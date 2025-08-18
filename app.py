from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import face_recognition
import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import uuid
import re
import logging

# ----------------- Import DB functions -----------------
from database import (
    register_person_to_db,
    get_all_registered_people,
    create_db,
    delete_person_by_id,
    get_person_by_id
)

# Import Flask-WTF forms
from forms import RegisterForm, SearchForm

# ----------------- Config -----------------
UPLOAD_FOLDER = 'static/uploads'
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Amir@123")  # ⚠️ fallback for local testing

# Flask setup
app = Flask(__name__)
app.secret_key = os.urandom(24)  # ⚠️ replace with a secure random value in production
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Enable logging
logging.basicConfig(level=logging.INFO)


# ----------------- Helpers -----------------
def fix_photo_path(person):
    """Ensure the photo path points to an existing file or fallback placeholder."""
    placeholder = "images/no-photo.png"
    raw = person.get("photo_path") or ""
    filename = os.path.basename(raw) if raw else ""
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            person["photo_path"] = f"uploads/{filename}"
        else:
            person["photo_path"] = placeholder
    else:
        person["photo_path"] = placeholder
    return person


def extract_face_encoding(photo_path):
    """Extract face encoding from an image file."""
    image = face_recognition.load_image_file(photo_path)
    face_encodings = face_recognition.face_encodings(image)
    if not face_encodings:
        raise ValueError("No face found in the image.")
    return face_encodings[0]


def compare_faces(search_encoding, registered_encoding):
    """Compare two face encodings with tolerance."""
    return face_recognition.compare_faces(
        [registered_encoding], search_encoding, tolerance=0.6
    )[0]


def _ensure_upload_folder():
    """Make sure upload folder exists."""
    folder = app.config.get('UPLOAD_FOLDER', '')
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


# ----------------- Routes -----------------
@app.route('/')
def home():
    all_people = get_all_registered_people()
    recent_matches = all_people[-5:] if len(all_people) > 5 else all_people
    recent = [fix_photo_path(p.copy()) for p in recent_matches]
    return render_template('index.html', recent_matches=recent)


# ----------------- Register -----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        full_name = form.full_name.data.strip()
        guardian_name = form.guardian_name.data.strip()
        age = form.age.data
        phone = form.phone_number.data.strip()
        address = form.address.data.strip()
        last_seen_date = form.last_seen_date.data
        last_seen_str = last_seen_date.isoformat() if last_seen_date else ""

        upload_photo = request.files.get('upload_photo')
        webcam_image_b64 = (request.form.get('webcam_image') or "").strip()
        has_upload = bool(upload_photo and upload_photo.filename.strip())
        has_webcam = bool(webcam_image_b64)

        if not has_upload and not has_webcam:
            flash('Please provide a photo using upload or webcam.', 'error')
            return redirect(url_for('register'))
        if has_upload and has_webcam:
            flash('Choose only one photo method — either upload OR webcam.', 'error')
            return redirect(url_for('register'))

        _ensure_upload_folder()
        try:
            if has_upload:
                filename = secure_filename(upload_photo.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                upload_photo.save(photo_path)
            else:
                m = re.match(r'^data:image/(png|jpeg|jpg);base64,(.+)$', webcam_image_b64, flags=re.I)
                if not m:
                    flash('Invalid webcam image format.', 'error')
                    return redirect(url_for('register'))
                ext = 'jpeg' if m.group(1).lower() == 'jpg' else m.group(1).lower()
                img_data = base64.b64decode(m.group(2))
                unique_filename = f"{uuid.uuid4().hex}_webcam.{ext}"
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                with open(photo_path, 'wb') as f:
                    f.write(img_data)
        except Exception as e:
            logging.error(f"Error saving photo: {e}")
            flash(f'Error saving photo: {e}', 'error')
            return redirect(url_for('register'))

        try:
            face_encoding = extract_face_encoding(photo_path)
        except ValueError as e:
            flash(str(e), 'error')
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return redirect(url_for('register'))

        register_person_to_db(
            name=full_name,
            age=age,
            phone=phone,
            address=address,
            guardian_name=guardian_name,
            last_seen=last_seen_str,
            face_encoding=face_encoding.tolist(),
            photo_path=unique_filename
        )

        flash('Person registered successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html', form=form)


# ----------------- Search -----------------
@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    matches = []
    photo_rel_path = None

    if form.validate_on_submit():
        photo = form.photo.data
        last_seen = form.last_seen.data or ""

        if not photo or not photo.filename.strip():
            flash('Please upload a photo before searching.', 'error')
            return redirect(url_for('search'))

        _ensure_upload_folder()
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(photo.filename)}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        photo_rel_path = f"uploads/{unique_filename}"
        photo.save(photo_path)

        try:
            search_encoding = extract_face_encoding(photo_path)
        except ValueError as e:
            os.remove(photo_path)
            flash(str(e), 'error')
            return redirect(url_for('search'))

        registered_people = get_all_registered_people()
        for person in registered_people:
            if compare_faces(search_encoding, np.array(person['face_encoding'])):
                person_copy = fix_photo_path(person.copy())
                if last_seen:
                    person_copy['last_seen'] = last_seen
                matches.append(person_copy)

        flash(f'Found {len(matches)} matching person(s).' if matches else 'No matching person found.',
              'success' if matches else 'error')

        os.remove(photo_path)

    return render_template('search.html', form=form, matches=matches, photo_path=photo_rel_path)


# ----------------- Delete -----------------
@app.route('/delete/<int:person_id>', methods=['POST'])
def delete(person_id):
    password = request.form.get('password', '')
    if password != ADMIN_PASSWORD:
        flash('Invalid admin password.', 'error')
        return redirect(url_for('search'))
    try:
        delete_person_by_id(person_id)
        flash('Person deleted successfully.', 'success')
    except Exception as e:
        logging.error(f"Error deleting person: {e}")
        flash(f'Error deleting person: {str(e)}', 'error')
    return redirect(url_for('search'))


# ----------------- API Search Frame -----------------
@app.route('/api/search_frame', methods=['POST'])
def search_frame():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"]
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        face_encodings = face_recognition.face_encodings(np.array(image))
        if not face_encodings:
            return jsonify({"match": False})

        search_encoding = face_encodings[0]
        registered_people = get_all_registered_people()
        for person in registered_people:
            if compare_faces(search_encoding, np.array(person['face_encoding'])):
                person_fixed = fix_photo_path(person.copy())
                return jsonify({
                    "match": True,
                    "person": {
                        "name": person_fixed["name"],
                        "phone": person_fixed["phone"],
                        "address": person_fixed["address"],
                        "guardian_name": person_fixed.get("guardian_name"),
                        "photo_path": person_fixed["photo_path"]
                    }
                })
        return jsonify({"match": False})
    except Exception as e:
        logging.error(f"Error in /api/search_frame: {e}")
        return jsonify({"error": "Server error", "details": str(e)}), 500


# ----------------- Upload Photo -----------------
@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    try:
        if 'upload_photo' in request.files:
            upload_photo = request.files['upload_photo']
            if not upload_photo.filename.strip():
                return jsonify({"error": "Empty file"}), 400
            filename = secure_filename(upload_photo.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            upload_photo.save(save_path)
            return jsonify({"success": True, "filename": unique_filename, "path": f"uploads/{unique_filename}"})
        return jsonify({"error": "No valid image provided"}), 400
    except Exception as e:
        logging.error(f"Error in /upload_photo: {e}")
        return jsonify({"error": "Server error", "details": str(e)}), 500


# ----------------- Static Pages -----------------
@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy-policy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


# ----------------- Main -----------------
if __name__ == "__main__":
    with app.app_context():
        create_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
