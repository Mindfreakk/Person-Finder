from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import face_recognition
import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import uuid
from database import (
    register_person_to_db,
    get_all_registered_people,
    create_db,
    delete_person_by_id,
    get_person_by_id
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Simple admin password for delete action (replace with env var or config)
ADMIN_PASSWORD = "Amir@123"


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        age = request.form.get('age')
        phone = request.form.get('phone')
        address = request.form.get('address')
        upload_photo = request.files.get('upload_photo')

        try:
            age = int(age)
            if age < 0:
                flash('Age cannot be negative', 'error')
                return redirect(url_for('register'))
        except (ValueError, TypeError):
            flash('Please enter a valid age', 'error')
            return redirect(url_for('register'))

        if not upload_photo:
            flash('Please upload a photo', 'error')
            return redirect(url_for('register'))

        filename = secure_filename(upload_photo.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        upload_photo.save(photo_path)

        try:
            face_encoding = extract_face_encoding(photo_path)
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('register'))

        face_encoding_list = face_encoding.tolist()

        register_person_to_db(
            full_name,
            age,
            phone,
            address,
            face_encoding_list,
            photo_path
        )

        flash('Person registered successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        photo = request.files.get('photo')
        if not photo or photo.filename == '':
            flash('Please upload a photo before searching.', 'error')
            return redirect(url_for('search'))

        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(photo.filename)}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        photo_rel_path = os.path.join('uploads', unique_filename).replace("\\", "/")  # For preview
        photo.save(photo_path)

        try:
            search_encoding = extract_face_encoding(photo_path)
        except ValueError as e:
            os.remove(photo_path)
            flash(str(e), 'error')
            return redirect(url_for('search'))

        registered_people = get_all_registered_people()
        matches = []
        for person in registered_people:
            registered_encoding = np.array(person['face_encoding'])
            if compare_faces(search_encoding, registered_encoding):
                relative_photo_path = os.path.relpath(person['photo_path'], 'static')
                person['photo_path'] = relative_photo_path.replace("\\", "/")
                matches.append(person)

        if matches:
            flash(f'Found {len(matches)} matching person(s).', 'success')
        else:
            flash('No matching person found.', 'error')

        try:
            os.remove(photo_path)
        except Exception as e:
            print(f"Warning: could not delete search photo: {e}")

        return render_template('search.html', matches=matches, photo_path=photo_rel_path)

    # GET request
    return render_template('search.html', matches=None, photo_path=None)


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
        flash(f'Error deleting person: {str(e)}', 'error')

    return redirect(url_for('search'))


def extract_face_encoding(photo_path):
    image = face_recognition.load_image_file(photo_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    else:
        raise ValueError("No face found in the image.")


def compare_faces(search_encoding, registered_encoding):
    return face_recognition.compare_faces([registered_encoding], search_encoding, tolerance=0.6)[0]


@app.route('/api/search_frame', methods=['POST'])
def search_frame():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"]
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": "Invalid image data", "details": str(e)}), 400

        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return jsonify({"match": False})

        search_encoding = face_encodings[0]
        registered_people = get_all_registered_people()

        for person in registered_people:
            registered_encoding = np.array(person['face_encoding'])
            if compare_faces(search_encoding, registered_encoding):
                return jsonify({
                    "match": True,
                    "person": {
                        "name": person["name"],
                        "phone": person["phone"],
                        "address": person["address"],
                        "photo_filename": os.path.basename(person["photo_path"])
                    }
                })

        return jsonify({"match": False})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route('/')
def home():
    # Fetch last 5 recent matches (ordered by newest first)
    all_people = get_all_registered_people()
    recent_matches = all_people[-5:] if len(all_people) > 5 else all_people
    return render_template('index.html', recent_matches=recent_matches)


@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy-policy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


if __name__ == "__main__":
    with app.app_context():
        create_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
