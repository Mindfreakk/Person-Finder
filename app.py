# app.py
import os
import io
import uuid
import base64
import logging
import json
import secrets
from math import ceil
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
from PIL import Image, ExifTags
import qrcode

import face_recognition
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session, current_app
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user,
)
from flask_mail import Mail, Message

# Import DB helpers and models from database.py
from database import (
    db, initialize_database, Person, SearchLog, PushSubscription, User,
    register_person_to_db, get_all_registered_people,
    get_person_by_id, delete_person_by_id,
    find_person_by_face, log_best_match_search, get_stats,
    authenticate_user, debug_find_person_by_image, clear_people_encodings_cache
)

# SocketIO
from flask_socketio import SocketIO

# ----------------- Load environment -----------------
load_dotenv()
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "<YOUR_PRIVATE_KEY>")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "<YOUR_PUBLIC_KEY>")
VAPID_CLAIMS = {"sub": os.getenv("VAPID_SUBJECT", "mailto:you@example.com")}

# ----------------- Flask App Config -----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET") or os.urandom(24)

# Ensure database is always inside project root (next to app.py)
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "personfinder.db")
db_uri = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")

app.config.update(
    SQLALCHEMY_DATABASE_URI=db_uri,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER=os.path.join(app.root_path, "static", "uploads"),
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32 MB
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("EMAIL_APP_PASSWORD"),
    VAPID_PUBLIC_KEY=VAPID_PUBLIC_KEY,
    VAPID_PRIVATE_KEY=VAPID_PRIVATE_KEY,
    ADMIN_EMAIL=os.getenv("ADMIN_EMAIL"),
    ADMIN_PHONE=os.getenv("ADMIN_PHONE"),
    RECAPTCHA_SITE_KEY=os.getenv("RECAPTCHA_SITE_KEY"),
    RECAPTCHA_SECRET_KEY=os.getenv("RECAPTCHA_SECRET_KEY")
)

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------- Initialize Mail -----------------
mail = Mail(app)

# ----------------- Version Control -----------------
VERSION_FILE = "VERSION"

def get_version():
    """Read the version from the VERSION file, or initialize it."""
    if not os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "w") as f:
            f.write("1.0.0.0")  # initial version
    with open(VERSION_FILE, "r") as f:
        version = f.read().strip()
        parts = version.split(".")
        # Ensure 4-part version
        while len(parts) < 4:
            parts.append("0")
        return ".".join(parts)

def bump_version():
    """
    Automatically increment version with cascading:
    build -> patch -> minor -> major
    Each rolls over at 9.
    """
    version = get_version()
    major, minor, patch, build = map(int, version.split("."))

    # Increment build first
    build += 1
    if build > 9:
        build = 0
        patch += 1
        if patch > 9:
            patch = 0
            minor += 1
            if minor > 9:
                minor = 0
                major += 1

    new_version = f"{major}.{minor}.{patch}.{build}"
    with open(VERSION_FILE, "w") as f:
        f.write(new_version)
    return new_version

# ----------------- Auto Bump on Actual Server Start -----------------
# Avoid double bump during Flask dev auto-reload
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_RUN_FROM_CLI") == "true":
    APP_VERSION = bump_version()  # auto cascading bump
else:
    APP_VERSION = get_version()

# ----------------- Context Processor -----------------
@app.context_processor
def inject_globals():
    """Inject global template variables."""
    return {
        "app_version": APP_VERSION,            # auto-updated 4-part version
        "current_year": datetime.now().year,   # dynamic year
        "personId": None,
        "VAPID_PUBLIC_KEY": current_app.config.get("VAPID_PUBLIC_KEY", None),
        "RECAPTCHA_SITE_KEY": current_app.config.get("RECAPTCHA_SITE_KEY", None)
    }

    # ----------------- reCAPTCHA Verification -----------------
import requests

def verify_recaptcha(token, action="submit", min_score=0.5):
    """
    Verify Google reCAPTCHA v3 token with Google's API.
    Returns True if valid, False otherwise.
    """
    secret = current_app.config.get("RECAPTCHA_SECRET_KEY")
    if not secret:
        current_app.logger.warning("reCAPTCHA secret key not configured.")
        return False

    try:
        response = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": secret, "response": token}
        )
        result = response.json()

        # Debug logging (optional)
        current_app.logger.debug(f"reCAPTCHA result: {json.dumps(result, indent=2)}")

        # Check Google API response
        if not result.get("success"):
            return False
        if result.get("action") != action:
            return False
        if float(result.get("score", 0)) < min_score:
            return False

        return True
    except Exception as e:
        current_app.logger.error(f"reCAPTCHA verification failed: {e}")
        return False

# ----------------- Helper: Auto-correct EXIF orientation -----------------
def auto_orient_image(image_path):
    try:
        image = Image.open(image_path)
        orientation = None
        for tag in ExifTags.TAGS.keys():
            if ExifTags.TAGS[tag] == "Orientation":
                orientation = tag
                break
        exif = image._getexif()
        if exif is not None and orientation in exif:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
        return image
    except Exception:
        return Image.open(image_path)

# ----------------- Initialize DB -----------------
# Important: call db.init_app once, then initialize_database once inside app context.
db.init_app(app)
with app.app_context():
    initialize_database(app)
    print(f"🔎 Using database at: {db_path}")

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Flask-Login -----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "error"

class UserLogin(UserMixin):
    def __init__(self, user: User):
        self.id = str(user.id)
        self.username = user.username
        self.role = getattr(user, "role", "user")

@login_manager.user_loader
def load_user(user_id):
    user = User.query.get(int(user_id))
    return UserLogin(user) if user else None

def require_role(*roles):
    def wrapper(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))
            if roles and getattr(current_user, "role", None) not in roles:
                flash("Unauthorized access!", "error")
                return redirect(url_for("home"))
            return fn(*args, **kwargs)
        return inner
    return wrapper

# ----------------- Helpers -----------------
def save_uploaded_file(storage_file) -> str:
    original = secure_filename(storage_file.filename or "upload")
    ext = os.path.splitext(original)[1] or ".png"
    unique = f"{uuid.uuid4().hex}{ext}"
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    storage_file.save(abs_path)
    return abs_path

def save_base64_image(data_url: str) -> str:
    encoded = data_url.split(",", 1)[-1]
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    img = Image.open(img_bytes).convert("RGB")
    unique = f"{uuid.uuid4().hex}.png"
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    img.save(abs_path, format="PNG")
    return abs_path

def fix_photo_path(person: dict) -> dict:
    placeholder = "images/no-photo.png"
    raw = person.get("photo_path") or ""
    filename = os.path.basename(raw) if raw else ""
    person["photo_path"] = (
        f"uploads/{filename}"
        if filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else placeholder
    )
    return person

def save_subscription_to_db(person_id, subscription: dict):
    if not person_id or not subscription:
        return
    try:
        existing = PushSubscription.query.filter_by(person_id=person_id, endpoint=subscription["endpoint"]).first()
        if not existing:
            new_sub = PushSubscription(
                person_id=int(person_id),
                endpoint=subscription["endpoint"],
                p256dh=subscription["keys"]["p256dh"],
                auth=subscription["keys"]["auth"]
            )
            db.session.add(new_sub)
            db.session.commit()
    except Exception:
        logger.exception("Failed to save push subscription")

# ----------------- Utility Function -----------------
def get_stats():
    try:
        registrations = Person.query.count()
        searches = SearchLog.query.count()
        searches_traced = SearchLog.query.filter(SearchLog.success == 1).count()
    except Exception:
        registrations = searches = searches_traced = 0

    return {
        "registrations": registrations,
        "searches": searches,
        "searches_traced": searches_traced,
    }

# ----------------- Home Route -----------------
@app.route("/")
def home():
    stats = get_stats()
    return render_template(
        "home.html",
        stats=stats
    )

# ----------------- API Route -----------------
@app.route("/api/stats")
def api_stats():
    stats = get_stats()
    return jsonify(stats)


# ----------------- Auth Routes -----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))
        login_user(UserLogin(user))
        flash("Logged in successfully!", "success")
        return redirect(url_for("admin_dashboard") if user.role == "admin" else url_for("home"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(username="admin").first()
        if not user or email != os.getenv("ADMIN_EMAIL"):
            flash("No admin account with this email.", "error")
            return redirect(url_for("forgot_password"))

        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_expiry = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()

        reset_link = url_for("reset_password", token=token, _external=True)
        msg = Message(
            "Reset Admin Password",
            sender=app.config.get('MAIL_USERNAME'),
            recipients=[email],
        )
        msg.body = f"Click link to reset password (15 min validity): {reset_link}"
        try:
            mail.send(msg)
        except Exception:
            logger.exception("Failed to send reset email")

        flash("Password reset link sent.", "success")
        return redirect(url_for("login"))
    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or datetime.utcnow() > (user.reset_expiry or datetime.utcnow()):
        flash("Invalid or expired link.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "").strip()
        if len(password) < 6:
            flash("Password must be >=6 chars", "error")
            return redirect(request.url)

        user.password_hash = generate_password_hash(password)
        user.reset_token = None
        user.reset_expiry = None
        db.session.commit()

        flash("Password updated! Login now.", "success")
        return redirect(url_for("login"))
    return render_template("reset_password.html", token=token)


# --- Register Person ---
@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Registration route for missing persons with optional webcam upload,
    Google reCAPTCHA v3 verification, and live form validation.
    """
    RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY", "")
    filename_for_preview = None
    errors = {}
    form_data = {}
    focus_field = None

    if request.method == "POST":
        raw_form = request.form or {}
        photo_file = request.files.get("photo") or request.files.get("photo_file")
        photo_base64 = (raw_form.get("photo_input") or "").strip()

        # Copy form values to repopulate in case of error
        keys_to_copy = [
            "full_name", "age", "gender", "guardian_name", "phone_number",
            "address", "last_seen", "registered_by_name",
            "registered_by_phone", "registered_by_relation", "agreement"
        ]
        for k in keys_to_copy:
            form_data[k] = raw_form.get(k, "") or ""

        # --- reCAPTCHA v3 verification ---
        recaptcha_token = raw_form.get("g-recaptcha-response", "")
        if not verify_recaptcha(recaptcha_token, action="register", min_score=0.5):
            errors["recaptcha"] = "reCAPTCHA verification failed. Please try again."
            focus_field = focus_field or "recaptcha"

        # --- Required fields validation ---
        required_fields = {
            "full_name": "Full name is required.",
            "age": "Age is required.",
            "gender": "Please select a gender.",
            "guardian_name": "Guardian name is required.",
            "phone_number": "Phone number is required.",
            "address": "Address is required.",
            "registered_by_name": "Your name is required.",
            "registered_by_phone": "Your phone number is required.",
            "registered_by_relation": "Relation is required."
        }
        for fid, msg in required_fields.items():
            val = (form_data.get(fid) or "").strip()
            if val == "":
                errors[fid] = msg
                focus_field = focus_field or fid

        # Agreement checkbox
        if not str(form_data.get("agreement", "")).lower() in ("1", "true", "on", "yes"):
            errors["agreement"] = "You must agree to the Privacy Policy and Terms."
            focus_field = focus_field or "agreement"

        # Photo validation
        if not (photo_file and getattr(photo_file, "filename", "").strip()) and not photo_base64:
            errors["photo"] = "Photo is required (upload or use webcam)."
            focus_field = focus_field or "photo"

        # --- Save uploaded photo ---
        photo_abs_path = None
        try:
            if photo_file and getattr(photo_file, "filename", "").strip():
                photo_abs_path = save_uploaded_file(photo_file)
            elif photo_base64:
                photo_abs_path = save_base64_image(photo_base64)
        except Exception as e:
            logger.exception("Error saving photo: %s", e)
            errors["photo"] = "Unable to save uploaded photo."
            focus_field = focus_field or "photo"

        # Return if errors exist
        if errors:
            if photo_abs_path:
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
            return render_template(
                "register.html",
                filename=None,
                errors=errors,
                form_data=form_data,
                focus_field=focus_field,
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 400

        # --- Face detection ---
        try:
            image = face_recognition.load_image_file(photo_abs_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                errors["photo"] = "No face detected. Please upload a clear, front-facing image."
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
                return render_template(
                    "register.html",
                    filename=None,
                    errors=errors,
                    form_data=form_data,
                    focus_field="photo",
                    RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
                ), 400
        except Exception as e:
            logger.exception("Face encoding error: %s", e)
            errors["photo"] = "Unable to process the uploaded photo."
            try:
                os.remove(photo_abs_path)
            except Exception:
                pass
            return render_template(
                "register.html",
                filename=None,
                errors=errors,
                form_data=form_data,
                focus_field="photo",
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 400

        # --- Prepare person data ---
        try:
            age_val = int(form_data.get("age")) if form_data.get("age") else None
        except ValueError:
            age_val = None

        person_data = {
            "full_name": form_data.get("full_name").strip(),
            "age": age_val,
            "gender": form_data.get("gender").strip(),
            "guardian_name": form_data.get("guardian_name").strip(),
            "phone_number": form_data.get("phone_number").strip(),
            "address": form_data.get("address").strip(),
            "last_seen": form_data.get("last_seen").strip(),
            "photo_path": os.path.basename(photo_abs_path) if photo_abs_path else None,
            "face_encoding": encodings[0].tolist() if encodings else None,
            "created_by": current_user.id if current_user.is_authenticated else None,
            "registered_by_name": form_data.get("registered_by_name").strip(),
            "registered_by_phone": form_data.get("registered_by_phone").strip(),
            "registered_by_relation": form_data.get("registered_by_relation").strip(),
        }

        # --- Save to DB ---
        try:
            register_person_to_db(person_data)
            flash("Person registered successfully!", "success")
            return redirect(url_for("home"))
        except Exception as e:
            logger.exception("DB insert failed: %s", e)
            flash("Failed to save person to DB.", "error")
            if photo_abs_path:
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
            return render_template(
                "register.html",
                filename=None,
                errors={"server": "DB error"},
                form_data=form_data,
                focus_field=None,
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 500

    # --- GET request ---
    return render_template(
        "register.html",
        filename=filename_for_preview,
        form_data={},
        errors={},
        focus_field=None,
        RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
    )


# --- Search Person (with AJAX support) ---
@app.route("/search", methods=["GET", "POST"])
def search():
    MATCH_CONFIDENCE_THRESHOLD = 60.0
    MAX_RESULTS = 10
    results = []
    uploaded_photo = None
    searched = False
    face_locations = []

    def serialize_person_obj(p):
        if not p:
            return {}
        if isinstance(p, dict):
            return dict(p)
        out = {}
        for attr in (
            "id", "full_name", "age", "gender", "guardian_name",
            "phone_number", "address", "last_seen", "photo_path",
            "registered_by_name", "registered_by_phone", "registered_by_relation",
            "notes", "last_seen_location"
        ):
            val = getattr(p, attr, None)
            if val is not None:
                out[attr] = val
        created = getattr(p, "created_at", None) or getattr(p, "registration_date", None)
        if created and "registration_date" not in out:
            out["registration_date"] = (
                created.strftime("%Y-%m-%d %H:%M:%S") if isinstance(created, datetime) else created
            )
        return out

    if request.method == "POST":
        searched = True

        # --- Handle uploaded photo ---
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": "Please upload a photo."}), 400
            flash("Please upload a photo.", "error")
            return redirect(url_for("search"))

        tmp_abs_path = save_uploaded_file(photo)
        uploaded_photo = "uploads/" + os.path.basename(tmp_abs_path)

        try:
            # --- Open and correct orientation ---
            img = auto_orient_image(tmp_abs_path)

            # Resize if too large
            max_dim = 800
            if max(img.width, img.height) > max_dim:
                scale = max_dim / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save compressed JPEG
            temp_jpg_path = tmp_abs_path + "_compressed.jpg"
            img.convert("RGB").save(temp_jpg_path, format="JPEG", quality=85, optimize=True)
            tmp_abs_path = temp_jpg_path

            # --- Face recognition ---
            import numpy as np
            image_np = np.array(img)
            face_locations = face_recognition.face_locations(image_np)
            encodings = face_recognition.face_encodings(image_np, face_locations)

            all_matches = []

            if encodings:
                seen_ids = set()
                for face_index, face_encoding in enumerate(encodings):
                    candidates = find_person_by_face(face_encoding, tolerance=0.6, max_results=20, debug=False) or []
                    for cand in candidates:
                        cand = dict(cand)
                        cand["detected_face_index"] = face_index + 1
                        if face_index < len(face_locations):
                            loc = face_locations[face_index]
                            cand["matched_face_location"] = [int(v) for v in loc]

                        try:
                            cand["match_confidence"] = float(cand.get("match_confidence", 0))
                        except Exception:
                            cand["match_confidence"] = 0.0

                        pid = cand.get("id")
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)

                        if cand["match_confidence"] >= MATCH_CONFIDENCE_THRESHOLD:
                            all_matches.append(cand)

                results = sorted(all_matches, key=lambda x: x.get("match_confidence", 0), reverse=True)[:MAX_RESULTS]

                # Enrich with DB data
                enriched = []
                for cand in results:
                    pid = cand.get("id")
                    if pid:
                        try:
                            person_obj = get_person_by_id(int(pid))
                            pdata = serialize_person_obj(person_obj)
                            cand.update(pdata)
                        except Exception:
                            app.logger.exception("Failed to enrich candidate (id=%s)", pid)
                    try:
                        cand = fix_photo_path(cand)
                    except Exception:
                        cand.setdefault("photo_path", "images/no-photo.png")
                    enriched.append(cand)
                results = enriched

            # --- Log search attempt ---
            try:
                uploaded_name_for_log = os.path.basename(tmp_abs_path) if tmp_abs_path else uploaded_photo
                log_best_match_search(uploaded_name_for_log, results)
            except Exception:
                logger.exception("Failed to record/log search")

        except Exception as e:
            app.logger.exception("Error searching photo")
            try:
                log_best_match_search(os.path.basename(tmp_abs_path) if tmp_abs_path else None, [])
            except Exception:
                pass
            flash("Error processing uploaded photo.", "error")
            try:
                os.remove(tmp_abs_path)
            except Exception:
                pass
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": "Error processing uploaded photo."}), 500
            return redirect(url_for("search"))

        # --- Return JSON if AJAX ---
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "success": True,
                "results": results,
                "searched": searched,
                "uploaded_photo": uploaded_photo,
                "face_locations": face_locations
            })

    # --- Normal GET fallback ---
    return render_template(
        "search.html",
        results=results,
        uploaded_photo=uploaded_photo,
        searched=searched,
        face_locations=face_locations
    )


# --- Admin Dashboard ---
@app.route("/admin/dashboard")
@login_required
@require_role("admin")
def admin_dashboard():
    page = int(request.args.get("page", 1))
    per_page = 10
    people_query = Person.query.order_by(Person.id.desc())
    total_people = people_query.count()
    people = people_query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = ceil(total_people / per_page)

    # call get_all_registered_people() so it's used and available for any utility needs
    try:
        all_people_for_util = get_all_registered_people()
    except Exception:
        all_people_for_util = []

    stats = get_stats()
    users = User.query.all()
    search_logs = SearchLog.query.order_by(SearchLog.ts.desc()).limit(10).all()

    return render_template(
        "admin_dashboard.html",
        stats=stats,
        users=users,
        people=people,
        search_logs=search_logs,
        page=page,
        total_pages=total_pages,
    )


# --- Clear_Search_Logs ---
@app.route('/clear_search_logs', methods=['POST'])
@login_required
def clear_search_logs():
    if getattr(current_user, "role", None) != 'admin':
        flash("You do not have permission to perform this action.", "error")
        return redirect(url_for('admin_dashboard'))

    page = request.form.get("page", 1, type=int)  # preserve current page

    try:
        SearchLog.query.delete()
        db.session.commit()
        flash("All search logs have been cleared.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error clearing logs: {e}", "error")

    return redirect(url_for('admin_dashboard', page=page))


# --- Delete_Person ---
@app.route("/delete_person/<int:person_id>", methods=["POST"])
@login_required
def delete_person(person_id):
    page = request.form.get("page", 1, type=int)
    try:
        # Use helper from database.py which enforces authorization rules
        success = delete_person_by_id(person_id, current_user)
        if not success:
            flash("Person not found or unauthorized to delete.", "error")
        else:
            try:
                clear_people_encodings_cache()
            except Exception:
                pass
            flash("Person deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting person: {e}", "error")

    return redirect(url_for("admin_dashboard", page=page))


# --- Delete_Multiple_Persons ---
@app.route("/delete_multiple_persons", methods=["POST"])
@login_required
def delete_multiple_persons():
    page = request.form.get("page", 1, type=int)  # keep same page
    ids = request.form.getlist("selected_ids")

    if not ids:
        flash("No persons selected for deletion.", "warning")
        return redirect(url_for("admin_dashboard", page=page))

    try:
        # Perform a bulk delete for performance, then clear enc cache
        Person.query.filter(Person.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
        try:
            clear_people_encodings_cache()
        except Exception:
            pass
        flash(f"Deleted {len(ids)} people successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting people: {e}", "error")

    return redirect(url_for("admin_dashboard", page=page))


# --- Static Pages ---
@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy-policy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/donate")
def donate_page():
    return render_template("donate.html")


@app.route("/about")
def about():
    return render_template("about.html", title="About App - PersonFinder")


@app.route("/developers")
def developers():
    return render_template("developers.html", current_year=datetime.now().year)


@app.route("/donate-qr")
def donate_qr():
    payment_link = "upi://pay?pa=yourupiid@upi&pn=PersonFinder&am=500"
    img = qrcode.make(payment_link)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route('/save-subscription', methods=['POST'])
def save_subscription():
    subscription_info = request.get_json()
    person_id = request.args.get('person_id')
    save_subscription_to_db(person_id, subscription_info)
    return jsonify({"success": True})


# --- Admin debug endpoint: upload an image and get per-face match diagnostics ---
@app.route("/admin/debug-match", methods=["POST"])
@login_required
@require_role("admin")
def admin_debug_match():
    f = request.files.get("photo")
    if not f or not f.filename:
        return jsonify({"error": "upload missing"}), 400
    tmp = save_uploaded_file(f)
    try:
        data = debug_find_person_by_image(tmp, tolerance=0.6, max_results=20)
        return jsonify(data)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ----------------- SocketIO Integration -----------------
from flask_socketio import SocketIO

# Initialize SocketIO (allow cross-origin for mobile devices)
socketio = SocketIO(app, cors_allowed_origins="*")  # <- important for mobile access

# ----------------- Run Flask + SocketIO -----------------
if __name__ == "__main__":
    import socket

    def get_local_ip():
        """Detect local network IP for mobile access."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # connect to external host
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    host_ip = get_local_ip()
    port = 5001

    print(f"🚀 Server starting!")
    print(f"  Desktop: http://127.0.0.1:{port}")
    print(f"  Mobile (same Wi-Fi): http://{host_ip}:{port}")

    # Run SocketIO on all interfaces
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
