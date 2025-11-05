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
from PIL import Image, ExifTags, ImageOps
import qrcode
import requests
import numpy as np  # ✅ FIX: used in face encoding path

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
from werkzeug.exceptions import RequestEntityTooLarge

# SocketIO
from flask_socketio import SocketIO

# Import DB helpers and models from database.py
from database import (
    db, initialize_database, Person, SearchLog, PushSubscription, User,
    register_person_to_db, get_all_registered_people,
    get_person_by_id, delete_person_by_id,
    find_person_by_face, log_best_match_search, get_stats as db_get_stats,
    authenticate_user, debug_find_person_by_image, clear_people_encodings_cache
)

# ----------------- Load environment -----------------
load_dotenv()
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "<YOUR_PRIVATE_KEY>")
VAPID_PUBLIC_KEY  = os.getenv("VAPID_PUBLIC_KEY", "<YOUR_PUBLIC_KEY>")
VAPID_CLAIMS      = {"sub": os.getenv("VAPID_SUBJECT", "mailto:you@example.com")}
DEV_MODE          = str(os.getenv("DEV_MODE", "true")).lower() in ("1", "true", "yes", "on")

# ----------------- Flask App Config -----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET") or os.urandom(24)

# Ensure database is always inside project root (next to app.py)
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "personfinder.db")
db_uri  = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")

# ---- Mail defaults / coercion helpers ----
def _bool_env(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")

MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT   = int(os.getenv("MAIL_PORT", "587"))
MAIL_USE_TLS = _bool_env("MAIL_USE_TLS", "true")
MAIL_USE_SSL = _bool_env("MAIL_USE_SSL", "false")
# Ensure only one of TLS/SSL is true; prefer TLS for Gmail: 587/TLS
if MAIL_USE_TLS and MAIL_USE_SSL:
    MAIL_USE_SSL = False

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # Gmail App Password (16 chars)
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER") or MAIL_USERNAME

app.config.update(
    SQLALCHEMY_DATABASE_URI=db_uri,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,

    # Uploads
    UPLOAD_FOLDER=os.path.join(app.root_path, "static", "uploads"),

    # Accept large uploads — upper bound here; your routes handle specifics
    MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH_BYTES", 32 * 1024 * 1024)),
    MAX_FORM_MEMORY_SIZE=int(os.getenv("MAX_FORM_MEMORY_SIZE_BYTES", 32 * 1024 * 1024)),

    # Flask-Mail
    MAIL_SERVER=MAIL_SERVER,
    MAIL_PORT=MAIL_PORT,
    MAIL_USE_TLS=MAIL_USE_TLS,
    MAIL_USE_SSL=MAIL_USE_SSL,
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_DEFAULT_SENDER=MAIL_DEFAULT_SENDER,
    MAIL_SUPPRESS_SEND=False,
    MAIL_DEBUG=DEV_MODE,

    # Push
    VAPID_PUBLIC_KEY=VAPID_PUBLIC_KEY,
    VAPID_PRIVATE_KEY=VAPID_PRIVATE_KEY,

    # Admin info
    ADMIN_EMAIL=os.getenv("ADMIN_EMAIL"),
    ADMIN_PHONE=os.getenv("ADMIN_PHONE"),

    # reCAPTCHA
    RECAPTCHA_SITE_KEY=os.getenv("RECAPTCHA_SITE_KEY"),
    RECAPTCHA_SECRET_KEY=os.getenv("RECAPTCHA_SECRET_KEY"),
)

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------- Initialize Mail -----------------
mail = Mail(app)

# ----------------- Version Control (dynamic) -----------------
import os
basedir = os.path.abspath(os.path.dirname(__file__))

# Prefer version.txt (set by CI), fall back to VERSION if present
_VERSION_CANDIDATES = [
    os.path.join(basedir, "version.txt"),
    os.path.join(basedir, "VERSION"),
]
for _p in _VERSION_CANDIDATES:
    if os.path.exists(_p):
        VERSION_FILE = _p
        break
else:
    VERSION_FILE = _VERSION_CANDIDATES[0]  # default to version.txt

# Tiny cache so we don't re-read on every request unless file changed
_version_cache = {"mtime": None, "val": "1.0.0.0"}

def get_app_version() -> str:
    """Return current version string from VERSION_FILE; re-read on mtime change."""
    try:
        st_mtime = os.stat(VERSION_FILE).st_mtime
        if st_mtime != _version_cache["mtime"]:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                val = (f.read() or "").strip() or "1.0.0.0"
            _version_cache.update({"mtime": st_mtime, "val": val})
    except Exception:
        # If file missing/unreadable, keep last known value
        pass
    return _version_cache["val"]

# (Optional) keep your bump_version if you still call it elsewhere
def bump_version():
    """
    Keeps your original 4-segment cascade bump logic.
    Only use this if you explicitly want to bump locally on demand.
    """
    # Ensure file exists
    if not os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write("1.0.0.0")

    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        version = (f.read() or "").strip() or "1.0.0.0"

    parts = version.split(".")
    while len(parts) < 4:
        parts.append("0")
    major, minor, patch, build = map(int, parts[:4])

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
    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        f.write(new_version)
    # Invalidate cache so next request sees the new version
    _version_cache["mtime"] = None
    return new_version

# ----------------- Context Processor -----------------
@app.context_processor
def inject_globals():
    """Inject global template variables."""
    return {
        "app_version": get_app_version(),  # <— dynamic now
        "current_year": datetime.now().year,
        "personId": None,
        "VAPID_PUBLIC_KEY": current_app.config.get("VAPID_PUBLIC_KEY", None),
        "RECAPTCHA_SITE_KEY": current_app.config.get("RECAPTCHA_SITE_KEY", None),
    }

# ----------------- reCAPTCHA Verification -----------------
def verify_recaptcha(token, action="submit", min_score=0.5):
    """
    Verify Google reCAPTCHA v3 token with Google's API.
    Returns True if valid, False otherwise. In DEV_MODE we fail-open so you can test locally.
    """
    secret = current_app.config.get("RECAPTCHA_SECRET_KEY")
    if not secret:
        current_app.logger.warning("reCAPTCHA secret key not configured.")
        return DEV_MODE
    try:
        response = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": secret, "response": token}
        )
        result = response.json()
        current_app.logger.debug(f"reCAPTCHA result: {json.dumps(result, indent=2)}")

        if not result.get("success"):
            return False
        if result.get("action") != action:
            return False
        if float(result.get("score", 0)) < min_score:
            return False
        return True
    except Exception as e:
        current_app.logger.error(f"reCAPTCHA verification failed: {e}")
        return DEV_MODE

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
# ----------------- Initialize DB -----------------
db.init_app(app)
with app.app_context():
    initialize_database(app)
    print(f"🔎 Using database at: {db_path}")

    # ---- Ensure `phone_number` column exists (if you ever need to be defensive) ----
    from sqlalchemy import inspect, text
    insp = inspect(db.engine)
    try:
        has_phone_number = any(col["name"] == "phone_number" for col in insp.get_columns("users"))
    except Exception:
        has_phone_number = True  # if inspect fails, assume it's there since your model defines it

    if not has_phone_number:
        try:
            db.session.execute(text('ALTER TABLE "users" ADD COLUMN phone_number VARCHAR(50)'))
            db.session.commit()
            print("✅ Added 'phone_number' column to 'users'.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Failed adding 'phone_number' column: {e}")

    # --- One-time backfill of admin phone from .env ---
    env_phone = (os.getenv("ADMIN_PHONE") or "").strip()
    if env_phone:
        admin = User.query.filter_by(username="admin").first()
        if admin:
            if not admin.phone_number or admin.phone_number.strip() != env_phone:
                admin.phone_number = env_phone
                db.session.commit()
                print(f"✅ Admin phone_number set/updated to: {env_phone}")
            else:
                print("ℹ️ Admin phone_number already up to date.")
        else:
            print("ℹ️ Admin user not found; skipping phone backfill.")
    else:
        print("⚠️ ADMIN_PHONE not set in environment.")


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
        existing = PushSubscription.query.filter_by(
            person_id=person_id, endpoint=subscription["endpoint"]
        ).first()
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
    return render_template("home.html", stats=stats)

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

# ----------------- Auto-detect LAN IP & build external URLs -----------------
import socket
from urllib.parse import urljoin

def get_local_ip() -> str:
    """Return LAN IP (e.g. 192.168.x.x). Falls back to 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # only to learn the outbound iface
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def build_external_url(endpoint: str, **values) -> str:
    """
    Reset links that open on phone:
    - Use PUBLIC_BASE_URL if set (e.g. https://your-domain.com)
    - Else use auto-detected LAN IP + PORT
    """
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if not base:
        local_ip = get_local_ip()
        port = os.getenv("PORT", "5001")
        base = f"http://{local_ip}:{port}"
    rel = url_for(endpoint, _external=False, **values).lstrip("/")
    return urljoin(base.rstrip("/") + "/", rel)

# ----------------- Mail helper -----------------
from typing import Optional, Tuple, List

def send_mail(subject: str, recipients: List[str], *, body: str = "", html: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Unified mail sender with clear error surfacing.
    Returns (ok, error_message).
    """
    sender = app.config.get("MAIL_DEFAULT_SENDER") or app.config.get("MAIL_USERNAME")
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        return False, "MAIL_USERNAME/EMAIL_APP_PASSWORD not set in environment."
    try:
        msg = Message(subject=subject, recipients=recipients, sender=sender)
        msg.body = body or ""
        if html:
            msg.html = html
        mail.send(msg)
        return True, None
    except Exception as e:
        logger.exception("Flask-Mail send failed")
        return False, str(e)

# ---------- Forgot / Reset (uses your combined template) ----------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """
    Renders forgot_password.html with token_valid=False (request-reset form).
    On POST, sends a real email; on failure shows the concrete error.
    """
    token_valid = False
    if request.method == "POST":
        email = (request.form.get("email", "")).strip().lower()
        admin_env = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

        # Admin user record (by your convention: username='admin')
        user = User.query.filter_by(username="admin").first()

        if not user or email != admin_env:
            # Don’t leak user existence. In DEV, also show explicit hint.
            if DEV_MODE:
                flash("No admin account with this email (or ADMIN_EMAIL mismatch).", "error")
            flash("If that email is registered, a reset link has been sent.", "success")
            return render_template("forgot_password.html", token_valid=token_valid)

        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_expiry = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()

        # Use LAN-friendly/public base URL so phone can open it
        reset_link = build_external_url("reset_password", token=token)

        ok, err = send_mail(
            "Reset Admin Password",
            [email],
            body=f"Click to reset password (valid 15 minutes): {reset_link}",
            html=f"""<p>You requested an admin password reset.</p>
                     <p><a href="{reset_link}">Reset Password</a></p>
                     <p>This link expires in 15 minutes.</p>"""
        )
        if not ok:
            # In DEV, print the URL to console to avoid being blocked while testing.
            if DEV_MODE:
                print("\n[DEV] Email send failed. Reset URL (copy/paste):\n", reset_link, "\n")
            flash(f"Could not send reset email: {err}", "error")
            return render_template("forgot_password.html", token_valid=token_valid)

        flash("Reset link sent successfully.", "success")
        return render_template("forgot_password.html", token_valid=token_valid)

    return render_template("forgot_password.html", token_valid=token_valid)

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """
    Uses the same template (forgot_password.html) with token_valid=True to render the set-password form.
    """
    user = User.query.filter_by(reset_token=token).first()
    if not user or datetime.utcnow() > (user.reset_expiry or datetime.utcnow()):
        flash("Invalid or expired link.", "error")
        return render_template("forgot_password.html", token_valid=False)

    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        confirm  = (request.form.get("confirm_password") or "").strip()
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("forgot_password.html", token_valid=True)
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("forgot_password.html", token_valid=True)

        user.password_hash = generate_password_hash(password)
        user.reset_token = None
        user.reset_expiry = None
        db.session.commit()

        flash("Password updated! Login now.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html", token_valid=True)

# ---- Optional: DEV-only test route to verify SMTP quickly ----
if DEV_MODE:
    @app.route("/debug/send-test-email")
    def debug_send_test_email():
        recipient = (os.getenv("ADMIN_EMAIL") or MAIL_USERNAME)
        ok, err = send_mail(
            "PersonFinder Test Email",
            [recipient],
            body="This is a test email from PersonFinder."
        )
        if ok:
            return f"✅ Test email sent to {recipient}"
        return f"❌ Failed to send test email: {err}", 500

# --- Register Person (no size/resolution limits; stream-safe; largest-face; duplicate short-circuit) ---
# Disable Pillow decompression bomb checks (accept all megapixels)
Image.MAX_IMAGE_PIXELS = None

# Ensure upload dir exists
app.config.setdefault("UPLOAD_FOLDER", os.getenv("UPLOAD_FOLDER", "static/uploads"))
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    """If some upstream still throws 413, render the form with a helpful message."""
    RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY", "")
    errors = {"photo": "The image was rejected by an upstream limit. Please try again now; "
                       "the server accepts very large images. If it persists, ask the admin to raise proxy limits."}
    return render_template(
        "register.html",
        filename=None,
        errors=errors,
        form_data=request.form if request.form else {},
        focus_field="photo",
        RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
    ), 413

@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Registration route (no size/resolution/pixel limits).
    - Accepts any size image (server-side limits set very high; Pillow guard disabled).
    - Streams from upload (no full file read into RAM).
    - Keeps original pixel dimensions in the saved file (no downscale).
    - Largest-face selection; duplicate short-circuit.
    - Agreement validation and reCAPTCHA.
    """
    RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY", "")

    # Face settings
    FACE_MODEL = "hog"              # "cnn" requires GPU/dlib-cnn; "hog" is CPU-friendly
    DUPLICATE_TOLERANCE = 0.45
    DUPLICATE_MIN_CONF = 90.0

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    UPLOAD_FOLDER = app.config.get("UPLOAD_FOLDER", "static/uploads")

    # ---------- helpers ----------
    def _now_ts():
        import time
        return int(time.time())

    def allowed_file(filename: str) -> bool:
        if not filename or "." not in filename:
            return False
        ext = filename.rsplit(".", 1)[-1].lower()
        return ext in ALLOWED_EXTENSIONS

    def unique_filename(basename: str, suffix=".jpg") -> str:
        base = secure_filename((basename or "upload").rsplit(".", 1)[0]) or "upload"
        token = secrets.token_hex(8)
        return f"{base}-{_now_ts()}-{token}{suffix}"

    def normalize_image_keep_pixels(img: Image.Image) -> Image.Image:
        # Auto-orient, drop alpha to white, and convert to RGB
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def save_as_jpeg_without_resizing(pil_img: Image.Image, out_abs_path: str, quality: int = 90):
        """Save as JPEG (no resizing), reasonable quality (no size cap enforced)."""
        os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)  # ✅ FIX: use the parameter
        with open(out_abs_path, "wb") as f:
            f.write(buf.getvalue())
        return out_abs_path

    def process_file_storage_streaming(file_storage) -> str:
        """
        Open the uploaded file from its stream (no full read), normalize, and save
        without any downscale/byte cap.
        """
        filename = file_storage.filename or "upload"
        if not allowed_file(filename):
            raise ValueError("Unsupported image type. Allowed: JPG, JPEG, PNG, WEBP.")

        try:
            file_storage.stream.seek(0)
        except Exception:
            pass

        try:
            img = Image.open(file_storage.stream)  # PIL lazy decode
        except Exception:
            raise ValueError("Invalid image file.")

        img = normalize_image_keep_pixels(img)
        out_name = unique_filename(filename, suffix=".jpg")
        out_abs_path = os.path.join(UPLOAD_FOLDER, out_name)
        return save_as_jpeg_without_resizing(img, out_abs_path, quality=90)

    def process_base64_image_no_resize(data_url_or_b64: str) -> str:
        s = (data_url_or_b64 or "").strip()
        if "," in s and s.lower().startswith("data:image"):
            s = s.split(",", 1)[1]
        try:
            raw = base64.b64decode(s, validate=True)
        except Exception:
            raise ValueError("Invalid base64 image data.")
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception:
            raise ValueError("Invalid base64 image payload.")
        img = normalize_image_keep_pixels(img)
        out_name = unique_filename("webcam", suffix=".jpg")
        out_abs_path = os.path.join(UPLOAD_FOLDER, out_name)
        return save_as_jpeg_without_resizing(img, out_abs_path, quality=90)

    def load_for_face(image_path: str):
        """
        For face encoding speed only, we may downscale a COPY (does not affect saved file).
        """
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        max_dim = max(pil.width, pil.height)
        if max_dim > 1600:  # purely for speed during encoding
            scale = 1600 / max_dim
            pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.Resampling.LANCZOS)
        return np.array(pil)

    def pick_largest_face(face_locations):
        if not face_locations:
            return None, None
        areas = [max(0, (b - t)) * max(0, (r - l)) for (t, r, b, l) in face_locations]
        idx = int(np.argmax(areas))
        return idx, face_locations[idx]

    filename_for_preview = None
    errors = {}
    form_data = {}
    focus_field = None

    if request.method == "POST":
        raw_form = request.form or {}

        # Prefer client base64 if present; else use the file
        photo_base64 = (raw_form.get("photo_input") or "").strip()
        photo_file = None if photo_base64 else (request.files.get("photo") or request.files.get("photo_file"))

        # Keep values to repopulate on error
        for k in [
            "full_name", "age", "gender", "guardian_name", "phone_number",
            "address", "last_seen", "registered_by_name",
            "registered_by_phone", "registered_by_relation", "agreement"
        ]:
            form_data[k] = raw_form.get(k, "") or ""

        # reCAPTCHA
        recaptcha_token = raw_form.get("g-recaptcha-response", "")
        if not verify_recaptcha(recaptcha_token, action="register", min_score=0.5):
            errors["recaptcha"] = "reCAPTCHA verification failed. Please try again."
            focus_field = focus_field or "agreement"

        # Required fields
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
            if (form_data.get(fid) or "").strip() == "":
                errors[fid] = msg
                focus_field = focus_field or fid

        # Agreement (server-side)
        if str(form_data.get("agreement", "")).lower() not in ("1", "true", "on", "yes"):
            errors["agreement"] = "You must agree to the Privacy Policy and Terms."
            focus_field = focus_field or "agreement"

        # Photo presence
        if not (photo_file and getattr(photo_file, "filename", "").strip()) and not photo_base64:
            errors["photo"] = "Photo is required (upload or use webcam)."
            focus_field = focus_field or "photo"

        # Early return on validation errors
        if errors:
            return render_template(
                "register.html",
                filename=None,
                errors=errors,
                form_data=form_data,
                focus_field=focus_field,
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 400

        # Save photo (NO resize/byte cap), streaming
        photo_abs_path = None
        try:
            if photo_base64:
                photo_abs_path = process_base64_image_no_resize(photo_base64)
            else:
                try:
                    photo_file.stream.seek(0)
                except Exception:
                    pass
                photo_abs_path = process_file_storage_streaming(photo_file)
        except ValueError as ve:
            errors["photo"] = str(ve)
        except Exception:
            app.logger.exception("Error saving/normalizing photo")
            errors["photo"] = "Unable to save uploaded photo."

        if errors:
            if photo_abs_path:
                try: os.remove(photo_abs_path)
                except Exception: pass
            return render_template(
                "register.html",
                filename=None,
                errors=errors,
                form_data=form_data,
                focus_field=focus_field or "photo",
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 400

        # Face detection & largest-face encoding (on COPY for speed only)
        try:
            image_np = load_for_face(photo_abs_path)
            face_locations = face_recognition.face_locations(image_np, model=FACE_MODEL)
            if not face_locations:
                errors["photo"] = "No face detected. Please upload a clear, front-facing image."
            else:
                idx, chosen_box = pick_largest_face(face_locations)
                chosen_locs = [chosen_box] if (idx is not None and chosen_box) else []
                enc_all = face_recognition.face_encodings(image_np, chosen_locs)
                if not enc_all:
                    errors["photo"] = "Unable to compute a face encoding. Try a clearer image."
                else:
                    chosen_encoding = enc_all[0]
        except Exception as e:
            app.logger.exception("Face encoding error: %s", e)
            errors["photo"] = "Unable to process the uploaded photo."

        if errors:
            try: os.remove(photo_abs_path)
            except Exception: pass
            return render_template(
                "register.html",
                filename=None,
                errors=errors,
                form_data=form_data,
                focus_field="photo",
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 400

        # Duplicate short-circuit (if your DB can search by encoding)
        try:
            dup_candidates = find_person_by_face(
                chosen_encoding, tolerance=0.45, max_results=1, debug=False
            ) or []
        except Exception:
            dup_candidates = []

        if dup_candidates:
            c = dict(dup_candidates[0])
            try:
                conf = float(c.get("match_confidence", 0.0))
            except Exception:
                conf = 0.0
            if conf >= 90.0:
                try: os.remove(photo_abs_path)
                except Exception: pass
                flash("This person appears to be already registered. Showing existing record.", "success")
                return redirect(url_for("home"))

        # Build person record
        try:
            age_val = int(form_data.get("age")) if form_data.get("age") else None
        except ValueError:
            age_val = None

        person_data = {
            "full_name": (form_data.get("full_name") or "").strip(),
            "age": age_val,
            "gender": (form_data.get("gender") or "").strip(),
            "guardian_name": (form_data.get("guardian_name") or "").strip(),
            "phone_number": (form_data.get("phone_number") or "").strip(),
            "address": (form_data.get("address") or "").strip(),
            "last_seen": (form_data.get("last_seen") or "").strip(),
            "photo_path": os.path.basename(photo_abs_path) if photo_abs_path else None,
            "face_encoding": chosen_encoding.tolist(),
            "created_by": current_user.id if current_user.is_authenticated else None,
            "registered_by_name": (form_data.get("registered_by_name") or "").strip(),
            "registered_by_phone": (form_data.get("registered_by_phone") or "").strip(),
            "registered_by_relation": (form_data.get("registered_by_relation") or "").strip(),
            "detected_faces_count": len(face_locations),
            "chosen_face_box": [int(v) for v in (chosen_box or (0, 0, 0, 0))]
        }

        # Save to DB
        try:
            register_person_to_db(person_data)
            flash("Person registered successfully!", "success")
            return redirect(url_for("home"))
        except Exception as e:
            app.logger.exception("DB insert failed: %s", e)
            flash("Failed to save person to DB.", "error")
            if photo_abs_path:
                try: os.remove(photo_abs_path)
                except Exception: pass
            return render_template(
                "register.html",
                filename=None,
                errors={"server": "DB error"},
                form_data=form_data,
                focus_field=None,
                RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
            ), 500

    # GET
    return render_template(
        "register.html",
        filename=None,
        form_data={},
        errors={},
        focus_field=None,
        RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY
    )


# --- Search Person (largest-face + optional name/gender filter) ---
import numpy as np
@app.route("/search", methods=["GET", "POST"])
def search():
    # ---------------- Tunables ----------------
    MATCH_CONFIDENCE_THRESHOLD = 60.0   # only show matches >= this
    MAX_RESULTS = 10

    # Post-face-match re-ranking / filtering (applies only if user provided fields)
    USE_NAME_FILTER = True
    USE_GENDER_FILTER = True

    # Soft vs strict behavior
    REQUIRE_NAME_AGREEMENT = False     # if True: drop candidates whose name doesn't meet threshold
    REQUIRE_GENDER_AGREEMENT = False   # if True: drop candidates whose gender mismatches

    NAME_SIM_THRESHOLD = 0.62          # similarity threshold when strict or to get strong bonus
    NAME_SOFT_BONUS = 10.0             # how many points to add for a good name match (soft mode)
    GENDER_SOFT_BONUS = 7.0            # how many points to add if genders match (soft mode)
    # -----------------------------------------

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

    # ---------- helpers for largest-face ----------
    def pick_largest_face(face_locs):
        if not face_locs:
            return None, None
        areas = []
        for (t, r, b, l) in face_locs:
            w = max(0, r - l)
            h = max(0, b - t)
            areas.append(w * h)
        idx = int(np.argmax(areas))
        return idx, face_locs[idx]

    # ---------- helpers for name/gender filtering ----------
    import re, difflib

    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def name_similarity(a: str, b: str) -> float:
        a, b = _norm(a), _norm(b)
        if not a or not b:
            return 0.0
        base = difflib.SequenceMatcher(None, a, b).ratio()
        toks = a.split()
        if toks and all(t in b for t in toks):
            base = max(base, min(0.99, 0.85 + 0.03 * len(toks)))
        return float(base)

    def gender_matches(query_gender: str, candidate_gender: str) -> bool:
        q = (query_gender or "").strip().lower()
        c = (candidate_gender or "").strip().lower()
        if not q or not c:
            return False
        m = {"m":"male","f":"female","o":"other"}
        q = m.get(q, q)
        c = m.get(c, c)
        return q == c

    if request.method == "POST":
        searched = True

        # Optional user-specified fields (used for filter/rerank after face match)
        query_name = (request.form.get("full_name") or "").strip()
        query_gender = (request.form.get("gender") or "").strip()

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

            # Resize for runtime speed (encoding copy)
            max_dim_for_runtime = 1600
            if max(img.width, img.height) > max_dim_for_runtime:
                scale = max_dim_for_runtime / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save compressed JPEG (optional temporary)
            temp_jpg_path = tmp_abs_path + "_compressed.jpg"
            img.convert("RGB").save(temp_jpg_path, format="JPEG", quality=85, optimize=True)
            tmp_abs_path = temp_jpg_path

            image_np = np.array(img)
            face_locations = face_recognition.face_locations(image_np, model="hog")
            if not face_locations:
                try:
                    log_best_match_search(os.path.basename(tmp_abs_path), [])
                except Exception:
                    logger.exception("Failed to record/log search")
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({
                        "success": True,
                        "results": [],
                        "searched": True,
                        "uploaded_photo": uploaded_photo,
                        "face_locations": []
                    })
                return render_template(
                    "search.html",
                    results=[],
                    uploaded_photo=uploaded_photo,
                    searched=True,
                    face_locations=[]
                )

            idx, chosen_box = pick_largest_face(face_locations)
            chosen_locs = [chosen_box] if (idx is not None and chosen_box) else []
            encodings = face_recognition.face_encodings(image_np, chosen_locs)

            all_matches = []
            if encodings:
                face_encoding = encodings[0]

                # IMPORTANT: ensure your DB encodings are np.float64 (or compatible) and length 128
                candidates = find_person_by_face(face_encoding, tolerance=0.6, max_results=50, debug=False) or []

                seen_ids = set()
                for cand in candidates:
                    cand = dict(cand)
                    cand["detected_face_index"] = 1
                    if chosen_box:
                        cand["matched_face_location"] = [int(v) for v in chosen_box]

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

                # Enrich with DB fields to apply name/gender filters
                enriched = []
                for cand in all_matches:
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

                filtered = []
                for cand in enriched:
                    keep = True
                    bonus = 0.0

                    # Gender
                    if USE_GENDER_FILTER and query_gender:
                        gm = gender_matches(query_gender, cand.get("gender"))
                        if REQUIRE_GENDER_AGREEMENT and not gm:
                            keep = False
                        elif gm:
                            bonus += GENDER_SOFT_BONUS

                    # Name
                    if USE_NAME_FILTER and query_name and cand.get("full_name"):
                        sim = name_similarity(query_name, cand.get("full_name"))
                        cand["_name_similarity"] = sim
                        if REQUIRE_NAME_AGREEMENT and sim < NAME_SIM_THRESHOLD:
                            keep = False
                        elif sim >= NAME_SIM_THRESHOLD:
                            bonus += NAME_SOFT_BONUS
                    else:
                        cand["_name_similarity"] = None

                    if keep:
                        cand["_rank_score"] = float(cand["match_confidence"]) + bonus
                        filtered.append(cand)

                if not filtered:
                    results = []
                else:
                    results = sorted(
                        filtered,
                        key=lambda x: (x.get("_rank_score", 0.0), x.get("match_confidence", 0.0)),
                        reverse=True
                    )[:MAX_RESULTS]
            else:
                results = []

            try:
                uploaded_name_for_log = os.path.basename(tmp_abs_path) if tmp_abs_path else uploaded_photo
                log_best_match_search(uploaded_name_for_log, results)
            except Exception:
                logger.exception("Failed to record/log search")

        except Exception:
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
            scrubbed = []
            for c in results:
                c2 = dict(c)
                c2.pop("_rank_score", None)
                c2.pop("_name_similarity", None)
                scrubbed.append(c2)
            return jsonify({
                "success": True,
                "results": scrubbed,
                "searched": searched,
                "uploaded_photo": uploaded_photo,
                "face_locations": [list(chosen_box)] if face_locations else []
            })

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

    page = request.form.get("page", 1, type=int)

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
    page = request.form.get("page", 1, type=int)
    ids = request.form.getlist("selected_ids")

    if not ids:
        flash("No persons selected for deletion.", "warning")
        return redirect(url_for("admin_dashboard", page=page))

    try:
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

# --- Admin debug endpoint ---
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
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",      # no eventlet/gevent needed
    logger=False,
    engineio_logger=os.getenv("ENGINEIO_LOG", "0") in ("1", "true", "yes", "on"),
    ping_interval=25,
    ping_timeout=60,
)

# ----------------- Run Flask + SocketIO -----------------
if __name__ == "__main__":
    import socket

    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))   # learn the outbound iface
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    host_ip = get_local_ip()
    port = int(os.getenv("PORT", "5001"))

    print("🚀 Server starting!")
    print(f"  Desktop: http://127.0.0.1:{port}")
    print(f"  Mobile (same Wi-Fi): http://{host_ip}:{port}")

    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=(str(os.getenv("DEV_MODE", "true")).lower() in ("1", "true", "yes", "on")),
        use_reloader=False,              # avoid double-start/version bump
        allow_unsafe_werkzeug=True,     # quiets warnings in dev
        # ssl_context="adhoc",          # <- optional if you want HTTPS locally
    )
