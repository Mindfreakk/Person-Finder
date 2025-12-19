# app.py (top)
import os
import io
import uuid
import base64
import logging
import json
import secrets
import time
import socket
import csv
from math import ceil
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Tuple, List
from urllib.parse import urljoin

from dotenv import load_dotenv
from PIL import Image, ExifTags, ImageOps
import qrcode
import requests
import numpy as np
import face_recognition
from werkzeug.routing import BuildError

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
    session,
    current_app,
    Response,
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from flask_mail import Mail, Message
from flask_socketio import SocketIO

from database import (
    db,
    initialize_database,
    Person,
    SearchLog,
    PushSubscription,
    User,
    Feedback,
    register_person_to_db,
    get_all_registered_people,
    get_person_by_id,
    delete_person_by_id,
    find_person_by_face,
    log_best_match_search,
    get_stats as db_get_stats,
    debug_find_person_by_image,
    clear_people_encodings_cache,
)
ENV: str = os.getenv("FLASK_ENV", "development").lower()
IS_PROD: bool = ENV == "production"

# ----------------- Load environment -----------------
load_dotenv()
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "<YOUR_PRIVATE_KEY>")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "<YOUR_PUBLIC_KEY>")
VAPID_CLAIMS = {"sub": os.getenv("VAPID_SUBJECT", "mailto:you@example.com")}
DEV_MODE = str(os.getenv("DEV_MODE", "true")).lower() in ("1", "true", "yes", "on")

# ----------------- Flask App Config -----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET") or os.urandom(24)

# Ensure database is always inside project root (next to app.py)
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "personfinder.db")
db_uri = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")


# ---- Mail defaults / coercion helpers ----
def _bool_env(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")


MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
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
    # Accepts large uploads — upper bound here; your routes handle specifics
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

# ----------------- Version Control (CI-friendly + local auto-bump) -----------------
# Prefer CI-managed version.txt; fall back to local VERSION
_VERSION_CANDIDATES = [
    os.path.join(basedir, "version.txt"),  # written by CI (GitHub Actions)
    os.path.join(basedir, "VERSION"),  # local/dev fallback
]
for _p in _VERSION_CANDIDATES:
    if os.path.exists(_p):
        VERSION_FILE = _p
        break
else:
    VERSION_FILE = _VERSION_CANDIDATES[1]  # default to local VERSION

_version_cache = {"mtime": None, "val": "1.0.0"}

def _ensure_file(path: str, default_val: str = "1.0.0") -> None:
    # Ensure parent dir exists and file exists (with default content if needed)
    parent = os.path.dirname(path) or basedir
    os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(default_val + "\n")
            # update mtime/val cache
            st = os.stat(path)
            _version_cache["mtime"] = st.st_mtime
            _version_cache["val"] = default_val
        except Exception:
            # best-effort; ignore errors here
            pass

def get_app_version() -> str:
    """
    Return current version string.
    Re-reads the version file only if the mtime has changed.
    """
    try:
        st = os.stat(VERSION_FILE)
        st_mtime = st.st_mtime

        if _version_cache["mtime"] != st_mtime:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                val = (f.read() or "").strip() or "1.0.0"
            _version_cache["mtime"] = st_mtime
            _version_cache["val"] = val

    except Exception as e:
        # Fail safe: never crash app due to version
        print(f"[version] read failed: {e}")

    return _version_cache["val"]

def _write_version(new_val: str) -> None:
    """
    Overwrite the version file and update cache immediately.
    """
    try:
        _ensure_file(VERSION_FILE, default_val=new_val)
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(new_val.rstrip() + "\n")
        try:
            st = os.stat(VERSION_FILE)
            _version_cache["mtime"] = st.st_mtime
        except Exception:
            _version_cache["mtime"] = None
        _version_cache["val"] = new_val
    except Exception:
        logger.exception("Failed to write version file.")

def bump_version(which: str = "patch") -> str:
    """
    3-part version MAJOR.MINOR.PATCH with 0..99 caps on MINOR/PATCH.
    Rolls over (99->0) and cascades upward.
    """
    _ensure_file(VERSION_FILE, default_val="1.0.0")
    cur = get_app_version()
    try:
        major, minor, patch = [int(x) for x in (cur.split(".")[:3] + ["0", "0"])[:3]]
    except Exception:
        major, minor, patch = 1, 0, 0

    which = (which or "patch").lower()
    if which == "major":
        major, minor, patch = major + 1, 0, 0
    elif which == "minor":
        minor += 1
        if minor > 99:
            minor = 0
            major += 1
        patch = 0
    else:  # patch
        patch += 1
        if patch > 99:
            patch = 0
            minor += 1
            if minor > 99:
                minor = 0
                major += 1

    newv = f"{major}.{minor}.{patch}"
    _write_version(newv)
    return newv

# ---- Auto-bump on real app start (works with or without reloader & SocketIO) ----
# Modes:
#  - Default: If CI's version.txt exists, we DO NOT bump locally. If not, we bump patch once on start.
#  - PF_FORCE_LOCAL_BUMP=true -> always bump locally (even if version.txt exists).
#  - PF_LOCAL_BUMP_ON_START=false -> disable local bumping entirely.
_FORCE_LOCAL = str(os.getenv("PF_FORCE_LOCAL_BUMP", "false")).lower() in ("1", "true", "yes", "on")
_LOCAL_BUMP_ENABLED = str(os.getenv("PF_LOCAL_BUMP_ON_START", "true")).lower() in ("1", "true", "yes", "on")
_USING_CI_VERSION = os.path.basename(VERSION_FILE).lower() == "version.txt"

def _should_bump_now() -> bool:
    # If local bumping disabled, no.
    if not _LOCAL_BUMP_ENABLED:
        return False
    # If CI version file is in use, only bump when explicitly forced.
    if _USING_CI_VERSION and not _FORCE_LOCAL:
        return False
    # Handle dev reloader and non-reloader cases:
    # - With reloader: WERKZEUG_RUN_MAIN == "true" only in the serving child -> bump there.
    # - Without reloader (or alt servers like SocketIO/Gunicorn): env var may be absent -> bump on first import.
    flag = os.environ.get("WERKZEUG_RUN_MAIN")
    return (flag == "true") or (flag is None)

try:
    _ensure_file(VERSION_FILE, default_val="1.0.0")
    if _should_bump_now():
        bumped = bump_version("patch")
        print(f"[version] Auto-bump -> {bumped} (file: {VERSION_FILE}, ci_file={_USING_CI_VERSION})")
except Exception as _e:
    print(f"[version] Warning: could not bump version: {_e}")

# Optionally store it in config for convenience
app.config["APP_VERSION"] = get_app_version()

# ----------------- Context Processor (smart role-based header rendering) -----------------
@app.context_processor
def inject_globals():
    """
    Inject safe globals for templates:
      - app_version, current_year, VAPID/RECAPTCHA keys (existing)
      - pf_user: simplified dict of current_user (id, username, role) or None
      - pf_is_authenticated, pf_is_admin
    This avoids using fragile exprs like getattr(...) in templates and keeps logic small.
    """
    user = None
    is_authenticated = False
    is_admin = False

    try:
        if getattr(current_user, "is_authenticated", False):
            is_authenticated = True
            uname = getattr(current_user, "username", "") or ""
            role = getattr(current_user, "role", "") or "user"
            # normalize role to string
            role = str(role)
            is_admin = role.lower() == "admin"
            user = {"id": getattr(current_user, "id", None), "username": uname, "role": role}
    except Exception:
        # Fail-safe: leave values as False/None
        logger.exception("Error while preparing pf_user context.")

    return {
        "app_version": get_app_version(),
        "current_year": datetime.now().year,
        "personId": None,
        "VAPID_PUBLIC_KEY": app.config.get("VAPID_PUBLIC_KEY"),
        "RECAPTCHA_SITE_KEY": app.config.get("RECAPTCHA_SITE_KEY"),

        # NEW (for header + role based templates)
        "pf_user": user,
        "pf_is_authenticated": is_authenticated,
        "pf_is_admin": is_admin,
    }

# ----------------- Admin unread-count API used by header JS -----------------
from typing import Callable, TypeVar
from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user

F = TypeVar("F", bound=Callable[..., object])

def require_role(*roles: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))

            if roles and getattr(current_user, "role", None) not in roles:
                flash("Unauthorized access!", "error")
                return redirect(url_for("home"))

            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator

@app.get("/api/admin/alerts/unread-count")
@login_required
@require_role("admin")
def api_admin_unread_count():
    """
    Return JSON { "unread": N } for initial polling by the header.
    If your app has a helper to compute the real unread count, expose it as
      def get_unread_admin_alerts_count() -> int
    in your module (e.g. in database.py or helpers), otherwise this endpoint returns 0.
    """
    try:
        fn = globals().get("get_unread_admin_alerts_count")
        if callable(fn):
            cnt = int(fn() or 0)
        else:
            # Optional: try to look up SearchLog or an Alerts model (if present)
            try:
                # Example: if you have a SearchLog model and store 'notified' flags, adapt as needed.
                SearchLog_model = globals().get("SearchLog")
                if SearchLog_model is not None:
                    # fallback: 0 for now (implement real logic if you track alert rows)
                    cnt = 0
                else:
                    cnt = 0
            except Exception:
                cnt = 0
        return jsonify({"unread": cnt})
    except Exception as e:
        logger.exception("Error computing unread alerts count")
        return jsonify({"unread": 0})


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
            data={"secret": secret, "response": token},
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


def _csv_bytes_response(csv_text: str, filename: str):
    """
    Wrap CSV text in a BytesIO so send_file works reliably with UTF-8.
    """
    buf = io.BytesIO(csv_text.encode("utf-8-sig"))  # BOM helps Excel
    return send_file(
        buf,
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )


# ----------------- Initialize DB -----------------
db.init_app(app)
with app.app_context():
    initialize_database(app)
    print(f"🔎 Using database at: {db_path}")

    # ---- Ensure `phone_number` column exists (defensive) ----
    from sqlalchemy import inspect, text, func

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
    user = db.session.get(User, int(user_id))
    return UserLogin(user) if user else None


from typing import Callable, TypeVar
from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user

F = TypeVar("F", bound=Callable[..., object])

def require_role(*roles: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))

            if roles and getattr(current_user, "role", None) not in roles:
                flash("Unauthorized access!", "error")
                return redirect(url_for("home"))

            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator

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
        if filename and os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], filename))
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
                auth=subscription["keys"]["auth"],
            )
            db.session.add(new_sub)
            db.session.commit()
    except Exception:
        logger.exception("Failed to save push subscription")


def pick_largest_face(face_locations):
    """Shared helper to pick largest detected face by area."""
    if not face_locations:
        return None, None
    areas = [max(0, (b - t)) * max(0, (r - l)) for (t, r, b, l) in face_locations]
    idx = int(np.argmax(areas))
    return idx, face_locations[idx]


# ----------------- Utility Function -----------------
def get_stats():
    """Wrapper around db_get_stats with a defensive fallback."""
    try:
        return db_get_stats()
    except Exception:
        logger.exception("Error fetching stats via db_get_stats()")
        return {"registrations": 0, "searches": 0, "searches_traced": 0}


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


# ----------------- Feedback API -----------------
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    Accept feedback from:
      - JSON (AJAX / fetch)  -> returns JSON
      - HTML form POST      -> redirects with flash messages

    Expected fields:
      name    (optional)
      email   (required)
      rating  (required, 1–5)
      message (required)
    """

    is_json = request.is_json or (
        request.content_type
        and "application/json" in request.content_type.lower()
    )
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    # --------- Parse input ---------
    if is_json:
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip()
        rating_raw = data.get("rating")
        message = (data.get("message") or "").strip()
    else:
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip()
        rating_raw = request.form.get("rating")
        message = (request.form.get("message") or "").strip()

    errors = {}

    # --------- Validation ---------
    if not email:
        errors["email"] = "Email is required."
    elif "@" not in email or "." not in email.split("@")[-1]:
        errors["email"] = "Please provide a valid email address."

    try:
        rating = int(rating_raw)
    except (TypeError, ValueError):
        rating = None

    if rating is None:
        errors["rating"] = "Rating is required."
    elif not 1 <= rating <= 5:
        errors["rating"] = "Rating must be between 1 and 5."

    if not message or len(message) < 5:
        errors["message"] = "Please describe your experience (at least 5 characters)."

    # --------- Validation failure ---------
    if errors:
        if is_json or is_ajax:
            return jsonify({"ok": False, "errors": errors}), 400

        for msg in errors.values():
            flash(msg, "error")
        return redirect(request.referrer or url_for("home"))

    # --------- Metadata ---------
    page_url = (
        request.headers.get("X-Feedback-Page")
        or request.referrer
        or ""
    )

    user_agent = request.user_agent.string if request.user_agent else None

    ip_address = (
        request.headers.get("X-Forwarded-For", "")
        .split(",")[0]
        .strip()
        or request.remote_addr
    )

    # --------- Save to DB ---------
    try:
        fb = Feedback(
            name=name or None,
            email=email,
            rating=rating,
            message=message,
            page_url=page_url[:500] if page_url else None,
            user_agent=user_agent[:500] if user_agent else None,
            ip_address=ip_address[:100] if ip_address else None,
        )
        db.session.add(fb)
        db.session.commit()

    except Exception:
        db.session.rollback()

        if is_json or is_ajax:
            return jsonify({
                "ok": False,
                "errors": {"server": "Could not save feedback. Please try again."}
            }), 500

        flash("Could not save feedback. Please try again later.", "error")
        return redirect(request.referrer or url_for("home"))

    # --------- Admin Notification ---------
    try:
        emit_admin_alert(
            "new_feedback",
            {
                "rating": rating,
                "email": email,
            },
        )
    except Exception:
        pass  # non-fatal

    # --------- Success ---------
    if is_json or is_ajax:
        return jsonify({"ok": True})

    flash("Thank you! Your feedback has been submitted.", "success")
    return redirect(request.referrer or url_for("home"))


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

        # Block deactivated accounts
        if hasattr(user, "is_active") and user.is_active is False:
            flash("This account has been deactivated by an administrator.", "error")
            return redirect(url_for("login"))

        # Block accounts that still have an active verification/reset token
        if user.reset_token and user.reset_expiry and datetime.utcnow() <= user.reset_expiry:
            flash(
                "This account has a pending verification/reset link. "
                "Please use the link sent to your email first.",
                "error",
            )
            return redirect(url_for("login"))

        login_user(UserLogin(user))

        # 1) If coming from a protected page, go back there
        next_url = request.args.get("next")

        # Normalize role
        role = (getattr(user, "role", "") or "").strip().lower()

        # 2) Admins → admin_dashboard with success flash
        if role == "admin":
            flash("Logged in successfully.", "success")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("admin_dashboard"))

        # 3) Normal users → home with warning flash
        flash(
            "You logged in with user-level access. Admin dashboard access is restricted to administrator accounts only.",
            "error",
        )
        if next_url:
            return redirect(next_url)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    try:
        socketio.emit(
            "admin_disconnected",
            {"user": current_user.id},
        )
    except Exception:
        pass
    flash("Logged out successfully.", "success")
    result = redirect(url_for("login"))

    # ---- Auto-clear admin alerts when dashboard is visited ----
    try:
        socketio.emit(
            "admin_alert_cleared",
            {
                "by": current_user.id,
                "ts": int(time.time()),
            },
            namespace="/",
        )
    except Exception:
        pass
    return result

# ----------------- Admin Dashboard -----------------
@app.route("/admin/dashboard", methods=["GET"])
@login_required
@require_role("admin")
def admin_dashboard():
    # People pagination (10 per page)
    page = request.args.get("page", 1, type=int)
    per_page = 10

    people_pagination = (
        Person.query.order_by(Person.id.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    people = people_pagination.items
    total_pages = people_pagination.pages or 1

    # Users
    users = User.query.order_by(User.id.asc()).all()

    # Search logs (recent)
    search_logs = (
        SearchLog.query.order_by(SearchLog.ts.desc())
        .limit(100)
        .all()
    )

    # Existing stats helper
    stats = get_stats() or {}

    # Optional recent events for the Activity section (empty for now)
    recent_events = []

    # App version for the System Info card
    app_version = get_app_version()

    # ---- Feedback stats for dashboard ----
    try:
        from sqlalchemy import func as sa_func

        feedback_total = Feedback.query.count()
        feedback_avg_rating = db.session.query(sa_func.avg(Feedback.rating)).scalar()
        feedback_negative_count = Feedback.query.filter(Feedback.rating <= 2).count()

        # Histogram for ratings 1–5
        feedback_hist = [0, 0, 0, 0, 0]
        rows = (
            db.session.query(Feedback.rating, sa_func.count(Feedback.id))
            .group_by(Feedback.rating)
            .all()
        )
        for rating, count in rows:
            if 1 <= rating <= 5:
                feedback_hist[rating - 1] = count

        # Latest feedback entries for dashboard panel
        recent_feedback = (
            Feedback.query.order_by(Feedback.created_at.desc())
            .limit(5)
            .all()
        )
    except Exception:
        feedback_total = 0
        feedback_avg_rating = None
        feedback_negative_count = 0
        feedback_hist = [0, 0, 0, 0, 0]
        recent_feedback = []

        # ---- Dashboard role badge for logged-in user (Super Admin vs Admin) ----
    # This email is your *primary* Super Admin. Set in env, e.g. ADMIN_EMAIL=ammehz09@gmail.com
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

    # Use Session.get to avoid the SQLAlchemy legacy warning
    admin_user = (
        db.session.get(User, int(current_user.id))
        if current_user.is_authenticated
        else None
    )

    is_super_admin_current = bool(
        admin_user
        and super_admin_email
        and (admin_user.email or "").strip().lower() == super_admin_email
    )

    if is_super_admin_current:
        dashboard_role_label = "Super Admin"
        dashboard_role_slug = "super-admin"
        dashboard_role_icon = "👑"
    else:
        # Normal admin (users cannot access this dashboard otherwise)
        dashboard_role_label = "Admin"
        dashboard_role_slug = "admin"
        dashboard_role_icon = "🛡️"

    # lowercase display username for the header
    display_username = ""
    if admin_user and getattr(admin_user, "username", None):
        display_username = (admin_user.username or "").strip().lower()

    # Expose super admin info and actor info to the template
    primary_super_admin_email = super_admin_email
    actor_is_super_admin = is_super_admin_current
    actor_user_id = admin_user.id if admin_user else None

    return render_template(
        "admin_dashboard.html",
        # people
        people=people,
        page=page,
        total_pages=total_pages,
        # users
        users=users,
        # logs
        search_logs=search_logs,
        # stats/system
        stats=stats,
        app_version=app_version,
        recent_events=recent_events,
        # feedback
        feedback_total=feedback_total,
        feedback_avg_rating=feedback_avg_rating,
        feedback_negative_count=feedback_negative_count,
        feedback_hist=feedback_hist,
        recent_feedback=recent_feedback,
        # dashboard greeting + role pill
        dashboard_role_label=dashboard_role_label,
        dashboard_role_slug=dashboard_role_slug,
        dashboard_role_icon=dashboard_role_icon,
        display_username=display_username,
        # super-admin info for template logic
        super_admin_email=super_admin_email,
        primary_super_admin_email=primary_super_admin_email,
        # actor flags for template (used in users table)
        actor_is_super_admin=actor_is_super_admin,
        actor_user_id=actor_user_id,
    )


@app.route(
    "/admin/search-logs/export",
    methods=["GET"],
    endpoint="export_search_logs_csv",
)
@login_required
@require_role("admin")
def export_search_logs_csv():
    SearchLog_model = globals().get("SearchLog")
    if SearchLog_model is None:
        flash("Search log export is not configured on this installation.", "error")
        return _redirect_back_to_dashboard()

    logs = SearchLog_model.query.order_by(SearchLog_model.ts.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["timestamp", "searched_name", "matches"])

    for log in logs:
        ts = log.ts.strftime("%Y-%m-%d %H:%M:%S") if log.ts else ""
        writer.writerow([ts, log.uploaded_name or "", log.matches or ""])

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="search_logs.csv"'},
    )

@app.route("/admin/export-people", methods=["GET"])
@login_required
@require_role("admin")
def export_people_csv():
    """
    Export people data as CSV.
    This is defensive: if the Person model or query fails, it still returns a valid CSV with just a header.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row – adjust columns to match your Person model / DB schema
    writer.writerow([
        "id",
        "full_name",
        "age",
        "gender",
        "guardian_name",
        "phone_number",
        "address",
        "last_seen",
    ])

    try:
        # If your model is named differently, change Person → your model name
        people = Person.query.order_by(Person.id.asc()).all()
        for p in people:
            writer.writerow([
                getattr(p, "id", ""),
                getattr(p, "full_name", ""),
                getattr(p, "age", ""),
                getattr(p, "gender", ""),
                getattr(p, "guardian_name", ""),
                getattr(p, "phone_number", getattr(p, "phone", "")),
                getattr(p, "address", ""),
                getattr(p, "last_seen", ""),
            ])
    except Exception:
        # Fail silently – at least the CSV will contain a header
        pass

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=people.csv"},
    )


@app.route("/admin/export-users", methods=["GET"])
@login_required
@require_role("admin")
def export_users_csv():
    """
    Export user accounts as CSV.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row – adjust if your User model has different fields
    writer.writerow([
        "id",
        "username",
        "email",
        "role",
        "is_active",
        "created_at",
    ])

    try:
        users = User.query.order_by(User.id.asc()).all()
        for u in users:
            writer.writerow([
                getattr(u, "id", ""),
                getattr(u, "username", ""),
                getattr(u, "email", ""),
                getattr(u, "role", ""),
                getattr(u, "is_active", ""),
                getattr(u, "created_at", ""),
            ])
    except Exception as e:
        # Log but still return at least the header
        logger.exception("Error exporting users CSV: %s", e)

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=users.csv"},
    )


@app.route("/admin/export-feedback", methods=["GET"])
@login_required
@require_role("admin")
def export_feedback_csv():
    """
    Export feedback as CSV.
    Honours optional rating filter (?rating=1..5).
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "id",
        "name",
        "email",
        "rating",
        "message",
        "page_url",
        "user_agent",
        "ip_address",
        "created_at",
    ])

    rating_filter = request.args.get("rating", type=int)

    query = Feedback.query
    if rating_filter:
        query = query.filter(Feedback.rating == rating_filter)

    try:
        rows = query.order_by(Feedback.created_at.desc()).all()
        for fb in rows:
            writer.writerow([
                getattr(fb, "id", ""),
                getattr(fb, "name", ""),
                getattr(fb, "email", ""),
                getattr(fb, "rating", ""),
                (getattr(fb, "message", "") or "").replace("\n", " ").replace("\r", " "),
                getattr(fb, "page_url", ""),
                getattr(fb, "user_agent", ""),
                getattr(fb, "ip_address", ""),
                getattr(fb, "created_at", ""),
            ])
    except Exception as e:
        logger.exception("Error exporting feedback CSV: %s", e)

    output.seek(0)

    filename = "feedback.csv"
    if rating_filter:
        filename = f"feedback_rating_{rating_filter}.csv"

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route(
    "/admin/feedback/<int:feedback_id>/delete",
    methods=["POST"],
    endpoint="admin_delete_feedback",
)
@login_required
@require_role("admin")
def admin_delete_feedback(feedback_id):
    """
    Delete a feedback entry.
    Only the primary admin (super admin) – the one matching ADMIN_EMAIL – may delete.
    """
    # Who is super admin?
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    current = User.query.get(int(current_user.id))

    if not current:
        flash("Current user not found.", "error")
        return redirect(url_for("admin_feedback"))

    if not super_admin_email or (current.email or "").strip().lower() != super_admin_email:
        flash("Only the primary (super) admin can delete feedback.", "error")
        # Preserve pagination + rating filter if provided
        page = request.form.get("page", 1)
        rating = request.form.get("rating") or None
        return redirect(url_for("admin_feedback", page=page, rating=rating))

    fb = Feedback.query.get_or_404(feedback_id)

    try:
        db.session.delete(fb)
        db.session.commit()
        flash("Feedback entry deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting feedback: {e}", "error")

    page = request.form.get("page", 1)
    rating = request.form.get("rating") or None
    return redirect(url_for("admin_feedback", page=page, rating=rating))


# ----------------- Admin: Feedback Listing -----------------
@app.route("/admin/feedback", methods=["GET"])
@login_required
@require_role("admin")  # Admin & super-admin
def admin_feedback():
    """
    Paginated list of user feedback, newest first.
    Optional rating filter: /admin/feedback?rating=5
    """
    page = request.args.get("page", 1, type=int)
    rating_filter = request.args.get("rating", type=int)
    per_page = 20

    # Build query with optional rating filter
    query = Feedback.query
    if rating_filter:
        query = query.filter(Feedback.rating == rating_filter)

    pagination = query.order_by(Feedback.created_at.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False,
    )

    stats = get_stats()  # reuse your existing stats helper

    # Determine if current user is "super admin" (primary admin email)
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    admin_user = User.query.get(int(current_user.id))
    is_super_admin = bool(
        admin_user
        and super_admin_email
        and (admin_user.email or "").strip().lower() == super_admin_email
    )

    return render_template(
        "admin_feedback.html",
        feedback_items=pagination.items,
        feedback_pagination=pagination,
        total_feedback=pagination.total,
        current_page=pagination.page,
        total_pages=pagination.pages,
        stats=stats,
        rating_filter=rating_filter,
        is_super_admin=is_super_admin,
    )


# ----------------- Auto-detect LAN IP & build external URLs -----------------
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


from werkzeug.routing import BuildError  # you already import this; keep it

def build_external_url(endpoint: str, **values) -> str:
    """
    Build an external URL that prefers PUBLIC_BASE_URL, else falls back to LAN IP + PORT.
    This version tries url_for() but has a robust fallback when url_for cannot build the endpoint
    (avoids BuildError causing a 500 in production).
    """
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if not base:
        local_ip = get_local_ip()
        port = os.getenv("PORT", "5001")
        base = f"http://{local_ip}:{port}"

    # Try the normal Flask url_for approach first
    try:
        rel = url_for(endpoint, _external=False, **values).lstrip("/")
    except BuildError:
        # Fallbacks when url_for can't build the endpoint:
        # 1) Common reset-password route pattern: "reset-password/<token>"
        # 2) Try replacing underscores with hyphens and append values as path segments
        # 3) Last resort: join key/value pairs as path segments
        token = values.get("token") or values.get("id") or None
        if token and ("reset" in endpoint or "password" in endpoint or "reset_password" in endpoint):
            rel = f"reset-password/{token}"
        else:
            # convert endpoint name to a path-ish string
            rel_base = str(endpoint).replace("_", "-").lstrip("/")
            if values:
                # append values in order (use only primitives)
                try:
                    segs = "/".join(str(v) for v in values.values())
                    rel = f"{rel_base}/{segs}"
                except Exception:
                    rel = rel_base
            else:
                rel = rel_base
        rel = rel.lstrip("/")

    return urljoin(base.rstrip("/") + "/", rel)


# ----------------- Mail helper -----------------
def send_mail(
    subject: str,
    recipients: List[str],
    *,
    body: str = "",
    html: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
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
    On POST, sends a reset email; on failure shows a generic message to users
    while logging useful details for developers when DEV_MODE is true.
    """
    token_valid = False

    if request.method == "POST":
        email = (request.form.get("email", "") or "").strip().lower()
        admin_env = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

        # Prefer looking up the canonical admin user by configured ADMIN_EMAIL
        try:
            user = None
            if admin_env:
                # optionally require role="admin" if you want extra safety:
                # user = User.query.filter_by(email=admin_env, role="admin").first()
                user = User.query.filter_by(email=admin_env).first()
            if not user:
                user = User.query.filter_by(username="admin").first()
        except Exception as e:
            app.logger.exception("Error querying admin user for forgot_password: %s", e)
            flash("If that email is registered, a reset link has been sent.", "success")
            return render_template("forgot_password.html", token_valid=token_valid)

        # Don't reveal existence. Log debug info in DEV only.
        if not user or email != admin_env:
            if DEV_MODE:
                app.logger.debug(
                    "forgot_password: submitted_email=%s, resolved_admin_email=%s, found_user=%s",
                    email,
                    admin_env,
                    getattr(user, "username", None) if user else None,
                )
            flash("If that email is registered, a reset link has been sent.", "success")
            return render_template("forgot_password.html", token_valid=token_valid)

        # Create reset token + expiry, persist to DB
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_expiry = datetime.utcnow() + timedelta(minutes=15)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Failed to save reset token for admin: %s", e)
            # Generic message for users
            flash("If that email is registered, a reset link has been sent.", "success")
            return render_template("forgot_password.html", token_valid=token_valid)

        # Build external reset link (LAN-friendly / PUBLIC_BASE_URL-aware)
        reset_link = build_external_url("reset_password", token=token)

        # Send email (plain-text + HTML)
        body = f"""Reset your PersonFinder admin password

You requested a password reset. Click the link below to set a new password (valid for 15 minutes):

{reset_link}

If you did not request this, you can ignore this email.
"""

        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width">
  </head>
  <body style="margin:0;padding:0;background:#f3f6fb;font-family:Helvetica,Arial,sans-serif;">

    <div style="display:none;max-height:0px;overflow:hidden;">
      Reset your PersonFinder admin password — link valid 15 minutes.
    </div>

    <table width="100%" style="background:#f3f6fb;padding:24px 0;">
      <tr>
        <td align="center">
          <table width="600"
                 style="background:#ffffff;border-radius:12px;overflow:hidden;
                        box-shadow:0 6px 18px rgba(20,30,60,0.08);">

            <tr>
              <td style="padding:28px 36px;text-align:center;
                         background:linear-gradient(90deg,#2563eb,#7c3aed);color:#fff;">
                <h1 style="margin:0;font-size:20px;font-weight:700;">
                  PersonFinder — Password Reset
                </h1>
              </td>
            </tr>

            <tr>
              <td style="padding:28px 36px;color:#1f2937;">
                <p style="font-size:15px;margin:0 0 14px;">Hello,</p>
                <p style="font-size:15px;margin:0 0 20px;line-height:1.5;">
                  A password reset was requested for your account.
                  Click the button below — the link is valid for <strong>15 minutes</strong>.
                </p>

                <p style="text-align:center;margin:20px 0;">
                  <a href="{reset_link}"
                    style="padding:12px 22px;border-radius:8px;
                           background:linear-gradient(90deg,#2563eb,#d946ef);
                           color:#fff;text-decoration:none;font-weight:600;">
                    Reset My Password
                  </a>
                </p>

                <p style="font-size:13px;color:#6b7280;margin-top:20px;">
                  Or paste this link in your browser:<br>
                  <a href="{reset_link}" style="color:#2563eb;">{reset_link}</a>
                </p>
              </td>
            </tr>

            <tr>
              <td style="background:#f8fafc;padding:18px 36px;text-align:center;
                         color:#9aa4b2;font-size:12px;">
                PersonFinder — Secure Access • Link expires in 15 minutes
              </td>
            </tr>

          </table>
        </td>
      </tr>
    </table>

  </body>
</html>
"""

        ok, err = send_mail("Reset Admin Password", [email], body=body, html=html)

        if not ok:
            # Log full error and (in DEV) the link for local testing; show generic message to user.
            app.logger.error("Failed to send reset email to %s: %s", email, err)
            if DEV_MODE:
                # Print reset URL for convenience when developing locally
                print("\n[DEV] Email send failed. Reset URL (copy/paste):\n", reset_link, "\n")
                app.logger.debug("DEV reset link (admin): %s", reset_link)

            # Generic message to avoid revealing whether the address is valid or whether email failed
            flash("If that email is registered, a reset link has been sent.", "success")
            return render_template("forgot_password.html", token_valid=token_valid)

        # On success show the same generic message (prevents enumeration)
        flash("If that email is registered, a reset link has been sent.", "success")
        return render_template("forgot_password.html", token_valid=token_valid)

    return render_template("forgot_password.html", token_valid=token_valid)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    # find user by token
    user = User.query.filter_by(reset_token=token).first()

    # token must exist and not be expired
    if (
        not user
        or not getattr(user, "reset_expiry", None)
        or datetime.utcnow() > user.reset_expiry
    ):
        flash("Invalid or expired link.", "error")
        return render_template("forgot_password.html", token_valid=False)

    # POST → update password
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        confirm = (request.form.get("confirm_password") or "").strip()

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("reset_password.html", token=token)

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("reset_password.html", token=token)

        user.password_hash = generate_password_hash(password)
        user.reset_token = None
        user.reset_expiry = None
        db.session.commit()

        flash("Password updated! Login now.", "success")
        return redirect(url_for("login"))

    # GET → Show reset form
    return render_template("reset_password.html", token=token)

# Verify User
@app.route("/verify-user/<token>")
def verify_user(token):
    user = User.query.filter_by(reset_token=token).first()

    if not user:
        flash("Invalid verification link.", "error")
        return redirect(url_for("login"))

    if user.reset_expiry and datetime.utcnow() > user.reset_expiry:
        flash("Verification link has expired. Ask an admin to resend it.", "error")
        return redirect(url_for("login"))

    # Mark as verified by clearing token/expiry
    user.reset_token = None
    user.reset_expiry = None
    db.session.commit()

    flash("Your account has been verified. You can now log in.", "success")
    return redirect(url_for("login"))


# ---- Optional: DEV-only test route to verify SMTP quickly ----
if DEV_MODE:

    @app.route("/debug/send-test-email")
    def debug_send_test_email():
        recipient = (os.getenv("ADMIN_EMAIL") or MAIL_USERNAME)
        ok, err = send_mail(
            "PersonFinder Test Email",
            [recipient],
            body="This is a test email from PersonFinder.",
        )
        if ok:
            return f"✅ Test email sent to {recipient}"
        return f"❌ Failed to send test email: {err}", 500


# --- Register Person (no size/resolution limits; stream-safe; largest-face; duplicate short-circuit) ---
# Disable Pillow decompression bomb checks (accept all megapixels)
Image.MAX_IMAGE_PIXELS = None

# Ensure upload dir exists (defensive; config already has one)
app.config.setdefault("UPLOAD_FOLDER", os.getenv("UPLOAD_FOLDER", "static/uploads"))
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    """If some upstream still throws 413, render the form with a helpful message."""
    recaptcha_site_key = os.getenv("RECAPTCHA_SITE_KEY", "")
    errors = {
        "photo": "The image was rejected by an upstream limit. Please try again now; "
        "the server accepts very large images. If it persists, ask the admin to raise proxy limits."
    }
    return (
        render_template(
            "register.html",
            filename=None,
            errors=errors,
            form_data=request.form if request.form else {},
            focus_field="photo",
            RECAPTCHA_SITE_KEY=recaptcha_site_key,
        ),
        413,
    )


# Route to create a new user for another person
@app.route("/admin/create-user", methods=["POST"])
@login_required
@require_role("admin")
def admin_create_user():
    # Fields coming from the modal
    username = (request.form.get("new_username") or "").strip()
    email = (request.form.get("new_email") or "").strip().lower()
    phone = (request.form.get("new_phone") or "").strip()
    role = (request.form.get("new_role") or "user").strip().lower()
    raw_password = (request.form.get("new_password") or "").strip()
    confirm_password = (request.form.get("new_password_confirm") or "").strip()
    admin_password = (request.form.get("admin_password") or "").strip()

    if not username or not email:
        flash("Username and email are required.", "error")
        return redirect(url_for("admin_dashboard"))

    # Verify admin password
    admin_user = User.query.get(int(current_user.id))
    if not admin_user or not check_password_hash(admin_user.password_hash, admin_password):
        flash("Admin password is incorrect. User was not created.", "error")
        return redirect(url_for("admin_dashboard"))

    # Unique username / email
    existing = User.query.filter(
        (User.username == username) | (User.email == email)
    ).first()
    if existing:
        flash("A user with this username or email already exists.", "error")
        return redirect(url_for("admin_dashboard"))

    if raw_password:
        if len(raw_password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("admin_dashboard"))
        if raw_password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("admin_dashboard"))
        chosen_password = raw_password
    else:
        chosen_password = secrets.token_urlsafe(8)  # auto temp password

    password_hash = generate_password_hash(chosen_password)

    # Use reset_token/expiry as a "verification token"
    verification_token = secrets.token_urlsafe(32)
    verification_expiry = datetime.utcnow() + timedelta(days=2)

    if role not in ("admin", "user"):
        role = "user"

    # IMPORTANT: do NOT pass phone / phone_number in the constructor
    new_user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=role,
    )

    # Now set phone only if the attribute exists on the model
    try:
        if hasattr(new_user, "phone"):
            new_user.phone = phone or None
        elif hasattr(new_user, "phone_number"):
            new_user.phone_number = phone or None
    except Exception:
        pass

    # Set verification fields only if they exist on the model
    if hasattr(new_user, "reset_token"):
        new_user.reset_token = verification_token
    if hasattr(new_user, "reset_expiry"):
        new_user.reset_expiry = verification_expiry

    db.session.add(new_user)
    db.session.commit()

    # Send verification email
    verify_link = build_external_url("verify_user", token=verification_token)
    login_link = build_external_url("login")

    subject = "Your PersonFinder admin account"
    body = f"""Hello {username},

An account has been created for you on PersonFinder.

Login page: {login_link}
Email: {email}
Temporary password: {chosen_password}

Before you can log in, you must verify your email by opening this link:
{verify_link}

This link will expire in 2 days.
"""

    html = f"""
    <p>Hello <strong>{username}</strong>,</p>
    <p>An account has been created for you on <strong>PersonFinder</strong>.</p>
    <ul>
      <li><strong>Login page:</strong> <a href="{login_link}">{login_link}</a></li>
      <li><strong>Email:</strong> {email}</li>
      <li><strong>Temporary password:</strong> <code>{chosen_password}</code></li>
    </ul>
    <p>Before you can log in, please verify your email:</p>
    <p><a href="{verify_link}">Verify my account</a></p>
    <p>This link will expire in 2 days.</p>
    """

    ok, err = send_mail(subject, [email], body=body, html=html)
    if not ok:
        flash(f"User created, but failed to send email: {err}", "error")
    else:
        flash("User created and verification email sent.", "success")

        emit_admin_alert(
    "user_created",
    {
        "username": username,
        "role": role,
    },
)

    return redirect(url_for("admin_dashboard"))


# ---------- Register ----------
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
    recaptcha_site_key = os.getenv("RECAPTCHA_SITE_KEY", "")

    # Face settings
    FACE_MODEL = "hog"  # "cnn" requires GPU/dlib-cnn; "hog" is CPU-friendly
    DUPLICATE_TOLERANCE = 0.45
    DUPLICATE_MIN_CONF = 90.0

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    UPLOAD_FOLDER = app.config.get("UPLOAD_FOLDER", "static/uploads")

    # ---------- helpers ----------
    def _now_ts():
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

    def save_as_jpeg_without_resizing(
        pil_img: Image.Image, out_abs_path: str, quality: int = 90
    ):
        """Save as JPEG (no resizing), reasonable quality (no size cap enforced)."""
        os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
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
            pil = pil.resize(
                (int(pil.width * scale), int(pil.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return np.array(pil)

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
            "full_name",
            "age",
            "gender",
            "guardian_name",
            "phone_number",
            "address",
            "last_seen",
            "registered_by_name",
            "registered_by_phone",
            "registered_by_relation",
            "agreement",
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
            "registered_by_relation": "Relation is required.",
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
            return (
                render_template(
                    "register.html",
                    filename=None,
                    errors=errors,
                    form_data=form_data,
                    focus_field=focus_field,
                    RECAPTCHA_SITE_KEY=recaptcha_site_key,
                ),
                400,
            )

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
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
            return (
                render_template(
                    "register.html",
                    filename=None,
                    errors=errors,
                    form_data=form_data,
                    focus_field=focus_field or "photo",
                    RECAPTCHA_SITE_KEY=recaptcha_site_key,
                ),
                400,
            )

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
            try:
                os.remove(photo_abs_path)
            except Exception:
                pass
            return (
                render_template(
                    "register.html",
                    filename=None,
                    errors=errors,
                    form_data=form_data,
                    focus_field="photo",
                    RECAPTCHA_SITE_KEY=recaptcha_site_key,
                ),
                400,
            )

        # Duplicate short-circuit (if your DB can search by encoding)
        try:
            dup_candidates = find_person_by_face(
                chosen_encoding, tolerance=DUPLICATE_TOLERANCE, max_results=1, debug=False
            ) or []
        except Exception:
            dup_candidates = []

        if dup_candidates:
            c = dict(dup_candidates[0])
            try:
                conf = float(c.get("match_confidence", 0.0))
            except Exception:
                conf = 0.0
            if conf >= DUPLICATE_MIN_CONF:
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
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
            "chosen_face_box": [int(v) for v in (chosen_box or (0, 0, 0, 0))],
        }

        try:
            register_person_to_db(person_data)
            emit_admin_alert(
    "person_registered",
    {
        "name": person_data.get("full_name"),
        "phone": person_data.get("phone_number"),
    },
)
            flash("Person registered successfully!", "success")
            return redirect(url_for("home"))
        except Exception as e:
            app.logger.exception("DB insert failed: %s", e)
            flash("Failed to save person to DB.", "error")
            if photo_abs_path:
                try:
                    os.remove(photo_abs_path)
                except Exception:
                    pass
            return (
                render_template(
                    "register.html",
                    filename=None,
                    errors={"server": "DB error"},
                    form_data=form_data,
                    focus_field=None,
                    RECAPTCHA_SITE_KEY=recaptcha_site_key,
                ),
                500,
            )

    # GET
    return render_template(
        "register.html",
        filename=None,
        form_data={},
        errors={},
        focus_field=None,
        RECAPTCHA_SITE_KEY=recaptcha_site_key,
    )


# --- Search Person (largest-face + optional name/gender filter) ---
@app.route("/search", methods=["GET", "POST"])
def search():
    # ---------------- Tunables ----------------
    MATCH_CONFIDENCE_THRESHOLD = 60.0
    MAX_RESULTS = 10

    USE_NAME_FILTER = True
    USE_GENDER_FILTER = True

    REQUIRE_NAME_AGREEMENT = False
    REQUIRE_GENDER_AGREEMENT = False

    NAME_SIM_THRESHOLD = 0.62
    NAME_SOFT_BONUS = 10.0
    GENDER_SOFT_BONUS = 7.0
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
            "id",
            "full_name",
            "age",
            "gender",
            "guardian_name",
            "phone_number",
            "address",
            "last_seen",
            "photo_path",
            "registered_by_name",
            "registered_by_phone",
            "registered_by_relation",
            "notes",
            "last_seen_location",
        ):
            val = getattr(p, attr, None)
            if val is not None:
                out[attr] = val
        created = getattr(p, "created_at", None) or getattr(p, "registration_date", None)
        if created and "registration_date" not in out:
            out["registration_date"] = (
                created.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(created, datetime)
                else created
            )
        return out

    import re
    import difflib

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
        mapping = {"m": "male", "f": "female", "o": "other"}
        return mapping.get(q, q) == mapping.get(c, c)

    if request.method == "POST":
        searched = True

        query_name = (request.form.get("full_name") or "").strip()
        query_gender = (request.form.get("gender") or "").strip()

        photo = request.files.get("photo")
        if not photo or not photo.filename:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": "Please upload a photo."}), 400
            flash("Please upload a photo.", "error")
            return redirect(url_for("search"))

        tmp_abs_path = save_uploaded_file(photo)
        uploaded_photo = "uploads/" + os.path.basename(tmp_abs_path)

        try:
            img = auto_orient_image(tmp_abs_path)

            if max(img.width, img.height) > 1600:
                scale = 1600 / max(img.width, img.height)
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )

            temp_jpg_path = tmp_abs_path + "_compressed.jpg"
            img.convert("RGB").save(temp_jpg_path, format="JPEG", quality=85, optimize=True)
            tmp_abs_path = temp_jpg_path

            image_np = np.array(img)
            face_locations = face_recognition.face_locations(image_np, model="hog")

            if not face_locations:
                log_best_match_search(os.path.basename(tmp_abs_path), [])
                return render_template(
                    "search.html",
                    results=[],
                    uploaded_photo=uploaded_photo,
                    searched=True,
                    face_locations=[],
                )

            idx, chosen_box = pick_largest_face(face_locations)
            encodings = face_recognition.face_encodings(image_np, [chosen_box])

            if encodings:
                candidates = find_person_by_face(encodings[0], tolerance=0.6, max_results=50) or []

                enriched = []
                seen_ids = set()
                for cand in candidates:
                    cand = dict(cand)
                    pid = cand.get("id")
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    if float(cand.get("match_confidence", 0)) >= MATCH_CONFIDENCE_THRESHOLD:
                        person = get_person_by_id(pid)
                        cand.update(serialize_person_obj(person))
                        cand = fix_photo_path(cand)
                        enriched.append(cand)

                results = sorted(
                    enriched,
                    key=lambda x: x.get("match_confidence", 0),
                    reverse=True,
                )[:MAX_RESULTS]

            # ---- LOG + ADMIN ALERT (POST ONLY) ----
            uploaded_name_for_log = os.path.basename(tmp_abs_path)
            log_best_match_search(uploaded_name_for_log, results)

            if results:
                emit_admin_alert(
                    "person_traced",
                    {
                        "count": len(results),
                        "top_match": results[0].get("full_name"),
                    },
                )

        except Exception:
            logger.exception("Error searching photo")
            flash("Error processing uploaded photo.", "error")
            return redirect(url_for("search"))

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(
                {
                    "success": True,
                    "results": results,
                    "searched": True,
                    "uploaded_photo": uploaded_photo,
                    "face_locations": [list(chosen_box)] if face_locations else [],
                }
            )

    return render_template(
        "search.html",
        results=results,
        uploaded_photo=uploaded_photo,
        searched=searched,
        face_locations=face_locations,
    )


# --- Admin helpers ----------------------------------------------------------
def _redirect_back_to_dashboard(default_page: int = 1):
    """Small helper to keep redirects to the same page if provided."""
    try:
        page = int(request.form.get("page", default_page))
    except Exception:
        page = default_page
    return redirect(url_for("admin_dashboard", page=page))


def _is_primary_super_admin(user_obj) -> bool:
    """
    Returns True if the given user_obj is the primary super admin,
    based on the ADMIN_EMAIL environment variable.
    """
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    if not super_admin_email or not user_obj:
        return False

    email = getattr(user_obj, "email", None)
    if not email:
        return False

    return email.strip().lower() == super_admin_email


def _current_actor() -> "User | None":
    """
    Safely load the DB User row for the current logged-in principal.
    This avoids relying on current_user having an email attribute.
    """
    user_id = getattr(current_user, "id", None)
    if user_id is None:
        return None
    return User.query.get(user_id)

@app.route(
    "/admin/persons/<int:person_id>/delete",
    methods=["POST"],
    endpoint="delete_person",
)
@login_required
@require_role("admin")
def delete_person(person_id):
    """
    Delete a single Person record from the admin dashboard.
    Used by the 'Delete' buttons on each row/card.
    """
    person = Person.query.get_or_404(person_id)

    try:
        db.session.delete(person)
        db.session.commit()
        flash(f"Person '{person.full_name}' has been deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting person: {e}", "error")

    # Preserve pagination if 'page' was posted
    return _redirect_back_to_dashboard()

@app.route("/admin/persons/delete-multiple", methods=["POST"])
@login_required
@require_role("admin")
def delete_multiple_persons():
    """
    Delete multiple Person records based on IDs sent from the admin dashboard.
    Expects a list of person IDs in form field 'selected_ids' (matches template JS).
    """
    raw_ids = request.form.getlist("selected_ids")

    if not raw_ids:
        flash("No persons selected for deletion.", "error")
        return _redirect_back_to_dashboard()

    ids = []
    for v in raw_ids:
        try:
            ids.append(int(v))
        except (TypeError, ValueError):
            continue

    if not ids:
        flash("No valid persons selected for deletion.", "error")
        return _redirect_back_to_dashboard()

    try:
        Person.query.filter(Person.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
        flash(f"Deleted {len(ids)} person(s).", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting persons: {e}", "error")

    return _redirect_back_to_dashboard()


@app.route("/admin/clear-search-logs", methods=["POST"])
@login_required
@require_role("admin")
def clear_search_logs():
    """
    Clear search logs used by the admin dashboard.
    Tries, in order:
      1) db_clear_search_logs() helper if it exists
      2) SearchLog model if it exists
    Fails gracefully if neither is present.
    """
    cleared = False
    error = None

    fn = globals().get("db_clear_search_logs")
    if callable(fn):
        try:
            fn()
            cleared = True
        except Exception as e:
            error = e

    if not cleared:
        SearchLog_model = globals().get("SearchLog")
        if SearchLog_model is not None:
            try:
                SearchLog_model.query.delete()
                db.session.commit()
                cleared = True
            except Exception as e:
                db.session.rollback()
                error = e

    if cleared:
        flash("Search logs cleared.", "success")
    else:
        msg = "Could not clear search logs."
        if DEV_MODE and error:
            msg += f" ({error})"
        flash(msg, "error")

    return _redirect_back_to_dashboard()


@app.route(
    "/admin/users/<int:user_id>/reset-password",
    methods=["POST"],
    endpoint="admin_reset_user_password",
)
@login_required
@require_role("admin")
def admin_reset_user_password(user_id):
    user = User.query.get_or_404(user_id)

    # Load the real actor row from DB
    actor = _current_actor()
    if actor is None:
        flash("Unable to determine the current admin account.", "error")
        return _redirect_back_to_dashboard()

    # Normalize emails
    actor_email = (getattr(actor, "email", "") or "").strip().lower()
    target_email = (user.email or "").strip().lower()
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

    # --- 1. SUPER ADMIN PROTECTION ---
    # Only the primary Super Admin can reset the primary Super Admin account
    if target_email == super_admin_email and actor_email != super_admin_email:
        flash("Only the Super Admin can reset the primary Super Admin password.", "error")
        return _redirect_back_to_dashboard()

    # --- 2. ADMIN CANNOT RESET ANOTHER ADMIN ---
    target_is_admin = (getattr(user, "role", "") or "").lower() == "admin"
    if target_is_admin and actor_email not in (target_email, super_admin_email):
        flash(
            "Only the Super Admin or this admin themselves can reset this admin password.",
            "error",
        )
        return _redirect_back_to_dashboard()

    # --- 3. Must have email ---
    if not user.email:
        flash("User has no email address on file; cannot send reset link.", "error")
        return _redirect_back_to_dashboard()

    # --- 4. Generate reset token ---
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + timedelta(minutes=30)

    if hasattr(user, "reset_token"):
        user.reset_token = token
    if hasattr(user, "reset_expiry"):
        user.reset_expiry = expiry

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Error preparing reset link: {e}", "error")
        return _redirect_back_to_dashboard()

    reset_link = build_external_url("reset_password", token=token)

    # --- 5. Email contents ---
    subject = "Your PersonFinder Password Reset Request"
    body = f"""Hello {user.username},

A password reset request was made for your PersonFinder account.

Click the link below to set a new password (valid for 30 minutes):
{reset_link}

If you did not request this change, please contact the system administrator immediately.
"""
    html = f"""
    <div style="font-family: Arial, sans-serif; color:#333;">
      <h2 style="margin-bottom: 10px;">PersonFinder Password Reset</h2>
      <p>Hello <strong>{user.username}</strong>,</p>
      <p>A password reset was requested for your PersonFinder account.</p>

      <p>
        <a href="{reset_link}"
           style="background:#2563eb;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none;">
           Reset My Password
        </a>
      </p>

      <p style="margin-top:20px;font-size:13px;color:#666;">
        This link is valid for <strong>30 minutes</strong>.
      </p>

      <hr style="border:none;border-top:1px solid #ddd;margin:20px 0;">

      <p style="font-size:12px;color:#777;">
        Security notice: If you did not request this password reset, please ignore this email
        and contact your administrator or the PersonFinder support team immediately.
      </p>

      <p style="font-size:12px;color:#999;">— PersonFinder Security Team</p>
    </div>
    """

    ok, err = send_mail(subject, [user.email], body=body, html=html)
    if not ok:
        flash(f"Reset token created, but email failed: {err}", "error")
    else:
        flash("Password reset link sent successfully.", "success")

    return _redirect_back_to_dashboard()


@app.route(
    "/admin/toggle-user-role/<int:user_id>",
    methods=["POST"],
    endpoint="admin_toggle_user_role",
)
@login_required
@require_role("admin")
def admin_toggle_user_role(user_id):
    user = User.query.get_or_404(user_id)
    actor = _current_actor()

    actor_is_super_admin = _is_primary_super_admin(actor)
    target_is_super_admin = _is_primary_super_admin(user)
    target_is_admin = (getattr(user, "role", "") or "").lower() == "admin"

    # Nobody can change the primary super admin role
    if target_is_super_admin:
        flash("You cannot change the role of the primary super admin.", "error")
        return _redirect_back_to_dashboard()

    # Only super admin can change roles for admin accounts
    if target_is_admin and not actor_is_super_admin:
        flash("Only the Super Admin can change roles for admin accounts.", "error")
        return _redirect_back_to_dashboard()

    # Toggle role normally (user <-> admin)
    if user.role == "admin":
        user.role = "user"
    else:
        user.role = "admin"

    try:
        db.session.commit()
        flash(f"User role updated to {user.role}.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating role: {e}", "error")

    return _redirect_back_to_dashboard()


@app.route(
    "/admin/users/<int:user_id>/toggle-active",
    methods=["POST"],
    endpoint="admin_toggle_user_active",
)
@login_required
@require_role("admin")
def admin_toggle_user_active(user_id):
    user = User.query.get_or_404(user_id)
    actor = _current_actor()

    actor_is_super_admin = _is_primary_super_admin(actor)
    target_is_super_admin = _is_primary_super_admin(user)
    target_is_admin = (getattr(user, "role", "") or "").lower() == "admin"

    if not hasattr(user, "is_active"):
        flash("This installation does not yet support enabling/disabling users.", "error")
        return _redirect_back_to_dashboard()

    # Prevent disabling yourself
    if user.id == current_user.id:
        flash("You cannot disable your own account.", "error")
        return _redirect_back_to_dashboard()

    # Nobody can disable the primary super admin account
    if target_is_super_admin:
        flash("You cannot disable the primary super admin account.", "error")
        return _redirect_back_to_dashboard()

    # Only super admin can activate/deactivate admin accounts
    if target_is_admin and not actor_is_super_admin:
        flash("Only the Super Admin can activate/deactivate admin accounts.", "error")
        return _redirect_back_to_dashboard()

    user.is_active = not bool(user.is_active)

    try:
        db.session.commit()
        state = "activated" if user.is_active else "deactivated"
        flash(f"User {user.username} has been {state}.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating active status: {e}", "error")

    return _redirect_back_to_dashboard()


@app.route(
    "/admin/users/<int:user_id>/delete",
    methods=["POST"],
    endpoint="admin_delete_user",
)
@login_required
@require_role("admin")
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    actor = _current_actor()

    actor_is_super_admin = _is_primary_super_admin(actor)
    target_is_super_admin = _is_primary_super_admin(user)
    target_is_admin = (getattr(user, "role", "") or "").lower() == "admin"

    # Prevent deleting yourself
    if user.id == current_user.id:
        flash("You cannot delete your own account.", "error")
        return _redirect_back_to_dashboard()

    # Nobody can delete the primary super admin
    if target_is_super_admin:
        flash("You cannot delete the primary super admin account.", "error")
        return _redirect_back_to_dashboard()

    # Only super admin can delete admin accounts
    if target_is_admin and not actor_is_super_admin:
        flash("Only the Super Admin can delete admin accounts.", "error")
        return _redirect_back_to_dashboard()

    try:
        db.session.delete(user)
        db.session.commit()
        flash(f"User {user.username} has been deleted.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {e}", "error")

    return _redirect_back_to_dashboard()


# --- Static Pages ---
@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy-policy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/about")
def about():
    return render_template("about.html", title="About App - PersonFinder")


@app.route("/developers")
def developers():
    return render_template("developers.html", current_year=datetime.now().year)


# ✅ Single, canonical donate route
@app.route("/donate", endpoint="donate_page")
def donate_page():
    return render_template(
        "donate.html",
        donation_account_holder=os.getenv("DONATION_ACCOUNT_HOLDER"),
        donation_account_number=os.getenv("DONATION_ACCOUNT_NUMBER"),
        donation_ifsc=os.getenv("DONATION_IFSC"),
        donation_bank_name=os.getenv("DONATION_BANK_NAME"),
        donation_upi_id=os.getenv("DONATION_UPI_ID"),
    )


@app.route("/save-subscription", methods=["POST"])
def save_subscription():
    subscription_info = request.get_json()
    person_id = request.args.get("person_id")
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
    async_mode="threading",  # Windows-safe
    logger=False,
    engineio_logger=os.getenv("ENGINEIO_LOG", "0") in ("1", "true", "yes", "on"),
    ping_interval=25,
    ping_timeout=60,
)

# ----------------- SocketIO Auth Guard -----------------
@socketio.on("connect")
def socket_connect():
    """
    Allow socket connection ONLY for authenticated admins.
    Prevents public / anonymous socket usage.
    """
    try:
        if not current_user.is_authenticated:
            return False

        # Only admin gets live alerts
        role = (getattr(current_user, "role", "") or "").lower()
        if role != "admin":
            return False

    except Exception:
        return False

@socketio.on("request_admin_unread_count")
def handle_request_unread():
    # compute unread count for admin session (example: query DB / cache)
    count = 0
    try:
        # compute actual count...
        socketio.emit("admin_unread_count", {"count": count}, to=request.sid)
    except Exception:
        pass


# ----------------- Admin Alert Emitter -----------------
from typing import Optional, Dict

def emit_admin_alert(event: str, payload: Optional[Dict] = None):
    try:
        # increment unread count
        session["admin_unread_alerts"] = session.get("admin_unread_alerts", 0) + 1

        socketio.emit(
            "admin_alert",
            {
                "event": event,
                "payload": payload or {},
                "ts": int(time.time()),
            },
            namespace="/",
        )
    except Exception:
        logger.exception("Failed to emit admin alert")


# ------------------------------------------------------------
# RUN SERVER (PRODUCTION SAFE)
# ------------------------------------------------------------
if __name__ == "__main__":
    host_ip = get_local_ip()
    port = int(os.getenv("PORT", 5001))

    print("\n🚀 PersonFinder Server Started")
    print(f"   ENV     : {'PRODUCTION' if IS_PROD else 'DEVELOPMENT'}")
    print(f"   Desktop : http://127.0.0.1:{port}")
    print(f"   Mobile  : http://{host_ip}:{port}\n")

    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
# End of app.py