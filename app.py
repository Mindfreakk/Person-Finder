# app.py
import os, io, uuid, base64, logging, json, random, secrets
from math import ceil
from datetime import datetime, timedelta
from functools import wraps
from PIL import Image
from dotenv import load_dotenv
import face_recognition
import qrcode
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from pywebpush import webpush, WebPushException
from flask_mail import Mail, Message

# ----------------- Import DB functions / ORM -----------------
from database import (
    db,
    Person, SearchLog, PushSubscription, User,
    register_person_to_db, get_all_registered_people,
    get_person_by_id, delete_person_by_id,
    find_person_by_face, log_best_match_search, get_stats,
    authenticate_user
)

# ----------------- Env / Config -----------------
load_dotenv()
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "<YOUR_PRIVATE_KEY>")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "<YOUR_PUBLIC_KEY>")
VAPID_CLAIMS = {"sub": "mailto:you@example.com"}

# ----------------- App Config -----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or os.urandom(24)
UPLOAD_DIR = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", 'sqlite:///personfinder.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='Ammehz09@gmail.com',
    MAIL_PASSWORD=os.getenv("EMAIL_APP_PASSWORD")
)
mail = Mail(app)

# ----------------- Initialize DB -----------------
db.init_app(app)
logging.basicConfig(level=logging.INFO)

# ----------------- Flask-Login -----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "error"

# ----------------- User Wrapper -----------------
class UserLogin(UserMixin):
    def __init__(self, user: User):
        self.id = str(user.id)
        self.username = user.username
        self.role = getattr(user, "role", "user")

    @property
    def is_active(self):
        return True

@login_manager.user_loader
def load_user(user_id):
    user = User.query.get(int(user_id))
    return UserLogin(user) if user else None

# ----------------- Ensure Admin Exists -----------------
def ensure_admin_exists():
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "Alhamdulillah@123"
    admin = User.query.filter_by(username=ADMIN_USERNAME).first()
    if not admin:
        admin = User(username=ADMIN_USERNAME, role="admin")
        admin.password_hash = generate_password_hash(ADMIN_PASSWORD)
        db.session.add(admin)
        db.session.commit()
        logging.info(f"✅ Admin created: {ADMIN_USERNAME}/{ADMIN_PASSWORD}")
    else:
        logging.info(f"ℹ️ Admin already exists: {ADMIN_USERNAME}")

# ----------------- Role Decorator -----------------
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
def fix_photo_path(person: dict) -> dict:
    placeholder = "images/no-photo.png"
    raw = person.get("photo_path") or ""
    filename = os.path.basename(raw) if raw else ""
    person["photo_path"] = f"uploads/{filename}" if filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)) else placeholder
    return person

def save_uploaded_file(storage_file) -> str:
    original = secure_filename(storage_file.filename or "upload")
    ext = os.path.splitext(original)[1] or ".png"
    unique = f"{uuid.uuid4().hex}{ext}"
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    storage_file.save(abs_path)
    return abs_path

def save_base64_image(data_url: str) -> str:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    img = Image.open(img_bytes)
    unique = f"{uuid.uuid4().hex}.png"
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    img.save(abs_path)
    return abs_path

# ----------------- Push Notifications -----------------
def save_subscription_to_db(person_id, subscription: dict):
    existing = PushSubscription.query.filter_by(person_id=person_id, endpoint=subscription["endpoint"]).first()
    if not existing:
        new_sub = PushSubscription(
            person_id=person_id,
            endpoint=subscription["endpoint"],
            p256dh=subscription["keys"]["p256dh"],
            auth=subscription["keys"]["auth"]
        )
        db.session.add(new_sub)
        db.session.commit()

def send_push_notification(person_id, title, message, url="/"):
    subs = PushSubscription.query.filter_by(person_id=person_id).all()
    for sub in subs:
        try:
            webpush(
                subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                data=json.dumps({"title": title, "body": message, "url": url}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
        except WebPushException as ex:
            logging.error(f"Push failed for {sub.id}: {ex}")

# ----------------- OTP Utilities -----------------
OTP_EXPIRY_MINUTES = 5
RESEND_COOLDOWN_SECONDS = 30

def _now(): return datetime.utcnow()
def _generate_otp(): return f"{random.randint(100000, 999999)}"
def _get_otp_dict(key):
    if key not in session: session[key] = {}
    return session[key]

def store_email_otp(email, otp):
    d = _get_otp_dict("email_otps")
    d[email] = {"otp": otp, "expiry": (_now()+timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat(), "last_sent": _now().isoformat()}
    session.modified = True

def store_phone_otp(phone, otp):
    d = _get_otp_dict("phone_otps")
    d[phone] = {"otp": otp, "expiry": (_now()+timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat(), "last_sent": _now().isoformat()}
    session.modified = True

def verify_otp(kind, key, otp_input):
    d = _get_otp_dict(f"{kind}_otps")
    record = d.get(key)
    if not record: return False, "OTP not found"
    if datetime.fromisoformat(record["expiry"]) <= _now(): d.pop(key, None); return False, "OTP expired"
    if otp_input != record["otp"]: d.pop(key, None); return False, "Invalid OTP"
    d.pop(key, None); session.modified = True
    return True, None

# ----------------- Routes -----------------

@app.route("/")
def home():
    stats = get_stats()
    return render_template("home.html", registrations_count=stats.get("registrations", 0), searches_count=stats.get("searches", 0))

# --- Admin Dashboard ---
@app.route("/admin/dashboard")
@login_required
@require_role("admin")
def admin_dashboard():
    page = int(request.args.get("page", 1))
    per_page = 10
    people_query = Person.query.order_by(Person.id.desc())
    total_people = people_query.count()
    people = people_query.offset((page-1)*per_page).limit(per_page).all()
    recent_threshold = datetime.utcnow() - timedelta(hours=24)
    for p in people:
        p.is_recent = getattr(p, "created_at", None) and p.created_at > recent_threshold
    total_pages = ceil(total_people / per_page)
    stats = get_stats()
    users = User.query.all()
    search_logs = SearchLog.query.order_by(SearchLog.ts.desc()).limit(10).all()
    return render_template("admin_dashboard.html", stats=stats, users=users, people=people, search_logs=search_logs, page=page, total_pages=total_pages)

# --- Login / Logout ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        admin_user = User.query.filter_by(username="admin").first()
        if not admin_user:
            flash("Admin account not found.", "error")
            return redirect(url_for("login"))
        if username.lower() != "admin":
            flash("Only admin can log in here.", "error")
            return redirect(url_for("login"))
        if check_password_hash(admin_user.password_hash, password):
            login_user(UserLogin(admin_user))
            flash("Admin logged in successfully!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid password for admin.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "success")
    return redirect(url_for("login"))

from datetime import datetime, timedelta
import secrets
from werkzeug.security import generate_password_hash

# --- Forgot Password ---
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        ADMIN_EMAIL = "ammehz09@gmail.com"
        user = User.query.filter_by(username="admin").first()

        if not user or email != ADMIN_EMAIL.lower():
            flash("No admin account found with this email.", "error")
            return redirect(url_for("forgot_password"))

        # Generate a token and store in DB
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_expiry = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()

        reset_link = url_for("reset_password", token=token, _external=True)

        # Send email
        msg = Message(
            "PersonFinder Admin Password Reset",
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )
        msg.body = f"Click the link below to reset your admin password (valid 15 min):\n\n{reset_link}"
        mail.send(msg)

        flash("Password reset link sent to your email.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")


# --- Reset Password ---
@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or datetime.utcnow() > user.reset_expiry:
        flash("Invalid or expired reset link.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("password", "").strip()
        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(request.url)

        user.password_hash = generate_password_hash(new_password)
        user.reset_token = None
        user.reset_expiry = None
        db.session.commit()

        flash("Password updated successfully! You can now login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)



# --- Register a Person ---
@app.route("/register", methods=["GET","POST"])
def register():
    filename_for_preview = None
    if request.method=="POST":
        try:
            full_name = request.form.get("full_name","").strip()
            age = request.form.get("age","").strip()
            gender = request.form.get("gender","").strip()
            guardian_name = request.form.get("guardian_name","").strip()
            phone_number = request.form.get("phone_number","").strip()
            address = request.form.get("address","").strip()
            last_seen = request.form.get("last_seen","").strip()
            photo_file = request.files.get("photo_file")
            photo_base64 = request.form.get("photo_input")
            photo_abs_path = None
            if photo_file and photo_file.filename:
                photo_abs_path = save_uploaded_file(photo_file)
            elif photo_base64:
                photo_abs_path = save_base64_image(photo_base64)
            else:
                flash("Photo is required", "error")
                return render_template("register.html", filename=None)
            filename_for_preview = "uploads/" + os.path.basename(photo_abs_path)
            image = face_recognition.load_image_file(photo_abs_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                flash("No face detected.", "error")
                return render_template("register.html", filename=None)
            face_encoding = encodings[0]
            person = {
                "full_name": full_name, "age": age, "gender": gender,
                "guardian_name": guardian_name, "phone_number": phone_number,
                "address": address, "last_seen": last_seen,
                "photo_path": os.path.basename(photo_abs_path),
                "face_encoding": face_encoding
            }
            register_person_to_db(person, user_id=None)
            flash("Person registered successfully!", "success")
            return redirect(url_for('home'))
        except Exception as e:
            logging.exception("Error during registration")
            flash(f"Error: {e}", "error")
            return render_template("register.html", filename=filename_for_preview)
    return render_template("register.html", filename=filename_for_preview)

# --- Search ---
@app.route("/search", methods=["GET","POST"])
def search():
    results, uploaded_photo_url, searched = [], None, False
    if request.method=="POST":
        searched = True
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            flash("Please upload a photo.", "error")
            return redirect(request.url)
        tmp_abs_path = save_uploaded_file(photo)
        uploaded_photo_url = "uploads/" + os.path.basename(tmp_abs_path)
        try:
            image = face_recognition.load_image_file(tmp_abs_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                flash("No face detected.", "error")
            else:
                search_encoding = encodings[0]
                results = find_person_by_face(search_encoding, tolerance=0.6) or []
                results = [fix_photo_path(p.copy()) for p in results]
                log_best_match_search(uploaded_name=photo.filename, matches=results)
        except Exception as e:
            logging.exception("Error processing uploaded photo")
            flash("Error processing uploaded photo.", "error")
    return render_template("search.html", results=results, uploaded_photo=uploaded_photo_url, searched=searched)

# --- Delete Person (Admin only) ---
@app.route("/delete/<int:person_id>", methods=["POST"])
@login_required
@require_role("admin")
def delete(person_id):
    try:
        ok = delete_person_by_id(person_id, current_user=current_user)
        if ok:
            person = get_person_by_id(person_id)
            photo_name = os.path.basename((person or {}).get("photo_path") or "")
            if photo_name:
                try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], photo_name))
                except: pass
            flash("Person deleted successfully.", "success")
        else:
            flash("You don’t have permission.", "error")
    except Exception as e:
        logging.exception("Error deleting person")
        flash(f"Error: {e}", "error")
    return redirect(url_for("search"))

# --- Static Pages ---
@app.route("/privacy-policy")
def privacy_policy(): return render_template("privacy-policy.html")
@app.route("/terms")
def terms(): return render_template("terms.html")
@app.route("/donate")
def donate_page(): return render_template("donate.html")
@app.route("/about")
def about(): return render_template("about.html", title="About App - PersonFinder")
@app.route("/developers")
def developers(): return render_template("developers.html", current_year=datetime.now().year)

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

def migrate_database():
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)

    with db.engine.begin() as conn:  # use begin() to auto-commit
        if 'users' in inspector.get_table_names():
            columns = [c['name'] for c in inspector.get_columns('users')]

            if 'email' not in columns:
                conn.execute(text('ALTER TABLE users ADD COLUMN email VARCHAR;'))

            if 'phone' not in columns:
                conn.execute(text('ALTER TABLE users ADD COLUMN phone VARCHAR;'))

            if 'reset_token' not in columns:
                conn.execute(text('ALTER TABLE users ADD COLUMN reset_token VARCHAR;'))

            if 'reset_expiry' not in columns:
                conn.execute(text('ALTER TABLE users ADD COLUMN reset_expiry TIMESTAMP;'))


# ----------------- Run -----------------
with app.app_context():
    db.create_all()
    migrate_database()
    ensure_admin_exists()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)