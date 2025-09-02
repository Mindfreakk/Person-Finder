import os
import base64
import logging
import numpy as np
import face_recognition
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from pywebpush import webpush, WebPushException
from werkzeug.security import generate_password_hash, check_password_hash
import json

logging.basicConfig(level=logging.INFO)

# ----------------- SQLAlchemy Setup -----------------
db = SQLAlchemy()

# ----------------- Models -----------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), default="admin")
    
    # New fields for reset
    reset_token = db.Column(db.String(128), nullable=True)
    reset_expiry = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


    # --- Reset token helpers ---
    def set_reset_token(self, token: str, expiry_minutes: int = 15):
        self.reset_token = token
        self.reset_expiry = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        db.session.commit()

    def clear_reset_token(self):
        self.reset_token = None
        self.reset_expiry = None
        db.session.commit()

    def is_reset_token_valid(self, token: str) -> bool:
        if not self.reset_token or self.reset_token != token:
            return False
        if not self.reset_expiry or datetime.utcnow() > self.reset_expiry:
            return False
        return True


class Person(db.Model):
    __tablename__ = 'people'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String)
    guardian_name = db.Column(db.String)
    phone_number = db.Column(db.String, nullable=False)
    address = db.Column(db.String)
    last_seen = db.Column(db.String)
    photo_path = db.Column(db.String)
    face_encoding = db.Column(db.Text)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    search_logs = db.relationship('SearchLog', backref='person', lazy=True, cascade="all, delete-orphan")
    subscriptions = db.relationship('PushSubscription', backref='person', lazy=True, cascade="all, delete-orphan")


class SearchLog(db.Model):
    __tablename__ = 'search_logs'
    id = db.Column(db.Integer, primary_key=True)
    ts = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_name = db.Column(db.String)
    success = db.Column(db.Integer, default=0)
    matches = db.Column(db.Integer, default=0)
    person_id = db.Column(db.Integer, db.ForeignKey('people.id'))


class PushSubscription(db.Model):
    __tablename__ = 'push_subscriptions'
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('people.id'), nullable=False)
    endpoint = db.Column(db.String, nullable=False)
    p256dh = db.Column(db.String, nullable=False)
    auth = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ----------------- Helpers -----------------
def _normalize_photo_rel_path(photo_path: str) -> str:
    if not photo_path:
        return None
    filename = os.path.basename(photo_path).replace("\\", "/")
    return f"uploads/{filename}"


# ----------------- Admin Helper -----------------
def ensure_admin_exists():
    """Create default admin if not present"""
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin")
        admin.set_password("Alhamdulillah@123")
        db.session.add(admin)
        db.session.commit()
        logging.info("Default admin created")


def authenticate_user(username: str, password: str) -> User:
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    return None


# ----------------- Person CRUD -----------------
def register_person_to_db(person: dict, user_id: int = None):
    if not person.get("face_encoding"):
        raise ValueError("Face encoding required")

    face_array = np.array(person["face_encoding"], dtype=np.float64).reshape((128,))
    face_encoding_b64 = base64.b64encode(face_array.tobytes()).decode()
    rel_photo_path = _normalize_photo_rel_path(person.get("photo_path"))

    age_val = int(person["age"]) if str(person.get("age") or "").isdigit() else None

    new_person = Person(
        full_name=person.get("full_name"),
        age=age_val,
        gender=person.get("gender"),
        guardian_name=person.get("guardian_name"),
        phone_number=person.get("phone_number"),
        address=person.get("address"),
        last_seen=person.get("last_seen"),
        photo_path=rel_photo_path,
        face_encoding=face_encoding_b64,
        created_by=user_id
    )
    db.session.add(new_person)
    db.session.commit()
    logging.info(f"Person registered: {person.get('full_name')}")
    return new_person


# ----------------- Face Search / Stats / Push -----------------
def get_all_registered_people(include_face_encoding=False) -> list:
    people = Person.query.order_by(Person.id.asc()).all()
    result = []
    for p in people:
        person_dict = {
            "id": p.id,
            "full_name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "guardian_name": p.guardian_name,
            "phone_number": p.phone_number,
            "address": p.address,
            "last_seen": p.last_seen,
            "photo_path": p.photo_path,
            "created_by": p.created_by
        }
        if include_face_encoding:
            person_dict["face_encoding"] = p.face_encoding
        result.append(person_dict)
    return result


def get_person_by_id(person_id: int) -> dict:
    p = Person.query.get(person_id)
    if not p:
        return None
    return {
        "id": p.id,
        "full_name": p.full_name,
        "age": p.age,
        "gender": p.gender,
        "guardian_name": p.guardian_name,
        "phone_number": p.phone_number,
        "address": p.address,
        "last_seen": p.last_seen,
        "photo_path": p.photo_path,
        "face_encoding": p.face_encoding,
        "created_by": p.created_by
    }


def delete_person_by_id(person_id: int, current_user: User) -> bool:
    p = Person.query.get(person_id)
    if not p:
        return False
    if current_user.role == "admin":
        for sub in p.subscriptions:
            db.session.delete(sub)
        db.session.delete(p)
        db.session.commit()
        logging.info(f"Person deleted: {p.id}")
        return True
    return False


def find_person_by_face(search_encoding: np.ndarray, tolerance=0.6) -> list:
    search_encoding = np.array(search_encoding, dtype=np.float64).reshape((128,))
    people = get_all_registered_people(include_face_encoding=True)
    matches = []

    for person in people:
        encoded_str = person.get("face_encoding")
        if not encoded_str:
            continue
        person_encoding = np.frombuffer(base64.b64decode(encoded_str), dtype=np.float64).reshape((128,))
        match = face_recognition.compare_faces([person_encoding], search_encoding, tolerance=tolerance)[0]
        distance = float(face_recognition.face_distance([person_encoding], search_encoding)[0])
        if match:
            confidence = int(max(0, min(1, 1.0 - distance)) * 100)
            person_copy = person.copy()
            person_copy["match_confidence"] = confidence
            matches.append(person_copy)

    matches.sort(key=lambda x: x.get("match_confidence", 0), reverse=True)
    return matches


def log_search(uploaded_name: str = None, success: int = 0, matches: int = 0, person_id: int = None):
    db.session.add(SearchLog(
        uploaded_name=uploaded_name,
        success=int(bool(success)),
        matches=int(matches),
        person_id=person_id
    ))
    db.session.commit()


def get_stats() -> dict:
    return {"registrations": Person.query.count(), "searches": SearchLog.query.count()}


VAPID_PRIVATE_KEY = "<YOUR_PRIVATE_VAPID_KEY>"
VAPID_CLAIMS = {"sub": "mailto:you@example.com"}


def send_push_notification(person_id, title, message):
    subscriptions = PushSubscription.query.filter_by(person_id=person_id).all()
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                data=json.dumps({"title": title, "message": message}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
        except WebPushException as ex:
            logging.error(f"Push failed for subscription {sub.id}: {ex}")


def log_best_match_search(uploaded_name: str, matches: list):
    if not matches:
        log_search(uploaded_name=uploaded_name, success=0, matches=0)
        return
    best_match = matches[0]
    log_search(
        uploaded_name=uploaded_name,
        success=1,
        matches=len(matches),
        person_id=best_match["id"]
    )
    send_push_notification(
        best_match["id"],
        title="Person Found!",
        message=f"Your registered person '{best_match['full_name']}' has been matched."
    )
