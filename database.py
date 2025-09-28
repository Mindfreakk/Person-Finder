# database.py (fixed & hardened)
import os
import json
import base64
import ast
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import face_recognition
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache
from sqlalchemy.exc import OperationalError
from sqlalchemy import inspect, text

# Try to import pywebpush but degrade gracefully if not installed
try:
    from pywebpush import webpush, WebPushException
    _HAS_PYWEBPUSH = True
except Exception:
    webpush = None
    WebPushException = Exception
    _HAS_PYWEBPUSH = False

# ---------------------------------------------------
# Logging & SQLAlchemy setup
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single shared DB instance
db = SQLAlchemy()

# ---------------------------------------------------
# Models
# ---------------------------------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(200))
    # single role column; default 'user' (can be 'admin')
    role = db.Column(db.String(50), default="user")
    phone_number = db.Column(db.String(50), nullable=True)
    reset_token = db.Column(db.String(128), nullable=True)
    reset_expiry = db.Column(db.DateTime, nullable=True)

    # --- Password helpers ---
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
    __tablename__ = "people"

    id = db.Column(db.Integer, primary_key=True)

    # Primary fields
    full_name = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String)
    guardian_name = db.Column(db.String)
    phone_number = db.Column(db.String, nullable=False)
    address = db.Column(db.String)
    last_seen = db.Column(db.String)
    photo_path = db.Column(db.String)

    # Face encoding stored as base64 float64[128]
    face_encoding = db.Column(db.Text)

    # Who created the record (FK to users.id)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # “Registered by” fields for non-logged registrations
    registered_by_name = db.Column(db.String)
    registered_by_phone = db.Column(db.String)
    registered_by_relation = db.Column(db.String)

    # Relationships
    search_logs = db.relationship(
        "SearchLog", backref="person", lazy=True, cascade="all, delete-orphan"
    )
    subscriptions = db.relationship(
        "PushSubscription", backref="person", lazy=True, cascade="all, delete-orphan"
    )


class SearchLog(db.Model):
    __tablename__ = "search_logs"

    id = db.Column(db.Integer, primary_key=True)
    ts = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_name = db.Column(db.String)
    success = db.Column(db.Integer, default=0)  # 0/1
    matches = db.Column(db.Integer, default=0)
    person_id = db.Column(db.Integer, db.ForeignKey("people.id"))


class PushSubscription(db.Model):
    __tablename__ = "push_subscriptions"

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey("people.id"), nullable=False)
    endpoint = db.Column(db.String, nullable=False)
    p256dh = db.Column(db.String, nullable=False)
    auth = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------
# Small helpers
# ---------------------------------------------------
def _normalize_photo_rel_path(photo_path: Optional[str]) -> Optional[str]:
    """Normalize stored photo path to 'uploads/<filename>' or None."""
    if not photo_path:
        return None
    filename = os.path.basename(photo_path).replace("\\", "/")
    return f"uploads/{filename}"


def _ensure_b64_face_encoding(val) -> Optional[str]:
    """
    Accepts a list/np.ndarray/bytes/base64 str and returns a base64 string
    representing float64[128]. Returns None if val is falsy.
    """
    if val is None:
        return None

    # If it's already a base64 string representing float64 bytes, validate & return
    if isinstance(val, str):
        try:
            arr = np.frombuffer(base64.b64decode(val), dtype=np.float64)
            if arr.size == 128:
                return val
        except Exception:
            pass  # fall through to other conversions

    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val, dtype=np.float64).reshape((128,))
        return base64.b64encode(arr.tobytes()).decode("utf-8")

    if isinstance(val, (bytes, bytearray)):
        # Assume it is raw float64 bytes or base64 bytes
        try:
            arr = np.frombuffer(val, dtype=np.float64).reshape((128,))
            return base64.b64encode(arr.tobytes()).decode("utf-8")
        except Exception:
            try:
                arr = np.frombuffer(base64.b64decode(val), dtype=np.float64).reshape((128,))
                return base64.b64encode(arr.tobytes()).decode("utf-8")
            except Exception:
                pass

    # fallback: if val is a JSON string containing list
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (list, tuple)):
                arr = np.array(parsed, dtype=np.float64).reshape((128,))
                return base64.b64encode(arr.tobytes()).decode("utf-8")
        except Exception:
            pass

    raise ValueError("Invalid face_encoding; expected 128-length vector or base64 float64 representation.")


def _decode_face_encoding_to_vec(b64_text: str) -> np.ndarray:
    """Decode base64 float64[128] -> np.ndarray(128,)."""
    arr = np.frombuffer(base64.b64decode(b64_text), dtype=np.float64)
    if arr.size != 128:
        raise ValueError("Stored face encoding has invalid length.")
    return arr.reshape((128,))


# ---------------------------------------------------
# Person operations
# ---------------------------------------------------
def register_person_to_db(person_data: dict) -> Person:
    """
    Register a new person in the database.
    Expects person_data to contain keys used by Person columns.
    Handles face_encoding normalization and photo path normalization.
    Performs minimal validation on required fields.
    """
    try:
        data = dict(person_data or {})

        # validate minimum required fields
        if not data.get("phone_number"):
            raise ValueError("phone_number is required")
        if not data.get("full_name"):
            raise ValueError("full_name is required")

        # Normalize photo_path to uploads/<filename>
        data["photo_path"] = _normalize_photo_rel_path(data.get("photo_path"))

        # Ensure face_encoding is stored in base64 format
        if "face_encoding" in data and data.get("face_encoding") is not None:
            data["face_encoding"] = _ensure_b64_face_encoding(data.get("face_encoding"))

        person = Person(
            full_name=data.get("full_name"),
            age=data.get("age"),
            gender=data.get("gender"),
            guardian_name=data.get("guardian_name"),
            phone_number=data.get("phone_number"),
            address=data.get("address"),
            last_seen=data.get("last_seen"),
            photo_path=data.get("photo_path"),
            face_encoding=data.get("face_encoding"),
            created_by=data.get("created_by"),
            registered_by_name=data.get("registered_by_name"),
            registered_by_phone=data.get("registered_by_phone"),
            registered_by_relation=data.get("registered_by_relation"),
        )

        db.session.add(person)
        db.session.commit()
        logger.info(f"✅ Registered person #{person.id}: {person.full_name}")
        # After successful registration you may want to clear encodings cache (if cached)
        try:
            clear_people_encodings_cache()
        except Exception:
            pass
        return person

    except Exception as e:
        db.session.rollback()
        logger.exception("Error registering person: %s", e)
        # raise a clearer error for caller
        raise RuntimeError(f"Database error while registering person: {e}")


def get_person_by_id(person_id: int) -> Optional[Dict[str, Any]]:
    """Return a dictionary of person info by ID."""
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
        "created_by": p.created_by,
        "registered_by_name": p.registered_by_name,
        "registered_by_phone": p.registered_by_phone,
        "registered_by_relation": p.registered_by_relation,
    }


def delete_person_by_id(person_id: int, current_user: User = None) -> bool:
    """
    Delete a person if current user is authorized:
      - admin OR
      - family who originally created the record.
    If current_user is None, deletion will only proceed if no restrictions apply (admin expected).
    """
    p = Person.query.get(person_id)
    if not p:
        return False

    allowed = False
    if current_user is None:
        # without a user object, require admin check outside (more secure)
        allowed = True
    else:
        if getattr(current_user, "role", None) == "admin":
            allowed = True
        elif getattr(current_user, "role", None) == "family" and p.created_by == getattr(current_user, "id", None):
            allowed = True

    if not allowed:
        return False

    try:
        db.session.delete(p)
        db.session.commit()
        logger.info(f"🗑️ Deleted person #{p.id}")
        # clear encodings cache to avoid stale results
        try:
            clear_people_encodings_cache()
        except Exception:
            pass
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting person {person_id}: {e}")
        return False


def get_all_registered_people(include_face_encoding: bool = False) -> List[Dict[str, Any]]:
    """Return all people as dicts, ordered by id ascending."""
    people = Person.query.order_by(Person.id.asc()).all()
    result = []
    for p in people:
        d = {
            "id": p.id,
            "full_name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "guardian_name": p.guardian_name,
            "phone_number": p.phone_number,
            "address": p.address,
            "last_seen": p.last_seen,
            "photo_path": p.photo_path,
            "created_by": p.created_by,
            "registered_by_name": p.registered_by_name,
            "registered_by_phone": p.registered_by_phone,
            "registered_by_relation": p.registered_by_relation,
        }
        if include_face_encoding:
            d["face_encoding"] = p.face_encoding
        result.append(d)
    return result


# ---------------------------------------------------
# Face matching: caching & utilities
# ---------------------------------------------------
@lru_cache(maxsize=1)
def _load_people_encodings_cached() -> List[Dict[str, Any]]:
    """
    Load people and decode face encodings into numpy arrays.
    Returns list of dicts with keys: id, full_name, photo_path, encoding(np.ndarray), ... (no SQLAlchemy objects)
    Cached for speed; call clear_people_encodings_cache() after updates.
    """
    out = []
    people = get_all_registered_people(include_face_encoding=True)
    for p in people:
        enc_b64 = p.get("face_encoding")
        if not enc_b64:
            continue
        try:
            enc_vec = _decode_face_encoding_to_vec(enc_b64)
            out.append({
                "id": p.get("id"),
                "full_name": p.get("full_name"),
                "photo_path": p.get("photo_path"),
                "encoding": enc_vec
            })
        except Exception:
            logger.debug("Skipping person id=%s due to invalid encoding", p.get("id"))
            continue
    logger.info("Loaded %d decoded encodings from DB into cache", len(out))
    return out


def clear_people_encodings_cache():
    """Clear the cached decoded encodings (call after registration/delete)."""
    try:
        _load_people_encodings_cached.cache_clear()
    except Exception:
        pass


def face_distance_to_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Map face distance -> confidence percentage (0..100).
    Heuristic: distances <= threshold map to ~60..100, distances above threshold map down to 0.
    """
    try:
        d = float(distance)
    except Exception:
        return 0.0
    if d <= 0:
        return 100.0
    if d <= threshold:
        # Map linearly 0 -> 100, threshold -> 60
        return round(100.0 - (d / threshold) * 40.0, 2)
    # Map threshold..(2*threshold) -> 60..0 (aggressive falloff)
    val = max(0.0, 60.0 - ((d - threshold) / threshold) * 60.0)
    return round(val, 2)


def _normalize_search_encoding(enc) -> np.ndarray:
    """
    Accept search encoding as np.ndarray/list/tuple/base64 string and return np.ndarray(128,).
    Raises ValueError if invalid.
    """
    if enc is None:
        raise ValueError("search encoding is None")

    if isinstance(enc, np.ndarray):
        arr = np.array(enc, dtype=np.float64)
    elif isinstance(enc, (list, tuple)):
        arr = np.array(enc, dtype=np.float64)
    elif isinstance(enc, str):
        # try base64 float64 bytes first
        try:
            arr = np.frombuffer(base64.b64decode(enc), dtype=np.float64)
        except Exception:
            # fallback to JSON list string or Python repr
            try:
                parsed = json.loads(enc)
                arr = np.array(parsed, dtype=np.float64)
            except Exception:
                try:
                    parsed = ast.literal_eval(enc)
                    arr = np.array(parsed, dtype=np.float64)
                except Exception:
                    raise ValueError("search_encoding string is not valid base64 nor JSON list")
    elif isinstance(enc, (bytes, bytearray)):
        arr = np.frombuffer(enc, dtype=np.float64)
    else:
        try:
            arr = np.array(list(enc), dtype=np.float64)
        except Exception:
            raise ValueError("Unsupported search_encoding type")

    if arr.size != 128:
        raise ValueError(f"search_encoding must be length 128; got {arr.size}")
    return arr.reshape((128,))


def find_person_by_face(search_encoding,
                        tolerance: float = 0.6,
                        max_results: int = 20,
                        debug: bool = False,
                        **kwargs) -> List[Dict[str, Any]]:
    """
    Robust DB-backed search:
      - accepts search_encoding (ndarray/list/base64)
      - computes distances to all DB encodings in bulk
      - returns top matches (by distance/confidence). Accepts aliases via kwargs.
    Returns list of dicts containing metadata (id, full_name, photo_path) plus:
      - distance (float)
      - match_confidence (float 0..100)
    """
    # allow aliases
    try:
        limit = int(kwargs.get("limit", kwargs.get("top_k", max_results)))
    except Exception:
        limit = int(max_results or 20)

    # normalize search vector
    try:
        search_vec = _normalize_search_encoding(search_encoding)
    except Exception as e:
        if debug:
            logger.exception("Invalid search encoding: %s", e)
        return []

    # load cached known encodings
    known = _load_people_encodings_cached()
    if not known:
        if debug:
            logger.debug("find_person_by_face: no known encodings found")
        return []

    known_encodings = [k["encoding"] for k in known]
    try:
        distances = face_recognition.face_distance(known_encodings, search_vec)
    except Exception as e:
        logger.exception("face_recognition.face_distance failed: %s", e)
        return []

    combined = []
    for meta, dist in zip(known, distances):
        conf = face_distance_to_confidence(float(dist), threshold=tolerance)
        entry = {
            "id": meta.get("id"),
            "full_name": meta.get("full_name"),
            "photo_path": _normalize_photo_rel_path(meta.get("photo_path")),
            "distance": float(dist),
            "match_confidence": conf
        }
        combined.append(entry)

    # sort by distance ascending
    combined.sort(key=lambda x: x["distance"])

    # Filter by tolerance first; if none pass, fallback to top-N
    filtered = [c for c in combined if c["distance"] <= float(tolerance)]
    if filtered:
        results = filtered[:limit]
    else:
        results = combined[:limit]

    # Return results sorted by match_confidence desc (UI expects best first)
    results.sort(key=lambda x: float(x.get("match_confidence", 0)), reverse=True)

    if debug:
        logger.debug("find_person_by_face: total_known=%d returned=%d tolerance=%s", len(combined), len(results), tolerance)
        for i, r in enumerate(results[:20], 1):
            logger.debug("  %d) id=%s dist=%.4f conf=%s", i, r.get("id"), r.get("distance"), r.get("match_confidence"))

    return results


# ---------------------------------------------------
# Debug helper: run search on an image and return diagnostic info
# ---------------------------------------------------
def debug_find_person_by_image(image_path: str, tolerance: float = 0.6, max_results: int = 20) -> Dict[str, Any]:
    """
    Load an image from disk (path) and run face detection + matching.
    Returns a dict:
      { "faces": [ { "face_index": int, "location": [top,right,bottom,left], "candidates": [ ... ] }, ... ] }
    Each candidate contains the same fields as find_person_by_face plus 'distance' and 'match_confidence'.
    This function is intended for debugging / admin endpoints.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, face_locations)

    output = {"faces": [], "detected_faces_count": len(encodings)}
    for idx, enc in enumerate(encodings):
        # get top candidates with debug logs
        candidates = find_person_by_face(enc, tolerance=tolerance, max_results=max_results, debug=True)
        face_info = {
            "face_index": idx + 1,
            "location": [int(v) for v in face_locations[idx]],
            "candidates": candidates
        }
        output["faces"].append(face_info)

    return output


# ---------------------------------------------------
# Search logging & notifications
# ---------------------------------------------------
from typing import Optional, List, Dict, Any

def log_search(uploaded_name: Optional[str] = None, success: int = 0, matches: int = 0, person_id: Optional[int] = None):
    """Log a face search attempt."""
    try:
        db.session.add(
            SearchLog(
                uploaded_name=uploaded_name,
                success=int(bool(success)),
                matches=int(matches),
                person_id=person_id,
            )
        )
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error logging search: {e}")


def log_best_match_search(uploaded_name: str, matches: List[Dict[str, Any]]):
    """Log best match and send push notification (if configured)."""
    if not matches:
        log_search(uploaded_name=uploaded_name, success=0, matches=0)
        return

    best_match = matches[0]
    log_search(
        uploaded_name=uploaded_name,
        success=1,
        matches=len(matches),
        person_id=best_match.get("id"),
    )

    try:
        send_push_notification(
            best_match["id"],
            title="Person Found!",
            message=f"Your registered person '{best_match['full_name']}' has been matched.",
        )
    except Exception as e:
        logger.error(f"Push notification error: {e}")


# ---------------------------------------------------
# Authentication / stats
# ---------------------------------------------------
def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password."""
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    return None


def get_stats() -> Dict[str, int]:
    """
    Return counts of registrations, total searches, 
    and successful searches (traced/matched).
    """
    try:
        registrations = db.session.query(Person).count()
        total_searches = db.session.query(SearchLog).count()
        successful_searches = db.session.query(SearchLog).filter(SearchLog.success == 1).count()
        return {
            "registrations": registrations,
            "searches": total_searches,
            "searches_traced": successful_searches
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"registrations": 0, "searches": 0, "searches_traced": 0}


# ---------------------------------------------------
# Migration helper
# ---------------------------------------------------
def add_missing_registered_columns(db_file: str):
    """Ensure 'registered_by_*' columns exist on 'people' table (SQLite)."""
    if not os.path.exists(db_file):
        logger.info(f"DB file does not exist: {db_file}")
        return
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='people';")
        if not cursor.fetchone():
            logger.info("Table 'people' does not exist. Skipping registered_by_* migration.")
            return

        columns = [
            ("registered_by_name", "TEXT"),
            ("registered_by_phone", "TEXT"),
            ("registered_by_relation", "TEXT"),
        ]

        cursor.execute("PRAGMA table_info(people);")
        existing_columns = [row[1] for row in cursor.fetchall()]

        for col_name, col_type in columns:
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE people ADD COLUMN {col_name} {col_type};")
                logger.info(f"Added missing column to people: {col_name}")

        conn.commit()
    except Exception as e:
        logger.exception(f"Error adding missing columns to people table: {e}")
    finally:
        if conn:
            conn.close()


def add_missing_user_columns(db_file: str):
    """Ensure users table has required columns (for backward compatibility)."""
    if not os.path.exists(db_file):
        logger.info(f"DB file does not exist: {db_file}")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        if not cursor.fetchone():
            logger.info("Table 'users' does not exist. Skipping users-table migrations.")
            return

        cursor.execute("PRAGMA table_info(users);")
        existing_cols = [row[1] for row in cursor.fetchall()]

        required_columns = {
            "email": "TEXT",
            "phone_number": "TEXT",
            "reset_token": "TEXT",
            "reset_expiry": "DATETIME"
        }

        for col, col_type in required_columns.items():
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type};")
                    logger.info(f"Added missing column to users: {col}")
                except Exception as inner_e:
                    logger.exception(f"Failed to add column {col} to users: {inner_e}")

        conn.commit()
    except Exception as e:
        logger.exception(f"Failed to add missing user columns: {e}")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------
# Admin seeding & DB initialization
# ---------------------------------------------------
def ensure_admin_exists():
    """Ensure at least one admin exists, with email + phone from .env"""
    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_phone = os.getenv("ADMIN_PHONE", "000-000-0000")

    # Only attempt to query if the users table exists
    try:
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
    except Exception as e:
        logger.debug("ensure_admin_exists: couldn't inspect DB engine: %s", e)
        tables = []

    if "users" not in tables:
        logger.info("users table not present; skipping ensure_admin_exists for now.")
        return

    try:
        admin = User.query.filter_by(username="admin").first()
    except Exception as e:
        logger.exception("ensure_admin_exists: DB query failed: %s", e)
        raise

    if not admin:
        admin = User(
            username="admin",
            email=admin_email,
            phone_number=admin_phone,
            role="admin",
            password_hash=generate_password_hash("Alhamdulillah@123")  # default password
        )
        db.session.add(admin)
        db.session.commit()
        logger.info("✅ Created default admin user (username='admin')")
    else:
        # Always keep env values up to date
        admin.email = admin_email
        admin.phone_number = admin_phone
        db.session.commit()


def initialize_database(app):
    """Create tables, apply migrations, seed admin.

    Behavior:
      - Ensure tables exist (db.create_all()).
      - Run table-specific migrations (ALTER TABLE) only if the table exists.
      - If an OperationalError occurs because of schema mismatch, optionally reset DB in development.
    """
    with app.app_context():
        # First, make sure SQLAlchemy is initialized
        db.create_all()

        # Use SQLAlchemy inspector to check actual tables created
        try:
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
        except Exception as e:
            logger.exception("Failed to inspect DB engine: %s", e)
            existing_tables = []

        if not existing_tables:
            # No tables found even after create_all(): log and try again (or recreate file)
            logger.warning("No tables detected after db.create_all(). Attempting to create tables again.")
            try:
                db.create_all()
                inspector = inspect(db.engine)
                existing_tables = inspector.get_table_names()
                logger.info("Tables after retry: %s", existing_tables)
            except Exception as e:
                logger.exception("Retry db.create_all() failed: %s", e)

        # If sqlite, run file-targeted migrations (only when db_file path is resolvable)
        uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if uri and ("sqlite" in uri):
            # normalize sqlite file path
            db_file = uri.replace("sqlite:///", "").replace("sqlite://", "").strip()
            if db_file:
                add_missing_registered_columns(db_file)
                add_missing_user_columns(db_file)

        # Ensure admin (this may raise if users table/columns are missing)
        try:
            ensure_admin_exists()
        except OperationalError as oe:
            logger.exception("OperationalError during ensure_admin_exists: %s", oe)
            # In development, try to recover by recreating schema
            if os.getenv("FLASK_ENV", "development").lower() == "development":
                logger.warning("Dev mode: attempting a DB reset to recover from schema issues.")
                try:
                    db.drop_all()
                    db.create_all()
                    # run migrations again (should be no-ops on fresh DB)
                    if uri and ("sqlite" in uri) and db_file:
                        add_missing_registered_columns(db_file)
                        add_missing_user_columns(db_file)
                    ensure_admin_exists()
                    logger.info("✅ Database reset & initialized successfully.")
                    return
                except Exception as e:
                    logger.exception("Failed while resetting database: %s", e)
                    raise
            else:
                raise
        except Exception as e:
            logger.exception("Unexpected error during DB init: %s", e)
            raise

        logger.info("✅ Database initialized successfully (tables: %s)", existing_tables)


# ---------------------------------------------------
# Push notifications
# ---------------------------------------------------
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = {"sub": os.getenv("VAPID_SUBJECT", "mailto:you@example.com")}


def send_push_notification(person_id: int, title: str, message: str):
    """Send push notification to all subscriptions for a person."""
    if not VAPID_PRIVATE_KEY:
        logger.warning("VAPID_PRIVATE_KEY not set; skipping push.")
        return

    if not _HAS_PYWEBPUSH:
        logger.warning("pywebpush not installed; skipping push.")
        return

    subs = PushSubscription.query.filter_by(person_id=person_id).all()
    for sub in subs:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=json.dumps({"title": title, "message": message}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS,
            )
            logger.info(f"Push sent to subscription {sub.id}")
        except WebPushException as ex:
            logger.error(f"Push failed for subscription {sub.id}: {ex}")
        except Exception as ex:
            logger.exception(f"Unexpected error sending push for subscription {sub.id}: {ex}")


# ---------------------------------------------------
# Standalone init helper (for CLI usage)
# ---------------------------------------------------
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///personfinder.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    try:
        initialize_database(app)
        print("✅ personfinder.db is ready.")
    except Exception as e:
        print(f"❌ Failed to initialize DB: {e}")


# ---------------------------------------------------
# Public API (for imports)
# ---------------------------------------------------
__all__ = [
    "db", "initialize_database", "Person", "SearchLog", "PushSubscription", "User",
    "register_person_to_db", "get_all_registered_people", "get_person_by_id",
    "delete_person_by_id", "find_person_by_face", "log_best_match_search",
    "get_stats", "authenticate_user", "debug_find_person_by_image",
    "clear_people_encodings_cache"
]
# End of database.py
