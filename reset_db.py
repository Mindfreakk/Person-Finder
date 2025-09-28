# reset_db.py
import os
from app import app
from database import db

# Always create DB in the same folder as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "personfinder.db")

def reset_database():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Deleted old {DB_FILE}")

    with app.app_context():
        db.drop_all()
        db.create_all()
        print(f"Created fresh {DB_FILE} with latest schema ✅")

if __name__ == "__main__":
    reset_database()
