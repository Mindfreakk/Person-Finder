import sqlite3
import os

DB_FILE = "personfinder.db"

def create_db():
    """Recreate the SQLite database with the correct schema."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)   # ❗ deletes old DB to avoid schema mismatch
        print(f"Old {DB_FILE} deleted.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # People table
    c.execute('''
        CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            guardian_name TEXT,
            phone_number TEXT NOT NULL,
            address TEXT,
            last_seen TEXT,
            photo_path TEXT,
            face_encoding TEXT
        )
    ''')

    # Search logs table
    c.execute('''
        CREATE TABLE search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            uploaded_name TEXT,
            success INTEGER DEFAULT 0,
            matches INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database {DB_FILE} created successfully with correct schema!")

if __name__ == "__main__":
    create_db()
