import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
DB_FILE = "personfinder.db"

# Define columns to ensure exist
columns_to_add = [
    ("registered_by_name", "TEXT"),
    ("registered_by_phone", "TEXT"),
    ("registered_by_relation", "TEXT")
]

def init_db(db_file):
    db_exists = os.path.exists(db_file)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    if not db_exists:
        logging.info(f"DB file does not exist, creating: {db_file}")

    # Create people table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        guardian_name TEXT,
        phone_number TEXT,
        address TEXT,
        last_seen TEXT,
        photo_path TEXT,
        face_encoding TEXT,
        created_by INTEGER
    );
    """)
    logging.info("Ensured people table exists")

    # Check existing columns
    cursor.execute("PRAGMA table_info(people);")
    existing_columns = [col[1] for col in cursor.fetchall()]

    # Add missing columns
    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE people ADD COLUMN {col_name} {col_type};")
            logging.info(f"Added missing column: {col_name}")

    conn.commit()
    conn.close()
    logging.info(f"Database initialized at {os.path.abspath(db_file)}")

if __name__ == "__main__":
    init_db(DB_FILE)
