import sqlite3
import json

DB_PATH = "people.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get existing columns (if table exists)
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='registered_people';")
    if c.fetchone():
        c.execute("PRAGMA table_info(registered_people);")
        existing_cols = [row[1] for row in c.fetchall()]
        _migrate_schema(conn, existing_cols)
    else:
        _create_fresh_table(conn)

    conn.commit()
    conn.close()

def _create_fresh_table(conn):
    print("🆕 Creating fresh table: registered_people")
    conn.execute("""
        CREATE TABLE registered_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            phone TEXT,
            address TEXT,
            guardian_name TEXT,
            last_seen TEXT,
            face_encoding TEXT,
            photo_path TEXT
        );
    """)

def _migrate_schema(conn, existing_cols):
    """Ensure table has the required columns without breaking if old backup exists."""
    required_cols = [
        "id", "name", "age", "phone", "address",
        "guardian_name", "last_seen", "face_encoding", "photo_path"
    ]

    missing_cols = [col for col in required_cols if col not in existing_cols]
    if not missing_cols:
        print("✅ Database schema is up-to-date.")
        return

    print("⚠ Schema mismatch detected — fixing database table...")

    c = conn.cursor()

    # Drop leftover backup table if it exists
    c.execute("DROP TABLE IF EXISTS registered_people_old;")

    # Rename old table
    c.execute("ALTER TABLE registered_people RENAME TO registered_people_old;")

    # Create new table
    _create_fresh_table(conn)

    # Copy matching columns from old to new
    common_cols = [col for col in required_cols if col in existing_cols]
    col_list = ", ".join(common_cols)
    c.execute(f"""
        INSERT INTO registered_people ({col_list})
        SELECT {col_list} FROM registered_people_old;
    """)

    print(f"✅ Migration complete. Added missing columns: {missing_cols}")

def register_person_to_db(name, age, phone, address, guardian_name, last_seen, face_encoding, photo_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO registered_people (name, age, phone, address, guardian_name, last_seen, face_encoding, photo_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, age, phone, address, guardian_name, last_seen, json.dumps(face_encoding), photo_path))
    conn.commit()
    conn.close()

def get_all_registered_people():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM registered_people")
    rows = c.fetchall()
    conn.close()

    # Convert rows to list of dicts
    people = []
    for row in rows:
        people.append({
            "id": row[0],
            "name": row[1],
            "age": row[2],
            "phone": row[3],
            "address": row[4],
            "guardian_name": row[5],
            "last_seen": row[6],
            "face_encoding": json.loads(row[7]) if row[7] else None,
            "photo_path": row[8]
        })
    return people

def delete_person_by_id(person_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM registered_people WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()

def get_person_by_id(person_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM registered_people WHERE id = ?", (person_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "age": row[2],
            "phone": row[3],
            "address": row[4],
            "guardian_name": row[5],
            "last_seen": row[6],
            "face_encoding": json.loads(row[7]) if row[7] else None,
            "photo_path": row[8]
        }
    return None
