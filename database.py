import sqlite3
import pickle

DB_NAME = 'database.db'

EXPECTED_COLUMNS = [
    'id', 'name', 'age', 'phone', 'address', 'face_encoding', 'photo_path'
]

def create_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()

        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='registered_people';")
        table_exists = c.fetchone()

        if not table_exists:
            # Create fresh table
            _create_table(c)
            conn.commit()
            return

        # Table exists → check columns
        c.execute("PRAGMA table_info(registered_people);")
        existing_cols = [row[1] for row in c.fetchall()]

        if existing_cols != EXPECTED_COLUMNS:
            print("⚠ Schema mismatch detected — fixing database table...")
            _migrate_schema(conn, existing_cols)
            print("✅ Database schema updated successfully.")

def _create_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registered_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            phone TEXT,
            address TEXT,
            face_encoding BLOB,
            photo_path TEXT
        )
    ''')

def _migrate_schema(conn, existing_cols):
    c = conn.cursor()
    # Rename old table
    c.execute("ALTER TABLE registered_people RENAME TO registered_people_old;")
    # Create new table
    _create_table(c)

    # Determine which columns to copy (intersection of expected and existing)
    common_cols = [col for col in EXPECTED_COLUMNS if col in existing_cols]
    cols_str = ", ".join(common_cols)

    # Copy data from old table
    c.execute(f"""
        INSERT INTO registered_people ({cols_str})
        SELECT {cols_str} FROM registered_people_old;
    """)

    # Drop old table
    c.execute("DROP TABLE registered_people_old;")
    conn.commit()

def register_person_to_db(name, age, phone, address, face_encoding, photo_path):
    face_encoding_blob = pickle.dumps(face_encoding)
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO registered_people (name, age, phone, address, face_encoding, photo_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, age, phone, address, face_encoding_blob, photo_path))
        conn.commit()

def get_all_registered_people():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM registered_people')
        people = c.fetchall()

    registered_people = []
    for person in people:
        try:
            face_encoding = pickle.loads(person[5])
        except Exception:
            continue  # skip corrupted rows
        registered_people.append({
            'id': person[0],
            'name': person[1],
            'age': person[2],
            'phone': person[3],
            'address': person[4],
            'face_encoding': face_encoding,
            'photo_path': person[6]
        })

    return registered_people

def get_person_by_id(person_id):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM registered_people WHERE id = ?', (person_id,))
        person = c.fetchone()

    if person:
        try:
            face_encoding = pickle.loads(person[5])
        except Exception:
            face_encoding = None
        return {
            'id': person[0],
            'name': person[1],
            'age': person[2],
            'phone': person[3],
            'address': person[4],
            'face_encoding': face_encoding,
            'photo_path': person[6]
        }

    return None

def delete_person_by_id(person_id):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('DELETE FROM registered_people WHERE id = ?', (person_id,))
        conn.commit()