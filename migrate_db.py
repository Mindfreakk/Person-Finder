import sqlite3

DB_NAME = 'database.db'

def migrate_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Check if 'age' column exists
    c.execute("PRAGMA table_info(registered_people)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'age' not in columns:
        print("Migrating database: Adding 'age' column.")
        c.execute("ALTER TABLE registered_people ADD COLUMN age INTEGER DEFAULT 0")
        conn.commit()
        print("Migration complete.")
    else:
        print("'age' column already exists. No migration needed.")
    
    conn.close()

if __name__ == '__main__':
    migrate_db()