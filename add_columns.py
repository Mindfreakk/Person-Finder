import sqlite3

db_file = "E:/Projects/PersonFinder/personfinder.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Add missing columns safely
try:
    cursor.execute("ALTER TABLE people ADD COLUMN registered_by_name TEXT;")
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE people ADD COLUMN registered_by_phone TEXT;")
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE people ADD COLUMN registered_by_relation TEXT;")
except sqlite3.OperationalError:
    pass

conn.commit()
conn.close()
print("Columns added successfully.")
