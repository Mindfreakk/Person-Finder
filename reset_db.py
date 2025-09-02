from app import db, User, ensure_admin_exists
import os

# --- Step 1: Delete old database ---
db_path = "app.db"  # change if your database has a different name
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Old database '{db_path}' deleted.")
else:
    print(f"No existing database found at '{db_path}'.")

# --- Step 2: Create all tables ---
db.create_all()
print("All tables created successfully.")

# --- Step 3: Create default admin user ---
ensure_admin_exists()
print("Default admin user ensured.")

# --- Step 4: Verify ---
admin = User.query.filter_by(username="admin").first()
if admin:
    print(f"Admin created: username='{admin.username}', role='{admin.role}'")
else:
    print("Admin creation failed!")
