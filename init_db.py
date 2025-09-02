# init_db.py
from app import db, app, ensure_admin_exists

# Make sure we are in the Flask app context
with app.app_context():
    # Drop all existing tables (optional, removes all existing data)
    db.drop_all()
    print("Dropped all existing tables.")

    # Create tables based on current models
    db.create_all()
    print("Created all tables based on current models.")

    # Ensure default admin exists
    ensure_admin_exists()
    print("Default admin created successfully (username: admin, password: Alhamdulillah@123).")
