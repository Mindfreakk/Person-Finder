import os
import logging
import sqlite3
from flask import Flask
from database import db, initialize_database, add_missing_registered_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Set Absolute DB Path -----------------
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DB_FILE = os.path.join(PROJECT_DIR, "personfinder.db")
logger.info(f"Database file path: {DB_FILE}")

# ----------------- Ensure folder exists -----------------
os.makedirs(PROJECT_DIR, exist_ok=True)

# ----------------- Flask App Setup -----------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_FILE}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ----------------- Initialize SQLAlchemy -----------------
db.init_app(app)

# ----------------- Run Initialization -----------------
with app.app_context():
    # Create all tables
    initialize_database(app)

    # Add missing registered_by_* columns if needed
    add_missing_registered_columns(DB_FILE)

logger.info("Database migration completed successfully!")
