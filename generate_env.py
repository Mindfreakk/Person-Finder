# generate_env.py
import os
from cryptography.fernet import Fernet

ENV_FILE = ".env"

def create_env():
    if os.path.exists(ENV_FILE):
        print("✅ .env file already exists. No changes made.")
        return

    key = Fernet.generate_key().decode()  # Generate secure Fernet key
    with open(ENV_FILE, "w") as f:
        f.write(f"FERNET_KEY={key}\n")

    print(f"🆕 .env file created with Fernet key:\n{key}")

if __name__ == "__main__":
    create_env()
