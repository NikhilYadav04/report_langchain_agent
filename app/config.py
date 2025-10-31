# app/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file.")

# --- Directory Setup (FOR RENDER) ---

# We will set an env var "DATA_DIR" in Render to "/data"
# This "/data" is the mount path for our persistent disk.
# We default to "." for local dev if the var isn't set.
DATA_DIR = Path(os.getenv("DATA_DIR", "."))

# Define directories relative to our new DATA_DIR
FAISS_INDEX_DIR = DATA_DIR / "faiss_indexes"
TEMP_UPLOAD_DIR = DATA_DIR / "temp_uploads"

# Ensure these directories exist
FAISS_INDEX_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)