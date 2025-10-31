# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# ... (your API key loading) ...
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# --- Directory Setup (Ephemeral for Render Free Tier) ---
# Use the /tmp directory, which is writeable by www-data.
# THIS IS NOT PERSISTENT. Data will be lost.
DATA_DIR = Path("/tmp")

FAISS_INDEX_DIR = DATA_DIR / "faiss_indexes"
TEMP_UPLOAD_DIR = DATA_DIR / "temp_uploads"

# Ensure these directories exist
FAISS_INDEX_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
