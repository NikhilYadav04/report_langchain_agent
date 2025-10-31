# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# ... (your API key loading) ...
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# --- Directory Setup (FOR RAILWAY) ---
# We get the path "/data" from the DATA_DIR env var.
# This path is our persistent volume.
DATA_DIR = Path(os.getenv("DATA_DIR", "."))

FAISS_INDEX_DIR = DATA_DIR / "faiss_indexes"
TEMP_UPLOAD_DIR = DATA_DIR / "temp_uploads"



# import os
# from dotenv import load_dotenv
# from pathlib import Path

# # Load environment variables from .env file
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)

# # --- API Keys ---
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file.")

# # --- Directory Setup ---
# # Base directory of the project (i.e., /health_rag_agent/)
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Define directories for storing data
# FAISS_INDEX_DIR = BASE_DIR / "faiss_indexes"
# TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

# # Ensure these directories exist
# FAISS_INDEX_DIR.mkdir(exist_ok=True)
# TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
