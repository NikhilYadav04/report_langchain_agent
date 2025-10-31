import uvicorn
from app.config import FAISS_INDEX_DIR, TEMP_UPLOAD_DIR

if __name__ == "__main__":
    print("--- Health RAG Agent Server ---")
    print(f"📁 Storing FAISS indexes in: {FAISS_INDEX_DIR.resolve()}")
    print(f"📂 Storing temp uploads in: {TEMP_UPLOAD_DIR.resolve()}")
    print("Starting Uvicorn server at http://127.0.0.1:8000")

    # 'app.main:app' points to the 'app' variable inside the 'app/main.py' file
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
