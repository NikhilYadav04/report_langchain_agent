import shutil
import ocrmypdf
from pathlib import Path
from app.config import FAISS_INDEX_DIR, GOOGLE_API_KEY

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# --- Initialize Global Components ---

# Use the 'models/' prefix for the v1.5 API
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

# --- Helper Function ---


def get_faiss_path(user_id: str) -> Path:
    """Gets the standardized path for a user's FAISS index."""
    return FAISS_INDEX_DIR / f"faiss_index_{user_id}"


# --- Core Service Functions ---


def create_vector_store(user_id: str, pdf_path: Path) -> bool:
    """Processes a PDF and creates a new FAISS vector store for the user."""
    index_path = get_faiss_path(user_id)

    # Clear any old index for this user
    if index_path.exists():
        shutil.rmtree(index_path)

    ocr_pdf_path = pdf_path.with_suffix(".ocr.pdf")

    try:
        # 1. OCR the PDF to make it searchable (from your notebook)
        print(f"Starting OCR for {pdf_path.name}...")
        ocrmypdf.ocr(
            pdf_path, ocr_pdf_path, language="eng", force_ocr=True, progress_bar=False
        )
        print("OCR complete.")

        # 2. Load the OCR'd PDF
        loader = PDFPlumberLoader(str(ocr_pdf_path))
        docs = loader.load()

        # 3. Split documents into chunks
        chunks = text_splitter.split_documents(docs)

        # 4. Create FAISS index from chunks
        print(f"Creating FAISS index for {user_id}...")
        vectorstore = FAISS.from_documents(embedding=embeddings, documents=chunks)

        # 5. Save the index locally
        vectorstore.save_local(str(index_path))
        print(f"FAISS index saved to {index_path}")

        return True

    except Exception as e:
        print(f"Error creating vector store for {user_id}: {e}")
        # Clean up failed artifacts
        if index_path.exists():
            shutil.rmtree(index_path)
        return False

    finally:
        # 6. Clean up temporary files
        if pdf_path.exists():
            pdf_path.unlink()
        if ocr_pdf_path.exists():
            ocr_pdf_path.unlink()


def delete_vector_store(user_id: str) -> bool:
    """Deletes a user's FAISS index folder."""
    index_path = get_faiss_path(user_id)

    if index_path.exists():
        try:
            shutil.rmtree(index_path)
            print(f"Deleted index for {user_id}")
            return True
        except Exception as e:
            print(f"Error deleting index for {user_id}: {e}")
            return False

    print(f"Index not found for {user_id}, nothing to delete.")
    return False  # False because it didn't exist to be deleted


def load_vector_store(user_id: str) -> FAISS | None:
    """Loads an existing FAISS vector store for a user."""
    index_path = get_faiss_path(user_id)
    print(f"Index path is {index_path}")

    if not index_path.exists():
        print(f"No index found for user {user_id}")
        return None

    try:
        return FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS index for {user_id}: {e}")
        return None


def get_retriever(user_id: str) -> VectorStoreRetriever | None:
    """
    Loads the FAISS index for a user and returns it as a retriever.
    This is based on Cell 31 of your notebook.
    """
    print(f"🔍 Loading FAISS index and creating retriever for user: {user_id} ...")
    vectorstore = load_vector_store(user_id)

    if vectorstore:
        # You can customize search_type and search_kwargs here if needed
        # e.g., return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        return vectorstore.as_retriever()
    else:
        return None
