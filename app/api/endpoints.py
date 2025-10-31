from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeleteRequest,
    DeleteResponse,
)
from app.services import vector_store, agent_service
from app.config import TEMP_UPLOAD_DIR
import shutil

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Uploads a PDF report, processes it with OCR, and creates a
    user-specific vector store.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF allowed.",
        )

    # Use a unique name for the temp file to avoid conflicts
    temp_path = TEMP_UPLOAD_DIR / f"{user_id}_{file.filename}"

    # Save uploaded file temporarily
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # Process and create vector store
    print(f"Processing upload for user {user_id}...")
    success = vector_store.create_vector_store(user_id, temp_path)

    if success:
        return UploadResponse(
            user_id=user_id,
            filename=file.filename,
            message="Report processed and indexed successfully.",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process and index the PDF.",
        )


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Sends a query to the LangChain agent, which will use the
    user's indexed report to answer.
    """
    print(f"Received query from {request.user_id}: {request.query}")
    response_text = agent_service.run_agent_query(request.user_id, request.query)
    return QueryResponse(
        user_id=request.user_id, query=request.query, response=response_text
    )


@router.delete("/delete_index", response_model=DeleteResponse)
async def delete_index(request: DeleteRequest):
    """
    Deletes the FAISS index folder associated with a user_id.
    """
    print(f"Received delete request for user {request.user_id}")
    success = vector_store.delete_vector_store(request.user_id)

    if success:
        return DeleteResponse(
            user_id=request.user_id, message="User index deleted successfully."
        )
    else:
        # This can mean it failed OR it didn't exist, which is still a "success"
        return DeleteResponse(
            user_id=request.user_id,
            message="User index not found or could not be deleted.",
        )
