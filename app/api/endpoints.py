from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Response
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeleteRequest,
    DeleteResponse,
    DeleteAllResponse,
)
from app.services import vector_store, agent_service
from app.config import TEMP_UPLOAD_DIR
from app.config import ADMIN
import shutil

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    response: Response, user_id: str = Form(...), file: UploadFile = File(...)
):
    """
    Uploads a PDF report, processes it with OCR, and creates a
    user-specific vector store.
    """
    if not file.filename or not file.filename.endswith(".pdf"):

        response.status_code = status.HTTP_400_BAD_REQUEST
        return UploadResponse(
            statusCode=400,
            filename="",
            message="Invalid file type. Only PDF allowed..",
        )

    # Ensure the temporary upload directory exists
    TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
    # ---

    # Use a unique name for the temp file to avoid conflicts
    temp_path = TEMP_UPLOAD_DIR / f"{user_id}_{file.filename}"

    # Save uploaded file temporarily
    try:
        # Save uploaded file temporarily
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()

        # Process and create vector store
        print(f"Processing upload for user {user_id}...")
        success = vector_store.create_vector_store(user_id, temp_path)

        if success:
            return UploadResponse(
                filename=file.filename,
                statusCode=200,
                message="Report processed and indexed successfully.",
            )
        else:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return UploadResponse(
                filename=file.filename,
                statusCode=500,
                message="Failed to process and index the PDF.",
            )

    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return UploadResponse(
            filename=file.filename,
            statusCode=500,
            message=f"Server Error : {e}",
        )


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest, response: Response):
    """
    Sends a query to the LangChain agent, which will use the
    user's indexed report to answer.
    """
    print(f"Received query from {request.user_id}: {request.query}")
    response_dict = agent_service.run_agent_query(request.user_id, request.query)
    response.status_code = response_dict["code"]
    return QueryResponse(
        query=request.query,
        data=response_dict["message"],
        statusCode=response_dict["code"],
    )


@router.delete("/delete_index", response_model=DeleteResponse)
async def delete_index(request: DeleteRequest, response: Response):
    """
    Deletes the FAISS index folder associated with a user_id.
    """
    print(f"Received delete request for user {request.user_id}")
    success = vector_store.delete_vector_store(request.user_id)

    if success:
        return DeleteResponse(message="User index deleted successfully", statusCode=200)
    else:
        # This can mean it failed OR it didn't exist, which is still a "success"
        response.status_code = status.HTTP_400_BAD_REQUEST
        return DeleteResponse(
            statusCode=400,
            message="User index not found or could not be deleted.",
        )


@router.delete("/delete/all", response_model=DeleteAllResponse)
async def delete_all(key: str, response: Response):
    """
    Deletes ALL FAISS index folders by recursively deleting the main
    storage directory and then recreating it.
    """
    print("Received request to delete ALL user indices.")
    try:

        if key != ADMIN:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return DeleteAllResponse(
                status_code=400,
                path_cleared="",
                message=f"Invalid Key => {key}",
            )
        # 1. Get the base path
        # We find the parent directory of where user indexes are stored.
        # This assumes get_faiss_path(id) returns something like /app/storage/user_id
        # so .parent gives us /app/storage
        base_path = vector_store.get_parent_faiss_path()
        print(base_path)

        if not base_path:
            print(f"Base index directory not found: {base_path}")
            response.status_code = status.HTTP_400_BAD_REQUEST
            return DeleteAllResponse(
                message="Base index directory not found. Nothing to delete.",
                path_cleared=str(base_path),
                statusCode=400,
            )

        # 2. Delete the entire directory tree
        print(f"Deleting directory: {base_path}")
        shutil.rmtree(base_path)

        # 3. CRITICAL: Recreate the base directory
        # If you don't do this, future attempts to save indexes will fail.
        base_path.mkdir(parents=True, exist_ok=True)
        print(f"Recreated empty directory: {base_path}")

        return DeleteAllResponse(
            message="All user indices have been deleted successfully.",
            statusCode=200,
            path_cleared=str(base_path),
        )

    except Exception as e:
        print(f"Error during delete_all operation: {e}")
        # Return a 500 server error
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return DeleteAllResponse(
            statusCode=500,
            message=f"An error occurred while deleting all indices: {e}",
            path_cleared="",
        )
