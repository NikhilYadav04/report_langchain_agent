from pydantic import BaseModel


class QueryRequest(BaseModel):
    user_id: str
    query: str


class QueryResponse(BaseModel):
    user_id: str
    query: str
    response: str


class UploadResponse(BaseModel):
    user_id: str
    filename: str
    message: str


class DeleteRequest(BaseModel):
    user_id: str


class DeleteResponse(BaseModel):
    user_id: str
    message: str


class DeleteAllResponse(BaseModel):
    """Response model for the delete_all endpoint."""

    message: str
    path_cleared: str
