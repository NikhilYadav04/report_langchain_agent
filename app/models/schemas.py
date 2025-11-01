from pydantic import BaseModel


class QueryRequest(BaseModel):
    user_id: str
    query: str


class QueryResponse(BaseModel):

    query: str
    data: str
    statusCode: int


class UploadResponse(BaseModel):

    filename: str
    message: str
    statusCode: int


class DeleteRequest(BaseModel):
    user_id: str


class DeleteResponse(BaseModel):

    message: str
    statusCode: int


class DeleteAllResponse(BaseModel):
    """Response model for the delete_all endpoint."""

    message: str
    path_cleared: str
    statusCode: int
