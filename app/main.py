from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Health Report RAG Agent",
    description="API for uploading health reports and querying them with a LangChain agent.",
    version="1.0.0",
)

# Include all the API routes from endpoints.py
app.include_router(api_router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to the Health RAG Agent API. Go to /docs to see the API endpoints."
    }
