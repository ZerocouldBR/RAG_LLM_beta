"""Entry point for the FastAPI application."""
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="RAG LLM")
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
