"""FastAPI routes for the application."""
from pathlib import Path
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..services.rag_service import RagService
from ..services.model_service import train_classifier
from ..utils.config import get_settings

router = APIRouter()
rag_service = RagService()
settings = get_settings()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a new file to the data folder."""
    data_path = Path("app/data") / file.filename
    with data_path.open("wb") as f:
        f.write(await file.read())
    return JSONResponse({"message": f"Uploaded {file.filename}"})


@router.post("/ingest")
async def ingest_data() -> JSONResponse:
    """Trigger ingestion of the data folder."""
    folder = Path("app/data")
    rag_service.ingest_folder(folder)
    return JSONResponse({"message": "Data ingested"})


@router.post("/train")
async def train_model() -> JSONResponse:
    """Retrain the classifier."""
    model = train_classifier(Path("app/data/train.csv"))
    torch.save(model.state_dict(), "classifier.pt")
    return JSONResponse({"message": "Model trained"})


@router.get("/query")
async def query_rag(q: str) -> JSONResponse:
    """Query the RAG system."""
    try:
        answer = rag_service.answer(q)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"answer": answer})
