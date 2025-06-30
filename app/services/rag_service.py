"""Service wrapping the RAG pipeline."""
from pathlib import Path
from typing import Optional

from ..models.rag_model import RagPipeline


class RagService:
    """High-level API for interacting with RAG pipeline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.pipeline = RagPipeline(model_name)
        self.data_path: Optional[Path] = None

    def ingest_folder(self, folder: Path) -> None:
        self.pipeline.ingest(folder)
        self.data_path = folder

    def answer(self, query: str) -> str:
        return self.pipeline.query(query)
