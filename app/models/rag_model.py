"""Retrieval-Augmented Generation components."""
from pathlib import Path
from typing import List

import numpy as np
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FAISS_AVAILABLE = False
from sentence_transformers import SentenceTransformer

from .nn_model import Classifier


class VectorStore:
    """Simple vector store using FAISS if available."""

    def __init__(self, dim: int) -> None:
        self.texts: List[str] = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.embeddings: list[np.ndarray] = []

    def add(self, embeddings: np.ndarray, texts: List[str]) -> None:
        if FAISS_AVAILABLE:
            self.index.add(embeddings)
        else:
            self.embeddings.extend(embeddings)
        self.texts.extend(texts)

    def search(self, embedding: np.ndarray, k: int = 2) -> List[str]:
        if FAISS_AVAILABLE:
            distances, indices = self.index.search(embedding, k)
            idx_list = indices[0]
        else:
            sims = np.matmul(np.stack(self.embeddings), embedding.T).squeeze()
            idx_list = np.argsort(-sims)[:k]
        return [self.texts[i] for i in idx_list if i < len(self.texts)]


class RagPipeline:
    """A minimal RAG pipeline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.embedder = SentenceTransformer(model_name)
        self.store: VectorStore | None = None

    def ingest(self, folder: Path) -> None:
        texts = []
        for file in folder.glob("*.txt"):
            texts.append(file.read_text())
        embeddings = self.embedder.encode(texts)
        self.store = VectorStore(embeddings.shape[1])
        self.store.add(np.array(embeddings), texts)

    def query(self, question: str) -> str:
        if not self.store:
            raise ValueError("Vector store is empty. Ingest data first.")
        question_emb = self.embedder.encode([question])
        context = self.store.search(np.array(question_emb))
        return "\n".join(context)
