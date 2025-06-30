import sys, os; sys.path.append(os.path.abspath("."))
"""Basic tests for RAG pipeline."""
from pathlib import Path
from app.services.rag_service import RagService

def test_ingest_and_query(tmp_path):
    data_file = tmp_path / "doc.txt"
    data_file.write_text("Hello world")
    service = RagService()
    service.ingest_folder(tmp_path)
    answer = service.answer("Hello")
    assert "Hello world" in answer
