"""Script to ingest documents into vector store."""
from pathlib import Path
from app.services.rag_service import RagService


def main() -> None:
    service = RagService()
    service.ingest_folder(Path("app/data"))
    print("Ingestion completed")


if __name__ == "__main__":
    main()
