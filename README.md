# RAG LLM Beta

This project demonstrates a simple Retrieval-Augmented Generation pipeline
combined with a neural network classifier. It exposes a FastAPI backend and
provides scripts for training and data ingestion.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py
```

## Ingestion

```bash
python scripts/ingest.py
```

## Running the API

```bash
uvicorn main:app --reload
```

## Querying

After ingestion, send a GET request to `/query?q=your+question`.
