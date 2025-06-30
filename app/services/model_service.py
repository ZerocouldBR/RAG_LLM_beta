"""Training and evaluation utilities."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import pandas as pd

from ..models.nn_model import Classifier, train_step


def build_dataloader(path: Path, batch_size: int = 4) -> DataLoader:
    """Create a dataloader from a CSV file."""
    df = pd.read_csv(path)
    # naive bag of words representation
    vocab = {word: idx for idx, word in enumerate(set(" ".join(df["text"]).split()))}
    features = []
    for text in df["text"]:
        vec = [0] * len(vocab)
        for word in text.split():
            idx = vocab[word]
            vec[idx] += 1
        features.append(vec)
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_classifier(data_path: Path) -> Classifier:
    """Train a simple classifier."""
    dl = build_dataloader(data_path)
    input_dim = next(iter(dl))[0].shape[1]
    model = Classifier(input_dim, 8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(2):
        for batch in dl:
            loss = train_step(model, optimizer, criterion, batch)
    return model


def cluster_texts(texts: list[str], num_clusters: int = 2) -> list[int]:
    """Cluster texts using sentence embeddings and k-means."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
    return kmeans.labels_.tolist()
