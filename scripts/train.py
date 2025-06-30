"""Script to train the classifier."""
from pathlib import Path
import torch

from app.services.model_service import train_classifier


def main() -> None:
    model = train_classifier(Path("app/data/train.csv"))
    model_path = Path("classifier.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
