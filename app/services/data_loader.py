"""Data loading utilities with progress logs."""
from pathlib import Path
from typing import Generator, Tuple
import pandas as pd
from tqdm import tqdm


def load_csv_batches(path: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    """Load CSV file in batches."""
    reader = pd.read_csv(path, chunksize=batch_size)
    for chunk in reader:
        yield chunk


def load_text_folder(folder: Path) -> Generator[Tuple[str, str], None, None]:
    """Yield filename and content from a text folder."""
    for file in tqdm(list(folder.glob("*.txt")), desc="Loading texts"):
        yield file.name, file.read_text()
