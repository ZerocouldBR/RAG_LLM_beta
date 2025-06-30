"""Simple NLP utilities for text preprocessing."""
import re
from typing import List
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return word_tokenize(text)


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply stemming to a list of tokens."""
    return [stemmer.stem(t) for t in tokens]
