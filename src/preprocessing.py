from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SPECIAL = ["<pad>", "<unk>"]
QUERY_EDGE_PUNCTUATION = string.punctuation + "“”‘’«»…"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(captions: list[str], min_frequency: int = 2) -> dict[str, int]:
    counts = Counter(token for caption in captions for token in tokenize(caption))
    words = sorted(word for word, count in counts.items() if count >= min_frequency)
    return {word: index for index, word in enumerate(SPECIAL + words)}


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")


def load_vocab(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary missing: {path}. Run python -m src.prepare_data first.")
    return json.loads(path.read_text(encoding="utf-8"))


def encode_caption(text: str, vocab: dict[str, int], max_tokens: int,
                   tokenization: str = "regex") -> list[int]:
    words = text.split() if tokenization == "whitespace" else tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in words[:max_tokens]]
    return ids or [vocab["<unk>"]]


def normalize_retrieval_query(text: str, vocab: dict[str, int]) -> str:
    """Recover known typed words without changing training tokenization.

    Exact vocabulary entries keep their capitalization and internal punctuation.
    Case and surrounding punctuation variants are recovered only when their
    replacement is present in the saved vocabulary. Unknown words stay unknown.
    """
    words = []
    for word in text.split():
        stripped = word.strip(QUERY_EDGE_PUNCTUATION)
        candidates = (word, word.lower(), stripped, stripped.lower())
        known = next((candidate for candidate in candidates if candidate in vocab), None)
        if known is not None:
            words.append(known)
        elif stripped:
            words.append(word)
    return " ".join(words)


def load_glove_matrix(
    path: Path, vocab: dict[str, int], dimension: int, seed: int, unknown_random: bool = True
) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"GloVe file missing: {path}. Run python -m src.prepare_data --download-glove.")
    rng = np.random.default_rng(seed)
    matrix = (rng.normal(0, 0.02, (len(vocab), dimension)) if unknown_random
              else np.zeros((len(vocab), dimension))).astype(np.float32)
    matrix[vocab["<pad>"]] = 0
    found = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            word, separator, values = line.partition(" ")
            if not separator or word not in vocab:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.size == dimension:
                matrix[vocab[word]] = vector
                found += 1
    return matrix, found
