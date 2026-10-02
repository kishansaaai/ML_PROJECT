"""Average-GloVe to VGG fc7 multi-response linear regression."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config, project_path
from .dataset import CaptionFeatureDataset
from .preprocessing import encode_caption, load_glove_matrix, load_vocab


def mean_word_vectors(texts: list[str], vectors: np.ndarray, vocab: dict[str, int], max_tokens: int,
                      tokenization: str = "regex") -> np.ndarray:
    output = np.empty((len(texts), vectors.shape[1]), dtype=np.float32)
    for index, text in enumerate(texts):
        ids = encode_caption(text, vocab, max_tokens, tokenization)
        known = [word_id for word_id in ids if word_id != vocab["<unk>"]]
        output[index] = vectors[known].mean(axis=0) if known else 0
    return output


def fit_baseline(cfg: dict, output: Path) -> Path:
    vocab = load_vocab(project_path(cfg, cfg["data"]["vocab"]))
    vectors, found = load_glove_matrix(
        project_path(cfg, cfg["data"]["glove"]), vocab,
        cfg["model"]["word_dim"], cfg["seed"], unknown_random=False,
    )
    print(f"Loaded GloVe for {found}/{len(vocab)} vocabulary entries.")
    if found == 0:
        raise ValueError("No vocabulary words were found in the GloVe file")
    dataset = CaptionFeatureDataset(cfg, "train", vocab)
    word_dim = vectors.shape[1]
    image_dim = dataset.features.shape[1]
    xtx = np.zeros((word_dim, word_dim), dtype=np.float64)
    xty = np.zeros((word_dim, image_dim), dtype=np.float64)
    batch_size = 1024
    for start in tqdm(range(0, len(dataset.pairs), batch_size), desc="Fitting baseline"):
        pairs = dataset.pairs[start:start + batch_size]
        x = mean_word_vectors([caption for _, caption in pairs], vectors, vocab, dataset.max_tokens,
                              dataset.tokenization).astype(np.float64)
        y = np.asarray(dataset.features[[index for index, _ in pairs]], dtype=np.float64)
        xtx += x.T @ x
        xty += x.T @ y
    weights = np.linalg.lstsq(xtx, xty, rcond=None)[0].astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, weights=weights, word_vectors=vectors)
    print(f"Saved baseline to {output}")
    return output


def load_baseline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline missing: {path}. Run python -m src.baseline.")
    with np.load(path) as data:
        return data["weights"], data["word_vectors"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fit_baseline(cfg, project_path(cfg, args.output or f"{cfg['training']['checkpoint_dir']}/baseline.npz"))


if __name__ == "__main__":
    main()
