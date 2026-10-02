"""Rank COCO images for a natural-language caption query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import load_baseline, mean_word_vectors
from .config import device_for_run, load_config, project_path
from .dataset import CaptionFeatureDataset
from .evaluate import load_neural_checkpoint
from .model import similarity_matrix
from .preprocessing import encode_caption, load_vocab


def retrieve(cfg: dict, method: str, checkpoint: Path, query: str,
             split: str = "test", top_k: int = 5, max_images: int = 1000) -> list[dict]:
    if split not in {"val", "test"}:
        raise ValueError("Retrieval examples must use validation or test images")
    if not query.strip():
        raise ValueError("Query must contain at least one word")
    if top_k < 1 or max_images < 1:
        raise ValueError("top_k and max_images must be positive")
    if method == "baseline":
        vocab = load_vocab(project_path(cfg, cfg["data"]["vocab"]))
        weights, vectors = load_baseline(checkpoint)
    else:
        model, vocab, variant = load_neural_checkpoint(checkpoint, device_for_run())
        if variant != method:
            raise ValueError(f"Checkpoint is {variant}, requested {method}")
    dataset = CaptionFeatureDataset(cfg, split, vocab)
    count = min(max_images, len(dataset.images))
    with torch.no_grad():
        if method == "baseline":
            images = torch.from_numpy(np.array(dataset.features[:count], copy=True))
            text = mean_word_vectors([query], vectors, vocab, dataset.max_tokens, dataset.tokenization) @ weights
            query_embedding = torch.from_numpy(text)
        else:
            device = device_for_run()
            projected = []
            for start in range(0, count, 128):
                features = torch.from_numpy(np.array(dataset.features[start:start + 128], copy=True)).to(device)
                projected.append(model.encode_images(features).cpu())
            images = torch.cat(projected)
            ids = encode_caption(query, vocab, dataset.max_tokens, dataset.tokenization)
            query_embedding = model.encode_captions(
                torch.tensor([ids], dtype=torch.long, device=device), torch.tensor([len(ids)])
            ).cpu()
        scores = similarity_matrix(images, query_embedding)[:, 0]
        values, indices = scores.topk(min(top_k, count))
    root = project_path(cfg, cfg["data"]["root"])
    return [
        {
            "rank": rank, "score": float(score), "image_id": dataset.images[int(index)]["image_id"],
            "path": str(root / dataset.images[int(index)]["path"]),
            "caption": dataset.images[int(index)]["captions"][0],
        }
        for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", choices=["baseline", "gru", "gru_glove", "lstm_glove"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    default = "baseline.npz" if args.method == "baseline" else f"{args.method}_best.pt"
    results = retrieve(cfg, args.method, project_path(cfg, args.checkpoint or f"{cfg['training']['checkpoint_dir']}/{default}"),
                       args.query, args.split, args.top_k, args.max_images)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
