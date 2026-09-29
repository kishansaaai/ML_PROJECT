"""Memory-mapped VGG features and caption pairs from COCO manifests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import project_path
from .features import read_manifest
from .preprocessing import encode_caption


class CaptionFeatureDataset(Dataset):
    def __init__(self, cfg: dict, split: str, vocab: dict[str, int]):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.images = read_manifest(project_path(cfg, cfg["data"]["manifest_dir"]) / f"{split}.jsonl")
        feature_path = project_path(cfg, cfg["data"]["feature_dir"]) / f"{split}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"VGG features missing: {feature_path}. Run python -m src.features.")
        self.features = np.load(feature_path, mmap_mode="r")
        if self.features.shape != (len(self.images), cfg["model"]["image_feature_dim"]):
            raise ValueError(f"Feature shape mismatch in {feature_path}: {self.features.shape}")
        self.pairs = [
            (index, caption)
            for index, image in enumerate(self.images)
            for caption in image["captions"]
        ]
        self.vocab = vocab
        self.max_tokens = cfg["data"]["max_tokens"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        image_index, caption = self.pairs[index]
        return {
            "feature": torch.from_numpy(np.array(self.features[image_index], copy=True)),
            "tokens": encode_caption(caption, self.vocab, self.max_tokens),
            "image_id": self.images[image_index]["image_id"],
            "image_index": image_index,
            "caption": caption,
        }


def collate_pairs(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)
    tokens = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(batch):
        tokens[index, :len(item["tokens"])] = torch.tensor(item["tokens"])
    return {
        "features": torch.stack([item["feature"] for item in batch]),
        "tokens": tokens,
        "lengths": lengths,
        "image_ids": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "image_indices": [item["image_index"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
