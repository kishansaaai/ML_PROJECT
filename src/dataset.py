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
        self._encoded_captions: list[list[int]] | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        if self._encoded_captions is None:
            # Reuse caption tokenization across epochs; the baseline never needs it.
            self._encoded_captions = [
                encode_caption(caption, self.vocab, self.max_tokens)
                for _, caption in self.pairs
            ]
        image_index, caption = self.pairs[index]
        return {
            "feature": self.features[image_index],
            "tokens": self._encoded_captions[index],
            "image_id": self.images[image_index]["image_id"],
            "image_index": image_index,
            "caption": caption,
        }


def collate_pairs(batch: list[dict]) -> dict:
    lengths_array = np.fromiter((len(item["tokens"]) for item in batch), dtype=np.int64, count=len(batch))
    tokens_array = np.zeros((len(batch), int(lengths_array.max())), dtype=np.int64)
    for index, item in enumerate(batch):
        tokens_array[index, :lengths_array[index]] = item["tokens"]
    return {
        "features": torch.from_numpy(np.stack([item["feature"] for item in batch])),
        "tokens": torch.from_numpy(tokens_array),
        "lengths": torch.from_numpy(lengths_array),
        "image_ids": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "image_indices": [item["image_index"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
