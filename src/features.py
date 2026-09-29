"""Extract and store VGG features once; training reads them with memory mapping."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .config import ROOT, device_for_run, load_config, project_path
from .image_encoder import VGG19FC7, ten_crop_transform


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest missing: {path}. Run python -m src.prepare_data first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def extract_split(cfg: dict, split: str, pretrained: bool = True, overwrite: bool = False) -> Path:
    images = read_manifest(project_path(cfg, cfg["data"]["manifest_dir"]) / f"{split}.jsonl")
    output_dir = project_path(cfg, cfg["data"]["feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}.npy"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Feature file already exists: {output}. Use --overwrite to re-extract.")
        output.unlink()
    if "TORCH_HOME" not in os.environ:
        torch.hub.set_dir(str(ROOT / "work/torch/hub"))
    model = VGG19FC7(pretrained=pretrained).to(device_for_run())
    image_root = project_path(cfg, cfg["data"]["root"])
    array = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(images), cfg["model"]["image_feature_dim"]),
    )
    try:
        for index, image in enumerate(tqdm(images, desc=f"VGG19 {split}")):
            path = image_root / image["path"]
            with Image.open(path) as picture:
                crops = ten_crop_transform(picture)
            chunks = []
            batch_size = cfg["data"]["image_batch_size"]
            for chunk in crops.split(batch_size):
                chunks.append(model(chunk.to(device_for_run())).cpu())
            array[index] = torch.cat(chunks).mean(dim=0).numpy()
        array.flush()
    except Exception:
        del array
        output.unlink(missing_ok=True)
        raise
    del array
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing VGG feature arrays")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(extract_split(cfg, split, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
