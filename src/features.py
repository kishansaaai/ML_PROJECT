"""Extract and store VGG features once; training reads them with memory mapping."""
from __future__ import annotations

import argparse
import hashlib
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


def extract_split(cfg: dict, split: str, pretrained: bool = True, overwrite: bool = False,
                  resume: bool = False) -> Path:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    manifest = project_path(cfg, cfg["data"]["manifest_dir"]) / f"{split}.jsonl"
    images = read_manifest(manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output_dir = project_path(cfg, cfg["data"]["feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}.npy"
    progress = output_dir / f"{split}.progress.json"
    if overwrite:
        output.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
    expected = {"manifest_sha256": manifest_hash, "image_count": len(images),
                "feature_dim": cfg["model"]["image_feature_dim"], "pretrained": pretrained}
    start = 0
    if resume:
        if not output.exists() or not progress.exists():
            raise FileNotFoundError(f"Feature array and progress record required to resume {split}")
        state = json.loads(progress.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Manifest or feature settings changed; cannot resume {split}")
        start = int(state["completed"])
        if not 0 <= start <= len(images):
            raise ValueError(f"Invalid feature progress for {split}: {start}")
        if start == len(images):
            return output
        array = np.load(output, mmap_mode="r+")
        if array.shape != (len(images), cfg["model"]["image_feature_dim"]):
            raise ValueError(f"Feature array shape changed; cannot resume {split}")
    else:
        if output.exists():
            raise FileExistsError(f"Feature file already exists: {output}. Use --resume or --overwrite.")
        array = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.float32,
            shape=(len(images), cfg["model"]["image_feature_dim"]),
        )

    def save_progress(completed: int) -> None:
        array.flush()
        temporary = progress.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**expected, "completed": completed}), encoding="utf-8")
        temporary.replace(progress)

    if not resume:
        save_progress(0)
    if "TORCH_HOME" not in os.environ:
        torch.hub.set_dir(str(ROOT / "work/torch/hub"))
    model = VGG19FC7(pretrained=pretrained).to(device_for_run())
    device = device_for_run()
    image_root = project_path(cfg, cfg["data"]["root"])
    try:
        for index in tqdm(range(start, len(images)), initial=start, total=len(images), desc=f"VGG19 {split}"):
            image = images[index]
            path = image_root / image["path"]
            with Image.open(path) as picture:
                crops = ten_crop_transform(picture)
            chunks = []
            batch_size = cfg["data"]["image_batch_size"]
            for chunk in crops.split(batch_size):
                chunks.append(model(chunk.to(device)).cpu())
            array[index] = torch.cat(chunks).mean(dim=0).numpy()
            if (index + 1) % 50 == 0 or index + 1 == len(images):
                save_progress(index + 1)
    finally:
        del model
    del array
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing VGG feature arrays")
    parser.add_argument("--resume", action="store_true", help="Continue interrupted VGG feature extraction")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(extract_split(cfg, split, overwrite=args.overwrite, resume=args.resume))


if __name__ == "__main__":
    main()
