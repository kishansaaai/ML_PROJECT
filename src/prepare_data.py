"""Download source files when requested and build Karpathy split manifests."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import load_config, project_path
from .preprocessing import build_vocab, save_vocab


KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
TRAIN_URL = "http://images.cocodataset.org/zips/train2014.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url} -> {destination}")
        partial = destination.with_name(destination.name + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return destination


def extract_named(archive: Path, destination: Path, target: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with zipfile.ZipFile(archive) as zf:
        matches = [name for name in zf.namelist() if Path(name).name == target]
        if len(matches) != 1:
            raise ValueError(f"Expected one {target} in {archive}, found {len(matches)}")
        with zf.open(matches[0]) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def extract_images(archive: Path, root: Path, folder: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.startswith(folder + "/")]
        if not members:
            raise ValueError(f"No {folder} images found in {archive}")
        for name in members:
            if name.endswith("/"):
                continue
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe archive member: {name}")
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_subset(annotation: Path, image_root: Path, per_split: int) -> None:
    if per_split < 1:
        raise ValueError("--download-subset must be positive")
    images = json.loads(annotation.read_text(encoding="utf-8"))["images"]
    counts = {"train": 0, "val": 0, "test": 0}
    for image in images:
        split = "train" if image["split"] == "restval" else image["split"]
        if split not in counts or counts[split] >= per_split:
            continue
        relative = Path(image["filepath"]) / image["filename"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid path in Karpathy annotations: {relative}")
        download(f"http://images.cocodataset.org/{relative.as_posix()}", image_root / relative)
        counts[split] += 1
        if all(count == per_split for count in counts.values()):
            break
    if any(count != per_split for count in counts.values()):
        raise ValueError(f"Could not find {per_split} images in every split: {counts}")


def prepare_manifests(cfg: dict, limit_per_split: int | None = None) -> dict[str, int]:
    annotation = project_path(cfg, cfg["data"]["karpathy_json"])
    image_root = project_path(cfg, cfg["data"]["root"])
    output = project_path(cfg, cfg["data"]["manifest_dir"])
    if not annotation.exists():
        raise FileNotFoundError(f"Karpathy annotations missing: {annotation}. Use --download-karpathy.")
    document = json.loads(annotation.read_text(encoding="utf-8"))
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0
    for image in document["images"]:
        split = "train" if image["split"] == "restval" else image["split"]
        if split not in splits:
            continue
        if limit_per_split is not None and len(splits[split]) >= limit_per_split:
            continue
        relative = Path(image.get("filepath", "")) / image["filename"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid image path in annotations: {relative}")
        if not (image_root / relative).is_file():
            missing += 1
            continue
        captions = [sentence.get("raw") or " ".join(sentence["tokens"]) for sentence in image["sentences"]]
        splits[split].append({
            "image_id": int(image.get("cocoid", image.get("imgid"))),
            "path": str(relative).replace("\\", "/"),
            "captions": captions,
        })
    if missing and limit_per_split is None:
        raise FileNotFoundError(
            f"{missing} COCO images from the Karpathy annotations are missing under {image_root}. "
            "Complete the train2014 and val2014 image folders, or use --limit-per-split for a subset."
        )
    if any(not value for value in splits.values()):
        raise ValueError(
            f"All splits need local images; counts: { {key: len(value) for key, value in splits.items()} }. "
            f"Missing files: {missing}. Download COCO 2014 train and val images or check data.root."
        )
    output.mkdir(parents=True, exist_ok=True)
    for split, images in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for image in images:
                handle.write(json.dumps(image, ensure_ascii=False) + "\n")
    train_captions = [caption for image in splits["train"] for caption in image["captions"]]
    vocab = build_vocab(train_captions, cfg["data"]["min_word_frequency"])
    save_vocab(vocab, project_path(cfg, cfg["data"]["vocab"]))
    counts = {split: len(images) for split, images in splits.items()}
    print(f"Prepared {counts}; vocabulary size {len(vocab)}; skipped missing images {missing}.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--download-karpathy", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--download-subset", type=int, default=None,
                        help="Download this many COCO images per Karpathy split")
    parser.add_argument("--download-glove", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = project_path(cfg, cfg["data"]["root"])
    if args.download_karpathy or args.download_subset:
        archive = download(KARPATHY_URL, root / "caption_datasets.zip")
        extract_named(archive, project_path(cfg, cfg["data"]["karpathy_json"]), "dataset_coco.json")
    if args.download_images:
        for url, folder in [(TRAIN_URL, "train2014"), (VAL_URL, "val2014")]:
            extract_images(download(url, root / f"{folder}.zip"), root, folder)
    if args.download_glove:
        glove = project_path(cfg, cfg["data"]["glove"])
        archive = download(GLOVE_URL, glove.parent / "glove.6B.zip")
        extract_named(archive, glove, "glove.6B.300d.txt")
    if args.download_subset:
        download_subset(project_path(cfg, cfg["data"]["karpathy_json"]), root, args.download_subset)
    limit = args.limit_per_split or args.download_subset
    prepare_manifests(cfg, limit)


if __name__ == "__main__":
    main()
