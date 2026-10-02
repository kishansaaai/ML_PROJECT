"""Verify and import the original feature archive linked by the authors.

The existing COCO annotations verify the image identity of every feature row.
The archive and arrays remain outside Git in the ignored data directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .config import load_config, project_path
from .preprocessing import save_vocab


def prepare_reference(cfg: dict, archive: Path) -> dict:
    destination = project_path(cfg, "data/paper_reference")
    destination.mkdir(parents=True, exist_ok=True)
    entries = [f"data/coco/{split}.txt" for split in ("train", "val", "test")]
    entries += [f"data/coco/images/10crop/{split}.npy" for split in ("train", "val", "test")]
    with zipfile.ZipFile(archive) as zipped:
        for name in entries:
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            # Reading every member verifies its CRC before replacing a complete file.
            temporary = output.with_suffix(output.suffix + ".tmp")
            with zipped.open(name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            temporary.replace(output)

    annotations = json.loads(project_path(cfg, cfg["data"]["karpathy_json"]).read_text(encoding="utf-8"))
    manifests = project_path(cfg, cfg["data"]["manifest_dir"])
    manifests.mkdir(parents=True, exist_ok=True)
    captions_by_split = {}
    counts = {}
    for split in ("train", "val", "test"):
        images = [image for image in annotations["images"]
                  if ("train" if image["split"] == "restval" else image["split"]) == split]
        captions = (destination / f"data/coco/{split}.txt").read_text(encoding="utf-8").splitlines()
        features = np.load(destination / f"data/coco/images/10crop/{split}.npy", mmap_mode="r")
        if features.shape != (len(images), 4096) or len(captions) != len(images) * 5:
            raise ValueError(f"Original archive sizes do not match the Karpathy {split} split")
        records = []
        for index, image in enumerate(images):
            expected = [" ".join(sentence["tokens"]) for sentence in image["sentences"][:5]]
            actual = captions[index * 5:(index + 1) * 5]
            if expected != actual:
                raise ValueError(f"Feature row identity cannot be verified at {split} row {index}")
            records.append({"image_id": image["cocoid"],
                            "path": str(Path(image["filepath"]) / image["filename"]),
                            "captions": actual})
        for start in range(0, len(features), 4096):
            block = features[start:start + 4096]
            if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) == 0):
                raise ValueError(f"Invalid original features in {split}")
        with (manifests / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        captions_by_split[split] = captions
        counts[split] = {"images": len(images), "captions": len(captions)}

    # The released trainer includes the first validation fold in its dictionary.
    frequencies = Counter(word for caption in captions_by_split["train"] + captions_by_split["val"][:5000]
                          for word in caption.split())
    words = list(frequencies)
    order = np.argsort(list(frequencies.values()))[::-1]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({words[index]: number + 2 for number, index in enumerate(order)})
    save_vocab(vocab, project_path(cfg, cfg["data"]["vocab"]))
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    provenance = {"archive_sha256": digest.hexdigest(), "splits": counts,
                  "vocabulary_entries": len(vocab), "feature_rows_verified": True,
                  "source_url": "https://www.cs.toronto.edu/~vendrov/order/coco.zip",
                  "author_source_commit": "181baae2a01dc747c8bd47de913a9e35830f1477"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_reference.yaml")
    parser.add_argument("--archive", default="data/paper_reference/coco.zip")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(prepare_reference(cfg, project_path(cfg, args.archive)), indent=2))


if __name__ == "__main__":
    main()
