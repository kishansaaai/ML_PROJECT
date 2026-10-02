"""Audit the released evaluator with five caption queries per test image.

This supplementary protocol does not replace the document's 1000 query table.
The checkpoint choices are fixed before testing: validation best and epoch 50.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import torch

from .config import device_for_run, load_config, project_path
from .dataset import CaptionFeatureDataset
from .evaluate import load_neural_checkpoint, neural_embeddings, summarize_ranks


def source_ranks(images: torch.Tensor, captions: torch.Tensor, targets: list[int],
                 device: torch.device) -> list[int]:
    """Direct squared distances and NumPy argsort, as in the author's t2i.

    Chunking bounds memory without changing the gallery or query ordering.
    """
    images = images.to(device)
    ranks = []
    with torch.no_grad():
        for start in range(0, len(captions), 16):
            queries = captions[start:start + 16].to(device)
            differences = queries[:, None, :] - images[None, :, :]
            errors = differences.square().sum(-1).cpu().numpy()
            for error, target in zip(errors, targets[start:start + 16]):
                ordering = np.argsort(error)
                ranks.append(int(np.where(ordering == target)[0][0]) + 1)
    return ranks


def audit(cfg: dict, selections: list[str]) -> dict:
    device = device_for_run()
    results = []
    for method in ("gru", "gru_glove", "lstm_glove"):
        for selection in selections:
            if selection == "paper":
                epoch = cfg["training"].get("paper_checkpoint_epochs", {}).get(method)
                if epoch is None:
                    continue
                filename = f"{method}_epoch_{epoch}.pt"
            else:
                filename = f"{method}_{selection}.pt"
            path = project_path(cfg, f"{cfg['training']['checkpoint_dir']}/{filename}")
            model, vocab, variant = load_neural_checkpoint(path, device)
            if variant != method:
                raise ValueError("Checkpoint variant does not match the selected method")
            dataset = CaptionFeatureDataset(cfg, "test", vocab)
            images, captions, targets, text = neural_embeddings(
                model, dataset, 1000, device, captions_per_image=5)
            if len(images) != 1000 or len(captions) != 5000:
                raise ValueError("The released evaluator requires 1000 images and 5000 captions")
            ranks = source_ranks(images, captions, targets, device)
            measured = summarize_ranks(ranks, text, len(images))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            measured.update(method=method, split="test", checkpoint=str(path),
                            checkpoint_selection=selection, epoch=checkpoint["epoch"],
                            protocol="released source five captions per image",
                            ranks=ranks)
            results.append(measured)
            print(json.dumps({key: measured[key] for key in (
                "method", "checkpoint_selection", "epoch", "gallery_images", "queries",
                "r_at_1", "r_at_10", "mean_rank")}), flush=True)
            del model, checkpoint, images, captions, dataset
    return {"completed_utc": datetime.now(timezone.utc).isoformat(),
            "author_source_commit": "181baae2a01dc747c8bd47de913a9e35830f1477",
            "selection_rule": "validation best, final epoch, and documented author epochs chosen before test evaluation",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_reference.yaml")
    parser.add_argument("--selection", choices=("best", "last", "paper", "both", "all"), default="both")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    selections = {"both": ["best", "last"], "all": ["best", "last", "paper"]}.get(args.selection, [args.selection])
    result = audit(cfg, selections)
    output = project_path(cfg, args.output or f"{cfg.get('evaluation', {}).get('metrics_dir', 'outputs/reference')}/source_protocol_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
