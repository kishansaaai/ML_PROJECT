"""Measure query case/punctuation recovery on validation data without training.

The controlled title-case stress test is separate from the paper benchmark.
It uses the existing GRU + GloVe checkpoint and only the validation split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import torch

from src.config import device_for_run, load_config, project_path
from src.dataset import CaptionFeatureDataset
from src.evaluate import calculate_ranks, load_neural_checkpoint, summarize_ranks
from src.preprocessing import encode_caption, normalize_retrieval_query


def validate(cfg: dict) -> dict:
    device = device_for_run()
    checkpoint = project_path(cfg, f"{cfg['training']['checkpoint_dir']}/gru_glove_best.pt")
    model, vocab, variant = load_neural_checkpoint(checkpoint, device)
    if variant != "gru_glove":
        raise ValueError("This validation checks the UI's GRU + GloVe checkpoint")
    dataset = CaptionFeatureDataset(cfg, "val", vocab)
    count = 1000
    if len(dataset.images) < count or dataset.tokenization != "whitespace":
        raise ValueError("Expected the complete original feature validation split and whitespace tokenizer")
    clean = [image["captions"][0] for image in dataset.images[:count]]
    typed = [caption.title() + "!" for caption in clean]
    recovered = [normalize_retrieval_query(caption, vocab) for caption in typed]

    def encode(text):
        return encode_caption(text, vocab, dataset.max_tokens, dataset.tokenization)

    clean_ids = [encode(text) for text in clean]
    unchanged = sum(encode(normalize_retrieval_query(text, vocab)) == ids
                    for text, ids in zip(clean, clean_ids))
    restored = sum(encode(text) == ids for text, ids in zip(recovered, clean_ids))
    if unchanged != count:
        raise AssertionError("Normalization changed an original validation caption's encoding")
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val", "method": variant, "checkpoint": str(checkpoint),
        "protocol": "first caption of each of the first 1000 validation images",
        "stress_transform": "str.title() followed by one exclamation mark",
        "scope": "controlled input robustness check; not a new paper test benchmark or training run",
        "clean_encodings_preserved": unchanged, "typed_encodings_restored": restored,
        "results": {},
    }
    with torch.inference_mode():
        projected = []
        for start in range(0, count, 128):
            features = torch.from_numpy(np.array(dataset.features[start:min(start + 128, count)], copy=True)).to(device)
            projected.append(model.encode_images(features).cpu())
        images = torch.cat(projected)
        if len(images) != count:
            raise AssertionError("Validation gallery must contain exactly 1000 embeddings")
        for name, queries in (("clean_original", clean), ("typed_before_fix", typed), ("typed_after_fix", recovered)):
            caption_vectors = []
            for start in range(0, count, 128):
                ids = [encode(text) for text in queries[start:start + 128]]
                lengths = torch.tensor([len(tokens) for tokens in ids])
                tokens = torch.zeros((len(ids), int(lengths.max())), dtype=torch.long, device=device)
                for row, token_ids in enumerate(ids):
                    tokens[row, :len(token_ids)] = torch.tensor(token_ids, device=device)
                caption_vectors.append(model.encode_captions(tokens, lengths).cpu())
            captions = torch.cat(caption_vectors)
            if len(captions) != count:
                raise AssertionError("Each condition must contain exactly 1000 caption queries")
            ranks = calculate_ranks(images, captions, list(range(count)))
            metrics = summarize_ranks(ranks, queries, count)
            metrics["ranks"] = ranks
            report["results"][name] = metrics
            print(json.dumps({"condition": name, **{key: metrics[key] for key in
                              ("gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank")}}), flush=True)
    if restored == count and report["results"]["clean_original"]["ranks"] != report["results"]["typed_after_fix"]["ranks"]:
        raise AssertionError("Restored encodings did not reproduce the clean caption rankings")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_author.yaml")
    parser.add_argument("--output", default="outputs/input_validation/query_normalization.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = validate(cfg)
    output = project_path(cfg, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
