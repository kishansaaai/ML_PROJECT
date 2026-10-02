"""1,000-image caption-to-image Recall@1, Recall@10, and mean rank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import load_baseline, mean_word_vectors
from .config import device_for_run, load_config, project_path
from .dataset import CaptionFeatureDataset
from .model import RetrievalModel, similarity_matrix
from .preprocessing import encode_caption, load_vocab, tokenize


def load_neural_checkpoint(path: Path, device: torch.device) -> tuple[RetrievalModel, dict, str]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Run python -m src.train first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RetrievalModel(
        vocab_size=len(checkpoint["vocab"]),
        image_feature_dim=checkpoint["image_feature_dim"],
        word_dim=checkpoint["word_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        recurrent_type=checkpoint["recurrent_type"],
        **checkpoint.get("model_options", {}),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint["vocab"], checkpoint["variant"]


def neural_embeddings(
    model: RetrievalModel, dataset: CaptionFeatureDataset, image_count: int,
    device: torch.device, batch_size: int = 128,
    captions_per_image: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    if captions_per_image not in {1, 5}:
        raise ValueError("Caption evaluation requires one or five captions per image")
    if not 2 <= image_count <= len(dataset.images):
        raise ValueError("Image count must fit the dataset and contain at least two images")
    images = []
    with torch.no_grad():
        for start in range(0, image_count, batch_size):
            stop = min(start + batch_size, image_count)
            features = torch.from_numpy(np.array(dataset.features[start:stop], copy=True)).to(device)
            images.append(model.encode_images(features).cpu())
        if any(len(image["captions"]) < captions_per_image for image in dataset.images[:image_count]):
            raise ValueError("An image has too few captions for the selected evaluation")
        pairs = [(index, caption)
                 for index, image in enumerate(dataset.images[:image_count])
                 for caption in image["captions"][:captions_per_image]]
        captions = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = [encode_caption(text, dataset.vocab, dataset.max_tokens, dataset.tokenization)
                       for _, text in batch]
            lengths = torch.tensor([len(ids) for ids in encoded], dtype=torch.long)
            tokens = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long)
            for row, ids in enumerate(encoded):
                tokens[row, :len(ids)] = torch.tensor(ids)
            captions.append(model.encode_captions(tokens.to(device), lengths).cpu())
    image_embeddings, caption_embeddings = torch.cat(images), torch.cat(captions)
    if len(image_embeddings) != image_count or len(caption_embeddings) != image_count * captions_per_image:
        raise ValueError("Actual embedding counts do not match the selected evaluation")
    return image_embeddings, caption_embeddings, [index for index, _ in pairs], [caption for _, caption in pairs]


def calculate_ranks(images: torch.Tensor, queries: torch.Tensor, targets: list[int]) -> list[int]:
    ranks = []
    with torch.no_grad():
        for start in range(0, len(queries), 256):
            scores = similarity_matrix(images, queries[start:start + 256]).T
            batch_targets = torch.tensor(targets[start:start + 256], dtype=torch.long)
            true_scores = scores[torch.arange(len(batch_targets)), batch_targets]
            # A deterministic earlier gallery item wins an exact tie.
            earlier = torch.arange(len(images))[None, :] < batch_targets[:, None]
            rank = 1 + (scores > true_scores[:, None]).sum(1) + ((scores == true_scores[:, None]) & earlier).sum(1)
            ranks.extend(rank.tolist())
    return ranks


def summarize_ranks(ranks: list[int], captions: list[str], image_count: int) -> dict:
    rank_array = np.asarray(ranks)
    lengths = np.asarray([len(tokenize(caption)) for caption in captions])
    by_length = []
    for length in sorted(set(lengths)):
        selection = lengths == length
        if selection.sum() >= 5:
            by_length.append({
                "words": int(length), "queries": int(selection.sum()),
                "r_at_10": float((rank_array[selection] <= 10).mean() * 100),
            })
    return {
        "gallery_images": image_count,
        "queries": len(ranks),
        "r_at_1": float((rank_array <= 1).mean() * 100),
        "r_at_10": float((rank_array <= 10).mean() * 100),
        "mean_rank": float(rank_array.mean()),
        "caption_length": by_length,
    }


def evaluate_method(
    cfg: dict, split: str, method: str, checkpoint: Path, max_images: int = 1000,
    model: RetrievalModel | None = None, vocab: dict[str, int] | None = None,
    captions_per_image: int = 1,
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation is restricted to validation and test splits")
    if method not in {"baseline", "baseline_weighted", "gru", "gru_glove", "lstm_glove"}:
        raise ValueError(f"Unknown method: {method}")
    if method.startswith("baseline") and captions_per_image != 1:
        raise ValueError("Five separate caption evaluation is restricted to neural methods")
    device = device_for_run()
    if method.startswith("baseline"):
        vocab = load_vocab(project_path(cfg, cfg["data"]["vocab"]))
        weights, vectors = load_baseline(checkpoint)
    elif model is None:
        model, vocab, variant = load_neural_checkpoint(checkpoint, device)
        if variant != method:
            raise ValueError(f"Checkpoint is {variant}, requested {method}")
    assert vocab is not None
    dataset = CaptionFeatureDataset(cfg, split, vocab)
    image_count = min(max_images, len(dataset.images))
    if image_count < 2:
        raise ValueError("Evaluation needs at least two images")
    if method.startswith("baseline"):
        images = torch.from_numpy(np.array(dataset.features[:image_count], copy=True))
        if method == "baseline_weighted":
            if any(len(image["captions"]) < 5 for image in dataset.images[:image_count]):
                raise ValueError("Pooled baseline requires five captions for every gallery image")
            pairs = [(i, caption)
                     for i, image in enumerate(dataset.images[:image_count])
                     for caption in image["captions"][:5]]
        else:
            pairs = [(i, image["captions"][0])
                     for i, image in enumerate(dataset.images[:image_count])]
        targets = [i for i, _ in pairs]
        captions = [text for _, text in pairs]
        features = mean_word_vectors(captions, vectors, vocab, dataset.max_tokens, dataset.tokenization)
        if method == "baseline_weighted":
            # Paper's oracle experiment: combine the five held-out captions for each image.
            pooled = [features[np.asarray(targets) == i].mean(axis=0) for i in range(image_count)]
            queries = torch.from_numpy(np.asarray(pooled) @ weights)
            targets = list(range(image_count))
            captions = [" ".join(text for j, text in pairs if j == i) for i in targets]
        else:
            queries = torch.from_numpy(features @ weights)
    else:
        assert model is not None
        images, queries, targets, captions = neural_embeddings(
            model, dataset, image_count, device, captions_per_image=captions_per_image)
    ranks = calculate_ranks(images, queries, targets)
    metrics = summarize_ranks(ranks, captions, image_count)
    metrics.update({"method": method, "split": split, "checkpoint": str(checkpoint)})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--method", choices=["baseline", "baseline_weighted", "gru", "gru_glove", "lstm_glove"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    default_name = "baseline.npz" if args.method.startswith("baseline") else f"{args.method}_best.pt"
    checkpoint = project_path(cfg, args.checkpoint or f"{cfg['training']['checkpoint_dir']}/{default_name}")
    metrics = evaluate_method(cfg, args.split, args.method, checkpoint, args.max_images)
    print(json.dumps({key: value for key, value in metrics.items() if key != "caption_length"}, indent=2))
    metrics_dir = cfg.get("evaluation", {}).get("metrics_dir", "outputs")
    output = project_path(cfg, args.output or f"{metrics_dir}/{args.method}_{args.split}_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
