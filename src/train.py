"""Train the paper's GRU, GRU+GloVe, or LSTM+GloVe retrieval model."""
from __future__ import annotations

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import device_for_run, load_config, project_path, set_seed
from .dataset import CaptionFeatureDataset, RandomCaptionDataset, collate_pairs
from .evaluate import evaluate_method
from .loss import ranking_loss
from .model import RetrievalModel
from .preprocessing import load_glove_matrix, load_vocab


def run_epoch(model, loader, optimizer, device, margin: float, training: bool,
              reduction: str = "mean") -> float:
    model.train(training)
    total = 0.0
    batches = 0
    for batch in tqdm(loader, desc="train" if training else "validation", leave=False):
        features = batch["features"].to(device)
        tokens = batch["tokens"].to(device)
        image_ids = batch["image_ids"].to(device)
        with torch.set_grad_enabled(training):
            image_embedding, caption_embedding = model(features, tokens, batch["lengths"])
            loss = ranking_loss(image_embedding, caption_embedding, image_ids, margin, reduction)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ranking loss; lower the learning rate for this dataset")
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def train(cfg: dict, variant: str, epochs: int | None = None, batch_size: int | None = None,
          learning_rate: float | None = None, resume: bool = False):
    if variant not in {"gru", "gru_glove", "lstm_glove"}:
        raise ValueError(f"Unknown model variant: {variant}")
    set_seed(cfg["seed"])
    device = device_for_run()
    print(f"Device: {device}")
    vocab = load_vocab(project_path(cfg, cfg["data"]["vocab"]))
    train_data = CaptionFeatureDataset(cfg, "train", vocab)
    val_data = CaptionFeatureDataset(cfg, "val", vocab)
    size = batch_size or cfg["training"]["batch_size"]
    image_sampling = cfg["training"].get("one_caption_per_image", False)
    train_loader = DataLoader(RandomCaptionDataset(train_data) if image_sampling else train_data,
                              batch_size=size, shuffle=True, drop_last=image_sampling,
                              num_workers=cfg["training"]["num_workers"], collate_fn=collate_pairs)
    validation_images = cfg["training"].get("validation_images", len(val_data.images))
    val_indices = [index for index, (image_index, _) in enumerate(val_data.pairs)
                   if image_index < validation_images]
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=size, shuffle=False,
                            num_workers=cfg["training"]["num_workers"], collate_fn=collate_pairs)
    glove = None
    if variant.endswith("glove"):
        glove, found = load_glove_matrix(
            project_path(cfg, cfg["data"]["glove"]), vocab,
            cfg["model"]["word_dim"], cfg["seed"],
            unknown_random=cfg["model"].get("unknown_glove_random", True),
        )
        print(f"Initialized {found}/{len(vocab)} vocabulary entries with pretrained GloVe.")
        if found == 0:
            raise ValueError("No vocabulary words were found in the GloVe file")
    recurrent_type = "lstm" if variant.startswith("lstm") else "gru"
    model_options = {key: cfg["model"][key] for key in (
        "normalize_embeddings", "projection_bias", "recurrent_implementation", "caption_padding_length"
    ) if key in cfg["model"]}
    model = RetrievalModel(
        len(vocab), cfg["model"]["image_feature_dim"], cfg["model"]["word_dim"],
        cfg["model"]["embedding_dim"], recurrent_type, glove, **model_options,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate if learning_rate is not None else cfg["training"]["learning_rate"],
    )
    checkpoint_dir = project_path(cfg, cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    best_val_loss = float("inf")
    stale = 0
    history = []
    best_path = checkpoint_dir / f"{variant}_best.pt"
    last_path = checkpoint_dir / f"{variant}_last.pt"
    maximum = epochs or cfg["training"]["epochs"]
    first_epoch = 1
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Training checkpoint missing: {last_path}")
        previous = torch.load(last_path, map_location=device, weights_only=False)
        if previous["variant"] != variant or previous["vocab"] != vocab:
            raise ValueError("Training checkpoint does not match the selected variant and vocabulary")
        if previous.get("model_options", {}) != model_options:
            raise ValueError("Training checkpoint uses different model options")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        if learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        best_recall = previous["best_recall"]
        best_val_loss = previous["best_val_loss"]
        stale = previous["stale"]
        history = previous["history"]
        first_epoch = previous["epoch"] + 1
        if "rng" in previous:
            random.setstate(previous["rng"]["python"])
            np.random.set_state(previous["rng"]["numpy"])
            torch.set_rng_state(previous["rng"]["torch"].cpu())
            if torch.cuda.is_available() and previous["rng"]["cuda"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in previous["rng"]["cuda"]])
        print(f"Resuming {variant} at epoch {first_epoch}")
        if stale >= cfg["training"]["patience"]:
            print("Training had already stopped early")
            return best_path
    for epoch in range(first_epoch, maximum + 1):
        reduction = cfg["training"].get("loss_reduction", "mean")
        train_loss = run_epoch(model, train_loader, optimizer, device, cfg["training"]["margin"], True, reduction)
        val_loss = run_epoch(model, val_loader, optimizer, device, cfg["training"]["margin"], False, reduction)
        metrics = evaluate_method(
            cfg, "val", variant, best_path,
            cfg["training"]["eval_max_images"], model=model, vocab=vocab,
        )
        recall = metrics["r_at_10"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_r_at_10": recall})
        improved = recall > best_recall or (recall == best_recall and val_loss < best_val_loss)
        if improved:
            best_recall, best_val_loss, stale = recall, val_loss, 0
        else:
            stale += 1
        state = {
            "variant": variant, "recurrent_type": recurrent_type, "model": model.state_dict(),
            "vocab": vocab,
            "image_feature_dim": cfg["model"]["image_feature_dim"],
            "word_dim": cfg["model"]["word_dim"],
            "embedding_dim": cfg["model"]["embedding_dim"],
            "model_options": model_options,
            "tokenization": train_data.tokenization,
            "max_tokens": train_data.max_tokens,
            "training_config": cfg["training"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch, "best_recall": best_recall, "best_val_loss": best_val_loss,
            "history": history,
        }
        if improved:
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(best_path)
        last_state = {**state, "optimizer": optimizer.state_dict(), "stale": stale,
                      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        temporary = last_path.with_suffix(".pt.tmp")
        torch.save(last_state, temporary)
        temporary.replace(last_path)
        (checkpoint_dir / f"{variant}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
              f"val R@10={recall:.2f}%, best={best_recall:.2f}%")
        if stale >= cfg["training"]["patience"]:
            print(f"Early stopping after {stale} epochs without validation improvement")
            break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", choices=["gru", "gru_glove", "lstm_glove"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue training from the last completed epoch")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = train(cfg, args.variant, args.epochs, args.batch_size, args.learning_rate, args.resume)
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
