"""Train the isolated model extension and evaluate its validation selected checkpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import device_for_run, load_config, project_path, set_seed
from .dataset import CaptionFeatureDataset, RandomCaptionDataset, collate_pairs
from .evaluate import calculate_ranks, load_neural_checkpoint, neural_embeddings, summarize_ranks
from .improved_model import BidirectionalRetrievalModel, contrastive_retrieval_loss
from .preprocessing import load_glove_matrix, load_vocab
from .retrieve import retrieve
from .source_evaluation import source_ranks


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_run(path: Path):
    """An operating system lock prevents two writers, including after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("An improvement run is already active") from error
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                    sort_keys=True).encode()).hexdigest()


def measured(model, dataset, device, image_count=1000, captions_per_image=5) -> dict:
    images, queries, targets, captions = neural_embeddings(
        model, dataset, image_count, device, captions_per_image=captions_per_image)
    ranks = calculate_ranks(images, queries, targets)
    result = summarize_ranks(ranks, captions, image_count)
    result["ranks"] = ranks
    return result


def run(cfg: dict, resume: bool = False, stop_after_epoch: int | None = None):
    output = project_path(cfg, cfg["evaluation"]["metrics_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output / "run.lock"):
        _run(cfg, resume, stop_after_epoch, output)


def _run(cfg, resume, stop_after_epoch, output):
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("config_fingerprint") not in {None, fingerprint(cfg)}:
        raise ValueError("Existing experiment configuration changed; use a separate output directory")
    if status.get("stage") == "complete":
        print("Improvement experiment already complete", flush=True)
        return
    if status and not resume:
        raise ValueError("An experiment exists; use --resume to preserve its progress")
    status.update(pid=os.getpid(), config=cfg.get("_config_name", "config_improved.yaml"),
                  config_fingerprint=fingerprint(cfg), experiment="bidirectional GRU extension")
    status.setdefault("started_utc", datetime.now(timezone.utc).isoformat())

    def save(stage, **values):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        write_json(status_path, status)

    try:
        set_seed(cfg["seed"])
        torch.set_num_threads(4)
        device = device_for_run()
        print(f"Device: {device}", flush=True)
        save("loading_data", error=None)
        vocab = load_vocab(project_path(cfg, cfg["data"]["vocab"]))
        train_data = CaptionFeatureDataset(cfg, "train", vocab)
        val_data = CaptionFeatureDataset(cfg, "val", vocab)
        train_ids = {row["image_id"] for row in train_data.images}
        val_ids = {row["image_id"] for row in val_data.images}
        if train_ids & val_ids:
            raise ValueError("Training and validation images overlap")
        loader = DataLoader(RandomCaptionDataset(train_data),
                            batch_size=cfg["training"]["batch_size"], shuffle=True,
                            drop_last=False, num_workers=0, collate_fn=collate_pairs)
        glove, found = load_glove_matrix(project_path(cfg, cfg["data"]["glove"]), vocab,
                                         cfg["model"]["word_dim"], cfg["seed"], unknown_random=False)
        if found == 0:
            raise ValueError("No GloVe words were loaded")
        options = {key: cfg["model"][key] for key in ("hidden_dim", "dropout")}
        model = BidirectionalRetrievalModel(len(vocab), cfg["model"]["image_feature_dim"],
            cfg["model"]["word_dim"], cfg["model"]["embedding_dim"], glove_matrix=glove,
            **options).to(device)
        training = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                       weight_decay=training["weight_decay"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=training["learning_rate_milestones"], gamma=training["learning_rate_gamma"])
        checkpoint_dir = project_path(cfg, training["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path, last_path = (checkpoint_dir / f"bigru_glove_{s}.pt" for s in ("best", "last"))
        history, best_key, stale, first_epoch = [], (-1.0, -1.0, -float("inf")), 0, 1
        training_finished = False
        if resume and last_path.exists():
            previous = torch.load(last_path, map_location=device, weights_only=False)
            if previous["config_fingerprint"] != fingerprint(cfg) or previous["vocab"] != vocab:
                raise ValueError("Resume configuration or vocabulary changed")
            model.load_state_dict(previous["model"])
            optimizer.load_state_dict(previous["optimizer"])
            scheduler.load_state_dict(previous["scheduler"])
            history, best_key, stale = previous["history"], tuple(previous["best_key"]), previous["stale"]
            first_epoch = previous["epoch"] + 1
            rng = previous["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([item.cpu() for item in rng["cuda"]])
            training_finished = previous["training_finished"]
        save("training", training_images=len(train_data.images), available_training_captions=len(train_data.pairs),
             validation_images=training["validation_images"],
             validation_queries=training["validation_images"] * training["validation_captions_per_image"],
             glove_words=found, total_epochs=training["epochs"], completed_epoch=first_epoch - 1,
             device=str(device), selection_rule="validation R@10, then R@1, then lower mean rank")
        for epoch in range(first_epoch, training["epochs"] + 1):
            if training_finished:
                break
            started = time.monotonic()
            model.train()
            total_loss = total_samples = 0
            lr = optimizer.param_groups[0]["lr"]
            hard_weight = training["hard_negative_weight"] if epoch > training["hard_negative_warmup_epochs"] else 0.0
            for batch_index, batch in enumerate(loader, 1):
                features, tokens = batch["features"].to(device), batch["tokens"].to(device)
                image_ids = batch["image_ids"].to(device)
                optimizer.zero_grad(set_to_none=True)
                images, captions = model(features, tokens, batch["lengths"])
                loss = contrastive_retrieval_loss(images, captions, image_ids,
                    training["temperature"], training["margin"], hard_weight)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non finite improvement loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(tokens)
                total_samples += len(tokens)
                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                    elapsed = time.monotonic() - started
                    save("training", epoch=epoch, batch=batch_index, batches=len(loader),
                         epoch_elapsed_seconds=round(elapsed, 2), train_loss=total_loss / total_samples)
                    print(f"epoch {epoch} batch {batch_index}/{len(loader)} loss={total_loss/total_samples:.4f} elapsed={elapsed:.1f}s", flush=True)
            model.eval()
            validation = measured(model, val_data, device, training["validation_images"],
                                  training["validation_captions_per_image"])
            key = (validation["r_at_10"], validation["r_at_1"], -validation["mean_rank"])
            if key > best_key:
                best_key, stale, improved = key, 0, True
            else:
                stale, improved = stale + 1, False
            record = {"epoch": epoch, "training_batches": len(loader), "training_samples": total_samples,
                      "train_loss": total_loss / total_samples, "learning_rate": lr,
                      "hard_negative_weight": hard_weight, "seconds": time.monotonic() - started,
                      **{k: validation[k] for k in ("gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank")}}
            history.append(record)
            scheduler.step()
            training_finished = epoch == training["epochs"] or (
                epoch >= training["minimum_epochs"] and stale >= training["patience"])
            state = {"variant": "bigru_glove", "model_type": "bidirectional_gru", "model": model.state_dict(),
                     "vocab": vocab, "model_options": options, "epoch": epoch,
                     "image_feature_dim": cfg["model"]["image_feature_dim"], "word_dim": cfg["model"]["word_dim"],
                     "embedding_dim": cfg["model"]["embedding_dim"], "tokenization": train_data.tokenization,
                     "max_tokens": train_data.max_tokens, "config_fingerprint": fingerprint(cfg),
                     "selection_rule": status["selection_rule"], "validation": validation}
            if improved:
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(state, temporary)
                temporary.replace(best_path)
            last_state = {**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "history": history, "best_key": best_key, "stale": stale,
                          "training_finished": training_finished,
                          "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                                  "torch": torch.get_rng_state(),
                                  "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
            temporary = last_path.with_suffix(".pt.tmp")
            torch.save(last_state, temporary)
            temporary.replace(last_path)
            write_json(checkpoint_dir / "bigru_glove_history.json", history)
            save("training", completed_epoch=epoch, best_validation_r_at_10=best_key[0],
                 last_epoch_seconds=record["seconds"], stale_epochs=stale)
            print(f"epoch {epoch} completed: validation R@10={key[0]:.2f}% R@1={key[1]:.2f}% best={best_key[0]:.2f}% time={record['seconds']:.1f}s", flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch and not training_finished:
                save("saved_for_resume")
                return
        save("evaluation", completed_epoch=history[-1]["epoch"])
        del model, optimizer, scheduler, train_data, loader, glove
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_model, best_vocab, _ = load_neural_checkpoint(best_path, device)
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        test_data = CaptionFeatureDataset(cfg, "test", best_vocab)
        test_ids = {row["image_id"] for row in test_data.images}
        if test_ids & (train_ids | val_ids):
            raise ValueError("Test images overlap with training or validation")
        image_count = cfg["evaluation"]["gallery_images"]
        test_images, test_captions, targets, text = neural_embeddings(
            best_model, test_data, image_count, device, captions_per_image=5)
        ranks = calculate_ranks(test_images, test_captions, targets)
        independent = source_ranks(test_images, test_captions, targets, device)
        if ranks != independent:
            raise ValueError("Matrix rankings differ from independent squared distance rankings")
        gallery_digest = hashlib.sha256(json.dumps([row["image_id"] for row in test_data.images[:image_count]]).encode()).hexdigest()
        final = {"method": "bigru_glove", "epoch": checkpoint["epoch"], "checkpoint": str(best_path),
                 "selection_rule": status["selection_rule"], "config_fingerprint": fingerprint(cfg),
                 "gallery_image_ids_sha256": gallery_digest, "independent_ranks_verified": True,
                 "validation": {k: checkpoint["validation"][k] for k in ("r_at_1", "r_at_10", "mean_rank", "gallery_images", "queries")},
                 "completed_utc": datetime.now(timezone.utc).isoformat()}
        for count, selection in ((5, list(range(len(ranks)))), (1, list(range(0, len(ranks), 5)))):
            selected_ranks = [ranks[i] for i in selection]
            result = {**summarize_ranks(selected_ranks, [text[i] for i in selection], image_count),
                      "method": "bigru_glove", "split": "test", "epoch": checkpoint["epoch"],
                      "captions_per_image": count, "ranks": selected_ranks,
                      "gallery_image_ids_sha256": gallery_digest}
            write_json(output / f"bigru_glove_test_{len(selection)}queries.json", result)
            final[f"test_{len(selection)}queries"] = {k: result[k] for k in ("gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank")}
        examples = {q: retrieve(cfg, "bigru_glove", best_path, q, max_images=image_count)
                    for q in ("dog", "a guy riding a bike next to a train")}
        if any(not Path(row["path"]).is_file() for rows in examples.values() for row in rows):
            raise ValueError("A retrieved image path is missing")
        write_json(output / "retrieval_examples.json", examples)
        write_json(output / "comparison.json", final)
        save("complete", measured_results=final, error=None)
        print(json.dumps(final, indent=2), flush=True)
    except Exception:
        save("failed", error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_improved.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Save after this epoch without testing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_config_name"] = args.config
    run(cfg, args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
