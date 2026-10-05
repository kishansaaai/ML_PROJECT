"""Choose an original and new encoder mixture using validation, then test it once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .config import device_for_run, load_config, project_path
from .dataset import CaptionFeatureDataset
from .evaluate import calculate_ranks, load_neural_checkpoint, neural_embeddings, summarize_ranks
from .improved_model import ValidationEnsemble
from .improve_run import exclusive_run, write_json
from .retrieve import retrieve
from .source_evaluation import source_ranks


WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def definition(state):
    keys = ("model_type", "vocab", "image_feature_dim", "word_dim", "embedding_dim",
            "recurrent_type", "model_options")
    return {key: state[key] for key in keys if key in state}


def choose_weight(original_images, original_captions, candidate_images, candidate_captions,
                  targets, captions):
    measurements = []
    for weight in WEIGHTS:
        images = torch.cat((weight ** 0.5 * original_images,
                            (1 - weight) ** 0.5 * candidate_images), dim=1)
        queries = torch.cat((weight ** 0.5 * original_captions,
                             (1 - weight) ** 0.5 * candidate_captions), dim=1)
        rank = calculate_ranks(images, queries, targets)
        result = summarize_ranks(rank, captions, len(images))
        measurements.append({"original_weight": weight,
                             **{k: result[k] for k in ("gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank")}})
    selected = max(measurements, key=lambda row: (row["r_at_10"], row["r_at_1"], -row["mean_rank"]))
    return selected, measurements


def run(cfg, plan_only=False):
    output = project_path(cfg, cfg["evaluation"]["metrics_dir"])
    with exclusive_run(output / "selection.lock"):
        plan_path = output / "selection_plan.json"
        plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "original_checkpoint": "checkpoints/author/gru_glove_best.pt",
                "candidate_checkpoint": f"{cfg['training']['checkpoint_dir']}/bigru_glove_best.pt",
                "original_weights": list(WEIGHTS), "validation_images": 1000, "validation_queries": 5000,
                "selection_rule": "validation R@10, then R@1, then lower mean rank",
                "test_used_for_selection": False}
        if not plan_path.exists():
            write_json(plan_path, plan)
        else:
            recorded = json.loads(plan_path.read_text())
            if any(recorded[k] != plan[k] for k in plan if k != "created_utc"):
                raise ValueError("Fixed validation selection plan changed")
        if plan_only:
            print(f"Fixed validation selection plan saved to {plan_path}", flush=True)
            return
        if (output / "selected_comparison.json").exists():
            print("Validation selected mixture already measured", flush=True)
            return
        status = json.loads((output / "status.json").read_text())
        if status["stage"] != "complete":
            raise RuntimeError("Complete the new encoder training before selecting a mixture")
        device = device_for_run()
        torch.set_num_threads(4)
        original_path = project_path(cfg, plan["original_checkpoint"])
        candidate_path = project_path(cfg, plan["candidate_checkpoint"])
        original, vocab, _ = load_neural_checkpoint(original_path, device)
        candidate, candidate_vocab, _ = load_neural_checkpoint(candidate_path, device)
        if vocab != candidate_vocab:
            raise ValueError("Encoder vocabularies differ")
        original_state = torch.load(original_path, map_location="cpu", weights_only=False)
        candidate_state = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if not original_state.get("model_options", {}).get("normalize_embeddings", False):
            raise ValueError("The original encoder must use normalized embeddings")
        if (original_state["max_tokens"], original_state["tokenization"]) != (
                candidate_state["max_tokens"], candidate_state["tokenization"]):
            raise ValueError("Encoder caption conventions differ")
        validation_data = CaptionFeatureDataset(cfg, "val", vocab)
        original_images, original_captions, targets, captions = neural_embeddings(
            original, validation_data, 1000, device, captions_per_image=5)
        candidate_images, candidate_captions, other_targets, other_captions = neural_embeddings(
            candidate, validation_data, 1000, device, captions_per_image=5)
        if targets != other_targets or captions != other_captions:
            raise ValueError("Validation query orders differ")
        selected, measurements = choose_weight(original_images, original_captions,
                                                candidate_images, candidate_captions, targets, captions)
        print(f"Selected original weight {selected['original_weight']} from validation only: {selected['r_at_10']:.2f}% R@10", flush=True)
        write_json(output / "selection_validation.json", {"selected": selected, "candidates": measurements,
                   "selection_completed_utc": datetime.now(timezone.utc).isoformat(), "test_used": False})
        ensemble = ValidationEnsemble(original, candidate, selected["original_weight"]).eval()
        path = project_path(cfg, f"{cfg['training']['checkpoint_dir']}/ensemble_glove_best.pt")
        state = {"model_type": "validation_ensemble", "variant": "ensemble_glove", "model": ensemble.state_dict(),
                 "original_definition": definition(original_state), "candidate_definition": definition(candidate_state),
                 "original_weight": selected["original_weight"], "vocab": vocab,
                 "image_feature_dim": cfg["model"]["image_feature_dim"], "word_dim": cfg["model"]["word_dim"],
                 "embedding_dim": original_state["embedding_dim"] + candidate_state["embedding_dim"],
                 "max_tokens": candidate_state["max_tokens"], "tokenization": candidate_state["tokenization"],
                 "epoch": candidate_state["epoch"], "validation": selected, "selection_rule": plan["selection_rule"]}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        del original_images, original_captions, candidate_images, candidate_captions
        test_data = CaptionFeatureDataset(cfg, "test", vocab)
        test_images, test_captions, targets, captions = neural_embeddings(
            ensemble, test_data, 1000, device, captions_per_image=5)
        ranks = calculate_ranks(test_images, test_captions, targets)
        if ranks != source_ranks(test_images, test_captions, targets, device):
            raise ValueError("Selected model rankings differ from independent squared distances")
        digest = hashlib.sha256(json.dumps([row["image_id"] for row in test_data.images[:1000]]).encode()).hexdigest()
        report = {"method": "ensemble_glove", "original_weight": selected["original_weight"],
                  "candidate_epoch": candidate_state["epoch"], "original_epoch": original_state["epoch"],
                  "validation": selected, "selection_rule": plan["selection_rule"], "test_used_for_selection": False,
                  "independent_ranks_verified": True, "gallery_image_ids_sha256": digest,
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        for count, indices in ((5, list(range(5000))), (1, list(range(0, 5000, 5)))):
            selected_ranks = [ranks[i] for i in indices]
            metrics = {**summarize_ranks(selected_ranks, [captions[i] for i in indices], 1000),
                       "method": "ensemble_glove", "captions_per_image": count,
                       "ranks": selected_ranks, "gallery_image_ids_sha256": digest}
            write_json(output / f"ensemble_glove_test_{len(indices)}queries.json", metrics)
            report[f"test_{len(indices)}queries"] = {k: metrics[k] for k in (
                "gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank")}
        examples = {q: retrieve(cfg, "ensemble_glove", path, q)
                    for q in ("dog", "a guy riding a bike next to a train")}
        if any(not Path(row["path"]).is_file() for rows in examples.values() for row in rows):
            raise ValueError("A selected model example image is missing")
        write_json(output / "selected_retrieval_examples.json", examples)
        write_json(output / "selected_comparison.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_improved.yaml")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), args.plan_only)


if __name__ == "__main__":
    main()
