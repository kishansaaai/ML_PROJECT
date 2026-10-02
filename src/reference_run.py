"""Resume the isolated source fidelity run and record only measured results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import traceback

from .baseline import fit_baseline
from .config import load_config, project_path
from .evaluate import evaluate_method
from .retrieve import retrieve
from .train import train


def run(cfg: dict):
    output = project_path(cfg, "outputs/reference")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"completed_training": []}
    if status.get("stage") == "complete":
        print("Reference run is already complete")
        return
    status["pid"] = os.getpid()
    status["config"] = "config_reference.yaml"

    def save(stage, **values):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)
        print(f"Reference stage: {stage}", flush=True)

    try:
        provenance = project_path(cfg, "data/paper_reference/provenance.json")
        if not provenance.exists():
            raise FileNotFoundError("Import and verify the original archive before training")
        status["data_provenance"] = json.loads(provenance.read_text())
        checkpoint_dir = project_path(cfg, cfg["training"]["checkpoint_dir"])
        if "baseline" not in status["completed_training"]:
            save("training_baseline")
            fit_baseline(cfg, checkpoint_dir / "baseline.npz")
            status["completed_training"].append("baseline")
        for variant in ("gru_glove", "gru", "lstm_glove"):
            if variant in status["completed_training"]:
                continue
            save("training_" + variant)
            train(cfg, variant, resume=(checkpoint_dir / f"{variant}_last.pt").exists())
            status["completed_training"].append(variant)
        save("evaluation")
        results = []
        for method in ("baseline", "baseline_weighted", "gru", "gru_glove", "lstm_glove"):
            filename = "baseline.npz" if method.startswith("baseline") else f"{method}_best.pt"
            measured = evaluate_method(cfg, "test", method, checkpoint_dir / filename, 1000)
            if measured["gallery_images"] != 1000 or measured["queries"] != 1000:
                raise ValueError("Final protocol requires exactly 1000 images and queries")
            (output / f"{method}_test_metrics.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
            results.append({key: measured[key] for key in ("method", "r_at_1", "r_at_10", "mean_rank")})
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        examples = {query: retrieve(cfg, "gru_glove", checkpoint_dir / "gru_glove_best.pt", query)
                    for query in ("dog", "a guy riding a bike next to a train")}
        (output / "retrieval_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
        save("complete", measured_results=results, error=None)
    except Exception:
        save("failed", error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_reference.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
