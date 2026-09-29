"""Create the paper's comparison table, R@10 curves, length analysis, and retrieval examples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .config import load_config, project_path
from .retrieve import retrieve


METHODS = ["baseline", "baseline_weighted", "gru", "gru_glove", "lstm_glove"]


def write_comparison(cfg: dict, split: str, output: Path) -> None:
    rows = []
    for method in METHODS:
        path = project_path(cfg, f"outputs/{method}_{split}_metrics.json")
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: metrics[key] for key in ["method", "split", "gallery_images", "queries", "r_at_1", "r_at_10", "mean_rank"]})
    if not rows:
        raise FileNotFoundError("No evaluation files found; run python -m src.evaluate for each trained model")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Method                 R@1       R@10     Mean rank")
    for row in rows:
        print(f"{row['method']:<20} {row['r_at_1']:>6.2f}%   {row['r_at_10']:>6.2f}%   {row['mean_rank']:>9.2f}")


def plot_experiments(cfg: dict, split: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    histories = []
    for method in METHODS[2:]:
        path = project_path(cfg, f"checkpoints/{method}_history.json")
        if path.exists():
            histories.append((method, json.loads(path.read_text(encoding="utf-8"))))
    if histories:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, history in histories:
            ax.plot([item["epoch"] for item in history], [item["val_r_at_10"] for item in history], label=method)
        ax.set(xlabel="Epoch", ylabel="Validation R@10 (%)", title="Retrieval recall during training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "recall_at_10_epochs.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for method in ["baseline", "gru", "gru_glove", "lstm_glove"]:
        path = project_path(cfg, f"outputs/{method}_{split}_metrics.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))["caption_length"]
        if not data:
            continue
        ax.plot([item["words"] for item in data], [item["r_at_10"] for item in data], label=method)
        plotted = True
    if plotted:
        ax.set(xlabel="Caption length (words)", ylabel="R@10 (%)", title=f"Recall by caption length ({split})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "caption_length.png", dpi=160)
    plt.close(fig)


def save_retrieval_examples(cfg: dict, neural_method: str, output: Path, split: str) -> None:
    queries = ["dog", "a guy riding a bike next to a train"]
    for query in queries:
        results = [
            ("baseline", retrieve(cfg, "baseline", project_path(cfg, "checkpoints/baseline.npz"), query, split)),
            (neural_method, retrieve(cfg, neural_method, project_path(cfg, f"checkpoints/{neural_method}_best.pt"), query, split)),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(13, 5))
        for row, (method, matches) in enumerate(results):
            for column, match in enumerate(matches):
                with Image.open(match["path"]) as source:
                    axes[row, column].imshow(source.convert("RGB"))
                axes[row, column].set_title(f"{method} #{match['rank']}", fontsize=9)
                axes[row, column].axis("off")
        fig.suptitle(query)
        fig.tight_layout()
        fig.savefig(output / f"retrieval_{'short' if query == 'dog' else 'long'}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--neural-method", choices=["gru", "gru_glove", "lstm_glove"], default="gru_glove")
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = project_path(cfg, "outputs/experiments")
    write_comparison(cfg, args.split, output)
    plot_experiments(cfg, args.split, output)
    if not args.skip_examples:
        save_retrieval_examples(cfg, args.neural_method, output, args.split)
    print(f"Saved experiments to {output}")


if __name__ == "__main__":
    main()
