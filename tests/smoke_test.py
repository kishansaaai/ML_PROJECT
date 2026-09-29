"""End-to-end structural smoke test on a synthetic COCO-format fixture.

These checks do not measure COCO retrieval quality or use pretrained VGG weights.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.baseline import fit_baseline
from src.config import set_seed
from src.evaluate import evaluate_method, load_neural_checkpoint
from src.features import extract_split
from src.loss import ranking_loss
from src.model import RetrievalModel
from src.prepare_data import prepare_manifests
from src.preprocessing import load_glove_matrix, load_vocab
from src.retrieve import retrieve
from src.train import train


def create_fixture(root: Path) -> dict:
    set_seed(7)
    coco = root / "data" / "coco"
    entries = []
    topics = ["red dog", "blue train", "green bicycle"]
    for split in ["train", "val", "test"]:
        folder = "train2014" if split == "train" else "val2014"
        (coco / folder).mkdir(parents=True, exist_ok=True)
        for index, topic in enumerate(topics):
            image_id = {"train": 100, "val": 200, "test": 300}[split] + index
            filename = f"COCO_{folder}_{image_id:012d}.jpg"
            color = [(225, 45, 50), (40, 70, 220), (35, 190, 65)][index]
            Image.new("RGB", (256, 256), color).save(coco / folder / filename)
            entries.append({
                "cocoid": image_id, "filename": filename, "filepath": folder, "split": split,
                "sentences": [{"raw": f"a {topic} near the street"} for _ in range(5)],
            })
    (coco / "dataset_coco.json").write_text(json.dumps({"images": entries}), encoding="utf-8")
    glove = root / "data" / "glove" / "glove.6B.300d.txt"
    glove.parent.mkdir(parents=True, exist_ok=True)
    with glove.open("w", encoding="utf-8") as handle:
        for index, word in enumerate("a red dog blue train green bicycle near the street".split()):
            vector = np.zeros(8, dtype=np.float32)
            vector[index % 8] = 1
            handle.write(word + " " + " ".join(map(str, vector)) + "\n")
    cfg = {
        "_root": str(root), "seed": 7,
        "data": {
            "root": "data/coco", "karpathy_json": "data/coco/dataset_coco.json",
            "manifest_dir": "data/manifests", "vocab": "data/vocab.json",
            "glove": "data/glove/glove.6B.300d.txt", "min_word_frequency": 1,
            "max_tokens": 12, "feature_dir": "embeddings/vgg19", "image_batch_size": 1,
            "image_workers": 0,
        },
        "model": {"image_feature_dim": 16, "word_dim": 8, "embedding_dim": 12},
        "training": {"batch_size": 6, "epochs": 1, "learning_rate": 0.001,
                     "margin": 0.1, "num_workers": 0, "patience": 2,
                     "eval_max_images": 3, "checkpoint_dir": "checkpoints"},
    }
    prepare_manifests(cfg)
    feature_dir = root / cfg["data"]["feature_dir"]
    feature_dir.mkdir(parents=True)
    for offset, split in enumerate(["train", "val", "test"]):
        features = np.eye(3, 16, dtype=np.float32) + offset * 0.01
        np.save(feature_dir / f"{split}.npy", features)
    return cfg


def run_smoke(vgg: bool, pretrained_vgg: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="ml_project_", dir=Path("work").resolve()) as directory:
        root = Path(directory)
        cfg = create_fixture(root)
        vocab = load_vocab(root / "data/vocab.json")
        matrix, found = load_glove_matrix(root / "data/glove/glove.6B.300d.txt", vocab, 8, 7)
        assert found >= 8
        model = RetrievalModel(len(vocab), 16, 8, 12, "gru", matrix)
        images = torch.rand(3, 16, requires_grad=True)
        tokens = torch.tensor([[2, 3, 0], [4, 5, 6], [7, 8, 9]])
        image_emb, caption_emb = model(images, tokens, torch.tensor([2, 3, 3]))
        loss = ranking_loss(image_emb, caption_emb, torch.tensor([1, 2, 3]), 0.1)
        loss.backward()
        assert torch.isfinite(loss) and images.grad is not None
        assert model.word_embeddings.weight.grad is not None  # pretrained words are fine-tuned
        aligned = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        swapped = aligned.flip(0)
        assert ranking_loss(aligned, aligned, torch.tensor([1, 2])) == 0
        assert ranking_loss(aligned, swapped, torch.tensor([1, 2])) > 0
        baseline_path = fit_baseline(cfg, root / "checkpoints/baseline.npz")
        baseline_metrics = evaluate_method(cfg, "test", "baseline", baseline_path, 3)
        weighted = evaluate_method(cfg, "test", "baseline_weighted", baseline_path, 3)
        assert baseline_metrics["queries"] == 3 and weighted["queries"] == 3
        assert len(retrieve(cfg, "baseline", baseline_path, "red dog", max_images=3)) == 3
        for variant in ["gru", "gru_glove", "lstm_glove"]:
            checkpoint = train(cfg, variant, epochs=1, batch_size=6)
            loaded, _, _ = load_neural_checkpoint(checkpoint, torch.device("cpu"))
            assert isinstance(loaded, RetrievalModel)
            metrics = evaluate_method(cfg, "test", variant, checkpoint, 3)
            assert metrics["queries"] == 3
            if variant == "gru":
                train(cfg, variant, epochs=2, batch_size=6, learning_rate=0.0002, resume=True)
                resumed = torch.load(root / "checkpoints/gru_last.pt", map_location="cpu", weights_only=False)
                assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.0002
            assert len(retrieve(cfg, variant, checkpoint, "blue train", max_images=3)) == 3
        if vgg:
            from src.image_encoder import VGG19FC7, ten_crop_transform
            image_path = root / "data/coco/train2014/COCO_train2014_000000000100.jpg"
            with Image.open(image_path) as image:
                crops = ten_crop_transform(image)
            assert crops.shape == (10, 3, 224, 224)
            extractor = VGG19FC7(pretrained=False)
            with torch.no_grad():
                assert extractor(crops[:1]).shape == (1, 4096)
        if pretrained_vgg:
            cfg_vgg = copy.deepcopy(cfg)
            cfg_vgg["model"]["image_feature_dim"] = 4096
            cfg_vgg["data"]["feature_dir"] = "embeddings/pretrained_vgg"
            cfg_vgg["data"]["manifest_dir"] = "data/vgg_manifest"
            source = root / "data/manifests/train.jsonl"
            manifest_dir = root / "data/vgg_manifest"
            manifest_dir.mkdir()
            (manifest_dir / "train.jsonl").write_text(source.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
            features = np.load(extract_split(cfg_vgg, "train", pretrained=True))
            assert features.shape == (1, 4096) and np.isfinite(features).all()
        print("PASS: data, GloVe, baseline, GRU, LSTM, loss/backward, training, checkpoint, evaluation, retrieval")
        if vgg:
            print("PASS: VGG-19 fc7 shape and ten-crop preprocessing (random weights only)")
        if pretrained_vgg:
            print("PASS: pretrained VGG-19 ten-crop feature extraction")
        print("Fixture is synthetic. These are not COCO results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vgg", action="store_true", help="Also run a VGG-19 forward pass with random weights")
    parser.add_argument("--pretrained-vgg", action="store_true", help="Also extract one image with pretrained VGG-19; downloads weights")
    args = parser.parse_args()
    Path("work").mkdir(exist_ok=True)
    run_smoke(args.vgg, args.pretrained_vgg)
