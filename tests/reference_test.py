"""Check source fidelity equations and checkpoint compatibility."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.dataset import CaptionFeatureDataset, RandomCaptionDataset
from src.evaluate import load_neural_checkpoint
from src.legacy_recurrent import LegacyRecurrent
from src.loss import ranking_loss
from src.model import RetrievalModel
from src.preprocessing import load_vocab
from src.train import train
from tests.smoke_test import create_fixture


class ReferenceTests(unittest.TestCase):
    def test_legacy_cells_against_numpy_equations(self):
        torch.manual_seed(8)
        for kind in ("gru", "lstm"):
            cell = LegacyRecurrent(3, 4, kind)
            value = torch.randn(2, 5, 3, requires_grad=True)
            kernel = cell.kernel.detach().numpy()
            recurrent = cell.recurrent_kernel.detach().numpy()
            bias = cell.bias.detach().numpy()
            expected = np.zeros((2, 4), dtype=np.float32)
            memory = expected.copy()
            hard = lambda x: np.clip(0.2 * x + 0.5, 0, 1)
            for step in range(5):
                projected = value.detach().numpy()[:, step] @ kernel + bias
                if kind == "gru":
                    update = hard(projected[:, :4] + expected @ recurrent[:, :4])
                    reset = hard(projected[:, 4:8] + expected @ recurrent[:, 4:8])
                    candidate = np.tanh(projected[:, 8:] + (reset * expected) @ recurrent[:, 8:])
                    expected = update * expected + (1 - update) * candidate
                else:
                    gates = projected + expected @ recurrent
                    memory = hard(gates[:, 4:8]) * memory + hard(gates[:, :4]) * np.tanh(gates[:, 8:12])
                    expected = hard(gates[:, 12:]) * np.tanh(memory)
            actual = cell(value)
            np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-6)
            actual.square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters()))

    def test_normalization_padding_and_sum_loss(self):
        model = RetrievalModel(10, 6, 3, 4, "gru", normalize_embeddings=True,
                               projection_bias=True, recurrent_implementation="keras_legacy",
                               caption_padding_length=8)
        features = torch.randn(2, 6)
        tokens = torch.tensor([[2, 3], [4, 5]])
        lengths = torch.tensor([2, 2])
        images, captions = model(features, tokens, lengths)
        self.assertTrue(torch.allclose(images.norm(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(captions.norm(dim=1), torch.ones(2), atol=1e-6))
        padded = torch.nn.functional.pad(tokens, (0, 6))
        self.assertTrue(torch.allclose(captions, model.encode_captions(padded, lengths)))
        scores = -(images[:, None] - captions[None]).square().sum(-1)
        expected = (0.1 + scores[0, 1] - scores[0, 0]).clamp_min(0)
        expected += (0.1 + scores[1, 0] - scores[1, 1]).clamp_min(0)
        expected += (0.1 + scores[0, 1] - scores[1, 1]).clamp_min(0)
        expected += (0.1 + scores[1, 0] - scores[0, 0]).clamp_min(0)
        actual = ranking_loss(images, captions, torch.tensor([1, 2]), reduction="sum")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_reference_training_and_old_checkpoints(self):
        Path("work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path("work"), prefix="reference_test_") as directory:
            cfg = create_fixture(Path(directory))
            cfg["model"].update(normalize_embeddings=True, projection_bias=True,
                                recurrent_implementation="keras_legacy", caption_padding_length=12,
                                unknown_glove_random=False)
            cfg["training"].update(batch_size=2, one_caption_per_image=True, loss_reduction="sum")
            vocab = load_vocab(Path(directory) / cfg["data"]["vocab"])
            pairs = CaptionFeatureDataset(cfg, "train", vocab)
            sampled = RandomCaptionDataset(pairs)
            self.assertEqual(len(sampled), 3)
            self.assertEqual(len({sampled[i]["image_id"] for i in range(3)}), 3)
            checkpoint = train(cfg, "gru_glove")
            loaded, _, _ = load_neural_checkpoint(checkpoint, torch.device("cpu"))
            self.assertTrue(loaded.normalize_embeddings)
            self.assertEqual(loaded.recurrent_implementation, "keras_legacy")
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state["learning_rate"], 0.001)
            old = RetrievalModel(len(vocab), 16, 8, 12)
            state["model"] = old.state_dict()
            state.pop("model_options")
            state["recurrent_type"] = "gru"
            torch.save(state, checkpoint)
            restored, _, _ = load_neural_checkpoint(checkpoint, torch.device("cpu"))
            self.assertFalse(restored.normalize_embeddings)
            del sampled, pairs


if __name__ == "__main__":
    unittest.main()
