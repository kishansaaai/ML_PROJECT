"""Check source fidelity equations and checkpoint compatibility."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import CaptionFeatureDataset, RandomCaptionDataset, SourceBatchSampler
from src.evaluate import load_neural_checkpoint, neural_embeddings, calculate_ranks
from src.legacy_recurrent import LegacyRecurrent
from src.legacy_optimizer import LegacyAdam
from src.source_evaluation import source_ranks
from src.loss import ranking_loss
from src.model import RetrievalModel
from src.preprocessing import encode_caption, load_vocab, normalize_retrieval_query
from src.train import train
from src.retrieve import retrieve
from tests.smoke_test import create_fixture


class ReferenceTests(unittest.TestCase):
    def test_query_normalization_preserves_vocabulary_and_unknown_words(self):
        vocab = {"<pad>": 0, "<unk>": 1, "US": 2, "x-ray": 3, "dog's": 4}
        original = "US x-ray dog's"
        self.assertEqual(normalize_retrieval_query(original, vocab), original)
        self.assertEqual(normalize_retrieval_query("“US” X-ray DOG'S.", vocab), original)
        self.assertEqual(normalize_retrieval_query("Zxqv!", vocab), "Zxqv!")
        self.assertEqual(normalize_retrieval_query("!!! ???", vocab), "")

    def test_retrieval_case_and_punctuation_keep_the_matching_image(self):
        torch.manual_seed(16)
        vocab = {"<pad>": 0, "<unk>": 1, "a": 2, "dog": 3, "catching": 4, "frisbee": 5}
        model = RetrievalModel(len(vocab), 4, 3, 4, "gru", normalize_embeddings=True,
                               recurrent_implementation="keras_legacy", caption_padding_length=8)
        with torch.no_grad():
            model.image_projection.weight.copy_(torch.eye(4))
            features = model.encode_captions(torch.tensor([[2, 3, 4, 2, 5], [1, 1, 1, 1, 1]]),
                                             torch.tensor([5, 5])).numpy()
        dataset = SimpleNamespace(features=features,
                                  images=[{"image_id": i, "path": f"{i}.jpg", "captions": ["a dog catching a frisbee"]}
                                          for i in range(2)], max_tokens=8, tokenization="whitespace")
        cfg = {"_root": ".", "data": {"root": "."}}
        # Training/evaluation encoding retains the original source convention.
        self.assertEqual(encode_caption("A Dog Catching A Frisbee", vocab, 8, "whitespace"), [1] * 5)
        with patch("src.retrieve.load_neural_checkpoint", return_value=(model, vocab, "gru")), \
             patch("src.retrieve.CaptionFeatureDataset", return_value=dataset), \
             patch("src.retrieve.device_for_run", return_value=torch.device("cpu")):
            original = retrieve(cfg, "gru", Path("unused"), "a dog catching a frisbee", top_k=2, max_images=2)
            for query in ("A Dog Catching A Frisbee!", "“A Dog Catching A Frisbee!”"):
                recovered = retrieve(cfg, "gru", Path("unused"), query, top_k=2, max_images=2)
                self.assertEqual([row["image_id"] for row in recovered], [row["image_id"] for row in original])
                self.assertEqual(recovered[0]["image_id"], 0)
                np.testing.assert_allclose([row["score"] for row in recovered], [row["score"] for row in original], atol=1e-6)
            for query in ("!!! ???", "zxqv qqqrrr", "<pad>"):
                with self.assertRaisesRegex(ValueError, "no known words"):
                    retrieve(cfg, "gru", Path("unused"), query, max_images=2)

    def test_retrieval_excludes_candidate_outside_gallery(self):
        vocab = {"<pad>": 0, "<unk>": 1, "word": 2}
        dataset = SimpleNamespace(features=np.eye(4, dtype=np.float32),
                                  images=[{"image_id": i, "path": f"{i}.jpg", "captions": ["word"]} for i in range(4)],
                                  max_tokens=3, tokenization="whitespace")
        class Encoder:
            def encode_images(self, features):
                return features
            def encode_captions(self, tokens, lengths):
                # This query would select the excluded fourth image if the
                # final encoder batch leaked beyond the requested boundary.
                return torch.tensor([[0., 0., 0., 1.]])
        with patch("src.retrieve.load_neural_checkpoint", return_value=(Encoder(), vocab, "gru")), \
             patch("src.retrieve.CaptionFeatureDataset", return_value=dataset), \
             patch("src.retrieve.device_for_run", return_value=torch.device("cpu")):
            rows = retrieve({"_root": ".", "data": {"root": "."}}, "gru", Path("unused"), "word",
                            top_k=3, max_images=3)
        self.assertEqual({row["image_id"] for row in rows}, {0, 1, 2})

    def test_legacy_adam_numpy_updates_and_resumption(self):
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64))
        optimizer = LegacyAdam([parameter])
        expected = parameter.detach().numpy().copy()
        average, square = np.zeros(3), np.zeros(3)
        for step in range(1, 5):
            gradient = np.array([1e-8 * step, -0.1 * step, 0.2])
            average = 0.9 * average + 0.1 * gradient
            square = 0.999 * square + 0.001 * gradient ** 2
            rate = 0.001 * np.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
            expected -= rate * average / (np.sqrt(square) + 1e-7)
            parameter.grad = torch.from_numpy(gradient)
            optimizer.step()
            np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=1e-12, atol=1e-12)
            if step == 2:
                restored = LegacyAdam([parameter])
                restored.load_state_dict(optimizer.state_dict())
                optimizer = restored

    def test_source_generator_ceil_steps_and_full_batches(self):
        pairs = SimpleNamespace(images=[{} for _ in range(5)],
                                pairs=[(i, "word") for i in range(5) for _ in range(5)])
        sampler = SourceBatchSampler(pairs, 2)
        batches = list(sampler)
        self.assertEqual(len(sampler), 3)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        self.assertEqual(len({idx // 5 for batch in batches[:2] for idx in batch}), 4)
        self.assertTrue(all(len({idx // 5 for idx in batch}) == 2 for batch in batches))

    def test_gallery_boundary_and_five_caption_query_targets(self):
        # Four available images but only three allowed: the partial final batch
        # must exclude the fourth image even when batch_size is two.
        vocab = {"<pad>": 0, "<unk>": 1, "word": 2}
        data = SimpleNamespace(features=np.eye(4, dtype=np.float32),
                               images=[{"captions": ["word"] * 5} for _ in range(4)],
                               vocab=vocab, max_tokens=3, tokenization="whitespace")
        class Encoder:
            def encode_images(self, x):
                return x
            def encode_captions(self, tokens, lengths):
                return torch.ones(len(tokens), 4)
        for count in (1, 5):
            images, queries, targets, captions = neural_embeddings(
                Encoder(), data, 3, torch.device("cpu"), batch_size=2, captions_per_image=count)
            self.assertEqual(tuple(images.shape), (3, 4))
            self.assertEqual(len(queries), 3 * count)
            self.assertEqual(targets, [i for i in range(3) for _ in range(count)])
            self.assertEqual(len(captions), len(targets))
            # The same squared distances ranked independently with NumPy.
            distances = ((queries.numpy()[:, None] - images.numpy()[None]) ** 2).sum(-1)
            expected = [int(np.where(np.argsort(row, kind="stable") == target)[0][0]) + 1
                        for row, target in zip(distances, targets)]
            self.assertEqual(calculate_ranks(images, queries, targets), expected)
            self.assertEqual(source_ranks(images, queries, targets, torch.device("cpu")), expected)

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
            cfg["training"].update(batch_size=2, one_caption_per_image=True, loss_reduction="sum",
                                   source_generator=True, optimizer_implementation="keras_legacy",
                                   validation_captions_per_image=5)
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
            self.assertEqual(state["optimizer_implementation"], "keras_legacy")
            self.assertEqual(state["history"][0]["training_batches"], 2)
            self.assertEqual(state["history"][0]["validation_queries"], 15)
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
