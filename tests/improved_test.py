"""Focused model extension checks before the full COCO experiment."""
from pathlib import Path
from contextlib import redirect_stdout
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.evaluate import load_neural_checkpoint, calculate_ranks
from src.improved_model import BidirectionalRetrievalModel, ValidationEnsemble, contrastive_retrieval_loss
from src.improve_select import choose_weight, definition
from src.model import RetrievalModel, similarity_matrix
from src.improve_run import exclusive_run, run
from src.source_evaluation import source_ranks
from tests.smoke_test import create_fixture


class ImprovementTests(unittest.TestCase):
    def test_ensemble_distance_and_saved_encoder_match(self):
        torch.manual_seed(21)
        original = RetrievalModel(12, 8, 4, 6, normalize_embeddings=True).eval()
        candidate = BidirectionalRetrievalModel(12, 8, 4, 6, hidden_dim=3).eval()
        ensemble = ValidationEnsemble(original, candidate, 0.25).eval()
        features, tokens, lengths = torch.randn(3, 8), torch.tensor([[2, 3], [4, 5], [6, 7]]), torch.tensor([2, 2, 2])
        oi, oc = original(features, tokens, lengths)
        ci, cc = candidate(features, tokens, lengths)
        images, captions = ensemble(features, tokens, lengths)
        torch.testing.assert_close(similarity_matrix(images, captions),
                                   0.25 * similarity_matrix(oi, oc) + 0.75 * similarity_matrix(ci, cc))
        base = {"vocab": {str(i): i for i in range(12)}, "image_feature_dim": 8, "word_dim": 4,
                "embedding_dim": 6, "recurrent_type": "gru", "model_options": {"normalize_embeddings": True}}
        new = {**base, "model_type": "bidirectional_gru", "model_options": {"hidden_dim": 3, "dropout": 0.1}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            torch.save({**base, "model_type": "validation_ensemble", "variant": "ensemble_glove",
                        "embedding_dim": 12,
                        "model": ensemble.state_dict(), "original_definition": definition(base),
                        "candidate_definition": definition(new), "original_weight": 0.25}, path)
            restored, _, variant = load_neural_checkpoint(path, torch.device("cpu"))
            self.assertEqual(variant, "ensemble_glove")
            ri, rc = restored(features, tokens, lengths)
            torch.testing.assert_close(ri, images)
            torch.testing.assert_close(rc, captions)
        selected, all_candidates = choose_weight(torch.eye(3), torch.eye(3),
                                                  torch.eye(3), torch.eye(3).roll(1, 0),
                                                  [0, 1, 2], ["a", "b", "c"])
        self.assertEqual(len(all_candidates), 5)
        self.assertEqual(selected["r_at_1"], 100.0)
        self.assertGreater(selected["original_weight"], 0.5)

    def test_padding_batch_order_and_checkpoint_restore(self):
        torch.manual_seed(7)
        model = BidirectionalRetrievalModel(12, 8, 4, 6, hidden_dim=3, dropout=0.1).eval()
        tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
        lengths = torch.tensor([2, 3])
        with torch.no_grad():
            expected = model.encode_captions(tokens, lengths)
            padded = torch.nn.functional.pad(tokens, (0, 5))
            torch.testing.assert_close(model.encode_captions(padded, lengths), expected)
            torch.testing.assert_close(model.encode_captions(tokens.flip(0), lengths.flip(0)), expected.flip(0))
            for i, length in enumerate(lengths):
                torch.testing.assert_close(model.encode_captions(tokens[i:i+1, :length], lengths[i:i+1]), expected[i:i+1])
            torch.testing.assert_close(expected.norm(dim=1), torch.ones(2))
        vocab = {str(i): i for i in range(12)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_type": "bidirectional_gru", "model": model.state_dict(), "variant": "bigru_glove",
                        "vocab": vocab, "image_feature_dim": 8, "word_dim": 4, "embedding_dim": 6,
                        "model_options": {"hidden_dim": 3, "dropout": 0.1}}, path)
            restored, loaded_vocab, variant = load_neural_checkpoint(path, torch.device("cpu"))
            self.assertEqual(loaded_vocab, vocab)
            self.assertEqual(variant, "bigru_glove")
            torch.testing.assert_close(restored.encode_captions(tokens, lengths), expected)

    def test_loss_alignment_gradient_and_duplicate_positives(self):
        aligned = torch.eye(3, requires_grad=True)
        ids = torch.arange(3)
        good = contrastive_retrieval_loss(aligned, aligned, ids)
        bad = contrastive_retrieval_loss(aligned, aligned.roll(1, 0), ids)
        self.assertLess(float(good.detach()), float(bad.detach()))
        bad.backward()
        self.assertTrue(bool(torch.isfinite(aligned.grad).all()))
        copies = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
        no_negative = contrastive_retrieval_loss(copies, copies, torch.tensor([5, 5]))
        torch.testing.assert_close(no_negative, torch.tensor(0.))
        no_negative.backward()
        self.assertTrue(bool(torch.isfinite(copies.grad).all()))
        logits = torch.eye(3) / 0.07
        expected = torch.nn.functional.cross_entropy(logits, ids)
        torch.testing.assert_close(contrastive_retrieval_loss(torch.eye(3), torch.eye(3), ids, hard_weight=0), expected)

    def test_independent_ranks_and_exclusive_lock(self):
        torch.manual_seed(11)
        images = torch.nn.functional.normalize(torch.randn(7, 9), dim=1)
        captions = torch.nn.functional.normalize(torch.randn(14, 9), dim=1)
        targets = [i // 2 for i in range(14)]
        self.assertEqual(calculate_ranks(images, captions, targets),
                         source_ranks(images, captions, targets, torch.device("cpu")))
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "run.lock"
            with exclusive_run(lock_path):
                if __import__("os").name == "nt":
                    with self.assertRaisesRegex(RuntimeError, "already active"):
                        with exclusive_run(lock_path):
                            self.fail("Second writer acquired the lock")
            with exclusive_run(lock_path):
                pass

    def test_resume_preserves_full_training_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            configs = []
            for name in ("continuous", "resumed"):
                cfg = create_fixture(base / name)
                cfg["model"].update(hidden_dim=3, dropout=0.1)
                cfg["training"].update(
                    batch_size=2, epochs=3, minimum_epochs=3, patience=3,
                    learning_rate=0.0002, weight_decay=0.0001,
                    learning_rate_milestones=[1], learning_rate_gamma=0.5,
                    gradient_clip=2.0, temperature=0.07, margin=0.2,
                    hard_negative_weight=0.25, hard_negative_warmup_epochs=1,
                    validation_images=3, validation_captions_per_image=5,
                    checkpoint_dir="checkpoints/improved")
                cfg["evaluation"] = {"gallery_images": 3, "metrics_dir": "outputs/improved"}
                configs.append(cfg)
            with patch("src.improve_run.device_for_run", return_value=torch.device("cpu")), \
                 patch("src.retrieve.device_for_run", return_value=torch.device("cpu")), \
                 redirect_stdout(io.StringIO()):
                run(configs[0])
                run(configs[1], stop_after_epoch=1)
                status_path = base / "resumed/outputs/improved/status.json"
                self.assertEqual(json.loads(status_path.read_text())["stage"], "saved_for_resume")
                changed = copy.deepcopy(configs[1])
                changed["training"]["learning_rate"] = 0.001
                with self.assertRaisesRegex(ValueError, "configuration"):
                    run(changed, resume=True)
                run(configs[1], resume=True)
            states = [torch.load(base / name / "checkpoints/improved/bigru_glove_last.pt",
                                 map_location="cpu", weights_only=False) for name in ("continuous", "resumed")]
            self.assertEqual(states[0]["epoch"], 3)
            for key, value in states[0]["model"].items():
                torch.testing.assert_close(value, states[1]["model"][key], rtol=0, atol=0)
            self.assertEqual(states[0]["best_key"], states[1]["best_key"])
            self.assertEqual(states[0]["optimizer"]["param_groups"][0]["lr"], 0.0001)
            status = json.loads(status_path.read_text())
            self.assertEqual(status["stage"], "complete")
            self.assertEqual(status["measured_results"]["test_15queries"]["queries"], 15)
            self.assertEqual(status["measured_results"]["test_3queries"]["queries"], 3)


if __name__ == "__main__":
    unittest.main()
