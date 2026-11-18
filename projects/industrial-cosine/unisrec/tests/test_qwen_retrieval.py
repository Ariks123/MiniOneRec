"""Focused item-ID, SID and exact-search handoff regressions; no training."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_retrieval import (MiniOneRecDrafter, _assert_training_checkpoint,
                            export_index, read_catalog, sha256_file)


def catalog_rows():
    return [
        {"embedding_row": 0, "item_id": "0", "asin": "ASIN0",
         "sid_tokens": ["<a_1>", "<b_2>"], "sid": "<a_1><b_2>"},
        {"embedding_row": 1, "item_id": "7", "asin": "ASIN7",
         "sid_tokens": ["<a_3>", "<b_4>"], "sid": "<a_3><b_4>"},
        {"embedding_row": 2, "item_id": "2", "asin": "ASIN2",
         "sid_tokens": ["<a_5>", "<b_6>"], "sid": "<a_5><b_6>"},
    ]


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "catalog.json"

    def write(self, catalog):
        self.path.write_text(json.dumps(catalog), encoding="utf-8")

    def test_original_zero_and_embedding_row_order_are_preserved(self):
        self.write(catalog_rows()[::-1])
        self.assertEqual([item["item_id"] for item in read_catalog(self.path)], ["0", "7", "2"])

    def test_industrial_null_asin_and_colliding_sid_preserve_item_ids(self):
        rows = catalog_rows()
        rows[0]["asin"] = None
        rows[1]["asin"] = None
        rows[1]["sid_tokens"] = rows[0]["sid_tokens"]
        rows[1]["sid"] = rows[0]["sid"]
        self.write(rows)
        loaded = read_catalog(self.path)
        self.assertEqual([item["item_id"] for item in loaded], ["0", "7", "2"])
        self.assertIsNone(loaded[0]["asin"])
        self.assertEqual(loaded[0]["sid"], loaded[1]["sid"])

    def test_mismatched_sid_fails(self):
        rows = catalog_rows()
        rows[1]["sid"] = "wrong"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "SID"):
            read_catalog(self.path)

    def test_missing_embedding_row_or_duplicate_item_fails(self):
        for mode in ["row", "item"]:
            rows = catalog_rows()
            if mode == "row":
                rows[1]["embedding_row"] = 8
            else:
                rows[1]["item_id"] = "0"
            self.write(rows)
            with self.assertRaises(ValueError):
                read_catalog(self.path)

    def test_random_preflight_checkpoint_cannot_be_exported(self):
        for checkpoint in [{}, {"epoch": -1}, {"epoch": True}, {"epoch": "0"}]:
            with self.assertRaisesRegex(ValueError, "trained checkpoint"):
                _assert_training_checkpoint(checkpoint)
        _assert_training_checkpoint({"epoch": 0})


HAS_NUMERIC = all(importlib.util.find_spec(name) is not None for name in ["numpy", "torch"])


@unittest.skipUnless(HAS_NUMERIC, "PyTorch and NumPy are required for inference-only tests")
class DrafterTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        self.np, self.torch = np, torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog_path = self.root / "catalog.json"
        self.catalog_path.write_text(json.dumps(catalog_rows()), encoding="utf-8")
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text("hidden_size: 2\n", encoding="utf-8")
        self.checkpoint = self.root / "trained.pth"
        self.checkpoint.write_bytes(b"test fixture, no optimizer or training")

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.max_seq_length = 3
                self.last_sequence = None
                self.last_lengths = None

            def encode_items(self, ids):
                table = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [-1., 0.]], device=ids.device)
                return table[ids]

            def encode_history(self, sequence, lengths):
                if self.training:
                    raise AssertionError("inference must disable dropout/noisy gates")
                self.last_sequence = sequence.detach().cpu().tolist()
                self.last_lengths = lengths.detach().cpu().tolist()
                return torch.tensor([[1., 0.]], device=sequence.device)

        self.model = FakeModel()
        # Original item token '0' deliberately maps to internal ID 2, not PAD.
        self.dataset = types.SimpleNamespace(
            iid_field="item_id", item_num=4,
            field2token_id={"item_id": {"[PAD]": 0, "7": 1, "0": 2, "2": 3}},
        )
        self.config = {"hidden_size": 2, "catalog_path": str(self.catalog_path)}
        runtime = types.ModuleType("qwen_runtime")
        runtime.load_experiment = lambda *args, **kwargs: (self.config, self.dataset, (), self.model)
        runtime.load_trained_checkpoint = lambda *args, **kwargs: {"epoch": 2}
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"qwen_runtime": runtime}).start()
        self.index = export_index(self.config_path, self.checkpoint, self.root / "index", batch_size=2)

    def drafter(self):
        return MiniOneRecDrafter(self.model, self.dataset, self.index, self.checkpoint)

    def test_export_uses_catalog_to_internal_mapping_and_normalized_scores(self):
        vectors = self.np.load(self.index / "item_vectors.npy")
        self.np.testing.assert_array_equal(vectors, [[1., 0.], [0., 1.], [-1., 0.]])
        candidates = self.drafter().draft(["7"], topk=3)
        self.assertEqual([item["item_id"] for item in candidates], ["0", "7", "2"])
        self.assertEqual([item["score"] for item in candidates], [1., 0., -1.])

    def test_repeat_candidates_are_kept_by_default_and_sid_is_handoff(self):
        drafter = self.drafter()
        payload = drafter.verification_payload(["0"], topk=2)
        self.assertEqual(payload["candidates"][0]["item_id"], "0")
        self.assertEqual(payload["candidate_sids"][0], "<a_1><b_2>")
        self.assertEqual(drafter.draft(["0"], topk=2, exclude_seen=True)[0]["item_id"], "7")

    def test_colliding_sid_payload_preserves_distinct_original_item_ids(self):
        rows = catalog_rows()
        rows[0]["asin"] = rows[1]["asin"] = None
        rows[1]["sid_tokens"] = rows[0]["sid_tokens"]
        rows[1]["sid"] = rows[0]["sid"]
        self.catalog_path.write_text(json.dumps(rows), encoding="utf-8")
        index = export_index(self.config_path, self.checkpoint, self.root / "industrial_index")
        drafter = MiniOneRecDrafter(self.model, self.dataset, index, self.checkpoint)
        payload = drafter.verification_payload(["0"], topk=2)
        self.assertEqual([item["item_id"] for item in payload["candidates"]], ["0", "7"])
        self.assertEqual(payload["candidate_sids"], ["<a_1><b_2>", "<a_1><b_2>"])
        self.assertIsNone(payload["candidates"][0]["asin"])

    def test_original_zero_right_padding_and_recent_history(self):
        drafter = self.drafter()
        drafter.draft(["0"])
        self.assertEqual(self.model.last_sequence, [[2, 0, 0]])
        self.assertEqual(self.model.last_lengths, [1])
        drafter.draft(["7", "7", "2", "0"])
        self.assertEqual(self.model.last_sequence, [[1, 3, 2]])
        self.assertEqual(self.model.last_lengths, [3])

    def test_mismatched_checkpoint_and_tampered_index_fail(self):
        other = self.root / "other.pth"
        other.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            MiniOneRecDrafter(self.model, self.dataset, self.index, other)
        with (self.index / "catalog.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.drafter()

    def test_empty_or_unknown_history_is_rejected(self):
        drafter = self.drafter()
        with self.assertRaisesRegex(ValueError, "at least one"):
            drafter.draft([])
        with self.assertRaisesRegex(ValueError, "absent"):
            drafter.draft(["999"])

    def test_existing_index_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            export_index(self.config_path, self.checkpoint, self.index)


if __name__ == "__main__":
    unittest.main()
