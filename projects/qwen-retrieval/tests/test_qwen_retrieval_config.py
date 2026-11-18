"""Catalog identities and SID collisions must survive the retrieval adapter."""
import json
from pathlib import Path
import tempfile
import unittest

from qwen_retrieval.config import build_catalog, checkpoint_identity, json_digest


class RetrievalConfigTests(unittest.TestCase):
    def test_unique_sid_rows_preserve_all_item_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sid.json"
            path.write_text(json.dumps({"10": ["<a_1>", "<b_2>", "<c_3>"],
                                        "2": ["<a_1>", "<b_2>", "<c_3>"],
                                        "3": ["<a_2>", "<b_2>", "<c_3>"]}))
            rows = build_catalog(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["item_ids"], ["2", "10"])
            self.assertEqual(rows[0]["sid"], "<a_1><b_2><c_3>")

    def test_original_three_semantic_tokens_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sid.json"
            for tokens in (["<a_1>", "<b_2>"], ["<a_1>", "<c_2>", "<b_3>"],
                           ["<a_1>", "<b_2>", "<c_3>", "<d_0>"]):
                path.write_text(json.dumps({"1": tokens}))
                with self.assertRaises(ValueError):
                    build_catalog(path)

    def test_index_identity_changes_with_head_weights_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"backbone-and-head-v1")
            (root / "tokenizer.json").write_text("{}")
            first = checkpoint_identity(root)
            (root / "model.safetensors").write_bytes(b"backbone-and-head-v2")
            second = checkpoint_identity(root)
            self.assertNotEqual(first["sha256"], second["sha256"])
            (root / "tokenizer.json").write_text('{"updated":true}')
            self.assertNotEqual(second["sha256"], checkpoint_identity(root)["sha256"])
            self.assertEqual(json_digest({"b": 2, "a": 1}), json_digest({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
