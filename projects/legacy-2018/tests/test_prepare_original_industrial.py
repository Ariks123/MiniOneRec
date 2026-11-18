"""Preservation and catalog-integrity checks for original Industrial inputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts import prepare_original_industrial as adapter


class PrepareOriginalIndustrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="original-industrial-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source, self.output, self.canonical = self.base / "original", self.base / "prepared", self.base / "canonical"
        for name in ("index", "info", *adapter.SPLITS):
            (self.source / name).mkdir(parents=True, exist_ok=True)
        self.indices = {"0": ["<a_1>", "<b_2>", "<c_3>"],
                        "1": ["<a_1>", "<b_2>", "<c_3>"],
                        "2": ["<a_4>", "<b_5>", "<c_6>"]}
        self.metadata = {"0": {"title": "Item A"}, "1": {"title": "Item B"}, "2": {"title": "Item C"}}
        self.paths = adapter.source_paths(self.source)
        self.paths["sid_index"].write_text(json.dumps(self.indices), encoding="utf-8")
        self.paths["item_metadata"].write_text(json.dumps(self.metadata), encoding="utf-8")
        self.paths["info"].write_bytes("".join(f"{''.join(tokens)}\t{self.metadata[item]['title']}\t{item}\r\n"
                                               for item, tokens in self.indices.items()).encode())
        self.write_rows("train", [([0], 1), ([0], 1)])
        self.write_rows("valid", [([1], 0)])
        self.write_rows("test", [([0, 1], 2), ([0, 1], 2)])
        self.sid_hash = adapter.sha256(self.paths["sid_index"])

    def write_rows(self, split, examples):
        with self.paths[split].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=adapter.FIELDS)
            writer.writeheader()
            for history, target in examples:
                writer.writerow({"user_id": "A7", "item_id": target,
                                 "history_item_id": repr(history), "item_title": self.metadata[str(target)]["title"],
                                 "history_item_title": repr([self.metadata[str(item)]["title"] for item in history]),
                                 "item_sid": "".join(self.indices[str(target)]),
                                 "history_item_sid": repr(["".join(self.indices[str(item)]) for item in history])})

    def prepare(self):
        return adapter.prepare(self.source, self.output, self.canonical, self.sid_hash)

    def test_byte_preservation_order_duplicates_and_original_sid_collisions(self):
        before = {name: path.read_bytes() for name, path in self.paths.items()}
        manifest = self.prepare()
        self.assertEqual(manifest["rows"], {"train": 2, "valid": 1, "test": 2})
        self.assertTrue(manifest["original_sid"])
        self.assertIsNone(manifest["selected_sid_epoch"])
        self.assertFalse(manifest["complete_sids_unique"])
        self.assertEqual(manifest["complete_sid_tokens"], 3)
        self.assertEqual(manifest["unique_sids"], 2)
        self.assertEqual(manifest["sid_collisions"], {"groups": 1, "items": 2, "excess_items": 1})
        self.assertEqual(manifest["source_audit"]["splits"]["train"]["duplicate_full_rows"], 1)
        self.assertEqual(manifest["source_audit"]["cold_test_rows"], 2)
        self.assertEqual(manifest["source_audit"]["cold_test_unique_items"], 1)
        for split in adapter.SPLITS:
            self.assertEqual((self.output / f"{split}.csv").read_bytes(), before[split])
        self.assertEqual((self.output / "info.txt").read_bytes(), before["info"])
        self.assertEqual((self.canonical / f"{adapter.DATASET}.index.json").read_bytes(), before["sid_index"])
        for name, path in self.paths.items():
            self.assertEqual(path.read_bytes(), before[name])
        lines = (self.canonical / f"{adapter.DATASET}.train.inter").read_text().splitlines()
        self.assertEqual(lines[1:], ["A7\t0\t1", "A7\t0\t1"])
        cleaning = json.loads((self.canonical / "cleaning_report.json").read_text())
        self.assertEqual(cleaning["split_boundaries"], {})
        self.assertIn("do not contain timestamps", cleaning["notes"][0])

    def test_unapproved_sid_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "SID SHA-256 mismatch"):
            adapter.prepare(self.source, self.output, self.canonical, "0" * 64)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.canonical.exists())

    def test_bad_history_title_is_rejected_before_writing(self):
        path = self.paths["test"]
        text = path.read_text()
        path.write_text(text.replace("['Item A', 'Item B']", "['Wrong', 'Item B']"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "history title mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_bad_target_sid_and_catalog_mapping_are_rejected(self):
        path = self.paths["test"]
        text = path.read_text()
        path.write_text(text.replace("<a_4><b_5><c_6>", "<a_9><b_9><c_9>"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "target SID mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_nonempty_destination_is_never_overwritten(self):
        self.output.mkdir()
        retained = self.output / "retain.txt"
        retained.write_text("original")
        with self.assertRaisesRegex(ValueError, "nonempty directory"):
            self.prepare()
        self.assertEqual(retained.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
