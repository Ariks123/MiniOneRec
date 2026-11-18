"""Numerical and provenance checks for same-split recommendation baselines."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import build_sft_baselines as baselines


class SFTBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-baseline-test-")
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.sid_index = self.data / "Video_Games.index.json"
        self.catalog = {str(i): [f"<a_{a}>", "<b_1>", "<c_1>", "<d_0>"]
                        for i, a in enumerate((2, 10, 3))}
        self.train = ["0", "1", "0", "1"]
        self.test = ["1", "1", "2", "0"]
        self.save()

    def save(self):
        self.sid_index.write_text(json.dumps(self.catalog), encoding="utf-8")
        for name, items in (("train", self.train), ("test", self.test)):
            with (self.data / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid"])
                writer.writeheader()
                for item in items:
                    writer.writerow({"item_id": item, "item_sid": "".join(self.catalog[item])})
        self.manifest = {
            "dataset": "Video_Games", "validation": "passed", "items": len(self.catalog),
            "rows": {"train": len(self.train), "test": len(self.test)},
            "sources": {"sid_index": {"sha256": baselines.sha256(self.sid_index)}},
            "expected_sid_sha256": baselines.sha256(self.sid_index),
            "outputs": {f"{name}.csv": {"sha256": baselines.sha256(self.data / f"{name}.csv")}
                        for name in ("train", "test")},
        }
        self.save_manifest()

    def save_manifest(self):
        (self.data / "preparation_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def build(self):
        return baselines.build_baselines(self.data, self.sid_index)

    def test_popularity_ties_are_lexical_and_duplicate_rows_keep_weight(self):
        result = self.build()
        # The tied SID containing a_10 sorts before the SID containing a_2.
        ordered = ["".join(self.catalog[item]) for item in ("1", "0", "2")]
        self.assertEqual(result["popularity"]["top_sids"], ordered)
        metrics = result["popularity"]["metrics"]
        self.assertEqual(metrics["HR@1"], .5)
        self.assertEqual(metrics["NDCG@1"], .5)
        self.assertEqual(metrics["HR@3"], 1)
        self.assertAlmostEqual(metrics["NDCG@3"], (2 + 1 / math.log2(3) + .5) / 4)
        self.assertEqual(result["train_rows"], 4)
        self.assertEqual(result["test_rows"], 4)
        self.assertFalse(result["methodology"]["test_row_deduplication"])

    def test_uniform_expectation_caps_k_at_catalog_size(self):
        metrics = self.build()["uniform_random"]["metrics"]
        self.assertEqual(metrics["HR@1"], 1 / 3)
        self.assertEqual(metrics["NDCG@1"], 1 / 3)
        self.assertEqual(metrics["HR@50"], 1)
        self.assertAlmostEqual(metrics["NDCG@3"], (1 + 1 / math.log2(3) + .5) / 3)
        self.assertEqual(metrics["NDCG@50"], metrics["NDCG@3"])

    def test_test_target_changes_never_change_popularity_ranking(self):
        first = self.build()
        self.test = ["2", "2", "2", "2"]
        self.save()
        second = self.build()
        self.assertEqual(first["popularity"]["top_sids"], second["popularity"]["top_sids"])
        self.assertEqual(second["popularity"]["metrics"]["HR@1"], 0)

    def test_changed_source_hash_and_manifest_row_count_are_rejected(self):
        with (self.data / "train.csv").open("a") as stream:
            stream.write("0,wrong\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.build()
        self.save()
        self.manifest["rows"]["test"] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Test row count"):
            self.build()

    def test_duplicate_catalog_sids_and_wrong_targets_are_rejected(self):
        self.catalog["1"] = self.catalog["0"][:]
        self.save()
        with self.assertRaisesRegex(ValueError, "SIDs must be unique"):
            self.build()
        self.catalog["1"][0] = "<a_10>"
        self.save()
        path = self.data / "test.csv"
        path.write_text("item_id,item_sid\n99,<a_0><b_0><c_0><d_0>\n", encoding="utf-8")
        self.manifest["outputs"]["test.csv"]["sha256"] = baselines.sha256(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "target item is not in the catalog"):
            self.build()

    def test_missing_approved_hash_is_rejected(self):
        del self.manifest["outputs"]["test.csv"]["sha256"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "no valid test_file SHA-256"):
            self.build()


if __name__ == "__main__":
    unittest.main()
