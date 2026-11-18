"""Known-vector ranking and provenance checks for the content baseline."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


HAS_EMBEDDING_STACK = all(importlib.util.find_spec(module) is not None for module in ("numpy", "torch"))


@unittest.skipUnless(HAS_EMBEDDING_STACK, "numpy and torch are required for content baseline tests")
class SFTContentBaselineTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scripts import evaluate_sft_content_baseline as content
        self.np, self.content = np, content
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-content-baseline-test-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.data, self.original, self.run = base / "prepared", base / "original", base / "run"
        for path in (self.data, self.original, self.run):
            path.mkdir()
        self.embeddings = base / "embeddings.npy"
        self.vectors = np.array([[2, 0], [1, 0], [0, 3], [-1, 0]], dtype=np.float32)
        self.catalog = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>", "<d_0>"] for i in range(4)}
        self.sids = {item: "".join(tokens) for item, tokens in self.catalog.items()}
        self.train = [("1", ["0"], "1"), ("1", ["1"], "1")]
        self.valid = [("1", ["0"], "0"), ("1", ["2"], "2"),
                      ("1", ["2"], "3"), ("1", ["2"], "3"),
                      ("1", ["0", "2"], "1")]
        self.save_all()

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def save_all(self):
        self.np.save(self.embeddings, self.vectors)
        self.write_json(self.original / "Video_Games.index.json", self.catalog)
        metadata = {str(i): {"title": f"item {i}"} for i in range(4)}
        self.write_json(self.original / "Video_Games.item.json", metadata)
        for split, rows in (("train", self.train), ("valid", self.valid)):
            with (self.original / f"Video_Games.{split}.inter").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["user_id:token", "item_id_list:token_seq", "item_id:token"])
                writer.writerows((user, " ".join(history), target) for user, history, target in rows)
            with (self.data / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid"])
                writer.writeheader()
                writer.writerows({"item_id": target, "item_sid": self.sids[target]} for _, _, target in rows)
        sid_hash = self.digest(self.original / "Video_Games.index.json")
        meta_hash = self.digest(self.original / "Video_Games.item.json")
        self.manifest = {
            "dataset": "Video_Games", "validation": "passed", "check_only": False,
            "items": 4, "rows": {"train": len(self.train), "valid": len(self.valid)},
            "sources": {"sid_index": {"sha256": sid_hash}, "item_metadata": {"sha256": meta_hash},
                        **{split: {"sha256": self.digest(self.original / f"Video_Games.{split}.inter")}
                           for split in ("train", "valid")}},
            "outputs": {f"{split}.csv": {"sha256": self.digest(self.data / f"{split}.csv")}
                        for split in ("train", "valid")},
        }
        self.write_json(self.data / "preparation_manifest.json", self.manifest)
        self.run_config = {"data": {"category": "Video_Games", "train_file": str(self.data / "train.csv"),
                                    "eval_file": str(self.data / "valid.csv"),
                                    "sha256": {"train_file": self.digest(self.data / "train.csv"),
                                               "eval_file": self.digest(self.data / "valid.csv"),
                                               "sid_index_path": sid_hash, "item_meta_path": meta_hash}}}
        self.write_json(self.run / "run_config.json", self.run_config)
        self.embedding_manifest = {
            "dataset": "Video_Games", "shape": [4, 2], "dtype": "float32",
            "item_metadata_sha256": meta_hash, "embedding_sha256": self.digest(self.embeddings),
            "row_order": "contiguous item_id ascending from zero", "validation": "passed",
        }
        self.write_json(str(self.embeddings) + ".manifest.json", self.embedding_manifest)
        self.write_json(str(self.embeddings) + ".ids.json", list(range(4)))

    def evaluate(self, batch_size=2):
        return self.content.evaluate(self.embeddings, self.original, self.data, self.run,
                                     split="valid", device="cpu", batch_size=batch_size)

    def test_normalize_before_mean_stable_ties_and_no_seen_filter(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "passed")
        strata = {row["key"]: row for row in result["strata"]}
        overall = strata["overall"]
        # Ranks are 1, 1, 4, 4, 2. The final rank is 2 only when item
        # vectors are normalized before pooling and ties prefer lower IDs.
        self.assertEqual(overall["rows"], 5)
        self.assertEqual(overall["hits_by_k"]["1"], 2)
        self.assertEqual(overall["hits_by_k"]["3"], 3)
        self.assertEqual(overall["metrics"]["HR@1"], .4)
        self.assertEqual(overall["metrics"]["HR@3"], .6)
        self.assertAlmostEqual(overall["metrics"]["NDCG@3"], (2 + 1 / math.log2(3)) / 5)
        self.assertEqual(overall["metrics"]["HR@50"], 1)
        self.assertEqual(strata["behavior_warm"]["rows"], 2)
        self.assertEqual(strata["behavior_cold"]["rows"], 3)
        self.assertAlmostEqual(strata["behavior_cold"]["metrics"]["HR@3"], 1 / 3)
        self.assertEqual(strata["behavior_cold"]["hits_by_k"]["3"], 1)

    def test_batch_size_does_not_change_scores(self):
        small = self.evaluate(batch_size=1)
        large = self.evaluate(batch_size=5)
        small_strata = {row["key"]: row for row in small["strata"]}
        large_strata = {row["key"]: row for row in large["strata"]}
        for key in small_strata:
            self.assertEqual(small_strata[key]["hits_by_k"], large_strata[key]["hits_by_k"])
            self.assertEqual(small_strata[key]["metrics"], large_strata[key]["metrics"])

    def test_stale_embedding_hash_is_rejected(self):
        self.np.save(self.embeddings, self.vectors * 2)
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_altered_row_ids_are_rejected(self):
        self.write_json(str(self.embeddings) + ".ids.json", [1, 0, 2, 3])
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_zero_norm_catalog_item_is_rejected(self):
        self.vectors[3] = [0, 0]
        self.save_all()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_zero_norm_pooled_history_is_rejected(self):
        self.valid[0] = ("1", ["0", "3"], "0")
        self.save_all()
        with self.assertRaises(ValueError):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
