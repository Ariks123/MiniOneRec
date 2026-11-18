"""Stratified metric, row-weighting and provenance tests for behavior shift."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import analyze_sft_behavior_shift as diagnostics


TOP_K = (1, 3, 5, 10, 20, 50)


class SFTBehaviorShiftTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-behavior-shift-test-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.data = base / "prepared"
        self.original = base / "original"
        self.run = base / "run"
        for path in (self.data, self.original, self.run / "evaluation"):
            path.mkdir(parents=True)
        self.catalog = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>", "<d_0>"] for i in range(4)}
        self.sids = {item: "".join(tokens) for item, tokens in self.catalog.items()}
        self.metadata = {str(i): {"title": title} for i, title in
                         enumerate(("history-only", "popular", "same title", "same title"))}
        self.train = [("1", ["0"], "1"), ("1", ["1"], "1")]
        self.test = [("1", ["0"], "0"), ("1", ["1"], "2"),
                     ("1", ["1"], "2"), ("1", ["0", "1"], "1"),
                     ("1", ["0", "1"], "0")]
        orders = [("0", "1", "2", "3"), ("1", "0", "3", "2"),
                  ("1", "2", "0", "3"), ("2", "1", "0", "3"),
                  ("0", "1", "2", "3")]
        self.predictions = [{"row_id": i, "target": self.sids[target],
                             "predictions": [self.sids[item] for item in order]}
                            for i, ((_, _, target), order) in enumerate(zip(self.test, orders))]
        self.save_all()

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def save_predictions(self):
        (self.run / "evaluation/predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.predictions), encoding="utf-8")

    def save_all(self):
        sid_path = self.original / "Video_Games.index.json"
        meta_path = self.original / "Video_Games.item.json"
        self.write_json(sid_path, self.catalog)
        self.write_json(meta_path, self.metadata)
        for name, rows in (("train", self.train), ("test", self.test)):
            with (self.original / f"Video_Games.{name}.inter").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["user_id:token", "item_id_list:token_seq", "item_id:token"])
                writer.writerows((user, " ".join(history), target) for user, history, target in rows)
            with (self.data / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid"])
                writer.writeheader()
                writer.writerows({"item_id": target, "item_sid": self.sids[target]} for _, _, target in rows)
        self.boundaries = {"train": {"first_target_utc": "2020-01-01T00:00:00Z", "last_target_utc": "2020-01-02T00:00:00Z"},
                           "test": {"first_target_utc": "2020-02-01T00:00:00Z", "last_target_utc": "2020-02-03T00:00:00Z"}}
        self.write_json(self.original / "cleaning_report.json", {"split_boundaries": self.boundaries})
        hashes = {"train_file": self.digest(self.data / "train.csv"),
                  "test_file": self.digest(self.data / "test.csv"),
                  "sid_index_path": self.digest(sid_path)}
        self.manifest = {
            "dataset": "Video_Games", "validation": "passed", "check_only": False,
            "items": len(self.catalog), "rows": {"train": len(self.train), "test": len(self.test)},
            "sources": {"sid_index": {"filename": sid_path.name, "sha256": hashes["sid_index_path"]},
                        "item_metadata": {"filename": meta_path.name, "sha256": self.digest(meta_path)},
                        **{name: {"filename": f"Video_Games.{name}.inter",
                                  "sha256": self.digest(self.original / f"Video_Games.{name}.inter")}
                           for name in ("train", "test")}},
            "outputs": {f"{name}.csv": {"filename": f"{name}.csv", "sha256": hashes[f"{name}_file"]}
                        for name in ("train", "test")},
        }
        self.write_json(self.data / "preparation_manifest.json", self.manifest)
        self.run_config = {"data": {"category": "Video_Games", "train_file": str(self.data / "train.csv"),
                                    "sha256": {**hashes, "item_meta_path": self.digest(meta_path)}}}
        self.write_json(self.run / "run_config.json", self.run_config)
        metrics = {}
        for k in TOP_K:
            ranks = [row["predictions"].index(row["target"]) + 1 for row in self.predictions]
            metrics[f"HR@{k}"] = sum(rank <= k for rank in ranks) / len(ranks)
            metrics[f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= k) / len(ranks)
        self.evaluation = {
            "status": "passed", "category": "Video_Games", "source_sha256": hashes,
            "sid_index_sha256": hashes["sid_index_path"], "item_meta_sha256": self.digest(meta_path),
            "expected_rows": len(self.test), "completed_rows": len(self.test),
            "coverage": {"missing_rows": 0, "duplicate_rows": 0}, "num_beams": 4,
            "metrics": metrics,
            "check_results": {name: "passed" for name in ("provenance", "prepared_input_hashes", "checkpoint_reload",
                "tokenizer_atomic_sid_tokens", "model_vocabulary", "prediction_coverage", "valid_sids", "unique_candidates")},
        }
        self.write_json(self.run / "evaluation/metrics.json", self.evaluation)
        self.save_predictions()
        train_counts = Counter(target for _, _, target in self.train)
        popularity_ids = sorted(self.catalog, key=lambda item: (-train_counts[item], self.sids[item]))
        self.baselines = {
            "status": "passed", "dataset": "Video_Games", "source_sha256": hashes,
            "catalog_items": len(self.catalog), "train_rows": len(self.train), "test_rows": len(self.test),
            "preparation_manifest_sha256": self.digest(self.data / "preparation_manifest.json"),
            "popularity": {"top_sids": [self.sids[item] for item in popularity_ids]},
        }
        self.write_json(self.run / "comparison_baselines.json", self.baselines)

    def analyze(self):
        return diagnostics.analyze(self.data, self.original, self.run)

    def test_behavior_union_duplicate_weighting_and_exact_stratified_metrics(self):
        result = self.analyze()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["overall_test_rows"], 5)
        self.assertEqual(result["counts"]["training_behavior_items"], 2)
        self.assertEqual(result["counts"]["test_target_unique_items"], 3)
        self.assertEqual(result["counts"]["cold_test_unique_items"], 1)
        shift = result["shift_summary"]
        self.assertEqual(shift["test_behavior_cold_rows"], 2)
        self.assertEqual(shift["test_behavior_cold_rate"], .4)
        self.assertEqual(shift["train_repeat_last_rows"], 1)
        self.assertEqual(shift["train_repeat_last_rate"], .5)
        self.assertEqual(shift["test_repeat_last_rows"], 2)
        self.assertEqual(shift["test_repeat_last_rate"], .4)
        strata = {row["key"]: row for row in result["strata"]}
        self.assertEqual(strata["behavior_warm"]["rows"], 3)
        self.assertEqual(strata["behavior_cold"]["rows"], 2)
        self.assertEqual(strata["repeat_last"]["rows"], 2)
        self.assertEqual(strata["repeat_any"]["rows"], 3)
        self.assertEqual(strata["not_repeat_any"]["rows"], 2)
        self.assertEqual(strata["overall"]["metrics"]["HR@1"], .4)
        self.assertEqual(strata["overall"]["metrics"]["HR@3"], .8)
        self.assertEqual(strata["overall"]["hits_by_k"]["3"], 4)
        self.assertEqual(strata["behavior_cold"]["metrics"]["HR@3"], .5)
        self.assertEqual(strata["behavior_cold"]["hits_by_k"]["3"], 1)
        self.assertAlmostEqual(strata["behavior_cold"]["metrics"]["NDCG@3"], 1 / math.log2(3) / 2)
        self.assertEqual(strata["behavior_warm"]["metrics"]["HR@1"], 2 / 3)
        self.assertEqual(strata["behavior_cold"]["model_top1_equals_last_count"], 2)
        self.assertEqual(strata["behavior_cold"]["model_top1_equals_last_rate"], 1)
        self.assertEqual(strata["overall"]["model_top1_equals_last_count"], 3)
        self.assertEqual(strata["overall"]["model_top1_copy_correct_count"], 1)
        self.assertEqual(strata["overall"]["model_top1_copy_wrong_count"], 2)
        self.assertEqual(strata["overall"]["model_top1_correct_count"], 2)
        self.assertAlmostEqual(strata["overall"]["model_top1_copy_accuracy"], 1 / 3)
        self.assertEqual(strata["behavior_cold"]["model_top1_copy_correct_count"], 0)
        self.assertEqual(strata["behavior_cold"]["model_top1_copy_wrong_count"], 2)
        self.assertEqual(strata["overall"]["popularity"]["HR@1"], .2)
        self.assertEqual(strata["behavior_cold"]["uniform_random"]["HR@1"], .25)
        self.assertEqual(result["temporal_boundaries"], self.boundaries)

    def test_reverse_title_collision_does_not_remove_sid_to_title_coverage(self):
        alignment = self.analyze()["alignment_coverage"]
        self.assertEqual(alignment["cold_unique_items"], 1)
        self.assertEqual(alignment["sid_to_title_items"], 1)
        self.assertEqual(alignment["title_to_sid_items"], 0)

    def test_source_change_is_rejected(self):
        with (self.data / "test.csv").open("a") as stream:
            stream.write("0,wrong\n")
        with self.assertRaises(ValueError):
            self.analyze()

    def test_duplicate_or_missing_prediction_rows_are_rejected(self):
        for changed in (self.predictions[:-1], self.predictions[:-1] + [self.predictions[0]]):
            with self.subTest(rows=[row["row_id"] for row in changed]):
                self.predictions = changed
                self.save_predictions()
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_test_order_mismatch_rejected_even_with_new_csv_hash(self):
        path = self.data / "test.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0], rows[1] = rows[1], rows[0]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid"])
            writer.writeheader()
            writer.writerows(rows)
        new_hash = self.digest(path)
        self.manifest["outputs"]["test.csv"]["sha256"] = new_hash
        self.write_json(self.data / "preparation_manifest.json", self.manifest)
        for document in (self.run_config["data"]["sha256"], self.evaluation["source_sha256"],
                         self.baselines["source_sha256"]):
            document["test_file"] = new_hash
        self.baselines["preparation_manifest_sha256"] = self.digest(self.data / "preparation_manifest.json")
        self.write_json(self.run / "run_config.json", self.run_config)
        self.write_json(self.run / "evaluation/metrics.json", self.evaluation)
        self.write_json(self.run / "comparison_baselines.json", self.baselines)
        with self.assertRaises(ValueError):
            self.analyze()

    def test_unaccepted_evaluation_is_rejected(self):
        self.evaluation["status"] = "running"
        self.write_json(self.run / "evaluation/metrics.json", self.evaluation)
        with self.assertRaises(ValueError):
            self.analyze()

    def test_predictions_must_reproduce_accepted_overall_metrics(self):
        self.evaluation["metrics"]["HR@1"] += .1
        self.write_json(self.run / "evaluation/metrics.json", self.evaluation)
        with self.assertRaisesRegex(ValueError, "Recomputed HR@1"):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
