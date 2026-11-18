"""Protocol and arithmetic checks for the paired Industrial title-history report."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import compare_industrial_title_history as comparison
from scripts import report_industrial_sft as reports


class TitleHistoryComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-title-comparison-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.inputs = self.root / "data", self.root / "inputs"
        self.baseline, self.augmented = self.root / "baseline", self.root / "augmented"
        self.data.mkdir()
        self.inputs.mkdir()
        self.indices = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(60)}
        self.indices["59"] = self.indices["58"][:]
        self.catalog = {key: "".join(tokens) for key, tokens in self.indices.items()}
        self.save(self.inputs / f"{reports.DATASET}.index.json", self.indices)
        self.save(self.inputs / f"{reports.DATASET}.item.json", {key: {"title": "item " + key} for key in self.indices})
        for name, examples in {"train": [([0], 1), ([1], 58)], "valid": [([0], 1)],
                               "test": [([0], 59), ([1], 2), ([0], 1)]}.items():
            with (self.data / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("item_id", "history_item_id", "item_sid"))
                writer.writeheader()
                for history, target in examples:
                    writer.writerow({"item_id": target, "history_item_id": repr(history), "item_sid": self.catalog[str(target)]})
        paths = {"train_file": self.data / "train.csv", "eval_file": self.data / "valid.csv",
                 "test_file": self.data / "test.csv", "sid_index_path": self.inputs / f"{reports.DATASET}.index.json",
                 "item_meta_path": self.inputs / f"{reports.DATASET}.item.json"}
        self.hashes = {key: reports.sha256(path) for key, path in paths.items()}
        self.save(self.data / "preparation_manifest.json", {
            "dataset": reports.DATASET, "validation": "passed", "original_sid": True,
            "rows": {"train": 2, "valid": 1, "test": 3},
            "outputs": {name: {"sha256": self.hashes[key]} for name, key in (
                ("train.csv", "train_file"), ("valid.csv", "eval_file"), ("test.csv", "test_file"))},
            "sources": {"sid_index": {"sha256": self.hashes["sid_index_path"]},
                        "item_metadata": {"sha256": self.hashes["item_meta_path"]}}})
        self.make_run(self.baseline, False)
        self.make_run(self.augmented, True)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def edit(self, run, filename, change):
        path = run / filename
        value = json.loads(path.read_text())
        change(value)
        self.save(path, value)

    def make_run(self, run, augmented):
        tasks = {"sid_sequence_recommendation": 2, "history_to_title": 2, "sid2title": 59, "title2sid": 60}
        if augmented:
            tasks["title_history_to_sid"] = 2
        self.save(run / "run_config.json", {
            "stage": "sft", "resume_from_checkpoint": None,
            "model": {"source": str(self.root / "weights/Qwen2.5-0.5B"), "train_from_scratch": False,
                      "identity": {"sha256": "a" * 64}},
            "data": {"category": reports.DATASET, "sample": -1, "include_title_history": augmented,
                     "sha256": self.hashes, "train_samples": sum(tasks.values()), "validation_samples": 1},
            "task_counts": tasks,
            "hyperparameters": {"world_size": 4, "micro_batch_size": 8, "gradient_accumulation_steps": 2,
                "global_batch_size": 64, "max_epochs": 10, "learning_rate": .0003, "warmup_steps": 20,
                "lr_scheduler_type": "cosine", "cutoff_len": 512, "seed": 42,
                "early_stopping_patience": 3, "eval_save_fraction": .05, "bf16": True}})
        final = str(run / "final_checkpoint")
        self.save(run / "training_summary.json", {
            "status": "completed", "final_checkpoint_contains_best_model": True,
            "best_checkpoint": str(run / "checkpoint-7"), "best_metric": .9,
            "final_checkpoint": final, "global_step": 8, "planned_max_steps": 10,
            "epoch": 8.0, "stop_reason": "early_stopping"})
        unique = list(dict.fromkeys(self.catalog.values()))
        predictions, ranks = [], []
        for row_id, target in enumerate((59, 2, 1)):
            sid = self.catalog[str(target)]
            hit = augmented or row_id == 2
            candidates = ([sid] if hit else []) + [item for item in unique if item != sid]
            predictions.append({"row_id": row_id, "target": sid, "predictions": candidates[:50],
                                "prompt_style": "original-eval"})
            ranks.append(1 if hit else math.inf)
        (run / "evaluation").mkdir()
        (run / "evaluation/predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        metrics = {f"{metric}@{k}": sum((1 if metric == "HR" else 1 / math.log2(rank + 1))
                    for rank in ranks if rank <= k) / 3 for metric in ("HR", "NDCG") for k in reports.TOP_K}
        self.save(run / "evaluation/metrics.json", {
            "status": "passed", "category": reports.DATASET, "scope": "complete_original_test_split",
            "model_path": final, "num_beams": 50, "prompt_style": "original-eval", "validation_prompt_style": "training",
            "source_sha256": {key: self.hashes[key] for key in ("train_file", "test_file", "sid_index_path")},
            "item_meta_sha256": self.hashes["item_meta_path"], "expected_rows": 3, "completed_rows": 3,
            "coverage": {"expected_rows": 3, "completed_rows": 3, "missing_rows": 0, "duplicate_rows": 0},
            "CC": 0, "validity_rate": 1, "unique_rate": 1, "metrics": metrics,
            "check_results": {key: "passed" for key in ("provenance", "prepared_input_hashes", "checkpoint_reload",
                "tokenizer_atomic_sid_tokens", "model_vocabulary", "validation_coverage", "prediction_coverage",
                "valid_sids", "unique_candidates")}})

    def compare(self):
        return comparison.compare_runs(self.baseline, self.augmented, self.data, self.inputs)

    def test_reaggregates_predictions_and_computes_percentage_point_deltas(self):
        result = self.compare()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["protocol"]["test_rows"], 3)
        self.assertEqual(result["baseline"]["train_samples"], 123)
        self.assertEqual(result["augmented"]["train_samples"], 125)
        rows = {(row["stratum"], row["metric"], row["k"]): row for row in result["metrics"]}
        overall = rows[("overall", "Recall", 10)]
        self.assertAlmostEqual(overall["baseline"], 1 / 3)
        self.assertEqual(overall["augmented"], 1)
        self.assertAlmostEqual(overall["difference_percentage_points"], 200 / 3)
        self.assertEqual(rows[("behavior_cold", "Recall", 50)]["rows"], 2)
        self.assertEqual(rows[("behavior_cold", "Recall", 50)]["difference_percentage_points"], 100)
        self.assertEqual(len(rows), len(comparison.GROUPS) * 2 * len(reports.TOP_K))

    def test_rejects_incomplete_training_and_different_model_identity(self):
        self.edit(self.baseline, "training_summary.json", lambda obj: obj.update(status="running"))
        with self.assertRaisesRegex(ValueError, "training is not completed"):
            self.compare()
        self.edit(self.baseline, "training_summary.json", lambda obj: obj.update(status="completed"))
        self.edit(self.augmented, "run_config.json", lambda obj: obj["model"]["identity"].update(sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "different pretrained model bytes"):
            self.compare()

    def test_rejects_wrong_model_or_task_flag(self):
        self.edit(self.baseline, "run_config.json", lambda obj: obj["model"].update(source="weights/Qwen2.5-1.5B"))
        with self.assertRaisesRegex(ValueError, "Qwen2.5-0.5B"):
            self.compare()
        self.edit(self.baseline, "run_config.json", lambda obj: obj["model"].update(source=str(self.root / "weights/Qwen2.5-0.5B")))
        self.edit(self.augmented, "run_config.json", lambda obj: obj["data"].update(include_title_history=False))
        with self.assertRaisesRegex(ValueError, "task flag"):
            self.compare()

    def test_rejects_changed_shared_learning_rate_and_partial_extra_task(self):
        self.edit(self.augmented, "run_config.json", lambda obj: obj["hyperparameters"].update(learning_rate=.0001))
        with self.assertRaisesRegex(ValueError, "hyperparameters differ"):
            self.compare()
        self.edit(self.augmented, "run_config.json", lambda obj: obj["hyperparameters"].update(learning_rate=.0003))
        self.edit(self.augmented, "run_config.json", lambda obj: obj["task_counts"].update(title_history_to_sid=1))
        with self.assertRaisesRegex(ValueError, "every training row"):
            self.compare()

    def test_rejects_stale_predictions_and_mismatched_evaluated_checkpoint(self):
        path = self.augmented / "evaluation/predictions.jsonl"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Predictions incomplete"):
            self.compare()
        path.write_text(original)
        self.edit(self.augmented, "evaluation/metrics.json", lambda obj: obj.update(model_path=str(self.baseline / "final_checkpoint")))
        with self.assertRaisesRegex(ValueError, "evaluated checkpoint"):
            self.compare()

    def test_writes_report_separately_and_does_not_change_training_inputs(self):
        result = self.compare()
        before = (self.baseline / "run_config.json").read_bytes()
        output = self.root / "reports"
        files = comparison.write_comparison(result, output)
        self.assertTrue(all(Path(path).is_file() for path in files.values()))
        saved = json.loads((output / "comparison.json").read_text())
        self.assertEqual(saved["metrics"], result["metrics"])
        with (output / "comparison.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(result["metrics"]))
        self.assertIn("Recall@10", (output / "index.html").read_text())
        self.assertEqual(before, (self.baseline / "run_config.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "separate from"):
            comparison.write_comparison(result, self.baseline / "comparison")


if __name__ == "__main__":
    unittest.main()
