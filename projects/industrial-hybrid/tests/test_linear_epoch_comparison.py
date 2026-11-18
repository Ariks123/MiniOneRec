"""Reaggregate final and actual epoch-3 snapshots without mixing their predictions."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_industrial_linear_epochs as comparison
from scripts import report_industrial_sft as reports


class LinearEpochComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-linear-comparison-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.inputs = self.root / "data", self.root / "inputs"
        self.three, self.five = self.root / "epochs3", self.root / "epochs5"
        self.data.mkdir()
        self.inputs.mkdir()
        self.indices = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(60)}
        self.indices["59"] = self.indices["58"][:]
        self.catalog = {key: "".join(tokens) for key, tokens in self.indices.items()}
        self.save(self.inputs / f"{reports.DATASET}.index.json", self.indices)
        self.save(self.inputs / f"{reports.DATASET}.item.json", {key: {"title": "item " + key} for key in self.indices})
        for split, rows in {"train": [([0], 1), ([1], 58)], "valid": [([0], 1)],
                            "test": [([0], 59), ([1], 2), ([0], 1)]}.items():
            with (self.data / (split + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("item_id", "history_item_id", "item_sid"))
                writer.writeheader()
                for history, target in rows:
                    writer.writerow({"item_id": target, "history_item_id": repr(history),
                                     "item_sid": self.catalog[str(target)]})
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
        self.tasks = {"sid_sequence_recommendation": 2, "history_to_title": 2,
                      "sid2title": 59, "title2sid": 60, "title_history_to_sid": 2}
        smaller_fixture = patch.object(comparison, "EXPECTED_TASK_COUNTS", self.tasks)
        smaller_fixture.start()
        self.addCleanup(smaller_fixture.stop)
        self.make_training(self.three, 3)
        self.make_training(self.five, 5)
        self.make_evaluation(self.three / "evaluation", self.three / "final_checkpoint", {2})
        self.make_evaluation(self.five / "evaluation", self.five / "final_checkpoint", {0, 1, 2})
        snapshot = self.five / "epoch_3_checkpoint"
        self.save(snapshot / "snapshot_metadata.json", {
            "format_version": 1, "snapshot_type": "epoch_end_training_weights",
            "capture_event": "on_epoch_end", "requested_epoch": 3, "epoch": 3.0,
            "global_step": 6, "steps_per_epoch": 2, "checkpoint_path": str(snapshot),
            "best_model_reloaded": False, "created_at_utc": "2026-11-17T00:00:00+00:00"})
        self.make_evaluation(self.five / "evaluation_epoch3", snapshot, {0, 2})

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def edit(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        self.save(path, value)

    def make_training(self, run, maximum):
        self.save(run / "run_config.json", {
            "stage": "sft", "resume_from_checkpoint": None,
            "model": {"source": str(self.root / "weights/Qwen2.5-0.5B"), "train_from_scratch": False,
                      "identity": {"sha256": "a" * 64}},
            "data": {"category": reports.DATASET, "sample": -1, "include_title_history": True,
                     "sha256": self.hashes, "train_samples": sum(self.tasks.values()), "validation_samples": 1},
            "task_counts": self.tasks,
            "hyperparameters": {"world_size": 4, "micro_batch_size": 8, "gradient_accumulation_steps": 2,
                "global_batch_size": 64, "max_epochs": maximum, "learning_rate": .0003, "warmup_steps": 20,
                "lr_scheduler_type": "linear", "cutoff_len": 512, "seed": 42,
                "early_stopping_patience": 3, "eval_save_fraction": .05, "bf16": True,
                "generative_validation": False, "model_selection_metric": "eval_loss",
                "save_epoch_checkpoint": 3 if maximum == 5 else 0, "minimum_epochs": 3 if maximum == 5 else 0}})
        self.save(run / "training_summary.json", {
            "status": "completed", "selection_metric": "eval_loss", "final_checkpoint_contains_best_model": True,
            "best_checkpoint": str(run / f"checkpoint-{2 * maximum - 2}"), "best_metric": 1.5,
            "final_checkpoint": str(run / "final_checkpoint"), "global_step": 2 * maximum,
            "planned_max_steps": 2 * maximum, "epoch": maximum, "stop_reason": "max_epochs"})
        self.save(run / "run_status.json", {"status": "completed", "runtime_image_id": "sha256:" + "b" * 64,
                                               "runtime_versions": {"torch": "2.8.0"},
                                               "source_sha256": {"sft.py": "c" * 64}})
        self.save(run / "trainer_state.json", {"log_history": [{"step": 2 * maximum, "eval_loss": 1.5}]})

    def make_evaluation(self, directory, model_path, hit_rows):
        directory.mkdir(parents=True, exist_ok=True)
        unique = list(dict.fromkeys(self.catalog.values()))
        predictions, ranks = [], []
        for row_id, target in enumerate((59, 2, 1)):
            sid = self.catalog[str(target)]
            hit = row_id in hit_rows
            candidates = ([sid] if hit else []) + [item for item in unique if item != sid]
            predictions.append({"row_id": row_id, "target": sid, "predictions": candidates[:50],
                                "prompt_style": "original-eval"})
            ranks.append(1 if hit else math.inf)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        metrics = {f"{metric}@{k}": sum((1 if metric == "HR" else 1 / math.log2(rank + 1))
                    for rank in ranks if rank <= k) / 3 for metric in ("HR", "NDCG") for k in reports.TOP_K}
        self.save(directory / "metrics.json", {
            "status": "passed", "category": reports.DATASET, "scope": "complete_original_test_split",
            "model_path": str(model_path), "num_beams": 50, "prompt_style": "original-eval",
            "validation_prompt_style": "training", "seed": 42, "cutoff_len": 512, "max_new_tokens": 5, "world_size": 4,
            "source_sha256": {key: self.hashes[key] for key in ("train_file", "test_file", "sid_index_path")},
            "item_meta_sha256": self.hashes["item_meta_path"], "expected_rows": 3, "completed_rows": 3,
            "coverage": {"expected_rows": 3, "completed_rows": 3, "missing_rows": 0, "duplicate_rows": 0},
            "CC": 0, "validity_rate": 1, "unique_rate": 1, "metrics": metrics,
            "check_results": {key: "passed" for key in ("provenance", "prepared_input_hashes", "checkpoint_reload",
                "tokenizer_atomic_sid_tokens", "model_vocabulary", "validation_coverage", "prediction_coverage",
                "valid_sids", "unique_candidates")}})

    def compare(self):
        return comparison.compare_runs(self.three, self.five, self.data, self.inputs)

    def test_snapshot_uses_its_own_predictions_and_actual_epoch_weights(self):
        result = self.compare()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["train_samples"], 125)
        self.assertEqual(result["epochs3"]["test_metrics"]["HR@10"], 1 / 3)
        self.assertEqual(result["epochs5"]["test_metrics"]["HR@10"], 1)
        self.assertEqual(result["epoch3_snapshot"]["test_metrics"]["HR@10"], 2 / 3)
        self.assertEqual(result["epoch3_snapshot"]["selected_epoch"], 3)
        self.assertEqual(result["epochs5"]["selected_epoch"], 4)
        self.assertNotEqual(result["epochs5"]["prediction_sha256"], result["epoch3_snapshot"]["prediction_sha256"])
        rows = {(row["arm"], row["stratum"], row["metric"], row["k"]): row for row in result["metrics"]}
        self.assertAlmostEqual(rows[("epoch3_snapshot", "overall", "Recall", 10)]["difference_percentage_points_vs_epochs3"], 100 / 3)
        self.assertEqual(rows[("epoch3_snapshot", "behavior_cold", "Recall", 10)]["value"], .5)

    def test_rejects_training_hyperparameter_or_model_identity_changes(self):
        path = self.five / "run_config.json"
        self.edit(path, lambda obj: obj["hyperparameters"].update(seed=43))
        with self.assertRaisesRegex(ValueError, "Shared training hyperparameters"):
            self.compare()
        self.edit(path, lambda obj: obj["hyperparameters"].update(seed=42))
        self.edit(path, lambda obj: obj["model"]["identity"].update(sha256="d" * 64))
        with self.assertRaisesRegex(ValueError, "pretrained model identities"):
            self.compare()

    def test_rejects_best_reloaded_or_wrong_step_epoch_snapshot(self):
        path = self.five / "epoch_3_checkpoint/snapshot_metadata.json"
        self.edit(path, lambda obj: obj.update(best_model_reloaded=True))
        with self.assertRaisesRegex(ValueError, "live training weights"):
            self.compare()
        self.edit(path, lambda obj: obj.update(best_model_reloaded=False, global_step=7))
        with self.assertRaisesRegex(ValueError, "snapshot epoch, step"):
            self.compare()

    def test_rejects_snapshot_evaluation_that_points_to_final_model(self):
        self.edit(self.five / "evaluation_epoch3/metrics.json",
                  lambda obj: obj.update(model_path=str(self.five / "final_checkpoint")))
        with self.assertRaisesRegex(ValueError, "requested exported model"):
            self.compare()

    def test_rejects_incomplete_snapshot_predictions_despite_success_metadata(self):
        path = self.five / "evaluation_epoch3/predictions.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Predictions incomplete"):
            self.compare()

    def test_rejects_ranking_metrics_used_during_training(self):
        self.edit(self.three / "trainer_state.json",
                  lambda obj: obj["log_history"].append({"eval_ndcg10": .5}))
        with self.assertRaisesRegex(ValueError, "unexpectedly used during training"):
            self.compare()

    def test_writes_three_model_report_without_mutating_source_files(self):
        source = self.five / "evaluation_epoch3/predictions.jsonl"
        before = source.read_bytes()
        result = self.compare()
        output = self.root / "comparison"
        files = comparison.write_comparison(result, output)
        self.assertEqual(set(files), {"comparison.json", "comparison.csv", "index.html"})
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(json.loads((output / "comparison.json").read_text())["epoch3_snapshot"]["selected_step"], 6)
        with (output / "comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3 * len(comparison.GROUPS) * 2 * len(reports.TOP_K))
        page = (output / "index.html").read_text()
        self.assertIn("第 3 epoch 即时权重", page)
        self.assertIn("33.3333%", page)
        self.assertIn("66.6667%", page)
        with self.assertRaisesRegex(ValueError, "outside both source"):
            comparison.write_comparison(result, self.five / "comparison")


if __name__ == "__main__":
    unittest.main()
