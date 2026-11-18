"""Integrity and cold-item semantics for the original Industrial reports."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import report_industrial_sft as reports
from scripts import compare_industrial_reproductions as comparisons


class IndustrialReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="industrial-report-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.data, self.raw, self.out = [self.root / name for name in ("run", "data", "raw", "reports")]
        for folder in (self.run, self.data, self.raw):
            folder.mkdir()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self):
        self.indices = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(60)}
        self.indices["59"] = self.indices["58"][:]
        catalog = {key: "".join(value) for key, value in self.indices.items()}
        self.save(self.raw / f"{reports.DATASET}.index.json", self.indices)
        self.save(self.raw / f"{reports.DATASET}.item.json", {key: {"title": "item " + key} for key in catalog})

        def write(name, rows):
            with (self.data / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid", "history_item_id"])
                writer.writeheader()
                for history, target in rows:
                    writer.writerow({"item_id": target, "item_sid": catalog[str(target)], "history_item_id": repr(history)})
        write("train.csv", [([0], 1), ([1], 58)])
        write("valid.csv", [([0], 1)])
        write("test.csv", [([0], 59), ([1], 2), ([0], 1)])
        paths = {"train_file": self.data / "train.csv", "eval_file": self.data / "valid.csv", "test_file": self.data / "test.csv",
                 "sid_index_path": self.raw / f"{reports.DATASET}.index.json",
                 "item_meta_path": self.raw / f"{reports.DATASET}.item.json"}
        self.hashes = {key: reports.sha256(path) for key, path in paths.items()}
        self.save(self.run / "run_config.json", {"model": {"source": "fixture-Base"}, "data": {
            "category": reports.DATASET, "sha256": self.hashes}})
        self.save(self.data / "preparation_manifest.json", {
            "dataset": reports.DATASET, "validation": "passed", "original_sid": True,
            "rows": {"train": 2, "valid": 1, "test": 3}, "items": 60,
            "sources": {"sid_index": {"sha256": self.hashes["sid_index_path"]},
                        "item_metadata": {"sha256": self.hashes["item_meta_path"]}},
            "outputs": {"train.csv": {"sha256": self.hashes["train_file"]},
                        "valid.csv": {"sha256": self.hashes["eval_file"]},
                        "test.csv": {"sha256": self.hashes["test_file"]}}})
        unique = list(dict.fromkeys(catalog.values()))
        preferred = [catalog["59"], catalog["2"], catalog["1"]]
        candidates = (preferred + [sid for sid in unique if sid not in preferred])[:50]
        self.predictions = [{"row_id": row_id, "target": catalog[str(target)], "predictions": candidates,
                             "prompt_style": "original-eval"}
                            for row_id, target in enumerate((59, 2, 1))]
        metric = {}
        for k in reports.TOP_K:
            metric[f"HR@{k}"] = sum(rank <= k for rank in (1, 2, 3)) / 3
            metric[f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for rank in (1, 2, 3) if rank <= k) / 3
        self.evaluation = {"status": "passed", "category": reports.DATASET, "num_beams": 50,
            "prompt_style": "original-eval", "validation_prompt_style": "training",
            "source_sha256": {key: self.hashes[key] for key in ("train_file", "test_file", "sid_index_path")},
            "item_meta_sha256": self.hashes["item_meta_path"], "expected_rows": 3, "completed_rows": 3,
            "coverage": {"expected_rows": 3, "completed_rows": 3, "missing_rows": 0, "duplicate_rows": 0},
            "CC": 0, "validity_rate": 1, "unique_rate": 1, "metrics": metric,
            "check_results": {key: "passed" for key in ("provenance", "prepared_input_hashes", "checkpoint_reload",
                "tokenizer_atomic_sid_tokens", "model_vocabulary", "validation_coverage", "prediction_coverage",
                "valid_sids", "unique_candidates")}}
        self.save(self.run / "evaluation/metrics.json", self.evaluation)
        self.save_predictions(self.predictions)
        self.save(self.run / "training_summary.json", {"status": "completed", "best_checkpoint": "checkpoint-2", "best_metric": 1.0})
        self.save(self.run / "run_status.json", {"status": "completed", "stage": "completed"})
        folder = self.run / "monitor"
        folder.mkdir(exist_ok=True)
        (folder / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss": 2, "grad_norm": .5, "learning_rate": .0003}) + "\n")

    def save_predictions(self, rows):
        path = self.run / "evaluation/predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_pending_report_does_not_claim_zero_or_completed(self):
        result = reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        self.assertEqual(result["training_status"], "pending")
        self.assertEqual(result["validation_status"], "pending")
        validation = reports.read_json(self.out / "validation_report.json")
        self.assertEqual(validation["strata"], [])
        self.assertFalse(validation["rendered_pdf"])
        self.assertEqual(validation["comparison_reference"]["metrics"]["HR@10"], .1127)
        self.assertEqual(validation["comparison_reference"]["metrics"]["NDCG@50"], .0997)
        self.assertIn("not independently verified", validation["reference_comparison_note"])

    def test_original_collisions_preserved_and_cold_defined_by_item_id(self):
        self.fixture()
        training, validation, _ = reports.build_reports(self.run, self.data, self.raw)
        self.assertEqual(training["status"], "completed")
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["catalog_items"], 60)
        self.assertEqual(validation["unique_sids"], 59)
        self.assertEqual(validation["collision_groups"], 1)
        self.assertEqual(validation["collision_items"], 2)
        self.assertEqual(validation["rows"], 3)
        self.assertEqual(validation["behavior_cold_sid_warm_alias_rows"], 1)
        groups = {row["key"]: row for row in validation["strata"]}
        self.assertEqual(groups["behavior_cold"]["rows"], 2)
        self.assertEqual(groups["behavior_cold_collision_affected"]["rows"], 1)
        self.assertEqual(groups["behavior_cold_collision_free"]["rows"], 1)
        self.assertEqual(groups["overall"]["metrics"], self.evaluation["metrics"])
        self.assertEqual(groups["behavior_cold"]["metrics"]["HR@1"], .5)

    def test_modified_input_hash_prevents_successful_report(self):
        self.fixture()
        with (self.data / "train.csv").open("a") as stream:
            stream.write("changed\n")
        validation = reports.build_reports(self.run, self.data, self.raw)[1]
        self.assertEqual(validation["status"], "incomplete")
        self.assertIn("Training source changed", validation["errors"][0])

    def test_missing_or_duplicate_predictions_prevent_successful_report(self):
        self.fixture()
        for rows in (self.predictions[:-1], [self.predictions[0], *self.predictions]):
            with self.subTest(rows=len(rows)):
                self.save_predictions(rows)
                validation = reports.build_reports(self.run, self.data, self.raw)[1]
                self.assertEqual(validation["status"], "incomplete")

    def test_accepted_metrics_must_match_reaggregated_predictions(self):
        self.fixture()
        self.evaluation["metrics"]["HR@10"] = .25
        self.save(self.run / "evaluation/metrics.json", self.evaluation)
        validation = reports.build_reports(self.run, self.data, self.raw)[1]
        self.assertEqual(validation["status"], "incomplete")
        self.assertIn("Recomputed overall metric", validation["errors"][0])

    def test_failed_final_evaluation_preserves_completed_training(self):
        self.fixture()
        self.evaluation["status"] = "failed"
        self.evaluation["error"] = "missing shard"
        self.save(self.run / "evaluation/metrics.json", self.evaluation)
        self.save(self.run / "run_status.json", {"status": "failed", "stage": "validating"})
        training, validation, _ = reports.build_reports(self.run, self.data, self.raw)
        self.assertEqual(training["status"], "completed")
        self.assertEqual(validation["status"], "failed")

    def test_model_and_scheduler_labels_follow_each_isolated_run(self):
        self.fixture()
        original = reports.read_json(self.run / "run_config.json")
        for model, scheduler in (("Qwen2.5-1.5B", "cosine"), ("Qwen2.5-0.5B", "linear")):
            config = {**original, "model": {"source": "/weights/" + model}, "hyperparameters": {
                "world_size": 4, "max_epochs": 3, "global_batch_size": 64, "lr_scheduler_type": scheduler}}
            self.save(self.run / "run_config.json", config)
            destination = self.out / scheduler
            reports.generate(self.run, self.data, self.raw, destination, plots=False)
            validation = reports.read_json(destination / "validation_report.json")
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["run_settings"]["model_name"], model)
            self.assertIn(f"{model}; scheduler={scheduler}; GPUs=4", validation["reference_comparison_note"])
            self.assertIn(f"scheduler={scheduler}", (destination / "index.html").read_text())
        prior = reports.read_json(self.out / "cosine/validation_report.json")
        self.assertEqual(prior["run_settings"]["model_name"], "Qwen2.5-1.5B")
        self.assertEqual(prior["run_settings"]["lr_scheduler_type"], "cosine")
        self.assertNotIn("1.5B", (self.out / "linear/index.html").read_text())
        pending = reports.run_settings({}, {"parameters": {"base_model": "/weights/Qwen2.5-0.5B",
            "world_size": 4, "lr_scheduler_type": "linear", "num_epochs": 3, "global_batch_size": 64}})
        self.assertIn("Qwen2.5-0.5B; scheduler=linear; GPUs=4", reports.run_label(pending))

    def comparison_fixture(self):
        self.fixture()
        self.reference = self.root / "reference"
        self.previous = self.root / "previous"
        config = reports.read_json(self.run / "run_config.json")
        config["model"]["source"] = "/weights/Qwen2.5-0.5B"
        config["hyperparameters"] = {"world_size": 4, "max_epochs": 3, "global_batch_size": 64, "lr_scheduler_type": "linear"}
        self.save(self.run / "run_config.json", config)
        shutil.copytree(self.run, self.previous)
        config["model"]["source"] = "/weights/Qwen2.5-1.5B"
        config["hyperparameters"]["lr_scheduler_type"] = "cosine"
        self.save(self.previous / "run_config.json", config)
        with (self.data / "test.csv").open() as stream:
            tests = list(csv.DictReader(stream))
        self.archived = []
        for row, prediction in zip(tests, self.predictions):
            history = json.loads(row["history_item_id"])
            prompt = "Can you predict the next possible item the user may expect, given the following chronological interaction history: " + ", ".join("".join(self.indices[str(item)]) for item in history)
            self.archived.append({"input": prompt, "output": prediction["target"] + "\n", "predict": prediction["predictions"]})
        for stage in ("sft", "rl"):
            self.save(self.reference / "results" / f"eval_{stage}_result.json", self.archived)

    def comparison(self, plots=False):
        return comparisons.generate(self.reference, self.previous, self.run, self.data, self.raw, self.out,
                                    plots=plots, expected_cold_rows=2, expected_cold_items=2)

    def test_four_source_comparison_preserves_original_aliases_and_all_rows(self):
        self.comparison_fixture()
        result = self.comparison()
        self.assertEqual(result["status"], "passed")
        comparison = reports.read_json(self.out / "industrial_reproduction_comparison.json")
        self.assertEqual(comparison["cohort"]["cold_sid_warm_alias_rows"], 1)
        self.assertEqual([row["stage"] for row in comparison["results"]], ["SFT", "SFT + RL", "SFT", "SFT"])
        for row in comparison["results"]:
            self.assertEqual(row["strata"]["overall"]["rows"], 3)
            self.assertEqual(row["strata"]["cold"]["rows"], 2)
            self.assertEqual(row["strata"]["overall"]["metrics"], self.evaluation["metrics"])
        self.assertEqual(len(comparison["current_minus_reference"]), 108)
        self.assertEqual(comparison["results"][2]["scheduler"], "cosine")
        self.assertEqual(comparison["results"][3]["scheduler"], "linear")

    def test_comparison_rejects_archive_history_changes_despite_matching_target(self):
        self.comparison_fixture()
        self.archived[0]["input"] += ", <a_0><b_0><c_0>"
        self.save(self.reference / "results/eval_sft_result.json", self.archived)
        with self.assertRaisesRegex(ValueError, "Archive history/order differs"):
            self.comparison()

    def test_comparison_rejects_missing_prediction_rows_and_wrong_cold_cohort(self):
        self.comparison_fixture()
        with self.assertRaisesRegex(ValueError, "Behavior-cold cohort differs"):
            comparisons.build_comparison(self.reference, self.previous, self.run, self.data, self.raw)
        self.save_predictions(self.predictions[:-1])
        with self.assertRaisesRegex(ValueError, "Prediction coverage is incomplete"):
            self.comparison()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib unavailable")
    def test_comparison_renders_static_overall_and_cold_figure(self):
        self.comparison_fixture()
        self.comparison(plots=True)
        self.assertTrue((self.out / "industrial_reproduction_comparison.png").read_bytes().startswith(b"\x89PNG"))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib unavailable")
    def test_real_rendering_outputs_live_and_completed_pdf_and_plots(self):
        for completed in (False, True):
            if completed:
                self.fixture()
            reports.generate(self.run, self.data, self.raw, self.out)
            for name in ("training-report.pdf", "validation-report.pdf"):
                self.assertTrue((self.out / name).read_bytes().startswith(b"%PDF"))
            self.assertTrue((self.out / "ranking.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((self.out / "grad_norm.png").read_bytes().startswith(b"\x89PNG"))
            if completed:
                self.assertEqual(reports.read_json(self.out / "validation_report.json")["status"], "passed")


if __name__ == "__main__":
    unittest.main()
