"""Verified parent provenance, real missing states, and validation-only RL choice."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import report_industrial_rl as reports


class IndustrialRLReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-rl-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.parent, self.data, self.raw, self.out = [self.root / name for name in (
            "rl", "parent", "data", "raw", "report")]
        for path in (self.run, self.parent, self.data, self.raw):
            path.mkdir()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixture(self):
        indices = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(60)}
        indices["59"] = indices["58"][:]
        self.catalog = {key: "".join(tokens) for key, tokens in indices.items()}
        self.save(self.raw / f"{reports.DATASET}.index.json", indices)
        self.save(self.raw / f"{reports.DATASET}.item.json", {key: {"title": key} for key in indices})
        for filename, rows in {"train.csv": [([0], 1), ([1], 58)], "valid.csv": [([0], 1)],
                               "test.csv": [([0], 59), ([1], 2), ([0], 1)]}.items():
            with (self.data / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "item_sid", "history_item_id"])
                writer.writeheader()
                for history, target in rows:
                    writer.writerow({"item_id": target, "item_sid": self.catalog[str(target)], "history_item_id": repr(history)})
        paths = {"train_file": self.data / "train.csv", "eval_file": self.data / "valid.csv",
                 "test_file": self.data / "test.csv", "sid_index_path": self.raw / f"{reports.DATASET}.index.json",
                 "item_meta_path": self.raw / f"{reports.DATASET}.item.json"}
        self.hashes = {key: reports.sha256(path) for key, path in paths.items()}
        self.save(self.data / "preparation_manifest.json", {
            "dataset": reports.DATASET, "validation": "passed", "original_sid": True,
            "rows": {"train": 2, "valid": 1, "test": 3},
            "outputs": {name: {"sha256": self.hashes[key]} for name, key in (
                ("train.csv", "train_file"), ("valid.csv", "eval_file"), ("test.csv", "test_file"))},
            "sources": {"sid_index": {"sha256": self.hashes["sid_index_path"]},
                        "item_metadata": {"sha256": self.hashes["item_meta_path"]}}})
        parent_model = self.parent / "final_checkpoint"
        self.parent_config = {"stage": "sft", "model": {"source": str(self.root / "Qwen2.5-0.5B")},
            "data": {"category": reports.DATASET, "include_title_history": True, "sha256": self.hashes},
            "hyperparameters": {"max_epochs": 3, "lr_scheduler_type": "linear"}}
        self.save(self.parent / "run_config.json", self.parent_config)
        self.save(self.parent / "training_summary.json", {"status": "completed", "selection_metric": "eval_loss",
            "best_checkpoint": str(self.parent / "checkpoint-5187"), "best_metric": 1.56,
            "final_checkpoint": str(parent_model), "final_checkpoint_contains_best_model": True})
        self.save(self.run / "run_config.json", {"stage": "rl", "model": {"source": str(parent_model)},
            "data": self.parent_config["data"], "task_counts": {"fixture": 2}, "hyperparameters": {"max_epochs": 1}})
        self.summary = {"status": "completed", "selection_metric": "eval_ndcg10", "best_metric": .7,
            "final_eval_metrics": {"eval_ndcg10": .7, "eval_generation_rows": 1}, "best_checkpoint": str(self.run / "checkpoint-2"),
            "final_checkpoint": str(self.run / "final_checkpoint"), "final_checkpoint_contains_best_model": True,
            "epoch": 1, "global_step": 2, "stop_reason": "max_epochs"}
        self.save(self.run / "training_summary.json", self.summary)
        self.save(self.run / "run_status.json", {"status": "completed", "stage": "completed"})
        self.save(self.run / "baseline_validation.json", {"status": "passed", "category": reports.DATASET,
            "scope": "complete_original_validation_split", "step": 0, "model_path": str(parent_model),
            "coverage": {"expected_rows": 1, "completed_rows": 1, "missing_rows": 0, "duplicate_rows": 0},
            "generation": {"num_beams": 50, "prompt_style": "original-eval", "full_catalog": True, "seen_item_filter": False},
            "metrics": {"ndcg10": .6}, "source_sha256": {
                "valid_file": self.hashes["eval_file"], "sid_index_path": self.hashes["sid_index_path"],
                "preparation_manifest": reports.sha256(self.data / "preparation_manifest.json")}})
        self.monitor = self.run / "monitor/metrics.jsonl"
        self.monitor.parent.mkdir()
        self.monitor.write_text("".join(json.dumps(row) + "\n" for row in (
            {"step": 1, "reward": 0, "kl": .02, "zero_reward_group_rate": 1, "grad_norm": 99,
             "step_skipped": True, "learning_rate": 1e-5, "loss": -.11},
            {"step": 2, "reward": .25, "kl": .03, "zero_reward_group_rate": .5, "grad_norm": .75,
             "learning_rate": 5e-6, "task/next_sid/reward": .3},
            {"step": 2, "eval_ndcg10": .7, "eval_recall10": .8})))
        preferred = [self.catalog[str(item)] for item in (59, 2, 1)]
        unique = list(dict.fromkeys(self.catalog.values()))
        candidates = (preferred + [sid for sid in unique if sid not in preferred])[:50]
        self.parent_predictions = [{"row_id": i, "target": target, "predictions": candidates,
                                    "prompt_style": "original-eval"} for i, target in enumerate(preferred)]
        self.rl_predictions = []
        for i, target in enumerate(preferred):
            front = [preferred[1], preferred[0]] if i < 2 else []
            choices = front + [sid for sid in unique if sid not in front and (i < 2 or sid != target)]
            self.rl_predictions.append({"row_id": i, "target": target, "predictions": choices[:50], "prompt_style": "original-eval"})
        for folder, predictions, model in ((self.parent, self.parent_predictions, parent_model),
                                            (self.run, self.rl_predictions, self.run / "final_checkpoint")):
            metric = {}
            ranks = [row["predictions"].index(row["target"]) + 1 if row["target"] in row["predictions"] else math.inf
                     for row in predictions]
            for k in reports.TOP_K:
                metric[f"HR@{k}"] = sum(rank <= k for rank in ranks) / len(ranks)
                metric[f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= k) / len(ranks)
            self.save(folder / "evaluation/metrics.json", {"status": "passed", "category": reports.DATASET,
                "scope": "complete_original_test_split", "model_path": str(model), "num_beams": 50,
                "prompt_style": "original-eval", "validation_prompt_style": "training",
                "source_sha256": {key: self.hashes[key] for key in ("train_file", "test_file", "sid_index_path")},
                "item_meta_sha256": self.hashes["item_meta_path"], "expected_rows": 3, "completed_rows": 3,
                "coverage": {"expected_rows": 3, "completed_rows": 3, "missing_rows": 0, "duplicate_rows": 0},
                "CC": 0, "validity_rate": 1, "unique_rate": 1, "metrics": metric,
                "check_results": {key: "passed" for key in ("provenance", "prepared_input_hashes", "checkpoint_reload",
                    "tokenizer_atomic_sid_tokens", "model_vocabulary", "validation_coverage", "prediction_coverage", "valid_sids", "unique_candidates")}})
            (folder / "evaluation/predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        pinned = {key: reports.sha256(self.parent / names[0]) for key, names in reports.PARENT_FILES.items()}
        self.supervisor = {"model_path": str(parent_model), "parent_sft_report_sources": {
            "run_dir": str(self.parent), "sha256": pinned}}
        self.save(self.run / "supervisor_config.json", self.supervisor)

    def build(self):
        return reports.build_reports(self.run, self.data, self.raw)

    def replace_rl_predictions(self, rows):
        path = self.run / "evaluation/predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        evaluation = reports.read_json(self.run / "evaluation/metrics.json")
        ranks = [row["predictions"].index(row["target"]) + 1 if row["target"] in row["predictions"] else math.inf
                 for row in rows]
        for k in reports.TOP_K:
            evaluation["metrics"][f"HR@{k}"] = sum(rank <= k for rank in ranks) / len(rows)
            evaluation["metrics"][f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= k) / len(rows)
        self.save(self.run / "evaluation/metrics.json", evaluation)

    def test_pending_has_no_invented_metric_or_pdf(self):
        result = reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        self.assertEqual((result["training_status"], result["validation_status"], result["comparison_status"]),
                         ("pending", "pending", "pending"))
        training = reports.read_json(self.out / "training_report.json")
        comparison = reports.read_json(self.out / "comparison.json")
        self.assertIsNone(training["metric_statistics"]["reward"]["mean"])
        self.assertEqual(comparison["rows"], [])
        self.assertIsNone(comparison["recommendation"]["recommended_model"])
        self.assertFalse(training["rendered_pdf"])
        self.assertFalse((self.out / "training-report.pdf").exists())

    def test_verified_comparison_preserves_cold_aliases_and_uses_validation_choice(self):
        self.fixture()
        training, validation, comparison, _ = self.build()
        self.assertEqual((training["status"], validation["status"], comparison["status"]), ("completed", "passed", "passed"))
        self.assertEqual(comparison["recommendation"]["recommended_model"], "RL")
        self.assertFalse(comparison["recommendation"]["test_used_for_selection"])
        overall = next(row for row in comparison["rows"] if row["stratum"] == "overall" and row["k"] == 10)
        self.assertEqual((overall["sft_recall"], overall["rl_recall"]), (1, 2 / 3))
        cold = next(row for row in comparison["rows"] if row["stratum"] == "behavior_cold" and row["k"] == 50)
        self.assertEqual((cold["rows"], cold["sft_hits"], cold["rl_hits"]), (2, 2, 2))
        self.assertEqual(validation["behavior_cold_sid_warm_alias_rows"], 1)
        self.assertEqual(training["grad_norm"]["count"], 1)
        self.assertEqual(training["grad_norm"]["mean"], .75)
        self.assertEqual(training["metric_statistics"]["reward"]["count"], 2)
        self.assertEqual(training["metric_statistics"]["reward"]["mean"], .125)
        self.assertEqual(validation["source_sha256"]["predictions"], reports.sha256(self.run / "evaluation/predictions.jsonl"))

    def test_pinned_cache_works_without_original_parent_metadata(self):
        self.fixture()
        for original, cached in reports.PARENT_FILES.values():
            (self.parent / original).replace(self.run / cached)
        parent = self.build()[2]["parent_sft"]
        self.assertEqual(parent["status"], "passed")
        self.assertEqual(parent["source"], "pinned_cache")
        self.assertEqual(parent["report_source_sha256"], self.supervisor["parent_sft_report_sources"]["sha256"])
        with (self.run / "parent_sft_run_config.json").open("a") as stream:
            stream.write(" ")
        comparison = self.build()[2]
        self.assertEqual(comparison["status"], "incomplete")
        self.assertIn("SHA-256", comparison["parent_sft"]["error"])

    def test_unpinned_cache_cannot_substitute_for_parent_identity(self):
        self.fixture()
        for original, cached in reports.PARENT_FILES.values():
            (self.parent / original).replace(self.run / cached)
        self.supervisor.pop("parent_sft_report_sources")
        self.save(self.run / "supervisor_config.json", self.supervisor)
        comparison = self.build()[2]
        self.assertEqual(comparison["status"], "incomplete")
        self.assertIn("pinned SHA-256", comparison["parent_sft"]["error"])

    def test_failed_final_evaluation_retains_training_and_missing_rl_values(self):
        self.fixture()
        self.save(self.run / "evaluation/metrics.json", {"status": "failed", "error": "missing shard"})
        self.save(self.run / "run_status.json", {"status": "failed", "stage": "validating", "error": "missing shard"})
        training, validation, comparison, _ = self.build()
        self.assertEqual(training["status"], "completed")
        self.assertEqual(validation["status"], "failed")
        self.assertEqual(comparison["status"], "failed")
        self.assertTrue(comparison["rows"])
        self.assertTrue(all(row["rl_recall"] is None and row["rl_hits"] is None for row in comparison["rows"]))

    def test_missing_predictions_and_changed_catalog_never_pass(self):
        self.fixture()
        predictions = self.run / "evaluation/predictions.jsonl"
        lines = predictions.read_text().splitlines()
        predictions.write_text("\n".join(lines[:-1]) + "\n")
        self.assertEqual(self.build()[1]["status"], "incomplete")
        predictions.write_text("\n".join(lines) + "\n")
        with (self.raw / f"{reports.DATASET}.item.json").open("a") as stream:
            stream.write(" ")
        validation = self.build()[1]
        self.assertEqual(validation["status"], "incomplete")
        self.assertIn("Training source changed", validation["error"])

    def test_baseline_hash_and_reloaded_score_are_required_for_recommendation(self):
        self.fixture()
        baseline = reports.read_json(self.run / "baseline_validation.json")
        baseline["source_sha256"]["valid_file"] = "wrong"
        self.save(self.run / "baseline_validation.json", baseline)
        comparison = self.build()[2]
        self.assertEqual(comparison["test_comparison_status"], "passed")
        self.assertEqual(comparison["status"], "incomplete")
        self.assertIsNone(comparison["recommendation"]["recommended_model"])
        baseline["source_sha256"]["valid_file"] = self.hashes["eval_file"]
        self.save(self.run / "baseline_validation.json", baseline)
        self.summary["final_eval_metrics"]["eval_ndcg10"] = .5
        self.save(self.run / "training_summary.json", self.summary)
        self.assertIn("Reloaded RL", self.build()[2]["recommendation"]["error"])

    def test_export_schema_has_all_k_rows_and_nulls_for_unobserved_metrics(self):
        self.fixture()
        result = reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        self.assertEqual(result["comparison_status"], "passed")
        with (self.out / "comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        overall = [row for row in rows if row["stratum"] == "overall"]
        self.assertEqual([int(row["k"]) for row in overall], list(reports.TOP_K))
        training = reports.read_json(self.out / "training_report.json")
        self.assertIsNone(training["metric_statistics"]["hit_group_rate"]["last"])
        self.assertNotIn("loss", training["metric_statistics"])
        self.assertIn("teacher-forced", training["objective"])
        for name in ("training_report.json", "validation_report.json", "comparison.json", "comparison.csv", "comparison.html", "index.html"):
            self.assertGreater((self.out / name).stat().st_size, 0)

    def test_missing_rl_telemetry_cannot_claim_completed_training(self):
        self.fixture()
        self.monitor.unlink()
        result = reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        training = reports.read_json(self.out / "training_report.json")
        self.assertEqual(result["training_status"], "incomplete")
        self.assertIsNone(training["metric_statistics"]["reward"]["mean"])
        self.assertIsNone(training["grad_norm"]["mean"])
        self.assertTrue(training["errors"])

    def test_validation_tie_retains_sft_and_failed_training_stays_failed(self):
        self.fixture()
        self.summary["best_metric"] = self.summary["final_eval_metrics"]["eval_ndcg10"] = .6
        self.save(self.run / "training_summary.json", self.summary)
        self.assertEqual(self.build()[2]["recommendation"]["recommended_model"], "SFT")
        self.summary["status"] = "failed"
        self.save(self.run / "training_summary.json", self.summary)
        self.save(self.run / "run_status.json", {"status": "failed", "stage": "training", "error": "nonfinite reward"})
        training, validation, comparison, _ = self.build()
        self.assertEqual(training["status"], "failed")
        self.assertEqual(validation["status"], "incomplete")
        self.assertIsNone(comparison["recommendation"]["recommended_model"])

    def test_parent_metadata_references_and_baseline_model_are_bound_to_verified_sft(self):
        self.fixture()
        config = reports.read_json(self.run / "run_config.json")
        config["parent_sft"] = {"model_path": str(self.parent / "final_checkpoint"), "run_dir": str(self.parent),
            "selected_step": 5187, "selected_checkpoint": str(self.parent / "checkpoint-5187"),
            "config_sha256": self.supervisor["parent_sft_report_sources"]["sha256"]["run_config"],
            "summary_sha256": "forged"}
        self.save(self.run / "run_config.json", config)
        comparison = self.build()[2]
        self.assertIn("metadata reference", comparison["parent_sft"]["error"])
        self.assertIsNone(comparison["recommendation"]["recommended_model"])
        config.pop("parent_sft")
        self.save(self.run / "run_config.json", config)
        baseline = reports.read_json(self.run / "baseline_validation.json")
        baseline["model_path"] = str(self.root / "another-model")
        self.save(self.run / "baseline_validation.json", baseline)
        self.assertIn("parent SFT model", self.build()[2]["recommendation"]["error"])

    def test_runtime_correction_provenance_and_comparison_limits_are_visible(self):
        self.fixture()
        metadata = {"applied": True, "version": "0.18.0", "original_source_sha256": "a" * 64,
                    "patched_source_sha256": "b" * 64, "reason": "Preserve earlier microbatch gradients"}
        config = reports.read_json(self.run / "run_config.json")
        config["runtime_compatibility"] = metadata
        self.save(self.run / "run_config.json", config)
        result = reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        self.assertEqual(result["comparison_status"], "passed")
        for name in ("training_report.json", "validation_report.json", "comparison.json"):
            report = reports.read_json(self.out / name)
            self.assertEqual(report["runtime_compatibility"], metadata)
            lines = "\n".join(reports.runtime_pdf_lines(report))
            self.assertIn(metadata["original_source_sha256"], lines)
            self.assertIn(metadata["patched_source_sha256"], lines)
            self.assertIn("0.18.0", lines)
            self.assertIn("Historical SFT", lines)
            self.assertIn("cannot isolate algorithm gains", lines)
            self.assertIn("Failed earlier attempts are not RL results", lines)
        for name in ("index.html", "comparison.html"):
            content = (self.out / name).read_text()
            self.assertIn(metadata["original_source_sha256"], content)
            self.assertIn(metadata["patched_source_sha256"], content)
            self.assertIn("没有因此自动重训 SFT", content)
            self.assertIn("历史 SFT", content)
            self.assertIn("此前失败尝试不作为 RL 结果", content)

    def test_absent_runtime_metadata_does_not_claim_a_correction_was_applied(self):
        self.fixture()
        reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        for name in ("training_report.json", "validation_report.json", "comparison.json"):
            report = reports.read_json(self.out / name)
            self.assertIsNone(report["runtime_compatibility"])
            self.assertIn("not recorded", reports.runtime_pdf_lines(report)[0])
            self.assertFalse(any(line.startswith("This RL run applies") for line in report["comparison_limitations"]))
        content = (self.out / "index.html").read_text()
        self.assertIn("未记录已应用", content)
        self.assertNotIn("本次 RL 已应用", content)
        comparison = reports.read_json(self.out / "comparison.json")
        self.assertEqual(comparison["status"], "passed")

    def test_expectation_improved_requires_validation_and_test_gains_with_stable_recall(self):
        self.fixture()
        unique = list(dict.fromkeys(self.catalog.values()))
        predictions = [{**row, "predictions": [row["target"], *[sid for sid in unique if sid != row["target"]]][:50]}
                       for row in self.rl_predictions]
        self.replace_rl_predictions(predictions)
        reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        comparison = reports.read_json(self.out / "comparison.json")
        assessment = comparison["expectation_assessment"]
        self.assertEqual(assessment["status"], "improved")
        self.assertEqual(assessment["primary_metrics"]["test_hr10"]["direction"], "unchanged")
        self.assertEqual(assessment["primary_metrics"]["test_ndcg10"]["direction"], "up")
        self.assertEqual(comparison["recommendation"]["recommended_model"], "RL")
        self.assertFalse(assessment["test_used_for_selection"])
        for name in ("training_report.json", "validation_report.json"):
            report = reports.read_json(self.out / name)
            self.assertEqual(report["expectation_assessment"], assessment)
            self.assertEqual(report["display_timezone"], "Asia/Shanghai")
            self.assertTrue(report["generated_at_local"].endswith("+08:00"))
        self.assertIn("Expectation assessment: improved", "\n".join(reports.expectation_pdf_lines(assessment)))
        self.assertIn("是否符合改善预期", (self.out / "index.html").read_text())

    def test_expectation_mixed_reports_warm_cold_hit_changes_without_reselecting(self):
        self.fixture()
        unique = list(dict.fromkeys(self.catalog.values()))
        self.rl_predictions[0]["predictions"] = [sid for sid in unique if sid != self.rl_predictions[0]["target"]][:50]
        self.replace_rl_predictions(self.rl_predictions)
        _, _, comparison, _ = self.build()
        assessment = comparison["expectation_assessment"]
        self.assertEqual(assessment["status"], "mixed")
        self.assertEqual(assessment["primary_metrics"]["validation_ndcg10"]["direction"], "up")
        self.assertEqual(assessment["primary_metrics"]["test_ndcg10"]["direction"], "down")
        groups = {row["stratum"]: row for row in assessment["strata_hr50"]}
        cold = groups["behavior_cold"]
        self.assertEqual((cold["rows"], cold["sft_hits"], cold["rl_hits"], cold["hit_difference"]), (2, 2, 1, -1))
        self.assertEqual(cold["difference_percentage_points"], -50)
        self.assertEqual(groups["behavior_warm"]["hit_difference"], -1)
        self.assertEqual(comparison["recommendation"]["recommended_model"], "RL")
        self.assertTrue(any("one training seed" in line for line in assessment["evidence_boundaries"]))
        self.assertTrue(any("2 test rows" in line for line in assessment["evidence_boundaries"]))

    def test_expectation_not_improved_and_incomplete_test_remains_pending(self):
        self.fixture()
        self.summary["best_metric"] = self.summary["final_eval_metrics"]["eval_ndcg10"] = .5
        self.save(self.run / "training_summary.json", self.summary)
        comparison = self.build()[2]
        self.assertEqual(comparison["expectation_assessment"]["status"], "not_improved")
        self.assertEqual(comparison["recommendation"]["recommended_model"], "SFT")
        (self.run / "evaluation/predictions.jsonl").unlink()
        comparison = self.build()[2]
        assessment = comparison["expectation_assessment"]
        self.assertEqual(assessment["status"], "pending")
        self.assertIsNone(assessment["primary_metrics"]["test_ndcg10"]["difference_percentage_points"])
        self.assertEqual(assessment["strata_hr50"], [])
        self.assertIn("no expectation conclusion", assessment["summary"])

    def test_selected_best_revalidation_cannot_overwrite_training_curve_at_same_step(self):
        self.fixture()
        records = [
            {"step": 1449, "kind": "train", "phase": "training", "reward": .25, "grad_norm": .8},
            {"step": 1449, "kind": "eval", "phase": "training", "eval_ndcg10": .0929206607157986,
             "eval_recall10": .13},
            {"step": 1449, "kind": "eval", "phase": "final_validation", "eval_ndcg10": .09316929779634507,
             "eval_recall10": .14},
            {"step": 1450, "kind": "train", "phase": "final_validation", "reward": .3, "grad_norm": .9},
        ]
        self.monitor.write_text("".join(json.dumps(row) + "\n" for row in records))
        training, _, _, steps = self.build()
        self.assertEqual(steps[0]["eval_ndcg10"], .0929206607157986)
        self.assertEqual(steps[0]["eval_recall10"], .13)
        self.assertEqual(steps[0]["reward"], .25)
        self.assertEqual(steps[1]["reward"], .3)
        self.assertEqual(training["metric_statistics"]["eval_ndcg10"]["count"], 1)
        self.assertEqual(training["training_summary"]["final_eval_metrics"]["eval_ndcg10"], .7)

    def test_posttraining_generator_hash_is_separate_from_training_time_source(self):
        self.fixture()
        training_hash = "c" * 64
        self.supervisor["source_sha256"] = {"scripts/report_industrial_rl.py": training_hash}
        self.save(self.run / "supervisor_config.json", self.supervisor)
        reports.generate(self.run, self.data, self.raw, self.out, plots=False)
        for name in ("training_report.json", "validation_report.json", "comparison.json"):
            report = reports.read_json(self.out / name)
            self.assertEqual(report["report_generator_sha256"], reports.sha256(Path(reports.__file__)))
            self.assertEqual(report["training_report_source_sha256"], training_hash)
            self.assertNotEqual(report["report_generator_sha256"], report["training_report_source_sha256"])


if __name__ == "__main__":
    unittest.main()
