"""Evidence, resume and archive-preservation checks for live SFT reports."""
import hashlib
import importlib.util
import json
import csv
from pathlib import Path
import tempfile
import unittest

from scripts import report_sft as reports


class SFTReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-report-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.run = self.folder / "run"
        self.report_root = self.folder / "reports"
        self.run.mkdir()

    def save(self, filename, value):
        path = self.run / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def save_records(self, filename, values, tail=""):
        path = self.run / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in values) + tail, encoding="utf-8")

    def passed_evaluation(self, rows=4):
        return {
            "status": "passed", "expected_rows": rows, "completed_rows": rows,
            "coverage": {"expected_rows": rows, "completed_rows": rows, "missing_rows": 0, "duplicate_rows": 0},
            "metrics": {f"{metric}@{k}": 0.0 for metric in ("HR", "NDCG") for k in reports.TOP_K},
            "CC": 0, "validity_rate": 1.0, "unique_rate": 1.0,
            "check_results": {name: "passed" for name in (
                "provenance", "tokenizer_atomic_sid_tokens", "model_vocabulary", "prediction_coverage",
                "valid_sids", "unique_candidates")},
            "validation": {"rows": 4, "answer_tokens": 20, "loss": 1.0, "masked_perplexity": 2.71828},
        }

    def comparison_inputs(self):
        hashes = {"train_file": "a" * 64, "test_file": "b" * 64, "sid_index_path": "c" * 64}
        config = {"stage": "sft", "model": {"source": "/weights/Qwen2.5-1.5B"},
                  "data": {"category": "Video_Games", "train_file": str(self.run / "prepared/train.csv"),
                           "sha256": {"train_file": hashes["train_file"], "sid_index_path": hashes["sid_index_path"]}},
                  "hyperparameters": {"max_epochs": 10, "learning_rate": .0003, "lr_scheduler_type": "linear"}}
        self.save("run_config.json", config)
        self.save("prepared/preparation_manifest.json", {"dataset": "Video_Games", "items": 100,
                  "rows": {"train": 8, "test": 4}, "history_length": {"maximum": 20}, "selected_sid_epoch": 2900})
        baseline = {"status": "passed", "dataset": "Video_Games", "catalog_items": 100,
                    "train_rows": 8, "test_rows": 4, "source_sha256": hashes,
                    "popularity": {"metrics": {f"{metric}@{k}": k * .0001 for metric in ("HR", "NDCG") for k in reports.TOP_K}},
                    "uniform_random": {"metrics": {f"{metric}@{k}": k * .00001 for metric in ("HR", "NDCG") for k in reports.TOP_K}}}
        evaluation = self.passed_evaluation()
        evaluation["source_sha256"] = dict(hashes)
        evaluation["metrics"] = {f"{metric}@{k}": k * .01 * (1 if metric == "HR" or k == 1 else .5)
                                 for metric in ("HR", "NDCG") for k in reports.TOP_K}
        self.save("comparison_baselines.json", baseline)
        self.save("evaluation/metrics.json", evaluation)
        self.save("baseline_validation.json", {"loss": 2, "masked_perplexity": 7.389})
        return baseline, evaluation

    def behavior_shift_inputs(self):
        _, evaluation = self.comparison_inputs()
        config = json.loads((self.run / "run_config.json").read_text())
        config["data"]["sha256"]["item_meta_path"] = "e" * 64
        self.save("run_config.json", config)
        sources = {**evaluation["source_sha256"], "item_meta_path": "e" * 64, "predictions": "f" * 64,
                   "evaluation_metrics": hashlib.sha256((self.run / "evaluation/metrics.json").read_bytes()).hexdigest()}
        strata = []
        for key, count, factor in (("overall", 4, 1), ("behavior_cold", 1, 0), ("behavior_warm", 3, 4/3),
                                    ("repeat_last", 1, 1), ("not_repeat_last", 3, 1)):
            strata.append({"key": key, "label": key, "rows": count, "share": count / 4,
                           "metrics": {name: value * factor for name, value in evaluation["metrics"].items()},
                           "popularity": {name: 0 for name in evaluation["metrics"]}, "uniform_random": {}, "hits_by_k": {}})
        diagnostic = {"status": "passed", "schema_version": 1, "source_sha256": sources,
                      "overall_test_rows": 4, "counts": {"train_rows": 8, "test_rows": 4, "catalog_items": 100},
                      "shift_summary": {"test_behavior_cold_rows": 1, "test_behavior_cold_rate": .25,
                                        "train_repeat_last_rate": .05, "test_repeat_last_rate": .25},
                      "alignment_coverage": {"cold_unique_items": 1, "sid_to_title_items": 1, "title_to_sid_items": 1},
                      "prediction_diagnostics": {"top1_equals_last_rate": .5, "copy_precision": .04,
                                                 "copy_share_of_all_top1_hits": .5, "cold_candidate_rate": .0002,
                                                 "cold_predicted_unique_items": 1}, "strata": strata}
        self.save("behavior_shift_analysis.json", diagnostic)
        return diagnostic, evaluation

    def test_empty_and_running_reports_never_claim_completion(self):
        pending, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(validation["status"], "pending")
        self.assertIsNone(pending["grad_norm"]["mean"])
        self.save("run_status.json", {"status": "running", "stage": "training"})
        self.save_records("monitor/metrics.jsonl", [{"step": 1, "loss": 2, "grad_norm": 3}])
        result = reports.generate(self.run, self.report_root, plots=False)
        self.assertEqual(result["training_status"], "running")
        self.assertEqual(result["validation_status"], "pending")
        text = (self.report_root / reports.TRAINING_DIRECTORY / "index.html").read_text()
        self.assertIn("本次训练尚未确认完成", text)
        record = json.loads((self.report_root / reports.TRAINING_DIRECTORY / "training_report.json").read_text())
        self.assertFalse(record["rendered_pdf"])
        self.assertFalse(record["rl_training_started"])

    def test_resume_deduplication_preserves_eval_and_excludes_invalid_updates(self):
        self.save_records("monitor/metrics.jsonl", [
            {"step": 1, "kind": "train", "loss": 3, "grad_norm": 2},
            {"step": 2, "kind": "train", "loss": 2, "grad_norm": 4},
            {"step": 2, "kind": "train", "loss": 1.5, "grad_norm": 6, "rolling_grad_norm_mean": 4},
            {"step": 2, "kind": "eval", "eval_loss": 1.3},
            {"step": 3, "loss": 1.1, "grad_norm": 100, "step_skipped": True},
            {"step": 4, "loss": 1.0},
            {"step": 5, "loss": .9, "grad_norm": None, "grad_norm_nonfinite": True},
            {"step": 6, "kind": "eval", "eval_loss": 1.2},
        ], tail='{"step": 7, "loss":')
        training, _, steps, _ = reports.build_reports(self.run)
        self.assertEqual(len(steps), 6)
        self.assertEqual(steps[1]["grad_norm"], 6)
        self.assertEqual(steps[1]["eval_loss"], 1.3)
        self.assertEqual(steps[1]["grad_norm_rolling_mean"], 4)
        grad = training["grad_norm"]
        self.assertEqual(grad["valid_count"], 2)
        self.assertEqual(grad["mean"], 4)
        self.assertEqual(grad["max"], 6)
        self.assertEqual(grad["max_step"], 2)
        self.assertEqual(grad["skipped_update_count"], 1)
        self.assertEqual(grad["missing_count"], 1)
        self.assertEqual(grad["nonfinite_or_invalid_count"], 1)
        self.assertFalse(training["ingestion_issues"])

    def test_complete_invalid_line_is_reported_but_partial_final_line_is_tolerated(self):
        self.save_records("monitor/metrics.jsonl", [], tail='not-json\n{"step":')
        training, _, _, _ = reports.build_reports(self.run)
        self.assertEqual(len(training["ingestion_issues"]), 1)

    def test_overall_failure_does_not_relabel_successful_training_as_failed(self):
        self.save("run_status.json", {"status": "failed", "stage": "validating"})
        self.save("training_summary.json", {"status": "completed", "best_checkpoint": "checkpoint-344"})
        self.save_records("monitor/metrics.jsonl", [{"step": 1, "loss": 2, "grad_norm": 3}])
        self.save("evaluation/metrics.json", {"status": "failed", "completed_rows": 2, "expected_rows": 4,
                                                "error": "missing shard"})
        training, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(training["status"], "completed")
        self.assertEqual(validation["status"], "failed")
        self.assertIn("missing shard", reports.validation_html(validation))

    def test_completed_training_without_gradient_evidence_is_incomplete(self):
        self.save("training_summary.json", {"status": "completed"})
        training, _, _, _ = reports.build_reports(self.run)
        self.assertEqual(training["status"], "incomplete")
        self.assertTrue(training["ingestion_issues"])

    def test_actual_monitor_schema_controls_window_and_preflight_is_preparing(self):
        self.save("run_config.json", {"hyperparameters": {"monitor_window": 2}})
        self.save("run_status.json", {"status": "running", "stage": "preflight"})
        self.save_records("monitor/metrics.jsonl", [
            {"step": 1, "loss": 3, "grad_norm": 100},
            {"step": 2, "loss": 2, "grad_norm": 2},
            {"step": 3, "loss": 1, "grad_norm": 4}])
        training, _, _, _ = reports.build_reports(self.run)
        self.assertEqual(training["status"], "preparing")
        self.assertEqual(training["grad_norm"]["window_size"], 2)
        self.assertEqual(training["grad_norm"]["rolling_mean"], 3)
        self.assertIn("最近 2 有效步", reports.training_html(training))
        self.save("monitor/summary.json", {"monitor_window": 1})
        training, _, _, _ = reports.build_reports(self.run)
        self.assertEqual(training["grad_norm"]["rolling_mean"], 4)

    def test_timing_waits_for_post_warmup_samples_and_uses_actual_max_steps(self):
        rows = [{"step": step, "loss": 1.0, "grad_norm": 2.0, "elapsed_seconds": 1000 + 2 * step}
                for step in range(1, 23)]
        self.assertIsNone(reports.estimate_training_timing(rows[:19], 100)["seconds_per_step"])
        self.assertIsNone(reports.estimate_training_timing(rows[:21], 100)["seconds_per_step"])
        timing = reports.estimate_training_timing(rows, 100)
        self.assertEqual(timing["seconds_per_step"], 2)
        self.assertEqual(timing["estimated_remaining_training_seconds"], 156)
        self.save("run_status.json", {"status": "running", "stage": "training"})
        self.save("monitor/summary.json", {"max_steps": 100})
        self.save_records("monitor/metrics.jsonl", rows)
        training, _, _, _ = reports.build_reports(self.run)
        self.assertEqual(training["timing"], timing)
        self.assertIn("不包含后续验证", reports.training_html(training))

    def test_timing_excludes_skipped_updates_and_evaluation_pauses(self):
        rows = [{"step": step, "loss": 1, "grad_norm": 2, "elapsed_seconds": elapsed}
                for step, elapsed in ((21, 42), (22, 500), (23, 502), (24, 504), (25, 505))]
        rows[1]["eval_loss"] = 1.0
        rows[-1]["step_skipped"] = True
        timing = reports.estimate_training_timing(rows, 100)
        self.assertEqual(timing["seconds_per_step"], 2)
        self.assertEqual(timing["estimated_remaining_training_seconds"], 152)

    def test_timing_uses_at_most_the_last_100_valid_update_records(self):
        rows = [{"step": step, "loss": 1, "grad_norm": 2,
                 "elapsed_seconds": 20 * step if step <= 50 else 1000 + 2 * (step - 50)}
                for step in range(21, 151)]
        timing = reports.estimate_training_timing(rows, 200)
        self.assertEqual(timing["seconds_per_step"], 2)
        self.assertEqual(timing["estimated_remaining_training_seconds"], 100)

    def test_validation_requires_complete_rows_atomic_model_checks_and_valid_candidates(self):
        for field, changed in (("completed_rows", 3), ("CC", 1), ("validity_rate", .9), ("unique_rate", .9)):
            with self.subTest(field=field):
                evaluation = self.passed_evaluation()
                evaluation[field] = changed
                self.save("evaluation/metrics.json", evaluation)
                self.assertEqual(reports.build_reports(self.run)[1]["status"], "incomplete")
        evaluation = self.passed_evaluation()
        evaluation["check_results"]["provenance"] = "failed"
        self.save("evaluation/metrics.json", evaluation)
        self.assertEqual(reports.build_reports(self.run)[1]["status"], "incomplete")
        evaluation = self.passed_evaluation()
        del evaluation["metrics"]["HR@50"]
        self.save("evaluation/metrics.json", evaluation)
        self.assertEqual(reports.build_reports(self.run)[1]["status"], "incomplete")

    def test_zero_recommendation_accuracy_can_be_a_valid_complete_measurement(self):
        self.save("evaluation/metrics.json", self.passed_evaluation())
        self.save("baseline_validation.json", {"loss": 2, "masked_perplexity": 7.389, "answer_token_count": 20})
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["ranking"][0]["HR"], 0)
        text = reports.validation_html(validation)
        self.assertIn("0.000%", text)
        self.assertIn("不是通用语言困惑度", text)
        self.assertIn("共享已知商品目录", text)
        self.assertIn("完整测试集已通过验证，共 4 条样本", text)
        self.assertIn("变化 -50.00%", text)
        observations = reports.validation_chart_data(validation)
        self.assertEqual(observations["ranking"][0]["HR"], 0)
        self.assertEqual(observations["loss"], [2, 1])
        self.assertEqual(observations["masked_perplexity"], [7.389, 2.71828])

    def test_final_charts_and_overview_reject_partial_evaluation_results(self):
        self.save("baseline_validation.json", {"loss": 2, "masked_perplexity": 7.389})
        for evaluation in ({}, {**self.passed_evaluation(), "completed_rows": 3}):
            with self.subTest(evaluation=evaluation):
                self.save("evaluation/metrics.json", evaluation)
                _, validation, _, _ = reports.build_reports(self.run)
                observations = reports.validation_chart_data(validation)
                self.assertFalse(any(observations.values()))
                text = reports.validation_html(validation)
                self.assertIn("最终结果将在全部测试样本和模型检查通过后列出", text)
                self.assertNotIn("完整测试集已通过验证，共", text)
                self.assertIn("ranking.png", text)
                self.assertIn("validation-comparison.png", text)

    def test_predeclared_expectations_require_complete_evaluation_and_bound_identities(self):
        self.comparison_inputs()
        self.save("evaluation/metrics.json", {})
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(validation["expectation_analysis"]["status"], "pending")
        comparison = validation["comparison"]
        self.assertEqual(comparison["same_dataset_baselines"]["status"], "pending")
        self.assertNotIn("metrics", comparison["same_dataset_baselines"])
        self.assertEqual(comparison["actual"]["history_max"], 20)
        self.assertEqual(comparison["actual"]["sid_selected_epoch"], 2900)
        settings = {row["key"]: row for row in comparison["comparability"]}
        self.assertEqual(settings["lr_scheduler_type"]["actual"], "linear")
        self.assertEqual(settings["lr_scheduler_type"]["paper"], "cosine")
        self.assertEqual(settings["max_epochs"]["assessment"], "matched")
        self.assertEqual(settings["learning_rate"]["assessment"], "matched")
        self.assertIn("暂不判断是否符合预期", reports.validation_html(validation))

    def test_verified_same_dataset_comparison_assesses_only_minimum_signal(self):
        self.comparison_inputs()
        _, validation, _, _ = reports.build_reports(self.run)
        analysis = validation["expectation_analysis"]
        self.assertEqual(analysis["status"], "complete")
        self.assertTrue(analysis["metric_consistency"]["passed"])
        self.assertTrue(analysis["minimum_signal_checks_passed"])
        self.assertEqual(analysis["primary_k"], 10)
        self.assertAlmostEqual(analysis["baseline_comparison"]["popularity"]["differences"]["HR"], .099)
        self.assertEqual(analysis["validation_loss"]["difference"], -1)
        self.assertIn("推荐竞争力和论文复现仍无法判定", reports.validation_html(validation))

    def test_mismatched_baseline_hashes_or_missing_test_provenance_prevent_verdict(self):
        for key in ("train_file", "test_file", "sid_index_path", "missing_evaluated_test"):
            with self.subTest(key=key):
                baseline, evaluation = self.comparison_inputs()
                if key == "missing_evaluated_test":
                    evaluation["source_sha256"].pop("test_file")
                    self.save("evaluation/metrics.json", evaluation)
                else:
                    baseline["source_sha256"][key] = "d" * 64
                    self.save("comparison_baselines.json", baseline)
                _, validation, _, _ = reports.build_reports(self.run)
                self.assertEqual(validation["comparison"]["same_dataset_baselines"]["status"], "unavailable")
                self.assertIsNone(validation["expectation_analysis"]["minimum_signal_checks_passed"])
                self.assertNotIn("metrics", validation["comparison"]["same_dataset_baselines"])

    def test_consistency_diagnostics_detect_bad_cumulative_and_top1_metrics(self):
        for key, value, diagnostic in (("HR@3", .001, "HR_nondecreasing"),
                                        ("NDCG@3", .04, "NDCG_not_above_HR"),
                                        ("NDCG@1", .005, "HR1_equals_NDCG1")):
            with self.subTest(key=key):
                _, evaluation = self.comparison_inputs()
                evaluation["metrics"][key] = value
                self.save("evaluation/metrics.json", evaluation)
                _, validation, _, _ = reports.build_reports(self.run)
                analysis = validation["expectation_analysis"]
                self.assertFalse(analysis["metric_consistency"]["checks"][diagnostic])
                self.assertFalse(analysis["minimum_signal_checks_passed"])

    def test_missing_before_loss_prevents_claiming_all_learning_checks_pass(self):
        self.comparison_inputs()
        self.save("baseline_validation.json", {})
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertIsNone(validation["expectation_analysis"]["minimum_signal_checks_passed"])
        self.assertEqual(validation["expectation_analysis"]["signal_quality"], "loss_evidence_unavailable")
        self.assertIn("缺少可比的训练前后答案 token loss", reports.validation_html(validation))

    def test_paper_export_never_interpolates_or_compares_cross_domain_improvement(self):
        self.comparison_inputs()
        reports.generate(self.run, self.report_root, plots=False)
        directory = self.report_root / reports.VALIDATION_DIRECTORY
        report = json.loads((directory / "validation_report.json").read_text())
        self.assertFalse(report["comparison"]["directly_comparable"])
        paper_rows = [row for row in reports.paper_comparison_rows(report) if row["source"].startswith("Paper")]
        self.assertEqual(len(paper_rows), 6)
        self.assertEqual({row["k"] for row in paper_rows}, {3, 5, 10})
        self.assertEqual({row["dataset"] for row in paper_rows}, {"Industrial", "Office"})
        self.assertTrue(report["comparison"]["paper"]["source"]["url"].endswith("2510.24431v1"))
        with (directory / "baseline_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertTrue((directory / "expectation_analysis.json").is_file())
        self.assertIn("只发表 K=3、5、10", (directory / "index.html").read_text())

    def test_verified_shift_preserves_overall_metrics_and_qualifies_learning_claim(self):
        _, evaluation = self.behavior_shift_inputs()
        reports.generate(self.run, self.report_root, plots=False)
        directory = self.report_root / reports.VALIDATION_DIRECTORY
        validation = json.loads((directory / "validation_report.json").read_text())
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["evaluation"]["metrics"], evaluation["metrics"])
        self.assertTrue(validation["expectation_analysis"]["minimum_signal_checks_passed"])
        self.assertEqual(validation["behavior_shift_analysis"]["status"], "available")
        self.assertIn("事后探索性分层", reports.validation_html(validation))
        text = reports.validation_html(validation)
        self.assertIn("这不表示行为冷组推荐能力通过验证", text)
        self.assertIn("行为冷不等于模型完全未见商品", text)
        self.assertIn("复制上一商品的比例", text)
        self.assertIn("不筛选候选目录", text)
        self.assertNotIn("基本学习与评估检查符合预期", text)
        with (directory / "behavior_shift_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertTrue((directory / "behavior_shift_analysis.json").is_file())

    def test_shift_identity_mismatch_does_not_change_accepted_overall_result(self):
        for key in ("train_file", "test_file", "sid_index_path", "item_meta_path", "evaluation_metrics"):
            with self.subTest(key=key):
                diagnostic, _ = self.behavior_shift_inputs()
                diagnostic["source_sha256"][key] = "d" * 64
                self.save("behavior_shift_analysis.json", diagnostic)
                _, validation, _, _ = reports.build_reports(self.run)
                self.assertEqual(validation["status"], "passed")
                self.assertEqual(validation["behavior_shift_analysis"]["status"], "unavailable")
                self.assertFalse(reports.behavior_shift_rows(validation))

    def test_inconsistent_conditional_metrics_are_not_published_as_verified(self):
        diagnostic, _ = self.behavior_shift_inputs()
        diagnostic["strata"][1]["metrics"]["HR@10"] = .1
        self.save("behavior_shift_analysis.json", diagnostic)
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["behavior_shift_analysis"]["status"], "unavailable")
        self.assertTrue(any("reconstruct" in reason for reason in validation["behavior_shift_analysis"]["reasons"]))

    def test_shift_analysis_waits_for_accepted_evaluation(self):
        self.behavior_shift_inputs()
        self.save("evaluation/metrics.json", {})
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(validation["behavior_shift_analysis"]["status"], "pending")
        self.assertFalse(reports.behavior_shift_rows(validation))

    def test_malformed_optional_shift_schema_preserves_accepted_evaluation(self):
        for key, value in (("strata", None), ("strata", [None]), ("strata", [{"key": []}]),
                           ("shift_summary", []), ("prediction_diagnostics", [])):
            with self.subTest(key=key, value=value):
                diagnostic, _ = self.behavior_shift_inputs()
                diagnostic[key] = value
                self.save("behavior_shift_analysis.json", diagnostic)
                _, validation, _, _ = reports.build_reports(self.run)
                self.assertEqual(validation["status"], "passed")
                self.assertEqual(validation["behavior_shift_analysis"]["status"], "unavailable")

    def test_shift_overall_stratum_must_cover_every_accepted_test_row(self):
        diagnostic, _ = self.behavior_shift_inputs()
        diagnostic["strata"][0].update(rows=2, share=.5)
        self.save("behavior_shift_analysis.json", diagnostic)
        _, validation, _, _ = reports.build_reports(self.run)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["behavior_shift_analysis"]["status"], "unavailable")
        self.assertTrue(any("overall stratum" in reason for reason in validation["behavior_shift_analysis"]["reasons"]))

    def test_gpu_schema_and_source_hashes_preserve_actual_observations(self):
        self.save_records("gpu_metrics.jsonl", [
            {"timestamp": "2026-11-15T12:00:00Z", "gpus": [
                {"index": 0, "memory_used_mib": 1234, "utilization_gpu_pct": 80},
                {"index": 1, "memory_used_mib": 1600, "utilization_gpu_pct": 100}]}])
        training, _, _, gpu = reports.build_reports(self.run)
        self.assertEqual(training["gpu_peak_memory_mib"], 1600)
        self.assertEqual(training["gpu_mean_utilization_percent"], 90)
        self.assertEqual(gpu[1]["gpu_index"], 1)
        source = training["sources"][0]
        self.assertEqual(source["sha256"], hashlib.sha256((self.run / source["path"]).read_bytes()).hexdigest())

    def test_collection_preserves_archived_manifest_entries_and_links_without_old_files(self):
        self.report_root.mkdir()
        previous_index = "<!doctype html><main><a href='01_cleaning/old.pdf'>old archive</a></main>"
        (self.report_root / "index.html").write_text(previous_index)
        archive = {"path": "01_cleaning/old.pdf", "bytes": 20, "sha256": "old-checksum"}
        old_report = {"directory": "01_cleaning", "pdf": "01_cleaning/old.pdf", "pages": 3}
        (self.report_root / "reports_manifest.json").write_text(json.dumps({
            "validation": "passed", "files": [archive], "reports": [old_report], "custom": "retain"}))
        for _ in range(2):
            reports.generate(self.run, self.report_root, plots=False)
        manifest = json.loads((self.report_root / "reports_manifest.json").read_text())
        self.assertEqual(manifest["files"][0], archive)
        self.assertEqual(manifest["reports"][0], old_report)
        self.assertEqual(manifest["custom"], "retain")
        self.assertEqual(len(manifest["reports"]), 3)
        self.assertEqual(manifest["sft_status"], {"training": "pending", "validation": "pending"})
        text = (self.report_root / "index.html").read_text()
        self.assertIn("<a href='01_cleaning/old.pdf'>old archive</a>", text)
        self.assertEqual(text.count("<!-- SFT_REPORTS_START -->"), 1)
        self.assertFalse((self.report_root / "01_cleaning").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "report plotting dependencies not installed")
    def test_rendering_produces_real_pdf_and_png_even_for_pending_run(self):
        for status in ("pending", "passed"):
            with self.subTest(status=status):
                if status == "passed":
                    self.behavior_shift_inputs()
                    self.save("baseline_validation.json", {"loss": 14.15, "masked_perplexity": 1400000})
                reports.generate(self.run, self.report_root, plots=True)
                training_dir = self.report_root / reports.TRAINING_DIRECTORY
                validation_dir = self.report_root / reports.VALIDATION_DIRECTORY
                self.assertTrue((training_dir / "training-report.pdf").read_bytes().startswith(b"%PDF"))
                self.assertTrue((validation_dir / "validation-report.pdf").read_bytes().startswith(b"%PDF"))
                self.assertTrue((training_dir / "grad_norm.png").read_bytes().startswith(b"\x89PNG"))
                for name in ("ranking.png", "validation-comparison.png", "paper-reference.png"):
                    self.assertTrue((validation_dir / name).read_bytes().startswith(b"\x89PNG"))
                manifest = json.loads((validation_dir / "report_manifest.json").read_text())
                self.assertGreaterEqual(manifest["pages"], 7)
                self.assertEqual(manifest["status"], status)


if __name__ == "__main__":
    unittest.main()
