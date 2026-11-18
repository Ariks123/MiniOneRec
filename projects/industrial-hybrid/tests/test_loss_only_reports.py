"""Loss-only runs must not publish pending or fabricated ranking results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import report_industrial_sft as reports


class LossOnlyReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-loss-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.data, self.inputs, self.output = [self.root / name
                                                      for name in ("run", "data", "inputs", "reports")]
        for path in (self.run, self.data, self.inputs):
            path.mkdir()
        self.save(self.run / "run_config.json", {
            "model": {"source": "/weights/Qwen2.5-0.5B"},
            "data": {"category": reports.DATASET, "validation_samples": 4532},
            "hyperparameters": {"lr_scheduler_type": "linear", "max_epochs": 3}})
        self.save(self.run / "run_status.json", {
            "status": "completed", "stage": "completed", "parameters": {"final_ranking_eval": False}})
        self.save(self.run / "training_summary.json", {
            "status": "completed", "selection_metric": "eval_loss", "best_metric": 1.5,
            "best_checkpoint": str(self.run / "checkpoint-100"),
            "final_eval_metrics": {"eval_loss": 1.5}, "stop_reason": "max_epochs"})
        self.save(self.run / "evaluation/metrics.json", {
            "status": "not_requested", "scope": "no_ranking_requested",
            "reason": "Ranking metrics disabled by user; validation loss only"})
        self.save(self.data / "preparation_manifest.json", {"rows": {"valid": 4532}})
        path = self.run / "monitor/metrics.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps({"step": 1, "kind": "train", "loss": 2,
                                    "grad_norm": .5, "learning_rate": .0003}) + "\n")

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_completed_loss_training_is_distinct_from_unrequested_ranking(self):
        with patch.object(reports, "validated_evaluation", side_effect=AssertionError("ranking must not run")):
            training, validation, _ = reports.build_reports(self.run, self.data, self.inputs)
        self.assertEqual(training["status"], "completed")
        self.assertFalse(validation["ranking_requested"])
        self.assertEqual(validation["status"], "not_requested")
        self.assertEqual(validation["scope"], "no_ranking_requested")
        self.assertEqual(validation["strata"], [])
        self.assertEqual(validation["validation_samples"], 4532)
        self.assertEqual(validation["validation_loss"], 1.5)
        self.assertEqual(validation["selection_metric"], "eval_loss")
        self.assertIsNone(validation["comparison_reference"])
        self.assertNotIn("metrics", validation["evaluation"])

    def test_explicit_disabled_flag_overrides_stale_ranking_artifacts(self):
        self.save(self.run / "evaluation/metrics.json", {"status": "passed", "metrics": {"HR@10": .9}})
        self.save(self.run / "ranking_selection.json", {"primary_metric": "ndcg10", "best_ndcg": .9})
        self.save(self.run / "evaluation_recall/metrics.json", {"status": "passed", "metrics": {"HR@10": .9}})
        with patch.object(reports, "validated_evaluation", side_effect=AssertionError("stale ranking must not run")):
            training, validation, _ = reports.build_reports(self.run, self.data, self.inputs)
        self.assertEqual(validation["status"], "not_requested")
        self.assertEqual(training["ranking_selection"], {})
        self.assertEqual(training["recall_evaluation"], {})
        self.assertNotIn("metrics", validation["evaluation"])

    def test_placeholder_disables_ranking_even_without_supervisor_parameters(self):
        self.save(self.run / "run_status.json", {"status": "completed", "stage": "completed"})
        with patch.object(reports, "validated_evaluation", side_effect=AssertionError("ranking must not run")):
            _, validation, _ = reports.build_reports(self.run, self.data, self.inputs)
        self.assertEqual(validation["status"], "not_requested")

    def test_live_loss_falls_back_to_latest_validation_observation(self):
        self.save(self.run / "training_summary.json", {})
        with (self.run / "monitor/metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": 2, "kind": "eval", "eval_loss": 1.7}) + "\n")
            stream.write(json.dumps({"step": 3, "kind": "eval", "eval_loss": 1.6}) + "\n")
        _, validation, _ = reports.build_reports(self.run, self.data, self.inputs)
        self.assertEqual(validation["validation_loss"], 1.6)
        self.assertEqual(validation["status"], "not_requested")

    def test_html_has_loss_results_without_ranking_tables_or_reference_scores(self):
        self.output.mkdir()
        (self.output / "validation_ranking.png").write_bytes(b"stale")
        result = reports.generate(self.run, self.data, self.inputs, self.output, plots=False)
        page = (self.output / "index.html").read_text()
        self.assertEqual(result["training_status"], "completed")
        self.assertEqual(result["validation_status"], "not_requested")
        self.assertIn("所选模型的验证 loss", page)
        self.assertIn("排名评估未请求", page)
        self.assertNotIn("完整测试结果尚未通过", page)
        self.assertNotIn("HR@", page)
        self.assertNotIn("NDCG", page)
        self.assertNotIn("SFT-only 对照", page)
        self.assertNotIn("beam=50", page)
        self.assertNotIn("validation_ranking.png", page)
        self.assertFalse((self.output / "conditional_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
