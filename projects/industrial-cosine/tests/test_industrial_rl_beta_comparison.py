"""Beta-only provenance, honest reward plots, and validation-only selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_industrial_rl_beta as comparison


class IndustrialRLBetaComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-beta-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.left = self.make_run("baseline", .001, .09)
        self.right = self.make_run("candidate", .01, .085)

    def make_run(self, label, beta, score):
        run = self.root / label
        run.mkdir()
        manifest = run / "rl_task_manifest.jsonl"
        manifest.write_text('{"sample_id":"same-training-row"}\n')
        parent = {"model_path": "/shared/sft/final_checkpoint", "model_sha256": "same-parent-weight"}
        config = {"stage": "rl", "resume_from_checkpoint": None, "parent_sft": parent,
                  "model": {"source": parent["model_path"], "identity": {"files": {
                      "model.safetensors": {"sha256": parent["model_sha256"]}}}},
                  "data": {"category": comparison.reports.DATASET, "train_samples": 52775,
                      "task_manifest": {"path": str(manifest), "sha256": comparison.reports.sha256(manifest), "rows": 57307}},
                  "task_counts": {"title_history_to_sid": 10000, "sid_sequence_recommendation": 36259},
                  "hyperparameters": {"beta": beta, "num_generations": 16, "learning_rate": 1e-5,
                      "model_selection_metric": "eval_ndcg10", "max_grad_norm": .3, "ref_model_sync_steps": 512},
                  "runtime_compatibility": {"applied": True, "installed_package_modified": False,
                      "version": "0.18.0", "original_source_sha256": "original", "patched_source_sha256": "patched",
                      "method": "process-local-accumulation-fix"}}
        supervisor = {"beta": beta, "image": "sha256:same-runtime", "parent_runtime_image_id": "sha256:same-runtime",
                      "parent_model_sha256": parent["model_sha256"],
                      "source_sha256": {key: "same-code-" + key for key in comparison.TRAINING_SOURCES}}
        strata = []
        for group, count in comparison.EXPECTED_COUNTS.items():
            # The candidate has stronger test metrics but weaker validation;
            # the recommendation must still retain the baseline.
            hits = 1 if label == "baseline" else 2
            metrics = {f"{metric}@{k}": hits / count for metric in ("HR", "NDCG") for k in comparison.reports.TOP_K}
            strata.append({"key": group, "rows": count, "hits_by_k": {str(k): hits for k in comparison.reports.TOP_K},
                           "metrics": metrics})
        steps = [{"step": step, "epoch": step / 100, "reward": .001, "rewards/rule_reward": .02,
                  "kl": (4 if label == "baseline" else .1), "grad_norm": .6,
                  "hit_group_rate": .32, "eval_ndcg10": score} for step in (1, 2)]
        return {"run_dir": str(run), "config": config, "supervisor": supervisor,
                "summary": {"status": "completed", "best_metric": score, "selection_metric": "eval_ndcg10",
                    "final_checkpoint_contains_best_model": True, "best_checkpoint": str(run / "checkpoint-1"),
                    "global_step": 2, "epoch": .02, "stop_reason": "max_epochs", "train_metrics": {"train_runtime": 10}},
                "state": {"status": "completed", "runtime_image_id": "sha256:same-runtime",
                          "runtime_versions": {"torch": "2.8.0+cu128", "devices": ["H20"] * 4}},
                "validation": {"status": "passed", "strata": strata, "source_sha256": {
                    "test_file": "same-test", "train_file": "same-train", "predictions": "different-" + label}},
                "steps": steps, "input_sha256": {"manifest": comparison.reports.sha256(manifest)}}

    def build(self):
        with patch.object(comparison, "load_run", side_effect=[self.left, self.right]):
            return comparison.build_comparison(self.left["run_dir"], self.right["run_dir"])

    def test_beta_only_pair_is_accepted_and_recorded(self):
        result = comparison.verify_pair(self.left, self.right)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["only_changed_training_hyperparameter"], "beta")
        self.assertEqual(result["expected_test_strata"]["behavior_cold"], 147)

    def test_supervisor_cannot_claim_point01_when_training_was_point001(self):
        self.right["config"]["hyperparameters"]["beta"] = .001
        with self.assertRaisesRegex(ValueError, "actual training beta"):
            comparison.verify_pair(self.left, self.right)

    def test_other_hyperparameters_must_remain_identical(self):
        self.right["config"]["hyperparameters"]["learning_rate"] = 5e-6
        with self.assertRaisesRegex(ValueError, "beyond beta"):
            comparison.verify_pair(self.left, self.right)

    def test_training_and_evaluation_code_drift_is_rejected(self):
        self.right["supervisor"]["source_sha256"]["minionerec_trainer.py"] = "different"
        with self.assertRaisesRegex(ValueError, "code differs"):
            comparison.verify_pair(self.left, self.right)

    def test_manifest_tampering_is_rejected(self):
        (Path(self.right["run_dir"]) / "rl_task_manifest.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            comparison.verify_pair(self.left, self.right)

    def test_unfixed_or_different_runtime_is_rejected(self):
        self.right["config"]["runtime_compatibility"]["applied"] = False
        with self.assertRaisesRegex(ValueError, "accumulation correction"):
            comparison.verify_pair(self.left, self.right)
        self.right["config"]["runtime_compatibility"]["applied"] = True
        self.right["supervisor"]["image"] = "sha256:other"
        with self.assertRaisesRegex(ValueError, "image identity"):
            comparison.verify_pair(self.left, self.right)

    def test_missing_cold_rows_is_rejected(self):
        self.right["validation"]["strata"][-1]["rows"] = 146
        with self.assertRaisesRegex(ValueError, "complete Industrial test coverage"):
            comparison.verify_pair(self.left, self.right)

    def test_validation_selects_even_when_candidate_test_is_better(self):
        report, _, _ = self.build()
        self.assertEqual(report["recommendation"]["recommended_arm"], "baseline")
        self.assertFalse(report["recommendation"]["test_used_for_selection"])
        row = next(row for row in report["rows"] if row["stratum"] == "overall" and row["k"] == 10)
        self.assertGreater(row["delta_recall_pp"], 0)
        self.assertEqual(row["candidate_hits"], 2)
        self.assertTrue(report["expectation_assessment"]["matched_step_kl_decreased"])
        self.assertFalse(report["expectation_assessment"]["validation_and_test10_non_decreasing"])

    def test_matched_step_statistics_do_not_compare_different_training_lengths(self):
        self.left["steps"].append({"step": 3, "reward": .1, "kl": 9999, "grad_norm": 5})
        report, _, _ = self.build()
        self.assertEqual(report["common_training_steps"]["last"], 2)
        self.assertEqual(report["baseline"]["matched_step_statistics"]["kl"]["mean"], 4)
        self.assertGreater(report["baseline"]["all_training_statistics"]["kl"]["mean"], 100)

    def test_invalid_gradient_updates_are_excluded_from_statistics(self):
        rows = [{"grad_norm": .7}, {"grad_norm": 100, "step_skipped": True},
                {"grad_norm": 999, "optimizer_overflow": True}]
        stats = comparison.telemetry_stats(rows, .3)["grad_norm"]
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["mean"], .7)
        self.assertEqual(stats["clip_exceedance_rate"], 1)

    def test_outputs_are_separate_from_inputs_and_reward_definitions_are_visible(self):
        with self.assertRaisesRegex(ValueError, "outside every input"):
            comparison.generate(self.left["run_dir"], self.right["run_dir"], Path(self.right["run_dir"]) / "report", plots=False)
        destination = self.root / "output"
        with patch.object(comparison, "load_run", side_effect=[self.left, self.right]):
            result = comparison.generate(self.left["run_dir"], self.right["run_dir"], destination, plots=False)
        self.assertEqual(result["status"], "passed")
        report = json.loads((destination / "comparison.json").read_text())
        self.assertEqual(report["kind"], comparison.KIND)
        self.assertEqual(report["rendered_plots"], [])
        self.assertIn("Combined reward", (destination / "index.html").read_text())
        self.assertIn("147", (destination / "comparison.html").read_text())
        self.assertIn("candidate_hits", (destination / "comparison.csv").read_text().splitlines()[0])

    def test_incomplete_run_fails_before_writing_passed_report(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "Incomplete beta comparison inputs"):
            comparison.generate(missing, self.right["run_dir"], self.root / "out", plots=False)
        self.assertFalse((self.root / "out/comparison.json").exists())

    def test_telemetry_loader_excludes_reloaded_best_evaluation(self):
        run = Path(self.left["run_dir"])
        (run / "monitor").mkdir()
        (run / "monitor/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
            {"kind": "train", "step": 1, "reward": .01},
            {"kind": "eval", "phase": "training", "step": 1, "eval_ndcg10": .08},
            {"kind": "eval", "phase": "final_validation", "step": 1, "eval_ndcg10": .12})))
        errors = []
        steps = comparison.reports.observations(run, errors)
        self.assertEqual(errors, [])
        self.assertEqual(steps[0]["eval_ndcg10"], .08)


if __name__ == "__main__":
    unittest.main()
