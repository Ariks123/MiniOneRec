"""Verify that this branch retains the complete selected Industrial cosine run."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class SelectedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = read_json(ROOT / "config/selected_run.json")
        cls.config = read_json(ROOT / "results/cosine/run_config.json")
        cls.summary = read_json(ROOT / "results/cosine/training_summary.json")
        cls.evaluation = read_json(ROOT / "results/cosine/evaluation_metrics.json")

    def test_retained_run_is_industrial_2018_with_cosine_and_selected_base(self):
        self.assertEqual(self.config["stage"], "sft")
        self.assertEqual(self.config["data"]["category"], "Industrial_and_Scientific")
        self.assertIn("/data/Amazon18/", self.config["data"]["train_file"])
        self.assertEqual(Path(self.config["model"]["source"]).name, "Qwen2.5-1.5B")
        self.assertFalse(self.config["model"]["train_from_scratch"])
        self.assertEqual(self.config["hyperparameters"]["lr_scheduler_type"], "cosine")
        self.assertEqual(self.selected["selection"]["lr_scheduler_type"], "cosine")
        self.assertEqual(self.selected["selection"]["model"], "Qwen2.5-1.5B Base")
        self.assertEqual(self.selected["hyperparameters"], self.config["hyperparameters"])

    def test_final_checkpoint_contains_validation_selected_best_model(self):
        selected = self.selected["selection"]
        self.assertEqual(self.summary["status"], "completed")
        self.assertEqual(self.summary["global_step"], self.summary["planned_max_steps"])
        self.assertEqual(self.summary["epoch"], 3)
        self.assertTrue(self.summary["final_checkpoint_contains_best_model"])
        self.assertEqual(Path(self.summary["best_checkpoint"]).name, "checkpoint-3572")
        self.assertEqual(selected["selection_metric"], "eval_loss")
        for key in ("best_checkpoint", "final_checkpoint", "best_metric",
                    "final_checkpoint_contains_best_model"):
            self.assertEqual(selected[key], self.summary[key])
        self.assertEqual(self.evaluation["model_path"], self.summary["final_checkpoint"])

    def test_evaluation_covers_all_original_test_rows_with_valid_candidates(self):
        evaluation = self.evaluation
        self.assertEqual(evaluation["status"], "passed")
        self.assertEqual(evaluation["category"], "Industrial_and_Scientific")
        self.assertEqual(evaluation["scope"], "complete_original_test_split")
        self.assertEqual(evaluation["coverage"], {
            "expected_rows": 4533, "completed_rows": 4533,
            "missing_rows": 0, "duplicate_rows": 0,
        })
        self.assertEqual(evaluation["expected_rows"], 4533)
        self.assertEqual(evaluation["completed_rows"], 4533)
        self.assertEqual(evaluation["num_beams"], 50)
        self.assertEqual(evaluation["prompt_style"], "original-eval")
        self.assertEqual(evaluation["CC"], 0)
        self.assertEqual(evaluation["validity_rate"], 1)
        self.assertEqual(evaluation["unique_rate"], 1)
        self.assertTrue(evaluation["check_results"])
        self.assertEqual(set(evaluation["check_results"].values()), {"passed"})
        self.assertEqual(evaluation["source_sha256"]["sid_index_path"],
                         self.config["data"]["sha256"]["sid_index_path"])
        for key in ("status", "scope", "expected_rows", "completed_rows", "coverage", "metrics"):
            self.assertEqual(self.selected["evaluation"][key], evaluation[key])

    def test_recall_metrics_are_the_retained_higher_recall_run(self):
        metrics = self.evaluation["metrics"]
        self.assertAlmostEqual(metrics["HR@10"], 0.12618574895212883)
        self.assertAlmostEqual(metrics["HR@50"], 0.21950143392896537)
        self.assertAlmostEqual(metrics["NDCG@10"], 0.08536859997506724)
        self.assertAlmostEqual(metrics["NDCG@50"], 0.10597144041917358)
        self.assertEqual(set(metrics), {
            f"{metric}@{k}" for metric in ("HR", "NDCG") for k in (1, 3, 5, 10, 20, 50)
        })


if __name__ == "__main__":
    unittest.main()
