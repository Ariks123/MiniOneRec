"""Reports must preserve evaluation provenance and disclose the historical confound."""
import csv
from pathlib import Path
import tempfile
import unittest

from qwen_retrieval.report import atomic_json, build_report, comparison_rows, write_report


class RetrievalReportTests(unittest.TestCase):
    def test_reports_metrics_and_timing_without_claiming_a_clean_control(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "sft"
            record = {"status": "passed", "expected_rows": 2, "completed_rows": 2,
                      "metrics": {"HR@10": .2, "NDCG@10": .1},
                      "retrieval_metrics": {"Recall@10": .3, "Recall@200": .8},
                      "rerank_metrics": {"Recall@10": .2, "NDCG@10": .1},
                      "groups": {"cold": {"rows": 1}},
                      "timing": {"inference_seconds": 20, "prediction_seconds": 20,
                                 "query_seconds": 3, "retrieval_seconds": 1}}
            for name in ("evaluation_valid", "evaluation_test", "evaluation_beam", "evaluation_reference_beam"):
                atomic_json(run / name / "metrics.json", record)
            atomic_json(run / "training_summary.json", {"status": "completed"})
            atomic_json(run / "smoke/smoke_result.json", {"status": "passed"})
            atomic_json(run / "isolation_verification.json", {"status": "passed"})
            atomic_json(run / "final_checkpoint/config.json", {})
            (run / "final_checkpoint/model.safetensors").write_bytes(b"fixture")
            atomic_json(run / "item_index/manifest.json", {"status": "passed"})
            for name in ("item_vectors.npy", "row_to_items.json"):
                (run / "item_index" / name).write_bytes(b"fixture")
            atomic_json(run / "stage_timings.json", {"beam": {"wall_seconds": 30}})
            report = write_report(run, Path(folder) / "reports")
            self.assertEqual(report["status"], "passed")
            row = next(row for row in report["comparison"] if row["method"] == "joint_retrieval_gr_rerank" and row["split"] == "test")
            self.assertEqual(row["Recall@10"], .2)
            self.assertAlmostEqual(row["delta_Recall@10_vs_historical"], .2 - .12949481579527908)
            retrieved = next(row for row in report["comparison"] if row["method"] == "joint_vector_retrieval" and row["split"] == "test")
            self.assertEqual(retrieved["Recall@200"], .8)
            self.assertEqual(retrieved["inference_seconds"], 4)
            self.assertIn("components", retrieved["timing_scope"])
            self.assertTrue(any("DeepSpeed" in note and "cannot isolate" in note for note in report["interpretation"]))
            self.assertTrue(any("No fresh generation-only" in note for note in report["interpretation"]))
            html = (Path(folder) / "reports/index.html").read_text()
            self.assertIn("Interaction cold/warm", html)
            with (Path(folder) / "reports/comparison.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 7)
            record["completed_rows"] = 1
            atomic_json(run / "evaluation_test/metrics.json", record)
            self.assertEqual(build_report(run)["status"], "incomplete")

    def test_missing_metrics_are_pending_and_never_zero_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            report = build_report(folder)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["comparison"][1]["status"], "pending")
            self.assertIsNone(report["comparison"][1]["Recall@10"])
            self.assertIsNone(report["comparison"][1]["inference_seconds"])

    def test_failed_supervisor_cannot_produce_passed_report(self):
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder) / "run_status.json", {"status": "failed", "error": "<unsafe>"})
            report = write_report(folder, Path(folder) / "reports")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["supervisor"]["error"], "<unsafe>")


if __name__ == "__main__":
    unittest.main()
