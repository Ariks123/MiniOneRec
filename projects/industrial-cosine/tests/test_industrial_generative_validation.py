"""Protocol, coverage, and Trainer integration checks for validation ranking."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import industrial_generative_validation as ranking


class IndustrialGenerativeValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="industrial-ranking-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.valid, self.sid = self.root / "valid.csv", self.root / "index.json"
        self.catalog = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(50)}
        self.catalog["50"] = self.catalog["0"][:]
        self.sid.write_text(json.dumps(self.catalog))
        with self.valid.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("item_id", "item_sid"))
            writer.writeheader()
            for item in ("0", "50"):
                writer.writerow({"item_id": item, "item_sid": "".join(self.catalog[item])})
        self.manifest = {"validation": "passed", "dataset": ranking.CATEGORY, "original_sid": True,
                         "outputs": {"valid.csv": {"sha256": ranking.sha256_file(self.valid)}},
                         "rows": {"valid": 2}}
        (self.root / "preparation_manifest.json").write_text(json.dumps(self.manifest))
        self.config = ranking.GenerativeValidationConfig(
            str(self.valid), str(self.sid), expected_sid_sha256=ranking.sha256_file(self.sid),
            expected_valid_sha256=ranking.sha256_file(self.valid), expected_rows=2)

    def test_test_split_and_different_generation_protocol_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid.csv"):
            ranking.GenerativeValidationConfig(str(self.root / "test.csv"), str(self.sid))
        with self.assertRaisesRegex(ValueError, "beam50"):
            ranking.GenerativeValidationConfig(str(self.valid), str(self.sid), num_beams=10)
        with self.assertRaisesRegex(ValueError, "Industrial"):
            ranking.GenerativeValidationConfig(str(self.valid), str(self.sid), category="Video_Games")

    def test_original_catalog_collisions_remain_valid_sid_level_targets(self):
        artifacts = ranking.validation_artifacts(self.config)
        self.assertEqual(artifacts["catalog_items"], 51)
        self.assertEqual(artifacts["unique_sids"], 50)
        self.assertEqual(artifacts["targets"][0], artifacts["targets"][1])
        self.assertEqual(artifacts["source_sha256"]["valid_file"], self.config.expected_valid_sha256)
        self.assertNotIn("test_file", artifacts["source_paths"])

    def test_changed_prepared_input_or_catalog_cannot_be_used_for_selection(self):
        self.valid.write_text(self.valid.read_text() + "0,<a_0><b_0><c_0>\n")
        with self.assertRaisesRegex(ValueError, "Validation CSV differs"):
            ranking.validation_artifacts(self.config)
        self.valid.write_text("item_id,item_sid\n0,<a_0><b_0><c_0>\n50,<a_0><b_0><c_0>\n")
        self.manifest["outputs"]["valid.csv"]["sha256"] = ranking.sha256_file(self.valid)
        (self.root / "preparation_manifest.json").write_text(json.dumps(self.manifest))
        self.config.expected_valid_sha256 = ranking.sha256_file(self.valid)
        self.sid.write_text(json.dumps({**self.catalog, "new": ["<a_999>", "<b_0>", "<c_0>"]}))
        with self.assertRaisesRegex(ValueError, "SID catalog differs"):
            ranking.validation_artifacts(self.config)

    def test_validation_row_count_and_original_catalog_marker_are_required(self):
        self.config.expected_rows = 3
        with self.assertRaisesRegex(ValueError, "row count"):
            ranking.validation_artifacts(self.config)
        self.config.expected_rows = 2
        self.manifest["original_sid"] = False
        (self.root / "preparation_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "original Industrial"):
            ranking.validation_artifacts(self.config)

    def test_sharding_covers_uneven_splits_and_empty_rank_without_padding(self):
        for n, world in ((7, 4), (3, 4), (4532, 4)):
            shards = [ranking.rank_row_ids(n, rank, world) for rank in range(world)]
            self.assertEqual(ranking.validate_coverage(shards, n), {
                "expected_rows": n, "completed_rows": n, "missing_rows": 0, "duplicate_rows": 0})
        with self.assertRaisesRegex(ValueError, "coverage"):
            ranking.validate_coverage([[0, 1], [1, 2]], 3)
        with self.assertRaisesRegex(ValueError, "coverage"):
            ranking.validate_coverage([[0], [2]], 3)

    def test_all_k_ranking_metrics_count_exact_targets_and_reject_invalid_candidates(self):
        sids = [f"<a_{i}><b_0><c_0>" for i in range(60)]
        stats = ranking.RankingAccumulator(sids, 50)
        for target in (sids[0], sids[9], sids[59]):
            stats.add({"row_id": stats.rows, "target": target, "predictions": sids[:50]})
        result = stats.result()["metrics"]
        self.assertEqual(result["HR@1"], 1 / 3)
        self.assertEqual(result["HR@10"], 2 / 3)
        self.assertEqual(result["HR@50"], 2 / 3)
        self.assertAlmostEqual(result["NDCG@10"], (1 + 1 / math.log2(11)) / 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stats.add({"target": sids[0], "predictions": [sids[0]] * 50})
        with self.assertRaisesRegex(ValueError, "invalid"):
            stats.add({"target": sids[0], "predictions": sids[:49] + ["bad"]})

    def fake_trainer(self, num_samples=2):
        class BaseTrainer:
            def __init__(self, **kwargs):
                self.args = SimpleNamespace(output_dir=str(self_root))
                self.state = SimpleNamespace(global_step=10, epoch=1.0)

            def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
                return SimpleNamespace(num_samples=num_samples, metrics={metric_key_prefix + "_loss": 1.25})

            def is_world_process_zero(self):
                return True
        self_root = self.root
        return ranking.make_generative_trainer_class(BaseTrainer)(generative_validation=self.config)

    def report(self, ndcg=0.2):
        return {"coverage": {"expected_rows": 2, "completed_rows": 2},
                "metrics": {"ndcg10": ndcg, "recall10": 0.5, "ndcg50": ndcg, "recall50": 0.5},
                "prediction_directory": str(self.root / "validation_ranking/step-fixture")}

    def test_trainer_returns_ranking_metrics_on_eval_and_freshly_ranks_best_reloaded_model(self):
        trainer = self.fake_trainer()
        first, second = self.report(0.2), self.report(0.4)
        with patch.object(ranking, "generate_validation", side_effect=[first, second]) as generate:
            before = trainer.evaluation_loop(None)
            after = trainer.evaluation_loop(None)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(before.metrics["eval_loss"], 1.25)
        self.assertEqual(before.metrics["eval_ndcg10"], 0.2)
        self.assertEqual(after.metrics["eval_ndcg10"], 0.4)
        self.assertEqual(after.metrics["eval_recall10"], 0.5)
        self.assertEqual(after.metrics["eval_generation_rows"], 2)
        latest = json.loads((self.root / "validation_ranking/latest_metrics.json").read_text())
        self.assertEqual(latest["metrics"]["ndcg10"], 0.4)

    def test_trainer_rejects_ranking_and_teacher_forced_validation_row_mismatch(self):
        trainer = self.fake_trainer(num_samples=3)
        with patch.object(ranking, "generate_validation", return_value=self.report()):
            with self.assertRaisesRegex(ValueError, "different rows"):
                trainer.evaluation_loop(None)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                         "Torch/NumPy are optional in the CPU-only workspace shell")
    def test_generation_guard_restores_python_numpy_and_torch_rng(self):
        import numpy as np
        import torch
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        with ranking.preserve_rng_state(torch.device("cpu")):
            random.random()
            np.random.rand()
            torch.rand(3)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
