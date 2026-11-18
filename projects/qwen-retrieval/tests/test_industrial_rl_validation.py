import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import types
from collections import defaultdict, namedtuple
import unittest
from unittest.mock import patch

from scripts.industrial_rl_validation import (
    FinalStepEvaluationCallback, RLMonitorCallback, make_rl_trainer_class, ranking_metrics, run_initial_baseline,
)


def report():
    return {"status": "passed", "coverage": {"expected_rows": 4532, "completed_rows": 4532,
            "missing_rows": 0, "duplicate_rows": 0}, "metrics": {"ndcg10": .1, "recall10": .2}}


class RankingTests(unittest.TestCase):
    def test_ranking_selection_has_no_grpo_loss(self):
        value = report()
        value["metrics"]["eval_loss"] = -9
        self.assertEqual(ranking_metrics(value), {"eval_ndcg10": .1, "eval_recall10": .2,
                                                 "eval_generation_rows": 4532})

    def test_rejects_incomplete_or_nonfinite_ranking(self):
        for key in ("completed_rows", "missing_rows", "duplicate_rows"):
            value = report()
            value["coverage"][key] += 1
            with self.assertRaises(ValueError):
                ranking_metrics(value)
        value = report()
        value["metrics"]["ndcg10"] = float("nan")
        with self.assertRaises(ValueError):
            ranking_metrics(value)

    def test_baseline_does_not_call_trainer_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = SimpleNamespace(state=SimpleNamespace(global_step=0),
                                      args=SimpleNamespace(output_dir=folder), base_model="parent-sft",
                                      is_world_process_zero=lambda: True)
            with patch("scripts.industrial_rl_validation.generate_validation", return_value=report()):
                run_initial_baseline(trainer)
            value = json.loads((Path(folder) / "baseline_validation.json").read_text())
            self.assertEqual(value["role"], "sft_baseline")
            self.assertEqual(value["model_path"], "parent-sft")
            trainer.state.global_step = 1
            with self.assertRaises(ValueError):
                run_initial_baseline(trainer)

    def test_final_step_includes_live_weights_and_respects_safety_stop(self):
        monitor = SimpleNamespace(abort_reason=None)
        callback = FinalStepEvaluationCallback(monitor)
        control = SimpleNamespace(should_evaluate=False, should_save=False)
        state = SimpleNamespace(global_step=1650, max_steps=1650)
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_evaluate and control.should_save)
        monitor.abort_reason = "nonfinite"
        control.should_evaluate = control.should_save = False
        callback.on_step_end(None, state, control)
        self.assertFalse(control.should_evaluate or control.should_save)

    def test_monitor_preserves_reward_and_ranking_without_eval_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = RLMonitorCallback(folder)
            monitor.primary = True
            state = SimpleNamespace(global_step=1, epoch=.1, max_steps=10,
                                    best_model_checkpoint=None, best_metric=None)
            control = SimpleNamespace()
            monitor.on_log(None, state, control, logs={"loss": .01, "grad_norm": .2,
                "learning_rate": 1e-5, "reward": .3, "kl": .02, "zero_reward_group_rate": .7,
                "task/title2sid/reward": .4})
            monitor.on_log(None, state, control, logs={"eval_ndcg10": .1, "eval_recall10": .2})
            records = [json.loads(line) for line in monitor.metrics_path.read_text().splitlines()]
            self.assertEqual(records[0]["task/title2sid/reward"], .4)
            self.assertEqual(records[1]["kind"], "eval")
            self.assertNotIn("eval_loss", records[1])

    def test_factory_skips_rl_evaluation_and_keeps_training_rewards_out_of_validation(self):
        class Base:
            def __init__(self, folder):
                self.args = SimpleNamespace(output_dir=folder)
                self.state = SimpleNamespace(best_model_checkpoint="checkpoint-1")
                self._metrics = defaultdict(list, reward=[.4], kl=[.01])

            def is_world_process_zero(self):
                return True

            def evaluation_loop(self, *args, **kwargs):
                raise AssertionError("RL rollout evaluation must not run")

            def evaluate(self):
                value = self.evaluation_loop(None)
                self.logged_pending = dict(self._metrics)
                return value.metrics

        utils = types.ModuleType("transformers.trainer_utils")
        utils.EvalLoopOutput = namedtuple("EvalLoopOutput", "predictions label_ids metrics num_samples")
        with tempfile.TemporaryDirectory() as folder:
            trainer = make_rl_trainer_class(Base)(folder, generative_validation=SimpleNamespace(expected_rows=4532))
            trainer.rl_validation_phase = "selected_model"
            value = {**report(), "prediction_directory": str(Path(folder) / "ranking")}
            with patch.dict(sys.modules, {"transformers.trainer_utils": utils}), patch(
                    "scripts.industrial_rl_validation.generate_validation", return_value=value):
                metrics = trainer.evaluate()
            self.assertEqual(metrics["eval_ndcg10"], .1)
            self.assertEqual(trainer.logged_pending, {})
            self.assertEqual(trainer._metrics["reward"], [.4])
            saved = json.loads((Path(folder) / "validation_ranking/latest_metrics.json").read_text())
            self.assertEqual(saved["weight_role"], "selected_best")
            self.assertEqual(saved["evaluated_checkpoint"], "checkpoint-1")


if __name__ == "__main__":
    unittest.main()
