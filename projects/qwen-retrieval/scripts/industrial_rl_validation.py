"""Ranking-only validation and durable monitoring for Industrial GRPO.

GRPO evaluation loss is an advantage/KL objective, not teacher-forced validation
cross entropy.  Checkpoint selection therefore uses the existing full-catalog
validation generator without entering ReReTrainer's rollout evaluation loop.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from scripts.industrial_generative_validation import generate_validation, validation_artifacts
from sft_monitor import SFTMonitorCallback, TrainerCallback, atomic_json, finite_number, json_value, utc_now


def ranking_metrics(report, prefix="eval"):
    if report.get("status") != "passed":
        raise ValueError("RL validation ranking has not passed")
    coverage = report.get("coverage", {})
    if (not coverage.get("expected_rows") or
            coverage.get("completed_rows") != coverage.get("expected_rows") or
            coverage.get("missing_rows") != 0 or coverage.get("duplicate_rows") != 0):
        raise ValueError("RL validation ranking is incomplete")
    values = {key: value for key, value in report.get("metrics", {}).items()
              if key.startswith(("ndcg", "recall"))}
    if "ndcg10" not in values or any(finite_number(value) is None for value in values.values()):
        raise ValueError("RL validation requires finite NDCG@10 and ranking metrics")
    result = {prefix + "_" + key: float(value) for key, value in values.items()}
    result[prefix + "_generation_rows"] = coverage["completed_rows"]
    return result


def run_initial_baseline(trainer):
    """Generate the unmodified SFT baseline without initializing a DS eval engine.

    Call before train(), with the policy already placed on its rank-local GPU.
    No Trainer evaluation callbacks fire, so this cannot set RL's best checkpoint
    or consume its early-stopping patience.
    """
    if int(trainer.state.global_step) != 0:
        raise ValueError("The SFT baseline must be evaluated before RL updates")
    report = generate_validation(trainer)
    ranking_metrics(report)
    report = {**report, "role": "sft_baseline", "model_path": getattr(trainer, "base_model", None)}
    if trainer.is_world_process_zero():
        atomic_json(Path(trainer.args.output_dir) / "baseline_validation.json", report)
    return report


def make_rl_trainer_class(base_trainer):
    class IndustrialRLValidationTrainer(base_trainer):
        def __init__(self, *args, generative_validation, **kwargs):
            self.generative_validation = generative_validation
            self._generation_artifacts = self._generation_data = self._generation_result = None
            self.validation_weight_role = "training_weights"
            super().__init__(*args, **kwargs)

        def _validation_artifacts(self):
            if self._generation_artifacts is None:
                self._generation_artifacts = validation_artifacts(self.generative_validation)
            return self._generation_artifacts

        def evaluate(self, *args, **kwargs):
            # ReReTrainer.log prefixes its pending rollout metrics with eval_.
            # Keep training rewards out of validation logs, including the log()
            # call made by Trainer.evaluate after evaluation_loop returns.
            pending = self._metrics
            self._metrics = defaultdict(list)
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                self._metrics = pending

        def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
            from transformers.trainer_utils import EvalLoopOutput

            report = generate_validation(self)
            metrics = ranking_metrics(report, metric_key_prefix)
            expected = self.generative_validation.expected_rows
            rows = report["coverage"]["completed_rows"]
            if expected is not None and rows != expected:
                raise ValueError("RL validation ranking covers a different validation split")
            phase = getattr(self, "rl_validation_phase", "training")
            selected = phase == "selected_model" or self.validation_weight_role == "selected_best"
            report["phase"] = phase
            report["weight_role"] = "selected_best" if selected else self.validation_weight_role
            report["evaluated_checkpoint"] = (getattr(self.state, "best_model_checkpoint", None)
                                               if selected else None)
            report["selected_checkpoint"] = report["evaluated_checkpoint"]
            self._generation_result = report
            if self.is_world_process_zero():
                atomic_json(Path(report["prediction_directory"]) / "metrics.json", report)
                atomic_json(Path(self.args.output_dir) / "validation_ranking/latest_metrics.json", report)
            return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=rows)

    return IndustrialRLValidationTrainer


class FinalStepEvaluationCallback(TrainerCallback):
    def __init__(self, monitor=None):
        self.monitor = monitor

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(self.monitor, "abort_reason", None):
            return control
        if state.max_steps > 0 and state.global_step >= state.max_steps:
            # Includes the final live weights in selection before Trainer
            # reloads the best checkpoint, even when the interval does not divide
            # the total number of optimizer updates.
            control.should_evaluate = True
            control.should_save = True
        return control


def terminal_eval_callback(monitor=None):
    return FinalStepEvaluationCallback(monitor)


class RLMonitorCallback(SFTMonitorCallback):
    """Preserve RL logs while retaining the proven global-gradient safeguards."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        common = {"step": int(state.global_step), "epoch": state.epoch, "phase": self.phase,
                  "timestamp_utc": utc_now(), "elapsed_seconds": self._elapsed()}
        if any(key.startswith("eval_") for key in logs):
            record = {**common, "kind": "eval", **json_value(logs)}
            self.evaluations.append(record)
            self._append(self.metrics_path, record)
            self._write_summary(state)
            return control
        if "loss" not in logs and "grad_norm" not in logs:
            return control
        raw_norm = logs.get("grad_norm")
        nonfinite = raw_norm is not None and finite_number(raw_norm) is None
        record = {**common, **json_value(logs), "kind": "train",
                  "loss": finite_number(logs.get("loss")),
                  "learning_rate": finite_number(logs.get("learning_rate")),
                  "grad_norm": finite_number(raw_norm),
                  "grad_norm_observed": str(raw_norm) if nonfinite else finite_number(raw_norm),
                  "grad_norm_nonfinite": nonfinite,
                  "step_skipped": self.last_step_skipped, "optimizer_overflow": self.last_overflow}
        if self.statistics.add(record):
            stats = self.statistics.summary()
            record.update({key: stats[key] for key in ("rolling_grad_norm_mean", "rolling_grad_norm_max",
                                                      "grad_norm_mean", "grad_norm_max", "grad_norm_max_step")})
            self.latest = record
            self._append(self.metrics_path, record)
            self._write_summary(state)
        critical = ("loss", "reward", "kl", "zero_reward_group_rate", "hit_group_rate")
        bad = nonfinite or any(key in logs and finite_number(logs[key]) is None for key in critical)
        if bad:
            self._stop(control, "nonfinite_rl_loss_reward_kl_or_gradient", state)
        elif self.require_grad_norm and self.statistics.records >= 10 and self.statistics.count == 0:
            self._stop(control, "grad_norm_missing_for_first_10_updates", state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        score = (metrics or {}).get("eval_ndcg10")
        bad = self.pending_bad_loss or finite_number(score) is None
        self.pending_bad_loss = False
        if self._synchronize_bad_flag(bad):
            self._stop(control, "nonfinite_or_missing_validation_ndcg10", state)
        return control
