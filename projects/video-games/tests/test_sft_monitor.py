"""Check durable gradient statistics and monitor failure/resume behavior on CPU."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from sft_monitor import (GradientStatistics, SFTMonitorCallback, atomic_json, file_sha256,
                         model_source_identity, validate_resume_identity, write_verified_run_config)


def state(step=0, primary=True):
    return SimpleNamespace(global_step=step, epoch=step / 10, max_steps=100,
                           is_world_process_zero=primary, best_model_checkpoint=None,
                           best_metric=None)


def control():
    return SimpleNamespace(should_training_stop=False, should_save=True, should_evaluate=True)


class GradientStatisticsTests(unittest.TestCase):
    def test_window_and_global_statistics_exclude_skipped_missing_and_nonfinite(self):
        stats = GradientStatistics(window=2)
        for row in [
            {"step": 1, "grad_norm": 2.0},
            {"step": 2, "grad_norm": 1000.0, "step_skipped": True},
            {"step": 3, "grad_norm": None},
            {"step": 4, "grad_norm": None, "grad_norm_nonfinite": True},
            {"step": 5, "grad_norm": 4.0},
            {"step": 6, "grad_norm": 9.0},
        ]:
            self.assertTrue(stats.add(row))
        result = stats.summary()
        self.assertEqual(result["update_records"], 6)
        self.assertEqual(result["grad_norm_count"], 3)
        self.assertEqual(result["grad_norm_mean"], 5.0)
        self.assertEqual(result["grad_norm_max"], 9.0)
        self.assertEqual(result["grad_norm_max_step"], 6)
        self.assertEqual(result["rolling_grad_norm_mean"], 6.5)
        self.assertEqual(result["rolling_grad_norm_max"], 9.0)
        self.assertEqual(result["missing_grad_norm_count"], 1)
        self.assertEqual(result["nonfinite_grad_norm_count"], 1)
        self.assertEqual(result["skipped_update_count"], 1)
        self.assertFalse(stats.add({"step": 6, "grad_norm": 10000}))
        self.assertEqual(stats.summary(), result)

    def test_invalid_windows_are_rejected(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                GradientStatistics(value)

    def test_skipped_nonfinite_record_counts_both_events(self):
        stats = GradientStatistics()
        stats.add({"step": 1, "grad_norm_nonfinite": True, "step_skipped": True})
        self.assertEqual(stats.nonfinite, 1)
        self.assertEqual(stats.skipped, 1)
        self.assertEqual(stats.count, 0)


class SFTMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-monitor-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def monitor(self, primary=True):
        monitor = SFTMonitorCallback(self.directory, window=2)
        monitor.on_train_begin(None, state(primary=primary), control())
        return monitor

    def rows(self):
        with (self.directory / "monitor/metrics.jsonl").open() as stream:
            return [json.loads(line) for line in stream]

    def test_logs_flush_values_and_evaluation_without_counting_eval_as_update(self):
        monitor = self.monitor()
        decision = control()
        monitor.on_log(None, state(1), decision,
                       logs={"loss": 3.0, "learning_rate": 0.001, "grad_norm": 2.0})
        monitor.on_log(None, state(1), decision, logs={"eval_loss": 2.5})
        monitor.on_log(None, state(2), decision,
                       logs={"loss": 2.0, "learning_rate": 0.0009, "grad_norm": 4.0})
        rows = self.rows()
        self.assertEqual([row["kind"] for row in rows], ["train", "eval", "train"])
        self.assertEqual(rows[-1]["grad_norm_mean"], 3.0)
        self.assertEqual(rows[-1]["rolling_grad_norm_max"], 4.0)
        summary = json.loads((self.directory / "monitor/summary.json").read_text())
        self.assertEqual(summary["update_records"], 2)
        self.assertEqual(summary["evaluations"][0]["eval_loss"], 2.5)
        self.assertFalse(decision.should_training_stop)

    def test_norm_above_clipping_threshold_does_not_stop_training(self):
        monitor = self.monitor()
        decision = control()
        monitor.on_log(None, state(1), decision, logs={"loss": 2.0, "grad_norm": 100.0})
        self.assertFalse(decision.should_training_stop)

    def test_nonfinite_logs_stop_and_remain_valid_json(self):
        monitor = self.monitor()
        decision = control()
        monitor.on_log(None, state(1), decision, logs={"loss": math.nan, "grad_norm": math.inf})
        self.assertTrue(decision.should_training_stop)
        self.assertFalse(decision.should_save)
        self.assertFalse(decision.should_evaluate)
        row = self.rows()[0]
        self.assertIsNone(row["loss"])
        self.assertIsNone(row["grad_norm"])
        self.assertEqual(row["grad_norm_observed"], "inf")
        self.assertTrue(row["grad_norm_nonfinite"])
        self.assertEqual(monitor.statistics.count, 0)

    def test_missing_gradients_fail_after_ten_logged_updates(self):
        monitor = self.monitor()
        decision = control()
        for step in range(1, 11):
            monitor.on_log(None, state(step), decision, logs={"loss": 2.0})
        self.assertTrue(decision.should_training_stop)
        self.assertEqual(monitor.abort_reason, "grad_norm_missing_for_first_10_updates")

    def test_synchronized_nonfinite_flag_stops_other_ranks(self):
        monitor = self.monitor(primary=False)
        monitor.trainer = SimpleNamespace(model_wrapped=SimpleNamespace(), optimizer=SimpleNamespace())
        seen = []
        monitor._synchronize_bad_flag = lambda bad: seen.append(bad) or True
        decision = control()
        monitor.on_step_end(None, state(1, primary=False), decision)
        self.assertEqual(seen, [False])
        self.assertTrue(decision.should_training_stop)
        self.assertFalse((self.directory / "monitor/metrics.jsonl").exists())

    def test_overflow_excludes_cached_previous_norm(self):
        monitor = self.monitor()
        engine = SimpleNamespace(optimizer=SimpleNamespace(overflow=True),
                                 get_global_grad_norm=lambda: 4.0)
        monitor.trainer = SimpleNamespace(model_wrapped=engine, optimizer=SimpleNamespace(step_was_skipped=True))
        monitor._synchronize_bad_flag = lambda bad: bad
        decision = control()
        monitor.on_step_end(None, state(1), decision)
        monitor.on_log(None, state(1), decision, logs={"loss": 2.0, "grad_norm": 4.0})
        self.assertEqual(monitor.statistics.count, 0)
        self.assertEqual(monitor.statistics.skipped, 1)
        self.assertTrue(decision.should_training_stop)

    def test_resume_keeps_checkpoint_prefix_and_archives_replayed_tail(self):
        monitor = self.monitor()
        for step, norm in [(1, 2), (2, 4), (3, 100)]:
            monitor.on_log(None, state(step), control(), logs={"loss": 2.0, "grad_norm": norm})
        resumed = SFTMonitorCallback(self.directory, window=2)
        resumed.on_train_begin(None, state(2), control())
        resumed.on_log(None, state(3), control(), logs={"loss": 1.0, "grad_norm": 6.0})
        self.assertEqual([row["step"] for row in self.rows()], [1, 2, 3])
        self.assertEqual(resumed.statistics.summary()["grad_norm_mean"], 4.0)
        self.assertEqual(len(list((self.directory / "monitor").glob("metrics.before_resume.*.jsonl"))), 1)

    def test_fresh_run_cannot_overwrite_existing_training_metrics(self):
        monitor = self.monitor()
        monitor.on_log(None, state(1), control(), logs={"loss": 2.0, "grad_norm": 1.0})
        with self.assertRaisesRegex(ValueError, "checkpoint resume"):
            self.monitor()

    def test_resume_archives_and_repairs_partial_tail_before_appending(self):
        monitor = self.monitor()
        for step, norm in [(1, 2), (2, 4)]:
            monitor.on_log(None, state(step), control(), logs={"loss": 2.0, "grad_norm": norm})
        with monitor.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write('{"step":3,"grad_norm":')
        resumed = SFTMonitorCallback(self.directory, window=2)
        resumed.on_train_begin(None, state(2), control())
        resumed.on_log(None, state(3), control(), logs={"loss": 1.0, "grad_norm": 6.0})
        rows = self.rows()  # Every line must decode independently after the repair.
        self.assertEqual([row["step"] for row in rows], [1, 2, 3])
        self.assertEqual(resumed.statistics.count, 3)
        self.assertEqual(resumed.statistics.summary()["grad_norm_mean"], 4.0)
        archives = list((self.directory / "monitor").glob("metrics.before_resume.*.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertTrue(archives[0].read_text().endswith('{"step":3,"grad_norm":'))

    def test_resume_normalizes_complete_json_tail_without_newline(self):
        monitor = self.monitor()
        monitor.on_log(None, state(1), control(), logs={"loss": 2.0, "grad_norm": 2.0})
        monitor.metrics_path.write_text(monitor.metrics_path.read_text().rstrip("\n"))
        resumed = SFTMonitorCallback(self.directory, window=2)
        resumed.on_train_begin(None, state(1), control())
        resumed.on_log(None, state(2), control(), logs={"loss": 1.0, "grad_norm": 4.0})
        self.assertEqual([row["step"] for row in self.rows()], [1, 2])
        self.assertEqual(resumed.statistics.count, 2)

    def test_nonprimary_rank_never_writes_monitor_files(self):
        monitor = self.monitor(primary=False)
        monitor.on_log(None, state(1, primary=False), control(), logs={"loss": 2.0, "grad_norm": 3.0})
        self.assertEqual(monitor.statistics.count, 1)
        self.assertFalse((self.directory / "monitor").exists())

    def test_atomic_json_sanitizes_nonfinite_values_and_hashes_actual_bytes(self):
        target = self.directory / "record.json"
        atomic_json(target, {"value": math.inf, "items": [math.nan, 1]})
        self.assertEqual(json.loads(target.read_text()), {"value": None, "items": [None, 1]})
        self.assertEqual(len(file_sha256(target)), 64)
        self.assertFalse(target.with_suffix(".json.tmp").exists())


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-identity-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "base"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"qwen2"}')
        (self.model / "model.safetensors").write_bytes(b"initial-weights")
        (self.model / "tokenizer.json").write_text('{"fixture":"tokenizer"}')
        self.inputs = {}
        for name in ("train_file", "eval_file", "sid_index_path", "item_meta_path"):
            path = self.directory / name
            path.write_text(name + " original")
            self.inputs[name] = str(path)
        self.broadcasts = []
        self.torch = SimpleNamespace(distributed=SimpleNamespace(
            is_available=lambda: True, is_initialized=lambda: True,
            broadcast_object_list=lambda values, **kwargs: self.broadcasts.append(json.loads(json.dumps(values))),
        ))

    def config(self):
        return {
            "model": {"source": str(self.model), "train_from_scratch": False},
            "data": {**self.inputs, "category": "Video_Games", "sample": -1},
            "hyperparameters": {"seed": 42, "global_batch_size": 1024,
                                "learning_rate": 3e-4, "max_epochs": 10},
        }

    def verified(self, config):
        config["model"]["identity"] = model_source_identity(self.model)
        config["data"]["sha256"] = {name: file_sha256(path) for name, path in self.inputs.items()}
        return config

    def test_identity_uses_weight_content_not_size_or_mtime(self):
        original = model_source_identity(self.model)
        weight = self.model / "model.safetensors"
        before = weight.stat()
        weight.write_bytes(b"changed-weights")
        self.assertEqual(weight.stat().st_size, before.st_size)
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        modified = model_source_identity(self.model)
        self.assertNotEqual(original["sha256"], modified["sha256"])
        self.assertNotEqual(original["files"]["model.safetensors"]["sha256"],
                            modified["files"]["model.safetensors"]["sha256"])

    def test_config_and_tokenizer_changes_change_source_identity(self):
        original = model_source_identity(self.model)
        (self.model / "tokenizer.json").write_text('{"fixture":"changed"}')
        changed_tokenizer = model_source_identity(self.model)
        self.assertNotEqual(original, changed_tokenizer)
        (self.model / "config.json").write_text('{"model_type":"different"}')
        self.assertNotEqual(changed_tokenizer, model_source_identity(self.model))

    def test_index_and_every_weight_shard_are_hashed(self):
        (self.model / "model.safetensors").unlink()
        (self.model / "model-1.safetensors").write_bytes(b"part1")
        (self.model / "model-2.safetensors").write_bytes(b"part2")
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
        identity = model_source_identity(self.model)
        self.assertIn("model.safetensors.index.json", identity["files"])
        self.assertIn("model-1.safetensors", identity["files"])
        self.assertIn("model-2.safetensors", identity["files"])
        (self.model / "model-2.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "shard is missing"):
            model_source_identity(self.model)

    def test_weight_index_cannot_escape_the_model_directory(self):
        (self.model / "model.safetensors").unlink()
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "../train_file"}}))
        with self.assertRaisesRegex(ValueError, "nonlocal filename"):
            model_source_identity(self.model)

    def test_equal_identity_is_accepted_and_changed_input_is_rejected(self):
        previous = self.verified(self.config())
        current = json.loads(json.dumps(previous))
        validate_resume_identity(previous, current)
        for name in self.inputs:
            with self.subTest(name=name):
                modified = json.loads(json.dumps(current))
                modified["data"]["sha256"][name] = "changed"
                with self.assertRaisesRegex(ValueError, name):
                    validate_resume_identity(previous, modified)

    def test_missing_identity_and_changed_training_configuration_are_rejected(self):
        previous = self.verified(self.config())
        with self.assertRaisesRegex(ValueError, "complete previous model"):
            validate_resume_identity(self.config(), previous)
        changed = json.loads(json.dumps(previous))
        changed["hyperparameters"]["seed"] = 123
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_resume_identity(previous, changed)

    def test_source_directory_is_part_of_the_identity(self):
        previous = self.verified(self.config())
        changed = json.loads(json.dumps(previous))
        changed["model"]["identity"]["source_directory"] = str(self.directory / "other-model")
        with self.assertRaisesRegex(ValueError, "source model"):
            validate_resume_identity(previous, changed)

    def test_rank_zero_writes_initial_identity_then_verifies_before_resume_overwrite(self):
        output = self.directory / "run"
        accelerator = SimpleNamespace(device="cpu")
        with patch.dict("sys.modules", {"torch": self.torch}):
            identity = write_verified_run_config(self.config(), output, None, True, accelerator)
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            write_verified_run_config(self.config(), output, str(checkpoint), True, accelerator)
            previous_bytes = (output / "run_config.json").read_bytes()
            Path(self.inputs["train_file"]).write_text("different training set")
            with self.assertRaisesRegex(ValueError, "train_file"):
                write_verified_run_config(self.config(), output, str(checkpoint), True, accelerator)
        self.assertEqual((output / "run_config.json").read_bytes(), previous_bytes)
        self.assertEqual(len(identity), 64)
        self.assertEqual(len(self.broadcasts), 3)
        self.assertIn("train_file", self.broadcasts[-1][0]["error"])

    def test_nonprimary_rank_receives_the_same_error_without_hashing_or_writing(self):
        def broadcast(values, **kwargs):
            values[0] = {"error": "ValueError: source identity mismatch"}
        self.torch.distributed.broadcast_object_list = broadcast
        with patch.dict("sys.modules", {"torch": self.torch}), \
             patch("sft_monitor.model_source_identity", side_effect=AssertionError("nonprimary hashing")):
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                write_verified_run_config(self.config(), self.directory / "not-created", "unused", False,
                                          SimpleNamespace(device="cpu"))
        self.assertFalse((self.directory / "not-created").exists())


if __name__ == "__main__":
    unittest.main()
