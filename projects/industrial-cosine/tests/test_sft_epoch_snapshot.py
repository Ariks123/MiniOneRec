"""Exact live epoch exports and minimum-epoch early-stop safety on CPU."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sft_monitor import atomic_json


ROOT = Path(__file__).resolve().parents[1]


class AlwaysStoppingParent:
    def __init__(self, **kwargs):
        self.evaluations = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluations += 1
        control.should_training_stop = True

    def state(self):
        return {"args": {"early_stopping_patience": 1},
                "attributes": {"evaluations": self.evaluations}}


class SimulatedDistributed:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.rank = 0
        self.payload = None

    def is_available(self):
        return self.enabled

    def is_initialized(self):
        return self.enabled

    def get_world_size(self):
        return 2

    def all_gather_object(self, destination, value):
        destination[:] = [value, value]

    def broadcast_object_list(self, values, src):
        if self.rank == src:
            self.payload = copy.deepcopy(values)
        else:
            values[:] = copy.deepcopy(self.payload)


def objects(distributed):
    names = {"EpochSnapshotCallback", "MinimumEpochEarlyStoppingCallback",
             "write_sid_provenance", "verify_task_resume_configuration"}
    nodes = [node for node in ast.parse((ROOT / "sft.py").read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"Path": Path, "math": math, "json": json, "hashlib": hashlib,
                 "os": __import__("os"), "atomic_json": atomic_json,
                 "utc_now": lambda: "2026-11-17T00:00:00+00:00",
                 "TrainerCallback": object, "EarlyStoppingCallback": AlwaysStoppingParent,
                 "torch": SimpleNamespace(distributed=distributed)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "sft.py"), "exec"), namespace)
    return namespace


class FakeTokenizer:
    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer.json").write_text('{"saved": true}')


class FakeTrainer:
    def __init__(self, payload, primary=True):
        self.payload, self.primary, self.calls = payload, primary, []

    def save_model(self, directory):
        self.calls.append(directory)
        if self.primary:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(self.payload)
            (path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')


class EpochSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="epoch-snapshot-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index, self.meta = self.root / "index.json", self.root / "meta.json"
        self.index.write_text('{"0": ["<a_0>", "<b_0>", "<c_0>"]}')
        self.meta.write_text('{"0": {"title": "A title"}}')
        self.dist = SimulatedDistributed()
        self.ns = objects(self.dist)

    def callback(self, namespace=None):
        return (namespace or self.ns)["EpochSnapshotCallback"](
            self.root / "run", 3, FakeTokenizer(), self.index, self.meta,
            "Qwen2.5-0.5B", "Industrial_and_Scientific")

    def state(self, epoch, primary=True):
        return SimpleNamespace(epoch=epoch, global_step=round(epoch * 1814),
                               is_world_process_zero=primary)

    def test_exact_epoch_three_saves_live_weights_before_best_reload_only_once(self):
        callback = self.callback()
        trainer = FakeTrainer(b"actual-live-epoch-three")
        callback.attach_trainer(trainer)
        control = SimpleNamespace(should_training_stop=True)
        callback.on_epoch_end(None, self.state(2.999), control)
        self.assertEqual(trainer.calls, [])
        callback.on_epoch_end(None, self.state(3.0), control)
        self.assertTrue(control.should_training_stop)
        path = self.root / "run/epoch_3_checkpoint"
        self.assertEqual((path / "model.safetensors").read_bytes(), b"actual-live-epoch-three")
        trainer.payload = b"validation-best-weights-loaded-after-train"
        callback.on_epoch_end(None, self.state(5.0), control)
        self.assertEqual(len(trainer.calls), 1)
        self.assertEqual((path / "model.safetensors").read_bytes(), b"actual-live-epoch-three")
        metadata = json.loads((path / "snapshot_metadata.json").read_text())
        self.assertEqual(metadata["epoch"], 3.0)
        self.assertEqual(metadata["global_step"], 5442)
        self.assertEqual(metadata["steps_per_epoch"], 1814)
        self.assertEqual(metadata["capture_event"], "on_epoch_end")
        self.assertFalse(metadata["best_model_reloaded"])
        self.assertTrue((path / "tokenizer.json").is_file())
        self.assertEqual(json.loads((path / "posttraining_provenance.json").read_text())["sid_index_sha256"],
                         hashlib.sha256(self.index.read_bytes()).hexdigest())

    def test_each_rank_enters_model_save_before_primary_metadata_publication(self):
        distributed = SimulatedDistributed(enabled=True)
        ns = objects(distributed)
        primary, secondary = self.callback(ns), self.callback(ns)
        first, second = FakeTrainer(b"shared-model", True), FakeTrainer(b"unused", False)
        primary.attach_trainer(first)
        secondary.attach_trainer(second)
        control = SimpleNamespace(should_training_stop=False)
        distributed.rank = 0
        primary.on_epoch_end(None, self.state(3.0, True), control)
        distributed.rank = 1
        secondary.on_epoch_end(None, self.state(3.0, False), control)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(primary.metadata, secondary.metadata)

    def test_missing_boundary_and_incomplete_existing_export_are_not_claimed_complete(self):
        callback = self.callback()
        callback.attach_trainer(FakeTrainer(b"late"))
        with self.assertRaisesRegex(RuntimeError, "boundary was not captured"):
            callback.on_epoch_end(None, self.state(4.0), SimpleNamespace())
        callback.directory.mkdir(parents=True)
        (callback.directory / "model.safetensors").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "incomplete epoch snapshot"):
            self.callback()

    def test_resume_rejects_changing_minimum_or_snapshot_epoch(self):
        checkpoint = self.root / "checkpoint-1"
        checkpoint.mkdir()
        previous = {"task_counts": {"a": 2}, "data": {"include_title_history": True},
                    "hyperparameters": {"early_stopping_patience": 3, "eval_save_fraction": .05,
                                        "save_epoch_checkpoint": 3, "minimum_epochs": 3}}
        (self.root / "run_config.json").write_text(json.dumps(previous))
        verify = self.ns["verify_task_resume_configuration"]
        verify(checkpoint, {"a": 2}, True, 3, .05, False, 4, 50, 3, 3)
        for snapshot, minimum in ((0, 3), (3, 0)):
            with self.subTest(snapshot=snapshot, minimum=minimum), self.assertRaises(ValueError):
                verify(checkpoint, {"a": 2}, True, 3, .05, False, 4, 50, snapshot, minimum)


class MinimumEpochStoppingTests(unittest.TestCase):
    def setUp(self):
        ns = objects(SimulatedDistributed())
        self.callback = ns["MinimumEpochEarlyStoppingCallback"](3, early_stopping_patience=1)
        self.args = SimpleNamespace(metric_for_best_model="eval_loss")

    def evaluate(self, epoch, already_stopping=False, loss=1.0):
        control = SimpleNamespace(should_training_stop=already_stopping)
        self.callback.on_evaluate(self.args, SimpleNamespace(epoch=epoch), control,
                                  metrics={"eval_loss": loss})
        return control

    def test_only_parent_patience_stop_is_delayed_until_epoch_three(self):
        self.assertFalse(self.evaluate(2.5).should_training_stop)
        self.assertTrue(self.evaluate(3.0).should_training_stop)
        self.assertEqual(self.callback.evaluations, 2)

    def test_preexisting_safety_stop_and_nonfinite_validation_stop_are_preserved(self):
        self.assertTrue(self.evaluate(1.0, already_stopping=True).should_training_stop)
        self.assertTrue(self.evaluate(1.0, loss=math.nan).should_training_stop)
        self.assertTrue(self.evaluate(1.0, loss=math.inf).should_training_stop)

    def test_resume_state_keeps_required_minimum_epoch_constructor_argument(self):
        self.evaluate(2.5)
        value = self.callback.state()
        self.assertEqual(value["args"]["minimum_epochs"], 3)
        self.assertEqual(value["args"]["early_stopping_patience"], 1)
        self.assertEqual(value["attributes"]["evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
