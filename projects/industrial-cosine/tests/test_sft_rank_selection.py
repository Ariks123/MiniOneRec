"""Validation-only checkpoint retention, independent metrics, and resume isolation."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from sft_monitor import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def objects(names):
    source = ast.parse((ROOT / "sft.py").read_text())
    selected = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {"Path": Path, "math": math, "os": os, "shutil": shutil, "tempfile": tempfile,
                 "json": json, "hashlib": hashlib, "atomic_json": atomic_json,
                 "TrainerCallback": object, "torch": SimpleNamespace(distributed=SimpleNamespace(
                     is_available=lambda: False, is_initialized=lambda: False))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / "sft.py"), "exec"), namespace)
    return namespace


class Tokenizer:
    def save_pretrained(self, destination):
        (Path(destination) / "tokenizer.json").write_text('{"tokenizer": "same-saved-tokenizer"}')


class RankSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sft-rank-selection-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index, self.meta = self.root / "index.json", self.root / "meta.json"
        self.index.write_text('{"0": ["<a_1>", "<b_2>", "<c_3>"]}')
        self.meta.write_text('{"0": {"title": "Known item"}}')
        self.ns = objects({"ranking_values", "recall_improves", "write_sid_provenance", "BestRecallCheckpointCallback"})
        self.callback = self.ns["BestRecallCheckpointCallback"](
            self.root, Tokenizer(), str(self.index), str(self.meta), "Qwen2.5-0.5B", "Industrial_and_Scientific")

    def evaluate(self, step, ndcg, recall, loss):
        state = SimpleNamespace(global_step=step, epoch=step / 2, is_world_process_zero=True)
        control = SimpleNamespace(should_save=False)
        self.callback.on_evaluate(None, state, control, {
            "eval_ndcg10": ndcg, "eval_recall10": recall, "eval_loss": loss})
        return state, control

    def save(self, state, payload):
        checkpoint = self.root / f"checkpoint-{state.global_step}"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(payload)
        (checkpoint / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer must not be mirrored")
        self.callback.on_save(None, state, SimpleNamespace(should_save=True))
        return checkpoint

    def test_recall_and_ndcg_keep_independent_best_models_despite_checkpoint_rotation(self):
        first, control = self.evaluate(1, .1, .4, 2)
        self.assertTrue(control.should_save)
        checkpoint = self.save(first, b"recall-best-first")
        second, control = self.evaluate(2, .3, .3, 1)
        self.assertFalse(control.should_save)
        self.save(second, b"ndcg-best-second")
        shutil.rmtree(checkpoint)  # Simulate Trainer's later checkpoint rotation.
        retained = self.root / "best_recall_checkpoint"
        self.assertEqual((retained / "model.safetensors").read_bytes(), b"recall-best-first")
        self.assertFalse((retained / "optimizer.pt").exists())
        self.assertTrue((retained / "tokenizer.json").is_file())
        provenance = json.loads((retained / "posttraining_provenance.json").read_text())
        self.assertEqual(provenance["sid_index_sha256"], hashlib.sha256(self.index.read_bytes()).hexdigest())
        saved = json.loads((self.root / "ranking_selection.json").read_text())
        self.assertEqual(saved["best_recall"]["step"], 1)
        self.assertEqual(saved["best_ndcg"]["step"], 2)
        self.assertEqual(saved["scope"], "complete_original_validation_split")

    def test_higher_recall_replaces_export_and_final_best_reload_does_not_reselect(self):
        first, _ = self.evaluate(1, .3, .3, 2)
        self.save(first, b"first")
        second, control = self.evaluate(2, .2, .4, 1.8)
        self.assertTrue(control.should_save)
        self.save(second, b"higher-recall")
        self.callback.active = False
        self.evaluate(99, .99, .99, .01)
        self.assertEqual(self.callback.best_recall["step"], 2)
        self.assertEqual(self.callback.best_ndcg["step"], 1)
        self.assertEqual((self.root / "best_recall_checkpoint/model.safetensors").read_bytes(), b"higher-recall")

    def test_incomplete_new_model_cannot_replace_existing_export(self):
        first, _ = self.evaluate(1, .1, .1, 2)
        self.save(first, b"complete-old")
        second, _ = self.evaluate(2, .2, .2, 1)
        checkpoint = self.root / "checkpoint-2"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "no complete model export"):
            self.callback.on_save(None, second, SimpleNamespace(should_save=True))
        self.assertEqual((self.root / "best_recall_checkpoint/model.safetensors").read_bytes(), b"complete-old")

    def test_selection_uses_recall_then_ndcg_and_rejects_nonfinite_values(self):
        better = self.ns["recall_improves"]
        previous = {"recall10": .3, "ndcg10": .2, "loss": 2}
        self.assertFalse(better({"recall10": .2, "ndcg10": .9, "loss": .01}, previous))
        self.assertTrue(better({"recall10": .3, "ndcg10": .21, "loss": 5}, previous))
        self.assertTrue(better({"recall10": .3, "ndcg10": .2, "loss": 1.9}, previous))
        for name, value in (("ndcg10", math.nan), ("recall10", 1.1), ("loss", -1)):
            metrics = {"eval_ndcg10": .1, "eval_recall10": .2, "eval_loss": 1}
            metrics["eval_" + name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ns["ranking_values"](metrics)


class RankResumeTests(unittest.TestCase):
    def test_loss_and_ranking_selection_cannot_share_optimizer_resume(self):
        ns = objects({"verify_task_resume_configuration"})
        with tempfile.TemporaryDirectory(prefix="rank-resume-") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            previous = {"task_counts": {"task": 2}, "data": {"include_title_history": True},
                        "hyperparameters": {"early_stopping_patience": 3, "eval_save_fraction": .1,
                                            "generative_validation": False}}
            (root / "run_config.json").write_text(json.dumps(previous))
            with self.assertRaisesRegex(ValueError, "generative_validation"):
                ns["verify_task_resume_configuration"](checkpoint, {"task": 2}, True, 3, .1, True, 4, 50)


if __name__ == "__main__":
    unittest.main()
