"""Identity, resource and completion contracts for the Industrial RL supervisor."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("industrial_rl_experiment", ROOT / "docker/industrial_rl_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class IndustrialRLExperimentTests(unittest.TestCase):
    def test_training_pins_selected_sft_and_original_rl_options(self):
        arguments = experiment.train_arguments({"run_dir": str(experiment.RUN)})
        # Read options by name; the torchrun prefix is intentionally independent.
        options = {argument: arguments[i + 1] for i, argument in enumerate(arguments[:-1])
                   if argument.startswith("--") and "=" not in argument}
        self.assertIn("rl.py", arguments)
        self.assertIn("--nproc_per_node=4", arguments)
        expected = {"--model_path": str(experiment.PARENT_MODEL), "--num_train_epochs": "2",
                    "--learning_rate": "1e-05", "--num_generations": "16", "--train_batch_size": "16",
                    "--gradient_accumulation_steps": "16", "--reward_type": "ranking",
                    "--beta": "0.001", "--sync_ref_model": "True", "--ref_model_sync_steps": "512",
                    "--ref_model_mixup_alpha": "0.6", "--eval_save_fraction": "0.125",
                    "--early_stopping_patience": "3", "--test_during_training": "False"}
        for key, value in expected.items():
            self.assertEqual(options[key], value, key)
        self.assertNotIn("Video_Games", " ".join(arguments))
        self.assertNotIn("--resume_from_checkpoint", arguments)
        with self.assertRaisesRegex(ValueError, "two-epoch"):
            experiment.train_arguments({"run_dir": "unused", "num_epochs": 3})
        with self.assertRaisesRegex(ValueError, "parent model"):
            experiment.train_arguments({"run_dir": "unused", "model_path": str(experiment.BASE_MODEL)})

    def test_evaluation_keeps_full_original_beam50_protocol(self):
        arguments = experiment.eval_arguments({"run_dir": str(experiment.RUN)})
        for name, value in [("--model-path", str(experiment.RUN / "final_checkpoint")),
                            ("--expected-sid-sha256", experiment.APPROVED_SID),
                            ("--num-beams", "50"), ("--prompt-style", "original-eval")]:
            self.assertEqual(arguments[arguments.index(name) + 1], value)

    def test_status_does_not_start_or_inspect_resources(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            with patch.object(experiment.subprocess, "run") as run, patch.object(experiment.subprocess, "Popen") as popen:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(experiment.main(["status", "--run-dir", folder]), 0)
            run.assert_not_called()
            popen.assert_not_called()

    def test_existing_run_is_preserved_before_resource_checks(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            marker = Path(folder) / "keep.txt"
            marker.write_text("keep")
            with patch.object(experiment, "check_prepared") as prepared, patch.object(experiment, "check_storage") as storage:
                with self.assertRaisesRegex(ValueError, "Existing Industrial RL run retained"):
                    experiment.main(["start", "--run-dir", folder])
            prepared.assert_not_called()
            storage.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_busy_or_missing_gpu_is_rejected(self):
        idle = "\n".join(f"{i}, NVIDIA H20, 10, 0" for i in range(4))
        experiment.validate_gpu_idle(idle)
        for csv in (idle.replace("10, 0", "4000, 70", 1), "0, NVIDIA H20, 10, 0",
                    idle.replace("H20", "A100"), idle.replace("10, 0", "[N/A], 0", 1)):
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                experiment.validate_gpu_idle(csv)

    def test_storage_guard_measures_local_disk_not_oss_capacity(self):
        with patch.object(experiment.shutil, "disk_usage", return_value=types.SimpleNamespace(free=49 * 1024 ** 3)) as usage:
            with self.assertRaisesRegex(ValueError, "requires 50 GiB"):
                experiment.check_storage()
            usage.assert_called_once_with("/tmp")
        with patch.object(experiment.shutil, "disk_usage", return_value=types.SimpleNamespace(free=51 * 1024 ** 3)):
            self.assertEqual(experiment.check_storage(), 51)

    def test_cached_baseline_requires_all_recorded_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            run = Path(folder)
            hashes = {}
            for key, (_, destination) in experiment.PARENT_REPORT_FILES.items():
                path = run / destination
                path.write_text(key)
                hashes[key] = experiment.sha256(path)
            experiment.check_parent_cache(run, {"sha256": hashes})
            (run / "parent_sft_predictions.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "predictions"):
                experiment.check_parent_cache(run, {"sha256": hashes})

    def test_post_launch_source_change_is_rejected(self):
        with patch.object(experiment, "source_fingerprints", return_value={"rl.py": "new"}):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                experiment.check_source_fingerprints({"rl.py": "old"})

    def make_parent(self, directory):
        parent = directory / "sft"
        model = parent / "final_checkpoint"
        selected = parent / "checkpoint-5187"
        model.mkdir(parents=True)
        selected.mkdir()
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                     "added_tokens.json", "special_tokens_map.json"):
            (model / name).write_text("model fixture")
        (selected / "model.safetensors").write_text("model fixture")
        experiment.atomic_json(model / "posttraining_provenance.json", {"sid_index_sha256": experiment.APPROVED_SID})
        experiment.atomic_json(parent / "run_config.json", {
            "stage": "sft", "data": {"category": experiment.CATEGORY, "include_title_history": True},
            "model": {"source": str(experiment.BASE_MODEL), "identity": {"sha256": experiment.PARENT_BASE_IDENTITY}},
            "hyperparameters": {"max_epochs": 3, "lr_scheduler_type": "linear"}})
        experiment.atomic_json(parent / "training_summary.json", {
            "status": "completed", "selection_metric": "eval_loss", "final_checkpoint_contains_best_model": True,
            "global_step": 5442, "epoch": 3.0, "best_checkpoint": str(selected), "final_checkpoint": str(model)})
        experiment.atomic_json(parent / "run_status.json", {"runtime_image_id": "sha256:fixture"})
        experiment.atomic_json(parent / "evaluation/metrics.json", {
            "status": "passed", "completed_rows": 4533, "expected_rows": 4533,
            "num_beams": 50, "prompt_style": "original-eval", "validity_rate": 1.0,
            "unique_rate": 1.0, "model_path": str(model)})
        (parent / "evaluation/predictions.jsonl").write_text("prediction fixture")
        return parent, model, selected

    def test_parent_export_must_match_selected_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            parent, model, selected = self.make_parent(Path(folder))
            with patch.object(experiment, "PARENT_MODEL", model), \
                    patch.object(experiment, "PARENT_WEIGHT_SHA256", experiment.sha256(model / "model.safetensors")):
                verified = experiment.validate_parent(model)
                self.assertEqual(verified["model_sha256"], experiment.sha256(selected / "model.safetensors"))
                self.assertEqual(set(verified["report_sources"]["sha256"]), set(experiment.PARENT_REPORT_FILES))
                (model / "model.safetensors").write_text("wrong model")
                with self.assertRaisesRegex(ValueError, "approved weight SHA-256"):
                    experiment.validate_parent(model)

    def test_completion_requires_selected_export_test_and_reports(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            config = {"run_dir": str(base / "run"), "report_root": str(base / "reports"),
                      "container": "unused", "model_path": str(experiment.PARENT_MODEL),
                      "parent_model_sha256": "model", "parent_sft_report_sources": {"run_dir": "parent"}}
            supervisor = experiment.Supervisor(config)
            supervisor.run.mkdir()
            commands, calls = [], []

            def command(arguments, stage):
                supervisor.update(stage)
                commands.append((arguments, stage))
                if stage == "training":
                    experiment.atomic_json(supervisor.run / "training_summary.json", {
                        "status": "completed", "selection_metric": "eval_ndcg10",
                        "final_checkpoint_contains_best_model": True})
                else:
                    experiment.atomic_json(supervisor.run / "evaluation/metrics.json", {
                        "status": "passed", "completed_rows": 4533})

            def render():
                supervisor.reports.mkdir(exist_ok=True)
                experiment.atomic_json(supervisor.reports / "training_report.json", {"status": "completed"})
                experiment.atomic_json(supervisor.reports / "validation_report.json", {"status": "passed"})
                experiment.atomic_json(supervisor.reports / "comparison.json", {"status": "passed"})
                for name in ("training-report.pdf", "validation-report.pdf"):
                    (supervisor.reports / name).write_bytes(b"fixture")

            supervisor.command = command
            supervisor.call = lambda arguments, **kwargs: calls.append(arguments)
            supervisor.prepare_container = lambda: None
            supervisor.render_reports = render
            parent = {"model_sha256": "model", "report_sources": {"run_dir": "parent"}}
            with patch.object(experiment, "check_prepared", return_value=parent):
                supervisor.execute()
            self.assertEqual([stage for _, stage in commands], ["training", "validating"])
            self.assertEqual(supervisor.status["status"], "completed")
            self.assertTrue(supervisor.status["rl_enabled"])
            self.assertTrue(supervisor.status["rl_started"])
            self.assertEqual(calls[-1], ["stop", "--timeout", "2", "unused"])

    def test_final_gate_rejects_missing_training_telemetry_despite_passed_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            supervisor = experiment.Supervisor({"run_dir": str(base / "run"), "report_root": str(base),
                                                "container": "unused"})
            experiment.atomic_json(base / "training_report.json", {"status": "incomplete"})
            experiment.atomic_json(base / "validation_report.json", {"status": "passed"})
            experiment.atomic_json(base / "comparison.json", {"status": "passed"})
            for name in ("training-report.pdf", "validation-report.pdf"):
                (base / name).write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "training report is incomplete"):
                supervisor.validate_final_reports()

    def test_final_gate_rejects_invalid_parent_or_recommendation_despite_passed_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            supervisor = experiment.Supervisor({"run_dir": str(base / "run"), "report_root": str(base),
                                                "container": "unused"})
            experiment.atomic_json(base / "training_report.json", {"status": "completed"})
            experiment.atomic_json(base / "validation_report.json", {"status": "passed"})
            experiment.atomic_json(base / "comparison.json", {"status": "incomplete"})
            for name in ("training-report.pdf", "validation-report.pdf"):
                (base / name).write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "parent baseline and recommendation"):
                supervisor.validate_final_reports()

    def test_interrupted_remote_training_is_stopped_and_log_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            config = {"run_dir": str(base / "run"), "report_root": str(base / "reports"), "container": "own-rl"}
            supervisor = experiment.Supervisor(config)
            supervisor.run.mkdir()
            spool = base / "spool.log"
            spool.write_text("partial training output")
            supervisor.status.update(stage="training", training_log_spool=str(spool), rl_started=True)
            from unittest.mock import Mock
            child = Mock()
            child.poll.return_value = None
            supervisor.child = child
            calls = []
            supervisor.call = lambda arguments, **kwargs: calls.append(arguments)
            supervisor.fail(KeyboardInterrupt("stopped"))
            self.assertEqual(calls, [["stop", "--timeout", "20", "own-rl"]])
            child.wait.assert_called_once_with(timeout=30)
            self.assertEqual((supervisor.run / "training.log").read_text(), "partial training output")
            self.assertEqual(supervisor.status["status"], "failed")


if __name__ == "__main__":
    unittest.main()
