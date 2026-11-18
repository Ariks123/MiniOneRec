"""Safety and data identity checks for the isolated original-data SFT workflow."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("industrial_experiment", ROOT / "docker/industrial_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class IndustrialExperimentTests(unittest.TestCase):
    def test_training_uses_base_weights_original_data_and_separate_outputs(self):
        args = experiment.train_arguments({"run_dir": str(experiment.RUN)})
        self.assertEqual(args[args.index("--base_model") + 1], str(experiment.MODEL))
        self.assertEqual(args[args.index("--category") + 1], "Industrial_and_Scientific")
        self.assertIn("--nproc_per_node=4", args)
        self.assertEqual(args[args.index("--batch_size") + 1], "64")
        self.assertEqual(args[args.index("--micro_batch_size") + 1], "8")
        self.assertEqual(args[args.index("--num_epochs") + 1], "3")
        self.assertEqual(args[args.index("--include_title_history") + 1], "False")
        self.assertEqual(args[args.index("--lr_scheduler_type") + 1], "linear")
        self.assertEqual(args[args.index("--deepspeed") + 1], "/opt/minionerec/config/h20_zero2.json")
        self.assertNotIn("Video_Games", " ".join(args))
        self.assertFalse(any(value.endswith("rl.py") for value in args))

    def test_evaluation_pins_original_sid_and_reloads_final_model(self):
        args = experiment.eval_arguments({"run_dir": str(experiment.RUN)})
        self.assertEqual(args[args.index("--expected-sid-sha256") + 1], experiment.APPROVED_SID)
        self.assertEqual(args[args.index("--num-beams") + 1], "50")
        self.assertEqual(args[args.index("--prompt-style") + 1], "original-eval")
        self.assertEqual(args[args.index("--model-path") + 1], str(experiment.RUN / "final_checkpoint"))
        self.assertNotIn("Video_Games", " ".join(args))

    def test_unsupported_scheduler_is_rejected(self):
        for scheduler in ("constant", "polynomial", "cosine_with_restarts"):
            with self.subTest(scheduler=scheduler), self.assertRaises(ValueError):
                experiment.train_arguments({"run_dir": str(experiment.RUN),
                                            "lr_scheduler_type": scheduler})

    def test_other_base_model_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment.train_arguments({"run_dir": str(experiment.RUN),
                                        "base_model": str(experiment.WORKSPACE / "weights/Qwen2.5-1.5B")})

    def test_augmented_run_enables_only_the_extra_title_history_task(self):
        config = {"run_dir": str(experiment.RUN), "include_title_history": True}
        args = experiment.train_arguments(config)
        self.assertEqual(args[args.index("--include_title_history") + 1], "True")
        self.assertEqual(args[args.index("--num_epochs") + 1], "3")
        self.assertEqual(args[args.index("--early_stopping_patience") + 1], "3")
        for epochs in (0, 11, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                experiment.train_arguments({"run_dir": str(experiment.RUN), "num_epochs": epochs})

    def test_runtime_sources_are_local_to_this_branch(self):
        for name in experiment.SOURCE_FILES:
            with self.subTest(name=name):
                self.assertEqual(experiment.runtime_source(name), ROOT / name)
                self.assertTrue(experiment.runtime_source(name).is_file())

    def test_start_preserves_existing_experiment(self):
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT) as folder:
            run = Path(folder) / "run"
            run.mkdir()
            marker = run / "keep.txt"
            marker.write_text("keep")
            with patch.object(experiment, "check_prepared"), patch.object(experiment.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "Existing Industrial run retained"):
                    experiment.main(["start", "--run-dir", str(run)])
            popen.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_status_never_starts_docker_or_training(self):
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT) as folder:
            with patch.object(experiment.subprocess, "run") as run, patch.object(experiment.subprocess, "Popen") as popen:
                self.assertEqual(experiment.main(["status", "--run-dir", folder]), 0)
            run.assert_not_called()
            popen.assert_not_called()

    def test_atomic_status_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT) as folder:
            path = Path(folder) / "run_status.json"
            experiment.atomic_json(path, {"status": "running", "stage": "training"})
            self.assertEqual(experiment.read_json(path)["stage"], "training")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_loss_only_workflow_never_runs_ranking_or_requires_ranking_success(self):
        with tempfile.TemporaryDirectory(prefix="industrial-loss-only-", dir=ROOT) as folder:
            base = Path(folder)
            config = {"run_dir": str(base / "run"), "report_root": str(base / "reports"),
                      "container": "unused", "generative_validation": False,
                      "final_ranking_eval": False}
            supervisor = experiment.Supervisor(config)
            supervisor.run.mkdir()
            commands, calls = [], []

            def command(arguments, stage):
                commands.append((arguments, stage))
                experiment.atomic_json(supervisor.run / "training_summary.json", {
                    "status": "completed", "final_eval_metrics": {"eval_loss": 1.25}})

            def render():
                supervisor.reports.mkdir(exist_ok=True)
                experiment.atomic_json(supervisor.reports / "validation_report.json", {"status": "not_requested"})
                for name in ("training-report.pdf", "validation-report.pdf"):
                    (supervisor.reports / name).write_bytes(b"fixture")

            supervisor.command = command
            supervisor.call = lambda arguments, **kwargs: calls.append(arguments)
            supervisor.prepare_container = lambda: None
            supervisor.render_reports = render
            with patch.object(experiment, "check_prepared"):
                supervisor.execute()
            self.assertEqual([stage for _, stage in commands], ["training"])
            self.assertNotIn("scripts/evaluate_sft.py", " ".join(str(call) for call in calls))
            placeholder = experiment.read_json(supervisor.run / "evaluation/metrics.json")
            self.assertEqual(placeholder["status"], "not_requested")
            self.assertNotIn("metrics", placeholder)
            self.assertEqual(placeholder["validation"]["loss"], 1.25)
            self.assertEqual(supervisor.status["status"], "completed")


if __name__ == "__main__":
    unittest.main()
