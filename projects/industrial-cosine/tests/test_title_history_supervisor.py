"""Failure isolation and source integrity for the two-arm SFT supervisor."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "industrial_title_history_experiment", ROOT / "docker/industrial_title_history_experiment.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class TitleHistorySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="comparison-supervisor-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.py"
        self.source.write_text("selected runtime source\n")
        self.plan = {
            "source_sha256": {"sft.py": hashlib.sha256(self.source.read_bytes()).hexdigest()},
            "image_id": "sha256:pinned-image",
            "comparison_reports": str(self.root / "reports"),
            "arms": [
                {"name": name, "include_title_history": enabled,
                 "run_dir": str(self.root / name / "sft"),
                 "report_root": str(self.root / "reports" / name),
                 "container": "isolated-test-" + name}
                for name, enabled in (("baseline", False), ("title_history", True))
            ],
        }
        (self.root / "comparison_plan.json").write_text(json.dumps(self.plan))
        self.comparison = SimpleNamespace(compare_runs=Mock(), write_comparison=Mock())
        self.addCleanup(patch.stopall)
        patch.object(supervisor.experiment, "SOURCE_FILES", ("sft.py",)).start()
        patch.object(supervisor.experiment, "runtime_source", return_value=self.source).start()
        patch.dict(sys.modules, {"scripts.compare_industrial_title_history": self.comparison}).start()

    def status(self):
        return json.loads((self.root / "comparison_status.json").read_text())

    def test_changed_runtime_source_fails_before_any_arm_launch(self):
        self.source.write_text("modified after plan creation\n")
        with patch.object(supervisor.experiment, "main") as launch, \
                patch.object(supervisor.time, "sleep") as sleep, redirect_stderr(io.StringIO()):
            result = supervisor.worker(self.root)
        self.assertEqual(result, 1)
        self.assertEqual(self.status()["status"], "failed")
        self.assertIn("Runtime sources changed", self.status()["error"])
        launch.assert_not_called()
        sleep.assert_not_called()
        self.comparison.compare_runs.assert_not_called()
        self.comparison.write_comparison.assert_not_called()

    def test_first_arm_failure_prevents_second_arm_and_comparison(self):
        def fail_first_arm(arguments):
            run = Path(arguments[arguments.index("--run-dir") + 1])
            run.mkdir(parents=True)
            (run / "run_status.json").write_text(json.dumps({
                "status": "failed", "stage": "training", "error": "injected training failure"}))

        with patch.object(supervisor.experiment, "main", side_effect=fail_first_arm) as launch, \
                patch.object(supervisor.time, "sleep") as sleep, redirect_stderr(io.StringIO()):
            result = supervisor.worker(self.root)
        self.assertEqual(result, 1)
        launch.assert_called_once()
        arguments = launch.call_args.args[0]
        self.assertEqual(arguments[arguments.index("--run-dir") + 1], self.plan["arms"][0]["run_dir"])
        state = self.status()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_arm"], "baseline")
        self.assertIn("injected training failure", state["error"])
        self.assertFalse(Path(self.plan["arms"][1]["run_dir"]).exists())
        sleep.assert_not_called()
        self.comparison.compare_runs.assert_not_called()
        self.comparison.write_comparison.assert_not_called()

    def test_insufficient_storage_refuses_to_start_any_training(self):
        self.plan["minimum_free_disk_bytes"] = 100
        (self.root / "comparison_plan.json").write_text(json.dumps(self.plan))
        with patch.object(supervisor.shutil, "disk_usage", return_value=SimpleNamespace(free=99)), \
                patch.object(supervisor.experiment, "main") as launch, redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.worker(self.root), 1)
        launch.assert_not_called()
        self.assertIn("storage budget", self.status()["error"])

    def test_status_reads_saved_state_without_preflight_launch_or_docker(self):
        expected = {"status": "running", "stage": "training", "current_arm": "baseline"}
        (self.root / "comparison_status.json").write_text(json.dumps(expected))
        output = io.StringIO()
        with patch.object(supervisor.experiment, "check_prepared") as check, \
                patch.object(supervisor, "worker") as worker, \
                patch.object(supervisor.subprocess, "check_output") as docker, \
                patch.object(supervisor.subprocess, "Popen") as launch, redirect_stdout(output):
            result = supervisor.main(["status", "--experiment-root", str(self.root)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertEqual(self.status(), expected)
        check.assert_not_called()
        worker.assert_not_called()
        docker.assert_not_called()
        launch.assert_not_called()

    def test_linear_epoch_plan_runs_three_then_five_and_passes_snapshot_options(self):
        self.plan.update(comparison_kind="linear_epochs", lr_scheduler_type="linear")
        for arm, epochs in zip(self.plan["arms"], (3, 5)):
            arm.update(num_epochs=epochs, include_title_history=True,
                       save_epoch_checkpoint=3 if epochs == 5 else 0,
                       minimum_epochs=3 if epochs == 5 else 0)
        (self.root / "comparison_plan.json").write_text(json.dumps(self.plan))

        def complete_arm(arguments):
            run = Path(arguments[arguments.index("--run-dir") + 1])
            run.mkdir(parents=True)
            (run / "run_status.json").write_text(json.dumps({
                "status": "completed", "stage": "completed", "runtime_image_id": self.plan["image_id"],
                "source_sha256": self.plan["source_sha256"]}))

        self.comparison.write_comparison.return_value = {"comparison.json": "fixture"}
        with patch.dict(sys.modules, {"scripts.compare_industrial_linear_epochs": self.comparison}), \
                patch.object(supervisor.experiment, "main", side_effect=complete_arm) as launch:
            self.assertEqual(supervisor.worker(self.root), 0)
        self.assertEqual(launch.call_count, 2)
        for call, epochs in zip(launch.call_args_list, (3, 5)):
            args = call.args[0]
            self.assertEqual(args[args.index("--num-epochs") + 1], str(epochs))
            self.assertEqual(args[args.index("--lr-scheduler-type") + 1], "linear")
            self.assertEqual(args[args.index("--save-epoch-checkpoint") + 1], "3" if epochs == 5 else "0")
            self.assertEqual(args[args.index("--minimum-epochs") + 1], "3" if epochs == 5 else "0")
            self.assertIn("--no-generative-validation", args)
            self.assertIn("--final-ranking-eval", args)
        self.assertEqual(self.status()["status"], "completed")
        self.comparison.compare_runs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
