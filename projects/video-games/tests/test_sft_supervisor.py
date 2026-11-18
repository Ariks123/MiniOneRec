"""Durability and phase-boundary checks without launching GPU work."""
from contextlib import redirect_stdout
import io
import json
import hashlib
import os
import subprocess
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from docker import sft_experiment as experiment
from docker.run_copy import CopyRunner, CONTAINER_WORKSPACE


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run = self.root / "weights/sft"
        self.reports = self.root / "reports"
        self.workspace_patch = patch.object(experiment, "WORKSPACE", self.root)
        self.workspace_patch.start()
        self.addCleanup(self.workspace_patch.stop)

    def config(self):
        args = experiment.parser().parse_args(["start", "--run-dir", str(self.run),
                                                "--report-root", str(self.reports)])
        return experiment.configuration(args)

    def test_original_global_batch_and_sft_only_arguments(self):
        config = self.config()
        self.assertEqual(config["gradient_accumulation_steps"], 16)
        self.assertFalse(config["rl_enabled"])
        args = experiment.training_arguments(config)
        self.assertNotIn("rl", args)
        self.assertEqual(args[args.index("--sft-batch-size") + 1], "1024")
        self.assertEqual(args[args.index("--monitor-window") + 1], "100")

    def test_refuses_to_overwrite_existing_artifacts(self):
        self.run.mkdir(parents=True)
        (self.run / "checkpoint-fixture").write_text("keep")
        with patch.object(experiment.subprocess, "Popen", side_effect=AssertionError("spawned")):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                experiment.start(experiment.parser().parse_args([
                    "start", "--run-dir", str(self.run), "--report-root", str(self.reports)]))
        self.assertEqual((self.run / "checkpoint-fixture").read_text(), "keep")

    def test_duplicate_start_keeps_running_job_untouched(self):
        experiment.atomic_json(self.run / "run_status.json", {"status": "running", "pid": 12345})
        with patch.object(experiment, "alive", return_value=True), \
             patch.object(experiment.subprocess, "Popen", side_effect=AssertionError("spawned")):
            with self.assertRaisesRegex(ValueError, "already running"):
                experiment.start(experiment.parser().parse_args([
                    "start", "--run-dir", str(self.run), "--report-root", str(self.reports)]))

    def test_status_is_read_only_even_when_docker_is_unavailable(self):
        experiment.atomic_json(self.run / "run_status.json", {"status": "running", "pid": 12345})
        output = io.StringIO()
        with patch.object(experiment, "alive", return_value=False), \
             patch.object(experiment.subprocess, "run", side_effect=AssertionError("Docker access")), \
             redirect_stdout(output):
            self.assertEqual(experiment.main(["status", "--run-dir", str(self.run)]), 0)
        self.assertFalse(json.loads(output.getvalue())["supervisor_alive"])

    def test_resume_must_keep_hyperparameters(self):
        config = self.config()
        experiment.atomic_json(self.run / "supervisor_config.json", config)
        experiment.atomic_json(self.run / "run_status.json", {"status": "failed"})
        checkpoint = self.run / "checkpoint-344"
        experiment.atomic_json(checkpoint / "trainer_state.json", {"global_step": 344})
        args = experiment.parser().parse_args([
            "start", "--run-dir", str(self.run), "--report-root", str(self.reports),
            "--resume-from-checkpoint", str(checkpoint), "--learning-rate", "2e-5"])
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            experiment.start(args)

    def test_pipeline_never_evaluates_if_training_evidence_is_missing(self):
        config = self.config()
        self.run.mkdir(parents=True)
        supervisor = experiment.Supervisor(config)
        actions = []
        supervisor.command = lambda action, arguments, stage: actions.append(action)
        supervisor.render_reports = lambda **kwargs: None
        supervisor.sync_small = lambda: None
        with self.assertRaisesRegex(RuntimeError, "completed monitored"):
            supervisor.run_pipeline()
        self.assertEqual(actions, ["prepare-data", "preflight", "sft"])
        self.assertNotIn("rl", actions)

    def test_evaluation_failure_cannot_be_relabelled_completed(self):
        config = self.config()
        self.run.mkdir(parents=True)
        supervisor = experiment.Supervisor(config)
        actions = []

        def command(action, arguments, stage):
            actions.append(action)
            if action == "sft":
                experiment.atomic_json(self.run / "training_summary.json", {"status": "completed"})
            if action == "evaluate-sft":
                experiment.atomic_json(self.run / "evaluation/metrics.json", {"status": "failed"})

        supervisor.command = command
        supervisor.render_reports = lambda **kwargs: None
        supervisor.sync_small = lambda: None
        with self.assertRaisesRegex(RuntimeError, "Final evaluation"):
            supervisor.run_pipeline()
        self.assertNotEqual(supervisor.status["status"], "completed")
        self.assertNotIn("rl", actions)

    def test_success_requires_both_training_and_evaluation_then_pdfs(self):
        config = self.config()
        self.run.mkdir(parents=True)
        supervisor = experiment.Supervisor(config)
        actions = []

        def command(action, arguments, stage):
            actions.append(action)
            if action == "sft":
                experiment.atomic_json(self.run / "training_summary.json", {"status": "completed"})
            if action == "evaluate-sft":
                experiment.atomic_json(self.run / "evaluation/metrics.json", {"status": "passed"})

        def reports(**kwargs):
            for name in experiment.REPORT_DIRECTORIES:
                (self.reports / name).mkdir(parents=True, exist_ok=True)
                (self.reports / name / "report.pdf").write_bytes(b"fixture")
            experiment.atomic_json(self.reports / experiment.REPORT_DIRECTORIES[0] / "training_report.json", {"status": "completed"})
            experiment.atomic_json(self.reports / experiment.REPORT_DIRECTORIES[1] / "validation_report.json", {"status": "passed"})

        supervisor.command, supervisor.render_reports = command, reports
        supervisor.sync_small = lambda: None
        supervisor.run_pipeline()
        self.assertEqual(actions, ["prepare-data", "preflight", "sft", "evaluate-sft"])
        self.assertEqual(supervisor.status["status"], "completed")
        self.assertFalse(supervisor.status["rl_started"])

    def test_host_gpu_log_survives_an_old_container_export(self):
        self.run.mkdir(parents=True)
        (self.run / "host_gpu_metrics.jsonl").write_text('{"timestamp":"new"}\n')
        (self.run / "gpu_metrics.jsonl").write_text('{"timestamp":"old"}\n')
        experiment.Supervisor(self.config()).restore_host_metrics()
        self.assertEqual((self.run / "gpu_metrics.jsonl").read_text(), '{"timestamp":"new"}\n')

    def test_live_sync_timeout_does_not_interrupt_training(self):
        self.run.mkdir(parents=True)
        supervisor = experiment.Supervisor(self.config())
        supervisor.last_gpu = supervisor.last_report = time.monotonic()
        with patch.object(supervisor, "sync_small", side_effect=subprocess.TimeoutExpired("docker cp", 60)), \
             patch.object(supervisor, "is_running", return_value=False), redirect_stdout(io.StringIO()):
            supervisor.tick()
        self.assertGreater(supervisor.last_sync, 0)

    def test_new_container_resume_copies_complete_checkpoint_once(self):
        local = self.root / "weights/checkpoint-344"
        local.mkdir(parents=True)
        state = b'{"global_step":344}\n'
        (local / "trainer_state.json").write_bytes(state)
        container_path = CONTAINER_WORKSPACE + "/weights/checkpoint-344"
        with patch.dict(os.environ, {"ZIWEIZHAO_WORKDIR": str(self.root)}):
            runner = CopyRunner("sft", ["--resume-from-checkpoint", container_path])
        absent = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")
        with patch.object(runner, "helper", return_value=absent) as helper, patch.object(runner, "call") as call:
            runner.upload_resume_if_needed()
        call.assert_called_once_with(["cp", str(local) + "/.", runner.name + ":" + container_path])
        self.assertIn(hashlib.sha256(state).hexdigest(), helper.call_args.args)
        present = subprocess.CompletedProcess([], 0, stdout=hashlib.sha256(state).hexdigest().encode(), stderr=b"")
        with patch.object(runner, "helper", return_value=present), patch.object(runner, "call") as call:
            runner.upload_resume_if_needed()
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
