"""GPU ownership and immutable-input gates for the deferred beta experiment."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("industrial_rl_beta_queue", ROOT / "docker/queue_industrial_rl_beta.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class BetaQueueTests(unittest.TestCase):
    def test_comparison_module_loads_on_host_without_gpu_libraries(self):
        module = queue.compare_module()
        self.assertIn("evaluation/predictions.jsonl", module.INPUT_FILES)
        self.assertIn("rl_task_manifest.jsonl", module.INPUT_FILES)

    def test_only_terminal_job_and_stopped_container_allow_handoff(self):
        config = {"waiting_container_name": "previous-training"}
        for status, running, expected in (("running", True, False), ("paused", False, False),
                                          ("running", False, False), ("completed", True, False),
                                          ("completed", False, True), ("failed", False, True)):
            with self.subTest(status=status, running=running):
                self.assertEqual(queue.waiting_job_finished(config,
                    {"container": "previous-training", "status": status}, {"Running": running}), expected)
        with self.assertRaisesRegex(ValueError, "different container"):
            queue.waiting_job_finished(config, {"container": "other", "status": "completed"}, {"Running": False})

    def test_idle_gpu_requires_all_four_and_no_compute_process(self):
        idle = "\n".join(f"{i}, NVIDIA H20, 10, 0" for i in range(4))
        self.assertTrue(queue.gpu_idle(idle, "")[0])
        for gpu, processes in ((idle, "1234, python, 100"), (idle.replace("10, 0", "4000, 90", 1), ""),
                               ("0, NVIDIA H20, 10, 0", ""), (idle.replace("10, 0", "N/A, 0", 1), "")):
            self.assertFalse(queue.gpu_idle(gpu, processes)[0])

    def test_wait_does_not_launch_on_pause_or_training_evaluation_gap(self):
        config = {"waiting_container_name": "previous-training", "waiting_container_id": "fixed-id",
                  "waiting_status": "unused", "poll_seconds": 30}
        jobs = [{"container": "previous-training", "status": value, "stage": "evaluating"}
                for value in ("running", "running", "completed")]
        states = [{"Running": True}, {"Running": False}, {"Running": False}]
        worker = queue.Queue("unused", config)
        worker.update = lambda *args, **kwargs: None
        with patch.object(queue.rl, "read_json", side_effect=jobs), \
                patch.object(queue, "container_state", side_effect=states) as inspect, \
                patch.object(queue.time, "sleep") as sleep, patch.object(queue, "execute") as execute:
            worker.wait_for_job()
        self.assertEqual(inspect.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(inspect.call_args.args, ("fixed-id",))
        execute.assert_not_called()

    def test_changed_pinned_source_fails_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trainer.py"
            path.write_text("reviewed source")
            snapshot = queue.files_snapshot([path])
            queue.verify_snapshot(snapshot)
            path.write_text("different source")
            with self.assertRaisesRegex(ValueError, "changed"):
                queue.verify_snapshot(snapshot)

    def test_insufficient_storage_waits_without_starting_a_container(self):
        worker = queue.Queue("unused", {"poll_seconds": 30})
        worker.update = lambda *args, **kwargs: None
        with patch.object(queue.rl, "check_storage", side_effect=[ValueError("too little space"), 60]), \
                patch.object(queue.time, "sleep") as sleep, patch.object(queue, "docker") as docker:
            worker.wait_for_storage()
        sleep.assert_called_once_with(30)
        docker.assert_not_called()

    def test_launcher_uses_new_beta_and_distinct_output_paths(self):
        config = {"run_dir": "/new/run", "report_root": "/new/report", "container": "new-rl",
                  "image": "sha256:fixed"}
        arguments = queue.launch_arguments(config)
        for flag, value in (("--beta", "0.01"), ("--run-dir", "/new/run"),
                            ("--container", "new-rl"), ("--image", "sha256:fixed")):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        self.assertNotIn("--resume_from_checkpoint", arguments)

    def test_status_is_read_only_and_retains_existing_queue(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder)
            queue.rl.atomic_json(path / "status.json", {"status": "waiting"})
            with patch.object(queue, "docker") as docker, patch.object(queue.subprocess, "Popen") as popen:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(queue.main(["status", "--queue-dir", folder]), 0)
                with self.assertRaisesRegex(ValueError, "Existing queue retained"):
                    queue.main(["start", "--queue-dir", folder, "--waiting-container", "previous",
                                "--waiting-status", folder + "/previous.json"])
            docker.assert_not_called()
            popen.assert_not_called()
            self.assertEqual(queue.rl.read_json(path / "status.json"), {"status": "waiting"})

    def test_cleanup_touches_only_owned_observer_or_reporting_container(self):
        worker = queue.Queue("unused", {"container": "own-new-rl"})
        with patch.object(queue, "docker") as docker:
            worker.cleanup()
            docker.assert_not_called()
            worker.probe = "own-probe-id"
            worker.cleanup()
            docker.assert_called_once_with(["rm", "--force", "own-probe-id"], check=False)
        worker.probe = None
        worker.comparison_container_started = True
        with patch.object(queue, "docker") as docker:
            worker.cleanup()
            docker.assert_called_once_with(["stop", "--timeout", "2", "own-new-rl"], check=False)


if __name__ == "__main__":
    unittest.main()
