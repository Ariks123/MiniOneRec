"""Isolation and stage-completion contracts for the new retrieval supervisor."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from docker import industrial_retrieval_experiment as experiment
from qwen_retrieval.report import atomic_json, read_json


class RetrievalSupervisorTests(unittest.TestCase):
    def test_commands_keep_fresh_base_and_test_out_of_training(self):
        config = {"run_dir": "/isolated/sft"}
        train = experiment.train_arguments(config)
        self.assertIn("qwen_retrieval.train", train)
        self.assertIn("--nproc_per_node=4", train)
        self.assertEqual(train[train.index("--base-model") + 1], str(experiment.BASE_MODEL))
        self.assertEqual(train[train.index("--train-file") + 1], str(experiment.DATA_DIR / "train.csv"))
        self.assertNotIn(str(experiment.DATA_DIR / "test.csv"), train)
        self.assertNotIn(str(experiment.HISTORICAL_MODEL), train)
        old = experiment.beam_arguments(config, reference=True)
        new = experiment.beam_arguments(config)
        self.assertEqual(old[old.index("--model-path") + 1], experiment.CONTAINER_REFERENCE)
        self.assertEqual(new[new.index("--model-path") + 1], "/isolated/sft/final_checkpoint")
        for command in (old, new):
            self.assertEqual(command[command.index("--num-beams") + 1], "50")
            self.assertEqual(command[command.index("--prompt-style") + 1], "original-eval")
        valid = experiment.retrieval_arguments(config, "valid")
        self.assertEqual(valid[valid.index("--top-delta") + 1], "200")
        with self.assertRaises(ValueError):
            experiment.retrieval_arguments(config, "train")

    def test_existing_run_is_rejected_before_any_external_action(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            run = Path(folder) / "sft"
            run.mkdir()
            marker = run / "keep.json"
            marker.write_text("keep", encoding="utf-8")
            with patch.object(experiment, "check_prepared") as prepared, patch.object(experiment.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "Existing experiment retained"):
                    experiment.main(["start", "--run-dir", str(run)])
            prepared.assert_not_called()
            popen.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_status_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            with patch.object(experiment.subprocess, "run") as run, patch.object(experiment.subprocess, "Popen") as popen:
                self.assertEqual(experiment.main(["status", "--run-dir", folder]), 0)
            run.assert_not_called()
            popen.assert_not_called()

    def test_status_follows_persisted_active_run_pointer(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            root = Path(folder)
            pointer = root / "active.json"
            atomic_json(pointer, {"run_dir": str(root / "sft")})
            atomic_json(root / "sft/run_status.json", {"status": "running", "stage": "training"})
            with patch.object(experiment, "ACTIVE_RUN", pointer), patch("builtins.print") as output:
                self.assertEqual(experiment.main(["status"]), 0)
            self.assertEqual(json.loads(output.call_args.args[0])["stage"], "training")

    def test_missing_required_source_is_not_silently_omitted(self):
        with patch.object(experiment, "STATIC_SOURCES", ("missing_required_source.py",)):
            with self.assertRaisesRegex(ValueError, "Missing required experiment source"):
                experiment.source_files()

    def test_gpu_preflight_rejects_occupied_devices(self):
        idle = "\n".join(f"{index}, NVIDIA H20, 0, 0" for index in range(4))
        with patch.object(experiment.subprocess, "check_output", return_value=idle):
            self.assertEqual(len(experiment.check_idle_gpus()), 4)
        occupied = "\n".join(f"{index}, NVIDIA H20, {10 if index == 0 else 0}, 0" for index in range(4))
        with patch.object(experiment.subprocess, "check_output", return_value=occupied):
            with self.assertRaisesRegex(ValueError, "occupied"):
                experiment.check_idle_gpus()

    def test_isolation_verification_checks_git_and_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "old"
            old.mkdir()
            source = old / "sft.py"
            source.write_text("original", encoding="utf-8")
            snapshot = Path(folder) / "snapshot.json"
            atomic_json(snapshot, {"old_worktree": str(old), "head": "fixed", "git_status": " M README.md\n",
                                   "sha256": {"sft.py": experiment.hash_file(source)}})
            with patch.object(experiment.subprocess, "check_output", side_effect=["fixed\n", " M README.md\n"]):
                self.assertEqual(experiment.verify_isolation(snapshot)["status"], "passed")
            source.write_text("changed", encoding="utf-8")
            with patch.object(experiment.subprocess, "check_output", side_effect=["different\n", ""]):
                result = experiment.verify_isolation(snapshot)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["errors"]), 3)

    def test_storage_preflight_accounts_for_transient_checkpoints(self):
        result = experiment.storage_requirement(100, 200, 300, 400)
        self.assertEqual(result["checkpoint_estimated_bytes"], 900)
        self.assertEqual(result["checkpoint_slots_including_transient"], 3)
        self.assertEqual(result["required_bytes"], 100 + 200 + 300 + 400 + 2700 + 200 + 2 * 1024**3)
        measured = experiment.storage_requirement(100, 200, 300, 400, checkpoint_bytes=1200)
        self.assertEqual(measured["checkpoint_estimated_bytes"], 1200)

    def test_low_disk_does_not_create_run_or_start_worker(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            run = Path(folder) / "sft"
            base = Path(folder) / "base"
            base.mkdir()
            (base / "model.safetensors").write_bytes(b"model")
            with patch.object(experiment, "BASE_MODEL", base), patch.object(experiment, "check_prepared"), \
                 patch.object(experiment, "verify_isolation", return_value={"status": "passed"}), \
                 patch.object(experiment.subprocess, "check_output", return_value=experiment.IMAGE_ID), \
                 patch.object(experiment, "directory_size", return_value=100), \
                 patch.object(experiment.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)), \
                 patch.object(experiment.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "Insufficient local Docker storage"):
                    experiment.main(["start", "--run-dir", str(run)])
            self.assertFalse(run.exists())
            popen.assert_not_called()

    def test_stage_order_and_isolation_before_completion(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            root = Path(folder)
            config = {"run_dir": str(root / "sft"), "report_root": str(root / "reports"),
                      "container": "isolated", "isolation_snapshot": "snapshot"}
            supervisor = experiment.Supervisor(config)
            supervisor.run.mkdir()
            stages = []
            calls = []
            def command(arguments, stage):
                stages.append(stage)
                names = {"smoke": ("smoke/smoke_result.json", "passed"),
                         "training": ("training_summary.json", "completed"),
                         "encoding": ("item_index/manifest.json", "passed"),
                         "evaluate_valid": ("evaluation_valid/metrics.json", "passed"),
                         "evaluate_test": ("evaluation_test/metrics.json", "passed"),
                         "beam": ("evaluation_beam/metrics.json", "passed"),
                         "reference_beam": ("evaluation_reference_beam/metrics.json", "passed")}
                if stage in names:
                    name, status = names[stage]
                    atomic_json(supervisor.run / name, {"status": status, "expected_rows": 1, "completed_rows": 1})
                if stage == "training":
                    final = supervisor.run / "final_checkpoint"
                    final.mkdir()
                    (final / "config.json").write_text("{}")
                    (final / "model.safetensors").write_bytes(b"fixture")
                if stage == "encoding":
                    for name in ("item_vectors.npy", "row_to_items.json"):
                        (supervisor.run / "item_index" / name).write_bytes(b"fixture")
            supervisor.command = command
            supervisor.call = lambda args, **kwargs: calls.append(args)
            supervisor.prepare_container = lambda: None
            with patch.object(experiment, "check_prepared"), patch.object(experiment, "verify_isolation", return_value={"status": "passed"}) as verify:
                supervisor.execute()
            self.assertEqual(stages, ["smoke", "training", "encoding", "evaluate_valid", "evaluate_test", "beam", "reference_beam"])
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(read_json(supervisor.run / "run_status.json")["status"], "completed")
            self.assertTrue((supervisor.reports / "index.html").is_file())
            self.assertEqual(calls[-1][:1], ["stop"])

    def test_incomplete_evaluation_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor = experiment.Supervisor({"run_dir": folder, "report_root": folder, "container": "unused"})
            atomic_json(Path(folder) / "metrics.json", {"status": "passed", "expected_rows": 10, "completed_rows": 9})
            with self.assertRaisesRegex(RuntimeError, "incomplete row coverage"):
                supervisor.require_passed("metrics.json")

    def test_worker_failure_preserves_failed_stage_and_never_touches_unowned_container(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT) as folder:
            run = Path(folder)
            atomic_json(run / "supervisor_config.json", {"run_dir": str(run), "report_root": str(run / "reports"),
                                                          "container": "unowned", "isolation_snapshot": "snapshot"})
            def fail(supervisor):
                supervisor.update("smoke", status="running")
                raise RuntimeError("synthetic smoke failure")
            with patch.object(experiment.Supervisor, "execute", fail), \
                 patch.object(experiment.Supervisor, "sync_small"), \
                 patch.object(experiment.Supervisor, "call") as call, \
                 patch.object(experiment, "verify_isolation", return_value={"status": "passed"}), \
                 patch.object(experiment.signal, "signal"), patch.object(experiment.traceback, "print_exc"):
                self.assertEqual(experiment.main(["worker", "--run-dir", str(run)]), 1)
            call.assert_not_called()
            status = read_json(run / "run_status.json")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["failed_stage"], "smoke")
            self.assertIn("synthetic smoke failure", status["error"])
            self.assertEqual(read_json(run / "reports/comparison.json")["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
