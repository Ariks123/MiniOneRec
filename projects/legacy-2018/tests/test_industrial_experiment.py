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
        self.assertEqual(args[args.index("--lr_scheduler_type") + 1], "cosine")
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

    def test_original_half_billion_linear_run_does_not_inherit_previous_adapters(self):
        model = experiment.WORKSPACE / "weights/Qwen2.5-0.5B"
        config = {"run_dir": str(experiment.WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_enddonkey_05b_linear/sft"),
                  "base_model": str(model), "lr_scheduler_type": "linear",
                  "deepspeed": None, "gradient_checkpointing": False,
                  "baseline_eval": False, "save_total_limit": 1, "initialize_on_cpu": True}
        args = experiment.train_arguments(config)
        self.assertEqual(args[args.index("--base_model") + 1], str(model))
        self.assertEqual(args[args.index("--lr_scheduler_type") + 1], "linear")
        self.assertEqual(args[args.index("--gradient_checkpointing") + 1], "False")
        self.assertNotIn("--deepspeed", args)
        self.assertEqual(args[args.index("--baseline_eval") + 1], "False")
        self.assertEqual(args[args.index("--save_total_limit") + 1], "1")
        self.assertEqual(args[args.index("--initialize_on_cpu") + 1], "True")
        self.assertIn("--nproc_per_node=4", args)
        self.assertEqual(args[args.index("--batch_size") + 1], "64")
        self.assertEqual(args[args.index("--micro_batch_size") + 1], "8")

    def test_start_preserves_existing_experiment(self):
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT / "tmp") as folder:
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
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT / "tmp") as folder:
            with patch.object(experiment.subprocess, "run") as run, patch.object(experiment.subprocess, "Popen") as popen:
                self.assertEqual(experiment.main(["status", "--run-dir", folder]), 0)
            run.assert_not_called()
            popen.assert_not_called()

    def test_atomic_status_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="industrial-supervisor-", dir=ROOT / "tmp") as folder:
            path = Path(folder) / "run_status.json"
            experiment.atomic_json(path, {"status": "running", "stage": "training"})
            self.assertEqual(experiment.read_json(path)["stage"], "training")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
