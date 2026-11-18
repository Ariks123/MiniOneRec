"""CPU-only checks for launch plans; these tests never launch training."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_h20 as h20

class H20LaunchTests(unittest.TestCase):
    def args(self, stage="sft", *extra):
        return h20.parser().parse_args([stage, "--devices", "0,1,2,3", *extra])

    def test_sft_four_workers_preserve_effective_batch(self):
        plan = h20.build_plan(self.args())
        self.assertEqual(plan["sft_gradient_accumulation"], 16)
        self.assertEqual(plan["sft_global_samples_per_step"], 1024)
        self.assertIn("torch.distributed.run", plan["command"])
        pos = plan["command"].index("--nproc_per_node")
        self.assertEqual(plan["command"][pos + 1], "4")
        self.assertNotIn("Embedding", plan["command"])

    def test_rl_complete_groups_and_effective_batch(self):
        plan = h20.build_plan(self.args("rl"))
        self.assertEqual(plan["rl_completions_per_step"], 1024)
        self.assertEqual(plan["rl_prompt_groups_per_step"], 64)
        self.assertIn("accelerate.commands.launch", plan["command"])
        self.assertIn("final_checkpoint", str(plan["model_dir"]))

    def test_rank_local_rl_batch_rejected_even_if_global_divisible(self):
        with self.assertRaisesRegex(ValueError, "each device"):
            h20.build_plan(self.args("rl", "--rl-batch-size", "4"))

    def test_invalid_gpu_sets_and_batch_are_rejected(self):
        for options in (["--devices", "0,1,2"], ["--devices", "0,1,2,2"],
                        ["--sft-batch-size", "1000"], ["--num-generations", "1"],
                        ["--rl-gradient-accumulation", "0"], ["--master-port", "0"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                h20.build_plan(self.args("sft", *options))

    def test_nccl_auto_respects_administrator_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True):
            env = h20.build_plan(self.args())["environment"]
            self.assertNotIn("NCCL_IB_DISABLE", env)
            self.assertNotIn("NCCL_P2P_DISABLE", env)
        with patch.dict(os.environ, {"NCCL_IB_DISABLE": "1"}, clear=True):
            self.assertEqual(h20.build_plan(self.args())["environment"]["NCCL_IB_DISABLE"], "1")
            self.assertEqual(h20.build_plan(self.args("sft", "--nccl-ib-disable", "0"))["environment"]["NCCL_IB_DISABLE"], "0")

    def test_dry_run_reports_missing_inputs_without_runtime_or_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            with patch.object(h20, "check_runtime", side_effect=AssertionError("GPU access")), \
                 patch.object(h20.subprocess, "run", side_effect=AssertionError("training launched")), \
                 redirect_stdout(out):
                result = h20.main(["sft", "--dry-run", "--devices", "0,1,2,3",
                                   "--input-dir", folder, "--data-dir", folder])
            report = json.loads(out.getvalue())
            self.assertEqual(result, 0)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["inputs_ready"])
            self.assertTrue(report["pending_inputs"])

    def test_wrong_sid_is_rejected_before_model_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Video_Games.index.json").write_text("{}", encoding="utf-8")
            plan = h20.build_plan(self.args("sft", "--input-dir", folder, "--data-dir", folder))
            with self.assertRaisesRegex(ValueError, "approved epoch-2900"):
                h20.check_inputs(plan)

    def test_embedding_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"architectures": ["Qwen3Model"]}), encoding="utf-8")
            plan = h20.build_plan(self.args("sft", "--input-dir", folder, "--data-dir", folder, "--base-model", folder))
            with self.assertRaisesRegex(ValueError, "CausalLM"):
                h20.check_inputs(plan)

    def test_prepared_data_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"; raw.mkdir()
            data = root / "data"; data.mkdir()
            for name in ("Video_Games.index.json", "Video_Games.item.json"):
                (raw / name).write_text("{}", encoding="utf-8")
            sources = {"sid_index": {"filename": "Video_Games.index.json", "sha256": h20.sha256(raw / "Video_Games.index.json")},
                       "item_metadata": {"filename": "Video_Games.item.json", "sha256": h20.sha256(raw / "Video_Games.item.json")}}
            outputs = {}
            for name in ("train.csv", "valid.csv", "test.csv", "info.txt"):
                (data / name).write_text("original", encoding="utf-8")
                outputs[name] = {"sha256": h20.sha256(data / name)}
            manifest = {"dataset": "Video_Games", "validation": "passed", "check_only": False,
                        "selected_sid_epoch": 2900, "preserve_all_interaction_rows": True,
                        "sampling_applied": False, "keep_longest_only": False, "sources": sources, "outputs": outputs}
            (data / "preparation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            plan = h20.build_plan(self.args("sft", "--input-dir", str(raw), "--data-dir", str(data),
                                            "--base-model", str(root / "missing-model")))
            with patch.object(h20, "SID_SHA256", sources["sid_index"]["sha256"]):
                self.assertTrue(h20.check_inputs(plan))  # Only the model is pending.
                (data / "train.csv").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed: train.csv"):
                    h20.check_inputs(plan)

    def test_preflight_only_never_spawns_training(self):
        output = io.StringIO()
        with patch.object(h20, "check_inputs", return_value=[]), \
             patch.object(h20, "check_runtime", return_value=["NVIDIA H20"] * 4), \
             patch.object(h20.subprocess, "run", side_effect=AssertionError("training launched")), \
             redirect_stdout(output):
            self.assertEqual(h20.main(["sft", "--devices", "0,1,2,3", "--preflight-only"]), 0)
        self.assertFalse(json.loads(output.getvalue())["training_started"])

    def test_rl_requires_matching_sft_sid_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"; raw.mkdir()
            model = root / "model"; model.mkdir()
            (raw / "Video_Games.index.json").write_text("{}", encoding="utf-8")
            (raw / "Video_Games.item.json").write_text("{}", encoding="utf-8")
            (model / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"fixture only; never loaded")
            plan = h20.build_plan(self.args("rl", "--input-dir", str(raw), "--data-dir", str(root), "--model-path", str(model)))
            approved = h20.sha256(raw / "Video_Games.index.json")
            with patch.object(h20, "SID_SHA256", approved):
                pending = h20.check_inputs(plan)
                self.assertTrue(any("posttraining_provenance" in x for x in pending))
                provenance = {"stage": "sft", "sid_index_sha256": "wrong"}
                path = model / "posttraining_provenance.json"
                path.write_text(json.dumps(provenance), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not trained with the approved"):
                    h20.check_inputs(plan)
                provenance.update(sid_index_sha256=approved, item_meta_sha256=h20.sha256(raw / "Video_Games.item.json"))
                path.write_text(json.dumps(provenance), encoding="utf-8")
                self.assertFalse(any("posttraining_provenance" in x for x in h20.check_inputs(plan)))

    def test_commands_only_pass_existing_training_arguments(self):
        for stage in ("sft", "rl"):
            with self.subTest(stage=stage):
                plan = h20.build_plan(self.args(stage))
                source_path = h20.ROOT / (stage + ".py")
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
                accepted = {x.arg for x in function.args.args + function.args.kwonlyargs}
                script_pos = plan["command"].index(str(source_path))
                options = {x[2:] for x in plan["command"][script_pos + 1:] if x.startswith("--")}
                self.assertLessEqual(options, accepted)

    def test_deepspeed_precision_has_one_authoritative_source(self):
        config = json.loads(h20.DS_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertIs(config["bf16"]["enabled"], True)
        yaml_text = h20.ACCELERATE_CONFIG.read_text(encoding="utf-8")
        self.assertIn("num_processes: 4", yaml_text)
        self.assertIn("deepspeed_config_file: config/h20_zero2.json", yaml_text)
        self.assertNotIn("mixed_precision:", yaml_text)

if __name__ == "__main__":
    unittest.main()
