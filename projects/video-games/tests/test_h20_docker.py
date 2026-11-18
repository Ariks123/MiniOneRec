"""Exercise Docker command construction without a daemon, GPU, or training."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "docker/run_h20.sh"


class H20DockerCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h20 docker test ")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.capture = self.folder / "docker argv"
        self.fake_docker = self.folder / "fake docker"
        self.fake_docker.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\0" "$@" > "${FAKE_DOCKER_CAPTURE:?}"\n'
            'exit "${FAKE_DOCKER_EXIT:-0}"\n',
            encoding="utf-8",
        )
        self.fake_docker.chmod(0o755)
        self.environment = {
            "PATH": os.defpath,
            "DOCKER": str(self.fake_docker),
            "FAKE_DOCKER_CAPTURE": str(self.capture),
            "ZIWEIZHAO_WORKDIR": str(self.folder / "mounted workspace"),
            "MINIONEREC_IMAGE": "minionerec:test-h20",
        }

    def invoke(self, *arguments, **environment):
        result = subprocess.run(
            ["bash", str(WRAPPER), *arguments],
            cwd=self.folder,
            env={**self.environment, **environment},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        captured = self.capture.read_bytes().split(b"\0")
        self.assertEqual(captured.pop(), b"")
        return result, [value.decode("utf-8") for value in captured]

    def container_arguments(self, captured):
        position = captured.index(self.environment["MINIONEREC_IMAGE"])
        return captured[:position], captured[position + 1:]

    def test_build_uses_repository_context_and_preserves_build_options(self):
        result, captured = self.invoke(
            "build", "--pull", "--build-arg", "BASE_IMAGE=example/pytorch:custom"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured, [
            "build", "-f", str(ROOT / "docker/Dockerfile.h20"),
            "-t", self.environment["MINIONEREC_IMAGE"], "--pull", "--build-arg",
            "BASE_IMAGE=example/pytorch:custom", str(ROOT),
        ])
        self.assertNotIn("--gpus", captured)

    def test_cpu_actions_do_not_request_gpus(self):
        for arguments in (
            (), ("dry-run", "sft"), ("dry-run", "rl"),
            ("check-env",), ("smoke-test",), ("download-model",),
            ("prepare-data", "--check-only"),
        ):
            with self.subTest(arguments=arguments):
                result, captured = self.invoke(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                options, forwarded = self.container_arguments(captured)
                self.assertNotIn("--gpus", options)
                self.assertEqual(forwarded, list(arguments) if arguments else ["dry-run"])

    def test_training_and_preflight_request_exactly_four_devices(self):
        for arguments in (("sft",), ("rl",), ("preflight", "sft"), ("preflight", "rl"), ("gpu-smoke",)):
            with self.subTest(arguments=arguments):
                result, captured = self.invoke(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                options, forwarded = self.container_arguments(captured)
                self.assertEqual(options.count("--gpus"), 1)
                self.assertEqual(options[options.index("--gpus") + 1], '"device=0,1,2,3"')
                self.assertEqual(forwarded, list(arguments))

    def test_selected_host_devices_have_rank_local_container_indices(self):
        result, captured = self.invoke("sft", H20_GPUS="4,5,6,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        options, _ = self.container_arguments(captured)
        self.assertEqual(options[options.index("--gpus") + 1], '"device=4,5,6,7"')
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3", options)

    def test_credentials_and_configuration_are_passed_by_environment_name(self):
        supplied = {
            "HF_TOKEN": "fixture-secret-never-include-in-argv",
            "HF_ENDPOINT": "https://fixture.invalid",
            "MODEL_REVISION": "fixture-commit",
            "BASE_MODEL": "/mnt/ossdata/ziweizhao/weights/local base",
            "SFT_MODEL": "/mnt/ossdata/ziweizhao/weights/local sft",
            "H20_OUTPUT_ROOT": "/mnt/ossdata/ziweizhao/weights/checkpoints/custom",
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "1",
            "OMP_NUM_THREADS": "7",
        }
        result, captured = self.invoke("dry-run", "rl", **supplied)
        self.assertEqual(result.returncode, 0, result.stderr)
        options, _ = self.container_arguments(captured)
        passed = [options[index + 1] for index, value in enumerate(options) if value == "-e"]
        for name, value in supplied.items():
            self.assertIn(name, passed)
            self.assertNotIn(value, captured)
            self.assertNotIn(name + "=" + value, captured)
        self.assertNotIn(supplied["HF_TOKEN"].encode(), result.stdout + result.stderr)

    def test_training_arguments_remain_individual_literal_arguments(self):
        arguments = [
            "sft", "--base-model", "/mnt/ossdata/ziweizhao/weights/model with spaces",
            "--sft-micro-batch-size", "8", "--learning-rate", "2e-5",
            "--output-dir", "/mnt/ossdata/ziweizhao/weights/checkpoints/$(literal);run",
        ]
        result, captured = self.invoke(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        _, forwarded = self.container_arguments(captured)
        self.assertEqual(forwarded, arguments)

    def test_mounts_and_shared_memory_preserve_workspace_artifacts(self):
        result, captured = self.invoke("dry-run", "sft")
        self.assertEqual(result.returncode, 0, result.stderr)
        options, _ = self.container_arguments(captured)
        mounts = [options[index + 1] for index, value in enumerate(options) if value == "--mount"]
        self.assertEqual(mounts, [
            f"type=bind,source={ROOT},target=/opt/minionerec",
            f"type=bind,source={self.environment['ZIWEIZHAO_WORKDIR']},target=/mnt/ossdata/ziweizhao",
        ])
        self.assertEqual(options[options.index("--shm-size") + 1], "16g")
        self.assertIn("--rm", options)
        self.assertNotIn("--ipc=host", options)

    def test_docker_failure_status_is_propagated(self):
        result, _ = self.invoke("preflight", "sft", FAKE_DOCKER_EXIT="17")
        self.assertEqual(result.returncode, 17)


class H20DockerModelPathTests(unittest.TestCase):
    def load_launcher(self, environment):
        # Use an isolated module so other tests keep their normal import-time defaults.
        with patch.dict(os.environ, environment, clear=True):
            spec = importlib.util.spec_from_file_location("h20_docker_fixture", ROOT / "scripts/launch_h20.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def test_default_base_is_qwen25_15b_base_and_global_batch_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="h20 model ") as directory:
            launcher = self.load_launcher({"ZIWEIZHAO_WORKDIR": directory})
            with patch.dict(os.environ, {}, clear=True):
                plan = launcher.build_plan(launcher.parser().parse_args(["sft"]))
            self.assertEqual(plan["model_dir"], Path(directory) / "weights/Qwen2.5-1.5B")
            self.assertEqual(plan["sft_global_samples_per_step"], 1024)
            self.assertEqual(plan["sft_gradient_accumulation"], 16)
            original_global_batch = 8 * 16 * 8
            self.assertEqual(4 * 16 * plan["sft_gradient_accumulation"], original_global_batch)
            command = plan["command"]
            self.assertEqual(command[command.index("--micro_batch_size") + 1], "16")
            self.assertEqual(command[command.index("--train_from_scratch") + 1], "False")
            with patch.dict(os.environ, {}, clear=True):
                smaller_batch = launcher.build_plan(launcher.parser().parse_args([
                    "sft", "--sft-micro-batch-size", "4"
                ]))
            self.assertEqual(smaller_batch["sft_gradient_accumulation"], 64)
            self.assertEqual(4 * 4 * smaller_batch["sft_gradient_accumulation"], original_global_batch)

    def test_output_root_links_sft_output_to_default_rl_input(self):
        with tempfile.TemporaryDirectory(prefix="h20 checkpoints ") as directory:
            output = Path(directory) / "weights/checkpoints/custom task"
            launcher = self.load_launcher({"H20_OUTPUT_ROOT": str(output)})
            with patch.dict(os.environ, {}, clear=True):
                sft = launcher.build_plan(launcher.parser().parse_args(["sft"]))
                rl = launcher.build_plan(launcher.parser().parse_args(["rl"]))
            self.assertEqual(sft["output_dir"], output / "sft")
            self.assertEqual(rl["output_dir"], output / "rl")
            self.assertEqual(rl["model_dir"], sft["output_dir"] / "final_checkpoint")

    def test_explicit_model_and_output_paths_override_container_defaults(self):
        with tempfile.TemporaryDirectory(prefix="h20 overrides ") as directory:
            launcher = self.load_launcher({"H20_OUTPUT_ROOT": str(Path(directory) / "default")})
            environment = {"BASE_MODEL": directory + "/env-base", "SFT_MODEL": directory + "/env-sft"}
            with patch.dict(os.environ, environment, clear=True):
                sft_args = launcher.parser().parse_args([
                    "sft", "--base-model", directory + "/chosen-base", "--output-dir", directory + "/chosen-output"
                ])
                rl_args = launcher.parser().parse_args(["rl", "--model-path", directory + "/chosen-sft"])
                sft, rl = launcher.build_plan(sft_args), launcher.build_plan(rl_args)
            self.assertEqual(sft["model_dir"], Path(directory) / "chosen-base")
            self.assertEqual(sft["output_dir"], Path(directory) / "chosen-output")
            self.assertEqual(rl["model_dir"], Path(directory) / "chosen-sft")


if __name__ == "__main__":
    unittest.main()
