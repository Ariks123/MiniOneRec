"""Exercise the persistent-container transport with a recording Docker stub only."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h20_copy_fixture", ROOT / "docker/run_copy.py")
copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copy)


class DockerCopyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h20 copy test ")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.repository = self.folder / "workspace/MiniOneRec"
        raw = self.repository / "data/Amazon18/Video_Games"
        raw.mkdir(parents=True)
        for filename in [*copy.RAW_FILES, "Video_Games.emb-qwen-td.npy", "Video_Games.review.json"]:
            (raw / filename).write_text("fixture", encoding="utf-8")
        self.record = self.folder / "docker calls.jsonl"
        fake = self.folder / "fake docker"
        fake.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "with open(os.environ['FAKE_RECORD'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "args = sys.argv[1:]\n"
            "if args[:2] == ['image', 'inspect']:\n"
            "    print('sha256:fixture-image')\n"
            "if args[:2] == ['container', 'inspect']:\n"
            "    if not os.environ.get('FAKE_CONTAINER_EXISTS'):\n"
            "        print('No such container: fixture', file=sys.stderr)\n"
            "        sys.exit(1)\n"
            "    print(os.environ.get('FAKE_CONTAINER_IMAGE', 'sha256:fixture-image'),\n"
            "          os.environ.get('FAKE_CONTAINER_RUNNING', 'false'))\n"
            "if args[0] == 'exec' and '/usr/local/bin/minionerec' in args:\n"
            "    sys.exit(int(os.environ.get('FAKE_ACTION_STATUS', '0')))\n"
            "if args[0] == 'cp' and args[1].startswith('minionerec-h20-workspace:'):\n"
            "    sys.exit(int(os.environ.get('FAKE_EXPORT_STATUS', '0')))\n"
            "if args[0] == 'exec' and '/bin/bash' in args:\n"
            "    if os.environ.get('FAKE_HELPER_FAILURE'):\n"
            "        print('fixture container authorization denied', file=sys.stderr)\n"
            "        sys.exit(77)\n"
            "    if any('test -f' in arg for arg in args):\n"
            "        sys.exit(int(os.environ.get('FAKE_MODEL_MISSING', '1')))\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.environment = {
            "PATH": os.defpath, "DOCKER": str(fake), "FAKE_RECORD": str(self.record),
            "ZIWEIZHAO_WORKDIR": str(self.repository.parent),
            "MINIONEREC_IMAGE": "minionerec:copy-test",
        }

    def invoke(self, *arguments, **environment):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {**self.environment, **environment}, clear=True), \
             patch.object(copy, "ROOT", self.repository), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            status = copy.main(list(arguments))
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        return status, calls, stdout.getvalue(), stderr.getvalue()

    def test_container_is_stopped_and_retained_without_mount_or_volume_calls(self):
        status, calls, stdout, _ = self.invoke("dry-run", "sft")
        self.assertEqual(status, 0)
        uploads = [call for call in calls if call[0] == "cp"]
        self.assertEqual([Path(call[1]).name for call in uploads], copy.RAW_FILES)
        forbidden = {"volume", "--mount", "-v", "rm", "--rm"}
        self.assertFalse(any(argument in forbidden for call in calls for argument in call))
        self.assertEqual(calls[-1], ["stop", "--time", "10", "minionerec-h20-workspace"])
        self.assertIn("Persistent container retained: minionerec-h20-workspace", stdout)

    def test_dry_run_and_preflight_do_not_convert_data(self):
        for arguments in (("dry-run", "sft"), ("preflight", "rl")):
            with self.subTest(arguments=arguments):
                self.record.unlink(missing_ok=True)
                status, calls, _, _ = self.invoke(*arguments)
                self.assertEqual(status, 0)
                self.assertFalse(any("prepare-data" in call for call in calls))

    def test_gpu_environment_and_secret_names_preserve_arguments(self):
        marker = "dummy-secret-never-log"
        extras = ["--num-epochs", "2", "--learning-rate", "2e-5"]
        status, calls, stdout, stderr = self.invoke(
            "sft", *extras, HF_TOKEN=marker, H20_GPUS="4,5,6,7", H20_SHM_SIZE="24g")
        self.assertEqual(status, 0)
        create = next(call for call in calls if call[0] == "create")
        self.assertEqual(create[create.index("--gpus") + 1], '"device=4,5,6,7"')
        self.assertIn("--shm-size=24g", create)
        for flag in ("--init", "--ipc=host", "--ulimit", "HF_TOKEN"):
            self.assertNotIn(flag, create)
        self.assertEqual(create[-3:], ["/bin/sleep", "minionerec:copy-test", "infinity"])
        action = next(call for call in calls if "/usr/local/bin/minionerec" in call)
        self.assertEqual(action[action.index("HF_TOKEN") - 1], "-e")
        self.assertEqual(action[-len(extras) - 1:], ["sft", *extras])
        self.assertNotIn(marker, json.dumps(calls) + stdout + stderr)

    def test_failed_training_still_exports_before_container_stop(self):
        status, calls, _, _ = self.invoke("rl", FAKE_ACTION_STATUS="23")
        self.assertEqual(status, 23)
        exports = [call for call in calls if call[0] == "cp" and call[1].startswith("minionerec-h20-workspace:")]
        self.assertEqual(len(exports), 1)
        self.assertIn("/weights/checkpoints/MiniOneRec/Video_Games_sid2900/rl/.", exports[0][1])
        self.assertEqual(exports[0][2], str(self.repository.parent / "weights/checkpoints/MiniOneRec/Video_Games_sid2900/rl"))
        self.assertLess(calls.index(exports[0]), len(calls) - 1)
        self.assertEqual(calls[-1][0], "stop")

    def test_failed_export_retains_all_container_data(self):
        status, calls, _, stderr = self.invoke("sft", FAKE_EXPORT_STATUS="1")
        self.assertEqual(status, 1)
        self.assertFalse(any(call[0] == "rm" for call in calls))
        self.assertIn("retained container minionerec-h20-workspace", stderr)
        self.assertEqual(calls[-1][0], "stop")

    def test_successful_download_exports_the_base_model_to_local_weights(self):
        status, calls, _, _ = self.invoke("download-model")
        self.assertEqual(status, 0)
        exports = [call for call in calls if call[0] == "cp"]
        self.assertEqual(len(exports), 1)
        self.assertTrue(exports[0][1].endswith(":/mnt/ossdata/ziweizhao/weights/Qwen2.5-1.5B/."))
        self.assertEqual(exports[0][2], str(self.repository.parent / "weights/Qwen2.5-1.5B"))

    def test_prepare_data_exports_converted_samples(self):
        status, calls, _, _ = self.invoke("prepare-data")
        self.assertEqual(status, 0)
        exported = calls[-2]
        self.assertEqual(exported[0], "cp")
        self.assertIn(copy.PREPARED_RELATIVE + "/.", exported[1])
        self.assertEqual(exported[2], str(self.repository.parent / copy.PREPARED_RELATIVE))

    def test_existing_local_model_is_uploaded_only_if_container_model_is_missing(self):
        model = self.repository.parent / "weights/Qwen2.5-1.5B"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        status, calls, _, _ = self.invoke("preflight", "sft")
        self.assertEqual(status, 0)
        self.assertTrue(any(call[:2] == ["cp", str(model) + "/."] for call in calls))
        self.record.unlink()
        status, calls, _, _ = self.invoke("preflight", "sft", FAKE_MODEL_MISSING="0")
        self.assertEqual(status, 0)
        self.assertFalse(any(call[:2] == ["cp", str(model) + "/."] for call in calls))

    def test_outputs_outside_the_workspace_are_rejected_before_docker(self):
        with self.assertRaisesRegex(ValueError, "inside /mnt/ossdata/ziweizhao"):
            self.invoke("sft", "--output-dir", "/tmp/unpersisted-training-output")
        self.assertFalse(self.record.exists())

    def test_helper_failure_reports_the_daemon_error_and_retains_container(self):
        status, calls, _, stderr = self.invoke("check-env", FAKE_HELPER_FAILURE="1")
        self.assertEqual(status, 1)
        self.assertIn("fixture container authorization denied", stderr)
        self.assertEqual(calls[-1][0], "stop")

    def test_existing_container_with_matching_image_is_reused(self):
        status, calls, _, _ = self.invoke("check-env", FAKE_CONTAINER_EXISTS="1")
        self.assertEqual(status, 0)
        self.assertFalse(any(call[0] == "create" for call in calls))
        self.assertIn(["start", "minionerec-h20-workspace"], calls)
        self.assertTrue(any("/usr/local/bin/minionerec" in call for call in calls))

    def test_running_container_is_rejected_without_exec_or_stop(self):
        status, calls, _, stderr = self.invoke("check-env", FAKE_CONTAINER_EXISTS="1", FAKE_CONTAINER_RUNNING="true")
        self.assertEqual(status, 1)
        self.assertFalse(any(call[0] in {"create", "start", "exec", "stop", "cp", "rm"} for call in calls))
        self.assertIn("already running and was left untouched", stderr)
        self.assertIn("Wait for its current action", stderr)

    def test_different_image_preserves_container_and_requests_a_new_name(self):
        status, calls, _, stderr = self.invoke("check-env", FAKE_CONTAINER_EXISTS="1", FAKE_CONTAINER_IMAGE="sha256:older-image")
        self.assertEqual(status, 1)
        self.assertFalse(any(call[0] in {"create", "start", "stop", "rm", "exec", "cp"} for call in calls))
        self.assertIn("Choose a new MINIONEREC_CONTAINER name", stderr)
        self.assertIn("still contains its models and checkpoints", stderr)


if __name__ == "__main__":
    unittest.main()
