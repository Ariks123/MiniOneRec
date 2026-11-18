#!/usr/bin/env python3
"""Use a persistent named container when the daemon cannot mount this workspace.

Only Docker CLI arguments are logged by Docker; credentials are passed by name.
The container is stopped and retained after each action. Artifacts use docker cp.
"""
import os
import hashlib
from pathlib import Path, PurePosixPath
import posixpath
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_WORKSPACE = "/mnt/ossdata/ziweizhao"
RAW_RELATIVE = "MiniOneRec/data/Amazon18/Video_Games"
PREPARED_RELATIVE = "MiniOneRec/data/Amazon18/posttraining/Video_Games_sid2900"
RAW_FILES = ["Video_Games." + suffix for suffix in
             ("index.json", "item.json", "train.inter", "valid.inter", "test.inter")]
ENVIRONMENT_NAMES = (
    "HF_TOKEN", "HF_ENDPOINT", "MODEL_REVISION", "BASE_MODEL", "SFT_MODEL", "H20_OUTPUT_ROOT",
    "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_DEBUG",
    "OMP_NUM_THREADS",
)


def option(arguments, name, default):
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments):
                raise ValueError(f"{name} requires a value")
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return default


class CopyRunner:
    def __init__(self, action, arguments):
        self.action, self.arguments = action, arguments
        self.workspace = Path(os.environ.get("ZIWEIZHAO_WORKDIR", str(ROOT.parent))).resolve()
        self.image = os.environ.get("MINIONEREC_IMAGE", "minionerec:h20-cu124")
        self.docker = os.environ.get("DOCKER", "docker")
        self.name = os.environ.get("MINIONEREC_CONTAINER", "minionerec-h20-workspace")
        self.started = False
        self.base_model = option(arguments, "--base-model", os.environ.get(
            "BASE_MODEL", CONTAINER_WORKSPACE + "/weights/Qwen2.5-1.5B"))
        output_root = os.environ.get(
            "H20_OUTPUT_ROOT", CONTAINER_WORKSPACE + "/weights/checkpoints/MiniOneRec/Video_Games_sid2900")
        self.stage_output = option(arguments, "--output-dir", output_root + "/" + action)
        self.sft_model = option(arguments, "--model-path", os.environ.get(
            "SFT_MODEL", output_root + "/sft/final_checkpoint"))
        self.prepared_output = option(arguments, "--output-dir",
                                      CONTAINER_WORKSPACE + "/" + PREPARED_RELATIVE)
        self.resume_checkpoint = option(arguments, "--resume-from-checkpoint", "")
        if action in {"sft", "rl", "evaluate-sft"}:
            self.local_path(self.stage_output)
        if action == "download-model":
            self.local_path(self.base_model)
        if action == "prepare-data":
            self.local_path(self.prepared_output)
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()

    def local_path(self, container_path):
        normalized = PurePosixPath(posixpath.normpath(container_path))
        try:
            relative = normalized.relative_to(CONTAINER_WORKSPACE)
        except ValueError:
            raise ValueError(
                "Docker copy mode requires model/export paths inside " + CONTAINER_WORKSPACE
                + "; received " + str(container_path)
            ) from None
        return self.workspace.joinpath(*relative.parts)

    def call(self, arguments, *, check=True, quiet=False):
        return subprocess.run(
            [self.docker, *arguments], check=check,
            stdout=subprocess.PIPE if quiet else None,
            stderr=subprocess.PIPE if quiet else None,
        )

    def helper(self, script, *arguments, check=True):
        result = self.call([
            "exec", self.name, "/bin/bash", "-c", script, "copy-helper", *arguments,
        ], check=False, quiet=True)
        if check and result.returncode:
            if result.stderr:
                print("Docker container helper: " + result.stderr.decode(errors="replace").strip(),
                      file=sys.stderr)
            result.check_returncode()
        return result

    def file_exists(self, path):
        result = self.helper('test -f "$1"', path, check=False)
        if result.returncode not in (0, 1):
            if result.stderr:
                print("Docker container helper: " + result.stderr.decode(errors="replace").strip(),
                      file=sys.stderr)
            raise RuntimeError("Could not inspect files in the persistent Docker container")
        return result.returncode == 0

    def export_directory(self, container_path):
        destination = self.local_path(container_path)
        destination.mkdir(parents=True, exist_ok=True)
        result = self.call(["cp", f"{self.name}:{container_path}/.", str(destination)], check=False)
        if result.returncode:
            print(f"Export failed; retained container {self.name}. "
                  f"Retry docker cp {self.name}:{container_path}/. {destination}", file=sys.stderr)
            return False
        print(f"Copied artifacts to {destination}", flush=True)
        return True

    def upload_model_if_needed(self, path):
        try:
            local = self.local_path(path)
        except ValueError:
            return  # The normal launcher will report unavailable model paths.
        if (local / "config.json").is_file() and not self.file_exists(path + "/config.json"):
            self.helper('mkdir -p -- "$1"', path)
            self.call(["cp", str(local) + "/.", self.name + ":" + path])

    def upload_resume_if_needed(self):
        if not self.resume_checkpoint:
            return
        local = self.local_path(self.resume_checkpoint)
        state = local / "trainer_state.json"
        if not state.is_file():
            raise ValueError("Resume checkpoint is missing trainer_state.json")
        fingerprint = hashlib.sha256(state.read_bytes()).hexdigest()
        marker = self.resume_checkpoint + "/.h20_resume_copy_complete"
        existing = self.helper('test -f "$1" && cat -- "$1"', marker, check=False)
        if existing.returncode == 0 and existing.stdout.decode().strip() == fingerprint:
            return
        self.helper('mkdir -p -- "$1"', self.resume_checkpoint)
        self.call(["cp", str(local) + "/.", self.name + ":" + self.resume_checkpoint])
        self.helper('printf "%s\\n" "$2" > "$1"', marker, fingerprint)

    def start_container(self):
        # Query only IDs/running state: never inspect or print container environment values.
        image = self.call(["image", "inspect", "--format", "{{.Id}}", self.image], quiet=True)
        expected_image = image.stdout.decode().strip()
        existing = self.call(["container", "inspect", "--format", "{{.Image}} {{.State.Running}}",
                              self.name], check=False, quiet=True)
        running = False
        if existing.returncode == 0:
            actual_image, state = existing.stdout.decode().strip().split()
            if actual_image != expected_image:
                raise RuntimeError(
                    f"Container {self.name} uses a different image and was retained. "
                    "Choose a new MINIONEREC_CONTAINER name for the updated image; "
                    "the previous container still contains its models and checkpoints."
                )
            running = state.lower() == "true"
            if running:
                raise RuntimeError(
                    f"Persistent container {self.name} is already running and was left untouched. "
                    "Wait for its current action to finish, or choose a different MINIONEREC_CONTAINER name."
                )
        else:
            reason = existing.stderr.decode(errors="replace").strip()
            if "no such" not in reason.lower() and "not found" not in reason.lower():
                raise RuntimeError("Could not inspect the persistent container: " + reason)
            self.call([
                "create", "--name", self.name,
                "--shm-size=" + os.environ.get("H20_SHM_SIZE", "16g"),
                "--gpus", '"device=' + os.environ.get("H20_GPUS", "0,1,2,3") + '"',
                "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3",
                "--entrypoint", "/bin/sleep", self.image, "infinity",
            ], quiet=True)
        if not running:
            self.call(["start", self.name], quiet=True)
        self.started = True

    def execute(self):
        self.start_container()
        directories = [CONTAINER_WORKSPACE + "/" + RAW_RELATIVE,
                       CONTAINER_WORKSPACE + "/" + PREPARED_RELATIVE,
                       CONTAINER_WORKSPACE + "/weights/checkpoints",
                       CONTAINER_WORKSPACE + "/.cache"]
        if self.action in {"sft", "rl", "evaluate-sft"}:
            directories.append(self.stage_output)
        self.helper('mkdir -p -- "$@"', *directories)

        if self.action in {"sft", "rl", "preflight", "dry-run", "prepare-data", "evaluate-sft"}:
            for filename in RAW_FILES:
                source = ROOT / "data/Amazon18/Video_Games" / filename
                if source.is_file():
                    self.call(["cp", str(source), self.name + ":" +
                               CONTAINER_WORKSPACE + "/" + RAW_RELATIVE + "/" + filename])
        stage = (self.arguments[0] if self.arguments else "sft") if self.action in {
            "preflight", "dry-run"} else self.action
        if stage == "sft":
            self.upload_model_if_needed(self.base_model)
            self.upload_resume_if_needed()
        elif stage in {"rl", "evaluate-sft"}:
            self.upload_model_if_needed(self.sft_model)

        # Keep secrets out of container creation and its persistent configuration.
        options = ["exec"] + (["-it"] if self.interactive else [])
        for name in ENVIRONMENT_NAMES:
            if name in os.environ:
                options += ["-e", name]
        result = self.call([*options, self.name, "/usr/local/bin/minionerec",
                            self.action, *self.arguments], check=False)
        return result.returncode

    def finish(self, status):
        if not self.started:
            return status
        if status in (130, 143):
            self.call(["stop", "--time", "10", self.name], check=False, quiet=True)
        exported = True
        try:
            if self.action in {"sft", "rl", "evaluate-sft"}:
                exported = self.export_directory(self.stage_output)
            elif status == 0 and self.action == "download-model":
                exported = self.export_directory(self.base_model)
            elif status == 0 and self.action == "prepare-data":
                exported = self.export_directory(self.prepared_output)
        except OSError as error:
            exported = False
            print(f"Export failed; retained container {self.name}: {error}", file=sys.stderr)
        finally:
            stopped = self.call(["stop", "--time", "10", self.name], check=False, quiet=True)
        if stopped.returncode:
            print(f"Could not stop persistent container {self.name}; its files remain retained.",
                  file=sys.stderr)
        print(f"Persistent container retained: {self.name}", flush=True)
        return status if status else (0 if exported and stopped.returncode == 0 else 1)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    action = arguments.pop(0) if arguments else "dry-run"
    runner = CopyRunner(action, arguments)
    status = 1
    try:
        status = runner.execute()
    except KeyboardInterrupt:
        status = 130
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Docker copy mode failed: {error}", file=sys.stderr)
    finally:
        status = runner.finish(status)
    return status


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"Docker copy mode failed: {error}", file=sys.stderr)
        sys.exit(2)
