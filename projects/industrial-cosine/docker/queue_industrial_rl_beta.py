#!/usr/bin/env python3
"""Wait for an existing GPU job, run the beta-only RL experiment, and compare it."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docker import industrial_rl_experiment as rl

TERMINAL = {"completed", "failed", "cancelled", "aborted", "stopped"}
BASELINE = rl.RUN
DEFAULT_ROOT = rl.WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_beta0p01_20261118"
COMPARE_SCRIPT = ROOT / "scripts/compare_industrial_rl_beta.py"


def execute(arguments, *, timeout=120, check=True):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError("Command failed: " + " ".join(arguments[:4]) + "\n" + result.stderr[-2500:])
    return result


def docker(arguments, **kwargs):
    return execute([os.environ.get("DOCKER", "docker"), *arguments], **kwargs)


def container_state(identity):
    result = docker(["inspect", "--format", "{{json .State}}", identity], check=False)
    if result.returncode:
        if "no such" in result.stderr.lower() or "not found" in result.stderr.lower():
            return {"Running": False, "Status": "removed"}
        raise RuntimeError("Cannot inspect waiting container: " + result.stderr[-1500:])
    return json.loads(result.stdout)


def waiting_job_finished(config, job, state):
    rl.require(job.get("container") == config["waiting_container_name"],
               "Waiting-job status belongs to a different container")
    # A pause, an evaluation gap, or low GPU utilization is not job completion.
    return job.get("status") in TERMINAL and not state.get("Running", True)


def gpu_idle(gpu_csv, process_csv):
    try:
        rl.validate_gpu_idle(gpu_csv)
        rl.require(not process_csv.strip(), "Another GPU compute process is still running")
        return True, None
    except ValueError as error:
        return False, str(error)


def files_snapshot(paths):
    result = {}
    for path in paths:
        path = Path(path)
        rl.require(path.is_file(), "Required experiment input is missing: " + str(path))
        result[str(path)] = rl.sha256(path)
    return result


def verify_snapshot(expected):
    rl.require(files_snapshot(expected) == expected,
               "Queued source or baseline changed; the reviewed experiment was not launched")


def launch_arguments(config):
    return [sys.executable, str(ROOT / "docker/industrial_rl_experiment.py"), "start",
            "--beta", "0.01", "--run-dir", config["run_dir"],
            "--report-root", config["report_root"], "--container", config["container"],
            "--image", config["image"]]


def compare_module():
    spec = importlib.util.spec_from_file_location("scripts.compare_industrial_rl_beta", COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Queue:
    def __init__(self, directory, config):
        self.directory, self.config = Path(directory), config
        self.probe = None
        self.comparison_container_started = False

    def update(self, stage, **values):
        state = rl.read_json(self.directory / "status.json", {})
        state.update(stage=stage, updated_at=rl.now(), worker_pid=os.getpid(), **values)
        rl.atomic_json(self.directory / "status.json", state)
        # Status is durable even when the interactive Codex session has ended.
        print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)

    def wait_for_job(self):
        while True:
            job = rl.read_json(Path(self.config["waiting_status"]), {})
            try:
                state = container_state(self.config["waiting_container_id"])
                finished = waiting_job_finished(self.config, job, state)
                reason = None
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                finished, reason = False, str(error)
            self.update("waiting_for_job", status="waiting", waiting_job_status=job.get("status"),
                        waiting_job_stage=job.get("stage"), probe_error=reason)
            if finished:
                self.update("waiting_for_gpu", waiting_job_finished_at=rl.now(),
                            waiting_job_final_status=job.get("status"))
                return
            time.sleep(self.config["poll_seconds"])

    def wait_for_gpu(self):
        # This observer is created only after the preceding job has fully ended.
        # nvidia-smi does not create a CUDA compute context or allocate model memory.
        result = docker(["create", "--name", self.config["probe_container"],
                         "--label", "codex.experiment=industrial-rl-beta-comparison-probe",
                         "--gpus", '"device=0,1,2,3"', "--entrypoint", "/bin/sleep",
                         self.config["image"], "infinity"])
        self.probe = result.stdout.strip()
        rl.require(bool(self.probe), "GPU observer has no container identity")
        docker(["start", self.probe])
        streak = 0
        while streak < self.config["idle_samples"]:
            try:
                gpu = docker(["exec", self.probe, "nvidia-smi",
                              "--query-gpu=index,name,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"]).stdout
                processes = docker(["exec", self.probe, "nvidia-smi",
                                    "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"]).stdout
                idle, reason = gpu_idle(gpu, processes)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                idle, reason = False, str(error)
            streak = streak + 1 if idle else 0
            self.update("waiting_for_gpu", status="waiting", idle_samples=streak,
                        required_idle_samples=self.config["idle_samples"], probe_error=reason)
            if streak < self.config["idle_samples"]:
                time.sleep(self.config["poll_seconds"])
        docker(["rm", "--force", self.probe])
        self.probe = None

    def wait_for_storage(self):
        while True:
            try:
                free = rl.check_storage()
                self.update("storage_ready", free_tmp_gib=free)
                return
            except ValueError as error:
                self.update("waiting_for_storage", status="waiting", storage_error=str(error))
                time.sleep(self.config["poll_seconds"])

    def wait_for_training(self):
        path = Path(self.config["run_dir"]) / "run_status.json"
        while True:
            state = rl.read_json(path, {})
            self.update("training_and_evaluation", status="running", rl_status=state.get("status"),
                        rl_stage=state.get("stage"),
                        progress=state.get("progress", {}).get("latest"))
            if state.get("status") == "completed":
                # Completion is written before the supervisor's final report and
                # container stop. Wait for that cleanup before CPU comparison.
                if not container_state(self.config["container"]).get("Running", True):
                    return
            elif state.get("status") in TERMINAL:
                raise RuntimeError("RL experiment did not complete: " + str(state.get("error", state)))
            time.sleep(self.config["poll_seconds"])

    def compare(self):
        self.update("comparing", status="running")
        verify_snapshot(self.config["pinned_files"])
        baseline = Path(self.config["baseline_run"])
        container = self.config["container"]
        # The owned training container runs /bin/sleep as its entrypoint. Restart
        # it solely for CPU reporting; never restart either training command.
        docker(["start", container])
        self.comparison_container_started = True
        docker(["cp", str(COMPARE_SCRIPT), container + ":/opt/minionerec/scripts/" + COMPARE_SCRIPT.name])
        module = compare_module()
        with tempfile.TemporaryDirectory(prefix="industrial-rl-beta-baseline-") as folder:
            temporary = Path(folder)
            for name in module.INPUT_FILES:
                source = baseline / name
                rl.require(source.is_file(), "Baseline comparison input is missing: " + name)
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            docker(["exec", container, "mkdir", "-p", str(baseline)])
            docker(["cp", str(temporary) + "/.", container + ":" + str(baseline)], timeout=300)
        comparison = Path(self.config["comparison_root"])
        result = docker(["exec", "-w", "/opt/minionerec", "-e", "CUDA_VISIBLE_DEVICES=",
                         "-e", "MPLCONFIGDIR=/tmp/industrial-rl-beta-matplotlib", container,
                         "python", "scripts/" + COMPARE_SCRIPT.name,
                         "--baseline-run", str(baseline), "--candidate-run", self.config["run_dir"],
                         "--report-root", str(comparison)], timeout=600)
        (self.directory / "comparison.log").write_text(result.stdout + result.stderr)
        comparison.mkdir(parents=True, exist_ok=True)
        docker(["cp", container + ":" + str(comparison) + "/.", str(comparison)], timeout=180)
        report = rl.read_json(comparison / "comparison.json", {})
        rl.require(report.get("status") == "passed", "Beta comparison did not pass validation")
        docker(["stop", "--timeout", "2", container])
        self.comparison_container_started = False
        self.update("completed", status="completed", finished_at=rl.now(),
                    comparison_report=str(comparison / "comparison.html"))

    def run(self):
        key = hashlib.sha256(str(self.directory.resolve()).encode()).hexdigest()[:20]
        with open("/tmp/industrial-rl-beta-" + key + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_snapshot(self.config["pinned_files"])
            self.wait_for_job()
            self.wait_for_storage()
            self.wait_for_gpu()
            verify_snapshot(self.config["pinned_files"])
            self.update("launching", status="running", training_started_at=rl.now())
            result = execute(launch_arguments(self.config), timeout=300)
            (self.directory / "launch.log").write_text(result.stdout + result.stderr)
            actual = rl.read_json(Path(self.config["run_dir"]) / "supervisor_config.json", {})
            rl.require(actual.get("beta") == 0.01, "Started RL configuration does not use beta=0.01")
            self.wait_for_training()
            self.compare()

    def cleanup(self):
        if self.probe:
            docker(["rm", "--force", self.probe], check=False)
        if self.comparison_container_started:
            docker(["stop", "--timeout", "2", self.config["container"]], check=False)


def workspace_path(path):
    path = Path(path).resolve()
    path.relative_to(rl.WORKSPACE.resolve())
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_ROOT / "queue")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_ROOT / "rl")
    parser.add_argument("--baseline-run", type=Path, default=BASELINE)
    parser.add_argument("--report-root", type=Path, default=ROOT / "output/reports/industrial_05b_rl_beta0p01_20261118")
    parser.add_argument("--comparison-root", type=Path, default=ROOT / "output/reports/industrial_05b_rl_beta_comparison_20261118")
    parser.add_argument("--container", default="minionerec-industrial05b-rl-beta0p01-20261118")
    parser.add_argument("--waiting-container")
    parser.add_argument("--waiting-status", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--idle-samples", type=int, default=3)
    args = parser.parse_args(argv)
    directory = workspace_path(args.queue_dir)
    if args.action == "status":
        print(json.dumps(rl.read_json(directory / "status.json", {"status": "not_queued"}), ensure_ascii=False, indent=2))
        return 0
    if args.action == "start":
        rl.require(args.waiting_container and args.waiting_status,
                   "Both the waiting container and its job-status file are required")
        rl.require(args.poll_seconds >= 10 and args.idle_samples >= 3,
                   "Require at least three GPU-idle observations, at least ten seconds apart")
        rl.require(not directory.exists(), "Existing queue retained; choose a new --queue-dir")
        for path in (args.run_dir, args.report_root, args.comparison_root):
            path = workspace_path(path)
            rl.require(not path.exists() or not any(path.iterdir()), "Existing experiment output retained: " + str(path))
        baseline = workspace_path(args.baseline_run)
        original = rl.read_json(baseline / "supervisor_config.json", {})
        rl.require(original.get("beta") == 0.001 and
                   rl.read_json(baseline / "run_status.json", {}).get("status") == "completed",
                   "Comparison requires the completed beta=0.001 baseline")
        parent = rl.check_prepared()
        rl.require(original.get("parent_model_sha256") == parent["model_sha256"],
                   "Baseline and candidate must start from identical SFT weights")
        # Trainer/reward/data sources must remain identical across the two RL runs.
        source_hashes = rl.source_fingerprints()
        training_files = [name for name in rl.SOURCE_FILES if "report_" not in name]
        rl.require(all(original["source_sha256"].get(name) == source_hashes[name]
                       for name in training_files), "Training sources differ from the beta=0.001 baseline")
        waiting_path = workspace_path(args.waiting_status)
        waiting = rl.read_json(waiting_path, {})
        identity = docker(["inspect", "--format", "{{.Id}} {{.Name}}", args.waiting_container]).stdout.strip().split()
        rl.require(len(identity) == 2 and waiting.get("container") == identity[1].lstrip("/"),
                   "Waiting job and container identity do not match")
        image = docker(["image", "inspect", "--format", "{{.Id}}", original["image"]]).stdout.strip()
        rl.require(image == parent["runtime_image_id"], "RL comparison runtime image differs")
        baseline_files = [baseline / name for name in compare_module().INPUT_FILES]
        pinned = files_snapshot([*(ROOT / name for name in source_hashes), Path(__file__), COMPARE_SCRIPT,
                                 *baseline_files])
        config = {"created_at": rl.now(), "baseline_run": str(baseline), "beta": 0.01,
                  "baseline_beta": 0.001, "run_dir": str(workspace_path(args.run_dir)),
                  "report_root": str(workspace_path(args.report_root)), "image": image,
                  "comparison_root": str(workspace_path(args.comparison_root)), "container": args.container,
                  "probe_container": args.container + "-idle-probe", "pinned_files": pinned,
                  "waiting_container_id": identity[0], "waiting_container_name": identity[1].lstrip("/"),
                  "waiting_status": str(waiting_path), "poll_seconds": args.poll_seconds,
                  "idle_samples": args.idle_samples, "parent_model_sha256": parent["model_sha256"],
                  "expected_training_hours": [1.5, 2.0]}
        config["launch_arguments"] = launch_arguments(config)
        directory.mkdir(parents=True, exist_ok=False)
        rl.atomic_json(directory / "config.json", config)
        rl.atomic_json(directory / "status.json", {"status": "waiting", "stage": "queued",
                       "created_at": config["created_at"], "updated_at": rl.now(),
                       "run_dir": config["run_dir"], "baseline_run": str(baseline), "beta": 0.01,
                       "waiting_container": config["waiting_container_name"]})
        with (directory / "worker.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                        "worker", "--queue-dir", str(directory)], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        state = rl.read_json(directory / "status.json")
        state["worker_pid"] = process.pid
        rl.atomic_json(directory / "status.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    config = rl.read_json(directory / "config.json")
    rl.require(isinstance(config, dict), "Queue has no saved configuration")
    queue = Queue(directory, config)
    try:
        queue.run()
        return 0
    except BaseException as error:
        traceback.print_exc()
        queue.update("failed", status="failed", error=str(error), finished_at=rl.now())
        return 1
    finally:
        queue.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
