#!/usr/bin/env python3
"""Durable, isolated Qwen 0.5B joint generation/retrieval experiment."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docker import industrial_experiment as legacy
from qwen_retrieval.config import (APPROVED_SID_SHA256, BASE_MODEL, CATEGORY, DATA_DIR,
                                   HISTORICAL_MODEL, INPUT_DIR, WORKSPACE, hash_file)
from qwen_retrieval.report import atomic_json, read_json, write_report

IMAGE = "minionerec:h20-original-torch28"
IMAGE_ID = "sha256:0da0b47810fbd4027b860446cabe30a5cb31b48dca96a76c2ac0002143957eac"
ISOLATION_SNAPSHOT = WORKSPACE / "weights/checkpoints/MiniOneRec/retrieval_isolation_20261118.json"
ACTIVE_RUN = ROOT / "output/qwen_retrieval_active_run.json"
CONTAINER_REFERENCE = "/opt/minionerec/reference_checkpoint"
STATIC_SOURCES = ("data.py", "sft_monitor.py", "LogitProcessor.py", "npu_support.py", "sft.py",
                  "docker/__init__.py", "docker/industrial_experiment.py", "docker/industrial_retrieval_experiment.py",
                  "scripts/evaluate_sft.py", "scripts/sft_eval_utils.py",
                  "scripts/deepspeed_zero2_compat.py", "scripts/industrial_generative_validation.py",
                  "scripts/report_industrial_sft.py", "config/h20_zero2.json")


def source_files():
    names = list(STATIC_SOURCES)
    for name in names:
        if not (ROOT / name).is_file():
            raise ValueError("Missing required experiment source: " + name)
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "qwen_retrieval").rglob("*.py"))]
    return names


def torchrun(module):
    return ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4", "-m", module]


def train_arguments(config):
    return torchrun("qwen_retrieval.train") + [
        "--base-model", str(BASE_MODEL), "--train-file", str(DATA_DIR / "train.csv"),
        "--eval-file", str(DATA_DIR / "valid.csv"), "--output-dir", config["run_dir"],
        "--sid-index-path", str(INPUT_DIR / (CATEGORY + ".index.json")),
        "--item-meta-path", str(INPUT_DIR / (CATEGORY + ".item.json")),
        "--deepspeed", "/opt/minionerec/config/h20_zero2.json"]


def smoke_arguments(config):
    return torchrun("qwen_retrieval.smoke") + ["--output-dir", str(Path(config["run_dir"]) / "smoke")]


def encode_arguments(config):
    run = Path(config["run_dir"])
    return ["python", "-m", "qwen_retrieval.encode_items", "--model-path", str(run / "final_checkpoint"),
            "--input-dir", str(INPUT_DIR), "--output-dir", str(run / "item_index"), "--batch-size", "128"]


def retrieval_arguments(config, split):
    if split not in ("valid", "test"):
        raise ValueError("Retrieval evaluation split must be valid or test")
    run = Path(config["run_dir"])
    return torchrun("qwen_retrieval.evaluate") + [
        "--model-path", str(run / "final_checkpoint"), "--index-dir", str(run / "item_index"),
        "--data-dir", str(DATA_DIR), "--input-dir", str(INPUT_DIR),
        "--output-dir", str(run / f"evaluation_{split}"), "--split", split,
        "--batch-size", "4", "--candidate-batch-size", "32", "--top-delta", "200"]


def beam_arguments(config, reference=False):
    run = Path(config["run_dir"])
    return ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4",
            "scripts/evaluate_sft.py", "--model-path", CONTAINER_REFERENCE if reference else str(run / "final_checkpoint"),
            "--data-dir", str(DATA_DIR), "--input-dir", str(INPUT_DIR),
            "--output-dir", str(run / ("evaluation_reference_beam" if reference else "evaluation_beam")),
            "--category", CATEGORY, "--expected-sid-sha256", APPROVED_SID_SHA256,
            "--prompt-style", "original-eval", "--num-beams", "50", "--batch-size", "4"]


def verify_isolation(snapshot_path):
    snapshot = read_json(snapshot_path)
    if not snapshot:
        raise ValueError("Missing old-worktree isolation snapshot: " + str(snapshot_path))
    old = Path(snapshot["old_worktree"])
    errors = []
    head = subprocess.check_output(["git", "-C", str(old), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(old), "status", "--short"], text=True)
    if head != snapshot["head"]:
        errors.append("Old worktree HEAD changed")
    if status != snapshot["git_status"]:
        errors.append("Old worktree git status changed")
    for name, expected in snapshot["sha256"].items():
        path = old / name
        if not path.is_file() or hash_file(path) != expected:
            errors.append("Old artifact changed: " + name)
    return {"status": "failed" if errors else "passed", "checked_at": legacy.now(),
            "snapshot_path": str(snapshot_path), "old_worktree": str(old),
            "files_checked": len(snapshot["sha256"]), "errors": errors}


def directory_size(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def storage_requirement(base_bytes, reference_bytes, input_bytes, source_bytes=0, checkpoint_bytes=0):
    # Actual historical ZeRO-2 snapshots include both standard model weights
    # and DeepSpeed model states, plus all four optimizer shards. Reserve at
    # least nine times Base weights and never undercut the measured snapshot.
    estimated_checkpoint_bytes = max(9 * base_bytes, checkpoint_bytes)
    return {"base_model_bytes": base_bytes, "reference_model_bytes": reference_bytes,
            "input_bytes": input_bytes, "source_bytes": source_bytes,
            "checkpoint_measured_bytes": checkpoint_bytes,
            "checkpoint_estimated_bytes": estimated_checkpoint_bytes, "checkpoint_slots_including_transient": 3,
            "required_bytes": base_bytes + reference_bytes + input_bytes + source_bytes
                              + 3 * estimated_checkpoint_bytes + 2 * base_bytes + 2 * 1024**3,
            "assumptions": "max(nine times Base weights, measured historical model + ZeRO-2 states + all optimizer shards); two retained checkpoints plus one transient save; final export and smoke reserve"}


def check_prepared():
    legacy.check_prepared(BASE_MODEL)
    source_files()
    for name in ("train.py", "smoke.py", "encode_items.py", "evaluate.py", "model.py", "report.py"):
        if not (ROOT / "qwen_retrieval" / name).is_file():
            raise ValueError("Missing retrieval source: " + name)
    if not (HISTORICAL_MODEL / "model.safetensors").is_file():
        raise ValueError("Historical best checkpoint is unavailable")


def check_idle_gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 4 or any(len(row) != 4 or "H20" not in row[1] for row in rows):
        raise ValueError("Experiment requires exactly four visible H20 GPUs")
    if any(int(row[2]) > 0 or int(row[3]) > 0 for row in rows):
        raise ValueError("H20 GPUs are occupied; existing processes were left untouched")
    return [{"index": int(row[0]), "name": row[1], "memory_used_mib": int(row[2]),
             "utilization_percent": int(row[3])} for row in rows]


class Supervisor(legacy.Supervisor):
    def __init__(self, config):
        super().__init__(config)
        self.created_container = False

    def render_reports(self):
        return write_report(self.run, self.reports)

    def prepare_container(self):
        image_id = self.call(["image", "inspect", "--format", "{{.Id}}", self.config["image"]]).stdout.decode().strip()
        if image_id != self.config["image_id"]:
            raise ValueError("Fixed training image identity changed")
        existing = self.call(["container", "inspect", self.container], check=False)
        if existing.returncode == 0:
            raise ValueError("Existing container retained; choose a new container name")
        if "no such" not in existing.stderr.decode(errors="replace").lower():
            raise RuntimeError("Cannot establish that experiment container is new")
        for name, expected in self.config["source_sha256"].items():
            if hash_file(ROOT / name) != expected:
                raise ValueError("Experiment source changed after start: " + name)
        self.call(["create", "--name", self.container, "--shm-size", "16g", "--gpus", '"device=0,1,2,3"',
                   "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3", "--entrypoint", "/bin/sleep", self.config["image"], "infinity"])
        self.created_container = True
        self.call(["start", self.container])
        directories = {str(Path("/opt/minionerec") / Path(name).parent) for name in self.config["source_sha256"]}
        directories.update(map(str, (INPUT_DIR, DATA_DIR, BASE_MODEL, self.run, self.reports)))
        directories.add(CONTAINER_REFERENCE)
        self.call(["exec", self.container, "mkdir", "-p", *sorted(directories)])
        for name in self.config["source_sha256"]:
            self.call(["cp", str(ROOT / name), f"{self.container}:/opt/minionerec/{name}"])
        for directory in (INPUT_DIR, DATA_DIR, BASE_MODEL):
            self.call(["cp", str(directory) + "/.", f"{self.container}:{directory}"], timeout=900)
        self.call(["cp", str(HISTORICAL_MODEL) + "/.", f"{self.container}:{CONTAINER_REFERENCE}"], timeout=900)
        self.call(["exec", self.container, "chmod", "-R", "a-w", CONTAINER_REFERENCE])
        result = self.call(["exec", self.container, "python", "-c",
                           "import torch,transformers,datasets,accelerate,json,sys; "
                           "assert torch.cuda.device_count()==4; "
                           "assert all('H20' in torch.cuda.get_device_name(i) for i in range(4)); "
                           "print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,"
                           "datasets=datasets.__version__,accelerate=accelerate.__version__,cuda=torch.version.cuda,"
                           "python=sys.version.split()[0],devices=[torch.cuda.get_device_name(i) for i in range(4)])))"])
        self.status["runtime_versions"] = json.loads(result.stdout.decode().strip().splitlines()[-1])
        self.status["runtime_image_id"] = image_id
        self.update()

    def sync_small(self):
        names = ["run_config.json", "training_summary.json", "trainer_state.json", "baseline_validation.json",
                 "monitor/metrics.jsonl", "monitor/retrieval_metrics.jsonl", "monitor/summary.json", "monitor/events.jsonl",
                 "smoke/smoke_result.json", "item_index/manifest.json"]
        for directory in ("evaluation_valid", "evaluation_test", "evaluation_beam", "evaluation_reference_beam"):
            names.append(directory + "/metrics.json")
            names += [f"{directory}/progress.rank{rank}.json" for rank in range(4)]
        import tempfile
        for name in names:
            destination = self.run / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".retrieval-sync-", dir=destination.parent) as folder:
                temporary = Path(folder) / destination.name
                result = self.call(["cp", f"{self.container}:{self.run}/{name}", str(temporary)], check=False, timeout=30)
                if result.returncode == 0 and temporary.is_file():
                    os.replace(temporary, destination)
        self.update()

    def command(self, arguments, stage):
        started = time.monotonic()
        started_at = legacy.now()
        success = False
        try:
            super().command(arguments, stage)
            success = True
        finally:
            timings = read_json(self.run / "stage_timings.json", {})
            timings[stage] = {"started_at": started_at, "finished_at": legacy.now(),
                              "wall_seconds": time.monotonic() - started,
                              "status": "completed" if success else "failed", "command": arguments}
            atomic_json(self.run / "stage_timings.json", timings)

    def require_passed(self, name, expected="passed"):
        record = read_json(self.run / name, {})
        if record.get("status") != expected:
            raise RuntimeError(f"Stage result did not report {expected}: {name}")
        if "expected_rows" in record and record.get("expected_rows") != record.get("completed_rows"):
            raise RuntimeError("Evaluation has incomplete row coverage: " + name)

    def execute(self):
        self.update("preparing", status="running")
        check_prepared()
        isolation = verify_isolation(self.config["isolation_snapshot"])
        atomic_json(self.run / "isolation_verification_start.json", isolation)
        if isolation["status"] != "passed":
            raise RuntimeError("Old worktree isolation check failed before training")
        self.prepare_container()
        self.render_reports()
        self.command(smoke_arguments(self.config), "smoke")
        self.require_passed("smoke/smoke_result.json")
        self.command(train_arguments(self.config), "training")
        self.require_passed("training_summary.json", "completed")
        self.command(encode_arguments(self.config), "encoding")
        self.require_passed("item_index/manifest.json")
        for split in ("valid", "test"):
            self.command(retrieval_arguments(self.config, split), f"evaluate_{split}")
            self.require_passed(f"evaluation_{split}/metrics.json")
        self.command(beam_arguments(self.config), "beam")
        self.require_passed("evaluation_beam/metrics.json")
        self.command(beam_arguments(self.config, reference=True), "reference_beam")
        self.require_passed("evaluation_reference_beam/metrics.json")
        self.update("exporting")
        self.call(["cp", f"{self.container}:{self.run}/.", str(self.run)], timeout=1200)
        isolation = verify_isolation(self.config["isolation_snapshot"])
        atomic_json(self.run / "isolation_verification.json", isolation)
        if isolation["status"] != "passed":
            raise RuntimeError("Old worktree isolation check failed after experiment")
        self.update("reporting")
        if self.render_reports()["status"] != "passed":
            raise RuntimeError("Experiment report has incomplete results")
        self.update("completed", status="completed", finished_at=legacy.now())
        self.render_reports()
        self.call(["stop", "--timeout", "2", self.container], check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status", "report"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--isolation-snapshot", type=Path, default=ISOLATION_SNAPSHOT)
    args = parser.parse_args(argv)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        run = args.run_dir.resolve()
    elif args.action == "start":
        run = (WORKSPACE / f"weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_qwen_retrieval_{timestamp}/sft").resolve()
    else:
        active = read_json(ACTIVE_RUN, {})
        if not active.get("run_dir"):
            if args.action == "status":
                print(json.dumps({"status": "not_started"}))
                return 0
            raise ValueError("No active run pointer; provide --run-dir")
        run = Path(active["run_dir"]).resolve()
    run.relative_to(WORKSPACE.resolve())
    if args.action == "status":
        print(json.dumps(read_json(run / "run_status.json", {"status": "not_started", "run_dir": str(run)}), ensure_ascii=False, indent=2))
        return 0
    if args.action == "report":
        config = read_json(run / "supervisor_config.json", {})
        write_report(run, args.report_root or config["report_root"])
        return 0
    if args.action == "start":
        reports = (args.report_root or ROOT / f"output/reports/qwen_retrieval_{timestamp}").resolve()
        reports.relative_to(ROOT.resolve())
        for path in (run, reports):
            if path.exists() and any(path.iterdir()):
                raise ValueError("Existing experiment retained; choose a new destination: " + str(path))
        check_prepared()
        isolation = verify_isolation(args.isolation_snapshot)
        if isolation["status"] != "passed":
            raise ValueError("Old-worktree snapshot no longer matches: " + "; ".join(isolation["errors"]))
        docker = os.environ.get("DOCKER", "docker")
        image_id = subprocess.check_output([docker, "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
        if image_id != IMAGE_ID:
            raise ValueError("Experiment requires the historical fixed Docker image")
        sources = source_files()
        historical_training = read_json(HISTORICAL_MODEL.parent / "training_summary.json", {})
        selected_name = Path(historical_training.get("best_checkpoint") or "checkpoint-5187").name
        selected_checkpoint = HISTORICAL_MODEL.parent / selected_name
        measured_checkpoint_bytes = directory_size(selected_checkpoint) if selected_checkpoint.is_dir() else 0
        storage = storage_requirement((BASE_MODEL / "model.safetensors").stat().st_size,
                                      directory_size(HISTORICAL_MODEL), directory_size(INPUT_DIR) + directory_size(DATA_DIR),
                                      sum((ROOT / name).stat().st_size for name in sources), measured_checkpoint_bytes)
        storage["measured_checkpoint_path"] = str(selected_checkpoint)
        storage["available_bytes"] = shutil.disk_usage("/tmp").free
        if storage["available_bytes"] < storage["required_bytes"]:
            raise ValueError(f"Insufficient local Docker storage: need {storage['required_bytes']} bytes, available {storage['available_bytes']}; existing artifacts retained")
        gpu_preflight = check_idle_gpus()
        config = {"run_dir": str(run), "report_root": str(reports),
                  "container": args.container or f"minionerec-industrial05b-retrieval-{timestamp}",
                  "image": args.image, "image_id": image_id, "isolation_snapshot": str(args.isolation_snapshot.resolve()),
                  "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {name: hash_file(ROOT / name) for name in sources},
                  "category": CATEGORY, "base_model": str(BASE_MODEL), "historical_model": str(HISTORICAL_MODEL),
                  "expected_sid_sha256": APPROVED_SID_SHA256, "storage_preflight": storage,
                  "gpu_preflight": gpu_preflight,
                  "num_epochs": 3, "lr_scheduler_type": "linear", "learning_rate": 3e-4,
                  "warmup_steps": 20, "seed": 42, "global_batch_size": 64, "micro_batch_size": 8,
                  "gradient_accumulation_steps": 2, "world_size": 4, "projection_dim": 256,
                  "retrieval_weight": 0.1, "temperature": 0.07, "top_delta": 200,
                  "include_title_history": True, "selection_metric": "eval_loss",
                  "deepspeed_accumulation_corrected": True, "fresh_generation_only_control": False,
                  "rl_enabled": False, "created_at": legacy.now()}
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "supervisor_config.json", config)
        atomic_json(run / "isolation_verification_start.json", isolation)
        atomic_json(run / "run_status.json", {"status": "running", "stage": "starting", "started_at": legacy.now(),
                                              "run_dir": str(run), "report_root": str(reports), "container": config["container"]})
        atomic_json(ACTIVE_RUN, {"run_dir": str(run), "report_root": str(reports), "container": config["container"],
                                 "created_at": config["created_at"]})
        with (run / "supervisor.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--run-dir", str(run)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status = read_json(run / "run_status.json", {})
        status["pid"] = process.pid
        atomic_json(run / "run_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    config = read_json(run / "supervisor_config.json")
    if not config:
        raise ValueError("Cannot run worker without its saved supervisor_config.json")
    supervisor = Supervisor(config)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Retrieval supervisor received signal " + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervisor.execute()
        return 0
    except BaseException as error:
        traceback.print_exc()
        previous_stage = supervisor.status.get("stage")
        supervisor.update("failed", status="failed", failed_stage=previous_stage, error=str(error), finished_at=legacy.now())
        if supervisor.created_container:
            # Stop only the fresh container owned by this worker. Retain its
            # filesystem and every checkpoint/log for inspection and export.
            try:
                supervisor.call(["stop", "--timeout", "10", supervisor.container], check=False, timeout=30)
                if supervisor.child is not None:
                    supervisor.child.wait(timeout=30)
                    supervisor.child = None
            except Exception:
                traceback.print_exc()
        try:
            supervisor.sync_small()
            isolation = verify_isolation(config["isolation_snapshot"])
            atomic_json(run / "isolation_verification.json", isolation)
            supervisor.render_reports()
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
