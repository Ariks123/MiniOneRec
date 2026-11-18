#!/usr/bin/env python3
"""Durable host supervisor for SFT, full evaluation, and offline reports. Never runs RL."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("ZIWEIZHAO_WORKDIR", str(ROOT.parent)))
DEFAULT_RUN = WORKSPACE / "weights/checkpoints/MiniOneRec/Video_Games_sid2900/sft"
DEFAULT_REPORTS = ROOT / "output/reports/Video_Games"
REPORT_DIRECTORIES = ("06_sft_training", "07_sft_validation")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "status", "worker"))
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--image", default=os.environ.get("SFT_IMAGE", "minionerec:h20-sft-monitor"))
    p.add_argument("--container", default=os.environ.get("SFT_CONTAINER", "minionerec-h20-sft"))
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--sft-batch-size", type=int, default=1024)
    p.add_argument("--sft-micro-batch-size", type=int, default=16)
    p.add_argument("--resume-from-checkpoint", default="")
    p.add_argument("--monitor-window", type=int, default=100)
    return p


def configuration(args):
    for name in ("num_epochs", "sft_batch_size", "sft_micro_batch_size", "monitor_window"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.learning_rate <= 0 or args.sft_batch_size % (4 * args.sft_micro_batch_size):
        raise ValueError("Invalid learning rate or global/micro batch combination")
    run = args.run_dir.resolve()
    run.relative_to(WORKSPACE.resolve())
    args.report_root.resolve().relative_to(WORKSPACE.resolve())
    return {
        "run_dir": str(run), "report_root": str(args.report_root.resolve()),
        "image": args.image, "container": args.container,
        "num_epochs": args.num_epochs, "learning_rate": args.learning_rate,
        "sft_batch_size": args.sft_batch_size, "sft_micro_batch_size": args.sft_micro_batch_size,
        "gradient_accumulation_steps": args.sft_batch_size // (4 * args.sft_micro_batch_size),
        "monitor_window": args.monitor_window, "resume_from_checkpoint": args.resume_from_checkpoint,
        "model": "Qwen/Qwen2.5-1.5B", "rl_enabled": False,
    }


def training_arguments(config):
    result = ["--output-dir", config["run_dir"], "--num-epochs", str(config["num_epochs"]),
              "--learning-rate", str(config["learning_rate"]),
              "--sft-batch-size", str(config["sft_batch_size"]),
              "--sft-micro-batch-size", str(config["sft_micro_batch_size"]),
              "--monitor-window", str(config["monitor_window"])]
    if config.get("resume_from_checkpoint"):
        result += ["--resume-from-checkpoint", config["resume_from_checkpoint"]]
    return result


def start(args):
    config = configuration(args)
    run = Path(config["run_dir"])
    previous = read_json(run / "run_status.json", {})
    if previous.get("status") == "running" and alive(previous.get("pid")):
        raise ValueError(f"SFT supervisor is already running with PID {previous['pid']}")
    if previous and not args.resume_from_checkpoint:
        raise ValueError("Existing SFT run retained. Use --resume-from-checkpoint or choose a new --run-dir")
    if not previous and run.exists() and any(run.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty run directory")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume checkpoint must contain trainer_state.json")
        old_config = read_json(run / "supervisor_config.json", {})
        for key in ("model", "num_epochs", "learning_rate", "sft_batch_size", "sft_micro_batch_size"):
            if old_config.get(key) != config[key]:
                raise ValueError("Resume configuration differs: " + key)
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "supervisor_config.json", config)
    # No credentials are persisted: only allowlisted experiment configuration.
    status = {"status": "running", "stage": "preparing", "started_at": now(),
              "updated_at": now(), "parameters": config, "rl_started": False}
    atomic_json(run / "run_status.json", status)
    with (run / "supervisor.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "worker", "--run-dir", str(run)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
    status["pid"] = process.pid
    atomic_json(run / "run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.run = Path(config["run_dir"])
        self.reports = Path(config["report_root"])
        self.container = config["container"]
        self.docker = os.environ.get("DOCKER", "docker")
        self.environment = os.environ.copy()
        self.environment.update(MINIONEREC_IMAGE=config["image"], MINIONEREC_CONTAINER=self.container,
                                H20_DOCKER_MODE="copy", PYTHONUNBUFFERED="1")
        self.status = read_json(self.run / "run_status.json", {})
        self.status["pid"] = os.getpid()
        self.last_gpu = self.last_sync = self.last_report = 0.0
        self.child = None

    def update(self, stage=None, **values):
        if stage:
            self.status["stage"] = stage
        self.status.update(values, updated_at=now(), rl_started=False)
        summary = read_json(self.run / "monitor/summary.json")
        if summary:
            self.status["progress"] = summary
        atomic_json(self.run / "run_status.json", self.status)

    def call(self, arguments, check=True, timeout=180):
        completed = subprocess.run([self.docker, *arguments], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout, env=self.environment)
        if check and completed.returncode:
            raise RuntimeError(f"Docker {' '.join(arguments[:3])} failed: "
                               + completed.stderr.decode(errors="replace")[-2000:])
        return completed

    def is_running(self):
        result = self.call(["container", "inspect", "--format", "{{.State.Running}}", self.container], check=False)
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def sample_gpu(self):
        columns = ("index", "name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw")
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(columns),
                                 "--format=csv,noheader,nounits"], capture_output=True, timeout=10)
        if result.returncode:
            return
        gpus = []
        for row in result.stdout.decode().splitlines():
            fields = [value.strip() for value in row.split(",")]
            if len(fields) != 7:
                continue
            entry = {"index": int(fields[0]), "name": fields[1]}
            for key, raw in zip(("memory_used_mib", "memory_total_mib", "utilization_gpu_pct", "temperature_c", "power_w"), fields[2:]):
                try:
                    entry[key] = float(raw)
                except ValueError:
                    entry[key] = None
            gpus.append(entry)
        # This authoritative host file is never overwritten by a container export.
        with (self.run / "host_gpu_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": now(), "gpus": gpus}) + "\n")
        self.restore_host_metrics()

    def restore_host_metrics(self):
        source = self.run / "host_gpu_metrics.jsonl"
        if source.is_file():
            temporary = self.run / "gpu_metrics.jsonl.tmp"
            shutil.copyfile(source, temporary)
            os.replace(temporary, self.run / "gpu_metrics.jsonl")

    def sync_small(self):
        # Snapshot into temporary files before replacing local copies read by status/report clients.
        for name in ("run_config.json", "baseline_validation.json", "training_summary.json", "trainer_state.json",
                     "monitor/metrics.jsonl", "monitor/summary.json", "monitor/events.jsonl", "evaluation/metrics.json"):
            destination = self.run / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".sft-sync-", dir=destination.parent) as folder:
                temporary = Path(folder) / destination.name
                result = self.call(["cp", f"{self.container}:{self.run}/{name}", str(temporary)], check=False, timeout=60)
                if result.returncode == 0 and temporary.is_file():
                    os.replace(temporary, destination)

    def render_reports(self, seed=False):
        running = self.is_running()
        if not running:
            self.call(["start", self.container])
        try:
            self.call(["exec", self.container, "mkdir", "-p", str(self.run), str(self.reports)])
            for path in (self.run / "run_status.json", self.run / "gpu_metrics.jsonl", self.run / "supervisor_config.json"):
                if path.is_file():
                    self.call(["cp", str(path), f"{self.container}:{path}"], timeout=60)
            if seed:
                for name in ("index.html", "reports_manifest.json"):
                    source = self.reports / name
                    if source.is_file():
                        self.call(["cp", str(source), f"{self.container}:{source}"], timeout=60)
            result = self.call(["exec", self.container, "python", "scripts/report_sft.py", "--run-dir", str(self.run),
                                "--report-root", str(self.reports)], timeout=180)
            if result.stdout:
                print(result.stdout.decode(errors="replace")[-1500:], flush=True)
            self.reports.mkdir(parents=True, exist_ok=True)
            for name in (*REPORT_DIRECTORIES, "index.html", "reports_manifest.json"):
                source = f"{self.container}:{self.reports}/{name}"
                if name in REPORT_DIRECTORIES:
                    target = self.reports / name
                    target.mkdir(parents=True, exist_ok=True)
                    source += "/."
                else:
                    target = self.reports / name
                self.call(["cp", source, str(target)], timeout=180)
        finally:
            if not running:
                self.call(["stop", "--timeout", "2", self.container], check=False)

    def tick(self):
        current = time.monotonic()
        if current - self.last_gpu >= 30:
            try:
                self.sample_gpu()
            except (OSError, subprocess.TimeoutExpired) as error:
                print("GPU sampling unavailable: " + str(error), flush=True)
            self.last_gpu = current
        if current - self.last_sync >= 60:
            try:
                self.sync_small()
                self.update()
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                print("Live metrics sync pending: " + str(error), flush=True)
            self.last_sync = current
        if current - self.last_report >= 60 and self.is_running():
            try:
                self.render_reports()
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                print("Live report refresh pending: " + str(error), flush=True)
            self.last_report = current

    def command(self, action, arguments, stage):
        self.update(stage)
        log_path = self.run / (stage + ".log")
        with log_path.open("a", encoding="utf-8") as stream:
            self.child = subprocess.Popen(["bash", str(ROOT / "docker/run_h20.sh"), action, *arguments],
                                          stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT,
                                          env=self.environment, stdin=subprocess.DEVNULL)
            while self.child.poll() is None:
                if stage in {"training", "validating"}:
                    self.tick()
                else:
                    self.update()
                time.sleep(5)
            code = self.child.returncode
        self.child = None
        self.restore_host_metrics()
        self.update()
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {log_path}")

    def run_pipeline(self):
        self.update("preparing", status="running")
        self.command("prepare-data", [], "preparing")
        self.command("preflight", ["sft", *training_arguments(self.config)], "preflight")
        self.render_reports(seed=True)
        self.command("sft", training_arguments(self.config), "training")
        self.sync_small()
        training = read_json(self.run / "training_summary.json", {})
        if training.get("status") != "completed":
            raise RuntimeError("SFT returned without a completed monitored training_summary.json")
        self.update("validating")
        self.render_reports()
        self.command("evaluate-sft", ["--model-path", str(self.run / "final_checkpoint"),
                                     "--data-dir", str(ROOT / "data/Amazon18/posttraining/Video_Games_sid2900"),
                                     "--input-dir", str(ROOT / "data/Amazon18/Video_Games"),
                                     "--output-dir", str(self.run / "evaluation"), "--num-beams", "50"], "validating")
        evaluation = read_json(self.run / "evaluation/metrics.json", {})
        if evaluation.get("status") != "passed":
            raise RuntimeError("Final evaluation did not pass its completeness and model checks")
        self.update("reporting")
        self.render_reports()
        for directory in REPORT_DIRECTORIES:
            if not any((self.reports / directory).glob("*.pdf")):
                raise RuntimeError("Final report PDF is missing: " + directory)
        training_report = read_json(self.reports / REPORT_DIRECTORIES[0] / "training_report.json", {})
        validation_report = read_json(self.reports / REPORT_DIRECTORIES[1] / "validation_report.json", {})
        if training_report.get("status") != "completed" or validation_report.get("status") != "passed":
            raise RuntimeError("Reports did not validate training evidence and complete evaluation results")
        self.update("completed", status="completed", finished_at=now())
        try:
            self.render_reports()
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            # The previous complete PDFs and JSON reports already passed acceptance.
            print("Final status refresh pending; accepted reports are retained: " + str(error), flush=True)


def worker(args):
    config = read_json(args.run_dir / "supervisor_config.json")
    if not config:
        raise ValueError("Missing supervisor_config.json")
    supervisor = Supervisor(config)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Supervisor received signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervisor.run_pipeline()
        return 0
    except BaseException as error:
        traceback.print_exc()
        if supervisor.child is not None and supervisor.child.poll() is None:
            supervisor.child.terminate()
            try:
                supervisor.child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("Container and checkpoints retained; child export is still pending", flush=True)
        supervisor.update("failed", status="failed", error=str(error), finished_at=now())
        try:
            supervisor.sync_small()
            supervisor.render_reports()
        except Exception:
            traceback.print_exc()
        return 1


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "start":
        return start(args)
    if args.action == "worker":
        return worker(args)
    state = read_json(args.run_dir / "run_status.json", {"status": "not_started", "rl_started": False})
    report_root = Path(state.get("parameters", {}).get("report_root", str(DEFAULT_REPORTS)))
    report = read_json(report_root / REPORT_DIRECTORIES[0] / "training_report.json", {})
    if report.get("timing"):
        state["training_timing"] = report["timing"]
    state["paths"] = {"training_report": str(report_root / REPORT_DIRECTORIES[0] / "index.html"),
                      "validation_report": str(report_root / REPORT_DIRECTORIES[1] / "index.html"),
                      "monitor_summary": str(args.run_dir / "monitor/summary.json")}
    if state.get("status") == "running" and not alive(state.get("pid")):
        state["supervisor_alive"] = False
        state["message"] = "Supervisor is unavailable; inspect the retained container before resuming"
    else:
        state["supervisor_alive"] = alive(state.get("pid"))
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
