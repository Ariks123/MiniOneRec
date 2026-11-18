#!/usr/bin/env python3
"""Durable, isolated Industrial SFT -> full evaluation -> report workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
CATEGORY = "Industrial_and_Scientific"
APPROVED_SID = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"
REFERENCE = WORKSPACE / "code/MiniOneRec-Reproduction"
REFERENCE_COMMIT = "dc60a84e7ceb790410b8f2f8553acdc07438b585"
INPUT = WORKSPACE / "MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey"
DATA = WORKSPACE / "MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey"
EXPERIMENT_ROOT = WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_cosine_ep10_20261117"
RUN = EXPERIMENT_ROOT / "baseline/sft"
REPORTS = ROOT / "output/reports/industrial_05b_cosine_ep10/baseline"
MODEL = WORKSPACE / "weights/Qwen2.5-0.5B"
SOURCE_FILES = ("sft.py", "data.py", "sft_monitor.py", "LogitProcessor.py", "npu_support.py",
                "scripts/evaluate_sft.py", "scripts/sft_eval_utils.py", "scripts/report_industrial_sft.py",
                "scripts/industrial_generative_validation.py",
                "config/h20_zero2.json")


def runtime_source(name):
    # data.py was recovered from the selected run's container and is vendored.
    return ROOT / name


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_prepared(model=MODEL):
    if Path(model).resolve() != MODEL.resolve():
        raise ValueError("This comparison requires the Qwen2.5-0.5B Base model")
    manifest = read_json(DATA / "preparation_manifest.json", {})
    if not (manifest.get("dataset") == CATEGORY and manifest.get("validation") == "passed"
            and manifest.get("original_sid") is True):
        raise ValueError("Original Industrial preparation must pass before launching")
    if sha256(INPUT / (CATEGORY + ".index.json")) != APPROVED_SID:
        raise ValueError("Original Industrial SID identity changed")
    if manifest.get("history_length", {}).get("maximum", 0) > 10:
        raise ValueError("Original Industrial experiment requires history length at most ten")
    for name in ("train.csv", "valid.csv", "test.csv", "info.txt"):
        if sha256(DATA / name) != manifest["outputs"][name]["sha256"]:
            raise ValueError("Prepared output changed: " + name)
    for name in SOURCE_FILES:
        if not runtime_source(name).is_file():
            raise ValueError("Missing experiment source: " + name)
    if not (Path(model) / "model.safetensors").is_file():
        raise ValueError("Requested Base model weights are unavailable: " + str(model))
    return manifest


def train_arguments(config):
    scheduler = config.get("lr_scheduler_type", "linear")
    if scheduler not in ("linear", "cosine"):
        raise ValueError("Industrial experiments support linear or cosine scheduling")
    if Path(config.get("base_model", MODEL)).resolve() != MODEL.resolve():
        raise ValueError("This comparison requires the Qwen2.5-0.5B Base model")
    epochs = config.get("num_epochs", 3)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10:
        raise ValueError("num_epochs must be an integer from 1 to 10")
    include_title_history = config.get("include_title_history", False)
    if not isinstance(include_title_history, bool):
        raise ValueError("include_title_history must be boolean")
    arguments = ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4",
            "sft.py", "--base_model", str(config.get("base_model", MODEL)), "--train_file", str(DATA / "train.csv"),
            "--eval_file", str(DATA / "valid.csv"), "--output_dir", config["run_dir"],
            "--category", CATEGORY, "--sid_index_path", str(INPUT / (CATEGORY + ".index.json")),
            "--item_meta_path", str(INPUT / (CATEGORY + ".item.json")), "--seed", "42",
            "--batch_size", "64", "--micro_batch_size", "8", "--num_epochs", str(epochs),
            "--learning_rate", "0.0003", "--lr_scheduler_type", scheduler,
            "--cutoff_len", "512", "--freeze_LLM", "False", "--train_from_scratch", "False",
            "--gradient_checkpointing", "True", "--attn_implementation", "sdpa",
            "--monitoring", "True", "--monitor_window", "100",
            "--baseline_eval", "True",
            "--include_title_history", str(include_title_history),
            "--generative_validation", str(config.get("generative_validation", False)),
            "--generation_batch_size", str(config.get("generation_batch_size", 4)),
            "--generation_num_beams", str(config.get("generation_num_beams", 50)),
            "--save_epoch_checkpoint", str(config.get("save_epoch_checkpoint", 0)),
            "--minimum_epochs", str(config.get("minimum_epochs", 0)),
            "--early_stopping_patience", str(config.get("early_stopping_patience", 3)),
            "--eval_save_fraction", str(config.get("eval_save_fraction", 0.05)),
            "--wandb_run_name", "Industrial_05b_" + scheduler + "_" + ("title_history" if include_title_history else "baseline")]
    arguments.extend(["--deepspeed", "/opt/minionerec/config/h20_zero2.json"])
    return arguments


def eval_arguments(config):
    return ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4",
            "scripts/evaluate_sft.py", "--category", CATEGORY, "--model-path", config["run_dir"] + "/final_checkpoint",
            "--data-dir", str(DATA), "--input-dir", str(INPUT), "--output-dir", config["run_dir"] + "/evaluation",
            "--expected-sid-sha256", APPROVED_SID, "--prompt-style", "original-eval",
            "--num-beams", "50", "--batch-size", "4"]


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.run = Path(config["run_dir"])
        self.reports = Path(config["report_root"])
        self.container = config["container"]
        self.docker = os.environ.get("DOCKER", "docker")
        self.status = read_json(self.run / "run_status.json", {})
        self.child = None
        self.last_sync = self.last_report = 0.0

    def call(self, args, check=True, timeout=300):
        result = subprocess.run([self.docker, *args], capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError("Docker " + " ".join(args[:3]) + ": " + result.stderr.decode(errors="replace")[-2500:])
        return result

    def update(self, stage=None, **fields):
        if stage:
            self.status["stage"] = stage
        self.status.update(fields, updated_at=now(), pid=os.getpid(), rl_started=False)
        progress = read_json(self.run / "monitor/summary.json")
        if progress:
            self.status["progress"] = progress
        atomic_json(self.run / "run_status.json", self.status)

    def prepare_container(self):
        model = Path(self.config.get("base_model", MODEL))
        existing = self.call(["container", "inspect", "--format", "{{.State.Running}} {{.Image}}", self.container], check=False)
        image_id = self.call(["image", "inspect", "--format", "{{.Id}}", self.config["image"]]).stdout.decode().strip()
        if existing.returncode == 0:
            running, actual_image = existing.stdout.decode().strip().split()
            if running == "true":
                raise ValueError("Industrial container is already running; it was left untouched")
            if actual_image != image_id:
                raise ValueError("Retained Industrial container belongs to another image")
        else:
            reason = existing.stderr.decode(errors="replace").lower()
            if "no such" not in reason and "not found" not in reason:
                raise RuntimeError("Cannot inspect Industrial container: " + reason)
            self.call(["create", "--name", self.container, "--shm-size", "16g", "--gpus", '"device=0,1,2,3"',
                       "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3", "--entrypoint", "/bin/sleep", self.config["image"], "infinity"])
        self.call(["start", self.container])
        self.call(["exec", self.container, "mkdir", "-p", str(INPUT), str(DATA), str(model),
                   str(self.run), str(self.reports), "/opt/minionerec/scripts", "/opt/minionerec/config"])
        for name in SOURCE_FILES:
            self.call(["cp", str(runtime_source(name)), self.container + ":/opt/minionerec/" + name])
        for directory in (INPUT, DATA):
            self.call(["cp", str(directory) + "/.", self.container + ":" + str(directory)])
        exists = self.call(["exec", self.container, "test", "-f", str(model / "model.safetensors")], check=False)
        if exists.returncode:
            self.call(["cp", str(model) + "/.", self.container + ":" + str(model)], timeout=900)
        for name in ("download_manifest.json", "model_identity_manifest.json", "official_repository_metadata.json"):
            manifest = model / name
            if manifest.is_file():
                target = self.run / ("base_model_" + name)
                self.call(["cp", str(manifest), f"{self.container}:{target}"])
                shutil.copyfile(manifest, target)
        result = self.call(["exec", self.container, "python", "-c",
                   "import torch, transformers, datasets, accelerate, json, sys; assert torch.cuda.device_count()==4; "
                   "assert all('H20' in torch.cuda.get_device_name(i) for i in range(4)); "
                   "print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,"
                   "datasets=datasets.__version__,accelerate=accelerate.__version__,cuda=torch.version.cuda,"
                   "python=sys.version.split()[0],devices=[torch.cuda.get_device_name(i) for i in range(4)])))"])
        self.status["runtime_versions"] = json.loads(result.stdout.decode().strip().splitlines()[-1])
        self.status["runtime_image_id"] = image_id
        self.status["source_sha256"] = {name: sha256(runtime_source(name)) for name in SOURCE_FILES}
        self.update()

    def sync_small(self):
        names = ["run_config.json", "baseline_validation.json", "training_summary.json", "trainer_state.json",
                 "monitor/metrics.jsonl", "monitor/summary.json", "monitor/events.jsonl", "evaluation/metrics.json",
                 "ranking_selection.json", "best_recall_checkpoint/selection_metrics.json",
                 "validation_ranking/latest_metrics.json", "evaluation_recall/metrics.json",
                 "evaluation_epoch3/metrics.json"]
        snapshot_epoch = self.config.get("save_epoch_checkpoint", 0)
        if snapshot_epoch:
            names.append(f"epoch_{snapshot_epoch}_checkpoint/snapshot_metadata.json")
        names += [f"evaluation/progress.rank{rank}.json" for rank in range(4)]
        names += [f"validation_ranking/progress.rank{rank}.json" for rank in range(4)]
        for name in names:
            destination = self.run / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".industrial-sync-", dir=destination.parent) as folder:
                temporary = Path(folder) / destination.name
                result = self.call(["cp", f"{self.container}:{self.run}/{name}", str(temporary)], check=False, timeout=60)
                if result.returncode == 0 and temporary.is_file():
                    os.replace(temporary, destination)
        self.update()

    def sample_gpu(self):
        result = self.call(["exec", self.container, "nvidia-smi",
                           "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                           "--format=csv,noheader,nounits"], check=False, timeout=15)
        if result.returncode == 0:
            with (self.run / "gpu_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": now(), "csv": result.stdout.decode().strip()}) + "\n")

    def render_reports(self):
        for name in ("run_status.json", "supervisor_config.json"):
            source = self.run / name
            if source.is_file():
                self.call(["cp", str(source), f"{self.container}:{source}"])
        result = self.call(["exec", "-w", "/opt/minionerec", self.container, "python", "scripts/report_industrial_sft.py",
                            "--run-dir", str(self.run), "--data-dir", str(DATA), "--input-dir", str(INPUT),
                            "--report-root", str(self.reports)], timeout=180)
        if result.stdout:
            print(result.stdout.decode(errors="replace")[-1500:], flush=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.call(["cp", f"{self.container}:{self.reports}/.", str(self.reports)], timeout=180)

    def command(self, arguments, stage):
        self.update(stage)
        command = [self.docker, "exec", "-w", "/opt/minionerec", "-e", "PYTHONUNBUFFERED=1",
                   "-e", "WANDB_MODE=disabled", "-e", "TOKENIZERS_PARALLELISM=false",
                   "-e", "NCCL_IB_DISABLE=1", "-e", "OMP_NUM_THREADS=8", self.container, *arguments]
        # OSS append handles cannot reliably be read concurrently. Keep the
        # active stream on local disk and publish complete atomic snapshots.
        spool = Path(tempfile.mkdtemp(prefix="industrial-log-")) / (stage + ".log")
        self.status[stage + "_log_spool"] = str(spool)
        self.update()
        def publish_log():
            destination = self.run / (stage + ".log")
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(spool, temporary)
            os.replace(temporary, destination)
        with spool.open("a", encoding="utf-8") as log:
            self.child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while self.child.poll() is None:
                current = time.monotonic()
                if current - self.last_sync >= 60:
                    try:
                        self.sync_small()
                        self.sample_gpu()
                        publish_log()
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                        print("Live sync pending: " + str(error), flush=True)
                    self.last_sync = current
                if current - self.last_report >= 120:
                    try:
                        self.render_reports()
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                        print("Live report pending: " + str(error), flush=True)
                    self.last_report = current
                time.sleep(5)
            status = self.child.returncode
        publish_log()
        self.child = None
        self.sync_small()
        if status:
            raise RuntimeError(f"Industrial {stage} exited {status}; see {self.run / (stage + '.log')}")

    def execute(self):
        self.update("preparing", status="running")
        check_prepared(Path(self.config.get("base_model", MODEL)))
        self.prepare_container()
        self.render_reports()
        self.command(train_arguments(self.config), "training")
        if read_json(self.run / "training_summary.json", {}).get("status") != "completed":
            raise RuntimeError("Training returned without a completed summary")
        final_ranking = self.config.get("final_ranking_eval", True)
        if final_ranking:
            self.command(eval_arguments(self.config), "validating")
            if read_json(self.run / "evaluation/metrics.json", {}).get("status") != "passed":
                raise RuntimeError("Full Industrial evaluation did not pass")
        else:
            training = read_json(self.run / "training_summary.json", {})
            placeholder = {"status": "not_requested", "scope": "no_ranking_requested",
                           "reason": "Ranking metrics disabled by user; validation loss only",
                           "training_performed": True, "model_path": str(self.run / "final_checkpoint"),
                           "validation": {"loss": training.get("final_eval_metrics", {}).get("eval_loss")}}
            path = self.run / "evaluation/metrics.json"
            atomic_json(path, placeholder)
            self.call(["exec", self.container, "mkdir", "-p", str(path.parent)])
            self.call(["cp", str(path), f"{self.container}:{path}"])
        if final_ranking and self.config.get("generative_validation", False):
            # The independent Recall-selected model is evaluated only after
            # training; all checkpoint selection uses valid.csv exclusively.
            arguments = eval_arguments(self.config)
            arguments[arguments.index("--model-path") + 1] = str(self.run / "best_recall_checkpoint")
            arguments[arguments.index("--output-dir") + 1] = str(self.run / "evaluation_recall")
            self.command(arguments, "validating_recall")
            if read_json(self.run / "evaluation_recall/metrics.json", {}).get("status") != "passed":
                raise RuntimeError("Full Recall-selected model evaluation did not pass")
        snapshot_epoch = self.config.get("save_epoch_checkpoint", 0)
        if final_ranking and snapshot_epoch:
            snapshot = self.run / f"epoch_{snapshot_epoch}_checkpoint"
            metadata = read_json(snapshot / "snapshot_metadata.json", {})
            if metadata.get("epoch") != snapshot_epoch or metadata.get("best_model_reloaded") is not False:
                raise RuntimeError("The requested epoch-end training-weight snapshot was not retained")
            arguments = eval_arguments(self.config)
            arguments[arguments.index("--model-path") + 1] = str(snapshot)
            arguments[arguments.index("--output-dir") + 1] = str(self.run / f"evaluation_epoch{snapshot_epoch}")
            self.command(arguments, "validating_epoch_snapshot")
            if read_json(self.run / f"evaluation_epoch{snapshot_epoch}/metrics.json", {}).get("status") != "passed":
                raise RuntimeError("Full epoch-end snapshot evaluation did not pass")
        self.update("exporting")
        self.call(["cp", f"{self.container}:{self.run}/.", str(self.run)], timeout=1200)
        # Host supervisor status remains authoritative across a container export.
        self.update("reporting")
        self.render_reports()
        validation = read_json(self.reports / "validation_report.json", {})
        expected_status = "passed" if final_ranking else "not_requested"
        if validation.get("status") != expected_status:
            raise RuntimeError("Industrial final report did not validate the complete evaluation")
        if not all((self.reports / name).is_file() for name in ("training-report.pdf", "validation-report.pdf")):
            raise RuntimeError("Industrial final PDF reports are missing")
        self.update("completed", status="completed", finished_at=now())
        self.render_reports()
        self.call(["stop", "--timeout", "2", self.container], check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--report-root", type=Path, default=REPORTS)
    parser.add_argument("--container", default="minionerec-h20-industrial-05b-cosine-baseline")
    parser.add_argument("--image", default="minionerec:h20-original-torch28")
    parser.add_argument("--base-model", type=Path, default=MODEL)
    parser.add_argument("--lr-scheduler-type", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--include-title-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generative-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-ranking-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-epoch-checkpoint", type=int, default=0)
    parser.add_argument("--minimum-epochs", type=float, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-num-beams", type=int, choices=(50,), default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--eval-save-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    run.relative_to(WORKSPACE.resolve())
    if args.action == "status":
        print(json.dumps(read_json(run / "run_status.json", {"status": "not_started"}), ensure_ascii=False, indent=2))
        return 0
    if args.action == "start":
        model = args.base_model.resolve()
        model.relative_to(WORKSPACE.resolve())
        check_prepared(model)
        if not 1 <= args.num_epochs <= 10 or args.early_stopping_patience < 1:
            raise ValueError("Maximum epochs must be 1–10 and early stopping patience must be positive")
        if not 0 < args.eval_save_fraction < 1:
            raise ValueError("eval_save_fraction must be between zero and one")
        if not 0 <= args.save_epoch_checkpoint <= args.num_epochs or not 0 <= args.minimum_epochs <= args.num_epochs:
            raise ValueError("Snapshot epoch and minimum epochs must be within the training epoch budget")
        if run.exists() and any(run.iterdir()):
            raise ValueError("Existing Industrial run retained; choose a new --run-dir")
        reports = args.report_root.resolve()
        reports.relative_to(WORKSPACE.resolve())
        config = {"run_dir": str(run), "report_root": str(reports), "container": args.container,
                  "image": args.image, "category": CATEGORY, "base_model": str(model),
                  "expected_sid_sha256": APPROVED_SID, "world_size": 4, "num_epochs": args.num_epochs,
                  "learning_rate": 3e-4, "global_batch_size": 64, "micro_batch_size": 8,
                  "gradient_accumulation_steps": 2, "lr_scheduler_type": args.lr_scheduler_type, "history_max": 10,
                  "gradient_checkpointing": True,
                  "baseline_eval": True, "save_total_limit": 2,
                  "include_title_history": args.include_title_history,
                  "generative_validation": args.generative_validation,
                  "final_ranking_eval": args.final_ranking_eval,
                  "save_epoch_checkpoint": args.save_epoch_checkpoint,
                  "minimum_epochs": args.minimum_epochs,
                  "generation_batch_size": args.generation_batch_size,
                  "generation_num_beams": args.generation_num_beams,
                  "selection_metric": "ndcg10" if args.generative_validation else "eval_loss",
                  "early_stopping_patience": args.early_stopping_patience,
                  "eval_save_fraction": args.eval_save_fraction,
                  "deepspeed": "/opt/minionerec/config/h20_zero2.json",
                  "reference_repository": "https://github.com/Enddonkey/MiniOneRec-Reproduction",
                  "reference_commit": REFERENCE_COMMIT,
                  "reference_sft_sha256": sha256(REFERENCE / "sft.py"),
                  "reference_readme_sha256": sha256(REFERENCE / "README.md"),
                  "runtime_data_source": str(runtime_source("data.py")),
                  "adaptation_notes": [
                      "Base model is explicitly recorded and loaded fresh; previous fine-tuned weights are not reused.",
                      f"This run starts fresh from Qwen2.5-0.5B Base, with an epoch cap of {args.num_epochs}, global batch 64, four GPUs with micro batch 8 and accumulation 2.",
                      f"The {args.lr_scheduler_type} scheduler spans the requested epoch budget with 20 warmup steps; the recorded selection_metric governs validation model selection and early stopping.",
                      "Original data tasks and original evaluation prompt are retained; DDP placement, monitoring, provenance, complete coverage and constrained-decoding boundaries adapt the workflow to four GPUs.",
                      "DeepSpeed ZeRO-2 and gradient checkpointing retain the previous experiment's numerical training setup.",
                      "The augmented arm adds one title-history-to-next-SID sample for each original training interaction; equal epochs therefore do not imply equal optimizer steps."
                  ], "rl_enabled": False}
        run.mkdir(parents=True)
        atomic_json(run / "supervisor_config.json", config)
        atomic_json(run / "run_status.json", {"status": "running", "stage": "preparing", "started_at": now(),
                                              "updated_at": now(), "parameters": config, "rl_started": False})
        with (run / "supervisor.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--run-dir", str(run)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        status = read_json(run / "run_status.json")
        status["pid"] = process.pid
        atomic_json(run / "run_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    supervisor = Supervisor(read_json(run / "supervisor_config.json"))
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Industrial supervisor received signal " + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervisor.execute()
        return 0
    except BaseException as error:
        traceback.print_exc()
        supervisor.update("failed", status="failed", error=str(error), finished_at=now())
        try:
            supervisor.sync_small()
            supervisor.render_reports()
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
