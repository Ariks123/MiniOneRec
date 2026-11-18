#!/usr/bin/env python3
"""Durable Industrial GRPO training from the verified linear-three-epoch SFT model."""
from __future__ import annotations

import argparse
import json
import math
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docker import industrial_experiment as sft

WORKSPACE, CATEGORY = sft.WORKSPACE, sft.CATEGORY
DATA, INPUT, APPROVED_SID = sft.DATA, sft.INPUT, sft.APPROVED_SID
BASE_MODEL = WORKSPACE / "weights/Qwen2.5-0.5B"
PARENT_RUN = (WORKSPACE / "weights/checkpoints/MiniOneRec/"
              "Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1/epochs3/sft")
PARENT_MODEL = PARENT_RUN / "final_checkpoint"
PARENT_BASE_IDENTITY = "ff170b0115ab6b9c2d872ad7e3a234fbd1efbefa2a5a8aa8dc7131cee40ee79a"
PARENT_WEIGHT_SHA256 = "e688140bbfe9aac2981324471dd79ff9da1ad0b329acfa9c18c63f4dca4f597c"
RUN = WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl"
REPORTS = ROOT / "output/reports/industrial_05b_rl_from_linear_ep3_retry1"
SOURCE_FILES = ("rl.py", "minionerec_trainer.py", "scripts/industrial_rl_data.py",
                "scripts/industrial_rl_validation.py", "scripts/deepspeed_zero2_compat.py",
                "sft_monitor.py", "data.py", "LogitProcessor.py",
                "npu_support.py", "scripts/industrial_generative_validation.py", "scripts/evaluate_sft.py",
                "scripts/sft_eval_utils.py", "scripts/report_industrial_sft.py", "scripts/report_industrial_rl.py",
                "config/h20_zero2.json")
CONTROL_FILES = ("docker/industrial_rl_experiment.py", "docker/industrial_experiment.py")
PARENT_REPORT_FILES = {"run_config": ("run_config.json", "parent_sft_run_config.json"),
                       "training_summary": ("training_summary.json", "parent_sft_training_summary.json"),
                       "evaluation_metrics": ("evaluation/metrics.json", "parent_sft_evaluation_metrics.json"),
                       "predictions": ("evaluation/predictions.jsonl", "parent_sft_predictions.jsonl")}
now, read_json, atomic_json, sha256 = sft.now, sft.read_json, sft.atomic_json, sft.sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_beta(value):
    beta = float(value)
    require(math.isfinite(beta) and beta > 0, "RL beta must be positive and finite")
    return beta


def check_storage(minimum_gib=50):
    free = shutil.disk_usage("/tmp").free / 1024 ** 3
    require(free >= minimum_gib,
            f"Local checkpoint storage has {free:.2f} GiB free; this RL workflow requires {minimum_gib} GiB")
    return free


def source_fingerprints():
    missing = [name for name in (*SOURCE_FILES, *CONTROL_FILES) if not (ROOT / name).is_file()]
    require(not missing, "Missing RL workflow source: " + ", ".join(missing))
    return {name: sha256(ROOT / name) for name in (*SOURCE_FILES, *CONTROL_FILES)}


def check_source_fingerprints(expected):
    require(expected and source_fingerprints() == expected,
            "RL workflow sources changed after the reviewed launch configuration was recorded")


def check_parent_cache(run, sources):
    for key, (_, destination) in PARENT_REPORT_FILES.items():
        path = Path(run) / destination
        require(path.is_file() and sha256(path) == sources.get("sha256", {}).get(key),
                "Cached parent SFT report source differs: " + destination)


def validate_parent(model_path=PARENT_MODEL):
    model = Path(model_path).resolve()
    require(model == PARENT_MODEL.resolve(), "RL must start from the selected linear-three-epoch SFT export")
    parent = model.parent
    config = read_json(parent / "run_config.json", {})
    summary = read_json(parent / "training_summary.json", {})
    status = read_json(parent / "run_status.json", {})
    evaluation = read_json(parent / "evaluation/metrics.json", {})
    hyper = config.get("hyperparameters", {})
    require(config.get("stage") == "sft" and config.get("data", {}).get("category") == CATEGORY,
            "Parent run is not the original Industrial SFT experiment")
    require(Path(config.get("model", {}).get("source", "/missing")).resolve() == BASE_MODEL.resolve()
            and config.get("model", {}).get("identity", {}).get("sha256") == PARENT_BASE_IDENTITY,
            "Parent run does not originate from the verified Qwen2.5-0.5B Base model")
    require(hyper.get("max_epochs") == 3 and hyper.get("lr_scheduler_type") == "linear"
            and config.get("data", {}).get("include_title_history") is True,
            "Parent run must be the linear-three-epoch title-history arm")
    selected = parent / "checkpoint-5187"
    require(summary.get("status") == "completed" and summary.get("selection_metric") == "eval_loss"
            and summary.get("final_checkpoint_contains_best_model") is True
            and summary.get("global_step") == 5442 and summary.get("epoch") == 3.0
            and Path(summary.get("best_checkpoint", "/missing")).resolve() == selected.resolve()
            and Path(summary.get("final_checkpoint", "/missing")).resolve() == model,
            "Parent SFT selection/export does not identify checkpoint-5187")
    required_files = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                      "added_tokens.json", "special_tokens_map.json", "posttraining_provenance.json")
    require(all((model / name).is_file() and (model / name).stat().st_size > 0 for name in required_files),
            "Selected parent model/config/tokenizer/provenance export is incomplete")
    model_hash = sha256(model / "model.safetensors")
    require(model_hash == PARENT_WEIGHT_SHA256, "Parent SFT weights differ from the approved weight SHA-256")
    require((selected / "model.safetensors").is_file()
            and model_hash == sha256(selected / "model.safetensors"),
            "Parent final weights differ from the selected checkpoint-5187 weights")
    provenance = read_json(model / "posttraining_provenance.json", {})
    require(provenance.get("sid_index_sha256") == APPROVED_SID,
            "Parent model SID provenance differs from the original Industrial SID catalog")
    require(evaluation.get("status") == "passed" and evaluation.get("completed_rows") == 4533
            and evaluation.get("expected_rows") == 4533 and evaluation.get("num_beams") == 50
            and evaluation.get("prompt_style") == "original-eval"
            and evaluation.get("validity_rate") == evaluation.get("unique_rate") == 1
            and Path(evaluation.get("model_path", "/missing")).resolve() == model,
            "Parent full-test baseline has not passed the matching beam50 protocol")
    report_hashes = {}
    for key, (source, _) in PARENT_REPORT_FILES.items():
        require((parent / source).is_file(), "Parent report source is missing: " + source)
        report_hashes[key] = sha256(parent / source)
    image_id = status.get("runtime_image_id")
    require(isinstance(image_id, str) and image_id.startswith("sha256:"),
            "Parent SFT runtime has no immutable image identity")
    return {"run_dir": str(parent), "model_path": str(model), "model_sha256": model_hash,
            "runtime_image_id": image_id, "report_sources": {"run_dir": str(parent), "sha256": report_hashes}}


def check_prepared(model_path=PARENT_MODEL):
    manifest = read_json(DATA / "preparation_manifest.json", {})
    require(manifest.get("dataset") == CATEGORY and manifest.get("validation") == "passed"
            and manifest.get("original_sid") is True, "Original Industrial data preparation must pass")
    require(sha256(INPUT / (CATEGORY + ".index.json")) == APPROVED_SID,
            "Original Industrial SID identity changed")
    require(manifest.get("history_length", {}).get("maximum", 0) <= 10,
            "Industrial histories must contain at most ten items")
    for name in ("train.csv", "valid.csv", "test.csv", "info.txt"):
        require(sha256(DATA / name) == manifest["outputs"][name]["sha256"],
                "Prepared original Industrial file changed: " + name)
    parent = validate_parent(model_path)
    source_fingerprints()
    return parent


def validate_gpu_idle(csv_text):
    rows = [line.strip().split(",") for line in csv_text.splitlines() if line.strip()]
    require(len(rows) == 4, "RL preflight requires exactly four visible H20 GPUs")
    for row in rows:
        require(len(row) == 4 and "H20" in row[1], "RL preflight found an unexpected accelerator")
        try:
            memory, utilization = float(row[2]), float(row[3])
        except ValueError as error:
            raise ValueError("RL preflight could not read GPU memory/utilization") from error
        require(memory < 2048 and utilization <= 5,
                "A requested GPU is busy; existing workloads were left untouched")


def train_arguments(config):
    require(config.get("num_epochs", 2) == 2, "This RL experiment has a fixed two-epoch maximum")
    require(Path(config.get("model_path", PARENT_MODEL)).resolve() == PARENT_MODEL.resolve(),
            "RL parent model differs from the selected SFT export")
    beta = positive_beta(config.get("beta", 0.001))
    arguments = ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                 "rl.py", "--model_path", str(config.get("model_path", PARENT_MODEL)),
                 "--train_file", str(DATA / "train.csv"), "--eval_file", str(DATA / "valid.csv"),
                 "--info_file", str(DATA / "info.txt"), "--sid_index_path", str(INPUT / (CATEGORY + ".index.json")),
                 "--item_meta_path", str(INPUT / (CATEGORY + ".item.json")), "--category", CATEGORY,
                 "--output_dir", config["run_dir"], "--deepspeed", "/opt/minionerec/config/h20_zero2.json"]
    options = {"num_train_epochs": 2, "learning_rate": 1e-5, "train_batch_size": 16, "eval_batch_size": 16,
               "gradient_accumulation_steps": 16, "num_generations": 16, "reward_type": "ranking",
               "beam_search": True, "sync_ref_model": True, "ref_model_sync_steps": 512,
               "ref_model_mixup_alpha": 0.6, "temperature": 1.0, "beta": beta,
               "add_gt": False, "dynamic_sampling": False, "dapo": False, "gspo": False,
               "sample_train": False, "test_during_training": False, "gradient_checkpointing": True,
               "attn_implementation": "sdpa", "seed": 42, "generation_batch_size": 4,
               "generation_num_beams": 50, "cutoff_len": 512, "monitor_window": 100,
               "eval_save_fraction": 0.125, "early_stopping_patience": 3}
    for key, value in options.items():
        arguments.extend(["--" + key, str(value)])
    return arguments


def eval_arguments(config):
    return sft.eval_arguments(config)


class Supervisor(sft.Supervisor):
    def update(self, stage=None, **fields):
        if stage:
            self.status["stage"] = stage
        self.status.update(fields, updated_at=now(), pid=os.getpid(), rl_enabled=True)
        self.status["rl_started"] = bool(self.status.get("rl_started") or stage == "training")
        progress = read_json(self.run / "monitor/summary.json")
        if progress:
            self.status["progress"] = progress
        atomic_json(self.run / "run_status.json", self.status)

    def prepare_container(self):
        check_source_fingerprints(self.config["source_sha256"])
        check_parent_cache(self.run, self.config["parent_sft_report_sources"])
        check_storage()
        model = Path(self.config["model_path"])
        existing = self.call(["container", "inspect", self.container], check=False)
        require(existing.returncode != 0, "Existing RL container retained; use a new --container")
        reason = existing.stderr.decode(errors="replace").lower()
        require("no such" in reason or "not found" in reason, "Cannot inspect RL container: " + reason)
        image_id = self.call(["image", "inspect", "--format", "{{.Id}}", self.config["image"]]).stdout.decode().strip()
        require(image_id == self.config["parent_runtime_image_id"], "RL runtime image differs from the parent SFT image")
        self.call(["create", "--name", self.container, "--shm-size", "16g", "--gpus", '"device=0,1,2,3"',
                   "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3", "--entrypoint", "/bin/sleep", image_id, "infinity"])
        self.call(["start", self.container])
        idle = self.call(["exec", self.container, "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"]).stdout.decode()
        validate_gpu_idle(idle)
        processes = self.call(["exec", self.container, "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"]).stdout.decode().strip()
        require(not processes, "A requested GPU has an active compute process; existing workloads were left untouched")
        self.call(["exec", self.container, "mkdir", "-p", str(INPUT), str(DATA), str(model), str(self.run),
                   str(self.reports), "/opt/minionerec/scripts", "/opt/minionerec/config"])
        for name in SOURCE_FILES:
            self.call(["cp", str(ROOT / name), self.container + ":/opt/minionerec/" + name])
        for directory in (INPUT, DATA):
            self.call(["cp", str(directory) + "/.", self.container + ":" + str(directory)])
        self.call(["cp", str(model) + "/.", self.container + ":" + str(model)], timeout=900)
        # The RL entry checks this export's pinned SHA-256. Its original run
        # metadata remains available without copying the duplicate SFT optimizer
        # or selected weight files into the new container.
        for cached, original in (("parent_sft_run_config.json", "run_config.json"),
                                 ("parent_sft_training_summary.json", "training_summary.json")):
            self.call(["cp", str(self.run / cached), f"{self.container}:{model.parent / original}"])
        for _, destination in PARENT_REPORT_FILES.values():
            self.call(["cp", str(self.run / destination), f"{self.container}:{self.run / destination}"])
        self.call(["cp", str(self.run / "supervisor_config.json"), f"{self.container}:{self.run / 'supervisor_config.json'}"])
        result = self.call(["exec", "-w", "/opt/minionerec", self.container, "python", "-c",
                    "import torch,transformers,datasets,accelerate,trl,bitsandbytes,deepspeed,json,sys; "
                    "import rl,minionerec_trainer; assert torch.cuda.device_count()==4; "
                    "assert transformers.__version__=='4.57.1' and trl.__version__=='0.24.0'; "
                    "print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,"
                    "datasets=datasets.__version__,accelerate=accelerate.__version__,trl=trl.__version__,"
                    "deepspeed=deepspeed.__version__,bitsandbytes=bitsandbytes.__version__,cuda=torch.version.cuda,"
                    "python=sys.version.split()[0],devices=[torch.cuda.get_device_name(i) for i in range(4)])))"])
        self.status["runtime_versions"] = json.loads(result.stdout.decode().strip().splitlines()[-1])
        self.status["runtime_image_id"] = image_id
        self.status["source_sha256"] = self.config["source_sha256"]
        self.update()

    def sync_small(self):
        names = ["run_config.json", "baseline_validation.json", "baseline_validation_ranking.json",
                 "training_summary.json", "trainer_state.json", "ranking_selection.json",
                 "monitor/metrics.jsonl", "monitor/summary.json", "monitor/events.jsonl",
                 "validation_ranking/latest_metrics.json", "evaluation/metrics.json"]
        names += [f"evaluation/progress.rank{rank}.json" for rank in range(4)]
        names += [f"validation_ranking/progress.rank{rank}.json" for rank in range(4)]
        for name in names:
            destination = self.run / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".industrial-rl-sync-", dir=destination.parent) as folder:
                temporary = Path(folder) / destination.name
                result = self.call(["cp", f"{self.container}:{self.run}/{name}", str(temporary)], check=False, timeout=60)
                if result.returncode == 0 and temporary.is_file():
                    os.replace(temporary, destination)
        self.update()

    def render_reports(self):
        for name in ("run_status.json", "supervisor_config.json"):
            source = self.run / name
            if source.is_file():
                self.call(["cp", str(source), f"{self.container}:{source}"])
        result = self.call(["exec", "-w", "/opt/minionerec", self.container, "python", "scripts/report_industrial_rl.py",
                            "--run-dir", str(self.run), "--data-dir", str(DATA), "--input-dir", str(INPUT),
                            "--report-root", str(self.reports)], timeout=180)
        if result.stdout:
            print(result.stdout.decode(errors="replace")[-1500:], flush=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.call(["cp", f"{self.container}:{self.reports}/.", str(self.reports)], timeout=180)

    def validate_final_reports(self):
        require(read_json(self.reports / "training_report.json", {}).get("status") == "completed",
                "Industrial RL training report is incomplete or lacks valid training telemetry")
        require(read_json(self.reports / "validation_report.json", {}).get("status") == "passed",
                "Industrial RL final report did not validate the complete evaluation")
        require(read_json(self.reports / "comparison.json", {}).get("status") == "passed",
                "Industrial RL comparison did not validate the parent baseline and recommendation")
        require(all((self.reports / name).is_file() for name in ("training-report.pdf", "validation-report.pdf")),
                "Industrial RL final PDF reports are missing")

    def command(self, arguments, stage):
        self.update(stage)
        command = [self.docker, "exec", "-w", "/opt/minionerec", "-e", "PYTHONUNBUFFERED=1",
                   "-e", "WANDB_MODE=disabled", "-e", "TOKENIZERS_PARALLELISM=false", "-e", "HF_HUB_OFFLINE=1",
                   "-e", "TRANSFORMERS_OFFLINE=1", "-e", "NCCL_IB_DISABLE=1", "-e", "OMP_NUM_THREADS=8",
                   self.container, *arguments]
        spool = Path(tempfile.mkdtemp(prefix="industrial-rl-log-")) / (stage + ".log")
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
                        print("Live RL sync pending: " + str(error), flush=True)
                    self.last_sync = current
                if current - self.last_report >= 60:
                    try:
                        self.render_reports()
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                        print("Live RL report pending: " + str(error), flush=True)
                    self.last_report = current
                time.sleep(5)
            result_code = self.child.returncode
        publish_log()
        self.child = None
        self.sync_small()
        if result_code:
            raise RuntimeError(f"Industrial RL {stage} exited {result_code}; see {self.run / (stage + '.log')}")

    def execute(self):
        self.update("preparing", status="running")
        parent = check_prepared(Path(self.config["model_path"]))
        require(parent["model_sha256"] == self.config["parent_model_sha256"],
                "Parent SFT weights changed after launch")
        require(parent["report_sources"] == self.config["parent_sft_report_sources"],
                "Parent SFT baseline sources changed after launch")
        self.prepare_container()
        self.render_reports()
        self.command(train_arguments(self.config), "training")
        summary = read_json(self.run / "training_summary.json", {})
        require(summary.get("status") == "completed"
                and summary.get("selection_metric") in ("eval_ndcg10", "ndcg10")
                and summary.get("final_checkpoint_contains_best_model") is True,
                "RL training returned without a verified best-NDCG model export")
        self.command(eval_arguments(self.config), "validating")
        evaluation = read_json(self.run / "evaluation/metrics.json", {})
        require(evaluation.get("status") == "passed" and evaluation.get("completed_rows") == 4533,
                "Full Industrial RL test evaluation did not pass")
        self.update("exporting")
        self.call(["cp", f"{self.container}:{self.run}/.", str(self.run)], timeout=1200)
        self.update("reporting")
        self.render_reports()
        self.validate_final_reports()
        self.update("completed", status="completed", finished_at=now())
        self.render_reports()
        self.call(["stop", "--timeout", "2", self.container], check=False)

    def fail(self, error):
        """Leave an auditable stopped container when interruption aborts training."""
        stage = self.status.get("stage", "training")
        spool = self.status.get(stage + "_log_spool")
        if spool and Path(spool).is_file():
            shutil.copyfile(spool, self.run / (stage + ".log"))
        if self.child is not None and self.child.poll() is None:
            # Terminating docker-exec alone does not terminate its remote process.
            # Stop only this run's container; its filesystem/checkpoints remain.
            self.call(["stop", "--timeout", "20", self.container], check=False)
            self.child.wait(timeout=30)
            self.child = None
        self.update("failed", status="failed", error=str(error), finished_at=now())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--report-root", type=Path, default=REPORTS)
    parser.add_argument("--model-path", type=Path, default=PARENT_MODEL)
    parser.add_argument("--container", default="minionerec-industrial05b-rl-linear-ep3-20261117-retry1")
    parser.add_argument("--image", default="minionerec:h20-original-torch28")
    parser.add_argument("--beta", type=positive_beta, default=0.001)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    run.relative_to(WORKSPACE.resolve())
    if args.action == "status":
        print(json.dumps(read_json(run / "run_status.json", {"status": "not_started"}), ensure_ascii=False, indent=2))
        return 0
    if args.action == "start":
        require(not run.exists() or not any(run.iterdir()), "Existing Industrial RL run retained; choose a new --run-dir")
        reports = args.report_root.resolve()
        reports.relative_to(WORKSPACE.resolve())
        require(not reports.exists() or not any(reports.iterdir()), "Existing RL report directory retained; choose a new --report-root")
        check_storage()
        parent = check_prepared(args.model_path)
        docker = os.environ.get("DOCKER", "docker")
        image = subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", args.image],
                               check=True, capture_output=True, text=True).stdout.strip()
        require(image == parent["runtime_image_id"], "Requested image differs from the parent's immutable runtime")
        config = {"run_dir": str(run), "report_root": str(reports), "container": args.container,
                  "image": image, "requested_image": args.image, "model_path": parent["model_path"],
                  "parent_run_dir": parent["run_dir"], "parent_model_sha256": parent["model_sha256"],
                  "parent_selected_weight_verification": "approved model SHA-256 and host equality with checkpoint-5187",
                  "parent_runtime_image_id": parent["runtime_image_id"],
                  "parent_sft_report_sources": parent["report_sources"], "source_sha256": source_fingerprints(),
                  "category": CATEGORY, "expected_sid_sha256": APPROVED_SID, "world_size": 4, "num_epochs": 2,
                  "learning_rate": 1e-5, "lr_scheduler_type": "cosine", "num_generations": 16,
                  "per_device_completion_batch": 16, "gradient_accumulation_steps": 16,
                  "prompt_groups_per_update": 64, "completions_per_update": 1024,
                  "reward_type": "ranking", "beta": args.beta, "sync_ref_model": True,
                  "ref_model_sync_steps": 512, "ref_model_mixup_alpha": 0.6,
                  "selection_metric": "eval_ndcg10", "early_stopping_patience": 3,
                  "eval_save_fraction": 0.125, "save_total_limit": 2, "task_samples": 52775,
                  "deepspeed": "/opt/minionerec/config/h20_zero2.json", "rl_enabled": True}
        run.mkdir(parents=True, exist_ok=True)
        for _, (source, destination) in PARENT_REPORT_FILES.items():
            shutil.copyfile(Path(parent["run_dir"]) / source, run / destination)
        check_parent_cache(run, parent["report_sources"])
        atomic_json(run / "supervisor_config.json", config)
        atomic_json(run / "run_status.json", {"status": "running", "stage": "preparing", "started_at": now(),
                                              "updated_at": now(), "parameters": config,
                                              "rl_enabled": True, "rl_started": False})
        with (run / "supervisor.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--run-dir", str(run)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        status = read_json(run / "run_status.json")
        status["worker_pid"] = process.pid
        atomic_json(run / "run_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    config = read_json(run / "supervisor_config.json")
    require(isinstance(config, dict), "RL worker has no launch configuration")
    supervisor = Supervisor(config)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Industrial RL supervisor received signal " + str(signum))

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervisor.execute()
        return 0
    except BaseException as error:
        traceback.print_exc()
        supervisor.fail(error)
        try:
            supervisor.sync_small()
            supervisor.render_reports()
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
