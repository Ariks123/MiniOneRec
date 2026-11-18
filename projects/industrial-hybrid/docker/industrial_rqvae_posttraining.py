#!/usr/bin/env python3
"""Run the selected Industrial SFT and GRPO recipes with trained RQ-VAE SIDs."""

import argparse
from ast import literal_eval
import csv
import json
from itertools import zip_longest
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docker import industrial_experiment as sft
from docker import industrial_rl_experiment as grpo


RUNS = sft.WORKSPACE / "weights/checkpoints/MiniOneRec"
DEFAULT_RUN = RUNS / "Industrial_and_Scientific_RQVAE_2K_20261118"
BASELINES = {
    "sft": grpo.PARENT_RUN,
    "rl": grpo.RUN,
}
CATEGORY = sft.CATEGORY
INPUT_FILES = ("train.csv", "valid.csv", "test.csv", "info.txt")
ROW_COUNTS = {"train": 36259, "valid": 4532, "test": 4533}
WORKDIR = "/opt/minionerec"
GRPO_ENTRY = "scripts/industrial_rqvae_grpo.py"
DATA_FIELDS = ("history_item_sid", "item_sid")
GUARDS = {
    "scripts/sft_eval_utils.py": (
        'require(manifest.get("original_sid") is True, "Industrial evaluation requires the original SID catalog")',
        'require(manifest.get("original_sid") is False, "Industrial RQ-VAE evaluation requires the new SID catalog")',
    ),
    "scripts/industrial_generative_validation.py": (
        'and manifest.get("original_sid") is True, "Ranking selection requires passed original Industrial preparation")',
        'and manifest.get("original_sid") is False, "Ranking selection requires passed RQ-VAE preparation")',
    ),
}
RUNTIME_FILES = tuple(name for name in dict.fromkeys((*sft.SOURCE_FILES, *grpo.SOURCE_FILES, GRPO_ENTRY))
                      if not name.startswith("scripts/report_"))
SYNC_FILES = (
    "run_config.json", "training_summary.json", "trainer_state.json",
    "baseline_validation.json", "baseline_validation_ranking.json",
    "monitor/metrics.jsonl", "monitor/events.jsonl", "monitor/summary.json",
    "validation_ranking/latest_metrics.json", "evaluation/metrics.json",
    *(f"evaluation/progress.rank{rank}.json" for rank in range(4)),
    *(f"validation_ranking/progress.rank{rank}.json" for rank in range(4)),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_catalog(directory):
    data, catalog = directory / "data", directory / "catalog"
    sid_path = catalog / f"{CATEGORY}.index.json"
    meta_path = catalog / f"{CATEGORY}.item.json"
    manifest = read(data / "preparation_manifest.json")
    base = read(sft.DATA / "preparation_manifest.json")
    grpo.require(manifest["original_sid"] is False and manifest["dataset"] == CATEGORY
                 and manifest["validation"] == "passed" and manifest["check_only"] is False,
                 "A completed new Industrial SID preparation manifest is required")
    grpo.require(manifest["rows"] == ROW_COUNTS and manifest["preserve_all_interaction_rows"] is True
                 and manifest["sampling_applied"] is False and manifest["keep_longest_only"] is False,
                 "Prepared data changed the archived row coverage")
    grpo.require(manifest["history_length"] == base["history_length"],
                 "Prepared history lengths changed")
    sid_hash = sft.sha256(sid_path)
    grpo.require(sid_hash != sft.APPROVED_SID and
                 manifest["expected_sid_sha256"] == manifest["sources"]["sid_index"]["sha256"] == sid_hash,
                 "New SID must match the manifest and differ from the original")
    grpo.require(sft.sha256(meta_path) == base["sources"]["item_metadata"]["sha256"]
                 == manifest["sources"]["item_metadata"]["sha256"], "Original item metadata differ")
    sids = read(sid_path)
    grpo.require(len(sids) == base["items"] == manifest["items"] and
                 set(sids) == set(read(meta_path)) and
                 all(isinstance(tokens, list) and len(tokens) == 3 and
                     all(isinstance(token, str) and re.fullmatch(rf"<{stage}_[0-9]+>", token)
                         for stage, token in zip("abc", tokens)) for tokens in sids.values()),
                 "The RQ-VAE export must cover the original catalog with three a/b/c tokens per item")
    for source in manifest["sources"].values():
        filename = source["filename"]
        grpo.require(Path(filename).name == filename and sft.sha256(catalog / filename) == source["sha256"],
                     "RQ-VAE manifest source changed: " + filename)
    output_hashes = {name: sft.sha256(data / name) for name in INPUT_FILES}
    grpo.require(all(output_hashes[name] == manifest["outputs"][name]["sha256"] for name in INPUT_FILES),
                 "RQ-VAE prepared file hash differs from the manifest")
    joined = {item: "".join(tokens) for item, tokens in sids.items()}
    for split, count in ROW_COUNTS.items():
        name = f"{split}.csv"
        grpo.require(sft.sha256(sft.DATA / name) == base["outputs"][name]["sha256"],
                     "Archived comparison data changed: " + name)
        with (data / name).open(encoding="utf-8", newline="") as f, (
                sft.DATA / name).open(encoding="utf-8", newline="") as b:
            new, old = csv.DictReader(f), csv.DictReader(b)
            grpo.require(new.fieldnames == old.fieldnames, "Prepared CSV headers changed: " + name)
            kept_fields = [field for field in old.fieldnames if field not in DATA_FIELDS]
            observed = 0
            for observed, (a, p) in enumerate(zip_longest(new, old), 1):
                grpo.require(a is not None and p is not None and all(a[k] == p[k] for k in kept_fields),
                             f"Interaction data other than SID changed: {name}:{observed}")
                previous = literal_eval(a["history_item_id"])
                expected_history = [joined[str(item)] for item in previous]
                grpo.require(a["item_sid"] == joined[a["item_id"]]
                             and literal_eval(a["history_item_sid"]) == expected_history,
                             f"History/target SID differs from RQ-VAE catalog: {name}:{observed}")
            grpo.require(observed == count, "Prepared CSV size changed: " + name)
    return {"sid_sha256": sid_hash, "output_sha256": output_hashes, "manifest_sha256": sft.sha256(
            data / "preparation_manifest.json"), "rows": ROW_COUNTS}


def commands(run, sid_hash):
    data, catalog = run / "data", run / "catalog"
    sft_run, rl_run = run / "epochs3/sft", run / "rl"
    outcome = {}
    for stage, directory in (("sft", sft_run), ("rl", rl_run)):
        archived = read(BASELINES[stage] / "supervisor_config.json")
        archived_arguments = (sft.train_arguments(archived) if stage == "sft" else grpo.train_arguments(archived))
        training = list(archived_arguments)
        overrides = {
            "--train_file": data / "train.csv", "--eval_file": data / "valid.csv",
            "--sid_index_path": catalog / f"{CATEGORY}.index.json",
            "--item_meta_path": catalog / f"{CATEGORY}.item.json", "--output_dir": directory,
        }
        if stage == "rl":
            training[training.index("rl.py")] = GRPO_ENTRY
            overrides.update({"--model_path": sft_run / "final_checkpoint", "--info_file": data / "info.txt"})
        for flag, value in overrides.items():
            training[training.index(flag) + 1] = str(value)
        archived_evaluation = sft.eval_arguments(archived)
        evaluation = list(archived_evaluation)
        for flag, value in {
            "--model-path": directory / "final_checkpoint", "--data-dir": data,
            "--input-dir": catalog, "--output-dir": directory / "evaluation",
            "--expected-sid-sha256": sid_hash,
        }.items():
            evaluation[evaluation.index(flag) + 1] = str(value)
        outcome[stage] = {
            "directory": str(directory), "training": training, "evaluation": evaluation,
            "source_run": str(BASELINES[stage]), "archived_training": archived_arguments,
            "archived_evaluation": archived_evaluation,
        }
    return outcome


def check_tokenizer(run):
    """Verify the same added-token IDs as SFT on the fixed image without GPUs."""
    image = read(BASELINES["sft"] / "run_status.json")["runtime_image_id"]
    docker = os.environ.get("DOCKER", "docker")
    code = '''
import json
import sys
from transformers import AutoTokenizer

index = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained("/tmp/qwen-tokenizer", local_files_only=True)
original_size = len(tokenizer)
tokens = sorted({token for parts in index.values() for token in parts})
tokenizer.add_tokens(tokens)
vocabulary = tokenizer.get_vocab()
assert all(tokenizer.encode(token, add_special_tokens=False) == [vocabulary[token]] for token in tokens)
assert all(tokenizer.encode("".join(parts), add_special_tokens=False) == [vocabulary[token] for token in parts]
           for parts in index.values())
print(json.dumps({"base_vocab_size": original_size, "sid_tokens": len(tokens),
                  "tokenizer_size": len(tokenizer), "products": len(index),
                  "unique_sid": len({tuple(parts) for parts in index.values()}),
                  "three_atomic_tokens_per_product": True}))
'''
    with tempfile.TemporaryDirectory(prefix="industrial-rqvae-tokenizer-") as temp:
        root = Path(temp)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
            shutil.copyfile(sft.MODEL / name, root / name)
        process = subprocess.run([docker, "run", "--rm", "--network", "none", "-i",
                                  "-v", f"{root}:/tmp/qwen-tokenizer:ro", "--entrypoint", "python",
                                  image, "-c", code],
                                 input=(run / "catalog" / f"{CATEGORY}.index.json").read_bytes(),
                                 capture_output=True, check=True, timeout=180)
    return json.loads(process.stdout)


class PostTraining:
    def __init__(self, options):
        self.run = options.run_dir
        self.container = options.container
        self.gate = options.wait_for
        self.docker = os.environ.get("DOCKER", "docker")
        self.owns_container = False
        self.status_file = self.run / "posttraining_status.json"
        self.status = {"status": "running", "stage": "waiting", "container": self.container,
                       "started_at": sft.now(), "pid": os.getpid(),
                       "gpu_gate": str(self.gate) if self.gate else None}

    def update(self, stage, **values):
        self.status.update(stage=stage, updated_at=sft.now(), **values)
        sft.atomic_json(self.status_file, self.status)

    def call(self, *args, timeout=300, optional=False):
        result = subprocess.run([self.docker, *map(str, args)], timeout=timeout, capture_output=True)
        if result.returncode and not optional:
            raise RuntimeError("Docker " + " ".join(map(str, args[:3])) + ": " +
                               result.stderr.decode(errors="replace").strip()[-2500:])
        return result

    def initialize(self, staging):
        source_ids = {label: read(path / "run_status.json") for label, path in BASELINES.items()}
        image = source_ids["rl"]["runtime_image_id"]
        grpo.require(image == source_ids["sft"]["runtime_image_id"], "Baseline runtime images differ")
        self.call("image", "inspect", image)
        changes = {}
        original_hashes = {name: digest for record in source_ids.values()
                           for name, digest in record["source_sha256"].items()}
        for name in RUNTIME_FILES:
            source = ROOT / name
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sft.sha256(source)
            if name != GRPO_ENTRY:
                grpo.require(original_hashes[name] == digest, "Archived runtime source drift: " + name)
            if name in GUARDS:
                old, new = GUARDS[name]
                content = source.read_text(encoding="utf-8")
                grpo.require(content.count(old) == 1, "Cannot verify exact corpus guard change: " + name)
                target.write_text(content.replace(old, new), encoding="utf-8")
            else:
                shutil.copyfile(source, target)
            changes[name] = {"archived_sha256": original_hashes.get(name), "runtime_sha256": sft.sha256(target)}
        self.call("create", "--name", self.container, "--shm-size", "16g",
                  "--gpus", '"device=0,1,2,3"', "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3",
                  "--entrypoint", "/bin/sleep", image, "infinity")
        self.owns_container = True
        self.call("start", self.container)
        self.call("exec", self.container, "mkdir", "-p", WORKDIR, sft.MODEL,
                  *(self.run / name for name in ("data", "catalog", "epochs3/sft", "rl")))
        self.call("cp", f"{staging}/.", f"{self.container}:{WORKDIR}")
        self.call("cp", f"{sft.MODEL}/.", f"{self.container}:{sft.MODEL}", timeout=900)
        for item in ("data", "catalog"):
            self.call("cp", f"{self.run / item}/.", f"{self.container}:{self.run / item}")
        self.status.update(runtime_image_id=image, source_sha256=changes)

    def validate_gpu(self):
        idle = self.call("exec", self.container, "nvidia-smi",
                         "--query-gpu=index,name,memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits").stdout.decode()
        grpo.validate_gpu_idle(idle)
        running = self.call("exec", self.container, "nvidia-smi",
                            "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits").stdout.decode().strip()
        grpo.require(not running, "Four H20s must be free before launching post-training")

    def pull_file(self, directory, name):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".sync")
        result = self.call("cp", f"{self.container}:{target}", temporary, optional=True, timeout=60)
        if not result.returncode:
            os.replace(temporary, target)

    def sync(self, directory, name, spool):
        for name_to_copy in SYNC_FILES:
            self.pull_file(directory, name_to_copy)
        summary = sft.read_json(directory / "monitor/summary.json", {})
        destination = directory / (name + ".log")
        temporary = destination.with_name(destination.name + ".sync")
        shutil.copyfile(spool, temporary)
        os.replace(temporary, destination)
        self.update(name, progress=summary)

    def execute_command(self, directory, name, args):
        self.update(name, progress={})
        directory.mkdir(parents=True, exist_ok=True)
        env = ("PYTHONUNBUFFERED=1", "WANDB_MODE=disabled", "HF_HUB_OFFLINE=1",
               "TRANSFORMERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "NCCL_IB_DISABLE=1", "OMP_NUM_THREADS=8")
        command = [self.docker, "exec", "-w", WORKDIR]
        for item in env:
            command.extend(["-e", item])
        with tempfile.TemporaryDirectory(prefix="industrial-rqvae-train-") as local:
            file = Path(local) / "output.log"
            with file.open("ab") as stream:
                child = subprocess.Popen([*command, self.container, *args], stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        try:
                            self.sync(directory, name, file)
                        except (OSError, subprocess.TimeoutExpired) as error:
                            print(f"Live progress sync will retry: {error}", flush=True)
            self.sync(directory, name, file)
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; inspect {directory / (name + '.log')}")

    def export(self, directory, stage):
        for name in ("final_checkpoint", "evaluation"):
            target = directory / name
            target.mkdir(parents=True, exist_ok=True)
            self.call("cp", f"{self.container}:{target}/.", target, timeout=900)
        if stage == "rl":
            self.pull_file(directory, "rl_task_manifest.jsonl")
        evaluation = read(directory / "evaluation/metrics.json")
        grpo.require(evaluation["status"] == "passed" and evaluation["completed_rows"] == ROW_COUNTS["test"],
                     "Final beam evaluation does not cover the original test split")
        training = read(directory / "training_summary.json")
        return {"checkpoint": training["best_checkpoint"], "selected_metric": training["best_metric"],
                "runtime": training["train_metrics"]["train_runtime"],
                "model": str(directory / "final_checkpoint"), "evaluation": str(directory / "evaluation")}

    def execute(self):
        self.update("waiting")
        while self.gate and not self.gate.exists():
            time.sleep(15)
        identity = verify_catalog(self.run)
        recipe = commands(self.run, identity["sid_sha256"])
        self.update("preparing", dataset=identity, recipe=recipe)
        with tempfile.TemporaryDirectory(prefix="industrial-rqvae-runtime-") as staging:
            self.initialize(Path(staging))
        self.validate_gpu()
        results = {}
        for stage, record in recipe.items():
            directory = Path(record["directory"])
            self.execute_command(directory, stage + "_training", record["training"])
            training = read(directory / "training_summary.json")
            if training["status"] != "completed" or not training["final_checkpoint_contains_best_model"]:
                raise RuntimeError(f"{stage} did not export its validation-selected checkpoint")
            self.execute_command(directory, stage + "_evaluation", record["evaluation"])
            self.update(stage + "_export")
            results[stage] = self.export(directory, stage)
            self.update(stage + "_completed", results=results)
        self.call("stop", "--timeout", 15, self.container)
        self.update("completed", status="completed", progress={}, finished_at=sft.now(), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "start", "worker", "status"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--container", default="minionerec-industrial-rqvae-posttraining-20261118")
    parser.add_argument("--wait-for", type=Path, help="Wait for this scheduler marker before any GPU work")
    parser.add_argument("--check-tokenizer", action="store_true", help="Verify tokenization using the CPU runtime")
    options = parser.parse_args()
    options.run_dir = options.run_dir.resolve()
    options.run_dir.relative_to(RUNS)
    grpo.require(options.run_dir != RUNS and all(options.run_dir != path and options.run_dir not in path.parents
                                                for path in BASELINES.values()),
                 "Choose a separate RQ-VAE experiment output directory")
    status = options.run_dir / "posttraining_status.json"
    if options.action == "status":
        print(json.dumps(sft.read_json(status, {"status": "not_started"}), indent=2))
        return
    if options.action == "preflight":
        identity = verify_catalog(options.run_dir)
        record = {"dataset": identity, "recipe": commands(options.run_dir, identity["sid_sha256"])}
        if options.check_tokenizer:
            record["tokenizer"] = check_tokenizer(options.run_dir)
            sft.atomic_json(options.run_dir / "posttraining_preflight.json", record)
        print(json.dumps(record, indent=2))
        return
    if options.action == "start":
        options.run_dir.mkdir(parents=True, exist_ok=True)
        with (options.run_dir / "posttraining_supervisor.log").open("xb") as stream:
            args = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--run-dir",
                    str(options.run_dir), "--container", options.container]
            if options.wait_for:
                args.extend(["--wait-for", str(options.wait_for.resolve())])
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"pid": child.pid, "status_file": str(status), "container": options.container}))
        return
    if status.exists():
        raise FileExistsError(f"Existing run status retained at {status}")
    worker = PostTraining(options)
    try:
        worker.execute()
    except BaseException as error:
        traceback.print_exc()
        if worker.owns_container:
            try:
                worker.call("stop", "--timeout", 20, worker.container, timeout=45, optional=True)
            except (OSError, subprocess.TimeoutExpired):
                traceback.print_exc()
        worker.update(worker.status["stage"], status="failed", error=str(error), finished_at=sft.now())
        raise


if __name__ == "__main__":
    main()
