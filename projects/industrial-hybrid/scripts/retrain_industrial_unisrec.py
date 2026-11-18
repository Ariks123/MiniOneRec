"""Run the archived 20-epoch UniSRec recipe with the new Industrial SID catalog."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


DATASET = "Industrial_and_Scientific"
SOURCE = Path("/opt/unisrec")
INPUT = Path("/opt/industrial-input")
ARCHIVE = Path("/opt/industrial-run")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(stage, *command):
    status(stage)
    with (ARCHIVE / f"{stage}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def status(stage, **fields):
    values = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    save(ARCHIVE / "status.json", values)
    print(json.dumps(values), flush=True)


def prepare(config, baseline, prepared):
    import yaml

    original = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    if (original["epochs"], original["nproc"], original["seed"]) != (20, 4, 42):
        raise ValueError("Unexpected original UniSRec budget or seed")
    current = dict(original, checkpoint_dir=str(ARCHIVE / "checkpoints"))
    config.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    run("prepare-data", sys.executable, "prepare_qwen_data.py", "--source-dir", str(INPUT / "catalog"),
        "--output-dir", str(prepared), "--embedding-file", str(INPUT / "embedding" / f"{DATASET}.emb-qwen-td.npy"),
        "--max-history-len", str(current["MAX_ITEM_LIST_LENGTH"]), "--overwrite")
    archived = load(INPUT / "reference_preparation_manifest.json")
    observed = load(prepared / "manifest.json")
    sid_name = f"{DATASET}.index.json"
    for artifact, expected in archived["source_sha256"].items():
        if artifact != sid_name and observed["source_sha256"].get(artifact) != expected:
            raise ValueError(f"Data preparation changed a fixed input: {artifact}")
    if observed["source_sha256"][sid_name] == archived["source_sha256"][sid_name]:
        raise ValueError("UniSRec preparation must use the newly generated SID catalog")
    for name, expected in archived["prepared_sha256"].items():
        if name != "catalog.json" and observed["prepared_sha256"].get(name) != expected:
            raise ValueError(f"UniSRec consumed data drifted after regeneration: {name}")
    save(ARCHIVE / "preparation_manifest.json", observed)


def train():
    config = ARCHIVE / "config.yaml"
    prepared = SOURCE / "prepared" / DATASET
    prepare(config, INPUT / "reference_config.yaml", prepared)
    run("preflight", "bash", "scripts/run_4gpu.sh", "preflight", "--config", str(config),
        "--report", str(ARCHIVE / "gpu_preflight.json"))
    run("train", "bash", "scripts/run_4gpu.sh", "train", "--config", str(config))
    (result_file,) = (ARCHIVE / "checkpoints").glob("*.result.json")
    selected = load(result_file)
    checkpoint = Path(selected["checkpoint"])
    if checkpoint.parent != ARCHIVE / "checkpoints" or not checkpoint.is_file():
        raise ValueError(f"Training did not select a new local checkpoint: {checkpoint}")
    checksum = sha(checkpoint)
    run("vector-validation", sys.executable, "qwen_vector_eval.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--split", "valid", "--device", "cuda:0",
        "--output", str(ARCHIVE / "validation"))
    run("vector-test", sys.executable, "qwen_vector_eval.py", "--config", str(config),
        "--checkpoint", str(checkpoint), "--split", "test", "--device", "cuda:0",
        "--output", str(ARCHIVE / "test"))
    run("export", sys.executable, str(INPUT / "drafter_export.py"), "--source", str(SOURCE),
        "--config", str(config), "--checkpoint", str(checkpoint), "--output", str(ARCHIVE / "export"),
        "--expected-checkpoint-sha256", checksum)
    export = load(ARCHIVE / "export/metadata.json")
    if export["checkpoint_sha256"] != checksum:
        raise ValueError("Export does not match the checkpoint selected on validation")
    status("completed", checkpoint=str(checkpoint.relative_to(ARCHIVE)), checkpoint_sha256=checksum,
           best_epoch=export["checkpoint_epoch_zero_based"] + 1, validation_nll=selected["best_valid_score"],
           sid_sha256=sha(INPUT / "catalog" / f"{DATASET}.index.json"))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if (ARCHIVE / "status.json").exists():
        raise ValueError("UniSRec run directory already contains a run status")
    try:
        train()
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
