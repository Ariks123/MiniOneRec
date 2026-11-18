#!/usr/bin/env python3
"""Sequential linear 3/5-epoch SFT, retaining the five-epoch run's epoch-three weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docker import industrial_experiment as experiment
from docker.industrial_title_history_experiment import update, worker

EXPERIMENT_ROOT = experiment.WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1"
REPORT_ROOT = ROOT / "output/reports/industrial_05b_linear_ep3_ep5_title_history_retry1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--image", default="minionerec:h20-original-torch28")
    args = parser.parse_args(argv)
    root, reports = args.experiment_root.resolve(), args.report_root.resolve()
    root.relative_to(experiment.WORKSPACE.resolve())
    if args.action == "status":
        print(json.dumps(experiment.read_json(root / "comparison_status.json", {"status": "not_started"}), ensure_ascii=False, indent=2))
        return 0
    if args.action == "worker":
        return worker(root)
    experiment.check_prepared()
    if shutil.disk_usage("/tmp").free < 50 * 1024**3:
        raise ValueError("The sequential runs require at least 50 GiB of free local Docker storage before starting")
    reports.relative_to(experiment.WORKSPACE.resolve())
    for path in (root, reports):
        if path.exists() and any(path.iterdir()):
            raise ValueError("Existing experiment retained; choose a new destination: " + str(path))
    docker = os.environ.get("DOCKER", "docker")
    image_id = subprocess.check_output([docker, "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    arms = [{"name": f"epochs{epochs}", "num_epochs": epochs, "include_title_history": True,
             "save_epoch_checkpoint": 3 if epochs == 5 else 0,
             "minimum_epochs": 3 if epochs == 5 else 0,
             "run_dir": str(root / f"epochs{epochs}/sft"), "report_root": str(reports / f"epochs{epochs}"),
             "container": f"minionerec-industrial05b-linear-{suffix}-ep{epochs}"}
            for epochs in (3, 5)]
    root.mkdir(parents=True)
    plan = {"created_at_utc": experiment.now(), "comparison_kind": "linear_epochs",
            "category": experiment.CATEGORY, "base_model": str(experiment.MODEL), "image_id": image_id,
            "lr_scheduler_type": "linear", "early_stopping_patience": 3, "eval_save_fraction": 0.05,
            "generative_validation": False, "final_ranking_eval": True,
            "minimum_free_disk_bytes": 28 * 1024**3,
            "source_sha256": {name: experiment.sha256(experiment.runtime_source(name)) for name in experiment.SOURCE_FILES},
            "arms": arms, "comparison_reports": str(reports / "comparison"),
            "interpretation": [
                "Both runs initialize independently from the same Base weights and use the same five tasks.",
                "Training checkpoint selection and early stopping use validation loss only; ranking is evaluated after training.",
                "The five-epoch arm cannot early-stop before epoch three, so the requested epoch-end snapshot is guaranteed.",
                "The epoch-three snapshot is saved before validation-best model reloading and is evaluated only after training.",
                "Linear decay follows each run's total epoch budget, so the two runs have different learning-rate trajectories."
            ]}
    experiment.atomic_json(root / "comparison_plan.json", plan)
    update(root, status="running", stage="starting", started_at_utc=experiment.now())
    with (root / "comparison_worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--experiment-root", str(root)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(update(root, worker_pid=process.pid), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
