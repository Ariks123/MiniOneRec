#!/usr/bin/env python3
"""Run both fresh 0.5B cosine SFT arms sequentially and compare full-test recall."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docker import industrial_experiment as experiment


def update(root, **fields):
    path = root / "comparison_status.json"
    status = experiment.read_json(path, {})
    status.update(fields, updated_at_utc=experiment.now())
    experiment.atomic_json(path, status)
    return status


def worker(root):
    plan = experiment.read_json(root / "comparison_plan.json")
    try:
        for arm in plan["arms"]:
            minimum_free = plan.get("minimum_free_disk_bytes", 0)
            if minimum_free and shutil.disk_usage("/tmp").free < minimum_free:
                raise RuntimeError("Local Docker storage budget is insufficient before starting " + arm["name"])
            actual_sources = {name: experiment.sha256(experiment.runtime_source(name)) for name in experiment.SOURCE_FILES}
            if actual_sources != plan["source_sha256"]:
                raise RuntimeError("Runtime sources changed after the comparison plan was created")
            run = Path(arm["run_dir"])
            update(root, status="running", current_arm=arm["name"], stage="preparing")
            if not (run / "run_status.json").exists():
                experiment.main([
                    "start", "--run-dir", str(run), "--report-root", arm["report_root"],
                    "--container", arm["container"], "--image", plan["image_id"],
                    "--base-model", str(experiment.MODEL), "--num-epochs", str(arm.get("num_epochs", plan.get("max_epochs", 10))),
                    "--lr-scheduler-type", plan.get("lr_scheduler_type", "cosine"),
                    "--early-stopping-patience", str(plan.get("early_stopping_patience", 3)),
                    "--eval-save-fraction", str(plan.get("eval_save_fraction", 0.05)),
                    "--save-epoch-checkpoint", str(arm.get("save_epoch_checkpoint", 0)),
                    "--minimum-epochs", str(arm.get("minimum_epochs", 0)),
                    "--no-generative-validation", "--final-ranking-eval",
                    "--include-title-history" if arm["include_title_history"] else "--no-include-title-history",
                ])
            while True:
                state = experiment.read_json(run / "run_status.json", {})
                update(root, stage=state.get("stage", "preparing"),
                       current_run_status=state.get("status", "pending"),
                       current_progress=state.get("progress", {}))
                if state.get("status") == "completed":
                    break
                if state.get("status") in ("failed", "cancelled"):
                    raise RuntimeError(arm["name"] + " did not complete: " + str(state.get("error", state)))
                time.sleep(15)
        statuses = [experiment.read_json(Path(arm["run_dir"]) / "run_status.json") for arm in plan["arms"]]
        if any(state.get("runtime_image_id") != plan["image_id"] for state in statuses):
            raise RuntimeError("Runtime image identity differs from the immutable comparison plan")
        if statuses[0].get("source_sha256") != statuses[1].get("source_sha256"):
            raise RuntimeError("Runtime sources changed between the comparison arms")
        update(root, stage="comparing")
        if plan.get("comparison_kind") == "linear_epochs":
            from scripts.compare_industrial_linear_epochs import compare_runs, write_comparison
        else:
            from scripts.compare_industrial_title_history import compare_runs, write_comparison
        result = compare_runs(plan["arms"][0]["run_dir"], plan["arms"][1]["run_dir"],
                              experiment.DATA, experiment.INPUT)
        files = write_comparison(result, plan["comparison_reports"])
        update(root, status="completed", stage="completed", files=files, finished_at_utc=experiment.now())
        return 0
    except BaseException as error:
        traceback.print_exc()
        update(root, status="failed", stage="failed", error=str(error), finished_at_utc=experiment.now())
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "worker", "status"))
    parser.add_argument("--experiment-root", type=Path, default=experiment.EXPERIMENT_ROOT)
    parser.add_argument("--report-root", type=Path, default=ROOT / "output/reports/industrial_05b_cosine_ep10")
    parser.add_argument("--image", default="minionerec:h20-original-torch28")
    args = parser.parse_args(argv)
    root = args.experiment_root.resolve()
    root.relative_to(experiment.WORKSPACE.resolve())
    if args.action == "status":
        print(json.dumps(experiment.read_json(root / "comparison_status.json", {"status": "not_started"}),
                         ensure_ascii=False, indent=2))
        return 0
    if args.action == "worker":
        return worker(root)
    experiment.check_prepared()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Existing comparison retained; choose a new --experiment-root")
    reports = args.report_root.resolve()
    reports.relative_to(experiment.WORKSPACE.resolve())
    if reports.exists() and any(reports.iterdir()):
        raise ValueError("Existing comparison reports retained; choose a new --report-root")
    docker = os.environ.get("DOCKER", "docker")
    image_id = subprocess.check_output([docker, "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    container_suffix = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    arms = [{"name": name, "include_title_history": extra,
             "run_dir": str(root / name / "sft"), "report_root": str(reports / name),
             "container": "minionerec-industrial05b-" + container_suffix + "-" + name.replace("_", "-")}
            for name, extra in (("baseline", False), ("title_history", True))]
    root.mkdir(parents=True)
    plan = {"created_at_utc": experiment.now(), "category": experiment.CATEGORY,
            "base_model": str(experiment.MODEL), "image_id": image_id,
            "max_epochs": 10, "lr_scheduler_type": "cosine", "early_stopping_patience": 3,
            "source_sha256": {name: experiment.sha256(experiment.runtime_source(name)) for name in experiment.SOURCE_FILES},
            "arms": arms, "comparison_reports": str(reports / "comparison")}
    experiment.atomic_json(root / "comparison_plan.json", plan)
    update(root, status="running", stage="starting", started_at_utc=experiment.now())
    with (root / "comparison_worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                                    "--experiment-root", str(root)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(update(root, worker_pid=process.pid), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
