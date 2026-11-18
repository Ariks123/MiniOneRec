#!/usr/bin/env python3
"""Compare completed 0.5B Industrial SFT runs with and without title-history input.

This command only reads existing runs. It reaggregates full test predictions and
writes a comparison after both runs pass the shared protocol and input checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_industrial_sft import (
    DATASET, TOP_K, atomic_text, read_json, require, validated_evaluation, write_json,
)

GROUPS = ("overall", "behavior_warm", "behavior_cold", "not_repeat_last", "collision_free")
BASE_TASKS = ("sid_sequence_recommendation", "history_to_title", "sid2title", "title2sid")
EXTRA_TASK = "title_history_to_sid"
REQUIRED_HYPERPARAMETERS = (
    "world_size", "micro_batch_size", "gradient_accumulation_steps", "global_batch_size",
    "max_epochs", "learning_rate", "warmup_steps", "lr_scheduler_type", "cutoff_len",
    "seed", "early_stopping_patience", "eval_save_fraction", "bf16",
)


def load_run(run_dir, data_dir, input_dir, preparation, include_title_history):
    """Load one completed run and recompute its accepted full-test ranking metrics."""
    run_dir = Path(run_dir).resolve()
    config = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "evaluation/metrics.json")
    label = "augmented" if include_title_history else "baseline"
    require(config.get("stage") == "sft", f"{label}: missing SFT run configuration")
    model = config.get("model", {})
    require(Path(model.get("source", "")).name == "Qwen2.5-0.5B",
            f"{label}: comparison requires Qwen2.5-0.5B Base")
    require(model.get("train_from_scratch") is False and config.get("resume_from_checkpoint") is None,
            f"{label}: comparison requires fresh pretrained Base model initialization")
    identity = model.get("identity", {}).get("sha256")
    require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity),
            f"{label}: missing verified source-model identity SHA-256")
    data = config.get("data", {})
    require(data.get("category") == DATASET and data.get("sample") == -1,
            f"{label}: comparison requires the complete Industrial training dataset")
    require(data.get("include_title_history") is include_title_history,
            f"{label}: title-history task flag differs from the requested comparison arm")
    hyper = config.get("hyperparameters", {})
    require(all(key in hyper for key in REQUIRED_HYPERPARAMETERS),
            f"{label}: incomplete training hyperparameters")
    require(hyper.get("max_epochs") == 10 and hyper.get("lr_scheduler_type") == "cosine",
            f"{label}: comparison requires maximum 10 epochs and cosine scheduling")
    tasks = config.get("task_counts", {})
    expected_keys = set(BASE_TASKS) | ({EXTRA_TASK} if include_title_history else set())
    require(set(tasks) == expected_keys and all(type(value) is int and value > 0 for value in tasks.values()),
            f"{label}: unexpected training tasks or invalid task counts")
    train_rows = preparation.get("rows", {}).get("train")
    require(tasks["sid_sequence_recommendation"] == tasks["history_to_title"] == train_rows,
            f"{label}: original sequence tasks must cover every training row")
    if include_title_history:
        require(tasks[EXTRA_TASK] == train_rows,
                f"{label}: title-history task must cover every training row")
    require(data.get("train_samples") == sum(tasks.values()),
            f"{label}: expanded training-sample count differs from its task counts")
    require(data.get("validation_samples") == preparation.get("rows", {}).get("valid"),
            f"{label}: validation sample count differs from prepared data")
    require(summary.get("status") == "completed", f"{label}: training is not completed")
    require(summary.get("final_checkpoint_contains_best_model") is True,
            f"{label}: final model must contain the validation-selected best checkpoint")
    final_checkpoint = run_dir / "final_checkpoint"
    require(Path(summary.get("final_checkpoint", "")).resolve() == final_checkpoint
            and Path(evaluation.get("model_path", "")).resolve() == final_checkpoint,
            f"{label}: evaluated checkpoint does not match this run's final best model")
    require(Path(summary.get("best_checkpoint", "")).resolve().parent == run_dir,
            f"{label}: best checkpoint belongs to another run")
    require(isinstance(summary.get("best_metric"), (float, int))
            and math.isfinite(summary["best_metric"]), f"{label}: missing finite best validation loss")
    steps, planned, epoch = (summary.get(key) for key in ("global_step", "planned_max_steps", "epoch"))
    require(type(steps) is int and type(planned) is int and 0 < steps <= planned,
            f"{label}: invalid achieved/planned training steps")
    require(isinstance(epoch, (float, int)) and 0 < epoch <= 10,
            f"{label}: achieved epoch is outside the ten-epoch limit")
    expected_stop = "early_stopping" if steps < planned else "max_epochs"
    require(summary.get("stop_reason") == expected_stop,
            f"{label}: training stop reason does not match achieved steps")
    require(evaluation.get("scope") == "complete_original_test_split",
            f"{label}: evaluation is not the complete original test split")
    accepted = validated_evaluation(run_dir, data_dir, input_dir, config, evaluation, preparation)
    require(accepted["status"] == "passed", f"{label}: prediction validation failed")
    return {"run_dir": str(run_dir), "config": config, "summary": summary, "accepted": accepted}


def compare_runs(baseline_run, augmented_run, data_dir, input_dir):
    """Return a comparison only after both arms pass the same evaluation contract."""
    baseline_run, augmented_run = map(lambda path: Path(path).resolve(), (baseline_run, augmented_run))
    require(baseline_run != augmented_run, "Comparison arms must use separate run directories")
    data_dir, input_dir = map(lambda path: Path(path).resolve(), (data_dir, input_dir))
    preparation = read_json(data_dir / "preparation_manifest.json")
    baseline = load_run(baseline_run, data_dir, input_dir, preparation, False)
    augmented = load_run(augmented_run, data_dir, input_dir, preparation, True)
    left, right = baseline["config"], augmented["config"]
    require(left["model"]["identity"]["sha256"] == right["model"]["identity"]["sha256"],
            "The two runs started from different pretrained model bytes")
    require(Path(left["model"]["source"]).resolve() == Path(right["model"]["source"]).resolve(),
            "The two runs used different pretrained model source paths")
    require(left["hyperparameters"] == right["hyperparameters"],
            "Shared training hyperparameters differ between the two arms")
    require(left["data"]["sha256"] == right["data"]["sha256"],
            "Training input identities differ between the two arms")
    require(all(left["task_counts"][key] == right["task_counts"][key] for key in BASE_TASKS),
            "Original task sample counts differ between the two arms")
    require(left["data"]["validation_samples"] == right["data"]["validation_samples"],
            "Validation sample counts differ between the two arms")
    require(baseline["accepted"]["source_sha256"].keys() == augmented["accepted"]["source_sha256"].keys(),
            "Accepted input identity fields differ")
    for key, value in baseline["accepted"]["source_sha256"].items():
        if key != "predictions":
            require(value == augmented["accepted"]["source_sha256"][key],
                    "Accepted input identity differs: " + key)
    groups = [{row["key"]: row for row in arm["accepted"]["strata"]} for arm in (baseline, augmented)]
    rows = []
    for group in GROUPS:
        require(group in groups[0] and group in groups[1], "Missing evaluation stratum: " + group)
        first, second = (item[group] for item in groups)
        require(first["rows"] == second["rows"], "Test stratum sizes differ: " + group)
        for metric, source_name in (("Recall", "HR"), ("NDCG", "NDCG")):
            for k in TOP_K:
                key = f"{source_name}@{k}"
                a, b = first["metrics"][key], second["metrics"][key]
                rows.append({"stratum": group, "rows": first["rows"], "metric": metric, "k": k,
                             "baseline": a, "augmented": b,
                             "difference_percentage_points": None if a is None or b is None else 100 * (b - a)})
    def run_metadata(arm):
        summary, config = arm["summary"], arm["config"]
        return {"run_dir": arm["run_dir"], "include_title_history": config["data"]["include_title_history"],
                "train_samples": config["data"]["train_samples"], "task_counts": config["task_counts"],
                "epoch": summary["epoch"], "global_step": summary["global_step"],
                "planned_max_steps": summary["planned_max_steps"], "stop_reason": summary["stop_reason"],
                "best_checkpoint": summary["best_checkpoint"], "best_validation_loss": summary["best_metric"],
                "final_checkpoint": summary["final_checkpoint"],
                "prediction_sha256": arm["accepted"]["source_sha256"]["predictions"]}
    return {"format_version": 1, "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET, "model": "Qwen2.5-0.5B Base",
            "source_model_identity_sha256": left["model"]["identity"]["sha256"],
            "shared_hyperparameters": left["hyperparameters"],
            "protocol": {"scope": "complete_original_test_split", "num_beams": 50,
                         "prompt_style": "original-eval", "validation_prompt_style": "training",
                         "test_rows": baseline["accepted"]["rows"],
                         "catalog_items": baseline["accepted"]["catalog_items"],
                         "unique_sids": baseline["accepted"]["unique_sids"],
                         "data_sha256": {key: value for key, value in baseline["accepted"]["source_sha256"].items()
                                         if key != "predictions"},
                         "model_selection": "Best validation loss; test is used only after training",
                         "metric_semantics": baseline["accepted"]["metric_semantics"]},
            "baseline": run_metadata(baseline), "augmented": run_metadata(augmented),
            "metrics": rows,
            "interpretation": [
                "The augmented arm adds history titles to next-item SID supervision; all original tasks are retained.",
                "Maximum epochs match, while sample counts and optimizer-step budgets increase with the additional task.",
                "This compares complete training recipes; it does not separate task content from extra training compute.",
                "Positive differences favor the augmented run. A single seed does not establish statistical significance.",
                "Coldness is defined by training behavior item IDs; cold items can still appear in catalog alignment tasks."
            ]}


def write_comparison(result, output_dir):
    """Write standalone JSON, CSV, and HTML without altering either source run."""
    output_dir = Path(output_dir).resolve()
    runs = [Path(result[arm]["run_dir"]).resolve() for arm in ("baseline", "augmented")]
    for run in runs:
        require(not output_dir.is_relative_to(run) and not run.is_relative_to(output_dir),
                "Comparison output must be separate from both training runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", result)
    buffer = io.StringIO(newline="")
    fields = ("stratum", "rows", "metric", "k", "baseline", "augmented", "difference_percentage_points")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result["metrics"])
    atomic_text(output_dir / "comparison.csv", buffer.getvalue())
    def pct(value):
        return "—" if value is None else f"{100 * value:.4f}%"
    def pp(value):
        return "—" if value is None else f"{value:+.4f}"
    metadata_rows = "".join(
        "<tr><td>" + html.escape(label) + "</td><td>" + html.escape(str(result["baseline"][key]))
        + "</td><td>" + html.escape(str(result["augmented"][key])) + "</td></tr>"
        for key, label in (("train_samples", "训练样本"), ("epoch", "实际 epoch"),
                           ("global_step", "实际 steps"), ("stop_reason", "停止原因"),
                           ("best_validation_loss", "最佳验证 loss")))
    metric_rows = "".join(
        "<tr><td>" + html.escape(row["stratum"]) + f"</td><td>{row['rows']}</td><td>"
        + html.escape(f"{row['metric']}@{row['k']}") + "</td><td>" + pct(row["baseline"])
        + "</td><td>" + pct(row["augmented"]) + "</td><td>" + pp(row["difference_percentage_points"])
        + "</td></tr>" for row in result["metrics"])
    notes = "".join("<li>" + html.escape(note) + "</li>" for note in result["interpretation"])
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial 0.5B title-history SFT comparison</title>
<style>body{font:16px/1.6 system-ui,sans-serif;margin:24px;color:#18313b}main{max-width:1100px;margin:auto}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #cad7dc;padding:8px;text-align:left}th{background:#edf3f5}td{font-variant-numeric:tabular-nums}a{color:#176b88}</style><main>
<h1>Industrial 2018 · 0.5B SFT 对照</h1>
<p>基线：原四类任务；新增组：保留原任务，并加入历史商品标题 → 下一商品 SID。两组均为 cosine、最多 10 epoch，以验证 loss 选择模型。</p>
<p><a href="comparison.json">JSON</a> · <a href="comparison.csv">CSV</a></p>
<table><tr><th>项目</th><th>基线</th><th>新增任务</th></tr>""" + metadata_rows + "</table>"
    page += "<p>完整测试集、完整商品候选目录、beam=50；增量单位为百分点，正数表示新增组更高。</p>"
    page += "<table><tr><th>分组</th><th>样本数</th><th>指标</th><th>基线</th><th>新增任务</th><th>变化（百分点）</th></tr>" + metric_rows + "</table><ul>" + notes + "</ul></main></html>"
    atomic_text(output_dir / "index.html", page)
    return {name: str(output_dir / name) for name in ("comparison.json", "comparison.csv", "index.html")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--augmented-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_runs(args.baseline_run, args.augmented_run, args.data_dir, args.input_dir)
    files = write_comparison(result, args.output_dir)
    selected = [row for row in result["metrics"] if row["stratum"] == "overall" and row["metric"] == "Recall" and row["k"] in (10, 20, 50)]
    print(json.dumps({"status": result["status"], "files": files, "overall_recall": selected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
