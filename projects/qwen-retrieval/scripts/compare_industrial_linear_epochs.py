#!/usr/bin/env python3
"""Compare fresh linear 3/5-epoch runs and the five-epoch run's epoch-3 weights.

Only completed post-training evaluations are read. Source runs, snapshots,
predictions, and data are never modified by this command.
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

EXPECTED_TASK_COUNTS = {
    "sid_sequence_recommendation": 36259, "history_to_title": 36259,
    "sid2title": 3670, "title2sid": 3646, "title_history_to_sid": 36259,
}
GROUPS = ("overall", "behavior_warm", "behavior_cold", "not_repeat_last", "collision_free")
ARM_LABELS = {
    "epochs3": "3 epoch 计划：验证 loss 最佳模型",
    "epochs5": "5 epoch 计划：验证 loss 最佳模型",
    "epoch3_snapshot": "5 epoch 计划：第 3 epoch 即时权重",
}
ALLOWED_HYPERPARAMETER_DIFFERENCES = {"max_epochs", "save_epoch_checkpoint", "minimum_epochs"}
REQUIRED_HYPERPARAMETERS = (
    "world_size", "micro_batch_size", "gradient_accumulation_steps", "global_batch_size",
    "max_epochs", "learning_rate", "warmup_steps", "lr_scheduler_type", "cutoff_len",
    "seed", "early_stopping_patience", "eval_save_fraction", "bf16", "generative_validation",
)


def load_training(run_dir, maximum_epochs, preparation):
    run_dir = Path(run_dir).resolve()
    config = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "training_summary.json")
    require(config.get("stage") == "sft", "Missing SFT configuration: " + str(run_dir))
    model = config.get("model", {})
    require(Path(model.get("source", "")).name == "Qwen2.5-0.5B"
            and model.get("train_from_scratch") is False
            and config.get("resume_from_checkpoint") is None,
            "Both runs must start fresh from Qwen2.5-0.5B Base")
    identity = model.get("identity", {}).get("sha256")
    require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity),
            "Missing verified model identity SHA-256")
    data = config.get("data", {})
    require(data.get("category") == DATASET and data.get("sample") == -1
            and data.get("include_title_history") is True,
            "Both runs must include all Industrial title-history training samples")
    require(config.get("task_counts") == EXPECTED_TASK_COUNTS
            and data.get("train_samples") == sum(EXPECTED_TASK_COUNTS.values()),
            "The complete five-task mixture or expanded sample count differs")
    require(preparation.get("rows", {}).get("train") == EXPECTED_TASK_COUNTS["sid_sequence_recommendation"]
            and data.get("validation_samples") == preparation.get("rows", {}).get("valid"),
            "Prepared training/validation counts differ")
    hyper = config.get("hyperparameters", {})
    require(all(key in hyper for key in REQUIRED_HYPERPARAMETERS), "Incomplete training hyperparameters")
    require(hyper["max_epochs"] == maximum_epochs and hyper["lr_scheduler_type"] == "linear"
            and hyper["generative_validation"] is False
            and hyper.get("model_selection_metric", "eval_loss") == "eval_loss",
            "Comparison requires linear 3/5-epoch plans with loss-only model selection")
    require(hyper["learning_rate"] == 3e-4 and hyper["warmup_steps"] == 20
            and hyper["eval_save_fraction"] == .05,
            "Linear learning rate, warmup, or original evaluation fraction differs")
    batch = hyper["global_batch_size"]
    require(type(batch) is int and batch > 0
            and batch == hyper["world_size"] * hyper["micro_batch_size"] * hyper["gradient_accumulation_steps"],
            "Invalid effective batch size")
    updates_per_epoch = math.ceil(data["train_samples"] / batch)
    require(summary.get("status") == "completed"
            and summary.get("selection_metric", "eval_loss") == "eval_loss"
            and summary.get("final_checkpoint_contains_best_model") is True,
            "Training must complete and reload the validation-loss best model")
    steps, planned, epoch = (summary.get(key) for key in ("global_step", "planned_max_steps", "epoch"))
    require(type(steps) is int and type(planned) is int and 0 < steps <= planned
            and planned == updates_per_epoch * maximum_epochs,
            "Achieved/planned training steps do not match the epoch budget")
    require(isinstance(epoch, (float, int)) and 0 < epoch <= maximum_epochs
            and summary.get("stop_reason") == ("early_stopping" if steps < planned else "max_epochs"),
            "Training epoch or stop reason differs from the achieved step budget")
    require(isinstance(summary.get("best_metric"), (float, int))
            and math.isfinite(summary["best_metric"]) and summary["best_metric"] >= 0,
            "Missing finite best validation loss")
    best = Path(summary.get("best_checkpoint", "")).resolve()
    match = re.fullmatch(r"checkpoint-([0-9]+)", best.name)
    require(best.parent == run_dir and match and 0 < int(match[1]) <= steps,
            "Best checkpoint does not belong to the completed run")
    final = run_dir / "final_checkpoint"
    require(Path(summary.get("final_checkpoint", "")).resolve() == final,
            "Final checkpoint belongs to another run")
    state = read_json(run_dir / "trainer_state.json")
    require(not any(any(key.startswith(("eval_ndcg", "eval_recall", "eval_HR")) for key in row)
                    for row in state.get("log_history", [])),
            "Ranking metrics were unexpectedly used during training")
    return {"run_dir": run_dir, "config": config, "summary": summary,
            "status": read_json(run_dir / "run_status.json"), "updates_per_epoch": updates_per_epoch,
            "model_path": final, "selected_step": int(match[1]),
            "selected_epoch": int(match[1]) / updates_per_epoch}


def checked_evaluation(arm, evaluation_dir, model_path, data_dir, input_dir, preparation):
    evaluation_dir, model_path = map(lambda value: Path(value).resolve(), (evaluation_dir, model_path))
    evaluation = read_json(evaluation_dir / "metrics.json")
    require(evaluation.get("scope") == "complete_original_test_split"
            and Path(evaluation.get("model_path", "")).resolve() == model_path,
            "Evaluation does not identify the requested exported model and full test split")
    for field in ("seed", "cutoff_len", "world_size"):
        require(evaluation.get(field) == arm["config"]["hyperparameters"][field],
                "Evaluation differs from the recorded training protocol: " + field)
    accepted = validated_evaluation(arm["run_dir"], data_dir, input_dir, arm["config"],
                                    evaluation, preparation,
                                    predictions_path=evaluation_dir / "predictions.jsonl")
    require(accepted["status"] == "passed", "Prediction validation failed")
    return {"accepted": accepted, "evaluation": evaluation,
            "evaluation_dir": str(evaluation_dir), "model_path": str(model_path)}


def compare_runs(three_epoch_run, five_epoch_run, data_dir, input_dir):
    """Check input/protocol identity and recompute all three completed evaluations."""
    three_epoch_run, five_epoch_run, data_dir, input_dir = map(
        lambda value: Path(value).resolve(), (three_epoch_run, five_epoch_run, data_dir, input_dir))
    require(three_epoch_run != five_epoch_run, "Three- and five-epoch runs must be separate")
    preparation = read_json(data_dir / "preparation_manifest.json")
    three, five = (load_training(path, maximum, preparation)
                   for path, maximum in ((three_epoch_run, 3), (five_epoch_run, 5)))
    left, right = three["config"], five["config"]
    require(left["model"]["identity"]["sha256"] == right["model"]["identity"]["sha256"]
            and Path(left["model"]["source"]).resolve() == Path(right["model"]["source"]).resolve(),
            "The two runs used different pretrained model identities")
    require(left["data"]["sha256"] == right["data"]["sha256"], "Training input hashes differ")
    common = lambda config: {key: value for key, value in config["hyperparameters"].items()
                             if key not in ALLOWED_HYPERPARAMETER_DIFFERENCES}
    require(common(left) == common(right), "Shared training hyperparameters differ")
    for field in ("runtime_image_id", "runtime_versions", "source_sha256"):
        require(three["status"].get(field) == five["status"].get(field),
                "Runtime identity differs: " + field)
    snapshot_path = five_epoch_run / "epoch_3_checkpoint"
    snapshot = read_json(snapshot_path / "snapshot_metadata.json")
    require(snapshot.get("format_version") == 1
            and snapshot.get("snapshot_type") == "epoch_end_training_weights"
            and snapshot.get("capture_event") == "on_epoch_end"
            and snapshot.get("requested_epoch") == 3
            and snapshot.get("best_model_reloaded") is False,
            "Epoch-3 snapshot was not captured from live training weights before best-model reload")
    require(isinstance(snapshot.get("epoch"), (float, int))
            and math.isclose(snapshot["epoch"], 3, abs_tol=1e-6, rel_tol=0)
            and type(snapshot.get("global_step")) is int
            and snapshot["global_step"] == 3 * five["updates_per_epoch"]
            and math.isclose(snapshot.get("steps_per_epoch", 0), five["updates_per_epoch"], abs_tol=1e-6, rel_tol=0)
            and Path(snapshot.get("checkpoint_path", "")).resolve() == snapshot_path,
            "Epoch-3 snapshot epoch, step, or export path differs from the training plan")
    require(five["summary"]["global_step"] >= snapshot["global_step"]
            and five["summary"]["epoch"] >= 3, "Five-epoch run ended before the actual epoch-3 snapshot")
    evaluated = {
        "epochs3": checked_evaluation(three, three_epoch_run / "evaluation", three["model_path"], data_dir, input_dir, preparation),
        "epochs5": checked_evaluation(five, five_epoch_run / "evaluation", five["model_path"], data_dir, input_dir, preparation),
        "epoch3_snapshot": checked_evaluation(five, five_epoch_run / "evaluation_epoch3", snapshot_path, data_dir, input_dir, preparation),
    }
    reference = evaluated["epochs3"]
    for item in evaluated.values():
        for field in ("seed", "cutoff_len", "max_new_tokens", "world_size"):
            require(item["evaluation"].get(field) == reference["evaluation"].get(field),
                    "Evaluation protocol differs: " + field)
        for key, value in reference["accepted"]["source_sha256"].items():
            if key != "predictions":
                require(item["accepted"]["source_sha256"].get(key) == value,
                        "Accepted evaluation input differs: " + key)
    groups = {name: {group["key"]: group for group in item["accepted"]["strata"]}
              for name, item in evaluated.items()}
    metrics = []
    for arm in ARM_LABELS:
        for group in GROUPS:
            current, baseline = groups[arm][group], groups["epochs3"][group]
            require(current["rows"] == baseline["rows"], "Evaluation stratum sizes differ: " + group)
            for label, metric in (("Recall", "HR"), ("NDCG", "NDCG")):
                for k in TOP_K:
                    value, initial = current["metrics"][f"{metric}@{k}"], baseline["metrics"][f"{metric}@{k}"]
                    metrics.append({"arm": arm, "stratum": group, "rows": current["rows"],
                                    "metric": label, "k": k, "value": value,
                                    "difference_percentage_points_vs_epochs3": None if value is None or initial is None else 100 * (value - initial)})
    def metadata(arm, training):
        item = evaluated[arm]
        return {"label": ARM_LABELS[arm], "run_dir": str(training["run_dir"]),
                "model_path": item["model_path"], "evaluation_dir": item["evaluation_dir"],
                "maximum_epochs": training["config"]["hyperparameters"]["max_epochs"],
                "achieved_epoch": training["summary"]["epoch"],
                "global_step": training["summary"]["global_step"],
                "stop_reason": training["summary"]["stop_reason"],
                "selected_epoch": snapshot["epoch"] if arm == "epoch3_snapshot" else training["selected_epoch"],
                "selected_step": snapshot["global_step"] if arm == "epoch3_snapshot" else training["selected_step"],
                "selection": "actual_epoch_end_training_weights" if arm == "epoch3_snapshot" else "best_validation_loss",
                "best_validation_loss": None if arm == "epoch3_snapshot" else training["summary"]["best_metric"],
                "test_metrics": item["evaluation"]["metrics"],
                "prediction_sha256": item["accepted"]["source_sha256"]["predictions"]}
    return {"format_version": 1, "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET, "model": "Qwen2.5-0.5B Base", "source_model_identity_sha256": left["model"]["identity"]["sha256"],
            "task_counts": left["task_counts"], "train_samples": left["data"]["train_samples"],
            "shared_hyperparameters": common(left),
            "epochs3": metadata("epochs3", three), "epochs5": metadata("epochs5", five),
            "epoch3_snapshot": {**metadata("epoch3_snapshot", five), "snapshot_metadata": snapshot},
            "protocol": {"scope": "complete_original_test_split", "test_rows": reference["accepted"]["rows"],
                         "catalog_items": reference["accepted"]["catalog_items"], "unique_sids": reference["accepted"]["unique_sids"],
                         "num_beams": 50, "prompt_style": "original-eval", "validation_prompt_style": "training",
                         "model_selection": "Validation loss only; ranking metrics are computed after training",
                         "metric_semantics": reference["accepted"]["metric_semantics"]},
            "metrics": metrics,
            "interpretation": [
                "Both runs start fresh from identical 0.5B Base weights and retain the complete title-history task mixture.",
                "Three- and five-epoch linear plans have different decay trajectories; their epoch-3 weights are not interchangeable.",
                "The epoch-3 snapshot contains live weights captured before validation-best model reload, not the five-epoch loss-best weights.",
                "The shared evaluation fraction .05 yields approximately .15-epoch versus .25-epoch loss-validation intervals.",
                "Maximum epochs, minimum epochs and the requested snapshot differ; all other recorded training hyperparameters match.",
                "Final model selection uses validation loss only. The three post-training test evaluations do not select or alter models.",
                "Positive differences are percentage-point improvements over the three-epoch run's loss-best model; one seed is descriptive."
            ]}


def write_comparison(result, output_dir):
    output_dir = Path(output_dir).resolve()
    for name in ("epochs3", "epochs5"):
        run = Path(result[name]["run_dir"]).resolve()
        require(not output_dir.is_relative_to(run) and not run.is_relative_to(output_dir),
                "Comparison output must be outside both source training runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", result)
    buffer = io.StringIO(newline="")
    fields = ("arm", "stratum", "rows", "metric", "k", "value", "difference_percentage_points_vs_epochs3")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result["metrics"])
    atomic_text(output_dir / "comparison.csv", buffer.getvalue())
    percent = lambda value: "—" if value is None else f"{100 * value:.4f}%"
    main_rows = "".join(
        "<tr><td>" + html.escape(result[name]["label"]) + f"</td><td>{result[name]['achieved_epoch']}</td><td>{result[name]['selected_epoch']:.4g}</td><td>"
        + percent(result[name]["test_metrics"]["HR@10"]) + "</td><td>"
        + percent(result[name]["test_metrics"]["HR@20"]) + "</td><td>"
        + percent(result[name]["test_metrics"]["HR@50"]) + "</td><td>"
        + percent(result[name]["test_metrics"]["NDCG@10"]) + "</td></tr>" for name in ARM_LABELS)
    detail_rows = "".join(
        "<tr><td>" + html.escape(row["arm"]) + "</td><td>" + html.escape(row["stratum"])
        + f"</td><td>{row['rows']}</td><td>{row['metric']}@{row['k']}</td><td>" + percent(row["value"])
        + "</td><td>" + ("—" if row["difference_percentage_points_vs_epochs3"] is None else f"{row['difference_percentage_points_vs_epochs3']:+.4f}")
        + "</td></tr>" for row in result["metrics"])
    notes = "".join("<li>" + html.escape(note) + "</li>" for note in result["interpretation"])
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial 0.5B linear 3/5 epoch comparison</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#18313b;margin:24px}main{max-width:1200px;margin:auto}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #cad7dc;text-align:left}th{background:#edf3f5}td{font-variant-numeric:tabular-nums}a{color:#176b88}</style><main>
<h1>Industrial 2018 · 0.5B linear · 3/5 epoch 对照</h1>
<p>两组均含标题历史 → 下一 SID 任务；训练中只计算验证 loss 并据此早停、选模型。以下排名指标均在训练结束后，以完整测试集、原始评估提示、beam50计算。</p>
<p>第3epoch即时权重是5epoch计划训练当时的模型，不代表该运行最终选择的验证loss最佳模型。</p>
<p><a href="comparison.json">JSON</a> · <a href="comparison.csv">CSV</a></p>
<table><tr><th>模型</th><th>所属运行实际epoch</th><th>模型对应epoch</th><th>Recall@10</th><th>Recall@20</th><th>Recall@50</th><th>NDCG@10</th></tr>""" + main_rows + "</table>"
    page += "<h2>全部指标及分组</h2><table><tr><th>模型</th><th>分组</th><th>样本数</th><th>指标</th><th>结果</th><th>相对3epoch模型变化（百分点）</th></tr>" + detail_rows + "</table><ul>" + notes + "</ul></main></html>"
    atomic_text(output_dir / "index.html", page)
    return {name: str(output_dir / name) for name in ("comparison.json", "comparison.csv", "index.html")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-epoch-run", type=Path, required=True)
    parser.add_argument("--five-epoch-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_runs(args.three_epoch_run, args.five_epoch_run, args.data_dir, args.input_dir)
    print(json.dumps({"status": result["status"], "files": write_comparison(result, args.output_dir)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
