#!/usr/bin/env python3
"""Build evidence-backed SFT training and validation reports, including live runs.

Only the Python standard library is imported until chart rendering is requested.
This entry point never starts training, touches source datasets, or imports torch.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PAPER_REFERENCE_PATH = ROOT / "config/minionerec_paper_reference.json"
TRAINING_DIRECTORY = "06_sft_training"
VALIDATION_DIRECTORY = "07_sft_validation"
TOP_K = (1, 3, 5, 10, 20, 50)
SOURCE_NAMES = (
    "run_status.json", "run_config.json", "baseline_validation.json",
    "training_summary.json", "monitor/metrics.jsonl", "monitor/summary.json",
    "monitor/events.jsonl", "gpu_metrics.jsonl", "evaluation/metrics.json", "comparison_baselines.json",
    "behavior_shift_analysis.json",
)
STATUS_LABELS = {
    "pending": "尚未开始", "preparing": "准备中", "running": "进行中",
    "training": "训练中", "validating": "验证中", "completed": "已完成",
    "passed": "检查通过", "failed": "失败", "incomplete": "数据不完整",
}
CSS = """*{box-sizing:border-box}body{margin:0;background:#f1f5f6;color:#172d35;font:16px/1.65 'Segoe UI','Noto Sans CJK SC','Microsoft YaHei',sans-serif}main{max-width:1180px;margin:auto;padding:40px 24px}h1{font-size:32px;line-height:1.3;margin:8px 0 16px}h2{font-size:23px;margin:0 0 14px}section,.card{background:#fff;border:1px solid #dae6e9;border-radius:12px;padding:24px;margin:20px 0}.eyebrow{color:#167d8d;letter-spacing:.1em;font-size:13px;font-weight:700}.muted{color:#657983;font-size:14px}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.kpis .card{margin:0}.card b{display:block;font-size:28px;color:#167d8d}a{color:#126f80}.button{display:inline-block;padding:8px 13px;margin:5px 8px 5px 0;background:#edf5f7;border-radius:6px;text-decoration:none}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e3ebee;vertical-align:top;overflow-wrap:anywhere}th{background:#eef5f7}.scroll{overflow:auto}img{display:block;width:100%;height:auto}.badge{display:inline-block;padding:3px 10px;border-radius:6px;background:#e8f2f4;color:#126f80}.failed{background:#ffe7e4;color:#9c3529}.warning{background:#fff4d9;border-left:4px solid #b77a19;padding:14px 18px}code{background:#edf5f7;padding:2px 5px;border-radius:4px}footer{font-size:13px;color:#657983;padding:20px 0}@media(max-width:740px){main{padding:24px 14px}.kpis{grid-template-columns:1fr}h1{font-size:27px}section{padding:18px}}"""


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first(*values):
    return next((value for value in values if value is not None), None)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path, issues=None):
    if not Path(path).is_file():
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        if issues is not None:
            issues.append(f"{Path(path).name}: {error}")
        return {}


def read_jsonl(path, issues=None):
    """Ignore an incomplete final append while preserving all complete records."""
    if not Path(path).is_file():
        return []
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        if issues is not None:
            issues.append(f"{Path(path).name}: {error}")
        return []
    lines = content.splitlines()
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
            elif issues is not None:
                issues.append(f"{Path(path).name}:{number}: expected an object")
        except ValueError:
            if number != len(lines) or content.endswith("\n"):
                if issues is not None:
                    issues.append(f"{Path(path).name}:{number}: invalid complete JSON line")
    return records


def merged_steps(records):
    """Resume attempts and evaluation-only log rows do not double-count steps."""
    merged = {}
    for record in records:
        step = first(record.get("step"), record.get("global_step"))
        if not finite(step) or int(step) < 0:
            continue
        key = int(step)
        row = merged.setdefault(key, {"step": key})
        row.update({name: value for name, value in record.items() if value is not None})
        row["step"] = key
        for output_key, input_key in (("grad_norm_rolling_mean", "rolling_grad_norm_mean"),
                                      ("grad_norm_rolling_max", "rolling_grad_norm_max")):
            if input_key in row:
                row[output_key] = row[input_key]
    return [merged[step] for step in sorted(merged)]


def gradient_statistics(records, window=100):
    valid = []
    missing = invalid = skipped = 0
    for row in records:
        # Evaluation logs without optimizer metrics are not missing-gradient updates.
        is_update = any(name in row for name in ("loss", "grad_norm", "step_skipped", "optimizer_step_skipped"))
        if not is_update:
            continue
        if first(row.get("step_skipped"), row.get("optimizer_step_skipped"), False):
            skipped += 1
            continue
        value = row.get("grad_norm")
        if row.get("grad_norm_nonfinite"):
            invalid += 1
        elif value is None:
            missing += 1
        elif not finite(value) or value < 0:
            invalid += 1
        else:
            valid.append((row["step"], float(value)))
    maximum = max(valid, key=lambda item: item[1]) if valid else (None, None)
    recent = valid[-window:]
    return {
        "definition": "pre-clip, unscaled, full-model global gradient L2 norm",
        "window_size": window, "valid_count": len(valid), "missing_count": missing,
        "nonfinite_or_invalid_count": invalid, "skipped_update_count": skipped,
        "mean": mean(value for _, value in valid) if valid else None,
        "max": maximum[1], "max_step": maximum[0],
        "rolling_mean": mean(value for _, value in recent) if recent else None,
        "rolling_max": max((value for _, value in recent), default=None),
    }


def estimate_training_timing(records, max_steps, warmup_steps=20, window=100):
    """Estimate remaining optimizer work from recent, post-warmup update timings.

    Only consecutive valid updates contribute. Intervals around evaluation log
    rows are excluded so a validation pause is not treated as model throughput.
    """
    result = {"seconds_per_step": None, "estimated_remaining_training_seconds": None,
              "estimation_scope": "remaining optimizer steps; excludes evaluation and early stopping"}
    by_step = {}
    for row in records:
        step, elapsed = first(row.get("step"), row.get("global_step")), row.get("elapsed_seconds")
        if (not finite(step) or step <= warmup_steps or not finite(elapsed) or elapsed < 0
                or not finite(row.get("loss")) or not finite(row.get("grad_norm"))
                or row.get("grad_norm", -1) < 0 or row.get("grad_norm_nonfinite")
                or first(row.get("step_skipped"), row.get("optimizer_step_skipped"), False)):
            continue
        by_step[int(step)] = {**row, "step": int(step)}
    recent = [by_step[step] for step in sorted(by_step)][-window:]
    intervals = []
    for previous, current in zip(recent, recent[1:]):
        delta = current["elapsed_seconds"] - previous["elapsed_seconds"]
        if (current["step"] - previous["step"] == 1 and delta > 0
                and "eval_loss" not in previous and "eval_loss" not in current):
            intervals.append(delta)
    if not intervals:
        return result
    result["seconds_per_step"] = mean(intervals)
    if finite(max_steps) and max_steps > 0:
        result["estimated_remaining_training_seconds"] = max(0, max_steps - recent[-1]["step"]) * result["seconds_per_step"]
    return result


def report_states(run_status, training, evaluation):
    stage = str(run_status.get("stage", run_status.get("status", "pending"))).lower()
    status = str(run_status.get("status", "pending")).lower()
    train_status = str(training.get("status", "")).lower()
    if train_status in {"completed", "succeeded", "passed", "success"}:
        train_status = "completed"
    elif train_status in {"failed", "error", "aborted"}:
        train_status = "failed"
    elif status == "failed" and stage not in {"validating", "evaluation", "reporting"}:
        train_status = "failed"
    elif stage in {"validating", "evaluation", "reporting", "completed"} or status == "completed":
        train_status = "completed" if training else "incomplete"
    elif status in {"preparing", "planned"} or stage in {"preparing", "preflight", "baseline"}:
        train_status = "preparing"
    elif stage in {"training", "train"} or status in {"running", "training"}:
        train_status = "running"
    else:
        train_status = "pending"
    eval_status = str(evaluation.get("status", "pending")).lower()
    if eval_status == "passed":
        coverage = evaluation.get("coverage", {})
        expected = first(evaluation.get("expected_rows"), coverage.get("expected_rows"))
        completed = first(evaluation.get("completed_rows"), coverage.get("completed_rows"))
        complete = (finite(expected) and expected > 0 and completed == expected
                    and coverage.get("missing_rows", 0) in (0, [])
                    and coverage.get("duplicate_rows", 0) in (0, [])
                    and evaluation.get("CC") == 0
                    and evaluation.get("validity_rate") == 1
                    and evaluation.get("unique_rate") == 1
                    and all(finite(row[field]) and 0 <= row[field] <= 1
                            for row in ranking_rows(evaluation) for field in ("HR", "NDCG"))
                    and all(evaluation.get("check_results", {}).get(name) in ("passed", True)
                            for name in ("provenance", "tokenizer_atomic_sid_tokens", "model_vocabulary",
                                         "prediction_coverage", "valid_sids", "unique_candidates")))
        eval_status = "passed" if complete else "incomplete"
    elif eval_status in {"failed", "error"}:
        eval_status = "failed"
    elif stage in {"validating", "evaluation"} and status not in {"failed", "completed"}:
        eval_status = "running"
    else:
        eval_status = "pending"
    return train_status, eval_status


def source_records(run_dir):
    result = []
    for filename in SOURCE_NAMES:
        path = Path(run_dir) / filename
        if path.is_file():
            data = path.read_bytes()
            result.append({"path": filename, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()})
    return result


def gpu_rows(records):
    result = []
    for record in records:
        children = record.get("gpus", [record])
        if not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            row = {**record, **child}
            row.pop("gpus", None)
            row["gpu_index"] = first(row.get("gpu_index"), row.get("index"), row.get("gpu"))
            row["memory_used_mib"] = first(row.get("memory_used_mib"), row.get("memory_used_mb"),
                                            row.get("memory.used"), row.get("memory_used"))
            row["utilization_percent"] = first(row.get("utilization_percent"), row.get("utilization_gpu_percent"),
                                                row.get("utilization_gpu_pct"),
                                                row.get("utilization.gpu"), row.get("utilization"))
            result.append(row)
    return result


def ranking_rows(evaluation):
    flat, nested = evaluation.get("metrics", {}), evaluation.get("metrics_by_k", {})
    return [{"k": k,
             "HR": first(flat.get(f"HR@{k}"), nested.get(str(k), {}).get("HR")),
             "NDCG": first(flat.get(f"NDCG@{k}"), nested.get(str(k), {}).get("NDCG"))}
            for k in TOP_K]


def verified_comparison_baselines(record, config, preparation, evaluation, evaluation_state):
    """Bind reference rankings to the exact training, test and SID identities."""
    result = {"status": "unavailable", "primary_k": 10, "reasons": [], "identity_checks": {}}
    if record.get("status") != "passed":
        result["reasons"].append("Completed popularity/uniform baselines are unavailable")
        return result
    data = config.get("data", {})
    hashes = data.get("sha256", {})
    observed = record.get("source_sha256", {})
    expected = {"train_file": hashes.get("train_file"), "sid_index_path": hashes.get("sid_index_path")}
    evaluated = evaluation.get("source_sha256", {})
    if evaluation_state == "passed":
        expected["test_file"] = evaluated.get("test_file")
    for name, value in expected.items():
        matched = (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                   and observed.get(name) == value)
        result["identity_checks"][name] = bool(matched)
        if not matched:
            result["reasons"].append(f"Missing or mismatched {name} SHA-256")
    for name in ("train_file", "sid_index_path"):
        if evaluated.get(name) is not None and evaluated[name] != observed.get(name):
            result["reasons"].append(f"Evaluation provenance disagrees with baseline {name}")
    if record.get("dataset") != data.get("category") or not data.get("category"):
        result["reasons"].append("Baseline dataset does not match the run configuration")
    expected_rows = preparation.get("rows", {})
    for key, expected_count in (("train_rows", expected_rows.get("train")),
                                ("catalog_items", preparation.get("items")),
                                ("test_rows", first(evaluation.get("expected_rows"), evaluation.get("coverage", {}).get("expected_rows")) if evaluation_state == "passed"
                                 else expected_rows.get("test"))):
        if expected_count is not None and record.get(key) != expected_count:
            result["reasons"].append(f"Baseline {key} disagrees with verified run metadata")
    for method in ("popularity", "uniform_random"):
        metrics = record.get(method, {}).get("metrics", {})
        if not all(finite(metrics.get(f"{metric}@{k}")) and 0 <= metrics[f"{metric}@{k}"] <= 1
                   for metric in ("HR", "NDCG") for k in TOP_K):
            result["reasons"].append(f"Incomplete or invalid {method} baseline metrics")
    if result["reasons"]:
        return result
    if evaluation_state != "passed":
        result.update(status="pending", reasons=["Awaiting accepted evaluation and its test-file identity"])
        return result
    result.update(status="available", dataset=record["dataset"], source_sha256=observed,
                  catalog_items=record.get("catalog_items"), train_rows=record.get("train_rows"),
                  test_rows=record.get("test_rows"),
                  metrics={method: record[method]["metrics"] for method in ("popularity", "uniform_random")})
    return result


def build_comparison(config, preparation, reference, baselines, evaluation, evaluation_state):
    data, parameters = config.get("data", {}), config.get("hyperparameters", {})
    actual = {"dataset": data.get("category"), "stage": "SFT only; RL not started",
              "backbone": config.get("model", {}).get("source"),
              "history_max": preparation.get("history_length", {}).get("maximum"),
              "sid_selected_epoch": preparation.get("selected_sid_epoch"),
              "catalog_items": preparation.get("items"),
              "train_rows": first(preparation.get("rows", {}).get("train"),
                                  config.get("task_counts", {}).get("sid_sequence_recommendation")),
              "test_rows": first(evaluation.get("expected_rows"), preparation.get("rows", {}).get("test")),
              "sft_task_examples": data.get("train_samples"),
              "learning_rate": parameters.get("learning_rate"), "max_epochs": parameters.get("max_epochs"),
              "lr_scheduler_type": parameters.get("lr_scheduler_type"), "gpus": parameters.get("world_size")}
    settings = reference.get("settings", {})
    paper_domains = ", ".join(row.get("dataset", "") for row in reference.get("results", []))
    rows = []
    for key, label, observed, original in (
            ("dataset", "数据集", actual["dataset"], paper_domains or None),
            ("backbone", "生成底模", actual["backbone"], reference.get("backbone")),
            ("stage", "训练阶段", actual["stage"], reference.get("stage")),
            ("history_max", "最大历史长度", actual["history_max"], settings.get("history_max")),
            ("sid_selection", "SID checkpoint / 训练预算", actual["sid_selected_epoch"], settings.get("rqvae_epochs")),
            ("lr_scheduler_type", "SFT 学习率调度", actual["lr_scheduler_type"], settings.get("lr_scheduler_type")),
            ("max_epochs", "SFT 最大 epoch", actual["max_epochs"], settings.get("max_epochs")),
            ("learning_rate", "SFT 学习率", actual["learning_rate"], settings.get("learning_rate"))):
        assessment = "unknown" if observed is None or original is None else "matched" if observed == original else "different"
        note = "selected quantizer epoch vs paper quantizer training budget" if key == "sid_selection" else ""
        rows.append({"key": key, "setting": label, "actual": observed, "paper": original,
                     "assessment": assessment, "note": note})
    return {"status": "available" if reference.get("results") else "unavailable",
            "paper": reference, "actual": actual, "comparability": rows, "directly_comparable": False,
            "limitation": "原论文没有 Video Games 结果；不同数据域、Base/Instruct、SFT/RL 阶段及 SID 等设置只可作背景参照，不能据此计算复现提升或判定复现成功。",
            "same_dataset_baselines": verified_comparison_baselines(
                baselines, config, preparation, evaluation, evaluation_state)}


def build_expectation_analysis(validation):
    result = {"status": "pending", "primary_k": 10,
              "interpretation": "Assess signal quality on this dataset; this is not paper reproduction or a statistical significance test."}
    if validation["status"] != "passed":
        result["reason"] = "Complete accepted evaluation is required before judging expectations"
        return result
    rows = validation["ranking"]
    tolerance = 1e-12
    checks = {f"{metric}_nondecreasing": all(a[metric] <= b[metric] + tolerance for a, b in zip(rows, rows[1:]))
              for metric in ("HR", "NDCG")}
    checks["NDCG_not_above_HR"] = all(row["NDCG"] <= row["HR"] + tolerance for row in rows)
    at_one = next(row for row in rows if row["k"] == 1)
    checks["HR1_equals_NDCG1"] = math.isclose(at_one["HR"], at_one["NDCG"], rel_tol=0, abs_tol=tolerance)
    result.update(status="complete", metric_consistency={"checks": checks, "passed": all(checks.values()), "observations": rows},
                  minimum_signal_checks_passed=None, signal_quality="baseline_unavailable")
    comparison = validation_chart_data(validation)
    if comparison["loss"]:
        before, after = comparison["loss"]
        result["validation_loss"] = {"before": before, "after": after, "difference": after - before,
                                     "decreased": after < before}
    baselines = validation["comparison"]["same_dataset_baselines"]
    if baselines["status"] != "available":
        result["baseline_comparison"] = {"status": "unavailable", "reasons": baselines["reasons"]}
        return result
    actual = next(row for row in rows if row["k"] == 10)
    compared = {"status": "available", "k": 10, "actual": {metric: actual[metric] for metric in ("HR", "NDCG")}}
    for method, metrics in baselines["metrics"].items():
        compared[method] = {metric: metrics[f"{metric}@10"] for metric in ("HR", "NDCG")}
        compared[method]["differences"] = {metric: actual[metric] - metrics[f"{metric}@10"] for metric in ("HR", "NDCG")}
        compared[method]["both_metrics_higher"] = all(delta > 0 for delta in compared[method]["differences"].values())
    result["baseline_comparison"] = compared
    criteria = [all(checks.values()), compared["popularity"]["both_metrics_higher"],
                compared["uniform_random"]["both_metrics_higher"]]
    result["ranking_signal_checks_passed"] = all(criteria)
    if "validation_loss" not in result:
        result["signal_quality"] = "loss_evidence_unavailable"
        return result
    criteria.append(result["validation_loss"]["decreased"])
    result["minimum_signal_checks_passed"] = all(criteria)
    result["signal_quality"] = "basic_signal_checks_pass" if all(criteria) else "mixed_or_weak_signal"
    return result


def verified_behavior_shift(record, config, evaluation, evaluation_state, evaluation_sha256):
    """Attach post-hoc diagnostics only when they identify this accepted evaluation."""
    result = {"status": "unavailable", "reasons": [], "identity_checks": {}}
    if record.get("status") != "passed":
        result["reasons"].append("Completed behavior-shift diagnostics are unavailable")
        return result
    if evaluation_state != "passed":
        result.update(status="pending", reasons=["Awaiting complete accepted evaluation"])
        return result
    if (not all(isinstance(record.get(key), dict) for key in ("source_sha256", "counts", "shift_summary", "alignment_coverage"))
            or not isinstance(record.get("strata"), list)
            or not all(isinstance(row, dict) for row in record["strata"])
            or not isinstance(record.get("prediction_diagnostics", {}), dict)):
        result["reasons"].append("Malformed diagnostic schema")
        return result
    if not all(isinstance(row.get("key"), str) and isinstance(row.get("label"), str) and isinstance(row.get("metrics"), dict)
               and all(isinstance(row.get(key, {}), dict) for key in ("metrics", "popularity", "uniform_random", "hits_by_k"))
               for row in record["strata"]):
        result["reasons"].append("Malformed diagnostic stratum schema")
        return result
    observed = record.get("source_sha256", {})
    configured = config.get("data", {}).get("sha256", {})
    evaluated = evaluation.get("source_sha256", {})
    expected = {"train_file": configured.get("train_file"), "sid_index_path": configured.get("sid_index_path"),
                "item_meta_path": configured.get("item_meta_path"), "test_file": evaluated.get("test_file"),
                "evaluation_metrics": evaluation_sha256}
    for key, value in expected.items():
        matched = (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                   and observed.get(key) == value)
        result["identity_checks"][key] = bool(matched)
        if not matched:
            result["reasons"].append(f"Missing or mismatched diagnostic {key} SHA-256")
    for key in ("train_file", "sid_index_path"):
        if evaluated.get(key) is not None and observed.get(key) != evaluated[key]:
            result["reasons"].append(f"Diagnostic disagrees with evaluated {key}")
    if not isinstance(observed.get("predictions"), str) or not re.fullmatch(r"[0-9a-f]{64}", observed["predictions"]):
        result["reasons"].append("Verified diagnostic prediction identity is missing")
    expected_rows = first(evaluation.get("expected_rows"), evaluation.get("coverage", {}).get("expected_rows"))
    if (record.get("overall_test_rows") != expected_rows
            or record.get("counts", {}).get("test_rows") != expected_rows):
        result["reasons"].append("Diagnostic test coverage differs from accepted evaluation")
    strata = record.get("strata", [])
    by_key = {row.get("key"): row for row in strata if isinstance(row, dict)}
    required = {"overall", "behavior_cold", "behavior_warm", "repeat_last", "not_repeat_last"}
    if not required <= by_key.keys() or len(by_key) != len(strata):
        result["reasons"].append("Diagnostic strata are missing or duplicated")
    if "overall" in by_key and (by_key["overall"].get("rows") != expected_rows or by_key["overall"].get("share") != 1):
        result["reasons"].append("Diagnostic overall stratum does not cover the accepted test rows")
    for row in strata:
        count = row.get("rows")
        valid_count = isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= (expected_rows or 0)
        if not valid_count or not finite(row.get("share")) or not math.isclose(row["share"], count / expected_rows, abs_tol=1e-12):
            result["reasons"].append(f"Invalid diagnostic stratum coverage: {row.get('key')}")
            continue
        if count and not all(finite(row.get("metrics", {}).get(f"{metric}@{k}"))
                             and 0 <= row["metrics"][f"{metric}@{k}"] <= 1
                             for metric in ("HR", "NDCG") for k in TOP_K):
            result["reasons"].append(f"Invalid diagnostic metrics: {row.get('key')}")
    if not result["reasons"]:
        for left, right in (("behavior_cold", "behavior_warm"), ("repeat_last", "not_repeat_last")):
            if by_key[left]["rows"] + by_key[right]["rows"] != expected_rows:
                result["reasons"].append(f"Diagnostic partition does not cover all rows: {left}/{right}")
        for metric in ("HR", "NDCG"):
            for k in TOP_K:
                key = f"{metric}@{k}"
                actual = evaluation.get("metrics", {}).get(key)
                overall = by_key["overall"].get("metrics", {}).get(key)
                reconstructed = sum(by_key[group]["rows"] * by_key[group].get("metrics", {}).get(key, 0)
                                    for group in ("behavior_cold", "behavior_warm") if by_key[group]["rows"]) / expected_rows
                if not finite(actual) or not finite(overall) or not math.isclose(actual, overall, abs_tol=1e-12) or not math.isclose(actual, reconstructed, abs_tol=1e-12):
                    result["reasons"].append(f"Conditional metrics do not reconstruct accepted {key}")
    if result["reasons"]:
        return result
    return {**record, "status": "available", "reasons": [], "identity_checks": result["identity_checks"],
            "scope": "post-hoc exploratory strata on the original full-catalog predictions; no candidate filtering",
            "prediction_integrity_scope": "Diagnostic utility verified prediction bytes; renderer binds its stored hash and exact evaluation-metrics identity without rehashing prediction data"}


def build_reports(run_dir):
    run_dir = Path(run_dir).resolve()
    issues = []
    status = read_json(run_dir / "run_status.json", issues)
    config = read_json(run_dir / "run_config.json", issues)
    summary = read_json(run_dir / "training_summary.json", issues)
    monitor = read_json(run_dir / "monitor/summary.json", issues)
    baseline = read_json(run_dir / "baseline_validation.json", issues)
    evaluation = read_json(run_dir / "evaluation/metrics.json", issues)
    reference = read_json(PAPER_REFERENCE_PATH, issues)
    baselines = read_json(run_dir / "comparison_baselines.json", issues)
    shift_record = read_json(run_dir / "behavior_shift_analysis.json", issues)
    train_file = config.get("data", {}).get("train_file")
    preparation_path = Path(train_file).parent / "preparation_manifest.json" if train_file else None
    preparation = read_json(preparation_path, issues) if preparation_path else {}
    steps = merged_steps(read_jsonl(run_dir / "monitor/metrics.jsonl", issues))
    gpu = gpu_rows(read_jsonl(run_dir / "gpu_metrics.jsonl", issues))
    events = read_jsonl(run_dir / "monitor/events.jsonl", issues)
    window = first(monitor.get("monitor_window"), monitor.get("window_size"), config.get("monitor_window"),
                   config.get("hyperparameters", {}).get("monitor_window"), 100)
    window = int(window) if finite(window) and window > 0 else 100
    gradients = gradient_statistics(steps, window)
    training_state, evaluation_state = report_states(status, summary, evaluation)
    if training_state == "completed" and not gradients["valid_count"]:
        training_state = "incomplete"
        issues.append("Completed training summary has no valid per-update gradient observations")
    timing = estimate_training_timing(
        steps, monitor.get("max_steps"),
        warmup_steps=config.get("hyperparameters", {}).get("warmup_steps", 20))
    if training_state == "completed":
        timing["estimated_remaining_training_seconds"] = 0.0
    timestamp = datetime.now(timezone.utc)
    common = {"generated_at_utc": timestamp.isoformat(),
              "generated_at_shanghai": timestamp.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
              "run_directory": str(run_dir), "sources": source_records(run_dir),
              "ingestion_issues": issues, "run_status": status,
              "rl_training_started": False}
    for path in (PAPER_REFERENCE_PATH, preparation_path):
        if path and path.is_file():
            content = path.read_bytes()
            common["sources"].append({"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    training = {**common, "report": "sft_training", "status": training_state,
                "config": config, "training_summary": summary, "monitor_summary": monitor,
                "grad_norm": gradients, "optimizer_log_rows": len(steps), "timing": timing,
                "baseline_validation": baseline, "events": events,
                "gpu_samples": len(gpu),
                "gpu_peak_memory_mib": max((r["memory_used_mib"] for r in gpu
                                             if finite(r.get("memory_used_mib"))), default=None),
                "gpu_mean_utilization_percent": mean([r["utilization_percent"] for r in gpu
                                                       if finite(r.get("utilization_percent"))])
                if any(finite(r.get("utilization_percent")) for r in gpu) else None}
    validation = {**common, "report": "sft_validation", "status": evaluation_state,
                  "evaluation": evaluation, "ranking": ranking_rows(evaluation),
                  "baseline_validation": baseline,
                  "comparison": build_comparison(config, preparation, reference, baselines, evaluation, evaluation_state),
                  "behavior_shift_analysis": verified_behavior_shift(
                      shift_record, config, evaluation, evaluation_state,
                      hashlib.sha256((run_dir / "evaluation/metrics.json").read_bytes()).hexdigest()
                      if (run_dir / "evaluation/metrics.json").is_file() else None),
                  "protocol": {"split": "full original test split; no row deduplication",
                               "checkpoint_selection": "validation loss only; test unused for selection",
                               "shared_catalog": True,
                               "validity_interpretation": "catalog-constrained decoding validity is a structural check, not recommendation accuracy",
                               "perplexity_interpretation": "masked answer-token perplexity; not general-language perplexity"}}
    validation["expectation_analysis"] = build_expectation_analysis(validation)
    validation["expectation_analysis"]["behavior_shift_status"] = validation["behavior_shift_analysis"]["status"]
    if validation["behavior_shift_analysis"]["status"] == "available":
        validation["expectation_analysis"]["conclusion_scope"] = "Optimization and numerical integrity only; behavior-cold recommendation quality is assessed separately using post-hoc strata"
    return training, validation, steps, gpu


def fmt(value, percent=False, precision=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if percent and finite(value):
        return f"{value * 100:.{precision}f}%"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(clean_json(value), ensure_ascii=False)
    return str(value)


def validation_chart_data(report):
    """Only complete, accepted evaluation results belong in final-result charts."""
    observations = {"ranking": [], "loss": [], "masked_perplexity": []}
    if report["status"] != "passed":
        return observations
    observations["ranking"] = report["ranking"]
    baseline = report["baseline_validation"]
    evaluation = report["evaluation"]
    final = evaluation.get("validation", {})
    for name, alias in (("loss", "eval_loss_answer_tokens"), ("masked_perplexity", "masked_ppl")):
        values = [first(baseline.get(name), baseline.get(alias)),
                  first(final.get(name), evaluation.get(alias))]
        if all(finite(value) and value >= 0 for value in values):
            if name != "masked_perplexity" or all(value > 0 for value in values):
                observations[name] = values
    return observations


def validation_result_summary(report):
    if report["status"] != "passed":
        state = STATUS_LABELS.get(report["status"], report["status"])
        return [f"完整结果验证尚未通过（{state}），最终结果将在全部测试样本和模型检查通过后列出。"]
    evaluation = report["evaluation"]
    coverage = evaluation.get("coverage", {})
    rows = first(evaluation.get("completed_rows"), coverage.get("completed_rows"))
    at_ten = next((row for row in report["ranking"] if row["k"] == 10), {})
    lines = [f"完整测试集已通过验证，共 {fmt(rows)} 条样本；"
             f"HR@10 为 {fmt(at_ten.get('HR'), True)}，NDCG@10 为 {fmt(at_ten.get('NDCG'), True)}。"]
    comparison = validation_chart_data(report)
    if comparison["loss"]:
        before, after = comparison["loss"]
        change = f"，变化 {(after - before) / before:+.2%}" if before > 0 else ""
        lines.append(f"最佳模型的答案 token loss：{fmt(before)} → {fmt(after)}{change}。")
    if comparison["masked_perplexity"]:
        before, after = comparison["masked_perplexity"]
        lines.append(f"答案 token masked perplexity：{fmt(before)} → {fmt(after)}。")
    return lines


def paper_comparison_rows(report):
    comparison = report["comparison"]
    actual = comparison["actual"]
    rows = [{"source": "Current experiment", "dataset": actual.get("dataset"), "stage": actual["stage"],
             "k": row["k"], "HR": row["HR"] if report["status"] == "passed" else None,
             "NDCG": row["NDCG"] if report["status"] == "passed" else None} for row in report["ranking"]]
    paper = comparison["paper"]
    for domain in paper.get("results", []):
        for k in paper.get("reported_k", []):
            rows.append({"source": "Paper Table 1; reference only", "dataset": domain["dataset"],
                         "stage": paper.get("stage"), "k": k,
                         "HR": domain.get("HR", {}).get(str(k)), "NDCG": domain.get("NDCG", {}).get(str(k))})
    return rows


def expectation_summary(report):
    analysis = report["expectation_analysis"]
    if analysis["status"] == "pending":
        return ["完整评估尚未通过，暂不判断是否符合预期。预先指定 K=10 进行同数据集 POP / 均匀随机基线对照。"]
    lines = ["评估曲线一致性检查：" + ("通过。" if analysis["metric_consistency"]["passed"] else "存在异常，需检查累计排名指标。")]
    comparison = analysis.get("baseline_comparison", {})
    if comparison.get("status") == "available":
        actual = comparison["actual"]
        lines.append(f"预先指定 K=10：模型 HR={fmt(actual['HR'], True)} / NDCG={fmt(actual['NDCG'], True)}；"
                     f"POP={fmt(comparison['popularity']['HR'], True)} / {fmt(comparison['popularity']['NDCG'], True)}；"
                     f"均匀随机期望={fmt(comparison['uniform_random']['HR'], True)} / {fmt(comparison['uniform_random']['NDCG'], True)}。")
        if analysis["minimum_signal_checks_passed"] is None:
            lines.append("缺少可比的训练前后答案 token loss，暂不作完整学习信号判断。")
        else:
            passed_message = ("优化与评估检查通过；这不表示行为冷组推荐能力通过验证，能力结论须结合下方事后分层结果。"
                              if report.get("behavior_shift_analysis", {}).get("status") == "available" else
                              "基本学习与评估检查符合预期；推荐竞争力和论文复现仍无法判定。")
            lines.append(passed_message if analysis["minimum_signal_checks_passed"] else
                         "基本信号检查未全部通过：模型可能落后于同数据集基线、曲线存在异常或验证 loss 未下降，应进一步诊断。")
    else:
        lines.append("同数据集基线无法完成来源核验，暂不作基线强弱结论：" + "; ".join(comparison.get("reasons", [])))
    lines.append("超过 POP / 随机仅是最低学习信号检查；所有差值均为描述性比较，不代表统计显著或论文复现成功。")
    return lines


def behavior_shift_summary(report):
    diagnostic = report.get("behavior_shift_analysis", {})
    if diagnostic.get("status") != "available":
        return ["行为迁移与事后分层诊断尚未完成来源核验；不改变已验收的总体指标。"]
    shift = diagnostic["shift_summary"]
    groups = {row["key"]: row for row in diagnostic["strata"]}
    cold, warm = groups["behavior_cold"], groups["behavior_warm"]
    alignment = diagnostic.get("alignment_coverage", {})
    lines = [
        "本节是事后探索性分层，沿用同一批完整目录约束 beam=50 预测，不筛选候选目录、不重排、不改变总体指标或预先指定的 K=10。",
        f"测试目标在训练行为（历史及目标）中未出现的比例为 {fmt(shift.get('test_behavior_cold_rate'), True)}；"
        f"重复上一商品的比例由训练 {fmt(shift.get('train_repeat_last_rate'), True)} 变为测试 {fmt(shift.get('test_repeat_last_rate'), True)}。",
        f"行为冷组 {cold['rows']} 条：HR@10={fmt(cold['metrics'].get('HR@10'), True)}，HR@50={fmt(cold['metrics'].get('HR@50'), True, 5)}；"
        f"行为暖组 {warm['rows']} 条：HR@10={fmt(warm['metrics'].get('HR@10'), True)}。总体值是这些组按原行数加权的混合，不能替代各组能力结论。",
        f"行为冷不等于模型完全未见商品：{fmt(alignment.get('cold_unique_items'))} 个行为冷商品中，"
        f"SID→标题对齐覆盖 {fmt(alignment.get('sid_to_title_items'))} 个，标题→SID 覆盖 {fmt(alignment.get('title_to_sid_items'))} 个。",
        "全局 POP 在行为冷目标上缺少行为支持，时间和行为分布迁移也可能削弱其表现；超过 POP 的倍数不是推荐竞争力或泛化质量的证明，不能把差异单独归因于时间迁移。",
        "分层是描述性诊断，尚无匹配的强序列基线、训练前同协议排名对照或多种子置信区间，不能作因果或显著性结论。",
    ]
    predictions = diagnostic.get("prediction_diagnostics", {})
    if predictions:
        lines.insert(4, f"模型 Top1 复制上一商品的比例为 {fmt(predictions.get('top1_equals_last_rate'), True)}，"
                     f"真实重复末项比例为 {fmt(shift.get('test_repeat_last_rate'), True)}；复制预测的正确率为 {fmt(predictions.get('copy_precision'), True)}，"
                     f"其贡献了全部 Top1 命中的 {fmt(predictions.get('copy_share_of_all_top1_hits'), True)}。")
        lines.insert(5, f"行为冷商品仅占全部生成候选位置的 {fmt(predictions.get('cold_candidate_rate'), True, 5)}，"
                     f"候选中覆盖 {fmt(predictions.get('cold_predicted_unique_items'))} 个不同行为冷商品。目录可生成性不等于实际召回覆盖。")
    return lines


def behavior_shift_rows(report):
    diagnostic = report.get("behavior_shift_analysis", {})
    if diagnostic.get("status") != "available":
        return []
    return [{"stratum": group["key"], "label": group["label"], "rows": group["rows"], "share": group["share"], "k": k,
             "HR": group.get("metrics", {}).get(f"HR@{k}"), "NDCG": group.get("metrics", {}).get(f"NDCG@{k}"),
             "hits": group.get("hits_by_k", {}).get(str(k)),
             "POP_HR": group.get("popularity", {}).get(f"HR@{k}"), "POP_NDCG": group.get("popularity", {}).get(f"NDCG@{k}"),
             "uniform_HR": group.get("uniform_random", {}).get(f"HR@{k}"), "uniform_NDCG": group.get("uniform_random", {}).get(f"NDCG@{k}")}
            for group in diagnostic["strata"] for k in TOP_K]


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join(
        f"<th>{html.escape(str(item))}</th>" for item in headers
    ) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(
        f"<td>{html.escape(fmt(cell))}</td>" for cell in row
    ) + "</tr>" for row in rows) + "</tbody></table></div>"


def links(pdf, raw_json, csv_name):
    return f"<p><a class='button' href='{pdf}'>下载 PDF</a><a class='button' href='{raw_json}'>报告 JSON</a><a class='button' href='{csv_name}'>指标 CSV</a><a class='button' href='../index.html'>报告合集</a></p>"


def document(title, report, content):
    state = report["status"]
    badge = "badge failed" if state in {"failed", "incomplete"} else "badge"
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
            f"<div class='eyebrow'>MINIONEREC · VIDEO GAMES · QWEN2.5-1.5B BASE</div><h1>{title}</h1>"
            f"<p><span class='{badge}'>{STATUS_LABELS.get(state, html.escape(state))}</span> "
            f"<span class='muted'>更新于 {report['generated_at_shanghai']}（北京时间）</span></p>"
            + content + "<footer>仅执行 SFT 与结果验证，未启动 RL。报告依据实际日志生成；空值表示尚无观测，不能解释为 0。</footer></main></body></html>")


def training_html(report):
    grad = report["grad_norm"]
    summary = report["training_summary"]
    content = links("training-report.pdf", "training_report.json", "metrics.csv")
    if report["status"] != "completed":
        content += "<p class='warning'>本次训练尚未确认完成。当前曲线和统计只覆盖已记录的更新步，不能作为最终训练结果。</p>"
    content += "<div class='kpis'>" + "".join(
        f"<div class='card'>{label}<b>{html.escape(fmt(value))}</b></div>" for label, value in (
            ("gradNorm 全程均值", grad["mean"]), ("gradNorm 全程最大值", grad["max"]),
            ("最大值所在 step", grad["max_step"]), (f"最近 {grad['window_size']} 有效步均值", grad["rolling_mean"]),
            (f"最近 {grad['window_size']} 有效步最大值", grad["rolling_max"]), ("有效 gradNorm 记录数", grad["valid_count"]))) + "</div>"
    content += "<section><h2>训练与监控口径</h2><p>gradNorm 为去除 loss scale、裁剪前的全模型全局 L2 范数；不对四个 rank 的范数再次平均。窗口统计仅包含有限且未跳过的有效更新步，范数大于裁剪阈值 1 不等于训练失败。</p>"
    content += table(["项目", "实际记录"], [(key, value) for key, value in (
        ("训练状态", report["status"]), ("最佳 checkpoint", first(summary.get("best_model_checkpoint"), summary.get("best_checkpoint"))),
        ("最佳验证 loss", first(summary.get("best_metric"), summary.get("best_eval_loss"))),
        ("停止原因", summary.get("stop_reason")), ("缺少 gradNorm 的更新步", grad["missing_count"]),
        ("非有限或无效 gradNorm", grad["nonfinite_or_invalid_count"]), ("跳过更新次数", grad["skipped_update_count"]),
        ("GPU 峰值显存 MiB", report["gpu_peak_memory_mib"]),
        ("GPU 平均利用率", fmt(report["gpu_mean_utilization_percent"])),
        ("最近有效更新步的秒 / step", report["timing"]["seconds_per_step"]),
        ("预计剩余训练小时", report["timing"]["estimated_remaining_training_seconds"] / 3600
         if finite(report["timing"]["estimated_remaining_training_seconds"]) else None),
    )]) + "</section>"
    content += "<p class='muted'>时间估计跳过 warmup，并使用最近最多 100 个有效更新步；不足两个 warmup 后样本时暂不估计。不包含后续验证与评估耗时，早停可能使训练提前结束。</p>"
    content += "<section><h2>实际运行配置及任务数量</h2>" + table(
        ["键", "值"], report["config"].items()) + "</section>"
    for title, filename in (("训练 / 验证 loss", "loss.png"), ("gradNorm 与窗口统计", "grad_norm.png"),
                            ("学习率", "learning_rate.png"), ("GPU 资源", "gpu_resources.png")):
        content += f"<section><h2>{title}</h2><img src='{filename}' alt='{title}'></section>"
    if report["events"]:
        content += "<section><h2>监控事件</h2>" + table(["事件"], [(event,) for event in report["events"][-30:]]) + "</section>"
    content += "<section><h2>证据与可追溯性</h2>" + table(["源文件", "字节", "SHA-256"],
        [(r["path"], r["bytes"], r["sha256"]) for r in report["sources"]]) + "</section>"
    return document("SFT 训练报告", report, content)


def validation_html(report):
    evaluation = report["evaluation"]
    coverage = evaluation.get("coverage", {})
    valid = evaluation.get("validation", {})
    baseline = report["baseline_validation"]
    content = links("validation-report.pdf", "validation_report.json", "metrics.csv")
    if report["status"] != "passed":
        content += "<p class='warning'>完整结果验证尚未通过。已有部分指标不代表完整测试集结果；未开始、失败及数据缺失均会保留实际状态。</p>"
    content += "<section><h2>结果概览</h2>" + "".join(
        "<p>" + html.escape(line) + "</p>" for line in validation_result_summary(report)) + "</section>"
    content += "<section><h2>完整性与生成检查</h2>" + table(["检查", "实际值"], [
        ("预期测试样本", first(evaluation.get("expected_rows"), coverage.get("expected_rows"))),
        ("已完成测试样本", first(evaluation.get("completed_rows"), coverage.get("completed_rows"))),
        ("缺失 / 重复处理行", [coverage.get("missing_rows"), coverage.get("duplicate_rows")]),
        ("SID 有效率", fmt(evaluation.get("validity_rate"), percent=True)),
        ("每条候选唯一率", fmt(evaluation.get("unique_rate"), percent=True)),
        ("无效候选数 CC", evaluation.get("CC")),
    ]) + table(["模型 / 来源检查", "状态"], evaluation.get("check_results", {}).items()) + "</section>"
    content += "<section><h2>推荐效果</h2>" + table(["K", "HR@K", "NDCG@K"],
        [(row["k"], fmt(row["HR"], percent=True), fmt(row["NDCG"], percent=True)) for row in report["ranking"]])
    content += "<p class='muted'>每条测试行只有一个目标商品，因此本协议下 HR@K 等于 Recall@K。指标按原测试行计算，保留原重复交互及其权重；测试集不参与最优 checkpoint 选择。SID 合法性由候选目录约束保证，不等同于模型推荐准确率。</p></section>"
    content += "<section><h2>HR / NDCG 随 K 的变化</h2><img src='ranking.png' alt='完整测试集 HR 与 NDCG 随 K 的变化'><p class='muted'>仅在完整结果验证通过后绘制最终指标；尚未通过时显示实际状态。</p></section>"
    content += "<section><h2>训练前后验证集对照</h2>" + table(
        ["阶段", "仅答案 token loss", "masked perplexity", "答案 token 数"], [
            ("训练前", first(baseline.get("loss"), baseline.get("eval_loss_answer_tokens")),
             first(baseline.get("masked_perplexity"), baseline.get("masked_ppl")),
             first(baseline.get("answer_tokens"), baseline.get("answer_token_count"))),
            ("最终最佳模型", first(valid.get("loss"), evaluation.get("eval_loss_answer_tokens")),
             first(valid.get("masked_perplexity"), evaluation.get("masked_ppl")), valid.get("answer_tokens")),
        ]) + "<p class='muted'>prompt 与 padding 标签被 mask，只统计答案 token；这里的 perplexity 不是通用语言困惑度。</p></section>"
    content += "<section><h2>训练前后验证指标图</h2><img src='validation-comparison.png' alt='训练前与最佳模型的答案 token loss 及 masked perplexity 对比'><p class='muted'>loss 使用线性坐标；masked perplexity 单独使用对数坐标，柱上标注实际数值。完整验证通过且前后观测齐全时绘制。</p></section>"
    comparison = report["comparison"]
    paper = comparison["paper"]
    source = paper.get("source", {})
    content += "<section><h2>与原论文的可比性</h2><p class='warning'>" + html.escape(comparison["limitation"]) + "</p>"
    content += "<p>来源：<a href='" + html.escape(source.get("url", ""), quote=True) + "'>" + html.escape(source.get("title", "Paper reference unavailable")) + "</a>；" + html.escape(fmt(source.get("version"))) + "，" + html.escape(fmt(source.get("locators"))) + "。</p>"
    content += "<p>" + html.escape(paper.get("backbone_size_note", "")) + "</p>"
    content += table(["设置", "本次实际配置", "原论文", "关系", "说明"],
                     [(row["setting"], row["actual"], row["paper"], row["assessment"], row["note"])
                      for row in comparison["comparability"]]) + "</section>"
    content += "<section><h2>实际结果与论文参考值</h2>" + table(
        ["来源", "数据域", "阶段", "K", "HR@K", "NDCG@K"],
        [(row["source"], row["dataset"], row["stage"], row["k"], fmt(row["HR"], True), fmt(row["NDCG"], True))
         for row in paper_comparison_rows(report)])
    content += "<p class='muted'>论文只发表 K=3、5、10；不补造或插值 K=1、20、50，也不计算跨数据域提升百分比。</p><img src='paper-reference.png' alt='不同数据域和训练阶段的 HR 与 NDCG 背景参考'><p class='muted'>虚线为论文 Industrial / Office 完整训练流程；实线为本次 Video Games SFT。不同数据域与阶段，只作参考。</p></section>"
    content += "<p class='muted'>本次四类任务的配比见训练配置，论文另包含描述 / 偏好等对齐任务。不同任务、tokenizer、mask 和加权口径下的 loss 不设跨论文数值目标；训练前后只比较本次最终验证的同口径答案 token 加权 loss。</p>"
    content += "<section><h2>是否符合预期</h2>" + "".join("<p>" + html.escape(line) + "</p>" for line in expectation_summary(report))
    analysis = report["expectation_analysis"]
    if analysis.get("metric_consistency"):
        content += table(["数值诊断", "通过"], analysis["metric_consistency"]["checks"].items())
    content += "<p><a href='comparison.json'>对比 JSON</a> · <a href='paper_comparison.csv'>参考指标 CSV</a> · <a href='comparability.csv'>配置对照 CSV</a> · <a href='expectation_analysis.json'>预期分析 JSON</a> · <a href='baseline_metrics.csv'>已核验基线 CSV</a></p></section>"
    content += "<section><h2>行为迁移与条件召回（事后诊断）</h2>" + "".join(
        "<p>" + html.escape(line) + "</p>" for line in behavior_shift_summary(report))
    diagnostic = report.get("behavior_shift_analysis", {})
    if diagnostic.get("status") == "available":
        content += table(["分层", "原始行数", "占比", "模型 HR@1", "模型 HR@10", "模型 NDCG@10", "模型 HR@50", "POP HR@10"],
                         [(group["label"], group["rows"], fmt(group["share"], True),
                           fmt(group["metrics"].get("HR@1"), True), fmt(group["metrics"].get("HR@10"), True),
                           fmt(group["metrics"].get("NDCG@10"), True), fmt(group["metrics"].get("HR@50"), True),
                           fmt(group.get("popularity", {}).get("HR@10"), True)) for group in diagnostic["strata"]])
        content += "<p class='muted'>行为冷 / 暖互斥，重复末项 / 不重复末项互斥；不同分层维度会重叠，不能把表中所有行数相加。行为冷指训练行为历史及目标中均未出现，商品 SID 与文字对齐仍可见。</p>"
        content += "<p><a href='behavior_shift_analysis.json'>完整分层诊断 JSON</a> · <a href='behavior_shift_metrics.csv'>分层指标 CSV</a></p>"
    content += "</section>"
    content += "<section><h2>评估设定</h2><p>使用训练时相同的推荐提示、确定性 beam search 和完整 SID 目录。商品元数据 / SID 对齐任务采用共享已知商品目录；不能据此声称商品目录在训练与测试之间完全隔离，也不宣称时间戳边界严格无重叠。</p></section>"
    examples = evaluation.get("examples", {})
    for key, title in (("hit", "命中实例"), ("miss", "未命中实例")):
        if examples.get(key):
            content += f"<section><h2>{title}</h2>" + table(["原始行", "目标 SID", "排名", "前 10 个预测"],
                [(r.get("row_id"), r.get("target"), r.get("rank"), r.get("predictions", [])[:10])
                 for r in examples[key][:5]]) + "</section>"
    if evaluation.get("error"):
        content += "<section><h2>验证错误</h2><p>" + html.escape(str(evaluation["error"])) + "</p></section>"
    content += "<section><h2>证据与可追溯性</h2>" + table(["源文件", "字节", "SHA-256"],
        [(r["path"], r["bytes"], r["sha256"]) for r in report["sources"]]) + "</section>"
    return document("SFT 结果验证报告", report, content)


def write_csv(path, rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key)
                         for key in fields})
    atomic_text(path, output.getvalue())


def render_charts(training, validation, steps, gpu, training_dir, validation_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                               "axes.labelcolor": "#172d35", "text.color": "#172d35",
                               "axes.edgecolor": "#cedde1", "grid.color": "#e3ebee",
                               "figure.facecolor": "white", "savefig.facecolor": "white"})
    for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                 Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break

    figures = []

    def save_figure(fig, filename, directory, collection):
        fig.savefig(directory / filename, dpi=160)
        collection.append(fig)

    def figure(title, filename, series, ylabel):
        fig, ax = plt.subplots(figsize=(10.8, 4.6), layout="constrained")
        found = False
        for label, points, color in series:
            points = [(x, y) for x, y in points if finite(x) and finite(y)]
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], label=label,
                        color=color, linewidth=1.4)
                found = True
        if found:
            ax.legend(loc="best")
        else:
            ax.text(.5, .5, "No observations yet", transform=ax.transAxes, ha="center", va="center")
        ax.set(title=title, xlabel="Optimizer update step", ylabel=ylabel)
        ax.grid(alpha=.6)
        save_figure(fig, filename, training_dir, figures)
        return fig

    figure("Training and validation loss", "loss.png", [
        ("Training", [(r["step"], r.get("loss")) for r in steps], "#167d8d"),
        ("Validation", [(r["step"], r.get("eval_loss")) for r in steps], "#c17b2d")], "Loss")
    points = [(r["step"], r.get("grad_norm")) for r in steps
              if finite(r.get("grad_norm")) and not first(r.get("step_skipped"), r.get("optimizer_step_skipped"), False)]
    window = training["grad_norm"]["window_size"]
    rolling = [(p[0], mean(v for _, v in points[max(0, i-window+1):i+1])) for i, p in enumerate(points)]
    maximum = [(p[0], max(v for _, v in points[max(0, i-window+1):i+1])) for i, p in enumerate(points)]
    figure("Global gradient norm before clipping", "grad_norm.png", [
        ("gradNorm", points, "#9cbec5"), (f"Mean, last {window} valid updates", rolling, "#167d8d"),
        (f"Max, last {window} valid updates", maximum, "#c17b2d")], "Global L2 norm")
    figure("Learning rate", "learning_rate.png", [
        ("Learning rate", [(r["step"], first(r.get("learning_rate"), r.get("lr"))) for r in steps], "#167d8d")], "Learning rate")
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), layout="constrained")
    indexes = sorted({str(r.get("gpu_index", "unknown")) for r in gpu})
    for index in indexes:
        samples = [r for r in gpu if str(r.get("gpu_index", "unknown")) == index]
        for ax, field, label in ((axes[0], "memory_used_mib", "GPU memory (MiB)"),
                                 (axes[1], "utilization_percent", "GPU utilization (%)")):
            values = [(n, r.get(field)) for n, r in enumerate(samples) if finite(r.get(field))]
            if values:
                ax.plot([x for x, _ in values], [y for _, y in values], label=f"GPU {index}")
            ax.set(ylabel=label, xlabel="Resource sample (30 second cadence)")
            ax.grid(alpha=.6)
    if gpu:
        for ax in axes:
            ax.legend(loc="best")
    else:
        for ax in axes:
            ax.text(.5, .5, "No GPU observations yet", transform=ax.transAxes, ha="center")
    fig.suptitle("Four-GPU resource monitoring")
    fig.savefig(training_dir / "gpu_resources.png", dpi=160)
    figures.append(fig)

    def text_page(title, lines):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.08, .94, title, fontsize=20, weight="bold", color="#167d8d", va="top")
        y = .88
        for line in lines:
            import textwrap
            for wrapped in textwrap.wrap(str(line), width=95, break_long_words=True) or [""]:
                fig.text(.08, y, wrapped, fontsize=10, va="top")
                y -= .024
        return fig

    grad, summary = training["grad_norm"], training["training_summary"]
    config = training["config"]
    parameters, data = config.get("hyperparameters", {}), config.get("data", {})
    cover = text_page("SFT Training Report", [
        "MiniOneRec / Video Games / Qwen2.5-1.5B Base", f"Status: {training['status']}",
        f"Updated (Asia/Shanghai): {training['generated_at_shanghai']}", "",
        f"Run: {training['run_directory']}",
        f"Samples: train={data.get('train_samples')}, validation={data.get('validation_samples')}",
        f"Task counts: {fmt(config.get('task_counts'))}",
        f"GPUs={parameters.get('world_size')}; micro batch={parameters.get('micro_batch_size')}; accumulation={parameters.get('gradient_accumulation_steps')}; global batch={parameters.get('global_batch_size')}",
        f"Epoch limit={parameters.get('max_epochs')}; learning rate={parameters.get('learning_rate')}; warmup={parameters.get('warmup_steps')}; cutoff={parameters.get('cutoff_len')}; seed={parameters.get('seed')}",
        f"BF16={parameters.get('bf16')}; attention={parameters.get('attn_implementation')}; gradient checkpointing={parameters.get('gradient_checkpointing')}",
        f"Best checkpoint: {first(summary.get('best_model_checkpoint'), summary.get('best_checkpoint'))}",
        f"Best validation loss: {first(summary.get('best_metric'), summary.get('best_eval_loss'))}",
        f"Stop reason: {summary.get('stop_reason')}", "",
        "Global gradient L2 norm: unscaled and before clipping; no rank averaging.",
        f"Valid updates: {grad['valid_count']} / missing: {grad['missing_count']} / skipped: {grad['skipped_update_count']}",
        f"Global mean: {fmt(grad['mean'])}; maximum: {fmt(grad['max'])} at step {fmt(grad['max_step'])}",
        f"Recent {grad['window_size']} mean: {fmt(grad['rolling_mean'])}; max: {fmt(grad['rolling_max'])}",
        f"Nonfinite/invalid norm observations: {grad['nonfinite_or_invalid_count']}", "",
        f"Recent seconds/update: {fmt(training['timing']['seconds_per_step'])}; remaining training hours: {fmt(training['timing']['estimated_remaining_training_seconds'] / 3600 if finite(training['timing']['estimated_remaining_training_seconds']) else None)}",
        "Estimate excludes warmup and evaluation; early stopping may finish sooner.",
        "This report only describes recorded observations. Missing values are not zero.",
        "RL was not started. Completion is never inferred merely from file existence.",
    ])
    with PdfPages(training_dir / "training-report.pdf") as pdf:
        pdf.savefig(cover)
        for fig in figures:
            pdf.savefig(fig)
    plt.close(cover)
    for fig in figures:
        plt.close(fig)

    evaluation = validation["evaluation"]
    baseline = validation["baseline_validation"]
    final = evaluation.get("validation", {})
    observations = validation_chart_data(validation)
    validation_figures = []
    pending_message = (f"Final evaluation not accepted\nStatus: {validation['status']}\n"
                       "Awaiting complete validation.")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    baselines = validation["comparison"]["same_dataset_baselines"]
    for ax, metric in zip(axes, ("HR", "NDCG")):
        values = [(row["k"], row.get(metric)) for row in observations["ranking"]
                  if finite(row.get(metric))]
        if values:
            ax.plot([k for k, _ in values], [value for _, value in values],
                    label="Current model", marker="o", linewidth=1.8, color="#167d8d")
            if baselines["status"] == "available":
                for method, label, color in (("popularity", "POP (train only)", "#c17b2d"),
                                             ("uniform_random", "Uniform random expectation", "#7c6b96")):
                    ax.plot(TOP_K, [baselines["metrics"][method][f"{metric}@{k}"] for k in TOP_K],
                            label=label, linestyle="--", marker=".", linewidth=1.2, color=color)
            ax.legend(loc="best", fontsize=8)
            largest = max(max(line.get_ydata()) for line in ax.lines)
            ax.set_ylim(0, min(1, max(.05, largest * 1.15)))
        else:
            ax.text(.5, .5, pending_message, transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(0, 1)
        ax.set(title=metric, xlabel="K", ylabel=metric, xticks=TOP_K)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.6)
    fig.suptitle("Video Games complete-test ranking quality (same-dataset baselines only)")
    save_figure(fig, "ranking.png", validation_dir, validation_figures)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    for ax, name, title, ylabel in (
            (axes[0], "loss", "Answer-token validation loss", "Loss (linear scale)"),
            (axes[1], "masked_perplexity", "Answer-token masked perplexity", "Masked perplexity (log scale)")):
        values = observations[name]
        if values:
            bars = ax.bar(["Before training", "Final best model"], values, color=["#9cbec5", "#167d8d"], width=.6)
            ax.bar_label(bars, labels=[fmt(value) for value in values], padding=4)
            if name == "masked_perplexity":
                ax.set_yscale("log")
                ax.set_ylim(min(values) * .7, max(values) * 2)
            else:
                ax.set_ylim(0, max(1, max(values) * 1.2))
        else:
            message = pending_message if validation["status"] != "passed" else "Complete before/after\nobservations unavailable"
            ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", wrap=True)
            ax.set_xticks([])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Validation before training and after best-checkpoint reload")
    save_figure(fig, "validation-comparison.png", validation_dir, validation_figures)

    comparison = validation["comparison"]
    paper = comparison["paper"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), layout="constrained")
    for ax, metric in zip(axes, ("HR", "NDCG")):
        for domain, color in zip(paper.get("results", []), ("#c17b2d", "#7c6b96")):
            points = [(k, domain.get(metric, {}).get(str(k))) for k in paper.get("reported_k", [])]
            points = [(k, value) for k, value in points if finite(value)]
            if points:
                ax.plot([k for k, _ in points], [value for _, value in points], linestyle="--", marker="s",
                        color=color, label=f"Paper {domain['dataset']} | Instruct SFT + RL", linewidth=1.5)
        if observations["ranking"]:
            ax.plot([row["k"] for row in observations["ranking"]],
                    [row[metric] for row in observations["ranking"]], color="#167d8d", marker="o",
                    label="Current Video Games | Base SFT only", linewidth=1.8)
        else:
            ax.text(.98, .04, f"Current model: {validation['status']}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9)
        if ax.lines:
            largest = max(max(line.get_ydata()) for line in ax.lines)
            ax.set_ylim(0, min(1, max(.05, largest * 1.2)))
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(.5, .5, "Paper reference unavailable", transform=ax.transAxes, ha="center")
        ax.set(title=metric, xlabel="K (paper reports 3, 5, 10 only)", ylabel=metric, xticks=TOP_K)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.6)
    fig.suptitle("Different datasets and training stages: reference only\nNo Video Games paper benchmark; no cross-domain improvement claim")
    save_figure(fig, "paper-reference.png", validation_dir, validation_figures)

    page = text_page("SFT Result Validation Report", [
        "MiniOneRec / Video Games / Qwen2.5-1.5B Base", f"Status: {validation['status']}",
        f"Updated (Asia/Shanghai): {validation['generated_at_shanghai']}", "",
        f"Test rows: {evaluation.get('completed_rows')} / expected: {evaluation.get('expected_rows')}",
        f"Valid SID rate: {fmt(evaluation.get('validity_rate'), True)}", f"Unique candidate rate: {fmt(evaluation.get('unique_rate'), True)}",
        f"Invalid candidate count (CC): {evaluation.get('CC')}", "",
        *[f"K={r['k']:2}: HR={fmt(r['HR'], True)}; NDCG={fmt(r['NDCG'], True)}" for r in validation["ranking"]], "",
        f"Before-training answer-token loss: {fmt(first(baseline.get('loss'), baseline.get('eval_loss_answer_tokens')))}",
        f"After-training answer-token loss: {fmt(first(final.get('loss'), evaluation.get('eval_loss_answer_tokens')))}",
        f"Before masked PPL: {fmt(first(baseline.get('masked_perplexity'), baseline.get('masked_ppl')))}",
        f"After masked PPL: {fmt(first(final.get('masked_perplexity'), evaluation.get('masked_ppl')))}", "",
        "PPL uses masked answer tokens only, not general language perplexity.",
        "Shared item catalog; original test rows and duplicate interaction weights retained.",
        "Test data is not used for checkpoint selection. Catalog validity is structural.",
        f"Failure details: {evaluation.get('error', 'none recorded')}",
    ])
    import textwrap

    def text_pages(title, lines):
        wrapped = [part for line in lines for part in (textwrap.wrap(str(line), width=95, break_long_words=True) or [""])]
        return [text_page(title if start == 0 else title + " (continued)", wrapped[start:start+32])
                for start in range(0, len(wrapped), 32)]

    source = paper.get("source", {})
    comparability_pages = text_pages("Paper comparability", [
        f"Source: {source.get('title')}", f"URL: {source.get('url')}",
        f"Version: {source.get('version')}; locators: {source.get('locators')}", "",
        "No Video Games result in paper Table 1; values from different domains are reference only.",
        paper.get("backbone_size_note", ""), "Actual configuration vs paper:",
        *[f"{row['key']}: actual={fmt(row['actual'])}; paper={fmt(row['paper'])}; {row['assessment']}"
          + (f" ({row['note']})" if row['note'] else "") for row in comparison["comparability"]], "",
        "SID selected quantizer epoch is distinct from the paper's quantizer training budget.",
        "Paper settings are not substituted for the actual run configuration.",
        "Differences in domains, catalogs, histories, SIDs and SFT/RL prevent causal attribution.",
        "No cross-domain improvement percentage or reproduction-success verdict is calculated.",
    ])
    reference_pages = text_pages("Actual results and paper references", [
        "Different datasets and stages: reference only. Original main Table 1 is used consistently.",
        "Paper: Industrial / Office, Qwen2.5-Instruct, SFT + RL. Current: Video Games, 1.5B Base, SFT only.",
        "One target per test row: HR@K equals Recall@K in the current protocol.", "",
        *[f"{row['dataset']} | {'current' if row['source'] == 'Current experiment' else 'paper Table 1'} | "
          f"K={row['k']}: HR={fmt(row['HR'], True)}; NDCG={fmt(row['NDCG'], True)}" for row in paper_comparison_rows(validation)], "",
        "Only K=3,5,10 are reported by the paper; K=1,20,50 are not inferred or interpolated.",
        "Paper values cannot establish expected Video Games performance or statistical significance.",
    ])
    analysis = validation["expectation_analysis"]
    assessment_lines = [f"Analysis status: {analysis['status']}; predeclared primary K=10",
                        "Exceeding POP/random is a minimum learning-signal check, not competitiveness or reproduction.",
                        "Descriptive comparisons only; no statistical significance claim.", ""]
    if analysis["status"] == "pending":
        assessment_lines.append(analysis["reason"])
    else:
        assessment_lines.extend(f"{name}: {passed}" for name, passed in analysis["metric_consistency"]["checks"].items())
        assessment_lines.append(f"Validation loss: {analysis.get('validation_loss', 'unavailable')}")
        assessed_baselines = analysis["baseline_comparison"]
        assessment_lines.append(f"Baseline identity status: {assessed_baselines['status']}")
        if assessed_baselines["status"] == "available":
            for name in ("actual", "popularity", "uniform_random"):
                item = assessed_baselines[name]
                assessment_lines.append(f"K=10 {name}: HR={fmt(item['HR'], True)}; NDCG={fmt(item['NDCG'], True)}")
            for name in ("popularity", "uniform_random"):
                assessment_lines.append(f"Model minus {name}, absolute metric differences: {assessed_baselines[name]['differences']}")
            assessment_lines.append(f"Minimum signal checks passed: {analysis['minimum_signal_checks_passed']}")
        else:
            assessment_lines.extend(assessed_baselines.get("reasons", []))
        assessment_lines.extend(["", "Even when minimum checks pass, competitive recommendation quality and paper reproduction remain undetermined."])
    assessment_pages = text_pages("Expectation and signal diagnostics", assessment_lines)
    diagnostic = validation.get("behavior_shift_analysis", {})
    shift_pages = []
    if diagnostic.get("status") == "available":
        shift = diagnostic["shift_summary"]
        alignment = diagnostic.get("alignment_coverage", {})
        shift_lines = [
            "Post-hoc exploratory analysis; original full-catalog beam=50 predictions are reused.",
            "No candidate subset filtering, reranking, metric changes or new primary-endpoint selection.",
            f"Behavior-cold test-target rate: {fmt(shift.get('test_behavior_cold_rate'), True)}",
            f"Repeat-last rate: train={fmt(shift.get('train_repeat_last_rate'), True)}; test={fmt(shift.get('test_repeat_last_rate'), True)}",
            "Behavior-cold means absent from training behavior histories AND targets, not unseen SID/text.",
            f"Cold catalog items={alignment.get('cold_unique_items')}; SID-to-title covered={alignment.get('sid_to_title_items')}; title-to-SID covered={alignment.get('title_to_sid_items')}", "",
            *[f"{group['key']}: n={group['rows']} ({fmt(group['share'], True)}); "
              f"HR1={fmt(group['metrics'].get('HR@1'), True)}; HR10={fmt(group['metrics'].get('HR@10'), True)}; "
              f"NDCG10={fmt(group['metrics'].get('NDCG@10'), True)}; HR50={fmt(group['metrics'].get('HR@50'), True)}; "
              f"POP HR10={fmt(group.get('popularity', {}).get('HR@10'), True)}" for group in diagnostic["strata"]], "",
            "Cold/warm partition the rows; repeat-last/non-repeat partition them separately. Dimensions overlap.",
            "Overall HR is the row-weighted mixture of conditional HR; this is not a causal counterfactual.",
            "Optimization/integrity passing does not imply behavior-cold transfer passing.",
            "Global POP lacks behavior support for cold targets; temporal and behavior shift can weaken POP.",
            "A multiple over POP is descriptive, not evidence of competitive recommendation quality.",
            "Shared text/SID alignment exposes these cold items; do not claim strict unseen-item generalization.",
            "No isolated time-shift causality, statistical significance or paper-reproduction conclusion.",
            "Matched strong sequence baselines, pre-SFT ranking and multi-seed uncertainty remain unavailable.",
        ]
        predictions = diagnostic.get("prediction_diagnostics", {})
        if predictions:
            shift_lines.extend([
                "", f"Model Top1 copies last item: {fmt(predictions.get('top1_equals_last_rate'), True)}; true repeat-last: {fmt(shift.get('test_repeat_last_rate'), True)}",
                f"Copy precision: {fmt(predictions.get('copy_precision'), True)}; share of all Top1 hits: {fmt(predictions.get('copy_share_of_all_top1_hits'), True)}",
                f"Behavior-cold share of generated candidate positions: {fmt(predictions.get('cold_candidate_rate'), True, 5)}; unique cold items generated: {predictions.get('cold_predicted_unique_items')}",
                "These are descriptive prediction patterns; they do not isolate a causal mechanism.",
            ])
        shift_pages = text_pages("Behavior shift and conditional recall", shift_lines)
    validation_pages = [page, *comparability_pages, *reference_pages, *assessment_pages, *shift_pages]
    with PdfPages(validation_dir / "validation-report.pdf") as pdf:
        for text_figure in validation_pages:
            pdf.savefig(text_figure)
        for fig in validation_figures:
            pdf.savefig(fig)
    page.savefig(validation_dir / "validation-summary.png", dpi=130)
    for text_figure in validation_pages:
        plt.close(text_figure)
    for fig in validation_figures:
        plt.close(fig)
    return {"training_pdf_pages": 5, "validation_pdf_pages": len(validation_pages) + len(validation_figures)}


def output_manifest(directory, report, pages):
    files = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name != "report_manifest.json" and not path.name.endswith(".tmp"):
            data = path.read_bytes()
            files.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    result = {"stage": report["report"], "status": report["status"], "run_directory": report["run_directory"],
              "created_at_utc": report["generated_at_utc"], "pages": pages,
              "sources": report["sources"], "files": files}
    write_json(Path(directory) / "report_manifest.json", result)
    return result


def update_collection(report_root, training, validation, manifests):
    """Append new sections without requiring archived 01-05 files in this container."""
    report_root = Path(report_root)
    start, end = "<!-- SFT_REPORTS_START -->", "<!-- SFT_REPORTS_END -->"
    section = start + "<section id='sft_reports'><div class='eyebrow'>06 / 07 · SFT</div><h2>SFT 训练与完整结果验证</h2>"
    section += (f"<p>训练：{STATUS_LABELS.get(training['status'], training['status'])}；"
                f"结果验证：{STATUS_LABELS.get(validation['status'], validation['status'])}。"
                "模型为 Qwen2.5-1.5B Base，RL 未启动。</p>")
    section += (f"<a class='button' href='{TRAINING_DIRECTORY}/index.html'>训练报告与 gradNorm</a>"
                f"<a class='button' href='{VALIDATION_DIRECTORY}/index.html'>结果验证报告</a></section>") + end
    index = report_root / "index.html"
    if index.is_file():
        original = index.read_bytes()
        text = original.decode("utf-8-sig")
        pattern = re.escape(start) + r".*?" + re.escape(end)
        if start in text and end in text:
            text = re.sub(pattern, lambda _: section, text, count=1, flags=re.S)
        elif "</main>" in text:
            text = text.replace("</main>", section + "</main>", 1)
        else:
            text += section
        atomic_text(index, text)
    else:
        atomic_text(index, "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
                    f"<title>Video Games Reports</title><style>{CSS}</style><main><h1>Video Games 报告合集</h1>"
                    + section + "</main></html>")
    manifest_path = report_root / "reports_manifest.json"
    manifest = read_json(manifest_path)
    directories = {TRAINING_DIRECTORY, VALIDATION_DIRECTORY}
    reports = [item for item in manifest.get("reports", []) if item.get("directory") not in directories]
    for directory, report, details in zip((TRAINING_DIRECTORY, VALIDATION_DIRECTORY),
                                          (training, validation), manifests):
        reports.append({"directory": directory, "entrypoint": directory + "/index.html",
                        "pdf": directory + ("/training-report.pdf" if directory == TRAINING_DIRECTORY else "/validation-report.pdf"),
                        "pages": details["pages"], "status": report["status"]})
    # Archive records remain byte-for-byte equivalent as JSON objects; do not
    # regenerate or validate unavailable archived subdirectories on this machine.
    files = [item for item in manifest.get("files", [])
             if item.get("path", "").split("/", 1)[0] not in directories]
    for directory, details in zip((TRAINING_DIRECTORY, VALIDATION_DIRECTORY), manifests):
        files.extend({**item, "path": directory + "/" + item["path"]} for item in details["files"])
        path = report_root / directory / "report_manifest.json"
        data = path.read_bytes()
        files.append({"path": directory + "/report_manifest.json", "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    manifest.update(entrypoint="index.html", reports=reports, files=files,
                    sft_status={"training": training["status"], "validation": validation["status"]},
                    updated_at_utc=training["generated_at_utc"])
    write_json(manifest_path, manifest)


def generate(run_dir, report_root, plots=True):
    report_root = Path(report_root)
    training_dir, validation_dir = report_root / TRAINING_DIRECTORY, report_root / VALIDATION_DIRECTORY
    training_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    training, validation, steps, gpu = build_reports(run_dir)
    pages = render_charts(training, validation, steps, gpu, training_dir, validation_dir) if plots else {
        "training_pdf_pages": 0, "validation_pdf_pages": 0}
    training["rendered_pdf"] = bool(plots)
    validation["rendered_pdf"] = bool(plots)
    write_json(training_dir / "training_report.json", training)
    write_json(validation_dir / "validation_report.json", validation)
    write_json(validation_dir / "comparison.json", validation["comparison"])
    write_json(validation_dir / "expectation_analysis.json", validation["expectation_analysis"])
    write_json(validation_dir / "behavior_shift_analysis.json", validation["behavior_shift_analysis"])
    write_csv(training_dir / "metrics.csv", steps, ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm",
              "grad_norm_rolling_mean", "grad_norm_rolling_max", "grad_norm_mean", "grad_norm_max", "step_skipped", "elapsed_seconds"])
    write_csv(training_dir / "gpu_metrics.csv", gpu, ["timestamp", "elapsed_seconds", "gpu_index", "memory_used_mib", "utilization_percent"])
    write_csv(validation_dir / "metrics.csv", validation["ranking"], ["k", "HR", "NDCG"])
    write_csv(validation_dir / "paper_comparison.csv", paper_comparison_rows(validation), ["source", "dataset", "stage", "k", "HR", "NDCG"])
    write_csv(validation_dir / "comparability.csv", validation["comparison"]["comparability"], ["key", "setting", "actual", "paper", "assessment", "note"])
    baselines = validation["comparison"]["same_dataset_baselines"]
    baseline_rows = [{"method": method, "k": k, "HR": metrics[f"HR@{k}"], "NDCG": metrics[f"NDCG@{k}"]}
                     for method, metrics in baselines.get("metrics", {}).items() for k in TOP_K]
    write_csv(validation_dir / "baseline_metrics.csv", baseline_rows, ["method", "k", "HR", "NDCG"])
    write_csv(validation_dir / "behavior_shift_metrics.csv", behavior_shift_rows(validation),
              ["stratum", "label", "rows", "share", "k", "HR", "NDCG", "hits", "POP_HR", "POP_NDCG", "uniform_HR", "uniform_NDCG"])
    atomic_text(training_dir / "index.html", training_html(training))
    atomic_text(validation_dir / "index.html", validation_html(validation))
    manifests = [output_manifest(training_dir, training, pages["training_pdf_pages"]),
                 output_manifest(validation_dir, validation, pages["validation_pdf_pages"])]
    update_collection(report_root, training, validation, manifests)
    return {"training_status": training["status"], "validation_status": validation["status"],
            "training_report": str(training_dir / "index.html"), "validation_report": str(validation_dir / "index.html")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=ROOT / "output/reports/Video_Games")
    parser.add_argument("--no-plots", action="store_true", help="Skip PDF/PNG rendering; useful only for lightweight checks")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.run_dir, args.report_root, not args.no_plots), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
