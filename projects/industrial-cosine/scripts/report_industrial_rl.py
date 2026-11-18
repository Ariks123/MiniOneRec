#!/usr/bin/env python3
"""Read-only RL telemetry and verified parent-SFT comparisons for Industrial.

Only report artifacts are written. Ranking metrics are reaggregated from the
complete saved predictions; validation, never test ranking, recommends a model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html
import io
import math
from pathlib import Path
from statistics import mean

try:
    from .report_industrial_sft import (
        DATASET, TOP_K, atomic_text, finite, fmt, read_json, read_jsonl,
        require, sha256, table, validated_evaluation, write_json,
    )
except ImportError:
    from report_industrial_sft import (
        DATASET, TOP_K, atomic_text, finite, fmt, read_json, read_jsonl,
        require, sha256, table, validated_evaluation, write_json,
    )


PARENT_FILES = {
    "run_config": ("run_config.json", "parent_sft_run_config.json"),
    "training_summary": ("training_summary.json", "parent_sft_training_summary.json"),
    "evaluation_metrics": ("evaluation/metrics.json", "parent_sft_evaluation_metrics.json"),
    "predictions": ("evaluation/predictions.jsonl", "parent_sft_predictions.jsonl"),
}
METRIC_ALIASES = {
    "reward": ("reward", "reward_mean", "rewards/mean"),
    "kl": ("kl", "objective/kl", "kl_mean"),
    "zero_reward_group_rate": ("zero_reward_group_rate",),
    "hit_group_rate": ("hit_group_rate",),
    "advantage_nonzero_group_rate": ("advantage_nonzero_group_rate",),
    "learning_rate": ("learning_rate",),
    "grad_norm": ("grad_norm",),
    "eval_ndcg10": ("eval_ndcg10",),
    "eval_recall10": ("eval_recall10",),
}
PLOT_SPECS = (
    ("RL reward", "reward.png", ("reward",)),
    ("KL divergence", "kl.png", ("kl",)),
    ("Learning rate", "learning_rate.png", ("learning_rate",)),
    ("Global gradient L2 norm before clipping", "grad_norm.png", ("grad_norm",)),
    ("Reward groups", "zero_reward_group_rate.png", (
        "zero_reward_group_rate", "hit_group_rate", "advantage_nonzero_group_rate")),
    ("Full validation ranking", "validation_ndcg.png", ("eval_ndcg10", "eval_recall10")),
)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load(path, errors):
    try:
        result = read_json(path)
        require(isinstance(result, dict), str(path) + ": expected a JSON object")
        return result
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(path) + ": " + str(error))
        return {}


def observations(run_dir, errors):
    merged = {}
    try:
        raw = read_jsonl(run_dir / "monitor/metrics.jsonl")
        for observed in raw:
            require(isinstance(observed, dict), "RL telemetry row is not an object")
            # Reloading the selected best checkpoint evaluates it again at the
            # final optimizer step. That score is a model-summary result, not
            # the training-time validation score at that step.
            if observed.get("kind") == "eval" and observed.get("phase", "training") != "training":
                continue
            step = observed.get("step", observed.get("global_step"))
            if not finite(step):
                continue
            row = merged.setdefault(int(step), {"step": int(step)})
            row.update({key: value for key, value in observed.items() if value is not None})
            for metric, aliases in METRIC_ALIASES.items():
                value = next((observed[key] for key in aliases if finite(observed.get(key))), None)
                if value is not None:
                    row[metric] = value
    except (OSError, ValueError, TypeError) as error:
        errors.append("RL telemetry: " + str(error))
    return [merged[key] for key in sorted(merged)]


def metric_statistics(steps):
    result = {}
    extra = {key for row in steps for key in row if key.startswith(("rewards/", "task/"))}
    for key in sorted(set(METRIC_ALIASES) | extra):
        values = [row[key] for row in steps if finite(row.get(key)) and (key != "grad_norm" or (
            row[key] >= 0 and not row.get("step_skipped") and not row.get("optimizer_overflow")
            and not row.get("grad_norm_nonfinite")))]
        result[key] = {"count": len(values), "mean": mean(values) if values else None,
                       "last": values[-1] if values else None, "max": max(values) if values else None}
    return result


def runtime_context(config):
    """Describe recorded corrections without inventing provenance for old runs."""
    compatibility = config.get("runtime_compatibility")
    notes = []
    if isinstance(compatibility, dict) and compatibility.get("applied") is True:
        notes.extend([
            "This RL run applies the recorded DeepSpeed gradient-accumulation compatibility correction.",
            "The SFT baseline uses its existing fixed export; it was not retrained for this comparison.",
            "Historical SFT training with the same DeepSpeed configuration may also have been affected. "
            "This comparison cannot isolate algorithm gains from changes in gradient-accumulation behavior.",
        ])
    else:
        notes.extend([
            "No applied gradient-accumulation compatibility correction is verified by this run configuration.",
            "The SFT baseline uses its existing fixed export; it was not retrained for this comparison.",
        ])
    notes.append("Failed earlier attempts are not RL results and are excluded from all metric comparisons.")
    return compatibility, notes


def runtime_pdf_lines(report):
    metadata = report.get("runtime_compatibility")
    return [f"Runtime compatibility metadata: {metadata if metadata is not None else 'not recorded'}",
            *report.get("comparison_limitations", [])]


def runtime_html(report):
    metadata = report.get("runtime_compatibility")
    result = "<h3>运行兼容修复与对照范围</h3>"
    if isinstance(metadata, dict) and metadata.get("applied") is True:
        result += ("<p>本次 RL 已应用所记录的 DeepSpeed 梯度累积兼容修复。"
                   "基线采用既有 SFT 固定导出，没有因此自动重训 SFT。历史 SFT 在同一 DeepSpeed 配置下也可能受影响，"
                   "因此本次对比无法分别量化算法和梯度累积行为变化的贡献。</p>")
    else:
        result += "<p>本次配置未记录已应用的梯度累积兼容修复；基线采用既有 SFT 固定导出。</p>"
    result += "<p>此前失败尝试不作为 RL 结果，不纳入任何指标比较。</p>"
    result += "<pre>" + html.escape(str(metadata) if metadata is not None else "runtime_compatibility: not recorded") + "</pre>"
    return result


def parent_checkpoint(config, supervisor):
    source = config.get("model", {}).get("source")
    declared = supervisor.get("model_path", supervisor.get("parameters", {}).get("model_path"))
    require(source or declared, "Parent SFT checkpoint is not yet recorded")
    if source and declared:
        require(Path(source).resolve() == Path(declared).resolve(), "RL and supervisor parent model paths differ")
    checkpoint = Path(source or declared).resolve()
    require(checkpoint.name == "final_checkpoint", "RL source must be the parent SFT final_checkpoint")
    return checkpoint


def accepted_parent(run_dir, config, supervisor, data_dir, input_dir, preparation):
    """Choose one complete original/cache set and verify its pinned identities."""
    checkpoint = parent_checkpoint(config, supervisor)
    manifest = supervisor.get("parent_sft_report_sources", {})
    parent_run = Path(manifest.get("run_dir", checkpoint.parent)).resolve()
    require(parent_run == checkpoint.parent, "Parent source manifest points to another SFT run")
    originals = {key: parent_run / names[0] for key, names in PARENT_FILES.items()}
    direct = all(path.is_file() for path in originals.values())
    paths = originals if direct else {key: run_dir / names[1] for key, names in PARENT_FILES.items()}
    require(all(path.is_file() for path in paths.values()), "Complete parent SFT metadata/predictions are unavailable")
    pinned = manifest.get("sha256", {})
    require(direct or all(key in pinned for key in PARENT_FILES),
            "Cached parent SFT sources require a pinned SHA-256 manifest")
    hashes = {key: sha256(path) for key, path in paths.items()}
    if manifest:
        require(all(pinned.get(key) == value for key, value in hashes.items()),
                "Parent SFT report source SHA-256 differs from its pinned identity")
    # Core metadata records the exact parent it verified before any RL update.
    # Compare those references with the independent supervisor-pinned files.
    for name in ("run_config.json", "training_summary.json"):
        reference = read_json(run_dir / name).get("parent_sft")
        if reference is not None:
            require(isinstance(reference, dict)
                    and Path(reference.get("model_path", "")).resolve() == checkpoint
                    and Path(reference.get("run_dir", "")).resolve() == parent_run
                    and reference.get("selected_step") == 5187
                    and Path(reference.get("selected_checkpoint", "")).resolve() == parent_run / "checkpoint-5187"
                    and reference.get("config_sha256") == hashes["run_config"]
                    and reference.get("summary_sha256") == hashes["training_summary"],
                    "RL parent SFT metadata reference differs: " + name)
            if supervisor.get("parent_model_sha256"):
                require(reference.get("model_sha256") == supervisor["parent_model_sha256"],
                        "RL parent SFT weight identity differs: " + name)
    parent_config = read_json(paths["run_config"])
    summary = read_json(paths["training_summary"])
    evaluation = read_json(paths["evaluation_metrics"])
    hyper = parent_config.get("hyperparameters", {})
    require(parent_config.get("stage") == "sft"
            and Path(parent_config.get("model", {}).get("source", "")).name == "Qwen2.5-0.5B"
            and hyper.get("max_epochs") == 3 and hyper.get("lr_scheduler_type") == "linear"
            and parent_config.get("data", {}).get("include_title_history") is True,
            "Parent is not the declared 0.5B linear-three-epoch title-history SFT")
    require(summary.get("status") == "completed" and summary.get("selection_metric") == "eval_loss"
            and summary.get("final_checkpoint_contains_best_model") is True
            and Path(summary.get("best_checkpoint", "")).name == "checkpoint-5187"
            and Path(summary.get("final_checkpoint", "")).resolve() == checkpoint,
            "Parent SFT does not identify the completed validation-loss best checkpoint-5187")
    require(Path(evaluation.get("model_path", "")).resolve() == checkpoint
            and evaluation.get("scope") == "complete_original_test_split", "Parent evaluation model/scope differs")
    analysis = validated_evaluation(parent_run, data_dir, input_dir, parent_config, evaluation, preparation,
                                    predictions_path=paths["predictions"])
    require(all(sha256(path) == hashes[key] for key, path in paths.items()),
            "Parent SFT report sources changed during analysis")
    return {"status": "passed", "run_dir": str(parent_run), "model_path": str(checkpoint),
            "source": "original" if direct else "pinned_cache", "report_source_sha256": hashes,
            "config": parent_config, "training_summary": summary, "evaluation": evaluation, **analysis}


def validation_recommendation(config, summary, baseline, data_dir, input_dir, preparation):
    if summary.get("status") != "completed":
        return {"status": "pending", "recommended_model": None, "test_used_for_selection": False}
    try:
        count = preparation.get("rows", {}).get("valid")
        coverage, generation = baseline.get("coverage", {}), baseline.get("generation", {})
        require(baseline.get("status") == "passed" and baseline.get("category") == DATASET
                and baseline.get("scope") == "complete_original_validation_split" and baseline.get("step") == 0,
                "Missing accepted pre-RL full-validation baseline")
        require(Path(baseline.get("model_path", "")).resolve() == Path(config.get("model", {}).get("source", "")).resolve(),
                "Pre-RL validation baseline does not identify the parent SFT model")
        require(type(count) is int and count > 0
                and coverage.get("expected_rows") == coverage.get("completed_rows") == count
                and coverage.get("missing_rows") == coverage.get("duplicate_rows") == 0,
                "Parent validation baseline coverage is incomplete")
        require(generation.get("num_beams") == 50 and generation.get("prompt_style") == "original-eval"
                and generation.get("full_catalog") is True and generation.get("seen_item_filter") is False,
                "Parent validation baseline generation protocol differs")
        for key, path in {"valid_file": data_dir / "valid.csv",
                          "sid_index_path": input_dir / f"{DATASET}.index.json",
                          "preparation_manifest": data_dir / "preparation_manifest.json"}.items():
            require(baseline.get("source_sha256", {}).get(key) == sha256(path),
                    "Parent validation baseline source differs: " + key)
        before, after = baseline.get("metrics", {}).get("ndcg10"), summary.get("best_metric")
        require(summary.get("selection_metric") == "eval_ndcg10" and finite(before) and finite(after)
                and 0 <= before <= 1 and 0 <= after <= 1
                and summary.get("final_checkpoint_contains_best_model") is True,
                "RL selection must identify the reloaded validation-NDCG@10 best model")
        final = summary.get("final_eval_metrics", {}).get("eval_ndcg10")
        require(summary.get("final_eval_metrics", {}).get("eval_generation_rows") == count,
                "Reloaded RL model validation ranking coverage is incomplete")
        require(finite(final) and math.isclose(final, after, abs_tol=1e-7, rel_tol=0),
                "Reloaded RL model validation NDCG@10 differs from its selected score")
        return {"status": "passed", "basis": "complete_validation_ndcg10", "test_used_for_selection": False,
                "recommended_model": "RL" if after > before + 1e-12 else "SFT",
                "sft_validation_ndcg10": before, "rl_validation_ndcg10": after,
                "difference_percentage_points": 100 * (after - before),
                "tie_rule": "Retain parent SFT when validation NDCG@10 does not improve"}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"status": "incomplete", "recommended_model": None, "test_used_for_selection": False,
                "error": str(error)}


def comparison_rows(parent, validation):
    left = {row["key"]: row for row in parent.get("strata", [])}
    right = {row["key"]: row for row in validation.get("strata", [])}
    result = []
    for group in dict.fromkeys([*left, *right]):
        a, b = left.get(group), right.get(group)
        if a and b:
            require(a["rows"] == b["rows"], "SFT/RL test stratum row counts differ: " + group)
        for k in TOP_K:
            row = {"stratum": group, "rows": (a or b)["rows"], "k": k}
            for label, record in (("sft", a), ("rl", b)):
                row[label + "_recall"] = record["metrics"][f"HR@{k}"] if record else None
                row[label + "_ndcg"] = record["metrics"][f"NDCG@{k}"] if record else None
                row[label + "_hits"] = record["hits_by_k"][str(k)] if record else None
            for metric in ("recall", "ndcg"):
                av, bv = row["sft_" + metric], row["rl_" + metric]
                row["delta_" + metric + "_pp"] = 100 * (bv - av) if finite(av) and finite(bv) else None
            result.append(row)
    return result


def expectation_assessment(comparison):
    """Describe the selected model's outcome without selecting a model on test."""
    definitions = {
        "improved": "Validation and test NDCG@10 both increase, and test HR@10/HR@50 do not decrease.",
        "mixed": "Some primary metrics increase and others decrease or remain unchanged, without meeting improved.",
        "not_improved": "None of validation NDCG@10, test NDCG@10, test HR@10 and test HR@50 increases.",
        "pending": "Complete verified validation and both complete test evaluations are required for a conclusion.",
    }
    tolerance = 1e-12

    def metric(before=None, after=None):
        delta = after - before if finite(before) and finite(after) else None
        return {"sft": before, "rl": after,
                "difference_percentage_points": 100 * delta if delta is not None else None,
                "direction": None if delta is None else "up" if delta > tolerance else (
                    "down" if delta < -tolerance else "unchanged")}

    primary = {key: metric() for key in ("validation_ndcg10", "test_ndcg10", "test_hr10", "test_hr50")}
    recommendation = comparison.get("recommendation", {})
    if recommendation.get("status") == "passed":
        primary["validation_ndcg10"] = metric(recommendation.get("sft_validation_ndcg10"),
                                               recommendation.get("rl_validation_ndcg10"))
    complete = comparison.get("status") == "passed" and comparison.get("test_comparison_status") == "passed"
    rows = {(row["stratum"], row["k"]): row for row in comparison.get("rows", [])} if complete else {}
    for key, k, name in (("test_ndcg10", 10, "ndcg"), ("test_hr10", 10, "recall"), ("test_hr50", 50, "recall")):
        row = rows.get(("overall", k), {})
        primary[key] = metric(row.get("sft_" + name), row.get("rl_" + name))
    strata = []
    for key in ("behavior_warm", "behavior_cold"):
        row = rows.get((key, 50))
        if row:
            strata.append({"stratum": key, "rows": row["rows"], "sft_hits": row["sft_hits"],
                           "rl_hits": row["rl_hits"], "hit_difference": row["rl_hits"] - row["sft_hits"],
                           **metric(row["sft_recall"], row["rl_recall"])})
    complete = complete and all(record["direction"] is not None for record in primary.values())
    status = "pending"
    if complete:
        directions = {key: value["direction"] for key, value in primary.items()}
        if directions["validation_ndcg10"] == directions["test_ndcg10"] == "up" and all(
                directions[key] != "down" for key in ("test_hr10", "test_hr50")):
            status = "improved"
        elif "up" in directions.values():
            status = "mixed"
        else:
            status = "not_improved"
    summaries = {
        "pending": "The full validation/test comparison has not passed; no expectation conclusion is drawn.",
        "improved": "The selected RL model meets the stated descriptive improvement criterion over fixed SFT.",
        "mixed": "The selected RL model shows partial gains, but does not meet the combined improvement criterion.",
        "not_improved": "The selected RL model shows no gain in the four primary validation/test metrics.",
    }
    summaries_zh = {
        "pending": "完整验证与测试对照尚未全部通过，暂不判断是否符合预期。",
        "improved": "所选 RL 模型满足约定的描述性改善标准。",
        "mixed": "部分指标改善，但未满足验证、测试和召回率共同改善的标准。",
        "not_improved": "四项主要验证与测试指标均未获得正收益，未达到改善预期。",
    }
    boundaries = ["Only one training seed was evaluated. Changes are descriptive; repeated-seed uncertainty was not estimated."]
    cold = next((row for row in strata if row["stratum"] == "behavior_cold"), None)
    if cold and cold["rows"]:
        boundaries.append(f"The behavior-cold group has {cold['rows']} test rows; one hit changes HR@50 by "
                          f"{100 / cold['rows']:.6g} percentage points. A few hits do not establish a reliable cold-item gain.")
    else:
        boundaries.append("Accepted cold-group row/hit counts are required before interpreting cold-item changes.")
    boundaries.extend(comparison.get("comparison_limitations", []))
    return {"status": status, "summary": summaries[status], "summary_zh": summaries_zh[status],
            "definitions": definitions, "absolute_score_tolerance": tolerance,
            "primary_metrics": primary, "strata_hr50": strata,
            "test_rows": rows.get(("overall", 10), {}).get("rows"),
            "evidence_boundaries": boundaries, "test_used_for_selection": False,
            "recommendation_changed": False}


def expectation_pdf_lines(assessment):
    lines = [f"Expectation assessment: {assessment['status']}", assessment["summary"],
             "This is descriptive analysis; model recommendation remains based only on validation NDCG@10.",
             *[f"Definition {key}: {value}" for key, value in assessment["definitions"].items()]]
    for key, value in assessment["primary_metrics"].items():
        lines.append(f"{key}: SFT={fmt(value['sft'], True)}; RL={fmt(value['rl'], True)}; "
                     f"difference={fmt(value['difference_percentage_points'])} percentage points; {value['direction']}")
    for row in assessment["strata_hr50"]:
        lines.append(f"{row['stratum']} HR@50 (n={row['rows']}): SFT hits={row['sft_hits']}, RL hits={row['rl_hits']}, "
                     f"hit difference={row['hit_difference']}; SFT={fmt(row['sft'], True)}, RL={fmt(row['rl'], True)}, "
                     f"difference={fmt(row['difference_percentage_points'])} percentage points")
    return [*lines, *assessment["evidence_boundaries"]]


def expectation_html(assessment):
    labels = {"validation_ndcg10": "验证 NDCG@10", "test_ndcg10": "独立测试 NDCG@10",
              "test_hr10": "独立测试 HR@10", "test_hr50": "独立测试 HR@50"}
    result = "<h3>是否符合改善预期</h3>"
    result += "<p>" + html.escape(assessment["status"] + "：" + assessment["summary_zh"]) + "</p>"
    result += ("<p>improved：验证与测试 NDCG@10 均升，测试 HR@10/50 不降；mixed：指标有涨有跌，"
               "或仅部分提升且未满足共同改善条件；not_improved：四项主指标均无正收益；"
               "pending：完整验证与双侧测试尚未通过。该描述性分析不改变仅按验证集进行的选模。</p>")
    result += table(["指标", "无 RL 的 SFT", "RL", "差值（百分点）", "方向"], [
        (labels[key], fmt(value["sft"], True), fmt(value["rl"], True), fmt(value["difference_percentage_points"]),
         fmt(value["direction"])) for key, value in assessment["primary_metrics"].items()])
    result += table(["HR@50 分组", "样本数", "SFT 命中", "RL 命中", "命中差", "SFT HR@50", "RL HR@50", "差值（百分点）"], [
        (row["stratum"], row["rows"], row["sft_hits"], row["rl_hits"], row["hit_difference"],
         fmt(row["sft"], True), fmt(row["rl"], True), fmt(row["difference_percentage_points"])) for row in assessment["strata_hr50"]])
    result += "<p>单次种子结果是描述性证据；冷数据按实际样本数和命中数解读，不能用少量命中证明稳定改善。</p>"
    result += "<ul>" + "".join("<li>" + html.escape(line) + "</li>" for line in assessment["evidence_boundaries"]) + "</ul>"
    return result


def build_reports(run_dir, data_dir, input_dir):
    run_dir, data_dir, input_dir = map(Path, (run_dir, data_dir, input_dir))
    errors = []
    config, state, summary, monitor, evaluation, preparation, supervisor, baseline = [load(path, errors) for path in (
        run_dir / "run_config.json", run_dir / "run_status.json", run_dir / "training_summary.json",
        run_dir / "monitor/summary.json", run_dir / "evaluation/metrics.json", data_dir / "preparation_manifest.json",
        run_dir / "supervisor_config.json", run_dir / "baseline_validation.json")]
    steps = observations(run_dir, errors)
    stats = metric_statistics(steps)
    stage = state.get("stage", "pending")
    if summary.get("status") == "completed":
        training_status = "completed"
        if config.get("stage") != "rl" or not stats["reward"]["count"] or not stats["grad_norm"]["count"]:
            training_status = "incomplete"
            errors.append("Completed RL training lacks RL configuration, reward or valid gradient telemetry")
        if summary.get("selection_metric") != "eval_ndcg10":
            training_status = "incomplete"
            errors.append("RL checkpoint selection is not validation NDCG@10")
    elif summary.get("status") == "failed" or state.get("status") == "failed":
        training_status = "failed"
    else:
        training_status = "running" if stage in ("training", "validating", "reporting") else "pending"
    analysis = {"status": "running" if stage == "validating" else "pending", "strata": []}
    if evaluation.get("status") == "passed":
        try:
            require(summary.get("status") == "completed" and summary.get("selection_metric") == "eval_ndcg10"
                    and summary.get("final_checkpoint_contains_best_model") is True,
                    "Final RL test evaluation requires completed validation-NDCG-selected training")
            require(evaluation.get("scope") == "complete_original_test_split"
                    and Path(evaluation.get("model_path", "")).resolve() == Path(summary.get("final_checkpoint", "")).resolve(),
                    "RL test evaluation does not identify its exported final model/full test scope")
            analysis = validated_evaluation(run_dir, data_dir, input_dir, config, evaluation, preparation)
        except (OSError, ValueError, TypeError, KeyError) as error:
            analysis = {"status": "incomplete", "strata": [], "error": str(error)}
            errors.append("RL test evaluation: " + str(error))
    elif evaluation.get("status") == "failed" or state.get("status") == "failed" and stage in (
            "validating", "reporting"):
        analysis = {"status": "failed", "strata": [], "error": evaluation.get("error", state.get("error"))}
        errors.append("RL test evaluation failed: " + str(analysis["error"]))
    parent = {"status": "pending", "strata": []}
    try:
        parent = accepted_parent(run_dir, config, supervisor, data_dir, input_dir, preparation)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parent = {"status": "incomplete" if config or supervisor else "pending", "strata": [], "error": str(error)}
    recommendation = validation_recommendation(config, summary, baseline, data_dir, input_dir, preparation)
    if recommendation["status"] == "passed" and parent["status"] != "passed":
        recommendation = {"status": "incomplete", "recommended_model": None, "test_used_for_selection": False,
                          "error": "Parent SFT identity and saved predictions have not passed verification"}
    rows = comparison_rows(parent, analysis)
    paired = parent["status"] == analysis["status"] == "passed"
    comp_status = "passed" if paired and recommendation["status"] == "passed" else (
        "failed" if analysis["status"] == "failed" else "incomplete" if summary.get("status") == "completed" else "pending")
    timestamp = datetime.now(timezone.utc).isoformat()
    compatibility, limitations = runtime_context(config)
    common = {"format_version": 1, "generated_at_utc": timestamp,
              "generated_at_local": datetime.fromisoformat(timestamp).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
              "display_timezone": "Asia/Shanghai", "dataset": DATASET,
              "run_directory": str(run_dir), "stage": "rl", "errors": errors,
              "report_generator_sha256": sha256(Path(__file__)),
              "training_report_source_sha256": supervisor.get("source_sha256", {}).get("scripts/report_industrial_rl.py"),
              "runtime_compatibility": compatibility, "comparison_limitations": limitations}
    training = {**common, "status": training_status, "run_status": state, "config": config,
                "training_summary": summary, "monitor_summary": monitor, "metric_statistics": stats,
                "grad_norm": stats["grad_norm"], "baseline_validation": baseline,
                "objective": "RL policy optimization; reward and KL are not teacher-forced SFT loss",
                "selection_metric": "eval_ndcg10", "parent_model_path": config.get("model", {}).get("source")}
    validation = {**common, **analysis, "evaluation": evaluation,
                  "model_path": summary.get("final_checkpoint"), "test_used_for_selection": False}
    comparison = {**common, "status": comp_status, "test_comparison_status": "passed" if paired else "pending",
                  "parent_sft": parent, "rl": {"status": analysis["status"],
                      "model_path": summary.get("final_checkpoint"), "source_sha256": analysis.get("source_sha256"),
                      "selected_checkpoint": summary.get("best_checkpoint"), "selected_validation_ndcg10": summary.get("best_metric")},
                  "recommendation": recommendation, "rows": rows,
                  "protocol": {"scope": "complete_original_test_split", "num_beams": 50,
                      "expected_test_rows": preparation.get("rows", {}).get("test"), "test_used_for_selection": False,
                      "metric_semantics": analysis.get("metric_semantics", parent.get("metric_semantics"))},
                  "interpretation": ["HR@K equals Recall@K for one relevant target per test row.",
                      "Behavior-cold targets are item IDs absent from training histories and targets; SID aliases remain possible.",
                      "Cold results use the displayed row and hit counts; a few hits do not establish a reliable improvement.",
                      "Model recommendation uses complete validation NDCG@10. Final test comparison does not select a model."]}
    assessment = expectation_assessment(comparison)
    for report in (training, validation, comparison):
        report["expectation_assessment"] = assessment
    return json_safe(training), json_safe(validation), json_safe(comparison), json_safe(steps)


def render(training, validation, comparison, steps, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    summary = training["training_summary"]
    figures = []
    for title, filename, keys in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
        for key in keys:
            points = [(row["step"], row[key]) for row in steps if finite(row.get(key)) and (
                key != "grad_norm" or not row.get("step_skipped") and not row.get("grad_norm_nonfinite")
                and not row.get("optimizer_overflow"))]
            if points:
                ax.plot(*zip(*points), label=key)
        if filename == "validation_ndcg.png":
            baseline = training.get("baseline_validation", {})
            if baseline.get("status") == "passed":
                for key, color in (("ndcg10", "#1f77b4"), ("recall10", "#ff7f0e")):
                    score = baseline.get("metrics", {}).get(key)
                    if finite(score):
                        ax.axhline(score, color=color, linestyle="--", alpha=.6, label="step-0 SFT " + key)
            selected = str(summary.get("best_checkpoint") or "").rsplit("-", 1)[-1]
            if selected.isdigit() and finite(summary.get("best_metric")):
                ax.scatter([int(selected)], [summary["best_metric"]], marker="*", s=130,
                           color="#16452c", zorder=5, label="Selected checkpoint " + selected)
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, "No accepted observations yet", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="Optimizer update step", ylabel=title)
        ax.grid(alpha=.3)
        fig.savefig(destination / filename, dpi=140)
        figures.append(fig)

    def pages(title, lines):
        wrapped = [part for line in lines for part in (textwrap.wrap(str(line), 95) or [""])]
        result = []
        for offset in range(0, len(wrapped), 33):
            page = plt.figure(figsize=(8.27, 11.69))
            page.text(.08, .94, title, fontsize=18, weight="bold", va="top")
            for index, line in enumerate(wrapped[offset:offset + 33]):
                page.text(.08, .88 - index * .024, line, fontsize=10, va="top")
            result.append(page)
        return result

    train_pages = pages("Industrial RL training report", [
        f"Status: {training['status']}; updated (Asia/Shanghai): {training['generated_at_local']}", training["objective"],
        *runtime_pdf_lines(training),
        f"Parent SFT model: {training['parent_model_path']}", f"Selected checkpoint: {summary.get('best_checkpoint')}",
        f"Selection: validation NDCG@10={summary.get('best_metric')}; never final test metrics",
        f"Epoch/step: {summary.get('epoch')} / {summary.get('global_step')}; stop: {summary.get('stop_reason')}",
        "Gradient norms are unscaled global L2 norms before clipping; skipped/nonfinite updates excluded.",
        *[f"{key}: {value}" for key, value in training["metric_statistics"].items()],
        f"Task counts: {training['config'].get('task_counts')}",
        f"Hyperparameters: {training['config'].get('hyperparameters')}", f"Errors: {training['errors']}"])
    lines = [f"RL test status: {validation['status']}; comparison status: {comparison['status']}",
             *expectation_pdf_lines(comparison["expectation_assessment"]),
             *runtime_pdf_lines(validation),
             f"Recommendation based only on validation: {comparison['recommendation']}",
             f"Parent source: {comparison['parent_sft'].get('source')}; status: {comparison['parent_sft']['status']}",
             f"Expected complete original test rows: {comparison['protocol']['expected_test_rows']}",
             *comparison["interpretation"]]
    for row in comparison["rows"]:
        lines.append(f"{row['stratum']} n={row['rows']} K={row['k']}: "
                     f"SFT R={fmt(row['sft_recall'], True)} NDCG={fmt(row['sft_ndcg'], True)} hits={row['sft_hits']}; "
                     f"RL R={fmt(row['rl_recall'], True)} NDCG={fmt(row['rl_ndcg'], True)} hits={row['rl_hits']}")
    lines += [f"Parent error: {comparison['parent_sft'].get('error')}", f"Errors: {validation['errors']}"]
    valid_pages = pages("Industrial RL validation report", lines)
    with PdfPages(destination / "training-report.pdf") as pdf:
        for figure in [*train_pages, *figures]:
            pdf.savefig(figure)
    with PdfPages(destination / "validation-report.pdf") as pdf:
        for figure in valid_pages:
            pdf.savefig(figure)
    for figure in [*train_pages, *valid_pages, *figures]:
        plt.close(figure)
    return {"training": len(train_pages) + len(figures), "validation": len(valid_pages)}


def comparison_html(comparison, include_runtime=True):
    result = "<h2>SFT / RL 完整测试对照</h2>"
    result += f"<p>状态：{html.escape(comparison['status'])}；测试指标用于最终报告。推荐仅依据完整验证集 NDCG@10。</p>"
    if include_runtime:
        result += runtime_html(comparison)
    result += "<pre>" + html.escape(str(comparison["recommendation"])) + "</pre>"
    result += expectation_html(comparison["expectation_assessment"])
    if comparison["parent_sft"].get("error"):
        result += "<p>父 SFT 校验：" + html.escape(comparison["parent_sft"]["error"]) + "</p>"
    result += "<p>HR 等同单目标 Recall；行为冷按训练商品 ID 判定。冷样本以实际行数与命中数解读；缺失值显示 —。</p>"
    result += table(["分组", "样本", "K", "SFT Recall", "RL Recall", "差值 pp", "SFT NDCG", "RL NDCG", "差值 pp", "SFT 命中", "RL 命中"], [
        (row["stratum"], row["rows"], row["k"], fmt(row["sft_recall"], True), fmt(row["rl_recall"], True),
         fmt(row["delta_recall_pp"]), fmt(row["sft_ndcg"], True), fmt(row["rl_ndcg"], True),
         fmt(row["delta_ndcg_pp"]), fmt(row["sft_hits"]), fmt(row["rl_hits"])) for row in comparison["rows"]])
    return result


def document(content, title):
    return "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" + (
        f"<title>{title}</title><style>body{{font:15px/1.6 sans-serif;background:#f4f7f8;color:#17333d}}main{{max-width:1200px;margin:auto;padding:24px}}"
        "table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d7e3e7;padding:7px;text-align:left}"
        "img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#167d8d}</style><main>" + content + "</main></html>")


def generate(run_dir, data_dir, input_dir, report_root, plots=True):
    destination = Path(report_root)
    destination.mkdir(parents=True, exist_ok=True)
    training, validation, comparison, steps = build_reports(run_dir, data_dir, input_dir)
    pages = render(training, validation, comparison, steps, destination) if plots else {"training": 0, "validation": 0}
    for report, kind in ((training, "training"), (validation, "validation")):
        report["rendered_pdf"], report["pdf_pages"] = bool(plots), pages[kind]
        write_json(destination / (kind + "_report.json"), report)
    write_json(destination / "comparison.json", comparison)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["stratum", "rows", "k", "sft_recall", "rl_recall", "delta_recall_pp",
                                              "sft_ndcg", "rl_ndcg", "delta_ndcg_pp", "sft_hits", "rl_hits"])
    writer.writeheader()
    writer.writerows(comparison["rows"])
    atomic_text(destination / "comparison.csv", buffer.getvalue())
    comp_html = comparison_html(comparison)
    atomic_text(destination / "comparison.html", document(comp_html, "Industrial SFT / RL comparison"))
    content = "<h1>Industrial_and_Scientific RL 训练报告</h1>"
    content += f"<p>训练：{training['status']}；最终测试：{validation['status']}；对照：{comparison['status']}</p>"
    content += runtime_html(training)
    links = [("training_report.json", "训练 JSON"), ("validation_report.json", "验证 JSON"),
             ("comparison.json", "对照 JSON"), ("comparison.csv", "对照 CSV"), ("comparison.html", "对照 HTML")]
    if plots:
        links = [("training-report.pdf", "训练 PDF"), ("validation-report.pdf", "验证 PDF"), *links]
    content += "<p>" + " · ".join(f"<a href='{path}'>{label}</a>" for path, label in links) + "</p>"
    content += "<p>RL 使用 reward、KL、梯度与验证排名监控；优化目标不等同 teacher-forced SFT loss。缺失指标不表示0。</p>"
    content += table(["指标", "观测数", "均值", "最近值"], [
        (key, value["count"], fmt(value["mean"]), fmt(value["last"])) for key, value in training["metric_statistics"].items()])
    if plots:
        content += "".join(f"<h3>{title}</h3><img src='{filename}' alt='{title}'>" for title, filename, _ in PLOT_SPECS)
    content += comparison_html(comparison, include_runtime=False)
    if training["errors"]:
        content += "<h2>检查问题</h2><pre>" + html.escape("\n".join(training["errors"])) + "</pre>"
    atomic_text(destination / "index.html", document(content, "Industrial RL reports"))
    return {"training_status": training["status"], "validation_status": validation["status"],
            "comparison_status": comparison["status"], "recommended_model": comparison["recommendation"]["recommended_model"],
            "training_report": str(destination / "training-report.pdf") if plots else None,
            "validation_report": str(destination / "validation-report.pdf") if plots else None,
            "comparison": str(destination / "comparison.json"), "index": str(destination / "index.html")}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "input-dir", "report-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.run_dir, args.data_dir, args.input_dir, args.report_root,
                              plots=not args.no_plots), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
