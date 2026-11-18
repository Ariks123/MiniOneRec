#!/usr/bin/env python3
"""Verify and compare the Industrial beta=0.001/0.01 GRPO experiments.

Inputs are read only. Full saved test predictions are revalidated with the
existing Industrial reporter; checkpoint choice uses validation NDCG@10 only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import io
import math
from pathlib import Path
from statistics import mean, median
import sys

try:
    from . import report_industrial_rl as reports
except ImportError:
    import report_industrial_rl as reports


WORKSPACE = Path("/mnt/ossdata/ziweizhao")
DATA_DIR = WORKSPACE / "MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey"
INPUT_DIR = WORKSPACE / "MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey"
EXPECTED_COUNTS = {"overall": 4533, "behavior_warm": 4386, "behavior_cold": 147}
# Lightweight baseline inputs required inside a candidate's reporting container.
# Parent caches avoid copying the original SFT checkpoints or optimizer states.
INPUT_FILES = (
    "run_config.json", "run_status.json", "supervisor_config.json", "training_summary.json",
    "baseline_validation.json", "monitor/summary.json", "monitor/metrics.jsonl",
    "evaluation/metrics.json", "evaluation/predictions.jsonl", "rl_task_manifest.jsonl",
    "parent_sft_run_config.json", "parent_sft_training_summary.json",
    "parent_sft_evaluation_metrics.json", "parent_sft_predictions.jsonl",
)
TRAINING_SOURCES = (
    "rl.py", "minionerec_trainer.py", "scripts/industrial_rl_data.py",
    "scripts/industrial_rl_validation.py", "scripts/deepspeed_zero2_compat.py", "sft_monitor.py",
    "data.py", "LogitProcessor.py", "npu_support.py", "scripts/industrial_generative_validation.py",
    "scripts/evaluate_sft.py", "scripts/sft_eval_utils.py", "config/h20_zero2.json",
)
PLOTS = (
    ("rule_reward.png", "Rule Reward (candidate exact-hit mean)", "rewards/rule_reward"),
    ("reward.png", "Combined reward (exact hit + ranking penalty)", "reward"),
    ("kl.png", "KL estimate on generated candidates", "kl"),
    ("grad_norm.png", "Global gradient L2 norm before clipping", "grad_norm"),
    ("validation_ndcg.png", "Full validation NDCG@10", "eval_ndcg10"),
)
KIND = "industrial_rl_beta_comparison"
require = reports.require


def load_run(run_dir, data_dir, input_dir):
    """Accept completed, internally verified source records; never trust a stale PDF."""
    run_dir = Path(run_dir).resolve()
    paths = {name: run_dir / name for name in INPUT_FILES}
    require(all(path.is_file() for path in paths.values()), "Incomplete beta comparison inputs: " + str(run_dir))
    fingerprints = {name: reports.sha256(path) for name, path in paths.items()}
    training, validation, comparison, steps = reports.build_reports(run_dir, data_dir, input_dir)
    require(training.get("status") == "completed" and training.get("run_status", {}).get("status") == "completed",
            "RL run is not completed: " + str(run_dir))
    require(validation.get("status") == comparison.get("status") == "passed" and not training.get("errors"),
            "RL prediction/provenance validation did not pass: " + str(run_dir) + ": " + str(training.get("errors")))
    require(comparison.get("recommendation", {}).get("status") == "passed",
            "Full-validation checkpoint selection has not passed")
    require(all(reports.sha256(path) == fingerprints[name] for name, path in paths.items()),
            "RL source records changed during comparison")
    return {"run_dir": str(run_dir), "config": training["config"],
            "supervisor": reports.read_json(run_dir / "supervisor_config.json"),
            "summary": training["training_summary"], "state": training["run_status"],
            "validation": validation, "steps": steps, "input_sha256": fingerprints}


def _same(left, right, message):
    require(left is not None and left == right, message)
    return left


def verify_pair(baseline, candidate):
    """Require a beta-only training contrast, including the actual recorded arguments."""
    configs = [run["config"] for run in (baseline, candidate)]
    supers = [run["supervisor"] for run in (baseline, candidate)]
    hypers = [config.get("hyperparameters", {}) for config in configs]
    for run, expected in ((baseline, .001), (candidate, .01)):
        config, supervisor = run["config"], run["supervisor"]
        beta = config.get("hyperparameters", {}).get("beta")
        require(reports.finite(beta) and math.isclose(beta, expected, rel_tol=0, abs_tol=1e-12),
                f"Expected actual training beta={expected}, observed {beta}")
        require(supervisor.get("beta") == beta, "Supervisor beta differs from actual RL beta")
        require(config.get("stage") == "rl" and config.get("resume_from_checkpoint") is None,
                "Beta comparison requires fresh RL runs from the shared SFT export")
        require(config.get("hyperparameters", {}).get("model_selection_metric") == "eval_ndcg10",
                "Checkpoint selection must use validation NDCG@10")
        patch = config.get("runtime_compatibility", {})
        require(patch.get("applied") is True and patch.get("installed_package_modified") is False,
                "Both beta runs must use the process-local DeepSpeed accumulation correction")
        image = supervisor.get("image")
        require(isinstance(image, str) and image.startswith("sha256:")
                and image == supervisor.get("parent_runtime_image_id") == run["state"].get("runtime_image_id"),
                "Runtime image identity is incomplete or inconsistent")
        parent = config.get("parent_sft", {})
        require(parent.get("model_sha256") == supervisor.get("parent_model_sha256")
                == config.get("model", {}).get("identity", {}).get("files", {}).get("model.safetensors", {}).get("sha256")
                and isinstance(parent.get("model_sha256"), str), "Parent SFT weight identity is inconsistent")
        manifest = config.get("data", {}).get("task_manifest", {})
        manifest_path = Path(run["run_dir"]) / "rl_task_manifest.jsonl"
        require(Path(manifest.get("path", "")).resolve() == manifest_path.resolve()
                and reports.sha256(manifest_path) == manifest.get("sha256"), "RL task manifest differs from its recorded identity")
        strata = {row["key"]: row for row in run["validation"]["strata"]}
        require(all(strata.get(key, {}).get("rows") == count for key, count in EXPECTED_COUNTS.items()),
                "Expected complete Industrial test coverage (4533 total, 147 behavior-cold)")
        require(run["summary"].get("selection_metric") == "eval_ndcg10"
                and run["summary"].get("final_checkpoint_contains_best_model") is True,
                "Final export is not the selected validation-NDCG checkpoint")
        score = run["summary"].get("best_metric")
        require(reports.finite(score) and 0 <= score <= 1, "Selected validation NDCG is invalid")
    _same({key: value for key, value in hypers[0].items() if key != "beta"},
          {key: value for key, value in hypers[1].items() if key != "beta"}, "Training hyperparameters differ beyond beta")
    require(hypers[0].get("num_generations") == 16, "Rule Reward comparison requires the shared G=16 recipe")
    for key in ("model", "parent_sft", "task_counts"):
        _same(configs[0].get(key), configs[1].get(key), "RL configuration differs: " + key)
    left_data, right_data = [config.get("data", {}) for config in configs]
    _same({key: value for key, value in left_data.items() if key != "task_manifest"},
          {key: value for key, value in right_data.items() if key != "task_manifest"}, "Training data or task composition differs")
    _same({key: value for key, value in left_data["task_manifest"].items() if key != "path"},
          {key: value for key, value in right_data["task_manifest"].items() if key != "path"}, "RL task manifest identities differ")
    image = _same(supers[0].get("image"), supers[1].get("image"), "Beta runs used different runtime images")
    _same(baseline["state"].get("runtime_versions"), candidate["state"].get("runtime_versions"),
          "Beta runs used different runtime package/device versions")
    for key in ("version", "original_source_sha256", "patched_source_sha256", "method"):
        _same(configs[0]["runtime_compatibility"].get(key), configs[1]["runtime_compatibility"].get(key),
              "DeepSpeed accumulation correction differs: " + key)
    for name in TRAINING_SOURCES:
        _same(supers[0].get("source_sha256", {}).get(name), supers[1].get("source_sha256", {}).get(name),
              "Training/evaluation code differs: " + name)
    sources = [{key: value for key, value in run["validation"].get("source_sha256", {}).items()
                if key != "predictions"} for run in (baseline, candidate)]
    require(sources[0], "Validated evaluation source hashes are missing")
    _same(sources[0], sources[1], "Validated test input identities differ")
    return {"status": "passed", "only_changed_training_hyperparameter": "beta", "runtime_image": image,
            "parent_model_sha256": configs[0]["parent_sft"]["model_sha256"],
            "task_manifest_sha256": left_data["task_manifest"]["sha256"],
            "training_sources": {key: supers[0]["source_sha256"][key] for key in TRAINING_SOURCES},
            "expected_test_strata": EXPECTED_COUNTS, "num_generations": 16,
            "test_input_sha256": sources[0]}


def compare_rows(baseline, candidate):
    # Reuse the verified reporter's stratum calculation and rename its arm labels.
    rows = reports.comparison_rows(baseline["validation"], candidate["validation"])
    return [{("baseline_" + key[4:] if key.startswith("sft_") else
              "candidate_" + key[3:] if key.startswith("rl_") else key): value
             for key, value in row.items()}
            for row in rows if row["stratum"] in EXPECTED_COUNTS]


def accepted_training_rows(run):
    return [row for row in run["steps"] if reports.finite(row.get("reward"))]


def telemetry_stats(rows, clip):
    result = {}
    for key in ("rewards/rule_reward", "reward", "kl", "grad_norm", "hit_group_rate"):
        values = [row[key] for row in rows if reports.finite(row.get(key)) and (key != "grad_norm" or not (
            row.get("step_skipped") or row.get("grad_norm_nonfinite") or row.get("optimizer_overflow")))]
        result[key] = {"count": len(values), "mean": mean(values) if values else None,
                       "median": median(values) if values else None, "max": max(values) if values else None}
        if key == "grad_norm":
            result[key]["clip_exceedance_rate"] = sum(value > clip for value in values) / len(values) if values else None
    return result


def build_comparison(baseline_run, candidate_run, data_dir=DATA_DIR, input_dir=INPUT_DIR):
    baseline = load_run(baseline_run, data_dir, input_dir)
    candidate = load_run(candidate_run, data_dir, input_dir)
    require(baseline["run_dir"] != candidate["run_dir"], "Baseline and candidate must be separate runs")
    contract = verify_pair(baseline, candidate)
    clip = baseline["config"]["hyperparameters"]["max_grad_norm"]
    train_rows = [accepted_training_rows(run) for run in (baseline, candidate)]
    require(all(rows for rows in train_rows), "Both runs require accepted RL training telemetry")
    common_steps = sorted(set(row["step"] for row in train_rows[0]) & set(row["step"] for row in train_rows[1]))
    require(common_steps, "No shared optimizer steps for the beta training contrast")
    common_set = set(common_steps)
    arms = {}
    for label, run, rows in zip(("baseline", "candidate"), (baseline, candidate), train_rows):
        arms[label] = {"run_directory": run["run_dir"], "beta": run["config"]["hyperparameters"]["beta"],
                       "selected_checkpoint": run["summary"]["best_checkpoint"],
                       "selected_validation_ndcg10": run["summary"]["best_metric"],
                       "completed_steps": run["summary"].get("global_step"), "completed_epoch": run["summary"].get("epoch"),
                       "stop_reason": run["summary"].get("stop_reason"),
                       "training_runtime_seconds": run["summary"].get("train_metrics", {}).get("train_runtime"),
                       "input_sha256": run["input_sha256"],
                       "all_training_statistics": telemetry_stats(rows, clip),
                       "last100_statistics": telemetry_stats(rows[-100:], clip),
                       "matched_step_statistics": telemetry_stats([row for row in rows if row["step"] in common_set], clip)}
    before, after = [arms[label]["selected_validation_ndcg10"] for label in ("baseline", "candidate")]
    recommended = "candidate" if after > before + 1e-12 else "baseline"
    rows = compare_rows(baseline, candidate)
    kl_before, kl_after = [arms[label]["matched_step_statistics"]["kl"]["mean"] for label in ("baseline", "candidate")]
    require(reports.finite(kl_before) and reports.finite(kl_after), "Both runs require finite KL telemetry")
    primary = {row["k"]: row for row in rows if row["stratum"] == "overall"}
    quality = after >= before - 1e-12 and primary[10]["delta_ndcg_pp"] >= -1e-10 and primary[10]["delta_recall_pp"] >= -1e-10
    notes = [
        "仅 beta 改为0.01；同一SFT起点、G=16、训练数据、优化参数、运行镜像和DeepSpeed修复均已核对。",
        "预期：更强KL约束可降低概率漂移；召回和排序效果是否提高需由验证与测试数据判断。",
        "KL比较使用双方共同优化器步数；各自末100步所处参考模型同步阶段可能不同。",
        "Rule Reward为候选精确命中均值；Combined reward还包含排名惩罚，两者分别绘图。",
        "Grad Norm为裁剪前的全模型L2范数，超过0.3不代表实际更新幅度等于日志数值。",
        "测试集只作结果报告；推荐仅按完整验证集NDCG@10，平分时保留beta=0.001。",
        "行为冷商品按训练历史与目标中未出现的商品ID判定；SID别名仍可能共享。",
        "冷样本只有147条，一次命中使HR改变约0.6803个百分点；少数命中变化不证明稳定提升。",
        "每组只训练一个随机种子；此对照描述观测结果，不提供显著性或跨种子稳定性结论。",
    ]
    report = {"kind": KIND, "format_version": 1, "status": "passed", "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": reports.DATASET,
              "report_generator_sha256": reports.sha256(Path(__file__)), "contract": contract, **arms,
              "recommendation": {"basis": "complete_validation_ndcg10", "recommended_arm": recommended,
                  "recommended_beta": arms[recommended]["beta"], "difference_percentage_points": 100 * (after - before),
                  "test_used_for_selection": False, "tie_rule": "retain beta=0.001"},
              "common_training_steps": {"count": len(common_steps), "first": common_steps[0], "last": common_steps[-1]},
              "expectation_assessment": {"matched_step_kl_decreased": kl_after < kl_before,
                  "validation_and_test10_non_decreasing": quality,
                  "status": "met" if kl_after < kl_before and quality else "mixed",
                  "test_hr50_change_pp": primary[50]["delta_recall_pp"]},
              "protocol": {"test_rows": 4533, "validation_rows": 4532, "num_beams": 50,
                  "prompt_style": "original-eval", "test_used_for_selection": False},
              "rows": rows, "interpretation": notes}
    return reports.json_safe(report), baseline, candidate


def render_plots(report, baseline, candidate, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import deque
    colors = ("#2166ac", "#b35806")
    for filename, title, key in PLOTS:
        fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
        for label, run, color in zip(("baseline", "candidate"), (baseline, candidate), colors):
            points = [(row["step"], row[key]) for row in run["steps"] if reports.finite(row.get(key)) and (
                key != "grad_norm" or not (row.get("step_skipped") or row.get("optimizer_overflow") or row.get("grad_norm_nonfinite")))]
            require(points, "Missing observations required for plot: " + key)
            x, y = zip(*points)
            beta = report[label]["beta"]
            if key == "eval_ndcg10":
                ax.plot(x, y, marker="o", color=color, label=f"beta={beta:g}")
                selected = str(run["summary"]["best_checkpoint"]).rsplit("-", 1)[-1]
                require(selected.isdigit(), "Invalid selected-checkpoint step")
                ax.scatter([int(selected)], [run["summary"]["best_metric"]], color=color,
                           marker="*", s=155, zorder=5, label=f"beta={beta:g} selected")
            else:
                ax.plot(x, y, color=color, alpha=.22, linewidth=.65)
                window, smoothed = deque(maxlen=100), []
                for value in y:
                    window.append(value)
                    smoothed.append(mean(window))
                ax.plot(x, smoothed, color=color, label=f"beta={beta:g}, rolling <=100 updates")
        if key == "kl":
            period = baseline["config"]["hyperparameters"]["ref_model_sync_steps"]
            for step in range(period, max(baseline["summary"]["global_step"], candidate["summary"]["global_step"]) + 1, period):
                ax.axvline(step, color="#667", alpha=.22, linestyle="--")
        if key == "grad_norm":
            ax.axhline(baseline["config"]["hyperparameters"]["max_grad_norm"], color="#667", linestyle="--", label="clipping threshold")
        ax.set(title=title, xlabel="Optimizer update step", ylabel=key)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)


def html_report(report, plots):
    content = "<h1>Industrial RL：beta=0.001 与 beta=0.01 对照</h1>"
    content += "<p>已重算双方全部测试预测并校验输入、起点、训练参数及运行环境。差值单位为百分点。</p>"
    content += "<p>推荐 beta=" + str(report["recommendation"]["recommended_beta"]) + "，仅依据验证集 NDCG@10。</p>"
    content += reports.table(["实验", "beta", "训练步数", "epoch", "所选检查点", "验证NDCG@10", "训练秒数"], [
        (label, report[label]["beta"], report[label]["completed_steps"], reports.fmt(report[label]["completed_epoch"]),
         Path(report[label]["selected_checkpoint"]).name, reports.fmt(report[label]["selected_validation_ndcg10"], True),
         reports.fmt(report[label]["training_runtime_seconds"])) for label in ("baseline", "candidate")])
    content += "<h2>观测与解释</h2><ul>" + "".join("<li>" + html.escape(note) + "</li>" for note in report["interpretation"]) + "</ul>"
    assessment = report["expectation_assessment"]
    content += "<p>共同训练步区间：" + html.escape(str(report["common_training_steps"])) + "；KL均值降低：" + str(assessment["matched_step_kl_decreased"]) + "；验证与测试@10均未下降：" + str(assessment["validation_and_test10_non_decreasing"]) + "。</p>"
    content += "<h2>同一步数区间训练统计</h2>"
    content += reports.table(["指标", "beta=0.001均值", "beta=0.01均值", "beta=0.001最大", "beta=0.01最大"], [
        (key, reports.fmt(report["baseline"]["matched_step_statistics"][key]["mean"]),
         reports.fmt(report["candidate"]["matched_step_statistics"][key]["mean"]),
         reports.fmt(report["baseline"]["matched_step_statistics"][key]["max"]),
         reports.fmt(report["candidate"]["matched_step_statistics"][key]["max"]))
        for key in ("rewards/rule_reward", "reward", "kl", "grad_norm", "hit_group_rate")])
    content += "<h2>完整测试与冷商品</h2>"
    content += reports.table(["分组", "样本", "K", "0.001 HR", "0.01 HR", "变化pp", "0.001 NDCG", "0.01 NDCG", "变化pp", "0.001命中", "0.01命中"], [
        (row["stratum"], row["rows"], row["k"], reports.fmt(row["baseline_recall"], True), reports.fmt(row["candidate_recall"], True),
         reports.fmt(row["delta_recall_pp"]), reports.fmt(row["baseline_ndcg"], True), reports.fmt(row["candidate_ndcg"], True),
         reports.fmt(row["delta_ndcg_pp"]), row["baseline_hits"], row["candidate_hits"]) for row in report["rows"]])
    if plots:
        content += "<h2>训练曲线</h2><p>细线为原始观测，粗线为最多100步移动均值。验证曲线仅包含训练时评估，星标为所选检查点；训练结束后的重载评估不进入曲线。KL虚线标记参考模型同步。</p>"
        content += "".join(f"<h3>{html.escape(title)}</h3><img src='{filename}' alt='{html.escape(title)}'>" for filename, title, _ in PLOTS)
    content += "<p><a href='comparison.json'>完整JSON与校验哈希</a> · <a href='comparison.csv'>逐指标CSV</a></p>"
    return reports.document(content, "Industrial RL beta comparison")


def generate(baseline_run, candidate_run, report_root, data_dir=DATA_DIR, input_dir=INPUT_DIR, plots=True):
    destination = Path(report_root).resolve()
    for source in map(lambda value: Path(value).resolve(), (baseline_run, candidate_run, data_dir, input_dir)):
        require(destination != source and not destination.is_relative_to(source), "Report output must be outside every input directory")
    report, baseline, candidate = build_comparison(baseline_run, candidate_run, data_dir, input_dir)
    if destination.exists() and any(destination.iterdir()):
        previous = reports.read_json(destination / "comparison.json")
        require(previous.get("kind") == KIND and all(previous.get(label, {}).get("run_directory") == report[label]["run_directory"]
                for label in ("baseline", "candidate")), "Existing output belongs to another report; choose a new report-root")
    destination.mkdir(parents=True, exist_ok=True)
    if plots:
        render_plots(report, baseline, candidate, destination)
    report["rendered_plots"] = [filename for filename, _, _ in PLOTS] if plots else []
    reports.write_json(destination / "comparison.json", report)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(report["rows"][0]))
    writer.writeheader()
    writer.writerows(report["rows"])
    reports.atomic_text(destination / "comparison.csv", buffer.getvalue())
    content = html_report(report, plots)
    reports.atomic_text(destination / "comparison.html", content)
    reports.atomic_text(destination / "index.html", content)
    return {"status": "passed", "recommended_beta": report["recommendation"]["recommended_beta"],
            "comparison": str(destination / "comparison.json"), "index": str(destination / "index.html")}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--no-plots", action="store_true", help="Validate and write tables without importing matplotlib")
    args = parser.parse_args(argv)
    try:
        result = generate(args.baseline_run, args.candidate_run, args.report_root, args.data_dir, args.input_dir, not args.no_plots)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print("Beta comparison failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
