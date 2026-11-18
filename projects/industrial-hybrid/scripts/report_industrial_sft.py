#!/usr/bin/env python3
"""Build live and final reports for the original Industrial SFT experiment.

Metric ingestion uses only the standard library. Plotting imports matplotlib
only when requested. This module never trains a model or changes its inputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


DATASET = "Industrial_and_Scientific"
TOP_K = (1, 3, 5, 10, 20, 50)
REFERENCE = {
    "url": "https://github.com/Enddonkey/MiniOneRec-Reproduction#复现实验报告",
    "locator": "README reproduction experiment report, section 6.2, SFT Only",
    "readme_blob_sha": "369fa5824ca3bef5e2cf3d7c8ca233bd34a4d5a8",
    "method": "Enddonkey/MiniOneRec-Reproduction, Qwen2.5-0.5B Base, SFT only",
    "evidence_status": "Repository README self-report; not independently verified by this experiment",
    "reported_settings": {"model": "Qwen2.5-0.5B Base", "epochs": 3, "global_batch": 64,
                          "micro_batch": 8, "gradient_accumulation": 8,
                          "hardware": "1 × NVIDIA A800 80GB", "scheduler": "cosine with warmup (README statement)"},
    "metrics": {"HR@1": .0532, "HR@3": .0761, "HR@5": .0893, "HR@10": .1127,
                "HR@20": .1487, "HR@50": .2076, "NDCG@1": .0532, "NDCG@3": .0663,
                "NDCG@5": .0716, "NDCG@10": .0792, "NDCG@20": .0881, "NDCG@50": .0997},
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else ({} if default is None else default)


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    result = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_rows(path, catalog):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and {"item_id", "history_item_id", "item_sid"} <= set(reader.fieldnames),
                f"{path}: missing target/history columns")
        for row_id, row in enumerate(reader):
            target = str(row["item_id"])
            history = ast.literal_eval(row["history_item_id"])
            require(isinstance(history, list) and 1 <= len(history) <= 10,
                    f"{path}:{row_id}: expected 1–10 history items")
            history = [str(item) for item in history]
            require(all(item in catalog for item in [target, *history]), f"{path}:{row_id}: unknown item ID")
            require(row["item_sid"].strip() == catalog[target], f"{path}:{row_id}: target SID mismatch")
            yield {"row_id": row_id, "target_id": target, "target": catalog[target], "history": history}


def validated_evaluation(run_dir, data_dir, input_dir, config, evaluation, preparation, predictions_path=None):
    """Recompute all metrics without conflating item IDs sharing an original SID."""
    require(evaluation.get("status") == "passed" and evaluation.get("category") == DATASET,
            "Evaluation has not passed for Industrial_and_Scientific")
    require(preparation.get("dataset") == DATASET and preparation.get("validation") == "passed"
            and preparation.get("original_sid") is True, "Prepared original-data manifest is invalid")
    paths = {"train_file": data_dir / "train.csv", "eval_file": data_dir / "valid.csv", "test_file": data_dir / "test.csv",
             "sid_index_path": input_dir / f"{DATASET}.index.json",
             "item_meta_path": input_dir / f"{DATASET}.item.json"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    configured = config.get("data", {}).get("sha256", {})
    require(config.get("data", {}).get("category") == DATASET, "Training category differs")
    for key in ("train_file", "eval_file", "sid_index_path", "item_meta_path"):
        require(configured.get(key) == hashes[key], "Training source changed: " + key)
    for key in ("train_file", "test_file", "sid_index_path"):
        require(evaluation.get("source_sha256", {}).get(key) == hashes[key], "Evaluation source differs: " + key)
    require(evaluation.get("item_meta_sha256") == hashes["item_meta_path"], "Evaluation item metadata differs")
    for key, filename in (("train_file", "train.csv"), ("eval_file", "valid.csv"), ("test_file", "test.csv")):
        require(preparation.get("outputs", {}).get(filename, {}).get("sha256") == hashes[key],
                "Prepared source changed: " + filename)
    for key, source in (("sid_index_path", "sid_index"), ("item_meta_path", "item_metadata")):
        require(preparation.get("sources", {}).get(source, {}).get("sha256") == hashes[key],
                "Prepared catalog differs: " + source)
    indices, metadata = read_json(paths["sid_index_path"]), read_json(paths["item_meta_path"])
    require(isinstance(indices, dict) and indices and set(indices) == set(metadata), "Catalog/metadata IDs differ")
    require(all(isinstance(tokens, list) and tokens and all(isinstance(t, str) and t for t in tokens)
                for tokens in indices.values()), "Invalid SID token sequences")
    catalog = {key: "".join(tokens) for key, tokens in indices.items()}
    sid_counts = Counter(catalog.values())
    valid_sids = set(sid_counts)
    require(type(evaluation.get("num_beams")) is int and evaluation["num_beams"] == 50
            and len(valid_sids) >= 50, "Original evaluation requires a complete beam=50 ranking")
    require(evaluation.get("prompt_style") == "original-eval"
            and evaluation.get("validation_prompt_style") == "training",
            "Original evaluation prompt identity differs from the declared protocol")
    behavior = set()
    train_rows = 0
    for row in csv_rows(paths["train_file"], catalog):
        train_rows += 1
        behavior.update(row["history"])
        behavior.add(row["target_id"])
    tests = list(csv_rows(paths["test_file"], catalog))
    total = len(tests)
    require(total > 0 and preparation.get("rows", {}).get("test") == total
            and preparation.get("rows", {}).get("train") == train_rows, "Prepared row counts differ")
    coverage = evaluation.get("coverage", {})
    require(evaluation.get("expected_rows") == evaluation.get("completed_rows") == total
            and coverage.get("expected_rows") == coverage.get("completed_rows") == total
            and coverage.get("missing_rows") == 0 and coverage.get("duplicate_rows") == 0,
            "Evaluation coverage is incomplete")
    require(evaluation.get("CC") == 0 and evaluation.get("validity_rate") == evaluation.get("unique_rate") == 1,
            "Evaluation candidates are invalid or duplicated")
    checks = ("provenance", "prepared_input_hashes", "checkpoint_reload", "tokenizer_atomic_sid_tokens",
              "model_vocabulary", "validation_coverage", "prediction_coverage", "valid_sids", "unique_candidates")
    require(all(evaluation.get("check_results", {}).get(name) in (True, "passed") for name in checks),
            "A model or prediction integrity check did not pass")
    keys = ("overall", "behavior_warm", "behavior_cold", "repeat_last", "not_repeat_last",
            "collision_free", "collision_affected", "behavior_cold_collision_free", "behavior_cold_collision_affected")
    groups = {key: {"rows": 0, "hits": Counter(), "discounts": Counter()} for key in keys}
    warm_sids = {catalog[item] for item in behavior}
    cold_ids = set()
    cold_alias_rows = 0
    predictions_path = Path(predictions_path) if predictions_path is not None else run_dir / "evaluation/predictions.jsonl"
    prediction_hash = sha256(predictions_path)
    seen = set()
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            predicted = json.loads(line)
            require(isinstance(predicted, dict), "Prediction row must be an object")
            row_id = predicted.get("row_id")
            require(type(row_id) is int and 0 <= row_id < total and row_id not in seen,
                    "Prediction row is missing, repeated or out of range")
            seen.add(row_id)
            row = tests[row_id]
            require(predicted.get("target") == row["target"], "Prediction target/order differs")
            require(predicted.get("prompt_style") == evaluation["prompt_style"],
                    "Prediction prompt style differs from the accepted evaluation")
            candidates = predicted.get("predictions")
            require(isinstance(candidates, list) and len(candidates) == evaluation.get("num_beams")
                    and all(sid in valid_sids for sid in candidates) and len(set(candidates)) == len(candidates),
                    "Prediction candidates invalid or repeated")
            cold = row["target_id"] not in behavior
            repeat = row["target_id"] == row["history"][-1]
            collided = sid_counts[row["target"]] > 1
            selected = ["overall", "behavior_cold" if cold else "behavior_warm",
                        "repeat_last" if repeat else "not_repeat_last",
                        "collision_affected" if collided else "collision_free"]
            if cold:
                cold_ids.add(row["target_id"])
                cold_alias_rows += row["target"] in warm_sids
                selected.append("behavior_cold_collision_affected" if collided else "behavior_cold_collision_free")
            rank = candidates.index(row["target"]) + 1 if row["target"] in candidates else math.inf
            for key in selected:
                group = groups[key]
                group["rows"] += 1
                for k in TOP_K:
                    if rank <= k:
                        group["hits"][k] += 1
                        group["discounts"][k] += 1 / math.log2(rank + 1)
    require(len(seen) == total and sha256(predictions_path) == prediction_hash, "Predictions incomplete or changed")
    strata = []
    for key, group in groups.items():
        count = group["rows"]
        strata.append({"key": key, "rows": count, "share": count / total,
                       "hits_by_k": {str(k): group["hits"][k] for k in TOP_K},
                       "metrics": {f"{metric}@{k}": group[field][k] / count if count else None
                                   for metric, field in (("HR", "hits"), ("NDCG", "discounts")) for k in TOP_K}})
    for name, value in strata[0]["metrics"].items():
        expected = evaluation.get("metrics", {}).get(name)
        require(finite(expected) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-10),
                "Recomputed overall metric differs: " + name)
    require(all(sha256(paths[key]) == value for key, value in hashes.items()), "Input changed during analysis")
    return {"status": "passed", "rows": total, "train_rows": train_rows,
            "catalog_items": len(catalog), "unique_sids": len(valid_sids),
            "collision_groups": sum(n > 1 for n in sid_counts.values()),
            "collision_items": sum(n for n in sid_counts.values() if n > 1),
            "behavior_cold_items": len(cold_ids), "behavior_cold_sid_warm_alias_rows": cold_alias_rows,
            "strata": strata, "source_sha256": {**hashes, "predictions": prediction_hash},
            "prompt_style": evaluation["prompt_style"],
            "validation_prompt_style": evaluation["validation_prompt_style"],
            "num_beams": evaluation["num_beams"],
            "metric_semantics": "Original SID-level matching. Item IDs sharing a SID are indistinguishable; coldness is based on item IDs, not SID strings.",
            "interpretation": "Shared catalog/metadata; every original test row has equal weight. No candidate filtering or row removal. Cold-group estimates may be based on a small number of targets."}


def build_reports(run_dir, data_dir, input_dir):
    run_dir, data_dir, input_dir = map(Path, (run_dir, data_dir, input_dir))
    config, state, summary, monitor, evaluation, preparation = [read_json(path) for path in (
        run_dir / "run_config.json", run_dir / "run_status.json", run_dir / "training_summary.json",
        run_dir / "monitor/summary.json", run_dir / "evaluation/metrics.json", data_dir / "preparation_manifest.json")]
    rows = {}
    for observed in read_jsonl(run_dir / "monitor/metrics.jsonl"):
        step = observed.get("step", observed.get("global_step"))
        if finite(step):
            rows.setdefault(int(step), {"step": int(step)}).update({k: v for k, v in observed.items() if v is not None})
    steps = [rows[key] for key in sorted(rows)]
    norms = [row["grad_norm"] for row in steps if finite(row.get("grad_norm")) and row["grad_norm"] >= 0
             and not row.get("step_skipped") and not row.get("grad_norm_nonfinite")]
    stage = state.get("stage", "pending")
    train_status = "completed" if summary.get("status") == "completed" and norms else "failed" if (
        summary.get("status") == "failed" or state.get("status") == "failed" and stage not in ("validating", "reporting")) else "running" if stage in (
        "training", "validating", "reporting") else "pending"
    if summary.get("status") == "completed" and not norms:
        train_status = "incomplete"
    ranking_requested = (state.get("parameters", {}).get("final_ranking_eval", True) is not False
                         and evaluation.get("status") != "not_requested")
    analysis = {"status": "pending", "strata": []}
    errors = []
    if not ranking_requested:
        # A loss-only run may share a report directory with stale ranking
        # artifacts. Its explicit request takes precedence over those files.
        evaluation = {"status": "not_requested", "scope": "no_ranking_requested",
                      "reason": evaluation.get("reason", "Ranking metrics disabled by user; validation loss only")}
        loss = summary.get("final_eval_metrics", {}).get("eval_loss")
        if not finite(loss):
            loss = next((row["eval_loss"] for row in reversed(steps) if finite(row.get("eval_loss"))), None)
        analysis = {"status": "not_requested", "strata": [], "scope": "no_ranking_requested",
                    "reason": evaluation["reason"], "validation_loss": loss,
                    "validation_samples": config.get("data", {}).get("validation_samples", preparation.get("rows", {}).get("valid")),
                    "selection_metric": "eval_loss", "selected_checkpoint": summary.get("best_checkpoint")}
    elif evaluation.get("status") == "passed":
        try:
            analysis = validated_evaluation(run_dir, data_dir, input_dir, config, evaluation, preparation)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            analysis = {"status": "incomplete", "strata": [], "error": str(error)}
    elif evaluation.get("status") == "failed" or state.get("status") == "failed" and stage == "validating":
        analysis = {"status": "failed", "strata": [], "error": evaluation.get("error", state.get("error"))}
    elif stage == "validating":
        analysis["status"] = "running"
    timestamp = datetime.now(timezone.utc)
    common = {"generated_at_utc": timestamp.isoformat(), "generated_at_shanghai": timestamp.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
              "dataset": DATASET, "run_directory": str(run_dir), "run_status": state,
              "base_model": config.get("model", {}).get("source"), "errors": errors, "rl_started": False,
              "ranking_requested": ranking_requested}
    training = {**common, "status": train_status, "config": config, "preparation": preparation,
                "training_summary": summary, "monitor_summary": monitor,
                "grad_norm": {"count": len(norms), "mean": mean(norms) if norms else None,
                              "max": max(norms) if norms else None, "last100_mean": mean(norms[-100:]) if norms else None},
                "baseline_validation": read_json(run_dir / "baseline_validation.json"),
                "ranking_selection": read_json(run_dir / "ranking_selection.json") if ranking_requested else {},
                "recall_evaluation": read_json(run_dir / "evaluation_recall/metrics.json") if ranking_requested else {}}
    model_label = Path(str(common["base_model"] or state.get("parameters", {}).get("base_model", "pending"))).name
    world_size = config.get("hyperparameters", {}).get("world_size", state.get("parameters", {}).get("world_size", "pending"))
    validation = {**common, **analysis, "evaluation": evaluation,
                  "comparison_reference": REFERENCE if ranking_requested else None,
                  "reference_comparison_note": f"The benchmark is Enddonkey/MiniOneRec-Reproduction README SFT Only, self-reported and not independently verified. It uses Qwen2.5-0.5B Base on one A800; this run records {model_label} on {world_size} H20 GPUs. Its actual run_config.json governs epochs, tasks, batch and scheduler. Score differences do not establish an exact reproduction." if ranking_requested else "Ranking comparison was not requested; model selection uses validation loss."}
    return training, validation, steps


def fmt(value, percent=False):
    if value is None:
        return "—"
    if finite(value):
        return f"{value * 100:.5f}%" if percent else f"{value:.6g}"
    return str(value)


def table(headers, rows):
    return "<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</table>"


def render(training, validation, steps, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    figures = []
    plot_specs = [
        ("Training / validation loss", "loss.png", [("train", "loss", "#167d8d"), ("validation", "eval_loss", "#c17b2d")], "Loss"),
        ("Global gradient L2 norm before clipping", "grad_norm.png", [("gradient norm", "grad_norm", "#167d8d")], "L2 norm"),
        ("Learning rate", "learning_rate.png", [("learning rate", "learning_rate", "#167d8d")], "Learning rate")]
    if validation.get("ranking_requested", True) and any(finite(row.get("eval_ndcg10")) for row in steps):
        plot_specs.append(("Full validation ranking", "validation_ranking.png",
                           [("NDCG@10", "eval_ndcg10", "#167d8d"), ("Recall@10", "eval_recall10", "#c17b2d")], "Score"))
    for title, name, series, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(10.8, 4.6), layout="constrained")
        for label, key, color in series:
            points = [(row["step"], row[key]) for row in steps if finite(row.get(key))]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], label=label, color=color)
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, "Awaiting observations", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="Optimizer update step", ylabel=ylabel)
        ax.grid(alpha=.3)
        fig.savefig(destination / name, dpi=150)
        figures.append(fig)
    fig = None
    if validation.get("ranking_requested", True):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
        groups = {row["key"]: row for row in validation.get("strata", [])}
        for ax, metric in zip(axes, ("HR", "NDCG")):
            if validation["status"] == "passed":
                for key, color in (("overall", "#153b52"), ("behavior_warm", "#167d8d"), ("behavior_cold", "#c17b2d")):
                    row = groups[key]
                    if row["rows"]:
                        ax.plot(TOP_K, [row["metrics"][f"{metric}@{k}"] for k in TOP_K], marker="o",
                                label=f"{key} (n={row['rows']})", color=color)
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "Awaiting full accepted test evaluation", transform=ax.transAxes, ha="center", wrap=True)
            ax.set(title=metric, xlabel="K", ylabel=metric, xticks=TOP_K, ylim=(0, None))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.3)
        fig.suptitle("Industrial and Scientific: original SID-level full-catalog evaluation")
        fig.savefig(destination / "ranking.png", dpi=150)

    def pages(title, lines):
        import textwrap
        lines = [part for line in lines for part in (textwrap.wrap(str(line), 95) or [""])]
        result = []
        for offset in range(0, len(lines), 32):
            page = plt.figure(figsize=(8.27, 11.69))
            page.text(.08, .94, title, fontsize=18, weight="bold", va="top")
            for index, line in enumerate(lines[offset:offset + 32]):
                page.text(.08, .88 - index * .024, line, fontsize=10, va="top")
            result.append(page)
        return result
    summary = training["training_summary"]
    cover = pages("Industrial SFT training report", [f"Status: {training['status']}", f"Updated: {training['generated_at_shanghai']}",
        f"Model: {training['base_model']}", "Original dataset/SID and history lengths 1-10 are preserved.",
        f"Best checkpoint: {summary.get('best_checkpoint')}",
        f"Best validation {summary.get('selection_metric', 'eval_loss')}: {summary.get('best_metric')}",
        f"Stop reason: {summary.get('stop_reason')}", f"Gradient norm statistics: {training['grad_norm']}",
        "Gradients are unscaled global full-model L2 norms before clipping; not averaged across ranks.",
        f"Task counts: {training['config'].get('task_counts')}", f"Hyperparameters: {training['config'].get('hyperparameters')}"])
    with PdfPages(destination / "training-report.pdf") as pdf:
        for page in [*cover, *figures]:
            pdf.savefig(page)
    lines = [f"Status: {validation['status']}", f"Updated: {validation['generated_at_shanghai']}",
             f"Model: {validation['base_model']}", "Metrics are averaged over original rows; no test-row deduplication.",
             "HR@K equals Recall@K because every row has one relevant target.",
             "Behavior-cold means target item ID absent from all training histories and targets.",
             f"Catalog items: {validation.get('catalog_items')}; distinct SIDs: {validation.get('unique_sids')}",
             f"SID collision groups/items: {validation.get('collision_groups')} / {validation.get('collision_items')}",
             f"Cold unique item IDs: {validation.get('behavior_cold_items')}; cold rows aliasing warm SIDs: {validation.get('behavior_cold_sid_warm_alias_rows')}",
             "Item IDs sharing an original SID are indistinguishable. Original SID matching is preserved.", ""]
    for row in validation.get("strata", []):
        lines.append(f"{row['key']}: rows={row['rows']}")
        for k in TOP_K:
            lines.append(f"K={k}: HR={fmt(row['metrics'][f'HR@{k}'], True)}; NDCG={fmt(row['metrics'][f'NDCG@{k}'], True)}; hits={row['hits_by_k'][str(k)]}")
        lines.append("")
    lines += [validation["reference_comparison_note"],
              f"Source: https://github.com/Enddonkey/MiniOneRec-Reproduction/blob/main/README.md ({REFERENCE['locator']})",
              "Repository README SFT-only results: " + str(REFERENCE["metrics"]),
              "Reported settings: " + str(REFERENCE["reported_settings"]),
              f"Errors: {validation['errors']}"]
    if not validation.get("ranking_requested", True):
        lines = [f"Training status: {training['status']}", f"Updated: {validation['generated_at_shanghai']}",
                 f"Model: {validation['base_model']}", "Ranking evaluation: not requested by the user.",
                 "Early stopping and model selection use validation loss only.",
                 f"Validation rows: {validation.get('validation_samples')}",
                 f"Validation loss at the selected model: {validation.get('validation_loss')}",
                 f"Selected checkpoint: {validation.get('selected_checkpoint')}",
                 f"Reason: {validation.get('reason')}", f"Errors: {validation['errors']}"]
    covers = pages("Industrial SFT validation report", lines)
    ranking_figures = [fig] if fig is not None else []
    with PdfPages(destination / "validation-report.pdf") as pdf:
        for page in [*covers, *ranking_figures]:
            pdf.savefig(page)
    page_counts = {"training": len(cover) + len(figures), "validation": len(covers) + len(ranking_figures)}
    for page in [*cover, *covers, *figures, *ranking_figures]:
        plt.close(page)
    return page_counts


def generate(run_dir, data_dir, input_dir, report_root, plots=True):
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    training, validation, steps = build_reports(run_dir, data_dir, input_dir)
    pages = render(training, validation, steps, root) if plots else {"training": 0, "validation": 0}
    training["rendered_pdf"] = validation["rendered_pdf"] = bool(plots)
    training["pdf_pages"], validation["pdf_pages"] = pages["training"], pages["validation"]
    write_json(root / "training_report.json", training)
    write_json(root / "validation_report.json", validation)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["stratum", "rows", "share", "k", "HR", "NDCG", "hits"])
    writer.writeheader()
    for row in validation.get("strata", []):
        for k in TOP_K:
            writer.writerow({"stratum": row["key"], "rows": row["rows"], "share": row["share"], "k": k,
                             "HR": row["metrics"][f"HR@{k}"], "NDCG": row["metrics"][f"NDCG@{k}"], "hits": row["hits_by_k"][str(k)]})
    ranking_requested = validation.get("ranking_requested", True)
    if ranking_requested:
        atomic_text(root / "conditional_metrics.csv", output.getvalue())
    content = "<h1>Industrial_and_Scientific 原项目 SFT 实验</h1>"
    content += f"<p>北京时间 {training['generated_at_shanghai']} · 训练：{training['status']} · 排名评估：{validation['status']}</p>"
    if ranking_requested:
        content += "<p>复用原始 SID 与数据划分，历史长度1–10；全量商品目录、beam=50、逐原始测试行计算。未运行 RL。</p>"
    else:
        content += "<p>本次只进行验证 loss 评估，并按验证 loss 早停和保存最佳模型。未请求排名评估；没有计算排名指标。</p>"
    content += "<p><a href='training-report.pdf'>训练 PDF</a> · <a href='validation-report.pdf'>验证 PDF</a> · <a href='training_report.json'>训练 JSON</a> · <a href='validation_report.json'>验证 JSON</a>"
    content += " · <a href='conditional_metrics.csv'>分组 CSV</a></p>" if ranking_requested else "</p>"
    content += "<h2>训练</h2>" + table(["项目", "记录"], [("底模", training["base_model"]), ("训练状态", training["status"]),
        ("最佳 checkpoint", training["training_summary"].get("best_checkpoint")), ("梯度统计", training["grad_norm"])])
    for name, title in (("loss.png", "训练与验证 loss"), ("grad_norm.png", "梯度范数"), ("learning_rate.png", "学习率")):
        content += f"<h3>{title}</h3><img src='{name}' alt='{title}'>"
    if ranking_requested and (root / "validation_ranking.png").is_file():
        content += "<h3>完整验证集 Recall / NDCG</h3><img src='validation_ranking.png' alt='完整验证集Recall与NDCG'>"
    if training.get("ranking_selection"):
        content += "<h3>验证集最佳模型记录</h3><pre>" + html.escape(json.dumps(training["ranking_selection"], ensure_ascii=False, indent=2)) + "</pre>"
    recall_evaluation = training.get("recall_evaluation", {})
    if validation.get("status") == "passed" and recall_evaluation.get("status") == "passed":
        content += "<h3>按验证指标选出的两个模型：最终测试结果</h3>"
        content += table(["模型选择依据", "测试Recall@10", "测试NDCG@10", "测试Recall@50"], [
            (label, fmt(record["metrics"]["HR@10"], True), fmt(record["metrics"]["NDCG@10"], True), fmt(record["metrics"]["HR@50"], True))
            for label, record in (("验证NDCG@10", validation["evaluation"]), ("验证Recall@10", recall_evaluation))])
    if not ranking_requested:
        content += "<h2>验证 loss</h2><p>排名评估未请求，这不是待完成或失败状态，也不表示排名指标为0。</p>"
        content += table(["项目", "记录"], [
            ("验证样本数", validation.get("validation_samples")),
            ("所选模型的验证 loss", validation.get("validation_loss")),
            ("模型选择与早停指标", validation.get("selection_metric")),
            ("最佳 checkpoint", validation.get("selected_checkpoint"))])
    else:
        content += "<h2>整体、冷暖组与 SID 冲突诊断</h2><p>行为冷按训练行为中的商品 ID 判定，商品仍可参加全目录 SID↔文本辅助训练。HR 在单目标协议下等同 Recall。</p>"
        if validation["status"] != "passed":
            content += "<p class='warning'>完整测试结果尚未通过。当前为空的排名指标不代表0。</p>"
        else:
            content += table(["项目", "实际值"], [("完整测试行", validation["rows"]), ("商品数 / 唯一 SID", f"{validation['catalog_items']} / {validation['unique_sids']}"),
                ("冲突 SID 组 / 涉及商品", f"{validation['collision_groups']} / {validation['collision_items']}"),
                ("行为冷商品数", validation["behavior_cold_items"]), ("冷目标与暖商品共用 SID 的行数", validation["behavior_cold_sid_warm_alias_rows"])])
        content += table(["分组", "样本数", "HR@10", "NDCG@10", "HR@50"], [(row["key"], row["rows"],
            fmt(row["metrics"]["HR@10"], True), fmt(row["metrics"]["NDCG@10"], True), fmt(row["metrics"]["HR@50"], True)) for row in validation.get("strata", [])])
        content += "<p>原始 SID 冲突意味着部分商品不可区分；主指标沿用原项目 SID 匹配，同时给出无冲突子组。冷暖、重复与冲突分层维度会重叠，不能把所有行数相加。</p><img src='ranking.png' alt='整体和冷暖商品 HR NDCG 随 K 变化'>"
        content += f"<h2>复现仓库 SFT-only 对照</h2><p>{html.escape(validation['reference_comparison_note'])}</p><p>参考值来自仓库 README 自报结果，未经本实验独立复核；本次底模为 {html.escape(str(training['base_model']))}，GPU与实际训练设置差异需一并比较。</p><p><a href='{REFERENCE['url']}'>Enddonkey/MiniOneRec-Reproduction 复现实验报告</a></p>"
        content += table(["K", "仓库 SFT-only HR", "仓库 SFT-only NDCG"], [(k, fmt(REFERENCE["metrics"][f"HR@{k}"], True), fmt(REFERENCE["metrics"][f"NDCG@{k}"], True)) for k in TOP_K])
    if validation["errors"]:
        content += "<h2>检查问题</h2><pre>" + html.escape("\n".join(validation["errors"])) + "</pre>"
    atomic_text(root / "index.html", "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Industrial SFT reports</title><style>body{font:16px/1.6 sans-serif;color:#17333d;background:#f4f7f8;margin:0}main{max-width:1100px;margin:auto;padding:28px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #d7e3e7;text-align:left;overflow-wrap:anywhere}img{max-width:100%;background:white}a{color:#167d8d}.warning{background:#fff0cd;padding:14px}pre{white-space:pre-wrap}</style><main>" + content + "</main></html>")
    return {"training_status": training["status"], "validation_status": validation["status"],
            "training_report": str(root / "training-report.pdf"), "validation_report": str(root / "validation-report.pdf"),
            "index": str(root / "index.html"), "cold_rows": next((row["rows"] for row in validation.get("strata", []) if row["key"] == "behavior_cold"), None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "input-dir", "report-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.run_dir, args.data_dir, args.input_dir, args.report_root,
                              plots=not args.no_plots), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
