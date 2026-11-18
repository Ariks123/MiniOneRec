"""Dependency-free reports for the isolated joint generator/retriever experiment."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from qwen_retrieval.config import HISTORICAL_METRICS


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric(metrics, name):
    if name in metrics:
        return metrics[name]
    if name.startswith("Recall@"):
        return metrics.get(name.replace("Recall@", "HR@"))
    return None


def training_retrieval_monitor(run):
    records = []
    path = Path(run) / "monitor/retrieval_metrics.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    return {"records": len(records), "first": records[0] if records else None,
            "latest": records[-1] if records else None,
            "scope": "Retrieval loss is globally reduced for the last microbatch; generation token mean is rank-zero's last microbatch. These are diagnostics, not full optimizer-step averages."}


def comparison_rows(run):
    run = Path(run)
    rows = []
    sources = [
        ("historical_best_record", "test", {"status": "historical_record", "metrics": HISTORICAL_METRICS}),
        ("historical_best_beam50_retimed", "test", read_json(run / "evaluation_reference_beam/metrics.json", {})),
        ("joint_model_beam50", "test", read_json(run / "evaluation_beam/metrics.json", {})),
        ("joint_vector_retrieval", "valid", read_json(run / "evaluation_valid/metrics.json", {})),
        ("joint_vector_retrieval", "test", read_json(run / "evaluation_test/metrics.json", {})),
        ("joint_retrieval_gr_rerank", "valid", read_json(run / "evaluation_valid/metrics.json", {})),
        ("joint_retrieval_gr_rerank", "test", read_json(run / "evaluation_test/metrics.json", {})),
    ]
    timings = read_json(run / "stage_timings.json", {})
    for name, split, record in sources:
        if name == "joint_vector_retrieval":
            metrics = record.get("retrieval_metrics", {})
        elif name == "joint_retrieval_gr_rerank":
            metrics = record.get("rerank_metrics", record.get("metrics", {}))
        else:
            metrics = record.get("metrics", {})
        timing = record.get("timing", {})
        inference_seconds = timing.get("inference_seconds", timing.get("prediction_seconds"))
        if name == "joint_vector_retrieval":
            parts = [timing.get("query_seconds"), timing.get("retrieval_seconds")]
            inference_seconds = sum(parts) if all(value is not None for value in parts) else None
        stage = {"historical_best_beam50_retimed": "reference_beam", "joint_model_beam50": "beam",
                 "joint_vector_retrieval": f"evaluate_{split}",
                 "joint_retrieval_gr_rerank": f"evaluate_{split}"}.get(name)
        row = {"method": name, "split": split, "status": record.get("status", "pending"),
               "rows": record.get("completed_rows"),
               **{key: metric(metrics, key) for key in ("Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "Recall@50", "NDCG@50", "Recall@200")},
               "inference_seconds": inference_seconds,
               "stage_wall_seconds": timings.get(stage, {}).get("wall_seconds"),
               "timing_scope": ("query + retrieval + rerank" if name == "joint_retrieval_gr_rerank" else
                                "query encoding + vector matrix search components" if name == "joint_vector_retrieval" else
                                "beam prediction phase" if inference_seconds is not None else "not measured"),
               "delta_Recall@10_vs_historical": None, "delta_NDCG@10_vs_historical": None}
        if split == "test" and name != "historical_best_record":
            for key, old in (("Recall@10", HISTORICAL_METRICS["HR@10"]), ("NDCG@10", HISTORICAL_METRICS["NDCG@10"])):
                if row[key] is not None:
                    row[f"delta_{key}_vs_historical"] = row[key] - old
        rows.append(row)
    return rows


def build_report(run_dir):
    run = Path(run_dir)
    status = read_json(run / "run_status.json", {})
    training = read_json(run / "training_summary.json", {})
    isolation = read_json(run / "isolation_verification.json", {})
    smoke = read_json(run / "smoke/smoke_result.json", {})
    index = read_json(run / "item_index/manifest.json", {})
    rows = comparison_rows(run)
    artifact_checks = {
        "final_model_config": (run / "final_checkpoint/config.json").is_file(),
        "final_model_weights": (run / "final_checkpoint/model.safetensors").is_file(),
        "index_manifest": index.get("status") == "passed",
        "item_vectors": (run / "item_index/item_vectors.npy").is_file(),
        "item_mapping": (run / "item_index/row_to_items.json").is_file(),
        "evaluation_coverage": all(isinstance(row["rows"], int) and row["rows"] > 0 for row in rows[1:]),
    }
    for name in ("evaluation_valid", "evaluation_test", "evaluation_beam", "evaluation_reference_beam"):
        evaluation = read_json(run / name / "metrics.json", {})
        artifact_checks[name + "_complete"] = (evaluation.get("status") == "passed"
            and isinstance(evaluation.get("expected_rows"), int) and evaluation["expected_rows"] > 0
            and evaluation["expected_rows"] == evaluation.get("completed_rows"))
    complete = (status.get("status") != "failed" and training.get("status") == "completed" and smoke.get("status") == "passed"
                and isolation.get("status") == "passed"
                and all(row["status"] == "passed" for row in rows[1:])
                and all(artifact_checks.values()))
    return {
        "status": "passed" if complete else "incomplete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run.resolve()), "supervisor": status,
        "configuration": read_json(run / "supervisor_config.json", {}),
        "training": training, "smoke": smoke, "isolation": isolation,
        "artifact_checks": artifact_checks, "item_index": index,
        "retrieval_training_monitor": training_retrieval_monitor(run),
        "comparison": rows,
        "validation": read_json(run / "evaluation_valid/metrics.json", {}),
        "test": read_json(run / "evaluation_test/metrics.json", {}),
        "beam": read_json(run / "evaluation_beam/metrics.json", {}),
        "reference_beam": read_json(run / "evaluation_reference_beam/metrics.json", {}),
        "stage_timings": read_json(run / "stage_timings.json", {}),
        "interpretation": [
            "This is one new joint-training run. No fresh generation-only control was trained.",
            "The new run corrects DeepSpeed ZeRO-2 gradient accumulation. Historical SFT did not use this correction, so historical differences cannot isolate the retrieval objective alone.",
            "Historical best test Recall@10=0.12949481579527908 and NDCG@10=0.08300085599636473; these are comparison records, not checkpoint-selection criteria.",
            "Checkpoint selection uses validation generation loss; retrieval evaluation uses validation first and the complete test split only after training.",
            "Cold groups describe interaction cold start. Catalog-wide SID/title auxiliary supervision is retained.",
            "All candidates use the original three-token Industrial SID. Teacher-forcing reranking scores SID tokens only.",
            "Stage wall time includes process launch, loading and file work. Beam prediction time and retrieval inference time exclude model loading but include their own prediction pipelines; consult detailed component times before claiming speedups.",
        ],
    }


def write_report(run_dir, report_root):
    destination = Path(report_root)
    destination.mkdir(parents=True, exist_ok=True)
    report = build_report(run_dir)
    atomic_json(destination / "comparison.json", report)
    rows = report["comparison"]
    temporary = destination / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination / "comparison.csv")
    columns = ("method", "split", "status", "rows", "Recall@10", "NDCG@10", "Recall@50", "Recall@200", "inference_seconds", "stage_wall_seconds")
    def cell(value):
        return html.escape("—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value))
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(key)}</th>" for key in columns) + "</tr></thead><tbody>"
    table += "".join("<tr>" + "".join(f"<td>{cell(row.get(key))}</td>" for key in columns) + "</tr>" for row in rows) + "</tbody></table>"
    body = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Industrial Qwen joint retrieval</title>"
            "<style>body{font:15px system-ui;margin:32px;line-height:1.6;color:#173047}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:9px;border:1px solid #ccd5dc;text-align:left}th{background:#eef4f8}li{margin:8px 0}code{white-space:pre-wrap}</style>"
            f"<h1>Amazon 2018 Industrial &amp; Scientific · Qwen 0.5B</h1><p>Report: {html.escape(report['status'])}; stage: {html.escape(str(report['supervisor'].get('stage', 'pending')))}</p>"
            + table + "<h2>Interpretation</h2><ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in report["interpretation"])
            + "</ul><h2>Interaction cold/warm groups</h2><pre>"
            + html.escape(json.dumps(report["test"].get("groups", {}), ensure_ascii=False, indent=2))
            + "</pre><h2>Training summary</h2><pre>"
            + html.escape(json.dumps(report["training"], ensure_ascii=False, indent=2))
            + "</pre><h2>Retrieval training diagnostics</h2><pre>"
            + html.escape(json.dumps(report["retrieval_training_monitor"], ensure_ascii=False, indent=2))
            + "</pre><p><a href='comparison.json'>Detailed JSON</a> · <a href='comparison.csv'>Comparison CSV</a></p></html>")
    temporary = destination / "index.html.tmp"
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(destination / "index.html")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_report(args.run_dir, args.report_root)
    print(json.dumps({"status": report["status"], "report_root": str(args.report_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
