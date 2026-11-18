#!/usr/bin/env python3
"""Compare archived Industrial SFT/RL predictions with two verified SFT runs.

Scoring and integrity checks use the standard library. Matplotlib is imported
only for the standalone comparison figure. Input files are never changed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path


DATASET = "Industrial_and_Scientific"
KS = (1, 3, 5, 10, 20, 50)
PREFIX = "industrial_reproduction_comparison"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def shared_data(data_dir, input_dir, expected_cold_rows, expected_cold_items):
    paths = {"train_file": data_dir / "train.csv", "eval_file": data_dir / "valid.csv",
             "test_file": data_dir / "test.csv", "sid_index_path": input_dir / f"{DATASET}.index.json",
             "item_meta_path": input_dir / f"{DATASET}.item.json"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    manifest = load(data_dir / "preparation_manifest.json")
    require(manifest.get("validation") == "passed" and manifest.get("dataset") == DATASET
            and manifest.get("original_sid") is True, "Invalid original-data preparation manifest")
    for key, filename in (("train_file", "train.csv"), ("eval_file", "valid.csv"), ("test_file", "test.csv")):
        require(manifest.get("outputs", {}).get(filename, {}).get("sha256") == hashes[key], "Prepared file changed: " + filename)
    for key, name in (("sid_index_path", "sid_index"), ("item_meta_path", "item_metadata")):
        require(manifest.get("sources", {}).get(name, {}).get("sha256") == hashes[key], "Prepared catalog changed: " + name)
    indices, metadata = load(paths["sid_index_path"]), load(paths["item_meta_path"])
    require(isinstance(indices, dict) and indices and set(indices) == set(metadata), "Catalog IDs differ from metadata")
    require(all(isinstance(tokens, list) and tokens and all(isinstance(t, str) and t for t in tokens)
                for tokens in indices.values()), "Invalid SID catalog")
    catalog = {key: "".join(tokens) for key, tokens in indices.items()}

    def rows(path):
        result = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                parsed = ast.literal_eval(row["history_item_id"])
                require(isinstance(parsed, list), "History item IDs must be a list")
                history = [str(item) for item in parsed]
                target = row["item_id"]
                require(1 <= len(history) <= 10 and all(item in catalog for item in [target, *history]), "Invalid target/history IDs")
                require(row["item_sid"].strip() == catalog[target], "Target SID differs from item ID")
                if row.get("history_item_sid"):
                    require(ast.literal_eval(row["history_item_sid"]) == [catalog[item] for item in history], "History SID differs from item IDs")
                result.append({"item_id": target, "target": catalog[target], "history": history})
        return result

    training, tests = rows(paths["train_file"]), rows(paths["test_file"])
    require(len(training) == manifest.get("rows", {}).get("train") and len(tests) == manifest.get("rows", {}).get("test"), "Prepared row counts differ")
    known = {item for row in training for item in [row["item_id"], *row["history"]]}
    sid_counts = Counter(catalog.values())
    warm_sids = {catalog[item] for item in known}
    cold = [row for row in tests if row["item_id"] not in known]
    require(len(cold) == expected_cold_rows and len({row["item_id"] for row in cold}) == expected_cold_items,
            "Behavior-cold cohort differs from the declared original split")
    cohort = {"test_rows": len(tests), "train_rows": len(training), "train_behavior_items": len(known),
              "warm_rows": len(tests) - len(cold), "cold_rows": len(cold), "cold_items": len({row["item_id"] for row in cold}),
              "cold_sid_warm_alias_rows": sum(row["target"] in warm_sids for row in cold),
              "cold_sid_collision_rows": sum(sid_counts[row["target"]] > 1 for row in cold),
              "catalog_items": len(catalog), "unique_sids": len(sid_counts),
              "collision_groups": sum(n > 1 for n in sid_counts.values()),
              "collision_items": sum(n for n in sid_counts.values() if n > 1)}
    return paths, hashes, catalog, known, tests, cohort


def aggregate(predictions, tests, known, catalog):
    groups = {key: [] for key in ("overall", "warm", "cold")}
    valid = set(catalog.values())
    seen = set()
    for row_id, target, candidates in predictions:
        require(type(row_id) is int and 0 <= row_id < len(tests) and row_id not in seen, "Missing, duplicate or invalid prediction row ID")
        seen.add(row_id)
        row = tests[row_id]
        require(target == row["target"], "Prediction target/order differs from the test data")
        require(isinstance(candidates, list) and len(candidates) == 50 and len(set(candidates)) == 50
                and all(sid in valid for sid in candidates), "Candidates must contain 50 unique catalog SIDs")
        rank = candidates.index(target) + 1 if target in candidates else math.inf
        record = (row["item_id"], rank)
        groups["overall"].append(record)
        groups["warm" if row["item_id"] in known else "cold"].append(record)
    require(len(seen) == len(tests), "Prediction coverage is incomplete")
    strata = {}
    for name, rows in groups.items():
        hits, unique, metrics = {}, {}, {}
        for k in KS:
            selected = [(item, rank) for item, rank in rows if rank <= k]
            hits[str(k)] = len(selected)
            unique[str(k)] = len({item for item, _ in selected})
            metrics[f"HR@{k}"] = len(selected) / len(rows) if rows else None
            metrics[f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for _, rank in selected) / len(rows) if rows else None
        strata[name] = {"rows": len(rows), "hits_by_k": hits, "hit_target_item_ids_by_k": unique, "metrics": metrics}
    return strata


def archived_result(path, stage, tests, known, catalog):
    before = sha256(path)
    rows = load(path)
    require(isinstance(rows, list) and len(rows) == len(tests), "Archive coverage differs: " + str(path))

    def predictions():
        for row_id, (row, test) in enumerate(zip(rows, tests)):
            expected = "Can you predict the next possible item the user may expect, given the following chronological interaction history: " + ", ".join(catalog[item] for item in test["history"])
            require(row.get("input") == expected, f"Archive history/order differs at row {row_id}: {path}")
            output = row.get("output")
            target = output[0].strip('"').strip(" ") if isinstance(output, list) and len(output) == 1 else output.strip(' \n"') if isinstance(output, str) else None
            candidates = row.get("predict")
            require(isinstance(candidates, list) and all(isinstance(sid, str) for sid in candidates), "Invalid archived prediction list")
            # Match the repository calc.py whitespace/quote normalization; preserve rank and every candidate.
            yield row_id, target, [sid.strip('"\n').strip() for sid in candidates]

    strata = aggregate(predictions(), tests, known, catalog)
    require(sha256(path) == before, "Archive changed during analysis")
    key = "archive_sft" if stage == "SFT" else "archive_rl"
    return {"key": key, "label": "Archive 0.5B " + ("SFT" if stage == "SFT" else "SFT + RL"),
            "stage": stage, "model": "Qwen2.5-0.5B (repository report)", "hardware": "1 A800 (repository report)",
            "scheduler": "Archive configuration unverified; checked-in SFT code defaults to linear; README states cosine",
            "evidence": "Archived predictions reaggregated; training and model identity not independently verified",
            "prediction_file": str(path), "prediction_sha256": before, "history_and_target_alignment": "passed",
            "coverage": {"expected_rows": len(tests), "completed_rows": len(rows), "missing_rows": 0, "duplicate_rows": 0},
            "strata": strata}


def executed_result(run, key, tests, known, catalog, hashes):
    config_path, metric_path, prediction_path = run / "run_config.json", run / "evaluation/metrics.json", run / "evaluation/predictions.jsonl"
    before = {"config": sha256(config_path), "metrics": sha256(metric_path), "predictions": sha256(prediction_path)}
    config, evaluation = load(config_path), load(metric_path)
    require(evaluation.get("status") == "passed" and evaluation.get("category") == DATASET, "Executed evaluation has not passed: " + str(run))
    require(config.get("data", {}).get("category") == DATASET, "Executed training category differs")
    for name in ("train_file", "eval_file", "sid_index_path", "item_meta_path"):
        require(config.get("data", {}).get("sha256", {}).get(name) == hashes[name], "Executed training input differs: " + name)
    for name in ("train_file", "test_file", "sid_index_path"):
        require(evaluation.get("source_sha256", {}).get(name) == hashes[name], "Executed evaluation input differs: " + name)
    require(evaluation.get("item_meta_sha256") == hashes["item_meta_path"], "Executed metadata differs")
    require(type(evaluation.get("num_beams")) is int and evaluation["num_beams"] == 50 and evaluation.get("prompt_style") == "original-eval"
            and evaluation.get("validation_prompt_style") == "training", "Executed decoding protocol differs")
    coverage = evaluation.get("coverage", {})
    require(evaluation.get("completed_rows") == evaluation.get("expected_rows") == len(tests)
            and coverage.get("expected_rows") == coverage.get("completed_rows") == len(tests)
            and coverage.get("missing_rows") == coverage.get("duplicate_rows") == 0, "Executed coverage is incomplete")
    require(evaluation.get("CC") == 0 and evaluation.get("validity_rate") == evaluation.get("unique_rate") == 1, "Executed candidate integrity differs")
    required = ("provenance", "prepared_input_hashes", "checkpoint_reload", "tokenizer_atomic_sid_tokens", "model_vocabulary",
                "validation_coverage", "prediction_coverage", "valid_sids", "unique_candidates")
    require(all(evaluation.get("check_results", {}).get(name) in (True, "passed") for name in required), "Executed integrity check failed")

    def predictions():
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                require(row.get("prompt_style") == "original-eval", "Executed prediction prompt differs")
                yield row.get("row_id"), row.get("target"), row.get("predictions")

    strata = aggregate(predictions(), tests, known, catalog)
    for name, actual in strata["overall"]["metrics"].items():
        accepted = evaluation.get("metrics", {}).get(name)
        require(isinstance(accepted, (int, float)) and not isinstance(accepted, bool) and math.isfinite(accepted)
                and math.isclose(actual, accepted, abs_tol=1e-10, rel_tol=0), "Executed overall metric differs: " + name)
    hyper = config.get("hyperparameters", {})
    model = config.get("model", {}).get("source")
    require(isinstance(model, str) and model and hyper.get("lr_scheduler_type") and type(hyper.get("world_size")) is int
            and hyper["world_size"] > 0, "Executed model/scheduler/world size not recorded")
    for name, path in (("config", config_path), ("metrics", metric_path), ("predictions", prediction_path)):
        require(sha256(path) == before[name], "Executed source changed during analysis")
    return {"key": key, "label": f"{Path(model).name} SFT / {hyper['lr_scheduler_type']} / {hyper['world_size']} GPUs",
            "stage": "SFT", "model": model, "scheduler": hyper["lr_scheduler_type"], "world_size": hyper["world_size"],
            "epochs": hyper.get("max_epochs"), "global_batch_size": hyper.get("global_batch_size"),
            "evidence": "Independently executed SFT; accepted complete evaluation reaggregated from predictions",
            "run_directory": str(run), "source_sha256": before, "coverage": coverage, "strata": strata}


def build_comparison(reference_root, previous_run, current_run, data_dir, input_dir, *, expected_cold_rows=147, expected_cold_items=37):
    reference_root, previous_run, current_run, data_dir, input_dir = map(Path, (reference_root, previous_run, current_run, data_dir, input_dir))
    require(previous_run.resolve() != current_run.resolve(), "Previous and current runs must be separate")
    paths, hashes, catalog, known, tests, cohort = shared_data(data_dir, input_dir, expected_cold_rows, expected_cold_items)
    archives = reference_root / "results" if (reference_root / "results").is_dir() else reference_root
    results = [archived_result(archives / "eval_sft_result.json", "SFT", tests, known, catalog),
               archived_result(archives / "eval_rl_result.json", "SFT + RL", tests, known, catalog),
               executed_result(previous_run, "previous_sft", tests, known, catalog, hashes),
               executed_result(current_run, "current_sft", tests, known, catalog, hashes)]
    require(all(sha256(paths[key]) == value for key, value in hashes.items()), "Shared input changed during analysis")
    deltas = []
    for reference in results[:-1]:
        for group in ("overall", "warm", "cold"):
            for k in KS:
                for metric in ("HR", "NDCG"):
                    name = f"{metric}@{k}"
                    deltas.append({"current": "current_sft", "reference": reference["key"], "stratum": group,
                                   "K": k, "metric": metric, "difference_percentage_points":
                                   100 * (results[-1]["strata"][group]["metrics"][name] - reference["strata"][group]["metrics"][name])})
    return {"status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
            "cohort": cohort, "K": list(KS), "source_sha256": hashes, "results": results, "current_minus_reference": deltas,
            "protocol": "Original SID-level matching, full catalog, beam=50, every test row equally weighted; HR equals Recall for one relevant target.",
            "cold_definition": "Original target item ID absent from the union of every training history and target ID; full-catalog auxiliary SID/text exposure is retained.",
            "limitations": ["Archive model/hardware are repository claims; original training was not independently verified.",
                            "SFT + RL differs in training stage and is labeled separately.",
                            "Model size, scheduler, hardware and other configuration differences do not identify a causal effect.",
                            "Original colliding SIDs remain indistinguishable; no candidates or test rows are dropped."]}


def render(comparison, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ("#798492", "#8c5ba3", "#2865a6", "#12816e")
    for row_axes, group, title in zip(axes, ("overall", "cold"), ("All test interactions", "Behavior-cold targets")):
        for ax, metric in zip(row_axes, ("HR", "NDCG")):
            for result, color, style in zip(comparison["results"], colors, ("--", "--", "-", "-")):
                values = [result["strata"][group]["metrics"][f"{metric}@{k}"] for k in KS]
                ax.plot(KS, values, marker="o", markersize=5, linewidth=2, linestyle=style, color=color, label=result["label"])
            n = comparison["results"][0]["strata"][group]["rows"]
            ax.set_title(f"{title}: {'Recall / HR' if metric == 'HR' else 'NDCG'} (n={n})", loc="left", fontweight="bold")
            ax.set_xscale("log"); ax.set_xticks(KS, labels=KS); ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(.88, 57); ax.set_ylim(bottom=0); ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("K (log scale)"); ax.set_ylabel("Recall / HR" if metric == "HR" else metric)
            ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    fig.suptitle("Industrial and Scientific: repository archives and independent SFT runs", x=.065, y=.98, ha="left", fontsize=17, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.055, .95), frameon=False, ncol=2, fontsize=10)
    cohort = comparison["cohort"]
    fig.text(.065, .064, f"Cold cohort: {cohort['cold_rows']} rows / {cohort['cold_items']} item IDs; cold-to-warm SID aliases: {cohort['cold_sid_warm_alias_rows']}. All runs use the same original test rows.", fontsize=9)
    fig.text(.065, .041, "Archives report 0.5B; original training is unverified. README states cosine; checked-in SFT code defaults to linear.", fontsize=9)
    fig.text(.065, .018, "SFT + RL is a separate training stage. Configuration differences prevent a causal claim about model size or scheduler.", fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.84, bottom=.13, hspace=.36, wspace=.22)
    fig.savefig(destination / (PREFIX + ".png"), dpi=180, facecolor="white")
    plt.close(fig)


def generate(reference_root, previous_run, current_run, data_dir, input_dir, report_root, plots=True, **cohort_options):
    comparison = build_comparison(reference_root, previous_run, current_run, data_dir, input_dir, **cohort_options)
    destination = Path(report_root)
    destination.mkdir(parents=True, exist_ok=True)
    if plots:
        render(comparison, destination)
    comparison["rendered_plot"] = bool(plots)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["result", "label", "stage", "stratum", "rows", "K", "hits", "hit_target_item_ids", "HR", "NDCG"])
    for result in comparison["results"]:
        for name, group in result["strata"].items():
            for k in KS:
                writer.writerow([result["key"], result["label"], result["stage"], name, group["rows"], k,
                                 group["hits_by_k"][str(k)], group["hit_target_item_ids_by_k"][str(k)],
                                 group["metrics"][f"HR@{k}"], group["metrics"][f"NDCG@{k}"]])
    atomic_text(destination / (PREFIX + ".csv"), output.getvalue())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(comparison["current_minus_reference"][0]))
    writer.writeheader(); writer.writerows(comparison["current_minus_reference"])
    atomic_text(destination / "industrial_reproduction_deltas.csv", output.getvalue())
    atomic_text(destination / (PREFIX + ".json"), json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"status": comparison["status"], "report_root": str(destination), "rendered_plot": bool(plots), "cohort": comparison["cohort"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "previous-run", "current-run", "data-dir", "input-dir", "report-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate(args.reference_root, args.previous_run, args.current_run, args.data_dir, args.input_dir,
                          args.report_root, plots=not args.no_plots)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
