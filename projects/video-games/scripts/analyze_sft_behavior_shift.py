#!/usr/bin/env python3
"""Audit behavior coverage and stratified metrics from an accepted SFT evaluation.

All metrics are derived from existing predictions. The script performs no model
inference or training and never changes the source artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import tempfile


TOP_K = (1, 3, 5, 10, 20, 50)
STRATA = (
    ("overall", "完整测试集"), ("behavior_warm", "训练行为已见目标"),
    ("behavior_cold", "训练行为未见目标"), ("repeat_last", "目标等于末次历史商品"),
    ("not_repeat_last", "目标不同于末次历史商品"), ("repeat_any", "目标出现于任一历史位置"),
    ("not_repeat_any", "目标未出现于历史"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inter_rows(path, catalog):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        require(next(reader, None) == ["user_id:token", "item_id_list:token_seq", "item_id:token"],
                f"{Path(path).name}: unexpected interaction header")
        for line, fields in enumerate(reader, 2):
            require(len(fields) == 3, f"{Path(path).name}:{line}: expected three fields")
            _, text, target = fields
            history = text.split()
            require(history and all(item in catalog for item in [*history, target]),
                    f"{Path(path).name}:{line}: empty history or item outside catalog")
            yield history, target


def paired_rows(inter_path, csv_path, catalog):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and {"item_id", "item_sid"} <= set(reader.fieldnames),
                f"{Path(csv_path).name}: missing target columns")
        for row_id, (raw, prepared) in enumerate(zip_longest(inter_rows(inter_path, catalog), reader)):
            require(raw is not None and prepared is not None,
                    f"{Path(csv_path).name}: raw/prepared row count differs")
            history, target = raw
            require(prepared.get("item_id") == target and prepared.get("item_sid") == catalog[target],
                    f"{Path(csv_path).name}:{row_id}: target/order differs from raw interaction")
            yield history, target


def analyze(data_dir, input_dir, run_dir):
    data_dir, input_dir, run_dir = map(lambda path: Path(path).resolve(strict=True),
                                      (data_dir, input_dir, run_dir))
    paths = {
        "train_file": data_dir / "train.csv", "test_file": data_dir / "test.csv",
        "sid_index_path": input_dir / "Video_Games.index.json",
        "item_meta_path": input_dir / "Video_Games.item.json",
        "train_inter": input_dir / "Video_Games.train.inter",
        "test_inter": input_dir / "Video_Games.test.inter",
        "predictions": run_dir / "evaluation/predictions.jsonl",
        "evaluation_metrics": run_dir / "evaluation/metrics.json",
        "preparation_manifest": data_dir / "preparation_manifest.json",
        "run_config": run_dir / "run_config.json",
        "comparison_baselines": run_dir / "comparison_baselines.json",
        "cleaning_report": input_dir / "cleaning_report.json",
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    preparation = read_json(paths["preparation_manifest"])
    config = read_json(paths["run_config"])
    evaluation = read_json(paths["evaluation_metrics"])
    baselines = read_json(paths["comparison_baselines"])
    cleaning = read_json(paths["cleaning_report"])
    require(preparation.get("validation") == "passed" and preparation.get("dataset") == "Video_Games",
            "Prepared data must be an accepted Video_Games conversion")
    require(evaluation.get("status") == "passed" and evaluation.get("category") == "Video_Games",
            "A passed Video_Games evaluation is required")
    require(config.get("data", {}).get("category") == "Video_Games", "Run category mismatch")
    require(baselines.get("status") == "passed" and baselines.get("dataset") == "Video_Games",
            "Accepted same-dataset comparison baselines are required")
    expected = {
        "train_file": preparation.get("outputs", {}).get("train.csv", {}).get("sha256"),
        "test_file": preparation.get("outputs", {}).get("test.csv", {}).get("sha256"),
        "sid_index_path": preparation.get("sources", {}).get("sid_index", {}).get("sha256"),
        "item_meta_path": preparation.get("sources", {}).get("item_metadata", {}).get("sha256"),
        "train_inter": preparation.get("sources", {}).get("train", {}).get("sha256"),
        "test_inter": preparation.get("sources", {}).get("test", {}).get("sha256"),
    }
    for key, value in expected.items():
        require(isinstance(value, str) and value == hashes[key], f"{key}: prepared source SHA-256 mismatch")
    for key in ("train_file", "sid_index_path", "item_meta_path"):
        require(config.get("data", {}).get("sha256", {}).get(key) == hashes[key],
                f"{key}: run configuration SHA-256 mismatch")
    configured_test_hash = config.get("data", {}).get("sha256", {}).get("test_file")
    if configured_test_hash is not None:
        require(configured_test_hash == hashes["test_file"], "test_file: run configuration SHA-256 mismatch")
    for key in ("train_file", "test_file", "sid_index_path"):
        require(evaluation.get("source_sha256", {}).get(key) == hashes[key],
                f"{key}: evaluation source SHA-256 mismatch")
        require(baselines.get("source_sha256", {}).get(key) == hashes[key],
                f"{key}: baseline source SHA-256 mismatch")
    require(evaluation.get("sid_index_sha256") == hashes["sid_index_path"]
            and evaluation.get("item_meta_sha256") == hashes["item_meta_path"],
            "Evaluation catalog/metadata identity mismatch")
    if baselines.get("preparation_manifest_sha256") is not None:
        require(baselines["preparation_manifest_sha256"] == hashes["preparation_manifest"],
                "Baseline preparation manifest identity mismatch")

    indices, metadata = read_json(paths["sid_index_path"]), read_json(paths["item_meta_path"])
    require(isinstance(indices, dict) and indices and set(indices) == set(metadata),
            "SID/metadata catalog IDs differ")
    require(all(isinstance(tokens, list) and tokens and all(isinstance(t, str) and t for t in tokens)
                for tokens in indices.values()), "SID catalog token lists are invalid")
    catalog = {item: "".join(tokens) for item, tokens in indices.items()}
    require(len(set(catalog.values())) == len(catalog), "Complete SIDs are not unique")
    valid_sids = set(catalog.values())
    require(preparation.get("items") == len(catalog) == baselines.get("catalog_items"), "Catalog item count mismatch")

    train_rows = train_repeat_last = train_repeat_any = 0
    behavior, train_target_counts = set(), Counter()
    for history, target in paired_rows(paths["train_inter"], paths["train_file"], catalog):
        train_rows += 1
        train_repeat_last += target == history[-1]
        train_repeat_any += target in history
        behavior.update(history)
        behavior.add(target)
        train_target_counts[catalog[target]] += 1
    require(train_rows > 0 and train_rows == preparation.get("rows", {}).get("train") == baselines.get("train_rows"),
            "Training row count mismatch")
    pop_sids = sorted(valid_sids, key=lambda sid: (-train_target_counts[sid], sid))[:max(TOP_K)]
    require(baselines.get("popularity", {}).get("top_sids") == pop_sids,
            "Popularity ranking differs from training-only target counts")
    pop_rank = {sid: rank for rank, sid in enumerate(pop_sids, 1)}
    rows, test_targets, cold_targets = [], set(), set()
    test_repeat_last = test_repeat_any = cold_rows = 0
    for history, target in paired_rows(paths["test_inter"], paths["test_file"], catalog):
        cold, repeat_last, repeat_any = target not in behavior, target == history[-1], target in history
        keys = ["overall", "behavior_cold" if cold else "behavior_warm",
                "repeat_last" if repeat_last else "not_repeat_last", "repeat_any" if repeat_any else "not_repeat_any"]
        rows.append((catalog[target], catalog[history[-1]], keys))
        test_targets.add(target)
        if cold:
            cold_targets.add(target)
        cold_rows += cold
        test_repeat_last += repeat_last
        test_repeat_any += repeat_any
    total = len(rows)
    require(total > 0 and total == preparation.get("rows", {}).get("test") == baselines.get("test_rows")
            == evaluation.get("expected_rows") == evaluation.get("completed_rows"), "Test row count mismatch")
    coverage = evaluation.get("coverage", {})
    require(coverage.get("missing_rows") in (0, []) and coverage.get("duplicate_rows") in (0, []),
            "Evaluation does not have complete unique row coverage")
    beams = evaluation.get("num_beams")
    require(isinstance(beams, int) and min(max(TOP_K), len(catalog)) <= beams <= len(catalog),
            "Evaluation does not provide enough ranked candidates")
    accumulators = {key: {"key": key, "label": label, "rows": 0,
                          "hits": Counter(), "discounted": Counter(), "pop_hits": Counter(), "pop_discounted": Counter(),
                          "model_top1_equals_last_count": 0, "model_top1_copy_correct_count": 0,
                          "model_top1_copy_wrong_count": 0, "model_top1_correct_count": 0}
                    for key, label in STRATA}
    cold_catalog_sids = {catalog[item] for item in set(catalog) - behavior}
    cold_candidate_occurrences = total_candidate_occurrences = cold_top1_predictions = 0
    cold_predicted_unique = set()
    seen = set()
    with paths["predictions"].open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            record = json.loads(text)
            row_id = record.get("row_id")
            require(isinstance(row_id, int) and not isinstance(row_id, bool) and 0 <= row_id < total,
                    f"predictions:{line}: invalid row_id")
            require(row_id not in seen, f"predictions:{line}: duplicate prediction row")
            seen.add(row_id)
            target, last, keys = rows[row_id]
            require(record.get("target") == target, f"predictions:{line}: target/order mismatch")
            predicted = record.get("predictions")
            require(isinstance(predicted, list) and len(predicted) == beams
                    and all(isinstance(sid, str) and sid in valid_sids for sid in predicted)
                    and len(set(predicted)) == len(predicted), f"predictions:{line}: invalid/duplicate candidates")
            rank = predicted.index(target) + 1 if target in predicted else math.inf
            popularity_rank = pop_rank.get(target, math.inf)
            copies, correct = predicted[0] == last, predicted[0] == target
            cold_predictions = cold_catalog_sids.intersection(predicted)
            cold_candidate_occurrences += len(cold_predictions)
            total_candidate_occurrences += len(predicted)
            cold_predicted_unique.update(cold_predictions)
            cold_top1_predictions += predicted[0] in cold_catalog_sids
            for key in keys:
                acc = accumulators[key]
                acc["rows"] += 1
                acc["model_top1_equals_last_count"] += copies
                acc["model_top1_copy_correct_count"] += copies and correct
                acc["model_top1_copy_wrong_count"] += copies and not correct
                acc["model_top1_correct_count"] += correct
                for k in TOP_K:
                    if rank <= k:
                        acc["hits"][k] += 1
                        acc["discounted"][k] += 1 / math.log2(rank + 1)
                    if popularity_rank <= k:
                        acc["pop_hits"][k] += 1
                        acc["pop_discounted"][k] += 1 / math.log2(popularity_rank + 1)
    require(len(seen) == total, "Prediction coverage is incomplete: missing rows")
    uniform = {f"{metric}@{k}": (min(k, len(catalog)) / len(catalog) if metric == "HR" else
                               sum(1 / math.log2(r + 1) for r in range(1, min(k, len(catalog)) + 1)) / len(catalog))
               for metric in ("HR", "NDCG") for k in TOP_K}
    strata = []
    for key, _ in STRATA:
        acc = accumulators[key]
        count = acc["rows"]
        result = {name: value for name, value in acc.items() if not isinstance(value, Counter)}
        result.update(share=count / total, hits_by_k={str(k): acc["hits"][k] for k in TOP_K},
                      metrics={f"{metric}@{k}": acc[field][k] / count if count else None
                               for metric, field in (("HR", "hits"), ("NDCG", "discounted")) for k in TOP_K},
                      popularity={f"{metric}@{k}": acc[field][k] / count if count else None
                                  for metric, field in (("HR", "pop_hits"), ("NDCG", "pop_discounted")) for k in TOP_K},
                      uniform_random=uniform.copy() if count else {key: None for key in uniform},
                      model_top1_equals_last_rate=acc["model_top1_equals_last_count"] / count if count else None,
                      model_top1_copy_accuracy=(acc["model_top1_copy_correct_count"] / acc["model_top1_equals_last_count"]
                                                if acc["model_top1_equals_last_count"] else None))
        strata.append(result)
    for name, value in strata[0]["metrics"].items():
        observed = evaluation.get("metrics", {}).get(name)
        require(isinstance(observed, (int, float)) and math.isfinite(observed)
                and math.isclose(value, observed, rel_tol=0, abs_tol=1e-10),
                f"Recomputed {name} differs from accepted evaluation")
    for method, values in (("popularity", strata[0]["popularity"]), ("uniform_random", uniform)):
        supplied = baselines.get(method, {}).get("metrics", {})
        if supplied:
            require(all(isinstance(supplied.get(key), (int, float))
                        and math.isclose(value, supplied[key], rel_tol=0, abs_tol=1e-10) for key, value in values.items()),
                    f"Recomputed {method} metrics differ from accepted baseline")

    # Match SidItemFeatDataset's catalog iteration and title-key overwrite rules.
    sid_to_title, title_to_sid = {}, {}
    for item, sid in catalog.items():
        title = metadata[item]["title"]
        sid_to_title[sid] = title
        title_to_sid[title] = sid
    reverse_sids = set(title_to_sid.values())
    cold_sids = {catalog[item] for item in cold_targets}
    cold_sid_rows = sum(1 for target, _, keys in rows if "behavior_cold" in keys and target in reverse_sids)
    overall = strata[0]
    prediction_diagnostics = {
        "top1_equals_last_rows": overall["model_top1_equals_last_count"],
        "top1_equals_last_rate": overall["model_top1_equals_last_rate"],
        "copy_correct_rows": overall["model_top1_copy_correct_count"],
        "copy_wrong_rows": overall["model_top1_copy_wrong_count"],
        "copy_precision": overall["model_top1_copy_accuracy"],
        "copy_share_of_all_top1_hits": (overall["model_top1_copy_correct_count"] / overall["model_top1_correct_count"]
                                         if overall["model_top1_correct_count"] else None),
        "cold_candidate_occurrences": cold_candidate_occurrences,
        "total_candidate_occurrences": total_candidate_occurrences,
        "cold_candidate_rate": cold_candidate_occurrences / total_candidate_occurrences,
        "cold_predicted_unique_items": len(cold_predicted_unique), "cold_top1_predictions": cold_top1_predictions,
        "cold_definition": "All catalog items absent from every training behavior history and target.",
    }
    for key, path in paths.items():
        require(sha256(path) == hashes[key], f"{key}: source changed while analyzing")
    return {
        "status": "passed", "schema_version": 1, "source_sha256": hashes,
        "source_paths": {key: str(path) for key, path in paths.items()}, "overall_test_rows": total,
        "counts": {"train_rows": train_rows, "test_rows": total, "catalog_items": len(catalog),
                   "training_behavior_items": len(behavior), "training_target_items": len(train_target_counts),
                   "test_target_unique_items": len(test_targets), "cold_test_unique_items": len(cold_targets),
                   "catalog_behavior_cold_items": len(set(catalog) - behavior)},
        "shift_summary": {
            "test_behavior_cold_rows": cold_rows, "test_behavior_cold_rate": cold_rows / total,
            "train_repeat_last_rows": train_repeat_last, "train_repeat_last_rate": train_repeat_last / train_rows,
            "test_repeat_last_rows": test_repeat_last, "test_repeat_last_rate": test_repeat_last / total,
            "train_repeat_any_rows": train_repeat_any, "train_repeat_any_rate": train_repeat_any / train_rows,
            "test_repeat_any_rows": test_repeat_any, "test_repeat_any_rate": test_repeat_any / total,
        },
        "temporal_boundaries": cleaning.get("split_boundaries", {}),
        "alignment_coverage": {
            "cold_unique_items": len(cold_targets), "sid_to_title_items": len(cold_sids & sid_to_title.keys()),
            "title_to_sid_items": len(cold_sids & reverse_sids), "title_to_sid_test_rows": cold_sid_rows,
            "cold_test_rows": cold_rows, "catalog_sid_to_title_items": len(sid_to_title),
            "catalog_title_to_sid_items": len(title_to_sid),
            "cold_items_missing_title_to_sid": sorted(item for item in cold_targets if catalog[item] not in reverse_sids),
            "definition": "Exact SidItemFeatDataset catalog iteration: every SID maps to its title; duplicate title keys retain the last catalog SID for title-to-SID tasks.",
        },
        "strata": strata, "prediction_diagnostics": prediction_diagnostics,
        "interpretation": [
            "Behavior-cold means absent from every training history and target, not an unknown catalog item or unknown metadata.",
            "Metadata/SID alignment uses the shared full catalog; duplicate titles reduce reverse title-to-SID task coverage.",
            "Every original test row has equal weight, including repeated interactions; item counts and row counts use different denominators.",
            "Repeat-last target frequency measures label distribution. Actual top-1 copying and its correct/wrong counts measure model behavior separately.",
            "These strata describe association and distribution shift; they do not identify a causal explanation or demonstrate a paper replication.",
            "Rolling histories may include earlier validation/test interactions. Equal timestamps can cross adjacent global chronological split boundaries.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = (args.output or args.run_dir / "behavior_shift_analysis.json").resolve()
    result = analyze(args.data_dir, args.input_dir, args.run_dir)
    require(output not in {Path(path) for path in result["source_paths"].values()},
            "Output must not overwrite a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".behavior-shift-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "output": str(output),
                      "test_rows": result["overall_test_rows"], "shift_summary": result["shift_summary"]}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
