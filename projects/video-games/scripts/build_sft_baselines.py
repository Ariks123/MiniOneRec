#!/usr/bin/env python3
"""Measure simple recommendation baselines on the approved prepared test rows.

Popularity uses training targets only. Uniform random metrics are exact expected
values, rather than the result of a random trial. No model or training stack is
loaded, and source data is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


TOP_K = (1, 3, 5, 10, 20, 50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path):
    catalog = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(catalog, dict) and catalog, "SID catalog must be a nonempty object")
    sid_by_item = {}
    for item, tokens in catalog.items():
        require(isinstance(tokens, list) and len(tokens) == 4,
                f"Item {item}: expected four SID tokens")
        require(all(isinstance(token, str) and re.fullmatch(r"<" + level + r"_(0|[1-9][0-9]*)>", token)
                    for level, token in zip("abcd", tokens)),
                f"Item {item}: invalid SID tokens")
        sid_by_item[item] = "".join(tokens)
    require(len(set(sid_by_item.values())) == len(sid_by_item), "Complete SIDs must be unique")
    return sid_by_item


def target_counts(path, sid_by_item):
    counts = Counter()
    rows = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and {"item_id", "item_sid"} <= set(reader.fieldnames),
                f"{Path(path).name}: missing item_id/item_sid columns")
        for line, row in enumerate(reader, 2):
            item, sid = row.get("item_id"), row.get("item_sid")
            require(item in sid_by_item, f"{Path(path).name}:{line}: target item is not in the catalog")
            require(sid == sid_by_item[item], f"{Path(path).name}:{line}: target SID differs from catalog")
            counts[sid] += 1
            rows += 1
    require(rows > 0, f"{Path(path).name}: no target rows")
    return counts, rows


def build_baselines(data_dir, sid_index_path):
    data_dir = Path(data_dir).resolve(strict=True)
    sid_index_path = Path(sid_index_path).resolve(strict=True)
    manifest_path = data_dir / "preparation_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    require(manifest.get("dataset") == "Video_Games" and manifest.get("validation") == "passed",
            "Prepared Video_Games manifest must have passed validation")
    paths = {"train_file": data_dir / "train.csv", "test_file": data_dir / "test.csv",
             "sid_index_path": sid_index_path}
    expected = {"train_file": manifest.get("outputs", {}).get("train.csv", {}).get("sha256"),
                "test_file": manifest.get("outputs", {}).get("test.csv", {}).get("sha256"),
                "sid_index_path": manifest.get("sources", {}).get("sid_index", {}).get("sha256")}
    for name, value in expected.items():
        require(isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value),
                f"Preparation manifest has no valid {name} SHA-256")
    hashes = {name: sha256(path) for name, path in paths.items()}
    for name in paths:
        require(hashes[name] == expected[name].lower(), f"{name}: preparation SHA-256 mismatch")
    if manifest.get("expected_sid_sha256") is not None:
        require(manifest["expected_sid_sha256"] == hashes["sid_index_path"],
                "Approved SID SHA-256 differs from source manifest")

    sid_by_item = load_catalog(sid_index_path)
    require(manifest.get("items") == len(sid_by_item), "Catalog size differs from preparation manifest")
    train_counts, train_rows = target_counts(paths["train_file"], sid_by_item)
    # Determine and freeze the ranking before reading any test targets.
    ranked_sids = sorted(sid_by_item.values(), key=lambda sid: (-train_counts[sid], sid))[:max(TOP_K)]
    ranks = {sid: rank for rank, sid in enumerate(ranked_sids, 1)}
    test_counts, test_rows = target_counts(paths["test_file"], sid_by_item)
    require(manifest.get("rows", {}).get("train") == train_rows,
            "Training row count differs from preparation manifest")
    require(manifest.get("rows", {}).get("test") == test_rows,
            "Test row count differs from preparation manifest")
    popularity, uniform = {}, {}
    size = len(sid_by_item)
    for k in TOP_K:
        hits = sum(count for sid, count in test_counts.items() if ranks.get(sid, math.inf) <= k)
        discounted = sum(count / math.log2(ranks[sid] + 1) for sid, count in test_counts.items()
                         if ranks.get(sid, math.inf) <= k)
        popularity[f"HR@{k}"] = hits / test_rows
        popularity[f"NDCG@{k}"] = discounted / test_rows
        uniform[f"HR@{k}"] = min(k, size) / size
        uniform[f"NDCG@{k}"] = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, size) + 1)) / size

    for name, path in paths.items():
        require(sha256(path) == hashes[name], f"{name}: source changed while computing baselines")
    require(sha256(manifest_path) == manifest_hash, "Preparation manifest changed while computing baselines")
    return {
        "status": "passed", "dataset": "Video_Games", "catalog_items": size,
        "train_rows": train_rows, "test_rows": test_rows,
        "source_sha256": hashes,
        "source_paths": {name: str(path) for name, path in paths.items()},
        "preparation_manifest_sha256": manifest_hash,
        "popularity": {
            "definition": "Global ranking by training target frequency; ties use ascending lexicographic SID; no seen-item exclusion.",
            "metrics": popularity, "top_sids": ranked_sids,
        },
        "uniform_random": {
            "definition": "Exact expected metrics for a uniformly random catalog permutation without replacement and one relevant target per test row.",
            "metrics": uniform,
        },
        "methodology": {
            "top_k": list(TOP_K), "popularity_fit_split": "train",
            "test_used_for_ranking": False, "validation_used_for_ranking": False,
            "seen_item_exclusion": False, "test_row_deduplication": False,
            "test_weighting": "Each original test row contributes once; repeated interactions retain their original weight.",
            "hr_equals_recall": "Each row has one relevant item; HR@K equals Recall@K.",
            "interpretation": "Descriptive same-split baselines; no statistical significance or paper replication claim.",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sid-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    sources = [args.sid_index, *(args.data_dir / name for name in
                                ("train.csv", "test.csv", "preparation_manifest.json"))]
    require(output not in {path.resolve() for path in sources}, "Output must not overwrite a source file")
    result = build_baselines(args.data_dir, args.sid_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".sft-baselines-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "output": str(output),
                      "catalog_items": result["catalog_items"], "test_rows": result["test_rows"]}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
