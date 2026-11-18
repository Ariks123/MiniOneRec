#!/usr/bin/env python3
"""Audit and copy the original released Industrial dataset without changing SIDs.

Original CSVs and info.txt are copied byte-for-byte. Canonical .inter files are
derived in the same row order for provenance checks and behavior diagnostics.
Original SID collisions are retained and explicitly recorded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATASET = "Industrial_and_Scientific"
SUFFIX = "_5_2016-10-2018-11"
APPROVED_SID_SHA256 = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"
FIELDS = ["user_id", "history_item_title", "item_title", "history_item_id", "item_id", "history_item_sid", "item_sid"]
SPLITS = ("train", "valid", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_paths(source_root):
    return {"sid_index": source_root / "index" / f"{DATASET}.index.json",
            "item_metadata": source_root / "index" / f"{DATASET}.item.json",
            "info": source_root / "info" / f"{DATASET}{SUFFIX}.txt",
            **{split: source_root / split / f"{DATASET}{SUFFIX}.csv" for split in SPLITS}}


def checked_rows(path, sids, metadata):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, f"{Path(path).name}: unexpected CSV columns")
        for row_id, row in enumerate(reader):
            prefix = f"{Path(path).name}:{row_id + 2}"
            try:
                raw_ids = ast.literal_eval(row["history_item_id"])
                history_sids = ast.literal_eval(row["history_item_sid"])
                history_titles = ast.literal_eval(row["history_item_title"])
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"{prefix}: invalid history literal") from error
            require(isinstance(raw_ids, list) and 1 <= len(raw_ids) <= 10,
                    f"{prefix}: history length must be 1–10")
            require(all(type(item) is int or isinstance(item, str) for item in raw_ids),
                    f"{prefix}: invalid history item ID type")
            history = [str(item) for item in raw_ids]
            target = row["item_id"]
            require(target in sids and all(item in sids for item in history), f"{prefix}: unmapped item ID")
            require(row["item_sid"] == sids[target], f"{prefix}: target SID mismatch")
            require(row["item_title"] == metadata[target]["title"], f"{prefix}: target title mismatch")
            require(history_sids == [sids[item] for item in history], f"{prefix}: history SID mismatch")
            require(history_titles == [metadata[item]["title"] for item in history], f"{prefix}: history title mismatch")
            require(bool(row["user_id"]) and not any(character in row["user_id"] for character in "\t\r\n"),
                    f"{prefix}: invalid user ID")
            yield row["user_id"], history, target


def audit(source_root, expected_sid_sha256=APPROVED_SID_SHA256):
    paths = source_paths(Path(source_root).resolve(strict=True))
    hashes = {name: sha256(path) for name, path in paths.items()}
    require(hashes["sid_index"] == expected_sid_sha256.lower(), "Approved original SID SHA-256 mismatch")
    indices = json.loads(paths["sid_index"].read_text(encoding="utf-8-sig"))
    metadata = json.loads(paths["item_metadata"].read_text(encoding="utf-8-sig"))
    require(isinstance(indices, dict) and indices and set(indices) == set(metadata), "SID/metadata catalog IDs differ")
    require(all(re.fullmatch(r"0|[1-9][0-9]*", item) for item in indices), "Catalog IDs must be nonnegative integer strings")
    for item, tokens in indices.items():
        require(isinstance(tokens, list) and len(tokens) in (3, 4), f"Item {item}: expected three or four original SID tokens")
        require(all(isinstance(token, str) and re.fullmatch(r"<" + level + r"_(0|[1-9][0-9]*)>", token)
                    for level, token in zip("abcd", tokens)), f"Item {item}: malformed original SID tokens")
        require(isinstance(metadata[item], dict) and isinstance(metadata[item].get("title"), str),
                f"Item {item}: missing title")
    sids = {item: "".join(tokens) for item, tokens in indices.items()}
    groups = defaultdict(list)
    for item, sid in sids.items():
        groups[sid].append(item)
    collision_groups = [{"sid": sid, "item_ids": items, "titles": [metadata[item]["title"] for item in items]}
                        for sid, items in groups.items() if len(items) > 1]
    collided_ids = {item for group in collision_groups for item in group["item_ids"]}
    train_behavior, all_behavior, train_targets, test_targets = set(), set(), set(), Counter()
    split_audits, global_users = {}, set()
    for split in SPLITS:
        users, targets, behavior, lengths, duplicates = set(), set(), set(), Counter(), Counter()
        count = repeat_last = repeat_any = collision_rows = 0
        for user, history, target in checked_rows(paths[split], sids, metadata):
            count += 1
            users.add(user)
            global_users.add(user)
            targets.add(target)
            behavior.update(history)
            behavior.add(target)
            lengths[len(history)] += 1
            duplicates[(user, tuple(history), target)] += 1
            repeat_last += target == history[-1]
            repeat_any += target in history
            collision_rows += target in collided_ids
            if split == "train":
                train_behavior.update(history)
                train_behavior.add(target)
                train_targets.add(target)
            if split == "test":
                test_targets[target] += 1
        require(count > 0, f"{split}: empty original split")
        all_behavior.update(behavior)
        split_audits[split] = {
            "rows": count, "users": len(users), "target_unique_items": len(targets), "behavior_unique_items": len(behavior),
            "history_min": min(lengths), "history_max": max(lengths), "history_length_counts": dict(lengths),
            "repeat_last_rows": repeat_last, "repeat_last_rate": repeat_last / count,
            "repeat_any_rows": repeat_any, "repeat_any_rate": repeat_any / count,
            "duplicate_full_rows": sum(value - 1 for value in duplicates.values()), "collided_target_rows": collision_rows,
        }
    info_ids = set()
    with paths["info"].open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.rstrip("\r\n").split("\t")
            require(len(parts) == 3, f"info:{line_number}: expected SID, title and item ID")
            sid, title, item = parts
            require(item in sids and item not in info_ids and sids[item] == sid and metadata[item]["title"] == title,
                    f"info:{line_number}: catalog mapping mismatch")
            info_ids.add(item)
    require(info_ids == set(indices), "Info file does not cover every original item ID")
    cold = set(test_targets) - train_behavior
    cold_rows = sum(test_targets[item] for item in cold)
    warm_sids = {sids[item] for item in train_behavior}
    ambiguous_cold = {item for item in cold if sids[item] in warm_sids}
    for name, path in paths.items():
        require(sha256(path) == hashes[name], f"{name}: original source changed during audit")
    result = {
        "catalog_items": len(indices), "catalog_ids_contiguous_from_zero": set(indices) == {str(i) for i in range(len(indices))},
        "unique_sids": len(groups), "complete_sids_unique": not collision_groups,
        "sid_token_length_counts": dict(Counter(map(len, indices.values()))),
        "sid_collisions": {"groups": len(collision_groups), "items": len(collided_ids), "excess_items": len(indices) - len(groups)},
        "sid_collision_groups": collision_groups, "splits": split_audits, "users_across_all_splits": len(global_users),
        "training_behavior_items": len(train_behavior), "training_target_items": len(train_targets),
        "catalog_behavior_cold_items": len(set(indices) - train_behavior), "catalog_items_absent_from_all_behavior": len(set(indices) - all_behavior),
        "test_target_unique_items": len(test_targets), "cold_test_unique_items": len(cold),
        "cold_test_rows": cold_rows, "cold_test_rate": cold_rows / split_audits["test"]["rows"],
        "cold_test_targets_in_collided_groups": sorted(cold & collided_ids, key=int),
        "cold_test_targets_with_warm_sid": sorted(ambiguous_cold, key=int),
        "cold_test_rows_with_warm_sid": sum(test_targets[item] for item in ambiguous_cold),
        "validation": "passed", "mismatches": 0,
    }
    return result, paths, hashes, sids, metadata


def prepare(source_root, output_dir, input_dir=None, expected_sid_sha256=APPROVED_SID_SHA256):
    source_root, output_dir = Path(source_root).resolve(strict=True), Path(output_dir).resolve()
    input_dir = (Path(input_dir) if input_dir is not None else source_root.parent / "Amazon18/Industrial_and_Scientific_original").resolve()
    require(input_dir != output_dir and input_dir not in output_dir.parents and output_dir not in input_dir.parents,
            "Canonical input and prepared output directories must be separate")
    for destination in (input_dir, output_dir):
        require(destination != source_root and source_root not in destination.parents and destination not in source_root.parents,
                "Do not place generated files inside or above original source data")
        require(not destination.exists() or not any(destination.iterdir()), f"Refusing to overwrite nonempty directory: {destination}")
    result, paths, hashes, sids, metadata = audit(source_root, expected_sid_sha256)
    for destination in (input_dir, output_dir):
        destination.parent.mkdir(parents=True, exist_ok=True)
    canonical_stage = Path(tempfile.mkdtemp(prefix=".industrial-original-input-", dir=input_dir.parent))
    prepared_stage = Path(tempfile.mkdtemp(prefix=".industrial-original-prepared-", dir=output_dir.parent))
    try:
        for name, filename in (("sid_index", f"{DATASET}.index.json"), ("item_metadata", f"{DATASET}.item.json")):
            shutil.copyfile(paths[name], canonical_stage / filename)
        sources = {"sid_index": {"filename": f"{DATASET}.index.json", "sha256": hashes["sid_index"]},
                   "item_metadata": {"filename": f"{DATASET}.item.json", "sha256": hashes["item_metadata"]}}
        for split in SPLITS:
            shutil.copyfile(paths[split], prepared_stage / f"{split}.csv")
            filename = f"{DATASET}.{split}.inter"
            with (canonical_stage / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["user_id:token", "item_id_list:token_seq", "item_id:token"])
                writer.writerows((user, " ".join(history), target) for user, history, target
                                 in checked_rows(paths[split], sids, metadata))
            sources[split] = {"filename": filename, "sha256": sha256(canonical_stage / filename)}
        shutil.copyfile(paths["info"], prepared_stage / "info.txt")
        rows = {split: result["splits"][split]["rows"] for split in SPLITS}
        cleaning = {"dataset": DATASET, "validation": "passed", "source": "Original repository released prepared CSVs",
                    "users": result["users_across_all_splits"], "items": result["catalog_items"],
                    "examples": sum(rows.values()), "split_examples": rows, "split_boundaries": {},
                    "notes": ["Source CSVs do not contain timestamps; filename dates are not independently verified temporal boundaries.",
                              "Original SIDs, collisions, row order, splits and duplicate interactions are preserved."],
                    "source_audit": result}
        write_json(canonical_stage / "cleaning_report.json", cleaning)
        lengths = sorted(int(length) for length in result["sid_token_length_counts"])
        manifest = {
            "schema_version": 1, "dataset": DATASET, "original_sid": True, "selected_sid_epoch": None,
            "expected_sid_sha256": hashes["sid_index"], "items": result["catalog_items"],
            "complete_sid_tokens": lengths[0] if len(lengths) == 1 else lengths,
            "complete_sids_unique": result["complete_sids_unique"], "unique_sids": result["unique_sids"],
            "sid_collisions": result["sid_collisions"], "sid_collision_groups": result["sid_collision_groups"],
            "history_length": {"minimum": min(row["history_min"] for row in result["splits"].values()),
                               "maximum": max(row["history_max"] for row in result["splits"].values())},
            "rows": rows, "preserve_all_interaction_rows": True, "sampling_applied": False, "keep_longest_only": False,
            "input_directory": str(input_dir), "output_directory": str(output_dir), "sources": sources,
            "original_sources": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
            "outputs": {name: {"filename": name, "sha256": sha256(prepared_stage / name), "bytes": (prepared_stage / name).stat().st_size}
                        for name in ("train.csv", "valid.csv", "test.csv", "info.txt")},
            "validation": "passed", "check_only": False, "source_audit": result,
            "evaluation_identity": "Original SID sequence equivalence; collided item IDs cannot be distinguished by SID-only scoring.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for split in SPLITS:
            require(manifest["outputs"][f"{split}.csv"]["sha256"] == hashes[split], "CSV copy changed bytes")
        require(manifest["outputs"]["info.txt"]["sha256"] == hashes["info"], "Info copy changed bytes")
        for name, path in paths.items():
            require(sha256(path) == hashes[name], f"{name}: original changed during preparation")
        write_json(prepared_stage / "preparation_manifest.json", manifest)
        for stage, destination in ((canonical_stage, input_dir), (prepared_stage, output_dir)):
            if destination.exists():
                destination.rmdir()
            os.replace(stage, destination)
        return manifest
    finally:
        for stage in (canonical_stage, prepared_stage):
            if stage.exists():
                shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/Amazon")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/Amazon18/Industrial_and_Scientific_original")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/Amazon18/posttraining/Industrial_and_Scientific_original")
    parser.add_argument("--expected-sid-sha256", default=APPROVED_SID_SHA256)
    args = parser.parse_args(argv)
    result = prepare(args.source_root, args.output_dir, args.input_dir, args.expected_sid_sha256)
    print(json.dumps({"validation": result["validation"], "input_directory": result["input_directory"],
                      "output_directory": result["output_directory"], "rows": result["rows"],
                      "items": result["items"], "unique_sids": result["unique_sids"], "sid_collisions": result["sid_collisions"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
