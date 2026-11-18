#!/usr/bin/env python3
"""Publish learned three-token SIDs and remap only SID columns in fixed splits."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from scripts.train_industrial_rqvae import BASELINE_ROWS, CATALOG_SIZE, DATASET, EMBEDDING_SHA, LEGACY_SID_SHA
from hybrid.rho_bundle import digest, publish


CANONICAL = Path("MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey")
POSTTRAIN = CANONICAL.parent / "posttraining" / CANONICAL.name
SID_FIELDS = {"item_sid", "history_item_sid"}
FIELD_NAMES = [
    "user_id", "history_item_title", "item_title", "history_item_id", "item_id", "history_item_sid", "item_sid",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    return {"filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def require_sha(path, sha):
    if digest(path) != sha:
        raise ValueError(f"Source SHA-256 mismatch for {path}")


def sid_catalog(run, assignment):
    rq = run / "rqvae"
    summary = read(rq / "training_summary.json")
    selected_checkpoint = rq / summary["selected_checkpoint"]
    if summary["status"] != "completed" or summary["sources"]["embeddings_sha256"] != EMBEDDING_SHA:
        raise ValueError("Export requires the completed RQ-VAE training on the frozen original embeddings")
    if summary["sources"]["baseline_sid_sha256"] != LEGACY_SID_SHA:
        raise ValueError("RQ-VAE was trained against a different baseline catalog")
    require_sha(selected_checkpoint, summary["model_sha256"])
    codes_path = rq / "catalog_codes.npy"
    require_sha(codes_path, summary["catalog_codes_sha256"])
    metadata = {"name": "raw", "algorithm": "RQ-VAE raw nearest-code assignments",
                "unique_sids": summary["selected_metrics"]["all"]["unique_sids"],
                "codes_sha256": summary["catalog_codes_sha256"]}
    if assignment != "raw":
        repair = read(rq / "collision_repair.json")
        if repair["status"] != "completed" or repair["model_sha256"] != summary["model_sha256"]:
            raise ValueError("Collision repair must use the same verified RQ-VAE checkpoint")
        selected = repair["metrics"][assignment]
        output = rq / selected["codes_path"]
        require_sha(output, selected["codes_sha256"])
        before, after = np.load(codes_path, allow_pickle=False), np.load(output, allow_pickle=False)
        if after.shape != before.shape or not np.array_equal(before[:, :2], after[:, :2]):
            raise ValueError("Collision repair changed the learned first two tokens")
        codes_path = output
        labels = {"sinkhorn": "RQ-VAE with final-layer Sinkhorn collision repair",
                  "prefix_matching": "RQ-VAE with prefix-constrained minimum-distance last-code matching"}
        metadata = {"name": assignment, "algorithm": labels[assignment],
                    "unique_sids": selected["metrics"]["all"]["unique_sids"],
                    "codes_sha256": selected["codes_sha256"]}
    codes = np.load(codes_path, allow_pickle=False)
    if codes.shape != (CATALOG_SIZE, 3) or codes.dtype.kind not in "ui" or codes.min() < 0 or codes.max() >= 256:
        raise ValueError("RQ-VAE export requires exactly three integer codes in [0, 255] per item")
    return {
        str(item): [f"<{level}_{int(code)}>" for level, code in zip("abc", path, strict=True)]
        for item, path in enumerate(codes)
    }, summary, metadata


def remap_csv(source, destination, old, new, meta, collided, split):
    stable_columns = [name for name in FIELD_NAMES if name not in SID_FIELDS]
    left_hash, right_hash = hashlib.sha256(), hashlib.sha256()

    def stable_hash(row, checksum):
        content = json.dumps([row[key] for key in stable_columns], ensure_ascii=False, separators=(",", ":"))
        checksum.update(content.encode() + b"\n")

    targets, items, total, collision_rows = Counter(), set(), 0, 0
    with source.open(encoding="utf-8-sig", newline="") as inp, destination.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(inp)
        if reader.fieldnames != FIELD_NAMES:
            raise ValueError(f"Unexpected input columns: {source}")
        writer = csv.DictWriter(out, fieldnames=FIELD_NAMES, lineterminator="\n")
        writer.writeheader()
        for total, row in enumerate(reader, 1):
            item = str(row["item_id"])
            history = [str(value) for value in ast.literal_eval(row["history_item_id"])]
            if not 1 <= len(history) <= 10 or len(row) != len(FIELD_NAMES) or None in row:
                raise ValueError(f"Invalid history or column count in {source} on data row {total}")
            if (row["item_sid"] != old[item] or ast.literal_eval(row["history_item_sid"]) != [old[i] for i in history]
                    or row["item_title"] != meta[item]["title"]
                    or ast.literal_eval(row["history_item_title"]) != [meta[i]["title"] for i in history]):
                raise ValueError(f"Old split no longer matches catalog in {source} on data row {total}")
            stable_hash(row, left_hash)
            row["item_sid"] = new[item]
            row["history_item_sid"] = repr([new[i] for i in history])
            writer.writerow(row)
            targets[item] += 1
            items.update(history)
            items.add(item)
            collision_rows += item in collided
    with destination.open(encoding="utf-8", newline="") as inp:
        written = 0
        for written, row in enumerate(csv.DictReader(inp), 1):
            stable_hash(row, right_hash)
            history = [str(value) for value in ast.literal_eval(row["history_item_id"])]
            if row["item_sid"] != new[row["item_id"]] or ast.literal_eval(row["history_item_sid"]) != [
                new[i] for i in history
            ]:
                raise ValueError(f"New SID readback mismatch in {destination} on data row {written}")
    if total != BASELINE_ROWS[split] or written != total or right_hash.digest() != left_hash.digest():
        raise ValueError(f"Rows, ordering or non-SID fields changed for {split}")
    return {"non_sid_fields_sha256": right_hash.hexdigest(), "collided_target_rows": collision_rows,
            "targets": targets, "behavior_items": items}


def remap_info(source, destination, old, new, meta):
    ids = set()
    with source.open(encoding="utf-8-sig", newline="") as inp, destination.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        for line in inp:
            sid, title, item = line.rstrip("\r\n").split("\t")
            if item in ids or sid != old[item] or title != meta[item]["title"]:
                raise ValueError(f"Original info.txt catalog mismatch for item {item}")
            out.write(f"{new[item]}\t{title}\t{item}\n")
            ids.add(item)
    if ids != set(new):
        raise ValueError("info.txt did not cover the entire catalog")


def publish_verified(source, destination):
    sha = digest(source)
    if not destination.exists():
        publish(source, destination)
    require_sha(destination, sha)


def export(workspace, run, assignment="raw"):
    source = workspace / POSTTRAIN
    catalog_source = workspace / CANONICAL
    previous = read(source / "preparation_manifest.json")
    if previous["expected_sid_sha256"] != LEGACY_SID_SHA or previous["rows"] != BASELINE_ROWS:
        raise ValueError("The serving baseline provenance differs")
    index_file, meta_file = f"{DATASET}.index.json", f"{DATASET}.item.json"
    require_sha(catalog_source / index_file, LEGACY_SID_SHA)
    require_sha(catalog_source / meta_file, previous["sources"]["item_metadata"]["sha256"])
    old_index, metadata = read(catalog_source / index_file), read(catalog_source / meta_file)
    new_index, trained, assignment_record = sid_catalog(run, assignment)
    if set(old_index) != set(new_index) or set(metadata) != set(new_index):
        raise ValueError("RQ-VAE changed the product ID domain")
    old = {i: "".join(value) for i, value in old_index.items()}
    new = {i: "".join(value) for i, value in new_index.items()}
    groups = defaultdict(list)
    for item, sid in new.items():
        groups[sid].append(item)
    collisions = [{"sid": sid, "item_ids": ids, "titles": [metadata[i]["title"] for i in ids]}
                  for sid, ids in groups.items() if len(ids) > 1]
    collided = {i for entry in collisions for i in entry["item_ids"]}
    collision_counts = {"groups": len(collisions), "items": len(collided), "excess_items": CATALOG_SIZE - len(groups)}
    if len(groups) != assignment_record["unique_sids"]:
        raise ValueError("SID collision audit contradicts selected code assignment")
    prepared, catalog = run / "data", run / "catalog"
    ready = prepared / "preparation_manifest.json"
    if ready.exists():
        current = read(ready)
        if current["rqvae_model_sha256"] != trained["model_sha256"] or current["code_assignment"] != assignment_record:
            raise ValueError("Existing completed data use a different checkpoint or code assignment")
        for record in current["sources"].values():
            require_sha(catalog / record["filename"], record["sha256"])
        for record in current["outputs"].values():
            require_sha(prepared / record["filename"], record["sha256"])
        return read(run / "sid_export_summary.json")

    with tempfile.TemporaryDirectory(prefix="industrial-rqvae-export-") as temporary:
        root = Path(temporary)
        stage_data, stage_catalog = root / "data", root / "catalog"
        stage_data.mkdir()
        stage_catalog.mkdir()
        write(stage_catalog / index_file, new_index)
        shutil.copyfile(catalog_source / meta_file, stage_catalog / meta_file)
        source_records = {"sid_index": fingerprint(stage_catalog / index_file),
                          "item_metadata": fingerprint(stage_catalog / meta_file)}
        source_audit = previous["source_audit"]
        records = {}
        for split in BASELINE_ROWS:
            original_csv = source / f"{split}.csv"
            require_sha(original_csv, previous["outputs"][original_csv.name]["sha256"])
            records[split] = remap_csv(original_csv, stage_data / original_csv.name, old, new, metadata, collided, split)
            record = previous["sources"][split]
            original_inter = catalog_source / record["filename"]
            require_sha(original_inter, record["sha256"])
            shutil.copyfile(original_inter, stage_catalog / record["filename"])
            source_records[split] = fingerprint(stage_catalog / record["filename"])
            source_audit["splits"][split]["collided_target_rows"] = records[split]["collided_target_rows"]
        require_sha(source / "info.txt", previous["outputs"]["info.txt"]["sha256"])
        remap_info(source / "info.txt", stage_data / "info.txt", old, new, metadata)
        warm = records["train"]["behavior_items"]
        targets = records["test"]["targets"]
        cold_test = set(targets) - warm
        cold_count = sum(targets[i] for i in cold_test)
        if (len(warm), len(cold_test), cold_count) != (3647, 37, 147):
            raise ValueError("Product-level cold split changed")
        warm_sids = {new[i] for i in warm}
        ambiguous = {i for i in cold_test if new[i] in warm_sids}
        source_audit.update({
            "unique_sids": len(groups), "complete_sids_unique": not collisions,
            "sid_collisions": collision_counts, "sid_collision_groups": collisions,
            "cold_test_targets_in_collided_groups": sorted(cold_test & collided, key=int),
            "cold_test_targets_with_warm_sid": sorted(ambiguous, key=int),
            "cold_test_rows_with_warm_sid": sum(targets[i] for i in ambiguous),
        })
        info = {
            "status": "passed", "sid_sha256": source_records["sid_index"]["sha256"],
            "rqvae_model_sha256": trained["model_sha256"], "selected_epoch": trained["best_epoch"],
            "algorithm": assignment_record["algorithm"], "complete_sid_tokens": 3,
            "code_assignment": assignment_record,
            "rqvae_centered_embeddings": trained["config"].get("center_embeddings", False),
            "catalog_items": CATALOG_SIZE, "unique_sids": len(groups), "collisions": collision_counts,
            "occupied_codes_by_level": [len(set(level)) for level in zip(*new_index.values(), strict=True)],
            "collision_target_rows_by_split": {
                split: records[split]["collided_target_rows"] for split in BASELINE_ROWS
            },
            "baseline_unique_sids": previous["unique_sids"], "baseline_collisions": previous["sid_collisions"],
            "rows": BASELINE_ROWS, "cold_test_rows": cold_count,
            "non_sid_fields_sha256": {split: records[split]["non_sid_fields_sha256"] for split in BASELINE_ROWS},
        }
        manifest = previous | {
            "original_sid": False, "sid_generation": info["algorithm"], "baseline_sid_sha256": LEGACY_SID_SHA,
            "rqvae_centered_embeddings": info["rqvae_centered_embeddings"], "code_assignment": assignment_record,
            "rqvae_model_sha256": info["rqvae_model_sha256"], "expected_sid_sha256": info["sid_sha256"],
            "selected_sid_epoch": info["selected_epoch"], "complete_sid_tokens": 3,
            "complete_sids_unique": not collisions, "unique_sids": len(groups),
            "sid_collisions": collision_counts, "sid_collision_groups": collisions,
            "input_directory": str(catalog), "output_directory": str(prepared),
            "sources": source_records, "source_audit": source_audit,
            "outputs": {path.name: fingerprint(path) for path in stage_data.iterdir()},
            "validation": "passed", "check_only": False,
            "evaluation_identity": "Item sequences fixed; SID collisions cannot be distinguished by SID-only scoring.",
            "changed_csv_columns": ["history_item_sid", "item_sid"],
            "non_sid_fields_sha256": info["non_sid_fields_sha256"],
        }
        write(stage_catalog / "cleaning_report.json", {
            "dataset": DATASET, "source": "Frozen Industrial release with three-code RQ-VAE SID remap",
            "validation": "passed", "users": source_audit["users_across_all_splits"], "items": CATALOG_SIZE,
            "examples": sum(BASELINE_ROWS.values()), "split_examples": BASELINE_ROWS, "source_audit": source_audit,
        })
        write(root / "sid_export_summary.json", info)
        for path in stage_catalog.iterdir():
            publish_verified(path, catalog / path.name)
        for path in stage_data.iterdir():
            publish_verified(path, prepared / path.name)
        publish_verified(root / "sid_export_summary.json", run / "sid_export_summary.json")
        manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(stage_data / "preparation_manifest.json", manifest)
        publish_verified(stage_data / "preparation_manifest.json", ready)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assignment", choices=("raw", "sinkhorn", "prefix_matching"), default="raw")
    args = parser.parse_args()
    print(json.dumps(export(args.workspace.resolve(), args.run_dir.resolve(), args.assignment), ensure_ascii=False))


if __name__ == "__main__":
    main()
