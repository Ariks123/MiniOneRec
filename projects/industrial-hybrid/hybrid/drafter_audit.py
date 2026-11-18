"""CPU audit of the deployed drafters and first-round approximate retrieval."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

import numpy as np

from .drafters import RAW_SHA, SID_SHA, UNI_SHA, _sha, build_drafter
from .experiment import dump

DELTAS = (2, 10, 20, 40, 100)


def checked_sha(path, expected):
    digest = _sha(path)
    if digest != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    return digest


def provenance(assets):
    prepared_manifest = assets / "raw_manifest.json"
    unisrec_manifest = assets / "unisrec/metadata.json"
    qwen_manifest = assets / "qwen/item_index/manifest.json"
    prepared = json.loads(prepared_manifest.read_text())
    uni = json.loads(unisrec_manifest.read_text())
    emb = prepared["embedding"]
    rows_path = assets / "raw_row_ids.json"
    ids = json.loads(rows_path.read_text())
    if ids != list(map(str, range(3686))):
        raise ValueError("Raw row IDs differ from the published contiguous mapping")
    provenance = uni["provenance"]
    raw_digests = {
        "prepared_manifest": checked_sha(prepared_manifest, provenance["manifest_sha256"]),
        "row_ids": checked_sha(rows_path, provenance["embedding_ids_sha256"]),
        "vectors": checked_sha(assets / "raw.npy", RAW_SHA),
        "sid_catalog": checked_sha(assets / "sid.json", SID_SHA),
    }
    if raw_digests["vectors"] != emb["sha256"] or raw_digests["vectors"] != provenance["embedding_sha256"]:
        raise ValueError("Raw vectors do not match the UniSRec feature input")
    internal = uni["internal_ids_by_item_id"]
    if sorted(internal) != list(range(1, len(ids) + 1)) or uni["checkpoint_sha256"] != UNI_SHA:
        raise ValueError("Unexpected UniSRec checkpoint or row mapping")
    uni_files = {name: checked_sha(assets / "unisrec" / name, sha)
                 for name, sha in uni["checksums"].items()}

    sys.path.insert(0, str(assets / "qwen_source"))
    from qwen_retrieval.config import checkpoint_identity
    from qwen_retrieval.encode_items import load_index

    identity = checkpoint_identity(assets / "qwen/final_checkpoint")
    _, sid_rows, qwen = load_index(assets / "qwen/item_index", assets / "sid.json", identity)
    return {
        "ann": {"sha256": raw_digests, "rows": len(ids), "dimensions": emb["shape"][1],
                "row_ids_source": emb["row_ids_source"],
                "row_order_provenance_verified": emb["row_order_provenance_verified"],
                "model_identity_verified": emb["model_identity_verified"],
                "companion_metadata_matches_catalog": emb["companion_metadata_matches_catalog"]},
        "unisrec": {"manifest_sha256": _sha(unisrec_manifest), "files_sha256": uni_files,
                    "checkpoint_sha256": UNI_SHA, "checkpoint_epoch_zero_based": uni["checkpoint_epoch_zero_based"],
                    "reference_parity_max_abs_on_export": uni["max_abs_error_cpu"],
                    "recbole_user_item_mapping": "Export reorders trained item outputs by numeric catalog ID, using the RecBole field-to-token table. PAD is internal ID 0.",
                    "items": len(internal), "internal_id_of_catalog_item_zero": internal[0],
                    "internal_tokens_sha256": provenance["internal_item_tokens_sha256"],
                    "feature_order_lineage": "Shares the direct drafter raw matrix and its historical row-order evidence limit."},
        "qwen": {"manifest_sha256": _sha(qwen_manifest), "checkpoint_identity": identity,
                 "projection_parameter": qwen["projection_parameter"], "vectors": qwen["matrix"],
                 "mapping": qwen["mapping"], "catalog_rows_sha256": qwen["catalog_rows_sha256"],
                 "sid_rows": len(sid_rows), "items": sum(len(row["item_ids"]) for row in sid_rows),
                 "row_order_check": "Independent build_catalog(sid.json) matches each SID and its complete original item ID list."},
    }


def audit_shard(args):
    rank, assets, samples = args
    cat = json.loads((assets / "sid.json").read_text())
    start = time.perf_counter()
    ann = build_drafter("ann", cat, "cpu", assets)
    built = time.perf_counter()
    queries = ann.prepare([history for _, history in samples])
    ready = time.perf_counter()
    overlaps = []
    for (row_id, _), query in zip(samples, queries, strict=True):
        exact = ann.exact_reference(query, max(DELTAS))
        result = {}
        for delta in DELTAS:
            picks = ann.draft(query, limit=delta)
            if len(set(picks)) != delta:
                raise ValueError(f"Missing or duplicate ANN candidates in row {row_id}")
            result[delta] = len(set(picks).intersection(exact[:delta]))
        overlaps.append(result)
    return {"rank": rank, "rows": len(samples), "cold_start_seconds": built - start,
            "prepare_seconds": ready - built, "index": ann.metadata, "overlaps": overlaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path, default=Path("/run/hybrid/ann_audit.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.workers <= 256:
        parser.error("workers must be between 1 and 256")
    valid = args.assets / "data/valid.csv"
    with valid.open() as stream:
        histories = [ast.literal_eval(row["history_item_id"]) for row in csv.DictReader(stream)]
    selection = sorted(random.Random(args.seed).sample(range(len(histories)), 256))
    sample = [(i, histories[i]) for i in selection]
    shards = [(r, args.assets, sample[r::args.workers]) for r in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(audit_shard, shards))
    overlap = [row for result in results for row in result.pop("overlaps")]
    stats = {}
    for delta in DELTAS:
        values = [o[delta] for o in overlap]
        stats[str(delta)] = {"hits": sum(values), "denominator": len(values) * delta,
                             "mean_candidate_recall": sum(values) / (len(values) * delta),
                             "perfect_queries": values.count(delta), "min_candidate_recall": min(values) / delta}
    report = {
        "metric": "First-round ANN candidate set recall versus exact full-catalog cosine, unfiltered; sorted ties by numeric item ID. No next-item labels used.",
        "split": "valid", "validation_file_sha256": _sha(valid), "validation_rows": len(histories),
        "seed": args.seed, "samples": len(sample), "row_ids_zero_based": selection,
        "versions": {p: version(p) for p in ("faiss-cpu", "numpy", "torch", "transformers")},
        "concurrent_cpu_build_and_prepare": results, "deltas": stats, "sources": provenance(args.assets),
    }
    dump(args.output, report)
    print(json.dumps({"output": str(args.output), "deltas": stats,
                      "rank_build_prepare": [[r["cold_start_seconds"], r["prepare_seconds"]] for r in results]}))


if __name__ == "__main__":
    main()
