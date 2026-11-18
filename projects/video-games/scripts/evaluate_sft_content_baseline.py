#!/usr/bin/env python3
"""Evaluate a fixed, training-free content retrieval diagnostic.

Normalize the saved item vectors, average each history's normalized vectors,
normalize that mean, and rank the full catalog by cosine similarity. Targets
are used only after ranking to compute metrics. This does not modify SFT.
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
import time


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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def paired_rows(raw_path, csv_path, catalog):
    with Path(raw_path).open(encoding="utf-8-sig", newline="") as raw_stream, \
            Path(csv_path).open(encoding="utf-8-sig", newline="") as csv_stream:
        raw = csv.reader(raw_stream, delimiter="\t")
        require(next(raw, None) == ["user_id:token", "item_id_list:token_seq", "item_id:token"],
                f"{Path(raw_path).name}: unexpected header")
        prepared = csv.DictReader(csv_stream)
        require(prepared.fieldnames is not None and {"item_id", "item_sid"} <= set(prepared.fieldnames),
                f"{Path(csv_path).name}: missing target columns")
        for row_id, (fields, row) in enumerate(zip_longest(raw, prepared)):
            require(fields is not None and row is not None and len(fields) == 3,
                    f"{Path(raw_path).name}: raw/prepared row count or format differs")
            _, history_text, target = fields
            history = history_text.split()
            require(1 <= len(history) <= 20 and all(item in catalog for item in [*history, target]),
                    f"{Path(raw_path).name}:{row_id}: history or target outside approved catalog")
            require(row.get("item_id") == target and row.get("item_sid") == catalog[target],
                    f"{Path(csv_path).name}:{row_id}: target/order differs from raw interaction")
            yield [int(item) for item in history], int(target)


def rank_histories(normalized_items, histories, limit=50):
    """Rank histories independently of their targets; ties prefer lower item ID."""
    import torch

    require(bool(histories), "A retrieval batch must not be empty")
    width = max(map(len, histories))
    ids = torch.zeros((len(histories), width), dtype=torch.long, device=normalized_items.device)
    mask = torch.zeros((len(histories), width), dtype=torch.bool, device=normalized_items.device)
    for index, history in enumerate(histories):
        require(bool(history), "History must not be empty")
        ids[index, :len(history)] = torch.tensor(history, dtype=torch.long, device=ids.device)
        mask[index, :len(history)] = True
    means = (normalized_items[ids] * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(means, dim=1, keepdim=True)
    require(bool(torch.isfinite(norms).all()) and bool((norms > 0).all()),
            "A history mean has zero or nonfinite norm; cosine retrieval is undefined")
    queries = means / norms
    scores = queries @ normalized_items.T
    # Columns are numeric item IDs in ascending order, so stable descending sort
    # resolves exact score ties by ascending item ID without perturbing scores.
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(limit, len(normalized_items))]


def sft_comparison(split, run_dir, hashes, strata, paths):
    if split != "test":
        return {"status": "unavailable", "reason": "No saved SFT validation ranking evaluation is used by this diagnostic."}
    metric_path, shift_path = run_dir / "evaluation/metrics.json", run_dir / "behavior_shift_analysis.json"
    if not metric_path.is_file() or not shift_path.is_file():
        return {"status": "unavailable", "reason": "Accepted SFT evaluation and behavior strata are unavailable."}
    for key, path in (("sft_evaluation_metrics", metric_path), ("sft_behavior_shift", shift_path)):
        paths[key] = path
        hashes[key] = sha256(path)
    evaluation, shift = read_json(metric_path), read_json(shift_path)
    matched = (evaluation.get("status") == shift.get("status") == "passed"
               and shift.get("source_sha256", {}).get("evaluation_metrics") == hashes["sft_evaluation_metrics"])
    for key in ("train_file", "test_file", "sid_index_path"):
        actual_key = "split_file" if key == "test_file" else key
        matched = matched and evaluation.get("source_sha256", {}).get(key) == hashes[actual_key]
        matched = matched and shift.get("source_sha256", {}).get(key) == hashes[actual_key]
    matched = matched and evaluation.get("item_meta_sha256") == hashes["item_meta_path"]
    if not matched:
        return {"status": "unavailable", "reason": "SFT comparison artifacts do not match these exact data identities."}
    sft_strata = {row.get("key"): row for row in shift.get("strata", []) if isinstance(row, dict)}
    result = []
    for row in strata:
        original = sft_strata.get(row["key"], {})
        if original.get("rows") != row["rows"]:
            return {"status": "unavailable", "reason": "SFT and content stratum row counts differ."}
        metrics = original.get("metrics", {})
        if not all(metrics.get(key) is None if value is None else
                   isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
                   for key, value in row["metrics"].items()):
            return {"status": "unavailable", "reason": "SFT stratum metrics are missing or invalid."}
        result.append({"key": row["key"], "rows": row["rows"], "sft_metrics": metrics,
                       "content_metrics": row["metrics"],
                       "content_minus_sft": {key: value - metrics[key] if value is not None else None
                                             for key, value in row["metrics"].items()}})
    return {"status": "available", "strata": result,
            "interpretation": "Post-hoc descriptive comparison; no hybrid, training update, or held-out confirmatory claim."}


def evaluate(embedding_path, input_dir, data_dir, run_dir, split="valid", device="auto", batch_size=256):
    started = time.monotonic()
    require(split in {"valid", "test"}, "split must be valid or test")
    require(device in {"auto", "cpu", "cuda"}, "device must be auto, cpu or cuda")
    require(isinstance(batch_size, int) and batch_size > 0, "batch_size must be positive")
    embedding_path = Path(embedding_path).resolve(strict=True)
    input_dir, data_dir, run_dir = map(lambda value: Path(value).resolve(strict=True), (input_dir, data_dir, run_dir))
    paths = {
        "embedding": embedding_path, "embedding_manifest": Path(str(embedding_path) + ".manifest.json"),
        "embedding_ids": Path(str(embedding_path) + ".ids.json"),
        "sid_index_path": input_dir / "Video_Games.index.json", "item_meta_path": input_dir / "Video_Games.item.json",
        "train_file": data_dir / "train.csv", "split_file": data_dir / f"{split}.csv",
        "train_inter": input_dir / "Video_Games.train.inter", "split_inter": input_dir / f"Video_Games.{split}.inter",
        "preparation_manifest": data_dir / "preparation_manifest.json", "run_config": run_dir / "run_config.json",
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    embedding_manifest, preparation, config = (read_json(paths[key]) for key in
                                               ("embedding_manifest", "preparation_manifest", "run_config"))
    indices, metadata, row_ids = (read_json(paths[key]) for key in ("sid_index_path", "item_meta_path", "embedding_ids"))
    require(isinstance(indices, dict) and indices and set(indices) == set(metadata), "Catalog/metadata IDs differ")
    size = len(indices)
    require(set(indices) == {str(index) for index in range(size)}, "Catalog item IDs must be contiguous from zero")
    require(isinstance(row_ids, list) and len(row_ids) == size
            and all(type(value) is int and value == index for index, value in enumerate(row_ids)),
            "Embedding row IDs must match contiguous ascending catalog IDs")
    require(all(isinstance(tokens, list) and tokens and all(isinstance(token, str) and token for token in tokens)
                for tokens in indices.values()), "Invalid SID catalog")
    catalog = {item: "".join(tokens) for item, tokens in indices.items()}
    require(len(set(catalog.values())) == size, "Complete SIDs must be unique")
    require(embedding_manifest.get("validation") == "passed" and embedding_manifest.get("dataset") == "Video_Games",
            "Embedding manifest must have passed Video_Games validation")
    require(embedding_manifest.get("embedding_sha256") == hashes["embedding"], "Embedding SHA-256 mismatch")
    require(embedding_manifest.get("item_metadata_sha256") == hashes["item_meta_path"], "Embedding metadata SHA-256 mismatch")
    require(embedding_manifest.get("row_order") == "contiguous item_id ascending from zero",
            "Embedding manifest row order is unsupported")
    require(preparation.get("validation") == "passed" and preparation.get("dataset") == "Video_Games"
            and preparation.get("items") == size, "Prepared data identity is invalid")
    expected = {
        "train_file": preparation.get("outputs", {}).get("train.csv", {}).get("sha256"),
        "split_file": preparation.get("outputs", {}).get(f"{split}.csv", {}).get("sha256"),
        "train_inter": preparation.get("sources", {}).get("train", {}).get("sha256"),
        "split_inter": preparation.get("sources", {}).get(split, {}).get("sha256"),
        "sid_index_path": preparation.get("sources", {}).get("sid_index", {}).get("sha256"),
        "item_meta_path": preparation.get("sources", {}).get("item_metadata", {}).get("sha256"),
    }
    for key, value in expected.items():
        require(isinstance(value, str) and value == hashes[key], f"{key}: preparation SHA-256 mismatch")
    require(config.get("data", {}).get("category") == "Video_Games", "SFT run category differs")
    config_hashes = config.get("data", {}).get("sha256", {})
    for key in ("train_file", "sid_index_path", "item_meta_path"):
        require(config_hashes.get(key) == hashes[key], f"{key}: SFT run SHA-256 mismatch")
    if split == "valid":
        require(config_hashes.get("eval_file") == hashes["split_file"], "Validation file differs from SFT run")

    behavior, train_rows = set(), 0
    for history, target in paired_rows(paths["train_inter"], paths["train_file"], catalog):
        behavior.update(history)
        behavior.add(target)
        train_rows += 1
    require(train_rows == preparation.get("rows", {}).get("train"), "Training row count mismatch")
    rows = [(history, target, target not in behavior)
            for history, target in paired_rows(paths["split_inter"], paths["split_file"], catalog)]
    require(rows and len(rows) == preparation.get("rows", {}).get(split), "Evaluation split row count mismatch")

    import numpy as np
    import torch

    array = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    require(array.ndim == 2 and array.shape[0] == size and array.shape[1] > 0 and array.dtype == np.float32,
            "Embedding matrix must be catalog-sized float32 vectors")
    require(list(array.shape) == embedding_manifest.get("shape") and embedding_manifest.get("dtype") == "float32",
            "Embedding shape/dtype differs from generation manifest")
    chosen_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    require(chosen_device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vectors = torch.from_numpy(np.array(array, copy=True)).to(chosen_device)
    require(bool(torch.isfinite(vectors).all()), "Embedding matrix contains nonfinite values")
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    require(bool((norms > 0).all()), "Embedding matrix contains a zero-norm item")
    vectors = vectors / norms
    accumulators = {key: {"key": key, "rows": 0, "hits": Counter(), "discounted": Counter()}
                    for key in ("overall", "behavior_warm", "behavior_cold")}
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            predicted = rank_histories(vectors, [history for history, _, _ in batch], max(TOP_K)).cpu().tolist()
            for (_, target, cold), candidates in zip(batch, predicted):
                rank = candidates.index(target) + 1 if target in candidates else math.inf
                for key in ("overall", "behavior_cold" if cold else "behavior_warm"):
                    acc = accumulators[key]
                    acc["rows"] += 1
                    for k in TOP_K:
                        if rank <= k:
                            acc["hits"][k] += 1
                            acc["discounted"][k] += 1 / math.log2(rank + 1)
    strata = []
    for key in accumulators:
        acc = accumulators[key]
        count = acc["rows"]
        strata.append({"key": key, "rows": count, "share": count / len(rows),
                       "hits_by_k": {str(k): acc["hits"][k] for k in TOP_K},
                       "metrics": {f"{metric}@{k}": acc[field][k] / count if count else None
                                   for metric, field in (("HR", "hits"), ("NDCG", "discounted")) for k in TOP_K}})
    comparison = sft_comparison(split, run_dir, hashes, strata, paths)
    for key, path in paths.items():
        require(sha256(path) == hashes[key], f"{key}: source changed during retrieval diagnostic")
    return {
        "schema_version": 1, "status": "passed", "dataset": "Video_Games", "split": split,
        "method": "normalized_history_mean_cosine_full_catalog",
        "method_name": "Equal-weight normalized-history content retrieval (no training)",
        "catalog_items": size, "split_rows": len(rows), "training_behavior_items": len(behavior),
        "source_sha256": hashes, "source_paths": {key: str(path) for key, path in paths.items()},
        "methodology": {
            "item_vectors": "L2-normalize every existing saved item vector",
            "query": "Equal-weight mean of normalized vectors for every ordered history position, followed by L2 normalization",
            "history_limit": 20, "catalog": "Full shared catalog; no seen-item exclusion",
            "ranking": "Descending cosine similarity; exact score ties resolved by ascending numeric item ID",
            "max_k": 50, "candidate_count": min(50, size), "target_used_for_ranking": False,
            "training_performed": False, "new_encoder_or_api_used": False,
            "cold_definition": "Target absent from all training behavior histories and targets; catalog metadata remains known",
            "test_row_deduplication": False, "choices_fixed_before_execution": True,
        },
        "precision": {"dtype": "float32", "matmul_precision": "highest", "tf32_allowed": False,
                      "amp_used": False, "device": chosen_device, "batch_size": batch_size,
                      "torch_version": torch.__version__, "numpy_version": np.__version__,
                      "embedding_was_normalized": embedding_manifest.get("normalized")},
        "strata": strata, "actual_baseline_comparison": comparison,
        "interpretation": [
            "Post-hoc mechanism diagnostic using already known item text vectors, not an improved trained SFT model or a trained hybrid solution.",
            "Cold/warm metrics preserve every original evaluation row and the full-catalog ranking protocol.",
            "Validation is evaluated first. Test results are descriptive diagnostics after prior test inspection and must not be used to tune this fixed method.",
            "Content retrieval gains, if observed, motivate a separate validation-selected candidate fusion experiment; they do not prove such a hybrid will improve final ranking.",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    result = evaluate(args.embedding_path, args.input_dir, args.data_dir, args.run_dir,
                      split=args.split, device=args.device, batch_size=args.batch_size)
    output = args.output.resolve()
    require(output not in {Path(path) for path in result["source_paths"].values()},
            "Output must not overwrite a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".content-diagnostic-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "split": result["split"], "output": str(output),
                      "split_rows": result["split_rows"], "strata": result["strata"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
