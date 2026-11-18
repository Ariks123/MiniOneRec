"""Measure SASRec on the archived Industrial recommendation timing workload."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hybrid.benchmark import timed_batches, warmup
from hybrid.experiment import dump
from sasrec_baseline.data import EXPECTED_ROWS, HISTORY_LIMIT, KS, read_json, read_split, sha256


class SASRecOnline:
    """Map original IDs and materialize the ranked results on each request."""

    def __init__(self, model, device, original_ids, popularity):
        if sorted(set(original_ids)) != original_ids or len(original_ids) != 3647:
            raise ValueError("Expected 3,647 distinct sorted training-seen item IDs")
        if len(popularity) != len(original_ids) or set(popularity) != set(original_ids):
            raise ValueError("Popularity order does not cover the training-seen items")
        self.model = model
        self.device = device
        self.internal = {item: idx for idx, item in enumerate(original_ids, start=1)}
        self.external = torch.tensor(original_ids, dtype=torch.long, device=device)
        self.popularity = popularity

    @torch.inference_mode()
    def online(self, examples, config, _catalog):
        k = config["k"]
        outputs = [None] * len(examples)
        padded, lengths, positions = [], [], []
        for idx, row in enumerate(examples):
            known = [self.internal[item] for item in row["history"] if item in self.internal]
            if not known:
                outputs[idx] = self.popularity[:k]
                continue
            padded.append(known + [0] * (HISTORY_LIMIT - len(known)))
            lengths.append(len(known))
            positions.append(idx)
        if padded:
            item_seq = torch.tensor(padded, dtype=torch.long).to(self.device)
            item_seq_len = torch.tensor(lengths, dtype=torch.long).to(self.device)
            scores = self.model(item_seq, item_seq_len)
            order = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
            for idx, items in zip(positions, self.external[order].tolist(), strict=True):
                outputs[idx] = items
        return outputs


def validation_sample(run, archive_protocol, asset_manifest, test_csv):
    protocol = read_json(archive_protocol)
    expected = {
        "device": "NVIDIA H20",
        "framework": "2.8.0+cu128",
        "seed": 42,
        "repeats": 3,
        "warmup_batches_per_shape": 3,
        "latency_batch_size": 1,
        "throughput_batch_size_per_gpu": 4,
    }
    if any(protocol[k] != value for k, value in expected.items()):
        raise ValueError("The archived workload or GPU environment changed")
    ids = protocol["sample_row_ids"]
    if len(ids) != 256 or len(set(ids)) != 256 or not all(0 <= row < EXPECTED_ROWS["test"] for row in ids):
        raise ValueError("The archived workload must identify 256 distinct test rows")
    if sha256(asset_manifest) != protocol["asset_manifest_sha256"]:
        raise ValueError("The asset manifest differs from the archived workload")
    test_sha256 = read_json(asset_manifest)["files"]["data/test.csv"]["sha256"]
    if test_csv.name != "test.csv":
        raise ValueError("Expected the archived split filename test.csv")
    test = read_split(test_csv.parent, "test", test_sha256)

    selection = read_json(run / "selection.json")
    if selection["split_sha256"]["test"] != test_sha256:
        raise ValueError("SASRec used a different test split")
    predictions = run / "test_predictions.jsonl"
    for path in (run / "model.ts", predictions):
        if sha256(path) != selection["artifact_sha256"][path.name]:
            raise ValueError(f"Selected SASRec artifact changed: {path.name}")
    with predictions.open(encoding="utf-8") as stream:
        predicted = [json.loads(line) for line in stream]
    if len(predicted) != len(test):
        raise ValueError("SASRec and archived inference must use all 4,533 test requests")
    warm = set(selection["sorted_warm_original_ids"])
    for idx, (row, result) in enumerate(zip(test, predicted, strict=True)):
        target = row.target
        if (result["row_id"] != idx or result["target"] != target
                or result["is_cold"] != (target not in warm)
                or len(result["ranking"]) != 50 or len(set(result["ranking"])) != 50
                or not warm.issuperset(result["ranking"])):
            raise ValueError(f"SASRec prediction and archived request differ at test row {idx}")
    examples = [{"row_id": idx, "history": test[idx].history, "expected": predicted[idx]["ranking"]} for idx in ids]
    return protocol, selection, examples


def verify_ranking(engine, examples):
    counts = {}
    for size in (1, 4):
        mismatches = []
        for offset in range(0, len(examples), size):
            batch = examples[offset:offset + size]
            observed = engine.online(batch, {"k": 50}, None)
            mismatches.extend(row["row_id"] for ranking, row in zip(observed, batch, strict=True)
                              if ranking != row["expected"])
        counts[f"batch{size}_exact_ranking"] = len(examples) - len(mismatches)
        if mismatches:
            raise ValueError(f"Exported batch={size} rankings differ for rows {mismatches[:10]}")
    return counts


def single_gpu(engine, examples, config, warmups, repeats):
    warmup(engine, None, examples, config, 1, warmups)
    torch.cuda.reset_peak_memory_stats()
    durations, first_items, request_times = timed_batches(
        engine, None, examples, config, 1, repeats, latency=True,
    )
    peak_batch1 = torch.cuda.max_memory_allocated()
    warmup(engine, None, examples, config, 4, warmups)
    torch.cuda.reset_peak_memory_stats()
    batch_times, batch_items, _ = timed_batches(engine, None, examples, config, 4, repeats)
    n_rows = len(examples) * repeats
    k = config["k"]
    if not (len(request_times) == n_rows and first_items == batch_items == n_rows * k):
        raise ValueError(f"Single-GPU timing lost requests or recommendations at K={k}")
    p50, p95, p99 = np.percentile(request_times, [50, 95, 99]).tolist()
    result = {
        "latency_p50_ms": p50, "latency_p95_ms": p95, "latency_p99_ms": p99,
        "latency_repeat_seconds": durations, "single_repeat_seconds": batch_times,
        "latency_samples": len(request_times), "logical_requests_1gpu": n_rows,
        "recommendations_batch1": first_items, "recommendations_1gpu": batch_items,
        "requests_s_1gpu": n_rows / sum(batch_times), "items_s_1gpu": batch_items / sum(batch_times),
        "gpu_peak_bytes_batch1": peak_batch1, "gpu_peak_bytes_batch4": torch.cuda.max_memory_allocated(),
    }
    samples = [(k, repeat, row["row_id"], latency)
               for repeat in range(1, repeats + 1)
               for row, latency in zip(examples, request_times[(repeat - 1) * len(examples):repeat * len(examples)],
                                       strict=True)]
    return result, samples


def four_gpu(engine, examples, config, rank, warmups, repeats):
    local = examples[rank::4]
    warmup(engine, None, local, config, 4, warmups)
    seconds, item_count = [], 0
    for _ in range(repeats):
        torch.cuda.synchronize()
        dist.barrier()
        local_seconds, local_items, _ = timed_batches(engine, None, local, config, 4, 1)
        outcomes = [None] * 4
        dist.all_gather_object(outcomes, (local_seconds[0], local_items))
        seconds.append(max(elapsed for elapsed, _ in outcomes))
        item_count += sum(count for _, count in outcomes)
    n_rows = len(examples) * repeats
    if item_count != n_rows * config["k"]:
        raise ValueError(f"Four-GPU timing lost recommendations at K={config['k']}")
    return {
        "requests_s_4gpu": n_rows / sum(seconds), "items_s_4gpu": item_count / sum(seconds),
        "logical_requests_4gpu": n_rows, "recommendations_4gpu": item_count, "four_gpu_repeat_seconds": seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "archive-protocol", "asset-manifest", "test-csv", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol, selection, examples = validation_sample(
        args.run_dir, args.archive_protocol, args.asset_manifest, args.test_csv,
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or local_rank != rank:
        raise ValueError("Start this benchmark on one four-GPU node")
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_name(local_rank) != protocol["device"] or str(torch.__version__) != protocol["framework"]:
        raise ValueError("This process is not running in the archived NVIDIA H20 timing environment")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", local_rank)
    engine = SASRecOnline(torch.jit.load(str(args.run_dir / "model.ts"), map_location=device).eval(), device,
                          selection["sorted_warm_original_ids"], selection["popularity_fallback"])
    dist.init_process_group("gloo")
    try:
        parity = verify_ranking(engine, examples)
        all_parity = [None] * 4
        dist.all_gather_object(all_parity, parity)
        configs = [{"id": f"sasrec-k{k}", "drafter": "sasrec", "k": k} for k in KS]
        repeats, warmups = protocol["repeats"], protocol["warmup_batches_per_shape"]
        singles = {}
        for cfg in configs[rank::4]:
            singles[cfg["k"]] = single_gpu(engine, examples, cfg, warmups, repeats)
        all_singles = [None] * 4
        dist.all_gather_object(all_singles, singles)
        measured_singles = {k: result for records in all_singles for k, result in records.items()}
        benchmarks, raw_samples = [], []
        for cfg in configs:
            single, samples = measured_singles[cfg["k"]]
            multi = four_gpu(engine, examples, cfg, rank, warmups, repeats)
            benchmarks.append(cfg | single | multi)
            raw_samples.extend(samples)
            if rank == 0:
                print(f"Finished K={cfg['k']} B1 latency, B4 single-GPU and four-GPU throughput", flush=True)
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            reported = {key: protocol[key] for key in (
                "sample_row_ids", "seed", "repeats", "warmup_batches_per_shape", "latency_batch_size",
                "throughput_batch_size_per_gpu", "latency_unit", "framework", "device", "four_gpu_throughput",
                "single_gpu_context", "asset_manifest_sha256",
            )}
            reported.update(
                timing="resident model, raw history item IDs to materialized ranked original item IDs, GPU synchronized",
                cache_scope="no cross-request scoring or result cache",
                model_sha256=selection["artifact_sha256"]["model.ts"], ranking_parity=all_parity,
                archival_protocol_sha256=sha256(args.archive_protocol), test_csv_sha256=sha256(args.test_csv),
                tpot="not applicable",
            )
            with (args.output_dir / "benchmark_latency.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("k", "repeat", "row_id", "latency_ms"))
                writer.writerows(raw_samples)
            dump(args.output_dir / "benchmark_protocol.json", reported)
            dump(args.output_dir / "benchmarks.json", benchmarks)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
