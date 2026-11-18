"""Evaluate Qwen unique-SID vector retrieval followed by causal-LM reranking."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

from qwen_retrieval.config import APPROVED_SID_SHA256, CATEGORY, checkpoint_identity, hash_file
from qwen_retrieval.encode_items import load_index, load_joint_model, synchronize, utc_now
from scripts.sft_eval_utils import atomic_json, require, verify_artifacts, verify_sid_tokenizer


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "data-dir", "input-dir", "index-dir", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--split", choices=("valid", "test"), default="test")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--candidate-batch-size", type=int, default=32)
    result.add_argument("--top-delta", type=int, default=200)
    result.add_argument("--cutoff-len", type=int, default=512)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=("auto", "cpu"), default="auto")
    result.add_argument("--expected-sid-sha256", default=APPROVED_SID_SHA256)
    return result


def sid_token_ids(tokenizer, sid):
    """Never score EOS/newline or partial SIDs; original catalog has three atoms."""
    import re

    tokens = re.findall(r"<[abc]_[0-9]+>", sid)
    require(len(tokens) == 3 and "".join(tokens) == sid, "Expected a complete original three-token SID")
    ids = tokenizer.encode(sid, add_special_tokens=False)
    require(len(ids) == 3 and all(tokenizer.encode(token, add_special_tokens=False) == [token_id]
                                for token, token_id in zip(tokens, ids)), "SID tokens are not atomic")
    return ids


def candidate_log_probabilities(logits, labels):
    """Mean full-vocabulary log P of the candidate, with the causal one-token shift."""
    import torch

    shifted = labels[:, 1:]
    mask = shifted != -100
    counts = mask.sum(dim=1)
    require(bool((counts == 3).all()), "Every candidate must score exactly three SID tokens")
    # Select only the three candidate positions before expanding bfloat16 to fp32.
    selected = logits[:, :-1, :][mask].float()
    targets = shifted[mask]
    values = torch.nn.functional.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
    require(bool(torch.isfinite(values).all()), "Non-finite candidate likelihood")
    return values.reshape(labels.shape[0], 3).mean(dim=1)


def score_candidates_bruteforce(model, tokenizer, prompts, candidate_ids, *, batch_size, device):
    """Score aligned prompt/candidate pairs; prompts are encoded separately by EvalSidDataset."""
    import torch

    require(len(prompts) == len(candidate_ids), "Prompt/candidate counts differ")
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            group = list(zip(prompts[offset:offset + batch_size], candidate_ids[offset:offset + batch_size]))
            require(all(prompt and len(ids) == 3 for prompt, ids in group), "Missing prompt or incomplete candidate")
            examples = [{"input_ids": prompt + ids, "attention_mask": [1] * (len(prompt) + 3)} for prompt, ids in group]
            encoded = tokenizer.pad(examples, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
            width = batch["input_ids"].shape[1]
            labels = torch.full_like(batch["input_ids"], -100)
            for index, (prompt, ids) in enumerate(group):
                padding = width - len(prompt) - 3
                start = padding + len(prompt) if tokenizer.padding_side == "left" else len(prompt)
                labels[index, start:start + 3] = torch.tensor(ids, device=device)
            output = model(**batch, use_cache=False)
            scores.extend(candidate_log_probabilities(output.logits, labels).cpu().tolist())
    return scores


def score_candidates(model, tokenizer, prompts, candidate_ids, *, batch_size, device):
    """Exact teacher forcing with one shared prefill per distinct history prompt.

    Transformers DynamicCache repeats/mutates in place. Each candidate chunk gets
    an independent copy, so suffix scoring cannot corrupt the original prefix.
    """
    import copy
    import torch

    require(len(prompts) == len(candidate_ids) and batch_size > 0, "Invalid candidate batches")
    groups = {}
    for index, (prompt, ids) in enumerate(zip(prompts, candidate_ids)):
        require(bool(prompt) and len(ids) == 3, "Missing prompt or incomplete candidate")
        groups.setdefault(tuple(prompt), []).append(index)
    scores = [None] * len(prompts)
    with torch.inference_mode():
        for prompt, positions in groups.items():
            prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            prefix = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                           use_cache=True, logits_to_keep=1)
            require(prefix.past_key_values is not None
                    and hasattr(prefix.past_key_values, "batch_repeat_interleave"),
                    "Qwen shared-prefix scoring requires a repeatable Transformers cache")
            first_distribution = torch.nn.functional.log_softmax(prefix.logits[0, -1].float(), dim=-1)
            for offset in range(0, len(positions), batch_size):
                chunk = positions[offset:offset + batch_size]
                ids = torch.tensor([candidate_ids[index] for index in chunk], dtype=torch.long, device=device)
                candidate_cache = copy.deepcopy(prefix.past_key_values)
                candidate_cache.batch_repeat_interleave(len(chunk))
                output = model(
                    input_ids=ids[:, :2],
                    attention_mask=torch.ones((len(chunk), len(prompt) + 2), dtype=torch.long, device=device),
                    past_key_values=candidate_cache, use_cache=False, logits_to_keep=2,
                )
                next_distributions = torch.nn.functional.log_softmax(output.logits.float(), dim=-1)
                following = next_distributions.gather(2, ids[:, 1:, None]).squeeze(-1)
                values = (first_distribution[ids[:, 0]] + following.sum(dim=1)) / 3
                require(bool(torch.isfinite(values).all()), "Non-finite cached candidate likelihood")
                for index, value in zip(chunk, values.cpu().tolist()):
                    scores[index] = value
                del output, candidate_cache
    return scores


def read_examples(path, catalog):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"item_id", "item_sid", "history_item_id", "history_item_sid"}
        require(reader.fieldnames and required <= set(reader.fieldnames), f"Missing recommendation fields: {path}")
        rows = []
        for row_id, row in enumerate(reader):
            target_id = str(row["item_id"])
            history = ast.literal_eval(row["history_item_id"])
            require(isinstance(history, list) and bool(history), f"Invalid history at row {row_id}")
            history = list(map(str, history))
            require(target_id in catalog and all(item in catalog for item in history), "Unknown interaction item ID")
            sid = row["item_sid"].strip()
            require(sid == "".join(catalog[target_id]), "Target SID does not match item ID")
            require(ast.literal_eval(row["history_item_sid"]) == ["".join(catalog[item]) for item in history],
                    "History SIDs do not match item IDs")
            rows.append({"row_id": row_id, "target_id": target_id, "target": sid, "history": history})
    require(bool(rows), f"Empty split: {path}")
    return rows


class Metrics:
    def __init__(self, ks=(10, 20, 50, 200)):
        self.ks, self.rows = tuple(ks), 0
        self.hits = {k: 0 for k in ks}
        self.ndcg = {k: 0.0 for k in ks}

    def add(self, target, predictions):
        require(len(predictions) == len(set(predictions)), "Duplicate SID prediction")
        rank = predictions.index(target) + 1 if target in predictions else None
        for k in self.ks:
            if rank is not None and rank <= k:
                self.hits[k] += 1
                self.ndcg[k] += 1 / math.log2(rank + 1)
        self.rows += 1

    def result(self):
        result = {}
        for k in self.ks:
            recall = self.hits[k] / self.rows if self.rows else None
            result.update({f"Recall@{k}": recall, f"HR@{k}": recall,
                           f"NDCG@{k}": self.ndcg[k] / self.rows if self.rows else None})
        return result


def merge_and_measure(paths, examples, valid_sids, top_delta, output_path):
    """Require exactly one prediction for every split row before reporting any metrics."""
    by_id = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                index = row.get("row_id")
                require(type(index) is int and 0 <= index < len(examples) and index not in by_id,
                        "Duplicate or invalid evaluation row ID")
                require(row.get("target") == examples[index]["target"], "Prediction target differs")
                for key, length in (("predictions", min(50, top_delta)), ("retrieval_predictions", top_delta)):
                    predictions = row.get(key)
                    require(isinstance(predictions, list) and len(predictions) == length
                            and len(set(predictions)) == length and all(sid in valid_sids for sid in predictions),
                            f"Invalid {key} coverage or catalog SID")
                require(set(row["predictions"]) <= set(row["retrieval_predictions"]), "Reranker added an unretrieved candidate")
                by_id[index] = row
    require(len(by_id) == len(examples), "Evaluation split coverage is incomplete")
    groups = {name: {"retrieval": Metrics(tuple(sorted({10, 20, 50, top_delta}))), "rerank": Metrics((10, 20, 50))} for name in (
        "overall", "behavior_warm", "behavior_cold", "collision_free", "collision_affected",
    )}
    with Path(output_path).open("w", encoding="utf-8") as destination:
        for index in range(len(examples)):
            row = by_id[index]
            expected_groups = examples[index]["groups"]
            require(row.get("groups") == expected_groups, "Prediction stratum differs from source item IDs")
            for name in ("overall", *expected_groups):
                groups[name]["retrieval"].add(row["target"], row["retrieval_predictions"])
                groups[name]["rerank"].add(row["target"], row["predictions"])
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {name: {"rows": pair["retrieval"].rows, "retrieval_metrics": pair["retrieval"].result(),
                   "rerank_metrics": pair["rerank"].result()} for name, pair in groups.items()}


def evaluate(args):
    import torch
    import torch.distributed as dist
    from data import EvalSidDataset, SidSFTDataset

    require(args.batch_size > 0 and args.candidate_batch_size > 0 and args.top_delta >= 50,
            "Batches must be positive and top-delta must be at least 50")
    world, rank, local = (int(os.environ.get(name, default)) for name, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda", local) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if cuda else "gloo", timeout=timedelta(hours=6))
    torch.manual_seed(args.seed)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(args.output_dir / "metrics.json", {"status": "running", "split": args.split,
                    "phase": "loading", "created_at_utc": utc_now()})
    artifacts = verify_artifacts(args.model_path, args.input_dir, args.data_dir, CATEGORY, args.expected_sid_sha256)
    identity = [checkpoint_identity(args.model_path) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(identity, src=0)
    vectors, index_rows, manifest = load_index(args.index_dir, args.input_dir / f"{CATEGORY}.index.json", identity[0])
    require(args.top_delta <= len(index_rows), "top-delta exceeds the unique-SID catalog")
    examples = read_examples(args.data_dir / f"{args.split}.csv", artifacts["catalog"])
    behavior = set()
    for row in read_examples(args.data_dir / "train.csv", artifacts["catalog"]):
        behavior.update(row["history"])
        behavior.add(row["target_id"])
    counts = Counter("".join(tokens) for tokens in artifacts["catalog"].values())
    for row in examples:
        row["groups"] = ["behavior_warm" if row["target_id"] in behavior else "behavior_cold",
                         "collision_free" if counts[row["target"]] == 1 else "collision_affected"]
    tokenizer, model = load_joint_model(args.model_path, device)
    verify_sid_tokenizer(tokenizer, model, artifacts["catalog"])
    matrix = torch.from_numpy(vectors).to(device)
    candidate_tokens = [sid_token_ids(tokenizer, row["sid"]) for row in index_rows]
    # Match training's left truncation: reserve complete SID, newline and EOS.
    answer_length = len(tokenizer.encode(index_rows[0]["sid"] + "\n", add_special_tokens=False)) + 1
    prompt_budget = args.cutoff_len - answer_length
    require(prompt_budget >= 1, "cutoff-len leaves no room for the history prompt")

    class LocalTrainingPrompt(SidSFTDataset):
        def get_inputs(self):
            self.inputs = {index: self.pre(index) for index in range(rank, len(self.data), world)}

    class LocalOriginalEvalPrompt(EvalSidDataset):
        get_inputs = LocalTrainingPrompt.get_inputs

    dataset_kwargs = dict(train_file=str(args.data_dir / f"{args.split}.csv"), tokenizer=tokenizer,
                          max_len=args.cutoff_len, test=True, sample=-1, seed=args.seed,
                          category="industrial and scientific items")
    query_data, gr_data = LocalTrainingPrompt(**dataset_kwargs), LocalOriginalEvalPrompt(**dataset_kwargs)
    require(len(query_data) == len(gr_data) == len(examples), "Prompt/source row counts differ")
    local_ids = list(range(rank, len(examples), world))
    synchronize(torch, device)
    loading_seconds = time.monotonic() - started
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    if world > 1:
        dist.barrier()
    prediction_started = time.monotonic()
    phase_seconds = {"query_seconds": 0.0, "retrieval_seconds": 0.0, "rerank_seconds": 0.0}
    truncations = {"query": 0, "rerank": 0}
    shard = args.output_dir / f"predictions.rank{rank}.jsonl"
    with shard.open("w", encoding="utf-8", buffering=1) as stream, torch.inference_mode():
        for offset in range(0, len(local_ids), args.batch_size):
            ids = local_ids[offset:offset + args.batch_size]
            query_prompts, scoring_prompts = [], []
            for row_id in ids:
                query, gr = query_data[row_id]["input_ids"], gr_data[row_id]["input_ids"]
                truncations["query"] += int(len(query) > prompt_budget)
                truncations["rerank"] += int(len(gr) > prompt_budget)
                query_prompts.append(query[-prompt_budget:])
                scoring_prompts.append(gr[-prompt_budget:])
            batch = tokenizer.pad([{"input_ids": prompt, "attention_mask": [1] * len(prompt)}
                                   for prompt in query_prompts], padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
            synchronize(torch, device)
            tick = time.monotonic()
            queries = model.encode(**batch)
            synchronize(torch, device)
            phase_seconds["query_seconds"] += time.monotonic() - tick
            tick = time.monotonic()
            similarities = queries.float() @ matrix.T
            # Stable sort also makes tied normalized vectors deterministic across ranks.
            selected = torch.argsort(similarities, dim=-1, descending=True, stable=True)[:, :args.top_delta]
            selected_scores = similarities.gather(1, selected).cpu().tolist()
            selected_rows = selected.cpu().tolist()
            synchronize(torch, device)
            phase_seconds["retrieval_seconds"] += time.monotonic() - tick
            tick = time.monotonic()
            pair_prompts = [prompt for prompt in scoring_prompts for _ in range(args.top_delta)]
            pair_candidates = [candidate_tokens[index] for indices in selected_rows for index in indices]
            scores = score_candidates(model, tokenizer, pair_prompts, pair_candidates,
                                      batch_size=args.candidate_batch_size, device=device)
            synchronize(torch, device)
            phase_seconds["rerank_seconds"] += time.monotonic() - tick
            for position, row_id in enumerate(ids):
                indices = selected_rows[position]
                values = scores[position * args.top_delta:(position + 1) * args.top_delta]
                ordered = sorted(range(len(indices)), key=lambda index: (-values[index], indices[index]))
                top = ordered[:50]
                record = {"row_id": row_id, "target_id": examples[row_id]["target_id"],
                          "target": examples[row_id]["target"], "groups": examples[row_id]["groups"],
                          "retrieval_predictions": [index_rows[index]["sid"] for index in indices],
                          "retrieval_scores": selected_scores[position],
                          "predictions": [index_rows[indices[index]]["sid"] for index in top],
                          "rerank_scores": [values[index] for index in top]}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            atomic_json(args.output_dir / f"progress.rank{rank}.json", {
                "status": "predicting", "completed_rows": min(offset + len(ids), len(local_ids)),
                "expected_rows": len(local_ids), "updated_at_utc": utc_now(),
            })
    synchronize(torch, device)
    inference_seconds = time.monotonic() - prediction_started
    timing = {"loading_seconds": loading_seconds, **phase_seconds,
              "inference_seconds": inference_seconds, "prediction_seconds": inference_seconds,
              "total_seconds": time.monotonic() - started}
    names = list(timing)
    times = torch.tensor([timing[key] for key in names], dtype=torch.float64, device=device)
    truncated = torch.tensor(list(truncations.values()), dtype=torch.int64, device=device)
    memory = torch.tensor([torch.cuda.max_memory_allocated(device) if cuda else 0], dtype=torch.int64, device=device)
    memories = [torch.zeros_like(memory) for _ in range(world)]
    if world > 1:
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        dist.all_reduce(truncated)
        dist.all_gather(memories, memory)
        synchronize(torch, device)
        dist.barrier()
    else:
        memories = [memory]
    if rank == 0:
        groups = merge_and_measure([args.output_dir / f"predictions.rank{index}.jsonl" for index in range(world)],
                                   examples, artifacts["valid_sids"], args.top_delta, args.output_dir / "predictions.jsonl")
        timing = dict(zip(names, times.cpu().tolist()))
        timing["rows_per_second"] = len(examples) / timing["inference_seconds"]
        timing["mean_seconds_per_query"] = timing["inference_seconds"] / len(examples)
        overall = groups["overall"]
        report = {
            "status": "passed", "created_at_utc": utc_now(), "category": CATEGORY, "split": args.split,
            "model_path": str(args.model_path.resolve()), "checkpoint_sha256": identity[0]["sha256"],
            "index_manifest_sha256": hash_file(args.index_dir / "manifest.json"),
            "world_size": world, "expected_rows": len(examples), "completed_rows": len(examples),
            "coverage": {"expected_rows": len(examples), "completed_rows": len(examples), "missing_rows": 0, "duplicate_rows": 0},
            "metrics": {**overall["rerank_metrics"], f"Recall@{args.top_delta}": overall["retrieval_metrics"].get(f"Recall@{args.top_delta}")},
            "retrieval_metrics": overall["retrieval_metrics"], "rerank_metrics": overall["rerank_metrics"],
            "groups": groups, "catalog_items": manifest["catalog_items"], "unique_sids": manifest["unique_sids"],
            "validity_rate": 1.0, "unique_rate": 1.0, "training_performed": False,
            "metric_semantics": artifacts["metric_semantics"], "timing": timing,
            "peak_memory_allocated_bytes_per_rank": [int(value.item()) for value in memories],
            "prompt_truncations": dict(zip(truncations, truncated.cpu().tolist())),
            "sid_index_sha256": artifacts["sid_sha256"], "source_sha256": artifacts["source_sha256"],
            "protocol": {"query_prompt_style": "training", "rerank_prompt_style": "original-eval",
                         "top_delta": args.top_delta, "query_batch_size": args.batch_size,
                         "candidate_batch_size": args.candidate_batch_size, "cutoff_len": args.cutoff_len,
                         "score": "mean full-vocabulary log probability of exactly 3 SID tokens; EOS/newline excluded",
                         "history_filtering": False, "timing_scope": "Prediction loop including padding, query encode, matrix search, candidate scoring and output I/O; excludes loading/index export/validation"},
        }
        atomic_json(args.output_dir / "metrics.json", report)
        print(json.dumps({key: report[key] for key in ("status", "split", "completed_rows", "metrics", "timing")}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return evaluate(args)
    except Exception as error:
        report = {"status": "failed", "split": args.split, "rank": int(os.environ.get("RANK", "0")),
                  "error": str(error), "error_type": type(error).__name__, "created_at_utc": utc_now()}
        atomic_json(args.output_dir / f"failure.rank{report['rank']}.json", report)
        atomic_json(args.output_dir / "metrics.json", report)
        raise


if __name__ == "__main__":
    main()
