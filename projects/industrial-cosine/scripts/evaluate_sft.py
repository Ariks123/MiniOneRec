#!/usr/bin/env python3
"""Reload and validate an SFT checkpoint on complete validation and test splits.

Run with torchrun --standalone --nproc_per_node=4 scripts/evaluate_sft.py ... .
Only this entrypoint imports training frameworks; metric/integrity helpers are
separately testable on CPU without Torch. This script never performs training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sft_eval_utils import (
    CATEGORY_PROMPTS, EvaluationError, RankingAccumulator,
    atomic_json, merge_prediction_shards, require, resolve_expected_sid_sha256,
    verify_artifacts, verify_sid_tokenizer,
)

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--category", choices=("Industrial_and_Scientific",), default="Industrial_and_Scientific")
    result.add_argument("--prompt-style", choices=("original-eval",), default="original-eval",
                        help="Test recommendation prompt; validation loss always uses the training prompt")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--num-beams", type=int, default=50)
    result.add_argument("--cutoff-len", type=int, default=512)
    result.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    result.add_argument("--device", choices=("auto", "cpu"), default="auto")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--expected-sid-sha256", default="53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8",
                        help="Pinned original Industrial SID hash")
    return result


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _partial_row_count(output_dir):
    seen = set()
    for path in Path(output_dir).glob("predictions.rank*.jsonl"):
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    seen.add(json.loads(line)["row_id"])
                except (ValueError, KeyError, TypeError):
                    continue
    return len(seen)


def evaluate(args):
    args.expected_sid_sha256 = resolve_expected_sid_sha256(args.category, args.expected_sid_sha256)
    prompt_category = CATEGORY_PROMPTS[args.category]
    prompt_style = getattr(args, "prompt_style", "training")
    require(prompt_style in ("training", "original-eval"), "Unsupported test prompt style")
    require(args.batch_size > 0 and args.num_beams > 0 and args.cutoff_len > 0,
            "Batch size, num_beams and cutoff_len must be positive")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, LogitsProcessorList
    from data import EvalSidDataset, SidSFTDataset
    from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = args.device != "cpu" and torch.cuda.is_available()
    require(world_size >= 1 and 0 <= rank < world_size, "Invalid distributed rank configuration")
    if cuda:
        require(0 <= local_rank < torch.cuda.device_count(), "LOCAL_RANK is outside visible CUDA devices")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    torch.manual_seed(args.seed)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if cuda else "gloo", timeout=timedelta(hours=6))
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(output_dir / "metrics.json", {
            "status": "running", "phase": "checking_checkpoint", "updated_at_utc": _utc_now(),
            "expected_rows": None, "completed_rows": 0,
        })
    if world_size > 1:
        # Publish this attempt's initial state before another rank can fail its
        # input checks; an old failed report must not describe a new attempt.
        dist.barrier(device_ids=[local_rank] if cuda else None)
    rank_progress = output_dir / f"progress.rank{rank}.json"
    atomic_json(rank_progress, {"status": "checking_checkpoint", "rank": rank, "updated_at_utc": _utc_now()})
    artifacts = verify_artifacts(args.model_path, args.input_dir, args.data_dir,
                                 args.category, args.expected_sid_sha256)
    targets = artifacts["test_targets"]
    if rank == 0:
        atomic_json(output_dir / "metrics.json", {
            "status": "running", "phase": "checkpoint_reload", "updated_at_utc": _utc_now(),
            "expected_rows": len(targets), "completed_rows": 0,
        })
    require(args.num_beams <= len(artifacts["valid_sids"]), "num_beams exceeds the complete catalog size")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    require(tokenizer.eos_token_id is not None, "Checkpoint tokenizer has no EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs = {"torch_dtype": torch.bfloat16 if cuda else torch.float32,
                   "attn_implementation": "sdpa", "local_files_only": True}
    if cuda:
        load_kwargs["device_map"] = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), **load_kwargs)
    if not cuda:
        model.to(device)
    model.eval()
    dimensions = verify_sid_tokenizer(tokenizer, model, artifacts["catalog"])
    model.config.pad_token_id = tokenizer.pad_token_id
    prefixes = build_prefix_allowed_tokens(tokenizer, (sid + "\n" for sid in sorted(artifacts["valid_sids"])))
    max_new_tokens = max(map(len, prefixes)) + 1
    require(max_new_tokens < args.cutoff_len, "SID answer length leaves no room for the training prompt")

    class ShardedSidSFTDataset(SidSFTDataset):
        def get_inputs(self):
            # Use the exact training implementation; tokenize only this rank's rows.
            self.inputs = {index: self.pre(index) for index in range(rank, len(self.data), world_size)}

    class ShardedOriginalEvalDataset(EvalSidDataset):
        get_inputs = ShardedSidSFTDataset.get_inputs

    validation_started = time.monotonic()
    atomic_json(rank_progress, {"status": "validation_loss", "rank": rank, "updated_at_utc": _utc_now()})
    valid_data = ShardedSidSFTDataset(
        train_file=str(args.data_dir / "valid.csv"), tokenizer=tokenizer, max_len=args.cutoff_len,
        sample=-1, test=False, seed=args.seed, category=prompt_category,
    )
    valid_ids = list(range(rank, len(valid_data), world_size))
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for offset in range(0, len(valid_ids), args.batch_size):
            batch = collator([valid_data[index] for index in valid_ids[offset:offset + args.batch_size]])
            batch = {key: value.to(device) for key, value in batch.items()}
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            require(token_count > 0, "A validation batch has no supervised answer tokens")
            prediction = model(**batch, use_cache=False)
            loss = float(prediction.loss.float().item())
            require(math.isfinite(loss), "Non-finite validation loss")
            total_loss += loss * token_count
            total_tokens += token_count
            del prediction, batch
    sums = torch.tensor([total_loss, float(total_tokens), float(len(valid_ids))], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(sums)
    require(int(sums[2].item()) == len(artifacts["valid_targets"]), "Validation split coverage is incomplete")
    valid_loss = float(sums[0].item() / sums[1].item())
    masked_ppl = math.exp(valid_loss) if valid_loss < 709 else None
    validation_result = {"rows": int(sums[2].item()), "answer_tokens": int(sums[1].item()),
                         "loss": valid_loss, "masked_perplexity": masked_ppl,
                         "perplexity_scope": "Supervised answer tokens only; prompt and padding labels are masked"}
    validation_seconds = time.monotonic() - validation_started
    del valid_data, collator

    test_dataset_type = ShardedOriginalEvalDataset if prompt_style == "original-eval" else ShardedSidSFTDataset
    test_data = test_dataset_type(
        train_file=str(args.data_dir / "test.csv"), tokenizer=tokenizer, max_len=args.cutoff_len,
        sample=-1, test=True, seed=args.seed, category=prompt_category,
    )
    require(len(test_data) == len(targets), "Test dataset row count differs from prepared test data")
    local_ids = list(range(rank, len(test_data), world_size))
    shard_path = output_dir / f"predictions.rank{rank}.jsonl"
    generation_started = time.monotonic()
    completed = 0
    prompt_truncations = 0
    accumulator = RankingAccumulator(artifacts["valid_sids"], args.num_beams)
    with shard_path.open("w", encoding="utf-8", buffering=1) as destination, torch.inference_mode():
        for offset in range(0, len(local_ids), args.batch_size):
            row_ids = local_ids[offset:offset + args.batch_size]
            prompt_budget = args.cutoff_len - max_new_tokens
            sequences = []
            for row_id in row_ids:
                ids = test_data[row_id]["input_ids"]
                prompt_truncations += int(len(ids) > prompt_budget)
                ids = ids[-prompt_budget:]
                sequences.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            batch = tokenizer.pad(sequences, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
            prompt_width = batch["input_ids"].shape[1]
            processor = ConstrainedLogitsProcessor(
                lambda batch_id, completion: prefixes.get(tuple(completion), []),
                num_beams=args.num_beams, eos_token_id=tokenizer.eos_token_id, prompt_length=prompt_width,
            )
            generated = model.generate(
                **batch, num_beams=args.num_beams, num_return_sequences=args.num_beams,
                do_sample=False, use_cache=True, max_new_tokens=max_new_tokens, length_penalty=0.0,
                top_k=None, top_p=None, output_scores=False, return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([processor]),
            )
            decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            for index, row_id in enumerate(row_ids):
                predictions = [item.strip() for item in decoded[index * args.num_beams:(index + 1) * args.num_beams]]
                record = {"row_id": row_id, "target": targets[row_id], "predictions": predictions,
                          "prompt_style": prompt_style}
                accumulator.add(record)
                destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                completed += 1
            atomic_json(rank_progress, {"status": "predicting", "rank": rank, "completed_rows": completed,
                                       "expected_rows": len(local_ids), "updated_at_utc": _utc_now(),
                                       "elapsed_seconds": time.monotonic() - generation_started})
            del generated, batch
    atomic_json(rank_progress, {"status": "completed", "rank": rank, "completed_rows": completed,
                               "expected_rows": len(local_ids), "updated_at_utc": _utc_now()})
    times = torch.tensor([validation_seconds, time.monotonic() - generation_started,
                          time.monotonic() - started], dtype=torch.float64, device=device)
    truncated = torch.tensor([prompt_truncations], dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        dist.all_reduce(truncated)
        # NCCL all_reduce queues CUDA work without waiting for remote CPU file
        # writes. Every rank must close its shard before rank zero reads any of
        # them, so use an explicit blocking synchronization boundary here.
        if cuda:
            torch.cuda.synchronize(device)
        dist.barrier(device_ids=[local_rank] if cuda else None)
    if rank == 0:
        rankings = merge_prediction_shards(
            [output_dir / f"predictions.rank{index}.jsonl" for index in range(world_size)],
            targets, artifacts["valid_sids"], args.num_beams, output_dir / "predictions.jsonl",
        )
        check_results = {name: "passed" for name in (
            "provenance", "prepared_input_hashes", "checkpoint_reload", "tokenizer_atomic_sid_tokens",
            "model_vocabulary", "validation_coverage", "prediction_coverage", "valid_sids", "unique_candidates")}
        report = {"status": "passed", "created_at_utc": _utc_now(), "model_path": str(args.model_path.resolve()),
                  "category": args.category, "world_size": world_size, "num_beams": args.num_beams,
                  "prompt_style": prompt_style, "validation_prompt_style": "training",
                  "scope": "complete_original_test_split", "seed": args.seed, "training_performed": False,
                  "max_new_tokens": max_new_tokens, "cutoff_len": args.cutoff_len,
                  "prompt_truncations": int(truncated.item()), **rankings, "validation": validation_result,
                  "eval_loss_answer_tokens": valid_loss, "masked_ppl": masked_ppl,
                  "check_results": check_results, "tokenizer_and_model": dimensions,
                  "sid_index_sha256": artifacts["sid_sha256"], "item_meta_sha256": artifacts["metadata_sha256"],
                  "catalog_items": artifacts["catalog_items"], "unique_sids": artifacts["unique_sids"],
                  "sid_tokens_per_item": artifacts["sid_tokens_per_item"],
                  "collision_groups": artifacts["collision_groups"], "collision_items": artifacts["collision_items"],
                  "metric_semantics": artifacts["metric_semantics"],
                  "source_sha256": artifacts["source_sha256"],
                  "provenance": artifacts["provenance"],
                  "timing": {"validation_seconds": times[0].item(), "prediction_seconds": times[1].item(),
                             "total_seconds": times[2].item()},
                  "interpretation": "SID validity is enforced by catalog constraints; HR/NDCG measure ranking quality."}
        atomic_json(output_dir / "metrics.json", report)
        print(json.dumps({key: report[key] for key in
                          ("status", "expected_rows", "completed_rows", "metrics", "validation", "CC")},
                         ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return evaluate(args)
    except Exception as error:
        output_dir = args.output_dir.resolve()
        completed = _partial_row_count(output_dir) if output_dir.is_dir() else 0
        try:
            import csv
            with (args.data_dir / "test.csv").open(encoding="utf-8", newline="") as source:
                expected = max(0, sum(1 for _ in csv.reader(source)) - 1)
        except (OSError, UnicodeError, csv.Error):
            expected = None
        report = {"status": "failed", "error": str(error), "error_type": type(error).__name__,
                  "rank": int(os.environ.get("RANK", "0")), "updated_at_utc": _utc_now(),
                  "expected_rows": expected, "completed_rows": completed,
                  "coverage": {"expected_rows": expected, "completed_rows": completed,
                               "missing_rows": None if expected is None else max(0, expected - completed)},
                  "partial_predictions_preserved": True}
        atomic_json(output_dir / f"failure.rank{report['rank']}.json", report)
        atomic_json(output_dir / "metrics.json", report)
        print(f"SFT result validation failed on rank {report['rank']}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
