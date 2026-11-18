"""Full Industrial validation ranking during Trainer evaluation.

The original SID catalog is used with its original collisions. Generation uses
the original evaluation prompt and the same beam-50 protocol as final testing.
This module reads valid.csv only and leaves checkpoint selection to Trainer.
Training frameworks are imported only when generation is actually requested.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import random
import re
import time
import uuid

from scripts.sft_eval_utils import RankingAccumulator, TOP_K, atomic_json, sha256_file


SCOPE = "complete_original_validation_split"
CATEGORY = "Industrial_and_Scientific"
APPROVED_SID_SHA256 = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class GenerativeValidationConfig:
    valid_file: str
    sid_index_path: str
    expected_sid_sha256: str | None = None
    expected_valid_sha256: str | None = None
    batch_size: int = 4
    cutoff_len: int = 512
    num_beams: int = 50
    category: str = CATEGORY
    expected_rows: int | None = 4532
    write_predictions: bool = True

    def __post_init__(self):
        self.valid_file = str(Path(self.valid_file).resolve())
        self.sid_index_path = str(Path(self.sid_index_path).resolve())
        require(Path(self.valid_file).name == "valid.csv", "Ranking selection requires valid.csv, never a test split")
        require(self.category == CATEGORY, "Ranking selection requires the Industrial category")
        require(self.num_beams == 50, "Industrial ranking selection requires beam50")
        require(type(self.batch_size) is int and self.batch_size > 0
                and type(self.cutoff_len) is int and self.cutoff_len > 0,
                "Generation batch size and cutoff must be positive integers")
        require(self.expected_rows is None or type(self.expected_rows) is int and self.expected_rows > 0,
                "Expected validation rows must be a positive integer")
        require(type(self.write_predictions) is bool, "write_predictions must be boolean")


def rank_row_ids(n, rank, world_size):
    require(type(n) is int and n > 0 and type(world_size) is int and world_size > 0
            and type(rank) is int and 0 <= rank < world_size, "Invalid validation sharding parameters")
    return list(range(rank, n, world_size))


def validate_coverage(shards, n):
    ids = [row for shard in shards for row in shard]
    require(all(type(row) is int for row in ids) and sorted(ids) == list(range(n)),
            "Validation ranking coverage is missing, duplicated, or out of range")
    return {"expected_rows": n, "completed_rows": len(ids), "missing_rows": 0, "duplicate_rows": 0}


def validation_artifacts(config):
    valid, sid = Path(config.valid_file), Path(config.sid_index_path)
    manifest_path = valid.parent / "preparation_manifest.json"
    sources = {"valid_file": valid, "sid_index_path": sid, "preparation_manifest": manifest_path}
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("validation") == "passed" and manifest.get("dataset") == CATEGORY
            and manifest.get("original_sid") is True, "Ranking selection requires passed original Industrial preparation")
    require(hashes["valid_file"] == manifest["outputs"]["valid.csv"]["sha256"],
            "Validation CSV differs from preparation")
    expected_sid = config.expected_sid_sha256 or manifest.get("sources", {}).get("sid_index", {}).get("sha256")
    require(expected_sid and hashes["sid_index_path"] == expected_sid,
            "Ranking SID catalog differs from its pinned hash")
    if config.expected_valid_sha256:
        require(hashes["valid_file"] == config.expected_valid_sha256, "Validation CSV differs from its pinned hash")
    catalog = json.loads(sid.read_text(encoding="utf-8"))
    require(isinstance(catalog, dict) and catalog, "Empty SID catalog")
    require(all(isinstance(tokens, list) and len(tokens) in (3, 4)
                and all(isinstance(token, str) and re.fullmatch(rf"<{layer}_[0-9]+>", token)
                        for layer, token in zip("abcd", tokens)) for tokens in catalog.values()),
            "Industrial catalog contains malformed SID tokens")
    valid_sids = {"".join(tokens) for tokens in catalog.values()}
    require(len(valid_sids) >= config.num_beams, "Catalog is smaller than the requested beam count")
    # Original Industrial data has SID collisions; ranking is deliberately at
    # SID level, just as in scripts/evaluate_sft.py and the historical reports.
    with valid.open(encoding="utf-8", newline="") as stream:
        targets = [row["item_sid"] for row in csv.DictReader(stream)]
    require(targets and len(targets) == manifest["rows"]["valid"]
            and all(target in valid_sids for target in targets),
            "Validation rows or targets differ from the prepared catalog")
    require(config.expected_rows is None or len(targets) == config.expected_rows,
            "Validation row count differs from the requested protocol")
    require(all(sha256_file(path) == hashes[key] for key, path in sources.items()),
            "Validation source changed while loading")
    return {"catalog": catalog, "valid_sids": valid_sids, "targets": targets,
            "source_sha256": hashes, "source_paths": {key: str(path) for key, path in sources.items()},
            "catalog_items": len(catalog), "unique_sids": len(valid_sids)}


@contextmanager
def preserve_rng_state(device):
    """Evaluation must not change the random stream used by subsequent training."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generate_validation(trainer):
    with preserve_rng_state(trainer.accelerator.device):
        return _generate_validation(trainer)


def _generate_validation(trainer):
    import torch
    import torch.distributed as dist
    from transformers import LogitsProcessorList
    from data import EvalSidDataset
    from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
    from scripts.sft_eval_utils import verify_sid_tokenizer

    config, artifacts = trainer.generative_validation, trainer._validation_artifacts()
    require(all(sha256_file(path) == artifacts["source_sha256"][key]
                for key, path in artifacts["source_paths"].items()), "Validation sources changed after pinning")
    accelerator = trainer.accelerator
    rank, world = accelerator.process_index, accelerator.num_processes
    require(not getattr(trainer, "is_fsdp_enabled", False), "Ranking validation supports replicated DDP/ZeRO-2, not FSDP")
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    require(plugin is None or int(plugin.zero_stage) <= 2, "Ranking validation does not support ZeRO-3")
    model = accelerator.unwrap_model(trainer.model_wrapped)
    tokenizer = trainer.processing_class
    require(tokenizer is not None and tokenizer.eos_token_id is not None,
            "Ranking validation requires the generation tokenizer and EOS")
    if trainer._generation_data is None:
        verify_sid_tokenizer(tokenizer, model, artifacts["catalog"])
        prefixes = build_prefix_allowed_tokens(tokenizer, (sid + "\n" for sid in sorted(artifacts["valid_sids"])))
        answer_tokens = max(map(len, prefixes)) + 1
        require(config.cutoff_len > answer_tokens, "Cutoff leaves no room for an original evaluation prompt")

        class ShardedOriginalEvalDataset(EvalSidDataset):
            def get_inputs(self):
                self.inputs = {index: self.pre(index) for index in range(rank, len(self.data), world)}

        dataset = ShardedOriginalEvalDataset(
            train_file=config.valid_file, tokenizer=tokenizer, max_len=config.cutoff_len,
            sample=-1, test=True, seed=None, category="industrial and scientific items")
        require(len(dataset) == len(artifacts["targets"]), "Original validation prompt coverage differs")
        trainer._generation_data = dataset, prefixes, answer_tokens
    dataset, prefixes, answer_tokens = trainer._generation_data
    local_ids = rank_row_ids(len(artifacts["targets"]), rank, world)
    identifier = [f"step-{trainer.state.global_step:09d}-{uuid.uuid4().hex[:12]}" if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(identifier, src=0, device=accelerator.device)
    directory = Path(trainer.args.output_dir) / "validation_ranking" / identifier[0]
    directory.mkdir(parents=True, exist_ok=True)
    stats, completed, truncations = RankingAccumulator(artifacts["valid_sids"], config.num_beams), [], 0
    stream = (directory / f"predictions.rank{rank}.jsonl").open("w", encoding="utf-8") if config.write_predictions else None
    started = time.monotonic()
    prior_training, prior_padding = model.training, tokenizer.padding_side
    prior_cache = getattr(model.config, "use_cache", None)
    local_error = None

    def write_progress(status, error=None):
        atomic_json(Path(trainer.args.output_dir) / f"validation_ranking/progress.rank{rank}.json", {
            "status": status, "rank": rank, "step": int(trainer.state.global_step),
            "completed_rows": len(completed), "expected_rows": len(local_ids),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started, "error": error})

    model.eval()
    tokenizer.padding_side = "left"
    try:
        write_progress("predicting")
        with torch.inference_mode(), accelerator.autocast():
            for offset in range(0, len(local_ids), config.batch_size):
                row_ids = local_ids[offset:offset + config.batch_size]
                sequences, budget = [], config.cutoff_len - answer_tokens
                for row_id in row_ids:
                    ids = dataset[row_id]["input_ids"]
                    truncations += int(len(ids) > budget)
                    ids = ids[-budget:]
                    sequences.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                batch = tokenizer.pad(sequences, padding=True, return_tensors="pt")
                batch = {key: value.to(accelerator.device) for key, value in batch.items()
                         if key in {"input_ids", "attention_mask"}}
                width = batch["input_ids"].shape[1]
                processor = ConstrainedLogitsProcessor(
                    lambda batch_id, completion: prefixes.get(tuple(completion), []),
                    num_beams=config.num_beams, eos_token_id=tokenizer.eos_token_id, prompt_length=width)
                # Explicit kwargs override training-time generation defaults,
                # including model.generation_config.use_cache=False. This is
                # the same beam protocol used by the standalone evaluator.
                output = model.generate(
                    **batch, num_beams=config.num_beams, num_return_sequences=config.num_beams,
                    do_sample=False, use_cache=True, max_new_tokens=answer_tokens,
                    length_penalty=0.0, top_k=None, top_p=None, output_scores=False,
                    return_dict_in_generate=False, synced_gpus=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([processor]))
                require(output.shape[0] == len(row_ids) * config.num_beams, "Incomplete generated beam batch")
                decoded = tokenizer.batch_decode(output[:, width:], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                for index, row_id in enumerate(row_ids):
                    predictions = [value.strip() for value in decoded[index * config.num_beams:(index + 1) * config.num_beams]]
                    record = {"row_id": row_id, "target": artifacts["targets"][row_id],
                              "predictions": predictions, "prompt_style": "original-eval"}
                    stats.add(record)
                    completed.append(row_id)
                    if stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                write_progress("predicting")
                del output, batch
        write_progress("completed")
    except Exception as error:
        local_error = f"rank {rank}: {type(error).__name__}: {error}"
        try:
            write_progress("failed", local_error)
        except OSError:
            pass
    finally:
        tokenizer.padding_side = prior_padding
        model.config.use_cache = prior_cache
        model.train(prior_training)
        if stream:
            stream.close()
    errors = [local_error]
    if world > 1:
        errors = [None] * world
        dist.all_gather_object(errors, local_error)
    if any(errors):
        if rank == 0:
            atomic_json(directory / "metrics.json", {"status": "failed", "scope": SCOPE, "errors": errors})
        raise ValueError("Validation generation failed: " + "; ".join(error for error in errors if error))
    values = torch.tensor([stats.rows, *(stats.hits[k] for k in TOP_K), *(stats.ndcg[k] for k in TOP_K), truncations],
                          dtype=torch.float64, device=accelerator.device)
    shards = [completed]
    if world > 1:
        dist.all_reduce(values)
        shards = [None] * world
        dist.all_gather_object(shards, completed)
    coverage = validate_coverage(shards, len(artifacts["targets"]))
    reduced = values.tolist()
    rows = reduced[0]
    require(int(rows) == len(artifacts["targets"]), "Reduced validation row count differs")
    require(all(sha256_file(path) == artifacts["source_sha256"][key]
                for key, path in artifacts["source_paths"].items()), "Validation sources changed during generation")
    metrics = {f"recall{k}": reduced[1 + index] / rows for index, k in enumerate(TOP_K)}
    metrics.update({f"ndcg{k}": reduced[1 + len(TOP_K) + index] / rows for index, k in enumerate(TOP_K)})
    report = {"status": "passed", "scope": SCOPE, "category": CATEGORY,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "step": int(trainer.state.global_step), "epoch": trainer.state.epoch,
              "coverage": coverage, "metrics": metrics, "CC": 0, "validity_rate": 1.0, "unique_rate": 1.0,
              "prompt_truncations": int(reduced[-1]), "source_sha256": artifacts["source_sha256"],
              "catalog_items": artifacts["catalog_items"], "unique_sids": artifacts["unique_sids"],
              "prediction_directory": str(directory), "world_size": world,
              "generation": {"num_beams": config.num_beams, "num_return_sequences": config.num_beams,
                             "max_new_tokens": answer_tokens, "cutoff_len": config.cutoff_len,
                             "length_penalty": 0.0, "early_stopping": False, "full_catalog": True,
                             "seen_item_filter": False, "prompt_style": "original-eval"},
              "elapsed_seconds": time.monotonic() - started}
    if rank == 0:
        atomic_json(directory / "metrics.json", report)
    return report


def make_generative_trainer_class(base_trainer):
    class IndustrialGenerativeValidationTrainer(base_trainer):
        def __init__(self, *args, generative_validation, **kwargs):
            self.generative_validation = generative_validation
            self._generation_artifacts = self._generation_data = self._generation_result = None
            super().__init__(*args, **kwargs)

        def _validation_artifacts(self):
            if self._generation_artifacts is None:
                self._generation_artifacts = validation_artifacts(self.generative_validation)
            return self._generation_artifacts

        def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
            output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
            result = generate_validation(self)
            require(output.num_samples == result["coverage"]["expected_rows"],
                    "Teacher-forced validation and generated ranking cover different rows")
            loss_key = metric_key_prefix + "_loss"
            loss = float(output.metrics[loss_key])
            require(math.isfinite(loss), "Teacher-forced validation loss is not finite")
            result["metrics"]["eval_loss"] = loss
            self._generation_result = result
            output.metrics.update({metric_key_prefix + "_" + key: value
                                   for key, value in result["metrics"].items() if key != "eval_loss"})
            output.metrics[metric_key_prefix + "_generation_rows"] = result["coverage"]["completed_rows"]
            if self.is_world_process_zero():
                atomic_json(Path(result["prediction_directory"]) / "metrics.json", result)
                atomic_json(Path(self.args.output_dir) / "validation_ranking/latest_metrics.json", result)
            return output

    return IndustrialGenerativeValidationTrainer
