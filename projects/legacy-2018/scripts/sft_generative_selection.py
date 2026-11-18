"""Full-validation constrained generation and lexicographic Trainer selection.

Use make_generative_trainer_class(Trainer), pass generative_validation=Config,
and configure matching epoch/steps eval/save strategies, load_best_model_at_end,
metric_for_best_model='selection_index', greater_is_better=True. Do not install
EarlyStoppingCallback: this module selects checkpoints lexicographically but
resets patience only for primary NDCG improvements. Framework imports are lazy.

Generation unwraps replicated DDP/ZeRO-2 parameters. ZeRO-3/FSDP are unsupported.
No code in this module opens a test split.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from scripts.sft_eval_utils import atomic_json, sha256_file


SCOPE = "complete_original_validation_split"
CATEGORY_PROMPTS = {'Video_Games': 'video games',
                    'Industrial_and_Scientific': 'industrial and scientific items'}


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
    cutoff_len: int = 2048
    num_beams: int = 10
    tolerance: float = 1e-6
    patience: int = 3
    write_predictions: bool = True
    category: str = 'Video_Games'
    expected_rows: int | None = None
    expected_manifest_sha256: str | None = None

    def __post_init__(self):
        self.valid_file = str(Path(self.valid_file).resolve())
        self.sid_index_path = str(Path(self.sid_index_path).resolve())
        require(self.category in CATEGORY_PROMPTS, 'Unsupported Amazon23 selection category')
        require(self.expected_rows is None or type(self.expected_rows) is int and self.expected_rows > 0,
                'Expected validation rows must be a positive integer')
        require(Path(self.valid_file).name == "valid.csv", "Selection requires the prepared valid.csv, never test.csv")
        require(self.num_beams == 10, "Selection protocol requires exactly beam10")
        require(type(self.batch_size) is int and self.batch_size > 0 and type(self.cutoff_len) is int and self.cutoff_len > 0,
                "Generation batch size/cutoff must be positive integers")
        SelectionPolicy(self.tolerance, self.patience)


class SelectionPolicy:
    """Tolerance applies at each lexicographic tier; no weighted metric sum."""
    def __init__(self, tolerance=1e-6, patience=3):
        require(math.isfinite(tolerance) and tolerance >= 0, "Selection tolerance must be finite and nonnegative")
        require(type(patience) is int and patience > 0, "Selection patience must be a positive integer")
        self.tolerance, self.patience = tolerance, patience
        self.best_metrics, self.best_step = None, None
        self.best_primary_ndcg, self.primary_improved = None, False
        self.patience_counter = self.evaluations = self.improvements = 0
        self.last_step = None

    @staticmethod
    def checked(metrics):
        values = {key: float(metrics[key]) for key in ("ndcg10", "recall10", "eval_loss")}
        require(all(math.isfinite(value) for value in values.values()), "Selection metrics must be finite")
        require(0 <= values["ndcg10"] <= 1 and 0 <= values["recall10"] <= 1 and values["eval_loss"] >= 0,
                "Selection metrics are outside their valid ranges")
        return values

    def would_improve(self, metrics):
        values = self.checked(metrics)
        if self.best_metrics is None:
            return True
        for key, direction in (("ndcg10", 1), ("recall10", 1), ("eval_loss", -1)):
            difference = direction * (values[key] - self.best_metrics[key])
            if difference > self.tolerance:
                return True
            if difference < -self.tolerance:
                return False
        return False

    def consider(self, metrics, step):
        require(type(step) is int and step >= 0 and (self.last_step is None or step > self.last_step),
                "Selection steps must be unique and increasing")
        values = self.checked(metrics)
        improved = self.would_improve(values)
        self.primary_improved = (self.best_primary_ndcg is None
                                 or values["ndcg10"] > self.best_primary_ndcg + self.tolerance)
        self.evaluations += 1
        self.last_step = step
        if improved:
            self.best_metrics, self.best_step = values, step
            self.improvements += 1
        if self.primary_improved:
            self.best_primary_ndcg = values["ndcg10"]
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return improved

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value):
        result = cls(value["tolerance"], value["patience"])
        if value["best_metrics"] is not None:
            result.checked(value["best_metrics"])
        result.__dict__.update(value)
        return result


def rank_row(target, predictions, valid_sids):
    require(target in valid_sids, "Validation target is outside the catalog")
    seen, invalid, duplicates, rank = set(), 0, 0, None
    for position, candidate in enumerate(predictions, 1):
        if not isinstance(candidate, str) or candidate not in valid_sids:
            invalid += 1
            continue
        if candidate in seen:
            duplicates += 1
            continue
        seen.add(candidate)
        if candidate == target:
            rank = position
    return {"rank": rank, "invalid_candidates": invalid, "duplicate_candidates": duplicates}


class RankingStats:
    def __init__(self, valid_sids, num_beams=10):
        self.valid_sids, self.num_beams = set(valid_sids), num_beams
        self.rows = self.hits = self.invalid_candidates = self.duplicate_candidates = self.invalid_rows = 0
        self.ndcg_sum = 0.0

    def add(self, target, predictions):
        require(isinstance(predictions, list) and len(predictions) == self.num_beams, "Generation returned an incomplete beam group")
        result = rank_row(target, predictions, self.valid_sids)
        self.rows += 1
        self.invalid_candidates += result["invalid_candidates"]
        self.duplicate_candidates += result["duplicate_candidates"]
        self.invalid_rows += int(bool(result["invalid_candidates"] or result["duplicate_candidates"]))
        if result["rank"] is not None and result["rank"] <= 10:
            self.hits += 1
            self.ndcg_sum += 1 / math.log2(result["rank"] + 1)

    def metrics(self):
        require(self.rows > 0, "No validation generation rows")
        return {"ndcg10": self.ndcg_sum / self.rows, "recall10": self.hits / self.rows,
                "validity_rate": 1 - self.invalid_candidates / (self.rows * self.num_beams),
                "unique_rate": 1 - self.duplicate_candidates / (self.rows * self.num_beams)}


def rank_row_ids(n, rank, world_size):
    require(type(n) is int and n > 0 and type(world_size) is int and world_size > 0
            and type(rank) is int and 0 <= rank < world_size, "Invalid validation sharding parameters")
    return list(range(rank, n, world_size))


def validate_coverage(shards, n):
    ids = [row for shard in shards for row in shard]
    require(all(type(row) is int for row in ids) and sorted(ids) == list(range(n)),
            "Validation generation coverage is missing, duplicated, or out of range")
    return {"expected_rows": n, "completed_rows": len(ids), "missing_rows": 0, "duplicate_rows": 0}


def validation_artifacts(config):
    valid, sid = Path(config.valid_file).resolve(), Path(config.sid_index_path).resolve()
    manifest_path = valid.parent / "preparation_manifest.json"
    sources = {"valid_file": valid, "sid_index_path": sid, "preparation_manifest": manifest_path}
    before = {key: sha256_file(path) for key, path in sources.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("validation") == "passed", "Selection requires passed sample preparation")
    if config.expected_manifest_sha256:
        require(before['preparation_manifest'] == config.expected_manifest_sha256, 'Pinned preparation manifest SHA256 differs')
    declared_category = manifest.get('category', manifest.get('dataset_name'))
    if declared_category is None and manifest.get('dataset', '').startswith('Amazon Reviews 2023 '):
        declared_category = manifest['dataset'].removeprefix('Amazon Reviews 2023 ')
    require(declared_category is None or declared_category == config.category, 'Selection category differs from the prepared manifest')
    require(before["valid_file"] == manifest["outputs"]["valid.csv"]["sha256"], "Validation CSV SHA256 differs from preparation")
    expected_sid = config.expected_sid_sha256 or manifest.get("source_sha256", {}).get(f'{config.category}.index.json')
    require(expected_sid and before["sid_index_path"] == expected_sid, "Selection SID catalog SHA256 differs")
    if config.expected_valid_sha256:
        require(before["valid_file"] == config.expected_valid_sha256, "Pinned validation SHA256 differs")
    catalog = json.loads(sid.read_text(encoding="utf-8"))
    require(isinstance(catalog, dict) and catalog, "Empty SID catalog")
    sids = ["".join(tokens) for tokens in catalog.values()]
    require(all(isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens)
                for tokens in catalog.values()) and len(set(sids)) == len(catalog), "Catalog SIDs must be complete and unique")
    require(len(sids) >= config.num_beams, "Catalog is smaller than beam10")
    valid_sids = set(sids)
    with valid.open(encoding="utf-8", newline="") as stream:
        targets = [row["item_sid"] for row in csv.DictReader(stream)]
    require(targets and len(targets) == manifest["rows"]["valid"] and all(target in valid_sids for target in targets),
            "Original validation rows/targets differ from prepared catalog")
    require(config.expected_rows is None or len(targets) == config.expected_rows, 'Pinned validation row count differs')
    require(all(sha256_file(path) == before[key] for key, path in sources.items()), "Validation sources changed during loading")
    return {"catalog": catalog, "valid_sids": valid_sids, "targets": targets, "source_sha256": before,
            'category': config.category, 'native_category': CATEGORY_PROMPTS[config.category],
            "source_paths": {key: str(path) for key, path in sources.items()}}


def generate_validation(trainer):
    import torch
    import torch.distributed as dist
    from transformers import GenerationConfig, LogitsProcessorList
    from data import SidSFTDataset
    from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
    from scripts.sft_eval_utils import verify_sid_tokenizer

    config, artifacts = trainer.generative_validation, trainer._validation_artifacts()
    require(all(sha256_file(path) == artifacts["source_sha256"][key] for key, path in artifacts["source_paths"].items()),
            "Validation source changed after its identity was pinned")
    accelerator = trainer.accelerator
    rank, world = accelerator.process_index, accelerator.num_processes
    require(not getattr(trainer, "is_fsdp_enabled", False), "Generative selection supports DDP/ZeRO-2, not FSDP")
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    require(plugin is None or int(plugin.zero_stage) <= 2, "Generative selection requires replicated parameters, not ZeRO-3")
    model = accelerator.unwrap_model(trainer.model_wrapped)
    tokenizer = trainer.processing_class
    require(tokenizer is not None and tokenizer.eos_token_id is not None, "Selection requires the training tokenizer and EOS")
    if trainer._generation_data is None:
        verify_sid_tokenizer(tokenizer, model, artifacts["catalog"])
        prefixes = build_prefix_allowed_tokens(tokenizer, (sid + "\n" for sid in sorted(artifacts["valid_sids"])))
        answer_tokens = max(map(len, prefixes)) + 1
        require(config.cutoff_len > answer_tokens, "Cutoff leaves no room for a SID prompt")

        class NativeShardedDataset(SidSFTDataset):
            def get_inputs(self):
                self.inputs = {index: self.pre(index) for index in range(rank, len(self.data), world)}

        # seed=None avoids resetting Python RNG in the middle of training.
        dataset = NativeShardedDataset(train_file=config.valid_file, tokenizer=tokenizer, max_len=config.cutoff_len,
                                       sample=-1, test=True, seed=None, category=CATEGORY_PROMPTS[config.category])
        require(len(dataset) == len(artifacts["targets"]), "Native validation dataset coverage differs")
        require(all(sha256_file(path) == artifacts["source_sha256"][key] for key, path in artifacts["source_paths"].items()),
                "Validation source changed during native prompt tokenization")
        trainer._generation_data = dataset, prefixes, answer_tokens
    dataset, prefixes, answer_tokens = trainer._generation_data
    local_ids = rank_row_ids(len(artifacts["targets"]), rank, world)
    identifier = [f"step-{trainer.state.global_step:09d}-{uuid.uuid4().hex[:12]}" if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(identifier, src=0, device=accelerator.device)
    directory = Path(trainer.args.output_dir) / "generative_validation" / identifier[0]
    directory.mkdir(parents=True, exist_ok=True)
    stats, completed, truncations = RankingStats(artifacts["valid_sids"], config.num_beams), [], 0
    prediction_path = directory / f"predictions.rank{rank}.jsonl"
    stream = prediction_path.open("w", encoding="utf-8") if config.write_predictions else None
    started, prior_training, prior_padding = time.monotonic(), model.training, tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode(), accelerator.autocast():
            for offset in range(0, len(local_ids), config.batch_size):
                row_ids = local_ids[offset:offset + config.batch_size]
                sequences = []
                budget = config.cutoff_len - answer_tokens
                for row_id in row_ids:
                    ids = dataset[row_id]["input_ids"]
                    truncations += int(len(ids) > budget)
                    ids = ids[-budget:]
                    sequences.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                batch = tokenizer.pad(sequences, padding=True, return_tensors="pt")
                batch = {key: value.to(accelerator.device) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
                width = batch["input_ids"].shape[1]
                processor = ConstrainedLogitsProcessor(lambda batch_id, completion: prefixes.get(tuple(completion), []),
                                                      num_beams=10, eos_token_id=tokenizer.eos_token_id, prompt_length=width)
                generation = GenerationConfig(num_beams=10, num_return_sequences=10, do_sample=False,
                                              use_cache=True, max_new_tokens=answer_tokens, length_penalty=0.0,
                                              early_stopping=False, renormalize_logits=False,
                                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                output = model.generate(**batch, generation_config=generation, synced_gpus=False,
                                        logits_processor=LogitsProcessorList([processor]))
                require(output.shape[0] == len(row_ids) * 10, "Incomplete generated beam batch")
                decoded = tokenizer.batch_decode(output[:, width:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for index, row_id in enumerate(row_ids):
                    candidates = [value.strip() for value in decoded[index * 10:(index + 1) * 10]]
                    target = artifacts["targets"][row_id]
                    stats.add(target, candidates)
                    completed.append(row_id)
                    if stream:
                        stream.write(json.dumps({"row_id": row_id, "target": target, "predictions": candidates}, ensure_ascii=False) + "\n")
    finally:
        tokenizer.padding_side = prior_padding
        model.train(prior_training)
        if stream:
            stream.close()
    values = torch.tensor([stats.rows, stats.hits, stats.ndcg_sum, stats.invalid_candidates, stats.duplicate_candidates,
                           stats.invalid_rows, truncations], dtype=torch.float64, device=accelerator.device)
    shards = [completed]
    if world > 1:
        dist.all_reduce(values)
        shards = [None] * world
        dist.all_gather_object(shards, completed)
    coverage = validate_coverage(shards, len(artifacts["targets"]))
    rows, hits, ndcg, invalid, duplicates, bad_rows, truncated = values.tolist()
    require(int(rows) == len(artifacts["targets"]), "Reduced validation row count differs")
    require(all(sha256_file(path) == artifacts["source_sha256"][key] for key, path in artifacts["source_paths"].items()),
            "Validation source changed during generation")
    report = {"status": "passed" if invalid == duplicates == 0 else "failed", "scope": SCOPE,
              'category': config.category, 'native_category': CATEGORY_PROMPTS[config.category],
              "coverage": coverage, "metrics": {"ndcg10": ndcg / rows, "recall10": hits / rows},
              "invalid_candidates": int(invalid), "duplicate_candidates": int(duplicates), "invalid_rows": int(bad_rows),
              "prompt_truncations": int(truncated), "source_sha256": artifacts["source_sha256"],
              "prediction_directory": str(directory), "world_size": world,
              "generation": {"num_beams": 10, "num_return_sequences": 10, "max_new_tokens": answer_tokens,
                             "cutoff_len": config.cutoff_len, "length_penalty": 0.0, "early_stopping": False,
                             "full_catalog": True, "seen_item_filter": False, "prompt_style": "native_SidSFTDataset_test_true"},
              "elapsed_seconds": time.monotonic() - started}
    if rank == 0:
        atomic_json(directory / "metrics.json", report)
    require(invalid == duplicates == 0, "Invalid/truncated or duplicate SID candidates invalidate selection; diagnostic predictions preserved")
    return report


def make_generative_trainer_class(base_trainer):
    class GenerativeSelectionTrainer(base_trainer):
        def __init__(self, *args, generative_validation, **kwargs):
            self.generative_validation = generative_validation
            self.selection_policy = SelectionPolicy(generative_validation.tolerance, generative_validation.patience)
            self._generation_artifacts = self._generation_data = self._generation_result = None
            self._selection_best = self._selection_latest = None
            super().__init__(*args, **kwargs)
            require(self.args.metric_for_best_model in ("selection_index", "eval_selection_index")
                    and self.args.greater_is_better is True and self.args.load_best_model_at_end,
                    "Use metric_for_best_model=selection_index, greater_is_better=True and load_best_model_at_end=True")
            strategy = lambda value: str(value).split(".")[-1].lower()
            require(strategy(self.args.eval_strategy) == strategy(self.args.save_strategy)
                    and strategy(self.args.save_strategy) in ("steps", "epoch"), "Evaluation and checkpoint saving must share epoch/steps cadence")
            if strategy(self.args.save_strategy) == "steps":
                require(self.args.eval_steps == self.args.save_steps, "Selection requires equal eval_steps and save_steps")
            require(not any(type(callback).__name__ == "EarlyStoppingCallback" for callback in self.callback_handler.callbacks),
                    "Remove EarlyStoppingCallback; generative selection owns checkpoint selection and patience")

        def _validation_artifacts(self):
            if self._generation_artifacts is None:
                self._generation_artifacts = validation_artifacts(self.generative_validation)
            return self._generation_artifacts

        def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
            require(metric_key_prefix == "eval", "Checkpoint selection supports validation scope only")
            output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
            result = generate_validation(self)
            require(output.num_samples == result["coverage"]["expected_rows"], "Teacher-forced validation and generation rows differ")
            result["metrics"]["eval_loss"] = float(output.metrics["eval_loss"])
            result.update(step=int(self.state.global_step), epoch=self.state.epoch)
            self.selection_policy.checked(result["metrics"])
            self._generation_result = result
            if self.is_world_process_zero():
                atomic_json(Path(result["prediction_directory"]) / "metrics.json", result)
            output.metrics.update(eval_ndcg10=result["metrics"]["ndcg10"], eval_recall10=result["metrics"]["recall10"],
                                  eval_generation_rows=result["coverage"]["completed_rows"],
                                  eval_selection_index=self.selection_policy.improvements + int(self.selection_policy.would_improve(result["metrics"])))
            return output

        def _selection_payload(self):
            return {"schema_version": 1, "scope": SCOPE, "selection_order": ["ndcg10", "recall10", "-eval_loss"],
                    "tolerance": self.selection_policy.tolerance, "patience": self.selection_policy.patience,
                    "early_stopping_metric": "ndcg10", "best_primary_ndcg": self.selection_policy.best_primary_ndcg,
                    "policy": self.selection_policy.to_dict(), "best": self._selection_best, "latest": self._selection_latest,
                    "source_sha256": self._validation_artifacts()["source_sha256"], "generation_config": asdict(self.generative_validation)}

        def _determine_best_metric(self, metrics, trial):
            require(self._generation_result is not None, "Generation validation must precede checkpoint selection")
            improved = self.selection_policy.consider(self._generation_result["metrics"], int(self.state.global_step))
            metrics["eval_selection_index"] = self.selection_policy.improvements
            delegated = super()._determine_best_metric(metrics, trial)
            require(bool(delegated) == improved, "Trainer best-metric state differs from lexicographic selection")
            checkpoint = str(Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}")
            record = {"step": int(self.state.global_step), "epoch": self.state.epoch, "checkpoint": checkpoint,
                      "metrics": dict(self._generation_result["metrics"]), "improved": improved,
                      "primary_improved": self.selection_policy.primary_improved,
                      "best_primary_ndcg": self.selection_policy.best_primary_ndcg,
                      "patience_counter": self.selection_policy.patience_counter,
                      **{key: self._generation_result[key] for key in ("coverage", "scope", "generation", "source_sha256", "prediction_directory")}}
            self._selection_latest = record
            if improved:
                self._selection_best = record
                self.control.should_save = True
            if self.selection_policy.patience_counter >= self.selection_policy.patience:
                self.control.should_training_stop = True
            if self.is_world_process_zero():
                directory = Path(self.args.output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "selection_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                atomic_json(directory / "selection_summary.json", {"status": "early_stopping" if self.control.should_training_stop else "training",
                            "evaluations": self.selection_policy.evaluations, **self._selection_payload()})
            return improved

        def _save_checkpoint(self, model, trial):
            result = super()._save_checkpoint(model, trial)
            if self.is_world_process_zero():
                path = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
                atomic_json(path / "generative_selection_state.json", self._selection_payload())
            return result

        def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
            if resume_from_checkpoint:
                if resume_from_checkpoint is True:
                    from transformers.trainer_utils import get_last_checkpoint
                    resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                require(resume_from_checkpoint, "No checkpoint found for selection resume")
                saved = json.loads((Path(resume_from_checkpoint) / "generative_selection_state.json").read_text(encoding="utf-8"))
                require(saved["generation_config"] == asdict(self.generative_validation)
                        and saved["source_sha256"] == self._validation_artifacts()["source_sha256"], "Selection resume protocol or source identity differs")
                self.selection_policy = SelectionPolicy.from_dict(saved["policy"])
                self._selection_best, self._selection_latest = saved["best"], saved["latest"]
                if self.is_world_process_zero():
                    path = Path(self.args.output_dir) / "selection_history.jsonl"
                    if path.exists():
                        lines = path.read_text(encoding="utf-8").splitlines()
                        kept = [line for line in lines if json.loads(line)["step"] <= self.selection_policy.last_step]
                        if len(kept) != len(lines):
                            shutil.copyfile(path, path.with_name(f"selection_history.before_resume.{time.time_ns()}.jsonl"))
                            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            else:
                require(not (Path(self.args.output_dir) / "selection_history.jsonl").exists(), "Existing selection history requires checkpoint resume or a new output directory")
            result = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
            if self.is_world_process_zero() and self.selection_policy.evaluations:
                atomic_json(Path(self.args.output_dir) / "selection_summary.json", {
                    "status": "early_stopping" if self.selection_policy.patience_counter >= self.selection_policy.patience else "completed",
                    "evaluations": self.selection_policy.evaluations,
                    "best_checkpoint": self.state.best_model_checkpoint,
                    "best_global_step": self.state.best_global_step, **self._selection_payload()})
            return result

    return GenerativeSelectionTrainer
