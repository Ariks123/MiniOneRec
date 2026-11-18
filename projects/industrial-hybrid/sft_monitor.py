"""Durable SFT metrics and masked validation for the pinned Trainer/ZeRO-2 stack.

The monitor consumes Trainer's existing pre-clipping global gradient norm. It
never reconstructs norms from local ZeRO partitions. This module's statistics
and file helpers also work without the optional training libraries installed.
"""
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

try:
    from transformers import TrainerCallback
except ImportError:  # Allows CPU-only statistics tests in the workspace shell.
    class TrainerCallback:
        pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_value(value.item())
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(json_value(payload), destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_source_identity(source, train_from_scratch=False):
    """Hash the local pretrained weights and tokenizer, once on the writing rank.

    File sizes are useful report metadata; SHA-256 values, rather than sizes or
    mtimes, determine identity. Hub snapshot weight symlinks are supported.
    """
    directory = Path(source).expanduser().resolve(strict=True)
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("SFT identity requires a local model directory with config.json")
    filenames = {"config.json"}
    if not train_from_scratch:
        if (directory / "model.safetensors").is_file():
            filenames.add("model.safetensors")
        elif (directory / "model.safetensors.index.json").is_file():
            filenames.add("model.safetensors.index.json")
            index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
            filenames.update(index["weight_map"].values())
        elif (directory / "pytorch_model.bin").is_file():
            filenames.add("pytorch_model.bin")
        elif (directory / "pytorch_model.bin.index.json").is_file():
            filenames.add("pytorch_model.bin.index.json")
            index = json.loads((directory / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
            filenames.update(index["weight_map"].values())
        else:
            raise ValueError("SFT identity could not find source model weights")
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                     "spiece.model", "generation_config.json"):
        if (directory / filename).is_file():
            filenames.add(filename)
    files = {}
    for filename in sorted(filenames):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Model weight index contains a nonlocal filename")
        path = directory / relative
        if not path.is_file():
            raise ValueError("Source model weight shard is missing: " + filename)
        files[filename] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    identity = {"format_version": 1, "source_directory": str(directory),
                "train_from_scratch": bool(train_from_scratch), "files": files}
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


def validate_resume_identity(previous, current):
    """Refuse resuming optimizer state against a changed source or training set."""
    previous_identity = previous.get("model", {}).get("identity")
    current_identity = current.get("model", {}).get("identity")
    if not previous_identity or not current_identity:
        raise ValueError("Resume requires a complete previous model SHA-256 identity in run_config.json")
    if previous_identity != current_identity:
        raise ValueError("Resume source model/config/tokenizer SHA-256 identity differs from the original run")
    required = ("train_file", "eval_file", "sid_index_path", "item_meta_path")
    previous_hashes = previous.get("data", {}).get("sha256", {})
    current_hashes = current.get("data", {}).get("sha256", {})
    for name in required:
        if not previous_hashes.get(name) or not current_hashes.get(name):
            raise ValueError("Resume requires both previous and current input SHA-256: " + name)
        if previous_hashes[name] != current_hashes[name]:
            raise ValueError("Resume input SHA-256 differs from the original run: " + name)
    for name in ("category", "sample"):
        if previous.get("data", {}).get(name) != current.get("data", {}).get(name):
            raise ValueError("Resume data configuration differs: " + name)
    # Changing these would silently turn a restart into a different experiment.
    for name in ("world_size", "micro_batch_size", "gradient_accumulation_steps", "global_batch_size",
                 "max_epochs", "learning_rate", "warmup_steps", "lr_scheduler_type", "cutoff_len",
                 "max_grad_norm", "bf16", "gradient_checkpointing", "attn_implementation", "seed",
                 "deepspeed_sha256"):
        if previous.get("hyperparameters", {}).get(name) != current.get("hyperparameters", {}).get(name):
            raise ValueError("Resume training configuration differs: " + name)


def write_verified_run_config(run_config, output_dir, resume_from_checkpoint, primary, accelerator):
    """Broadcast rank-zero identity/write errors before any baseline collective."""
    import torch

    response = [None]
    if primary:
        try:
            source = run_config["model"]["source"]
            run_config["model"]["identity"] = model_source_identity(
                source, train_from_scratch=run_config["model"]["train_from_scratch"]
            )
            run_config["data"]["sha256"] = {
                name: file_sha256(run_config["data"][name]) for name in
                ("train_file", "eval_file", "sid_index_path", "item_meta_path")
            }
            deepspeed = run_config.get("hyperparameters", {}).get("deepspeed")
            if deepspeed:
                run_config["hyperparameters"]["deepspeed_sha256"] = file_sha256(deepspeed)
            if resume_from_checkpoint:
                checkpoint = Path(resume_from_checkpoint).expanduser().resolve(strict=True)
                previous_path = checkpoint.parent / "run_config.json"
                if not previous_path.is_file():
                    raise ValueError("Resume checkpoint is missing its parent run_config.json: " + str(checkpoint))
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                validate_resume_identity(previous, run_config)
                target = Path(output_dir) / "run_config.json"
                if target.is_file() and target.resolve() != previous_path.resolve():
                    validate_resume_identity(json.loads(target.read_text(encoding="utf-8")), run_config)
                run_config["resume_verified_against"] = str(previous_path)
            atomic_json(Path(output_dir) / "run_config.json", run_config)
            response[0] = {"error": None, "identity_sha256": run_config["model"]["identity"]["sha256"]}
        except Exception as error:
            response[0] = {"error": f"{type(error).__name__}: {error}"}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(response, src=0, device=accelerator.device)
    elif not primary:
        raise RuntimeError("Distributed SFT identity verification requires an initialized process group")
    if response[0] is None or response[0].get("error"):
        error = response[0].get("error") if response[0] else "Missing rank-zero identity response"
        raise ValueError("SFT identity verification failed: " + error)
    return response[0]["identity_sha256"]


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


class GradientStatistics:
    def __init__(self, window=100):
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("monitor_window must be a positive integer")
        self.window = window
        self.recent = deque(maxlen=window)
        self.count = 0
        self.total = 0.0
        self.maximum = None
        self.maximum_step = None
        self.records = 0
        self.skipped = 0
        self.missing = 0
        self.nonfinite = 0
        self.last_step = -1

    def add(self, record):
        step = int(record["step"])
        if step <= self.last_step:
            return False
        self.last_step = step
        self.records += 1
        skipped = bool(record.get("step_skipped"))
        nonfinite = bool(record.get("grad_norm_nonfinite"))
        self.skipped += int(skipped)
        self.nonfinite += int(nonfinite)
        if skipped or nonfinite:
            return True
        value = finite_number(record.get("grad_norm"))
        if value is None:
            self.missing += 1
            return True
        self.count += 1
        self.total += value
        self.recent.append(value)
        if self.maximum is None or value > self.maximum:
            self.maximum, self.maximum_step = value, step
        return True

    def summary(self):
        return {
            "monitor_window": self.window,
            "update_records": self.records,
            "grad_norm_count": self.count,
            "grad_norm_mean": self.total / self.count if self.count else None,
            "grad_norm_max": self.maximum,
            "grad_norm_max_step": self.maximum_step,
            "rolling_grad_norm_count": len(self.recent),
            "rolling_grad_norm_mean": sum(self.recent) / len(self.recent) if self.recent else None,
            "rolling_grad_norm_max": max(self.recent) if self.recent else None,
            "skipped_update_count": self.skipped,
            "missing_grad_norm_count": self.missing,
            "nonfinite_grad_norm_count": self.nonfinite,
        }


class SFTMonitorCallback(TrainerCallback):
    def __init__(self, output_dir, window=100, require_grad_norm=True):
        self.directory = Path(output_dir) / "monitor"
        self.metrics_path = self.directory / "metrics.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.events_path = self.directory / "events.jsonl"
        self.statistics = GradientStatistics(window)
        self.require_grad_norm = require_grad_norm
        self.trainer = None
        self.forward_hook = None
        self.pending_bad_loss = False
        self.abort_reason = None
        self.last_step_skipped = False
        self.last_overflow = False
        self.start_time = time.monotonic()
        self.previous_elapsed = 0.0
        self.latest = {}
        self.evaluations = []
        self.phase = "initialized"
        self.primary = False
        self.stream = None

    def attach_trainer(self, trainer):
        self.trainer = trainer
        self.forward_hook = trainer.model.register_forward_hook(self._inspect_forward)
        return self

    def _inspect_forward(self, module, inputs, output):
        loss = getattr(output, "loss", None)
        if loss is not None:
            import torch
            self.pending_bad_loss |= not bool(torch.isfinite(loss.detach()).all().item())

    def _append(self, path, payload):
        if not self.primary:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_value(payload), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def _elapsed(self):
        return self.previous_elapsed + time.monotonic() - self.start_time

    def _write_summary(self, state=None):
        if not self.primary:
            return
        payload = {
            "status": self.phase,
            "updated_at_utc": utc_now(),
            "elapsed_seconds": self._elapsed(),
            "grad_norm_definition": "unscaled global full-model L2 norm before gradient clipping",
            "statistics_exclude_skipped_nonfinite_missing": True,
            "latest": self.latest,
            "evaluations": self.evaluations,
            "abort_reason": self.abort_reason,
            **self.statistics.summary(),
        }
        if state is not None:
            payload.update(global_step=state.global_step, epoch=state.epoch,
                           max_steps=state.max_steps,
                           best_checkpoint=getattr(state, "best_model_checkpoint", None),
                           best_metric=getattr(state, "best_metric", None))
        atomic_json(self.summary_path, payload)

    def _restore(self, step):
        if not self.metrics_path.is_file():
            return
        records = []
        parse_error = False
        unterminated_line = False
        with self.metrics_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                unterminated_line |= bool(line.strip()) and not line.endswith("\n")
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_error = True
                    # An interrupted final append is retained in the archive, not
                    # mistaken for a valid optimizer update.
                    if line.strip():
                        self._append(self.events_path, {"event": "incomplete_metric_line",
                                     "line_number": line_number, "timestamp_utc": utc_now()})
        if records and step == 0:
            raise ValueError("Existing monitor logs require a checkpoint resume or a new output directory")
        retained = [record for record in records if int(record.get("step", 0)) <= step]
        if self.primary and (len(retained) != len(records) or parse_error or unterminated_line):
            archive = self.directory / ("metrics.before_resume." + str(time.time_ns()) + ".jsonl")
            shutil.copyfile(self.metrics_path, archive)
            temporary = self.metrics_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for record in retained:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            os.replace(temporary, self.metrics_path)
        for record in retained:
            if record.get("kind", "train") == "train":
                self.statistics.add(record)
                self.latest = record
            elif record.get("kind") == "eval":
                self.evaluations.append(record)
            self.previous_elapsed = max(self.previous_elapsed, float(record.get("elapsed_seconds", 0)))

    def on_train_begin(self, args, state, control, **kwargs):
        self.primary = bool(state.is_world_process_zero)
        self.phase = "training"
        self.start_time = time.monotonic()
        # Read shared logs on every rank to make missing-norm stop decisions equal.
        self._restore(int(state.global_step))
        self._write_summary(state)
        return control

    def _stop(self, control, reason, state):
        self.abort_reason = self.abort_reason or reason
        self.phase = "failed"
        control.should_training_stop = True
        control.should_save = False
        control.should_evaluate = False
        self._append(self.events_path, {"event": "training_aborted", "reason": self.abort_reason,
                     "step": state.global_step, "timestamp_utc": utc_now()})
        self._write_summary(state)

    def _synchronize_bad_flag(self, bad):
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self.trainer.accelerator.device
            flag = torch.tensor([int(bad)], dtype=torch.int32, device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())
        return bool(bad)

    def on_step_end(self, args, state, control, **kwargs):
        wrapped = getattr(self.trainer, "model_wrapped", None)
        optimizer = getattr(self.trainer, "optimizer", None)
        engine_optimizer = getattr(wrapped, "optimizer", None)
        self.last_step_skipped = bool(getattr(optimizer, "step_was_skipped", False))
        self.last_overflow = bool(getattr(engine_optimizer, "overflow", False))
        self.last_step_skipped |= self.last_overflow
        norm = None
        if hasattr(wrapped, "get_global_grad_norm"):
            norm = wrapped.get_global_grad_norm()
        bad_norm = norm is not None and finite_number(norm) is None
        bad = self.pending_bad_loss or bad_norm or self.last_overflow
        self.pending_bad_loss = False
        if self._synchronize_bad_flag(bad):
            self._stop(control, "nonfinite_loss_or_gradient", state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        common = {"step": int(state.global_step), "epoch": state.epoch, "phase": self.phase,
                  "timestamp_utc": utc_now(), "elapsed_seconds": self._elapsed()}
        if "eval_loss" in logs:
            record = {**common, "kind": "eval", **json_value(logs)}
            self.evaluations.append(record)
            self._append(self.metrics_path, record)
            self._write_summary(state)
        if "loss" not in logs and "grad_norm" not in logs:
            return control
        raw_norm = logs.get("grad_norm")
        nonfinite = raw_norm is not None and finite_number(raw_norm) is None
        record = {
            **common, "kind": "train", "loss": finite_number(logs.get("loss")),
            "learning_rate": finite_number(logs.get("learning_rate")),
            "grad_norm": finite_number(raw_norm),
            "grad_norm_observed": str(raw_norm) if nonfinite else finite_number(raw_norm),
            "grad_norm_nonfinite": nonfinite,
            "step_skipped": self.last_step_skipped,
            "optimizer_overflow": self.last_overflow,
        }
        if self.statistics.add(record):
            stats = self.statistics.summary()
            record.update({key: stats[key] for key in
                           ("rolling_grad_norm_mean", "rolling_grad_norm_max", "grad_norm_mean",
                            "grad_norm_max", "grad_norm_max_step")})
            self.latest = record
            self._append(self.metrics_path, record)
            self._write_summary(state)
        if nonfinite or ("loss" in logs and finite_number(logs["loss"]) is None):
            self._stop(control, "nonfinite_loss_or_gradient", state)
        elif self.require_grad_norm and self.statistics.records >= 10 and self.statistics.count == 0:
            self._stop(control, "grad_norm_missing_for_first_10_updates", state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = (metrics or {}).get("eval_loss")
        bad = self.pending_bad_loss or (value is not None and finite_number(value) is None)
        self.pending_bad_loss = False
        if self._synchronize_bad_flag(bad):
            self._stop(control, "nonfinite_validation_loss", state)
        return control

    def on_save(self, args, state, control, **kwargs):
        self._append(self.events_path, {
            "event": "checkpoint_saved", "step": state.global_step,
            "checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}"),
            "best_checkpoint": getattr(state, "best_model_checkpoint", None),
            "best_metric": getattr(state, "best_metric", None), "timestamp_utc": utc_now(),
        })
        self._write_summary(state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.phase = "failed" if self.abort_reason else "trained_pending_save"
        self._write_summary(state)
        return control

    def finish(self, state, status, error=None):
        self.phase = status
        if error:
            self._append(self.events_path, {"event": "exception", "error": str(error),
                         "timestamp_utc": utc_now(), "step": state.global_step})
        self._write_summary(state)
        if self.forward_hook is not None:
            self.forward_hook.remove()
            self.forward_hook = None


def masked_validation(model, dataset, collator, batch_size, device, rank=0, world_size=1):
    """Token-weighted causal CE over unmasked answer labels, without ZeRO init."""
    import torch
    from torch.utils.data import DataLoader, Subset

    started = time.monotonic()
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    rows = Subset(dataset, range(rank, len(dataset), world_size))
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collator,
                        num_workers=0, pin_memory=str(device).startswith("cuda"))
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                count = batch["labels"][:, 1:].ne(-100).sum()
                output = model(**batch)
                finite = bool(torch.isfinite(output.loss.detach()).item())
                if finite and count.item() > 0:
                    totals[0] += output.loss.detach().double() * count
                    totals[1] += count
                else:
                    totals[3] += 1
                totals[2] += batch["input_ids"].shape[0]
                del output, batch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        values = totals.cpu().tolist()
        if values[3] or values[1] <= 0 or int(values[2]) != len(dataset):
            raise RuntimeError("Masked validation found a nonfinite/empty loss or incomplete row coverage: " + str(values))
        loss = values[0] / values[1]
        return {
            "status": "passed", "loss": loss,
            "masked_perplexity": math.exp(loss) if loss < 709 else None,
            "perplexity_overflow": loss >= 709,
            "loss_sum": values[0], "answer_token_count": int(values[1]),
            "sample_count": int(values[2]), "world_size": world_size,
            "definition": "token-weighted causal cross entropy on nonmasked answer labels only",
            "elapsed_seconds": time.monotonic() - started,
            "created_at_utc": utc_now(),
        }
    finally:
        model.train(was_training)
