"""Isolated five-task SFT plus Qwen self-retrieval; run with four-rank torchrun."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "train-file", "eval-file", "output-dir", "sid-index-path", "item-meta-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deepspeed", default="config/h20_zero2.json")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lr-scheduler-type", choices=("linear",), default="linear")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cutoff-len", type=int, default=512)
    p.add_argument("--projection-dim", type=int, choices=(256,), default=256)
    p.add_argument("--retrieval-weight", type=float, default=.1)
    p.add_argument("--temperature", type=float, default=.07)
    p.add_argument("--eval-save-fraction", type=float, default=.05)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    return p


def make_joint_trainer_class(base):
    class JointTrainer(base):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Native Qwen accepts token counts spanning all accumulated batches.
            # Keep this official generation contract, and scale only the new
            # query-mean retrieval objective for the surrounding Trainer path.
            if "retrieval_mask" in inputs and num_items_in_batch is not None:
                divisor = self.current_gradient_accumulation_steps
                if self.args.average_tokens_across_devices:
                    divisor *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
                inputs = {**inputs, "retrieval_loss_scale": 1.0 / divisor}
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        def log(self, logs, *a, **kw):
            if self.model.training:
                logs = {**logs, **{("retrieval/" + key): value
                    for key, value in self.model.last_retrieval_stats.items()}}
            return super().log(logs, *a, **kw)
    return JointTrainer


def train(args):
    import npu_support
    import torch
    import transformers
    from datasets import Dataset as HFDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
    from sft import TokenExtender, set_seed, write_sid_provenance
    from sft_monitor import (SFTMonitorCallback, atomic_json, file_sha256, masked_validation,
                             utc_now, write_verified_run_config)
    from scripts.deepspeed_zero2_compat import apply_zero2_accumulation_fix
    from qwen_retrieval.config import ITEM_TEMPLATE
    from qwen_retrieval.model import QwenRetrievalForCausalLM
    from qwen_retrieval.training_data import build_training_data, RetrievalCollator

    world, local_rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    if args.num_epochs < 1 or args.global_batch_size % (args.micro_batch_size * world):
        raise ValueError("Epoch budget must be positive and global batch divisible by micro batch times ranks")
    if Path(args.train_file).name != "train.csv" or Path(args.eval_file).name != "valid.csv":
        raise ValueError("Only prepared train.csv/valid.csv may enter training and selection")
    if not 0 < args.eval_save_fraction <= 1:
        raise ValueError("Evaluation/save fraction must be in (0,1]")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    accumulation = args.global_batch_size // (args.micro_batch_size * world)
    compatibility = apply_zero2_accumulation_fix() if args.deepspeed else {"applied": False}
    compatibility = {**compatibility, "already_applied": False, "scope": "current_retrieval_sft_process_only"}
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        **({"device_map": {"": local_rank}} if torch.cuda.is_available() else {}))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    original_vocab = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokens = TokenExtender(str(Path(args.sid_index_path).parent),
                           Path(args.sid_index_path).name.split('.')[0]).get_new_tokens()
    tokenizer.add_tokens(tokens)
    # Exact historical ordering: load native Qwen, extend tokenizer/embeddings,
    # then add the head under a private RNG fork without reinitializing Qwen.
    model.resize_token_embeddings(len(tokenizer))
    model.__class__ = QwenRetrievalForCausalLM
    model.config.retrieval_dim = args.projection_dim
    model.config.retrieval_weight = args.retrieval_weight
    model.config.retrieval_temperature = args.temperature
    model.config.retrieval_item_template = ITEM_TEMPLATE
    model.enable_retrieval()
    model.config.use_cache = False
    raw_train, raw_valid, counts = build_training_data(args.train_file, args.eval_file,
        args.item_meta_path, args.sid_index_path, tokenizer, args.cutoff_len, args.seed)
    expected = {"sid_sequence_recommendation": 36259, "history_to_title": 36259,
                "sid2title": 3670, "title2sid": 3646, "title_history_to_sid": 36259}
    if counts != expected or len(raw_train) != 116093 or len(raw_valid) != 4532:
        raise ValueError("Pinned original Industrial five-task mixture or validation coverage changed")
    train_data = HFDataset.from_list(list(raw_train)).shuffle(seed=42)
    valid_data = HFDataset.from_list(list(raw_valid)).shuffle(seed=args.seed).shuffle(seed=42)
    collator = RetrievalCollator(tokenizer)
    monitor = SFTMonitorCallback(args.output_dir, window=100)

    JointTrainer = make_joint_trainer_class(transformers.Trainer)
    class RetrievalArtifactsCallback(TrainerCallback):
        def on_log(self, callback_args, state, control, logs=None, **kwargs):
            diagnostics = {key: value for key, value in (logs or {}).items() if key.startswith("retrieval/")}
            if state.is_world_process_zero and diagnostics:
                path = Path(args.output_dir) / "monitor/retrieval_metrics.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                        "scope": "last_microbatch; retrieval globally averaged; generation rank0 token mean",
                        **diagnostics}, allow_nan=False) + "\n")

        def on_save(self, callback_args, state, control, **kwargs):
            error = [None]
            if state.is_world_process_zero:
                try:
                    checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    tokenizer.save_pretrained(checkpoint)
                    write_sid_provenance(checkpoint, args.sid_index_path, args.item_meta_path,
                                         args.base_model, "Industrial_and_Scientific")
                except Exception as exception:
                    error[0] = str(exception)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(error, src=0, device=trainer.accelerator.device)
            if error[0]:
                raise RuntimeError("Checkpoint metadata publication failed: " + error[0])

    trainer = JointTrainer(model=model, train_dataset=train_data, eval_dataset=valid_data,
        processing_class=tokenizer, data_collator=collator,
        args=transformers.TrainingArguments(output_dir=args.output_dir,
            deepspeed=args.deepspeed or None, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=accumulation, warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs, learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type, bf16=True, logging_steps=1,
            logging_nan_inf_filter=False, max_grad_norm=1., optim="adamw_torch",
            eval_strategy="steps", eval_steps=args.eval_save_fraction,
            save_strategy="steps", save_steps=args.eval_save_fraction,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, ddp_find_unused_parameters=False if world > 1 else None,
            remove_unused_columns=False, group_by_length=False, report_to="none",
            seed=args.seed, data_seed=args.seed),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), monitor, RetrievalArtifactsCallback()])
    monitor.attach_trainer(trainer)
    monitor.primary = trainer.is_world_process_zero()
    source_root = Path(__file__).resolve().parents[1]
    sources = [str(path.relative_to(source_root)) for path in sorted((source_root / "qwen_retrieval").glob("*.py"))]
    sources += ["sft.py", "data.py", "sft_monitor.py", "scripts/deepspeed_zero2_compat.py"]
    config = {"stage": "sft", "method": "qwen_shared_self_retrieval_v1", "created_at_utc": utc_now(),
        "model": {"source": args.base_model, "train_from_scratch": False,
                  "original_vocab_size": original_vocab, "tokenizer_vocab_size": len(tokenizer),
                  "added_sid_tokens": len(tokens)},
        "data": {"train_file": args.train_file, "eval_file": args.eval_file,
                 "sid_index_path": args.sid_index_path, "item_meta_path": args.item_meta_path,
                 "category": "Industrial_and_Scientific", "sample": -1,
                 "include_title_history": True, "train_samples": len(train_data),
                 "validation_samples": len(valid_data)}, "task_counts": counts,
        "hyperparameters": {"world_size": world, "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": accumulation, "global_batch_size": args.global_batch_size,
            "max_epochs": args.num_epochs, "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps, "lr_scheduler_type": args.lr_scheduler_type,
            "cutoff_len": args.cutoff_len, "max_grad_norm": 1., "bf16": True,
            "gradient_checkpointing": True, "attn_implementation": "sdpa", "seed": args.seed,
            "deepspeed": args.deepspeed, "eval_save_fraction": args.eval_save_fraction,
            "early_stopping_patience": args.early_stopping_patience, "model_selection_metric": "eval_loss",
            "generation_loss_normalization": "native Qwen ForCausalLMLoss + Trainer num_items_in_batch contract",
            "average_tokens_across_devices": trainer.args.average_tokens_across_devices,
            "projection_dim": args.projection_dim, "retrieval_weight": args.retrieval_weight,
            "retrieval_temperature": args.temperature, "retrieval_item_template": ITEM_TEMPLATE},
        "runtime_compatibility": compatibility,
        "historical_comparison_caveat": "Historical SFT did not enable the DeepSpeed 0.18.0 accumulation correction; this run does.",
        "source_sha256": {name: file_sha256(source_root / name) for name in sources},
        "resume_from_checkpoint": args.resume_from_checkpoint}
    if args.resume_from_checkpoint:
        previous = json.loads((Path(args.resume_from_checkpoint).parent / "run_config.json").read_text())
        for name in ("task_counts", "source_sha256", "runtime_compatibility"):
            if previous.get(name) != config.get(name):
                raise ValueError("Resume differs in " + name)
        for name in ("projection_dim", "retrieval_weight", "retrieval_temperature", "retrieval_item_template",
                     "eval_save_fraction", "early_stopping_patience"):
            if previous["hyperparameters"].get(name) != config["hyperparameters"].get(name):
                raise ValueError("Resume retrieval configuration differs in " + name)
    write_verified_run_config(config, args.output_dir, args.resume_from_checkpoint,
                              trainer.is_world_process_zero(), trainer.accelerator)
    final = str(Path(args.output_dir) / "final_checkpoint")
    try:
        if not args.resume_from_checkpoint:
            baseline = masked_validation(model, valid_data, collator, args.micro_batch_size,
                trainer.accelerator.device, rank=trainer.accelerator.process_index,
                world_size=trainer.accelerator.num_processes)
            if trainer.is_world_process_zero():
                atomic_json(Path(args.output_dir) / "baseline_validation.json", baseline)
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if monitor.abort_reason:
            raise RuntimeError("Monitor stopped training: " + monitor.abort_reason)
        monitor.phase = "final_validation"
        metrics = trainer.evaluate()
        trainer.save_model(final)
        trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final)
            write_sid_provenance(final, args.sid_index_path, args.item_meta_path,
                                 args.base_model, "Industrial_and_Scientific")
            # Resume checkpoints also need their own tokenizer/provenance.
            for checkpoint in Path(args.output_dir).glob("checkpoint-*"):
                if checkpoint.is_dir():
                    tokenizer.save_pretrained(checkpoint)
                    write_sid_provenance(checkpoint, args.sid_index_path, args.item_meta_path,
                                         args.base_model, "Industrial_and_Scientific")
            atomic_json(Path(args.output_dir) / "training_summary.json", {
                "status": "completed", "stage": "sft", "finished_at_utc": utc_now(),
                "train_metrics": result.metrics, "final_eval_metrics": metrics,
                "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
                "selection_metric": "eval_loss", "final_checkpoint_contains_best_model": True,
                "final_checkpoint": final, "epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step, "planned_max_steps": trainer.state.max_steps,
                "stop_reason": "early_stopping" if trainer.state.global_step < trainer.state.max_steps else "max_epochs"})
        trainer.accelerator.wait_for_everyone()
        monitor.finish(trainer.state, "completed")
    except Exception as error:
        monitor.finish(trainer.state, "failed", error=error)
        if trainer.is_world_process_zero():
            atomic_json(Path(args.output_dir) / "training_summary.json", {
                "status": "failed", "error": str(error), "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch, "final_checkpoint": None})
        raise


if __name__ == "__main__":
    train(parser().parse_args())
