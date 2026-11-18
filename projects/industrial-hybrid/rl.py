"""Continue the pinned Industrial 0.5B SFT model with MiniOneRec GRPO."""
import os
from pathlib import Path
import random

import npu_support
import numpy as np
import torch
from datasets import Dataset
from fire import Fire
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import GRPOConfig

from minionerec_trainer import ReReTrainer
from scripts.deepspeed_zero2_compat import apply_zero2_accumulation_fix
from scripts.industrial_generative_validation import GenerativeValidationConfig
from scripts.industrial_rl_data import (
    CATEGORY, EXPECTED_DATA_SHA256, build_rl_tasks, make_ndcg_rule_reward,
    require, rule_reward, verify_info_catalog, verify_parent_sft, write_task_manifest,
)
from scripts.industrial_rl_validation import (
    RLMonitorCallback, make_rl_trainer_class, run_initial_baseline, terminal_eval_callback,
)
from sft_monitor import atomic_json, file_sha256, utc_now, write_verified_run_config


class MonitoredTrainerMixin:
    """Custom RL loss has no model-output .loss for the SFT forward hook."""
    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        monitor = getattr(self, "rl_monitor", None)
        if monitor is not None:
            value = loss[0] if isinstance(loss, tuple) else loss
            monitor.pending_bad_loss |= not bool(torch.isfinite(value.detach()).all().item())
        return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    model_path: str = "", train_file: str = "", eval_file: str = "", info_file: str = "",
    sid_index_path: str = "", item_meta_path: str = "", output_dir: str = "",
    category: str = CATEGORY, deepspeed: str = "", num_train_epochs: int = 2,
    learning_rate: float = 1e-5, train_batch_size: int = 16, eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 16, num_generations: int = 16,
    reward_type: str = "ranking", beam_search: bool = True, sync_ref_model: bool = True,
    ref_model_sync_steps: int = 512, ref_model_mixup_alpha: float = 0.6,
    temperature: float = 1.0, beta: float = 0.001, seed: int = 42,
    gradient_checkpointing: bool = True, attn_implementation: str = "sdpa",
    generation_batch_size: int = 4, generation_num_beams: int = 50,
    cutoff_len: int = 512, max_completion_length: int = 128,
    eval_save_fraction: float = 0.125, early_stopping_patience: int = 3,
    monitor_window: int = 100, add_gt: bool = False, dynamic_sampling: bool = False,
    dapo: bool = False, gspo: bool = False, sample_train: bool = False,
    test_during_training: bool = False, wandb_run_name: str = "industrial_05b_grpo_ranking",
):
    require(category == CATEGORY and reward_type == "ranking", "Only Industrial ranking GRPO is supported")
    require(beam_search and sync_ref_model, "This experiment requires beam rollouts and reference synchronization")
    require(not any([add_gt, dynamic_sampling, dapo, gspo, sample_train, test_during_training]),
            "GT injection, alternative losses, sampling filters and extra test passes are disabled")
    require(seed == 42 and num_generations == 16, "This experiment pins seed42 and beam16 rollout groups")
    require(type(num_train_epochs) is int and 1 <= num_train_epochs <= 2,
            "Industrial RL is capped at two epochs")
    require(learning_rate > 0 and beta > 0 and 0 < eval_save_fraction < 1,
            "RL learning rate, KL beta and evaluation fraction must be positive")
    require(cutoff_len == 512 and generation_num_beams == 50,
            "Ranking selection uses the unchanged cutoff512/beam50 protocol")
    require(all(type(value) is int and value > 0 for value in
                [gradient_accumulation_steps, generation_batch_size, monitor_window, early_stopping_patience]),
            "RL batch/monitoring values must be positive integers")
    require(model_path and output_dir and deepspeed and Path(deepspeed).is_file(),
            "A local parent model, output directory and ZeRO-2 configuration are required")
    output = Path(output_dir).resolve()
    parent_model = Path(model_path).resolve()
    require(output != parent_model and parent_model not in output.parents
            and output not in parent_model.parents, "RL output must be separate from its SFT parent")
    require(not (output / "training_summary.json").exists()
            and not (output / "run_config.json").exists(), "Use a fresh RL run directory")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    require(world_size == 4 and torch.cuda.is_available(), "Industrial RL requires the configured four GPUs")
    require(0 <= local_rank < torch.cuda.device_count(), "LOCAL_RANK is outside the visible GPUs")
    torch.cuda.set_device(local_rank)
    set_seed(seed)
    compatibility = apply_zero2_accumulation_fix()
    os.environ["WANDB_MODE"] = "disabled"
    data_paths = {"train_file": train_file, "eval_file": eval_file,
                  "sid_index_path": sid_index_path, "item_meta_path": item_meta_path}
    parent = verify_parent_sft(model_path, data_paths, file_sha256)
    verify_info_catalog(info_file, sid_index_path)
    rows, valid_rows, task_counts = build_rl_tasks(train_file, eval_file, item_meta_path, sid_index_path, seed)
    training_data = Dataset.from_list(rows).shuffle(seed=seed)
    validation_data = Dataset.from_list(valid_rows).shuffle(seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    monitor = RLMonitorCallback(output, window=monitor_window, require_grad_norm=True)
    callbacks = [monitor, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                 terminal_eval_callback(monitor)]
    ranking = GenerativeValidationConfig(
        valid_file=eval_file, sid_index_path=sid_index_path,
        expected_sid_sha256=EXPECTED_DATA_SHA256["sid_index_path"],
        expected_valid_sha256=EXPECTED_DATA_SHA256["eval_file"], batch_size=generation_batch_size,
        cutoff_len=cutoff_len, num_beams=generation_num_beams, category=category, expected_rows=4532)
    validation_trainer = make_rl_trainer_class(ReReTrainer)

    class IndustrialRLTrainer(MonitoredTrainerMixin, validation_trainer):
        pass

    arguments = GRPOConfig(
        output_dir=str(output), model_init_kwargs={"torch_dtype": "bfloat16",
            "attn_implementation": attn_implementation, "device_map": {"": local_rank}},
        deepspeed=deepspeed, use_vllm=False, bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, num_generations=num_generations,
        num_train_epochs=num_train_epochs, learning_rate=learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03, max_grad_norm=0.3,
        optim="paged_adamw_32bit", beta=beta, temperature=temperature,
        sync_ref_model=sync_ref_model, ref_model_sync_steps=ref_model_sync_steps,
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        max_prompt_length=cutoff_len, max_completion_length=max_completion_length,
        eval_strategy="steps", eval_steps=eval_save_fraction,
        save_strategy="steps", save_steps=eval_save_fraction, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_ndcg10", greater_is_better=True,
        logging_steps=1, logging_nan_inf_filter=False, log_completions=False,
        report_to="none", seed=seed, data_seed=seed, ddp_find_unused_parameters=False,
        run_name=wandb_run_name,
    )
    trainer = IndustrialRLTrainer(
        model=str(parent_model), base_model=str(parent_model),
        reward_funcs=[rule_reward, make_ndcg_rule_reward(num_generations)], args=arguments,
        info_file=info_file, processing_class=tokenizer, beam_search=True,
        train_dataset=training_data, eval_dataset=validation_data,
        callbacks=callbacks, generative_validation=ranking)
    trainer.rl_monitor = monitor
    monitor.attach_trainer(trainer)
    monitor.primary = trainer.is_world_process_zero()
    run_config = {
        "stage": "rl", "created_at_utc": utc_now(), "runtime_compatibility": compatibility,
        "model": {"source": str(parent_model), "train_from_scratch": False,
                  "tokenizer_vocab_size": len(tokenizer)},
        "parent_sft": parent,
        "data": {**data_paths, "info_file": info_file, "info_sha256": file_sha256(info_file),
                 "category": category, "sample": -1, "include_title_history": True,
                 "train_samples": len(rows), "validation_samples": len(valid_rows)},
        "task_counts": task_counts,
        "hyperparameters": {
            "world_size": world_size, "micro_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_batch_size": train_batch_size * world_size * gradient_accumulation_steps,
            "global_prompt_groups_per_step": train_batch_size * world_size * gradient_accumulation_steps // num_generations,
            "max_epochs": num_train_epochs, "learning_rate": learning_rate, "lr_scheduler_type": "cosine",
            "warmup_ratio": .03, "max_grad_norm": .3, "bf16": True,
            "gradient_checkpointing": gradient_checkpointing, "attn_implementation": attn_implementation,
            "deepspeed": deepspeed, "seed": seed, "reward_type": reward_type,
            "num_generations": num_generations, "beam_search": True, "temperature": temperature,
            "beta": beta, "sync_ref_model": sync_ref_model,
            "ref_model_sync_steps": ref_model_sync_steps, "ref_model_mixup_alpha": ref_model_mixup_alpha,
            "cutoff_len": cutoff_len, "max_completion_length": max_completion_length,
            "early_stopping_patience": early_stopping_patience, "eval_save_fraction": eval_save_fraction,
            "generative_validation": True, "generation_batch_size": generation_batch_size,
            "generation_num_beams": generation_num_beams, "model_selection_metric": "eval_ndcg10",
            "optim": "paged_adamw_32bit", "monitor_window": monitor_window,
            "add_gt": False, "dynamic_sampling": False, "sample_train": False,
            "dapo": False, "gspo": False, "test_during_training": False,
        },
        "resume_from_checkpoint": None,
    }
    manifest_error = None
    if trainer.is_world_process_zero():
        try:
            run_config["data"]["task_manifest"] = write_task_manifest(
                output / "rl_task_manifest.jsonl", rows, valid_rows, seed)
        except Exception as error:
            manifest_error = f"Task manifest write failed: {type(error).__name__}: {error}"
    # All ranks either proceed to the shared run-config/baseline collectives or
    # fail together. A rank-zero filesystem error must not strand other ranks.
    trainer._coordinated_require(manifest_error)
    write_verified_run_config(run_config, str(output), None, trainer.is_world_process_zero(), trainer.accelerator)
    final = output / "final_checkpoint"
    try:
        run_initial_baseline(trainer)
        result = trainer.train()
        if monitor.abort_reason:
            raise RuntimeError("RL monitoring stopped training: " + monitor.abort_reason)
        require(trainer.state.best_model_checkpoint is not None, "RL finished without a validation-selected checkpoint")
        trainer.rl_validation_phase = "selected_model"
        monitor.phase = "final_validation"
        final_metrics = trainer.evaluate()
        if monitor.abort_reason:
            raise RuntimeError("RL monitoring stopped validation: " + monitor.abort_reason)
        trainer.save_model(str(final))
        trainer.save_state()
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            state = trainer.state
            atomic_json(output / "training_summary.json", {
                "status": "completed", "stage": "rl", "finished_at_utc": utc_now(),
                "train_metrics": getattr(result, "metrics", {}), "final_eval_metrics": final_metrics,
                "best_checkpoint": state.best_model_checkpoint, "best_metric": state.best_metric,
                "selection_metric": "eval_ndcg10", "selected_validation_metrics": final_metrics,
                "final_checkpoint": str(final), "global_step": state.global_step, "epoch": state.epoch,
                "planned_max_steps": state.max_steps,
                "stop_reason": "early_stopping" if state.global_step < state.max_steps else "max_epochs",
                "final_checkpoint_contains_best_model": True, "parent_sft": parent,
                "runtime_compatibility": compatibility,
            })
        trainer.accelerator.wait_for_everyone()
        monitor.finish(trainer.state, "completed")
    except Exception as error:
        monitor.finish(trainer.state, "failed", error=error)
        if trainer.is_world_process_zero():
            atomic_json(output / "training_summary.json", {
                "status": "failed", "stage": "rl", "finished_at_utc": utc_now(),
                "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                "stop_reason": monitor.abort_reason or "exception", "error": str(error),
                "best_checkpoint": trainer.state.best_model_checkpoint, "final_checkpoint": None,
            })
        raise


if __name__ == "__main__":
    Fire(train)
