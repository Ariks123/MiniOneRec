import npu_support
import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import hashlib
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from sft_monitor import SFTMonitorCallback, atomic_json, masked_validation, utc_now, write_verified_run_config


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def write_sid_provenance(output_dir, sid_index_path, item_meta_path, source_model, category):
    """Record the SID catalog used for a saved SFT checkpoint on the writing rank."""
    if not sid_index_path or not os.path.isfile(sid_index_path):
        return None

    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    metadata_hash = None
    if item_meta_path:
        if not os.path.isfile(item_meta_path):
            raise ValueError("Cannot record SFT provenance: item metadata file is missing")
        metadata_hash = file_sha256(item_meta_path)
    record = {
        "format_version": 1,
        "stage": "sft",
        "sid_index_sha256": file_sha256(sid_index_path),
        "item_meta_sha256": metadata_hash,
        "source_model": str(source_model),
        "category": category,
    }
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "posttraining_provenance.json")
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(record, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary, target)
    return target


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    lr_scheduler_type: str = "linear",
    save_total_limit: int = 2,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    deepspeed: str = None,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "sdpa",
    monitoring: bool = True,
    monitor_window: int = 100,
    baseline_eval: bool = True,
    initialize_on_cpu: bool = False,
):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be a positive integer")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (batch_size, micro_batch_size)):
        raise ValueError("batch_size and micro_batch_size must be positive integers")
    effective_micro_batch = micro_batch_size * world_size
    if batch_size % effective_micro_batch:
        raise ValueError(
            "batch_size must be divisible by micro_batch_size * WORLD_SIZE "
            f"({micro_batch_size} * {world_size} = {effective_micro_batch})"
        )
    gradient_accumulation_steps = batch_size // effective_micro_batch
    if isinstance(monitor_window, bool) or not isinstance(monitor_window, int) or monitor_window <= 0:
        raise ValueError("monitor_window must be a positive integer")
    if resume_from_checkpoint and not monitoring:
        raise ValueError("SFT checkpoint resume requires monitoring and SHA-256 identity verification")
    ddp = world_size > 1
    if ddp and "LOCAL_RANK" not in os.environ:
        raise ValueError("Distributed SFT requires LOCAL_RANK; launch with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK {local_rank} is outside the visible CUDA devices")
        # Bind the rank before seeding or loading weights onto a GPU.
        torch.cuda.set_device(local_rank)
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_key = category
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books", "Video_Games": "video games"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='Qwen/Qwen2.5-1.5B'"
    print(f"SFT world_size={world_size}, per_device_batch={micro_batch_size}, "
          f"gradient_accumulation={gradient_accumulation_steps}, global_batch={batch_size}")
    model_load_kwargs = {"torch_dtype": torch.bfloat16}
    if attn_implementation:
        model_load_kwargs["attn_implementation"] = attn_implementation
    if ddp and cuda_available and not initialize_on_cpu:
        # Each process owns one complete model replica; never auto-shard it across ranks.
        model_load_kwargs["device_map"] = {"": local_rank}
    # Source-faithful initialization leaves weights and new SID embeddings on CPU.
    # Trainer subsequently moves the complete model onto the already-bound rank.
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)
    else:
        config = AutoConfig.from_pretrained(base_model)
        config_kwargs = {key: value for key, value in model_load_kwargs.items() if key != "device_map"}
        model = AutoModelForCausalLM.from_config(config, **config_kwargs)
        if ddp and cuda_available and not initialize_on_cpu:
            model.to(torch.device("cuda", local_rank))
        print("Training from scratch!")
    # Cached decoding is incompatible with checkpointed training and is unnecessary for SFT.
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    new_tokens = []
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data2)
    train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    train_datasets.append(train_data3)
    # train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data5)
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    monitor = SFTMonitorCallback(output_dir, window=monitor_window) if monitoring else None
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if monitor is not None:
        callbacks.append(monitor)
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            deepspeed=deepspeed or None,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            bf16=True,
            logging_steps=1,
            logging_nan_inf_filter=False,
            max_grad_norm=1.0,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none",
        ),
        data_collator=collator,
        callbacks=callbacks,
        # optimizers=(optimizer, lr_scheduler) 
    )
    if monitor is not None:
        monitor.attach_trainer(trainer)
        monitor.primary = trainer.is_world_process_zero()
        alignment_counts = {}
        for example in getattr(train_data2, "data", []):
            if isinstance(example, dict) and "task" in example:
                alignment_counts[example["task"]] = alignment_counts.get(example["task"], 0) + 1
        run_config = {
            "stage": "sft", "created_at_utc": utc_now(),
            "model": {"source": base_model, "train_from_scratch": train_from_scratch,
                      "original_vocab_size": original_vocab_size,
                      "tokenizer_vocab_size": len(tokenizer), "added_sid_tokens": len(new_tokens)},
            "data": {"train_file": train_file, "eval_file": eval_file,
                     "sid_index_path": sid_index_path, "item_meta_path": item_meta_path,
                     "category": category_key, "sample": sample,
                     "train_samples": len(hf_train_dataset), "validation_samples": len(hf_val_dataset)},
            "task_counts": {"sid_sequence_recommendation": len(train_data1),
                            "history_to_title": len(train_data3),
                            **(alignment_counts or {"sid_item_alignment": len(train_data2)})},
            "hyperparameters": {"world_size": world_size, "micro_batch_size": micro_batch_size,
                                "gradient_accumulation_steps": gradient_accumulation_steps,
                                "global_batch_size": batch_size, "max_epochs": num_epochs,
                                "learning_rate": learning_rate, "warmup_steps": 20,
                                "lr_scheduler_type": str(trainer.args.lr_scheduler_type).split(".")[-1].lower(), "cutoff_len": cutoff_len,
                                "max_grad_norm": 1.0, "bf16": True,
                                "gradient_checkpointing": gradient_checkpointing,
                                "initialize_on_cpu": initialize_on_cpu,
                                "attn_implementation": attn_implementation,
                                "deepspeed": deepspeed, "seed": seed,
                                "early_stopping_patience": 3, "eval_save_fraction": eval_step,
                                "save_total_limit": save_total_limit, "baseline_eval": baseline_eval,
                                "monitor_window": monitor_window},
            "resume_from_checkpoint": resume_from_checkpoint,
        }
        write_verified_run_config(run_config, output_dir, resume_from_checkpoint,
                                  trainer.is_world_process_zero(), trainer.accelerator)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    try:
        if baseline_eval and not resume_from_checkpoint:
            baseline = masked_validation(
                model, hf_val_dataset, collator, micro_batch_size, trainer.accelerator.device,
                rank=trainer.accelerator.process_index, world_size=trainer.accelerator.num_processes,
            )
            if trainer.is_world_process_zero():
                atomic_json(os.path.join(output_dir, "baseline_validation.json"), baseline)
        # Baseline inference is deliberately outside the ZeRO-2 engine. Trainer's
        # evaluate-before-train path initializes DeepSpeed in unsupported inference mode.
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if monitor is not None and monitor.abort_reason:
            raise RuntimeError("SFT monitoring stopped training: " + monitor.abort_reason)
        if monitor is not None:
            monitor.phase = "final_validation"
        final_eval_metrics = trainer.evaluate() if monitoring else None
        if monitor is not None and monitor.abort_reason:
            raise RuntimeError("SFT monitoring stopped validation: " + monitor.abort_reason)
        trainer.save_model(output_dir)

        # All ranks must enter Trainer saving so distributed state gathering remains safe.
        trainer.save_model(final_output_dir)
        if monitoring:
            trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_output_dir)
            write_sid_provenance(
                final_output_dir, sid_index_path, item_meta_path, base_model, category_key
            )
            if monitoring:
                state = trainer.state
                atomic_json(os.path.join(output_dir, "training_summary.json"), {
                    "status": "completed", "stage": "sft", "finished_at_utc": utc_now(),
                    "train_metrics": getattr(result, "metrics", {}),
                    "final_eval_metrics": final_eval_metrics,
                    "best_checkpoint": state.best_model_checkpoint, "best_metric": state.best_metric,
                    "final_checkpoint": final_output_dir,
                    "global_step": state.global_step, "epoch": state.epoch,
                    "planned_max_steps": state.max_steps,
                    "stop_reason": "early_stopping" if state.global_step < state.max_steps else "max_epochs",
                    "final_checkpoint_contains_best_model": True,
                })
        if monitoring:
            trainer.accelerator.wait_for_everyone()
        if monitor is not None:
            monitor.finish(trainer.state, "completed")
    except Exception as error:
        if monitor is not None:
            monitor.finish(trainer.state, "failed", error=error)
        if monitoring and trainer.is_world_process_zero():
            atomic_json(os.path.join(output_dir, "training_summary.json"), {
                "status": "failed", "stage": "sft", "finished_at_utc": utc_now(),
                "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                "stop_reason": getattr(monitor, "abort_reason", None) or "exception",
                "error": str(error), "final_checkpoint": None,
                "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            })
        raise



if __name__ == "__main__":
    fire.Fire(train)
