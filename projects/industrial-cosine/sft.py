import npu_support
import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
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
from pathlib import Path
import shutil
import tempfile
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from sft_monitor import SFTMonitorCallback, atomic_json, masked_validation, utc_now, write_verified_run_config, file_sha256


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


def build_training_tasks(train_file, item_meta_path, sid_index_path, tokenizer,
                         cutoff_len, sample, seed, category, include_title_history=False):
    """Build tasks from training interactions and the shared item catalog only."""
    if not isinstance(include_title_history, bool):
        raise ValueError("include_title_history must be a boolean")
    common = dict(tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
                  seed=seed, category=category)
    sequence = SidSFTDataset(train_file=train_file, **common)
    alignment = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path,
                                   **common)
    history_to_title = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path,
                                          index_file=sid_index_path, **common)
    datasets = [sequence, alignment, history_to_title]
    alignment_counts = {}
    for example in getattr(alignment, "data", []):
        if isinstance(example, dict) and "task" in example:
            task = example["task"]
            alignment_counts[task] = alignment_counts.get(task, 0) + 1
    counts = {"sid_sequence_recommendation": len(sequence),
              "history_to_title": len(history_to_title),
              **(alignment_counts or {"sid_item_alignment": len(alignment)})}
    if include_title_history:
        title_history = TitleHistory2SidSFTDataset(
            train_file=train_file, item_file=item_meta_path, index_file=sid_index_path,
            **common)
        datasets.append(title_history)
        counts["title_history_to_sid"] = len(title_history)
    return ConcatDataset(datasets), counts


def ranking_values(metrics):
    """Validate the validation-only metrics used for independent checkpoint retention."""
    values = {name: float(metrics["eval_" + name]) for name in ("ndcg10", "recall10", "loss")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Checkpoint selection requires finite validation metrics")
    if not 0 <= values["ndcg10"] <= 1 or not 0 <= values["recall10"] <= 1 or values["loss"] < 0:
        raise ValueError("Checkpoint selection metrics are outside their valid ranges")
    return values


def recall_improves(candidate, previous):
    """Keep maximum Recall@10; resolve exact ties with NDCG@10, then lower loss."""
    key = lambda values: (values["recall10"], values["ndcg10"], -values["loss"])
    return previous is None or key(candidate) > key(previous)


class BestRecallCheckpointCallback(TrainerCallback):
    """Mirror the just-saved Recall-best model before later checkpoint rotation.

    Trainer saves the full model collectively first. This callback only copies
    inference files on rank zero, then broadcasts any copy error to every rank.
    It never calls a ZeRO save collective from rank zero alone.
    """
    def __init__(self, output_dir, tokenizer, sid_index_path, item_meta_path,
                 base_model, category):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.sid_index_path, self.item_meta_path = sid_index_path, item_meta_path
        self.base_model, self.category = base_model, category
        self.best_recall = self.best_ndcg = self.pending = None
        self.active = True
        path = self.output_dir / "ranking_selection.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            self.best_recall = previous.get("best_recall")
            self.best_ndcg = previous.get("best_ndcg")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.active:
            return control
        values = ranking_values(metrics or {})
        record = {"step": state.global_step, "epoch": state.epoch, "metrics": values,
                  "source_checkpoint": str(self.output_dir / f"checkpoint-{state.global_step}"),
                  "scope": "complete_original_validation_split"}
        if self.best_ndcg is None or values["ndcg10"] > self.best_ndcg["metrics"]["ndcg10"]:
            self.best_ndcg = dict(record)
        if recall_improves(values, None if self.best_recall is None else self.best_recall["metrics"]):
            self.best_recall = dict(record)
            self.pending = dict(record)
            control.should_save = True
        return control

    def _publish_recall_model(self, record):
        source = Path(record["source_checkpoint"])
        if not source.is_dir():
            raise ValueError("Recall-best source checkpoint is unavailable: " + str(source))
        target = self.output_dir / "best_recall_checkpoint"
        stage = Path(tempfile.mkdtemp(prefix=".best-recall-", dir=self.output_dir))
        backup = self.output_dir / ".previous_best_recall_checkpoint"
        try:
            weights = []
            for path in source.iterdir():
                if path.is_file() and (path.name.endswith(".safetensors")
                                      or path.name.startswith("pytorch_model")
                                      or path.name in ("config.json", "generation_config.json",
                                                       "model.safetensors.index.json")):
                    shutil.copyfile(path, stage / path.name)
                    if path.suffix in (".safetensors", ".bin"):
                        weights.append(path.name)
            if not weights or not (stage / "config.json").is_file():
                raise ValueError("Recall-best checkpoint has no complete model export")
            self.tokenizer.save_pretrained(stage)
            write_sid_provenance(stage, self.sid_index_path, self.item_meta_path,
                                 self.base_model, self.category)
            atomic_json(stage / "selection_metrics.json", record)
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                os.replace(target, backup)
            os.replace(stage, target)
            if backup.exists():
                shutil.rmtree(backup)
        except BaseException:
            if not target.exists() and backup.exists():
                os.replace(backup, target)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def on_save(self, args, state, control, **kwargs):
        error = [None]
        if state.is_world_process_zero:
            try:
                if self.pending is not None:
                    self._publish_recall_model(self.pending)
                atomic_json(self.output_dir / "ranking_selection.json", {
                    "scope": "complete_original_validation_split", "primary_metric": "ndcg10",
                    "early_stopping_metric": "ndcg10", "best_ndcg": self.best_ndcg,
                    "best_recall": self.best_recall,
                    "best_recall_checkpoint": str(self.output_dir / "best_recall_checkpoint"),
                })
            except Exception as exception:
                error[0] = str(exception)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError("Could not retain validation-best Recall model: " + error[0])
        self.pending = None
        return control


class MinimumEpochEarlyStoppingCallback(EarlyStoppingCallback):
    """Delay this callback's patience stop while preserving safety stop flags."""
    def __init__(self, minimum_epochs, **kwargs):
        super().__init__(**kwargs)
        self.minimum_epochs = float(minimum_epochs)

    def state(self):
        value = super().state()
        value["args"]["minimum_epochs"] = self.minimum_epochs
        return value

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        already_stopping = bool(control.should_training_stop)
        result = super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        control = control if result is None else result
        if already_stopping:
            control.should_training_stop = True
            return control
        metric_name = args.metric_for_best_model
        if not metric_name.startswith("eval_"):
            metric_name = "eval_" + metric_name
        # Non-finite metrics must retain any stop requested by the parent;
        # the monitor's own safety decisions are never countermanded here.
        for name in (metric_name, "eval_loss"):
            value = (metrics or {}).get(name)
            if value is not None and not math.isfinite(float(value)):
                return control
        if float(state.epoch or 0.0) + 1e-9 < self.minimum_epochs:
            control.should_training_stop = False
        return control


class EpochSnapshotCallback(TrainerCallback):
    """Export live weights at an exact epoch boundary before best-model reload."""
    def __init__(self, output_dir, target_epoch, tokenizer, sid_index_path,
                 item_meta_path, base_model, category):
        self.output_dir, self.target_epoch = Path(output_dir), int(target_epoch)
        self.directory = self.output_dir / f"epoch_{self.target_epoch}_checkpoint"
        self.tokenizer, self.trainer = tokenizer, None
        self.sid_index_path, self.item_meta_path = sid_index_path, item_meta_path
        self.base_model, self.category = base_model, category
        self.metadata = None
        existing = self.directory / "snapshot_metadata.json"
        if existing.is_file():
            self.metadata = json.loads(existing.read_text(encoding="utf-8"))
            if (self.metadata.get("requested_epoch") != self.target_epoch
                    or self.metadata.get("snapshot_type") != "epoch_end_training_weights"
                    or self.metadata.get("capture_event") != "on_epoch_end"
                    or self.metadata.get("epoch") != float(self.target_epoch)
                    or Path(self.metadata.get("checkpoint_path", "")).resolve() != self.directory.resolve()
                    or self.metadata.get("best_model_reloaded") is not False):
                raise ValueError("Existing epoch snapshot is not the requested live-training export")
        elif self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError("Existing incomplete epoch snapshot retained: " + str(self.directory))

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = float(state.epoch or 0.0)
        if self.metadata is None and epoch > self.target_epoch + 1e-9:
            raise RuntimeError("The requested exact epoch snapshot boundary was not captured")
        if self.metadata is not None or not math.isclose(
                epoch, float(self.target_epoch), rel_tol=0.0, abs_tol=1e-9):
            return control
        error = None
        try:
            if self.trainer is None:
                raise ValueError("Epoch snapshot callback is not attached to Trainer")
            # Every rank participates: Trainer/ZeRO may gather the full model.
            self.trainer.save_model(str(self.directory))
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        errors = [error]
        if distributed:
            errors = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("Could not save exact epoch training weights: " +
                               "; ".join(value for value in errors if value))
        publication = [{"error": None, "metadata": None}]
        if state.is_world_process_zero:
            try:
                self.tokenizer.save_pretrained(self.directory)
                write_sid_provenance(self.directory, self.sid_index_path, self.item_meta_path,
                                     self.base_model, self.category)
                record = {"format_version": 1, "snapshot_type": "epoch_end_training_weights",
                          "capture_event": "on_epoch_end", "requested_epoch": self.target_epoch,
                          "epoch": epoch, "global_step": int(state.global_step),
                          "steps_per_epoch": int(state.global_step) / epoch,
                          "checkpoint_path": str(self.directory), "best_model_reloaded": False,
                          "created_at_utc": utc_now()}
                atomic_json(self.directory / "snapshot_metadata.json", record)
                publication[0]["metadata"] = record
            except Exception as exception:
                publication[0]["error"] = f"{type(exception).__name__}: {exception}"
        if distributed:
            torch.distributed.broadcast_object_list(publication, src=0)
        if publication[0]["error"] is not None:
            raise RuntimeError("Could not publish epoch snapshot metadata: " + publication[0]["error"])
        self.metadata = publication[0]["metadata"]
        return control


def verify_task_resume_configuration(resume_from_checkpoint, task_counts,
                                     include_title_history, early_stopping_patience,
                                     eval_save_fraction, generative_validation=False,
                                     generation_batch_size=4, generation_num_beams=50,
                                     save_epoch_checkpoint=0, minimum_epochs=0):
    """Prevent optimizer resumes from silently changing the task mixture."""
    if not resume_from_checkpoint:
        return
    previous_path = Path(resume_from_checkpoint).resolve().parent / "run_config.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("task_counts") != task_counts:
        raise ValueError("Resume training task counts differ from the original run")
    if previous.get("data", {}).get("include_title_history", False) != include_title_history:
        raise ValueError("Resume include_title_history differs from the original run")
    for name, value in (("early_stopping_patience", early_stopping_patience),
                        ("eval_save_fraction", eval_save_fraction)):
        if previous.get("hyperparameters", {}).get(name) != value:
            raise ValueError("Resume training configuration differs: " + name)
    previous_hyper = previous.get("hyperparameters", {})
    if previous_hyper.get("generative_validation", False) != generative_validation:
        raise ValueError("Resume generative_validation differs from the original run")
    for name, value in (("save_epoch_checkpoint", save_epoch_checkpoint),
                        ("minimum_epochs", minimum_epochs)):
        if previous_hyper.get(name, 0) != value:
            raise ValueError("Resume training configuration differs: " + name)
    if generative_validation:
        for name, value in (("generation_batch_size", generation_batch_size),
                            ("generation_num_beams", generation_num_beams),
                            ("model_selection_metric", "ndcg10")):
            if previous_hyper.get(name) != value:
                raise ValueError("Resume ranking configuration differs: " + name)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    lr_scheduler_type: str = "linear",
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="Industrial_and_Scientific",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    deepspeed: str = None,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "sdpa",
    monitoring: bool = True,
    monitor_window: int = 100,
    baseline_eval: bool = True,
    include_title_history: bool = False,
    early_stopping_patience: int = 3,
    eval_save_fraction: float = 0.05,
    generative_validation: bool = False,
    generation_batch_size: int = 4,
    generation_num_beams: int = 50,
    save_epoch_checkpoint: int = 0,
    minimum_epochs: float = 0,
):
    if lr_scheduler_type not in ("linear", "cosine") or category != "Industrial_and_Scientific":
        raise ValueError("Industrial_and_Scientific training supports linear or cosine scheduling")
    if not isinstance(include_title_history, bool):
        raise ValueError("include_title_history must be a boolean")
    if not isinstance(generative_validation, bool):
        raise ValueError("generative_validation must be a boolean")
    if generative_validation and not monitoring:
        raise ValueError("Ranking model selection requires monitoring and provenance")
    if type(generation_batch_size) is not int or generation_batch_size < 1 or generation_num_beams != 50:
        raise ValueError("Generation requires a positive batch size and beam50")
    if isinstance(early_stopping_patience, bool) or not isinstance(early_stopping_patience, int) or early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")
    if isinstance(eval_save_fraction, bool) or not isinstance(eval_save_fraction, (int, float)) or not 0 < eval_save_fraction <= 1:
        raise ValueError("eval_save_fraction must be a number in (0, 1]")
    if type(save_epoch_checkpoint) is not int or not 0 <= save_epoch_checkpoint <= num_epochs:
        raise ValueError("save_epoch_checkpoint must be an integer inside the training epoch budget")
    if (isinstance(minimum_epochs, bool) or not isinstance(minimum_epochs, (int, float))
            or not math.isfinite(minimum_epochs) or not 0 <= minimum_epochs <= num_epochs):
        raise ValueError("minimum_epochs must be finite and inside the training epoch budget")
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
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items"}
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
    if ddp and cuda_available:
        # Each process owns one complete model replica; never auto-shard it across ranks.
        model_load_kwargs["device_map"] = {"": local_rank}
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)
    else:
        config = AutoConfig.from_pretrained(base_model)
        config_kwargs = {key: value for key, value in model_load_kwargs.items() if key != "device_map"}
        model = AutoModelForCausalLM.from_config(config, **config_kwargs)
        if ddp and cuda_available:
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
        
    train_data, task_counts = build_training_tasks(
        train_file, item_meta_path, sid_index_path, tokenizer, cutoff_len,
        sample, seed, category, include_title_history=include_title_history)
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
    eval_step = eval_save_fraction
    monitor = SFTMonitorCallback(output_dir, window=monitor_window) if monitoring else None
    early_stopping = (MinimumEpochEarlyStoppingCallback(
        minimum_epochs, early_stopping_patience=early_stopping_patience) if minimum_epochs > 0
        else EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    callbacks = [early_stopping]
    if monitor is not None:
        callbacks.append(monitor)
    trainer_class = transformers.Trainer
    trainer_kwargs = {}
    recall_callback = None
    epoch_snapshot_callback = None
    if save_epoch_checkpoint:
        epoch_snapshot_callback = EpochSnapshotCallback(
            output_dir, save_epoch_checkpoint, tokenizer, sid_index_path,
            item_meta_path, base_model, category_key)
        callbacks.append(epoch_snapshot_callback)
    if generative_validation:
        from scripts.industrial_generative_validation import (
            GenerativeValidationConfig, make_generative_trainer_class,
        )
        ranking_config = GenerativeValidationConfig(
            valid_file=eval_file, sid_index_path=sid_index_path,
            expected_sid_sha256=file_sha256(sid_index_path),
            expected_valid_sha256=file_sha256(eval_file),
            batch_size=generation_batch_size, cutoff_len=cutoff_len,
            num_beams=generation_num_beams, category=category_key,
            expected_rows=len(hf_val_dataset),
        )
        trainer_class = make_generative_trainer_class(transformers.Trainer)
        trainer_kwargs = {"generative_validation": ranking_config, "processing_class": tokenizer}
        recall_callback = BestRecallCheckpointCallback(
            output_dir, tokenizer, sid_index_path, item_meta_path, base_model, category_key)
        callbacks.append(recall_callback)
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    trainer = trainer_class(
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
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="ndcg10" if generative_validation else "eval_loss",
            greater_is_better=generative_validation,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none",
        ),
        data_collator=collator,
        callbacks=callbacks,
        **trainer_kwargs,
        # optimizers=(optimizer, lr_scheduler) 
    )
    if epoch_snapshot_callback is not None:
        epoch_snapshot_callback.attach_trainer(trainer)
    if monitor is not None:
        monitor.attach_trainer(trainer)
        monitor.primary = trainer.is_world_process_zero()
        run_config = {
            "stage": "sft", "created_at_utc": utc_now(),
            "model": {"source": base_model, "train_from_scratch": train_from_scratch,
                      "original_vocab_size": original_vocab_size,
                      "tokenizer_vocab_size": len(tokenizer), "added_sid_tokens": len(new_tokens)},
            "data": {"train_file": train_file, "eval_file": eval_file,
                     "sid_index_path": sid_index_path, "item_meta_path": item_meta_path,
                     "category": category_key, "sample": sample,
                     "include_title_history": include_title_history,
                     "train_samples": len(hf_train_dataset), "validation_samples": len(hf_val_dataset)},
            "task_counts": task_counts,
            "hyperparameters": {"world_size": world_size, "micro_batch_size": micro_batch_size,
                                "gradient_accumulation_steps": gradient_accumulation_steps,
                                "global_batch_size": batch_size, "max_epochs": num_epochs,
                                "learning_rate": learning_rate, "warmup_steps": 20,
                                "lr_scheduler_type": str(trainer.args.lr_scheduler_type).split(".")[-1].lower(), "cutoff_len": cutoff_len,
                                "max_grad_norm": 1.0, "bf16": True,
                                "gradient_checkpointing": gradient_checkpointing,
                                "attn_implementation": attn_implementation,
                                "deepspeed": deepspeed, "seed": seed,
                                "early_stopping_patience": early_stopping_patience, "eval_save_fraction": eval_step,
                                "generative_validation": generative_validation,
                                "generation_batch_size": generation_batch_size,
                                "generation_num_beams": generation_num_beams,
                                "model_selection_metric": "ndcg10" if generative_validation else "eval_loss",
                                "save_epoch_checkpoint": save_epoch_checkpoint,
                                "minimum_epochs": minimum_epochs,
                                "monitor_window": monitor_window},
            "resume_from_checkpoint": resume_from_checkpoint,
        }
        verify_task_resume_configuration(
            resume_from_checkpoint, task_counts, include_title_history,
            early_stopping_patience, eval_save_fraction, generative_validation,
            generation_batch_size, generation_num_beams, save_epoch_checkpoint,
            minimum_epochs)
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
        if (epoch_snapshot_callback is not None and epoch_snapshot_callback.metadata is None
                and float(trainer.state.epoch or 0.0) + 1e-9 >= save_epoch_checkpoint):
            raise RuntimeError("Training returned without the requested exact epoch snapshot")
        if monitor is not None and monitor.abort_reason:
            raise RuntimeError("SFT monitoring stopped training: " + monitor.abort_reason)
        if monitor is not None:
            monitor.phase = "final_validation"
        if recall_callback is not None:
            # Trainer has reloaded the NDCG-best weights. A repeated final
            # evaluation must not label these as the last training checkpoint.
            recall_callback.active = False
        final_eval_metrics = trainer.evaluate() if monitoring else None
        if monitor is not None and monitor.abort_reason:
            raise RuntimeError("SFT monitoring stopped validation: " + monitor.abort_reason)
        trainer.save_model(output_dir)

        # All ranks must enter Trainer saving so distributed state gathering remains safe.
        trainer.save_model(final_output_dir)
        best_ndcg_output_dir = os.path.join(output_dir, "best_ndcg_checkpoint") if generative_validation else None
        if generative_validation:
            trainer.save_model(best_ndcg_output_dir)
        if monitoring:
            trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_output_dir)
            write_sid_provenance(
                final_output_dir, sid_index_path, item_meta_path, base_model, category_key
            )
            if generative_validation:
                tokenizer.save_pretrained(best_ndcg_output_dir)
                write_sid_provenance(
                    best_ndcg_output_dir, sid_index_path, item_meta_path, base_model, category_key)
                if recall_callback.best_recall is None or recall_callback.best_ndcg is None:
                    raise ValueError("Ranking training finished without validation-selected checkpoints")
                atomic_json(Path(best_ndcg_output_dir) / "selection_metrics.json", recall_callback.best_ndcg)
                atomic_json(Path(output_dir) / "ranking_selection.json", {
                    "scope": "complete_original_validation_split", "primary_metric": "ndcg10",
                    "early_stopping_metric": "ndcg10", "best_ndcg": recall_callback.best_ndcg,
                    "best_recall": recall_callback.best_recall,
                    "best_ndcg_checkpoint": best_ndcg_output_dir,
                    "best_recall_checkpoint": str(Path(output_dir) / "best_recall_checkpoint"),
                    "final_checkpoint": final_output_dir,
                })
            if monitoring:
                state = trainer.state
                atomic_json(os.path.join(output_dir, "training_summary.json"), {
                    "status": "completed", "stage": "sft", "finished_at_utc": utc_now(),
                    "train_metrics": getattr(result, "metrics", {}),
                    "final_eval_metrics": final_eval_metrics,
                    "best_checkpoint": state.best_model_checkpoint, "best_metric": state.best_metric,
                    "selection_metric": "ndcg10" if generative_validation else "eval_loss",
                    "validation_loss_at_selected_checkpoint": None if final_eval_metrics is None else final_eval_metrics.get("eval_loss"),
                    "minimum_observed_validation_loss": min(
                        (row["eval_loss"] for row in state.log_history if "eval_loss" in row), default=None),
                    "best_ndcg_checkpoint": best_ndcg_output_dir,
                    "best_recall_checkpoint": str(Path(output_dir) / "best_recall_checkpoint") if generative_validation else None,
                    "selected_validation_metrics": None if recall_callback is None else recall_callback.best_ndcg["metrics"],
                    "best_recall_validation_metrics": None if recall_callback is None else recall_callback.best_recall["metrics"],
                    "epoch_snapshot_checkpoint": None if epoch_snapshot_callback is None or epoch_snapshot_callback.metadata is None else str(epoch_snapshot_callback.directory),
                    "epoch_snapshot_metadata": None if epoch_snapshot_callback is None else epoch_snapshot_callback.metadata,
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
