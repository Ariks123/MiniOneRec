#!/usr/bin/env python3
"""Offline four-GPU ZeRO-2 smoke for Industrial validation ranking and dual best exports.

All models and samples are synthetic. No production split or pretrained weights
are opened. Launch with torchrun --standalone --nproc_per_node=4.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_ACCELERATOR"] = "cuda"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(work):
    """Create sixty distinct SIDs and one alias, including uneven validation shards."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    import torch

    base, inputs, prepared = (work / name for name in ("base", "inputs", "prepared"))
    for folder in (base, inputs, prepared):
        folder.mkdir(parents=True)
    catalog = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(60)}
    catalog["60"] = catalog["0"][:]
    sid_tokens = sorted({token for tokens in catalog.values() for token in tokens})
    words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "recommend", "item", *sid_tokens]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    backend.add_tokens([AddedToken(token, normalized=False) for token in sid_tokens])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[EOS]",
        bos_token="[BOS]", eos_token="[EOS]", padding_side="left", model_max_length=512,
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512, attention_dropout=0.0,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=True, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(42)
    Qwen2ForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    sid_path = inputs / "Industrial_and_Scientific.index.json"
    metadata_path = inputs / "Industrial_and_Scientific.item.json"
    sid_path.write_text(json.dumps(catalog) + "\n")
    metadata_path.write_text(json.dumps({key: {"title": "item " + key, "description": "synthetic"}
                                         for key in catalog}) + "\n")
    outputs = {}
    for split, count in (("train", 32), ("valid", 7)):
        path = prepared / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("user_id", "history_item_sid", "item_sid"))
            writer.writeheader()
            for row in range(count):
                item = "60" if split == "valid" and row == count - 1 else str(row % 60)
                writer.writerow({"user_id": f"U{row}", "history_item_sid": repr(["".join(catalog["0"])]),
                                 "item_sid": "".join(catalog[item])})
        outputs[path.name] = {"sha256": sha256(path)}
    manifest = {"dataset": "Industrial_and_Scientific", "validation": "passed", "original_sid": True,
                "rows": {"train": 32, "valid": 7}, "outputs": outputs,
                "sources": {"sid_index": {"sha256": sha256(sid_path)}}}
    (prepared / "preparation_manifest.json").write_text(json.dumps(manifest) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Trainer, TrainingArguments)
    from data import SidSFTDataset
    from sft import BestRecallCheckpointCallback, write_sid_provenance
    from scripts.industrial_generative_validation import (
        GenerativeValidationConfig, generate_validation, make_generative_trainer_class,
    )
    from scripts.sft_eval_utils import verify_sid_tokenizer

    rank, local_rank, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4, "This smoke requires exactly four GPU ranks")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    work = args.output_dir.resolve()
    if rank == 0:
        require(not work.exists() or not any(work.iterdir()), "Choose a fresh smoke output directory")
        work.mkdir(parents=True, exist_ok=True)
        make_fixture(work)
    dist.barrier(device_ids=[local_rank])
    base, inputs, prepared, run = (work / name for name in ("base", "inputs", "prepared", "run"))
    sid_path, metadata_path = (inputs / ("Industrial_and_Scientific" + suffix) for suffix in (".index.json", ".item.json"))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, attn_implementation="sdpa",
    )
    train = SidSFTDataset(str(prepared / "train.csv"), tokenizer, max_len=512, sample=-1,
                          seed=42, category="industrial and scientific items")
    valid = SidSFTDataset(str(prepared / "valid.csv"), tokenizer, max_len=512, sample=-1,
                          seed=None, category="industrial and scientific items")
    callback = BestRecallCheckpointCallback(str(run), tokenizer, str(sid_path), str(metadata_path),
                                           str(base), "Industrial_and_Scientific")
    trainer = make_generative_trainer_class(Trainer)(
        model=model, train_dataset=train, eval_dataset=valid,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(run), deepspeed=str(ROOT / "config/h20_zero2.json"), bf16=True,
            per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=6, learning_rate=0.0, warmup_steps=0, lr_scheduler_type="cosine",
            eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="ndcg10",
            greater_is_better=True, optim="adamw_torch", logging_steps=1,
            report_to="none", disable_tqdm=True, ddp_find_unused_parameters=False,
            dataloader_num_workers=0, seed=42,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), callback],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
        generative_validation=GenerativeValidationConfig(
            valid_file=str(prepared / "valid.csv"), sid_index_path=str(sid_path),
            expected_sid_sha256=sha256(sid_path), expected_valid_sha256=sha256(prepared / "valid.csv"),
            expected_rows=7, num_beams=50, batch_size=2, cutoff_len=512,
        ),
    )
    result = trainer.train()
    require(trainer.is_deepspeed_enabled and trainer.model_wrapped.zero_optimization_stage() == 2,
            "Smoke did not exercise real DeepSpeed ZeRO-2")
    require(trainer.state.global_step == 4, "Constant-ranking early stopping should stop at step four")
    require(Path(trainer.state.best_model_checkpoint).name == "checkpoint-1", "NDCG best is not first checkpoint")
    require(callback.best_ndcg["step"] == callback.best_recall["step"] == 1,
            "Constant-ranking independent best records should retain step one")
    evaluations = [entry for entry in trainer.state.log_history if "eval_ndcg10" in entry]
    require(len(evaluations) == 4 and all(entry["eval_generation_rows"] == 7 for entry in evaluations),
            "Each training-time validation must cover all seven original rows")
    require(all(entry["eval_ndcg10"] == evaluations[0]["eval_ndcg10"]
                and entry["eval_recall10"] == evaluations[0]["eval_recall10"] for entry in evaluations),
            "Zero-LR validation metrics unexpectedly changed")

    # Exercise restoration directly without invoking selection again after best reload.
    callback.active = False
    bare = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    bare.train(True)
    bare.config.use_cache = False
    tokenizer.padding_side = "right"
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(local_rank).clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    validation = generate_validation(trainer)
    rng_unchanged = (torch.equal(cpu_rng, torch.get_rng_state())
                     and torch.equal(cuda_rng, torch.cuda.get_rng_state(local_rank))
                     and python_rng == random.getstate()
                     and numpy_rng[0] == np.random.get_state()[0]
                     and np.array_equal(numpy_rng[1], np.random.get_state()[1])
                     and numpy_rng[2:] == np.random.get_state()[2:])
    restored = bare.training and bare.config.use_cache is False and tokenizer.padding_side == "right"
    flags = torch.tensor([int(rng_unchanged), int(restored)], dtype=torch.int64, device=trainer.accelerator.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    require(flags.tolist() == [1, 1], "Generation changed RNG, model mode/cache, or tokenizer padding")
    require(validation["coverage"]["completed_rows"] == 7
            and validation["catalog_items"] == 61 and validation["unique_sids"] == 60,
            "Collision-aware uneven validation coverage was not retained")
    require(validation["generation"]["num_beams"] == 50
            and all(f"{metric}{k}" in validation["metrics"]
                    for metric in ("recall", "ndcg") for k in (1, 3, 5, 10, 20, 50)),
            "Full beam50 Recall/NDCG metrics were not computed")
    tokenizer.padding_side = "left"
    final, ndcg_export = run / "final_checkpoint", run / "best_ndcg_checkpoint"
    trainer.save_model(str(final))
    trainer.save_model(str(ndcg_export))
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        for path in (final, ndcg_export):
            tokenizer.save_pretrained(path)
            write_sid_provenance(path, str(sid_path), str(metadata_path), str(base), "Industrial_and_Scientific")
        paths = [Path(trainer.state.best_model_checkpoint), final, ndcg_export, run / "best_recall_checkpoint"]
        reference = None
        for path in paths:
            loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
            state = loaded.state_dict()
            if reference is None:
                reference = {key: value.clone() for key, value in state.items()}
            require(all(torch.equal(reference[key], value) for key, value in state.items()),
                    "Reloaded best exports differ from retained best checkpoint")
            exported_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            verify_sid_tokenizer(exported_tokenizer, loaded, json.loads(sid_path.read_text()))
            if path != paths[0]:
                provenance = json.loads((path / "posttraining_provenance.json").read_text())
                require(provenance["sid_index_sha256"] == sha256(sid_path), "Exported SID provenance changed")
            del loaded
        summary = {"status": "passed", "world_size": world, "zero_stage": 2, "stopped_step": result.global_step,
                   "best_ndcg_step": callback.best_ndcg["step"], "best_recall_step": callback.best_recall["step"],
                   "train_validation_events": len(evaluations), "validation_rows": 7, "catalog_items": 61,
                   "unique_sids": 60, "num_beams": 50, "all_k_metrics": validation["metrics"],
                   "uneven_rank_coverage": "passed", "rng_restoration": "passed", "mode_restoration": "passed",
                   "checkpoint_reload": "passed", "no_test_split_opened": not (prepared / "test.csv").exists(),
                   "work_dir": str(work)}
        (work / "smoke_result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
