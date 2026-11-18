"""Offline four-H20 NCCL/BF16 and real ZeRO-2 SFT/RL training smoke.

Run with torchrun --standalone --nproc-per-node=4 docker/smoke_distributed.py.
The default RL optimizer matches production: paged_adamw_32bit. All weights,
tokenizers and examples are synthetic; no pretrained model or production data
is read, downloaded or saved. Checkpoint saving and experiment reporting are off.
"""
import argparse
import contextlib
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_ACCELERATOR"] = "cuda"
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tiny_tokenizer():
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    sid_tokens = ["<a_0>", "<b_0>", "<c_0>", "<d_0>", "<d_1>"]
    words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "recommend", "game"] + sid_tokens
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    backend.add_tokens([AddedToken(token, normalized=False) for token in sid_tokens])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]", padding_side="left", model_max_length=64,
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    catalog = ["<a_0><b_0><c_0><d_0>", "<a_0><b_0><c_0><d_1>"]
    require(all(len(tokenizer.encode(item, add_special_tokens=False)) == 4 for item in catalog),
            "Smoke tokenizer must preserve atomic SID tokens")
    return tokenizer, catalog


def tiny_model(tokenizer, device):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def common_args(output_dir, ds_config):
    return dict(
        output_dir=str(output_dir), deepspeed=str(ds_config), bf16=True, fp16=False,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_steps=1,
        learning_rate=1e-3, warmup_steps=0, eval_strategy="no", save_strategy="no",
        logging_strategy="no", report_to="none", disable_tqdm=True,
        ddp_find_unused_parameters=False, dataloader_num_workers=0,
    )


def check_update(model, before, result, stage):
    import torch

    require(result.global_step == 1 and math.isfinite(result.training_loss),
            f"{stage}: failed to finish one finite training step")
    require(not torch.equal(before, model.get_input_embeddings().weight.detach()),
            f"{stage}: optimizer did not update model weights")


def sft_step(directory, ds_config, tokenizer, catalog, device):
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model = tiny_model(tokenizer, device)
    prompt = tokenizer.encode("recommend game", add_special_tokens=False)
    rows = []
    for sid in catalog:
        answer = tokenizer.encode(sid, add_special_tokens=False) + [tokenizer.eos_token_id]
        sequence = prompt + answer
        rows.append({"input_ids": sequence, "attention_mask": [1] * len(sequence),
                     "labels": [-100] * len(prompt) + answer})
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**common_args(directory / "sft", ds_config), optim="adamw_torch"),
        train_dataset=Dataset.from_list(rows * 8),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
    before = model.get_input_embeddings().weight.detach().clone()
    result = trainer.train()
    require(trainer.is_deepspeed_enabled, "SFT did not use DeepSpeed")
    require(trainer.model_wrapped.zero_optimization_stage() == 2, "SFT did not use ZeRO-2")
    check_update(model, before, result, "SFT")
    report = {"status": "passed", "optimizer": "adamw_torch", "optimizer_steps": 1,
              "gradient_accumulation_steps": 2, "global_samples_per_step": 16,
              "zero_stage": 2, "loss": result.training_loss}
    del trainer, model, before
    return report


def rl_step(directory, ds_config, tokenizer, catalog, device, optimizer):
    from datasets import Dataset
    from trl import GRPOConfig
    from minionerec_trainer import ReReTrainer

    info_file = directory / "info.txt"
    info_file.write_text("".join(f"{sid}\tSmoke item {i}\t{i}\n" for i, sid in enumerate(catalog)),
                         encoding="utf-8")
    seen_completions = []

    def smoke_reward(prompts, completions, **kwargs):
        seen_completions.extend(completions)
        return [1.0 if completion.strip() == catalog[0] else 0.0 for completion in completions]

    model = tiny_model(tokenizer, device)
    args = GRPOConfig(
        **common_args(directory / "rl", ds_config),
        optim=optimizer, use_vllm=False, model_init_kwargs=None,
        max_prompt_length=16, max_completion_length=5, num_generations=2,
        temperature=1.0, beta=0.04, sync_ref_model=True, log_completions=False,
    )
    trainer = ReReTrainer(
        model=model, base_model="random-tiny-qwen2", reward_funcs=smoke_reward,
        args=args, beam_search=True, test_during_training=False, test_beam=2,
        info_file=str(info_file), processing_class=tokenizer,
        train_dataset=Dataset.from_list([{"prompt": "recommend game"} for _ in range(8)]),
    )
    before = model.get_input_embeddings().weight.detach().clone()
    result = trainer.train()
    require(trainer.is_deepspeed_enabled, "RL did not use DeepSpeed")
    require(trainer.model_wrapped.zero_optimization_stage() == 2, "RL did not use ZeRO-2")
    check_update(model, before, result, "RL")
    observed = [item.strip() for item in seen_completions]
    require(len(observed) == 4 and all(set(observed[i:i + 2]) == set(catalog) for i in (0, 2)),
            f"Each RL microstep must return the two valid SID candidates: {seen_completions}")
    report = {"status": "passed", "trainer": "ReReTrainer", "optimizer": optimizer,
              "optimizer_steps": 1, "gradient_accumulation_steps": 2,
              "global_completions_per_step": 16, "zero_stage": 2, "loss": result.training_loss,
              "valid_rollout_completions_per_rank": len(seen_completions), "reference_model": True}
    del trainer, model, before
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "communication", "sft", "rl"), default="all")
    parser.add_argument("--rl-optimizer", choices=("paged_adamw_32bit", "adamw_torch"),
                        default="paged_adamw_32bit")
    parser.add_argument("--deepspeed-config", type=Path, default=ROOT / "config/h20_zero2.json")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    require(int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "Launch with torchrun --standalone --nproc-per-node=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4,
            "Exactly four visible CUDA GPUs are required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_name(local_rank)
    require("H20" in name.upper(), f"Expected H20; rank {local_rank} found {name}")
    require(torch.cuda.is_bf16_supported(), f"Rank {local_rank} does not support BF16")
    require(dist.is_nccl_available(), "This PyTorch build does not provide NCCL")
    config = json.loads(args.deepspeed_config.read_text(encoding="utf-8"))
    require(config.get("zero_optimization", {}).get("stage") == 2
            and config.get("bf16", {}).get("enabled") is True,
            "Smoke requires the actual BF16 ZeRO-2 configuration")

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=180))
    tensor = torch.tensor([local_rank + 1], dtype=torch.bfloat16, device=device)
    dist.all_reduce(tensor)
    require(tensor.item() == 10, f"Rank {local_rank} BF16 NCCL all-reduce returned {tensor.item()}, expected 10")
    report = {"device": "four-h20", "world_size": dist.get_world_size(), "gpu_name": name,
              "cuda_build": torch.version.cuda, "nccl_bf16_all_reduce": "passed",
              "per_device_micro_batch_size": 2, "gradient_accumulation_steps": 2,
              "deepspeed_config": str(args.deepspeed_config), "synthetic_weights_only": True,
              "production_data_used": False, "downloads": False, "checkpoints_written": False}

    with tempfile.TemporaryDirectory(prefix=f"minionerec-smoke-rank{local_rank}-") as temporary:
        directory = Path(temporary)
        tokenizer, catalog = tiny_tokenizer()
        with contextlib.redirect_stdout(sys.stderr):
            if args.stage in {"all", "sft"}:
                report["sft"] = sft_step(directory, args.deepspeed_config, tokenizer, catalog, device)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
            if args.stage in {"all", "rl"}:
                report["rl"] = rl_step(directory, args.deepspeed_config, tokenizer, catalog,
                                       device, args.rl_optimizer)

    # Reaching this collective confirms every rank completed its local assertions.
    completed = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(completed)
    require(completed.item() == 4, "Not all four ranks finished")
    report["all_ranks_passed"] = True
    if dist.get_rank() == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
