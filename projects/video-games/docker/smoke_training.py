"""Offline CPU checks of the real pinned SFT and recommendation-RL frameworks.

Uses a random, one-layer Qwen2 model and an in-memory tokenizer. No pretrained
weights, production data, GPU, distributed process or network access is needed.
Run inside the H20 image: python docker/smoke_training.py
"""
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


# Set these before importing any ML libraries, including the project modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "true"
os.environ["DS_ACCELERATOR"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    import torch
    from datasets import Dataset
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        DataCollatorForSeq2Seq,
        LogitsProcessorList,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
        Trainer,
        TrainingArguments,
    )
    from trl import GRPOConfig

    from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
    from minionerec_trainer import ReReTrainer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    sid_tokens = ["<a_0>", "<b_0>", "<c_0>", "<d_0>", "<d_1>"]
    words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "recommend", "game"] + sid_tokens
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    backend.add_tokens([AddedToken(token, normalized=False) for token in sid_tokens])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        padding_side="left",
        model_max_length=64,
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    catalog = ["<a_0><b_0><c_0><d_0>", "<a_0><b_0><c_0><d_1>"]
    answers = [tokenizer.encode(sid, add_special_tokens=False) for sid in catalog]
    require(all(len(ids) == 4 for ids in answers), "Smoke tokenizer did not preserve atomic SID tokens")
    prompt = "recommend game"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).float().cpu()
    report = {"device": "cpu", "model": "random-tiny-qwen2", "parameters": model.num_parameters(),
              "downloads": False, "gpu_training_tested": False,
              "distributed_tested": False, "deepspeed_training_tested": False}

    with tempfile.TemporaryDirectory(prefix="minionerec-smoke-") as temporary:
        directory = Path(temporary)
        rows = []
        for answer in answers:
            sequence = prompt_ids + answer + [tokenizer.eos_token_id]
            rows.append({"input_ids": sequence, "attention_mask": [1] * len(sequence),
                         "labels": [-100] * len(prompt_ids) + sequence[len(prompt_ids):]})
        sft_args = TrainingArguments(
            output_dir=str(directory / "sft"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=1,
            learning_rate=1e-3,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )
        sft_trainer = Trainer(
            model=model,
            args=sft_args,
            train_dataset=Dataset.from_list(rows),
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
            processing_class=tokenizer,
        )
        before = model.get_input_embeddings().weight.detach().clone()
        with contextlib.redirect_stdout(sys.stderr):
            sft_result = sft_trainer.train()
        require(sft_result.global_step == 1 and math.isfinite(sft_result.training_loss),
                "SFT failed to produce one finite training step")
        require(not torch.equal(before, model.get_input_embeddings().weight.detach()),
                "SFT optimizer did not update model weights")
        report["sft"] = {"status": "passed", "optimizer_steps": 1, "loss": sft_result.training_loss,
                         "gradient_checkpointing": True, "attention": "sdpa"}
        model.gradient_checkpointing_disable()
        model.eval()

        # Exercise the actual catalog processor, including sparse initial prefixes
        # and two unique deterministic beams, before the full RL trainer step.
        prefixes = build_prefix_allowed_tokens(tokenizer, [sid + "\n" for sid in catalog])
        processor = ConstrainedLogitsProcessor(
            lambda batch_id, tokens: prefixes.get(tuple(tokens), []),
            num_beams=2,
            eos_token_id=tokenizer.eos_token_id,
            prompt_length=len(prompt_ids),
        )
        with torch.no_grad():
            generated = model.generate(
                input_ids=torch.tensor([prompt_ids]),
                attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long),
                max_new_tokens=5,
                num_beams=2,
                num_return_sequences=2,
                do_sample=False,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, len(prompt_ids):].tolist()
        expected = {tuple(answer + [tokenizer.eos_token_id]) for answer in answers}
        require({tuple(ids) for ids in completion_ids} == expected,
                f"Constrained generation returned invalid or duplicate catalog items: {completion_ids}")
        report["constrained_generation"] = {"status": "passed", "unique_valid_items": 2}

        info_file = directory / "info.txt"
        info_file.write_text("".join(f"{sid}\tSmoke item {i}\t{i}\n" for i, sid in enumerate(catalog)),
                             encoding="utf-8")
        seen_completions = []

        def smoke_reward(prompts, completions, **kwargs):
            seen_completions.extend(completions)
            return [1.0 if completion.strip() == catalog[0] else 0.0 for completion in completions]

        rl_args = GRPOConfig(
            output_dir=str(directory / "rl"),
            use_cpu=True,
            bf16=False,
            fp16=False,
            use_vllm=False,
            model_init_kwargs=None,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            max_prompt_length=16,
            max_completion_length=5,
            num_generations=2,
            temperature=1.0,
            beta=0.04,
            sync_ref_model=True,
            learning_rate=1e-3,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            log_completions=False,
            disable_tqdm=True,
        )
        with contextlib.redirect_stdout(sys.stderr):
            rl_trainer = ReReTrainer(
                model=model,
                base_model="random-tiny-qwen2",
                reward_funcs=smoke_reward,
                args=rl_args,
                beam_search=True,
                test_during_training=False,
                test_beam=2,
                info_file=str(info_file),
                processing_class=tokenizer,
                train_dataset=Dataset.from_list([{"prompt": prompt}]),
            )
            before = model.get_input_embeddings().weight.detach().clone()
            rl_result = rl_trainer.train()
        require(rl_result.global_step == 1 and math.isfinite(rl_result.training_loss),
                "ReReTrainer failed to produce one finite rollout/loss/backward/optimizer step")
        require(not torch.equal(before, model.get_input_embeddings().weight.detach()),
                "RL optimizer did not update model weights")
        require(len(seen_completions) == 2 and {item.strip() for item in seen_completions} == set(catalog),
                f"RL rollout did not contain the two valid SID candidates: {seen_completions}")
        report["rl"] = {"status": "passed", "trainer": "ReReTrainer", "optimizer_steps": 1,
                        "rollout_completions": len(seen_completions), "loss": rl_result.training_loss,
                        "reference_model": True, "reward_normalization": True}

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
