#!/usr/bin/env python3
"""Four-H20 smoke of row-local GRPO, ranking selection and ZeRO-2 exports.

All weights and data are synthetic. Real beam50 validation is run; only the
NDCG@10 selection score is scripted to make best reload/early-stop assertions
independent of incidental ordering changes in the tiny random model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
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


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, LogitsProcessorList, TrainerCallback
    from trl import GRPOConfig
    from docker.smoke_industrial_rank_selection import make_fixture, sha256
    from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
    from minionerec_trainer import ReReTrainer
    from scripts.industrial_generative_validation import GenerativeValidationConfig
    from scripts.industrial_rl_data import rule_reward, make_ndcg_rule_reward
    from scripts.deepspeed_zero2_compat import apply_zero2_accumulation_fix
    import scripts.industrial_rl_validation as validation

    compatibility = apply_zero2_accumulation_fix()

    rank, local_rank, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4, "The RL smoke requires four GPU ranks")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    work = args.output_dir.resolve()
    if rank == 0:
        require(not work.exists() or not any(work.iterdir()), "Choose a new smoke directory")
        work.mkdir(parents=True, exist_ok=True)
        make_fixture(work)
        catalog = json.loads((work / "inputs/Industrial_and_Scientific.index.json").read_text())
        (work / "info.txt").write_text("".join("".join(tokens) + f"\titem {item}\t{item}\n"
                                                   for item, tokens in catalog.items()))
    dist.barrier(device_ids=[local_rank])
    base, prepared = work / "base", work / "prepared"
    sid = work / "inputs/Industrial_and_Scientific.index.json"
    catalog = json.loads(sid.read_text())
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
                                                device_map={"": local_rank}, attn_implementation="sdpa")
    prefixes = build_prefix_allowed_tokens(tokenizer, ["".join(tokens) + "\n" for tokens in catalog.values()])
    batch = tokenizer("recommend item", return_tensors="pt").to(model.device)
    width = batch["input_ids"].shape[1]
    processor = ConstrainedLogitsProcessor(lambda _, tokens: prefixes.get(tuple(tokens), []),
                                         num_beams=16, eos_token_id=tokenizer.eos_token_id, prompt_length=width)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**batch, num_beams=16, num_return_sequences=16, do_sample=False,
                                 max_new_tokens=4, length_penalty=0.0, use_cache=True,
                                 logits_processor=LogitsProcessorList([processor]),
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    answers = [text.strip() for text in tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)]
    require(len(set(answers)) == 16, "The synthetic preflight must produce sixteen unique valid SIDs")
    rows = [{"prompt": "recommend item", "completion": answers[index % 2] + "\n",
             "target_sid": answers[index % 2], "sample_id": f"train:{index}",
             "task_type": "sid_sequence_recommendation" if index % 2 else "title2sid"} for index in range(32)]
    original = model.get_input_embeddings().weight.detach().clone()
    run = work / "run"
    monitor = validation.RLMonitorCallback(str(run))
    last_weights = {}

    class CaptureLast(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:
                last_weights["weight"] = trainer.accelerator.unwrap_model(trainer.model_wrapped).get_input_embeddings().weight.detach().clone()
            return control

    trainer = validation.make_rl_trainer_class(ReReTrainer)(
        model=model, base_model=str(base), processing_class=tokenizer,
        reward_funcs=[rule_reward, make_ndcg_rule_reward(16)], beam_search=True,
        test_during_training=False, info_file=str(work / "info.txt"),
        train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows[:7]),
        generative_validation=GenerativeValidationConfig(
            valid_file=str(prepared / "valid.csv"), sid_index_path=str(sid),
            expected_sid_sha256=sha256(sid), expected_valid_sha256=sha256(prepared / "valid.csv"),
            expected_rows=7, num_beams=50, batch_size=2, cutoff_len=512),
        args=GRPOConfig(
            output_dir=str(run), deepspeed=str(ROOT / "config/h20_zero2.json"),
            bf16=True, fp16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=2,
            num_generations=16, use_vllm=False, model_init_kwargs=None,
            max_prompt_length=128, max_completion_length=4, temperature=1.0,
            beta=.001, sync_ref_model=True, ref_model_sync_steps=1, ref_model_mixup_alpha=.6,
            max_steps=5, learning_rate=1e-3, warmup_steps=0, lr_scheduler_type="cosine",
            optim="paged_adamw_32bit", max_grad_norm=.3, remove_unused_columns=False,
            eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="ndcg10",
            greater_is_better=True, logging_steps=1, logging_nan_inf_filter=False,
            report_to="none", disable_tqdm=True, ddp_find_unused_parameters=False,
            dataloader_num_workers=0, seed=42),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), monitor, CaptureLast(),
                   validation.terminal_eval_callback(monitor)],
    )
    monitor.attach_trainer(trainer)
    monitor.primary = trainer.is_world_process_zero()
    reference_before = next(trainer.ref_model.parameters()).detach().clone()
    real_generate = validation.generate_validation
    baseline = validation.run_initial_baseline(trainer)
    require(trainer.state.best_model_checkpoint is None, "Baseline unexpectedly selected an RL checkpoint")

    def scored_generation(current):
        report = real_generate(current)
        report["smoke_actual_ndcg10"] = report["metrics"]["ndcg10"]
        report["metrics"]["ndcg10"] = {1: .9, 2: .7, 3: .6}[current.state.global_step]
        report["smoke_scripted_selection_score"] = True
        return report

    validation.generate_validation = scored_generation
    result = trainer.train()
    require(result.global_step == 3, "Patience two must stop at the third scripted ranking check")
    require(Path(trainer.state.best_model_checkpoint).name == "checkpoint-1", "Validation best checkpoint was not selected")
    bare = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    require(not torch.equal(original, bare.get_input_embeddings().weight), "RL did not update policy weights")
    require(not torch.equal(last_weights["weight"], bare.get_input_embeddings().weight), "Best weights were not reloaded")
    require(not torch.equal(reference_before, next(trainer.ref_model.parameters())), "Reference synchronization did not update weights")
    eval_logs = [entry for entry in trainer.state.log_history if "eval_ndcg10" in entry]
    require(len(eval_logs) == 3 and all("eval_loss" not in entry and "eval_reward" not in entry for entry in eval_logs),
            "Ranking validation was contaminated with GRPO loss or training rewards")
    train_logs = [entry for entry in trainer.state.log_history if "reward" in entry]
    require(train_logs and any(entry.get("hit_group_rate", 0) > 0 for entry in train_logs), "No successful reward groups observed")
    require(any(entry.get("advantage_nonzero_group_rate", 0) > 0 for entry in train_logs), "No nonzero grouped advantages observed")
    validation.generate_validation = real_generate
    bare.train(True)
    bare.config.use_cache = False
    tokenizer.padding_side = "right"
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(local_rank).clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    generated = real_generate(trainer)
    require(bare.training and bare.config.use_cache is False and tokenizer.padding_side == "right", "Validation changed training mode/cache/padding")
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(local_rank))
            and python_rng == random.getstate() and np.array_equal(numpy_rng[1], np.random.get_state()[1]),
            "Validation changed the training random stream")
    require(generated["coverage"]["completed_rows"] == 7, "Uneven validation shards were not fully covered")
    tokenizer.padding_side = "left"
    final = run / "final_checkpoint"
    trainer.save_model(str(final))
    if rank == 0:
        tokenizer.save_pretrained(final)
    dist.barrier(device_ids=[local_rank])
    restored = AutoModelForCausalLM.from_pretrained(final, local_files_only=True, torch_dtype=torch.bfloat16)
    require(torch.equal(restored.get_input_embeddings().weight, bare.get_input_embeddings().weight.detach().cpu()), "Export reload differs from best policy")
    require(len(list(run.glob("checkpoint-*"))) <= 2, "Checkpoint rotation exceeded its two-checkpoint limit")
    monitor.finish(trainer.state, "completed")
    report = {"status": "passed", "world_size": world, "zero_stage": trainer.model_wrapped.zero_optimization_stage(),
              "runtime_compatibility": compatibility,
              "optimizer": "paged_adamw_32bit", "num_generations": 16, "synthetic_data_and_weights": True,
              "scripted_selection_scores_only": True, "optimizer_steps": result.global_step,
              "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
              "row_local_labels": True, "duplicate_prompts_with_distinct_targets": True,
              "policy_updated": True, "reference_updated": True, "best_reloaded": True,
              "ranking_only_validation": True, "complete_validation_rows": 7,
              "rng_and_training_state_restored": True, "export_reload": True,
              "baseline_ndcg10": baseline["metrics"]["ndcg10"]}
    if rank == 0:
        (work / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
