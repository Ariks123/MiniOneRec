"""Synthetic CPU + four-GPU retrieval/ZeRO-2 correctness smoke, never real training."""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path


def fixture(path):
    import torch
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    words = ["[UNK]", "[EOS]", "[BOS]", "history", "answer", "Item", "SID:", "###", "Embedding:"]
    words += [f"<{letter}_{number}>" for letter in "abc" for number in range(6)]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_tokens([AddedToken(word, normalized=False) for word in words if word.startswith('<')])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
        pad_token="[EOS]", eos_token="[EOS]", bos_token="[BOS]", padding_side="left")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0., tie_word_embeddings=True, use_cache=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, max_position_embeddings=512)
    config._attn_implementation = "sdpa"
    config.retrieval_dim = 8
    config.retrieval_weight = .1
    config.retrieval_temperature = .07
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    Qwen2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return tokenizer, config


def sample(tokenizer, index=0, sequence=True, long_answer=False):
    sid = f"<a_{index % 6}><b_0><c_0>"
    prefix = tokenizer.encode("history " * (1 + index % 3) + "answer", add_special_tokens=False)
    answer = tokenizer.encode(sid, add_special_tokens=False)
    if long_answer:
        answer = answer + tokenizer.encode("history history history history", add_special_tokens=False)
    answer += [tokenizer.eos_token_id]
    return {"input_ids": prefix + answer, "attention_mask": [1] * (len(prefix) + len(answer)),
        "labels": [-100] * len(prefix) + answer, "task_type": "sid_sequence_recommendation" if sequence else "sid2title",
        "target_item_id": str(index), "target_sid": sid, "retrieval_sid_id": index % 6 if sequence else -1}


def cpu_checks(work):
    import torch
    from transformers import Qwen2ForCausalLM, Trainer, TrainingArguments
    from qwen_retrieval.model import QwenRetrievalForCausalLM
    from qwen_retrieval.training_data import RetrievalCollator
    from qwen_retrieval.train import make_joint_trainer_class
    torch.set_num_threads(1)
    tokenizer, config = fixture(work / "base")
    collator = RetrievalCollator(tokenizer)
    native = Qwen2ForCausalLM.from_pretrained(work / "base", local_files_only=True)
    joint = QwenRetrievalForCausalLM.from_pretrained(work / "base", local_files_only=True)
    native_state = native.state_dict()
    assert all(torch.equal(value, joint.state_dict()[key]) for key, value in native_state.items())
    # Constructor/extra initialization must preserve the baseline RNG state.
    torch.manual_seed(13)
    Qwen2ForCausalLM(copy.deepcopy(config))
    native_rng = torch.get_rng_state()
    torch.manual_seed(13)
    QwenRetrievalForCausalLM(copy.deepcopy(config))
    assert torch.equal(native_rng, torch.get_rng_state()), "Head initialization consumed the base RNG"
    joint.config.retrieval_weight = 0.
    examples = [sample(tokenizer, 0), sample(tokenizer, 1, long_answer=True)]
    batch = collator(examples)
    plain = {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}
    native.train(); joint.train()
    # Different answer lengths exercise token-count, rather than example-mean, scaling.
    count = torch.tensor(int((batch["labels"][:, 1:] != -100).sum()) + 11)
    a = native(**plain, num_items_in_batch=count).loss
    b = joint(**batch, num_items_in_batch=count).loss
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    a.backward(); b.backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(joint.named_parameters())[name].grad, atol=3e-6, rtol=3e-5)
    boundary = (batch["labels"] != -100).long().argmax(1) - 1
    joint.eval()
    q1 = joint.encode(batch["input_ids"], batch["attention_mask"], boundary)
    changed = batch["input_ids"].clone()
    changed[batch["labels"] != -100] = tokenizer.unk_token_id
    q2 = joint.encode(changed, batch["attention_mask"], boundary)
    torch.testing.assert_close(q1, q2, atol=1e-7, rtol=1e-6)
    mixed = collator([sample(tokenizer, 1, sequence=False), sample(tokenizer, 3)])
    assert mixed["retrieval_mask"].tolist() == [False, True]
    assert mixed["retrieval_sid_ids"].tolist() == [3]
    noseq = collator([sample(tokenizer, 1, sequence=False)])
    assert noseq["item_input_ids"].shape == (0, noseq["input_ids"].shape[1])
    joint.train(); joint.zero_grad()
    joint(**noseq).loss.backward()
    assert joint.retrieval_head.weight.grad is not None and joint.retrieval_head.weight.grad.count_nonzero() == 0
    # Compare the real Trainer contract across two unequal-length microbatches.
    native.zero_grad(); joint.zero_grad()
    training_args = dict(output_dir=str(work / "cpu_contract"), per_device_train_batch_size=1,
                         gradient_accumulation_steps=2, report_to="none", use_cpu=True,
                         average_tokens_across_devices=False)
    old_trainer = Trainer(model=native, args=TrainingArguments(**training_args))
    JointTrainer = make_joint_trainer_class(Trainer)
    new_trainer = JointTrainer(model=joint, args=TrainingArguments(**training_args))
    assert old_trainer.model_accepts_loss_kwargs and new_trainer.model_accepts_loss_kwargs
    old_trainer.current_gradient_accumulation_steps = new_trainer.current_gradient_accumulation_steps = 2
    batches = [collator([row]) for row in examples]
    total = sum((row["labels"][:, 1:] != -100).sum() for row in batches)
    for row in batches:
        base_row = {key: row[key] for key in ("input_ids", "attention_mask", "labels")}
        old_trainer.compute_loss(native, base_row, num_items_in_batch=total).backward()
        new_trainer.compute_loss(joint, row, num_items_in_batch=total).backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(joint.named_parameters())[name].grad, atol=3e-6, rtol=3e-5)
    joint.save_pretrained(work / "cpu_reload")
    reloaded = QwenRetrievalForCausalLM.from_pretrained(work / "cpu_reload", local_files_only=True)
    assert all(torch.equal(value, reloaded.state_dict()[key]) for key, value in joint.state_dict().items())
    return {"rng_unchanged": True, "legacy_token_loss_and_accumulated_gradients": True,
            "left_padding_boundary": True, "answer_invariant_queries": True,
            "mixed_and_zero_seq": True, "head_checkpoint_reload": True}


def distributed_checks(device, rank, world):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from qwen_retrieval.model import distributed_retrieval_loss
    records = {}
    cases = {"unequal_zero_duplicate": [[ ], [1], [1, 2, 3], [3, 4]],
             "all_zero": [[], [], [], []], "one_unique": [[], [2], [2], []]}
    for case, ids_by_rank in cases.items():
        torch.manual_seed(61)
        head = torch.nn.Linear(7, 5, bias=False, device=device)
        reference = copy.deepcopy(head)
        ids = torch.tensor(ids_by_rank[rank], dtype=torch.long, device=device)
        def inputs(values, query):
            if not values:
                return torch.empty((0, 7), device=device)
            return torch.stack([torch.sin(torch.arange(7, device=device).float() * .4 + value + (.1 if query else 0))
                                for value in values])
        q = F.normalize(head(inputs(ids_by_rank[rank], True)), dim=-1)
        e = F.normalize(head(inputs(ids_by_rank[rank], False)), dim=-1)
        loss, stats = distributed_retrieval_loss(q, e, ids, .2)
        assert torch.isfinite(loss)
        loss.backward()
        assert head.weight.grad is not None
        dist.all_reduce(head.weight.grad)
        head.weight.grad /= world
        flat_ids = [value for values in ids_by_rank for value in values]
        unique = sorted(set(flat_ids))
        if len(unique) > 1:
            qref = F.normalize(reference(inputs(flat_ids, True)), dim=-1)
            eref = F.normalize(reference(inputs(unique, False)), dim=-1)
            targets = torch.tensor([unique.index(value) for value in flat_ids], device=device)
            expected = F.cross_entropy(qref @ eref.T / .2, targets)
        else:
            expected = reference.weight.sum() * 0
        expected.backward()
        torch.testing.assert_close(head.weight.grad, reference.weight.grad, atol=5e-6, rtol=5e-5)
        records[case] = {"passed": True, **stats}
    return records


def zero2_checks(work, device, rank, world):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from torch.utils.data import SequentialSampler
    from qwen_retrieval.model import QwenRetrievalForCausalLM
    from qwen_retrieval.training_data import RetrievalCollator
    from qwen_retrieval.train import make_joint_trainer_class
    from scripts.deepspeed_zero2_compat import apply_zero2_accumulation_fix
    metadata = {**apply_zero2_accumulation_fix(), "scope": "current_retrieval_smoke_process_only"}
    tokenizer = AutoTokenizer.from_pretrained(work / "base", local_files_only=True)
    model = QwenRetrievalForCausalLM.from_pretrained(work / "base", local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device.index})
    model.config.retrieval_weight = .1
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    model.enable_retrieval()
    assert torch.equal(cpu_rng, torch.get_rng_state()), "Enabling head changed CPU RNG"
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device)), "Enabling head changed CUDA RNG"
    initial_head = model.retrieval_head.weight.detach().clone()
    rows = [sample(tokenizer, index, sequence=(index // 8) % 2 == 0,
                   long_answer=index % 2 == 0) for index in range(32)]
    valid = [{key: value for key, value in sample(tokenizer, index).items()
              if key in ("input_ids", "attention_mask", "labels")} for index in range(8)]
    JointTrainer = make_joint_trainer_class(Trainer)
    class SequentialJointTrainer(JointTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def evaluation_loop(self, *a, metric_key_prefix="eval", **kw):
            result = super().evaluation_loop(*a, metric_key_prefix=metric_key_prefix, **kw)
            assert torch.isfinite(torch.tensor(result.metrics[metric_key_prefix + "_loss"]))
            result.metrics[metric_key_prefix + "_actual_loss"] = result.metrics[metric_key_prefix + "_loss"]
            result.metrics[metric_key_prefix + "_loss"] = float(self.state.global_step)
            return result
    class Capture(TrainerCallback):
        def __init__(self):
            self.changed = []
        def on_step_end(self, args, state, control, model=None, **kw):
            changed = not torch.equal(model.retrieval_head.weight.detach(), initial_head)
            self.changed.append(changed)
            assert changed, "Zero-SEQ final microbatch overwrote earlier projection gradients"
    args = TrainingArguments(output_dir=str(work / "run"),
        deepspeed=str(Path(__file__).resolve().parents[1] / "config/h20_zero2.json"), bf16=True,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=2, learning_rate=.01, warmup_steps=0, lr_scheduler_type="linear", optim="adamw_torch",
        max_grad_norm=1., eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, report_to="none", disable_tqdm=True, ddp_find_unused_parameters=False,
        remove_unused_columns=False, seed=42, data_seed=42, dataloader_num_workers=0)
    capture = Capture()
    trainer = SequentialJointTrainer(model=model, args=args, train_dataset=rows, eval_dataset=valid,
        processing_class=tokenizer, data_collator=RetrievalCollator(tokenizer), callbacks=[capture])
    trainer.train()
    assert trainer.is_deepspeed_enabled and trainer.model_wrapped.zero_optimization_stage() == 2
    assert trainer.state.global_step == 2 and capture.changed == [True, True]
    assert Path(trainer.state.best_model_checkpoint).name == "checkpoint-1"
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    trainer.save_model(str(work / "final_checkpoint"))
    dist.barrier(device_ids=[device.index])
    loaded = QwenRetrievalForCausalLM.from_pretrained(work / "final_checkpoint", local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device.index})
    assert all(torch.equal(value, loaded.state_dict()[key].cpu()) for key, value in expected.items())
    # Isolate the second synthetic Trainer's Accelerator registry. Production
    # resumes launch a new process, so there is never a second plugin there.
    trainer.accelerator.free_memory()
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state()
    # Resume the exact optimizer/scheduler/RNG checkpoint and replay step two.
    resumed = SequentialJointTrainer(model=loaded, args=args, train_dataset=rows, eval_dataset=valid,
        processing_class=tokenizer, data_collator=RetrievalCollator(tokenizer))
    resumed.train(resume_from_checkpoint=str(work / "run/checkpoint-1"))
    assert resumed.state.global_step == 2
    for key, value in expected.items():
        torch.testing.assert_close(value, loaded.state_dict()[key].cpu(), atol=1e-5, rtol=1e-5)
    for wrapped in (trainer.model_wrapped, resumed.model_wrapped):
        wrapped.destroy()
        wrapped.optimizer = None
    return {"passed": True, "steps": 2, "gradient_accumulation_steps": 2,
            "earlier_best_checkpoint_reload": "checkpoint-1", "cpu_and_cuda_rng_unchanged": True,
            "zero_stage": 2, "gradient_checkpointing": True,
            "nonzero_first_micro_zero_seq_last_micro": True,
            "joint_head_save_reload": True, "optimizer_scheduler_resume": True,
            "runtime_compatibility": metadata}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args(argv)
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        result = {"status": "passed", "cpu": cpu_checks(work), "production_training": False}
        (work / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    import torch
    import torch.distributed as dist
    rank, local_rank, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 4:
        raise ValueError("This smoke requires four GPU ranks")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    if rank == 0:
        # CPU Trainer checks cannot join an already initialized NCCL process;
        # use a clean subprocess with distributed environment removed.
        import subprocess, sys
        cpu_env = {key: value for key, value in os.environ.items() if key not in
                   {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"}}
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, "-m", "qwen_retrieval.smoke", "--cpu-only",
                        "--output-dir", str(work)], check=True, env=cpu_env)
    dist.barrier(device_ids=[local_rank])
    distributed = distributed_checks(torch.device("cuda", local_rank), rank, world)
    zero2 = zero2_checks(work, torch.device("cuda", local_rank), rank, world)
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        result = json.loads((work / "smoke_result.json").read_text())
        result.update(world_size=world, distributed=distributed, zero2=zero2)
        (work / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "passed", "output": str(work / "smoke_result.json")}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
