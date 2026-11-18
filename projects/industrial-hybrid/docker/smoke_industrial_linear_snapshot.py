#!/usr/bin/env python3
"""Four-GPU ZeRO-2 smoke for loss-only early stopping and live epoch snapshots.

Only synthetic tiny-model fixtures are used. The real validation forward pass
still runs, but its selection loss is scripted to worsen after step one so the
test can prove an epoch-three snapshot precedes best-model reload. No ranking
or generate() call is permitted in this smoke.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    import torch.distributed as dist
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments)
    from data import SidSFTDataset
    from docker.smoke_industrial_rank_selection import make_fixture, sha256
    from scripts.sft_eval_utils import verify_sid_tokenizer
    from sft import EpochSnapshotCallback, MinimumEpochEarlyStoppingCallback, write_sid_provenance

    rank, local_rank, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4, "This smoke requires exactly four GPU ranks")
    require("scripts.industrial_generative_validation" not in sys.modules,
            "Loss-only smoke unexpectedly imported the ranking validation module")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    work = args.output_dir.resolve()
    if rank == 0:
        require(not work.exists() or not any(work.iterdir()), "Choose a fresh smoke output directory")
        work.mkdir(parents=True, exist_ok=True)
        # This helper constructs local data/tokenizer/weights only. It does not
        # import or execute its ranking smoke main() function.
        make_fixture(work)
    dist.barrier(device_ids=[local_rank])
    base, inputs, prepared, run = (work / name for name in ("base", "inputs", "prepared", "run"))
    sid_path = inputs / "Industrial_and_Scientific.index.json"
    metadata_path = inputs / "Industrial_and_Scientific.item.json"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, attn_implementation="sdpa",
    )
    generation_calls = []

    def forbid_generation(*unused_args, **unused_kwargs):
        generation_calls.append(True)
        raise AssertionError("Loss-only training must never invoke generate()")

    model.generate = forbid_generation
    train = SidSFTDataset(str(prepared / "train.csv"), tokenizer, max_len=512, sample=-1,
                          seed=42, category="industrial and scientific items")
    valid = SidSFTDataset(str(prepared / "valid.csv"), tokenizer, max_len=512, sample=-1,
                          seed=None, category="industrial and scientific items")

    class ScriptedLossTrainer(Trainer):
        """Keep real teacher-forced validation, override only this smoke's criterion."""
        def evaluation_loop(self, *loop_args, metric_key_prefix="eval", **loop_kwargs):
            output = super().evaluation_loop(*loop_args, metric_key_prefix=metric_key_prefix, **loop_kwargs)
            key = metric_key_prefix + "_loss"
            actual = float(output.metrics[key])
            require(math.isfinite(actual), "Real teacher-forced smoke loss is not finite")
            output.metrics[metric_key_prefix + "_actual_loss"] = actual
            output.metrics[key] = float(self.state.global_step)
            return output

    class CaptureLiveEpochThree(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.state_dict = None
            self.step = None

        def on_epoch_end(self, callback_args, state, control, **kwargs):
            if rank == 0 and math.isclose(float(state.epoch or 0), 3.0, abs_tol=1e-9):
                bare = self.trainer.accelerator.unwrap_model(self.trainer.model_wrapped)
                self.state_dict = {key: value.detach().cpu().clone() for key, value in bare.state_dict().items()}
                self.step = int(state.global_step)
            return control

    minimum = MinimumEpochEarlyStoppingCallback(minimum_epochs=3, early_stopping_patience=1)
    snapshot = EpochSnapshotCallback(str(run), 3, tokenizer, str(sid_path), str(metadata_path),
                                     str(base), "Industrial_and_Scientific")
    capture = CaptureLiveEpochThree()
    trainer = ScriptedLossTrainer(
        model=model, train_dataset=train, eval_dataset=valid, processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(run), deepspeed=str(ROOT / "config/h20_zero2.json"), bf16=True,
            per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            num_train_epochs=5, learning_rate=1e-3, warmup_steps=0, lr_scheduler_type="linear",
            eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, optim="adamw_torch", logging_steps=1,
            report_to="none", disable_tqdm=True, ddp_find_unused_parameters=False,
            dataloader_num_workers=0, seed=42,
        ),
        callbacks=[minimum, snapshot, capture],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
    )
    snapshot.attach_trainer(trainer)
    capture.trainer = trainer
    result = trainer.train()
    require(trainer.is_deepspeed_enabled and trainer.model_wrapped.zero_optimization_stage() == 2,
            "Smoke did not exercise real DeepSpeed ZeRO-2")
    require(trainer.state.max_steps == 10, "Fixture must have two optimizer steps per epoch and five planned epochs")
    require(trainer.state.global_step == 6 and math.isclose(float(trainer.state.epoch), 3.0),
            "Minimum-three-epoch guard did not stop at the exact epoch-three boundary")
    require(Path(trainer.state.best_model_checkpoint).name == "checkpoint-1",
            "Scripted loss should retain the first checkpoint as the best model")
    require(minimum.early_stopping_patience_counter >= 1,
            "Patience never requested early stopping before the minimum epoch boundary")
    require(snapshot.metadata and snapshot.metadata["global_step"] == 6
            and snapshot.metadata["epoch"] == 3.0
            and snapshot.metadata["best_model_reloaded"] is False,
            "Snapshot metadata does not identify live epoch-three weights")
    evaluations = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    require([entry["step"] for entry in evaluations] == list(range(1, 7)),
            "Minimum-epoch guard must permit all six loss-only validation events")
    require([entry["eval_loss"] for entry in evaluations] == list(map(float, range(1, 7))),
            "The scripted early-stopping loss sequence changed")
    require(not any(any("ndcg" in key.lower() or "recall" in key.lower() or key.startswith("eval_HR")
                        for key in entry) for entry in evaluations),
            "Training unexpectedly computed ranking metrics")
    require(not generation_calls and "scripts.industrial_generative_validation" not in sys.modules,
            "Loss-only training invoked ranking generation")
    require(not (run / "validation_ranking").exists(), "Loss-only smoke created ranking-validation artifacts")
    final = run / "final_checkpoint"
    trainer.save_model(str(final))
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        tokenizer.save_pretrained(final)
        write_sid_provenance(final, str(sid_path), str(metadata_path), str(base), "Industrial_and_Scientific")
        require(capture.state_dict is not None and capture.step == 6,
                "Live epoch-three weights were not captured before best-model reload")
        catalog = json.loads(sid_path.read_text())
        states = {}
        for name, path in (("snapshot", snapshot.directory), ("latest", run / "checkpoint-6"),
                           ("best", Path(trainer.state.best_model_checkpoint)), ("final", final)):
            loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
            states[name] = {key: value.clone() for key, value in loaded.state_dict().items()}
            saved_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            verify_sid_tokenizer(saved_tokenizer, loaded, catalog)
            del loaded
        def equal(first, second):
            return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)
        require(equal(states["snapshot"], capture.state_dict), "Epoch snapshot differs from live weights at capture")
        require(equal(states["snapshot"], states["latest"]), "Epoch snapshot differs from raw checkpoint six")
        require(equal(states["final"], states["best"]), "Final model did not reload the first/best-loss checkpoint")
        require(not equal(states["snapshot"], states["final"]),
                "Epoch snapshot incorrectly contains the reloaded best model instead of raw training weights")
        metadata = json.loads((snapshot.directory / "snapshot_metadata.json").read_text())
        require(metadata["capture_event"] == "on_epoch_end" and metadata["requested_epoch"] == 3
                and metadata["steps_per_epoch"] == 2.0, "Snapshot was not recorded at the exact epoch boundary")
        provenance = json.loads((snapshot.directory / "posttraining_provenance.json").read_text())
        require(provenance["sid_index_sha256"] == sha256(sid_path)
                and provenance["item_meta_sha256"] == sha256(metadata_path), "Snapshot provenance changed")
        report = {"status": "passed", "world_size": world, "zero_stage": 2,
                  "scheduler": "linear", "planned_epochs": 5, "planned_steps": trainer.state.max_steps,
                  "actual_epoch": trainer.state.epoch, "actual_step": result.global_step,
                  "minimum_epochs": 3, "patience": 1, "best_loss_checkpoint": trainer.state.best_model_checkpoint,
                  "epoch_snapshot": str(snapshot.directory), "snapshot_metadata": metadata,
                  "snapshot_matches_live_weights": True, "snapshot_matches_checkpoint_6": True,
                  "snapshot_differs_from_best_model": True, "final_matches_best_model": True,
                  "snapshot_precedes_reload": True, "no_ranking_calls": True, "generation_calls": 0,
                  "loss_evaluation_steps": [entry["step"] for entry in evaluations],
                  "no_test_split_opened": not (prepared / "test.csv").exists(), "work_dir": str(work)}
        (work / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
