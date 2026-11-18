# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""MiniOneRec GRPO mathematics and deterministic beam rollouts for Industrial.

This is the narrow industrial adaptation of the original ReReTrainer. It keeps
the group-normalized reward, policy gradient and reference KL unchanged, while
carrying each source row's identity and target through distributed generation.
"""
from collections import defaultdict
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from accelerate.utils import gather, gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList, Trainer
from transformers import TemperatureLogitsWarper
from trl import SyncRefModelCallback
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax

from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
from scripts.industrial_rl_data import (
    normalize_sid, require, validate_generation_batch_sizes, validate_rollout_groups,
)


class RepeatRandomSampler(Sampler):
    """The original sampler: keep each source row's G copies contiguous."""
    def __init__(self, data_source, repeat_count, seed=None):
        self.data_source, self.repeat_count = data_source, repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        return iter([index for index in torch.randperm(self.num_samples, generator=self.generator).tolist()
                     for _ in range(self.repeat_count)])

    def __len__(self):
        return self.num_samples * self.repeat_count


class ReReTrainer(Trainer):
    def __init__(self, model, base_model, reward_funcs, args, *, info_file,
                 train_dataset=None, eval_dataset=None, processing_class=None,
                 callbacks=None, beam_search=True, add_gt=False,
                 dynamic_sampling=False, test_during_training=False,
                 dapo=False, gspo=False, length_penalty=0.0, **kwargs):
        require(beam_search and not any([add_gt, dynamic_sampling, test_during_training, dapo, gspo]),
                "Industrial GRPO uses deterministic beams without GT injection or alternative losses")
        require(not args.use_vllm, "Industrial GRPO uses the original Transformers beam path")
        self.base_model = str(base_model)
        self.num_generations = args.num_generations
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.beta = args.beta
        self.temperature = args.temperature
        self.beam_search = True
        self.reward_funcs = list(reward_funcs)
        self._metrics = defaultdict(list)
        self._tag_names = ["trl", "grpo", "minionerec-industrial"]
        if isinstance(model, str):
            init_kwargs = dict(args.model_init_kwargs or {})
            if isinstance(init_kwargs.get("torch_dtype"), str):
                init_kwargs["torch_dtype"] = getattr(torch, init_kwargs["torch_dtype"])
            init_kwargs["use_cache"] = False
            model = AutoModelForCausalLM.from_pretrained(model, **init_kwargs)
        model.config.use_cache = False
        self.ref_model = create_reference_model(model)
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
        processing_class.padding_side = "left"
        processing_class.pad_token = processing_class.eos_token
        weights = args.reward_weights
        self.reward_weights = torch.tensor(weights if weights is not None else [1.0] * len(self.reward_funcs),
                                           dtype=torch.float32)
        require(len(self.reward_weights) == len(self.reward_funcs), "Reward weights/functions differ")
        model.warnings_issued["estimate_tokens"] = True
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         processing_class=processing_class, data_collator=lambda features: features,
                         callbacks=callbacks, **kwargs)
        validate_generation_batch_sizes(args.per_device_train_batch_size, args.per_device_eval_batch_size,
                                        self.num_generations, self.accelerator.num_processes,
                                        beam_search=True, eval_enabled=args.eval_strategy != "no")
        # Custom compute_loss must receive Trainer's normal accumulation scaling.
        self.model_accepts_loss_kwargs = False
        if self.is_deepspeed_enabled:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        with open(info_file, encoding="utf-8") as source:
            candidates = [line.split("\t")[0].strip() for line in source if line.strip()]
        self.catalog_sids = set(candidates)
        self.hash_dict = build_prefix_allowed_tokens(processing_class, (sid + "\n" for sid in candidates))
        require(self.max_completion_length >= max(map(len, self.hash_dict)) + 1,
                "Completion length cannot finish every catalog SID")
        require(len(self.catalog_sids) >= self.num_generations, "Catalog has too few unique SID candidates")
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length, length_penalty=length_penalty,
            num_beams=self.num_generations, num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id, eos_token_id=processing_class.eos_token_id,
            do_sample=False, top_k=None, top_p=None, temperature=self.temperature,
        )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "sample_id", "task_type", "target_sid"]

    def _get_train_sampler(self, train_dataset=None):
        return RepeatRandomSampler(self.train_dataset if train_dataset is None else train_dataset,
                                   self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _coordinated_require(self, error):
        errors = [error]
        if dist.is_available() and dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            raise ValueError("Distributed RL integrity check failed: " + "; ".join(e for e in errors if e))

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       logits_to_keep=logits_to_keep + 1).logits[:, :-1, :]
        return selective_log_softmax(logits[:, -logits_to_keep:, :], input_ids[:, -logits_to_keep:])

    def _prepare_inputs(self, inputs):
        error = None
        try:
            validate_rollout_groups(inputs, self.num_generations)
        except (ValueError, KeyError, TypeError) as exc:
            error = f"rank {self.accelerator.process_index}: {exc}"
        self._coordinated_require(error)
        device = self.accelerator.device
        prompts = [row["prompt"] for row in inputs]
        targets = [row["target_sid"] for row in inputs]
        batch = self.processing_class(prompts, return_tensors="pt", padding=True,
                                      padding_side="left", add_special_tokens=False)
        batch = super()._prepare_inputs(batch)
        prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
        if self.max_prompt_length is not None:
            self._metrics["prompt_truncation_rate"].append(float(
                (prompt_mask.sum(1) > self.max_prompt_length).float().mean().item()))
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        width = prompt_ids.shape[1]
        processor = ConstrainedLogitsProcessor(
            lambda batch_id, completion: self.hash_dict.get(tuple(completion), []),
            num_beams=self.num_generations, base_model=self.base_model,
            eos_token_id=self.processing_class.eos_token_id, prompt_length=width)
        processors = LogitsProcessorList([TemperatureLogitsWarper(self.temperature), processor])
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prior_training = unwrapped_model.training
            try:
                unwrapped_model.eval()
                with torch.no_grad():
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids[::self.num_generations], attention_mask=prompt_mask[::self.num_generations],
                        generation_config=self.generation_config, logits_processor=processors,
                        do_sample=False, use_cache=True, synced_gpus=False)
            finally:
                unwrapped_model.train(prior_training)
        self._coordinated_require(None if prompt_completion_ids.shape[0] == len(inputs)
                                  else "Beam output does not contain one full group per source row")
        prompt_ids = prompt_completion_ids[:, :width]
        completion_ids = prompt_completion_ids[:, width:]
        decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
        completions = [normalize_sid(value) for value in decoded]
        error = None
        if any(value not in self.catalog_sids for value in completions):
            error = "Beam produced a SID outside the complete catalog"
        elif any(len(set(completions[offset:offset + self.num_generations])) != self.num_generations
                 for offset in range(0, len(completions), self.num_generations)):
            error = "Beam produced repeated SID candidates inside a group"
        self._coordinated_require(error)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_index = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_index[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (indices <= eos_index.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        with torch.inference_mode():
            ref_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids,
                                                 attention_mask, completion_ids.shape[1])
        reward_kwargs = {key: [row[key] for row in inputs]
                         for key in ("sample_id", "task_type", "target_sid")}
        local_rewards = torch.tensor([
            function(prompts=prompts, completions=completions, **reward_kwargs)
            for function in self.reward_funcs], dtype=torch.float32, device=device).T
        self._coordinated_require(None if torch.isfinite(local_rewards).all().item()
                                  else "Nonfinite row-local RL rewards")
        rewards_per_func = gather(local_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(1)
        grouped = rewards.view(-1, self.num_generations)
        means, stds = grouped.mean(1), grouped.std(1)
        # The original MiniOneRec GRPO advantage, without clipping or alternate losses.
        advantages = (rewards - means.repeat_interleave(self.num_generations)) / (
            stds.repeat_interleave(self.num_generations) + 1e-4)
        local_hits = torch.tensor([any(completions[i] == targets[i]
                                        for i in range(offset, offset + self.num_generations))
                                   for offset in range(0, len(inputs), self.num_generations)],
                                  dtype=torch.float32, device=device)
        zero_groups = (grouped == 0).all(1).float()
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(stds.mean().item())
        self._metrics["zero_reward_group_rate"].append(zero_groups.mean().item())
        self._metrics["hit_group_rate"].append(gather(local_hits).mean().item())
        self._metrics["advantage_nonzero_group_rate"].append((stds > 0).float().mean().item())
        self._metrics["candidate_validity_rate"].append(1.0)
        self._metrics["candidate_unique_rate"].append(1.0)
        for index, function in enumerate(self.reward_funcs):
            self._metrics["rewards/" + function.__name__].append(rewards_per_func[:, index].mean().item())
        tasks = gather_object([row["task_type"] for row in inputs[::self.num_generations]])
        require(len(tasks) == len(means), "Task labels and global reward groups differ")
        for task in sorted(set(tasks)):
            mask = torch.tensor([name == task for name in tasks], dtype=torch.bool, device=device)
            self._metrics[f"task/{task}/reward"].append(means[mask].mean().item())
            self._metrics[f"task/{task}/zero_reward_group_rate"].append(zero_groups[mask].mean().item())
        start = self.accelerator.process_index * len(inputs)
        return {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
                "completion_ids": completion_ids, "completion_mask": completion_mask,
                "ref_per_token_logps": ref_logps,
                "advantages": advantages[start:start + len(inputs)],
                "sliced_rewards": rewards[start:start + len(inputs)]}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The MiniOneRec GRPO loss does not return model outputs")
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logps = self._get_per_token_logps(model, input_ids, attention_mask,
                                          inputs["completion_ids"].shape[1])
        difference = inputs["ref_per_token_logps"] - logps
        per_token_kl = torch.exp(difference) - difference - 1
        per_token_loss = torch.exp(logps - logps.detach()) * inputs["advantages"].unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
        lengths = completion_mask.sum(1)
        self._metrics["completion_length"].append(gather(lengths.float()).mean().item())
        kl = ((per_token_kl * completion_mask).sum(1) / lengths).mean()
        self._metrics["kl"].append(gather(kl.reshape(1)).mean().item())
        return loss

    def log(self, logs, start_time=None):
        averaged = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        if logs and next(iter(logs)).startswith("eval_"):
            averaged = {"eval_" + key: value for key, value in averaged.items()}
        super().log({**logs, **averaged}, start_time)
        self._metrics.clear()

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        source = Path(self.base_model) / "posttraining_provenance.json"
        if source.is_file():
            shutil.copyfile(source, Path(output_dir or self.args.output_dir) / source.name)
