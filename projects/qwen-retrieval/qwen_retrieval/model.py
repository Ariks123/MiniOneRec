"""Shared causal Qwen generator and SID retriever; no changes to legacy Qwen keys."""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


def distributed_retrieval_loss(queries, items, sid_ids, temperature):
    """Local queries against globally unique SID items, with differentiable gathering.

    DeepSpeed/DDP average parameter gradients across ranks. The W/N scaling below
    therefore reproduces the global sample mean, including unequal local counts.
    Every rank enters the same autograd collective, even with no local SEQ rows.
    """
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    count = torch.tensor([len(sid_ids)], dtype=torch.long, device=items.device)
    if world > 1:
        counts = [torch.empty_like(count) for _ in range(world)]
        dist.all_gather(counts, count)
        counts = [int(value.item()) for value in counts]
        width = max(1, max(counts))
        padded = F.pad(items, (0, 0, 0, width - len(items)))
        ids = F.pad(sid_ids, (0, width - len(sid_ids)), value=-1)
        from torch.distributed.nn.functional import all_gather
        gathered = all_gather(padded)
        gathered_ids = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(gathered_ids, ids)
        all_items = torch.cat([value[:n] for value, n in zip(gathered, counts)])
        all_ids = torch.cat([value[:n] for value, n in zip(gathered_ids, counts)])
        # Include padding in the zero dependency: a rank with zero items must
        # still execute the all_gather backward when every rank has zero rows.
        dependency = sum(value.sum() * 0 for value in gathered)
    else:
        all_items, all_ids, counts = items, sid_ids, [len(sid_ids)]
        dependency = items.sum() * 0
    dependency = dependency + queries.sum() * 0
    total = sum(counts)
    if total == 0:
        return dependency, {"queries": 0, "unique_items": 0}
    # The exact same SID template yields the same deterministic item encoding;
    # retaining its first rank/order occurrence removes false negatives and
    # prevents duplicate target popularity from reweighting the denominator.
    unique_ids = torch.unique(all_ids, sorted=True)
    first = torch.stack([(all_ids == value).nonzero(as_tuple=False)[0, 0] for value in unique_ids])
    candidates = all_items[first]
    if len(unique_ids) < 2 or len(queries) == 0:
        return dependency, {"queries": total, "unique_items": len(unique_ids)}
    targets = torch.searchsorted(unique_ids, sid_ids)
    scores = queries.float() @ candidates.float().T / temperature
    loss = F.cross_entropy(scores, targets, reduction="sum") * (world / total)
    return loss + dependency, {"queries": total, "unique_items": len(unique_ids)}


class QwenRetrievalForCausalLM(Qwen2ForCausalLM):
    """Native Qwen checkpoint with a shared, saved retrieval projection."""
    def __init__(self, config):
        super().__init__(config)
        self.enable_retrieval()

    def enable_retrieval(self):
        """Register after legacy token extension when starting from Base weights."""
        config = self.config
        config.retrieval_dim = int(getattr(config, "retrieval_dim", 256))
        config.retrieval_weight = float(getattr(config, "retrieval_weight", .1))
        config.retrieval_temperature = float(getattr(config, "retrieval_temperature", .07))
        config.retrieval_format_version = 1
        if config.retrieval_dim < 1 or config.retrieval_weight < 0 or config.retrieval_temperature <= 0:
            raise ValueError("Invalid retrieval dimension, weight, or temperature")
        # A new head must not consume the RNG stream used by the historical
        # SID embedding extension or subsequent training sampler.
        with torch.random.fork_rng(devices=[]):
            self.retrieval_head = nn.Linear(config.hidden_size, config.retrieval_dim, bias=False)
            self.retrieval_head._retrieval_projection = True
            self._init_weights(self.retrieval_head)
            reference = self.get_input_embeddings().weight
            self.retrieval_head.to(device=reference.device, dtype=reference.dtype)
        self.last_retrieval_stats = {}

    def _init_weights(self, module):
        if getattr(module, "_retrieval_projection", False):
            if not module.weight.is_meta:
                generator = torch.Generator(device=module.weight.device)
                generator.manual_seed(int(getattr(self.config, "retrieval_head_seed", 1042)))
                with torch.no_grad():
                    module.weight.normal_(mean=0., std=self.config.initializer_range, generator=generator)
        else:
            super()._init_weights(module)

    def _project(self, hidden):
        return F.normalize(self.retrieval_head(hidden).float(), p=2, dim=-1, eps=1e-12)

    def encode(self, input_ids, attention_mask, positions=None):
        if positions is None:
            arange = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
            positions = arange.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if (positions < 0).any() or not attention_mask.gather(1, positions[:, None]).bool().all():
            raise ValueError("Embedding positions must identify valid input tokens")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
        return self._project(outputs.last_hidden_state[torch.arange(len(input_ids), device=input_ids.device), positions])

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                retrieval_mask=None, retrieval_sid_ids=None,
                item_input_ids=None, item_attention_mask=None, retrieval_loss_scale=1.0, **kwargs):
        if retrieval_mask is None or not self.training:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels, **kwargs)
        if labels is None or attention_mask is None:
            raise ValueError("Joint training requires labels and attention_mask")
        # Collator already pads both branches to the generation width. One
        # decoder call avoids repeated checkpointed use of a ZeRO parameter.
        batch_size = len(input_ids)
        selected = retrieval_mask.bool().nonzero(as_tuple=False).flatten()
        if len(selected) != len(retrieval_sid_ids) or len(selected) != len(item_input_ids):
            raise ValueError("SEQ metadata and item batches are not aligned")
        combined_ids = torch.cat((input_ids, item_input_ids), dim=0)
        combined_mask = torch.cat((attention_mask, item_attention_mask), dim=0)
        outputs = self.model(input_ids=combined_ids, attention_mask=combined_mask,
                             use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden[:batch_size])
        loss_kwargs = {}
        if kwargs.get("num_items_in_batch") is not None:
            loss_kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"]
        gen_loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.vocab_size, **loss_kwargs)
        seq_labels = labels[selected]
        boundary = (seq_labels != -100).long().argmax(dim=1) - 1
        if len(selected) and ((boundary < 0).any() or
                not attention_mask[selected].gather(1, boundary[:, None]).bool().all()):
            raise ValueError("SEQ query boundary has been truncated or padded")
        queries = self._project(hidden[selected, boundary])
        item_positions = torch.arange(combined_ids.shape[1], device=input_ids.device).expand_as(item_input_ids)
        item_positions = item_positions.masked_fill(~item_attention_mask.bool(), -1).max(dim=1).values
        items = self._project(hidden[batch_size + torch.arange(len(selected), device=input_ids.device), item_positions])
        ret_loss, stats = distributed_retrieval_loss(
            queries, items, retrieval_sid_ids, self.config.retrieval_temperature)
        loss = gen_loss + self.config.retrieval_weight * ret_loss * retrieval_loss_scale
        global_ret = ret_loss.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_ret)
            global_ret /= dist.get_world_size()
        count = (labels[:, 1:] != -100).sum().clamp_min(1)
        gen_token_mean = gen_loss.detach()
        if loss_kwargs:
            gen_token_mean = gen_token_mean * loss_kwargs["num_items_in_batch"] / count
        self.last_retrieval_stats = {
            "gen_token_mean_last_micro": float(gen_token_mean),
            "loss_global_mean_last_micro": float(global_ret),
            "global_queries_last_micro": stats["queries"],
            "global_unique_items_last_micro": stats["unique_items"]}
        return CausalLMOutputWithPast(loss=loss, logits=logits)
