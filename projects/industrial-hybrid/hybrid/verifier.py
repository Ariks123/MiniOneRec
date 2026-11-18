"""MiniOneRec original-eval beam and shared-history teacher-forcing inference.

A session is scoped to one batch of online requests. Its KV and small likelihood
tables are discarded with that batch. Only legal SID columns are retained, but
every probability is normalized over the complete model vocabulary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass, field
from heapq import nlargest
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from data import BaseDataset, Tokenizer
from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens
from scripts.sft_eval_utils import verify_sid_tokenizer


INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request. \n\n"
    "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
)
QUESTION = "Can you predict the next possible item the user may expect, given the following chronological interaction history: "


@dataclass
class Session:
    ids: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    kv: object
    roots: list[dict]
    second: list[dict] = field(default_factory=list)
    third: list[dict] = field(default_factory=list)


def _position_ids(mask):
    return (mask.long().cumsum(1) - 1).masked_fill(mask == 0, 1)


def _fork_cache(original, rows):
    """Copy selected KV slots; appending a suffix cannot mutate another branch.

    The supported 4.57 Qwen2 cache has DynamicLayer entries. Shallow-copy the
    cache metadata before selecting rows so an entire redundant copy of the
    original, followed by a second batch-expanded copy, is avoided.
    """
    if not hasattr(original, "layers"):
        raise RuntimeError("Expected Transformers 4.57 DynamicCache with per-layer storage")
    fresh = copy(original)
    fresh.layers = []
    for layer in original.layers:
        branch = copy(layer)
        branch.batch_select_indices(rows)
        fresh.layers.append(branch)
    return fresh


def _logprob_columns(logits, columns):
    # Subtract the same complete-vocabulary normalization constant from every
    # selected token. These columns are never normalized among themselves.
    full = logits.float()
    return full.index_select(-1, columns) - torch.logsumexp(full, dim=-1, keepdim=True)


class Verifier:
    def __init__(self, model_path, catalog, device="cuda:0", chunk_size=64):
        device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        options = dict(local_files_only=True, attn_implementation="sdpa",
                       torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        if device.type == "cuda":
            options["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **options)
        self._configure(model, tokenizer, catalog, device, chunk_size, 50)

    def _configure(self, model, tokenizer, catalog, device, chunk_size, beam_width):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.full_width = int(beam_width)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        atoms = {item: re.findall(r"<[abc]_\d+>", sid) if isinstance(sid, str) else list(sid)
                 for item, sid in catalog.items()}
        if not all(len(parts) == 3 and all(re.fullmatch(fr"<{letter}_\d+>", atom)
                   for letter, atom in zip("abc", parts)) for parts in atoms.values()):
            raise ValueError("Industrial catalog must contain complete original three-token SIDs")
        verify_sid_tokenizer(tokenizer, model, atoms)
        self.tokens = {''.join(parts): tuple(tokenizer.convert_tokens_to_ids(parts)) for parts in atoms.values()}
        if not all(tokenizer.encode(s, add_special_tokens=False) == list(t) for s,t in self.tokens.items()):
            raise ValueError("Concatenated SID does not preserve its atomic tokenization")
        self.atom_for_id = {tokenizer.convert_tokens_to_ids(a): a for parts in atoms.values() for a in parts}
        self.trie = build_prefix_allowed_tokens(tokenizer, (s + "\n" for s in sorted(self.tokens)))
        self.answer_length = max(map(len, self.trie)) + 1
        self.all_columns = [sorted({v[i] for v in self.tokens.values()}) for i in range(3)]
        self.column_tensors = [torch.tensor(values, device=self.device) for values in self.all_columns]
        self.offsets = [{v: i for i,v in enumerate(values)} for values in self.all_columns]
        self.instruction = Tokenizer(tokenizer).encode(INSTRUCTION, bos=True, eos=False)
        self.prompt_formatter = BaseDataset(tokenizer=tokenizer)
        self.counters = Counter()
        def count_forward(module, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is not None:
                self.counters["forward_calls"] += 1
                self.counters["model_input_tokens_padded"] += ids.numel()
        self._forward_counter = model.register_forward_pre_hook(count_forward, with_kwargs=True)

    def prompts(self, histories):
        wrapper = self.prompt_formatter.tokenizer
        result = []
        for history in histories:
            if not history or any(sid not in self.tokens for sid in history):
                raise ValueError("Expected nonempty, catalog-valid history SID strings")
            formatted = self.prompt_formatter.generate_prompt({"input": QUESTION + ", ".join(history), "output": ""})
            sequence = self.instruction + wrapper.encode(formatted, bos=False, eos=False)
            result.append(sequence[-(512 - self.answer_length):])
        return result

    def _pad(self, prompts):
        if not prompts or any(not p for p in prompts):
            raise ValueError("Missing nonempty history prompts")
        inputs = [{"input_ids": list(ids), "attention_mask": [1] * len(ids)} for ids in prompts]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    @torch.inference_mode()
    def prefill(self, prompts):
        ids, mask = self._pad(prompts)
        output = self.model(input_ids=ids, attention_mask=mask, position_ids=_position_ids(mask),
                            logits_to_keep=1, use_cache=True)
        first = _logprob_columns(output.logits[:, -1, :], self.column_tensors[0]).cpu().tolist()
        self.counters["prefill_requests"] += len(prompts)
        self.counters["prefill_tokens"] += sum(len(p) for p in prompts)
        return Session(ids, mask, mask.sum(1), output.past_key_values,
                       [dict(zip(self.all_columns[0], row)) for row in first],
                       [{} for _ in prompts], [{} for _ in prompts])

    def _request_rows(self, session, rows):
        indices = list(range(len(session.roots))) if rows is None else list(rows)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(session.roots) for i in indices):
            raise ValueError("Invalid request indices in verifier session")
        return indices

    def _suffix_forward(self, session, requests, suffixes):
        device = self.device
        rows = torch.as_tensor(requests, device=device)
        tokens = torch.as_tensor(suffixes, device=device)
        mask = torch.cat((session.mask.index_select(0, rows), torch.ones_like(tokens)), dim=1)
        positions = session.lengths.index_select(0, rows)[:, None] + torch.arange(tokens.shape[1], device=device)
        cache = _fork_cache(session.kv, rows)
        self.counters["candidate_tokens"] += tokens.numel()
        self.counters["kv_branch_rows"] += len(requests)
        return self.model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                          past_key_values=cache, use_cache=False, logits_to_keep=tokens.shape[1]).logits

    def _fill_suffixes(self, session, missing, depth):
        # A single branch [a,b] produces distributions for both b and c in a
        # single forward. Every item sharing (a,b) uses the same final distribution.
        for start in range(0, len(missing), self.chunk_size):
            work = missing[start:start + self.chunk_size]
            # BF16 kernels change accumulation order with branch batch shape.
            # Use one physical shape in the grid and live execution so that a
            # close pair of scores cannot flip merely because more items were drafted.
            padded = work + [work[-1]] * (self.chunk_size - len(work))
            requests = [request for request, _ in padded]
            paths = [path if len(path) == 2 else (path[0], self.trie[path][0])
                     for _, path in padded]
            logits = self._suffix_forward(session, requests, paths)[:len(work)]
            second = _logprob_columns(logits[:, 0, :], self.column_tensors[1]).cpu().tolist()
            third = None
            if depth == 2:
                third = _logprob_columns(logits[:, 1, :], self.column_tensors[2]).cpu().tolist()
            for j,(request,path) in enumerate(work):
                # Store only the legal children in each small per-history table.
                a = path[0]
                if a not in session.second[request]:
                    session.second[request][a] = {b: second[j][self.offsets[1][b]] for b in self.trie[(a,)]}
                if third is not None:
                    session.third[request][path] = {c: third[j][self.offsets[2][c]] for c in self.trie[path]}

    @torch.inference_mode()
    def token_scores(self, session, candidate_sids, rows=None):
        indices = self._request_rows(session, rows)
        if len(indices) != len(candidate_sids):
            raise ValueError("Candidate groups must align to the requested history indices")
        groups = [[self.tokens[sid] for sid in group] for group in candidate_sids]
        missing = sorted({(i, path[:2]) for i,group in zip(indices, groups)
                          for path in group if path[:2] not in session.third[i]})
        self._fill_suffixes(session, missing, 2)
        return [[(session.roots[i][a], session.second[i][a][b], session.third[i][(a,b)][c])
                 for a,b,c in group] for i,group in zip(indices,groups)]

    def score(self, session, candidate_sids, rows=None):
        triples = self.token_scores(session, candidate_sids, rows)
        return [[sum(values) / len(values) for values in group] for group in triples]

    @torch.inference_mode()
    def prefixes(self, session, width, depth, rows=None):
        if width < 1 or depth not in (1,2):
            raise ValueError("Industrial prefix beams require positive width and depth 1 or 2")
        indices = self._request_rows(session, rows)
        # Autoregressive pruning uses the requested width at each depth.
        # Equal scores resolve deterministically in tokenizer ID order.
        roots = [nlargest(width, session.roots[i], key=lambda a: (session.roots[i][a], -a)) for i in indices]
        if depth == 1:
            return [[self.atom_for_id[a] for a in root] for root in roots]
        jobs = [(i, (a,)) for i,root in zip(indices,roots) for a in root if a not in session.second[i]]
        self._fill_suffixes(session, jobs, 1)
        result = []
        for i, root in zip(indices,roots):
            beams = ((session.roots[i][a] + value, a, b) for a in root for b,value in session.second[i][a].items())
            best = nlargest(width, beams, key=lambda x: (x[0], -x[1], -x[2]))
            result.append([self.atom_for_id[a] + self.atom_for_id[b] for _,a,b in best])
        return result

    def _beam(self, ids, mask, **kwargs):
        from LogitProcessor import ConstrainedLogitsProcessor
        processor = ConstrainedLogitsProcessor(lambda batch, seq: self.trie.get(tuple(seq), []),
                             self.full_width, eos_token_id=self.tokenizer.eos_token_id, prompt_length=ids.shape[1])
        generated = self.model.generate(
            input_ids=ids, attention_mask=mask, num_beams=self.full_width, num_return_sequences=self.full_width,
            do_sample=False, use_cache=True, max_new_tokens=self.answer_length, length_penalty=0.0,
            top_k=None, top_p=None, output_scores=False, return_dict_in_generate=False,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([processor]), **kwargs)
        decoded = self.tokenizer.batch_decode(generated[:, ids.shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        output = []
        for j in range(0, len(decoded), self.full_width):
            row = [sid.strip() for sid in decoded[j:j + self.full_width]]
            if len(set(row)) != self.full_width or any(sid not in self.tokens for sid in row):
                raise RuntimeError("Incomplete or invalid full beam output")
            output.append(row)
        self.counters["full_beam_requests"] += len(output)
        return output

    @torch.inference_mode()
    def beam(self, prompts):
        ids,mask = self._pad(prompts)
        self.counters["prefill_tokens"] += sum(len(p) for p in prompts)
        self.counters["prefill_requests"] += len(prompts)
        return self._beam(ids, mask)

    @torch.inference_mode()
    def beam_from_session(self, session, rows=None):
        indices = self._request_rows(session, rows)
        if not indices:
            return []
        rows_gpu = torch.tensor(indices, device=self.device)
        ids = session.ids.index_select(0, rows_gpu)
        mask = session.mask.index_select(0, rows_gpu)
        # HF takes the first generation decision from a model call. Replay only
        # the last prompt token, with a private, already beam-expanded KV.
        cache = _fork_cache(session.kv, rows_gpu.repeat_interleave(self.full_width))
        cache.crop(ids.shape[1] - 1)
        self.counters["beam_boundary_replay_tokens"] += len(indices) * self.full_width
        return self._beam(ids, mask, past_key_values=cache)

    @torch.inference_mode()
    def bruteforce(self, prompts, candidate_sids, *, per_token=False):
        if len(prompts) != len(candidate_sids):
            raise ValueError("Candidate groups do not align with prompts")
        result = [[] for _ in prompts]
        work = [(i, list(p), self.tokens[sid]) for i,(p,group) in enumerate(zip(prompts,candidate_sids))
                for sid in group]
        for offset in range(0, len(work), self.chunk_size):
            jobs = work[offset:offset + self.chunk_size]
            # Right ends align under left padding; the final three logit vectors
            # predict positions a,b,c when the supplied suffix is only [a,b].
            ids,mask = self._pad([p + list(path[:2]) for _,p,path in jobs])
            output = self.model(input_ids=ids, attention_mask=mask, position_ids=_position_ids(mask),
                                use_cache=False, logits_to_keep=3)
            labels = torch.tensor([path for _,_,path in jobs], device=self.device)
            full = output.logits.float().log_softmax(-1)
            values = full.gather(-1, labels.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            for (i,_,_),score in zip(jobs,values):
                result[i].append(score if per_token else sum(score) / len(score))
        return result
