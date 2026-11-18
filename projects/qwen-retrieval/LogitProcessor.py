from typing import Callable, Iterable, List, Optional

import torch
from transformers.generation import LogitsProcessor


def build_prefix_allowed_tokens(tokenizer, candidates: Iterable[str]):
    """Index complete answers, without a prompt header, BOS, or chat template.

    Candidates include their training-time answer formatting (e.g. SID + newline).
    Use the same tokenizer as generation, including any added SID tokens. The empty
    tuple is the root; every complete answer is followed by the tokenizer's EOS.
    SID strings may contain any number of tokenizer tokens.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("Constrained decoding requires a tokenizer EOS token")
    prefixes = {}
    for candidate in candidates:
        if not candidate.strip():
            raise ValueError("Constrained decoding candidates must not be empty")
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if not tokens or tokenizer.eos_token_id in tokens:
            raise ValueError("Candidates must tokenize to nonempty answers without EOS")
        tokens = list(tokens) + [tokenizer.eos_token_id]
        for index, token_id in enumerate(tokens):
            prefixes.setdefault(tuple(tokens[:index]), set()).add(token_id)
    if not prefixes:
        raise ValueError("Constrained decoding requires a nonempty item catalog")
    return {prefix: sorted(tokens) for prefix, tokens in prefixes.items()}


class ConstrainedLogitsProcessor(LogitsProcessor):
    """Constrain only tokens generated after this call's padded prompt boundary.

    Create a processor for each generate() call and pass input_ids.shape[1] as
    prompt_length. Neither model names, prompt suffixes nor call counters locate
    that boundary reliably. base_model remains accepted for caller compatibility.
    """

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, List[int]], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        prompt_length: Optional[int] = None,
    ):
        if not isinstance(prompt_length, int) or prompt_length < 0:
            raise ValueError("Pass the padded generation input width as prompt_length")
        if not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.eos_token_id = eos_token_id
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("input_ids is shorter than the recorded prompt_length")
        if input_ids.shape[0] != scores.shape[0] or input_ids.shape[0] % self._num_beams:
            raise ValueError("Input rows must match scores and contain complete beam groups")

        # Preserve full-vocabulary log probabilities for the existing beam ranking.
        # Normalizing after masking would rank by conditional probabilities instead.
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        completions = input_ids[:, self.prompt_length:].tolist()
        for row, completion in enumerate(completions):
            batch_id = row // self._num_beams
            # Finished sampling rows and completed beams can remain in the batch.
            if self.eos_token_id is not None and self.eos_token_id in completion:
                eos_index = completion.index(self.eos_token_id)
                allowed = self._prefix_allowed_tokens_fn(batch_id, completion[:eos_index])
                if self.eos_token_id not in allowed:
                    raise ValueError(f"EOS after an incomplete catalog prefix: {completion}")
                allowed = [self.eos_token_id]
            else:
                allowed = self._prefix_allowed_tokens_fn(batch_id, completion)
            if not allowed:
                raise ValueError(
                    f"No valid catalog continuation for batch {batch_id}, "
                    f"completion {completion}, prompt_length={self.prompt_length}. "
                    "Check that generation and the catalog use the same tokenizer."
                )
            mask[row, allowed] = 0
        return scores + mask
