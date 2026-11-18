"""CPU regressions for completion-relative catalog constraints.

The small tokenizer fixture deliberately splits a semantic ID across tokens.
Generation uses a real, randomly initialized GPT-2; no weights or tokenizers
are downloaded, and these tests make no recommendation-quality claims.
"""
import ast
from pathlib import Path
import sys
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LogitProcessor import ConstrainedLogitsProcessor, build_prefix_allowed_tokens


class CatalogTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    pieces = {
        2: "\n", 3: "<a_", 4: "1><b_2>", 5: "<c_3>", 6: "<c_4>",
        7: "<a_2>", 8: "<b_3>", 9: "<a_3>",
    }
    encodings = {
        "<a_1><b_2><c_3>\n": [3, 4, 5, 2],
        "<a_1><b_2><c_4>\n": [3, 4, 6, 2],
        "<a_2><b_3>\n": [7, 8, 2],
        "<a_3>\n": [9, 2],
    }

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        if add_special_tokens:
            raise AssertionError("answer constraints must not insert BOS/chat tokens")
        return list(self.encodings[text])

    def decode(self, ids):
        return "".join(self.pieces[int(value)] for value in ids
                       if int(value) not in (self.eos_token_id, self.pad_token_id))


class PreferInvalidToken(LogitsProcessor):
    """Make a token outside the catalog overwhelmingly attractive."""
    def __call__(self, input_ids, scores):
        scores = torch.zeros_like(scores)
        scores[:, 31] = 1000
        return scores


class ConstrainedDecodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CatalogTokenizer()
        self.candidates = list(self.tokenizer.encodings)
        self.table = build_prefix_allowed_tokens(self.tokenizer, self.candidates)

    def processor(self, prompt_length, beams=1, table=None):
        table = self.table if table is None else table
        return ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda batch, ids: table.get(tuple(ids), []),
            num_beams=beams, eos_token_id=1, prompt_length=prompt_length,
        )

    def allowed(self, processor, rows):
        ids = torch.tensor(rows, dtype=torch.long)
        result = processor(ids, torch.arange(32, dtype=torch.float).expand(len(rows), -1).clone())
        return [torch.isfinite(row).nonzero(as_tuple=True)[0].tolist() for row in result]

    def test_catalog_is_completion_only_and_preserves_subword_paths(self):
        self.assertEqual(set(self.table[()]), {3, 7, 9})
        self.assertEqual(set(self.table[(3, 4)]), {5, 6})
        self.assertEqual(self.table[(3, 4, 5, 2)], [1])
        self.assertEqual(self.table[(9, 2)], [1])
        self.assertTrue(all(not special for _, special in self.tokenizer.calls))
        self.assertTrue(all(isinstance(key, tuple) for key in self.table))

    def test_duplicate_candidates_do_not_duplicate_edges(self):
        table = build_prefix_allowed_tokens(self.tokenizer, self.candidates * 2)
        self.assertEqual(table, self.table)

    def test_prompt_headers_and_lengths_do_not_enter_the_constraint(self):
        for prompt in ([20], [20, 21, 22], [25, 26, 27, 28, 20, 21, 22]):
            with self.subTest(prompt=prompt):
                processor = self.processor(len(prompt))
                self.assertEqual(set(self.allowed(processor, [prompt])[0]), {3, 7, 9})
                self.assertEqual(self.allowed(processor, [prompt + [3, 4]]), [[5, 6]])

    def test_left_padding_uses_tensor_width_and_preserves_batch_beam_groups(self):
        calls = []
        def callback(batch, ids):
            calls.append((batch, list(ids)))
            return self.table.get(tuple(ids), [])
        processor = ConstrainedLogitsProcessor(callback, 2, eos_token_id=1, prompt_length=4)
        self.assertEqual(self.allowed(processor, [
            [0, 0, 20, 21, 3], [0, 0, 20, 21, 7],
            [22, 23, 24, 25, 9], [22, 23, 24, 25, 3],
        ]), [[4], [8], [2], [4]])
        self.assertEqual(calls, [(0, [3]), (0, [7]), (1, [9]), (1, [3])])

    def test_repeated_or_skipped_steps_do_not_change_prefix(self):
        processor = self.processor(2)
        self.assertEqual(self.allowed(processor, [[20, 21, 3, 4]]), [[5, 6]])
        self.assertEqual(self.allowed(processor, [[20, 21, 3, 4]]), [[5, 6]])
        self.assertEqual(set(self.allowed(processor, [[20, 21]])[0]), {3, 7, 9})
        self.assertEqual(self.allowed(processor, [[20, 21, 7, 8, 2]]), [[1]])

    def test_separate_generation_calls_can_use_different_prompt_widths(self):
        short = self.processor(1)
        long = self.processor(6)
        self.assertEqual(self.allowed(short, [[20, 3, 4]]), [[5, 6]])
        self.assertEqual(self.allowed(long, [[20, 21, 22, 23, 24, 25, 7]]), [[8]])
        self.assertEqual(self.allowed(short, [[20, 9]]), [[2]])

    def test_terminal_candidate_requires_eos(self):
        self.assertEqual(self.allowed(self.processor(2), [[20, 21, 9, 2]]), [[1]])

    def test_finished_sampling_rows_can_keep_emitting_eos(self):
        self.assertEqual(self.allowed(self.processor(2), [[20, 21, 9, 2, 1]]), [[1]])
        self.assertEqual(self.allowed(self.processor(2), [[20, 21, 9, 2, 1, 0, 0]]), [[1]])
        with self.assertRaisesRegex(ValueError, "EOS after an incomplete catalog prefix"):
            self.allowed(self.processor(2), [[20, 21, 3, 1]])

    def test_unknown_prefix_raises_instead_of_returning_a_truncated_item(self):
        with self.assertRaises(ValueError):
            self.allowed(self.processor(2), [[20, 21, 31]])

    def test_empty_constraint_callback_raises(self):
        with self.assertRaises(ValueError):
            self.allowed(self.processor(2, table={}), [[20, 21]])

    def test_prompt_boundary_is_required(self):
        with self.assertRaises((TypeError, ValueError)):
            ConstrainedLogitsProcessor(lambda batch, ids: [3], 1, eos_token_id=1)

    def test_scores_are_masked_without_renormalizing_catalog_probabilities(self):
        original = torch.arange(32, dtype=torch.float).unsqueeze(0)
        result = self.processor(2)(torch.tensor([[20, 21]]), original)
        expected = original.log_softmax(dim=-1)
        torch.testing.assert_close(result[:, [3, 7, 9]], expected[:, [3, 7, 9]])
        self.assertTrue(torch.isneginf(result[0, 31]))

    def test_real_transformers_generate_keeps_beams_and_samples_in_catalog(self):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=20, eos_token_id=1, pad_token_id=0,
        )).eval()
        model.generation_config.do_sample = True
        prompt_ids = torch.tensor([[0, 0, 20, 21], [22, 23, 24, 25]])
        prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        for candidates, beams, sample in ((self.candidates, 2, False),
                                          (self.candidates, 1, True),
                                          (self.candidates, 2, True),
                                          (self.candidates[:1], 4, False)):
            table = build_prefix_allowed_tokens(self.tokenizer, candidates)
            with self.subTest(catalog_size=len(candidates), num_beams=beams, do_sample=sample), torch.no_grad():
                output = model.generate(
                    input_ids=prompt_ids, attention_mask=prompt_mask,
                    num_beams=beams, num_return_sequences=3 if beams == 1 else beams,
                    do_sample=sample, max_new_tokens=8, top_k=0 if sample else None,
                    logits_processor=LogitsProcessorList([
                        PreferInvalidToken(), self.processor(prompt_ids.shape[1], beams=beams, table=table),
                    ]),
                )
                completions = output[:, prompt_ids.shape[1]:]
                decoded = [self.tokenizer.decode(row).strip() for row in completions]
                catalog = {candidate.strip() for candidate in candidates}
                invalid_count = sum(item not in catalog for item in decoded)
                self.assertEqual(invalid_count, 0, decoded)
                self.assertTrue(all(1 in row.tolist() for row in completions))


    def test_sparse_beam_sampling_never_silently_returns_invalid_items(self):
        # Transformers beam sampling can fill zero-probability slots with tokens
        # outside the catalog. The processor must expose that inconsistency;
        # production beam generation explicitly selects do_sample=False.
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=20, eos_token_id=1, pad_token_id=0,
        )).eval()
        table = build_prefix_allowed_tokens(self.tokenizer, self.candidates[:1])
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids=torch.tensor([[20, 21]]),
                    attention_mask=torch.tensor([[1, 1]]),
                    num_beams=4, num_return_sequences=4, do_sample=True,
                    max_new_tokens=8, top_k=0,
                    logits_processor=LogitsProcessorList([
                        PreferInvalidToken(), self.processor(2, beams=4, table=table),
                    ]),
                )
        except ValueError as error:
            self.assertIn("No valid catalog continuation", str(error))
        else:
            # Upstream implementations may avoid filling the empty beam slots.
            # Success is also valid if every returned answer is in the catalog.
            decoded = [self.tokenizer.decode(row[2:]).strip() for row in output]
            self.assertEqual(decoded, [self.candidates[0].strip()] * len(decoded))

    def test_trainer_beam_and_test_calls_explicitly_disable_sampling(self):
        tree = ast.parse((ROOT / "minionerec_trainer.py").read_text(encoding="utf-8"))
        found = set()
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "generate" or not call.args:
                continue
            kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
            config = kwargs.get("generation_config")
            if not isinstance(config, ast.Attribute):
                continue
            if config.attr == "test_generation_config":
                found.add("test")
            elif (config.attr == "generation_config"
                  and isinstance(call.args[0], ast.Name)
                  and call.args[0].id == "dedup_prompt_ids"):
                found.add("beam")
            else:
                continue
            self.assertIsInstance(kwargs.get("do_sample"), ast.Constant)
            self.assertIs(kwargs["do_sample"].value, False)
        self.assertEqual(found, {"beam", "test"})


if __name__ == "__main__":
    unittest.main()
