"""Small Qwen model regression tests for prompt KV reuse and complete beam ranks."""
import importlib.util
import re
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class CatalogTokenizer:
    """Only the tokenizer operations needed by the verifier, without external files."""
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = None

    def __init__(self, atom_set):
        self.names = ["<pad>", self.eos_token, "H", "I", "\n", *sorted(atom_set)]
        self.ids = {s:i for i,s in enumerate(self.names)}

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        return self.ids

    def convert_tokens_to_ids(self, values):
        return self.ids[values] if isinstance(values,str) else [self.ids[v] for v in values]

    def encode(self, value, **kwargs):
        if value in self.ids:
            return [self.ids[value]]
        tokens = re.findall(r"<[abc]_\d+>|\n", value)
        if tokens and ''.join(tokens) == value:
            return self.convert_tokens_to_ids(tokens)
        return [2,3,2]

    def pad(self, rows, **kwargs):
        import torch
        width = max(len(row["input_ids"]) for row in rows)
        ids = [([self.pad_token_id] * (width-len(row["input_ids"])) + row["input_ids"]) for row in rows]
        masks = [[0] * (width-len(row["input_ids"])) + row["attention_mask"] for row in rows]
        return {"input_ids":torch.tensor(ids), "attention_mask":torch.tensor(masks)}

    def batch_decode(self, ids, **kwargs):
        return [''.join(self.names[t] for t in row if t!=1) for row in ids.cpu().tolist()]


@unittest.skipUnless(AVAILABLE, "requires Torch and the pinned Transformers runtime")
class VerifierTests(unittest.TestCase):
    def setUp(self):
        import torch
        from hybrid.verifier import Verifier
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(1718)
        catalog = {}
        for a in range(3):
            for b in range(3):
                for c in range(2):
                    catalog[len(catalog)] = [f"<a_{a}>",f"<b_{b}>",f"<c_{c}>"]
        tok = CatalogTokenizer({x for v in catalog.values() for x in v})
        config = Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=48,
                             num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                             bos_token_id=None,eos_token_id=1,pad_token_id=1)
        model = Qwen2ForCausalLM(config)
        self.v = Verifier.__new__(Verifier)
        self.v._configure(model,tok,catalog,"cpu",3,4)
        self.prompts = [[2,3,2],[2,2,3,2,3,3]]
        self.sids = list(self.v.tokens)

    def test_amortized_scores_match_independent_full_forwards(self):
        import torch
        s = self.v.prefill(self.prompts)
        paths = [self.sids[:12], self.sids[::-1]]
        actual = self.v.token_scores(s,paths)
        ref = self.v.bruteforce(self.prompts,paths,per_token=True)
        for a,b in zip(actual,ref):
            torch.testing.assert_close(torch.as_tensor(a),torch.as_tensor(b),atol=2e-5,rtol=0)
        before = self.v.counters.copy()
        self.assertEqual(self.v.score(s,[paths[1][:2]],rows=[1])[0],
                         [sum(row)/len(row) for row in actual[1][:2]])
        self.assertEqual(self.v.counters,before)
        self.assertEqual(s.kv.get_seq_length(),len(self.prompts[1]))

    def test_beam_cache_preserves_ranks_when_batch_subset_reordered(self):
        s = self.v.prefill(self.prompts)
        self.v.score(s,[self.sids,self.sids])
        self.assertEqual(self.v.beam_from_session(s,rows=[1,0]),self.v.beam(self.prompts[::-1]))
        self.assertEqual(self.v.beam_from_session(s,rows=[0]),self.v.beam(self.prompts[:1]))
        self.assertEqual(s.kv.get_seq_length(),len(self.prompts[1]))

    def test_prefixes_pruned_at_every_depth(self):
        s = self.v.prefill(self.prompts)
        one = self.v.prefixes(s,1,1)
        two = self.v.prefixes(s,1,2)
        self.assertTrue(all(s2[0].startswith(s1[0]) for s1,s2 in zip(one,two)))
        wide = self.v.prefixes(s,5,2)
        for row in wide:
            self.assertEqual(len(set(row)),5)
            self.assertTrue(all(any(full.startswith(p) for full in self.sids) for p in row))

if __name__ == "__main__":
    unittest.main()
