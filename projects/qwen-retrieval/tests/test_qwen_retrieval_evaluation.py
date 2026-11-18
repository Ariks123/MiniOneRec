"""CPU checks for causal candidate scoring, index identity and metric coverage."""
import json
import importlib.util
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from qwen_retrieval.config import ITEM_TEMPLATE, PROJECTION_DIM, TEMPLATE_VERSION, build_catalog, hash_file, json_digest
from qwen_retrieval.encode_items import load_index, load_joint_model
from qwen_retrieval.evaluate import Metrics, candidate_log_probabilities, merge_and_measure, score_candidates, score_candidates_bruteforce, sid_token_ids


class FakeTokenizer:
    padding_side = "left"
    eos_token_id = 31
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        mapping = {"<a_1>": [2], "<b_2>": [3], "<c_3>": [4], "<a_1><b_2><c_3>": [2, 3, 4]}
        return mapping[text]

    def pad(self, examples, **kwargs):
        import torch
        width = max(len(row["input_ids"]) for row in examples)
        return {"input_ids": torch.tensor([[31] * (width - len(row["input_ids"])) + row["input_ids"] for row in examples]),
                "attention_mask": torch.tensor([[0] * (width - len(row["input_ids"])) + row["attention_mask"] for row in examples])}


class NextTokenModel:
    def __init__(self):
        self.observed = []

    def __call__(self, input_ids, attention_mask, **kwargs):
        import torch
        self.observed.append((input_ids.clone(), attention_mask.clone()))
        logits = torch.zeros((*input_ids.shape, 32))
        for b in range(input_ids.shape[0]):
            for t in range(input_ids.shape[1] - 1):
                logits[b, t, input_ids[b, t + 1]] = 10.0
        return types.SimpleNamespace(logits=logits)


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers")),
                     "Torch/Transformers checks run in the fixed training container")
class CandidateScoringTests(unittest.TestCase):
    def test_left_padding_causal_shift_and_only_three_sid_tokens(self):
        import torch
        model, tokenizer = NextTokenModel(), FakeTokenizer()
        scores = score_candidates_bruteforce(model, tokenizer, [[8, 9], [9]], [[2, 3, 4], [2, 3, 4]],
                                  batch_size=2, device=torch.device("cpu"))
        expected = -math.log1p(31 * math.exp(-10))
        self.assertAlmostEqual(scores[0], expected, places=6)
        self.assertAlmostEqual(scores[1], expected, places=6)
        ids, mask = model.observed[0]
        self.assertEqual(ids.tolist(), [[8, 9, 2, 3, 4], [31, 9, 2, 3, 4]])
        self.assertEqual(mask.sum(dim=1).tolist(), [5, 4])

    def test_sid_encoding_excludes_eos_and_newline(self):
        self.assertEqual(sid_token_ids(FakeTokenizer(), "<a_1><b_2><c_3>"), [2, 3, 4])
        for malformed in ("<a_1><b_2>", "<a_1><b_2><c_3>\n", "<a_1><b_2><c_3><d_0>"):
            with self.assertRaises(ValueError):
                sid_token_ids(FakeTokenizer(), malformed)

    def test_partial_candidate_is_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            candidate_log_probabilities(torch.zeros(1, 5, 32), torch.tensor([[-100, -100, -100, 2, 3]]))

    def test_missing_projection_head_is_rejected(self):
        import torch
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch(
            "qwen_retrieval.model.QwenRetrievalForCausalLM.from_pretrained",
            return_value=(types.SimpleNamespace(), {"missing_keys": ["retrieval_head.weight"]}),
        ):
            with self.assertRaisesRegex(ValueError, "projection head"):
                load_joint_model("unused", torch.device("cpu"))

    def test_cached_scoring_matches_full_teacher_forcing_and_preserves_pair_order(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(3)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            attn_implementation="sdpa",
        )).eval()
        prompts = [[8, 9], [9], [8, 9], [9]]
        candidates = [[2, 3, 4], [4, 3, 2], [5, 6, 7], [2, 3, 4]]
        expected = score_candidates_bruteforce(model, FakeTokenizer(), prompts, candidates,
                                               batch_size=4, device=torch.device("cpu"))
        for chunk in (1, 2):
            actual = score_candidates(model, FakeTokenizer(), prompts, candidates,
                                      batch_size=chunk, device=torch.device("cpu"))
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-5, atol=1e-6)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy checks run in the fixed training container")
class IndexTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "Industrial_and_Scientific.index.json"
        self.catalog.write_text(json.dumps({"1": ["<a_1>", "<b_2>", "<c_3>"],
                                            "2": ["<a_1>", "<b_2>", "<c_3>"],
                                            "3": ["<a_4>", "<b_5>", "<c_6>"]}))
        self.rows = build_catalog(self.catalog)
        self.identity = {"files_sha256": {"model.safetensors": "joint-head-and-trunk"}, "sha256": "identity"}
        self.matrix = np.eye(2, PROJECTION_DIM, dtype=np.float32)
        np.save(self.root / "item_vectors.npy", self.matrix, allow_pickle=False)
        (self.root / "row_to_items.json").write_text(json.dumps(self.rows))
        self.manifest = {"status": "passed", "format_version": 1, "category": "Industrial_and_Scientific",
                         "checkpoint_identity": self.identity, "projection_dim": PROJECTION_DIM,
                         "template_version": TEMPLATE_VERSION, "item_template": ITEM_TEMPLATE,
                         "sid_index_sha256": hash_file(self.catalog), "catalog_rows_sha256": json_digest(self.rows),
                         "unique_sids": 2, "catalog_items": 3,
                         "matrix": {"filename": "item_vectors.npy", "sha256": hash_file(self.root / "item_vectors.npy"),
                                    "shape": [2, PROJECTION_DIM], "dtype": "float32", "normalized": True},
                         "mapping": {"filename": "row_to_items.json", "sha256": hash_file(self.root / "row_to_items.json")}}
        self.publish()

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_unique_sid_rows_preserve_all_original_ids(self):
        matrix, rows, manifest = load_index(self.root, self.catalog, self.identity)
        self.assertEqual(matrix.shape, (2, PROJECTION_DIM))
        self.assertEqual(rows[0]["item_ids"], ["1", "2"])
        self.assertEqual(manifest["catalog_items"], 3)

    def test_model_projection_or_tokenizer_identity_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "model/projection/tokenizer"):
            load_index(self.root, self.catalog, {"sha256": "different"})

    def test_template_mismatch_fails(self):
        self.manifest["item_template"] = "Wrong template {sid}"
        self.publish()
        with self.assertRaisesRegex(ValueError, "template"):
            load_index(self.root, self.catalog, self.identity)

    def test_modified_vector_bytes_fail(self):
        with (self.root / "item_vectors.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_index(self.root, self.catalog, self.identity)

    def test_reordered_mapping_fails_even_with_updated_file_checksum(self):
        (self.root / "row_to_items.json").write_text(json.dumps(list(reversed(self.rows))))
        self.manifest["mapping"]["sha256"] = hash_file(self.root / "row_to_items.json")
        self.publish()
        with self.assertRaisesRegex(ValueError, "row order"):
            load_index(self.root, self.catalog, self.identity)


class MetricTests(unittest.TestCase):
    def test_recall200_is_candidate_coverage_and_ndcg_is_order_sensitive(self):
        metric = Metrics()
        metric.add("target", [str(index) for index in range(100)] + ["target"])
        result = metric.result()
        self.assertEqual(result["Recall@200"], 1)
        self.assertEqual(result["Recall@50"], 0)
        self.assertAlmostEqual(result["NDCG@200"], 1 / math.log2(102))

    def test_merge_rejects_incomplete_and_duplicate_distributed_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.jsonl"
            sids = [str(index) for index in range(60)]
            row = {"row_id": 0, "target": "0", "predictions": sids[:50], "retrieval_predictions": sids[:50],
                   "groups": ["behavior_cold", "collision_free"]}
            path.write_text(json.dumps(row) + "\n")
            examples = [{"target": "0", "groups": row["groups"]}, {"target": "1", "groups": ["behavior_warm", "collision_free"]}]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                merge_and_measure([path], examples, set(sids), 50, Path(directory) / "merged.jsonl")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge_and_measure([path, path], examples, set(sids), 50, Path(directory) / "merged.jsonl")

    def test_merge_produces_sid_level_cold_and_collision_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.jsonl"
            sids = [str(index) for index in range(60)]
            row = {"row_id": 0, "target": "0", "predictions": sids[:50], "retrieval_predictions": sids[:50],
                   "groups": ["behavior_cold", "collision_affected"]}
            path.write_text(json.dumps(row) + "\n")
            groups = merge_and_measure([path], [{"target": "0", "groups": row["groups"]}], set(sids), 50,
                                       Path(directory) / "merged.jsonl")
            self.assertEqual(groups["behavior_cold"]["rows"], 1)
            self.assertEqual(groups["collision_affected"]["rerank_metrics"]["Recall@10"], 1)
            self.assertIsNone(groups["behavior_warm"]["rerank_metrics"]["NDCG@10"])


if __name__ == "__main__":
    unittest.main()
