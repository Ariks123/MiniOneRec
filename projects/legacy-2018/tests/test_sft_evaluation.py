"""Ranking and artifact-integrity tests; no GPU, model weights or ML imports."""
import ast
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.sft_eval_utils import (
    APPROVED_VIDEO_GAMES_SID_SHA256, CATEGORY_PROMPTS, EvaluationError, RankingAccumulator,
    atomic_json, merge_prediction_shards, read_targets, resolve_expected_sid_sha256,
    sha256_file, verify_artifacts, verify_sid_tokenizer,
)


class RankingTests(unittest.TestCase):
    def test_exact_hr_and_discounted_gain_with_misses(self):
        accumulator = RankingAccumulator({"A", "B", "C", "D"}, 3)
        for row_id, target, predictions in (
            (0, "A", ["A", "B", "C"]),
            (1, "B", ["A", "C", "B"]),
            (2, "D", ["A", "B", "C"]),
        ):
            accumulator.add({"row_id": row_id, "target": target, "predictions": predictions})
        result = accumulator.result()
        self.assertAlmostEqual(result["metrics"]["HR@1"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["HR@3"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["NDCG@3"], (1 + 1 / math.log2(4)) / 3)
        self.assertEqual(result["CC"], 0)
        self.assertEqual(result["validity_rate"], 1)
        self.assertEqual(result["unique_rate"], 1)
        self.assertEqual(result["examples"]["hit"][1]["rank"], 3)
        self.assertIsNone(result["examples"]["miss"][0]["rank"])

    def test_invalid_truncated_duplicate_and_wrong_width_candidates_fail(self):
        for predictions, message in ((["A", "B", "Z"], "invalid"),
                                     (["A", "B", "A"], "duplicate"),
                                     (["A", "B"], "expected")):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(EvaluationError, message):
                RankingAccumulator({"A", "B", "C"}, 3).add(
                    {"row_id": 0, "target": "A", "predictions": predictions})

    def test_unknown_target_and_empty_evaluation_fail(self):
        with self.assertRaisesRegex(EvaluationError, "unknown target"):
            RankingAccumulator({"A", "B"}, 2).add({"row_id": 0, "target": "Z", "predictions": ["A", "B"]})
        with self.assertRaisesRegex(EvaluationError, "No predictions"):
            RankingAccumulator({"A"}, 1).result()


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def shard(self, name, rows):
        path = self.directory / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def row(self, row_id, target):
        return {"row_id": row_id, "target": target, "predictions": ["A", "B", "C"]}

    def test_rank_modulo_shards_merge_into_original_row_order(self):
        paths = [self.shard("rank0", [self.row(0, "A"), self.row(2, "D")]),
                 self.shard("rank1", [self.row(1, "B")])]
        destination = self.directory / "predictions.jsonl"
        result = merge_prediction_shards(paths, ["A", "B", "D"], {"A", "B", "C", "D"}, 3, destination)
        self.assertEqual([row["row_id"] for row in map(json.loads, destination.read_text().splitlines())], [0, 1, 2])
        self.assertEqual(result["coverage"], {"expected_rows": 3, "completed_rows": 3,
                                            "missing_rows": 0, "duplicate_rows": 0})
        self.assertFalse(destination.with_name("predictions.jsonl.partial").exists())

    def test_missing_duplicate_and_mismatched_target_are_rejected(self):
        scenarios = [
            ([self.row(0, "A"), self.row(2, "C")], [], "coverage"),
            ([self.row(0, "A")], [self.row(0, "A")], "coverage"),
            ([self.row(0, "B")], [self.row(1, "B"), self.row(2, "C")], "target differs"),
            ([self.row(0, "A")], [self.row(1, "B")], "coverage"),
        ]
        for index, (first, second, message) in enumerate(scenarios):
            with self.subTest(index=index), self.assertRaisesRegex(EvaluationError, message):
                merge_prediction_shards(
                    [self.shard("first", first), self.shard("second", second)],
                    ["A", "B", "C"], {"A", "B", "C"}, 3, self.directory / f"invalid-{index}.jsonl")
            self.assertFalse((self.directory / f"invalid-{index}.jsonl").exists())
            self.assertTrue((self.directory / f"invalid-{index}.jsonl.partial").exists())

    def test_each_shard_must_have_strictly_ascending_integer_ids(self):
        for ids in ((1, 0), (0, 0), (True,)):
            with self.subTest(ids=ids), self.assertRaises(EvaluationError):
                merge_prediction_shards([self.shard("bad", [self.row(index, "A") for index in ids])],
                                        ["A", "A"], {"A", "B", "C"}, 3, self.directory / "bad.jsonl")

    def test_distributed_merge_waits_for_closed_prediction_files(self):
        # A four-GPU regression exposed that asynchronous NCCL all_reduce is not
        # a CPU filesystem barrier. Guard the close -> blocking barrier -> merge
        # ordering even when the test shell has no Torch/CUDA runtime installed.
        entrypoint = Path(__file__).resolve().parents[1] / "scripts/evaluate_sft.py"
        tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "evaluate")
        shard_writer = next(node for node in ast.walk(function) if isinstance(node, ast.With)
                            and any(isinstance(item.optional_vars, ast.Name)
                                    and item.optional_vars.id == "destination" for item in node.items))
        merge = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "merge_prediction_shards")
        barriers = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "barrier"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "dist"
                    and shard_writer.end_lineno < node.lineno < merge.lineno]
        self.assertTrue(barriers, "Rank zero must wait for all closed prediction files before merging")
        self.assertTrue(all(not any(keyword.arg == "async_op" and isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value is True for keyword in node.keywords)
                            for node in barriers), "The filesystem boundary must block the CPU caller")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw, self.data, self.model = [self.directory / name for name in ("raw", "data", "model")]
        for path in (self.raw, self.data, self.model):
            path.mkdir()
        self.sid = self.raw / "Video_Games.index.json"
        self.category = "Video_Games"
        atomic_json(self.sid, {"0": ["<a_0>", "<b_0>", "<c_0>", "<d_0>"]})
        self.metadata = self.raw / "Video_Games.item.json"
        atomic_json(self.metadata, {"0": {"title": "A"}})
        self.sid_hash = sha256_file(self.sid)
        self.provenance = {"stage": "sft", "category": "Video_Games", "sid_index_sha256": self.sid_hash,
                           "item_meta_sha256": sha256_file(self.metadata), "source_model": "fixture-only"}
        atomic_json(self.model / "posttraining_provenance.json", self.provenance)
        atomic_json(self.model / "config.json", {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 10})
        (self.model / "model.safetensors").write_bytes(b"fixture never loaded")
        for split in ("train", "valid", "test"):
            with (self.data / f"{split}.csv").open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(["item_sid", "history_item_sid"])
                writer.writerow(["<a_0><b_0><c_0><d_0>", "[]"])
        (self.data / "info.txt").write_text("<a_0><b_0><c_0><d_0>\tA\t0\n")
        self.manifest = {
            "validation": "passed", "check_only": False, "dataset": "Video_Games", "selected_sid_epoch": 2900,
            "preserve_all_interaction_rows": True, "sampling_applied": False, "keep_longest_only": False,
            "rows": {"train": 1, "valid": 1, "test": 1},
            "sources": {"sid_index": {"filename": self.sid.name, "sha256": self.sid_hash},
                        "item_metadata": {"filename": self.metadata.name, "sha256": sha256_file(self.metadata)}},
            "outputs": {name: {"sha256": sha256_file(self.data / name)}
                        for name in ("train.csv", "valid.csv", "test.csv", "info.txt")},
        }
        atomic_json(self.data / "preparation_manifest.json", self.manifest)

    def verify(self):
        return verify_artifacts(self.model, self.raw, self.data, self.category, self.sid_hash)

    def industrial_fixture(self, collisions=True, width=3):
        self.category = "Industrial_and_Scientific"
        self.sid = self.raw / f"{self.category}.index.json"
        self.metadata = self.raw / f"{self.category}.item.json"
        catalog = {item_id: [f"<{layer}_{code}>" for layer in "abcd"[:width]]
                   for item_id, code in (("4", 0), ("12", 0 if collisions else 1), ("90", 2))}
        atomic_json(self.sid, catalog)
        atomic_json(self.metadata, {item_id: {"title": f"Item {item_id}"} for item_id in catalog})
        self.sid_hash = sha256_file(self.sid)
        self.provenance = {"stage": "sft", "category": self.category, "sid_index_sha256": self.sid_hash,
                           "item_meta_sha256": sha256_file(self.metadata), "source_model": "fixture-only"}
        atomic_json(self.model / "posttraining_provenance.json", self.provenance)
        target = "".join(catalog["4"])
        for split in ("train", "valid", "test"):
            with (self.data / f"{split}.csv").open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(["item_sid", "history_item_sid"])
                writer.writerow([target, "[]"])
        (self.data / "info.txt").write_text("".join(f"{''.join(tokens)}\tItem {item_id}\t{item_id}\n"
                                                  for item_id, tokens in catalog.items()))
        self.manifest = {"validation": "passed", "check_only": False, "dataset": self.category,
                         "original_sid": True, "selected_sid_epoch": None,
                         "unique_sids": 2 if collisions else 3,
                         "sid_collisions": {"groups": 1 if collisions else 0, "items": 2 if collisions else 0,
                                            "excess_items": 1 if collisions else 0},
                         "preserve_all_interaction_rows": True, "sampling_applied": False, "keep_longest_only": False,
                         "rows": {"train": 1, "valid": 1, "test": 1},
                         "sources": {"sid_index": {"filename": self.sid.name, "sha256": self.sid_hash},
                                     "item_metadata": {"filename": self.metadata.name, "sha256": sha256_file(self.metadata)}},
                         "outputs": {name: {"sha256": sha256_file(self.data / name)}
                                     for name in ("train.csv", "valid.csv", "test.csv", "info.txt")}}
        for split in ("train", "valid", "test"):
            source = self.raw / f"{self.category}.{split}.inter"
            source.write_text("user_id:token\titem_id_list:token_seq\titem_id:token\n5\t90\t4\n")
            self.manifest["sources"][split] = {"filename": source.name, "sha256": sha256_file(source)}
        atomic_json(self.data / "preparation_manifest.json", self.manifest)
        return catalog

    def test_matching_checkpoint_and_full_prepared_split_pass(self):
        result = self.verify()
        self.assertEqual(result["test_targets"], ["<a_0><b_0><c_0><d_0>"])
        self.assertEqual(len(result["catalog"]), 1)
        self.assertEqual(result["source_sha256"], {
            "train_file": sha256_file(self.data / "train.csv"),
            "test_file": sha256_file(self.data / "test.csv"),
            "sid_index_path": self.sid_hash,
        })

    def amazon23_fixture(self):
        self.manifest = {**self.manifest, 'dataset': 'Amazon Reviews 2023 Video_Games',
                         'history_length_range': [1, 50], 'complete_sids_unique': True,
                         'complete_sid_tokens': 4,
                         'source_sha256': {self.sid.name: self.sid_hash,
                                           self.metadata.name: sha256_file(self.metadata)}}
        # Amazon23 provenance deliberately does not impersonate the old epoch2900 run.
        for key in ('check_only', 'selected_sid_epoch', 'keep_longest_only', 'sources'):
            self.manifest.pop(key, None)
        atomic_json(self.data / 'preparation_manifest.json', self.manifest)

    def test_amazon23_history50_uses_its_own_catalog_without_epoch2900_assumption(self):
        self.amazon23_fixture()
        result = self.verify()
        self.assertIn('Amazon23', result['metric_semantics'])
        self.assertEqual(result['valid_targets'], result['test_targets'])

    def test_amazon23_rejects_old_history_or_changed_catalog_identity(self):
        self.amazon23_fixture()
        self.manifest['history_length_range'] = [1, 10]
        atomic_json(self.data / 'preparation_manifest.json', self.manifest)
        with self.assertRaisesRegex(EvaluationError, 'history-50'):
            self.verify()
        self.manifest['history_length_range'] = [1, 50]
        self.manifest['source_sha256'][self.sid.name] = '0' * 64
        atomic_json(self.data / 'preparation_manifest.json', self.manifest)
        with self.assertRaisesRegex(EvaluationError, 'catalog differs'):
            self.verify()

    def test_industrial_amazon23_routes_to_unique_four_token_history50_protocol(self):
        self.industrial_fixture(collisions=False, width=4)
        self.manifest.update(dataset='Amazon Reviews 2023 Industrial_and_Scientific',
                             history_length_range=[1, 50], complete_sids_unique=True, complete_sid_tokens=4,
                             source_sha256={self.sid.name: self.sid_hash,
                                            self.metadata.name: sha256_file(self.metadata)})
        self.manifest.pop('original_sid')
        atomic_json(self.data / 'preparation_manifest.json', self.manifest)
        result = self.verify()
        self.assertEqual(result['sid_tokens_per_item'], 4)
        self.assertEqual(result['unique_sids'], 3)
        self.assertIn('Amazon23', result['metric_semantics'])
    def test_wrong_provenance_and_modified_inputs_fail(self):
        for key in ("sid_index_sha256", "item_meta_sha256"):
            record = {**self.provenance, key: "wrong"}
            atomic_json(self.model / "posttraining_provenance.json", record)
            with self.subTest(key=key), self.assertRaisesRegex(EvaluationError, "another|other"):
                self.verify()
        atomic_json(self.model / "posttraining_provenance.json", self.provenance)
        with (self.data / "test.csv").open("a") as destination:
            destination.write("changed\n")
        with self.assertRaisesRegex(EvaluationError, "Prepared output changed"):
            self.verify()

    def test_missing_model_shard_and_wrong_manifest_counts_fail(self):
        atomic_json(self.model / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
        with self.assertRaisesRegex(EvaluationError, "weight shards"):
            self.verify()
        (self.model / "model.safetensors.index.json").unlink()
        self.manifest["rows"]["test"] = 2
        atomic_json(self.data / "preparation_manifest.json", self.manifest)
        with self.assertRaisesRegex(EvaluationError, "row counts"):
            self.verify()

    def test_atomic_sid_token_and_model_dimensions_are_checked(self):
        catalog = {"0": ["a", "b", "c", "d"]}
        class Tokenizer:
            def get_vocab(self):
                return {"a": 1, "b": 2, "c": 3, "d": 4}
            def encode(self, token, add_special_tokens=False):
                return [self.get_vocab()[token]]
            def __len__(self):
                return 6
        embedding = SimpleNamespace(weight=SimpleNamespace(shape=(10, 4)))
        model = SimpleNamespace(config=SimpleNamespace(vocab_size=10), get_input_embeddings=lambda: embedding,
                                get_output_embeddings=lambda: embedding)
        tokenizer = Tokenizer()
        self.assertEqual(verify_sid_tokenizer(tokenizer, model, catalog)["atomic_sid_tokens"], 4)
        tokenizer.encode = lambda *args, **kwargs: [1, 2]
        with self.assertRaisesRegex(EvaluationError, "atomic"):
            verify_sid_tokenizer(tokenizer, model, catalog)
        tokenizer = Tokenizer()
        model.config.vocab_size = 9
        with self.assertRaisesRegex(EvaluationError, "dimensions"):
            verify_sid_tokenizer(tokenizer, model, catalog)

    def test_original_industrial_preserves_noncontiguous_ids_and_sid_collisions(self):
        catalog = self.industrial_fixture()
        result = self.verify()
        self.assertEqual(result["catalog"], catalog)
        self.assertEqual(set(result["catalog"]), {"4", "12", "90"})
        self.assertEqual(result["catalog_items"], 3)
        self.assertEqual(result["unique_sids"], 2)
        self.assertEqual(result["collision_groups"], 1)
        self.assertEqual(result["collision_items"], 2)
        self.assertEqual(result["sid_tokens_per_item"], 3)
        self.assertEqual(result["test_targets"], ["<a_0><b_0><c_0>"])
        self.assertIn("colliding item IDs indistinguishable", result["metric_semantics"])

    def test_industrial_optional_collision_token_and_pinned_hash_are_checked(self):
        self.industrial_fixture(collisions=False, width=4)
        self.assertEqual(self.verify()["sid_tokens_per_item"], 4)
        with self.assertRaisesRegex(EvaluationError, "approved catalog"):
            verify_artifacts(self.model, self.raw, self.data, self.category, "f" * 64)
        with self.assertRaisesRegex(EvaluationError, "explicit"):
            verify_artifacts(self.model, self.raw, self.data, self.category, None)
        catalog = json.loads(self.sid.read_text())
        catalog["90"] = catalog["90"][:3]
        atomic_json(self.sid, catalog)
        self.sid_hash = sha256_file(self.sid)
        with self.assertRaisesRegex(EvaluationError, "consistent"):
            self.verify()

    def test_industrial_original_identity_and_collision_declarations_cannot_be_bypassed(self):
        for key, value, message in (("original_sid", False, "original SID"),
                                    ("unique_sids", 3, "unique count"),
                                    ("sid_collisions", {"groups": 0, "items": 0, "excess_items": 0}, "collision metadata")):
            with self.subTest(key=key):
                self.industrial_fixture()
                self.manifest[key] = value
                atomic_json(self.data / "preparation_manifest.json", self.manifest)
                with self.assertRaisesRegex(EvaluationError, message):
                    self.verify()

    def test_industrial_modified_source_or_checkpoint_category_is_rejected(self):
        self.industrial_fixture()
        atomic_json(self.model / "posttraining_provenance.json", {**self.provenance, "category": "Video_Games"})
        with self.assertRaisesRegex(EvaluationError, "category"):
            self.verify()
        atomic_json(self.model / "posttraining_provenance.json", self.provenance)
        (self.raw / f"{self.category}.train.inter").write_text("changed")
        with self.assertRaisesRegex(EvaluationError, "Source changed"):
            self.verify()

    def test_video_games_still_requires_four_unique_sid_tokens(self):
        atomic_json(self.sid, {"0": ["<a_0>", "<b_0>", "<c_0>"]})
        self.sid_hash = sha256_file(self.sid)
        with self.assertRaisesRegex(EvaluationError, "four SID tokens"):
            self.verify()
        tokens = ["<a_0>", "<b_0>", "<c_0>", "<d_0>"]
        atomic_json(self.sid, {"0": tokens, "1": tokens})
        atomic_json(self.metadata, {"0": {"title": "A"}, "1": {"title": "B"}})
        self.sid_hash = sha256_file(self.sid)
        with self.assertRaisesRegex(EvaluationError, "not unique"):
            self.verify()


class EvaluationArgumentsTests(unittest.TestCase):
    def test_default_video_games_hash_and_exact_training_prompt_categories(self):
        self.assertEqual(resolve_expected_sid_sha256("Video_Games"), APPROVED_VIDEO_GAMES_SID_SHA256)
        self.assertEqual(CATEGORY_PROMPTS["Video_Games"], "video games")
        self.assertEqual(CATEGORY_PROMPTS["Industrial_and_Scientific"], "industrial and scientific items")

    def test_industrial_missing_pin_fails_before_model_framework_import(self):
        from scripts.evaluate_sft import evaluate, parser
        args = parser().parse_args(["--model-path", "/tmp/model", "--input-dir", "/tmp/raw",
                                   "--data-dir", "/tmp/data", "--output-dir", "/tmp/output",
                                   "--category", "Industrial_and_Scientific"])
        with self.assertRaisesRegex(EvaluationError, "explicit --expected-sid-sha256"):
            evaluate(args)
        args.expected_sid_sha256 = "A" * 64
        self.assertEqual(resolve_expected_sid_sha256(args.category, args.expected_sid_sha256), "a" * 64)

    def test_original_eval_prompt_is_explicit_and_video_games_default_stays_training(self):
        from scripts.evaluate_sft import parser
        arguments = ["--model-path", "/tmp/model", "--input-dir", "/tmp/raw",
                     "--data-dir", "/tmp/data", "--output-dir", "/tmp/output"]
        self.assertEqual(parser().parse_args(arguments).prompt_style, "training")
        args = parser().parse_args(arguments + ["--category", "Industrial_and_Scientific",
                                               "--expected-sid-sha256", "a" * 64,
                                               "--prompt-style", "original-eval"])
        self.assertEqual(args.prompt_style, "original-eval")

    def test_training_and_original_eval_use_source_prompts_without_target_answer(self):
        # Execute only the pure prompt-building methods; no Torch/pandas imports.
        source = Path(__file__).resolve().parents[1] / "data.py"
        tree = ast.parse(source.read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        def method(class_name, name):
            function = next(node for node in classes[class_name].body
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            namespace = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
            return namespace[name]
        generate_prompt = method("BaseDataset", "generate_prompt")
        prompts = {}
        for class_name in ("SidSFTDataset", "EvalSidDataset"):
            history = method(class_name, "get_history")(None, {
                "history_item_sid": "['<a_1><b_1><c_1>']", "item_sid": "<a_9><b_9><c_9>"})
            history["output"] = ""
            prompts[class_name] = generate_prompt(None, history)
            self.assertIn("<a_1><b_1><c_1>", prompts[class_name])
            self.assertNotIn("<a_9><b_9><c_9>", prompts[class_name])
        self.assertIn("The user has interacted with items", prompts["SidSFTDataset"])
        self.assertIn("given the following chronological interaction history:", prompts["EvalSidDataset"])
        self.assertNotEqual(prompts["SidSFTDataset"], prompts["EvalSidDataset"])


if __name__ == "__main__":
    unittest.main()
