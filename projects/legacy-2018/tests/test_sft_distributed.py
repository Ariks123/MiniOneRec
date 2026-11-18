"""Exercise SFT launch behavior without importing ML libraries or loading a model."""
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "sft.py"


class Rows(list):
    def shuffle(self, **kwargs):
        return self

    def select(self, indices):
        return Rows(self[index] for index in indices)


class DatasetStub:
    @staticmethod
    def from_dict(values):
        keys = list(values)
        return Rows(
            {key: values[key][index] for key in keys}
            for index in range(len(values[keys[0]]))
        )


class SFTDistributedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile the actual train function; skip module imports and its Fire entry point.
        tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"), filename=str(SFT_PATH))
        train = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "write_sid_provenance")
        cls.provenance_code = compile(ast.Module(body=[helper], type_ignores=[]), str(SFT_PATH), "exec")
        cls.train_code = compile(
            ast.Module(body=[train], type_ignores=[]), str(SFT_PATH), "exec"
        )

    def run_train(self, env=None, *, cuda=True, primary=False, **overrides):
        trace = SimpleNamespace(
            events=[], loads=[], arguments=None, saves=[], tokenizer_saves=[],
            categories=[], provenance_calls=[], trained=False, cache_at_trainer_creation=None, error=None,
            resize_device=None, model_device_at_trainer_creation=None,
        )
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        model_device = ["cpu"]

        def move_model(device):
            trace.events.append(("to", device))
            model_device[0] = device
            return model

        model.to = move_model
        def resize_embeddings(size):
            trace.resize_device = model_device[0]
            trace.events.append(("resize", size))
        model.resize_token_embeddings = resize_embeddings

        class TokenizerStub:
            eos_token = "<eos>"
            eos_token_id = 2
            added_tokens = 0

            def __len__(self):
                return 8 + self.added_tokens

            def add_tokens(self, tokens):
                self.added_tokens += len(tokens)

            def save_pretrained(self, target):
                trace.tokenizer_saves.append(target)

        def load_model(*args, **kwargs):
            trace.events.append(("load", None))
            trace.loads.append(kwargs)
            if "device_map" in kwargs:
                model_device[0] = ("cuda", kwargs["device_map"][""])
            return model

        def data_stub(**kwargs):
            trace.categories.append(kwargs["category"])
            return Rows([
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}
            ])

        def training_arguments(**kwargs):
            trace.arguments = kwargs
            return SimpleNamespace(**kwargs)

        class TrainerStub:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                trace.cache_at_trainer_creation = self.model.config.use_cache
                trace.model_device_at_trainer_creation = model_device[0]
                trace.events.append(("trainer", None))

            def train(self, **kwargs):
                trace.trained = True

            def save_model(self, target):
                trace.saves.append(target)

            def is_world_process_zero(self):
                return primary

        cuda_stub = SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 4,
            set_device=lambda rank: trace.events.append(("device", rank)),
        )
        namespace = {
            "os": SimpleNamespace(environ=dict(env or {}), path=os.path),
            "torch": SimpleNamespace(
                bfloat16="bf16", device=lambda *parts: parts, cuda=cuda_stub
            ),
            "set_seed": lambda seed: trace.events.append(("seed", seed)),
            "print": lambda *args, **kwargs: None,
            "AutoModelForCausalLM": SimpleNamespace(
                from_pretrained=load_model, from_config=load_model
            ),
            "AutoConfig": SimpleNamespace(
                from_pretrained=lambda path: SimpleNamespace()
            ),
            "AutoTokenizer": SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: TokenizerStub()
            ),
            "TokenExtender": lambda **kwargs: SimpleNamespace(get_new_tokens=lambda: ["<a_1>"]),
            "SidSFTDataset": data_stub,
            "SidItemFeatDataset": data_stub,
            "FusionSeqRecDataset": data_stub,
            "ConcatDataset": lambda parts: Rows(row for part in parts for row in part),
            "HFDataset": DatasetStub,
            "transformers": SimpleNamespace(
                Trainer=TrainerStub,
                TrainingArguments=training_arguments,
                DataCollatorForSeq2Seq=lambda *args, **kwargs: None,
            ),
            "EarlyStoppingCallback": lambda **kwargs: None,
            "write_sid_provenance": lambda *args: trace.provenance_calls.append(args),
        }
        exec(self.train_code, namespace)
        kwargs = dict(
            base_model="stub-causal-model", train_file="stub.csv", eval_file="stub.csv",
            output_dir="out", category="Video_Games", batch_size=1024,
            micro_batch_size=16, deepspeed="config/h20_zero2.json",
            monitoring=False, baseline_eval=False,
        )
        kwargs.update(overrides)
        try:
            namespace["train"](**kwargs)
        except Exception as error:
            trace.error = error
        return trace

    def assert_success(self, trace):
        if trace.error is not None:
            raise trace.error
        self.assertTrue(trace.trained)

    def test_four_card_batch_and_rank_local_model_loading(self):
        trace = self.run_train({"WORLD_SIZE": "4", "LOCAL_RANK": "2"})
        self.assert_success(trace)
        self.assertEqual(trace.arguments["gradient_accumulation_steps"], 16)
        self.assertEqual(trace.arguments["per_device_train_batch_size"], 16)
        self.assertTrue(trace.arguments["bf16"])
        self.assertEqual(trace.loads[0]["device_map"], {"": 2})
        self.assertEqual(trace.loads[0]["attn_implementation"], "sdpa")
        self.assertEqual(
            [event[0] for event in trace.events[:3]], ["device", "seed", "load"]
        )
        self.assertTrue(all(category == "video games" for category in trace.categories))

    def test_non_main_rank_enters_trainer_saves_but_does_not_write_tokenizer(self):
        trace = self.run_train({"WORLD_SIZE": "4", "LOCAL_RANK": "3"})
        self.assert_success(trace)
        self.assertEqual(trace.saves, ["out", os.path.join("out", "final_checkpoint")])
        self.assertEqual(trace.tokenizer_saves, [])
        self.assertEqual(trace.provenance_calls, [])

    def test_main_rank_writes_final_tokenizer_once(self):
        trace = self.run_train(
            {"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, primary=True
        )
        self.assert_success(trace)
        self.assertEqual(trace.saves, ["out", os.path.join("out", "final_checkpoint")])
        self.assertEqual(
            trace.tokenizer_saves, [os.path.join("out", "final_checkpoint")]
        )

    def test_checkpointing_deepspeed_and_attention_options_reach_trainer(self):
        trace = self.run_train({"WORLD_SIZE": "4", "LOCAL_RANK": "1"})
        self.assert_success(trace)
        self.assertEqual(trace.arguments["deepspeed"], "config/h20_zero2.json")
        self.assertTrue(trace.arguments["gradient_checkpointing"])
        self.assertFalse(trace.arguments["logging_nan_inf_filter"])
        self.assertEqual(trace.arguments["max_grad_norm"], 1.0)
        self.assertEqual(trace.arguments["save_total_limit"], 2)
        self.assertEqual(trace.arguments["report_to"], "none")
        self.assertEqual(
            trace.arguments["gradient_checkpointing_kwargs"], {"use_reentrant": False}
        )
        self.assertFalse(trace.cache_at_trainer_creation)
        override = self.run_train(
            {"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, primary=True,
            deepspeed=None, gradient_checkpointing=False, attn_implementation="eager",
            lr_scheduler_type="linear", save_total_limit=1,
        )
        self.assert_success(override)
        self.assertIsNone(override.arguments["deepspeed"])
        self.assertFalse(override.arguments["gradient_checkpointing"])
        self.assertIsNone(override.arguments["gradient_checkpointing_kwargs"])
        self.assertEqual(override.loads[0]["attn_implementation"], "eager")
        self.assertEqual(override.arguments["lr_scheduler_type"], "linear")
        self.assertEqual(override.arguments["save_total_limit"], 1)

    def test_single_process_keeps_existing_category_and_batch_behavior(self):
        trace = self.run_train(
            cuda=False, primary=True, category="Office_Products",
            batch_size=128, micro_batch_size=4, deepspeed=None,
        )
        self.assert_success(trace)
        self.assertEqual(trace.arguments["gradient_accumulation_steps"], 32)
        self.assertNotIn("device_map", trace.loads[0])
        self.assertFalse(any(event[0] == "device" for event in trace.events))
        self.assertTrue(all(category == "office products" for category in trace.categories))

    def test_scratch_model_is_moved_to_the_rank_local_device(self):
        trace = self.run_train(
            {"WORLD_SIZE": "4", "LOCAL_RANK": "1"}, train_from_scratch=True
        )
        self.assert_success(trace)
        self.assertNotIn("device_map", trace.loads[0])
        self.assertIn(("to", ("cuda", 1)), trace.events)

    def test_cpu_initialization_binds_rank_and_resizes_before_trainer_placement(self):
        with tempfile.TemporaryDirectory() as temporary:
            sid = Path(temporary) / "Industrial_and_Scientific.index.json"
            sid.write_text("{}")
            trace = self.run_train(
                {"WORLD_SIZE": "4", "LOCAL_RANK": "2"}, initialize_on_cpu=True,
                category="Industrial_and_Scientific", sid_index_path=str(sid),
                batch_size=64, micro_batch_size=8, deepspeed=None,
                gradient_checkpointing=False, lr_scheduler_type="linear", save_total_limit=1,
            )
        self.assert_success(trace)
        self.assertEqual(trace.events[:3], [("device", 2), ("seed", 42), ("load", None)])
        self.assertNotIn("device_map", trace.loads[0])
        self.assertEqual(trace.resize_device, "cpu")
        self.assertEqual(trace.model_device_at_trainer_creation, "cpu")
        self.assertLess(next(i for i, event in enumerate(trace.events) if event[0] == "resize"),
                        next(i for i, event in enumerate(trace.events) if event[0] == "trainer"))
        self.assertFalse(any(event[0] == "to" for event in trace.events))
        self.assertEqual(trace.arguments["gradient_accumulation_steps"], 2)
        self.assertIsNone(trace.arguments["deepspeed"])
        self.assertFalse(trace.arguments["gradient_checkpointing"])
        self.assertEqual(trace.arguments["save_total_limit"], 1)

    def test_cpu_scratch_initialization_defers_manual_cuda_move(self):
        trace = self.run_train(
            {"WORLD_SIZE": "4", "LOCAL_RANK": "1"}, train_from_scratch=True, initialize_on_cpu=True
        )
        self.assert_success(trace)
        self.assertNotIn("device_map", trace.loads[0])
        self.assertEqual(trace.model_device_at_trainer_creation, "cpu")
        self.assertFalse(any(event[0] == "to" for event in trace.events))

    def test_invalid_launch_configuration_fails_before_model_loading(self):
        cases = [
            ({"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, {"batch_size": 0}),
            ({"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, {"micro_batch_size": 0}),
            ({"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, {"batch_size": 65}),
            ({"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, {"batch_size": True}),
            ({"WORLD_SIZE": "0"}, {}),
            ({"WORLD_SIZE": "4"}, {}),
            ({"WORLD_SIZE": "4", "LOCAL_RANK": "4"}, {}),
        ]
        for env, overrides in cases:
            with self.subTest(env=env, overrides=overrides):
                trace = self.run_train(env, **overrides)
                self.assertIsInstance(trace.error, ValueError)
                self.assertFalse(trace.loads)
                self.assertFalse(trace.trained)

    def test_main_rank_passes_the_original_category_and_sources_to_provenance(self):
        trace = self.run_train(
            {"WORLD_SIZE": "4", "LOCAL_RANK": "0"}, primary=True,
            sid_index_path="nonexistent-stub-sid.json", item_meta_path="stub-metadata.json",
        )
        self.assert_success(trace)
        self.assertEqual(trace.provenance_calls, [
            (os.path.join("out", "final_checkpoint"), "nonexistent-stub-sid.json",
             "stub-metadata.json", "stub-causal-model", "Video_Games")
        ])

    def provenance_writer(self):
        namespace = {"os": os, "json": json, "hashlib": hashlib}
        exec(self.provenance_code, namespace)
        return namespace["write_sid_provenance"]

    def test_provenance_hashes_actual_catalog_and_metadata_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sid = directory / "catalog.json"
            metadata = directory / "items.json"
            sid_bytes = b'{"0": ["<a_1>", "<b_2>", "<c_3>", "<d_0>"]}' + bytes([10])
            meta_bytes = '{"0": {"title": "游戏"}}'.encode("utf-8") + bytes([10])
            sid.write_bytes(sid_bytes)
            metadata.write_bytes(meta_bytes)
            output = directory / "final_checkpoint"
            path = self.provenance_writer()(
                output, sid, metadata, "local-causal-model", "Video_Games"
            )
            self.assertEqual(Path(path), output / "posttraining_provenance.json")
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(record["stage"], "sft")
            self.assertEqual(record["sid_index_sha256"], hashlib.sha256(sid_bytes).hexdigest())
            self.assertEqual(record["item_meta_sha256"], hashlib.sha256(meta_bytes).hexdigest())
            self.assertEqual(record["source_model"], "local-causal-model")
            self.assertEqual(record["category"], "Video_Games")
            self.assertFalse((output / "posttraining_provenance.json.tmp").exists())

    def test_generic_call_without_an_existing_sid_does_not_write_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            write = self.provenance_writer()
            self.assertIsNone(write(output, "", "", "model", "Office_Products"))
            self.assertIsNone(write(output, Path(temporary) / "missing.json", "", "model", "Office_Products"))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
