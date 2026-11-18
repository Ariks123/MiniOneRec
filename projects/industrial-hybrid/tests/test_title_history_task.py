"""Task-mixture isolation and actual title-history sample semantics on CPU."""
import ast
import csv
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source_objects(path, names, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FakeConcat:
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))


class SizedSamples:
    def __init__(self, length, data=None):
        self.length = length
        if data is not None:
            self.data = data

    def __len__(self):
        return self.length


class TrainingTaskTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def factory(name, length, data=None):
            def create(**kwargs):
                self.calls.append((name, kwargs))
                return SizedSamples(length, data)
            return create

        self.namespace = source_objects(ROOT / "sft.py", {"build_training_tasks"}, {
            "SidSFTDataset": factory("sequence", 36259),
            "SidItemFeatDataset": factory("alignment", 7316,
                [{"task": "sid2title"}] * 3670 + [{"task": "title2sid"}] * 3646),
            "FusionSeqRecDataset": factory("history_to_title", 36259),
            "TitleHistory2SidSFTDataset": factory("title_history", 36259),
            "ConcatDataset": FakeConcat,
        })

    def build(self, enabled):
        return self.namespace["build_training_tasks"](
            train_file="TRAIN_ONLY.csv", item_meta_path="catalog.json",
            sid_index_path="index.json", tokenizer=object(), cutoff_len=512,
            sample=-1, seed=42, category="industrial and scientific items",
            include_title_history=enabled)

    def test_baseline_keeps_original_four_tasks_and_does_not_load_title_history(self):
        dataset, counts = self.build(False)
        self.assertEqual(len(dataset), 79834)
        self.assertEqual(counts, {
            "sid_sequence_recommendation": 36259, "history_to_title": 36259,
            "sid2title": 3670, "title2sid": 3646,
        })
        self.assertNotIn("title_history", [name for name, _ in self.calls])

    def test_added_task_uses_only_training_interactions_and_shared_catalog(self):
        dataset, counts = self.build(True)
        self.assertEqual(len(dataset), 116093)
        self.assertEqual(counts["title_history_to_sid"], 36259)
        self.assertEqual(counts["title_history_to_sid"], counts["sid_sequence_recommendation"])
        self.assertEqual(sum(counts.values()), len(dataset))
        for name, kwargs in self.calls:
            if name != "alignment":
                self.assertEqual(kwargs["train_file"], "TRAIN_ONLY.csv")
        title_kwargs = next(kwargs for name, kwargs in self.calls if name == "title_history")
        self.assertEqual(title_kwargs["item_file"], "catalog.json")
        self.assertEqual(title_kwargs["index_file"], "index.json")
        self.assertEqual(title_kwargs["max_len"], 512)
        self.assertNotIn("test", title_kwargs)

    def test_task_flag_rejects_truthy_strings(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.build("False")
        self.assertEqual(self.calls, [])


class Frame:
    def __init__(self, rows):
        self.rows = rows
        self.iloc = self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CharacterTokenizer:
    bos_token_id = 1000000
    eos_token_id = 1000001

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens if token < self.bos_token_id)


class TitleHistorySampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="title-history-task-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train.csv"
        with self.train.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "item_id", "history_item_title", "history_item_id", "item_title"))
            writer.writeheader()
            writer.writerow({"item_id": "2", "history_item_title": repr(["First tool", "Second tool"]),
                             "history_item_id": "[0, 1]", "item_title": "TARGET_TITLE_NEVER_INPUT"})
        self.metadata = self.root / "items.json"
        self.metadata.write_text(json.dumps({"2": {"title": "TARGET_TITLE_NEVER_INPUT"}}))
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"2": ["<a_7>", "<b_8>", "<c_9>"]}))

        def read_csv(path):
            with Path(path).open(newline="", encoding="utf-8") as stream:
                return Frame(list(csv.DictReader(stream)))

        self.namespace = source_objects(ROOT / "data.py", {
            "Tokenizer", "BaseDataset", "TitleHistory2SidSFTDataset",
        }, {"Dataset": object, "List": list, "random": random,
            "json": json, "pd": SimpleNamespace(read_csv=read_csv), "tqdm": lambda rows: rows})

    def build(self, max_len=2048, test=False):
        return self.namespace["TitleHistory2SidSFTDataset"](
            train_file=self.train, item_file=self.metadata, index_file=self.index,
            tokenizer=CharacterTokenizer(), max_len=max_len, test=test,
            category="industrial and scientific items", seed=42)

    def test_input_contains_ordered_history_titles_and_supervises_only_next_item_sid(self):
        dataset = self.build()
        item = dataset[0]
        prompt_tokens = [token for token, label in zip(item["input_ids"], item["labels"]) if label == -100]
        supervised = [label for label in item["labels"] if label != -100]
        tokenizer = CharacterTokenizer()
        prompt = tokenizer.decode(prompt_tokens)
        self.assertLess(prompt.index('"First tool"'), prompt.index('"Second tool"'))
        self.assertNotIn("TARGET_TITLE_NEVER_INPUT", prompt)
        self.assertNotIn("<a_7>", prompt)
        self.assertEqual(tokenizer.decode(supervised), "<a_7><b_8><c_9>\n")
        self.assertEqual(supervised[-1], tokenizer.eos_token_id)
        self.assertEqual(len(item["input_ids"]), len(item["labels"]))

    def test_left_truncation_preserves_complete_target_supervision(self):
        item = self.build(max_len=24)[0]
        supervised = [label for label in item["labels"] if label != -100]
        self.assertEqual(len(item["input_ids"]), 24)
        self.assertEqual(CharacterTokenizer().decode(supervised), "<a_7><b_8><c_9>\n")
        self.assertEqual(supervised[-1], CharacterTokenizer.eos_token_id)


class TaskResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="task-resume-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint-5"
        self.checkpoint.mkdir()
        self.counts = {"sid_sequence_recommendation": 2, "title_history_to_sid": 2}
        self.previous = {"task_counts": self.counts.copy(),
                         "data": {"include_title_history": True},
                         "hyperparameters": {"early_stopping_patience": 3, "eval_save_fraction": 0.05}}
        self.namespace = source_objects(ROOT / "sft.py", {"verify_task_resume_configuration"},
                                        {"Path": Path, "json": json})

    def check(self):
        (self.root / "run_config.json").write_text(json.dumps(self.previous))
        self.namespace["verify_task_resume_configuration"](
            self.checkpoint, self.counts, True, 3, 0.05)

    def test_same_task_configuration_can_resume(self):
        self.check()

    def test_different_task_counts_or_flag_cannot_resume(self):
        self.previous["task_counts"]["title_history_to_sid"] = 1
        with self.assertRaisesRegex(ValueError, "task counts"):
            self.check()
        self.previous["task_counts"] = self.counts.copy()
        self.previous["data"]["include_title_history"] = False
        with self.assertRaisesRegex(ValueError, "include_title_history"):
            self.check()

    def test_changed_stopping_or_evaluation_schedule_cannot_resume(self):
        for key, changed in (("early_stopping_patience", 5), ("eval_save_fraction", 0.1)):
            with self.subTest(key=key):
                original = self.previous["hyperparameters"][key]
                self.previous["hyperparameters"][key] = changed
                with self.assertRaisesRegex(ValueError, key):
                    self.check()
                self.previous["hyperparameters"][key] = original


if __name__ == "__main__":
    unittest.main()
