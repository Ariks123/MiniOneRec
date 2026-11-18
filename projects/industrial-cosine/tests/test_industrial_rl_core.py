"""Row-local RL labels, unchanged ranking reward and parent lineage guards."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.industrial_rl_data import (
    EXPECTED_DATA_SHA256, EXPECTED_PARENT_MODEL_SHA256, labeled_rows,
    make_ndcg_rule_reward, rule_reward, validate_generation_batch_sizes,
    validate_rollout_groups, verify_info_catalog, verify_parent_sft, write_task_manifest,
)


def group(sample_id, target, task="sid_sequence_recommendation", size=16):
    return [{"sample_id": sample_id, "target_sid": target, "prompt": "identical historical prompt",
             "task_type": task} for _ in range(size)]


def rewards(function, rows, completions):
    return function(prompts=[row["prompt"] for row in rows], completions=completions,
                    **{key: [row[key] for row in rows] for key in ["target_sid", "sample_id", "task_type"]})


class RowLocalRLTests(unittest.TestCase):
    def test_task_manifest_preserves_duplicate_prompt_rows_split_and_exact_draw(self):
        data = [{"prompt": "same history 商品", "completion": "SID-A\n"},
                {"prompt": "same history 商品", "completion": "SID-B\n"}]
        train = labeled_rows(data, "title_history_to_sid", "train:title_history")
        valid = labeled_rows(data[:1], "sid_sequence_recommendation", "valid:sid")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl_task_manifest.jsonl"
            first = write_task_manifest(path, train, valid, 42)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(first["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual((first["rows"], first["train_rows"], first["validation_rows"], first["seed"]),
                             (3, 2, 1, 42))
            self.assertEqual([row["target_sid"] for row in records], ["SID-A", "SID-B", "SID-A"])
            self.assertEqual([row["split"] for row in records], ["train", "train", "valid"])
            self.assertEqual(len({row["sample_id"] for row in records}), 3)
            self.assertEqual(len({row["prompt_sha256"] for row in records}), 1)
            self.assertTrue(all("prompt" not in row for row in records))
            self.assertEqual(first, write_task_manifest(path, train, valid, 42))
            changed = copy.deepcopy(train)
            changed[0]["target_sid"] = "SID-C"
            self.assertNotEqual(first["sha256"], write_task_manifest(path, changed, valid, 42)["sha256"])

    def test_task_manifest_write_errors_propagate_to_distributed_entry_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "not-a-directory"
            blocker.write_text("retained")
            with self.assertRaises(OSError):
                write_task_manifest(blocker / "manifest.jsonl", [], [], 42)
            self.assertEqual(blocker.read_text(), "retained")

    def test_duplicate_history_keeps_distinct_training_and_validation_labels(self):
        data = [{"prompt": "same history", "completion": "SID-A\n"},
                {"prompt": "same history", "completion": "SID-B\n"}]
        train = labeled_rows(data, "sid_sequence_recommendation", "train:sid")
        valid = labeled_rows(data[:1], "sid_sequence_recommendation", "valid:sid")
        self.assertEqual([row["target_sid"] for row in train], ["SID-A", "SID-B"])
        self.assertEqual(len({row["sample_id"] for row in train + valid}), 3)
        batch = group(train[0]["sample_id"], "SID-A") + group(train[1]["sample_id"], "SID-B")
        candidates = ["SID-A"] + ["wrong"] * 15 + ["SID-A"] + ["wrong"] * 15
        self.assertTrue(validate_rollout_groups(batch, 16))
        self.assertEqual(rewards(rule_reward, batch, candidates), [1.0] + [0.0] * 31)
        self.assertEqual(valid[0]["target_sid"], "SID-A")

    def test_original_ranking_reward_preserved_for_hit_at_different_positions(self):
        rows = group("first", "target") + group("second", "target")
        candidates = ["wrong"] * 32
        candidates[0], candidates[16 + 9] = "target\n", ' "target" '
        actual = rewards(make_ndcg_rule_reward(16), rows, candidates)
        raw = [-1 / math.log2(i + 2) for i in range(16)]
        expected = [-value / sum(raw) for value in raw]
        for offset, position in [(0, 0), (16, 9)]:
            for index in range(16):
                self.assertAlmostEqual(actual[offset + index], 0 if index == position else expected[index])
        hit_rewards = rewards(rule_reward, rows, candidates)
        self.assertEqual(sum(hit_rewards), 2.0)
        self.assertTrue(all(value < 0 for index, value in enumerate(actual) if index not in [0, 25]))

    def test_all_zero_groups_have_no_artificial_negative_ranking_reward(self):
        rows = group("first", "target") + group("second", "different")
        values = rewards(make_ndcg_rule_reward(16), rows, ["wrong"] * 32)
        self.assertEqual(values, [0.0] * 32)

    def test_group_integrity_rejects_cross_row_or_cross_task_labels(self):
        for field, value in [("sample_id", "other-row"), ("target_sid", "other-target"),
                             ("task_type", "title2sid"), ("prompt", "other-prompt")]:
            rows = group("first", "target")
            rows[8][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rollout_groups(rows, 16)
        for rows in [[], group("row", "target")[:-1]]:
            with self.assertRaises(ValueError):
                validate_rollout_groups(rows, 16)

    def test_complete_rank_local_beam_groups_required(self):
        validate_generation_batch_sizes(16, 16, 16, 4)
        validate_generation_batch_sizes(32, 16, 16, 4)
        for batch in [4, 8, 15, 17]:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                validate_generation_batch_sizes(batch, 16, 16, 4)

    def test_reward_requires_one_target_per_completion(self):
        rows = group("first", "target")
        with self.assertRaises(ValueError):
            rewards(rule_reward, rows, ["target"] * 15)
        with self.assertRaises(ValueError):
            rewards(make_ndcg_rule_reward(16), rows[:-1], ["target"] * 15)


class ParentIdentityTests(unittest.TestCase):
    def test_catalog_must_include_all_items_and_original_sid_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, info = root / "index.json", root / "info.txt"
            index.write_text(json.dumps({"item1": ["SID-A"], "item2": ["SID-A"], "item3": ["SID-B"]}))
            info.write_text("SID-A\ttitle1\nSID-A\ttitle2\nSID-B\ttitle3\n")
            self.assertTrue(verify_info_catalog(info, index))
            info.write_text("SID-A\ttitle1\nSID-B\ttitle3\n")
            with self.assertRaises(ValueError):
                verify_info_catalog(info, index)

    def fixture(self, root):
        model = root / "epochs3/sft/final_checkpoint"
        model.mkdir(parents=True)
        summary = {"status": "completed", "selection_metric": "eval_loss",
                   "final_checkpoint_contains_best_model": True,
                   "best_checkpoint": str(model.parent / "checkpoint-5187"), "epoch": 3.0}
        config = {"hyperparameters": {"max_epochs": 3, "lr_scheduler_type": "linear"},
                  "data": {"include_title_history": True, "sha256": EXPECTED_DATA_SHA256},
                  "model": {"identity": {"sha256":
                      "ff170b0115ab6b9c2d872ad7e3a234fbd1efbefa2a5a8aa8dc7131cee40ee79a"}}}
        provenance = {"category": "Industrial_and_Scientific",
                      "sid_index_sha256": EXPECTED_DATA_SHA256["sid_index_path"],
                      "item_meta_sha256": EXPECTED_DATA_SHA256["item_meta_path"]}
        for path, value in [(model.parent / "training_summary.json", summary),
                            (model.parent / "run_config.json", config),
                            (model / "posttraining_provenance.json", provenance)]:
            path.write_text(json.dumps(value))
        paths = {key: root / key for key in EXPECTED_DATA_SHA256}
        hashes = {str(path): EXPECTED_DATA_SHA256[key] for key, path in paths.items()}
        hashes[str(model / "model.safetensors")] = EXPECTED_PARENT_MODEL_SHA256
        return model, paths, hashes, summary

    def test_exact_selected_parent_accepted_without_duplicate_model_export(self):
        with tempfile.TemporaryDirectory() as directory:
            model, paths, hashes, _ = self.fixture(Path(directory))
            result = verify_parent_sft(model, paths, lambda path: hashes.get(str(path), "metadata-hash"))
            self.assertEqual(result["selected_step"], 5187)
            self.assertEqual(result["model_sha256"], EXPECTED_PARENT_MODEL_SHA256)

    def test_changed_parent_selection_data_or_weights_rejected(self):
        for change in ["selection", "data", "weights"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                model, paths, hashes, summary = self.fixture(Path(directory))
                if change == "selection":
                    summary["best_checkpoint"] = str(model.parent / "checkpoint-8626")
                    (model.parent / "training_summary.json").write_text(json.dumps(summary))
                elif change == "data":
                    hashes[str(paths["train_file"])] = "wrong-data"
                else:
                    hashes[str(model / "model.safetensors")] = "wrong-weights"
                with self.assertRaises(ValueError):
                    verify_parent_sft(model, paths, lambda path: hashes.get(str(path), "metadata-hash"))


if __name__ == "__main__":
    unittest.main()
