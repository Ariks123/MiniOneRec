"""Lexicographic checkpoints, exact generation coverage and native validation."""
import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import sft_generative_selection as selection


def metrics(ndcg=.1, recall=.2, loss=1.):
    return {"ndcg10": ndcg, "recall10": recall, "eval_loss": loss}


class SelectionTests(unittest.TestCase):
    def test_ndcg_then_recall_then_loss_strict_priority_and_three_rejections(self):
        policy = selection.SelectionPolicy()
        self.assertTrue(policy.consider(metrics(), 1))
        self.assertFalse(policy.consider(metrics(.09, .99, .01), 2))
        self.assertEqual(policy.patience_counter, 1)
        self.assertTrue(policy.consider(metrics(.1000001, .21, 100), 3))
        self.assertEqual(policy.patience_counter, 2)
        self.assertTrue(policy.consider(metrics(.1000002, .2100001, 99), 4))
        self.assertEqual(policy.patience_counter, 3)
        self.assertTrue(policy.consider(metrics(.11, .1, 150), 5))
        self.assertEqual(policy.patience_counter, 0)
        for step in (6, 7, 8):
            self.assertFalse(policy.consider(metrics(.11, .1, 150), step))
        self.assertEqual(policy.patience_counter, 3)
        self.assertEqual(policy.best_step, 5)
        self.assertEqual(policy.improvements, 4)
        restored = selection.SelectionPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(restored.to_dict(), policy.to_dict())
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            restored.consider(metrics(), 8)

    def test_flat_ndcg_falling_loss_improves_checkpoint_but_stops_after_three_validations(self):
        policy = selection.SelectionPolicy(tolerance=1e-6, patience=3)
        decisions, primary = [], []
        for step, loss in enumerate((4., 3., 2., 1.), 1):
            decisions.append(policy.consider(metrics(.1, .2, loss), step))
            primary.append(policy.primary_improved)
        self.assertEqual(decisions, [True] * 4)
        self.assertEqual(primary, [True, False, False, False])
        self.assertEqual(policy.best_step, 4)
        self.assertEqual(policy.best_primary_ndcg, .1)
        self.assertEqual(policy.patience_counter, 3)

    def test_nonfinite_out_of_range_and_invalid_configuration_are_rejected(self):
        for values in (metrics(float("nan")), metrics(recall=2), metrics(loss=-1), metrics(loss=float("inf"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                selection.SelectionPolicy().consider(values, 1)
        with self.assertRaises(ValueError):
            selection.SelectionPolicy(tolerance=-1)
        with self.assertRaises(ValueError):
            selection.GenerativeValidationConfig("test.csv", "index.json")

    def test_invalid_and_duplicate_candidates_keep_original_slot_rank(self):
        catalog = {str(i) for i in range(12)}
        predictions = ["bad", "1", "1", "0", "2", "3", "4", "5", "6", "7"]
        result = selection.rank_row("0", predictions, catalog)
        self.assertEqual(result, {"rank": 4, "invalid_candidates": 1, "duplicate_candidates": 1})
        stats = selection.RankingStats(catalog)
        stats.add("0", predictions)
        stats.add("11", list(map(str, range(10))))
        self.assertEqual(stats.metrics()["recall10"], .5)
        self.assertAlmostEqual(stats.metrics()["ndcg10"], 1 / math.log2(5) / 2)
        self.assertEqual(stats.invalid_rows, 1)
        self.assertEqual(selection.rank_row("0", [None, "0", "0"], catalog)["rank"], 2)

    def test_strided_shards_have_no_sampler_padding_and_exact_coverage(self):
        shards = [selection.rank_row_ids(17, rank, 4) for rank in range(4)]
        self.assertEqual(selection.validate_coverage(shards, 17)["completed_rows"], 17)
        with self.assertRaises(ValueError):
            selection.validate_coverage([*shards, [0]], 17)
        with self.assertRaises(ValueError):
            selection.validate_coverage([[0, 1]], 3)


class ArtifactAndTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="generative-selection-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.sid = self.folder / "Video_Games.index.json"
        self.sid.write_text(json.dumps({str(i): [f"<a_{i}>", "<b_0>", "<c_0>", "<d_0>"] for i in range(12)}))
        self.valid = self.folder / "valid.csv"
        with self.valid.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["item_sid"])
            writer.writeheader()
            for item in (0, 0, 1):
                writer.writerow({"item_sid": f"<a_{item}><b_0><c_0><d_0>"})
        self.manifest = {"validation": "passed", "rows": {"valid": 3},
                         "outputs": {"valid.csv": {"sha256": selection.sha256_file(self.valid)}},
                         "source_sha256": {"Video_Games.index.json": selection.sha256_file(self.sid)}}
        (self.folder / "preparation_manifest.json").write_text(json.dumps(self.manifest))
        self.config = selection.GenerativeValidationConfig(self.valid, self.sid)

    def test_original_validation_duplicate_weight_and_hash_identity_without_opening_test(self):
        # No test.csv exists: selection can only open validation and catalog.
        artifacts = selection.validation_artifacts(self.config)
        self.assertEqual(len(artifacts["targets"]), 3)
        self.assertEqual(artifacts["targets"][0], artifacts["targets"][1])
        self.assertNotIn("test", artifacts["source_sha256"])
        self.valid.write_text(self.valid.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            selection.validation_artifacts(self.config)

    def trainer(self):
        class BaseTrainer:
            def __init__(base, **kwargs):
                base.args = SimpleNamespace(metric_for_best_model="selection_index", greater_is_better=True,
                                            load_best_model_at_end=True, eval_strategy="epoch", save_strategy="epoch",
                                            output_dir=str(self.folder / "run"))
                base.callback_handler = SimpleNamespace(callbacks=[])
                base.state = SimpleNamespace(global_step=1, epoch=1., best_metric=None, best_global_step=None,
                                             best_model_checkpoint=None)
                base.control = SimpleNamespace(should_save=False, should_training_stop=False)

            def evaluation_loop(base, *args, **kwargs):
                return SimpleNamespace(num_samples=3, metrics={"eval_loss": 1.})

            def _determine_best_metric(base, values, trial):
                if base.state.best_metric is None or values["eval_selection_index"] > base.state.best_metric:
                    base.state.best_metric = values["eval_selection_index"]
                    base.state.best_global_step = base.state.global_step
                    return True
                return False

            def _save_checkpoint(base, model, trial):
                path = Path(base.args.output_dir) / f"checkpoint-{base.state.global_step}"
                path.mkdir(parents=True)
                base.state.best_model_checkpoint = str(Path(base.args.output_dir) / f"checkpoint-{base.state.best_global_step}")

            def _get_output_dir(base, trial=None):
                return base.args.output_dir

            def is_world_process_zero(base):
                return True

        return selection.make_generative_trainer_class(BaseTrainer)(generative_validation=self.config)

    def test_metrics_enter_loop_before_best_hook_and_checkpoint_uses_same_decision(self):
        trainer = self.trainer()
        report = {"status": "passed", "metrics": {"ndcg10": .1, "recall10": .2},
                  "coverage": {"expected_rows": 3, "completed_rows": 3, "missing_rows": 0, "duplicate_rows": 0},
                  "scope": selection.SCOPE, "generation": {"num_beams": 10},
                  "source_sha256": trainer._validation_artifacts()["source_sha256"],
                  "prediction_directory": str(self.folder / "predictions")}
        with mock.patch.object(selection, "generate_validation", return_value=report):
            output = trainer.evaluation_loop(None, "evaluation")
        self.assertEqual(output.metrics["eval_ndcg10"], .1)
        self.assertEqual(output.metrics["eval_selection_index"], 1)
        self.assertTrue(trainer._determine_best_metric(output.metrics, None))
        trainer._save_checkpoint(None, None)
        self.assertEqual(trainer.state.best_global_step, 1)
        saved = Path(trainer.state.best_model_checkpoint) / "generative_selection_state.json"
        self.assertTrue(saved.is_file())
        for step in (2, 3, 4):
            trainer.state.global_step = step
            self.assertFalse(trainer._determine_best_metric(output.metrics, None))
        self.assertTrue(trainer.control.should_training_stop)
        summary = json.loads((Path(trainer.args.output_dir) / "selection_summary.json").read_text())
        self.assertEqual(summary["best"]["step"], 1)
        self.assertEqual(summary["latest"]["patience_counter"], 3)
        self.assertEqual(summary["status"], "early_stopping")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Pinned training image required for real Trainer generation integration")
class FrameworkIntegrationTests(unittest.TestCase):
    def test_real_trainer_full_catalog_generation_best_rotation_and_early_stop(self):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from data import SidSFTDataset

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="selection-real-trainer-") as folder:
            folder = Path(folder)
            catalog = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>", "<d_0>"] for i in range(12)}
            sid_tokens = sorted({token for sid in catalog.values() for token in sid})
            vocab = {token: index for index, token in enumerate(["[UNK]", "[EOS]", "[PAD]", "[BOS]", *sid_tokens])}
            backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            backend.decoder = decoders.Fuse()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]",
                                               pad_token="[PAD]", bos_token="[BOS]")
            tokenizer.add_tokens(sid_tokens, special_tokens=False)
            sid_path, valid_path = folder / "Video_Games.index.json", folder / "valid.csv"
            sid_path.write_text(json.dumps(catalog))
            with valid_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["user_id", "history_item_sid", "item_sid"])
                writer.writeheader()
                for item in range(4):
                    writer.writerow({"user_id": "A1", "history_item_sid": repr(["".join(catalog["0"])]),
                                     "item_sid": "".join(catalog[str(item)])})
            (folder / "preparation_manifest.json").write_text(json.dumps({
                "validation": "passed", "rows": {"valid": 4},
                "outputs": {"valid.csv": {"sha256": selection.sha256_file(valid_path)}},
                "source_sha256": {"Video_Games.index.json": selection.sha256_file(sid_path)}}))
            dataset = SidSFTDataset(str(valid_path), tokenizer, max_len=128, sample=-1, seed=None)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2,
                                             n_positions=128, bos_token_id=tokenizer.bos_token_id,
                                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
            arguments = TrainingArguments(output_dir=str(folder / "run"), use_cpu=True, max_steps=6,
                                          per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                          learning_rate=0., eval_strategy="steps", eval_steps=1,
                                          save_strategy="steps", save_steps=1, save_total_limit=2,
                                          load_best_model_at_end=True, metric_for_best_model="selection_index",
                                          greater_is_better=True, report_to="none", disable_tqdm=True)
            trainer = selection.make_generative_trainer_class(Trainer)(
                model=model, args=arguments, train_dataset=dataset, eval_dataset=dataset,
                processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer),
                generative_validation=selection.GenerativeValidationConfig(str(valid_path), str(sid_path), batch_size=2, cutoff_len=128))
            trainer.train()
            self.assertEqual(trainer.state.global_step, 4)
            self.assertEqual(trainer.selection_policy.best_step, 1)
            self.assertEqual(trainer.selection_policy.patience_counter, 3)
            self.assertTrue(trainer.state.best_model_checkpoint.endswith("checkpoint-1"))
            self.assertTrue((folder / "run/checkpoint-1/generative_selection_state.json").is_file())
            self.assertEqual(len(list((folder / "run").glob("checkpoint-*"))), 2)
            records = [json.loads(line) for line in (folder / "run/selection_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["improved"] for row in records], [True, False, False, False])
            self.assertTrue(all(row["coverage"]["completed_rows"] == 4 for row in records))
            self.assertTrue(all(row["generation"]["num_beams"] == 10 for row in records))


if __name__ == "__main__":
    unittest.main()
