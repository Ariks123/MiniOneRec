"""Exercise the loss-only Trainer configuration without loading model weights."""
import ast
import builtins
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def training_function():
    return next(node for node in ast.parse((ROOT / "sft.py").read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "train")


def assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


class LinearLossTrainingTests(unittest.TestCase):
    def guard(self):
        function = copy.deepcopy(training_function())
        boundary = next(index for index, node in enumerate(function.body)
                        if assigned_name(node) == "world_size")
        function.body = function.body[:boundary] + [ast.Return(value=ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Load()) for name in
                  ("num_epochs", "lr_scheduler_type", "generative_validation")], ctx=ast.Load()))]
        namespace = {"math": math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(ROOT / "sft.py"), "exec"), namespace)
        return namespace["train"]

    def test_default_recipe_is_three_epoch_linear_loss_only(self):
        self.assertEqual(self.guard()(), (3, "linear", False))

    def test_linear_is_supported_without_removing_historical_cosine(self):
        train = self.guard()
        self.assertEqual(train(lr_scheduler_type="linear")[1], "linear")
        self.assertEqual(train(lr_scheduler_type="cosine")[1], "cosine")
        for scheduler in ("constant", "cosine_with_restarts", "unknown"):
            with self.subTest(scheduler=scheduler), self.assertRaises(ValueError):
                train(lr_scheduler_type=scheduler)

    def test_loss_only_setup_never_imports_ranking_or_creates_recall_callback(self):
        function = training_function()
        start = next(index for index, node in enumerate(function.body)
                     if assigned_name(node) == "eval_step")
        end = next(index for index, node in enumerate(function.body[start:], start)
                   if assigned_name(node) == "trainer")
        setup = ast.FunctionDef(
            name="setup", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                              kw_defaults=[], defaults=[]),
            body=copy.deepcopy(function.body[start:end + 1]) +
                 [ast.Return(value=ast.Name(id="trainer", ctx=ast.Load()))],
            decorator_list=[],
        )
        imports = []
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            imports.append(name)
            if "generative_validation" in name:
                raise AssertionError("Loss-only training must not import ranking validation")
            return original_import(name, *args, **kwargs)

        class FakeTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def forbidden_recall_callback(*args, **kwargs):
            raise AssertionError("Loss-only training must not create Recall-best retention")

        namespace = {
            "__builtins__": dict(vars(builtins), __import__=guarded_import),
            "transformers": SimpleNamespace(Trainer=FakeTrainer,
                TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs),
                DataCollatorForSeq2Seq=lambda *args, **kwargs: object()),
            "EarlyStoppingCallback": lambda **kwargs: SimpleNamespace(**kwargs),
            "BestRecallCheckpointCallback": forbidden_recall_callback,
            "eval_save_fraction": .05, "monitoring": False, "generative_validation": False,
            "minimum_epochs": 0, "save_epoch_checkpoint": 0,
            "early_stopping_patience": 3, "tokenizer": object(), "model": object(),
            "hf_train_dataset": object(), "hf_val_dataset": object(),
            "deepspeed": "/opt/minionerec/config/h20_zero2.json", "gradient_checkpointing": True,
            "wandb_run_name": "linear-loss-test", "micro_batch_size": 8,
            "gradient_accumulation_steps": 2, "num_epochs": 3,
            "learning_rate": .0003, "lr_scheduler_type": "linear",
            "output_dir": "/tmp/linear-loss-test", "ddp": True, "group_by_length": False,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[setup], type_ignores=[])),
                     str(ROOT / "sft.py"), "exec"), namespace)
        trainer = namespace["setup"]()
        self.assertIs(type(trainer), FakeTrainer)
        arguments = trainer.kwargs["args"]
        self.assertEqual(arguments.lr_scheduler_type, "linear")
        self.assertEqual(arguments.num_train_epochs, 3)
        self.assertEqual(arguments.metric_for_best_model, "eval_loss")
        self.assertFalse(arguments.greater_is_better)
        self.assertTrue(arguments.load_best_model_at_end)
        self.assertEqual(len(trainer.kwargs["callbacks"]), 1)
        self.assertEqual(trainer.kwargs["callbacks"][0].early_stopping_patience, 3)
        self.assertNotIn("generative_validation", trainer.kwargs)
        self.assertEqual(imports, [])


if __name__ == "__main__":
    unittest.main()
