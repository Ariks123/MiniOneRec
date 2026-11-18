"""Run the archived GRPO recipe with a newly loss-selected RQ-VAE SFT parent."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sft_eval_utils import sha256_file


BASE_IDENTITY = "ff170b0115ab6b9c2d872ad7e3a234fbd1efbefa2a5a8aa8dc7131cee40ee79a"
CATEGORY = "Industrial_and_Scientific"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_parent(model_path, data_paths, hash_file):
    model = Path(model_path).resolve()
    parent = model.parent
    require(model.name == "final_checkpoint" and parent.name == "sft"
            and parent.parent.name == "epochs3", "Expected the new loss-selected epochs3 SFT export")
    summary = load(parent / "training_summary.json")
    config = load(parent / "run_config.json")
    provenance = load(model / "posttraining_provenance.json")
    hyper = config["hyperparameters"]
    require(summary.get("status") == "completed" and summary.get("selection_metric") == "eval_loss"
            and summary.get("final_checkpoint_contains_best_model") is True,
            "SFT parent does not export the loss-selected checkpoint")
    require(hyper.get("max_epochs") == 3 and hyper.get("world_size") == 4
            and hyper.get("lr_scheduler_type") == "linear"
            and config["data"].get("include_title_history") is True,
            "SFT parent recipe changed")
    require(config["model"]["identity"]["sha256"] == BASE_IDENTITY,
            "SFT parent was not freshly trained from the pinned Qwen2.5-0.5B Base weights")
    require(provenance.get("category") == CATEGORY and provenance.get("stage") == "sft",
            "SFT parent provenance identifies a different task")
    expected = {name: hash_file(path) for name, path in data_paths.items()}
    require(expected == config["data"]["sha256"], "GRPO data differ from the SFT parent data")
    require(provenance["sid_index_sha256"] == expected["sid_index_path"]
            and provenance["item_meta_sha256"] == expected["item_meta_path"],
            "GRPO/SFT SID or item metadata do not match")
    manifest = load(Path(data_paths["train_file"]).parent / "preparation_manifest.json")
    require(manifest["original_sid"] is False and manifest["expected_sid_sha256"] == expected["sid_index_path"],
            "GRPO requires the newly trained RQ-VAE SID and its preparation manifest")
    best = parent / Path(summary["best_checkpoint"]).name
    require(best.name.startswith("checkpoint-"), "SFT summary lacks a selected training step")
    selected_step = int(best.name.removeprefix("checkpoint-"))
    state = load(parent / "trainer_state.json")
    selected = [entry for entry in state["log_history"]
                if entry.get("step") == selected_step and entry.get("eval_loss") == summary["best_metric"]]
    require(selected and min(entry["eval_loss"] for entry in state["log_history"] if "eval_loss" in entry)
            == summary["best_metric"], "SFT selection differs from the minimum recorded validation loss")
    weights_hash = hash_file(model / "model.safetensors")
    require(weights_hash == hash_file(best / "model.safetensors"),
            "Exported SFT weight differs from the selected checkpoint")
    return {
        "model_path": str(model), "run_dir": str(parent), "selected_checkpoint": str(best),
        "selected_step": selected_step, "selected_epoch": selected[0]["epoch"], "model_sha256": weights_hash,
        "summary_sha256": hash_file(parent / "training_summary.json"),
        "config_sha256": hash_file(parent / "run_config.json"), "provenance": provenance,
    }


def run(**kwargs):
    from rl import train
    import rl

    names = ("train_file", "eval_file", "sid_index_path", "item_meta_path")
    prepared = load(Path(kwargs["train_file"]).parent / "preparation_manifest.json")
    expected = {
        "train_file": prepared["outputs"]["train.csv"]["sha256"],
        "eval_file": prepared["outputs"]["valid.csv"]["sha256"],
        "sid_index_path": prepared["expected_sid_sha256"],
        "item_meta_path": prepared["sources"]["item_metadata"]["sha256"],
    }
    require({name: sha256_file(kwargs[name]) for name in names} == expected,
            "New GRPO inputs differ from the preparation manifest")
    # Only parameterize lineage. The source of GRPO, rollout grouping, reward,
    # optimizer, selection and numerical runtime remains the archived module.
    rl.EXPECTED_DATA_SHA256 = expected
    rl.verify_parent_sft = validate_parent
    train(**kwargs)


if __name__ == "__main__":
    from fire import Fire
    Fire(run)
