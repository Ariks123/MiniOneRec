"""Row-local labels and original task mixture for Industrial GRPO.

The helpers outside ``build_rl_tasks`` deliberately have no training dependencies.
Repeated histories may have different legitimate next items; a prompt/history
dictionary must never be used as the reward label source.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


CATEGORY = "Industrial_and_Scientific"
EXPECTED_PARENT_MODEL_SHA256 = "e688140bbfe9aac2981324471dd79ff9da1ad0b329acfa9c18c63f4dca4f597c"
EXPECTED_TASK_COUNTS = {
    "sid_sequence_recommendation": 36259,
    "title2sid": 3646,
    "description2sid": 2870,
    "title_history_to_sid": 10000,
}
EXPECTED_DATA_SHA256 = {
    "train_file": "da5f2af4f5047e18023dd4792f65a48bd7c8eab90447dd682fb48ca255d25f8c",
    "eval_file": "12bbb02bced0bd754a8b4e0f9ca4086416e30d12b0b73a2c95e7b195dd67385a",
    "sid_index_path": "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8",
    "item_meta_path": "937600c7a03652f55864a1d12d6e9fe8ca3e24ecc4e255a88216b25c5f949496",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_sid(value):
    require(isinstance(value, str), "A SID label/completion must be a string")
    return value.strip('\n" ')


def verify_info_catalog(info_file, sid_index_path):
    index = json.loads(Path(sid_index_path).read_text())
    expected = Counter("".join(tokens) for tokens in index.values())
    actual = Counter(line.split("\t")[0].strip() for line in Path(info_file).read_text().splitlines()
                     if line.strip())
    require(actual == expected, "RL info.txt must contain the complete pinned SID catalog, including collisions")
    return True


def labeled_rows(dataset, task_type, prefix):
    """Preserve every row and its own target, including duplicate prompts."""
    rows = []
    for index, sample in enumerate(dataset):
        name = task_type(index) if callable(task_type) else task_type
        target = normalize_sid(sample["completion"])
        require(target and sample["prompt"], "RL samples require nonempty prompts and targets")
        rows.append({"sample_id": f"{prefix}:{index}", "task_type": name,
                     "prompt": sample["prompt"], "completion": sample["completion"],
                     "target_sid": target})
    return rows


def write_task_manifest(path, train_rows, validation_rows, seed=42):
    """Persist the exact deterministic task draw without duplicating prompts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    digest = hashlib.sha256()
    total = 0
    with temporary.open("wb") as stream:
        for split, rows in [("train", train_rows), ("valid", validation_rows)]:
            for row in rows:
                record = {"split": split, "sample_id": row["sample_id"], "task_type": row["task_type"],
                          "target_sid": row["target_sid"],
                          "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()}
                encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                           + "\n").encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                total += 1
        stream.flush()
    os.replace(temporary, destination)
    return {"path": str(destination), "sha256": digest.hexdigest(), "rows": total,
            "train_rows": len(train_rows), "validation_rows": len(validation_rows), "seed": seed}


def build_rl_tasks(train_file, eval_file, item_meta_path, sid_index_path, seed=42):
    from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset

    require(Path(train_file).name == "train.csv" and Path(eval_file).name == "valid.csv",
            "RL training/selection must use train.csv/valid.csv, never test.csv")
    require(Path(train_file).resolve().parent == Path(eval_file).resolve().parent,
            "RL train/valid must come from the same prepared split")
    require(seed == 42, "The pinned Industrial RL task mixture uses seed 42")
    category = "industrial and scientific items"
    sequential = SidDataset(train_file, category=category, sample=-1, seed=seed)
    metadata = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path,
                                 category=category, sample=-1, seed=seed)
    title_history = RLSeqTitle2SidDataset(train_file, category=category, sample=10000, seed=seed)
    validation = SidDataset(eval_file, category=category, sample=-1, seed=seed)
    rows = labeled_rows(sequential, "sid_sequence_recommendation", "train:sid")
    rows += labeled_rows(metadata, lambda index: metadata.data[index]["task"], "train:metadata")
    rows += labeled_rows(title_history, "title_history_to_sid", "train:title_history")
    valid_rows = labeled_rows(validation, "sid_sequence_recommendation", "valid:sid")
    counts = dict(Counter(row["task_type"] for row in rows))
    require(counts == EXPECTED_TASK_COUNTS, "Industrial RL task counts changed: " + str(counts))
    require(len(rows) == 52775 and len(valid_rows) == 4532, "Industrial RL split coverage changed")
    require(len({row["sample_id"] for row in rows + valid_rows}) == len(rows) + len(valid_rows),
            "RL sample IDs must distinguish split, task and original row")
    return rows, valid_rows, counts


def validate_rollout_groups(rows, num_generations):
    require(type(num_generations) is int and num_generations >= 2,
            "Grouped rewards require at least two candidates")
    require(rows and len(rows) % num_generations == 0,
            "Each rank must contain complete rollout groups")
    fields = ("sample_id", "task_type", "target_sid", "prompt")
    for offset in range(0, len(rows), num_generations):
        group = rows[offset:offset + num_generations]
        for field in fields:
            require(all(isinstance(row.get(field), str) and row[field] for row in group),
                    "Missing row-local RL field: " + field)
            require(len({row[field] for row in group}) == 1,
                    "A rank-local rollout group mixes " + field)
        require(all(normalize_sid(row["target_sid"]) == row["target_sid"] for row in group),
                "RL targets must already be normalized")
    return True


def validate_generation_batch_sizes(train_batch_size, eval_batch_size, num_generations,
                                    world_size, *, beam_search=True, eval_enabled=True):
    sizes = [train_batch_size, num_generations, world_size]
    if eval_enabled:
        sizes.append(eval_batch_size)
    require(all(type(value) is int and value > 0 for value in sizes),
            "RL batch sizes, group size and world size must be positive integers")
    require(num_generations >= 2, "Grouped rewards require at least two candidates")
    for batch in [train_batch_size] + ([eval_batch_size] if eval_enabled else []):
        require(batch * world_size % num_generations == 0, "Global batch splits a rollout group")
        require(not beam_search or batch % num_generations == 0,
                "Beam rollout groups must remain complete on each rank")


def rule_reward(prompts, completions, *, target_sid, sample_id, task_type):
    require(len(prompts) == len(completions) == len(target_sid) == len(sample_id) == len(task_type),
            "Reward rows and row-local labels have different lengths")
    return [float(normalize_sid(completion) == normalize_sid(target))
            for completion, target in zip(completions, target_sid)]


def make_ndcg_rule_reward(num_generations):
    # Keep the original MiniOneRec ranking reward exactly: correct candidates
    # receive 0 here, wrong candidates receive normalized negative discounts;
    # entirely unsuccessful groups receive 0. rule_reward supplies the +1 hit.
    discounts = [-1.0 / math.log2(index + 2) for index in range(num_generations)]
    penalties = [-value / sum(discounts) for value in discounts]

    def ndcg_rule_reward(prompts, completions, *, target_sid, sample_id, task_type):
        require(len(completions) % num_generations == 0, "Ranking reward received a partial group")
        hits = rule_reward(prompts, completions, target_sid=target_sid,
                           sample_id=sample_id, task_type=task_type)
        rewards = []
        for offset in range(0, len(completions), num_generations):
            group_hits = hits[offset:offset + num_generations]
            if any(group_hits):
                rewards.extend(0.0 if hit else penalties[index] for index, hit in enumerate(group_hits))
            else:
                rewards.extend([0.0] * num_generations)
        return rewards

    return ndcg_rule_reward


def verify_parent_sft(model_path, data_paths, hash_file):
    """Pin this experiment to the completed three-epoch, loss-selected model."""
    model = Path(model_path).resolve()
    require(model.name == "final_checkpoint" and model.parent.name == "sft"
            and model.parent.parent.name == "epochs3", "RL must start at the selected epochs3 final_checkpoint")
    summary = json.loads((model.parent / "training_summary.json").read_text())
    config = json.loads((model.parent / "run_config.json").read_text())
    provenance = json.loads((model / "posttraining_provenance.json").read_text())
    require(summary.get("status") == "completed" and summary.get("selection_metric") == "eval_loss"
            and summary.get("final_checkpoint_contains_best_model") is True,
            "Parent SFT must be complete and contain its loss-best model")
    require(Path(summary["best_checkpoint"]).name == "checkpoint-5187"
            and float(summary.get("epoch", 0)) == 3.0,
            "Parent SFT is not the selected three-epoch checkpoint")
    hyper = config["hyperparameters"]
    require(hyper.get("max_epochs") == 3 and hyper.get("lr_scheduler_type") == "linear"
            and config["data"].get("include_title_history") is True,
            "Parent SFT recipe changed")
    require(config.get("model", {}).get("identity", {}).get("sha256") ==
            "ff170b0115ab6b9c2d872ad7e3a234fbd1efbefa2a5a8aa8dc7131cee40ee79a",
            "Parent SFT must originate from the pinned 0.5B Base model")
    require(provenance.get("category") == CATEGORY, "Parent checkpoint category differs")
    for name, expected in EXPECTED_DATA_SHA256.items():
        require(hash_file(data_paths[name]) == expected == config["data"]["sha256"][name],
                "Parent/RL training data identity differs: " + name)
    require(provenance.get("sid_index_sha256") == EXPECTED_DATA_SHA256["sid_index_path"]
            and provenance.get("item_meta_sha256") == EXPECTED_DATA_SHA256["item_meta_path"],
            "Parent SID provenance differs")
    parent_weights_sha256 = hash_file(model / "model.safetensors")
    require(parent_weights_sha256 == EXPECTED_PARENT_MODEL_SHA256,
            "Parent final weights differ from the selected checkpoint-5187")
    return {"model_path": str(model), "run_dir": str(model.parent),
            "selected_checkpoint": str(model.parent / "checkpoint-5187"), "selected_step": 5187,
            "selected_epoch": 5187 / 1814, "model_sha256": parent_weights_sha256,
            "summary_sha256": hash_file(model.parent / "training_summary.json"),
            "config_sha256": hash_file(model.parent / "run_config.json"),
            "provenance": provenance}
