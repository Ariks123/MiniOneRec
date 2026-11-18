"""Validate an isolated RQ-VAE experiment's assets without weakening the legacy checks."""
import argparse
import csv
from itertools import zip_longest
import json
from pathlib import Path
import re

from hybrid.drafters import RAW_SHA, SID_SHA, UniSRecDrafter
from hybrid.experiment import Data
from hybrid.sweep import TEST_ROWS, VALID_ROWS, sha256, write_json


MANIFEST = "rqvae_manifest.json"
SPLIT_ROWS = {"train": 36259, "valid": VALID_ROWS, "test": TEST_ROWS}
EXCLUDED_COLUMNS = {"history_item_sid", "item_sid"}
MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors", "posttraining_provenance.json",
    "tokenizer.json", "tokenizer_config.json",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_catalog(path):
    content = load_json(path)
    if not isinstance(content, dict) or set(content) != {str(i) for i in range(3686)}:
        raise ValueError("RQ-VAE SID index must identify the same 3,686 catalog items")
    if any(
        not isinstance(tokens, list) or len(tokens) != 3
        or any(not isinstance(token, str) or re.fullmatch(fr"<{letter}_\d+>", token) is None
               for letter, token in zip("abc", tokens))
        for tokens in content.values()
    ):
        raise ValueError("An RQ-VAE SID must consist of exactly one a, b and c token")
    unique = len({tuple(tokens) for tokens in content.values()})
    if unique < 50:
        raise ValueError("At least 50 unique SIDs are needed for the original beam search")
    return unique


def hash_files(root, paths):
    return {name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
            for name in paths}


def audit_rows(legacy_data, new_data):
    """Prove split identity before comparing any prediction or cold/warm metric."""
    observed = {}
    for name, expected in SPLIT_ROWS.items():
        relative = name + ".csv"
        count = 0
        with (legacy_data / relative).open(newline="", encoding="utf-8") as before, (
            new_data / relative
        ).open(newline="", encoding="utf-8") as after:
            old = csv.DictReader(before)
            new = csv.DictReader(after)
            schema = set(old.fieldnames or ())
            if set(new.fieldnames or ()) != schema or not EXCLUDED_COLUMNS <= schema:
                raise ValueError(f"Prepared CSV schema differs in {relative}")
            fields = schema - EXCLUDED_COLUMNS
            for count, pair in enumerate(zip_longest(old, new), 1):
                a, b = pair
                if a is None or b is None or any(a[field] != b[field] for field in fields):
                    raise ValueError(f"Preparation changed a non-SID field at {relative}:{count}")
        if count != expected:
            raise ValueError(f"Expected {expected} archived requests in {relative}; got {count}")
        observed[name] = count
    return observed


def seal(assets, legacy_data, legacy_run, rl_run, unisrec_checkpoint):
    files = hash_files(assets, [
        "sid.json", "reference_test_beam.jsonl", "unisrec/metadata.json",
        *(f"data/{name}.csv" for name in SPLIT_ROWS),
        *(f"rl/{name}" for name in MODEL_FILES),
    ])
    sid_sha = files["sid.json"]["sha256"]
    if sid_sha == SID_SHA:
        raise ValueError("The new experiment must use the newly generated RQ-VAE mapping")
    unique_sids = inspect_catalog(assets / "sid.json")
    rows = audit_rows(legacy_data, assets / "data")
    legacy_catalog = load_json(legacy_run / "catalog.json")
    old_manifest = load_json(legacy_run / "manifest.json")
    for split in SPLIT_ROWS:
        name = split + ".csv"
        if sha256(legacy_data / name) != old_manifest["data"]["split_files"][name]["sha256"]:
            raise ValueError(f"Legacy reference CSV does not match the archived benchmark: {name}")
    aligned = Data(assets, expected_unique_sids=unique_sids)
    for split in SPLIT_ROWS:
        if len(aligned.read(split)) != rows[split]:
            raise ValueError(f"Unexpected {split} request count")
    for name in ("target_freq", "cold"):
        if aligned.record[name] != legacy_catalog[name]:
            raise ValueError(f"Training-defined item {name} differs across experiments")

    provenance = load_json(assets / "rl/posttraining_provenance.json")
    rl = load_json(rl_run / "training_summary.json")
    rl_config = load_json(rl_run / "run_config.json")
    metrics = load_json(rl_run / "evaluation/metrics.json")
    if (provenance["sid_index_sha256"] != sid_sha
            or rl["parent_sft"]["provenance"]["sid_index_sha256"] != sid_sha
            or rl_config["hyperparameters"]["beta"] != 0.001
            or not rl["final_checkpoint_contains_best_model"]):
        raise ValueError("RQ-VAE RL export does not identify the selected β=0.001 run on this SID catalog")
    if (metrics["status"], metrics["world_size"], metrics["num_beams"], metrics["completed_rows"],
            metrics["prompt_style"], metrics["sid_index_sha256"]) != (
            "passed", 4, 50, TEST_ROWS, "original-eval", sid_sha):
        raise ValueError("RQ-VAE RL evaluation must be the completed original four-rank beam50 protocol")
    if (metrics["source_sha256"]["test_file"] != files["data/test.csv"]["sha256"]
            or sha256(rl_run / "evaluation/predictions.jsonl") != files["reference_test_beam.jsonl"]["sha256"]
            or sha256(rl_run / "final_checkpoint/model.safetensors") != files["rl/model.safetensors"]["sha256"]):
        raise ValueError("Staged RL checkpoint/test artifacts differ from the archived new run")

    meta = load_json(assets / "unisrec/metadata.json")
    checkpoint_sha = sha256(unisrec_checkpoint)
    if (meta["checkpoint_sha256"] != checkpoint_sha
            or meta["provenance"]["embedding_sha256"] != RAW_SHA
            or meta["item_count"] != 3686):
        raise ValueError("Staged UniSRec index must be from the newly trained checkpoint and original item embeddings")
    result = dict(
        schema=1, variant="rqvae", files=files, rows=rows, unique_sids=unique_sids,
        unisrec_checkpoint_sha256=checkpoint_sha, unisrec_selected_epoch=meta["checkpoint_epoch_zero_based"] + 1,
        rl_selected_checkpoint=Path(rl["best_checkpoint"]).name,
        legacy_catalog_sha256=sha256(legacy_run / "catalog.json"),
        legacy_reference_sha256=old_manifest["data"]["reference_test_beam"]["sha256"],
        legacy_split_sha256s={key: old_manifest["data"]["split_files"][key + ".csv"]["sha256"]
                              for key in SPLIT_ROWS},
        rl_training_summary_sha256=sha256(rl_run / "training_summary.json"),
        rl_evaluation_metrics_sha256=sha256(rl_run / "evaluation/metrics.json"),
    )
    write_json(assets / MANIFEST, result)
    return result


class RQVAEData(Data):
    """Read only prepared RQ-VAE assets and expose only the retrained UniSRec drafter."""

    def __init__(self, assets):
        assets = Path(assets)
        self.manifest = load_json(assets / MANIFEST)
        if (self.manifest["schema"], self.manifest["variant"], self.manifest["rows"]) != (1, "rqvae", SPLIT_ROWS):
            raise ValueError("Invalid RQ-VAE inference asset manifest")
        expected = self.manifest["files"]
        paths = {"sid.json", "reference_test_beam.jsonl", "unisrec/metadata.json",
                 *(f"data/{name}.csv" for name in SPLIT_ROWS), *(f"rl/{name}" for name in MODEL_FILES)}
        if paths != expected.keys() or hash_files(assets, sorted(paths)) != expected:
            raise ValueError("RQ-VAE inference asset identity differs from its preparation manifest")
        if expected["sid.json"]["sha256"] == SID_SHA:
            raise ValueError("Legacy SID cannot be silently reused for an RQ-VAE run")
        super().__init__(assets, expected_unique_sids=self.manifest["unique_sids"])

    def build_drafter(self, name, device):
        if name != "unisrec":
            raise ValueError("RQ-VAE inference accepts only UniSRec and its no-retrieval beam baseline")
        return UniSRecDrafter(self.catalog, self.assets, device,
                              checkpoint_sha256=self.manifest["unisrec_checkpoint_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "legacy-data", "legacy-run", "rl-run", "unisrec-checkpoint"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.assets, args.legacy_data, args.legacy_run, args.rl_run,
                          args.unisrec_checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
