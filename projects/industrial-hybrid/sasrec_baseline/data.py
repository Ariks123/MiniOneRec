"""Read the fixed Industrial request splits for an ID-only SASRec baseline."""

import ast
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


KS = (1, 5, 10, 20, 50)
HISTORY_LIMIT = 10
CATALOG_COUNT = 3686
EXPECTED_ROWS = {"train": 36259, "valid": 4532, "test": 4533}
EXPECTED_COLD_TARGETS = {"train": 0, "valid": 125, "test": 147}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Request:
    row_id: int
    history: tuple[int, ...]
    target: int


def read_split(directory, split, expected_sha):
    path = directory / f"{split}.csv"
    if sha256(path) != expected_sha:
        raise ValueError(f"{split} differs from the latest RQ-VAE preparation manifest")
    requests = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row_id, row in enumerate(csv.DictReader(stream)):
            history = ast.literal_eval(row["history_item_id"])
            target = int(row["item_id"])
            if (not isinstance(history, list) or not 1 <= len(history) <= HISTORY_LIMIT
                    or any(type(item) is not int or not 0 <= item < CATALOG_COUNT for item in history)
                    or not 0 <= target < CATALOG_COUNT):
                raise ValueError(f"Invalid product ID or history length in {split} row {row_id}")
            requests.append(Request(row_id, tuple(history), target))
    if len(requests) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split} must preserve all {EXPECTED_ROWS[split]:,} archived requests")
    return requests


class Dataset:
    def __init__(self, directory: Path):
        self.directory = directory
        manifest = read_json(directory / "preparation_manifest.json")
        if (manifest["original_sid"] is not False or manifest["items"] != CATALOG_COUNT
                or manifest["rows"] != EXPECTED_ROWS):
            raise ValueError("Expected the completed Industrial RQ-VAE preparation manifest")
        self.split_sha256 = {split: manifest["outputs"][f"{split}.csv"]["sha256"]
                             for split in EXPECTED_ROWS}
        self.preparation_sha256 = sha256(directory / "preparation_manifest.json")
        self.training = self.read("train")
        frequencies = Counter(row.target for row in self.training)
        seen = set(frequencies)
        seen.update(item for row in self.training for item in row.history)
        if len(seen) != 3647:
            raise ValueError("Training must identify exactly 3,647 train-seen products")
        self.candidates = sorted(seen)
        self.compact = {item: i for i, item in enumerate(self.candidates, 1)}
        self.popularity = sorted(seen, key=lambda item: (-frequencies[item], item))

    def read(self, split):
        return read_split(self.directory, split, self.split_sha256[split])

    def encode_history(self, history):
        encoded = [self.compact[item] for item in history if item in self.compact]
        return encoded + [0] * (HISTORY_LIMIT - len(encoded)), len(encoded)

    def diagnostics(self, split, requests):
        unknown = [sum(item not in self.compact for item in row.history) for row in requests]
        cold = sum(row.target not in self.compact for row in requests)
        if cold != EXPECTED_COLD_TARGETS[split]:
            raise ValueError(f"The training-defined {split} cold split changed")
        return {
            "rows": len(requests),
            "cold_targets": cold,
            "unknown_history_rows": sum(count > 0 for count in unknown),
            "unknown_history_items": sum(unknown),
            "empty_history_rows": sum(count == len(row.history) for count, row in zip(unknown, requests, strict=True)),
            "repeat_target_rows": sum(row.target in row.history for row in requests),
        }
