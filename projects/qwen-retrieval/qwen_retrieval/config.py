"""Versioned, dependency-free contracts shared by training and retrieval."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


CATEGORY = "Industrial_and_Scientific"
ITEM_TEMPLATE = "Item SID: {sid}\n### Embedding:"
TEMPLATE_VERSION = 1
PROJECTION_DIM = 256
RETRIEVAL_WEIGHT = 0.1
TEMPERATURE = 0.07
TOP_DELTA = 200
APPROVED_SID_SHA256 = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
INPUT_DIR = WORKSPACE / "MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey"
DATA_DIR = WORKSPACE / "MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey"
BASE_MODEL = WORKSPACE / "weights/Qwen2.5-0.5B"
HISTORICAL_MODEL = WORKSPACE / "weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1/epochs3/sft/final_checkpoint"
HISTORICAL_METRICS = {"HR@10": 0.12949481579527908, "NDCG@10": 0.08300085599636473,
                      "HR@20": 0.16479152878888154, "NDCG@20": 0.09190191960470526,
                      "HR@50": 0.22104566512243548, "NDCG@50": 0.10308449676418517}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_catalog(path):
    """Return deterministic unique-SID rows, retaining every original item ID."""
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("SID catalog must be a nonempty object")
    groups = defaultdict(list)
    for item_id, tokens in catalog.items():
        if not isinstance(tokens, list) or len(tokens) != 3 or any(
            not isinstance(token, str) or re.fullmatch(rf"<{layer}_[0-9]+>", token) is None
            for layer, token in zip("abc", tokens)
        ):
            raise ValueError(f"Industrial original SID must contain exactly a/b/c tokens: {item_id}")
        groups["".join(tokens)].append(str(item_id))
    return [{"sid": sid, "item_ids": sorted(ids, key=lambda value: (not value.isdecimal(),
                                                                 int(value) if value.isdecimal() else value))}
            for sid, ids in sorted(groups.items())]


def checkpoint_identity(path):
    """Content identity includes the joint weight file (and hence projection head)."""
    path = Path(path)
    files = [p for p in path.iterdir() if p.is_file() and (
        p.suffix == ".safetensors" or p.name.startswith("pytorch_model")
        or p.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json"})]
    if not (path / "config.json").is_file() or not any(
        p.suffix == ".safetensors" or p.name.startswith("pytorch_model") for p in files
    ):
        raise ValueError("Checkpoint is missing configuration or model weights")
    hashes = {p.name: hash_file(p) for p in sorted(files)}
    return {"files_sha256": hashes, "sha256": json_digest(hashes)}
