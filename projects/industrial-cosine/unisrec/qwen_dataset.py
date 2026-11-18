"""RecBole benchmark dataset backed by the existing, frozen Qwen .npy features."""

import json
from pathlib import Path

import numpy as np
import torch
from recbole.data.dataset import SequentialDataset
from torch import nn


def _config_value(config, key, default=None):
    value = config[key] if key in config else None
    return default if value is None else value


def _read_ids(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Expected a nonempty JSON list of item IDs: {path}")
    if any(isinstance(v, bool) or not isinstance(v, (int, str)) for v in values):
        raise ValueError(f"Item IDs must be integers or strings: {path}")
    tokens = [str(value) for value in values]
    if any(not token or token == "[PAD]" for token in tokens):
        raise ValueError(f"Empty IDs and the reserved [PAD] token are forbidden: {path}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate item IDs: {path}")
    return tokens


class QwenUniSRecDataset(SequentialDataset):
    """Keep the supplied train/valid/test examples and map feature rows by raw token.

    Raw product ID ``0`` is a real product. RecBole's separate internal ID ``0``
    is padding, so feature rows are never indexed with internal IDs directly.
    The prepared ``.item`` file must enumerate the full retrieval catalog.
    """

    def __init__(self, config):
        if list(_config_value(config, "benchmark_filename", [])) != ["train", "valid", "test"]:
            raise ValueError("QwenUniSRec requires benchmark_filename: [train, valid, test]")
        super().__init__(config)
        self.plm_size = int(config["plm_size"])
        root = Path(config["data_path"])
        feature_path = Path(_config_value(
            config, "embedding_path", root / f"{self.dataset_name}.emb-qwen-td.npy"
        ))
        ids_path = Path(_config_value(config, "embedding_ids_path", f"{feature_path}.ids.json"))
        train_ids_path = Path(_config_value(config, "train_item_ids_path", root / "train_item_ids.json"))
        self.embedding_path = feature_path
        self.embedding_ids_path = ids_path

        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        raw_tokens = _read_ids(ids_path)
        expected_shape = (len(raw_tokens), self.plm_size)
        if features.shape != expected_shape or not np.issubdtype(features.dtype, np.floating):
            raise ValueError(
                f"Expected floating Qwen features with shape {expected_shape}; "
                f"found {features.shape}, {features.dtype}"
            )
        for start in range(0, len(features), 4096):
            chunk = np.asarray(features[start:start + 4096], dtype=np.float32)
            if not np.isfinite(chunk).all() or np.any(np.linalg.norm(chunk, axis=1) == 0):
                raise ValueError("Qwen features contain NaN/Inf or zero item vectors")

        row_by_token = {token: row for row, token in enumerate(raw_tokens)}
        internal_tokens = [str(token) for token in self.field2id_token[self.iid_field]]
        catalog_tokens = set(internal_tokens[1:])
        if set(raw_tokens) != catalog_tokens:
            missing_features = sorted(catalog_tokens - set(raw_tokens))[:8]
            missing_catalog = sorted(set(raw_tokens) - catalog_tokens)[:8]
            raise ValueError(
                "Prepared .item catalog and Qwen IDs must match exactly; "
                f"missing feature IDs={missing_features}, missing catalog IDs={missing_catalog}"
            )
        mapped = np.zeros((self.item_num, self.plm_size), dtype=np.float32)
        for internal_id, token in enumerate(internal_tokens[1:], start=1):
            mapped[internal_id] = features[row_by_token[token]]
        self.plm_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(mapped), freeze=True, padding_idx=0
        )

        train_tokens = _read_ids(train_ids_path)
        unknown = sorted(set(train_tokens) - catalog_tokens)
        if unknown:
            raise ValueError(f"Training item IDs are outside the prepared catalog: {unknown[:8]}")
        token_to_internal = self.field2token_id[self.iid_field]
        self.train_item_ids = torch.tensor(
            sorted(token_to_internal[token] for token in train_tokens), dtype=torch.long
        )
