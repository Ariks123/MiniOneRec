"""Framework-independent integrity checks and ranking metrics for SFT evaluation."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import uuid


TOP_K = (1, 3, 5, 10, 20, 50)
APPROVED_VIDEO_GAMES_SID_SHA256 = "ca9d730905a67cc36cd77c665a3165de823f2689660b19741df9ee884993085f"
CATEGORY_PROMPTS = {"Video_Games": "video games", "Industrial_and_Scientific": "industrial and scientific items"}


class EvaluationError(ValueError):
    """An input or prediction does not meet the evaluation contract."""


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def resolve_expected_sid_sha256(category, expected_sid_sha256=None):
    require(category in CATEGORY_PROMPTS, "Unsupported evaluation category")
    if expected_sid_sha256 is None:
        require(category == "Video_Games", "Industrial evaluation requires an explicit --expected-sid-sha256")
        expected_sid_sha256 = APPROVED_VIDEO_GAMES_SID_SHA256
    require(isinstance(expected_sid_sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected_sid_sha256) is not None,
            "Expected SID SHA-256 must be an explicit 64-digit hexadecimal hash")
    return expected_sid_sha256.lower()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    os.replace(temporary, path)


def read_targets(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames is not None and "item_sid" in reader.fieldnames,
                f"{path}: missing item_sid column")
        targets = [row["item_sid"].strip() for row in reader]
    require(bool(targets) and all(targets), f"{path}: empty dataset or target")
    return targets


def verify_artifacts(model_path, input_dir, data_dir, category, expected_sid_sha256):
    """Verify immutable input identities and the checkpoint's SFT provenance."""
    expected_sid_sha256 = resolve_expected_sid_sha256(category, expected_sid_sha256)
    model_path, input_dir, data_dir = map(Path, (model_path, input_dir, data_dir))
    sid_path = input_dir / f"{category}.index.json"
    metadata_path = input_dir / f"{category}.item.json"
    sid_hash = sha256_file(sid_path)
    require(sid_hash == expected_sid_sha256, "SID differs from the approved catalog")
    metadata_hash = sha256_file(metadata_path)
    catalog = json.loads(sid_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(isinstance(catalog, dict) and bool(catalog), "Catalog must be a nonempty object")
    require(set(catalog) == set(metadata), "Catalog and metadata IDs do not match")
    if category == "Video_Games":
        require(all(isinstance(tokens, list) and len(tokens) == 4
                    and all(isinstance(token, str) and token for token in tokens)
                    for tokens in catalog.values()), "Each catalog item must have four SID tokens")
    else:
        require(all(isinstance(tokens, list) and len(tokens) in (3, 4)
                    and all(isinstance(token, str) and re.fullmatch(rf"<{layer}_[0-9]+>", token)
                            for layer, token in zip("abcd", tokens)) for tokens in catalog.values()),
                "Original Industrial SIDs must contain a/b/c tokens and an optional d token")
        require(len({len(tokens) for tokens in catalog.values()}) == 1,
                "Original Industrial SID token lengths must be consistent")
    valid_sids = {"".join(tokens) for tokens in catalog.values()}
    sid_counts = Counter("".join(tokens) for tokens in catalog.values())
    collision_stats = {"groups": sum(count > 1 for count in sid_counts.values()),
                       "items": sum(count for count in sid_counts.values() if count > 1),
                       "excess_items": len(catalog) - len(valid_sids)}
    if category == "Video_Games":
        require(len(valid_sids) == len(catalog), "Complete catalog SIDs are not unique")

    provenance = json.loads((model_path / "posttraining_provenance.json").read_text(encoding="utf-8"))
    require(provenance.get("stage") == "sft" and provenance.get("category") == category,
            "Checkpoint provenance does not identify this category's SFT")
    require(provenance.get("sid_index_sha256") == sid_hash,
            "Checkpoint was trained with another SID catalog")
    require(provenance.get("item_meta_sha256") == metadata_hash,
            "Checkpoint was trained with other item metadata")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    require(any("ForCausalLM" in name for name in config.get("architectures", [])),
            "Checkpoint is not a causal language model")
    indices = [model_path / name for name in
               ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
    index_path = next((path for path in indices if path.is_file()), None)
    if index_path:
        shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
        require(bool(shards) and all(Path(name).name == name and (model_path / name).is_file()
                                    for name in shards), "Model weight shards are missing or unsafe")
    else:
        require(any((model_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")),
                "Checkpoint model weights are missing")

    manifest = json.loads((data_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    amazon23 = manifest.get('dataset') in ('Amazon Reviews 2023 Video_Games', 'Amazon Reviews 2023 Industrial_and_Scientific')
    if amazon23:
        require(manifest.get('dataset') == f'Amazon Reviews 2023 {category}' and manifest.get('validation') == 'passed',
                'Amazon23 requires the verified matching-category preparation')
        require(manifest.get('history_length_range') == [1, 50]
                and manifest.get('complete_sids_unique') is True
                and manifest.get('complete_sid_tokens') == 4
                and manifest.get('sampling_applied') is False,
                'Amazon23 evaluation requires the complete verified history-50 data')
        require(len(valid_sids) == len(catalog) and all(len(tokens) == 4 for tokens in catalog.values()),
                'Amazon23 requires unique complete four-token SIDs')
        require(manifest['source_sha256'][f'{category}.index.json'] == sid_hash
                and manifest['source_sha256'][f'{category}.item.json'] == metadata_hash,
                'Amazon23 prepared catalog differs from the trained catalog')
        for name, expected in manifest['source_sha256'].items():
            require(Path(name).name == name and sha256_file(input_dir / name) == expected,
                    'Amazon23 frozen source changed: ' + name)
        for name in ('train.csv', 'valid.csv', 'test.csv', 'info.txt'):
            require(sha256_file(data_dir / name) == manifest['outputs'][name]['sha256'],
                    'Amazon23 prepared output changed: ' + name)
        test_targets, valid_targets = read_targets(data_dir / 'test.csv'), read_targets(data_dir / 'valid.csv')
        require(len(test_targets) == manifest['rows']['test'] and len(valid_targets) == manifest['rows']['valid'],
                'Amazon23 evaluation split row counts changed')
        require(all(target in valid_sids for target in test_targets + valid_targets),
                'Amazon23 evaluation target is outside the fixed catalog')
        return {'catalog': catalog, 'valid_sids': valid_sids, 'provenance': provenance,
                'config': config, 'test_targets': test_targets, 'valid_targets': valid_targets,
                'sid_sha256': sid_hash, 'metadata_sha256': metadata_hash,
                'catalog_items': len(catalog), 'unique_sids': len(valid_sids),
                'sid_tokens_per_item': 4, 'collision_groups': 0, 'collision_items': 0,
                'metric_semantics': 'unique-item SID-level; Amazon23 history-50 full test',
                'source_sha256': {'train_file': manifest['outputs']['train.csv']['sha256'],
                                  'eval_file': manifest['outputs']['valid.csv']['sha256'],
                                  'test_file': manifest['outputs']['test.csv']['sha256'],
                                  'sid_index_path': sid_hash}}
    require(manifest.get("validation") == "passed" and manifest.get("check_only") is False,
            "Prepared-data manifest is not a completed successful conversion")
    require(manifest.get("dataset") == category, "Prepared data has a different category")
    if category == "Video_Games":
        require(manifest.get("selected_sid_epoch") == 2900, "Prepared data has a different category or SID epoch")
    else:
        require(manifest.get("original_sid") is True, "Industrial evaluation requires the original SID catalog")
        require(manifest.get("unique_sids") == len(valid_sids), "Original SID unique count differs from the prepared manifest")
        declared_collisions = manifest.get("sid_collisions", {})
        require(isinstance(declared_collisions, dict)
                and all(declared_collisions.get(key) == value for key, value in collision_stats.items()),
                "Original SID collision metadata differs from the catalog")
    require(manifest.get("preserve_all_interaction_rows") is True
            and manifest.get("sampling_applied") is False
            and manifest.get("keep_longest_only") is False,
            "Prepared data must retain all interaction rows and original splits")
    require(manifest["sources"]["sid_index"]["sha256"] == sid_hash
            and manifest["sources"]["item_metadata"]["sha256"] == metadata_hash,
            "Prepared inputs differ from checkpoint inputs")
    for source in manifest["sources"].values():
        name = source["filename"]
        require(Path(name).name == name, "Manifest source path is not local")
        require(sha256_file(input_dir / name) == source["sha256"], "Source changed: " + name)
    for name in ("train.csv", "valid.csv", "test.csv", "info.txt"):
        require(sha256_file(data_dir / name) == manifest["outputs"][name]["sha256"],
                "Prepared output changed: " + name)
    test_targets, valid_targets = read_targets(data_dir / "test.csv"), read_targets(data_dir / "valid.csv")
    require(len(test_targets) == manifest["rows"]["test"]
            and len(valid_targets) == manifest["rows"]["valid"], "Prepared split row counts changed")
    require(all(target in valid_sids for target in test_targets + valid_targets),
            "An evaluation target is outside the catalog")
    return {"catalog": catalog, "valid_sids": valid_sids, "provenance": provenance,
            "config": config, "test_targets": test_targets, "valid_targets": valid_targets,
            "sid_sha256": sid_hash, "metadata_sha256": metadata_hash,
            "catalog_items": len(catalog), "unique_sids": len(valid_sids),
            "sid_tokens_per_item": len(next(iter(catalog.values()))),
            "collision_groups": collision_stats["groups"], "collision_items": collision_stats["items"],
            "metric_semantics": ("original SID-level; colliding item IDs indistinguishable"
                                 if category == "Industrial_and_Scientific" else "unique-item SID-level"),
            "source_sha256": {
                "train_file": manifest["outputs"]["train.csv"]["sha256"],
                "test_file": manifest["outputs"]["test.csv"]["sha256"],
                "sid_index_path": sid_hash,
            }}


def verify_sid_tokenizer(tokenizer, model, catalog):
    tokens = {token for sid in catalog.values() for token in sid}
    vocabulary = tokenizer.get_vocab()
    require(all(token in vocabulary and tokenizer.encode(token, add_special_tokens=False) == [vocabulary[token]]
                for token in tokens), "Checkpoint tokenizer is missing atomic SID tokens")
    input_size = int(model.get_input_embeddings().weight.shape[0])
    output_embeddings = model.get_output_embeddings()
    require(output_embeddings is not None, "Checkpoint has no language-model output embeddings")
    output_size = int(output_embeddings.weight.shape[0])
    config_size = int(model.config.vocab_size)
    require(input_size == output_size == config_size and input_size >= len(tokenizer)
            and input_size > max(vocabulary[token] for token in tokens),
            "Model vocabulary dimensions do not contain the checkpoint tokenizer")
    return {"atomic_sid_tokens": len(tokens), "tokenizer_size": len(tokenizer),
            "model_vocabulary_size": input_size}


class RankingAccumulator:
    def __init__(self, valid_sids, num_beams, top_k=TOP_K):
        self.valid_sids = set(valid_sids)
        require(num_beams >= 1, "num_beams must be positive")
        self.num_beams = num_beams
        self.top_k = tuple(k for k in top_k if k <= num_beams)
        self.rows = 0
        self.hits = {k: 0 for k in self.top_k}
        self.ndcg = {k: 0.0 for k in self.top_k}
        self.examples = {"hit": [], "miss": []}

    def add(self, row):
        target, predictions = row["target"], row["predictions"]
        require(target in self.valid_sids, f"Row {row.get('row_id')}: unknown target SID")
        require(isinstance(predictions, list) and len(predictions) == self.num_beams,
                f"Row {row.get('row_id')}: expected {self.num_beams} candidates")
        require(all(isinstance(item, str) and item in self.valid_sids for item in predictions),
                f"Row {row.get('row_id')}: invalid or truncated prediction SID")
        require(len(set(predictions)) == self.num_beams,
                f"Row {row.get('row_id')}: duplicate candidates")
        rank = predictions.index(target) + 1 if target in predictions else None
        for k in self.top_k:
            if rank is not None and rank <= k:
                self.hits[k] += 1
                self.ndcg[k] += 1.0 / math.log2(rank + 1)
        example_type = "hit" if rank is not None else "miss"
        if len(self.examples[example_type]) < 5:
            self.examples[example_type].append({**row, "rank": rank})
        self.rows += 1

    def result(self):
        require(self.rows > 0, "No predictions were evaluated")
        by_k = {str(k): {"HR": self.hits[k] / self.rows, "NDCG": self.ndcg[k] / self.rows}
                for k in self.top_k}
        flat = {f"{metric}@{k}": value for k, values in by_k.items() for metric, value in values.items()}
        return {"metrics": flat, "metrics_by_k": by_k, "completed_rows": self.rows,
                "validity_rate": 1.0, "unique_rate": 1.0, "CC": 0, "examples": self.examples}


def _shard_rows(path):
    previous = -1
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            require(bool(line.strip()), f"{path}:{line_number}: empty prediction line")
            row = json.loads(line)
            row_id = row.get("row_id")
            require(isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > previous,
                    f"{path}:{line_number}: row IDs must be unique and ascending")
            previous = row_id
            yield row


def merge_prediction_shards(paths, targets, valid_sids, num_beams, output_path):
    """Merge sorted rank shards in O(world_size) row memory and enforce exact coverage."""
    require(bool(paths), "No prediction shards supplied")
    iterators = [iter(_shard_rows(path)) for path in paths]
    heap = []
    for index, iterator in enumerate(iterators):
        row = next(iterator, None)
        if row is not None:
            heapq.heappush(heap, (row["row_id"], index, row))
    accumulator = RankingAccumulator(valid_sids, num_beams)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    next_id = 0
    with temporary.open("w", encoding="utf-8") as destination:
        while heap:
            row_id, index, row = heapq.heappop(heap)
            require(row_id == next_id and row_id < len(targets),
                    f"Prediction coverage failed: expected row {next_id}, received {row_id}")
            require(row.get("target") == targets[row_id], f"Row {row_id}: prediction target differs from test data")
            accumulator.add(row)
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            next_id += 1
            following = next(iterators[index], None)
            if following is not None:
                heapq.heappush(heap, (following["row_id"], index, following))
        require(next_id == len(targets), f"Prediction coverage failed: {next_id}/{len(targets)} rows")
    os.replace(temporary, output_path)
    return {**accumulator.result(), "expected_rows": len(targets),
            "coverage": {"expected_rows": len(targets), "completed_rows": next_id,
                         "missing_rows": 0, "duplicate_rows": 0}}
