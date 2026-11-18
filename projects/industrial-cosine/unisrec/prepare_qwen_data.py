#!/usr/bin/env python3
"""Prepare MiniOneRec's fixed Qwen features and existing sequence splits.

This command uses only Python's standard library. It never loads a model or
starts training. Original item tokens (including the valid item ``0``) are
preserved; RecBole is responsible for adding its own internal padding ID.
"""

import argparse
from array import array
import ast
import csv
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = next((parent for parent in PROJECT_DIR.parents if (parent / "env.sh").is_file()), PROJECT_DIR.parent.parent)
DEFAULT_DATASET = "Industrial_and_Scientific"
DEFAULT_SOURCE_DIR = WORKSPACE_DIR / "MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey"
DEFAULT_EMBEDDING_FILE = WORKSPACE_DIR / f"MiniOneRec/data/Amazon/index/{DEFAULT_DATASET}.emb-qwen-td.npy"
FIELDS = ["user_id:token", "item_id_list:token_seq", "item_id:token"]
SPLITS = ("train", "valid", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source, destination):
    """Prefer a link, but support OSS mounts that do not implement symlinks."""
    try:
        Path(destination).symlink_to(Path(source))
        return "symlink"
    except OSError as error:
        if error.errno not in (errno.EOPNOTSUPP, errno.EPERM, errno.ENOSYS):
            raise
        shutil.copy2(source, destination)
        return "copy"


def item_token(value):
    require(isinstance(value, (str, int)) and not isinstance(value, bool), "item IDs must be strings or integers")
    token = str(value)
    require(token and token != "[PAD]" and not any(c.isspace() for c in token), f"invalid item token: {token!r}")
    return token


def inspect_embeddings(path, expected_dim):
    """Validate C-order float16/float32 NPY without importing NumPy or Torch."""
    path = Path(path)
    with path.open("rb") as handle:
        require(handle.read(6) == b"\x93NUMPY", "embedding file is not a .npy array")
        version = handle.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "unsupported .npy version")
        size_bytes = 2 if version[0] == 1 else 4
        length_bytes = handle.read(size_bytes)
        require(len(length_bytes) == size_bytes, "truncated .npy header")
        header_length = struct.unpack("<H" if size_bytes == 2 else "<I", length_bytes)[0]
        require(header_length <= 1024 * 1024, "unreasonably large .npy header")
        header_bytes = handle.read(header_length)
        require(len(header_bytes) == header_length, "truncated .npy header")
        header = ast.literal_eval(header_bytes.decode("utf-8" if version[0] == 3 else "latin1").strip())
        require(isinstance(header, dict), "invalid .npy header")
        shape = header.get("shape")
        require(isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(n, int) and n > 0 for n in shape),
                "embeddings must have nonempty shape (items, dimensions)")
        require(shape[1] == expected_dim, f"expected {expected_dim} embedding dimensions, got {shape[1]}")
        descriptor = header.get("descr")
        require(descriptor in ("<f4", ">f4", "=f4", "<f2", ">f2", "=f2"), "embeddings must have float16 or float32 dtype")
        itemsize = int(descriptor[-1])
        dtype = "float32" if itemsize == 4 else "float16"
        require(header.get("fortran_order") is False, "embeddings must use C row order")
        offset = handle.tell()
        require(path.stat().st_size == offset + shape[0] * shape[1] * itemsize, "embedding payload size does not match .npy shape")
        norms = []
        remaining = shape[0]
        while remaining:
            row_count = min(128, remaining)
            raw = handle.read(row_count * shape[1] * itemsize)
            if itemsize == 2:
                values = [value[0] for value in struct.iter_unpack(descriptor[0] + "e", raw)]
            else:
                values = array("f")
                values.frombytes(raw)
                require(values.itemsize == 4, "platform float size is not 32 bits")
                file_byteorder = sys.byteorder if descriptor[0] == "=" else ("little" if descriptor[0] == "<" else "big")
                if file_byteorder != sys.byteorder:
                    values.byteswap()
            require(all(math.isfinite(value) for value in values), "embeddings contain non-finite values")
            for start in range(0, len(values), shape[1]):
                norm = math.sqrt(math.fsum(value * value for value in values[start:start + shape[1]]))
                require(norm > 0, "embeddings contain zero vectors")
                norms.append(norm)
            remaining -= row_count
    return {"shape": list(shape), "dtype": dtype, "all_finite": True, "nonzero_rows": shape[0],
            "l2_norm": {"min": min(norms), "max": max(norms), "mean": math.fsum(norms) / len(norms)},
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def read_asins(path, expected_ids):
    asin_by_id = {}
    seen_asins = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 2, f"invalid item2id line {number}")
            asin, raw_id = fields
            token = item_token(raw_id)
            require(asin and asin not in seen_asins and token not in asin_by_id, "duplicate ASIN/item ID in item2id")
            asin_by_id[token] = asin
            seen_asins.add(asin)
    require(set(asin_by_id) == expected_ids, "item2id IDs do not match embedding row IDs")
    return asin_by_id


def convert_split(source, destination, catalog_ids, max_history_len, previous_split_keys, split):
    seen = set()
    train_ids = set()
    users = set()
    duplicates = 0
    repeat_targets = 0
    lengths = []
    with Path(source).open(encoding="utf-8-sig", newline="") as source_file, Path(destination).open("w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(source_file, delimiter="\t")
        require(reader.fieldnames == FIELDS, f"unexpected split header in {source}: {reader.fieldnames}")
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELDS)
        for number, row in enumerate(reader, 2):
            require(set(row) == set(FIELDS) and all(value is not None for value in row.values()), f"malformed row {source}:{number}")
            user = row[FIELDS[0]]
            require(user and user != "[PAD]" and not any(c.isspace() for c in user), f"invalid user token at {source}:{number}")
            full_history = tuple(item_token(value) for value in row[FIELDS[1]].split())
            target = item_token(row[FIELDS[2]])
            require(full_history, f"empty history at {source}:{number}")
            require(set(full_history).issubset(catalog_ids) and target in catalog_ids, f"unknown item ID at {source}:{number}")
            history = full_history[-max_history_len:]
            key = (user, history, target)
            require(key not in previous_split_keys, f"same sequence sample occurs in different splits at {source}:{number}")
            duplicates += key in seen
            seen.add(key)
            repeat_targets += target in history
            users.add(user)
            lengths.append(len(history))
            train_ids.update(full_history)
            train_ids.add(target)
            writer.writerow((user, " ".join(history), target))
    require(lengths, f"empty {split} split")
    return ({"rows": len(lengths), "users": len(users), "history_length_min": min(lengths),
             "history_length_max": max(lengths), "history_length_mean": math.fsum(lengths) / len(lengths),
             "within_split_duplicate_rows_preserved": duplicates, "target_already_in_history_rows": repeat_targets},
            seen, train_ids)


def prepare(source_dir, output_dir, dataset=DEFAULT_DATASET, max_history_len=10, embedding_dim=2560, overwrite=False,
            embedding_file=None, row_ids_file=None):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", dataset)), "dataset name must contain only letters, digits, underscore or dash")
    require(max_history_len > 0 and embedding_dim > 0, "history length and embedding dimension must be positive")
    require(source_dir != output_dir, "output directory must differ from source directory")
    require(not output_dir.exists() or overwrite, f"output already exists: {output_dir}; use --overwrite to replace generated files")
    embedding_name = f"{dataset}.emb-qwen-td.npy"
    row_ids_name = embedding_name + ".ids.json"
    embedding_file = Path(embedding_file).resolve() if embedding_file else source_dir / embedding_name
    require(embedding_file.is_file(), f"required embedding file missing: {embedding_file}")
    if row_ids_file is not None:
        require(Path(row_ids_file).is_file(), f"explicit row-ID file missing: {row_ids_file}")
    row_ids_file = Path(row_ids_file).resolve() if row_ids_file else Path(str(embedding_file) + ".ids.json")
    generation_file = Path(str(embedding_file) + ".manifest.json")
    source_names = [f"{dataset}.item.json", f"{dataset}.index.json"] + [f"{dataset}.{split}.inter" for split in SPLITS]
    for name in source_names:
        require((source_dir / name).is_file(), f"required source file missing: {source_dir / name}")
    embeddings = inspect_embeddings(embedding_file, embedding_dim)
    raw_row_ids = load_json(row_ids_file) if row_ids_file.is_file() else list(range(embeddings["shape"][0]))
    require(isinstance(raw_row_ids, list), "embedding row IDs must be a JSON list")
    row_ids = [item_token(value) for value in raw_row_ids]
    require(len(row_ids) == embeddings["shape"][0], "embedding row ID count does not match shape")
    require(len(set(row_ids)) == len(row_ids), "embedding row IDs must be unique")
    catalog_ids = set(row_ids)
    metadata = load_json(source_dir / f"{dataset}.item.json")
    require(isinstance(metadata, dict) and set(metadata) == catalog_ids, "item metadata IDs do not match embedding row IDs")
    sids = load_json(source_dir / f"{dataset}.index.json")
    require(isinstance(sids, dict) and set(sids) == catalog_ids, "SID mapping must cover the entire embedding catalog exactly")
    for token, tokens in sids.items():
        require(isinstance(tokens, list) and tokens and all(isinstance(t, str) and re.fullmatch(r"<[^<>\s]+>", t) for t in tokens),
                f"invalid SID token list for item {token}")
    sid_groups = {}
    for token, tokens in sids.items():
        sid_groups.setdefault("".join(tokens), []).append(token)
    collision_groups = [{"sid": sid, "item_ids": tokens} for sid, tokens in sid_groups.items() if len(tokens) > 1]
    item2id_file = source_dir / f"{dataset}.item2id"
    asin_by_id = read_asins(item2id_file, catalog_ids) if item2id_file.is_file() else {token: metadata[token].get("asin") for token in row_ids}
    generation = load_json(generation_file) if generation_file.is_file() else None
    if generation is not None:
        require(generation.get("shape") == embeddings["shape"], "embedding generation manifest shape mismatch")
        require(generation.get("dtype") == embeddings["dtype"], "embedding generation manifest dtype mismatch")
        require(generation.get("embedding_sha256") == embeddings["sha256"], "embedding hash differs from generation manifest")
    source_hashes = {name: sha256_file(source_dir / name) for name in source_names if name != embedding_name}
    source_hashes[embedding_name] = embeddings["sha256"]
    if generation is not None:
        require(generation.get("item_metadata_sha256") == source_hashes[f"{dataset}.item.json"], "item metadata hash differs from embedding generation manifest")
    companion_metadata = embedding_file.parent / f"{dataset}.item.json"
    companion_matches = None
    if companion_metadata.is_file():
        companion_matches = load_json(companion_metadata) == metadata
        require(companion_matches, "embedding companion metadata differs from interaction catalog metadata")
    for optional_file in (row_ids_file, generation_file, item2id_file):
        if optional_file.is_file():
            source_hashes[optional_file.name] = sha256_file(optional_file)
    catalog = [{"embedding_row": row, "item_id": token, "asin": asin_by_id[token],
                "sid_tokens": sids[token], "sid": "".join(sids[token])} for row, token in enumerate(row_ids)]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-prepare-", dir=output_dir.parent) as staging_name:
        staging = Path(staging_name)
        split_stats = {}
        previous_keys = set()
        train_ids = set()
        for split in SPLITS:
            name = f"{dataset}.{split}.inter"
            stats, keys, visible_ids = convert_split(source_dir / name, staging / name, catalog_ids,
                                                    max_history_len, previous_keys, split)
            split_stats[split] = stats
            previous_keys.update(keys)
            if split == "train":
                train_ids = visible_ids
        with (staging / f"{dataset}.item").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["item_id:token"])
            writer.writerows((token,) for token in row_ids)
        write_json(staging / "train_item_ids.json", [token for token in row_ids if token in train_ids])
        write_json(staging / "catalog.json", catalog)
        embedding_storage = link_or_copy(embedding_file, staging / embedding_name)
        if row_ids_file.is_file():
            row_ids_storage = link_or_copy(row_ids_file, staging / row_ids_name)
        else:
            write_json(staging / row_ids_name, row_ids)
            row_ids_storage = "generated"
        manifest = {"schema_version": 1, "dataset": dataset, "source_dir": str(source_dir), "prepared_dir": str(output_dir),
                    "training_started": False, "item_count": len(row_ids), "train_item_count": len(train_ids),
                    "train_unseen_item_count": len(catalog_ids - train_ids), "max_history_len": max_history_len,
                    "embedding": {**embeddings, "file": embedding_name, "row_ids": row_ids_name,
                                  "frozen": True, "source_file": str(embedding_file),
                                  "storage_mode": embedding_storage, "row_ids_storage_mode": row_ids_storage,
                                  "source_model": generation.get("model_id") if generation else None,
                                  "generation_manifest_available": generation is not None,
                                  "model_identity_verified": False,
                                  "row_ids_source": str(row_ids_file) if row_ids_file.is_file() else "inferred contiguous catalog IDs 0..N-1; original row provenance unavailable",
                                  "row_order_provenance_verified": row_ids_file.is_file(),
                                  "companion_metadata_matches_catalog": companion_matches,
                                  "generation_settings": generation},
                    "splits": split_stats, "source_sha256": source_hashes,
                    "sid_index": f"{dataset}.index.json", "sid_collisions": {"groups": collision_groups,
                                  "group_count": len(collision_groups), "unique_sids": len(sid_groups),
                                  "affected_items": sum(len(group["item_ids"]) for group in collision_groups)},
                    "cross_split_duplicate_samples": 0,
                    "notes": ["Original item tokens are preserved, including valid item ID 0; RecBole assigns padding internally.",
                              "Existing benchmark train/valid/test splits are retained; no re-splitting or sequence augmentation.",
                              "Within-split duplicate rows and repeat-purchase targets are preserved from the source data.",
                              "SID mapping uses MiniOneRec's final .index.json, as read by its SFT data preparation.",
                              "RQ-VAE sid.mapping.csv partitions are unrelated to interaction train/valid/test splits.",
                              "Train-item IDs include source train histories and targets; strict cold-item training must restrict its CE candidates to these IDs.",
                              "The full .item catalog includes train-unseen items for inductive evaluation and later export.",
                              "Existing SID collisions are preserved; item-level recall and SID-level verification can differ.",
                              "Missing embedding generation manifests and row-ID files are recorded explicitly, without claiming verified model identity.",
                              "Feature files and existing row-ID files use symlinks where supported, otherwise byte-preserving copies; storage modes are recorded."]}
        manifest["prepared_sha256"] = {path.name: sha256_file(path) for path in staging.iterdir()}
        write_json(staging / "manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Publish manifest last so a completed manifest describes all generated files.
        for path in sorted(staging.iterdir(), key=lambda path: path.name == "manifest.json"):
            destination = output_dir / path.name
            require(not destination.is_dir(), f"cannot replace directory with generated file: {destination}")
            path.replace(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / f"prepared/{DEFAULT_DATASET}")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--embedding-file", type=Path,
                        help=".npy path; the default Industrial catalog uses the shared Amazon/index feature file")
    parser.add_argument("--row-ids-file", type=Path)
    parser.add_argument("--max-history-len", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    embedding_file = args.embedding_file
    if embedding_file is None and args.dataset == DEFAULT_DATASET and args.source_dir.resolve() == DEFAULT_SOURCE_DIR.resolve():
        embedding_file = DEFAULT_EMBEDDING_FILE
    try:
        manifest = prepare(args.source_dir, args.output_dir, args.dataset, args.max_history_len, args.embedding_dim, args.overwrite,
                           embedding_file, args.row_ids_file)
    except (ValueError, OSError, SyntaxError, UnicodeError, json.JSONDecodeError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")
    print(json.dumps({"prepared_dir": manifest["prepared_dir"], "items": manifest["item_count"],
                      "train_items": manifest["train_item_count"],
                      "split_rows": {split: stats["rows"] for split, stats in manifest["splits"].items()},
                      "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
