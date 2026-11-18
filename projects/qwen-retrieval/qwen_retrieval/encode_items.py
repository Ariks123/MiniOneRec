"""Export the joint Qwen model's normalized unique-SID item embeddings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from qwen_retrieval.config import (
    APPROVED_SID_SHA256, CATEGORY, ITEM_TEMPLATE, PROJECTION_DIM, TEMPLATE_VERSION,
    build_catalog, checkpoint_identity, hash_file, json_digest,
)
from scripts.sft_eval_utils import atomic_json, require, verify_sid_tokenizer


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--input-dir", type=Path)
    result.add_argument("--sid-index-path", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--device", choices=("auto", "cpu"), default="auto")
    result.add_argument("--expected-sid-sha256", default=APPROVED_SID_SHA256)
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_joint_model(model_path, device):
    import torch
    from transformers import AutoTokenizer
    from qwen_retrieval.model import QwenRetrievalForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    require(tokenizer.eos_token_id is not None, "Tokenizer must define EOS")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model, loading = QwenRetrievalForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa", local_files_only=True, output_loading_info=True,
    )
    require("retrieval_head.weight" not in loading.get("missing_keys", [])
            and not any("retrieval_head" in str(key) for key in loading.get("mismatched_keys", [])),
            "Checkpoint projection head is missing or mismatched; refusing random retrieval vectors")
    model.to(device)
    require(getattr(model.config, "retrieval_dim", None) == PROJECTION_DIM, "Unexpected projection dimension")
    require(getattr(model.config, "retrieval_item_template", None) == ITEM_TEMPLATE, "Checkpoint item template differs")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def encode(args):
    import numpy as np
    import torch
    import os

    require(int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "Item encoding is single-process; run without torchrun or with one rank")
    require(args.batch_size > 0, "batch-size must be positive")
    require(args.sid_index_path is not None or args.input_dir is not None,
            "Specify --input-dir or --sid-index-path")
    sid_path = args.sid_index_path or args.input_dir / f"{CATEGORY}.index.json"
    require(hash_file(sid_path) == args.expected_sid_sha256, "SID catalog identity changed")
    rows = build_catalog(sid_path)
    catalog = json.loads(sid_path.read_text(encoding="utf-8"))
    device = torch.device("cuda", 0) if args.device != "cpu" and torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(not (args.output_dir / "manifest.json").exists(), "Index already exists; choose a fresh output directory")
    atomic_json(args.output_dir / "status.json", {"status": "running", "created_at_utc": utc_now()})
    started = time.monotonic()
    identity = checkpoint_identity(args.model_path)
    tokenizer, model = load_joint_model(args.model_path, device)
    dimensions = verify_sid_tokenizer(tokenizer, model, catalog)
    synchronize(torch, device)
    load_seconds = time.monotonic() - started
    encoding_started = time.monotonic()
    batches = []
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            texts = [ITEM_TEMPLATE.format(sid=row["sid"]) for row in rows[offset:offset + args.batch_size]]
            encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
            vectors = model.encode(**batch)
            require(vectors.shape == (len(texts), PROJECTION_DIM), "Invalid encoded item shape")
            batches.append(vectors.float().cpu().numpy())
    synchronize(torch, device)
    vectors = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    require(np.isfinite(vectors).all(), "Item vectors contain non-finite values")
    require(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4), "Item vectors are not normalized")
    matrix_path, mapping_path = args.output_dir / "item_vectors.npy", args.output_dir / "row_to_items.json"
    # Publish the manifest last; an interrupted export cannot be mistaken for a usable index.
    with matrix_path.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    atomic_json(mapping_path, rows)
    manifest = {
        "format_version": 1, "status": "passed", "created_at_utc": utc_now(),
        "category": CATEGORY, "checkpoint_identity": identity,
        "model_path": str(args.model_path.resolve()), "projection_parameter": "retrieval_head.weight",
        "projection_dim": PROJECTION_DIM, "template_version": TEMPLATE_VERSION, "item_template": ITEM_TEMPLATE,
        "sid_index_sha256": hash_file(sid_path), "catalog_rows_sha256": json_digest(rows),
        "unique_sids": len(rows), "catalog_items": sum(len(row["item_ids"]) for row in rows),
        "matrix": {"filename": matrix_path.name, "sha256": hash_file(matrix_path),
                   "shape": list(vectors.shape), "dtype": "float32", "normalized": True},
        "mapping": {"filename": mapping_path.name, "sha256": hash_file(mapping_path)},
        "tokenizer_and_model": dimensions,
        "timing": {"loading_seconds": load_seconds, "encoding_seconds": time.monotonic() - encoding_started,
                   "total_seconds": time.monotonic() - started},
        "metric_semantics": "One vector per complete SID; item_ids preserves all original IDs sharing each SID",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    atomic_json(args.output_dir / "status.json", {"status": "passed", "created_at_utc": utc_now()})
    print(json.dumps({key: manifest[key] for key in ("status", "unique_sids", "catalog_items", "timing")}), flush=True)
    return manifest


def load_index(index_dir, sid_path, identity):
    """Reject stale model, projection, tokenizer, catalog, template or index bytes."""
    import numpy as np

    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "passed" and manifest.get("format_version") == 1,
            "Index export is incomplete or unsupported")
    require(manifest.get("category") == CATEGORY, "Index category differs")
    require(manifest.get("checkpoint_identity") == identity, "Index model/projection/tokenizer version differs")
    require(manifest.get("item_template") == ITEM_TEMPLATE and manifest.get("template_version") == TEMPLATE_VERSION,
            "Index item template version differs")
    require(manifest.get("projection_dim") == PROJECTION_DIM, "Index dimension differs")
    require(manifest.get("sid_index_sha256") == hash_file(sid_path), "Index SID catalog version differs")
    matrix_name, mapping_name = manifest["matrix"]["filename"], manifest["mapping"]["filename"]
    require(Path(matrix_name).name == matrix_name and Path(mapping_name).name == mapping_name,
            "Index paths must be local filenames")
    matrix_path, mapping_path = index_dir / matrix_name, index_dir / mapping_name
    require(hash_file(matrix_path) == manifest["matrix"]["sha256"], "Index matrix checksum differs")
    require(hash_file(mapping_path) == manifest["mapping"]["sha256"], "Index mapping checksum differs")
    rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    require(rows == build_catalog(sid_path) and json_digest(rows) == manifest.get("catalog_rows_sha256"),
            "Index row order or complete item mapping differs")
    vectors = np.load(matrix_path, allow_pickle=False)
    require(vectors.dtype == np.float32 and vectors.shape == (len(rows), PROJECTION_DIM), "Index matrix shape/dtype differs")
    require(manifest["matrix"].get("shape") == list(vectors.shape)
            and manifest["matrix"].get("dtype") == "float32" and manifest["matrix"].get("normalized") is True,
            "Index matrix declaration differs")
    require(manifest.get("unique_sids") == len(rows)
            and manifest.get("catalog_items") == sum(len(row["item_ids"]) for row in rows), "Index counts differ")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4),
            "Index vectors are non-finite or not normalized")
    return vectors, rows, manifest


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        encode(args)
    except Exception as error:
        atomic_json(args.output_dir / "status.json", {
            "status": "failed", "error": str(error), "error_type": type(error).__name__, "created_at_utc": utc_now(),
        })
        raise


if __name__ == "__main__":
    main()
