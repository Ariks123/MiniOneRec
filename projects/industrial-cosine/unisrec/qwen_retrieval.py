"""Export a trained UniSRec catalog and draft item IDs/SIDs for MiniOneRec.

No training or embedding-model calls occur in this module. NumPy, PyTorch and
RecBole are imported only when an export or inference command is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_catalog(path: str | Path) -> list[dict]:
    """Read the preparation script's explicit embedding-row/item/ASIN/SID map."""
    with Path(path).open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog must be a nonempty JSON list")
    seen_ids, seen_rows = set(), set()
    for item in catalog:
        if not isinstance(item, dict):
            raise ValueError("each catalog record must be an object")
        token = str(item.get("item_id", ""))
        row = item.get("embedding_row")
        sid_tokens = item.get("sid_tokens")
        if not token or token == "[PAD]" or token in seen_ids:
            raise ValueError(f"empty, reserved or duplicate catalog item ID: {token!r}")
        if not isinstance(row, int) or isinstance(row, bool) or row < 0 or row in seen_rows:
            raise ValueError(f"invalid or duplicate embedding row for item {token}")
        if not isinstance(sid_tokens, list) or not sid_tokens or not all(
            isinstance(part, str) and part for part in sid_tokens
        ):
            raise ValueError(f"missing SID tokens for item {token}")
        sid = "".join(sid_tokens)
        if item.get("sid") != sid:
            raise ValueError(f"inconsistent final SID for item {token}")
        # Some MiniOneRec catalogs preserve opaque item IDs without ASIN
        # metadata, and several distinct items can share one final SID.
        # Item IDs, rather than SIDs or ASINs, identify retrieval candidates.
        asin = item.get("asin")
        if asin is not None and (not isinstance(asin, str) or not asin):
            raise ValueError(f"ASIN must be a nonempty string or null for item {token}")
        item["item_id"] = token
        item["asin"] = asin
        seen_ids.add(token)
        seen_rows.add(row)
    if seen_rows != set(range(len(catalog))):
        raise ValueError("catalog embedding rows must cover 0..N-1 exactly")
    return sorted(catalog, key=lambda item: item["embedding_row"])


def _write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _catalog_path(config) -> Path:
    # RecBole appends dataset name to data_path during configuration loading.
    configured = config["catalog_path"]
    if configured:
        return Path(configured)
    return Path(config["data_path"]) / "catalog.json"


def _item_token_map(dataset) -> dict[str, int]:
    field = getattr(dataset, "iid_field", "item_id")
    return {str(token): int(index) for token, index in dataset.field2token_id[field].items()}


def _assert_training_checkpoint(checkpoint: dict) -> None:
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("export requires a trained checkpoint with a nonnegative epoch")


def export_index(config_path: str | Path, checkpoint_path: str | Path,
                 output_dir: str | Path, batch_size: int = 512,
                 device: str = "cpu") -> Path:
    """Apply the trained adaptor in eval mode to the complete fixed catalog.

    The output is a normalized float32 .npy matrix plus a row-aligned item/SID
    map. Exact inner-product search is provided below, so FAISS is optional.
    """
    import numpy as np
    import torch
    from qwen_runtime import load_experiment, load_trained_checkpoint

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"trained checkpoint does not exist: {checkpoint_path}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing index: {output_dir}")
    config, dataset, _, model = load_experiment(config_path, device=device)
    checkpoint = load_trained_checkpoint(model, checkpoint_path)
    _assert_training_checkpoint(checkpoint)
    catalog_path = _catalog_path(config)
    catalog = read_catalog(catalog_path)
    token_map = _item_token_map(dataset)
    if len(catalog) != dataset.item_num - 1:
        raise ValueError("catalog must include every non-padding model item")
    internal_ids = []
    for item in catalog:
        internal = token_map.get(item["item_id"])
        if internal is None or internal == 0:
            raise ValueError(f"catalog item absent from model vocabulary: {item['item_id']}")
        internal_ids.append(internal)
    model.eval()
    model_device = next(model.parameters()).device
    hidden_size = int(config["hidden_size"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        vectors = np.lib.format.open_memmap(
            staging / "item_vectors.npy", mode="w+", dtype=np.float32,
            shape=(len(catalog), hidden_size),
        )
        with torch.inference_mode():
            for start in range(0, len(catalog), batch_size):
                ids = torch.tensor(internal_ids[start:start + batch_size],
                                   dtype=torch.long, device=model_device)
                encoded = model.encode_items(ids).detach().cpu().float().numpy()
                norms = np.linalg.norm(encoded, axis=1)
                if encoded.shape != (len(ids), hidden_size) or not np.isfinite(encoded).all():
                    raise ValueError("adaptor returned invalid catalog vectors")
                if not np.allclose(norms, 1.0, atol=1e-5):
                    raise ValueError("adaptor catalog vectors must be nonzero and L2-normalized")
                vectors[start:start + len(ids)] = encoded
        vectors.flush()
        del vectors
        _write_json(staging / "catalog.json", catalog)
        _write_json(staging / "item_ids.json", [item["item_id"] for item in catalog])
        _write_json(staging / "index_manifest.json", {
            "format_version": 1,
            "item_count": len(catalog),
            "dimension": hidden_size,
            "dtype": "float32",
            "normalized": True,
            "similarity": "inner_product",
            "train_stage": "inductive_ft",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_catalog_sha256": sha256_file(catalog_path),
            "catalog_sha256": sha256_file(staging / "catalog.json"),
            "vectors_sha256": sha256_file(staging / "item_vectors.npy"),
            "item_ids_sha256": sha256_file(staging / "item_ids.json"),
        })
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


class MiniOneRecDrafter:
    """History -> UniSRec query -> catalog item IDs -> MiniOneRec final SIDs.

    The query and candidates share UniSRec's trained 300-dimensional space.
    The SID is an opaque handoff identifier; no hidden-state alignment occurs.
    """

    def __init__(self, model, dataset, index_dir: str | Path,
                 checkpoint_path: str | Path):
        import numpy as np

        self.model = model.eval()
        self.token_map = _item_token_map(dataset)
        self.index_dir = Path(index_dir).resolve()
        self.catalog = read_catalog(self.index_dir / "catalog.json")
        manifest = json.loads((self.index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("train_stage") != "inductive_ft":
            raise ValueError("unsupported index format or training mode")
        if not manifest.get("normalized") or manifest.get("similarity") != "inner_product":
            raise ValueError("expected a normalized inner-product catalog index")
        if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
            raise ValueError("query model checkpoint differs from catalog index checkpoint")
        for name, field in [("catalog.json", "catalog_sha256"),
                            ("item_vectors.npy", "vectors_sha256"),
                            ("item_ids.json", "item_ids_sha256")]:
            if sha256_file(self.index_dir / name) != manifest.get(field):
                raise ValueError(f"index file hash mismatch: {name}")
        ids = json.loads((self.index_dir / "item_ids.json").read_text(encoding="utf-8"))
        if ids != [item["item_id"] for item in self.catalog]:
            raise ValueError("index vector IDs and catalog rows disagree")
        self.vectors = np.load(self.index_dir / "item_vectors.npy", mmap_mode="r", allow_pickle=False)
        expected_shape = (len(self.catalog), int(manifest["dimension"]))
        if self.vectors.shape != expected_shape or len(self.catalog) != manifest["item_count"]:
            raise ValueError("index shape does not match catalog manifest")
        if not np.isfinite(self.vectors).all() or not np.allclose(
            np.linalg.norm(self.vectors, axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("index vectors must be finite and L2-normalized")
        self.max_history = int(model.max_seq_length)
        if self.max_history <= 0:
            raise ValueError("model max_seq_length must be positive")

    def draft(self, history_item_ids, topk: int = 100,
              exclude_seen: bool = False) -> list[dict]:
        import numpy as np
        import torch

        if isinstance(history_item_ids, (str, bytes)):
            raise ValueError("history_item_ids must be a list of original item IDs")
        history = [str(item) for item in history_item_ids]
        if not history:
            raise ValueError("at least one history item is required")
        if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        recent = history[-self.max_history:]
        internal = []
        for token in recent:
            item_id = self.token_map.get(token)
            if item_id is None or item_id == 0:
                raise ValueError(f"history item absent from current catalog: {token}")
            internal.append(item_id)
        model_device = next(self.model.parameters()).device
        sequence = torch.zeros((1, self.max_history), dtype=torch.long, device=model_device)
        sequence[0, :len(internal)] = torch.tensor(internal, dtype=torch.long, device=model_device)
        lengths = torch.tensor([len(internal)], dtype=torch.long, device=model_device)
        self.model.eval()
        with torch.inference_mode():
            query = self.model.encode_history(sequence, lengths).detach().cpu().float().numpy()
        if query.shape != (1, self.vectors.shape[1]) or not np.isfinite(query).all():
            raise ValueError("history encoder returned an invalid query")
        if not np.allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-5):
            raise ValueError("history query must be nonzero and L2-normalized")
        scores = np.asarray(self.vectors @ query[0], dtype=np.float32)
        if exclude_seen:
            seen = set(history)
            for index, item in enumerate(self.catalog):
                if item["item_id"] in seen:
                    scores[index] = -np.inf
        # Stable sorting makes tied scores deterministic in embedding-row order.
        order = np.argsort(-scores, kind="stable")
        candidates = []
        for index in order[:min(topk, len(order))]:
            score = float(scores[index])
            if not math.isfinite(score):
                continue
            item = self.catalog[int(index)]
            candidates.append({
                "item_id": item["item_id"], "asin": item["asin"],
                "sid_tokens": list(item["sid_tokens"]), "sid": item["sid"],
                "score": score,
            })
        return candidates

    def verification_payload(self, history_item_ids, topk: int = 100,
                             exclude_seen: bool = False) -> dict:
        """Return a serializable handoff for a MiniOneRec candidate verifier.

        The receiving verifier can tokenize ``candidate_sids`` with its own
        existing SID tokenizer and compare candidate likelihoods. Distinct
        item IDs sharing one SID remain separate candidates; MiniOneRec gives
        them the same SID verification signal and cannot distinguish their
        original IDs through that SID alone. This method does not invoke or
        change MiniOneRec and does not claim verification.
        """
        history = [str(item) for item in history_item_ids]
        candidates = self.draft(history, topk=topk, exclude_seen=exclude_seen)
        return {
            "history_item_ids": history,
            "candidates": candidates,
            "candidate_sids": [candidate["sid"] for candidate in candidates],
        }


def load_drafter(config_path: str | Path, checkpoint_path: str | Path,
                 index_dir: str | Path, device: str = "cpu") -> MiniOneRecDrafter:
    from qwen_runtime import load_experiment, load_trained_checkpoint

    _, dataset, _, model = load_experiment(config_path, device=device)
    checkpoint = load_trained_checkpoint(model, checkpoint_path)
    _assert_training_checkpoint(checkpoint)
    return MiniOneRecDrafter(model, dataset, index_dir, checkpoint_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="export a trained normalized catalog")
    export.add_argument("--config", default="props/qwen_industrial.yaml")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--batch-size", type=int, default=512)
    export.add_argument("--device", default="cpu")
    draft = subparsers.add_parser("draft", help="retrieve candidate item IDs and final SIDs")
    draft.add_argument("--config", default="props/qwen_industrial.yaml")
    draft.add_argument("--checkpoint", required=True)
    draft.add_argument("--index", required=True)
    history = draft.add_mutually_exclusive_group(required=True)
    history.add_argument("--history", help='JSON original item ID list, e.g. ["0", "12"]')
    history.add_argument("--history-file", help="path to a JSON original item ID list")
    draft.add_argument("--topk", type=int, default=100)
    draft.add_argument("--exclude-seen", action="store_true", help="explicitly suppress history items")
    draft.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "export":
        output = export_index(args.config, args.checkpoint, args.output,
                              batch_size=args.batch_size, device=args.device)
        print(json.dumps({"index_dir": str(output)}, ensure_ascii=False))
    else:
        raw = args.history if args.history is not None else Path(args.history_file).read_text(encoding="utf-8")
        history_ids = json.loads(raw)
        if not isinstance(history_ids, list):
            parser.error("history must be a JSON list of original item IDs")
        drafter = load_drafter(args.config, args.checkpoint, args.index, device=args.device)
        print(json.dumps(drafter.verification_payload(
            history_ids, topk=args.topk, exclude_seen=args.exclude_seen,
        ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
