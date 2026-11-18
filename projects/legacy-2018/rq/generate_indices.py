#!/usr/bin/env python3
"""Export the original RQ-VAE SIDs, with catalog checks and collision statistics."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import npu_support  # noqa: F401 - enables the existing Ascend device shim

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset

PREFIX = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
MAX_COLLISION_ROUNDS = 20


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str)


def get_collision_item(all_indices_str):
    index2id = collections.defaultdict(list)
    for item, index in enumerate(all_indices_str):
        index2id[index].append(item)
    return [items for items in index2id.values() if len(items) > 1]


def collision_statistics(all_indices_str):
    counts = get_indices_count(all_indices_str)
    total = len(all_indices_str)
    unique = len(counts)
    return {
        "items": total,
        "unique_sids": unique,
        "collision_groups": sum(count > 1 for count in counts.values()),
        "items_in_collision_groups": sum(count for count in counts.values() if count > 1),
        "excess_items": total - unique,
        # Preserve the original exporter's definition, not the fraction of items in groups.
        "collision_rate": (total - unique) / total,
        "max_collision_group_size": max(counts.values()),
    }


def validate_catalog(data_path, item_meta_path):
    embeddings = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or 0 in embeddings.shape:
        raise ValueError("Embeddings must be a nonempty (items, dimensions) array")
    if not np.issubdtype(embeddings.dtype, np.number) or not np.isrealobj(embeddings):
        raise ValueError("Embeddings must contain real numeric values")
    for start in range(0, len(embeddings), 4096):
        block = embeddings[start:start + 4096]
        if not np.isfinite(block).all():
            raise ValueError("Embeddings contain NaN or Inf; refusing automatic replacement")
        with np.errstate(over="ignore", invalid="ignore"):
            finite_float32 = np.isfinite(block.astype(np.float32)).all()
        if not finite_float32:
            raise ValueError("Embeddings cannot be represented as finite float32 values")
    metadata = json.loads(item_meta_path.read_text(encoding="utf-8"))
    expected_ids = {str(item) for item in range(len(embeddings))}
    if not isinstance(metadata, dict) or set(metadata) != expected_ids:
        raise ValueError("Metadata keys must exactly match embedding row IDs 0..N-1")
    shape = [int(value) for value in embeddings.shape]
    del embeddings
    return shape


def file_source(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, help="Override the embedding path saved in the checkpoint")
    parser.add_argument("--item_meta_path", type=Path, required=True,
                        help="Dataset.item.json; row i must describe item ID i")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch_size must be positive and --num_workers must be nonnegative")
    return args


def main():
    cli = parse_args()
    ckpt_path = cli.ckpt_path.resolve(strict=True)
    item_meta_path = cli.item_meta_path.resolve(strict=True)
    output_file = cli.output_file.resolve()
    if output_file.suffix.lower() != ".json":
        raise ValueError("--output_file must have a .json suffix")
    stats_file = output_file.with_suffix(".stats.json")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an available --device explicitly")

    # Load only a trusted, locally trained RQ-VAE checkpoint (it contains argparse args).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    data_path = (cli.data_path or Path(args.data_path)).resolve(strict=True)
    protected = {ckpt_path, data_path, item_meta_path}
    if output_file in protected or stats_file in protected:
        raise ValueError("Output files must not overwrite checkpoint, embeddings, or metadata")
    shape = validate_catalog(data_path, item_meta_path)
    if not 1 <= len(args.num_emb_list) <= len(PREFIX):
        raise ValueError("The original SID format supports one to five quantization levels")
    if len(args.num_emb_list) != len(args.sk_epsilons):
        raise ValueError("Checkpoint codebook and Sinkhorn level counts do not match")
    for name, value in state_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {name} contains NaN or Inf")

    from models.rqvae import RQVAE

    data = EmbDataset(str(data_path))
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=getattr(args, "beta", 0.25),
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    data_loader = DataLoader(data, num_workers=cli.num_workers,
                             batch_size=cli.batch_size, shuffle=False,
                             pin_memory=device.type == "cuda")

    all_indices = []
    all_indices_str = []
    for batch in tqdm(data_loader, desc="Encoding item SIDs"):
        indices = model.get_indices(batch.to(device), use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [PREFIX[level].format(int(value)) for level, value in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))
    # Object arrays avoid truncating a newly assigned, longer token/string during collision repair.
    all_indices = np.array(all_indices, dtype=object)
    all_indices_str = np.array(all_indices_str, dtype=object)
    initial_stats = collision_statistics(all_indices_str)

    # Original algorithm: only the final quantizer uses Sinkhorn for colliding groups.
    for quantizer in model.rq.vq_layers[:-1]:
        quantizer.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003
    rounds = 0
    while rounds < MAX_COLLISION_ROUNDS and not check_collision(all_indices_str):
        collision_groups = get_collision_item(all_indices_str)
        print(f"Collision round {rounds + 1}: {len(collision_groups)} groups")
        for collision_items in collision_groups:
            batch = data[collision_items].to(device)
            indices = model.get_indices(batch, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [PREFIX[level].format(int(value)) for level, value in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        rounds += 1

    final_stats = collision_statistics(all_indices_str)
    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    if len(all_indices_dict) != shape[0]:
        raise ValueError("Exported SID count does not match the embedding catalog")
    stats = {
        "algorithm": "rqvae",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(output_file),
        "embedding_shape": shape,
        "item_id_mapping": "embedding row i -> metadata item ID str(i)",
        "metadata_alignment": "passed",
        "initial": initial_stats,
        "final": final_stats,
        "collision_rounds": rounds,
        "max_collision_rounds": MAX_COLLISION_ROUNDS,
        "collision_rate_definition": "(items - unique_sids) / items",
        "collision_policy": "original_last_level_sinkhorn; unresolved collisions retained; no suffix",
        "parameters": {
            "checkpoint_args": vars(args),
            "checkpoint_epoch": ckpt.get("epoch"),
            "device": str(device),
            "batch_size": cli.batch_size,
            "num_workers": cli.num_workers,
            "export_sk_epsilons": [quantizer.sk_epsilon for quantizer in model.rq.vq_layers],
        },
        "sources": {
            "checkpoint": file_source(ckpt_path),
            "embeddings": file_source(data_path),
            "item_metadata": file_source(item_meta_path),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(all_indices_dict, ensure_ascii=False) + "\n", encoding="utf-8")
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
                          encoding="utf-8")
    print(json.dumps({"output_file": str(output_file), "stats_file": str(stats_file),
                      **final_stats, "collision_rounds": rounds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
