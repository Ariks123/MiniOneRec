#!/usr/bin/env python3
"""Evaluate SID collision repair while preserving the trained first two codes."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rqvae-dir", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-matching", action="store_true",
                        help="Also compare minimum-distance assignments within the trained two-token prefix")
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.legacy_root.resolve() / "rq"))
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    import train_video_games_rqvae as legacy
    from export_video_games_sid import code_statistics, repair_semantic, REPAIR_EPSILONS, MAX_ROUNDS

    run = arguments.rqvae_dir
    summary = json.loads((run / "training_summary.json").read_text())
    checkpoint = run / summary["selected_checkpoint"]
    raw_path = run / "catalog_codes.npy"
    if (legacy.digest(checkpoint) != summary["model_sha256"]
            or legacy.digest(raw_path) != summary["catalog_codes_sha256"]
            or legacy.digest(arguments.embeddings) != summary["sources"]["embeddings_sha256"]):
        raise ValueError("Quantizer input identities or selected model changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = legacy.restore_model(payload, arguments.device).eval()
    x = np.load(arguments.embeddings, allow_pickle=False).astype(np.float32)
    if summary["config"].get("center_embeddings"):
        center_path = run / "embedding_center.npy"
        if legacy.digest(center_path) != summary["sources"]["embedding_center_sha256"]:
            raise ValueError("Learned RQ-VAE embedding center changed")
        x -= np.load(center_path, allow_pickle=False)
    raw = np.load(raw_path, allow_pickle=False)
    data = torch.from_numpy(x).to(arguments.device)
    groups = {
        "fit": payload["split"]["train_ids"],
        "warm_holdout": payload["split"]["val_ids"],
        "cold_catalog": payload["split"]["cold_ids"],
        "all": list(range(len(x))),
    }

    @torch.no_grad()
    def describe(codes):
        embedding = [layer.embedding.weight for layer in model.rq.vq_layers]
        all_mse = []
        for offset in range(0, len(data), 1024):
            indexes = torch.from_numpy(codes[offset:offset + 1024]).to(data.device)
            dequantized = sum(book[indexes[:, level]] for level, book in enumerate(embedding))
            decoded = model.decoder(dequantized)
            all_mse.extend((decoded - data[offset:offset + len(indexes)]).square().mean(-1).cpu().tolist())
        mse = np.array(all_mse)
        return {
            name: code_statistics(codes[items]) | {
                "mse": float(mse[items].mean()),
                "changed_items": int(np.any(codes[items] != raw[items], axis=1).sum()),
                "occupied_codes": [int(len(np.unique(codes[items, j]))) for j in range(3)],
            }
            for name, items in groups.items()
        }

    @torch.no_grad()
    def match_prefixes():
        """Reassign only colliding last tokens to distinct available codewords."""
        books = [layer.embedding.weight for layer in model.rq.vq_layers]
        latent = torch.cat([model.encoder(batch) for batch in data.split(1024)])
        residual = latent - books[0][raw[:, 0]] - books[1][raw[:, 1]]
        d = torch.cdist(residual, books[2]).square().cpu().numpy()
        result = raw.copy()
        prefixes, inverse = np.unique(raw[:, :2], axis=0, return_inverse=True)
        max_prefix_bucket = max(int((inverse == i).sum()) for i in range(len(prefixes)))
        for i in range(len(prefixes)):
            items = np.flatnonzero(inverse == i)
            untouched = np.bincount(raw[items, 2], minlength=256)[raw[items, 2]] == 1
            moved = items[~untouched]
            if not len(moved):
                continue
            remaining = np.setdiff1d(np.arange(256), raw[items[untouched], 2])
            item_indexes, code_indexes = linear_sum_assignment(d[np.ix_(moved, remaining)])
            result[moved[item_indexes], 2] = remaining[code_indexes]
        return result, max_prefix_bucket

    raw_audit = describe(raw)
    fixed, rounds, trajectory = repair_semantic(model, x, raw, arguments.device, [256] * 3)
    methods = {"raw": raw_audit}

    def save_variant(name, codes):
        if not np.array_equal(raw[:, :2], codes[:, :2]):
            raise ValueError("Collision repair changed either fixed semantic prefix token")
        output = run / f"catalog_codes_{name}.npy"
        np.save(output, codes, allow_pickle=False)
        methods[name] = {"codes_path": output.name, "codes_sha256": legacy.digest(output),
                         "metrics": describe(codes)}

    save_variant("sinkhorn", fixed)
    methods["sinkhorn"].update({"rounds": rounds, "trajectory": trajectory, "epsilons": REPAIR_EPSILONS,
                                "max_rounds": MAX_ROUNDS})
    if arguments.prefix_matching:
        matched, max_bucket = match_prefixes()
        save_variant("prefix_matching", matched)
        methods["prefix_matching"]["max_prefix_bucket"] = max_bucket
    report = {"status": "completed", "model_sha256": summary["model_sha256"], "metrics": methods,
              "selection_data": "Token assignment and reconstruction on fixed product vectors only."}
    legacy.write_json(run / "collision_repair.json", report)
    print(json.dumps({
        name: {k: details[k] for k in ("warm_holdout", "all")}
        for name, payload in methods.items() for details in [payload.get("metrics", payload)]
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
