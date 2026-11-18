#!/usr/bin/env python3
"""Train three-level RQ-VAE for the frozen Amazon 2018 Industrial catalog."""

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time


DATASET = "Industrial_and_Scientific"
CATALOG_SIZE = 3686
BASELINE_ROWS = {"train": 36259, "valid": 4532, "test": 4533}
EMBEDDING_SHA = "91a1c875e4b009f3a07b5febf00c2ef62a3813a55cfe873b825e18e32df1dd0b"
LEGACY_SID_SHA = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_assets(arguments, legacy, np):
    source = arguments.data
    manifest = read(source / "preparation_manifest.json")
    if manifest["rows"] != BASELINE_ROWS or manifest["expected_sid_sha256"] != LEGACY_SID_SHA:
        raise ValueError("Frozen baseline data identity changed")
    meta = read(arguments.metadata)
    embedding_sha = legacy.digest(arguments.embeddings)
    if embedding_sha != EMBEDDING_SHA or set(meta) != {str(index) for index in range(CATALOG_SIZE)}:
        raise ValueError("Raw embedding or product catalog identity changed")
    original_meta = manifest["original_sources"]["item_metadata"]["sha256"]
    if original_meta != legacy.digest(arguments.metadata):
        raise ValueError("Product metadata changed")
    raw = np.load(arguments.embeddings, allow_pickle=False)
    if raw.shape != (CATALOG_SIZE, 2560) or raw.dtype != np.float16 or not np.isfinite(raw).all():
        raise ValueError("Expected original 3686 × 2560 finite FP16 raw embeddings")
    training = source / "train.csv"
    if legacy.digest(training) != manifest["outputs"]["train.csv"]["sha256"]:
        raise ValueError("Frozen train split SHA-256 mismatch")
    warm = set()
    with training.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            warm.update(map(int, ast.literal_eval(row["history_item_id"])))
            warm.add(int(row["item_id"]))
    if len(warm) != 3647 or not warm <= set(range(CATALOG_SIZE)):
        raise ValueError("Unexpected warm product catalog")
    return raw.astype(np.float32), meta, warm, {
        "embeddings_sha256": embedding_sha, "metadata_sha256": original_meta,
        "baseline_sid_sha256": LEGACY_SID_SHA,
        "training_split_sha256": manifest["outputs"]["train.csv"]["sha256"],
        "embedding_row_order": "Contiguous released product ID order, same assumption as original UniSRec.",
    }


def split_items(meta, warm, arguments, legacy, np, group_split):
    texts = legacy.cleaned_texts(meta)
    sorted_warm = np.asarray(sorted(warm))
    grouped = np.asarray([texts[item] for item in sorted_warm], dtype=object)
    train_rows, val_rows = next(group_split(
        n_splits=1, test_size=.1, random_state=arguments.seed
    ).split(sorted_warm, groups=grouped))
    train_ids, val_ids = sorted_warm[train_rows], sorted_warm[val_rows]
    if set(grouped[train_rows]) & set(grouped[val_rows]) or len(train_ids) < 256 or len(val_ids) == 0:
        raise ValueError("Grouped warm product split is invalid")
    return {
        "train_ids": train_ids.tolist(), "val_ids": val_ids.tolist(),
        "cold_ids": sorted(set(range(CATALOG_SIZE)) - warm),
        "group_key": "exact cleaned title+description text", "seed": arguments.seed,
    }


def model_config(arguments):
    return argparse.Namespace(
        lr=1e-3, epochs=arguments.epochs, min_epochs=arguments.min_epochs,
        batch_size=20480, eval_step=arguments.eval_step, patience=30, relative_improvement=.001,
        warmup_epochs=50, dropout_prob=0., bn=False, loss_type="mse", kmeans_init=True,
        kmeans_iters=100, kmeans_n_init=arguments.kmeans_n_init, sk_epsilons=[0., 0., 0.], sk_iters=50,
        num_emb_list=[256, 256, 256], e_dim=32, quant_loss_weight=1., beta=.25,
        layers=[2048, 1024, 512, 256, 128, 64], seed=arguments.seed,
        optimizer="AdamW", weight_decay=0., grad_clip=1., dtype="float32", center_embeddings=arguments.center_embeddings,
    )


def train(arguments):
    for key in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "4")
    sys.path.insert(0, str(arguments.legacy_root.resolve() / "rq"))
    import numpy as np
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    import train_video_games_rqvae as legacy

    run = arguments.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise ValueError(f"RQ-VAE run directory must be empty: {run}")
    cfg = model_config(arguments)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    raw, metadata, warm, sources = load_assets(arguments, legacy, np)
    split = split_items(metadata, warm, arguments, legacy, np, GroupShuffleSplit)
    data = torch.from_numpy(raw).to(arguments.device)
    train_index = torch.as_tensor(split["train_ids"], device=data.device)
    mean = data[train_index].mean(0)
    if cfg.center_embeddings:
        np.save(run / "embedding_center.npy", mean.cpu().numpy(), allow_pickle=False)
        sources["embedding_center_sha256"] = legacy.digest(run / "embedding_center.npy")
        data = data - mean
        mean = data[train_index].mean(0)
    groups = {"train": split["train_ids"], "val": split["val_ids"], "all": list(range(len(data)))}
    baseline = {name: float((data[ids] - mean).square().mean()) for name, ids in groups.items()}
    manifest = {
        "dataset": DATASET, "status": "training", "config": vars(cfg), "sources": sources,
        "split_counts": {name: len(value) for name, value in groups.items()} | {"cold": len(split["cold_ids"])},
        "selection": "Strict minimum warm holdout reconstruction MSE; ties keep the earlier epoch.",
        "gpu": torch.cuda.get_device_name(data.device),
    }
    legacy.write_json(run / "split.json", split)
    legacy.write_json(run / "manifest.json", manifest)
    model = legacy.build_model(cfg, data.shape[1]).to(data.device)
    manifest["initialization"] = legacy.initialize(model, data, train_index, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = cfg.warmup_epochs * math.ceil(len(train_index) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / warmup, 1.))
    history = [{"epoch": 0, "elapsed_seconds": time.perf_counter() - start,
                **legacy.evaluate(model, data, split, baseline)}]
    best_mse = significant_best = float("inf")
    best_epoch = stale = 0
    stop_reason = "maximum_epochs"
    selected = run / "best_val_model.pth"
    names = ("epoch", "train_total", "train_recon", "train_quant", "lr", "epoch_seconds", "elapsed_seconds")
    torch.cuda.reset_peak_memory_stats(data.device)
    with (run / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            tick = time.perf_counter()
            model.train()
            batches = train_index[torch.randperm(len(train_index), device=data.device)].split(cfg.batch_size)
            total = torch.zeros(3, dtype=data.dtype, device=data.device)
            for batch_ids in batches:
                batch = data[batch_ids]
                optimizer.zero_grad(set_to_none=True)
                decoded, quant, _ = model(batch, use_sk=False)
                loss, recon = model.compute_loss(decoded, quant, xs=batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                total += torch.stack((loss.detach(), recon.detach(), quant.detach())) * len(batch)
            losses = (total / len(train_index)).tolist()
            if not all(math.isfinite(value) for value in losses):
                raise RuntimeError(f"Nonfinite RQ-VAE training loss at epoch {epoch}")
            observation = dict(zip(names, (epoch, *losses, optimizer.param_groups[0]["lr"],
                                           time.perf_counter() - tick, time.perf_counter() - start)))
            writer.writerow(observation)
            if epoch % cfg.eval_step and epoch != cfg.epochs:
                continue
            metrics = legacy.evaluate(model, data, split, baseline)
            history.append(observation | metrics)
            mse = metrics["val"]["mse"]
            if not math.isfinite(mse):
                raise RuntimeError(f"Nonfinite RQ-VAE holdout reconstruction MSE at epoch {epoch}")
            if mse < best_mse:
                best_mse, best_epoch = mse, epoch
                temporary = selected.with_suffix(".tmp")
                torch.save({"args": cfg, "input_dim": data.shape[1], "epoch": epoch,
                            "state_dict": model.state_dict(), "quantizer_initialized": [True] * 3,
                            "embedding_sha256": sources["embeddings_sha256"], "split": split}, temporary)
                temporary.replace(selected)
            if mse < significant_best * (1 - cfg.relative_improvement):
                significant_best, stale = mse, 0
            else:
                stale += 1
            stream.flush()
            legacy.write_json(run / "evaluation_history.json", history)
            progress = {
                "status": "training", "epoch": epoch, "max_epochs": cfg.epochs, "best_epoch": best_epoch,
                "best_val_mse": best_mse, "validation_mse": mse, "stale_evaluations": stale,
                "elapsed_seconds": time.perf_counter() - start,
                "all_collision_rate": metrics["all"]["collision_rate"],
                "train_unique_sids": metrics["train"]["unique_sids"],
                "val_unique_sids": metrics["val"]["unique_sids"],
            }
            legacy.write_json(run / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if epoch >= cfg.min_epochs and stale >= cfg.patience:
                stop_reason = "validation_plateau"
                break

    best = torch.load(selected, map_location="cpu", weights_only=False)
    model = legacy.restore_model(best, arguments.device).eval()
    chosen = legacy.evaluate(model, data, split, baseline)
    raw_codes = torch.cat([model.get_indices(batch, use_sk=False) for batch in data.split(1024)])
    codes = raw_codes.cpu().numpy()
    if codes.shape != (CATALOG_SIZE, 3) or codes.min() < 0 or codes.max() >= 256:
        raise ValueError("Selected RQ-VAE did not produce valid three-token codes")
    if chosen["val"]["mse"] != best_mse:
        raise ValueError("Restored checkpoint does not reproduce selected validation MSE")
    np.save(run / "catalog_codes.npy", codes, allow_pickle=False)
    summary = manifest | {
        "status": "completed", "selected_checkpoint": selected.name, "model_sha256": legacy.digest(selected),
        "stop_reason": stop_reason, "best_epoch": best_epoch, "best_val_mse": best_mse,
        "epochs_completed": epoch, "selected_metrics": chosen,
        "catalog_codes_sha256": legacy.digest(run / "catalog_codes.npy"),
        "training_seconds": time.perf_counter() - start,
        "peak_gpu_allocated_GiB": torch.cuda.max_memory_allocated(data.device) / 2**30,
    }
    legacy.write_json(run / "training_summary.json", summary)
    legacy.write_json(run / "progress.json", summary)
    legacy.write_json(run / "manifest.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "best_epoch", "best_val_mse", "training_seconds")}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--min-epochs", type=int, default=3000)
    parser.add_argument("--eval-step", type=int, default=50)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--center-embeddings", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.min_epochs, args.eval_step, args.kmeans_n_init) < 1:
        parser.error("Training counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
