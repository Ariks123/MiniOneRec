"""Generate and compare the frozen UniSRec 2K scheme on the new SID catalog."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from hybrid.experiment import Engine, KS, dump
from hybrid.paired import summarize
from hybrid.rqvae_assets import MANIFEST, RQVAEData, load_json
from hybrid.sweep import (
    Catalog, STRATA, TEST_ROWS, VALID_ROWS, evaluate, make_config, merge_selected,
    read_reference, score_of, sha256, stream_rows, files_for, write_csv, write_json,
)


def generate_split(root, split, data, engine, rank):
    rows = data.read(split)[rank::4]
    target = root / split / f"rank{rank}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    record = target.with_suffix(".manifest.json")
    identity = {
        "assets": sha256(data.assets / MANIFEST), "rank": rank, "split": split, "rows": len(rows),
        "source": {name: sha256(Path(__file__).with_name(name)) for name in (
            "rqvae_quality.py", "experiment.py", "drafters.py", "verifier.py",
        )},
    }
    if record.exists() and target.exists():
        previous = load_json(record)
        if previous["identity"] != identity or previous["sha256"] != sha256(target):
            raise ValueError(f"Cached shard has a different RQ-VAE identity at {target}")
        return
    reference = read_reference(data.assets)[0] if split == "test" else None
    if reference is not None:
        for row in rows:
            if data.sids[row["target"]] != reference[row["row_id"]][1]:
                raise ValueError(f"RL evaluation used a different target at test:{row['row_id']}")
    pending = target.with_suffix(".partial")
    start = time.monotonic()
    with pending.open("w", encoding="utf-8") as f:
        for offset in range(0, len(rows), 4):
            group = rows[offset:offset + 4]
            records = engine.primitives(group, smoke=offset == 0 and rank == 0, widths=(1,))
            for rec in records:
                if reference is not None and rec["beam"] != reference[rec["row_id"]][0]:
                    raise ValueError(f"New RL beam evaluation did not reproduce on test:{rec['row_id']}")
                f.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")
            if offset % 40 == 0 or offset + 4 >= len(rows):
                f.flush()
                dump(root / f"progress-{rank}.json", dict(
                    split=split, done=min(offset + 4, len(rows)), total=len(rows),
                    seconds=time.monotonic() - start,
                ))
    pending.replace(target)
    write_json(record, {"identity": identity, "sha256": sha256(target)})


def generate(args, data):
    import torch

    rank = int(os.environ.get("RANK", "0"))
    if os.environ.get("WORLD_SIZE") != "4" or int(os.environ.get("LOCAL_RANK", "-1")) != rank:
        raise ValueError("RQ-VAE quality generation requires four local GPU ranks and original B4 groups")
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    if rank == 0:
        dump(args.run_dir / "catalog.json", data.record)
    engine = Engine(data, f"cuda:{rank}", names=("unisrec",))
    for split in args.splits:
        generate_split(args.run_dir, split, data, engine, rank)


def verify_shards(root, split, data):
    result = {}
    asset_sha = sha256(data.assets / MANIFEST)
    for rank in range(4):
        name = f"rank{rank}.jsonl"
        path = root / split / name
        rec = load_json(path.with_suffix(".manifest.json"))
        digest = sha256(path)
        if (rec["identity"]["assets"] != asset_sha
                or rec["identity"]["split"] != split or rec["identity"]["rank"] != rank
                or rec["sha256"] != digest):
            raise ValueError(f"Stale or incomplete RQ-VAE quality shard {path}")
        result[name] = digest
    return result


def old_configurations(root):
    previous = load_json(root / "report/selection.json")
    by_id = {cfg["id"]: cfg for cfg in previous["locked"]}
    winners = {}
    beams = {}
    for k in KS:
        chosen = by_id[previous["winners"][str(k)]]
        wanted = -4.032527049382527 if k == 1 else None
        if (chosen["drafter"], chosen["k"], chosen["quota"], chosen["alpha"], chosen["prefix_beams"],
                chosen["threshold"], chosen.get("rho", 2), chosen.get("max_rounds", 3)) != (
                "unisrec", k, k, 1.0, 1, wanted, 2, 3):
            raise ValueError(f"Archive is not the final fixed 2K plan at K={k}: {chosen['id']}")
        winners[k] = dict(chosen, rho=2, max_rounds=3)
        pure = [cfg for cfg in previous["locked"] if cfg["drafter"] == "beam" and cfg["k"] == k]
        if len(pure) != 1 or pure[0]["quota"] != 0:
            raise ValueError(f"Invalid original pure beam configuration at K={k}")
        beams[k] = dict(pure[0], rho=2, max_rounds=3)
    return previous, winners, beams


def legacy_guard(legacy, manifest):
    if sha256(legacy / "catalog.json") != manifest["legacy_catalog_sha256"]:
        raise ValueError("Catalog of the archived comparison changed")
    reference = legacy / "reference_test_beam.jsonl"
    if sha256(reference) != manifest["legacy_reference_sha256"]:
        raise ValueError("Prepared test beam does not match the original benchmark")
    return Catalog.from_dict(load_json(legacy / "catalog.json"))


def calibrate_k1(old, new, previous, threshold):
    # Fix the empirical first-round rejection rate of the original K=1 route.
    # The two models use the same validation requests and sample size.
    fixed = random.Random(42).sample(range(VALID_ROWS), 512)
    if set(fixed) != set(previous["calibration"]["rows"]):
        raise ValueError("K=1 calibration requests do not match the original validation sample")
    wanted = set(fixed)

    def collect(root):
        values = {}
        for row in stream_rows(files_for(root, "valid")):
            idx = row["row_id"]
            if idx in wanted:
                picks = row["drafts"]["unisrec"]["1"]["first"]
                if len(picks) != 2 or idx in values:
                    raise ValueError("K=1 calibration requires two first-round scores per request")
                values[idx] = (row["target"], [score_of(row["scores"], p) for p in picks])
        if set(values) != wanted:
            raise ValueError("K=1 calibration sample is incomplete")
        return values

    old_rows = collect(old)
    new_rows = collect(new)
    if any(old_rows[i][0] != new_rows[i][0] for i in fixed):
        raise ValueError("Calibrating K=1 on different validation target items")
    base = np.asarray([old_rows[i][1] for i in fixed]).ravel()
    new_scores = np.asarray([new_rows[i][1] for i in fixed]).ravel()
    rate = float((base <= threshold).mean())
    chosen = float(np.quantile(new_scores, rate))
    return {
        "method": "match archived marginal first-round rejection rate on the same 512 validation histories",
        "sample_rows": fixed, "old_threshold": threshold, "old_rejection_rate": rate, "threshold": chosen,
        "observed_new_rejection_rate": float((new_scores <= chosen).mean()),
    }


def select(args, data):
    root = args.run_dir
    legacy = args.legacy_run
    legacy_guard(legacy, data.manifest)
    previous, winners, beams = old_configurations(legacy)
    valid_inputs = verify_shards(root, "valid", data)
    for name, digest in previous["valid_inputs"].items():
        if sha256(legacy / "valid" / name) != digest:
            raise ValueError(f"Original calibration archive changed: {name}")
    primary = [c for k in KS for c in (winners[k], beams[k])]
    calibration = calibrate_k1(legacy, root, previous, winners[1]["threshold"])
    calibrated = dict(make_config("unisrec", 1, 1, 1, calibration["threshold"], [1.0]),
                      rho=2, max_rounds=3)
    calibrated["id"] += "-rqvae-k1-recalibrated"
    locked = [*primary, calibrated]
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    rows = evaluate(root, "valid", Catalog.from_dict(data.record), locked)
    write_csv(report / "validation.csv", rows)
    write_json(report / "selection.json", {
        "scheme": "frozen_uniSRec_2K", "winners": {str(k): c["id"] for k, c in winners.items()},
        "beam": {str(k): c["id"] for k, c in beams.items()}, "locked": locked,
        "valid_inputs": valid_inputs, "asset_manifest_sha256": sha256(data.assets / MANIFEST),
        "k1_validation_calibration": {**calibration, "config_id": calibrated["id"]},
        "primary_rule": "exactly replay the historical 2K validation winners; calibration is supplementary",
    })
    write_json(report / "benchmark_configs.json", primary)


def rankings_from_archive(root, keys, targets):
    out = {key: np.zeros(TEST_ROWS, dtype=np.int16) for key in keys}
    seen = set()
    with gzip.open(root / "report/predictions/validation_winners.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key, row_id = row["config_id"], row["row_id"]
            if key not in out:
                continue
            if not 0 <= row_id < TEST_ROWS or (key, row_id) in seen or row["target"] != targets[row_id]:
                raise ValueError(f"Inconsistent archived recommendation at {key}:{row_id}")
            ranking = [item["item_id"] for item in row["ranking"]]
            if not 0 < len(ranking) <= keys[key] or len(set(ranking)) != len(ranking):
                raise ValueError(f"Mis-sized or duplicate ranking at {key}:{row_id}")
            seen.add((key, row_id))
            out[key][row_id] = item_rank(targets[row_id], ranking)
    if len(seen) != TEST_ROWS * len(keys):
        raise ValueError(f"Incomplete ranked recommendations in {root}")
    return out


def item_rank(target, recommendations):
    return recommendations.index(target) + 1 if target in recommendations else 0


def primitive_ranks(root, targets, catalog, reference):
    result = {method: {k: np.zeros(TEST_ROWS, dtype=np.int16) for k in KS}
              for method in ("unisrec_original_rank", "beam")}
    seen = set()
    for row in stream_rows(files_for(root, "test")):
        idx, target = row["row_id"], row["target"]
        if (not 0 <= idx < TEST_ROWS or idx in seen or target != targets[idx]
                or (row["beam"], catalog.sid_by_item[target]) != reference[idx]):
            raise ValueError(f"Test primitive is not aligned to the original beam and target at {root}:{idx}")
        seen.add(idx)
        draft_50 = row["drafts"]["unisrec"]["50"]["first"]
        if len(draft_50) != 100:
            raise ValueError("Expected 2K first-round candidates")
        beam = [hit.item_id for hit in merge_selected([], row["beam"], {}, catalog, 50, 0)]
        for k in KS:
            if row["drafts"]["unisrec"][str(k)]["first"] != draft_50[:2 * k]:
                raise ValueError(f"UniSRec first-round ranks are inconsistent across K at {root}:{idx}")
            result["unisrec_original_rank"][k][idx] = item_rank(target, draft_50[:k])
            result["beam"][k][idx] = item_rank(target, beam[:k])
    if seen != set(range(TEST_ROWS)):
        raise ValueError(f"Missing test rows in {root}")
    return result


def metric(ranks):
    hits = ranks[ranks > 0]
    return dict(rows=len(ranks), item_hits=len(hits), item_hr=float(len(hits) / len(ranks)),
                item_ndcg=float((1 / np.log2(hits.astype(np.float64) + 1)).sum() / len(ranks)))


def index_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {(row["id"], row["stratum"]): row for row in csv.DictReader(stream)}


def confirm_metric(check, actual, context):
    if any(not math.isclose(float(check[f]), actual[f], abs_tol=1e-12) for f in actual):
        raise ValueError(f"Independent paired metric differs from archived totals at {context}")


def paired_quality(legacy, root, data, selection, test_metrics):
    report = root / "report"
    archived = index_csv(legacy / "report/test.csv")
    actual = {(row["id"], row["stratum"]): row for row in test_metrics}
    targets = [item["target"] for item in data.read("test")]
    new_catalog = Catalog.from_dict(data.record)
    old_catalog = legacy_guard(legacy, data.manifest)
    cold = np.asarray(new_catalog.cold, dtype=bool)[targets]
    if (len(targets), int(cold.sum())) != (TEST_ROWS, 147):
        raise ValueError("Expected the same 4,533 test histories including 147 cold targets")
    masks = dict(zip(STRATA, (np.ones(TEST_ROWS, dtype=bool), ~cold, cold), strict=True))
    old_ranks = primitive_ranks(legacy, targets, old_catalog, read_reference(legacy)[0])
    new_ranks = primitive_ranks(root, targets, new_catalog, read_reference(data.assets)[0])
    k1_key = selection["k1_validation_calibration"]["config_id"]
    primary = {key: int(k) for k, key in selection["winners"].items()}
    old_final = rankings_from_archive(legacy, primary, targets)
    new_final = rankings_from_archive(root, {**primary, k1_key: 1}, targets)
    old_ranks["unisrec_2k"] = {k: old_final[selection["winners"][str(k)]] for k in KS}
    new_ranks["unisrec_2k"] = {k: new_final[selection["winners"][str(k)]] for k in KS}
    rows, pair_rows = [], []
    for k in KS:
        for method in ("beam", "unisrec_original_rank", "unisrec_2k"):
            for i, (group, mask) in enumerate(masks.items()):
                a, b = old_ranks[method][k][mask], new_ranks[method][k][mask]
                before, after = metric(a), metric(b)
                if method != "unisrec_original_rank":
                    key = selection["beam" if method == "beam" else "winners"][str(k)]
                    confirm_metric(archived[key, group], before, (legacy, key, group))
                    confirm_metric(actual[key, group], after, (root, key, group))
                paired = summarize(a, b, np.random.default_rng(np.random.SeedSequence([42, k, i])))
                rows.append(dict(k=k, method=method, stratum=group, old=before, new=after))
                pair_rows.append(dict(k=k, method=method, stratum=group, comparison="rqvae_minus_rqkmeans", **paired))

    for i, (group, mask) in enumerate(masks.items()):
        baseline = new_ranks["unisrec_2k"][1][mask]
        recalibrated = new_final[k1_key][mask]
        confirm_metric(actual[k1_key, group], metric(recalibrated), (root, k1_key, group))
        pair_rows.append(dict(k=1, method="unisrec_2k", stratum=group, comparison="new_calibration_minus_fixed",
                              **summarize(baseline, recalibrated,
                                          np.random.default_rng(np.random.SeedSequence([42, 1, i])))))
    write_json(report / "quality_comparison.json", {
        "schema": 1, "test_rows": TEST_ROWS, "cold_test_rows": int(cold.sum()), "metric": "exact_item_id",
        "primary": rows, "calibrated_k1": [dict(stratum=group, **metric(new_final[k1_key][mask]))
                                          for group, mask in masks.items()],
    })
    write_json(report / "paired_comparison.json", {
        "replicates": 10000, "confidence": 0.95, "unit": "percentage_points",
        "comparisons": pair_rows,
    })
    write_csv(report / "quality_comparison.csv", [
        dict(k=row["k"], method=row["method"], stratum=row["stratum"],
             **{f"{side}_{key}": v for side in ("old", "new") for key, v in row[side].items()})
        for row in rows
    ])


def quality(args, data):
    root = args.run_dir
    report = root / "report"
    selection = load_json(report / "selection.json")
    if (selection["asset_manifest_sha256"] != sha256(data.assets / MANIFEST)
            or selection["valid_inputs"] != verify_shards(root, "valid", data)):
        raise ValueError("The frozen selection belongs to different data or checkpoint")
    test_inputs = verify_shards(root, "test", data)
    configurations = selection["locked"]
    primary = set(selection["winners"].values())
    detail = primary | {selection["k1_validation_calibration"]["config_id"]}
    reference = read_reference(data.assets)[0]
    rows = evaluate(root, "test", Catalog.from_dict(data.record), configurations,
                    reference=reference, winner_ids=detail)
    write_csv(report / "test.csv", rows)
    paired_quality(args.legacy_run, root, data, selection, rows)
    write_json(report / "quality_status.json", {
        "status": "passed", "test_inputs": test_inputs, "paired_test_rows": TEST_ROWS,
        "configs": len(configurations), "frozen_primary_configs": len(primary),
        "asset_manifest_sha256": sha256(data.assets / MANIFEST),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "select", "quality"))
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--legacy-run", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=("valid", "test"))
    args = parser.parse_args()
    if args.run_dir.resolve() == args.legacy_run.resolve():
        raise ValueError("Choose an isolated RQ-VAE run directory")
    {"generate": generate, "select": select, "quality": quality}[args.stage](args, RQVAEData(args.assets))


if __name__ == "__main__":
    main()
