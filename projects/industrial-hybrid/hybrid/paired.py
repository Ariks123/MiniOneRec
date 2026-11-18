"""Paired item-level test uncertainty for configurations selected on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .sweep import KS, TEST_ROWS, write_json


def rank(items, targets):
    matched = items == targets[:, None]
    return np.where(matched.any(axis=1), 1 + matched.argmax(axis=1), 0)


def summarize(base_ranks, candidate_ranks, rng, repeats=10000):
    """Sample requests with replacement, jointly for both models and metrics."""
    base_ranks = np.asarray(base_ranks)
    candidate_ranks = np.asarray(candidate_ranks)
    if base_ranks.shape != candidate_ranks.shape or base_ranks.ndim != 1 or len(base_ranks) == 0:
        raise ValueError("Paired summaries require two equally sized, nonempty rank arrays")
    base_hit = base_ranks > 0
    candidate_hit = candidate_ranks > 0
    def discounted(ranks):
        return np.where(ranks > 0, 1 / np.log2(1.0 + np.maximum(1, ranks).astype(np.float64)), 0.0)
    differences = np.column_stack([
        candidate_hit.astype(float) - base_hit.astype(float),
        discounted(candidate_ranks) - discounted(base_ranks),
    ])
    # Resampling counts of identical (delta HR, delta NDCG) pairs is exactly
    # row bootstrap. This avoids materializing a repeats-by-test-size index array.
    values, counts = np.unique(differences, axis=0, return_counts=True)
    n = len(differences)
    estimates = np.empty((repeats, 2), dtype=float)
    for i in range(0, repeats, 256):
        size = min(256, repeats-i)
        multiplicities = rng.multinomial(n, counts/n, size=size)
        estimates[i:i+size] = (multiplicities @ values) * (100/n)
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    means = differences.mean(axis=0) * 100
    return {
        "rows": n, "baseline_hits": int(base_hit.sum()), "candidate_hits": int(candidate_hit.sum()),
        "gained_hits": int((~base_hit & candidate_hit).sum()),
        "lost_hits": int((base_hit & ~candidate_hit).sum()),
        "retained_hits": int((base_hit & candidate_hit).sum()),
        "both_missed": int((~base_hit & ~candidate_hit).sum()),
        "hr_delta_pp": float(means[0]), "hr_ci95_pp": bounds[:, 0].tolist(),
        "ndcg_delta_pp": float(means[1]), "ndcg_ci95_pp": bounds[:, 1].tolist(),
        "all_observed_hr_deltas_zero": bool(np.all(differences[:,0] == 0)),
        "all_observed_ndcg_deltas_zero": bool(np.all(differences[:,1] == 0)),
    }


def build(root):
    root = Path(root)
    report = root/"report"
    selection = json.loads((report/"selection.json").read_text())
    catalog = json.loads((root/"catalog.json").read_text())
    cold = np.asarray(catalog["cold"], dtype=bool)
    winners = selection["winners"]
    baseline = {c["k"]: c["id"] for c in selection["locked"] if c["drafter"]=="beam"}
    rows_seen = np.zeros(TEST_ROWS, dtype=bool)
    is_cold = np.zeros(TEST_ROWS, dtype=bool)
    ranks = {k: np.zeros((2, TEST_ROWS), dtype=np.int16) for k in KS}
    for shard in sorted((report/"predictions").glob("batch-*.npz")):
        with np.load(shard, allow_pickle=False) as data:
            row_ids = data["row_ids"]
            targets = data["targets"]
            if (np.any(row_ids<0) or np.any(row_ids>=TEST_ROWS) or len(np.unique(row_ids))!=len(row_ids)
                    or rows_seen[row_ids].any()):
                raise ValueError(f"Invalid paired test row IDs in {shard}")
            rows_seen[row_ids] = True
            is_cold[row_ids] = cold[targets]
            for k in KS:
                for index, key in enumerate((baseline[k], winners[str(k)])):
                    items = data[key+"_item"]
                    if items.shape != (len(row_ids), k):
                        raise ValueError(f"Malformed rankings for {key}: {items.shape}")
                    ranks[k][index, row_ids] = rank(items, targets)
    if not np.all(rows_seen) or is_cold.sum()!=147:
        raise ValueError("Paired evaluation requires all 4,533 test rows including 147 cold targets")
    summaries = {}
    for k in KS:
        groups = {}
        for index, (stratum,mask) in enumerate((("overall", rows_seen),("warm", ~is_cold),("cold",is_cold))):
            rng = np.random.default_rng(np.random.SeedSequence([42,k,index]))
            groups[stratum] = summarize(ranks[k][0,mask],ranks[k][1,mask],rng)
        summaries[str(k)] = {"baseline":baseline[k],"winner":winners[str(k)],"groups":groups}
    result = {
        "status":"passed","metric":"exact_item_id","comparison":"validation_winner_minus_original_beam",
        "unit":"percentage_points","method":"paired percentile bootstrap of test rows within each stratum",
        "replicates":10000,"seed":42,"confidence":0.95,"multiplicity_adjustment":None,"comparisons":summaries,
    }
    write_json(report/"paired.json", result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path)
    build(parser.parse_args().run_dir)
