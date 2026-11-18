"""Compare a validation-selected SASRec with frozen hybrid recommendations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hybrid.paired import summarize
from hybrid.rho_bundle import digest
from hybrid.rqvae_quality import confirm_metric, index_csv, metric, rankings_from_archive
from hybrid.sweep import KS, write_json
from .data import Dataset


STRATA = ("overall", "warm", "cold")
SEED = 42
BOOTSTRAP = 10000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_ranks(path, targets, warm):
    ranks = np.zeros(len(targets), dtype=np.int16)
    rows = set()
    with path.open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            index = row["row_id"]
            target = row["target"]
            if (type(index) is not int or not 0 <= index < len(targets) or index in rows
                    or target != targets[index] or row["is_cold"] != (target not in warm)):
                raise ValueError(f"Inconsistent SASRec prediction at {path}:{index}")
            items = row["ranking"]
            if len(items) != 50 or len(set(items)) != 50 or not set(items) <= warm:
                raise ValueError(f"Malformed SASRec ranking at {path}:{index}")
            rows.add(index)
            if target in items:
                ranks[index] = items.index(target) + 1
    if len(rows) != len(targets):
        raise ValueError(f"SASRec predictions omitted rows in {path}")
    return ranks


def at_k(ranks, k):
    return np.where(ranks <= k, ranks, 0)


def relative(hybrid, baseline):
    return (hybrid / baseline - 1) * 100 if baseline else None


def write_csv(path, rows):
    flattened = []
    for row in rows:
        result = {key: row[key] for key in ("k", "stratum", "relative_hr_gain_pct", "relative_ndcg_gain_pct")}
        for side in ("sasrec", "hybrid"):
            result.update({f"{side}_{key}": value for key, value in row[side].items()})
        paired = row["paired"]
        for key in ("hr", "ndcg"):
            lo, hi = paired[f"{key}_ci95_pp"]
            result.update({f"{key}_delta_pp": paired[f"{key}_delta_pp"],
                           f"{key}_ci95_low_pp": lo, f"{key}_ci95_high_pp": hi})
        result.update({key: paired[key] for key in ("gained_hits", "lost_hits", "retained_hits", "both_missed")})
        flattened.append(result)
    pending = path.with_suffix(".csv.next")
    with pending.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    pending.replace(path)


def build(run, experiment):
    published = experiment / "inference"
    source = experiment / "data"
    if (published / "stage.status").read_text().strip() != "completed":
        raise ValueError("Joint speculative inference must have finished")
    selection_path = published / "report/selection.json"
    selected = read_json(selection_path)
    ids = selected["winners"]
    if set(ids) != {str(k) for k in KS}:
        raise ValueError("Expected five frozen configurations for the joint system")
    baseline = read_json(run / "selection.json")
    valid_path, test_path = (run / f"{split}_predictions.jsonl" for split in ("valid", "test"))
    for path in (valid_path, test_path):
        if digest(path) != baseline["artifact_sha256"][path.name]:
            raise ValueError(f"SASRec prediction hash changed: {path.name}")

    dataset = Dataset(source)
    if (baseline["status"] != "completed" or baseline["split_sha256"] != dataset.split_sha256
            or baseline["preparation_sha256"] != dataset.preparation_sha256):
        raise ValueError("SASRec selection does not identify the frozen RQ-VAE interactions")
    valid_targets = [row.target for row in dataset.read("valid")]
    test = dataset.read("test")
    targets = [row.target for row in test]
    warm = set(dataset.candidates)
    catalog = read_json(published / "catalog.json")
    cold = np.asarray(catalog["cold"], dtype=bool)
    if len(warm) != 3647 or len(cold) != 3686 or set(np.flatnonzero(~cold)) != warm:
        raise ValueError("Training eligibility differs from the reported cold/warm catalog")
    if baseline["sorted_warm_original_ids"] != sorted(warm):
        raise ValueError("SASRec used a different candidate catalog")
    test_cold = cold[targets]
    if int(test_cold.sum()) != 147:
        raise ValueError("Expected 147 cold test targets")
    sas_rec_valid = metric(at_k(load_ranks(valid_path, valid_targets, warm), 10))
    for field in ("hr", "ndcg"):
        if not np.isclose(sas_rec_valid[f"item_{field}"], baseline["selected"][f"validation_{field}10"],
                          rtol=0, atol=1e-12):
            raise ValueError(f"SASRec winning validation score differs from saved rankings: {field}")
    sasrec = load_ranks(test_path, targets, warm)
    archived = rankings_from_archive(published, {ids[str(k)]: k for k in KS}, targets)
    source_metrics = index_csv(published / "report/test.csv")

    rows = []
    masks = np.ones(len(targets), dtype=bool), ~test_cold, test_cold
    for k in KS:
        hybrid = archived[ids[str(k)]]
        for index, (stratum, mask) in enumerate(zip(STRATA, masks, strict=True)):
            a, b = at_k(sasrec[mask], k), hybrid[mask]
            base, joint = metric(a), metric(b)
            confirm_metric(source_metrics[ids[str(k)], stratum], joint, (k, stratum))
            rng = np.random.default_rng(np.random.SeedSequence([SEED, k, index]))
            rows.append({"k": k, "stratum": stratum, "sasrec": base, "hybrid": joint,
                         "paired": summarize(a, b, rng, repeats=BOOTSTRAP),
                         "relative_hr_gain_pct": relative(joint["item_hr"], base["item_hr"]),
                         "relative_ndcg_gain_pct": relative(joint["item_ndcg"], base["item_ndcg"])})
    result = {
        "status": "passed", "metric": "exact_item_id",
        "test_rows": len(targets), "validation_rows": len(valid_targets),
        "warm_rows": int((~test_cold).sum()), "cold_rows": int(test_cold.sum()),
        "repeat_purchase_test_rows": sum(row.target in row.history for row in test),
        "hybrid_configs": ids, "comparisons": rows,
        "bootstrap": {"replicates": BOOTSTRAP, "confidence": .95, "unit": "paired_test_request", "seed": SEED},
        "source_sha256": {
            "sasrec_selection": digest(run / "selection.json"),
            "sasrec_test_predictions": digest(test_path),
            "hybrid_selection": digest(selection_path),
            "hybrid_test_predictions": digest(published / "report/predictions/validation_winners.jsonl.gz"),
            "test_csv": dataset.split_sha256["test"],
        },
    }
    write_csv(run / "comparison.csv", rows)
    write_json(run / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sasrec-run", type=Path, required=True)
    parser.add_argument("--rqvae-run", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.sasrec_run, args.rqvae_run)
    print(json.dumps({k: result[k] for k in ("status", "test_rows", "cold_rows", "repeat_purchase_test_rows")}))


if __name__ == "__main__":
    main()
