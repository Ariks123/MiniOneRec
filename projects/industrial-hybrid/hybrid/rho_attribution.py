"""Split UniSRec K-minus-2K NDCG into disjoint contributions on frozen test requests."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def rank_vectors(report, arm, cases, test_size):
    vectors = {case: [-1] * test_size for case in cases}
    targets = [None] * test_size
    path = report / "predictions" / arm / "validation_winners.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            case = row["config_id"]
            if case not in vectors:
                continue
            i, target = row["row_id"], row["target"]
            if not 0 <= i < test_size or vectors[case][i] != -1:
                raise ValueError(f"Duplicate or invalid row for {case}: {i}")
            if targets[i] is not None and targets[i] != target:
                raise ValueError(f"Target differs for request {i}")
            targets[i] = target
            items = [entry["item_id"] for entry in row["ranking"]]
            if len(items) != cases[case] or len(set(items)) != len(items):
                raise ValueError(f"Invalid recommendation list for {case}: {i}")
            vectors[case][i] = items.index(target) + 1 if target in items else 0
    if any(-1 in r for r in vectors.values()) or None in targets:
        raise ValueError(f"Incomplete archived predictions for {arm}")
    return targets, vectors


def discount(rank):
    return 1 / math.log2(rank + 1) if rank else 0.0


def decompose(ranks1, ranks2, indices):
    gains, losses, shifts = [], [], []
    up = same = down = 0
    for i in indices:
        one, two = ranks1[i], ranks2[i]
        if one and two:
            up += one < two
            down += one > two
            same += one == two
            shifts.append(discount(one) - discount(two))
        elif one:
            gains.append(discount(one))
        elif two:
            losses.append(-discount(two))
    scale = 100 / len(indices)
    parts = [scale * math.fsum(a) for a in (gains, losses, shifts)]
    return dict(rows=len(indices), gained_hits=len(gains), lost_hits=len(losses), shared_hits=len(shifts),
                shared_up=up, shared_same=same, shared_down=down, gained_ndcg_pp=parts[0],
                lost_ndcg_pp=parts[1], shared_ndcg_pp=parts[2], total_ndcg_pp=math.fsum(parts))


def build(root):
    report = root / "report"
    selection = read_json(report / "selection.json")
    configs = selection["locked"] + selection["test_additional"]
    groups = {}
    for kstr, winners in selection["arm_winners"].items():
        k = int(kstr)
        fixed = {}
        for arm in ("unisrec1", "unisrec2"):
            options = [c["id"] for c in configs if c["arm"] == arm and c["k"] == k
                       and c["quota"] == k and c["prefix_beams"] == 1 and c["threshold"] is None]
            if len(options) != 1:
                raise ValueError(f"Missing unique all-accept control for {arm}/{k}")
            fixed[arm] = options[0]
        groups[kstr] = {"fixed_all_accept": fixed,
                        "tuned_champions": {arm:winners[arm] for arm in fixed}}

    test_size = read_json(report / "quality.json")["test_rows"]
    masks = read_json(root / "catalog.json")["cold"]
    vectors = {}
    target_lists = []
    for arm in ("unisrec1", "unisrec2"):
        cases = {case[arm]:int(k) for k, entries in groups.items() for case in entries.values()}
        targets, arm_vectors = rank_vectors(report, arm, cases, test_size)
        vectors.update(arm_vectors)
        target_lists.append(targets)
    if target_lists[0] != target_lists[1]:
        raise ValueError("Test requests/targets do not match across the two arms")
    ids = list(range(test_size))
    strata = {"overall":ids, "warm":[i for i in ids if not masks[targets[i]]],
              "cold":[i for i in ids if masks[targets[i]]]}
    with (report / "test.csv").open() as fp:
        frozen = {(r["id"], r["stratum"]):r for r in csv.DictReader(fp)}
    result = {}
    for k, entries in groups.items():
        result[k] = {}
        for name, case in entries.items():
            one, two = case["unisrec1"], case["unisrec2"]
            rows = {}
            for tag, indices in strata.items():
                part = decompose(vectors[one], vectors[two], indices)
                newer, old = frozen[one,tag], frozen[two,tag]
                delta = 100 * (float(newer["item_ndcg"]) - float(old["item_ndcg"]))
                if (not math.isclose(part["total_ndcg_pp"], delta, abs_tol=1e-10) or
                        part["gained_hits"] + part["shared_hits"] != int(newer["item_hits"]) or
                        part["lost_hits"] + part["shared_hits"] != int(old["item_hits"]) or
                        len(indices) != int(newer["rows"])):
                    raise ValueError(f"Contribution/independent metric mismatch: K={k} {name} {tag}")
                rows[tag] = part
            result[k][name] = {"rho1_id":one, "rho2_id":two, "groups":rows}
    output = {"status":"passed", "direction":"UniSRec K minus UniSRec 2K",
              "unit":"percentage points of exact item NDCG per stratum",
              "definition":"One relevant item per request; gain/loss membership is measured at K. "
                           "Shared rank-up means the target has a smaller rank under UniSRec K.",
              "by_k":result}
    dest = report / "rank-attribution.json"
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("/run/hybrid-rho1"))
    build(ap.parse_args().run_dir)
