"""Strict product-ID direct-ranking baselines and target attribution for locked winners."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import NamedTuple

from .sweep import (
    Catalog, DRAFTERS, KS, STRATA, TEST_ROWS, files_for, merge_selected, score_of, stream_rows, write_json,
)


class Outcome(NamedTuple):
    target: int
    rank: int
    source: str | None
    round: int
    retrieved_count: int


def saved_outcomes(path, winners):
    outcomes = {key: {} for key in winners.values()}
    for rec in stream_rows([path]):
        known = outcomes[rec["config_id"]]
        row_id, target, items = rec["row_id"], rec["target"], rec["ranking"]
        if row_id in known or len({entry["item_id"] for entry in items}) != len(items):
            raise ValueError(f"Duplicate prediction or recommendation on row {row_id}")
        rank, source, rnd = 0, None, 0
        for position, item in enumerate(items, 1):
            if item["item_id"] == target:
                rank, source, rnd = position, item["source"], item["round"]
        known[row_id] = Outcome(target, rank, source, rnd, sum(i["source"] == "retrieval" for i in items))
    if any(set(rows) != set(range(TEST_ROWS)) for rows in outcomes.values()):
        raise ValueError("Every validation winner needs an observed ranking for every test request")
    return outcomes


def rank_of(target, ordered):
    try:
        return 1 + ordered.index(target)
    except ValueError:
        return 0


def record_rank(counter, rank):
    counter["rows"] += 1
    if rank:
        counter["hits"] += 1
        counter["discount"] += 1.0 / math.log2(1 + rank)


def metrics(counter):
    n = counter["rows"]
    return dict(rows=n, hits=counter["hits"], item_hr=counter["hits"]/n, item_ndcg=counter["discount"]/n)


def trace_target(row, cfg, observed, catalog):
    target, k, quota = row["target"], cfg["k"], cfg["quota"]
    theta = cfg["threshold"] if cfg["threshold"] is not None else -math.inf
    reached, accepted = set(), []
    raw_rank, draft_round, passed, rounds = 0, 0, False, 0
    if quota:
        data = row["drafts"][cfg["drafter"]][str(k)]
        raw_rank = rank_of(target, data["first"][:k])
        for rounds, ids in enumerate((data["first"], *data[str(cfg["prefix_beams"])]), 1):
            reached.update(ids)
            for item in ids:
                passes = score_of(row["scores"], item) > theta
                if passes:
                    accepted.append((item, rounds))
                if item == target:
                    draft_round, passed = rounds, passes
            if len(accepted) >= quota or len(reached) >= len(catalog.sid_by_item):
                break
    entries = merge_selected(accepted, row["beam"], row["scores"], catalog, k, quota)
    rank = rank_of(target, [entry.item_id for entry in entries])
    source = entries[rank-1].source if rank else None
    rnd = entries[rank-1].round if rank else 0
    r = sum(entry.source == "retrieval" for entry in entries)
    actual = Outcome(target, rank, source, rnd, r)
    if actual != observed:
        raise ValueError(f"Winner prediction differs from actually executed rounds on row {row['row_id']}: {actual}, {observed}")
    retrieval_hit, beam_hit = source == "retrieval", source == "beam"
    return {
        "rows": 1, "ran_first_round": rounds >= 1, "ran_later_rounds": rounds >= 2,
        "first_drafted": draft_round == 1, "first_passed": draft_round == 1 and passed,
        "first_retrieved": retrieval_hit and rnd == 1,
        "later_drafted": draft_round > 1, "later_passed": draft_round > 1 and passed,
        "later_retrieved": retrieval_hit and rnd > 1,
        "any_drafted": bool(draft_round), "any_passed": passed,
        "retrieved": retrieval_hit, "beam_hits": beam_hit, "final_hits": bool(rank),
        "beam_hits_when_retrieval_quota_short": beam_hit and r < quota,
        "beam_hits_when_retrieval_quota_filled": beam_hit and r == quota,
        "target_missed_in_drafts": bool(quota) and not draft_round,
        "target_rejected_at_threshold": bool(draft_round) and not passed,
        "target_passed_outside_retrieval_quota": passed and not retrieval_hit,
        "winner_only_vs_raw": bool(quota and rank and not raw_rank),
        "raw_only_vs_winner": bool(quota and raw_rank and not rank),
        "shared_hits_vs_raw": bool(quota and raw_rank and rank),
        "shared_ranks_improved": bool(quota and raw_rank and rank and rank < raw_rank),
        "shared_ranks_worsened": bool(quota and raw_rank and rank and rank > raw_rank),
        "winner_only_from_first_below_k": bool(not raw_rank and retrieval_hit and rnd == 1),
        "winner_only_from_later_retrieval": bool(not raw_rank and retrieval_hit and rnd > 1),
        "winner_only_from_beam": bool(quota and not raw_rank and beam_hit),
    }


def build(root):
    root = Path(root)
    report = root/"report"
    selected = json.loads((report/"selection.json").read_text())
    winners = selected["winners"]
    cfgs = {r["id"]:r for r in selected["locked"]}
    catalog = Catalog.from_dict(json.loads((root/"catalog.json").read_text()))
    outcomes = saved_outcomes(report/"predictions"/"validation_winners.jsonl.gz", winners)
    methods = (*DRAFTERS, "beam", "winner")
    ranks = {(k,s,m):Counter() for k in KS for s in STRATA for m in methods}
    events = {(k,s):Counter() for k in KS for s in STRATA}
    seen = set()
    for row in stream_rows(files_for(root, "test")):
        i, target = row["row_id"], row["target"]
        if i in seen or not 0 <= i < TEST_ROWS:
            raise ValueError(f"Duplicate or invalid test row {i}")
        seen.add(i)
        group = "cold" if catalog.cold[target] else "warm"
        beam = [catalog.aliases[sid][0] for sid in row["beam"]]
        for k in KS:
            entry = winners[str(k)]
            observed = outcomes[entry][i]
            current = {d:rank_of(target,row["drafts"][d][str(k)]["first"][:k]) for d in DRAFTERS}
            current.update(beam=rank_of(target,beam[:k]), winner=observed.rank)
            funnel = trace_target(row, cfgs[entry], observed, catalog)
            for stratum in ("overall", group):
                for name, position in current.items():
                    record_rank(ranks[k,stratum,name], position)
                events[k,stratum].update(funnel)
    if seen != set(range(TEST_ROWS)):
        raise ValueError("Direct comparisons need the full test split")
    with (report/"test.csv").open() as fd:
        independent = {(r["id"],r["stratum"]):r for r in csv.DictReader(fd)}
    by_k = {}
    for k in KS:
        winner = winners[str(k)]
        baseline = next(c["id"] for c in cfgs.values() if c["k"] == k and c["drafter"] == "beam")
        groups = {}
        for s in STRATA:
            m = {name:metrics(ranks[k,s,name]) for name in methods}
            for name, key in (("beam",baseline),("winner",winner)):
                saved = independent[key,s]
                if (m[name]["rows"] != int(saved["rows"]) or m[name]["hits"] != int(saved["item_hits"])
                        or not math.isclose(m[name]["item_ndcg"], float(saved["item_ndcg"]), abs_tol=1e-10)):
                    raise ValueError(f"Diagnostics metrics disagree with independent test table for {key}/{s}")
            f = dict(events[k,s])
            if f["final_hits"] != f["retrieved"] + f["beam_hits"] or f["final_hits"] != m["winner"]["hits"]:
                raise ValueError("Incomplete attribution of final target hits")
            d = cfgs[winner]["drafter"]
            if d != "beam" and (f["raw_only_vs_winner"] + f["shared_hits_vs_raw"] != m[d]["hits"]
                    or f["winner_only_vs_raw"] + f["shared_hits_vs_raw"] != m["winner"]["hits"]
                    or f["winner_only_from_first_below_k"] + f["winner_only_from_later_retrieval"] +
                    f["winner_only_from_beam"] != f["winner_only_vs_raw"]):
                raise ValueError("Incomplete division of hits gained from original drafter top K")
            groups[s] = {"rankings":m, "funnel":f}
        by_k[str(k)] = {"config_id":winner,"drafter":cfgs[winner]["drafter"],"groups":groups}
    result = dict(metric="exact_item_id", test_rows=TEST_ROWS, selection="locked_validation_winner_per_k",
                  direct_drafter="First K in the recorded unfiltered first-round 2K ranking, before verifier scoring",
                  cutoff="Strict score > threshold; later rounds count only if actually reached",
                  beam_attribution="Beam hit split by whether actual retrieval filled its target quota on that request",
                  inputs=["test/rank*.jsonl","catalog.json","report/test.csv","report/selection.json",
                          "report/predictions/validation_winners.jsonl.gz"], by_k=by_k)
    write_json(report/"funnel.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    build(parser.parse_args().run_dir)
