"""Record per-round draft decisions by observing the production Engine.online path."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from hybrid.experiment import DRAFTERS, Data, Engine, dump
from hybrid.sweep import Catalog


class TraceEngine(Engine):
    """Trace one request at a time so each observed stage has a single owner."""

    def trace_request(self, example, cfg, catalog):
        self.rounds = []
        self.beam_sids = []
        self.rescores = {}
        self.threshold = cfg["threshold"]
        ranking, = self.online([example], cfg, catalog)
        ranks = {entry.item_id: rank for rank, entry in enumerate(ranking, start=1)}
        if len(ranks) != len(ranking) or len(ranking) > cfg["k"]:
            raise ValueError("Nonunique or overfull final ranking")
        retained = {entry.item_id for entry in ranking if entry.source == "retrieval"}
        accepted = []
        drafted = set()
        for rnd, trace in enumerate(self.rounds, start=1):
            for item in trace["candidates"]:
                item_id = item["item_id"]
                if item_id in drafted:
                    raise ValueError("Candidate redrafted across rounds")
                drafted.add(item_id)
                if item["threshold_pass"]:
                    accepted.append((item_id, item["score"], rnd))
                rank = ranks.get(item_id)
                item["final_rank"] = rank
                item["final_source"] = ranking[rank-1].source if rank is not None else None
        expected = sorted(accepted, key=lambda x: -x[1])[:cfg["quota"]]
        if {item for item, _, _ in expected} != retained:
            raise ValueError("Accepted candidate quota/ranking and live output differ")
        if self.rounds and self.rounds[-1]["cumulative_accepted"] != len(accepted):
            raise ValueError("Round acceptance totals differ")
        if len(retained) < cfg["k"] and not self.beam_sids:
            raise ValueError("Missing beam completion")
        if len(self.beam_sids) != len(set(self.beam_sids)):
            raise ValueError("Duplicate full beam SID")
        beam_order = {sid: i for i, sid in enumerate(self.beam_sids, start=1)}
        entries = []
        for rank, entry in enumerate(ranking, start=1):
            sid = self.data.sids[entry.item_id]
            if entry.source == "beam" and sid not in beam_order:
                raise ValueError("Final beam item is missing in the live beam")
            entries.append(dict(rank=rank, item_id=entry.item_id, sid=sid,
                                cold=bool(catalog.cold[entry.item_id]), source=entry.source,
                                accepted_round=entry.round or None, final_score=entry.score,
                                original_beam_rank=beam_order.get(sid)))
        stop = ("retrieval_quota_reached" if len(accepted) >= cfg["quota"] else
                "catalog_exhausted" if len(drafted) == len(self.data.catalog) else "round_limit")
        if stop == "round_limit" and len(self.rounds) != 3:
            raise ValueError("Trace lost a draft stage")
        return dict(row_id=example["row_id"], history=example["history"], target=example["target"],
                    target_cold=bool(catalog.cold[example["target"]]), config=cfg, rounds=self.rounds,
                    stop_reason=stop, accepted_count=len(accepted), retrieval_count=len(retained),
                    nominal_beam_quota=cfg["k"]-cfg["quota"],
                    retrieval_quota_shortfall=cfg["quota"]-len(retained),
                    actual_beam_count=sum(e.source == "beam" for e in ranking),
                    alpha_eff=len(retained)/cfg["k"], returned=len(ranking),
                    full_beam_sids=self.beam_sids,
                    beam_only_rescore_items=[dict(item_id=i, score=s) for i,s in self.rescores.items()],
                    final_ranking=entries)

    def request_drafts(self, drafter, query, allowed, excluded, limit):
        result = super().request_drafts(drafter, query, allowed, excluded, limit)
        # Counts describe the eligible catalog, independent of whether a given
        # HNSW node was visited internally. Apply prefix before cross-round exclusion.
        paths = set(allowed) if allowed is not None else None
        def matches(sid):
            return paths is None or any(sid[:len(path)] == path for path in paths)
        matched = {item for item, sid in self.data.catalog.items() if matches(sid)}
        previously = set(excluded)
        earlier = {c["item_id"] for rnd in self.rounds for c in rnd["candidates"]}
        if previously != earlier or not previously.issubset(self.data.catalog):
            raise ValueError("Live exclusion state disagrees with observed history")
        available = matched - previously
        if not set(result).issubset(available) or len(set(result)) != len(result):
            raise ValueError("Returned draft violates the catalog filters")
        level = len(self.rounds)
        if (level == 0) != (paths is None):
            raise ValueError("Unexpected unrestricted round")
        if paths is not None and any(len(path) != level for path in paths):
            raise ValueError("Beam prefix depth disagrees with the current drafting round")
        candidates = []
        for item in result:
            sid = self.data.sids[item]
            pieces = self.data.catalog[item]
            model_tokens = self.v.tokens.get(sid)
            if (sid != "".join(pieces) or model_tokens is None or
                    list(model_tokens) != self.v.tokenizer.encode(sid, add_special_tokens=False)):
                raise ValueError(f"Candidate SID/token mismatch for catalog item {item}")
            candidates.append(dict(draft_rank=len(candidates)+1, item_id=item, sid=sid,
                                   token_ids=list(model_tokens), cold=bool(self.data.record["cold"][item])))
        self.rounds.append(dict(
            round=level+1, allowed_prefixes=None if allowed is None else [list(p) for p in allowed],
            draft_limit=limit, returned=len(result),
            filters=dict(catalog_total=len(self.data.catalog), already_drafted_total=len(previously),
                         prefix_mismatch=len(self.data.catalog)-len(matched), prefix_matched=len(matched),
                         already_drafted_within_prefix=len(matched & previously),
                         eligible=len(available), eligible_not_returned=len(available)-len(result)),
            returned_checks=dict(catalog_member=True, sid_matches_catalog_and_tokenizer=True,
                                 prefix_matches=True, unique_item_ids=True, unseen_in_prior_round=True),
            candidates=candidates))
        return result

    def run_stage(self, key, fn, *args, **kwargs):
        result = super().run_stage(key, fn, *args, **kwargs)
        if key == "verification":
            scores, = result
            trace = self.rounds[-1]
            if set(scores) != {c["item_id"] for c in trace["candidates"]}:
                raise ValueError("Verification candidate membership differs")
            for candidate in trace["candidates"]:
                score = scores[candidate["item_id"]]
                if not math.isfinite(score):
                    raise ValueError("Invalid verifier score")
                passed = self.threshold is None or score > self.threshold
                candidate.update(score=score, threshold_pass=passed,
                                 threshold_decision="accepted" if passed else "rejected_at_or_below_threshold")
            num = sum(c["threshold_pass"] for c in trace["candidates"])
            trace.update(accepted_count=num, rejected_count=len(scores)-num,
                         cumulative_accepted=num + (self.rounds[-2]["cumulative_accepted"] if len(self.rounds)>1 else 0))
        elif key == "full_beam":
            self.beam_sids, = result
        elif key == "beam_rescore":
            self.rescores, = result
        return result


def resolve_threshold(spec, report):
    if spec == "all":
        return None
    if spec.startswith("p"):
        selection = json.loads((report/"selection.json").read_text())
        calibration = selection["calibration"]
        values = dict(zip(calibration["percentiles"], calibration["values"], strict=True))
        return float(values[int(spec[1:])])
    value = float(spec)
    if not math.isfinite(value):
        raise ValueError("Specify 'all', a validation percentile, or a finite log-probability threshold")
    return value


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--assets", default=Path("/assets"), type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--thresholds", nargs=2, default=["all", "0.0"], metavar=("LOWER", "UPPER"),
                    help="Two score thresholds. 'all' accepts every finite score; p10/p25/p50/p75/p90 use selection.json.")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    report = args.run_dir/"report"
    values = [resolve_threshold(s, report) for s in args.thresholds]
    if values[1] is None or (values[0] is not None and values[0] >= values[1]):
        ap.error("Thresholds must increase strictly")
    data = Data(args.assets)
    test = data.read("test")
    if not 1 <= args.samples <= len(test):
        ap.error("Invalid test sample size")
    indices = random.Random(42).sample(range(len(test)), args.samples)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    engine = TraceEngine(data, str(device))
    catalog = Catalog.from_dict(data.record)
    traces = []
    for name in DRAFTERS:
        for label, threshold in zip(args.thresholds, values):
            cfg = dict(drafter=name, k=10, alpha=0.4, quota=4, prefix_beams=5, rho=2,
                       max_rounds=3, threshold=threshold, threshold_label=label)
            for i in indices:
                traces.append(engine.trace_request(test[i], cfg, catalog))
    decisions = Counter(c["threshold_decision"] for t in traces for rnd in t["rounds"] for c in rnd["candidates"])
    coverage = dict(candidate_decisions=decisions, traces_with_acceptance=sum(t["accepted_count"]>0 for t in traces),
                    traces_with_extra_beam_slots=sum(t["retrieval_quota_shortfall"]>0 for t in traces),
                    traces_using_later_prefixes=sum(len(t["rounds"])>1 for t in traces))
    if args.thresholds == ["all", "0.0"] and (
        not coverage["traces_with_acceptance"] or not coverage["traces_with_extra_beam_slots"]
        or not coverage["traces_using_later_prefixes"]):
        raise ValueError("Default boundary trace did not cover all required branches")
    dest = args.run_dir/"trace.json"
    dump(dest, dict(split="test", sample_seed=42, sampled_row_ids=indices, execution="Engine.online, batch=1",
                    thresholds=dict(zip(args.thresholds,values)), coverage=coverage, traces=traces,
                    score_definition="Mean original-vocabulary log-probability over this catalog's three semantic tokens",
                    filters_definition="Catalog counts: prefix mismatch, then previously drafted within prefix. "
                                       "eligible_not_returned counts eligible items outside this round's returned candidates.",
                    threshold_rule="strictly greater; all accepts every finite score. Zero is a forced-fallback boundary.",
                    beam_score_definition="null final score denotes original pure beam order when no retrieval item was retained."))
    print(f"{len(traces)} online traces written to {dest}", flush=True)


if __name__ == "__main__":
    main()
