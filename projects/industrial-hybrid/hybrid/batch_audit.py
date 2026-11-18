"""Compare frozen quality rankings with live B1 and original round-robin B4 execution."""
import argparse
from collections import defaultdict
from datetime import timedelta
import gzip
import json
import math
import os
from pathlib import Path
import random

from hybrid.experiment import Data, Engine, KS, dump
from hybrid.sweep import Catalog, Selected


class AuditEngine(Engine):
    def observe(self, examples, cfg, catalog):
        self.scored = [{} for _ in examples]
        ranking = self.online(examples, cfg, catalog)
        return ranking, self.scored

    def get_scores(self, session, rows, items):
        values = super().get_scores(session, rows, items)
        for i, scores in zip(rows, values, strict=True):
            self.scored[i].update(scores)
        return values


def original_group(row_id, n):
    # Generation took test[r::4], then contiguous groups of four on rank r.
    first = 16 * (row_id // 16) + row_id % 4
    return tuple(range(first, min(n, first + 16), 4))


def read_archived(report, wanted, config_ids, examples):
    records = {}
    path = report/"predictions"/"validation_winners.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rid, cfg = obj["row_id"], obj["config_id"]
            if rid not in wanted or cfg not in config_ids:
                continue
            key = rid, cfg
            if key in records or obj["target"] != examples[rid]["target"]:
                raise ValueError(f"Duplicate or misaligned archived winner at {key}")
            records[key] = [Selected(e["item_id"], e["source"], e["round"], e["score"])
                            for e in obj["ranking"]]
    if len(records) != len(wanted) * len(config_ids):
        raise ValueError("Missing archived quality rankings for the audit requests")
    return records


def max_score_difference(a, b):
    shared = [i for i in a.keys() & b.keys() if a[i] is not None and b[i] is not None]
    if not shared:
        return dict(count=0, abs_diff=None, item_id=None)
    if not all(math.isfinite(a[i]) and math.isfinite(b[i]) for i in shared):
        raise ValueError("Nonfinite score in a compared ranking")
    item = max(shared, key=lambda i: (abs(a[i]-b[i]), -i))
    return dict(count=len(shared), abs_diff=abs(a[item]-b[item]), item_id=item,
                reference=a[item], candidate=b[item])


def compare(reference, candidate, target):
    lids = [entry.item_id for entry in reference]
    rids = [entry.item_id for entry in candidate]
    if len(set(lids)) != len(lids) or len(set(rids)) != len(rids):
        raise ValueError("A compared ranking contains a duplicate item")
    lm, rm = {e.item_id:e for e in reference}, {e.item_id:e for e in candidate}
    shared = lm.keys() & rm.keys()
    left_scores = {i:e.score for i,e in lm.items()}
    right_scores = {i:e.score for i,e in rm.items()}
    order_equal = lids == rids
    set_equal = lm.keys() == rm.keys()
    # Detect reordering that can be explained entirely by exact score ties in the archive.
    tied_only = False
    if set_equal and not order_equal:
        tied_only = all(lid == rid or (left_scores[lid] is not None and left_scores[lid] == left_scores[rid])
                        for lid, rid in zip(lids,rids,strict=True))
    return dict(set_equal=set_equal, order_equal=order_equal, exact_tie_reorder=tied_only,
                common_source_round_equal=all(lm[i].source==rm[i].source and lm[i].round==rm[i].round
                                              for i in shared),
                reference_target_rank=lids.index(target)+1 if target in lm else 0,
                candidate_target_rank=rids.index(target)+1 if target in rm else 0,
                common_scores=max_score_difference(left_scores, right_scores))


def make_row(rid, target, cfg, group, lens, archive, single, batch, single_scores, batch_scores):
    comparisons = dict(b1_vs_archive=compare(archive,single,target),
                       b4_online_vs_archive=compare(archive,batch,target),
                       b1_vs_b4_online=compare(batch,single,target))
    result = dict(row_id=rid, k=cfg["k"], config_id=cfg["id"], target=target,
                  batch_row_ids=group, prompt_tokens=lens[group.index(rid)], padded_prompt_tokens=max(lens),
                  comparisons=comparisons, common_verified=max_score_difference(batch_scores,single_scores))
    if (not all(s["order_equal"] and s["common_source_round_equal"] for s in comparisons.values())
        or rid==0):
        def compact(seq):
            return [dict(item_id=e.item_id, source=e.source, round=e.round, score=e.score) for e in seq]
        result["rankings"] = dict(archive=compact(archive), b1=compact(single), b4_online=compact(batch))
    return result


def paired_stats(comparisons):
    ref = [x["reference_target_rank"] for x in comparisons]
    cand = [x["candidate_target_rank"] for x in comparisons]
    n = len(comparisons)
    def ndcg(ranks):
        return sum(1.0/math.log2(1+r) for r in ranks if r)/n
    ref_hr, cand_hr = sum(bool(r) for r in ref)/n, sum(bool(r) for r in cand)/n
    ref_ndcg, cand_ndcg = ndcg(ref), ndcg(cand)
    return dict(n=n, set_equal=sum(x["set_equal"] for x in comparisons),
                order_equal=sum(x["order_equal"] for x in comparisons),
                exact_tie_reorders=sum(x["exact_tie_reorder"] for x in comparisons),
                same_set_and_source_round=sum(x["set_equal"] and x["common_source_round_equal"] for x in comparisons),
                reference_hr=ref_hr, candidate_hr=cand_hr, reference_ndcg=ref_ndcg, candidate_ndcg=cand_ndcg,
                hr_delta=cand_hr-ref_hr, ndcg_delta=cand_ndcg-ref_ndcg,
                gained_hits=sum(not a and bool(b) for a,b in zip(ref,cand)),
                lost_hits=sum(bool(a) and not b for a,b in zip(ref,cand)),
                target_rank_changed=sum(a!=b for a,b in zip(ref,cand)),
                common_selected_scored=sum(x["common_scores"]["count"] for x in comparisons),
                max_common_selected_score_diff=max((x["common_scores"]["abs_diff"] for x in comparisons
                                                    if x["common_scores"]["count"]), default=None))


def summarize(records):
    result = []
    for k in KS:
        selected = [r for r in records if r["k"] == k]
        stats = {key:paired_stats([r["comparisons"][key] for r in selected])
                 for key in ("b1_vs_archive","b4_online_vs_archive","b1_vs_b4_online")}
        p = stats["b1_vs_archive"]
        largest = max((r for r in selected if r["common_verified"]["count"]),
                      key=lambda r:r["common_verified"]["abs_diff"], default=None)
        result.append(dict(k=k, config_id=selected[0]["config_id"], n=p["n"],
                           set_equal=p["set_equal"], order_equal=p["order_equal"],
                           b1_hr=p["candidate_hr"], b4_hr=p["reference_hr"],
                           b1_ndcg=p["candidate_ndcg"], b4_ndcg=p["reference_ndcg"],
                           b1_gained_hits=p["gained_hits"], b1_lost_hits=p["lost_hits"],
                           b1_minus_b4_hr=p["hr_delta"], b1_minus_b4_ndcg=p["ndcg_delta"],
                           b4_online_order_equal=stats["b4_online_vs_archive"]["order_equal"],
                           max_common_selected_score_diff=p["max_common_selected_score_diff"],
                           max_common_verified_score_diff=largest["common_verified"]["abs_diff"] if largest else None,
                           max_common_verified_example=({"row_id":largest["row_id"],**largest["common_verified"]}
                                                        if largest else None),
                           common_verified_scored=sum(r["common_verified"]["count"] for r in selected),
                           pairs=stats))
    return result


def execute(engine, catalog, examples, mine, configs, archive):
    groups = defaultdict(list)
    for rid in sorted(mine):
        groups[original_group(rid, len(examples))].append(rid)
    lengths = {group:[len(ids) for ids in engine.v.prompts([examples[r]["history_sids"] for r in group])]
               for group in groups}
    rows = []
    for cfg in configs:
        for group, wanted in groups.items():
            batch, batch_scores = engine.observe([examples[r] for r in group], cfg, catalog)
            for rid in wanted:
                lane = group.index(rid)
                single, single_scores = engine.observe([examples[rid]], cfg, catalog)
                rows.append(make_row(rid, examples[rid]["target"], cfg, group, lengths[group],
                                     archive[rid,cfg["id"]], single[0], batch[lane], single_scores[0],
                                     batch_scores[lane]))
        print(f'Rank {os.environ.get("RANK","0")}: completed {len(mine)} rows at K={cfg["k"]}', flush=True)
    return rows


def main():
    import torch
    import torch.distributed as dist
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--assets", type=Path, default=Path("/assets"))
    ap.add_argument("--samples", type=int, default=256, help="Use the first N of the fixed benchmark requests")
    ap.add_argument("--rqvae", action="store_true", help="Use sealed RQ-VAE and retrained UniSRec assets")
    args = ap.parse_args()
    if not 1 <= args.samples <= 256:
        ap.error("samples must be between 1 and 256")
    rank, world = int(os.getenv("RANK",0)), int(os.getenv("WORLD_SIZE",1))
    if world not in (1,2,4):
        ap.error("Launch with one, two, or four workers")
    device = int(os.getenv("LOCAL_RANK",rank))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    report = args.run_dir/"report"
    if args.rqvae:
        from hybrid.rqvae_assets import RQVAEData

        data = RQVAEData(args.assets)
    else:
        data = Data(args.assets)
    test = data.read("test")
    protocol = json.loads((report/"benchmark_protocol.json").read_text())
    fixed = protocol["sample_row_ids"]
    expected = random.Random(42).sample(range(len(test)),len(fixed))
    if fixed != expected or len(fixed)<args.samples:
        raise ValueError("Performance benchmark request IDs differ from the pinned sample")
    sample = fixed[:args.samples]
    selection = json.loads((report/"selection.json").read_text())
    by_id = {c["id"]:c for c in selection["locked"]}
    configs = [by_id[selection["winners"][str(k)]] for k in KS]
    if [c["k"] for c in configs] != list(KS):
        raise ValueError("Frozen winners are misaligned")
    # Row zero documents the reported example, independently of the random sample.
    wanted = set(sample) | {0}
    mine = {rid for rid in wanted if (rid%4)%world==rank}
    archive = read_archived(report, mine, {c["id"] for c in configs}, test)
    names = sorted({c["drafter"] for c in configs if c["quota"]})
    engine = AuditEngine(data,f"cuda:{device}",names=names)
    rows = execute(engine,Catalog.from_dict(data.record),test,mine,configs,archive)
    if world > 1:
        dist.init_process_group("gloo",timeout=timedelta(hours=1))
        parts = [None]*world if rank==0 else None
        dist.gather_object(rows,parts,dst=0)
        dist.destroy_process_group()
        if rank!=0:
            return
        rows = [r for part in parts for r in part]
    if {(r["k"],r["row_id"]) for r in rows} != {(k,r) for k in KS for r in wanted} or len(rows)!=len(wanted)*len(KS):
        raise ValueError("Audit rows are missing or duplicated")
    rows.sort(key=lambda r:(r["k"],r["row_id"]))
    sample_set = set(sample)
    sample_rows = [r for r in rows if r["row_id"] in sample_set]
    out = dict(status="measured", sample_row_ids=sample, seed=42, rows=summarize(sample_rows),
               score_scope="Max absolute difference on the same item. Selected-item comparisons require a numeric "
                           "score in both rankings. Verified comparisons include all items actually scored by both live calls.",
               metric="Exact catalog item ID HR and single-relevant-item NDCG; fractions in [0,1], deltas candidate minus reference.",
               batch_rule="Original generation rank=r%4; group starts at 16*(r//16)+r%4 and strides four through at most four rows.",
               rows_detail=sample_rows, diagnostic_row_zero=[r for r in rows if r["row_id"]==0])
    dump(report/"batch_audit.json",out)


if __name__=="__main__":
    main()
