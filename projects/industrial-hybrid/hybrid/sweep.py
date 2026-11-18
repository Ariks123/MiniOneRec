"""Quota-constrained Industrial recommendation. Quality replay uses the live model's primitives."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import sys
import time
from typing import NamedTuple

import numpy as np

KS = (1, 5, 10, 20, 50)
WIDTHS = (1, 5, 10, 20, 50)
DRAFTERS = ("ann", "unisrec", "qwen")
STRATA = ("overall", "warm", "cold")
VALID_ROWS, TEST_ROWS = 4532, 4533
REF_PATH = Path("/mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl/evaluation/predictions.jsonl")


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Expected at least one result row")
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items()})


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Catalog:
    sid_by_item: list
    target_freq: list
    cold: list
    aliases: dict
    sid_index: dict
    sid_codes: np.ndarray
    alias_matrix: np.ndarray

    @classmethod
    def from_dict(cls, data):
        sids = data["sid_by_item"]
        freq = data["target_freq"]
        cold = data["cold"]
        if len(sids) != len(freq) or len(sids) != len(cold) or not sids:
            raise ValueError("Catalog must have a frequency, cold flag and SID for every item")
        aliases = defaultdict(list)
        for i,sid in enumerate(sids):
            aliases[sid].append(i)
        aliases = {s:sorted(items, key=lambda j:(-freq[j],j)) for s,items in aliases.items()}
        si = {s:i for i,s in enumerate(aliases)}
        matrix = np.full((len(si), max(map(len,aliases.values()))), -1, dtype=np.int32)
        for s, items in aliases.items():
            matrix[si[s], :len(items)] = items
        return cls(sids,freq,cold,aliases,si,np.array([si[s] for s in sids],dtype=np.int32),matrix)


class Selected(NamedTuple):
    item_id: int
    source: str
    round: int
    score: float | None


def score_of(scores, item):
    try:
        score = scores[item]
    except (KeyError, IndexError):
        score = scores[str(item)]
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"Missing finite verifier score for item {item}")
    return score


def merge_selected(accepted, beam, scores, catalog, k, quota):
    """Shared live/reference finalization. accepted is ordered by round and draft order."""
    ordered = dict()
    for item, rnd in accepted:
        ordered.setdefault(int(item), int(rnd))
    result = [Selected(i,"retrieval",r,score_of(scores,i)) for i,r in ordered.items()]
    result.sort(key=lambda row:-row.score)
    result = result[:quota]
    taken = {row.item_id for row in result}
    r = len(taken)
    seen_sids = set()
    for sid in beam:
        if len(result) == k:
            break
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        item = next((i for i in catalog.aliases[sid] if i not in taken), None)
        if item is not None:
            taken.add(item)
            result.append(Selected(item,"beam",0,score_of(scores,item) if r else None))
    if r:
        result.sort(key=lambda row:-row.score)
    return result


def alpha_groups(k):
    groups = defaultdict(list)
    for a in sorted({*(Fraction(i,10) for i in range(11)),Fraction(1,k)}):
        groups[int(k*a)].append(float(a))
    return dict(groups)


def make_config(d,k,q,b,t,aliases):
    sid = "all" if t is None else float(t).hex()
    ident = f"{d}-k{k}-q{q}-b{b}-" + ("all" if t is None else hashlib.sha256(sid.encode()).hexdigest()[:12])
    return dict(id=ident,drafter=d,k=k,quota=q,prefix_beams=b,alpha=aliases[0],
                alpha_aliases=aliases,threshold=t)


def configurations(thresholds):
    for k in KS:
        groups = alpha_groups(k)
        yield make_config("beam",k,0,0,None,groups[0])
        for d in DRAFTERS:
            for b in WIDTHS:
                for q,aliases in groups.items():
                    if q:
                        for t in thresholds:
                            yield make_config(d,k,q,b,t,aliases)


def files_for(root, split):
    directory=root/split
    paths = sorted({*directory.glob("*.jsonl"),*directory.glob("*.jsonl.gz")}) if directory.is_dir() else []
    if not paths:
        paths=[p for p in [root/f"{split}.jsonl",root/f"{split}.jsonl.gz"] if p.is_file()]
    if not paths:
        raise ValueError(f"Missing {split} cached primitives at {directory}")
    return paths


def stream_rows(paths):
    for path in paths:
        op = gzip.open if path.suffix==".gz" else open
        with op(path,"rt",encoding="utf-8") as fd:
            for line in fd:
                if line.strip():
                    yield json.loads(line)


def calibrate(root):
    wanted=set(random.Random(42).sample(range(VALID_ROWS),512))
    scores=[]
    found=set()
    for row in stream_rows(files_for(root,"valid")):
        i=row["row_id"]
        if i not in wanted:
            continue
        if i in found:
            raise ValueError("Duplicate validation calibration row")
        found.add(i)
        # Pool actual first rounds for each drafter and K. No target labels enter calibration.
        for d in DRAFTERS:
            for k in KS:
                ids=row["drafts"][d][str(k)]["first"]
                scores.extend(score_of(row["scores"],int(item)) for item in ids)
    if found!=wanted or not scores:
        raise ValueError("Incomplete 512-row calibration sample")
    return dict(seed=42,rows=sorted(found),positions=len(scores),percentiles=[10,25,50,75,90],
                values=np.percentile(scores,[10,25,50,75,90]).tolist())


class Packed:
    def __init__(self,rows,catalog):
        self.rows=rows
        self.catalog=catalog
        self.n=len(rows)
        self.ix=np.arange(self.n)[:,None]
        self.items=len(catalog.sid_by_item)
        self.row_ids=np.array([r["row_id"] for r in rows],dtype=np.int32)
        self.targets=np.array([r["target"] for r in rows],dtype=np.int32)
        self.cold=np.asarray(catalog.cold,dtype=bool)[self.targets]
        self.beam_codes=np.array([[catalog.sid_index[s] for s in r["beam"]] for r in rows],dtype=np.int32)
        if self.beam_codes.shape!=(self.n,50) or any(len(set(r["beam"]))!=50 for r in rows):
            raise ValueError("Full beam must contain exactly 50 distinct valid SIDs")
        self.aliases=catalog.alias_matrix[self.beam_codes]
        self.raw_scores=np.full((self.n,self.items+1),np.nan,dtype=np.float64)
        for j,r in enumerate(rows):
            pairs=r["scores"]
            if isinstance(pairs,dict):
                ids=np.fromiter(map(int,pairs),dtype=np.int32,count=len(pairs))
                if ids.size and (ids.min()<0 or ids.max()>=self.items):
                    raise ValueError("Score for unknown item")
                self.raw_scores[j,ids]=np.fromiter(pairs.values(),dtype=np.float64,count=len(pairs))
            else:
                self.raw_scores[j,:self.items]=pairs
        self.used=np.zeros((self.n,self.items+1),dtype=np.int32)
        self.stamp=0

    def items_in(self,d,k,b):
        cols=2*k
        all_ids=np.full((self.n,3*cols),-1,dtype=np.int32)
        for i,row in enumerate(self.rows):
            data=row["drafts"][d][str(k)]
            rounds=(data["first"],*data[str(b)])
            if len(rounds)!=3:
                raise ValueError("Expected first round and two prefix-guided rounds")
            ids=[int(x) for group in rounds for x in group]
            if len(ids)!=len(set(ids)) or any(v<0 or v>=self.items for v in ids):
                raise ValueError(f"Repeated or invalid draft on row {row['row_id']}")
            for j,group in enumerate(rounds):
                if len(group)>cols:
                    raise ValueError("Draft round exceeds 2K")
                all_ids[i,j*cols:j*cols+len(group)]=group
        scores=self.raw_scores[self.ix,all_ids]
        if not np.all(np.isfinite(scores[all_ids>=0])):
            raise ValueError("Missing finite draft verifier score")
        scores=np.where(all_ids<0,-np.inf,scores)
        exists=all_ids.reshape(self.n,3,cols)>=0
        all_positions=np.broadcast_to(np.arange(3*cols,dtype=np.int32),all_ids.shape)
        sorted_prefixes=[]
        for j in (1,2,3):
            length=j*cols
            idx=np.argsort(-scores[:,:length],axis=1,kind="stable")[:,:k]
            sorted_prefixes.append((all_ids[self.ix,idx],scores[self.ix,idx],
                                    1+all_positions[self.ix,idx]//cols))
        return scores.reshape(self.n,3,cols),exists.sum(axis=2),sorted_prefixes

    def pure(self,k):
        z=np.zeros(self.n,dtype=np.int32)
        out=self.aliases[:,:k,0]
        return out,np.zeros_like(out),np.full(out.shape,np.nan),z,z,z,z

    def replay(self,c,primitives,threshold_state):
        """Vectorized form of merge_selected with per-request stop and accepted-count accounting."""
        k,q=c["k"],c["quota"]
        _,size,prefixes=primitives
        cumulative=threshold_state
        # An empty eligible prefix need not exhaust the next frontier.
        exhausted=np.cumsum(size,axis=1)>=self.items
        stops=np.where((cumulative[:,0]>=q)|exhausted[:,0],0,np.where((cumulative[:,1]>=q)|exhausted[:,1],1,2))
        r=np.minimum(q,cumulative[np.arange(self.n),stops])
        accepted=cumulative[np.arange(self.n),stops]
        verified=size[:,0]+(stops>=1)*size[:,1]+(stops>=2)*size[:,2]
        rounds=stops+1
        ret,ret_sc,ret_round=(np.empty((self.n,q),dtype=dt) for dt in (np.int32,float,np.int32))
        for j,(ids,sc,rnd) in enumerate(prefixes):
            ix=stops==j
            ret[ix]=ids[ix,:q]
            ret_sc[ix]=sc[ix,:q]
            ret_round[ix]=rnd[ix,:q]
        mask=np.arange(q)[None,:]<r[:,None]
        ret[~mask]=-1
        self.stamp+=1
        self.used[self.ix,ret]=self.stamp
        # Each unique beam SID can contribute one available alias.
        available=self.aliases>=0
        for j in range(self.aliases.shape[2]):
            available[:,:,j]&=self.used[self.ix,self.aliases[:,:,j]]!=self.stamp
        first=available.argmax(axis=2)
        bm=np.take_along_axis(self.aliases,first[:,:,None],axis=2)[:,:,0]
        bm[~available.any(axis=2)]=-1
        bm=bm[:,:k+q]  # at most r<=q distinct beam groups can be exhausted
        bm[np.cumsum(bm>=0,axis=1)>(k-r)[:,None]]=-1
        beam_sc=self.raw_scores[self.ix,bm]
        need_score=(r[:,None]>0)&(bm>=0)
        if not np.isfinite(beam_sc[need_score]).all():
            raise ValueError("Missing finite verifier score for a selected beam item/alias")
        combos=np.concatenate([ret,bm],axis=1)
        sc=np.concatenate([ret_sc,beam_sc],axis=1)
        order_sc=np.where(combos<0,-np.inf,np.where(r[:,None]==0,0,sc))
        order=np.argsort(-order_sc,axis=1,kind="stable")[:,:k]
        out=combos[self.ix,order]
        source_round=np.concatenate([np.where(mask,ret_round,0),np.zeros_like(bm)],axis=1)[self.ix,order]
        out_sc=np.where((out>=0)&(r[:,None]>0),sc[self.ix,order],np.nan)
        return out,source_round,out_sc,r,rounds,verified,accepted


def blocks(root, split, catalog, total, reference=None, size=128):
    rows=[]; seen=set()
    for row in stream_rows(files_for(root,split)):
        rid = row["row_id"]
        if not isinstance(rid,int) or rid<0 or rid>=total or rid in seen:
            raise ValueError(f"Invalid/duplicate {split} row {rid}")
        if reference is not None and (row["beam"]!=reference[rid][0] or
                                      catalog.sid_by_item[row["target"]]!=reference[rid][1]):
            raise ValueError(f"Original beam predictions/target differ on test row {rid}")
        seen.add(rid); rows.append(row)
        if len(rows)>=size:
            yield Packed(rows,catalog); rows=[]
    if rows:
        yield Packed(rows,catalog)
    if len(seen)!=total:
        raise ValueError(f"{split} has {len(seen)}/{total} rows")


def accumulate(block, replay, accumulator):
    items,rounds_by_item,scores,r,rounds,verified,accepted=replay
    valid=items>=0
    hits=valid & (items==block.targets[:,None])
    sid_hits=valid & (block.catalog.sid_codes[items]==block.catalog.sid_codes[block.targets,None])
    pos=np.where(hits.any(axis=1),hits.argmax(axis=1)+1,0)
    spos=np.where(sid_hits.any(axis=1),sid_hits.argmax(axis=1)+1,0)
    # Sum integer counts and discounted hits in float64. One target per test row.
    flags=np.arange(block.n)
    for j,subset in enumerate((flags, flags[~block.cold],flags[block.cold])):
        if not len(subset): continue
        p,sp = pos[subset],spos[subset]
        h,sh = p>0,sp>0
        accumulator[j]+=np.array([len(p),h.sum(),np.sum(1/np.log2(1+p[h])),
            sh.sum(),np.sum(1/np.log2(1+sp[sh])),r[subset].sum(),rounds[subset].sum(),
            verified[subset].sum(),accepted[subset].sum(),(r[subset]<items.shape[1]).sum(),
            np.sum(valid[subset])],dtype=np.float64)


def result_rows(configs, totals):
    rows=[]
    for c in configs:
        for j,stratum in enumerate(STRATA):
            n,h,n_sum,sh,s_sum,r,rnds,verified,accepted,beam_calls,returned=totals[c["id"]][j]
            if not n: continue
            rows.append({**c,"stratum":stratum,"rows":int(n),"item_hits":int(h),"item_hr":h/n,
                         "item_ndcg":n_sum/n,"sid_hits":int(sh),"sid_hr":sh/n,"sid_ndcg":s_sum/n,
                         "alpha_eff":r/(n*c["k"]),"mean_retrieved":r/n,"mean_rounds":rnds/n,
                         "mean_verified":verified/n,"acceptance_rate":accepted/verified if verified else 0.0,
                         "below_retrieval_quota_rate":None,"beam_call_rate":beam_calls/n,"mean_returned":returned/n})
    return rows


def evaluate(root,split,catalog,configs,*,reference=None,winner_ids=()):
    root=Path(root)
    total=VALID_ROWS if split=="valid" else TEST_ROWS
    groups=defaultdict(list)
    for c in configs:
        groups[c["drafter"],c["k"],c["prefix_beams"]].append(c)
    counters={c["id"]:np.zeros((3,11),dtype=np.float64) for c in configs}
    fallback={c["id"]:np.zeros(3,dtype=np.int64) for c in configs}
    is_test=split=="test"
    outdir=root/"report"/"predictions"
    if is_test:
        outdir.mkdir(parents=True,exist_ok=True)
        winner_file=gzip.open(outdir/"validation_winners.jsonl.gz.tmp","wt",encoding="utf-8",compresslevel=2)
    n=0; tick=time.monotonic()
    try:
        for blockid,blk in enumerate(blocks(root,split,catalog,total,reference)):
            archive={"row_ids":blk.row_ids,"targets":blk.targets}
            for (d,k,b),cs in groups.items():
                prim=None if d=="beam" else blk.items_in(d,k,b)
                for t in dict.fromkeys(c["threshold"] for c in cs):
                    state=None if prim is None else np.cumsum((prim[0]>(-np.inf if t is None else t)).sum(axis=2),axis=1)
                    for c in (x for x in cs if x["threshold"]==t):
                        out=blk.pure(k) if prim is None else blk.replay(c,prim,state)
                        accumulate(blk,out,counters[c["id"]])
                        misses=out[3]<c["quota"]
                        fallback[c["id"]]+=(misses.sum(),np.sum(misses&~blk.cold),np.sum(misses&blk.cold))
                        if is_test:
                            ident=c["id"]
                            archive[ident+"_item"]=out[0].astype(np.int16)
                            archive[ident+"_round"]=out[1].astype(np.int8)
                            archive[ident+"_score"]=out[2].astype(np.float32)
                            if ident in winner_ids:
                                for j,rid in enumerate(blk.row_ids):
                                    ranking=[dict(item_id=int(i),sid=catalog.sid_by_item[i],source="retrieval" if rnd else "beam",
                                                  round=int(rnd),score=float(v) if np.isfinite(v) else None)
                                             for i,rnd,v in zip(out[0][j],out[1][j],out[2][j]) if i>=0]
                                    winner_file.write(json.dumps(dict(config_id=ident,row_id=int(rid),target=int(blk.targets[j]),
                                        alpha_eff=float(out[3][j]/k),ranking=ranking),separators=(",",":"))+"\n")
            if is_test:
                # ZIP rewrites its headers; keep random writes on local storage and
                # publish the finished archive sequentially on the OSS mount.
                target=outdir/f"batch-{blockid:04d}.npz"
                pending=target.with_suffix(".npz.tmp")
                with tempfile.TemporaryFile() as spool:
                    np.savez_compressed(spool,**archive)
                    spool.seek(0)
                    with pending.open("wb") as sink:
                        shutil.copyfileobj(spool,sink,4 << 20)
                pending.replace(target)
            n+=blk.n
            print(f"{split} {n}/{total} rows elapsed={time.monotonic()-tick:.1f}s",file=sys.stderr,flush=True)
    finally:
        if is_test: winner_file.close()
    if is_test:
        (outdir/"validation_winners.jsonl.gz.tmp").replace(outdir/"validation_winners.jsonl.gz")
    rows=result_rows(configs,counters)
    for row in rows:
        row["below_retrieval_quota_rate"]=fallback[row["id"]][STRATA.index(row["stratum"])]/row["rows"]
    return rows


def rank_key(row):
    return (-row["item_ndcg"],-row["item_hr"],row["mean_verified"])


def choose(configs, rows):
    by_id={c["id"]:c for c in configs}
    overall=[r for r in rows if r["stratum"]=="overall"]
    by_combo=defaultdict(list)
    for row in overall:
        by_combo[row["drafter"],row["k"],row["quota"],row["prefix_beams"]].append(row)
    locked=[min(group,key=rank_key) for group in by_combo.values()]
    winners={str(k):min((r for r in locked if r["k"]==k),key=rank_key)["id"] for k in KS}
    bench={}
    def retain(row,label):
        c=by_id[row["id"]]
        if c["id"] not in bench: bench[c["id"]]={**c,"labels":[]}
        bench[c["id"]]["labels"].append(label)
    for k in KS:
        pure=next(r for r in locked if r["k"]==k and r["drafter"]=="beam")
        retain(pure,"pure_beam")
        for d in DRAFTERS:
            sub=[r for r in locked if r["k"]==k and r["drafter"]==d]
            retain(min(sub,key=rank_key),"best_"+d)
            for alpha in (Fraction(2,5),Fraction(1)):
                q=int(alpha*k)
                hit=min((r for r in sub if r["quota"]==q),key=rank_key) if q else pure
                retain(hit,f"{d}_alpha_{float(alpha):g}")
    return [by_id[r["id"]] for r in locked], winners, list(bench.values())


def read_reference(root):
    path=root/"reference_test_beam.jsonl"
    if not path.is_file():
        path=REF_PATH
    reference={}
    for row in stream_rows([path]):
        if row["row_id"] in reference:
            raise ValueError("Duplicate row in frozen original beam reference")
        reference[row["row_id"]]=(row["predictions"],row["target"])
    if sorted(reference)!=list(range(TEST_ROWS)):
        raise ValueError("Missing original beam reference rows")
    return reference,sha256(path)


def select(root):
    report=root/"report"
    report.mkdir(parents=True,exist_ok=True)
    catalog=Catalog.from_dict(json.loads((root/"catalog.json").read_text()))
    calibration=calibrate(root)
    thresholds=[None,*dict.fromkeys(calibration["values"])]
    configs=list(configurations(thresholds))
    rows=evaluate(root,"valid",catalog,configs)
    locked,winners,bench=choose(configs,rows)
    write_csv(report/"validation.csv",rows)
    write_json(report/"benchmark_configs.json",bench)
    write_json(report/"selection.json",dict(primary="validation exact item NDCG@K",
        tie_break=["higher item HR","fewer mean verified candidates","predefined grid order"],
        valid_rows=VALID_ROWS,seed=42,validation_configurations=len(configs),test_configurations=len(locked),
        catalog_sha256=sha256(root/"catalog.json"),
        valid_inputs={p.name:sha256(p) for p in files_for(root,"valid")},
        calibration=calibration,winners=winners,locked=locked))


def quality(root):
    report=root/"report"
    if not (report/"selection.json").is_file():
        select(root)
    selection=json.loads((report/"selection.json").read_text())
    if selection["valid_inputs"]!={p.name:sha256(p) for p in files_for(root,"valid")}:
        raise ValueError("Validation primitive cache changed after model selection")
    if selection["catalog_sha256"]!=sha256(root/"catalog.json"):
        raise ValueError("Catalog changed after model selection")
    catalog=Catalog.from_dict(json.loads((root/"catalog.json").read_text()))
    reference,ref_hash=read_reference(root)
    rows=evaluate(root,"test",catalog,selection["locked"],reference=reference,
                  winner_ids=set(selection["winners"].values()))
    original={r["k"]:r for r in rows if r["drafter"]=="beam" and r["stratum"]=="overall"}
    for k,strict,compatible in zip(KS,[188,409,544,699,910],[259,490,625,781,995]):
        if original[k]["item_hits"]!=strict or original[k]["sid_hits"]!=compatible:
            raise ValueError(f"Pure beam K={k} failed frozen strict-ID/SID count regression: {original[k]}")
    for r in rows:
        if r["stratum"]=="cold" and r["rows"]!=147:
            raise ValueError("Test cold stratum must have exactly 147 rows")
    write_csv(report/"test.csv",rows)
    write_json(report/"quality.json",dict(status="passed",test_rows=TEST_ROWS,validation_rows=VALID_ROWS,
        catalog_items=len(catalog.sid_by_item),catalog_sids=len(catalog.sid_index),
        primary="exact_item_id_ndcg_at_k",fixed_before_test=True,locked_configs=len(selection["locked"]),
        reference_sha256=ref_hash,original_beam_sid_order_checked=TEST_ROWS,
        original_beam_hits={str(k):[original[k]["item_hits"],original[k]["sid_hits"]] for k in KS},
        winners={str(k):[r for r in rows if r["id"]==selection["winners"][str(k)]] for k in KS},
        prediction_layout={"files":"predictions/batch-*.npz","index":"row_ids","target":"targets",
            "per_configuration":"<config_id>_item, _round, _score: row x K; item -1 is unfilled",
            "round":"0 beam, 1-3 retrieval","score":"NaN on pure beam or unfilled positions",
            "winner_detail":"predictions/validation_winners.jsonl.gz",
            "draft_trace":"Input primitive shards retain candidate IDs for all three rounds; verifier threshold and stop can be replayed."}))
    from .paired import build as paired_intervals
    from .report import render
    paired_intervals(root)
    render(root)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True,type=Path)
    p.add_argument("stage",choices=["select","quality","report"])
    args=p.parse_args(argv)
    if args.stage=="report":
        from .report import render
        render(args.run_dir)
    elif args.stage=="select":
        select(args.run_dir)
    else:
        quality(args.run_dir)


if __name__=="__main__":
    main()
