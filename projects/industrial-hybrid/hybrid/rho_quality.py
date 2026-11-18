"""Validation-only selection and frozen test replay for UniSRec K versus 2K drafts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from . import diagnostics, paired, sweep
from .sweep import KS, WIDTHS, STRATA, TEST_ROWS, VALID_ROWS

ARMS = ("beam", "ann", "unisrec2", "qwen", "unisrec1")
DIRECT_HITS = (169, 455, 638, 853, 1177)
DIRECT_COLD_HITS = (1, 6, 9, 10, 15)


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cfg_for(arm, k, quota, width, threshold, aliases):
    name = "unisrec" if arm.startswith("unisrec") else arm
    c = sweep.make_config(name, k, quota, width, threshold, aliases)
    if arm == "unisrec1":
        c["id"] = c["id"].replace("unisrec-", "unisrec-rho1-", 1)
    return {**c, "arm":arm, "rho":1 if arm=="unisrec1" else 2}


def grid(thresholds):
    configs=[]
    for k in KS:
        groups=sweep.alpha_groups(k)
        configs.append(cfg_for("beam",k,0,0,None,groups[0]))
        for arm in ARMS[1:]:
            for width in WIDTHS:
                for q,aliases in groups.items():
                    if q:
                        configs.extend(cfg_for(arm,k,q,width,t,aliases) for t in thresholds)
    return configs


def inputs(path, split):
    paths=sweep.files_for(path,split)
    if [p.name for p in paths]!=[f"rank{i}.jsonl" for i in range(4)]:
        raise ValueError(f"Incomplete {split} primitive shards in {path}")
    return {p.name:sweep.sha256(p) for p in paths}


def calibrate(old, root, historical):
    wanted=set(random.Random(42).sample(range(VALID_ROWS),512))
    percentiles=[10,25,50,75,90]
    if set(historical["rows"])!=wanted or historical["percentiles"]!=percentiles:
        raise ValueError("Historical calibration sample differs from the planned protocol")
    values=[]
    for source,names in ((old,("ann","qwen")),(root,("unisrec",))):
        found=set()
        for row in sweep.stream_rows(sweep.files_for(source,"valid")):
            i=row["row_id"]
            if i not in wanted:
                continue
            if i in found:
                raise ValueError("Duplicate calibration request")
            found.add(i)
            for d in names:
                for k in KS:
                    drafts=row["drafts"][d][str(k)]["first"]
                    limit=k if d=="unisrec" else 2*k
                    if len(drafts)!=limit:
                        raise ValueError("Unexpected first-round calibration draft size")
                    values.extend(sweep.score_of(row["scores"],item) for item in drafts)
        if found!=wanted:
            raise ValueError("Missing calibration requests")
    if len(values)!=512*sum(KS)*5:
        raise ValueError("Missing first-round candidate positions")
    return dict(seed=42,rows=sorted(wanted),percentiles=percentiles,positions=len(values),
                values=np.percentile(values,percentiles).tolist())


def eval_one(job):
    source, output, catalog_record, split, configs, detail, reference_root = job
    catalog=sweep.Catalog.from_dict(catalog_record)
    if split=="valid":
        return sweep.evaluate(source,split,catalog,configs)
    reference,_=sweep.read_reference(reference_root)
    # The legacy evaluator writes compressed predictions alongside its input.
    # Give every arm a private output folder while keeping the source cache read-only.
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="industrial-rho-eval-") as tmp:
        view=Path(tmp)
        (view/split).symlink_to(source/split,target_is_directory=True)
        (view/"report").mkdir()
        (view/"report"/"predictions").symlink_to(output,target_is_directory=True)
        return sweep.evaluate(view,split,catalog,configs,reference=reference,winner_ids=detail)


def eval_configs(old, root, configs, split, catalog, detail=()):
    jobs=[]
    for arm in ARMS:
        subset=[c for c in configs if c["arm"]==arm]
        if subset:
            source=root if arm=="unisrec1" else old
            output=root/"report"/"predictions"/arm
            jobs.append((source,output,catalog,split,subset,set(detail),old))
    rows=[]
    with ProcessPoolExecutor(max_workers=min(5,len(jobs))) as pool:
        for part in pool.map(eval_one,jobs):
            rows.extend(part)
    return rows


def restore(old_rows, by_id):
    result=[]
    for row in old_rows:
        c=by_id[row["id"]]
        metrics={k:(int(v) if k in ("rows","item_hits","sid_hits") else float(v))
                 for k,v in row.items() if k not in c and k!="stratum"}
        result.append(dict(**c,stratum=row["stratum"],**metrics))
    return result


def best(configs, numbers):
    return min(configs,key=lambda c:sweep.rank_key(numbers[c["id"],"overall"]))


def per_arm(candidates, numbers):
    return {str(k):{arm:best([c for c in candidates if c["k"]==k and c["arm"]==arm],numbers)["id"]
            for arm in ARMS} for k in KS}


def performance_plan(lock, arm_winners, legacy, by_id, numbers):
    cases={}
    def add(c,label):
        case=cases.setdefault(c["id"],dict(**c,labels=[]))
        case["labels"].append(label)
    for k in KS:
        for arm in ARMS:
            add(by_id[arm_winners[str(k)][arm]],"best_"+arm)
            if arm=="beam":
                continue
            for alpha in (Fraction(2,5),Fraction(1)):
                quota=int(alpha*k)
                candidates=[c for c in lock if c["k"]==k and c["quota"]==quota and c["arm"]==(arm if quota else "beam")]
                add(best(candidates,numbers),f"{arm}_alpha_{float(alpha):g}")
        add(by_id[legacy[str(k)]],"published_winner")
    return list(cases.values())


def select(old, root):
    report=root/"report"
    report.mkdir(parents=True,exist_ok=True)
    previous=read(old/"report"/"selection.json")
    valid_hashes={"old":inputs(old,"valid"),"new":inputs(root,"valid")}
    if previous["valid_inputs"]!=valid_hashes["old"] or previous["catalog_sha256"]!=sweep.sha256(old/"catalog.json"):
        raise ValueError("Historical frozen validation inputs or catalog changed")
    old_calibration=previous["calibration"]
    new_calibration=calibrate(old,root,old_calibration)
    original=[None,*old_calibration["values"]]
    thresholds=list(dict.fromkeys([*original,*new_calibration["values"]]))
    configs=grid(thresholds)
    by_id={c["id"]:c for c in configs}
    if len(configs)!=len(by_id) or len(configs)!=5+4*190*len(thresholds):
        raise ValueError("Invalid unique validation grid size")
    old_rows=csv_rows(old/"report"/"validation.csv")
    if len(old_rows)!=3425*3:
        raise ValueError("Expected complete previous validation grid")
    records=restore(old_rows,by_id)
    available={row["id"] for row in records}
    todo=[c for c in configs if c["id"] not in available]
    records.extend(eval_configs(old,root,todo,"valid",read(old/"catalog.json")))
    numbers={(r["id"],r["stratum"]):r for r in records}
    if len(records)!=len(numbers) or len(numbers)!=len(configs)*3:
        raise ValueError("Incomplete validation outcome table")
    sweep.write_csv(report/"validation.csv",[numbers[c["id"],s] for c in configs for s in STRATA])
    combo=defaultdict(list)
    for c in configs:
        combo[c["k"],c["arm"],c["quota"],c["prefix_beams"]].append(c)
    lock=[best(group,numbers) for group in combo.values()]
    if len(lock)!=765:
        raise ValueError("Expected 765 threshold-frozen test configurations")
    aw=per_arm(lock,numbers)
    original_cfg=[c for c in configs if c["threshold"] in original]
    fixed=per_arm(original_cfg,numbers)
    legacy=previous["winners"]
    winners={str(k):best([by_id[idx] for idx in aw[str(k)].values()],numbers)["id"] for k in KS}
    supplementary={c["id"]:c for c in lock}
    detail=set(winners.values())|set(legacy.values())
    for k in KS:
        detail.update(aw[str(k)].values())
        detail.update(fixed[str(k)].values())
        choices=sweep.alpha_groups(k)[k]
        for width in WIDTHS:
            ident=cfg_for("unisrec1",k,k,width,None,choices)["id"]
            supplementary[ident]=by_id[ident]
        for arm in ("unisrec1","unisrec2"):
            detail.add(cfg_for(arm,k,k,1,None,choices)["id"])
    for ident in detail:
        supplementary[ident]=by_id[ident]
    controlled=[]
    for c in original_cfg:
        if c["arm"]!="unisrec1":
            continue
        peer=cfg_for("unisrec2",c["k"],c["quota"],c["prefix_beams"],c["threshold"],c["alpha_aliases"])
        row={k:c[k] for k in ("k","quota","alpha","alpha_aliases","prefix_beams","threshold")}
        row.update(unisrec1_id=c["id"],unisrec2_id=peer["id"])
        for stratum in STRATA:
            for field in ("item_hr","item_ndcg","mean_verified","alpha_eff","mean_rounds"):
                a,b=(numbers[ident,stratum][field] for ident in (c["id"],peer["id"]))
                row.update({f"{stratum}_rho1_{field}":a,f"{stratum}_rho2_{field}":b,
                            f"{stratum}_rho1_minus_rho2_{field}":a-b})
        controlled.append(row)
    sweep.write_csv(report/"controlled.csv",controlled)
    benchmark=performance_plan(lock,aw,legacy,by_id,numbers)
    selection=dict(primary="validation exact item NDCG@K",tie_break=previous["tie_break"],seed=42,
        validation_configurations=len(configs),test_configurations=len(lock),
        evaluated_test_configurations=len(supplementary),historical_validation_rows_reused=3425,
        controlled_validation_pairs=len(controlled),catalog_sha256=previous["catalog_sha256"],
        valid_inputs=valid_hashes,legacy_validation_csv_sha256=sweep.sha256(old/"report"/"validation.csv"),
        calibration={"old":old_calibration,"new":new_calibration},thresholds={"old":original,
        "new":[None,*new_calibration["values"]],"union":thresholds},arm_order=list(ARMS),
        winners=winners,arm_winners=aw,old_grid_winners=fixed,legacy_winners=legacy,locked=lock,
        test_additional=[c for i,c in supplementary.items() if i not in {x["id"] for x in lock}],
        prediction_detail_ids=sorted(detail))
    sweep.write_json(report/"selection.json",selection)
    sweep.write_json(report/"benchmark_configs.json",benchmark)
    print("Selection complete",len(configs),len(benchmark),winners,flush=True)


def export_alpha(report, selection, val, test):
    rows=[]
    for k in KS:
        for q,aliases in sweep.alpha_groups(k).items():
            for arm in (ARMS[1:] if q else ("beam",)):
                cfg=best([c for c in selection["locked"] if c["k"]==k and c["quota"]==q and c["arm"]==arm],val)
                record={**cfg,"retrieval_quota":q,"beam_quota":k-q}
                for split,data in (("validation",val),("test",test)):
                    for st in ("overall","cold"):
                        for field in ("item_hr","item_ndcg","alpha_eff"):
                            record[split+("_cold" if st=="cold" else "")+"_"+field]=float(data[cfg["id"],st][field])
                rows.extend({**record,"alpha":a} for a in aliases)
    sweep.write_csv(report/"alpha-comparison.csv",rows)


def audit_arms(root, old, selection, configs, test, catalog):
    report=root/"report"
    by_id={c["id"]:c for c in configs}
    observed={}
    events={}
    direct={}
    champion_ids=set(selection["winners"].values())
    dest=report/"predictions"/"validation_winners.jsonl.gz"
    all_detail=selection["prediction_detail_ids"]
    with gzip.open(dest.with_suffix(".gz.tmp"),"wt",compresslevel=2) as out:
        for arm in ARMS:
            expected_details={i for i in all_detail if by_id[i]["arm"]==arm}
            if not expected_details:
                continue
            path=report/"predictions"/arm/"validation_winners.jsonl.gz"
            for row in sweep.stream_rows([path]):
                ident=row["config_id"]
                if ident not in expected_details:
                    raise ValueError("Unexpected detailed prediction")
                if ident in champion_ids:
                    out.write(json.dumps(row,separators=(",",":"))+"\n")
            observed.update(diagnostics.saved_outcomes(path,{i:i for i in expected_details}))
    dest.with_suffix(".gz.tmp").replace(dest)
    for arm in ARMS:
        ids=[i for i in all_detail if by_id[i]["arm"]==arm]
        if not ids:
            continue
        accum={(i,s):Counter() for i in ids for s in STRATA}
        source=root if arm=="unisrec1" else old
        for row in sweep.stream_rows(sweep.files_for(source,"test")):
            rid,target=row["row_id"],row["target"]
            subset="cold" if catalog.cold[target] else "warm"
            for ident in ids:
                cfg=by_id[ident]
                actual=observed[ident][rid]
                if actual.target!=target:
                    raise ValueError("Detailed prediction target mismatch")
                funnel=diagnostics.trace_target(row,cfg,actual,catalog)
                raw=(row["drafts"][cfg["drafter"]][str(cfg["k"])]["first"][:cfg["k"]] if cfg["quota"] else
                     [catalog.aliases[s][0] for s in row["beam"][:cfg["k"]]])
                funnel["raw_topk_hits"]=target in raw
                for s in ("overall",subset):
                    accum[ident,s].update(funnel)
        for ident in ids:
            events[ident]={s:dict(accum[ident,s]) for s in STRATA}
            for s in STRATA:
                if events[ident][s]["final_hits"]!=test[ident,s]["item_hits"]:
                    raise ValueError("Funnel hits disagree with test table")
    sweep.write_json(report/"funnel.json",dict(by_config=events,source="actual executed rounds and final ranks",
                                               metric="exact_item_id"))
    cold=np.array([catalog.cold[observed[next(iter(observed))][i].target] for i in range(TEST_ROWS)])
    if cold.sum()!=147:
        raise ValueError("Cold target count must be 147")
    masks={"overall":np.ones(TEST_ROWS,dtype=bool),"warm":~cold,"cold":cold}
    params={}
    for k in KS:
        ks=str(k)
        arms=selection["arm_winners"][ks]
        ids={"unisrec1_minus_unisrec2":(arms["unisrec1"],arms["unisrec2"]),
             "winner_minus_legacy":(selection["winners"][ks],selection["legacy_winners"][ks]),
             "winner_minus_beam":(selection["winners"][ks],arms["beam"])}
        params[ks]={}
        for label,(new,base) in ids.items():
            x,y=[np.array([observed[i][r].rank for r in range(TEST_ROWS)]) for i in (base,new)]
            entry={"baseline":base,"candidate":new}
            for j,(s,mask) in enumerate(masks.items()):
                entry[s]=paired.summarize(x[mask],y[mask],np.random.default_rng(np.random.SeedSequence([42,k,j])))
            params[ks][label]=entry
    sweep.write_json(report/"paired.json",dict(by_k=params,seed=42,replicates=10000,
        method="Paired percentile bootstrap over requests, stratified summaries, unadjusted for multiple K",
        unit="percentage points"))
    # Independently check membership for every request and every width, including cold ones.
    shard_dir=report/"predictions"/"unisrec1"
    direct_top={row["row_id"]:{k:row["drafts"]["unisrec"][str(k)]["first"] for k in KS}
                for row in sweep.stream_rows(sweep.files_for(root,"test"))}
    for k,expected,c_expected in zip(KS,DIRECT_HITS,DIRECT_COLD_HITS,strict=True):
        aliases=sweep.alpha_groups(k)[k]
        for width in WIDTHS:
            ident=cfg_for("unisrec1",k,k,width,None,aliases)["id"]
            if (test[ident,"overall"]["item_hits"],test[ident,"cold"]["item_hits"])!=(expected,c_expected):
                raise ValueError(f"Direct-UniSRec hit count failed: {ident}")
            if test[ident,"overall"]["beam_call_rate"] or test[ident,"overall"]["mean_rounds"]!=1:
                raise ValueError(f"Unexpected extra round or beam call for {ident}")
        direct[str(k)]=dict(overall_hits=expected,cold_hits=c_expected,rows=TEST_ROWS,widths_checked=list(WIDTHS))
    covered=set()
    for file in sorted(shard_dir.glob("batch-*.npz")):
        with np.load(file,allow_pickle=False) as data:
            rid=list(map(int,data["row_ids"]))
            if covered.intersection(rid):
                raise ValueError("Duplicate direct audit row")
            covered.update(rid)
            for k in KS:
                raw=np.sort([direct_top[i][k] for i in rid],axis=1)
                for width in WIDTHS:
                    ident=cfg_for("unisrec1",k,k,width,None,sweep.alpha_groups(k)[k])["id"]
                    if not np.array_equal(np.sort(data[ident+"_item"],axis=1),raw):
                        raise ValueError(f"Direct top-K membership failed: {ident}")
    if covered!=set(range(TEST_ROWS)):
        raise ValueError("Incomplete direct membership audit")
    return direct


def quality(old, root):
    report=root/"report"
    sel=read(report/"selection.json")
    # Freeze the predeclared all-accept controls before consuming test predictions.
    # This also supports a selection prepared by the first version of this runner.
    extra={c["id"]:c for c in sel["test_additional"]}
    locked_ids={c["id"] for c in sel["locked"]}
    detail=set(sel["prediction_detail_ids"])
    for k in KS:
        for arm in ("unisrec1","unisrec2"):
            c=cfg_for(arm,k,k,1,None,sweep.alpha_groups(k)[k])
            detail.add(c["id"])
            if c["id"] not in locked_ids:
                extra[c["id"]]=c
    if detail!=set(sel["prediction_detail_ids"]) or len(extra)!=len(sel["test_additional"]):
        sel["test_additional"]=list(extra.values())
        sel["prediction_detail_ids"]=sorted(detail)
        sel["evaluated_test_configurations"]=len(locked_ids)+len(extra)
        sweep.write_json(report/"selection.json",sel)
    if sel["valid_inputs"]!={"old":inputs(old,"valid"),"new":inputs(root,"valid")}:
        raise ValueError("Validation data changed after freeze")
    record=read(old/"catalog.json")
    if sweep.sha256(old/"catalog.json")!=sel["catalog_sha256"]:
        raise ValueError("Catalog changed after freeze")
    configs=sel["locked"]+sel["test_additional"]
    detail=sel["prediction_detail_ids"]
    rows=eval_configs(old,root,configs,"test",record,detail)
    numbers={(r["id"],r["stratum"]):r for r in rows}
    if len(numbers)!=len(configs)*3 or any(r["rows"]!=(147 if r["stratum"]=="cold" else
            4386 if r["stratum"]=="warm" else TEST_ROWS) for r in rows):
        raise ValueError("Incomplete frozen test grid")
    ordered=[numbers[c["id"],s] for c in configs for s in STRATA]
    sweep.write_csv(report/"test.csv",ordered)
    historical=csv_rows(old/"report"/"test.csv")
    checked=0
    for r in historical:
        key=r["id"],r["stratum"]
        if key in numbers:
            checked+=1
            actual=numbers[key]
            for field in ("item_hits","item_ndcg","sid_hits","sid_ndcg","alpha_eff","mean_verified","mean_rounds"):
                if not np.isclose(float(r[field]),float(actual[field]),rtol=0,atol=1e-13):
                    raise ValueError(f"Historical strict regression {key} {field}")
    if not checked:
        raise ValueError("No shared frozen reference settings")
    val={(r["id"],r["stratum"]):r for r in csv_rows(report/"validation.csv")}
    # Validation values read from CSV need numeric fields for identical ranking keys.
    for r in val.values():
        for field in ("item_ndcg","item_hr","mean_verified"):
            r[field]=float(r[field])
    export_alpha(report,sel,val,numbers)
    direct=audit_arms(root,old,sel,configs,numbers,sweep.Catalog.from_dict(record))
    sweep.write_json(report/"quality.json",dict(status="passed",test_rows=TEST_ROWS,validation_rows=VALID_ROWS,
      locked_configs=len(sel["locked"]),test_configs_with_supplementary=len(configs),historical_test_rows_reproduced=checked,
      reference_rankings_verified=TEST_ROWS,direct_uni_k_topk=direct,validation_selection_sha256=sweep.sha256(report/"selection.json"),
      test_inputs={"old":inputs(old,"test"),"new":inputs(root,"test")},winners=sel["winners"],
      prediction_layout="predictions/<arm>/batch-*.npz, predictions/validation_winners.jsonl.gz",
      detail_config_ids=detail))
    print("Frozen test grid, direct top-K membership and paired rankings verified",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old",type=Path,default=Path("/run/hybrid"))
    parser.add_argument("--run-dir",type=Path,default=Path("/run/hybrid-rho1"))
    parser.add_argument("stage",choices=("select","quality"))
    args=parser.parse_args()
    (select if args.stage=="select" else quality)(args.old.resolve(),args.run_dir.resolve())


if __name__=="__main__":
    main()
