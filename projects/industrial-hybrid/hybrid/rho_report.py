"""Render the paired UniSRec K-versus-2K experiment using frozen validation choices."""
from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from .report import batch_audit_table, configure_plots, pct, read_csv, table
from .sweep import KS, alpha_groups, write_csv, write_json


ARMS = ("beam", "ann", "qwen", "unisrec2", "unisrec1")
NAMES = {"beam": "纯 beam", "ann": "向量 HNSW", "qwen": "Qwen 检索器",
         "unisrec1": "UniSRec-K", "unisrec2": "UniSRec-2K"}
GROUPS = {"overall": "整体", "warm": "暖商品", "cold": "冷商品"}
COLORS = {"beam": "#69757d", "ann": "#92594b", "qwen": "#3165b0",
          "unisrec1": "#15826d", "unisrec2": "#714fb1"}


def load(path):
    return json.loads(path.read_text())


def metric_rows(path):
    result = {}
    for row in read_csv(path):
        for key in ("k", "quota", "rho", "prefix_beams", "rows", "item_hits"):
            row[key] = int(row[key])
        for key in ("alpha", "item_hr", "item_ndcg", "alpha_eff", "mean_verified",
                    "mean_rounds", "below_retrieval_quota_rate", "beam_call_rate"):
            row[key] = float(row[key])
        row["threshold"] = float(row["threshold"]) if row["threshold"] else None
        row["alpha_aliases"] = json.loads(row["alpha_aliases"])
        key = row["id"], row["stratum"]
        if key in result:
            raise ValueError(f"Duplicate {path.name} metric {key}")
        result[key] = row
    return result


def objective(row):
    return -row["item_ndcg"], -row["item_hr"], row["mean_verified"]


def signed(n):
    return f"{n:+.3f}"


def change(score, base):
    return signed(100 * (score - base))


def threshold(t):
    return "全接受" if t is None else f"{float(t):.4f}"


def choice_alpha(row):
    return "、".join(f"{a:g}" for a in row["alpha_aliases"])


def route(row):
    if row["arm"] == "beam":
        return "纯 beam50"
    return (f"{NAMES[row['arm']]}，α={choice_alpha(row)}，配额 {row['quota']}/{row['k']}，"
            f"前缀宽 {row['prefix_beams']}，阈值{threshold(row['threshold'])}")


def ci(group, name):
    low, high = group[f"{name}_ci95_pp"]
    return f"{signed(group[f'{name}_delta_pp'])} [{signed(low)}, {signed(high)}]"


class Results:
    def __init__(self, run_dir, preview):
        self.report = Path(run_dir) / "report"
        self.selection = load(self.report / "selection.json")
        self.val = metric_rows(self.report / "validation.csv")
        self.test = metric_rows(self.report / "test.csv")
        self.funnel = load(self.report / "funnel.json")["by_config"]
        self.paired = load(self.report / "paired.json")["by_k"]
        self.quality = load(self.report / "quality.json")
        if (self.quality.get("status") != "passed" or self.quality["winners"] != self.selection["winners"]
                or self.quality["locked_configs"] != len(self.selection["locked"])):
            raise ValueError("Report requires a complete test evaluation of the frozen validation winners")
        attr_path = self.report / "rank-attribution.json"
        self.attribution = load(attr_path)["by_k"] if attr_path.exists() or not preview else {}
        self.chosen = self.selection["winners"]
        self.arm_winners = self.selection["arm_winners"]
        self.legacy = self.selection["legacy_winners"]
        self.overall = [r for (i, group), r in self.val.items() if group == "overall"]
        self.grid = defaultdict(list)
        for row in self.overall:
            self.grid[row["k"], row["quota"], row["arm"]].append(row)
        self.best_quota = {key: min(group, key=objective) for key,group in self.grid.items()}
        self.timings = {}
        for name in ["benchmarks.json", "benchmark_protocol.json", "batch_audit.json"]:
            if not preview and not (self.report / name).is_file():
                raise ValueError(f"Final report requires {name}")
        if (self.report / "benchmarks.json").is_file():
            timed = load(self.report / "benchmarks.json")
            self.timings = {b["id"]:b for b in timed}
            planned = load(self.report / "benchmark_configs.json")
            if (len(self.timings) != len(timed) or set(self.timings) != {c["id"] for c in planned}
                    or any(not all(math.isfinite(float(b[m])) and b[m] > 0 for m in
                                   ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                                    "requests_s_1gpu", "requests_s_4gpu")) for b in timed)):
                raise ValueError("Live timing must cover every planned configuration with finite measurements")
        for k in KS:
            for name in ("unisrec1", "unisrec2", "beam"):
                self.test[self.arm_winners[str(k)][name], "overall"]
            if not preview:
                for i in {self.chosen[str(k)], self.arm_winners[str(k)]["unisrec1"],
                          self.arm_winners[str(k)]["unisrec2"], self.legacy[str(k)]}:
                    self.timings[i]
        self.preview = preview

    def arms(self,k):
        win = self.arm_winners[str(k)]
        return win["unisrec1"], win["unisrec2"]

    def timing(self, ident, metric, digits=2):
        if ident not in self.timings:
            return "待测"
        b = self.timings[ident]
        if metric == "drafted":
            number = b["retrieval_counters_1gpu"].get("drafted", 0) / b["logical_requests_1gpu"]
        else:
            number = b[metric]
        return f"{number:.{digits}f}"


def alpha_section(r):
    sections = ["<h2 id='alpha'>融合比例 α 与验证效果</h2>",
        "<p>检索目标为 ⌊αK⌋ 件，配额不足时由完整 beam 顺延补齐。α_eff 是每条请求最终从检索留下的件数除以 K 后的平均值。"
        "横轴是目标 α，每个非零配额和每种草拟器独立在验证集选择前缀宽度及阈值；右图用相同配置评估测试集。"
        "K=20 与 50 除十分位网格外分别含 0.05、0.02。等价配额共同使用一条排名。</p>"]
    thresholds_old = set(r.selection["calibration"]["old"]["values"]) | {None}
    for k in (10, 1, 5, 20, 50):
        plt = configure_plots(r.report)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.55), layout="constrained")
        comparisons, control, peak_text = [], [], []
        quotas = alpha_groups(k)
        for arm in ARMS[1:]:
            vals, tests, alphas = [], [], []
            for q,aliases in quotas.items():
                v = r.best_quota[k,q,arm if q else "beam"]
                t = r.test[v["id"],"overall"]
                vals += [v["item_ndcg"]*100]*len(aliases)
                tests += [t["item_ndcg"]*100]*len(aliases)
                alphas += aliases
                if q or arm == ARMS[1]:
                    comparisons.append((q, [choice_alpha(v), NAMES[v["arm"]], q,
                        v["prefix_beams"] if q else "—", threshold(v["threshold"]) if q else "—",
                        pct(v["alpha_eff"]), pct(v["item_hr"]), pct(v["item_ndcg"]),
                        pct(t["alpha_eff"]), pct(t["item_hr"]), pct(t["item_ndcg"])]))
            label = {"ann":"Vector HNSW", "qwen":"Qwen", "unisrec1":"UniSRec-K", "unisrec2":"UniSRec-2K"}[arm]
            for axis,scores in zip(ax,(vals,tests)):
                axis.plot(alphas,scores,label=label,color=COLORS[arm],marker="o",markersize=3,linewidth=1.7)
            top = max(vals)
            peaks = "、".join(f"{a:g}" for a, val in zip(alphas,vals) if val==top)
            peak_text.append(f"{NAMES[arm]} {peaks}（{top:.3f}%）")
        beam = r.best_quota[k,0,"beam"]
        for axis,b,title in ((ax[0],beam,"Validation"),(ax[1],r.test[beam["id"],"overall"],"Test")):
            axis.axhline(100*b["item_ndcg"],color=COLORS["beam"],ls="--",lw=1,label="Beam")
            axis.set_xticks(sorted({a for aliases in quotas.values() for a in aliases}))
            axis.tick_params(axis="x",labelsize=8,rotation=40 if k==50 else 0)
            axis.set_xlabel("Target retrieval fraction " + chr(945))
            axis.set_title(title)
            axis.grid(axis="y",alpha=.18)
        ax[0].set_ylabel(f"Item NDCG@{k} (%)")
        ax[1].legend(loc="best",fontsize=8)
        name=f"alpha-k{k}.png"
        fig.savefig(r.report/name,dpi=140)
        plt.close(fig)
        for q,aliases in quotas.items():
            pair=[]
            for arm in ("unisrec1","unisrec2"):
                fixed = (next(row for row in r.grid[k,q,arm] if row["prefix_beams"]==5 and row["threshold"] is None)
                         if q else beam)
                pair.append(fixed)
            a,b=pair
            control.append([", ".join(f"{x:g}" for x in aliases), q, pct(a["item_hr"]), pct(b["item_hr"]),
                            pct(a["item_ndcg"]), pct(b["item_ndcg"]), change(a["item_ndcg"],b["item_ndcg"])])
        sections += [f"<details{' open' if k==10 else ''}><summary>K={k} 的 α 曲线与逐点结果</summary>"
                     f"<img src='{name}' alt='K={k} 四种检索方案各目标比例的验证与测试 NDCG'>",
                     "<p>各方法验证 NDCG 最高时的目标 α 为 " + "，".join(peak_text)+"。</p>",
                     "<details><summary>各 α 的 HR、NDCG 与实际检索占比</summary>",
                     table(["目标 α","方案","检索配额","前缀宽","阈值","验证 α_eff","验证 HR","验证 NDCG",
                            "测试 α_eff","测试 HR","测试 NDCG"],[x[1] for x in sorted(comparisons,key=lambda x:x[0])]),
                     "</details><details><summary>固定前缀宽度 5、全接受时的候选数对照</summary>"
                     "<p>下表使用验证集，固定同一列目标配额、阈值与前缀宽度，仅改变每轮候选数。</p>",
                     table(["目标 α","检索配额","K 候选 HR","2K 候选 HR","K 候选 NDCG","2K 候选 NDCG","差值 pp"],control),
                     "</details></details>"]
    keys = lambda row:(row["k"],row["quota"],row["prefix_beams"],row["threshold"])
    fixed={arm:{keys(v):v for v in r.overall if v["arm"]==arm and v["threshold"] in thresholds_old}
           for arm in ("unisrec1","unisrec2")}
    one,two=fixed["unisrec1"],fixed["unisrec2"]
    if one.keys()!=two.keys() or len(one)!=1140:
        raise ValueError("Incomplete matched original-grid UniSRec controls")
    ds=[one[key]["item_ndcg"]-two[key]["item_ndcg"] for key in one]
    equal=sum(abs(x)<=1e-12 for x in ds)
    plus=sum(x>1e-12 for x in ds)
    sections.append(f"<p>原阈值网格下有 {len(ds):,} 对同参数配置。每轮改取 K 后，验证 NDCG 上升 {plus} 对，"
                    f"持平 {equal} 对，下降 {len(ds)-equal-plus} 对。各配置使用相同请求，计数描述配置分布。"
                    "<a href='controlled.csv'>全部固定参数对照</a> · "
                    "<a href='alpha-comparison.csv'>目标 α、实际 α 和冷商品指标</a></p>")
    return "".join(sections)


def paired_section(r):
    rows=[]
    for k in KS:
        for comp,label in (("unisrec1_minus_unisrec2","UniSRec-K 较 UniSRec-2K"),
                           ("winner_minus_legacy","本次冠军较上版冠军"),("winner_minus_beam","本次冠军较纯 beam")):
            for stratum,cn in GROUPS.items():
                g=r.paired[str(k)][comp][stratum]
                if comp=="unisrec1_minus_unisrec2":
                    a,b=r.arms(k)
                    if [g["candidate_hits"],g["baseline_hits"]]!=[r.test[i,stratum]["item_hits"] for i in (a,b)]:
                        raise ValueError(f"Paired test hits differ at K={k} {stratum}")
                rows.append([k,label,cn,g["rows"],g["gained_hits"],g["lost_hits"],g["retained_hits"],
                             ci(g,"hr"),ci(g,"ndcg")])
    ten = r.paired["10"]["unisrec1_minus_unisrec2"]
    cold=ten["cold"]
    lo,hi=cold["ndcg_ci95_pp"]
    if lo<=0<=hi:
        inference="该冷商品区间覆盖零，本次样本支持的变化方向仍有不确定性。"
    else:
        inference="这 147 条冷请求的配对区间位于零的同一侧。"
    return ("<h2 id='paired'>同一请求上的增益与不确定性</h2>"
            f"<p>K=10 两个 UniSRec 验证冠军逐条比较，每轮 K 的配置在整体上新增 {ten['overall']['gained_hits']} 次命中，"
            f"丢失 {ten['overall']['lost_hits']} 次。暖目标净变化 {ten['warm']['gained_hits']-ten['warm']['lost_hits']:+d} 次，"
            f"冷目标净变化 {cold['gained_hits']-cold['lost_hits']:+d} 次。冷商品 NDCG 的差值与 95% 区间为 "
            f"{ci(cold,'ndcg')} 个百分点。{inference}</p>"
            "<p>每个 K、每个分组对相同测试行执行 10,000 次有放回成对重采样。"
            "表内差值均为前一方案减去后一方案，单位为百分点。每个冷目标命中约对应 0.680 个百分点的冷 HR。"
            "这些区间未经多重比较校正；当两组逐行结果相同，经验区间为 [0, 0]。</p>"
            "<details><summary>全部 K 的整体、暖商品、冷商品配对结果</summary>"+
            table(["K","比较","分组","请求数","新增命中","丢失命中","均命中","ΔHR [95%]","ΔNDCG [95%]"],rows)+
            "</details><p><a href='paired.json'>成对变化和区间数据</a></p>")


def funnel_section(r):
    rows=[]
    summaries=[]
    for k in KS:
        one,two=r.arms(k)
        for ident in (one,two):
            for key,cn in GROUPS.items():
                f=r.funnel[ident][key]
                if f["first_retrieved"]+f["later_retrieved"]+f["beam_hits"]!=r.test[ident,key]["item_hits"]:
                    raise ValueError(f"Funnel losses at {ident} {key}")
                rows.append([k,NAMES[r.val[ident,"overall"]["arm"]],cn,f["rows"],f["first_drafted"],
                             f["later_drafted"],f["any_passed"],f["target_rejected_at_threshold"],
                             f["target_passed_outside_retrieval_quota"],f["retrieved"],f["beam_hits"],f["final_hits"]])
        a,b=r.funnel[one]["overall"],r.funnel[two]["overall"]
        ac,bc=r.funnel[one]["cold"],r.funnel[two]["cold"]
        summaries.append([k,f"{b['first_drafted']} → {a['first_drafted']}",f"{b['any_passed']} → {a['any_passed']}",
                          f"{b['retrieved']} → {a['retrieved']}",f"{b['beam_hits']} → {a['beam_hits']}",
                          f"{bc['any_drafted']} → {ac['any_drafted']}",f"{bc['final_hits']} → {ac['final_hits']}"])
    one,two=r.arms(10)
    a,b=r.funnel[one]["cold"],r.funnel[two]["cold"]
    wide_one,wide_two=r.arms(50)
    small,large=r.funnel[wide_one]["cold"],r.funnel[wide_two]["cold"]
    return ("<h2 id='coverage'>冷目标在哪一步留下或丢失</h2>"
            "<p>下表的箭头从 UniSRec-2K 验证冠军指向 UniSRec-K 验证冠军。每个数字都是包含正确目标的请求数。"
            "检索入选和 beam 命中相加等于最终命中；beam 可以补进未出现在草拟候选中的商品。</p>"+
            table(["K","首轮覆盖","过阈值（全部轮）","检索入选","beam 命中","冷目标获草拟","冷目标最终命中"],summaries)+
            f"<p>K=10 的 147 条冷请求中，每轮取 K 时，{147-a['any_drafted']} 条的正确目标从未被草拟，"
            f"{a['target_rejected_at_threshold']} 条在草拟后未过阈值，{a['target_passed_outside_retrieval_quota']} 条过阈但未获检索席位。"
            f"最终由检索命中 {a['retrieved']} 条、由 beam 命中 {a['beam_hits']} 条。每轮取 2K 时这两种来源分别贡献 "
            f"{b['retrieved']} 和 {b['beam_hits']} 条。两臂各自按验证集选参，分解包含候选数、阈值与配额的共同影响。</p>"
            f"<p>K=50 时，2K 组在实际草拟轮次中覆盖了 {large['any_drafted']} 个正确冷目标，"
            f"最终命中 {large['final_hits']} 个，其中 {large['target_passed_outside_retrieval_quota']} 个过阈目标未获得检索席位。"
            f"K 组覆盖了 {small['any_drafted']} 个正确冷目标，最终命中 {small['final_hits']} 个。"
            "首轮覆盖量和最终保留量共同决定这一分组的效果。</p>"
            "<details><summary>按 K、候选数和冷暖分组查看目标去向</summary>"+
            table(["K","方案","分组","请求","首轮覆盖","后续覆盖","过阈目标","阈值拒绝","过阈未获席位",
                   "检索命中","beam 命中","最终命中"],rows)+
            "</details><p><a href='funnel.json'>逐阶段计数</a></p>")


def controlled_test(r):
    rows = []
    for idx,k in enumerate(KS):
        for arm in ("unisrec2","unisrec1"):
            cfg = next(v for v in r.grid[k,k,arm] if v["prefix_beams"]==1 and v["threshold"] is None)
            t = r.test[cfg["id"],"overall"]
            c = r.test[cfg["id"],"cold"]
            if arm=="unisrec1" and (t["item_hits"] != (169,455,638,853,1177)[idx] or
                                      c["item_hits"] != (1,6,9,10,15)[idx]):
                raise ValueError(f"Direct top-K membership regression at K={k}")
            rows.append([k, NAMES[arm], f"{t['mean_verified']:.2f}", t["item_hits"],
                         pct(t["item_hr"]), pct(t["item_ndcg"]), c["item_hits"],
                         pct(c["item_hr"]), pct(c["item_ndcg"])])
    ten=[row for row in rows if row[0]==10]
    old,new=ten
    return ("<h2 id='controls'>只改候选数时会发生什么</h2>"
            "<p>这里统一取 α=1、前缀宽 1、全接受。两组都在首轮结束。取 2K 时，验证器既决定哪些商品入选，也决定它们的顺序；"
            "改取 K 时，首轮每件候选均获一个席位，验证器只改变顺序。该配置可以单独检查扩大候选池的作用。</p>"
            +table(["K","方案","平均验证候选数","整体命中","整体 HR","整体 NDCG","冷命中/147","冷 HR","冷 NDCG"],rows)
            +f"<p>K=10 按这个固定设置，每轮取 K 的整体命中是 {new[3]} 条，每轮取 2K 是 {old[3]} 条；"
             f"冷目标命中分别为 {new[6]} 和 {old[6]} 条。直接 UniSRec 的前 K 集合在 K 候选版本中完整保留，"
             "因此其冷、暖命中数均可逐行回核。NDCG 还取决于命中目标被排到第几位。</p>")


def rank_attribution(r):
    if not r.attribution:
        return ""
    data=r.attribution
    text=[]
    rows=[]
    components={"gained_ndcg_pp":"新增命中","lost_ndcg_pp":"丢失命中","shared_ndcg_pp":"共同命中的名次变化"}
    for k in KS:
        key=str(k)
        for comp,label in (("fixed_all_accept","固定 α=1、全接受"),("tuned_champions","各臂验证冠军")):
            event=data[key][comp]
            if comp=="tuned_champions" and (event["rho1_id"],event["rho2_id"])!=r.arms(k):
                raise ValueError(f"Contribution analysis uses incorrect winner IDs at K={k}")
            for stratum,cn in GROUPS.items():
                g=event["groups"][stratum]
                total=g["total_ndcg_pp"]
                if not math.isclose(sum(g[field] for field in components),total,abs_tol=1e-7):
                    raise ValueError(f"NDCG contributions must add to the observed change at K={k}, {comp}, {cn}")
                ref,new=(r.test[event[which],stratum] for which in ("rho2_id","rho1_id"))
                if not math.isclose(total,100*(new["item_ndcg"]-ref["item_ndcg"]),abs_tol=1e-7):
                    raise ValueError(f"Contribution total differs from the frozen test metrics at K={k}")
                rows.append([k,label,cn,g["gained_hits"],g["lost_hits"],g["shared_up"],g["shared_same"],g["shared_down"],
                             *[signed(g[f]) for f in components],signed(total)])
        if k in (10,20,50):
            g=data[key]["tuned_champions"]["groups"]["overall"]
            sign=g["total_ndcg_pp"]
            exclusive=g["gained_ndcg_pp"]+g["lost_ndcg_pp"]
            if exclusive<0<sign:
                explanation=(f"两组各自独有的命中合计 {signed(exclusive)} 个百分点。共同命中的名次提升抵消了这项损失，"
                             "使总差转为正值。")
            elif exclusive<sign<0:
                explanation=(f"两组各自独有的命中合计 {signed(exclusive)} 个百分点。共同命中的名次改善仅抵消了部分损失。")
            else:
                explanation=f"两组各自独有的命中合计贡献 {signed(exclusive)} 个百分点。"
            interval=r.paired[key]["unisrec1_minus_unisrec2"]["overall"]["ndcg_ci95_pp"]
            assessment=("该区间跨过零，测试样本中观察到的差异仍与零增益相容。" if interval[0]<=0<=interval[1]
                        else "本组配对区间位于零以上。" if interval[0]>0 else "本组配对区间位于零以下。")
            text.append(f"<p><strong>K={k}。</strong> 相对 2K 组，K 组新增 {g['gained_hits']} 次命中，"
                        f"丢失 {g['lost_hits']} 次，净变化 {g['gained_hits']-g['lost_hits']:+d} 次。两组共同命中的请求中，"
                        f"{g['shared_up']} 个目标提前、{g['shared_down']} 个目标后移。新增、丢失和共同命中的名次变化"
                        f"分别贡献 {signed(g['gained_ndcg_pp'])}、{signed(g['lost_ndcg_pp'])}、"
                        f"{signed(g['shared_ndcg_pp'])} 个 NDCG 百分点。{explanation}"
                        f"合计 {signed(sign)} 个百分点，95% 配对区间 [{signed(interval[0])}, {signed(interval[1])}]。{assessment}</p>")
    return ("<h3>名次变化怎样影响 NDCG</h3><p>对相同测试请求，将 UniSRec-K 较 UniSRec-2K 的 NDCG 差拆成三部分。"
            "新增和丢失命中按其实际名次折扣，共同命中按两种名次的折扣差计入。三项除以同一分组的请求数后可直接相加。</p>"
            +"".join(text)+"<details><summary>固定参数与各自调优两种口径的完整排名贡献</summary>"
            +table(["K","比较","分组","新增命中","丢失命中","共同命中提前","名次相同","共同命中后移",
                    "新增贡献 pp","丢失贡献 pp","共同命中名次贡献 pp","总差 pp"],rows)+
            "</details><p><a href='rank-attribution.json'>排名贡献原始统计</a></p>")


def arm_comparisons(r):
    rows=[]
    for k in KS:
        for ident in r.arms(k):
            v=r.val[ident,"overall"]
            t=r.test[ident,"overall"]
            w,c=r.test[ident,"warm"],r.test[ident,"cold"]
            rows.append([k,route(v),pct(v["item_ndcg"]),pct(t["item_hr"]),pct(t["item_ndcg"]),
                         f"{w['item_hits']}/{w['rows']}",pct(w["item_ndcg"]),
                         f"{c['item_hits']}/{c['rows']}",pct(c["item_ndcg"]),
                         f"{t['mean_verified']:.2f}",f"{t['mean_rounds']:.2f}",pct(t["beam_call_rate"])])
    a,b=r.arms(10)
    va,vb=[r.val[i,"overall"] for i in (a,b)]
    ta,tb=[r.test[i,"overall"] for i in (a,b)]
    return ("<h2 id='arms'>分别调优后的 UniSRec 冠军</h2>"
            "<p>先在每个候选上限内独立确定最优比例、前缀宽度和阈值，再在相同测试请求上比对。"
            "暖、冷分组按目标商品是否出现在训练历史或训练目标中划分。</p>"+
            table(["K","验证选出的参数","验证 NDCG","测试 HR","测试 NDCG","暖命中/样本","暖 NDCG","冷命中/样本",
                   "冷 NDCG","实际草拟数/请求","草拟轮数/请求","完整 beam 调用率"],rows)+
            f"<p>K=10 的 UniSRec-K 相对 UniSRec-2K，验证 NDCG 差 {change(va['item_ndcg'],vb['item_ndcg'])} 个百分点，"
            f"测试 NDCG 差 {change(ta['item_ndcg'],tb['item_ndcg'])} 个百分点。实际每条请求平均验证 "
            f"{ta['mean_verified']:.2f} 与 {tb['mean_verified']:.2f} 件。平均数包含提前退出和实际执行的后续轮次。</p>")


def performance(r):
    if not r.timings:
        return "<h2 id='performance'>实测延迟与吞吐</h2><p>质量结果已生成，性能表将在在线测量完成后写入。</p>"
    protocol=load(r.report/"benchmark_protocol.json")
    compact=[]
    for k in KS:
        for role,ident in (("UniSRec-K",r.arms(k)[0]),("UniSRec-2K",r.arms(k)[1]),
                           ("纯 beam",r.arm_winners[str(k)]["beam"])):
            b=r.timings[ident]
            compact.append([k,role,r.timing(ident,"drafted"),f"{b['latency_p50_ms']:.2f}",f"{b['latency_p95_ms']:.2f}",
                            f"{b['latency_p99_ms']:.2f}",f"{b['requests_s_1gpu']:.2f}",f"{b['requests_s_4gpu']:.2f}"])
    all_rows=[]
    for b in r.timings.values():
        v=r.val[b["id"],"overall"]
        all_rows.append([v["k"],route(v),f"{b['latency_p50_ms']:.2f}",
                         f"{b['requests_s_1gpu']:.2f}",f"{b['requests_s_4gpu']:.2f}",
                         f"{b['items_s_1gpu']:.1f}",f"{b['verifier_input_tokens_s_1gpu']:.1f}",
                         f"{b['gpu_peak_bytes_batch4']/2**30:.2f}"])
    one,two=r.arms(10)
    a,b=r.timings[one],r.timings[two]
    device=html.escape(str(protocol.get("device","NVIDIA H20")))
    plt=configure_plots(r.report)
    fig,axes=plt.subplots(1,len(KS),figsize=(13,3.2))
    fig.subplots_adjust(left=.055,right=.99,top=.76,bottom=.20,wspace=.33)
    labels={"beam":"Beam","ann":"HNSW","qwen":"Qwen","unisrec1":"UniSRec-K","unisrec2":"UniSRec-2K"}
    for k,axis in zip(KS,axes):
        for arm in ARMS:
            cases=[(r.timings[i],v) for (i,s),v in r.test.items()
                   if s=="overall" and v["k"]==k and v["arm"]==arm and i in r.timings]
            axis.scatter([b["latency_p50_ms"] for b,v in cases],
                         [100*v["item_ndcg"] for b,v in cases],
                         color=COLORS[arm],label=labels[arm],s=24,alpha=.85)
        i=r.chosen[str(k)]
        axis.scatter(r.timings[i]["latency_p50_ms"],100*r.test[i,"overall"]["item_ndcg"],
                     facecolors="none",edgecolors="#202e36",linewidths=1.2,s=90)
        axis.set_title(f"K={k}")
        axis.set_xlabel("P50 (ms)")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Test item NDCG (%)")
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="upper center",ncol=len(names),frameon=False)
    fig.savefig(r.report/"quality-latency.png",dpi=150)
    plt.close(fig)
    fastest=[]
    for k in KS:
        samples=[cfg for cfg in r.timings.values() if cfg["k"]==k]
        for field,label,select in (("latency_p50_ms","最低中位时延（ms）",min),
                                   ("requests_s_4gpu","最高四卡吞吐（请求/s）",max)):
            case=select(samples,key=lambda row:row[field])
            fastest.append([k,label,route(r.val[case["id"],"overall"]),f"{case[field]:.2f}",
                            pct(r.test[case["id"],"overall"]["item_ndcg"])])
    return ("<h2 id='performance'>实测延迟与吞吐</h2>"
            f"<p>本轮 {len(r.timings)} 个代表配置在 {device} 上使用同样的 {len(protocol['sample_row_ids'])} 条历史，"
            f"每个配置重复 {protocol['repeats']} 遍。P50/P95/P99 为 batch=1 的单请求延迟，单位毫秒。"
            "吞吐采用每卡 batch=4，单位请求/秒。草拟数为这些计时请求中单卡 batch=4 的实际候选总数除以请求数。"
            "图中黑圈标记各 K 验证冠军。</p><img src='quality-latency.png' alt='五个 K 的测试 NDCG 与端到端时延对照'>"+
            table(["K","方案","实际草拟数/请求","P50 ms","P95 ms","P99 ms","单卡请求/s","四卡请求/s"],compact)+
            f"<p>K=10 两臂各取自己的验证冠军后，单请求 P50 从 2K 组的 {b['latency_p50_ms']:.2f} ms "
            f"变为 K 组的 {a['latency_p50_ms']:.2f} ms；四卡吞吐从 {b['requests_s_4gpu']:.2f} 变为 "
            f"{a['requests_s_4gpu']:.2f} 请求/秒。完整 beam 调用率分别为 "
            f"{pct(r.test[two,'overall']['beam_call_rate'])}、{pct(r.test[one,'overall']['beam_call_rate'])}。"
            "端到端延迟还包括固定的历史编码以及实际触发的验证或完整 beam 解码。</p>"
            "<details><summary>各 K 实测时延和吞吐最优配置</summary>"+
            table(["K","实测项目","配置","数值","测试 NDCG"],fastest)+
            "</details><details><summary>全部计时配置的速度与显存资料</summary>"+
            table(["K","配置","P50 ms","单卡请求/s","四卡请求/s","单卡商品/s","验证器物理输入 token/s","单卡峰值显存 GiB"],
                  sorted(all_rows,key=lambda z:z[0]))+
            "</details><p>物理 token 数计入填充和 beam 分支，完整 JSON 同时提供逻辑历史、草拟 SID 与输出 SID 的 token/s。"
            "<a href='benchmarks.json'>逐配置在线计时和显存原始数据</a> · "
            "<a href='benchmark_protocol.json'>固定请求和重复计时协议</a></p>")


def parameter_reason(r, win):
    if win["arm"]=="beam":
        return "配额为零，按原始 beam50 顺序直接取推荐。"
    k,q,b=win["k"],win["quota"],win["prefix_beams"]
    scope=[row for row in r.grid[k,q,win["arm"]] if row["threshold"]==win["threshold"]]
    widths=sorted(row["prefix_beams"] for row in scope if objective(row)==objective(win))
    observed=r.test[win["id"],"overall"]
    msg=f"目标 α={choice_alpha(win)} 对应 {q} 个检索席位，实测检索占比 {pct(observed['alpha_eff'])}。"
    competing=[v for (size,count,arm),v in r.best_quota.items()
               if size==k and count!=q and arm in (win["arm"],"beam")]
    runner=min(competing,key=objective)
    gap=100*(win["item_ndcg"]-runner["item_ndcg"])
    msg+=(f"该方案的其余配额（含纯 beam 的零配额）中，以 {runner['quota']} 件检索席位的验证结果最好，"
          f"与本次配额的 NDCG 差 {gap:.3f} 个百分点（各配额独立选宽度与阈值）。")
    if observed["mean_rounds"]==1.0 and observed["beam_call_rate"]==0:
        msg += "全部测试请求首轮填满，后续前缀搜索和完整 beam 的调用次数均为零。"
    else:
        msg += (f"每条请求平均草拟 {observed['mean_rounds']:.2f} 轮，"
                f"{pct(observed['below_retrieval_quota_rate'])} 的请求需要把检索缺额转给 beam。")
    if len(widths)>1:
        msg += (f"固定该配额与阈值时，前缀宽度 {', '.join(map(str,widths))} 的三项验证选优指标相同，"
                f"按预设网格顺序记录宽度 {b}。")
    return msg


def champion_section(r):
    rows=[]
    reasons=[]
    records=[]
    params=[]
    for k in KS:
        key=str(k); wid=r.chosen[key]; old_id=r.legacy[key]; legacy=r.val[old_id,"overall"]
        v,t=r.val[wid,"overall"],r.test[wid,"overall"]
        cold,warm=r.test[wid,"cold"],r.test[wid,"warm"]
        old_test=r.test[old_id,"overall"]
        win_arm=v["arm"]
        contenders=[r.val[i,"overall"] for arm,i in r.arm_winners[key].items() if arm!=win_arm]
        second=min(contenders,key=objective)
        if objective(v)>objective(second):
            raise ValueError(f"Winner at K={k} fails validation selection ordering")
        ndcg_edge=100*(v["item_ndcg"]-second["item_ndcg"])
        if ndcg_edge:
            why=f"验证 NDCG 比另一方案中最高的 {NAMES[second['arm']]} 高 {ndcg_edge:.3f} 个百分点。"
        elif v["item_hr"]>second["item_hr"]:
            why=f"与 {NAMES[second['arm']]} 的验证 NDCG 相同，验证 HR 更高。"
        elif v["mean_verified"]<second["mean_verified"]:
            why=f"与 {NAMES[second['arm']]} 的验证 NDCG、HR 相同，平均验证候选数更少。"
        else:
            why=f"与 {NAMES[second['arm']]} 的三项选优指标相同，按预设枚举顺序入选。"
        comparison=r.paired[key]["winner_minus_legacy"]
        conclusion=("本项沿用上版同一配置。" if wid==old_id else
                    f"上版冠军为 {route(legacy)}，本次测试 NDCG 与其相差 "
                    f"{ci(comparison['overall'],'ndcg')} 个百分点（含 95% 成对区间）。"
                    f"暖、冷商品分别净增 {comparison['warm']['candidate_hits']-comparison['warm']['baseline_hits']:+d}、"
                    f"{comparison['cold']['candidate_hits']-comparison['cold']['baseline_hits']:+d} 次命中。")
        if wid!=old_id and r.attribution:
            comp=r.attribution[key]["tuned_champions"]
            if (comp["rho1_id"],comp["rho2_id"])==(wid,old_id):
                g=comp["groups"]["overall"]
                exclusive=g["gained_ndcg_pp"]+g["lost_ndcg_pp"]
                conclusion+=(f"独有命中的折扣分合计贡献 {signed(exclusive)} 个百分点，"
                             f"共同命中的名次变化贡献 {signed(g['shared_ndcg_pp'])} 个百分点。")
        if wid!=old_id and comparison["overall"]["ndcg_ci95_pp"][0]<=0<=comparison["overall"]["ndcg_ci95_pp"][1]:
            conclusion+="测试 NDCG 差的区间覆盖零；此配置由预先指定的验证集标准选出。"
        candidate_size=k*v["rho"] if v["quota"] else 0
        live=r.timings.get(wid,{})
        record={"k":k,"winner_id":wid,"arm":win_arm,"drafter":v["drafter"],
                "rho":v["rho"] if v["quota"] else 0,"delta":candidate_size,"alpha":v["alpha"],
                "alpha_aliases":v["alpha_aliases"],"retrieval_quota":v["quota"],"beam_quota":k-v["quota"],
                "prefix_beams":v["prefix_beams"],"threshold":v["threshold"],"max_rounds":3 if v["quota"] else 0,
                "full_beam_width":50,"rl_beta":.001,"test_alpha_eff":t["alpha_eff"],
                "validation_hr":v["item_hr"],"validation_ndcg":v["item_ndcg"],
                "test_hr":t["item_hr"],"test_ndcg":t["item_ndcg"],
                "test_warm_hits":warm["item_hits"],"test_warm_hr":warm["item_hr"],"test_warm_ndcg":warm["item_ndcg"],
                "test_cold_hits":cold["item_hits"],"test_cold_hr":cold["item_hr"],"test_cold_ndcg":cold["item_ndcg"],
                "test_mean_rounds":t["mean_rounds"],"test_mean_verified":t["mean_verified"],
                "test_beam_call_rate":t["beam_call_rate"],"legacy_winner_id":old_id,
                "test_ndcg_delta_vs_legacy_pp":100*(t["item_ndcg"]-old_test["item_ndcg"]),
                "validation_runner_up_arm":second["arm"],"validation_ndcg_margin_to_next_arm_pp":ndcg_edge}
        for field in ("latency_p50_ms","latency_p95_ms","latency_p99_ms","requests_s_1gpu","requests_s_4gpu"):
            record[field]=live.get(field)
        records.append(record)
        params.append([k,NAMES[win_arm],record["rho"],candidate_size,choice_alpha(v),f"{v['quota']} + {k-v['quota']}",
                       pct(t["alpha_eff"]),v["prefix_beams"] if v["quota"] else "—",
                       threshold(v["threshold"]) if v["quota"] else "—",record["max_rounds"],f"{t['mean_rounds']:.2f}",
                       f"{t['mean_verified']:.2f}"])
        rows.append([k,NAMES[win_arm],pct(v["item_hr"]),pct(v["item_ndcg"]),pct(t["item_hr"]),pct(t["item_ndcg"]),
                     f"{cold['item_hits']}/{cold['rows']}",pct(cold["item_ndcg"]),r.timing(wid,"latency_p50_ms"),
                     r.timing(wid,"requests_s_1gpu"),r.timing(wid,"requests_s_4gpu")])
        reasons.append(f"<p><strong>K={k} · {NAMES[win_arm]}。</strong> {why}{parameter_reason(r,v)}{conclusion}</p>")
    write_csv(r.report/"champions.csv",records)
    return ("<section id='champions'><h2>各 K 的冠军配置与选择理由</h2>"
            "<p>下列五项均由验证集锁定。纯 beam、ANN、Qwen 以及 UniSRec 的两种候选数共同参选。"
            "质量指标按精确商品 ID 计算；P50 为单请求实测。</p>"+
            table(["K","最终冠军","验证 HR","验证 NDCG","测试 HR","测试 NDCG","冷命中/样本","冷 NDCG","P50 ms",
                   "单卡请求/s","四卡请求/s"],rows)+
            "<h3>实际生效的参数</h3><p>ρ 表示每轮草拟数相对于 K 的倍率，δ 为每轮候选上限。"
            "全部使用完整 beam 宽度 50、RL KL 系数 0.001、checkpoint-828。</p>"+
            table(["K","方案","ρ","δ","目标 α","检索 + beam 席位","实际 α_eff","前缀宽","阈值",
                   "最多轮数","实际轮数/请求","实际草拟数/请求"],params)+
            "".join(reasons)+"<p><a href='champions.csv'>五个冠军的参数、质量与实测性能 CSV</a></p></section>")


def calibration_section(r):
    c=r.selection["calibration"]
    old,new=c["old"],c["new"]
    if old["rows"]!=new["rows"] or old["percentiles"]!=new["percentiles"]:
        raise ValueError("Recalibration must use the historical validation request sample and quantiles")
    compare=[]
    for k in KS:
        for arm in ("unisrec1","unisrec2"):
            key=str(k)
            before=r.val[r.selection["old_grid_winners"][key][arm],"overall"]
            after=r.val[r.arm_winners[key][arm],"overall"]
            if objective(after)>objective(before):
                raise ValueError("Expanded threshold grid omitted a previous point")
            compare.append([k,NAMES[arm],route(before),pct(before["item_ndcg"]),
                            route(after),pct(after["item_ndcg"]),change(after["item_ndcg"],before["item_ndcg"])])
    return ("<h2 id='design'>如何确定冠军</h2>"
            f"<p>五个 K 共比较 {r.selection['validation_configurations']:,} 个唯一配置。每种方法、配额和前缀宽度先按其"
            f"验证表现选阈值，锁定 {r.selection['test_configurations']:,} 个配置；计入已公布冠军与固定参数验收后，"
            f"共有 {r.selection['evaluated_test_configurations']:,} 个不同配置接受测试。冠军优先比较整体验证集的精确商品 NDCG，"
            "并列时依次比较 HR、较少的实际验证候选数、预设网格顺序。测试集为原实验的同一批请求。</p>"
            "<p>阈值使用固定的 512 条验证历史计算。旧分位数来自 ANN、Qwen、UniSRec 全部每轮取 2K；"
            "新分位数将其中的 UniSRec 首轮改取 K，另外两路保持 2K。候选分数为三个 SID token 的全词表条件对数概率均值，"
            "阈值采用严格大于。所有检索方案都从新旧阈值并集及“全接受”中选参。</p>"+
            table(["首轮分数分位数","原阈值","重新校准阈值"],
                  [[f"P{p}",f"{a:.5f}",f"{b:.5f}"] for p,a,b in zip(new["percentiles"],old["values"],new["values"],strict=True)])+
            "<details><summary>相同候选数下，扩大阈值网格带来的变化</summary>"+
            table(["K","候选数方案","仅原阈值网格冠军","验证 NDCG","联合阈值网格冠军","验证 NDCG","差值 pp"],compare)+
            "</details><p><a href='selection.json'>阈值、选参规则与冻结配置</a> · "
            "<a href='validation.csv'>全验证网格</a> · <a href='test.csv'>全部测试指标和冷暖分组</a> · "
            "<a href='quality.json'>数据与候选集合验收</a></p>")


def headlines(r):
    by_arm=defaultdict(list)
    for k in KS:
        by_arm[r.val[r.chosen[str(k)],"overall"]["arm"]].append(k)
    summary="；".join(NAMES[arm]+" 在 K="+", ".join(map(str,ks))+" 取得最高验证排名" for arm,ks in by_arm.items())
    winner=r.chosen["10"]
    v,t=r.val[winner,"overall"],r.test[winner,"overall"]
    legacy=r.legacy["10"]; oldt=r.test[legacy,"overall"]
    oldc,newc=r.test[legacy,"cold"],r.test[winner,"cold"]
    return (f"<p class='lead'>{summary}。K=10 的最终方案为 <strong>{route(v)}</strong>。"
            f"相对已公布的 K=10 冠军，测试 HR 从 {pct(oldt['item_hr'])} 到 {pct(t['item_hr'])}，"
            f"NDCG 从 {pct(oldt['item_ndcg'])} 到 {pct(t['item_ndcg'])}，冷目标命中从 "
            f"{oldc['item_hits']}/{oldc['rows']} 到 {newc['item_hits']}/{newc['rows']}。</p>")


STYLE="""
:root{color-scheme:light;--ink:#142f38;--detail:#38525b;--line:#d5dfe0}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:16px/1.62 system-ui,-apple-system,sans-serif}
main{max-width:1240px;margin:auto;padding:32px clamp(18px,4vw,58px) 72px}h1,h2,h3{line-height:1.23}
h1{font-size:clamp(26px,4vw,44px);letter-spacing:-.035em}h2{font-size:clamp(22px,3vw,30px);border-top:1px solid var(--line);padding-top:26px;margin-top:40px}
.lead{font-size:1.11rem;color:var(--detail)}nav{display:flex;gap:8px 17px;flex-wrap:wrap}a{color:#075f73;text-underline-offset:.16em}
.scroll{overflow-x:auto;margin:.9rem 0}table{font-size:.86rem;font-variant-numeric:tabular-nums;border-collapse:collapse;width:100%;white-space:nowrap}
th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line)}th{background:#e7f0ef;color:#264449}
td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f8faf9}
details{margin:.75rem 0}summary{font-weight:650;cursor:pointer;color:#0c505d}details[open]>summary{margin-bottom:.45rem}
img{max-width:100%;height:auto;display:block}p{max-width:96ch}section#champions{padding-bottom:2rem}strong{color:#0d4e49}
@media print {main{max-width:none;padding:0}nav{display:none}h2{break-after:avoid}table{font-size:8pt}.scroll{overflow:visible}}
"""


def render(run_dir, original_run_dir, preview=False):
    r=Results(run_dir, preview)
    old=load(Path(original_run_dir)/"report"/"selection.json")
    if old["winners"]!=r.legacy:
        raise ValueError("Published winner IDs differ from the immutable previous selection")
    page=["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
          "<meta name='viewport' content='width=device-width,initial-scale=1'><title>UniSRec 草拟 K 与 2K 的对照验证</title>",
          f"<style>{STYLE}</style></head><body><main><h1>UniSRec 草拟 K 与 2K 的对照验证</h1>",
          "<p>Amazon Reviews 2018 · Industrial and Scientific · 4,532 条验证请求 · 4,533 条测试请求</p>",
          headlines(r),
          "<nav aria-label='报告目录'><a href='#controls'>固定参数对照</a><a href='#alpha'>融合比例</a>"
          "<a href='#coverage'>冷目标去向</a><a href='#performance'>时延与吞吐</a><a href='#champions'>五个 K 的最终冠军</a>"
          "<a href='previous/index.html'>上版完整验证报告</a></nav>",
          calibration_section(r),controlled_test(r),arm_comparisons(r),alpha_section(r),
          funnel_section(r),paired_section(r),rank_attribution(r),performance(r),batch_audit_table(r.report),
          "<h2>指标口径与逐请求结果</h2><p>每条用户历史有一个真实下一商品。精确商品 HR 在该设置下等于 Recall，"
          "NDCG 按命中名次作对数折扣。共享 SID 的商品按不同商品 ID 区分，完整 SID 命中指标另存于 CSV。"
          "目录共 3,686 件商品、3,670 个完整 SID，训练交互中未曾出现的目标被归入冷组。"
          "生成模型为 RL β=0.001 的 Qwen 0.5B checkpoint-828，单个候选由冻结模型打分。"
          "全量质量评估保持原评估的每卡 batch=4。各部分复用同一套已冻结配置及测试请求。</p>"
          "<p><a href='predictions/validation_winners.jsonl.gz'>五个冠军的逐请求预测、来源与分数</a></p>",
          champion_section(r),"</main></body></html>"]
    for file in (r.report/"index.html",r.report/"validation-report.html"):
        tmp=file.with_suffix(".tmp")
        tmp.write_text("".join(page),encoding="utf-8")
        tmp.replace(file)
    write_json(r.report/"report_status.json",dict(quality="passed",performance="passed" if r.timings else "pending",
        completed=not preview,configurations_measured=len(r.timings),configurations_planned=len(load(r.report/"benchmark_configs.json")),
        validation_winners=r.chosen))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--original-run-dir",type=Path,required=True)
    p.add_argument("--preview",action="store_true",help="Allow rendering before online benchmark and batch-audit results arrive")
    args=p.parse_args()
    render(args.run_dir,args.original_run_dir,args.preview)
    print(args.run_dir/"report"/"index.html",flush=True)


if __name__=="__main__":
    main()
