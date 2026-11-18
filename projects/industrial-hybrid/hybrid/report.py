"""Render offline validation selection, full-test outcome and live-latency summaries."""
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path

import numpy as np

from .sweep import DRAFTERS, KS, WIDTHS, alpha_groups, write_csv, write_json


def table(headers, rows):
    escape=lambda v:html.escape(str(v))
    return ("<div class='scroll'><table><thead><tr>"+"".join(f"<th>{escape(v)}</th>" for v in headers)
            +"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{escape(v)}</td>" for v in row)+"</tr>"
                                            for row in rows)+"</tbody></table></div>")


def pct(v):
    return f"{float(v)*100:.3f}%"


def delta(v):
    return f"{v:+.3f}"


def drafter_name(key):
    return {"beam":"原始 beam50","ann":"向量 HNSW","unisrec":"UniSRec","qwen":"Qwen 联合检索器"}[key]


def describe(row):
    if row["drafter"]=="beam":
        return "原始 beam50"
    return (f"{drafter_name(row['drafter'])}，配额 {row['quota']}/{row['k']}，"
            f"前缀宽 {row['prefix_beams']}，阈值 "+("全接受" if row['threshold'] is None else f"{float(row['threshold']):.3f}"))


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def split_tensor_rows(rows):
    return {(r["id"],r["stratum"]):r for r in rows}


def configure_plots(report):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hybrid-inference-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def alpha_comparison_rows(locked, val, test):
    """Expand stored aliases after choosing a single validation setting per drafter and quota."""
    grouped = {}
    for cfg in locked:
        grouped.setdefault((cfg["k"], cfg["quota"], cfg["drafter"]), []).append(cfg)

    def selection_key(cfg):
        row = val[cfg["id"], "overall"]
        return -float(row["item_ndcg"]), -float(row["item_hr"]), float(row["mean_verified"])

    rows = []
    for k in KS:
        for quota, aliases in alpha_groups(k).items():
            for drafter in DRAFTERS if quota else ("beam",):
                cfg = min(grouped[k, quota, drafter], key=selection_key)
                if cfg["alpha_aliases"] != aliases:
                    raise ValueError(f"Unexpected stored alpha aliases for {cfg['id']}")
                record = dict(k=k, drafter=drafter, retrieval_quota=quota, beam_quota=k-quota,
                              prefix_beams=cfg["prefix_beams"], threshold=cfg["threshold"],
                              config_id=cfg["id"], alpha_aliases=aliases)
                for split, metrics in (("validation", val), ("test", test)):
                    overall = metrics[cfg["id"], "overall"]
                    cold = metrics[cfg["id"], "cold"]
                    for key in ("item_hr", "item_ndcg", "alpha_eff"):
                        record[f"{split}_{key}"] = float(overall[key])
                    for key in ("item_hr", "item_ndcg"):
                        record[f"{split}_cold_{key}"] = float(cold[key])
                for alpha in aliases:
                    rows.append({"alpha": alpha, **record})
    return rows


def alpha_plot(report, k, rows):
    plt = configure_plots(report)
    fig, ax = plt.subplots(figsize=(8.4, 3.9), layout="constrained")
    baseline = next(r["validation_item_ndcg"] for r in rows if r["drafter"] == "beam")
    ax.axhline(100 * baseline, color="#59616a", lw=1.2, ls="--", label="Original beam")
    colors = {"ann": "#4266ae", "unisrec": "#14867a", "qwen": "#b84b32"}
    labels = {"ann": "HNSW", "unisrec": "UniSRec", "qwen": "Qwen retriever"}
    for drafter, marker in zip(DRAFTERS, ("o", "s", "^")):
        curve = [r for r in rows if r["drafter"] in (drafter, "beam")]
        ax.plot([r["alpha"] for r in curve], [100*r["validation_item_ndcg"] for r in curve],
                color=colors[drafter], label=labels[drafter], marker=marker, markersize=4)
    ticks = sorted({r["alpha"] for r in rows})
    ax.set_xticks(ticks, [f"{a:g}" for a in ticks], rotation=35 if k==50 else 0)
    ax.set_xlabel("Target retrieval fraction α")
    ax.set_ylabel(f"Validation item NDCG@{k} (%)")
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.15), frameon=False)
    fig.savefig(report/f"alpha-k{k}.png", dpi=150)
    plt.close(fig)


def fixed_k10_alpha_table(val):
    k = 10
    fixed = {}
    for (ident, stratum), row in val.items():
        if stratum != "overall" or int(row["k"]) != k:
            continue
        if row["drafter"] == "beam" or (int(row["prefix_beams"]) == 5 and row["threshold"] == ""):
            key = (int(row["quota"]), row["drafter"])
            if key in fixed:
                raise ValueError(f"Duplicate fixed-setting validation score for {ident}")
            fixed[key] = pct(row["item_ndcg"])
    rows = []
    for quota, aliases in alpha_groups(k).items():
        values = [fixed[quota, drafter if quota else "beam"] for drafter in DRAFTERS]
        rows.append([", ".join(f"{a:g}" for a in aliases), quota, *values])
    return ("<details><summary>K=10 固定前缀宽度和阈值，仅改变 α</summary>"
            "<p>以下为全量验证集的精确商品 NDCG@10。固定前缀宽度为 5，阈值为全接受。"
            "首轮在全目录取 20 个候选，足以满足最多 10 件的检索配额，通常首轮即停止。"
            "α=0 共享纯 beam 基线。</p>"
            + table(["目标 α","检索配额","向量 HNSW","UniSRec","Qwen 联合检索器"], rows)
            + "</details>")


def alpha_peaks(rows):
    peaks = []
    for drafter in DRAFTERS:
        candidates = [r for r in rows if r["drafter"] in ("beam", drafter)]
        score = max(r["validation_item_ndcg"] for r in candidates)
        highest = [r for r in candidates if r["validation_item_ndcg"] == score]
        values = []
        for route in ("beam", drafter):
            alphas = [f"{r['alpha']:g}" for r in highest if r["drafter"] == route]
            if alphas:
                values.append("、".join(alphas) + ("（纯 beam）" if route == "beam" else ""))
        peaks.append(f"{drafter_name(drafter)} 在 α={'、'.join(values)} 时验证 NDCG 最高，达到 <strong>{pct(score)}</strong>")
    return "<p>" + "。".join(peaks) + "。</p>"


def alpha_comparison_section(report, locked, val, test):
    rows = alpha_comparison_rows(locked, val, test)
    write_csv(report/"alpha-comparison.csv", rows)
    section = ["<section id='alpha-comparison'><h2>融合比例 α 对 HR 和 NDCG 的影响</h2>",
               "<p>五个 K 均验证了 α=0、0.1、0.2、…、1，并包含首个非零配额断点 1/K。"
               "K=20 多测了 0.05，K=50 多测了 0.02。检索目标配额为 ⌊αK⌋，配额为 0 时共用原始 beam。"
               "同一 K 下产生相同配额的 α 合并展示，CSV 按原始 α 展开。实际检索占比 α_eff 为每条请求最终保留的检索商品数除以 K 后取平均。</p>",
               "<p>每个目标 α 下，对每种草拟器按整体精确商品验证 NDCG 选前缀宽度和阈值。并列时先取较高验证 HR，再取较小的平均验证候选数，仍并列则沿用网格顺序。"
               "曲线各点显示该 α 调参后的验证结果，点之间的线仅用于阅读。对应测试指标使用同一个已冻结配置。"
               "<a href='alpha-comparison.csv'>下载各目标 α 的验证与测试明细（含冷商品 HR/NDCG）</a></p>"]
    for k in (10, *(k for k in KS if k != 10)):
        group = [r for r in rows if r["k"] == k]
        alpha_plot(report, k, group)
        values = []
        for r in group:
            if r["alpha"] != r["alpha_aliases"][0]:
                continue
            beam = r["drafter"] == "beam"
            threshold = "—" if beam else ("全接受" if r["threshold"] is None else f"{r['threshold']:.3f}")
            values.append([", ".join(f"{a:g}" for a in r["alpha_aliases"]), drafter_name(r["drafter"]),
                           f"{r['retrieval_quota']} + {r['beam_quota']}", r["prefix_beams"] if not beam else "—",
                           threshold, *[pct(r[field]) for field in ("validation_alpha_eff", "validation_item_hr",
                           "validation_item_ndcg", "test_alpha_eff", "test_item_hr", "test_item_ndcg")]])
        section.extend([f"<details{' open' if k==10 else ''}><summary>K={k} 的融合比例对照</summary>",
                        f"<img src='alpha-k{k}.png' alt='K={k} 三种草拟器在各目标 α 上选参后的验证 NDCG'>",
                        alpha_peaks(group),
                        table(["目标 α", "草拟器", "检索 + beam 配额", "前缀宽", "阈值", "验证 α_eff", "验证 HR",
                               "验证 NDCG", "测试 α_eff", "测试 HR", "测试 NDCG"], values)])
        if k == 10:
            section.append(fixed_k10_alpha_table(val))
        section.append("</details>")
    section.append("</section>")
    return "".join(section)


def quality_heatmaps(report,locked,val):
    plt=configure_plots(report)
    for k in KS:
        sub=[r for r in locked if r["k"]==k]
        qs=sorted(alpha_groups(k))
        pure=next(r for r in sub if r["drafter"]=="beam")
        pure_val=float(val[pure["id"],"overall"]["item_ndcg"])
        import matplotlib.colors as colors
        figure, axes=plt.subplots(3,1,figsize=(10,8),sharex=True, layout="constrained")
        matrices=[]
        for d in DRAFTERS:
            arr=np.full((len(WIDTHS),len(qs)),np.nan)
            arr[:,0]=0
            for r in sub:
                if r["drafter"]==d:
                    arr[WIDTHS.index(r["prefix_beams"]),qs.index(r["quota"])]=100*(float(val[r["id"],"overall"]["item_ndcg"])-pure_val)
            matrices.append(arr)
        limit=max(1e-5,max(float(np.nanmax(np.abs(m))) for m in matrices))
        norm=colors.TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        for d,ax,m in zip(DRAFTERS,axes,matrices):
            im=ax.imshow(m,interpolation="nearest",aspect="auto",cmap="RdBu",norm=norm)
            ax.set_yticks(range(len(WIDTHS)),WIDTHS)
            ax.set_ylabel("Prefix beams")
            ax.set_title(d.upper(),loc="left",fontsize=10,fontweight="bold")
        axes[-1].set_xticks(range(len(qs)),qs)
        axes[-1].set_xlabel(f"Retrieval quota floor(alpha x {k})")
        figure.suptitle(f"Validation item NDCG@{k}: difference from original beam (percentage points)")
        figure.colorbar(im,ax=axes,label="NDCG difference (pp)", shrink=.7)
        figure.savefig(report/f"validation-k{k}.png",dpi=135)
        plt.close(figure)


def latency_chart(report,benches,test,configs):
    plt=configure_plots(report)
    fig,axes=plt.subplots(1,len(KS),figsize=(17,3.8),layout="constrained")
    palette={"beam":"#222831","ann":"#4266ae","qwen":"#b84b32","unisrec":"#14867a"}
    for k,ax in zip(KS,axes):
        for name in ("beam",*DRAFTERS):
            rows=[r for r in benches if configs[r["id"]]["k"]==k and configs[r["id"]]["drafter"]==name]
            x=[r["latency_p50_ms"] for r in rows]
            y=[100*float(test[r["id"],"overall"]["item_ndcg"]) for r in rows]
            ax.scatter(x,y,label=name,color=palette[name],s=34)
        ax.set_title(f"K={k}")
        ax.set_xlabel("p50 latency (ms)")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Test item NDCG (%)")
    axes[-1].legend(fontsize=8)
    fig.savefig(report/"quality-latency.png",dpi=150)
    plt.close(fig)


def compare_quality(k, winner, baseline, test):
    result={"k":k,"winner":winner,"baseline":baseline}
    for stratum in ("overall","warm","cold"):
        a,b=test[baseline,stratum],test[winner,stratum]
        result[stratum]={
            "rows":int(a["rows"]),
            "hits":[int(a["item_hits"]),int(b["item_hits"])],
            "delta_hits":int(b["item_hits"])-int(a["item_hits"]),
            "delta_item_hr_pp":100*(float(b["item_hr"])-float(a["item_hr"])),
            "delta_item_ndcg_pp":100*(float(b["item_ndcg"])-float(a["item_ndcg"])),
            "delta_sid_hr_pp":100*(float(b["sid_hr"])-float(a["sid_hr"])),
        }
    return result


def k10_summary(result,test,configs):
    winner,beam=result["winner"],result["baseline"]
    wr,br=test[winner,"overall"],test[beam,"overall"]
    cold=result["cold"]
    dr=int(wr["item_hits"])-int(br["item_hits"])
    return (f"<p><strong>K=10 的验证冠军为 {html.escape(describe(configs[winner]))}。</strong>"
            f" 在测试集上，它的精确商品 HR 从纯 beam 的 {pct(br['item_hr'])} 变为 {pct(wr['item_hr'])}"
            f"（命中 {br['item_hits']} → {wr['item_hits']}，净变化 {dr:+d} 条），"
            f"NDCG 从 {pct(br['item_ndcg'])} 变为 {pct(wr['item_ndcg'])}。"
            f"冷商品命中从 {cold['hits'][0]} 变为 {cold['hits'][1]}，共 {cold['rows']} 条。"
            f"该冠军按完整 SID 匹配的 HR 为 {pct(wr['sid_hr'])}，"
            f"按精确商品 ID 的 HR 为 {pct(wr['item_hr'])}。</p>")


def paired_table(report, quality_comparisons):
    path = report / "paired.json"
    if not path.is_file():
        return ""
    data = json.loads(path.read_text())
    rows = []
    for k in KS:
        comparison = data["comparisons"][str(k)]
        reference = quality_comparisons[str(k)]
        if comparison["winner"] != reference["winner"] or comparison["baseline"] != reference["baseline"]:
            raise ValueError(f"Paired results and quality report use different configurations at K={k}")
        for key, label in (("overall", "整体"), ("warm", "暖商品"), ("cold", "冷商品")):
            group = comparison["groups"][key]
            expected = reference[key]
            if (group["rows"] != expected["rows"] or
                    [group["baseline_hits"], group["candidate_hits"]] != expected["hits"] or
                    not math.isclose(group["ndcg_delta_pp"], expected["delta_item_ndcg_pp"], abs_tol=1e-7)):
                raise ValueError(f"Paired results and test scores disagree for {key} at K={k}")
            intervals = []
            for metric in ("hr", "ndcg"):
                low, high = group[f"{metric}_ci95_pp"]
                intervals.append(f"{delta(group[f'{metric}_delta_pp'])} [{delta(low)}, {delta(high)}]")
            rows.append([k, label, group["rows"], group["gained_hits"], group["lost_hits"],
                         group["retained_hits"], *intervals])
    return ("<h2>同一测试请求上的成对变化</h2>"
            f"<p>对每个 K 的验证冠军和原始 beam，按测试行比较精确商品命中及排名。"
            f"在每个分组内有放回重采样 {data['replicates']:,} 次，基础随机种子为 {data['seed']}，"
            "报告变化的百分点与 95% 百分位区间。新增表示仅冠军命中，丢失表示仅 beam 命中。"
            "两者都命中时，目标商品的名次仍可影响 NDCG。</p>"
            + table(["K", "分组", "样本", "新增", "丢失", "均命中", "HR 变化 [95% 区间]",
                     "NDCG 变化 [95% 区间]"], rows)
            + "<p>多 K 和多个分组的比较属于探索性分析，区间未做多重比较校正。"
            "冷商品只有 147 条测试请求。如果某项指标在每条观察上都没有变化，"
            "经验重采样会得到 [0, 0]。这一结果仅描述当前观察样本，仍可能遗漏稀有的变化。"
            "<a href='paired.json'>成对统计与区间原始数据</a></p>")


def diagnostics_tables(report, winners):
    path = report / "funnel.json"
    if not path.is_file():
        return ""
    diagnostics = json.loads(path.read_text())["by_k"]
    rows, stages = [], []
    for k in KS:
        result = diagnostics[str(k)]
        if result["config_id"] != winners[str(k)]:
            raise ValueError(f"Diagnostics use a different validation winner at K={k}")
        groups = result["groups"]
        for key in (*DRAFTERS, "beam", "winner"):
            label = "验证冠军" if key == "winner" else drafter_name(key)
            values = []
            for s in ("overall", "warm", "cold"):
                metric = groups[s]["rankings"][key]
                values += [pct(metric["item_hr"]), pct(metric["item_ndcg"])]
            rows.append([k, label, *values])
        for key, label in (("overall", "整体"), ("warm", "暖商品"), ("cold", "冷商品")):
            f = groups[key]["funnel"]
            stages.append([k, label, f["rows"], f["first_drafted"], f["first_passed"], f["first_retrieved"],
                           f["later_drafted"], f["later_passed"], f["later_retrieved"], f["beam_hits"],
                           f["beam_hits_when_retrieval_quota_short"], f["final_hits"]])
    ten = diagnostics["10"]
    section = "<h2>直接返回草拟结果时的效果</h2>"
    if ten["drafter"] != "beam":
        g = ten["groups"]["overall"]
        f = g["funnel"]
        raw, champion = g["rankings"][ten["drafter"]], g["rankings"]["winner"]
        section += (f"<p>K=10 直接按{drafter_name(ten['drafter'])}的顺序推荐，整体 HR 为 {pct(raw['item_hr'])}，"
                    f"NDCG 为 {pct(raw['item_ndcg'])}。应用验证冠军配置后，HR 为 {pct(champion['item_hr'])}，"
                    f"NDCG 为 {pct(champion['item_ndcg'])}。逐条对照有 {f['winner_only_vs_raw']} 条新增命中，"
                    f"{f['raw_only_vs_winner']} 条丢失命中。新增命中中，{f['winner_only_from_first_below_k']} 条由首轮"
                    f"第 11–20 位进入结果，{f['winner_only_from_later_retrieval']} 条来自后续检索轮次，"
                    f"{f['winner_only_from_beam']} 条来自 beam。双方共同命中的 {f['shared_hits_vs_raw']} 条中，"
                    f"验证冠军将目标排得更靠前的有 {f['shared_ranks_improved']} 条，更靠后的有 {f['shared_ranks_worsened']} 条。</p>")
    return (section + "<p>三种草拟器均取无前缀首轮实际返回的 2K 个合法商品中的前 K 个，沿用草拟器自己的顺序。"
            "表中所有模型均以精确商品 ID 计算测试指标。</p>"
            + table(["K","排序来源","整体 HR","整体 NDCG","暖 HR","暖 NDCG","冷 HR","冷 NDCG"], rows)
            + "<h3>验证冠军的目标商品去向</h3>"
            "<p>下表计数的单位是请求。只有在线规则实际走到的轮次才参与计数。首轮和后续轮次分别记录目标出现、"
            "通过分数阈值以及按检索配额入选的请求数。最终命中等于两列检索入选加 beam 命中；"
            "其中“配额不足时”列是 beam 命中的子集，表示该请求实际检索商品数低于目标配额。</p>"
            + table(["K","分组","样本","首轮出现","首轮过阈","首轮入选","后续出现","后续过阈",
                     "后续入选","beam 命中","其中配额不足时","最终命中"], stages)
            + "<p><a href='funnel.json'>草拟器直接排序、分轮计数和逐阶段归因原始数据</a></p>")


def batch_audit_table(report):
    path = report / "batch_audit.json"
    if not path.is_file():
        return ""
    audit = json.loads(path.read_text())
    rows = []
    if sorted(e["k"] for e in audit["rows"]) != list(KS):
        raise ValueError("Batch audit must contain one row for every evaluated K")
    for entry in sorted(audit["rows"], key=lambda e: e["k"]):
        n = entry["n"]
        order, members = entry["order_equal"], entry["set_equal"]
        if not 0 <= order <= members <= n or n <= 0:
            raise ValueError(f"Invalid request agreement counts for K={entry['k']}")
        hr_change = 100 * (entry["b1_hr"] - entry["b4_hr"])
        ndcg_change = 100 * (entry["b1_ndcg"] - entry["b4_ndcg"])
        rows.append([entry["k"], f"{members}/{n}", f"{order}/{n}", f"{entry['b4_online_order_equal']}/{n}",
                     pct(entry["b4_hr"]), pct(entry["b1_hr"]),
                     delta(hr_change), pct(entry["b4_ndcg"]), pct(entry["b1_ndcg"]), delta(ndcg_change)])
    return ("<h2>单请求与批量推理的一致性</h2>"
            "<p>在同一组固定测试请求上，逐一比对各 K 验证冠军的推荐商品 ID。集合一致表示两种运行方式选中"
            "相同商品，完整顺序一致还要求每一位相同。以全量质量评估的批量协议（每卡 4 条）为基准，"
            "差值为单请求指标减去同一批抽样请求的基准指标，单位为百分点。批量复跑一列使用原先的批次分组与请求位置，"
            "核验在线代码与归档排序的逐位一致性。</p>"
            + table(["K", "单请求集合一致/请求", "单请求顺序一致/请求", "批量复跑顺序一致/请求", "批量 HR", "单请求 HR", "HR 变化",
                     "批量 NDCG", "单请求 NDCG", "NDCG 变化"], rows)
            + "<p>BF16 推理时，批量形状及历史补齐长度可能影响条件概率的数值舍入。"
            "候选得分接近时，顺序可能随之变化。目标商品仅在列表内部换位时，HR 保持相同，NDCG 可能变化。"
            "<a href='batch_audit.json'>成对请求与批形状审计数据</a></p>")


def measured_summary(benches,configs,winners,baselines,test):
    by_id={b["id"]:b for b in benches}
    profiles=[]; extremes=[]; records={}
    for k in KS:
        winner,beam=winners[str(k)],baselines[k]
        a,b=by_id[beam],by_id[winner]
        for role,c in (("原始 beam",a),("验证效果最优",b)):
            profiles.append([k,role,describe(configs[c["id"]]),
                *[f"{c[name]:.2f}" for name in ("latency_p50_ms","latency_p95_ms","requests_s_1gpu","requests_s_4gpu")],
                f"{a['latency_p50_ms']/c['latency_p50_ms']:.2f}×",
                f"{c['requests_s_4gpu']/a['requests_s_4gpu']:.2f}×"])
        eligible=[c for c in benches if configs[c["id"]]["k"]==k]
        fastest=min(eligible,key=lambda c:c["latency_p50_ms"])
        most1=max(eligible,key=lambda c:c["requests_s_1gpu"])
        most4=max(eligible,key=lambda c:c["requests_s_4gpu"])
        records[str(k)]=dict(lowest_median_latency=fastest["id"],highest_single_gpu_throughput=most1["id"],
            highest_four_gpu_throughput=most4["id"],validation_winner=winner,beam=beam,
            winner_median_speedup=a["latency_p50_ms"]/b["latency_p50_ms"],
            winner_p95_speedup=a["latency_p95_ms"]/b["latency_p95_ms"],
            winner_4gpu_throughput_ratio=b["requests_s_4gpu"]/a["requests_s_4gpu"])
        for title,c,metric,unit in (("最低中位时延",fastest,"latency_p50_ms","ms"),
                ("最高单卡吞吐",most1,"requests_s_1gpu","请求/s"),("最高四卡吞吐",most4,"requests_s_4gpu","请求/s")):
            extremes.append([k,title,describe(configs[c["id"]]),f"{c[metric]:.2f} {unit}",
                pct(test[c["id"],"overall"]["item_ndcg"]),pct(test[c["id"],"cold"]["item_hr"])])
    return records,profiles,extremes


def render(root):
    root=Path(root)
    report=root/"report"
    selection=json.loads((report/"selection.json").read_text())
    quality=json.loads((report/"quality.json").read_text())
    val=split_tensor_rows(read_csv(report/"validation.csv"))
    test=split_tensor_rows(read_csv(report/"test.csv"))
    locked=selection["locked"]
    configs={r["id"]:r for r in locked}
    bench_plan=json.loads((report/"benchmark_configs.json").read_text())
    planned={r["id"]:r for r in bench_plan}
    benchpath=report/"benchmarks.json"
    benches=json.loads(benchpath.read_text()) if benchpath.is_file() else []
    if isinstance(benches,dict):
        benches=benches["rows"]
    for row in benches:
        if row["id"] not in planned or not all(math.isfinite(float(row[k])) and float(row[k])>0 for k in
            ("latency_p50_ms","latency_p95_ms","latency_p99_ms","requests_s_1gpu","requests_s_4gpu")):
            raise ValueError("Invalid live benchmark row: "+str(row.get("id")))
    if benches and sorted(r["id"] for r in benches)!=sorted(planned):
        raise ValueError("Live benchmark table does not cover every representative exactly once")
    winner_tab=[]; cold_tab=[]; comparison_rows=[]; baselines={}; quality_comparisons={}
    for k in KS:
        best=selection["winners"][str(k)]
        beam=next(r["id"] for r in locked if r["k"]==k and r["drafter"]=="beam")
        baselines[k]=beam
        quality_comparisons[str(k)]=compare_quality(k,best,beam,test)
        winner_tab.append([k,describe(configs[best]),pct(val[best,"overall"]["item_ndcg"]),
            pct(test[best,"overall"]["item_hr"]),pct(test[best,"overall"]["item_ndcg"]),
            delta(100*(float(test[best,"overall"]["item_ndcg"])-float(test[beam,"overall"]["item_ndcg"]))),
            pct(test[best,"overall"]["sid_ndcg"])])
        cold_tab.append([k,*[test[best,s]["item_hits"]+"/"+test[best,s]["rows"] for s in ("warm","cold")],
            pct(test[best,"cold"]["item_ndcg"]),test[beam,"cold"]["item_hits"]+"/"+test[beam,"cold"]["rows"],
            pct(test[best,"overall"]["alpha_eff"]),pct(test[best,"overall"]["below_retrieval_quota_rate"]),
            pct(test[best,"overall"]["beam_call_rate"])])
        for d in ("beam",*DRAFTERS):
            options=[r for r in locked if r["k"]==k and r["drafter"]==d]
            chosen=min(options,key=lambda r:(-float(val[r["id"],"overall"]["item_ndcg"]),
                                             -float(val[r["id"],"overall"]["item_hr"]),
                                             float(val[r["id"],"overall"]["mean_verified"])))
            i=chosen["id"]
            comparison_rows.append([k,describe(chosen),pct(val[i,"overall"]["item_ndcg"]),
                pct(test[i,"overall"]["item_ndcg"]),pct(test[i,"cold"]["item_hr"]),pct(test[i,"overall"]["alpha_eff"]),
                f"{float(test[i,'overall']['mean_rounds']):.2f}",f"{float(test[i,'overall']['mean_verified']):.2f}"])
    quality_heatmaps(report,locked,val)
    base_css="""
      body{font:16px/1.5 system-ui,sans-serif;background:#fff;color:#172d39;margin:0 auto;padding:2rem 3vw;max-width:1220px}
      h1{font-size:2rem} h2{margin-top:2rem} .scroll{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.89rem}
      td,th{padding:.42rem .55rem;border-bottom:1px solid #d6dfe2;text-align:right;font-variant-numeric:tabular-nums}
      th{background:#eaf2f3;font-weight:650}td:first-child,th:first-child{text-align:left} img{width:100%;max-width:1000px}
      a{color:#164f8c} details{margin:.9rem 0} summary{cursor:pointer} .warning{color:#783e04}"""
    title="Industrial 混合推理验证报告"
    h=[f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>{base_css}</style><main><h1>{title}</h1>',
       f"<p>验证器使用 MiniOneRec RL β=0.001 的 checkpoint-828。共有 {selection['validation_configurations']:,} 个不同验证配置在 {quality['validation_rows']:,} 条样本上计分，冻结 {quality['locked_configs']:,} 个配置后评估全部 {quality['test_rows']:,} 条测试样本。</p>",
       k10_summary(quality_comparisons["10"],test,configs),
       alpha_comparison_section(report,locked,val,test),
       "<h2>按验证集精确商品 NDCG 选出的配置</h2>",
       table(["K","验证最优配置","验证 NDCG","测试 HR","测试 NDCG","较 beam 变化 pp","测试 SID NDCG"],winner_tab),
       "<p>阈值和推荐配置均在测试前锁定。同分时依次比较验证 HR 和平均验证候选数。下面的变化均相对相同 K、相同商品 ID 命中规则的原始 beam50，单位为百分点。</p>",
       table(["K","整体 HR 变化","整体 NDCG 变化","整体净增命中","暖商品净增命中","冷商品净增命中","冷 HR 变化","冷 NDCG 变化"],
          [[k,delta(q["overall"]["delta_item_hr_pp"]),delta(q["overall"]["delta_item_ndcg_pp"]),
            f"{q['overall']['delta_hits']:+d}",f"{q['warm']['delta_hits']:+d}",f"{q['cold']['delta_hits']:+d}",
            delta(q["cold"]["delta_item_hr_pp"]),delta(q["cold"]["delta_item_ndcg_pp"])]
           for k in KS for q in [quality_comparisons[str(k)]]]),
       "<h2>冷暖商品与补位</h2>",table(["K","暖商品命中/样本","冷商品命中/样本","冷商品 NDCG","beam 冷命中/样本",
              "实际检索占比","检索配额不足率","完整 beam 调用率"],cold_tab),
       "<p>冷暖按训练交互的历史与目标商品 ID 判定，147 条冷测试样本中一次命中对应约 0.680 个百分点。实际检索占比是逐条请求实际保留的检索商品数除以 K 再取平均，受验证阈值和候选覆盖影响。检索配额不足率表示检索未填满目标配额的请求比例，完整 beam 调用率还包括预先为 beam 留位的请求。</p>",
       paired_table(report,quality_comparisons),
       diagnostics_tables(report,selection["winners"]),
       "<h2>各草拟器的验证最优混合配置</h2>",table(["K","配置","验证 NDCG","测试 NDCG","测试冷 HR","实际检索占比","平均轮数","平均验证数"],comparison_rows),
       "<h2>验证网格</h2><p>色阶显示各草拟器相对于原始 beam 的验证 NDCG 变化，单位为百分点。横轴为整数检索配额，给定 K 的等价 α 共享同一结果；原始比例在 CSV 中列出。每个单元格采用该配置验证集选出的阈值。</p>"]
    for k in KS:
        h.append(f"<details{' open' if k==10 else ''}><summary>K={k}</summary><img src='validation-k{k}.png' alt='K={k} 验证热图'></details>")
    h.append(batch_audit_table(report))
    h.append("<h2>推理性能</h2>")
    speed_summary={}
    if benches:
        latency_chart(report,benches,test,configs)
        speed_summary,profiles,extremes=measured_summary(benches,configs,selection["winners"],baselines,test)
        k10=speed_summary["10"]
        h.append(f"<p><strong>K=10 验证冠军相对纯 beam 的中位时延加速比为 {k10['winner_median_speedup']:.2f}×</strong>，"
                 f"p95 时延加速比为 {k10['winner_p95_speedup']:.2f}×，四卡吞吐比为 {k10['winner_4gpu_throughput_ratio']:.2f}×。"
                 "时延加速比按基线时延除以配置时延计算，吞吐比按配置吞吐除以基线吞吐计算，大于 1 表示该项改善。</p>")
        h.append(table(["K","选择依据","配置","p50 ms","p95 ms","单卡请求/s","四卡请求/s","p50 加速比","四卡吞吐比"],profiles))
        h.append("<h3>代表配置中的实测速度冠军</h3>")
        h.append(table(["K","选择依据","配置","测量值","测试 NDCG","测试冷 HR"],extremes))
        protocol_file=report/"benchmark_protocol.json"
        if protocol_file.is_file():
            protocol=json.loads(protocol_file.read_text())
            h.append(f"<p>设备为 {html.escape(str(protocol.get('device','GPU')))}。固定 {len(protocol['sample_row_ids'])} 条测试输入，每种配置重复 {protocol['repeats']} 轮，"
                     f"每种 batch 形状预热 {protocol['warmup_batches_per_shape']} 批。单请求时延使用 batch={protocol['latency_batch_size']}，"
                     f"吞吐使用每卡 batch={protocol['throughput_batch_size_per_gpu']}。四卡并行以每轮最慢 rank 为完成时间。"
                     "各模型和索引常驻，请求之间重建历史缓存。单卡测量时同机其他卡可能并行测量其他配置。</p>")
        h.append("<p>每个实测配置使用同一批输入。解码、检索、验证和补位随请求实际执行，表中单请求时延为毫秒。运行条件与逐阶段数据保存在 benchmark JSON。</p>")
        h.append("<img src='quality-latency.png' alt='测试 NDCG 与端到端 p50 时延'>")
        h.append(table(["K","配置","p50 ms","p95 ms","p99 ms","单卡请求/s","四卡请求/s"],
            [[configs[r['id']]['k'],describe(configs[r['id']]),*[f"{float(r[f]):.2f}" for f in ("latency_p50_ms","latency_p95_ms","latency_p99_ms","requests_s_1gpu","requests_s_4gpu")]] for r in benches]))
        h.append("<p><a href='benchmarks.json'>在线性能记录与阶段分解</a> · <a href='benchmark_protocol.json'>在线测量协议及抽样行号</a></p>")
    else:
        h.append("<p class='warning'>效果评估已完成。在线性能结果尚未写入本报告。</p>")
    h.extend(["<h2>数据与评测口径</h2><p>目录共有 3,686 件商品、3,670 个完整 SID。精确 HR 要求返回目标商品 ID；历史 SID 口径接受同完整 SID 的商品，取第一次匹配的名次。两套分数的差距来自共享 SID 的商品。beam 的别名按训练目标频数优先，同频选最小数值 ID，每个 beam SID 最多入选一件商品。三位 SID 的各位均进入评分。</p>",
        "<p>向量 HNSW 的输入是原始 3,686×2,560 商品向量，依照配套元数据行序并使用归一化历史均值查询。该向量库缺少独立生成记录。UniSRec 使用保留的 20 轮模型和全目录 300 维向量。Qwen 联合检索器使用其独立检索头和 3,670×256 SID 索引，映射到全目录商品 ID。验证器为单独的 RL 权重。</p>",
        "<p>草拟量为 2K，最多三轮。各商品的条件对数概率取自模型全词表，各候选独立使用同一历史。混合列表先按配额选成员，再由验证器排序。纯 beam 保留原顺序。测试逐行核对全部 50 个 SID 的顺序，并校验商品 ID 命中数。</p>",
        "<p><a href='../ann_audit.json'>HNSW 近邻核验</a> · <a href='../trace.json'>逐轮草拟与过滤示例</a> · <a href='../catalog.json'>商品与 SID 目录及训练频数</a></p>",
        "<p><a href='validation.csv'>完整验证网格 CSV</a> · <a href='test.csv'>完整测试与分组 CSV</a> · <a href='selection.json'>阈值与锁定配置</a> · <a href='quality.json'>验收与预测格式</a> · <a href='outcomes.json'>各 K 相对基线的结果</a> · <a href='predictions/validation_winners.jsonl.gz'>验证冠军逐样本测试推荐</a></p></main></html>"])
    page="".join(h)
    for name in ("index.html","validation-report.html"):
        path=report/name
        temp=path.with_suffix(".tmp");temp.write_text(page,encoding="utf-8");temp.replace(path)
    write_json(report/"report_status.json",dict(quality="passed",performance="passed" if benches else "pending",
        configurations_measured=len(benches),configurations_planned=len(bench_plan),
        completed=bool(benches),validation_winners=selection["winners"]))
    write_json(report/"outcomes.json",dict(test_changes_against_beam=quality_comparisons,measured_speed=speed_summary))
