"""Reconcile the final UniSRec-2K evaluation with its archived rankings."""

import csv
import gzip
from collections import Counter, defaultdict
from html import escape
import json
from math import isclose, log2
from pathlib import Path

from hybrid.rho_report import metric_rows, objective
from hybrid.sweep import KS, alpha_groups


STAGES = {
    "sft": "线性 SFT · 3 轮",
    "rl001": "GRPO β=0.001",
    "rl01": "GRPO β=0.01",
}
METHODS = {
    "beam": "纯 beam50",
    "raw": "UniSRec 原排",
    "unisrec2": "UniSRec-2K + GRPO 验证",
    "unisrec1": "UniSRec-K + GRPO 验证",
    "ann": "向量 HNSW + GRPO 验证",
    "qwen": "Qwen 检索 + GRPO 验证",
}
GROUPS = {"overall": "整体", "warm": "暖商品", "cold": "冷商品"}
COLORS = {
    "sft": "#64748b", "rl001": "#075985", "rl01": "#7e22ce",
    "beam": "#64748b", "raw": "#b45309", "unisrec2": "#075985",
    "unisrec1": "#047857", "ann": "#a74335", "qwen": "#7937a2",
}
BASELINES = ("beam", "raw", "unisrec2", "unisrec1")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def write_csv(output, stem, rows):
    path = output / "data" / f"validation-{stem}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    def cells(values, tag):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)

    body = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in rows)
    return (f'<div class="table-wrap"><table><thead><tr>{cells(headers, "th")}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def start(section, title):
    return f'<section><h2 id="{section}">{title}</h2>'


def figure(stem, caption):
    return (f'<figure><img src="figures/validation-{stem}.png" alt="{escape(caption)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def link(stem, label):
    return f'<a href="data/validation-{stem}.csv">{label}</a>'


def pct(value):
    return f"{100 * value:.3f}%"


def pp(value):
    return f"{100 * value:+.3f}"


def close(left, right):
    return isclose(left, right, abs_tol=1e-12, rel_tol=0)


def assess(rank_rows, cutoffs):
    counts, hits, scores = Counter(), Counter(), Counter()
    for cold, rank in rank_rows:
        for group in ("overall", "cold" if cold else "warm"):
            counts[group] += 1
            for k in cutoffs:
                if rank is not None and rank <= k:
                    hits[k, group] += 1
                    scores[k, group] += 1 / log2(rank + 1)
    return {
        (k, group): dict(rows=n, hits=hits[k, group], hr=hits[k, group] / n,
                         ndcg=scores[k, group] / n)
        for k in cutoffs for group, n in counts.items()
    }


def rank(target, items):
    return items.index(target) + 1 if target in items else None


def evaluate_generation(checkpoints, sample, sids, cold):
    rl = checkpoints / "Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1" / "rl"
    beta01 = checkpoints / "Industrial_and_Scientific_05b_rl_beta0p01_20261118" / "rl"
    inputs = {
        "sft": (rl / "parent_sft_evaluation_metrics.json", rl / "parent_sft_predictions.jsonl"),
        "rl001": (rl / "evaluation/metrics.json", rl / "evaluation/predictions.jsonl"),
        "rl01": (beta01 / "evaluation/metrics.json", beta01 / "evaluation/predictions.jsonl"),
    }
    results = {}
    for name, (metric_path, prediction_path) in inputs.items():
        report = load(metric_path)
        if (report["status"], report["completed_rows"], report["num_beams"], report["prompt_style"]) != (
            "passed", len(sample), 50, "original-eval"
        ):
            raise ValueError(f"Unexpected SID evaluation protocol at {metric_path}")
        found = {}
        for row in records(prediction_path):
            key = row["row_id"]
            target = sample[key]["target"]
            if key in found or row["target"] != sids[target] or row["prompt_style"] != "original-eval":
                raise ValueError(f"SID evaluation request mismatch at {prediction_path}, row {key}")
            found[key] = cold[target], rank(row["target"], row["predictions"])
        if set(found) != set(sample):
            raise ValueError(f"Incomplete SID evaluation: {prediction_path}")
        result = assess(found.values(), (1, 3, 5, 10, 20, 50))
        for k, group in result:
            scores = result[k, group]
            if group == "overall" and any(
                not close(scores[field], report["metrics_by_k"][str(k)][field.upper()])
                for field in ("hr", "ndcg")
            ):
                raise ValueError(f"SID metrics disagree with manifest at {metric_path}, K={k}")
        for (k, group), scores in result.items():
            results[name, k, group] = scores
    return results


def evaluate_items(old, new, sample, sids, cold, frequency, selected, timings, test):
    alias = {}
    for i, sid in enumerate(sids):
        if sid not in alias or (-frequency[i], i) < (-frequency[alias[sid]], alias[sid]):
            alias[sid] = i
    results, overlap = {}, Counter()
    raw = {i: (cold[q["target"]], rank(q["target"], q["drafts"]["unisrec"]["50"]["first"])) for i, q in sample.items()}
    results.update({("raw", k, group): row for (k, group), row in assess(raw.values(), KS).items()})
    for k in KS:
        for arm in ("beam", "unisrec2", "unisrec1", "ann", "qwen"):
            ident = selected[str(k)][arm]
            if ident not in timings:
                raise ValueError(f"Missing timing for {ident}")
            for group in GROUPS:
                source = test[ident, group]
                results[arm, k, group] = dict(rows=source["rows"], hits=source["item_hits"],
                    hr=source["item_hr"], ndcg=source["item_ndcg"])
    beam = assess(((cold[q["target"]], rank(q["target"], [alias[sid] for sid in q["beam"]]))
                   for q in sample.values()), KS)
    for (k, group), row in beam.items():
        archive = results["beam", k, group]
        if any(not close(value, archive[field]) for field, value in row.items()):
            raise ValueError(f"Beam archive differs from reference item ranking, K={k}, {group}")

    for arm, folder in (("unisrec2", old), ("unisrec1", new)):
        expected = {selected[str(k)][arm]: k for k in KS}
        if arm == "unisrec2":
            prediction_path = folder / "report/predictions/validation_winners.jsonl.gz"
        else:
            prediction_path = folder / "report/predictions/unisrec1/validation_winners.jsonl.gz"
        found = defaultdict(dict)
        for row in records(prediction_path):
            k = expected.get(row["config_id"])
            if k is None:
                continue
            key, target = row["row_id"], row["target"]
            items = [value["item_id"] for value in row["ranking"]]
            if key in found[k] or sample[key]["target"] != target or len(items) != k or len(set(items)) != k:
                raise ValueError(f"Invalid saved {arm} ranking at K={k}, row {key}")
            result_rank = rank(target, items)
            found[k][key] = cold[target], result_rank
            if arm == "unisrec2":
                raw_rank = raw[key][1]
                overlap[k, raw_rank is not None and raw_rank <= k, result_rank is not None] += 1
        for k in KS:
            if set(found[k]) != set(sample):
                raise ValueError(f"Incomplete saved {arm} rankings at K={k}")
            for (_, group), row in assess(found[k].values(), (k,)).items():
                archive = results[arm, k, group]
                if any(not close(value, archive[field]) for field, value in row.items()):
                    raise ValueError(f"Saved {arm} ranking metrics differ, K={k}, {group}")
    return results, overlap


def alpha_grid(validation, test, selection):
    by_route = defaultdict(list)
    for (_, group), row in validation.items():
        if group == "overall":
            by_route[row["k"], row["quota"], row["arm"]].append(row)
    rows = []
    for k in KS:
        chosen = selection["arm_winners"][str(k)]
        for quota, alphas in alpha_groups(k).items():
            arms = ("ann", "qwen", "unisrec1", "unisrec2") if quota else ("beam",)
            for arm in arms:
                v = min(by_route[k, quota, arm], key=objective)
                t = test[v["id"], "overall"]
                cold = test[v["id"], "cold"]
                rows.append(dict(k=k, arm=arm, config_id=v["id"], quota=quota,
                    alpha_values=" ".join(f"{alpha:g}" for alpha in alphas), prefix_beams=v["prefix_beams"],
                    threshold=v["threshold"], validation_hr=v["item_hr"], validation_ndcg=v["item_ndcg"],
                    validation_alpha_eff=v["alpha_eff"], test_hr=t["item_hr"], test_ndcg=t["item_ndcg"],
                    test_alpha_eff=t["alpha_eff"], test_cold_hr=cold["item_hr"],
                    test_cold_ndcg=cold["item_ndcg"], is_arm_winner=chosen[arm] == v["id"]))
    return rows


def charts(output, sid, product, grid, timing, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "sft": "3-epoch SFT", "rl001": "GRPO beta=0.001", "rl01": "GRPO beta=0.01",
        "beam": "Beam50", "raw": "UniSRec raw", "unisrec2": "2K + GRPO scorer",
        "unisrec1": "K + GRPO scorer", "ann": "HNSW", "qwen": "Qwen retriever",
    }

    def save(fig, stem, axes):
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
        fig.savefig(output / "figures" / f"validation-{stem}.png", dpi=155)
        plt.close(fig)

    def quality(name, values, arms, columns, ks):
        fig, axes = plt.subplots(1, len(columns), figsize=(11, 3.9), layout="constrained")
        for ax, (metric, group, title) in zip(axes, columns):
            for arm in arms:
                scores = [values[arm, k, group][metric] for k in ks]
                ax.plot(range(len(ks)), [100 * score for score in scores], "o-", markersize=4,
                        color=COLORS[arm], label=labels[arm], linewidth=2 if arm == "unisrec2" else 1.4)
            ax.set_xticks(range(len(ks)), ks)
            ax.set_xlabel("K")
            ax.set_ylabel("Percent")
            ax.set_title(title)
        save(fig, name, axes)

    quality("sid-training-stages", sid, STAGES, (("hr", "overall", "SID HR"),
            ("ndcg", "overall", "SID NDCG"), ("hr", "cold", "Cold target SID HR")), (1, 3, 5, 10, 20, 50))
    quality("item-quality", product, BASELINES, (("hr", "overall", "Item HR"),
            ("ndcg", "overall", "Item NDCG"), ("hr", "cold", "Cold target item HR")), KS)

    for k in KS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
        selected_grid = [r for r in grid if r["k"] == k]
        beam = selected_grid[0]
        for arm in ("ann", "qwen", "unisrec1", "unisrec2"):
            arm_grid = [beam, *(r for r in selected_grid if r["arm"] == arm)]
            points = [(float(a), r) for r in arm_grid for a in r["alpha_values"].split()]
            for axis, field in zip(axes, ("validation_ndcg", "test_ndcg")):
                axis.plot([a for a, _ in points], [100 * r[field] for _, r in points], "o-",
                          markersize=3, color=COLORS[arm], label=labels[arm],
                          linewidth=2 if arm == "unisrec2" else 1.4)
        for axis, title in zip(axes, ("Validation", "Test, frozen configs")):
            axis.set_title(f"{title}  |  K={k}")
            axis.set_xlabel("Target retrieval fraction α")
            axis.set_ylabel("Item NDCG (%)")
        save(fig, f"alpha-k{k}", axes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    for axis, field, title in zip(axes, ("latency_p50_ms", "requests_s_4gpu"),
                                  ("Batch=1 P50 (ms)", "4-GPU throughput (requests/s)")):
        for arm in ("beam", "unisrec2", "unisrec1"):
            b = [timing[selected[str(k)][arm]][field] for k in KS]
            axis.plot(range(len(KS)), b, "o-", color=COLORS[arm], label=labels[arm])
        axis.set_xticks(range(len(KS)), KS)
        axis.set_xlabel("K")
        axis.set_title(title)
    save(fig, "latency-throughput", axes)

    fig, ax = plt.subplots(figsize=(9, 3.9), layout="constrained")
    stack = [0.0] * 4
    cases = [(10, "unisrec2"), (20, "unisrec2"), (50, "unisrec2"), (50, "beam")]
    for stage, label, color in (("verifier_tokenize", "Tokenizer", "#afc5dc"),
                                ("verifier_prefill", "Prompt prefill", "#60a5fa"),
                                ("drafter_query", "UniSRec encode", "#0d9488"),
                                ("draft_search", "Vector search", "#115e59"),
                                ("verification", "Candidate verification", "#1d4e89"),
                                ("full_beam", "Full beam", "#65768a")):
        values = [timing[selected[str(k)][arm]]["profile_batch4_stage_ms_per_request"].get(stage, 0.0)
                  for k, arm in cases]
        ax.bar(range(len(cases)), values, bottom=stack, color=color, label=label)
        stack = [base + value for base, value in zip(stack, values)]
    ax.set_xticks(range(len(cases)), [f"{labels[arm]} @ {k}" for k, arm in cases])
    ax.set_ylabel("Profiled ms/request (batch=4)")
    save(fig, "stage-time", (ax,))


def render_generation(sid):
    overall = [[k, STAGES[stage], pct(sid[stage, k, "overall"]["hr"]),
                pct(sid[stage, k, "overall"]["ndcg"]), *(f'{sid[stage, k, g]["hits"]}/{sid[stage, k, g]["rows"]}'
                for g in ("warm", "cold"))] for k in (1, 3, 5, 10, 20, 50) for stage in STAGES]
    a10, b10, c10 = (sid[stage, 10, "overall"] for stage in STAGES)
    a50, b50 = (sid[stage, 50, "overall"] for stage in ("sft", "rl001"))
    cold_10 = [sid[stage, 10, "cold"]["hits"] for stage in STAGES]
    cold_50 = [sid[stage, 50, "cold"]["hits"] for stage in STAGES]
    return ("<p>三组使用同一批 4,533 条测试历史、原项目评估提示和受约束 beam50。逐条核对目标 SID 后重新计算得分，"
            "整体结果与已归档评估指标一致。冷暖分组根据目标商品 ID 在训练交互中的出现情况划分。组内命中仍按 SID 计算。</p>"
            + figure("sid-training-stages", "三种检查点的生成排名评估。冷组分母为 147。")
            + table(["K", "检查点", "SID HR", "SID NDCG", "暖 SID 命中", "冷 SID 命中"], overall)
            + f"<p>相对线性 SFT 三轮，β=0.001 的 NDCG@10 从 {pct(a10['ndcg'])} 升至 {pct(b10['ndcg'])}，"
            f"HR@10 增加 {pp(b10['hr']-a10['hr'])} 个百分点。K=50 时 HR 变动 {pp(b50['hr']-a50['hr'])} 个百分点，"
            f"NDCG 仍提高 {pp(b50['ndcg']-a50['ndcg'])} 个百分点。命中总数略降，高名次命中的加权贡献同时增加。"
            f"β=0.01 的 NDCG@10 为 {pct(c10['ndcg'])}，与 β=0.001 相差 {pp(c10['ndcg']-b10['ndcg'])} 个百分点。"
            f"三组在 K=10 的冷目标命中依次为 {'、'.join(map(str,cold_10))} 条，K=50 为 {'、'.join(map(str,cold_50))} 条，均应结合 147 条样本解读。</p>"
            f"<p>{link('sid-checkpoints', '三组 SID 指标及冷暖分组 CSV')}</p></section>")


def render_product(product, overlap, paired, old_paired):
    leaders = []
    for k in KS:
        for arm in BASELINES:
            a, w, c = (product[arm, k, g] for g in GROUPS)
            leaders.append([k, METHODS[arm], f"{a['hits']}/{a['rows']}", pct(a["hr"]), pct(a["ndcg"]),
                pct(w["hr"]), pct(w["ndcg"]), f"{c['hits']}/{c['rows']}", pct(c["hr"]), pct(c["ndcg"])])
    also = [[k, METHODS[arm], pct(product[arm, k, "overall"]["hr"]),
             pct(product[arm, k, "overall"]["ndcg"]), f'{product[arm, k, "cold"]["hits"]}/147']
            for k in KS for arm in ("ann", "qwen")]
    def paired_range(record):
        low, high = record["ndcg_ci95_pp"]
        return f"{record['ndcg_delta_pp']:+.3f} [{low:+.3f}, {high:+.3f}]"

    differences = [[k, paired_range(old_paired[str(k)]["groups"]["overall"]),
                    paired_range(paired[str(k)]["unisrec1_minus_unisrec2"]["overall"])] for k in KS]
    raw10, two10 = (product[arm, 10, "overall"] for arm in ("raw", "unisrec2"))
    raw50, two50, one50 = (product[arm, 50, "overall"] for arm in ("raw", "unisrec2", "unisrec1"))
    cold_raw50, cold_two50 = (product[arm, 50, "cold"]["hits"] for arm in ("raw", "unisrec2"))
    return (figure("item-quality", "同一批测试请求上的四组商品级结果。UniSRec 原排直接取前 K 件。")
            + table(["K", "方案", "整体命中", "整体 HR", "整体 NDCG", "暖 HR", "暖 NDCG", "冷命中", "冷 HR", "冷 NDCG"], leaders)
            + f"<p>K=10 时，从 UniSRec 前 20 件里用 GRPO 选 10 件，命中由 {raw10['hits']} 增至 {two10['hits']}。"
            f"逐条对照是新增 {overlap[10,False,True]} 条、丢失 {overlap[10,True,False]} 条。"
            f"K=50 时，从 100 件筛到 50 件新增 {overlap[50,False,True]} 条、丢失 {overlap[50,True,False]} 条，"
            f"净命中由 {raw50['hits']} 变为 {two50['hits']}，而 NDCG 从 {pct(raw50['ndcg'])} 升至 {pct(two50['ndcg'])}，"
            f"冷目标命中从 {cold_raw50}/147 变为 {cold_two50}/147。扩大候选池同时改变了入选集合和名次。</p>"
            + f"<p>K=50 的单倍候选在同一批请求上命中 {one50['hits']} 条，NDCG 为 {pct(one50['ndcg'])}。此时全部前 50 件均保留，"
            "相对于 UniSRec 原排，同一商品集合上的 NDCG 差来自排序。双倍候选的 K=50 方案则更改了商品集合，"
            "因此要用自己对应的 HR、NDCG 与冷组结果。</p>"
            + "<h3>纯 beam、双倍及单倍候选的配对区间</h3><p>表中均为测试 NDCG 差值和 95% 区间，单位为百分点。"
            "按同一个测试请求配对，使用 10,000 次 bootstrap，区间按单独比较计算。K=20 的单倍与双倍候选差值区间跨零。"
            "K=50 单倍方案的区间整体偏正，同时单请求时延和四卡吞吐也更好。</p>"
            + table(["K", "2K 减纯 beam", "K 候选减 2K"], differences)
            + "<details><summary>另外两路草拟器的验证选优配置测试结果</summary>"
            + table(["K", "方案", "HR", "NDCG", "冷命中"], also) + "</details>"
            + f"<p>{link('item-test', '逐方法与冷暖分组完整 CSV')} · {link('paired', '配对差值及区间 CSV')}</p></section>")


def render_alpha(grid, controls, selection):
    deltas = [float(row["overall_rho1_minus_rho2_item_ndcg"]) for row in controls]
    n_same = sum(close(delta, 0.0) for delta in deltas)
    n_up = sum(delta > 1e-12 for delta in deltas)
    n_down = len(deltas) - n_same - n_up
    calibration = []
    for a, old, new in zip((10, 25, 50, 75, 90), selection["thresholds"]["old"][1:],
                           selection["thresholds"]["new"][1:]):
        calibration.append([f"P{a}", f"{old:.6f}", f"{new:.6f}"])
    sections = ["<p>目标 α 取 0、0.1 至 1.0，并加入各 K 的第一个非零配额断点 1/K。"
                "检索名额为 ⌊αK⌋，相同整数配额复用配置。图中 α=0 时各路线共用纯 beam。"
                "每个非零配额和草拟器独立在验证集选择前缀宽度与阈值。前缀宽度为 1、5、10、20、50，完整 beam 宽度固定 50。"
                "实际检索比例 α_eff 由填槽后成功保留的检索商品数决定。</p>"]
    for k in KS:
        two = [r for r in grid if r["k"] == k and r["arm"] in ("beam", "unisrec2")]
        rows = [[r["alpha_values"], r["quota"], r["prefix_beams"] or "—", threshold_name(r["threshold"]),
                 pct(r["validation_alpha_eff"]), pct(r["validation_ndcg"]),
                 pct(r["test_alpha_eff"]), pct(r["test_hr"]), pct(r["test_ndcg"])] for r in two]
        sections += [f"<details{' open' if k == 10 else ''}><summary>K={k} 的曲线与 2K 路线逐配额指标</summary>",
            figure(f"alpha-k{k}", f"K={k} 各候选方案随目标检索比例的变化。测试数据使用验证集锁定的配置。"),
            table(["目标 α", "检索名额", "前缀宽", "阈值", "验证 α_eff", "验证 NDCG",
                   "测试 α_eff", "测试 HR", "测试 NDCG"], rows), "</details>"]
    sections += ["<h3>阈值来源与固定条件对照</h3><p>两组都使用 seed=42 抽取同一组 512 条验证请求，从三路草拟器的"
                 "首轮验证分数计算以下分位数。旧组所有草拟器每轮取 2K，新组仅把 UniSRec 改为每轮取 K。"
                 "两列与全接受合并得到 11 个阈值选项。所有路线共享同一个并集。</p>",
                 table(["分位数", "原 2K 候选池", "UniSRec-K 候选池"], calibration),
                 f"<p>同一旧阈值、前缀宽度和配额下，共有 {len(deltas):,} 对配置。取 K 时验证 NDCG 上升 {n_up} 对、"
                 f"持平 {n_same} 对、下降 {n_down} 对。整个网格使用同一批请求，这些计数描述配置分布。"
                 "各 K 的 2K 臂冠军在旧网格和扩展网格中完全一致，验证 NDCG 的峰值都出现在 α=1。</p>",
                 f"<p>{link('alpha', '四种草拟器各配额的验证与测试 CSV')} · "
                 f"{link('controlled', '1,140 对固定参数对照 CSV')}</p></section>"]
    return "".join(sections)


def threshold_name(threshold):
    return "全接受" if threshold is None else f"{threshold:.6f}"


def render_performance(timing, arms, batch, protocol):
    rows = []
    stages = []
    timed = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "requests_s_1gpu", "requests_s_4gpu")
    for k in KS:
        for arm in ("beam", "unisrec2", "unisrec1"):
            b = timing[arms[str(k)][arm]]
            rows.append([k, METHODS[arm], *(f"{b[f]:.2f}" for f in timed), f"{b['gpu_peak_bytes_batch4']/2**30:.3f}"])
        stage = timing[arms[str(k)]["unisrec2"]]["profile_batch4_stage_ms_per_request"]
        stages.append([k, f"{stage.get('drafter_query', 0) + stage.get('draft_search', 0):.2f}",
                      *(f"{stage.get(f, 0):.2f}" for f in ("verifier_prefill", "verification", "full_beam"))])
    audits = [[a["k"], f"{a['set_equal']}/{a['n']}", f"{a['order_equal']}/{a['n']}",
              f"{a['b4_online_order_equal']}/{a['n']}", pp(a["b1_minus_b4_ndcg"])] for a in batch]
    dev = escape(protocol["device"])
    return (f"<p>在 {dev} 上用固定的 {len(protocol['sample_row_ids'])} 条测试历史重复 {protocol['repeats']} 次。"
            "模型和索引已驻留，计时从用户历史输入到排序响应，CUDA 同步计时。单条时延使用 batch=1。"
            "吞吐使用每卡 batch=4，四卡吞吐按整体请求数除以每次最慢 GPU 的时间之和。显存为单卡 batch=4 的 CUDA 已分配峰值。</p>"
            + figure("latency-throughput", "三条路线在不同 K 的在线实测。每条线使用同一批请求。")
            + table(["K", "方案", "P50 ms", "P95 ms", "P99 ms", "单卡请求/s", "四卡请求/s", "单卡峰值 GiB"], rows)
            + figure("stage-time", "单卡 batch=4 阶段计时。不同于 batch=1 的端到端延迟。")
            + table(["K", "2K 历史编码及检索 ms/请求", "验证模型 prefill", "候选验证", "完整 beam"], stages)
            + "<p>双倍候选的检索和历史编码约 0.5 毫秒/请求，候选验证在 K=10、20、50 约为 4.15、8.41、19.05 毫秒/请求。"
            "因此在这套小商品库中，增加 K 带来的主要算力开销发生在 GRPO 验证。K=50 的纯 beam 解码阶段约 72.19 毫秒/请求，"
            "双倍候选配置在首轮即可填满，用批量计算代替了完整 beam 搜索。</p>"
            "<h3>采用最终 2K 配置的数值复跑</h3><p>完整质量结果沿用每卡 batch=4。下表在相同的 256 个请求上比较 batch=1 "
            "及原始 batch=4 分组。排序对 BF16 批形状敏感，尤其是 K=50，当候选分数接近时，输出次序和入选集合均可能变化。"
            "表中的精确顺序一致需要整条列表逐位相同。P50 与质量数字适用的批设置在部署评估中应分别对照。</p>"
            + table(["K", "B1/B4 商品集合一致", "B1/B4 完整顺序一致", "B4 与归档顺序一致", "B1 减 B4 NDCG（百分点）"], audits)
            + f"<p>{link('benchmarks', '时延、吞吐、token 与显存 CSV')} · {link('batch-audit', '2K 配置批大小审计 CSV')}</p></section>")


def render_final(final):
    rows = [[c["k"], c["draft_limit"], f"{c['alpha']:g}", c["prefix_beams"], threshold_name(c["threshold"]),
             f"{c['mean_rounds']:.4f}", pct(c["beam_call_rate"]), pct(c["validation_ndcg"]), pct(c["test_hr"]),
             pct(c["test_ndcg"]), f"{c['cold_hits']}/147", f"{c['latency_p50_ms']:.2f}"] for c in final]
    return ("<p>项目最终统一采用每轮 2K，检索配额上限为 K。五档在 2K 臂中各自按验证 NDCG 选出的配置，"
            "均使用 α=1、前缀宽 1、最多 3 轮和完整 beam 宽度 50。采用 Qwen2.5-0.5B 的 GRPO β=0.001 第 828 步检查点。"
            "K=1 的阈值选在验证 P25，进入额外一轮的请求平均占比很小，最终仅 1 条调用完整 beam。"
            "其余 4 档均全接受，测试请求首轮填满，前缀宽度作为后续轮次的配置保留。"
            "宽度或阈值同分时使用已归档的固定枚举顺序。</p>"
            + table(["K", "每轮候选", "α", "前缀宽", "阈值", "实测轮数/请求", "完整 beam 调用率",
                     "验证 NDCG", "测试 HR", "测试 NDCG", "冷目标命中", "P50 ms"], rows)
            + "<p>K=1、5、10 的 2K 也是跨路线验证冠军。K=20、50 的跨路线验证冠军是单倍候选 K。"
            "本报告按统一双倍候选的要求发布配置，单倍候选的收益保留在各对照章节。"
            "K=50 的 2K 相比直接生成仍有明显的质量和延迟收益。相对单倍候选，2K 用更多验证算力换来的候选覆盖被模型筛选抵消，"
            "最终测试 HR、NDCG 和冷命中均低于单倍候选。</p>"
            f"<p>{link('final-2k', '最终五档配置、验证与测试结果 CSV')}</p></section>")


def export(output, sid, product, alpha_rows, controls, benchmark, batch, final, old_paired, paired):
    write_csv(output, "sid-checkpoints", [dict(stage=stage, k=k, group=group, **v)
                                           for (stage, k, group), v in sid.items()])
    write_csv(output, "item-test", [dict(method=arm, k=k, group=group, **v)
                                     for (arm, k, group), v in product.items()])
    write_csv(output, "alpha", alpha_rows)
    write_csv(output, "controlled", controls)
    timing_fields = ("id", "k", "arm", "rho", "quota", "alpha", "threshold", "prefix_beams",
                     "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_samples",
                     "requests_s_1gpu", "requests_s_4gpu", "items_s_1gpu", "items_s_4gpu",
                     "gpu_peak_bytes_batch1", "gpu_peak_bytes_batch4", "verifier_input_tokens_s_1gpu",
                     "verifier_input_tokens_s_4gpu", "profile_requests")
    stage_fields = sorted({key for b in benchmark for key in b["profile_batch4_stage_ms_per_request"]})
    write_csv(output, "benchmarks", [{**{field: row[field] for field in timing_fields},
               **{f"profile_ms_req_{field}": row["profile_batch4_stage_ms_per_request"].get(field, 0.0)
                  for field in stage_fields}} for row in benchmark])
    audit_fields = ("k", "config_id", "n", "set_equal", "order_equal", "b4_online_order_equal", "b1_ndcg", "b4_ndcg",
                    "b1_minus_b4_ndcg", "b1_hr", "b4_hr", "b1_minus_b4_hr")
    write_csv(output, "batch-audit", [{key: record[key] for key in audit_fields} for record in batch])
    write_csv(output, "paired", [dict(k=k, group=group, comparison=key,
                  ndcg_delta_pp=measure["ndcg_delta_pp"], ci95_low_pp=measure["ndcg_ci95_pp"][0],
                  ci95_high_pp=measure["ndcg_ci95_pp"][1], hr_delta_pp=measure["hr_delta_pp"],
                  gained_hits=measure["gained_hits"], lost_hits=measure["lost_hits"], rows=measure["rows"])
                  for k in KS for group in GROUPS for key, measure in
                  (("2k_minus_beam", old_paired[str(k)]["groups"][group]),
                   ("k_minus_2k", paired[str(k)]["unisrec1_minus_unisrec2"][group]))])
    write_csv(output, "final-2k", final)


def render(workspace: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    checkpoints = workspace / "weights/checkpoints/MiniOneRec"
    old = checkpoints / "Industrial_hybrid_beta001_inference_20261118"
    new = checkpoints / "Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118"
    report = new / "report"
    old_report = old / "report"
    selection = load(report / "selection.json")
    arms = selection["arm_winners"]
    quality = load(report / "quality.json")
    old_selection = load(old_report / "selection.json")
    validation = metric_rows(report / "validation.csv")
    test = metric_rows(report / "test.csv")
    paired = load(report / "paired.json")["by_k"]
    old_paired = load(old_report / "paired.json")["comparisons"]
    benchmark = load(report / "benchmarks.json")
    timing = {b["id"]: b for b in benchmark}
    protocol = load(report / "benchmark_protocol.json")
    audit = load(old_report / "batch_audit.json")
    batch = audit["rows"]
    catalog = load(old / "catalog.json")
    if catalog != load(new / "catalog.json"):
        raise ValueError("Product catalogs differ across experiments")
    sids, cold, frequency = (catalog[key] for key in ("sid_by_item", "cold", "target_freq"))
    sample = {}
    for shard in sorted((new / "test").glob("rank[0-3].jsonl")):
        for q in records(shard):
            if q["row_id"] in sample:
                raise ValueError(f"Repeated test row {q['row_id']}")
            sample[q["row_id"]] = q
    overall, warm, n_cold = len(sample), sum(not cold[q["target"]] for q in sample.values()), sum(
            cold[q["target"]] for q in sample.values())
    if (len(sids), len(set(sids)), sum(cold), overall, warm, n_cold) != (3686, 3670, 39, 4533, 4386, 147):
        raise ValueError("Unexpected item catalog or test population")
    if set(sample) != set(range(overall)) or quality["status"] != "passed" or quality["test_rows"] != overall:
        raise ValueError("Incomplete or unverified test artifacts")
    if (len(validation) != 3 * selection["validation_configurations"]
            or len(test) != 3 * selection["evaluated_test_configurations"]):
        raise ValueError("Incomplete validation/test metric grids")
    if len(benchmark) != 47 or len(timing) != len(benchmark):
        raise ValueError("Missing live timing cases")
    if audit["status"] != "measured" or audit["sample_row_ids"] != protocol["sample_row_ids"]:
        raise ValueError("The original 2K batch audit does not match the benchmark requests")
    reference = {q["row_id"]: q["predictions"] for q in records(old / "reference_test_beam.jsonl")}
    if reference.keys() != sample.keys() or any(reference[key] != q["beam"] for key, q in sample.items()):
        raise ValueError("New test cache changed the frozen baseline beam ranking")
    if any(any(q["drafts"]["unisrec"][str(k)]["first"] != q["drafts"]["unisrec"]["50"]["first"][:k]
               for k in KS) for q in sample.values()):
        raise ValueError("UniSRec first-round rankings are inconsistent across K")

    final = []
    for k, audit_row in zip(KS, batch, strict=True):
        key = str(k)
        ident = arms[key]["unisrec2"]
        if not (ident == audit_row["config_id"] == old_selection["winners"][key] == selection["legacy_winners"][key]
                == selection["old_grid_winners"][key]["unisrec2"] == old_paired[key]["winner"]):
            raise ValueError(f"The frozen 2K configuration identity differs at K={k}")
        v, t, c = validation[ident, "overall"], test[ident, "overall"], test[ident, "cold"]
        minimum = min((row for (_, group), row in validation.items()
                       if group == "overall" and row["k"] == k and row["arm"] == "unisrec2"), key=objective)
        if minimum["id"] != ident or (v["rho"], v["quota"], v["alpha"], v["prefix_beams"]) != (2, k, 1.0, 1):
            raise ValueError(f"Unverified final 2K route at K={k}")
        if audit_row["n"] != len(protocol["sample_row_ids"]) or audit_row["b4_online_order_equal"] != audit_row["n"]:
            raise ValueError(f"Original batch protocol ranking not reproducible at K={k}")
        n_beam = round(t["beam_call_rate"] * overall)
        if (n_beam, v["threshold"]) != ((1, selection["thresholds"]["old"][2]) if k == 1 else (0, None)):
            raise ValueError(f"Unexpected threshold or beam call rate at K={k}")
        final.append(dict(k=k, config_id=ident, rho=2, draft_limit=2*k, alpha=v["alpha"], quota=v["quota"],
                          prefix_beams=v["prefix_beams"], threshold=v["threshold"], max_rounds=3, full_beam_width=50,
                          mean_rounds=t["mean_rounds"], mean_verified=t["mean_verified"], alpha_eff=t["alpha_eff"],
                          beam_call_rate=t["beam_call_rate"], validation_hr=v["item_hr"], validation_ndcg=v["item_ndcg"],
                          test_hr=t["item_hr"], test_ndcg=t["item_ndcg"], test_hits=t["item_hits"], cold_hits=c["item_hits"],
                          cold_hr=c["item_hr"], cold_ndcg=c["item_ndcg"], latency_p50_ms=timing[ident]["latency_p50_ms"],
                          four_gpu_requests_s=timing[ident]["requests_s_4gpu"]))
    sid = evaluate_generation(checkpoints, sample, sids, cold)
    product, overlap = evaluate_items(old, new, sample, sids, cold, frequency, arms, timing, test)
    grid = alpha_grid(validation, test, selection)
    alpha_rows = [dict(r, target_alpha=float(value)) for r in grid for value in r["alpha_values"].split()]
    with (report / "alpha-comparison.csv").open(newline="", encoding="utf-8") as stream:
        alpha_identity = {(r["id"], float(r["alpha"])) for r in csv.DictReader(stream)}
    with (report / "controlled.csv").open(newline="", encoding="utf-8") as stream:
        controls = list(csv.DictReader(stream))
    if (len(alpha_rows) != 183 or {(r["config_id"], r["target_alpha"]) for r in alpha_rows} != alpha_identity
            or len(controls) != selection["controlled_validation_pairs"]):
        raise ValueError("Incomplete alpha grid or paired fixed-settings evidence")
    export(output, sid, product, alpha_rows, controls, benchmark, batch, final, old_paired, paired)
    charts(output, sid, product, grid, timing, arms)

    contents = [("val-protocol", "评估数据、指标与选参口径"), ("val-sid", "SFT 与 GRPO 的 SID 排名评估"),
                ("val-item", "商品级质量与冷暖分组"), ("val-alpha", "融合比例、阈值和候选倍率"),
                ("val-timing", "在线时延、吞吐和批大小审计"), ("val-final", "最终 2K 的五档配置与选择依据")]
    description = (f"<p>验证集 {quality['validation_rows']:,} 条用于选参，测试集 {overall:,} 条用于锁定后的比较。"
                   f"测试目标中暖商品 {warm:,} 条、冷商品 {n_cold} 条。冷目标占 {pct(n_cold/overall)}。"
                   "训练交互中曾作为历史或目标出现的商品为暖。目录共 3,686 件商品，映射到 3,670 个不同 SID。"
                   "每条请求只有一个目标，在给定的前 K 个结果内找到目标，HR 与 Recall 的数值相等。"
                   "NDCG 对命中名次 r 按 1/log₂(r+1) 加权，再在组内取平均。</p>"
                   "<p>纯生成训练阶段对照使用 SID 口径。检索及联合推理使用精确商品 ID。多个商品共享 SID 时，"
                   "生成出的同一 SID 可能映射到不同商品，因而两个口径的 HR 与 NDCG 分值会不同。"
                   "纯 beam 在商品评估中按训练目标频次及商品 ID 将 SID 映射回目录。</p>"
                   f"<p>扩展搜索共验证 {selection['validation_configurations']:,} 个配置。先按验证 NDCG、HR、"
                   "较少的平均验证候选数和固定枚举顺序锁定配置，再计算测试指标。"
                   f"测试包含 {selection['test_configurations']:,} 个锁定配置及 "
                   f"{selection['evaluated_test_configurations']-selection['test_configurations']} 个预先指定补充配置。"
                   "最终配置按已指定的统一每轮 2K 策略选择。逐 K 跨路线冠军作为对照展示。</p></section>")
    sections = (description, render_generation(sid), render_product(product, overlap, paired, old_paired),
                render_alpha(grid, controls, selection), render_performance(timing, arms, batch, protocol),
                render_final(final))
    return "".join(start(section, title) + section_body
                   for (section, title), section_body in zip(contents, sections, strict=True)), contents
