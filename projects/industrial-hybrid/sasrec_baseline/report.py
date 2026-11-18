"""Add the independent SASRec baseline to the latest RQ-VAE evaluation report."""

import csv
from html import escape
import json
import math
from pathlib import Path
import shutil

from rqvae_project_report.algorithm import math as formula
from rqvae_project_report.validation import GROUPS, KS, mark, pct, section, table

from .data import read_json, sha256

METHODS = ("SASRec", "联合投机推理")
COLORS = ("#31587c", "#007c73")
GROUP_NAMES = {"overall": "整体", "warm": "暖目标", "cold": "冷目标"}
LATENCY_KEYS = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
THROUGHPUT_KEYS = ("requests_s_1gpu", "requests_s_4gpu", "items_s_1gpu", "items_s_4gpu")


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def save_csv(output, stem, rows):
    with (output / "data" / f"sasrec-{stem}.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(stem, title):
    return f'<a href="data/sasrec-{stem}.csv">{escape(title)}</a>'


def figure(stem, caption):
    quoted = escape(caption, quote=True)
    return (
        f'<figure><img src="figures/sasrec-{stem}.png" alt="{quoted}">'
        f'<figcaption>{escape(caption)}</figcaption></figure>'
    )


def draw(output, stem, panels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), sharex=False, figsize=(9.4, 3.4),
                             layout="constrained", squeeze=False)
    try:
        for axis, settings in zip(axes[0], panels, strict=True):
            for line in settings["lines"]:
                x, y, label, color, style = line
                axis.plot(x, y, style, linewidth=1.55, markersize=3, label=label, color=color)
            axis.set(xlabel=settings["xlabel"], ylabel=settings["ylabel"])
            if "xticks" in settings:
                axis.set_xticks(range(len(settings["xticks"])), settings["xticks"])
            if settings.get("log"):
                axis.set_yscale("log")
            axis.grid(axis="y", color="#cad5de", linewidth=.55, alpha=.72)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, fontsize=8)
        fig.savefig(output / "figures" / f"sasrec-{stem}.png", dpi=160)
    finally:
        plt.close(fig)


def training_section(run, output, record):
    chosen = record["selected"]
    cases = record["cases"]
    grid = {(case["hidden_size"], float(case["learning_rate"])) for case in cases}
    expected = {(width, lr) for width in (64, 300) for lr in (.001, .002)}
    if len(cases) != 4 or grid != expected or sum(case["case_id"] == chosen["case_id"] for case in cases) != 1:
        raise ValueError("Incomplete SASRec validation grid or missing chosen case")
    validations = []
    ndcg_curves = []
    loss_curves = []
    grid_rows = []
    palette = ("#31587c", "#9b6835", "#007c73", "#8d1227")
    for case, color in zip(cases, palette, strict=True):
        hist = read_jsonl(run / "cases" / case["case_id"] / "epochs.jsonl")
        if [rec["epoch"] for rec in hist] != list(range(1, case["epochs_trained"] + 1)):
            raise ValueError(f"Incomplete validation history for {case['case_id']}")
        selected_epoch = hist[case["epoch"] - 1]
        for key in ("validation_hr10", "validation_ndcg10"):
            if not math.isclose(selected_epoch[key], case[key], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Chosen training epoch and validation record differ for {case['case_id']}")
        label = f"{case['hidden_size']} dim, lr {case['learning_rate']:g}"
        epochs = [row["epoch"] for row in hist]
        ndcg_curves.append((epochs, [100 * row["validation_ndcg10"] for row in hist], label, color, "-o"))
        loss_curves.append((epochs, [row["train_loss"] for row in hist], label, color, "-o"))
        validations.extend({**row, "case_id": case["case_id"], "hidden_size": case["hidden_size"],
                            "learning_rate": case["learning_rate"]} for row in hist)
        winner = case["case_id"] == chosen["case_id"]
        stop = "早停" if case["early_stopped"] else "达到上限"
        grid_rows.append([case["hidden_size"], f"{case['learning_rate']:g}", case["epochs_trained"],
                          stop, case["epoch"], pct(case["validation_hr10"]),
                          mark(pct(case["validation_ndcg10"]), winner), "所选" if winner else "—"])
        if winner:
            for key in ("epoch", "validation_hr10", "validation_ndcg10", "hidden_size", "learning_rate"):
                if chosen[key] != case[key]:
                    raise ValueError(f"Selected metadata mismatch for {key}")

    save_csv(output, "validation-history", validations)
    draw(output, "training", (
        dict(xlabel="Epoch", ylabel="Training cross-entropy", lines=loss_curves),
        dict(xlabel="Epoch", ylabel="Validation NDCG@10 (%)", lines=ndcg_curves),
    ))
    return (
        "<p>本实验的 GRPO 以每条训练请求的目标 SID 为标签。精确命中得 1，组内出现命中时，"
        "错误候选按名次加上负折扣。全组未命中时奖励为 0。独立 SASRec 在本次单独训练，用于推理对照。</p>"
        "<p>独立 SASRec 只以商品 ID 序列为输入，用相同的 36,259 条训练历史和 4,532 条验证历史训练与选点。"
        "模型有两层、两头自注意力，前馈维度为 256，dropout 为 0.2。各组使用 Adam、batch 512、"
        "seed 42，每组最多 30 轮。验证 NDCG@10 连续 5 轮未改善时早停，"
        "选点同分依次比较验证 HR@10、较早轮次和固定配置顺序。</p>"
        "<p>对历史 <var>H</var>，用最新位置表示 <var>h(H)</var> 与商品向量 <var>e<sub>j</sub></var> 的内积打分。"
        "用真实下一商品 <var>y</var> 在训练可见商品集合上的交叉熵训练，每批取请求损失均值。</p>"
        + formula("""
<mrow>
  <msub><mi>s</mi><mi>j</mi></msub><mo>=</mo><msup><mrow><mi>h</mi><mo>(</mo><mi>H</mi><mo>)</mo></mrow><mi>⊤</mi></msup>
  <msub><mi>e</mi><mi>j</mi></msub><mo>,</mo><mspace width="1em"/>
  <mi>ℒ</mi><mo>=</mo><mo>−</mo><mi>log</mi>
  <mfrac><msup><mi>e</mi><msub><mi>s</mi><mi>y</mi></msub></msup>
  <mrow><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><msub><mi>ℐ</mi><mtext>train</mtext></msub></mrow></munder>
  <msup><mi>e</mi><msub><mi>s</mi><mi>j</mi></msub></msup></mrow></mfrac>
</mrow>""", "SASRec 商品得分和真实下一商品的交叉熵")
        + table(("隐藏维度", "学习率", "实际轮数", "结束方式", "选中轮次",
                 "验证 HR@10", "验证 NDCG@10", "采用"), grid_rows)
        + figure("training", "四组独立 SASRec 的训练交叉熵与验证 NDCG@10 曲线。")
        + f"<p>所选权重为 {chosen['hidden_size']} 维、学习率 {chosen['learning_rate']:g}、第 {chosen['epoch']} 轮。"
        f"验证 NDCG@10 为 {pct(chosen['validation_ndcg10'])}。测试集只用于评估这组锁定权重。</p>"
        + f"<p>{download('validation-history', '下载四组训练损失与各轮验证结果')}</p>"
    )


def quality_record(run):
    comparison = read_json(run / "comparison.json")
    pairs = comparison["comparisons"]
    rows = {(int(row["k"]), row["stratum"]): row for row in pairs}
    counts = {"overall": comparison["test_rows"], "warm": comparison["warm_rows"],
              "cold": comparison["cold_rows"]}
    if (comparison["status"], comparison["validation_rows"], *counts.values()) != ("passed", 4532, 4533, 4386, 147):
        raise ValueError("SASRec comparison does not cover the frozen evaluation split")
    if (set(rows) != {(k, group) for k in KS for group in GROUPS} or len(rows) != len(pairs)
            or comparison["bootstrap"]["replicates"] != 10_000 or comparison["bootstrap"]["confidence"] != .95):
        raise ValueError("Missing SASRec cutoffs, evaluation groups or bootstrap metadata")
    for (_, group), row in rows.items():
        for side in ("sasrec", "hybrid"):
            item = row[side]
            if not (item["rows"] == counts[group] and 0 <= item["item_ndcg"] <= item["item_hr"] <= 1
                    and math.isclose(item["item_hr"], item["item_hits"] / counts[group], abs_tol=1e-10)):
                raise ValueError("Hit and ranking metrics disagree")
        for metric in ("hr", "ndcg"):
            old, new = (row[side][f"item_{metric}"] for side in ("sasrec", "hybrid"))
            delta = row["paired"][f"{metric}_delta_pp"]
            lo, hi = row["paired"][f"{metric}_ci95_pp"]
            if not math.isclose(delta, 100 * (new - old), abs_tol=1e-8) or not lo <= hi:
                raise ValueError("Paired point estimates disagree with per-method metrics")
        if group == "cold" and row["sasrec"]["item_hits"] != 0:
            raise ValueError("SASRec cold hits were measured outside the supported candidate scope")
    return comparison, rows


def paired_table(rows, group):
    data = []
    for k in KS:
        row = rows[k, group]
        for metric in ("hr", "ndcg"):
            old, new = (row[key][f"item_{metric}"] for key in ("sasrec", "hybrid"))
            values = [
                mark(pct(rate) if metric == "ndcg" else f"{pct(rate)}（{row[method]['item_hits']:,}）",
                     rate == max(old, new))
                for rate, method in ((old, "sasrec"), (new, "hybrid"))
            ]
            paired = row["paired"]
            l95, u95 = paired[f"{metric}_ci95_pp"]
            relative = row[f"relative_{metric}_gain_pct"]
            data.append([k, metric.upper(), *values, f"{paired[f'{metric}_delta_pp']:+.3f}",
                         f"[{l95:+.3f}, {u95:+.3f}]", "—" if relative is None else f"{relative:+.2f}%"])
    return table(("K", "指标", *METHODS, "差值（百分点）", "配对 95% 区间（百分点）", "相对差值"), data)


def quality_section(run, output, rows):
    sections = []
    for group in GROUPS:
        n = rows[KS[0], group]["sasrec"]["rows"]
        parts = paired_table(rows, group)
        if group == "overall":
            sections.append(f"<h3>{GROUP_NAMES[group]}（{n:,} 条）</h3>{parts}")
        else:
            sections.append(f"<details><summary>{GROUP_NAMES[group]}（{n:,} 条）</summary>{parts}</details>")
    metrics = []
    for key in ("hr", "ndcg"):
        series = [
            (range(len(KS)), [100 * rows[k, "overall"][method][f"item_{key}"] for k in KS],
             label, color, "-o")
            for method, label, color in zip(("sasrec", "hybrid"), ("SASRec", "Speculative"), COLORS, strict=True)
        ]
        metrics.append(dict(xlabel="K", ylabel=f"Test {key.upper()} (%)", lines=series, xticks=KS))
    draw(output, "quality", metrics)
    shutil.copyfile(run / "comparison.csv", output / "data/sasrec-comparison.csv")

    first = rows[1, "overall"]["paired"]
    last = rows[50, "overall"]["paired"]
    ndcg_baseline = "、".join(str(k) for k in KS if rows[k, "overall"]["paired"]["ndcg_delta_pp"] < 0)
    hr_hybrid = "、".join(str(k) for k in KS if rows[k, "overall"]["paired"]["hr_delta_pp"] > 0)
    hit_delta = {group: rows[50, group]["hybrid"]["item_hits"] - rows[50, group]["sasrec"]["item_hits"]
                 for group in GROUPS}
    first_lo, first_hi = first["ndcg_ci95_pp"]
    last_lo, last_hi = last["hr_ci95_pp"]
    return (
        "<p>两条路线在相同测试请求上按精确商品 ID 评估，历史中出现过的商品仍可推荐。"
        "SASRec 在训练历史或目标出现过的 3,647 件商品中打分。历史里的未知商品按原顺序滤除，"
        "空历史按训练目标频次和商品 ID 排序。全体测试请求含 147 条冷目标，"
        "冷商品超出该纯商品 ID 基线的候选范围。单独查看暖目标结果可比较双方在已见商品上的排序。</p>"
        + figure("quality", "相同 4,533 条请求的测试 HR 和 NDCG，按推荐件数绘制。")
        + "<p>差值为联合投机推理减 SASRec。配对区间按相同测试请求重采样 10,000 次，各 K 和各指标单独计算。"
          "表中 HR 括号内为命中数，深红加粗表示同一 K 与分组下较高的指标，同分同时标出。"
          "SASRec 指标为零时，列出绝对差值。</p>"
        + "".join(sections)
        + f"<p>整体 NDCG 在 K={ndcg_baseline} 时由 SASRec 领先，整体 HR 在 K={hr_hybrid} 时由联合投机推理领先。"
          f"K=1 时，联合投机推理的首位命中率低 {abs(first['ndcg_delta_pp']):.3f} 个百分点，"
          f"配对区间为 [{first_lo:+.3f}, {first_hi:+.3f}]。"
          f"K=50 时，联合投机推理的净命中差为 {hit_delta['overall']:+d} 条"
          f"（暖目标 {hit_delta['warm']:+d}、冷目标 {hit_delta['cold']:+d}），"
          f"HR 差值 {last['hr_delta_pp']:+.3f} 个百分点，配对区间 [{last_lo:+.3f}, {last_hi:+.3f}] 涵盖零。"
          "联合投机推理在较长榜单取得更高覆盖率，SASRec 把更多命中排在靠前位置。</p>"
        + f"<p>{download('comparison', '下载五档整体、冷暖分组及逐请求配对汇总')}</p>"
    )


def timing_index(data, expected_ids=None):
    if not isinstance(data, list):
        raise ValueError("Benchmark must contain one record per measured configuration")
    result = {}
    for row in data:
        if expected_ids is not None and row["id"] not in expected_ids:
            continue
        k = int(row["k"])
        if k in result:
            raise ValueError(f"Duplicate benchmark at K={k}")
        if (any(row[field] != 768 for field in ("latency_samples", "logical_requests_1gpu", "logical_requests_4gpu"))
                or any(row[field] != k * 768 for field in ("recommendations_batch1", "recommendations_1gpu",
                                                          "recommendations_4gpu"))):
            raise ValueError(f"Benchmark request or recommendation count mismatch at K={k}")
        fields = LATENCY_KEYS + THROUGHPUT_KEYS
        if any(not math.isfinite(row[key]) or row[key] <= 0 for key in fields):
            raise ValueError(f"Invalid measured timing at K={k}")
        result[k] = row
    if set(result) != set(KS):
        raise ValueError("Benchmark misses one or more cutoffs")
    return result


def runtime_section(rqvae, run, output, comparison, selection):
    archived = rqvae / "inference/report"
    protocol = read_json(run / "benchmark_protocol.json")
    archived_protocol = read_json(archived / "benchmark_protocol.json")
    settings = ("sample_row_ids", "seed", "repeats", "warmup_batches_per_shape", "latency_batch_size",
                "throughput_batch_size_per_gpu", "framework", "device")
    if (any(protocol[key] != archived_protocol[key] for key in settings)
            or protocol["archival_protocol_sha256"] != sha256(archived / "benchmark_protocol.json")
            or protocol["model_sha256"] != selection["artifact_sha256"]["model.ts"]
            or protocol["test_csv_sha256"] != comparison["source_sha256"]["test_csv"]):
        raise ValueError("SASRec and joint inference do not use the same benchmark protocol")
    sample_count = len(protocol["sample_row_ids"])
    parity = {f"batch{size}_exact_ranking": sample_count for size in (1, 4)}
    if sample_count != 256 or protocol["ranking_parity"] != [parity] * 4:
        raise ValueError("SASRec inference rankings must match offline Top50 at batch sizes 1 and 4 on all four GPUs")
    configuration = {int(k): key for k, key in comparison["hybrid_configs"].items()}
    baseline = timing_index(read_json(run / "benchmarks.json"))
    hybrid = timing_index(read_json(archived / "benchmarks.json"), set(configuration.values()))
    if {k: row["id"] for k, row in hybrid.items()} != configuration:
        raise ValueError("Joint inference benchmark does not match the evaluated configurations")

    speed_tables = [[], []]
    speed_data = []
    for k in KS:
        for i, (label, row) in enumerate(((METHODS[0], baseline[k]), (METHODS[1], hybrid[k]))):
            methods = (baseline[k], hybrid[k])
            speed_tables[0].append([k, label, *(mark(f"{row[field]:,.3f}", row[field] == min(
                item[field] for item in methods)) for field in LATENCY_KEYS)])
            speed_tables[1].append([k, label, *(mark(f"{row[field]:,.2f}", row[field] == max(
                item[field] for item in methods)) for field in THROUGHPUT_KEYS)])
            speed_data.append({"k": k, "method": ("sasrec", "hybrid")[i],
                               **{field: row[field] for field in LATENCY_KEYS + THROUGHPUT_KEYS}})
    save_csv(output, "performance", speed_data)
    speed_charts = []
    for key, label in (("latency_p50_ms", "P50 latency (ms)"), ("requests_s_4gpu", "Four-GPU requests/s")):
        curves = [
            (range(len(KS)), [stats[k][key] for k in KS], method, color, "-o")
            for stats, method, color in zip((baseline, hybrid), ("SASRec", "Speculative"), COLORS, strict=True)
        ]
        speed_charts.append(dict(xlabel="K", ylabel=label, lines=curves, log=True, xticks=KS))
    draw(output, "runtime", speed_charts)
    old = baseline[50]
    new = hybrid[50]
    return (
        f"<p>在 <b>{escape(protocol['device'])}</b>、PyTorch {escape(protocol['framework'])} 下，"
        f"两条路线用随机种子 {protocol['seed']} 抽取的同一批 {sample_count} 条历史。"
        "四张卡的 SASRec 在 batch=1 和 batch=4 下，均逐条复现样本的离线前 50 名商品顺序。"
        f"每种输入形状预热 {protocol['warmup_batches_per_shape']} 批并测 {protocol['repeats']} 轮。"
        f"单请求时延按 batch={protocol['latency_batch_size']}，从收到商品历史到返回已排序商品 ID。"
        f"吞吐采用每卡 batch={protocol['throughput_batch_size_per_gpu']}，四卡逐轮取最慢卡的耗时。"
        "深红加粗表示同一 K 中较低的时延或较高的吞吐。</p>"
        + figure("runtime", "同协议实测的单请求 P50 与四卡请求吞吐，纵轴为对数刻度。")
        + table(("K", "路线", "P50（ms）", "P95（ms）", "P99（ms）"), speed_tables[0])
        + table(("K", "路线", "单卡请求/秒", "四卡请求/秒", "单卡商品/秒", "四卡商品/秒"), speed_tables[1])
        + f"<p>K=50 时，SASRec 的 P50 为 {old['latency_p50_ms']:.3f} ms，"
          f"联合投机推理的 P50 为 {new['latency_p50_ms']:.3f} ms。四卡吞吐分别为 "
          f"{old['requests_s_4gpu']:,.2f} 和 {new['requests_s_4gpu']:,.2f} 请求/秒。"
          f"联合投机推理的 P50 是 SASRec 的 {new['latency_p50_ms']/old['latency_p50_ms']:.2f} 倍，"
          f"四卡吞吐是 SASRec 的 {100*new['requests_s_4gpu']/old['requests_s_4gpu']:.2f}%。"
          "SASRec 直接对商品打分，TPOT 不适用。联合投机推理在本文测速口径下整批返回商品。</p>"
        + f"<p>{download('performance', '下载两条路线的五档延迟和吞吐')}</p>"
    )


def render(rqvae: Path, run: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    """Render the SASRec addendum into the existing RQ-VAE report bundle."""
    comparison, rows = quality_record(run)
    record = read_json(run / "selection.json")
    sections = [
        ("sasrec-training", "独立 SASRec 基线的验证选点", training_section(run, output, record)),
        ("sasrec-quality", "SASRec 与联合投机推理的推荐质量", quality_section(run, output, rows)),
        ("sasrec-runtime", "SASRec 与联合投机推理的推理性能", runtime_section(rqvae, run, output, comparison, record)),
    ]
    return "".join(section(*entry) for entry in sections), [(anchor, title) for anchor, title, _ in sections]
