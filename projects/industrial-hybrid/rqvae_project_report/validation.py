"""Render the newest RQ-VAE run's evaluation and performance report."""

from collections import Counter
import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import median

from hybrid.rho_bundle import digest


KS = (1, 5, 10, 20, 50)
GENERATION_KS = (1, 3, 5, 10, 20, 50)
GROUPS = ("overall", "warm", "cold")
METHODS = {
    "beam": "直接生成（beam 50）",
    "unisrec_original_rank": "UniSRec 原排",
    "unisrec_2k": "联合投机推理",
}
COMBINED = "unisrec_2k"
TIMED = ("beam", COMBINED)
TIMING_FIELDS = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                 "requests_s_1gpu", "requests_s_4gpu", "items_s_4gpu", "recommended_sid_tokens_s_4gpu")
COLORS = ("#2c5475", "#956134", "#007b74")
ORIGINAL = Path("MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey")
CATALOG = "Industrial_and_Scientific.index.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.3f}%"


def mark(text, best):
    value = escape(str(text))
    return f'<strong class="best">{value}</strong>' if best else value


def table(headers, rows):
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(anchor, title, body):
    return f'<section><h2 id="{anchor}">{escape(title)}</h2>{body}</section>'


def figure(stem, caption):
    description = escape(caption)
    return (f'<figure><img src="figures/validation-{stem}.png" alt="{description}">'
            f'<figcaption>{description}</figcaption></figure>')


def download(stem, text, extension="csv"):
    return f'<a href="data/validation-{stem}.{extension}">{escape(text)}</a>'


def formula(label, expression):
    return (f'<div class="formula"><math display="block" aria-label="{escape(label, quote=True)}">'
            f"{expression}</math></div>")


def write_csv(output, stem, rows):
    with (output / "data" / f"validation-{stem}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(output, stem, record):
    (output / "data" / f"validation-{stem}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quality_rows(report):
    record = read(report / "quality_comparison.json")
    rows = {(int(row["k"]), row["method"], row["stratum"]): row["new"] for row in record["primary"]}
    expected = {(k, m, g) for k in KS for m in METHODS for g in GROUPS}
    if (set(rows) != expected or len(record["primary"]) != len(expected) or
            (record["test_rows"], record["cold_test_rows"]) != (4533, 147)):
        raise ValueError("Incomplete RQ-VAE product-level evaluation")
    return rows


def configuration_rows(path, selection):
    route = {key: (int(k), method) for method, name in (("beam", "beam"), (COMBINED, "winners"))
             for k, key in selection[name].items()}
    fields = ("item_hr", "item_ndcg", "alpha_eff", "mean_rounds", "mean_verified", "beam_call_rate", "mean_returned")
    with path.open(newline="", encoding="utf-8") as stream:
        return {(*route[row["id"]], row["stratum"]): {
            "rows": int(row["rows"]), "item_hits": int(row["item_hits"]),
            **{field: float(row[field]) for field in fields},
        } for row in csv.DictReader(stream) if row["id"] in route}


def timing_rows(report, selection, protocol):
    expected = len(protocol["sample_row_ids"]) * protocol["repeats"]
    if expected != 768 or protocol["latency_batch_size"] != 1 or protocol["throughput_batch_size_per_gpu"] != 4:
        raise ValueError("The latest timing must cover the fixed 256 histories in the documented batch protocol")
    by_id = {row["id"]: row for row in read(report / "benchmarks.json")}
    result = {}
    for k in KS:
        for method, key in (("beam", "beam"), (COMBINED, "winners")):
            row = by_id[selection[key][str(k)]]
            if (row["k"] != k or not all(row[label] == expected for label in
                    ("latency_samples", "logical_requests_1gpu", "logical_requests_4gpu")) or
                    row["recommendations_batch1"] != k * expected):
                raise ValueError("Timing was measured on a different route or has missing outputs")
            wall_ms = 1000 * sum(row["latency_repeat_seconds"])
            result[k, method] = {
                "k": k, "method": method,
                **{field: row[field] for field in TIMING_FIELDS},
                "items_s_1gpu": row["items_s_1gpu"],
                "recommended_sid_tokens_s_1gpu": row["recommended_sid_tokens_s_1gpu"],
                "verified_input_tokens_s_1gpu": row["verifier_input_tokens_s_1gpu"],
                "verified_input_tokens_s_4gpu": row["verifier_input_tokens_s_4gpu"],
                "batch1_peak_gib": row["gpu_peak_bytes_batch1"] / 2**30,
                "batch4_peak_gib": row["gpu_peak_bytes_batch4"] / 2**30,
                "batch1_e2e_ms_per_delivered_sid_token": wall_ms / (3 * row["recommendations_batch1"]),
                "profile_batch4_ms_per_request": row["profile_batch4_stage_ms_per_request"],
            }
    return result


def sid_statistics(index):
    catalog = list(index.values())
    if len(catalog) != 3686 or any(len(tokens) != 3 for tokens in catalog):
        raise ValueError("Expected the same 3,686-item catalog with three SID positions")
    buckets = [Counter(tuple(tokens[:n]) for tokens in catalog) for n in (1, 2, 3)]
    occupied = [len({tokens[pos] for tokens in catalog}) for pos in range(3)]
    return {
        "items": len(catalog), "unique_sids": len(buckets[-1]),
        "collision_groups": sum(n > 1 for n in buckets[-1].values()),
        "collision_items": sum(n for n in buckets[-1].values() if n > 1),
        "excess_items": len(catalog) - len(buckets[-1]),
        "slot_occupancy": occupied,
        "prefix_counts": [len(x) for x in buckets],
        "max_prefix_buckets": [max(x.values()) for x in buckets],
    }


def sid_comparison(workspace, run, output, prepared):
    old_file = workspace / ORIGINAL / CATALOG
    new_file = run / "catalog" / CATALOG
    if (digest(old_file), digest(new_file)) != (prepared["baseline_sid_sha256"], prepared["expected_sid_sha256"]):
        raise ValueError("SID comparison files disagree with frozen preparation")
    old_index, new_index = read(old_file), read(new_file)
    if set(old_index) != set(new_index):
        raise ValueError("Cannot compare SID capacity on different catalogs")
    old, new = map(sid_statistics, (old_index, new_index))
    collision = read(run / "rqvae/collision_repair.json")
    exported = read(run / "sid_export_summary.json")
    if (new["unique_sids"] != prepared["unique_sids"] or collision["model_sha256"] != exported["rqvae_model_sha256"]):
        raise ValueError("RQ-VAE SID statistics are not tied to the selected quantizer")
    variants = {name: value.get("metrics", value) for name, value in collision["metrics"].items()}
    repaired = variants["prefix_matching"]
    holdout = {key: val["warm_holdout"]["mse"] for key, val in variants.items()}
    write_json(output, "sid-audit", {"rq_kmeans": old, "rq_vae": new,
        "rq_vae_decoding": {name: {"unique_sids": record["all"]["unique_sids"], "warm_holdout_mse": holdout[name]}
                            for name, record in variants.items()}})
    axes = pyplot()
    fig, (a, b) = axes.subplots(1, 2, figsize=(9.2, 3.3), layout="constrained")
    for i, (label, row, color) in enumerate(zip(("RQ-KMeans", "RQ-VAE"), (old, new), COLORS)):
        x = [pos + (i - .5) * .34 for pos in range(3)]
        a.bar(x, row["slot_occupancy"], width=.34, color=color, label=label)
        b.bar(x, row["max_prefix_buckets"], width=.34, color=color, label=label)
    a.set(xlabel="SID position", ylabel="Distinct codes", xticks=range(3), xticklabels=["a", "b", "c"])
    b.set(xlabel="Prefix length", ylabel="Largest bucket (items)", xticks=range(3), xticklabels=[1, 2, 3])
    a.legend(frameon=False)
    save(axes, fig, output, "sid")
    fields = [
        ("目录商品", "items", None), ("唯一完整 SID", "unique_sids", max), ("完整 SID 碰撞组", "collision_groups", min),
        ("碰撞组内商品", "collision_items", min), ("超额重复编码", "excess_items", min),
        ("各位置的实际用码数", "slot_occupancy", None), ("不同前缀数（长度 1 / 2 / 3）", "prefix_counts", None),
        ("最大前缀桶（长度 1 / 2 / 3）", "max_prefix_buckets", None),
    ]
    rows = []
    for label, field, best in fields:
        values = [record[field] for record in (old, new)]
        leader = best(values) if best else None
        rows.append([label, *(mark(" / ".join(map(str, x)) if isinstance(x, list) else f"{x:,}",
                                   x == leader) for x in values)])
    return (
        "<p>在同一份商品目录上对比原 RQ-KMeans 发布索引与本轮 RQ-VAE 最终导出。"
        "超额重复编码等于商品数减去不同完整 SID 数。前缀统计用来解释候选验证中的状态复用。</p>"
        + figure("sid", "两种索引的分位置用码数及各长度的最大前缀桶。")
        + table(("目录审计", "原 RQ-KMeans", "本轮 RQ-VAE"), rows)
        + f"<p>本轮 RQ-VAE 先对训练商品向量中心化，原始最近码得到 {variants['raw']['all']['unique_sids']:,} 个完整 SID。"
          f"固定前两位并按残差距离分配第三位后，得到 {repaired['all']['unique_sids']:,} 个唯一 SID，"
          f"暖商品留出重建 MSE 由 {holdout['raw']:.6f} 变为 {holdout['prefix_matching']:.6f}。"
          "RQ-KMeans 的发布产物支持目录编码对比。其重建误差与训练中间量的归档缺失，重建 MSE 只展示本轮 RQ-VAE 的量化校验。</p>"
        + f"<p>{download('sid-audit', '下载 SID 目录与编码校验数据', 'json')}</p>"
    )


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(plt, fig, output, stem):
    for axis in fig.axes:
        axis.grid(axis="y", alpha=.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "figures" / f"validation-{stem}.png", dpi=170, facecolor="white")
    plt.close(fig)


def generation(run, output, sid_sha):
    by_stage = {}
    for key, source in (("SFT", "epochs3/sft"), ("GRPO", "rl")):
        report = read(run / source / "evaluation/metrics.json")
        if (report["status"], report["completed_rows"], report["sid_index_sha256"], report["num_beams"]) != (
                "passed", 4533, sid_sha, 50):
            raise ValueError(f"{key} validation belongs to different assets")
        by_stage[key] = {k: tuple(report["metrics"][f"{field}@{k}"] for field in ("HR", "NDCG"))
                         for k in GENERATION_KS}
    rows = []
    for k in GENERATION_KS:
        a, b = (by_stage[stage][k] for stage in ("SFT", "GRPO"))
        winner = [max(pair) for pair in zip(a, b)]
        rows.append([k, *(mark(pct(n), n == w) for group in (a, b) for n, w in zip(group, winner))])
    write_csv(output, "generation", [dict(stage=stage, k=k, hr=values[0], ndcg=values[1])
                                    for stage, v in by_stage.items() for k, values in v.items()])
    a, b = (by_stage[stage][10] for stage in ("SFT", "GRPO"))
    return (
        "<p>在相同 RQ-VAE SID 目录下，用同一测试切分和 beam 50 比较两个训练阶段的选中权重。"
        "目录中完整 SID 与商品 ID 一一对应，所以本轮 SID 命中与商品命中一致。</p>"
        + table(("K", "SFT HR", "SFT NDCG", "GRPO HR", "GRPO NDCG"), rows)
        + f"<p>从 SFT 到 GRPO，HR@10 由 {pct(a[0])} 升至 {pct(b[0])}，NDCG@10 由 {pct(a[1])} 升至 {pct(b[1])}。"
          f"此处只考察模型直接生成，联合路径的效果见下一节。{download('generation', '下载本轮生成评估结果')}</p>"
    )


def recommendation_quality(quality, output):
    rows = []
    fields = ("item_hr", "item_ndcg")
    plt = pyplot()
    fig, plots = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for method, label, color in zip(METHODS, ("Beam 50", "UniSRec", "Speculative"), COLORS):
        for axis, group, metric, unit in zip(plots, ("overall", "overall", "cold"),
                (*fields, "item_hits"), ("Exact item HR (%)", "Exact item NDCG (%)", "Cold hits / 147")):
            axis.plot(range(len(KS)), [(1 if metric == "item_hits" else 100) * quality[k, method, group][metric]
                                      for k in KS], "o-", label=label, color=color, linewidth=1.6, markersize=4)
            axis.set(xlabel="K", ylabel=unit, xticks=range(len(KS)), xticklabels=KS)
    plots[0].legend(frameon=False, fontsize=8)
    save(plt, fig, output, "quality")
    for k in KS:
        for method, label in METHODS.items():
            row = [k, label]
            for group in GROUPS:
                record = quality[k, method, group]
                if group == "cold":
                    row.append(mark(f"{record['item_hits']}/{record['rows']}", record["item_hits"] ==
                                    max(quality[k, m, group]["item_hits"] for m in METHODS)))
                row.extend(mark(pct(record[field]), record[field] == max(quality[k, m, group][field] for m in METHODS))
                           for field in fields)
            rows.append(row)
    write_csv(output, "quality", [dict(k=k, method=m, stratum=g, **row)
                                  for (k, m, g), row in sorted(quality.items())])
    a, b, c = (quality[50, m, "overall"] for m in METHODS)
    raw_cold, joint_cold = (quality[50, method, "cold"]["item_hits"]
                            for method in ("unisrec_original_rank", COMBINED))
    return (
        "<p>三个方案使用同一次运行的产物。直接生成取 GRPO beam 50 的前 K 件，UniSRec 原排取内积排序前 K 件，"
        "联合投机推理对候选 SID 打分后返回 K 件。深红粗体表示同一 K、同一分组和同一指标的最佳值，并列同样标出。</p>"
        + figure("quality", "本轮三个方案的整体商品 HR、NDCG 与冷目标命中数。")
        + table(("K", "方案", "整体 HR", "整体 NDCG", "暖 HR", "暖 NDCG", "冷命中", "冷 HR", "冷 NDCG"), rows)
        + f"<p>K=1 的商品 HR 以直接生成最高。K=5、10 的联合方案同时取得最高整体 HR 和 NDCG。"
          f"K=50 的联合方案 NDCG 为 {pct(c['item_ndcg'])}，高于 UniSRec 原排的 {pct(b['item_ndcg'])} "
          f"和直接生成的 {pct(a['item_ndcg'])}。此时 UniSRec 原排的整体 HR 更高，"
          f"冷目标命中也更高（{raw_cold} 条与 {joint_cold} 条）。"
          "重排有助于把部分命中排到前面，也会把部分候选排出最终列表。</p>"
        + f"<p>{download('quality', '下载三个方案的整体、暖组和冷组指标')}</p>"
    )


def speed_cell(timed, k, method, field, digits=2):
    value = timed[k, method][field]
    best = min if field.startswith(("latency", "batch1_e2e", "batch4_peak")) else max
    return mark(f"{value:,.{digits}f}", value == best(timed[k, route][field] for route in TIMED))


def runtime(report, asset_sha, protocol, timed, output):
    data = read(report / "tpot_rqvae.json")
    audit = read(report / "batch_audit.json")
    sample = protocol["sample_row_ids"]
    expected = len(sample) * protocol["repeats"]
    if ((data["status"], data["variant"], data["model_sha256"], data["benchmark_protocol_sha256"]) !=
            ("passed", "rqvae", asset_sha, digest(report / "benchmark_protocol.json")) or
            data["sample_row_ids"] != sample or audit["sample_row_ids"] != sample or
            data["device"] != protocol["device"] or len(data["raw"]) != expected or data["summary"]["samples"] != expected or
            data["rank_parity"]["tested"] != 24 or data["rank_parity"]["exact_full_beam_order_matches"] != 24):
        raise ValueError("The streaming-boundary measurement used different assets or samples")
    repeat_index = {(i, row) for i in range(1, protocol["repeats"] + 1) for row in sample}
    if {(row["repeat"], row["row_id"]) for row in data["raw"]} != repeat_index:
        raise ValueError("Incomplete TPOT sample coverage")
    plot = pyplot()
    fig, axes = plot.subplots(1, 2, figsize=(9.2, 3.4), layout="constrained")
    for method, label, color in zip(TIMED, ("Beam 50", "Speculative"), (COLORS[0], COLORS[2])):
        for axis, fields, ylabel in zip(axes, (("latency_p50_ms", "latency_p99_ms"), ("requests_s_4gpu",)),
                ("End-to-end latency (ms)", "Four-GPU requests/s")):
            for field in fields:
                p99 = field.endswith("p99_ms")
                axis.plot(range(len(KS)), [timed[k, method][field] for k in KS if (k, method) in timed],
                          "o--" if p99 else "o-", color=color, linewidth=1.5, markersize=3.5,
                          label=f"{label} {'P99' if p99 else 'P50'}" if len(fields) > 1 else label)
            axis.set(xlabel="K", ylabel=ylabel, xticks=range(len(KS)), xticklabels=KS)
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    save(plot, fig, output, "latency")
    rows = [[k, METHODS[method], *(speed_cell(timed, k, method, key) for key in TIMING_FIELDS)]
            for k in KS for method in TIMED]
    tokens = data["summary"]
    token_rows = [[label, *(f"{tokens[key][p]:.3f}" for p in ("mean", "p50", "p95", "p99"))]
                  for key, label in (("logical_ttft_ms", "内部首 SID token 时延（TTFT）"),
                                     ("tpot_ms", "后续 SID token 时延（TPOT）"))]
    audit_rows = [[x["k"], f"{x['set_equal']}/{x['n']}", f"{x['order_equal']}/{x['n']}",
                   f"{x['b4_online_order_equal']}/{x['n']}", f"{100*x['b1_minus_b4_ndcg']:+.3f}"]
                  for x in audit["rows"]]
    write_csv(output, "performance", [{k: v for k, v in row.items() if k != "profile_batch4_ms_per_request"}
                                      for row in timed.values()])
    write_json(output, "performance-details", list(timed.values()))
    write_json(output, "beam-token-latency", data)
    write_json(output, "batch-audit", {"sample_row_ids": sample, "results": audit["rows"]})
    return (
        f"<p>在 {escape(protocol['device'])} 上使用 {len(sample)} 条固定请求，各配置预热后重复 {protocol['repeats']} 遍。"
        "P50、P95 和 P99 是单卡 batch=1 的整条请求耗时，吞吐使用每卡 batch=4。"
        "深红粗体表示对应 K 下实测两条路线中最低的时延或最高的吞吐。UniSRec 原排用于质量对照。</p>"
        + figure("latency", "直接生成与联合投机推理的端到端分位数时延及四卡吞吐。")
        + table(("K", "路线", "P50（ms）", "P95（ms）", "P99（ms）", "单卡请求/s", "四卡请求/s",
                 "四卡商品/s", "四卡输出 token/s"), rows)
        + "<h3>逐 token 计时与整批响应口径</h3>"
          "<p>纯 beam50 在各 K 使用固定解码宽度，下面的内部 token 指标共用一次测量。TTFT 从请求准备开始"
          "计到首个 SID token 选定。TPOT 的起止点是第一个与第三个 SID token 的选定时刻，"
          "排除换行和 EOS。抽检的 24 次完整 beam 排序在打点前后相同。</p>"
        + formula("TPOT 等于第三个 SID token 选定时刻减去第一个选定时刻，再除以二",
                  '<mi mathvariant="normal">TPOT</mi><mo>=</mo><mfrac><mrow>'
                  '<msub><mi>t</mi><mn>3</mn></msub><mo>−</mo><msub><mi>t</mi><mn>1</mn></msub>'
                  '</mrow><mn>2</mn></mfrac>')
        + table(("纯 beam50 指标（ms）", "均值", "P50", "P95", "P99"), token_rows)
        + "<p>联合路径用已知候选并行打分，一次交付整个列表。其每输出 SID token 摊销时延"
          "以 batch=1 端到端总时长除以实际交付 token 数计算，每件商品计三个 token。下节列出各 K 的具体值。</p>"
          "<h3>单请求与批量复核</h3>"
          "<p>原批次按 batch=4 复跑，与归档排序逐位核对。另列 batch=1 结果与 batch=4 的商品集合和全序一致数。"
          "全序要求每个商品名次都一样，所以比仅看集合严格。</p>"
        + table(("K", "B1 与 B4 集合一致", "B1 与 B4 全序一致", "B4 与归档全序一致", "B1 减 B4 NDCG（百分点）"), audit_rows)
        + f"<p>{download('performance', '下载延迟与吞吐 CSV')} · {download('performance-details', '下载阶段耗时和显存', 'json')} · "
          f"{download('beam-token-latency', '下载逐请求 TTFT 与 TPOT', 'json')} · {download('batch-audit', '下载批量复核结果', 'json')}</p>"
    )


def prefix_timing(inference, protocol, output):
    sampled = set(protocol["sample_row_ids"])
    sids = read(inference / "catalog.json")["sid_by_item"]
    prefixes = [sid.split("><c_", 1)[0] for sid in sids]
    rows = []
    for shard in sorted((inference / "test").glob("rank[0-3].jsonl")):
        with shard.open(encoding="utf-8") as stream:
            for item in map(json.loads, stream):
                if item["row_id"] in sampled:
                    candidates = item["drafts"]["unisrec"]["50"]["first"]
                    if len(candidates) != 100:
                        raise ValueError("Unexpected draft budget or incomplete prefix diagnostic")
                    rows.append({"row_id": item["row_id"],
                                 "unique_two_token_prefixes": len({prefixes[c] for c in candidates})})
    if len(rows) != len(sampled) or {row["row_id"] for row in rows} != sampled:
        raise ValueError("Prefix diagnostic did not reuse the latency sample")
    rows.sort(key=lambda row: row["row_id"])
    write_csv(output, "prefix-sample", rows)
    sizes = [row["unique_two_token_prefixes"] for row in rows]
    return (f"<p>用相同的 {len(sampled)} 条计时历史，复查 K=50 时每条请求首轮的 100 件候选。两位前缀数的"
            f"中位数为 {median(sizes):g}，其中 {sum(n > 64 for n in sizes)} 条超过 64 个。验证器复用相同前缀，"
            f"每个验证块处理至多 64 支，超过一块时需要多次前向。{download('prefix-sample', '下载逐请求前缀数')}</p>")


def selected_metrics(quality, configured, validated, timed, output, diagnostics):
    summary, speed, workflow, public = [], [], [], []
    speed_fields = (*TIMING_FIELDS, "batch1_e2e_ms_per_delivered_sid_token", "batch4_peak_gib")
    for k in KS:
        overall, warm, cold = (quality[k, COMBINED, group] for group in GROUPS)
        diagnostic = configured[k, COMBINED, "overall"]
        key = k, COMBINED, "overall"
        row = [k, pct(validated[key]["item_ndcg"])]
        for group, item in zip(GROUPS, (overall, warm, cold)):
            row.extend(mark(pct(item[field]), item[field] == max(quality[k, m, group][field] for m in METHODS))
                       for field in ("item_hr", "item_ndcg"))
        summary.append([*row[:6],
                        mark(f"{cold['item_hits']}/{cold['rows']}", cold["item_hits"] ==
                             max(quality[k, m, "cold"]["item_hits"] for m in METHODS)), *row[6:]])
        speed.append([k, *(speed_cell(timed, k, COMBINED, key, 3 if key.startswith("batch") else 2)
                          for key in speed_fields)])
        workflow.append([k, pct(diagnostic["alpha_eff"]), f"{diagnostic['mean_rounds']:.3f}",
                         f"{diagnostic['mean_verified']:.2f}", pct(diagnostic["beam_call_rate"])])
        public.append({
            "k": k, "validation_ndcg": validated[key]["item_ndcg"], "test_hr": overall["item_hr"],
            "test_ndcg": overall["item_ndcg"], "warm_hr": warm["item_hr"], "warm_ndcg": warm["item_ndcg"],
            "cold_hits": cold["item_hits"], "cold_hr": cold["item_hr"], "cold_ndcg": cold["item_ndcg"],
            **{field: diagnostic[field] for field in ("alpha_eff", "mean_rounds", "mean_verified", "beam_call_rate")},
            **{field: timed[k, COMBINED][field] for field in speed_fields},
        })
    write_csv(output, "selected-metrics", public)
    q = public[-1]
    return (
        "<p>以下集中展示本轮预先固定的联合投机推理方案。验证集指标用于核对同一配置在测试前的表现。"
        "测试质量和实测速度沿用上文的最佳值标注规则。</p>"
        + table(("K", "验证 NDCG", "测试 HR", "测试 NDCG", "暖 HR", "暖 NDCG", "冷命中", "冷 HR", "冷 NDCG"), summary)
        + table(("K", "P50（ms）", "P95（ms）", "P99（ms）", "单卡请求/s", "四卡请求/s", "四卡商品/s",
                 "四卡输出 token/s", "单位输出摊销（ms/token）", "B4 峰值显存（GiB）"), speed)
        + table(("K", "实际检索占比", "草拟轮数/请求", "验证候选/请求", "完整 beam 补位率"), workflow)
        + f"<p>K=50 的测试 NDCG 为 {pct(q['test_ndcg'])}，中位单请求耗时 {q['latency_p50_ms']:.2f} ms，"
          f"P99 为 {q['latency_p99_ms']:.2f} ms。评估时延预算时可以同时查看典型请求和尾部请求的分位数。"
          "逐 token 的 TTFT 与 TPOT 实测适用于上节的纯 beam 路径。</p>"
        + diagnostics
        + f"<p>{download('selected-metrics', '下载选定方案的五档质量与在线性能')}</p>"
    )


def render(workspace: Path, run: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    report = run / "inference/report"
    prepared = read(run / "data/preparation_manifest.json")
    assets = read(run / "inference/asset_manifest.json")
    selection = read(report / "selection.json")
    protocol = read(report / "benchmark_protocol.json")
    completed = read(report / "comparison_status.json")
    if (completed["status"] != "passed" or completed["quality_sha256"] != digest(report / "quality_comparison.json")
            or selection["asset_manifest_sha256"] != digest(run / "inference/asset_manifest.json")
            or protocol["asset_manifest_sha256"] != selection["asset_manifest_sha256"]
            or (prepared["original_sid"], prepared["unique_sids"]) != (False, 3686)):
        raise ValueError("Latest RQ-VAE quality or performance evidence is incomplete")
    quality = quality_rows(report)
    configured = configuration_rows(report / "test.csv", selection)
    validated = configuration_rows(report / "validation.csv", selection)
    expected = {(k, m, g) for k in KS for m in TIMED for g in GROUPS}
    if set(configured) != expected or set(validated) != expected:
        raise ValueError("Frozen model configurations omit routes, cutoffs or strata")
    for key, row in configured.items():
        if row["rows"] != quality[key]["rows"] or any(not math.isclose(row[field], quality[key][field])
                for field in ("item_hr", "item_ndcg", "item_hits")):
            raise ValueError("Joined archival rows disagree on the evaluated item ranking")
    timed = timing_rows(report, selection, protocol)
    cutoff = ('<munder><mo>∑</mo><mrow><mi>i</mi><mo>:</mo><msub><mi>r</mi><mi>i</mi></msub>'
              '<mo>≤</mo><mi>K</mi></mrow></munder>')
    definitions = (
        "<p>本报告使用最新 RQ-VAE 重训的工业品实验。验证切分有 4,532 条请求，测试切分有 4,533 条，"
        "其中暖目标 4,386 条、冷目标 147 条。冷商品在训练交互的历史和目标中均未出现。</p>"
        "<p>每条历史只对应一个正确商品。设一个分组有 <var>N</var> 条请求，第 <var>i</var> 条请求的正确商品"
        "排在第 <var>r</var><sub><var>i</var></sub> 位。前 K 件找不到正确商品时，该请求的贡献记为零。"
        "HR 等于此口径下的 Recall，NDCG 对命中的名次作对数折扣。测试排名以精确商品 ID 为准。</p>"
        + formula("HR 等于前 K 件找到正确商品的请求数除以总请求数",
                  '<msub><mi mathvariant="normal">HR</mi><mi>K</mi></msub><mo>=</mo>'
                  f'<mfrac><mn>1</mn><mi>N</mi></mfrac>{cutoff}<mn>1</mn>')
        + formula("NDCG 是所有请求的名次折扣得分的平均",
                  '<msub><mi mathvariant="normal">NDCG</mi><mi>K</mi></msub><mo>=</mo>'
                  f'<mfrac><mn>1</mn><mi>N</mi></mfrac>{cutoff}'
                  '<mfrac><mn>1</mn><mrow><msub><mi mathvariant="normal">log</mi><mn>2</mn></msub>'
                  '<mo>(</mo><msub><mi>r</mi><mi>i</mi></msub><mo>+</mo><mn>1</mn><mo>)</mo></mrow></mfrac>')
    )
    sections = [
        ("evaluation", "评估对象与指标口径", definitions),
        ("sid", "RQ-VAE 与 RQ-KMeans 的编码结果", sid_comparison(workspace, run, output, prepared)),
        ("generation", "SFT 与 GRPO 的直接生成评估", generation(run, output, prepared["expected_sid_sha256"])),
        ("quality", "推荐质量与冷暖分组", recommendation_quality(quality, output)),
        ("runtime", "在线时延、吞吐与批次复核",
         runtime(report, assets["files"]["rl/model.safetensors"]["sha256"], protocol, timed, output)),
        ("selected", "所选方案在各 K 下的质量与在线性能",
         selected_metrics(quality, configured, validated, timed, output, prefix_timing(run / "inference", protocol, output))),
    ]
    return "".join(section(*item) for item in sections), [(key, title) for key, title, _ in sections]
