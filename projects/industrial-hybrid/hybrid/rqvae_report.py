"""Render the retrained RQ-VAE 2K comparison from confirmed quality and performance."""
import argparse
from html import escape
import json
from pathlib import Path

from hybrid.experiment import KS
from hybrid.rqvae_assets import MANIFEST, load_json
from hybrid.sweep import sha256, write_csv, write_json


METHODS = {
    "beam": "直接生成·Beam 50",
    "unisrec_original_rank": "UniSRec 原排",
    "unisrec_2k": "UniSRec 2K + RL 打分",
}
STRATA = {"overall": "整体", "warm": "暖目标", "cold": "冷目标"}
TIMING_FIELDS = (
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "requests_s_1gpu", "requests_s_4gpu", "verifier_input_tokens_s_1gpu",
    "verifier_input_tokens_s_4gpu", "gpu_peak_bytes_batch4",
)
PROTOCOL_KEYS = (
    "sample_row_ids", "seed", "repeats", "warmup_batches_per_shape",
    "latency_batch_size", "throughput_batch_size_per_gpu",
)


def performance(report, legacy_timing, selection, legacy_audit=None):
    current = load_json(report / "benchmark_protocol.json")
    baseline = load_json(legacy_timing / "report/benchmark_protocol.json")
    if any(current[key] != baseline[key] for key in PROTOCOL_KEYS):
        raise ValueError("Measured quality and timing require the original fixed request and repetition protocol")
    if current["asset_manifest_sha256"] != selection["asset_manifest_sha256"]:
        raise ValueError("Performance measurement differs from the quality asset identity")
    if len(current["sample_row_ids"]) != 256:
        raise ValueError("Expected 256 original timing histories")
    timings = [
        {row["id"]: row for row in load_json(directory / "benchmarks.json")}
        for directory in (legacy_timing / "report", report)
    ]
    rows = []
    for k in KS:
        for method, key in (("beam", "beam"), ("unisrec_2k", "winners")):
            ident = selection[key][str(k)]
            options = [b[ident] for b in timings]
            for measure in options:
                if (measure["k"], measure["quota"], measure["alpha"]) != (
                    k, k if method == "unisrec_2k" else 0, 1.0 if method == "unisrec_2k" else 0.0,
                ):
                    raise ValueError(f"Timing config drift for {ident}")
                expected = len(current["sample_row_ids"]) * current["repeats"]
                if any(measure[key] != expected for key in (
                    "latency_samples", "logical_requests_1gpu", "logical_requests_4gpu",
                )):
                    raise ValueError(f"Incomplete benchmark samples for {ident}")
            old, new = [{name: m[name] for name in TIMING_FIELDS} | {
                "profile_batch4_stage_ms_per_request": m["profile_batch4_stage_ms_per_request"]
            } for m in options]
            rows.append(dict(k=k, method=method, config_id=ident, old=old, new=new))
    audits = {}
    for side, folder in (("old", (legacy_audit or legacy_timing) / "report"), ("new", report)):
        audit = load_json(folder / "batch_audit.json")
        if side == "old" and any(
            row["config_id"] != selection["winners"][str(row["k"])] for row in audit["rows"]
        ):
            # The published comparison measured some final 1K champions. Their
            # audit is never used for the 2K primary result.
            continue
        if audit["sample_row_ids"] != current["sample_row_ids"] or any(
            row["config_id"] != selection["winners"][str(row["k"])] for row in audit["rows"]
        ):
            raise ValueError(f"Batch shape audit differs from the fixed 2K selection ({side})")
        audits[side] = [{key: row[key] for key in (
            "k", "config_id", "n", "set_equal", "order_equal", "b4_online_order_equal",
            "b1_minus_b4_hr", "b1_minus_b4_ndcg",
        )} for row in audit["rows"]]
    result = {
        "schema": 1, "protocol": {key: current[key] for key in PROTOCOL_KEYS},
        "devices": {"old": baseline["device"], "new": current["device"]},
        "frameworks": {"old": baseline["framework"], "new": current["framework"]},
        "rows": rows, "batch_audit": audits,
        "baseline_benchmark_sha256": sha256(legacy_timing / "report/benchmarks.json"),
    }
    write_json(report / "performance_comparison.json", result)
    write_csv(report / "performance_comparison.csv", [
        dict(k=row["k"], method=row["method"], config_id=row["config_id"],
             **{f"{side}_{field}": row[side][field] for side in ("old", "new") for field in TIMING_FIELDS})
        for row in rows
    ])
    return result


def table(headers, rows):
    head = "".join(f"<th scope='col'>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def percentage(value):
    return f"{value * 100:.3f}%"


def paired_label(row, field):
    low, high = row[f"{field}_ci95_pp"]
    return f"{row[f'{field}_delta_pp']:+.3f} [{low:+.3f}, {high:+.3f}]"


def make_charts(report, quality, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = report / "figures"
    figure_dir.mkdir(exist_ok=True)
    grouped = {(r["k"], r["method"], r["stratum"]): r for r in quality["primary"]}
    fig, panels = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    indexes = np.arange(len(KS))
    labels = ("Beam 50", "UniSRec rank", "UniSRec 2K")
    for axis, field, title in zip(panels, ("item_hr", "item_ndcg", "item_hits"),
                                  ("Exact item HR (%)", "Exact item NDCG (%)", "Cold target hits / 147")):
        segment = "cold" if field == "item_hits" else "overall"
        scale = 1 if field == "item_hits" else 100
        for method, name, color in zip(METHODS, labels, ("#0f607c", "#9c4921", "#008878")):
            for variant, style in (("old", "--"), ("new", "-")):
                axis.plot(indexes, [scale * grouped[k, method, segment][variant][field] for k in KS],
                          label=f"{name} {'RQ-VAE' if variant == 'new' else 'RQ-KMeans'}",
                          linestyle=style, color=color, marker="o", markersize=3.2, linewidth=1.3)
        axis.set(title=title, xlabel="K", xticks=indexes, xticklabels=KS)
        axis.grid(axis="y", alpha=.2)
    fig.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(figure_dir / "rqvae-quality.png", dpi=160, facecolor="white")
    plt.close(fig)

    timed = {(r["k"], r["method"]): r for r in timing["rows"]}
    fig, panels = plt.subplots(1, 2, figsize=(9.4, 3.4), layout="constrained")
    for axis, field, title in zip(panels, ("latency_p50_ms", "requests_s_4gpu"),
                                  ("B1 latency P50 (ms)", "Four-GPU throughput (requests/s)")):
        for method, name, color in (("beam", "Beam 50", "#0f607c"), ("unisrec_2k", "UniSRec 2K", "#008878")):
            for variant, style in (("old", "--"), ("new", "-")):
                axis.plot(indexes, [timed[k, method][variant][field] for k in KS], color=color, linestyle=style,
                          marker="o", label=f"{name} {'RQ-VAE' if variant == 'new' else 'RQ-KMeans'}")
        axis.set(title=title, xlabel="K", xticks=indexes, xticklabels=KS)
        axis.grid(axis="y", alpha=.2)
    fig.legend(loc="outside lower center", ncols=2, frameon=False, fontsize=8)
    fig.savefig(figure_dir / "rqvae-timing.png", dpi=160, facecolor="white")
    plt.close(fig)


def render_section(report, selection, quality, paired, timing, assets):
    data = {(r["k"], r["method"], r["stratum"]): r for r in quality["primary"]}
    pairs = {(r["k"], r["method"], r["stratum"]): r for r in paired["comparisons"]
             if r["comparison"] == "rqvae_minus_rqkmeans"}
    rows = []
    for k in KS:
        for method in METHODS:
            key = k, method, "overall"
            sides = [data[key][variant] for variant in ("old", "new")]
            cold = [data[k, method, "cold"][variant]["item_hits"] for variant in ("old", "new")]
            rows.append(
                [k, METHODS[method], *(percentage(m["item_hr"]) for m in sides),
                 *(percentage(m["item_ndcg"]) for m in sides), paired_label(pairs[key], "ndcg"), *cold]
            )
    cal = selection["k1_validation_calibration"]
    res_cal = next(r for r in quality["calibrated_k1"] if r["stratum"] == "overall")
    fixed_cal = data[1, "unisrec_2k", "overall"]["new"]
    final = data[50, "unisrec_2k", "overall"]
    final_cold = data[50, "unisrec_2k", "cold"]
    pair50 = pairs[50, "unisrec_2k", "overall"]
    new_raw50 = data[50, "unisrec_original_rank", "overall"]["new"]
    benchmark_rows = [
        [r["k"], METHODS[r["method"]],
         *(f"{r[side][field]:.2f}" for field in TIMING_FIELDS[:5] for side in ("old", "new"))]
        for r in timing["rows"]
    ]
    audit = timing["batch_audit"]["new"]
    audit_rows = [[r["k"], "RQ-VAE" if side == "new" else "RQ-KMeans",
                   f"{r['set_equal']}/{r['n']}", f"{r['order_equal']}/{r['n']}",
                   f"{r['b4_online_order_equal']}/{r['n']}", f"{r['b1_minus_b4_ndcg']*100:+.3f}"]
                  for side, rows in timing["batch_audit"].items() for r in rows]
    no_b4_drift = all(r["b4_online_order_equal"] == r["n"] for r in audit)
    hardware = "同一型号 GPU 和 PyTorch 版本" if (
        timing["devices"]["old"] == timing["devices"]["new"]
        and timing["frameworks"]["old"] == timing["frameworks"]["new"]
    ) else "GPU 或 PyTorch 版本存在差异，时延与吞吐按各自运行环境解读"
    return (
        "<section><h2 id='rqvae-quality'>同一请求上的商品级对照</h2>"
        f"<p>新方案用 RQ-VAE 三位 SID，共 {assets['unique_sids']:,} 个唯一 SID。"
        f"新 RL 选中 {escape(assets['rl_selected_checkpoint'])}，UniSRec 使用第 {assets['unisrec_selected_epoch']} 轮导出。"
        f"每组测试均含 {quality['test_rows']:,} 条目标，其中 {quality['cold_test_rows']} 条冷目标。"
        "原始与重训方案逐请求对齐商品历史、目标和冷暖标记。所有模型全链重训后的变化共同计入差值。</p>"
        "<p>主方案对五个 K 均固定 UniSRec 每轮取 2K，α=1，前缀宽度1，最多3轮。"
        "K=1 沿用原阈值 −4.032527，其余四档采用全接受。生成基线为同一版本 RL 的受约束 beam50。"
        "UniSRec 原排直接取前 K 件，用于观察检索与后续重排各自的影响。</p>"
        "<figure><img src='figures/rqvae-quality.png' alt='RQ-VAE 重训后各 K 的 HR、NDCG 与冷目标命中'>"
        "<figcaption>虚线对应原 RQ-KMeans 方案，实线对应重训 RQ-VAE 方案。</figcaption></figure>"
        + table(["K", "方案", "旧 HR", "新 HR", "旧 NDCG", "新 NDCG",
                 "NDCG Δ百分点［95%区间］", "旧冷命中/147", "新冷命中/147"], rows)
        + f"<p>K=50 的最终 2K 方案从 {percentage(final['old']['item_hr'])} HR 和 "
        f"{percentage(final['old']['item_ndcg'])} NDCG 变为 {percentage(final['new']['item_hr'])} 与 "
        f"{percentage(final['new']['item_ndcg'])}。逐请求新增 {pair50['gained_hits']} 条命中，"
        f"丢失 {pair50['lost_hits']} 条。冷命中为 {final_cold['old']['item_hits']}→{final_cold['new']['item_hits']}。"
        f"新 UniSRec 不经重排时 HR@50 为 {percentage(new_raw50['item_hr'])}，NDCG@50 为 "
        f"{percentage(new_raw50['item_ndcg'])}。</p>"
        "<p>区间以同一请求的排名结果做 10,000 次配对 bootstrap。暖组与冷组的完整指标可查看附件。"
        "生成得分依据完整 SID。共享 SID 的商品经目录映射后，评估仍按精确商品 ID。</p>"
        "<p><a href='quality_comparison.csv'>整体与分组质量 CSV</a> · "
        "<a href='paired_comparison.json'>配对区间与命中差分</a> · "
        "<a href='validation.csv'>冻结方案的验证指标</a></p></section>"
        "<section><h2 id='rqvae-threshold'>K=1 阈值敏感性</h2>"
        f"<p>使用与旧校准相同的 512 条验证历史。以原 UniSRec 首轮候选的拒绝比例 "
        f"{percentage(cal['old_rejection_rate'])} 在新模型上取对应分位数，得到阈值 {cal['threshold']:.6f}，"
        f"新验证样本上的实际首轮拒绝比例为 {percentage(cal['observed_new_rejection_rate'])}。"
        f"固定阈值的测试 HR@1 为 {percentage(fixed_cal['item_hr'])}，校准对照为 "
        f"{percentage(res_cal['item_hr'])}。此对照以同一验证样本事先锁定，并列在主对照之外。</p>"
        "<p><a href='selection.json'>冻结参数与校准请求清单</a></p></section>"
        "<section><h2 id='rqvae-performance'>在线时延与吞吐</h2>"
        f"<p>原方案使用 {escape(timing['devices']['old'])}，新方案使用 {escape(timing['devices']['new'])}。{hardware}。"
        "固定 256 条历史，每种配置预热后运行3轮。时延是单卡 batch=1 的端到端毫秒，吞吐采用每卡 batch=4。"
        "阶段分析和显存统计包含在计时附件中。</p>"
        "<figure><img src='figures/rqvae-timing.png' alt='各 K 两个版本的单请求时延和四卡吞吐'>"
        "<figcaption>计时覆盖从用户历史输入到返回商品排名的整条路径。</figcaption></figure>"
        + table(["K", "路线", "旧 P50", "新 P50", "旧 P95", "新 P95", "旧 P99", "新 P99",
                 "旧单卡请求/s", "新单卡请求/s", "旧四卡请求/s", "新四卡请求/s"], benchmark_rows)
        + f"<p>按归档的原始 batch=4 分组重放，{'所有抽样配置的排序一致' if no_b4_drift else '部分请求的排序发生变化'}。"
        "另比较 batch=1 与 batch=4 的商品集合、完全排序一致比例及 NDCG，差值单位为百分点。</p>"
        + table(["K", "SID 来源", "集合一致", "排序一致", "B4 与归档排序一致", "B1 减 B4 NDCG"], audit_rows)
        + "<p><a href='performance_comparison.csv'>时延、吞吐与显存对照 CSV</a> · "
        "<a href='performance_comparison.json'>含各推理阶段耗时的详细对照</a> · "
        "<a href='batch_audit.json'>单请求与批量排序审计</a></p></section>"
    )


def render(run_dir, assets, legacy_timing, legacy_audit=None):
    report = run_dir / "report"
    status = load_json(report / "quality_status.json")
    selection = load_json(report / "selection.json")
    if (status["status"] != "passed" or status["asset_manifest_sha256"] != sha256(assets / MANIFEST)
            or status["asset_manifest_sha256"] != selection["asset_manifest_sha256"]):
        raise ValueError("Cannot publish a comparison without completed paired quality on the same prepared assets")
    quality = load_json(report / "quality_comparison.json")
    paired = load_json(report / "paired_comparison.json")
    timing = performance(report, legacy_timing, selection, legacy_audit)
    make_charts(report, quality, timing)
    body = render_section(report, selection, quality, paired, timing, load_json(assets / MANIFEST))
    page = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>工业品 RQ-VAE 重训与 UniSRec 2K 对照</title>"
        "<style>body{font:16px/1.62 system-ui,sans-serif;color:#18232d;background:#f3f6f7;margin:0}"
        "main{max-width:1180px;margin:auto;background:white;padding:clamp(15px,4vw,45px)}"
        "h1,h2{color:#172d3a}h2{border-top:1px solid #cfdbdc;padding-top:1rem}"
        "p{max-width:94ch}a{color:#006960}figure{margin:1rem 0}img{max-width:100%;height:auto}"
        ".table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}"
        "td,th{text-align:right;padding:.35rem .55rem;border-bottom:1px solid #cfdbdc}"
        "th{background:#e7f1ef}td:first-child,th:first-child{text-align:left}"
        "@media print{body,main{background:white;padding:0}.table-wrap{overflow:visible}table{font-size:8pt}}"
        "</style></head><body><main><h1>工业品 RQ-VAE 重训与 UniSRec 2K 对照</h1>"
        "<nav aria-label='目录'><ol><li><a href='#rqvae-quality'>商品级质量</a></li>"
        "<li><a href='#rqvae-threshold'>K=1 阈值敏感性</a></li>"
        "<li><a href='#rqvae-performance'>时延与吞吐</a></li></ol></nav>"
        + body + "</main></body></html>"
    )
    (report / "comparison.html").write_text(page, encoding="utf-8")
    (report / "comparison-section.html").write_text(body, encoding="utf-8")
    write_json(report / "comparison_status.json", {
        "status": "passed", "quality_sha256": sha256(report / "quality_comparison.json"),
        "performance_sha256": sha256(report / "performance_comparison.json"),
        "comparison_html_sha256": sha256(report / "comparison.html"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run-dir", "assets", "legacy-timing"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--legacy-audit", type=Path,
                        help="Original 2K run when timing comes from the later K-versus-2K comparison")
    args = parser.parse_args()
    render(args.run_dir, args.assets, args.legacy_timing, args.legacy_audit)
    print(json.dumps({"report": str(args.run_dir / "report/comparison.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
