"""Create the offline RQ-VAE vs released RQ-KMeans end-to-end comparison."""

import argparse
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean, median
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from hybrid.rho_bundle import digest, publish
from project_report.__main__ import validate_documents
from project_report.provenance import BASELINE, COMPARISON, RL, SFT, UNISREC, UNI_EXPORT
from project_report.training import figure as training_figure, save, table


NAME = "Industrial-2018-RQVAE-vs-RQKMeans-2K"
KS = (1, 5, 10, 20, 50)
GENERATION_KS = (1, 3, 5, 10, 20, 50)
OLD = "#236782"
NEW = "#ad531d"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.3f}%"


def comparison_chart(root, name, panels, data):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(panels), figsize=(4.15 * len(panels), 3.0),
                               layout="constrained", squeeze=False)
    for axis, (key, ylabel, xlabel) in zip(axes[0], panels, strict=True):
        for label, color, rows in data:
            x, y = zip(*rows[key])
            axis.plot(x, y, label=label, color=color, linewidth=1.2)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend(frameon=False)
    save(figure, root, name)


def quantization(run, output):
    a = run / "rqvae_reference"
    b = run / "rqvae"
    original, chosen = (load(p / "training_summary.json") for p in (a, b))
    repair = load(b / "collision_repair.json")
    exported = load(run / "sid_export_summary.json")
    if chosen["model_sha256"] != exported["rqvae_model_sha256"] or chosen["model_sha256"] != repair["model_sha256"]:
        raise ValueError("Quantizer training, diagnosis, and exported SID must use the same checkpoint")
    curves = []
    for label, color, folder in (("Original recipe", OLD, a), ("Mean-centered", NEW, b)):
        history = load(folder / "evaluation_history.json")
        curves.append((label, color, {
            "mse": [(x["epoch"], x["val"]["mse"]) for x in history],
            "coverage": [(x["epoch"], 100 * (1 - x["all"]["collision_rate"])) for x in history],
        }))
    comparison_chart(output, "RQ-VAE", (("mse", "Warm holdout MSE", "Epoch"),
                                       ("coverage", "Unique SID share (%)", "Epoch")), curves)
    decisions = [
        ["原 RQ-KMeans 发布索引", "—", "—", exported["baseline_unique_sids"],
         exported["baseline_collisions"]["excess_items"], "—"],
        ["RQ-VAE 原配方", original["epochs_completed"], original["best_epoch"],
         original["selected_metrics"]["all"]["unique_sids"],
         exported["catalog_items"] - original["selected_metrics"]["all"]["unique_sids"],
         f'{original["best_val_mse"]:.5f}'],
    ]
    checks = repair["metrics"]
    for key, title in (("raw", "中心化 RQ-VAE 原始三位"), ("sinkhorn", "末层 Sinkhorn"),
                       ("prefix_matching", "前缀固定的第三位匹配")):
        item = checks[key] if key == "raw" else checks[key]["metrics"]
        decisions.append([title, chosen["epochs_completed"], chosen["best_epoch"], item["all"]["unique_sids"],
                          item["all"]["excess_items"], f'{item["warm_holdout"]["mse"]:.5f}'])
    with (output / "data/quantization.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scheme", "epochs", "selected_epoch", "unique_sids", "excess_items", "warm_holdout_mse"))
        writer.writerows(decisions)
    if exported["unique_sids"] != checks["prefix_matching"]["metrics"]["all"]["unique_sids"]:
        raise ValueError("Reported code repair was not used in the downstream catalog")
    occupied = "、".join(map(str, exported["occupied_codes_by_level"]))
    return (
        '<p>两组使用相同的 2,560 维原商品向量、三层 256 码本、32 维潜空间与训练配方。'
        '模型只在 3,280 件训练可见商品上拟合，367 件可见商品用于按重建 MSE 选择权重。'
        '39 件未在推荐训练交互出现的冷商品在权重冻结后进入编码。</p>'
        + training_figure("RQ-VAE", "两个 RQ-VAE 训练组的重建和码覆盖。曲线中的 SID 使用各层原始最近邻。")
        + table(("方案", "训练轮数", "选中轮", "唯一 SID", "超额重复", "量化验证 MSE"), decisions)
        + '<p>直接沿用原配方时，同一编码覆盖过多商品。中心化使用拟合子集的均值去除共享向量分量，'
          '在重建时加回该均值。导出保留前两位，仅把每个前缀内发生冲突的商品分配到不同第三码，'
          '以总残差距离最小为目标。对照的 Sinkhorn 也只修改第三位。'
          '采用最小残差匹配由量化验证集与目录诊断决定，推荐测试结果未参与。</p>'
        + f'<p>新索引三层分别实际使用 {occupied} 个码。商品最终仍用三个 token，完整索引的商品级编码唯一。</p>'
          '<p><a href="data/quantization.csv">下载第三位分配与重建误差对照</a></p>'
    )


def generative_item_metrics(workspace, run, output):
    prior = load(workspace / BASELINE / "catalog.json")
    sid_file = run / "catalog/Industrial_and_Scientific.index.json"
    new_index = load(sid_file)
    old_sids = prior["sid_by_item"]
    new_sids = ["".join(new_index[str(i)]) for i in range(len(old_sids))]
    with (run / "data/test.csv").open(newline="", encoding="utf-8-sig") as stream:
        targets = [int(row["item_id"]) for row in csv.DictReader(stream)]
    cold = prior["cold"]
    observations = []
    for stage, name, directory, sids in (
        ("SFT", "RQ-KMeans", workspace / SFT, old_sids), ("SFT", "RQ-VAE", run / "epochs3/sft", new_sids),
        ("GRPO", "RQ-KMeans", workspace / RL, old_sids), ("GRPO", "RQ-VAE", run / "rl", new_sids),
    ):
        preferred = {}
        for item in sorted(range(len(sids)), key=lambda i: (-prior["target_freq"][i], i)):
            preferred.setdefault(sids[item], item)
        ranks = {}
        with (directory / "evaluation/predictions.jsonl").open(encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                index = row["row_id"]
                item = targets[index]
                if index in ranks or row["target"] != sids[item]:
                    raise ValueError("Beam predictions use a different test row or SID")
                result = [preferred[sid] for sid in row["predictions"]]
                ranks[index] = result.index(item) + 1 if item in result else math.inf
        if set(ranks) != set(range(len(targets))):
            raise ValueError("Beam predictions omit test rows")
        for k in GENERATION_KS:
            for group in ("overall", "cold", "warm"):
                group_ranks = [rank for row, rank in ranks.items()
                               if group == "overall" or cold[targets[row]] == (group == "cold")]
                hit_ranks = [rank for rank in group_ranks if rank <= k]
                n = len(group_ranks)
                observations.append(dict(stage=stage, sid=name, k=k, group=group, rows=n, hits=len(hit_ranks),
                                         hr=len(hit_ranks)/n, ndcg=sum(1/math.log2(x+1) for x in hit_ranks)/n))
    with (output / "data/generative-item.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    show = [[r["stage"], r["sid"], r["k"], pct(r["hr"]), pct(r["ndcg"])]
            for r in observations if r["group"] == "overall"]
    return (
        "<h3>训练检查点按精确商品 ID 复算</h3>"
        "<p>把相同测试行的 beam SID 映射回商品，旧 SID 冲突时沿用训练目标频次、商品 ID 的映射规则。"
        "该口径用于判断 SID 指标变化是否意味着商品级变化。</p>"
        + table(("阶段", "SID", "K", "商品 HR", "商品 NDCG"), show)
        + '<p><a href="data/generative-item.csv">下载生成排名的六档整体与冷暖组 CSV</a></p>'
    )


def training(workspace, run, output):
    post = load(run / "posttraining_status.json")
    uni = load(run / "unisrec/status.json")
    if post["status"] != "completed" or uni["stage"] != "completed":
        raise ValueError("Post-training and UniSRec must finish before producing the comparison")
    series = {"sft": [], "rl": []}
    table_rows = []
    task_count_rows = []
    for stage, title, legacy, current in (
        ("sft", "线性三轮 SFT", workspace / SFT, run / "epochs3/sft"),
        ("rl", "GRPO β=0.001", workspace / RL, run / "rl"),
    ):
        original, rerun = (load(directory / "run_config.json") for directory in (legacy, current))
        if original["hyperparameters"] != rerun["hyperparameters"]:
            raise ValueError(f"{stage} hyperparameters changed across SID runs")
        for task, size in original["task_counts"].items():
            observed = rerun["task_counts"][task]
            if observed != size:
                task_count_rows.append([title, "SID → 标题" if task == "sid2title" else task, size, observed])
        keys = (("loss", "eval_loss") if stage == "sft" else
                ("rewards/rule_reward", "kl", "grad_norm", "eval_ndcg10"))
        for label, color, directory in (("RQ-KMeans SID", OLD, legacy), ("RQ-VAE SID", NEW, current)):
            summary = load(directory / "training_summary.json")
            history = load(directory / "trainer_state.json")["log_history"]
            metrics = load(directory / "evaluation/metrics.json")["metrics"]
            steps = {key: [(row["step"], float(row[key])) for row in history if "step" in row and key in row]
                     for key in keys}
            series[stage].append((label, color, steps))
            scores = [pct(metrics[name]) for name in ("HR@10", "NDCG@10", "HR@50", "NDCG@50")]
            selected = Path(summary["best_checkpoint"]).name
            table_rows.append([title, label, summary["global_step"], selected, summary["stop_reason"],
                               f'{summary["train_metrics"]["train_runtime"] / 60:.2f}', *scores])
    with (output / "data/posttraining.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stage", "sid", "steps", "checkpoint", "stop_reason", "minutes",
                         "sid_hr10", "sid_ndcg10", "sid_hr50", "sid_ndcg50"))
        writer.writerows(table_rows)
    comparison_chart(output, "sft",
                     (("loss", "Training loss", "SFT step"), ("eval_loss", "Validation loss", "SFT step")), series["sft"])
    comparison_chart(output, "grpo", (("rewards/rule_reward", "Rule reward", "Optimizer step"),
                                     ("kl", "KL", "Optimizer step"), ("grad_norm", "Grad norm", "Optimizer step")),
                     series["rl"])
    comparison_chart(output, "grpo-rank", (("eval_ndcg10", "Validation SID NDCG@10", "Optimizer step"),), series["rl"])
    common_rl = min(timeline["kl"][-1][0] for _, _, timeline in series["rl"])
    rl_end = []
    for label, _, timeline in series["rl"]:
        stats = [mean(value for step, value in timeline[key] if common_rl - 100 < step <= common_rl)
                 for key in ("rewards/rule_reward", "kl", "grad_norm")]
        rl_end.append((label, *(f"{value:.5f}" for value in stats)))
    uni_old = load(workspace / UNISREC / "model_selection.json")
    old_vector = load(workspace / UNISREC / "test_vector_quality/report.json")["summary"]
    new_vector = load(run / "unisrec/test/report.json")["summary"]
    old_export = load(workspace / UNI_EXPORT / "metadata.json")
    new_export = load(run / "unisrec/export/metadata.json")
    vectors_equal = old_export["checksums"]["item_vectors.npy"] == new_export["checksums"]["item_vectors.npy"]
    recall_rows = [[k, pct(old_vector[f"tie_expected_recall@{k}_mean"]),
                    pct(new_vector[f"tie_expected_recall@{k}_mean"])] for k in (10, 20, 50)]
    uni_data = []
    for label, color, log in (("RQ-KMeans SID", OLD, workspace / UNISREC / "train.log"),
                              ("RQ-VAE SID", NEW, run / "unisrec/train.log")):
        text = log.read_text(encoding="utf-8")
        pattern = r"epoch (\d+) (?:training|evaluating) \[time: [\d.]+s, (train loss|valid_score): ([\d.]+)"
        points = {"train loss": [], "valid_score": []}
        for epoch, metric, value in re.findall(pattern, text):
            points[metric].append((int(epoch) + 1, float(value)))
        uni_data.append((label, color, points))
    comparison_chart(output, "unisrec", (("train loss", "Train batch-CE sum", "Epoch"),
                                         ("valid_score", "Validation NLL", "Epoch")), uni_data)
    return (
        '<p>SFT 沿用独立线性三轮组、标题历史任务和验证 loss 选点。GRPO 从本次 SFT 选中权重继续训练，'
        '保持 β=0.001、原奖励和验证 NDCG@10 选点。两个阶段的归档超参数字段逐项一致。'
        'SID 变化会改变词表和按 SID 去重的任务数。'
        '下表的命中单位是当前各自的 SID，共用编码会影响商品级解释。</p>'
        + table(("阶段", "样本数变化的任务", "原样本数", "新样本数"), task_count_rows)
        + table(("阶段", "SID", "步数", "选中检查点", "停止原因", "训练分钟", "SID HR@10", "SID NDCG@10",
                 "SID HR@50", "SID NDCG@50"), table_rows)
        + '<p><a href="data/posttraining.csv">下载 SFT 和 GRPO 对照</a></p>'
        + generative_item_metrics(workspace, run, output)
        + training_figure("sft", "训练损失和验证损失对照，按优化步数。")
        + training_figure("grpo", "逐步奖励、KL 和裁剪前梯度范数。")
        + training_figure("grpo-rank", "GRPO 在原验证切分上的 SID 排名。")
        + f'<p>下表取两组共同训练区间的末 100 个优化步，即第 {common_rl - 99} 到 {common_rl} 步。'
          'Rule Reward 反映 rollout 的 SID 命中，KL 衡量相对参考策略的偏离，Grad Norm 为裁剪前更新强度。</p>'
        + table(("SID", "Rule Reward 均值", "KL 均值", "Grad Norm 均值"), rl_end)
        + f'<p>UniSRec 重训沿用 20 轮上限、相同的商品向量和交互，按验证目录 NLL 选点。'
          f'原方案选第 {uni_old["best_epoch"]} 轮，新方案选第 {uni["best_epoch"]} 轮'
          f'，新方案验证 NLL 为 {uni["validation_nll"]:.6f}。'
          '它的训练和原排按商品 ID，SID 参与后续联合推理的前缀筛选与模型评分。</p>'
        + training_figure("unisrec", "两次 UniSRec 训练采用同一商品输入。左图为逐轮训练批损失之和，右图为验证目录 NLL。")
        + table(("K", "原同分期望 Recall", "新同分期望 Recall"), recall_rows)
        + f'<p>两次导出的商品向量文件 SHA-256 {"相同" if vectors_equal else "不同"}。'
          f'测试目录 NLL 从 {old_vector["vector_nll"]:.8f} 变为 {new_vector["vector_nll"]:.8f}。'
          '同分期望指标来自独立 UniSRec 评估，联合推理使用实际返回的有序商品列表。</p>'
    )


def token_latency(workspace, inference, output, selection, before, after):
    from hybrid.rqvae_tpot import b1_amortization

    report = inference / "report"
    historical = workspace / COMPARISON
    old_model = load(historical / "manifest.json")["verifier"]["files"]["model.safetensors"]["sha256"]
    new_model = load(inference / "asset_manifest.json")["files"]["rl/model.safetensors"]["sha256"]
    old_protocol = historical / "report/benchmark_protocol.json"
    new_protocol = report / "benchmark_protocol.json"
    expected_rows = load(old_protocol)["sample_row_ids"]
    if expected_rows != load(new_protocol)["sample_row_ids"]:
        raise ValueError("Benchmarks did not compare the same histories")
    summaries = []
    for variant, label, model, protocol in (
        ("rqkmeans", "原 RQ-KMeans SID", old_model, old_protocol),
        ("rqvae", "新 RQ-VAE SID", new_model, new_protocol),
    ):
        name = f"tpot_{variant}.json"
        source = report / name
        data = load(source)
        if (data["status"], data["variant"], data["full_beam_width"], data["summary"]["samples"],
                data["model_sha256"], data["benchmark_protocol_sha256"], data["sample_row_ids"]) != (
                "passed", variant, 50, 768, model, digest(protocol), expected_rows):
            raise ValueError(f"TPOT did not use the benchmark configuration: {variant}")
        expected = {(i, row) for i in range(1, 4) for row in expected_rows}
        if len(data["raw"]) != len(expected) or {(r["repeat"], r["row_id"]) for r in data["raw"]} != expected:
            raise ValueError("TPOT raw timings omit requests or repeats")
        parity = data["rank_parity"]
        if parity["tested"] != 24 or parity["exact_full_beam_order_matches"] != 24:
            raise ValueError("TPOT instrumentation changed beam ranking")
        timing = data["summary"]["tpot_ms"]
        summaries.append([label, *(f'{timing[p]:.3f}' for p in ("mean", "p50", "p95", "p99"))])
        shutil.copyfile(source, output / "data" / name)
    all_rows, displayed = [], []
    for arm in ("beam", "winners"):
        for k in KS:
            case = selection[arm][str(k)]
            values = [b1_amortization(bench[case]) for bench in (before, after)]
            all_rows.extend(dict(route=arm, k=k, version=version, **record)
                            for version, record in zip(("rqkmeans", "rqvae"), values, strict=True))
            if arm == "winners":
                displayed.append([k, *(f'{r["mean_ms_per_delivered_sid_token"]:.3f}' for r in values)])
    columns = tuple(dict.fromkeys(key for row in all_rows for key in row))
    with (output / "data/token-amortized-e2e.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    return (
        '<h3>TPOT 与联合推理的单位输出时延</h3>'
        '<p>纯 beam50 的 TPOT 取内部第三个 SID token 选中时刻减第一个 SID token 选中时刻，再除以 2。'
        '两组各用同一批 256 条历史重复 3 次，batch=1。计时排除换行和 EOS 模型前向。'
        '最终 K 只控制从固定 beam50 中截取的结果数，因此五档共享实测解码 TPOT。每组抽检 24 次，'
        '打点前后的完整 beam 排序逐项一致。</p>'
        + table(("纯 beam50", "平均 TPOT（ms）", "P50（ms）", "P95（ms）", "P99（ms）"), summaries)
        + '<p>UniSRec-2K 先检索和批量验证，再一次返回商品列表。下表把未插入计时打点的 batch=1 端到端总时长'
          '除以实际交付的 SID token 总数，每件商品计三个 token。它描述检索和打分成本由多少个输出分摊，'
          '与上表的逐 token 解码时间是不同口径。</p>'
        + table(("K", "原每输出 SID token 摊销（ms）", "新每输出 SID token 摊销（ms）"), displayed)
        + '<p><a href="data/tpot_rqkmeans.json">原 TPOT 逐请求数据</a> · '
          '<a href="data/tpot_rqvae.json">新 TPOT 逐请求数据</a> · '
          '<a href="data/token-amortized-e2e.csv">各路线单位输出摊销 CSV</a></p>'
    )


def k50_prefix_diagnostics(workspace, inference, output):
    ids = set(load(inference / "report/benchmark_protocol.json")["sample_row_ids"])
    versions = []
    for root in (workspace / BASELINE, inference):
        catalog = load(root / "catalog.json")["sid_by_item"]
        prefix_ids = [sid.split("><c_", 1)[0] for sid in catalog]
        ranks = {}
        for shard in sorted((root / "test").glob("rank[0-3].jsonl")):
            with shard.open(encoding="utf-8") as stream:
                for row in map(json.loads, stream):
                    i = row["row_id"]
                    if i in ids:
                        ranks[i] = row["drafts"]["unisrec"]["50"]["first"]
        if set(ranks) != ids or any(len(items) != 100 for items in ranks.values()):
            raise ValueError("Incomplete Top100 prefix comparison for K=50")
        sizes = {i: len({prefix_ids[j] for j in items}) for i, items in ranks.items()}
        versions.append((ranks, sizes))
    (old, a), (new, b) = versions
    matching = sum(old[i] == new[i] for i in ids)
    with (output / "data/k50-prefixes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("row_id", "same_candidate_order", "old_unique_two_token_prefixes", "new_unique_two_token_prefixes"))
        writer.writerows((i, old[i] == new[i], a[i], b[i]) for i in sorted(ids))
    above = [sum(n > 64 for n in values.values()) for values in (a, b)]
    return (
        f'<p>K=50 的同一批 {len(ids)} 条计时历史中，{matching} 条的 UniSRec 前 100 件候选及顺序完全一致。'
        f'候选的不同两位前缀中位数从 {median(a.values()):g} 降到 {median(b.values()):g}，'
        f'超过 64 个前缀的请求从 {above[0]} 条降至 {above[1]} 条。'
        '验证器会复用相同前缀的计算，每个验证块最多 64 支。新 SID 需要跨块的请求更少，'
        '这与 K=50 的 P50 降幅大于 P95 和四卡吞吐的变化一致。'
        '<a href="data/k50-prefixes.csv">下载逐请求前缀统计</a></p>'
    )


def validation(workspace, inference, output):
    report = inference / "report"
    verification = load(report / "comparison_status.json")
    if verification["status"] != "passed" or any(digest(report / name) != verification[field] for name, field in (
        ("quality_comparison.json", "quality_sha256"), ("performance_comparison.json", "performance_sha256"),
        ("comparison.html", "comparison_html_sha256"),
    )):
        raise ValueError("Validation results changed after inference report verification")
    quality = load(report / "quality_comparison.json")
    paired = load(report / "paired_comparison.json")
    selection = load(report / "selection.json")
    bench = {item["id"]: item for item in load(report / "benchmarks.json")}
    old_bench = {item["id"]: item for item in load(workspace / COMPARISON / "report/benchmarks.json")}
    if len(quality["primary"]) != 45 or quality["test_rows"] != 4533 or quality["cold_test_rows"] != 147:
        raise ValueError("Validation must cover all original requests and all five cutoffs")
    final = {(x["k"], x["stratum"]): x for x in quality["primary"] if x["method"] == "unisrec_2k"}
    ci = {(x["k"], x["stratum"]): x for x in paired["comparisons"]
          if x["method"] == "unisrec_2k" and x["comparison"] == "rqvae_minus_rqkmeans"}
    names = ("quality_comparison.json", "quality_comparison.csv", "paired_comparison.json",
             "performance_comparison.json", "performance_comparison.csv", "benchmark_protocol.json",
             "benchmarks.json", "selection.json", "batch_audit.json", "test.csv", "validation.csv",
             "comparison.html", "figures/rqvae-quality.png", "figures/rqvae-timing.png")
    detail_root = output / "inference"
    (detail_root / "figures").mkdir(parents=True)
    for name in names:
        shutil.copyfile(report / name, detail_root / name)
    validate_documents(detail_root, {"comparison.html": [
        ("rqvae-quality", "商品级质量"), ("rqvae-threshold", "K=1 阈值敏感性"), ("rqvae-performance", "时延与吞吐")
    ]})
    rows = []
    for k in KS:
        sample, cold, delta = final[k, "overall"], final[k, "cold"], ci[k, "overall"]
        config = selection["winners"][str(k)]
        p, q = old_bench[config], bench[config]
        before, after = sample["old"], sample["new"]
        low, high = delta["ndcg_ci95_pp"]
        rows.append([k, pct(before["item_hr"]), pct(after["item_hr"]), pct(before["item_ndcg"]),
                     pct(after["item_ndcg"]), f'{delta["ndcg_delta_pp"]:+.3f} [{low:+.3f}, {high:+.3f}]',
                     f'{cold["old"]["item_hits"]} → {cold["new"]["item_hits"]} / 147',
                     f'{p["latency_p50_ms"]:.2f} → {q["latency_p50_ms"]:.2f}',
                     f'{p["requests_s_4gpu"]:.2f} → {q["requests_s_4gpu"]:.2f}'])
    curves = []
    for side, label, color in (("old", "RQ-KMeans SID", OLD), ("new", "RQ-VAE SID", NEW)):
        curves.append((label, color, {
            "hr": [(k, 100 * final[k, "overall"][side]["item_hr"]) for k in KS],
            "ndcg": [(k, 100 * final[k, "overall"][side]["item_ndcg"]) for k in KS],
            "cold": [(k, final[k, "cold"][side]["item_hits"]) for k in KS],
        }))
    comparison_chart(output, "item-2k", (("hr", "Item HR (%)", "K"), ("ndcg", "Item NDCG (%)", "K"),
                                        ("cold", "Cold target hits / 147", "K")), curves)
    detail = []
    for x in quality["primary"]:
        if x["stratum"] != "overall":
            continue
        method = {"beam": "纯 beam50", "unisrec_original_rank": "UniSRec 原排",
                  "unisrec_2k": "UniSRec-2K + GRPO"}[x["method"]]
        detail.append([x["k"], method, pct(x["old"]["item_ndcg"]), pct(x["new"]["item_ndcg"]),
                       x["old"]["item_hits"], x["new"]["item_hits"]])
    changed = ci[50, "overall"]
    low, high = changed["ndcg_ci95_pp"]
    conclusion = ("区间跨过零，K=50 的测试结果未给出方向明确的质量变化。"
                  if low <= 0 <= high else f"K=50 的配对区间整体{'高于' if low > 0 else '低于'}零。")
    raw = [r for r in quality["primary"] if r["method"] == "unisrec_original_rank" and r["stratum"] == "overall"]
    raw_count = sum(a["old"]["item_hits"] == a["new"]["item_hits"] for a in raw)
    rows_overall = [final[k, "overall"] for k in KS]
    improved_ndcg = sum(x["new"]["item_ndcg"] > x["old"]["item_ndcg"] for x in rows_overall)
    improved_hr = sum(x["new"]["item_hr"] > x["old"]["item_hr"] for x in rows_overall)
    one = final[1, "overall"]
    one_beam = next(row for row in quality["primary"] if
                    (row["k"], row["method"], row["stratum"]) == (1, "beam", "overall"))
    one_low, one_high = ci[1, "overall"]["ndcg_ci95_pp"]
    return (
        '<p>主对照固定每轮 2K、α=1、前缀宽 1、最多三轮和完整 beam50。'
        'K=1 使用旧阈值，其余取全接受。冷目标按训练历史或目标是否出现划分。'
        '商品级 HR 和 NDCG 使用完全相同的 4,533 条历史、目标商品 ID 和冷暖标签。</p>'
        + training_figure("item-2k", "最终 2K 方案在相同测试行上的质量与冷目标变化。")
        + table(("K", "原 HR", "新 HR", "原 NDCG", "新 NDCG", "NDCG 差与 95% 区间（pp）",
                 "冷命中", "P50 ms", "四卡请求/s"), rows)
        + f'<p>新方案的 NDCG 在 {improved_ndcg}/5 档提高，HR 在 {improved_hr}/5 档提高。K=1 的 2K 商品 HR '
          f'由 {pct(one["old"]["item_hr"])} 升至 {pct(one["new"]["item_hr"])}，配对区间 '
          f'[{one_low:+.3f}, {one_high:+.3f}] 个百分点。'
          f'用新权重直接做 beam50 并截取一件时，HR 是 {pct(one_beam["new"]["item_hr"])}。'
          '这个对照说明新 SID 对生成路径和固定 2K 重排的影响不同。</p>'
        + f'<p>K=50 新旧结果逐请求配对，新增命中 {changed["gained_hits"]} 条、丢失 {changed["lost_hits"]} 条。'
        + conclusion
        + '区间用配对 bootstrap，保留相同请求的相关性。单请求 P50 用 batch=1，四卡吞吐按每卡 batch=4 测量。</p>'
        + k50_prefix_diagnostics(workspace, inference, output)
        + '<h3>纯生成与 UniSRec 原排对照</h3>'
        + table(("K", "方法", "原 NDCG", "新 NDCG", "原商品命中", "新商品命中"), detail)
        + token_latency(workspace, inference, output, selection, old_bench, bench)
        + f'<p>纯 beam 比较完整 SID 生成和映射回商品的效果。UniSRec 原排在 {raw_count}/5 档的商品总命中相同，'
          '可用来判断联合结果变化是否来自检索侧。'
          'K=1 的阈值重校准是独立的验证集敏感性分析，主表使用固定旧阈值。'
          '<a href="inference/comparison.html">查看质量、性能与批量差异详情</a> · '
          '<a href="inference/quality_comparison.csv">下载全部冷暖分组</a> · '
          '<a href="inference/paired_comparison.json">下载配对区间</a> · '
          '<a href="inference/performance_comparison.json">下载性能对照</a> · '
          '<a href="inference/selection.json">下载冻结参数</a></p>'
    )


def build(workspace, run, inference, destination):
    prep = load(run / "data/preparation_manifest.json")
    prepared_rows = prep["rows"]
    if prep["original_sid"] or tuple(prepared_rows[s] for s in ("train", "valid", "test")) != (36259, 4532, 4533):
        raise ValueError("Unexpected data preparation")
    with tempfile.TemporaryDirectory(prefix="industrial-rqvae-report-") as directory:
        output = Path(directory) / NAME
        (output / "figures").mkdir(parents=True)
        (output / "data").mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1] / "project_report/report.css", output / "report.css")
        sections = [
            ("data", "数据与实验口径",
             "<p>工业品目录含 3,686 件商品。训练、验证、测试仍是 36,259、4,532、4,533 条。"
             "数据重写仅替换历史和目标 SID。用户、商品、交互顺序和切分按原始记录保留。"
             "比较范围覆盖 RQ-VAE、SFT、GRPO、UniSRec 与联合推理。</p>"),
            ("quantizer", "RQ-VAE 训练与 SID 导出", quantization(run, output)),
            ("training", "SFT、GRPO 与 UniSRec 重训", training(workspace, run, output)),
            ("validation", "2K 验证、性能与结果差异", validation(workspace, inference, output)),
        ]
        links = "".join(f'<li><a href="#{key}">{escape(title)}</a></li>' for key, title, _ in sections)
        content = "".join(f'<section><h2 id="{key}">{escape(title)}</h2>{body}</section>'
                          for key, title, body in sections)
        (output / "index.html").write_text(
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>工业品 RQ-VAE 重训验证</title><link rel="stylesheet" href="report.css"></head>'
            '<body><header class="hero"><div class="frame"><p class="dataset">Amazon Reviews 2018 · Industrial and Scientific</p>'
            '<h1>RQ-VAE 重训与 RQ-KMeans 对照</h1><p class="summary">统一 UniSRec 2K 联合推理</p></div></header>'
            f'<main class="frame"><nav class="toc"><p class="toc-title">目录</p><ol>{links}</ol></nav>{content}</main>'
            '</body></html>', encoding="utf-8")
        validate_documents(output, {"index.html": [(key, title) for key, title, _ in sections]})
        archive = output.parent / (NAME + ".zip")
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
            for filename in sorted(output.rglob("*")):
                if filename.is_file():
                    bundle.write(filename, filename.relative_to(output.parent))
        with ZipFile(archive) as bundle:
            if bundle.testzip():
                raise ValueError("Report archive failed CRC checks")
        publish(archive, destination)
        if digest(archive) != digest(destination):
            raise OSError("Published report differs from the staging archive")
        print(json.dumps({"report": str(destination), "sha256": digest(destination)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/industrial-rqvae-report-mpl")
    build(args.workspace, args.run_dir, args.inference_dir, args.destination)


if __name__ == "__main__":
    main()
