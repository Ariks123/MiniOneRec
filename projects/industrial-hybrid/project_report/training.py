"""Render the historical Industrial and Scientific training dossier."""

from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import re
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MINI = Path("weights/checkpoints/MiniOneRec")
UNI = Path("weights/checkpoints/UniSRec")
PREP = Path("MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey")
SID = Path("MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey/Industrial_and_Scientific.index.json")
SFT_ROOT = MINI / "Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1"
UNI_PREFIX = "Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3"
UNI_RUNS = (
    (15, UNI / f"{UNI_PREFIX}_ep15_20261117_182158"),
    (20, UNI / f"{UNI_PREFIX}_ep20_20261117_181242"),
    (25, UNI / f"{UNI_PREFIX}_ep25_20261117_182943"),
    (50, UNI / f"{UNI_PREFIX}_ep50_retry1_20261117_180455"),
    (100, UNI / f"{UNI_PREFIX}_20261117_173712"),
)
SFT_RUNS = (
    ("原配置 · Linear 3", MINI / "Industrial_and_Scientific_enddonkey_05b_linear/sft", "evaluation"),
    ("Cosine 10 · 原任务", MINI / "Industrial_and_Scientific_05b_cosine_ep10_20261117/baseline/sft", "evaluation"),
    ("Cosine 10 · 加标题历史", MINI / "Industrial_and_Scientific_05b_cosine_ep10_20261117/title_history/sft", "evaluation"),
    ("Cosine 5 · 按 NDCG@10 选点", MINI / "Industrial_and_Scientific_05b_cosine_ep5_title_history_rank50_20261117/sft", "evaluation"),
    ("Linear 5 · 第 3 轮快照", SFT_ROOT / "epochs5/sft", "evaluation_epoch3"),
    ("Linear 5 · 选定权重", SFT_ROOT / "epochs5/sft", "evaluation"),
    ("Linear 3 · 主线", SFT_ROOT / "epochs3/sft", "evaluation"),
)
RL_RUNS = (
    ("0.001", MINI / "Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl"),
    ("0.01", MINI / "Industrial_and_Scientific_05b_rl_beta0p01_20261118/rl"),
)
SECTIONS = [
    ("training-data", "数据准备与训练样本"),
    ("training-sid", "语义 ID 与 RQ-VAE 归档"),
    ("training-sft", "Qwen SFT 训练"),
    ("training-rl", "GRPO 强化学习"),
    ("training-unisrec", "UniSRec 训练与向量导出"),
    ("training-summary", "训练结果解释与检查点谱系"),
]
BLUE = "#0d5774"
TEAL = "#087d6d"
ORANGE = "#ae581a"
COLORS = (BLUE, ORANGE)
RL_FIELDS = ("loss", "learning_rate", "reward", "rewards/rule_reward", "rewards/ndcg_rule_reward",
             "kl", "grad_norm", "zero_reward_group_rate", "eval_ndcg10", "eval_recall10")
RL_PANELS = (
    ("rewards/rule_reward", "Exact-SID reward"),
    ("rewards/ndcg_rule_reward", "Rank penalty"),
    ("kl", "KL estimate"),
    ("grad_norm", "Gradient L2 (pre-clip)"),
    ("zero_reward_group_rate", "Zero-reward group ratio"),
    ("eval_ndcg10", "Validation SID NDCG@10"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_file(output, name, fields, rows):
    with (output / "data" / f"training-{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100 * value:.3f}%"


def table(headers, rows):
    head = "".join(f'<th scope="col">{escape(str(cell))}</th>' for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(name, caption):
    fig = f"figures/training-{name}.png"
    return f'<figure><img src="{fig}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def evidence(*paths):
    return '<p class="note">工作区证据 ' + "、".join(f"<code>{escape(str(path))}</code>" for path in paths) + "</p>"


def download(name, label):
    return f'<p><a href="data/training-{name}.csv">{escape(label)}</a></p>'


def save(fig, output, name):
    for panel in fig.axes:
        panel.grid(axis="y", alpha=0.2)
        for spine in ("top", "right"):
            panel.spines[spine].set_visible(False)
    fig.savefig(output / "figures" / f"training-{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trend(panel, values, color, label, size=100):
    """Include step-level values under nonoverlapping block means."""
    x, y = zip(*values)
    panel.plot(x, y, color=color, alpha=.18, linewidth=.55)
    blocks = [values[start:start + size] for start in range(0, len(values), size)]
    panel.plot([mean(point[0] for point in group) for group in blocks],
               [mean(point[1] for point in group) for group in blocks], label=label, color=color, linewidth=1.5)


def data_section(prepared, sft_conf, rl_conf, output):
    rows = prepared["rows"]
    audit = prepared["source_audit"]
    lengths = list(range(1, 11))
    split_records = []
    fig, panel = plt.subplots(figsize=(8, 3.05), layout="constrained")
    for split, color in (("train", BLUE), ("valid", TEAL), ("test", ORANGE)):
        info = audit["splits"][split]
        counts = [info["history_length_counts"].get(str(size), 0) for size in lengths]
        if sum(counts) != rows[split]:
            raise ValueError(f"Incomplete history length audit for {split}")
        shares = [count / rows[split] for count in counts]
        panel.plot(lengths, [share * 100 for share in shares], marker="o", markersize=4,
                   linewidth=1.65, color=color, label=split.capitalize())
        split_records.extend(dict(split=split, history_length=size, rows=count, fraction=share)
                             for size, count, share in zip(lengths, counts, shares))
    panel.set(xlabel="History length (items)", ylabel="Share of split (%)", xticks=lengths)
    panel.legend(frameon=False, ncols=3)
    save(fig, output, "history-length")
    csv_file(output, "history-length", tuple(split_records[0]), split_records)

    dataset = table(("用途", "行数", "输入与标签"), (
        ("训练交互", f"{rows['train']:,}", "用户顺序历史与真实下一件商品"),
        ("验证交互", f"{rows['valid']:,}", "选择训练检查点、校准联合推理参数"),
        ("测试交互", f"{rows['test']:,}", "使用锁定模型和配置评估"),
        ("SFT 训练任务", f"{sft_conf['data']['train_samples']:,}", "从交互和商品元数据扩展为五类任务"),
        ("RL 训练任务", f"{rl_conf['data']['train_samples']:,}", "从交互及商品元数据构造四类奖励任务"),
    ))
    task_rows = []
    tasks = (
        ("sid_sequence_recommendation", "历史 SID → 下一商品 SID"),
        ("history_to_title", "历史商品 → 下一商品标题"),
        ("sid2title", "商品 SID → 商品标题"),
        ("title2sid", "商品标题 → 商品 SID"),
        ("title_history_to_sid", "历史商品标题 → 下一商品 SID"),
        ("description2sid", "商品描述 → 商品 SID"),
    )
    for key, label in tasks:
        counts = [f"{cfg['task_counts'][key]:,}" if key in cfg["task_counts"] else "—" for cfg in (sft_conf, rl_conf)]
        task_rows.append((label, *counts))
    return (f"<p>归档样本覆盖 {audit['users_across_all_splits']:,} 名用户、{prepared['items']:,} 件商品，"
            f"合计 {sum(rows.values()):,} 条交互请求。每条请求以 1–10 件历史商品预测下一件商品。</p>"
            + dataset
            + figure("history-length", "原切分中的历史长度分布。纵轴为每个切分内部的比例。")
            + "<p>训练集中 1–4 件商品的历史最多，验证和测试中峰值均位于 4 件。"
            "保留了原训练切分中 3 条完整重复行，验证和测试均无完整重复行。"
            "原始 CSV 无时间戳，切分沿用上游预处理结果。</p>"
            + download("history-length", "下载各切分的历史长度分布")
            + table(("训练任务", "SFT 条数", "RL 条数"), task_rows)
            + "<p>目录商品、按时间切分的交互、由交互或商品元数据扩展出的训练任务分别计数。"
            "同一段交互可以为不同训练目标产生多条任务。数据准备按原切分保留全部交互行，标准化商品与 SID 对应关系，"
            "历史保留末尾最多 10 件商品。训练、验证和测试 CSV 在主线运行中通过文件 SHA-256 固定。</p>"
            "<p>RL 中的历史标题任务固定抽取 10,000 条，历史 SID 任务仍使用全部 36,259 条训练交互。"
            "验证历史的目标来自原验证切分，SFT 用答案 token 的损失选点，RL 在相同 4,532 条历史上评估生成排名。</p>"
            + evidence(PREP / "preparation_manifest.json", SFT_ROOT / "epochs3/sft/run_config.json", RL_RUNS[0][1] / "rl_task_manifest.jsonl"))


def sid_section(workspace, output, prepared):
    catalog = read(workspace / SID)
    if not prepared["original_sid"] or len(catalog) != prepared["items"]:
        raise ValueError("Industrial training must use the archived original SID catalog")
    counts = [Counter(int(code[3:-1]) for code in codes) for codes in zip(*catalog.values())]
    usage = [dict(position=level, code=code, item_count=c[code])
             for level, c in zip("abc", counts) for code in range(256)]
    csv_file(output, "sid-usage", ("position", "code", "item_count"), usage)
    fig, panels = plt.subplots(1, 3, figsize=(10, 2.5), layout="constrained")
    for level, c, ax in zip("abc", counts, panels):
        ax.bar(sorted(c), [c[v] for v in sorted(c)], color=TEAL, width=1)
        ax.set(xlabel=f"{level} code ({len(c)} active)", xlim=(-4, 260), xticks=(0, 64, 128, 192, 255))
    panels[0].set_ylabel("Catalog items")
    save(fig, output, "sid-usage")
    return (
        "<p>工业品主线加载复现仓库已经提供的<strong>残差 K-Means（RQ-KMeans）SID</strong>。训练用的词表、目录约束和后续推理映射"
        "均沿用这套三位编码。预处理清单标记原始 SID，记录了目录 SHA-256，未选择任何 RQ-VAE epoch。</p>"
        "<p>本地 RQ-VAE 训练记录来自 Video Games。工业品目录的可核验材料是上游发布的映射及其入库审计。"
        "工业品的 RQ-VAE 训练轮数、学习率、重建误差和量化损失曲线均缺少归档。本节以工业品 SID 产物的实测质量记录这一阶段。</p>"
        + table(("目录商品", "唯一完整 SID", "碰撞组", "涉及商品", "使用的 a/b/c token"), [
            (prepared["items"], prepared["unique_sids"], prepared["sid_collisions"]["groups"],
             prepared["sid_collisions"]["items"], " / ".join(str(len(c)) for c in counts)),
        ])
        + figure("sid-usage", "按目录统计三位 SID 的 code 使用频次。第一位使用 48 个不同 code，第二、三位各使用 256 个。")
        + "<p>第一位编码决定后续生成的搜索分支。目录有 15 组完整 SID 对应多件商品，命中这些 SID 后，最终商品 ID 仍由映射规则决定。"
        "后续验证报告列出 SID 命中和精确商品命中。</p>"
        + download("sid-usage", "下载三层 code 使用频次")
        + evidence(SID, Path("code/MiniOneRec-Reproduction/README.md"), PREP / "preparation_manifest.json")
    )


def sft_section(workspace, output):
    comparisons = []
    selected = {}
    for name, path, scope in SFT_RUNS:
        root = workspace / path
        summary = read(root / "training_summary.json")
        metrics = read(root / scope / "metrics.json")["metrics"]
        snapshot = scope == "evaluation_epoch3"
        entry = dict(run=name, epoch=3.0 if snapshot else summary["epoch"],
                     actual_steps=5442 if snapshot else summary["global_step"],
                     checkpoint="epoch_3_checkpoint" if snapshot else Path(summary["best_checkpoint"]).name,
                     stop_reason="epoch_snapshot" if snapshot else summary["stop_reason"],
                     run_minutes=None if snapshot else summary["train_metrics"]["train_runtime"] / 60,
                     hr10=metrics["HR@10"], ndcg10=metrics["NDCG@10"], hr50=metrics["HR@50"], ndcg50=metrics["NDCG@50"],
                     evidence=str(path / scope / "metrics.json"))
        comparisons.append(entry)
        if scope == "evaluation" and path.parent.name in ("epochs3", "epochs5"):
            h = read(root / "trainer_state.json")["log_history"]
            selected[path.parent.name] = (entry, h)

    csv_file(output, "sft-comparison", tuple(comparisons[0]), comparisons)
    fig, panels = plt.subplots(1, 3, figsize=(11, 3), layout="constrained")
    points = []
    for key, color in zip(("epochs3", "epochs5"), COLORS):
        entry, history = selected[key]
        label = "Linear 3" if key == "epochs3" else "Linear 5"
        for row in history:
            if "step" in row:
                points.append({key: row.get(key) for key in ("step", "epoch", "loss", "eval_loss", "learning_rate")}
                              | {"run": label})
        trend(panels[0], [(row["step"], row["loss"]) for row in history if "loss" in row], color, label)
        vals = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
        panels[1].plot(*zip(*vals), color=color, marker=".", markersize=3, label=label)
        mark = int(entry["checkpoint"].split("-")[-1])
        y = next(y for step, y in vals if step == mark)
        panels[1].scatter([mark], [y], color=color, edgecolors="#1a3137", s=58, zorder=5)
        lr = [(row["step"], row["learning_rate"]) for row in history if "learning_rate" in row]
        panels[2].plot(*zip(*lr), color=color, label=label)
    for p, y in zip(panels, ("Answer-token train CE", "Answer-token validation CE", "Learning rate")):
        p.set(xlabel="Optimizer step", ylabel=y)
    panels[0].legend(frameon=False)
    save(fig, output, "sft")
    csv_file(output, "sft-curves", ("run", "step", "epoch", "loss", "eval_loss", "learning_rate"), points)

    return (
        "<p>主线从 Qwen2.5-0.5B 预训练权重开始，全量微调 24 层自回归 Transformer。"
        "模型隐藏维度 896，前馈层维度 4,864，注意力有 14 个查询头与 2 个 KV 头。"
        "新增 560 个原子 SID token，词表由 151,665 扩充到 152,225。训练只在非掩码答案 token 上计算交叉熵。</p>"
        "<p>线性训练采用峰值学习率 3×10<sup>−4</sup>、20 步 warm-up、4 卡，每卡 micro-batch 8、累计 2 步，"
        "全局 batch 64，最大长度 512。采用 BF16、ZeRO-2 与梯度裁剪 1.0。验证损失每约 5% 计划步数记录一次。"
        "patience 为 3，两个线性组均达到各自的 epoch 上限。</p>"
        + figure("sft", "线性 3 轮与线性 5 轮的训练损失、验证损失和学习率。细线是逐步原值，粗线为每 100 步分块均值，圆点标明最终选点。")
        + "<p>训练前，新增 SID token 后的底座答案 token 验证交叉熵为 14.4963。图中从训练开始后的实际优化记录绘制。"
        "三轮组在第 5,187 步达到最低验证损失 1.5607，最终导出恢复该权重。五轮组即使训练损失更低，"
        "最低验证损失仍为 1.5791，测试 NDCG@10 降至 6.073%。五轮组的第 3 轮快照由更长的线性学习率计划得到，"
        "公开的训练损失与测试排名支持按实际轨迹看待这两个第 3 轮模型。</p>"
        + table(("训练配置", "停止轮数或快照轮数", "优化步", "评估权重", "训练分钟", "测试 SID HR@10", "测试 SID NDCG@10"), [
            (r["run"], f"{r['epoch']:g}", f"{r['actual_steps']:,}", r["checkpoint"],
             f"{r['run_minutes']:.2f}" if r["run_minutes"] is not None else "—", percent(r["hr10"]), percent(r["ndcg10"]))
            for r in comparisons
        ])
        + "<p>相同 10 轮 cosine 上限下，加标题历史组相对原任务组将 SID NDCG@10 从 2.793% 提高到 5.250%，"
        "同时增加每轮任务和步数。线性三轮加标题历史组达到 8.300%。这些对照同时受任务量、训练轨迹和选点方法影响，"
        "适合作为本项目的配置比较。</p>"
        + download("sft-curves", "下载 SFT 逐步曲线数据")
        + download("sft-comparison", "下载 SFT 配置与测试指标")
        + evidence(SFT_ROOT / "epochs3/sft", SFT_ROOT / "epochs5/sft")
    )


def rl_section(workspace, output):
    fig, panels = plt.subplots(2, 3, figsize=(11, 5.4), layout="constrained")
    comps, points = [], []
    for ((beta, path), color) in zip(RL_RUNS, COLORS):
        root = workspace / path
        s = read(root / "training_summary.json")
        m = read(root / "evaluation/metrics.json")["metrics"]
        base = read(root / "baseline_validation.json")["metrics"]
        history = read(root / "trainer_state.json")["log_history"]
        rows = [row for row in history if "kl" in row and "step" in row]
        best = Path(s["best_checkpoint"]).name
        baseline = dict(step=0, epoch=0.0, eval_ndcg10=base["ndcg10"], eval_recall10=base["recall10"])
        verifications = {row["step"]: row for row in history if "eval_ndcg10" in row}
        check_step = int(best.split("-")[-1])
        if verifications[check_step]["eval_ndcg10"] != s["best_metric"]:
            raise ValueError(f"GRPO best validation metric differs at step {check_step}")
        for ax, (field, label) in zip(panels.flat, RL_PANELS):
            if field == "eval_ndcg10":
                val = [baseline] + [verifications[step] for step in sorted(verifications)]
                ax.plot([x["step"] for x in val], [x[field] * 100 for x in val], marker=".", color=color, label=f"beta={beta}")
                ax.scatter([check_step], [s["best_metric"] * 100], color=color, edgecolors="#1a3137", s=54, zorder=5)
                ax.set_ylabel("Validation SID NDCG@10 (%)")
            else:
                trend(ax, [(row["step"], row[field]) for row in rows], color, f"beta={beta}")
                ax.set_ylabel(label)
            ax.set_xlabel("Optimizer step")
        for row in [baseline] + [row for row in history if "step" in row]:
            points.append({key: row.get(key) for key in ("step", "epoch", *RL_FIELDS)} | {"beta": beta})
        for window, run in (("last100", rows[-100:]), ("common_steps", [r for r in rows if r["step"] <= 1449])):
            comps.append(dict(beta=beta, window=window, count=len(run), stop_step=s["global_step"],
                              stop_epoch=s["epoch"], stop_reason=s["stop_reason"], checkpoint=best,
                              runtime_minutes=s["train_metrics"]["train_runtime"] / 60,
                              validation_ndcg10=s["best_metric"], test_hr10=m["HR@10"], test_ndcg10=m["NDCG@10"],
                              test_hr50=m["HR@50"], test_ndcg50=m["NDCG@50"],
                              kl=mean(r["kl"] for r in run), grad_norm=mean(r["grad_norm"] for r in run),
                              rule=mean(r["rewards/rule_reward"] for r in run),
                              ndcg_penalty=mean(r["rewards/ndcg_rule_reward"] for r in run),
                              zero_group=mean(r["zero_reward_group_rate"] for r in run), evidence=str(path)))
    panels[0, 0].legend(frameon=False)
    save(fig, output, "grpo")
    csv_file(output, "grpo-curves", ("beta", "step", "epoch", *RL_FIELDS), points)
    csv_file(output, "grpo-comparison", tuple(comps[0]), comps)
    return (
        "<p>两个 β 配置都从线性三轮组的同一 SFT 权重继续训练。最大 2 轮，4 卡，"
        "每优化步 64 个完整提示组、合计 1,024 个完成候选。每个提示由受约束 beam 产生 16 条候选。"
        "学习率为 10<sup>−5</sup>，cosine 调度，warm-up 占 3%。每 512 步按 0.6 的混合系数同步参考策略。"
        "训练按验证 SID NDCG@10 选点，patience 为 3。</p>"
        "<p>Rule Reward 是候选生成的 SID 完整命中时给 1，否则给 0。组里出现命中时，错误候选按位次得到归一化的负折扣。"
        "组内标准化奖励得到相对优势。KL 是完成 token 上的策略漂移估计，β 控制其权重。Grad Norm 是裁剪前的全局 L2 范数，"
        "实际裁剪上限 0.3。全零奖励组仍可产生 KL 项，候选匹配比例与最终测试 HR 使用不同的分母。</p>"
        + figure("grpo", "两种 β 的规则奖励、排名惩罚、KL、梯度范数、零奖励组比例与验证 NDCG@10。粗线为每 100 步分块均值，验证圆点突出选定的检查点。")
        + table(("KL β", "训练结束", "导出检查点", "训练分钟", "验证 SID NDCG@10", "测试 SID HR@10", "测试 SID NDCG@10", "测试 SID HR@50", "测试 SID NDCG@50"), [
            (r["beta"], f"{r['stop_step']:,} 步 · {'早停' if r['stop_reason'] == 'early_stopping' else '达到上限'}",
             r["checkpoint"], f"{r['runtime_minutes']:.2f}", percent(r["validation_ndcg10"]),
             percent(r["test_hr10"]), percent(r["test_ndcg10"]), percent(r["test_hr50"]), percent(r["test_ndcg50"]))
            for r in comps if r["window"] == "last100"
        ])
        + table(("KL β", "统计窗口", "记录条数", "平均 Rule Reward", "平均 KL", "平均 Grad Norm"), [
            (r["beta"], "各组末 100 条" if r["window"] == "last100" else "相同步数 1–1,449",
             f"{r['count']:,}", percent(r["rule"]), f"{r['kl']:.4f}", f"{r['grad_norm']:.4f}") for r in comps
        ])
        + "<p>提高 β 后，KL 在相同步数内明显下降，Grad Norm 也降低。β=0.01 的最佳验证 NDCG@10 略高，"
        "同一个测试集上的 NDCG@10 与 NDCG@50 均略低于 β=0.001。最终联合推理沿用归档的 β=0.001 权重。"
        "因此 β 的实验结果支持较强约束带来更小策略漂移，而两配置测试排名的差距较小。</p>"
        + download("grpo-curves", "下载 GRPO 逐步奖励、梯度和验证数据")
        + download("grpo-comparison", "下载 β 对照及统计窗口")
        + evidence(*(path for _, path in RL_RUNS))
    )


def unisrec_section(workspace, output):
    main = next(path for cap, path in UNI_RUNS if cap == 20)
    log = (workspace / main / "train.log").read_text(encoding="utf-8")
    pattern = r"epoch (\d+) (?:training|evaluating) \[time: ([\d.]+)s, (?:train loss|valid_score): ([\d.]+)"
    observations = [(int(epoch) + 1, float(sec), float(loss)) for epoch, sec, loss in re.findall(pattern, log)]
    train = observations[::2]
    valid = observations[1::2]
    if [t[0] for t in train] != list(range(1, 21)) or [v[0] for v in valid] != list(range(1, 21)):
        raise ValueError("Expected training and validation logs for all 20 UniSRec epochs")
    curve = [dict(epoch=t[0], summed_batch_ce=t[2], validation_vector_nll=v[2],
                  training_seconds=t[1], validation_seconds=v[1]) for t, v in zip(train, valid)]
    csv_file(output, "unisrec-curves", tuple(curve[0]), curve)
    fig, panels = plt.subplots(1, 2, figsize=(8, 2.7), layout="constrained")
    for ax, series, color, name in zip(panels, (train, valid), (BLUE, TEAL),
                                      ("Training batch-CE sum", "Validation vector NLL")):
        ax.plot([x[0] for x in series], [x[2] for x in series], color=color, marker=".", linewidth=1.6)
        ax.set(xlabel="Epoch", ylabel=name, xticks=(1, 5, 10, 15, 20))
    panels[1].scatter([20], [valid[-1][2]], color=TEAL, edgecolors="#1a3137", s=62, zorder=5)
    save(fig, output, "unisrec")
    comparisons = []
    for cap, path in UNI_RUNS:
        summary = read(workspace / path / "training_summary.json")
        values = summary["test_vector_quality"]
        cold = values["groups"]["train_unseen"]
        row = dict(epoch_cap=cap, completed_epochs=summary["completed_epochs"], best_epoch=summary["best_epoch"],
                   validation_nll=summary.get("best_validation_loss", summary.get("best_valid_vector_nll")),
                   test_nll=values["vector_nll"], evidence=str(path / "training_summary.json"))
        row.update({f"recall{k}": values[f"tie_expected_recall@{k}_mean"] for k in (10, 20, 50)})
        row.update({f"cold_recall{k}": cold[f"tie_expected_recall@{k}_mean"] for k in (10, 20, 50)})
        comparisons.append(row)
    csv_file(output, "unisrec-comparison", tuple(comparisons[0]), comparisons)
    return (
        "<p>UniSRec 读取归档的 2,560 维商品文本向量，用 8 个带软门控的专家映射到 300 维。"
        "历史加入位置嵌入后通过 2 层 2 头的 SASRec Transformer，前馈隐藏宽 256，取最后一个有效位置作为用户表示。"
        "归一化用户和商品向量后作内积。训练在训练目录上使用温度 0.07 的商品交叉熵，验证和推理扩展到全目录。</p>"
        "<p>主线使用 4 卡、全局 batch 512 和 Adam。前 5 轮 warm-up，学习率峰值 0.002，cosine 最低 0.0002。"
        "训练日志以 0 起始，这里转换成人类可读的 1–20 轮。每轮记录训练批损失累计值和验证目录 NLL，"
        "两张图采用各自的纵轴。轮内训练合计 81.89 秒，轮内验证合计 14.48 秒，容器运行约 120.3 秒。</p>"
        + figure("unisrec", "UniSRec 20 轮组的训练与验证轨迹。验证目录 NLL 从 7.523539 降到 7.236538，出口权重为第 20 轮。")
        + table(("计划轮数", "实际轮数", "组内最佳轮", "验证 NLL", "测试 Recall@10", "测试 Recall@20", "测试 Recall@50", "冷 Recall@50"), [
            (r["epoch_cap"], r["completed_epochs"], r["best_epoch"], f"{r['validation_nll']:.6f}",
             percent(r["recall10"]), percent(r["recall20"]), percent(r["recall50"]), percent(r["cold_recall50"]))
            for r in comparisons
        ])
        + "<p>上表沿用独立 UniSRec 报告的同分期望 Recall。20 轮组是用户选定的配置，组内按最低验证目录 NLL 保存第 20 轮。"
        "25 轮组的最佳验证 NLL 更低，测试 Recall@10 与 Recall@50 则由选定的 20 轮组取得最高值。"
        "这说明全目录概率拟合与前 K 排名可以有不同的排序。最终导出复用该权重的历史编码器与 3,686×300 商品向量。</p>"
        + download("unisrec-curves", "下载 UniSRec 20 轮训练和验证曲线")
        + download("unisrec-comparison", "下载所有已完成 UniSRec 轮数对照")
        + evidence(main / "train.log", main / "training_summary.json", main / "model_selection.json")
    )


def render(workspace: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return training report sections and their table-of-contents entries."""
    prepared = read(workspace / PREP / "preparation_manifest.json")
    sft_conf = read(workspace / SFT_ROOT / "epochs3/sft/run_config.json")
    rl_conf = read(workspace / RL_RUNS[0][1] / "run_config.json")
    summary = (
        "<p>主线权重依次为<strong>原工业品三位 SID → Qwen2.5-0.5B 线性三轮 SFT"
        "（第 5,187 步）→ GRPO β=0.001（第 828 步）</strong>。"
        "UniSRec 第 20 轮权重在另一条训练路径导出，并在推理端提供商品候选。</p>"
        "<p>线性三轮 SFT 的 SID HR@10 为 12.949%，NDCG@10 为 8.300%。加入所选 RL 权重后为 13.788% 和 9.234%。"
        "在 K=50，SID HR 由 22.105% 降为 21.950%，而 NDCG 提高到 11.039%。"
        "较窄展示范围和按名次折扣的指标得到改善，较宽候选范围的总命中略有缩小。</p>"
        "<p>训练结论采用原训练生成验证的 SID 口径。最终联合方案由 UniSRec 每轮最多草拟 2K 件，"
        "用 RL 权重给完整 SID 打分，推荐评估按精确商品 ID 计算。两种口径在存在 SID 碰撞时会产生差异，"
        "后续验证报告列出对齐后的对照数据。</p>"
    )
    contents = (
        data_section(prepared, sft_conf, rl_conf, output), sid_section(workspace, output, prepared),
        sft_section(workspace, output), rl_section(workspace, output),
        unisrec_section(workspace, output), summary,
    )
    body = "".join(f'<section><h2 id="{key}">{title}</h2>{text}</section>'
                   for (key, title), text in zip(SECTIONS, contents))
    return body, SECTIONS
