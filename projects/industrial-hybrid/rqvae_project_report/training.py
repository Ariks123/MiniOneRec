"""Render the training curves and objectives for the RQ-VAE experiment."""

import csv
from html import escape
import json
from pathlib import Path
import re
from statistics import mean

import yaml

from project_report.training import csv_file, download, figure, percent, plt, save, table, trend


SECTIONS = (
    ("training-data", "数据准备与训练样本"),
    ("training-rqvae", "RQ-VAE 损失与语义 ID"),
    ("training-sft", "监督微调"),
    ("training-grpo", "GRPO 强化学习"),
    ("training-unisrec", "UniSRec 向量检索训练"),
    ("training-checkpoints", "检查点与训练结论"),
)
BLUE, TEAL, ORANGE = "#236782", "#06756b", "#a95123"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def formula(label, content):
    return (
        '<div class="formula"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" '
        f'aria-label="{escape(label)}"><mrow>{content}</mrow></math></div>'
    )


def selected_point(axis, step, value):
    axis.axvline(step, color=TEAL, linestyle=":", linewidth=1.1, zorder=1)
    axis.scatter([step], [value], color=TEAL, edgecolors="white", linewidths=0.8, s=48, zorder=5)


def section_data(preparation, sft, rl, output):
    sizes = preparation["rows"]
    audit = preparation["source_audit"]
    rows = []
    fig, axis = plt.subplots(figsize=(9, 3.0), layout="constrained")
    for split, label, color in (("train", "Train", BLUE), ("valid", "Validation", TEAL), ("test", "Test", ORANGE)):
        counts = audit["splits"][split]["history_length_counts"]
        bins = [(int(n), count) for n, count in sorted(counts.items(), key=lambda item: int(item[0]))]
        if sum(count for _, count in bins) != sizes[split]:
            raise ValueError(f"History counts are incomplete for {split}")
        axis.plot([n for n, _ in bins], [100 * count / sizes[split] for _, count in bins],
                  color=color, marker="o", markersize=3.5, label=label)
        rows.extend({"split": split, "history_items": n, "requests": count, "share": count / sizes[split]}
                    for n, count in bins)
    axis.set(xlabel="Items in history", ylabel="Request share (%)", xticks=range(1, 11))
    axis.legend(frameon=False, ncols=3)
    save(fig, output, "history")
    csv_file(output, "history", tuple(rows[0]), rows)
    labels = {
        "sid_sequence_recommendation": "历史 SID → 下一商品 SID",
        "history_to_title": "历史 SID → 下一商品标题",
        "sid2title": "商品 SID → 商品标题",
        "title2sid": "商品标题 → 商品 SID",
        "title_history_to_sid": "历史标题 → 下一商品 SID",
        "description2sid": "商品描述 → 商品 SID",
    }
    tasks = [
        (title, sft["task_counts"].get(task), rl["task_counts"].get(task))
        for task, title in labels.items()
    ]
    for run in (sft, rl):
        if sum(run["task_counts"].values()) != run["data"]["train_samples"]:
            raise ValueError("Post-training task counts do not match the archive")
    return (
        f'<p>本次重训覆盖 {audit["users_across_all_splits"]:,} 位用户和 {preparation["items"]:,} 件商品。'
        '每条请求由最近 1–10 件历史商品与一个下一商品目标组成。'
        '数据准备保留请求顺序和商品 ID，仅按新目录更新历史 SID 与目标 SID。</p>'
        + table(("用途", "条数"), [
            ("训练交互", f'{sizes["train"]:,}'), ("验证请求", f'{sizes["valid"]:,}'),
            ("测试请求", f'{sizes["test"]:,}'), ("测试冷目标", f'{audit["cold_test_rows"]:,}'),
        ])
        + figure("history", "各数据切分的历史长度分布。纵轴为各自切分中的请求占比。")
        + download("history", "下载历史长度分布")
        + table(("训练任务", "SFT 训练样本", "GRPO 训练样本"), [
            (label, f"{left:,}" if left is not None else "—", f"{right:,}" if right is not None else "—")
            for label, left, right in tasks
        ] + [("合计", f'{sft["data"]["train_samples"]:,}', f'{rl["data"]["train_samples"]:,}')])
        + '<p>SFT 用标题与 SID 的互译任务对齐文本和商品语义，在历史任务上学习下一商品预测。'
          'GRPO 用对应任务的真实 SID 作为奖励目标，并从标题历史任务采样 10,000 条。</p>'
    )


def section_rqvae(run, output):
    root = run / "rqvae"
    summary = read(root / "training_summary.json")
    history = read(root / "evaluation_history.json")
    exported = read(run / "sid_export_summary.json")
    repair = read(root / "collision_repair.json")
    if summary["model_sha256"] != exported["rqvae_model_sha256"] or summary["model_sha256"] != repair["model_sha256"]:
        raise ValueError("The coding diagnostics must refer to the selected RQ-VAE")
    config, counts = summary["config"], summary["split_counts"]
    chosen, end = summary["best_epoch"], summary["epochs_completed"]
    if chosen != exported["selected_epoch"] or history[-1]["epoch"] != end:
        raise ValueError("RQ-VAE training history and exported epoch differ")
    with (root / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != end:
        raise ValueError("RQ-VAE epoch loss data is incomplete")
    fields = ("epoch", "train_total", "train_recon", "train_quant", "lr")
    csv_file(output, "rqvae-epochs", fields, [{key: row[key] for key in fields} for row in raw])
    check_rows = [
        {"epoch": row["epoch"], "warm_holdout_mse": row["val"]["mse"], "catalog_mse": row["all"]["mse"],
         "raw_unique_sids": row["all"]["unique_sids"], "raw_collision_rate": row["all"]["collision_rate"]}
        for row in history
    ]
    csv_file(output, "rqvae-validation", tuple(check_rows[0]), check_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.35), layout="constrained")
    for field, color, label in (("train_total", BLUE, "Total"), ("train_recon", TEAL, "Reconstruction"),
                                ("train_quant", ORANGE, "Quantization")):
        trend(axes[0], [(int(row["epoch"]), float(row[field])) for row in raw], color, label, size=25)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Objective components", xlim=(0, end))
    axes[0].legend(frameon=False)
    axes[1].plot([r["epoch"] for r in check_rows], [r["warm_holdout_mse"] for r in check_rows],
                 color=BLUE, linewidth=1.5)
    selected_point(axes[1], chosen, summary["best_val_mse"])
    axes[1].set(xlabel="Epoch", ylabel="Warm holdout MSE", title="Checkpoint selection", xlim=(0, end))
    save(fig, output, "rqvae")

    raw_codes = repair["metrics"]["raw"]
    repaired = repair["metrics"]["prefix_matching"]["metrics"]
    if exported["unique_sids"] != repaired["all"]["unique_sids"]:
        raise ValueError("The final catalog does not match the recorded prefix repair")
    after = repaired["warm_holdout"]["mse"]
    diagnostics = [
        {"assignment": algorithm, "scope": scope, "items": row["items"], "unique_sids": row["unique_sids"],
         "excess_items": row["excess_items"], "mse": row["mse"]}
        for algorithm, data in (("nearest_codes", raw_codes), ("prefix_matching", repaired))
        for scope, row in data.items()
    ]
    csv_file(output, "rqvae-codes", tuple(diagnostics[0]), diagnostics)

    return (
        f'<p>用 {counts["train"]:,} 件训练可见商品拟合中心化均值与量化器，再用 {counts["val"]:,} 件暖商品的'
        f'重建误差选点。{counts["cold"]} 件冷商品参加冻结模型后的全库编码。'
        f'量化器使用 {len(config["num_emb_list"])} 级码本，每级 {config["num_emb_list"][0]} 个 {config["e_dim"]} 维向量。'
        f'训练使用 {config["optimizer"]}，预热 {config["warmup_epochs"]} 轮后学习率为 {config["lr"]:g}。</p>'
        '<h3>训练损失</h3>'
        + formula("RQ-VAE 的重建与量化损失", """
        <msub><mi>ℒ</mi><mtext>RQ</mtext></msub><mo>=</mo>
        <mi mathvariant="normal">MSE</mi><mo>(</mo><mi>x</mi><mo>,</mo><mover><mi>x</mi><mo>^</mo></mover><mo>)</mo>
        <mo>+</mo><mfrac><mi>λ</mi><mn>3</mn></mfrac>
        <munderover><mo>∑</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mn>3</mn></munderover>
        <msub><mi>ℓ</mi><mi>l</mi></msub>""")
        + formula("各级码本与承诺损失", """
        <msub><mi>ℓ</mi><mi>l</mi></msub><mo>=</mo>
        <mi mathvariant="normal">MSE</mi><mo>(</mo><mi mathvariant="normal">sg</mi><mo>(</mo>
        <msub><mi>r</mi><mrow><mi>l</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>)</mo><mo>,</mo>
        <msub><mi>c</mi><mi>l</mi></msub><mo>)</mo><mo>+</mo><mi>β</mi>
        <mi mathvariant="normal">MSE</mi><mo>(</mo><msub><mi>r</mi><mrow><mi>l</mi><mo>−</mo><mn>1</mn></mrow></msub>
        <mo>,</mo><mi mathvariant="normal">sg</mi><mo>(</mo><msub><mi>c</mi><mi>l</mi></msub><mo>)</mo><mo>)</mo>""")
        + f'<p><i>x</i> 是中心化的商品向量，<i>r</i> 是各级输入残差，<i>c</i> 是选中的码向量。'
          f'第一个量化项更新码本，第二个约束编码器残差靠近已选码。sg 表示停止梯度。'
          f'实验取 λ={config["quant_loss_weight"]:g}、β={config["beta"]:g}。</p>'
        + figure("rqvae", "左图为逐轮损失及每 25 轮的分块均值。右图为每 50 轮的暖商品验证误差，圆点标示选中权重。")
        + f'<p>训练在第 {end:,} 轮触发验证平台早停，记录耗时 {summary["training_seconds"]:.2f} 秒。'
          f'选中第 {chosen:,} 轮，暖商品 MSE 为 {summary["best_val_mse"]:.5f}。'
          f'末轮的训练重建 MSE 已降至 {float(raw[-1]["train_recon"]):.5f}，'
          f'暖商品验证 MSE 回升到 {history[-1]["val"]["mse"]:.5f}，因此导出采用更早的检查点。</p>'
        + f'<p>选中权重的逐层近邻分配得到 {raw_codes["all"]["unique_sids"]:,} 个不同 SID。'
          f'固定前两位并以最小残差距离匹配第三位后，{exported["unique_sids"]:,} 件商品的三位 SID 一一对应。'
          f'暖商品 MSE 为 {after:.5f}，比原分配增加 {(after / raw_codes["warm_holdout"]["mse"] - 1) * 100:.2f}%。</p>'
        + download("rqvae-epochs", "下载逐轮训练损失")
        + download("rqvae-validation", "下载量化验证曲线")
        + download("rqvae-codes", "下载选中权重的 SID 编码统计")
    )


def section_sft(run, config, output):
    root = run / "epochs3" / "sft"
    stats = read(root / "training_summary.json")
    history = read(root / "trainer_state.json")["log_history"]
    before = read(root / "baseline_validation.json")
    selected = int(Path(stats["best_checkpoint"]).name.rsplit("-", 1)[-1])
    train = [row for row in history if "loss" in row]
    evaluations = [(i, row) for i, row in enumerate(history) if "eval_loss" in row]
    replay_index, replay = evaluations[-1]
    if replay["eval_loss"] != stats["final_eval_metrics"]["eval_loss"]:
        raise ValueError("SFT replay loss does not match the exported checkpoint")
    valid = [(row["step"], row["eval_loss"]) for _, row in evaluations[:-1]]
    if len(train) != stats["global_step"] or (selected, stats["best_metric"]) not in valid:
        raise ValueError("SFT logs do not cover the final and the selected steps")
    keys = ("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm")
    rows = []
    for i, row in enumerate(history):
        if "step" not in row or not ("loss" in row or "eval_loss" in row):
            continue
        phase = "validation" if "eval_loss" in row else "training"
        rows.append({"phase": "restored_checkpoint_validation" if i == replay_index else phase,
                     **{key: row.get(key, "") for key in keys}})
    csv_file(output, "sft", ("phase", *keys), rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.25), layout="constrained")
    trend(axes[0], [(r["step"], r["loss"]) for r in train], BLUE, "100-step mean")
    axes[0].set(xlabel="Optimizer step", ylabel="Cross-entropy", title="Training objective")
    axes[0].legend(frameon=False)
    axes[1].plot(*zip(*valid), marker="o", markersize=2.5, color=BLUE, linewidth=1.3)
    selected_point(axes[1], selected, stats["best_metric"])
    axes[1].set(xlabel="Optimizer step", ylabel="Cross-entropy", title="Validation")
    axes[2].plot([r["step"] for r in train], [r["learning_rate"] for r in train], color=BLUE)
    axes[2].set(xlabel="Optimizer step", ylabel="Learning rate", title="Schedule")
    save(fig, output, "sft")
    hp, model = config["hyperparameters"], config["model"]

    return (
        f'<p>用 Qwen2.5-0.5B 初始化，在词表中新增 {model["added_sid_tokens"]} 个原子 SID token。'
        f'训练共 {stats["epoch"]:g} 轮，峰值学习率 {hp["learning_rate"]:g}，前 {hp["warmup_steps"]} 步预热后线性衰减。'
        f'全局批量为 {hp["global_batch_size"]}，最大序列长度为 {hp["cutoff_len"]} token。</p>'
        '<h3>训练损失</h3>'
        + formula("只在答案 token 上计算的自回归交叉熵", """
        <msub><mi>ℒ</mi><mtext>SFT</mtext></msub><mo>=</mo>
        <mo>−</mo><mfrac>
          <mrow><munder><mo>∑</mo><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></munder>
            <msub><mi>m</mi><mrow><mi>b</mi><mi>t</mi></mrow></msub>
            <mi mathvariant="normal">log</mi><msub><mi>p</mi><mi>θ</mi></msub><mo>(</mo>
            <msub><mi>y</mi><mrow><mi>b</mi><mi>t</mi></mrow></msub><mo>∣</mo>
            <msub><mi>h</mi><mi>b</mi></msub><mo>,</mo>
            <msub><mi>y</mi><mrow><mi>b</mi><mo>,</mo><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>)</mo></mrow>
          <mrow><munder><mo>∑</mo><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></munder>
            <msub><mi>m</mi><mrow><mi>b</mi><mi>t</mi></mrow></msub></mrow>
        </mfrac>""")
        + '<p>掩码 <i>m</i> 在有效答案位置取 1，在提示和填充位置取 0。'
          '目标答案之前的 token 作为上下文，每个参与监督的 token 贡献一项负对数概率。</p>'
        + figure("sft", "监督微调的训练损失、选点验证损失与学习率。左图淡线为逐步读数，实线为每 100 步均值。")
        + f'<p>模型在 {stats["global_step"]:,} 步跑满三轮，耗时 {stats["train_metrics"]["train_runtime"]/60:.2f} 分钟。'
          f'微调前验证交叉熵为 {before["loss"]:.4f}，第 {selected:,} 步达到最低验证损失 {stats["best_metric"]:.4f}。'
          '导出权重对应最低验证损失的检查点，作为 GRPO 的起点。</p>'
        + download("sft", "下载 SFT 曲线与验证点")
    )


def section_grpo(run, config, output):
    root = run / "rl"
    stats = read(root / "training_summary.json")
    history = read(root / "trainer_state.json")["log_history"]
    before = read(root / "baseline_validation.json")["metrics"]["ndcg10"]
    hp = config["hyperparameters"]
    selected = int(Path(stats["best_checkpoint"]).name.rsplit("-", 1)[-1])
    curve = [row for row in history if "kl" in row and "loss" in row]
    if len(curve) != stats["global_step"]:
        raise ValueError("GRPO optimizer step log is incomplete")
    validation = {0: before}
    for row in history:
        if "eval_ndcg10" in row:
            # The final replay at the same step uses restored best weights.
            validation.setdefault(row["step"], row["eval_ndcg10"])
    if validation[selected] != stats["best_metric"]:
        raise ValueError("GRPO validation score disagrees with the selected checkpoint")
    keys = ("step", "loss", "kl", "rewards/rule_reward", "rewards/ndcg_rule_reward", "grad_norm",
            "zero_reward_group_rate", "learning_rate")
    csv_file(output, "grpo-steps", keys, [{key: row[key] for key in keys} for row in curve])
    csv_file(output, "grpo-validation", ("step", "sid_ndcg_at_10", "selected"), [
        {"step": step, "sid_ndcg_at_10": val, "selected": step == selected} for step, val in validation.items()
    ])
    fig, axes = plt.subplots(2, 3, figsize=(11.9, 5.45), layout="constrained")
    panels = (
        ("loss", "GRPO loss", BLUE, .01),
        ("rewards/rule_reward", "Exact SID reward", TEAL, None),
        ("kl", "KL estimate", ORANGE, 10),
        ("grad_norm", "Gradient norm before clipping", BLUE, 1),
        ("zero_reward_group_rate", "Zero-reward group ratio", ORANGE, None),
    )
    for axis, (key, title, color, linthresh) in zip(axes.flat, panels):
        trend(axis, [(r["step"], r[key]) for r in curve], color, "50-step mean", size=50)
        axis.set(xlabel="Optimizer step", title=title)
        if linthresh is not None:
            axis.set_yscale("symlog", linthresh=linthresh)
    axis = axes[1, 2]
    axis.plot(list(validation), [100 * v for v in validation.values()], color=BLUE, marker="o", markersize=3)
    selected_point(axis, selected, 100 * stats["best_metric"])
    axis.set(xlabel="Optimizer step", ylabel="NDCG@10 (%)", title="Validation ranking")
    save(fig, output, "grpo")
    tail = curve[-100:]
    peak = max(curve, key=lambda row: row["kl"])

    return (
        f'<p>以选中的 SFT 权重初始化策略和参考模型。每个提示用受限 beam 产生 {hp["num_generations"]} 个合法 SID，'
        '按目标 SID 命中和候选顺位打分。命中的候选得到 1。'
        '若组内有命中，未命中候选按顺位获得归一化的负对数折扣。全组未命中时奖励均为零。'
        f'每 {hp["ref_model_sync_steps"]} 步按系数 {hp["ref_model_mixup_alpha"]:g} 将策略权重混入参考模型。</p>'
        '<h3>训练损失</h3>'
        + formula("组标准化优势和逐 token KL 估计", """
        <msub><mi>A</mi><mi>i</mi></msub><mo>=</mo>
        <mfrac><mrow><msub><mi>R</mi><mi>i</mi></msub><mo>−</mo><mover><mi>R</mi><mo>¯</mo></mover></mrow>
        <mrow><msub><mi>σ</mi><mi>R</mi></msub><mo>+</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow></mfrac>
        <mspace width="1.4em"/>
        <msub><mi>κ</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>=</mo>
        <msup><mi>e</mi><msub><mi>δ</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub></msup><mo>−</mo>
        <msub><mi>δ</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>−</mo><mn>1</mn>""")
        + formula("GRPO 策略损失与参考模型正则项", """
        <msub><mi>ℒ</mi><mtext>GRPO</mtext></msub><mo>=</mo>
        <mfrac><mn>1</mn><mi>N</mi></mfrac>
        <munder><mo>∑</mo><mi>i</mi></munder>
        <mfrac><mn>1</mn><msub><mi>T</mi><mi>i</mi></msub></mfrac>
        <munderover><mo>∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>T</mi><mi>i</mi></msub></munderover>
        <mrow><mo>[</mo><mo>−</mo><msub><mi>A</mi><mi>i</mi></msub>
        <msup><mi>e</mi><mrow><msub><mi>z</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>−</mo>
        <mi mathvariant="normal">sg</mi><mo>(</mo><msub><mi>z</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>)</mo></mrow></msup>
        <mo>+</mo><mi>β</mi><msub><mi>κ</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>]</mo></mrow>""")
        + f'<p><i>A</i> 用同一提示组的奖励均值和样本标准差计算。<i>z</i> 是策略的 token 对数概率，'
          'δ 是参考模型对数概率减去 <i>z</i>。先对每条答案的 <i>T</i><sub>i</sub> 个有效 token 取平均，再对 '
          f'<i>N</i> 个候选取平均，β={hp["beta"]:g}。</p>'
        + figure("grpo", "淡线为逐步值，实线为每 50 步均值。损失、KL 与梯度范数采用对称对数纵轴，以同时展示峰值和常态。验证曲线圆点标示选中权重。")
        + f'<p>验证 SID NDCG@10 由训练前的 {percent(before)} 提升至第 {selected:,} 步的 '
          f'{percent(stats["best_metric"])}。随后连续 {hp["early_stopping_patience"]} 次验证未超过该值，'
          f'第 {stats["global_step"]:,} 步早停，耗时 {stats["train_metrics"]["train_runtime"]/60:.2f} 分钟。</p>'
        + f'<p>第 {peak["step"]:,} 步的 KL 达到本次峰值 {peak["kl"]:,.2f}，'
          f'同一步的裁剪前梯度范数为 {peak["grad_norm"]:,.2f}。末 100 步的平均 KL 为 {mean(r["kl"] for r in tail):.3f}，'
          f'无奖励候选组比例为 {percent(mean(r["zero_reward_group_rate"] for r in tail))}。'
          '无命中组的组内优势为零，KL 仍约束策略偏离。验证选点依据商品 SID 的排名表现。</p>'
        + download("grpo-steps", "下载 GRPO 训练目标与诊断")
        + download("grpo-validation", "下载 GRPO 验证选点")
    )


def section_unisrec(run, output):
    root = run / "unisrec"
    summary = read(root / "status.json")
    vectors = read(root / "test/report.json")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    text = (root / "train.log").read_text(encoding="utf-8")
    pattern = r"epoch (\d+) (?:training|evaluating) \[time: ([\d.]+)s, (train loss|valid_score): ([\d.]+)\]"
    points = {"train loss": {}, "valid_score": {}}
    for epoch, seconds, key, value in re.findall(pattern, text):
        points[key][int(epoch) + 1] = (float(seconds), float(value))
    epochs = list(range(1, config["epochs"] + 1))
    if any(sorted(series) != epochs for series in points.values()):
        raise ValueError("UniSRec epoch trace is incomplete")
    picked = summary["best_epoch"]
    measures = [{"epoch": i, "train_seconds": points["train loss"][i][0],
                 "train_batch_loss_sum": points["train loss"][i][1],
                 "validation_seconds": points["valid_score"][i][0],
                 "validation_nll": points["valid_score"][i][1]} for i in epochs]
    csv_file(output, "unisrec", tuple(measures[0]), measures)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), layout="constrained")
    axes[0].plot(epochs, [x["train_batch_loss_sum"] for x in measures], color=BLUE, marker="o", markersize=3)
    axes[0].set(xlabel="Epoch", ylabel="Sum of batch losses", title="Training catalogue CE")
    axes[1].plot(epochs, [x["validation_nll"] for x in measures], color=BLUE, marker="o", markersize=3)
    selected_point(axes[1], picked, summary["validation_nll"])
    axes[1].set(xlabel="Epoch", ylabel="NLL", title="Full catalogue validation")
    save(fig, output, "unisrec")
    provenance = vectors["evaluation"]["qwen_provenance"]["architecture"]
    return (
        f'<p>通过历史序列编码器和商品向量适配器学习归一化的查询向量 <i>q</i> 与商品向量 <i>v</i>。'
        f'向量空间为 {provenance["hidden_size"]} 维，训练批量 {config["train_batch_size"]}，'
        f'学习率预热 {config["warmup_epochs"]} 轮后按余弦调度。</p>'
        '<h3>训练损失</h3>'
        + formula("训练目录上的下一商品交叉熵", """
        <msub><mi>ℒ</mi><mtext>UniSRec</mtext></msub><mo>=</mo>
        <mo>−</mo><mfrac><mn>1</mn><mi>B</mi></mfrac><munder><mo>∑</mo><mi>b</mi></munder>
        <mi mathvariant="normal">log</mi><mfrac>
          <mrow><mi mathvariant="normal">exp</mi><mo>(</mo><msubsup><mi>q</mi><mi>b</mi><mo>⊤</mo></msubsup>
                <msub><mi>v</mi><mrow><mi>y</mi><mo>(</mo><mi>b</mi><mo>)</mo></mrow></msub>
                <mo>/</mo><mi>τ</mi><mo>)</mo></mrow>
          <mrow><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><msub><mi>𝒞</mi><mtext>train</mtext></msub></mrow></munder>
                <mi mathvariant="normal">exp</mi><mo>(</mo>
                <msubsup><mi>q</mi><mi>b</mi><mo>⊤</mo></msubsup><msub><mi>v</mi><mi>j</mi></msub>
                <mo>/</mo><mi>τ</mi><mo>)</mo></mrow>
        </mfrac>""")
        + f'<p><i>B</i> 为训练批量，<i>y</i>(<i>b</i>) 是真实下一商品。'
          f'分母为训练候选目录内商品，τ={vectors["evaluation"]["temperature"]:g}。'
          '验证在全商品目录计算 NLL，两个阶段使用各自的候选范围和归约口径。</p>'
        + figure("unisrec", "训练曲线为各批交叉熵之和。验证曲线为全目录 NLL，圆点标示选中权重。")
        + f'<p>第 {picked} 轮达到最低验证 NLL {summary["validation_nll"]:.6f}，并导出用于投机推理的'
          f'历史编码器和 {vectors["evaluation"]["candidate_count"]:,} 个商品向量。</p>'
        + download("unisrec", "下载 UniSRec 训练和验证曲线")
    )


def section_checkpoints(run, output):
    q = read(run / "rqvae/training_summary.json")
    sft = read(run / "epochs3/sft/training_summary.json")
    rl = read(run / "rl/training_summary.json")
    uni = read(run / "unisrec/status.json")
    records = [
        {"stage": "RQ-VAE", "budget": f'{q["epochs_completed"]:,} 轮', "checkpoint": f'第 {q["best_epoch"]:,} 轮',
         "validation_metric": "暖商品重建 MSE", "value": f'{q["best_val_mse"]:.5f}'},
        {"stage": "SFT", "budget": f'{sft["global_step"]:,} 步', "checkpoint": Path(sft["best_checkpoint"]).name,
         "validation_metric": "答案 token 交叉熵", "value": f'{sft["best_metric"]:.4f}'},
        {"stage": "GRPO", "budget": f'{rl["global_step"]:,} 步', "checkpoint": Path(rl["best_checkpoint"]).name,
         "validation_metric": "SID NDCG@10", "value": percent(rl["best_metric"])},
        {"stage": "UniSRec", "budget": f'{uni["best_epoch"]} 轮', "checkpoint": f'第 {uni["best_epoch"]} 轮',
         "validation_metric": "全目录 NLL", "value": f'{uni["validation_nll"]:.6f}'},
    ]
    csv_file(output, "checkpoints", tuple(records[0]), records)
    return (
        table(("阶段", "实跑规模", "选中权重", "验证依据", "选点指标"), [tuple(r.values()) for r in records])
        + '<p>RQ-VAE 的产物是完整商品 SID。SFT 学得按历史生成 SID 的策略，GRPO 在同一输出空间内调整排名。'
          '投机推理使用 UniSRec 的商品向量召回和 GRPO 的 SID 打分。</p>'
        + download("checkpoints", "下载训练检查点摘要")
        + '<p><a href="validation.html">查看新模型的排序、冷目标和服务性能</a></p>'
    )


def render(workspace: Path, run: Path, output: Path):
    """Return the standalone training page body and its table of contents."""
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    preparation = read(run / "data/preparation_manifest.json")
    sft = read(run / "epochs3/sft/run_config.json")
    rl = read(run / "rl/run_config.json")
    contents = [
        section_data(preparation, sft, rl, output), section_rqvae(run, output), section_sft(run, sft, output),
        section_grpo(run, rl, output), section_unisrec(run, output), section_checkpoints(run, output),
    ]
    return "".join(f'<section><h2 id="{key}">{title}</h2>{html}</section>'
                   for (key, title), html in zip(SECTIONS, contents, strict=True)), list(SECTIONS)
