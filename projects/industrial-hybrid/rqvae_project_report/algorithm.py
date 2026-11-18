"""Render the algorithm chapter from the completed Industrial RQ-VAE run."""

from html import escape
import json
from pathlib import Path

from project_report.algorithm import table


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def math(markup, label):
    return (f'<div class="formula"><math xmlns="http://www.w3.org/1998/Math/MathML" '
            f'display="block" aria-label="{escape(label, quote=True)}">{markup}</math></div>')


def inference_diagram():
    """Draw the implemented verification, retry, and fallback paths."""
    colors = {
        "input": ("#eef2f8", "#6481a7", "#223a5a"),
        "retrieve": ("#eaf6f3", "#4e9583", "#215849"),
        "verify": ("#edf2f9", "#6888ae", "#294c76"),
        "decision": ("#fff3df", "#b38742", "#75531c"),
        "fallback": ("#f4f1f9", "#8b72a9", "#504269"),
        "output": ("#173c51", "#173c51", "#ffffff"),
    }

    def node(x, y, width, height, theme, label, detail):
        fill, border, ink = colors[theme]
        center = x + width / 2
        return (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" '
            f'fill="{fill}" stroke="{border}" stroke-width="1.5"/>'
            f'<text x="{center:g}" y="{y + 27}" font-size="19" font-weight="700" fill="{ink}">{escape(label)}</text>'
            f'<text x="{center:g}" y="{y + 49}" font-size="14" fill="{ink}">{escape(detail)}</text>'
        )

    nodes = (
        (296, 16, 308, 62, "input", "用户历史商品", "输入有序商品 ID，构造历史 SID 序列"),
        (56, 125, 313, 64, "retrieve", "UniSRec 检索候选", "首轮全库检索，跨轮去重"),
        (531, 125, 313, 64, "verify", "Qwen 历史预填充", "一次计算并保留历史 KV"),
        (297, 233, 306, 66, "verify", "SID 批量验证", "按三位条件对数概率评分"),
        (327, 329, 246, 66, "decision", "检索席位填满？", "或达到停止条件"),
        (16, 330, 230, 66, "retrieve", "推进 SID 前缀", "限制下一轮候选目录"),
        (654, 330, 230, 66, "fallback", "受约束 beam 补位", "去重后填补输出空缺"),
        (292, 469, 316, 66, "output", "合并入选商品并重排", "输出最多 K 件商品"),
    )
    connectors = (
        "M450 78 V99 H212.5 V124", "M450 78 V99 H687.5 V124",
        "M450 211 V232",
        "M450 299 V328", "M327 363 H247", "M573 363 H653",
        "M131 330 V189", "M450 395 V468", "M769 396 V442 H581 V468",
    )
    arrows = "".join(f'<path d="{d}" marker-end="url(#rq-infer-arrow)"/>' for d in connectors)
    labels = ((285, 346, "不足可继续"), (614, 346, "不足且停止"), (104, 289, "下一轮"), (479, 436, "已填满"))
    branch_names = "".join(
        f'<text x="{x}" y="{y}" fill="#455971" font-size="14">{escape(label)}</text>'
        for x, y, label in labels
    )
    return (
        '<figure><div style="overflow-x:auto"><svg viewBox="0 0 900 552" role="img" '
        'aria-labelledby="rq-flow-title rq-flow-desc" '
        'style="display:block;width:100%;height:auto;min-width:610px;max-width:900px;margin-inline:auto">'
        '<title id="rq-flow-title">联合投机推理流程图</title>'
        '<desc id="rq-flow-desc">UniSRec 检索与 Qwen 历史编码共同进入批量验证。'
        '已接受商品填满席位后合并排序，仍可草拟时推进前缀，未填满且停止时用受约束 beam 补位。</desc>'
        '<defs><marker id="rq-infer-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0 0 L7 3.5 L0 7 Z" fill="#58718c"/></marker></defs>'
        '<g fill="none" stroke="#58718c" stroke-width="1.6" stroke-linejoin="round">'
        f'<path d="M212.5 189 V211 H687.5 V189"/>{arrows}</g>'
        '<g text-anchor="middle" style="font-family:inherit">'
        + "".join(node(*entry) for entry in nodes)
        + branch_names
        + '</g></svg></div><figcaption>联合投机推理的三条决策路径。检索和生成分别确定入选商品，再对合并结果排序。</figcaption></figure>'
    )


def quantizer(rq, sid, emb):
    config = rq["config"]
    dims = [emb, *config["layers"], config["e_dim"]]
    spans = " → ".join(f"{part:,}" for part in dims)
    back = " → ".join(f"{part:,}" for part in reversed(dims))
    layers = "、".join(str(x) for x in sid["occupied_codes_by_level"])
    return f"""
<p>RQ-VAE 把商品的 {emb:,} 维文本向量编码为三位 SID。中心化均值仅由 {rq['split_counts']['train']:,} 件量化训练商品计算。
另用 {rq['split_counts']['val']:,} 件暖商品按重建 MSE 选择权重，冻结后为 {rq['split_counts']['cold']} 件冷商品编码。</p>
""" + table(("组件", "模型架构"), (
        ("编码器", f"对中心化特征做 MLP 投影，维度 {spans}。隐藏层使用 ReLU。"),
        ("残差量化器", f"三层依次量化残差。每层 {config['num_emb_list'][0]} 个 {config['e_dim']} 维码向量，逐层做 K-means 初始化。"),
        ("解码器", f"输入是三层码向量之和，镜像 MLP 维度 {back}，最后加回均值。"),
    )) + (
        '<p>编码器先得到潜向量。每层选取距当前残差最近的码向量，再减去该向量，把余量交给下一层。</p>'
        + math("""
<mrow>
  <msub><mi>r</mi><mn>0</mn></msub><mo>=</mo><mi>E</mi><mo>(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo>)</mo>
  <mo>,</mo><mspace width="1em"/>
  <msub><mi>j</mi><mi>ℓ</mi></msub><mo>=</mo><munder><mi mathvariant="normal">arg min</mi><mi>j</mi></munder>
  <msup><mrow><mo>‖</mo><msub><mi>r</mi><mrow><mi>ℓ</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>−</mo><msub><mi>e</mi><mrow><mi>ℓ</mi><mo>,</mo><mi>j</mi></mrow></msub><mo>‖</mo></mrow><mn>2</mn></msup>
</mrow>""", "中心化编码，逐层寻找与残差最近的码") +
        f'<p>每次从残差中减去已选中的码向量。按暖商品重建误差选中第 {rq["best_epoch"]:,} 轮。'
        '导出时在同一两位前缀内保留已唯一的第三位，把发生冲突的商品与尚未使用的第三位码做一对一匹配，'
        f'最小化分配后的残差平方距离之和。三层分别使用 {layers} 个码，'
        f'{sid["catalog_items"]:,} 件商品得到 {sid["unique_sids"]:,} 个唯一三位 SID。</p>'
    )


def supervised(qwen, sft, best):
    tasks = (
        ("历史 SID", "下一商品 SID", "sid_sequence_recommendation"),
        ("历史 SID", "下一商品标题", "history_to_title"),
        ("商品 SID", "商品标题", "sid2title"),
        ("商品标题", "商品 SID", "title2sid"),
        ("历史商品标题", "下一商品 SID", "title_history_to_sid"),
    )
    return f"""
<p>生成模型采用 Qwen2.5-0.5B decoder-only Transformer。输入嵌入依次进入 {qwen['num_hidden_layers']} 层因果解码块，
隐藏宽度 {qwen['hidden_size']}。每块使用 {qwen['num_attention_heads']} 个查询头与 {qwen['num_key_value_heads']} 个 KV 头的分组查询注意力，
位置采用 RoPE，归一化使用 RMSNorm，SwiGLU 前馈中间宽度为 {qwen['intermediate_size']:,}。
输入词嵌入与输出投影共享权重，当前词表含 {qwen['vocab_size']:,} 个 token。</p>
<p>SFT 把商品语义编码与自然语言标题任务放入同一模型，使用因果掩码做答案 token 的监督训练。
令 <var>x</var> 为输入，<var>y</var> 为答案，<var>m</var> 为有效答案掩码。</p>
""" + math("""
<mrow>
  <msub><mi>ℒ</mi><mtext>SFT</mtext></msub><mo>=</mo><mo>−</mo>
  <mfrac>
    <mrow><munder><mo>∑</mo><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></munder>
      <msub><mi>m</mi><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></msub>
      <mi mathvariant="normal">log</mi><msub><mi>p</mi><mi>θ</mi></msub>
      <mo>(</mo><msub><mi>y</mi><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>∣</mo>
      <msub><mi>x</mi><mi>b</mi></msub><mo>,</mo><msub><mi>y</mi><mrow><mi>b</mi><mo>,</mo><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>)</mo>
    </mrow>
    <mrow><munder><mo>∑</mo><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></munder>
      <msub><mi>m</mi><mrow><mi>b</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow>
  </mfrac>
</mrow>""", "SFT 损失等于有效答案 token 的平均条件负对数概率") + table(
        ("输入", "监督目标", "训练样本"), [
            (prompt, target, f"{sft['task_counts'][key]:,}") for prompt, target, key in tasks
        ]
    ) + f'<p>按验证损失选择第 {int(best["best_checkpoint"].split("-")[-1]):,} 步的权重，为后续策略训练提供初始化。</p>'


def policy(rl, best, qwen):
    h = rl["hyperparameters"]
    return f"""
<p>GRPO 维护策略模型和参考模型。两者均为 {qwen['num_hidden_layers']} 层、隐藏宽度 {qwen['hidden_size']} 的 Qwen 解码器，
从选中的 SFT 权重初始化。
策略模型在受约束 beam 中为每个提示产生 {h['num_generations']} 个候选，再根据组内奖励更新。
参考模型提供 token 级对数概率，每 {h['ref_model_sync_steps']} 个优化步做权重混合更新。</p>
<p>完整 SID 命中得 +1。组内出现命中时，错误候选根据名次获得归一化负折扣，高位错误惩罚较大。全组无命中时奖励为 0。</p>
""" + math("""
<mrow>
  <msub><mover><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo>
  <mfrac><mrow><msub><mi>R</mi><mi>i</mi></msub><mo>−</mo><mover><mi>R</mi><mo>¯</mo></mover></mrow>
  <mrow><mi mathvariant="normal">std</mi><mo>(</mo><mi>R</mi><mo>)</mo><mo>+</mo>
  <msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow></mfrac>
  <mo>,</mo><mspace width="1em"/>
  <mi>κ</mi><mo>=</mo><msup><mi mathvariant="normal">e</mi><mi>Δ</mi></msup><mo>−</mo><mi>Δ</mi><mo>−</mo><mn>1</mn>
</mrow>""", "组内归一化优势，及逐 token KL 估计") + """
<p><var>R</var> 为同一提示下的奖励组，<var>Δ</var> 为参考模型减策略模型的 token 对数概率。
逐 token 目标如下，其中 <var>ℓ</var> 为策略对数概率，<span class="math-term">sg</span> 表示停止梯度。</p>
""" + math("""
<mrow>
  <msub><mi>ℒ</mi><mrow><mi mathvariant="normal">token</mi><mo>,</mo><mi>i</mi></mrow></msub><mo>=</mo>
  <mo>−</mo><msup><mi mathvariant="normal">e</mi><mrow><mi>ℓ</mi><mo>−</mo><mi mathvariant="normal">sg</mi><mo>(</mo><mi>ℓ</mi><mo>)</mo></mrow></msup>
  <msub><mover><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>+</mo><mi>β</mi><mi>κ</mi>
</mrow>""", "GRPO 的逐 token 目标为负策略优势与 KL 惩罚之和") + (
        f'<p>训练先对每条候选的有效生成 token 求平均，再对候选求平均。KL 权重为 {h["beta"]:.3f}。'
        f'验证 NDCG@10 选中第 {int(best["best_checkpoint"].split("-")[-1]):,} 步。'
        '该策略权重用于在线 SID 验证和受约束生成。</p>'
    )


def retrieval(uni, checkpoint, warm_items):
    return f"""
<p>UniSRec 读取冻结的 {uni['plm_size']:,} 维商品文本向量。历史与商品分支共用含 {uni['n_exps']} 个专家的门控适配器，
投影到 {uni['hidden_size']} 维。商品向量在适配后做 L2 归一化。历史序列叠加位置嵌入后进入 {uni['n_layers']} 层因果 Transformer 编码器，
每层 {uni['n_heads']} 个注意力头，前馈宽度 {uni['inner_size']}。取最后有效位置作为历史表示，再做 L2 归一化。</p>
<p>训练将真实下一商品作为正类，将历史表示和 {warm_items:,} 件训练可见商品的内积除以温度 {uni['temperature']:g} 后计算交叉熵。
推理使用第 {checkpoint['best_epoch']} 轮的历史编码器和全库归一化向量，按内积召回。每个请求的历史查询向量可在多轮检索间复用。</p>
"""


def joint_inference():
    return (
        inference_diagram()
        + '<ol><li><strong>草拟。</strong> 首轮从全库检索。后续按当前合法 SID 前缀缩小候选池，并排除该请求已经草拟过的商品 ID。</li>'
          '<li><strong>验证。</strong> 复用历史 KV。历史末位 logits 给出第一位 SID 概率，将候选前两位作为独立分支输入，一次前向取得后两位的概率。'
          '使用全词表归一化后的对数概率，批量计算完整 SID 得分。满足当前接受条件的商品进入排序池。</li>'
          '<li><strong>分配与补位。</strong> 按得分保留检索席位。席位不足且还有轮次时，前缀逐位延长后重草拟。'
          '达到停止条件后按受约束 beam 的生成顺序去重补位。检索和 beam 都入选时只对合并集合做统一排序。'
          '纯 beam 路径沿用生成顺序。</li></ol>'
        + math("""
<mrow>
  <mi>v</mi><mo>(</mo><mi>i</mi><mo>,</mo><mi>H</mi><mo>)</mo><mo>=</mo>
  <mfrac><mn>1</mn><mn>3</mn></mfrac><munderover><mo>∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mn>3</mn></munderover>
  <mi mathvariant="normal">log</mi><msub><mi>p</mi><mi>θ</mi></msub>
  <mo>(</mo><msub><mi>c</mi><mi>t</mi></msub><mo>∣</mo><mi>H</mi><mo>,</mo>
  <msub><mi>c</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>)</mo>
</mrow>""", "联合投机推理的商品分数是三位 SID 的条件对数概率均值")
        + '<p><var>H</var> 是用户历史提示，<var>c</var> 是商品的三位 SID。历史、换行和 EOS 排除在评分范围外。'
          '已知候选可并行验证，同一请求共享的前缀分支可以复用中间计算。</p>'
        + table(("推理参数", "采用值"), (
            ("输出上限 K", "1、5、10、20、50"),
            ("每轮草拟上限", "2K"),
            ("检索份额 α", "1"),
            ("最大草拟轮数", "3"),
            ("引导前缀 beam 宽度", "1"),
            ("完整生成 beam 宽度", "50"),
        ))
    )


def render(workspace: Path, run: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    del workspace, output
    prep = read(run / "data/preparation_manifest.json")
    rq = read(run / "rqvae/training_summary.json")
    sid = read(run / "sid_export_summary.json")
    qwen = read(run / "epochs3/sft/final_checkpoint/config.json")
    rl_qwen = read(run / "rl/final_checkpoint/config.json")
    sft = read(run / "epochs3/sft/run_config.json")
    sft_best = read(run / "epochs3/sft/training_summary.json")
    rl = read(run / "rl/run_config.json")
    rl_best = read(run / "rl/training_summary.json")
    uni = read(run / "unisrec/export/metadata.json")
    uni_best = read(run / "unisrec/status.json")
    architecture = uni["provenance"]["architecture"]
    model_keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")
    if any(qwen[key] != rl_qwen[key] for key in model_keys):
        raise ValueError("Expected the same decoder architecture across SFT and GRPO")
    if prep["original_sid"] or prep["unique_sids"] != sid["catalog_items"] or rq["best_epoch"] != sid["selected_epoch"]:
        raise ValueError("Algorithm report requires the completed RQ-VAE catalog")
    fingerprint = prep["expected_sid_sha256"]
    if not (fingerprint == sid["sid_sha256"] == sft["data"]["sha256"]["sid_index_path"]
            == rl["data"]["sha256"]["sid_index_path"] == uni_best["sid_sha256"]):
        raise ValueError("Algorithm report received mismatched training stages")
    select = read(run / "inference/report/selection.json")
    chosen = [item for item in select["locked"] if item["id"] in select["winners"].values()]
    if {item["k"] for item in chosen} != {1, 5, 10, 20, 50} or any(
        (item["rho"], item["alpha"], item["prefix_beams"], item["max_rounds"]) != (2, 1, 1, 3)
        for item in chosen
    ):
        raise ValueError("Algorithm parameters did not match the frozen evaluation routes")

    rows = prep["rows"]
    intro = (
        f'<p>工业品目录含 {sid["catalog_items"]:,} 件商品。固定切分有 {rows["train"]:,} 条训练、'
        f'{rows["valid"]:,} 条验证与 {rows["test"]:,} 条测试请求，每条历史最多 10 件商品。'
        f'按训练历史和目标中的商品覆盖定义冷暖，测试有 {sid["cold_test_rows"]} 条冷目标。</p>'
        '<p>离线先学习商品的三位语义 ID，再训练 Qwen 的 SFT 和 GRPO，以及独立的 UniSRec 检索模型。'
        '在线由 UniSRec 提出商品，Qwen 验证完整 SID。最终输出以商品 ID 标识的推荐排序。</p>'
    )
    sections = [
        ("algorithm-data", "系统架构与数据流", intro),
        ("algorithm-rqvae", "RQ-VAE 模型与语义编码", quantizer(rq, sid, architecture["plm_size"])),
        ("algorithm-sft", "生成模型架构与监督微调", supervised(qwen, sft, sft_best)),
        ("algorithm-grpo", "策略模型与 GRPO", policy(rl, rl_best, rl_qwen)),
        ("algorithm-unisrec", "UniSRec 检索模型", retrieval(
            architecture, uni_best, sid["catalog_items"] - rq["split_counts"]["cold"]
        )),
        ("algorithm-inference", "联合投机推理", joint_inference()),
    ]
    toc = [(anchor, title) for anchor, title, _ in sections]
    return "\n".join(f'<section><h2 id="{anchor}">{title}</h2>{body}</section>'
                     for anchor, title, body in sections), toc
