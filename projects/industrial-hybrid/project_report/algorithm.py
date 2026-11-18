"""Render the algorithm report from the Industrial project manifests."""
import gzip
import json
from html import escape
from pathlib import Path


SFT = "Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1/epochs3/sft"
RL = "Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl"
HYBRID = "Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118"
KS = (1, 5, 10, 20, 50)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    cells = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table></div>"


def pre(text):
    return f"<pre><code>{escape(text.strip())}</code></pre>"


def example(run, config):
    predictions = run / "report" / "predictions" / "validation_winners.jsonl.gz"
    with gzip.open(predictions, "rt", encoding="utf-8") as handle:
        selected = {row["row_id"]: row for row in map(json.loads, handle) if row["config_id"] == config["id"]}
    for path in sorted((run / "test").glob("rank[0-3].jsonl")):
        with path.open(encoding="utf-8") as handle:
            for source in map(json.loads, handle):
                target = int(source["target"])
                prediction = selected.get(source["row_id"])
                drafts = source["drafts"]["unisrec"]["20"]["first"]
                # In the new archive, ratio 1 at K=20 gives the same unrestricted top 20.
                if prediction is None or target not in drafts or drafts.index(target) < 10:
                    continue
                ranked = prediction["ranking"]
                output_ids = [entry["item_id"] for entry in ranked]
                if target in output_ids and len(drafts) == 20 and all(x["round"] == 1 for x in ranked):
                    assert all(x in drafts for x in output_ids)
                    return source["row_id"], target, drafts.index(target) + 1, output_ids.index(target) + 1, ranked
    raise ValueError("Missing the verified top-20 to top-10 example")


def render(workspace: Path, output: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return the report sections and their navigation anchors."""
    checkpoint = workspace / "weights" / "checkpoints"
    model_root = checkpoint / "MiniOneRec"
    run = model_root / HYBRID
    prepare = read(workspace / "MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_original/preparation_manifest.json")
    catalog = read(run / "catalog.json")
    selection = read(run / "report/selection.json")
    sft = read(model_root / SFT / "run_config.json")
    rl = read(model_root / RL / "run_config.json")
    qwen = read(model_root / RL / "checkpoint-828/config.json")
    manifest = read(run / "manifest.json")
    uni = read(checkpoint / "UniSRec/industrial_hybrid_inference_ep20_export_v1/metadata.json")
    spec = uni["provenance"]["architecture"]
    sid_record = manifest["data"]["original_sid_index"]
    assert sid_record["sha256"] == prepare["expected_sid_sha256"] == sft["data"]["sha256"]["sid_index_path"] == rl["data"]["sha256"]["sid_index_path"]
    configs = {row["id"]: row for row in selection["locked"]}
    final = {k: configs[selection["arm_winners"][str(k)]["unisrec2"]] for k in KS}
    assert all(c["rho"] == 2 and c["alpha"] == 1 and c["prefix_beams"] == 1 for c in final.values())
    row_id, target, draft_rank, result_rank, ranked = example(run, final[10])

    parts = []
    contents = []

    def section(anchor, title, body):
        contents.append((anchor, title))
        parts.append(f"<section><h2 id='{anchor}'>{title}</h2>{body}</section>")

    section("architecture", "总体架构与数据流", """
<p>最终方案对所有 K 使用 <strong>UniSRec 每轮取 2K</strong>。UniSRec 按商品 ID 从目录召回，Qwen2.5-0.5B 的 RL 检查点按完整 SID 给候选打分。系统先确定检索与 beam 各自的入选商品，再输出最多 K 件。验证报告保留 K 对 2K 的比较，包括 K=20、50 上单独取 K 的结果。</p>
<figure class="algorithm-flow">
<figcaption>图 1 · 离线资产与在线推理流程</figcaption>
<ol>
<li><strong>离线数据</strong> · 复核上游交互切分、商品元数据与预置 SID，生成训练读取格式和冷暖标记。</li>
<li><strong>生成路径</strong> · 商品 SID 与标题构成 SFT 样本 → Qwen2.5-0.5B → GRPO → RL checkpoint-828。</li>
<li><strong>检索路径</strong> · 商品的固定 Qwen 文本特征与训练交互 → UniSRec → 导出历史编码器和全库商品向量。</li>
<li><strong>在线主路径</strong> · 用户商品历史 → UniSRec 召回最多 2K 件 → 按目录与前缀过滤 → RL 批量评分 → 接受并选出检索席位。</li>
<li><strong>在线补位</strong> · 剩余席位由受约束 beam 按原始顺序填补 → 入选集合统一评分排序 → 输出商品 ID、SID、来源、分数和轮次。</li>
</ol>
</figure>
<p>检索阶段的排序单位是商品，验证阶段的打分单位是完整 SID。两件商品共用 SID 时，它们共享验证分数，但依然是不同的可推荐商品。质量评估以精确商品 ID 为主。</p>""")

    splits = prepare["source_audit"]["splits"]
    section("data", "数据清洗、切分与目录", """
<p>上游 Amazon 2018 数据处理流程按时间窗口过滤评论，清洗商品标题并排除无有效标题的商品，迭代执行用户与商品 5-core，按时间构造历史和下一商品，再按全局目标时间做约 8:1:1 切分。本次使用上游预处理快照，其中 2016-10 至 2018-11 来自文件名。归档 CSV 不含时间戳，实际日期边界无法从 CSV 独立核验。本地准备脚本校验 CSV、SID、元数据和值域，保留切分、行序和重复交互，按哈希记录来源。</p>
""" + table(("数据集", "请求数", "切分内用户", "历史长度范围", "完整重复行"), [
        (name, f"{splits[key]['rows']:,}", f"{splits[key]['users']:,}", f"{splits[key]['history_min']}–{splits[key]['history_max']}", splits[key]["duplicate_full_rows"])
        for name, key in (("训练", "train"), ("验证", "valid"), ("测试", "test"))
    ]) + f"""
<p>全局 {prepare['source_audit']['users_across_all_splits']:,} 名用户，目录 {prepare['items']:,} 件商品。历史从早到晚，保留已购商品再次成为目标的可能。暖商品在训练历史或目标中至少出现一次。目录有 {sum(catalog['cold'])} 件冷商品，测试集中有 {prepare['source_audit']['cold_test_rows']} 条冷目标请求。共享商品目录同时提供商品 ID、标题、完整 SID、冷暖状态和训练目标频次。UniSRec 的 RecBole 内部 ID 另做映射，内部 0 是 padding，目录商品 0 有效。</p>
<p>商品文本特征复用原仓的 3,686 × 2,560 Qwen 向量。原始特征生成清单与行 ID 未随资产提供，当前索引按连续目录 ID 推定行序并固定向量哈希。UniSRec 导出另核对了 RecBole 内部 ID 与目录映射。</p>""")

    positions = [len({sid.split("><")[i].strip("<>") for sid in catalog["sid_by_item"]}) for i in range(3)]
    c = prepare["sid_collisions"]
    section("sid", "语义 ID、残差量化与 RQ-VAE", f"""
<p><strong>本次采用的 SID</strong>来自原仓工业品目录。上游 README §3.3 将其描述为预置 RQ-KMeans SID。残差 K-means 逐级对上一级未表示的残差聚类，聚类中心的下标构成离散语义标识。当前工程按哈希复用实际索引，具体训练超参数与原始码本产物未随目录归档。</p>
<p>实现读入的 SID 为三个 token <code>&lt;a_*&gt;&lt;b_*&gt;&lt;c_*&gt;</code>。各槽实测使用 {positions[0]}、{positions[1]}、{positions[2]} 种 token，共 {sum(positions)} 种。{prepare['items']:,} 件商品对应 {prepare['unique_sids']:,} 个唯一 SID，包含 {c['groups']} 个碰撞组和 {c['items']} 件碰撞组内商品。推理只使用目录确认的三位 SID。冷暖商品均评分三位，现有目录没有可单独排除的末尾消歧位。</p>
<h3>RQ-VAE 的可训练结构</h3>
<p>备用流程使用 MLP 编码器 <em>E</em> 将文本向量 <em>x</em> 投影到潜空间，初始化残差 <em>r₀=E(x)</em>。第 <em>l</em> 层在自己的码本中找到最接近残差的向量 <em>eₗ[jₗ]</em>，设置 <em>rₗ=rₗ₋₁−eₗ[jₗ]</em>。各层向量之和进入 MLP 解码器重建 <em>x</em>，各层下标 <em>(j₁,j₂,j₃)</em> 导出为 SID。直通估计让重建损失可以反传至编码器。</p>
<p>总损失为 <em>L=L<sub>recon</sub>+λ·mean(L<sub>VQ,l</sub>)</em>，仓库默认用 MSE 重建。各层量化项为 <em>‖sg(rₗ₋₁)−eₗ‖²+β<sub>commit</sub>‖rₗ₋₁−sg(eₗ)‖²</em>，<em>sg</em> 表示停止梯度。码本损失将中心拉向编码残差，commitment 损失约束编码器输出向已选中心收敛。</p>
<p>上游参考代码使用 2,560 维输入时，默认编码器为 2,560 → 2,048 → 1,024 → 512 → 256 → 128 → 64 → 32，解码器反向镜像，三层码本各有 256 个向量。训练入口 <code>rq/rqvae.py</code>，导出入口 <code>rq/generate_indices.py</code>。该默认架构用于解释参考实现。可复现实验应同时记录重建误差、各层码本利用率、全库 SID 碰撞率与选定检查点。</p>
<p class="note">Industrial 最终链路沿用预置索引。已归档的 Amazon 2018 RQ-VAE 权重和损失日志属于 Video Games，Industrial 报告只展示上述目录统计及 RQ-VAE 算法接口。</p>""")

    s = sft["hyperparameters"]
    r = rl["hyperparameters"]
    counts = sft["task_counts"]
    tasks = (
        ("历史 SID → 下一商品 SID", "sid_sequence_recommendation"),
        ("历史商品标题 → 下一商品标题", "history_to_title"),
        ("商品 SID → 商品标题", "sid2title"),
        ("商品标题 → 商品 SID", "title2sid"),
        ("历史商品标题 → 下一商品 SID", "title_history_to_sid"),
    )
    section("llm", "Qwen 架构、SFT 与 GRPO", f"""
<p>生成骨干为 Qwen2.5-0.5B decoder-only Transformer。实际配置含 {qwen['num_hidden_layers']} 层、隐藏维度 {qwen['hidden_size']}、{qwen['num_attention_heads']} 个 query 头和 {qwen['num_key_value_heads']} 个 KV 头，前馈中间维度 {qwen['intermediate_size']:,}。SFT 将实际 SID token 加入词表，扩展 {sft['model']['added_sid_tokens']} 个，词表增至 {qwen['vocab_size']:,}。输入使用历史 SID 或自然语言标题模板，输出按任务类型生成 SID 或标题。</p>
<h3>多任务监督微调</h3>
<p>把每种任务构造为 prompt 与答案。因果语言模型在答案 token 上最小化条件负对数似然。当前父模型采用线性学习率，初值 {s['learning_rate']:.0e}，warmup {s['warmup_steps']} 步，最多 {s['max_epochs']} 轮，global batch {s['global_batch_size']}。学习率随训练步数衰减，并在验证时按 <code>eval_loss</code> 选模型。传给 RL 的是 step {rl['parent_sft']['selected_step']:,}（epoch {rl['parent_sft']['selected_epoch']:.3f}）对应的导出权重。</p>
""" + table(("SFT 任务", "样本数"), [(name, f"{counts[key]:,}") for name, key in tasks]) + f"""
<h3>推荐奖励与组内策略梯度</h3>
<p>RL 从该 SFT 权重起步，在每个训练样本上执行受约束 beam search，返回 {r['num_generations']} 个生成候选。完整、规范化后的 SID 等于该行目标 SID 时，命中规则给 +1。同组至少有一个命中时，错误候选按其生成名次给归一化负折扣。折扣幅度与 1/log₂(j+1) 成正比。全组未命中时，两个奖励项均为零。RL 提示包含 SID 历史推荐、标题转 SID、描述转 SID、标题历史转 SID，共 {rl['data']['train_samples']:,} 条训练提示。</p>
<p>组内用 <em>Aⱼ=(Rⱼ−mean(R))/(std(R)+10⁻⁴)</em> 计算优势。对输出 token 的当前策略对数概率 <em>ℓ</em>，KL 估计为 <em>exp(d)−d−1</em>，其中 <em>d=log p<sub>ref</sub>−ℓ</em>。训练对有效生成 token 计算 <em>−exp(ℓ−stopgrad(ℓ))·Aⱼ+β<sub>KL</sub>·KL</em>，先按生成长度归一化，再在 batch 中求平均。参考策略起始于同一个 SFT，按配置周期同步。此处训练 KL 系数为 {r['beta']:g}，与推理前缀宽度是不同参数。</p>
<p>RL 使用最高 {r['max_epochs']} 轮预算、初始学习率 {r['learning_rate']:.0e}、cosine 衰减、{r['warmup_ratio']:.0%} warmup，按验证 NDCG@10 选用 <code>checkpoint-828</code>。训练用完整 SID 判定奖励，所以共享 SID 的不同商品不产生不同奖励。</p>""")

    section("retrieval", "UniSRec 的模型与训练目标", f"""
<p>UniSRec 使用冻结的 {spec['plm_size']:,} 维商品文本特征。商品与历史共享一个带 {spec['n_exps']} 个专家的 MoE 适配器，输出 {spec['hidden_size']} 维。适配器按输入计算专家门控权重并汇总各专家投影。历史分支叠加位置表示后通过 {spec['n_layers']} 层因果自注意力编码器（每层 {spec['n_heads']} 头，前馈维度 {spec['inner_size']}），读出最后一个有效历史位置。商品分支直接归一化适配结果，历史向量在序列编码后归一化。</p>
<p>训练采用商品级交叉熵。设历史向量为 <em>q(H)</em>、商品向量为 <em>eᵢ</em>，logits 为 <em>q(H)·eᵢ/{spec['temperature']:g}</em>。真实下一商品对应正类，分母覆盖训练商品目录。推理时用不含温度的内积对全库排序，包括具备文本向量的冷商品。离线导出第 {uni['checkpoint_epoch_zero_based'] + 1} 轮的历史编码器与 {uni['item_count']:,} × {spec['hidden_size']} 全库向量，常驻推理服务。</p>
<p>首轮使用全库。继续草拟时只在当前允许 SID 前缀包含的商品中搜索，并剔除之前草拟过的商品 ID。同分按商品 ID 升序处理。同一个用户的查询向量在所有草拟轮间复用。</p>""")

    section("inference", "核心推理算法与 2K 数据流", """
<p>对从早到晚的商品历史 <em>H</em>、输出数量 <em>K</em>、融合比例 <em>α</em>、阈值 <em>τ</em> 和前缀宽度 <em>B</em>，每轮草拟上限是 <em>2K</em>。检索目标席位 <em>q=⌊αK⌋</em>，初始 beam 席位为 <em>K−q</em>。以商品 ID 作为去重单位。配置为全接受时，合法候选均通过验证。</p>
<h3>从候选 SID 提取评分</h3>
<p>用项目历史 SID prompt 做一次 prefill，缓存历史 KV。历史末位的 logits 预测候选第一位 <em>c₁</em>。在各独立候选分支输入已知的 <em>c₁,c₂</em>，一次因果前向得到预测 <em>c₂,c₃</em> 的 logits。验证器按原始全词表计算 log-softmax，在对应位置取真实候选 token 的分数。padding 通过掩码隔离。评分不计入历史、标记、换行与 EOS。</p>
<p><em>v(i,H) = [log p(c₁ | H) + log p(c₂ | H,c₁) + log p(c₃ | H,c₁,c₂)] / 3</em>。</p>
<p>不同候选可并行前向，共享已算过的同历史前缀概率。当前实现按块复制对应历史 KV 行并保持候选后续状态独立。候选共用完整 SID 时，只计算一次并把分数回填到各商品 ID。</p>
<h3>候选校验、分支循环与补位</h3>
""" + pre("""
q = floor(alpha * K)
if q == 0: return native_beam50(H).to_items().first(K)

initialize prompt KV, UniSRec query, drafted IDs, accepted items
for depth in 0..2:
    prefix = unrestricted if depth == 0 else best_legal_prefixes(H, depth, B)
    candidates = UniSRec.top(2*K, prefix, exclude=drafted IDs)
    validate catalog item IDs, canonical SID, and exact prefix membership
    mark the candidates as drafted
    scores = teacher_force(prompt KV, candidates)
    record items with score > tau, including scores and acceptance rounds
    if accepted count >= q or all catalog items drafted: break

retained = highest scored q accepted items, preserving order on ties
if len(retained) < K:
    fill from native_beam50 in original rank order, skipping selected IDs
if retained is nonempty:
    use the same verifier to score any selected beam items with missing scores
    stable-sort only the selected items by that score
return all selected items (up to K), with alpha_eff = len(retained) / K
""") + """
<p>第一轮从全库取候选。之后使用长度分别为 1、2 的合法前缀，逐位精确匹配 SID 开头。某一轮前缀桶没有候选时，仍可尝试下一层更长前缀。已接受商品未达到检索配额时，空缺自动转给 beam。纯 beam 路径按原生成排名输出。多个商品共享 beam 的同一个 SID 时，按训练目标频次降序、商品 ID 升序选未入选商品。</p>
<p>混合重排发生在两路席位已经确定之后，稳定排序保留同分商品的原选择顺序。α=1 且首轮达到 K 时直接输出检索排序。完整 beam 也无法填满剩余席位时，返回实际取得的商品数。</p>""")

    section("deployment", "最终配置、真实示例与复现入口", """
<p>以下是每个 K 在 2K 组内按验证集选出的配置。共有参数为 α=1，前缀宽 1，最多 3 轮，完整 beam 宽 50，RL β=0.001。验证先看商品 NDCG，再看商品 HR，再按平均验证候选数更少优先，最终用固定枚举顺序解平。</p>
""" + table(("K", "每轮草拟上限", "阈值", "检索目标席位", "配置 ID"), [
        (k, 2*k, f"{conf['threshold']:.15g}" if conf["threshold"] is not None else "全接受（all）", conf["quota"], conf["id"])
        for k, conf in final.items()
    ]) + f"""
<h3>K=10 的真实请求</h3>
<p>测试集零基第 {row_id} 行的真实商品 ID 为 {target}。UniSRec 取前 20 件时它排第 {draft_rank}。RL 一次验证这 20 件，在其中选分数最高的 10 件，该商品进入最终第 {result_rank} 名。最终 10 件商品均来自首轮检索。</p>
""" + table(("最终名次", "商品 ID", "完整 SID", "验证分数"), [
        (index, entry["item_id"], entry["sid"], f"{entry['score']:.5f}") for index, entry in enumerate(ranked, 1)
    ]) + """
<h3>单条推理命令</h3>
<p>在已配置 RL 检查点、UniSRec 导出、SID 目录及数据的推理容器内，从 <code>/work</code> 用显式参数复现实例配置。离线质量评估使用每批 4 条历史，单条 BF16 推理产生的细微分差可能改变名次。</p>
""" + pre(f"""
python -m hybrid.infer --test-row {row_id} --k 10 --drafter unisrec \\
  --alpha 1 --rho 2 --prefix-beams 1 --threshold all --max-rounds 3 \\
  --assets /assets --device cuda:0
""") + f"""
<p>对需复核的产物，输入目录和原 SID 哈希在 <code>{HYBRID}/manifest.json</code>，验证选参在 <code>report/selection.json</code>，运行接口在 <code>hybrid/infer.py</code>，核心路径在 <code>hybrid/experiment.py</code> 与 <code>hybrid/verifier.py</code>。原 SID 的 SHA-256 为 <code>{sid_record['sha256']}</code>。</p>""")

    section("references", "算法参考和实现来源", """
<ul>
<li>商品离散编码与残差量化。<a href="https://arxiv.org/abs/2305.05065">TIGER</a></li>
<li>语义 ID、推荐任务对齐与策略优化。<a href="https://arxiv.org/abs/2510.24431">MiniOneRec</a></li>
<li>序列表示学习、商品特征适配与检索。<a href="https://arxiv.org/abs/2206.05941">UniSRec</a></li>
<li>检索草拟、验证和引导重草拟。<a href="https://arxiv.org/html/2410.02939v2#S3.SS3">SpecGR 方法</a></li>
<li>工业品原始数据和预置 SID 说明。<a href="https://github.com/Enddonkey/MiniOneRec-Reproduction">MiniOneRec-Reproduction</a></li>
</ul>
<p>本方案的 2K 上限、两路配额、阈值、指标和速度以本地实验参数与归档记录为准。</p>""")

    return "\n".join(parts), contents
