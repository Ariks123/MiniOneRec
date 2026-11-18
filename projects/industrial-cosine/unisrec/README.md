# 工业分支的 UniSRec 草拟器

用于 `amazon2018industrialandscientific-cosine` 的独立适配目录。已准备数据、配置、训练入口和商品 ID→SID 检索桥接；已完成100、50、20、15、25轮上限的五组从零训练。用户已选定保留20轮组，最佳权重来自第20轮，CPU复评通过；其他组的权重、生成向量和对应停止容器已清理，配置、日志及指标仍保留。当前保留结果见 [artifacts/active_run.json](artifacts/active_run.json)。

基于 [RUCAIBox/UniSRec](https://github.com/RUCAIBox/UniSRec)，固定版本见 [UPSTREAM.json](UPSTREAM.json)，原始说明见 [README.upstream.md](README.upstream.md)，保留上游 MIT LICENSE 和原始实现。新增的 `qwen_*.py` 使用上游 MoE adaptor 和 SASRec 编码器。

## 已核对的输入

| 内容 | 工业分支实际数据 |
| --- | --- |
| 交互及原 SID | `/mnt/ossdata/ziweizhao/MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey/` |
| 已有向量 | `/mnt/ossdata/ziweizhao/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy` |
| 向量形状/存储 | 3686 × 2560，float16；加载为固定 float32 表 |
| train / valid / test | 36259 / 4532 / 4533，历史长度 1–10 |
| 训练出现的商品 / 全目录 | 3647 / 3686，39 个商品未在训练历史和目标中出现 |
| 原始 SID | 3 token，3670 个不同 SID；15 组碰撞，涉及 31 个商品 |

工业向量没有生成 manifest 或原始行 ID 文件，因此**无法确认其具体 Qwen 模型、pooling 和生成设置，也无法独立证明向量行序**。当前依据完整连续商品 ID `0..3685` 推断行序，且核对向量所在目录的配套商品元数据与本分支商品元数据完全一致。数据 manifest 明确记录 `model_identity_verified: false`、`row_order_provenance_verified: false`，不将这份工业向量宣称为已经验证的 Qwen3-Embedding-4B 产物。准备代码和前向检查可继续；正式实验及新商品编码前应补齐来源证据。

## 模型和数据处理

- `inductive_ft`，不包含可训练商品 ID embedding；现有 2560 维向量冻结。
- 同一个可训练 MoE adaptor 将历史和候选的 2560 维特征适配为 300 维；序列编码器从零初始化并可训练。不加载官方 768 维 checkpoint，不冻结编码器。
- 监督信号是原训练文件中的“历史→下一商品”，交叉熵候选默认仅包含训练历史或目标中出现过的 3647 个商品。相比官方全目录 CE，这避免让训练未见商品先作为负类参与优化；`train_catalog_only: false` 可切回全目录候选。
- 原 train/valid/test 行、用户、重复购买及划分均保留；不重新增广或切分。超过长度时仅保留最近历史。原始商品 `0` 是合法 ID，RecBole 内部 `0` 单独用作 padding。
- 验证和检索使用全部 3686 个商品，屏蔽 padding。默认保留历史商品，因为验证/测试各有 236/128 条目标已在历史中出现的样本。`exclude_seen: true` 会显式屏蔽历史，因而也会屏蔽这类重复购买目标。
- 验证集的全目录 **vector_nll（越低越好）** 用于选模型和早停，Recall/NDCG作为辅助指标；测试仅在训练结束后评估。UniSRec 指标按商品 ID 计算，原 MiniOneRec 的 SID 指标无法区分碰撞商品，二者不能直接视为同一指标。
- 加载时检查准备文件 SHA256；checkpoint 绑定特征、目录、划分、内部 ID 次序和模型结构，防止错配。

## 准备和无训练检查

宿主可用 `python3` 执行数据准备，仅依赖标准库：

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine/unisrec
python3 prepare_qwen_data.py
```

本轮已经执行该命令。输出在 `prepared/Industrial_and_Scientific/`：三份 benchmark、全目录 `.item`、特征与行 ID 文件、`train_item_ids.json`、`catalog.json`、`manifest.json`。OSS 不支持符号链接时自动复制，向量内容保持原样。脚本拒绝默认覆盖；重建需显式 `--overwrite`。

模型检查需要独立 Python 3.11 + PyTorch 环境；宿主 `/usr/bin/python3` 没有这些依赖。可使用 [Dockerfile.qwen](Dockerfile.qwen) 创建隔离环境，也可在已有独立环境安装 [requirements-qwen.txt](requirements-qwen.txt)，不要改变正在运行的 SFT 容器。

```bash
python -m pip install -r requirements-qwen.txt
python -m pytest -q tests
python qwen_preflight.py --report artifacts/preflight.json
```

`qwen_preflight.py` 强制 CPU、`eval()`、`no_grad()`，核对全量数据加载、固定向量、可训练模块、合法 ID 0、300 维输出、查询×候选矩阵与 full-sort 分数一致、CE 数值有限和参数未变。它不创建 optimizer、不反向传播、不调用 `trainer.fit`，结果保存在 [artifacts/preflight.json](artifacts/preflight.json)。

此前单卡适配的独立 CPU 容器验证：**33 项测试全部通过，无跳过**，全量工业数据前向检查通过，默认训练入口检查通过。依赖版本与目录读取核验见 [artifacts/validation.json](artifacts/validation.json)。

Docker 的复制方式也适用于当前 DSW 的远程 Docker：

```bash
docker build -f Dockerfile.qwen -t unisrec-qwen:torch26 .
docker create --name unisrec-qwen-check unisrec-qwen:torch26
docker cp prepared unisrec-qwen-check:/opt/unisrec/prepared
docker start -a unisrec-qwen-check
```

## 训练入口与保留结果

当前默认最大轮数已恢复为20，继续以验证集交叉熵 loss（代码名 `vector_nll`）最小作为每次训练中的早停和最佳模型标准，patience为10。四卡、全局batch512、峰值学习率0.002及5轮预热保留；余弦衰减终点为第20轮。五组均从相同seed42重新初始化；除轮数预算和保存路径外，其他参数、数据划分及模型实现相同。改变轮数预算也会改变余弦衰减速度。

当前保留目录为 `/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/`，[artifacts/active_run.json](artifacts/active_run.json) 已指向该组。训练正常完成（退出码0），实际20轮，最佳为第20轮。CPU加载该最佳权重得到测试NLL 7.284757006，与GPU结果的差值约为 `3.1e-8`。[保留权重](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/checkpoints/QwenUniSRec-Nov-17-2026_10-13-24.pth)、[训练摘要](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/training_summary.json) 及本组完整测试结果均保留。五组历史实验和两个Qwen基线的完整对比仍见 [对比报告](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep25_20261117_182943/comparison.html) 和 [CSV](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep25_20261117_182943/metrics_comparison.csv)。

以下均评估完整4533条测试、3686个候选，冷商品子集为147条；召回率采用同分组内均匀排序的期望（容差 `1e-6`）。每组先按验证 loss 选权重，再进行测试。

| 轮数上限 | 实际 / 最佳轮 | 验证 loss↓ | 测试 NLL↓ | 整体 R10 | 整体 R20 | 整体 R50 | 冷 Top10 | 冷 Top50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 21 / 10 | **7.205852** | **7.235976** | 10.63% | 14.23% | 20.87% | 0.68% | 7.48% |
| 50 | 15 / 4 | 7.330349 | 7.373654 | 12.50% | 16.69% | 23.38% | 5.44% | **14.97%** |
| 20 | 20 / 20 | 7.236538 | 7.284757 | **14.04%** | **18.82%** | **25.97%** | **6.12%** | 10.20% |
| 15 | 15 / 13 | 7.232750 | 7.299058 | 13.34% | 17.98% | 24.93% | 4.76% | 8.84% |
| 25 | 25 / 20 | 7.206554 | 7.261403 | 13.25% | 17.59% | 25.50% | 4.76% | 10.20% |

25轮组实际完成25轮、最佳为第20轮；相对20轮组测试NLL降低0.023354，整体R50降低0.46个百分点，冷Top50相同。历史指标中，验证 loss 最优为100轮上限组，整体召回最优为20轮组，冷Top50最优为50轮组。用户查看完整指标后已选定保留20轮组：其他四组的4份权重、8份查询/目标向量文件及9个对应停止容器（含失败启动容器）均已删除；各组配置、日志、指标、逐行诊断和源码/输入快照继续保留，20轮组全部结果保留。

首组的完整分析包含原始Qwen历史均值、最后商品基线、按用户/冷目标商品聚类的配对bootstrap及逐轮曲线，见 [结果分析报告](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_20261117_173712/analysis/report.html)。五组整体固定温度NLL均优于两个Qwen基线，但冷Top50均低于历史均值17.69%和最后商品21.09%，冷商品匹配仍需改善。保留的20轮组正样本余弦均值为0.338324、成对区分率为81.02%、最难负样本间隔为−0.277597；成对区分率不等于top1准确率。

20轮组查询与对应目标向量均为 `4533 × 300` float32，CPU复评详情见 [report.json](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/test_vector_quality/report.json)。训练过程中，后台监控每20秒将日志、质量记录和最佳检查点同步回工作区；检查点先临时保存再原子发布。保留组状态见 [run_status.json](/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/run_status.json)，完整训练日志为同目录 `train.log`，权重在 `checkpoints/`。清理后已复核保留权重、两份向量和报告的SHA256均未变化。

```bash
/usr/bin/python3 scripts/monitor_run.py status --run-dir \
  /mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242
```

默认训练配置已切换为单机四卡 DDP。四卡及较大 batch 的初始参数见 [props/qwen_industrial_4gpu.yaml](props/qwen_industrial_4gpu.yaml)：

| 参数 | 原单卡配置 | 新四卡配置 |
| --- | --- | --- |
| 全局 batch | 128 | 512 |
| 每卡 batch | 128 | 128 |
| 峰值学习率 | 0.001 | 0.002 |
| 学习率策略 | 固定 | 前5轮线性预热，随后余弦衰减 |
| 最低学习率 | — | 0.0002 |
| 最大轮数 / 验证早停耐心 | 20 / 10 | 20 / 10 |

采用 Adam，学习率从原值按 batch 比例的平方根放大：`0.001 × sqrt(512/128) = 0.002`。这是经验初始设置，并非已调优结果。预热按轮设置，第1轮0.0004，第5轮0.002，随后衰减至第20轮0.0002；若早停，训练会在到达最终学习率之前结束。梯度累积为1，梯度裁剪仍为1.0。

四卡训练使用 RecBole 的训练分片和梯度同步。36259条训练交互按4进程补齐到36260条，因此每轮有1个采样重复、71次优化更新；原始数据文件不改变。验证4532条和测试4533条由rank0完整评估，其他进程接收同一指标，统一早停；不使用会补齐测试样本的分布式评估。仅rank0保存检查点和结果，各进程使用同一检查点路径。

仅检查四进程通信、数据分片和无梯度前向：

```bash
bash scripts/run_4gpu.sh preflight --report artifacts/ddp_preflight_gpu.json
```

该脚本不传模式时也默认为 `preflight`。实际训练必须显式选择 `train`：

```bash
bash scripts/run_4gpu.sh train
# 等价命令：
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  qwen_train.py --config props/qwen_industrial_4gpu.yaml --start-training
```

未分配GPU的隔离环境可检查CPU四进程通信与前向：

```bash
QWEN_DDP_BACKEND=gloo bash scripts/run_4gpu.sh preflight \
  --report artifacts/ddp_preflight_cpu.json
```

CPU通信检查本身不验证NCCL、四卡反向同步、GPU显存或加速效果。此前代码准备阶段未启动训练；当前运行已通过实际NCCL前向检查并完成四卡训练批次和完整验证。每个进程都要计算3647个训练候选的MoE，提速结论仍需同条件单卡对照。

此前四卡适配阶段：**56项测试通过，无跳过**；四进程CPU/Gloo检查通过，包含训练分片、完整评估数据、参数/映射一致性及实际主进程结果/异常广播。四卡配置也已通过单进程CPU全量加载与检索分数一致性检查。记录见 [artifacts/validation_4gpu.json](artifacts/validation_4gpu.json)、[artifacts/ddp_preflight_cpu.json](artifacts/ddp_preflight_cpu.json)；未进行GPU训练或反向测试。

四卡检查点单独写入 `/mnt/ossdata/ziweizhao/weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3/`。单进程直接启动默认四卡训练会报错提示使用torchrun；以下显式单卡入口仍保留：

```bash
python qwen_train.py --config props/qwen_industrial.yaml --start-training
```

不传 `--start-training` 时只打印说明，不加载数据或训练。原单卡配置 [props/qwen_industrial.yaml](props/qwen_industrial.yaml) 的设置与检查点目录保留。若在远程Docker中训练，需要分配四张GPU并将权重目录挂载或复制回工作区；当前Dockerfile默认命令只做单进程检查。

## 向量匹配质量是主要评估目标

这里的“向量准确度”指历史查询向量对真实下一商品的匹配质量。查询 `q` 是历史经共享 adaptor 和编码器后的归一化300维向量；目标 `v+` 是真实商品的固定内容向量经训练后 adaptor 得到的归一化300维向量。没有独立给定的“正确300维坐标”，不能直接将输出与原始2560维Qwen向量逐坐标计算误差。

| 评估项 | 方向及含义 |
| --- | --- |
| `vector_nll` | 主指标，越低越好；固定温度0.07和完整3686商品目录下，真实下一商品的负对数概率 |
| `positive_cosine` | 查询与真实目标的余弦；记录均值及分位数，需结合其他指标解释 |
| `margin_vs_hardest_negative` | 目标相似度减去最高分的其他候选相似度；越大越好，正值表示目标严格胜出 |
| `margin_vs_mean_negative` | 目标与其他候选平均相似度的差距 |
| `pairwise_accuracy` | 目标胜过其他候选的比例；平局计0.5，不能替代难候选差距 |
| `alignment_squared_l2` | 同一300维空间的查询—目标距离平方，等于 `2−2cos(q,v+)` |
| 几何诊断 | 查询及商品向量的坐标方差、异向量平均余弦、协方差有效秩，检查向量是否趋同 |
| 冷商品分组 | 训练历史和目标均未出现过的商品；与训练已见商品分开报告上述指标 |

全目录均匀预测的NLL基线为 `log(3686)≈8.212297`，报告同时给出相对均匀基线的提升。只比较同一目录、温度和划分下的NLL；它不是百分比准确率。其他候选只是未观察的对照，不代表用户明确不喜欢。

如果所有向量都趋同，即使正样本余弦达到1，也不能据此认为准确：此时分数间隔为0、成对准确率为0.5、NLL等于均匀基线。排序辅助指标使用同分组内均匀排列的期望，避免同分或商品向量重复时虚高。原SID碰撞不合并商品，评估仍按商品ID进行。

训练时每次验证会记录向量质量摘要及几何诊断，写入TensorBoard和检查点旁的 `*.vector_quality.jsonl`。两份配置均已改为 `valid_metric: vector_nll`、越低越好；RecBole对自定义指标方向的默认处理也已覆盖。训练损失仍为原来的下一商品交叉熵，Recall/NDCG继续作为辅助指标。

训练后可单独导出完整验证集或测试集的逐行诊断：

```bash
python qwen_vector_eval.py --config props/qwen_industrial_4gpu.yaml \
  --checkpoint /path/to/trained.pth --split valid \
  --output /path/to/new_vector_quality --save-vectors
```

输出 `report.json`（总体、冷暖分组及几何摘要）、`rows.jsonl`（原商品ID/SID、目标冷热标记、逐行指标），可选 `query_vectors.npy` 和 `target_item_vectors.npy`。使用 `--split test` 可在模型选定后进行测试。该工具必须提供数据指纹匹配的训练checkpoint，拒绝不存在的checkpoint和已有输出目录。

前向检查仅验证实现正确，不是训练后向量质量结论。准备阶段只验证评估公式、边界情况、checkpoint限制和选模型逻辑；当前实际训练的验证向量质量记录，保存在上述运行目录的 `checkpoints/*.vector_quality.jsonl` 及状态文件中。训练早期值不等同于最终测试结果。

加入向量质量评估后，完整 **89项测试通过，无跳过**。验证集冷目标有125行（28个商品ID），测试集有147行（37个商品ID）；这些是数据分组计数，不是模型效果。测试与评估准备状态见 [artifacts/validation_vector_quality.json](artifacts/validation_vector_quality.json)。

## 训练后的检索与 SID 交接（本轮未执行）

```bash
python qwen_retrieval.py export --config props/qwen_industrial_4gpu.yaml \
  --checkpoint /path/to/trained.pth --output /path/to/new_index
python qwen_retrieval.py draft --config props/qwen_industrial_4gpu.yaml \
  --checkpoint /path/to/trained.pth --index /path/to/new_index \
  --history '["0", "12", "23"]' --topk 100
```

导出在 `eval()` 下关闭 adaptor dropout 和 gating 噪声，将候选归一化后保存；当前目录仅 3686 个商品，内置精确内积检索，无需 FAISS。导出必须提供训练入口生成且数据指纹匹配的 checkpoint，不会把预检随机权重当作训练结果。

四卡训练的检查点可由单进程CPU/GPU加载导出，模型权重不带DDP包装前缀，导出不需要四张GPU。显式传入训练时对应的配置即可。

`MiniOneRecDrafter.verification_payload()` 返回 `history_item_ids`、逐商品 `candidates` 和 `candidate_sids`，可供 MiniOneRec 现有 SID tokenizer/候选评分器消费。原 SID 碰撞和缺失 ASIN 保留，多个原商品 ID 可以对应同一 SID。该桥接提供候选验证接口数据，尚未调用真实 MiniOneRec 验证模型。

查询与候选只需在 UniSRec 的 300 维空间中匹配，通过商品 ID→原 SID 交接，无需对齐 MiniOneRec 隐藏状态，也不重训 RQ-VAE、SID 或 Qwen。`encode_features()` 可转换任意同来源新商品向量；真正新增目录还需要维护 ID/SID 映射及更新索引，本轮未导入新商品。由于工业向量来源证据缺失，在补齐来源前不能保证新编码向量与现有特征一致。
