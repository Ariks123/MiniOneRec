# Amazon Reviews 2018 · Industrial_and_Scientific

当前采用 **Qwen2.5-0.5B Base + 标题历史任务 + linear**，顺序比较epoch上限3轮与5轮。两组均从相同Base权重重新开始，训练期间仅计算验证loss并据此早停、保存最佳模型，不计算HR/NDCG；每组训练结束后才做完整4533条测试样本的beam50排名评估。

5轮组额外保存**第3轮结束时、重载最佳模型之前**的实际训练权重，目录为 `epoch_3_checkpoint/`，含tokenizer、SID来源和 `snapshot_metadata.json`。该组至少训练到第3轮以保证比较权重存在，此后继续按验证loss早停，最多5轮。其余批量、模型、五类任务、ZeRO-2和梯度检查点设置保持上一轮配置。

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine
python docker/industrial_linear_epoch_experiment.py start
python docker/industrial_linear_epoch_experiment.py status
```

运行目录为 `/mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1/`，包含 `epochs3/sft/` 和 `epochs5/sft/`，以及持久化计划与状态文件。报告位于 `output/reports/industrial_05b_linear_ep3_ep5_title_history_retry1/`。

最终 `comparison/` 中的JSON、CSV和HTML汇总三份模型：3轮组最佳loss、5轮组最佳loss、5轮组第3轮原始权重。后者的独立测试记录位于 `epochs5/sft/evaluation_epoch3/`。验证与测试不进入训练任务；测试不用于选择模型。linear按照各自3/5轮总预算衰减，因此两组第3轮的学习率轨迹并不相同。

**此前实验记录**

此前的5epoch实验采用 **新增标题历史任务、Qwen2.5-0.5B Base、cosine、上限5epoch**。cosine在5epoch预算内衰减，比此前10epoch更快；其余训练设置保持一致。

每0.5epoch在完整4532条验证样本上，用原始评估提示、完整目录、beam50计算Recall/NDCG。以验证NDCG@10选择主模型并早停（patience=3）；独立保留验证Recall@10最好的模型。测试4533条样本只在训练结束后评估两个选定模型，不用于早停或选模型。

- `best_ndcg_checkpoint/`：验证NDCG@10最高的模型，`final_checkpoint/`导出同一模型。
- `best_recall_checkpoint/`：验证Recall@10最高的模型；同分时依次比较NDCG@10、验证loss。
- `ranking_selection.json`：两个最佳模型的步骤、epoch和验证指标。
- `validation_ranking/`：每次完整验证的指标与逐行预测。
- `evaluation/`、`evaluation_recall/`：两个选定模型的最终测试结果。

本轮运行与报告保存在独立目录，不覆盖此前结果：

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine
python docker/industrial_experiment.py start \
  --run-dir /mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_cosine_ep5_title_history_rank50_20261117/sft \
  --report-root /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine/output/reports/industrial_05b_cosine_ep5_title_history_rank50 \
  --container minionerec-industrial05b-title-ep5-rank50-20261117 \
  --include-title-history --num-epochs 5 --eval-save-fraction 0.1 \
  --generative-validation --generation-num-beams 50 --generation-batch-size 4
```

查看本轮状态，将上述 `start` 改为 `status`，并保留相同 `--run-dir`。报告会增加验证Recall/NDCG曲线、独立最佳检查点记录，以及两个选定模型的最终测试对照。

**此前10epoch对照记录**

此前完成的两组对照实验统一使用 **Qwen2.5-0.5B Base、cosine、最多 10 epoch**，比较增加“历史商品标题 → 下一商品 SID”任务后的召回率。此前的 1.5B cosine 实验保留为历史归档。

| 组别 | 每轮训练样本 | 任务 |
| --- | ---: | --- |
| baseline | 79,834 | SID历史→下一SID、SID历史→下一标题、SID→标题、标题→SID |
| title_history | 116,093 | 保留基线任务，增加36,259条标题历史→下一SID |

两组均从相同 Base 权重重新初始化，使用固定的同一 Docker 镜像；seed 42、学习率 `3e-4`、warmup 20 steps、BF16、4张H20、每卡batch8、梯度累积2、全局batch64、cutoff512、DeepSpeed ZeRO-2和gradient checkpointing保持一致。

早停按原SID推荐验证集loss判断，patience为3；最多10epoch时约每0.5epoch验证一次。训练结束后重载最佳验证loss模型，在相同的完整4533条测试样本、完整商品目录、原评估提示和beam50下计算Recall/NDCG及冷暖分组。测试集不参与模型选择。

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine
python docker/industrial_title_history_experiment.py status
```

完整实验由持久化监督器按顺序运行两组，再自动生成对照报告。启动新实验：

```bash
python docker/industrial_title_history_experiment.py start
```

默认运行目录为 `/mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_cosine_ep10_20261117/`，包含 `comparison_plan.json`、`comparison_status.json`、`baseline/sft/` 和 `title_history/sft/`。报告位于本工作目录的 `output/reports/industrial_05b_cosine_ep10/`，最终汇总为 `comparison/comparison.json`、`comparison.csv` 和 `index.html`。已有实验目录不会被覆盖；重跑时使用新的 `--experiment-root` 和 `--report-root`。

`docker/industrial_experiment.py` 可单独运行一组；新增组使用 `--include-title-history`，基线使用 `--no-include-title-history`。新增任务只读取训练集的历史标题，监督目标是下一件商品的完整SID，提示部分不计算loss。两组使用同样epoch上限，但新增任务会增加训练样本和优化更新次数；对照结论应按整个训练方案解释。

共享固定输入位于 `/mnt/ossdata/ziweizhao/MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey/` 和 `data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey/`；Base模型位于 `/mnt/ossdata/ziweizhao/weights/Qwen2.5-0.5B/`。大体积数据、权重和新实验产物通过共享路径引用，并由Git忽略。

历史1.5B cosine实验的3epoch结果：Recall@10/20/50为 **12.6186% / 16.5674% / 21.9501%**。该结果属于此前运行，不能视为此次0.5B新实验结果。其固定配置见 [config/selected_run.json](config/selected_run.json)，原始记录位于 `results/cosine/`，报告位于 `reports/`；历史源码快照提交为 `1a5746d`。原检查点位于 `/mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_enddonkey/sft/final_checkpoint/`，已加载最佳 `checkpoint-3572`。

本地检查：`python -m unittest discover -s tests -q`。训练前还在固定镜像中离线加载本地tokenizer，检查完整任务构造与监督标签。
