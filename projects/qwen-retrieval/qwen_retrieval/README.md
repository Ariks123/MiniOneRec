# Qwen 生成与检索联合实验

这个独立分支适用于 Amazon Reviews 2018 Industrial and Scientific。
Qwen2.5-0.5B Base 保留原五任务训练，新增共享896→256投影头；只有SID历史→下一SID样本参与对比学习。
用户历史使用原SFT prompt的末端状态，物品输入使用固定 `Item SID: {sid}\n### Embedding:` 模板。
主干与投影头均共享参数，训练时历史向量读取第一个有效答案label之前的位置，避免看到答案。

配置沿用历史最佳的linear三轮预算、学习率3e-4、warmup20、seed42、四卡每卡batch8、梯度累积2、BF16、cutoff512和ZeRO-2。
检索loss权重0.1，温度0.07；四卡共享同一microbatch内的SEQ负例，梯度累积不会扩大负例池。
相同SID只保留一个候选，避免把重复目标或SID冲突当负例。

## 首轮实际结果

运行 `20261118_004220` 已完成三轮、5442次更新，训练约41分钟，未早停。
最佳权重为第2.8594轮的 `checkpoint-5187`，已导出到该运行的 `final_checkpoint/`。
完整4532条验证的生成loss为 **1.519214**，历史最佳为1.560734；验证集向量Recall@200为35.9003%，重排Recall@10/NDCG@10为12.4228%/8.7299%。

完整4533条测试，以下均为百分数：

| 方法 | Recall@10 | NDCG@10 | Recall@50 |
| --- | ---: | ---: | ---: |
| 历史最佳SFT beam50（重新评估） | 12.9495 | 8.3001 | 22.1046 |
| 联合模型 beam50 | 11.8465 | 7.9597 | 21.6413 |
| 联合模型纯向量检索 | 11.5597 | 7.9918 | 21.2221 |
| 向量Top200→GR重排 | 12.0450 | 8.2501 | 20.4721 |

测试向量Recall@200为34.9879%。147条交互冷启动样本中，向量Top200召回48条（32.6531%），但GR重排后的Recall@50为0。
四卡完整测试推理阶段：历史beam50约80.92秒，召回与重排约147.46秒；这组结果暂不替换历史最佳SFT。

产物根目录为 `/mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_qwen_retrieval_20261118_004220/sft/`。
原始报告位于 `output/reports/qwen_retrieval_20261118_004220/`。
所有训练及评估阶段完成，但原监督器在末尾严格隔离检查处保留了failed状态：旧分支在训练期间新增了独立RL提交，HEAD由798908f变为de1658f。
补充审计记录与原始失败记录分别保存；已核对的旧SFT文件、数据和历史模型权重保持一致，不修改旧分支或原隔离快照来消除告警。

## 启动与产物

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-qwen-retrieval
python docker/industrial_retrieval_experiment.py start
python docker/industrial_retrieval_experiment.py status
```

监督器使用固定的 `minionerec:h20-original-torch28` 镜像和专属容器，依次验证、训练、建库和评估。
新运行使用独立checkpoint/report目录；不会覆盖历史结果。
准确路径、当前阶段及异常记录在监督器返回的状态文件中。
训练保存主干、投影头、tokenizer和优化器状态，按验证生成loss选择最终权重；测试集不参与训练或选模型。

物品索引按3670个唯一SID建行，并保留3686个商品ID的映射；保存模型权重、tokenizer、模板和SID版本校验。
评估先检索200个不同SID，再用Qwen对三个SID token的平均条件log probability重排。
GR评分沿用历史original-eval prompt，查询向量沿用SFT prompt；换行和EOS不参与评分。

对照报告含历史SFT beam50、新模型beam50、纯向量检索和向量召回→GR重排的Recall/NDCG、Recall@200、分组结果及计时。
原始工业SID有15组冲突，因此主指标保留SID级命中语义；商品ID用于交互冷暖划分。
全目录SID↔标题任务保持原样，冷启动结论仅指交互冷启动。

## 运行差异与复用

新训练启用经过源码SHA256检查的DeepSpeed 0.18.0梯度累积兼容修复，仅作用于当前进程，不修改已安装包。
历史SFT未记录该修复；用户选择只跑一组新增训练，因此历史结果差异不能全部归因于检索loss。
首版不包括冻结索引第二阶段、阈值接受、前缀再草拟和提前退出。

独立入口为 `python -m qwen_retrieval.train`、`python -m qwen_retrieval.encode_items`、
`python -m qwen_retrieval.evaluate`；各入口 `--help` 提供参数。
下次训练使用新的输出目录；模型更新后必须重建物品向量库。
