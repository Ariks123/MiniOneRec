# Amazon Reviews 2018 实验归档

本分支归档 Video Games 与 Industrial and Scientific 的训练代码、实验结果和报告。快照整理于 2026-11-18，原始工作区保留不变。

| 内容 | 入口 |
| --- | --- |
| 最新 Industrial RQ-VAE、GRPO 与 UniSRec 报告 | [可浏览报告](reports/industrial-latest/index.html)、[完整报告包](reports/bundles/) |
| 独立 SASRec 与联合投机推理对照 | [含 SASRec 的完整报告包](reports/bundles/Industrial-2018-RQVAE-Speculative-Recommendation-SASRec-Report.zip) |
| Industrial cosine / linear / 标题历史 / RL / UniSRec | [代码与说明](projects/industrial-cosine/README.md)、[实验报告](projects/industrial-cosine/output/reports/) |
| RQ-VAE 与混合推理、SASRec | [代码](projects/industrial-hybrid/)、[混合推理说明](projects/industrial-hybrid/hybrid/README.md) |
| Qwen 联合生成与检索 | [代码与说明](projects/qwen-retrieval/README.md)、[完整结果与补充审计](projects/qwen-retrieval/output/reports/) |
| Video Games | [项目代码](projects/video-games/README.md)、[完整报告](reports/original-2018/Video_Games/index.html) |
| 历史实现与原论文复现对比 | [早期本地实现](projects/legacy-2018/)、[原复现代码与结果](projects/reference-reproduction/) |
| 正式运行的训练与评估记录 | [原始结果](results/) |
| 权重清单与上传程序 | [发布说明](publication/README.md)、[权重来源](weight_sources.json) |

GitHub 上的 HTML 可下载后在浏览器打开，或使用 `python -m http.server 8000` 本地浏览。PDF、CSV、JSON、图表和 ZIP 同时保留，便于核查。

Industrial 最新报告采用 36,259 / 4,532 / 4,533 条训练 / 验证 / 测试样本，3,686 个商品。RQ-VAE 的三位唯一 SID、旧 RQ-KMeans SID、Qwen 联合检索和 SASRec 属于不同实验，请按各自配置、模型选择规则和报告解读。Qwen 联合实验包含监督器隔离状态失败的原记录以及后续完成结果、补充审计，均完整保留。

模型权重按[发布说明](publication/README.md)单独上传至 [Hugging Face 私有模型仓库](https://huggingface.co/Ryze5426/Amazon2018-MiniOneRec-Checkpoints)，上传完成以程序生成的远端校验回执为准；代码归档和本地权重清单本身不代表上传已完成。本分支不包含模型二进制文件、优化器状态或通用下载基座。具体模型、tokenizer、SID 和索引对应关系记录在清单中。训练脚本中的原始绝对路径反映归档时的环境，迁移运行时需要调整路径与依赖。

源码快照分别来自 `Ariks123/MiniOneRec` 的本地 Video Games、Industrial cosine、Qwen retrieval 分支及 Industrial hybrid 独立仓库；参考复现来自 `Enddonkey/MiniOneRec-Reproduction`。完整提交和工作区来源见 [provenance.json](provenance.json)，文件哈希见 [archive_manifest.json](archive_manifest.json)。各项目保留原 LICENSE。
