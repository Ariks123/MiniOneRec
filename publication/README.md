# 权重发布

[weight_sources.json](../weight_sources.json) 列出计划发布的正式训练运行推理权重、tokenizer、SID 和检索资产。清单排除重复训练步骤检查点、独立优化器状态、失败和 smoke 运行。Video Games 的 RQ-VAE 比较保留完整原始 `.pth`，其中包含恢复训练所需的状态。独立比较所需的第 3 轮原始权重和 RQ-VAE 参考方案保留。[模型卡](model_card.md)是上传时使用的文档；仅生成本地清单或模型卡不表示权重已上传。

从原工作区上传（凭据不写入本仓库）：

```bash
source /mnt/ossdata/ziweizhao/env.sh
python -m pip install huggingface_hub
hf auth login
python publication/upload_weights.py --repo-id Ryze5426/Amazon2018-MiniOneRec-Checkpoints --private
```

程序先核对全部源文件，创建轻量软链接 staging，再创建 Hugging Face model 仓库并上传；重新执行可续传。默认 staging 位于 `/tmp/amazon2018-models-ziweizhao`，其中只有指向工作区权重的链接与清单文件，模型文件仍留在原工作区。可用 `--prepare-only` 仅生成本地清单。不会移动或修改原权重。

上传完成后，普通 SFT/RL 检查点可通过 `AutoModelForCausalLM.from_pretrained(repo_id, subfolder="checkpoints/MiniOneRec/.../final_checkpoint")` 加载。Qwen 联合模型需要 [qwen_retrieval/model.py](../projects/qwen-retrieval/qwen_retrieval/model.py) 中的 `QwenRetrievalForCausalLM`，在 `projects/qwen-retrieval` 加入 Python 模块搜索路径后可用 `from qwen_retrieval.model import QwenRetrievalForCausalLM` 导入；UniSRec 和 SASRec 请使用各自项目加载代码。RQ-VAE 模型与 catalog SID 必须匹配。

上传程序将验证远端清单内每个权重和资产文件的大小、LFS SHA-256 或普通 Git blob SHA-1，并把成功回执写入代码归档根目录下的 `.runtime/huggingface_receipt.json`。该回执包含模型仓库 URL、提交版本和已验证文件数量。模型卡、权重来源及文件哈希清单同时上传。
