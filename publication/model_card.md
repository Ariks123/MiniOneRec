---
license: apache-2.0
language:
- en
tags:
- recommendation
- amazon-reviews-2018
- rq-vae
- qwen2
- sasrec
- unisrec
base_model:
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-1.5B
---

# Amazon Reviews 2018 MiniOneRec checkpoints

Selected trained recommendation checkpoints for Video Games and Industrial and Scientific, with tokenizer, semantic item ID (SID), quantizer, retrieval index and baseline assets. This card is prepared locally before upload; files become available on Hugging Face after the upload completes. The publishing script records a receipt only after verifying the remote checkpoint and asset files.

Source code and experimental reports: [Ariks123/MiniOneRec, amazon2018-archive-20261118](https://github.com/Ariks123/MiniOneRec/tree/amazon2018-archive-20261118).

`weight_sources.json` describes every selected model and its source experiment. `weight_manifest.json` lists all files, byte sizes and SHA-256 checksums. Checkpoints under different run directories are distinct experiments. Preserve the matching catalog and tokenizer when loading a model.

The selected archive includes Qwen2.5 1.5B and 0.5B SFT variants, title-history and learning-rate-schedule comparisons, GRPO runs, the separate epoch-3 comparison snapshot, a custom joint Qwen generation/retrieval model, RQ-VAE models, UniSRec encoders and SASRec baselines. Standalone optimizer files and repeated intermediate checkpoints are excluded. The selected original Video Games RQ-VAE `.pth` checkpoints retain the training state bundled in their source format. After successful upload, a standard SFT/RL checkpoint can be loaded as follows; replace `repo_id` with the repository URL's account and name.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "Ryze5426/Amazon2018-MiniOneRec-Checkpoints"
subfolder = "checkpoints/MiniOneRec/Industrial_and_Scientific_RQVAE_2K_20261118/rl/final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder)
```

The joint generation/retrieval checkpoint requires the `QwenRetrievalForCausalLM` class in `projects/qwen-retrieval/qwen_retrieval/model.py` in the source archive. Add `projects/qwen-retrieval` to Python's module search path and import it with `from qwen_retrieval.model import QwenRetrievalForCausalLM`. It contains a trained retrieval head in the model weights. UniSRec and SASRec need their corresponding project loading code and item mappings; they are not Transformers causal language models.

Models are research artifacts evaluated on fixed Amazon Reviews 2018 splits. See the source archive for training configurations, validation-based checkpoint selection, independent test results, and the joint model's original isolation failure record plus supplemental audit. Results from different SID systems or validation rules should be compared using the documented protocols.
