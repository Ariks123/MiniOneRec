"""Download the exact Qwen2.5-1.5B Base repository into the mounted weights directory."""
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    target = Path(os.environ["BASE_MODEL"])
    snapshot_download(
        repo_id="Qwen/Qwen2.5-1.5B",
        revision=os.environ.get("MODEL_REVISION", "main"),
        local_dir=str(target),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("hidden_size") != 1536:
        raise RuntimeError("Downloaded checkpoint is not the expected Qwen2.5-1.5B CausalLM")
    print(f"Qwen/Qwen2.5-1.5B Base ready: {target}")


if __name__ == "__main__":
    main()
