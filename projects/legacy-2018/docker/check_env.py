"""Check real SFT/RL imports without requiring a GPU or loading model weights."""
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.launch_h20 import PINNED


def main():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    for name, expected in PINNED.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}=={expected} required; found {versions[name]}")

    import torch
    if torch.version.cuda != "12.4":
        raise RuntimeError(f"CUDA 12.4 PyTorch required; found {torch.version.cuda}")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("This PyTorch build is missing NCCL")
    # Exercise the actual project import graph, including TRL private helpers.
    import sft  # noqa: F401
    import rl  # noqa: F401

    print(json.dumps({"python": sys.version.split()[0], "versions": versions,
                      "cuda_build": torch.version.cuda, "nccl": True,
                      "sft_import": "passed", "rl_import": "passed",
                      "gpu_training_tested": False}, indent=2))


if __name__ == "__main__":
    main()
