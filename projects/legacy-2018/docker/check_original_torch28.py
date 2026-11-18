"""Audit the isolated Torch 2.8 image and optionally exercise one CUDA device."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


AUDIT = Path("/opt/original-torch28")


def packages():
    return {dist.metadata["Name"].lower().replace("_", "-"): dist.version
            for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}


def allowed(name):
    return name in {"torch", "triton", "sympy"} or name.startswith("nvidia-")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    current = packages()
    if args.snapshot:
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "packages-before.json").write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        (AUDIT / "preserved-constraints.txt").write_text("".join(
            f"{name}=={version}\n" for name, version in sorted(current.items()) if not allowed(name)))
        print("Preserved non-Torch dependency constraints written", flush=True)
        return
    before = json.loads((AUDIT / "packages-before.json").read_text())
    changes = {name: {"before": before.get(name), "after": current.get(name)}
               for name in sorted(set(before) | set(current)) if before.get(name) != current.get(name)}
    forbidden = {name: change for name, change in changes.items() if not allowed(name)}
    if forbidden:
        raise RuntimeError("An unrelated dependency changed: " + json.dumps(forbidden))
    for name, version in {"transformers": "4.57.1", "datasets": "4.2.0", "accelerate": "1.10.1",
                          "deepspeed": "0.18.0", "trl": "0.24.0"}.items():
        if current.get(name) != version:
            raise RuntimeError(f"Preserved {name} version differs: {current.get(name)}")
    import torch
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("Unexpected Torch/CUDA build")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is unavailable")
    sys.path.insert(0, "/opt/minionerec")
    import sft
    import rl
    import transformers
    import datasets
    import deepspeed
    import trl
    import bitsandbytes

    report = {"status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "versions": {name: current[name] for name in ("torch", "triton", "sympy", "transformers", "datasets",
                                                            "accelerate", "deepspeed", "trl", "bitsandbytes")},
              "package_changes": changes, "non_torch_dependencies_preserved": True,
              "sft_import": "passed", "rl_import": "passed", "nccl_available": True,
              "python_reference_difference": "Image retains Python 3.11; archived reproduction used Python 3.12.3.",
              "cuda_toolkit_note": "nvcc is inherited from the CUDA 12.4 devel base; the Torch wheel supplies CUDA 12.8 runtime dependencies.",
              "gpu_checked": False}
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        loss = (a @ b).float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(t.grad).all() for t in (a, b)):
            raise RuntimeError("BF16 CUDA matmul/backward produced nonfinite values")
        q, k, v = [torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(3)]
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attention_loss = output.float().square().mean()
        attention_loss.backward()
        if not torch.isfinite(output).all() or not all(torch.isfinite(t.grad).all() for t in (q, k, v)):
            raise RuntimeError("BF16 SDPA/backward produced nonfinite values")
        torch.cuda.synchronize()
        report.update(gpu_checked=True, gpu_name=torch.cuda.get_device_name(0),
                      bf16_matmul_backward="passed", bf16_sdpa_backward="passed",
                      matmul_loss=float(loss), sdpa_loss=float(attention_loss))
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    name = "gpu-audit.json" if args.gpu else "image-audit.json"
    (AUDIT / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
