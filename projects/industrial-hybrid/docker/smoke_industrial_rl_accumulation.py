"""Prove four-GPU ZeRO-2 accumulation with the production compatibility hook.

Each isolated worker process uses two or sixteen microbatches, different rank-local
gradients, and an analytically checkable BF16 linear model. No dataset or
production checkpoint is loaded. Run in the pinned four-H20 preflight runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASES = {"first_nonzero_last_zero": (0.0, 0.0), "addition": (2.0, 0.0),
         "cancellation": (-1.0, 0.0), "clipping": (2.0, 0.3), "sparse_gas16": (0.0, 0.0)}


def worker(case, result_dir):
    sys.path.insert(0, str(ROOT))
    import torch
    from accelerate import Accelerator, DeepSpeedPlugin
    from trl.models import unwrap_model_for_generation
    from scripts.deepspeed_zero2_compat import apply_zero2_accumulation_fix

    metadata = apply_zero2_accumulation_fix()
    second, clipping = CASES[case]
    accumulation_steps = 16 if case == "sparse_gas16" else 2
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    if world != 4:
        raise ValueError("This accumulation smoke requires exactly four GPU ranks")
    config = {
        "bf16": {"enabled": True}, "fp16": {"enabled": False},
        "zero_optimization": {"stage": 2, "overlap_comm": True,
            "contiguous_gradients": True, "reduce_scatter": True,
            "reduce_bucket_size": 2048, "allgather_bucket_size": 2048},
        "gradient_accumulation_steps": accumulation_steps, "gradient_clipping": clipping,
        "train_batch_size": accumulation_steps * world, "train_micro_batch_size_per_gpu": 1,
        "zero_allow_untested_optimizer": True,
    }
    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
    model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    model.is_gradient_checkpointing = False
    torch.nn.init.constant_(model.weight, 1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    records = []
    for micro in range(accumulation_steps):
        accelerator.gradient_state._set_sync_gradients(micro == accumulation_steps - 1)
        with unwrap_model_for_generation(model, accelerator) as unwrapped:
            with torch.no_grad():
                unwrapped(torch.ones((1, 4), device=accelerator.device, dtype=torch.bfloat16))
        prediction = model(torch.ones((1, 4), device=accelerator.device, dtype=torch.bfloat16))
        loss = prediction.sum() * (1.0 if micro == 0 else second) * (rank + 1)
        accelerator.backward(loss / accumulation_steps, scale_wrt_gas=False)
        norm = model.get_global_grad_norm()
        records.append({"micro": micro, "boundary": model.is_gradient_accumulation_boundary(),
                        "global_grad_norm": None if norm is None else float(norm),
                        "weight": float(accelerator.unwrap_model(model).weight.detach().float().mean())})
    gradient = (1 + second * (accumulation_steps - 1)) * ((world + 1) / 2) / accumulation_steps
    expected_norm = abs(gradient) * 4
    if clipping and expected_norm > clipping:
        gradient *= clipping / expected_norm
    expected_weight = 1.0 - .1 * gradient
    weight = float(accelerator.unwrap_model(model).weight.detach().float().mean())
    passed = (abs(weight - expected_weight) < .004 and
              abs(records[-1]["global_grad_norm"] - expected_norm) < .01)
    report = {"case": case, "rank": rank, "world_size": world, "passed": passed,
              "gradient_accumulation_steps": accumulation_steps,
              "runtime_compatibility": metadata, "microbatches": records,
              "expected_norm": expected_norm, "actual_norm": records[-1]["global_grad_norm"],
              "expected_weight": expected_weight, "actual_weight": weight,
              "weight_tolerance": .004, "norm_tolerance": .01}
    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{case}.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.distributed.destroy_process_group()
    if not passed:
        raise AssertionError(json.dumps(report))


def run(output):
    cases = {}
    with tempfile.TemporaryDirectory(prefix="industrial-rl-accumulation-smoke-") as directory:
        for case in CASES:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                       str(Path(__file__).resolve()), "--worker", "--case", case, "--result-dir", directory]
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(case + " failed:\n" + (result.stdout + result.stderr)[-12000:])
            records = [json.loads((Path(directory) / f"{case}.rank{rank}.json").read_text())
                       for rank in range(4)]
            if not all(row["passed"] for row in records):
                raise AssertionError("An accumulation smoke rank did not pass")
            if len({row["actual_norm"] for row in records}) != 1 or len({row["actual_weight"] for row in records}) != 1:
                raise AssertionError("Accumulated gradients or updated weights differ across GPU ranks")
            cases[case] = records
            print(json.dumps({"case": case, "passed": True, "norm": records[0]["actual_norm"],
                              "weight": records[0]["actual_weight"]}), flush=True)
    report = {"status": "passed", "world_size": 4, "gradient_accumulation_steps": 2,
              "gradient_accumulation_steps_tested": [2, 16],
              "cases": cases, "production_checkpoint_loaded": False}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": list(cases), "world_size": 4}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--result-dir", default="")
    args = parser.parse_args()
    if args.worker:
        worker(args.case, args.result_dir)
    else:
        run(args.output)
