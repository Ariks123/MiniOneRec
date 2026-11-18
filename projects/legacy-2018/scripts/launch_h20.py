#!/usr/bin/env python3
"""Single-node four-H20 SFT/RL launcher. --dry-run never starts training."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SID_SHA256 = "ca9d730905a67cc36cd77c665a3165de823f2689660b19741df9ee884993085f"
DATASET = "Video_Games"
DEFAULT_INPUT = ROOT / "data/Amazon18/Video_Games"
DEFAULT_DATA = ROOT / "data/Amazon18/posttraining/Video_Games_sid2900"
DEFAULT_OUTPUT = Path(os.environ.get("H20_OUTPUT_ROOT", str(ROOT / "output/h20_4gpu/Video_Games_sid2900")))
DEFAULT_BASE_MODEL = Path(os.environ.get("ZIWEIZHAO_WORKDIR", str(ROOT.parent))) / "weights/Qwen2.5-1.5B"
DS_CONFIG = ROOT / "config/h20_zero2.json"
ACCELERATE_CONFIG = ROOT / "config/h20_4gpu.yaml"
PINNED = {"torch": "2.6.0", "transformers": "4.57.1", "accelerate": "1.10.1",
          "trl": "0.24.0", "deepspeed": "0.18.0"}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def path_from_root(value):
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sft", "rl"))
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", str(DEFAULT_BASE_MODEL)),
                   help="Local CausalLM directory for SFT; defaults to weights/Qwen2.5-1.5B (Base)")
    p.add_argument("--model-path", default=os.environ.get("SFT_MODEL", ""),
                   help="Local SFT final_checkpoint directory for RL")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    p.add_argument("--master-port", type=int)
    p.add_argument("--nccl-ib-disable", choices=("auto", "0", "1"), default="auto",
                   help="auto preserves administrator settings and otherwise leaves NCCL auto-selection enabled")
    p.add_argument("--sft-batch-size", type=int, default=1024)
    p.add_argument("--sft-micro-batch-size", type=int, default=16)
    p.add_argument("--monitor-window", type=int, default=100)
    p.add_argument("--resume-from-checkpoint", default="")
    p.add_argument("--rl-batch-size", type=int, default=16)
    p.add_argument("--rl-eval-batch-size", type=int, default=16)
    p.add_argument("--rl-gradient-accumulation", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print command and pending inputs; no training or GPU imports")
    modes.add_argument("--preflight-only", action="store_true", help="Check data/model/environment/four GPUs; never train")
    return p

def build_plan(args):
    devices = [x.strip() for x in args.devices.split(",")]
    require(len(devices) == 4 and all(devices) and len(set(devices)) == 4,
            "This launcher requires exactly four distinct visible GPUs")
    for name in ("sft_batch_size", "sft_micro_batch_size", "rl_batch_size",
                 "rl_eval_batch_size", "rl_gradient_accumulation", "num_generations"):
        require(getattr(args, name) > 0, name + " must be positive")
    require(args.sft_batch_size % (4 * args.sft_micro_batch_size) == 0,
            "SFT global batch must be divisible by 4 * per-device micro batch")
    require(args.num_generations >= 2, "RL num_generations must be at least 2")
    require(args.rl_batch_size % args.num_generations == 0
            and args.rl_eval_batch_size % args.num_generations == 0,
            "RL beam generation requires each device's train/eval batch to be divisible by num_generations")
    require(args.num_epochs is None or args.num_epochs > 0, "num_epochs must be positive")
    require(args.learning_rate is None or args.learning_rate > 0, "learning_rate must be positive")
    require(args.monitor_window > 0, "monitor_window must be positive")
    port = args.master_port if args.master_port is not None else (29501 if args.stage == "sft" else 29503)
    require(1 <= port <= 65535, "master_port must be between 1 and 65535")
    raw = path_from_root(args.input_dir)
    data = path_from_root(args.data_dir)
    output = path_from_root(args.output_dir or DEFAULT_OUTPUT / args.stage)
    model_arg = args.base_model if args.stage == "sft" else (args.model_path or str(DEFAULT_OUTPUT / "sft/final_checkpoint"))
    model = str(path_from_root(model_arg)) if model_arg else "<BASE_MODEL>"
    common = ["--train_file", str(data / "train.csv"), "--eval_file", str(data / "valid.csv"),
              "--output_dir", str(output), "--category", DATASET,
              "--seed", "42", "--sid_index_path", str(raw / (DATASET + ".index.json")),
              "--item_meta_path", str(raw / (DATASET + ".item.json")),
              "--gradient_checkpointing", "True", "--attn_implementation", args.attn_implementation]
    if args.stage == "sft":
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
                   "--nproc_per_node", "4", "--master_addr", "127.0.0.1",
                   "--master_port", str(port), str(ROOT / "sft.py"), "--base_model", model,
                   "--batch_size", str(args.sft_batch_size), "--micro_batch_size", str(args.sft_micro_batch_size),
                   "--num_epochs", str(args.num_epochs or 10),
                   "--learning_rate", str(args.learning_rate or 3e-4),
                   "--cutoff_len", "512", "--deepspeed", str(DS_CONFIG),
                   "--monitor_window", str(args.monitor_window),
                   "--train_from_scratch", "False", "--freeze_LLM", "False",
                   "--wandb_run_name", "Video_Games_sid2900_sft_h20x4"] + common
        if args.resume_from_checkpoint:
            command += ["--resume_from_checkpoint", str(path_from_root(args.resume_from_checkpoint))]
    else:
        command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", str(ACCELERATE_CONFIG),
                   "--num_processes", "4", "--main_process_port", str(port), str(ROOT / "rl.py"),
                   "--model_path", model, "--info_file", str(data / "info.txt"),
                   "--train_batch_size", str(args.rl_batch_size), "--eval_batch_size", str(args.rl_eval_batch_size),
                   "--gradient_accumulation_steps", str(args.rl_gradient_accumulation),
                   "--num_train_epochs", str(args.num_epochs or 2),
                   "--learning_rate", str(args.learning_rate or 1e-5), "--num_generations", str(args.num_generations),
                   "--reward_type", "ranking", "--beam_search", "True", "--sync_ref_model", "True",
                   "--sample_train", "False", "--eval_step", "0.0999", "--test_during_training", "False",
                   "--temperature", "1.0", "--beta", "0.001", "--add_gt", "False",
                   "--dynamic_sampling", "False", "--mask_all_zero", "False", "--dapo", "False",
                   "--wandb_run_name", "Video_Games_sid2900_rl_h20x4"] + common
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("WANDB_MODE", "disabled")
    if args.nccl_ib_disable != "auto":
        env["NCCL_IB_DISABLE"] = args.nccl_ib_disable
    return {"stage": args.stage, "selected_sid_epoch": 2900, "expected_sid_sha256": SID_SHA256,
            "gpu_processes": 4, "devices": devices, "precision": "bf16", "zero_stage": 2,
            "sft_gradient_accumulation": args.sft_batch_size // (4 * args.sft_micro_batch_size),
            "sft_global_samples_per_step": args.sft_batch_size,
            "rl_completions_per_step": 4 * args.rl_batch_size * args.rl_gradient_accumulation,
            "rl_prompt_groups_per_step": 4 * args.rl_batch_size * args.rl_gradient_accumulation // args.num_generations,
            "command": command, "environment": env, "input_dir": raw, "data_dir": data,
            "output_dir": output, "model_dir": None if not model_arg else Path(model)}

def check_inputs(plan):
    """Verify approved SID and converted output hashes, without loading model weights."""
    pending = []
    raw, data, model = plan["input_dir"], plan["data_dir"], plan["model_dir"]
    sid = raw / (DATASET + ".index.json")
    metadata = raw / (DATASET + ".item.json")
    for path in (sid, metadata):
        if not path.is_file():
            pending.append("Missing input: " + str(path))
    if sid.is_file():
        require(sha256(sid) == SID_SHA256, "SID hash differs from the approved epoch-2900 catalog")
    manifest_path = data / "preparation_manifest.json"
    if not manifest_path.is_file():
        pending.append("Prepare CSV/info first: python scripts/prepare_video_games_posttraining.py --input-dir "
                       + shlex.quote(str(raw)) + " --output-dir " + shlex.quote(str(data)))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("validation") == "passed" and not manifest.get("check_only")
                and manifest.get("selected_sid_epoch") == 2900, "Invalid prepared-data manifest")
        require(manifest.get("dataset") == DATASET and manifest.get("preserve_all_interaction_rows") is True
                and manifest.get("sampling_applied") is False and manifest.get("keep_longest_only") is False,
                "Prepared data must preserve all Video Games rows and the original splits")
        require(manifest["sources"]["sid_index"]["sha256"] == SID_SHA256, "Prepared CSV uses a different SID catalog")
        for source in manifest["sources"].values():
            require(Path(source["filename"]).name == source["filename"], "Manifest source filename must be local")
            require((raw / source["filename"]).is_file() and sha256(raw / source["filename"]) == source["sha256"],
                    "Prepared data source changed: " + source["filename"])
        if metadata.is_file():
            require(manifest["sources"]["item_metadata"]["sha256"] == sha256(metadata),
                    "Prepared CSV metadata differs from current metadata")
        for filename in ("train.csv", "valid.csv", "test.csv", "info.txt"):
            path = data / filename
            require(path.is_file() and sha256(path) == manifest["outputs"][filename]["sha256"],
                    "Prepared output missing or changed: " + filename)
    if model is None or not (model / "config.json").is_file():
        pending.append("Set --base-model (SFT) or --model-path (RL) to a local CausalLM checkpoint")
    else:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        require(any("ForCausalLM" in x for x in config.get("architectures", [])),
                "The model must be a CausalLM; Qwen3-Embedding-4B is an encoder")
        weights = [model / "model.safetensors", model / "pytorch_model.bin"]
        indices = [model / "model.safetensors.index.json", model / "pytorch_model.bin.index.json"]
        index_path = next((x for x in indices if x.is_file()), None)
        if index_path:
            shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
            require(all((model / x).is_file() for x in shards), "Model weight shards are missing")
        else:
            require(any(x.is_file() for x in weights), "Model weight files are missing")
        if plan["stage"] == "rl":
            provenance_path = model / "posttraining_provenance.json"
            if not provenance_path.is_file():
                pending.append("RL needs the SID-bound SFT final_checkpoint with posttraining_provenance.json")
            else:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                require(provenance.get("stage") == "sft" and provenance.get("sid_index_sha256") == SID_SHA256,
                        "RL SFT checkpoint was not trained with the approved epoch-2900 SID")
                if metadata.is_file():
                    require(provenance.get("item_meta_sha256") == sha256(metadata),
                            "RL SFT checkpoint used different item metadata")
    return pending

def check_runtime(plan):
    require(platform.system() == "Linux", "Four-H20 NCCL/DeepSpeed training requires a Linux CUDA server")
    for name, expected in PINNED.items():
        installed = importlib.metadata.version(name).split("+")[0]
        require(installed == expected, f"{name}=={expected} required; found {installed}")
    # Import only after the dry-run branch has returned. No model is loaded.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(plan["devices"])
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Exactly four visible CUDA GPUs are required")
    names = [torch.cuda.get_device_name(i) for i in range(4)]
    require(all("H20" in x.upper() for x in names), "This hardware profile expects four H20 GPUs; detected " + str(names))
    require(torch.distributed.is_nccl_available(), "This PyTorch build does not provide NCCL")
    require(torch.cuda.is_bf16_supported(), "The CUDA environment must support BF16")
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(plan["model_dir"]), local_files_only=True)
    if plan["stage"] == "rl":
        # RL must use the SID-aware SFT checkpoint, not the untouched base model.
        catalog = json.loads((plan["input_dir"] / (DATASET + ".index.json")).read_text(encoding="utf-8"))
        tokens = {token for sid in catalog.values() for token in sid}
        vocab = tokenizer.get_vocab()
        require(all(token in vocab and tokenizer.encode(token, add_special_tokens=False) == [vocab[token]]
                    for token in tokens), "RL checkpoint tokenizer is missing atomic SID tokens; use the SFT final_checkpoint")
        model_config = json.loads((plan["model_dir"] / "config.json").read_text(encoding="utf-8"))
        require(int(model_config.get("vocab_size", 0)) > max(vocab[token] for token in tokens),
                "RL model vocabulary does not include all SID tokens")
    return names

def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
        pending = check_inputs(plan)
        printable = {key: value for key, value in plan.items() if key not in
                     ("environment", "input_dir", "data_dir", "output_dir", "model_dir")}
        printable.update(mode="dry_run" if args.dry_run else ("preflight" if args.preflight_only else "launch"),
                         training_started=False, inputs_ready=not pending, pending_inputs=pending,
                         nccl={key: plan["environment"][key] for key in
                               ("NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_DEBUG")
                               if key in plan["environment"]},
                         shell_command=shlex.join(plan["command"]))
        if args.dry_run:
            print(json.dumps(printable, ensure_ascii=False, indent=2))
            return 0
        require(not pending, "\n".join(pending))
        names = check_runtime(plan)
        printable["gpu_names"] = names
        print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            return 0
        return subprocess.run(plan["command"], cwd=ROOT, env=plan["environment"], check=False).returncode
    except (ValueError, OSError, KeyError, json.JSONDecodeError, importlib.metadata.PackageNotFoundError) as error:
        p.exit(2, f"H20 preflight failed: {error}\n")

if __name__ == "__main__":
    sys.exit(main())
