"""Fingerprint the files and software actually present in the inference container."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess

from .experiment import dump

EXPECTED_WEIGHTS = "580c46ddcf508b2af8680928994f9dda4809f573b07ca28655281594d52c925b"
EXPECTED_BEAM = "9e8e53ba51269f97f0adecc71e1534d93660476909703cd599146c4434e8a20f"
EXPECTED_SID = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"


def fingerprint(path):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": sha, "bytes": path.stat().st_size}


def require_digest(found, expected, label):
    if found != expected:
        raise ValueError(f"{label} SHA256 differs: {found} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--run-dir", type=Path, default=Path("/run/hybrid"))
    args = parser.parse_args()
    assets = args.assets
    root = Path(__file__).resolve().parent.parent
    data_dir = assets / "data"
    prep_path = data_dir / "preparation_manifest.json"
    prepared = json.loads(prep_path.read_text())
    splits = {name: fingerprint(data_dir / name) for name in ("train.csv", "valid.csv", "test.csv", "info.txt")}
    for name, value in splits.items():
        require_digest(value["sha256"], prepared["outputs"][name]["sha256"], name)
    sid = fingerprint(assets / "sid.json")
    require_digest(sid["sha256"], EXPECTED_SID, "original SID catalog")
    require_digest(sid["sha256"], prepared["expected_sid_sha256"], "data preparation SID catalog")
    verifier = {p.name: fingerprint(p) for p in sorted((assets / "rl").iterdir()) if p.is_file()}
    require_digest(verifier["model.safetensors"]["sha256"], EXPECTED_WEIGHTS, "RL verifier")
    legacy = fingerprint(assets / "legacy_test.jsonl")
    replay = fingerprint(args.run_dir / "reference_test_beam.jsonl")
    for key, record in (("legacy beam", legacy), ("replay beam", replay)):
        require_digest(record["sha256"], EXPECTED_BEAM, key)

    shared = ["data.py", "npu_support.py", "LogitProcessor.py", "scripts/sft_eval_utils.py"]
    code = sorted([*(root / "hybrid").glob("*.py"), *(root / "hybrid").glob("*.sh"),
                   *(root / "tests").glob("test_hybrid*.py"),
                   *(root / p for p in shared)])
    sources = {str(p.relative_to(root)): fingerprint(p) for p in code}
    qwen = assets / "qwen_source"
    qwen_sources = {str(p.relative_to(qwen)): fingerprint(p)
                    for p in sorted((qwen / "qwen_retrieval").glob("*.py"))}
    if not qwen_sources:
        raise ValueError("Missing deployed joint Qwen retrieval source")
    packages = ("torch", "transformers", "tokenizers", "numpy", "faiss-cpu", "safetensors",
                "accelerate", "pandas", "matplotlib")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assets_root": str(assets), "executed_source_root": str(root),
        "verifier": {"rl_kl_beta": 0.001, "selected_training_step": 828, "files": verifier},
        "data": {"split_files": splits, "row_counts": prepared["rows"],
                 "preparation_manifest": fingerprint(prep_path), "original_sid_index": sid,
                 "reference_test_beam": replay, "legacy_test_beam": legacy},
        "draft_assets": {
            p: fingerprint(assets / p) for p in ("raw.npy", "raw_manifest.json", "raw_row_ids.json",
                                                 "unisrec/metadata.json", "qwen/item_index/manifest.json")},
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "versions": {p: metadata.version(p) for p in packages},
                     "visible_gpus_smi": smi.stdout.strip().splitlines()},
        "source": {"work": sources, "qwen_source": qwen_sources},
    }
    path = args.run_dir / "manifest.json"
    dump(path, result)
    print(json.dumps({"path": str(path), "sha256": fingerprint(path)["sha256"],
                      "source_files": len(sources) + len(qwen_sources),
                      "rl_sha256": EXPECTED_WEIGHTS, "beam_sha256": EXPECTED_BEAM}))


if __name__ == "__main__":
    main()
