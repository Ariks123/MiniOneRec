"""Check the model, data and index lineage before publishing the report."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from hybrid.rho_bundle import digest


MINI = Path("weights/checkpoints/MiniOneRec")
UNI = Path("weights/checkpoints/UniSRec")
SFT = MINI / "Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1/epochs3/sft"
RL = MINI / "Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl"
UNISREC = UNI / "Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242"
UNI_EXPORT = UNI / "industrial_hybrid_inference_ep20_export_v1"
BASELINE = MINI / "Industrial_hybrid_beta001_inference_20261118"
COMPARISON = MINI / "Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118"
DATA = Path("MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey")
CATALOG = Path("MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(observed, expected, context):
    if observed != expected:
        raise ValueError(f"Archived evidence differs: {context}")


def audit(workspace):
    sft = read(workspace / SFT / "training_summary.json")
    rl = read(workspace / RL / "training_summary.json")
    rl_config = read(workspace / RL / "run_config.json")
    uni = read(workspace / UNISREC / "model_selection.json")
    export = read(workspace / UNI_EXPORT / "metadata.json")
    baseline = read(workspace / BASELINE / "manifest.json")
    comparison = read(workspace / COMPARISON / "manifest.json")
    prep = read(workspace / DATA / "preparation_manifest.json")
    clean = read(workspace / CATALOG / "cleaning_report.json")

    parent = rl["parent_sft"]
    require_equal(rl_config["parent_sft"], parent, "RL parent checkpoint metadata")
    require_equal(parent["selected_checkpoint"], sft["best_checkpoint"], "SFT → RL selected checkpoint")
    require_equal(parent["model_path"], sft["final_checkpoint"], "SFT → RL export")
    require_equal(parent["selected_step"], 5187, "SFT best step")
    require_equal(Path(rl["best_checkpoint"]).name, "checkpoint-828", "RL best step")
    require_equal((sft["final_checkpoint_contains_best_model"], rl["final_checkpoint_contains_best_model"]),
                  (True, True), "selected weights exported")
    require_equal((uni["best_epoch"], export["checkpoint_epoch_zero_based"]), (20, 19), "UniSRec epoch")
    require_equal(uni["checkpoint_sha256"], export["checkpoint_sha256"], "UniSRec export source")
    require_equal(export["item_count"], prep["items"], "UniSRec catalog size")
    require_equal(baseline["data"], comparison["data"], "input and original beam identities across sweeps")
    require_equal(baseline["verifier"], comparison["verifier"], "verifier identity across sweeps")
    require_equal(prep["rows"], baseline["data"]["row_counts"], "preparation and inference sample counts")
    require_equal(parent["provenance"]["sid_index_sha256"],
                  prep["expected_sid_sha256"], "SFT and hybrid SID")

    weights = "model.safetensors"
    sid = CATALOG / "Industrial_and_Scientific.index.json"
    uni_checkpoint = UNISREC / "checkpoints" / Path(uni["checkpoint"]).name
    expected = {
        DATA / "preparation_manifest.json": baseline["data"]["preparation_manifest"]["sha256"],
        sid: prep["expected_sid_sha256"],
        SFT / "run_config.json": parent["config_sha256"],
        SFT / "training_summary.json": parent["summary_sha256"],
        SFT / "checkpoint-5187" / weights: parent["model_sha256"],
        SFT / "final_checkpoint" / weights: parent["model_sha256"],
        RL / "checkpoint-828" / weights: baseline["verifier"]["files"][weights]["sha256"],
        RL / "final_checkpoint" / weights: baseline["verifier"]["files"][weights]["sha256"],
        RL / "evaluation/predictions.jsonl": baseline["data"]["reference_test_beam"]["sha256"],
        uni_checkpoint: export["checkpoint_sha256"],
        UNI_EXPORT / "metadata.json": baseline["draft_assets"]["unisrec/metadata.json"]["sha256"],
    }
    require_equal(expected[sid], baseline["data"]["original_sid_index"]["sha256"], "inference catalog hash")
    for name, content in baseline["data"]["split_files"].items():
        expected[DATA / name] = content["sha256"]
    for name, item in export["checksums"].items():
        expected[UNI_EXPORT / name] = item if isinstance(item, str) else item["sha256"]

    supporting = [
        CATALOG / "cleaning_report.json",
        SFT / "trainer_state.json",
        SFT / "evaluation/metrics.json",
        RL / "run_config.json",
        RL / "trainer_state.json",
        RL / "training_summary.json",
        RL / "evaluation/metrics.json",
        UNISREC / "config.yaml",
        UNISREC / "model_selection.json",
        UNISREC / "train.log",
    ]
    supporting.extend(root / filename for root in (BASELINE, COMPARISON) for filename in (
        "manifest.json", "report/selection.json", "report/test.csv", "report/validation.csv",
        "report/benchmarks.json", "report/batch_audit.json", "report/paired.json",
    ))

    def inspect(path):
        full = workspace / path
        sha = digest(full)
        if path in expected:
            require_equal(sha, expected[path], str(path))
        return {"path": str(path), "bytes": full.stat().st_size, "sha256": sha,
                "matched_archived_hash": path in expected}

    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(inspect, sorted(set(expected) | set(supporting))))
    return {
        "dataset": "Amazon Reviews 2018 Industrial and Scientific",
        "catalog_items": clean["items"],
        "users": clean["users"],
        "rows_by_split": prep["rows"],
        "cold_test_rows": clean["source_audit"]["cold_test_rows"],
        "unique_sids": prep["unique_sids"],
        "sft_checkpoint": "checkpoint-5187",
        "rl_checkpoint": "checkpoint-828",
        "unisrec_epoch": uni["best_epoch"],
        "inputs": inventory,
    }
