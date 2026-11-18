"""Four-rank smoke of production SFT monitoring, baseline, save, and reload. No RL."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["DS_ACCELERATOR"] = "cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sft-monitor-smoke"))
    parser.add_argument("--catalog-size", type=int, default=50)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from docker.smoke_distributed import tiny_model
    from sft import train

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    work = args.work_dir
    raw, prepared, base, output = (work / part for part in ("raw", "prepared", "base", "run"))
    if rank == 0:
        if output.exists():
            raise RuntimeError("Choose a fresh smoke work directory; existing artifacts are retained")
        for folder in (raw, prepared, base):
            folder.mkdir(parents=True, exist_ok=True)
        # SID vocabulary is genuinely added by production SFT, rather than preinstalled.
        words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "recommend", "game"]
        backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer, backend.decoder = pre_tokenizers.WhitespaceSplit(), decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                           bos_token="[BOS]", eos_token="[EOS]", padding_side="left", model_max_length=512)
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        catalog = [f"<a_0><b_0><c_0><d_{i}>" for i in range(args.catalog_size)]
        model = tiny_model(tokenizer, torch.device("cpu"))
        model.config.max_position_embeddings = 512
        model.save_pretrained(base)
        tokenizer.save_pretrained(base)
        indices = {str(i): ["<a_0>", "<b_0>", "<c_0>", f"<d_{i}>"] for i in range(args.catalog_size)}
        metadata = {str(i): {"title": "game " + str(i), "description": "game " + str(i)} for i in range(args.catalog_size)}
        (raw / "Video_Games.index.json").write_text(json.dumps(indices) + "\n")
        (raw / "Video_Games.item.json").write_text(json.dumps(metadata) + "\n")
        sources = {
            "sid_index": {"filename": "Video_Games.index.json"},
            "item_metadata": {"filename": "Video_Games.item.json"},
        }
        fields = ["user_id", "history_item_title", "item_title", "history_item_id", "item_id", "history_item_sid", "item_sid"]
        outputs = {}
        for split, count in (("train", 14), ("valid", 8), ("test", 8)):
            path = prepared / (split + ".csv")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i in range(count):
                    writer.writerow({"user_id": "A" + str(i), "history_item_title": repr(["game 0"]),
                                     "item_title": "game " + str(i % args.catalog_size), "history_item_id": "[0]", "item_id": i % args.catalog_size,
                                     "history_item_sid": repr([catalog[0]]), "item_sid": catalog[i % args.catalog_size]})
            outputs[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        info = prepared / "info.txt"
        info.write_text("".join(sid + "\tgame " + str(i) + "\t" + str(i) + "\n" for i, sid in enumerate(catalog)))
        outputs[info.name] = {"sha256": hashlib.sha256(info.read_bytes()).hexdigest()}
        for record in sources.values():
            record["sha256"] = hashlib.sha256((raw / record["filename"]).read_bytes()).hexdigest()
        manifest = {"validation": "passed", "check_only": False, "dataset": "Video_Games", "selected_sid_epoch": 2900,
                    "preserve_all_interaction_rows": True, "sampling_applied": False, "keep_longest_only": False,
                    "rows": {"train": 14, "valid": 8, "test": 8}, "sources": sources, "outputs": outputs}
        (prepared / "preparation_manifest.json").write_text(json.dumps(manifest))
    dist.barrier()
    train(base_model=str(base), train_file=str(prepared / "train.csv"), eval_file=str(prepared / "valid.csv"),
          output_dir=str(output), category="Video_Games", batch_size=16, micro_batch_size=2,
          num_epochs=1, learning_rate=1e-3, cutoff_len=512, sid_index_path=str(raw / "Video_Games.index.json"),
          item_meta_path=str(raw / "Video_Games.item.json"), deepspeed=str(ROOT / "config/h20_zero2.json"),
          monitoring=True, monitor_window=2, baseline_eval=True)
    dist.barrier()
    if rank == 0:
        summary = json.loads((output / "monitor/summary.json").read_text())
        training = json.loads((output / "training_summary.json").read_text())
        baseline = json.loads((output / "baseline_validation.json").read_text())
        if not (summary["grad_norm_count"] > 0 and summary["grad_norm_max"] >= summary["grad_norm_mean"] >= 0):
            raise RuntimeError("Global gradNorm statistics were not recorded")
        if training["status"] != "completed" or baseline["status"] != "passed":
            raise RuntimeError("Production SFT/baseline did not complete")
        reloaded = AutoModelForCausalLM.from_pretrained(output / "final_checkpoint", local_files_only=True)
        best = AutoModelForCausalLM.from_pretrained(training["best_checkpoint"], local_files_only=True)
        if any(not torch.equal(value, best.state_dict()[key]) for key, value in reloaded.state_dict().items()):
            raise RuntimeError("Final model weights differ from the selected best checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(output / "final_checkpoint", local_files_only=True)
        catalog = json.loads((raw / "Video_Games.index.json").read_text())
        for tokens in catalog.values():
            for token in tokens:
                if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
                    raise RuntimeError("Saved SID tokenizer is not atomic")
        provenance = json.loads((output / "final_checkpoint/posttraining_provenance.json").read_text())
        expected = hashlib.sha256((raw / "Video_Games.index.json").read_bytes()).hexdigest()
        if provenance["sid_index_sha256"] != expected:
            raise RuntimeError("Saved model provenance is incorrect")
        result = {"status": "passed", "world_size": 4, "baseline": baseline,
                  "gradient_statistics": summary, "checkpoint_reload": "passed", "rl_started": False,
                  "best_checkpoint_weights_match": True, "vocabulary_extension": "passed",
                  "work_dir": str(work), "fixture_sid_sha256": expected}
        (work / "smoke_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        del reloaded, best
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
