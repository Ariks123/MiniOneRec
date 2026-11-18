"""Check the real five-task token stream without loading weights or training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    from transformers import AutoTokenizer
    from sft import TokenExtender
    from qwen_retrieval.training_data import build_training_data, RetrievalCollator
    data, inputs = Path(a.data_dir), Path(a.input_dir)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(TokenExtender(str(inputs), "Industrial_and_Scientific").get_new_tokens())
    training, valid, counts = build_training_data(str(data / "train.csv"), str(data / "valid.csv"),
        str(inputs / "Industrial_and_Scientific.item.json"),
        str(inputs / "Industrial_and_Scientific.index.json"), tokenizer)
    observed = Counter()
    for wrapped in training.datasets:
        for index in range(len(wrapped)):
            sample, old = wrapped[index], wrapped.source[index]
            assert all(sample[key] == old[key] for key in ("input_ids", "attention_mask", "labels"))
            observed[sample["task_type"]] += 1
    assert dict(observed) == counts and len(training) == 116093 and len(valid) == 4532
    seq = training.datasets[0]
    collator = RetrievalCollator(tokenizer)
    batch = collator([seq[index] for index in (0, 1, len(seq) - 1)] + [training.datasets[1][0]])
    assert batch["retrieval_mask"].tolist() == [True, True, True, False]
    result = {"status": "passed", "all_generated_input_tokens_and_labels_identical": True,
        "train_samples": len(training), "validation_samples": len(valid), "task_counts": counts,
        "added_sid_tokens": len(set(token for row in json.loads(
            (inputs / "Industrial_and_Scientific.index.json").read_text()).values() for token in row)),
        "mixed_task_collator": True, "production_training": False}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
