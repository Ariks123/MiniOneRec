"""Metadata wrappers preserve the old five-task generated tokens and labels."""
from __future__ import annotations

import torch
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from qwen_retrieval.config import ITEM_TEMPLATE, build_catalog


class LabeledTaskDataset(Dataset):
    def __init__(self, source, task_type, catalog, sequence=False):
        self.source, self.task_type, self.sequence = source, task_type, sequence
        self.sid_ids = {row["sid"]: index for index, row in enumerate(catalog)}

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = dict(self.source[index])
        sample["task_type"] = self.task_type(index) if callable(self.task_type) else self.task_type
        sample["target_item_id"] = ""
        sample["target_sid"] = ""
        sample["retrieval_sid_id"] = -1
        if self.sequence:
            row = self.source.data.iloc[index]
            sid = str(row["item_sid"])
            sample.update(target_item_id=str(row["item_id"]), target_sid=sid,
                          retrieval_sid_id=self.sid_ids[sid])
        return sample


def build_training_data(train_file, eval_file, item_meta_path, sid_index_path,
                        tokenizer, cutoff_len=512, seed=42):
    from sft import build_training_tasks
    from data import SidSFTDataset
    from torch.utils.data import ConcatDataset
    catalog = build_catalog(sid_index_path)
    original, counts = build_training_tasks(train_file, item_meta_path, sid_index_path,
        tokenizer, cutoff_len, -1, seed, "industrial and scientific items", include_title_history=True)
    sequence, alignment, history, title_history = original.datasets
    wrapped = ConcatDataset([
        LabeledTaskDataset(sequence, "sid_sequence_recommendation", catalog, sequence=True),
        LabeledTaskDataset(alignment, lambda index: alignment.data[index]["task"], catalog),
        LabeledTaskDataset(history, "history_to_title", catalog),
        LabeledTaskDataset(title_history, "title_history_to_sid", catalog),
    ])
    valid = SidSFTDataset(eval_file, tokenizer, max_len=cutoff_len, sample=-1,
                          seed=seed, category="industrial and scientific items")
    return wrapped, valid, counts


class RetrievalCollator:
    def __init__(self, tokenizer, template=ITEM_TEMPLATE):
        self.tokenizer, self.template = tokenizer, template
        self.base = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    def __call__(self, features):
        samples = [{key: list(value) for key, value in row.items()
                    if key in ("input_ids", "attention_mask", "labels")} for row in features]
        batch = self.base(samples)
        # Validation retains the exact legacy generation-only path.
        if "retrieval_sid_id" not in features[0]:
            return batch
        selected = [index for index, row in enumerate(features) if row["retrieval_sid_id"] >= 0]
        mask = torch.zeros(len(features), dtype=torch.bool)
        mask[selected] = True
        batch["retrieval_mask"] = mask
        batch["retrieval_sid_ids"] = torch.tensor(
            [features[index]["retrieval_sid_id"] for index in selected], dtype=torch.long)
        width = batch["input_ids"].shape[1]
        encoded = [self.tokenizer.encode(self.template.format(sid=features[index]["target_sid"]),
                                        add_special_tokens=False) for index in selected]
        if any(not tokens or len(tokens) > width for tokens in encoded):
            raise ValueError("Item template exceeds the generation batch width")
        batch["item_input_ids"] = torch.tensor(
            [[self.tokenizer.pad_token_id] * (width - len(tokens)) + tokens for tokens in encoded],
            dtype=torch.long).reshape(len(selected), width)
        batch["item_attention_mask"] = torch.tensor(
            [[0] * (width - len(tokens)) + [1] * len(tokens) for tokens in encoded],
            dtype=torch.long).reshape(len(selected), width)
        if selected:
            labels = batch["labels"][mask]
            boundary = (labels != -100).long().argmax(1) - 1
            if (boundary < 0).any() or not batch["attention_mask"][mask].gather(1, boundary[:, None]).bool().all():
                raise ValueError("Missing valid answer-free SEQ query boundary")
        return batch
