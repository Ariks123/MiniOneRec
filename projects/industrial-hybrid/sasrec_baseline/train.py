"""Train the four prescribed SASRec variants and test only the validation winner."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Dataset, KS, read_json, sha256, write_json


SEED = 42
BATCH = 512
EPOCHS = 30
PATIENCE = 5
GRID = [(hidden, lr) for hidden in (64, 300) for lr in (.001, .002)]


class ItemDomain:
    """Only expose the compact ID domain to the unchanged RecBole architecture."""

    def __init__(self, size):
        self.size = size

    def num(self, field):
        if field != "item_id":
            raise ValueError(f"Unexpected RecBole field {field}")
        return self.size


class Scoring(nn.Module):
    """Return item-ID scores for the tied learned embeddings, excluding padding."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, history, lengths):
        query = self.encoder(history, lengths)
        return query @ self.encoder.item_embedding.weight[1:].T


class PortableScoring(nn.Module):
    """Run the trained SASRec modules with position IDs that follow the export device."""

    def __init__(self, encoder):
        super().__init__()
        self.item_embedding = encoder.item_embedding
        self.position_embedding = encoder.position_embedding
        self.layer_norm = encoder.LayerNorm
        self.dropout = encoder.dropout
        self.transformer = encoder.trm_encoder
        positions = torch.arange(encoder.position_embedding.num_embeddings).unsqueeze(0)
        self.register_buffer("positions", positions)

    def forward(self, history, lengths):
        position_embedding = self.position_embedding(self.positions.expand_as(history))
        inputs = self.dropout(self.layer_norm(self.item_embedding(history) + position_embedding))

        attention = (history != 0).unsqueeze(1).unsqueeze(2)
        attention = torch.tril(attention.expand(-1, -1, history.size(-1), -1))
        attention = torch.where(attention, 0.0, -10000.0)
        transformer_output = self.transformer(inputs, attention, output_all_encoded_layers=True)[-1]
        indexes = (lengths - 1).view(-1, 1, 1).expand(-1, -1, transformer_output.size(-1))
        query = transformer_output.gather(dim=1, index=indexes).squeeze(1)
        return query @ self.item_embedding.weight[1:].T


def initialize(device):
    if version("recbole") != "1.2.1":
        raise ValueError("This run uses RecBole 1.2.1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(SEED)


def model_config(hidden, device):
    return {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "MAX_ITEM_LIST_LENGTH": 10,
        "device": device,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": hidden,
        "inner_size": 256,
        "hidden_dropout_prob": .2,
        "attn_dropout_prob": .2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": .02,
        "loss_type": "CE",
    }


def model_for(dataset, hidden, device):
    from recbole.model.sequential_recommender.sasrec import SASRec

    return Scoring(SASRec(model_config(hidden, device), ItemDomain(len(dataset.candidates) + 1))).to(device)


def encode(dataset, rows, device):
    prepared = [dataset.encode_history(row.history) for row in rows]
    histories, lengths = zip(*prepared)
    targets = [dataset.compact.get(row.target, 0) - 1 for row in rows]
    return (torch.tensor(histories, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device))


@torch.inference_mode()
def rank(model, prepared, dataset, k, batch):
    model.eval()
    histories, lengths, _ = prepared
    candidates = torch.tensor(dataset.candidates, dtype=torch.long, device=histories.device)
    for start in range(0, len(histories), batch):
        end = start + batch
        x, lens = histories[start:end], lengths[start:end]
        with torch.amp.autocast(device_type=histories.device.type, enabled=False):
            scores = model(x, lens.clamp_min(1))
        indexes = scores.argsort(descending=True, stable=True)[:, :k]
        ranking = candidates[indexes].cpu().numpy()
        for position in torch.nonzero(lens == 0, as_tuple=True)[0].tolist():
            ranking[position] = dataset.popularity[:k]
        yield ranking


def metrics(rankings, rows, dataset, cutoffs):
    target = np.array([row.target for row in rows], dtype=np.int64)
    matched = rankings == target[:, None]
    position = np.where(matched.any(1), matched.argmax(1) + 1, 0)
    cold = np.array([row.target not in dataset.compact for row in rows])
    result = {}
    for group, mask in (("overall", np.ones(len(rows), dtype=bool)), ("warm", ~cold), ("cold", cold)):
        values = {}
        ranks = position[mask]
        for k in cutoffs:
            hits = ranks[(ranks > 0) & (ranks <= k)]
            values[str(k)] = dict(rows=len(ranks), item_hits=len(hits), item_hr=len(hits)/len(ranks),
                                  item_ndcg=float(np.sum(1/np.log2(hits + 1)) / len(ranks)))
        result[group] = values
    return result


def case_name(index):
    hidden, lr = GRID[index]
    return f"h{hidden}-lr{str(lr).replace('.', 'p')}"


def selection_key(item):
    return (-item["validation_ndcg10"], -item["validation_hr10"], item["epoch"], item["grid_order"])


def train_case(dataset, run, index, device):
    hidden, lr = GRID[index]
    name = case_name(index)
    directory = run / "cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    summary_file = directory / "summary.json"
    if summary_file.is_file():
        summary = read_json(summary_file)
        if (summary["case_id"] != name or summary["split_sha256"] != dataset.split_sha256
                or summary["checkpoint_sha256"] != sha256(directory / "weights.pt")
                or summary["log_sha256"] != sha256(directory / "epochs.jsonl")):
            raise ValueError(f"Existing result {name} does not match the current inputs")
        return
    start = time.perf_counter()
    model = model_for(dataset, hidden, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_history, train_lengths, labels = encode(dataset, dataset.training, device)
    if torch.any(labels < 0) or torch.any(train_lengths == 0):
        raise ValueError("Training contains a missing candidate or an empty encoded request")
    validation = dataset.read("valid")
    dataset.diagnostics("valid", validation)
    valid_prepared = encode(dataset, validation, device)
    best = None
    max_ndcg = -1.0
    stale = 0
    log_path = directory / "epochs.jsonl"
    checkpoint = directory / "weights.pt"
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            accumulated = torch.zeros((), device=device, dtype=torch.float64)
            for indexes in torch.randperm(len(labels), device=device).split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                output = model(train_history[indexes], train_lengths[indexes])
                loss = F.cross_entropy(output, labels[indexes])
                loss.backward()
                optimizer.step()
                accumulated += loss.detach().double() * len(indexes)
            train_loss = accumulated.item() / len(labels)
            ranked = np.concatenate(list(rank(model, valid_prepared, dataset, 10, 128)))
            valid = metrics(ranked, validation, dataset, (10,))["overall"]["10"]
            if not np.isfinite(train_loss):
                raise ValueError(f"Nonfinite training loss in {name} epoch {epoch}")
            row = dict(case_id=name, grid_order=index, hidden_size=hidden, learning_rate=lr,
                       epoch=epoch, train_loss=train_loss, validation_hr10=valid["item_hr"],
                       validation_ndcg10=valid["item_ndcg"], elapsed_seconds=time.perf_counter()-start)
            if best is None or selection_key(row) < selection_key(best):
                best = row.copy()
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({key: value.cpu() for key, value in model.state_dict().items()}, temporary)
                temporary.replace(checkpoint)
            if valid["item_ndcg"] > max_ndcg:
                max_ndcg = valid["item_ndcg"]
                stale = 0
            else:
                stale += 1
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            if stale == PATIENCE:
                break
    write_json(summary_file, {**best, "epochs_trained": epoch, "early_stopped": stale == PATIENCE,
                              "checkpoint_sha256": sha256(checkpoint), "log_sha256": sha256(log_path),
                              "split_sha256": dataset.split_sha256, "runtime_seconds": time.perf_counter()-start,
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})


def write_predictions(path, rankings, rows, dataset):
    if len(rankings) != len(rows):
        raise ValueError("Incorrect prediction row count")
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for items, row in zip(rankings, rows, strict=True):
            record = dict(row_id=row.row_id, target=row.target, is_cold=row.target not in dataset.compact,
                          ranking=items.tolist())
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def export_trace(model, dataset, validation, run, device):
    cpu = torch.device("cpu")
    model = model.cpu().eval()
    rows = [row for row in validation if all(i in dataset.compact for i in row.history)][:48]
    histories, lens, _ = encode(dataset, rows, cpu)
    artifact = run / "model.ts"
    temporary = artifact.with_suffix(".ts.tmp")
    portable = PortableScoring(model.encoder).eval()
    traced = torch.jit.trace(portable, (histories[:4], lens[:4]), strict=True)
    traced.save(str(temporary))
    eager = model.to(device)
    restored = torch.jit.load(str(temporary), map_location=device).eval()
    checked = {}
    for batch in (1, 4):
        shape_results = []
        for history, lengths in zip(histories.split(batch), lens.split(batch), strict=True):
            actual = restored(history.to(device), lengths.to(device))
            expected = eager(history.to(device), lengths.to(device))
            torch.testing.assert_close(actual, expected)
            actual_order = actual.argsort(descending=True, stable=True)[:, :50]
            expected_order = expected.argsort(descending=True, stable=True)[:, :50]
            if not actual_order.equal(expected_order):
                raise ValueError(f"Exported SASRec changed exact Top50 ranking at batch {batch}")
            shape_results.append(len(history))
        checked[str(batch)] = sum(shape_results)
    temporary.replace(artifact)
    return eager, checked


def select(dataset, run, device):
    candidates = [read_json(run / "cases" / case_name(i) / "summary.json") for i in range(len(GRID))]
    for i, item in enumerate(candidates):
        directory = run / "cases" / case_name(i)
        if (item["case_id"] != case_name(i) or item["grid_order"] != i
                or item["checkpoint_sha256"] != sha256(directory / "weights.pt")
                or item["log_sha256"] != sha256(directory / "epochs.jsonl")
                or item["split_sha256"] != dataset.split_sha256):
            raise ValueError("An archived grid result does not match the current run")
    best = min(candidates, key=selection_key)
    model = model_for(dataset, best["hidden_size"], device)
    model.load_state_dict(torch.load(run / "cases" / best["case_id"] / "weights.pt", weights_only=True, map_location="cpu"))
    validation = dataset.read("valid")
    model, parity = export_trace(model, dataset, validation, run, device)
    results = {}
    diagnostics = {"train": dataset.diagnostics("train", dataset.training)}
    for split in ("valid", "test"):
        requests = validation if split == "valid" else dataset.read(split)
        diagnostics[split] = dataset.diagnostics(split, requests)
        prepared = encode(dataset, requests, device)
        # Reproduce the selection batch shape, then match the archived test protocol.
        batch_size = 128 if split == "valid" else 4
        ranked = np.concatenate(list(rank(model, prepared, dataset, max(KS), batch_size)))
        results[split] = metrics(ranked, requests, dataset, KS)
        write_predictions(run / f"{split}_predictions.jsonl", ranked, requests, dataset)
    valid = results["valid"]["overall"]["10"]
    if (valid["item_hr"], valid["item_ndcg"]) != (best["validation_hr10"], best["validation_ndcg10"]):
        raise ValueError("Exported validation predictions differ from checkpoint selection")
    # Selection is published last so dependent jobs can consume it atomically.
    summary = {
        "status": "completed",
        "selected": best,
        "cases": candidates,
        "recbole": version("recbole"),
        "torch": str(torch.__version__),
        "seed": SEED,
        "recipe": {"layers": 2, "heads": 2, "inner_size": 256, "dropout": .2, "activation": "gelu",
                   "optimizer": "Adam", "loss": "item_id_cross_entropy_train_seen", "batch_size": BATCH,
                   "max_epochs": EPOCHS, "patience": PATIENCE, "validation_ranking_batch_size": 128,
                   "test_ranking_batch_size": 4, "selection": ["validation_ndcg10", "validation_hr10",
                                                                  "earlier_epoch", "grid_order"]},
        "split_counts": {split: values["rows"] for split, values in diagnostics.items()},
        "split_diagnostics": diagnostics,
        "split_sha256": dataset.split_sha256,
        "preparation_sha256": dataset.preparation_sha256,
        "sorted_warm_original_ids": dataset.candidates,
        "popularity_fallback": dataset.popularity,
        "model_interface": {"history_length": 10, "pad": 0, "compact_id_offset": 1,
                            "scores_in_original_id_order": True, "score_dtype": "float32"},
        "export_parity_top50": parity,
        "quality": results,
        "artifact_sha256": {name: sha256(run / name)
                            for name in ("model.ts", "valid_predictions.jsonl", "test_predictions.jsonl")},
    }
    write_json(run / "selection.json", summary)
    print(json.dumps({"winner": best["case_id"], "epoch": best["epoch"], "validation": best["validation_ndcg10"],
                      "test": results["test"]["overall"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--case", type=int, choices=range(len(GRID)))
    task.add_argument("--select", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    initialize(device)
    dataset = Dataset(args.data)
    if args.select:
        select(dataset, args.run, device)
    else:
        train_case(dataset, args.run, args.case, device)


if __name__ == "__main__":
    main()
