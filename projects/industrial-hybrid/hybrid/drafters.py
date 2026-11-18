"""Frozen drafters operating on original numeric item IDs. SID aliases stay distinct."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


RAW_SHA = "91a1c875e4b009f3a07b5febf00c2ef62a3813a55cfe873b825e18e32df1dd0b"
SID_SHA = "53f1aefbe8b8aa774326c122606c6525300c5a325e7b78817739d465aeb335f8"
UNI_SHA = "50f8bd01b5bdb5251ee1019ba0a3a003a6fd8ecfc987179cd3f16d13c36a64cb"


def stable_top(ids, scores, count):
    """Sort descending by score, then ascending by numeric product ID."""
    ids = np.asarray(ids)
    order = np.lexsort((ids, -scores))
    return ids[order[:count]].tolist()


class CatalogDrafter:
    def __init__(self, catalog):
        self.catalog = {int(item): tuple(tokens) for item, tokens in catalog.items()}
        self.all_ids = np.asarray(sorted(self.catalog), dtype=np.int32)
        if not np.array_equal(self.all_ids, np.arange(len(self.all_ids))):
            raise ValueError("Industrial catalog IDs must enumerate every vector row")
        groups = defaultdict(list)
        for item in self.all_ids:
            for width in (1, 2):
                groups[self.catalog[item][:width]].append(item)
        self.groups = {p: np.asarray(items, dtype=np.int32) for p, items in groups.items()}

    def allowed_groups(self, prefixes):
        if prefixes is None:
            return [((), self.all_ids)]
        prefixes = sorted(set(tuple(p) for p in prefixes))
        if any(len(p) not in (1, 2) for p in prefixes) or len({len(p) for p in prefixes}) > 1:
            raise ValueError("A drafting round requires uniform one- or two-token prefixes")
        return [(p, self.groups[p]) for p in prefixes if p in self.groups]

    def draft(self, query, allowed_prefixes=None, excluded=(), limit=100):
        groups = self.allowed_groups(allowed_prefixes)
        if not groups or limit <= 0:
            return []
        ids = np.concatenate([items for _, items in groups]) if len(groups) > 1 else groups[0][1]
        if excluded:
            ids = ids[~np.isin(ids, list(excluded))]
        return stable_top(ids, query[ids], limit)

    def validate_histories(self, histories):
        parsed = [[int(i) for i in history] for history in histories]
        if any(not history or len(history)>10 or any(i not in self.catalog for i in history)
               for history in parsed):
            raise ValueError("Missing history, unknown item, or more than 10 history items")
        return parsed


class AnnDrafter(CatalogDrafter):
    """Exact scoring for exhausted buckets, FAISS HNSW retrieval otherwise."""
    def __init__(self, catalog, assets):
        import faiss
        super().__init__(catalog)
        if _sha(assets/"raw.npy") != RAW_SHA:
            raise ValueError("Raw embedding matrix checksum differs")
        self.vectors = np.load(assets/"raw.npy", allow_pickle=False).astype(np.float32)
        n = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if self.vectors.shape != (len(self.catalog),2560) or not np.all(np.isfinite(n) & (n>0)):
            raise ValueError("Unexpected direct-vector catalog")
        self.vectors /= n
        faiss.omp_set_num_threads(1)  # deterministic single-threaded insertion order
        self.faiss = faiss
        self.indexes = {(): self.make_index(self.all_ids)}
        for prefix, ids in self.groups.items():
            if len(ids) > 2:  # below min requested delta (=2), exact exhausts the bucket
                self.indexes[prefix] = self.make_index(ids)
        self.metadata = {"library": "faiss", "version": faiss.__version__, "metric":"normalized_inner_product",
                         "raw_sha256":RAW_SHA, "query":"normalized_mean_of_normalized_item_vectors",
                         "M":16,"efConstruction":80,"efSearch":128,"indexes":len(self.indexes)}

    def make_index(self, ids):
        index = self.faiss.IndexHNSWFlat(self.vectors.shape[1], 16, self.faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 128
        index.add(np.ascontiguousarray(self.vectors[ids]))
        return index

    def prepare(self, histories):
        result = []
        for h in self.validate_histories(histories):
            q = self.vectors[h].mean(0)
            norm = np.linalg.norm(q)
            if not norm > 0:
                raise ValueError("Zero vector for direct retrieval history")
            result.append(q/norm)
        return result

    def draft(self, query, allowed_prefixes=None, excluded=(), limit=100):
        if limit <= 0:
            return []
        excluded = set(excluded)
        candidates = []
        scores = []
        for prefix, ids in self.allowed_groups(allowed_prefixes):
            # Fetch enough rows to cover exclusions even if these all score highest.
            n = min(len(ids), limit + len(excluded))
            if n == len(ids) or prefix not in self.indexes:
                picked = ids
                sim = np.einsum('ij,j->i', self.vectors[picked], query, optimize=False)
            else:
                score, local = self.indexes[prefix].search(query.reshape(1,-1), n)
                ok = local[0] >= 0
                picked, sim = ids[local[0][ok]], score[0][ok]
                if len(picked) < n:
                    # Graph connectivity is approximate; complete the eligible bucket
                    # when the backend returns fewer neighbors than requested.
                    picked = ids
                    sim = np.einsum('ij,j->i', self.vectors[picked], query, optimize=False)
            candidates.extend(picked)
            scores.extend(sim)
        candidates = np.asarray(candidates, dtype=np.int32)
        scores = np.asarray(scores, dtype=np.float32)
        keep = ~np.isin(candidates, list(excluded)) if excluded else slice(None)
        return stable_top(candidates[keep], scores[keep], limit)

    def exact_reference(self, query, limit=100):
        return stable_top(self.all_ids,np.einsum('ij,j->i',self.vectors,query,optimize=False),limit)


class UniSRecDrafter(CatalogDrafter):
    def __init__(self, catalog, assets, device, *, checkpoint_sha256=UNI_SHA):
        super().__init__(catalog)
        import torch
        directory = assets/'unisrec'
        meta = json.loads((directory/'metadata.json').read_text())
        if meta['checkpoint_sha256'] != checkpoint_sha256 or meta['provenance']['embedding_sha256'] != RAW_SHA:
            raise ValueError("UniSRec checkpoint/input identity differs")
        for name, sha in meta['checksums'].items():
            if _sha(directory/name) != sha:
                raise ValueError(f"UniSRec {name} identity differs")
        self.device = torch.device(device)
        self.model = torch.jit.load(str(directory/'encoder.pt'),map_location=self.device).eval()
        self.ids = np.asarray(meta['internal_ids_by_item_id'],dtype=np.int64)
        if set(self.ids.tolist()) != set(range(1,1+len(self.catalog))):
            raise ValueError("RecBole internal ID mapping incomplete or confused with PAD")
        matrix = np.load(directory/'item_vectors.npy',allow_pickle=False)
        if matrix.shape != (len(self.catalog),300) or not np.allclose(np.linalg.norm(matrix,axis=1),1.,atol=1e-5):
            raise ValueError("UniSRec index mismatch")
        self.matrix = torch.from_numpy(matrix).to(self.device)
        with np.load(directory/'parity.npz',allow_pickle=False) as golden, torch.inference_mode():
            query = self.model(torch.as_tensor(golden['internal_ids'],device=self.device),
                               torch.as_tensor(golden['lengths'],device=self.device))
            diff = float(np.max(np.abs(query.cpu().numpy()-golden['expected'])))
            if diff>5e-5:
                raise ValueError(f"UniSRec portable encoder differs from original by {diff}")
        self.metadata = {'checkpoint_sha256':checkpoint_sha256,'max_abs_error_device':diff,'rows':len(matrix),'dimension':300}

    def prepare(self,histories):
        import torch
        histories = self.validate_histories(histories)
        lengths = [len(h) for h in histories]
        ids = np.zeros((len(histories),10),dtype=np.int64)
        for i, h in enumerate(histories):
            ids[i,:len(h)] = self.ids[h]
        with torch.inference_mode():
            queries = self.model(torch.from_numpy(ids).to(self.device),torch.tensor(lengths,device=self.device))
            return (queries @ self.matrix.T).cpu().numpy()


class QwenDrafter(CatalogDrafter):
    def __init__(self,catalog,assets,device):
        super().__init__(catalog)
        import torch
        from data import SidSFTDataset, Tokenizer
        sys.path.insert(0,str(assets/'qwen_source'))
        from qwen_retrieval.config import checkpoint_identity
        from qwen_retrieval.encode_items import load_index,load_joint_model
        directory = assets/'qwen'
        identity = checkpoint_identity(directory/'final_checkpoint')
        vectors, rows, manifest = load_index(directory/'item_index',assets/'sid.json',identity)
        row_by_item = np.full(len(self.catalog), -1, dtype=np.int64)
        for idx, row in enumerate(rows):
            for item in row['item_ids']:
                if ''.join(self.catalog[int(item)])!=row['sid'] or row_by_item[int(item)]!=-1:
                    raise ValueError("Qwen index resolved a product to the wrong SID")
                row_by_item[int(item)]=idx
        if (row_by_item<0).any():
            raise ValueError("Incomplete Qwen product catalog")
        self.device = torch.device(device)
        self.tokenizer, self.model = load_joint_model(directory/'final_checkpoint',self.device)
        self.matrix = torch.from_numpy(vectors).to(self.device)
        self.row_by_item = row_by_item
        # Reuse the trained SidSFTDataset.pre method using history-only request rows.
        self.formatter = object.__new__(SidSFTDataset)
        self.formatter.test = True
        self.formatter.tokenizer = Tokenizer(self.tokenizer)
        self.metadata = {'model_identity':identity['sha256'],'unique_sids':len(rows),
                         'catalog_rows_sha256':manifest['catalog_rows_sha256'],'dimension':256}

    def prepare(self,histories):
        import torch
        history_rows = []
        for h in self.validate_histories(histories):
            values = [''.join(self.catalog[i]) for i in h]
            # SidSFTDataset.pre strips output in test mode. A sentinel avoids
            # accessing labels when encoding a request.
            history_rows.append({'history_item_sid':repr(values),'item_sid':''})
        self.formatter.data = SimpleNamespace(iloc=history_rows)
        seqs = []
        for i in range(len(history_rows)):
            ids = self.formatter.pre(i)['input_ids'][-507:]
            seqs.append({'input_ids':ids, 'attention_mask':[1]*len(ids)})
        batch = self.tokenizer.pad(seqs, padding=True, return_tensors='pt')
        with torch.inference_mode():
            queries = self.model.encode(input_ids=batch['input_ids'].to(self.device),
                                        attention_mask=batch['attention_mask'].to(self.device))
            scores = (queries.float() @ self.matrix.T).cpu().numpy()
        return np.ascontiguousarray(scores[:,self.row_by_item])


def build_drafter(name,catalog,device='cuda:0',assets='/assets'):
    assets = Path(assets)
    if _sha(assets/'sid.json') != SID_SHA:
        raise ValueError("Expected RL checkpoint-828 SID catalog")
    expected = json.loads((assets/'sid.json').read_text())
    if {str(i): list(v) for i,v in catalog.items()} != expected:
        raise ValueError("Caller catalog differs from the pinned SID mapping")
    if name == 'ann':
        return AnnDrafter(catalog,assets)
    if name == 'unisrec':
        return UniSRecDrafter(catalog,assets,device)
    if name == 'qwen':
        return QwenDrafter(catalog,assets,device)
    raise ValueError(f"Unknown drafter {name}")
