#!/usr/bin/env python3
"""Verify a local Qwen model and profile the original MiniOneRec embedding method."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('G:/Qwen')
DATA = ROOT / 'data/Amazon18/Video_Games'
REPORT = DATA / 'Video_Games.embedding_preflight.json'
EXPECTED = {
    'model-00001-of-00002.safetensors': (4965826464, 'e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8'),
    'model-00002-of-00002.safetensors': (3077765624, 'ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1'),
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    started = time.perf_counter()
    files = []
    for path in sorted(MODEL.iterdir()):
        if not path.is_file():
            continue
        print('Checking file:', path.name, flush=True)
        size, sha = path.stat().st_size, digest(path)
        if path.name in EXPECTED:
            assert (size, sha) == EXPECTED[path.name], path.name
        files.append({'name': path.name, 'bytes': size, 'sha256': sha})
    assert set(EXPECTED).issubset({entry['name'] for entry in files})
    config = json.loads((MODEL / 'config.json').read_text(encoding='utf-8'))
    assert config['model_type'] == 'qwen3' and config['hidden_size'] == 2560
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    assert set(index['weight_map'].values()) == set(EXPECTED)
    from safetensors import safe_open
    for name in EXPECTED:
        with safe_open(str(MODEL / name), framework='pt', device='cpu') as weights:
            assert set(weights.keys()) == {key for key, value in index['weight_map'].items() if value == name}
    print('All model files and official weight SHA-256 checks passed.', flush=True)
    sys.path.insert(0, str(ROOT / 'rq/text2emb'))
    import amazon_text2emb as pipeline
    texts = pipeline.preprocess_text(SimpleNamespace(root=str(DATA), dataset='Video_Games'))
    assert sorted(item for item, _ in texts) == list(range(16572))
    metadata = json.loads((DATA / 'Video_Games.item.json').read_text(encoding='utf-8'))
    tokenizer = pipeline.AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = []
    for start in range(0, len(texts), 256):
        encoded = tokenizer([text for _, text in texts[start:start + 256]], max_length=2048, truncation=True, return_length=True)
        lengths.extend(encoded['length'])
    longest = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
    print('Token lengths:', {'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths))}, flush=True)
    tokenizer, model = pipeline.load_qwen_model(str(MODEL), local_files_only=True)
    model = model.to('cuda:0').eval()
    profiles = []
    for batch_size in (1, 4, 8):
        batch = [texts[i][1] for i in longest[:batch_size]]
        encoded = tokenizer(batch, max_length=2048, truncation=True, return_tensors='pt', padding=True).to('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        with torch.no_grad():
            output = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, use_cache=False)
            hidden = output.last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        torch.cuda.synchronize()
        assert tuple(vector.shape) == (batch_size, 2560) and torch.isfinite(vector).all()
        profiles.append({'batch_size': batch_size, 'padded_tokens': encoded.input_ids.shape[1],
                         'seconds': time.perf_counter() - step_start,
                         'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                         'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30})
        print('GPU profile:', json.dumps(profiles[-1]), flush=True)
        del encoded, output, hidden, mask, vector
        torch.cuda.empty_cache()
    result = {'validation': 'passed', 'model_id': 'Qwen/Qwen3-Embedding-4B', 'model_directory': str(MODEL),
              'files': files, 'weight_sha256_verified_against_official': True,
              'item_count': len(texts), 'embedding_dimension': 2560, 'pooling': 'masked_mean', 'normalized': False,
              'max_sent_len': 2048, 'token_length': {'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths)),
              'p50': float(np.percentile(lengths, 50)), 'p95': float(np.percentile(lengths, 95))},
              'text_cleaning': {'description_discarded_by_original_2000_char_rule': sum(pipeline.clean_text(value['description']) == '' for value in metadata.values()),
                                'originally_empty_descriptions': sum(value['description'] == '' for value in metadata.values())},
              'profiles': profiles, 'selected_batch_size': 8, 'gpu': torch.cuda.get_device_name(0),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Preflight passed:', REPORT, flush=True)


if __name__ == '__main__':
    main()
