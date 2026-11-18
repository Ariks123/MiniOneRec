#!/usr/bin/env python3
"""Download and verify the exact official embedding model used by MiniOneRec."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = 'Qwen/Qwen3-Embedding-4B'
REVISION = '5cf2132abc99cad020ac570b19d031efec650f2b'
DESTINATION = ROOT / 'models' / 'Qwen3-Embedding-4B'
EXPECTED = {
    'model-00001-of-00002.safetensors': (4965826464, 'e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8'),
    'model-00002-of-00002.safetensors': (3077765624, 'ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1'),
}
FILES = ['config.json', 'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'README.md', *EXPECTED]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    os.environ.setdefault('HF_HOME', str(ROOT / '.runtime' / 'huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    from huggingface_hub import snapshot_download
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {MODEL_ID}@{REVISION} to {DESTINATION}', flush=True)
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=DESTINATION,
                      allow_patterns=FILES, token=False, max_workers=2)
    records = []
    for name in FILES:
        path = DESTINATION / name
        size, sha256 = path.stat().st_size, digest(path)
        if name in EXPECTED:
            assert (size, sha256) == EXPECTED[name], f'Weight integrity check failed: {name}'
        records.append({'name': name, 'bytes': size, 'sha256': sha256})
    config = json.loads((DESTINATION / 'config.json').read_text(encoding='utf-8'))
    assert config['hidden_size'] == 2560 and config['model_type'] == 'qwen3'
    index = json.loads((DESTINATION / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    assert set(index['weight_map'].values()) == set(EXPECTED)
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'source': f'https://huggingface.co/{MODEL_ID}/tree/{REVISION}',
                'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'validation': 'passed',
                'files': records, 'total_bytes': sum(item['bytes'] for item in records)}
    (DESTINATION / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
