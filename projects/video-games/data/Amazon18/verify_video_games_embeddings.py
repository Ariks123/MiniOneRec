#!/usr/bin/env python3
"""Validate the complete Video Games embeddings and their local provenance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/Amazon18/Video_Games'
EXPECTED_WEIGHTS = {
    'model-00001-of-00002.safetensors': (4965826464, 'e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8'),
    'model-00002-of-00002.safetensors': (3077765624, 'ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path, digest=None):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': digest if digest is not None else sha256(path)}


def validate_catalog(metadata, item_map, row_ids, count):
    expected = set(range(count))
    require(isinstance(metadata, dict) and len(metadata) == count, 'Metadata item count mismatch')
    require(set(metadata) == {str(i) for i in expected}, 'Metadata keys must be exactly 0..N-1')
    require(isinstance(row_ids, list) and all(type(value) is int for value in row_ids), 'Row IDs must be an integer list')
    require(row_ids == list(range(count)), 'Embedding row IDs are not exactly 0..N-1 in order')
    require(len(item_map) == count, 'ASIN mapping row count mismatch')
    asins, ids = [], []
    for line_number, line in enumerate(item_map, start=1):
        fields = line.rstrip('\r\n').split('\t')
        require(len(fields) == 2 and fields[0] != '', f'Malformed ASIN mapping at line {line_number}')
        asins.append(fields[0])
        try:
            ids.append(int(fields[1]))
        except ValueError as error:
            raise ValueError(f'Non-integer item ID at mapping line {line_number}') from error
    require(len(set(asins)) == count, 'Duplicate ASIN in item2id')
    require(len(set(ids)) == count and set(ids) == expected, 'ASIN mapping IDs do not match metadata and embedding rows')
    return {'items': count, 'unique_asins': len(set(asins)), 'minimum_item_id': 0,
            'maximum_item_id': count - 1, 'metadata_and_item2id_agree': True,
            'embedding_rows_in_item_id_order': True}


def validate_array(array, shape, chunk_size=512):
    require(array.shape == tuple(shape), f'Embedding shape {array.shape} != {tuple(shape)}')
    require(array.dtype == np.dtype('float32'), f'Embedding dtype {array.dtype} != float32')
    norms = np.empty(shape[0], dtype=np.float64)
    for start in range(0, shape[0], chunk_size):
        block = np.asarray(array[start:start + chunk_size])
        require(np.isfinite(block).all(), f'NaN or Inf in embedding rows beginning at {start}')
        block_norms = np.linalg.norm(block.astype(np.float64), axis=1)
        require(np.isfinite(block_norms).all() and (block_norms > 0).all(),
                f'Zero or invalid embedding row beginning at {start}')
        norms[start:start + len(block)] = block_norms
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': True,
            'nonzero_rows': int(shape[0]), 'zero_rows': 0,
            'l2_norm': {'min': float(norms.min()), 'max': float(norms.max()),
                        'mean': float(norms.mean()), 'median': float(np.median(norms))}}


def validate_manifest(manifest, array_stats, metadata_hash, embedding_hash):
    require(manifest['dataset'] == 'Video_Games', 'Manifest dataset mismatch')
    require(manifest['validation'] == 'passed', 'Embedding manifest did not pass validation')
    require(manifest['shape'] == array_stats['shape'], 'Manifest shape mismatch')
    require(manifest['dtype'] == array_stats['dtype'], 'Manifest dtype mismatch')
    require(manifest['item_metadata_sha256'] == metadata_hash, 'Metadata SHA-256 differs from generation manifest')
    require(manifest['embedding_sha256'] == embedding_hash, 'Embedding SHA-256 differs from generation manifest')
    require(manifest['features'] == ['title', 'description'], 'Unexpected embedding text fields')
    require(manifest['pooling'] == 'masked_mean' and manifest['normalized'] is False,
            'Embedding settings differ from the original MiniOneRec pipeline')
    require(manifest['max_sent_len'] == 2048 and manifest['word_drop_ratio'] == -1,
            'Token limit or word-drop settings differ from the requested pipeline')
    require(type(manifest['batch_size']) is int and manifest['batch_size'] >= 1, 'Invalid inference batch size')


def validate_preflight(preflight, count, dimension, checkpoint):
    require(preflight['validation'] == 'passed', 'Model preflight did not pass')
    require(preflight['model_id'] == 'Qwen/Qwen3-Embedding-4B', 'Unexpected embedding model')
    require(preflight['item_count'] == count and preflight['embedding_dimension'] == dimension,
            'Preflight catalog shape mismatch')
    require(preflight['weight_sha256_verified_against_official'] is True, 'Official weights were not verified')
    require(preflight['pooling'] == 'masked_mean' and preflight['normalized'] is False and preflight['max_sent_len'] == 2048,
            'Preflight settings differ from the requested embedding pipeline')
    require(Path(preflight['model_directory']).resolve() == Path(checkpoint).resolve(), 'Checkpoint path differs from the verified model directory')
    files = {entry['name']: entry for entry in preflight['files']}
    for name, (size, digest) in EXPECTED_WEIGHTS.items():
        require(name in files and files[name]['bytes'] == size and files[name]['sha256'] == digest,
                f'Official weight verification mismatch for {name}')
    return {'model_id': preflight['model_id'], 'model_directory': preflight['model_directory'],
            'weight_sha256_verified_against_official': True,
            'weights': [files[name] for name in EXPECTED_WEIGHTS],
            'preflight_completed_at_utc': preflight['completed_at_utc'],
            'gpu': preflight['gpu'], 'text_cleaning': preflight['text_cleaning'],
            'token_length': preflight['token_length']}


def validate_execution(execution):
    require(execution.get('done') is True and execution.get('exit_code') == 0, 'Generation process did not finish successfully')
    require(execution.get('signal') is None, 'Generation process ended with a signal')
    require(isinstance(execution.get('command'), list) and execution['command'], 'Missing execution command')
    command = execution['command']
    require(all(isinstance(argument, str) for argument in command), 'Execution command arguments must be strings')
    normalized_arguments = [argument.replace('\\', '/').lower() for argument in command]
    explicit_dataset = any(
        argument == '--dataset=video_games' or
        (argument == '--dataset' and index + 1 < len(normalized_arguments) and normalized_arguments[index + 1] == 'video_games')
        for index, argument in enumerate(normalized_arguments)
    )
    wrapper = re.search(r'(?:^|[\s/\x27\x22])rq/text2emb/video_games_text2emb\.sh(?:$|[\s\x27\x22])',
                        ' '.join(normalized_arguments)) is not None
    require(explicit_dataset or wrapper, 'Execution command is not for Video_Games')
    started = datetime.fromisoformat(execution['started_at_utc'].replace('Z', '+00:00'))
    finished = datetime.fromisoformat(execution['finished_at_utc'].replace('Z', '+00:00'))
    require(finished >= started, 'Execution timestamps are out of order')
    log = Path(execution['log_file'])
    require(log.is_file(), 'Generation log file is missing')
    return (finished - started).total_seconds(), log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--execution', type=Path, required=True, help='Completed generation log.execution.json')
    parser.add_argument('--preflight', type=Path, help='Defaults to Video_Games.embedding_preflight.json in data-dir')
    parser.add_argument('--output', type=Path, help='Defaults to Video_Games.embedding_validation.json in data-dir')
    args = parser.parse_args()
    base = args.data_dir.resolve(strict=True)
    paths = {
        'embedding': base / 'Video_Games.emb-qwen-td.npy',
        'row_ids': base / 'Video_Games.emb-qwen-td.npy.ids.json',
        'generation_manifest': base / 'Video_Games.emb-qwen-td.npy.manifest.json',
        'item_metadata': base / 'Video_Games.item.json',
        'item2id': base / 'Video_Games.item2id',
        'preflight': args.preflight or base / 'Video_Games.embedding_preflight.json',
        'execution_record': args.execution,
    }
    output = (args.output or base / 'Video_Games.embedding_validation.json').resolve()
    require(output not in {path.resolve() for path in paths.values()}, 'Validation report must not overwrite an input file')
    for key, path in paths.items():
        require(path.is_file(), f'Missing {key}: {path}')
    metadata = read_json(paths['item_metadata'])
    row_ids = read_json(paths['row_ids'])
    manifest = read_json(paths['generation_manifest'])
    preflight = read_json(paths['preflight'])
    execution = read_json(paths['execution_record'])
    item_map = paths['item2id'].read_text(encoding='utf-8').splitlines()
    catalog = validate_catalog(metadata, item_map, row_ids, 16572)
    array = np.load(paths['embedding'], mmap_mode='r', allow_pickle=False)
    array_stats = validate_array(array, (16572, 2560))
    hashes = {key: sha256(path) for key, path in paths.items()}
    validate_manifest(manifest, array_stats, hashes['item_metadata'], hashes['embedding'])
    model = validate_preflight(preflight, 16572, 2560, manifest['checkpoint'])
    seconds, log_path = validate_execution(execution)
    require(output != log_path.resolve(), 'Validation report must not overwrite the execution log')
    files = {key: file_record(path, hashes[key]) for key, path in paths.items()}
    files['log'] = file_record(log_path)
    report = {
        'validation': 'passed', 'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Video_Games', 'catalog': catalog, 'embeddings': array_stats,
        'embedding_file_bytes': files['embedding']['bytes'],
        'embedding_file_decimal_MB': files['embedding']['bytes'] / 1e6,
        'model_verification': model,
        'generation_settings': {key: manifest[key] for key in ('features', 'pooling', 'normalized', 'max_sent_len', 'batch_size', 'word_drop_ratio')},
        'execution': {**execution, 'wall_seconds': seconds}, 'files': files,
        'notes': ['Model weight hashes are copied from the completed preflight that verified the local official weights.',
                  'Validation reads the saved embedding matrix and checks all rows; no model is loaded and no GPU is used.'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'validation': report['validation'], 'output': str(output),
                      'embeddings': array_stats, 'embedding_file_bytes': files['embedding']['bytes'],
                      'wall_seconds': seconds}, ensure_ascii=False))


if __name__ == '__main__':
    main()
