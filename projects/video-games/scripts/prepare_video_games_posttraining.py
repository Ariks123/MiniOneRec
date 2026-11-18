#!/usr/bin/env python3
"""Prepare post-training CSVs from the approved Video Games epoch-2900 SIDs.

This CPU-only conversion preserves every input row and its train/valid/test split.
It never trains a model or modifies the cleaned catalog, SID map, or interactions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'Video_Games'
SELECTED_EPOCH = 2900
EXPECTED_SID_SHA256 = 'ca9d730905a67cc36cd77c665a3165de823f2689660b19741df9ee884993085f'
SPLITS = ('train', 'valid', 'test')
INTER_HEADER = ['user_id:token', 'item_id_list:token_seq', 'item_id:token']
CSV_FIELDS = ['user_id', 'history_item_title', 'item_title', 'history_item_id',
              'item_id', 'history_item_sid', 'item_sid']
OUTPUT_FILES = [*(split + '.csv' for split in SPLITS), 'info.txt', 'preparation_manifest.json']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_catalog(input_dir: Path, expected_sid_sha256: str):
    require(bool(re.fullmatch(r'[0-9a-fA-F]{64}', expected_sid_sha256)),
            'expected SID SHA-256 must contain exactly 64 hexadecimal digits')
    sid_path = input_dir / (DATASET + '.index.json')
    meta_path = input_dir / (DATASET + '.item.json')
    sid_bytes = sid_path.read_bytes()
    meta_bytes = meta_path.read_bytes()
    sid_hash = hashlib.sha256(sid_bytes).hexdigest()
    require(sid_hash == expected_sid_sha256.lower(),
            f'Approved epoch-2900 SID hash mismatch: expected {expected_sid_sha256.lower()}, got {sid_hash}')
    sids = json.loads(sid_bytes.decode('utf-8-sig'))
    metadata = json.loads(meta_bytes.decode('utf-8-sig'))
    require(isinstance(sids, dict) and bool(sids), 'SID catalog must be a nonempty object')
    require(isinstance(metadata, dict) and set(metadata) == set(sids), 'Metadata/SID item IDs differ')
    require(set(sids) == {str(i) for i in range(len(sids))}, 'Catalog IDs must be complete contiguous integers starting at zero')
    sid_strings = {}
    titles = {}
    for item_id in range(len(sids)):
        key = str(item_id)
        tokens = sids[key]
        require(isinstance(tokens, list) and len(tokens) == 4, f'Item {key}: expected four SID tokens')
        for level, token in zip('abcd', tokens):
            require(isinstance(token, str) and bool(re.fullmatch(r'<' + level + r'_(0|[1-9][0-9]*)>', token)),
                    f'Item {key}: invalid {level}-level SID token')
            if level != 'd':
                require(int(token[3:-1]) < 256, f'Item {key}: semantic code is outside [0,255]')
        sid_strings[key] = ''.join(tokens)
        item = metadata[key]
        require(isinstance(item, dict), f'Item {key}: metadata must be an object')
        title = item.get('title', f'Item_{key}')
        require(isinstance(title, str), f'Item {key}: title must be text')
        titles[key] = title
    require(len(set(sid_strings.values())) == len(sids), 'Complete four-token SIDs must be unique')
    sources = {
        'sid_index': {'filename': sid_path.name, 'sha256': sid_hash},
        'item_metadata': {'filename': meta_path.name, 'sha256': hashlib.sha256(meta_bytes).hexdigest()},
    }
    return sid_strings, titles, sources


def interaction_rows(path: Path, sid_strings: dict, titles: dict):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header = next(reader, None)
        require(header == INTER_HEADER, f'{path.name}: unexpected interaction header')
        for line_number, fields in enumerate(reader, 2):
            prefix = f'{path.name}:{line_number}'
            require(len(fields) == 3, f'{prefix}: expected three tab-separated fields')
            user_id, history_text, target_text = fields
            require(bool(re.fullmatch(r'0|[1-9][0-9]*', user_id)), f'{prefix}: invalid user ID')
            history = history_text.split()
            require(1 <= len(history) <= 20, f'{prefix}: history length must be 1-20')
            require(all(bool(re.fullmatch(r'0|[1-9][0-9]*', item)) for item in history),
                    f'{prefix}: history contains an invalid item ID')
            require(bool(re.fullmatch(r'0|[1-9][0-9]*', target_text)), f'{prefix}: invalid target item ID')
            require(all(item in sid_strings for item in history), f'{prefix}: history contains an unmapped item ID')
            require(target_text in sid_strings, f'{prefix}: target item ID is not in the catalog')
            yield {
                'user_id': 'A' + user_id,
                'history_item_title': repr([titles[item] for item in history]),
                'item_title': titles[target_text],
                'history_item_id': repr([int(item) for item in history]),
                'item_id': int(target_text),
                'history_item_sid': repr([sid_strings[item] for item in history]),
                'item_sid': sid_strings[target_text],
            }


def prepare(input_dir: Path, output_dir: Path, expected_sid_sha256: str = EXPECTED_SID_SHA256,
            check_only: bool = False) -> dict:
    input_dir = Path(input_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    require(input_dir != output_dir, 'Keep post-training output separate from the cleaned input directory')
    sid_strings, titles, sources = load_catalog(input_dir, expected_sid_sha256)
    row_counts = {}
    for split in SPLITS:
        path = input_dir / f'{DATASET}.{split}.inter'
        before = sha256(path)
        count = sum(1 for _ in interaction_rows(path, sid_strings, titles))
        require(count > 0, f'{path.name}: split contains no samples')
        require(sha256(path) == before, f'{path.name}: input changed during validation')
        row_counts[split] = count
        sources[split] = {'filename': path.name, 'sha256': before}
    # Check metadata and SID were stable throughout the complete preflight as well.
    for key in ('sid_index', 'item_metadata'):
        record = sources[key]
        require(sha256(input_dir / record['filename']) == record['sha256'], 'Catalog changed during validation')
    manifest = {
        'schema_version': 1,
        'dataset': DATASET,
        'selected_sid_epoch': SELECTED_EPOCH,
        'expected_sid_sha256': expected_sid_sha256.lower(),
        'items': len(sid_strings),
        'complete_sid_tokens': 4,
        'complete_sids_unique': True,
        'history_length': {'minimum': 1, 'maximum': 20},
        'rows': row_counts,
        'preserve_all_interaction_rows': True,
        'sampling_applied': False,
        'keep_longest_only': False,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'sources': sources,
        'validation': 'passed',
        'check_only': bool(check_only),
    }
    if check_only:
        return manifest

    # All inputs have passed validation before any output files are created.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.video-games-sid2900-', dir=output_dir.parent)).resolve()
    require(stage.parent == output_dir.parent, 'Unexpected staging directory')
    try:
        for split in SPLITS:
            count = 0
            with (stage / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator='\n')
                writer.writeheader()
                for row in interaction_rows(input_dir / f'{DATASET}.{split}.inter', sid_strings, titles):
                    writer.writerow(row)
                    count += 1
            require(count == row_counts[split], f'{split}: input row count changed during conversion')
        with (stage / 'info.txt').open('w', encoding='utf-8', newline='') as stream:
            for item_id in range(len(sid_strings)):
                key = str(item_id)
                title = titles[key].replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
                stream.write(f'{sid_strings[key]}\t{title}\t{key}\n')
        for record in sources.values():
            require(sha256(input_dir / record['filename']) == record['sha256'],
                    'Input changed during conversion; no generated files have been published')
        manifest['outputs'] = {
            name: {'filename': name, 'sha256': sha256(stage / name), 'bytes': (stage / name).stat().st_size}
            for name in OUTPUT_FILES if name != 'preparation_manifest.json'
        }
        manifest['created_at_utc'] = datetime.now(timezone.utc).isoformat()
        (stage / 'preparation_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        output_dir.mkdir(parents=True, exist_ok=True)
        # Commit the manifest last. Consumers can verify its output hashes before training.
        for name in OUTPUT_FILES:
            os.replace(stage / name, output_dir / name)
    finally:
        # Only known files in our own verified staging directory are removed; no recursive deletion.
        for name in OUTPUT_FILES:
            path = stage / name
            if path.is_file():
                path.unlink()
        stage.rmdir()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/Amazon18/Video_Games')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/Amazon18/posttraining/Video_Games_sid2900')
    parser.add_argument('--expected-sid-sha256', default=EXPECTED_SID_SHA256,
                        help='Approved complete SID file SHA-256; change only for an explicitly approved catalog')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without writing any output')
    args = parser.parse_args()
    try:
        result = prepare(args.input_dir, args.output_dir, args.expected_sid_sha256, args.check_only)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f'Preparation failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
