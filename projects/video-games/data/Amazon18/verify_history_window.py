#!/usr/bin/env python3
"""Verify every regenerated history against complete sequences and the previous run."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(histogram):
    count = sum(histogram.values())
    total = sum(length * frequency for length, frequency in histogram.items())
    return {'min': min(histogram), 'max': max(histogram), 'mean': total / count,
            'samples': count, 'total_history_items': total,
            'samples_over_10': sum(n for length, n in histogram.items() if length > 10),
            'histogram': {str(length): histogram[length] for length in range(1, max(histogram) + 1)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--max-history-len', type=int, required=True)
    args = parser.parse_args()
    assert args.max_history_len >= 1
    report = json.loads((args.baseline_dir / 'cleaning_report.json').read_text(encoding='utf-8'))
    previous_hashes = json.loads((args.baseline_dir / 'baseline_hashes.json').read_text(encoding='utf-8'))
    execution = json.loads(args.execution.read_text(encoding='utf-8'))
    assert execution['exit_code'] == 0
    dataset = report['dataset']
    sequences = json.loads((args.data_dir / f'{dataset}.inter.json').read_text(encoding='utf-8'))
    assert len(sequences) == report['users']
    assert sum(map(len, sequences.values())) == report['interactions']
    assert len({item for sequence in sequences.values() for item in sequence}) == report['items']
    assert min(map(len, sequences.values())) == report['min_user_interactions']
    split_names = {f'{dataset}.{split}.inter' for split in ('train', 'valid', 'test')}
    unchanged_hashes = {}
    for name, old_hash in previous_hashes.items():
        if name not in split_names:
            new_hash = sha256(args.data_dir / name)
            assert new_hash == old_hash, f'Unexpected non-sequence change: {name}'
            unchanged_hashes[name] = new_hash
    positions = defaultdict(lambda: 1)
    all_histogram, old_histogram = Counter(), Counter()
    split_stats = {}
    longer = 0
    previous_max = report['history_length_range'][1]
    for split in ('train', 'valid', 'test'):
        name = f'{dataset}.{split}.inter'
        histogram = Counter()
        with (args.data_dir / name).open(encoding='utf-8', newline='') as current, (args.baseline_dir / name).open(encoding='utf-8', newline='') as previous:
            current_rows = csv.DictReader(current, delimiter='\t')
            previous_rows = csv.DictReader(previous, delimiter='\t')
            assert current_rows.fieldnames == previous_rows.fieldnames == ['user_id:token', 'item_id_list:token_seq', 'item_id:token']
            for row_number, (row, old) in enumerate(zip_longest(current_rows, previous_rows), start=2):
                assert row is not None and old is not None, (split, row_number, 'row-count change')
                uid, target = row['user_id:token'], int(row['item_id:token'])
                assert uid == old['user_id:token'] and target == int(old['item_id:token']), (split, row_number, 'target or user changed')
                history = list(map(int, row['item_id_list:token_seq'].split()))
                old_history = list(map(int, old['item_id_list:token_seq'].split()))
                position = positions[uid]
                sequence = sequences[uid]
                assert target == sequence[position], (split, row_number, 'target position')
                assert history == sequence[max(0, position - args.max_history_len):position], (split, row_number, 'history mismatch')
                assert old_history == sequence[max(0, position - previous_max):position], (split, row_number, 'baseline mismatch')
                assert 1 <= len(history) <= args.max_history_len
                positions[uid] += 1
                histogram[len(history)] += 1
                old_histogram[len(old_history)] += 1
                longer += len(history) > len(old_history)
        assert sum(histogram.values()) == report['split_examples'][split]
        split_stats[split] = summarize(histogram)
        all_histogram.update(histogram)
    assert len(positions) == len(sequences)
    assert all(positions[uid] == len(sequence) for uid, sequence in sequences.items())
    assert sum(all_histogram.values()) == report['examples'] == report['interactions'] - report['users']
    current_stats, old_stats = summarize(all_histogram), summarize(old_histogram)
    previous_bytes = sum((args.baseline_dir / name).stat().st_size for name in split_names)
    current_bytes = sum((args.data_dir / name).stat().st_size for name in split_names)
    report['history_length_range'] = [current_stats['min'], current_stats['max']]
    report['configured_max_history_len'] = args.max_history_len
    report['history_stats'] = {**current_stats, 'by_split': split_stats}
    report['history_comparison'] = {
        'previous_max': previous_max, 'current_max': args.max_history_len,
        'previous_mean': old_stats['mean'], 'current_mean': current_stats['mean'],
        'samples_with_longer_history': longer,
        'previous_split_file_bytes': previous_bytes, 'current_split_file_bytes': current_bytes,
        'examples_unchanged': True, 'users_unchanged': True, 'items_unchanged': True,
        'interactions_unchanged': True, 'targets_and_splits_unchanged': True,
        'previous_histogram': old_stats['histogram'],
    }
    for entry in report['files']:
        entry['bytes'] = (args.data_dir / entry['name']).stat().st_size
    report['total_data_file_bytes'] = sum(entry['bytes'] for entry in report['files'])
    report['execution'] = execution
    report['history_validation'] = {
        'all_rows_checked': report['examples'],
        'checked_against_full_user_sequences': True,
        'checked_against_previous_split_rows': True,
        'unchanged_non_sample_sha256': unchanged_hashes,
        'baseline_directory': str(args.baseline_dir),
        'verifier': str(Path(__file__).resolve()),
    }
    report['validation'] = 'passed'
    destination = args.data_dir / 'cleaning_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'validation': 'passed', 'examples': report['examples'], 'history_stats': report['history_stats'],
                      'history_comparison': report['history_comparison'], 'total_data_file_bytes': report['total_data_file_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
