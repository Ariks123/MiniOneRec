#!/usr/bin/env python3
"""Summarize a completed retained run and its predecessors without training."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def result_of(run):
    paths = list((run/'checkpoints').glob('*.result.json'))
    if len(paths) != 1:
        raise ValueError('Expected one completed result JSON: ' + str(run))
    return read(paths[0])


def experiment_row(run):
    spec = read(run/'run_spec.json')
    result = result_of(run)
    records = [json.loads(line) for path in (run/'checkpoints').glob('*.vector_quality.jsonl')
               for line in path.read_text().splitlines()]
    valid = [row for row in records if not row['checkpoint_loaded']]
    best = min(valid, key=lambda row: row['summary']['vector_nll'])
    test = result['vector_quality']['summary']
    if abs(best['summary']['vector_nll'] - result['best_valid_score']) > 1e-8:
        raise ValueError('Best validation loss does not match the saved result')
    return {
        'run_dir': str(run), 'max_epochs': spec['configuration']['epochs'],
        'completed_epochs': len(valid), 'best_epoch': best['epoch_index'] + 1,
        'best_validation_loss': result['best_valid_score'], 'test_loss': test['vector_nll'],
        'test_rows': test['count'], 'candidate_count': test['candidate_count'],
        'test_cold_rows': test['groups']['train_unseen']['count'],
        'evaluation_temperature': result['qwen_provenance']['architecture']['temperature'],
        'embedding_sha256': result['qwen_provenance']['embedding_sha256'],
        'test_positive_cosine': test['positive_cosine_mean'],
        'test_pairwise_accuracy': test['pairwise_accuracy_mean'],
        'test_hardest_margin': test['margin_vs_hardest_negative_mean'],
        'test_recall10': test['tie_expected_recall@10_mean'],
        'test_recall20': test['tie_expected_recall@20_mean'],
        'test_recall50': test['tie_expected_recall@50_mean'],
        'test_cold_loss': test['groups']['train_unseen']['vector_nll'],
        'test_cold_recall10': test['groups']['train_unseen']['tie_expected_recall@10_mean'],
        'test_cold_recall20': test['groups']['train_unseen']['tie_expected_recall@20_mean'],
        'test_cold_recall50': test['groups']['train_unseen']['tie_expected_recall@50_mean'],
    }


def read_npz_column(archive, name):
    """Read a one-dimensional NumPy column using only the standard library."""
    with archive.open(name + '.npy') as handle:
        if handle.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid NumPy magic: ' + name)
        version = handle.read(2)
        if len(version) != 2 or version[0] not in (1, 2, 3):
            raise ValueError('Unsupported NumPy format: ' + name)
        length_size = 2 if version[0] == 1 else 4
        header_length = struct.unpack('<H' if length_size == 2 else '<I',
                                      handle.read(length_size))[0]
        header = ast.literal_eval(handle.read(header_length).decode(
            'utf-8' if version[0] == 3 else 'latin1'))
        shape = header['shape']
        if len(shape) != 1 or header['fortran_order']:
            raise ValueError('Expected a contiguous one-dimensional column: ' + name)
        descriptor = header['descr']
        if not isinstance(descriptor, str) or descriptor[0] not in '<>=|':
            raise ValueError('Unsupported NumPy dtype: ' + name)
        formats = {'i8': 'q', 'u8': 'Q', 'f8': 'd', 'i4': 'i', 'u4': 'I',
                   'f4': 'f', 'b1': '?', 'u1': 'B', 'i1': 'b'}
        if descriptor[1:] not in formats:
            raise ValueError('Unsupported NumPy column dtype: ' + descriptor)
        endian = descriptor[0] if descriptor[0] in '<>=' else '<'
        fmt = endian + formats[descriptor[1:]]
        payload = handle.read()
        if len(payload) != shape[0] * struct.calcsize(fmt):
            raise ValueError('NumPy column payload length differs from its shape: ' + name)
        return [value[0] for value in struct.iter_unpack(fmt, payload)]


def recall_from_rank_bounds(optimistic, pessimistic, cutoff, mask=None):
    """Expected Recall@K under the existing uniform ordering of target ties."""
    if len(optimistic) != len(pessimistic) or (mask is not None and len(mask) != len(optimistic)):
        raise ValueError('Baseline rank columns and mask have inconsistent lengths')
    total, count = 0.0, 0
    for index, (lower, upper) in enumerate(zip(optimistic, pessimistic)):
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError('Baseline rank bounds must be finite')
        if lower != int(lower) or upper != int(upper) or lower < 1 or upper < lower:
            raise ValueError('Baseline rank bounds must be positive ordered integers')
        if mask is not None and not mask[index]:
            continue
        tied = upper - lower + 1
        total += max(0, min(cutoff - (lower - 1), tied)) / tied
        count += 1
    return total / count if count else None


def overall_metrics_comparison(comparisons):
    """Combine learned runs and immutable raw-Qwen baselines without rewriting either."""
    if not comparisons:
        raise ValueError('At least one completed learned run is required')
    original = Path(comparisons[-1]['run_dir'])
    analysis_path = original/'analysis/analysis.json'
    arrays_path = original/'analysis/per_row_baselines.npz'
    analysis = read(analysis_path)
    reference = comparisons[0]
    for row in comparisons:
        for key in ('test_rows', 'test_cold_rows', 'candidate_count',
                    'evaluation_temperature', 'embedding_sha256'):
            if row[key] != reference[key]:
                raise ValueError('Learned runs differ in comparison scope: ' + key)
    if (analysis['test_rows_verified'] != reference['test_rows']
            or analysis['candidate_count'] != reference['candidate_count']
            or analysis['temperature'] != reference['evaluation_temperature']
            or analysis['source_sha256']['raw_embeddings'] != reference['embedding_sha256']):
        raise ValueError('Raw baseline and learned runs use different data or evaluation settings')
    methods = [{'method': 'UniSRec max_epochs=' + str(row['max_epochs']), **row}
               for row in comparisons]
    with zipfile.ZipFile(arrays_path) as archive:
        seen = read_npz_column(archive, 'target_train_seen')
        if len(seen) != reference['test_rows'] or not all(isinstance(value, bool) for value in seen):
            raise ValueError('Baseline training-seen mask is not a matching boolean column')
        cold = [not value for value in seen]
        if sum(cold) != reference['test_cold_rows']:
            raise ValueError('Baseline cold-item group differs from learned runs')
        for name in ('raw_qwen_history_mean', 'raw_qwen_last_item'):
            quality = analysis['baselines'][name]
            cold_quality = quality['groups']['train_unseen']
            lower = read_npz_column(archive, name + '__optimistic_rank')
            upper = read_npz_column(archive, name + '__pessimistic_rank')
            if len(lower) != reference['test_rows']:
                raise ValueError('Baseline row count differs from learned runs')
            row = {'method': name, 'max_epochs': None, 'completed_epochs': None,
                   'best_epoch': None, 'test_rows': quality['count'],
                   'test_cold_rows': cold_quality['count'],
                   'candidate_count': quality['candidate_count'],
                   'evaluation_temperature': analysis['temperature'],
                   'test_loss': quality['vector_nll'], 'test_cold_loss': cold_quality['vector_nll']}
            for cutoff in (10, 20, 50):
                row['test_recall' + str(cutoff)] = recall_from_rank_bounds(lower, upper, cutoff)
                row['test_cold_recall' + str(cutoff)] = recall_from_rank_bounds(lower, upper, cutoff, cold)
                if cutoff in (10, 50):
                    expected = quality['tie_expected_recall@' + str(cutoff) + '_mean']
                    cold_expected = cold_quality['tie_expected_recall@' + str(cutoff) + '_mean']
                    if (abs(row['test_recall' + str(cutoff)] - expected) > 1e-12
                            or abs(row['test_cold_recall' + str(cutoff)] - cold_expected) > 1e-12):
                        raise ValueError('Baseline rank calculation differs from existing metrics: ' + name)
            methods.append(row)
    return {
        'test_rows': reference['test_rows'], 'cold_rows': reference['test_cold_rows'],
        'candidate_count': reference['candidate_count'],
        'temperature': reference['evaluation_temperature'],
        'tie_policy': 'Uniform expected ordering of target score ties, tolerance 1e-6',
        'selection': 'Each learned run uses its minimum validation cross-entropy loss checkpoint',
        'baseline_source_run_dir': str(original),
        'baseline_source_sha256': {
            'analysis_json': hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
            'per_row_arrays': hashlib.sha256(arrays_path.read_bytes()).hexdigest()},
        'methods': methods,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    state = read(run/'run_status.json')
    if state['status'] != 'completed' or state['exit_code'] != 0:
        raise ValueError('Training is not completed and exported successfully')
    spec, result = read(run/'run_spec.json'), result_of(run)
    current = experiment_row(run)
    comparisons, seen, cursor = [], set(), run
    while str(cursor) not in seen:
        seen.add(str(cursor))
        comparisons.append(experiment_row(cursor))
        previous = read(cursor/'run_spec.json').get('previous_run_dir')
        if not previous:
            break
        cursor = Path(previous)
    overall = overall_metrics_comparison(comparisons)
    summary = {
        'status': 'completed', **current,
        'early_stopping_metric': 'validation full-catalog cross-entropy loss (=vector_nll)',
        'early_stopping_direction': 'minimize',
        'early_stopping_patience_setting': spec['configuration']['stopping_step'],
        'test_vector_quality': result['vector_quality']['summary'],
        'test_geometry': result['vector_quality']['geometry'],
        'checkpoint': result['checkpoint'], 'initialization': spec['initialization'],
        'config': spec['configuration'],
        'comparison_previous_runs': comparisons,
    }
    offline = run/'test_vector_quality/report.json'
    if offline.exists():
        report = read(offline)
        if abs(report['summary']['vector_nll'] - current['test_loss']) > 1e-4:
            raise ValueError('CPU/GPU test loss mismatch')
        if report['evaluation']['checkpoint_epoch'] + 1 != current['best_epoch']:
            raise ValueError('Offline report used a different checkpoint epoch')
        summary.update({'offline_cpu_evaluation_consistent': True,
                        'vector_quality_report': str(offline)})
    (run/'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (run/'comparison_previous_runs.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (run/'overall_metrics_comparison.json').write_text(json.dumps(overall, indent=2, allow_nan=False) + '\n')
    project = Path(__file__).resolve().parents[1]
    active_path = project/'artifacts/active_run.json'
    active = read(active_path)
    if active['run_dir'] == str(run):
        active.update({'training_status': 'completed',
                       'completed_epochs': current['completed_epochs'],
                       'best_epoch': current['best_epoch'],
                       'best_validation_loss': current['best_validation_loss'],
                       'test_loss': current['test_loss'],
                       'summary_file': str(run/'training_summary.json'),
                       'comparison_file': str(run/'comparison_previous_runs.json'),
                       'overall_metrics_comparison_file': str(run/'overall_metrics_comparison.json')})
        if offline.exists():
            active['vector_quality_report'] = str(offline)
        active_path.write_text(json.dumps(active, indent=2) + '\n')
        history_path = project/'artifacts/run_history.json'
        history = read(history_path)
        for index, item in enumerate(history):
            if item['run_dir'] == str(run):
                history[index] = active
        history_path.write_text(json.dumps(history, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
