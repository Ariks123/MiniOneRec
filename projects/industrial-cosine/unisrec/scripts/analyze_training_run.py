#!/usr/bin/env python3
"""Compare a completed UniSRec run with fixed-Qwen CPU baselines.

This script only reads inputs and writes a new analysis. It does not train,
load an optimizer, use a GPU, or change the checkpoint/configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


METRICS = (
    'positive_cosine', 'negative_mean_cosine', 'hardest_negative_cosine',
    'margin_vs_mean_negative', 'margin_vs_hardest_negative', 'pairwise_accuracy',
    'all_catalog_vector_nll', 'positive_softmax_probability', 'alignment_squared_l2',
    'optimistic_rank', 'pessimistic_rank', 'average_rank', 'tie_expected_mrr',
    'tie_expected_recall@10', 'tie_expected_ndcg@10', 'pessimistic_recall@10',
    'tie_expected_recall@50', 'tie_expected_ndcg@50', 'pessimistic_recall@50',
)
PAIRED_METRICS = (
    'all_catalog_vector_nll', 'pairwise_accuracy', 'margin_vs_hardest_negative',
    'tie_expected_recall@10', 'tie_expected_recall@50', 'tie_expected_mrr',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized_rows(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite nonempty vector matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('A baseline has a zero vector and cannot be normalized')
    return values / norms


def score_metrics(scores, targets, temperature=.07, tie_tolerance=1e-6):
    """Same full-catalog, exact tie policy as qwen_vector_metrics, without SVD."""
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite [rows, candidates] matrix')
    if targets.shape != (len(scores),) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError('Expected one integer target per score row')
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError('Target row outside catalog')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive')
    count = scores.shape[1]
    rows = np.arange(len(scores))
    positive = scores[rows, targets].copy()
    negative_mean = (scores.sum(axis=1) - positive) / (count - 1)
    logits = scores / temperature
    maxima = logits.max(axis=1)
    nll = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1)) - positive / temperature
    difference = scores - positive[:, None]
    above = np.count_nonzero(difference > tie_tolerance, axis=1)
    tied = np.count_nonzero(np.abs(difference) <= tie_tolerance, axis=1)
    below = count - above - tied
    optimistic, pessimistic = above + 1, above + tied
    alternatives = scores.copy()
    alternatives[rows, targets] = -np.inf
    hardest = alternatives.max(axis=1)
    result = {
        'positive_cosine': positive, 'negative_mean_cosine': negative_mean,
        'hardest_negative_cosine': hardest,
        'margin_vs_mean_negative': positive - negative_mean,
        'margin_vs_hardest_negative': positive - hardest,
        'pairwise_accuracy': (below + .5 * (tied - 1)) / (count - 1),
        'all_catalog_vector_nll': nll,
        'positive_softmax_probability': np.exp(-nll),
        'alignment_squared_l2': np.maximum(0., 2. - 2. * positive),
        'optimistic_rank': optimistic, 'pessimistic_rank': pessimistic,
        'average_rank': (optimistic + pessimistic) / 2.,
    }
    harmonic = np.concatenate(([0.], np.cumsum(1. / np.arange(1, count + 1))))
    discount = np.concatenate(([0.], np.cumsum(1. / np.log2(np.arange(1, count + 1) + 1))))
    result['tie_expected_mrr'] = (harmonic[pessimistic] - harmonic[above]) / tied
    for cutoff in (10, 50):
        result[f'tie_expected_recall@{cutoff}'] = np.clip(cutoff - above, 0, tied) / tied
        last = np.maximum(above, np.minimum(pessimistic, cutoff))
        result[f'tie_expected_ndcg@{cutoff}'] = (discount[last] - discount[above]) / tied
        result[f'pessimistic_recall@{cutoff}'] = (pessimistic <= cutoff).astype(float)
    return result


def summarize(values, mask, candidate_count, tie_tolerance=1e-6):
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    result = {'count': count, 'candidate_count': candidate_count}
    for key in METRICS:
        result[key + '_mean'] = float(np.mean(values[key][mask])) if count else None
    result['vector_nll'] = result['all_catalog_vector_nll_mean']
    result['uniform_catalog_nll'] = math.log(candidate_count)
    result['nll_gain_over_uniform'] = (result['uniform_catalog_nll'] - result['vector_nll']) if count else None
    result['quantiles'] = {}
    for key in ('positive_cosine', 'margin_vs_hardest_negative', 'average_rank'):
        result['quantiles'][key] = {
            name: float(np.quantile(values[key][mask], probability)) if count else None
            for name, probability in (('p05', .05), ('p25', .25), ('p50', .5), ('p75', .75), ('p95', .95))
        }
    margin = values['margin_vs_hardest_negative'][mask]
    result['target_vs_hardest_alternative'] = {
        'strict_win_fraction': float(np.mean(margin > tie_tolerance)) if count else None,
        'tie_fraction': float(np.mean(np.abs(margin) <= tie_tolerance)) if count else None,
        'strict_loss_fraction': float(np.mean(margin < -tie_tolerance)) if count else None,
    }
    return result


def grouped_summary(values, seen_rows, candidate_count):
    import numpy as np
    summary = summarize(values, np.ones(len(seen_rows), dtype=bool), candidate_count)
    summary['groups'] = {
        'train_seen': summarize(values, seen_rows, candidate_count),
        'train_unseen': summarize(values, ~seen_rows, candidate_count),
    }
    return summary


def clustered_bootstrap(differences, clusters, *, replicates=1000, seed=42):
    """Row-weighted paired mean, resampling whole clusters with replacement."""
    import numpy as np
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters)
    if differences.ndim != 2 or len(differences) != len(clusters):
        raise ValueError('Differences and cluster IDs must have aligned rows')
    if not len(differences):
        return {'rows': 0, 'clusters': 0, 'estimate': None, 'ci_low': None, 'ci_high': None}
    if replicates < 1 or not np.isfinite(differences).all():
        raise ValueError('Finite differences and positive replicate count required')
    labels, inverse = np.unique(clusters, return_inverse=True)
    count = len(labels)
    sums = np.zeros((count, differences.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, differences)
    sizes = np.bincount(inverse)
    random = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 64):
        draw = random.integers(0, count, size=(min(64, replicates-start), count))
        samples.append(sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None])
    estimates = np.concatenate(samples)
    return {
        'rows': len(differences), 'clusters': count,
        'estimate': differences.mean(axis=0).tolist(),
        'ci_low': np.quantile(estimates, .025, axis=0).tolist(),
        'ci_high': np.quantile(estimates, .975, axis=0).tolist(),
        'replicates': replicates, 'seed': seed,
        'method': 'Percentile 95% CI; resample entire clusters, retain all rows in each selected cluster',
        'estimand': 'row-weighted paired mean delta (trained minus baseline)',
    }


def paired_report(trained, baseline, users, target_ids, seen_rows, *, replicates, seed):
    import numpy as np
    difference = np.column_stack([trained[key] - baseline[key] for key in PAIRED_METRICS])
    groups = {
        'all_user_clustered': (np.ones(len(users), dtype=bool), users, 'user'),
        'train_seen_user_clustered': (seen_rows, users, 'user'),
        'train_unseen_item_clustered': (~seen_rows, target_ids, 'target_item'),
    }
    result = {}
    for name, (mask, cluster, unit) in groups.items():
        bootstrap = clustered_bootstrap(difference[mask], cluster[mask], replicates=replicates, seed=seed)
        metric_result = {}
        for index, key in enumerate(PAIRED_METRICS):
            metric_result[key] = {
                'delta_trained_minus_baseline': bootstrap['estimate'][index] if bootstrap['estimate'] is not None else None,
                'ci95_low': bootstrap['ci_low'][index] if bootstrap['ci_low'] is not None else None,
                'ci95_high': bootstrap['ci_high'][index] if bootstrap['ci_high'] is not None else None,
                'trained_improvement_direction': 'negative' if key == 'all_catalog_vector_nll' else 'positive',
            }
        result[name] = {key: value for key, value in bootstrap.items() if key not in ('estimate', 'ci_low', 'ci_high')}
        result[name].update({'cluster_unit': unit, 'metrics': metric_result})
    return result


def load_split(path, item_rows):
    import numpy as np
    with Path(path).open(encoding='utf-8') as handle:
        original = list(csv.DictReader(handle, delimiter='\t'))
    records = []
    for row in original:
        history = row['item_id_list:token_seq'].split()
        target = row['item_id:token']
        if not history or target not in item_rows or any(item not in item_rows for item in history):
            raise ValueError('Unknown item or empty source history')
        records.append({'user_id': row['user_id:token'], 'target_item_id': target,
                        'history': np.asarray([item_rows[item] for item in history], dtype=np.int64),
                        'target_row': item_rows[target]})
    return records


def baseline_queries(records, candidate_vectors, kind):
    import numpy as np
    if kind == 'raw_qwen_last_item':
        return np.asarray([candidate_vectors[row['history'][-1]] for row in records])
    if kind == 'raw_qwen_history_mean':
        return normalized_rows(np.asarray([candidate_vectors[row['history']].mean(axis=0) for row in records]))
    raise ValueError('Unknown baseline kind')


def evaluate_baseline(records, candidates, kind, *, batch_size, temperature):
    import numpy as np
    columns = {key: [] for key in METRICS}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        queries = baseline_queries(batch, candidates, kind)
        target = np.asarray([row['target_row'] for row in batch], dtype=np.int64)
        calculated = score_metrics(queries @ candidates.T, target, temperature=temperature)
        for key in columns:
            columns[key].append(calculated[key])
    return {key: np.concatenate(chunks) for key, chunks in columns.items()}


def analyze(run_dir, *, batch_size=128, replicates=1000, seed=42, include_valid=False):
    import numpy as np
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'analysis'
    if (output/'analysis.json').exists() or (output/'per_row_baselines.npz').exists():
        raise FileExistsError('Refusing to overwrite an existing baseline analysis')
    manifests = list((run_dir/'source/prepared').glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected one immutable prepared dataset in run/source/prepared')
    prepared = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    dataset = manifest['dataset']
    feature_path = prepared/manifest['embedding']['file']
    ids_path = prepared/manifest['embedding']['row_ids']
    if sha256_file(feature_path) != manifest['embedding']['sha256']:
        raise ValueError('Raw features differ from the recorded run manifest')
    ids = [str(value) for value in json.loads(ids_path.read_text())]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate raw feature row IDs')
    candidates = normalized_rows(np.load(feature_path, mmap_mode='r', allow_pickle=False))
    if len(candidates) != len(ids):
        raise ValueError('Raw feature row IDs and shape differ')
    item_rows = {token: row for row, token in enumerate(ids)}
    test = load_split(prepared/(dataset+'.test.inter'), item_rows)
    train_ids = {str(value) for value in json.loads((prepared/'train_item_ids.json').read_text())}
    seen_rows = np.asarray([row['target_item_id'] in train_ids for row in test], dtype=bool)
    users = np.asarray([row['user_id'] for row in test])
    targets = np.asarray([row['target_item_id'] for row in test])
    trained_report = json.loads((run_dir/'test_vector_quality/report.json').read_text())
    trained_rows = [json.loads(line) for line in (run_dir/'test_vector_quality/rows.jsonl').read_text().splitlines()]
    if len(trained_rows) != len(test):
        raise ValueError('Trained row count differs from original test split')
    for index, (record, original) in enumerate(zip(trained_rows, test)):
        if not (record['row_index'] == index
                and record['target_item_id'] == original['target_item_id']
                and record['user_id'] == original['user_id']
                and record['history_length'] == len(original['history'])
                and record['target_train_seen'] == bool(seen_rows[index])):
            raise ValueError('Trained row ordering/labels differ from source at row '+str(index))
    trained = {key: np.asarray([row[key] for row in trained_rows], dtype=np.float64) for key in METRICS}
    temperature = float(trained_report['evaluation']['temperature'])
    baseline = {}
    comparisons = {}
    arrays = {'user_ids': users, 'target_item_ids': targets, 'target_train_seen': seen_rows}
    for key, values in trained.items():
        arrays['trained__'+key] = values
    for kind in ('raw_qwen_history_mean', 'raw_qwen_last_item'):
        print('Evaluating '+kind, flush=True)
        values = evaluate_baseline(test, candidates, kind, batch_size=batch_size, temperature=temperature)
        baseline[kind] = grouped_summary(values, seen_rows, len(candidates))
        comparisons[kind] = paired_report(trained, values, users, targets, seen_rows,
                                          replicates=replicates, seed=seed)
        for key, column in values.items():
            arrays[kind+'__'+key] = column
    report = {
        'status': 'completed', 'training_started': False, 'gpu_used': False,
        'run_dir': str(run_dir), 'test_rows_verified': len(test),
        'candidate_count': len(candidates), 'temperature': temperature,
        'trained': grouped_summary(trained, seen_rows, len(candidates)),
        'trained_geometry': trained_report['geometry'],
        'baselines': baseline, 'paired_comparisons': comparisons,
        'baseline_definitions': {
            'raw_qwen_history_mean': 'L2-normalize each fixed Qwen item vector; average recent source history; L2-normalize the mean',
            'raw_qwen_last_item': 'L2-normalized fixed Qwen vector of the last source history item',
            'catalog': 'All recorded raw item vectors; PAD absent; history candidates retained',
            'dimensions': {'trained': 300, 'raw_qwen': int(candidates.shape[1])},
            'arithmetic': 'float64 score calculation, tie tolerance 1e-6; same metric definitions as trained report',
        },
        'limitations': [
            'Raw and learned coordinates differ; absolute positive cosine or coordinate L2 errors must not be compared as representation accuracy.',
            'Both methods use the same observed next-item task, catalog, temperature, splits and tie policy; NLL, rank and pairwise comparisons are task-relative.',
            'Raw baselines are not temperature-tuned; ranking metrics are temperature-invariant.',
            'Bootstrap CIs describe clustered sampling uncertainty on this held-out split, not training-seed uncertainty or a new population.',
            'Cold group has few distinct target items; its item-cluster bootstrap is descriptive and does not resolve provenance concerns.',
            'Original raw-Qwen model identity and row-order provenance remain unverified as recorded in the run manifest.',
        ],
        'source_sha256': {'raw_embeddings': sha256_file(feature_path),
                          'embedding_ids': sha256_file(ids_path),
                          'trained_row_report': sha256_file(run_dir/'test_vector_quality/rows.jsonl'),
                          'script': sha256_file(__file__)},
    }
    if include_valid:
        print('Evaluating raw mean on validation split', flush=True)
        valid = load_split(prepared/(dataset+'.valid.inter'), item_rows)
        valid_seen = np.asarray([row['target_item_id'] in train_ids for row in valid], dtype=bool)
        valid_values = evaluate_baseline(valid, candidates, 'raw_qwen_history_mean',
                                        batch_size=batch_size, temperature=temperature)
        report['validation_raw_qwen_history_mean'] = grouped_summary(valid_values, valid_seen, len(candidates))
        for key, column in valid_values.items():
            arrays['valid_raw_qwen_history_mean__'+key] = column
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.analysis-', dir=run_dir) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging/'per_row_baselines.npz', **arrays)
        (staging/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        if (output/'analysis.json').exists() or (output/'per_row_baselines.npz').exists():
            raise FileExistsError('Analysis was published while this process was running')
        os.replace(staging/'per_row_baselines.npz', output/'per_row_baselines.npz')
        os.replace(staging/'analysis.json', output/'analysis.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-valid', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0:
        parser.error('Batch size and bootstrap replicates must be positive')
    report = analyze(args.run_dir, batch_size=args.batch_size,
                     replicates=args.bootstrap_replicates, seed=args.seed,
                     include_valid=args.include_valid)
    print(json.dumps({'status': report['status'], 'output': str(args.run_dir/'analysis/analysis.json'),
                      'trained_vector_nll': report['trained']['vector_nll'],
                      'raw_mean_vector_nll': report['baselines']['raw_qwen_history_mean']['vector_nll']}, indent=2))


if __name__ == '__main__':
    main()
