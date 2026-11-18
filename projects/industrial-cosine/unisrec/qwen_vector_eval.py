"""Evaluate learned query/target vector quality from a trained UniSRec checkpoint.

Only inference occurs here. Coordinate-wise errors against the original 2560-D
features are not meaningful for the learned 300-D retrieval space.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


def _json_value(value):
    """Convert NumPy results without allowing non-finite JSON numbers."""
    if hasattr(value, 'tolist'):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _dataset_tokens(dataset, field, ids):
    mapping = getattr(dataset, 'field2id_token', {}).get(field)
    if mapping is None:
        return [str(value) for value in ids]
    return [str(mapping[int(value)]) for value in ids]


def evaluate_model_vectors(model, eval_dataset, *, temperature,
                           topk=(10, 20, 50), batch_size=128,
                           item_batch_size=512, include_vectors=False):
    """Measure all-catalog vector matching on one complete evaluation dataset.

    This lower-level API also accepts synthetic models in tests. It deliberately
    does not label its output as a trained-model result; the checkpoint CLI below
    supplies that provenance gate. ``per_row`` is a dictionary of aligned NumPy
    columns, with original example order retained. PAD is never a candidate.
    """
    import numpy as np
    import torch
    from qwen_vector_metrics import evaluate_vectors

    if batch_size <= 0 or item_batch_size <= 0:
        raise ValueError('batch_size and item_batch_size must be positive')
    row_count = len(eval_dataset)
    if row_count == 0:
        raise ValueError('Vector evaluation requires a nonempty held-out split')
    item_count = int(eval_dataset.item_num) - 1
    if item_count < 2:
        raise ValueError('Vector evaluation requires at least two non-padding items')
    interaction = eval_dataset.inter_feat
    target_ids = interaction[model.POS_ITEM_ID].detach().cpu().numpy().astype(np.int64)
    history_lengths = interaction[model.ITEM_SEQ_LEN].detach().cpu().numpy().astype(np.int64)
    if target_ids.shape != (row_count,) or np.any(target_ids <= 0) or np.any(target_ids > item_count):
        raise ValueError('Evaluation targets must be valid non-padding internal item IDs')
    train_ids = getattr(eval_dataset, 'train_item_ids', None)
    if train_ids is None:
        raise ValueError('Dataset train_item_ids is required for genuine cold-item groups')
    if hasattr(train_ids, 'detach'):
        train_ids = train_ids.detach().cpu().numpy()
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if train_ids.ndim != 1 or np.any(train_ids <= 0) or np.any(train_ids > item_count):
        raise ValueError('Training item IDs must exclude PAD and fit the catalog')
    seen_mask = np.zeros(item_count, dtype=bool)
    seen_mask[train_ids - 1] = True
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    was_training = model.training
    candidate_chunks, query_chunks = [], []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(1, item_count + 1, item_batch_size):
                ids = torch.arange(start, min(item_count + 1, start + item_batch_size), device=device)
                encoded = model.encode_items(ids)
                candidate_chunks.append(encoded.detach().cpu().float().numpy())
            for start in range(0, row_count, batch_size):
                batch = interaction[start:start + batch_size]
                encoded = model.encode_history(batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
                query_chunks.append(encoded.detach().cpu().float().numpy())
    finally:
        model.train(was_training)
    candidates = np.concatenate(candidate_chunks, axis=0)
    queries = np.concatenate(query_chunks, axis=0)
    target_indices = target_ids - 1
    report = evaluate_vectors(queries, candidates, target_indices,
                              temperature=temperature, train_seen_mask=seen_mask,
                              topk=tuple(topk))
    per_row = report['per_row']
    per_row['row_index'] = np.arange(row_count, dtype=np.int64)
    per_row['target_internal_id'] = target_ids
    per_row['target_train_seen'] = seen_mask[target_indices]
    per_row['history_length'] = history_lengths
    item_field = getattr(eval_dataset, 'iid_field', 'item_id')
    per_row['target_item_id'] = np.asarray(_dataset_tokens(eval_dataset, item_field, target_ids))
    user_field = getattr(eval_dataset, 'uid_field', 'user_id')
    if user_field in interaction:
        users = interaction[user_field].detach().cpu().numpy()
        per_row['user_id'] = np.asarray(_dataset_tokens(eval_dataset, user_field, users))
    report['evaluation'] = {
        'row_count': row_count, 'candidate_count': item_count,
        'candidate_internal_ids': '1..item_num-1; PAD excluded',
        'temperature': float(temperature), 'topk': list(topk),
        'history_candidates_excluded': False,
        'cold_definition': 'Target item absent from all training histories and targets',
        'model_training_mode_restored': bool(model.training == was_training),
        'training_started': False, 'optimizer_created': False, 'backward_called': False,
        'checkpoint_verified': False,
        'true_target_vector': 'L2-normalized learned adaptor(item_feature), not original Qwen coordinates',
        'interpretation': (
            'Query and target share the learned item-retrieval space. '
            'Other catalog items are unobserved alternatives, not verified dislikes. '
            'Absolute cosine alone is insufficient; inspect margins, NLL and collapse diagnostics.'),
    }
    if include_vectors:
        report['vectors'] = {'queries': queries, 'targets': candidates[target_indices]}
    return report


def _row_records(columns):
    if not columns:
        return
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError('Per-row vector metric columns have inconsistent lengths')
    for index in range(next(iter(lengths))):
        yield {key: _json_value(values[index]) for key, values in columns.items()}


def evaluate_checkpoint(config_path, checkpoint_path, output_dir, *, split='valid',
                        device='cpu', batch_size=128, item_batch_size=512,
                        save_vectors=False):
    """Write held-out quality metrics only after validating a trained checkpoint."""
    if split not in ('valid', 'test'):
        raise ValueError('Only valid or test splits may be evaluated')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Vector quality evaluation uses one complete split; do not launch with torchrun')
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Trained checkpoint does not exist: {checkpoint_path}')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite vector evaluation output: {output_dir}')

    from qwen_runtime import ROOT, load_experiment, load_trained_checkpoint, sha256_file
    from qwen_retrieval import _assert_training_checkpoint, read_catalog
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    config, _, loaders, model = load_experiment(path, device=device)
    checkpoint = load_trained_checkpoint(model, checkpoint_path)
    _assert_training_checkpoint(checkpoint)
    eval_dataset = loaders[1 if split == 'valid' else 2].dataset
    report = evaluate_model_vectors(
        model, eval_dataset, temperature=float(config['temperature']),
        topk=tuple(config['topk']), batch_size=batch_size,
        item_batch_size=item_batch_size, include_vectors=save_vectors)
    catalog_path = Path(config['catalog_path']) if config['catalog_path'] else Path(config['data_path']) / 'catalog.json'
    catalog = {row['item_id']: row for row in read_catalog(catalog_path)}
    row_columns = report.pop('per_row')
    missing = set(str(item) for item in row_columns['target_item_id']) - set(catalog)
    if missing:
        raise ValueError(f'Evaluation targets are absent from the SID catalog: {sorted(missing)[:8]}')
    vectors = report.pop('vectors', None)
    report['evaluation'].update({
        'split': split, 'checkpoint_verified': True, 'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_sha256': sha256_file(checkpoint_path),
        'config': str(path), 'config_sha256': sha256_file(path),
        'catalog_sha256': sha256_file(catalog_path),
        'qwen_provenance': model.qwen_provenance,
        'saved_vectors': bool(save_vectors),
        'sid_note': 'Metrics use item IDs; original SID collisions are retained in row records.',
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}-', dir=output_dir.parent))
    try:
        with (staging / 'rows.jsonl').open('w', encoding='utf-8') as handle:
            for row in _row_records(row_columns):
                item = catalog[str(row['target_item_id'])]
                row['target_sid'] = item['sid']
                row['target_sid_tokens'] = item['sid_tokens']
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        (staging / 'report.json').write_text(
            json.dumps(_json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n',
            encoding='utf-8')
        if vectors is not None:
            import numpy as np
            np.save(staging / 'query_vectors.npy', vectors['queries'], allow_pickle=False)
            np.save(staging / 'target_item_vectors.npy', vectors['targets'], allow_pickle=False)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='props/qwen_industrial_4gpu.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--item-batch-size', type=int, default=512)
    parser.add_argument('--save-vectors', action='store_true')
    args = parser.parse_args(argv)
    output = evaluate_checkpoint(
        args.config, args.checkpoint, args.output, split=args.split,
        device=args.device, batch_size=args.batch_size,
        item_batch_size=args.item_batch_size, save_vectors=args.save_vectors)
    print(json.dumps({'output_dir': str(output), 'training_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
