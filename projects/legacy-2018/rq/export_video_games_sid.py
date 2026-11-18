#!/usr/bin/env python3
"""Export raw, repaired, and collision-free Video Games SIDs from one RQ-VAE run."""
from pathlib import Path
from datetime import datetime, timezone
from argparse import Namespace
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/python_deps'))
sys.path.append(str(ROOT))
import npu_support  # noqa: F401 - preserve the project's device shim

PREFIXES = ('<a_{}>', '<b_{}>', '<c_{}>')
MAX_ROUNDS = 20
REPAIR_EPSILONS = [0.0, 0.0, 0.003]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_nonfinite_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_nonfinite_constant)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(value)
    os.replace(temporary, path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n').encode('utf-8')


def atomic_array(path, value):
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def artifact_path(manifest, key):
    value = manifest['artifacts'][key]
    require(isinstance(value, str), f'Artifact {key} must be an absolute path string')
    path = Path(value)
    require(path.is_absolute(), f'Artifact {key} must be an absolute path: {path}')
    return path.resolve(strict=True)


def validate_inputs(embedding_path, metadata_path, mapping_path, split_path):
    embeddings = np.load(embedding_path, mmap_mode='r', allow_pickle=False)
    require(embeddings.ndim == 2 and all(embeddings.shape), 'Expected a nonempty two-dimensional embedding matrix')
    require(embeddings.dtype == np.dtype('float32'), 'Expected the verified float32 embedding matrix')
    for start in range(0, len(embeddings), 512):
        block = embeddings[start:start + 512]
        require(np.isfinite(block).all(), f'Embedding NaN/Inf beginning at row {start}')
        require((np.linalg.norm(block.astype(np.float64), axis=1) > 0).all(), f'Zero embedding beginning at row {start}')
    n = len(embeddings)
    metadata = read_json(metadata_path)
    require(isinstance(metadata, dict) and set(metadata) == {str(i) for i in range(n)}, 'Metadata IDs must exactly match embedding rows 0..N-1')
    asin_by_id = {}
    seen_asins = set()
    for number, row in enumerate(mapping_path.read_text(encoding='utf-8-sig').splitlines(), start=1):
        fields = row.split('\t')
        require(len(fields) == 2 and bool(fields[0]), f'Malformed item2id row {number}')
        asin, item = fields[0], int(fields[1])
        require(item not in asin_by_id and asin not in seen_asins, f'Duplicate item ID or ASIN at row {number}')
        asin_by_id[item] = asin
        seen_asins.add(asin)
    require(set(asin_by_id) == set(range(n)), 'ASIN mapping must cover exactly 0..N-1')
    if split_path.suffix.casefold() == '.json':
        read_json(split_path)
    elif split_path.suffix.casefold() == '.npz':
        with np.load(split_path, allow_pickle=False) as split:
            for key in split.files:
                value = split[key]
                require(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(), f'Invalid split array {key}')
    return embeddings, metadata, asin_by_id


def code_statistics(codes):
    require(codes.ndim == 2 and len(codes) > 0, 'Codes must be nonempty and two-dimensional')
    counts = Counter(tuple(map(int, row)) for row in codes)
    n, unique = len(codes), len(counts)
    return {'items': n, 'unique_sids': unique, 'collision_groups': sum(count > 1 for count in counts.values()),
            'items_in_collision_groups': sum(count for count in counts.values() if count > 1),
            'excess_items': n - unique, 'collision_rate': (n - unique) / n,
            'max_collision_group_size': max(counts.values())}


def collision_groups(codes):
    groups = defaultdict(list)
    for item, row in enumerate(codes):
        groups[tuple(map(int, row))].append(item)
    return [items for items in groups.values() if len(items) > 1]


def validate_codes(codes, n, sizes):
    require(codes.shape == (n, len(sizes)) and np.issubdtype(codes.dtype, np.integer), 'Code shape or integer dtype mismatch')
    for level, size in enumerate(sizes):
        require(((codes[:, level] >= 0) & (codes[:, level] < size)).all(), f'Out-of-range code at level {level}')


def model_codes(model, batch, use_sk):
    require(torch.isfinite(batch).all().item(), 'Non-finite model input')
    with torch.no_grad():
        codes = model.get_indices(batch, use_sk=use_sk)
    require(torch.isfinite(codes).all().item(), 'Non-finite model codes')
    return codes.reshape(-1, codes.shape[-1]).detach().cpu().numpy().astype(np.int64, copy=False)


def encode_raw(model, embeddings, device, batch_size, sizes):
    result = []
    for start in range(0, len(embeddings), batch_size):
        batch = torch.from_numpy(np.array(embeddings[start:start + batch_size], copy=True)).to(device)
        result.append(model_codes(model, batch, use_sk=False))
    codes = np.concatenate(result, axis=0)
    validate_codes(codes, len(embeddings), sizes)
    return codes


def repair_semantic(model, embeddings, raw_codes, device, sizes):
    for quantizer, epsilon in zip(model.rq.vq_layers, REPAIR_EPSILONS):
        quantizer.sk_epsilon = epsilon
    codes = raw_codes.copy()
    trajectory = []
    rounds = 0
    while rounds < MAX_ROUNDS:
        groups = collision_groups(codes)
        if not groups:
            break
        before = codes.copy()
        print(f'Sinkhorn collision round {rounds + 1}: {len(groups)} groups', flush=True)
        for item_ids in groups:
            batch = torch.from_numpy(np.array(embeddings[item_ids], copy=True)).to(device)
            changed = model_codes(model, batch, use_sk=True)
            validate_codes(changed, len(item_ids), sizes)
            codes[item_ids] = changed
        rounds += 1
        trajectory.append({'round': rounds, 'items_changed_this_round': int(np.any(codes != before, axis=1).sum()),
                           **code_statistics(codes)})
    validate_codes(codes, len(embeddings), sizes)
    return codes, rounds, trajectory


def add_identity_suffix(semantic_codes, asin_by_id):
    groups = defaultdict(list)
    for item, code in enumerate(semantic_codes):
        groups[tuple(map(int, code))].append(item)
    ranks = np.zeros(len(semantic_codes), dtype=np.int64)
    for items in groups.values():
        for rank, item in enumerate(sorted(items, key=lambda item: asin_by_id[item])):
            ranks[item] = rank
    final = np.column_stack([semantic_codes, ranks])
    require(code_statistics(final)['unique_sids'] == len(final), 'Identity suffix did not produce unique SIDs')
    require(np.array_equal(final[:, :3], semantic_codes), 'Identity suffix changed semantic codes')
    return final


def sid_dictionary(codes):
    templates = (*PREFIXES, '<d_{}>')
    require(codes.shape[1] in (3, 4), 'SID must have three semantic tokens or four final tokens')
    return {str(item): [templates[level].format(int(value)) for level, value in enumerate(row)]
            for item, row in enumerate(codes)}


def collision_csv(raw, semantic, final, metadata, asin_by_id):
    raw_counts = Counter(tuple(map(int, row)) for row in raw)
    semantic_counts = Counter(tuple(map(int, row)) for row in semantic)
    stream = io.StringIO(newline='')
    fields = ['item_id', 'asin', 'title', 'raw_group_size', 'semantic_group_size', 'raw_sid', 'semantic_sid', 'final_sid', 'd_rank', 'semantic_code_changed']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    raw_dict, semantic_dict, final_dict = sid_dictionary(raw), sid_dictionary(semantic), sid_dictionary(final)
    for item in range(len(raw)):
        raw_size, semantic_size = raw_counts[tuple(map(int, raw[item]))], semantic_counts[tuple(map(int, semantic[item]))]
        if raw_size == 1 and semantic_size == 1:
            continue
        writer.writerow({'item_id': item, 'asin': asin_by_id[item], 'title': metadata[str(item)].get('title', ''),
                         'raw_group_size': raw_size, 'semantic_group_size': semantic_size,
                         'raw_sid': ''.join(raw_dict[str(item)]), 'semantic_sid': ''.join(semantic_dict[str(item)]),
                         'final_sid': ''.join(final_dict[str(item)]), 'd_rank': int(final[item, 3]),
                         'semantic_code_changed': bool(np.any(raw[item] != semantic[item]))})
    return stream.getvalue().encode('utf-8-sig')


def sync_catalog(run_dir, catalog_dir, payloads, dataset='Video_Games'):
    marker = catalog_dir / f'{dataset}.sid_export_origin.json'
    previous = read_json(marker) if marker.exists() else None
    same_run = previous is not None and Path(previous['run_directory']).resolve() == run_dir
    for name, payload in payloads.items():
        target = catalog_dir / name
        if target.exists() and target.read_bytes() != payload:
            require(same_run, f'Refusing to overwrite SID from another or unknown run: {target}')
    for name, payload in payloads.items():
        target = catalog_dir / name
        if not target.exists() or target.read_bytes() != payload:
            atomic_bytes(target, payload)
    atomic_bytes(marker, json_bytes({'run_directory': str(run_dir), 'synced_at_utc': datetime.now(timezone.utc).isoformat(),
                                    'files': {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                                              for name, payload in payloads.items()}}))
    return {name: str(catalog_dir / name) for name in payloads}


def validate_checkpoint_role(manifest, checkpoint_path, checkpoint, role):
    if role == 'validation_best':
        require(checkpoint_path.name == 'best_val_model.pth',
                'validation_best must select best_val_model.pth')
        return
    require(role == 'fixed_epoch', f'Unknown checkpoint role: {role}')
    expected_epoch = manifest.get('evaluation_checkpoint_epoch')
    require(type(expected_epoch) is int and expected_epoch > 0,
            'fixed_epoch requires a positive integer manifest.evaluation_checkpoint_epoch')
    actual_epoch = checkpoint.get('epoch')
    require(type(actual_epoch) is int and actual_epoch == expected_epoch,
            f'Checkpoint epoch {actual_epoch!r} does not match evaluation_checkpoint_epoch {expected_epoch}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=('Video_Games', 'Industrial_and_Scientific'), default=None,
                        help='Defaults to manifest category or the legacy Video_Games catalog')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--checkpoint-role', choices=('validation_best', 'fixed_epoch'), default='validation_best')
    parser.add_argument('--skip-catalog-sync', action='store_true', help='write SID artifacts only to run-dir')
    parser.add_argument('--output-dir',type=Path,default=None,help='Isolated SID catalog destination; required for syncing a 512-codebook export')
    cli = parser.parse_args()
    require(cli.batch_size > 0, 'batch-size must be positive')
    run_dir = cli.run_dir.resolve(strict=True)
    manifest_path = run_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    inputs = {key: artifact_path(manifest, key) for key in ('checkpoint', 'embeddings', 'item_metadata', 'item2id', 'split')}
    file_category = inputs['item_metadata'].name.removesuffix('.item.json')
    inferred = manifest.get('category') or manifest.get('config', {}).get('dataset') or (file_category if file_category in ('Video_Games', 'Industrial_and_Scientific') else None)
    dataset = cli.dataset or inferred or 'Video_Games'
    require(dataset in ('Video_Games', 'Industrial_and_Scientific') and (inferred is None or inferred == dataset),
            'Requested SID dataset differs from the manifest/catalog identity')
    before_sources = {key: source_record(path) for key, path in inputs.items()}
    embeddings, metadata, asin_by_id = validate_inputs(inputs['embeddings'], inputs['item_metadata'], inputs['item2id'], inputs['split'])
    checkpoint = torch.load(inputs['checkpoint'], map_location='cpu', weights_only=False)
    validate_checkpoint_role(manifest, inputs['checkpoint'], checkpoint, cli.checkpoint_role)
    args = Namespace(**checkpoint['args']) if isinstance(checkpoint['args'], dict) else checkpoint['args']
    sizes = list(args.num_emb_list)
    require(len(sizes) == 3 and all(type(size) is int and size > 0 for size in sizes), 'Expected exactly three positive codebook sizes')
    catalog_dir=cli.output_dir.resolve() if cli.output_dir else inputs['item_metadata'].parent
    if sizes==[512]*3 and not cli.skip_catalog_sync:
        require(cli.output_dir is not None and catalog_dir!=inputs['item_metadata'].parent.resolve(),
                '512-codebook SID export requires an isolated --output-dir or --skip-catalog-sync')
    require(len(args.sk_epsilons) == 3 and np.isfinite(args.sk_epsilons).all(), 'Invalid checkpoint Sinkhorn configuration')
    for name, tensor in checkpoint['state_dict'].items():
        if torch.is_tensor(tensor):
            require(torch.isfinite(tensor).all().item(), f'Non-finite checkpoint tensor {name}')
    from models.rqvae import RQVAE
    import models.vq as vq_module
    original_sinkhorn = vq_module.sinkhorn_algorithm
    def checked_sinkhorn(distances, epsilon, iterations):
        require(torch.isfinite(distances).all().item(), 'Non-finite Sinkhorn distances')
        result = original_sinkhorn(distances, epsilon, iterations)
        require(torch.isfinite(result).all().item(), 'Non-finite Sinkhorn transport matrix')
        return result
    vq_module.sinkhorn_algorithm = checked_sinkhorn
    model = RQVAE(in_dim=embeddings.shape[1], num_emb_list=sizes, e_dim=args.e_dim,
                  layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn, loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight, beta=getattr(args, 'beta', 0.25),
                  kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for quantizer in model.rq.vq_layers:
        quantizer.initted = True
    device = torch.device(cli.device)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'Requested CUDA device is unavailable')
    model = model.to(device).eval()
    raw = encode_raw(model, embeddings, device, cli.batch_size, sizes)
    semantic, rounds, trajectory = repair_semantic(model, embeddings, raw, device, sizes)
    final = add_identity_suffix(semantic, asin_by_id)
    names = {'raw_codes': 'raw_codes.npy', 'semantic_codes': 'semantic_codes.npy',
             'raw_sid': f'{dataset}.raw.index.json', 'semantic_sid': f'{dataset}.semantic.index.json',
             'final_sid': f'{dataset}.index.json', 'export_stats': 'export_stats.json', 'collision_items': 'collision_items.csv'}
    outputs = {key: run_dir / value for key, value in names.items()}
    require(not set(outputs.values()) & set(inputs.values()), 'Export output would overwrite an input artifact')
    atomic_array(outputs['raw_codes'], raw)
    atomic_array(outputs['semantic_codes'], semantic)
    sid_payloads = {names[key]: json_bytes(sid_dictionary(codes))
                    for key, codes in [('raw_sid', raw), ('semantic_sid', semantic), ('final_sid', final)]}
    for name, payload in sid_payloads.items():
        atomic_bytes(run_dir / name, payload)
    atomic_bytes(outputs['collision_items'], collision_csv(raw, semantic, final, metadata, asin_by_id))
    for key, path in inputs.items():
        require(source_record(path) == before_sources[key], f'Input artifact changed during export: {key}')
    catalog_dir.mkdir(parents=True,exist_ok=True)
    catalog_sync = ({'status': 'skipped', 'reason': '--skip-catalog-sync',
                     'catalog_directory': str(inputs['item_metadata'].parent)}
                    if cli.skip_catalog_sync else sync_catalog(run_dir, catalog_dir, sid_payloads, dataset))
    stats = {'algorithm': 'original_rqvae_nearest_then_last_level_sinkhorn_then_asin_rank', 'category': dataset,
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_directory': str(run_dir),
             'checkpoint_role': cli.checkpoint_role, 'checkpoint_epoch': checkpoint.get('epoch'),
             'items': len(embeddings), 'embedding_dimensions': embeddings.shape[1], 'codebook_sizes': sizes,
             'raw': code_statistics(raw), 'semantic': code_statistics(semantic), 'final': code_statistics(final),
             'collision_rate_definition': '(items - unique_sids) / items',
             'code_changes': {'items_raw_to_semantic': int(np.any(raw != semantic, axis=1).sum()),
                              'tokens_raw_to_semantic': int((raw != semantic).sum()),
                              'by_level_raw_to_semantic': (raw != semantic).sum(axis=0).tolist(),
                              'identity_suffix_max_rank': int(final[:, 3].max()),
                              'identity_suffix_nonzero_items': int((final[:, 3] > 0).sum()),
                              'semantic_prefix_preserved': True},
             'collision_rounds': rounds, 'max_collision_rounds': MAX_ROUNDS, 'repair_trajectory': trajectory,
             'parameters': {'checkpoint_args': vars(args), 'checkpoint_epoch': checkpoint.get('epoch'),
                            'codebook_sizes': sizes, 'repair_sk_epsilons': REPAIR_EPSILONS,
                            'device': str(device), 'batch_size': cli.batch_size,
                            'torch_backends': {'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                                               'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
                                               'float32_matmul_precision': torch.get_float32_matmul_precision()},
                            'identity_suffix_policy': 'For each semantic 3-code group, rank ASINs lexicographically from zero; singleton rank is zero.'},
             'sources': before_sources, 'catalog_sync': catalog_sync,
             'outputs': {key: source_record(path) for key, path in outputs.items() if key != 'export_stats'},
             'validation': 'passed'}
    atomic_bytes(outputs['export_stats'], json_bytes(stats))
    latest_manifest = read_json(manifest_path)
    require(all(artifact_path(latest_manifest, key) == value for key, value in inputs.items()), 'Manifest input artifacts changed during export')
    latest_manifest['artifacts'].update({key: str(path) for key, path in outputs.items()})
    latest_manifest['sid_export'] = {'completed_at_utc': stats['created_at_utc'], 'validation': 'passed',
                                     'items': len(embeddings), 'codebook_sizes': sizes, 'unique_final_sids': stats['final']['unique_sids'],
                                     'checkpoint_role': cli.checkpoint_role, 'checkpoint_epoch': checkpoint.get('epoch'),
                                     'collision_rounds': rounds, 'catalog_sync': catalog_sync}
    atomic_bytes(manifest_path, json_bytes(latest_manifest))
    print(json.dumps({'validation': 'passed', 'run_directory': str(run_dir), 'raw': stats['raw'],
                      'semantic': stats['semantic'], 'final': stats['final'], 'collision_rounds': rounds,
                      'export_stats': str(outputs['export_stats'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
