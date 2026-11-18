#!/usr/bin/env python3
"""Validate Video Games RQ-VAE SIDs without SFT or recommendation training."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import html
import itertools
import json
import re
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.runtime/python_deps'))
sys.path.insert(0, str(ROOT / 'rq'))
GROUPS = [('Controllers', '手柄'), ('Headsets', '耳机'), ('Gaming Mice', '鼠标'), ('Gaming Keyboards', '键盘')]
SEED = 2024


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'min': None, 'p05': None, 'median': None, 'p95': None, 'max': None}
    check(np.isfinite(values).all(), 'Nonfinite distribution')
    return {'count': int(len(values)), 'mean': float(values.mean()), 'min': float(values.min()),
            'p05': float(np.percentile(values, 5)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def clean_text(raw_text):
    # Exact text cleaning used by rq/text2emb/utils.py for this string-only catalog.
    if isinstance(raw_text, list):
        cleaned = ' '.join(re.sub(r'["\n\r]*', '', re.sub(r'</?\w+[^>]*>', '', html.unescape(raw))).strip() for raw in raw_text)
    else:
        cleaned = str(raw_text)[1:-1].strip() if isinstance(raw_text, dict) else raw_text.strip()
        cleaned = re.sub(r'["\n\r]*', '', re.sub(r'</?\w+[^>]*>', '', html.unescape(cleaned)))
    index = -1
    while -index < len(cleaned) and cleaned[index] == '.':
        index -= 1
    index += 1
    cleaned = cleaned + '.' if index == 0 else cleaned[:index] + '.'
    return '' if len(cleaned) >= 2000 else cleaned


def text_fingerprints(metadata, n):
    result = []
    for item in range(n):
        fields = [clean_text(metadata[str(item)].get(key, '')) for key in ('title', 'description')]
        text = ' '.join(field.strip() for field in fields if field.strip()) or 'unknown item'
        result.append(hashlib.sha256(text.encode('utf-8')).hexdigest())
    return result


def validate_ids_and_codes(raw, semantic, raw_sid, semantic_sid, final_sid, n, sizes):
    expected = {str(i) for i in range(n)}
    for name, codes in [('raw', raw), ('semantic', semantic)]:
        check(codes.shape == (n, 3) and np.issubdtype(codes.dtype, np.integer), f'{name} codes must be integer (N,3)')
        for layer, size in enumerate(sizes):
            check(((codes[:, layer] >= 0) & (codes[:, layer] < size)).all(), f'{name} code out of range')
    check(np.array_equal(raw[:, :2], semantic[:, :2]), 'Collision repair changed an earlier semantic prefix')
    for mapping in (raw_sid, semantic_sid, final_sid):
        check(isinstance(mapping, dict) and set(mapping) == expected, 'SID catalog IDs mismatch')
    for item in range(n):
        key = str(item)
        expected_raw = [f'<{letter}_{int(value)}>' for letter, value in zip('abc', raw[item])]
        expected_semantic = [f'<{letter}_{int(value)}>' for letter, value in zip('abc', semantic[item])]
        check(raw_sid[key] == expected_raw, f'Raw SID/codes mismatch at {item}')
        check(semantic_sid[key] == expected_semantic, f'Semantic SID/codes mismatch at {item}')
        check(len(final_sid[key]) == 4 and final_sid[key][:3] == expected_semantic and re.fullmatch(r'<d_\d+>', final_sid[key][3]), f'Final SID format mismatch at {item}')
    check(len({tuple(value) for value in final_sid.values()}) == n, 'Final four-token SIDs are not unique')


def collision_stats(codes):
    counts = Counter(map(tuple, codes.tolist()))
    return {'items': len(codes), 'unique_semantic_sids': len(counts), 'collision_groups': sum(n > 1 for n in counts.values()),
            'items_in_collision_groups': sum(n for n in counts.values() if n > 1),
            'excess_items': len(codes) - len(counts), 'collision_rate': (len(codes) - len(counts)) / len(codes),
            'maximum_group_size': max(counts.values())}


def codebook_usage(codes, split_ids, sizes):
    result, rows = {}, []
    for split, ids in split_ids.items():
        layers = []
        for layer, size in enumerate(sizes):
            counts = np.bincount(codes[ids, layer], minlength=size)
            p = counts[counts > 0] / len(ids)
            entropy = float(-(p * np.log(p)).sum())
            layers.append({'layer': layer + 1, 'codewords': size, 'used': int((counts > 0).sum()),
                           'dead': int((counts == 0).sum()), 'entropy_nats': entropy,
                           'normalized_entropy': entropy / np.log(size), 'perplexity': float(np.exp(entropy)),
                           'maximum_share': float(counts.max() / len(ids)), 'counts': counts.tolist()})
            for code, count in enumerate(counts): rows.append({'split': split, 'layer': layer + 1, 'code': code, 'count': int(count), 'share': float(count / len(ids))})
        result[split] = layers
    return result, rows


def build_model(path, device):
    from models.rqvae import RQVAE
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    if isinstance(args, dict): args = SimpleNamespace(**args)
    required = ['num_emb_list', 'e_dim', 'layers', 'dropout_prob', 'bn', 'loss_type', 'quant_loss_weight', 'kmeans_init', 'kmeans_iters', 'sk_epsilons', 'sk_iters']
    check(all(hasattr(args, key) for key in required), 'Incomplete checkpoint architecture args')
    state = ckpt['state_dict']
    in_dim = int(state['encoder.mlp_layers.1.weight'].shape[1])
    check(in_dim == 2560 and list(args.num_emb_list) == [256, 256, 256], 'Unexpected RQ-VAE dimensions')
    model = RQVAE(in_dim=in_dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim, layers=args.layers,
                  dropout_prob=args.dropout_prob, bn=args.bn, loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight, beta=getattr(args, 'beta', .25),
                  kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, args, ckpt


@torch.no_grad()
def represent(model, array, codes, device, batch_size=512):
    latent = np.empty((len(array), model.e_dim), dtype=np.float32)
    quantized = np.empty_like(latent)
    reconstructed = np.empty_like(array, dtype=np.float32)
    codebooks = [q.embedding.weight for q in model.rq.vq_layers]
    for start in range(0, len(array), batch_size):
        batch = torch.as_tensor(np.asarray(array[start:start + batch_size]), dtype=torch.float32, device=device)
        indices = torch.as_tensor(codes[start:start + batch_size], dtype=torch.long, device=device)
        z = model.encoder(batch)
        zq = sum(codebook[indices[:, level]] for level, codebook in enumerate(codebooks))
        output = model.decoder(zq)
        latent[start:start + len(batch)] = z.cpu().numpy()
        quantized[start:start + len(batch)] = zq.cpu().numpy()
        reconstructed[start:start + len(batch)] = output.cpu().numpy()
    for value in (latent, quantized, reconstructed): check(np.isfinite(value).all(), 'Nonfinite RQ-VAE representation')
    return latent, quantized, reconstructed


def reconstruction_stats(original, reconstructed, quantized, latent, split_ids, train_mean):
    error = np.mean((np.asarray(original, dtype=np.float64) - reconstructed) ** 2, axis=1)
    baseline = np.mean((np.asarray(original, dtype=np.float64) - train_mean) ** 2, axis=1)
    denom = np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1)
    cosine = np.clip(np.sum(original * reconstructed, axis=1) / np.maximum(denom, 1e-12), -1, 1)
    quant_error = np.mean((latent - quantized) ** 2, axis=1)
    result = {}
    for split, ids in split_ids.items():
        mse, mean_mse = float(error[ids].mean()), float(baseline[ids].mean())
        result[split] = {'items': len(ids), 'mse': mse, 'cosine': distribution(cosine[ids]),
                        'train_mean_baseline_mse': mean_mse, 'nmse_vs_train_mean': mse / mean_mse if mean_mse > 0 else None,
                        'zero_reconstruction_rows': int((np.linalg.norm(reconstructed[ids], axis=1) == 0).sum()),
                        'latent_quantization_mse': float(quant_error[ids].mean())}
    return result


@torch.no_grad()
def nearest_neighbors(array, query_ids, device, k=20, batch_size=128):
    candidates = torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
    candidates = torch.nn.functional.normalize(candidates, dim=1)
    result = []
    for start in range(0, len(query_ids), batch_size):
        ids = np.asarray(query_ids[start:start + batch_size])
        query = candidates[torch.as_tensor(ids, dtype=torch.long, device=device)]
        score = query @ candidates.T
        score[torch.arange(len(ids), device=device), torch.as_tensor(ids, device=device)] = -float('inf')
        result.append(score.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(result)


def overlap_stats(reference, proposed, query_ids, split_ids, ks=(5, 10, 20)):
    is_val = np.isin(query_ids, split_ids['val'])
    result = {}
    for name, mask in [('train_queries', ~is_val), ('val_queries', is_val), ('all_queries', np.ones(len(query_ids), dtype=bool))]:
        ids = np.where(mask)[0]
        row = {'queries': len(ids)}
        for k in ks:
            overlaps = [len(set(reference[i, :k]) & set(proposed[i, :k])) / k for i in ids]
            row[f'overlap_at_{k}'] = float(np.mean(overlaps)) if overlaps else None
        result[name] = row
    return result


def lcp(a, b):
    equal = np.asarray(a) == np.asarray(b)
    return np.cumprod(equal, axis=-1).sum(axis=-1)


def sample_prefix_pairs(codes, fingerprints, limit=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    output, summaries = [], []
    for depth in range(4):
        buckets = defaultdict(list)
        for item, code in enumerate(codes): buckets[tuple(code[:depth])].append(item)
        eligible_buckets, weights, total_available = [], [], 0
        for ids in buckets.values():
            if len(ids) < 2: continue
            count = len(ids) * (len(ids) - 1) // 2
            if depth < 3:
                for n in Counter(int(codes[i, depth]) for i in ids).values(): count -= n * (n - 1) // 2
            by_text = defaultdict(list)
            for i in ids: by_text[fingerprints[i]].append(i)
            for same_text in by_text.values():
                excluded = len(same_text) * (len(same_text) - 1) // 2
                if depth < 3:
                    for n in Counter(int(codes[i, depth]) for i in same_text).values(): excluded -= n * (n - 1) // 2
                count -= excluded
            if count > 0:
                eligible_buckets.append(np.asarray(ids, dtype=np.int32)); weights.append(len(ids) * (len(ids) - 1) // 2); total_available += count
        desired = min(limit, total_available)
        selected, attempts = set(), 0
        if desired:
            probabilities = np.asarray(weights, dtype=np.float64); probabilities /= probabilities.sum()
            if total_available <= limit and max(map(len, eligible_buckets)) <= 256:
                for ids in eligible_buckets:
                    for a, b in itertools.combinations(map(int, ids), 2):
                        if fingerprints[a] != fingerprints[b] and int(lcp(codes[a], codes[b])) == depth: selected.add((min(a,b),max(a,b)))
                attempts = len(selected)
            else:
                while len(selected) < desired and attempts < max(200000, desired * 50):
                    ids = eligible_buckets[int(rng.choice(len(eligible_buckets), p=probabilities))]
                    a, b = map(int, rng.choice(ids, size=2, replace=False)); attempts += 1
                    if fingerprints[a] == fingerprints[b] or int(lcp(codes[a], codes[b])) != depth: continue
                    selected.add((min(a,b),max(a,b)))
        for a, b in sorted(selected): output.append({'lcp': depth, 'item_a': a, 'item_b': b})
        summaries.append({'lcp': depth, 'eligible_distinct_text_pairs': total_available, 'sampled_pairs': len(selected),
                          'target_sample_count': desired, 'sampling_incomplete': len(selected) < desired, 'attempts': attempts})
    return output, summaries


def prefix_diagnostics(codes, fingerprints, original, stage, out):
    rows, stats = sample_prefix_pairs(codes, fingerprints)
    norm = np.linalg.norm(original, axis=1)
    for row in rows:
        a, b = row['item_a'], row['item_b']
        row.update({'stage': stage, 'cosine_similarity': float(np.clip(np.dot(original[a], original[b]) / (norm[a] * norm[b]), -1, 1))})
    for summary in stats:
        values = [row['cosine_similarity'] for row in rows if row['lcp'] == summary['lcp']]
        summary['original_cosine'] = distribution(values)
        summary['cosine_samples'] = values
    write_csv(out / f'prefix_pairs_{stage}.csv', rows)
    return stats


def weak_labels(metadata, n):
    labels = np.full(n, -1, dtype=np.int32)
    for item in range(n):
        tokens = {value.strip() for value in html.unescape(metadata[str(item)].get('categories', '')).split(',')}
        matches = [i for i, (name, _) in enumerate(GROUPS) if name in tokens]
        if len(matches) == 1: labels[item] = matches[0]
    return labels


def same_label_retrieval(representations, labels, device):
    ids = np.where(labels >= 0)[0]
    result = {'labeled_items': len(ids), 'coverage': float(len(ids) / len(labels)), 'candidate_scope': 'only labeled four-category subset', 'metrics': {}}
    for stage, array in representations.items():
        neighbors = nearest_neighbors(array[ids], np.arange(len(ids)), device, k=10)
        stage_rows = []
        for group, (name, zh) in enumerate(GROUPS):
            mask = labels[ids] == group
            stage_rows.append({'group': name, 'group_zh': zh, 'items': int(mask.sum()),
                               'same_label_at_1': float((labels[ids][neighbors[mask, :1]] == group).mean()),
                               'same_label_at_5': float((labels[ids][neighbors[mask, :5]] == group).mean()),
                               'same_label_at_10': float((labels[ids][neighbors[mask, :10]] == group).mean())})
        result['metrics'][stage] = {'by_group': stage_rows, 'macro': {f'same_label_at_{k}': float(np.mean([row[f'same_label_at_{k}'] for row in stage_rows])) for k in (1,5,10)}}
    return result


@torch.no_grad()
def noise_stability(model, original, raw_codes, val_ids, device):
    rng = np.random.default_rng(SEED)
    rows, summary = [], []
    for ratio in (.005, .01, .02):
        codes = []
        for start in range(0, len(val_ids), 256):
            ids = val_ids[start:start + 256]
            batch = np.asarray(original[ids], dtype=np.float32)
            noise = rng.normal(size=batch.shape).astype(np.float32)
            noise *= (np.linalg.norm(batch, axis=1) * ratio / np.linalg.norm(noise, axis=1))[:, None]
            proposed = model.get_indices(torch.as_tensor(batch + noise, device=device), use_sk=False).cpu().numpy()
            codes.append(proposed)
        proposed = np.concatenate(codes)
        equality = proposed == raw_codes[val_ids]
        prefix = np.cumprod(equality, axis=1)
        row = {'relative_l2_noise': ratio, 'items': len(val_ids), 'prefix_1_retention': float(prefix[:, 0].mean()),
               'prefix_2_retention': float(prefix[:, 1].mean()), 'prefix_3_retention': float(prefix[:, 2].mean())}
        summary.append(row)
        for item, retains in zip(val_ids, prefix): rows.append({'item_id': int(item), 'relative_l2_noise': ratio,
                                                               'prefix_1_retained': int(retains[0]), 'prefix_2_retained': int(retains[1]), 'prefix_3_retained': int(retains[2])})
    return {'seed': SEED, 'repetitions_per_level': 1, 'noise_definition': 'Independent Gaussian direction normalized to exactly ratio*original vector L2 norm for every item', 'baseline': 'raw nearest-codeword semantic SID, identity suffix excluded', 'levels': summary}, rows



def prefix_neighbor_retention(reference_neighbors, query_ids, codes_by_stage, split_ids, permutations=100, seed=SEED, ks=(5,10,20)):
    """Compare original-vector neighbors with prefix sharing under occupancy-preserving nulls."""
    is_val = np.isin(query_ids, split_ids['val'])
    masks = {'train_queries': ~is_val, 'val_queries': is_val, 'all_queries': np.ones(len(query_ids),dtype=bool)}
    result = {'reference_metric':'original_vector_cosine', 'exclude_self':True, 'permutations':permutations,
              'seed':seed, 'null_definition':'Uniformly permute item-to-code assignments, preserving every code occupancy; reuse fixed original-vector neighbor IDs',
              'interval_definition':'2.5th and 97.5th empirical percentiles across permutations, not a recommendation confidence interval', 'stages':{}}
    rows = []
    for stage, codes in codes_by_stage.items():
        rng = np.random.default_rng(seed)
        observed_equal = np.cumprod(codes[reference_neighbors] == codes[np.asarray(query_ids),None,:],axis=2)
        observed, null = {}, {}
        for split, mask in masks.items():
            for k in ks:
                for depth in (1,2,3):
                    key=(split,k,depth)
                    observed[key]=float(observed_equal[mask,:k,depth-1].mean())
                    null[key]=[]
        for _ in range(permutations):
            shuffled = codes[rng.permutation(len(codes))]
            equal = np.cumprod(shuffled[reference_neighbors] == shuffled[np.asarray(query_ids),None,:],axis=2)
            for split, mask in masks.items():
                for k in ks:
                    for depth in (1,2,3):null[(split,k,depth)].append(float(equal[mask,:k,depth-1].mean()))
        result['stages'][stage]={}
        for split, mask in masks.items():
            stage_rows=[]
            for k in ks:
                for depth in (1,2,3):
                    key=(split,k,depth); draws=np.asarray(null[key])
                    applicable=len(np.unique(codes[:,:depth],axis=0)) < len(codes)
                    row={'k':k,'prefix_depth':depth,'queries':int(mask.sum()),'observed':observed[key],
                         'null_mean':float(draws.mean()),'null_p025':float(np.percentile(draws,2.5)),
                         'null_p975':float(np.percentile(draws,97.5)), 'applicable':applicable,
                         'above_null_p975':bool(observed[key] > np.percentile(draws,97.5)) if applicable else None}
                    stage_rows.append(row);rows.append({'stage':stage,'split':split,**row})
            result['stages'][stage][split]=stage_rows
    return result, rows


def assess_quality(reconstruction, knn, prefix_retention):
    warnings=[]
    nmse=reconstruction['semantic']['val']['nmse_vs_train_mean']
    nmse_pass=nmse is not None and nmse < 1
    if not nmse_pass:warnings.append('碰撞修复后验证集重构 NMSE 未优于训练均值向量基线。')
    knn_checks=[]
    for stage in ('raw_reconstruction','semantic_reconstruction'):
        for k in (5,10,20):
            observed=knn['metrics'][stage]['val_queries'][f'overlap_at_{k}']
            initial=knn['metrics']['initial_reconstruction']['val_queries'][f'overlap_at_{k}']
            random=knn['random_overlap_baseline'][str(k)]
            passed=observed is not None and observed > initial and observed > random
            knn_checks.append({'stage':stage,'k':k,'observed':observed,'initial':initial,'random':random,'supported':bool(passed)})
            if not passed:warnings.append(f'{stage} 验证集 overlap@{k} 未同时高于初始化和随机基线。')
    prefix_checks=[]
    for stage in ('raw','semantic'):
        for row in prefix_retention['stages'][stage]['val_queries']:
            prefix_checks.append({'stage':stage,**row})
            if row['applicable'] and not row['above_null_p975']:
                warnings.append(f'{stage} 验证集 Top{row["k"]} 前缀{row["prefix_depth"]}共享率未高于100次置换null的97.5百分位。')
    uninformative=[{'stage':row['stage'],'k':row['k'],'prefix_depth':row['prefix_depth'],'reason':'全部商品在此深度前缀唯一，精确共享率恒为0，无鉴别力。'} for row in prefix_checks if not row['applicable']]
    return {'structural_validation':'passed','quality_status':'supported_on_defined_diagnostics' if not warnings else 'needs_review',
            'semantic_val_nmse':{'observed':nmse,'threshold':1.0,'better_than_train_mean':bool(nmse_pass)},
            'knn_validation_comparisons':knn_checks,'prefix_validation_comparisons':prefix_checks,
            'non_applicable_prefix_checks':uninformative,'warnings':warnings,
            'interpretation':'Structural pass only verifies valid artifacts; semantic quality is separately assessed by these item-content diagnostics, never recommendation accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-train-limit', type=int, default=2048)
    args = parser.parse_args()
    if args.query_train_limit < 1: parser.error('--query-train-limit must be positive')
    run = args.run_dir.resolve(strict=True); out = run / 'validation'; out.mkdir(exist_ok=True)
    manifest = read_json(run / 'manifest.json'); artifacts = manifest['artifacts']
    def path(key):
        value = Path(artifacts[key]); return value if value.is_absolute() else run / value
    required = ['checkpoint','initial_checkpoint','split','raw_codes','semantic_codes','raw_sid','semantic_sid','final_sid','embeddings','item_metadata']
    for key in required: check(path(key).is_file(), f'Missing artifact {key}: {path(key)}')
    original = np.load(path('embeddings'), allow_pickle=False)
    check(original.shape == (16572,2560) and original.dtype == np.float32 and np.isfinite(original).all(), 'Invalid original embeddings')
    n = len(original); split = read_json(path('split'))
    check(split['seed']==SEED, 'Unexpected split seed')
    check(sha256(path('embeddings'))==manifest['embedding_sha256']==split['embedding_sha256'], 'Embedding hash differs from training/split provenance')
    train_ids = np.asarray(split['train_ids'], dtype=np.int64); val_ids = np.asarray(split['val_ids'], dtype=np.int64)
    check(len(train_ids)==14919 and len(val_ids)==1653, 'Unexpected split sizes')
    check(np.array_equal(np.sort(np.concatenate([train_ids,val_ids])), np.arange(n)), 'Split is not a disjoint catalog partition')
    split_ids = {'train': train_ids, 'val': val_ids, 'all': np.arange(n)}
    metadata = read_json(path('item_metadata')); check(set(metadata)=={str(i) for i in range(n)}, 'Metadata alignment mismatch')
    fingerprints = text_fingerprints(metadata,n)
    overlap = set(fingerprints[i] for i in train_ids) & set(fingerprints[i] for i in val_ids)
    check(not overlap, 'Exact cleaned embedding text overlaps between train and validation')
    raw_codes = np.load(path('raw_codes'), allow_pickle=False); semantic_codes = np.load(path('semantic_codes'), allow_pickle=False)
    validate_ids_and_codes(raw_codes,semantic_codes,read_json(path('raw_sid')),read_json(path('semantic_sid')),read_json(path('final_sid')),n,[256]*3)
    device = torch.device(args.device); torch.set_num_threads(4)
    if device.type=='cuda': check(torch.cuda.is_available(), 'CUDA unavailable')
    model, model_args, checkpoint = build_model(path('checkpoint'),device)
    latent, raw_quantized, raw_reconstructed = represent(model,original,raw_codes,device)
    _, semantic_quantized, semantic_reconstructed = represent(model,original,semantic_codes,device)
    # Raw codes must be independently reproduced from the selected checkpoint.
    with torch.no_grad():
        for start in range(0,n,512):
            proposed=model.get_indices(torch.as_tensor(original[start:start+512],device=device),use_sk=False).cpu().numpy()
            check(np.array_equal(proposed,raw_codes[start:start+512]), 'Raw SID codes cannot be reproduced from checkpoint')
    train_mean=np.asarray(original[train_ids],dtype=np.float64).mean(axis=0)
    reconstruction={'raw':reconstruction_stats(original,raw_reconstructed,raw_quantized,latent,split_ids,train_mean),
                    'semantic':reconstruction_stats(original,semantic_reconstructed,semantic_quantized,latent,split_ids,train_mean)}
    initial_model, _, initial_checkpoint=build_model(path('initial_checkpoint'),device)
    initial_codes=[]
    with torch.no_grad():
        for start in range(0,n,512): initial_codes.append(initial_model.get_indices(torch.as_tensor(original[start:start+512],device=device),use_sk=False).cpu().numpy())
    initial_codes=np.concatenate(initial_codes)
    initial_latent,initial_quantized,initial_reconstructed=represent(initial_model,original,initial_codes,device)
    reconstruction['initial']=reconstruction_stats(original,initial_reconstructed,initial_quantized,initial_latent,split_ids,train_mean)
    del initial_model,initial_latent,initial_quantized
    if device.type=='cuda':torch.cuda.empty_cache()
    rng=np.random.default_rng(SEED)
    sampled_train=np.sort(rng.choice(train_ids,size=min(args.query_train_limit,len(train_ids)),replace=False))
    queries=np.concatenate([np.sort(val_ids),sampled_train])
    reference=nearest_neighbors(original,queries,device)
    latent_reference=nearest_neighbors(latent,queries,device)
    knn={'metric':'cosine','candidate_items':n,'exclude_self':True,'query_ids':queries.tolist(),'seed':SEED,'random_overlap_baseline':{str(k):k/(n-1) for k in (5,10,20)},'metrics':{}}
    for stage,array,ref in [('initial_reconstruction',initial_reconstructed,reference),('raw_reconstruction',raw_reconstructed,reference),('semantic_reconstruction',semantic_reconstructed,reference),('raw_quantized_latent',raw_quantized,latent_reference),('semantic_quantized_latent',semantic_quantized,latent_reference)]:
        proposed=nearest_neighbors(array,queries,device)
        knn['metrics'][stage]=overlap_stats(ref,proposed,queries,split_ids)
    prefix_retention,prefix_neighbor_rows=prefix_neighbor_retention(reference,queries,{'raw':raw_codes,'semantic':semantic_codes},split_ids)
    write_csv(out/'prefix_neighbor_retention.csv',prefix_neighbor_rows)
    del initial_reconstructed
    prefixes={'raw':prefix_diagnostics(raw_codes,fingerprints,original,'raw',out),'semantic':prefix_diagnostics(semantic_codes,fingerprints,original,'semantic',out)}
    usage={};usage_rows=[]
    for stage,codes in [('raw',raw_codes),('semantic',semantic_codes)]:
        usage[stage],rows=codebook_usage(codes,split_ids,[256]*3)
        usage_rows.extend([{'stage':stage,**row} for row in rows])
    write_csv(out/'codebook_usage.csv',usage_rows)
    noise,noise_rows=noise_stability(model,original,raw_codes,val_ids,device);write_csv(out/'noise_stability.csv',noise_rows)
    weak=same_label_retrieval({'original':original,'raw_reconstruction':raw_reconstructed,'semantic_reconstruction':semantic_reconstructed},weak_labels(metadata,n),device)
    old_analysis=read_json(ROOT/'output/reports/Video_Games/02_embedding_similarity/analysis.json')
    anchors=[row['anchor_id'] for row in old_analysis['examples']]
    example_neighbors={name:nearest_neighbors(array,np.asarray(anchors),device,k=5) for name,array in [('original',original),('raw_reconstruction',raw_reconstructed),('semantic_reconstruction',semantic_reconstructed)]}
    examples=[];example_rows=[]
    original_norm=original/np.linalg.norm(original,axis=1,keepdims=True)
    for pos,anchor in enumerate(anchors):
        entry={'anchor_id':anchor,'title':metadata[str(anchor)]['title'],'raw_sid':raw_codes[anchor].tolist(),'semantic_sid':semantic_codes[anchor].tolist(),'neighbors':{}}
        for stage,neighbors in example_neighbors.items():
            stage_rows=[]
            for rank,item in enumerate(neighbors[pos],start=1):
                row={'anchor_id':anchor,'stage':stage,'rank':rank,'neighbor_id':int(item),'title':metadata[str(int(item))]['title'],
                     'original_cosine':float(np.dot(original_norm[anchor],original_norm[item])),'raw_lcp':int(lcp(raw_codes[anchor],raw_codes[item])),
                     'semantic_lcp':int(lcp(semantic_codes[anchor],semantic_codes[item]))}
                stage_rows.append(row);example_rows.append(row)
            entry['neighbors'][stage]=stage_rows
        examples.append(entry)
    write_csv(out/'neighbor_examples.csv',example_rows)
    changed=np.any(raw_codes!=semantic_codes,axis=1)
    changed_ids=np.where(changed)[0]
    if len(changed_ids):
        raw_error=np.mean((original[changed_ids]-raw_reconstructed[changed_ids])**2,axis=1)
        semantic_error=np.mean((original[changed_ids]-semantic_reconstructed[changed_ids])**2,axis=1)
        distortion={'changed_items':int(changed.sum()),'raw_mse':float(raw_error.mean()),'semantic_mse':float(semantic_error.mean()),'mse_delta':distribution(semantic_error-raw_error)}
    else:distortion={'changed_items':0,'raw_mse':None,'semantic_mse':None,'mse_delta':distribution([])}
    metric_rows=[]
    for stage,parts in reconstruction.items():
        for split_name,row in parts.items():metric_rows.append({'stage':stage,'split':split_name,'mse':row['mse'],'mean_cosine':row['cosine']['mean'],'train_mean_baseline_mse':row['train_mean_baseline_mse'],'nmse_vs_train_mean':row['nmse_vs_train_mean'],'latent_quantization_mse':row['latent_quantization_mse']})
    write_csv(out/'reconstruction_metrics.csv',metric_rows)
    knn_rows=[]
    for stage,parts in knn['metrics'].items():
        for split_name,row in parts.items():
            for k in (5,10,20):knn_rows.append({'stage':stage,'split':split_name,'queries':row['queries'],'k':k,'overlap':row[f'overlap_at_{k}'],'random_baseline':k/(n-1)})
    write_csv(out/'knn_overlap.csv',knn_rows)
    report={'validation':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'run_directory':str(run),'items':n,'dimensions':2560,'split':{'train':len(train_ids),'val':len(val_ids),'seed':SEED,'exact_cleaned_text_overlap':0},
            'reconstruction':reconstruction,'knn':knn,'prefix_neighbor_retention':prefix_retention,'quality_assessment':assess_quality(reconstruction,knn,prefix_retention),'prefix_similarity':prefixes,'codebook_usage':usage,'noise_stability':noise,
            'collisions':{'raw':collision_stats(raw_codes),'semantic':collision_stats(semantic_codes),'final_four_token_unique':True,'repair_distortion':distortion},
            'weak_label_retrieval':weak,'examples':examples,'identity_suffix_excluded_from_semantic_metrics':True,
            'checks':{'embedding_shape_dtype_finite':True,'disjoint_complete_group_split':True,'exact_cleaned_text_train_val_overlap_zero':True,'item_ids_and_codes_aligned':True,'code_ranges_valid':True,'raw_codes_reproduced_from_checkpoint':True,'all_representations_finite':True,'semantic_repair_first_two_levels_unchanged':True,'final_four_token_sids_unique':True},
            'baseline_definition':'NMSE = reconstruction MSE / MSE of constant training-mean vector; train mean fitted on training items only',
            'sources':{key:{'path':str(path(key)),'sha256':sha256(path(key))} for key in required},
            'limitations':['Catalog content diagnostics and held-out item reconstruction are not recommendation accuracy.',
                           'LCP is categorical prefix agreement; numeric codeword IDs have no ordinal distance.',
                           'Nearest neighbors retain same-title and near-duplicate items after excluding self.',
                           'Prefix pairs exclude identical cleaned encoding text and may be unavailable at LCP=3.',
                           'Four-category labels are noisy and cover only a small catalog subset.']}
    (run/'sid_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'validation':'passed','report':str(run/'sid_validation.json'),'reconstruction':reconstruction,'collisions':report['collisions'],'knn':knn['metrics']},ensure_ascii=False))


if __name__=='__main__':main()
