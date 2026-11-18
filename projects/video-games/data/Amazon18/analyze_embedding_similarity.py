#!/usr/bin/env python3
"""Catalog-based diagnostics of Video Games product embedding distances."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import html
import json
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.runtime/python_deps'))
from sklearn.decomposition import PCA
DATA = ROOT / 'data/Amazon18/Video_Games'
OUT = ROOT / 'output/reports/Video_Games/02_embedding_similarity'
GROUPS = [('Controllers', '手柄'), ('Headsets', '耳机'), ('Gaming Mice', '鼠标'), ('Gaming Keyboards', '键盘')]
SEED = 20261114


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(value):
    return re.sub(r'\s+', ' ', html.unescape(value).strip()).casefold()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    items = json.loads((DATA / 'Video_Games.item.json').read_text(encoding='utf-8'))
    array = np.load(DATA / 'Video_Games.emb-qwen-td.npy', allow_pickle=False)
    assert array.shape == (16572, 2560) and np.isfinite(array).all()
    n = len(array)
    titles = [items[str(i)]['title'] for i in range(n)]
    norm_titles = [normalize(title) for title in titles]
    brands = [normalize(items[str(i)]['brand']) for i in range(n)]
    labels = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        tokens = {part.strip() for part in html.unescape(items[str(i)]['categories']).split(',')}
        matches = [j for j, (token, _) in enumerate(GROUPS) if token in tokens]
        assert len(matches) <= 1, (i, matches)
        if matches:
            labels[i] = matches[0]
    grouped = [np.where(labels == i)[0] for i in range(len(GROUPS))]
    labeled = np.where(labels >= 0)[0]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    raw = torch.from_numpy(array).to(device)
    normalized = torch.nn.functional.normalize(raw, p=2, dim=1)

    def metrics(left, right):
        cos, l2 = [], []
        for start in range(0, len(left), 1024):
            a = torch.as_tensor(left[start:start + 1024], dtype=torch.long, device=device)
            b = torch.as_tensor(right[start:start + 1024], dtype=torch.long, device=device)
            cos.extend((normalized[a] * normalized[b]).sum(dim=1).clamp(-1, 1).cpu().tolist())
            l2.extend(torch.linalg.vector_norm(raw[a] - raw[b], dim=1).cpu().tolist())
        return np.asarray(cos), np.asarray(l2)

    anchor_ids, positive_ids, negative_ids = [], [], []
    for group, ids in enumerate(grouped):
        anchors = rng.choice(ids, size=min(300, len(ids)), replace=False)
        for anchor in anchors:
            candidates = [int(i) for i in ids if i != anchor and norm_titles[i] != norm_titles[anchor]]
            if not candidates:
                continue
            positive = int(rng.choice(candidates))
            other_group = int(rng.choice([j for j in range(len(GROUPS)) if j != group]))
            negative_candidates = [int(i) for i in grouped[other_group] if norm_titles[i] != norm_titles[anchor]]
            anchor_ids.append(int(anchor)); positive_ids.append(positive); negative_ids.append(int(rng.choice(negative_candidates)))
    positive_cos, positive_l2 = metrics(anchor_ids, positive_ids)
    negative_cos, negative_l2 = metrics(anchor_ids, negative_ids)
    anchor_labels = labels[anchor_ids]
    pair_rows, by_group = [], []
    for group, (name, zh) in enumerate(GROUPS):
        mask = anchor_labels == group
        by_group.append({'group': name, 'group_zh': zh, 'catalog_items': int(len(grouped[group])), 'anchors': int(mask.sum()),
                         'same_cos_mean': float(positive_cos[mask].mean()), 'different_cos_mean': float(negative_cos[mask].mean()),
                         'cos_gap': float((positive_cos[mask] - negative_cos[mask]).mean()),
                         'same_l2_mean': float(positive_l2[mask].mean()), 'different_l2_mean': float(negative_l2[mask].mean()),
                         'cosine_win_rate': float(((positive_cos[mask] > negative_cos[mask]) + .5 * (positive_cos[mask] == negative_cos[mask])).mean()),
                         'l2_win_rate': float(((positive_l2[mask] < negative_l2[mask]) + .5 * (positive_l2[mask] == negative_l2[mask])).mean())})
    for i, anchor in enumerate(anchor_ids):
        for kind, partner, cos, l2 in [('same_group', positive_ids[i], positive_cos[i], positive_l2[i]), ('different_group', negative_ids[i], negative_cos[i], negative_l2[i])]:
            pair_rows.append({'anchor_id': anchor, 'partner_id': partner, 'anchor_group': GROUPS[labels[anchor]][0], 'kind': kind,
                              'cosine_similarity': float(cos), 'cosine_distance': float(1 - cos), 'raw_l2_distance': float(l2),
                              'anchor_title': titles[anchor], 'partner_title': titles[partner]})
    with (OUT / 'sampled_pairs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)

    # Exact retrieval, restricted to the independently labeled subset.
    sub = normalized[labeled]
    cosine_matrix = (sub @ sub.T).clamp(-1, 1)
    sub_raw = raw[labeled]
    squared_norm = (sub_raw * sub_raw).sum(dim=1)
    squared_l2 = (squared_norm[:, None] + squared_norm[None, :] - 2 * (sub_raw @ sub_raw.T)).clamp_min(0)
    cosine_matrix.fill_diagonal_(-2)
    squared_l2.fill_diagonal_(float('inf'))
    cosine_top = cosine_matrix.topk(10, dim=1).indices.cpu().numpy()
    l2_top = squared_l2.topk(10, dim=1, largest=False).indices.cpu().numpy()
    sub_labels = labels[labeled]
    retrieval = {'candidate_items': int(len(labeled)), 'queries': int(len(labeled)), 'ks': [1, 5, 10], 'by_group': []}
    for group, (name, zh) in enumerate(GROUPS):
        q = sub_labels == group
        entry = {'group': name, 'group_zh': zh, 'queries': int(q.sum()), 'random_baseline': float((q.sum() - 1) / (len(labeled) - 1))}
        for metric, neighbors in [('cosine', cosine_top), ('l2', l2_top)]:
            entry[metric] = [float((sub_labels[neighbors[q, :k]] == group).mean()) for k in retrieval['ks']]
        retrieval['by_group'].append(entry)
    for metric in ['cosine', 'l2']:
        retrieval[metric + '_macro'] = np.mean([row[metric] for row in retrieval['by_group']], axis=0).tolist()
    retrieval['random_macro'] = float(np.mean([row['random_baseline'] for row in retrieval['by_group']]))
    with (OUT / 'labeled_retrieval.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['group', 'queries', 'metric', 'k', 'known_same_label_fraction', 'random_baseline'])
        for row in retrieval['by_group']:
            for metric in ['cosine', 'l2']:
                for k, score in zip(retrieval['ks'], row[metric]): writer.writerow([row['group'], row['queries'], metric, k, score, row['random_baseline']])

    # Brand controls use all unordered valid pairs in the labeled subset; at most 2,000 per cell.
    aa, bb = np.triu_indices(len(labeled), k=1)
    ids_a, ids_b = labeled[aa], labeled[bb]
    valid_title = np.asarray([norm_titles[a] != norm_titles[b] for a, b in zip(ids_a, ids_b)])
    valid_brand = np.asarray([bool(brands[a]) and bool(brands[b]) for a, b in zip(ids_a, ids_b)])
    same_brand = np.asarray([brands[a] == brands[b] for a, b in zip(ids_a, ids_b)])
    same_group = labels[ids_a] == labels[ids_b]
    controls = []
    control_names = [(True, True, '同类同品牌'), (True, False, '同类异品牌'), (False, True, '异类同品牌'), (False, False, '异类异品牌')]
    for category_equal, brand_equal, zh in control_names:
        available = np.where(valid_title & valid_brand & (same_group == category_equal) & (same_brand == brand_equal))[0]
        selection = rng.choice(available, size=min(2000, len(available)), replace=False)
        cos, l2 = metrics(ids_a[selection], ids_b[selection])
        controls.append({'condition_zh': zh, 'eligible_pairs': int(len(available)), 'sampled_pairs': int(len(selection)),
                         'cos_mean': float(cos.mean()), 'cos_median': float(np.median(cos)), 'l2_mean': float(l2.mean()),
                         'cos_samples': cos.tolist(), 'l2_samples': l2.tolist()})
    with (OUT / 'brand_controls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['condition', 'eligible_pairs', 'sampled_pairs', 'cosine_mean', 'cosine_median', 'raw_l2_mean'])
        for row in controls: writer.writerow([row['condition_zh'], row['eligible_pairs'], row['sampled_pairs'], row['cos_mean'], row['cos_median'], row['l2_mean']])

    # Anchors are selected from metadata before any neighbor scores are inspected.
    anchor_examples = [int(ids.min()) for ids in grouped]
    anchor_examples.append(7)  # Super Mario Galaxy, fixed independently of vector results.
    cod = [i for i in range(n) if 'call of duty' in titles[i].casefold() and 'Games' in html.unescape(items[str(i)]['categories']).split(',')]
    if cod: anchor_examples.append(min(cod))
    examples, neighbor_rows = [], []
    for anchor in anchor_examples:
        score = (normalized[anchor] @ normalized.T).clamp(-1, 1)
        score[anchor] = -2
        values, indices = score.topk(5)
        ids = indices.cpu().tolist()
        _, distances = metrics([anchor] * len(ids), ids)
        info = {'anchor_id': anchor, 'title': titles[anchor], 'brand': items[str(anchor)]['brand'], 'group_zh': GROUPS[labels[anchor]][1] if labels[anchor] >= 0 else '游戏作品',
                'description_discarded_by_original_rule': len(items[str(anchor)]['description']) >= 2000, 'neighbors': []}
        for rank, (partner, cosine, l2) in enumerate(zip(ids, values.cpu().tolist(), distances), start=1):
            row = {'anchor_id': anchor, 'anchor_title': titles[anchor], 'rank': rank, 'neighbor_id': partner,
                   'neighbor_title': titles[partner], 'neighbor_brand': items[str(partner)]['brand'],
                   'cosine_similarity': float(cosine), 'cosine_distance': float(1 - cosine), 'raw_l2_distance': float(l2),
                   'same_weak_label': bool(labels[anchor] >= 0 and labels[partner] == labels[anchor])}
            info['neighbors'].append(row); neighbor_rows.append(row)
        examples.append(info)
    with (OUT / 'nearest_neighbors.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(neighbor_rows[0])); writer.writeheader(); writer.writerows(neighbor_rows)

    normalized_cpu = normalized[labeled].cpu().numpy()
    pca = PCA(n_components=2, svd_solver='randomized', random_state=SEED)
    coordinates = pca.fit_transform(normalized_cpu)
    projection = {'method': 'PCA on L2-normalized vectors, labeled subset only', 'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                  'points': [{'id': int(item), 'group': int(labels[item]), 'x': float(x), 'y': float(y)} for item, (x, y) in zip(labeled, coordinates)]}
    norms = np.linalg.norm(array, axis=1)
    summary = {'dataset': 'Video_Games', 'model_id': 'Qwen/Qwen3-Embedding-4B', 'items': n, 'dimensions': array.shape[1],
               'labeled_items': int(len(labeled)), 'labeled_coverage': float(len(labeled) / n), 'seed': SEED,
               'pair_anchors': len(anchor_ids), 'by_group': by_group,
               'macro_cosine_gap': float(np.mean([row['cos_gap'] for row in by_group])),
               'macro_cosine_win_rate': float(np.mean([row['cosine_win_rate'] for row in by_group])),
               'macro_l2_win_rate': float(np.mean([row['l2_win_rate'] for row in by_group])),
               'norms': {'min': float(norms.min()), 'max': float(norms.max()), 'mean': float(norms.mean())}}
    result = {'summary': summary,
              'distributions': {'same_cosine': positive_cos.tolist(), 'different_cosine': negative_cos.tolist(), 'same_l2': positive_l2.tolist(), 'different_l2': negative_l2.tolist()},
              'brand_control': controls, 'retrieval': retrieval, 'projection': projection, 'examples': examples,
              'provenance': {'embedding_sha256': sha256(DATA / 'Video_Games.emb-qwen-td.npy'), 'metadata_sha256': sha256(DATA / 'Video_Games.item.json'),
                             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'device': device,
                             'group_definition': 'Exact known category token after HTML unescape; no score-based labels or anchor selection.',
                             'pair_sampling': 'At most 300 unique anchors per category, one distinct-title same-category partner and one uniformly chosen other category partner per anchor.',
                             'brand_control_sampling': 'Up to 2000 distinct-title, nonempty-brand unordered pairs per observed category/brand cell; not a causal experiment.',
                             'distance_definition': 'cosine = x dot y / (norm(x)*norm(y)); raw_l2 = norm(x-y). Source embeddings are unchanged.',
                             'limitations': [f'Weak category labels cover {len(labeled)/n:.2%} of the catalog and include some mislabeled accessories.', 'Two-dimensional PCA is a projection, not the original 2560-dimensional distance.', 'No human relevance judgments or downstream recommendation accuracy are measured.']}}
    (OUT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'summary': summary, 'retrieval': retrieval, 'brand_control': [{k:v for k,v in row.items() if not k.endswith('_samples')} for row in controls],
                      'examples': [{'id': row['anchor_id'], 'title': row['title'], 'first_neighbor': row['neighbors'][0]} for row in examples]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
