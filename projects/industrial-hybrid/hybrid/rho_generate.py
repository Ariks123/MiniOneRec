"""Generate UniSRec delta=K primitives in the original four-rank evaluation batches."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from collections import Counter

from hybrid.experiment import Data, Engine, KS, WIDTHS, dump, split_prefix


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_batch(engine, rows, archived, counters):
    for row, old in zip(rows, archived, strict=True):
        if (row['row_id'], row['target']) != (old['row_id'], old['target']):
            raise ValueError('Mismatched original request or label')
    session = engine.v.prefill(engine.v.prompts([row['history_sids'] for row in rows]))
    prefixes = {}
    for width in WIDTHS:
        for depth in (1, 2):
            prefixes[width, depth] = [
                [split_prefix(s) for s in group]
                for group in engine.v.prefixes(session, width, depth)
            ]

    drafter = engine.d['unisrec']
    queries = drafter.prepare([row['history'] for row in rows])
    touched, grids = [], []
    for j, (old, query) in enumerate(zip(archived, queries, strict=True)):
        grid = {}
        # Preserve every alias because retained retrieval items can change which
        # alias is available when replay takes a SID from the original beam.
        items = {item for sid in old['beam'] for item in engine.data.by_sid[sid]}
        old_grid = old['drafts']['unisrec']
        for k in KS:
            first = engine.request_drafts(drafter, query, None, set(), k)
            if len(first) != k or first != old_grid[str(k)]['first'][:k]:
                raise ValueError(f"Original UniSRec first K differs at row {old['row_id']} K={k}")
            counters['first_round_top_k_matches'] += 1
            items.update(first)
            case = {'first': first}
            for width in WIDTHS:
                seen = set(first)
                rounds = []
                for depth in (1, 2):
                    group = engine.request_drafts(drafter, query, prefixes[width, depth][j], seen, k)
                    rounds.append(group)
                    items.update(group)
                    seen.update(group)
                case[str(width)] = rounds
            # Equal draft sizes must generate identical per-round candidates even
            # though the output K and the downstream stopping quota can differ.
            if k in (10, 20):
                if case != old_grid[str(k // 2)]:
                    raise ValueError(f"Cross-K equal-size drafts differ at row {old['row_id']} K={k}")
                counters['equal_size_grid_matches'] += 1
            grid[str(k)] = case
        grids.append(grid)
        touched.append(sorted(items))

    scores = engine.get_scores(session, list(range(len(rows))), touched)
    result = []
    for row, old, grid, evaluated, ids in zip(rows, archived, grids, scores, touched, strict=True):
        if set(evaluated) != set(ids) or not all(math.isfinite(v) for v in evaluated.values()):
            raise ValueError(f"Incomplete/nonfinite scores at row {row['row_id']}")
        old_scores = old['scores']
        for item, value in evaluated.items():
            if str(item) in old_scores:
                err = abs(old_scores[str(item)] - value)
                counters['max_common_score_abs_diff'] = max(counters['max_common_score_abs_diff'], err)
                counters['scored_in_both'] += 1
            else:
                counters['newly_scored_items'] += 1
        counters['scored_items'] += len(evaluated)
        result.append(dict(row_id=row['row_id'], target=row['target'], beam=old['beam'],
                           rho=1, drafts={'unisrec': grid}, scores=evaluated))
    return result


def generate_split(args, engine, split):
    old_file = args.original / split / f'rank{args.rank}.jsonl'
    out = args.run_dir / split / old_file.name
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = out.with_suffix('.audit.json')
    identity = dict(rho=1, split=split, rank=args.rank, ranks=4, batch_size=4,
                    archive_sha256=digest(old_file), generator_sha256=digest(Path(__file__)))
    if split == 'valid':
        old_hash = json.loads((args.original / 'report/selection.json').read_text())['valid_inputs'][old_file.name]
        if identity['archive_sha256'] != old_hash:
            raise ValueError(f'Frozen old validation shard changed: {old_file}')
    if out.is_file() and audit.is_file():
        record = json.loads(audit.read_text())
        if record['identity'] != identity or record['output_sha256'] != digest(out):
            raise ValueError(f'Incompatible completed shard at {out}')
        return
    rows = engine.data.read(split)[args.rank::4]
    counters = Counter()
    started = time.perf_counter()
    pending = out.with_suffix('.partial')
    progress = args.run_dir / f'progress-{args.rank}.json'
    with old_file.open() as reference, pending.open('w') as output:
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            old_rows = [json.loads(next(reference)) for _ in batch]
            records = build_batch(engine, batch, old_rows, counters)
            for rec in records:
                output.write(json.dumps(rec, separators=(',', ':'), allow_nan=False) + '\n')
            done = start + len(batch)
            if start % 40 == 0 or done == len(rows):
                output.flush()
                dump(progress, dict(split=split, done=done, total=len(rows),
                                    elapsed=time.perf_counter()-started))
        if reference.readline():
            raise ValueError('Extra archived requests after the last expected row')
    pending.replace(out)
    dump(audit, dict(identity=identity, rows=len(rows), counts=dict(counters),
                     output_sha256=digest(out), seconds=time.perf_counter()-started))
    print(split, args.rank, len(rows), dict(counters), flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, default=Path('/assets'))
    ap.add_argument('--original', type=Path, default=Path('/run/hybrid'))
    ap.add_argument('--run-dir', type=Path, default=Path('/run/hybrid-rho1'))
    ap.add_argument('--splits', choices=('valid','test'), nargs='+', default=('valid','test'))
    args = ap.parse_args()
    args.rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 4 or args.rank != int(os.environ['LOCAL_RANK']):
        raise ValueError('Use four local ranks to reproduce original request batch shapes')
    torch.cuda.set_device(args.rank)
    torch.set_num_threads(2)
    data = Data(args.assets)
    if json.loads((args.original / 'catalog.json').read_text()) != data.record:
        raise ValueError('Frozen item/SID catalog differs')
    if args.rank == 0:
        dump(args.run_dir / 'catalog.json', data.record)
    engine = Engine(data, f'cuda:{args.rank}', names=('unisrec',))
    for split in args.splits:
        generate_split(args, engine, split)


if __name__ == '__main__':
    main()
