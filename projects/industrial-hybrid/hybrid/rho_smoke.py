"""Compare live rho=1 routing with the independently stored four-request primitives."""
import argparse
import json
from itertools import islice
from pathlib import Path

from hybrid.audit import replay
from hybrid.experiment import Data, Engine, dump
from hybrid.sweep import Catalog


class RecordingEngine(Engine):
    def run_stage(self, key, fn, *args, **kwargs):
        result = super().run_stage(key, fn, *args, **kwargs)
        if key == 'verification':
            _, lanes, candidate_groups = args
            for lane, items, scores in zip(lanes, candidate_groups, result, strict=True):
                self.observed[lane].append((items, scores))
        return result


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=Path('/run/hybrid-rho1'))
    ap.add_argument('--assets', type=Path, default=Path('/assets'))
    args = ap.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    data = Data(args.assets)
    examples = data.read('test')
    catalog = Catalog.from_dict(data.record)
    engine = RecordingEngine(data, 'cuda:0', names=('unisrec',))
    settings = [
        ('pure_beam', 10, 0, 1, None),
        ('mixed', 10, 4, 5, None),
        ('mixed_threshold', 10, 4, 5, -1.75),
        ('k10_accept_all', 10, 10, 1, None),
        ('k50_accept_all', 50, 50, 1, None),
        ('strict_forced_beam', 50, 50, 20, 0.0),
        ('k50_multiround', 50, 40, 10, -2.0),
    ]
    results = []
    for rank in (0, 3):
        with (args.run_dir / 'test' / f'rank{rank}.jsonl').open() as f:
            archived = [json.loads(line) for line in islice(f, 4)]
        for lane,row in enumerate(archived):
            if row['row_id'] != rank + lane*4: raise ValueError('Original cohort differs')
            row['scores'] = {int(i):v for i,v in row['scores'].items()}
        for name,k,q,w,theta in settings:
            cfg = dict(drafter='unisrec',k=k,quota=q,prefix_beams=w,threshold=theta,rho=1,max_rounds=3)
            engine.observed = [[] for _ in archived]
            live = engine.online([examples[r['row_id']] for r in archived], cfg, catalog)
            for lane, (row, actual) in enumerate(zip(archived, live, strict=True)):
                expected = replay(row,cfg,catalog)
                candidates = row['drafts']['unisrec'][str(k)]
                followed, accepted = [], 0
                if q:
                    for ids in [candidates['first'], *candidates[str(w)]]:
                        followed.append(ids)
                        accepted += sum(theta is None or row['scores'][i]>theta for i in ids)
                        if accepted >= q: break
                actual_groups = engine.observed[lane]
                same_rounds = [ids for ids,_ in actual_groups] == followed
                maxerr = max((abs(v-row['scores'][i]) for _, scores in actual_groups
                              for i,v in scores.items()), default=0)
                expected_tuples = [tuple(x[:3]) for x in expected]
                actual_tuples = [tuple(x[:3]) for x in actual]
                same_ranks = actual_tuples == expected_tuples
                selected_error = max((abs(e.score-a.score) for e,a in zip(expected,actual)
                                      if e.item_id == a.item_id and e.score is not None and a.score is not None),
                                     default=0)
                results.append(dict(case=name,config=cfg,row_id=row['row_id'],rounds=len(actual_groups),
                                    draft_counts=[len(items) for items,_ in actual_groups],
                                    exact_round_candidates=same_rounds,exact_rank_item_source_round=same_ranks,
                                    drafted_score_max_abs_error=maxerr,selected_score_max_abs_error=selected_error,
                                    accepted=accepted,retained_retrieval=sum(x.source=='retrieval' for x in actual),
                                    expected=expected_tuples,observed=actual_tuples))
    passed = all(r['exact_round_candidates'] and r['exact_rank_item_source_round']
                 and max(r['drafted_score_max_abs_error'],r['selected_score_max_abs_error']) < 1e-4
                 for r in results)
    passed = passed and any(r['rounds']==3 and r['retained_retrieval']>0 for r in results)
    output = dict(status='passed' if passed else 'failed',batch_size=4,rho=1,
                  comparisons=len(results),cases=[x[0] for x in settings],checks=results)
    dump(args.run_dir/'rho_smoke.json', output)
    print(output['status'], len(results), flush=True)
    if not passed: raise RuntimeError('Live rho=1 routing parity check requires investigation')


if __name__ == '__main__':
    main()
