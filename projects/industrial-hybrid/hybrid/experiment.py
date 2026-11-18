"""Generate exact rollout primitives for the Industrial inference grid and serve live requests."""
import argparse
import ast
import csv
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

DRAFTERS = ('ann', 'unisrec', 'qwen')
KS = (1, 5, 10, 20, 50)
WIDTHS = (1, 5, 10, 20, 50)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def split_prefix(prefix):
    atoms = tuple(re.findall(r'<[abc]_\d+>', prefix))
    if not atoms or ''.join(atoms) != prefix:
        raise ValueError(f'Invalid SID prefix {prefix!r}')
    return atoms


def draft_size(k, rho):
    ratio = Fraction(str(rho))
    if ratio <= 0:
        raise ValueError('rho must be positive')
    return max(1, int(k * ratio))


class Data:
    def __init__(self, assets, expected_unique_sids=3670):
        self.assets = Path(assets)
        self.catalog = {int(k): tuple(v) for k,v in json.loads((self.assets/'sid.json').read_text()).items()}
        self.sids = {i: ''.join(t) for i,t in self.catalog.items()}
        self.by_sid = defaultdict(list)
        for i,sid in self.sids.items():
            self.by_sid[sid].append(i)
        if len(self.catalog) != 3686 or len(self.by_sid) != expected_unique_sids or set(self.sids) != set(range(3686)):
            raise ValueError("Industrial catalog size, item IDs, or unique SID count differs")
        if not all(len(v) == 3 for v in self.catalog.values()):
            raise ValueError("Industrial inference requires three tokens per SID")
        freq, warm = Counter(), set()
        for row in self.read('train'):
            freq[row['target']] += 1
            warm.update(row['history'])
            warm.add(row['target'])
        self.record = dict(sid_by_item=[self.sids[i] for i in range(3686)],
                           target_freq=[freq[i] for i in range(3686)], cold=[i not in warm for i in range(3686)])

    def read(self, split):
        result=[]
        with (self.assets/'data'/f'{split}.csv').open() as f:
            for i,row in enumerate(csv.DictReader(f)):
                hist = [int(x) for x in ast.literal_eval(row['history_item_id'])]
                sids = [self.sids[x] for x in hist]
                target = int(row['item_id'])
                if (ast.literal_eval(row['history_item_sid'])!=sids or row['item_sid']!=self.sids[target]
                    or not hist):
                    raise ValueError(f'Catalog alignment error at {split}:{i}')
                result.append(dict(row_id=i,history=hist,history_sids=sids,target=target))
        return result

    def build_drafter(self, name, device):
        from hybrid.drafters import build_drafter

        return build_drafter(name, self.catalog, device=device, assets=self.assets)


class Engine:
    def __init__(self, data, device, names=DRAFTERS):
        import torch
        from hybrid.verifier import Verifier
        self.data, self.device = data, device
        self.v = Verifier(data.assets/'rl', data.catalog, device=device, chunk_size=64)
        self.d = {name: data.build_drafter(name, device) for name in names}
        self.clock = Counter()
        self.work = Counter()
        self.profiling = False
        torch.set_grad_enabled(False)

    def run_stage(self, key, fn, *args, **kwargs):
        if not self.profiling:
            return fn(*args, **kwargs)
        import torch
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=fn(*args, **kwargs)
        torch.cuda.synchronize()
        self.clock[key] += time.perf_counter()-start
        return result

    def request_drafts(self, d, query, allowed, excluded, limit):
        result=list(map(int,d.draft(query,allowed,excluded,limit)))
        if len(result)>limit or len(result)!=len(set(result)) or set(result)&excluded:
            raise ValueError('Draft size or uniqueness violation')
        for item in result:
            sid = self.data.catalog[item]
            if allowed is not None and not any(sid[:len(p)]==p for p in allowed):
                raise ValueError('Drafter returned an item outside allowed prefixes')
        return result

    def get_scores(self, session, rows, items):
        sid_lists=[list(dict.fromkeys(self.data.sids[i] for i in ids)) for ids in items]
        values=self.v.score(session,sid_lists,rows=rows)
        output=[]
        for ids,sids,scores in zip(items,sid_lists,values,strict=True):
            mapping=dict(zip(sids,scores,strict=True))
            output.append({i:float(mapping[self.data.sids[i]]) for i in ids})
        return output

    def primitives(self, examples, smoke=False, *, widths=WIDTHS):
        history=[r['history'] for r in examples]
        prompt_ids=self.v.prompts([r['history_sids'] for r in examples])
        session=self.v.prefill(prompt_ids)
        beam=self.v.beam(prompt_ids)
        prefixes = {w: {depth:[[split_prefix(s) for s in group] for group in self.v.prefixes(session,w,depth)]
                        for depth in (1,2)} for w in widths}
        drafts=[{} for _ in examples]
        touched=[{i for sid in top for i in self.data.by_sid[sid]} for top in beam]
        for name,d in self.d.items():
            queries=d.prepare(history)
            for j,query in enumerate(queries):
                grid={}
                for k in KS:
                    first=self.request_drafts(d,query,None,set(),2*k)
                    candidates={'first':first}
                    touched[j].update(first)
                    for w in widths:
                        excluded=set(first)
                        rounds=[]
                        for depth in (1,2):
                            ids=self.request_drafts(d,query,prefixes[w][depth][j],excluded,2*k)
                            rounds.append(ids)
                            excluded.update(ids)
                            touched[j].update(ids)
                        candidates[str(w)]=rounds
                    grid[str(k)]=candidates
                drafts[j][name]=grid
        items=[sorted(x) for x in touched]
        scores=self.get_scores(session,list(range(len(examples))),items)
        if smoke:
            sid_lists=[[self.data.sids[i] for i in x[:7]] for x in items]
            reference=self.v.bruteforce(prompt_ids,sid_lists)
            errors=[abs(scores[j][i]-r) for j,g in enumerate(reference) for i,r in zip(items[j][:7],g,strict=True)]
            if max(errors) > .12:
                raise ValueError(f'Cached teacher forcing differs from reference: {max(errors)}')
            print('teacher_forcing_max_abs_error',max(errors),flush=True)
        return [dict(row_id=r['row_id'],target=r['target'],beam=beam[j],scores=scores[j],drafts=drafts[j])
                for j,r in enumerate(examples)]

    def online(self, examples, cfg, catalog, audit=False):
        from hybrid.sweep import merge_selected
        k,quota=cfg['k'],cfg['quota']
        max_rounds=cfg.get('max_rounds',3)
        if not (1 <= k <= 50 and 0 <= quota <= k and max_rounds in (1,2,3)):
            raise ValueError('Require K in 1..50, quota in 0..K and one to three draft rounds')
        delta=draft_size(k,cfg.get('rho',2))
        pids=self.run_stage('verifier_tokenize',self.v.prompts,[x['history_sids'] for x in examples])
        count=len(examples)
        if quota==0:
            beams=self.run_stage('full_beam',self.v.beam,pids)
            return [merge_selected([],b,{},catalog,k,0) for b in beams]
        ses=self.run_stage('verifier_prefill',self.v.prefill,pids)
        drafter=self.d[cfg['drafter']]
        queries=self.run_stage('drafter_query',drafter.prepare,[x['history'] for x in examples])
        score_maps=[{} for _ in examples]
        accepted=[[] for _ in examples]
        seen=[set() for _ in examples]
        theta=cfg['threshold'] if cfg['threshold'] is not None else -float('inf')
        active=list(range(count))
        for depth in range(max_rounds):
            if not active: break
            prefixes=([None]*len(active) if depth==0 else
                      [[split_prefix(p) for p in group] for group in self.run_stage(
                          'prefix_beam',self.v.prefixes,ses,cfg['prefix_beams'],depth,rows=active)])
            ids=[self.run_stage('draft_search',self.request_drafts,drafter,queries[j],pf,seen[j],delta)
                 for j,pf in zip(active,prefixes,strict=True)]
            values=self.run_stage('verification',self.get_scores,ses,active,ids)
            for j,group,sc in zip(active,ids,values,strict=True):
                seen[j].update(group);score_maps[j].update(sc)
                accepted[j].extend((i,depth+1) for i in group if sc[i]>theta)
                self.work['drafted']+=len(group)
                self.work['rounds']+=1
            # An empty prefix-restricted round does not necessarily exhaust a
            # different width/depth frontier. Empty global inventory does.
            active=[j for j in active if len(accepted[j])<quota and len(seen[j])<len(self.data.sids)]
        need=[j for j in range(count) if min(quota,len(accepted[j]))<k]
        beams=[[] for _ in examples]
        if need:
            generated=self.run_stage('full_beam',self.v.beam_from_session,ses,rows=need)
            for j,b in zip(need,generated,strict=True): beams[j]=b
        rescore=[j for j in need if accepted[j]]
        if rescore:
            from hybrid.sweep import merge_selected as merge
            # Dummy scores here only identify beam members selected by original
            # rank; accepted retrieval candidates retain their real scores.
            missing=[]
            for j in rescore:
                provisional={i:0. for sid in beams[j] for i in self.data.by_sid[sid]}
                provisional.update(score_maps[j])
                provisional_items=merge(accepted[j],beams[j],provisional,catalog,k,quota)
                missing.append([i for i,source,*_ in provisional_items if source=='beam' and i not in score_maps[j]])
            positions=[n for n,x in enumerate(missing) if x]
            if positions:
                rows=[rescore[n] for n in positions]
                fresh=self.run_stage('beam_rescore',self.get_scores,ses,rows,[missing[n] for n in positions])
                for j,sc in zip(rows,fresh,strict=True): score_maps[j].update(sc)
        return [merge_selected(accepted[j],beams[j],score_maps[j],catalog,k,quota) for j in range(count)]


def run_quality(args,data,engine):
    run=Path(args.run_dir)
    if args.rank==0:
        dump(run/'catalog.json',data.record)
    for split in args.splits:
        all_rows=data.read(split)
        rows=all_rows[args.rank::args.world]
        if args.limit: rows=rows[:args.limit]
        target=run/split/f'rank{args.rank}.jsonl'
        target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_suffix('.partial')
        if target.is_file(): continue
        reference=None
        if split=='test':
            with (data.assets/'legacy_test.jsonl').open() as f:
                reference={r['row_id']:r['predictions'] for r in map(json.loads,f)}
        started=time.perf_counter()
        with pending.open('w') as f:
            for n in range(0,len(rows),4):
                records=engine.primitives(rows[n:n+4],smoke=(n==0 and args.rank==0))
                for rec in records:
                    if reference is not None and rec['beam']!=reference[rec['row_id']]:
                        raise ValueError(f"Historical beam order mismatch at {split}:{rec['row_id']}")
                    f.write(json.dumps(rec,separators=(',',':'),allow_nan=False)+'\n')
                if n%40==0 or n+4>=len(rows):
                    f.flush()
                    dump(run/f'progress-{args.rank}.json',dict(split=split,done=min(n+4,len(rows)),total=len(rows),
                        seconds=time.perf_counter()-started))
        pending.replace(target)


def main():
    import torch
    ap=argparse.ArgumentParser()
    ap.add_argument('--assets',default='/assets')
    ap.add_argument('--run-dir',default='/run/hybrid')
    ap.add_argument('--rank',type=int,default=int(os.getenv('RANK','0')))
    ap.add_argument('--world',type=int,default=int(os.getenv('WORLD_SIZE','1')))
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--splits',nargs='+',choices=['valid','test'],default=['valid','test'])
    args=ap.parse_args()
    torch.cuda.set_device(args.rank)
    torch.set_num_threads(2)
    data=Data(args.assets)
    run_quality(args,data,Engine(data,f'cuda:{args.rank}'))


if __name__=='__main__':
    main()
