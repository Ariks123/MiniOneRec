"""Check live vs cached routing and test the frozen native beam on the actual GPU."""
import argparse
import json
from pathlib import Path
import torch
from .experiment import Data, Engine, dump
from .sweep import Catalog, merge_selected


def replay(row,cfg,catalog):
    name,k,quota,b=cfg['drafter'],cfg['k'],cfg['quota'],str(cfg['prefix_beams'])
    accepted=[]
    threshold=cfg['threshold'] if cfg['threshold'] is not None else -float('inf')
    if quota:
        rounds=row['drafts'][name][str(k)]
        for depth,items in enumerate((rounds['first'],*rounds[b]),1):
            accepted.extend((i,depth) for i in items if row['scores'][i]>threshold)
            if len(accepted)>=quota: break
    return merge_selected(accepted,row['beam'],row['scores'],catalog,k,quota)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',default='/run/hybrid',type=Path)
    ap.add_argument('--assets',default='/assets',type=Path)
    a=ap.parse_args()
    torch.cuda.set_device(0)
    data=Data(a.assets)
    cat=Catalog.from_dict(data.record)
    engine=Engine(data,'cuda:0')
    examples=data.read('test')[:16:4]
    primitives=engine.primitives(examples,smoke=True)
    with (a.assets/'legacy_test.jsonl').open() as f:
        frozen={r['row_id']:r['predictions'] for r in map(json.loads,f)}
    assert all(r['beam']==frozen[r['row_id']] for r in primitives)
    checks=[]
    for d in ('ann','unisrec','qwen'):
        for k,q,w,t in [(1,0,1,None),(10,4,5,None),(10,4,5,-2.),(5,5,20,None),(5,5,1,0.)]:
            cfg=dict(drafter=d,k=k,quota=q,prefix_beams=w,threshold=t)
            live=engine.online(examples,cfg,cat)
            for i,(row,observed) in enumerate(zip(primitives,live)):
                expected=replay(row,cfg,cat)
                left=[(x.item_id,x.source,x.round) for x in expected]
                right=[(x.item_id,x.source,x.round) for x in observed]
                err=max((abs(x.score-y.score) for x,y in zip(expected,observed)
                         if x.score is not None and y.score is not None),default=0.)
                checks.append(dict(**cfg,row_id=row['row_id'],exact=left==right,score_abs_error=err))
                if left!=right:
                    print('live vs offline difference',cfg,row['row_id'],left,right,flush=True)
    report={'status':'passed' if all(x['exact'] for x in checks) else 'failed',
            'rows':[r['row_id'] for r in examples], 'comparisons':len(checks), 'checks':checks}
    dump(a.run_dir/'audit.json',report)
    if report['status']!='passed':
        raise ValueError('Online/replay ordering or source differs; see audit.json')
    print('passed:',len(checks),'request/config comparisons',flush=True)

if __name__=='__main__':
    main()
