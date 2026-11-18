"""Measure live Industrial inference on fixed test requests, with durable per-config results.

torchrun --standalone --nproc-per-node=4 -m hybrid.benchmark --run-dir /run/hybrid
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from hybrid.experiment import Data, Engine, dump
from hybrid.sweep import Catalog


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def timed_batches(engine, catalog, examples, cfg, batch_size, repeats, latency=False):
    latencies, repetition_times = [], []
    items = 0
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for offset in range(0,len(examples),batch_size):
            t = time.perf_counter() if latency else None
            outputs = engine.online(examples[offset:offset+batch_size],cfg,catalog)
            items += sum(len(r) for r in outputs)
            if latency:
                torch.cuda.synchronize()
                latencies.append((time.perf_counter()-t)*1000)
        torch.cuda.synchronize()
        repetition_times.append(time.perf_counter()-started)
    return repetition_times,items,latencies


def initialize_arm(engine, name):
    active = list(engine.d)
    wanted = [] if name == 'beam' else [name]
    if active != wanted:
        engine.d.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if wanted:
            engine.d[name] = engine.data.build_drafter(name, engine.device)
    torch.cuda.synchronize()


def warmup(engine,catalog,examples,cfg,batch_size,steps):
    for i in range(steps):
        offset = (i*batch_size) % len(examples)
        group=[examples[(offset+j)%len(examples)] for j in range(min(batch_size,len(examples)))]
        engine.online(group,cfg,catalog)
    torch.cuda.synchronize()


def single(engine,catalog,examples,cfg,args):
    warmup(engine,catalog,examples,cfg,1,args.warmup)
    torch.cuda.reset_peak_memory_stats()
    mem_static=torch.cuda.memory_allocated()
    durations,items,lat = timed_batches(engine,catalog,examples,cfg,1,args.repeats,latency=True)
    peak1 = torch.cuda.max_memory_allocated()
    warmup(engine,catalog,examples,cfg,4,args.warmup)
    torch.cuda.reset_peak_memory_stats()
    engine.work.clear()
    engine.v.counters.clear()
    times,items4,_ = timed_batches(engine,catalog,examples,cfg,4,args.repeats)
    peak4=torch.cuda.max_memory_allocated()
    metrics=dict(engine.v.counters)
    work=dict(engine.work)
    n=len(examples)*args.repeats
    timing=sum(times)
    percent=np.percentile(lat,[50,95,99]).tolist()
    # Stage timers synchronize independently. Their overhead is excluded from
    # the preceding end-to-end results and the throughput denominator.
    prof=examples[:min(16,len(examples))]
    engine.profiling=True
    engine.clock.clear()
    try:
        for off in range(0,len(prof),4):
            engine.online(prof[off:off+4],cfg,catalog)
    finally:
        engine.profiling=False
    return {
        'latency_p50_ms':percent[0],'latency_p95_ms':percent[1],'latency_p99_ms':percent[2],
        'requests_s_1gpu':n/timing,'items_s_1gpu':items4/timing,
        'latency_samples':len(lat),'latency_repeat_seconds':durations,'single_repeat_seconds':times,
        'logical_requests_1gpu':n,'recommendations_batch1':items,'recommendations_1gpu':items4,
        'verifier_forward_calls_1gpu':metrics.get('forward_calls',0),
        'verifier_input_tokens_s_1gpu':metrics.get('model_input_tokens_padded',0)/timing,
        'logical_prompt_tokens_s_1gpu':metrics.get('prefill_tokens',0)/timing,
        'drafted_sid_tokens_s_1gpu':3*work.get('drafted',0)/timing,
        'recommended_sid_tokens_s_1gpu':3*items4/timing,
        'verifier_counters_1gpu':metrics,'retrieval_counters_1gpu':work,
        'gpu_resident_model_bytes':mem_static,'gpu_peak_bytes_batch1':peak1,'gpu_peak_bytes_batch4':peak4,
        'profile_requests':len(prof),'profile_batch4_stage_ms_per_request':
            {stage:seconds*1000/len(prof) for stage,seconds in engine.clock.items()},
    }


def multi(engine,catalog,examples,cfg,args):
    rank,world=dist.get_rank(),dist.get_world_size()
    local=examples[rank::world]
    warmup(engine,catalog,local,cfg,4,args.warmup)
    torch.cuda.reset_peak_memory_stats()
    engine.v.counters.clear()
    engine.work.clear()
    elapsed=[]
    total_items=0
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        dist.barrier()
        times,items,_=timed_batches(engine,catalog,local,cfg,4,1)
        outcomes=[None]*world
        dist.all_gather_object(outcomes,(times[0],items))
        elapsed.append(max(s for s,_ in outcomes))
        total_items+=sum(count for _,count in outcomes)
    info=[None]*world
    dist.all_gather_object(info,(torch.cuda.max_memory_allocated(),dict(engine.v.counters),dict(engine.work)))
    n=len(examples)*args.repeats
    counters={key:sum(c.get(key,0) for _,c,_ in info) for key in set().union(*(set(c) for _,c,_ in info))}
    work={key:sum(c.get(key,0) for _,_,c in info) for key in set().union(*(set(c) for _,_,c in info))}
    return {'requests_s_4gpu':n/sum(elapsed),'items_s_4gpu':total_items/sum(elapsed),
            'logical_requests_4gpu':n,'recommendations_4gpu':total_items,'four_gpu_repeat_seconds':elapsed,
            'verifier_counters_4gpu':counters,'retrieval_counters_4gpu':work,
            'logical_prompt_tokens_s_4gpu':counters.get('prefill_tokens',0)/sum(elapsed),
            'drafted_sid_tokens_s_4gpu':3*work.get('drafted',0)/sum(elapsed),
            'recommended_sid_tokens_s_4gpu':3*total_items/sum(elapsed),
            'peak_bytes_per_rank_batch4':[x[0] for x in info],
            'verifier_input_tokens_s_4gpu':counters.get('model_input_tokens_padded',0)/sum(elapsed)}


def saved(path, identity):
    if not path.is_file():
        return None
    record=json.loads(path.read_text())
    if record.get('identity')!=identity:
        raise ValueError(f"Saved benchmark does not match active model/protocol: {path}")
    return record


def merge(configs,state,report,identity):
    result=[]
    for cfg in configs:
        entries=[saved(state/mode/(cfg['id']+'.json'),identity) for mode in ('single','multi')]
        if None in entries:
            return False
        result.append({**cfg,**entries[0]['measurement'],**entries[1]['measurement']})
    dump(report/'benchmarks.json',result)
    return True


def progress(state,mode,rank,done,count,start,last=None):
    elapsed=time.monotonic()-start
    row={'mode':mode,'rank':rank,'done':done,'total':count,'seconds':elapsed,'last':last,
         'estimated_seconds_remaining':(count-done)*elapsed/done if done else None}
    dump(state/f'progress-{mode}-{rank}.json',row)
    print(json.dumps(row,separators=(',',':')),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=Path('/assets'))
    parser.add_argument('--mode',choices=['single','multi','both','merge'],default='both')
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--rqvae', action='store_true', help='Use sealed RQ-VAE and retrained UniSRec assets')
    args=parser.parse_args()
    if args.samples<=0 or args.repeats<=0 or args.warmup<1:
        parser.error('Benchmark sample size, repeats and warmup must be positive')
    report=args.run_dir/'report'
    configs=json.loads((report/'benchmark_configs.json').read_text())
    if not configs or len({cfg['id'] for cfg in configs}) != len(configs):
        raise ValueError("Benchmark requires distinct validation-selected configurations")
    if args.samples>4533:
        raise ValueError("More requests than available test rows")
    indices=random.Random(42).sample(range(4533),args.samples)
    source=Path(__file__).parent
    protocol={'sample_row_ids':indices,'seed':42,'repeats':args.repeats,'warmup_batches_per_shape':args.warmup,
        'ranking_config_sha256':fingerprint(configs),'source_sha256':
            {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
             ('benchmark.py','experiment.py','verifier.py','drafters.py','sweep.py')},
        'latency_batch_size':1,'throughput_batch_size_per_gpu':4,'latency_unit':'ms/request',
        'timing':'resident model and index, historical item input to ranked response, GPU synchronized',
        'stage_timing':'separate runs, CUDA synchronization enclosing each stage',
        'four_gpu_throughput':'all logical test requests / sum of per-repetition slowest rank times',
        'single_gpu_context':'independent configurations may run concurrently on other GPUs on the same host',
        'verifier_tokens':'actual padded model input tokens processed, including repeated beam tokens',
        'logical_tokens':'prompt counted once per request; each drafted or delivered SID has three tokens',
        'cache_scope':'request-local prompt and candidate branch state only; no cross-request or grid cache'}
    data_factory = Data
    if args.rqvae:
        from hybrid.rqvae_assets import MANIFEST, RQVAEData

        if any(c["drafter"] not in ("beam", "unisrec") for c in configs):
            raise ValueError("RQ-VAE benchmark allows only UniSRec and pure beam configurations")
        protocol["asset_manifest_sha256"] = hashlib.sha256((args.assets / MANIFEST).read_bytes()).hexdigest()
        data_factory = RQVAEData
    identity=fingerprint(protocol)
    state=report/'benchmark_state'/identity
    if args.mode=='merge':
        if not merge(configs,state,report,identity):
            raise RuntimeError("Incomplete measurement files")
        return
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    device=int(os.environ.get('LOCAL_RANK',rank))
    if args.mode in ('multi','both') and (world!=4 or args.samples<4):
        raise ValueError("Four-GPU measurement needs torchrun nproc=4 and at least four requests")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if rank==0:
        dump(report/'benchmark_protocol.json',{**protocol,'identity':identity,'framework':torch.__version__,
            'device':torch.cuda.get_device_name(device)})
    data=data_factory(args.assets)
    test=data.read('test')
    examples=[test[i] for i in indices]
    catalog=Catalog.from_dict(data.record)
    engine=Engine(data,f'cuda:{device}',names=())
    ordered=sorted(configs,key=lambda c:(c['drafter'],c['id']))
    modes=('single','multi') if args.mode=='both' else (args.mode,)
    for mode in modes:
        if mode=='multi' and not dist.is_initialized():
            # Gloo only coordinates clocks, warmup and small metrics. Each GPU
            # hosts an inference replica; no model collective is required.
            dist.init_process_group(backend='gloo',timeout=timedelta(hours=6))
        mine=ordered[rank::world] if mode=='single' else ordered
        start=time.monotonic()
        progress(state,mode,rank,0,len(mine),start)
        for j,cfg in enumerate(mine,1):
            path=state/mode/(cfg['id']+'.json')
            cached=bool(saved(path,identity)) if mode=='single' or rank==0 else False
            if mode=='multi':
                signal=[cached]
                dist.broadcast_object_list(signal,src=0)
                cached=signal[0]
            if not cached:
                initialize_arm(engine,cfg['drafter'])
                measurement=(single if mode=='single' else multi)(engine,catalog,examples,cfg,args)
                if mode=='single' or rank==0:
                    dump(path,{'identity':identity,'configuration':cfg['id'],'measurement':measurement})
            progress(state,mode,rank,j,len(mine),start,cfg['id'])
        if mode=='multi':
            dist.barrier()
            if rank==0 and not merge(configs,state,report,identity):
                print('Four-GPU measurements complete; waiting for single-GPU measurements before merge',flush=True)
            dist.destroy_process_group()


if __name__=='__main__':
    main()
