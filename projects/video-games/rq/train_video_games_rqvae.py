#!/usr/bin/env python3
"""Train the original Video Games RQ-VAE with grouped holdout and resumable early stopping."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,os,random,sys,time
from datetime import datetime,timezone
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime/python_deps'))
sys.path.append(str(ROOT))
import npu_support
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from models.rqvae import RQVAE


def atomic_replace(source,target):
    for attempt in range(30):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt==29:raise
            time.sleep(min(.1*(attempt+1),.5))


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');atomic_replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def build_model(args,dim=2560):
    return RQVAE(in_dim=dim,num_emb_list=args.num_emb_list,e_dim=args.e_dim,layers=args.layers,dropout_prob=args.dropout_prob,bn=args.bn,loss_type=args.loss_type,quant_loss_weight=args.quant_loss_weight,beta=args.beta,kmeans_init=args.kmeans_init,kmeans_iters=args.kmeans_iters,sk_epsilons=args.sk_epsilons,sk_iters=args.sk_iters)


def restore_model(ckpt,device='cuda:0'):
    model=build_model(ckpt['args'],ckpt.get('input_dim',2560));model.load_state_dict(ckpt['state_dict'])
    for q,initialized in zip(model.rq.vq_layers,ckpt.get('quantizer_initialized',[True]*3)):q.initted=bool(initialized)
    return model.to(device)


def cleaned_texts(metadata):
    spec=importlib.util.spec_from_file_location('embedding_text_utils',ROOT/'rq/text2emb/utils.py')
    util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
    values=[]
    for i in range(len(metadata)):
        feature=metadata[str(i)];parts=[util.clean_text(feature[key]).strip() for key in ['title','description']]
        values.append(' '.join(part for part in parts if part) or 'unknown item')
    return values


def initialize(model,data,train_ids,args):
    model.eval()
    with torch.no_grad():latent=torch.cat([model.encoder(data[train_ids[start:start+1024]]) for start in range(0,len(train_ids),1024)])
    residual=latent.detach().cpu().numpy();records=[]
    for level,q in enumerate(model.rq.vq_layers):
        print(f'Initializing codebook {level+1}: KMeans on {len(residual)} training residuals',flush=True)
        km=KMeans(n_clusters=q.n_e,init='k-means++',n_init=args.kmeans_n_init,max_iter=args.kmeans_iters,random_state=args.seed+level).fit(residual)
        with torch.no_grad():q.embedding.weight.copy_(torch.from_numpy(km.cluster_centers_).to(data.device));q.initted=True
        residual=residual-km.cluster_centers_[km.labels_]
        records.append({'level':level+1,'training_items':len(train_ids),'k':q.n_e,'n_init':args.kmeans_n_init,'max_iter':args.kmeans_iters,'seed':args.seed+level,'iterations':int(km.n_iter_),'inertia':float(km.inertia_),'residual_mse_after':float(np.mean(residual**2))})
    return records


@torch.no_grad()
def evaluate(model,data,split,baseline):
    model.eval();all_mse=[];all_cos=[];all_codes=[]
    for start in range(0,len(data),1024):
        batch=data[start:start+1024];out,_,codes=model(batch,use_sk=False)
        all_mse.append(((out-batch)**2).mean(1).cpu().numpy());all_cos.append(torch.nn.functional.cosine_similarity(out,batch,dim=1).cpu().numpy());all_codes.append(codes.cpu().numpy())
    mse=np.concatenate(all_mse);cos=np.concatenate(all_cos);codes=np.concatenate(all_codes)
    result={}
    for name,ids in [('train',split['train_ids']),('val',split['val_ids']),('all',list(range(len(data))))]:
        idx=np.asarray(ids,dtype=int);code=codes[idx];n=len(idx);usage=[]
        for level in range(3):
            counts=np.bincount(code[:,level],minlength=256);p=counts[counts>0]/n;entropy=float(-(p*np.log(p)).sum())
            usage.append({'used':int((counts>0).sum()),'dead':int((counts==0).sum()),'entropy':entropy,'perplexity':math.exp(entropy),'max_share':float(counts.max()/n),'counts':counts.tolist()})
        unique=len(np.unique(code,axis=0));m=float(mse[idx].mean())
        result[name]={'mse':m,'cosine_mean':float(cos[idx].mean()),'baseline_mse':baseline[name],'nmse':m/baseline[name],'unique_sids':unique,'collision_rate':1-unique/n,'codebook':usage}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=50000);p.add_argument('--min-epochs',type=int,default=3000)
    p.add_argument('--eval-step',type=int,default=50);p.add_argument('--patience',type=int,default=30);p.add_argument('--relative-improvement',type=float,default=.001)
    p.add_argument('--batch-size',type=int,default=20480);p.add_argument('--device',default='cuda:0');p.add_argument('--resume',type=Path)
    p.add_argument('--disable-early-stop',action='store_true');p.add_argument('--seed',type=int,default=2024);p.add_argument('--kmeans-n-init',type=int,default=10)
    cli=p.parse_args();run=cli.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    data_dir=ROOT/'data/Amazon18/Video_Games';embedding=data_dir/'Video_Games.emb-qwen-td.npy';metadata_path=data_dir/'Video_Games.item.json'
    args=argparse.Namespace(data_path=str(embedding),lr=1e-3,epochs=cli.epochs,batch_size=cli.batch_size,num_workers=0,eval_step=cli.eval_step,learner='AdamW',lr_scheduler_type='constant',warmup_epochs=50,weight_decay=0.,dropout_prob=0.,bn=False,loss_type='mse',kmeans_init=True,kmeans_iters=100,kmeans_n_init=cli.kmeans_n_init,sk_epsilons=[0.,0.,0.],sk_iters=50,device=cli.device,num_emb_list=[256,256,256],e_dim=32,quant_loss_weight=1.,beta=.25,layers=[2048,1024,512,256,128,64],ckpt_dir=str(run),seed=cli.seed,min_epochs=cli.min_epochs,patience=cli.patience,relative_improvement=cli.relative_improvement,disable_early_stop=cli.disable_early_stop)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    x=np.load(embedding,allow_pickle=False);assert x.shape==(16572,2560) and x.dtype==np.float32 and np.isfinite(x).all()
    source_hash=digest(embedding);metadata=json.loads(metadata_path.read_text(encoding='utf-8'));texts=cleaned_texts(metadata)
    train_ids,val_ids=next(GroupShuffleSplit(n_splits=1,test_size=.1,random_state=args.seed).split(np.arange(len(x)),groups=np.asarray(texts,dtype=object)))
    assert (len(train_ids),len(val_ids))==(14919,1653)
    assert not set(texts[i] for i in train_ids)&set(texts[i] for i in val_ids)
    split={'train_ids':train_ids.tolist(),'val_ids':val_ids.tolist(),'seed':args.seed,'validation_fraction_requested':.1,'group_key':'exact cleaned title+description text','identical_text_group_overlap':0,'embedding_sha256':source_hash}
    if (run/'split.json').exists():assert json.loads((run/'split.json').read_text(encoding='utf-8'))==split
    else:write_json(run/'split.json',split)
    data=torch.from_numpy(x).to(args.device);train_index=torch.as_tensor(train_ids,device=data.device)
    mean=data[train_index].mean(0)
    baseline={name:float(((data[torch.as_tensor(ids,device=data.device)]-mean)**2).mean().item()) for name,ids in [('train',train_ids),('val',val_ids),('all',np.arange(len(x)))]}
    manifest_path=run/'manifest.json'
    manifest={'run_dir':str(run),'status':'training','started_at_utc':datetime.now(timezone.utc).isoformat(),'embedding_sha256':source_hash,'config':vars(args),'split_counts':{'train':len(train_ids),'val':len(val_ids),'all':len(x)},'artifacts':{'checkpoint':str(run/'best_val_model.pth'),'initial_checkpoint':str(run/'initial_model.pth'),'split':str(run/'split.json'),'embeddings':str(embedding),'item_metadata':str(metadata_path),'item2id':str(data_dir/'Video_Games.item2id'),'epoch_metrics':str(run/'epoch_metrics.csv'),'evaluation_history':str(run/'evaluation_history.json'),'training_summary':str(run/'training_summary.json')}}
    start=time.perf_counter();history=[];best_val=float('inf');best_epoch=0;best_collision=float('inf');significant_best=float('inf');stale=0;first_epoch=1;initialization=[]
    if cli.resume:
        ckpt=torch.load(cli.resume,map_location='cpu',weights_only=False)
        assert ckpt['embedding_sha256']==source_hash and ckpt['split']==split
        start=time.perf_counter()-float(ckpt.get('elapsed_seconds',ckpt['evaluation_history'][-1]['elapsed_seconds']))
        model=restore_model(ckpt,args.device);initialization=ckpt['initialization'];history=ckpt['evaluation_history'];first_epoch=ckpt['epoch']+1;best_val=ckpt['best_val'];best_epoch=ckpt['best_epoch'];best_collision=ckpt['best_collision'];significant_best=ckpt['significant_best'];stale=ckpt['stale']
    else:
        model=build_model(args).to(args.device);initialization=initialize(model,data,train_index,args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    steps_per_epoch=math.ceil(len(train_ids)/args.batch_size);warmup_steps=50*steps_per_epoch
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(step/max(1,warmup_steps),1.))
    if cli.resume:
        optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
        random.setstate(ckpt['rng']['python']);np.random.set_state(ckpt['rng']['numpy']);torch.set_rng_state(ckpt['rng']['torch']);torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        print(f'Resumed at epoch {first_epoch}; initialized={[q.initted for q in model.rq.vq_layers]}',flush=True)
    def save_checkpoint(name,epoch):
        payload={'args':args,'input_dim':2560,'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'quantizer_initialized':[q.initted for q in model.rq.vq_layers],'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'split':split,'embedding_sha256':source_hash,'initialization':initialization,'evaluation_history':history,'best_val':best_val,'best_epoch':best_epoch,'best_collision':best_collision,'significant_best':significant_best,'stale':stale,'elapsed_seconds':time.perf_counter()-start}
        target=run/name;tmp=target.with_name(target.name+'.tmp');torch.save(payload,tmp);atomic_replace(tmp,target)
    if not cli.resume:
        initial_eval=evaluate(model,data,split,baseline);history.append({'epoch':0,'train_total':None,'train_recon':None,'train_quant':None,'lr':0.,'elapsed_seconds':time.perf_counter()-start,**initial_eval});save_checkpoint('initial_model.pth',0);write_json(run/'evaluation_history.json',history)
    if cli.resume and manifest_path.exists():
        previous_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['started_at_utc']=previous_manifest['started_at_utc']
        manifest['resume_events']=previous_manifest.get('resume_events',[])+[{'checkpoint':str(cli.resume),'first_epoch':first_epoch,'resumed_at_utc':datetime.now(timezone.utc).isoformat()}]
    manifest['initialization']=initialization;manifest['baseline_mse']=baseline;write_json(manifest_path,manifest)
    epoch_file=run/'epoch_metrics.csv';append=cli.resume is not None and epoch_file.exists()
    if append:
        with epoch_file.open(encoding='utf-8',newline='') as old_stream:
            old_reader=csv.DictReader(old_stream);fieldnames=old_reader.fieldnames
            retained=[row for row in old_reader if int(row['epoch'])<first_epoch]
        replacement=epoch_file.with_name(epoch_file.name+'.tmp')
        with replacement.open('w',encoding='utf-8',newline='') as output:
            w=csv.DictWriter(output,fieldnames=fieldnames);w.writeheader();w.writerows(retained)
        atomic_replace(replacement,epoch_file)
    stream=epoch_file.open('a' if append else 'w',encoding='utf-8',newline='');writer=csv.DictWriter(stream,fieldnames=['epoch','train_total','train_recon','train_quant','lr','epoch_seconds','elapsed_seconds'])
    if not append:writer.writeheader()
    print(f'Training {len(train_ids)} items; holdout {len(val_ids)}; max_epochs={args.epochs}; batch={args.batch_size}',flush=True)
    stop_reason='maximum_epochs';last_epoch=first_epoch-1;torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(first_epoch,args.epochs+1):
            tick=time.perf_counter();model.train();order=train_index[torch.randperm(len(train_index),device=data.device)];sums=np.zeros(3,dtype=float)
            for pos in range(0,len(order),args.batch_size):
                batch=data[order[pos:pos+args.batch_size]];optimizer.zero_grad(set_to_none=True)
                out,quant,_=model(batch,use_sk=False);loss,recon=model.compute_loss(out,quant,xs=batch)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
                sums+=np.asarray([loss.item(),recon.item(),quant.item()])*len(batch)
            sums/=len(train_ids);row={'epoch':epoch,'train_total':float(sums[0]),'train_recon':float(sums[1]),'train_quant':float(sums[2]),'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-start};writer.writerow(row);last_epoch=epoch
            if epoch%args.eval_step==0 or epoch==args.epochs:
                metrics=evaluate(model,data,split,baseline);history.append({**row,**metrics});current=metrics['val']['mse'];collision=metrics['all']['collision_rate']
                if current<best_val:best_val=current;best_epoch=epoch;improved=True
                else:improved=False
                if current<significant_best*(1-args.relative_improvement):significant_best=current;stale=0
                else:stale+=1
                collision_improved=collision<best_collision
                if collision_improved:best_collision=collision
                if improved:save_checkpoint('best_val_model.pth',epoch)
                if collision_improved:save_checkpoint('best_collision_model.pth',epoch)
                save_checkpoint('last_model.pth',epoch);stream.flush();write_json(run/'evaluation_history.json',history)
                progress={'status':'training','epoch':epoch,'max_epochs':args.epochs,'best_epoch':best_epoch,'best_val_mse':best_val,'validation_mse':current,'validation_nmse':metrics['val']['nmse'],'raw_collision_rate':collision,'stale_evaluations':stale,'elapsed_seconds':time.perf_counter()-start,'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                write_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
                if not args.disable_early_stop and epoch>=args.min_epochs and stale>=args.patience:stop_reason='validation_plateau';break
    finally:stream.close()
    best=torch.load(run/'best_val_model.pth',map_location='cpu',weights_only=False);selected=restore_model(best,args.device);selected_metrics=evaluate(selected,data,split,baseline)
    summary={'status':'completed','stop_reason':stop_reason,'epochs_completed':last_epoch,'best_epoch':best_epoch,'best_val_mse':best_val,'config':vars(args),'split_counts':manifest['split_counts'],'initialization':initialization,'baseline_mse':baseline,'selected_metrics':selected_metrics,'training_seconds':time.perf_counter()-start,'gpu':torch.cuda.get_device_name(0),'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'embedding_sha256':source_hash,'selected_checkpoint':str(run/'best_val_model.pth'),'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(run/'training_summary.json',summary);write_json(run/'progress.json',summary);manifest.update(status='trained',completed_at_utc=summary['completed_at_utc']);write_json(manifest_path,manifest)
    print('TRAINING_COMPLETE '+json.dumps({k:summary[k] for k in ['stop_reason','epochs_completed','best_epoch','best_val_mse','training_seconds']}),flush=True)


if __name__=='__main__':main()
