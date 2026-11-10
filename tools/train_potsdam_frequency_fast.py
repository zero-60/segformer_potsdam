#!/usr/bin/env python3
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,json,math,random,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pf_model import PotsdamModel
from pf_data import FastSpec,FastTrainSet,EpochSampler,worker_init
from pf_evaluate import evaluate

ROOT=Path(__file__).resolve().parent.parent

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def lovasz(raw,labels):
    lab=F.interpolate(labels[:,None].float(),size=raw.shape[-2:],mode='nearest')[:,0].long().flatten()
    prob=raw.float().softmax(1).permute(0,2,3,1).reshape(-1,6)
    losses=[]
    for c in range(6):
        fg=(lab==c).float()
        if not fg.any():continue
        err,idx=(fg-prob[:,c]).abs().sort(descending=True);fg=fg[idx]
        inter=fg.sum()-fg.cumsum(0);union=fg.sum()+(1-fg).cumsum(0)
        grad=1-inter/union.clamp_min(1);grad=torch.cat((grad[:1],grad[1:]-grad[:-1]))
        losses.append(torch.dot(err,grad))
    return torch.stack(losses).mean() if losses else raw.sum()*0

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--protocol',type=Path,default=Path('data/processed/potsdam_frequency_quick_v1'))
    p.add_argument('--raw-root',type=Path,default=Path('data/raw/potsdam'))
    p.add_argument('--rgbir-dir',type=Path,default=Path('data/raw/potsdam/_expanded/4_Ortho_RGBIR'))
    p.add_argument('--pretrained',type=Path,default=Path('../hf-cache/segformer-b2-finetuned-ade-512-512'))
    p.add_argument('--variant',choices=['baseline','frequency'],default='frequency')
    p.add_argument('--out',type=Path)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--val-every',type=int,default=5);p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--base-lr',type=float,default=3e-5);p.add_argument('--new-lr',type=float,default=3e-4)
    p.add_argument('--lovasz-weight',type=float,default=.5);p.add_argument('--cutoff',type=float,default=.15)
    p.add_argument('--no-amp',action='store_true');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache-dir',type=Path,default=Path('data/processed/potsdam_frequency_fast_cache'))
    p.add_argument('--memory',choices=['ram','mmap'],default='ram')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--eval-batch-size',type=int,default=2)
    p.add_argument('--fusion-storage',choices=['ram','disk'],default='ram')
    p.add_argument('--prepare-cache-only',action='store_true')
    p.add_argument('--smoke-test',action='store_true');p.add_argument('--resume',type=Path)
    return p

def resolve_args(a):
    a.project_root=a.project_root.resolve()
    for key in ['protocol','raw_root','rgbir_dir','pretrained','out','resume','cache_dir']:
        v=getattr(a,key)
        if v is not None:setattr(a,key,(a.project_root/v).resolve() if not v.is_absolute() else v.resolve())
    if a.out is None:a.out=a.project_root/f'outputs/training/potsdam_quick_{a.variant}_seed{a.seed}'
    for key in ['epochs','batch_size','grad_accum','val_every']:
        if getattr(a,key)<1:raise ValueError(key+' must be positive')
    if a.workers<0 or a.cpu_threads<1 or a.eval_batch_size<1:raise ValueError('Invalid worker/thread/eval batch count')
    if not 0<a.cutoff<=.5:raise ValueError('cutoff must be in (0,.5] cycles/pixel')
    return a

def make_spec(a):
    spec=FastSpec(a.protocol,a.raw_root,a.rgbir_dir,a.cache_dir,a.memory)
    spec.eval_batch_size=a.eval_batch_size;spec.fusion_storage=a.fusion_storage
    return spec
def config_dict(a):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def duration(s):
    h,r=divmod(int(s),3600);m,s=divmod(r,60);return f'{h:02d}:{m:02d}:{s:02d}'

def main():
    a=resolve_args(parser().parse_args());torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1)
    seed(a.seed);device=torch.device(a.device)
    spec=make_spec(a)
    spec.prepare('train')
    if not a.smoke_test:spec.prepare('val')
    if a.prepare_cache_only:
        print('[CACHE READY] Training/validation caches prepared; no model loaded.');return
    spec.preload('train');ds=FastTrainSet(spec)
    print(f'[PATH] protocol={a.protocol} raw={a.raw_root} rgbir={a.rgbir_dir} pretrained={a.pretrained}',flush=True)
    print(f'[DATA] train_tiles={len(spec.split["train"])} val_tiles={len(spec.split["val"])} samples/epoch={len(ds)} crop=512 val_windows=512',flush=True)
    model=PotsdamModel(a.pretrained,a.variant=='frequency',a.cutoff).to(device)
    amp=device.type=='cuda' and not a.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    if a.smoke_test:
        model.train();r,n,y=ds[0];r=r[None].to(device);n=n[None].to(device);y=y[None].to(device)
        with torch.autocast(device_type=device.type,enabled=amp):
            raw=model(r,n,raw=True);up=F.interpolate(raw,size=y.shape[-2:],mode='bilinear',align_corners=False)
            loss=F.cross_entropy(up.float(),y)+a.lovasz_weight*lovasz(raw,y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradients')
        if model.frequency is not None:
            g=sum(float(p.grad.abs().sum()) for p in model.frequency.parameters() if p.grad is not None)
            if g<=0:raise RuntimeError('Frequency branch has no gradient')
            print(f'[FREQUENCY] gradient_abs_sum={g:.6g}')
        print(f'[SMOKE PASS] variant={a.variant} image={tuple(r.shape)} logits={tuple(raw.shape)} loss={loss.item():.6f}')
        return
    if a.out.exists() and any(a.out.iterdir()) and a.resume is None:raise FileExistsError('Output is nonempty; use a new --out or --resume')
    a.out.mkdir(parents=True,exist_ok=True)
    sampler=EpochSampler(len(ds),a.seed)
    loader_options=dict(batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,
                        pin_memory=device.type=='cuda',worker_init_fn=worker_init)
    if a.workers:
        # Linux fork shares preloaded arrays; workers do CPU-only operations.
        loader_options.update(persistent_workers=True,prefetch_factor=2,multiprocessing_context='fork')
    loader=DataLoader(ds,**loader_options)
    # Explicit per-epoch deterministic permutation, independent of module RNG.
    base=[];new=[]
    for name,p in model.named_parameters():
        (new if name.startswith(('adapters','frequency')) or name.startswith('decode_head.classifier') else base).append(p)
    optimizer=torch.optim.AdamW([{'params':base,'lr':a.base_lr},{'params':new,'lr':a.new_lr}],weight_decay=.01)
    updates=math.ceil(len(loader)/a.grad_accum)*a.epochs;warmup=max(1,int(.05*updates))
    def factor(step):return (step+1)/warmup if step<warmup else max(0,(1-(step-warmup)/max(1,updates-warmup)))**.9
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    first=0;best=-1.
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        for k in ['variant','seed','epochs','batch_size','grad_accum','base_lr','new_lr','cutoff','lovasz_weight']:
            if ck['config'][k]!=config_dict(a)[k]:raise ValueError('Resume config mismatch: '+k)
        if ck['protocol_sha256']!=spec.hash:raise ValueError('Resume protocol mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        first=ck['epoch'];best=ck['best_miou']
        old_best=a.resume.parent/'best.pt'
        if old_best.is_file() and old_best.resolve()!=(a.out/'best.pt').resolve() and not (a.out/'best.pt').exists():
            shutil.copy2(old_best,a.out/'best.pt')
        print(f'[RESUME] completed_epoch={first} next_epoch={first+1} inherited_best_mIoU={best:.6f}',flush=True)
    (a.out/'run_config_fast_v2.json').write_text(json.dumps(config_dict(a),indent=2),encoding='utf8')
    (a.out/'dataset_protocol.json').write_text(json.dumps(spec.p,indent=2),encoding='utf8')
    print(f'[FAST] workers={a.workers} cpu_threads={a.cpu_threads} memory={a.memory} eval_batch={a.eval_batch_size}',flush=True)
    for epoch in range(first,a.epochs):
        seed(a.seed+epoch);ds.epoch=epoch
        sampler.epoch=epoch;epoch_loader=loader
        model.train();optimizer.zero_grad(set_to_none=True);begin=time.time();loss_sum=0.;skips=0;wait_sum=0.;compute_sum=0.;last_end=time.perf_counter()
        for i,(r,n,y) in enumerate(epoch_loader):
            batch_ready=time.perf_counter();wait_sum+=batch_ready-last_end
            r=r.to(device,non_blocking=True);n=n.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            group_start=(i//a.grad_accum)*a.grad_accum;group_n=min(a.grad_accum,len(epoch_loader)-group_start)
            # Correct weighting even if the last batch has fewer examples.
            group_samples=min(a.grad_accum*a.batch_size,len(ds)-group_start*a.batch_size)
            with torch.autocast(device_type=device.type,enabled=amp):
                raw=model(r,n,raw=True);up=F.interpolate(raw,size=y.shape[-2:],mode='bilinear',align_corners=False)
                ce=F.cross_entropy(up.float(),y);lv=lovasz(raw,y);loss=ce+a.lovasz_weight*lv
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss*(y.shape[0]/group_samples)).backward();loss_sum+=float(loss.detach())*y.shape[0]
            if (i+1)%a.grad_accum==0 or i+1==len(epoch_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=old:scheduler.step()
                else:skips+=1
                optimizer.zero_grad(set_to_none=True)
            compute_sum+=time.perf_counter()-batch_ready;last_end=time.perf_counter()
            if (i+1)%16==0 or i+1==len(epoch_loader):
                elapsed=time.time()-begin;remaining=elapsed/(i+1)*(len(epoch_loader)-i-1)
                eta=remaining+elapsed/(i+1)*len(epoch_loader)*(a.epochs-epoch-1)
                print(f'[train] epoch={epoch+1:03d}/{a.epochs} batch={i+1}/{len(epoch_loader)} loss={loss.item():.5f} ce={ce.item():.5f} lovasz={lv.item():.5f} epoch_eta={duration(remaining)} train_eta_excl_val={duration(eta)} data_wait_ms={1000*wait_sum/(i+1):.1f} step_wall_ms={1000*compute_sum/(i+1):.1f}',flush=True)
        result=None
        if (epoch+1)%a.val_every==0 or epoch+1==a.epochs:
            result=evaluate(model,spec,'val',device,a.out/f'val_epoch_{epoch+1:03d}',amp)
        improved=result is not None and result['mIoU']>best
        if improved:best=result['mIoU']
        ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'epoch':epoch+1,'best_miou':best,'config':config_dict(a),'protocol_sha256':spec.hash}
        save(a.out/'last.pt',ck)
        if improved:save(a.out/'best.pt',ck)
        with (a.out/'train_log.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':epoch+1,'train_loss':loss_sum/len(ds),'val_miou':None if result is None else result['mIoU'],'best_miou':best,'amp_skipped_updates':skips,'data_wait_seconds':wait_sum,'step_wall_seconds':compute_sum,'loader_version':'fast-v2'})+'\n')
    print('[DONE]',a.out)
if __name__=='__main__':main()
