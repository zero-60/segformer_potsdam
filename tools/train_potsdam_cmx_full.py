#!/usr/bin/env python3
"""Train the existing CMX adapter on the frozen full Potsdam 18/6/14 split."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cmx_full_model import PotsdamCMX
from cmx_full_data import FastSpec,FastTrainSet,EpochSampler,worker_init,Spec,TrainSet
from cmx_full_evaluate import evaluate
from cmx_full_console import disable_quick_edit
from cmx_full_runtime import duration,heartbeat,write_json,validation_due,validation_count,estimate_training_eta

ROOT=Path(__file__).resolve().parent.parent
ARCHITECTURE='POTSDAM_CMX_FULL_V1'


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def lovasz(raw,labels):
    # Preserve the tested quick-version objective, including quarter-scale Lovasz.
    lab=F.interpolate(labels[:,None].float(),size=raw.shape[-2:],mode='nearest')[:,0].long().flatten()
    prob=raw.float().softmax(1).permute(0,2,3,1).reshape(-1,6);losses=[]
    for c in range(6):
        fg=(lab==c).float()
        if not fg.any():continue
        err,idx=(fg-prob[:,c]).abs().sort(descending=True);fg=fg[idx]
        inter=fg.sum()-fg.cumsum(0);union=fg.sum()+(1-fg).cumsum(0)
        grad=1-inter/union.clamp_min(1);grad=torch.cat((grad[:1],grad[1:]-grad[:-1]))
        losses.append(torch.dot(err,grad))
    return torch.stack(losses).mean() if losses else raw.sum()*0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--protocol',type=Path,default=Path('data/processed/potsdam_cmx_full_v1'))
    p.add_argument('--raw-root',type=Path,default=Path('data/raw/potsdam'))
    p.add_argument('--rgbir-dir',type=Path,default=Path('data/raw/potsdam/_expanded/4_Ortho_RGBIR'))
    p.add_argument('--pretrained',type=Path,default=Path('../hf-cache/segformer-b2-finetuned-ade-512-512'))
    p.add_argument('--variant',choices=['baseline','frm','cmx'],default='cmx')
    p.add_argument('--out',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--val-every',type=int,default=5);p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--base-lr',type=float,default=3e-5);p.add_argument('--new-lr',type=float,default=3e-4)
    p.add_argument('--lovasz-weight',type=float,default=.5);p.add_argument('--log-every',type=int,default=16)
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache-dir',type=Path,default=Path('data/processed/potsdam_cmx_full_cache_v1'))
    p.add_argument('--memory',choices=['ram','mmap'],default='ram')
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--eval-batch-size',type=int,default=1)
    p.add_argument('--fusion-storage',choices=['ram','disk'],default='ram')
    p.add_argument('--prepare-cache-only',action='store_true');p.add_argument('--check-data',action='store_true')
    p.add_argument('--smoke-test',action='store_true');p.add_argument('--keep-quick-edit',action='store_true')
    return p


def resolve_args(a):
    a.project_root=a.project_root.resolve()
    for name in ['protocol','raw_root','rgbir_dir','pretrained','out','resume','cache_dir']:
        value=getattr(a,name)
        if value is not None:setattr(a,name,(a.project_root/value).resolve() if not value.is_absolute() else value.resolve())
    if a.out is None:a.out=a.resume.parent if a.resume else a.project_root/f'outputs/training/potsdam_cmx_full_{a.variant}_v1_seed{a.seed}'
    for name in ['epochs','batch_size','grad_accum','val_every','log_every','cpu_threads','eval_batch_size']:
        if getattr(a,name)<1:raise ValueError(name+' must be positive')
    if a.workers<0:raise ValueError('workers must be nonnegative')
    if a.base_lr<=0 or a.new_lr<=0 or a.lovasz_weight<0:raise ValueError('Invalid learning rate / loss weight')
    if not 0<=a.seed<2**32:raise ValueError('seed must be in [0,2**32)')
    if os.name=='nt' and a.workers:
        print('[CONFIG] Windows uses workers=0 to avoid spawning copies of the RAM cache.',flush=True);a.workers=0
    return a


def make_spec(a):
    s=FastSpec(a.protocol,a.raw_root,a.rgbir_dir,a.cache_dir,a.memory)
    s.eval_batch_size=a.eval_batch_size;s.fusion_storage=a.fusion_storage
    return s


def config_dict(a):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}


def atomic_checkpoint(path,obj):
    temp=path.with_name(path.name+'.tmp')
    with heartbeat('saving '+path.name):torch.save(obj,temp);temp.replace(path)


def load_full_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('architecture')!=ARCHITECTURE:
        raise ValueError('Expected a full-protocol checkpoint. Quick-subset checkpoints cannot be resumed here.')
    return ck


def checkpoint_model(ck):
    from transformers import SegformerConfig
    cfg=SegformerConfig.from_dict(ck['model_config'])
    return PotsdamCMX(config=cfg,variant=ck['config']['variant'])


def smoke(a,device):
    # One real 512 crop: no need to construct 24 full-tile caches first.
    s=Spec(a.protocol,a.raw_root,a.rgbir_dir);dataset=TrainSet(s)
    with heartbeat('one-crop smoke input'):rgb,nir,labels=dataset[0]
    model=PotsdamCMX(a.pretrained,variant=a.variant).to(device).train()
    rgb=rgb[None].to(device);nir=nir[None].to(device);labels=labels[None].to(device)
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda' and not a.no_amp):
        raw=model(rgb,nir,raw=True);up=F.interpolate(raw,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.cross_entropy(up.float(),labels)+a.lovasz_weight*lovasz(raw,labels)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite smoke gradient')
    for name in ['frms','ffms']:
        if hasattr(model,name):
            total=sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)
            if total<=0:raise RuntimeError(name+' has no gradient')
            print(f'[SMOKE] {name}_gradient_abs_sum={total:.6g}',flush=True)
    if device.type=='cuda':print(f'[CUDA] peak_allocated_GiB={torch.cuda.max_memory_allocated(device)/2**30:.3f}',flush=True)
    print(f'[SMOKE PASS] variant={a.variant} rgb={tuple(rgb.shape)} nir={tuple(nir.shape)} loss={loss.item():.6f}',flush=True)


def main():
    a=resolve_args(parser().parse_args())
    if not a.keep_quick_edit:disable_quick_edit()
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1);seed(a.seed)
    s=make_spec(a);s.check_sources('train');s.check_sources('val')
    print(f'[FULL PROTOCOL] train=18 val=6 test=14 samples_per_epoch=1152 patch=512 val_windows=1536',flush=True)
    print(f'[NIR] full_train_mean={s.norm["nir"]["mean"]:.10f} std={s.norm["nir"]["std"]:.10f}',flush=True)
    if a.check_data:
        print('[CHECK PASS] Metadata hashes and train/val paths passed; raw GT hashes are checked during cache construction.',flush=True);return
    ck=None
    if a.resume:
        ck=load_full_checkpoint(a.resume)
        if 'optimizer' not in ck:raise ValueError('Resume from last.pt, not the inference-only best.pt')
        if a.out.resolve()!=a.resume.parent.resolve():raise ValueError('Resume must use the same run directory; omit --out')
        if ck['protocol_sha256']!=s.hash:raise ValueError('Resume protocol mismatch')
        for name in ['variant','seed','epochs','batch_size','grad_accum','base_lr','new_lr','lovasz_weight','val_every','no_amp']:
            if ck['config'][name]!=config_dict(a)[name]:raise ValueError('Resume config mismatch: '+name)
    elif a.out.exists() and any(a.out.iterdir()) and not (a.smoke_test or a.prepare_cache_only):
        raise FileExistsError('Output directory is nonempty; use --resume path/to/last.pt or a new --out')
    if a.prepare_cache_only:
        s.prepare('train');s.prepare('val');print('[CACHE READY] Full train/val caches built; no model loaded.',flush=True);return
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the active environment')
    if a.smoke_test:smoke(a,device);return
    s.prepare('train');s.prepare('val');s.preload('train')
    dataset=FastTrainSet(s);sampler=EpochSampler(len(dataset),a.seed)
    options=dict(batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=device.type=='cuda',worker_init_fn=worker_init,
                 generator=torch.Generator().manual_seed(a.seed))
    if a.workers:options.update(persistent_workers=True,prefetch_factor=2,multiprocessing_context='fork')
    loader=DataLoader(dataset,**options)
    print('[MODEL] loading full CMX adapter...',flush=True)
    with heartbeat('loading model'):model=(checkpoint_model(ck) if ck else PotsdamCMX(a.pretrained,variant=a.variant)).to(device)
    amp=device.type=='cuda' and not a.no_amp;scaler=torch.amp.GradScaler('cuda',enabled=amp)
    base=[];new=[]
    for name,p in model.named_parameters():
        (new if name.startswith(('adapters','frms','ffms','ffm_scales','decode_head.classifier')) else base).append(p)
    optimizer=torch.optim.AdamW([{'params':base,'lr':a.base_lr},{'params':new,'lr':a.new_lr}],weight_decay=.01)
    updates=math.ceil(len(loader)/a.grad_accum)*a.epochs;warmup=max(1,int(.05*updates))
    def factor(step):return (step+1)/warmup if step<warmup else max(0,1-(step-warmup)/max(1,updates-warmup))**.9
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    first=0;best=-1.;best_epoch=0;val_times=[];train_times=[]
    if ck:
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        first=ck['epoch'];best=ck['best_miou'];best_epoch=ck['best_epoch']
        val_times=ck['val_times'];train_times=ck['train_times']
        print(f'[RESUME] completed={first}/{a.epochs} best_epoch={best_epoch} best_mIoU={best:.6f}',flush=True)
        del ck
    a.out.mkdir(parents=True,exist_ok=True)
    import transformers
    scripts=[p for p in Path(__file__).parent.glob('*.py') if p.name.startswith('cmx_full_') or p.name in ('train_potsdam_cmx_full.py','validate_potsdam_cmx_full.py')]
    environment={'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
                 'python':platform.python_version(),'platform':platform.platform(),'cuda_runtime':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                 'official_cmx_commit':'e251d860aebc2f583a6c4919877e6bebe7f1aff3',
                 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}}
    write_json(a.out/'run_environment.json',environment);write_json(a.out/'run_config_cmx_full_v1.json',config_dict(a))
    write_json(a.out/'dataset_protocol.json',s.p)
    print(f'[START] variant={a.variant} params_M={sum(p.numel() for p in model.parameters())/1e6:.3f} epochs={a.epochs} batches={len(loader)} amp={amp} out={a.out}',flush=True)
    print('[ETA] Epoch 1 includes full validation to calibrate total_eta_est. Before that, total_eta_lower_bound excludes validation.',flush=True)
    for epoch in range(first,a.epochs):
        seed((a.seed+epoch)%2**32);dataset.epoch=epoch;sampler.epoch=epoch
        model.train();optimizer.zero_grad(set_to_none=True);begin=time.monotonic()
        loss_sum=0.;skips=0;wait_sum=0.;compute_sum=0.;last_end=time.perf_counter()
        for i,(rgb,nir,labels) in enumerate(loader):
            ready=time.perf_counter();wait_sum+=ready-last_end
            rgb=rgb.to(device,non_blocking=True);nir=nir.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            group_start=(i//a.grad_accum)*a.grad_accum
            group_samples=min(a.grad_accum*a.batch_size,len(dataset)-group_start*a.batch_size)
            with torch.autocast(device_type=device.type,enabled=amp):
                raw=model(rgb,nir,raw=True);up=F.interpolate(raw,size=labels.shape[-2:],mode='bilinear',align_corners=False)
                ce=F.cross_entropy(up.float(),labels);lv=lovasz(raw,labels);loss=ce+a.lovasz_weight*lv
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop without overwriting last complete epoch')
            scaler.scale(loss*(labels.shape[0]/group_samples)).backward()
            loss_sum+=float(loss.detach())*labels.shape[0]
            if (i+1)%a.grad_accum==0 or i+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=previous:scheduler.step()
                else:skips+=1
                optimizer.zero_grad(set_to_none=True)
            compute_sum+=time.perf_counter()-ready;last_end=time.perf_counter()
            if (i+1)%a.log_every==0 or i+1==len(loader):
                elapsed=time.monotonic()-begin
                pending=validation_count(epoch+1,a.epochs,a.val_every)
                e_eta,t_eta,total=estimate_training_eta(elapsed,i+1,len(loader),a.epochs-epoch-1,train_times,val_times,pending)
                eta=f'total_eta_est={duration(total)}' if total is not None else f'total_eta_lower_bound={duration(t_eta)} waiting_first_validation=1'
                print(f'[train] epoch={epoch+1:03d}/{a.epochs} batch={i+1}/{len(loader)} loss={loss.item():.5f} ce={ce.item():.5f} lovasz={lv.item():.5f} epoch_eta={duration(e_eta)} train_eta_excl_val={duration(t_eta)} {eta} data_wait_ms={1000*wait_sum/(i+1):.1f} step_wall_ms={1000*compute_sum/(i+1):.1f}',flush=True)
                write_json(a.out/'progress.json',{'phase':'train','epoch':epoch+1,'batch':i+1,'total_batches':len(loader),
                           'epoch_eta_seconds':e_eta,'train_eta_excl_val_seconds':t_eta,'total_eta_est_seconds':total})
        train_times.append(time.monotonic()-begin)
        result=None
        if validation_due(epoch+1,a.epochs,a.val_every):
            def future_eta(current_validation_estimate):
                remaining_train=sum(train_times[-3:])/len(train_times[-3:])*(a.epochs-epoch-1)
                future_val=validation_count(epoch+2,a.epochs,a.val_every)
                # During first validation use elapsed throughput; after it use measured history.
                average_val=sum(val_times)/len(val_times) if val_times else current_validation_estimate
                return remaining_train+average_val*future_val
            write_json(a.out/'progress.json',{'phase':'val','epoch':epoch+1,'split':'val','windows':1536})
            result=evaluate(model,s,'val',device,a.out/f'val_epoch_{epoch+1:03d}',amp,future_eta=future_eta)
            val_times.append(result['seconds'])
        improved=result is not None and result['mIoU']>best
        if improved:best=result['mIoU'];best_epoch=epoch+1
        scales=model.ffm_scales.detach().tanh().cpu().tolist() if a.variant=='cmx' else None
        log={'epoch':epoch+1,'train_loss':loss_sum/len(dataset),'val_miou':None if result is None else result['mIoU'],
             'val_vegetation_MAE_pp':None if result is None else result['vegetation_MAE_pp'],'best_miou':best,
             'best_epoch':best_epoch,'amp_skipped_updates':skips,'data_wait_seconds':wait_sum,'step_wall_seconds':compute_sum,
             'train_epoch_seconds':train_times[-1],'ffm_scales':scales,'loader_version':'cmx-full-v1'}
        # One file per completed epoch makes re-running an interrupted epoch idempotent.
        write_json(a.out/f'epoch_{epoch+1:03d}.json',log)
        history=[json.loads(p.read_text()) for p in sorted(a.out.glob('epoch_*.json')) if int(p.stem.split('_')[-1])<=epoch+1]
        (a.out/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
        payload={'architecture':ARCHITECTURE,'model':model.state_dict(),'model_config':model.rgb_encoder.config.to_dict(),
                 'epoch':epoch+1,'best_miou':best,'best_epoch':best_epoch,'config':config_dict(a),'protocol_sha256':s.hash,
                 'metrics':result}
        if improved:atomic_checkpoint(a.out/'best.pt',payload)
        payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                       train_times=train_times,val_times=val_times)
        atomic_checkpoint(a.out/'last.pt',payload)
        remaining=sum(train_times[-3:])/len(train_times[-3:])*(a.epochs-epoch-1)
        remaining+=sum(val_times)/len(val_times)*validation_count(epoch+2,a.epochs,a.val_every) if val_times else 0.
        write_json(a.out/'progress.json',{'phase':'epoch_complete','epoch':epoch+1,'epochs':a.epochs,'best_epoch':best_epoch,
                   'best_miou':best,'total_eta_est_seconds':remaining})
        print(f'[epoch DONE] {epoch+1}/{a.epochs} best_epoch={best_epoch} best_mIoU={best:.6f} total_eta_est={duration(remaining)} ffm_scales={scales}',flush=True)
    write_json(a.out/'progress.json',{'phase':'done','epochs':a.epochs,'best_epoch':best_epoch,'best_miou':best,'total_eta_est_seconds':0})
    print(f'[DONE] best_epoch={best_epoch} best_mIoU={best:.6f} best={a.out/"best.pt"}',flush=True)


if __name__=='__main__':main()
