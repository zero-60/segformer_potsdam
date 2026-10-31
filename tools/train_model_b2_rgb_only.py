#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a standard RGB-only SegFormer-B2 baseline on TLSNMSP.

The model reads the standalone RGB files from the existing manifest, not the
multispectral TIFF. Split files, labels, crop size, augmentation, loss,
optimizer and training schedule are kept close to the BGR+RedEdge experiment.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.segformer_b2_rgb_only import build_model_b2_rgb_only, CLASS_NAMES, NUM_CLASSES

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL_DIR = ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_PRETRAINED = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
DEFAULT_OUTPUT = ROOT / "outputs" / "training" / "meris_b2_rgb_only_seed20261029"
CROP = 512
IGNORE_INDEX = 255
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def read_rows(path: Path):
    if not path.is_file(): raise FileNotFoundError(f"缺少协议清单：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(root: Path, rel: str):
    return (root / Path(rel.replace("/", os.sep))).resolve()


def load_rgb(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None: raise RuntimeError(f"RGB 影像读取失败：{path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"RGB 影像维度异常：{arr.shape}, {path}")
    if arr.dtype != np.uint8:
        raise RuntimeError(f"RGB 影像预期 uint8，实际 {arr.dtype}：{path}")
    return np.ascontiguousarray(arr)


def load_label(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None: raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:,:,0], arr[:,:,1]) and np.array_equal(arr[:,:,0], arr[:,:,2]):
            arr = arr[:,:,0]
        else:
            raise RuntimeError(f"mask_converted 必须是单通道 ID 图：{path}, shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}：{path}")
    ids = set(np.unique(arr).astype(int).tolist())
    if ids - set(range(1,8)):
        raise RuntimeError(f"标签出现 1..7 以外的 ID：{sorted(ids)}；{path}")
    return arr.astype(np.uint8, copy=False)


def to_tensor(rgb, label_raw):
    image = rgb.astype(np.float32) / 255.0
    image = (image - RGB_MEAN.reshape(1,1,3)) / RGB_STD.reshape(1,1,3)
    label = label_raw.astype(np.int16) - 1  # raw IDs 1..7 -> train IDs 0..6
    if label.size and (label.min() < 0 or label.max() >= NUM_CLASSES):
        raise RuntimeError(f"标签映射越界：{label.min()}..{label.max()}")
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()
    y = torch.from_numpy(np.ascontiguousarray(label.astype(np.int64))).long()
    return x, y


class RandomCropDataset(Dataset):
    def __init__(self, rows, root, seed, augment):
        self.rows, self.root, self.seed, self.augment, self.epoch = rows, root, int(seed), augment, 0
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb = load_rgb(resolve_path(self.root, row["rgb_path"]))
        lab = load_label(resolve_path(self.root, row["label_path"]))
        if rgb.shape[:2] != lab.shape:
            raise RuntimeError(f"{row['sample_id']} RGB/标签尺寸不一致：{rgb.shape}, {lab.shape}")
        h,w = lab.shape
        if h < CROP or w < CROP:
            raise RuntimeError(f"影像小于 {CROP}x{CROP}：{row['sample_id']} {w}x{h}")
        rng = np.random.default_rng(self.seed + self.epoch*1_000_003 + idx*9_176)
        y0 = int(rng.integers(0,h-CROP+1)); x0 = int(rng.integers(0,w-CROP+1))
        rgb,lab = rgb[y0:y0+CROP,x0:x0+CROP], lab[y0:y0+CROP,x0:x0+CROP]
        if self.augment:
            k = int(rng.integers(0,4))
            if k: rgb,lab = np.rot90(rgb,k),np.rot90(lab,k)
            if bool(rng.integers(0,2)): rgb,lab = rgb[:,::-1],lab[:,::-1]
            if bool(rng.integers(0,2)): rgb,lab = rgb[::-1,:],lab[::-1,:]
        x,y = to_tensor(np.ascontiguousarray(rgb),np.ascontiguousarray(lab))
        return {"image":x,"label":y,"sample_id":row["sample_id"]}


def starts(length):
    # Non-overlapping tiles; last partial tile is padded and ignored beyond its valid area.
    return list(range(0,length,CROP))


class EvalPatchDataset(Dataset):
    def __init__(self, rows, root):
        self.rows,self.root=rows,root
        self.items=[]
        for ri,row in enumerate(rows):
            lab=load_label(resolve_path(root,row["label_path"]))
            for y in starts(lab.shape[0]):
                for x in starts(lab.shape[1]):
                    self.items.append((ri,y,x,min(CROP,lab.shape[0]-y),min(CROP,lab.shape[1]-x)))
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        ri,y0,x0,vh,vw=self.items[idx]; row=self.rows[ri]
        rgb=load_rgb(resolve_path(self.root,row["rgb_path"]))
        lab=load_label(resolve_path(self.root,row["label_path"]))
        rgb=rgb[y0:y0+vh,x0:x0+vw]; lab=lab[y0:y0+vh,x0:x0+vw]
        if vh<CROP or vw<CROP:
            rgb=np.pad(rgb,((0,CROP-vh),(0,CROP-vw),(0,0)),mode="edge")
            lab=np.pad(lab,((0,CROP-vh),(0,CROP-vw)),mode="edge")
        x,y=to_tensor(rgb,lab)
        return {"image":x,"label":y,"sample_id":row["sample_id"],"valid_h":vh,"valid_w":vw}


def lovasz_grad(gt_sorted):
    p=gt_sorted.numel(); gts=gt_sorted.sum()
    inter=gts-gt_sorted.float().cumsum(0)
    union=gts+(1.0-gt_sorted).float().cumsum(0)
    j=1.0-inter/torch.clamp(union,min=1.0)
    if p>1: j[1:p]=j[1:p]-j[:-1]
    return j


def lovasz_softmax(raw_logits, labels_full):
    labels=F.interpolate(labels_full.unsqueeze(1).float(),size=raw_logits.shape[-2:],mode="nearest").squeeze(1).long()
    probs=torch.softmax(raw_logits.float(),dim=1).permute(0,2,3,1).reshape(-1,NUM_CLASSES)
    labels=labels.reshape(-1); valid=labels!=IGNORE_INDEX; probs,labels=probs[valid],labels[valid]
    if labels.numel()==0: return raw_logits.sum()*0.
    losses=[]
    for cls in range(NUM_CLASSES):
        fg=(labels==cls).float()
        if fg.sum().item()==0: continue
        errors=torch.abs(fg-probs[:,cls]); es,perm=torch.sort(errors,descending=True)
        losses.append(torch.dot(es,lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else raw_logits.sum()*0.


def confusion_update(cm, logits, labels, valid_h=None, valid_w=None):
    if valid_h is not None:
        # evaluation batch size is one
        logits=logits[:,:,:int(valid_h),:int(valid_w)]
        labels=labels[:,:int(valid_h),:int(valid_w)]
    pred=logits.argmax(1).reshape(-1).long()
    gt=labels.reshape(-1).long()
    valid=(gt>=0)&(gt<NUM_CLASSES)
    encoded=gt[valid]*NUM_CLASSES+pred[valid]
    cm += torch.bincount(encoded,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES).cpu()


def metrics(cm):
    a=cm.numpy().astype(np.float64); diag=np.diag(a); g=a.sum(1); p=a.sum(0); total=a.sum()
    oa=float(diag.sum()/total) if total else 0.
    union=g+p-diag; ious=np.divide(diag,union,out=np.full(NUM_CLASSES,np.nan),where=union>0)
    miou=float(np.nanmean(ious)) if np.any(union>0) else 0.
    pe=float((g*p).sum()/total**2) if total else 0.
    kap=float((oa-pe)/(1-pe)) if total and abs(1-pe)>1e-12 else 0.
    cls=[]
    for i,n in enumerate(CLASS_NAMES):
        cls.append({"train_id":i,"raw_label_id":i+1,"class_name":n,"gt_pixels":int(g[i]),"pred_pixels":int(p[i]),
                    "IoU":None if not np.isfinite(ious[i]) else float(ious[i]),
                    "precision":float(diag[i]/p[i]) if p[i] else 0.,
                    "recall":float(diag[i]/g[i]) if g[i] else 0.})
    return {"OA":oa,"mIoU":miou,"Kappa":kap,"total_pixels":int(total),"per_class":cls}


def coverage_from_cm(cm):
    # Vegetation = tree + low vegetation (train IDs 4 and 5).
    a=cm.numpy().astype(np.int64)
    gt_veg=int(a[4:6,:].sum()); pred_veg=int(a[:,4:6].sum()); total=int(a.sum())
    gt_cov=gt_veg*100/total if total else 0.
    pred_cov=pred_veg*100/total if total else 0.
    err=pred_cov-gt_cov
    return {"gt_vegetation_pixels":gt_veg,"pred_vegetation_pixels":pred_veg,
            "gt_vegetation_coverage_pct":gt_cov,"pred_vegetation_coverage_pct":pred_cov,
            "coverage_signed_error_pp":err,"coverage_absolute_error_pp":abs(err),
            "coverage_relative_absolute_error_pct":abs(err/gt_cov*100) if gt_cov else None}


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    os.replace(tmp,path)


def atomic_save(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)


def load_ckpt(path):
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")


def evaluate(model,loader,device,amp,split,outdir):
    model.eval(); cm=torch.zeros((NUM_CLASSES,NUM_CLASSES),dtype=torch.int64); loss_sum=0.; n=0; t0=time.time()
    with torch.no_grad():
        for b in loader:
            x=b["image"].to(device,non_blocking=True); y=b["label"].to(device,non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                low=model(pixel_values=x).logits
                logits=F.interpolate(low.float(),size=y.shape[-2:],mode="bilinear",align_corners=False)
                loss=F.cross_entropy(logits,y)
            loss_sum+=float(loss.item()); n+=1
            vh=int(b["valid_h"][0]); vw=int(b["valid_w"][0])
            confusion_update(cm,logits,y,vh,vw)
    met=metrics(cm); cov=coverage_from_cm(cm)
    result={"split":split,**{k:met[k] for k in ("OA","mIoU","Kappa","total_pixels")},
            **cov,"mean_patch_CE":loss_sum/max(n,1),"patch_count":n,"evaluation_seconds":time.time()-t0,
            "per_class":met["per_class"]}
    outdir.mkdir(parents=True,exist_ok=True)
    save_json(outdir/f"{split.lower()}_metrics.json",result)
    with (outdir/f"{split.lower()}_class_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(met["per_class"][0].keys()));w.writeheader();w.writerows(met["per_class"])
    np.savetxt(outdir/f"{split.lower()}_confusion_matrix.csv",cm.numpy(),fmt="%d",delimiter=",")
    return result


def fmt(sec):
    sec=max(0,int(sec)); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def get_args():
    p=argparse.ArgumentParser(description="Train RGB-only SegFormer-B2 baseline")
    p.add_argument("--data-root",type=Path,default=DEFAULT_DATA_ROOT)
    p.add_argument("--protocol-dir",type=Path,default=DEFAULT_PROTOCOL_DIR)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--pretrained-checkpoint",type=Path,default=DEFAULT_PRETRAINED)
    p.add_argument("--epochs",type=int,default=40);p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accum-steps",type=int,default=4);p.add_argument("--base-lr",type=float,default=3e-5)
    p.add_argument("--new-lr",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--warmup-steps",type=int,default=100);p.add_argument("--lovasz-weight",type=float,default=.5)
    p.add_argument("--num-workers",type=int,default=0);p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--save-every",type=int,default=5);p.add_argument("--early-stopping-patience",type=int,default=10)
    p.add_argument("--resume",default="none",help="none、auto 或 checkpoint 路径")
    p.add_argument("--no-amp",action="store_true");p.add_argument("--no-gradient-checkpointing",action="store_true")
    p.add_argument("--smoke-test",action="store_true")
    return p.parse_args()


def main():
    a=get_args()
    if min(a.epochs,a.batch_size,a.grad_accum_steps)<=0 or a.num_workers<0 or a.warmup_steps<0:
        raise SystemExit("训练参数不合法")
    root=a.data_root.resolve(); protocol=a.protocol_dir.resolve(); out=a.output_dir.resolve(); pretrained=a.pretrained_checkpoint.resolve()
    if not root.is_dir(): raise SystemExit(f"数据目录不存在：{root}")
    train_rows=read_rows(protocol/"train.csv"); val_rows=read_rows(protocol/"val.csv"); test_rows=read_rows(protocol/"test.csv")
    if not train_rows or not val_rows or not test_rows: raise RuntimeError("train/val/test 清单为空")
    set_seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp=device.type=="cuda" and not a.no_amp
    print(f"[DATA] {root}\n[PROTOCOL] {protocol}\n[OUTPUT] {out}\n[DEVICE] {device} AMP={amp}")
    print(f"[INPUT] standalone RGB files | ImageNet mean/std | classes={NUM_CLASSES}")
    print(f"[SPLIT] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_ds=RandomCropDataset(train_rows,root,a.seed,True); val_ds=EvalPatchDataset(val_rows,root); test_ds=EvalPatchDataset(test_rows,root)
    train_loader=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,
                            pin_memory=device.type=="cuda",drop_last=False)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=="cuda")
    test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=="cuda")
    print(f"[PATCHES] train samples/epoch={len(train_ds)} val patches={len(val_ds)} test patches={len(test_ds)}")
    model,meta=build_model_b2_rgb_only(pretrained);model.to(device)
    if not a.no_gradient_checkpointing and hasattr(model,"gradient_checkpointing_enable"):
        try: model.gradient_checkpointing_enable();print("[GRAD-CHECKPOINTING] enabled")
        except Exception as e: print(f"[WARN] gradient checkpointing enable failed: {e}")
    if a.smoke_test:
        b=train_ds[0];x=b["image"].unsqueeze(0).to(device);y=b["label"].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            low=model(pixel_values=x).logits
            logits=F.interpolate(low.float(),size=y.shape[-2:],mode="bilinear",align_corners=False)
            ce=F.cross_entropy(logits,y);lov=lovasz_softmax(low,y);loss=ce+a.lovasz_weight*lov
        if not torch.isfinite(loss):raise RuntimeError("Smoke test loss 非有限")
        loss.backward()
        print(f"[SMOKE PASS] input={tuple(x.shape)} logits={tuple(low.shape)} loss={loss.item():.6f} grads={sum(p.grad is not None for p in model.parameters())}")
        if device.type=="cuda": print(f"[CUDA] peak_allocated_GiB={torch.cuda.max_memory_allocated()/1024**3:.3f}")
        return

    # Separate learning rates for pretrained backbone and newly initialized classifier.
    groups={"base_decay":[],"base_no_decay":[],"new_decay":[],"new_no_decay":[]}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        is_new=name.startswith("decode_head.classifier.")
        no_decay=param.ndim==1 or name.endswith(".bias")
        groups[("new" if is_new else "base")+("_no_decay" if no_decay else "_decay")].append(param)
    param_groups=[]
    for key,params in groups.items():
        if params:param_groups.append({"params":params,"lr":a.new_lr if key.startswith("new") else a.base_lr,
                                       "weight_decay":0. if key.endswith("no_decay") else a.weight_decay})
    optimizer=AdamW(param_groups)
    updates=math.ceil(len(train_loader)/a.grad_accum_steps);total_updates=updates*a.epochs
    def lr_factor(step):
        if a.warmup_steps>0 and step<a.warmup_steps:return max(1e-4,(step+1)/a.warmup_steps)
        progress=min(max((step-a.warmup_steps)/max(1,total_updates-a.warmup_steps),0.),1.)
        return (1.-progress)**.9
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    checkpoints=out/"checkpoints";log=out/"training_history.jsonl";out.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    if a.resume.lower()=="none" and (any(checkpoints.glob("*.pt")) or log.exists()):
        raise RuntimeError(f"输出目录已有训练记录：{out}。续训用 --resume auto 或指定新输出目录。")
    manifest_hash=hashlib.sha256((protocol/"train.csv").read_bytes()).hexdigest()
    start_epoch=0;step=0;best=-1.;best_epoch=0;stale=0
    resume=None
    if a.resume.lower()=="auto":
        resume=checkpoints/"latest.pt"
        if not resume.is_file():resume=None
    elif a.resume.lower()!="none":resume=Path(a.resume).resolve()
    if resume:
        c=load_ckpt(resume)
        if c.get("model_id")!="B2_RGB_ONLY_MERIS7":raise RuntimeError("检查点不是 RGB-only 模型")
        if c.get("train_manifest_sha256")!=manifest_hash:raise RuntimeError("train.csv 哈希变化，拒绝续训")
        model.load_state_dict(c["model"],strict=True);optimizer.load_state_dict(c["optimizer"]);scheduler.load_state_dict(c["scheduler"]);scaler.load_state_dict(c["scaler"])
        start_epoch=int(c["epoch"])+1;step=int(c["global_step"]);best=float(c["best_miou"]);best_epoch=int(c["best_epoch"]);stale=int(c["stale_epochs"])
        try:
            random.setstate(c["rng_python"]);np.random.set_state(c["rng_numpy"]);torch.set_rng_state(c["rng_torch"])
            if torch.cuda.is_available() and c.get("rng_cuda") is not None:torch.cuda.set_rng_state_all(c["rng_cuda"])
        except Exception as e:print(f"[WARN] RNG restore failed: {e}")
        print(f"[RESUME] {resume} next_epoch={start_epoch+1} step={step} best_mIoU={best:.4f}")
    protocol_record={"protocol_version":"TLSNMSP_RGB_ONLY_7CLASS_v1","model":meta,"data_root":str(root),
        "protocol_dir":str(protocol),"input":"standalone RGB file","normalization":{"type":"ImageNet","mean":RGB_MEAN.tolist(),"std":RGB_STD.tolist()},
        "split_counts":{"train":len(train_rows),"val":len(val_rows),"test":len(test_rows)},"crop_size":CROP,
        "loss":{"cross_entropy":1.,"lovasz_softmax_weight":a.lovasz_weight},"optimizer":"AdamW","seed":a.seed,
        "epochs":a.epochs,"batch_size":a.batch_size,"grad_accum_steps":a.grad_accum_steps,"base_lr":a.base_lr,"new_lr":a.new_lr,
        "train_manifest_sha256":manifest_hash}
    save_json(out/"protocol.json",protocol_record)
    epoch_durations=[]
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                item=json.loads(line)
                if item.get("event")=="epoch_done":epoch_durations.append(float(item.get("epoch_seconds",0)))
            except Exception:pass
    run_started=time.time()
    for epoch in range(start_epoch,a.epochs):
        model.train();train_ds.set_epoch(epoch);optimizer.zero_grad(set_to_none=True)
        sums={"loss":0.,"ce":0.,"lovasz":0.};t0=time.time();n_batches=len(train_loader)
        for bi,b in enumerate(train_loader):
            x=b["image"].to(device,non_blocking=True);y=b["label"].to(device,non_blocking=True)
            group_start=(bi//a.grad_accum_steps)*a.grad_accum_steps
            group_size=min(a.grad_accum_steps,n_batches-group_start)
            with torch.cuda.amp.autocast(enabled=amp):
                low=model(pixel_values=x).logits
                logits=F.interpolate(low.float(),size=y.shape[-2:],mode="bilinear",align_corners=False)
                ce=F.cross_entropy(logits,y);lov=lovasz_softmax(low,y);loss=ce+a.lovasz_weight*lov
            if not torch.isfinite(loss.detach()):raise RuntimeError(f"Loss 非有限：epoch={epoch+1} batch={bi+1}")
            scaler.scale(loss/group_size).backward()
            sums["loss"]+=float(loss.detach());sums["ce"]+=float(ce.detach());sums["lovasz"]+=float(lov.detach())
            end=((bi+1)%a.grad_accum_steps==0) or (bi+1==n_batches)
            grad_norm=float("nan")
            if end:
                scaler.unscale_(optimizer);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);grad_norm=float(gn)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if (bi+1)%10==0 or bi+1==n_batches:
                elapsed=time.time()-t0;batch_eta=elapsed/(bi+1)*(n_batches-bi-1)
                avg_ep=sum(epoch_durations)/len(epoch_durations) if epoch_durations else elapsed/(bi+1)*n_batches
                total_eta=batch_eta+max(0,a.epochs-epoch-1)*avg_ep
                finish=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()+total_eta))
                print(f"[TRAIN] epoch={epoch+1:03d}/{a.epochs:03d} batch={bi+1:03d}/{n_batches:03d} "
                      f"loss={loss.item():.5f} ce={ce.item():.5f} lovasz={lov.item():.5f} step={step} "
                      f"grad_norm={grad_norm:.3f} elapsed={elapsed:.1f}s epoch_eta={fmt(batch_eta)} total_eta={fmt(total_eta)} finish~{finish}",flush=True)
            del x,y,low,logits,ce,lov,loss
        train_seconds=time.time()-t0
        val=evaluate(model,val_loader,device,amp,"VAL",out/"metrics")
        record={"event":"epoch_done","epoch":epoch+1,"global_step":step,"train_loss":sums["loss"]/max(1,n_batches),
                "train_ce":sums["ce"]/max(1,n_batches),"train_lovasz":sums["lovasz"]/max(1,n_batches),
                "epoch_seconds":train_seconds+val["evaluation_seconds"],"val_OA":val["OA"],"val_mIoU":val["mIoU"],"val_Kappa":val["Kappa"]}
        with log.open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
        print(f"[VAL] epoch={epoch+1} OA={val['OA']:.4f} mIoU={val['mIoU']:.4f} Kappa={val['Kappa']:.4f} "
              f"veg_abs_error={val['coverage_absolute_error_pp']:.4f}pp",flush=True)
        if val["mIoU"]>best:best=float(val["mIoU"]);best_epoch=epoch+1;stale=0
        else:stale+=1
        payload={"model_id":"B2_RGB_ONLY_MERIS7","model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"global_step":step,
                 "best_miou":best,"best_epoch":best_epoch,"stale_epochs":stale,"model_meta":meta,
                 "train_manifest_sha256":manifest_hash,"protocol":protocol_record,
                 "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),"rng_torch":torch.get_rng_state(),
                 "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_save(payload,checkpoints/"latest.pt")
        if best_epoch==epoch+1:atomic_save(payload,checkpoints/"best.pt")
        if a.save_every>0 and (epoch+1)%a.save_every==0:atomic_save(payload,checkpoints/f"epoch_{epoch+1:03d}.pt")
        epoch_durations.append(record["epoch_seconds"])
        eta=max(0,a.epochs-(epoch+1))*sum(epoch_durations)/len(epoch_durations)
        finish=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()+eta))
        save_json(out/"progress.json",{"epoch_completed":epoch+1,"epochs_total":a.epochs,"global_step":step,
                 "best_epoch":best_epoch,"best_val_mIoU":best,"estimated_remaining_seconds":eta,"estimated_finish_local":finish})
        print(f"[EPOCH ETA] completed={epoch+1}/{a.epochs} remaining={fmt(eta)} finish~{finish}",flush=True)
        if a.early_stopping_patience>0 and stale>=a.early_stopping_patience:
            print(f"[EARLY STOP] best_epoch={best_epoch} best_val_mIoU={best:.4f}");break
    if not (checkpoints/"best.pt").is_file():raise RuntimeError("未生成 best.pt")
    best_ck=load_ckpt(checkpoints/"best.pt");model.load_state_dict(best_ck["model"],strict=True)
    test=evaluate(model,test_loader,device,amp,"TEST",out/"metrics")
    atomic_save({"model_id":"B2_RGB_ONLY_MERIS7","model":model.state_dict(),"best_epoch":best_epoch,
                 "best_val_mIoU":best,"test_metrics":test,"model_meta":meta,"protocol":protocol_record,
                 "train_manifest_sha256":manifest_hash},checkpoints/"final.pt")
    print("\n[TRAINING COMPLETE]")
    print(f"Best epoch: {best_epoch}; best validation mIoU={best:.4f}")
    print(f"TEST OA={test['OA']:.4f}; mIoU={test['mIoU']:.4f}; Kappa={test['Kappa']:.4f}")
    print(f"TEST vegetation coverage GT={test['gt_vegetation_coverage_pct']:.4f}% "
          f"Pred={test['pred_vegetation_coverage_pct']:.4f}% abs_error={test['coverage_absolute_error_pp']:.4f}pp")
    print(f"Best checkpoint: {checkpoints/'best.pt'}")
    print(f"Final model: {checkpoints/'final.pt'}")
    print(f"Output directory: {out}")


if __name__=="__main__":
    try: main()
    except KeyboardInterrupt:
        print("\n用户中断。若训练已开始，已保存的 latest.pt 保留。",file=sys.stderr)
        raise SystemExit(130)
