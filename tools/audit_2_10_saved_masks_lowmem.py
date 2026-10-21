#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight 2_10 GT audit, optionally compare an ALREADY SAVED raw class-ID prediction.

Run from the repository root after placing this script in tools/:
  python tools/audit_2_10_saved_masks_lowmem.py
  python tools/audit_2_10_saved_masks_lowmem.py --pred /path/to/2_10_pred.npy

No model weights and no inference are loaded. Predictions must contain raw 0..5
class IDs at the full 6000x6000 tile resolution. RGB visualization PNGs are NOT
accepted because colors are not canonical class IDs.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset

NAMES=['Impervious surfaces','Building','Low vegetation','Tree','Car','Clutter/background']
COLORS=np.array([[220,220,220],[220,40,40],[80,190,80],[20,100,40],[40,90,220],[230,180,30]],dtype=np.uint8)
TILE='2_10'

def resolve(p):
    p=Path(p).expanduser(); return p if p.is_absolute() else ROOT/p

def read_prediction(path, shape):
    path=resolve(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if path.suffix.lower()=='.npy':
        arr=np.load(path,mmap_mode='r',allow_pickle=False)
    elif path.suffix.lower() in ('.png','.tif','.tiff'):
        with Image.open(path) as im:
            if im.mode not in ('P','L','I','I;16'):
                raise ValueError(f'Prediction image mode={im.mode}, not a class-ID image. Do not use an RGB/colorized PNG!')
            arr=np.asarray(im).copy()
    else:
        raise ValueError('Only .npy or single-band class-ID .png/.tif/.tiff are accepted')
    if arr.shape!=shape: raise ValueError(f'Pred shape {arr.shape} != GT shape {shape}; cannot compare without original-grid alignment')
    vals=np.unique(arr)
    if np.any((vals<0)|(vals>5)):
        raise ValueError(f'Prediction must have classes 0..5 (seen {vals[:20]}). Avoid 255/invalid predicted IDs.')
    return arr

def save_thumb(mask,out,size=1200):
    h,w=mask.shape
    scale=min(1.,size/max(h,w)); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    # Shrink class ID first, THEN colorize: avoids holding a 6000x6000 RGB array.
    m=Image.fromarray(np.asarray(mask,dtype=np.uint8),'L').resize((nw,nh),Image.Resampling.NEAREST)
    a=np.asarray(m)
    rgb=np.zeros((nh,nw,3),dtype=np.uint8)
    for i in range(6): rgb[a==i]=COLORS[i]
    Image.fromarray(rgb,'RGB').save(out)
    return rgb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, default=None, help='Existing FULL TILE 0..5 class-ID .npy or grayscale PNG/TIFF; no checkpoint needed')
    ap.add_argument('--output', type=Path, default=Path('outputs/coverage_audit_2_10_lowmem'))
    ap.add_argument('--split', choices=['val','test'], default='val')
    args=ap.parse_args(); out=resolve(args.output);out.mkdir(parents=True,exist_ok=True)
    ds=PotsdamSlidingWindowDataset(ROOT,split=args.split)
    if TILE not in ds.tile_ids: raise RuntimeError(f'{TILE} not in {args.split}: {ds.tile_ids}')
    gt=ds.load_full_label(TILE).cpu().numpy()
    valid=(gt>=0)&(gt<6)
    valid_n=int(valid.sum()); gt_counts=np.bincount(gt[valid].astype(np.int64),minlength=6)
    gtveg=int(gt_counts[2]+gt_counts[3]); gt_cov=100*gtveg/valid_n
    result={'tile_id':TILE,'split':args.split,'gt_valid_pixels':valid_n,'gt_vegetation_pixels':gtveg,
            'gt_coverage_percent':gt_cov,'gt_counts_by_class':dict(zip(NAMES,map(int,gt_counts))),
            'notion_GT_reference_vegetation_pixels':17861449,
            'notion_reference_matches':gtveg==17861449,
            'vegetation_definition':'class 2 Low vegetation + class 3 Tree; only GT 0..5 valid'}
    save_thumb(gt,out/'GT_preview.png')
    rows=[dict(class_id=i,class_name=NAMES[i],GT_pixels=int(gt_counts[i])) for i in range(6)]
    if args.pred:
        pred=read_prediction(args.pred,gt.shape)
        pr_counts=np.zeros(6,dtype=np.int64); cm=np.zeros((6,6),dtype=np.int64)
        # Rowwise accumulation avoids full 6000x6000 GT/pred uint64 temporary arrays.
        for y0 in range(0,gt.shape[0],256):
            g=gt[y0:y0+256]; p=np.asarray(pred[y0:y0+256]); v=(g<6)
            pp=p[v].astype(np.int64,copy=False); gg=g[v].astype(np.int64,copy=False)
            pr_counts+=np.bincount(pp,minlength=6)
            cm+=np.bincount(6*gg+pp,minlength=36).reshape(6,6)
        prveg=int(pr_counts[2]+pr_counts[3]); prcov=100*prveg/valid_n
        result.update({'prediction_path':str(resolve(args.pred)),'pred_vegetation_pixels':prveg,
                       'pred_coverage_percent':prcov,'signed_error_pp':prcov-gt_cov,
                       'absolute_error_pp':abs(prcov-gt_cov),
                       'relative_error_percent':100*abs(prveg-gtveg)/gtveg if gtveg else None,
                       'confusion_matrix_gt_rows_pred_columns':cm.tolist()})
        for row,n in zip(rows,pr_counts):row['Pred_pixels']=int(n)
        save_thumb(pred,out/'Pred_preview.png')
        # Derive the error preview at reduced size only.
        size=1200; sh=(round(gt.shape[1]*min(1,size/max(gt.shape))),round(gt.shape[0]*min(1,size/max(gt.shape))))
        gg=np.asarray(Image.fromarray(np.asarray(gt,dtype=np.uint8),'L').resize(sh,Image.Resampling.NEAREST))
        pp=np.asarray(Image.fromarray(np.asarray(pred,dtype=np.uint8),'L').resize(sh,Image.Resampling.NEAREST))
        v=gg<6; gv=np.isin(gg,[2,3]);pv=np.isin(pp,[2,3]);err=np.zeros((*gg.shape,3),np.uint8)
        err[v&(gg==pp)]=(245,245,245);err[v&gv&~pv]=(230,50,50);err[v&~gv&pv]=(50,100,240)
        err[v&(gg!=pp)&(gv==pv)]=(245,200,40)
        Image.fromarray(err,'RGB').save(out/'error_preview.png')
        with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['GT / Pred']+NAMES)
            for i in range(6):w.writerow([NAMES[i]]+cm[i].tolist())
        print(f'Pred vegetation pixels: {prveg:,} | coverage: {prcov:.4f}%')
        print(f'Absolute error: {abs(prcov-gt_cov):.4f} pp | relative error: {result["relative_error_percent"]:.4f}%')
    with (out/'class_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['class_id','class_name','GT_pixels']+(['Pred_pixels'] if args.pred else []));w.writeheader();w.writerows(rows)
    (out/'coverage_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'GT vegetation pixels: {gtveg:,} | coverage: {gt_cov:.4f}% | Notion match: {gtveg==17861449}')
    print(f'Outputs: {out}')
    if not args.pred:print('GT-only mode: prediction not computed. Use --pred with an EXISTING raw class-ID mask.')
if __name__=='__main__':main()
