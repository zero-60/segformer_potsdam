"""Full-tile sliding evaluation with global confusion matrices and ETA."""
import csv
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from cmx_full_data import starts
from cmx_full_runtime import close_mapping, duration, write_json


def metrics(cm):
    cm=np.asarray(cm,dtype=np.int64);tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(len(cm),np.nan),where=union>0)
    total=int(cm.sum())
    if total<=0:raise ValueError('Cannot evaluate an empty confusion matrix')
    po=float(tp.sum()/total)
    pe=float(np.dot(cm.sum(0).astype(float),cm.sum(1))/float(total)**2)
    return {'OA':po,'mIoU':float(np.nanmean(iou)),
            'IoU':[None if np.isnan(v) else float(v) for v in iou],
            'Kappa':float((po-pe)/(1-pe)) if pe<1 else None,'confusion_matrix':cm.tolist()}


@torch.inference_mode()
def evaluate(model,spec,split,device,out,amp=True,save_predictions=False,future_eta=None):
    if split not in ('val','test'):raise ValueError('Evaluation requires val or test')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    was_training=model.training;model.eval();begin=time.monotonic()
    cfg=spec.p['validation_test_inference'];size=cfg['tile_size'][0]
    crop=cfg['crop_size'][0];stride=cfg['stride'][0]
    if cfg['overlap_fusion']!='uniform mean-logit accumulation' or cfg['tta']:
        raise ValueError('Expected mean-logit fusion, no TTA')
    coords=starts(size,crop,stride)
    if coords!=cfg['starts']:raise ValueError('Window coordinates differ from protocol')
    windows=[(y,x) for y in coords for x in coords]
    total_windows=len(windows)*len(spec.split[split]);completed=0
    global_cm=np.zeros((6,6),np.int64);rows=[]
    def progress(tid,tile_windows):
        elapsed=time.monotonic()-begin
        remaining=elapsed/max(1,completed)*(total_windows-completed)
        estimated_full_validation=elapsed/max(1,completed)*total_windows
        total=None if future_eta is None else remaining+future_eta(estimated_full_validation)
        suffix='' if total is None else f' total_eta_est={duration(total)}'
        print(f'[{split}] tile={tid} window={tile_windows}/{len(windows)} all_windows={completed}/{total_windows} {split}_eta={duration(remaining)}{suffix}',flush=True)
    try:
        for tid in spec.split[split]:
            image=spec.image(tid);gt=spec.label_ids(tid)
            if image.shape!=(size,size,4) or gt.shape!=(size,size):raise ValueError('Wrong cached tile shape')
            cm=np.zeros((6,6),np.int64);veg=np.zeros((2,2),np.int64)
            gtveg=predveg=0
            prediction=np.empty((size,size),np.uint8) if save_predictions else None
            with tempfile.TemporaryDirectory(prefix='fusion_',dir=out) as tmp:
                sums=None
                try:
                    sums=np.zeros((6,size,size),np.float32) if spec.fusion_storage=='ram' else np.memmap(Path(tmp)/'logits.bin',dtype='float32',mode='w+',shape=(6,size,size))
                    sums[:]=0
                    count=np.zeros((size,size),np.uint16)
                    for offset in range(0,len(windows),spec.eval_batch_size):
                        batch=windows[offset:offset+spec.eval_batch_size]
                        normalized=[spec.normalize(image[y:y+crop,x:x+crop]) for y,x in batch]
                        rgb=torch.stack([p[0] for p in normalized]).to(device,non_blocking=True)
                        nir=torch.stack([p[1] for p in normalized]).to(device,non_blocking=True)
                        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                            logits=model(rgb,nir)
                        values=logits.float().cpu().numpy()
                        for value,(y,x) in zip(values,batch):
                            sums[:,y:y+crop,x:x+crop]+=value;count[y:y+crop,x:x+crop]+=1
                        completed+=len(batch);wi=offset+len(batch)
                        if wi//64!=offset//64 or wi==len(windows):progress(tid,wi)
                    # A stripe keeps temporary full-resolution logits off the GPU.
                    for y in range(0,size,256):
                        cnt=count[y:y+256]
                        if np.any(cnt==0):raise RuntimeError('Uncovered pixels')
                        pred=(np.asarray(sums[:,y:y+256])/cnt[None]).argmax(0).astype(np.uint8)
                        target=gt[y:y+256]
                        if target.max()>=6:raise ValueError('Unexpected label ID')
                        cm+=np.bincount((target.astype(np.int64)*6+pred).ravel(),minlength=36).reshape(6,6)
                        pv=np.isin(pred,[2,3]);gv=np.isin(target,[2,3])
                        predveg+=int(pv.sum());gtveg+=int(gv.sum())
                        veg+=np.bincount((gv.astype(np.int64)*2+pv).ravel(),minlength=4).reshape(2,2)
                        if prediction is not None:prediction[y:y+256]=pred
                finally:
                    # Close disk mappings before TemporaryDirectory deletion on Windows.
                    if sums is not None:close_mapping(sums)
                    del sums
            row={'tile_id':tid,**metrics(cm),'vegetation_gt_percent':100*gtveg/size**2,
                 'vegetation_pred_percent':100*predveg/size**2,
                 'vegetation_error_pp':100*(predveg-gtveg)/size**2,
                 'vegetation_abs_error_pp':100*abs(predveg-gtveg)/size**2,
                 'vegetation_confusion_matrix':veg.tolist()}
            rows.append(row);global_cm+=cm
            if prediction is not None:
                from PIL import Image
                Image.fromarray(prediction).save(out/f'{tid}_ids.png')
                Image.fromarray(spec.colors[prediction]).save(out/f'{tid}_color.png')
            del image,gt
            spec.release_tile(tid)
            print(f'[{split}] {tid} mIoU={row["mIoU"]:.4f} vegetation_error_pp={row["vegetation_error_pp"]:+.3f}',flush=True)
        vegetation_cm=sum(np.array(r['vegetation_confusion_matrix']) for r in rows)
        result={'split':split,'protocol_sha256':spec.hash,**metrics(global_cm),'per_tile':rows,
                'vegetation_MAE_pp':float(np.mean([r['vegetation_abs_error_pp'] for r in rows])),
                'vegetation_max_abs_error_pp':float(max(r['vegetation_abs_error_pp'] for r in rows)),
                'vegetation_binary_metrics':metrics(vegetation_cm),
                'seconds':time.monotonic()-begin,'absent_class_rule':'exclude zero-union classes from mIoU; IoU=null'}
        write_json(out/'metrics.json',result)
        with (out/'coverage_per_tile.csv').open('w',encoding='utf-8-sig',newline='') as f:
            names=['tile_id','mIoU','OA','vegetation_gt_percent','vegetation_pred_percent','vegetation_error_pp','vegetation_abs_error_pp']
            writer=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        print(f'[{split} DONE] mIoU={result["mIoU"]:.6f} OA={result["OA"]:.6f} vegetation_MAE_pp={result["vegetation_MAE_pp"]:.4f} elapsed={duration(result["seconds"])}',flush=True)
        return result
    finally:
        model.train(was_training)
