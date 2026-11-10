import json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from cmx_data import encode,starts

def metrics(cm):
    cm=np.asarray(cm,dtype=np.int64);tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(6,np.nan),where=union>0)
    total=cm.sum();po=float(tp.sum()/total) if total else float('nan')
    pe=float(np.dot(cm.sum(0).astype(float),cm.sum(1))/total**2) if total else float('nan')
    return {'OA':po,'mIoU':float(np.nanmean(iou)),'IoU':[None if np.isnan(v) else float(v) for v in iou],
            'Kappa':float((po-pe)/(1-pe)) if pe<1 else None,'confusion_matrix':cm.tolist()}

@torch.inference_mode()
def evaluate(model,spec,split,device,out,amp=True,save_predictions=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model.eval();global_cm=np.zeros((6,6),np.int64);rows=[];begin=time.time()
    cfg=spec.p['validation_test_inference'];crop=cfg['crop_size'][0];stride=cfg['stride'][0]
    if cfg['overlap_fusion']!='uniform mean-logit accumulation':raise ValueError('Unexpected fusion')
    size=cfg["tile_size"][0]
    if cfg["tile_size"]!=[size,size]:raise ValueError("Square tiles required")
    coords=starts(size,crop,stride)
    for tid in spec.split[split]:
        image=spec.image(tid);gt=spec.label_ids(tid) if hasattr(spec,'label_ids') else spec.gt(tid);cm=np.zeros((6,6),np.int64)
        veg=np.zeros((2,2),np.int64);gtveg=predveg=0
        # Disk-backed logit sums: ~0.9GB temporary/tile, avoids GPU full-tile allocations.
        with tempfile.TemporaryDirectory(prefix='fusion_',dir=out) as tmp:
            sums=np.zeros((6,size,size),np.float32) if getattr(spec,'fusion_storage','disk')=='ram' else np.memmap(Path(tmp)/'logits.bin',dtype='float32',mode='w+',shape=(6,size,size));sums[:]=0
            count=np.zeros((size,size),np.uint16)
            prediction=np.empty((size,size),np.uint8) if save_predictions else None
            windows=[(y,x) for y in coords for x in coords]
            batch_size=getattr(spec,'eval_batch_size',1)
            for offset in range(0,len(windows),batch_size):
                batch=windows[offset:offset+batch_size]
                normalized=[spec.normalize(image[y:y+crop,x:x+crop]) for y,x in batch]
                rgb=torch.stack([r for r,n in normalized]);nir=torch.stack([n for r,n in normalized])
                with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                    logits=model(rgb.to(device),nir.to(device))
                values=logits.float().cpu().numpy()
                for value,(y,x) in zip(values,batch):
                    sums[:,y:y+crop,x:x+crop]+=value;count[y:y+crop,x:x+crop]+=1
                wi=offset+len(batch)
                if wi%64==0 or wi==len(windows):print(f'[{split}] tile={tid} window={wi}/{len(windows)}',flush=True)
            for y in range(0,size,256):
                cnt=count[y:y+256]
                if np.any(cnt==0):raise RuntimeError('Uncovered validation pixels')
                pred=(np.asarray(sums[:,y:y+256])/cnt[None]).argmax(0).astype(np.uint8)
                target=gt[y:y+256] if gt.ndim==2 else encode(gt[y:y+256],spec.colors)
                cm+=np.bincount((target.astype(np.int64)*6+pred).ravel(),minlength=36).reshape(6,6)
                pv=np.isin(pred,[2,3]);gv=np.isin(target,[2,3]);predveg+=int(pv.sum());gtveg+=int(gv.sum())
                veg+=np.bincount((gv.astype(np.int64)*2+pv).ravel(),minlength=4).reshape(2,2)
                if prediction is not None:prediction[y:y+256]=pred
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
        print(f'[{split}] {tid} mIoU={row["mIoU"]:.4f} vegetation_error_pp={row["vegetation_error_pp"]:.3f}',flush=True)
    result={'split':split,'protocol_sha256':spec.hash,**metrics(global_cm),'per_tile':rows,
            'vegetation_MAE_pp':float(np.mean([r['vegetation_abs_error_pp'] for r in rows])),
            'seconds':time.time()-begin,'absent_class_rule':'exclude zero-union classes from mIoU; IoU=null'}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
