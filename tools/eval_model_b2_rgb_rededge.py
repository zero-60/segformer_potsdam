#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the trained four-band SegFormer-B2 and calculate vegetation coverage.

Input MUST match train_model_b2_bgr_re4.py: first four multispectral TIFF
bands in B, G, R, RedEdge order. This script does not use the separate RGB file.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.segformer_b2_bgr_re4 import BAND_NAMES, CLASS_NAMES, NUM_CLASSES, build_model_b2_bgr_re4
from tools.train_model_b2_bgr_re4 import load_ms_chw, load_label, read_csv_rows, resolve_manifest_path

DEFAULT_DATA = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL = ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_RUN = ROOT / "outputs" / "training" / "meris_b2_bgr_re4_seed20261029"
DEFAULT_PRETRAINED = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
TILE = 512
VEG_IDS = (4, 5)  # model train IDs: tree=4, low_vegetation=5


def load_stats(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练时的波段归一化参数：{path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("band_order") != BAND_NAMES:
        raise RuntimeError(f"波段顺序不匹配：{d.get('band_order')}，预期 {BAND_NAMES}")
    mean, std = np.asarray(d["mean_dn"], np.float32), np.asarray(d["std_dn"], np.float32)
    if mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("band_normalization.json 中 mean/std 格式或数值异常")
    return mean, std


def confusion(gt, pred):
    encoded = gt.astype(np.int64).ravel() * NUM_CLASSES + pred.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def calc_metrics(cm):
    cmf = cm.astype(np.float64)
    diag, gt_n, pred_n = np.diag(cmf), cmf.sum(1), cmf.sum(0)
    total = cmf.sum()
    oa = float(diag.sum() / total) if total else 0.
    union = gt_n + pred_n - diag
    iou = np.divide(diag, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    miou = float(np.nanmean(iou)) if np.any(union > 0) else 0.
    pe = float((gt_n * pred_n).sum() / total**2) if total else 0.
    kappa = float((oa-pe)/(1-pe)) if total and abs(1-pe) > 1e-12 else 0.
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        classes.append({
            "train_id": i, "raw_label_id": i+1, "class_name": name,
            "gt_pixels": int(gt_n[i]), "pred_pixels": int(pred_n[i]),
            "intersection_pixels": int(diag[i]),
            "IoU": None if not np.isfinite(iou[i]) else float(iou[i]),
            "precision": float(diag[i]/pred_n[i]) if pred_n[i] else 0.,
            "recall": float(diag[i]/gt_n[i]) if gt_n[i] else 0.,
        })
    return {"OA": oa, "mIoU": miou, "Kappa": kappa, "total_valid_pixels": int(total), "per_class": classes}


def coverage(gt, pred):
    n = gt.size
    gt_px = int(np.isin(gt, VEG_IDS).sum())
    pr_px = int(np.isin(pred, VEG_IDS).sum())
    gt_pct, pr_pct = gt_px*100/n, pr_px*100/n
    err = pr_pct-gt_pct
    rel = err/gt_pct*100 if gt_pct else None
    return {
        "valid_pixels": n, "gt_vegetation_pixels": gt_px, "pred_vegetation_pixels": pr_px,
        "gt_vegetation_coverage_pct": gt_pct, "pred_vegetation_coverage_pct": pr_pct,
        "coverage_signed_error_pp": err, "coverage_absolute_error_pp": abs(err),
        "coverage_relative_signed_error_pct": rel,
        "coverage_relative_absolute_error_pct": abs(rel) if rel is not None else None,
    }


def infer_image(model, ms, mean, std, device, amp):
    # Match training: first 4 bands from multispectral TIFF, order B,G,R,RedEdge.
    bands = np.moveaxis(ms[:4], 0, -1).astype(np.float32)
    bands = np.ascontiguousarray((bands-mean.reshape(1,1,4))/std.reshape(1,1,4))
    h, w = bands.shape[:2]
    pred = np.zeros((h,w), np.uint8)
    # Non-overlapping windows: every source pixel contributes exactly once.
    model.eval()
    with torch.no_grad():
        for y in range(0,h,TILE):
            for x in range(0,w,TILE):
                tile = bands[y:min(y+TILE,h), x:min(x+TILE,w)]
                vh,vw = tile.shape[:2]
                if vh<TILE or vw<TILE:
                    tile = np.pad(tile, ((0,TILE-vh),(0,TILE-vw),(0,0)), mode="edge")
                t = torch.from_numpy(np.ascontiguousarray(tile.transpose(2,0,1))).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    low = model(pixel_values=t).logits
                full = F.interpolate(low.float(), size=(TILE,TILE), mode="bilinear", align_corners=False)
                p = full.argmax(1)[0,:vh,:vw].cpu().numpy().astype(np.uint8)
                pred[y:y+vh,x:x+vw] = p
    return pred


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="四波段 B,G,R,RedEdge 模型验证及植物覆盖度计算")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED)
    ap.add_argument("--split", choices=["val","test","both"], default="test")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    data_root, protocol, run = a.data_root.resolve(), a.protocol_dir.resolve(), a.run_dir.resolve()
    ckpt_path = a.checkpoint.resolve() if a.checkpoint else run/"checkpoints"/"best.pt"
    out = a.output_dir.resolve() if a.output_dir else run/"validation_rgb_rededge"/ckpt_path.stem
    mean,std = load_stats(run/"band_normalization.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("model_id") != "B2_BGR_REDEDGE_4BAND_MERIS7":
        raise RuntimeError(f"检查点不是本四波段模型：{ckpt.get('model_id')}")
    model,_ = build_model_b2_bgr_re4(a.pretrained_checkpoint.resolve())
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    splits = ["val","test"] if a.split == "both" else [a.split]
    print(f"[CHECKPOINT] {ckpt_path}")
    print(f"[MODEL] {ckpt.get('model_id')} best_epoch={ckpt.get('best_epoch', ckpt.get('epoch'))}")
    print(f"[DEVICE] {device} AMP={amp}")
    print(f"[INPUT] bands={BAND_NAMES} source=multispectral TIFF first 4 bands (B,G,R,RedEdge)")
    print(f"[OUTPUT] {out}")
    all_results = {}
    for split in splits:
        rows = read_csv_rows(protocol/f"{split}.csv")
        if not rows: raise RuntimeError(f"{split}.csv 为空")
        cm_all = np.zeros((NUM_CLASSES,NUM_CLASSES), np.int64)
        image_rows, abs_pp, abs_rel = [], [], []
        total_px = gt_veg = pr_veg = 0
        t0 = time.time()
        for i,row in enumerate(rows,1):
            ms_path = resolve_manifest_path(data_root,row["multispectral_path"])
            label_path = resolve_manifest_path(data_root,row["label_path"])
            ms, label_raw = load_ms_chw(ms_path), load_label(label_path)
            if ms.shape[1:] != label_raw.shape:
                raise RuntimeError(f"{row['sample_id']} 多光谱与标签尺寸不一致")
            gt = (label_raw.astype(np.int16)-1).astype(np.uint8)
            pr = infer_image(model,ms,mean,std,device,amp)
            cm = confusion(gt,pr); cm_all += cm
            cov = coverage(gt,pr)
            total_px += cov["valid_pixels"]; gt_veg += cov["gt_vegetation_pixels"]; pr_veg += cov["pred_vegetation_pixels"]
            abs_pp.append(cov["coverage_absolute_error_pp"])
            if cov["coverage_relative_absolute_error_pct"] is not None: abs_rel.append(cov["coverage_relative_absolute_error_pct"])
            image_rows.append({"sample_id":row["sample_id"],"split":split,"width":gt.shape[1],"height":gt.shape[0],**cov})
            if a.save_predictions:
                import cv2
                pdir=out/split/"prediction_masks_raw_ids"; pdir.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(pdir/f"{row['sample_id']}.png"),(pr+1).astype(np.uint8)):
                    raise RuntimeError(f"写入预测掩膜失败：{row['sample_id']}")
            if i%5==0 or i==len(rows):
                print(f"[{split.upper()}] {i}/{len(rows)} images elapsed={time.time()-t0:.1f}s",flush=True)
        met=calc_metrics(cm_all)
        gt_cov=gt_veg*100/total_px; pr_cov=pr_veg*100/total_px; err=pr_cov-gt_cov
        result={
            "split":split,"checkpoint":str(ckpt_path),"model_id":ckpt.get("model_id"),
            "best_epoch":ckpt.get("best_epoch",ckpt.get("epoch")),"input_bands":BAND_NAMES,
            "input_source":"first four multispectral TIFF bands; same order and normalization as training",
            "image_count":len(rows),**{k:met[k] for k in ("OA","mIoU","Kappa","total_valid_pixels")},
            "gt_vegetation_pixels":gt_veg,"pred_vegetation_pixels":pr_veg,
            "gt_vegetation_coverage_pct":gt_cov,"pred_vegetation_coverage_pct":pr_cov,
            "coverage_signed_error_pp":err,"coverage_absolute_error_pp":abs(err),
            "coverage_relative_signed_error_pct":err/gt_cov*100 if gt_cov else None,
            "coverage_relative_absolute_error_pct":abs(err/gt_cov*100) if gt_cov else None,
            "per_image_mean_absolute_error_pp":float(np.mean(abs_pp)),
            "per_image_median_absolute_error_pp":float(np.median(abs_pp)),
            "per_image_max_absolute_error_pp":float(np.max(abs_pp)),
            "per_image_max_absolute_relative_error_pct":float(np.max(abs_rel)) if abs_rel else None,
            "evaluation_seconds":time.time()-t0,"per_class":met["per_class"],
        }
        sd=out/split
        save_json(sd/"metrics_summary.json",result)
        write_csv(sd/"per_image_coverage.csv",image_rows)
        write_csv(sd/"per_class_metrics.csv",met["per_class"])
        np.savetxt(sd/"confusion_matrix.csv",cm_all,fmt="%d",delimiter=",")
        all_results[split]=result
        print(f"[{split.upper()} RESULT] OA={result['OA']:.4f} mIoU={result['mIoU']:.4f} Kappa={result['Kappa']:.4f} "
              f"GT_cov={gt_cov:.4f}% Pred_cov={pr_cov:.4f}% abs_error={abs(err):.4f}pp "
              f"max_image_abs_error={result['per_image_max_absolute_error_pp']:.4f}pp",flush=True)
    save_json(out/"validation_summary.json",{"checkpoint":str(ckpt_path),"results":all_results,
        "normalization":{"band_order":BAND_NAMES,"mean_dn":mean.tolist(),"std_dn":std.tolist()}})
    print(f"[DONE] Results saved to {out}")


if __name__ == "__main__":
    main()
