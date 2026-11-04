#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate TLSNMSP five-band SegFormer-B2 and calculate vegetation coverage."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.segformer_b2_bgr_re_nir5 import BAND_NAMES, CLASS_NAMES, NUM_CLASSES, build_model_b2_bgr_re_nir5

DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
PROTOCOL_DIR = ROOT / "data" / "processed" / "meris_tlsnmsp"
RUN_DIR = ROOT / "outputs" / "training" / "meris_b2_fiveband_weighted_ce"
PRETRAINED = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
TILE = 512
VEG_IDS = (4, 5)  # train IDs: Tree and Low vegetation


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"协议清单不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"协议清单为空：{path}")
    required = {"sample_id", "multispectral_path", "label_path"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"{path.name} 缺少字段：{sorted(missing)}")
    return rows


def resolve(root, rel):
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (root / Path(rel.replace("/", os.sep))).resolve()


def load_bands(path):
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维数组，实际 {arr.shape}: {path}")
    if arr.shape[0] == 5:
        chw = arr
    elif arr.shape[-1] == 5:
        chw = np.moveaxis(arr, -1, 0)
    else:
        raise RuntimeError(f"无法识别五波段 TIFF 布局：{arr.shape}: {path}")
    if chw.dtype != np.uint16:
        raise RuntimeError(f"多光谱 TIFF 预期 uint16，实际 {chw.dtype}: {path}")
    return np.ascontiguousarray(np.moveaxis(chw, 0, -1))


def load_label(path):
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"标签应为单通道类别 ID 图：{path}, shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}: {path}")
    ids = set(np.unique(arr).astype(int).tolist())
    if ids - set(range(1, NUM_CLASSES + 1)):
        raise RuntimeError(f"标签出现 1..7 以外的 ID：{sorted(ids)}; {path}")
    return (arr.astype(np.int16) - 1).astype(np.uint8)


def load_stats(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少归一化统计文件：{path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    saved, expected = list(stats.get("band_order", [])), list(BAND_NAMES)
    if saved != expected:
        raise RuntimeError(f"波段顺序不匹配：saved={saved}, expected={expected}")
    if len(stats.get("mean_dn", [])) != 5 or len(stats.get("std_dn", [])) != 5:
        raise RuntimeError("band_normalization.json 中 mean_dn/std_dn 必须各有 5 个值")
    if np.any(np.asarray(stats["std_dn"], dtype=float) <= 0):
        raise RuntimeError("波段标准差存在非正数")
    return stats


def infer(model, bands, stats, device, amp):
    mean = np.asarray(stats["mean_dn"], np.float32).reshape(1, 1, 5)
    std = np.asarray(stats["std_dn"], np.float32).reshape(1, 1, 5)
    img = np.ascontiguousarray((bands.astype(np.float32) - mean) / np.maximum(std, 1e-6))
    h, w = img.shape[:2]
    pred = np.zeros((h, w), dtype=np.uint8)
    model.eval()
    # Non-overlapping windows + edge padding: every original pixel counted exactly once.
    with torch.inference_mode():
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                tile = img[y:min(y + TILE, h), x:min(x + TILE, w)]
                vh, vw = tile.shape[:2]
                if vh < TILE or vw < TILE:
                    tile = np.pad(tile, ((0, TILE-vh), (0, TILE-vw), (0, 0)), mode="edge")
                tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    raw = model(pixel_values=tensor).logits
                logits = F.interpolate(raw.float(), size=(TILE, TILE), mode="bilinear", align_corners=False)
                pred[y:y+vh, x:x+vw] = logits.argmax(1)[0, :vh, :vw].cpu().numpy().astype(np.uint8)
                del tensor, raw, logits
    return pred


def confusion(gt, pred):
    if gt.shape != pred.shape:
        raise RuntimeError(f"GT/Pred 尺寸不一致：{gt.shape} vs {pred.shape}")
    enc = gt.astype(np.int64).ravel() * NUM_CLASSES + pred.astype(np.int64).ravel()
    return np.bincount(enc, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(cm):
    cm = cm.astype(np.float64)
    diag, rows, cols = np.diag(cm), cm.sum(1), cm.sum(0)
    total = cm.sum()
    oa = float(diag.sum()/total) if total else 0.0
    union = rows + cols - diag
    ious = np.divide(diag, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    miou = float(np.nanmean(ious)) if np.any(union > 0) else 0.0
    pe = float((rows*cols).sum()/(total*total)) if total else 0.0
    kappa = float((oa-pe)/(1-pe)) if total and abs(1-pe) > 1e-12 else 0.0
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        classes.append({
            "train_id": i, "raw_label_id": i+1, "class_name": name,
            "gt_pixels": int(rows[i]), "pred_pixels": int(cols[i]),
            "intersection_pixels": int(diag[i]),
            "IoU": float(ious[i]) if np.isfinite(ious[i]) else None,
            "precision": float(diag[i]/cols[i]) if cols[i] else 0.0,
            "recall": float(diag[i]/rows[i]) if rows[i] else 0.0,
        })
    return {"OA": oa, "mIoU": miou, "Kappa": kappa, "total_pixels": int(total), "per_class": classes}


def coverage(gt, pred):
    n = int(gt.size)
    gtv = int(np.isin(gt, VEG_IDS).sum())
    prv = int(np.isin(pred, VEG_IDS).sum())
    gtp = 100.0 * gtv / n if n else 0.0
    prp = 100.0 * prv / n if n else 0.0
    err = prp - gtp
    rel = err / gtp * 100.0 if gtp else None
    return {
        "valid_pixels": n, "gt_vegetation_pixels": gtv, "pred_vegetation_pixels": prv,
        "gt_vegetation_coverage_pct": gtp, "pred_vegetation_coverage_pct": prp,
        "coverage_signed_error_pp": err, "coverage_absolute_error_pp": abs(err),
        "coverage_relative_signed_error_pct": rel,
        "coverage_relative_absolute_error_pct": abs(rel) if rel is not None else None,
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="五波段加权CE模型验证、七类指标与植物覆盖度核验")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=PRETRAINED)
    ap.add_argument("--checkpoint", type=Path, default=None, help="默认使用 checkpoints/best.pt")
    ap.add_argument("--split", choices=["val", "test", "both"], default="test")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    data_root, protocol_dir, run_dir = [p.expanduser().resolve() for p in (args.data_root, args.protocol_dir, args.run_dir)]
    pretrained = args.pretrained_checkpoint.expanduser().resolve()
    ckpt_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "validation" / ckpt_path.stem
    stats = load_stats(run_dir / "band_normalization.json")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_id") != "B2_BGR_REDEDGE_NIR_5BAND_MERIS7":
        raise RuntimeError(f"检查点不是五波段模型：model_id={ckpt.get('model_id')}")
    training_protocol = ckpt.get("protocol", {})
    if training_protocol.get("experiment_id") != "MERIS5_WEIGHTED_CE_v1":
        raise RuntimeError("检查点不是本次五波段加权CE实验，请检查 --run-dir 或 --checkpoint")
    if training_protocol.get("class_weight_mode") != "sqrt_inverse":
        raise RuntimeError("检查点没有预期的 sqrt_inverse 类别加权记录")
    manifest_hash = hashlib.sha256((protocol_dir / "train.csv").read_bytes()).hexdigest()
    if manifest_hash != ckpt.get("train_manifest_sha256"):
        raise RuntimeError("当前train.csv与检查点训练协议不一致")
    saved_stats = training_protocol.get("band_normalization", {})
    for key in ("mean_dn", "std_dn"):
        if key not in saved_stats or not np.allclose(np.asarray(stats[key]), np.asarray(saved_stats[key]), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"运行目录归一化统计与检查点不同：{key}")
    print("[TRAINING CE WEIGHTS]", training_protocol.get("class_weights"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model, model_meta = build_model_b2_bgr_re_nir5(pretrained)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    print(f"[CHECKPOINT] {ckpt_path}")
    print(f"[MODEL] best_epoch={ckpt.get('best_epoch', ckpt.get('epoch'))} | device={device} | AMP={amp}")
    print(f"[BANDS] {list(BAND_NAMES)}")
    print(f"[OUTPUT] {out_dir}")

    all_results = {}
    for split in splits:
        rows = read_rows(protocol_dir / f"{split}.csv")
        cm_all = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        image_rows, abs_pps, abs_rels = [], [], []
        total_pixels = total_gtv = total_predv = 0
        start = time.time()
        for idx, row in enumerate(rows, 1):
            ms_path = resolve(data_root, row["multispectral_path"])
            lab_path = resolve(data_root, row["label_path"])
            bands, gt = load_bands(ms_path), load_label(lab_path)
            if bands.shape[:2] != gt.shape:
                raise RuntimeError(f"{row['sample_id']} 影像/标签尺寸不一致：{bands.shape[:2]} vs {gt.shape}")
            pred = infer(model, bands, stats, device, amp)
            cm_all += confusion(gt, pred)
            cov = coverage(gt, pred)
            total_pixels += cov["valid_pixels"]
            total_gtv += cov["gt_vegetation_pixels"]
            total_predv += cov["pred_vegetation_pixels"]
            abs_pps.append(cov["coverage_absolute_error_pp"])
            if cov["coverage_relative_absolute_error_pct"] is not None:
                abs_rels.append(cov["coverage_relative_absolute_error_pct"])
            image_rows.append({"sample_id": row["sample_id"], "split": split, "width": gt.shape[1], "height": gt.shape[0], **cov})
            if args.save_predictions:
                import cv2
                pred_dir = out_dir / split / "prediction_masks_raw_ids"
                pred_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(pred_dir / f"{row['sample_id']}.png"), (pred + 1).astype(np.uint8)):
                    raise RuntimeError(f"预测掩膜保存失败：{row['sample_id']}")
            if idx % 5 == 0 or idx == len(rows):
                print(f"[{split.upper()}] {idx}/{len(rows)} images | elapsed={time.time()-start:.1f}s", flush=True)

        m = metrics(cm_all)
        gt_cov = 100.0 * total_gtv / total_pixels if total_pixels else 0.0
        pred_cov = 100.0 * total_predv / total_pixels if total_pixels else 0.0
        err = pred_cov - gt_cov
        result = {
            "split": split, "checkpoint": str(ckpt_path), "model_id": ckpt["model_id"],
            "experiment_id": training_protocol["experiment_id"],
            "training_class_weights": training_protocol.get("class_weights"),
            "vegetation_raw_ids": [i + 1 for i in VEG_IDS],
            "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch")), "input_bands": list(BAND_NAMES),
            "image_count": len(rows), "OA": m["OA"], "mIoU": m["mIoU"], "Kappa": m["Kappa"],
            "total_pixels": m["total_pixels"], "gt_vegetation_pixels": total_gtv,
            "pred_vegetation_pixels": total_predv, "gt_vegetation_coverage_pct": gt_cov,
            "pred_vegetation_coverage_pct": pred_cov, "coverage_signed_error_pp": err,
            "coverage_absolute_error_pp": abs(err),
            "coverage_relative_signed_error_pct": err / gt_cov * 100.0 if gt_cov else None,
            "coverage_relative_absolute_error_pct": abs(err / gt_cov * 100.0) if gt_cov else None,
            "per_image_mean_absolute_error_pp": float(np.mean(abs_pps)) if abs_pps else None,
            "per_image_median_absolute_error_pp": float(np.median(abs_pps)) if abs_pps else None,
            "per_image_max_absolute_error_pp": float(np.max(abs_pps)) if abs_pps else None,
            "per_image_max_absolute_relative_error_pct": float(np.max(abs_rels)) if abs_rels else None,
            "evaluation_seconds": time.time()-start, "per_class": m["per_class"],
        }
        sd = out_dir / split
        write_json(sd / "metrics_summary.json", result)
        write_csv(sd / "per_image_coverage.csv", image_rows)
        write_csv(sd / "per_class_metrics.csv", m["per_class"])
        np.savetxt(sd / "confusion_matrix.csv", cm_all, fmt="%d", delimiter=",")
        all_results[split] = result
        print(f"[{split.upper()} RESULT] OA={m['OA']:.4f} mIoU={m['mIoU']:.4f} Kappa={m['Kappa']:.4f} "
              f"GT_cov={gt_cov:.4f}% Pred_cov={pred_cov:.4f}% abs_error={abs(err):.4f}pp "
              f"max_image_abs_error={result['per_image_max_absolute_error_pp']:.4f}pp", flush=True)

    write_json(out_dir / "validation_summary.json", {
        "checkpoint": str(ckpt_path), "model_meta": model_meta, "results": all_results,
        "training_loss": training_protocol.get("loss"),
        "evaluation_note": "Class weighting applies during training only; predictions use ordinary argmax and metrics are unweighted.",
        "inference_note": "Non-overlapping 512x512 tiles with edge padding; each source pixel counted once.",
    })
    print(f"[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
