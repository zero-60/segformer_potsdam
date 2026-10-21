#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit GT/Pred vegetation coverage for one full Potsdam tile using the repository's
SegFormer-B2 RGB+NIR Fixed-NIR evaluator/model/data pipeline.

Run from repository root:
python tools/audit_potsdam_tile_coverage.py \
  --checkpoint outputs/training/b2_rgbnir_fixed_clean_g1p0_seed20260917/checkpoints/final.pt \
  --tile-id 2_10 --device cuda

Outputs: per-tile GT/Pred class masks (.npy), palette PNGs, pixel counts,
coverage errors (percentage points and relative percent), and JSON/CSV reports.
This first version intentionally targets the fixed-NIR checkpoint architecture.
It does not silently pretend to load RR-DARF checkpoints.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from models.segformer_rgb import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
import train_model_b2_rgb as progress_base
from models.segformer_b2_rgbnir_fixed_clean import build_model_b2_rgbnir_fixed_clean

COLORS = np.array([
    [220, 220, 220],  # impervious
    [220, 40, 40],    # building
    [80, 190, 80],    # low vegetation
    [20, 100, 40],    # tree
    [40, 90, 220],    # car
    [230, 180, 30],   # clutter/background
    [0, 0, 0],        # ignore
], dtype=np.uint8)

def resolve(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p).resolve()

def save_palette(mask: np.ndarray, path: Path):
    # Label 255 is rendered black. Class IDs are 0..5.
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        rgb[mask == i] = COLORS[i]
    rgb[mask == IGNORE_INDEX] = COLORS[6]
    Image.fromarray(rgb).save(path)

def save_error(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, path: Path):
    # White = correct, red = false negative vegetation, blue = false positive vegetation,
    # yellow = other class mismatch; black = ignored GT.
    gt_veg = np.isin(gt, [2, 3])
    pr_veg = np.isin(pred, [2, 3])
    img = np.zeros((*gt.shape, 3), dtype=np.uint8)
    img[valid & (gt == pred)] = (245, 245, 245)
    img[valid & gt_veg & ~pr_veg] = (230, 50, 50)
    img[valid & ~gt_veg & pr_veg] = (50, 100, 240)
    img[valid & (gt != pred) & (gt_veg == pr_veg)] = (245, 200, 40)
    Image.fromarray(img).save(path)

def confusion_from_arrays(gt, pred):
    valid = (gt != IGNORE_INDEX) & (gt >= 0) & (gt < NUM_CLASSES)
    enc = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    return np.bincount(enc, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES), valid

def main():
    ap = argparse.ArgumentParser(description="Full-tile GT/Pred vegetation coverage audit.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tile-id", default="2_10", help="Exact tile ID, default 2_10.")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--device", default="cuda", help="Use cuda or cpu.")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--row-chunk", type=int, default=256)
    ap.add_argument("--save-npy", action="store_true", help="Save full GT and prediction arrays.")
    args = ap.parse_args()

    ckpt_path = resolve(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_id") != "B2_RGBNIR_FIXED_CLEAN":
        raise RuntimeError(
            f"This script expects B2_RGBNIR_FIXED_CLEAN; got {ckpt.get('model_id')!r}. "
            "Do not use it on RR-DARF. Use the model-specific evaluator for RR-DARF."
        )
    strength = ckpt.get("fixed_nir_strength", ckpt.get("protocol", {}).get("fixed_nir_strength"))
    if strength is None:
        raise RuntimeError("Checkpoint does not record fixed_nir_strength.")
    strength = float(strength)

    device = progress_base.get_device(args.device)
    ds = PotsdamSlidingWindowDataset(ROOT, split=args.split)
    matches = [i for i, tid in enumerate(ds.tile_ids) if str(tid).lower() == args.tile_id.lower()]
    if not matches:
        raise RuntimeError(f"Tile {args.tile_id!r} not found in split={args.split}. Available IDs: {ds.tile_ids}")
    ti = matches[0]
    tile_id = ds.tile_ids[ti]
    output = resolve(args.output_dir) if args.output_dir else ROOT / "outputs" / "coverage_audit" / f"{tile_id}_{args.split}_fixedg{strength:g}"
    output.mkdir(parents=True, exist_ok=True)

    model, _meta = build_model_b2_rgbnir_fixed_clean(ROOT, fixed_strength=strength)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    tile_size = int(ds.spec.tile_size)
    windows_per_tile = len(ds.window_coordinates)
    start = ti * windows_per_tile
    # Disk-backed accumulator avoids allocating a 6x6000x6000 tensor in RAM.
    scratch = output / "logit_sum.dat"
    if scratch.exists(): scratch.unlink()
    acc = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(NUM_CLASSES, tile_size, tile_size))
    acc[:] = 0
    started = time.time()
    use_amp = device.type == "cuda"
    for wi in range(windows_per_tile):
        item = ds[start + wi]
        rgb = item["rgb"].unsqueeze(0).to(device)
        nir = item["nir"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(rgb, nir)
        arr = logits[0].detach().float().cpu().numpy()
        x, y = int(item["x"]), int(item["y"])
        h, w = arr.shape[-2:]
        acc[:, y:y+h, x:x+w] += arr
        if (wi + 1) % 25 == 0 or wi + 1 == windows_per_tile:
            print(f"[infer] tile={tile_id} windows={wi+1}/{windows_per_tile}", flush=True)
        del rgb, nir, logits, arr
    acc.flush()
    gt = ds.load_full_label(tile_id).cpu().numpy().astype(np.uint8, copy=False)
    pred = np.zeros((tile_size, tile_size), dtype=np.uint8)
    # Row chunks bound RAM while argmaxing the full tile.
    for y0 in range(0, tile_size, args.row_chunk):
        y1 = min(tile_size, y0 + args.row_chunk)
        pred[y0:y1] = np.argmax(np.asarray(acc[:, y0:y1, :]), axis=0).astype(np.uint8)
    del acc
    try: scratch.unlink()
    except OSError: pass

    cm, valid = confusion_from_arrays(gt, pred)
    gt_veg = valid & np.isin(gt, [2, 3])
    pred_veg = valid & np.isin(pred, [2, 3])
    valid_n = int(valid.sum())
    gt_n, pred_n = int(gt_veg.sum()), int(pred_veg.sum())
    gt_cov = gt_n / valid_n * 100 if valid_n else float("nan")
    pred_cov = pred_n / valid_n * 100 if valid_n else float("nan")
    abs_pp = abs(pred_cov - gt_cov)
    rel_pct = abs_pp / gt_cov * 100 if gt_cov > 0 else None

    per_class = []
    for cid, name in enumerate(CLASS_NAMES):
        gt_count = int(cm[cid, :].sum())
        pred_count = int(cm[:, cid].sum())
        tp = int(cm[cid, cid])
        union = gt_count + pred_count - tp
        per_class.append({"class_id": cid, "class_name": name, "gt_pixels": gt_count,
                          "pred_pixels": pred_count, "true_positive_pixels": tp,
                          "iou": tp / union if union else None})
    report = {
        "tile_id": str(tile_id), "split": args.split, "checkpoint": str(ckpt_path),
        "model_id": ckpt.get("model_id"), "fixed_nir_strength": strength,
        "vegetation_definition": {"class_ids": [2, 3], "class_names": [CLASS_NAMES[2], CLASS_NAMES[3]]},
        "valid_pixels": valid_n, "gt_vegetation_pixels": gt_n, "pred_vegetation_pixels": pred_n,
        "gt_coverage_percent": gt_cov, "pred_coverage_percent": pred_cov,
        "absolute_coverage_error_percentage_points": abs_pp,
        "relative_coverage_error_percent": rel_pct,
        "gt_class_counts": {r["class_name"]: r["gt_pixels"] for r in per_class},
        "pred_class_counts": {r["class_name"]: r["pred_pixels"] for r in per_class},
        "per_class": per_class, "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
        "elapsed_seconds": time.time() - started,
        "notes": "Coverage denominator is valid GT pixels; vegetation is class 2 Low vegetation + class 3 Tree. Confirm this matches your project acceptance definition."
    }
    (output / "coverage_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "coverage_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tile_id","valid_pixels","GT_veg_pixels","Pred_veg_pixels","GT_coverage_percent",
                    "Pred_coverage_percent","absolute_error_pp","relative_error_percent"])
        w.writerow([tile_id, valid_n, gt_n, pred_n, gt_cov, pred_cov, abs_pp, rel_pct])
        w.writerow([])
        w.writerow(["class_id","class_name","GT_pixels","Pred_pixels","TP","IoU"])
        for r in per_class:
            w.writerow([r["class_id"], r["class_name"], r["gt_pixels"], r["pred_pixels"], r["true_positive_pixels"], r["iou"]])
    save_palette(gt, output / "GT_class_map.png")
    save_palette(pred, output / "Pred_class_map.png")
    save_error(gt, pred, valid, output / "GT_Pred_error_map.png")
    if args.save_npy:
        np.save(output / "GT_labels.npy", gt)
        np.save(output / "Pred_labels.npy", pred)
    print("\n===== COVERAGE AUDIT =====")
    print(f"Tile: {tile_id} | split={args.split} | device={device}")
    print(f"Valid pixels: {valid_n:,}")
    print(f"GT vegetation: {gt_n:,} ({gt_cov:.4f}%)")
    print(f"Pred vegetation: {pred_n:,} ({pred_cov:.4f}%)")
    print(f"Absolute coverage error: {abs_pp:.4f} percentage points")
    print(f"Relative coverage error: {rel_pct if rel_pct is not None else 'undefined (GT coverage is zero)'}")
    print(f"Saved to: {output}")
    print("Maps: GT_class_map.png, Pred_class_map.png, GT_Pred_error_map.png")
    print("NOTE: this evaluator only accepts the fixed-NIR checkpoint architecture.")
if __name__ == "__main__":
    main()
