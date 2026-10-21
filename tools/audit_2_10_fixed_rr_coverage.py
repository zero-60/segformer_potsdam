#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused GT/Pred/coverage diagnostics for Potsdam tile 2_10.

Uses existing Joint Robust-4 validators and their official degradation protocol.
Runs only Fixed Clean + Fog L3 and RR-DARF Clean + Gaussian Blur L3.
Place this file in tools/ and run from the repository root.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    import validate_model_b2_rgbnir_fixed1_joint_robust4_coverage as fixedval
    import validate_model_b2_darf_rr_coverage as rrval
except Exception as exc:
    raise RuntimeError(
        "Place this script in tools/ and confirm both existing validator scripts "
        "are present. Import error: " + repr(exc)
    ) from exc

TILE_ID = "2_10"
REFERENCE_GT_VEG_PIXELS = 17_861_449
REFERENCE_GT_COVERAGE_PCT = 49.615
CLASS_COLORS = np.array([
    [220, 220, 220], [220, 40, 40], [80, 190, 80],
    [20, 100, 40], [40, 90, 220], [230, 180, 30],
], dtype=np.uint8)
ACTIVE_OUTPUT_DIR: Optional[Path] = None
ACTIVE_CONDITION = "unknown"
ACTIVE_VARIANT = "unknown"
ACTIVE_CHECKPOINT = "unknown"
SAVE_NPY = False

ORIGINAL_FIXED_INFER = fixedval.infer_one_tile_joint
ORIGINAL_EVALUATE_CONDITION = fixedval.base.evaluate_condition
ORIGINAL_FIXED_BUILD_DATASET = fixedval.build_dataset
ORIGINAL_FIXED_PROBE = fixedval.joint_degradation_probe
ORIGINAL_FIXED_CONDITIONS = fixedval.suite_conditions
ORIGINAL_RR_BUILD_DATASET = rrval.build_dataset
ORIGINAL_RR_CLEAN_PROBE = rrval.clean_probe
ORIGINAL_RR_CONDITIONS = rrval.conditions_for_suite
ORIGINAL_PROGRESS = fixedval.JointValidationProgress

class SingleTileView:
    """Expose one tile but retain its original global index into the dataset."""
    _LOCAL = {"_dataset", "_tile_id", "_original_tile_index", "tile_ids", "window_coordinates", "spec"}
    def __init__(self, dataset, tile_id):
        ids = list(dataset.tile_ids)
        if tile_id not in ids:
            raise RuntimeError(f"Tile {tile_id!r} not found. Available: {ids}")
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_tile_id", tile_id)
        object.__setattr__(self, "_original_tile_index", ids.index(tile_id))
        object.__setattr__(self, "tile_ids", [tile_id])
        object.__setattr__(self, "window_coordinates", dataset.window_coordinates)
        object.__setattr__(self, "spec", dataset.spec)
    def __len__(self):
        return len(self.window_coordinates)
    def __getitem__(self, index):
        n = len(self.window_coordinates)
        if index < 0 or index >= n:
            raise IndexError(index)
        return self._dataset[self._original_tile_index * n + index]
    def load_full_label(self, tile_id):
        return self._dataset.load_full_label(tile_id)
    def __getattr__(self, name):
        return getattr(self._dataset, name)
    def __setattr__(self, name, value):
        if name in self._LOCAL: object.__setattr__(self, name, value)
        else: setattr(self._dataset, name, value)
    def __delattr__(self, name):
        if name in self._LOCAL: object.__delattr__(self, name)
        else: delattr(self._dataset, name)

def as_tile_view(dataset, tile_id=TILE_ID):
    if isinstance(dataset, SingleTileView) and dataset.tile_ids == [tile_id]:
        return dataset
    return SingleTileView(dataset, tile_id)

def to_label_numpy(value):
    if isinstance(value, torch.Tensor): value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
    if arr.ndim != 2: raise RuntimeError(f"Expected 2D label, got {arr.shape}")
    return arr.astype(np.uint8, copy=False)

def colorize(mask, nclasses):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid in range(nclasses): out[mask == cid] = CLASS_COLORS[cid]
    out[~((mask >= 0) & (mask < nclasses))] = (0, 0, 0)
    return out

def make_error_map(gt, pred, nclasses, veg_ids):
    valid = (gt >= 0) & (gt < nclasses)
    gv, pv = np.isin(gt, list(veg_ids)), np.isin(pred, list(veg_ids))
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[valid & (gt == pred)] = (245, 245, 245)
    out[valid & gv & ~pv] = (230, 50, 50)       # vegetation false negative
    out[valid & ~gv & pv] = (50, 100, 240)      # vegetation false positive
    out[valid & (gt != pred) & (gv == pv)] = (245, 200, 40)
    return out

def save_preview(gt_rgb, pred_rgb, err_rgb, path):
    side = 1200
    ims = []
    for arr in (gt_rgb, pred_rgb, err_rgb):
        im = Image.fromarray(arr, "RGB")
        im.thumbnail((side, side), Image.Resampling.NEAREST)
        ims.append(im.copy())
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    canvas = Image.new("RGB", (w * 3, h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (im, title) in enumerate(zip(ims, ["GT classes", "Pred classes", "Vegetation errors"])):
        x = idx * w
        draw.text((x + 8, 8), title, fill="black")
        canvas.paste(im, (x, 35))
    canvas.save(path)

def save_diagnostics(dataset, tile_id, prediction, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    gt = to_label_numpy(dataset.load_full_label(tile_id))
    pred = np.asarray(prediction).astype(np.uint8, copy=False)
    if gt.shape != pred.shape: raise RuntimeError(f"{tile_id}: GT {gt.shape} != Pred {pred.shape}")
    stats = fixedval.vegetation_tile_statistics(dataset=dataset, tile_id=tile_id, prediction=pred)
    nclasses = int(fixedval.base.NUM_CLASSES)
    names = list(fixedval.CLASS_NAMES)
    veg_ids = tuple(fixedval.VEGETATION_CLASS_IDS)
    valid = (gt >= 0) & (gt < nclasses)
    gt_v, pred_v = gt[valid].astype(np.int64), pred[valid].astype(np.int64)
    cm = np.bincount(gt_v * nclasses + pred_v, minlength=nclasses*nclasses).reshape(nclasses, nclasses)
    Image.fromarray(gt, "L").save(outdir / "GT_labels_id.png")
    Image.fromarray(pred, "L").save(outdir / "Pred_labels_id.png")
    gt_rgb, pred_rgb = colorize(gt, nclasses), colorize(pred, nclasses)
    err_rgb = make_error_map(gt, pred, nclasses, veg_ids)
    Image.fromarray(gt_rgb, "RGB").save(outdir / "GT_classes_color.png")
    Image.fromarray(pred_rgb, "RGB").save(outdir / "Pred_classes_color.png")
    Image.fromarray(err_rgb, "RGB").save(outdir / "vegetation_error_map.png")
    save_preview(gt_rgb, pred_rgb, err_rgb, outdir / "comparison_preview.png")
    if SAVE_NPY:
        np.save(outdir / "GT_labels.npy", gt)
        np.save(outdir / "Pred_labels.npy", pred)
    with (outdir / "class_pixel_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "gt_pixels", "pred_pixels"])
        writer.writeheader(); writer.writerows(stats.get("per_class_pixel_counts", []))
    with (outdir / "confusion_matrix_GT_rows_Pred_cols.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["GT \\ Pred"] + names)
        for i, name in enumerate(names): writer.writerow([name] + cm[i].astype(int).tolist())
    gt_count = int(stats["vegetation_gt_pixels"])
    payload: Dict[str, Any] = {
        "tile_id": tile_id, "variant": ACTIVE_VARIANT, "condition": ACTIVE_CONDITION,
        "checkpoint": ACTIVE_CHECKPOINT, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_names": names, "vegetation_class_ids": list(veg_ids),
        "vegetation_class_names": [names[i] for i in veg_ids], "coverage_metrics": stats,
        "reference_from_notion": {
            "gt_vegetation_pixels": REFERENCE_GT_VEG_PIXELS,
            "gt_coverage_percent_rounded": REFERENCE_GT_COVERAGE_PCT,
            "gt_pixel_count_matches": gt_count == REFERENCE_GT_VEG_PIXELS,
        },
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
        "visualization_legend": {
            "white": "GT class equals Pred class", "red": "GT vegetation but Pred non-vegetation (FN)",
            "blue": "Pred vegetation but GT non-vegetation (FP)", "yellow": "Other class mismatch", "black": "Ignored GT pixels",
        },
    }
    (outdir / "coverage_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnostic] {ACTIVE_VARIANT} | {ACTIVE_CONDITION} | GT veg={gt_count:,} ({stats['vegetation_gt_coverage_pct']:.4f}%) | "
          f"Pred veg={stats['vegetation_pred_pixels']:,} ({stats['vegetation_pred_coverage_pct']:.4f}%) | "
          f"abs error={stats['vegetation_coverage_abs_error_pp']:.4f} pp | "
          f"relative error={stats['vegetation_coverage_relative_error_pct']:.4f}%")
    if gt_count != REFERENCE_GT_VEG_PIXELS:
        print(f"[WARNING] GT vegetation count != Notion reference {REFERENCE_GT_VEG_PIXELS:,}; check tile, labels, class mapping and valid-pixel rule.")

def rebuild_summary(root):
    rows = []
    for p in root.rglob("coverage_audit.json"):
        try:
            x = json.loads(p.read_text(encoding="utf-8")); s = x["coverage_metrics"]
            rows.append({
                "tile_id": x.get("tile_id"), "variant": x.get("variant"), "condition": x.get("condition"),
                "checkpoint": x.get("checkpoint"), "valid_pixels": s.get("coverage_valid_pixels"),
                "GT_vegetation_pixels": s.get("vegetation_gt_pixels"), "Pred_vegetation_pixels": s.get("vegetation_pred_pixels"),
                "GT_coverage_percent": s.get("vegetation_gt_coverage_pct"), "Pred_coverage_percent": s.get("vegetation_pred_coverage_pct"),
                "signed_error_pp_Pred_minus_GT": s.get("vegetation_coverage_diff_pp"),
                "absolute_error_pp": s.get("vegetation_coverage_abs_error_pp"),
                "relative_error_percent": s.get("vegetation_coverage_relative_error_pct"),
                "relative_error_lt_10pct": s.get("vegetation_relative_error_lt_10pct"), "diagnostic_dir": str(p.parent),
            })
        except Exception as exc: print(f"[warning] Could not summarize {p}: {exc}")
    fields = ["tile_id","variant","condition","checkpoint","valid_pixels","GT_vegetation_pixels","Pred_vegetation_pixels","GT_coverage_percent","Pred_coverage_percent","signed_error_pp_Pred_minus_GT","absolute_error_pp","relative_error_percent","relative_error_lt_10pct","diagnostic_dir"]
    out = root / "summary_2_10_GT_Pred_coverage.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in sorted(rows, key=lambda r: (str(r["variant"]), str(r["condition"]))): w.writerow(row)
    print(f"Summary saved: {out}")

def patch_runtime():
    def wrapped_evaluate_condition(*args, **kwargs):
        global ACTIVE_OUTPUT_DIR, ACTIVE_CONDITION
        old_dir, old_condition = ACTIVE_OUTPUT_DIR, ACTIVE_CONDITION
        ACTIVE_OUTPUT_DIR = Path(kwargs["output_dir"])
        ACTIVE_CONDITION = str(kwargs.get("item", {}).get("condition", "unknown"))
        try: return ORIGINAL_EVALUATE_CONDITION(*args, **kwargs)
        finally: ACTIVE_OUTPUT_DIR, ACTIVE_CONDITION = old_dir, old_condition

    def wrapped_infer(**kwargs):
        prediction, runtime = ORIGINAL_FIXED_INFER(**kwargs)
        if str(kwargs.get("tile_id", "")) == TILE_ID and ACTIVE_OUTPUT_DIR is not None:
            save_diagnostics(kwargs["dataset"], TILE_ID, prediction, ACTIVE_OUTPUT_DIR / "tile_diagnostics" / TILE_ID)
        return prediction, runtime

    def fixed_build(**kwargs): return as_tile_view(ORIGINAL_FIXED_BUILD_DATASET(**kwargs))
    def fixed_probe(*, clean_dataset, degraded_dataset, condition):
        return ORIGINAL_FIXED_PROBE(clean_dataset=as_tile_view(clean_dataset), degraded_dataset=as_tile_view(degraded_dataset), condition=condition)
    def fixed_conditions(suite):
        return [x for x in ORIGINAL_FIXED_CONDITIONS(suite)
                if x.get("kind") == "clean" or (x.get("family") == "fog" and x.get("level") == "L3")]
    def rr_build(**kwargs): return as_tile_view(ORIGINAL_RR_BUILD_DATASET(**kwargs))
    def rr_clean(): return as_tile_view(ORIGINAL_RR_CLEAN_PROBE())
    def rr_conditions(suite):
        if suite != "formal": return []
        return [x for x in ORIGINAL_RR_CONDITIONS(suite)
                if x.get("kind") == "clean" or (x.get("family") == "gaussian_blur" and x.get("level") == "L3")]
    def one_tile_progress(*, total_conditions, tiles_per_condition, output_path, device):
        return ORIGINAL_PROGRESS(total_conditions=total_conditions, tiles_per_condition=1, output_path=output_path, device=device)

    fixedval.base.evaluate_condition = wrapped_evaluate_condition
    fixedval.infer_one_tile_joint = wrapped_infer
    fixedval.base.infer_one_tile = wrapped_infer
    fixedval.build_dataset = fixed_build
    fixedval.joint_degradation_probe = fixed_probe
    fixedval.suite_conditions = fixed_conditions
    fixedval.JointValidationProgress = one_tile_progress
    rrval.build_dataset = rr_build
    rrval.clean_probe = rr_clean
    rrval.conditions_for_suite = rr_conditions
    rrval.joint_val.JointValidationProgress = one_tile_progress
    rrval.joint_val.infer_one_tile_joint = wrapped_infer

def run_fixed(checkpoint, output_root, device, batch_size):
    global ACTIVE_VARIANT, ACTIVE_CHECKPOINT
    ACTIVE_VARIANT, ACTIVE_CHECKPOINT = "Fixed_g1.0_Joint_Robust4", str(checkpoint)
    old = sys.argv[:]
    sys.argv = ["validate_model_b2_rgbnir_fixed1_joint_robust4_coverage.py", "--variant", "fixed", "--checkpoint", str(checkpoint), "--output-root", str(output_root), "--suite", "all", "--device", device, "--batch-size", str(batch_size), "--num-workers", "0", "--save-predictions", "--force"]
    try:
        print("\n========== Fixed g=1.0: Clean + Fog L3 ==========")
        fixedval.main()
    finally: sys.argv = old

def run_rr(checkpoint, output_root, device, batch_size):
    global ACTIVE_VARIANT, ACTIVE_CHECKPOINT
    ACTIVE_VARIANT, ACTIVE_CHECKPOINT = "RR_DARF_Joint_Robust4", str(checkpoint)
    old = sys.argv[:]
    sys.argv = ["validate_model_b2_darf_rr_coverage.py", "--suite", "formal", "--checkpoint", str(checkpoint), "--output-root", str(output_root), "--device", device, "--batch-size", str(batch_size), "--num-workers", "0", "--save-predictions", "--force"]
    try:
        print("\n========== RR-DARF: Clean + Gaussian Blur L3 ==========")
        rrval.main()
    finally: sys.argv = old

def main():
    p = argparse.ArgumentParser(description="GT/Pred/coverage diagnostics for Potsdam tile 2_10")
    p.add_argument("--variant", choices=["fixed", "rr", "both"], default="both")
    p.add_argument("--fixed-checkpoint", type=Path, default=Path("outputs/training/b2_rgbnir_fixed1_joint_robust4/checkpoints/final.pt"))
    p.add_argument("--rr-checkpoint", type=Path, default=Path("outputs/training/b2_darf_rr_joint_robust4/checkpoints/final.pt"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/coverage_audit_2_10"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--save-npy", action="store_true", help="Also save full-resolution GT/Pred arrays; uses additional disk space.")
    args = p.parse_args()
    global SAVE_NPY
    SAVE_NPY = args.save_npy
    root = args.output_root.expanduser()
    if not root.is_absolute(): root = (PROJECT_ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    def resolve_checkpoint(cp):
        cp = cp.expanduser()
        return cp.resolve() if cp.is_absolute() else (PROJECT_ROOT / cp).resolve()
    fixed_cp, rr_cp = resolve_checkpoint(args.fixed_checkpoint), resolve_checkpoint(args.rr_checkpoint)
    patch_runtime()
    errors = []
    if args.variant in ("fixed", "both"):
        try:
            if not fixed_cp.is_file(): raise FileNotFoundError(f"Fixed checkpoint not found: {fixed_cp}")
            run_fixed(fixed_cp, root / "fixed_joint_robust4", args.device, args.batch_size)
        except Exception as exc:
            errors.append(("Fixed", repr(exc))); print(f"\n[ERROR] Fixed failed: {exc!r}")
    if args.variant in ("rr", "both"):
        try:
            if not rr_cp.is_file(): raise FileNotFoundError(f"RR-DARF checkpoint not found: {rr_cp}")
            run_rr(rr_cp, root / "rr_darf_joint_robust4", args.device, args.batch_size)
        except Exception as exc:
            errors.append(("RR-DARF", repr(exc))); print(f"\n[ERROR] RR-DARF failed: {exc!r}")
    rebuild_summary(root)
    if errors:
        print("\n有模型未完成。请检查 checkpoint 路径、完整性和首个报错。")
        for name, err in errors: print(f" - {name}: {err}")
        return 1
    print(f"\n完成。仅重跑 2_10 指定条件，不会重训。输出目录：{root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
