#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CLEAN Potsdam fixed g=1.0 category model or its matched baseline.
Preserves original 512 windows, stride=384, full-tile overlapping logit fusion.
Reports six-class metrics and vegetation coverage (Potsdam class IDs 2+3).
Optional prediction masks contain class IDs 0..5; --tiles supports a split subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset  # noqa: E402
from models.segformer_rgb import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
def get_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请检查当前环境；需要CPU检查可传 --device cpu")
    return device

from models.segformer_b2_rgbnir_fixed_category_clean import (  # noqa: E402
    build_model_b2_rgbnir_fixed_category_clean,
)


MODULE_VERSION = "1.0.0"
EVAL_PROTOCOL_VERSION = "B2 RGBNIR Fixed Clean Full-Tile Evaluation v1"
EXPECTED_MODEL_ID = "B2_RGBNIR_FIXED_CATEGORY_CLEAN"
DEFAULT_SPLIT = "val"
DEFAULT_ROW_CHUNK = 256


def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def strength_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return "g" + text.replace("-", "m").replace(".", "p")




def format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS for progress/ETA output."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-tile clean evaluation for fixed RGB+NIR SegFormer-B2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pretrained-checkpoint", type=Path, default=None)
    p.add_argument("--tiles", nargs="+", default=None, help="仅评估指定split内的tile，例如2_10")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT)
    p.add_argument(
        "--fixed-nir-strength",
        type=float,
        default=None,
        help="Normally inferred from checkpoint. If supplied, it must match the checkpoint.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--row-chunk", type=int, default=DEFAULT_ROW_CHUNK)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--keep-logit-memmaps",
        action="store_true",
        help="Keep per-tile float32 fused-logit sum files for debugging (very large).",
    )
    x = p.parse_args()
    if x.fixed_nir_strength is not None and not (0.0 <= x.fixed_nir_strength <= 1.0):
        p.error("--fixed-nir-strength must be in [0,1]")
    if x.row_chunk <= 0:
        p.error("--row-chunk must be > 0")
    return x


def checkpoint_strength(checkpoint: Mapping[str, Any]) -> float:
    value = checkpoint.get("fixed_nir_strength")
    if value is None:
        value = checkpoint.get("protocol", {}).get("fixed_nir_strength")
    if value is None:
        raise RuntimeError("Checkpoint does not record fixed_nir_strength.")
    value = float(value)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise RuntimeError(f"Invalid checkpoint fixed_nir_strength={value}")
    if value != 1.0:raise RuntimeError("本实验要求fixed NIR strength=1.0")
    return value


def validate_checkpoint_identity(checkpoint: Mapping[str, Any]) -> None:
    model_id = checkpoint.get("model_id")
    if model_id != EXPECTED_MODEL_ID:
        raise RuntimeError(
            f"Expected model_id={EXPECTED_MODEL_ID}, got {model_id}. "
            "Use this evaluator only for the clean fixed-NIR experiment."
        )
    if checkpoint.get("regime") != "clean":
        raise RuntimeError(f"Expected clean checkpoint, got regime={checkpoint.get('regime')}")
    if "model" not in checkpoint:
        raise RuntimeError("Checkpoint has no 'model' state_dict.")


def confusion_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    encoded = gt[valid].astype(np.int64, copy=False) * num_classes + pred[valid].astype(
        np.int64, copy=False
    )
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def kappa_from_confusion(cm):
    total=float(cm.sum())
    if total==0:return None
    oa=float(np.trace(cm))/total
    pe=float(np.dot(cm.sum(0).astype(float),cm.sum(1).astype(float)))/(total*total)
    return (oa-pe)/(1-pe) if abs(1-pe)>1e-12 else None


def vegetation_from_confusion(cm):
    total=int(cm.sum());ids=[2,3]  # frozen Potsdam IDs, not TLSNMSP IDs
    gt=int(cm[ids,:].sum());pred=int(cm[:,ids].sum());tp=int(cm[np.ix_(ids,ids)].sum())
    gt_cov=100.0*gt/total if total else None;pred_cov=100.0*pred/total if total else None
    err=pred_cov-gt_cov if total else None
    union=gt+pred-tp
    return dict(vegetation_class_ids=ids,valid_pixels=total,gt_vegetation_pixels=gt,pred_vegetation_pixels=pred,
                gt_coverage_pct=gt_cov,pred_coverage_pct=pred_cov,signed_error_pp=err,
                absolute_error_pp=abs(err) if err is not None else None,
                relative_signed_error_pct=100.0*(pred-gt)/gt if gt else None,
                relative_absolute_error_pct=100.0*abs(pred-gt)/gt if gt else None,
                vegetation_binary_iou=tp/union if union else None)


def metrics_from_confusion(confusion: np.ndarray) -> Dict[str, Any]:
    confusion = confusion.astype(np.int64, copy=False)
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp

    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    precision = np.divide(
        tp, pred_count, out=np.full_like(tp, np.nan), where=pred_count > 0
    )
    recall = np.divide(tp, gt_count, out=np.full_like(tp, np.nan), where=gt_count > 0)
    f1 = np.divide(2.0 * tp, gt_count + pred_count, out=np.full_like(tp, np.nan), where=(gt_count + pred_count)>0)
    total = float(confusion.sum())
    pixel_accuracy = float(tp.sum() / total) if total > 0 else float("nan")

    per_class: List[Dict[str, Any]] = []
    for class_id in range(NUM_CLASSES):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "iou": None if not np.isfinite(iou[class_id]) else float(iou[class_id]),
                "precision": (
                    None
                    if not np.isfinite(precision[class_id])
                    else float(precision[class_id])
                ),
                "recall": (
                    None if not np.isfinite(recall[class_id]) else float(recall[class_id])
                ),
                "f1": None if not np.isfinite(f1[class_id]) else float(f1[class_id]),
                "gt_pixels": int(gt_count[class_id]),
                "pred_pixels": int(pred_count[class_id]),
                "true_positive_pixels": int(tp[class_id]),
            }
        )

    return {
        "miou": float(np.nanmean(iou)),
        "pixel_accuracy": pixel_accuracy,
        "OA": pixel_accuracy,
        "Kappa": kappa_from_confusion(confusion),
        "coverage": vegetation_from_confusion(confusion),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "valid_pixels": int(confusion.sum()),
    }


def add_window_logits(
    accumulator: np.memmap,
    logits: np.ndarray,
    *,
    x: int,
    y: int,
) -> None:
    if logits.ndim != 3 or logits.shape[0] != NUM_CLASSES:
        raise RuntimeError(f"Unexpected window logits shape: {logits.shape}")
    h, w = int(logits.shape[1]), int(logits.shape[2])
    accumulator[:, y : y + h, x : x + w] += logits.astype(np.float32, copy=False)


def evaluate_one_tile(
    *,
    model,
    dataset: PotsdamSlidingWindowDataset,
    tile_index: int,
    device: torch.device,
    amp: bool,
    output_dir: Path,
    row_chunk: int,
    keep_memmap: bool,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    tile_id = dataset.tile_ids[tile_index]
    windows_per_tile = len(dataset.window_coordinates)
    first_index = tile_index * windows_per_tile
    tile_size = int(dataset.spec.tile_size)

    scratch_dir = output_dir / "scratch_logits"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = scratch_dir / f"{dataset.split_name}_{tile_id}_logit_sum_f32.dat"
    if memmap_path.exists():
        memmap_path.unlink()

    accumulator = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(NUM_CLASSES, tile_size, tile_size),
    )
    accumulator[:] = 0.0

    started = time.time()
    for window_offset in range(windows_per_tile):
        item = dataset[first_index + window_offset]
        if item["tile_id"] != tile_id:
            raise RuntimeError(
                f"Dataset is not tile-major: expected {tile_id}, got {item['tile_id']}"
            )

        rgb = item["rgb"].unsqueeze(0).to(device, non_blocking=True)
        nir = item["nir"].unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(rgb, nir)

        if logits.shape != (1, NUM_CLASSES, int(item["height"]), int(item["width"])):
            raise RuntimeError(f"Unexpected model logits shape: {tuple(logits.shape)}")
        logits_np = logits[0].detach().float().cpu().numpy()
        add_window_logits(
            accumulator,
            logits_np,
            x=int(item["x"]),
            y=int(item["y"]),
        )
        if (window_offset + 1) % 32 == 0 or (window_offset + 1) == windows_per_tile:
            windows_done = window_offset + 1
            window_elapsed = time.time() - started
            avg_window_seconds = window_elapsed / max(1, windows_done)
            tile_eta_seconds = avg_window_seconds * max(0, windows_per_tile - windows_done)
            print(
                f"[eval] tile={tile_id} windows={windows_done:03d}/{windows_per_tile:03d} "
                f"elapsed={format_duration(window_elapsed)} "
                f"tile_eta={format_duration(tile_eta_seconds)}",
                flush=True,
            )
        del rgb, nir, logits, logits_np

    accumulator.flush()

    # Mean-logit fusion would divide every class at a pixel by the same positive
    # overlap count. That division cannot change argmax, so the exact prediction
    # can be obtained directly from the accumulated logit sums. This avoids a
    # second full-tile count map without changing the class prediction.
    gt = dataset.load_full_label(tile_id).cpu().numpy()
    if gt.shape != (tile_size, tile_size):
        raise RuntimeError(f"Unexpected full GT shape for {tile_id}: {gt.shape}")

    tile_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    prediction = np.empty((tile_size, tile_size), dtype=np.uint8) if save_predictions else None
    for y0 in range(0, tile_size, row_chunk):
        y1 = min(tile_size, y0 + row_chunk)
        pred_chunk = np.argmax(np.asarray(accumulator[:, y0:y1, :]), axis=0).astype(
            np.int64, copy=False
        )
        if prediction is not None:prediction[y0:y1,:]=pred_chunk
        tile_confusion += confusion_from_arrays(
            gt[y0:y1, :],
            pred_chunk,
            NUM_CLASSES,
            IGNORE_INDEX,
        )

    if prediction is not None:
        import tifffile
        pred_dir=output_dir/"prediction_masks_class_ids";pred_dir.mkdir(parents=True,exist_ok=True)
        tifffile.imwrite(pred_dir/f"{tile_id}.tif",prediction,photometric="minisblack")
        del prediction
    del accumulator
    if not keep_memmap:
        try:
            memmap_path.unlink()
        except FileNotFoundError:
            pass

    result = metrics_from_confusion(tile_confusion)
    result.update(
        {
            "tile_id": tile_id,
            "windows": windows_per_tile,
            "seconds": float(time.time() - started),
            "logit_accumulator": (
                str(memmap_path) if keep_memmap else "deleted_after_scoring"
            ),
        }
    )
    return result


def main() -> None:
    x = parse_args()
    checkpoint_path = resolve(x.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(checkpoint)
    strength = checkpoint_strength(checkpoint)
    if x.fixed_nir_strength is not None and not math.isclose(
        float(x.fixed_nir_strength), strength, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Requested fixed strength {x.fixed_nir_strength} does not match checkpoint {strength}."
        )

    if x.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / f"b2_rgbnir_fixed_category_clean_{strength_tag(strength)}_{x.split}"
        )
    else:
        output_dir = resolve(x.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(x.device)
    amp = device.type == "cuda" and not x.no_amp

    dataset = PotsdamSlidingWindowDataset(PROJECT_ROOT, split=x.split)
    category_mode=checkpoint.get("protocol",{}).get("category_mode","full")
    if x.output_dir is None and category_mode == "baseline":
        output_dir=output_dir.with_name(output_dir.name+"_baseline")
        output_dir.mkdir(parents=True,exist_ok=True)
    local_default=Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
    pretrained=x.pretrained_checkpoint or (local_default if local_default.is_dir() else Path(checkpoint["model_meta"]["checkpoint"]))
    model, model_meta = build_model_b2_rgbnir_fixed_category_clean(
        PROJECT_ROOT, checkpoint=str(pretrained),category_mode=category_mode,fixed_strength=strength,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    print("=" * 120)
    print("Clean full-tile evaluation: SegFormer-B2 RGB+NIR Fixed")
    print("=" * 120)
    print(f"checkpoint          : {checkpoint_path}")
    print(f"split               : {x.split}")
    print(f"tiles               : {len(dataset.tile_ids)}")
    print(f"windows/tile        : {len(dataset.window_coordinates)}")
    print(f"fixed NIR strength  : {strength:.6f}")
    print(f"device              : {device}")
    print(f"AMP                 : {amp}")
    print("corruption          : NONE")
    print(f"output              : {output_dir}")
    print("=" * 120)

    started = time.time()
    global_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    tile_results: List[Dict[str, Any]] = []

    selected=[(i,t) for i,t in enumerate(dataset.tile_ids) if x.tiles is None or t in x.tiles]
    if x.tiles and set(x.tiles)-set(dataset.tile_ids):raise ValueError(f"指定tile不属于{x.split}划分")
    if not selected:raise ValueError("没有选择任何tile")
    for selected_index,(tile_index,tile_id) in enumerate(selected):
        print(
            f"[tile] {selected_index + 1:02d}/{len(selected):02d} {tile_id}",
            flush=True,
        )
        tile_result = evaluate_one_tile(
            model=model,
            dataset=dataset,
            tile_index=tile_index,
            device=device,
            amp=amp,
            output_dir=output_dir,
            row_chunk=x.row_chunk,
            keep_memmap=x.keep_logit_memmaps,
            save_predictions=x.save_predictions,
        )
        tile_results.append(tile_result)
        global_confusion += np.asarray(tile_result["confusion_matrix"], dtype=np.int64)
        tiles_done = selected_index + 1
        eval_elapsed = time.time() - started
        average_tile_seconds = eval_elapsed / max(1, tiles_done)
        eval_eta_seconds = average_tile_seconds * max(0, len(selected) - tiles_done)
        print(
            f"[tile done] {tile_id} | mIoU={tile_result['miou']:.6f} | "
            f"pixel_acc={tile_result['pixel_accuracy']:.6f} | "
            f"seconds={tile_result['seconds']:.1f} | "
            f"eval_eta={format_duration(eval_eta_seconds)}",
            flush=True,
        )

    global_metrics = metrics_from_confusion(global_confusion)
    absolute_errors=[r["coverage"]["absolute_error_pp"] for r in tile_results]
    relative_errors=[r["coverage"]["relative_absolute_error_pct"] for r in tile_results if r["coverage"]["relative_absolute_error_pct"] is not None]
    global_metrics["tile_coverage_mae_pp"]=float(np.mean(absolute_errors))
    global_metrics["tile_coverage_max_abs_error_pp"]=float(max(absolute_errors))
    global_metrics["tile_coverage_max_abs_relative_error_pct"]=float(max(relative_errors)) if relative_errors else None
    coverage_rows=[dict(tile_id=r["tile_id"],**r["coverage"]) for r in tile_results]
    for name,rows in [("per_tile_coverage.csv",coverage_rows),("per_class_metrics.csv",global_metrics["per_class"])]:
        with (output_dir/name).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savetxt(output_dir/"confusion_matrix.csv",global_confusion,delimiter=",",fmt="%d")
    print("[COVERAGE]",global_metrics["coverage"],"tile MAE pp:",global_metrics["tile_coverage_mae_pp"])

    result: Dict[str, Any] = {
        "evaluation_protocol_version": EVAL_PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "model_id": EXPECTED_MODEL_ID,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_1based": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_global_step": int(
            checkpoint.get("global_step", checkpoint.get("step", 0))
        ),
        "regime": "clean",
        "synthetic_corruption_enabled": False,
        "split": x.split,
        "fixed_nir_strength": strength,
        "fusion": "RGB-anchored fixed-strength NIR residual fusion",
        "full_tile_evaluation": {
            "tile_size": int(dataset.spec.tile_size),
            "crop_size": int(dataset.spec.crop_size),
            "stride": int(dataset.spec.sliding_stride),
            "windows_per_tile": len(dataset.window_coordinates),
            "fusion": "overlapping window logit sum; argmax equivalent to mean-logit fusion",
            "global_confusion_matrix": True,
        },
        "model_meta_from_builder": dict(model_meta),
        "evaluated_tile_ids": [r["tile_id"] for r in tile_results],
        "subset_evaluation": x.tiles is not None,
        "global": global_metrics,
        "tiles": tile_results,
        "duration_seconds": float(time.time() - started),
    }

    json_path = output_dir / f"{x.split}_clean_metrics.json"
    write_json_atomic(json_path, result)

    text_path = output_dir / f"{x.split}_clean_metrics.txt"
    with text_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"split: {x.split}\n")
        f.write(f"fixed_nir_strength: {strength}\n")
        f.write("synthetic_corruption: NONE\n")
        f.write(f"mIoU: {global_metrics['miou']:.8f}\n")
        f.write(f"pixel_accuracy: {global_metrics['pixel_accuracy']:.8f}\n")
        f.write("per_class_iou:\n")
        for row in global_metrics["per_class"]:
            f.write(
                f"  {row['class_id']} {row['class_name']}: "
                f"IoU={row['iou']} precision={row['precision']} "
                f"recall={row['recall']} f1={row['f1']}\n"
            )
        f.write("confusion_matrix:\n")
        for row in global_metrics["confusion_matrix"]:
            f.write("  " + " ".join(str(int(v)) for v in row) + "\n")

    print("=" * 120)
    print(f"[result] split={x.split} mIoU={global_metrics['miou']:.6f}")
    print(f"[result] pixel_accuracy={global_metrics['pixel_accuracy']:.6f}")
    for row in global_metrics["per_class"]:
        print(
            f"[class] {row['class_id']} {row['class_name']} | IoU={row['iou']}",
            flush=True,
        )
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
