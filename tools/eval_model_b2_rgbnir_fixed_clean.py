#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a CLEAN fixed-NIR SegFormer-B2 checkpoint on Potsdam val/test.

Project placement
-----------------
Copy this file to:
    tools/eval_model_b2_rgbnir_fixed_clean.py

The evaluation follows the repository's frozen full-tile protocol:
1) deterministic 512x512 sliding windows (stride=384),
2) accumulate overlapping window logits over each full 6000x6000 tile,
3) argmax only after full-tile fusion,
4) build tile/global confusion matrices from full-tile GT,
5) report per-class IoU and global mIoU.

No corruption/degradation is applied to RGB or NIR.

Run
---
    python tools/eval_model_b2_rgbnir_fixed_clean.py \
        --checkpoint outputs/training/b2_rgbnir_fixed_clean_g1p0_seed20260917/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
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
import train_model_b2_rgb as progress_base  # noqa: E402
from models.segformer_b2_rgbnir_fixed_clean import (  # noqa: E402
    build_model_b2_rgbnir_fixed_clean,
)


MODULE_VERSION = "1.0.0"
EVAL_PROTOCOL_VERSION = "B2 RGBNIR Fixed Clean Full-Tile Evaluation v1"
EXPECTED_MODEL_ID = "B2_RGBNIR_FIXED_CLEAN"
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
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.full_like(tp, np.nan),
        where=np.isfinite(precision + recall) & ((precision + recall) > 0),
    )
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
            print(
                f"[eval] tile={tile_id} windows={window_offset + 1:03d}/{windows_per_tile:03d}",
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
    for y0 in range(0, tile_size, row_chunk):
        y1 = min(tile_size, y0 + row_chunk)
        pred_chunk = np.argmax(np.asarray(accumulator[:, y0:y1, :]), axis=0).astype(
            np.int64, copy=False
        )
        tile_confusion += confusion_from_arrays(
            gt[y0:y1, :],
            pred_chunk,
            NUM_CLASSES,
            IGNORE_INDEX,
        )

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
            / f"b2_rgbnir_fixed_clean_{strength_tag(strength)}_{x.split}"
        )
    else:
        output_dir = resolve(x.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = progress_base.get_device(x.device)
    amp = device.type == "cuda" and not x.no_amp

    dataset = PotsdamSlidingWindowDataset(PROJECT_ROOT, split=x.split)
    model, model_meta = build_model_b2_rgbnir_fixed_clean(
        PROJECT_ROOT,
        fixed_strength=strength,
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

    for tile_index, tile_id in enumerate(dataset.tile_ids):
        print(
            f"[tile] {tile_index + 1:02d}/{len(dataset.tile_ids):02d} {tile_id}",
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
        )
        tile_results.append(tile_result)
        global_confusion += np.asarray(tile_result["confusion_matrix"], dtype=np.int64)
        print(
            f"[tile done] {tile_id} | mIoU={tile_result['miou']:.6f} | "
            f"pixel_acc={tile_result['pixel_accuracy']:.6f} | "
            f"seconds={tile_result['seconds']:.1f}",
            flush=True,
        )

    global_metrics = metrics_from_confusion(global_confusion)
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
