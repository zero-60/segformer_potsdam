#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the RGB-only SegFormer-B2 TLSNMSP model.

This script is for checkpoints created by tools/train_model_b2_rgb_only.py.
It reads the separate RGB images listed by rgb_path in the existing protocol,
uses ImageNet normalization, evaluates every source pixel exactly once, and
reports OA/mIoU/Kappa, per-class metrics, confusion matrix, and vegetation
coverage error.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.segformer_b2_rgb_only import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    build_model_b2_rgb_only,
)

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL_DIR = ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_RUN_DIR = ROOT / "outputs" / "training" / "meris_b2_rgb_only_seed20261029"
DEFAULT_PRETRAINED = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
TILE = 512
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
VEGETATION_IDS = (4, 5)  # train IDs: tree=4, low vegetation=5


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少协议清单：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"协议清单为空：{path}")
    required = {"sample_id", "rgb_path", "label_path"}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"{path.name} 缺少字段 {sorted(missing)}，现有字段：{list(rows[0].keys())}")
    return rows


def resolve_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = root / Path(rel.replace("/", os.sep))
    return p.resolve()


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"RGB 影像读取失败：{path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"RGB 影像维度异常：{arr.shape}，文件：{path}")
    if arr.dtype != np.uint8:
        raise RuntimeError(f"RGB 图像预期 uint8，实际 {arr.dtype}：{path}")
    return np.ascontiguousarray(arr)


def load_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"标签必须是单通道类别 ID 图：{path}，shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}：{path}")
    ids = set(np.unique(arr).astype(int).tolist())
    if ids - set(range(1, NUM_CLASSES + 1)):
        raise RuntimeError(f"标签出现 1..{NUM_CLASSES} 以外的 ID：{sorted(ids)}；文件：{path}")
    # Raw mask IDs 1..7 -> model training IDs 0..6.
    return (arr.astype(np.int16) - 1).astype(np.uint8)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD)


def infer_image(model, rgb: np.ndarray, device: torch.device, amp: bool) -> np.ndarray:
    """Non-overlapping 512x512 inference; each original image pixel is counted once."""
    image = normalize_rgb(rgb)
    h, w = image.shape[:2]
    prediction = np.zeros((h, w), dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                tile = image[y0:min(y0 + TILE, h), x0:min(x0 + TILE, w)]
                valid_h, valid_w = tile.shape[:2]
                if valid_h < TILE or valid_w < TILE:
                    tile = np.pad(
                        tile,
                        ((0, TILE - valid_h), (0, TILE - valid_w), (0, 0)),
                        mode="edge",
                    )
                tensor = torch.from_numpy(
                    np.ascontiguousarray(tile.transpose(2, 0, 1))
                ).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits_low = model(pixel_values=tensor).logits
                logits = F.interpolate(
                    logits_low.float(), size=(TILE, TILE),
                    mode="bilinear", align_corners=False
                )
                pred_tile = logits.argmax(dim=1)[0, :valid_h, :valid_w].cpu().numpy().astype(np.uint8)
                prediction[y0:y0 + valid_h, x0:x0 + valid_w] = pred_tile
                del tensor, logits_low, logits, pred_tile
    return prediction


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape != pred.shape:
        raise RuntimeError(f"GT/Pred 尺寸不一致：{gt.shape} vs {pred.shape}")
    valid = (gt >= 0) & (gt < NUM_CLASSES)
    encoded = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(cm: np.ndarray) -> dict[str, Any]:
    cmf = cm.astype(np.float64)
    diagonal = np.diag(cmf)
    gt_counts = cmf.sum(axis=1)
    pred_counts = cmf.sum(axis=0)
    total = float(cmf.sum())
    oa = float(diagonal.sum() / total) if total else 0.0
    union = gt_counts + pred_counts - diagonal
    iou = np.divide(diagonal, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    miou = float(np.nanmean(iou)) if np.any(union > 0) else 0.0
    pe = float((gt_counts * pred_counts).sum() / total**2) if total else 0.0
    kappa = float((oa - pe) / (1.0 - pe)) if total and abs(1.0 - pe) > 1e-12 else 0.0

    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        per_class.append({
            "train_id": i,
            "raw_label_id": i + 1,
            "class_name": name,
            "gt_pixels": int(gt_counts[i]),
            "pred_pixels": int(pred_counts[i]),
            "intersection_pixels": int(diagonal[i]),
            "IoU": None if not np.isfinite(iou[i]) else float(iou[i]),
            "precision": float(diagonal[i] / pred_counts[i]) if pred_counts[i] else 0.0,
            "recall": float(diagonal[i] / gt_counts[i]) if gt_counts[i] else 0.0,
        })
    return {
        "OA": oa,
        "mIoU": miou,
        "Kappa": kappa,
        "total_valid_pixels": int(total),
        "per_class": per_class,
    }


def calculate_coverage(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    total = int(gt.size)
    gt_veg = int(np.isin(gt, VEGETATION_IDS).sum())
    pred_veg = int(np.isin(pred, VEGETATION_IDS).sum())
    gt_pct = gt_veg * 100.0 / total if total else 0.0
    pred_pct = pred_veg * 100.0 / total if total else 0.0
    error_pp = pred_pct - gt_pct
    relative_error = error_pp / gt_pct * 100.0 if gt_pct > 0 else None
    return {
        "valid_pixels": total,
        "gt_vegetation_pixels": gt_veg,
        "pred_vegetation_pixels": pred_veg,
        "gt_vegetation_coverage_pct": gt_pct,
        "pred_vegetation_coverage_pct": pred_pct,
        "coverage_signed_error_pp": error_pp,
        "coverage_absolute_error_pp": abs(error_pp),
        "coverage_relative_signed_error_pct": relative_error,
        "coverage_relative_absolute_error_pct": abs(relative_error) if relative_error is not None else None,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"检查点不存在：{path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB-only SegFormer-B2 验证与植物覆盖度计算")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="默认加载 run-dir/checkpoints/best.pt")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    pretrained_path = args.pretrained_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "validation_rgb_only" / checkpoint_path.stem

    if not data_root.is_dir():
        raise SystemExit(f"数据目录不存在：{data_root}")
    if not (run_dir / "protocol.json").is_file():
        raise FileNotFoundError(
            f"缺少训练协议文件：{run_dir / 'protocol.json'}。"
            "请确认 run-dir 指向 RGB-only 训练输出目录。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("model_id") != "B2_RGB_ONLY_MERIS7":
        raise RuntimeError(
            f"检查点模型标识不匹配：{checkpoint.get('model_id')}。"
            "本脚本只用于 tools/train_model_b2_rgb_only.py 训练出的 RGB-only 模型。"
        )

    model, model_meta = build_model_b2_rgb_only(pretrained_path)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[MODEL] {checkpoint.get('model_id')} | best_epoch={checkpoint.get('best_epoch', checkpoint.get('epoch'))}")
    print(f"[DEVICE] {device} | AMP={amp}")
    print("[INPUT] standalone RGB image | RGB order | ImageNet mean/std")
    print(f"[OUTPUT] {output_dir}")

    all_results: dict[str, Any] = {}
    for split in splits:
        rows = read_rows(protocol_dir / f"{split}.csv")
        cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        per_image_rows: list[dict[str, Any]] = []
        abs_errors_pp: list[float] = []
        abs_relative_errors: list[float] = []
        total_gt_veg = total_pred_veg = total_pixels = 0
        started = time.time()

        for index, row in enumerate(rows, 1):
            rgb_path = resolve_path(data_root, row["rgb_path"])
            label_path = resolve_path(data_root, row["label_path"])
            rgb = load_rgb(rgb_path)
            gt = load_label(label_path)
            if rgb.shape[:2] != gt.shape:
                raise RuntimeError(
                    f"{row['sample_id']} RGB/标签尺寸不一致：RGB={rgb.shape[:2]}, GT={gt.shape}"
                )
            pred = infer_image(model, rgb, device, amp)
            cm = confusion_matrix(gt, pred)
            cm_total += cm
            coverage = calculate_coverage(gt, pred)
            total_pixels += coverage["valid_pixels"]
            total_gt_veg += coverage["gt_vegetation_pixels"]
            total_pred_veg += coverage["pred_vegetation_pixels"]
            abs_errors_pp.append(coverage["coverage_absolute_error_pp"])
            if coverage["coverage_relative_absolute_error_pct"] is not None:
                abs_relative_errors.append(coverage["coverage_relative_absolute_error_pct"])

            per_image_rows.append({
                "sample_id": row["sample_id"],
                "split": split,
                "width": int(gt.shape[1]),
                "height": int(gt.shape[0]),
                **coverage,
            })

            if args.save_predictions:
                pred_dir = output_dir / split / "prediction_masks_raw_ids"
                pred_dir.mkdir(parents=True, exist_ok=True)
                # Convert train IDs 0..6 back to raw mask IDs 1..7.
                raw_pred = (pred + 1).astype(np.uint8)
                out_path = pred_dir / f"{row['sample_id']}.png"
                if not cv2.imwrite(str(out_path), raw_pred):
                    raise RuntimeError(f"预测掩膜保存失败：{out_path}")

            if index % 5 == 0 or index == len(rows):
                print(
                    f"[{split.upper()}] {index}/{len(rows)} images "
                    f"| last={row['sample_id']} | elapsed={time.time()-started:.1f}s",
                    flush=True,
                )

        metrics = calculate_metrics(cm_total)
        gt_cov = total_gt_veg * 100.0 / total_pixels if total_pixels else 0.0
        pred_cov = total_pred_veg * 100.0 / total_pixels if total_pixels else 0.0
        error_pp = pred_cov - gt_cov
        result = {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "model_id": checkpoint.get("model_id"),
            "best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
            "input": "standalone RGB image, normalized using ImageNet mean/std",
            "image_count": len(rows),
            **{k: metrics[k] for k in ("OA", "mIoU", "Kappa", "total_valid_pixels")},
            "gt_vegetation_pixels": total_gt_veg,
            "pred_vegetation_pixels": total_pred_veg,
            "gt_vegetation_coverage_pct": gt_cov,
            "pred_vegetation_coverage_pct": pred_cov,
            "coverage_signed_error_pp": error_pp,
            "coverage_absolute_error_pp": abs(error_pp),
            "coverage_relative_signed_error_pct": error_pp / gt_cov * 100.0 if gt_cov else None,
            "coverage_relative_absolute_error_pct": abs(error_pp / gt_cov * 100.0) if gt_cov else None,
            "per_image_mean_absolute_error_pp": float(np.mean(abs_errors_pp)) if abs_errors_pp else None,
            "per_image_median_absolute_error_pp": float(np.median(abs_errors_pp)) if abs_errors_pp else None,
            "per_image_max_absolute_error_pp": float(np.max(abs_errors_pp)) if abs_errors_pp else None,
            "per_image_max_absolute_relative_error_pct": float(np.max(abs_relative_errors)) if abs_relative_errors else None,
            "evaluation_seconds": time.time() - started,
            "per_class": metrics["per_class"],
        }

        split_dir = output_dir / split
        write_json(split_dir / "metrics_summary.json", result)
        write_csv(split_dir / "per_image_coverage.csv", per_image_rows)
        write_csv(split_dir / "per_class_metrics.csv", metrics["per_class"])
        np.savetxt(split_dir / "confusion_matrix.csv", cm_total, fmt="%d", delimiter=",")
        all_results[split] = result

        print(
            f"[{split.upper()} RESULT] OA={result['OA']:.4f} "
            f"mIoU={result['mIoU']:.4f} Kappa={result['Kappa']:.4f} "
            f"GT_cov={gt_cov:.4f}% Pred_cov={pred_cov:.4f}% "
            f"abs_error={abs(error_pp):.4f}pp "
            f"max_image_abs_error={result['per_image_max_absolute_error_pp']:.4f}pp",
            flush=True,
        )

    write_json(output_dir / "validation_summary.json", {
        "checkpoint": str(checkpoint_path),
        "model_meta": model_meta,
        "results": all_results,
    })
    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
