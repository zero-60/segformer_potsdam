#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a trained TLSNMSP SegFormer-B2 RGB+NIR model and calculate vegetation coverage.

This script follows the existing TLSNMSP seven-class protocol. It evaluates full images by
stitching sliding-window logits, so overlapping edge windows do not double-count pixels in
coverage or confusion-matrix metrics.

Vegetation definition (raw label IDs):
  5 = tree, 6 = low vegetation
Coverage = (tree pixels + low-vegetation pixels) / all valid labeled pixels * 100.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer_b2_rgbnir_meris import (  # noqa: E402
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    RAW_LABEL_TO_TRAIN_ID,
    build_model_b2_rgbnir_meris,
)

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PRETRAINED_CHECKPOINT = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "outputs" / "training" / "meris_b2_rgbnir_fixed_g1_seed20261028"
CROP_SIZE = 512
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据清单：{path}\n请先运行 tools/prepare_meris_tlsnmsp.py")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据配置：{path}\n请先运行 tools/prepare_meris_tlsnmsp.py")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("protocol_version") != "TLSNMSP_RGB_NIR_7CLASS_v1":
        raise RuntimeError(f"不支持的 protocol_version：{cfg.get('protocol_version')}")
    if len(cfg.get("class_names_train_id_order", [])) != NUM_CLASSES:
        raise RuntimeError("dataset_config.json 的类别数量与模型不一致")
    return cfg


def resolve_manifest_path(data_root: Path, rel: str) -> Path:
    return (data_root / Path(rel.replace("/", os.sep))).resolve()


def load_sample_arrays(rgb_path: Path, ms_path: Path, label_path: Path):
    """Load one image using the same RGB/NIR/label conventions as the training script."""
    rgb_raw = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if rgb_raw is None:
        raise RuntimeError(f"RGB 读取失败：{rgb_path}")
    if rgb_raw.ndim == 2:
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_GRAY2RGB)
    elif rgb_raw.ndim == 3 and rgb_raw.shape[2] == 4:
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGRA2RGB)
    elif rgb_raw.ndim == 3 and rgb_raw.shape[2] >= 3:
        rgb = cv2.cvtColor(rgb_raw[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"不支持的 RGB 形状：{rgb_raw.shape}，文件 {rgb_path}")
    if rgb.dtype != np.uint8:
        raise RuntimeError(f"RGB 必须是 uint8，实际为 {rgb.dtype}：{rgb_path}")

    ms = tifffile.imread(str(ms_path))
    if ms.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维，实际为 {ms.shape}：{ms_path}")
    if ms.shape[0] == 5:
        nir = ms[4]
    elif ms.shape[-1] == 5:
        nir = ms[:, :, 4]
    else:
        raise RuntimeError(f"无法识别五波段 TIFF 布局：{ms.shape}：{ms_path}")
    if nir.dtype != np.uint16:
        raise RuntimeError(f"NIR 应为 uint16，实际为 {nir.dtype}：{ms_path}")

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"标签读取失败：{label_path}")
    if label.ndim == 3:
        if (
            label.shape[2] >= 3
            and np.array_equal(label[:, :, 0], label[:, :, 1])
            and np.array_equal(label[:, :, 0], label[:, :, 2])
        ):
            label = label[:, :, 0]
        else:
            raise RuntimeError(f"mask_converted 应为单通道类别 ID 图：{label_path}，shape={label.shape}")

    if rgb.shape[:2] != nir.shape or label.shape != nir.shape:
        raise RuntimeError(
            f"RGB/NIR/标签尺寸不一致：RGB={rgb.shape}, NIR={nir.shape}, "
            f"label={label.shape}; 文件={rgb_path.stem}"
        )

    label_ids = set(np.unique(label).astype(int).tolist())
    unknown = label_ids - set(int(k) for k in RAW_LABEL_TO_TRAIN_ID.keys())
    if unknown:
        raise RuntimeError(f"标签出现未知原始 ID：{sorted(unknown)}；文件 {label_path}")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(nir), np.ascontiguousarray(label.astype(np.uint8))


def label_to_train_ids(label_raw: np.ndarray) -> np.ndarray:
    mapped = np.full(label_raw.shape, IGNORE_INDEX, dtype=np.int64)
    for raw_id, train_id in RAW_LABEL_TO_TRAIN_ID.items():
        mapped[label_raw == int(raw_id)] = int(train_id)
    if np.any(mapped == IGNORE_INDEX):
        unknown = np.unique(label_raw[mapped == IGNORE_INDEX]).astype(int).tolist()
        raise RuntimeError(f"原始标签无法映射为训练类别 ID：{unknown}")
    return mapped


def pack_inputs(rgb: np.ndarray, nir: np.ndarray, label_raw: np.ndarray, cfg: dict[str, Any]):
    rgb_f = rgb.astype(np.float32) / 255.0
    rgb_f = (rgb_f - RGB_MEAN[None, None, :]) / RGB_STD[None, None, :]
    nir_cfg = cfg["nir_normalization"]
    nir_mean = float(nir_cfg["mean_dn"])
    nir_std = max(float(nir_cfg["std_dn"]), 1e-6)
    nir_f = (nir.astype(np.float32) - nir_mean) / nir_std
    label = label_to_train_ids(label_raw)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_f.transpose(2, 0, 1))).float()
    nir_tensor = torch.from_numpy(np.ascontiguousarray(nir_f[None, :, :])).float()
    label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long()
    return rgb_tensor, nir_tensor, label_tensor


def window_starts(length: int, crop: int = CROP_SIZE) -> list[int]:
    if length < crop:
        raise RuntimeError(f"影像边长 {length} 小于窗口 {crop}；与训练协议一致，本脚本不自动缩放标签")
    starts = list(range(0, length - crop + 1, crop))
    last = length - crop
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def load_checkpoint(path: Path, device: str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_fixed_strength(payload: Any) -> float:
    """Try to recover fixed NIR strength from checkpoint metadata; fall back to protocol default 1.0."""
    wanted = {"fixed_strength", "fixed_nir_strength", "nir_fixed_strength"}

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key) in wanted and isinstance(value, (int, float)):
                    return float(value)
            for value in obj.values():
                result = walk(value)
                if result is not None:
                    return result
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                result = walk(value)
                if result is not None:
                    return result
        return None

    result = walk(payload)
    return 1.0 if result is None else result


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        state = payload
    else:
        raise RuntimeError("无法从 checkpoint 中识别模型权重；需要包含 model 或 state_dict 的 .pt 文件")

    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    return state


def confusion_update(confmat: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    valid = (gt >= 0) & (gt < NUM_CLASSES) & (gt != IGNORE_INDEX)
    if not np.any(valid):
        return
    encoded = gt[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    confmat += np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confmat: np.ndarray) -> dict[str, Any]:
    cm = confmat.astype(np.float64)
    total = float(cm.sum())
    diag = np.diag(cm)
    gt_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    oa = float(diag.sum() / total) if total else 0.0
    union = gt_counts + pred_counts - diag
    ious = np.divide(diag, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    present = union > 0
    miou = float(np.nanmean(ious)) if present.any() else 0.0
    pe = float((gt_counts * pred_counts).sum() / (total * total)) if total else 0.0
    kappa = float((oa - pe) / (1.0 - pe)) if total and abs(1.0 - pe) > 1e-12 else 0.0
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        precision = float(diag[i] / pred_counts[i]) if pred_counts[i] else 0.0
        recall = float(diag[i] / gt_counts[i]) if gt_counts[i] else 0.0
        per_class.append({
            "class_id_train": i,
            "class_name": str(name),
            "gt_pixels": int(gt_counts[i]),
            "pred_pixels": int(pred_counts[i]),
            "intersection_pixels": int(diag[i]),
            "IoU": None if not np.isfinite(ious[i]) else float(ious[i]),
            "precision": precision,
            "recall": recall,
        })
    return {
        "OA": oa,
        "mIoU": miou,
        "Kappa": kappa,
        "valid_pixels": int(total),
        "per_class": per_class,
    }


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned[:160] or "sample"


def coverage_values(gt: np.ndarray, pred: np.ndarray, vegetation_train_ids: list[int]) -> dict[str, Any]:
    valid = (gt >= 0) & (gt < NUM_CLASSES) & (gt != IGNORE_INDEX)
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise RuntimeError("整幅影像没有有效标签像素，无法计算植物覆盖度")
    gt_veg_pixels = int((valid & np.isin(gt, vegetation_train_ids)).sum())
    pred_veg_pixels = int((valid & np.isin(pred, vegetation_train_ids)).sum())
    gt_cov = 100.0 * gt_veg_pixels / valid_pixels
    pred_cov = 100.0 * pred_veg_pixels / valid_pixels
    signed_error_pp = pred_cov - gt_cov
    relative_error_pct = (signed_error_pp / gt_cov * 100.0) if gt_cov > 0 else None
    return {
        "valid_pixels": valid_pixels,
        "gt_vegetation_pixels": gt_veg_pixels,
        "pred_vegetation_pixels": pred_veg_pixels,
        "gt_vegetation_coverage_pct": gt_cov,
        "pred_vegetation_coverage_pct": pred_cov,
        "vegetation_coverage_error_pp": signed_error_pp,
        "vegetation_coverage_abs_error_pp": abs(signed_error_pp),
        "vegetation_coverage_relative_error_pct": relative_error_pct,
        "vegetation_coverage_abs_relative_error_pct": abs(relative_error_pct) if relative_error_pct is not None else None,
    }


def infer_full_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    nir: np.ndarray,
    label_raw: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer one full image; average logits where edge-aligned windows overlap."""
    height, width = label_raw.shape
    ys = window_starts(height)
    xs = window_starts(width)
    logit_sum = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    vote_count = np.zeros((height, width), dtype=np.uint16)

    model.eval()
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                rgb_patch = np.ascontiguousarray(rgb[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE])
                nir_patch = np.ascontiguousarray(nir[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE])
                label_patch = np.ascontiguousarray(label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE])
                rgb_t, nir_t, _ = pack_inputs(rgb_patch, nir_patch, label_patch, cfg)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                nir_t = nir_t.unsqueeze(0).to(device, non_blocking=True)

                if amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(rgb_t, nir_t)
                else:
                    logits = model(rgb_t, nir_t)
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("out"))
                if not torch.is_tensor(logits) or logits.ndim != 4:
                    raise RuntimeError("模型输出不是预期的 [B,C,H,W] logits 张量")
                if logits.shape[1] != NUM_CLASSES:
                    raise RuntimeError(f"模型输出类别数={logits.shape[1]}，预期={NUM_CLASSES}")
                if logits.shape[-2:] != (CROP_SIZE, CROP_SIZE):
                    logits = F.interpolate(logits.float(), size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
                patch_logits = logits[0].float().cpu().numpy()
                y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
                logit_sum[:, y0:y1, x0:x1] += patch_logits
                vote_count[y0:y1, x0:x1] += 1
                del rgb_t, nir_t, logits, patch_logits

    if np.any(vote_count == 0):
        raise RuntimeError("滑窗拼接后仍存在未预测像素，请检查影像尺寸与窗口起点")
    logit_sum /= vote_count[None, :, :].astype(np.float32)
    pred = np.argmax(logit_sum, axis=0).astype(np.int64)
    gt = label_to_train_ids(label_raw)
    return gt, pred


def save_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(path: Path, confmat: np.ndarray):
    names = [str(n) for n in CLASS_NAMES]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + names)
        for idx, name in enumerate(names):
            writer.writerow([name] + [int(v) for v in confmat[idx]])


def evaluate_split(
    split_name: str,
    rows: list[dict[str, str]],
    data_root: Path,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    amp: bool,
    vegetation_train_ids: list[int],
    output_dir: Path,
    save_predictions: bool,
    max_images: int,
) -> dict[str, Any]:
    if max_images > 0:
        rows = rows[:max_images]
    if not rows:
        raise RuntimeError(f"{split_name} 清单为空")

    split_dir = output_dir / split_name.lower()
    split_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = split_dir / "prediction_masks_raw_ids"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_rows: list[dict[str, Any]] = []
    gt_vegetation_pixels_total = 0
    pred_vegetation_pixels_total = 0
    valid_pixels_total = 0
    started = time.time()

    for image_idx, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", f"{split_name}_{image_idx:04d}")
        rgb_path = resolve_manifest_path(data_root, row["rgb_path"])
        ms_path = resolve_manifest_path(data_root, row["multispectral_path"])
        label_path = resolve_manifest_path(data_root, row["label_path"])
        for p in (rgb_path, ms_path, label_path):
            if not p.is_file():
                raise FileNotFoundError(f"样本 {sample_id} 缺少文件：{p}")

        rgb, nir, label_raw = load_sample_arrays(rgb_path, ms_path, label_path)
        gt, pred = infer_full_image(model, rgb, nir, label_raw, cfg, device, amp)
        image_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        confusion_update(image_cm, gt, pred)
        confusion_update(confmat, gt, pred)
        image_metrics = metrics_from_confusion(image_cm)
        coverage = coverage_values(gt, pred, vegetation_train_ids)

        out_row = {
            "split": split_name,
            "sample_id": sample_id,
            "width": int(label_raw.shape[1]),
            "height": int(label_raw.shape[0]),
            "OA": image_metrics["OA"],
            "mIoU": image_metrics["mIoU"],
            "Kappa": image_metrics["Kappa"],
            **coverage,
        }
        image_rows.append(out_row)
        gt_vegetation_pixels_total += coverage["gt_vegetation_pixels"]
        pred_vegetation_pixels_total += coverage["pred_vegetation_pixels"]
        valid_pixels_total += coverage["valid_pixels"]

        if save_predictions:
            # Convert train IDs back to original raw IDs 1..7 for easy comparison with mask_converted.
            pred_raw = np.zeros(pred.shape, dtype=np.uint8)
            for raw_id, train_id in RAW_LABEL_TO_TRAIN_ID.items():
                pred_raw[pred == int(train_id)] = int(raw_id)
            mask_path = prediction_dir / f"{safe_filename(sample_id)}_pred_raw_ids.png"
            if not cv2.imwrite(str(mask_path), pred_raw):
                raise RuntimeError(f"预测掩膜写入失败：{mask_path}")

        rel_err = coverage["vegetation_coverage_relative_error_pct"]
        rel_text = "N/A" if rel_err is None else f"{rel_err:+.2f}%"
        print(
            f"[{split_name.upper()} {image_idx:03d}/{len(rows):03d}] {sample_id} "
            f"OA={image_metrics['OA']:.4f} mIoU={image_metrics['mIoU']:.4f} "
            f"GT_veg={coverage['gt_vegetation_coverage_pct']:.2f}% "
            f"Pred_veg={coverage['pred_vegetation_coverage_pct']:.2f}% "
            f"coverage_error={coverage['vegetation_coverage_error_pp']:+.2f}pp "
            f"relative_error={rel_text}",
            flush=True,
        )
        del rgb, nir, label_raw, gt, pred

    overall_metrics = metrics_from_confusion(confmat)
    gt_cov = 100.0 * gt_vegetation_pixels_total / max(1, valid_pixels_total)
    pred_cov = 100.0 * pred_vegetation_pixels_total / max(1, valid_pixels_total)
    signed_error_pp = pred_cov - gt_cov
    relative_error_pct = (signed_error_pp / gt_cov * 100.0) if gt_cov > 0 else None
    abs_errors = [float(r["vegetation_coverage_abs_error_pp"]) for r in image_rows]
    rel_errors = [r["vegetation_coverage_abs_relative_error_pct"] for r in image_rows if r["vegetation_coverage_abs_relative_error_pct"] is not None]

    vegetation_raw_ids = [
        int(raw_id) for raw_id, train_id in RAW_LABEL_TO_TRAIN_ID.items()
        if int(train_id) in vegetation_train_ids
    ]
    vegetation_definition = " + ".join(
        f"raw ID {raw_id} ({CLASS_NAMES[int(RAW_LABEL_TO_TRAIN_ID[raw_id])]})"
        for raw_id in vegetation_raw_ids
    )
    coverage_summary = {
        "vegetation_raw_label_ids": vegetation_raw_ids,
        "vegetation_definition": vegetation_definition,
        "vegetation_train_ids": vegetation_train_ids,
        "denominator": "all valid labeled pixels in each full image; GT and prediction use the same valid-GT mask",
        "valid_pixels": int(valid_pixels_total),
        "gt_vegetation_pixels": int(gt_vegetation_pixels_total),
        "pred_vegetation_pixels": int(pred_vegetation_pixels_total),
        "gt_vegetation_coverage_pct": gt_cov,
        "pred_vegetation_coverage_pct": pred_cov,
        "vegetation_coverage_error_pp": signed_error_pp,
        "vegetation_coverage_abs_error_pp": abs(signed_error_pp),
        "vegetation_coverage_relative_error_pct": relative_error_pct,
        "vegetation_coverage_abs_relative_error_pct": abs(relative_error_pct) if relative_error_pct is not None else None,
        "mean_image_abs_coverage_error_pp": float(np.mean(abs_errors)) if abs_errors else None,
        "median_image_abs_coverage_error_pp": float(np.median(abs_errors)) if abs_errors else None,
        "max_image_abs_coverage_error_pp": float(np.max(abs_errors)) if abs_errors else None,
        "mean_image_abs_relative_error_pct": float(np.mean(rel_errors)) if rel_errors else None,
    }
    summary = {
        "split": split_name,
        "image_count_evaluated": len(image_rows),
        "metrics_scope": "full-resolution image pixels; overlapping sliding-window logits averaged before argmax",
        "classification_metrics": overall_metrics,
        "vegetation_coverage": coverage_summary,
        "elapsed_seconds": time.time() - started,
        "per_class": overall_metrics["per_class"],
    }
    (split_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_csv(split_dir / "per_image_coverage.csv", image_rows)
    save_csv(split_dir / "per_class_metrics.csv", overall_metrics["per_class"])
    save_confusion_csv(split_dir / "confusion_matrix.csv", confmat)
    np.save(split_dir / "confusion_matrix.npy", confmat)

    print(
        f"[{split_name.upper()} SUMMARY] OA={overall_metrics['OA']:.4f} "
        f"mIoU={overall_metrics['mIoU']:.4f} Kappa={overall_metrics['Kappa']:.4f} "
        f"GT_vegetation={gt_cov:.2f}% Pred_vegetation={pred_cov:.2f}% "
        f"coverage_error={signed_error_pp:+.2f}pp "
        f"mean_image_abs_error={coverage_summary['mean_image_abs_coverage_error_pp']:.2f}pp "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    print(f"[OUTPUT] {split_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate TLSNMSP SegFormer-B2 RGB+NIR and calculate full-image vegetation coverage."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TRAINING_DIR / "checkpoints" / "best.pt",
                        help="训练检查点：best.pt、latest.pt 或 final.pt")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CHECKPOINT,
                        help="本地 ADE20K SegFormer-B2 目录，用于按训练时相同结构构建模型")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果目录；默认写入 outputs/training/.../validation/<checkpoint名>")
    parser.add_argument("--split", choices=["val", "test", "both"], default="val",
                        help="默认只验证 val；正式最终评估时再使用 test 或 both")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fixed-nir-strength", type=float, default=None,
                        help="固定 NIR 融合强度；默认从 checkpoint 元数据读取，读取不到时用 1.0")
    parser.add_argument("--vegetation-raw-ids", type=int, nargs="+", default=[5, 6],
                        help="原始标签中定义为植被的类别 ID，默认 5 6（树木+低矮植被）")
    parser.add_argument("--max-images", type=int, default=0,
                        help="仅评估前 N 张影像用于快速检查；0 表示完整评估")
    parser.add_argument("--save-predictions", action="store_true",
                        help="额外保存全分辨率预测掩膜 PNG，像素值为原始标签 ID 1..7")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度推理")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    pretrained_path = args.pretrained_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据根目录不存在：{data_root}")
    if not pretrained_path.is_dir():
        raise SystemExit(f"本地预训练模型目录不存在：{pretrained_path}")
    if not (pretrained_path / "config.json").is_file():
        raise SystemExit(f"本地预训练目录缺少 config.json：{pretrained_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"训练检查点不存在：{checkpoint_path}")
    if args.max_images < 0:
        raise SystemExit("--max-images 必须大于或等于 0")

    raw_ids = [int(x) for x in args.vegetation_raw_ids]
    if not raw_ids or len(set(raw_ids)) != len(raw_ids):
        raise SystemExit("--vegetation-raw-ids 不能为空且不能重复")
    missing_raw_ids = [x for x in raw_ids if x not in RAW_LABEL_TO_TRAIN_ID]
    if missing_raw_ids:
        raise SystemExit(f"植被原始标签 ID 不在数据集映射中：{missing_raw_ids}")
    vegetation_train_ids = [int(RAW_LABEL_TO_TRAIN_ID[x]) for x in raw_ids]

    cfg = load_config(protocol_dir / "dataset_config.json")
    split_rows = {
        "val": read_csv_rows(protocol_dir / "val.csv"),
        "test": read_csv_rows(protocol_dir / "test.csv"),
    }
    if args.split == "val":
        selected_splits = ["val"]
    elif args.split == "test":
        selected_splits = ["test"]
    else:
        selected_splits = ["val", "test"]

    payload = load_checkpoint(checkpoint_path, "cpu")
    model_id = payload.get("model_id") if isinstance(payload, dict) else None
    if model_id and model_id != "B2_RGBNIR_FIXED_RESIDUAL_MERIS7":
        raise RuntimeError(
            f"checkpoint model_id={model_id!r}，与当前 TLSNMSP 七类固定残差 RGB+NIR 模型不一致，拒绝加载。"
        )

    # Guard against accidentally evaluating a checkpoint against a different split manifest.
    train_csv = protocol_dir / "train.csv"
    saved_manifest_hash = payload.get("train_manifest_sha256") if isinstance(payload, dict) else None
    if saved_manifest_hash and train_csv.is_file():
        import hashlib
        h = hashlib.sha256()
        with train_csv.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        current_manifest_hash = h.hexdigest()
        if current_manifest_hash != saved_manifest_hash:
            raise RuntimeError(
                "当前 train.csv 与 checkpoint 记录的 SHA-256 不一致。为避免用错误的数据划分验证，已停止。"
            )

    fixed_strength = args.fixed_nir_strength
    if fixed_strength is None:
        fixed_strength = find_fixed_strength(payload)
    if not 0.0 <= fixed_strength <= 1.0:
        raise SystemExit(f"fixed-nir-strength 必须位于 [0,1]，实际为 {fixed_strength}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前 PyTorch 无法使用 CUDA")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    amp = bool(device.type == "cuda" and not args.no_amp)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (
        DEFAULT_TRAINING_DIR / "validation" / checkpoint_path.stem
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROJECT] {PROJECT_ROOT}")
    print(f"[DATA] {data_root}")
    print(f"[PROTOCOL] {protocol_dir}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[PRETRAINED] {pretrained_path}")
    print(f"[DEVICE] {device} | AMP={amp} | fixed_nir_strength={fixed_strength}")
    print(f"[VEGETATION] raw IDs={raw_ids} | train IDs={vegetation_train_ids} | denominator=all valid labeled pixels")
    print(f"[SPLITS] val={len(split_rows['val'])} images, test={len(split_rows['test'])} images")
    print(f"[OUTPUT] {output_dir}")

    model, model_meta = build_model_b2_rgbnir_meris(
        PROJECT_ROOT, checkpoint=str(pretrained_path), fixed_strength=float(fixed_strength)
    )
    state_dict = extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    del payload, state_dict
    print(f"[MODEL LOAD PASS] strict=True | model_meta={json.dumps(model_meta, ensure_ascii=False, default=str)}")

    run_started = time.time()
    all_summaries = {}
    for split_name in selected_splits:
        all_summaries[split_name] = evaluate_split(
            split_name=split_name,
            rows=split_rows[split_name],
            data_root=data_root,
            cfg=cfg,
            model=model,
            device=device,
            amp=amp,
            vegetation_train_ids=vegetation_train_ids,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
            max_images=args.max_images,
        )
    run_summary = {
        "checkpoint": str(checkpoint_path),
        "model_id": model_id or "B2_RGBNIR_FIXED_RESIDUAL_MERIS7",
        "fixed_nir_strength": float(fixed_strength),
        "vegetation_raw_label_ids": raw_ids,
        "vegetation_train_ids": vegetation_train_ids,
        "vegetation_definition": "tree + low vegetation; default raw IDs 5 and 6",
        "splits_evaluated": selected_splits,
        "max_images_per_split": args.max_images,
        "save_predictions": bool(args.save_predictions),
        "elapsed_seconds": time.time() - run_started,
        "results": all_summaries,
    }
    (output_dir / "validation_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\n[VALIDATION COMPLETE] elapsed={time.time() - run_started:.1f}s")
    print(f"[SUMMARY] {output_dir / 'validation_run_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n验证被用户中断。已写入的结果文件会保留。", file=sys.stderr)
        raise SystemExit(130)
