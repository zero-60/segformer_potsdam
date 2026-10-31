#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a SegFormer-B2 trained with Red + RedEdge (2-channel) on TLSNMSP.

This script requires a checkpoint trained for TWO input channels (Red, RedEdge).
It intentionally rejects the existing B,G,R,RedEdge four-channel checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

NUM_CLASSES = 7
TILE = 512
IGNORE_INDEX = 255
BANDS = ["Red", "RedEdge"]
CLASS_NAMES = [
    "building", "impervious_surface", "water", "barren",
    "tree", "low_vegetation", "background_other",
]
VEGETATION_IDS = (4, 5)  # train IDs: tree=4, low vegetation=5


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少协议清单：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"mask_converted 必须为单通道类别 ID 图：{path}, shape={arr.shape}")
    ids = set(np.unique(arr).astype(int).tolist())
    if ids - set(range(1, 8)):
        raise RuntimeError(f"标签 ID 必须为 1..7，实际为 {sorted(ids)}：{path}")
    return arr.astype(np.uint8) - 1


def read_red_rededge(path: Path) -> np.ndarray:
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
        raise RuntimeError(f"预期 uint16，实际 {chw.dtype}: {path}")
    # Assumed source order: Blue, Green, Red, RedEdge, NIR.
    return np.ascontiguousarray(chw[[2, 3]])


def resolve(data_root: Path, rel_path: str) -> Path:
    return (data_root / Path(rel_path.replace("/", os.sep))).resolve()


def load_stats(checkpoint: dict, checkpoint_path: Path) -> dict:
    protocol = checkpoint.get("protocol", {}) if isinstance(checkpoint, dict) else {}
    stats = protocol.get("band_normalization")
    if stats is None and isinstance(checkpoint, dict):
        stats = checkpoint.get("band_normalization")
    if stats is None:
        # Expected layout: outputs/training/<run>/checkpoints/best.pt
        candidates = [
            checkpoint_path.parent.parent / "band_normalization.json",
            checkpoint_path.parent / "band_normalization.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                stats = json.loads(candidate.read_text(encoding="utf-8"))
                break
    if not stats:
        raise RuntimeError(
            "找不到 Red+RedEdge 专用归一化参数。需要检查点 protocol.band_normalization，"
            "或训练输出目录中的 band_normalization.json。"
        )
    if stats.get("band_order") != BANDS:
        raise RuntimeError(f"归一化 band_order 必须是 {BANDS}，实际为 {stats.get('band_order')}")
    if len(stats.get("mean_dn", [])) != 2 or len(stats.get("std_dn", [])) != 2:
        raise RuntimeError("mean_dn/std_dn 必须各有两个值，对应 Red、RedEdge")
    return stats


class TwoBandTiles(Dataset):
    def __init__(self, rows, data_root: Path, stats):
        self.rows = rows
        self.data_root = data_root
        self.mean = np.asarray(stats["mean_dn"], dtype=np.float32).reshape(2, 1, 1)
        self.std = np.maximum(np.asarray(stats["std_dn"], dtype=np.float32), 1e-6).reshape(2, 1, 1)
        self.tiles = []
        for row_idx, row in enumerate(rows):
            label = read_label(resolve(data_root, row["label_path"]))
            h, w = label.shape
            # Non-overlapping tiles; partial edge tiles are padded rather than shifted.
            for y in range(0, h, TILE):
                for x in range(0, w, TILE):
                    self.tiles.append((row_idx, y, x, min(TILE, h-y), min(TILE, w-x)))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        row_idx, y, x, vh, vw = self.tiles[index]
        row = self.rows[row_idx]
        ms = read_red_rededge(resolve(self.data_root, row["multispectral_path"]))
        label = read_label(resolve(self.data_root, row["label_path"]))
        bands = ms[:, y:y+vh, x:x+vw].astype(np.float32)
        bands = (bands - self.mean) / self.std

        img_pad = np.zeros((2, TILE, TILE), dtype=np.float32)
        lab_pad = np.full((TILE, TILE), IGNORE_INDEX, dtype=np.int64)
        img_pad[:, :vh, :vw] = bands
        lab_pad[:vh, :vw] = label[y:y+vh, x:x+vw]
        return {
            "image": torch.from_numpy(img_pad),
            "label": torch.from_numpy(lab_pad),
            "sample_id": row["sample_id"],
            "x": x, "y": y, "valid_h": vh, "valid_w": vw,
        }


def build_model(pretrained_dir: Path, state: dict):
    if not pretrained_dir.is_dir() or not (pretrained_dir / "config.json").is_file():
        raise FileNotFoundError(f"本地预训练模型目录无效：{pretrained_dir}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(pretrained_dir), num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True, local_files_only=True,
    )
    old = model.segformer.encoder.patch_embeddings[0].proj
    if not isinstance(old, nn.Conv2d) or old.in_channels != 3:
        raise RuntimeError("无法适配预训练 SegFormer-B2 的首层三通道卷积")
    new = nn.Conv2d(
        2, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, dilation=old.dilation, groups=1,
        bias=old.bias is not None, padding_mode=old.padding_mode,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        # Initialization is overwritten by the trained two-channel checkpoint below.
        new.weight[:, 0:1].copy_(old.weight[:, 2:3])
        new.weight[:, 1:2].copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new
    model.config.num_channels = 2
    model.segformer.config.num_channels = 2

    key = "segformer.encoder.patch_embeddings.0.proj.weight"
    if key not in state:
        raise RuntimeError(f"检查点缺少模型参数 {key}")
    if state[key].ndim != 4 or state[key].shape[1] != 2:
        raise RuntimeError(
            f"该检查点不是两通道模型，首层权重形状为 {tuple(state[key].shape)}。"
            "当前 BGR+RedEdge 四通道 best.pt 不能直接用于 Red+RedEdge 验证。"
        )
    model.load_state_dict(state, strict=True)
    return model


def metrics_from_cm(cm: np.ndarray):
    f = cm.astype(np.float64)
    total = f.sum()
    diag = np.diag(f)
    gt, pred = f.sum(1), f.sum(0)
    oa = float(diag.sum()/total) if total else 0.0
    union = gt + pred - diag
    iou = np.divide(diag, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    miou = float(np.nanmean(iou)) if np.any(union > 0) else 0.0
    pe = float((gt*pred).sum()/(total*total)) if total else 0.0
    kappa = float((oa-pe)/(1-pe)) if total and abs(1-pe) > 1e-12 else 0.0
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "class_id": i, "class_name": name, "gt_pixels": int(gt[i]),
            "pred_pixels": int(pred[i]), "intersection": int(diag[i]),
            "iou": None if not np.isfinite(iou[i]) else float(iou[i]),
            "precision": float(diag[i]/pred[i]) if pred[i] else 0.0,
            "recall": float(diag[i]/gt[i]) if gt[i] else 0.0,
        })
    return {"OA": oa, "mIoU": miou, "Kappa": kappa, "total_pixels": int(total), "per_class": rows}


def evaluate(model, loader, device, split, output_dir: Path, amp: bool, save_predictions: bool):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_cov = {}
    loss_sum = 0.0
    t0 = time.time()
    pred_dir = output_dir / f"{split.lower()}_prediction_masks"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                raw = model(pixel_values=image).logits
                logits = F.interpolate(raw.float(), size=(TILE, TILE), mode="bilinear", align_corners=False)
            pred = logits.argmax(1).cpu().numpy()[0]
            gt = label.cpu().numpy()[0]
            vh, vw = int(batch["valid_h"][0]), int(batch["valid_w"][0])
            pred, gt = pred[:vh, :vw].astype(np.uint8), gt[:vh, :vw].astype(np.uint8)
            valid = gt != IGNORE_INDEX
            encoded = gt[valid].astype(np.int64)*NUM_CLASSES + pred[valid].astype(np.int64)
            cm += np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

            sid = batch["sample_id"][0]
            entry = image_cov.setdefault(sid, {"sample_id": sid, "valid_pixels": 0, "gt_vegetation_pixels": 0, "pred_vegetation_pixels": 0})
            entry["valid_pixels"] += int(valid.sum())
            entry["gt_vegetation_pixels"] += int(np.isin(gt[valid], VEGETATION_IDS).sum())
            entry["pred_vegetation_pixels"] += int(np.isin(pred[valid], VEGETATION_IDS).sum())

            loss_sum += float(F.cross_entropy(
                logits[:, :, :vh, :vw], label[:, :vh, :vw], ignore_index=IGNORE_INDEX
            ).item())

            if save_predictions:
                cv2.imwrite(str(pred_dir / f"{sid}_x{int(batch['x'][0])}_y{int(batch['y'][0])}.png"), pred)

    result = metrics_from_cm(cm)
    result.update({"split": split, "patch_count": len(loader), "mean_patch_CE": loss_sum/max(1, len(loader)), "seconds": time.time()-t0})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/f"{split.lower()}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir/f"{split.lower()}_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(result["per_class"])
    np.savetxt(output_dir/f"{split.lower()}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    cov_rows = []
    total_valid = total_gt = total_pred = 0
    for entry in image_cov.values():
        n = entry["valid_pixels"]
        gt_cov = entry["gt_vegetation_pixels"]/n*100 if n else 0.0
        pred_cov = entry["pred_vegetation_pixels"]/n*100 if n else 0.0
        signed = pred_cov - gt_cov
        cov_rows.append({
            **entry, "gt_coverage_percent": gt_cov, "pred_coverage_percent": pred_cov,
            "signed_error_pp": signed, "absolute_error_pp": abs(signed),
            "relative_error_percent": signed/gt_cov*100 if gt_cov else None,
        })
        total_valid += n
        total_gt += entry["gt_vegetation_pixels"]
        total_pred += entry["pred_vegetation_pixels"]
    with (output_dir/f"{split.lower()}_per_image_vegetation_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()) if cov_rows else [])
        writer.writeheader()
        writer.writerows(cov_rows)

    gt_cov = total_gt/total_valid*100 if total_valid else 0.0
    pred_cov = total_pred/total_valid*100 if total_valid else 0.0
    coverage = {
        "split": split, "valid_pixels": total_valid,
        "gt_vegetation_pixels": total_gt, "pred_vegetation_pixels": total_pred,
        "gt_coverage_percent": gt_cov, "pred_coverage_percent": pred_cov,
        "signed_error_pp": pred_cov-gt_cov, "absolute_error_pp": abs(pred_cov-gt_cov),
        "relative_error_percent": (pred_cov-gt_cov)/gt_cov*100 if gt_cov else None,
        "mean_per_image_absolute_error_pp": float(np.mean([r["absolute_error_pp"] for r in cov_rows])) if cov_rows else 0.0,
        "max_per_image_absolute_error_pp": float(np.max([r["absolute_error_pp"] for r in cov_rows])) if cov_rows else 0.0,
    }
    (output_dir/f"{split.lower()}_vegetation_coverage_summary.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{split}] OA={result['OA']:.4f} mIoU={result['mIoU']:.4f} Kappa={result['Kappa']:.4f} CE={result['mean_patch_CE']:.5f}")
    print(f"[{split} COVERAGE] GT={gt_cov:.4f}% Pred={pred_cov:.4f}% signed_error={pred_cov-gt_cov:+.4f} pp")
    print(f"[OUTPUT] {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Validate a trained Red+RedEdge two-channel SegFormer-B2")
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\MRISE\dataset\TLSNMSP"))
    ap.add_argument("--protocol-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "meris_tlsnmsp")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512"))
    ap.add_argument("--split", choices=["val", "test", "both"], default="test")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    ckpt_path = args.checkpoint.expanduser().resolve()
    pretrained_dir = args.pretrained_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据目录不存在：{data_root}")
    if not ckpt_path.is_file():
        raise SystemExit(f"检查点不存在：{ckpt_path}")
    cfg_path = protocol_dir / "dataset_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"缺少协议配置：{cfg_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("检查点格式不正确")
    model_id = checkpoint.get("model_id")
    if model_id not in (None, "B2_R_REDEDGE_2BAND_MERIS7", "B2_RED_REDEDGE_2BAND_MERIS7"):
        raise RuntimeError(
            f"model_id={model_id}，不是 Red+RedEdge 两通道模型。"
            "你当前提供的 BGR+RedEdge best.pt 是四通道模型，不能直接用于本脚本。"
        )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    stats = load_stats(checkpoint, ckpt_path)
    model = build_model(pretrained_dir, state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model.to(device)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else ckpt_path.parent.parent / "validation_red_rededge"
    print(f"[CHECKPOINT] {ckpt_path}")
    print(f"[INPUT] {BANDS} (TIFF band indices 3 and 4)")
    print(f"[DEVICE] {device} AMP={amp}")
    print("[TILING] non-overlapping 512x512; edge patches padded then cropped to valid pixels")
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        rows = read_csv(protocol_dir / f"{split}.csv")
        ds = TwoBandTiles(rows, data_root, stats)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        evaluate(model, loader, device, split.upper(), output_dir, amp, args.save_predictions)


if __name__ == "__main__":
    main()
