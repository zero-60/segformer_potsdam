#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train/evaluate the existing SegFormer-B2 RGB+NIR architecture on TLSNMSP.

Default paths target the user's Windows training machine. This script does not
modify the original Potsdam training scripts or source dataset.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
# Prevent accidental Hugging Face Hub attempts; the model is loaded from a local folder.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training" / "meris_b2_rgbnir_fixed_g1_seed20261028"
CROP_SIZE = 512
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少协议清单：{path}\n请先运行 tools/prepare_meris_tlsnmsp.py")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据配置：{path}\n请先运行 tools/prepare_meris_tlsnmsp.py")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("protocol_version") != "TLSNMSP_RGB_NIR_7CLASS_v1":
        raise RuntimeError(f"不支持的 protocol_version: {cfg.get('protocol_version')}")
    if len(cfg.get("class_names_train_id_order", [])) != NUM_CLASSES:
        raise RuntimeError("dataset_config.json 的类别数量与模型不一致")
    return cfg


def resolve_manifest_path(data_root: Path, rel: str) -> Path:
    # Protocol paths are stored as slash-separated paths relative to data_root.
    return (data_root / Path(rel.replace("/", os.sep))).resolve()


@lru_cache(maxsize=10)
def load_sample_arrays(rgb_path_text: str, ms_path_text: str, label_path_text: str):
    """LRU cache: retain a few full-resolution samples to reduce repeated disk I/O."""
    rgb_path, ms_path, label_path = Path(rgb_path_text), Path(ms_path_text), Path(label_path_text)
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
        raise RuntimeError(f"RGB 必须是 uint8，实际 {rgb.dtype}：{rgb_path}")

    ms = tifffile.imread(str(ms_path))
    if ms.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维，实际 {ms.shape}：{ms_path}")
    if ms.shape[0] == 5:
        nir = ms[4]
    elif ms.shape[-1] == 5:
        nir = ms[:, :, 4]
    else:
        raise RuntimeError(f"无法识别五波段 TIFF 布局：{ms.shape}：{ms_path}")
    if nir.dtype != np.uint16:
        raise RuntimeError(f"NIR 应为 uint16，实际 {nir.dtype}：{ms_path}")

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"标签读取失败：{label_path}")
    if label.ndim == 3:
        if label.shape[2] >= 3 and np.array_equal(label[:, :, 0], label[:, :, 1]) and np.array_equal(label[:, :, 0], label[:, :, 2]):
            label = label[:, :, 0]
        else:
            raise RuntimeError(f"mask_converted 应是单通道类别 ID 图：{label_path}, shape={label.shape}")
    if rgb.shape[:2] != nir.shape or label.shape != nir.shape:
        raise RuntimeError(
            f"RGB/NIR/标签尺寸不一致：RGB={rgb.shape}, NIR={nir.shape}, label={label.shape}; {rgb_path.stem}"
        )
    ids = set(np.unique(label).astype(int).tolist())
    if ids - set(RAW_LABEL_TO_TRAIN_ID):
        raise RuntimeError(f"标签出现 1..7 以外的值：{sorted(ids)}; {label_path}")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(nir), np.ascontiguousarray(label.astype(np.uint8))


def label_to_train_ids(label_raw: np.ndarray) -> np.ndarray:
    # Verified raw label IDs are 1..7; map to contiguous model indices 0..6.
    mapped = label_raw.astype(np.int16) - 1
    if mapped.size and (mapped.min() < 0 or mapped.max() >= NUM_CLASSES):
        raise RuntimeError(f"映射后标签越界：min={mapped.min()}, max={mapped.max()}")
    return mapped.astype(np.uint8)


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


class MerisRandomCropDataset(Dataset):
    """One deterministic-random 512x512 crop per labeled image per epoch."""
    def __init__(self, rows, data_root: Path, cfg: dict[str, Any], *, seed: int, augment: bool):
        self.rows = rows
        self.data_root = data_root
        self.cfg = cfg
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb_path = resolve_manifest_path(self.data_root, row["rgb_path"])
        ms_path = resolve_manifest_path(self.data_root, row["multispectral_path"])
        label_path = resolve_manifest_path(self.data_root, row["label_path"])
        rgb, nir, label_raw = load_sample_arrays(str(rgb_path), str(ms_path), str(label_path))
        h, w = label_raw.shape
        if h < CROP_SIZE or w < CROP_SIZE:
            raise RuntimeError(f"样本小于 {CROP_SIZE}x{CROP_SIZE}: {row['sample_id']} {w}x{h}")
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx * 9_176)
        y0 = int(rng.integers(0, h - CROP_SIZE + 1))
        x0 = int(rng.integers(0, w - CROP_SIZE + 1))
        rgb = rgb[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        nir = nir[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        label_raw = label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

        if self.augment:
            k = int(rng.integers(0, 4))
            if k:
                rgb = np.rot90(rgb, k)
                nir = np.rot90(nir, k)
                label_raw = np.rot90(label_raw, k)
            if bool(rng.integers(0, 2)):
                rgb, nir, label_raw = rgb[:, ::-1], nir[:, ::-1], label_raw[:, ::-1]
            if bool(rng.integers(0, 2)):
                rgb, nir, label_raw = rgb[::-1, :], nir[::-1, :], label_raw[::-1, :]

        rgb_t, nir_t, label_t = pack_inputs(np.ascontiguousarray(rgb), np.ascontiguousarray(nir), np.ascontiguousarray(label_raw), self.cfg)
        return {"rgb": rgb_t, "nir": nir_t, "label": label_t, "sample_id": row["sample_id"]}


def window_starts(length: int, crop: int = CROP_SIZE) -> list[int]:
    if length < crop:
        raise RuntimeError(f"影像边长 {length} 小于窗口 {crop}，当前脚本不自动缩放标签")
    starts = list(range(0, length - crop + 1, crop))
    last = length - crop
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


class MerisSlidingPatchDataset(Dataset):
    """Non-overlapping 512 windows (last window aligned to edge if needed)."""
    def __init__(self, rows, data_root: Path, cfg: dict[str, Any]):
        self.rows = rows
        self.data_root = data_root
        self.cfg = cfg
        self.items = []
        for row_idx, row in enumerate(rows):
            rgb_path = resolve_manifest_path(data_root, row["rgb_path"])
            ms_path = resolve_manifest_path(data_root, row["multispectral_path"])
            label_path = resolve_manifest_path(data_root, row["label_path"])
            rgb, nir, label = load_sample_arrays(str(rgb_path), str(ms_path), str(label_path))
            ys = window_starts(label.shape[0])
            xs = window_starts(label.shape[1])
            for y in ys:
                for x in xs:
                    self.items.append((row_idx, y, x))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_idx, y0, x0 = self.items[idx]
        row = self.rows[row_idx]
        rgb_path = resolve_manifest_path(self.data_root, row["rgb_path"])
        ms_path = resolve_manifest_path(self.data_root, row["multispectral_path"])
        label_path = resolve_manifest_path(self.data_root, row["label_path"])
        rgb, nir, label_raw = load_sample_arrays(str(rgb_path), str(ms_path), str(label_path))
        rgb = rgb[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        nir = nir[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        label_raw = label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        rgb_t, nir_t, label_t = pack_inputs(rgb, nir, label_raw, self.cfg)
        return {"rgb": rgb_t, "nir": nir_t, "label": label_t, "sample_id": row["sample_id"], "x": x0, "y": y0}


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = int(gt_sorted.numel())
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard


def lovasz_softmax(raw_logits: torch.Tensor, labels_full: torch.Tensor) -> torch.Tensor:
    labels = F.interpolate(labels_full.unsqueeze(1).float(), size=raw_logits.shape[-2:], mode="nearest").squeeze(1).long()
    probs = torch.softmax(raw_logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
    labels = labels.reshape(-1)
    valid = labels != IGNORE_INDEX
    probs, labels = probs[valid], labels[valid]
    if labels.numel() == 0:
        return raw_logits.sum() * 0.0
    losses = []
    for class_id in range(NUM_CLASSES):
        fg = (labels == class_id).float()
        if fg.sum().item() == 0:
            continue
        errors = torch.abs(fg - probs[:, class_id])
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else raw_logits.sum() * 0.0


def make_class_weights(protocol_dir: Path, mode: str, device: torch.device):
    if mode == "none":
        return None
    rows = read_csv_rows(protocol_dir / "class_pixel_counts_by_split.csv")
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for row in rows:
        if row["split"] == "train":
            idx = int(row["train_label_id"])
            counts[idx] = float(row["pixel_count"])
    if np.any(counts <= 0):
        raise RuntimeError(f"训练集至少有一个类别像素数为 0，无法计算类别权重：{counts.tolist()}")
    weights = np.sqrt(counts.sum() / counts)
    weights /= weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_optimizer(model, base_lr: float, new_lr: float, weight_decay: float):
    groups = {"base_decay": [], "base_no_decay": [], "new_decay": [], "new_no_decay": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = name.startswith("nir_adapters.") or name.startswith("decode_head.classifier.")
        no_decay = param.ndim == 1 or name.endswith(".bias")
        key = ("new" if is_new else "base") + ("_no_decay" if no_decay else "_decay")
        groups[key].append(param)
    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        is_new = key.startswith("new")
        no_decay = key.endswith("no_decay")
        param_groups.append({
            "params": params,
            "lr": new_lr if is_new else base_lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "group_name": key,
        })
    return AdamW(param_groups)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-4, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return (1.0 - progress) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def confusion_update(confmat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
    pred = logits.argmax(dim=1).reshape(-1).to(torch.int64)
    target = labels.reshape(-1).to(torch.int64)
    valid = (target >= 0) & (target < NUM_CLASSES)
    encoded = target[valid] * NUM_CLASSES + pred[valid]
    confmat += torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def metrics_from_confusion(confmat: np.ndarray):
    cm = confmat.astype(np.float64)
    total = cm.sum()
    diag = np.diag(cm)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    oa = float(diag.sum() / total) if total else 0.0
    union = row + col - diag
    ious = np.divide(diag, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    present = union > 0
    miou = float(np.nanmean(ious)) if present.any() else 0.0
    pe = float((row * col).sum() / (total * total)) if total else 0.0
    kappa = float((oa - pe) / (1.0 - pe)) if total and abs(1.0 - pe) > 1e-12 else 0.0
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        precision = float(diag[i] / col[i]) if col[i] else 0.0
        recall = float(diag[i] / row[i]) if row[i] else 0.0
        per_class.append({
            "class_id": i,
            "class_name": name,
            "gt_pixels": int(row[i]),
            "pred_pixels": int(col[i]),
            "intersection": int(diag[i]),
            "iou": None if not np.isfinite(ious[i]) else float(ious[i]),
            "precision": precision,
            "recall": recall,
        })
    return {"OA": oa, "mIoU": miou, "Kappa": kappa, "total_pixels": int(total), "per_class": per_class}


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def torch_load_checkpoint(path: Path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_transfer_checkpoint(model, checkpoint_path: Path):
    """Load compatible Potsdam RGB+NIR weights while keeping the new 7-class head random."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"初始化 checkpoint 不存在：{checkpoint_path}")
    payload = torch_load_checkpoint(checkpoint_path, "cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        source_state = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        source_state = payload["state_dict"]
    elif isinstance(payload, dict):
        source_state = payload
    else:
        raise RuntimeError("无法识别初始化 checkpoint 格式：未找到 state_dict/model 字典")

    target_state = model.state_dict()
    compatible = {}
    unexpected = []
    shape_mismatch = []
    for original_key, value in source_state.items():
        if not torch.is_tensor(value):
            continue
        key = str(original_key)
        while key.startswith("module."):
            key = key[len("module."):]
        # Do not transfer Potsdam's six-class classifier to the different seven-class task.
        if key in ("decode_head.classifier.weight", "decode_head.classifier.bias"):
            continue
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        compatible[key] = value

    required_keys = {
        key for key in target_state
        if key not in ("decode_head.classifier.weight", "decode_head.classifier.bias")
    }
    missing = sorted(required_keys - set(compatible))
    if missing or shape_mismatch or unexpected:
        raise RuntimeError(
            "初始化权重与当前 RGB+NIR 模型结构不完全匹配，已拒绝部分加载以避免静默错误。\n"
            f"缺失参数（前 12 项）: {missing[:12]}\n"
            f"形状不匹配（前 12 项）: {shape_mismatch[:12]}\n"
            f"额外参数（前 12 项）: {unexpected[:12]}\n"
            "请确认 checkpoint 来自本仓库的固定残差 RGB+NIR 模型，而不是 DARF/门控模型或 RGB-only 模型。"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint_path),
        "compatible_parameter_tensors_loaded": len(compatible),
        "classifier_transferred": False,
        "note": "已加载兼容的编码器/融合模块参数；七类分类头保持新初始化。",
    }


def evaluate(model, loader, device, amp: bool, split_name: str, output_dir: Path | None = None):
    model.eval()
    confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            nir = batch["nir"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(rgb, nir)
                loss = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX)
            loss_sum += float(loss.item())
            n_batches += 1
            confusion_update(confmat, logits, labels)
    cm_np = confmat.numpy()
    metrics = metrics_from_confusion(cm_np)
    metrics["split"] = split_name
    metrics["CE_loss_mean_patch"] = loss_sum / max(1, n_batches)
    metrics["patch_count"] = n_batches
    if output_dir is not None:
        save_json(output_dir / f"{split_name.lower()}_metrics.json", metrics)
        with (output_dir / f"{split_name.lower()}_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics["per_class"][0].keys()))
            writer.writeheader()
            writer.writerows(metrics["per_class"])
        np.save(output_dir / f"{split_name.lower()}_confusion_matrix.npy", cm_np)
    return metrics


def get_args():
    ap = argparse.ArgumentParser(description="Train TLSNMSP using the repository's SegFormer-B2 RGB+NIR fixed fusion")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CHECKPOINT,
                    help="本地 ADE20K SegFormer-B2 文件夹，必须含 config.json 和 pytorch_model.bin/model.safetensors；不联网下载")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--base-lr", type=float, default=3e-5)
    ap.add_argument("--new-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lovasz-weight", type=float, default=0.5)
    ap.add_argument("--fixed-nir-strength", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 默认 0，避免多进程启动问题")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--class-weight-mode", choices=["none", "sqrt_inverse"], default="none")
    ap.add_argument("--resume", default="none", help="none、auto 或一个 checkpoint 路径")
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="可选：从兼容的已训练 Potsdam RGB+NIR 固定残差 checkpoint 初始化非分类头参数")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--smoke-test", action="store_true", help="仅运行一次前向+反向检查，不进入训练")
    return ap.parse_args()


def main():
    args = get_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise SystemExit("epochs、batch-size、grad-accum-steps 必须为正数")
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lovasz_weight < 0:
        raise SystemExit("num-workers/warmup-steps/lovasz-weight 参数不合法")
    if not 0.0 <= args.fixed_nir_strength <= 1.0:
        raise SystemExit("fixed-nir-strength 必须在 [0,1] 范围内")

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pretrained_checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据目录不存在：{data_root}")
    if not pretrained_checkpoint.is_dir():
        raise SystemExit(
            f"本地预训练模型目录不存在：{pretrained_checkpoint}\n"
            "请确认路径为 D:\\MRISE\\hf-cache\\segformer-b2-finetuned-ade-512-512，"
            "或通过 --pretrained-checkpoint 指定实际目录。"
        )
    if not (pretrained_checkpoint / "config.json").is_file():
        raise SystemExit(f"本地预训练目录缺少 config.json：{pretrained_checkpoint}")
    if not any((pretrained_checkpoint / name).is_file() for name in (
        "pytorch_model.bin", "model.safetensors",
        "pytorch_model.bin.index.json", "model.safetensors.index.json"
    )):
        raise SystemExit(f"本地预训练目录缺少模型权重文件：{pretrained_checkpoint}")
    cfg = load_config(protocol_dir / "dataset_config.json")
    train_rows = read_csv_rows(protocol_dir / "train.csv")
    val_rows = read_csv_rows(protocol_dir / "val.csv")
    test_rows = read_csv_rows(protocol_dir / "test.csv")
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("train/val/test 任一清单为空")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"[PROJECT] {PROJECT_ROOT}")
    print(f"[DATA] {data_root}")
    print(f"[PROTOCOL] {protocol_dir}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[PRETRAINED] {pretrained_checkpoint} | local_files_only=True")
    print(f"[DEVICE] {device} | AMP={amp} | classes={NUM_CLASSES}")
    print(f"[SPLIT] train={len(train_rows)} images, val={len(val_rows)} images, test={len(test_rows)} images")
    print(f"[NIR] train-only mean={cfg['nir_normalization']['mean_dn']:.3f}, std={cfg['nir_normalization']['std_dn']:.3f} DN")

    train_ds = MerisRandomCropDataset(train_rows, data_root, cfg, seed=args.seed, augment=True)
    val_ds = MerisSlidingPatchDataset(val_rows, data_root, cfg)
    test_ds = MerisSlidingPatchDataset(test_rows, data_root, cfg)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_common)
    # Validation/test are evaluated one patch at a time for memory safety and exact confusion aggregation.
    eval_loader_kwargs = dict(batch_size=1, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)
    print(f"[PATCHES] train samples/epoch={len(train_ds)}, val patches={len(val_ds)}, test patches={len(test_ds)}")

    model, model_meta = build_model_b2_rgbnir_meris(
        PROJECT_ROOT, checkpoint=str(pretrained_checkpoint), fixed_strength=args.fixed_nir_strength
    )
    init_checkpoint_info = None
    if args.init_checkpoint is not None:
        init_checkpoint_info = load_transfer_checkpoint(model, args.init_checkpoint)
        print("[INIT-CHECKPOINT]", json.dumps(init_checkpoint_info, ensure_ascii=False))
    else:
        print("[INIT-CHECKPOINT] 未指定；使用本地 ADE20K 预训练 B2 编码器初始化，七类分类头随机初始化。")
    model.to(device)
    if not args.no_gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        try:
            print("[GRAD-CHECKPOINTING]", model.enable_gradient_checkpointing())
        except Exception as exc:
            print(f"[WARN] gradient checkpointing could not be enabled: {exc}")

    if args.smoke_test:
        model.train()
        sample = train_ds[0]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        nir = sample["nir"].unsqueeze(0).to(device)
        label = sample["label"].unsqueeze(0).to(device)
        optimizer = make_optimizer(model, args.base_lr, args.new_lr, args.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            details = model(rgb, nir, return_details=True)
            ce = F.cross_entropy(details["logits"].float(), label, ignore_index=IGNORE_INDEX)
            lovasz = lovasz_softmax(details["raw_logits"], label)
            loss = ce + args.lovasz_weight * lovasz
        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke-test loss non-finite: {loss.item()}")
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        print("[SMOKE PASS]",
              f"sample={sample['sample_id']}",
              f"RGB={tuple(rgb.shape)}", f"NIR={tuple(nir.shape)}",
              f"label_range={int(label.min())}..{int(label.max())}",
              f"logits={tuple(details['logits'].shape)}", f"loss={loss.item():.6f}",
              f"parameters_with_grad={grad_count}")
        if device.type == "cuda":
            print(f"[CUDA] peak_allocated_GiB={torch.cuda.max_memory_allocated() / 1024**3:.3f}")
        return

    class_weights = make_class_weights(protocol_dir, args.class_weight_mode, device)
    if class_weights is not None:
        print("[CLASS-WEIGHTS]", [round(float(x), 4) for x in class_weights.detach().cpu()])

    optimizer = make_optimizer(model, args.base_lr, args.new_lr, args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, total_updates, args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    checkpoints = output_dir / "checkpoints"
    log_path = output_dir / "training_history.jsonl"
    has_old_run = checkpoints.exists() and any(checkpoints.glob("*.pt"))
    if args.resume.lower() == "none" and (has_old_run or log_path.is_file()):
        raise RuntimeError(
            f"输出目录已存在训练检查点或日志：{output_dir}。为避免混合/覆盖实验，"
            "请使用 --resume auto 续训，或为新实验指定不同的 --output-dir。"
        )
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_miou = -1.0
    best_epoch = 0
    stale_epochs = 0
    train_manifest_hash = sha256_file(protocol_dir / "train.csv")

    resume_path = None
    if args.resume.lower() == "auto":
        candidate = checkpoints / "latest.pt"
        resume_path = candidate if candidate.is_file() else None
    elif args.resume.lower() != "none":
        resume_path = Path(args.resume).expanduser().resolve()
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint 不存在：{resume_path}")
        ckpt = torch_load_checkpoint(resume_path, "cpu")
        if ckpt.get("train_manifest_sha256") != train_manifest_hash:
            raise RuntimeError("resume checkpoint 的 train.csv 哈希与当前协议不同，拒绝自动续训")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_miou = float(ckpt.get("best_miou", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        try:
            if "rng_python" in ckpt:
                random.setstate(ckpt["rng_python"])
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])
            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])
            if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception as exc:
            print(f"[WARN] 恢复随机状态失败，将继续使用当前随机状态：{exc}")
        print(f"[RESUME] {resume_path} start_epoch={start_epoch + 1}, global_step={global_step}, best_epoch={best_epoch}, best_val_mIoU={best_miou:.4f}")

    protocol_record = {
        "protocol_version": "TLSNMSP_RGB_NIR_7CLASS_v1",
        "model": model_meta,
        "init_checkpoint": init_checkpoint_info,
        "data_root": str(data_root),
        "protocol_dir": str(protocol_dir),
        "train_manifest_sha256": train_manifest_hash,
        "split_groups": cfg["split_groups"],
        "sample_counts": cfg["sample_counts"],
        "crop_size": CROP_SIZE,
        "train_sampling": "one seeded random 512x512 crop per training image per epoch; D4 augmentation",
        "validation_test": "fixed 512x512 windows; aggregate confusion matrix over all pixels",
        "rgb_normalization": cfg["rgb_normalization"],
        "nir_normalization": cfg["nir_normalization"],
        "raw_label_to_train_id": cfg["raw_label_to_train_id"],
        "class_weight_mode": args.class_weight_mode,
        "loss": {"cross_entropy": 1.0, "lovasz_softmax": args.lovasz_weight},
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "base_lr": args.base_lr,
        "new_lr": args.new_lr,
        "weight_decay": args.weight_decay,
        "amp": amp,
        "device": str(device),
    }
    save_json(output_dir / "protocol.json", protocol_record)

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = ce_sum = lovasz_sum = 0.0
        epoch_started = time.time()
        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            nir = batch["nir"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            group_start = (batch_idx // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(train_loader) - group_start)
            with torch.cuda.amp.autocast(enabled=amp):
                details = model(rgb, nir, return_details=True)
                ce = F.cross_entropy(details["logits"].float(), labels, weight=class_weights, ignore_index=IGNORE_INDEX)
                lovasz = lovasz_softmax(details["raw_logits"], labels)
                loss = ce + args.lovasz_weight * lovasz
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Loss 非有限值：epoch={epoch + 1}, batch={batch_idx + 1}, loss={loss.item()}")
            scaler.scale(loss / group_size).backward()
            loss_sum += float(loss.detach().item())
            ce_sum += float(ce.detach().item())
            lovasz_sum += float(lovasz.detach().item())
            group_end = ((batch_idx + 1) % args.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if group_end:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(train_loader):
                elapsed = time.time() - epoch_started
                batches_done = batch_idx + 1
                batches_total = len(train_loader)
                seconds_per_batch = elapsed / max(1, batches_done)
                eta_epoch = seconds_per_batch * (batches_total - batches_done)

                # Estimate the remaining run using the current epoch's observed speed.
                # Once this process has completed epochs, use their mean duration for future epochs.
                completed_this_process = epoch - start_epoch
                if completed_this_process > 0:
                    prior_epoch_seconds = (time.time() - started - elapsed) / completed_this_process
                    future_epoch_seconds = max(1.0, prior_epoch_seconds)
                else:
                    future_epoch_seconds = max(1.0, elapsed / max(1, batches_done) * batches_total)
                eta_total = eta_epoch + future_epoch_seconds * max(0, args.epochs - epoch - 1)
                finish_at = datetime.now() + timedelta(seconds=eta_total)

                print(
                    f"[TRAIN] epoch={epoch + 1:03d}/{args.epochs:03d} "
                    f"batch={batches_done:03d}/{batches_total:03d} "
                    f"loss={loss.item():.5f} ce={ce.item():.5f} lovasz={lovasz.item():.5f} "
                    f"step={global_step} grad_norm={float(grad_norm):.3f} "
                    f"elapsed={format_duration(elapsed)} "
                    f"ETA_epoch={format_duration(eta_epoch)} "
                    f"ETA_total={format_duration(eta_total)} "
                    f"finish~{finish_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    flush=True,
                )
            del details, rgb, nir, labels, ce, lovasz, loss

        train_summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss_mean": loss_sum / max(1, len(train_loader)),
            "train_ce_mean": ce_sum / max(1, len(train_loader)),
            "train_lovasz_mean": lovasz_sum / max(1, len(train_loader)),
            "epoch_seconds": time.time() - epoch_started,
        }
        val_metrics = evaluate(model, val_loader, device, amp, "VAL", output_dir)
        train_summary.update({k: val_metrics[k] for k in ("OA", "mIoU", "Kappa", "CE_loss_mean_patch")})
        train_summary["event"] = "epoch_done"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(train_summary, ensure_ascii=False) + "\n")
        print(
            f"[VAL] epoch={epoch + 1} OA={val_metrics['OA']:.4f} "
            f"mIoU={val_metrics['mIoU']:.4f} Kappa={val_metrics['Kappa']:.4f} "
            f"CE={val_metrics['CE_loss_mean_patch']:.5f}", flush=True
        )

        improved = val_metrics["mIoU"] > best_miou
        if improved:
            best_miou = float(val_metrics["mIoU"])
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        ckpt_payload = {
            "format_version": 1,
            "model_id": "B2_RGBNIR_FIXED_RESIDUAL_MERIS7",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "model_meta": model_meta,
            "protocol": protocol_record,
            "train_manifest_sha256": train_manifest_hash,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_torch_save(ckpt_payload, checkpoints / "latest.pt")
        if improved:
            atomic_torch_save(ckpt_payload, checkpoints / "best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            atomic_torch_save(ckpt_payload, checkpoints / f"epoch_{epoch + 1:03d}.pt")
        elapsed_run = time.time() - started
        completed_run_epochs = epoch - start_epoch + 1
        mean_epoch_seconds = elapsed_run / max(1, completed_run_epochs)
        eta_remaining_seconds = mean_epoch_seconds * max(0, args.epochs - (epoch + 1))
        eta_finish_at = datetime.now() + timedelta(seconds=eta_remaining_seconds)
        save_json(output_dir / "progress.json", {
            "epoch_completed": epoch + 1,
            "epochs_total": args.epochs,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_val_mIoU": best_miou,
            "latest_checkpoint": str(checkpoints / "latest.pt"),
            "elapsed_seconds": elapsed_run,
            "eta_remaining_seconds": eta_remaining_seconds,
            "eta_remaining_formatted": format_duration(eta_remaining_seconds),
            "estimated_finish_time": eta_finish_at.isoformat(timespec="seconds"),
        })
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print(f"[EARLY STOP] {stale_epochs} epochs without val mIoU improvement; best epoch={best_epoch}")
            break

    best_path = checkpoints / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"没有生成 best checkpoint：{best_path}")
    best_ckpt = torch_load_checkpoint(best_path, "cpu")
    model.load_state_dict(best_ckpt["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device, amp, "TEST", output_dir)
    final_path = checkpoints / "final.pt"
    atomic_torch_save({
        "format_version": 1,
        "model_id": "B2_RGBNIR_FIXED_RESIDUAL_MERIS7",
        "model": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_mIoU": best_miou,
        "test_metrics": test_metrics,
        "model_meta": model_meta,
        "protocol": protocol_record,
        "train_manifest_sha256": train_manifest_hash,
    }, final_path)
    print("\n[TRAINING COMPLETE]")
    print(f"Best epoch: {best_epoch}; best validation mIoU={best_miou:.4f}")
    print(f"TEST OA={test_metrics['OA']:.4f}; mIoU={test_metrics['mIoU']:.4f}; Kappa={test_metrics['Kappa']:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Final model: {final_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。最新检查点（若已开始训练）保留在 outputs 目录。", file=sys.stderr)
        raise SystemExit(130)
