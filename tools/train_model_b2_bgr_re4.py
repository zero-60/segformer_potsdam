#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train SegFormer-B2 with Blue+Green+Red+RedEdge bands on TLSNMSP.

Uses the first four bands of the 5-band multispectral TIFF (assumed order:
Blue, Green, Red, RedEdge, NIR). The standalone RGB image is not used as an
input in this experiment. Dataset split, labels, loss, optimizer, metrics,
checkpointing and crop strategy follow the existing MERIS RGB+NIR baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer_b2_bgr_re4 import (  # noqa: E402
    BAND_NAMES,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    build_model_b2_bgr_re4,
)

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PRETRAINED_CHECKPOINT = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training" / "meris_b2_bgr_re4_seed20261029"
CROP_SIZE = 512
BAND_SAMPLE_STRIDE = 8
RAW_LABEL_TO_TRAIN_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


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
        raise FileNotFoundError(f"缺少数据配置：{path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("protocol_version") != "TLSNMSP_RGB_NIR_7CLASS_v1":
        raise RuntimeError(f"不支持的数据协议：{cfg.get('protocol_version')}")
    if len(cfg.get("class_names_train_id_order", [])) != NUM_CLASSES:
        raise RuntimeError("数据配置类别数量与模型不一致")
    return cfg


def resolve_manifest_path(data_root: Path, rel: str) -> Path:
    return (data_root / Path(rel.replace("/", os.sep))).resolve()


def load_ms_chw(path: Path) -> np.ndarray:
    """Return five bands as CHW; expected band order B, G, R, RedEdge, NIR."""
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
    return np.ascontiguousarray(chw)


def load_label(path: Path) -> np.ndarray:
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"mask_converted 应为单通道类别 ID 图：{path}, shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}: {path}")
    ids = set(np.unique(arr).astype(int).tolist())
    if ids - set(RAW_LABEL_TO_TRAIN_ID):
        raise RuntimeError(f"标签出现 1..7 以外的值：{sorted(ids)}; {path}")
    return arr.astype(np.uint8, copy=False)


@lru_cache(maxsize=8)
def load_sample_arrays(ms_path_text: str, label_path_text: str):
    ms = load_ms_chw(Path(ms_path_text))
    bands = np.ascontiguousarray(np.moveaxis(ms[:4], 0, -1))  # H,W,4: B,G,R,RedEdge
    label = load_label(Path(label_path_text))
    if bands.shape[:2] != label.shape:
        raise RuntimeError(f"多光谱/标签尺寸不一致：bands={bands.shape}, label={label.shape}; {ms_path_text}")
    return bands, label


def compute_band_stats(train_rows, data_root: Path, stride: int) -> dict[str, Any]:
    """Train-only streaming mean/std for the four selected bands."""
    sums = np.zeros(4, dtype=np.float64)
    sums_sq = np.zeros(4, dtype=np.float64)
    count = 0
    for idx, row in enumerate(train_rows, 1):
        path = resolve_manifest_path(data_root, row["multispectral_path"])
        ms = load_ms_chw(path)
        sampled = ms[:4, ::stride, ::stride].astype(np.float64, copy=False)
        sums += sampled.sum(axis=(1, 2), dtype=np.float64)
        sums_sq += np.square(sampled).sum(axis=(1, 2), dtype=np.float64)
        count += int(sampled.shape[1] * sampled.shape[2])
        if idx % 25 == 0 or idx == len(train_rows):
            print(f"[BAND-STATS] {idx}/{len(train_rows)} train images")
    if count < 1000:
        raise RuntimeError(f"四波段训练统计样本不足：{count}")
    means = sums / count
    variances = np.maximum(sums_sq / count - means * means, 0.0)
    stds = np.sqrt(variances)
    stds = np.maximum(stds, 1e-6)
    return {
        "method": "train-only sampled z-score",
        "sample_stride": stride,
        "sampled_pixel_count": count,
        "band_order": BAND_NAMES,
        "mean_dn": [float(x) for x in means],
        "std_dn": [float(x) for x in stds],
    }


def load_or_compute_band_stats(output_dir: Path, train_rows, data_root: Path, stride: int):
    path = output_dir / "band_normalization.json"
    if path.is_file():
        stats = json.loads(path.read_text(encoding="utf-8"))
        if stats.get("band_order") != BAND_NAMES:
            raise RuntimeError(f"现有 band_normalization.json 波段顺序不匹配：{stats.get('band_order')}")
        print(f"[BAND-STATS] reuse {path}")
        return stats
    stats = compute_band_stats(train_rows, data_root, stride)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[BAND-STATS]", json.dumps(stats, ensure_ascii=False))
    return stats


def label_to_train_ids(label_raw: np.ndarray) -> np.ndarray:
    mapped = label_raw.astype(np.int16) - 1
    if mapped.size and (mapped.min() < 0 or mapped.max() >= NUM_CLASSES):
        raise RuntimeError(f"标签映射越界：min={mapped.min()}, max={mapped.max()}")
    return mapped.astype(np.uint8)


def pack_inputs(bands: np.ndarray, label_raw: np.ndarray, stats: dict[str, Any]):
    values = bands.astype(np.float32, copy=False)
    means = np.asarray(stats["mean_dn"], dtype=np.float32).reshape(1, 1, 4)
    stds = np.maximum(np.asarray(stats["std_dn"], dtype=np.float32), 1e-6).reshape(1, 1, 4)
    normalized = (values - means) / stds
    image_tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()
    label_tensor = torch.from_numpy(np.ascontiguousarray(label_to_train_ids(label_raw))).long()
    return image_tensor, label_tensor


class MerisFourBandRandomCropDataset(Dataset):
    """One deterministic-random 512x512 crop per labeled image per epoch."""
    def __init__(self, rows, data_root: Path, stats, *, seed: int, augment: bool):
        self.rows, self.data_root, self.stats = rows, data_root, stats
        self.seed, self.augment, self.epoch = int(seed), bool(augment), 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ms_path = resolve_manifest_path(self.data_root, row["multispectral_path"])
        label_path = resolve_manifest_path(self.data_root, row["label_path"])
        bands, label_raw = load_sample_arrays(str(ms_path), str(label_path))
        h, w = label_raw.shape
        if h < CROP_SIZE or w < CROP_SIZE:
            raise RuntimeError(f"样本小于 {CROP_SIZE}x{CROP_SIZE}: {row['sample_id']} {w}x{h}")
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx * 9_176)
        y0 = int(rng.integers(0, h - CROP_SIZE + 1))
        x0 = int(rng.integers(0, w - CROP_SIZE + 1))
        bands = bands[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        label_raw = label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        if self.augment:
            k = int(rng.integers(0, 4))
            if k:
                bands, label_raw = np.rot90(bands, k), np.rot90(label_raw, k)
            if bool(rng.integers(0, 2)):
                bands, label_raw = bands[:, ::-1], label_raw[:, ::-1]
            if bool(rng.integers(0, 2)):
                bands, label_raw = bands[::-1, :], label_raw[::-1, :]
        image_t, label_t = pack_inputs(np.ascontiguousarray(bands), np.ascontiguousarray(label_raw), self.stats)
        return {"image": image_t, "label": label_t, "sample_id": row["sample_id"]}


def window_starts(length: int, crop: int = CROP_SIZE) -> list[int]:
    if length < crop:
        raise RuntimeError(f"影像边长 {length} 小于窗口 {crop}，不自动缩放标签")
    starts = list(range(0, length - crop + 1, crop))
    last = length - crop
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


class MerisFourBandSlidingPatchDataset(Dataset):
    """Fixed 512x512 windows; last window is aligned to image edge."""
    def __init__(self, rows, data_root: Path, stats):
        self.rows, self.data_root, self.stats = rows, data_root, stats
        self.items = []
        for row_idx, row in enumerate(rows):
            ms_path = resolve_manifest_path(data_root, row["multispectral_path"])
            label_path = resolve_manifest_path(data_root, row["label_path"])
            bands, label = load_sample_arrays(str(ms_path), str(label_path))
            for y in window_starts(label.shape[0]):
                for x in window_starts(label.shape[1]):
                    self.items.append((row_idx, y, x))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_idx, y0, x0 = self.items[idx]
        row = self.rows[row_idx]
        ms_path = resolve_manifest_path(self.data_root, row["multispectral_path"])
        label_path = resolve_manifest_path(self.data_root, row["label_path"])
        bands, label_raw = load_sample_arrays(str(ms_path), str(label_path))
        bands = bands[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        label_raw = label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        image_t, label_t = pack_inputs(bands, label_raw, self.stats)
        return {"image": image_t, "label": label_t, "sample_id": row["sample_id"], "x": x0, "y": y0}


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
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else raw_logits.sum() * 0.0


def make_class_weights(protocol_dir: Path, mode: str, device: torch.device):
    if mode == "none":
        return None
    rows = read_csv_rows(protocol_dir / "class_pixel_counts_by_split.csv")
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for row in rows:
        if row["split"] == "train":
            counts[int(row["train_label_id"])] = float(row["pixel_count"])
    if np.any(counts <= 0):
        raise RuntimeError(f"训练集中至少一个类别像素数为 0：{counts.tolist()}")
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
        is_new = name.startswith("decode_head.classifier.")
        no_decay = param.ndim == 1 or name.endswith(".bias")
        key = ("new" if is_new else "base") + ("_no_decay" if no_decay else "_decay")
        groups[key].append(param)
    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        param_groups.append({
            "params": params,
            "lr": new_lr if key.startswith("new") else base_lr,
            "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay,
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
    diag, row, col = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
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
            "class_id": i, "class_name": name, "gt_pixels": int(row[i]),
            "pred_pixels": int(col[i]), "intersection": int(diag[i]),
            "iou": None if not np.isfinite(ious[i]) else float(ious[i]),
            "precision": precision, "recall": recall,
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


def evaluate(model, loader, device, amp: bool, split_name: str, output_dir: Path | None = None):
    model.eval()
    confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum, n_batches = 0.0, 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                raw_logits = model(pixel_values=image).logits
                logits = F.interpolate(raw_logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
            loss_sum += float(loss.item())
            n_batches += 1
            confusion_update(confmat, logits, labels)
    cm_np = confmat.numpy()
    metrics = metrics_from_confusion(cm_np)
    metrics["split"] = split_name
    metrics["CE_loss_mean_patch"] = loss_sum / max(1, n_batches)
    metrics["patch_count"] = n_batches
    metrics["evaluation_seconds"] = time.time() - started
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / f"{split_name.lower()}_metrics.json", metrics)
        with (output_dir / f"{split_name.lower()}_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics["per_class"][0].keys()))
            writer.writeheader()
            writer.writerows(metrics["per_class"])
        np.savetxt(output_dir / f"{split_name.lower()}_confusion_matrix.csv", cm_np, fmt="%d", delimiter=",")
    return metrics


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_args():
    ap = argparse.ArgumentParser(description="Train SegFormer-B2 on Blue+Green+Red+RedEdge four-band TLSNMSP data")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CHECKPOINT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--base-lr", type=float, default=3e-5)
    ap.add_argument("--new-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lovasz-weight", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--class-weight-mode", choices=["none", "sqrt_inverse"], default="none")
    ap.add_argument("--resume", default="none", help="none、auto 或一个 checkpoint 路径")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    return ap.parse_args()


def main():
    args = get_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise SystemExit("epochs、batch-size、grad-accum-steps 必须为正数")
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lovasz_weight < 0:
        raise SystemExit("num-workers/warmup-steps/lovasz-weight 参数不合法")

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pretrained_checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据目录不存在：{data_root}")
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
    print(f"[INPUT BANDS] {BAND_NAMES} | source=first 4 bands of multispectral TIFF")
    print(f"[SPLIT] train={len(train_rows)} images, val={len(val_rows)} images, test={len(test_rows)} images")
    print(f"[LABEL] raw IDs 1..7 -> train IDs 0..6; class names={CLASS_NAMES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    band_stats = load_or_compute_band_stats(output_dir, train_rows, data_root, BAND_SAMPLE_STRIDE)
    train_ds = MerisFourBandRandomCropDataset(train_rows, data_root, band_stats, seed=args.seed, augment=True)
    val_ds = MerisFourBandSlidingPatchDataset(val_rows, data_root, band_stats)
    test_ds = MerisFourBandSlidingPatchDataset(test_rows, data_root, band_stats)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_common)
    eval_kwargs = dict(batch_size=1, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, **eval_kwargs)
    test_loader = DataLoader(test_ds, **eval_kwargs)
    print(f"[PATCHES] train samples/epoch={len(train_ds)}, val patches={len(val_ds)}, test patches={len(test_ds)}")

    model, model_meta = build_model_b2_bgr_re4(pretrained_checkpoint)
    model.to(device)
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("[GRAD-CHECKPOINTING] enabled")
        except Exception as exc:
            print(f"[WARN] gradient checkpointing could not be enabled: {exc}")

    if args.smoke_test:
        model.train()
        sample = train_ds[0]
        image = sample["image"].unsqueeze(0).to(device)
        labels = sample["label"].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            raw_logits = model(pixel_values=image).logits
            logits = F.interpolate(raw_logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
            lovasz = lovasz_softmax(raw_logits, labels)
            loss = ce + args.lovasz_weight * lovasz
        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke-test loss 非有限值：{loss.item()}")
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        print("[SMOKE PASS]", f"sample={sample['sample_id']}", f"image={tuple(image.shape)}",
              f"labels={tuple(labels.shape)}", f"raw_logits={tuple(raw_logits.shape)}",
              f"loss={loss.item():.6f}", f"parameters_with_grad={grad_count}")
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
        raise RuntimeError(f"输出目录已有检查点或日志：{output_dir}。续训请使用 --resume auto，或更换输出目录。")
    checkpoints.mkdir(parents=True, exist_ok=True)

    start_epoch, global_step = 0, 0
    best_miou, best_epoch, stale_epochs = -1.0, 0, 0
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
            raise RuntimeError("resume checkpoint 的 train.csv 哈希与当前协议不同，拒绝续训")
        if ckpt.get("model_id") != "B2_BGR_REDEDGE_4BAND_MERIS7":
            raise RuntimeError("检查点不是本四波段实验的模型，拒绝加载")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_miou = float(ckpt.get("best_miou", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        try:
            if "rng_python" in ckpt: random.setstate(ckpt["rng_python"])
            if "rng_numpy" in ckpt: np.random.set_state(ckpt["rng_numpy"])
            if "rng_torch" in ckpt: torch.set_rng_state(ckpt["rng_torch"])
            if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception as exc:
            print(f"[WARN] 恢复随机状态失败：{exc}")
        print(f"[RESUME] {resume_path} start_epoch={start_epoch + 1}, global_step={global_step}, best_epoch={best_epoch}, best_val_mIoU={best_miou:.4f}")

    protocol_record = {
        "protocol_version": "TLSNMSP_BGR_REDEDGE_7CLASS_v1",
        "model": model_meta,
        "data_root": str(data_root),
        "protocol_dir": str(protocol_dir),
        "train_manifest_sha256": train_manifest_hash,
        "split_groups": cfg["split_groups"],
        "sample_counts": cfg["sample_counts"],
        "crop_size": CROP_SIZE,
        "input_bands": BAND_NAMES,
        "input_source": "first four bands of multispectral TIFF; expected B,G,R,RedEdge,NIR order",
        "band_normalization": band_stats,
        "train_sampling": "one seeded random 512x512 crop per training image per epoch; D4 augmentation",
        "validation_test": "fixed 512x512 windows; aggregate confusion matrix over all pixels",
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

    run_started = time.time()
    completed_epoch_seconds = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("event") == "epoch_done" and item.get("epoch_seconds"):
                    completed_epoch_seconds.append(float(item["epoch_seconds"]))
            except Exception:
                pass
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = ce_sum = lovasz_sum = 0.0
        epoch_started = time.time()
        epoch_steps = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            group_start = (batch_idx // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(train_loader) - group_start)
            with torch.cuda.amp.autocast(enabled=amp):
                raw_logits = model(pixel_values=image).logits
                logits = F.interpolate(raw_logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=IGNORE_INDEX)
                lovasz = lovasz_softmax(raw_logits, labels)
                loss = ce + args.lovasz_weight * lovasz
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Loss 非有限值：epoch={epoch + 1}, batch={batch_idx + 1}, loss={loss.item()}")
            scaler.scale(loss / group_size).backward()
            loss_sum += float(loss.detach().item())
            ce_sum += float(ce.detach().item())
            lovasz_sum += float(lovasz.detach().item())
            grad_norm = torch.tensor(float("nan"))
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
                sec_per_batch = elapsed / max(1, batch_idx + 1)
                epoch_eta = sec_per_batch * (epoch_steps - batch_idx - 1)
                prior = completed_epoch_seconds
                avg_epoch = (sum(prior) / len(prior)) if prior else sec_per_batch * epoch_steps
                total_eta = epoch_eta + max(0, args.epochs - epoch - 1) * avg_epoch
                finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_eta))
                print(
                    f"[TRAIN] epoch={epoch + 1:03d}/{args.epochs:03d} "
                    f"batch={batch_idx + 1:03d}/{epoch_steps:03d} "
                    f"loss={loss.item():.5f} ce={ce.item():.5f} lovasz={lovasz.item():.5f} "
                    f"step={global_step} grad_norm={float(grad_norm):.3f} elapsed={elapsed:.1f}s "
                    f"epoch_train_eta={format_duration(epoch_eta)} total_eta={format_duration(total_eta)} finish~{finish}",
                    flush=True,
                )
            del raw_logits, logits, image, labels, ce, lovasz, loss

        train_seconds = time.time() - epoch_started
        train_summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss_mean": loss_sum / max(1, len(train_loader)),
            "train_ce_mean": ce_sum / max(1, len(train_loader)),
            "train_lovasz_mean": lovasz_sum / max(1, len(train_loader)),
            "epoch_train_seconds": train_seconds,
        }
        val_metrics = evaluate(model, val_loader, device, amp, "VAL", output_dir)
        train_summary.update({k: val_metrics[k] for k in ("OA", "mIoU", "Kappa", "CE_loss_mean_patch")})
        train_summary["epoch_seconds"] = train_seconds + val_metrics["evaluation_seconds"]
        train_summary["event"] = "epoch_done"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(train_summary, ensure_ascii=False) + "\n")
        print(
            f"[VAL] epoch={epoch + 1} OA={val_metrics['OA']:.4f} "
            f"mIoU={val_metrics['mIoU']:.4f} Kappa={val_metrics['Kappa']:.4f} "
            f"CE={val_metrics['CE_loss_mean_patch']:.5f} val_time={format_duration(val_metrics['evaluation_seconds'])}",
            flush=True,
        )

        improved = val_metrics["mIoU"] > best_miou
        if improved:
            best_miou, best_epoch, stale_epochs = float(val_metrics["mIoU"]), epoch + 1, 0
        else:
            stale_epochs += 1
        ckpt_payload = {
            "format_version": 1,
            "model_id": "B2_BGR_REDEDGE_4BAND_MERIS7",
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
        completed_epoch_seconds.append(train_summary["epoch_seconds"])
        remaining_epochs = max(0, args.epochs - (epoch + 1))
        avg_epoch = sum(completed_epoch_seconds) / len(completed_epoch_seconds)
        total_eta = remaining_epochs * avg_epoch
        finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_eta))
        save_json(output_dir / "progress.json", {
            "epoch_completed": epoch + 1,
            "epochs_total": args.epochs,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_val_mIoU": best_miou,
            "latest_checkpoint": str(checkpoints / "latest.pt"),
            "elapsed_seconds_this_run": time.time() - run_started,
            "estimated_remaining_seconds": total_eta,
            "estimated_finish_local": finish,
        })
        print(f"[EPOCH ETA] completed={epoch + 1}/{args.epochs} remaining={format_duration(total_eta)} finish~{finish}", flush=True)
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
        "model_id": "B2_BGR_REDEDGE_4BAND_MERIS7",
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
