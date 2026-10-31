#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch-predict unlabeled TLSNMSP images using the trained SegFormer-B2 RGB+NIR model.

- Pairs RGB images with five-band multispectral TIFFs by filename stem.
- Excludes any sample that has an original mask OR a converted mask.
- Uses the 5th multispectral band as NIR, and the same normalization as training.
- Loads a trained checkpoint strictly (no training, no checkpoint modification).
- Saves raw predicted IDs (1..7), color masks, RGB overlays, comparison images,
  contact sheets, and predicted class/vegetation coverage statistics.

Place this file at: <project_root>\\tools\\predict_unlabeled_meris_rgbnir.py
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
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep CLI help usable even before the script is copied into the project.
# The project model implementation is imported in main() after argument parsing.
NUM_CLASSES = 7
CLASS_NAMES = [
    "building", "impervious_surface", "water", "barren", "tree",
    "low_vegetation", "background_other",
]
build_model_b2_rgbnir_meris = None

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "training" / "meris_b2_rgbnir_fixed_g1_seed20261028"
    / "checkpoints" / "best.pt"
)
DEFAULT_PRETRAINED_CHECKPOINT = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "unlabeled_predictions" / "meris_b2_rgbnir_fixed_g1_seed20261028"
)

RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MS_EXTENSIONS = {".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff"}
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

# The palette is copied from prepare_meris_tlsnmsp.py. Model IDs are 0..6;
# raw prediction IDs written to disk are 1..7.
CLASS_NAMES_FALLBACK = [
    "building", "impervious_surface", "water", "barren", "tree",
    "low_vegetation", "background_other",
]
RAW_ID_TO_RGB = {
    1: (178, 74, 64),       # #B24A40 building
    2: (128, 128, 128),     # #808080 impervious surface
    3: (135, 206, 235),     # #87CEEB water
    4: (210, 180, 100),     # #D2B464 barren
    5: (34, 139, 34),       # #228B22 tree
    6: (144, 238, 144),     # #90EE90 low vegetation
    7: (255, 255, 0),       # #FFFF00 background / other
}


def scan_index(folder: Path, extensions: set[str], *, required: bool = True) -> dict[str, Path]:
    if not folder.is_dir():
        if required:
            raise FileNotFoundError(f"缺少目录：{folder}")
        return {}
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem.casefold()
        if key in index:
            duplicates.setdefault(key, [str(index[key])]).append(str(path))
        else:
            index[key] = path
    if duplicates:
        examples = list(duplicates.items())[:5]
        raise RuntimeError(f"目录中存在重复文件名 stem，无法安全配对：{folder}\n示例：{examples}")
    return index


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"RGB 读取失败：{path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"不支持的 RGB 形状 {arr.shape}：{path}")
    if arr.dtype != np.uint8:
        raise RuntimeError(f"RGB 必须为 uint8，实际为 {arr.dtype}：{path}")
    return np.ascontiguousarray(arr)


def load_nir(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维数组，实际为 {arr.shape}：{path}")
    if arr.shape[0] == 5:
        nir = arr[4]
    elif arr.shape[-1] == 5:
        nir = arr[:, :, 4]
    else:
        raise RuntimeError(f"无法识别五波段 TIFF 布局（预期第 5 波段为 NIR），实际为 {arr.shape}：{path}")
    if nir.dtype != np.uint16:
        raise RuntimeError(f"NIR 必须为 uint16，实际为 {nir.dtype}：{path}")
    return np.ascontiguousarray(nir)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}\n请先运行数据准备脚本生成数据协议。")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_model_state(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("无法识别 checkpoint：预期为字典格式。")
    model_id = payload.get("model_id")
    if model_id and model_id != "B2_RGBNIR_FIXED_RESIDUAL_MERIS7":
        raise RuntimeError(
            f"checkpoint 的 model_id={model_id!r}，不是预期的 MERIS 七类 RGB+NIR 固定融合模型。"
        )
    if isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    elif payload and all(torch.is_tensor(v) for v in payload.values()):
        state = payload
    else:
        raise RuntimeError("checkpoint 中没有可识别的 model/state_dict 参数字典。")
    meta = payload.get("model_meta", {}) if isinstance(payload.get("model_meta", {}), dict) else {}
    protocol = payload.get("protocol", {}) if isinstance(payload.get("protocol", {}), dict) else {}
    return state, {"model_id": model_id, "model_meta": meta, "protocol": protocol}


def get_fixed_strength(meta: dict[str, Any], explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    candidates: list[dict[str, Any]] = []
    model_meta = meta.get("model_meta", {})
    protocol = meta.get("protocol", {})
    if isinstance(model_meta, dict):
        candidates.append(model_meta)
    if isinstance(protocol, dict):
        if isinstance(protocol.get("model"), dict):
            candidates.append(protocol["model"])
        candidates.append(protocol)
    for obj in candidates:
        for key in ("fixed_strength", "fixed_nir_strength", "nir_strength", "g"):
            if key in obj:
                try:
                    value = float(obj[key])
                    if 0.0 <= value <= 1.0:
                        return value
                except (TypeError, ValueError):
                    pass
    # The current experiment was trained with fixed_nir_strength=1.0.
    return 1.0


def window_starts(length: int, crop: int) -> list[int]:
    """Match the training/evaluation window policy: fixed crop, final window edge-aligned."""
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, crop))
    last = length - crop
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def pad_to_crop(patch: np.ndarray, crop: int) -> np.ndarray:
    h, w = patch.shape[:2]
    pad_h, pad_w = max(0, crop - h), max(0, crop - w)
    if not pad_h and not pad_w:
        return patch
    # Reflection padding is visually and spectrally less disruptive than zero padding.
    mode = cv2.BORDER_REFLECT_101 if h > 1 and w > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, mode)


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("logits", "out"):
            if torch.is_tensor(output.get(key)):
                return output[key]
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"模型返回了无法识别的输出类型：{type(output).__name__}")


def predict_one_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    nir: np.ndarray,
    nir_mean: float,
    nir_std: float,
    device: torch.device,
    amp: bool,
    crop_size: int,
) -> tuple[np.ndarray, int]:
    h, w = nir.shape
    if rgb.shape[:2] != (h, w):
        raise RuntimeError(f"RGB/NIR 尺寸不一致：RGB={rgb.shape[:2]}，NIR={nir.shape}")
    y_starts = window_starts(h, crop_size)
    x_starts = window_starts(w, crop_size)
    logits_sum = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count = np.zeros((h, w, 1), dtype=np.float32)
    tile_count = 0

    for y0 in y_starts:
        for x0 in x_starts:
            y1, x1 = min(y0 + crop_size, h), min(x0 + crop_size, w)
            rgb_patch = pad_to_crop(rgb[y0:y1, x0:x1], crop_size)
            nir_patch = pad_to_crop(nir[y0:y1, x0:x1], crop_size)

            rgb_f = rgb_patch.astype(np.float32) / 255.0
            rgb_f = (rgb_f - RGB_MEAN[None, None, :]) / RGB_STD[None, None, :]
            nir_f = (nir_patch.astype(np.float32) - nir_mean) / max(nir_std, 1e-6)
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_f.transpose(2, 0, 1))).unsqueeze(0).to(device)
            nir_t = torch.from_numpy(np.ascontiguousarray(nir_f[None, :, :])).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(rgb_t, nir_t)
                    logits = extract_logits(output)
                    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != NUM_CLASSES:
                        raise RuntimeError(f"模型输出 shape 异常：{tuple(logits.shape)}，预期 (1,{NUM_CLASSES},H,W)")
                    if logits.shape[-2:] != (crop_size, crop_size):
                        logits = F.interpolate(logits.float(), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            tile_logits = logits[0].float().permute(1, 2, 0).cpu().numpy()
            valid_h, valid_w = y1 - y0, x1 - x0
            logits_sum[y0:y1, x0:x1] += tile_logits[:valid_h, :valid_w]
            count[y0:y1, x0:x1] += 1.0
            tile_count += 1
            del rgb_t, nir_t, logits, output

    logits_avg = logits_sum / np.maximum(count, 1.0)
    pred_train_ids = np.argmax(logits_avg, axis=2).astype(np.uint8)
    return pred_train_ids, tile_count


def colorize_train_ids(pred_train_ids: np.ndarray) -> np.ndarray:
    palette = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for raw_id, rgb in RAW_ID_TO_RGB.items():
        train_id = raw_id - 1
        if 0 <= train_id < NUM_CLASSES:
            palette[train_id] = rgb
    return palette[pred_train_ids]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resize_keep_aspect(image: np.ndarray, max_w: int, max_h: int, background=(35, 35, 35)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[:] = background
    x = (max_w - nw) // 2
    y = (max_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def save_comparison(path: Path, rgb: np.ndarray, color_mask: np.ndarray, overlay: np.ndarray, max_panel_width: int = 768) -> None:
    panels = []
    for label, image in (("RGB original", rgb), ("Predicted classes", color_mask), ("Overlay", overlay)):
        h, w = image.shape[:2]
        scale = min(1.0, max_panel_width / max(1, w))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        header = np.zeros((32, new_w, 3), dtype=np.uint8)
        header[:] = (245, 245, 245)
        cv2.putText(header, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, resized]))
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            padded.append(np.vstack([panel, pad]))
        else:
            padded.append(panel)
    comparison_rgb = np.hstack(padded)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(comparison_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"比较图写入失败：{path}")


def save_contact_sheets(records: list[dict[str, Any]], out_dir: Path, cols: int = 4, rows: int = 3) -> int:
    if not records:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows
    cell_w, header_h, image_h = 420, 26, 156
    cell_h = header_h + image_h
    total_sheets = math.ceil(len(records) / per_sheet)
    for sheet_idx in range(total_sheets):
        group = records[sheet_idx * per_sheet:(sheet_idx + 1) * per_sheet]
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        sheet[:] = (38, 38, 38)
        for j, record in enumerate(group):
            r, c = divmod(j, cols)
            x0, y0 = c * cell_w, r * cell_h
            title = record["sample_id"]
            cv2.putText(sheet, title[:48], (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            rgb_bgr = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
            overlay_bgr = cv2.imread(str(record["overlay_path"]), cv2.IMREAD_COLOR)
            if rgb_bgr is None or overlay_bgr is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            half_w = cell_w // 2
            left = resize_keep_aspect(rgb, half_w, image_h, background=(55, 55, 55))
            right = resize_keep_aspect(overlay_rgb, half_w, image_h, background=(55, 55, 55))
            cv2.putText(left, "RGB", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right, "Overlay", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tile = np.hstack([left, right])
            sheet[y0 + header_h:y0 + header_h + image_h, x0:x0 + cell_w] = tile
        out_path = out_dir / f"contact_sheet_{sheet_idx + 1:03d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return total_sheets


def main() -> int:
    ap = argparse.ArgumentParser(description="使用已训练的 MERIS RGB+NIR 模型分割无标签 TLSNMSP 影像")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CHECKPOINT,
                    help="用于构建同一模型结构的本地 B2 预训练目录；不联网下载")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--fixed-nir-strength", type=float, default=None,
                    help="默认从 checkpoint 元数据读取；若读取不到，使用当前实验的 g=1.0")
    ap.add_argument("--tile-size", type=int, default=512, help="推理切块大小；应与训练时的 512 一致")
    ap.add_argument("--max-images", type=int, default=0,
                    help="只处理前 N 张用于试运行；0 表示处理全部无标签影像")
    ap.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度")
    ap.add_argument("--skip-contact-sheets", action="store_true", help="跳过总览拼图生成")
    args = ap.parse_args()

    global NUM_CLASSES, CLASS_NAMES, build_model_b2_rgbnir_meris
    try:
        from models.segformer_b2_rgbnir_meris import (
            CLASS_NAMES as MODEL_CLASS_NAMES,
            NUM_CLASSES as MODEL_NUM_CLASSES,
            build_model_b2_rgbnir_meris as MODEL_BUILDER,
        )
        CLASS_NAMES = list(MODEL_CLASS_NAMES)
        NUM_CLASSES = int(MODEL_NUM_CLASSES)
        build_model_b2_rgbnir_meris = MODEL_BUILDER
    except Exception as exc:
        raise SystemExit(
            "无法导入 models.segformer_b2_rgbnir_meris。请将本脚本放到项目的 tools 目录，"
            "并确认项目内存在 models/segformer_b2_rgbnir_meris.py。\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        )

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    pretrained_dir = args.pretrained_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.tile_size <= 0 or args.max_images < 0:
        raise SystemExit("tile-size 必须为正数，max-images 必须 >= 0")
    if args.fixed_nir_strength is not None and not 0.0 <= args.fixed_nir_strength <= 1.0:
        raise SystemExit("fixed-nir-strength 必须在 [0,1] 范围内")
    if not data_root.is_dir():
        raise SystemExit(f"数据根目录不存在：{data_root}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"训练 checkpoint 不存在：{checkpoint_path}\n请检查 --checkpoint 路径。")
    if not pretrained_dir.is_dir() or not (pretrained_dir / "config.json").is_file():
        raise SystemExit(f"本地预训练模型目录或 config.json 不存在：{pretrained_dir}")
    if not any((pretrained_dir / name).is_file() for name in (
        "pytorch_model.bin", "model.safetensors",
        "pytorch_model.bin.index.json", "model.safetensors.index.json",
    )):
        raise SystemExit(f"本地预训练模型目录缺少权重文件：{pretrained_dir}")

    config = read_json(protocol_dir / "dataset_config.json")
    if config.get("protocol_version") != "TLSNMSP_RGB_NIR_7CLASS_v1":
        raise RuntimeError(f"数据协议版本不匹配：{config.get('protocol_version')}")
    nir_cfg = config.get("nir_normalization", {})
    nir_mean = float(nir_cfg["mean_dn"])
    nir_std = float(nir_cfg["std_dn"])
    if not np.isfinite(nir_mean) or not np.isfinite(nir_std) or nir_std <= 0:
        raise RuntimeError(f"NIR 归一化参数异常：mean={nir_mean}, std={nir_std}")

    indexes = {
        "RGB": scan_index(data_root / "RGB", RGB_EXTENSIONS),
        "multispectral": scan_index(data_root / "multispectral", MS_EXTENSIONS),
        "mask": scan_index(data_root / "mask", MASK_EXTENSIONS, required=False),
        "mask_converted": scan_index(data_root / "mask_converted", MASK_EXTENSIONS, required=False),
    }
    rgb_keys, ms_keys = set(indexes["RGB"]), set(indexes["multispectral"])
    paired_keys = sorted(rgb_keys & ms_keys)
    missing_ms = sorted(rgb_keys - ms_keys)
    missing_rgb = sorted(ms_keys - rgb_keys)
    labeled_keys = set(indexes["mask"]) | set(indexes["mask_converted"])
    unlabeled_keys = [key for key in paired_keys if key not in labeled_keys]
    incomplete_label_keys = sorted((set(indexes["mask"]) ^ set(indexes["mask_converted"])) & set(paired_keys))
    # Any sample with either mask type is excluded from inference, including incomplete annotation pairs.
    if args.max_images:
        unlabeled_keys = unlabeled_keys[:args.max_images]
    if not unlabeled_keys:
        raise SystemExit(
            "没有找到同时具备 RGB+多光谱 TIFF 且不含原始/转换标签的影像。\n"
            f"RGB={len(rgb_keys)}, multispectral={len(ms_keys)}, paired={len(paired_keys)}, labeled_stems={len(labeled_keys)}"
        )

    payload = torch_load_checkpoint(checkpoint_path)
    state_dict, checkpoint_meta = get_model_state(payload)
    fixed_strength = get_fixed_strength(checkpoint_meta, args.fixed_nir_strength)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前 PyTorch 无法使用 CUDA。")
    amp = bool(device.type == "cuda" and not args.no_amp)

    print(f"[DATA] {data_root}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[MODEL_ID] {checkpoint_meta.get('model_id')}")
    print(f"[DEVICE] {device} | AMP={amp} | fixed_nir_strength={fixed_strength}")
    print(f"[FILES] RGB={len(rgb_keys)} | multispectral={len(ms_keys)} | paired={len(paired_keys)}")
    print(f"[LABELS] raw mask={len(indexes['mask'])} | converted mask={len(indexes['mask_converted'])} | excluded labeled stems={len(labeled_keys)}")
    print(f"[UNLABELED] to process={len(unlabeled_keys)} | incomplete label pairs skipped={len(incomplete_label_keys)}")
    print(f"[NIR] train-only mean={nir_mean:.6f}, std={nir_std:.6f} DN")

    model, model_meta = build_model_b2_rgbnir_meris(
        PROJECT_ROOT, checkpoint=str(pretrained_dir), fixed_strength=fixed_strength
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Do not allow partial checkpoint loading. strict=False is used only to produce a clearer diagnostic.
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint 与模型结构不完全匹配，已停止推理。\n"
            f"missing keys（前 20 项）={missing[:20]}\n"
            f"unexpected keys（前 20 项）={unexpected[:20]}"
        )
    model.to(device)
    model.eval()
    print(f"[MODEL] loaded strict-equivalent checkpoint; model_meta={json.dumps(model_meta, ensure_ascii=False, default=str)}")

    dirs = {
        "raw": output_dir / "pred_raw_ids",
        "color": output_dir / "pred_color_masks",
        "overlay": output_dir / "overlays",
        "comparison": output_dir / "comparisons",
        "contact": output_dir / "contact_sheets",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    class_names = list(CLASS_NAMES) if len(CLASS_NAMES) == NUM_CLASSES else CLASS_NAMES_FALLBACK
    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    started = time.time()
    completed = 0

    for idx, key in enumerate(unlabeled_keys, 1):
        rgb_path = indexes["RGB"][key]
        ms_path = indexes["multispectral"][key]
        sample_id = rgb_path.stem
        image_started = time.time()
        try:
            rgb = load_rgb(rgb_path)
            nir = load_nir(ms_path)
            if rgb.shape[:2] != nir.shape:
                raise RuntimeError(f"RGB/NIR 尺寸不一致：RGB={rgb.shape[:2]}，NIR={nir.shape}")
            pred_train_ids, tile_count = predict_one_image(
                model, rgb, nir, nir_mean, nir_std, device, amp, args.tile_size
            )
            pred_raw_ids = (pred_train_ids.astype(np.uint16) + 1).astype(np.uint8)
            color_mask = colorize_train_ids(pred_train_ids)
            overlay = cv2.addWeighted(rgb, 0.56, color_mask, 0.44, 0)

            raw_path = dirs["raw"] / f"{sample_id}_pred_raw_ids.png"
            color_path = dirs["color"] / f"{sample_id}_pred_color.png"
            overlay_path = dirs["overlay"] / f"{sample_id}_overlay.jpg"
            comparison_path = dirs["comparison"] / f"{sample_id}_comparison.jpg"
            if not cv2.imwrite(str(raw_path), pred_raw_ids):
                raise RuntimeError(f"原始类别 ID 掩膜写入失败：{raw_path}")
            if not cv2.imwrite(str(color_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"彩色掩膜写入失败：{color_path}")
            if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise RuntimeError(f"叠加图写入失败：{overlay_path}")
            save_comparison(comparison_path, rgb, color_mask, overlay)

            counts = np.bincount(pred_train_ids.reshape(-1), minlength=NUM_CLASSES)
            total = int(pred_train_ids.size)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "rgb_path": str(rgb_path),
                "multispectral_path": str(ms_path),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "tiles_inferred": int(tile_count),
                "predicted_pixels": total,
                "predicted_vegetation_pixels_tree_plus_lowveg": int(counts[4] + counts[5]),
                "predicted_vegetation_coverage_percent_tree_plus_lowveg": round(100.0 * float(counts[4] + counts[5]) / max(1, total), 4),
                "raw_id_mask_path": str(raw_path),
                "color_mask_path": str(color_path),
                "overlay_path": str(overlay_path),
                "comparison_path": str(comparison_path),
                "status": "PASS",
                "error": "",
            }
            for train_id, class_name in enumerate(class_names):
                safe_name = str(class_name).lower().replace(" ", "_").replace("/", "_")
                row[f"pred_pixels_{safe_name}"] = int(counts[train_id])
                row[f"pred_percent_{safe_name}"] = round(100.0 * float(counts[train_id]) / max(1, total), 4)
            result_rows.append(row)
            contact_records.append({"sample_id": sample_id, "rgb_path": str(rgb_path), "overlay_path": str(overlay_path)})
            completed += 1
            elapsed = time.time() - started
            avg = elapsed / max(1, completed)
            eta = avg * (len(unlabeled_keys) - completed)
            print(
                f"[PREDICT] {idx:03d}/{len(unlabeled_keys):03d} {sample_id} "
                f"size={rgb.shape[1]}x{rgb.shape[0]} tiles={tile_count} "
                f"vegetation={row['predicted_vegetation_coverage_percent_tree_plus_lowveg']:.2f}% "
                f"image={time.time() - image_started:.1f}s elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                flush=True,
            )
        except Exception as exc:
            error_rows.append({"sample_id": sample_id, "rgb_path": str(rgb_path), "multispectral_path": str(ms_path),
                               "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {sample_id}: {type(exc).__name__}: {exc}", flush=True)

    write_csv(output_dir / "unlabeled_prediction_summary.csv", result_rows)
    write_csv(output_dir / "prediction_errors.csv", error_rows)
    if not args.skip_contact_sheets:
        sheets = save_contact_sheets(contact_records, dirs["contact"])
    else:
        sheets = 0

    total_elapsed = time.time() - started
    summary = {
        "task": "unlabeled TLSNMSP RGB+NIR semantic segmentation",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "protocol_dir": str(protocol_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_id": checkpoint_meta.get("model_id"),
        "fixed_nir_strength": fixed_strength,
        "device": str(device),
        "amp": amp,
        "tile_size": args.tile_size,
        "rgb_file_count": len(rgb_keys),
        "multispectral_file_count": len(ms_keys),
        "paired_rgb_nir_count": len(paired_keys),
        "raw_mask_file_count": len(indexes["mask"]),
        "converted_mask_file_count": len(indexes["mask_converted"]),
        "labeled_stem_count_excluded": len(labeled_keys),
        "unlabeled_candidates_before_max_images": len([k for k in paired_keys if k not in labeled_keys]),
        "images_attempted": len(unlabeled_keys),
        "images_predicted_successfully": len(result_rows),
        "images_failed": len(error_rows),
        "rgb_without_multispectral_count": len(missing_ms),
        "multispectral_without_rgb_count": len(missing_rgb),
        "incomplete_label_pair_count": len(incomplete_label_keys),
        "predicted_vegetation_definition": "predicted tree + low_vegetation pixels; model estimate only, not ground-truth accuracy",
        "output_dir": str(output_dir),
        "contact_sheet_count": sheets,
        "elapsed_seconds": round(total_elapsed, 2),
        "outputs": {
            "raw_ids": str(dirs["raw"]),
            "color_masks": str(dirs["color"]),
            "overlays": str(dirs["overlay"]),
            "comparisons": str(dirs["comparison"]),
            "contact_sheets": str(dirs["contact"]),
            "csv_summary": str(output_dir / "unlabeled_prediction_summary.csv"),
            "errors": str(output_dir / "prediction_errors.csv"),
        },
        "notes": [
            "No source RGB, multispectral, mask, or model checkpoint files are modified.",
            "Any sample with either original mask or converted mask is excluded from unlabeled inference.",
            "Vegetation coverage is the model-predicted proportion of tree + low_vegetation classes; without ground truth it is not an accuracy measurement.",
            "Raw prediction IDs are written as 1..7 to match mask_converted conventions; model training IDs are 0..6 internally.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[COMPLETE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if error_rows:
        print(f"[WARNING] 有 {len(error_rows)} 张影像推理失败，请查看：{output_dir / 'prediction_errors.csv'}")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断推理。原始数据和 checkpoint 未被修改。", file=sys.stderr)
        raise SystemExit(130)
