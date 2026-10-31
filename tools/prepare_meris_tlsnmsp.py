#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a reproducible RGB+NIR+label protocol for TLSNMSP. Read-only on source data."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
REQUIRED_FOLDERS = ("RGB", "multispectral", "mask", "mask_converted")
EXTENSIONS = {
    "RGB": {".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "multispectral": {".tif", ".tiff"},
    "mask": {".png", ".tif", ".tiff"},
    "mask_converted": {".png", ".tif", ".tiff"},
}
SPLIT_GROUPS = {
    "train": {"100", "101", "103", "106", "107"},
    "val": {"102"},
    "test": {"108"},
}
CLASS_NAMES = [
    "building", "impervious_surface", "water", "barren", "tree", "low_vegetation", "background_other"
]
RAW_LABEL_TO_TRAIN_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
RGB_COLOR_TO_RAW_ID = {
    (178, 74, 64): 1,       # #B24A40 building
    (128, 128, 128): 2,     # #808080 impervious surface
    (135, 206, 235): 3,    # #87CEEB water
    (210, 180, 100): 4,    # #D2B464 barren
    (34, 139, 34): 5,      # #228B22 tree
    (144, 238, 144): 6,    # #90EE90 low vegetation
    (255, 255, 0): 7,      # #FFFF00 background / other objects
}
GROUP_RE = re.compile(r"^(\d+)FPLAN_DJI_", re.IGNORECASE)


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"RGB 读取失败: {path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"不支持的 RGB shape {arr.shape}: {path}")
    if arr.dtype != np.uint8:
        raise RuntimeError(f"RGB 应为 uint8，实际 {arr.dtype}: {path}")
    return arr


def load_ms(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维数组，实际 {arr.shape}: {path}")
    if arr.shape[0] == 5:
        chw = arr
    elif arr.shape[-1] == 5:
        chw = np.moveaxis(arr, -1, 0)
    else:
        raise RuntimeError(f"多光谱 TIFF 无法识别五波段布局，实际 {arr.shape}: {path}")
    if chw.dtype != np.uint16:
        raise RuntimeError(f"多光谱 TIFF 预期 uint16，实际 {chw.dtype}: {path}")
    return chw


def load_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败: {path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"转换标签必须是单通道 ID 图，实际 {arr.shape}: {path}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}: {path}")
    return arr


def verify_original_mask(original_path: Path, converted: np.ndarray, sample_id: str):
    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"原始彩色标签读取失败: {original_path}")
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    if original.shape[:2] != converted.shape:
        raise RuntimeError(f"{sample_id}: 原始 mask 与 converted 尺寸不一致")
    packed = (original[:, :, 0].astype(np.uint32) << 16) | (original[:, :, 1].astype(np.uint32) << 8) | original[:, :, 2].astype(np.uint32)
    color_to_id = {((r << 16) | (g << 8) | b): cls for (r, g, b), cls in RGB_COLOR_TO_RAW_ID.items()}
    expected = np.full(packed.shape, 0, dtype=np.uint8)
    unknown = np.zeros(packed.shape, dtype=bool)
    for color_key in np.unique(packed):
        cls = color_to_id.get(int(color_key))
        mask = packed == color_key
        if cls is None:
            unknown |= mask
        else:
            expected[mask] = cls
    if unknown.any():
        unknown_colors = [f"#{int(v):06X}" for v in np.unique(packed[unknown])[:10]]
        raise RuntimeError(f"{sample_id}: 原始 mask 有未知颜色 {unknown_colors}")
    mismatch = int(np.count_nonzero(expected != converted))
    if mismatch:
        raise RuntimeError(f"{sample_id}: 原始 mask 与 converted 有 {mismatch} 个像素不一致")


def get_group(sample_id: str) -> str:
    m = GROUP_RE.match(sample_id)
    if not m:
        raise RuntimeError(f"无法从样本名识别航带组: {sample_id}")
    return m.group(1)


def scan_dir(root: Path, folder: str):
    d = root / folder
    if not d.is_dir():
        raise FileNotFoundError(f"缺少目录: {d}")
    idx = defaultdict(list)
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSIONS[folder]:
            idx[p.stem.casefold()].append(p)
    dup = {k: v for k, v in idx.items() if len(v) != 1}
    if dup:
        example = list(dup.items())[:5]
        raise RuntimeError(f"{folder} 中存在重复文件名 stem: {[(k,[str(p) for p in v]) for k,v in example]}")
    return {k: v[0] for k, v in idx.items()}


def main():
    ap = argparse.ArgumentParser(description="为 TLSNMSP 生成 train/val/test 清单和 NIR 归一化统计")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--verify-color-mask", action="store_true", default=True,
                    help="逐像素核对原始彩色标签和转换标签（默认开启）")
    ap.add_argument("--skip-color-mask-check", action="store_false", dest="verify_color_mask",
                    help="跳过原始彩色标签核对；不建议使用")
    ap.add_argument("--nir-sample-stride", type=int, default=8,
                    help="NIR 统计抽样步长，默认 8，即每 8 像素采一个点")
    args = ap.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"数据目录不存在: {root}\n请用 --data-root 指定训练电脑上的真实路径。")
    if args.nir_sample_stride < 1:
        raise SystemExit("--nir-sample-stride 必须 >= 1")

    print(f"[ROOT] {root}")
    print(f"[OUT ] {out}")
    print("[NOTE] 原始数据只读，不会修改、移动或删除。")
    indexes = {folder: scan_dir(root, folder) for folder in REQUIRED_FOLDERS}
    for folder, idx in indexes.items():
        print(f"[SCAN] {folder}: {len(idx)} files")

    # Only labeled, complete pairs enter supervised protocol.
    candidate_keys = sorted(set(indexes["RGB"]) & set(indexes["multispectral"]) & set(indexes["mask"]) & set(indexes["mask_converted"]))
    if not candidate_keys:
        raise SystemExit("没有找到同时具有 RGB、多光谱、mask、mask_converted 的样本")

    rows_by_split = {"train": [], "val": [], "test": []}
    all_rows = []
    split_class_pixels = {k: Counter() for k in rows_by_split}
    class_pixel_counts = Counter()
    nir_sample_chunks = []
    report_lines = []
    dims_seen = Counter()
    label_mapping_pass = 0
    errors = []

    for n, key in enumerate(candidate_keys, 1):
        rgb_path = indexes["RGB"][key]
        ms_path = indexes["multispectral"][key]
        mask_path = indexes["mask"][key]
        label_path = indexes["mask_converted"][key]
        sample_id = rgb_path.stem
        group = get_group(sample_id)
        assigned = [split for split, groups in SPLIT_GROUPS.items() if group in groups]
        if len(assigned) != 1:
            raise RuntimeError(f"{sample_id}: 航带组 {group} 没有唯一的 train/val/test 划分，请先修订 SPLIT_GROUPS")
        split = assigned[0]

        try:
            rgb = load_rgb(rgb_path)
            ms = load_ms(ms_path)
            label = load_label(label_path)
            h, w = label.shape
            if rgb.shape[:2] != (h, w) or ms.shape[1:] != (h, w):
                raise RuntimeError(f"尺寸不一致 RGB={rgb.shape}, MS={ms.shape}, label={label.shape}")
            if set(np.unique(label).astype(int).tolist()) - set(RAW_LABEL_TO_TRAIN_ID):
                raise RuntimeError(f"转换标签存在 1..7 以外的 ID: {np.unique(label).tolist()}")
            if args.verify_color_mask:
                verify_original_mask(mask_path, label, sample_id)
                label_mapping_pass += 1

            values, counts = np.unique(label, return_counts=True)
            for value, count in zip(values, counts):
                class_pixel_counts[int(value)] += int(count)
                split_class_pixels[split][int(value)] += int(count)

            if split == "train":
                nir = ms[4].astype(np.float32, copy=False)
                sampled = nir[::args.nir_sample_stride, ::args.nir_sample_stride].reshape(-1)
                # DN values in this dataset are positive; still guard against non-finite values.
                sampled = sampled[np.isfinite(sampled)]
                if sampled.size:
                    nir_sample_chunks.append(sampled.astype(np.float64, copy=False))

            row = {
                "sample_id": sample_id,
                "group": group,
                "split": split,
                "rgb_path": rgb_path.relative_to(root).as_posix(),
                "multispectral_path": ms_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
                "label_path": label_path.relative_to(root).as_posix(),
                "width": w,
                "height": h,
                "rgb_channels_used": 3,
                "multispectral_band_count": 5,
                "nir_band_index_1based": 5,
                "label_ids": ";".join(str(int(v)) for v in values),
                "label_mapping_verified": bool(args.verify_color_mask),
            }
            rows_by_split[split].append(row)
            all_rows.append(row)
            dims_seen[(w, h)] += 1
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {sample_id}: {exc}")
        if n % 25 == 0 or n == len(candidate_keys):
            print(f"[CHECK] {n}/{len(candidate_keys)}")

    if errors:
        err_path = out / "preparation_errors.csv"
        write_csv(err_path, errors, ["sample_id", "error"])
        raise SystemExit(f"发现 {len(errors)} 个样本问题，已写入 {err_path}。修复前不应训练。")

    for split, rows in rows_by_split.items():
        if not rows:
            raise SystemExit(f"{split} split 为空，请检查 SPLIT_GROUPS")
    # Split groups must remain disjoint.
    groups = {split: {r["group"] for r in rows} for split, rows in rows_by_split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise SystemExit(f"航带组泄漏：{a}/{b} 重叠 {groups[a] & groups[b]}")

    nir_samples = np.concatenate(nir_sample_chunks) if nir_sample_chunks else np.array([], dtype=np.float64)
    if nir_samples.size < 1000:
        raise SystemExit("NIR 训练集统计样本不足，拒绝生成训练协议")
    nir_mean = float(nir_samples.mean(dtype=np.float64))
    nir_std = float(nir_samples.std(dtype=np.float64))
    if not np.isfinite(nir_mean) or not np.isfinite(nir_std) or nir_std <= 0:
        raise SystemExit(f"NIR 统计异常 mean={nir_mean}, std={nir_std}")
    nir_p02, nir_p98 = [float(x) for x in np.percentile(nir_samples, [2, 98])]

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest_all.csv", all_rows)
    for split, rows in rows_by_split.items():
        write_csv(out / f"{split}.csv", rows)
    class_rows = []
    for split, counter in split_class_pixels.items():
        total = sum(counter.values())
        for raw_id in range(1, 8):
            count = int(counter.get(raw_id, 0))
            class_rows.append({
                "split": split,
                "raw_label_id": raw_id,
                "train_label_id": RAW_LABEL_TO_TRAIN_ID[raw_id],
                "class_name": CLASS_NAMES[RAW_LABEL_TO_TRAIN_ID[raw_id]],
                "pixel_count": count,
                "pixel_percent": round(count * 100.0 / total, 8) if total else 0.0,
                "total_pixels": total,
            })
    write_csv(out / "class_pixel_counts_by_split.csv", class_rows)

    config = {
        "protocol_version": "TLSNMSP_RGB_NIR_7CLASS_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(root),
        "project_root": str(PROJECT_ROOT),
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_groups": {split: sorted(v) for split, v in groups.items()},
        "class_names_train_id_order": CLASS_NAMES,
        "raw_label_to_train_id": {str(k): v for k, v in RAW_LABEL_TO_TRAIN_ID.items()},
        "ignore_index": 255,
        "rgb_normalization": {"type": "ImageNet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "nir_normalization": {
            "method": "train-only sampled z-score on raw DN values",
            "band_index_1based": 5,
            "sample_stride": args.nir_sample_stride,
            "mean_dn": nir_mean,
            "std_dn": nir_std,
            "p02_dn": nir_p02,
            "p98_dn": nir_p98,
            "sampled_value_count": int(nir_samples.size),
        },
        "crop_size": 512,
        "input_modalities": ["RGB", "NIR"],
        "original_mask_color_mapping_verified_samples": label_mapping_pass,
        "dimension_counts": {f"{w}x{h}": int(c) for (w, h), c in dims_seen.items()},
        "source_files_untouched": True,
    }
    (out / "dataset_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "TLSNMSP 训练协议准备报告", "=" * 44,
        f"数据根目录: {root}", f"输出目录: {out}",
        f"有标签完整配对样本: {len(all_rows)}",
        f"train/val/test: {len(rows_by_split['train'])}/{len(rows_by_split['val'])}/{len(rows_by_split['test'])}",
        f"航带组划分: {json.dumps({k: sorted(v) for k,v in groups.items()}, ensure_ascii=False)}",
        f"尺寸分布: {dict((f'{w}x{h}', c) for (w,h),c in dims_seen.items())}",
        f"原始 mask 与转换标签逐像素核对通过: {label_mapping_pass}/{len(all_rows)}",
        f"NIR band 5 train-only sampled mean/std: {nir_mean:.6f}/{nir_std:.6f} DN",
        f"NIR sampled p02/p98: {nir_p02:.3f}/{nir_p98:.3f} DN (for audit only; z-score uses mean/std)",
        "", "注意:",
        "1. 本脚本只生成协议文件，不改动原始影像。",
        "2. 训练/验证/测试按航带组划分，三组不重叠。",
        "3. NIR 归一化统计只从训练组计算，避免使用验证/测试分布。",
        "4. 标签原始 ID 1..7 映射到训练 ID 0..6。",
        "5. 尺寸一致不等于严格证明像素级配准；请人工检查 RGB/NIR/标签边界。",
        "6. 该 split 是本项目自定义的实验划分，不是数据集官方划分。",
    ]
    (out / "preparation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n[PASS] 数据协议生成完成")
    print(f"[SPLIT] train={len(rows_by_split['train'])}, val={len(rows_by_split['val'])}, test={len(rows_by_split['test'])}")
    print(f"[NIR] mean={nir_mean:.4f}, std={nir_std:.4f} DN; sampled={nir_samples.size:,}")
    print(f"[OUT] {out / 'preparation_report.txt'}")
    print(f"[OUT] {out / 'dataset_config.json'}")
    print("下一步先运行 smoke test，不要立即启动长时间训练。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。源数据未修改。", file=sys.stderr)
        raise SystemExit(130)
