#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pixel-level audit of TLSNMSP original color masks vs mask_converted labels.

The protocol ZIP contains only a summary saying 176/176 samples passed; it does
not include the authoritative RGB-color -> raw-label-ID table. This script can:
  1) audit per-pixel agreement against an authoritative JSON color map; or
  2) without a map, report whether each observed RGB color consistently maps to
     one converted label ID. Mode 2 is a consistency audit, NOT proof that the
     semantic class assigned to each color is correct.

Expected color map JSON format (RGB order, raw label IDs 1..7):
{
  "color_space": "RGB",
  "colors": {
    "#RRGGBB": 1,
    "#RRGGBB": 2
  }
}
Replace the example colors with the verified palette from the dataset/prep code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASS_NAMES_RAW = {
    1: "building",
    2: "impervious_surface",
    3: "water",
    4: "barren",
    5: "tree",
    6: "low_vegetation",
    7: "background_other",
}

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audits" / "meris_raw_mask_mapping"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到协议清单：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(root: Path, rel: str) -> Path:
    return (root / Path(rel.replace("/", "\\"))).resolve()


def read_original_mask(path: Path) -> tuple[np.ndarray, str]:
    """Return original mask as RGB uint8. OpenCV decodes color images as BGR."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"原始 mask 读取失败：{path}")
    if arr.ndim == 2:
        # Grayscale or indexed PNG decoded as a single channel; preserve values.
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
        kind = "single_channel_replicated"
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
        kind = "color_rgb"
    else:
        raise RuntimeError(f"不支持的原始 mask 形状 {arr.shape}: {path}")
    if rgb.dtype != np.uint8:
        if rgb.min() < 0 or rgb.max() > 255:
            raise RuntimeError(f"原始 mask 颜色通道超出 uint8 范围：dtype={rgb.dtype}, {path}")
        rgb = rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb), kind


def read_converted_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"mask_converted 读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"mask_converted 不是单通道类别 ID 图：shape={arr.shape}, path={path}")
    if arr.ndim != 2:
        raise RuntimeError(f"mask_converted 维度异常：shape={arr.shape}, path={path}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise RuntimeError(f"mask_converted 应为整数类别 ID，实际 dtype={arr.dtype}, path={path}")
    return arr.astype(np.int16, copy=False)


def parse_rgb_key(key: str) -> tuple[int, int, int]:
    s = key.strip().upper()
    if s.startswith("#"):
        s = s[1:]
    s = s.replace(" ", "")
    if len(s) != 6:
        raise ValueError(f"颜色键必须是 #RRGGBB 格式：{key!r}")
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"无效 RGB 颜色键：{key!r}") from exc


def load_color_map(path: Path | None) -> dict[tuple[int, int, int], int] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"指定的颜色映射 JSON 不存在：{path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(obj, dict) and "colors" in obj:
        color_space = str(obj.get("color_space", "RGB")).upper()
        if color_space != "RGB":
            raise ValueError("当前脚本的 color-map JSON 必须声明 color_space=RGB")
        entries = obj["colors"]
    else:
        entries = obj
    if not isinstance(entries, dict):
        raise ValueError("颜色映射 JSON 应为对象，或包含 colors 对象")
    mapping: dict[tuple[int, int, int], int] = {}
    for key, value in entries.items():
        rgb = parse_rgb_key(str(key))
        raw_id = int(value)
        if raw_id not in CLASS_NAMES_RAW:
            raise ValueError(f"颜色 {key} 对应 raw ID={raw_id}，必须为 1..7")
        if rgb in mapping and mapping[rgb] != raw_id:
            raise ValueError(f"同一颜色被映射到多个 raw ID：{key}")
        mapping[rgb] = raw_id
    return mapping


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="逐像素核验 TLSNMSP 原始彩色 mask 与 mask_converted")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    ap.add_argument("--color-map-json", type=Path, default=None,
                    help="权威 RGB 颜色->原始标签 ID 映射 JSON；提供后才能判断语义映射是否逐像素正确")
    ap.add_argument("--max-mismatch-examples", type=int, default=1000)
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据根目录不存在：{data_root}")
    if args.max_mismatch_examples < 0:
        raise SystemExit("--max-mismatch-examples 必须 >= 0")

    manifest = read_csv_rows(protocol_dir / "manifest_all.csv")
    if args.split != "all":
        manifest = [r for r in manifest if r.get("split", "").lower() == args.split]
    if not manifest:
        raise SystemExit(f"manifest 中没有 split={args.split} 的样本")
    color_map_path = args.color_map_json.expanduser().resolve() if args.color_map_json else None
    color_map = load_color_map(color_map_path)

    image_rows: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    global_color_label_counts: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    global_color_pixel_counts: Counter[tuple[int, int, int]] = Counter()
    global_color_map_mismatch: Counter[tuple[int, int, int]] = Counter()
    all_observed_colors: set[tuple[int, int, int]] = set()
    map_unused_colors = set(color_map or {})
    total_pixels = 0
    total_invalid_converted = 0
    total_mismatch_pixels = 0
    total_unknown_colors = 0
    shape_errors = 0
    read_errors = 0
    consistent_image_count = 0
    start_time = time.time()

    for index, row in enumerate(manifest, 1):
        sample_id = row.get("sample_id", f"sample_{index}")
        raw_path = resolve_path(data_root, row["mask_path"])
        label_path = resolve_path(data_root, row["label_path"])
        base = {
            "sample_id": sample_id,
            "split": row.get("split", ""),
            "mask_path": str(raw_path),
            "label_path": str(label_path),
        }
        try:
            rgb, raw_kind = read_original_mask(raw_path)
            label = read_converted_label(label_path)
        except Exception as exc:
            read_errors += 1
            image_rows.append({**base, "status": "READ_ERROR", "error": str(exc)})
            print(f"[ERROR] {sample_id}: {exc}", flush=True)
            continue

        if rgb.shape[:2] != label.shape:
            shape_errors += 1
            image_rows.append({**base, "status": "SHAPE_MISMATCH", "raw_shape": str(rgb.shape),
                               "converted_shape": str(label.shape), "error": "原始 mask 与转换标签尺寸不一致"})
            print(f"[ERROR] {sample_id}: shape mismatch raw={rgb.shape}, converted={label.shape}", flush=True)
            continue

        h, w = label.shape
        n_pix = int(label.size)
        total_pixels += n_pix
        valid_ids = np.isin(label, list(CLASS_NAMES_RAW))
        invalid_count = int((~valid_ids).sum())
        total_invalid_converted += invalid_count

        flat_rgb = rgb.reshape(-1, 3)
        flat_label = label.reshape(-1)
        unique_rgb, inverse, color_counts = np.unique(flat_rgb, axis=0, return_inverse=True, return_counts=True)
        local_mismatch = 0
        local_unknown = 0
        local_color_consistent = True
        for color_idx, color_arr in enumerate(unique_rgb):
            color = tuple(int(x) for x in color_arr.tolist())
            pix_idx = np.flatnonzero(inverse == color_idx)
            ids, id_counts = np.unique(flat_label[pix_idx], return_counts=True)
            distribution = {int(cid): int(cnt) for cid, cnt in zip(ids, id_counts)}
            global_color_label_counts[color].update(distribution)
            global_color_pixel_counts[color] += int(color_counts[color_idx])
            all_observed_colors.add(color)
            if len(distribution) != 1:
                local_color_consistent = False

            if color_map is not None:
                if color not in color_map:
                    local_unknown += int(color_counts[color_idx])
                    total_unknown_colors += int(color_counts[color_idx])
                    if len(mismatch_examples) < args.max_mismatch_examples:
                        for flat_pos in pix_idx[:max(0, args.max_mismatch_examples - len(mismatch_examples))]:
                            yy, xx = divmod(int(flat_pos), w)
                            mismatch_examples.append({"sample_id": sample_id, "x": xx, "y": yy,
                                "rgb": rgb_hex(color), "expected_raw_id": "UNKNOWN_COLOR",
                                "observed_converted_id": int(flat_label[flat_pos]), "reason": "原始颜色未出现在权威映射表"})
                else:
                    map_unused_colors.discard(color)
                    expected_id = color_map[color]
                    wrong_mask = flat_label[pix_idx] != expected_id
                    wrong_count = int(wrong_mask.sum())
                    if wrong_count:
                        local_mismatch += wrong_count
                        total_mismatch_pixels += wrong_count
                        global_color_map_mismatch[color] += wrong_count
                        if len(mismatch_examples) < args.max_mismatch_examples:
                            wrong_positions = pix_idx[np.flatnonzero(wrong_mask)]
                            remaining = args.max_mismatch_examples - len(mismatch_examples)
                            for flat_pos in wrong_positions[:remaining]:
                                yy, xx = divmod(int(flat_pos), w)
                                mismatch_examples.append({"sample_id": sample_id, "x": xx, "y": yy,
                                    "rgb": rgb_hex(color), "expected_raw_id": expected_id,
                                    "observed_converted_id": int(flat_label[flat_pos]),
                                    "reason": "颜色映射对应的标签 ID 与转换标签不一致"})

        if local_color_consistent:
            consistent_image_count += 1
        local_status = "CONSISTENT_ONLY" if color_map is None else ("PASS" if local_mismatch == 0 and local_unknown == 0 and invalid_count == 0 else "FAIL")
        image_rows.append({**base, "status": local_status, "height": h, "width": w,
            "raw_mask_decode_type": raw_kind, "unique_rgb_colors": int(len(unique_rgb)),
            "pixel_count": n_pix, "invalid_converted_label_pixels": invalid_count,
            "color_to_label_consistency": bool(local_color_consistent),
            "mismatched_pixels": local_mismatch, "unknown_color_pixels": local_unknown,
            "mismatch_rate_percent": round(100.0 * local_mismatch / max(1, n_pix), 8),
            "error": ""})
        if index % 10 == 0 or index == len(manifest):
            print(f"[AUDIT] {index}/{len(manifest)} images | {sample_id} | colors={len(unique_rgb)} | "
                  f"mismatch={local_mismatch} | unknown={local_unknown}", flush=True)

    # Aggregate color -> observed converted-ID distributions.
    color_rows: list[dict[str, Any]] = []
    global_consistency_pass = True
    for color in sorted(global_color_pixel_counts, key=lambda c: (-global_color_pixel_counts[c], c)):
        counts = global_color_label_counts[color]
        total_for_color = int(sum(counts.values()))
        dominant_id, dominant_count = max(counts.items(), key=lambda item: item[1])
        consistent = len(counts) == 1
        global_consistency_pass = global_consistency_pass and consistent
        expected_id = color_map.get(color) if color_map is not None else None
        color_rows.append({
            "rgb_hex": rgb_hex(color), "r": color[0], "g": color[1], "b": color[2],
            "pixel_count": total_for_color,
            "observed_converted_id_distribution_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
            "observed_converted_class_distribution_json": json.dumps({CLASS_NAMES_RAW.get(k, f"invalid_{k}"): v for k, v in sorted(counts.items())}, ensure_ascii=False),
            "dominant_converted_raw_id": dominant_id,
            "dominant_class_name": CLASS_NAMES_RAW.get(dominant_id, f"invalid_{dominant_id}"),
            "dominant_id_purity_percent": round(100.0 * dominant_count / max(1, total_for_color), 8),
            "one_color_one_id_consistent": consistent,
            "expected_raw_id_from_authoritative_map": expected_id if expected_id is not None else "",
            "mismatched_pixels_vs_authoritative_map": int(global_color_map_mismatch[color]),
        })

    image_count_ok = sum(1 for r in image_rows if r.get("status") in ("PASS", "CONSISTENT_ONLY"))
    authoritative_pass = bool(color_map is not None and read_errors == 0 and shape_errors == 0
                              and total_invalid_converted == 0 and total_mismatch_pixels == 0
                              and total_unknown_colors == 0 and len(image_rows) == len(manifest))
    summary = {
        "audit_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root), "protocol_dir": str(protocol_dir),
        "split_requested": args.split, "manifest_samples": len(manifest),
        "images_read_successfully": len(manifest) - read_errors - shape_errors,
        "read_errors": read_errors, "shape_errors": shape_errors,
        "total_valid_image_pixels": total_pixels,
        "total_converted_label_pixels_outside_1_to_7": total_invalid_converted,
        "observed_unique_rgb_colors": len(all_observed_colors),
        "images_with_one_color_to_one_converted_id_consistency": consistent_image_count,
        "global_color_to_converted_id_consistency_pass": bool(global_consistency_pass and read_errors == 0 and shape_errors == 0 and total_invalid_converted == 0),
        "authoritative_color_map_provided": color_map is not None,
        "authoritative_color_map_path": str(color_map_path) if color_map_path else None,
        "authoritative_map_color_entries": len(color_map) if color_map is not None else 0,
        "observed_colors_missing_from_authoritative_map": len(all_observed_colors - set(color_map or {})) if color_map is not None else None,
        "authoritative_map_entries_not_observed": len(map_unused_colors) if color_map is not None else None,
        "unknown_color_pixels": total_unknown_colors,
        "pixels_not_matching_authoritative_mapping": total_mismatch_pixels,
        "pixelwise_semantic_mapping_pass": authoritative_pass if color_map is not None else None,
        "interpretation": (
            "AUTHORITATIVE_PIXELWISE_SEMANTIC_MAPPING_PASS" if authoritative_pass else
            "AUTHORITATIVE_PIXELWISE_SEMANTIC_MAPPING_FAIL" if color_map is not None else
            "CONSISTENCY_ONLY: no authoritative RGB-to-class map supplied; this run cannot prove semantic correctness"
        ),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "notes": [
            "Converted label IDs are expected to be raw IDs 1..7; the model training script later maps them to 0..6.",
            "Without --color-map-json, the script only checks whether each observed RGB color consistently corresponds to converted IDs; it does not prove the semantic class is correct.",
            "A semantic PASS requires an authoritative color map independently confirmed from the dataset specification or original preparation script.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image_pixel_audit.csv", image_rows)
    write_csv(output_dir / "color_to_converted_id_distribution.csv", color_rows,
              ["rgb_hex", "r", "g", "b", "pixel_count", "observed_converted_id_distribution_json",
               "observed_converted_class_distribution_json", "dominant_converted_raw_id", "dominant_class_name",
               "dominant_id_purity_percent", "one_color_one_id_consistent",
               "expected_raw_id_from_authoritative_map", "mismatched_pixels_vs_authoritative_map"])
    write_csv(output_dir / "mismatch_pixel_examples.csv", mismatch_examples,
              ["sample_id", "x", "y", "rgb", "expected_raw_id", "observed_converted_id", "reason"])
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OUTPUT] {output_dir}")
    if color_map is None:
        print("[IMPORTANT] 当前未提供权威颜色映射表；本次结果只能证明颜色-ID一致性，不能证明语义映射正确。")
        return 0 if summary["global_color_to_converted_id_consistency_pass"] else 2
    return 0 if authoritative_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断核验。", file=sys.stderr)
        raise SystemExit(130)
