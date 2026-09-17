#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISPRS Potsdam Dataset Protocol v1 的正式 Dataset 实现。

本模块只负责数据层：
- 读取四份冻结 metadata：
    data/processed/potsdam/labels_manifest.json
    data/processed/potsdam/tile_split.json
    data/processed/potsdam/rgbir_semantics.json
    data/processed/potsdam/dataset_protocol.json
- RGBIR 只读取 4_Ortho_RGBIR，使用 tifffile
- Train：512x512 deterministic on-the-fly crop + deterministic D4
- Val/Test：deterministic 512x512 sliding windows, stride=384
- RGB：ImageNet normalization
- NIR：冻结的 train-only mean/std
- GT：六色 RGB -> 0..5，unknown hard error，255 仅为 sentinel

本模块明确不包含：
- 模型
- 训练循环
- corruption
- RGB+NIR 4-channel SegFormer
- dual encoder
- fixed fusion
- Quality Gate
- sliding-window logit fusion（只提供固定窗口坐标；融合在后续 inference 模块实现）

设计原则：
1. Train Dataset 的 index -> (tile, sample_slot) 采用 tile-major：
   index = tile_index * 64 + sample_slot
   与 audit_dataset_protocol.py 的 schedule 顺序一致。
2. 坐标 RNG 与 D4 RNG 使用相同的 SHA256 stateless 规则。
3. Val/Test Dataset 不返回 patch GT，防止误用 patch mIoU。
   推理完成后通过 load_full_label(tile_id) 获取整 tile GT。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


MODULE_VERSION = "1.0.0"
PROTOCOL_NAME = "Dataset Protocol v1"

EXPECTED_DATA_SEED = 20260917
EXPECTED_TILE_SIZE = 6000
EXPECTED_CROP_SIZE = 512
EXPECTED_CROPS_PER_TILE_PER_EPOCH = 64
EXPECTED_SAMPLES_PER_EPOCH = 1152
EXPECTED_CAT_MAX_RATIO = 0.75
EXPECTED_MAX_CANDIDATES = 10
EXPECTED_SLIDING_STRIDE = 384
EXPECTED_IGNORE_INDEX = 255
EXPECTED_NUM_CLASSES = 6
EXPECTED_NIR_CHANNEL = 3
EXPECTED_WINDOW_COORD_SHA256 = (
    "58a2f857c7d84adb0a17f2ce0f5b1802452886127963f42fd9ac125c7dd84e1a"
)

EXPECTED_CLASSES = [
    {"id": 0, "name": "Impervious surfaces", "rgb": [255, 255, 255]},
    {"id": 1, "name": "Building",            "rgb": [0, 0, 255]},
    {"id": 2, "name": "Low vegetation",      "rgb": [0, 255, 255]},
    {"id": 3, "name": "Tree",                "rgb": [0, 255, 0]},
    {"id": 4, "name": "Car",                 "rgb": [255, 255, 0]},
    {"id": 5, "name": "Clutter/background",  "rgb": [255, 0, 0]},
]

IMAGE_EXTS = (".tif", ".tiff")
TILE_ID_RE = re.compile(r"^\d+_\d+$")
RGBIR_FILENAME_RE = re.compile(r"(?:^|_)(\d+_\d+)_RGBIR(?:\.|_)", re.IGNORECASE)


class DatasetProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetProtocolError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"缺少冻结文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise DatasetProtocolError(f"JSON 读取失败：{path}\n{e}") from e


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------
# split schema
# ---------------------------------------------------------------------

def _is_tile_list_like(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(
            isinstance(value.get(key), list)
            for key in ("tiles", "tile_ids", "ids", "items")
        )
    return False


def _looks_like_split_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    lower = {str(k).lower(): k for k in obj.keys()}
    train_key = next((lower[k] for k in ("train", "training") if k in lower), None)
    val_key = next((lower[k] for k in ("val", "valid", "validation") if k in lower), None)
    test_key = next((lower[k] for k in ("test", "testing") if k in lower), None)
    if train_key is None or val_key is None or test_key is None:
        return False
    return (
        _is_tile_list_like(obj[train_key])
        and _is_tile_list_like(obj[val_key])
        and _is_tile_list_like(obj[test_key])
    )


def _find_split_dict(obj: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, dict):
        if _looks_like_split_dict(obj.get("splits")):
            return obj["splits"]
        if _looks_like_split_dict(obj):
            return obj
        for v in obj.values():
            found = _find_split_dict(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_split_dict(v)
            if found is not None:
                return found
    return None


def _normalize_tile_list(value: Any, name: str) -> List[str]:
    if isinstance(value, dict):
        for key in ("tiles", "tile_ids", "ids", "items"):
            if key in value:
                value = value[key]
                break
    require(isinstance(value, list), f"tile_split 中 {name} 不是 tile list")
    out = [str(x) for x in value]
    require(all(TILE_ID_RE.match(x) for x in out),
            f"tile_split 中 {name} 含非法 tile id")
    require(len(out) == len(set(out)), f"tile_split 中 {name} 存在重复 tile")
    return out


def parse_split(split_obj: Any) -> Dict[str, List[str]]:
    if isinstance(split_obj, dict) and _looks_like_split_dict(split_obj.get("splits")):
        d = split_obj["splits"]
    else:
        d = _find_split_dict(split_obj)
    require(d is not None, "无法识别 tile_split.json 的 train/val/test tile list")

    lower = {str(k).lower(): k for k in d.keys()}
    train_key = next(lower[k] for k in ("train", "training") if k in lower)
    val_key = next(lower[k] for k in ("val", "valid", "validation") if k in lower)
    test_key = next(lower[k] for k in ("test", "testing") if k in lower)

    train = _normalize_tile_list(d[train_key], "train")
    val = _normalize_tile_list(d[val_key], "val")
    test = _normalize_tile_list(d[test_key], "test")

    require(not (set(train) & set(val)), "train/val overlap")
    require(not (set(train) & set(test)), "train/test overlap")
    require(not (set(val) & set(test)), "val/test overlap")
    require(len(train) == 18 and len(val) == 6 and len(test) == 14,
            f"split 数量异常：{len(train)}/{len(val)}/{len(test)}")
    require(len(set(train) | set(val) | set(test)) == 38,
            "split union 不是 38 tiles")

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------
# rgbir semantics
# ---------------------------------------------------------------------

def _canon_channel_name(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = re.sub(r"[^A-Z0-9]+", "", x.upper())
    aliases = {
        "R": "R", "RED": "R",
        "G": "G", "GREEN": "G",
        "B": "B", "BLUE": "B",
        "IR": "NIR", "NIR": "NIR", "NEARIR": "NIR",
        "NEARINFRARED": "NIR", "INFRARED": "NIR",
    }
    return aliases.get(s)


def _parse_index_key(k: Any) -> Optional[int]:
    if isinstance(k, int) and 0 <= k <= 3:
        return k
    s = str(k).strip().lower()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i <= 3 else None
    m = re.fullmatch(r"(?:ch|channel|band)[_\-\s]*([0-3])", s)
    return int(m.group(1)) if m else None


def collect_channel_mapping_candidates(obj: Any) -> List[List[str]]:
    candidates: List[List[str]] = []

    def add(seq: Sequence[Any]) -> None:
        if len(seq) != 4:
            return
        canon = [_canon_channel_name(v) for v in seq]
        if all(v is not None for v in canon):
            candidates.append([str(v) for v in canon])

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            idx_to_name = {}
            for k, v in x.items():
                idx = _parse_index_key(k)
                name = _canon_channel_name(v)
                if idx is not None and name is not None:
                    idx_to_name[idx] = name
            if set(idx_to_name) == {0, 1, 2, 3}:
                add([idx_to_name[i] for i in range(4)])

            name_to_idx = {}
            for k, v in x.items():
                name = _canon_channel_name(k)
                if name is not None and isinstance(v, int) and 0 <= v <= 3:
                    name_to_idx[name] = v
            if set(name_to_idx) >= {"R", "G", "B", "NIR"}:
                seq = [None] * 4
                for name, idx in name_to_idx.items():
                    if name in {"R", "G", "B", "NIR"}:
                        seq[idx] = name
                if all(seq):
                    add(seq)

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            if len(x) == 4 and all(isinstance(v, str) for v in x):
                add(x)

            if len(x) >= 4 and all(isinstance(v, dict) for v in x):
                idx_to_name = {}
                for item in x:
                    idx = None
                    name = None
                    for k in ("index", "channel_index", "band_index", "id"):
                        if k in item and isinstance(item[k], int):
                            idx = item[k]
                            break
                    for k in ("name", "semantic", "channel", "band"):
                        if k in item:
                            name = _canon_channel_name(item[k])
                            if name:
                                break
                    if idx is not None and 0 <= idx <= 3 and name is not None:
                        idx_to_name[idx] = name
                if set(idx_to_name) == {0, 1, 2, 3}:
                    add([idx_to_name[i] for i in range(4)])

            for v in x:
                walk(v)

    walk(obj)

    unique = []
    seen = set()
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c)
    return unique


def validate_rgbir_semantics(obj: Any) -> None:
    candidates = collect_channel_mapping_candidates(obj)
    require(["R", "G", "B", "NIR"] in candidates,
            f"rgbir_semantics.json 未确认 [R,G,B,NIR]；识别到：{candidates}")


# ---------------------------------------------------------------------
# labels_manifest canonical GT path parser
# ---------------------------------------------------------------------

def _find_tile_id_in_dict(d: Mapping[str, Any]) -> Optional[str]:
    for key in ("tile_id", "tile", "id", "name"):
        if key in d and isinstance(d[key], (str, int)):
            s = str(d[key])
            if TILE_ID_RE.match(s):
                return s
    return None


def collect_manifest_entries(obj: Any) -> Dict[str, List[Any]]:
    found: Dict[str, List[Any]] = defaultdict(list)

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            direct = _find_tile_id_in_dict(x)
            if direct:
                found[direct].append(x)
            for k, v in x.items():
                ks = str(k)
                if TILE_ID_RE.match(ks):
                    found[ks].append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return dict(found)


def _canonical_source_hint(entry: Any) -> Optional[str]:
    hints: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if isinstance(v, str):
                    vl = v.lower().strip()
                    if (
                        ("source" in kl or "selected" in kl or "canonical" in kl)
                        and vl in ("participant", "all", "all_gt", "participant_gt")
                    ):
                        hints.append("participant" if "participant" in vl else "all")
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(entry)
    if not hints:
        return None
    return hints[0] if len(set(hints)) == 1 else None


def _collect_image_path_candidates(entry: Any) -> List[Tuple[int, str, Tuple[str, ...]]]:
    source_hint = _canonical_source_hint(entry)
    out: List[Tuple[int, str, Tuple[str, ...]]] = []

    def walk(x: Any, keys: Tuple[str, ...] = ()) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, keys + (str(k).lower(),))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, keys + (str(i),))
        elif isinstance(x, str):
            if not x.lower().endswith(IMAGE_EXTS):
                return
            ctx = "/".join(keys).lower()
            score = 0
            if any(tok in ctx for tok in ("canonical_gt", "canonical", "selected", "chosen", "resolved")):
                score += 300
            if any(tok in ctx for tok in ("gt_path", "label_path", "ground_truth", "label", "gt")):
                score += 120
            if any(tok in ctx for tok in ("path", "file", "filename")):
                score += 30
            if source_hint == "participant" and "participant" in ctx:
                score += 90
            if source_hint == "all" and re.search(r"(^|[/_])all($|[/_])", ctx):
                score += 90
            if "candidate" in ctx:
                score -= 30
            out.append((score, x, keys))

    walk(entry)
    return out


def _resolve_manifest_path(raw: str, project_root: Path, manifest_path: Path) -> Optional[Path]:
    p = Path(raw)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            project_root / "data/raw/potsdam" / p,
            project_root / "data/raw/potsdam/Potsdam" / p,
            project_root / p,
            manifest_path.parent / p,
            manifest_path.parent.parent / p,
        ])

    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return c.resolve()
    return None


def resolve_canonical_gt_paths(
    manifest_obj: Any,
    manifest_path: Path,
    project_root: Path,
    expected_tiles: Sequence[str],
) -> Dict[str, Path]:
    entries = collect_manifest_entries(manifest_obj)
    missing_entries = [t for t in expected_tiles if t not in entries]
    require(not missing_entries,
            f"labels_manifest.json 缺少 tile entries：{missing_entries}")

    resolved: Dict[str, Path] = {}
    for tile in expected_tiles:
        all_candidates: List[Tuple[int, str, Tuple[str, ...], Path]] = []
        for entry in entries[tile]:
            for score, raw, keys in _collect_image_path_candidates(entry):
                p = _resolve_manifest_path(raw, project_root, manifest_path)
                if p is not None:
                    all_candidates.append((score, raw, keys, p))

        require(
            all_candidates,
            f"{tile}: manifest 中找不到可解析的 canonical GT 路径；"
            "Dataset 不允许扫描多个 5_Labels* 目录重新选择 GT。"
        )

        best_score = max(x[0] for x in all_candidates)
        best = [x for x in all_candidates if x[0] == best_score]

        unique_best: Dict[str, Tuple[int, str, Tuple[str, ...], Path]] = {}
        for item in best:
            unique_best[str(item[3].resolve())] = item
        best = list(unique_best.values())

        require(
            len(best) == 1,
            f"{tile}: manifest 中存在多个同优先级 GT 路径，拒绝猜测："
            + "\n"
            + "\n".join(f"  score={x[0]} path={x[3]}" for x in best)
        )
        resolved[tile] = best[0][3]

    require(len({str(p) for p in resolved.values()}) == len(expected_tiles),
            "canonical GT 路径不是唯一 tile->file 映射")
    return resolved


# ---------------------------------------------------------------------
# 4_Ortho_RGBIR 定位
# ---------------------------------------------------------------------

def _direct_rgbir_tiles_in_dir(root: Path) -> Dict[str, Path]:
    tile_to_paths: Dict[str, List[Path]] = defaultdict(list)

    if not root.is_dir():
        return {}

    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = RGBIR_FILENAME_RE.search(p.name)
        if m:
            tile_to_paths[m.group(1)].append(p.resolve())

    duplicates = {
        tile: paths for tile, paths in tile_to_paths.items()
        if len(paths) != 1
    }
    require(not duplicates,
            f"RGBIR 目录存在同 tile 多文件：{root}\n{duplicates}")

    return {tile: paths[0] for tile, paths in tile_to_paths.items()}


def locate_rgbir_root(project_root: Path, expected_tiles: Sequence[str]) -> Path:
    search_root = project_root / "data/raw/potsdam"
    require(search_root.is_dir(), f"缺少 data/raw/potsdam：{search_root}")

    candidates = sorted(
        {p.resolve() for p in search_root.rglob("4_Ortho_RGBIR") if p.is_dir()},
        key=lambda p: (len(p.parts), str(p)),
    )
    require(candidates, "未找到名为 4_Ortho_RGBIR 的目录")

    expected = set(expected_tiles)
    valid: List[Path] = []

    for root in candidates:
        direct_map = _direct_rgbir_tiles_in_dir(root)
        if set(direct_map.keys()) == expected and len(direct_map) == len(expected_tiles):
            valid.append(root)

    require(
        len(valid) == 1,
        "无法唯一确定实际 4_Ortho_RGBIR TIFF 数据目录。"
        f"\n候选：{[str(x) for x in candidates]}"
        f"\n满足冻结 38 tiles 的候选：{[str(x) for x in valid]}"
    )
    return valid[0]


def build_rgbir_map(rgbir_root: Path, expected_tiles: Sequence[str]) -> Dict[str, Path]:
    direct_map = _direct_rgbir_tiles_in_dir(rgbir_root)
    expected = set(expected_tiles)
    found = set(direct_map.keys())
    require(found == expected and len(direct_map) == len(expected_tiles),
            f"RGBIR tile 集不等于冻结 tile 集：missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}")
    return {tile: direct_map[tile] for tile in expected_tiles}


# ---------------------------------------------------------------------
# TIFF store
# ---------------------------------------------------------------------

class _TiffStore:
    """
    先尝试 tifffile.memmap；若 TIFF 不可 mmap，则退回 tifffile.imread。
    两种方式均严格使用 tifffile。

    mmap cache 仅保存轻量映射对象；
    imread fallback 只缓存最近 1 个完整 array，避免每 worker 占用过大内存。
    """

    def __init__(self, max_memmaps: int = 8):
        self.max_memmaps = int(max_memmaps)
        self._memmaps: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._fallback_key: Optional[str] = None
        self._fallback_array: Optional[np.ndarray] = None

    def clear(self) -> None:
        self._memmaps.clear()
        self._fallback_key = None
        self._fallback_array = None

    def read(self, path: Path) -> np.ndarray:
        key = str(path.resolve())

        if key in self._memmaps:
            arr = self._memmaps.pop(key)
            self._memmaps[key] = arr
            return arr

        if self._fallback_key == key and self._fallback_array is not None:
            return self._fallback_array

        try:
            arr = tifffile.memmap(str(path), mode="r")
            if len(self._memmaps) >= self.max_memmaps:
                self._memmaps.popitem(last=False)
            self._memmaps[key] = arr
            return arr
        except Exception:
            try:
                arr = tifffile.imread(str(path))
            except Exception as e:
                raise DatasetProtocolError(f"tifffile 读取失败：{path}\n{e}") from e
            self._fallback_key = key
            self._fallback_array = arr
            return arr


# ---------------------------------------------------------------------
# stateless RNG / D4 / sliding coords
# ---------------------------------------------------------------------

def _key_bytes(parts: Sequence[Any], stream: str, counter: int) -> bytes:
    return (
        "|".join(str(x) for x in parts) + f"|{stream}|{counter}"
    ).encode("utf-8")


def stateless_randbelow(parts: Sequence[Any], n: int, stream: str) -> int:
    require(n > 0, "stateless_randbelow: n 必须 > 0")
    space = 1 << 256
    limit = space - (space % n)
    counter = 0
    while True:
        digest = hashlib.sha256(_key_bytes(parts, stream, counter)).digest()
        value = int.from_bytes(digest, "big", signed=False)
        if value < limit:
            return value % n
        counter += 1


def deterministic_candidate_xy(
    seed: int,
    epoch: int,
    tile_id: str,
    sample_slot: int,
    candidate_id: int,
    max_start: int,
) -> Tuple[int, int]:
    parts = (seed, epoch, tile_id, sample_slot, candidate_id)
    x = stateless_randbelow(parts, max_start + 1, "x")
    y = stateless_randbelow(parts, max_start + 1, "y")
    return x, y


def deterministic_d4_code(
    seed: int,
    epoch: int,
    tile_id: str,
    sample_slot: int,
) -> int:
    return stateless_randbelow(
        (seed, epoch, tile_id, sample_slot), 8, "d4"
    )


def apply_d4(arr: np.ndarray, code: int) -> np.ndarray:
    require(0 <= code <= 7, f"非法 D4 code={code}")
    if code < 4:
        out = np.rot90(arr, k=code, axes=(0, 1))
    else:
        out = np.fliplr(np.rot90(arr, k=code - 4, axes=(0, 1)))
    return np.ascontiguousarray(out)


def sliding_starts(length: int, crop: int, stride: int) -> List[int]:
    require(length >= crop, "sliding window: length < crop")
    last = length - crop
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_window_coordinates(starts: Sequence[int]) -> List[Tuple[int, int]]:
    # 与 audit 一致：y-major then x-major
    return [(y, x) for y in starts for x in starts]


# ---------------------------------------------------------------------
# 冻结 metadata loader
# ---------------------------------------------------------------------

class FrozenPotsdamSpec:
    """
    四份冻结 metadata 的唯一入口。
    初始化时做轻量一致性验证，但不会重新做昂贵的 full dataset audit。
    """

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.metadata_dir = self.project_root / "data/processed/potsdam"

        self.labels_manifest_path = self.metadata_dir / "labels_manifest.json"
        self.tile_split_path = self.metadata_dir / "tile_split.json"
        self.rgbir_semantics_path = self.metadata_dir / "rgbir_semantics.json"
        self.dataset_protocol_path = self.metadata_dir / "dataset_protocol.json"

        self.labels_manifest = load_json(self.labels_manifest_path)
        self.tile_split_obj = load_json(self.tile_split_path)
        self.rgbir_semantics_obj = load_json(self.rgbir_semantics_path)
        self.protocol = load_json(self.dataset_protocol_path)

        self._validate_frozen_metadata_hashes()
        self.split = parse_split(self.tile_split_obj)
        validate_rgbir_semantics(self.rgbir_semantics_obj)
        self._validate_protocol()

        self.all_tiles = (
            list(self.split["train"])
            + list(self.split["val"])
            + list(self.split["test"])
        )

        self.gt_paths = resolve_canonical_gt_paths(
            self.labels_manifest,
            self.labels_manifest_path,
            self.project_root,
            self.all_tiles,
        )
        self.rgbir_root = locate_rgbir_root(self.project_root, self.all_tiles)
        self.rgbir_paths = build_rgbir_map(self.rgbir_root, self.all_tiles)

        self.data_seed = int(self.protocol["training_sampling"]["data_seed"])
        self.tile_size = int(self.protocol["validation_test_inference"]["tile_size"][0])
        self.crop_size = int(self.protocol["training_sampling"]["crop_size"][0])
        self.crops_per_tile = int(
            self.protocol["training_sampling"]["crops_per_tile_per_epoch"]
        )
        self.cat_max_ratio = float(
            self.protocol["training_sampling"]["cat_max_ratio"]
        )
        self.max_candidates = int(
            self.protocol["training_sampling"]["max_candidates"]
        )
        self.ignore_index = int(
            self.protocol["labels"]["ignore_index_reserved_sentinel"]
        )

        self.rgb_mean = np.asarray(
            self.protocol["normalization"]["rgb"]["mean"], dtype=np.float32
        )
        self.rgb_std = np.asarray(
            self.protocol["normalization"]["rgb"]["std"], dtype=np.float32
        )
        self.nir_mean = float(
            self.protocol["normalization"]["nir"]["mean"]
        )
        self.nir_std = float(
            self.protocol["normalization"]["nir"]["std"]
        )

        self.sliding_stride = int(
            self.protocol["validation_test_inference"]["stride"][0]
        )
        self.sliding_starts = [
            int(x)
            for x in self.protocol["validation_test_inference"]["starts"]
        ]
        self.window_coordinates = build_window_coordinates(self.sliding_starts)

        self.class_specs = self.protocol["labels"]["classes"]
        self._rgb_code_to_class = self._build_rgb_code_map()

    def _validate_frozen_metadata_hashes(self) -> None:
        expected = self.protocol.get("source_metadata_sha256")
        require(isinstance(expected, dict),
                "dataset_protocol.json 缺少 source_metadata_sha256")

        actual = {
            "labels_manifest.json": sha256_file(self.labels_manifest_path),
            "tile_split.json": sha256_file(self.tile_split_path),
            "rgbir_semantics.json": sha256_file(self.rgbir_semantics_path),
        }

        for name, actual_hash in actual.items():
            require(
                expected.get(name) == actual_hash,
                f"{name} SHA256 与 dataset_protocol.json 冻结值不一致。"
                f"\nactual={actual_hash}\nexpected={expected.get(name)}"
            )

    def _validate_protocol(self) -> None:
        p = self.protocol

        require(
            p.get("protocol_version") == PROTOCOL_NAME,
            f"protocol_version 不是 {PROTOCOL_NAME}: {p.get('protocol_version')}"
        )
        require(
            p.get("status") == "FROZEN_AFTER_AUDIT_PASS",
            f"dataset_protocol.json 未处于冻结 PASS 状态：{p.get('status')}"
        )

        protocol_split = p.get("split")
        require(
            protocol_split == self.split,
            "dataset_protocol.json 的 split 与 tile_split.json 不一致"
        )

        labels = p["labels"]
        require(int(labels["num_classes"]) == EXPECTED_NUM_CLASSES,
                "num_classes != 6")
        require(int(labels["ignore_index_reserved_sentinel"]) == EXPECTED_IGNORE_INDEX,
                "ignore index != 255")
        require(labels.get("reduce_labels") is False,
                "reduce_labels 必须为 False")
        require(labels.get("unknown_rgb_policy") == "hard_error",
                "unknown_rgb_policy 必须为 hard_error")
        require(labels.get("classes") == EXPECTED_CLASSES,
                "六类 ID/RGB 映射与冻结协议不一致")

        train = p["training_sampling"]
        require(int(train["data_seed"]) == EXPECTED_DATA_SEED,
                "DATA_SEED 与冻结协议不一致")
        require(train["crop_size"] == [EXPECTED_CROP_SIZE, EXPECTED_CROP_SIZE],
                "train crop_size 与冻结协议不一致")
        require(int(train["crops_per_tile_per_epoch"]) == EXPECTED_CROPS_PER_TILE_PER_EPOCH,
                "crops_per_tile_per_epoch != 64")
        require(int(train["samples_per_epoch"]) == EXPECTED_SAMPLES_PER_EPOCH,
                "samples_per_epoch != 1152")
        require(math.isclose(float(train["cat_max_ratio"]), EXPECTED_CAT_MAX_RATIO),
                "cat_max_ratio != 0.75")
        require(int(train["max_candidates"]) == EXPECTED_MAX_CANDIDATES,
                "max_candidates != 10")

        infer = p["validation_test_inference"]
        require(infer["tile_size"] == [EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE],
                "tile_size != 6000x6000")
        require(infer["crop_size"] == [EXPECTED_CROP_SIZE, EXPECTED_CROP_SIZE],
                "inference crop_size != 512x512")
        require(infer["stride"] == [EXPECTED_SLIDING_STRIDE, EXPECTED_SLIDING_STRIDE],
                "stride != 384")
        require(int(infer["starts_per_axis"]) == 16,
                "starts_per_axis != 16")
        require(int(infer["windows_per_tile"]) == 256,
                "windows_per_tile != 256")
        require(infer["starts"][-1] == 5488,
                "sliding final anchor != 5488")
        require(
            infer["window_coordinates_sha256"] == EXPECTED_WINDOW_COORD_SHA256,
            "sliding-window coordinate SHA256 与冻结 audit 不一致"
        )

        coords = [
            [y, x]
            for y, x in build_window_coordinates(infer["starts"])
        ]
        require(
            sha256_json_canonical(coords) == infer["window_coordinates_sha256"],
            "dataset_protocol.json 中的 sliding starts 无法重现冻结 coordinate SHA256"
        )

        nir = p["normalization"]["nir"]
        require(int(nir["pixel_count"]) == 648000000,
                "NIR normalization pixel_count != 648000000")
        require(float(nir["std"]) > 0,
                "NIR std 必须 > 0")

    def _build_rgb_code_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for spec in self.class_specs:
            cid = int(spec["id"])
            r, g, b = [int(v) for v in spec["rgb"]]
            code = (r << 16) | (g << 8) | b
            require(code not in mapping, "GT RGB mapping 中存在重复颜色")
            mapping[code] = cid
        require(set(mapping.values()) == set(range(6)),
                "GT class IDs 必须恰好是 0..5")
        return mapping

    def gt_rgb_to_class_index(
        self,
        gt_arr: np.ndarray,
        context: str,
    ) -> np.ndarray:
        require(gt_arr.ndim == 3 and gt_arr.shape[2] in (3, 4),
                f"{context}: GT shape 应为 HxWx3/4，实际 {gt_arr.shape}")
        require(gt_arr.dtype == np.uint8,
                f"{context}: GT dtype 必须为 uint8，实际 {gt_arr.dtype}")

        if gt_arr.shape[2] == 4:
            alpha = gt_arr[..., 3]
            require(np.all(alpha == 255),
                    f"{context}: GT 含非全 255 alpha，拒绝 overlay")
            rgb = gt_arr[..., :3]
        else:
            rgb = gt_arr

        code = (
            (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32)
        )

        labels = np.full(code.shape, self.ignore_index, dtype=np.uint8)
        for rgb_code, cid in self._rgb_code_to_class.items():
            labels[code == rgb_code] = cid

        unknown = labels == self.ignore_index
        if np.any(unknown):
            unknown_codes, counts = np.unique(code[unknown], return_counts=True)
            order = np.argsort(counts)[::-1][:10]
            examples = []
            for i in order:
                c = int(unknown_codes[i])
                examples.append({
                    "rgb": [(c >> 16) & 255, (c >> 8) & 255, c & 255],
                    "count": int(counts[i]),
                })
            raise DatasetProtocolError(
                f"{context}: canonical GT 出现 unknown RGB，"
                f"count={int(unknown.sum())}, examples={examples}"
            )

        return labels

    def normalize_rgb_nir(
        self,
        rgbir_crop: np.ndarray,
        context: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        require(
            rgbir_crop.ndim == 3 and rgbir_crop.shape[2] == 4,
            f"{context}: RGBIR crop 应为 HxWx4，实际 {rgbir_crop.shape}"
        )
        require(
            rgbir_crop.dtype == np.uint8,
            f"{context}: RGBIR dtype 必须为 uint8，实际 {rgbir_crop.dtype}"
        )

        rgb = rgbir_crop[..., 0:3].astype(np.float32) / 255.0
        nir = rgbir_crop[..., EXPECTED_NIR_CHANNEL].astype(np.float32) / 255.0

        rgb = (rgb - self.rgb_mean.reshape(1, 1, 3)) / self.rgb_std.reshape(1, 1, 3)
        nir = (nir - self.nir_mean) / self.nir_std

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        nir_t = torch.from_numpy(np.ascontiguousarray(nir[None, ...]))

        require(rgb_t.dtype == torch.float32, "RGB tensor dtype != float32")
        require(nir_t.dtype == torch.float32, "NIR tensor dtype != float32")

        return rgb_t, nir_t


# ---------------------------------------------------------------------
# Dataset base
# ---------------------------------------------------------------------

class _PotsdamDatasetBase(Dataset):
    def __init__(self, project_root: Path | str, split: str):
        super().__init__()
        self.spec = FrozenPotsdamSpec(project_root)
        require(split in ("train", "val", "test"), f"非法 split={split}")
        self.split_name = split
        self.tile_ids = list(self.spec.split[split])

        self._rgbir_store = _TiffStore(max_memmaps=8)
        self._gt_store = _TiffStore(max_memmaps=8)

    def __getstate__(self):
        state = self.__dict__.copy()
        # DataLoader worker pickle/fork 后不继承现有 array cache。
        state["_rgbir_store"] = _TiffStore(max_memmaps=8)
        state["_gt_store"] = _TiffStore(max_memmaps=8)
        return state

    def _read_rgbir(self, tile_id: str) -> np.ndarray:
        arr = self._rgbir_store.read(self.spec.rgbir_paths[tile_id])

        require(arr.ndim == 3, f"{tile_id}: RGBIR TIFF 不是 3D array")
        if arr.shape[0] == 4 and arr.shape[-1] != 4:
            arr = np.moveaxis(arr, 0, -1)

        require(
            arr.shape == (
                self.spec.tile_size,
                self.spec.tile_size,
                4,
            ),
            f"{tile_id}: RGBIR shape 异常：{arr.shape}"
        )
        require(arr.dtype == np.uint8,
                f"{tile_id}: RGBIR dtype 应为 uint8，实际 {arr.dtype}")
        return arr

    def _read_gt_rgb(self, tile_id: str) -> np.ndarray:
        arr = self._gt_store.read(self.spec.gt_paths[tile_id])

        require(arr.ndim == 3, f"{tile_id}: GT TIFF 不是 3D array")
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

        require(
            arr.shape[0] == self.spec.tile_size
            and arr.shape[1] == self.spec.tile_size
            and arr.shape[2] in (3, 4),
            f"{tile_id}: GT shape 异常：{arr.shape}"
        )
        require(arr.dtype == np.uint8,
                f"{tile_id}: GT dtype 应为 uint8，实际 {arr.dtype}")
        return arr


# ---------------------------------------------------------------------
# Train Dataset
# ---------------------------------------------------------------------

class PotsdamTrainDataset(_PotsdamDatasetBase):
    """
    每个 epoch：
        18 tiles * 64 sample slots = 1152 items

    Dataset index 顺序：
        tile-major
        [tile0 slot0..63, tile1 slot0..63, ...]

    注意：
    Dataset 只固定“index -> crop”的映射。
    未来训练循环如果要 shuffle DataLoader index，必须使用统一 deterministic
    sampler/generator；该训练顺序在本模块中不擅自定义。
    """

    def __init__(self, project_root: Path | str, epoch: int = 0):
        super().__init__(project_root=project_root, split="train")
        self.epoch = 0
        self.set_epoch(epoch)

        require(
            len(self.tile_ids) * self.spec.crops_per_tile
            == EXPECTED_SAMPLES_PER_EPOCH,
            "Train Dataset 长度无法得到 1152"
        )

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        require(epoch >= 0, "epoch 必须 >= 0")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.tile_ids) * self.spec.crops_per_tile

    def index_to_tile_slot(self, index: int) -> Tuple[str, int]:
        if index < 0:
            index += len(self)
        require(0 <= index < len(self), f"Train Dataset index 越界：{index}")

        tile_index = index // self.spec.crops_per_tile
        sample_slot = index % self.spec.crops_per_tile
        return self.tile_ids[tile_index], sample_slot

    def _candidate_label_counts(
        self,
        gt_rgb: np.ndarray,
        tile_id: str,
        x: int,
        y: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        patch_rgb = gt_rgb[
            y:y + self.spec.crop_size,
            x:x + self.spec.crop_size,
        ]
        require(
            patch_rgb.shape[0] == self.spec.crop_size
            and patch_rgb.shape[1] == self.spec.crop_size,
            f"{tile_id}: GT candidate crop 越界"
        )

        labels = self.spec.gt_rgb_to_class_index(
            patch_rgb,
            context=f"{tile_id} GT crop x={x} y={y}",
        )
        counts = np.bincount(labels.ravel(), minlength=6)[:6].astype(np.int64)
        require(
            int(counts.sum()) == self.spec.crop_size * self.spec.crop_size,
            f"{tile_id}: candidate class counts 异常"
        )
        return labels, counts

    def _choose_crop(
        self,
        tile_id: str,
        sample_slot: int,
        gt_rgb: np.ndarray,
    ) -> Dict[str, Any]:
        max_start = self.spec.tile_size - self.spec.crop_size
        candidates: List[Dict[str, Any]] = []

        for candidate_id in range(self.spec.max_candidates):
            x, y = deterministic_candidate_xy(
                self.spec.data_seed,
                self.epoch,
                tile_id,
                sample_slot,
                candidate_id,
                max_start,
            )
            labels, counts = self._candidate_label_counts(
                gt_rgb, tile_id, x, y
            )
            ratio = float(counts.max() / counts.sum())

            row = {
                "candidate_id": candidate_id,
                "x": x,
                "y": y,
                "labels": labels,
                "class_counts": counts,
                "dominant_ratio": ratio,
            }
            candidates.append(row)

            if ratio <= self.spec.cat_max_ratio:
                return {
                    **row,
                    "fallback": False,
                    "rejected_candidates": candidate_id,
                }

        best_index = int(np.argmin(
            np.asarray(
                [c["dominant_ratio"] for c in candidates],
                dtype=np.float64,
            )
        ))
        chosen = candidates[best_index]
        return {
            **chosen,
            "fallback": True,
            "rejected_candidates": self.spec.max_candidates,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id, sample_slot = self.index_to_tile_slot(index)

        gt_rgb = self._read_gt_rgb(tile_id)
        chosen = self._choose_crop(tile_id, sample_slot, gt_rgb)

        x = int(chosen["x"])
        y = int(chosen["y"])
        d4_code = deterministic_d4_code(
            self.spec.data_seed,
            self.epoch,
            tile_id,
            sample_slot,
        )

        rgbir = self._read_rgbir(tile_id)
        rgbir_crop = rgbir[
            y:y + self.spec.crop_size,
            x:x + self.spec.crop_size,
            :,
        ]
        require(
            rgbir_crop.shape == (
                self.spec.crop_size,
                self.spec.crop_size,
                4,
            ),
            f"{tile_id}: RGBIR crop shape 异常：{rgbir_crop.shape}"
        )

        labels = chosen["labels"]

        rgbir_crop = apply_d4(rgbir_crop, d4_code)
        labels = apply_d4(labels, d4_code)

        rgb_t, nir_t = self.spec.normalize_rgb_nir(
            rgbir_crop,
            context=f"train {tile_id} epoch={self.epoch} slot={sample_slot}",
        )
        label_t = torch.from_numpy(
            np.ascontiguousarray(labels.astype(np.int64, copy=False))
        )

        require(rgb_t.shape == (3, self.spec.crop_size, self.spec.crop_size),
                "Train RGB tensor shape 异常")
        require(nir_t.shape == (1, self.spec.crop_size, self.spec.crop_size),
                "Train NIR tensor shape 异常")
        require(label_t.shape == (self.spec.crop_size, self.spec.crop_size),
                "Train label tensor shape 异常")
        require(label_t.dtype == torch.int64,
                "Train label dtype 必须为 torch.int64")
        require(
            int(label_t.min()) >= 0 and int(label_t.max()) <= 5,
            "Train label 必须全部属于 0..5"
        )

        return {
            "rgb": rgb_t,
            "nir": nir_t,
            "labels": label_t,
            "tile_id": tile_id,
            "epoch": int(self.epoch),
            "sample_slot": int(sample_slot),
            "x": x,
            "y": y,
            "d4_code": int(d4_code),
            "candidate_id": int(chosen["candidate_id"]),
            "rejected_candidates": int(chosen["rejected_candidates"]),
            "fallback": bool(chosen["fallback"]),
            "dominant_ratio": float(chosen["dominant_ratio"]),
            "class_counts": torch.from_numpy(
                chosen["class_counts"].astype(np.int64, copy=True)
            ),
        }


# ---------------------------------------------------------------------
# Val/Test sliding-window Dataset
# ---------------------------------------------------------------------

class PotsdamSlidingWindowDataset(_PotsdamDatasetBase):
    """
    split 必须是 val 或 test。

    __getitem__ 只返回 normalized RGB/NIR + tile/window coordinates。
    不返回 patch GT。

    原因：
    protocol 的主指标要求：
        window logits -> full-tile mean-logit fusion
        -> full-tile prediction
        -> global confusion matrix
    因此 GT 必须在整 tile prediction 完成后通过 load_full_label() 读取。
    """

    def __init__(self, project_root: Path | str, split: str):
        require(split in ("val", "test"),
                "PotsdamSlidingWindowDataset split 只能是 val/test")
        super().__init__(project_root=project_root, split=split)

        self.window_coordinates = list(self.spec.window_coordinates)
        require(len(self.window_coordinates) == 256,
                "window_coordinates 数量必须为 256")

    def __len__(self) -> int:
        return len(self.tile_ids) * len(self.window_coordinates)

    def index_to_tile_window(
        self,
        index: int,
    ) -> Tuple[str, int, int, int]:
        if index < 0:
            index += len(self)
        require(0 <= index < len(self),
                f"{self.split_name} Dataset index 越界：{index}")

        windows_per_tile = len(self.window_coordinates)
        tile_index = index // windows_per_tile
        window_index = index % windows_per_tile
        y, x = self.window_coordinates[window_index]

        return self.tile_ids[tile_index], window_index, x, y

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id, window_index, x, y = self.index_to_tile_window(index)

        rgbir = self._read_rgbir(tile_id)
        crop = rgbir[
            y:y + self.spec.crop_size,
            x:x + self.spec.crop_size,
            :,
        ]
        require(
            crop.shape == (
                self.spec.crop_size,
                self.spec.crop_size,
                4,
            ),
            f"{tile_id}: sliding crop shape 异常：{crop.shape}"
        )

        rgb_t, nir_t = self.spec.normalize_rgb_nir(
            crop,
            context=(
                f"{self.split_name} {tile_id} "
                f"window={window_index} x={x} y={y}"
            ),
        )

        return {
            "rgb": rgb_t,
            "nir": nir_t,
            "tile_id": tile_id,
            "tile_index": int(self.tile_ids.index(tile_id)),
            "window_index": int(window_index),
            "x": int(x),
            "y": int(y),
            "height": int(self.spec.crop_size),
            "width": int(self.spec.crop_size),
        }

    def load_full_label(self, tile_id: str) -> torch.Tensor:
        """
        返回完整 6000x6000 class-index GT，dtype=torch.int64。

        仅用于 full-tile logits 融合完成后的指标计算。
        """
        require(tile_id in self.tile_ids,
                f"{tile_id} 不属于 {self.split_name} split")

        gt_rgb = self._read_gt_rgb(tile_id)
        labels = self.spec.gt_rgb_to_class_index(
            gt_rgb,
            context=f"{self.split_name} full GT {tile_id}",
        )

        require(
            labels.shape == (self.spec.tile_size, self.spec.tile_size),
            f"{tile_id}: full label shape 异常"
        )
        return torch.from_numpy(
            np.ascontiguousarray(labels.astype(np.int64, copy=False))
        )

    def get_tile_window_coordinates(
        self,
        tile_id: str,
    ) -> List[Tuple[int, int]]:
        require(tile_id in self.tile_ids,
                f"{tile_id} 不属于 {self.split_name} split")
        return list(self.window_coordinates)


# ---------------------------------------------------------------------
# convenience factory
# ---------------------------------------------------------------------

def build_potsdam_dataset(
    project_root: Path | str,
    split: str,
    epoch: int = 0,
) -> Dataset:
    if split == "train":
        return PotsdamTrainDataset(project_root=project_root, epoch=epoch)
    if split in ("val", "test"):
        return PotsdamSlidingWindowDataset(
            project_root=project_root,
            split=split,
        )
    raise DatasetProtocolError(f"非法 split={split}")


# ---------------------------------------------------------------------
# 内置 smoke-test（无训练）
# ---------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _tensor_fingerprint(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def run_smoke_test(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()

    print("=" * 72)
    print("Potsdam Dataset 实现 smoke-test")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("本测试不会启动模型或训练。")
    print()

    print("[1/5] 加载四份冻结 metadata...")
    spec = FrozenPotsdamSpec(project_root)
    print("  PASS: metadata / hash / split / channel / protocol 一致。")
    print(f"  RGBIR root: {spec.rgbir_root}")
    print(
        f"  frozen NIR mean/std = "
        f"{spec.nir_mean:.12f} / {spec.nir_std:.12f}"
    )

    print("[2/5] Train Dataset 实例化与真实样本读取...")
    train = PotsdamTrainDataset(project_root, epoch=0)
    require(len(train) == 1152, f"train len={len(train)} != 1152")

    probe_indices = [0, 63, 64, len(train) - 1]
    train_rows = []
    for idx in probe_indices:
        item = train[idx]
        require(item["rgb"].shape == (3, 512, 512), "RGB shape mismatch")
        require(item["nir"].shape == (1, 512, 512), "NIR shape mismatch")
        require(item["labels"].shape == (512, 512), "label shape mismatch")
        require(item["rgb"].dtype == torch.float32, "RGB dtype mismatch")
        require(item["nir"].dtype == torch.float32, "NIR dtype mismatch")
        require(item["labels"].dtype == torch.int64, "label dtype mismatch")
        require(0.0 <= item["dominant_ratio"] <= 0.75 + 1e-12,
                "非 fallback sample dominant ratio > 0.75")

        train_rows.append({
            "index": idx,
            "tile_id": item["tile_id"],
            "sample_slot": item["sample_slot"],
            "x": item["x"],
            "y": item["y"],
            "d4_code": item["d4_code"],
            "candidate_id": item["candidate_id"],
            "fallback": item["fallback"],
            "dominant_ratio": item["dominant_ratio"],
            "rgb_sha256": _tensor_fingerprint(item["rgb"]),
            "nir_sha256": _tensor_fingerprint(item["nir"]),
            "label_sha256": _tensor_fingerprint(item["labels"]),
        })
        print(
            f"  idx={idx:4d} tile={item['tile_id']} slot={item['sample_slot']:02d} "
            f"x={item['x']:4d} y={item['y']:4d} d4={item['d4_code']} "
            f"ratio={item['dominant_ratio']:.6f}"
        )

    # 同 epoch / 同 index 再读一次，要求 metadata 和 tensor 都完全重现。
    first_a = train[0]
    first_b = train[0]
    for key in ("tile_id", "sample_slot", "x", "y", "d4_code",
                "candidate_id", "fallback", "dominant_ratio"):
        require(first_a[key] == first_b[key],
                f"Train deterministic repeat metadata mismatch: {key}")
    require(torch.equal(first_a["rgb"], first_b["rgb"]),
            "Train deterministic repeat RGB mismatch")
    require(torch.equal(first_a["nir"], first_b["nir"]),
            "Train deterministic repeat NIR mismatch")
    require(torch.equal(first_a["labels"], first_b["labels"]),
            "Train deterministic repeat label mismatch")

    # index mapping 必须保持 audit 的 tile-major 结构。
    require(train.index_to_tile_slot(0) == (spec.split["train"][0], 0),
            "train index 0 mapping mismatch")
    require(train.index_to_tile_slot(63) == (spec.split["train"][0], 63),
            "train index 63 mapping mismatch")
    require(train.index_to_tile_slot(64) == (spec.split["train"][1], 0),
            "train index 64 mapping mismatch")
    print("  PASS: train length / shapes / dtype / deterministic repeat / tile-major mapping。")

    print("[3/5] epoch 切换只改变 deterministic schedule，不改变 Dataset 长度...")
    epoch0_meta = (
        first_a["tile_id"], first_a["sample_slot"],
        first_a["x"], first_a["y"], first_a["d4_code"]
    )
    train.set_epoch(1)
    require(len(train) == 1152, "set_epoch 后 train length 改变")
    epoch1 = train[0]
    epoch1_meta = (
        epoch1["tile_id"], epoch1["sample_slot"],
        epoch1["x"], epoch1["y"], epoch1["d4_code"]
    )
    require(epoch0_meta != epoch1_meta,
            "epoch 0/1 的 index 0 schedule 完全相同，异常")
    train.set_epoch(0)
    first_c = train[0]
    require(
        (first_c["x"], first_c["y"], first_c["d4_code"])
        == (first_a["x"], first_a["y"], first_a["d4_code"]),
        "set_epoch(1)->set_epoch(0) 后无法重现 epoch 0 sample"
    )
    print("  PASS: epoch schedule 可切换且可回到 epoch 0 精确重现。")

    print("[4/5] Val/Test sliding-window Dataset...")
    val = PotsdamSlidingWindowDataset(project_root, split="val")
    test = PotsdamSlidingWindowDataset(project_root, split="test")
    require(len(val) == 6 * 256, f"val len={len(val)} != 1536")
    require(len(test) == 14 * 256, f"test len={len(test)} != 3584")

    require(val.index_to_tile_window(0)[1:] == (0, 0, 0),
            "val first window mismatch")
    first_tile_last = val.index_to_tile_window(255)
    require(first_tile_last[1:] == (255, 5488, 5488),
            f"val first tile last window mismatch: {first_tile_last}")

    val_first = val[0]
    test_first = test[0]
    require(val_first["rgb"].shape == (3, 512, 512), "val RGB shape mismatch")
    require(val_first["nir"].shape == (1, 512, 512), "val NIR shape mismatch")
    require(test_first["rgb"].shape == (3, 512, 512), "test RGB shape mismatch")
    require(test_first["nir"].shape == (1, 512, 512), "test NIR shape mismatch")

    print(
        f"  val len={len(val)}, test len={len(test)}, "
        f"first=(0,0), first-tile-last=(5488,5488)"
    )
    print("  PASS: sliding-window index / shape / edge anchor。")

    print("[5/5] Full-tile GT 接口（只验证一个 Validation tile）...")
    val_tile = val.tile_ids[0]
    full_gt = val.load_full_label(val_tile)
    require(full_gt.shape == (6000, 6000), "full GT shape mismatch")
    require(full_gt.dtype == torch.int64, "full GT dtype mismatch")
    unique_classes = sorted(int(x) for x in torch.unique(full_gt).tolist())
    require(all(0 <= x <= 5 for x in unique_classes),
            f"full GT 出现非法 class: {unique_classes}")
    print(
        f"  tile={val_tile}, shape={tuple(full_gt.shape)}, "
        f"classes={unique_classes}"
    )
    print("  PASS: full-tile GT 只能得到 0..5。")

    report = {
        "status": "PASS",
        "module_version": MODULE_VERSION,
        "protocol_version": spec.protocol["protocol_version"],
        "dataset_protocol_sha256": sha256_file(spec.dataset_protocol_path),
        "split_counts": {
            "train": len(spec.split["train"]),
            "val": len(spec.split["val"]),
            "test": len(spec.split["test"]),
        },
        "normalization": {
            "rgb_mean": spec.rgb_mean.tolist(),
            "rgb_std": spec.rgb_std.tolist(),
            "nir_mean": spec.nir_mean,
            "nir_std": spec.nir_std,
        },
        "train": {
            "length": len(train),
            "epoch0_probe_samples": train_rows,
            "deterministic_repeat": True,
            "epoch_switch_reproducible": True,
            "index_order": "tile-major",
        },
        "validation": {
            "length": len(val),
            "windows_per_tile": 256,
        },
        "test": {
            "length": len(test),
            "windows_per_tile": 256,
        },
        "sliding_window_coordinates_sha256": (
            spec.protocol["validation_test_inference"]["window_coordinates_sha256"]
        ),
        "full_gt_probe": {
            "tile_id": val_tile,
            "shape": list(full_gt.shape),
            "classes": unique_classes,
        },
    }

    print()
    print("FINAL STATUS: PASS")
    return report


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _report_to_text(report: Dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return (
            "Potsdam Dataset smoke-test\n"
            "========================================================================\n"
            f"状态: FAIL\n"
            f"error_type: {report.get('error_type')}\n"
            f"error: {report.get('error')}\n"
        )

    lines = [
        "Potsdam Dataset smoke-test",
        "=" * 72,
        "状态: PASS",
        f"模块版本: {report['module_version']}",
        f"protocol: {report['protocol_version']}",
        f"dataset_protocol_sha256: {report['dataset_protocol_sha256']}",
        "",
        "[Split]",
        f"train/val/test = "
        f"{report['split_counts']['train']} / "
        f"{report['split_counts']['val']} / "
        f"{report['split_counts']['test']}",
        "",
        "[Normalization]",
        f"NIR mean = {report['normalization']['nir_mean']:.12f}",
        f"NIR std = {report['normalization']['nir_std']:.12f}",
        "",
        "[Train]",
        f"length = {report['train']['length']}",
        f"index_order = {report['train']['index_order']}",
        f"deterministic_repeat = {report['train']['deterministic_repeat']}",
        f"epoch_switch_reproducible = {report['train']['epoch_switch_reproducible']}",
        "",
        "[Val/Test]",
        f"val length = {report['validation']['length']}",
        f"test length = {report['test']['length']}",
        f"windows_per_tile = {report['validation']['windows_per_tile']}",
        f"window_coordinates_sha256 = "
        f"{report['sliding_window_coordinates_sha256']}",
        "",
        "[Full GT probe]",
        f"tile = {report['full_gt_probe']['tile_id']}",
        f"shape = {report['full_gt_probe']['shape']}",
        f"classes = {report['full_gt_probe']['classes']}",
        "",
        "FINAL STATUS: PASS",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Potsdam Dataset Protocol v1 正式 Dataset + smoke-test"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录；默认 data_pipeline/ 的父目录。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行真实数据 smoke-test；不启动模型或训练。",
    )
    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "Dataset 模块已加载。若要运行真实数据检查：\n"
            "  python data_pipeline/potsdam_dataset.py --smoke-test"
        )
        return 0

    project_root = args.project_root.resolve()
    out_dir = project_root / "outputs/dataset_check/dataset_implementation"
    out_dir.mkdir(parents=True, exist_ok=True)

    console_path = out_dir / "console.txt"
    json_path = out_dir / "dataset_smoke_test.json"
    txt_path = out_dir / "dataset_smoke_test.txt"

    with console_path.open("w", encoding="utf-8", buffering=1) as console_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, console_file)
        sys.stderr = Tee(old_stderr, console_file)

        try:
            report = run_smoke_test(project_root)
            exit_code = 0
        except Exception as e:
            print()
            print("=" * 72)
            print("FINAL STATUS: FAIL")
            print(f"{type(e).__name__}: {e}")
            print("=" * 72)
            traceback.print_exc()
            report = {
                "status": "FAIL",
                "module_version": MODULE_VERSION,
                "project_root": str(project_root),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            exit_code = 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    _write_json(json_path, report)
    txt_path.write_text(_report_to_text(report), encoding="utf-8")

    print(f"console: {console_path}")
    print(f"json: {json_path}")
    print(f"txt: {txt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
