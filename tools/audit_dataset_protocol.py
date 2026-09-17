#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Protocol v1 独立审计脚本。

用途：
1) 只从冻结元数据读取 GT / split / RGBIR channel 定义；
2) 只使用 Train 18 tiles 的 RGBIR ch3 全像素计算 NIR normalization；
3) 审计 deterministic 512x512 train sampling；
4) 审计 canonical GT RGB -> class index；
5) 审计 RGB/NIR/GT 同坐标、同 D4 空间变换；
6) 审计 Val/Test 6000x6000, crop=512, stride=384 的 deterministic sliding windows；
7) PASS 后冻结 data/processed/potsdam/dataset_protocol.json。

本脚本不包含模型、训练、corruption、4-channel model、dual encoder、fusion 或 Quality Gate。
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile


SCRIPT_VERSION = "1.0.4"
PROTOCOL_VERSION = "Dataset Protocol v1"

DATA_SEED = 20260917
TILE_SIZE = 6000
CROP_SIZE = 512
TRAIN_CROPS_PER_TILE_PER_EPOCH = 64
AUDIT_EPOCHS = (0, 1, 2)
CAT_MAX_RATIO = 0.75
MAX_CANDIDATES = 10
SLIDING_STRIDE = 384
IGNORE_INDEX = 255
NIR_CHANNEL_INDEX = 3
NIR_SCALE = 255.0

# “几乎完全采不到”的保守 hard threshold。
# 这不是 class balancing 目标，只用于发现 sampler / label 映射发生明显异常。
NEAR_MISSING_MIN_CROP_PRESENCE_FRACTION = 0.005   # 0.5%
NEAR_MISSING_MIN_PIXEL_FRACTION = 1e-5            # 0.001%

RGB_IMAGENET_MEAN = [0.485, 0.456, 0.406]
RGB_IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_SPECS = [
    {"id": 0, "name": "Impervious surfaces", "rgb": [255, 255, 255]},
    {"id": 1, "name": "Building",            "rgb": [0, 0, 255]},
    {"id": 2, "name": "Low vegetation",      "rgb": [0, 255, 255]},
    {"id": 3, "name": "Tree",                "rgb": [0, 255, 0]},
    {"id": 4, "name": "Car",                 "rgb": [255, 255, 0]},
    {"id": 5, "name": "Clutter/background",  "rgb": [255, 0, 0]},
]

EXPECTED_TRAIN_TILES = [
    "2_11", "2_12", "3_10", "3_11", "4_12", "5_10", "5_12", "6_7", "6_8",
    "6_9", "6_10", "6_11", "6_12", "7_7", "7_8", "7_9", "7_10", "7_12",
]
EXPECTED_VAL_TILES = ["2_10", "3_12", "4_10", "4_11", "5_11", "7_11"]
EXPECTED_TEST_TILES = [
    "2_13", "2_14", "3_13", "3_14", "4_13", "4_14", "4_15",
    "5_13", "5_14", "5_15", "6_13", "6_14", "6_15", "7_13",
]
EXPECTED_ALL_TILES = EXPECTED_TRAIN_TILES + EXPECTED_VAL_TILES + EXPECTED_TEST_TILES

TILE_ID_RE = re.compile(r"^\d+_\d+$")
RGBIR_FILENAME_RE = re.compile(r"(?:^|_)(\d+_\d+)_RGBIR(?:\.|_)", re.IGNORECASE)


class AuditError(RuntimeError):
    pass


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


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


def relpath_str(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def load_json(path: Path) -> Any:
    require(path.is_file(), f"缺少冻结文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise AuditError(f"JSON 读取失败：{path}\n{e}") from e


# ---------------------------
# split 解析与冻结一致性检查
# ---------------------------

def _is_tile_list_like(value: Any) -> bool:
    """
    判断一个值是否“像 split tile list”。

    重要：tile_split.json 同时可能含有
        counts = {"train": 18, "val": 6, "test": 14}
    和
        splits = {"train": [...], "val": [...], "test": [...]}

    只有后者才允许作为 split 定义；绝不能因为键名相同就把 counts
    误识别成真正的 split。
    """
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
        # tile_split.json 的正式 schema 优先使用根节点/当前节点的 "splits"。
        # 这一步可避免先撞到 counts={"train":18,...}。
        splits_value = obj.get("splits")
        if _looks_like_split_dict(splits_value):
            return splits_value

        # 当前 dict 本身只有在三个值都确实是 tile list（或 list wrapper）时才接受。
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
        # 支持 {"tiles": [...]} 之类包装。
        for key in ("tiles", "tile_ids", "ids", "items"):
            if key in value:
                value = value[key]
                break
    require(isinstance(value, list), f"tile_split 中 {name} 不是 list")
    out = [str(x) for x in value]
    bad = [x for x in out if not TILE_ID_RE.match(x)]
    require(not bad, f"tile_split 中 {name} 含非法 tile id：{bad[:10]}")
    require(len(out) == len(set(out)), f"tile_split 中 {name} 存在重复 tile")
    return out


def parse_and_validate_split(split_obj: Any) -> Dict[str, List[str]]:
    # 正式 tile_split schema：优先根节点 "splits"。
    if (
        isinstance(split_obj, dict)
        and _looks_like_split_dict(split_obj.get("splits"))
    ):
        d = split_obj["splits"]
    else:
        d = _find_split_dict(split_obj)

    require(d is not None, "无法从 tile_split.json 识别 train/val/test tile list 结构")

    lower = {str(k).lower(): k for k in d.keys()}
    train_key = next(lower[k] for k in ("train", "training") if k in lower)
    val_key = next(lower[k] for k in ("val", "valid", "validation") if k in lower)
    test_key = next(lower[k] for k in ("test", "testing") if k in lower)

    train = _normalize_tile_list(d[train_key], "train")
    val = _normalize_tile_list(d[val_key], "val")
    test = _normalize_tile_list(d[test_key], "test")

    require(set(train) == set(EXPECTED_TRAIN_TILES),
            f"Train split 与冻结定义不一致。\n实际：{train}\n期望：{EXPECTED_TRAIN_TILES}")
    require(set(val) == set(EXPECTED_VAL_TILES),
            f"Validation split 与冻结定义不一致。\n实际：{val}\n期望：{EXPECTED_VAL_TILES}")
    require(set(test) == set(EXPECTED_TEST_TILES),
            f"Test split 与冻结定义不一致。\n实际：{test}\n期望：{EXPECTED_TEST_TILES}")

    require(not (set(train) & set(val)), "train/val 存在 overlap")
    require(not (set(train) & set(test)), "train/test 存在 overlap")
    require(not (set(val) & set(test)), "val/test 存在 overlap")
    require(set(train) | set(val) | set(test) == set(EXPECTED_ALL_TILES),
            "train/val/test union 不是冻结的 38 tiles")

    # 采样顺序固定为 protocol 中明确列出的顺序，避免 JSON 键/列表顺序影响未来 schedule。
    return {
        "train": list(EXPECTED_TRAIN_TILES),
        "val": list(EXPECTED_VAL_TILES),
        "test": list(EXPECTED_TEST_TILES),
    }


# ---------------------------
# labels_manifest 解析
# 只从 manifest 选择 canonical path，不扫描 GT 目录。
# ---------------------------

IMAGE_EXTS = (".tif", ".tiff")


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
    hints = []

    def walk(x: Any, path_keys: Tuple[str, ...] = ()) -> None:
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
                walk(v, path_keys + (kl,))
        elif isinstance(x, list):
            for v in x:
                walk(v, path_keys)

    walk(entry)
    if not hints:
        return None
    if len(set(hints)) == 1:
        return hints[0]
    return None


def _collect_image_path_candidates(entry: Any) -> List[Tuple[int, str, Tuple[str, ...]]]:
    source_hint = _canonical_source_hint(entry)
    out: List[Tuple[int, str, Tuple[str, ...]]] = []

    def walk(x: Any, keys: Tuple[str, ...] = ()) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                walk(v, keys + (kl,))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, keys + (str(i),))
        elif isinstance(x, str):
            low = x.lower()
            if not low.endswith(IMAGE_EXTS):
                return

            ctx = "/".join(keys).lower()
            score = 0

            # 明确 canonical / selected 路径优先。
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

            # 候选/原始路径不应压过明确 canonical 路径。
            if "candidate" in ctx:
                score -= 30

            out.append((score, x, keys))

    walk(entry)
    return out


def _resolve_manifest_path(raw: str, project_root: Path, manifest_path: Path) -> Optional[Path]:
    """
    将 labels_manifest.json 中记录的 GT 路径解析为真实文件。

    重要约束：
    - 这里只“解析 manifest 已明确给出的路径”；
    - 不扫描 5_Labels* 目录；
    - 不在多个 GT 中重新选择 canonical GT。

    当前项目的 manifest / split 中可能保存形如
        _expanded/5_Labels_for_participants/.../top_potsdam_2_11_label.tif
    的相对路径。该路径通常是相对于 Potsdam raw 根目录
        data/raw/potsdam
    而不是相对于项目根目录或 processed metadata 目录。
    """
    p = Path(raw)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        potsdam_raw_root = project_root / "data/raw/potsdam"
        potsdam_dataset_root = potsdam_raw_root / "Potsdam"
        candidates.extend([
            potsdam_raw_root / p,           # labels_manifest 的 gt_relpath（如 _expanded/...）优先
            potsdam_dataset_root / p,       # 兼容相对于 data/raw/potsdam/Potsdam 的路径
            project_root / p,               # 兼容 project-relative manifest
            manifest_path.parent / p,       # 兼容 metadata-relative manifest
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
    manifest_obj: Any, manifest_path: Path, project_root: Path
) -> Dict[str, Path]:
    entries = collect_manifest_entries(manifest_obj)
    missing_entries = [t for t in EXPECTED_ALL_TILES if t not in entries]
    require(not missing_entries,
            f"labels_manifest.json 缺少 tile entries：{missing_entries}")

    resolved: Dict[str, Path] = {}
    for tile in EXPECTED_ALL_TILES:
        all_candidates: List[Tuple[int, str, Tuple[str, ...], Path]] = []
        for entry in entries[tile]:
            for score, raw, keys in _collect_image_path_candidates(entry):
                p = _resolve_manifest_path(raw, project_root, manifest_path)
                if p is not None:
                    all_candidates.append((score, raw, keys, p))

        if not all_candidates:
            raw_candidates = []
            for entry in entries[tile]:
                for score, raw, keys in _collect_image_path_candidates(entry):
                    raw_candidates.append({
                        "score": score,
                        "raw": raw,
                        "context": "/".join(keys),
                    })

            diagnostic = (
                f"{tile}: manifest 中找不到可解析到真实文件的 canonical GT 路径。"
                "脚本不会扫描 5_Labels* 目录替你重新选择 GT。"
            )
            if raw_candidates:
                diagnostic += (
                    "\nmanifest 中检测到的 TIFF 路径候选：\n"
                    + "\n".join(
                        f"  score={x['score']} raw={x['raw']} context={x['context']}"
                        for x in raw_candidates[:20]
                    )
                    + f"\n尝试的 raw 根目录 1：{project_root / 'data/raw/potsdam'}"
                    + f"\n尝试的 raw 根目录 2：{project_root / 'data/raw/potsdam/Potsdam'}"
                )

                attempted = []
                for x in raw_candidates[:20]:
                    raw_p = Path(x["raw"])
                    if raw_p.is_absolute():
                        attempted.append(raw_p)
                    else:
                        attempted.extend([
                            project_root / "data/raw/potsdam" / raw_p,
                            project_root / "data/raw/potsdam/Potsdam" / raw_p,
                            project_root / raw_p,
                            manifest_path.parent / raw_p,
                            manifest_path.parent.parent / raw_p,
                        ])

                uniq = []
                seen = set()
                for ap in attempted:
                    key = str(ap)
                    if key not in seen:
                        seen.add(key)
                        uniq.append(ap)

                diagnostic += "\n实际尝试路径：\n" + "\n".join(
                    f"  exists={ap.is_file()}  {ap}" for ap in uniq[:40]
                )
            else:
                diagnostic += "\n该 tile entry 中没有检测到任何 .tif/.tiff 路径字段。"

            raise AuditError(diagnostic)

        best_score = max(x[0] for x in all_candidates)
        best = [x for x in all_candidates if x[0] == best_score]

        # 去掉同一路径重复出现。
        unique_best: Dict[str, Tuple[int, str, Tuple[str, ...], Path]] = {}
        for item in best:
            unique_best[str(item[3].resolve())] = item
        best = list(unique_best.values())

        require(len(best) == 1,
                f"{tile}: manifest 中存在多个同优先级 GT 路径，拒绝猜测 canonical GT：\n" +
                "\n".join(f"  score={x[0]} path={x[3]}" for x in best))

        resolved[tile] = best[0][3]

    require(len({str(p) for p in resolved.values()}) == 38,
            "canonical GT 路径不是 38 个唯一文件")
    return resolved


# ---------------------------
# RGBIR semantics 解析
# ---------------------------

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
            # 形式 1: {"0":"R", "1":"G", ...} / {"ch0":"R", ...}
            idx_to_name = {}
            for k, v in x.items():
                idx = _parse_index_key(k)
                name = _canon_channel_name(v)
                if idx is not None and name is not None:
                    idx_to_name[idx] = name
            if set(idx_to_name) == {0, 1, 2, 3}:
                add([idx_to_name[i] for i in range(4)])

            # 形式 2: {"R":0, "G":1, "B":2, "NIR":3}
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

            # 形式 3: {"index": 0, "name":"R"} 由 list 上层处理也可；
            # 这里继续递归。
            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            if len(x) == 4 and all(isinstance(v, str) for v in x):
                add(x)

            # [{"index":0, "name":"R"}, ...]
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


def validate_rgbir_semantics(obj: Any) -> Dict[str, Any]:
    candidates = collect_channel_mapping_candidates(obj)
    expected = ["R", "G", "B", "NIR"]
    require(candidates,
            "无法从 rgbir_semantics.json 识别 4-channel mapping；"
            "请不要让脚本猜测 channel semantics。")
    require(expected in candidates,
            f"rgbir_semantics.json 未确认冻结映射 [R,G,B,NIR]。识别到：{candidates}")
    return {
        "expected": expected,
        "recognized_candidates": candidates,
        "nir_channel_index": NIR_CHANNEL_INDEX,
    }


# ---------------------------
# RGBIR 文件只允许来自 4_Ortho_RGBIR
# ---------------------------

def _direct_rgbir_tiles_in_dir(root: Path) -> Dict[str, Path]:
    """
    只检查 root 的“直接子文件”，不递归。
    这样可以区分：
        .../_expanded/4_Ortho_RGBIR/                 # 外层容器
        .../_expanded/4_Ortho_RGBIR/4_Ortho_RGBIR/  # 实际 TIFF 数据目录

    返回该目录中能识别出的 tile_id -> TIFF。
    若同一 tile 在同一目录出现多个 TIFF，则直接报错。
    """
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
    require(
        not duplicates,
        f"RGBIR 目录中存在同 tile 多文件：{root}\n"
        + "\n".join(
            f"  {tile}: {[str(p) for p in paths]}"
            for tile, paths in sorted(duplicates.items())
        )
    )

    return {tile: paths[0] for tile, paths in tile_to_paths.items()}


def locate_rgbir_root(project_root: Path) -> Path:
    """
    定位正式 RGBIR 数据目录。

    规则：
    1) 只考虑目录名恰好为 4_Ortho_RGBIR 的候选；
    2) 不把外层容器和内层同名数据目录混为一谈；
    3) 候选目录必须“直接包含”冻结的 38 个 RGBIR tiles，且不得有缺失/额外 tile；
    4) 必须恰好有一个候选通过，否则 hard error。

    这不是在多个影像源之间重新选择数据；
    所有候选都属于同一正式 source：4_Ortho_RGBIR，只是在解析 ZIP 解压后的目录层级。
    """
    search_root = project_root / "data/raw/potsdam"
    require(search_root.is_dir(), f"缺少 data/raw/potsdam：{search_root}")

    candidates = [p.resolve() for p in search_root.rglob("4_Ortho_RGBIR") if p.is_dir()]
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))

    require(
        candidates,
        f"未找到任何名为 4_Ortho_RGBIR 的目录：{search_root}"
    )

    expected = set(EXPECTED_ALL_TILES)
    valid: List[Tuple[Path, Dict[str, Path]]] = []
    diagnostics = []

    for root in candidates:
        direct_map = _direct_rgbir_tiles_in_dir(root)
        found = set(direct_map.keys())
        missing = sorted(expected - found)
        extra = sorted(found - expected)

        diagnostics.append({
            "root": str(root),
            "direct_rgbir_tiff_count": len(direct_map),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing": missing,
            "extra": extra,
        })

        if found == expected and len(direct_map) == len(EXPECTED_ALL_TILES):
            valid.append((root, direct_map))

    if len(valid) != 1:
        msg = [
            "无法唯一确定实际 4_Ortho_RGBIR TIFF 数据目录。",
            "要求：目录直接包含且只包含冻结 38 tiles 对应的 RGBIR TIFF。",
            "候选检查结果：",
        ]
        for d in diagnostics:
            msg.append(
                f"  root={d['root']} "
                f"direct_tiffs={d['direct_rgbir_tiff_count']} "
                f"missing={d['missing_count']} extra={d['extra_count']}"
            )
            if d["missing"]:
                msg.append(f"    missing={d['missing']}")
            if d["extra"]:
                msg.append(f"    extra={d['extra']}")
        raise AuditError("\n".join(msg))

    return valid[0][0]


def build_rgbir_map(rgbir_root: Path) -> Dict[str, Path]:
    direct_map = _direct_rgbir_tiles_in_dir(rgbir_root)

    found = set(direct_map.keys())
    expected = set(EXPECTED_ALL_TILES)
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    require(
        not missing and not extra and len(direct_map) == 38,
        f"4_Ortho_RGBIR 实际数据目录 tile 集不等于冻结 38 tiles。"
        f"\nroot={rgbir_root}"
        f"\nmissing={missing}"
        f"\nextra={extra}"
        f"\ncount={len(direct_map)}"
    )

    out = {}
    for tile in EXPECTED_ALL_TILES:
        out[tile] = direct_map[tile]
        require(
            "4_Ortho_RGBIR" in out[tile].parts,
            f"{tile}: RGBIR 路径不在 4_Ortho_RGBIR：{out[tile]}"
        )

    return out


# ---------------------------
# TIFF 读取 / GT 转换
# ---------------------------

def read_tiff_hwc(path: Path, expected_channels: Optional[int] = None) -> np.ndarray:
    try:
        arr = tifffile.imread(str(path))
    except Exception as e:
        raise AuditError(f"tifffile 读取失败：{path}\n{e}") from e

    require(arr.ndim == 3, f"{path}: TIFF 应为 3D array，实际 shape={arr.shape}")

    # 支持极少数 planar TIFF；Potsdam 正常应直接是 HWC。
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    require(arr.shape[0] == TILE_SIZE and arr.shape[1] == TILE_SIZE,
            f"{path}: 期望 spatial shape=({TILE_SIZE},{TILE_SIZE})，实际={arr.shape[:2]}")

    if expected_channels is not None:
        require(arr.shape[2] == expected_channels,
                f"{path}: 期望 {expected_channels} channels，实际 shape={arr.shape}")

    return arr


def gt_rgb_to_class_index(gt_arr: np.ndarray, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    require(gt_arr.dtype == np.uint8,
            f"{path}: canonical GT 必须为 uint8，实际 dtype={gt_arr.dtype}")
    require(gt_arr.shape[2] in (3, 4),
            f"{path}: canonical GT 期望 RGB/RGBA，实际 shape={gt_arr.shape}")

    if gt_arr.shape[2] == 4:
        alpha = gt_arr[..., 3]
        # 允许完全不透明 alpha，但不允许 RGB alpha overlay 悄悄通过。
        require(np.all(alpha == 255),
                f"{path}: GT 含非全 255 alpha，疑似 overlay；canonical GT 审计拒绝继续。")

    rgb = gt_arr[..., :3]
    code = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )

    labels = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    expected_codes = {}
    for spec in CLASS_SPECS:
        r, g, b = spec["rgb"]
        c = (r << 16) | (g << 8) | b
        expected_codes[c] = spec["id"]
        labels[code == c] = spec["id"]

    unknown_mask = labels == IGNORE_INDEX
    unknown_count = int(unknown_mask.sum())
    unknown_examples = []
    if unknown_count:
        unknown_codes, counts = np.unique(code[unknown_mask], return_counts=True)
        order = np.argsort(counts)[::-1][:10]
        for i in order:
            c = int(unknown_codes[i])
            unknown_examples.append({
                "rgb": [(c >> 16) & 255, (c >> 8) & 255, c & 255],
                "count": int(counts[i]),
            })

    require(unknown_count == 0,
            f"{path}: canonical GT 出现 unknown RGB，共 {unknown_count} pixels；"
            f"示例：{unknown_examples}")

    class_counts = np.bincount(labels.ravel(), minlength=6)[:6].astype(np.int64)
    ignore_count = int(np.count_nonzero(labels == IGNORE_INDEX))
    require(ignore_count == 0,
            f"{path}: RGB->class 后 ignore pixels 应为 0，实际 {ignore_count}")

    return labels, {
        "unknown_pixels": 0,
        "ignore_pixels": 0,
        "class_pixel_counts": [int(x) for x in class_counts],
        "rgba_input": bool(gt_arr.shape[2] == 4),
    }


# ---------------------------
# Stateless deterministic RNG
# ---------------------------

def _key_bytes(parts: Sequence[Any], stream: str, counter: int) -> bytes:
    text = "|".join(str(x) for x in parts) + f"|{stream}|{counter}"
    return text.encode("utf-8")


def stateless_randbelow(parts: Sequence[Any], n: int, stream: str) -> int:
    """
    SHA256-based deterministic rejection sampler.
    对 [0, n) 做无 modulo bias 的 stateless uniform sampling。
    """
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
    seed: int, epoch: int, tile_id: str, sample_slot: int, candidate_id: int,
    max_start: int,
) -> Tuple[int, int]:
    parts = (seed, epoch, tile_id, sample_slot, candidate_id)
    x = stateless_randbelow(parts, max_start + 1, "x")
    y = stateless_randbelow(parts, max_start + 1, "y")
    # 自检：同 key 必须精确重现。
    require(x == stateless_randbelow(parts, max_start + 1, "x"),
            "stateless RNG x 自检失败")
    require(y == stateless_randbelow(parts, max_start + 1, "y"),
            "stateless RNG y 自检失败")
    return x, y


def deterministic_d4_code(seed: int, epoch: int, tile_id: str, sample_slot: int) -> int:
    parts = (seed, epoch, tile_id, sample_slot)
    code = stateless_randbelow(parts, 8, "d4")
    require(code == stateless_randbelow(parts, 8, "d4"),
            "stateless RNG D4 自检失败")
    return code


# ---------------------------
# D4
# 0..3: rot90 k=0..3
# 4..7: 先 rot90 k=(code-4)，再左右镜像
# ---------------------------

def apply_d4(arr: np.ndarray, code: int) -> np.ndarray:
    require(0 <= code <= 7, f"非法 D4 code={code}")
    if code < 4:
        out = np.rot90(arr, k=code, axes=(0, 1))
    else:
        out = np.fliplr(np.rot90(arr, k=code - 4, axes=(0, 1)))
    return np.ascontiguousarray(out)


def build_d4_inverse_table() -> Dict[int, int]:
    marker = np.arange(25, dtype=np.int32).reshape(5, 5)
    inv = {}
    for a in range(8):
        transformed = apply_d4(marker, a)
        matches = [b for b in range(8) if np.array_equal(apply_d4(transformed, b), marker)]
        require(len(matches) == 1, f"D4 inverse 构造失败：code={a}, matches={matches}")
        inv[a] = matches[0]
    return inv


D4_INVERSE = build_d4_inverse_table()


def audit_real_crop_alignment(
    rgbir: np.ndarray,
    gt_idx: np.ndarray,
    x: int,
    y: int,
    d4_code: int,
    tile_id: str,
) -> Dict[str, Any]:
    rgb = rgbir[y:y + CROP_SIZE, x:x + CROP_SIZE, 0:3]
    nir = rgbir[y:y + CROP_SIZE, x:x + CROP_SIZE, NIR_CHANNEL_INDEX]
    gt = gt_idx[y:y + CROP_SIZE, x:x + CROP_SIZE]

    require(rgb.shape == (CROP_SIZE, CROP_SIZE, 3), f"{tile_id}: RGB crop shape 异常")
    require(nir.shape == (CROP_SIZE, CROP_SIZE), f"{tile_id}: NIR crop shape 异常")
    require(gt.shape == (CROP_SIZE, CROP_SIZE), f"{tile_id}: GT crop shape 异常")

    rgb_t = apply_d4(rgb, d4_code)
    nir_t = apply_d4(nir, d4_code)
    gt_t = apply_d4(gt, d4_code)

    # 对真实 crop 做 joint transform，再与三个 modality 分别 transform 的结果逐像素比较。
    joint = np.concatenate(
        [rgb, nir[..., None], gt[..., None]], axis=2
    )
    joint_t = apply_d4(joint, d4_code)

    require(np.array_equal(joint_t[..., 0:3], rgb_t),
            f"{tile_id}: RGB D4 与 joint transform 不一致")
    require(np.array_equal(joint_t[..., 3], nir_t),
            f"{tile_id}: NIR D4 与 joint transform 不一致")
    require(np.array_equal(joint_t[..., 4], gt_t),
            f"{tile_id}: GT D4 与 joint transform 不一致")

    inv = D4_INVERSE[d4_code]
    require(np.array_equal(apply_d4(rgb_t, inv), rgb),
            f"{tile_id}: RGB D4 inverse 失败")
    require(np.array_equal(apply_d4(nir_t, inv), nir),
            f"{tile_id}: NIR D4 inverse 失败")
    require(np.array_equal(apply_d4(gt_t, inv), gt),
            f"{tile_id}: GT D4 inverse 失败")

    sample_hash = sha256_json_canonical({
        "tile": tile_id,
        "x": x,
        "y": y,
        "d4": d4_code,
        "rgb_sha256": hashlib.sha256(rgb_t.tobytes()).hexdigest(),
        "nir_sha256": hashlib.sha256(nir_t.tobytes()).hexdigest(),
        "gt_sha256": hashlib.sha256(gt_t.tobytes()).hexdigest(),
    })

    return {
        "tile_id": tile_id,
        "x": x,
        "y": y,
        "d4_code": d4_code,
        "inverse_d4_code": inv,
        "status": "PASS",
        "sample_fingerprint_sha256": sample_hash,
    }


# ---------------------------
# train sampling
# ---------------------------

def patch_class_counts(gt_idx: np.ndarray, x: int, y: int) -> np.ndarray:
    patch = gt_idx[y:y + CROP_SIZE, x:x + CROP_SIZE]
    require(patch.shape == (CROP_SIZE, CROP_SIZE), "train crop 越界")
    counts = np.bincount(patch.ravel(), minlength=6)[:6].astype(np.int64)
    require(int(counts.sum()) == CROP_SIZE * CROP_SIZE, "crop class count 不等于 patch pixels")
    return counts


def choose_training_crop(
    gt_idx: np.ndarray,
    seed: int,
    epoch: int,
    tile_id: str,
    sample_slot: int,
) -> Dict[str, Any]:
    max_start = TILE_SIZE - CROP_SIZE
    candidates = []

    for candidate_id in range(MAX_CANDIDATES):
        x, y = deterministic_candidate_xy(
            seed, epoch, tile_id, sample_slot, candidate_id, max_start
        )
        counts = patch_class_counts(gt_idx, x, y)
        ratio = float(counts.max() / counts.sum())
        candidates.append({
            "candidate_id": candidate_id,
            "x": x,
            "y": y,
            "counts": counts,
            "dominant_ratio": ratio,
        })
        if ratio <= CAT_MAX_RATIO:
            return {
                **candidates[-1],
                "fallback": False,
                "rejected_candidates": candidate_id,
                "d4_code": deterministic_d4_code(seed, epoch, tile_id, sample_slot),
            }

    ratios = [c["dominant_ratio"] for c in candidates]
    best_idx = int(np.argmin(np.asarray(ratios, dtype=np.float64)))
    chosen = candidates[best_idx]
    require(abs(chosen["dominant_ratio"] - min(ratios)) < 1e-15,
            "fallback 未选择 10 个 candidate 中 dominant ratio 最小者")

    return {
        **chosen,
        "fallback": True,
        "rejected_candidates": MAX_CANDIDATES,
        "d4_code": deterministic_d4_code(seed, epoch, tile_id, sample_slot),
    }


def summarize_ratios(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(a.size > 0, "ratio list 为空")
    qs = np.quantile(a, [0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    return {
        "min": float(a.min()),
        "q25": float(qs[0]),
        "median": float(qs[1]),
        "q75": float(qs[2]),
        "q90": float(qs[3]),
        "q95": float(qs[4]),
        "q99": float(qs[5]),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


# ---------------------------
# NIR statistics
# ---------------------------

def accumulate_nir_stats(rgbir: np.ndarray, state: Dict[str, Any]) -> None:
    require(rgbir.dtype == np.uint8,
            f"RGBIR 必须为 uint8，实际 dtype={rgbir.dtype}")
    require(rgbir.shape == (TILE_SIZE, TILE_SIZE, 4),
            f"RGBIR shape 必须为 (6000,6000,4)，实际={rgbir.shape}")

    nir = rgbir[..., NIR_CHANNEL_INDEX]

    # 分块 float64，避免为整个 6000x6000 NIR 一次性创建 float64 副本。
    row_chunk = 512
    for y0 in range(0, TILE_SIZE, row_chunk):
        block = nir[y0:min(y0 + row_chunk, TILE_SIZE)].astype(np.float64)
        block /= NIR_SCALE
        state["count"] += int(block.size)
        state["sum"] += float(block.sum(dtype=np.float64))
        state["sumsq"] += float(np.einsum("ij,ij->", block, block, dtype=np.float64))


def finalize_nir_stats(state: Dict[str, Any]) -> Dict[str, Any]:
    n = int(state["count"])
    require(n > 0, "NIR pixel count 为 0")
    mean = float(state["sum"] / n)
    variance = float(state["sumsq"] / n - mean * mean)
    # 只容忍极小的浮点负误差。
    require(variance >= -1e-15, f"NIR variance 出现异常负值：{variance}")
    variance = max(variance, 0.0)
    std = float(math.sqrt(variance))
    require(std > 0.0, "NIR std 必须 > 0")
    return {
        "mean": mean,
        "std": std,
        "pixel_count": n,
        "sum": float(state["sum"]),
        "sumsq": float(state["sumsq"]),
        "scale_divisor": NIR_SCALE,
        "value_domain_after_scaling": "[0,1]",
        "scope": "train-only full pixels",
        "std_definition": "population_std",
    }


# ---------------------------
# sliding windows
# ---------------------------

def sliding_starts(length: int, crop: int, stride: int) -> List[int]:
    require(length >= crop, "sliding window: length < crop")
    last = length - crop
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def audit_sliding_windows() -> Dict[str, Any]:
    starts = sliding_starts(TILE_SIZE, CROP_SIZE, SLIDING_STRIDE)
    expected_starts = [
        0, 384, 768, 1152, 1536, 1920, 2304, 2688,
        3072, 3456, 3840, 4224, 4608, 4992, 5376, 5488,
    ]
    require(starts == expected_starts,
            f"sliding starts 与冻结协议不一致：{starts}")
    require(len(starts) == 16, f"每方向 starts 应为 16，实际={len(starts)}")
    require(starts[-1] == TILE_SIZE - CROP_SIZE == 5488,
            "最后一个 window 未 anchor 到 5488")

    coords = [[y, x] for y in starts for x in starts]
    require(len(coords) == 256, f"每 tile windows 应为 256，实际={len(coords)}")

    coverage = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
    for y, x in coords:
        coverage[y:y + CROP_SIZE, x:x + CROP_SIZE] += 1

    min_cov = int(coverage.min())
    max_cov = int(coverage.max())
    require(min_cov > 0, f"sliding coverage 存在 0，min={min_cov}")

    unique, counts = np.unique(coverage, return_counts=True)
    coverage_hist = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    return {
        "tile_size": [TILE_SIZE, TILE_SIZE],
        "crop_size": [CROP_SIZE, CROP_SIZE],
        "stride": [SLIDING_STRIDE, SLIDING_STRIDE],
        "starts": starts,
        "starts_per_axis": len(starts),
        "windows_per_tile": len(coords),
        "window_coordinate_order": "y-major then x-major",
        "window_coordinates_sha256": sha256_json_canonical(coords),
        "starts_sha256": sha256_json_canonical(starts),
        "coverage_min": min_cov,
        "coverage_max": max_cov,
        "coverage_histogram_pixels": coverage_hist,
        "all_pixels_covered": bool(min_cov > 0),
        "edge_rule": "append L-crop if regular stride does not land on final anchor",
        "fusion_protocol_future": "uniform mean-logit accumulation (not implemented in this audit)",
    }


# ---------------------------
# 文本报告
# ---------------------------

def build_txt_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"{PROTOCOL_VERSION} 审计报告")
    lines.append("=" * 72)
    lines.append(f"状态: {report.get('status', 'UNKNOWN')}")
    lines.append(f"脚本版本: {SCRIPT_VERSION}")

    if report.get("status") == "FAIL":
        lines.append("")
        lines.append("失败原因:")
        lines.append(str(report.get("error", "unknown error")))
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append("[冻结输入]")
    for k, v in report["inputs"]["sha256"].items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("[Split]")
    lines.append("train/val/test = 18 / 6 / 14")
    lines.append("overlap = False")
    lines.append("union = 38")

    nir = report["nir_normalization"]
    lines.append("")
    lines.append("[NIR normalization: 仅 Train 18 tiles 全像素]")
    lines.append(f"pixel_count = {nir['pixel_count']}")
    lines.append(f"mean = {nir['mean']:.12f}")
    lines.append(f"std = {nir['std']:.12f}")

    samp = report["training_sampling"]
    lines.append("")
    lines.append("[Deterministic train sampling]")
    lines.append(f"epochs = {samp['epochs']}")
    lines.append(f"samples_per_epoch = {samp['samples_per_epoch']}")
    lines.append(f"total_samples = {samp['total_samples']}")
    lines.append(f"cat_max_ratio = {samp['cat_max_ratio']}")
    lines.append(f"max_candidates = {samp['max_candidates']}")
    lines.append(f"rejected_candidate_count = {samp['rejected_candidate_count']}")
    lines.append(f"fallback_crop_count = {samp['fallback_crop_count']}")
    lines.append(f"schedule_sha256 = {samp['schedule_sha256']}")
    lines.append(f"D4 histogram = {samp['d4_histogram']}")
    lines.append(f"dominant ratio summary = {samp['dominant_ratio_summary']}")

    lines.append("")
    lines.append("[Sampled class exposure]")
    for row in samp["class_exposure"]:
        lines.append(
            f"class {row['class_id']} {row['class_name']}: "
            f"pixels={row['sampled_pixels']}, "
            f"pixel_fraction={row['sampled_pixel_fraction']:.8f}, "
            f"crops_present={row['crops_present']}, "
            f"crop_presence_fraction={row['crop_presence_fraction']:.8f}"
        )
    lines.append(f"near_missing_classes = {samp['near_missing_classes']}")

    gt = report["gt_audit"]
    lines.append("")
    lines.append("[Canonical GT]")
    lines.append(f"tiles_checked = {gt['tiles_checked']}")
    lines.append(f"unknown_pixels_total = {gt['unknown_pixels_total']}")
    lines.append(f"ignore_pixels_total = {gt['ignore_pixels_total']}")
    lines.append("reduce_labels = False")

    d4 = report["d4_alignment_audit"]
    lines.append("")
    lines.append("[RGB/NIR/GT 同坐标 + D4]")
    lines.append(f"real_tiles_checked = {d4['real_tiles_checked']}")
    lines.append(f"status = {d4['status']}")
    lines.append(f"audit_fingerprint_sha256 = {d4['audit_fingerprint_sha256']}")

    sw = report["sliding_window_audit"]
    lines.append("")
    lines.append("[Val/Test sliding windows]")
    lines.append(f"starts = {sw['starts']}")
    lines.append(f"starts_per_axis = {sw['starts_per_axis']}")
    lines.append(f"windows_per_tile = {sw['windows_per_tile']}")
    lines.append(f"coverage_min = {sw['coverage_min']}")
    lines.append(f"coverage_max = {sw['coverage_max']}")
    lines.append(f"window_coordinates_sha256 = {sw['window_coordinates_sha256']}")

    lines.append("")
    lines.append("FINAL STATUS: PASS")
    return "\n".join(lines) + "\n"


def critical_protocol_fingerprint(protocol: Dict[str, Any]) -> str:
    return sha256_json_canonical(protocol)


def freeze_dataset_protocol(path: Path, protocol: Dict[str, Any]) -> Dict[str, Any]:
    """
    第一次 PASS 时写入。
    若文件已存在，只接受“关键内容完全一致”的重复审计；绝不静默覆盖不同 protocol。
    """
    new_fp = critical_protocol_fingerprint(protocol)
    if path.exists():
        existing = load_json(path)
        old_fp = critical_protocol_fingerprint(existing)
        require(old_fp == new_fp,
                "dataset_protocol.json 已存在且与本次 PASS 结果不同。"
                "拒绝自动覆盖冻结 metadata；请先人工审查差异。")
        return {"action": "verified_existing", "sha256": sha256_file(path)}

    json_dump(path, protocol)
    return {"action": "created", "sha256": sha256_file(path)}


# ---------------------------
# 主流程
# ---------------------------

def run_audit(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()

    metadata_dir = project_root / "data/processed/potsdam"
    labels_manifest_path = metadata_dir / "labels_manifest.json"
    tile_split_path = metadata_dir / "tile_split.json"
    rgbir_semantics_path = metadata_dir / "rgbir_semantics.json"

    output_dir = project_root / "outputs/dataset_check/dataset_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "status": "RUNNING",
        "protocol_version": PROTOCOL_VERSION,
        "script_version": SCRIPT_VERSION,
        "project_root": str(project_root),
    }

    print("=" * 72)
    print(f"{PROTOCOL_VERSION} 独立审计")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("本脚本不会训练模型，也不会实现 corruption / 4-channel model / dual encoder / fusion / gate。")
    print()

    # 1. 冻结文件
    print("[1/7] 读取并验证三个冻结元数据...")
    labels_manifest = load_json(labels_manifest_path)
    tile_split = load_json(tile_split_path)
    rgbir_semantics = load_json(rgbir_semantics_path)

    input_hashes = {
        "labels_manifest.json": sha256_file(labels_manifest_path),
        "tile_split.json": sha256_file(tile_split_path),
        "rgbir_semantics.json": sha256_file(rgbir_semantics_path),
    }
    split = parse_and_validate_split(tile_split)
    semantics_info = validate_rgbir_semantics(rgbir_semantics)

    gt_paths = resolve_canonical_gt_paths(
        labels_manifest, labels_manifest_path, project_root
    )
    rgbir_root = locate_rgbir_root(project_root)
    rgbir_paths = build_rgbir_map(rgbir_root)

    report["inputs"] = {
        "sha256": input_hashes,
        "labels_manifest": relpath_str(labels_manifest_path, project_root),
        "tile_split": relpath_str(tile_split_path, project_root),
        "rgbir_semantics": relpath_str(rgbir_semantics_path, project_root),
        "rgbir_root": relpath_str(rgbir_root, project_root),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    report["split"] = split
    report["rgbir_semantics_audit"] = semantics_info
    print("  PASS: split=18/6/14；RGBIR=[R,G,B,NIR]；canonical GT 仅来自 manifest。")

    # 2. GT / Train sampling / NIR stats / real D4
    print("[2/7] Train 18 tiles：GT 转换 + NIR 全像素统计 + 3 epochs deterministic sampling...")

    nir_state = {"count": 0, "sum": 0.0, "sumsq": 0.0}
    gt_tile_results: Dict[str, Any] = {}

    total_class_pixels = np.zeros(6, dtype=np.int64)
    total_crop_presence = np.zeros(6, dtype=np.int64)
    total_dominant_ratios: List[float] = []
    total_samples = 0
    total_rejections = 0
    total_fallbacks = 0
    crops_with_rejection = 0
    d4_hist = Counter()
    tile_crop_counts = defaultdict(int)
    epoch_summaries: Dict[str, Any] = {}
    epoch_hashers = {epoch: hashlib.sha256() for epoch in AUDIT_EPOCHS}
    global_schedule_hasher = hashlib.sha256()
    d4_alignment_rows = []

    # 先创建 epoch accumulator
    epoch_acc = {}
    for epoch in AUDIT_EPOCHS:
        epoch_acc[epoch] = {
            "class_pixels": np.zeros(6, dtype=np.int64),
            "crop_presence": np.zeros(6, dtype=np.int64),
            "dominant_ratios": [],
            "fallbacks": 0,
            "rejections": 0,
            "tile_counts": defaultdict(int),
        }

    for tile_idx, tile in enumerate(split["train"], start=1):
        gt_arr = read_tiff_hwc(gt_paths[tile])
        gt_idx, gt_info = gt_rgb_to_class_index(gt_arr, gt_paths[tile])
        gt_tile_results[tile] = gt_info
        del gt_arr

        rgbir = read_tiff_hwc(rgbir_paths[tile], expected_channels=4)
        require(rgbir.dtype == np.uint8,
                f"{rgbir_paths[tile]}: RGBIR 必须为 uint8，实际={rgbir.dtype}")
        accumulate_nir_stats(rgbir, nir_state)

        for epoch in AUDIT_EPOCHS:
            for slot in range(TRAIN_CROPS_PER_TILE_PER_EPOCH):
                s = choose_training_crop(gt_idx, DATA_SEED, epoch, tile, slot)
                counts = s["counts"]
                present = counts > 0

                total_class_pixels += counts
                total_crop_presence += present.astype(np.int64)
                total_dominant_ratios.append(float(s["dominant_ratio"]))
                total_samples += 1
                total_rejections += int(s["rejected_candidates"])
                if s["rejected_candidates"] > 0:
                    crops_with_rejection += 1
                total_fallbacks += int(bool(s["fallback"]))
                d4_hist[int(s["d4_code"])] += 1
                tile_crop_counts[(epoch, tile)] += 1

                ea = epoch_acc[epoch]
                ea["class_pixels"] += counts
                ea["crop_presence"] += present.astype(np.int64)
                ea["dominant_ratios"].append(float(s["dominant_ratio"]))
                ea["fallbacks"] += int(bool(s["fallback"]))
                ea["rejections"] += int(s["rejected_candidates"])
                ea["tile_counts"][tile] += 1

                row_text = (
                    f"{epoch}|{tile}|{slot}|{s['candidate_id']}|"
                    f"{s['x']}|{s['y']}|{s['d4_code']}|"
                    f"{int(s['fallback'])}|{s['dominant_ratio']:.12f}\n"
                ).encode("utf-8")
                epoch_hashers[epoch].update(row_text)
                global_schedule_hasher.update(row_text)

                # 每个 Train tile 用一个真实 crop 做 RGB/NIR/GT alignment + D4 audit。
                if epoch == AUDIT_EPOCHS[0] and slot == 0:
                    d4_alignment_rows.append(
                        audit_real_crop_alignment(
                            rgbir=rgbir,
                            gt_idx=gt_idx,
                            x=int(s["x"]),
                            y=int(s["y"]),
                            d4_code=int(s["d4_code"]),
                            tile_id=tile,
                        )
                    )

        print(
            f"  [{tile_idx:02d}/18] {tile}: "
            f"NIR pixels={TILE_SIZE*TILE_SIZE:,}; "
            f"sampling={len(AUDIT_EPOCHS)*TRAIN_CROPS_PER_TILE_PER_EPOCH} crops; "
            "GT unknown=0"
        )
        del rgbir
        del gt_idx

    expected_nir_count = len(split["train"]) * TILE_SIZE * TILE_SIZE
    require(nir_state["count"] == expected_nir_count,
            f"NIR pixel count 错误：actual={nir_state['count']}, expected={expected_nir_count}")
    nir_stats = finalize_nir_stats(nir_state)
    report["nir_normalization"] = nir_stats

    # 每 tile 每 epoch 必须严格 64。
    for epoch in AUDIT_EPOCHS:
        for tile in split["train"]:
            require(tile_crop_counts[(epoch, tile)] == TRAIN_CROPS_PER_TILE_PER_EPOCH,
                    f"tile-balanced sampling 失败：epoch={epoch} tile={tile} "
                    f"count={tile_crop_counts[(epoch, tile)]}")

    expected_samples_per_epoch = len(split["train"]) * TRAIN_CROPS_PER_TILE_PER_EPOCH
    expected_total_samples = expected_samples_per_epoch * len(AUDIT_EPOCHS)
    require(total_samples == expected_total_samples,
            f"sampling 总数错误：actual={total_samples}, expected={expected_total_samples}")

    require(set(d4_hist.keys()) == set(range(8)),
            f"3 epochs 中 D4 8 种变换未全部出现：{dict(d4_hist)}")

    total_sampled_pixels = total_samples * CROP_SIZE * CROP_SIZE
    class_exposure = []
    near_missing_classes = []
    for spec in CLASS_SPECS:
        cid = spec["id"]
        pix = int(total_class_pixels[cid])
        crops = int(total_crop_presence[cid])
        pixel_frac = pix / total_sampled_pixels
        crop_frac = crops / total_samples
        row = {
            "class_id": cid,
            "class_name": spec["name"],
            "sampled_pixels": pix,
            "sampled_pixel_fraction": float(pixel_frac),
            "crops_present": crops,
            "crop_presence_fraction": float(crop_frac),
        }
        class_exposure.append(row)
        if (
            pix == 0
            or crops == 0
            or pixel_frac < NEAR_MISSING_MIN_PIXEL_FRACTION
            or crop_frac < NEAR_MISSING_MIN_CROP_PRESENCE_FRACTION
        ):
            near_missing_classes.append(cid)

    require(not near_missing_classes,
            "deterministic sampler 出现“几乎完全采不到”的类别："
            f"{near_missing_classes}。阈值：crop_presence>="
            f"{NEAR_MISSING_MIN_CROP_PRESENCE_FRACTION}, pixel_fraction>="
            f"{NEAR_MISSING_MIN_PIXEL_FRACTION}。请检查真实 class exposure，禁止自动放宽标准。")

    for epoch in AUDIT_EPOCHS:
        ea = epoch_acc[epoch]
        require(all(ea["tile_counts"][t] == TRAIN_CROPS_PER_TILE_PER_EPOCH for t in split["train"]),
                f"epoch {epoch}: tile counts 不平衡")
        epoch_summaries[str(epoch)] = {
            "sample_count": expected_samples_per_epoch,
            "class_sampled_pixels": [int(x) for x in ea["class_pixels"]],
            "class_crops_present": [int(x) for x in ea["crop_presence"]],
            "dominant_ratio_summary": summarize_ratios(ea["dominant_ratios"]),
            "fallback_crop_count": int(ea["fallbacks"]),
            "rejected_candidate_count": int(ea["rejections"]),
            "schedule_sha256": epoch_hashers[epoch].hexdigest(),
            "tile_crop_counts": {t: int(ea["tile_counts"][t]) for t in split["train"]},
        }

    report["training_sampling"] = {
        "data_seed": DATA_SEED,
        "epochs": list(AUDIT_EPOCHS),
        "crop_size": [CROP_SIZE, CROP_SIZE],
        "train_tiles": len(split["train"]),
        "crops_per_tile_per_epoch": TRAIN_CROPS_PER_TILE_PER_EPOCH,
        "samples_per_epoch": expected_samples_per_epoch,
        "total_samples": total_samples,
        "tile_balanced": True,
        "cat_max_ratio": CAT_MAX_RATIO,
        "max_candidates": MAX_CANDIDATES,
        "fallback_rule": "if all 10 fail, choose candidate with minimum dominant-class ratio",
        "coordinate_rng": "SHA256 stateless rejection sampler keyed by (seed,epoch,tile_id,sample_slot,candidate_id)",
        "d4_rng": "SHA256 stateless rejection sampler keyed by (seed,epoch,tile_id,sample_slot)",
        "dominant_ratio_summary": summarize_ratios(total_dominant_ratios),
        "rejected_candidate_count": int(total_rejections),
        "crops_with_at_least_one_rejection": int(crops_with_rejection),
        "fallback_crop_count": int(total_fallbacks),
        "d4_histogram": {str(k): int(d4_hist[k]) for k in range(8)},
        "class_exposure": class_exposure,
        "near_missing_definition": {
            "min_crop_presence_fraction": NEAR_MISSING_MIN_CROP_PRESENCE_FRACTION,
            "min_pixel_fraction": NEAR_MISSING_MIN_PIXEL_FRACTION,
        },
        "near_missing_classes": near_missing_classes,
        "per_epoch": epoch_summaries,
        "schedule_sha256": global_schedule_hasher.hexdigest(),
    }

    d4_audit_fp = sha256_json_canonical(d4_alignment_rows)
    report["d4_alignment_audit"] = {
        "status": "PASS",
        "real_tiles_checked": len(d4_alignment_rows),
        "expected_real_tiles_checked": len(split["train"]),
        "definition": "same raw crop coordinates; same D4 code; separate transforms equal one joint transform; inverse exact",
        "samples": d4_alignment_rows,
        "audit_fingerprint_sha256": d4_audit_fp,
    }
    require(len(d4_alignment_rows) == 18, "D4 real-crop audit 未覆盖全部 18 train tiles")

    print("  PASS: Train sampling / NIR stats / 18 个真实 crop 的 RGB-NIR-GT D4 对齐。")
    print(f"  NIR pixel_count={nir_stats['pixel_count']:,}")
    print(f"  NIR mean={nir_stats['mean']:.12f}")
    print(f"  NIR std={nir_stats['std']:.12f}")
    print(f"  sampling schedule SHA256={report['training_sampling']['schedule_sha256']}")

    # 3. Val/Test GT mapping 审计（不参与 NIR normalization）
    print("[3/7] Validation/Test canonical GT RGB->class 审计（不读取其 NIR 统计）...")
    for group_name in ("val", "test"):
        tiles = split[group_name]
        for tile_idx, tile in enumerate(tiles, start=1):
            gt_arr = read_tiff_hwc(gt_paths[tile])
            _gt_idx, gt_info = gt_rgb_to_class_index(gt_arr, gt_paths[tile])
            gt_tile_results[tile] = gt_info
            del gt_arr
            del _gt_idx
        print(f"  {group_name}: {len(tiles)} tiles PASS")

    unknown_total = sum(int(v["unknown_pixels"]) for v in gt_tile_results.values())
    ignore_total = sum(int(v["ignore_pixels"]) for v in gt_tile_results.values())
    require(len(gt_tile_results) == 38, f"GT audit 应覆盖 38 tiles，实际={len(gt_tile_results)}")
    require(unknown_total == 0, f"canonical GT unknown pixels total={unknown_total}")
    require(ignore_total == 0, f"canonical GT ignore pixels total={ignore_total}")

    report["gt_audit"] = {
        "tiles_checked": 38,
        "mapping": CLASS_SPECS,
        "ignore_index_reserved_sentinel": IGNORE_INDEX,
        "unknown_pixels_total": unknown_total,
        "ignore_pixels_total": ignore_total,
        "reduce_labels": False,
        "hard_error_on_unknown_rgb": True,
        "per_tile": gt_tile_results,
    }
    print("  PASS: 38/38 canonical GT unknown=0, ignore=0, reduce_labels=False。")

    # 4. sliding windows
    print("[4/7] 审计 Val/Test deterministic sliding-window coordinates...")
    sw = audit_sliding_windows()
    report["sliding_window_audit"] = sw
    print(f"  starts={sw['starts']}")
    print(f"  windows/tile={sw['windows_per_tile']}, coverage min/max={sw['coverage_min']}/{sw['coverage_max']}")
    print(f"  coordinate SHA256={sw['window_coordinates_sha256']}")

    # 5. hard criteria 汇总
    print("[5/7] 汇总 hard criteria...")
    hard_criteria = {
        "three_frozen_metadata_loaded": True,
        "split_exact_18_6_14": True,
        "rgbir_semantics_exact_R_G_B_NIR": True,
        "rgbir_source_only_4_Ortho_RGBIR": True,
        "nir_stats_train_only_18_tiles": nir_stats["pixel_count"] == expected_nir_count,
        "nir_pixel_count_exact": nir_stats["pixel_count"] == 648000000,
        "nir_std_positive": nir_stats["std"] > 0,
        "train_samples_per_epoch_1152": expected_samples_per_epoch == 1152,
        "train_total_samples_3_epochs_3456": total_samples == 3456,
        "each_train_tile_64_crops_per_epoch": True,
        "stateless_coordinate_rng": True,
        "cat_max_ratio_0_75": CAT_MAX_RATIO == 0.75,
        "max_candidates_10": MAX_CANDIDATES == 10,
        "fallback_min_dominant_ratio": True,
        "all_six_classes_sampled_not_near_missing": len(near_missing_classes) == 0,
        "canonical_gt_38_tiles_unknown_zero": unknown_total == 0,
        "canonical_gt_ignore_zero": ignore_total == 0,
        "reduce_labels_false": True,
        "d4_all_8_codes_observed": set(d4_hist.keys()) == set(range(8)),
        "real_crop_alignment_18_train_tiles": len(d4_alignment_rows) == 18,
        "sliding_starts_16_per_axis": sw["starts_per_axis"] == 16,
        "sliding_windows_256_per_tile": sw["windows_per_tile"] == 256,
        "sliding_final_anchor_5488": sw["starts"][-1] == 5488,
        "sliding_full_coverage": sw["coverage_min"] > 0,
    }
    failed_criteria = [k for k, v in hard_criteria.items() if not bool(v)]
    require(not failed_criteria, f"hard criteria 未通过：{failed_criteria}")
    report["hard_criteria"] = hard_criteria

    # 6. protocol metadata
    print("[6/7] 生成冻结 dataset_protocol.json 内容...")
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "FROZEN_AFTER_AUDIT_PASS",
        "source_metadata_sha256": input_hashes,
        "data_source": {
            "rgbir": "data/raw/potsdam/Potsdam/4_Ortho_RGBIR (resolved directory must be named 4_Ortho_RGBIR)",
            "gt": "canonical path from labels_manifest.json only",
            "split": "tile_split.json only",
            "channel_semantics": "rgbir_semantics.json only",
        },
        "split": {
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
        },
        "labels": {
            "classes": CLASS_SPECS,
            "num_classes": 6,
            "ignore_index_reserved_sentinel": IGNORE_INDEX,
            "expected_ignore_pixels": 0,
            "reduce_labels": False,
            "unknown_rgb_policy": "hard_error",
        },
        "normalization": {
            "rgb": {
                "input_scale_divisor": 255.0,
                "mean": RGB_IMAGENET_MEAN,
                "std": RGB_IMAGENET_STD,
                "scope": "ImageNet pretrained RGB normalization",
            },
            "nir": nir_stats,
        },
        "training_sampling": {
            "data_seed": DATA_SEED,
            "crop_size": [CROP_SIZE, CROP_SIZE],
            "crops_per_tile_per_epoch": TRAIN_CROPS_PER_TILE_PER_EPOCH,
            "samples_per_epoch": 1152,
            "tile_balanced": True,
            "cat_max_ratio": CAT_MAX_RATIO,
            "max_candidates": MAX_CANDIDATES,
            "fallback_rule": "minimum dominant-class ratio among 10 candidates",
            "coordinate_rng": "SHA256 stateless rejection sampler keyed by (seed,epoch,tile_id,sample_slot,candidate_id)",
            "d4": {
                "enabled": True,
                "codes": {
                    "0": "rot90 k=0",
                    "1": "rot90 k=1",
                    "2": "rot90 k=2",
                    "3": "rot90 k=3",
                    "4": "fliplr(rot90 k=0)",
                    "5": "fliplr(rot90 k=1)",
                    "6": "fliplr(rot90 k=2)",
                    "7": "fliplr(rot90 k=3)",
                },
                "rng": "SHA256 stateless rejection sampler keyed by (seed,epoch,tile_id,sample_slot)",
            },
            "photometric_augmentation": False,
            "resize_or_scale_jitter": False,
            "audit_schedule_sha256_epochs_0_1_2": report["training_sampling"]["schedule_sha256"],
        },
        "validation_test_inference": {
            "tile_size": [TILE_SIZE, TILE_SIZE],
            "crop_size": [CROP_SIZE, CROP_SIZE],
            "stride": [SLIDING_STRIDE, SLIDING_STRIDE],
            "starts": sw["starts"],
            "starts_per_axis": 16,
            "windows_per_tile": 256,
            "edge_rule": sw["edge_rule"],
            "window_coordinate_order": sw["window_coordinate_order"],
            "window_coordinates_sha256": sw["window_coordinates_sha256"],
            "overlap_fusion": "uniform mean-logit accumulation",
            "tta": False,
            "random_crop": False,
        },
        "metrics": {
            "primary_miou": "6-class mIoU from global split confusion matrix",
            "per_tile_confusion_and_iou": True,
            "do_not_average_patch_miou": True,
            "do_not_use_mean_tile_miou_as_primary": True,
        },
        "corruption_separation_rule": {
            "implemented_here": False,
            "future_rule": "robustness corruption is separate from train augmentation; RGB only; NIR and GT unchanged",
        },
        "audit_fingerprints": {
            "sampling_schedule_sha256_epochs_0_1_2": report["training_sampling"]["schedule_sha256"],
            "d4_real_crop_audit_sha256": d4_audit_fp,
            "sliding_window_coordinates_sha256": sw["window_coordinates_sha256"],
        },
    }

    frozen_path = metadata_dir / "dataset_protocol.json"
    freeze_info = freeze_dataset_protocol(frozen_path, protocol)
    report["frozen_dataset_protocol"] = {
        "path": relpath_str(frozen_path, project_root),
        **freeze_info,
    }

    # 7. PASS
    report["status"] = "PASS"
    print("[7/7] 冻结完成。")
    print(f"  dataset_protocol.json: {freeze_info['action']}")
    print("FINAL STATUS: PASS")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and freeze Dataset Protocol v1 before any SegFormer training."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录；默认取 tools/ 的父目录。",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = project_root / "outputs/dataset_check/dataset_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)

    console_path = output_dir / "console.txt"
    audit_json_path = output_dir / "dataset_protocol_audit.json"
    audit_txt_path = output_dir / "dataset_protocol_audit.txt"

    # console.txt 每次审计覆盖，避免把多次运行拼在一起。
    with console_path.open("w", encoding="utf-8", buffering=1) as console_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, console_file)
        sys.stderr = Tee(old_stderr, console_file)

        report: Dict[str, Any]
        exit_code = 1
        try:
            report = run_audit(project_root)
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
                "protocol_version": PROTOCOL_VERSION,
                "script_version": SCRIPT_VERSION,
                "project_root": str(project_root),
                "error_type": type(e).__name__,
                "error": str(e),
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    json_dump(audit_json_path, report)
    audit_txt_path.write_text(build_txt_report(report), encoding="utf-8")

    # 最后再在终端告诉用户报告位置（不写进 console，避免文件自引用式噪声）。
    print(f"console: {console_path}")
    print(f"audit json: {audit_json_path}")
    print(f"audit txt: {audit_txt_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
