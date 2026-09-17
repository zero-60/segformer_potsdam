#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISPRS Potsdam full-tile sliding-window evaluation protocol.

职责：
1) 接收 Val/Test 每个 512x512 window 的 6-class logits；
2) 严格按照冻结的 y-major window coordinates 做 uniform mean-logit accumulation；
3) 流式拼成完整 6000x6000 prediction，避免分配 6x6000x6000 的完整 logits；
4) 按完整 tile GT 计算 6x6 confusion matrix；
5) 主 mIoU 来自整个 split 的 global confusion matrix；
6) 同时保留 per-tile IoU / mIoU。

本模块不包含：
- SegFormer 或任何模型
- 训练
- corruption
- RGB+NIR model
- dual encoder / fusion / Quality Gate
- ΔmIoU / Retention / NIR Gain / Gate Gain 的实验聚合
  （这些属于后续 robustness experiment aggregation）

重要：
- uniform mean-logit，而不是 window overwrite；
- window coordinate 必须与 dataset_protocol.json 完全一致；
- GT 必须是 full-tile 0..5 class index；
- patch mIoU 不作为主指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Evaluation Protocol v1"

NUM_CLASSES = 6
IGNORE_INDEX = 255
EXPECTED_TILE_SIZE = 6000
EXPECTED_CROP_SIZE = 512
EXPECTED_STRIDE = 384
EXPECTED_WINDOWS_PER_TILE = 256
EXPECTED_WINDOW_COORD_SHA256 = (
    "58a2f857c7d84adb0a17f2ce0f5b1802452886127963f42fd9ac125c7dd84e1a"
)

CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]


class EvaluationProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationProtocolError(message)


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


def load_json(path: Path) -> Any:
    require(path.is_file(), f"缺少文件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise EvaluationProtocolError(f"JSON 读取失败：{path}\n{e}") from e


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# Frozen evaluation spec
# ---------------------------------------------------------------------

class FrozenEvaluationSpec:
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        meta = self.project_root / "data/processed/potsdam"

        self.dataset_protocol_path = meta / "dataset_protocol.json"
        self.dataloader_protocol_path = meta / "dataloader_protocol.json"

        self.dataset_protocol = load_json(self.dataset_protocol_path)
        self.dataloader_protocol = load_json(self.dataloader_protocol_path)

        self._validate()

        infer = self.dataset_protocol["validation_test_inference"]
        self.tile_size = int(infer["tile_size"][0])
        self.crop_size = int(infer["crop_size"][0])
        self.stride = int(infer["stride"][0])
        self.starts = [int(v) for v in infer["starts"]]
        self.window_coordinates = [
            (y, x)
            for y in self.starts
            for x in self.starts
        ]
        self.window_coordinates_sha256 = str(
            infer["window_coordinates_sha256"]
        )

    def _validate(self) -> None:
        dp = self.dataset_protocol
        dl = self.dataloader_protocol

        require(
            dp.get("protocol_version") == "Dataset Protocol v1"
            and dp.get("status") == "FROZEN_AFTER_AUDIT_PASS",
            "dataset_protocol.json 不是冻结 PASS 状态"
        )
        require(
            dl.get("protocol_version") == "DataLoader Protocol v1"
            and dl.get("status") == "FROZEN_AFTER_AUDIT_PASS",
            "dataloader_protocol.json 不是冻结 PASS 状态"
        )

        # DataLoader protocol 必须指向当前 dataset_protocol 文件。
        expected_dp_hash = dl["source"]["dataset_protocol_sha256"]
        actual_dp_hash = sha256_file(self.dataset_protocol_path)
        require(
            expected_dp_hash == actual_dp_hash,
            "dataloader_protocol.json 记录的 dataset_protocol SHA256 "
            "与当前文件不一致"
        )

        labels = dp["labels"]
        require(int(labels["num_classes"]) == NUM_CLASSES,
                "num_classes != 6")
        require(
            int(labels["ignore_index_reserved_sentinel"]) == IGNORE_INDEX,
            "ignore_index != 255"
        )
        require(labels["reduce_labels"] is False,
                "reduce_labels 必须为 False")

        infer = dp["validation_test_inference"]
        require(infer["tile_size"] == [6000, 6000],
                "tile_size != 6000x6000")
        require(infer["crop_size"] == [512, 512],
                "crop_size != 512x512")
        require(infer["stride"] == [384, 384],
                "stride != 384")
        require(int(infer["starts_per_axis"]) == 16,
                "starts_per_axis != 16")
        require(int(infer["windows_per_tile"]) == EXPECTED_WINDOWS_PER_TILE,
                "windows_per_tile != 256")
        require(infer["starts"][-1] == 5488,
                "final sliding anchor != 5488")
        require(
            infer["window_coordinate_order"] == "y-major then x-major",
            "window coordinate order 不是 y-major then x-major"
        )
        require(
            infer["overlap_fusion"] == "uniform mean-logit accumulation",
            "overlap fusion 不是 uniform mean-logit accumulation"
        )
        require(
            infer["window_coordinates_sha256"] == EXPECTED_WINDOW_COORD_SHA256,
            "window coordinate SHA256 与冻结值不一致"
        )

        coords = [
            [y, x]
            for y in infer["starts"]
            for x in infer["starts"]
        ]
        require(
            sha256_json_canonical(coords)
            == infer["window_coordinates_sha256"],
            "starts 无法重现冻结 window coordinate SHA256"
        )

        metrics = dp["metrics"]
        require(
            metrics["primary_miou"]
            == "6-class mIoU from global split confusion matrix",
            "primary mIoU 定义与冻结协议不一致"
        )
        require(metrics["per_tile_confusion_and_iou"] is True,
                "必须保留 per-tile confusion/IoU")
        require(metrics["do_not_average_patch_miou"] is True,
                "禁止平均 patch mIoU")
        require(metrics["do_not_use_mean_tile_miou_as_primary"] is True,
                "mean tile mIoU 不能作为主指标")


# ---------------------------------------------------------------------
# Streaming uniform mean-logit stitcher
# ---------------------------------------------------------------------

class StreamingMeanLogitStitcher:
    """
    流式 uniform mean-logit accumulation。

    expected_coordinates 必须是 y-major，并且同一 y 的 x 递增。
    当进入下一个 y band 时，所有 global row < new_y 的像素以后不会再被
    新 window 覆盖，因此可以立即：
        mean_logits = logit_sum / count
        prediction = argmax(mean_logits)
    然后只保留仍可能被未来 window 覆盖的 overlap rows。

    因此内存近似：
        num_classes * crop_h * tile_w * float32
    而不是：
        num_classes * tile_h * tile_w * float32
    """

    def __init__(
        self,
        *,
        tile_height: int,
        tile_width: int,
        crop_height: int,
        crop_width: int,
        num_classes: int,
        expected_coordinates: Sequence[Tuple[int, int]],
    ):
        self.tile_height = int(tile_height)
        self.tile_width = int(tile_width)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.num_classes = int(num_classes)
        self.expected_coordinates = [
            (int(y), int(x)) for y, x in expected_coordinates
        ]

        require(self.tile_height >= self.crop_height,
                "tile_height < crop_height")
        require(self.tile_width >= self.crop_width,
                "tile_width < crop_width")
        require(self.num_classes > 1,
                "num_classes 必须 > 1")
        require(self.expected_coordinates,
                "expected_coordinates 为空")

        self._validate_coordinate_sequence()

        first_y = self.expected_coordinates[0][0]
        require(first_y == 0,
                "当前 streaming protocol 要求第一个 y=0")

        self.base_y = first_y
        self.current_band_y = first_y
        self.next_coordinate_index = 0

        self.logit_sum = np.zeros(
            (
                self.num_classes,
                self.crop_height,
                self.tile_width,
            ),
            dtype=np.float32,
        )
        self.coverage = np.zeros(
            (self.crop_height, self.tile_width),
            dtype=np.uint16,
        )

        self.prediction = np.empty(
            (self.tile_height, self.tile_width),
            dtype=np.uint8,
        )

        self.flushed_until_y = 0
        self.coverage_min: Optional[int] = None
        self.coverage_max: Optional[int] = None
        self.finalized = False

    def _validate_coordinate_sequence(self) -> None:
        seen = set()
        previous: Optional[Tuple[int, int]] = None

        for coord in self.expected_coordinates:
            y, x = coord
            require(coord not in seen,
                    f"重复 window coordinate：{coord}")
            seen.add(coord)

            require(
                0 <= y <= self.tile_height - self.crop_height,
                f"window y 越界：{coord}"
            )
            require(
                0 <= x <= self.tile_width - self.crop_width,
                f"window x 越界：{coord}"
            )

            if previous is not None:
                py, px = previous
                require(
                    y > py or (y == py and x > px),
                    "expected_coordinates 必须严格 y-major / x-increasing"
                )
            previous = coord

        ys = sorted({y for y, _ in self.expected_coordinates})
        require(ys[0] == 0,
                "第一个 y start 必须为 0")
        require(
            ys[-1] + self.crop_height == self.tile_height,
            "最后一个 y window 未覆盖 tile bottom"
        )

    def _update_coverage_extrema(self, block: np.ndarray) -> None:
        block_min = int(block.min())
        block_max = int(block.max())
        require(block_min > 0,
                "mean-logit finalize 遇到 coverage=0 像素")

        if self.coverage_min is None:
            self.coverage_min = block_min
            self.coverage_max = block_max
        else:
            self.coverage_min = min(self.coverage_min, block_min)
            self.coverage_max = max(self.coverage_max, block_max)

    def _flush_rows(self, global_end_y: int) -> None:
        """
        finalize [base_y, global_end_y)。
        """
        global_end_y = int(global_end_y)
        require(
            self.base_y < global_end_y <= self.base_y + self.crop_height,
            "非法 flush 范围"
        )

        n = global_end_y - self.base_y
        counts = self.coverage[:n, :]
        self._update_coverage_extrema(counts)

        # 显式执行 mean-logit；不是依赖 argmax(sum)==argmax(mean) 的简化。
        mean_logits = (
            self.logit_sum[:, :n, :]
            / counts[None, :, :].astype(np.float32)
        )
        pred = np.argmax(mean_logits, axis=0).astype(np.uint8, copy=False)

        self.prediction[self.base_y:global_end_y, :] = pred
        self.flushed_until_y = global_end_y

    def _advance_to_band(self, new_y: int) -> None:
        new_y = int(new_y)
        require(new_y > self.current_band_y,
                "new_y 必须大于 current_band_y")

        # row < new_y 不可能被未来 window 覆盖，先 finalize。
        self._flush_rows(new_y)

        shift = new_y - self.base_y
        require(
            0 < shift <= self.crop_height,
            f"相邻 y starts gap 超出 crop height：{shift}"
        )

        overlap = self.crop_height - shift

        new_sum = np.zeros_like(self.logit_sum)
        new_cov = np.zeros_like(self.coverage)

        if overlap > 0:
            new_sum[:, :overlap, :] = self.logit_sum[:, shift:, :]
            new_cov[:overlap, :] = self.coverage[shift:, :]

        self.logit_sum = new_sum
        self.coverage = new_cov
        self.base_y = new_y
        self.current_band_y = new_y

    @staticmethod
    def _to_float32_numpy(logits: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(logits, torch.Tensor):
            require(logits.device.type == "cpu",
                    "Stitcher 只接受 CPU logits；请在调用前 detach().cpu()")
            arr = logits.detach().contiguous().numpy()
        else:
            arr = np.asarray(logits)

        require(np.issubdtype(arr.dtype, np.floating),
                f"logits 必须是 floating dtype，实际 {arr.dtype}")

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        require(np.all(np.isfinite(arr)),
                "logits 出现 NaN/Inf")
        return arr

    def add_window(
        self,
        logits: np.ndarray | torch.Tensor,
        *,
        y: int,
        x: int,
    ) -> None:
        require(not self.finalized,
                "Stitcher 已 finalize，不能继续 add_window")
        require(
            self.next_coordinate_index < len(self.expected_coordinates),
            "add_window 数量超过 expected coordinates"
        )

        expected = self.expected_coordinates[self.next_coordinate_index]
        actual = (int(y), int(x))
        require(
            actual == expected,
            f"window coordinate/order 不一致："
            f"expected={expected}, actual={actual}, "
            f"index={self.next_coordinate_index}"
        )

        y, x = actual

        if y != self.current_band_y:
            self._advance_to_band(y)

        require(self.base_y == y,
                "streaming buffer base_y 与当前 window y 不一致")

        arr = self._to_float32_numpy(logits)
        require(
            arr.shape == (
                self.num_classes,
                self.crop_height,
                self.crop_width,
            ),
            f"logits shape 异常：{arr.shape}"
        )

        x1 = x + self.crop_width
        self.logit_sum[:, :, x:x1] += arr
        self.coverage[:, x:x1] += 1
        self.next_coordinate_index += 1

    def finalize(self) -> np.ndarray:
        require(not self.finalized,
                "Stitcher.finalize() 不能重复调用")
        require(
            self.next_coordinate_index == len(self.expected_coordinates),
            f"window 数量不完整："
            f"{self.next_coordinate_index}/"
            f"{len(self.expected_coordinates)}"
        )

        self._flush_rows(self.tile_height)

        require(
            self.flushed_until_y == self.tile_height,
            "没有 finalize 到 tile bottom"
        )
        require(
            self.coverage_min is not None
            and self.coverage_max is not None,
            "coverage extrema 未生成"
        )

        self.finalized = True
        return self.prediction


# ---------------------------------------------------------------------
# Dense reference（仅 smoke-test 小图）
# ---------------------------------------------------------------------

def dense_mean_logit_reference(
    *,
    tile_height: int,
    tile_width: int,
    crop_height: int,
    crop_width: int,
    num_classes: int,
    windows: Sequence[Tuple[int, int, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(
        (num_classes, tile_height, tile_width),
        dtype=np.float32,
    )
    counts = np.zeros(
        (tile_height, tile_width),
        dtype=np.uint16,
    )

    for y, x, logits in windows:
        arr = np.asarray(logits, dtype=np.float32)
        require(
            arr.shape == (num_classes, crop_height, crop_width),
            "dense reference logits shape 异常"
        )
        sums[:, y:y+crop_height, x:x+crop_width] += arr
        counts[y:y+crop_height, x:x+crop_width] += 1

    require(int(counts.min()) > 0,
            "dense reference coverage=0")
    means = sums / counts[None, :, :].astype(np.float32)
    pred = np.argmax(means, axis=0).astype(np.uint8)
    return pred, counts


# ---------------------------------------------------------------------
# Confusion matrix / IoU
# ---------------------------------------------------------------------

def confusion_matrix_6class(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    chunk_rows: int = 512,
) -> np.ndarray:
    if isinstance(prediction, torch.Tensor):
        pred = prediction.detach().cpu().numpy()
    else:
        pred = np.asarray(prediction)

    if isinstance(target, torch.Tensor):
        gt = target.detach().cpu().numpy()
    else:
        gt = np.asarray(target)

    require(pred.ndim == 2 and gt.ndim == 2,
            "prediction/target 必须为 HxW")
    require(pred.shape == gt.shape,
            f"prediction/target shape 不一致：{pred.shape}/{gt.shape}")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    h = pred.shape[0]
    for y0 in range(0, h, int(chunk_rows)):
        y1 = min(h, y0 + int(chunk_rows))
        p = pred[y0:y1].astype(np.int64, copy=False)
        g = gt[y0:y1].astype(np.int64, copy=False)

        require(
            np.all((p >= 0) & (p < NUM_CLASSES)),
            f"prediction 出现非法 class，rows={y0}:{y1}"
        )
        require(
            np.all((g >= 0) & (g < NUM_CLASSES)),
            f"target 出现非法 class / ignore，rows={y0}:{y1}"
        )

        encoded = g.ravel() * NUM_CLASSES + p.ravel()
        block = np.bincount(
            encoded,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        cm += block.astype(np.int64, copy=False)

    require(
        int(cm.sum()) == int(pred.size),
        "confusion matrix pixel count != image pixels"
    )
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    require(cm.shape == (NUM_CLASSES, NUM_CLASSES),
            f"confusion matrix shape 异常：{cm.shape}")
    require(np.all(cm >= 0),
            "confusion matrix 不能有负数")

    tp = np.diag(cm).astype(np.float64)
    gt_count = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp

    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid]

    total = float(cm.sum())
    pixel_accuracy = (
        float(tp.sum() / total) if total > 0 else float("nan")
    )
    miou = (
        float(np.nanmean(iou))
        if np.any(valid)
        else float("nan")
    )

    return {
        "confusion_matrix": cm.tolist(),
        "class_iou": [
            None if not np.isfinite(v) else float(v)
            for v in iou
        ],
        "class_union_pixels": [int(v) for v in union],
        "valid_iou_classes": int(valid.sum()),
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "pixel_count": int(cm.sum()),
    }


class SplitConfusionEvaluator:
    """
    主指标：
        sum(tile confusion matrices) -> global confusion matrix -> mIoU

    同时保存 per-tile confusion/IoU。
    """

    def __init__(self):
        self.global_cm = np.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=np.int64,
        )
        self.per_tile: Dict[str, Dict[str, Any]] = {}

    def add_tile(
        self,
        tile_id: str,
        prediction: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> Dict[str, Any]:
        require(tile_id not in self.per_tile,
                f"tile 重复加入 evaluator：{tile_id}")

        cm = confusion_matrix_6class(prediction, target)
        metrics = metrics_from_confusion_matrix(cm)
        self.per_tile[tile_id] = metrics
        self.global_cm += cm
        return metrics

    def summary(self) -> Dict[str, Any]:
        global_metrics = metrics_from_confusion_matrix(self.global_cm)

        # mean(tile mIoU) 仅作为诊断字段，明确不是主指标。
        tile_mious = [
            float(v["miou"])
            for v in self.per_tile.values()
            if math.isfinite(float(v["miou"]))
        ]
        diagnostic_mean_tile_miou = (
            float(np.mean(tile_mious))
            if tile_mious
            else float("nan")
        )

        return {
            "primary_metric_definition": (
                "global confusion matrix -> 6-class IoU -> mIoU"
            ),
            "global": global_metrics,
            "per_tile": self.per_tile,
            "diagnostic_mean_tile_miou_not_primary": (
                diagnostic_mean_tile_miou
            ),
        }


# ---------------------------------------------------------------------
# Oracle helper（仅 smoke-test）
# ---------------------------------------------------------------------

def oracle_logits_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    require(labels.ndim == 2,
            "oracle labels 必须为 HxW")
    require(np.all((labels >= 0) & (labels < NUM_CLASSES)),
            "oracle labels 必须为 0..5")

    h, w = labels.shape
    logits = np.zeros(
        (NUM_CLASSES, h, w),
        dtype=np.float32,
    )
    flat = logits.reshape(NUM_CLASSES, -1)
    indices = labels.astype(np.int64, copy=False).ravel()
    positions = np.arange(indices.size, dtype=np.int64)
    flat[indices, positions] = 1.0
    return logits


# ---------------------------------------------------------------------
# Protocol freezing
# ---------------------------------------------------------------------

def freeze_protocol(
    path: Path,
    protocol: Dict[str, Any],
) -> Dict[str, str]:
    new_fp = sha256_json_canonical(protocol)

    if path.exists():
        existing = load_json(path)
        old_fp = sha256_json_canonical(existing)
        require(
            old_fp == new_fp,
            "evaluation_protocol.json 已存在且与本次结果不同；"
            "拒绝静默覆盖，请先人工审查。"
        )
        return {
            "action": "verified_existing",
            "sha256": sha256_file(path),
        }

    write_json(path, protocol)
    return {
        "action": "created",
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------

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


def _ensure_project_import(project_root: Path) -> None:
    root_str = str(project_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_smoke_test(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    _ensure_project_import(project_root)

    from data_pipeline.potsdam_dataset import (
        PotsdamSlidingWindowDataset,
    )

    spec = FrozenEvaluationSpec(project_root)

    print("=" * 72)
    print("Potsdam full-tile mean-logit / metric evaluation audit")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("本测试不会启动模型或训练。")
    print()

    print("[1/5] 验证冻结 evaluation 输入协议...")
    require(len(spec.window_coordinates) == 256,
            "window coordinates != 256")
    require(
        spec.window_coordinates_sha256 == EXPECTED_WINDOW_COORD_SHA256,
        "window coordinate SHA256 mismatch"
    )
    print("  PASS: Dataset/DataLoader protocol 均为冻结 PASS 状态。")
    print(
        f"  window coordinate SHA256 = "
        f"{spec.window_coordinates_sha256}"
    )

    print("[2/5] 小图冲突窗口：证明执行的是 mean-logit 而不是 overwrite...")
    # tile 4x6，两张 4x4 window，x overlap=[2,4)。
    # window 1: class0 logit=10
    # window 2: class1 logit=6
    # overlap mean: class0=5, class1=3 -> class0
    # 若简单 overwrite，则 overlap 会变 class1。
    coords_small = [(0, 0), (0, 2)]
    w1 = np.zeros((2, 4, 4), dtype=np.float32)
    w2 = np.zeros((2, 4, 4), dtype=np.float32)
    w1[0, :, :] = 10.0
    w2[1, :, :] = 6.0

    stitch_small = StreamingMeanLogitStitcher(
        tile_height=4,
        tile_width=6,
        crop_height=4,
        crop_width=4,
        num_classes=2,
        expected_coordinates=coords_small,
    )
    stitch_small.add_window(w1, y=0, x=0)
    stitch_small.add_window(w2, y=0, x=2)
    pred_stream = stitch_small.finalize()

    pred_dense, count_dense = dense_mean_logit_reference(
        tile_height=4,
        tile_width=6,
        crop_height=4,
        crop_width=4,
        num_classes=2,
        windows=[
            (0, 0, w1),
            (0, 2, w2),
        ],
    )

    require(np.array_equal(pred_stream, pred_dense),
            "streaming mean-logit 与 dense reference 不一致")
    require(np.all(pred_stream[:, 2:4] == 0),
            "overlap 未得到 mean-logit class0")
    overwrite_overlap = np.ones((4, 2), dtype=np.uint8)
    require(
        not np.array_equal(pred_stream[:, 2:4], overwrite_overlap),
        "测试构造无法区分 mean 与 overwrite"
    )
    require(int(count_dense.min()) == 1 and int(count_dense.max()) == 2,
            "small dense coverage 应为 1..2")
    print("  PASS: streaming == dense mean-logit；overlap 结果明确不同于 overwrite。")

    print("[3/5] 真实 Validation tile：256 windows full-tile oracle stitching...")
    val = PotsdamSlidingWindowDataset(project_root, split="val")
    tile_id = val.tile_ids[0]

    full_gt_t = val.load_full_label(tile_id)
    require(
        tuple(full_gt_t.shape) == (6000, 6000),
        "Validation full GT shape != 6000x6000"
    )
    full_gt = full_gt_t.numpy().astype(np.uint8, copy=True)
    del full_gt_t

    stitch = StreamingMeanLogitStitcher(
        tile_height=6000,
        tile_width=6000,
        crop_height=512,
        crop_width=512,
        num_classes=6,
        expected_coordinates=spec.window_coordinates,
    )

    for i, (y, x) in enumerate(spec.window_coordinates):
        labels_crop = full_gt[y:y+512, x:x+512]
        logits = oracle_logits_from_labels(labels_crop)
        stitch.add_window(logits, y=y, x=x)

        if (i + 1) % 64 == 0:
            print(f"  processed windows: {i+1}/256")

    oracle_prediction = stitch.finalize()

    require(
        stitch.coverage_min == 1 and stitch.coverage_max == 9,
        f"真实 6000x6000 coverage 应为 1..9，实际 "
        f"{stitch.coverage_min}..{stitch.coverage_max}"
    )
    require(
        np.array_equal(oracle_prediction, full_gt),
        "oracle mean-logit stitching 后 prediction != full GT"
    )
    print(
        f"  PASS: tile={tile_id}, 256 windows, "
        f"coverage={stitch.coverage_min}..{stitch.coverage_max}, "
        "oracle prediction == full GT"
    )

    print("[4/5] 审计 confusion matrix / global mIoU 计算...")
    evaluator = SplitConfusionEvaluator()
    tile_metrics = evaluator.add_tile(
        tile_id,
        oracle_prediction,
        full_gt,
    )
    summary = evaluator.summary()

    require(tile_metrics["valid_iou_classes"] == 6,
            "真实 Validation tile 未覆盖全部 6 类")
    require(
        abs(float(tile_metrics["miou"]) - 1.0) < 1e-12,
        f"oracle tile mIoU != 1: {tile_metrics['miou']}"
    )
    require(
        abs(float(summary["global"]["miou"]) - 1.0) < 1e-12,
        f"oracle global mIoU != 1: {summary['global']['miou']}"
    )
    require(
        all(abs(float(v) - 1.0) < 1e-12
            for v in tile_metrics["class_iou"]),
        "oracle per-class IoU != 1"
    )

    expected_pixels = 6000 * 6000
    require(
        int(summary["global"]["pixel_count"]) == expected_pixels,
        "global confusion matrix pixel count != 36,000,000"
    )
    print(
        "  PASS: 6-class confusion matrix / per-class IoU / "
        "global mIoU；oracle mIoU=1.0。"
    )

    print("[5/5] 冻结 evaluation_protocol.json...")
    evaluator_file = project_root / "evaluation/potsdam_evaluation.py"
    dataset_file = project_root / "data_pipeline/potsdam_dataset.py"

    require(evaluator_file.is_file(),
            f"缺少 evaluator 文件：{evaluator_file}")
    require(dataset_file.is_file(),
            f"缺少 dataset 文件：{dataset_file}")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "status": "FROZEN_AFTER_AUDIT_PASS",
        "sources": {
            "dataset_protocol": (
                "data/processed/potsdam/dataset_protocol.json"
            ),
            "dataset_protocol_sha256": sha256_file(
                spec.dataset_protocol_path
            ),
            "dataloader_protocol": (
                "data/processed/potsdam/dataloader_protocol.json"
            ),
            "dataloader_protocol_sha256": sha256_file(
                spec.dataloader_protocol_path
            ),
            "dataset_module": "data_pipeline/potsdam_dataset.py",
            "dataset_module_sha256": sha256_file(dataset_file),
            "evaluation_module": "evaluation/potsdam_evaluation.py",
            "evaluation_module_sha256": sha256_file(evaluator_file),
        },
        "sliding_window": {
            "tile_size": [6000, 6000],
            "crop_size": [512, 512],
            "stride": [384, 384],
            "windows_per_tile": 256,
            "coordinate_order": "y-major then x-major",
            "window_coordinates_sha256": (
                spec.window_coordinates_sha256
            ),
            "fusion": "uniform mean-logit accumulation",
            "implementation": (
                "streaming row-band accumulation; mathematically "
                "equivalent to dense sum/count mean logits"
            ),
            "logit_accumulation_dtype": "float32",
            "coverage_count_dtype": "uint16",
            "prediction_dtype": "uint8",
            "expected_coverage_min": 1,
            "expected_coverage_max": 9,
            "simple_overwrite_forbidden": True,
            "tta": False,
        },
        "metrics": {
            "num_classes": 6,
            "class_names": CLASS_NAMES,
            "ignore_index_reserved": 255,
            "expected_ignore_pixels": 0,
            "per_tile": {
                "confusion_matrix": True,
                "class_iou": True,
                "miou": True,
            },
            "primary_split_miou": (
                "sum all full-tile 6x6 confusion matrices, then compute "
                "6 class IoUs and their mean"
            ),
            "patch_miou_forbidden_as_primary": True,
            "mean_tile_miou_forbidden_as_primary": True,
            "absent_class_per_tile_iou": "NaN/null when union=0",
        },
        "audit": {
            "small_conflict_test": (
                "streaming prediction equals dense mean-logit and "
                "differs from overwrite in overlap"
            ),
            "real_oracle_tile": tile_id,
            "real_oracle_windows": 256,
            "real_oracle_coverage": [
                stitch.coverage_min,
                stitch.coverage_max,
            ],
            "real_oracle_prediction_equals_gt": True,
            "real_oracle_miou": 1.0,
            "real_oracle_pixel_count": expected_pixels,
        },
    }

    protocol_path = (
        project_root
        / "data/processed/potsdam/evaluation_protocol.json"
    )
    freeze_info = freeze_protocol(protocol_path, protocol)

    print(f"  evaluation_protocol.json: {freeze_info['action']}")
    print()
    print("FINAL STATUS: PASS")

    return {
        "status": "PASS",
        "module_version": MODULE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "window_coordinates_sha256": spec.window_coordinates_sha256,
        "small_mean_vs_overwrite_test": {
            "streaming_equals_dense": True,
            "overlap_prediction_class": 0,
            "overwrite_would_predict_class": 1,
            "coverage_min": 1,
            "coverage_max": 2,
        },
        "real_oracle": {
            "tile_id": tile_id,
            "windows": 256,
            "coverage_min": stitch.coverage_min,
            "coverage_max": stitch.coverage_max,
            "prediction_equals_gt": True,
            "pixel_count": int(summary["global"]["pixel_count"]),
            "class_iou": tile_metrics["class_iou"],
            "miou": tile_metrics["miou"],
            "confusion_matrix": tile_metrics["confusion_matrix"],
        },
        "metric_definition": summary["primary_metric_definition"],
        "frozen_evaluation_protocol": {
            "path": "data/processed/potsdam/evaluation_protocol.json",
            **freeze_info,
        },
    }


def report_to_text(report: Dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return (
            "Potsdam evaluation protocol audit\n"
            + "=" * 72
            + "\n状态: FAIL\n"
            + f"error_type: {report.get('error_type')}\n"
            + f"error: {report.get('error')}\n"
        )

    oracle = report["real_oracle"]
    lines = [
        "Potsdam full-tile mean-logit / metric evaluation audit",
        "=" * 72,
        "状态: PASS",
        f"模块版本: {report['module_version']}",
        f"protocol: {report['protocol_version']}",
        "",
        "[Sliding window]",
        f"window_coordinates_sha256 = "
        f"{report['window_coordinates_sha256']}",
        "fusion = uniform mean-logit accumulation",
        "",
        "[Mean vs overwrite]",
        "streaming == dense mean-logit = True",
        "overlap mean prediction = class 0",
        "simple overwrite would predict = class 1",
        "",
        "[Real oracle full tile]",
        f"tile = {oracle['tile_id']}",
        f"windows = {oracle['windows']}",
        f"coverage min/max = "
        f"{oracle['coverage_min']} / {oracle['coverage_max']}",
        f"prediction_equals_gt = "
        f"{oracle['prediction_equals_gt']}",
        f"pixel_count = {oracle['pixel_count']}",
        f"class_iou = {oracle['class_iou']}",
        f"mIoU = {oracle['miou']}",
        "",
        "[Metric definition]",
        report["metric_definition"],
        "",
        "[Frozen metadata]",
        f"path = {report['frozen_evaluation_protocol']['path']}",
        f"action = {report['frozen_evaluation_protocol']['action']}",
        f"sha256 = "
        f"{report['frozen_evaluation_protocol']['sha256']}",
        "",
        "FINAL STATUS: PASS",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Potsdam full-tile uniform mean-logit stitching + "
            "global confusion matrix audit"
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录；默认 evaluation/ 的父目录。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行 synthetic + real full-tile oracle audit；不启动模型。",
    )
    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "Evaluation 模块已加载。运行审计：\n"
            "  python evaluation/potsdam_evaluation.py --smoke-test"
        )
        return 0

    project_root = args.project_root.resolve()
    out_dir = (
        project_root
        / "outputs/dataset_check/evaluation_protocol"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    console_path = out_dir / "console.txt"
    json_path = out_dir / "evaluation_protocol_audit.json"
    txt_path = out_dir / "evaluation_protocol_audit.txt"

    with console_path.open("w", encoding="utf-8", buffering=1) as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)

        try:
            report = run_smoke_test(project_root)
            exit_code = 0
        except Exception as e:
            print()
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

    write_json(json_path, report)
    txt_path.write_text(report_to_text(report), encoding="utf-8")

    print(f"console: {console_path}")
    print(f"json: {json_path}")
    print(f"txt: {txt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
