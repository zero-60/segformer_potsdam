#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Model A (RGB baseline) on the frozen Potsdam validation protocol.

This script is intentionally aligned with the project's Dataset Protocol v1:

    512x512 validation windows, stride=384
        -> model full-resolution window logits
        -> overlap regions fused by MEAN LOGIT
        -> one 6000x6000 prediction per tile
        -> one GLOBAL confusion matrix over all validation tiles
        -> global mIoU + six class IoUs

Important
---------
The main metric is NOT patch mIoU and NOT the arithmetic mean of tile mIoUs.
It is computed from the global confusion matrix after full-tile mean-logit
fusion, exactly as required by PotsdamSlidingWindowDataset's protocol.

Expected location
-----------------
Save this file as:
    tools/validate_model_a_rgb.py

Then run from the repository root:
    python tools/validate_model_a_rgb.py

Default checkpoint:
    outputs/training/model_a_rgb/checkpoints/final.pt

Default outputs:
    outputs/evaluation/model_a_rgb/clean_val/
        metrics.json
        per_class_metrics.csv
        per_tile_metrics.jsonl
        confusion_matrix.csv
        run_config.json

Optional:
    python tools/validate_model_a_rgb.py --batch-size 8
    python tools/validate_model_a_rgb.py --checkpoint path/to/checkpoint.pt
    python tools/validate_model_a_rgb.py --save-predictions

Notes on I/O
------------
The validation dataset is tile-major: all 256 windows of one tile are visited
before moving to the next tile. The project's TIFF store therefore only needs
to decode each validation RGBIR tile once when num_workers=0. For that reason
num_workers=0 is the safe/default choice here; unlike shuffled training, it
does not repeatedly jump across tiles.

No NIR tensor is passed to Model A.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Project / experiment constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make `python tools/validate_model_a_rgb.py` work without PYTHONPATH=.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from models.segformer_rgb import build_model_a_rgb


MODEL_NAME = "Model A (RGB baseline)"
NUM_CLASSES = 6
IGNORE_INDEX = 255

CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_a_rgb"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "clean_val"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Model A RGB baseline using full-tile mean-logit fusion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model A training checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for validation metrics/results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Dataset split. Use val for model development/model selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of 512x512 windows inferred at once.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. 0 is recommended because validation is "
            "tile-major and avoids duplicate full-TIFF decoding across workers."
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin host tensors before CUDA transfer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Examples: "cuda", "cuda:0", "cpu".',
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA FP16 autocast during inference.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save each full-tile class prediction as uint8 .npy. "
            "Disabled by default to avoid unnecessary disk use."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=8,
        help="Print progress every N inference batches within each tile.",
    )
    parser.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
        help=(
            "Rows processed at once when building the CPU confusion matrix. "
            "This bounds temporary RAM use."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")
    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    return args


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def resolve_from_project(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def get_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Fix the CUDA environment or pass --device cpu for debugging."
        )

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    return device


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(payload, ensure_ascii=False, default=str)
            + "\n"
        )


def unwrap_checkpoint_state_dict(
    checkpoint_obj: Any,
) -> tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    """
    Accept this project's training checkpoint and, as a convenience, a raw
    state_dict. Returns (state_dict, metadata).
    """
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping/dict. "
            f"Got {type(checkpoint_obj)!r}."
        )

    if "model" in checkpoint_obj:
        state = checkpoint_obj["model"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint['model'] is not a state_dict mapping")
        metadata = {
            k: v
            for k, v in checkpoint_obj.items()
            if k != "model"
        }
        return state, metadata

    # Raw state_dict path.
    if checkpoint_obj and all(
        isinstance(k, str) for k in checkpoint_obj.keys()
    ):
        tensor_like = all(
            isinstance(v, (torch.Tensor, torch.nn.Parameter))
            for v in checkpoint_obj.values()
        )
        if tensor_like:
            return checkpoint_obj, {}

    raise KeyError(
        "Could not find model state_dict. Expected checkpoint['model'] "
        "or a raw state_dict."
    )


def validate_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    """
    Fail on obvious experiment mix-ups, but remain compatible with older
    checkpoints that may not contain the newer metadata fields.
    """
    ckpt_model_name = metadata.get("model_name")
    if ckpt_model_name is not None and ckpt_model_name != MODEL_NAME:
        raise RuntimeError(
            "Checkpoint appears to belong to a different model: "
            f"{ckpt_model_name!r} != {MODEL_NAME!r}"
        )

    protocol = metadata.get("protocol")
    if isinstance(protocol, Mapping):
        protocol_model = protocol.get("model")
        if protocol_model is not None and protocol_model != "A_RGB":
            raise RuntimeError(
                "Checkpoint protocol is not Model A: "
                f"protocol.model={protocol_model!r}"
            )

        modalities = protocol.get("input_modalities")
        if modalities is not None and list(modalities) != ["RGB"]:
            raise RuntimeError(
                "Checkpoint protocol is not RGB-only: "
                f"input_modalities={modalities!r}"
            )

        n_channels = protocol.get("num_input_channels")
        if n_channels is not None and int(n_channels) != 3:
            raise RuntimeError(
                "Checkpoint protocol input channels != 3: "
                f"{n_channels}"
            )

        n_classes = protocol.get("num_classes")
        if n_classes is not None and int(n_classes) != NUM_CLASSES:
            raise RuntimeError(
                "Checkpoint protocol num_classes != 6: "
                f"{n_classes}"
            )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_from_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int,
    chunk_rows: int,
) -> np.ndarray:
    """
    Build a num_classes x num_classes confusion matrix.

    Rows = ground truth
    Cols = prediction

    Work is chunked by rows so a 6000x6000 tile does not create a very large
    temporary int64 index vector all at once.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shapes differ: "
            f"{prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 2:
        raise ValueError(
            f"Expected 2D full-tile prediction, got {prediction.shape}"
        )

    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    height = target.shape[0]

    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)

        gt = target[y0:y1]
        pred = prediction[y0:y1]

        valid = (
            (gt != ignore_index)
            & (gt >= 0)
            & (gt < num_classes)
        )

        if not np.any(valid):
            continue

        gt_valid = gt[valid].astype(np.int64, copy=False)
        pred_valid = pred[valid].astype(np.int64, copy=False)

        if pred_valid.size and (
            pred_valid.min() < 0
            or pred_valid.max() >= num_classes
        ):
            raise RuntimeError(
                "Prediction contains an out-of-range class id."
            )

        ids = num_classes * gt_valid + pred_valid
        hist += np.bincount(
            ids,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    return hist


def metrics_from_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"Confusion matrix must be {NUM_CLASSES}x{NUM_CLASSES}, "
            f"got {confusion.shape}"
        )

    cm = confusion.astype(np.float64, copy=False)

    tp = np.diag(cm)
    gt_support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)

    fp = pred_support - tp
    fn = gt_support - tp
    union = tp + fp + fn

    iou = np.divide(
        tp,
        union,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=union > 0,
    )

    class_accuracy = np.divide(
        tp,
        gt_support,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=gt_support > 0,
    )

    valid_iou = np.isfinite(iou)
    valid_acc = np.isfinite(class_accuracy)

    total = cm.sum()
    pixel_accuracy = float(tp.sum() / total) if total > 0 else float("nan")
    mean_accuracy = (
        float(np.mean(class_accuracy[valid_acc]))
        if np.any(valid_acc)
        else float("nan")
    )
    miou = (
        float(np.mean(iou[valid_iou]))
        if np.any(valid_iou)
        else float("nan")
    )

    per_class: List[Dict[str, Any]] = []
    for cid, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": cid,
                "class_name": name,
                "iou": None if not np.isfinite(iou[cid]) else float(iou[cid]),
                "accuracy": (
                    None
                    if not np.isfinite(class_accuracy[cid])
                    else float(class_accuracy[cid])
                ),
                "tp": int(tp[cid]),
                "fp": int(fp[cid]),
                "fn": int(fn[cid]),
                "gt_pixels": int(gt_support[cid]),
                "pred_pixels": int(pred_support[cid]),
                "union_pixels": int(union[cid]),
            }
        )

    return {
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": mean_accuracy,
        "valid_pixels": int(total),
        "per_class": per_class,
    }


def write_confusion_csv(
    path: Path,
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["gt\\pred", *class_names]
        )
        for name, row in zip(class_names, confusion.tolist()):
            writer.writerow([name, *row])


def write_per_class_csv(
    path: Path,
    per_class: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "class_id",
        "class_name",
        "iou",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "gt_pixels",
        "pred_pixels",
        "union_pixels",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_class:
            writer.writerow({key: row.get(key) for key in fields})


# ---------------------------------------------------------------------------
# Inference / full-tile fusion
# ---------------------------------------------------------------------------

def validate_batch_metadata(
    batch: Mapping[str, Any],
    expected_tile_id: str,
) -> None:
    required = {
        "rgb",
        "tile_id",
        "window_index",
        "x",
        "y",
        "height",
        "width",
    }
    missing = sorted(required - set(batch.keys()))
    if missing:
        raise RuntimeError(
            f"Validation batch is missing keys: {missing}"
        )

    tile_ids = list(batch["tile_id"])
    if not tile_ids:
        raise RuntimeError("Empty validation batch.")

    if any(str(t) != expected_tile_id for t in tile_ids):
        raise RuntimeError(
            "A per-tile validation DataLoader unexpectedly mixed tile ids: "
            f"expected={expected_tile_id}, got={tile_ids}"
        )

    rgb = batch["rgb"]
    if not isinstance(rgb, torch.Tensor):
        raise TypeError("batch['rgb'] must be a torch.Tensor")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise RuntimeError(
            "Model A validation must be RGB-only [B,3,H,W], "
            f"got {tuple(rgb.shape)}"
        )
    if rgb.dtype != torch.float32:
        raise RuntimeError(
            "Potsdam validation RGB is expected to be normalized float32, "
            f"got {rgb.dtype}"
        )


def infer_one_tile(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    tile_id: str,
    tile_index: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Infer all frozen sliding windows for one tile and mean-fuse logits.

    Accumulators live on GPU by default:
        logits_sum : [6, H, W] float32
        coverage   : [H, W]    float32

    For 6000x6000, this is ~1.0 GiB total, which is modest on the target GPU
    and avoids repeatedly copying window logits to CPU.
    """
    windows_per_tile = len(dataset.window_coordinates)
    start_index = tile_index * windows_per_tile
    stop_index = start_index + windows_per_tile

    subset = Subset(
        dataset,
        range(start_index, stop_index),
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
    )

    tile_size = int(dataset.spec.tile_size)
    crop_size = int(dataset.spec.crop_size)

    logits_sum = torch.zeros(
        (NUM_CLASSES, tile_size, tile_size),
        dtype=torch.float32,
        device=device,
    )
    coverage = torch.zeros(
        (tile_size, tile_size),
        dtype=torch.float32,
        device=device,
    )

    tile_start = time.time()
    windows_seen = 0

    for batch_idx, batch in enumerate(loader):
        validate_batch_metadata(batch, expected_tile_id=tile_id)

        rgb = batch["rgb"].to(
            device,
            non_blocking=(pin_memory and device.type == "cuda"),
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(rgb)

        if (
            logits.ndim != 4
            or logits.shape[1] != NUM_CLASSES
            or tuple(logits.shape[-2:]) != (crop_size, crop_size)
        ):
            raise RuntimeError(
                "Unexpected Model A validation logits shape: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all().item():
            raise FloatingPointError(
                f"Non-finite logits for tile={tile_id}, batch={batch_idx}"
            )

        # Always accumulate in FP32 even if inference uses FP16 autocast.
        logits = logits.float()

        xs = batch["x"]
        ys = batch["y"]

        batch_count = int(logits.shape[0])

        for j in range(batch_count):
            x = int(xs[j])
            y = int(ys[j])

            if not (
                0 <= x <= tile_size - crop_size
                and 0 <= y <= tile_size - crop_size
            ):
                raise RuntimeError(
                    f"Out-of-range sliding window: "
                    f"tile={tile_id}, x={x}, y={y}"
                )

            logits_sum[
                :,
                y:y + crop_size,
                x:x + crop_size,
            ].add_(logits[j])

            coverage[
                y:y + crop_size,
                x:x + crop_size,
            ].add_(1.0)

        windows_seen += batch_count

        if (
            batch_idx % log_every == 0
            or batch_idx + 1 == len(loader)
        ):
            print(
                f"  tile {tile_id} | "
                f"batch {batch_idx + 1:03d}/{len(loader):03d} | "
                f"windows {windows_seen:03d}/{windows_per_tile:03d}",
                flush=True,
            )

        del logits, rgb

    if windows_seen != windows_per_tile:
        raise RuntimeError(
            f"Tile {tile_id}: expected {windows_per_tile} windows, "
            f"got {windows_seen}"
        )

    coverage_min = float(coverage.min().item())
    coverage_max = float(coverage.max().item())

    if coverage_min <= 0:
        zero_pixels = int((coverage == 0).sum().item())
        raise RuntimeError(
            f"Tile {tile_id} has {zero_pixels} uncovered pixels. "
            "Frozen sliding-window coordinates are not covering the tile."
        )

    # Mean-logit fusion.
    logits_sum.div_(coverage.unsqueeze(0))

    # Argmax is initially int64; cast to uint8 before copying to CPU.
    prediction = (
        logits_sum
        .argmax(dim=0)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )

    elapsed = time.time() - tile_start

    del logits_sum, coverage

    if device.type == "cuda":
        # Makes peak/runtime reporting cleaner between tiles. This is not
        # needed for correctness.
        torch.cuda.empty_cache()

    info = {
        "tile_id": tile_id,
        "windows": windows_seen,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "inference_seconds": elapsed,
    }

    return prediction, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_from_project(args.checkpoint)
    output_dir = resolve_from_project(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = get_device(args.device)
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 86)
    print(f"{MODEL_NAME} - CLEAN {args.split.upper()} VALIDATION")
    print("=" * 86)
    print(f"project root : {PROJECT_ROOT}")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"output dir   : {output_dir}")
    print(f"device       : {device}")
    print(f"AMP          : {amp_enabled}")
    print(f"batch size   : {args.batch_size}")
    print(f"workers      : {args.num_workers}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("[1] loading frozen sliding-window dataset")
    dataset = PotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split=args.split,
    )

    tile_ids = list(dataset.tile_ids)
    windows_per_tile = len(dataset.window_coordinates)

    expected_len = len(tile_ids) * windows_per_tile
    if len(dataset) != expected_len:
        raise RuntimeError(
            f"Dataset length mismatch: len={len(dataset)}, "
            f"tiles={len(tile_ids)}, windows/tile={windows_per_tile}"
        )

    if windows_per_tile != 256:
        raise RuntimeError(
            f"Frozen protocol expects 256 windows/tile, got "
            f"{windows_per_tile}"
        )

    print(
        f"  split={args.split} | tiles={len(tile_ids)} | "
        f"windows/tile={windows_per_tile} | total={len(dataset)}"
    )
    print(f"  tile ids: {tile_ids}")

    # ------------------------------------------------------------------
    # Model + trained checkpoint
    # ------------------------------------------------------------------
    print("[2] building Model A architecture")
    model, model_meta = build_model_a_rgb(PROJECT_ROOT)

    print("[3] loading trained checkpoint")
    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, checkpoint_meta = unwrap_checkpoint_state_dict(
        checkpoint_obj
    )
    validate_checkpoint_metadata(checkpoint_meta)

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    # strict=True should already guarantee these are empty, but make it visible.
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    ckpt_epoch = checkpoint_meta.get("epoch")
    ckpt_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get("step"),
    )

    print(
        "  checkpoint loaded strictly"
        + (
            f" | epoch={int(ckpt_epoch) + 1}"
            if ckpt_epoch is not None
            else ""
        )
        + (
            f" | global_step={ckpt_step}"
            if ckpt_step is not None
            else ""
        )
    )

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------
    run_config = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "condition": "Clean",
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": ckpt_epoch,
        "checkpoint_global_step": ckpt_step,
        "input_modalities": ["RGB"],
        "nir_used": False,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "fusion": "full-tile mean-logit sliding-window fusion",
        "tile_size": int(dataset.spec.tile_size),
        "crop_size": int(dataset.spec.crop_size),
        "windows_per_tile": windows_per_tile,
        "validation_tiles": tile_ids,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_meta": model_meta,
    }
    write_json(output_dir / "run_config.json", run_config)

    per_tile_path = output_dir / "per_tile_metrics.jsonl"
    # Avoid silently appending a second run to the same result file.
    if per_tile_path.exists():
        per_tile_path.unlink()

    prediction_dir = output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Full-tile validation
    # ------------------------------------------------------------------
    print("[4] full-tile mean-logit validation")

    global_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    per_tile_results: List[Dict[str, Any]] = []
    validation_start = time.time()

    for tile_index, tile_id in enumerate(tile_ids):
        print(
            f"[tile {tile_index + 1}/{len(tile_ids)}] {tile_id}",
            flush=True,
        )

        prediction, tile_runtime = infer_one_tile(
            model=model,
            dataset=dataset,
            tile_id=tile_id,
            tile_index=tile_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            device=device,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
        )

        if prediction.shape != (
            dataset.spec.tile_size,
            dataset.spec.tile_size,
        ):
            raise RuntimeError(
                f"Full prediction shape error for {tile_id}: "
                f"{prediction.shape}"
            )

        if prediction.dtype != np.uint8:
            raise RuntimeError(
                f"Prediction dtype should be uint8, got {prediction.dtype}"
            )

        unique_pred = np.unique(prediction)
        if not np.all(
            (unique_pred >= 0)
            & (unique_pred < NUM_CLASSES)
        ):
            raise RuntimeError(
                f"Tile {tile_id} prediction has invalid classes: "
                f"{unique_pred.tolist()}"
            )

        # Dataset protocol explicitly requires full-tile GT only AFTER
        # full-tile logit fusion.
        target_t = dataset.load_full_label(tile_id)
        if target_t.dtype != torch.int64:
            raise RuntimeError(
                f"Full GT dtype must be int64, got {target_t.dtype}"
            )

        target = target_t.numpy()

        tile_confusion = confusion_from_prediction(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            chunk_rows=args.confusion_chunk_rows,
        )

        global_confusion += tile_confusion
        tile_metrics = metrics_from_confusion(
            tile_confusion,
            CLASS_NAMES,
        )

        tile_record = {
            **tile_runtime,
            "miou": tile_metrics["miou"],
            "pixel_accuracy": tile_metrics["pixel_accuracy"],
            "mean_class_accuracy": tile_metrics["mean_class_accuracy"],
            "valid_pixels": tile_metrics["valid_pixels"],
            "per_class_iou": {
                item["class_name"]: item["iou"]
                for item in tile_metrics["per_class"]
            },
        }

        per_tile_results.append(tile_record)
        append_jsonl(per_tile_path, tile_record)

        if args.save_predictions:
            np.save(
                prediction_dir / f"{tile_id}_pred.npy",
                prediction,
                allow_pickle=False,
            )

        print(
            f"  fused tile mIoU={tile_metrics['miou']:.6f} | "
            f"pixel_acc={tile_metrics['pixel_accuracy']:.6f} | "
            f"coverage={tile_runtime['coverage_min']:.0f}"
            f"..{tile_runtime['coverage_max']:.0f} | "
            f"time={tile_runtime['inference_seconds']:.1f}s",
            flush=True,
        )

        del prediction, target, target_t

    # ------------------------------------------------------------------
    # Global metrics: this is the primary result.
    # ------------------------------------------------------------------
    print("[5] computing GLOBAL confusion-matrix metrics")

    global_metrics = metrics_from_confusion(
        global_confusion,
        CLASS_NAMES,
    )

    validation_seconds = time.time() - validation_start

    result = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "condition": "Clean",
        "split": args.split,
        "metric_scope": (
            "GLOBAL confusion matrix after full-tile mean-logit fusion"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": ckpt_epoch,
        "checkpoint_global_step": ckpt_step,
        "miou": global_metrics["miou"],
        "pixel_accuracy": global_metrics["pixel_accuracy"],
        "mean_class_accuracy": global_metrics["mean_class_accuracy"],
        "valid_pixels": global_metrics["valid_pixels"],
        "per_class": global_metrics["per_class"],
        "confusion_matrix": global_confusion.tolist(),
        "tiles": per_tile_results,
        "num_tiles": len(tile_ids),
        "windows_per_tile": windows_per_tile,
        "total_windows": len(dataset),
        "validation_seconds": validation_seconds,
    }

    write_json(output_dir / "metrics.json", result)
    write_per_class_csv(
        output_dir / "per_class_metrics.csv",
        global_metrics["per_class"],
    )
    write_confusion_csv(
        output_dir / "confusion_matrix.csv",
        global_confusion,
        CLASS_NAMES,
    )

    # ------------------------------------------------------------------
    # Human-readable final report
    # ------------------------------------------------------------------
    print("=" * 86)
    print(
        f"RESULT | {MODEL_NAME} | Clean {args.split} | "
        "GLOBAL full-tile metrics"
    )
    print("=" * 86)
    print(f"mIoU               : {global_metrics['miou']:.6f}")
    print(
        f"Pixel Accuracy      : "
        f"{global_metrics['pixel_accuracy']:.6f}"
    )
    print(
        f"Mean Class Accuracy : "
        f"{global_metrics['mean_class_accuracy']:.6f}"
    )
    print(f"Valid pixels        : {global_metrics['valid_pixels']:,}")
    print("-" * 86)

    for item in global_metrics["per_class"]:
        iou = item["iou"]
        iou_text = "NaN" if iou is None else f"{iou:.6f}"
        print(
            f"class {item['class_id']} | "
            f"{item['class_name']:<20} | IoU={iou_text} | "
            f"GT={item['gt_pixels']:,}"
        )

    print("-" * 86)
    print(f"metrics             : {output_dir / 'metrics.json'}")
    print(
        f"per-class CSV       : "
        f"{output_dir / 'per_class_metrics.csv'}"
    )
    print(
        f"confusion matrix    : "
        f"{output_dir / 'confusion_matrix.csv'}"
    )
    print(f"per-tile diagnostics: {per_tile_path}")
    print(f"runtime             : {validation_seconds:.1f}s")
    print("=" * 86)


if __name__ == "__main__":
    main()
