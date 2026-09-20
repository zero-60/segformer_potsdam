#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B complete validation under the FINAL RGB Degradation Protocol v2.

Model B
-------
RGB + clean NIR -> direct 4-channel SegFormer-B0.

This script runs / resumes:
1. Clean validation.
2. Gaussian Noise L1/L2/L3.
3. Gaussian Blur L1/L2/L3.
4. RGB Underexposure L1/L2/L3.
5. A-vs-B comparison and NIR Gain analysis.

Evaluation is identical in structure to Model A:
    512x512 windows
        -> full-resolution logits
        -> full-tile MEAN-LOGIT fusion
        -> one 6000x6000 prediction per tile
        -> one GLOBAL confusion matrix over all six val tiles
        -> global mIoU + six class IoUs

NIR is ALWAYS clean in degraded conditions.

Before running
--------------
A) Replace:
    evaluation/rgb_degradation_protocol.py
with the FINAL v2 implementation.

B) Run once:
    python tools/finalize_model_a_protocol_v2.py

C) Then run:
    python tools/validate_model_b_v2.py

Expected Model B checkpoint:
    outputs/training/model_b_rgbnir/checkpoints/final.pt

Outputs
-------
outputs/evaluation/model_b_rgbnir/
├── clean_val/
│   ├── metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   └── per_tile_metrics.jsonl
│
└── robustness_val_v2/
    ├── degradation_protocol.json
    ├── robustness_summary.json
    ├── robustness_summary.csv
    ├── model_a_vs_b_comparison.json
    ├── model_a_vs_b_comparison.csv
    ├── per_class_nir_gain.csv
    ├── gaussian_noise_L1/
    ├── ...
    └── rgb_underexposure_L3/

Definitions
-----------
NIR Gain(condition):
    mIoU_B(condition) - mIoU_A(condition)

Drop_X(condition):
    mIoU_X(Clean) - mIoU_X(condition)

Drop Reduction:
    Drop_A - Drop_B
Positive means Model B loses less mIoU than RGB-only Model A.

Relative Drop Reduction (%):
    100 * (Drop_A - Drop_B) / Drop_A
Positive means NIR/direct fusion recovers part of Model A's degradation loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from data_pipeline.potsdam_dataset import (
    PotsdamSlidingWindowDataset,
)
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL,
    DEGRADATION_PROTOCOL_VERSION,
    IMPLEMENTATION_REVISION,
    DegradedPotsdamSlidingWindowDataset,
    available_corruptions,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    write_degradation_protocol,
)
from models.segformer_rgbnir import (
    INPUT_CHANNELS,
    NUM_CLASSES,
    build_model_b_rgbnir,
)

try:
    from validate_model_a_rgb import (
        CLASS_NAMES,
        IGNORE_INDEX,
        confusion_from_prediction,
        metrics_from_confusion,
        write_confusion_csv,
        write_per_class_csv,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import tools/validate_model_a_rgb.py. "
        "Keep the successful Model A Clean validator in tools/."
    ) from exc


MODEL_ID = "B_RGBNIR_4CH"
MODEL_NAME = "Model B (RGB+NIR direct 4-channel)"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_b_rgbnir"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_b_rgbnir"
)

DEFAULT_A_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "clean_val"
    / "metrics.json"
)

DEFAULT_A_V1_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val"
)

DEFAULT_A_V2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val_v2"
)

CONDITION_ORDER = [
    ("gaussian_noise", "L1"),
    ("gaussian_noise", "L2"),
    ("gaussian_noise", "L3"),
    ("gaussian_blur", "L1"),
    ("gaussian_blur", "L2"),
    ("gaussian_blur", "L3"),
    ("rgb_underexposure", "L1"),
    ("rgb_underexposure", "L2"),
    ("rgb_underexposure", "L3"),
]


# ---------------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Model B Clean + final Protocol-v2 robustness and "
            "compare against Model A."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--model-a-clean",
        type=Path,
        default=DEFAULT_A_CLEAN,
    )
    parser.add_argument(
        "--model-a-v1-dir",
        type=Path,
        default=DEFAULT_A_V1_DIR,
    )
    parser.add_argument(
        "--model-a-v2-dir",
        type=Path,
        default=DEFAULT_A_V2_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
    )
    parser.add_argument(
        "--force-robustness",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")
    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    return args


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return obj


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
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


def append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def get_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    if (
        device.type == "cuda"
        and device.index is not None
    ):
        torch.cuda.set_device(device.index)

    return device


# ---------------------------------------------------------------------------
# Model B checkpoint
# ---------------------------------------------------------------------------

def unwrap_model_b_checkpoint(
    checkpoint: Any,
) -> tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping, got {type(checkpoint)!r}."
        )

    if "model" not in checkpoint:
        raise KeyError(
            "Expected Model B training checkpoint with checkpoint['model']."
        )

    state = checkpoint["model"]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint['model'] is not a state_dict.")

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model"
    }

    return state, metadata


def validate_model_b_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    model_id = metadata.get("model_id")
    if model_id is not None and model_id != MODEL_ID:
        raise RuntimeError(
            f"Checkpoint model_id={model_id!r}, expected {MODEL_ID!r}."
        )

    model_name = metadata.get("model_name")
    if (
        model_name is not None
        and model_name != MODEL_NAME
    ):
        raise RuntimeError(
            f"Checkpoint model_name={model_name!r}, expected {MODEL_NAME!r}."
        )

    protocol = metadata.get("protocol")
    if isinstance(protocol, Mapping):
        if protocol.get("model") != MODEL_ID:
            raise RuntimeError(
                f"Checkpoint protocol model={protocol.get('model')!r}."
            )

        if list(
            protocol.get("input_modalities", [])
        ) != ["RGB", "NIR"]:
            raise RuntimeError(
                "Checkpoint is not RGB+NIR."
            )

        if int(
            protocol.get("num_input_channels", -1)
        ) != INPUT_CHANNELS:
            raise RuntimeError(
                "Checkpoint protocol is not 4-channel."
            )

        if bool(protocol.get("quality_gate", True)):
            raise RuntimeError(
                "Model B checkpoint unexpectedly uses a Quality Gate."
            )

        if bool(protocol.get("dual_encoder", True)):
            raise RuntimeError(
                "Model B checkpoint unexpectedly uses a dual encoder."
            )


# ---------------------------------------------------------------------------
# Full-tile Model B inference
# ---------------------------------------------------------------------------

def validate_batch(
    batch: Mapping[str, Any],
    *,
    tile_id: str,
) -> None:
    required = {
        "rgb",
        "nir",
        "tile_id",
        "x",
        "y",
    }

    missing = sorted(
        required - set(batch.keys())
    )
    if missing:
        raise RuntimeError(
            f"Validation batch missing keys: {missing}"
        )

    rgb = batch["rgb"]
    nir = batch["nir"]

    if (
        not isinstance(rgb, torch.Tensor)
        or rgb.ndim != 4
        or rgb.shape[1] != 3
    ):
        raise RuntimeError(
            f"RGB must be [B,3,H,W], got {getattr(rgb, 'shape', None)}."
        )

    if (
        not isinstance(nir, torch.Tensor)
        or nir.ndim != 4
        or nir.shape[1] != 1
    ):
        raise RuntimeError(
            f"NIR must be [B,1,H,W], got {getattr(nir, 'shape', None)}."
        )

    if (
        rgb.shape[0] != nir.shape[0]
        or rgb.shape[-2:] != nir.shape[-2:]
    ):
        raise RuntimeError(
            f"RGB/NIR shape mismatch: {tuple(rgb.shape)} vs {tuple(nir.shape)}."
        )

    tile_ids = list(batch["tile_id"])
    if any(str(x) != tile_id for x in tile_ids):
        raise RuntimeError(
            f"Per-tile loader mixed tile ids: expected={tile_id}, got={tile_ids}"
        )


def infer_one_tile_b(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    tile_id: str,
    tile_index: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    windows_per_tile = len(
        dataset.window_coordinates
    )

    start = (
        tile_index
        * windows_per_tile
    )
    stop = (
        start
        + windows_per_tile
    )

    subset = Subset(
        dataset,
        range(start, stop),
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    tile_size = int(
        dataset.spec.tile_size
    )
    crop_size = int(
        dataset.spec.crop_size
    )

    logits_sum = torch.zeros(
        (
            NUM_CLASSES,
            tile_size,
            tile_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    coverage = torch.zeros(
        (
            tile_size,
            tile_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    started = time.time()
    windows_seen = 0

    for batch_idx, batch in enumerate(loader):
        validate_batch(
            batch,
            tile_id=tile_id,
        )

        rgb = batch["rgb"].to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        nir = batch["nir"].to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(
                    rgb,
                    nir,
                )

        if (
            logits.ndim != 4
            or logits.shape[1] != NUM_CLASSES
            or tuple(logits.shape[-2:])
            != (
                crop_size,
                crop_size,
            )
        ):
            raise RuntimeError(
                f"Unexpected Model B logits: {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all().item():
            raise FloatingPointError(
                f"Non-finite logits: tile={tile_id}, batch={batch_idx}"
            )

        logits = logits.float()

        xs = batch["x"]
        ys = batch["y"]

        for j in range(
            int(logits.shape[0])
        ):
            x = int(xs[j])
            y = int(ys[j])

            logits_sum[
                :,
                y:y + crop_size,
                x:x + crop_size,
            ].add_(
                logits[j]
            )

            coverage[
                y:y + crop_size,
                x:x + crop_size,
            ].add_(1.0)

        windows_seen += int(
            logits.shape[0]
        )

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

        del rgb, nir, logits

    if windows_seen != windows_per_tile:
        raise RuntimeError(
            f"{tile_id}: expected {windows_per_tile} windows, got {windows_seen}."
        )

    coverage_min = float(
        coverage.min().item()
    )
    coverage_max = float(
        coverage.max().item()
    )

    if coverage_min <= 0.0:
        raise RuntimeError(
            f"{tile_id}: full tile has uncovered pixels."
        )

    logits_sum.div_(
        coverage.unsqueeze(0)
    )

    prediction = (
        logits_sum
        .argmax(dim=0)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )

    elapsed = (
        time.time()
        - started
    )

    del logits_sum, coverage

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return prediction, {
        "tile_id": tile_id,
        "windows": windows_seen,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "inference_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def degradation_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: DegradedPotsdamSlidingWindowDataset,
) -> None:
    clean = clean_dataset[0]
    degraded = degraded_dataset[0]

    for key in (
        "tile_id",
        "window_index",
        "x",
        "y",
        "height",
        "width",
    ):
        if clean[key] != degraded[key]:
            raise RuntimeError(
                f"Degradation changed frozen metadata key={key}."
            )

    if not torch.equal(
        clean["nir"],
        degraded["nir"],
    ):
        diff = float(
            (
                clean["nir"]
                - degraded["nir"]
            )
            .abs()
            .max()
            .item()
        )
        raise RuntimeError(
            "NIR changed under RGB-only degradation. "
            f"max_abs_diff={diff}"
        )

    if torch.equal(
        clean["rgb"],
        degraded["rgb"],
    ):
        raise RuntimeError(
            "Degraded RGB equals clean RGB on probe window."
        )

    change = float(
        (
            clean["rgb"]
            - degraded["rgb"]
        )
        .abs()
        .mean()
        .item()
    )

    print(
        "[probe] PASS | same window | NIR bit-identical | "
        f"RGB normalized mean_abs_change={change:.6f}"
    )


def evaluate_dataset(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    condition: str,
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_global_step: Optional[int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    confusion_chunk_rows: int,
    save_predictions: bool,
    corruption: Optional[str] = None,
    severity_level: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_tile_path = (
        output_dir
        / "per_tile_metrics.jsonl"
    )
    if per_tile_path.exists():
        per_tile_path.unlink()

    pred_dir = (
        output_dir
        / "predictions"
    )
    if save_predictions:
        pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    tile_ids = list(
        dataset.tile_ids
    )
    windows_per_tile = len(
        dataset.window_coordinates
    )

    if (
        len(tile_ids) != 6
        or windows_per_tile != 256
    ):
        raise RuntimeError(
            "Frozen validation protocol changed: "
            f"tiles={len(tile_ids)}, windows/tile={windows_per_tile}"
        )

    global_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    per_tile: List[Dict[str, Any]] = []
    started = time.time()

    for tile_index, tile_id in enumerate(
        tile_ids
    ):
        print(
            f"[{condition}] tile "
            f"{tile_index + 1}/{len(tile_ids)} | {tile_id}",
            flush=True,
        )

        prediction, runtime = infer_one_tile_b(
            model=model,
            dataset=dataset,
            tile_id=tile_id,
            tile_index=tile_index,
            batch_size=batch_size,
            device=device,
            amp_enabled=amp_enabled,
            log_every=log_every,
        )

        target_t = dataset.load_full_label(
            tile_id
        )
        target = target_t.numpy()

        tile_cm = confusion_from_prediction(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            chunk_rows=confusion_chunk_rows,
        )

        global_confusion += tile_cm

        tile_metrics = metrics_from_confusion(
            tile_cm,
            CLASS_NAMES,
        )

        record = {
            **runtime,
            "condition": condition,
            "miou": tile_metrics["miou"],
            "pixel_accuracy": tile_metrics["pixel_accuracy"],
            "mean_class_accuracy": tile_metrics["mean_class_accuracy"],
            "valid_pixels": tile_metrics["valid_pixels"],
            "per_class_iou": {
                row["class_name"]: row["iou"]
                for row in tile_metrics["per_class"]
            },
        }

        if corruption is not None:
            record["corruption"] = corruption
            record["severity_level"] = severity_level
            record["severity_parameters"] = condition_spec(
                corruption,
                str(severity_level),
            )

        per_tile.append(record)
        append_jsonl(
            per_tile_path,
            record,
        )

        if save_predictions:
            np.save(
                pred_dir
                / f"{tile_id}_pred.npy",
                prediction,
                allow_pickle=False,
            )

        print(
            f"  tile mIoU={tile_metrics['miou']:.6f} | "
            f"pixel_acc={tile_metrics['pixel_accuracy']:.6f} | "
            f"time={runtime['inference_seconds']:.1f}s"
        )

        del prediction, target, target_t

        if isinstance(
            dataset,
            DegradedPotsdamSlidingWindowDataset,
        ):
            dataset.clear_degradation_cache()

    metrics = metrics_from_confusion(
        global_confusion,
        CLASS_NAMES,
    )

    elapsed = (
        time.time()
        - started
    )

    result: Dict[str, Any] = {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "condition": condition,
        "split": "val",
        "metric_scope": (
            "GLOBAL confusion matrix after full-tile mean-logit fusion"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "input_modalities": ["RGB", "NIR"],
        "nir_used": True,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "miou": metrics["miou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_class_accuracy": metrics["mean_class_accuracy"],
        "valid_pixels": metrics["valid_pixels"],
        "per_class": metrics["per_class"],
        "confusion_matrix": global_confusion.tolist(),
        "tiles": per_tile,
        "num_tiles": len(tile_ids),
        "windows_per_tile": windows_per_tile,
        "total_windows": len(dataset),
        "validation_seconds": elapsed,
    }

    if corruption is not None:
        result.update(
            {
                "corruption": corruption,
                "severity_level": severity_level,
                "severity_rank": int(
                    str(severity_level)[1:]
                ),
                "severity_parameters": condition_spec(
                    corruption,
                    str(severity_level),
                ),
                "degradation_protocol_version": (
                    DEGRADATION_PROTOCOL_VERSION
                ),
                "degradation_implementation_revision": (
                    IMPLEMENTATION_REVISION
                ),
                "degradation_protocol_sha256": (
                    degradation_protocol_sha256()
                ),
                "rgb_degraded": True,
                "nir_source": "clean / unchanged",
            }
        )

    write_json(
        output_dir / "metrics.json",
        result,
    )
    write_per_class_csv(
        output_dir / "per_class_metrics.csv",
        metrics["per_class"],
    )
    write_confusion_csv(
        output_dir / "confusion_matrix.csv",
        global_confusion,
        CLASS_NAMES,
    )

    return result


def load_existing_clean(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_global_step: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = read_json(path)
    except Exception:
        return None

    checks = [
        obj.get("model") == MODEL_ID,
        obj.get("condition") == "Clean",
        obj.get("split") == "val",
        Path(str(obj.get("checkpoint", ""))).name
        == checkpoint_path.name,
    ]

    if checkpoint_global_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(checkpoint_global_step)
        )

    if all(checks):
        return obj

    return None


def load_existing_degraded(
    path: Path,
    *,
    condition: str,
    checkpoint_path: Path,
    checkpoint_global_step: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = read_json(path)
    except Exception:
        return None

    checks = [
        obj.get("model") == MODEL_ID,
        obj.get("condition") == condition,
        obj.get("split") == "val",
        obj.get("degradation_protocol_sha256")
        == degradation_protocol_sha256(),
        int(
            obj.get(
                "degradation_implementation_revision",
                -1,
            )
        )
        == IMPLEMENTATION_REVISION,
        Path(str(obj.get("checkpoint", ""))).name
        == checkpoint_path.name,
    ]

    if checkpoint_global_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(checkpoint_global_step)
        )

    if all(checks):
        return obj

    return None


# ---------------------------------------------------------------------------
# Model A references / comparison
# ---------------------------------------------------------------------------

def validate_model_a_references(
    *,
    a_clean: Mapping[str, Any],
    a_v2_summary: Mapping[str, Any],
) -> None:
    if (
        a_clean.get("model") != "A_RGB"
        or a_clean.get("condition") != "Clean"
        or a_clean.get("split") != "val"
    ):
        raise RuntimeError(
            "Model A Clean reference is invalid."
        )

    if a_v2_summary.get("model") != "A_RGB":
        raise RuntimeError(
            "Model A v2 robustness summary is invalid."
        )

    if (
        a_v2_summary.get(
            "degradation_protocol_sha256"
        )
        != degradation_protocol_sha256()
    ):
        raise RuntimeError(
            "Model A v2 summary is NOT finalized to the current protocol.\n"
            "Run: python tools/finalize_model_a_protocol_v2.py"
        )

    if int(
        a_v2_summary.get(
            "degradation_implementation_revision",
            -1,
        )
    ) != IMPLEMENTATION_REVISION:
        raise RuntimeError(
            "Model A v2 implementation revision mismatch."
        )

    clean_ref = a_v2_summary.get(
        "clean_reference"
    )

    if not isinstance(
        clean_ref,
        Mapping,
    ):
        raise RuntimeError(
            "Model A v2 summary lacks clean_reference."
        )

    if not math.isclose(
        float(clean_ref["miou"]),
        float(a_clean["miou"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Model A Clean mIoU mismatch between files."
        )


def a_summary_map(
    summary: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rows = summary.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(
            "Model A v2 summary has no results list."
        )

    out = {
        str(row["condition"]): dict(row)
        for row in rows
    }

    expected = {
        condition_name(c, l)
        for c, l in CONDITION_ORDER
    }

    if set(out) != expected:
        raise RuntimeError(
            "Model A v2 summary does not contain all nine conditions."
        )

    return out


def model_a_detail_path(
    condition: str,
    *,
    a_clean_path: Path,
    a_v1_dir: Path,
    a_v2_dir: Path,
) -> Path:
    if condition == "Clean":
        return a_clean_path

    if (
        condition.startswith("gaussian_noise_")
        or condition.startswith("gaussian_blur_")
    ):
        return (
            a_v1_dir
            / condition
            / "metrics.json"
        )

    if condition.startswith("rgb_underexposure_"):
        return (
            a_v2_dir
            / condition
            / "metrics.json"
        )

    raise KeyError(condition)


def per_class_map(
    metrics: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    rows = metrics.get("per_class")
    if not isinstance(rows, list):
        raise RuntimeError(
            "metrics.json missing per_class."
        )

    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[int(row["class_id"])] = dict(row)

    if set(out) != set(range(NUM_CLASSES)):
        raise RuntimeError(
            f"Per-class ids incomplete: {sorted(out)}"
        )

    return out


def make_comparison(
    *,
    a_clean: Mapping[str, Any],
    a_summary: Mapping[str, Any],
    b_clean: Mapping[str, Any],
    b_results: Mapping[str, Mapping[str, Any]],
    a_clean_path: Path,
    a_v1_dir: Path,
    a_v2_dir: Path,
    robustness_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    a_by_condition = a_summary_map(
        a_summary
    )

    a_clean_miou = float(
        a_clean["miou"]
    )
    b_clean_miou = float(
        b_clean["miou"]
    )

    comparison_rows: List[
        Dict[str, Any]
    ] = []

    # Clean
    comparison_rows.append(
        {
            "condition": "Clean",
            "corruption": "Clean",
            "severity_level": "",
            "severity_rank": 0,
            "model_a_miou": a_clean_miou,
            "model_b_miou": b_clean_miou,
            "nir_gain_miou": (
                b_clean_miou
                - a_clean_miou
            ),
            "model_a_drop": 0.0,
            "model_b_drop": 0.0,
            "drop_reduction": 0.0,
            "relative_drop_reduction_pct": None,
        }
    )

    for corruption, level in CONDITION_ORDER:
        cond = condition_name(
            corruption,
            level,
        )

        a_row = a_by_condition[cond]
        b_row = b_results[cond]

        a_miou = float(
            a_row["miou"]
        )
        b_miou = float(
            b_row["miou"]
        )

        a_drop = (
            a_clean_miou
            - a_miou
        )
        b_drop = (
            b_clean_miou
            - b_miou
        )

        drop_reduction = (
            a_drop
            - b_drop
        )

        relative_drop_reduction_pct = (
            100.0
            * drop_reduction
            / a_drop
            if abs(a_drop) > 1e-15
            else None
        )

        comparison_rows.append(
            {
                "condition": cond,
                "corruption": corruption,
                "severity_level": level,
                "severity_rank": int(level[1:]),
                "model_a_miou": a_miou,
                "model_b_miou": b_miou,
                "nir_gain_miou": (
                    b_miou
                    - a_miou
                ),
                "model_a_drop": a_drop,
                "model_b_drop": b_drop,
                "drop_reduction": drop_reduction,
                "relative_drop_reduction_pct": (
                    relative_drop_reduction_pct
                ),
            }
        )

    # Per-class NIR gain across Clean + all 9 degraded conditions.
    per_class_rows: List[
        Dict[str, Any]
    ] = []

    for comparison in comparison_rows:
        cond = str(
            comparison["condition"]
        )

        a_detail = read_json(
            model_a_detail_path(
                cond,
                a_clean_path=a_clean_path,
                a_v1_dir=a_v1_dir,
                a_v2_dir=a_v2_dir,
            )
        )

        if cond == "Clean":
            b_detail = b_clean
        else:
            b_detail = b_results[cond]

        a_classes = per_class_map(
            a_detail
        )
        b_classes = per_class_map(
            b_detail
        )

        for class_id in range(
            NUM_CLASSES
        ):
            a_iou = float(
                a_classes[class_id]["iou"]
            )
            b_iou = float(
                b_classes[class_id]["iou"]
            )

            per_class_rows.append(
                {
                    "condition": cond,
                    "corruption": comparison["corruption"],
                    "severity_level": comparison["severity_level"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "model_a_iou": a_iou,
                    "model_b_iou": b_iou,
                    "nir_gain_iou": (
                        b_iou
                        - a_iou
                    ),
                }
            )

    # RQ-oriented trend diagnostics.
    trend_analysis: Dict[str, Any] = {}

    for corruption, _ in [
        ("gaussian_noise", None),
        ("gaussian_blur", None),
        ("rgb_underexposure", None),
    ]:
        rows = [
            row
            for row in comparison_rows
            if row["corruption"] == corruption
        ]

        rows.sort(
            key=lambda x: int(
                x["severity_rank"]
            )
        )

        gains = [
            float(
                row["nir_gain_miou"]
            )
            for row in rows
        ]

        drop_reductions = [
            float(
                row["drop_reduction"]
            )
            for row in rows
        ]

        trend_analysis[corruption] = {
            "levels": [
                row["severity_level"]
                for row in rows
            ],
            "nir_gain_miou": gains,
            "drop_reduction": drop_reductions,
            "nir_gain_monotonic_nondecreasing": all(
                gains[i + 1] + 1e-12
                >= gains[i]
                for i in range(
                    len(gains) - 1
                )
            ),
            "drop_reduction_monotonic_nondecreasing": all(
                drop_reductions[i + 1] + 1e-12
                >= drop_reductions[i]
                for i in range(
                    len(drop_reductions) - 1
                )
            ),
        }

    comparison_json = {
        "comparison": "Model A RGB vs Model B RGB+NIR",
        "split": "val",
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL_VERSION
        ),
        "degradation_implementation_revision": (
            IMPLEMENTATION_REVISION
        ),
        "degradation_protocol_sha256": (
            degradation_protocol_sha256()
        ),
        "definitions": {
            "nir_gain_miou": (
                "Model B mIoU - Model A mIoU at the same condition"
            ),
            "drop": (
                "same-model Clean mIoU - same-model degraded mIoU"
            ),
            "drop_reduction": (
                "Model A Drop - Model B Drop; positive means Model B loses less"
            ),
        },
        "clean_gain_miou": (
            b_clean_miou
            - a_clean_miou
        ),
        "results": comparison_rows,
        "trend_analysis": trend_analysis,
    }

    return (
        comparison_rows,
        per_class_rows,
        comparison_json,
    )


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "condition",
        "corruption",
        "severity_level",
        "severity_rank",
        "model_a_miou",
        "model_b_miou",
        "nir_gain_miou",
        "model_a_drop",
        "model_b_drop",
        "drop_reduction",
        "relative_drop_reduction_pct",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def write_per_class_gain_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "condition",
        "corruption",
        "severity_level",
        "class_id",
        "class_name",
        "model_a_iou",
        "model_b_iou",
        "nir_gain_iou",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def write_robustness_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "model",
        "condition",
        "corruption",
        "severity_level",
        "severity_rank",
        "clean_miou",
        "miou",
        "drop_miou",
        "delta_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        "validation_seconds",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )
    output_root = resolve_path(
        args.output_root
    )
    clean_dir = (
        output_root
        / "clean_val"
    )
    robustness_dir = (
        output_root
        / "robustness_val_v2"
    )

    a_clean_path = resolve_path(
        args.model_a_clean
    )
    a_v1_dir = resolve_path(
        args.model_a_v1_dir
    )
    a_v2_dir = resolve_path(
        args.model_a_v2_dir
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    a_clean = read_json(
        a_clean_path
    )
    a_v2_summary = read_json(
        a_v2_dir
        / "robustness_summary.json"
    )

    validate_model_a_references(
        a_clean=a_clean,
        a_v2_summary=a_v2_summary,
    )

    device = get_device(
        args.device
    )
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 96)
    print("MODEL B | CLEAN + RGB DEGRADATION PROTOCOL v2 VALIDATION")
    print("=" * 96)
    print(f"checkpoint      : {checkpoint_path}")
    print(f"output root     : {output_root}")
    print(f"device / AMP    : {device} / {amp_enabled}")
    print(f"batch size      : {args.batch_size}")
    print(f"protocol        : {DEGRADATION_PROTOCOL_VERSION}")
    print(f"implementation  : revision {IMPLEMENTATION_REVISION}")
    print(f"protocol hash   : {degradation_protocol_sha256()}")
    print(
        "noise RNG ns    : "
        f"{DEGRADATION_PROTOCOL['rng_policy']['gaussian_noise_seed_namespace']}"
    )
    print("=" * 96)

    print("[1] building Model B")
    model, model_meta = build_model_b_rgbnir(
        PROJECT_ROOT
    )

    print("[2] loading trained Model B checkpoint")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        unwrap_model_b_checkpoint(
            checkpoint
        )
    )
    validate_model_b_checkpoint_metadata(
        checkpoint_meta
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.to(device)
    model.eval()

    checkpoint_epoch = checkpoint_meta.get(
        "epoch"
    )
    checkpoint_global_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get(
            "step"
        ),
    )

    print(
        f"[checkpoint] epoch_zero_based={checkpoint_epoch} | "
        f"global_step={checkpoint_global_step}"
    )

    # --------------------------------------------------------------
    # Clean
    # --------------------------------------------------------------
    print("[3] Model B Clean validation")

    clean_metrics_path = (
        clean_dir
        / "metrics.json"
    )

    b_clean = None
    if not args.force_clean:
        b_clean = load_existing_clean(
            clean_metrics_path,
            checkpoint_path=checkpoint_path,
            checkpoint_global_step=(
                int(checkpoint_global_step)
                if checkpoint_global_step is not None
                else None
            ),
        )

    if b_clean is not None:
        print(
            "[resume] compatible Model B Clean result exists; "
            f"skipping: {clean_metrics_path}"
        )
    else:
        clean_dataset = PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )

        b_clean = evaluate_dataset(
            model=model,
            dataset=clean_dataset,
            condition="Clean",
            output_dir=clean_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=(
                int(checkpoint_epoch)
                if checkpoint_epoch is not None
                else None
            ),
            checkpoint_global_step=(
                int(checkpoint_global_step)
                if checkpoint_global_step is not None
                else None
            ),
            batch_size=args.batch_size,
            device=device,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
            confusion_chunk_rows=(
                args.confusion_chunk_rows
            ),
            save_predictions=(
                args.save_predictions
            ),
        )

        del clean_dataset

    b_clean_miou = float(
        b_clean["miou"]
    )
    a_clean_miou = float(
        a_clean["miou"]
    )

    print(
        f"[Clean] A={a_clean_miou:.6f} | "
        f"B={b_clean_miou:.6f} | "
        f"NIR Gain={b_clean_miou - a_clean_miou:+.6f}"
    )

    # --------------------------------------------------------------
    # Robustness
    # --------------------------------------------------------------
    print("[4] Model B Protocol-v2 robustness validation")

    robustness_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_degradation_protocol(
        robustness_dir
        / "degradation_protocol.json"
    )

    b_results: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for condition_index, (
        corruption,
        level,
    ) in enumerate(
        CONDITION_ORDER,
        start=1,
    ):
        cond = condition_name(
            corruption,
            level,
        )

        cond_dir = (
            robustness_dir
            / cond
        )
        metrics_path = (
            cond_dir
            / "metrics.json"
        )

        print()
        print(
            f"[condition {condition_index}/9] {cond} "
            f"{condition_spec(corruption, level)}"
        )

        existing = None
        if not args.force_robustness:
            existing = load_existing_degraded(
                metrics_path,
                condition=cond,
                checkpoint_path=checkpoint_path,
                checkpoint_global_step=(
                    int(checkpoint_global_step)
                    if checkpoint_global_step is not None
                    else None
                ),
            )

        if existing is not None:
            print(
                "[resume] compatible result exists; "
                f"skipping: {metrics_path}"
            )
            result = existing
        else:
            degraded_dataset = (
                DegradedPotsdamSlidingWindowDataset(
                    PROJECT_ROOT,
                    split="val",
                    corruption=corruption,
                    level=level,
                )
            )

            clean_probe = (
                PotsdamSlidingWindowDataset(
                    PROJECT_ROOT,
                    split="val",
                )
            )
            degradation_probe(
                clean_dataset=clean_probe,
                degraded_dataset=degraded_dataset,
            )
            del clean_probe

            result = evaluate_dataset(
                model=model,
                dataset=degraded_dataset,
                condition=cond,
                output_dir=cond_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=(
                    int(checkpoint_epoch)
                    if checkpoint_epoch is not None
                    else None
                ),
                checkpoint_global_step=(
                    int(checkpoint_global_step)
                    if checkpoint_global_step is not None
                    else None
                ),
                batch_size=args.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                log_every=args.log_every,
                confusion_chunk_rows=(
                    args.confusion_chunk_rows
                ),
                save_predictions=(
                    args.save_predictions
                ),
                corruption=corruption,
                severity_level=level,
            )

            del degraded_dataset

        degraded_miou = float(
            result["miou"]
        )
        drop = (
            b_clean_miou
            - degraded_miou
        )

        result["clean_reference_miou"] = (
            b_clean_miou
        )
        result["drop_miou"] = drop
        result["delta_miou"] = -drop
        result["relative_drop_pct"] = (
            100.0
            * drop
            / b_clean_miou
            if b_clean_miou != 0
            else None
        )
        result["retention_pct"] = (
            100.0
            * degraded_miou
            / b_clean_miou
            if b_clean_miou != 0
            else None
        )

        write_json(
            metrics_path,
            result,
        )

        b_results[cond] = result

        print(
            f"[{cond}] "
            f"mIoU={degraded_miou:.6f} | "
            f"Drop={drop:.6f} | "
            f"Retention={result['retention_pct']:.2f}%"
        )

    # --------------------------------------------------------------
    # B robustness summary
    # --------------------------------------------------------------
    summary_rows: List[
        Dict[str, Any]
    ] = []

    for corruption, level in CONDITION_ORDER:
        cond = condition_name(
            corruption,
            level,
        )
        result = b_results[cond]

        summary_rows.append(
            {
                "model": MODEL_ID,
                "condition": cond,
                "corruption": corruption,
                "severity_level": level,
                "severity_rank": int(level[1:]),
                "clean_miou": b_clean_miou,
                "miou": result["miou"],
                "drop_miou": result["drop_miou"],
                "delta_miou": result["delta_miou"],
                "relative_drop_pct": result["relative_drop_pct"],
                "retention_pct": result["retention_pct"],
                "pixel_accuracy": result["pixel_accuracy"],
                "mean_class_accuracy": result["mean_class_accuracy"],
                "validation_seconds": result["validation_seconds"],
            }
        )

    b_trends: Dict[str, Any] = {}

    for corruption in (
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    ):
        rows = [
            row
            for row in summary_rows
            if row["corruption"] == corruption
        ]
        rows.sort(
            key=lambda x: int(
                x["severity_rank"]
            )
        )

        mious = [
            float(row["miou"])
            for row in rows
        ]
        drops = [
            float(row["drop_miou"])
            for row in rows
        ]

        b_trends[corruption] = {
            "levels": [
                row["severity_level"]
                for row in rows
            ],
            "miou": mious,
            "drop_miou": drops,
            "monotonic_nonincreasing_miou": all(
                mious[i + 1]
                <= mious[i] + 1e-12
                for i in range(
                    len(mious) - 1
                )
            ),
            "monotonic_nondecreasing_drop": all(
                drops[i + 1] + 1e-12
                >= drops[i]
                for i in range(
                    len(drops) - 1
                )
            ),
        }

    robustness_summary = {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "split": "val",
        "clean_reference": {
            "metrics_path": str(
                clean_metrics_path
            ),
            "miou": b_clean_miou,
            "pixel_accuracy": b_clean.get(
                "pixel_accuracy"
            ),
            "mean_class_accuracy": b_clean.get(
                "mean_class_accuracy"
            ),
            "checkpoint_global_step": (
                checkpoint_global_step
            ),
        },
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL_VERSION
        ),
        "degradation_implementation_revision": (
            IMPLEMENTATION_REVISION
        ),
        "degradation_protocol_sha256": (
            degradation_protocol_sha256()
        ),
        "results": summary_rows,
        "trend_analysis": b_trends,
    }

    write_json(
        robustness_dir
        / "robustness_summary.json",
        robustness_summary,
    )
    write_robustness_summary_csv(
        robustness_dir
        / "robustness_summary.csv",
        summary_rows,
    )

    # --------------------------------------------------------------
    # A vs B
    # --------------------------------------------------------------
    print("[5] computing Model A vs Model B NIR Gain")

    comparison_rows, per_class_rows, comparison_json = (
        make_comparison(
            a_clean=a_clean,
            a_summary=a_v2_summary,
            b_clean=b_clean,
            b_results=b_results,
            a_clean_path=a_clean_path,
            a_v1_dir=a_v1_dir,
            a_v2_dir=a_v2_dir,
            robustness_dir=robustness_dir,
        )
    )

    write_json(
        robustness_dir
        / "model_a_vs_b_comparison.json",
        comparison_json,
    )
    write_comparison_csv(
        robustness_dir
        / "model_a_vs_b_comparison.csv",
        comparison_rows,
    )
    write_per_class_gain_csv(
        robustness_dir
        / "per_class_nir_gain.csv",
        per_class_rows,
    )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------
    print()
    print("=" * 112)
    print("MODEL A vs MODEL B | NIR GAIN SUMMARY")
    print("=" * 112)
    print(
        f"{'Condition':<28} "
        f"{'A mIoU':>10} "
        f"{'B mIoU':>10} "
        f"{'NIR Gain':>11} "
        f"{'A Drop':>10} "
        f"{'B Drop':>10} "
        f"{'Drop Red.':>11}"
    )
    print("-" * 112)

    for row in comparison_rows:
        print(
            f"{row['condition']:<28} "
            f"{float(row['model_a_miou']):>10.6f} "
            f"{float(row['model_b_miou']):>10.6f} "
            f"{float(row['nir_gain_miou']):>+11.6f} "
            f"{float(row['model_a_drop']):>10.6f} "
            f"{float(row['model_b_drop']):>10.6f} "
            f"{float(row['drop_reduction']):>+11.6f}"
        )

    print("-" * 112)

    for corruption, trend in comparison_json[
        "trend_analysis"
    ].items():
        print(
            f"{corruption}: "
            f"NIR Gain L1->L3 nondecreasing="
            f"{trend['nir_gain_monotonic_nondecreasing']} | "
            f"Drop reduction nondecreasing="
            f"{trend['drop_reduction_monotonic_nondecreasing']}"
        )

    print("-" * 112)
    print(
        "Clean Gain = "
        f"{comparison_json['clean_gain_miou']:+.6f}"
    )
    print(
        f"B Clean metrics : "
        f"{clean_dir / 'metrics.json'}"
    )
    print(
        f"B robustness    : "
        f"{robustness_dir / 'robustness_summary.json'}"
    )
    print(
        f"A vs B CSV      : "
        f"{robustness_dir / 'model_a_vs_b_comparison.csv'}"
    )
    print(
        f"Per-class Gain  : "
        f"{robustness_dir / 'per_class_nir_gain.csv'}"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
