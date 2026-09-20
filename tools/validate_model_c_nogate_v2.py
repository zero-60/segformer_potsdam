#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Model C-noGate (Dual Encoder + Fixed 0.5/0.5 Multi-Scale Fusion)
under Clean + FINAL RGB Degradation Protocol v2.

This script runs/resumes:

    Clean

    Gaussian Noise
        L1 / L2 / L3

    Gaussian Blur
        L1 / L2 / L3

    RGB Underexposure
        L1 / L2 / L3

Evaluation protocol is identical to Model A / Model B:

    512x512 windows
        -> full-resolution logits
        -> full-tile MEAN-LOGIT fusion
        -> one 6000x6000 prediction per tile
        -> one GLOBAL confusion matrix over all 6 val tiles
        -> global mIoU + six class IoUs

The script also builds a 3-model comparison:

    Model A        : RGB only
    Model B        : RGB+NIR direct 4-channel early fusion
    Model C-noGate : dual encoders + fixed 0.5/0.5 four-scale fusion

Key comparison quantities
-------------------------
C-vs-A Gain:
    mIoU_C_noGate(condition) - mIoU_A(condition)

Fixed-Fusion Gain vs B:
    mIoU_C_noGate(condition) - mIoU_B(condition)

Drop_X:
    mIoU_X(Clean) - mIoU_X(condition)

Drop Reduction vs A:
    Drop_A - Drop_C_noGate

Drop Reduction vs B:
    Drop_B - Drop_C_noGate

Positive Drop Reduction means C-noGate loses less performance from its own
Clean score under the same RGB degradation.

Expected files
--------------
evaluation/rgb_degradation_protocol.py
    FINAL v2 implementation with implementation_revision=2.

models/segformer_dual_fixed.py
tools/validate_model_a_rgb.py

Existing reference results:
outputs/evaluation/model_a_rgb/clean_val/metrics.json
outputs/evaluation/model_a_rgb/robustness_val/
outputs/evaluation/model_a_rgb/robustness_val_v2/
outputs/evaluation/model_b_rgbnir/clean_val/metrics.json
outputs/evaluation/model_b_rgbnir/robustness_val_v2/

Checkpoint:
outputs/training/model_c_nogate/checkpoints/final.pt

Outputs
-------
outputs/evaluation/model_c_nogate/
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
    ├── three_model_comparison.json
    ├── three_model_comparison.csv
    ├── per_class_three_model_comparison.csv
    ├── gaussian_noise_L1/
    ├── ...
    └── rgb_underexposure_L3/

Run
---
    python tools/validate_model_c_nogate_v2.py

Resume behavior
---------------
Compatible existing Clean/degraded metrics are reused automatically.
Use --force-clean and/or --force-robustness to recompute.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ============================================================================
# Project imports
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL,
    DEGRADATION_PROTOCOL_VERSION,
    IMPLEMENTATION_REVISION,
    DegradedPotsdamSlidingWindowDataset,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    write_degradation_protocol,
)
from models.segformer_dual_fixed import (
    FIXED_NIR_WEIGHT,
    FIXED_RGB_WEIGHT,
    MODEL_ID,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_SCALES,
    build_model_c_nogate,
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
        "Keep the successful Model A validator in tools/."
    ) from exc


# ============================================================================
# Paths / condition order
# ============================================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_c_nogate"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_c_nogate"
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

DEFAULT_B_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_b_rgbnir"
    / "clean_val"
    / "metrics.json"
)

DEFAULT_B_V2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_b_rgbnir"
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


# ============================================================================
# CLI / generic utilities
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Model C-noGate Clean + final Protocol-v2 robustness "
            "and compare A/B/C-noGate."
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
        "--model-b-clean",
        type=Path,
        default=DEFAULT_B_CLEAN,
    )
    parser.add_argument(
        "--model-b-v2-dir",
        type=Path,
        default=DEFAULT_B_V2_DIR,
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
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return obj


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def get_device(
    device_arg: str,
) -> torch.device:
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
        torch.cuda.set_device(
            device.index
        )

    return device


# ============================================================================
# Checkpoint validation
# ============================================================================

def unwrap_checkpoint(
    checkpoint: Any,
) -> tuple[
    Mapping[str, torch.Tensor],
    Dict[str, Any],
]:
    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            f"Checkpoint must be dict-like, got {type(checkpoint)!r}."
        )

    if "model" not in checkpoint:
        raise KeyError(
            "C-noGate checkpoint has no checkpoint['model'] state_dict."
        )

    state = checkpoint["model"]

    if not isinstance(
        state,
        Mapping,
    ):
        raise TypeError(
            "checkpoint['model'] is not a state_dict mapping."
        )

    metadata = {
        key: value
        for key, value
        in checkpoint.items()
        if key != "model"
    }

    return state, metadata


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    checkpoint_model_id = metadata.get(
        "model_id"
    )

    if (
        checkpoint_model_id is not None
        and checkpoint_model_id != MODEL_ID
    ):
        raise RuntimeError(
            "Checkpoint model_id mismatch: "
            f"{checkpoint_model_id!r} != {MODEL_ID!r}."
        )

    checkpoint_model_name = metadata.get(
        "model_name"
    )

    if (
        checkpoint_model_name is not None
        and checkpoint_model_name != MODEL_NAME
    ):
        raise RuntimeError(
            "Checkpoint model_name mismatch: "
            f"{checkpoint_model_name!r} != {MODEL_NAME!r}."
        )

    protocol = metadata.get(
        "protocol"
    )

    if not isinstance(
        protocol,
        Mapping,
    ):
        raise RuntimeError(
            "C-noGate checkpoint does not contain protocol metadata."
        )

    if protocol.get(
        "model"
    ) != MODEL_ID:
        raise RuntimeError(
            "Checkpoint protocol is not Model C-noGate."
        )

    if not bool(
        protocol.get(
            "dual_encoder",
            False,
        )
    ):
        raise RuntimeError(
            "C-noGate checkpoint protocol is not dual-encoder."
        )

    if bool(
        protocol.get(
            "quality_gate",
            True,
        )
    ):
        raise RuntimeError(
            "Checkpoint unexpectedly enables a Quality Gate."
        )

    if int(
        protocol.get(
            "fusion_scales",
            -1,
        )
    ) != NUM_SCALES:
        raise RuntimeError(
            "Checkpoint fusion scale count does not equal 4."
        )

    if not math.isclose(
        float(
            protocol.get(
                "fixed_rgb_weight",
                -1.0,
            )
        ),
        FIXED_RGB_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Checkpoint RGB fixed fusion weight is not 0.5."
        )

    if not math.isclose(
        float(
            protocol.get(
                "fixed_nir_weight",
                -1.0,
            )
        ),
        FIXED_NIR_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Checkpoint NIR fixed fusion weight is not 0.5."
        )


# ============================================================================
# C-noGate full-tile inference
# ============================================================================

def validate_batch(
    batch: Mapping[str, Any],
    *,
    expected_tile_id: str,
) -> None:
    required = {
        "rgb",
        "nir",
        "tile_id",
        "x",
        "y",
    }

    missing = sorted(
        required
        - set(
            batch.keys()
        )
    )

    if missing:
        raise RuntimeError(
            f"Validation batch missing keys: {missing}"
        )

    rgb = batch["rgb"]
    nir = batch["nir"]

    if (
        not isinstance(
            rgb,
            torch.Tensor,
        )
        or rgb.ndim != 4
        or rgb.shape[1] != 3
    ):
        raise RuntimeError(
            "RGB must be [B,3,H,W], "
            f"got {getattr(rgb, 'shape', None)}."
        )

    if (
        not isinstance(
            nir,
            torch.Tensor,
        )
        or nir.ndim != 4
        or nir.shape[1] != 1
    ):
        raise RuntimeError(
            "NIR must be [B,1,H,W], "
            f"got {getattr(nir, 'shape', None)}."
        )

    if (
        rgb.shape[0]
        != nir.shape[0]
        or rgb.shape[-2:]
        != nir.shape[-2:]
    ):
        raise RuntimeError(
            "RGB/NIR batch or spatial shapes differ."
        )

    tile_ids = list(
        batch["tile_id"]
    )

    if any(
        str(tile_id)
        != expected_tile_id
        for tile_id
        in tile_ids
    ):
        raise RuntimeError(
            "Per-tile validation loader mixed tile IDs: "
            f"expected={expected_tile_id}, got={tile_ids}."
        )


def audit_fixed_fusion_weights(
    weights: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    expected_shape = (
        batch_size,
        NUM_SCALES,
        2,
    )

    if tuple(
        weights.shape
    ) != expected_shape:
        raise RuntimeError(
            "C-noGate fusion weight shape mismatch: "
            f"expected={expected_shape}, got={tuple(weights.shape)}."
        )

    expected = torch.full_like(
        weights,
        0.5,
    )

    if not torch.equal(
        weights,
        expected,
    ):
        max_error = float(
            (
                weights
                - expected
            )
            .abs()
            .max()
            .item()
        )

        raise RuntimeError(
            "C-noGate returned non-fixed fusion weights. "
            f"max_abs_error_from_0.5={max_error}"
        )

    print(
        "[fixed-fusion audit] PASS | "
        "all four scales exactly [0.5, 0.5]"
    )


def infer_one_tile(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    tile_id: str,
    tile_index: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    audit_first_forward: bool,
) -> tuple[
    np.ndarray,
    Dict[str, Any],
    bool,
]:
    windows_per_tile = len(
        dataset.window_coordinates
    )

    start_index = (
        tile_index
        * windows_per_tile
    )
    stop_index = (
        start_index
        + windows_per_tile
    )

    subset = Subset(
        dataset,
        range(
            start_index,
            stop_index,
        ),
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

    tile_start = time.time()
    windows_seen = 0
    audit_completed = False

    for batch_index, batch in enumerate(
        loader
    ):
        validate_batch(
            batch,
            expected_tile_id=tile_id,
        )

        rgb = batch["rgb"].to(
            device,
            non_blocking=(
                device.type
                == "cuda"
            ),
        )

        nir = batch["nir"].to(
            device,
            non_blocking=(
                device.type
                == "cuda"
            ),
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if (
                    audit_first_forward
                    and not audit_completed
                ):
                    details = model(
                        rgb,
                        nir,
                        return_details=True,
                    )

                    logits = details[
                        "logits"
                    ]

                    audit_fixed_fusion_weights(
                        details[
                            "fusion_weights"
                        ],
                        batch_size=int(
                            rgb.shape[0]
                        ),
                    )

                    # Release explicit feature-pyramid references immediately.
                    del details
                    audit_completed = True
                else:
                    logits = model(
                        rgb,
                        nir,
                    )

        if (
            logits.ndim != 4
            or logits.shape[1]
            != NUM_CLASSES
            or tuple(
                logits.shape[-2:]
            )
            != (
                crop_size,
                crop_size,
            )
        ):
            raise RuntimeError(
                "Unexpected C-noGate logits shape: "
                f"{tuple(logits.shape)}."
            )

        if not torch.isfinite(
            logits
        ).all().item():
            raise FloatingPointError(
                "Non-finite C-noGate logits: "
                f"tile={tile_id}, batch={batch_index}."
            )

        logits = logits.float()

        xs = batch["x"]
        ys = batch["y"]

        batch_count = int(
            logits.shape[0]
        )

        for local_index in range(
            batch_count
        ):
            x = int(
                xs[local_index]
            )
            y = int(
                ys[local_index]
            )

            if not (
                0
                <= x
                <= tile_size
                - crop_size
                and 0
                <= y
                <= tile_size
                - crop_size
            ):
                raise RuntimeError(
                    f"Invalid window coordinates: tile={tile_id}, x={x}, y={y}."
                )

            logits_sum[
                :,
                y:y + crop_size,
                x:x + crop_size,
            ].add_(
                logits[
                    local_index
                ]
            )

            coverage[
                y:y + crop_size,
                x:x + crop_size,
            ].add_(1.0)

        windows_seen += (
            batch_count
        )

        if (
            batch_index
            % log_every
            == 0
            or batch_index + 1
            == len(loader)
        ):
            print(
                f"  tile {tile_id} | "
                f"batch {batch_index + 1:03d}/{len(loader):03d} | "
                f"windows {windows_seen:03d}/{windows_per_tile:03d}",
                flush=True,
            )

        del rgb
        del nir
        del logits

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
        zero_count = int(
            (
                coverage == 0
            )
            .sum()
            .item()
        )

        raise RuntimeError(
            f"{tile_id}: {zero_count} full-tile pixels were never covered."
        )

    logits_sum.div_(
        coverage.unsqueeze(0)
    )

    prediction = (
        logits_sum
        .argmax(
            dim=0
        )
        .to(
            dtype=torch.uint8
        )
        .cpu()
        .numpy()
    )

    elapsed = (
        time.time()
        - tile_start
    )

    del logits_sum
    del coverage

    if device.type == "cuda":
        torch.cuda.empty_cache()

    runtime = {
        "tile_id": tile_id,
        "windows": windows_seen,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "inference_seconds": elapsed,
    }

    return (
        prediction,
        runtime,
        audit_completed,
    )


# ============================================================================
# Degradation sanity probe
# ============================================================================

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
        if (
            clean[key]
            != degraded[key]
        ):
            raise RuntimeError(
                f"Degradation changed frozen window metadata: {key}."
            )

    if not torch.equal(
        clean["nir"],
        degraded["nir"],
    ):
        max_diff = float(
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
            f"max_abs_diff={max_diff}"
        )

    if torch.equal(
        clean["rgb"],
        degraded["rgb"],
    ):
        raise RuntimeError(
            "Degraded RGB equals Clean RGB on the probe window."
        )

    rgb_change = float(
        (
            clean["rgb"]
            - degraded["rgb"]
        )
        .abs()
        .mean()
        .item()
    )

    print(
        "[degradation probe] PASS | "
        "same frozen window | NIR bit-identical | "
        f"RGB normalized mean_abs_change={rgb_change:.6f}"
    )


# ============================================================================
# Generic condition evaluation
# ============================================================================

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
    audit_fixed_fusion: bool,
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

    prediction_dir = (
        output_dir
        / "predictions"
    )

    if save_predictions:
        prediction_dir.mkdir(
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
        len(tile_ids)
        != 6
        or windows_per_tile
        != 256
    ):
        raise RuntimeError(
            "Frozen validation protocol changed unexpectedly: "
            f"tiles={len(tile_ids)}, windows/tile={windows_per_tile}."
        )

    global_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    per_tile: List[
        Dict[str, Any]
    ] = []

    validation_start = (
        time.time()
    )

    fixed_fusion_audited = (
        not audit_fixed_fusion
    )

    for tile_index, tile_id in enumerate(
        tile_ids
    ):
        print(
            f"[{condition}] "
            f"tile {tile_index + 1}/{len(tile_ids)} | {tile_id}",
            flush=True,
        )

        (
            prediction,
            tile_runtime,
            audit_completed,
        ) = infer_one_tile(
            model=model,
            dataset=dataset,
            tile_id=tile_id,
            tile_index=tile_index,
            batch_size=batch_size,
            device=device,
            amp_enabled=amp_enabled,
            log_every=log_every,
            audit_first_forward=(
                not fixed_fusion_audited
            ),
        )

        if audit_completed:
            fixed_fusion_audited = True

        target_t = (
            dataset
            .load_full_label(
                tile_id
            )
        )

        target = target_t.numpy()

        tile_confusion = (
            confusion_from_prediction(
                prediction,
                target,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                chunk_rows=(
                    confusion_chunk_rows
                ),
            )
        )

        global_confusion += (
            tile_confusion
        )

        tile_metrics = (
            metrics_from_confusion(
                tile_confusion,
                CLASS_NAMES,
            )
        )

        record = {
            **tile_runtime,
            "condition": condition,
            "miou": (
                tile_metrics[
                    "miou"
                ]
            ),
            "pixel_accuracy": (
                tile_metrics[
                    "pixel_accuracy"
                ]
            ),
            "mean_class_accuracy": (
                tile_metrics[
                    "mean_class_accuracy"
                ]
            ),
            "valid_pixels": (
                tile_metrics[
                    "valid_pixels"
                ]
            ),
            "per_class_iou": {
                row[
                    "class_name"
                ]: row[
                    "iou"
                ]
                for row
                in tile_metrics[
                    "per_class"
                ]
            },
        }

        if corruption is not None:
            record[
                "corruption"
            ] = corruption
            record[
                "severity_level"
            ] = severity_level
            record[
                "severity_parameters"
            ] = condition_spec(
                corruption,
                str(
                    severity_level
                ),
            )

        per_tile.append(
            record
        )

        append_jsonl(
            per_tile_path,
            record,
        )

        if save_predictions:
            np.save(
                prediction_dir
                / f"{tile_id}_pred.npy",
                prediction,
                allow_pickle=False,
            )

        print(
            f"  fused tile mIoU="
            f"{tile_metrics['miou']:.6f} | "
            f"pixel_acc="
            f"{tile_metrics['pixel_accuracy']:.6f} | "
            f"time="
            f"{tile_runtime['inference_seconds']:.1f}s",
            flush=True,
        )

        del prediction
        del target
        del target_t

        if isinstance(
            dataset,
            DegradedPotsdamSlidingWindowDataset,
        ):
            dataset.clear_degradation_cache()

    if (
        audit_fixed_fusion
        and not fixed_fusion_audited
    ):
        raise RuntimeError(
            "Requested fixed-fusion audit was never completed."
        )

    global_metrics = (
        metrics_from_confusion(
            global_confusion,
            CLASS_NAMES,
        )
    )

    validation_seconds = (
        time.time()
        - validation_start
    )

    result: Dict[str, Any] = {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "condition": condition,
        "split": "val",
        "metric_scope": (
            "GLOBAL confusion matrix after full-tile mean-logit fusion"
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch_zero_based": (
            checkpoint_epoch
        ),
        "checkpoint_global_step": (
            checkpoint_global_step
        ),
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "dual_encoder": True,
        "quality_gate": False,
        "fusion_rule": (
            "F_i = 0.5*F_RGB_i + 0.5*F_NIR_i"
        ),
        "fixed_rgb_weight": (
            FIXED_RGB_WEIGHT
        ),
        "fixed_nir_weight": (
            FIXED_NIR_WEIGHT
        ),
        "num_classes": (
            NUM_CLASSES
        ),
        "class_names": (
            CLASS_NAMES
        ),
        "ignore_index": (
            IGNORE_INDEX
        ),
        "miou": (
            global_metrics[
                "miou"
            ]
        ),
        "pixel_accuracy": (
            global_metrics[
                "pixel_accuracy"
            ]
        ),
        "mean_class_accuracy": (
            global_metrics[
                "mean_class_accuracy"
            ]
        ),
        "valid_pixels": (
            global_metrics[
                "valid_pixels"
            ]
        ),
        "per_class": (
            global_metrics[
                "per_class"
            ]
        ),
        "confusion_matrix": (
            global_confusion
            .tolist()
        ),
        "tiles": per_tile,
        "num_tiles": len(
            tile_ids
        ),
        "windows_per_tile": (
            windows_per_tile
        ),
        "total_windows": len(
            dataset
        ),
        "validation_seconds": (
            validation_seconds
        ),
        "fixed_fusion_audit_passed": (
            fixed_fusion_audited
        ),
    }

    if corruption is not None:
        result.update(
            {
                "corruption": (
                    corruption
                ),
                "severity_level": (
                    severity_level
                ),
                "severity_rank": int(
                    str(
                        severity_level
                    )[1:]
                ),
                "severity_parameters": (
                    condition_spec(
                        corruption,
                        str(
                            severity_level
                        ),
                    )
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
                "nir_source": (
                    "clean / unchanged"
                ),
            }
        )

    write_json(
        output_dir
        / "metrics.json",
        result,
    )

    write_per_class_csv(
        output_dir
        / "per_class_metrics.csv",
        global_metrics[
            "per_class"
        ],
    )

    write_confusion_csv(
        output_dir
        / "confusion_matrix.csv",
        global_confusion,
        CLASS_NAMES,
    )

    return result


# ============================================================================
# Resume helpers
# ============================================================================

def load_existing_clean(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_global_step: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = read_json(
            path
        )
    except Exception:
        return None

    checks = [
        obj.get(
            "model"
        ) == MODEL_ID,
        obj.get(
            "condition"
        ) == "Clean",
        obj.get(
            "split"
        ) == "val",
        bool(
            obj.get(
                "fixed_fusion_audit_passed",
                False,
            )
        ),
        Path(
            str(
                obj.get(
                    "checkpoint",
                    "",
                )
            )
        ).name
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
            == int(
                checkpoint_global_step
            )
        )

    if all(
        checks
    ):
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
        obj = read_json(
            path
        )
    except Exception:
        return None

    checks = [
        obj.get(
            "model"
        ) == MODEL_ID,
        obj.get(
            "condition"
        ) == condition,
        obj.get(
            "split"
        ) == "val",
        obj.get(
            "degradation_protocol_sha256"
        )
        == degradation_protocol_sha256(),
        int(
            obj.get(
                "degradation_implementation_revision",
                -1,
            )
        )
        == IMPLEMENTATION_REVISION,
        Path(
            str(
                obj.get(
                    "checkpoint",
                    "",
                )
            )
        ).name
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
            == int(
                checkpoint_global_step
            )
        )

    if all(
        checks
    ):
        return obj

    return None


# ============================================================================
# Reference A / B validation
# ============================================================================

def validate_model_a_references(
    *,
    a_clean: Mapping[str, Any],
    a_v2_summary: Mapping[str, Any],
) -> None:
    if (
        a_clean.get(
            "model"
        )
        != "A_RGB"
        or a_clean.get(
            "condition"
        )
        != "Clean"
        or a_clean.get(
            "split"
        )
        != "val"
    ):
        raise RuntimeError(
            "Model A Clean reference is invalid."
        )

    if a_v2_summary.get(
        "model"
    ) != "A_RGB":
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
            "Model A v2 summary does not match the FINAL protocol hash."
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


def validate_model_b_references(
    *,
    b_clean: Mapping[str, Any],
    b_v2_summary: Mapping[str, Any],
) -> None:
    if (
        b_clean.get(
            "model"
        )
        != "B_RGBNIR_4CH"
        or b_clean.get(
            "condition"
        )
        != "Clean"
        or b_clean.get(
            "split"
        )
        != "val"
    ):
        raise RuntimeError(
            "Model B Clean reference is invalid."
        )

    if b_v2_summary.get(
        "model"
    ) != "B_RGBNIR_4CH":
        raise RuntimeError(
            "Model B v2 robustness summary is invalid."
        )

    if (
        b_v2_summary.get(
            "degradation_protocol_sha256"
        )
        != degradation_protocol_sha256()
    ):
        raise RuntimeError(
            "Model B robustness summary protocol hash differs from FINAL v2."
        )

    if int(
        b_v2_summary.get(
            "degradation_implementation_revision",
            -1,
        )
    ) != IMPLEMENTATION_REVISION:
        raise RuntimeError(
            "Model B degradation implementation revision mismatch."
        )


def robustness_summary_map(
    summary: Mapping[str, Any],
    *,
    expected_model: str,
) -> Dict[str, Dict[str, Any]]:
    if summary.get(
        "model"
    ) != expected_model:
        raise RuntimeError(
            "Unexpected model in robustness summary."
        )

    rows = summary.get(
        "results"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            "Robustness summary has no results list."
        )

    result = {
        str(
            row[
                "condition"
            ]
        ): dict(
            row
        )
        for row
        in rows
    }

    expected = {
        condition_name(
            corruption,
            level,
        )
        for corruption, level
        in CONDITION_ORDER
    }

    if set(
        result
    ) != expected:
        raise RuntimeError(
            "Robustness summary condition set does not match all nine v2 "
            f"conditions. Got: {sorted(result)}"
        )

    return result


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
        condition.startswith(
            "gaussian_noise_"
        )
        or condition.startswith(
            "gaussian_blur_"
        )
    ):
        return (
            a_v1_dir
            / condition
            / "metrics.json"
        )

    if condition.startswith(
        "rgb_underexposure_"
    ):
        return (
            a_v2_dir
            / condition
            / "metrics.json"
        )

    raise KeyError(
        condition
    )


def model_b_detail_path(
    condition: str,
    *,
    b_clean_path: Path,
    b_v2_dir: Path,
) -> Path:
    if condition == "Clean":
        return b_clean_path

    return (
        b_v2_dir
        / condition
        / "metrics.json"
    )


def per_class_map(
    metrics: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    rows = metrics.get(
        "per_class"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            "metrics.json does not contain per_class."
        )

    result: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for row in rows:
        class_id = int(
            row[
                "class_id"
            ]
        )
        result[
            class_id
        ] = dict(
            row
        )

    if set(
        result
    ) != set(
        range(
            NUM_CLASSES
        )
    ):
        raise RuntimeError(
            "Per-class result does not contain exactly class ids 0..5."
        )

    return result


# ============================================================================
# Three-model comparisons
# ============================================================================

def build_three_model_comparison(
    *,
    a_clean: Mapping[str, Any],
    a_v2_summary: Mapping[str, Any],
    b_clean: Mapping[str, Any],
    b_v2_summary: Mapping[str, Any],
    c_clean: Mapping[str, Any],
    c_results: Mapping[str, Mapping[str, Any]],
    a_clean_path: Path,
    a_v1_dir: Path,
    a_v2_dir: Path,
    b_clean_path: Path,
    b_v2_dir: Path,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    a_by_condition = (
        robustness_summary_map(
            a_v2_summary,
            expected_model="A_RGB",
        )
    )

    b_by_condition = (
        robustness_summary_map(
            b_v2_summary,
            expected_model=(
                "B_RGBNIR_4CH"
            ),
        )
    )

    a_clean_miou = float(
        a_clean[
            "miou"
        ]
    )
    b_clean_miou = float(
        b_clean[
            "miou"
        ]
    )
    c_clean_miou = float(
        c_clean[
            "miou"
        ]
    )

    rows: List[
        Dict[str, Any]
    ] = []

    rows.append(
        {
            "condition": "Clean",
            "corruption": "Clean",
            "severity_level": "",
            "severity_rank": 0,
            "model_a_miou": (
                a_clean_miou
            ),
            "model_b_miou": (
                b_clean_miou
            ),
            "model_c_nogate_miou": (
                c_clean_miou
            ),
            "b_gain_vs_a": (
                b_clean_miou
                - a_clean_miou
            ),
            "c_nogate_gain_vs_a": (
                c_clean_miou
                - a_clean_miou
            ),
            "c_nogate_gain_vs_b": (
                c_clean_miou
                - b_clean_miou
            ),
            "model_a_drop": 0.0,
            "model_b_drop": 0.0,
            "model_c_nogate_drop": 0.0,
            "c_drop_reduction_vs_a": 0.0,
            "c_drop_reduction_vs_b": 0.0,
        }
    )

    for (
        corruption,
        level,
    ) in CONDITION_ORDER:
        condition = condition_name(
            corruption,
            level,
        )

        a_miou = float(
            a_by_condition[
                condition
            ][
                "miou"
            ]
        )

        b_miou = float(
            b_by_condition[
                condition
            ][
                "miou"
            ]
        )

        c_miou = float(
            c_results[
                condition
            ][
                "miou"
            ]
        )

        a_drop = (
            a_clean_miou
            - a_miou
        )
        b_drop = (
            b_clean_miou
            - b_miou
        )
        c_drop = (
            c_clean_miou
            - c_miou
        )

        rows.append(
            {
                "condition": condition,
                "corruption": corruption,
                "severity_level": (
                    level
                ),
                "severity_rank": int(
                    level[
                        1:
                    ]
                ),
                "model_a_miou": (
                    a_miou
                ),
                "model_b_miou": (
                    b_miou
                ),
                "model_c_nogate_miou": (
                    c_miou
                ),
                "b_gain_vs_a": (
                    b_miou
                    - a_miou
                ),
                "c_nogate_gain_vs_a": (
                    c_miou
                    - a_miou
                ),
                "c_nogate_gain_vs_b": (
                    c_miou
                    - b_miou
                ),
                "model_a_drop": (
                    a_drop
                ),
                "model_b_drop": (
                    b_drop
                ),
                "model_c_nogate_drop": (
                    c_drop
                ),
                "c_drop_reduction_vs_a": (
                    a_drop
                    - c_drop
                ),
                "c_drop_reduction_vs_b": (
                    b_drop
                    - c_drop
                ),
            }
        )

    # ------------------------------------------------------------------
    # Six-class comparison for all 10 conditions.
    # ------------------------------------------------------------------
    per_class_rows: List[
        Dict[str, Any]
    ] = []

    for comparison_row in rows:
        condition = str(
            comparison_row[
                "condition"
            ]
        )

        a_metrics = read_json(
            model_a_detail_path(
                condition,
                a_clean_path=(
                    a_clean_path
                ),
                a_v1_dir=(
                    a_v1_dir
                ),
                a_v2_dir=(
                    a_v2_dir
                ),
            )
        )

        b_metrics = read_json(
            model_b_detail_path(
                condition,
                b_clean_path=(
                    b_clean_path
                ),
                b_v2_dir=(
                    b_v2_dir
                ),
            )
        )

        if condition == "Clean":
            c_metrics = (
                c_clean
            )
        else:
            c_metrics = (
                c_results[
                    condition
                ]
            )

        a_classes = (
            per_class_map(
                a_metrics
            )
        )
        b_classes = (
            per_class_map(
                b_metrics
            )
        )
        c_classes = (
            per_class_map(
                c_metrics
            )
        )

        for class_id in range(
            NUM_CLASSES
        ):
            a_iou = float(
                a_classes[
                    class_id
                ][
                    "iou"
                ]
            )
            b_iou = float(
                b_classes[
                    class_id
                ][
                    "iou"
                ]
            )
            c_iou = float(
                c_classes[
                    class_id
                ][
                    "iou"
                ]
            )

            per_class_rows.append(
                {
                    "condition": (
                        condition
                    ),
                    "corruption": (
                        comparison_row[
                            "corruption"
                        ]
                    ),
                    "severity_level": (
                        comparison_row[
                            "severity_level"
                        ]
                    ),
                    "class_id": (
                        class_id
                    ),
                    "class_name": (
                        CLASS_NAMES[
                            class_id
                        ]
                    ),
                    "model_a_iou": (
                        a_iou
                    ),
                    "model_b_iou": (
                        b_iou
                    ),
                    "model_c_nogate_iou": (
                        c_iou
                    ),
                    "b_gain_vs_a_iou": (
                        b_iou
                        - a_iou
                    ),
                    "c_nogate_gain_vs_a_iou": (
                        c_iou
                        - a_iou
                    ),
                    "c_nogate_gain_vs_b_iou": (
                        c_iou
                        - b_iou
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Trend diagnostics.
    # ------------------------------------------------------------------
    trend_analysis: Dict[
        str,
        Any,
    ] = {}

    for corruption in (
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    ):
        family_rows = [
            row
            for row
            in rows
            if row[
                "corruption"
            ]
            == corruption
        ]

        family_rows.sort(
            key=lambda x: int(
                x[
                    "severity_rank"
                ]
            )
        )

        c_mious = [
            float(
                row[
                    "model_c_nogate_miou"
                ]
            )
            for row
            in family_rows
        ]

        gain_vs_a = [
            float(
                row[
                    "c_nogate_gain_vs_a"
                ]
            )
            for row
            in family_rows
        ]

        gain_vs_b = [
            float(
                row[
                    "c_nogate_gain_vs_b"
                ]
            )
            for row
            in family_rows
        ]

        drop_reduction_vs_a = [
            float(
                row[
                    "c_drop_reduction_vs_a"
                ]
            )
            for row
            in family_rows
        ]

        trend_analysis[
            corruption
        ] = {
            "levels": [
                row[
                    "severity_level"
                ]
                for row
                in family_rows
            ],
            "c_nogate_miou": (
                c_mious
            ),
            "c_nogate_gain_vs_a": (
                gain_vs_a
            ),
            "c_nogate_gain_vs_b": (
                gain_vs_b
            ),
            "c_drop_reduction_vs_a": (
                drop_reduction_vs_a
            ),
            "c_miou_monotonic_nonincreasing": all(
                c_mious[
                    index + 1
                ]
                <= c_mious[
                    index
                ]
                + 1e-12
                for index
                in range(
                    len(
                        c_mious
                    )
                    - 1
                )
            ),
            "c_gain_vs_a_monotonic_nondecreasing": all(
                gain_vs_a[
                    index + 1
                ]
                + 1e-12
                >= gain_vs_a[
                    index
                ]
                for index
                in range(
                    len(
                        gain_vs_a
                    )
                    - 1
                )
            ),
            "c_gain_vs_b_monotonic_nondecreasing": all(
                gain_vs_b[
                    index + 1
                ]
                + 1e-12
                >= gain_vs_b[
                    index
                ]
                for index
                in range(
                    len(
                        gain_vs_b
                    )
                    - 1
                )
            ),
        }

    comparison_json = {
        "comparison": (
            "Model A vs Model B vs Model C-noGate"
        ),
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
            "b_gain_vs_a": (
                "Model B mIoU - Model A mIoU"
            ),
            "c_nogate_gain_vs_a": (
                "Model C-noGate mIoU - Model A mIoU"
            ),
            "c_nogate_gain_vs_b": (
                "Model C-noGate mIoU - Model B mIoU"
            ),
            "drop": (
                "same-model Clean mIoU - same-model degraded mIoU"
            ),
            "c_drop_reduction_vs_a": (
                "Model A Drop - Model C-noGate Drop"
            ),
            "c_drop_reduction_vs_b": (
                "Model B Drop - Model C-noGate Drop"
            ),
        },
        "clean": {
            "model_a_miou": (
                a_clean_miou
            ),
            "model_b_miou": (
                b_clean_miou
            ),
            "model_c_nogate_miou": (
                c_clean_miou
            ),
            "b_gain_vs_a": (
                b_clean_miou
                - a_clean_miou
            ),
            "c_nogate_gain_vs_a": (
                c_clean_miou
                - a_clean_miou
            ),
            "c_nogate_gain_vs_b": (
                c_clean_miou
                - b_clean_miou
            ),
        },
        "results": rows,
        "trend_analysis": (
            trend_analysis
        ),
    }

    return (
        rows,
        per_class_rows,
        comparison_json,
    )


def write_three_model_comparison_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    fields = [
        "condition",
        "corruption",
        "severity_level",
        "severity_rank",
        "model_a_miou",
        "model_b_miou",
        "model_c_nogate_miou",
        "b_gain_vs_a",
        "c_nogate_gain_vs_a",
        "c_nogate_gain_vs_b",
        "model_a_drop",
        "model_b_drop",
        "model_c_nogate_drop",
        "c_drop_reduction_vs_a",
        "c_drop_reduction_vs_b",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                    field: (
                        row.get(
                            field
                        )
                    )
                    for field
                    in fields
                }
            )


def write_per_class_comparison_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    fields = [
        "condition",
        "corruption",
        "severity_level",
        "class_id",
        "class_name",
        "model_a_iou",
        "model_b_iou",
        "model_c_nogate_iou",
        "b_gain_vs_a_iou",
        "c_nogate_gain_vs_a_iou",
        "c_nogate_gain_vs_b_iou",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                    field: (
                        row.get(
                            field
                        )
                    )
                    for field
                    in fields
                }
            )


def write_robustness_summary_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                    field: (
                        row.get(
                            field
                        )
                    )
                    for field
                    in fields
                }
            )


# ============================================================================
# Main
# ============================================================================

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

    b_clean_path = resolve_path(
        args.model_b_clean
    )
    b_v2_dir = resolve_path(
        args.model_b_v2_dir
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    # ------------------------------------------------------------------
    # Reference results must already use the final protocol.
    # ------------------------------------------------------------------
    a_clean = read_json(
        a_clean_path
    )
    a_v2_summary = read_json(
        a_v2_dir
        / "robustness_summary.json"
    )

    b_clean = read_json(
        b_clean_path
    )
    b_v2_summary = read_json(
        b_v2_dir
        / "robustness_summary.json"
    )

    validate_model_a_references(
        a_clean=a_clean,
        a_v2_summary=a_v2_summary,
    )

    validate_model_b_references(
        b_clean=b_clean,
        b_v2_summary=b_v2_summary,
    )

    device = get_device(
        args.device
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 104)
    print(
        "MODEL C-noGate | CLEAN + FINAL RGB DEGRADATION PROTOCOL v2"
    )
    print("=" * 104)
    print(
        f"checkpoint       : {checkpoint_path}"
    )
    print(
        f"output root      : {output_root}"
    )
    print(
        f"device / AMP     : {device} / {amp_enabled}"
    )
    print(
        f"batch size       : {args.batch_size}"
    )
    print(
        f"protocol         : {DEGRADATION_PROTOCOL_VERSION}"
    )
    print(
        f"implementation   : revision {IMPLEMENTATION_REVISION}"
    )
    print(
        f"protocol hash    : {degradation_protocol_sha256()}"
    )
    print(
        "fusion           : "
        "four scales, fixed [0.5 RGB, 0.5 NIR]"
    )
    print("=" * 104)

    # ------------------------------------------------------------------
    # Build / load model
    # ------------------------------------------------------------------
    print(
        "[1] building Model C-noGate"
    )

    model, model_meta = (
        build_model_c_nogate(
            PROJECT_ROOT
        )
    )

    print(
        "[2] loading trained checkpoint"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        unwrap_checkpoint(
            checkpoint
        )
    )

    validate_checkpoint_metadata(
        checkpoint_meta
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(
        device
    )
    model.eval()

    checkpoint_epoch = (
        checkpoint_meta.get(
            "epoch"
        )
    )

    checkpoint_global_step = (
        checkpoint_meta.get(
            "global_step",
            checkpoint_meta.get(
                "step"
            ),
        )
    )

    print(
        f"[checkpoint] "
        f"epoch_zero_based={checkpoint_epoch} | "
        f"global_step={checkpoint_global_step}"
    )

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------
    print(
        "[3] Clean validation"
    )

    clean_metrics_path = (
        clean_dir
        / "metrics.json"
    )

    c_clean = None

    if not args.force_clean:
        c_clean = load_existing_clean(
            clean_metrics_path,
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_global_step=(
                int(
                    checkpoint_global_step
                )
                if checkpoint_global_step
                is not None
                else None
            ),
        )

    if c_clean is not None:
        print(
            "[resume] compatible C-noGate Clean result exists; "
            f"skipping: {clean_metrics_path}"
        )
    else:
        clean_dataset = (
            PotsdamSlidingWindowDataset(
                PROJECT_ROOT,
                split="val",
            )
        )

        c_clean = evaluate_dataset(
            model=model,
            dataset=clean_dataset,
            condition="Clean",
            output_dir=clean_dir,
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_epoch=(
                int(
                    checkpoint_epoch
                )
                if checkpoint_epoch
                is not None
                else None
            ),
            checkpoint_global_step=(
                int(
                    checkpoint_global_step
                )
                if checkpoint_global_step
                is not None
                else None
            ),
            batch_size=(
                args.batch_size
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
            log_every=(
                args.log_every
            ),
            confusion_chunk_rows=(
                args.confusion_chunk_rows
            ),
            save_predictions=(
                args.save_predictions
            ),
            audit_fixed_fusion=True,
        )

        del clean_dataset

    c_clean_miou = float(
        c_clean[
            "miou"
        ]
    )

    print(
        "[Clean] "
        f"A={float(a_clean['miou']):.6f} | "
        f"B={float(b_clean['miou']):.6f} | "
        f"C-noGate={c_clean_miou:.6f} | "
        f"C-A={c_clean_miou - float(a_clean['miou']):+.6f} | "
        f"C-B={c_clean_miou - float(b_clean['miou']):+.6f}"
    )

    # ------------------------------------------------------------------
    # Protocol-v2 degraded conditions
    # ------------------------------------------------------------------
    print(
        "[4] Protocol-v2 robustness validation"
    )

    robustness_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_degradation_protocol(
        robustness_dir
        / "degradation_protocol.json"
    )

    c_results: Dict[
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
        condition = condition_name(
            corruption,
            level,
        )

        condition_dir = (
            robustness_dir
            / condition
        )

        metrics_path = (
            condition_dir
            / "metrics.json"
        )

        print()
        print(
            f"[condition {condition_index}/9] "
            f"{condition} "
            f"{condition_spec(corruption, level)}"
        )

        existing = None

        if not args.force_robustness:
            existing = (
                load_existing_degraded(
                    metrics_path,
                    condition=(
                        condition
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    checkpoint_global_step=(
                        int(
                            checkpoint_global_step
                        )
                        if checkpoint_global_step
                        is not None
                        else None
                    ),
                )
            )

        if existing is not None:
            print(
                "[resume] compatible degraded result exists; "
                f"skipping: {metrics_path}"
            )
            result = existing
        else:
            degraded_dataset = (
                DegradedPotsdamSlidingWindowDataset(
                    PROJECT_ROOT,
                    split="val",
                    corruption=(
                        corruption
                    ),
                    level=(
                        level
                    ),
                )
            )

            clean_probe_dataset = (
                PotsdamSlidingWindowDataset(
                    PROJECT_ROOT,
                    split="val",
                )
            )

            degradation_probe(
                clean_dataset=(
                    clean_probe_dataset
                ),
                degraded_dataset=(
                    degraded_dataset
                ),
            )

            del clean_probe_dataset

            result = evaluate_dataset(
                model=model,
                dataset=(
                    degraded_dataset
                ),
                condition=(
                    condition
                ),
                output_dir=(
                    condition_dir
                ),
                checkpoint_path=(
                    checkpoint_path
                ),
                checkpoint_epoch=(
                    int(
                        checkpoint_epoch
                    )
                    if checkpoint_epoch
                    is not None
                    else None
                ),
                checkpoint_global_step=(
                    int(
                        checkpoint_global_step
                    )
                    if checkpoint_global_step
                    is not None
                    else None
                ),
                batch_size=(
                    args.batch_size
                ),
                device=device,
                amp_enabled=(
                    amp_enabled
                ),
                log_every=(
                    args.log_every
                ),
                confusion_chunk_rows=(
                    args.confusion_chunk_rows
                ),
                save_predictions=(
                    args.save_predictions
                ),
                audit_fixed_fusion=False,
                corruption=(
                    corruption
                ),
                severity_level=(
                    level
                ),
            )

            del degraded_dataset

        degraded_miou = float(
            result[
                "miou"
            ]
        )

        drop_miou = (
            c_clean_miou
            - degraded_miou
        )

        result[
            "clean_reference_miou"
        ] = c_clean_miou
        result[
            "drop_miou"
        ] = drop_miou
        result[
            "delta_miou"
        ] = (
            -drop_miou
        )

        result[
            "relative_drop_pct"
        ] = (
            100.0
            * drop_miou
            / c_clean_miou
            if c_clean_miou
            != 0.0
            else None
        )

        result[
            "retention_pct"
        ] = (
            100.0
            * degraded_miou
            / c_clean_miou
            if c_clean_miou
            != 0.0
            else None
        )

        write_json(
            metrics_path,
            result,
        )

        c_results[
            condition
        ] = result

        print(
            f"[{condition}] "
            f"mIoU={degraded_miou:.6f} | "
            f"Drop={drop_miou:.6f} | "
            f"Retention={float(result['retention_pct']):.2f}%"
        )

    # ------------------------------------------------------------------
    # C-noGate robustness summary
    # ------------------------------------------------------------------
    summary_rows: List[
        Dict[str, Any]
    ] = []

    for (
        corruption,
        level,
    ) in CONDITION_ORDER:
        condition = condition_name(
            corruption,
            level,
        )

        result = c_results[
            condition
        ]

        summary_rows.append(
            {
                "model": (
                    MODEL_ID
                ),
                "condition": (
                    condition
                ),
                "corruption": (
                    corruption
                ),
                "severity_level": (
                    level
                ),
                "severity_rank": int(
                    level[
                        1:
                    ]
                ),
                "clean_miou": (
                    c_clean_miou
                ),
                "miou": (
                    result[
                        "miou"
                    ]
                ),
                "drop_miou": (
                    result[
                        "drop_miou"
                    ]
                ),
                "delta_miou": (
                    result[
                        "delta_miou"
                    ]
                ),
                "relative_drop_pct": (
                    result[
                        "relative_drop_pct"
                    ]
                ),
                "retention_pct": (
                    result[
                        "retention_pct"
                    ]
                ),
                "pixel_accuracy": (
                    result[
                        "pixel_accuracy"
                    ]
                ),
                "mean_class_accuracy": (
                    result[
                        "mean_class_accuracy"
                    ]
                ),
                "validation_seconds": (
                    result[
                        "validation_seconds"
                    ]
                ),
            }
        )

    c_trends: Dict[
        str,
        Any,
    ] = {}

    for corruption in (
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    ):
        family_rows = [
            row
            for row
            in summary_rows
            if row[
                "corruption"
            ]
            == corruption
        ]

        family_rows.sort(
            key=lambda x: int(
                x[
                    "severity_rank"
                ]
            )
        )

        mious = [
            float(
                row[
                    "miou"
                ]
            )
            for row
            in family_rows
        ]

        drops = [
            float(
                row[
                    "drop_miou"
                ]
            )
            for row
            in family_rows
        ]

        c_trends[
            corruption
        ] = {
            "levels": [
                row[
                    "severity_level"
                ]
                for row
                in family_rows
            ],
            "miou": (
                mious
            ),
            "drop_miou": (
                drops
            ),
            "monotonic_nonincreasing_miou": all(
                mious[
                    index + 1
                ]
                <= mious[
                    index
                ]
                + 1e-12
                for index
                in range(
                    len(
                        mious
                    )
                    - 1
                )
            ),
            "monotonic_nondecreasing_drop": all(
                drops[
                    index + 1
                ]
                + 1e-12
                >= drops[
                    index
                ]
                for index
                in range(
                    len(
                        drops
                    )
                    - 1
                )
            ),
        }

    robustness_summary = {
        "model": (
            MODEL_ID
        ),
        "model_name": (
            MODEL_NAME
        ),
        "split": "val",
        "clean_reference": {
            "metrics_path": str(
                clean_metrics_path
            ),
            "miou": (
                c_clean_miou
            ),
            "pixel_accuracy": (
                c_clean.get(
                    "pixel_accuracy"
                )
            ),
            "mean_class_accuracy": (
                c_clean.get(
                    "mean_class_accuracy"
                )
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
        "fusion": {
            "quality_gate": False,
            "num_scales": (
                NUM_SCALES
            ),
            "rgb_weight": (
                FIXED_RGB_WEIGHT
            ),
            "nir_weight": (
                FIXED_NIR_WEIGHT
            ),
        },
        "results": (
            summary_rows
        ),
        "trend_analysis": (
            c_trends
        ),
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

    # ------------------------------------------------------------------
    # A / B / C-noGate comparison
    # ------------------------------------------------------------------
    print(
        "[5] computing A / B / C-noGate comparison"
    )

    (
        comparison_rows,
        per_class_rows,
        comparison_json,
    ) = build_three_model_comparison(
        a_clean=(
            a_clean
        ),
        a_v2_summary=(
            a_v2_summary
        ),
        b_clean=(
            b_clean
        ),
        b_v2_summary=(
            b_v2_summary
        ),
        c_clean=(
            c_clean
        ),
        c_results=(
            c_results
        ),
        a_clean_path=(
            a_clean_path
        ),
        a_v1_dir=(
            a_v1_dir
        ),
        a_v2_dir=(
            a_v2_dir
        ),
        b_clean_path=(
            b_clean_path
        ),
        b_v2_dir=(
            b_v2_dir
        ),
    )

    write_json(
        robustness_dir
        / "three_model_comparison.json",
        comparison_json,
    )

    write_three_model_comparison_csv(
        robustness_dir
        / "three_model_comparison.csv",
        comparison_rows,
    )

    write_per_class_comparison_csv(
        robustness_dir
        / "per_class_three_model_comparison.csv",
        per_class_rows,
    )

    # ------------------------------------------------------------------
    # Final terminal report
    # ------------------------------------------------------------------
    print()
    print("=" * 134)
    print(
        "MODEL A vs MODEL B vs MODEL C-noGate | SUMMARY"
    )
    print("=" * 134)
    print(
        f"{'Condition':<28} "
        f"{'A mIoU':>9} "
        f"{'B mIoU':>9} "
        f"{'C-noGate':>10} "
        f"{'C-A':>9} "
        f"{'C-B':>9} "
        f"{'A Drop':>9} "
        f"{'B Drop':>9} "
        f"{'C Drop':>9} "
        f"{'C DropRed A':>11}"
    )
    print("-" * 134)

    for row in comparison_rows:
        print(
            f"{row['condition']:<28} "
            f"{float(row['model_a_miou']):>9.6f} "
            f"{float(row['model_b_miou']):>9.6f} "
            f"{float(row['model_c_nogate_miou']):>10.6f} "
            f"{float(row['c_nogate_gain_vs_a']):>+9.6f} "
            f"{float(row['c_nogate_gain_vs_b']):>+9.6f} "
            f"{float(row['model_a_drop']):>9.6f} "
            f"{float(row['model_b_drop']):>9.6f} "
            f"{float(row['model_c_nogate_drop']):>9.6f} "
            f"{float(row['c_drop_reduction_vs_a']):>+11.6f}"
        )

    print("-" * 134)

    for (
        corruption,
        trend,
    ) in comparison_json[
        "trend_analysis"
    ].items():
        print(
            f"{corruption}: "
            f"C mIoU severity-monotonic="
            f"{trend['c_miou_monotonic_nonincreasing']} | "
            f"C-vs-A gain nondecreasing="
            f"{trend['c_gain_vs_a_monotonic_nondecreasing']} | "
            f"C-vs-B gain nondecreasing="
            f"{trend['c_gain_vs_b_monotonic_nondecreasing']}"
        )

    print("-" * 134)

    clean_comparison = (
        comparison_json[
            "clean"
        ]
    )

    print(
        "Clean | "
        f"C-noGate - A = "
        f"{float(clean_comparison['c_nogate_gain_vs_a']):+.6f} | "
        f"C-noGate - B = "
        f"{float(clean_comparison['c_nogate_gain_vs_b']):+.6f}"
    )

    print(
        f"C-noGate Clean     : "
        f"{clean_dir / 'metrics.json'}"
    )
    print(
        f"C-noGate robustness: "
        f"{robustness_dir / 'robustness_summary.json'}"
    )
    print(
        f"3-model comparison : "
        f"{robustness_dir / 'three_model_comparison.csv'}"
    )
    print(
        f"Per-class compare  : "
        f"{robustness_dir / 'per_class_three_model_comparison.csv'}"
    )

    print("=" * 134)


if __name__ == "__main__":
    main()
