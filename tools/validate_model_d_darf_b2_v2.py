#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Model D / DARF-B2:
SegFormer-B2 + Degradation-Aware Residual Fusion.

Runs:
    Clean

    Gaussian Noise
        L1 / L2 / L3

    Gaussian Blur
        L1 / L2 / L3

    RGB Underexposure
        L1 / L2 / L3

Evaluation protocol
-------------------
Exactly the same semantic-evaluation definition used by the earlier models:

    512x512 sliding windows
        -> full-resolution 6-class logits
        -> overlap MEAN-LOGIT fusion
        -> one 6000x6000 prediction per tile
        -> one GLOBAL confusion matrix over all 6 validation tiles
        -> global mIoU + six class IoUs

The FINAL RGB Degradation Protocol v2 is reused:
- RGB only is degraded.
- NIR is clean / unchanged.
- GT is unchanged.
- full raw tile is degraded before crop and normalization.
- deterministic Gaussian Noise maps are shared with the previous models.

Important scientific note
-------------------------
Model D uses a DIFFERENT training regime from A/B/C-noGate/C:
- SegFormer-B2 instead of B0.
- corruption-aware RGB training.
- CE + Lovasz + gate-quality auxiliary loss.

Therefore this script reports five-model performance comparisons, but labels
Model D as an "enhanced robust-training model", NOT a pure architecture-only
ablation against the clean-trained B0 models.

DARF gate semantics
-------------------
The gate is NIR residual strength:

    F_fused_i = F_RGB_i + g_NIR_i * Adapter_i(F_NIR_i)

So interpretability should test:

    Clean -> L1 -> L2 -> L3

and ask whether mean g_NIR is NON-DECREASING as RGB quality gets worse.

Per-window gate records
-----------------------
For every validation window and every scale this script stores:

    condition
    tile_id
    window_index
    x
    y
    scale
    g_nir
    gate_logit

Outputs
-------
outputs/evaluation/model_d_darf_b2/
├── clean_val/
│   ├── metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   ├── per_tile_metrics.jsonl
│   └── gate_strength_windows.csv
│
└── robustness_val_v2/
    ├── degradation_protocol.json
    ├── robustness_summary.json
    ├── robustness_summary.csv
    ├── five_model_comparison.json
    ├── five_model_comparison.csv
    ├── per_class_five_model_comparison.csv
    ├── gate_statistics.json
    ├── gate_statistics.csv
    ├── gate_monotonicity.json
    └── target_check.json

Run
---
    python tools/validate_model_d_darf_b2_v2.py

If evaluation OOMs:
    python tools/validate_model_d_darf_b2_v2.py --batch-size 1
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


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for item in (PROJECT_ROOT, TOOLS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL_VERSION,
    IMPLEMENTATION_REVISION,
    DegradedPotsdamSlidingWindowDataset,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    write_degradation_protocol,
)
from models.segformer_b2_darf import (
    GATE_TYPE,
    MODEL_ID,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_SCALES,
    build_model_d_darf_b2,
)
from validate_model_a_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    confusion_from_prediction,
    metrics_from_confusion,
    write_confusion_csv,
    write_per_class_csv,
)


# =============================================================================
# Constants / reference paths
# =============================================================================

MODEL_A_ID = "A_RGB"
MODEL_B_ID = "B_RGBNIR_4CH"
MODEL_FIXED_ID = "C_NOGATE_FIXED_FUSION"
MODEL_C_ID = "C_QUALITY_GATE"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_d_darf_b2"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_d_darf_b2"
)

A_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "clean_val"
    / "metrics.json"
)
A_V1 = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val"
)
A_V2 = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val_v2"
)

B_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_b_rgbnir"
    / "clean_val"
    / "metrics.json"
)
B_V2 = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_b_rgbnir"
    / "robustness_val_v2"
)

FIXED_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_c_nogate"
    / "clean_val"
    / "metrics.json"
)
FIXED_V2 = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_c_nogate"
    / "robustness_val_v2"
)

C_CLEAN = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_c_quality_gate"
    / "clean_val"
    / "metrics.json"
)
C_V2 = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_c_quality_gate"
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

CLEAN_MIOU_TARGET = 0.80


# =============================================================================
# Generic helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Model D / DARF-B2 on Clean + final Protocol-v2 "
            "degradations and compare against A/B/C-noGate/C."
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
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
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


def resolve(path: Path) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        obj = json.load(file)

    if not isinstance(
        obj,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return obj


def save_json(
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
    ) as file:
        file.write(
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
    device = torch.device(
        device_arg
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    return device


# =============================================================================
# Model D checkpoint
# =============================================================================

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
            "Checkpoint does not contain checkpoint['model']."
        )

    state_dict = checkpoint[
        "model"
    ]

    if not isinstance(
        state_dict,
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

    return (
        state_dict,
        metadata,
    )


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    if metadata.get(
        "model_id"
    ) not in (
        None,
        MODEL_ID,
    ):
        raise RuntimeError(
            "Checkpoint model_id does not match Model D."
        )

    protocol = metadata.get(
        "protocol"
    )

    if not isinstance(
        protocol,
        Mapping,
    ):
        raise RuntimeError(
            "Model D checkpoint has no protocol metadata."
        )

    if protocol.get(
        "model"
    ) != MODEL_ID:
        raise RuntimeError(
            "Checkpoint protocol is not Model D / DARF-B2."
        )

    if protocol.get(
        "backbone"
    ) != "SegFormer-B2":
        raise RuntimeError(
            "Checkpoint backbone is not SegFormer-B2."
        )

    if not bool(
        protocol.get(
            "quality_gate",
            False,
        )
    ):
        raise RuntimeError(
            "Model D checkpoint does not enable the quality gate."
        )

    corruption_training = protocol.get(
        "corruption_training"
    )

    if not isinstance(
        corruption_training,
        Mapping,
    ) or not bool(
        corruption_training.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "Model D checkpoint is not the degradation-aware training run."
        )


# =============================================================================
# Gate statistics
# =============================================================================

class GateAccumulator:
    def __init__(
        self,
        num_scales: int,
    ):
        self.num_scales = int(
            num_scales
        )

        self.values: List[
            List[float]
        ] = [
            []
            for _ in range(
                self.num_scales
            )
        ]

    def update(
        self,
        gate_strength: torch.Tensor,
    ) -> None:
        if (
            gate_strength.ndim != 2
            or gate_strength.shape[1]
            != self.num_scales
        ):
            raise RuntimeError(
                "Invalid DARF gate-strength shape: "
                f"{tuple(gate_strength.shape)}"
            )

        arr = (
            gate_strength
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if not np.isfinite(
            arr
        ).all():
            raise RuntimeError(
                "DARF gate strength contains NaN/Inf."
            )

        if (
            arr.min() < 0.0
            or arr.max() > 1.0
        ):
            raise RuntimeError(
                "DARF gate strength escaped [0,1]."
            )

        for scale_index in range(
            self.num_scales
        ):
            self.values[
                scale_index
            ].extend(
                arr[
                    :,
                    scale_index,
                ]
                .astype(
                    np.float64
                )
                .tolist()
            )

    def finalize(
        self,
        *,
        condition: str,
        corruption: str,
        severity_level: str,
        severity_rank: int,
    ) -> List[Dict[str, Any]]:
        rows = []

        for scale_index, values in enumerate(
            self.values,
            start=1,
        ):
            arr = np.asarray(
                values,
                dtype=np.float64,
            )

            if arr.size == 0:
                raise RuntimeError(
                    f"No gate samples for scale {scale_index}."
                )

            q = np.quantile(
                arr,
                [
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                ],
            )

            rows.append(
                {
                    "condition": (
                        condition
                    ),
                    "corruption": (
                        corruption
                    ),
                    "severity_level": (
                        severity_level
                    ),
                    "severity_rank": (
                        severity_rank
                    ),
                    "scale": (
                        scale_index
                    ),
                    "count": int(
                        arr.size
                    ),
                    "g_nir_mean": float(
                        arr.mean()
                    ),
                    "g_nir_std": float(
                        arr.std(
                            ddof=0
                        )
                    ),
                    "g_nir_min": float(
                        arr.min()
                    ),
                    "g_nir_p05": float(
                        q[
                            0
                        ]
                    ),
                    "g_nir_p25": float(
                        q[
                            1
                        ]
                    ),
                    "g_nir_median": float(
                        q[
                            2
                        ]
                    ),
                    "g_nir_p75": float(
                        q[
                            3
                        ]
                    ),
                    "g_nir_p95": float(
                        q[
                            4
                        ]
                    ),
                    "g_nir_max": float(
                        arr.max()
                    ),
                }
            )

        return rows


GATE_WINDOW_FIELDS = [
    "condition",
    "corruption",
    "severity_level",
    "tile_id",
    "window_index",
    "x",
    "y",
    "scale",
    "g_nir",
    "gate_logit",
]


def open_gate_csv(
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = path.open(
        "w",
        encoding="utf-8",
        newline="",
    )

    writer = csv.DictWriter(
        handle,
        fieldnames=(
            GATE_WINDOW_FIELDS
        ),
    )

    writer.writeheader()

    return (
        handle,
        writer,
    )


def write_gate_batch(
    *,
    writer: csv.DictWriter,
    condition: str,
    corruption: str,
    severity_level: str,
    batch: Mapping[str, Any],
    strength: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    g = (
        strength
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    z = (
        logits
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    batch_size = int(
        g.shape[
            0
        ]
    )

    if g.shape != (
        batch_size,
        NUM_SCALES,
    ):
        raise RuntimeError(
            f"Bad DARF gate shape: {g.shape}"
        )

    if z.shape != (
        batch_size,
        NUM_SCALES,
    ):
        raise RuntimeError(
            f"Bad DARF gate-logit shape: {z.shape}"
        )

    for sample_index in range(
        batch_size
    ):
        for scale_index in range(
            NUM_SCALES
        ):
            writer.writerow(
                {
                    "condition": (
                        condition
                    ),
                    "corruption": (
                        corruption
                    ),
                    "severity_level": (
                        severity_level
                    ),
                    "tile_id": str(
                        batch[
                            "tile_id"
                        ][
                            sample_index
                        ]
                    ),
                    "window_index": int(
                        batch[
                            "window_index"
                        ][
                            sample_index
                        ]
                    ),
                    "x": int(
                        batch[
                            "x"
                        ][
                            sample_index
                        ]
                    ),
                    "y": int(
                        batch[
                            "y"
                        ][
                            sample_index
                        ]
                    ),
                    "scale": (
                        scale_index
                        + 1
                    ),
                    "g_nir": float(
                        g[
                            sample_index,
                            scale_index,
                        ]
                    ),
                    "gate_logit": float(
                        z[
                            sample_index,
                            scale_index,
                        ]
                    ),
                }
            )


# =============================================================================
# Sanity probes
# =============================================================================

def validate_batch(
    batch: Mapping[str, Any],
    *,
    tile_id: str,
) -> None:
    required = {
        "rgb",
        "nir",
        "tile_id",
        "window_index",
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

    rgb = batch[
        "rgb"
    ]

    nir = batch[
        "nir"
    ]

    if (
        rgb.ndim != 4
        or rgb.shape[1] != 3
    ):
        raise RuntimeError(
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}."
        )

    if (
        nir.ndim != 4
        or nir.shape[1] != 1
    ):
        raise RuntimeError(
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}."
        )

    if (
        rgb.shape[0] != nir.shape[0]
        or rgb.shape[-2:] != nir.shape[-2:]
    ):
        raise RuntimeError(
            "RGB/NIR batch alignment changed."
        )

    if any(
        str(
            value
        )
        != tile_id
        for value
        in list(
            batch[
                "tile_id"
            ]
        )
    ):
        raise RuntimeError(
            "Per-tile loader mixed tile IDs."
        )


def degradation_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: DegradedPotsdamSlidingWindowDataset,
) -> None:
    clean = clean_dataset[
        0
    ]

    degraded = degraded_dataset[
        0
    ]

    for key in (
        "tile_id",
        "window_index",
        "x",
        "y",
        "height",
        "width",
    ):
        if clean[
            key
        ] != degraded[
            key
        ]:
            raise RuntimeError(
                f"Degradation changed frozen metadata key={key}."
            )

    if not torch.equal(
        clean[
            "nir"
        ],
        degraded[
            "nir"
        ],
    ):
        raise RuntimeError(
            "NIR changed under RGB-only degradation."
        )

    if torch.equal(
        clean[
            "rgb"
        ],
        degraded[
            "rgb"
        ],
    ):
        raise RuntimeError(
            "Degraded RGB equals Clean RGB on probe window."
        )

    print(
        "[degradation probe] PASS | "
        "same window | NIR bit-identical | RGB changed"
    )


# =============================================================================
# Full-tile inference
# =============================================================================

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
    condition: str,
    corruption: str,
    severity_level: str,
    gate_writer: csv.DictWriter,
    gate_accumulator: GateAccumulator,
) -> tuple[
    np.ndarray,
    Dict[str, Any],
]:
    windows_per_tile = len(
        dataset.window_coordinates
    )

    subset = Subset(
        dataset,
        range(
            tile_index
            * windows_per_tile,
            (
                tile_index
                + 1
            )
            * windows_per_tile,
        ),
    )

    loader = DataLoader(
        subset,
        batch_size=(
            batch_size
        ),
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

    # Keep fusion definition identical to previous validators.
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

    started = (
        time.time()
    )

    windows_seen = 0

    for batch_index, batch in enumerate(
        loader
    ):
        validate_batch(
            batch,
            tile_id=(
                tile_id
            ),
        )

        rgb = (
            batch[
                "rgb"
            ]
            .to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )
        )

        nir = (
            batch[
                "nir"
            ]
            .to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=(
                    device.type
                ),
                dtype=torch.float16,
                enabled=(
                    amp_enabled
                ),
            ):
                details = model(
                    rgb,
                    nir,
                    return_details=True,
                )

        logits = details[
            "logits"
        ]

        strength = details[
            "nir_gate_strength"
        ]

        gate_logits = details[
            "nir_gate_logits"
        ]

        if (
            logits.ndim != 4
            or logits.shape[1]
            != NUM_CLASSES
            or tuple(
                logits.shape[
                    -2:
                ]
            )
            != (
                crop_size,
                crop_size,
            )
        ):
            raise RuntimeError(
                "Unexpected DARF logits shape: "
                f"{tuple(logits.shape)}."
            )

        if not torch.isfinite(
            logits
        ).all().item():
            raise FloatingPointError(
                "DARF logits contain NaN/Inf."
            )

        gate_accumulator.update(
            strength
        )

        write_gate_batch(
            writer=(
                gate_writer
            ),
            condition=(
                condition
            ),
            corruption=(
                corruption
            ),
            severity_level=(
                severity_level
            ),
            batch=(
                batch
            ),
            strength=(
                strength
            ),
            logits=(
                gate_logits
            ),
        )

        logits = logits.float()

        for local_index in range(
            int(
                logits.shape[
                    0
                ]
            )
        ):
            x = int(
                batch[
                    "x"
                ][
                    local_index
                ]
            )

            y = int(
                batch[
                    "y"
                ][
                    local_index
                ]
            )

            logits_sum[
                :,
                y:y
                + crop_size,
                x:x
                + crop_size,
            ].add_(
                logits[
                    local_index
                ]
            )

            coverage[
                y:y
                + crop_size,
                x:x
                + crop_size,
            ].add_(
                1.0
            )

        windows_seen += int(
            logits.shape[
                0
            ]
        )

        if (
            batch_index
            % log_every
            == 0
            or batch_index
            + 1
            == len(
                loader
            )
        ):
            gate_mean = (
                strength
                .detach()
                .float()
                .mean(
                    dim=0
                )
                .cpu()
                .tolist()
            )

            print(
                f"  tile {tile_id} | "
                f"batch {batch_index + 1:03d}/{len(loader):03d} | "
                f"windows {windows_seen:03d}/{windows_per_tile:03d} | "
                f"gNIR={[round(v, 4) for v in gate_mean]}",
                flush=True,
            )

        del details
        del logits
        del strength
        del gate_logits
        del rgb
        del nir

    if windows_seen != windows_per_tile:
        raise RuntimeError(
            f"{tile_id}: expected {windows_per_tile} windows, got {windows_seen}."
        )

    if float(
        coverage
        .min()
        .item()
    ) <= 0.0:
        raise RuntimeError(
            f"{tile_id}: uncovered full-tile pixels."
        )

    coverage_min = float(
        coverage
        .min()
        .item()
    )

    coverage_max = float(
        coverage
        .max()
        .item()
    )

    logits_sum.div_(
        coverage.unsqueeze(
            0
        )
    )

    prediction = (
        logits_sum
        .argmax(
            dim=0
        )
        .to(
            torch.uint8
        )
        .cpu()
        .numpy()
    )

    elapsed = (
        time.time()
        - started
    )

    del logits_sum
    del coverage

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        prediction,
        {
            "tile_id": (
                tile_id
            ),
            "windows": (
                windows_seen
            ),
            "coverage_min": (
                coverage_min
            ),
            "coverage_max": (
                coverage_max
            ),
            "inference_seconds": (
                elapsed
            ),
        },
    )


# =============================================================================
# Evaluate one condition
# =============================================================================

def evaluate_condition(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    condition: str,
    corruption: str,
    severity_level: str,
    severity_rank: int,
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
) -> tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
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

    (
        gate_handle,
        gate_writer,
    ) = open_gate_csv(
        output_dir
        / "gate_strength_windows.csv"
    )

    gate_accumulator = (
        GateAccumulator(
            NUM_SCALES
        )
    )

    tile_ids = list(
        dataset.tile_ids
    )

    if (
        len(
            tile_ids
        )
        != 6
        or len(
            dataset.window_coordinates
        )
        != 256
    ):
        gate_handle.close()

        raise RuntimeError(
            "Frozen validation protocol changed."
        )

    global_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    tile_rows: List[
        Dict[str, Any]
    ] = []

    started = (
        time.time()
    )

    try:
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
                runtime,
            ) = infer_one_tile(
                model=(
                    model
                ),
                dataset=(
                    dataset
                ),
                tile_id=(
                    tile_id
                ),
                tile_index=(
                    tile_index
                ),
                batch_size=(
                    batch_size
                ),
                device=(
                    device
                ),
                amp_enabled=(
                    amp_enabled
                ),
                log_every=(
                    log_every
                ),
                condition=(
                    condition
                ),
                corruption=(
                    corruption
                ),
                severity_level=(
                    severity_level
                ),
                gate_writer=(
                    gate_writer
                ),
                gate_accumulator=(
                    gate_accumulator
                ),
            )

            target_t = (
                dataset
                .load_full_label(
                    tile_id
                )
            )

            target = (
                target_t
                .numpy()
            )

            tile_confusion = (
                confusion_from_prediction(
                    prediction,
                    target,
                    num_classes=(
                        NUM_CLASSES
                    ),
                    ignore_index=(
                        IGNORE_INDEX
                    ),
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
                **runtime,
                "condition": (
                    condition
                ),
                "corruption": (
                    corruption
                ),
                "severity_level": (
                    severity_level
                ),
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

            tile_rows.append(
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
                f"  tile mIoU={tile_metrics['miou']:.6f} | "
                f"time={runtime['inference_seconds']:.1f}s"
            )

            if isinstance(
                dataset,
                DegradedPotsdamSlidingWindowDataset,
            ):
                dataset.clear_degradation_cache()

            del prediction
            del target
            del target_t

    finally:
        gate_handle.close()

    global_metrics = (
        metrics_from_confusion(
            global_confusion,
            CLASS_NAMES,
        )
    )

    gate_statistics = (
        gate_accumulator
        .finalize(
            condition=(
                condition
            ),
            corruption=(
                corruption
            ),
            severity_level=(
                severity_level
            ),
            severity_rank=(
                severity_rank
            ),
        )
    )

    result: Dict[
        str,
        Any,
    ] = {
        "model": (
            MODEL_ID
        ),
        "model_name": (
            MODEL_NAME
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "training_regime": (
            "degradation-aware robust training"
        ),
        "condition": (
            condition
        ),
        "corruption": (
            corruption
        ),
        "severity_level": (
            severity_level
        ),
        "severity_rank": (
            severity_rank
        ),
        "split": (
            "val"
        ),
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
        "tiles": (
            tile_rows
        ),
        "num_tiles": (
            len(
                tile_ids
            )
        ),
        "windows_per_tile": (
            256
        ),
        "total_windows": (
            len(
                dataset
            )
        ),
        "validation_seconds": (
            time.time()
            - started
        ),
        "gate_semantics": (
            "g_NIR residual correction strength; larger means more NIR correction"
        ),
        "gate_statistics": (
            gate_statistics
        ),
        "gate_strength_windows_csv": str(
            output_dir
            / "gate_strength_windows.csv"
        ),
    }

    if condition != "Clean":
        result.update(
            {
                "severity_parameters": (
                    condition_spec(
                        corruption,
                        severity_level,
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
                "rgb_degraded": (
                    True
                ),
                "nir_source": (
                    "clean / unchanged"
                ),
            }
        )

    save_json(
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

    return (
        result,
        gate_statistics,
    )


# =============================================================================
# Resume
# =============================================================================

def compatible_existing(
    path: Path,
    *,
    condition: str,
    checkpoint_path: Path,
    checkpoint_global_step: Optional[int],
    degraded: bool,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = load_json(
            path
        )
    except Exception:
        return None

    checks = [
        obj.get(
            "model"
        )
        == MODEL_ID,
        obj.get(
            "condition"
        )
        == condition,
        obj.get(
            "split"
        )
        == "val",
        Path(
            str(
                obj.get(
                    "checkpoint",
                    "",
                )
            )
        ).name
        == checkpoint_path.name,
        isinstance(
            obj.get(
                "gate_statistics"
            ),
            list,
        ),
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

    if degraded:
        checks.extend(
            [
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
            ]
        )

    return (
        obj
        if all(
            checks
        )
        else None
    )


# =============================================================================
# Reference model comparisons
# =============================================================================

def validate_reference_clean(
    obj: Mapping[str, Any],
    *,
    expected_model: str,
    label: str,
) -> None:
    if (
        obj.get(
            "model"
        )
        != expected_model
        or obj.get(
            "condition"
        )
        != "Clean"
        or obj.get(
            "split"
        )
        != "val"
    ):
        raise RuntimeError(
            f"{label} Clean reference is invalid."
        )


def validate_reference_summary(
    obj: Mapping[str, Any],
    *,
    expected_model: str,
    label: str,
) -> None:
    if obj.get(
        "model"
    ) != expected_model:
        raise RuntimeError(
            f"{label} summary model id mismatch."
        )

    if (
        obj.get(
            "degradation_protocol_sha256"
        )
        != degradation_protocol_sha256()
    ):
        raise RuntimeError(
            f"{label} summary does not use FINAL v2 protocol."
        )

    if int(
        obj.get(
            "degradation_implementation_revision",
            -1,
        )
    ) != IMPLEMENTATION_REVISION:
        raise RuntimeError(
            f"{label} degradation implementation revision mismatch."
        )


def summary_map(
    obj: Mapping[str, Any],
    *,
    expected_model: str,
) -> Dict[str, Dict[str, Any]]:
    if obj.get(
        "model"
    ) != expected_model:
        raise RuntimeError(
            "Reference summary model mismatch."
        )

    rows = {
        str(
            row[
                "condition"
            ]
        ): dict(
            row
        )
        for row
        in obj[
            "results"
        ]
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
        rows
    ) != expected:
        raise RuntimeError(
            "Reference robustness condition set mismatch."
        )

    return rows


def build_five_model_comparison(
    *,
    a_clean,
    a_summary,
    b_clean,
    b_summary,
    fixed_clean,
    fixed_summary,
    c_clean,
    c_summary,
    d_clean,
    d_results,
):
    maps = {
        "A": summary_map(
            a_summary,
            expected_model=(
                MODEL_A_ID
            ),
        ),
        "B": summary_map(
            b_summary,
            expected_model=(
                MODEL_B_ID
            ),
        ),
        "Fixed": summary_map(
            fixed_summary,
            expected_model=(
                MODEL_FIXED_ID
            ),
        ),
        "C": summary_map(
            c_summary,
            expected_model=(
                MODEL_C_ID
            ),
        ),
    }

    clean_values = {
        "A": float(
            a_clean[
                "miou"
            ]
        ),
        "B": float(
            b_clean[
                "miou"
            ]
        ),
        "Fixed": float(
            fixed_clean[
                "miou"
            ]
        ),
        "C": float(
            c_clean[
                "miou"
            ]
        ),
        "D": float(
            d_clean[
                "miou"
            ]
        ),
    }

    rows = [
        {
            "condition": (
                "Clean"
            ),
            "corruption": (
                "Clean"
            ),
            "severity_level": (
                ""
            ),
            "severity_rank": (
                0
            ),
            "model_a_miou": (
                clean_values[
                    "A"
                ]
            ),
            "model_b_miou": (
                clean_values[
                    "B"
                ]
            ),
            "model_c_nogate_miou": (
                clean_values[
                    "Fixed"
                ]
            ),
            "model_c_miou": (
                clean_values[
                    "C"
                ]
            ),
            "model_d_miou": (
                clean_values[
                    "D"
                ]
            ),
            "d_gain_vs_a": (
                clean_values[
                    "D"
                ]
                - clean_values[
                    "A"
                ]
            ),
            "d_gain_vs_b": (
                clean_values[
                    "D"
                ]
                - clean_values[
                    "B"
                ]
            ),
            "d_gain_vs_fixed": (
                clean_values[
                    "D"
                ]
                - clean_values[
                    "Fixed"
                ]
            ),
            "d_gain_vs_c": (
                clean_values[
                    "D"
                ]
                - clean_values[
                    "C"
                ]
            ),
            "model_a_drop": (
                0.0
            ),
            "model_b_drop": (
                0.0
            ),
            "model_c_nogate_drop": (
                0.0
            ),
            "model_c_drop": (
                0.0
            ),
            "model_d_drop": (
                0.0
            ),
        }
    ]

    for corruption, level in CONDITION_ORDER:
        condition = condition_name(
            corruption,
            level,
        )

        values = {
            "A": float(
                maps[
                    "A"
                ][
                    condition
                ][
                    "miou"
                ]
            ),
            "B": float(
                maps[
                    "B"
                ][
                    condition
                ][
                    "miou"
                ]
            ),
            "Fixed": float(
                maps[
                    "Fixed"
                ][
                    condition
                ][
                    "miou"
                ]
            ),
            "C": float(
                maps[
                    "C"
                ][
                    condition
                ][
                    "miou"
                ]
            ),
            "D": float(
                d_results[
                    condition
                ][
                    "miou"
                ]
            ),
        }

        drops = {
            key: (
                clean_values[
                    key
                ]
                - values[
                    key
                ]
            )
            for key in values
        }

        rows.append(
            {
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
                "model_a_miou": (
                    values[
                        "A"
                    ]
                ),
                "model_b_miou": (
                    values[
                        "B"
                    ]
                ),
                "model_c_nogate_miou": (
                    values[
                        "Fixed"
                    ]
                ),
                "model_c_miou": (
                    values[
                        "C"
                    ]
                ),
                "model_d_miou": (
                    values[
                        "D"
                    ]
                ),
                "d_gain_vs_a": (
                    values[
                        "D"
                    ]
                    - values[
                        "A"
                    ]
                ),
                "d_gain_vs_b": (
                    values[
                        "D"
                    ]
                    - values[
                        "B"
                    ]
                ),
                "d_gain_vs_fixed": (
                    values[
                        "D"
                    ]
                    - values[
                        "Fixed"
                    ]
                ),
                "d_gain_vs_c": (
                    values[
                        "D"
                    ]
                    - values[
                        "C"
                    ]
                ),
                "model_a_drop": (
                    drops[
                        "A"
                    ]
                ),
                "model_b_drop": (
                    drops[
                        "B"
                    ]
                ),
                "model_c_nogate_drop": (
                    drops[
                        "Fixed"
                    ]
                ),
                "model_c_drop": (
                    drops[
                        "C"
                    ]
                ),
                "model_d_drop": (
                    drops[
                        "D"
                    ]
                ),
            }
        )

    degraded_rows = rows[
        1:
    ]

    mean_drop = {
        "A": float(
            np.mean(
                [
                    row[
                        "model_a_drop"
                    ]
                    for row in degraded_rows
                ]
            )
        ),
        "B": float(
            np.mean(
                [
                    row[
                        "model_b_drop"
                    ]
                    for row in degraded_rows
                ]
            )
        ),
        "C_noGate": float(
            np.mean(
                [
                    row[
                        "model_c_nogate_drop"
                    ]
                    for row in degraded_rows
                ]
            )
        ),
        "C": float(
            np.mean(
                [
                    row[
                        "model_c_drop"
                    ]
                    for row in degraded_rows
                ]
            )
        ),
        "D": float(
            np.mean(
                [
                    row[
                        "model_d_drop"
                    ]
                    for row in degraded_rows
                ]
            )
        ),
    }

    condition_best = {}

    for row in rows:
        values = {
            "A": float(
                row[
                    "model_a_miou"
                ]
            ),
            "B": float(
                row[
                    "model_b_miou"
                ]
            ),
            "C_noGate": float(
                row[
                    "model_c_nogate_miou"
                ]
            ),
            "C": float(
                row[
                    "model_c_miou"
                ]
            ),
            "D": float(
                row[
                    "model_d_miou"
                ]
            ),
        }

        maximum = max(
            values.values()
        )

        condition_best[
            str(
                row[
                    "condition"
                ]
            )
        ] = [
            key
            for key, value
            in values.items()
            if math.isclose(
                value,
                maximum,
                abs_tol=1e-12,
                rel_tol=0.0,
            )
        ]

    return (
        rows,
        {
            "comparison": (
                "A/B/C-noGate/C are clean-trained B0 baselines; "
                "D is B2 + degradation-aware robust training."
            ),
            "warning": (
                "D versus A/B/C-noGate/C is a system-level comparison, "
                "not a pure architecture-only ablation."
            ),
            "clean_miou_target": (
                CLEAN_MIOU_TARGET
            ),
            "clean_miou_target_met": (
                clean_values[
                    "D"
                ]
                >= CLEAN_MIOU_TARGET
            ),
            "clean": (
                clean_values
            ),
            "results": (
                rows
            ),
            "mean_degraded_drop": (
                mean_drop
            ),
            "best_miou_model_by_condition": (
                condition_best
            ),
        },
    )


FIVE_FIELDS = [
    "condition",
    "corruption",
    "severity_level",
    "severity_rank",
    "model_a_miou",
    "model_b_miou",
    "model_c_nogate_miou",
    "model_c_miou",
    "model_d_miou",
    "d_gain_vs_a",
    "d_gain_vs_b",
    "d_gain_vs_fixed",
    "d_gain_vs_c",
    "model_a_drop",
    "model_b_drop",
    "model_c_nogate_drop",
    "model_c_drop",
    "model_d_drop",
]


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
    fields: Sequence[
        str
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                list(
                    fields
                )
            ),
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
                    for field in fields
                }
            )


# =============================================================================
# Per-class five-model comparison
# =============================================================================

def model_a_detail_path(
    condition: str,
) -> Path:
    if condition == "Clean":
        return A_CLEAN

    if (
        condition.startswith(
            "gaussian_noise_"
        )
        or condition.startswith(
            "gaussian_blur_"
        )
    ):
        return (
            A_V1
            / condition
            / "metrics.json"
        )

    return (
        A_V2
        / condition
        / "metrics.json"
    )


def standard_detail_path(
    condition: str,
    *,
    clean_path: Path,
    v2_dir: Path,
) -> Path:
    if condition == "Clean":
        return clean_path

    return (
        v2_dir
        / condition
        / "metrics.json"
    )


def per_class_map(
    metrics: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(
            row[
                "class_id"
            ]
        ): dict(
            row
        )
        for row
        in metrics[
            "per_class"
        ]
    }


def build_per_class_comparison(
    *,
    five_rows,
    d_clean,
    d_results,
):
    output = []

    for comparison_row in five_rows:
        condition = str(
            comparison_row[
                "condition"
            ]
        )

        a = load_json(
            model_a_detail_path(
                condition
            )
        )

        b = load_json(
            standard_detail_path(
                condition,
                clean_path=(
                    B_CLEAN
                ),
                v2_dir=(
                    B_V2
                ),
            )
        )

        fixed = load_json(
            standard_detail_path(
                condition,
                clean_path=(
                    FIXED_CLEAN
                ),
                v2_dir=(
                    FIXED_V2
                ),
            )
        )

        c = load_json(
            standard_detail_path(
                condition,
                clean_path=(
                    C_CLEAN
                ),
                v2_dir=(
                    C_V2
                ),
            )
        )

        d = (
            d_clean
            if condition
            == "Clean"
            else d_results[
                condition
            ]
        )

        maps = {
            "A": per_class_map(
                a
            ),
            "B": per_class_map(
                b
            ),
            "Fixed": per_class_map(
                fixed
            ),
            "C": per_class_map(
                c
            ),
            "D": per_class_map(
                d
            ),
        }

        for class_id in range(
            NUM_CLASSES
        ):
            values = {
                key: float(
                    value[
                        class_id
                    ][
                        "iou"
                    ]
                )
                for key, value
                in maps.items()
            }

            output.append(
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
                        values[
                            "A"
                        ]
                    ),
                    "model_b_iou": (
                        values[
                            "B"
                        ]
                    ),
                    "model_c_nogate_iou": (
                        values[
                            "Fixed"
                        ]
                    ),
                    "model_c_iou": (
                        values[
                            "C"
                        ]
                    ),
                    "model_d_iou": (
                        values[
                            "D"
                        ]
                    ),
                    "d_gain_vs_a_iou": (
                        values[
                            "D"
                        ]
                        - values[
                            "A"
                        ]
                    ),
                    "d_gain_vs_b_iou": (
                        values[
                            "D"
                        ]
                        - values[
                            "B"
                        ]
                    ),
                    "d_gain_vs_fixed_iou": (
                        values[
                            "D"
                        ]
                        - values[
                            "Fixed"
                        ]
                    ),
                    "d_gain_vs_c_iou": (
                        values[
                            "D"
                        ]
                        - values[
                            "C"
                        ]
                    ),
                }
            )

    return output


# =============================================================================
# DARF gate monotonicity
# =============================================================================

def gate_monotonicity(
    gate_rows: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    lookup = {
        (
            str(
                row[
                    "condition"
                ]
            ),
            int(
                row[
                    "scale"
                ]
            ),
        ): row
        for row in gate_rows
    }

    output = {}

    for corruption in (
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    ):
        sequence = [
            "Clean",
            condition_name(
                corruption,
                "L1",
            ),
            condition_name(
                corruption,
                "L2",
            ),
            condition_name(
                corruption,
                "L3",
            ),
        ]

        scales = {}

        for scale in range(
            1,
            NUM_SCALES
            + 1,
        ):
            means = [
                float(
                    lookup[
                        (
                            condition,
                            scale,
                        )
                    ][
                        "g_nir_mean"
                    ]
                )
                for condition
                in sequence
            ]

            nondecreasing = all(
                means[
                    index
                    + 1
                ]
                + 1e-12
                >= means[
                    index
                ]
                for index in range(
                    len(
                        means
                    )
                    - 1
                )
            )

            scales[
                f"scale_{scale}"
            ] = {
                "g_nir_mean": (
                    means
                ),
                "delta_from_clean": [
                    value
                    - means[
                        0
                    ]
                    for value
                    in means
                ],
                "monotonic_nondecreasing": (
                    nondecreasing
                ),
            }

        output[
            corruption
        ] = {
            "sequence": (
                sequence
            ),
            "scales": (
                scales
            ),
            "all_scales_monotonic_nondecreasing": all(
                item[
                    "monotonic_nondecreasing"
                ]
                for item
                in scales.values()
            ),
        }

    return output


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    checkpoint_path = resolve(
        args.checkpoint
    )

    output_root = resolve(
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

    # -------------------------------------------------------------------------
    # Reference baselines
    # -------------------------------------------------------------------------
    a_clean = load_json(
        A_CLEAN
    )
    a_summary = load_json(
        A_V2
        / "robustness_summary.json"
    )

    b_clean = load_json(
        B_CLEAN
    )
    b_summary = load_json(
        B_V2
        / "robustness_summary.json"
    )

    fixed_clean = load_json(
        FIXED_CLEAN
    )
    fixed_summary = load_json(
        FIXED_V2
        / "robustness_summary.json"
    )

    c_clean = load_json(
        C_CLEAN
    )
    c_summary = load_json(
        C_V2
        / "robustness_summary.json"
    )

    for obj, model_id, label in (
        (
            a_clean,
            MODEL_A_ID,
            "Model A",
        ),
        (
            b_clean,
            MODEL_B_ID,
            "Model B",
        ),
        (
            fixed_clean,
            MODEL_FIXED_ID,
            "Model C-noGate",
        ),
        (
            c_clean,
            MODEL_C_ID,
            "Model C",
        ),
    ):
        validate_reference_clean(
            obj,
            expected_model=(
                model_id
            ),
            label=(
                label
            ),
        )

    for obj, model_id, label in (
        (
            a_summary,
            MODEL_A_ID,
            "Model A",
        ),
        (
            b_summary,
            MODEL_B_ID,
            "Model B",
        ),
        (
            fixed_summary,
            MODEL_FIXED_ID,
            "Model C-noGate",
        ),
        (
            c_summary,
            MODEL_C_ID,
            "Model C",
        ),
    ):
        validate_reference_summary(
            obj,
            expected_model=(
                model_id
            ),
            label=(
                label
            ),
        )

    device = get_device(
        args.device
    )

    amp_enabled = (
        device.type
        == "cuda"
        and not args.no_amp
    )

    print(
        "="
        * 112
    )
    print(
        "MODEL D / DARF-B2 | CLEAN + FINAL RGB DEGRADATION PROTOCOL v2"
    )
    print(
        "="
        * 112
    )
    print(
        f"checkpoint    : {checkpoint_path}"
    )
    print(
        f"device / AMP  : {device} / {amp_enabled}"
    )
    print(
        f"batch size    : {args.batch_size}"
    )
    print(
        f"protocol hash : {degradation_protocol_sha256()}"
    )
    print(
        f"clean target  : mIoU >= {CLEAN_MIOU_TARGET:.2f}"
    )
    print(
        "comparison    : D is robust-trained B2; A/B/C-noGate/C are clean-trained B0"
    )
    print(
        "="
        * 112
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model, model_meta = (
        build_model_d_darf_b2(
            PROJECT_ROOT
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    (
        state_dict,
        checkpoint_meta,
    ) = unwrap_checkpoint(
        checkpoint
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

    checkpoint_step = (
        checkpoint_meta.get(
            "global_step",
            checkpoint_meta.get(
                "step"
            ),
        )
    )

    print(
        f"[checkpoint] epoch_zero_based={checkpoint_epoch} | "
        f"global_step={checkpoint_step}"
    )

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------
    print(
        "[1] Clean validation"
    )

    clean_metrics_path = (
        clean_dir
        / "metrics.json"
    )

    d_clean = None

    if not args.force_clean:
        d_clean = compatible_existing(
            clean_metrics_path,
            condition=(
                "Clean"
            ),
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_global_step=(
                int(
                    checkpoint_step
                )
                if checkpoint_step
                is not None
                else None
            ),
            degraded=(
                False
            ),
        )

    if d_clean is None:
        clean_dataset = (
            PotsdamSlidingWindowDataset(
                PROJECT_ROOT,
                split="val",
            )
        )

        (
            d_clean,
            clean_gate_rows,
        ) = evaluate_condition(
            model=(
                model
            ),
            dataset=(
                clean_dataset
            ),
            condition=(
                "Clean"
            ),
            corruption=(
                "Clean"
            ),
            severity_level=(
                ""
            ),
            severity_rank=(
                0
            ),
            output_dir=(
                clean_dir
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
                    checkpoint_step
                )
                if checkpoint_step
                is not None
                else None
            ),
            batch_size=(
                args.batch_size
            ),
            device=(
                device
            ),
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
        )
    else:
        clean_gate_rows = list(
            d_clean[
                "gate_statistics"
            ]
        )

        print(
            f"[resume] {clean_metrics_path}"
        )

    d_clean_miou = float(
        d_clean[
            "miou"
        ]
    )

    print(
        f"[Clean] D mIoU={d_clean_miou:.6f} | "
        f"target >= {CLEAN_MIOU_TARGET:.2f}: "
        f"{d_clean_miou >= CLEAN_MIOU_TARGET}"
    )
    print(
        "[Clean gNIR] "
        f"{[round(float(row['g_nir_mean']), 6) for row in clean_gate_rows]}"
    )

    # -------------------------------------------------------------------------
    # Degraded
    # -------------------------------------------------------------------------
    print(
        "[2] Protocol-v2 degraded validation"
    )

    robustness_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_degradation_protocol(
        robustness_dir
        / "degradation_protocol.json"
    )

    d_results = {}

    all_gate_rows = [
        dict(
            row
        )
        for row
        in clean_gate_rows
    ]

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
            f"{condition} {condition_spec(corruption, level)}"
        )

        result = None

        if not args.force_robustness:
            result = compatible_existing(
                metrics_path,
                condition=(
                    condition
                ),
                checkpoint_path=(
                    checkpoint_path
                ),
                checkpoint_global_step=(
                    int(
                        checkpoint_step
                    )
                    if checkpoint_step
                    is not None
                    else None
                ),
                degraded=(
                    True
                ),
            )

        if result is None:
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

            (
                result,
                gate_rows,
            ) = evaluate_condition(
                model=(
                    model
                ),
                dataset=(
                    degraded_dataset
                ),
                condition=(
                    condition
                ),
                corruption=(
                    corruption
                ),
                severity_level=(
                    level
                ),
                severity_rank=int(
                    level[
                        1:
                    ]
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
                        checkpoint_step
                    )
                    if checkpoint_step
                    is not None
                    else None
                ),
                batch_size=(
                    args.batch_size
                ),
                device=(
                    device
                ),
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
            )
        else:
            gate_rows = list(
                result[
                    "gate_statistics"
                ]
            )

            print(
                f"[resume] {metrics_path}"
            )

        condition_miou = float(
            result[
                "miou"
            ]
        )

        drop = (
            d_clean_miou
            - condition_miou
        )

        result.update(
            {
                "clean_reference_miou": (
                    d_clean_miou
                ),
                "drop_miou": (
                    drop
                ),
                "delta_miou": (
                    -drop
                ),
                "relative_drop_pct": (
                    100.0
                    * drop
                    / d_clean_miou
                ),
                "retention_pct": (
                    100.0
                    * condition_miou
                    / d_clean_miou
                ),
            }
        )

        save_json(
            metrics_path,
            result,
        )

        d_results[
            condition
        ] = (
            result
        )

        all_gate_rows.extend(
            [
                dict(
                    row
                )
                for row
                in gate_rows
            ]
        )

        print(
            f"[{condition}] "
            f"mIoU={condition_miou:.6f} | "
            f"Drop={drop:.6f} | "
            f"gNIR="
            f"{[round(float(row['g_nir_mean']), 4) for row in gate_rows]}"
        )

    # -------------------------------------------------------------------------
    # Model D robustness summary
    # -------------------------------------------------------------------------
    robustness_rows = []

    for corruption, level in CONDITION_ORDER:
        condition = condition_name(
            corruption,
            level,
        )

        result = d_results[
            condition
        ]

        robustness_rows.append(
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
                    d_clean_miou
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

    robustness_summary = {
        "model": (
            MODEL_ID
        ),
        "model_name": (
            MODEL_NAME
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "training_regime": (
            "degradation-aware robust training"
        ),
        "split": (
            "val"
        ),
        "clean_reference": {
            "metrics_path": str(
                clean_metrics_path
            ),
            "miou": (
                d_clean_miou
            ),
            "checkpoint_global_step": (
                checkpoint_step
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
        "results": (
            robustness_rows
        ),
    }

    save_json(
        robustness_dir
        / "robustness_summary.json",
        robustness_summary,
    )

    write_csv(
        robustness_dir
        / "robustness_summary.csv",
        robustness_rows,
        [
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
        ],
    )

    # -------------------------------------------------------------------------
    # Gate interpretability
    # -------------------------------------------------------------------------
    save_json(
        robustness_dir
        / "gate_statistics.json",
        {
            "model": (
                MODEL_ID
            ),
            "gate_semantics": (
                "g_NIR residual correction strength"
            ),
            "rows": (
                all_gate_rows
            ),
        },
    )

    write_csv(
        robustness_dir
        / "gate_statistics.csv",
        all_gate_rows,
        [
            "condition",
            "corruption",
            "severity_level",
            "severity_rank",
            "scale",
            "count",
            "g_nir_mean",
            "g_nir_std",
            "g_nir_min",
            "g_nir_p05",
            "g_nir_p25",
            "g_nir_median",
            "g_nir_p75",
            "g_nir_p95",
            "g_nir_max",
        ],
    )

    monotonicity = gate_monotonicity(
        all_gate_rows
    )

    save_json(
        robustness_dir
        / "gate_monotonicity.json",
        monotonicity,
    )

    # -------------------------------------------------------------------------
    # Five-model comparison
    # -------------------------------------------------------------------------
    (
        five_rows,
        five_json,
    ) = build_five_model_comparison(
        a_clean=(
            a_clean
        ),
        a_summary=(
            a_summary
        ),
        b_clean=(
            b_clean
        ),
        b_summary=(
            b_summary
        ),
        fixed_clean=(
            fixed_clean
        ),
        fixed_summary=(
            fixed_summary
        ),
        c_clean=(
            c_clean
        ),
        c_summary=(
            c_summary
        ),
        d_clean=(
            d_clean
        ),
        d_results=(
            d_results
        ),
    )

    save_json(
        robustness_dir
        / "five_model_comparison.json",
        five_json,
    )

    write_csv(
        robustness_dir
        / "five_model_comparison.csv",
        five_rows,
        FIVE_FIELDS,
    )

    # -------------------------------------------------------------------------
    # Per-class
    # -------------------------------------------------------------------------
    per_class_rows = (
        build_per_class_comparison(
            five_rows=(
                five_rows
            ),
            d_clean=(
                d_clean
            ),
            d_results=(
                d_results
            ),
        )
    )

    write_csv(
        robustness_dir
        / "per_class_five_model_comparison.csv",
        per_class_rows,
        [
            "condition",
            "corruption",
            "severity_level",
            "class_id",
            "class_name",
            "model_a_iou",
            "model_b_iou",
            "model_c_nogate_iou",
            "model_c_iou",
            "model_d_iou",
            "d_gain_vs_a_iou",
            "d_gain_vs_b_iou",
            "d_gain_vs_fixed_iou",
            "d_gain_vs_c_iou",
        ],
    )

    # -------------------------------------------------------------------------
    # Target check
    # -------------------------------------------------------------------------
    d_wins = sum(
        (
            "D"
            in winners
        )
        for winners in five_json[
            "best_miou_model_by_condition"
        ].values()
    )

    target_check = {
        "clean_miou_target": (
            CLEAN_MIOU_TARGET
        ),
        "clean_miou": (
            d_clean_miou
        ),
        "clean_target_met": (
            d_clean_miou
            >= CLEAN_MIOU_TARGET
        ),
        "best_miou_condition_count_including_clean": (
            d_wins
        ),
        "total_conditions_including_clean": (
            10
        ),
        "mean_degraded_drop": (
            five_json[
                "mean_degraded_drop"
            ]
        ),
        "training_regime_warning": (
            five_json[
                "warning"
            ]
        ),
    }

    save_json(
        robustness_dir
        / "target_check.json",
        target_check,
    )

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------
    print()
    print(
        "="
        * 164
    )
    print(
        "FINAL FIVE-MODEL PERFORMANCE | "
        "A/B/C-noGate/C = clean-trained B0 | D = robust-trained B2 DARF"
    )
    print(
        "="
        * 164
    )
    print(
        f"{'Condition':<28} "
        f"{'A':>9} "
        f"{'B':>9} "
        f"{'Fixed':>9} "
        f"{'C':>9} "
        f"{'D/DARF':>9} "
        f"{'D-A':>9} "
        f"{'D-B':>9} "
        f"{'D-Fix':>9} "
        f"{'D-C':>9} "
        f"{'D Drop':>9}"
    )
    print(
        "-"
        * 164
    )

    for row in five_rows:
        print(
            f"{row['condition']:<28} "
            f"{float(row['model_a_miou']):>9.6f} "
            f"{float(row['model_b_miou']):>9.6f} "
            f"{float(row['model_c_nogate_miou']):>9.6f} "
            f"{float(row['model_c_miou']):>9.6f} "
            f"{float(row['model_d_miou']):>9.6f} "
            f"{float(row['d_gain_vs_a']):>+9.6f} "
            f"{float(row['d_gain_vs_b']):>+9.6f} "
            f"{float(row['d_gain_vs_fixed']):>+9.6f} "
            f"{float(row['d_gain_vs_c']):>+9.6f} "
            f"{float(row['model_d_drop']):>9.6f}"
        )

    print(
        "-"
        * 164
    )
    print(
        f"CLEAN TARGET | "
        f"D={d_clean_miou:.6f} | "
        f"target >= {CLEAN_MIOU_TARGET:.2f} | "
        f"met={d_clean_miou >= CLEAN_MIOU_TARGET}"
    )
    print(
        "Mean degraded Drop | "
        f"{five_json['mean_degraded_drop']}"
    )

    print(
        "-"
        * 164
    )
    print(
        "DARF GATE mean g_NIR | Clean -> L1 -> L2 -> L3"
    )

    for family, result in monotonicity.items():
        print(
            f"{family}:"
        )

        for scale_name, scale_result in result[
            "scales"
        ].items():
            print(
                f"  {scale_name}: "
                f"{[round(float(v), 6) for v in scale_result['g_nir_mean']]} | "
                f"monotonic_nondecreasing="
                f"{scale_result['monotonic_nondecreasing']}"
            )

        print(
            "  all_scales_monotonic_nondecreasing="
            f"{result['all_scales_monotonic_nondecreasing']}"
        )

    print(
        "-"
        * 164
    )
    print(
        f"D Clean metrics   : {clean_dir / 'metrics.json'}"
    )
    print(
        f"D robustness      : {robustness_dir / 'robustness_summary.json'}"
    )
    print(
        f"5-model CSV       : {robustness_dir / 'five_model_comparison.csv'}"
    )
    print(
        f"Gate statistics   : {robustness_dir / 'gate_statistics.csv'}"
    )
    print(
        f"Gate monotonicity : {robustness_dir / 'gate_monotonicity.json'}"
    )
    print(
        f"Target check      : {robustness_dir / 'target_check.json'}"
    )
    print(
        "="
        * 164
    )


if __name__ == "__main__":
    main()
