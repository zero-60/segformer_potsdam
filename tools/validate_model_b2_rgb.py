#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the fair SegFormer-B2 RGB-only baselines (M0/M1) on the complete
Potsdam robustness suite.

Default use for the just-trained M0 checkpoint:
    python tools/validate_model_b2_rgb.py

Explicit:
    python tools/validate_model_b2_rgb.py \
        --checkpoint outputs/training/b2_rgb_clean/checkpoints/final.pt

M1 later:
    python tools/validate_model_b2_rgb.py \
        --regime robust3 \
        --checkpoint outputs/training/b2_rgb_robust3/checkpoints/final.pt

Validation suite
----------------
1) Clean
2) Gaussian Noise      L1 / L2 / L3
3) Gaussian Blur       L1 / L2 / L3
4) RGB Underexposure   L1 / L2 / L3
5) Atmospheric Fog     L1 / L2 / L3

Total = 13 conditions x 6 validation tiles.

Scientific fairness
-------------------
- Uses the same frozen 512x512 sliding-window / stride=384 validation protocol.
- Full-resolution window logits are fused by overlap MEAN LOGIT.
- One 6000x6000 prediction is produced per tile.
- Primary metric is one GLOBAL confusion matrix over all six validation tiles.
- Standard degradations are imported from evaluation/rgb_degradation_protocol.py.
- Fog is imported from tools/validate_model_d_fog.py so B2-RGB and Model D see
  pixel-identical synthetic fog fields for the same tile + level.
- RGB-only B2 never consumes NIR.

Training-awareness
------------------
For B2_RGB_CLEAN (M0):
    all 12 degraded conditions are unseen at training time.

For B2_RGB_ROBUST3 (M1):
    Noise / Blur / Underexposure are seen degradation families.
    Fog remains unseen/OOD.

Process visualization
---------------------
During validation the terminal shows:
    condition progress
    tile progress
    overall percentage
    current global partial mIoU
    tile runtime
    elapsed time
    ETA
    estimated LOCAL finish time

A machine-readable snapshot is continuously written to:
    <output_root>/validation_progress.json

Outputs
-------
outputs/evaluation/b2_rgb_clean/
├── validation_progress.json
├── all_conditions_summary.json
├── all_conditions_summary.csv
├── clean_val/
│   ├── metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   └── per_tile_metrics.jsonl
├── robustness_val_v2/
│   ├── degradation_protocol.json
│   ├── robustness_summary.json
│   ├── robustness_summary.csv
│   ├── gaussian_noise_L1/
│   ├── ...
│   └── rgb_underexposure_L3/
└── fog_ood_val/
    ├── fog_protocol.json
    ├── fog_summary.json
    ├── fog_summary.csv
    ├── fog_L1/
    ├── fog_L2/
    └── fog_L3/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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

from train_model_b2_rgb import (
    build_b2_rgb_model,
    model_id_for_regime,
    model_name_for_regime,
)

from validate_model_a_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    append_jsonl,
    confusion_from_prediction,
    get_device,
    infer_one_tile,
    metrics_from_confusion,
    unwrap_checkpoint_state_dict,
    write_confusion_csv,
    write_per_class_csv,
)

# Reuse the exact fog implementation already used for Model D.
from validate_model_d_fog import (
    FOG_IMPLEMENTATION_REVISION,
    FOG_PROTOCOL,
    FOG_PROTOCOL_VERSION,
    FogPotsdamSlidingWindowDataset,
    fog_protocol_sha256,
    fog_spec,
)


# =============================================================================
# Constants
# =============================================================================

VALIDATOR_VERSION = "1.0.0"

DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 0
DEFAULT_LOG_EVERY = 16
DEFAULT_CONFUSION_CHUNK_ROWS = 512
DEFAULT_FOG_CHUNK_ROWS = 128

STANDARD_CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("gaussian_noise", "L1"),
    ("gaussian_noise", "L2"),
    ("gaussian_noise", "L3"),
    ("gaussian_blur", "L1"),
    ("gaussian_blur", "L2"),
    ("gaussian_blur", "L3"),
    ("rgb_underexposure", "L1"),
    ("rgb_underexposure", "L2"),
    ("rgb_underexposure", "L3"),
)

FOG_LEVELS: Tuple[str, ...] = (
    "L1",
    "L2",
    "L3",
)


# =============================================================================
# Small utilities
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def local_now() -> datetime:
    return datetime.now().astimezone()


def format_local_datetime(dt: Optional[datetime]) -> str:
    if dt is None:
        return "calibrating"
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"

    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)

    if hours < 100:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def save_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def default_checkpoint(regime: str) -> Path:
    if regime == "robust3":
        return (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_rgb_robust3"
            / "checkpoints"
            / "final.pt"
        )

    # auto defaults to the currently completed M0 run.
    return (
        PROJECT_ROOT
        / "outputs"
        / "training"
        / "b2_rgb_clean"
        / "checkpoints"
        / "final.pt"
    )


def default_output_root(regime: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / (
            "b2_rgb_robust3"
            if regime == "robust3"
            else "b2_rgb_clean"
        )
    )


def clean_dir(output_root: Path) -> Path:
    return output_root / "clean_val"


def robustness_root(output_root: Path) -> Path:
    return output_root / "robustness_val_v2"


def fog_root(output_root: Path) -> Path:
    return output_root / "fog_ood_val"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate B2-RGB Clean/Robust3 with Clean + 9 standard "
            "degradations + 3 Fog conditions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--regime",
        choices=(
            "auto",
            "clean",
            "robust3",
        ),
        default="auto",
        help=(
            "auto infers regime from checkpoint metadata. If no checkpoint is "
            "given, auto uses the completed clean/M0 checkpoint."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--suite",
        choices=(
            "all",
            "clean",
            "standard",
            "fog",
        ),
        default="all",
        help=(
            "all = Clean + standard v2 robustness + Fog."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Default 2 matches the established Model-D evaluation batch size "
            "for more comparable runtime reporting."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Keep 0. Validation is tile-major and degraded datasets cache one "
            "full 6000x6000 tile."
        ),
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
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
    )

    parser.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=DEFAULT_CONFUSION_CHUNK_ROWS,
    )

    parser.add_argument(
        "--fog-chunk-rows",
        type=int,
        default=DEFAULT_FOG_CHUNK_ROWS,
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore compatible existing metrics.json files and re-run."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    if args.num_workers != 0:
        parser.error(
            "--num-workers must be 0 for the frozen full-tile degradation "
            "evaluation/cache protocol."
        )

    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    if args.fog_chunk_rows <= 0:
        parser.error("--fog-chunk-rows must be > 0")

    return args


# =============================================================================
# Checkpoint validation
# =============================================================================

def infer_regime_from_metadata(
    metadata: Mapping[str, Any],
) -> str:
    top_regime = metadata.get(
        "regime"
    )

    protocol = metadata.get(
        "protocol"
    )

    protocol_regime = (
        protocol.get(
            "regime"
        )
        if isinstance(
            protocol,
            Mapping,
        )
        else None
    )

    values = [
        str(value)
        for value in (
            top_regime,
            protocol_regime,
        )
        if value is not None
    ]

    if not values:
        raise RuntimeError(
            "Checkpoint contains no regime metadata. "
            "Pass --regime clean or --regime robust3 explicitly."
        )

    if len(set(values)) != 1:
        raise RuntimeError(
            f"Checkpoint regime metadata disagrees: {values}"
        )

    regime = values[0]

    if regime not in (
        "clean",
        "robust3",
    ):
        raise RuntimeError(
            f"Unsupported checkpoint regime: {regime!r}"
        )

    return regime


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    requested_regime: str,
) -> str:
    regime = infer_regime_from_metadata(
        metadata
    )

    if (
        requested_regime
        != "auto"
        and requested_regime
        != regime
    ):
        raise RuntimeError(
            "Requested regime does not match checkpoint: "
            f"{requested_regime!r} != {regime!r}"
        )

    expected_model_id = (
        model_id_for_regime(
            regime
        )
    )

    expected_model_name = (
        model_name_for_regime(
            regime
        )
    )

    if metadata.get(
        "model_id"
    ) != expected_model_id:
        raise RuntimeError(
            "Checkpoint model_id mismatch: "
            f"{metadata.get('model_id')!r} != {expected_model_id!r}"
        )

    if metadata.get(
        "model_name"
    ) != expected_model_name:
        raise RuntimeError(
            "Checkpoint model_name mismatch: "
            f"{metadata.get('model_name')!r} != {expected_model_name!r}"
        )

    protocol = metadata.get(
        "protocol"
    )

    if not isinstance(
        protocol,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint protocol metadata is missing."
        )

    if protocol.get(
        "backbone"
    ) != "SegFormer-B2":
        raise RuntimeError(
            "Checkpoint is not SegFormer-B2."
        )

    if list(
        protocol.get(
            "input_modalities",
            [],
        )
    ) != [
        "RGB"
    ]:
        raise RuntimeError(
            "Checkpoint is not RGB-only."
        )

    if bool(
        protocol.get(
            "nir_used",
            True,
        )
    ):
        raise RuntimeError(
            "Checkpoint metadata says NIR is used."
        )

    corruption_training = (
        protocol.get(
            "corruption_training"
        )
    )

    if not isinstance(
        corruption_training,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint has no corruption_training metadata."
        )

    if bool(
        corruption_training.get(
            "fog_in_training",
            True,
        )
    ):
        raise RuntimeError(
            "This validation design requires Fog to remain absent from training."
        )

    if regime == "clean":
        if bool(
            corruption_training.get(
                "enabled",
                True,
            )
        ):
            raise RuntimeError(
                "Clean checkpoint unexpectedly has corruption training enabled."
            )

    if regime == "robust3":
        expected = {
            "gaussian_noise",
            "gaussian_blur",
            "rgb_underexposure",
        }

        actual = {
            str(x)
            for x in corruption_training.get(
                "families",
                [],
            )
        }

        if actual != expected:
            raise RuntimeError(
                "Robust3 corruption families mismatch: "
                f"{sorted(actual)} != {sorted(expected)}"
            )

    return regime


# =============================================================================
# Condition semantics
# =============================================================================

def training_relation(
    *,
    regime: str,
    family: str,
) -> str:
    if family == "clean":
        return "clean_reference"

    if family == "fog":
        return "unseen_ood"

    if regime == "robust3":
        return "seen_degradation_family"

    return "unseen_degradation"


def suite_conditions(
    suite: str,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    if suite in (
        "all",
        "clean",
    ):
        rows.append(
            {
                "kind": "clean",
                "condition": "Clean",
                "family": "clean",
                "corruption": None,
                "level": None,
                "severity_rank": 0,
            }
        )

    if suite in (
        "all",
        "standard",
    ):
        for corruption, level in STANDARD_CONDITIONS:
            rows.append(
                {
                    "kind": "standard",
                    "condition": (
                        condition_name(
                            corruption,
                            level,
                        )
                    ),
                    "family": corruption,
                    "corruption": corruption,
                    "level": level,
                    "severity_rank": int(
                        level[1:]
                    ),
                }
            )

    if suite in (
        "all",
        "fog",
    ):
        for level in FOG_LEVELS:
            rows.append(
                {
                    "kind": "fog",
                    "condition": (
                        f"fog_{level}"
                    ),
                    "family": "fog",
                    "corruption": "fog",
                    "level": level,
                    "severity_rank": int(
                        level[1:]
                    ),
                }
            )

    return rows


def condition_output_dir(
    *,
    output_root: Path,
    item: Mapping[str, Any],
) -> Path:
    kind = str(
        item[
            "kind"
        ]
    )

    if kind == "clean":
        return clean_dir(
            output_root
        )

    if kind == "standard":
        return (
            robustness_root(
                output_root
            )
            / str(
                item[
                    "condition"
                ]
            )
        )

    if kind == "fog":
        return (
            fog_root(
                output_root
            )
            / str(
                item[
                    "condition"
                ]
            )
        )

    raise ValueError(
        kind
    )


# =============================================================================
# Overall validation progress / ETA
# =============================================================================

class ValidationProgress:
    def __init__(
        self,
        *,
        total_conditions: int,
        tiles_per_condition: int,
        output_path: Path,
    ):
        self.total_conditions = int(
            total_conditions
        )

        self.tiles_per_condition = int(
            tiles_per_condition
        )

        self.total_tiles = (
            self.total_conditions
            * self.tiles_per_condition
        )

        self.completed_tiles = 0

        self.started_mono = (
            time.monotonic()
        )

        self.started_wall = (
            local_now()
        )

        self.tile_ema: Optional[
            float
        ] = None

        self.ema_alpha = 0.25

        self.output_path = (
            output_path
        )

        self.is_tty = (
            sys.stdout.isatty()
        )

    @staticmethod
    def bar(
        fraction: float,
        width: int = 24,
    ) -> str:
        fraction = min(
            max(
                float(
                    fraction
                ),
                0.0,
            ),
            1.0,
        )

        filled = int(
            round(
                width
                * fraction
            )
        )

        return (
            "["
            + "=" * filled
            + "." * (
                width
                - filled
            )
            + "]"
        )

    def tile_done(
        self,
        *,
        condition_index: int,
        condition: str,
        tile_index: int,
        tile_id: str,
        tile_seconds: float,
        partial_miou: float,
    ) -> None:
        self.completed_tiles += 1

        if self.tile_ema is None:
            self.tile_ema = float(
                tile_seconds
            )
        else:
            alpha = self.ema_alpha

            self.tile_ema = (
                alpha
                * float(
                    tile_seconds
                )
                + (
                    1.0
                    - alpha
                )
                * self.tile_ema
            )

        remaining_tiles = max(
            0,
            self.total_tiles
            - self.completed_tiles,
        )

        eta_seconds = (
            self.tile_ema
            * remaining_tiles
        )

        finish = (
            local_now()
            + timedelta(
                seconds=eta_seconds
            )
        )

        elapsed = (
            time.monotonic()
            - self.started_mono
        )

        fraction = (
            self.completed_tiles
            / self.total_tiles
        )

        payload = {
            "status": "running",
            "updated_at_local": (
                format_local_datetime(
                    local_now()
                )
            ),
            "started_at_local": (
                format_local_datetime(
                    self.started_wall
                )
            ),
            "condition_index": (
                condition_index
            ),
            "total_conditions": (
                self.total_conditions
            ),
            "condition": (
                condition
            ),
            "tile_index": (
                tile_index
            ),
            "tiles_per_condition": (
                self.tiles_per_condition
            ),
            "tile_id": (
                tile_id
            ),
            "completed_tiles": (
                self.completed_tiles
            ),
            "total_tiles": (
                self.total_tiles
            ),
            "overall_progress_pct": (
                100.0
                * fraction
            ),
            "partial_global_miou": (
                float(
                    partial_miou
                )
            ),
            "last_tile_seconds": (
                float(
                    tile_seconds
                )
            ),
            "tile_ema_seconds": (
                float(
                    self.tile_ema
                )
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "eta_seconds": (
                eta_seconds
            ),
            "estimated_finish_local": (
                format_local_datetime(
                    finish
                )
            ),
        }

        save_json_atomic(
            self.output_path,
            payload,
        )

        line = (
            f"OVERALL "
            f"{self.bar(fraction)} "
            f"{100.0 * fraction:6.2f}% | "
            f"C {condition_index:02d}/{self.total_conditions:02d} "
            f"{condition:<24} | "
            f"T {tile_index:02d}/{self.tiles_per_condition:02d} "
            f"{tile_id:<5} | "
            f"partial mIoU {partial_miou:.6f} | "
            f"tile {tile_seconds:.1f}s | "
            f"elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta_seconds)} | "
            f"finish {format_local_datetime(finish)}"
        )

        print(
            line,
            flush=True,
        )

    def mark_skipped_condition(
        self,
        *,
        condition_index: int,
        condition: str,
    ) -> None:
        """
        Existing compatible result counts as completed work for progress.
        """
        for tile_index in range(
            1,
            self.tiles_per_condition
            + 1,
        ):
            self.completed_tiles += 1

        fraction = (
            self.completed_tiles
            / self.total_tiles
        )

        elapsed = (
            time.monotonic()
            - self.started_mono
        )

        payload = {
            "status": "running",
            "updated_at_local": (
                format_local_datetime(
                    local_now()
                )
            ),
            "condition_index": (
                condition_index
            ),
            "total_conditions": (
                self.total_conditions
            ),
            "condition": (
                condition
            ),
            "condition_status": (
                "reused_existing_result"
            ),
            "completed_tiles": (
                self.completed_tiles
            ),
            "total_tiles": (
                self.total_tiles
            ),
            "overall_progress_pct": (
                100.0
                * fraction
            ),
            "elapsed_seconds": (
                elapsed
            ),
        }

        save_json_atomic(
            self.output_path,
            payload,
        )

        print(
            f"[resume] {condition} compatible metrics reused | "
            f"overall={100.0 * fraction:.2f}%",
            flush=True,
        )

    def finish(
        self,
        *,
        output_root: Path,
    ) -> None:
        elapsed = (
            time.monotonic()
            - self.started_mono
        )

        payload = {
            "status": "finished",
            "finished_at_local": (
                format_local_datetime(
                    local_now()
                )
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "elapsed": (
                format_duration(
                    elapsed
                )
            ),
            "output_root": str(
                output_root
            ),
            "completed_tiles": (
                self.total_tiles
            ),
            "total_tiles": (
                self.total_tiles
            ),
            "overall_progress_pct": 100.0,
        }

        save_json_atomic(
            self.output_path,
            payload,
        )


# =============================================================================
# Dataset / degradation checks
# =============================================================================

def build_dataset(
    *,
    item: Mapping[str, Any],
    fog_chunk_rows: int,
) -> PotsdamSlidingWindowDataset:
    kind = str(
        item[
            "kind"
        ]
    )

    if kind == "clean":
        return PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )

    if kind == "standard":
        return DegradedPotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
            corruption=str(
                item[
                    "corruption"
                ]
            ),
            level=str(
                item[
                    "level"
                ]
            ),
        )

    if kind == "fog":
        return FogPotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
            level=str(
                item[
                    "level"
                ]
            ),
            fog_chunk_rows=fog_chunk_rows,
        )

    raise ValueError(
        kind
    )


def degradation_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: PotsdamSlidingWindowDataset,
    condition: str,
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
        if (
            clean[
                key
            ]
            != degraded[
                key
            ]
        ):
            raise RuntimeError(
                f"{condition}: frozen metadata changed at {key}."
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
            f"{condition}: degraded RGB equals Clean RGB on probe window."
        )

    print(
        f"[probe] {condition}: PASS | "
        "same frozen window | RGB changed",
        flush=True,
    )


def clear_dataset_cache(
    dataset: PotsdamSlidingWindowDataset,
) -> None:
    method = getattr(
        dataset,
        "clear_degradation_cache",
        None,
    )

    if callable(
        method
    ):
        method()


# =============================================================================
# Existing result compatibility
# =============================================================================

def compatible_existing(
    path: Path,
    *,
    model_id: str,
    condition: str,
    checkpoint_path: Path,
    checkpoint_global_step: Optional[int],
    item: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    checks = [
        obj.get(
            "model"
        )
        == model_id,
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
    ]

    if (
        checkpoint_global_step
        is not None
    ):
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

    kind = str(
        item[
            "kind"
        ]
    )

    if kind == "standard":
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

    if kind == "fog":
        checks.extend(
            [
                obj.get(
                    "fog_protocol_sha256"
                )
                == fog_protocol_sha256(),
                int(
                    obj.get(
                        "fog_implementation_revision",
                        -1,
                    )
                )
                == FOG_IMPLEMENTATION_REVISION,
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
# Evaluate one condition
# =============================================================================

def evaluate_condition(
    *,
    model: torch.nn.Module,
    dataset: PotsdamSlidingWindowDataset,
    item: Mapping[str, Any],
    output_dir: Path,
    model_id: str,
    model_name: str,
    regime: str,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_global_step: Optional[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    confusion_chunk_rows: int,
    save_predictions: bool,
    condition_index: int,
    progress: ValidationProgress,
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
        len(
            tile_ids
        )
        != 6
        or windows_per_tile
        != 256
    ):
        raise RuntimeError(
            "Frozen validation protocol changed: "
            f"tiles={len(tile_ids)}, "
            f"windows_per_tile={windows_per_tile}"
        )

    condition = str(
        item[
            "condition"
        ]
    )

    family = str(
        item[
            "family"
        ]
    )

    relation = training_relation(
        regime=regime,
        family=family,
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

    started = time.time()

    for tile_index, tile_id in enumerate(
        tile_ids
    ):
        print(
            f"[{condition}] "
            f"tile {tile_index + 1}/"
            f"{len(tile_ids)} | "
            f"{tile_id}",
            flush=True,
        )

        prediction, runtime = (
            infer_one_tile(
                model=model,
                dataset=dataset,
                tile_id=tile_id,
                tile_index=tile_index,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                device=device,
                amp_enabled=amp_enabled,
                log_every=log_every,
            )
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
            "family": (
                family
            ),
            "corruption": (
                item.get(
                    "corruption"
                )
            ),
            "severity_level": (
                item.get(
                    "level"
                )
            ),
            "severity_rank": (
                item.get(
                    "severity_rank"
                )
            ),
            "training_relation": (
                relation
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
                for row in tile_metrics[
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

        partial_metrics = (
            metrics_from_confusion(
                global_confusion,
                CLASS_NAMES,
            )
        )

        progress.tile_done(
            condition_index=(
                condition_index
            ),
            condition=(
                condition
            ),
            tile_index=(
                tile_index
                + 1
            ),
            tile_id=(
                tile_id
            ),
            tile_seconds=float(
                runtime[
                    "inference_seconds"
                ]
            ),
            partial_miou=float(
                partial_metrics[
                    "miou"
                ]
            ),
        )

        print(
            f"  fused tile "
            f"mIoU={tile_metrics['miou']:.6f} | "
            f"pixel_acc="
            f"{tile_metrics['pixel_accuracy']:.6f} | "
            f"coverage="
            f"{runtime['coverage_min']:.0f}"
            f".."
            f"{runtime['coverage_max']:.0f} | "
            f"time="
            f"{runtime['inference_seconds']:.1f}s",
            flush=True,
        )

        clear_dataset_cache(
            dataset
        )

        del (
            prediction,
            target,
            target_t,
        )

    global_metrics = (
        metrics_from_confusion(
            global_confusion,
            CLASS_NAMES,
        )
    )

    result: Dict[
        str,
        Any,
    ] = {
        "validator_version": (
            VALIDATOR_VERSION
        ),
        "model": (
            model_id
        ),
        "model_name": (
            model_name
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "regime": (
            regime
        ),
        "condition": (
            condition
        ),
        "family": (
            family
        ),
        "corruption": (
            item.get(
                "corruption"
            )
        ),
        "severity_level": (
            item.get(
                "level"
            )
        ),
        "severity_rank": (
            item.get(
                "severity_rank"
            )
        ),
        "training_relation": (
            relation
        ),
        "split": "val",
        "metric_scope": (
            "GLOBAL confusion matrix after "
            "full-tile mean-logit fusion"
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
            "RGB"
        ],
        "nir_used": False,
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
            windows_per_tile
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
    }

    kind = str(
        item[
            "kind"
        ]
    )

    if kind == "standard":
        result.update(
            {
                "severity_parameters": (
                    condition_spec(
                        str(
                            item[
                                "corruption"
                            ]
                        ),
                        str(
                            item[
                                "level"
                            ]
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
            }
        )

    if kind == "fog":
        result.update(
            {
                "severity_parameters": (
                    fog_spec(
                        str(
                            item[
                                "level"
                            ]
                        )
                    )
                ),
                "fog_protocol_version": (
                    FOG_PROTOCOL_VERSION
                ),
                "fog_implementation_revision": (
                    FOG_IMPLEMENTATION_REVISION
                ),
                "fog_protocol_sha256": (
                    fog_protocol_sha256()
                ),
                "fog_seen_during_training": False,
                "fog_equation": (
                    "I(x) = J(x) * t(x) + A * (1 - t(x))"
                ),
                "rgb_degraded": True,
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

    return result


# =============================================================================
# Summaries
# =============================================================================

def add_clean_relative_metrics(
    result: Dict[str, Any],
    *,
    clean_miou: float,
) -> None:
    current = float(
        result[
            "miou"
        ]
    )

    drop = (
        clean_miou
        - current
    )

    result[
        "clean_reference_miou"
    ] = clean_miou

    result[
        "drop_miou"
    ] = drop

    result[
        "delta_miou"
    ] = -drop

    result[
        "relative_drop_pct"
    ] = (
        100.0
        * drop
        / clean_miou
    )

    result[
        "retention_pct"
    ] = (
        100.0
        * current
        / clean_miou
    )


def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": result.get(
            "model"
        ),
        "regime": result.get(
            "regime"
        ),
        "condition": result.get(
            "condition"
        ),
        "family": result.get(
            "family"
        ),
        "severity_level": result.get(
            "severity_level"
        ),
        "severity_rank": result.get(
            "severity_rank"
        ),
        "training_relation": result.get(
            "training_relation"
        ),
        "clean_miou": result.get(
            "clean_reference_miou",
            result.get(
                "miou"
            )
            if result.get(
                "condition"
            )
            == "Clean"
            else None,
        ),
        "miou": result.get(
            "miou"
        ),
        "drop_miou": result.get(
            "drop_miou",
            0.0
            if result.get(
                "condition"
            )
            == "Clean"
            else None,
        ),
        "relative_drop_pct": result.get(
            "relative_drop_pct",
            0.0
            if result.get(
                "condition"
            )
            == "Clean"
            else None,
        ),
        "retention_pct": result.get(
            "retention_pct",
            100.0
            if result.get(
                "condition"
            )
            == "Clean"
            else None,
        ),
        "pixel_accuracy": result.get(
            "pixel_accuracy"
        ),
        "mean_class_accuracy": result.get(
            "mean_class_accuracy"
        ),
        "validation_seconds": result.get(
            "validation_seconds"
        ),
    }


def write_summary_bundle(
    *,
    output_root: Path,
    results: Sequence[Mapping[str, Any]],
    model_id: str,
    model_name: str,
    regime: str,
    checkpoint_path: Path,
    checkpoint_step: Optional[int],
) -> None:
    rows = [
        summary_row(
            result
        )
        for result in results
    ]

    save_json(
        output_root
        / "all_conditions_summary.json",
        {
            "model": (
                model_id
            ),
            "model_name": (
                model_name
            ),
            "regime": (
                regime
            ),
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_global_step": (
                checkpoint_step
            ),
            "results": (
                rows
            ),
        },
    )

    fields = [
        "model",
        "regime",
        "condition",
        "family",
        "severity_level",
        "severity_rank",
        "training_relation",
        "clean_miou",
        "miou",
        "drop_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        "validation_seconds",
    ]

    write_csv(
        output_root
        / "all_conditions_summary.csv",
        rows,
        fields,
    )

    standard_rows = [
        row
        for row in rows
        if row[
            "family"
        ]
        in {
            "gaussian_noise",
            "gaussian_blur",
            "rgb_underexposure",
        }
    ]

    if standard_rows:
        save_json(
            robustness_root(
                output_root
            )
            / "robustness_summary.json",
            {
                "model": (
                    model_id
                ),
                "model_name": (
                    model_name
                ),
                "regime": (
                    regime
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
                "results": (
                    standard_rows
                ),
            },
        )

        write_csv(
            robustness_root(
                output_root
            )
            / "robustness_summary.csv",
            standard_rows,
            fields,
        )

    fog_rows = [
        row
        for row in rows
        if row[
            "family"
        ]
        == "fog"
    ]

    if fog_rows:
        save_json(
            fog_root(
                output_root
            )
            / "fog_summary.json",
            {
                "model": (
                    model_id
                ),
                "model_name": (
                    model_name
                ),
                "regime": (
                    regime
                ),
                "scientific_role": (
                    "Fog remains unseen/OOD for B2-RGB training."
                ),
                "fog_protocol_version": (
                    FOG_PROTOCOL_VERSION
                ),
                "fog_implementation_revision": (
                    FOG_IMPLEMENTATION_REVISION
                ),
                "fog_protocol_sha256": (
                    fog_protocol_sha256()
                ),
                "results": (
                    fog_rows
                ),
            },
        )

        write_csv(
            fog_root(
                output_root
            )
            / "fog_summary.csv",
            fog_rows,
            fields,
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    requested_regime = (
        args.regime
    )

    # With --regime auto and no explicit checkpoint, use completed M0.
    checkpoint_path = (
        resolve(
            args.checkpoint
        )
        if args.checkpoint
        is not None
        else default_checkpoint(
            "robust3"
            if requested_regime
            == "robust3"
            else "clean"
        ).resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    (
        state_dict,
        checkpoint_meta,
    ) = unwrap_checkpoint_state_dict(
        checkpoint_obj
    )

    regime = (
        validate_checkpoint_metadata(
            checkpoint_meta,
            requested_regime=(
                requested_regime
            ),
        )
    )

    model_id = (
        model_id_for_regime(
            regime
        )
    )

    model_name = (
        model_name_for_regime(
            regime
        )
    )

    output_root = (
        resolve(
            args.output_root
        )
        if args.output_root
        is not None
        else default_output_root(
            regime
        ).resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device(
        args.device
    )

    amp_enabled = (
        device.type
        == "cuda"
        and not args.no_amp
    )

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

    checkpoint_epoch_int = (
        int(
            checkpoint_epoch
        )
        if checkpoint_epoch
        is not None
        else None
    )

    checkpoint_step_int = (
        int(
            checkpoint_step
        )
        if checkpoint_step
        is not None
        else None
    )

    training_protocol = (
        checkpoint_meta[
            "protocol"
        ]
    )

    planned_steps = (
        training_protocol.get(
            "total_update_steps"
        )
    )

    print("=" * 122)
    print(
        f"{model_name} | COMPLETE VALIDATION SUITE"
    )
    print("=" * 122)
    print(
        f"checkpoint      : {checkpoint_path}"
    )
    print(
        f"regime          : {regime}"
    )
    print(
        f"epoch           : "
        f"{checkpoint_epoch_int + 1 if checkpoint_epoch_int is not None else 'unknown'}"
    )
    print(
        f"global_step     : {checkpoint_step_int}"
    )
    print(
        f"planned updates : {planned_steps}"
    )
    print(
        f"device / AMP    : {device} / {amp_enabled}"
    )
    print(
        f"batch size      : {args.batch_size}"
    )
    print(
        f"output root     : {output_root}"
    )
    print(
        f"start local     : {format_local_datetime(local_now())}"
    )
    print("=" * 122)

    if (
        planned_steps
        is not None
        and checkpoint_step_int
        is not None
        and checkpoint_step_int
        != int(
            planned_steps
        )
    ):
        difference = (
            int(
                planned_steps
            )
            - checkpoint_step_int
        )

        print(
            "[checkpoint note] "
            f"global_step differs from planned updates by {difference}. "
            "This validator does not reject the checkpoint because CUDA AMP "
            "overflow/skip events can legitimately reduce successful optimizer "
            "steps while all requested epochs still complete.",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print(
        "[1] building exact B2-RGB architecture from train_model_b2_rgb.py"
    )

    model, model_meta = (
        build_b2_rgb_model()
    )

    incompatible = (
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading unexpectedly returned incompatibilities."
        )

    model.to(
        device
    )

    model.eval()

    print(
        f"  strict load PASS | "
        f"parameters="
        f"{model_meta['parameters']['total']:,} | "
        "input=RGB only"
    )

    # -------------------------------------------------------------------------
    # Protocol snapshots
    # -------------------------------------------------------------------------

    standard_root = (
        robustness_root(
            output_root
        )
    )

    standard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_degradation_protocol(
        standard_root
        / "degradation_protocol.json"
    )

    fog_output_root = (
        fog_root(
            output_root
        )
    )

    fog_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fog_protocol_payload = dict(
        FOG_PROTOCOL
    )

    fog_protocol_payload[
        "sha256"
    ] = fog_protocol_sha256()

    save_json(
        fog_output_root
        / "fog_protocol.json",
        fog_protocol_payload,
    )

    # -------------------------------------------------------------------------
    # Conditions / reference dataset
    # -------------------------------------------------------------------------

    conditions = suite_conditions(
        args.suite
    )

    if not conditions:
        raise RuntimeError(
            "No validation conditions selected."
        )

    clean_probe_dataset = (
        PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )
    )

    if (
        len(
            clean_probe_dataset.tile_ids
        )
        != 6
        or len(
            clean_probe_dataset.window_coordinates
        )
        != 256
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    progress = (
        ValidationProgress(
            total_conditions=(
                len(
                    conditions
                )
            ),
            tiles_per_condition=6,
            output_path=(
                output_root
                / "validation_progress.json"
            ),
        )
    )

    results: List[
        Dict[str, Any]
    ] = []

    # Existing Clean result may be needed if user runs only standard/fog.
    clean_result: Optional[
        Dict[str, Any]
    ] = None

    existing_clean_path = (
        clean_dir(
            output_root
        )
        / "metrics.json"
    )

    if existing_clean_path.is_file():
        try:
            candidate = json.loads(
                existing_clean_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                candidate.get(
                    "model"
                )
                == model_id
                and candidate.get(
                    "condition"
                )
                == "Clean"
                and (
                    checkpoint_step_int
                    is None
                    or int(
                        candidate.get(
                            "checkpoint_global_step",
                            -1,
                        )
                    )
                    == checkpoint_step_int
                )
            ):
                clean_result = (
                    candidate
                )
        except Exception:
            clean_result = None

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    print(
        f"[2] validating {len(conditions)} conditions "
        f"({len(conditions) * 6} full tiles)"
    )

    for condition_index, item in enumerate(
        conditions,
        start=1,
    ):
        condition = str(
            item[
                "condition"
            ]
        )

        relation = (
            training_relation(
                regime=regime,
                family=str(
                    item[
                        "family"
                    ]
                ),
            )
        )

        output_dir = (
            condition_output_dir(
                output_root=output_root,
                item=item,
            )
        )

        metrics_path = (
            output_dir
            / "metrics.json"
        )

        print()
        print(
            "-" * 122
        )
        print(
            f"[Condition {condition_index:02d}/"
            f"{len(conditions):02d}] "
            f"{condition} | "
            f"relation={relation}"
        )
        print(
            "-" * 122
        )

        existing = None

        if not args.force:
            existing = (
                compatible_existing(
                    metrics_path,
                    model_id=model_id,
                    condition=condition,
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    checkpoint_global_step=(
                        checkpoint_step_int
                    ),
                    item=item,
                )
            )

        if existing is not None:
            result = existing

            progress.mark_skipped_condition(
                condition_index=(
                    condition_index
                ),
                condition=(
                    condition
                ),
            )

        else:
            dataset = (
                build_dataset(
                    item=item,
                    fog_chunk_rows=(
                        args.fog_chunk_rows
                    ),
                )
            )

            if item[
                "kind"
            ] != "clean":
                degradation_probe(
                    clean_dataset=(
                        clean_probe_dataset
                    ),
                    degraded_dataset=(
                        dataset
                    ),
                    condition=(
                        condition
                    ),
                )

            result = (
                evaluate_condition(
                    model=model,
                    dataset=dataset,
                    item=item,
                    output_dir=(
                        output_dir
                    ),
                    model_id=(
                        model_id
                    ),
                    model_name=(
                        model_name
                    ),
                    regime=(
                        regime
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    checkpoint_epoch=(
                        checkpoint_epoch_int
                    ),
                    checkpoint_global_step=(
                        checkpoint_step_int
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                    num_workers=(
                        args.num_workers
                    ),
                    pin_memory=(
                        args.pin_memory
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
                    condition_index=(
                        condition_index
                    ),
                    progress=(
                        progress
                    ),
                )
            )

            clear_dataset_cache(
                dataset
            )

            del dataset

        if condition == "Clean":
            clean_result = (
                result
            )

        results.append(
            result
        )

        print(
            f"[condition result] "
            f"{condition:<24} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"pixel_acc="
            f"{float(result['pixel_accuracy']):.6f} | "
            f"time="
            f"{float(result['validation_seconds']):.1f}s"
        )

    # -------------------------------------------------------------------------
    # Clean-relative metrics
    # -------------------------------------------------------------------------

    if clean_result is None:
        raise RuntimeError(
            "Clean result is required to compute degradation drop/retention. "
            "Run --suite all or --suite clean first."
        )

    clean_miou = float(
        clean_result[
            "miou"
        ]
    )

    for result in results:
        if result[
            "condition"
        ] == "Clean":
            result[
                "clean_reference_miou"
            ] = (
                clean_miou
            )
            result[
                "drop_miou"
            ] = 0.0
            result[
                "delta_miou"
            ] = 0.0
            result[
                "relative_drop_pct"
            ] = 0.0
            result[
                "retention_pct"
            ] = 100.0
        else:
            add_clean_relative_metrics(
                result,
                clean_miou=(
                    clean_miou
                ),
            )

        # Update per-condition metrics with clean-relative fields.
        result_dir = (
            condition_output_dir(
                output_root=output_root,
                item={
                    "kind": (
                        "clean"
                        if result[
                            "condition"
                        ]
                        == "Clean"
                        else (
                            "fog"
                            if result[
                                "family"
                            ]
                            == "fog"
                            else "standard"
                        )
                    ),
                    "condition": (
                        result[
                            "condition"
                        ]
                    ),
                },
            )
        )

        save_json(
            result_dir
            / "metrics.json",
            result,
        )

    write_summary_bundle(
        output_root=(
            output_root
        ),
        results=(
            results
        ),
        model_id=(
            model_id
        ),
        model_name=(
            model_name
        ),
        regime=(
            regime
        ),
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint_step=(
            checkpoint_step_int
        ),
    )

    progress.finish(
        output_root=(
            output_root
        )
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print()
    print("=" * 138)
    print(
        f"{model_name} | FINAL VALIDATION SUMMARY"
    )
    print("=" * 138)
    print(
        f"{'Condition':<26} "
        f"{'Relation':<24} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'RelDrop%':>10} "
        f"{'Retention%':>12} "
        f"{'Time(s)':>10}"
    )
    print(
        "-" * 138
    )

    for result in results:
        print(
            f"{str(result['condition']):<26} "
            f"{str(result['training_relation']):<24} "
            f"{float(result['miou']):>10.6f} "
            f"{float(result['drop_miou']):>10.6f} "
            f"{float(result['relative_drop_pct']):>10.3f} "
            f"{float(result['retention_pct']):>12.3f} "
            f"{float(result['validation_seconds']):>10.1f}"
        )

    print(
        "-" * 138
    )
    print(
        f"Clean mIoU       : {clean_miou:.6f}"
    )
    print(
        f"Summary JSON     : "
        f"{output_root / 'all_conditions_summary.json'}"
    )
    print(
        f"Summary CSV      : "
        f"{output_root / 'all_conditions_summary.csv'}"
    )
    print(
        f"Progress         : "
        f"{output_root / 'validation_progress.json'}"
    )
    print("=" * 138)


if __name__ == "__main__":
    main()
