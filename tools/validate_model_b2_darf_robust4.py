#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate M3: SegFormer-B2 DARF Robust-4.

Run:
    python tools/validate_model_b2_darf_robust4.py

Default checkpoint:
    outputs/training/b2_darf_robust4/checkpoints/final.pt

Evaluation suite:
    Clean
    Gaussian Noise L1/L2/L3
    Gaussian Blur L1/L2/L3
    RGB Underexposure L1/L2/L3
    Fog L1/L2/L3

Total:
    13 conditions x 6 Potsdam validation tiles

Metric protocol:
    512x512 sliding windows
    -> full-resolution logits
    -> overlap MEAN-LOGIT fusion
    -> one 6000x6000 prediction per tile
    -> one GLOBAL confusion matrix over all six validation tiles
    -> global mIoU + per-class IoU

Robust-4 semantics:
    Noise / Blur / Underexposure / Fog are all SEEN degradation families.
    Only RGB is degraded. NIR and GT stay unchanged.

M3-specific diagnostics:
    - per-window Gate strength/logit CSV
    - 4-scale Gate statistics for every condition
    - Clean -> L1 -> L2 -> L3 Gate monotonicity for all 4 families
    - M3 vs M2 automatic mIoU comparison
    - optional per-class M3 vs M2 comparison when M2 metrics exist

Expected project files:
    tools/train_model_b2_darf_robust4.py
    tools/validate_model_d_darf_b2_v2.py
    tools/validate_model_d_fog.py
    models/segformer_b2_darf.py

Outputs:
outputs/evaluation/b2_darf_robust4/
├── validation_progress.json
├── all_conditions_summary.json
├── all_conditions_summary.csv
├── comparison_vs_m2.json
├── comparison_vs_m2.csv
├── per_class_comparison_vs_m2.csv
├── gate_statistics.json
├── gate_statistics.csv
├── gate_monotonicity.json
├── clean_val/
│   ├── metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   ├── per_tile_metrics.jsonl
│   └── gate_strength_windows.csv
├── robustness_val_v2/
│   ├── degradation_protocol.json
│   ├── gaussian_noise_L1/
│   ├── ...
│   └── rgb_underexposure_L3/
└── fog_seen_val/
    ├── fog_protocol.json
    ├── fog_L1/
    ├── fog_L2/
    └── fog_L3/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_d_darf_b2_v2 as base

from models.segformer_b2_darf import (
    GATE_TYPE,
    NUM_SCALES,
    build_model_d_darf_b2,
)
from train_model_b2_darf_robust4 import (
    MODEL_ID,
    MODEL_NAME,
    PROTOCOL_VERSION,
)
from validate_model_d_fog import (
    FOG_IMPLEMENTATION_REVISION,
    FOG_PROTOCOL,
    FOG_PROTOCOL_VERSION,
    FogPotsdamSlidingWindowDataset,
    fog_protocol_sha256,
    fog_spec,
)


VALIDATOR_VERSION = "1.0.0"
REGIME = "robust4"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_darf_robust4"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_darf_robust4"
)

DEFAULT_M2_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgbnir_fixed_robust4"
    / "all_conditions_summary.json"
)

DEFAULT_M2_EVAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgbnir_fixed_robust4"
)

STANDARD_CONDITIONS = (
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

FOG_LEVELS = (
    "L1",
    "L2",
    "L3",
)

GATE_FIELDS = (
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
)


# =============================================================================
# Generic helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

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
    obj: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
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
            fieldnames=list(fields),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field
                    )
                    for field in fields
                }
            )


def local_now() -> datetime:
    return datetime.now().astimezone()


def format_dt(
    value: Optional[datetime],
) -> str:
    if value is None:
        return "calibrating"

    return value.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def format_duration(
    seconds: Optional[float],
) -> str:
    if (
        seconds is None
        or not math.isfinite(
            seconds
        )
        or seconds < 0
    ):
        return "--:--:--"

    total = int(
        round(
            seconds
        )
    )

    hours, rem = divmod(
        total,
        3600,
    )

    minutes, secs = divmod(
        rem,
        60,
    )

    if hours < 100:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    days, hours = divmod(
        hours,
        24,
    )

    return (
        f"{days}d "
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def condition_output_dir(
    *,
    output_root: Path,
    kind: str,
    condition: str,
) -> Path:
    if kind == "clean":
        return (
            output_root
            / "clean_val"
        )

    if kind == "standard":
        return (
            output_root
            / "robustness_val_v2"
            / condition
        )

    if kind == "fog":
        return (
            output_root
            / "fog_seen_val"
            / condition
        )

    raise ValueError(
        kind
    )


def training_relation(
    family: str,
) -> str:
    if family == "clean":
        return "clean_reference"

    if family in {
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
        "fog",
    }:
        return "seen_degradation_family"

    return "unknown_relation"


# =============================================================================
# Validation progress
# =============================================================================

class ValidationProgress:
    def __init__(
        self,
        *,
        total_conditions: int,
        output_path: Path,
    ):
        self.total_conditions = int(
            total_conditions
        )

        self.completed = 0

        self.started_wall = (
            local_now()
        )

        self.started_mono = (
            time.monotonic()
        )

        self.condition_ema: Optional[
            float
        ] = None

        self.alpha = 0.30

        self.output_path = (
            output_path
        )

    def condition_done(
        self,
        *,
        condition: str,
        condition_seconds: float,
        miou: float,
        reused: bool,
    ) -> None:
        self.completed += 1

        if not reused:
            if self.condition_ema is None:
                self.condition_ema = float(
                    condition_seconds
                )
            else:
                self.condition_ema = (
                    self.alpha
                    * float(
                        condition_seconds
                    )
                    + (
                        1.0
                        - self.alpha
                    )
                    * self.condition_ema
                )

        remaining = max(
            0,
            self.total_conditions
            - self.completed,
        )

        eta_seconds = (
            None
            if self.condition_ema is None
            else (
                self.condition_ema
                * remaining
            )
        )

        finish = (
            None
            if eta_seconds is None
            else (
                local_now()
                + timedelta(
                    seconds=eta_seconds
                )
            )
        )

        elapsed = (
            time.monotonic()
            - self.started_mono
        )

        payload = {
            "status": (
                "running"
                if self.completed
                < self.total_conditions
                else "finished"
            ),
            "updated_at_local": (
                format_dt(
                    local_now()
                )
            ),
            "started_at_local": (
                format_dt(
                    self.started_wall
                )
            ),
            "condition": (
                condition
            ),
            "completed_conditions": (
                self.completed
            ),
            "total_conditions": (
                self.total_conditions
            ),
            "progress_pct": (
                100.0
                * self.completed
                / self.total_conditions
            ),
            "condition_seconds": (
                float(
                    condition_seconds
                )
            ),
            "condition_reused": (
                bool(
                    reused
                )
            ),
            "condition_miou": (
                float(
                    miou
                )
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "eta_seconds": (
                eta_seconds
            ),
            "estimated_finish_local": (
                format_dt(
                    finish
                )
                if finish is not None
                else None
            ),
        }

        save_json(
            self.output_path,
            payload,
        )

        print(
            f"[overall] "
            f"{self.completed:02d}/"
            f"{self.total_conditions:02d} | "
            f"{100.0 * self.completed / self.total_conditions:6.2f}% | "
            f"{condition:<24} | "
            f"mIoU={miou:.6f} | "
            f"elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta_seconds)} | "
            f"finish={format_dt(finish)}",
            flush=True,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate M3 / B2-DARF-Robust4 on Clean + 12 Robust-4 conditions."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    p.add_argument(
        "--m2-summary",
        type=Path,
        default=DEFAULT_M2_SUMMARY,
    )

    p.add_argument(
        "--m2-eval-root",
        type=Path,
        default=DEFAULT_M2_EVAL_ROOT,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--no-amp",
        action="store_true",
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=8,
    )

    p.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
    )

    p.add_argument(
        "--fog-chunk-rows",
        type=int,
        default=128,
    )

    p.add_argument(
        "--save-predictions",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore compatible existing metrics and recompute all conditions."
        ),
    )

    x = p.parse_args()

    if x.batch_size <= 0:
        p.error(
            "--batch-size must be > 0"
        )

    if x.log_every <= 0:
        p.error(
            "--log-every must be > 0"
        )

    if x.confusion_chunk_rows <= 0:
        p.error(
            "--confusion-chunk-rows must be > 0"
        )

    if x.fog_chunk_rows <= 0:
        p.error(
            "--fog-chunk-rows must be > 0"
        )

    return x


# =============================================================================
# Checkpoint audit
# =============================================================================

def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    if metadata.get(
        "model_id"
    ) != MODEL_ID:
        raise RuntimeError(
            "Checkpoint model_id mismatch: "
            f"{metadata.get('model_id')!r} != {MODEL_ID!r}"
        )

    if metadata.get(
        "model_name"
    ) != MODEL_NAME:
        raise RuntimeError(
            "Checkpoint model_name mismatch: "
            f"{metadata.get('model_name')!r} != {MODEL_NAME!r}"
        )

    if metadata.get(
        "regime"
    ) != REGIME:
        raise RuntimeError(
            "Checkpoint regime mismatch."
        )

    protocol = metadata.get(
        "protocol"
    )

    if not isinstance(
        protocol,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint protocol metadata missing."
        )

    if protocol.get(
        "protocol_version"
    ) != PROTOCOL_VERSION:
        raise RuntimeError(
            "Checkpoint training protocol mismatch: "
            f"{protocol.get('protocol_version')!r} != {PROTOCOL_VERSION!r}"
        )

    if protocol.get(
        "model_id"
    ) != MODEL_ID:
        raise RuntimeError(
            "Protocol model_id mismatch."
        )

    if protocol.get(
        "model_name"
    ) != MODEL_NAME:
        raise RuntimeError(
            "Protocol model_name mismatch."
        )

    if protocol.get(
        "backbone"
    ) != "SegFormer-B2":
        raise RuntimeError(
            "Checkpoint backbone is not SegFormer-B2."
        )

    if list(
        protocol.get(
            "input_modalities",
            [],
        )
    ) != [
        "RGB",
        "NIR",
    ]:
        raise RuntimeError(
            "Checkpoint is not RGB+NIR."
        )

    if not bool(
        protocol.get(
            "nir_used",
            False,
        )
    ):
        raise RuntimeError(
            "Checkpoint protocol says NIR is disabled."
        )

    if not bool(
        protocol.get(
            "quality_gate",
            False,
        )
    ):
        raise RuntimeError(
            "Checkpoint protocol does not enable DARF quality gate."
        )

    if not bool(
        protocol.get(
            "gate_supervision",
            False,
        )
    ):
        raise RuntimeError(
            "Checkpoint protocol does not enable gate supervision."
        )

    if protocol.get(
        "gate_type"
    ) != GATE_TYPE:
        raise RuntimeError(
            "Checkpoint gate type mismatch."
        )

    gate_target = protocol.get(
        "gate_target"
    )

    if not isinstance(
        gate_target,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint gate_target metadata missing."
        )

    if abs(
        float(
            gate_target.get(
                "clean",
                float(
                    "nan"
                ),
            )
        )
        - 0.05
    ) > 1e-12:
        raise RuntimeError(
            "Unexpected Clean gate target."
        )

    if (
        gate_target.get(
            "degraded_formula"
        )
        != "0.15 + 0.80 * severity"
    ):
        raise RuntimeError(
            "Unexpected degraded gate target formula."
        )

    corruption = protocol.get(
        "corruption_training"
    )

    if not isinstance(
        corruption,
        Mapping,
    ):
        raise RuntimeError(
            "corruption_training metadata missing."
        )

    expected = {
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
        "fog",
    }

    actual = {
        str(
            item
        )
        for item in corruption.get(
            "families",
            [],
        )
    }

    if actual != expected:
        raise RuntimeError(
            "Robust-4 family mismatch: "
            f"{sorted(actual)} != {sorted(expected)}"
        )

    if not bool(
        corruption.get(
            "fog_in_training",
            False,
        )
    ):
        raise RuntimeError(
            "This is not the intended Fog-seen Robust-4 checkpoint."
        )

    if str(
        corruption.get(
            "nir",
            ""
        )
    ) != "clean / unchanged":
        raise RuntimeError(
            "Formal experiment requires clean/unchanged NIR during corruption training."
        )


# =============================================================================
# Base validator patching
# =============================================================================

def patch_base_identity() -> None:
    """
    validate_model_d_darf_b2_v2.py contains the correct DARF inference and gate
    collection implementation. Patch only experiment identity, not inference.
    """
    base.MODEL_ID = (
        MODEL_ID
    )

    base.MODEL_NAME = (
        MODEL_NAME
    )


@contextmanager
def fog_protocol_patch():
    """
    Temporarily make the existing DARF evaluator understand corruption='fog'.

    The actual Fog RGB generation remains the frozen implementation in
    validate_model_d_fog.py.
    """
    old_condition_spec = (
        base.condition_spec
    )

    old_protocol_version = (
        base.DEGRADATION_PROTOCOL_VERSION
    )

    old_revision = (
        base.IMPLEMENTATION_REVISION
    )

    old_hash = (
        base.degradation_protocol_sha256
    )

    def condition_spec_patched(
        corruption: str,
        level: str,
    ):
        if corruption == "fog":
            return fog_spec(
                level
            )

        return old_condition_spec(
            corruption,
            level,
        )

    base.condition_spec = (
        condition_spec_patched
    )

    base.DEGRADATION_PROTOCOL_VERSION = (
        FOG_PROTOCOL_VERSION
    )

    base.IMPLEMENTATION_REVISION = (
        FOG_IMPLEMENTATION_REVISION
    )

    base.degradation_protocol_sha256 = (
        fog_protocol_sha256
    )

    try:
        yield
    finally:
        base.condition_spec = (
            old_condition_spec
        )

        base.DEGRADATION_PROTOCOL_VERSION = (
            old_protocol_version
        )

        base.IMPLEMENTATION_REVISION = (
            old_revision
        )

        base.degradation_protocol_sha256 = (
            old_hash
        )


# =============================================================================
# Existing result compatibility
# =============================================================================

def compatible_existing(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_step: Optional[int],
    condition: str,
    kind: str,
) -> Optional[
    Dict[str, Any]
]:
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
        isinstance(
            obj.get(
                "gate_statistics"
            ),
            list,
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

    if checkpoint_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(
                checkpoint_step
            )
        )

    if kind == "standard":
        checks.extend(
            [
                obj.get(
                    "degradation_protocol_sha256"
                )
                == base.degradation_protocol_sha256(),
                int(
                    obj.get(
                        "degradation_implementation_revision",
                        -1,
                    )
                )
                == int(
                    base.IMPLEMENTATION_REVISION
                ),
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
# Result semantics
# =============================================================================

def apply_common_semantics(
    result: Dict[str, Any],
    *,
    family: str,
) -> None:
    result[
        "validator_version"
    ] = VALIDATOR_VERSION

    result[
        "regime"
    ] = REGIME

    result[
        "training_relation"
    ] = training_relation(
        family
    )

    result[
        "input_modalities"
    ] = [
        "RGB",
        "NIR",
    ]

    result[
        "nir_used"
    ] = True

    result[
        "fusion"
    ] = (
        "RGB-anchored degradation-aware residual NIR fusion"
    )

    result[
        "quality_gate"
    ] = True

    result[
        "gate_supervision"
    ] = True

    result[
        "gate_type"
    ] = GATE_TYPE

    result[
        "robust4_training_families"
    ] = [
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
        "fog",
    ]

    result[
        "fog_seen_during_training"
    ] = (
        family
        == "fog"
    )


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
    ] = (
        clean_miou
    )

    result[
        "drop_miou"
    ] = (
        drop
    )

    result[
        "delta_miou"
    ] = (
        -drop
    )

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


# =============================================================================
# Gate analysis
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

    families = {
        "gaussian_noise": [
            "Clean",
            base.condition_name(
                "gaussian_noise",
                "L1",
            ),
            base.condition_name(
                "gaussian_noise",
                "L2",
            ),
            base.condition_name(
                "gaussian_noise",
                "L3",
            ),
        ],
        "gaussian_blur": [
            "Clean",
            base.condition_name(
                "gaussian_blur",
                "L1",
            ),
            base.condition_name(
                "gaussian_blur",
                "L2",
            ),
            base.condition_name(
                "gaussian_blur",
                "L3",
            ),
        ],
        "rgb_underexposure": [
            "Clean",
            base.condition_name(
                "rgb_underexposure",
                "L1",
            ),
            base.condition_name(
                "rgb_underexposure",
                "L2",
            ),
            base.condition_name(
                "rgb_underexposure",
                "L3",
            ),
        ],
        "fog": [
            "Clean",
            "fog_L1",
            "fog_L2",
            "fog_L3",
        ],
    }

    output: Dict[
        str,
        Any,
    ] = {}

    all_family_flags = []

    for family, sequence in families.items():
        scales = {}

        scale_flags = []

        for scale in range(
            1,
            NUM_SCALES
            + 1,
        ):
            missing = [
                condition
                for condition in sequence
                if (
                    condition,
                    scale,
                )
                not in lookup
            ]

            if missing:
                raise RuntimeError(
                    f"Missing Gate rows for {family}, "
                    f"scale={scale}: {missing}"
                )

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
                for condition in sequence
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

            scale_flags.append(
                nondecreasing
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
                    for value in means
                ],
                "monotonic_nondecreasing": (
                    nondecreasing
                ),
            }

        family_flag = all(
            scale_flags
        )

        all_family_flags.append(
            family_flag
        )

        output[
            family
        ] = {
            "sequence": (
                sequence
            ),
            "scales": (
                scales
            ),
            "all_scales_monotonic_nondecreasing": (
                family_flag
            ),
        }

    return {
        "model": (
            MODEL_ID
        ),
        "gate_semantics": (
            "g_NIR residual correction strength; larger means more NIR correction"
        ),
        "scientific_question": (
            "With all four degradation families included in Robust-4 training, "
            "does predicted NIR correction increase as validation degradation "
            "severity increases?"
        ),
        "families": (
            output
        ),
        "all_families_all_scales_monotonic_nondecreasing": (
            all(
                all_family_flags
            )
        ),
    }


# =============================================================================
# M3 vs M2
# =============================================================================

def load_m2_summary_map(
    path: Path,
) -> Dict[
    str,
    Dict[
        str,
        Any,
    ]
]:
    obj = load_json(
        path
    )

    rows = obj.get(
        "results"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            "M2 summary has no results list."
        )

    return {
        str(
            row[
                "condition"
            ]
        ): dict(
            row
        )
        for row in rows
        if isinstance(
            row,
            Mapping,
        )
        and "condition"
        in row
    }


def write_comparison_vs_m2(
    *,
    output_root: Path,
    results: Sequence[
        Mapping[str, Any]
    ],
    m2_summary_path: Path,
) -> None:
    if not m2_summary_path.is_file():
        print(
            f"[M2 comparison] skipped; summary not found: "
            f"{m2_summary_path}",
            flush=True,
        )

        return

    m2 = load_m2_summary_map(
        m2_summary_path
    )

    rows = []

    for result in results:
        condition = str(
            result[
                "condition"
            ]
        )

        if condition not in m2:
            continue

        m2_miou = float(
            m2[
                condition
            ][
                "miou"
            ]
        )

        m3_miou = float(
            result[
                "miou"
            ]
        )

        rows.append(
            {
                "condition": (
                    condition
                ),
                "family": (
                    result.get(
                        "family",
                        result.get(
                            "corruption"
                        ),
                    )
                ),
                "severity_level": (
                    result.get(
                        "severity_level"
                    )
                ),
                "m2_fixed_robust4_miou": (
                    m2_miou
                ),
                "m3_darf_robust4_miou": (
                    m3_miou
                ),
                "absolute_gain_miou": (
                    m3_miou
                    - m2_miou
                ),
                "relative_gain_vs_m2_pct": (
                    100.0
                    * (
                        m3_miou
                        - m2_miou
                    )
                    / m2_miou
                    if m2_miou
                    != 0
                    else None
                ),
                "m2_clean_relative_drop_pct": (
                    m2[
                        condition
                    ].get(
                        "relative_drop_pct"
                    )
                ),
                "m3_clean_relative_drop_pct": (
                    result.get(
                        "relative_drop_pct"
                    )
                ),
            }
        )

    degraded = [
        row
        for row in rows
        if row[
            "condition"
        ]
        != "Clean"
    ]

    mean_gain = (
        float(
            np.mean(
                [
                    row[
                        "absolute_gain_miou"
                    ]
                    for row in degraded
                ]
            )
        )
        if degraded
        else None
    )

    save_json(
        output_root
        / "comparison_vs_m2.json",
        {
            "m2_source": str(
                m2_summary_path
            ),
            "m3_model": (
                MODEL_ID
            ),
            "scientific_question": (
                "What is the contribution of adaptive DARF gating relative "
                "to fixed g=0.5 when the dual B2 encoders, NIR residual path, "
                "Robust-4 training, loss, and evaluation protocol are matched?"
            ),
            "mean_absolute_gain_across_12_degraded_conditions": (
                mean_gain
            ),
            "comparison": (
                rows
            ),
        },
    )

    write_csv(
        output_root
        / "comparison_vs_m2.csv",
        rows,
        (
            "condition",
            "family",
            "severity_level",
            "m2_fixed_robust4_miou",
            "m3_darf_robust4_miou",
            "absolute_gain_miou",
            "relative_gain_vs_m2_pct",
            "m2_clean_relative_drop_pct",
            "m3_clean_relative_drop_pct",
        ),
    )

    if mean_gain is not None:
        print(
            "[M2 comparison] mean M3-M2 gain across "
            f"12 degraded conditions: {mean_gain:+.6f} mIoU",
            flush=True,
        )


def m2_metrics_path(
    *,
    m2_root: Path,
    family: str,
    condition: str,
) -> Path:
    if condition == "Clean":
        return (
            m2_root
            / "clean_val"
            / "metrics.json"
        )

    if family == "fog":
        return (
            m2_root
            / "fog_seen_val"
            / condition
            / "metrics.json"
        )

    return (
        m2_root
        / "robustness_val_v2"
        / condition
        / "metrics.json"
    )


def write_per_class_comparison_vs_m2(
    *,
    output_root: Path,
    results: Sequence[
        Mapping[str, Any]
    ],
    m2_root: Path,
) -> None:
    rows = []

    for result in results:
        condition = str(
            result[
                "condition"
            ]
        )

        family = str(
            result.get(
                "family",
                result.get(
                    "corruption",
                    "clean",
                ),
            )
        )

        path = m2_metrics_path(
            m2_root=(
                m2_root
            ),
            family=(
                family
            ),
            condition=(
                condition
            ),
        )

        if not path.is_file():
            continue

        try:
            m2 = load_json(
                path
            )
        except Exception:
            continue

        m2_classes = {
            str(
                row[
                    "class_name"
                ]
            ): row
            for row in m2.get(
                "per_class",
                []
            )
            if isinstance(
                row,
                Mapping,
            )
            and "class_name"
            in row
        }

        m3_classes = {
            str(
                row[
                    "class_name"
                ]
            ): row
            for row in result.get(
                "per_class",
                []
            )
            if isinstance(
                row,
                Mapping,
            )
            and "class_name"
            in row
        }

        for class_name in sorted(
            set(
                m2_classes
            )
            & set(
                m3_classes
            )
        ):
            m2_iou = float(
                m2_classes[
                    class_name
                ][
                    "iou"
                ]
            )

            m3_iou = float(
                m3_classes[
                    class_name
                ][
                    "iou"
                ]
            )

            rows.append(
                {
                    "condition": (
                        condition
                    ),
                    "family": (
                        family
                    ),
                    "class_name": (
                        class_name
                    ),
                    "m2_iou": (
                        m2_iou
                    ),
                    "m3_iou": (
                        m3_iou
                    ),
                    "gain_iou": (
                        m3_iou
                        - m2_iou
                    ),
                }
            )

    if not rows:
        print(
            "[per-class M2 comparison] skipped; no compatible M2 metrics found.",
            flush=True,
        )

        return

    write_csv(
        output_root
        / "per_class_comparison_vs_m2.csv",
        rows,
        (
            "condition",
            "family",
            "class_name",
            "m2_iou",
            "m3_iou",
            "gain_iou",
        ),
    )


# =============================================================================
# Summary
# =============================================================================

def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": (
            MODEL_ID
        ),
        "condition": (
            result.get(
                "condition"
            )
        ),
        "family": (
            result.get(
                "family",
                result.get(
                    "corruption"
                ),
            )
        ),
        "severity_level": (
            result.get(
                "severity_level"
            )
        ),
        "severity_rank": (
            result.get(
                "severity_rank"
            )
        ),
        "training_relation": (
            result.get(
                "training_relation"
            )
        ),
        "clean_miou": (
            result.get(
                "clean_reference_miou"
            )
        ),
        "miou": (
            result.get(
                "miou"
            )
        ),
        "drop_miou": (
            result.get(
                "drop_miou"
            )
        ),
        "relative_drop_pct": (
            result.get(
                "relative_drop_pct"
            )
        ),
        "retention_pct": (
            result.get(
                "retention_pct"
            )
        ),
        "pixel_accuracy": (
            result.get(
                "pixel_accuracy"
            )
        ),
        "mean_class_accuracy": (
            result.get(
                "mean_class_accuracy"
            )
        ),
        "validation_seconds": (
            result.get(
                "validation_seconds"
            )
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    x = parse_args()

    checkpoint_path = resolve(
        x.checkpoint
    )

    output_root = resolve(
        x.output_root
    )

    m2_summary_path = resolve(
        x.m2_summary
    )

    m2_eval_root = resolve(
        x.m2_eval_root
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_base_identity()

    device = base.get_device(
        x.device
    )

    amp_enabled = (
        device.type
        == "cuda"
        and not x.no_amp
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, metadata = (
        base.unwrap_checkpoint(
            checkpoint
        )
    )

    validate_checkpoint_metadata(
        metadata
    )

    checkpoint_epoch = metadata.get(
        "epoch"
    )

    checkpoint_step = metadata.get(
        "global_step",
        metadata.get(
            "step"
        ),
    )

    checkpoint_epoch = (
        int(
            checkpoint_epoch
        )
        if checkpoint_epoch
        is not None
        else None
    )

    checkpoint_step = (
        int(
            checkpoint_step
        )
        if checkpoint_step
        is not None
        else None
    )

    planned_updates = (
        metadata[
            "protocol"
        ].get(
            "total_update_steps"
        )
    )

    print(
        "=" * 132
    )

    print(
        f"{MODEL_NAME} | COMPLETE 13-CONDITION VALIDATION"
    )

    print(
        "=" * 132
    )

    print(
        f"checkpoint      : {checkpoint_path}"
    )

    print(
        f"epoch           : "
        f"{checkpoint_epoch + 1 if checkpoint_epoch is not None else 'unknown'}"
    )

    print(
        f"global_step     : {checkpoint_step}"
    )

    print(
        f"planned updates : {planned_updates}"
    )

    print(
        f"device / AMP    : {device} / {amp_enabled}"
    )

    print(
        f"batch size      : {x.batch_size}"
    )

    print(
        "seen families   : Noise / Blur / Underexposure / Fog"
    )

    print(
        "NIR corruption  : none; clean / unchanged"
    )

    print(
        f"output root     : {output_root}"
    )

    print(
        f"start local     : {format_dt(local_now())}"
    )

    print(
        "=" * 132
    )

    if (
        planned_updates
        is not None
        and checkpoint_step
        is not None
        and checkpoint_step
        != int(
            planned_updates
        )
    ):
        print(
            "[checkpoint note] "
            f"successful optimizer steps differ from nominal by "
            f"{int(planned_updates) - checkpoint_step}. "
            "This is acceptable when all epochs completed and AMP skipped a "
            "small number of overflowed optimizer updates.",
            flush=True,
        )

    model, model_meta = (
        build_model_d_darf_b2(
            PROJECT_ROOT
        )
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
            "Strict M3 checkpoint loading returned incompatibilities."
        )

    model.to(
        device
    )

    model.eval()

    print(
        "[model] strict load PASS | "
        f"parameters={model_meta['parameters']['total']:,} | "
        "RGB+NIR | dynamic DARF gate",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Freeze protocol snapshots
    # -------------------------------------------------------------------------

    standard_root = (
        output_root
        / "robustness_val_v2"
    )

    standard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.write_degradation_protocol(
        standard_root
        / "degradation_protocol.json"
    )

    fog_root = (
        output_root
        / "fog_seen_val"
    )

    fog_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fog_payload = dict(
        FOG_PROTOCOL
    )

    fog_payload[
        "sha256"
    ] = fog_protocol_sha256()

    fog_payload[
        "evaluation_role_for_this_model"
    ] = (
        "seen degradation family under Robust-4 training; "
        "RGB degraded, NIR clean"
    )

    save_json(
        fog_root
        / "fog_protocol.json",
        fog_payload,
    )

    # -------------------------------------------------------------------------
    # Condition plan
    # -------------------------------------------------------------------------

    conditions: List[
        Dict[str, Any]
    ] = [
        {
            "kind": (
                "clean"
            ),
            "condition": (
                "Clean"
            ),
            "family": (
                "clean"
            ),
            "corruption": (
                "Clean"
            ),
            "level": (
                ""
            ),
            "severity_rank": (
                0
            ),
        }
    ]

    for corruption, level in STANDARD_CONDITIONS:
        conditions.append(
            {
                "kind": (
                    "standard"
                ),
                "condition": (
                    base.condition_name(
                        corruption,
                        level,
                    )
                ),
                "family": (
                    corruption
                ),
                "corruption": (
                    corruption
                ),
                "level": (
                    level
                ),
                "severity_rank": int(
                    level[
                        1:
                    ]
                ),
            }
        )

    for level in FOG_LEVELS:
        conditions.append(
            {
                "kind": (
                    "fog"
                ),
                "condition": (
                    f"fog_{level}"
                ),
                "family": (
                    "fog"
                ),
                "corruption": (
                    "fog"
                ),
                "level": (
                    level
                ),
                "severity_rank": int(
                    level[
                        1:
                    ]
                ),
            }
        )

    clean_dataset = (
        base.PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )
    )

    if (
        len(
            clean_dataset.tile_ids
        )
        != 6
        or len(
            clean_dataset.window_coordinates
        )
        != 256
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    progress = ValidationProgress(
        total_conditions=(
            len(
                conditions
            )
        ),
        output_path=(
            output_root
            / "validation_progress.json"
        ),
    )

    results: List[
        Dict[str, Any]
    ] = []

    all_gate_rows: List[
        Dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    for index, item in enumerate(
        conditions,
        start=1,
    ):
        kind = str(
            item[
                "kind"
            ]
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

        out_dir = (
            condition_output_dir(
                output_root=(
                    output_root
                ),
                kind=(
                    kind
                ),
                condition=(
                    condition
                ),
            )
        )

        metrics_path = (
            out_dir
            / "metrics.json"
        )

        print()

        print(
            "-" * 132
        )

        print(
            f"[Condition {index:02d}/{len(conditions):02d}] "
            f"{condition} | "
            f"{training_relation(family)}"
        )

        print(
            "-" * 132
        )

        started = time.time()

        existing = None

        if not x.force:
            existing = compatible_existing(
                metrics_path,
                checkpoint_path=(
                    checkpoint_path
                ),
                checkpoint_step=(
                    checkpoint_step
                ),
                condition=(
                    condition
                ),
                kind=(
                    kind
                ),
            )

        reused = (
            existing
            is not None
        )

        if existing is not None:
            result = (
                existing
            )

            gate_rows = [
                dict(
                    row
                )
                for row in result[
                    "gate_statistics"
                ]
            ]

            print(
                f"[resume] compatible existing result: {metrics_path}",
                flush=True,
            )

        else:
            if kind == "clean":
                dataset = (
                    clean_dataset
                )

                result, gate_rows = (
                    base.evaluate_condition(
                        model=(
                            model
                        ),
                        dataset=(
                            dataset
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
                            out_dir
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                        checkpoint_epoch=(
                            checkpoint_epoch
                        ),
                        checkpoint_global_step=(
                            checkpoint_step
                        ),
                        batch_size=(
                            x.batch_size
                        ),
                        device=(
                            device
                        ),
                        amp_enabled=(
                            amp_enabled
                        ),
                        log_every=(
                            x.log_every
                        ),
                        confusion_chunk_rows=(
                            x.confusion_chunk_rows
                        ),
                        save_predictions=(
                            x.save_predictions
                        ),
                    )
                )

            elif kind == "standard":
                dataset = (
                    base.DegradedPotsdamSlidingWindowDataset(
                        PROJECT_ROOT,
                        split="val",
                        corruption=(
                            item[
                                "corruption"
                            ]
                        ),
                        level=(
                            item[
                                "level"
                            ]
                        ),
                    )
                )

                base.degradation_probe(
                    clean_dataset=(
                        clean_dataset
                    ),
                    degraded_dataset=(
                        dataset
                    ),
                )

                result, gate_rows = (
                    base.evaluate_condition(
                        model=(
                            model
                        ),
                        dataset=(
                            dataset
                        ),
                        condition=(
                            condition
                        ),
                        corruption=(
                            item[
                                "corruption"
                            ]
                        ),
                        severity_level=(
                            item[
                                "level"
                            ]
                        ),
                        severity_rank=(
                            item[
                                "severity_rank"
                            ]
                        ),
                        output_dir=(
                            out_dir
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                        checkpoint_epoch=(
                            checkpoint_epoch
                        ),
                        checkpoint_global_step=(
                            checkpoint_step
                        ),
                        batch_size=(
                            x.batch_size
                        ),
                        device=(
                            device
                        ),
                        amp_enabled=(
                            amp_enabled
                        ),
                        log_every=(
                            x.log_every
                        ),
                        confusion_chunk_rows=(
                            x.confusion_chunk_rows
                        ),
                        save_predictions=(
                            x.save_predictions
                        ),
                    )
                )

            else:
                dataset = (
                    FogPotsdamSlidingWindowDataset(
                        PROJECT_ROOT,
                        split="val",
                        level=(
                            item[
                                "level"
                            ]
                        ),
                        fog_chunk_rows=(
                            x.fog_chunk_rows
                        ),
                    )
                )

                base.degradation_probe(
                    clean_dataset=(
                        clean_dataset
                    ),
                    degraded_dataset=(
                        dataset
                    ),
                )

                with fog_protocol_patch():
                    result, gate_rows = (
                        base.evaluate_condition(
                            model=(
                                model
                            ),
                            dataset=(
                                dataset
                            ),
                            condition=(
                                condition
                            ),
                            corruption=(
                                "fog"
                            ),
                            severity_level=(
                                item[
                                    "level"
                                ]
                            ),
                            severity_rank=(
                                item[
                                    "severity_rank"
                                ]
                            ),
                            output_dir=(
                                out_dir
                            ),
                            checkpoint_path=(
                                checkpoint_path
                            ),
                            checkpoint_epoch=(
                                checkpoint_epoch
                            ),
                            checkpoint_global_step=(
                                checkpoint_step
                            ),
                            batch_size=(
                                x.batch_size
                            ),
                            device=(
                                device
                            ),
                            amp_enabled=(
                                amp_enabled
                            ),
                            log_every=(
                                x.log_every
                            ),
                            confusion_chunk_rows=(
                                x.confusion_chunk_rows
                            ),
                            save_predictions=(
                                x.save_predictions
                            ),
                        )
                    )

                # Replace generic degradation metadata with explicit Fog fields.
                result.pop(
                    "degradation_protocol_version",
                    None,
                )

                result.pop(
                    "degradation_implementation_revision",
                    None,
                )

                result.pop(
                    "degradation_protocol_sha256",
                    None,
                )

                result[
                    "fog_protocol_version"
                ] = (
                    FOG_PROTOCOL_VERSION
                )

                result[
                    "fog_implementation_revision"
                ] = (
                    FOG_IMPLEMENTATION_REVISION
                )

                result[
                    "fog_protocol_sha256"
                ] = (
                    fog_protocol_sha256()
                )

                result[
                    "fog_seen_during_training"
                ] = (
                    True
                )

                save_json(
                    metrics_path,
                    result,
                )

        apply_common_semantics(
            result,
            family=(
                family
            ),
        )

        result[
            "family"
        ] = (
            family
        )

        save_json(
            metrics_path,
            result,
        )

        results.append(
            result
        )

        all_gate_rows.extend(
            dict(
                row
            )
            for row in gate_rows
        )

        elapsed = (
            time.time()
            - started
        )

        print(
            f"[condition result] "
            f"{condition:<24} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"gNIR="
            f"{[round(float(row['g_nir_mean']), 4) for row in gate_rows]} | "
            f"time={float(result['validation_seconds']):.1f}s",
            flush=True,
        )

        progress.condition_done(
            condition=(
                condition
            ),
            condition_seconds=(
                elapsed
            ),
            miou=float(
                result[
                    "miou"
                ]
            ),
            reused=(
                reused
            ),
        )

        if kind != "clean":
            method = getattr(
                dataset,
                "clear_degradation_cache",
                None,
            )

            if callable(
                method
            ):
                method()

            del dataset

    # -------------------------------------------------------------------------
    # Clean-relative metrics
    # -------------------------------------------------------------------------

    clean_result = next(
        (
            result
            for result in results
            if result[
                "condition"
            ]
            == "Clean"
        ),
        None,
    )

    if clean_result is None:
        raise RuntimeError(
            "Clean validation result missing."
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
            ] = (
                0.0
            )

            result[
                "delta_miou"
            ] = (
                0.0
            )

            result[
                "relative_drop_pct"
            ] = (
                0.0
            )

            result[
                "retention_pct"
            ] = (
                100.0
            )
        else:
            add_clean_relative_metrics(
                result,
                clean_miou=(
                    clean_miou
                ),
            )

        family = str(
            result.get(
                "family",
                result.get(
                    "corruption",
                    "clean",
                ),
            )
        )

        if result[
            "condition"
        ] == "Clean":
            kind = "clean"
        elif family == "fog":
            kind = "fog"
        else:
            kind = "standard"

        save_json(
            condition_output_dir(
                output_root=(
                    output_root
                ),
                kind=(
                    kind
                ),
                condition=str(
                    result[
                        "condition"
                    ]
                ),
            )
            / "metrics.json",
            result,
        )

    # -------------------------------------------------------------------------
    # Main summary
    # -------------------------------------------------------------------------

    summary_rows = [
        summary_row(
            result
        )
        for result in results
    ]

    save_json(
        output_root
        / "all_conditions_summary.json",
        {
            "validator_version": (
                VALIDATOR_VERSION
            ),
            "model": (
                MODEL_ID
            ),
            "model_name": (
                MODEL_NAME
            ),
            "backbone": (
                "SegFormer-B2"
            ),
            "regime": (
                REGIME
            ),
            "input_modalities": [
                "RGB",
                "NIR",
            ],
            "fusion": (
                "RGB-anchored degradation-aware residual NIR fusion"
            ),
            "quality_gate": (
                True
            ),
            "gate_type": (
                GATE_TYPE
            ),
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_global_step": (
                checkpoint_step
            ),
            "robust_training": {
                "families": [
                    "gaussian_noise",
                    "gaussian_blur",
                    "rgb_underexposure",
                    "fog",
                ],
                "fog_seen_during_training": (
                    True
                ),
                "nir_degraded": (
                    False
                ),
            },
            "results": (
                summary_rows
            ),
        },
    )

    write_csv(
        output_root
        / "all_conditions_summary.csv",
        summary_rows,
        (
            "model",
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
        ),
    )

    # -------------------------------------------------------------------------
    # Gate statistics + monotonicity
    # -------------------------------------------------------------------------

    save_json(
        output_root
        / "gate_statistics.json",
        {
            "model": (
                MODEL_ID
            ),
            "gate_semantics": (
                "g_NIR residual correction strength; larger means more NIR correction"
            ),
            "rows": (
                all_gate_rows
            ),
        },
    )

    write_csv(
        output_root
        / "gate_statistics.csv",
        all_gate_rows,
        GATE_FIELDS,
    )

    monotonicity = gate_monotonicity(
        all_gate_rows
    )

    save_json(
        output_root
        / "gate_monotonicity.json",
        monotonicity,
    )

    # -------------------------------------------------------------------------
    # M3 vs M2
    # -------------------------------------------------------------------------

    write_comparison_vs_m2(
        output_root=(
            output_root
        ),
        results=(
            results
        ),
        m2_summary_path=(
            m2_summary_path
        ),
    )

    write_per_class_comparison_vs_m2(
        output_root=(
            output_root
        ),
        results=(
            results
        ),
        m2_root=(
            m2_eval_root
        ),
    )

    # -------------------------------------------------------------------------
    # Final console report
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 154
    )

    print(
        f"{MODEL_NAME} | FINAL VALIDATION SUMMARY"
    )

    print(
        "=" * 154
    )

    print(
        f"{'Condition':<26} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'RelDrop%':>10} "
        f"{'Retention%':>12} "
        f"{'g1':>8} "
        f"{'g2':>8} "
        f"{'g3':>8} "
        f"{'g4':>8}"
    )

    print(
        "-" * 154
    )

    gate_lookup = {}

    for row in all_gate_rows:
        gate_lookup[
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
            )
        ] = float(
            row[
                "g_nir_mean"
            ]
        )

    for result in results:
        condition = str(
            result[
                "condition"
            ]
        )

        gates = [
            gate_lookup[
                (
                    condition,
                    scale,
                )
            ]
            for scale in range(
                1,
                NUM_SCALES
                + 1,
            )
        ]

        print(
            f"{condition:<26} "
            f"{float(result['miou']):>10.6f} "
            f"{float(result['drop_miou']):>10.6f} "
            f"{float(result['relative_drop_pct']):>10.3f} "
            f"{float(result['retention_pct']):>12.3f} "
            f"{gates[0]:>8.4f} "
            f"{gates[1]:>8.4f} "
            f"{gates[2]:>8.4f} "
            f"{gates[3]:>8.4f}"
        )

    print(
        "-" * 154
    )

    print(
        f"Clean mIoU       : {clean_miou:.6f}"
    )

    print(
        "Gate monotonic   : "
        f"{monotonicity['all_families_all_scales_monotonic_nondecreasing']}"
    )

    print(
        f"Summary          : "
        f"{output_root / 'all_conditions_summary.csv'}"
    )

    print(
        f"M3 vs M2         : "
        f"{output_root / 'comparison_vs_m2.csv'}"
    )

    print(
        f"Per-class M3-M2  : "
        f"{output_root / 'per_class_comparison_vs_m2.csv'}"
    )

    print(
        f"Gate statistics  : "
        f"{output_root / 'gate_statistics.csv'}"
    )

    print(
        f"Gate monotonicity: "
        f"{output_root / 'gate_monotonicity.json'}"
    )

    print(
        f"Progress         : "
        f"{output_root / 'validation_progress.json'}"
    )

    print(
        "=" * 154
    )


if __name__ == "__main__":
    main()
