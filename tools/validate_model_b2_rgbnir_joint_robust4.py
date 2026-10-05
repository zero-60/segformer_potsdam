#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified 13-condition validation for the formal Joint RGB+NIR Robust-4 experiments.

Supports
--------
M2' / fixed:
    B2-RGBNIR-Fixed-Joint-Robust4

M3' / darf:
    B2-DARF-Joint-Robust4

Examples
--------
    python tools/validate_model_b2_rgbnir_joint_robust4.py --variant fixed
    python tools/validate_model_b2_rgbnir_joint_robust4.py --variant darf

Protocol
--------
13 conditions:
    Clean
    Gaussian Noise L1/L2/L3
    Gaussian Blur L1/L2/L3
    Underexposure L1/L2/L3
    Fog L1/L2/L3

Critical multimodal rule:
    For every degraded condition, RGB AND NIR are degraded on the full raw
    6000x6000 tile before any 512x512 sliding-window crop.

Coupling:
    Gaussian Noise:
        same sigma level, independent deterministic RGB/NIR noise realization.
    Gaussian Blur:
        same sigma.
    Underexposure:
        same attenuation alpha.
    Fog:
        same low-frequency t_rgb spatial field, while
            t_nir = 1 - nir_fog_scatter_ratio * (1 - t_rgb)
        with default ratio 0.65. NIR is never left clean.

Metric protocol:
    - 6 frozen Potsdam validation tiles
    - 512x512 sliding windows
    - stride / coordinates inherited from frozen Potsdam Dataset Protocol
    - overlap mean-logit fusion
    - one full 6000x6000 prediction per tile
    - one GLOBAL confusion matrix across all 6 tiles
    - global mIoU + per-class IoU + per-tile metrics
    - deterministic full-tile degradation

Fairness / compatibility:
    The RGB side deliberately reuses the already-frozen RGB validation
    degradation implementations so M0/M1 RGB conditions remain directly
    comparable.  This script changes the multimodal validation assumption:
    NIR is degraded jointly rather than kept clean.

Runtime visibility:
    - condition / tile / batch progress
    - elapsed time
    - ETA and estimated finish time
    - GPU allocated/reserved/peak memory
    - fixed-fusion residual response or DARF gate response

Expected location:
    tools/validate_model_b2_rgbnir_joint_robust4.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgb as base
import validate_model_d_fog as fog_base
import train_model_b2_rgbnir_joint_robust4 as train_joint

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset, require
from evaluation import rgb_degradation_protocol as rgb_protocol
from joint_multimodal_robust4 import DEFAULT_NIR_FOG_SCATTER_RATIO


# =============================================================================
# Identity / protocol constants
# =============================================================================

VALIDATOR_VERSION = "1.0.0"
JOINT_VALIDATION_PROTOCOL_VERSION = "Joint RGB+NIR Validation Protocol v1"
JOINT_VALIDATION_IMPLEMENTATION_REVISION = 1

# Only used to make the NIR Gaussian-noise realization deterministic and
# independent from the frozen RGB Gaussian-noise realization.
JOINT_VALIDATION_SEED = 20261002

REGIME = "joint_robust4"
NUM_SCALES = int(train_joint.NUM_SCALES)
FIXED_NIR_STRENGTH = float(train_joint.FIXED_NIR_STRENGTH)

DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 0
DEFAULT_LOG_EVERY = 16
DEFAULT_CONFUSION_CHUNK_ROWS = 512
DEFAULT_FOG_CHUNK_ROWS = 128

VARIANT_CONFIG = {
    "fixed": {
        "model_id": train_joint.FIXED_MODEL_ID,
        "model_name": train_joint.FIXED_MODEL_NAME,
        "checkpoint": (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_rgbnir_fixed_joint_robust4"
            / "checkpoints"
            / "final.pt"
        ),
        "output_root": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "b2_rgbnir_fixed_joint_robust4"
        ),
        "reference_summary": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "b2_rgb_robust4"
            / "all_conditions_summary.json"
        ),
        "reference_name": "M1 B2-RGB-Robust4",
    },
    "darf": {
        "model_id": train_joint.DARF_MODEL_ID,
        "model_name": train_joint.DARF_MODEL_NAME,
        "checkpoint": (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_darf_joint_robust4"
            / "checkpoints"
            / "final.pt"
        ),
        "output_root": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "b2_darf_joint_robust4"
        ),
        "reference_summary": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "b2_rgbnir_fixed_joint_robust4"
            / "all_conditions_summary.json"
        ),
        "reference_name": "M2' B2-RGBNIR-Fixed-Joint-Robust4",
    },
}


# Preserve the established RGB condition names so old M0/M1 summaries line up.
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
FOG_LEVELS = ("L1", "L2", "L3")


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def joint_validation_protocol() -> Dict[str, Any]:
    standard = {}
    for corruption in (
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    ):
        standard[corruption] = {
            level: rgb_protocol.condition_spec(corruption, level)
            for level in ("L1", "L2", "L3")
        }

    fog_levels = {
        level: fog_base.fog_spec(level)
        for level in FOG_LEVELS
    }

    return {
        "protocol_version": JOINT_VALIDATION_PROTOCOL_VERSION,
        "implementation_revision": JOINT_VALIDATION_IMPLEMENTATION_REVISION,
        "seed": JOINT_VALIDATION_SEED,
        "scientific_role": (
            "formal deterministic validation for Joint RGB+NIR Robust-4"
        ),
        "scope": {
            "rgb": "clean in Clean; degraded in every non-clean condition",
            "nir": "clean in Clean; degraded in every non-clean condition",
            "gt": "unchanged",
            "application_domain": "raw uint8 RGBIR before normalization",
            "spatial_scope": "full 6000x6000 tile before sliding-window crop",
            "overlap_consistency": True,
        },
        "normalization": {
            "rgb": "ImageNet mean/std from frozen Potsdam dataset protocol",
            "nir": "frozen Potsdam train-only independent mean/std",
            "order": (
                "raw full-tile joint degradation -> crop -> modality-specific "
                "normalization"
            ),
        },
        "metric_protocol": {
            "validation_tiles": 6,
            "crop_size": 512,
            "windows_per_tile": 256,
            "fusion": "overlap mean-logit fusion",
            "prediction_scope": "one full prediction per tile",
            "metric_scope": "one GLOBAL confusion matrix across all six tiles",
        },
        "coupling": {
            "gaussian_noise": (
                "same sigma level; deterministic independent noise realization "
                "for RGB and NIR"
            ),
            "gaussian_blur": "same sigma for RGB and NIR",
            "underexposure": "same alpha for RGB and NIR",
            "fog": (
                "shared low-frequency t_rgb spatial field; "
                "t_nir = 1 - r * (1 - t_rgb)"
            ),
        },
        "nir_fog_scatter_ratio": float(DEFAULT_NIR_FOG_SCATTER_RATIO),
        "conditions": {
            "standard": standard,
            "fog": fog_levels,
        },
        "rgb_reference_compatibility": {
            "standard_protocol_version": (
                rgb_protocol.DEGRADATION_PROTOCOL_VERSION
            ),
            "standard_protocol_sha256": (
                rgb_protocol.degradation_protocol_sha256()
            ),
            "fog_protocol_version": fog_base.FOG_PROTOCOL_VERSION,
            "fog_protocol_sha256": fog_base.fog_protocol_sha256(),
            "purpose": (
                "keep RGB degradation pixel-compatible with completed M0/M1 "
                "validation; only the auxiliary NIR assumption changes"
            ),
        },
        "training_correspondence": {
            "noise_sigma_255_range": [5.0, 50.0],
            "blur_sigma_range": [0.5, 4.0],
            "underexposure_alpha_approx_range": [0.94, 0.40],
            "fog_rgb_mean_transmission": "0.90 - 0.50 * severity",
            "nir_fog_rule": (
                "1 - nir_fog_scatter_ratio * (1 - t_rgb)"
            ),
        },
        "physical_note": (
            "NIR fog attenuation ratio is a controlled experimental "
            "approximation, not a calibrated radiative-transfer constant."
        ),
    }


def joint_validation_protocol_sha256() -> str:
    return hashlib.sha256(
        _canonical_json_bytes(joint_validation_protocol())
    ).hexdigest()


# =============================================================================
# Generic helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in fields}
            )


def condition_output_dir(
    output_root: Path,
    item: Mapping[str, Any],
) -> Path:
    kind = str(item["kind"])
    if kind == "clean":
        return output_root / "clean_val"
    if kind == "fog":
        return output_root / "joint_fog_val" / str(item["condition"])
    return output_root / "joint_robustness_val" / str(item["condition"])


def suite_conditions(suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if suite in ("all", "clean"):
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

    if suite in ("all", "standard"):
        for corruption, level in STANDARD_CONDITIONS:
            family = (
                "underexposure"
                if corruption == "rgb_underexposure"
                else corruption
            )
            rows.append(
                {
                    "kind": "standard",
                    # Keep the established condition key for direct comparison.
                    "condition": rgb_protocol.condition_name(
                        corruption,
                        level,
                    ),
                    "family": family,
                    "corruption": corruption,
                    "level": level,
                    "severity_rank": int(level[1:]),
                }
            )

    if suite in ("all", "fog"):
        for level in FOG_LEVELS:
            rows.append(
                {
                    "kind": "fog",
                    "condition": f"fog_{level}",
                    "family": "fog",
                    "corruption": "fog",
                    "level": level,
                    "severity_rank": int(level[1:]),
                }
            )

    return rows


def training_relation(
    *,
    regime: str,
    family: str,
) -> str:
    if family == "clean":
        return "clean_reference"
    return "seen_joint_degradation_family"


def gpu_memory_snapshot(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": False,
            "device": str(device),
        }

    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    gib = float(1024 ** 3)

    free_bytes = None
    total_bytes = None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    except Exception:
        pass

    payload: Dict[str, Any] = {
        "available": True,
        "device": str(device),
        "allocated_gib": float(
            torch.cuda.memory_allocated(index) / gib
        ),
        "reserved_gib": float(
            torch.cuda.memory_reserved(index) / gib
        ),
        "peak_allocated_gib": float(
            torch.cuda.max_memory_allocated(index) / gib
        ),
        "peak_reserved_gib": float(
            torch.cuda.max_memory_reserved(index) / gib
        ),
    }

    if free_bytes is not None and total_bytes is not None:
        payload["free_gib"] = float(free_bytes / gib)
        payload["total_gib"] = float(total_bytes / gib)

    return payload


def gpu_memory_text(device: torch.device) -> str:
    m = gpu_memory_snapshot(device)
    if not m.get("available"):
        return "GPUmem=n/a"
    return (
        f"GPUmem alloc={m['allocated_gib']:.2f}G "
        f"reserved={m['reserved_gib']:.2f}G "
        f"peak={m['peak_allocated_gib']:.2f}G"
    )


class JointValidationProgress(base.ValidationProgress):
    """
    Reuse the established condition/tile ETA logic and append GPU-memory
    information to validation_progress.json.
    """

    def __init__(
        self,
        *,
        total_conditions: int,
        tiles_per_condition: int,
        output_path: Path,
        device: torch.device,
    ):
        super().__init__(
            total_conditions=total_conditions,
            tiles_per_condition=tiles_per_condition,
            output_path=output_path,
        )
        self.device = device

    def _append_runtime_state(self) -> None:
        if not self.output_path.is_file():
            return
        try:
            payload = json.loads(
                self.output_path.read_text(encoding="utf-8")
            )
        except Exception:
            return

        payload["gpu_memory"] = gpu_memory_snapshot(self.device)
        save_json(self.output_path, payload)

    def tile_done(self, **kwargs) -> None:
        super().tile_done(**kwargs)
        self._append_runtime_state()

    def mark_skipped_condition(self, **kwargs) -> None:
        super().mark_skipped_condition(**kwargs)
        self._append_runtime_state()

    def finish(self, *, output_root: Path) -> None:
        super().finish(output_root=output_root)
        self._append_runtime_state()


# =============================================================================
# Deterministic Joint RGB+NIR full-tile degradation
# =============================================================================

def _nir_noise_seed(
    *,
    tile_id: str,
    level: str,
) -> int:
    payload = (
        f"{JOINT_VALIDATION_PROTOCOL_VERSION}|"
        f"{JOINT_VALIDATION_SEED}|"
        f"{tile_id}|nir_independent_gaussian_noise|{level}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
        signed=False,
    )


def _gaussian_noise_nir_uint8(
    nir: np.ndarray,
    *,
    sigma_255: float,
    seed: int,
    chunk_rows: int = 256,
) -> np.ndarray:
    if nir.ndim != 2 or nir.dtype != np.uint8:
        raise ValueError(
            f"NIR full tile must be uint8 [H,W], got "
            f"{nir.shape} {nir.dtype}"
        )
    if sigma_255 <= 0:
        raise ValueError("sigma_255 must be > 0")

    rng = np.random.default_rng(seed)
    out = np.empty_like(nir)

    for y0 in range(0, nir.shape[0], chunk_rows):
        y1 = min(nir.shape[0], y0 + chunk_rows)
        src = nir[y0:y1].astype(np.float32, copy=False)
        noise = rng.normal(
            loc=0.0,
            scale=float(sigma_255),
            size=src.shape,
        ).astype(np.float32)
        degraded = src + noise
        np.clip(degraded, 0.0, 255.0, out=degraded)
        np.rint(degraded, out=degraded)
        out[y0:y1] = degraded.astype(np.uint8)

    return np.ascontiguousarray(out)


def _gaussian_blur_nir_uint8(
    nir: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    if nir.ndim != 2 or nir.dtype != np.uint8:
        raise ValueError(
            f"NIR full tile must be uint8 [H,W], got "
            f"{nir.shape} {nir.dtype}"
        )
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    out = cv2.GaussianBlur(
        nir,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )

    if out.shape != nir.shape or out.dtype != np.uint8:
        raise RuntimeError(
            "NIR GaussianBlur returned unexpected result: "
            f"{out.shape} {out.dtype}"
        )

    return np.ascontiguousarray(out)


def _underexposure_nir_uint8(
    nir: np.ndarray,
    *,
    alpha: float,
    chunk_rows: int = 512,
) -> np.ndarray:
    if nir.ndim != 2 or nir.dtype != np.uint8:
        raise ValueError(
            f"NIR full tile must be uint8 [H,W], got "
            f"{nir.shape} {nir.dtype}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Underexposure alpha must be in (0,1), got {alpha}"
        )

    out = np.empty_like(nir)

    for y0 in range(0, nir.shape[0], chunk_rows):
        y1 = min(nir.shape[0], y0 + chunk_rows)
        degraded = (
            nir[y0:y1].astype(np.float32, copy=False)
            * float(alpha)
        )
        np.clip(degraded, 0.0, 255.0, out=degraded)
        np.rint(degraded, out=degraded)
        out[y0:y1] = degraded.astype(np.uint8)

    return np.ascontiguousarray(out)


def _fog_field_params(
    tile_id: str,
    level: str,
) -> Dict[str, np.ndarray]:
    """
    Reproduce the already-frozen RGB fog field exactly.
    """
    rng = np.random.default_rng(
        fog_base.fog_seed(tile_id, level)
    )
    return {
        "fx": rng.uniform(
            0.35,
            1.60,
            4,
        ).astype(np.float32),
        "fy": rng.uniform(
            0.35,
            1.60,
            4,
        ).astype(np.float32),
        "phase": rng.uniform(
            0.0,
            2.0 * math.pi,
            4,
        ).astype(np.float32),
        "weights": np.asarray(
            [0.34, 0.27, 0.22, 0.17],
            dtype=np.float32,
        ),
        "signs": rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=4,
        ).astype(np.float32),
    }


def _fog_t_rgb_chunk(
    *,
    y0: int,
    y1: int,
    h: int,
    w: int,
    level: str,
    params: Mapping[str, np.ndarray],
) -> np.ndarray:
    spec = fog_base.fog_spec(level)

    x = (
        np.arange(w, dtype=np.float32)
        / max(w - 1, 1)
    )[None, :]
    y = (
        np.arange(y0, y1, dtype=np.float32)
        / max(h - 1, 1)
    )[:, None]

    field = np.zeros(
        (y1 - y0, w),
        dtype=np.float32,
    )
    two_pi = np.float32(2.0 * math.pi)

    for i in range(4):
        phase = (
            two_pi
            * (
                params["fx"][i] * x
                + params["fy"][i] * y
            )
            + params["phase"][i]
        )
        field += (
            params["weights"][i]
            * params["signs"][i]
            * np.sin(phase).astype(np.float32, copy=False)
        )

    np.clip(field, -1.0, 1.0, out=field)

    t_rgb = (
        np.float32(spec["transmission_mean"])
        + np.float32(spec["transmission_variation"])
        * field
    )
    np.clip(
        t_rgb,
        np.float32(spec["transmission_min"]),
        np.float32(spec["transmission_max"]),
        out=t_rgb,
    )
    return t_rgb


def _joint_fog_nir_uint8(
    nir: np.ndarray,
    *,
    tile_id: str,
    level: str,
    nir_fog_scatter_ratio: float,
    chunk_rows: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if nir.ndim != 2 or nir.dtype != np.uint8:
        raise ValueError(
            f"NIR full tile must be uint8 [H,W], got "
            f"{nir.shape} {nir.dtype}"
        )

    ratio = float(nir_fog_scatter_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            "nir_fog_scatter_ratio must be in (0,1]."
        )

    h, w = nir.shape
    params = _fog_field_params(tile_id, level)

    # The frozen RGB fog uses (245,245,245), so the shared scalar
    # atmospheric light is 245 in raw 8-bit space.
    atmosphere = np.float32(
        float(fog_base.ATMOSPHERIC_LIGHT_255[0])
    )

    out = np.empty_like(nir)
    t_rgb_sum = 0.0
    t_nir_sum = 0.0
    pixels = 0

    for y0 in range(0, h, int(chunk_rows)):
        y1 = min(h, y0 + int(chunk_rows))

        t_rgb = _fog_t_rgb_chunk(
            y0=y0,
            y1=y1,
            h=h,
            w=w,
            level=level,
            params=params,
        )

        t_nir = (
            np.float32(1.0)
            - np.float32(ratio)
            * (
                np.float32(1.0)
                - t_rgb
            )
        )

        np.clip(
            t_nir,
            np.float32(0.20),
            np.float32(0.995),
            out=t_nir,
        )

        src = nir[y0:y1].astype(np.float32, copy=False)
        fogged = (
            src * t_nir
            + atmosphere * (
                np.float32(1.0)
                - t_nir
            )
        )

        np.clip(fogged, 0.0, 255.0, out=fogged)
        np.rint(fogged, out=fogged)
        out[y0:y1] = fogged.astype(np.uint8)

        t_rgb_sum += float(t_rgb.sum(dtype=np.float64))
        t_nir_sum += float(t_nir.sum(dtype=np.float64))
        pixels += int(t_rgb.size)

    return (
        np.ascontiguousarray(out),
        {
            "mean_t_rgb": t_rgb_sum / max(pixels, 1),
            "mean_t_nir": t_nir_sum / max(pixels, 1),
        },
    )


def apply_joint_degradation_full_tile(
    rgbir: np.ndarray,
    *,
    tile_id: str,
    corruption: str,
    level: str,
    nir_fog_scatter_ratio: float,
    fog_chunk_rows: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if (
        rgbir.ndim != 3
        or rgbir.shape[2] != 4
        or rgbir.dtype != np.uint8
    ):
        raise ValueError(
            "Expected raw uint8 RGBIR [H,W,4], got "
            f"{rgbir.shape} {rgbir.dtype}"
        )

    rgb = rgbir[..., :3]
    nir = rgbir[..., 3]

    diagnostics: Dict[str, Any] = {
        "tile_id": tile_id,
        "corruption": corruption,
        "level": level,
    }

    if corruption in {
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
    }:
        spec = rgb_protocol.condition_spec(
            corruption,
            level,
        )

        # Preserve the established deterministic RGB realization exactly.
        rgb_out = rgb_protocol.apply_rgb_degradation_full_tile(
            rgb,
            tile_id=tile_id,
            corruption=corruption,
            level=level,
        )

        if corruption == "gaussian_noise":
            sigma_255 = float(spec["sigma_255"])
            nir_out = _gaussian_noise_nir_uint8(
                nir,
                sigma_255=sigma_255,
                seed=_nir_noise_seed(
                    tile_id=tile_id,
                    level=level,
                ),
            )
            diagnostics.update(
                {
                    "sigma_255_rgb": sigma_255,
                    "sigma_255_nir": sigma_255,
                    "noise_realization_coupling": "independent",
                }
            )

        elif corruption == "gaussian_blur":
            sigma = float(spec["sigma"])
            nir_out = _gaussian_blur_nir_uint8(
                nir,
                sigma=sigma,
            )
            diagnostics.update(
                {
                    "sigma_rgb": sigma,
                    "sigma_nir": sigma,
                }
            )

        else:
            alpha = float(spec["alpha"])
            nir_out = _underexposure_nir_uint8(
                nir,
                alpha=alpha,
            )
            diagnostics.update(
                {
                    "alpha_rgb": alpha,
                    "alpha_nir": alpha,
                }
            )

    elif corruption == "fog":
        # Preserve the established deterministic RGB fog exactly.
        rgb_out = fog_base.apply_fog_full_tile(
            rgb,
            tile_id=tile_id,
            level=level,
            chunk_rows=int(fog_chunk_rows),
        )

        nir_out, fog_diag = _joint_fog_nir_uint8(
            nir,
            tile_id=tile_id,
            level=level,
            nir_fog_scatter_ratio=(
                nir_fog_scatter_ratio
            ),
            chunk_rows=int(fog_chunk_rows),
        )

        diagnostics.update(
            {
                **fog_diag,
                "nir_fog_scatter_ratio": float(
                    nir_fog_scatter_ratio
                ),
                "nir_transmission_rule": (
                    "t_nir = 1 - r * (1 - t_rgb)"
                ),
            }
        )

    else:
        raise ValueError(
            f"Unsupported corruption: {corruption}"
        )

    if rgb_out.shape != rgb.shape or rgb_out.dtype != np.uint8:
        raise RuntimeError(
            f"{tile_id}: bad degraded RGB "
            f"{rgb_out.shape} {rgb_out.dtype}"
        )

    if nir_out.shape != nir.shape or nir_out.dtype != np.uint8:
        raise RuntimeError(
            f"{tile_id}: bad degraded NIR "
            f"{nir_out.shape} {nir_out.dtype}"
        )

    out = np.empty_like(rgbir)
    out[..., :3] = rgb_out
    out[..., 3] = nir_out

    return np.ascontiguousarray(out), diagnostics


class JointDegradedPotsdamSlidingWindowDataset(
    PotsdamSlidingWindowDataset
):
    """
    Full-tile Joint RGB+NIR deterministic degradation.

    Exactly one 6000x6000 degraded RGBIR tile is cached at a time.
    Validation must use num_workers=0.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        split: str,
        corruption: str,
        level: str,
        nir_fog_scatter_ratio: float,
        fog_chunk_rows: int,
    ):
        super().__init__(
            project_root=project_root,
            split=split,
        )

        if corruption == "fog":
            fog_base.fog_spec(level)
        else:
            rgb_protocol.condition_spec(
                corruption,
                level,
            )

        self.corruption = str(corruption)
        self.level = str(level)
        self.condition = (
            f"fog_{level}"
            if corruption == "fog"
            else rgb_protocol.condition_name(
                corruption,
                level,
            )
        )
        self.nir_fog_scatter_ratio = float(
            nir_fog_scatter_ratio
        )
        self.fog_chunk_rows = int(fog_chunk_rows)

        self._degraded_cache_tile_id: Optional[str] = None
        self._degraded_cache_rgbir: Optional[np.ndarray] = None
        self._degraded_cache_diagnostics: Optional[
            Dict[str, Any]
        ] = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_degraded_cache_tile_id"] = None
        state["_degraded_cache_rgbir"] = None
        state["_degraded_cache_diagnostics"] = None
        return state

    def clear_degradation_cache(self) -> None:
        self._degraded_cache_tile_id = None
        self._degraded_cache_rgbir = None
        self._degraded_cache_diagnostics = None

    def _get_degraded_rgbir(
        self,
        tile_id: str,
        rgbir: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if (
            self._degraded_cache_tile_id == tile_id
            and self._degraded_cache_rgbir is not None
            and self._degraded_cache_diagnostics is not None
        ):
            return (
                self._degraded_cache_rgbir,
                self._degraded_cache_diagnostics,
            )

        degraded, diagnostics = (
            apply_joint_degradation_full_tile(
                rgbir,
                tile_id=tile_id,
                corruption=self.corruption,
                level=self.level,
                nir_fog_scatter_ratio=(
                    self.nir_fog_scatter_ratio
                ),
                fog_chunk_rows=self.fog_chunk_rows,
            )
        )

        require(
            degraded.shape
            == (
                self.spec.tile_size,
                self.spec.tile_size,
                4,
            ),
            f"{tile_id}: degraded RGBIR shape invalid: "
            f"{degraded.shape}",
        )
        require(
            degraded.dtype == np.uint8,
            f"{tile_id}: degraded RGBIR dtype invalid: "
            f"{degraded.dtype}",
        )

        self._degraded_cache_tile_id = tile_id
        self._degraded_cache_rgbir = degraded
        self._degraded_cache_diagnostics = diagnostics

        return degraded, diagnostics

    def __getitem__(self, index: int) -> Dict[str, Any]:
        (
            tile_id,
            window_index,
            x,
            y,
        ) = self.index_to_tile_window(index)

        rgbir = self._read_rgbir(tile_id)
        degraded_rgbir, diagnostics = (
            self._get_degraded_rgbir(
                tile_id,
                rgbir,
            )
        )

        h = int(self.spec.crop_size)
        w = int(self.spec.crop_size)

        crop = degraded_rgbir[
            y:y + h,
            x:x + w,
            :,
        ]

        require(
            crop.shape == (h, w, 4),
            f"{tile_id}: joint degraded crop invalid: "
            f"{crop.shape}",
        )

        rgb_t, nir_t = self.spec.normalize_rgb_nir(
            crop,
            context=(
                f"{self.split_name} {tile_id} "
                f"{self.condition} window={window_index} "
                f"x={x} y={y}"
            ),
        )

        row = {
            "rgb": rgb_t,
            "nir": nir_t,
            "tile_id": tile_id,
            "tile_index": int(
                self.tile_ids.index(tile_id)
            ),
            "window_index": int(window_index),
            "x": int(x),
            "y": int(y),
            "height": h,
            "width": w,
            "degradation_condition": self.condition,
            "corruption": self.corruption,
            "severity_level": self.level,
            "rgb_degraded": True,
            "nir_degraded": True,
        }

        if self.corruption == "fog":
            row["fog_mean_t_rgb"] = float(
                diagnostics["mean_t_rgb"]
            )
            row["fog_mean_t_nir"] = float(
                diagnostics["mean_t_nir"]
            )

        return row


def build_dataset(
    *,
    item: Mapping[str, Any],
    nir_fog_scatter_ratio: float,
    fog_chunk_rows: int,
) -> PotsdamSlidingWindowDataset:
    if item["kind"] == "clean":
        return PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )

    return JointDegradedPotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
        corruption=str(item["corruption"]),
        level=str(item["level"]),
        nir_fog_scatter_ratio=(
            nir_fog_scatter_ratio
        ),
        fog_chunk_rows=fog_chunk_rows,
    )


def joint_degradation_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: PotsdamSlidingWindowDataset,
    condition: str,
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
                f"{condition}: frozen window metadata "
                f"changed at {key}."
            )

    if torch.equal(
        clean["rgb"],
        degraded["rgb"],
    ):
        raise RuntimeError(
            f"{condition}: RGB did not change."
        )

    if torch.equal(
        clean["nir"],
        degraded["nir"],
    ):
        raise RuntimeError(
            f"{condition}: NIR did not change. "
            "Joint validation forbids clean/unchanged NIR."
        )

    print(
        f"[joint probe] {condition}: PASS | "
        "same frozen window | RGB changed | NIR changed",
        flush=True,
    )


# =============================================================================
# Checkpoint / model
# =============================================================================

def identity(
    variant: str,
) -> Tuple[str, str]:
    cfg = VARIANT_CONFIG[variant]
    return (
        str(cfg["model_id"]),
        str(cfg["model_name"]),
    )


def build_model(
    variant: str,
):
    if variant == "fixed":
        return train_joint.build_model_b2_rgbnir_fixed(
            PROJECT_ROOT,
            fixed_strength=FIXED_NIR_STRENGTH,
        )

    if variant == "darf":
        return train_joint.build_model_d_darf_b2(
            PROJECT_ROOT
        )

    raise ValueError(variant)


def validate_checkpoint_metadata(
    meta: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    model_id, model_name = identity(variant)

    if meta.get("model_id") != model_id:
        raise RuntimeError(
            "Checkpoint model_id mismatch: "
            f"{meta.get('model_id')!r} != {model_id!r}"
        )

    if meta.get("model_name") != model_name:
        raise RuntimeError(
            "Checkpoint model_name mismatch: "
            f"{meta.get('model_name')!r} != {model_name!r}"
        )

    if meta.get("variant") != variant:
        raise RuntimeError(
            "Checkpoint variant mismatch: "
            f"{meta.get('variant')!r} != {variant!r}"
        )

    if meta.get("regime") != REGIME:
        raise RuntimeError(
            "Checkpoint is not joint_robust4."
        )

    protocol = meta.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(
            "Checkpoint protocol metadata missing."
        )

    if (
        protocol.get("protocol_version")
        != train_joint.PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "Joint training protocol_version mismatch."
        )

    if protocol.get("backbone") != "SegFormer-B2":
        raise RuntimeError(
            "Checkpoint is not SegFormer-B2."
        )

    if list(
        protocol.get(
            "input_modalities",
            [],
        )
    ) != ["RGB", "NIR"]:
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
            "Checkpoint metadata says NIR is not used."
        )

    if protocol.get("regime") != REGIME:
        raise RuntimeError(
            "Checkpoint protocol regime is not joint_robust4."
        )

    corruption = protocol.get(
        "joint_corruption_training"
    )
    if not isinstance(corruption, Mapping):
        raise RuntimeError(
            "joint_corruption_training metadata missing."
        )

    if not bool(
        corruption.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "Joint corruption training is disabled."
        )

    expected_families = {
        "gaussian_noise",
        "gaussian_blur",
        "underexposure",
        "fog",
    }
    actual_families = {
        str(x)
        for x in corruption.get(
            "families",
            [],
        )
    }

    if actual_families != expected_families:
        raise RuntimeError(
            "Joint training family mismatch: "
            f"{sorted(actual_families)} != "
            f"{sorted(expected_families)}"
        )

    if str(corruption.get("rgb")) != "degraded":
        raise RuntimeError(
            "Training metadata does not say RGB is degraded."
        )

    if str(corruption.get("nir")) != "degraded":
        raise RuntimeError(
            "Training metadata does not say NIR is degraded."
        )

    checkpoint_ratio = float(
        corruption.get(
            "fog",
            {},
        ).get(
            "nir_fog_scatter_ratio",
            float("nan"),
        )
    )

    if not math.isfinite(checkpoint_ratio):
        raise RuntimeError(
            "Checkpoint NIR fog ratio missing."
        )

    if variant == "fixed":
        if bool(
            protocol.get(
                "quality_gate",
                True,
            )
        ):
            raise RuntimeError(
                "Fixed M2' must not use a quality gate."
            )

        g = float(
            protocol.get(
                "fixed_nir_strength",
                float("nan"),
            )
        )
        if (
            not math.isfinite(g)
            or abs(
                g - FIXED_NIR_STRENGTH
            ) > 1e-12
        ):
            raise RuntimeError(
                "Fixed M2' coefficient mismatch."
            )

    else:
        if not bool(
            protocol.get(
                "quality_gate",
                False,
            )
        ):
            raise RuntimeError(
                "DARF M3' must use a quality gate."
            )

        if bool(
            protocol.get(
                "gate_supervision",
                True,
            )
        ):
            raise RuntimeError(
                "Formal M3' must have Gate BCE disabled."
            )

        gate_training = protocol.get(
            "gate_training"
        )
        if not isinstance(
            gate_training,
            Mapping,
        ):
            raise RuntimeError(
                "M3' gate_training metadata missing."
            )

        if bool(
            gate_training.get(
                "auxiliary_gate_bce",
                True,
            )
        ):
            raise RuntimeError(
                "Formal M3' must not use auxiliary Gate BCE."
            )


# =============================================================================
# Full-tile inference with multimodal diagnostics
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
        required - set(batch.keys())
    )
    if missing:
        raise RuntimeError(
            f"Validation batch missing keys: {missing}"
        )

    rgb = batch["rgb"]
    nir = batch["nir"]

    if (
        rgb.ndim != 4
        or rgb.shape[1] != 3
    ):
        raise RuntimeError(
            f"RGB must be [B,3,H,W], got "
            f"{tuple(rgb.shape)}"
        )

    if (
        nir.ndim != 4
        or nir.shape[1] != 1
    ):
        raise RuntimeError(
            f"NIR must be [B,1,H,W], got "
            f"{tuple(nir.shape)}"
        )

    if (
        rgb.shape[0] != nir.shape[0]
        or rgb.shape[-2:] != nir.shape[-2:]
    ):
        raise RuntimeError(
            "RGB/NIR batch alignment changed."
        )

    if any(
        str(x) != tile_id
        for x in list(batch["tile_id"])
    ):
        raise RuntimeError(
            "Per-tile DataLoader mixed tile IDs."
        )


def _stat_runtime_from_gate(
    *,
    values: torch.Tensor,
) -> Dict[str, Any]:
    """
    values: [N, NUM_SCALES] on CPU float64
    """
    if values.ndim != 2 or values.shape[1] != NUM_SCALES:
        raise RuntimeError(
            f"Bad gate values shape: {tuple(values.shape)}"
        )

    n = int(values.shape[0])
    sums = values.sum(dim=0)
    sumsq = (
        values * values
    ).sum(dim=0)

    return {
        "gate_count": n,
        "gate_sum_by_scale": [
            float(x)
            for x in sums.tolist()
        ],
        "gate_sumsq_by_scale": [
            float(x)
            for x in sumsq.tolist()
        ],
        "gate_min_by_scale": [
            float(x)
            for x in values.min(dim=0).values.tolist()
        ],
        "gate_max_by_scale": [
            float(x)
            for x in values.max(dim=0).values.tolist()
        ],
    }


def infer_one_tile_joint(
    *,
    model,
    dataset,
    tile_id,
    tile_index,
    batch_size,
    num_workers,
    pin_memory,
    device,
    amp_enabled,
    log_every,
):
    if int(num_workers) != 0:
        raise RuntimeError(
            "Joint full-tile degradation validation "
            "requires num_workers=0."
        )

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

    loader = DataLoader(
        Subset(
            dataset,
            range(start, stop),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
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
            base.NUM_CLASSES,
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    started = time.time()
    windows_seen = 0

    variant_detected: Optional[str] = None

    residual_sum = torch.zeros(
        NUM_SCALES,
        dtype=torch.float64,
    )
    residual_weight = 0

    gate_batches: List[
        torch.Tensor
    ] = []

    for batch_index, batch in enumerate(
        loader
    ):
        validate_batch(
            batch,
            tile_id=tile_id,
        )

        rgb = (
            batch["rgb"]
            .to(
                device,
                non_blocking=(
                    pin_memory
                    and device.type == "cuda"
                ),
            )
        )
        nir = (
            batch["nir"]
            .to(
                device,
                non_blocking=(
                    pin_memory
                    and device.type == "cuda"
                ),
            )
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                details = model(
                    rgb,
                    nir,
                    return_details=True,
                )

        logits = (
            details["logits"]
            .float()
        )

        if (
            logits.ndim != 4
            or logits.shape[1] != base.NUM_CLASSES
            or tuple(
                logits.shape[-2:]
            )
            != (
                crop_size,
                crop_size,
            )
        ):
            raise RuntimeError(
                "Unexpected logits shape: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(
            logits
        ).all().item():
            raise FloatingPointError(
                "Logits contain NaN/Inf."
            )

        batch_n = int(
            logits.shape[0]
        )

        if "nir_gate_strength" in details:
            current_variant = "darf"
            gate = (
                details[
                    "nir_gate_strength"
                ]
                .detach()
                .float()
                .cpu()
                .double()
            )

            if gate.shape != (
                batch_n,
                NUM_SCALES,
            ):
                raise RuntimeError(
                    "Bad DARF gate shape: "
                    f"{tuple(gate.shape)}"
                )

            gate_batches.append(
                gate
            )

            diagnostic_text = (
                "gNIR="
                + str(
                    [
                        round(
                            float(x),
                            4,
                        )
                        for x in gate.mean(
                            dim=0
                        ).tolist()
                    ]
                )
            )

        elif "residual_abs_mean" in details:
            current_variant = "fixed"
            residual = (
                details[
                    "residual_abs_mean"
                ]
                .detach()
                .float()
                .cpu()
                .double()
            )

            if residual.numel() != NUM_SCALES:
                raise RuntimeError(
                    "Fixed residual diagnostic "
                    "must have 4 scales."
                )

            residual_sum += (
                residual
                * batch_n
            )
            residual_weight += batch_n

            running = (
                residual_sum
                / max(
                    residual_weight,
                    1,
                )
            )

            diagnostic_text = (
                "residual="
                + str(
                    [
                        round(
                            float(x),
                            5,
                        )
                        for x in running.tolist()
                    ]
                )
            )

        else:
            raise RuntimeError(
                "Model details contain neither "
                "nir_gate_strength nor residual_abs_mean."
            )

        if variant_detected is None:
            variant_detected = current_variant
        elif variant_detected != current_variant:
            raise RuntimeError(
                "Model diagnostic type changed within tile."
            )

        for j in range(batch_n):
            x = int(batch["x"][j])
            y = int(batch["y"][j])

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
            ].add_(
                1.0
            )

        windows_seen += batch_n

        if (
            batch_index % log_every == 0
            or batch_index + 1 == len(loader)
        ):
            tile_elapsed = (
                time.time()
                - started
            )
            tile_fraction = (
                windows_seen
                / windows_per_tile
            )
            tile_eta = (
                tile_elapsed
                * (
                    1.0
                    - tile_fraction
                )
                / max(
                    tile_fraction,
                    1e-12,
                )
            )

            print(
                f"  tile {tile_id} | "
                f"batch {batch_index + 1:03d}/"
                f"{len(loader):03d} | "
                f"windows {windows_seen:03d}/"
                f"{windows_per_tile:03d} | "
                f"elapsed "
                f"{base.format_duration(tile_elapsed)} | "
                f"tile ETA "
                f"{base.format_duration(tile_eta)} | "
                f"{diagnostic_text} | "
                f"{gpu_memory_text(device)}",
                flush=True,
            )

        del details
        del logits
        del rgb
        del nir

    if windows_seen != windows_per_tile:
        raise RuntimeError(
            f"{tile_id}: expected "
            f"{windows_per_tile} windows, "
            f"got {windows_seen}."
        )

    if float(
        coverage.min().item()
    ) <= 0.0:
        raise RuntimeError(
            f"{tile_id}: uncovered full-tile pixels."
        )

    coverage_min = float(
        coverage.min().item()
    )
    coverage_max = float(
        coverage.max().item()
    )

    logits_sum.div_(
        coverage.unsqueeze(0)
    )

    prediction = (
        logits_sum
        .argmax(dim=0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    elapsed = (
        time.time()
        - started
    )

    runtime: Dict[str, Any] = {
        "tile_id": tile_id,
        "windows": windows_seen,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "inference_seconds": elapsed,
        "variant_detected": variant_detected,
        "gpu_memory": gpu_memory_snapshot(
            device
        ),
    }

    if variant_detected == "fixed":
        if residual_weight <= 0:
            raise RuntimeError(
                "No fixed residual diagnostics collected."
            )

        residual_mean = (
            residual_sum
            / residual_weight
        )
        for s in range(NUM_SCALES):
            runtime[
                f"nir_residual_abs_mean_scale{s + 1}"
            ] = float(
                residual_mean[s]
            )

    elif variant_detected == "darf":
        if not gate_batches:
            raise RuntimeError(
                "No DARF gate values collected."
            )

        gate_values = torch.cat(
            gate_batches,
            dim=0,
        )
        runtime.update(
            _stat_runtime_from_gate(
                values=gate_values
            )
        )

    else:
        raise RuntimeError(
            "Failed to detect model variant."
        )

    del logits_sum
    del coverage

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return prediction, runtime


# Patch only this Python process.  The established evaluator still owns:
#   full-tile GT, global confusion matrix, per-class IoU, prediction saving,
#   and tile/condition progress.
base.infer_one_tile = infer_one_tile_joint
base.training_relation = training_relation


# =============================================================================
# Condition diagnostics / result semantics
# =============================================================================

def add_joint_semantics(
    result: Dict[str, Any],
    *,
    variant: str,
    nir_fog_scatter_ratio: float,
) -> None:
    result["validator_version"] = VALIDATOR_VERSION
    result["regime"] = REGIME
    result["variant"] = variant
    result["input_modalities"] = [
        "RGB",
        "NIR",
    ]
    result["nir_used"] = True
    result["quality_gate"] = (
        variant == "darf"
    )
    result["gate_supervision"] = False
    result["joint_validation"] = True

    result["rgb_degraded"] = (
        result["condition"] != "Clean"
    )
    result["nir_degraded"] = (
        result["condition"] != "Clean"
    )

    result["training_relation"] = (
        training_relation(
            regime=REGIME,
            family=str(
                result["family"]
            ),
        )
    )

    result[
        "joint_validation_protocol_version"
    ] = JOINT_VALIDATION_PROTOCOL_VERSION
    result[
        "joint_validation_implementation_revision"
    ] = JOINT_VALIDATION_IMPLEMENTATION_REVISION
    result[
        "joint_validation_protocol_sha256"
    ] = joint_validation_protocol_sha256()

    result["nir_fog_scatter_ratio"] = float(
        nir_fog_scatter_ratio
    )

    if variant == "fixed":
        result["fusion"] = (
            "RGB-anchored fixed NIR residual fusion"
        )
        result["fusion_rule"] = (
            "F_i = F_RGB_i + 0.5 * Adapter_i(F_NIR_i)"
        )
        result["fixed_nir_strength"] = (
            FIXED_NIR_STRENGTH
        )
    else:
        result["fusion"] = (
            "dynamic DARF RGB-anchored NIR residual fusion"
        )
        result["fusion_rule"] = (
            "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
        )
        result["gate_learning"] = (
            "semantic segmentation loss only; no Gate BCE"
        )

    if result["condition"] != "Clean":
        result[
            "rgb_reference_protocol_compatibility"
        ] = {
            "standard_protocol_version": (
                rgb_protocol.DEGRADATION_PROTOCOL_VERSION
            ),
            "standard_protocol_sha256": (
                rgb_protocol.degradation_protocol_sha256()
            ),
            "fog_protocol_version": (
                fog_base.FOG_PROTOCOL_VERSION
            ),
            "fog_protocol_sha256": (
                fog_base.fog_protocol_sha256()
            ),
        }


def fixed_residual_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": result["condition"],
        "family": result["family"],
        "severity_level": result.get(
            "severity_level"
        ),
        "fixed_nir_strength": (
            FIXED_NIR_STRENGTH
        ),
    }

    for s in range(1, NUM_SCALES + 1):
        key = (
            f"nir_residual_abs_mean_scale{s}"
        )
        values = np.asarray(
            [
                float(tile[key])
                for tile in result["tiles"]
            ],
            dtype=np.float64,
        )

        row[f"scale{s}_mean"] = float(
            values.mean()
        )
        row[f"scale{s}_std_across_tiles"] = float(
            values.std(ddof=0)
        )
        row[f"scale{s}_effective_mean"] = float(
            FIXED_NIR_STRENGTH
            * values.mean()
        )

    return row


def darf_gate_rows(
    result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    tiles = result["tiles"]

    total_count = sum(
        int(tile["gate_count"])
        for tile in tiles
    )

    if total_count <= 0:
        raise RuntimeError(
            "DARF gate count is zero."
        )

    rows: List[Dict[str, Any]] = []

    for s in range(NUM_SCALES):
        total_sum = sum(
            float(
                tile[
                    "gate_sum_by_scale"
                ][s]
            )
            for tile in tiles
        )
        total_sumsq = sum(
            float(
                tile[
                    "gate_sumsq_by_scale"
                ][s]
            )
            for tile in tiles
        )

        mean = (
            total_sum
            / total_count
        )
        second = (
            total_sumsq
            / total_count
        )
        std = math.sqrt(
            max(
                second
                - mean * mean,
                0.0,
            )
        )

        minimum = min(
            float(
                tile[
                    "gate_min_by_scale"
                ][s]
            )
            for tile in tiles
        )
        maximum = max(
            float(
                tile[
                    "gate_max_by_scale"
                ][s]
            )
            for tile in tiles
        )

        rows.append(
            {
                "condition": result["condition"],
                "family": result["family"],
                "severity_level": result.get(
                    "severity_level"
                ),
                "severity_rank": result.get(
                    "severity_rank"
                ),
                "scale": s + 1,
                "count": total_count,
                "g_nir_mean": mean,
                "g_nir_std": std,
                "g_nir_min": minimum,
                "g_nir_max": maximum,
            }
        )

    return rows


def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": result["model"],
        "variant": result["variant"],
        "regime": result["regime"],
        "condition": result["condition"],
        "family": result["family"],
        "severity_level": result.get(
            "severity_level"
        ),
        "severity_rank": result.get(
            "severity_rank"
        ),
        "training_relation": result[
            "training_relation"
        ],
        "clean_miou": result.get(
            "clean_reference_miou",
            result["miou"]
            if result["condition"] == "Clean"
            else None,
        ),
        "miou": result["miou"],
        "drop_miou": result.get(
            "drop_miou",
            0.0
            if result["condition"] == "Clean"
            else None,
        ),
        "relative_drop_pct": result.get(
            "relative_drop_pct",
            0.0
            if result["condition"] == "Clean"
            else None,
        ),
        "retention_pct": result.get(
            "retention_pct",
            100.0
            if result["condition"] == "Clean"
            else None,
        ),
        "pixel_accuracy": result[
            "pixel_accuracy"
        ],
        "mean_class_accuracy": result[
            "mean_class_accuracy"
        ],
        "validation_seconds": result[
            "validation_seconds"
        ],
    }


def aggregate_metrics(
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    lookup = {
        str(r["condition"]): r
        for r in results
    }

    clean = float(
        lookup["Clean"]["miou"]
    )

    degraded = [
        float(r["miou"])
        for r in results
        if r["condition"] != "Clean"
    ]

    l3 = [
        float(r["miou"])
        for r in results
        if r.get("severity_level") == "L3"
    ]

    return {
        "clean_miou": clean,
        "mean_degraded_miou_12_conditions": (
            float(
                np.mean(
                    np.asarray(
                        degraded,
                        dtype=np.float64,
                    )
                )
            )
            if degraded
            else None
        ),
        "mean_L3_miou_4_families": (
            float(
                np.mean(
                    np.asarray(
                        l3,
                        dtype=np.float64,
                    )
                )
            )
            if l3
            else None
        ),
        "num_conditions": len(results),
        "num_degraded_conditions": len(
            degraded
        ),
    }


def _condition_aliases(
    condition: str,
) -> Tuple[str, ...]:
    if condition.startswith(
        "rgb_underexposure_"
    ):
        return (
            condition,
            condition.replace(
                "rgb_underexposure_",
                "underexposure_",
                1,
            ),
        )
    if condition.startswith(
        "underexposure_"
    ):
        return (
            condition,
            condition.replace(
                "underexposure_",
                "rgb_underexposure_",
                1,
            ),
        )
    return (condition,)


def write_reference_comparison(
    *,
    results: Sequence[Mapping[str, Any]],
    reference_path: Path,
    reference_name: str,
    output_root: Path,
    current_model_id: str,
) -> None:
    if not reference_path.is_file():
        print(
            f"[reference] skipped; not found: "
            f"{reference_path}",
            flush=True,
        )
        return

    payload = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )

    lookup: Dict[str, Mapping[str, Any]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, Mapping):
            continue
        cond = str(
            row.get(
                "condition",
                "",
            )
        )
        if cond:
            lookup[cond] = row

    rows = []

    for result in results:
        condition = str(
            result["condition"]
        )

        ref = None
        ref_condition = None
        for alias in _condition_aliases(
            condition
        ):
            if alias in lookup:
                ref = lookup[alias]
                ref_condition = alias
                break

        if ref is None:
            continue

        old = float(ref["miou"])
        new = float(result["miou"])

        rows.append(
            {
                "condition": condition,
                "reference_condition": (
                    ref_condition
                ),
                "family": result["family"],
                "severity_level": result.get(
                    "severity_level"
                ),
                "reference_model": payload.get(
                    "model"
                ),
                "reference_miou": old,
                "current_model": current_model_id,
                "current_miou": new,
                "absolute_gain_miou": (
                    new - old
                ),
                "relative_gain_pct": (
                    100.0
                    * (
                        new - old
                    )
                    / old
                    if old != 0
                    else None
                ),
            }
        )

    if not rows:
        print(
            "[reference] no matching condition rows.",
            flush=True,
        )
        return

    save_json(
        output_root
        / "comparison_vs_reference.json",
        {
            "reference_name": reference_name,
            "reference_source": str(
                reference_path
            ),
            "reference_model": payload.get(
                "model"
            ),
            "current_model": current_model_id,
            "rows": rows,
        },
    )

    write_csv(
        output_root
        / "comparison_vs_reference.csv",
        rows,
        [
            "condition",
            "reference_condition",
            "family",
            "severity_level",
            "reference_model",
            "reference_miou",
            "current_model",
            "current_miou",
            "absolute_gain_miou",
            "relative_gain_pct",
        ],
    )

    print(
        f"[reference] written: "
        f"{output_root / 'comparison_vs_reference.csv'}",
        flush=True,
    )


def compatible_existing(
    path: Path,
    *,
    model_id: str,
    variant: str,
    condition: str,
    checkpoint_path: Path,
    checkpoint_step: Optional[int],
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
        obj.get("model") == model_id,
        obj.get("variant") == variant,
        obj.get("regime") == REGIME,
        obj.get("condition") == condition,
        obj.get("split") == "val",
        obj.get("joint_validation") is True,
        obj.get(
            "joint_validation_protocol_sha256"
        )
        == joint_validation_protocol_sha256(),
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

    return obj if all(checks) else None


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate M2'/M3' under deterministic "
            "Joint RGB+NIR Robust-4 conditions."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    p.add_argument(
        "--variant",
        choices=(
            "fixed",
            "darf",
        ),
        required=True,
        help=(
            "fixed = M2', darf = M3'"
        ),
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Default depends on --variant."
        ),
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Default depends on --variant."
        ),
    )

    p.add_argument(
        "--reference-summary",
        type=Path,
        default=None,
        help=(
            "Optional comparison summary. "
            "Default: M1 for fixed, M2' for darf."
        ),
    )

    p.add_argument(
        "--suite",
        choices=(
            "all",
            "clean",
            "standard",
            "fog",
        ),
        default="all",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
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
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
    )

    p.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=DEFAULT_CONFUSION_CHUNK_ROWS,
    )

    p.add_argument(
        "--fog-chunk-rows",
        type=int,
        default=DEFAULT_FOG_CHUNK_ROWS,
    )

    p.add_argument(
        "--nir-fog-scatter-ratio",
        type=float,
        default=DEFAULT_NIR_FOG_SCATTER_RATIO,
        help=(
            "Keep 0.65 for the formal experiment. "
            "NIR is still degraded."
        ),
    )

    p.add_argument(
        "--save-predictions",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore compatible completed metrics and rerun."
        ),
    )

    args = p.parse_args()

    if args.batch_size <= 0:
        p.error(
            "--batch-size must be > 0"
        )

    if args.num_workers != 0:
        p.error(
            "--num-workers must stay 0 because "
            "full-tile degraded RGBIR is cached "
            "and validation is tile-major."
        )

    if args.log_every <= 0:
        p.error(
            "--log-every must be > 0"
        )

    if args.confusion_chunk_rows <= 0:
        p.error(
            "--confusion-chunk-rows must be > 0"
        )

    if args.fog_chunk_rows <= 0:
        p.error(
            "--fog-chunk-rows must be > 0"
        )

    if not (
        0.0
        < args.nir_fog_scatter_ratio
        <= 1.0
    ):
        p.error(
            "--nir-fog-scatter-ratio "
            "must be in (0,1]"
        )

    return args


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    cfg = VARIANT_CONFIG[
        args.variant
    ]

    checkpoint_path = resolve(
        args.checkpoint
        if args.checkpoint is not None
        else Path(
            cfg["checkpoint"]
        )
    )

    output_root = resolve(
        args.output_root
        if args.output_root is not None
        else Path(
            cfg["output_root"]
        )
    )

    reference_path = resolve(
        args.reference_summary
        if args.reference_summary is not None
        else Path(
            cfg["reference_summary"]
        )
    )

    model_id, model_name = identity(
        args.variant
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Freeze the exact validation protocol next to the results.
    protocol_payload = (
        joint_validation_protocol()
    )
    protocol_payload["sha256"] = (
        joint_validation_protocol_sha256()
    )
    protocol_payload[
        "runtime_nir_fog_scatter_ratio"
    ] = float(
        args.nir_fog_scatter_ratio
    )

    save_json(
        output_root
        / "joint_validation_protocol.json",
        protocol_payload,
    )

    device = base.get_device(
        args.device
    )
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        base.unwrap_checkpoint_state_dict(
            checkpoint_obj
        )
    )

    validate_checkpoint_metadata(
        checkpoint_meta,
        variant=args.variant,
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

    checkpoint_epoch = (
        int(checkpoint_epoch)
        if checkpoint_epoch is not None
        else None
    )
    checkpoint_step = (
        int(checkpoint_step)
        if checkpoint_step is not None
        else None
    )

    training_protocol = checkpoint_meta[
        "protocol"
    ]

    trained_ratio = float(
        training_protocol[
            "joint_corruption_training"
        ][
            "fog"
        ][
            "nir_fog_scatter_ratio"
        ]
    )

    if abs(
        trained_ratio
        - float(
            args.nir_fog_scatter_ratio
        )
    ) > 1e-12:
        print(
            "[warning] validation NIR fog ratio "
            f"{args.nir_fog_scatter_ratio:.6f} "
            "differs from training metadata "
            f"{trained_ratio:.6f}. "
            "For the formal experiment keep them equal.",
            flush=True,
        )

    print(
        "=" * 132
    )
    print(
        f"{model_name} | "
        "JOINT RGB+NIR 13-CONDITION VALIDATION"
    )
    print(
        "=" * 132
    )
    print(
        f"variant             : {args.variant}"
    )
    print(
        f"model id            : {model_id}"
    )
    print(
        f"checkpoint          : {checkpoint_path}"
    )
    print(
        f"epoch               : "
        f"{checkpoint_epoch + 1 if checkpoint_epoch is not None else 'unknown'}"
    )
    print(
        f"global_step         : {checkpoint_step}"
    )
    print(
        f"device / AMP        : "
        f"{device} / {amp_enabled}"
    )
    print(
        f"batch size          : "
        f"{args.batch_size}"
    )
    print(
        "degradation scope   : "
        "full raw 6000x6000 RGB + NIR"
    )
    print(
        "fusion              : "
        + (
            f"fixed g={FIXED_NIR_STRENGTH:.3f}"
            if args.variant == "fixed"
            else "dynamic DARF; no Gate BCE"
        )
    )
    print(
        f"NIR fog ratio       : "
        f"{args.nir_fog_scatter_ratio:.3f}"
    )
    print(
        f"validation protocol : "
        f"{JOINT_VALIDATION_PROTOCOL_VERSION}"
    )
    print(
        f"protocol sha256     : "
        f"{joint_validation_protocol_sha256()}"
    )
    print(
        f"output              : {output_root}"
    )
    print(
        f"start local         : "
        f"{base.format_local_datetime(base.local_now())}"
    )
    print(
        f"{gpu_memory_text(device)}"
    )
    print(
        "=" * 132
    )

    model, model_meta = build_model(
        args.variant
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading returned "
            "incompatibilities."
        )

    model.to(
        device
    )
    model.eval()

    print(
        f"[model] strict load PASS | "
        f"parameters="
        f"{model_meta['parameters']['total']:,}",
        flush=True,
    )

    conditions = suite_conditions(
        args.suite
    )

    if not conditions:
        raise RuntimeError(
            "No validation conditions selected."
        )

    clean_probe = (
        PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )
    )

    if (
        len(
            clean_probe.tile_ids
        )
        != 6
        or len(
            clean_probe.window_coordinates
        )
        != 256
        or int(
            clean_probe.spec.tile_size
        )
        != 6000
        or int(
            clean_probe.spec.crop_size
        )
        != 512
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    print(
        "[dataset] frozen validation PASS | "
        "6 tiles | 6000x6000 | "
        "512 windows | 256 windows/tile",
        flush=True,
    )
    print(
        f"[dataset] NIR normalization | "
        f"mean={float(clean_probe.spec.nir_mean):.12f} | "
        f"std={float(clean_probe.spec.nir_std):.12f}",
        flush=True,
    )

    progress = JointValidationProgress(
        total_conditions=len(
            conditions
        ),
        tiles_per_condition=6,
        output_path=(
            output_root
            / "validation_progress.json"
        ),
        device=device,
    )

    results: List[
        Dict[str, Any]
    ] = []
    clean_result: Optional[
        Dict[str, Any]
    ] = None

    total_started = time.time()

    for condition_index, item in enumerate(
        conditions,
        start=1,
    ):
        condition = str(
            item["condition"]
        )
        relation = training_relation(
            regime=REGIME,
            family=str(
                item["family"]
            ),
        )

        out_dir = condition_output_dir(
            output_root,
            item,
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
            f"[Condition "
            f"{condition_index:02d}/"
            f"{len(conditions):02d}] "
            f"{condition} | "
            f"{relation}"
        )
        print(
            "-" * 132
        )

        existing = None
        if not args.force:
            existing = compatible_existing(
                metrics_path,
                model_id=model_id,
                variant=args.variant,
                condition=condition,
                checkpoint_path=(
                    checkpoint_path
                ),
                checkpoint_step=(
                    checkpoint_step
                ),
            )

        if existing is not None:
            result = existing

            progress.mark_skipped_condition(
                condition_index=(
                    condition_index
                ),
                condition=condition,
            )

            print(
                f"[resume] compatible metrics reused | "
                f"mIoU={float(result['miou']):.6f}",
                flush=True,
            )

        else:
            dataset = build_dataset(
                item=item,
                nir_fog_scatter_ratio=(
                    args.nir_fog_scatter_ratio
                ),
                fog_chunk_rows=(
                    args.fog_chunk_rows
                ),
            )

            if item["kind"] != "clean":
                joint_degradation_probe(
                    clean_dataset=(
                        clean_probe
                    ),
                    degraded_dataset=(
                        dataset
                    ),
                    condition=(
                        condition
                    ),
                )

            result = base.evaluate_condition(
                model=model,
                dataset=dataset,
                item=item,
                output_dir=out_dir,
                model_id=model_id,
                model_name=model_name,
                regime=REGIME,
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
                    args.batch_size
                ),
                num_workers=(
                    args.num_workers
                ),
                pin_memory=(
                    args.pin_memory
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
                condition_index=(
                    condition_index
                ),
                progress=progress,
            )

            add_joint_semantics(
                result,
                variant=args.variant,
                nir_fog_scatter_ratio=(
                    args.nir_fog_scatter_ratio
                ),
            )

            save_json(
                metrics_path,
                result,
            )

            base.clear_dataset_cache(
                dataset
            )
            del dataset

        # Existing result should already have semantics, but normalize again
        # so future minor metadata additions stay consistent.
        add_joint_semantics(
            result,
            variant=args.variant,
            nir_fog_scatter_ratio=(
                args.nir_fog_scatter_ratio
            ),
        )

        if condition == "Clean":
            clean_result = result

        results.append(
            result
        )

        print(
            f"[result] {condition:<26} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"{gpu_memory_text(device)}",
            flush=True,
        )

    if clean_result is None:
        # For a partial suite without Clean, use an already completed Clean
        # result only when it belongs to this same checkpoint/protocol.
        clean_path = (
            output_root
            / "clean_val"
            / "metrics.json"
        )
        clean_result = compatible_existing(
            clean_path,
            model_id=model_id,
            variant=args.variant,
            condition="Clean",
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_step=(
                checkpoint_step
            ),
        )

    if clean_result is None:
        if args.suite != "clean":
            raise RuntimeError(
                "Clean reference is required for "
                "clean-relative robustness metrics. "
                "Run --suite clean first or use --suite all."
            )

    clean_miou = (
        float(
            clean_result["miou"]
        )
        if clean_result is not None
        else None
    )

    if clean_miou is not None:
        for result in results:
            if result["condition"] == "Clean":
                result[
                    "clean_reference_miou"
                ] = clean_miou
                result["drop_miou"] = 0.0
                result["delta_miou"] = 0.0
                result[
                    "relative_drop_pct"
                ] = 0.0
                result["retention_pct"] = 100.0
            else:
                base.add_clean_relative_metrics(
                    result,
                    clean_miou=clean_miou,
                )

            add_joint_semantics(
                result,
                variant=args.variant,
                nir_fog_scatter_ratio=(
                    args.nir_fog_scatter_ratio
                ),
            )

            item_for_path = {
                "kind": (
                    "clean"
                    if result["condition"] == "Clean"
                    else "fog"
                    if result["family"] == "fog"
                    else "standard"
                ),
                "condition": result[
                    "condition"
                ],
            }

            save_json(
                condition_output_dir(
                    output_root,
                    item_for_path,
                )
                / "metrics.json",
                result,
            )

    rows = [
        summary_row(r)
        for r in results
    ]

    summary_payload: Dict[str, Any] = {
        "validator_version": (
            VALIDATOR_VERSION
        ),
        "joint_validation_protocol_version": (
            JOINT_VALIDATION_PROTOCOL_VERSION
        ),
        "joint_validation_protocol_sha256": (
            joint_validation_protocol_sha256()
        ),
        "model": model_id,
        "model_name": model_name,
        "variant": args.variant,
        "regime": REGIME,
        "backbone": "SegFormer-B2",
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "joint_degradation": True,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_global_step": (
            checkpoint_step
        ),
        "nir_fog_scatter_ratio": float(
            args.nir_fog_scatter_ratio
        ),
        "results": rows,
    }

    if (
        args.suite == "all"
        and len(results) == 13
    ):
        summary_payload[
            "aggregates"
        ] = aggregate_metrics(
            results
        )

    save_json(
        output_root
        / "all_conditions_summary.json",
        summary_payload,
    )

    summary_fields = [
        "model",
        "variant",
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
        summary_fields,
    )

    # Variant-specific diagnostic bundle.
    if args.variant == "fixed":
        residual_rows = [
            fixed_residual_row(r)
            for r in results
        ]

        save_json(
            output_root
            / "residual_statistics.json",
            {
                "model": model_id,
                "definition": (
                    "mean absolute NIR adapter "
                    "response by scale"
                ),
                "fixed_nir_strength": (
                    FIXED_NIR_STRENGTH
                ),
                "rows": residual_rows,
            },
        )

        residual_fields = [
            "condition",
            "family",
            "severity_level",
            "fixed_nir_strength",
        ]
        for s in range(
            1,
            NUM_SCALES + 1,
        ):
            residual_fields.extend(
                [
                    f"scale{s}_mean",
                    (
                        f"scale{s}_std_"
                        "across_tiles"
                    ),
                    (
                        f"scale{s}_"
                        "effective_mean"
                    ),
                ]
            )

        write_csv(
            output_root
            / "residual_statistics.csv",
            residual_rows,
            residual_fields,
        )

    else:
        gate_rows: List[
            Dict[str, Any]
        ] = []

        for result in results:
            gate_rows.extend(
                darf_gate_rows(
                    result
                )
            )

        clean_gate = {
            int(r["scale"]): float(
                r["g_nir_mean"]
            )
            for r in gate_rows
            if r["condition"] == "Clean"
        }

        for row in gate_rows:
            scale = int(
                row["scale"]
            )
            row[
                "delta_mean_vs_clean"
            ] = (
                float(
                    row["g_nir_mean"]
                )
                - clean_gate[scale]
                if scale in clean_gate
                else None
            )

        save_json(
            output_root
            / "gate_statistics.json",
            {
                "model": model_id,
                "gate_semantics": (
                    "g_NIR residual correction strength"
                ),
                "gate_supervision": False,
                "analysis_note": (
                    "No monotonic-severity behavior "
                    "is assumed or enforced."
                ),
                "rows": gate_rows,
            },
        )

        write_csv(
            output_root
            / "gate_statistics.csv",
            gate_rows,
            [
                "condition",
                "family",
                "severity_level",
                "severity_rank",
                "scale",
                "count",
                "g_nir_mean",
                "g_nir_std",
                "g_nir_min",
                "g_nir_max",
                "delta_mean_vs_clean",
            ],
        )

    write_reference_comparison(
        results=results,
        reference_path=(
            reference_path
        ),
        reference_name=str(
            cfg["reference_name"]
        ),
        output_root=output_root,
        current_model_id=model_id,
    )

    progress.finish(
        output_root=(
            output_root
        )
    )

    total_seconds = (
        time.time()
        - total_started
    )

    print()
    print(
        "=" * 132
    )
    print(
        "FINAL JOINT RGB+NIR VALIDATION SUMMARY"
    )
    print(
        "=" * 132
    )
    print(
        f"{'Condition':<28} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'RelDrop%':>10} "
        f"{'Retention%':>12}"
    )
    print(
        "-" * 132
    )

    for row in rows:
        print(
            f"{str(row['condition']):<28} "
            f"{float(row['miou']):>10.6f} "
            f"{float(row['drop_miou']):>10.6f} "
            f"{float(row['relative_drop_pct']):>10.3f} "
            f"{float(row['retention_pct']):>12.3f}"
        )

    print(
        "-" * 132
    )

    if (
        args.suite == "all"
        and len(results) == 13
    ):
        agg = aggregate_metrics(
            results
        )
        print(
            f"Clean mIoU             : "
            f"{agg['clean_miou']:.6f}"
        )
        print(
            f"12 degraded mean mIoU : "
            f"{agg['mean_degraded_miou_12_conditions']:.6f}"
        )
        print(
            f"4-family L3 mean mIoU : "
            f"{agg['mean_L3_miou_4_families']:.6f}"
        )

    print(
        f"elapsed                : "
        f"{base.format_duration(total_seconds)}"
    )
    print(
        f"{gpu_memory_text(device)}"
    )
    print(
        f"protocol               : "
        f"{output_root / 'joint_validation_protocol.json'}"
    )
    print(
        f"summary                : "
        f"{output_root / 'all_conditions_summary.csv'}"
    )

    if args.variant == "fixed":
        print(
            f"residual stats         : "
            f"{output_root / 'residual_statistics.csv'}"
        )
    else:
        print(
            f"gate stats             : "
            f"{output_root / 'gate_statistics.csv'}"
        )

    print(
        f"reference comparison   : "
        f"{output_root / 'comparison_vs_reference.csv'}"
    )
    print(
        f"progress               : "
        f"{output_root / 'validation_progress.json'}"
    )
    print(
        "=" * 132
    )


if __name__ == "__main__":
    main()
