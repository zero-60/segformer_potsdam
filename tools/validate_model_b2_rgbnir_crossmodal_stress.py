#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-modal stress validation for the Joint RGB+NIR Robust-4 study.

Scientific purpose
------------------
The formal Joint validation showed:
    M2' fixed g=0.5 < fixed g=1.0 ~= M3' DARF,
while the learned DARF gates were almost saturated at 1.

This validator tests the mechanism under conditions where RELATIVE modality
reliability changes. These are auxiliary stress tests, not replacements for
the formal 13-condition Joint RGB+NIR validation table.

Primary comparison:
    Fixed g=1.0  vs  M3' DARF

Stress axes
-----------
1) RGB clean / NIR degraded
   - NIR-only Gaussian noise L1/L2/L3
   - NIR-only Gaussian blur L1/L2/L3
   - NIR-only underexposure L1/L2/L3
   - NIR-only fog L1/L2/L3

2) Full NIR dropout
   - RGB stays clean
   - raw NIR is replaced by the frozen train-set NIR mean, so normalized NIR
     carries approximately zero information.

3) RGB/NIR severity mismatch
   For each Robust-4 family:
       a) RGB L1 / NIR L3  (NIR is relatively worse)
       b) RGB L3 / NIR L1  (RGB is relatively worse)

4) RGB-NIR misregistration
   - RGB stays clean
   - NIR is shifted on the full raw tile before crop/normalization
   - default integer displacement magnitudes: 2 / 8 / 16 pixels

Frozen metric protocol
----------------------
- 6 Potsdam validation tiles
- full raw 6000x6000 RGBIR tile transform before sliding-window crop
- 512x512 sliding windows
- frozen window coordinates / overlap
- overlap mean-logit fusion
- one full prediction per tile
- one GLOBAL confusion matrix across all six tiles
- global mIoU + per-class IoU + per-tile metrics
- deterministic transforms
- num_workers=0 so exactly one transformed full tile is cached at a time

Dependencies
------------
Place this file in tools/ beside:
    validate_model_b2_rgbnir_fixed1_joint_robust4.py

Examples
--------
Fixed g=1.0:
    python tools/validate_model_b2_rgbnir_crossmodal_stress.py --variant fixed1

DARF:
    python tools/validate_model_b2_rgbnir_crossmodal_stress.py --variant darf

Run only one axis:
    python tools/validate_model_b2_rgbnir_crossmodal_stress.py \
        --variant darf --suite mismatch
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgbnir_fixed1_joint_robust4 as joint_val

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset, require


# =============================================================================
# Constants / identities
# =============================================================================

VALIDATOR_VERSION = "1.0.0"
STRESS_PROTOCOL_VERSION = "Cross-Modal Stress Protocol v1"
STRESS_IMPLEMENTATION_REVISION = 1
STRESS_SEED = 20261004

REGIME = joint_val.REGIME
NUM_SCALES = joint_val.NUM_SCALES
NIR_FOG_SCATTER_RATIO = float(
    joint_val.DEFAULT_NIR_FOG_SCATTER_RATIO
)

DEFAULT_BATCH_SIZE = joint_val.DEFAULT_BATCH_SIZE
DEFAULT_NUM_WORKERS = 0
DEFAULT_LOG_EVERY = joint_val.DEFAULT_LOG_EVERY
DEFAULT_CONFUSION_CHUNK_ROWS = joint_val.DEFAULT_CONFUSION_CHUNK_ROWS
DEFAULT_FOG_CHUNK_ROWS = joint_val.DEFAULT_FOG_CHUNK_ROWS

MISREGISTRATION_PIXELS = {
    "L1": 2,
    "L2": 8,
    "L3": 16,
}

FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "underexposure",
    "fog",
)

LEVELS = ("L1", "L2", "L3")

VARIANT_CONFIG = {
    "fixed1": {
        "internal_variant": "fixed",
        "model_id": "B2_RGBNIR_FIXED1_JOINT_ROBUST4",
        "model_name": "SegFormer-B2 RGB+NIR Fixed g=1.0 Joint Robust-4",
        "checkpoint": (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_rgbnir_fixed1_joint_robust4"
            / "checkpoints"
            / "final.pt"
        ),
        "output_root": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "b2_rgbnir_fixed1_joint_robust4"
            / "crossmodal_stress"
        ),
    },
    "darf": {
        "internal_variant": "darf",
        "model_id": joint_val.train_joint.DARF_MODEL_ID,
        "model_name": joint_val.train_joint.DARF_MODEL_NAME,
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
            / "crossmodal_stress"
        ),
    },
}


# =============================================================================
# Generic helpers
# =============================================================================

def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def protocol_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            stress_protocol()
        )
    ).hexdigest()


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


def deterministic_seed(*parts: Any) -> int:
    payload = (
        STRESS_PROTOCOL_VERSION
        + "|"
        + str(STRESS_SEED)
        + "|"
        + "|".join(
            str(x)
            for x in parts
        )
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
        signed=False,
    )


def family_to_rgb_protocol_name(family: str) -> str:
    if family == "underexposure":
        return "rgb_underexposure"
    return family


def class_column_name(name: str) -> str:
    chars = [
        ch if ch.isalnum() else "_"
        for ch in name.lower()
    ]
    return "iou_" + "_".join(
        part
        for part in "".join(chars).split("_")
        if part
    )


# =============================================================================
# Frozen stress protocol
# =============================================================================

def stress_protocol() -> Dict[str, Any]:
    nir_only: Dict[str, Any] = {}

    for family in FAMILIES:
        nir_only[family] = {}
        for level in LEVELS:
            if family == "fog":
                spec = joint_val.fog_base.fog_spec(level)
            else:
                spec = joint_val.rgb_protocol.condition_spec(
                    family_to_rgb_protocol_name(family),
                    level,
                )
            nir_only[family][level] = spec

    mismatch = {
        family: [
            {
                "role": "nir_worse",
                "rgb_level": "L1",
                "nir_level": "L3",
            },
            {
                "role": "rgb_worse",
                "rgb_level": "L3",
                "nir_level": "L1",
            },
        ]
        for family in FAMILIES
    }

    return {
        "protocol_version": STRESS_PROTOCOL_VERSION,
        "implementation_revision": STRESS_IMPLEMENTATION_REVISION,
        "seed": STRESS_SEED,
        "role": (
            "auxiliary mechanism stress test; not the formal Joint 13-condition "
            "main validation table"
        ),
        "metric_protocol": {
            "validation_tiles": 6,
            "tile_size": 6000,
            "crop_size": 512,
            "windows_per_tile": 256,
            "fusion": "overlap mean-logit",
            "metric_scope": (
                "one GLOBAL confusion matrix across all six validation tiles"
            ),
            "transform_scope": (
                "full raw RGBIR tile before crop and modality-specific normalization"
            ),
        },
        "axes": {
            "nir_only": {
                "rgb": "clean",
                "nir": "degraded",
                "families": nir_only,
                "hypothesis": (
                    "g_NIR should decline as NIR reliability degrades; DARF should "
                    "lose less performance than fixed g=1.0"
                ),
            },
            "nir_dropout": {
                "rgb": "clean",
                "nir": (
                    "full modality replaced by frozen train-set NIR mean in raw "
                    "space; normalized NIR approximately zero-information"
                ),
                "hypothesis": "g_NIR should fall strongly relative to Clean",
            },
            "severity_mismatch": {
                "pairs": mismatch,
                "noise": (
                    "independent deterministic RGB/NIR noise realizations with "
                    "different sigma levels"
                ),
                "blur": "different sigma levels",
                "underexposure": "different alpha levels",
                "fog": (
                    "same deterministic low-frequency spatial field; RGB and NIR "
                    "use different severity levels; NIR retains the 0.65 spectral "
                    "attenuation rule"
                ),
                "hypothesis": (
                    "g_NIR(rgb L3, nir L1) should exceed "
                    "g_NIR(rgb L1, nir L3)"
                ),
            },
            "misregistration": {
                "rgb": "clean",
                "nir": "integer translation; no interpolation blur",
                "pixels": dict(MISREGISTRATION_PIXELS),
                "border_fill": "frozen train-set raw NIR mean",
                "direction": (
                    "deterministic per tile, one of four cardinal directions"
                ),
                "hypothesis": (
                    "g_NIR should decline as displacement increases"
                ),
            },
        },
        "nir_fog_scatter_ratio": NIR_FOG_SCATTER_RATIO,
        "formal_joint_protocol_reference": {
            "protocol_version": joint_val.JOINT_VALIDATION_PROTOCOL_VERSION,
            "protocol_sha256": joint_val.joint_validation_protocol_sha256(),
        },
    }


def stress_conditions(suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "kind": "clean",
            "stress_axis": "clean",
            "condition": "Clean",
            "family": "clean",
            "corruption": None,
            "severity_level": None,
            "severity_rank": 0,
            "rgb_state": "clean",
            "nir_state": "clean",
        }
    ]

    if suite in ("all", "nir_only"):
        for family in FAMILIES:
            for level in LEVELS:
                rows.append(
                    {
                        "kind": "nir_only",
                        "stress_axis": "nir_only",
                        "condition": f"nir_only_{family}_{level}",
                        "family": family,
                        "corruption": family,
                        "severity_level": level,
                        "severity_rank": int(level[1:]),
                        "rgb_state": "clean",
                        "nir_state": f"{family}_{level}",
                        "nir_level": level,
                    }
                )

    if suite in ("all", "dropout"):
        rows.append(
            {
                "kind": "nir_dropout",
                "stress_axis": "nir_dropout",
                "condition": "nir_dropout_full",
                "family": "nir_dropout",
                "corruption": "nir_dropout",
                "severity_level": "FULL",
                "severity_rank": 4,
                "rgb_state": "clean",
                "nir_state": "dropout_full",
            }
        )

    if suite in ("all", "mismatch"):
        for family in FAMILIES:
            for role, rgb_level, nir_level in (
                ("nir_worse", "L1", "L3"),
                ("rgb_worse", "L3", "L1"),
            ):
                rows.append(
                    {
                        "kind": "severity_mismatch",
                        "stress_axis": "severity_mismatch",
                        "condition": (
                            f"mismatch_{family}_rgb{rgb_level}_nir{nir_level}"
                        ),
                        "family": family,
                        "corruption": family,
                        "severity_level": f"RGB{rgb_level}_NIR{nir_level}",
                        "severity_rank": None,
                        "rgb_state": f"{family}_{rgb_level}",
                        "nir_state": f"{family}_{nir_level}",
                        "mismatch_role": role,
                        "rgb_level": rgb_level,
                        "nir_level": nir_level,
                    }
                )

    if suite in ("all", "misregistration"):
        for level in LEVELS:
            rows.append(
                {
                    "kind": "misregistration",
                    "stress_axis": "misregistration",
                    "condition": f"misregistration_{level}",
                    "family": "misregistration",
                    "corruption": "misregistration",
                    "severity_level": level,
                    "severity_rank": int(level[1:]),
                    "rgb_state": "clean",
                    "nir_state": f"shift_{MISREGISTRATION_PIXELS[level]}px",
                    "shift_pixels": int(MISREGISTRATION_PIXELS[level]),
                }
            )

    if suite == "clean":
        return rows[:1]

    return rows


# =============================================================================
# Raw full-tile transforms
# =============================================================================

def nir_only_degradation(
    nir: np.ndarray,
    *,
    tile_id: str,
    family: str,
    level: str,
    fog_chunk_rows: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if family == "gaussian_noise":
        spec = joint_val.rgb_protocol.condition_spec(
            "gaussian_noise",
            level,
        )
        sigma_255 = float(spec["sigma_255"])
        out = joint_val._gaussian_noise_nir_uint8(
            nir,
            sigma_255=sigma_255,
            seed=deterministic_seed(
                tile_id,
                "nir_only",
                family,
                level,
            ),
        )
        return out, {"sigma_255": sigma_255}

    if family == "gaussian_blur":
        spec = joint_val.rgb_protocol.condition_spec(
            "gaussian_blur",
            level,
        )
        sigma = float(spec["sigma"])
        return (
            joint_val._gaussian_blur_nir_uint8(
                nir,
                sigma=sigma,
            ),
            {"sigma": sigma},
        )

    if family == "underexposure":
        spec = joint_val.rgb_protocol.condition_spec(
            "rgb_underexposure",
            level,
        )
        alpha = float(spec["alpha"])
        return (
            joint_val._underexposure_nir_uint8(
                nir,
                alpha=alpha,
            ),
            {"alpha": alpha},
        )

    if family == "fog":
        out, diag = joint_val._joint_fog_nir_uint8(
            nir,
            tile_id=tile_id,
            level=level,
            nir_fog_scatter_ratio=NIR_FOG_SCATTER_RATIO,
            chunk_rows=fog_chunk_rows,
        )
        return (
            out,
            {
                **diag,
                "nir_fog_scatter_ratio": NIR_FOG_SCATTER_RATIO,
            },
        )

    raise ValueError(f"Unsupported NIR-only family: {family}")


def _apply_fog_raw(
    image: np.ndarray,
    transmission: np.ndarray,
    *,
    atmosphere_255: float,
) -> np.ndarray:
    src = image.astype(np.float32, copy=False)
    t = transmission.astype(np.float32, copy=False)

    if image.ndim == 3:
        t = t[..., None]

    out = (
        src * t
        + np.float32(atmosphere_255)
        * (np.float32(1.0) - t)
    )

    np.clip(out, 0.0, 255.0, out=out)
    np.rint(out, out=out)
    return np.ascontiguousarray(out.astype(np.uint8))


def mismatch_fog(
    rgb: np.ndarray,
    nir: np.ndarray,
    *,
    tile_id: str,
    rgb_level: str,
    nir_level: str,
    fog_chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Same spatial field for both modalities, but different severity levels.
    L2 is used only to freeze the field seed; level-specific transmission
    parameters are still taken from rgb_level / nir_level.
    """
    h, w = nir.shape
    params = joint_val._fog_field_params(tile_id, "L2")
    atmosphere = float(
        joint_val.fog_base.ATMOSPHERIC_LIGHT_255[0]
    )

    rgb_out = np.empty_like(rgb)
    nir_out = np.empty_like(nir)

    t_rgb_sum = 0.0
    t_nir_sum = 0.0
    pixels = 0

    for y0 in range(0, h, int(fog_chunk_rows)):
        y1 = min(h, y0 + int(fog_chunk_rows))

        t_rgb = joint_val._fog_t_rgb_chunk(
            y0=y0,
            y1=y1,
            h=h,
            w=w,
            level=rgb_level,
            params=params,
        )

        nir_visible_equivalent = joint_val._fog_t_rgb_chunk(
            y0=y0,
            y1=y1,
            h=h,
            w=w,
            level=nir_level,
            params=params,
        )

        t_nir = (
            np.float32(1.0)
            - np.float32(NIR_FOG_SCATTER_RATIO)
            * (
                np.float32(1.0)
                - nir_visible_equivalent
            )
        )
        np.clip(
            t_nir,
            np.float32(0.20),
            np.float32(0.995),
            out=t_nir,
        )

        rgb_out[y0:y1] = _apply_fog_raw(
            rgb[y0:y1],
            t_rgb,
            atmosphere_255=atmosphere,
        )
        nir_out[y0:y1] = _apply_fog_raw(
            nir[y0:y1],
            t_nir,
            atmosphere_255=atmosphere,
        )

        t_rgb_sum += float(t_rgb.sum(dtype=np.float64))
        t_nir_sum += float(t_nir.sum(dtype=np.float64))
        pixels += int(t_rgb.size)

    return (
        np.ascontiguousarray(rgb_out),
        np.ascontiguousarray(nir_out),
        {
            "mean_t_rgb": t_rgb_sum / max(pixels, 1),
            "mean_t_nir": t_nir_sum / max(pixels, 1),
            "rgb_level": rgb_level,
            "nir_level": nir_level,
            "nir_fog_scatter_ratio": NIR_FOG_SCATTER_RATIO,
            "shared_field_seed_level": "L2",
        },
    )


def severity_mismatch(
    rgb: np.ndarray,
    nir: np.ndarray,
    *,
    tile_id: str,
    family: str,
    rgb_level: str,
    nir_level: str,
    fog_chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if family == "fog":
        return mismatch_fog(
            rgb,
            nir,
            tile_id=tile_id,
            rgb_level=rgb_level,
            nir_level=nir_level,
            fog_chunk_rows=fog_chunk_rows,
        )

    rgb_out = joint_val.rgb_protocol.apply_rgb_degradation_full_tile(
        rgb,
        tile_id=tile_id,
        corruption=family_to_rgb_protocol_name(family),
        level=rgb_level,
    )

    nir_out, nir_diag = nir_only_degradation(
        nir,
        tile_id=tile_id,
        family=family,
        level=nir_level,
        fog_chunk_rows=fog_chunk_rows,
    )

    return (
        rgb_out,
        nir_out,
        {
            "rgb_level": rgb_level,
            "nir_level": nir_level,
            "nir_parameters": nir_diag,
        },
    )


def integer_shift_direction(tile_id: str) -> Tuple[int, int]:
    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )
    index = deterministic_seed(
        tile_id,
        "misregistration_direction",
    ) % len(directions)
    return directions[int(index)]


def integer_shift_nir(
    nir: np.ndarray,
    *,
    tile_id: str,
    pixels: int,
    fill_value: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if pixels <= 0:
        raise ValueError("pixels must be > 0")

    unit_dx, unit_dy = integer_shift_direction(tile_id)
    dx = int(unit_dx * pixels)
    dy = int(unit_dy * pixels)

    h, w = nir.shape
    out = np.full(
        (h, w),
        int(fill_value),
        dtype=np.uint8,
    )

    if dx >= 0:
        src_x0, src_x1 = 0, w - dx
        dst_x0, dst_x1 = dx, w
    else:
        src_x0, src_x1 = -dx, w
        dst_x0, dst_x1 = 0, w + dx

    if dy >= 0:
        src_y0, src_y1 = 0, h - dy
        dst_y0, dst_y1 = dy, h
    else:
        src_y0, src_y1 = -dy, h
        dst_y0, dst_y1 = 0, h + dy

    out[
        dst_y0:dst_y1,
        dst_x0:dst_x1,
    ] = nir[
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    return (
        np.ascontiguousarray(out),
        {
            "shift_pixels": pixels,
            "dx": dx,
            "dy": dy,
            "fill_value_uint8": int(fill_value),
            "interpolation": "none_integer_copy",
        },
    )


def apply_stress_full_tile(
    rgbir: np.ndarray,
    *,
    tile_id: str,
    item: Mapping[str, Any],
    nir_mean: float,
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

    kind = str(item["kind"])
    rgb = rgbir[..., :3]
    nir = rgbir[..., 3]

    diagnostics: Dict[str, Any] = {
        "tile_id": tile_id,
        "condition": str(item["condition"]),
        "stress_axis": str(item["stress_axis"]),
    }

    rgb_out = np.ascontiguousarray(rgb)
    nir_out = np.ascontiguousarray(nir)

    if kind == "nir_only":
        nir_out, diag = nir_only_degradation(
            nir,
            tile_id=tile_id,
            family=str(item["family"]),
            level=str(item["nir_level"]),
            fog_chunk_rows=fog_chunk_rows,
        )
        diagnostics.update(diag)

    elif kind == "nir_dropout":
        fill_uint8 = int(
            np.clip(
                np.rint(float(nir_mean) * 255.0),
                0,
                255,
            )
        )
        nir_out = np.full_like(nir, fill_uint8)
        diagnostics.update(
            {
                "dropout": "full_modality",
                "fill_value_uint8": fill_uint8,
                "fill_value_definition": "frozen train-set NIR mean",
            }
        )

    elif kind == "severity_mismatch":
        rgb_out, nir_out, diag = severity_mismatch(
            rgb,
            nir,
            tile_id=tile_id,
            family=str(item["family"]),
            rgb_level=str(item["rgb_level"]),
            nir_level=str(item["nir_level"]),
            fog_chunk_rows=fog_chunk_rows,
        )
        diagnostics.update(diag)
        diagnostics["mismatch_role"] = str(
            item["mismatch_role"]
        )

    elif kind == "misregistration":
        fill_uint8 = int(
            np.clip(
                np.rint(float(nir_mean) * 255.0),
                0,
                255,
            )
        )
        nir_out, diag = integer_shift_nir(
            nir,
            tile_id=tile_id,
            pixels=int(item["shift_pixels"]),
            fill_value=fill_uint8,
        )
        diagnostics.update(diag)

    else:
        raise ValueError(f"Unsupported stress kind: {kind}")

    out = np.empty_like(rgbir)
    out[..., :3] = rgb_out
    out[..., 3] = nir_out

    return np.ascontiguousarray(out), diagnostics


# =============================================================================
# Dataset
# =============================================================================

class CrossModalStressPotsdamDataset(
    PotsdamSlidingWindowDataset
):
    """One deterministic full-tile stress transform cached at a time."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        split: str,
        item: Mapping[str, Any],
        fog_chunk_rows: int,
    ):
        super().__init__(
            project_root=project_root,
            split=split,
        )
        self.item = dict(item)
        self.condition = str(item["condition"])
        self.fog_chunk_rows = int(fog_chunk_rows)

        self._stress_cache_tile_id: Optional[str] = None
        self._stress_cache_rgbir: Optional[np.ndarray] = None
        self._stress_cache_diagnostics: Optional[
            Dict[str, Any]
        ] = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_stress_cache_tile_id"] = None
        state["_stress_cache_rgbir"] = None
        state["_stress_cache_diagnostics"] = None
        return state

    def clear_degradation_cache(self) -> None:
        self._stress_cache_tile_id = None
        self._stress_cache_rgbir = None
        self._stress_cache_diagnostics = None

    def _get_stressed_rgbir(
        self,
        tile_id: str,
        rgbir: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if (
            self._stress_cache_tile_id == tile_id
            and self._stress_cache_rgbir is not None
            and self._stress_cache_diagnostics is not None
        ):
            return (
                self._stress_cache_rgbir,
                self._stress_cache_diagnostics,
            )

        stressed, diagnostics = apply_stress_full_tile(
            rgbir,
            tile_id=tile_id,
            item=self.item,
            nir_mean=float(self.spec.nir_mean),
            fog_chunk_rows=self.fog_chunk_rows,
        )

        require(
            stressed.shape
            == (
                self.spec.tile_size,
                self.spec.tile_size,
                4,
            ),
            f"{tile_id}: stress RGBIR shape invalid: {stressed.shape}",
        )
        require(
            stressed.dtype == np.uint8,
            f"{tile_id}: stress RGBIR dtype invalid: {stressed.dtype}",
        )

        self._stress_cache_tile_id = tile_id
        self._stress_cache_rgbir = stressed
        self._stress_cache_diagnostics = diagnostics

        return stressed, diagnostics

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id, window_index, x, y = self.index_to_tile_window(index)

        raw = self._read_rgbir(tile_id)
        stressed, diagnostics = self._get_stressed_rgbir(
            tile_id,
            raw,
        )

        h = int(self.spec.crop_size)
        w = int(self.spec.crop_size)

        crop = stressed[
            y:y + h,
            x:x + w,
            :,
        ]

        require(
            crop.shape == (h, w, 4),
            f"{tile_id}: bad stress crop {crop.shape}",
        )

        rgb_t, nir_t = self.spec.normalize_rgb_nir(
            crop,
            context=(
                f"{self.split_name} {tile_id} {self.condition} "
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
            "height": h,
            "width": w,
            "degradation_condition": self.condition,
            "stress_axis": str(self.item["stress_axis"]),
            "rgb_state": str(self.item["rgb_state"]),
            "nir_state": str(self.item["nir_state"]),
            "stress_diagnostics_json": json.dumps(
                diagnostics,
                sort_keys=True,
            ),
        }


def build_dataset(
    *,
    item: Mapping[str, Any],
    fog_chunk_rows: int,
) -> PotsdamSlidingWindowDataset:
    if item["kind"] == "clean":
        return PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )

    return CrossModalStressPotsdamDataset(
        PROJECT_ROOT,
        split="val",
        item=item,
        fog_chunk_rows=fog_chunk_rows,
    )


def stress_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    stressed_dataset: PotsdamSlidingWindowDataset,
    item: Mapping[str, Any],
) -> None:
    if item["kind"] == "clean":
        return

    clean = clean_dataset[0]
    stressed = stressed_dataset[0]

    for key in (
        "tile_id",
        "window_index",
        "x",
        "y",
        "height",
        "width",
    ):
        if clean[key] != stressed[key]:
            raise RuntimeError(
                f"{item['condition']}: window metadata changed at {key}"
            )

    rgb_changed = not torch.equal(
        clean["rgb"],
        stressed["rgb"],
    )
    nir_changed = not torch.equal(
        clean["nir"],
        stressed["nir"],
    )

    expected_rgb_changed = (
        item["kind"] == "severity_mismatch"
    )

    if rgb_changed != expected_rgb_changed:
        raise RuntimeError(
            f"{item['condition']}: RGB changed={rgb_changed}, "
            f"expected={expected_rgb_changed}"
        )

    if not nir_changed:
        raise RuntimeError(
            f"{item['condition']}: NIR did not change."
        )

    print(
        f"[stress probe] {item['condition']}: PASS | "
        f"RGB_changed={rgb_changed} | NIR_changed={nir_changed}",
        flush=True,
    )


# =============================================================================
# Model/checkpoint
# =============================================================================

def identity(variant: str) -> Tuple[str, str]:
    cfg = VARIANT_CONFIG[variant]
    return (
        str(cfg["model_id"]),
        str(cfg["model_name"]),
    )


def build_model(variant: str):
    internal = str(
        VARIANT_CONFIG[variant]["internal_variant"]
    )
    return joint_val.build_model(internal)


def validate_checkpoint(
    meta: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    internal = str(
        VARIANT_CONFIG[variant]["internal_variant"]
    )

    joint_val.validate_checkpoint_metadata(
        meta,
        variant=internal,
    )

    expected_model_id = str(
        VARIANT_CONFIG[variant]["model_id"]
    )

    if meta.get("model_id") != expected_model_id:
        raise RuntimeError(
            "Stress validator model_id mismatch: "
            f"{meta.get('model_id')!r} != {expected_model_id!r}"
        )


# =============================================================================
# Result semantics / diagnostics
# =============================================================================

def stress_training_relation(
    *,
    regime: str,
    family: str,
) -> str:
    if family == "clean":
        return "clean_reference"
    return "cross_modal_stress_outside_formal_joint_distribution"


# Imported validator already patches base.infer_one_tile to collect fixed
# residual diagnostics / DARF gates. Only the training-relation label changes.
joint_val.base.training_relation = stress_training_relation


def output_dir_for_item(
    output_root: Path,
    item: Mapping[str, Any],
) -> Path:
    if item["kind"] == "clean":
        return output_root / "clean_reference"

    return (
        output_root
        / str(item["stress_axis"])
        / str(item["condition"])
    )


def add_stress_semantics(
    result: Dict[str, Any],
    *,
    item: Mapping[str, Any],
    public_variant: str,
) -> None:
    result["validator_version"] = VALIDATOR_VERSION
    result["crossmodal_stress"] = True
    result["crossmodal_stress_protocol_version"] = (
        STRESS_PROTOCOL_VERSION
    )
    result["crossmodal_stress_implementation_revision"] = (
        STRESS_IMPLEMENTATION_REVISION
    )
    result["crossmodal_stress_protocol_sha256"] = protocol_sha256()

    result["public_variant"] = public_variant
    result["stress_axis"] = str(item["stress_axis"])
    result["stress_kind"] = str(item["kind"])
    result["rgb_state"] = str(item["rgb_state"])
    result["nir_state"] = str(item["nir_state"])
    result["formal_joint_main_table"] = False
    result["scientific_role"] = "auxiliary mechanism stress test"

    for key in (
        "mismatch_role",
        "rgb_level",
        "nir_level",
        "shift_pixels",
    ):
        if key in item:
            result[key] = item[key]

    if public_variant == "fixed1":
        result["fusion"] = (
            "RGB-anchored fixed NIR residual fusion"
        )
        result["fusion_rule"] = (
            "F_i = F_RGB_i + 1.0 * Adapter_i(F_NIR_i)"
        )
        result["fixed_nir_strength"] = 1.0
        result["quality_gate"] = False
    else:
        result["fusion"] = (
            "dynamic DARF RGB-anchored NIR residual fusion"
        )
        result["fusion_rule"] = (
            "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
        )
        result["quality_gate"] = True
        result["gate_supervision"] = False


def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": result["model"],
        "public_variant": result["public_variant"],
        "condition": result["condition"],
        "stress_axis": result["stress_axis"],
        "family": result["family"],
        "severity_level": result.get("severity_level"),
        "mismatch_role": result.get("mismatch_role"),
        "rgb_level": result.get("rgb_level"),
        "nir_level": result.get("nir_level"),
        "shift_pixels": result.get("shift_pixels"),
        "rgb_state": result["rgb_state"],
        "nir_state": result["nir_state"],
        "miou": float(result["miou"]),
        "pixel_accuracy": float(result["pixel_accuracy"]),
        "mean_class_accuracy": float(
            result["mean_class_accuracy"]
        ),
        "clean_reference_miou": result.get(
            "clean_reference_miou"
        ),
        "drop_miou": result.get("drop_miou"),
        "relative_drop_pct": result.get(
            "relative_drop_pct"
        ),
        "retention_pct": result.get("retention_pct"),
        "validation_seconds": float(
            result["validation_seconds"]
        ),
    }

    for class_item in result["per_class"]:
        row[
            class_column_name(
                str(class_item["class_name"])
            )
        ] = float(class_item["iou"])

    return row


def gate_rows_for_result(
    result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows = joint_val.darf_gate_rows(result)
    enriched = []

    for row in rows:
        x = dict(row)
        x["stress_axis"] = result["stress_axis"]
        x["mismatch_role"] = result.get("mismatch_role")
        x["rgb_level"] = result.get("rgb_level")
        x["nir_level"] = result.get("nir_level")
        x["shift_pixels"] = result.get("shift_pixels")
        enriched.append(x)

    return enriched


def fixed_residual_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    row = joint_val.fixed_residual_row(result)
    row["stress_axis"] = result["stress_axis"]
    row["mismatch_role"] = result.get("mismatch_role")
    row["rgb_level"] = result.get("rgb_level")
    row["nir_level"] = result.get("nir_level")
    row["shift_pixels"] = result.get("shift_pixels")
    return row


def compatible_existing(
    path: Path,
    *,
    model_id: str,
    public_variant: str,
    condition: str,
    checkpoint_step: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None

    checks = [
        obj.get("model") == model_id,
        obj.get("public_variant") == public_variant,
        obj.get("condition") == condition,
        obj.get("crossmodal_stress") is True,
        obj.get("crossmodal_stress_protocol_sha256")
        == protocol_sha256(),
    ]

    if checkpoint_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(checkpoint_step)
        )

    return obj if all(checks) else None


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cross-modal mechanism stress validation for "
            "Fixed g=1.0 vs M3' DARF."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    p.add_argument(
        "--variant",
        choices=("fixed1", "darf"),
        required=True,
    )

    p.add_argument(
        "--suite",
        choices=(
            "all",
            "clean",
            "nir_only",
            "dropout",
            "mismatch",
            "misregistration",
        ),
        default="all",
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
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
        "--save-predictions",
        action="store_true",
    )
    p.add_argument(
        "--force",
        action="store_true",
    )

    x = p.parse_args()

    if x.batch_size <= 0:
        p.error("--batch-size must be > 0")

    if x.num_workers != 0:
        p.error(
            "--num-workers must remain 0 for "
            "full-tile transformed cache correctness."
        )

    if x.log_every <= 0:
        p.error("--log-every must be > 0")

    if x.confusion_chunk_rows <= 0:
        p.error("--confusion-chunk-rows must be > 0")

    if x.fog_chunk_rows <= 0:
        p.error("--fog-chunk-rows must be > 0")

    return x


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    x = parse_args()
    cfg = VARIANT_CONFIG[x.variant]

    checkpoint_path = resolve(
        x.checkpoint
        if x.checkpoint is not None
        else Path(cfg["checkpoint"])
    )

    output_root = resolve(
        x.output_root
        if x.output_root is not None
        else Path(cfg["output_root"])
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = stress_protocol()
    protocol["sha256"] = protocol_sha256()

    save_json(
        output_root
        / "crossmodal_stress_protocol.json",
        protocol,
    )

    device = joint_val.base.get_device(x.device)
    amp_enabled = (
        device.type == "cuda"
        and not x.no_amp
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        joint_val.base.unwrap_checkpoint_state_dict(
            checkpoint_obj
        )
    )

    validate_checkpoint(
        checkpoint_meta,
        variant=x.variant,
    )

    checkpoint_epoch = checkpoint_meta.get("epoch")
    checkpoint_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get("step"),
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

    model_id, model_name = identity(x.variant)
    model, model_meta = build_model(x.variant)

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint load returned incompatibilities."
        )

    model.to(device)
    model.eval()

    conditions = stress_conditions(x.suite)

    clean_probe = PotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
    )

    if (
        len(clean_probe.tile_ids) != 6
        or len(clean_probe.window_coordinates) != 256
        or int(clean_probe.spec.tile_size) != 6000
        or int(clean_probe.spec.crop_size) != 512
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    print("=" * 136)
    print("CROSS-MODAL STRESS VALIDATION")
    print("=" * 136)
    print(f"variant             : {x.variant}")
    print(f"model               : {model_id}")
    print(f"checkpoint          : {checkpoint_path}")
    print(f"checkpoint step     : {checkpoint_step}")
    print(f"suite               : {x.suite}")
    print(f"conditions          : {len(conditions)}")
    print(f"device / AMP        : {device} / {amp_enabled}")
    print(f"protocol            : {STRESS_PROTOCOL_VERSION}")
    print(f"protocol sha256     : {protocol_sha256()}")
    print(
        "metric protocol     : 6 tiles | 6000x6000 full-tile transform | "
        "512 windows | overlap mean-logit | GLOBAL confusion"
    )
    print(f"output              : {output_root}")
    print(joint_val.gpu_memory_text(device))
    print("=" * 136)

    progress = joint_val.JointValidationProgress(
        total_conditions=len(conditions),
        tiles_per_condition=6,
        output_path=(
            output_root
            / "validation_progress.json"
        ),
        device=device,
    )

    results: List[Dict[str, Any]] = []
    clean_result: Optional[Dict[str, Any]] = None
    started = time.time()

    for condition_index, item in enumerate(
        conditions,
        start=1,
    ):
        condition = str(item["condition"])

        out_dir = output_dir_for_item(
            output_root,
            item,
        )
        metrics_path = out_dir / "metrics.json"

        print()
        print("-" * 136)
        print(
            f"[Condition {condition_index:02d}/{len(conditions):02d}] "
            f"{condition} | axis={item['stress_axis']} | "
            f"RGB={item['rgb_state']} | NIR={item['nir_state']}"
        )
        print("-" * 136)

        existing = None
        if not x.force:
            existing = compatible_existing(
                metrics_path,
                model_id=model_id,
                public_variant=x.variant,
                condition=condition,
                checkpoint_step=checkpoint_step,
            )

        if existing is not None:
            result = existing
            progress.mark_skipped_condition(
                condition_index=condition_index,
                condition=condition,
            )
            print(
                f"[resume] reused | "
                f"mIoU={float(result['miou']):.6f}",
                flush=True,
            )

        else:
            dataset = build_dataset(
                item=item,
                fog_chunk_rows=x.fog_chunk_rows,
            )

            stress_probe(
                clean_dataset=clean_probe,
                stressed_dataset=dataset,
                item=item,
            )

            result = joint_val.base.evaluate_condition(
                model=model,
                dataset=dataset,
                item=item,
                output_dir=out_dir,
                model_id=model_id,
                model_name=model_name,
                regime=REGIME,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_global_step=checkpoint_step,
                batch_size=x.batch_size,
                num_workers=x.num_workers,
                pin_memory=x.pin_memory,
                device=device,
                amp_enabled=amp_enabled,
                log_every=x.log_every,
                confusion_chunk_rows=x.confusion_chunk_rows,
                save_predictions=x.save_predictions,
                condition_index=condition_index,
                progress=progress,
            )

            add_stress_semantics(
                result,
                item=item,
                public_variant=x.variant,
            )

            save_json(
                metrics_path,
                result,
            )

            joint_val.base.clear_dataset_cache(
                dataset
            )
            del dataset

        if condition == "Clean":
            clean_result = result

        results.append(result)

        print(
            f"[result] {condition:<44} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"{joint_val.gpu_memory_text(device)}",
            flush=True,
        )

    if clean_result is None:
        clean_result = compatible_existing(
            output_root
            / "clean_reference"
            / "metrics.json",
            model_id=model_id,
            public_variant=x.variant,
            condition="Clean",
            checkpoint_step=checkpoint_step,
        )

    if clean_result is None:
        raise RuntimeError(
            "A compatible Clean stress reference is required."
        )

    clean_miou = float(clean_result["miou"])

    condition_lookup = {
        str(item["condition"]): item
        for item in conditions
    }

    for result in results:
        if result["condition"] == "Clean":
            result["clean_reference_miou"] = clean_miou
            result["drop_miou"] = 0.0
            result["delta_miou"] = 0.0
            result["relative_drop_pct"] = 0.0
            result["retention_pct"] = 100.0
        else:
            joint_val.base.add_clean_relative_metrics(
                result,
                clean_miou=clean_miou,
            )

        item = condition_lookup[
            str(result["condition"])
        ]

        add_stress_semantics(
            result,
            item=item,
            public_variant=x.variant,
        )

        save_json(
            output_dir_for_item(
                output_root,
                item,
            )
            / "metrics.json",
            result,
        )

    rows = [
        summary_row(r)
        for r in results
    ]

    class_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("iou_")
        }
    )

    summary_fields = [
        "model",
        "public_variant",
        "condition",
        "stress_axis",
        "family",
        "severity_level",
        "mismatch_role",
        "rgb_level",
        "nir_level",
        "shift_pixels",
        "rgb_state",
        "nir_state",
        "miou",
        "clean_reference_miou",
        "drop_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        *class_fields,
        "validation_seconds",
    ]

    nonclean = [
        float(r["miou"])
        for r in results
        if r["condition"] != "Clean"
    ]

    summary_payload = {
        "validator_version": VALIDATOR_VERSION,
        "protocol_version": STRESS_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(),
        "model": model_id,
        "model_name": model_name,
        "public_variant": x.variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint_step,
        "suite": x.suite,
        "clean_miou": clean_miou,
        "mean_nonclean_stress_miou": (
            float(
                np.mean(
                    np.asarray(
                        nonclean,
                        dtype=np.float64,
                    )
                )
            )
            if nonclean
            else None
        ),
        "results": rows,
    }

    save_json(
        output_root
        / "crossmodal_stress_summary.json",
        summary_payload,
    )

    write_csv(
        output_root
        / "crossmodal_stress_summary.csv",
        rows,
        summary_fields,
    )

    if x.variant == "darf":
        gate_rows: List[Dict[str, Any]] = []

        for result in results:
            gate_rows.extend(
                gate_rows_for_result(result)
            )

        clean_gate = {
            int(row["scale"]): float(
                row["g_nir_mean"]
            )
            for row in gate_rows
            if row["condition"] == "Clean"
        }

        for row in gate_rows:
            scale = int(row["scale"])
            row["delta_mean_vs_clean"] = (
                float(row["g_nir_mean"])
                - clean_gate[scale]
                if scale in clean_gate
                else None
            )

        save_json(
            output_root / "gate_statistics.json",
            {
                "model": model_id,
                "protocol_sha256": protocol_sha256(),
                "gate_semantics": (
                    "g_NIR residual correction strength"
                ),
                "gate_supervision": False,
                "rows": gate_rows,
            },
        )

        write_csv(
            output_root / "gate_statistics.csv",
            gate_rows,
            [
                "condition",
                "stress_axis",
                "family",
                "severity_level",
                "mismatch_role",
                "rgb_level",
                "nir_level",
                "shift_pixels",
                "scale",
                "count",
                "g_nir_mean",
                "g_nir_std",
                "g_nir_min",
                "g_nir_max",
                "delta_mean_vs_clean",
            ],
        )

    else:
        residual_rows = [
            fixed_residual_row(result)
            for result in results
        ]

        fields = [
            "condition",
            "stress_axis",
            "family",
            "severity_level",
            "mismatch_role",
            "rgb_level",
            "nir_level",
            "shift_pixels",
            "fixed_nir_strength",
        ]

        for s in range(1, NUM_SCALES + 1):
            fields.extend(
                [
                    f"scale{s}_mean",
                    f"scale{s}_std_across_tiles",
                    f"scale{s}_effective_mean",
                ]
            )

        save_json(
            output_root / "residual_statistics.json",
            {
                "model": model_id,
                "protocol_sha256": protocol_sha256(),
                "fixed_nir_strength": 1.0,
                "rows": residual_rows,
            },
        )

        write_csv(
            output_root / "residual_statistics.csv",
            residual_rows,
            fields,
        )

    progress.finish(output_root=output_root)

    elapsed = time.time() - started

    print()
    print("=" * 136)
    print("CROSS-MODAL STRESS SUMMARY")
    print("=" * 136)
    print(
        f"{'Condition':<48} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'Retention%':>12}"
    )
    print("-" * 136)

    for row in rows:
        print(
            f"{str(row['condition']):<48} "
            f"{float(row['miou']):>10.6f} "
            f"{float(row['drop_miou']):>10.6f} "
            f"{float(row['retention_pct']):>12.3f}"
        )

    print("-" * 136)
    print(
        f"elapsed              : "
        f"{joint_val.base.format_duration(elapsed)}"
    )
    print(
        f"summary              : "
        f"{output_root / 'crossmodal_stress_summary.csv'}"
    )

    if x.variant == "darf":
        print(
            f"gate statistics      : "
            f"{output_root / 'gate_statistics.csv'}"
        )
    else:
        print(
            f"residual statistics  : "
            f"{output_root / 'residual_statistics.csv'}"
        )

    print(
        f"progress             : "
        f"{output_root / 'validation_progress.json'}"
    )
    print(joint_val.gpu_memory_text(device))
    print("=" * 136)


if __name__ == "__main__":
    main()
