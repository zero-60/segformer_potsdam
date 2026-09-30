#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atmospheric Fog/Haze Degradation Protocol v1 for segformer_potsdam.

This file is intentionally separate from evaluation/rgb_degradation_protocol.py
so the existing RGB Degradation Protocol v2 hash and completed results remain
unchanged.

Scientific role:
- Current Robust-3 training sees Gaussian Noise / Gaussian Blur /
  RGB Underexposure.
- Fog/Haze is an UNSEEN / OOD evaluation degradation.
- RGB only is degraded.
- NIR and GT remain unchanged.
- Fog is applied on the full raw RGB tile before crop and normalization.
- Same tile + level gets deterministic pixel-identical fog for all models.

Atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))

Potsdam has no depth map, so t(x) is a deterministic low-frequency spatial
proxy, not a depth-derived or meteorologically exact Chongqing fog simulation.

Frozen fog levels:
    L1 transmission range [0.78, 0.92]
    L2 transmission range [0.58, 0.78]
    L3 transmission range [0.38, 0.62]

Expected location:
    evaluation/fog_degradation_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from data_pipeline.potsdam_dataset import (
    PotsdamSlidingWindowDataset,
    require,
)
from evaluation.rgb_degradation_protocol import (
    DegradedPotsdamSlidingWindowDataset,
)


FOG_PROTOCOL_VERSION = "Atmospheric Fog Degradation Protocol v1"
FOG_IMPLEMENTATION_REVISION = 1
FOG_SEED = 20260928
FOG_CORRUPTION_NAME = "atmospheric_fog"
FOG_ATMOSPHERIC_LIGHT_255 = (245.0, 245.0, 245.0)

FOG_PROTOCOL: Dict[str, Any] = {
    "protocol_version": FOG_PROTOCOL_VERSION,
    "implementation_revision": FOG_IMPLEMENTATION_REVISION,
    "seed": FOG_SEED,
    "scientific_role": (
        "UNSEEN/OOD evaluation degradation for the current Robust-3 training "
        "regime; do not add to training unless intentionally defining a new "
        "Robust-4 experiment."
    ),
    "scope": {
        "degrade": "RGB only",
        "nir": "clean / unchanged",
        "gt": "unchanged",
        "application_domain": "raw uint8 RGB, before normalization",
        "spatial_scope": "full 6000x6000 tile before sliding-window crop",
        "overlap_consistency": True,
    },
    "physical_model": {
        "equation": "I(x) = J(x) * t(x) + A * (1 - t(x))",
        "atmospheric_light_255": list(FOG_ATMOSPHERIC_LIGHT_255),
        "transmission_source": (
            "deterministic low-frequency spatial proxy; no depth/DEM is used"
        ),
        "interpretation_limit": (
            "controlled synthetic atmospheric degradation, not an exact "
            "meteorological reconstruction of real Chongqing fog"
        ),
    },
    "rng_policy": {
        "namespace": FOG_PROTOCOL_VERSION,
        "seed_derivation": (
            "sha256(protocol_version|seed|tile_id|atmospheric_fog|level)"
        ),
        "cross_model_fairness": (
            "same tile and level receive pixel-identical fog degradation"
        ),
    },
    "corruptions": {
        FOG_CORRUPTION_NAME: {
            "description": (
                "Spatially varying atmospheric scattering degradation applied "
                "to raw RGB while NIR remains clean."
            ),
            "levels": {
                "L1": {
                    "transmission_min": 0.78,
                    "transmission_max": 0.92,
                },
                "L2": {
                    "transmission_min": 0.58,
                    "transmission_max": 0.78,
                },
                "L3": {
                    "transmission_min": 0.38,
                    "transmission_max": 0.62,
                },
            },
        },
    },
}


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fog_protocol_sha256() -> str:
    return hashlib.sha256(
        _canonical_json_bytes(FOG_PROTOCOL)
    ).hexdigest()


def write_fog_protocol(path: Path) -> None:
    payload = dict(FOG_PROTOCOL)
    payload["sha256"] = fog_protocol_sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def available_fog_levels() -> Tuple[str, ...]:
    return tuple(
        FOG_PROTOCOL["corruptions"][FOG_CORRUPTION_NAME]["levels"].keys()
    )


def _validate_fog_condition(corruption: str, level: str) -> None:
    if corruption != FOG_CORRUPTION_NAME:
        raise ValueError(
            f"Fog protocol only supports {FOG_CORRUPTION_NAME!r}, "
            f"got {corruption!r}."
        )

    levels = FOG_PROTOCOL["corruptions"][FOG_CORRUPTION_NAME]["levels"]
    if level not in levels:
        raise ValueError(
            f"Unknown fog level {level!r}; available={list(levels)}"
        )


def fog_condition_name(corruption: str, level: str) -> str:
    _validate_fog_condition(corruption, level)
    return f"{corruption}_{level}"


def fog_condition_spec(corruption: str, level: str) -> Dict[str, Any]:
    _validate_fog_condition(corruption, level)
    return dict(
        FOG_PROTOCOL["corruptions"][FOG_CORRUPTION_NAME]["levels"][level]
    )


def _validate_rgb_full_tile(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"RGB full tile must be [H,W,3], got {rgb.shape}"
        )
    if rgb.dtype != np.uint8:
        raise ValueError(
            f"RGB full tile must be uint8, got {rgb.dtype}"
        )


def _fog_seed(
    *,
    tile_id: str,
    corruption: str,
    level: str,
) -> int:
    _validate_fog_condition(corruption, level)
    payload = (
        f"{FOG_PROTOCOL_VERSION}|{FOG_SEED}|"
        f"{tile_id}|{corruption}|{level}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _build_low_frequency_components(
    *,
    seed: int,
) -> Tuple[Tuple[float, float, float, float], ...]:
    rng = np.random.default_rng(int(seed))

    base_cycles = (0.65, 1.10, 1.75, 2.60)
    base_weights = (1.00, 0.70, 0.48, 0.30)

    components = []
    for cycles, weight in zip(base_cycles, base_weights):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        anisotropy = float(rng.uniform(0.85, 1.15))
        frequency_x = float(cycles) * math.cos(angle) * anisotropy
        frequency_y = float(cycles) * math.sin(angle) / anisotropy
        phase = float(rng.uniform(0.0, 2.0 * math.pi))

        components.append(
            (
                frequency_x,
                frequency_y,
                phase,
                float(weight),
            )
        )

    return tuple(components)


def _fog_transmission_chunk(
    *,
    y0: int,
    y1: int,
    height: int,
    width: int,
    transmission_min: float,
    transmission_max: float,
    components: Tuple[Tuple[float, float, float, float], ...],
) -> np.ndarray:
    if not (
        0.0 < transmission_min < transmission_max <= 1.0
    ):
        raise ValueError(
            "Transmission must satisfy 0 < min < max <= 1."
        )

    yy = (
        np.arange(y0, y1, dtype=np.float32)[:, None]
        / max(float(height - 1), 1.0)
    )
    xx = (
        np.arange(width, dtype=np.float32)[None, :]
        / max(float(width - 1), 1.0)
    )

    field = np.zeros((y1 - y0, width), dtype=np.float32)
    total_weight = 0.0

    for frequency_x, frequency_y, phase, weight in components:
        angle = (
            2.0
            * math.pi
            * (
                float(frequency_x) * xx
                + float(frequency_y) * yy
            )
            + float(phase)
        )

        field += (
            float(weight)
            * np.sin(angle).astype(np.float32, copy=False)
        )
        total_weight += abs(float(weight))

    if total_weight <= 0.0:
        raise RuntimeError("Fog field has zero total weight.")

    field /= float(total_weight)
    np.clip(field, -1.0, 1.0, out=field)

    density = (field + 1.0) * 0.5
    transmission = (
        float(transmission_max)
        - density
        * (
            float(transmission_max)
            - float(transmission_min)
        )
    )

    return transmission.astype(np.float32, copy=False)


def apply_fog_degradation_full_tile(
    rgb: np.ndarray,
    *,
    tile_id: str,
    level: str,
    corruption: str = FOG_CORRUPTION_NAME,
    chunk_rows: int = 256,
) -> np.ndarray:
    _validate_rgb_full_tile(rgb)
    _validate_fog_condition(corruption, level)

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")

    spec = fog_condition_spec(corruption, level)
    transmission_min = float(spec["transmission_min"])
    transmission_max = float(spec["transmission_max"])

    seed = _fog_seed(
        tile_id=tile_id,
        corruption=corruption,
        level=level,
    )
    components = _build_low_frequency_components(seed=seed)

    atmospheric_light = np.asarray(
        FOG_ATMOSPHERIC_LIGHT_255,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    height = int(rgb.shape[0])
    width = int(rgb.shape[1])
    out = np.empty_like(rgb)

    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)

        transmission = _fog_transmission_chunk(
            y0=y0,
            y1=y1,
            height=height,
            width=width,
            transmission_min=transmission_min,
            transmission_max=transmission_max,
            components=components,
        )

        src = rgb[y0:y1].astype(np.float32, copy=False)
        t3 = transmission[:, :, None]

        degraded = (
            src * t3
            + atmospheric_light * (1.0 - t3)
        )

        np.clip(degraded, 0.0, 255.0, out=degraded)
        np.rint(degraded, out=degraded)
        out[y0:y1] = degraded.astype(np.uint8)

    return np.ascontiguousarray(out)


class FogDegradedPotsdamSlidingWindowDataset(
    DegradedPotsdamSlidingWindowDataset
):
    """
    Existing-validator-compatible Potsdam wrapper for Fog L1/L2/L3.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        split: str,
        level: str,
    ):
        # Bypass the RGB-v2 degraded wrapper constructor because Fog is kept
        # intentionally outside the existing v2 protocol.
        PotsdamSlidingWindowDataset.__init__(
            self,
            project_root=project_root,
            split=split,
        )

        _validate_fog_condition(
            FOG_CORRUPTION_NAME,
            level,
        )

        self.corruption = FOG_CORRUPTION_NAME
        self.level = level
        self.condition = fog_condition_name(
            self.corruption,
            self.level,
        )

        self._degraded_cache_tile_id: str | None = None
        self._degraded_cache_rgb: np.ndarray | None = None

    def _get_degraded_rgb(
        self,
        tile_id: str,
        rgbir: np.ndarray,
    ) -> np.ndarray:
        if (
            self._degraded_cache_tile_id == tile_id
            and self._degraded_cache_rgb is not None
        ):
            return self._degraded_cache_rgb

        rgb_clean = rgbir[..., 0:3]

        degraded = apply_fog_degradation_full_tile(
            rgb_clean,
            tile_id=tile_id,
            level=self.level,
            corruption=self.corruption,
        )

        require(
            degraded.shape
            == (
                self.spec.tile_size,
                self.spec.tile_size,
                3,
            ),
            f"{tile_id}: fog RGB shape invalid: {degraded.shape}",
        )

        require(
            degraded.dtype == np.uint8,
            f"{tile_id}: fog RGB dtype invalid: {degraded.dtype}",
        )

        self._degraded_cache_tile_id = tile_id
        self._degraded_cache_rgb = degraded

        return degraded


def _self_test() -> None:
    rgb = np.full(
        (96, 128, 3),
        96,
        dtype=np.uint8,
    )

    means = {}

    for level in available_fog_levels():
        first = apply_fog_degradation_full_tile(
            rgb,
            tile_id="self_test_tile",
            level=level,
            chunk_rows=31,
        )
        second = apply_fog_degradation_full_tile(
            rgb,
            tile_id="self_test_tile",
            level=level,
            chunk_rows=47,
        )

        if not np.array_equal(first, second):
            raise RuntimeError(
                f"{level}: result changed with chunk_rows."
            )
        if np.array_equal(first, rgb):
            raise RuntimeError(
                f"{level}: result equals clean RGB."
            )

        means[level] = float(first.mean())

    if not (
        means["L1"] < means["L2"] < means["L3"]
    ):
        raise RuntimeError(
            f"Fog severity sanity check failed: {means}"
        )

    print("[fog protocol self-test] PASS")
    print(f"protocol_sha256={fog_protocol_sha256()}")
    print(f"mean_uint8_by_level={means}")


if __name__ == "__main__":
    _self_test()
