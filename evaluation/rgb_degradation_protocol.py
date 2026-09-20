#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB Degradation Protocol v2 — final implementation revision.

This file supersedes the earlier v2 implementation while KEEPING the same
scientific v2 conditions:

    Gaussian Noise:
        L1 sigma_255=10
        L2 sigma_255=25
        L3 sigma_255=50

    Gaussian Blur:
        L1 sigma=1.0
        L2 sigma=2.0
        L3 sigma=4.0

    RGB Underexposure:
        L1 alpha=0.8
        L2 alpha=0.6
        L3 alpha=0.4

Critical compatibility fix
--------------------------
Model A Gaussian Noise was evaluated under protocol v1. The original v1 noise
generator derived its deterministic per-tile seed from the string
"RGB Degradation Protocol v1".

The first v2 implementation changed that seed namespace to "... v2", which
would produce a DIFFERENT Gaussian noise realization for Model B even though
sigma remained unchanged.

This final implementation explicitly freezes:

    gaussian_noise RNG namespace = "RGB Degradation Protocol v1"

so Model A / B / C-noGate / C receive pixel-identical Gaussian noise for the
same tile + level. Blur and underexposure are deterministic and are unaffected.

The scientific protocol remains "RGB Degradation Protocol v2"; the
implementation_revision and rng_policy fields make this detail auditable.

Expected location:
    evaluation/rgb_degradation_protocol.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from data_pipeline.potsdam_dataset import (
    PotsdamSlidingWindowDataset,
    require,
)


DEGRADATION_PROTOCOL_VERSION = "RGB Degradation Protocol v2"
IMPLEMENTATION_REVISION = 2
DEGRADATION_SEED = 20260917

LEGACY_V1_PROTOCOL_SHA256 = (
    "1eff7b2c809c3ce551cf2c0d8194922640158b923adc7382278c3f43b10cb8f1"
)

PRE_FIX_V2_PROTOCOL_SHA256 = (
    "574f9d508f2e063aadb11c173d9e49518261e553d778c16c24a66afed002e7bf"
)

GAUSSIAN_NOISE_RNG_NAMESPACE = "RGB Degradation Protocol v1"


DEGRADATION_PROTOCOL: Dict[str, Any] = {
    "protocol_version": DEGRADATION_PROTOCOL_VERSION,
    "implementation_revision": IMPLEMENTATION_REVISION,
    "seed": DEGRADATION_SEED,
    "scope": {
        "degrade": "RGB only",
        "nir": "clean / unchanged",
        "gt": "unchanged",
        "application_domain": "raw uint8 RGB, before normalization",
        "spatial_scope": "full 6000x6000 tile before sliding-window crop",
        "overlap_consistency": True,
    },
    "rng_policy": {
        "gaussian_noise_seed_namespace": GAUSSIAN_NOISE_RNG_NAMESPACE,
        "reason": (
            "Preserve pixel-identical Gaussian noise realizations from the "
            "completed Model A protocol-v1 evaluation when protocol v2 adds "
            "RGB underexposure."
        ),
    },
    "corruptions": {
        "gaussian_noise": {
            "description": (
                "Additive zero-mean independent Gaussian noise in raw 8-bit "
                "RGB space; result is rounded and clipped to [0,255]."
            ),
            "levels": {
                "L1": {"sigma_255": 10.0},
                "L2": {"sigma_255": 25.0},
                "L3": {"sigma_255": 50.0},
            },
        },
        "gaussian_blur": {
            "description": (
                "Full-tile Gaussian blur using cv2.GaussianBlur with "
                "automatic kernel size and BORDER_REFLECT_101."
            ),
            "levels": {
                "L1": {"sigma": 1.0},
                "L2": {"sigma": 2.0},
                "L3": {"sigma": 4.0},
            },
        },
        "rgb_underexposure": {
            "description": (
                "Visible-channel exposure degradation in raw 8-bit RGB "
                "space: RGB' = round(clip(alpha * RGB, 0, 255)). "
                "NIR remains clean."
            ),
            "levels": {
                "L1": {"alpha": 0.8},
                "L2": {"alpha": 0.6},
                "L3": {"alpha": 0.4},
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


def degradation_protocol_sha256() -> str:
    return hashlib.sha256(
        _canonical_json_bytes(DEGRADATION_PROTOCOL)
    ).hexdigest()


def write_degradation_protocol(path: Path) -> None:
    payload = dict(DEGRADATION_PROTOCOL)
    payload["sha256"] = degradation_protocol_sha256()
    payload["legacy_v1_protocol_sha256"] = LEGACY_V1_PROTOCOL_SHA256
    payload["pre_fix_v2_protocol_sha256"] = PRE_FIX_V2_PROTOCOL_SHA256
    payload["implementation_fix"] = (
        "Gaussian Noise keeps the v1 RNG namespace so every model sees the "
        "same deterministic noise realization used by Model A."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def available_corruptions() -> Tuple[str, ...]:
    return tuple(DEGRADATION_PROTOCOL["corruptions"].keys())


def available_levels(corruption: str) -> Tuple[str, ...]:
    _validate_condition(corruption, "L1")
    return tuple(
        DEGRADATION_PROTOCOL["corruptions"][corruption]["levels"].keys()
    )


def condition_name(corruption: str, level: str) -> str:
    _validate_condition(corruption, level)
    return f"{corruption}_{level}"


def condition_spec(corruption: str, level: str) -> Dict[str, Any]:
    _validate_condition(corruption, level)
    return dict(
        DEGRADATION_PROTOCOL["corruptions"][corruption]["levels"][level]
    )


def _validate_condition(corruption: str, level: str) -> None:
    corruptions = DEGRADATION_PROTOCOL["corruptions"]

    if corruption not in corruptions:
        raise ValueError(
            f"Unknown corruption {corruption!r}; "
            f"available={list(corruptions)}"
        )

    levels = corruptions[corruption]["levels"]
    if level not in levels:
        raise ValueError(
            f"Unknown level {level!r} for {corruption}; "
            f"available={list(levels)}"
        )


def _gaussian_noise_seed(
    *,
    tile_id: str,
    corruption: str,
    level: str,
) -> int:
    """
    EXACTLY reproduce the protocol-v1 Gaussian Noise seed derivation.

    v1 used:
        f"{DEGRADATION_PROTOCOL_VERSION}|{seed}|{tile}|{corruption}|{level}"

    where its DEGRADATION_PROTOCOL_VERSION string was
        "RGB Degradation Protocol v1"

    We freeze that namespace explicitly here.
    """
    payload = (
        f"{GAUSSIAN_NOISE_RNG_NAMESPACE}|{DEGRADATION_SEED}|"
        f"{tile_id}|{corruption}|{level}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(
        digest[:8],
        "big",
        signed=False,
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


def _gaussian_noise_uint8(
    rgb: np.ndarray,
    *,
    sigma_255: float,
    seed: int,
    chunk_rows: int = 256,
) -> np.ndarray:
    if sigma_255 <= 0:
        raise ValueError("sigma_255 must be > 0")

    rng = np.random.default_rng(seed)
    out = np.empty_like(rgb)

    for y0 in range(0, rgb.shape[0], chunk_rows):
        y1 = min(rgb.shape[0], y0 + chunk_rows)

        src = rgb[y0:y1].astype(
            np.float32,
            copy=False,
        )

        noise = rng.normal(
            loc=0.0,
            scale=float(sigma_255),
            size=src.shape,
        ).astype(np.float32)

        degraded = src + noise
        np.clip(degraded, 0.0, 255.0, out=degraded)
        np.rint(degraded, out=degraded)

        out[y0:y1] = degraded.astype(np.uint8)

    return out


def _gaussian_blur_uint8(
    rgb: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    out = cv2.GaussianBlur(
        rgb,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )

    if out.dtype != np.uint8 or out.shape != rgb.shape:
        raise RuntimeError(
            "GaussianBlur returned unexpected result: "
            f"shape={out.shape}, dtype={out.dtype}"
        )

    return np.ascontiguousarray(out)


def _rgb_underexposure_uint8(
    rgb: np.ndarray,
    *,
    alpha: float,
    chunk_rows: int = 512,
) -> np.ndarray:
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"Underexposure alpha must satisfy 0<alpha<1, got {alpha}"
        )

    out = np.empty_like(rgb)

    for y0 in range(0, rgb.shape[0], chunk_rows):
        y1 = min(rgb.shape[0], y0 + chunk_rows)

        degraded = (
            rgb[y0:y1].astype(
                np.float32,
                copy=False,
            )
            * float(alpha)
        )

        np.clip(degraded, 0.0, 255.0, out=degraded)
        np.rint(degraded, out=degraded)

        out[y0:y1] = degraded.astype(np.uint8)

    return out


def apply_rgb_degradation_full_tile(
    rgb: np.ndarray,
    *,
    tile_id: str,
    corruption: str,
    level: str,
) -> np.ndarray:
    _validate_rgb_full_tile(rgb)
    _validate_condition(corruption, level)

    spec = condition_spec(corruption, level)

    if corruption == "gaussian_noise":
        return _gaussian_noise_uint8(
            rgb,
            sigma_255=float(spec["sigma_255"]),
            seed=_gaussian_noise_seed(
                tile_id=tile_id,
                corruption=corruption,
                level=level,
            ),
        )

    if corruption == "gaussian_blur":
        return _gaussian_blur_uint8(
            rgb,
            sigma=float(spec["sigma"]),
        )

    if corruption == "rgb_underexposure":
        return _rgb_underexposure_uint8(
            rgb,
            alpha=float(spec["alpha"]),
        )

    raise AssertionError(
        f"Validated corruption has no implementation: {corruption}"
    )


class DegradedPotsdamSlidingWindowDataset(
    PotsdamSlidingWindowDataset
):
    """
    Frozen validation/test dataset with full-tile RGB degradation.

    RGB is degraded before project normalization.
    NIR is copied from the clean raw RGBIR tile.
    GT and window coordinates are unchanged.

    Use num_workers=0: the class caches one 6000x6000 degraded RGB tile and
    validation is tile-major.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        split: str,
        corruption: str,
        level: str,
    ):
        super().__init__(
            project_root=project_root,
            split=split,
        )

        _validate_condition(corruption, level)

        self.corruption = corruption
        self.level = level
        self.condition = condition_name(corruption, level)

        self._degraded_cache_tile_id: str | None = None
        self._degraded_cache_rgb: np.ndarray | None = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_degraded_cache_tile_id"] = None
        state["_degraded_cache_rgb"] = None
        return state

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

        degraded = apply_rgb_degradation_full_tile(
            rgb_clean,
            tile_id=tile_id,
            corruption=self.corruption,
            level=self.level,
        )

        require(
            degraded.shape
            == (
                self.spec.tile_size,
                self.spec.tile_size,
                3,
            ),
            f"{tile_id}: degraded RGB shape invalid: {degraded.shape}",
        )
        require(
            degraded.dtype == np.uint8,
            f"{tile_id}: degraded RGB dtype invalid: {degraded.dtype}",
        )

        self._degraded_cache_tile_id = tile_id
        self._degraded_cache_rgb = degraded

        return degraded

    def clear_degradation_cache(self) -> None:
        self._degraded_cache_tile_id = None
        self._degraded_cache_rgb = None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id, window_index, x, y = self.index_to_tile_window(index)

        rgbir = self._read_rgbir(tile_id)
        degraded_rgb = self._get_degraded_rgb(tile_id, rgbir)

        h = int(self.spec.crop_size)
        w = int(self.spec.crop_size)

        rgb_crop = degraded_rgb[
            y:y + h,
            x:x + w,
            :,
        ]

        nir_crop = rgbir[
            y:y + h,
            x:x + w,
            3:4,
        ]

        require(
            rgb_crop.shape == (h, w, 3),
            f"{tile_id}: degraded RGB crop invalid: {rgb_crop.shape}",
        )
        require(
            nir_crop.shape == (h, w, 1),
            f"{tile_id}: clean NIR crop invalid: {nir_crop.shape}",
        )

        rgbir_for_normalization = np.empty(
            (h, w, 4),
            dtype=np.uint8,
        )
        rgbir_for_normalization[..., :3] = rgb_crop
        rgbir_for_normalization[..., 3:4] = nir_crop

        rgb_t, nir_t = self.spec.normalize_rgb_nir(
            rgbir_for_normalization,
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
            "corruption": self.corruption,
            "severity_level": self.level,
        }
