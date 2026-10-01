#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train M1: SegFormer-B2 RGB-only Robust-4 baseline.

Robust-4 families:
    1) Gaussian noise
    2) Gaussian blur
    3) RGB underexposure
    4) Atmospheric fog

Scientific role
---------------
M0 / B2-RGB-Clean remains unchanged and MUST NOT be retrained with fog.
This script trains the robust RGB baseline that will later be compared fairly
against:
    M2 / B2-RGBNIR-Fixed-Robust4
    M3 / B2-DARF-Robust4

Fog remains absent from the clean M0 baseline, but IS part of M1/M2/M3
Robust-4 training.

Training recipe
---------------
- Backbone: SegFormer-B2
- RGB only
- 6 Potsdam classes
- 120 epochs
- batch size 1
- gradient accumulation 16
- base LR 3e-5
- new classifier LR 3e-4
- weight decay 0.01
- warmup 500 successful optimizer updates
- CE + 0.5 * Lovasz
- AMP
- grad clip 1.0
- clean warm-up 10 epochs
- corruption ramp 20 epochs
- max corruption probability 0.50
- corruption family sampled uniformly from 4 families
- severity sampled continuously from [0.10, 1.00]

Fog augmentation
----------------
Input RGB is first inverse-ImageNet-normalized to [0,1].

Atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))

Severity mapping:
    mean transmission = 0.90 - 0.50 * severity
    spatial variation = 0.04 + 0.11 * severity
    atmospheric light A ~ Uniform(0.94, 0.99)

At severity ~= 1.0, mean transmission ~= 0.40, consistent with the severe
Fog-L3 validation stress level. Training uses a continuous distribution, not
only the discrete validation L1/L2/L3 points.

The fog field is low-frequency and spatially varying inside each training crop.
This is intentionally a training augmentation; validation continues to use the
frozen full-tile deterministic fog protocol in validate_model_d_fog.py /
validate_model_b2_rgb.py.

Run
---
    python tools/train_model_b2_rgb_robust4.py

Resume
------
    python tools/train_model_b2_rgb_robust4.py --resume auto

Output
------
outputs/training/b2_rgb_robust4/
├── protocol.json
├── progress.json
├── train_log.jsonl
└── checkpoints/
    ├── latest.pt
    └── final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model_b2_rgb as base

from data_pipeline.potsdam_dataloader import (
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from train_model_a_rgb import (
    CachedPotsdamTrainDataset,
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_NUM_WORKERS,
    atomic_torch_save,
    build_poly_scheduler,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from train_model_d_darf_b2 import (
    corruption_probability,
    lovasz_softmax,
)

# =============================================================================
# Frozen Robust-4 protocol
# =============================================================================

MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "B2 RGB Robust-4 Protocol v1"

MODEL_ID = "B2_RGB_ROBUST4"
MODEL_NAME = "SegFormer-B2 RGB-only Robust-4 Baseline"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_rgb_robust4"
)

DEFAULT_SEED = 20260917
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16

DEFAULT_BASE_LR = 3e-5
DEFAULT_NEW_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP = 500
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LOVASZ_WEIGHT = 0.50

DEFAULT_CLEAN_WARMUP_EPOCHS = 10
DEFAULT_CORRUPTION_RAMP_EPOCHS = 20
DEFAULT_MAX_CORRUPTION_PROB = 0.50

DEFAULT_PROGRESS_EVERY = 5
DEFAULT_ETA_WARMUP_BATCHES = 20
DEFAULT_ETA_EMA_ALPHA = 0.08

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROBUST4_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "rgb_underexposure",
    "fog",
)


# =============================================================================
# Helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _rgb_denormalize(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 3, 1, 1)

    return torch.clamp(
        rgb * std + mean,
        0.0,
        1.0,
    )


def _rgb_normalize(rgb01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=rgb01.device,
        dtype=rgb01.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=rgb01.device,
        dtype=rgb01.dtype,
    ).view(1, 3, 1, 1)

    return (
        rgb01 - mean
    ) / std


def _gaussian_blur_single(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    sigma = float(sigma)

    radius = max(
        1,
        int(
            math.ceil(
                3.0 * sigma
            )
        ),
    )

    kernel_size = min(
        2 * radius + 1,
        31,
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = (
        kernel_size // 2
    )

    coords = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )

    kernel_1d = torch.exp(
        -(
            coords
            * coords
        )
        / (
            2.0
            * sigma
            * sigma
        )
    )

    kernel_1d = (
        kernel_1d
        / kernel_1d.sum()
    )

    kernel_2d = (
        kernel_1d[:, None]
        * kernel_1d[None, :]
    )

    weight = (
        kernel_2d
        .view(
            1,
            1,
            kernel_size,
            kernel_size,
        )
        .expand(
            3,
            1,
            kernel_size,
            kernel_size,
        )
        .contiguous()
    )

    x = image.unsqueeze(
        0
    )

    x = F.pad(
        x,
        (
            radius,
            radius,
            radius,
            radius,
        ),
        mode="reflect",
    )

    y = F.conv2d(
        x,
        weight,
        groups=3,
    )

    return y.squeeze(
        0
    )


def _fog_single(
    image: torch.Tensor,
    severity: float,
) -> torch.Tensor:
    """
    Apply spatially varying atmospheric fog to one RGB crop in [0,1].

    image: [3,H,W]
    """
    severity = float(
        min(
            max(
                severity,
                0.0,
            ),
            1.0,
        )
    )

    _, h, w = image.shape

    device = image.device
    dtype = image.dtype

    mean_t = (
        0.90
        - 0.50
        * severity
    )

    variation = (
        0.04
        + 0.11
        * severity
    )

    # Continuous atmospheric light, close to the validation protocol's 245/255.
    atmosphere = (
        0.94
        + 0.05
        * torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
    )

    yy = torch.linspace(
        0.0,
        1.0,
        h,
        device=device,
        dtype=dtype,
    ).view(
        h,
        1,
    )

    xx = torch.linspace(
        0.0,
        1.0,
        w,
        device=device,
        dtype=dtype,
    ).view(
        1,
        w,
    )

    field = torch.zeros(
        (
            h,
            w,
        ),
        device=device,
        dtype=dtype,
    )

    # Four low-frequency components, matching the validation protocol concept.
    weights = (
        0.34,
        0.27,
        0.22,
        0.17,
    )

    for weight in weights:
        fx = (
            0.35
            + 1.25
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        fy = (
            0.35
            + 1.25
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        phase = (
            2.0
            * math.pi
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        sign = torch.where(
            torch.rand(
                (),
                device=device,
            )
            < 0.5,
            torch.tensor(
                -1.0,
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                1.0,
                device=device,
                dtype=dtype,
            ),
        )

        field = (
            field
            + float(
                weight
            )
            * sign
            * torch.sin(
                2.0
                * math.pi
                * (
                    fx
                    * xx
                    + fy
                    * yy
                )
                + phase
            )
        )

    field = torch.clamp(
        field,
        -1.0,
        1.0,
    )

    t = (
        mean_t
        + variation
        * field
    )

    # Keep a physically valid transmission range.
    t_min = max(
        0.20,
        mean_t
        - variation,
    )

    t_max = min(
        0.98,
        mean_t
        + variation,
    )

    t = torch.clamp(
        t,
        t_min,
        t_max,
    )

    t = t.unsqueeze(
        0
    )

    out = (
        image
        * t
        + atmosphere
        * (
            1.0
            - t
        )
    )

    return torch.clamp(
        out,
        0.0,
        1.0,
    )


def degrade_rgb_batch_robust4(
    rgb_normalized: torch.Tensor,
    *,
    probability: float,
) -> tuple[
    torch.Tensor,
    Dict[str, int],
    torch.Tensor,
]:
    """
    Robust-4 RGB augmentation.

    Returns
    -------
    augmented normalized RGB
    family counts
    severity [B] (0 for clean)
    """
    with torch.autocast(
        device_type=(
            rgb_normalized.device.type
        ),
        enabled=False,
    ):
        rgb01 = _rgb_denormalize(
            rgb_normalized.float()
        )

        out = rgb01.clone()

        batch = int(
            rgb01.shape[0]
        )

        severity_out = torch.zeros(
            (
                batch,
            ),
            device=rgb01.device,
            dtype=torch.float32,
        )

        counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "rgb_underexposure": 0,
            "fog": 0,
        }

        for index in range(
            batch
        ):
            do_corrupt = bool(
                (
                    torch.rand(
                        (),
                        device=rgb01.device,
                    )
                    < probability
                )
                .item()
            )

            if not do_corrupt:
                counts[
                    "clean"
                ] += 1
                continue

            severity = float(
                (
                    0.10
                    + 0.90
                    * torch.rand(
                        (),
                        device=rgb01.device,
                    )
                )
                .item()
            )

            family = int(
                torch.randint(
                    low=0,
                    high=4,
                    size=(),
                    device=rgb01.device,
                )
                .item()
            )

            if family == 0:
                sigma_255 = (
                    5.0
                    + 45.0
                    * severity
                )

                noise = (
                    torch.randn_like(
                        out[
                            index
                        ]
                    )
                    * (
                        sigma_255
                        / 255.0
                    )
                )

                out[
                    index
                ] = torch.clamp(
                    out[
                        index
                    ]
                    + noise,
                    0.0,
                    1.0,
                )

                counts[
                    "gaussian_noise"
                ] += 1

            elif family == 1:
                sigma = (
                    0.5
                    + 3.5
                    * severity
                )

                out[
                    index
                ] = (
                    _gaussian_blur_single(
                        out[
                            index
                        ],
                        sigma,
                    )
                )

                counts[
                    "gaussian_blur"
                ] += 1

            elif family == 2:
                alpha = (
                    1.0
                    - 0.60
                    * severity
                )

                out[
                    index
                ] = torch.clamp(
                    out[
                        index
                    ]
                    * alpha,
                    0.0,
                    1.0,
                )

                counts[
                    "rgb_underexposure"
                ] += 1

            else:
                out[
                    index
                ] = _fog_single(
                    out[
                        index
                    ],
                    severity,
                )

                counts[
                    "fog"
                ] += 1

            severity_out[
                index
            ] = severity

        out_normalized = (
            _rgb_normalize(
                out
            )
        )

    return (
        out_normalized,
        counts,
        severity_out,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train M1 / SegFormer-B2 RGB-only Robust-4 baseline."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=DEFAULT_BASE_LR,
    )
    parser.add_argument(
        "--new-lr",
        type=float,
        default=DEFAULT_NEW_LR,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--lovasz-weight",
        type=float,
        default=DEFAULT_LOVASZ_WEIGHT,
    )

    parser.add_argument(
        "--clean-warmup-epochs",
        type=int,
        default=DEFAULT_CLEAN_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--corruption-ramp-epochs",
        type=int,
        default=DEFAULT_CORRUPTION_RAMP_EPOCHS,
    )
    parser.add_argument(
        "--max-corruption-prob",
        type=float,
        default=DEFAULT_MAX_CORRUPTION_PROB,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=DEFAULT_DATA_CACHE_DIR,
    )
    parser.add_argument(
        "--rebuild-data-cache",
        action="store_true",
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
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--resume",
        default="",
        help='Checkpoint path, or "auto" for latest.pt.',
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
    )

    parser.add_argument(
        "--eta-warmup-batches",
        type=int,
        default=DEFAULT_ETA_WARMUP_BATCHES,
    )

    parser.add_argument(
        "--eta-ema-alpha",
        type=float,
        default=DEFAULT_ETA_EMA_ALPHA,
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")

    if args.base_lr <= 0 or args.new_lr <= 0:
        parser.error("learning rates must be > 0")

    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")

    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")

    if args.grad_clip <= 0:
        parser.error("--grad-clip must be > 0")

    if args.lovasz_weight < 0:
        parser.error("--lovasz-weight must be >= 0")

    if args.clean_warmup_epochs < 0:
        parser.error("--clean-warmup-epochs must be >= 0")

    if args.corruption_ramp_epochs < 0:
        parser.error("--corruption-ramp-epochs must be >= 0")

    if not (
        0.0
        <= args.max_corruption_prob
        <= 1.0
    ):
        parser.error(
            "--max-corruption-prob must be in [0,1]"
        )

    if args.progress_every <= 0:
        parser.error("--progress-every must be > 0")

    if args.eta_warmup_batches <= 0:
        parser.error("--eta-warmup-batches must be > 0")

    if not (
        0.0
        < args.eta_ema_alpha
        <= 1.0
    ):
        parser.error(
            "--eta-ema-alpha must be in (0,1]"
        )

    return args


# =============================================================================
# Protocol / checkpoint
# =============================================================================

def build_protocol(
    *,
    args,
    model_meta,
    optimizer_summary,
    updates_per_epoch,
    total_updates,
    warmup,
    device,
    amp,
    cache,
    checkpointing_status,
):
    return {
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "module_version": (
            MODULE_VERSION
        ),
        "model_id": (
            MODEL_ID
        ),
        "model_name": (
            MODEL_NAME
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "checkpoint": (
            base.DEFAULT_CHECKPOINT
        ),
        "input_modalities": [
            "RGB"
        ],
        "nir_used": False,
        "fusion": None,
        "regime": (
            "robust4"
        ),
        "epochs": (
            args.epochs
        ),
        "batch_size": (
            args.batch_size
        ),
        "grad_accum_steps": (
            args.grad_accum_steps
        ),
        "effective_batch_size_nominal": (
            args.batch_size
            * args.grad_accum_steps
        ),
        "base_lr": (
            args.base_lr
        ),
        "new_classifier_lr": (
            args.new_lr
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "warmup_steps": (
            warmup
        ),
        "grad_clip": (
            args.grad_clip
        ),
        "loss": {
            "ce_weight": 1.0,
            "lovasz_weight": (
                args.lovasz_weight
            ),
            "gate_loss": False,
        },
        "corruption_training": {
            "enabled": True,
            "version": (
                "Robust-4"
            ),
            "fog_in_training": True,
            "clean_warmup_epochs": (
                args.clean_warmup_epochs
            ),
            "ramp_epochs": (
                args.corruption_ramp_epochs
            ),
            "max_probability": (
                args.max_corruption_prob
            ),
            "families": list(
                ROBUST4_FAMILIES
            ),
            "family_sampling": (
                "uniform"
            ),
            "severity_sampling": (
                "continuous Uniform[0.10,1.00]"
            ),
            "noise_sigma_255": [
                5.0,
                50.0,
            ],
            "blur_sigma": [
                0.5,
                4.0,
            ],
            "underexposure_alpha": [
                0.94,
                0.40,
            ],
            "fog": {
                "equation": (
                    "I(x)=J(x)*t(x)+A*(1-t(x))"
                ),
                "mean_transmission": (
                    "0.90 - 0.50 * severity"
                ),
                "spatial_variation": (
                    "0.04 + 0.11 * severity"
                ),
                "atmospheric_light": [
                    0.94,
                    0.99,
                ],
                "field_components": 4,
            },
        },
        "optimizer": (
            optimizer_summary
        ),
        "updates_per_epoch": (
            updates_per_epoch
        ),
        "total_update_steps": (
            total_updates
        ),
        "seed": (
            args.seed
        ),
        "data_seed": int(
            DATA_SEED
        ),
        "amp": (
            amp
        ),
        "gradient_checkpointing_requested": (
            args.gradient_checkpointing
        ),
        "gradient_checkpointing_status": (
            checkpointing_status
        ),
        "device": str(
            device
        ),
        "data_cache_dir": str(
            cache
        ),
        "model_meta": dict(
            model_meta
        ),
        "scientific_note": (
            "M0 clean is preserved unchanged. M1/M2/M3 Robust-4 must share "
            "the same four corruption families for fair architectural ablation."
        ),
    }


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    protocol,
    model_meta,
):
    return {
        "format_version": 2,
        "model_id": (
            MODEL_ID
        ),
        "model_name": (
            MODEL_NAME
        ),
        "regime": (
            "robust4"
        ),
        "model": (
            model.state_dict()
        ),
        "optimizer": (
            optimizer.state_dict()
        ),
        "scheduler": (
            scheduler.state_dict()
        ),
        "scaler": (
            scaler.state_dict()
        ),
        "epoch": int(
            epoch
        ),
        "global_step": int(
            global_step
        ),
        "step": int(
            global_step
        ),
        "protocol": dict(
            protocol
        ),
        "model_meta": dict(
            model_meta
        ),
        "rng_state": (
            capture_rng_state()
        ),
    }


def resolve_resume(
    value: str,
    checkpoint_dir: Path,
) -> Optional[Path]:
    if not value:
        return None

    if value.lower() == "auto":
        path = (
            checkpoint_dir
            / "latest.pt"
        )
    else:
        path = Path(
            value
        ).expanduser()

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    return path


def load_resume(
    *,
    path,
    model,
    optimizer,
    scheduler,
    scaler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get(
        "model_id"
    ) != MODEL_ID:
        raise RuntimeError(
            "Resume checkpoint model_id mismatch."
        )

    if checkpoint.get(
        "regime"
    ) != "robust4":
        raise RuntimeError(
            "Resume checkpoint is not Robust-4."
        )

    model.load_state_dict(
        checkpoint[
            "model"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler"
        ]
    )

    scaler.load_state_dict(
        checkpoint[
            "scaler"
        ]
    )

    restore_rng_state(
        checkpoint.get(
            "rng_state"
        )
    )

    return (
        int(
            checkpoint[
                "epoch"
            ]
        )
        + 1,
        int(
            checkpoint.get(
                "global_step",
                checkpoint.get(
                    "step",
                    0,
                ),
            )
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    output_dir = resolve(
        args.output_dir
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = resolve(
        args.data_cache_dir
    )

    print("=" * 118)
    print(MODEL_NAME)
    print("=" * 118)
    print(f"model id       : {MODEL_ID}")
    print("regime         : robust4")
    print("families       : Noise / Blur / Underexposure / Fog")
    print(f"output         : {output_dir}")
    print(
        f"start local    : "
        f"{base.format_local_datetime(base.local_now())}"
    )
    print("=" * 118)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    dataset = CachedPotsdamTrainDataset(
        PROJECT_ROOT,
        epoch=0,
        cache_root=cache,
    )

    dataset.prepare_cache(
        rebuild=args.rebuild_data_cache
    )

    sampler = DeterministicEpochSampler(
        dataset,
        seed=DATA_SEED,
        epoch=0,
    )

    loader = build_train_dataloader(
        dataset,
        sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    updates_per_epoch = math.ceil(
        len(loader)
        / args.grad_accum_steps
    )

    total_updates = (
        updates_per_epoch
        * args.epochs
    )

    warmup = min(
        args.warmup_steps,
        max(
            0,
            total_updates
            - 1,
        ),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    device = base.get_device(
        args.device
    )

    model, model_meta = (
        base.build_b2_rgb_model()
    )

    checkpointing_status = {
        "rgb_encoder": False
    }

    if args.gradient_checkpointing:
        checkpointing_status = (
            model
            .enable_gradient_checkpointing()
        )

    model.to(
        device
    )

    optimizer, optimizer_summary = (
        base.build_optimizer(
            model,
            base_lr=args.base_lr,
            new_lr=args.new_lr,
            weight_decay=args.weight_decay,
        )
    )

    scheduler = (
        build_poly_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup,
            total_steps=total_updates,
            power=1.0,
        )
    )

    amp = (
        device.type
        == "cuda"
        and not args.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
    )

    protocol = build_protocol(
        args=args,
        model_meta=model_meta,
        optimizer_summary=optimizer_summary,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        warmup=warmup,
        device=device,
        amp=amp,
        cache=cache,
        checkpointing_status=checkpointing_status,
    )

    write_json_atomic(
        output_dir
        / "protocol.json",
        protocol,
    )

    print(
        f"parameters      : "
        f"{model_meta['parameters']['total']:,}"
    )
    print(
        f"epochs          : "
        f"{args.epochs}"
    )
    print(
        f"batches/epoch   : "
        f"{len(loader)}"
    )
    print(
        f"updates/epoch   : "
        f"{updates_per_epoch}"
    )
    print(
        f"total updates   : "
        f"{total_updates}"
    )
    print(
        f"effective batch : "
        f"{args.batch_size * args.grad_accum_steps}"
    )
    print(
        f"loss            : "
        f"CE + {args.lovasz_weight} * Lovasz"
    )
    print(
        f"max p_corrupt   : "
        f"{args.max_corruption_prob}"
    )
    print(
        f"clean warm-up   : "
        f"{args.clean_warmup_epochs} epochs"
    )
    print(
        f"ramp            : "
        f"{args.corruption_ramp_epochs} epochs"
    )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0
    global_step = 0

    resume = resolve_resume(
        args.resume,
        checkpoint_dir,
    )

    if resume is not None:
        (
            start_epoch,
            global_step,
        ) = load_resume(
            path=resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print(
            f"[resume] {resume} | "
            f"start_epoch={start_epoch} | "
            f"global_step={global_step}"
        )

    if start_epoch >= args.epochs:
        print(
            "[done] requested epochs already completed."
        )
        return

    # -------------------------------------------------------------------------
    # Live progress
    # -------------------------------------------------------------------------

    progress = (
        base.LiveTrainingProgress(
            total_epochs=args.epochs,
            batches_per_epoch=len(loader),
            start_epoch=start_epoch,
            progress_every=args.progress_every,
            eta_warmup_batches=args.eta_warmup_batches,
            ema_alpha=args.eta_ema_alpha,
            output_path=(
                output_dir
                / "progress.json"
            ),
            device=device,
            enabled=(
                not args.no_progress
            ),
        )
    )

    log_path = (
        output_dir
        / "train_log.jsonl"
    )

    training_start = time.time()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        epoch_start = time.time()

        set_train_epoch(
            dataset,
            sampler,
            epoch,
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        p_corrupt = (
            corruption_probability(
                epoch,
                clean_warmup_epochs=(
                    args.clean_warmup_epochs
                ),
                ramp_epochs=(
                    args.corruption_ramp_epochs
                ),
                maximum=(
                    args.max_corruption_prob
                ),
            )
        )

        total_valid = 0
        ce_numerator = 0.0
        lovasz_sum = 0.0
        total_loss_sum = 0.0
        mini_batches = 0
        updates = 0
        amp_skips = 0
        last_grad_norm = float(
            "nan"
        )

        family_counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "rgb_underexposure": 0,
            "fog": 0,
        }

        severity_sum = 0.0
        degraded_samples = 0

        accumulation_target = (
            args.grad_accum_steps
        )

        for batch_index, batch in enumerate(
            loader
        ):
            batch_started = (
                time.monotonic()
            )

            if (
                batch_index
                % args.grad_accum_steps
                == 0
            ):
                remaining = (
                    len(loader)
                    - batch_index
                )

                accumulation_target = min(
                    args.grad_accum_steps,
                    remaining,
                )

            rgb = batch[
                "rgb"
            ].to(
                device,
                non_blocking=args.pin_memory,
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=args.pin_memory,
            ).long()

            (
                rgb_model,
                counts,
                severities,
            ) = degrade_rgb_batch_robust4(
                rgb,
                probability=p_corrupt,
            )

            for key in family_counts:
                family_counts[
                    key
                ] += int(
                    counts[
                        key
                    ]
                )

            degraded_mask = (
                severities
                > 0
            )

            if degraded_mask.any():
                severity_sum += float(
                    severities[
                        degraded_mask
                    ]
                    .sum()
                    .item()
                )

                degraded_samples += int(
                    degraded_mask
                    .sum()
                    .item()
                )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                details = model(
                    rgb_model,
                    return_details=True,
                )

                logits = details[
                    "logits"
                ]

                raw_logits = details[
                    "raw_logits"
                ]

                ce = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=(
                        base.IGNORE_INDEX
                    ),
                )

                lovasz = (
                    lovasz_softmax(
                        raw_logits,
                        labels,
                    )
                )

                total_loss = (
                    ce
                    + args.lovasz_weight
                    * lovasz
                )

                loss_for_backward = (
                    total_loss
                    / accumulation_target
                )

            if not torch.isfinite(
                total_loss
            ).item():
                raise FloatingPointError(
                    "Non-finite loss at "
                    f"epoch={epoch + 1}, "
                    f"batch={batch_index + 1}"
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            valid_pixels = int(
                (
                    labels
                    != base.IGNORE_INDEX
                )
                .sum()
                .item()
            )

            total_valid += (
                valid_pixels
            )

            ce_numerator += (
                float(
                    ce
                    .detach()
                    .item()
                )
                * valid_pixels
            )

            lovasz_sum += float(
                lovasz
                .detach()
                .item()
            )

            total_loss_sum += float(
                total_loss
                .detach()
                .item()
            )

            mini_batches += 1

            should_step = (
                (
                    (
                        batch_index
                        + 1
                    )
                    % args.grad_accum_steps
                    == 0
                )
                or (
                    batch_index
                    + 1
                    == len(loader)
                )
            )

            if should_step:
                scaler.unscale_(
                    optimizer
                )

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                    )
                )

                last_grad_norm = float(
                    grad_norm
                    .detach()
                    .item()
                )

                before = (
                    scaler
                    .get_scale()
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                after = (
                    scaler
                    .get_scale()
                )

                step_happened = (
                    not amp
                    or after >= before
                )

                if step_happened:
                    scheduler.step()
                    global_step += 1
                    updates += 1
                else:
                    amp_skips += 1

                optimizer.zero_grad(
                    set_to_none=True
                )

            lrs = {
                group.get(
                    "group_name",
                    str(index),
                ): float(
                    group[
                        "lr"
                    ]
                )
                for index, group in enumerate(
                    optimizer.param_groups
                )
            }

            batch_seconds = (
                time.monotonic()
                - batch_started
            )

            progress.observe_batch(
                batch_seconds
            )

            progress.update(
                epoch_zero_based=epoch,
                batch_zero_based=batch_index,
                loss=float(
                    total_loss
                    .detach()
                    .item()
                ),
                ce=float(
                    ce
                    .detach()
                    .item()
                ),
                lovasz=float(
                    lovasz
                    .detach()
                    .item()
                ),
                base_lr=lrs.get(
                    "pretrained_decay",
                    args.base_lr,
                ),
                new_lr=lrs.get(
                    "new_decay",
                    args.new_lr,
                ),
                p_corrupt=p_corrupt,
            )

            if (
                batch_index
                % args.log_every
                == 0
                or batch_index
                + 1
                == len(loader)
            ):
                append_jsonl(
                    log_path,
                    {
                        "event": (
                            "batch_log"
                        ),
                        "time_local": (
                            base.format_local_datetime(
                                base.local_now()
                            )
                        ),
                        "epoch": (
                            epoch
                            + 1
                        ),
                        "batch": (
                            batch_index
                            + 1
                        ),
                        "global_step": (
                            global_step
                        ),
                        "loss": float(
                            total_loss
                            .detach()
                            .item()
                        ),
                        "ce": float(
                            ce
                            .detach()
                            .item()
                        ),
                        "lovasz": float(
                            lovasz
                            .detach()
                            .item()
                        ),
                        "corruption_probability": (
                            p_corrupt
                        ),
                        "lr_groups": (
                            lrs
                        ),
                        "batch_seconds": (
                            batch_seconds
                        ),
                    },
                )

            del (
                details,
                logits,
                raw_logits,
                total_loss,
                loss_for_backward,
                rgb,
                rgb_model,
                labels,
            )

        progress.epoch_boundary()

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        epoch_record = {
            "event": (
                "epoch_done"
            ),
            "time_local": (
                base.format_local_datetime(
                    base.local_now()
                )
            ),
            "epoch": (
                epoch
                + 1
            ),
            "global_step": (
                global_step
            ),
            "optimizer_updates_this_epoch": (
                updates
            ),
            "amp_skipped_updates": (
                amp_skips
            ),
            "mini_batches": (
                mini_batches
            ),
            "valid_pixels": (
                total_valid
            ),
            "train_ce": (
                ce_numerator
                / total_valid
            ),
            "train_lovasz_mean": (
                lovasz_sum
                / mini_batches
            ),
            "train_total_loss_mean": (
                total_loss_sum
                / mini_batches
            ),
            "corruption_probability": (
                p_corrupt
            ),
            "corruption_family_counts": (
                family_counts
            ),
            "mean_degraded_severity": (
                severity_sum
                / degraded_samples
                if degraded_samples
                > 0
                else 0.0
            ),
            "last_grad_norm_before_clip": (
                last_grad_norm
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
            "epoch_duration": (
                base.format_duration(
                    epoch_seconds
                )
            ),
        }

        append_jsonl(
            log_path,
            epoch_record,
        )

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            protocol=protocol,
            model_meta=model_meta,
        )

        atomic_torch_save(
            payload,
            checkpoint_dir
            / "latest.pt",
        )

        if (
            args.save_every
            > 0
            and (
                epoch
                + 1
            )
            % args.save_every
            == 0
        ):
            atomic_torch_save(
                payload,
                checkpoint_dir
                / f"epoch_{epoch + 1:03d}.pt",
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/{args.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"Total={epoch_record['train_total_loss_mean']:.6f} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"families={family_counts} | "
            f"epoch={base.format_duration(epoch_seconds)} | "
            f"global_step={global_step}",
            flush=True,
        )

    final = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=(
            args.epochs
            - 1
        ),
        global_step=global_step,
        protocol=protocol,
        model_meta=model_meta,
    )

    atomic_torch_save(
        final,
        checkpoint_dir
        / "final.pt",
    )

    progress.finish(
        output_dir=output_dir
    )

    total_seconds = (
        time.time()
        - training_start
    )

    print("=" * 118)
    print(
        f"[finished] {MODEL_NAME}"
    )
    print(
        f"[finished] duration   : "
        f"{base.format_duration(total_seconds)}"
    )
    print(
        f"[finished] checkpoint : "
        f"{checkpoint_dir / 'final.pt'}"
    )
    print(
        f"[finished] protocol   : "
        f"{output_dir / 'protocol.json'}"
    )
    print(
        f"[finished] progress   : "
        f"{output_dir / 'progress.json'}"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
