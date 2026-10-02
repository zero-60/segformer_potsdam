#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the revised NIR experiments with JOINT RGB+NIR Robust-4 degradation.

Variants
--------
M2-Joint:
    python tools/train_model_b2_rgbnir_joint_robust4.py --variant fixed

M3-Joint:
    python tools/train_model_b2_rgbnir_joint_robust4.py --variant darf

Why this script exists
----------------------
The previous NIR experiments degraded RGB while keeping NIR perfectly clean.
That protocol can overestimate the value of NIR because the auxiliary modality
is unrealistically reliable.

The revised protocol applies the same degradation event to BOTH RGB and NIR.

Shared Joint Robust-4 protocol
------------------------------
If a sample is selected for corruption:
    - RGB and NIR share the same degradation family.
    - RGB and NIR share the same base severity.

Families:
    1) Gaussian noise:
       same sigma, independent random noise realization in RGB and NIR.

    2) Gaussian blur:
       same blur sigma in RGB and NIR.

    3) Underexposure:
       same attenuation factor in RGB and NIR.

    4) Atmospheric fog:
       same low-frequency fog field.
       NIR is degraded too, but less strongly by default:
           t_nir = 1 - 0.65 * (1 - t_rgb)
       This is a controlled approximation, not a calibrated atmosphere model.

M2-Joint
--------
    F_i = F_RGB_i + 0.5 * Adapter_i(F_NIR_i)

M3-Joint
--------
    F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)

Important change to DARF training
---------------------------------
The OLD gate supervision:
    more degradation severity -> larger NIR gate

is invalid once NIR itself is also degraded.

Therefore M3-Joint:
    - removes severity-based Gate BCE supervision
    - initializes all DARF gates at 0.5
    - learns dynamic gates only through the segmentation objective

This creates a cleaner comparison:
    M2-Joint: fixed g = 0.5
    M3-Joint: learned dynamic g, initialized at 0.5

Both use the SAME:
    - dual B2 encoders
    - NIR residual adapters
    - joint degradation distribution
    - CE + 0.5 Lovasz semantic loss
    - optimizer schedule
    - dataset/sampler protocol

Outputs
-------
Fixed:
    outputs/training/b2_rgbnir_fixed_joint_robust4/

DARF:
    outputs/training/b2_darf_joint_robust4/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model_b2_rgb as progress_base
import train_model_d_darf_b2 as darf_base

from data_pipeline.potsdam_dataloader import (
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from models.segformer_b2_darf import (
    GATE_TYPE,
    IGNORE_INDEX,
    NUM_SCALES,
    build_model_d_darf_b2,
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
from train_model_b2_rgbnir_fixed_robust4 import (
    FIXED_NIR_STRENGTH,
    build_model_b2_rgbnir_fixed,
    build_optimizer as build_fixed_optimizer,
)
from joint_multimodal_robust4 import (
    DEFAULT_NIR_FOG_SCATTER_RATIO,
    ROBUST4_FAMILIES,
    degrade_rgb_nir_batch_joint_robust4,
)


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Joint RGB+NIR Robust-4 Protocol v1"

FIXED_MODEL_ID = "B2_RGBNIR_FIXED_JOINT_ROBUST4"
FIXED_MODEL_NAME = "SegFormer-B2 RGB+NIR Fixed Joint Robust-4"

DARF_MODEL_ID = "B2_DARF_JOINT_ROBUST4"
DARF_MODEL_NAME = "SegFormer-B2 DARF Joint Robust-4"

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

DARF_INITIAL_GATE = 0.50


# =============================================================================
# Generic helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def write_json_atomic(
    path: Path,
    obj: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

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

    os.replace(
        tmp,
        path,
    )


def append_jsonl(
    path: Path,
    obj: Mapping[str, Any],
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
                obj,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def default_output_dir(
    variant: str,
) -> Path:
    if variant == "fixed":
        return (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_rgbnir_fixed_joint_robust4"
        )

    if variant == "darf":
        return (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "b2_darf_joint_robust4"
        )

    raise ValueError(
        variant
    )


def identity(
    variant: str,
) -> Tuple[
    str,
    str,
]:
    if variant == "fixed":
        return (
            FIXED_MODEL_ID,
            FIXED_MODEL_NAME,
        )

    if variant == "darf":
        return (
            DARF_MODEL_ID,
            DARF_MODEL_NAME,
        )

    raise ValueError(
        variant
    )


def reset_darf_gate_prior(
    model,
    *,
    prior: float,
) -> None:
    """
    Reinitialize only the final gate projections.

    The original DARF architecture starts at g=0.05 because old training used
    explicit severity supervision.  Under the revised joint-degradation
    experiment there is no reason to prefer almost-zero NIR at initialization.

    M3-Joint therefore starts from g=0.5, exactly matching the fixed M2-Joint
    coefficient before dynamic learning begins.

    Residual adapters remain exact-zero output, so the fused network still
    starts on the pretrained RGB path at step 0.
    """
    prior = float(
        prior
    )

    if not (
        0.0
        < prior
        < 1.0
    ):
        raise ValueError(
            "Gate prior must be in (0,1)."
        )

    prior_logit = math.log(
        prior
        / (
            1.0
            - prior
        )
    )

    if not hasattr(
        model,
        "quality_gates",
    ):
        raise RuntimeError(
            "DARF model has no quality_gates."
        )

    with torch.no_grad():
        for gate in model.quality_gates:
            gate.fc2.weight.zero_()
            gate.fc2.bias.fill_(
                prior_logit
            )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train M2/M3 with joint RGB+NIR Robust-4 degradation."
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
            "fixed = M2-Joint, darf = M3-Joint"
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )

    p.add_argument(
        "--base-lr",
        type=float,
        default=DEFAULT_BASE_LR,
    )

    p.add_argument(
        "--new-lr",
        type=float,
        default=DEFAULT_NEW_LR,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    p.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    p.add_argument(
        "--lovasz-weight",
        type=float,
        default=DEFAULT_LOVASZ_WEIGHT,
    )

    p.add_argument(
        "--clean-warmup-epochs",
        type=int,
        default=DEFAULT_CLEAN_WARMUP_EPOCHS,
    )

    p.add_argument(
        "--corruption-ramp-epochs",
        type=int,
        default=DEFAULT_CORRUPTION_RAMP_EPOCHS,
    )

    p.add_argument(
        "--max-corruption-prob",
        type=float,
        default=DEFAULT_MAX_CORRUPTION_PROB,
    )

    p.add_argument(
        "--nir-fog-scatter-ratio",
        type=float,
        default=DEFAULT_NIR_FOG_SCATTER_RATIO,
        help=(
            "Relative NIR fog attenuation. "
            "1.0 means same fog attenuation as RGB; "
            "values below 1 make NIR less affected but still degraded."
        ),
    )

    p.add_argument(
        "--darf-initial-gate",
        type=float,
        default=DARF_INITIAL_GATE,
        help=(
            "Initial M3-Joint gate prior. "
            "Keep 0.5 for the formal revised experiment."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--data-cache-dir",
        type=Path,
        default=DEFAULT_DATA_CACHE_DIR,
    )

    p.add_argument(
        "--rebuild-data-cache",
        action="store_true",
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
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default depends on --variant."
        ),
    )

    p.add_argument(
        "--resume",
        default="",
        help=(
            'Checkpoint path, or "auto" for <output-dir>/checkpoints/latest.pt.'
        ),
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=0,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
    )

    p.add_argument(
        "--eta-warmup-batches",
        type=int,
        default=DEFAULT_ETA_WARMUP_BATCHES,
    )

    p.add_argument(
        "--eta-ema-alpha",
        type=float,
        default=DEFAULT_ETA_EMA_ALPHA,
    )

    p.add_argument(
        "--no-progress",
        action="store_true",
    )

    x = p.parse_args()

    for name, value in (
        (
            "epochs",
            x.epochs,
        ),
        (
            "batch-size",
            x.batch_size,
        ),
        (
            "grad-accum-steps",
            x.grad_accum_steps,
        ),
        (
            "base-lr",
            x.base_lr,
        ),
        (
            "new-lr",
            x.new_lr,
        ),
        (
            "grad-clip",
            x.grad_clip,
        ),
        (
            "log-every",
            x.log_every,
        ),
        (
            "progress-every",
            x.progress_every,
        ),
        (
            "eta-warmup-batches",
            x.eta_warmup_batches,
        ),
    ):
        if value <= 0:
            p.error(
                f"--{name} must be > 0"
            )

    if x.weight_decay < 0:
        p.error(
            "--weight-decay must be >= 0"
        )

    if x.warmup_steps < 0:
        p.error(
            "--warmup-steps must be >= 0"
        )

    if x.lovasz_weight < 0:
        p.error(
            "--lovasz-weight must be >= 0"
        )

    if (
        x.clean_warmup_epochs
        < 0
        or x.corruption_ramp_epochs
        < 0
    ):
        p.error(
            "warm-up/ramp epochs must be >= 0"
        )

    if not (
        0.0
        <= x.max_corruption_prob
        <= 1.0
    ):
        p.error(
            "--max-corruption-prob must be in [0,1]"
        )

    if not (
        0.0
        < x.nir_fog_scatter_ratio
        <= 1.0
    ):
        p.error(
            "--nir-fog-scatter-ratio must be in (0,1]"
        )

    if not (
        0.0
        < x.darf_initial_gate
        < 1.0
    ):
        p.error(
            "--darf-initial-gate must be in (0,1)"
        )

    if not (
        0.0
        < x.eta_ema_alpha
        <= 1.0
    ):
        p.error(
            "--eta-ema-alpha must be in (0,1]"
        )

    return x


# =============================================================================
# Model / optimizer
# =============================================================================

def build_model_and_optimizer(
    *,
    variant: str,
    base_lr: float,
    new_lr: float,
    weight_decay: float,
    darf_initial_gate: float,
):
    if variant == "fixed":
        model, model_meta = (
            build_model_b2_rgbnir_fixed(
                PROJECT_ROOT,
                fixed_strength=(
                    FIXED_NIR_STRENGTH
                ),
            )
        )

        optimizer, optimizer_summary = (
            build_fixed_optimizer(
                model,
                base_lr=(
                    base_lr
                ),
                new_lr=(
                    new_lr
                ),
                weight_decay=(
                    weight_decay
                ),
            )
        )

        return (
            model,
            model_meta,
            optimizer,
            optimizer_summary,
        )

    model, model_meta = (
        build_model_d_darf_b2(
            PROJECT_ROOT
        )
    )

    reset_darf_gate_prior(
        model,
        prior=(
            darf_initial_gate
        ),
    )

    optimizer, optimizer_summary = (
        darf_base.build_optimizer(
            model,
            base_lr=(
                base_lr
            ),
            new_lr=(
                new_lr
            ),
            weight_decay=(
                weight_decay
            ),
        )
    )

    return (
        model,
        model_meta,
        optimizer,
        optimizer_summary,
    )


# =============================================================================
# Protocol / checkpoint
# =============================================================================

def build_protocol(
    *,
    x,
    model_id: str,
    model_name: str,
    model_meta,
    optimizer_summary,
    updates_per_epoch: int,
    total_updates: int,
    warmup: int,
    device,
    amp: bool,
    cache: Path,
    checkpointing_status,
    nir_mean: float,
    nir_std: float,
):
    fixed = (
        x.variant
        == "fixed"
    )

    return {
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "module_version": (
            MODULE_VERSION
        ),
        "model_id": (
            model_id
        ),
        "model_name": (
            model_name
        ),
        "variant": (
            x.variant
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "nir_used": True,
        "dual_encoder": True,
        "fusion": (
            "fixed RGB-anchored NIR residual"
            if fixed
            else "dynamic DARF RGB-anchored NIR residual"
        ),
        "fusion_rule": (
            "F_i = F_RGB_i + 0.5 * Adapter_i(F_NIR_i)"
            if fixed
            else "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
        ),
        "fixed_nir_strength": (
            float(
                FIXED_NIR_STRENGTH
            )
            if fixed
            else None
        ),
        "quality_gate": (
            not fixed
        ),
        "gate_type": (
            None
            if fixed
            else GATE_TYPE
        ),
        "gate_supervision": False,
        "gate_training": (
            None
            if fixed
            else {
                "initial_gate": float(
                    x.darf_initial_gate
                ),
                "auxiliary_gate_bce": False,
                "learning_signal": (
                    "semantic segmentation loss only"
                ),
                "reason": (
                    "Once NIR is also degraded, absolute degradation severity "
                    "does not define the optimal NIR residual strength."
                ),
            }
        ),
        "regime": (
            "joint_robust4"
        ),
        "epochs": (
            x.epochs
        ),
        "batch_size": (
            x.batch_size
        ),
        "grad_accum_steps": (
            x.grad_accum_steps
        ),
        "effective_batch_size_nominal": (
            x.batch_size
            * x.grad_accum_steps
        ),
        "base_lr": (
            x.base_lr
        ),
        "new_lr": (
            x.new_lr
        ),
        "weight_decay": (
            x.weight_decay
        ),
        "warmup_steps": (
            warmup
        ),
        "grad_clip": (
            x.grad_clip
        ),
        "loss": {
            "ce_weight": 1.0,
            "lovasz_weight": (
                x.lovasz_weight
            ),
            "gate_bce_weight": 0.0,
        },
        "joint_corruption_training": {
            "enabled": True,
            "version": (
                "Joint RGB+NIR Robust-4 v1"
            ),
            "clean_warmup_epochs": (
                x.clean_warmup_epochs
            ),
            "ramp_epochs": (
                x.corruption_ramp_epochs
            ),
            "max_probability": (
                x.max_corruption_prob
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
            "event_coupling": (
                "RGB and NIR share family and base severity"
            ),
            "noise": {
                "rgb_sigma_255": [
                    5.0,
                    50.0,
                ],
                "nir_sigma_255": [
                    5.0,
                    50.0,
                ],
                "random_realizations": (
                    "independent by modality/channel"
                ),
            },
            "blur_sigma": [
                0.5,
                4.0,
            ],
            "underexposure_alpha": [
                0.94,
                0.40,
            ],
            "fog": {
                "rgb_mean_transmission": (
                    "0.90 - 0.50 * severity"
                ),
                "shared_spatial_field": True,
                "nir_transmission": (
                    "1 - nir_fog_scatter_ratio * (1 - t_rgb)"
                ),
                "nir_fog_scatter_ratio": float(
                    x.nir_fog_scatter_ratio
                ),
                "physical_status": (
                    "controlled approximation; not radiative-transfer calibration"
                ),
            },
            "rgb": (
                "degraded"
            ),
            "nir": (
                "degraded"
            ),
            "labels": (
                "unchanged"
            ),
            "normalization": {
                "rgb": (
                    "ImageNet mean/std"
                ),
                "nir_mean": float(
                    nir_mean
                ),
                "nir_std": float(
                    nir_std
                ),
            },
            "implementation_source": (
                "tools/joint_multimodal_robust4.py"
            ),
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
            x.seed
        ),
        "data_seed": int(
            DATA_SEED
        ),
        "amp": (
            amp
        ),
        "gradient_checkpointing_requested": (
            x.gradient_checkpointing
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
    }


def checkpoint_payload(
    *,
    model_id: str,
    model_name: str,
    variant: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    protocol,
    model_meta,
):
    return {
        "format_version": 2,
        "model_id": (
            model_id
        ),
        "model_name": (
            model_name
        ),
        "variant": (
            variant
        ),
        "regime": (
            "joint_robust4"
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
) -> Optional[
    Path
]:
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
    expected_model_id: str,
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
    ) != expected_model_id:
        raise RuntimeError(
            "Resume checkpoint model_id mismatch."
        )

    if checkpoint.get(
        "regime"
    ) != "joint_robust4":
        raise RuntimeError(
            "Resume checkpoint is not joint_robust4."
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
    x = parse_args()

    seed_everything(
        x.seed
    )

    model_id, model_name = identity(
        x.variant
    )

    output_dir = (
        resolve(
            x.output_dir
        )
        if x.output_dir
        is not None
        else default_output_dir(
            x.variant
        )
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
        x.data_cache_dir
    )

    print("=" * 132)
    print(
        model_name
    )
    print("=" * 132)
    print(
        f"model id       : {model_id}"
    )
    print(
        f"variant        : {x.variant}"
    )
    print(
        "regime         : joint_robust4"
    )
    print(
        "modalities     : RGB + NIR"
    )
    print(
        "corruption     : BOTH RGB and NIR degraded"
    )
    print(
        f"NIR fog ratio  : {x.nir_fog_scatter_ratio:.3f}"
    )

    if x.variant == "fixed":
        print(
            f"fusion         : fixed g={FIXED_NIR_STRENGTH:.3f}"
        )
    else:
        print(
            f"fusion         : dynamic DARF, initial g={x.darf_initial_gate:.3f}"
        )
        print(
            "gate BCE       : disabled"
        )

    print(
        f"output         : {output_dir}"
    )
    print(
        f"start local    : "
        f"{progress_base.format_local_datetime(progress_base.local_now())}"
    )
    print("=" * 132)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    dataset = (
        CachedPotsdamTrainDataset(
            PROJECT_ROOT,
            epoch=0,
            cache_root=(
                cache
            ),
        )
    )

    dataset.prepare_cache(
        rebuild=(
            x.rebuild_data_cache
        )
    )

    nir_mean = float(
        dataset.spec.nir_mean
    )

    nir_std = float(
        dataset.spec.nir_std
    )

    if (
        not math.isfinite(
            nir_mean
        )
        or not math.isfinite(
            nir_std
        )
        or nir_std
        <= 0
    ):
        raise RuntimeError(
            f"Invalid frozen NIR normalization: mean={nir_mean}, std={nir_std}"
        )

    print(
        f"NIR norm       : mean={nir_mean:.9f}, std={nir_std:.9f}"
    )

    sampler = (
        DeterministicEpochSampler(
            dataset,
            seed=(
                DATA_SEED
            ),
            epoch=0,
        )
    )

    loader = (
        build_train_dataloader(
            dataset,
            sampler,
            batch_size=(
                x.batch_size
            ),
            num_workers=(
                x.num_workers
            ),
            pin_memory=(
                x.pin_memory
            ),
        )
    )

    updates_per_epoch = math.ceil(
        len(
            loader
        )
        / x.grad_accum_steps
    )

    total_updates = (
        updates_per_epoch
        * x.epochs
    )

    warmup = min(
        x.warmup_steps,
        max(
            0,
            total_updates
            - 1,
        ),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    device = (
        progress_base
        .get_device(
            x.device
        )
    )

    (
        model,
        model_meta,
        optimizer,
        optimizer_summary,
    ) = build_model_and_optimizer(
        variant=(
            x.variant
        ),
        base_lr=(
            x.base_lr
        ),
        new_lr=(
            x.new_lr
        ),
        weight_decay=(
            x.weight_decay
        ),
        darf_initial_gate=(
            x.darf_initial_gate
        ),
    )

    model_meta = dict(
        model_meta
    )

    model_meta[
        "experiment_model_id"
    ] = model_id

    model_meta[
        "experiment_model_name"
    ] = model_name

    model_meta[
        "experiment_protocol_version"
    ] = PROTOCOL_VERSION

    checkpointing_status = {
        "rgb_encoder": False,
        "nir_encoder": False,
    }

    if x.gradient_checkpointing:
        checkpointing_status = (
            model
            .enable_gradient_checkpointing()
        )

    model.to(
        device
    )

    scheduler = (
        build_poly_scheduler(
            optimizer=(
                optimizer
            ),
            warmup_steps=(
                warmup
            ),
            total_steps=(
                total_updates
            ),
            power=1.0,
        )
    )

    amp = (
        device.type
        == "cuda"
        and not x.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            amp
        ),
    )

    protocol = build_protocol(
        x=x,
        model_id=(
            model_id
        ),
        model_name=(
            model_name
        ),
        model_meta=(
            model_meta
        ),
        optimizer_summary=(
            optimizer_summary
        ),
        updates_per_epoch=(
            updates_per_epoch
        ),
        total_updates=(
            total_updates
        ),
        warmup=(
            warmup
        ),
        device=(
            device
        ),
        amp=(
            amp
        ),
        cache=(
            cache
        ),
        checkpointing_status=(
            checkpointing_status
        ),
        nir_mean=(
            nir_mean
        ),
        nir_std=(
            nir_std
        ),
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
        f"epochs          : {x.epochs}"
    )
    print(
        f"batches/epoch   : {len(loader)}"
    )
    print(
        f"updates/epoch   : {updates_per_epoch}"
    )
    print(
        f"total updates   : {total_updates}"
    )
    print(
        f"effective batch : "
        f"{x.batch_size * x.grad_accum_steps}"
    )
    print(
        f"loss            : CE + {x.lovasz_weight} * Lovasz"
    )
    print(
        f"max p_corrupt   : {x.max_corruption_prob}"
    )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0
    global_step = 0

    resume_path = resolve_resume(
        x.resume,
        checkpoint_dir,
    )

    if resume_path is not None:
        (
            start_epoch,
            global_step,
        ) = load_resume(
            path=(
                resume_path
            ),
            expected_model_id=(
                model_id
            ),
            model=(
                model
            ),
            optimizer=(
                optimizer
            ),
            scheduler=(
                scheduler
            ),
            scaler=(
                scaler
            ),
        )

        print(
            f"[resume] {resume_path} | "
            f"start_epoch={start_epoch} | "
            f"global_step={global_step}",
            flush=True,
        )

    if start_epoch >= x.epochs:
        print(
            "[done] requested epochs already completed."
        )
        return

    # -------------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------------

    progress = (
        progress_base
        .LiveTrainingProgress(
            total_epochs=(
                x.epochs
            ),
            batches_per_epoch=(
                len(
                    loader
                )
            ),
            start_epoch=(
                start_epoch
            ),
            progress_every=(
                x.progress_every
            ),
            eta_warmup_batches=(
                x.eta_warmup_batches
            ),
            ema_alpha=(
                x.eta_ema_alpha
            ),
            output_path=(
                output_dir
                / "progress.json"
            ),
            device=(
                device
            ),
            enabled=(
                not x.no_progress
            ),
        )
    )

    log_path = (
        output_dir
        / "train_log.jsonl"
    )

    training_start = (
        time.time()
    )

    first_batch_checked = False

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        x.epochs,
    ):
        epoch_start = (
            time.time()
        )

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
            darf_base
            .corruption_probability(
                epoch,
                clean_warmup_epochs=(
                    x.clean_warmup_epochs
                ),
                ramp_epochs=(
                    x.corruption_ramp_epochs
                ),
                maximum=(
                    x.max_corruption_prob
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
            "underexposure": 0,
            "fog": 0,
        }

        severity_sum = 0.0
        degraded_samples = 0

        fog_t_rgb_sum = 0.0
        fog_t_nir_sum = 0.0
        fog_samples = 0

        gate_sum = (
            torch.zeros(
                NUM_SCALES,
                dtype=torch.float64,
            )
            if x.variant
            == "darf"
            else None
        )

        gate_sq_sum = (
            torch.zeros(
                NUM_SCALES,
                dtype=torch.float64,
            )
            if x.variant
            == "darf"
            else None
        )

        gate_samples = 0

        residual_sum = (
            torch.zeros(
                NUM_SCALES,
                dtype=torch.float64,
            )
            if x.variant
            == "fixed"
            else None
        )

        residual_batches = 0

        accumulation_target = (
            x.grad_accum_steps
        )

        for batch_index, batch in enumerate(
            loader
        ):
            batch_started = (
                time.monotonic()
            )

            if (
                batch_index
                % x.grad_accum_steps
                == 0
            ):
                remaining = (
                    len(
                        loader
                    )
                    - batch_index
                )

                accumulation_target = min(
                    x.grad_accum_steps,
                    remaining,
                )

            rgb = (
                batch[
                    "rgb"
                ]
                .to(
                    device,
                    non_blocking=(
                        x.pin_memory
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
                        x.pin_memory
                    ),
                )
            )

            labels = (
                batch[
                    "labels"
                ]
                .to(
                    device,
                    non_blocking=(
                        x.pin_memory
                    ),
                )
                .long()
            )

            if not first_batch_checked:
                if (
                    rgb.ndim
                    != 4
                    or rgb.shape[
                        1
                    ]
                    != 3
                ):
                    raise RuntimeError(
                        f"Bad RGB batch: {tuple(rgb.shape)}"
                    )

                if (
                    nir.ndim
                    != 4
                    or nir.shape[
                        1
                    ]
                    != 1
                ):
                    raise RuntimeError(
                        f"Bad NIR batch: {tuple(nir.shape)}"
                    )

                if (
                    rgb.shape[
                        0
                    ]
                    != nir.shape[
                        0
                    ]
                    or rgb.shape[
                        -2:
                    ]
                    != nir.shape[
                        -2:
                    ]
                ):
                    raise RuntimeError(
                        "RGB/NIR batch alignment failed."
                    )

                if labels.ndim != 3:
                    raise RuntimeError(
                        f"Bad labels batch: {tuple(labels.shape)}"
                    )

                first_batch_checked = True

                print(
                    "[first batch] PASS | "
                    f"RGB={tuple(rgb.shape)} | "
                    f"NIR={tuple(nir.shape)} | "
                    f"labels={tuple(labels.shape)}",
                    flush=True,
                )

            (
                rgb_model,
                nir_model,
                counts,
                severities,
                corruption_diag,
            ) = (
                degrade_rgb_nir_batch_joint_robust4(
                    rgb,
                    nir,
                    probability=(
                        p_corrupt
                    ),
                    nir_mean=(
                        nir_mean
                    ),
                    nir_std=(
                        nir_std
                    ),
                    nir_fog_scatter_ratio=(
                        x.nir_fog_scatter_ratio
                    ),
                )
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

            current_fog_samples = int(
                corruption_diag[
                    "fog_samples"
                ]
            )

            if current_fog_samples > 0:
                fog_samples += current_fog_samples

                fog_t_rgb_sum += (
                    float(
                        corruption_diag[
                            "fog_mean_t_rgb"
                        ]
                    )
                    * current_fog_samples
                )

                fog_t_nir_sum += (
                    float(
                        corruption_diag[
                            "fog_mean_t_nir"
                        ]
                    )
                    * current_fog_samples
                )

            with torch.autocast(
                device_type=(
                    device.type
                ),
                dtype=torch.float16,
                enabled=(
                    amp
                ),
            ):
                details = model(
                    rgb_model,
                    nir_model,
                    return_details=True,
                )

                logits = (
                    details[
                        "logits"
                    ]
                )

                raw_logits = (
                    details[
                        "raw_logits"
                    ]
                )

                ce = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=(
                        IGNORE_INDEX
                    ),
                )

                lovasz = (
                    darf_base
                    .lovasz_softmax(
                        raw_logits,
                        labels,
                    )
                )

                total_loss = (
                    ce
                    + x.lovasz_weight
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
                    f"Non-finite loss at epoch={epoch + 1}, "
                    f"batch={batch_index + 1}"
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            valid_pixels = int(
                (
                    labels
                    != IGNORE_INDEX
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

            if x.variant == "darf":
                gate = (
                    details[
                        "nir_gate_strength"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .double()
                )

                gate_sum += gate.sum(
                    dim=0
                )

                gate_sq_sum += (
                    gate
                    * gate
                ).sum(
                    dim=0
                )

                gate_samples += int(
                    gate.shape[
                        0
                    ]
                )

            else:
                residual = (
                    details[
                        "residual_abs_mean"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .double()
                )

                residual_sum += (
                    residual
                )

                residual_batches += 1

            should_step = (
                (
                    (
                        batch_index
                        + 1
                    )
                    % x.grad_accum_steps
                    == 0
                )
                or (
                    batch_index
                    + 1
                    == len(
                        loader
                    )
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
                        x.grad_clip,
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
                    or after
                    >= before
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
                    str(
                        index
                    ),
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
                epoch_zero_based=(
                    epoch
                ),
                batch_zero_based=(
                    batch_index
                ),
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
                base_lr=(
                    lrs.get(
                        "pretrained_decay",
                        x.base_lr,
                    )
                ),
                new_lr=(
                    lrs.get(
                        "new_decay",
                        x.new_lr,
                    )
                ),
                p_corrupt=(
                    p_corrupt
                ),
            )

            if (
                batch_index
                % x.log_every
                == 0
                or batch_index
                + 1
                == len(
                    loader
                )
            ):
                extra = {}

                if x.variant == "darf":
                    extra[
                        "gate_strength_mean"
                    ] = [
                        float(
                            v
                        )
                        for v in (
                            details[
                                "nir_gate_strength"
                            ]
                            .detach()
                            .float()
                            .mean(
                                dim=0
                            )
                            .cpu()
                            .tolist()
                        )
                    ]

                else:
                    extra[
                        "residual_abs_mean"
                    ] = [
                        float(
                            v
                        )
                        for v in (
                            details[
                                "residual_abs_mean"
                            ]
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                        )
                    ]

                append_jsonl(
                    log_path,
                    {
                        "event": (
                            "batch_log"
                        ),
                        "time_local": (
                            progress_base
                            .format_local_datetime(
                                progress_base
                                .local_now()
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
                        "family_counts_running": dict(
                            family_counts
                        ),
                        "lr_groups": (
                            lrs
                        ),
                        "batch_seconds": (
                            batch_seconds
                        ),
                        **extra,
                    },
                )

            del (
                details,
                logits,
                raw_logits,
                total_loss,
                loss_for_backward,
                rgb,
                nir,
                rgb_model,
                nir_model,
                labels,
            )

        progress.epoch_boundary()

        if (
            total_valid
            <= 0
            or mini_batches
            <= 0
        ):
            raise RuntimeError(
                "Epoch contains no valid data."
            )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        epoch_record: Dict[
            str,
            Any,
        ] = {
            "event": (
                "epoch_done"
            ),
            "time_local": (
                progress_base
                .format_local_datetime(
                    progress_base
                    .local_now()
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
            "fog_mean_t_rgb": (
                fog_t_rgb_sum
                / fog_samples
                if fog_samples
                > 0
                else None
            ),
            "fog_mean_t_nir": (
                fog_t_nir_sum
                / fog_samples
                if fog_samples
                > 0
                else None
            ),
            "last_grad_norm_before_clip": (
                last_grad_norm
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
            "epoch_duration": (
                progress_base
                .format_duration(
                    epoch_seconds
                )
            ),
        }

        if x.variant == "darf":
            if gate_samples <= 0:
                raise RuntimeError(
                    "No DARF gate samples collected."
                )

            gate_mean = (
                gate_sum
                / gate_samples
            )

            gate_second = (
                gate_sq_sum
                / gate_samples
            )

            gate_std = torch.sqrt(
                torch.clamp(
                    gate_second
                    - gate_mean
                    * gate_mean,
                    min=0.0,
                )
            )

            epoch_record[
                "gate_nir_strength_mean_by_scale"
            ] = [
                float(
                    value
                )
                for value in gate_mean.tolist()
            ]

            epoch_record[
                "gate_nir_strength_std_by_scale"
            ] = [
                float(
                    value
                )
                for value in gate_std.tolist()
            ]

        else:
            if residual_batches <= 0:
                raise RuntimeError(
                    "No fixed-fusion residual diagnostics collected."
                )

            residual_mean = (
                residual_sum
                / residual_batches
            )

            epoch_record[
                "residual_abs_mean_by_scale"
            ] = [
                float(
                    value
                )
                for value in residual_mean.tolist()
            ]

        append_jsonl(
            log_path,
            epoch_record,
        )

        payload = checkpoint_payload(
            model_id=(
                model_id
            ),
            model_name=(
                model_name
            ),
            variant=(
                x.variant
            ),
            model=(
                model
            ),
            optimizer=(
                optimizer
            ),
            scheduler=(
                scheduler
            ),
            scaler=(
                scaler
            ),
            epoch=(
                epoch
            ),
            global_step=(
                global_step
            ),
            protocol=(
                protocol
            ),
            model_meta=(
                model_meta
            ),
        )

        atomic_torch_save(
            payload,
            checkpoint_dir
            / "latest.pt",
        )

        if (
            x.save_every
            > 0
            and (
                epoch
                + 1
            )
            % x.save_every
            == 0
        ):
            atomic_torch_save(
                payload,
                checkpoint_dir
                / (
                    f"epoch_"
                    f"{epoch + 1:03d}.pt"
                ),
            )

        if x.variant == "darf":
            diagnostic_text = (
                " | g="
                + str(
                    [
                        round(
                            float(
                                value
                            ),
                            4,
                        )
                        for value in epoch_record[
                            "gate_nir_strength_mean_by_scale"
                        ]
                    ]
                )
            )
        else:
            diagnostic_text = (
                " | residual="
                + str(
                    [
                        round(
                            float(
                                value
                            ),
                            5,
                        )
                        for value in epoch_record[
                            "residual_abs_mean_by_scale"
                        ]
                    ]
                )
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/{x.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"Total={epoch_record['train_total_loss_mean']:.6f} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"families={family_counts}"
            f"{diagnostic_text} | "
            f"epoch={progress_base.format_duration(epoch_seconds)} | "
            f"global_step={global_step}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    final = checkpoint_payload(
        model_id=(
            model_id
        ),
        model_name=(
            model_name
        ),
        variant=(
            x.variant
        ),
        model=(
            model
        ),
        optimizer=(
            optimizer
        ),
        scheduler=(
            scheduler
        ),
        scaler=(
            scaler
        ),
        epoch=(
            x.epochs
            - 1
        ),
        global_step=(
            global_step
        ),
        protocol=(
            protocol
        ),
        model_meta=(
            model_meta
        ),
    )

    atomic_torch_save(
        final,
        checkpoint_dir
        / "final.pt",
    )

    progress.finish(
        output_dir=(
            output_dir
        )
    )

    total_seconds = (
        time.time()
        - training_start
    )

    print("=" * 132)
    print(
        f"[finished] {model_name}"
    )
    print(
        f"[finished] duration   : "
        f"{progress_base.format_duration(total_seconds)}"
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
        f"[finished] train log  : "
        f"{log_path}"
    )
    print(
        f"[finished] progress   : "
        f"{output_dir / 'progress.json'}"
    )
    print("=" * 132)


if __name__ == "__main__":
    main()
