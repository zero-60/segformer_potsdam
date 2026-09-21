#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Model D / DARF-B2:
SegFormer-B2 + Degradation-Aware Residual Fusion.

This training recipe is designed to address the failure mode observed in the
previous clean-only convex Quality Gate:

- Stronger SegFormer-B2 backbone raises the capacity ceiling.
- RGB-anchored residual fusion preserves the pretrained RGB pathway.
- Lovasz-Softmax directly optimizes an IoU surrogate.
- Continuous RGB corruption augmentation teaches robustness.
- Explicit gate supervision teaches the gate to represent RGB quality rather
  than merely semantic content.
- Differential learning rates let new fusion/gate/classifier parameters learn
  faster while preserving the pretrained encoders/decoder.

Training corruption policy
--------------------------
The Dataset itself remains unchanged. RGB corruption is applied AFTER loading
the normalized training tensor by:
    1) inverse ImageNet normalization -> raw RGB in [0,1]
    2) synthetic RGB-only corruption
    3) ImageNet normalization again

NIR and labels remain unchanged.

After a short clean warm-up, each sample is degraded with probability 0.5.
The family is sampled uniformly from:
    Gaussian noise
    Gaussian blur
    RGB underexposure

Severity is CONTINUOUS rather than fixed to evaluation L1/L2/L3:
    noise sigma_255: 5 .. 50
    blur sigma:      0.5 .. 4.0
    exposure alpha:  0.94 .. 0.40

This means the model is not trained on only three discrete test points.

Loss
----
    L = CE
        + lambda_lovasz * LovaszSoftmax(raw logits, downsampled GT)
        + lambda_gate * BCE(g_NIR, quality_target)

Quality target
--------------
Clean:
    target g_NIR = 0.05

Degraded:
    target g_NIR = 0.15 + 0.80 * severity

The gate target is auxiliary supervision only. At inference, the network is
given RGB+NIR and predicts g_NIR from features; corruption type/severity is not
an input.

Default optimization
--------------------
- epochs: 120
- batch size: 1
- grad accumulation: 16
- effective batch: 16
- base pretrained LR: 3e-5
- new modules/classifier LR: 3e-4
- weight decay: 0.01
- warmup successful updates: 500
- polynomial linear decay
- AMP
- grad clip 1.0
- gradient checkpointing enabled where supported

Expected location
-----------------
tools/train_model_d_darf_b2.py

Run
---
python models/segformer_b2_darf.py
python tools/train_model_d_darf_b2.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline.potsdam_dataloader import (
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from models.segformer_b2_darf import (
    GATE_TYPE,
    IGNORE_INDEX,
    INITIAL_NIR_GATE,
    MODEL_ID,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_SCALES,
    build_model_d_darf_b2,
)

try:
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
except ImportError as exc:
    raise ImportError(
        "Could not import the successful tools/train_model_a_rgb.py."
    ) from exc


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_d_darf_b2"
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
DEFAULT_GATE_WEIGHT = 0.10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--base-lr", type=float, default=DEFAULT_BASE_LR)
    p.add_argument("--new-lr", type=float, default=DEFAULT_NEW_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--lovasz-weight", type=float, default=DEFAULT_LOVASZ_WEIGHT)
    p.add_argument("--gate-weight", type=float, default=DEFAULT_GATE_WEIGHT)

    p.add_argument("--clean-warmup-epochs", type=int, default=10)
    p.add_argument("--corruption-ramp-epochs", type=int, default=20)
    p.add_argument("--max-corruption-prob", type=float, default=0.50)

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--data-cache-dir", type=Path, default=DEFAULT_DATA_CACHE_DIR)
    p.add_argument("--rebuild-data-cache", action="store_true")

    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--resume",
        default="",
        help='Checkpoint path, or "auto" for latest.pt',
    )
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)

    x = p.parse_args()

    positive = [
        ("epochs", x.epochs),
        ("batch-size", x.batch_size),
        ("grad-accum-steps", x.grad_accum_steps),
        ("base-lr", x.base_lr),
        ("new-lr", x.new_lr),
        ("grad-clip", x.grad_clip),
        ("log-every", x.log_every),
    ]
    for name, value in positive:
        if value <= 0:
            p.error(f"--{name} must be > 0")

    if x.weight_decay < 0 or x.warmup_steps < 0:
        p.error("weight-decay/warmup-steps must be non-negative")

    if x.lovasz_weight < 0 or x.gate_weight < 0:
        p.error("loss weights must be non-negative")

    if not (0.0 <= x.max_corruption_prob <= 1.0):
        p.error("--max-corruption-prob must be in [0,1]")

    if x.clean_warmup_epochs < 0 or x.corruption_ramp_epochs < 0:
        p.error("warm-up/ramp epochs must be non-negative")

    return x


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def get_device(name: str) -> torch.device:
    d = torch.device(name)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return d


def resume_path(arg: str, ckpt_dir: Path) -> Optional[Path]:
    if not arg:
        return None

    if arg.lower() == "auto":
        p = ckpt_dir / "latest.pt"
    else:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()

    if not p.is_file():
        raise FileNotFoundError(p)

    return p


# ---------------------------------------------------------------------------
# Corruption augmentation
# ---------------------------------------------------------------------------

def corruption_probability(
    epoch_zero_based: int,
    *,
    clean_warmup_epochs: int,
    ramp_epochs: int,
    maximum: float,
) -> float:
    if epoch_zero_based < clean_warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return float(maximum)

    progress = (
        epoch_zero_based
        - clean_warmup_epochs
        + 1
    ) / float(ramp_epochs)

    return float(maximum) * min(max(progress, 0.0), 1.0)


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

    return (rgb01 - mean) / std


def _gaussian_blur_single(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    image: [3,H,W], float32 in [0,1]
    """
    sigma = float(sigma)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    kernel_size = min(kernel_size, 31)

    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2

    coords = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )

    kernel_1d = torch.exp(
        -(coords * coords)
        / (2.0 * sigma * sigma)
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
        .view(1, 1, kernel_size, kernel_size)
        .expand(3, 1, kernel_size, kernel_size)
        .contiguous()
    )

    x = image.unsqueeze(0)
    x = F.pad(
        x,
        (radius, radius, radius, radius),
        mode="reflect",
    )

    y = F.conv2d(
        x,
        weight,
        groups=3,
    )

    return y.squeeze(0)


def degrade_rgb_batch(
    rgb_normalized: torch.Tensor,
    *,
    probability: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, int],
    torch.Tensor,
]:
    """
    Returns:
        augmented normalized RGB
        gate target [B]
        family counts
        severity [B] (0 for clean)
    """
    with torch.autocast(
        device_type=rgb_normalized.device.type,
        enabled=False,
    ):
        rgb01 = _rgb_denormalize(
            rgb_normalized.float()
        )

        out = rgb01.clone()

        batch = int(
            rgb01.shape[0]
        )

        target = torch.full(
            (batch,),
            float(INITIAL_NIR_GATE),
            device=rgb01.device,
            dtype=torch.float32,
        )

        severity_out = torch.zeros(
            (batch,),
            device=rgb01.device,
            dtype=torch.float32,
        )

        counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "rgb_underexposure": 0,
        }

        for i in range(batch):
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
                counts["clean"] += 1
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
                    high=3,
                    size=(),
                    device=rgb01.device,
                ).item()
            )

            if family == 0:
                sigma_255 = (
                    5.0
                    + 45.0
                    * severity
                )

                noise = (
                    torch.randn_like(
                        out[i]
                    )
                    * (
                        sigma_255
                        / 255.0
                    )
                )

                out[i] = torch.clamp(
                    out[i]
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

                out[i] = _gaussian_blur_single(
                    out[i],
                    sigma,
                )

                counts[
                    "gaussian_blur"
                ] += 1

            else:
                alpha = (
                    1.0
                    - 0.60
                    * severity
                )

                out[i] = torch.clamp(
                    out[i]
                    * alpha,
                    0.0,
                    1.0,
                )

                counts[
                    "rgb_underexposure"
                ] += 1

            severity_out[
                i
            ] = severity

            # Explicit quality target:
            # 0.05 clean -> about 0.23 mild -> 0.95 severe.
            target[
                i
            ] = (
                0.15
                + 0.80
                * severity
            )

        out_normalized = (
            _rgb_normalize(
                out
            )
        )

    return (
        out_normalized,
        target,
        counts,
        severity_out,
    )


# ---------------------------------------------------------------------------
# Lovasz-Softmax
# ---------------------------------------------------------------------------

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = int(
        gt_sorted.numel()
    )

    gts = gt_sorted.sum()

    intersection = (
        gts
        - gt_sorted.float().cumsum(0)
    )

    union = (
        gts
        + (
            1.0
            - gt_sorted
        )
        .float()
        .cumsum(0)
    )

    jaccard = (
        1.0
        - intersection
        / torch.clamp(
            union,
            min=1.0,
        )
    )

    if p > 1:
        jaccard[
            1:p
        ] = (
            jaccard[
                1:p
            ]
            - jaccard[
                :-1
            ]
        )

    return jaccard


def lovasz_softmax(
    raw_logits: torch.Tensor,
    labels_full: torch.Tensor,
) -> torch.Tensor:
    """
    Lovasz on the 1/4-resolution raw SegFormer logits to keep sorting cost low.
    """
    labels = (
        F.interpolate(
            labels_full
            .unsqueeze(1)
            .float(),
            size=raw_logits.shape[-2:],
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )

    probabilities = torch.softmax(
        raw_logits.float(),
        dim=1,
    )

    probabilities = (
        probabilities
        .permute(
            0,
            2,
            3,
            1,
        )
        .reshape(
            -1,
            NUM_CLASSES,
        )
    )

    labels = labels.reshape(
        -1
    )

    valid = (
        labels
        != IGNORE_INDEX
    )

    probabilities = (
        probabilities[
            valid
        ]
    )

    labels = (
        labels[
            valid
        ]
    )

    if labels.numel() == 0:
        return (
            raw_logits.sum()
            * 0.0
        )

    losses = []

    for class_id in range(
        NUM_CLASSES
    ):
        foreground = (
            labels
            == class_id
        ).float()

        if foreground.sum().item() == 0:
            continue

        class_probability = (
            probabilities[
                :,
                class_id
            ]
        )

        errors = torch.abs(
            foreground
            - class_probability
        )

        (
            errors_sorted,
            permutation,
        ) = torch.sort(
            errors,
            descending=True,
        )

        foreground_sorted = (
            foreground[
                permutation
            ]
        )

        losses.append(
            torch.dot(
                errors_sorted,
                lovasz_grad(
                    foreground_sorted
                ),
            )
        )

    if not losses:
        return (
            raw_logits.sum()
            * 0.0
        )

    return torch.stack(
        losses
    ).mean()


# ---------------------------------------------------------------------------
# Optimizer / protocol / checkpoint
# ---------------------------------------------------------------------------

def build_optimizer(
    model: torch.nn.Module,
    *,
    base_lr: float,
    new_lr: float,
    weight_decay: float,
):
    base_decay = []
    base_no_decay = []
    new_decay = []
    new_no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        is_new = (
            name.startswith(
                "nir_adapters."
            )
            or name.startswith(
                "quality_gates."
            )
            or name.startswith(
                "decode_head.classifier."
            )
        )

        no_decay = (
            parameter.ndim == 1
            or name.endswith(
                ".bias"
            )
        )

        if is_new and no_decay:
            new_no_decay.append(
                parameter
            )
        elif is_new:
            new_decay.append(
                parameter
            )
        elif no_decay:
            base_no_decay.append(
                parameter
            )
        else:
            base_decay.append(
                parameter
            )

    groups = [
        {
            "params": base_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "pretrained_decay",
        },
        {
            "params": base_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "group_name": "pretrained_no_decay",
        },
        {
            "params": new_decay,
            "lr": new_lr,
            "weight_decay": weight_decay,
            "group_name": "new_decay",
        },
        {
            "params": new_no_decay,
            "lr": new_lr,
            "weight_decay": 0.0,
            "group_name": "new_no_decay",
        },
    ]

    groups = [
        group
        for group
        in groups
        if group[
            "params"
        ]
    ]

    optimizer = AdamW(
        groups
    )

    summary = {
        group[
            "group_name"
        ]: {
            "parameters": int(
                sum(
                    p.numel()
                    for p
                    in group[
                        "params"
                    ]
                )
            ),
            "lr": float(
                group[
                    "lr"
                ]
            ),
            "weight_decay": float(
                group[
                    "weight_decay"
                ]
            ),
        }
        for group
        in groups
    }

    return (
        optimizer,
        summary,
    )


def protocol_dict(
    *,
    x,
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
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "backbone": "SegFormer-B2",
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "dual_encoder": True,
        "fusion": (
            "RGB-anchored degradation-aware residual NIR fusion"
        ),
        "fusion_rule": (
            "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
        ),
        "quality_gate": True,
        "gate_type": (
            GATE_TYPE
        ),
        "gate_target_clean": (
            INITIAL_NIR_GATE
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
            "gate_bce_weight": (
                x.gate_weight
            ),
        },
        "corruption_training": {
            "enabled": True,
            "clean_warmup_epochs": (
                x.clean_warmup_epochs
            ),
            "ramp_epochs": (
                x.corruption_ramp_epochs
            ),
            "max_probability": (
                x.max_corruption_prob
            ),
            "families": [
                "gaussian_noise",
                "gaussian_blur",
                "rgb_underexposure",
            ],
            "severity_sampling": (
                "continuous Uniform(0.10,1.00)"
            ),
            "noise_sigma_255_range": [
                5.0,
                50.0,
            ],
            "blur_sigma_range": [
                0.5,
                4.0,
            ],
            "underexposure_alpha_range": [
                0.40,
                0.94,
            ],
            "nir": (
                "clean / unchanged"
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
        "gradient_checkpointing_effective": (
            any(
                bool(value)
                for value
                in checkpointing_status.values()
            )
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
            "This is no longer a clean-training-only robustness experiment; "
            "it is a degradation-aware robust training method."
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
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step": int(global_step),
        "protocol": dict(protocol),
        "model_meta": dict(model_meta),
        "rng_state": capture_rng_state(),
    }


def load_resume(
    *,
    path,
    model,
    optimizer,
    scheduler,
    scaler,
):
    c = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if c.get(
        "model_id"
    ) != MODEL_ID:
        raise RuntimeError(
            f"Wrong resume model id: {c.get('model_id')}"
        )

    model.load_state_dict(
        c[
            "model"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        c[
            "optimizer"
        ]
    )

    scheduler.load_state_dict(
        c[
            "scheduler"
        ]
    )

    scaler.load_state_dict(
        c[
            "scaler"
        ]
    )

    restore_rng_state(
        c.get(
            "rng_state"
        )
    )

    return (
        int(
            c[
                "epoch"
            ]
        )
        + 1,
        int(
            c.get(
                "global_step",
                c.get(
                    "step",
                    0,
                ),
            )
        ),
    )


def main():
    x = parse_args()

    seed_everything(
        x.seed
    )

    out = resolve(
        x.output_dir
    )

    ckpt_dir = (
        out
        / "checkpoints"
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = resolve(
        x.data_cache_dir
    )

    print("=" * 104)
    print(
        MODEL_NAME
    )
    print("=" * 104)

    dataset = CachedPotsdamTrainDataset(
        PROJECT_ROOT,
        epoch=0,
        cache_root=(
            cache
        ),
    )

    dataset.prepare_cache(
        rebuild=(
            x.rebuild_data_cache
        )
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

    device = get_device(
        x.device
    )

    print(
        "[1] building SegFormer-B2 DARF"
    )

    model, model_meta = (
        build_model_d_darf_b2(
            PROJECT_ROOT
        )
    )

    checkpointing_status = {
        "rgb_encoder": False,
        "nir_encoder": False,
    }

    if x.gradient_checkpointing:
        checkpointing_status = (
            model
            .enable_gradient_checkpointing()
        )

        if not any(
            checkpointing_status.values()
        ):
            print(
                "[gradient-checkpointing] requested but unsupported by the "
                "installed Transformers SegFormer implementation; continuing "
                "safely with checkpointing disabled."
            )

    model.to(
        device
    )

    print(
        f"  parameters       : "
        f"{model_meta['parameters']['total']:,}"
    )
    print(
        f"  hidden sizes     : "
        f"{model_meta['hidden_sizes']}"
    )
    print(
        f"  grad checkpoint  : "
        f"{checkpointing_status}"
    )

    (
        optimizer,
        optimizer_summary,
    ) = build_optimizer(
        model,
        base_lr=(
            x.base_lr
        ),
        new_lr=(
            x.new_lr
        ),
        weight_decay=(
            x.weight_decay
        ),
    )

    scheduler = build_poly_scheduler(
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

    protocol = protocol_dict(
        x=x,
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
    )

    write_json(
        out
        / "protocol.json",
        protocol,
    )

    start_epoch = 0
    global_step = 0

    rp = resume_path(
        x.resume,
        ckpt_dir,
    )

    if rp is not None:
        (
            start_epoch,
            global_step,
        ) = load_resume(
            path=(
                rp
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
            f"[resume] start_epoch={start_epoch}, "
            f"global_step={global_step}"
        )

    if start_epoch >= x.epochs:
        print(
            "[done] checkpoint already reached requested epochs."
        )
        return

    log_path = (
        out
        / "train_log.jsonl"
    )

    training_start = (
        time.time()
    )

    first_batch_checked = False

    print(
        "[2] start training"
    )

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

        p_corrupt = corruption_probability(
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

        total_valid = 0
        ce_numerator = 0.0
        lovasz_sum = 0.0
        gate_sum = 0.0
        total_loss_sum = 0.0
        mini_batches = 0
        updates = 0
        amp_skips = 0
        last_grad_norm = float(
            "nan"
        )

        gate_strength_sum = torch.zeros(
            NUM_SCALES,
            dtype=torch.float64,
        )

        gate_strength_sq = torch.zeros(
            NUM_SCALES,
            dtype=torch.float64,
        )

        gate_samples = 0

        family_counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "rgb_underexposure": 0,
        }

        severity_sum = 0.0
        degraded_samples = 0

        accumulation_target = (
            x.grad_accum_steps
        )

        for bi, batch in enumerate(
            loader
        ):
            if bi % x.grad_accum_steps == 0:
                remaining = len(
                    loader
                ) - bi

                accumulation_target = min(
                    x.grad_accum_steps,
                    remaining,
                )

            rgb = batch[
                "rgb"
            ].to(
                device,
                non_blocking=(
                    x.pin_memory
                ),
            )

            nir = batch[
                "nir"
            ].to(
                device,
                non_blocking=(
                    x.pin_memory
                ),
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=(
                    x.pin_memory
                ),
            ).long()

            (
                rgb_aug,
                gate_target,
                counts,
                severities,
            ) = degrade_rgb_batch(
                rgb,
                probability=(
                    p_corrupt
                ),
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
                device_type=(
                    device.type
                ),
                dtype=torch.float16,
                enabled=(
                    amp
                ),
            ):
                details = model(
                    rgb_aug,
                    nir,
                    return_details=True,
                )

                logits = details[
                    "logits"
                ]

                raw_logits = details[
                    "raw_logits"
                ]

                gate_strength = details[
                    "nir_gate_strength"
                ]

                gate_logits = details[
                    "nir_gate_logits"
                ]

                ce = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=(
                        IGNORE_INDEX
                    ),
                )

                lovasz = lovasz_softmax(
                    raw_logits,
                    labels,
                )

                target_matrix = (
                    gate_target
                    .view(
                        -1,
                        1,
                    )
                    .expand(
                        -1,
                        NUM_SCALES,
                    )
                )

                # AMP-safe and numerically stable gate supervision.
                #
                # Do NOT apply BCE to sigmoid probabilities under autocast.
                # The model already returns the pre-sigmoid gate logits, so
                # supervise those directly with BCEWithLogits.
                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_logits.float(),
                    target_matrix.float(),
                )

                total_loss = (
                    ce
                    + x.lovasz_weight
                    * lovasz
                    + x.gate_weight
                    * gate_loss
                )

                loss_for_backward = (
                    total_loss
                    / accumulation_target
                )

            if not torch.isfinite(
                total_loss
            ).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch+1}, batch={bi}."
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

            gate_sum += float(
                gate_loss
                .detach()
                .item()
            )

            total_loss_sum += float(
                total_loss
                .detach()
                .item()
            )

            gw = (
                gate_strength
                .detach()
                .float()
                .cpu()
                .to(
                    torch.float64
                )
            )

            gate_strength_sum += (
                gw.sum(
                    dim=0
                )
            )

            gate_strength_sq += (
                (
                    gw
                    * gw
                )
                .sum(
                    dim=0
                )
            )

            gate_samples += int(
                gw.shape[
                    0
                ]
            )

            mini_batches += 1

            should_step = (
                (
                    (
                        bi
                        + 1
                    )
                    % x.grad_accum_steps
                    == 0
                )
                or (
                    bi
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

            if (
                bi
                % x.log_every
                == 0
                or bi
                + 1
                == len(
                    loader
                )
            ):
                gate_mean_batch = (
                    gate_strength
                    .detach()
                    .float()
                    .mean(
                        dim=0
                    )
                    .cpu()
                    .tolist()
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
                    for index, group
                    in enumerate(
                        optimizer
                        .param_groups
                    )
                }

                print(
                    f"epoch {epoch+1:03d}/{x.epochs:03d} | "
                    f"batch {bi+1:05d}/{len(loader):05d} | "
                    f"loss {float(total_loss.detach().item()):.5f} | "
                    f"CE {float(ce.detach().item()):.5f} | "
                    f"Lovasz {float(lovasz.detach().item()):.5f} | "
                    f"Gate {float(gate_loss.detach().item()):.5f} | "
                    f"p_corrupt {p_corrupt:.3f} | "
                    f"gNIR {[round(v,4) for v in gate_mean_batch]} | "
                    f"step {global_step}/{total_updates} | "
                    f"lr_base {lrs.get('pretrained_decay', x.base_lr):.2e} | "
                    f"lr_new {lrs.get('new_decay', x.new_lr):.2e}"
                )

            del (
                details,
                logits,
                raw_logits,
                gate_strength,
                gate_logits,
                total_loss,
                loss_for_backward,
                rgb,
                rgb_aug,
                nir,
                labels,
            )

        if total_valid <= 0 or mini_batches <= 0:
            raise RuntimeError(
                "Epoch contains no valid training data."
            )

        gate_mean = (
            gate_strength_sum
            / float(
                gate_samples
            )
        )

        gate_var = torch.clamp(
            (
                gate_strength_sq
                / float(
                    gate_samples
                )
                - gate_mean
                * gate_mean
            ),
            min=0.0,
        )

        gate_std = torch.sqrt(
            gate_var
        )

        epoch_record = {
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
            "train_gate_bce_mean": (
                gate_sum
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
            "gate_nir_strength_mean_by_scale": [
                float(
                    value
                )
                for value
                in gate_mean.tolist()
            ],
            "gate_nir_strength_std_by_scale": [
                float(
                    value
                )
                for value
                in gate_std.tolist()
            ],
            "last_grad_norm_before_clip": (
                last_grad_norm
            ),
            "epoch_seconds": (
                time.time()
                - epoch_start
            ),
            "lr_groups": {
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
                for index, group
                in enumerate(
                    optimizer
                    .param_groups
                )
            },
        }

        append_jsonl(
            log_path,
            epoch_record,
        )

        payload = checkpoint_payload(
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
            ckpt_dir
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
                ckpt_dir
                / f"epoch_{epoch+1:03d}.pt",
            )

        print(
            f"[epoch done] {epoch+1:03d}/{x.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"Gate={epoch_record['train_gate_bce_mean']:.6f} | "
            f"gNIR="
            f"{[round(v,4) for v in epoch_record['gate_nir_strength_mean_by_scale']]} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"updates={updates} | skips={amp_skips} | "
            f"global_step={global_step}"
        )

    final = checkpoint_payload(
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
        ckpt_dir
        / "final.pt",
    )

    print("=" * 104)
    print(
        f"[finished] {MODEL_NAME} | "
        f"epochs={x.epochs} | "
        f"global_step={global_step} | "
        f"time={(time.time()-training_start)/3600.0:.2f}h"
    )
    print(
        f"[finished] final checkpoint : "
        f"{ckpt_dir/'final.pt'}"
    )
    print(
        f"[finished] training log     : "
        f"{log_path}"
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
