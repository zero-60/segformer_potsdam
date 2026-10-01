#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train M2: SegFormer-B2 RGB+NIR Fixed Residual Fusion + Robust-4.

Scientific role
---------------
M0: B2-RGB-Clean
M1: B2-RGB-Robust4
M2: B2-RGBNIR-Fixed-Robust4   <-- this script
M3: B2-DARF-Robust4            <-- next

M2 is intentionally designed as the direct architectural bridge between M1
and M3:

    F_i = F_RGB_i + g_fixed * Adapter_i(F_NIR_i)

with:
    g_fixed = 0.5

Why 0.5?
--------
It is the pre-specified midpoint of the DARF gate range [0,1].  It is NOT tuned
on validation results.  The model has the same dual B2 encoders, NIR residual
adapters, decoder, Robust-4 augmentation, optimizer, LR schedule and semantic
loss that M3 will use, but has NO quality gate and NO gate supervision.

Therefore:
    M1 -> M2 isolates the contribution of adding NIR residual information.
    M2 -> M3 isolates the contribution of adaptive degradation-aware gating.

Initialization
--------------
- RGB encoder: pretrained SegFormer-B2 encoder.
- NIR encoder: exact copy of pretrained RGB encoder.
- NIR first patch projection: 3->1 using mean of pretrained RGB kernels.
- NIR residual adapters: same structure as DARF, zero-output initialized.
- Therefore M2 starts EXACTLY as the pretrained RGB path:
      Adapter_i(...) == 0  =>  F_fused_i == F_RGB_i
  despite g_fixed=0.5.

Robust-4 training
-----------------
This script imports degrade_rgb_batch_robust4() from
tools/train_model_b2_rgb_robust4.py.  Thus M1 and M2 use the EXACT SAME
training corruption implementation:

    Gaussian noise
    Gaussian blur
    RGB underexposure
    Atmospheric fog

Only RGB is degraded.  NIR and labels stay unchanged.

Default optimization
--------------------
- 120 epochs
- batch size 1
- gradient accumulation 16
- effective batch 16
- pretrained/base LR 3e-5
- new adapter/classifier LR 3e-4
- weight decay 0.01
- warmup 500 successful optimizer updates
- CE + 0.5 Lovasz-Softmax
- AMP
- grad clip 1.0
- clean warm-up 10 epochs
- corruption ramp 20 epochs
- max corruption probability 0.50

Run
---
    python tools/train_model_b2_rgbnir_fixed_robust4.py

Resume
------
    python tools/train_model_b2_rgbnir_fixed_robust4.py --resume auto

Output
------
outputs/training/b2_rgbnir_fixed_robust4/
├── protocol.json
├── progress.json
├── train_log.jsonl
└── checkpoints/
    ├── latest.pt
    └── final.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import SegformerConfig, SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model_b2_rgb as train_base

from data_pipeline.potsdam_dataloader import (
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from models.segformer_b2_darf import (
    DEFAULT_CHECKPOINT,
    EXPECTED_HIDDEN_SIZES,
    NIRResidualAdapter,
    _adapt_nir_first_layer,
    require,
)
from models.segformer_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    parameter_counts,
    validate_pretrained_loading_info,
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
from train_model_b2_rgb_robust4 import (
    ROBUST4_FAMILIES,
    degrade_rgb_batch_robust4,
)
from train_model_d_darf_b2 import (
    corruption_probability,
    lovasz_softmax,
)

# =============================================================================
# Frozen M2 protocol
# =============================================================================

MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "B2 RGBNIR Fixed Residual Robust-4 Protocol v1"

MODEL_ID = "B2_RGBNIR_FIXED_ROBUST4"
MODEL_NAME = "SegFormer-B2 RGB+NIR Fixed Residual Fusion Robust-4"

NUM_SCALES = 4
FIXED_NIR_STRENGTH = 0.50

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_rgbnir_fixed_robust4"
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


# =============================================================================
# Generic helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
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


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


# =============================================================================
# M2 model
# =============================================================================

class SegFormerB2RGBNIRFixed(nn.Module):
    """
    Dual B2 encoder with fixed-strength RGB-anchored NIR residual fusion.

        fused_i = rgb_i + fixed_strength * Adapter_i(nir_i)

    There is deliberately NO quality gate.
    """

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        nir_encoder: nn.Module,
        decode_head: nn.Module,
        hidden_sizes: Sequence[int],
        fixed_strength: float = FIXED_NIR_STRENGTH,
    ):
        super().__init__()

        hidden_sizes = tuple(
            int(x)
            for x in hidden_sizes
        )

        require(
            hidden_sizes
            == EXPECTED_HIDDEN_SIZES,
            f"Unexpected B2 hidden sizes: {hidden_sizes}",
        )

        fixed_strength = float(
            fixed_strength
        )

        require(
            0.0
            <= fixed_strength
            <= 1.0,
            "fixed_strength must be in [0,1].",
        )

        self.rgb_encoder = rgb_encoder
        self.nir_encoder = nir_encoder
        self.decode_head = decode_head
        self.hidden_sizes = hidden_sizes
        self.fixed_strength = (
            fixed_strength
        )

        self.nir_adapters = nn.ModuleList(
            [
                NIRResidualAdapter(
                    channels
                )
                for channels in hidden_sizes
            ]
        )

    def enable_gradient_checkpointing(
        self,
    ) -> Dict[str, bool]:
        status: Dict[
            str,
            bool,
        ] = {}

        for name, encoder in (
            (
                "rgb_encoder",
                self.rgb_encoder,
            ),
            (
                "nir_encoder",
                self.nir_encoder,
            ),
        ):
            supported = bool(
                getattr(
                    encoder,
                    "supports_gradient_checkpointing",
                    False,
                )
            )

            if not supported:
                status[
                    name
                ] = False
                continue

            method = getattr(
                encoder,
                "gradient_checkpointing_enable",
                None,
            )

            if method is None:
                status[
                    name
                ] = False
                continue

            try:
                method()
            except (
                ValueError,
                NotImplementedError,
            ):
                status[
                    name
                ] = False
            else:
                status[
                    name
                ] = True

        return status

    def _encode(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ):
        rgb_out = (
            self.rgb_encoder(
                pixel_values=rgb,
                output_hidden_states=True,
                return_dict=True,
            )
        )

        nir_out = (
            self.nir_encoder(
                pixel_values=nir,
                output_hidden_states=True,
                return_dict=True,
            )
        )

        rgb_features = tuple(
            rgb_out.hidden_states
        )

        nir_features = tuple(
            nir_out.hidden_states
        )

        require(
            len(
                rgb_features
            )
            == NUM_SCALES,
            f"RGB encoder returned {len(rgb_features)} scales.",
        )

        require(
            len(
                nir_features
            )
            == NUM_SCALES,
            f"NIR encoder returned {len(nir_features)} scales.",
        )

        for index, (
            rgb_feature,
            nir_feature,
            channels,
        ) in enumerate(
            zip(
                rgb_features,
                nir_features,
                self.hidden_sizes,
            ),
            start=1,
        ):
            require(
                rgb_feature.shape
                == nir_feature.shape,
                f"Scale {index}: RGB/NIR shapes differ.",
            )

            require(
                rgb_feature.shape[
                    1
                ]
                == channels,
                f"Scale {index}: expected C={channels}, "
                f"got {rgb_feature.shape[1]}.",
            )

        return (
            rgb_features,
            nir_features,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        require(
            rgb.ndim == 4
            and rgb.shape[
                1
            ]
            == 3,
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}",
        )

        require(
            nir.ndim == 4
            and nir.shape[
                1
            ]
            == 1,
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}",
        )

        require(
            rgb.shape[
                0
            ]
            == nir.shape[
                0
            ]
            and rgb.shape[
                -2:
            ]
            == nir.shape[
                -2:
            ],
            "RGB/NIR inputs are not aligned.",
        )

        (
            rgb_features,
            nir_features,
        ) = self._encode(
            rgb,
            nir,
        )

        fused = []
        residual_abs_mean = []

        for (
            rgb_feature,
            nir_feature,
            adapter,
        ) in zip(
            rgb_features,
            nir_features,
            self.nir_adapters,
        ):
            residual = adapter(
                nir_feature
            )

            fused_feature = (
                rgb_feature
                + self.fixed_strength
                * residual
            )

            require(
                torch.isfinite(
                    fused_feature
                )
                .all()
                .item(),
                "Fused feature contains NaN/Inf.",
            )

            fused.append(
                fused_feature
            )

            with torch.no_grad():
                residual_abs_mean.append(
                    residual
                    .detach()
                    .float()
                    .abs()
                    .mean()
                )

        raw_logits = (
            self.decode_head(
                tuple(
                    fused
                )
            )
        )

        require(
            raw_logits.ndim
            == 4
            and raw_logits.shape[
                0
            ]
            == rgb.shape[
                0
            ]
            and raw_logits.shape[
                1
            ]
            == NUM_CLASSES,
            f"Bad raw logits shape: {tuple(raw_logits.shape)}",
        )

        logits = F.interpolate(
            raw_logits,
            size=rgb.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )

        require(
            torch.isfinite(
                logits
            )
            .all()
            .item(),
            "Logits contain NaN/Inf.",
        )

        if not return_details:
            return logits

        return {
            "logits": (
                logits
            ),
            "raw_logits": (
                raw_logits
            ),
            "fixed_nir_strength": (
                self.fixed_strength
            ),
            "residual_abs_mean": (
                torch.stack(
                    residual_abs_mean
                )
            ),
        }


def _audit_zero_residual(
    model: SegFormerB2RGBNIRFixed,
) -> Dict[str, Any]:
    rows = []

    for scale, adapter in enumerate(
        model.nir_adapters,
        start=1,
    ):
        w_nonzero = int(
            torch.count_nonzero(
                adapter.expand.weight
            )
            .item()
        )

        b_nonzero = int(
            torch.count_nonzero(
                adapter.expand.bias
            )
            .item()
        )

        require(
            w_nonzero == 0
            and b_nonzero == 0,
            f"Scale {scale}: residual adapter final projection is not zero-init.",
        )

        rows.append(
            {
                "scale": (
                    scale
                ),
                "channels": (
                    adapter.channels
                ),
                "bottleneck": (
                    adapter.hidden
                ),
                "zero_output_initialized": (
                    True
                ),
            }
        )

    return {
        "all_zero_output_initialized": (
            True
        ),
        "stages": (
            rows
        ),
    }


def build_model_b2_rgbnir_fixed(
    project_root: Path | str = PROJECT_ROOT,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    fixed_strength: float = FIXED_NIR_STRENGTH,
) -> Tuple[
    SegFormerB2RGBNIRFixed,
    Dict[str, Any],
]:
    _ = Path(
        project_root
    ).resolve()

    require(
        checkpoint
        == DEFAULT_CHECKPOINT,
        "M2 protocol requires the same pretrained B2 checkpoint as M1/M3.",
    )

    id2label = {
        index: name
        for index, name in enumerate(
            CLASS_NAMES
        )
    }

    label2id = {
        name: index
        for index, name in id2label.items()
    }

    config = (
        SegformerConfig
        .from_pretrained(
            checkpoint
        )
    )

    require(
        int(
            config.num_labels
        )
        == 150,
        "Expected ADE20K 150-label checkpoint.",
    )

    require(
        int(
            getattr(
                config,
                "num_channels",
                3,
            )
        )
        == 3,
        "Pretrained B2 checkpoint is not RGB.",
    )

    require(
        tuple(
            int(x)
            for x in config.hidden_sizes
        )
        == EXPECTED_HIDDEN_SIZES,
        f"Unexpected B2 hidden sizes: {config.hidden_sizes}",
    )

    config.num_labels = (
        NUM_CLASSES
    )

    config.id2label = (
        id2label
    )

    config.label2id = (
        label2id
    )

    config.semantic_loss_ignore_index = (
        IGNORE_INDEX
    )

    base_model, loading_info = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
    )

    loading_summary = (
        validate_pretrained_loading_info(
            loading_info
        )
    )

    rgb_encoder = (
        base_model.segformer
    )

    nir_encoder = (
        copy.deepcopy(
            rgb_encoder
        )
    )

    nir_init = (
        _adapt_nir_first_layer(
            nir_encoder
        )
    )

    decode_head = (
        base_model.decode_head
    )

    model = (
        SegFormerB2RGBNIRFixed(
            rgb_encoder=(
                rgb_encoder
            ),
            nir_encoder=(
                nir_encoder
            ),
            decode_head=(
                decode_head
            ),
            hidden_sizes=(
                config.hidden_sizes
            ),
            fixed_strength=(
                fixed_strength
            ),
        )
    )

    residual_audit = (
        _audit_zero_residual(
            model
        )
    )

    meta = {
        "variant": "M2",
        "model_id": (
            MODEL_ID
        ),
        "name": (
            MODEL_NAME
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "checkpoint": (
            checkpoint
        ),
        "resolved_revision": (
            getattr(
                base_model.config,
                "_commit_hash",
                None,
            )
        ),
        "backbone": (
            "SegFormer-B2"
        ),
        "num_labels": (
            NUM_CLASSES
        ),
        "class_names": list(
            CLASS_NAMES
        ),
        "hidden_sizes": list(
            EXPECTED_HIDDEN_SIZES
        ),
        "decoder_hidden_size": int(
            config.decoder_hidden_size
        ),
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "dual_encoder": (
            True
        ),
        "fusion": (
            "RGB-anchored fixed-strength NIR residual fusion"
        ),
        "fusion_rule": (
            "F_i = F_RGB_i + 0.5 * Adapter_i(F_NIR_i)"
        ),
        "fixed_nir_strength": (
            float(
                fixed_strength
            )
        ),
        "quality_gate": (
            False
        ),
        "gate_supervision": (
            False
        ),
        "nir_encoder_initialization": (
            nir_init
        ),
        "residual_adapter_audit": (
            residual_audit
        ),
        "loading_info": (
            loading_summary
        ),
        "parameters": (
            parameter_counts(
                model
            )
        ),
        "semantic_loss_ignore_index": (
            IGNORE_INDEX
        ),
    }

    return (
        model,
        meta,
    )


# =============================================================================
# Optimizer
# =============================================================================

def build_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    new_lr: float,
    weight_decay: float,
):
    pretrained_decay = []
    pretrained_no_decay = []
    new_decay = []
    new_no_decay = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        is_new = (
            name.startswith(
                "nir_adapters."
            )
            or name.startswith(
                "decode_head.classifier."
            )
        )

        no_decay = (
            parameter.ndim
            == 1
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
            pretrained_no_decay.append(
                parameter
            )
        else:
            pretrained_decay.append(
                parameter
            )

    groups = [
        {
            "params": (
                pretrained_decay
            ),
            "lr": (
                base_lr
            ),
            "weight_decay": (
                weight_decay
            ),
            "group_name": (
                "pretrained_decay"
            ),
        },
        {
            "params": (
                pretrained_no_decay
            ),
            "lr": (
                base_lr
            ),
            "weight_decay": (
                0.0
            ),
            "group_name": (
                "pretrained_no_decay"
            ),
        },
        {
            "params": (
                new_decay
            ),
            "lr": (
                new_lr
            ),
            "weight_decay": (
                weight_decay
            ),
            "group_name": (
                "new_decay"
            ),
        },
        {
            "params": (
                new_no_decay
            ),
            "lr": (
                new_lr
            ),
            "weight_decay": (
                0.0
            ),
            "group_name": (
                "new_no_decay"
            ),
        },
    ]

    groups = [
        group
        for group in groups
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
                    for p in group[
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
        for group in groups
    }

    return (
        optimizer,
        summary,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train M2 / B2 RGB+NIR fixed residual fusion with Robust-4."
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
        "--fixed-nir-strength",
        type=float,
        default=FIXED_NIR_STRENGTH,
        help=(
            "Pre-specified fixed residual coefficient. "
            "Keep 0.5 for the formal M2 experiment."
        ),
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

    if not (
        0.0
        <= args.fixed_nir_strength
        <= 1.0
    ):
        parser.error(
            "--fixed-nir-strength must be in [0,1]"
        )

    if args.clean_warmup_epochs < 0:
        parser.error(
            "--clean-warmup-epochs must be >= 0"
        )

    if args.corruption_ramp_epochs < 0:
        parser.error(
            "--corruption-ramp-epochs must be >= 0"
        )

    if not (
        0.0
        <= args.max_corruption_prob
        <= 1.0
    ):
        parser.error(
            "--max-corruption-prob must be in [0,1]"
        )

    if args.progress_every <= 0:
        parser.error(
            "--progress-every must be > 0"
        )

    if args.eta_warmup_batches <= 0:
        parser.error(
            "--eta-warmup-batches must be > 0"
        )

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
            DEFAULT_CHECKPOINT
        ),
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "nir_used": (
            True
        ),
        "dual_encoder": (
            True
        ),
        "fusion": (
            "RGB-anchored fixed-strength NIR residual fusion"
        ),
        "fusion_rule": (
            "F_i = F_RGB_i + 0.5 * Adapter_i(F_NIR_i)"
        ),
        "fixed_nir_strength": (
            float(
                args.fixed_nir_strength
            )
        ),
        "quality_gate": (
            False
        ),
        "gate_supervision": (
            False
        ),
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
        "new_lr": (
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
            "ce_weight": (
                1.0
            ),
            "lovasz_weight": (
                args.lovasz_weight
            ),
            "gate_loss": (
                False
            ),
        },
        "corruption_training": {
            "enabled": (
                True
            ),
            "version": (
                "Robust-4"
            ),
            "fog_in_training": (
                True
            ),
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
            "rgb": (
                "degraded according to Robust-4"
            ),
            "nir": (
                "clean / unchanged"
            ),
            "labels": (
                "unchanged"
            ),
            "implementation_source": (
                "degrade_rgb_batch_robust4 from "
                "tools/train_model_b2_rgb_robust4.py"
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
            "M2 differs from M1 by adding the NIR encoder/residual pathway. "
            "M2 differs from future M3 only by replacing adaptive g_i with "
            "the pre-specified constant g=0.5 and removing gate supervision."
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
        "format_version": (
            2
        ),
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

    # Formal M2 uses exactly 0.5. Allowing CLI override is useful for debugging,
    # but force the user to explicitly acknowledge non-formal runs in metadata.
    if abs(
        float(
            args.fixed_nir_strength
        )
        - FIXED_NIR_STRENGTH
    ) > 1e-12:
        print(
            "[WARNING] --fixed-nir-strength differs from formal M2 value 0.5. "
            "Do not mix this checkpoint into the formal ablation table.",
            flush=True,
        )

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

    print("=" * 122)
    print(
        MODEL_NAME
    )
    print("=" * 122)
    print(
        f"model id       : {MODEL_ID}"
    )
    print(
        "regime         : robust4"
    )
    print(
        "modalities     : RGB + NIR"
    )
    print(
        f"fixed g_NIR    : {args.fixed_nir_strength:.3f}"
    )
    print(
        "quality gate   : False"
    )
    print(
        "families       : Noise / Blur / Underexposure / Fog"
    )
    print(
        f"output         : {output_dir}"
    )
    print(
        f"start local    : "
        f"{train_base.format_local_datetime(train_base.local_now())}"
    )
    print("=" * 122)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    dataset = (
        CachedPotsdamTrainDataset(
            PROJECT_ROOT,
            epoch=0,
            cache_root=cache,
        )
    )

    dataset.prepare_cache(
        rebuild=(
            args.rebuild_data_cache
        )
    )

    sampler = (
        DeterministicEpochSampler(
            dataset,
            seed=DATA_SEED,
            epoch=0,
        )
    )

    loader = (
        build_train_dataloader(
            dataset,
            sampler,
            batch_size=(
                args.batch_size
            ),
            num_workers=(
                args.num_workers
            ),
            pin_memory=(
                args.pin_memory
            ),
        )
    )

    updates_per_epoch = math.ceil(
        len(
            loader
        )
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

    device = (
        train_base.get_device(
            args.device
        )
    )

    model, model_meta = (
        build_model_b2_rgbnir_fixed(
            PROJECT_ROOT,
            fixed_strength=(
                args.fixed_nir_strength
            ),
        )
    )

    checkpointing_status = {
        "rgb_encoder": (
            False
        ),
        "nir_encoder": (
            False
        ),
    }

    if args.gradient_checkpointing:
        checkpointing_status = (
            model
            .enable_gradient_checkpointing()
        )

        if not any(
            checkpointing_status.values()
        ):
            print(
                "[gradient-checkpointing] requested but unsupported by the "
                "installed SegFormer implementation; continuing safely.",
                flush=True,
            )

    model.to(
        device
    )

    optimizer, optimizer_summary = (
        build_optimizer(
            model,
            base_lr=(
                args.base_lr
            ),
            new_lr=(
                args.new_lr
            ),
            weight_decay=(
                args.weight_decay
            ),
        )
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
        and not args.no_amp
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=(
                amp
            ),
        )
    )

    protocol = build_protocol(
        args=args,
        model_meta=model_meta,
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
        f"hidden sizes    : "
        f"{model_meta['hidden_sizes']}"
    )
    print(
        f"grad checkpoint : "
        f"{checkpointing_status}"
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

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0
    global_step = 0

    resume = (
        resolve_resume(
            args.resume,
            checkpoint_dir,
        )
    )

    if resume is not None:
        (
            start_epoch,
            global_step,
        ) = load_resume(
            path=(
                resume
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
            f"[resume] {resume} | "
            f"start_epoch={start_epoch} | "
            f"global_step={global_step}",
            flush=True,
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
        train_base.LiveTrainingProgress(
            total_epochs=(
                args.epochs
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
                args.progress_every
            ),
            eta_warmup_batches=(
                args.eta_warmup_batches
            ),
            ema_alpha=(
                args.eta_ema_alpha
            ),
            output_path=(
                output_dir
                / "progress.json"
            ),
            device=(
                device
            ),
            enabled=(
                not args.no_progress
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

    first_batch_checked = (
        False
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs,
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

        residual_sum = (
            torch.zeros(
                NUM_SCALES,
                dtype=torch.float64,
            )
        )

        residual_samples = 0

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
                    len(
                        loader
                    )
                    - batch_index
                )

                accumulation_target = min(
                    args.grad_accum_steps,
                    remaining,
                )

            rgb = (
                batch[
                    "rgb"
                ]
                .to(
                    device,
                    non_blocking=(
                        args.pin_memory
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
                        args.pin_memory
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
                        args.pin_memory
                    ),
                )
                .long()
            )

            if not first_batch_checked:
                require(
                    rgb.ndim == 4
                    and rgb.shape[
                        1
                    ]
                    == 3,
                    f"Bad RGB batch: {tuple(rgb.shape)}",
                )

                require(
                    nir.ndim == 4
                    and nir.shape[
                        1
                    ]
                    == 1,
                    f"Bad NIR batch: {tuple(nir.shape)}",
                )

                require(
                    rgb.shape[
                        0
                    ]
                    == nir.shape[
                        0
                    ]
                    and rgb.shape[
                        -2:
                    ]
                    == nir.shape[
                        -2:
                    ],
                    "RGB/NIR batch alignment failed.",
                )

                require(
                    labels.ndim
                    == 3,
                    f"Bad labels batch: {tuple(labels.shape)}",
                )

                if not torch.isfinite(
                    rgb
                ).all().item():
                    raise FloatingPointError(
                        "RGB input contains NaN/Inf."
                    )

                if not torch.isfinite(
                    nir
                ).all().item():
                    raise FloatingPointError(
                        "NIR input contains NaN/Inf."
                    )

                first_batch_checked = (
                    True
                )

                print(
                    "[first batch] PASS | "
                    f"RGB={tuple(rgb.shape)} | "
                    f"NIR={tuple(nir.shape)} | "
                    f"labels={tuple(labels.shape)} | "
                    f"fixed_g={args.fixed_nir_strength:.3f}",
                    flush=True,
                )

            rgb_model, counts, severities = (
                degrade_rgb_batch_robust4(
                    rgb,
                    probability=(
                        p_corrupt
                    ),
                )
            )

            # NIR deliberately remains clean.
            nir_model = (
                nir
            )

            for key in (
                family_counts
            ):
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
                dtype=(
                    torch.float16
                ),
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
                    f"Non-finite loss at epoch={epoch + 1}, "
                    f"batch={batch_index + 1}"
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            with torch.no_grad():
                residual_sum += (
                    details[
                        "residual_abs_mean"
                    ]
                    .detach()
                    .cpu()
                    .double()
                )

                residual_samples += 1

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
                        args.base_lr,
                    )
                ),
                new_lr=(
                    lrs.get(
                        "new_decay",
                        args.new_lr,
                    )
                ),
                p_corrupt=(
                    p_corrupt
                ),
            )

            if (
                batch_index
                % args.log_every
                == 0
                or batch_index
                + 1
                == len(
                    loader
                )
            ):
                append_jsonl(
                    log_path,
                    {
                        "event": (
                            "batch_log"
                        ),
                        "time_local": (
                            train_base
                            .format_local_datetime(
                                train_base
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
                        "fixed_nir_strength": (
                            float(
                                args.fixed_nir_strength
                            )
                        ),
                        "residual_abs_mean": [
                            float(
                                x
                            )
                            for x in details[
                                "residual_abs_mean"
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
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
                "Epoch contains no valid training data."
            )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        residual_mean = (
            (
                residual_sum
                / residual_samples
            )
            .tolist()
            if residual_samples
            > 0
            else [
                0.0
                for _ in range(
                    NUM_SCALES
                )
            ]
        )

        epoch_record = {
            "event": (
                "epoch_done"
            ),
            "time_local": (
                train_base
                .format_local_datetime(
                    train_base
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
            "fixed_nir_strength": (
                float(
                    args.fixed_nir_strength
                )
            ),
            "residual_abs_mean": (
                [
                    float(
                        value
                    )
                    for value in residual_mean
                ]
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
                train_base
                .format_duration(
                    epoch_seconds
                )
            ),
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
                / (
                    f"epoch_"
                    f"{epoch + 1:03d}.pt"
                ),
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/{args.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"Total={epoch_record['train_total_loss_mean']:.6f} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"families={family_counts} | "
            f"residual="
            f"{[round(float(v), 5) for v in residual_mean]} | "
            f"epoch={train_base.format_duration(epoch_seconds)} | "
            f"global_step={global_step}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

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
            args.epochs
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

    print("=" * 122)
    print(
        f"[finished] {MODEL_NAME}"
    )
    print(
        f"[finished] duration   : "
        f"{train_base.format_duration(total_seconds)}"
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
    print("=" * 122)


if __name__ == "__main__":
    main()
