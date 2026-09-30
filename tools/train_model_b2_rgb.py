#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the fair SegFormer-B2 RGB-only baselines for the Potsdam experiments.

This ONE script intentionally supports two regimes so the M0/M1 comparison
cannot drift because of copied training code:

    M0 / clean:
        python tools/train_model_b2_rgb.py --regime clean

    M1 / robust3:
        python tools/train_model_b2_rgb.py --regime robust3

The architecture, optimizer, loss, seed, epochs, batch/accumulation, scheduler
and pretrained checkpoint are identical.  The ONLY intended difference is:

    clean   : RGB is never synthetically degraded.
    robust3 : reuse the exact Model-D Robust-3 corruption functions:
              Gaussian noise + Gaussian blur + RGB underexposure.

Fog is intentionally NOT used in training.  Fog remains an unseen/OOD test.

Fair protocol relative to current Model D / DARF-B2
----------------------------------------------------
- Backbone: nvidia/segformer-b2-finetuned-ade-512-512
- RGB only, 3 channels
- 6 Potsdam classes
- 120 epochs
- batch size 1
- gradient accumulation 16
- nominal effective batch 16
- pretrained/base LR 3e-5
- new 6-class classifier LR 3e-4
- weight decay 0.01
- 500 successful-update warmup steps
- polynomial linear decay (power=1.0)
- CE + 0.5 * Lovasz-Softmax
- AMP on CUDA
- grad clip 1.0
- same deterministic Potsdam sampler/cache used by Model D

Robust-3 regime
---------------
The exact functions are imported from tools/train_model_d_darf_b2.py to avoid
silently creating a slightly different corruption distribution:
    - corruption_probability(...)
    - degrade_rgb_batch(...)
    - lovasz_softmax(...)

Therefore robust3 uses the same:
    clean warm-up = 10 epochs
    corruption ramp = 20 epochs
    max corruption probability = 0.50
    continuous corruption severity
    Gaussian noise / blur / RGB underexposure

Process visualization / ETA
---------------------------
The terminal progress line shows:
    - epoch / batch
    - epoch progress bar
    - total training percentage
    - CE / Lovasz / total loss
    - base/new LR
    - CUDA allocated/reserved memory
    - elapsed time
    - ETA
    - estimated LOCAL finish date/time

ETA is intentionally shown as "calibrating" during the first few batches.
After --eta-warmup-batches observations it is updated from an exponential
moving average of real batch wall-clock time.

A machine-readable snapshot is also continuously written to:
    <output_dir>/progress.json

Checkpointing
-------------
latest.pt is written every epoch and can be resumed with:
    --resume auto

Examples
--------
M0 Clean:
    python tools/train_model_b2_rgb.py --regime clean

M1 Robust-3:
    python tools/train_model_b2_rgb.py --regime robust3

Resume:
    python tools/train_model_b2_rgb.py --regime clean --resume auto

If CUDA memory is tight:
    python tools/train_model_b2_rgb.py --regime clean --batch-size 1

Expected output directories
---------------------------
outputs/training/b2_rgb_clean/
outputs/training/b2_rgb_robust3/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# =============================================================================
# Project imports
# =============================================================================

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
from train_model_d_darf_b2 import (
    corruption_probability,
    degrade_rgb_batch,
    lovasz_softmax,
)

# =============================================================================
# Frozen protocol constants
# =============================================================================

MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "B2 RGB Fair Baseline Protocol v1"

DEFAULT_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
EXPECTED_HIDDEN_SIZES = (64, 128, 320, 512)

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

ROBUST3_CLEAN_WARMUP_EPOCHS = 10
ROBUST3_RAMP_EPOCHS = 20
ROBUST3_MAX_CORRUPTION_PROB = 0.50

DEFAULT_PROGRESS_EVERY = 5
DEFAULT_ETA_WARMUP_BATCHES = 20
DEFAULT_ETA_EMA_ALPHA = 0.08


# =============================================================================
# Generic helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def default_output_dir(regime: str) -> Path:
    if regime == "clean":
        return PROJECT_ROOT / "outputs" / "training" / "b2_rgb_clean"
    if regime == "robust3":
        return PROJECT_ROOT / "outputs" / "training" / "b2_rgb_robust3"
    raise ValueError(regime)


def model_id_for_regime(regime: str) -> str:
    return "B2_RGB_CLEAN" if regime == "clean" else "B2_RGB_ROBUST3"


def model_name_for_regime(regime: str) -> str:
    return (
        "SegFormer-B2 RGB-only Clean Baseline"
        if regime == "clean"
        else "SegFormer-B2 RGB-only Robust-3 Baseline"
    )


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


def get_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


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


def local_now() -> datetime:
    return datetime.now().astimezone()


def format_local_datetime(dt: Optional[datetime]) -> str:
    if dt is None:
        return "calibrating"
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def cuda_memory_text(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"{allocated:.2f}/{reserved:.2f}G"


# =============================================================================
# Model
# =============================================================================

class SegFormerB2RGB(nn.Module):
    """
    RGB-only SegFormer-B2 wrapper exposing both raw 1/4-resolution logits and
    full 512x512 logits so CE/Lovasz match the Model-D training definition.
    """

    def __init__(self, hf_model: SegformerForSemanticSegmentation):
        super().__init__()
        self.hf_model = hf_model

    def enable_gradient_checkpointing(self) -> Dict[str, bool]:
        status = {"rgb_encoder": False}

        encoder = self.hf_model.segformer
        supported = bool(
            getattr(
                encoder,
                "supports_gradient_checkpointing",
                False,
            )
        )

        if not supported:
            return status

        method = getattr(
            encoder,
            "gradient_checkpointing_enable",
            None,
        )

        if method is None:
            return status

        try:
            method()
        except (ValueError, NotImplementedError):
            status["rgb_encoder"] = False
        else:
            status["rgb_encoder"] = True

        return status

    def forward(
        self,
        rgb: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise RuntimeError(
                f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}."
            )

        output = self.hf_model(
            pixel_values=rgb,
            return_dict=True,
        )

        raw_logits = output.logits

        if (
            raw_logits.ndim != 4
            or raw_logits.shape[0] != rgb.shape[0]
            or raw_logits.shape[1] != NUM_CLASSES
        ):
            raise RuntimeError(
                f"Unexpected raw logits shape: {tuple(raw_logits.shape)}."
            )

        logits = F.interpolate(
            raw_logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("B2-RGB logits contain NaN/Inf.")

        if not return_details:
            return logits

        return {
            "logits": logits,
            "raw_logits": raw_logits,
        }


def build_b2_rgb_model() -> tuple[SegFormerB2RGB, Dict[str, Any]]:
    id2label = {
        index: name
        for index, name in enumerate(CLASS_NAMES)
    }
    label2id = {
        name: index
        for index, name in id2label.items()
    }

    config = SegformerConfig.from_pretrained(
        DEFAULT_CHECKPOINT
    )

    original_num_labels = int(
        config.num_labels
    )

    if original_num_labels != 150:
        raise RuntimeError(
            f"Expected ADE20K 150 labels, got {original_num_labels}."
        )

    if int(getattr(config, "num_channels", 3)) != 3:
        raise RuntimeError(
            "B2 pretrained checkpoint is not 3-channel RGB."
        )

    hidden_sizes = tuple(
        int(value)
        for value in config.hidden_sizes
    )

    if hidden_sizes != EXPECTED_HIDDEN_SIZES:
        raise RuntimeError(
            f"Unexpected SegFormer-B2 hidden sizes: {hidden_sizes}."
        )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    base, loading_info = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            DEFAULT_CHECKPOINT,
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

    model = SegFormerB2RGB(
        base
    )

    meta = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "resolved_revision": getattr(
            base.config,
            "_commit_hash",
            None,
        ),
        "backbone": "SegFormer-B2",
        "input_modalities": ["RGB"],
        "input_channels": 3,
        "num_labels": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "hidden_sizes": list(
            EXPECTED_HIDDEN_SIZES
        ),
        "decoder_hidden_size": int(
            config.decoder_hidden_size
        ),
        "loading_info": loading_summary,
        "parameters": parameter_counts(
            model
        ),
    }

    return model, meta


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

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # New 6-class classifier is randomly initialized after 150 -> 6.
        is_new = (
            name.startswith(
                "hf_model.decode_head.classifier."
            )
        )

        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
        )

        if is_new and no_decay:
            new_no_decay.append(parameter)
        elif is_new:
            new_decay.append(parameter)
        elif no_decay:
            pretrained_no_decay.append(parameter)
        else:
            pretrained_decay.append(parameter)

    groups = [
        {
            "params": pretrained_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "pretrained_decay",
        },
        {
            "params": pretrained_no_decay,
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
        for group in groups
        if group["params"]
    ]

    optimizer = AdamW(
        groups
    )

    summary = {
        group["group_name"]: {
            "parameters": int(
                sum(
                    p.numel()
                    for p in group["params"]
                )
            ),
            "lr": float(
                group["lr"]
            ),
            "weight_decay": float(
                group["weight_decay"]
            ),
        }
        for group in groups
    }

    return optimizer, summary


# =============================================================================
# ETA / live terminal progress
# =============================================================================

class LiveTrainingProgress:
    """
    Lightweight progress visualization with no third-party dependency.

    ETA is based on an EMA of actual mini-batch wall time.  The first
    eta_warmup_batches are used for calibration before a finish time is shown.
    """

    def __init__(
        self,
        *,
        total_epochs: int,
        batches_per_epoch: int,
        start_epoch: int,
        progress_every: int,
        eta_warmup_batches: int,
        ema_alpha: float,
        output_path: Path,
        device: torch.device,
        enabled: bool,
    ):
        self.total_epochs = int(total_epochs)
        self.batches_per_epoch = int(batches_per_epoch)
        self.start_epoch = int(start_epoch)
        self.progress_every = int(progress_every)
        self.eta_warmup_batches = int(eta_warmup_batches)
        self.ema_alpha = float(ema_alpha)
        self.output_path = output_path
        self.device = device
        self.enabled = bool(enabled)

        self.total_batches_all = (
            self.total_epochs
            * self.batches_per_epoch
        )

        self.training_started_wall = (
            local_now()
        )
        self.training_started_mono = (
            time.monotonic()
        )

        self.observed_batches = 0
        self.ema_batch_seconds: Optional[
            float
        ] = None

        self.is_tty = (
            sys.stdout.isatty()
        )

    def observe_batch(
        self,
        seconds: float,
    ) -> None:
        seconds = float(seconds)

        if (
            not math.isfinite(seconds)
            or seconds <= 0
        ):
            return

        self.observed_batches += 1

        if self.ema_batch_seconds is None:
            self.ema_batch_seconds = seconds
        else:
            alpha = self.ema_alpha
            self.ema_batch_seconds = (
                alpha
                * seconds
                + (
                    1.0 - alpha
                )
                * self.ema_batch_seconds
            )

    def estimates(
        self,
        *,
        epoch_zero_based: int,
        batch_zero_based: int,
    ) -> tuple[
        Optional[float],
        Optional[datetime],
    ]:
        if (
            self.observed_batches
            < self.eta_warmup_batches
            or self.ema_batch_seconds
            is None
        ):
            return None, None

        completed_all = (
            epoch_zero_based
            * self.batches_per_epoch
            + batch_zero_based
            + 1
        )

        remaining = max(
            0,
            self.total_batches_all
            - completed_all,
        )

        eta_seconds = (
            self.ema_batch_seconds
            * remaining
        )

        finish = (
            local_now()
            + timedelta(
                seconds=eta_seconds
            )
        )

        return (
            eta_seconds,
            finish,
        )

    @staticmethod
    def _bar(
        fraction: float,
        *,
        width: int = 24,
    ) -> str:
        fraction = min(
            max(
                float(fraction),
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

    def update(
        self,
        *,
        epoch_zero_based: int,
        batch_zero_based: int,
        loss: float,
        ce: float,
        lovasz: float,
        base_lr: float,
        new_lr: float,
        p_corrupt: float,
        force: bool = False,
    ) -> None:
        batch_one_based = (
            batch_zero_based
            + 1
        )

        if (
            not force
            and batch_one_based
            % self.progress_every
            != 0
            and batch_one_based
            != self.batches_per_epoch
        ):
            return

        epoch_fraction = (
            batch_one_based
            / self.batches_per_epoch
        )

        completed_all = (
            epoch_zero_based
            * self.batches_per_epoch
            + batch_one_based
        )

        total_fraction = (
            completed_all
            / self.total_batches_all
        )

        eta_seconds, finish = (
            self.estimates(
                epoch_zero_based=(
                    epoch_zero_based
                ),
                batch_zero_based=(
                    batch_zero_based
                ),
            )
        )

        elapsed = (
            time.monotonic()
            - self.training_started_mono
        )

        snapshot = {
            "status": "running",
            "updated_at_local": (
                format_local_datetime(
                    local_now()
                )
            ),
            "training_started_at_local": (
                format_local_datetime(
                    self.training_started_wall
                )
            ),
            "epoch": (
                epoch_zero_based
                + 1
            ),
            "epochs": (
                self.total_epochs
            ),
            "batch": (
                batch_one_based
            ),
            "batches_per_epoch": (
                self.batches_per_epoch
            ),
            "epoch_progress_pct": (
                100.0
                * epoch_fraction
            ),
            "total_progress_pct": (
                100.0
                * total_fraction
            ),
            "loss": float(
                loss
            ),
            "ce": float(
                ce
            ),
            "lovasz": float(
                lovasz
            ),
            "base_lr": float(
                base_lr
            ),
            "new_lr": float(
                new_lr
            ),
            "corruption_probability": (
                float(
                    p_corrupt
                )
            ),
            "cuda_memory": (
                cuda_memory_text(
                    self.device
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
                if finish
                is not None
                else None
            ),
            "eta_calibration": {
                "observed_batches": (
                    self.observed_batches
                ),
                "required_batches": (
                    self.eta_warmup_batches
                ),
                "ema_batch_seconds": (
                    self.ema_batch_seconds
                ),
            },
        }

        write_json_atomic(
            self.output_path,
            snapshot,
        )

        if not self.enabled:
            return

        eta_text = (
            format_duration(
                eta_seconds
            )
            if eta_seconds
            is not None
            else (
                "calibrating "
                f"{self.observed_batches}/"
                f"{self.eta_warmup_batches}"
            )
        )

        finish_text = (
            format_local_datetime(
                finish
            )
            if finish
            is not None
            else "calibrating"
        )

        line = (
            f"E{epoch_zero_based + 1:03d}/{self.total_epochs:03d} "
            f"{self._bar(epoch_fraction)} "
            f"B{batch_one_based:04d}/{self.batches_per_epoch:04d} "
            f"ALL {100.0 * total_fraction:6.2f}% | "
            f"L {loss:.4f} "
            f"CE {ce:.4f} "
            f"Lov {lovasz:.4f} | "
            f"pCor {p_corrupt:.2f} | "
            f"LR {base_lr:.2e}/{new_lr:.2e} | "
            f"GPU {cuda_memory_text(self.device)} | "
            f"elapsed {format_duration(elapsed)} | "
            f"ETA {eta_text} | "
            f"finish {finish_text}"
        )

        if self.is_tty:
            sys.stdout.write(
                "\r"
                + line[:240]
                .ljust(240)
            )
            sys.stdout.flush()
        else:
            print(
                line,
                flush=True,
            )

    def epoch_boundary(self) -> None:
        if (
            self.enabled
            and self.is_tty
        ):
            sys.stdout.write(
                "\n"
            )
            sys.stdout.flush()

    def finish(
        self,
        *,
        output_dir: Path,
    ) -> None:
        finished = (
            local_now()
        )

        elapsed = (
            time.monotonic()
            - self.training_started_mono
        )

        payload = {
            "status": "finished",
            "finished_at_local": (
                format_local_datetime(
                    finished
                )
            ),
            "training_started_at_local": (
                format_local_datetime(
                    self.training_started_wall
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
            "output_dir": str(
                output_dir
            ),
        }

        write_json_atomic(
            self.output_path,
            payload,
        )


# =============================================================================
# Protocol / checkpoint
# =============================================================================

def protocol_dict(
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
    robust = (
        args.regime
        == "robust3"
    )

    return {
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "module_version": (
            MODULE_VERSION
        ),
        "model_id": (
            model_id_for_regime(
                args.regime
            )
        ),
        "model_name": (
            model_name_for_regime(
                args.regime
            )
        ),
        "backbone": "SegFormer-B2",
        "checkpoint": (
            DEFAULT_CHECKPOINT
        ),
        "input_modalities": [
            "RGB"
        ],
        "nir_used": False,
        "fusion": None,
        "regime": (
            args.regime
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
            "enabled": robust,
            "fog_in_training": False,
            "source": (
                "exact helper functions imported from "
                "tools/train_model_d_darf_b2.py"
                if robust
                else None
            ),
            "clean_warmup_epochs": (
                ROBUST3_CLEAN_WARMUP_EPOCHS
                if robust
                else None
            ),
            "ramp_epochs": (
                ROBUST3_RAMP_EPOCHS
                if robust
                else None
            ),
            "max_probability": (
                ROBUST3_MAX_CORRUPTION_PROB
                if robust
                else 0.0
            ),
            "families": (
                [
                    "gaussian_noise",
                    "gaussian_blur",
                    "rgb_underexposure",
                ]
                if robust
                else []
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
        "progress_visualization": {
            "terminal_enabled": (
                not args.no_progress
            ),
            "progress_every_batches": (
                args.progress_every
            ),
            "eta_warmup_batches": (
                args.eta_warmup_batches
            ),
            "eta_ema_alpha": (
                args.eta_ema_alpha
            ),
            "progress_json": (
                "progress.json"
            ),
        },
        "scientific_note": (
            "M0/M1 share one training implementation. The intended "
            "difference is only clean versus Robust-3 RGB augmentation."
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
            protocol[
                "model_id"
            ]
        ),
        "model_name": (
            protocol[
                "model_name"
            ]
        ),
        "regime": (
            protocol[
                "regime"
            ]
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


def resume_path(
    arg: str,
    ckpt_dir: Path,
) -> Optional[Path]:
    if not arg:
        return None

    if arg.lower() == "auto":
        path = (
            ckpt_dir
            / "latest.pt"
        )
    else:
        path = Path(
            arg
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
    expected_model_id,
    expected_regime,
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
            "Resume model_id mismatch: "
            f"{checkpoint.get('model_id')} "
            f"!= {expected_model_id}"
        )

    if checkpoint.get(
        "regime"
    ) != expected_regime:
        raise RuntimeError(
            "Resume regime mismatch: "
            f"{checkpoint.get('regime')} "
            f"!= {expected_regime}"
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
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train fair SegFormer-B2 RGB-only Clean/Robust3 baselines "
            "with live progress and wall-clock ETA."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--regime",
        choices=(
            "clean",
            "robust3",
        ),
        default="clean",
        help=(
            "clean=M0; robust3=M1. Fog is never used for training."
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
        default=None,
        help=(
            "Default is outputs/training/b2_rgb_<regime>."
        ),
    )
    parser.add_argument(
        "--resume",
        default="",
        help=(
            'Checkpoint path, or "auto" for latest.pt.'
        ),
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
        help=(
            "Write a detailed text-style training log message every N batches."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Refresh live terminal/progress.json every N mini-batches."
        ),
    )
    parser.add_argument(
        "--eta-warmup-batches",
        type=int,
        default=DEFAULT_ETA_WARMUP_BATCHES,
        help=(
            "Observed mini-batches before ETA/finish timestamp is shown."
        ),
    )
    parser.add_argument(
        "--eta-ema-alpha",
        type=float,
        default=DEFAULT_ETA_EMA_ALPHA,
        help=(
            "EMA smoothing factor for observed mini-batch wall time."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable terminal live progress; progress.json is still written."
        ),
    )

    args = parser.parse_args()

    positive = [
        (
            "epochs",
            args.epochs,
        ),
        (
            "batch-size",
            args.batch_size,
        ),
        (
            "grad-accum-steps",
            args.grad_accum_steps,
        ),
        (
            "base-lr",
            args.base_lr,
        ),
        (
            "new-lr",
            args.new_lr,
        ),
        (
            "grad-clip",
            args.grad_clip,
        ),
        (
            "log-every",
            args.log_every,
        ),
        (
            "progress-every",
            args.progress_every,
        ),
        (
            "eta-warmup-batches",
            args.eta_warmup_batches,
        ),
    ]

    for name, value in positive:
        if value <= 0:
            parser.error(
                f"--{name} must be > 0"
            )

    if (
        args.weight_decay < 0
        or args.warmup_steps < 0
        or args.lovasz_weight < 0
    ):
        parser.error(
            "weight-decay/warmup/lovasz must be non-negative"
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
# Main
# =============================================================================

def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    output_dir = (
        resolve(
            args.output_dir
        )
        if args.output_dir
        is not None
        else default_output_dir(
            args.regime
        ).resolve()
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

    model_id = model_id_for_regime(
        args.regime
    )

    model_name = (
        model_name_for_regime(
            args.regime
        )
    )

    print("=" * 118)
    print(model_name)
    print("=" * 118)
    print(f"model id       : {model_id}")
    print(f"regime         : {args.regime}")
    print(f"pretrained     : {DEFAULT_CHECKPOINT}")
    print(f"output         : {output_dir}")
    print(f"start local    : {format_local_datetime(local_now())}")
    print("fog in training: False (reserved for unseen/OOD validation)")
    print("=" * 118)

    # -------------------------------------------------------------------------
    # Dataset / loader
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
            total_updates - 1,
        ),
    )

    # -------------------------------------------------------------------------
    # Model / optimizer
    # -------------------------------------------------------------------------

    device = get_device(
        args.device
    )

    print("[1] building SegFormer-B2 RGB-only")

    model, model_meta = (
        build_b2_rgb_model()
    )

    checkpointing_status = {
        "rgb_encoder": False
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
                "[gradient-checkpointing] requested but unsupported by "
                "installed Transformers SegFormer; continuing safely."
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
        base_lr=args.base_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_poly_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup,
        total_steps=total_updates,
        power=1.0,
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

    protocol = protocol_dict(
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

    print("[2] frozen training protocol")
    print(f"  epochs           : {args.epochs}")
    print(f"  mini-batch       : {args.batch_size}")
    print(f"  grad accumulation: {args.grad_accum_steps}")
    print(
        f"  effective batch  : "
        f"{args.batch_size * args.grad_accum_steps}"
    )
    print(f"  batches/epoch    : {len(loader)}")
    print(f"  updates/epoch    : {updates_per_epoch}")
    print(f"  total updates    : {total_updates}")
    print(f"  CE + Lovasz      : 1.0 + {args.lovasz_weight}")
    print(
        f"  LR base/new      : "
        f"{args.base_lr:.2e} / {args.new_lr:.2e}"
    )
    print(f"  AMP              : {amp}")
    print(
        f"  progress file    : "
        f"{output_dir / 'progress.json'}"
    )

    if args.regime == "robust3":
        print(
            "  corruption       : Noise / Blur / Underexposure "
            "(exact Model-D Robust-3 helper functions)"
        )
    else:
        print(
            "  corruption       : disabled"
        )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0
    global_step = 0

    resume = resume_path(
        args.resume,
        checkpoint_dir,
    )

    if resume is not None:
        (
            start_epoch,
            global_step,
        ) = load_resume(
            path=resume,
            expected_model_id=model_id,
            expected_regime=args.regime,
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
            "[done] checkpoint already reached requested epochs."
        )
        return

    # -------------------------------------------------------------------------
    # Progress / logs
    # -------------------------------------------------------------------------

    progress = LiveTrainingProgress(
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

    print(
        "[ETA] exact wall-clock end time depends on your GPU/data throughput. "
        f"A calibrated local finish time will appear after "
        f"{args.eta_warmup_batches} observed mini-batches."
    )

    log_path = (
        output_dir
        / "train_log.jsonl"
    )

    training_start = time.time()
    first_batch_checked = False

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("[3] start training")

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

        if args.regime == "robust3":
            p_corrupt = (
                corruption_probability(
                    epoch,
                    clean_warmup_epochs=(
                        ROBUST3_CLEAN_WARMUP_EPOCHS
                    ),
                    ramp_epochs=(
                        ROBUST3_RAMP_EPOCHS
                    ),
                    maximum=(
                        ROBUST3_MAX_CORRUPTION_PROB
                    ),
                )
            )
        else:
            p_corrupt = 0.0

        total_valid = 0
        ce_numerator = 0.0
        lovasz_sum = 0.0
        total_loss_sum = 0.0
        mini_batches = 0
        updates = 0
        amp_skips = 0
        last_grad_norm = float("nan")

        family_counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "rgb_underexposure": 0,
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

            # Dataset may provide NIR, but this B2-RGB baseline MUST NOT move,
            # consume, encode or fuse it.
            if not first_batch_checked:
                if (
                    rgb.ndim != 4
                    or rgb.shape[1] != 3
                ):
                    raise RuntimeError(
                        f"Bad RGB batch shape: {tuple(rgb.shape)}"
                    )

                if labels.ndim != 3:
                    raise RuntimeError(
                        f"Bad labels shape: {tuple(labels.shape)}"
                    )

                if not torch.isfinite(
                    rgb
                ).all().item():
                    raise FloatingPointError(
                        "Input RGB contains NaN/Inf."
                    )

                valid_labels = (
                    labels[
                        labels
                        != IGNORE_INDEX
                    ]
                )

                if valid_labels.numel() > 0:
                    label_min = int(
                        valid_labels.min().item()
                    )
                    label_max = int(
                        valid_labels.max().item()
                    )

                    if (
                        label_min < 0
                        or label_max
                        >= NUM_CLASSES
                    ):
                        raise RuntimeError(
                            "Label range invalid: "
                            f"{label_min}..{label_max}"
                        )

                first_batch_checked = True

                print(
                    "[first batch] PASS | "
                    f"RGB={tuple(rgb.shape)} | "
                    f"labels={tuple(labels.shape)} | "
                    "NIR intentionally unused"
                )

            if args.regime == "robust3":
                (
                    rgb_model,
                    _unused_gate_target,
                    counts,
                    severities,
                ) = degrade_rgb_batch(
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

            else:
                rgb_model = rgb

                family_counts[
                    "clean"
                ] += int(
                    rgb.shape[0]
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
                    ignore_index=IGNORE_INDEX,
                )

                lovasz = lovasz_softmax(
                    raw_logits,
                    labels,
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
                    f"Non-finite loss at "
                    f"epoch={epoch + 1}, "
                    f"batch={batch_index}."
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
                    group["lr"]
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
                eta_seconds, finish = (
                    progress.estimates(
                        epoch_zero_based=epoch,
                        batch_zero_based=batch_index,
                    )
                )

                event = {
                    "event": (
                        "batch_log"
                    ),
                    "time_local": (
                        format_local_datetime(
                            local_now()
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
                    "batches_per_epoch": (
                        len(loader)
                    ),
                    "global_step": (
                        global_step
                    ),
                    "total_updates": (
                        total_updates
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
                    "lr_groups": lrs,
                    "cuda_memory": (
                        cuda_memory_text(
                            device
                        )
                    ),
                    "batch_seconds": (
                        batch_seconds
                    ),
                    "eta_seconds": (
                        eta_seconds
                    ),
                    "estimated_finish_local": (
                        format_local_datetime(
                            finish
                        )
                        if finish
                        is not None
                        else None
                    ),
                }

                append_jsonl(
                    log_path,
                    event,
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

        if (
            total_valid <= 0
            or mini_batches <= 0
        ):
            raise RuntimeError(
                "Epoch contains no valid training data."
            )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        eta_seconds, finish = (
            progress.estimates(
                epoch_zero_based=epoch,
                batch_zero_based=(
                    len(loader)
                    - 1
                ),
            )
        )

        epoch_record = {
            "event": (
                "epoch_done"
            ),
            "time_local": (
                format_local_datetime(
                    local_now()
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
                format_duration(
                    epoch_seconds
                )
            ),
            "lr_groups": {
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
            },
            "eta_seconds": (
                eta_seconds
            ),
            "estimated_finish_local": (
                format_local_datetime(
                    finish
                )
                if finish
                is not None
                else None
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
                / (
                    f"epoch_"
                    f"{epoch + 1:03d}.pt"
                ),
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/{args.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz="
            f"{epoch_record['train_lovasz_mean']:.6f} | "
            f"Total="
            f"{epoch_record['train_total_loss_mean']:.6f} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"epoch={format_duration(epoch_seconds)} | "
            f"ETA={format_duration(eta_seconds)} | "
            f"finish="
            f"{epoch_record['estimated_finish_local']} | "
            f"updates={updates} | "
            f"skips={amp_skips} | "
            f"global_step={global_step}"
        )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

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

    finished_local = (
        local_now()
    )

    print("=" * 118)
    print(
        f"[finished] {model_name}"
    )
    print(
        f"[finished] epochs      : {args.epochs}"
    )
    print(
        f"[finished] global_step : {global_step}"
    )
    print(
        f"[finished] duration    : "
        f"{format_duration(total_seconds)}"
    )
    print(
        f"[finished] local time  : "
        f"{format_local_datetime(finished_local)}"
    )
    print(
        f"[finished] checkpoint  : "
        f"{checkpoint_dir / 'final.pt'}"
    )
    print(
        f"[finished] train log   : "
        f"{log_path}"
    )
    print(
        f"[finished] progress    : "
        f"{output_dir / 'progress.json'}"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
