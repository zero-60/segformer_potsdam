#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train M3: SegFormer-B2 + DARF + Robust-4.

M0: B2-RGB-Clean
M1: B2-RGB-Robust4
M2: B2-RGBNIR-Fixed-Robust4
M3: B2-DARF-Robust4  <-- this script

Fusion:
    F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)

Robust-4 families:
    Gaussian noise
    Gaussian blur
    RGB underexposure
    Atmospheric fog

Only RGB is degraded. NIR and labels remain unchanged.

Gate supervision:
    Clean:    target g_NIR = 0.05
    Degraded: target g_NIR = 0.15 + 0.80 * severity

Loss:
    CE + 0.50 * LovaszSoftmax + 0.10 * GateBCEWithLogits

Run:
    python tools/train_model_b2_darf_robust4.py

Resume:
    python tools/train_model_b2_darf_robust4.py --resume auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    INITIAL_NIR_GATE,
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
from train_model_b2_rgb_robust4 import (
    ROBUST4_FAMILIES,
    degrade_rgb_batch_robust4,
)

MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "B2 DARF Robust-4 Protocol v1"
MODEL_ID = "B2_DARF_ROBUST4"
MODEL_NAME = "SegFormer-B2 DARF Robust-4"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "training" / "b2_darf_robust4"
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
DEFAULT_CLEAN_WARMUP_EPOCHS = 10
DEFAULT_CORRUPTION_RAMP_EPOCHS = 20
DEFAULT_MAX_CORRUPTION_PROB = 0.50
DEFAULT_PROGRESS_EVERY = 5
DEFAULT_ETA_WARMUP_BATCHES = 20
DEFAULT_ETA_EMA_ALPHA = 0.08

GATE_TARGET_DEGRADED_BASE = 0.15
GATE_TARGET_SEVERITY_SCALE = 0.80


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


def make_gate_target(severities: torch.Tensor) -> torch.Tensor:
    if severities.ndim != 1:
        raise RuntimeError(
            f"Expected severity [B], got {tuple(severities.shape)}"
        )

    target = torch.full_like(
        severities,
        float(INITIAL_NIR_GATE),
        dtype=torch.float32,
    )

    degraded = severities > 0

    if degraded.any():
        target[degraded] = (
            GATE_TARGET_DEGRADED_BASE
            + GATE_TARGET_SEVERITY_SCALE
            * severities[degraded].float()
        )

    if not torch.isfinite(target).all().item():
        raise FloatingPointError("Gate target contains NaN/Inf.")

    if float(target.min().item()) < 0.0 or float(target.max().item()) > 1.0:
        raise RuntimeError("Gate target escaped [0,1].")

    return target


def parse_args():
    p = argparse.ArgumentParser(
        description="Train M3 / SegFormer-B2 DARF Robust-4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help='Checkpoint path, or "auto" for latest.pt.',
    )
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
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
    p.add_argument("--no-progress", action="store_true")

    x = p.parse_args()

    for name, value in (
        ("epochs", x.epochs),
        ("batch-size", x.batch_size),
        ("grad-accum-steps", x.grad_accum_steps),
        ("base-lr", x.base_lr),
        ("new-lr", x.new_lr),
        ("grad-clip", x.grad_clip),
        ("log-every", x.log_every),
        ("progress-every", x.progress_every),
        ("eta-warmup-batches", x.eta_warmup_batches),
    ):
        if value <= 0:
            p.error(f"--{name} must be > 0")

    if x.weight_decay < 0 or x.warmup_steps < 0:
        p.error("weight-decay/warmup must be non-negative")
    if x.lovasz_weight < 0 or x.gate_weight < 0:
        p.error("loss weights must be non-negative")
    if x.clean_warmup_epochs < 0 or x.corruption_ramp_epochs < 0:
        p.error("warm-up/ramp epochs must be non-negative")
    if not 0.0 <= x.max_corruption_prob <= 1.0:
        p.error("--max-corruption-prob must be in [0,1]")
    if not 0.0 < x.eta_ema_alpha <= 1.0:
        p.error("--eta-ema-alpha must be in (0,1]")

    return x


def build_protocol(
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
        "protocol_version": PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "backbone": "SegFormer-B2",
        "checkpoint": model_meta.get("checkpoint"),
        "input_modalities": ["RGB", "NIR"],
        "nir_used": True,
        "dual_encoder": True,
        "fusion": "RGB-anchored degradation-aware residual NIR fusion",
        "fusion_rule": "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)",
        "quality_gate": True,
        "gate_type": GATE_TYPE,
        "gate_supervision": True,
        "gate_target": {
            "clean": float(INITIAL_NIR_GATE),
            "degraded_formula": "0.15 + 0.80 * severity",
            "applies_to_families": list(ROBUST4_FAMILIES),
        },
        "regime": "robust4",
        "epochs": x.epochs,
        "batch_size": x.batch_size,
        "grad_accum_steps": x.grad_accum_steps,
        "effective_batch_size_nominal": x.batch_size * x.grad_accum_steps,
        "base_lr": x.base_lr,
        "new_lr": x.new_lr,
        "weight_decay": x.weight_decay,
        "warmup_steps": warmup,
        "grad_clip": x.grad_clip,
        "loss": {
            "ce_weight": 1.0,
            "lovasz_weight": x.lovasz_weight,
            "gate_bce_weight": x.gate_weight,
        },
        "corruption_training": {
            "enabled": True,
            "version": "Robust-4",
            "fog_in_training": True,
            "clean_warmup_epochs": x.clean_warmup_epochs,
            "ramp_epochs": x.corruption_ramp_epochs,
            "max_probability": x.max_corruption_prob,
            "families": list(ROBUST4_FAMILIES),
            "family_sampling": "uniform",
            "severity_sampling": "continuous Uniform[0.10,1.00]",
            "rgb": "degraded according to Robust-4",
            "nir": "clean / unchanged",
            "labels": "unchanged",
            "implementation_source": (
                "degrade_rgb_batch_robust4 from "
                "tools/train_model_b2_rgb_robust4.py"
            ),
        },
        "optimizer": optimizer_summary,
        "updates_per_epoch": updates_per_epoch,
        "total_update_steps": total_updates,
        "seed": x.seed,
        "data_seed": int(DATA_SEED),
        "amp": amp,
        "gradient_checkpointing_requested": x.gradient_checkpointing,
        "gradient_checkpointing_status": checkpointing_status,
        "device": str(device),
        "data_cache_dir": str(cache),
        "model_meta": dict(model_meta),
        "scientific_note": (
            "M3 keeps the M2 dual-encoder/NIR-residual/Robust-4 setup but "
            "replaces fixed g=0.5 with the learned degradation-aware gate "
            "and its auxiliary gate supervision."
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
        "regime": "robust4",
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


def resolve_resume(value: str, checkpoint_dir: Path) -> Optional[Path]:
    if not value:
        return None

    if value.lower() == "auto":
        path = checkpoint_dir / "latest.pt"
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_resume(*, path, model, optimizer, scheduler, scaler):
    c = torch.load(path, map_location="cpu", weights_only=False)

    if c.get("model_id") != MODEL_ID:
        raise RuntimeError("Resume checkpoint model_id mismatch.")
    if c.get("regime") != "robust4":
        raise RuntimeError("Resume checkpoint is not Robust-4.")

    model.load_state_dict(c["model"], strict=True)
    optimizer.load_state_dict(c["optimizer"])
    scheduler.load_state_dict(c["scheduler"])
    scaler.load_state_dict(c["scaler"])
    restore_rng_state(c.get("rng_state"))

    return (
        int(c["epoch"]) + 1,
        int(c.get("global_step", c.get("step", 0))),
    )



def main():
    x = parse_args()
    seed_everything(x.seed)

    out = resolve(x.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cache = resolve(x.data_cache_dir)

    print("=" * 126)
    print(MODEL_NAME)
    print("=" * 126)
    print(f"model id       : {MODEL_ID}")
    print("regime         : robust4")
    print("modalities     : RGB + clean NIR")
    print("fusion         : dynamic DARF gate")
    print(f"initial g_NIR  : {INITIAL_NIR_GATE:.3f}")
    print(f"gate loss      : {x.gate_weight:.3f}")
    print("families       : Noise / Blur / Underexposure / Fog")
    print(f"output         : {out}")
    print(
        f"start local    : "
        f"{progress_base.format_local_datetime(progress_base.local_now())}"
    )
    print("=" * 126)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    dataset = CachedPotsdamTrainDataset(
        PROJECT_ROOT,
        epoch=0,
        cache_root=cache,
    )
    dataset.prepare_cache(
        rebuild=x.rebuild_data_cache
    )

    sampler = DeterministicEpochSampler(
        dataset,
        seed=DATA_SEED,
        epoch=0,
    )

    loader = build_train_dataloader(
        dataset,
        sampler,
        batch_size=x.batch_size,
        num_workers=x.num_workers,
        pin_memory=x.pin_memory,
    )

    updates_per_epoch = math.ceil(
        len(loader) / x.grad_accum_steps
    )
    total_updates = updates_per_epoch * x.epochs
    warmup = min(
        x.warmup_steps,
        max(0, total_updates - 1),
    )

    # -------------------------------------------------------------------------
    # Model / optimizer
    # -------------------------------------------------------------------------

    device = progress_base.get_device(
        x.device
    )

    model, architecture_meta = build_model_d_darf_b2(
        PROJECT_ROOT
    )

    model_meta = dict(
        architecture_meta
    )
    model_meta["architecture_model_id"] = architecture_meta.get(
        "model_id"
    )
    model_meta["architecture_protocol_version"] = architecture_meta.get(
        "protocol_version"
    )
    model_meta["model_id"] = MODEL_ID
    model_meta["name"] = MODEL_NAME
    model_meta["training_protocol_version"] = PROTOCOL_VERSION
    model_meta["training_regime"] = "robust4"

    checkpointing_status = {
        "rgb_encoder": False,
        "nir_encoder": False,
    }

    if x.gradient_checkpointing:
        checkpointing_status = (
            model.enable_gradient_checkpointing()
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
        darf_base.build_optimizer(
            model,
            base_lr=x.base_lr,
            new_lr=x.new_lr,
            weight_decay=x.weight_decay,
        )
    )

    scheduler = build_poly_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup,
        total_steps=total_updates,
        power=1.0,
    )

    amp = (
        device.type == "cuda"
        and not x.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
    )

    protocol = build_protocol(
        x=x,
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
        out / "protocol.json",
        protocol,
    )

    print(f"parameters      : {model_meta['parameters']['total']:,}")
    print(f"hidden sizes    : {model_meta['hidden_sizes']}")
    print(f"grad checkpoint : {checkpointing_status}")
    print(f"epochs          : {x.epochs}")
    print(f"batches/epoch   : {len(loader)}")
    print(f"updates/epoch   : {updates_per_epoch}")
    print(f"total updates   : {total_updates}")
    print(
        f"effective batch : "
        f"{x.batch_size * x.grad_accum_steps}"
    )
    print(
        f"loss            : "
        f"CE + {x.lovasz_weight} Lovasz + "
        f"{x.gate_weight} GateBCE"
    )
    print(f"max p_corrupt   : {x.max_corruption_prob}")

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0
    global_step = 0

    rp = resolve_resume(
        x.resume,
        ckpt_dir,
    )

    if rp is not None:
        start_epoch, global_step = load_resume(
            path=rp,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print(
            f"[resume] {rp} | "
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
    # Live progress
    # -------------------------------------------------------------------------

    progress = progress_base.LiveTrainingProgress(
        total_epochs=x.epochs,
        batches_per_epoch=len(loader),
        start_epoch=start_epoch,
        progress_every=x.progress_every,
        eta_warmup_batches=x.eta_warmup_batches,
        ema_alpha=x.eta_ema_alpha,
        output_path=out / "progress.json",
        device=device,
        enabled=(not x.no_progress),
    )

    log_path = out / "train_log.jsonl"
    training_start = time.time()
    first_batch_checked = False

    print(
        "[training] Robust-4 + DARF started. "
        "ETA will stabilize after the calibration batches.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        x.epochs,
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

        p_corrupt = darf_base.corruption_probability(
            epoch,
            clean_warmup_epochs=x.clean_warmup_epochs,
            ramp_epochs=x.corruption_ramp_epochs,
            maximum=x.max_corruption_prob,
        )

        total_valid = 0
        ce_numerator = 0.0
        lovasz_sum = 0.0
        gate_loss_sum = 0.0
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
            "fog": 0,
        }

        severity_sum = 0.0
        degraded_samples = 0

        gate_strength_sum = torch.zeros(
            NUM_SCALES,
            dtype=torch.float64,
        )
        gate_strength_sq_sum = torch.zeros(
            NUM_SCALES,
            dtype=torch.float64,
        )
        gate_samples = 0

        gate_target_sum = 0.0
        gate_target_count = 0

        accumulation_target = (
            x.grad_accum_steps
        )

        for batch_index, batch in enumerate(
            loader
        ):
            batch_started = time.monotonic()

            if (
                batch_index
                % x.grad_accum_steps
                == 0
            ):
                remaining = (
                    len(loader)
                    - batch_index
                )
                accumulation_target = min(
                    x.grad_accum_steps,
                    remaining,
                )

            rgb = batch["rgb"].to(
                device,
                non_blocking=x.pin_memory,
            )

            nir = batch["nir"].to(
                device,
                non_blocking=x.pin_memory,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=x.pin_memory,
            ).long()

            if not first_batch_checked:
                if (
                    rgb.ndim != 4
                    or rgb.shape[1] != 3
                ):
                    raise RuntimeError(
                        f"Bad RGB batch: {tuple(rgb.shape)}"
                    )

                if (
                    nir.ndim != 4
                    or nir.shape[1] != 1
                ):
                    raise RuntimeError(
                        f"Bad NIR batch: {tuple(nir.shape)}"
                    )

                if (
                    rgb.shape[0] != nir.shape[0]
                    or rgb.shape[-2:] != nir.shape[-2:]
                ):
                    raise RuntimeError(
                        "RGB/NIR batch alignment failed."
                    )

                if labels.ndim != 3:
                    raise RuntimeError(
                        f"Bad labels: {tuple(labels.shape)}"
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
                counts,
                severities,
            ) = degrade_rgb_batch_robust4(
                rgb,
                probability=p_corrupt,
            )

            # Formal protocol: NIR stays clean.
            nir_model = nir

            gate_target = make_gate_target(
                severities
            )

            for key in family_counts:
                family_counts[key] += int(
                    counts[key]
                )

            degraded_mask = (
                severities > 0
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
                    nir_model,
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

                if (
                    gate_strength.ndim != 2
                    or tuple(
                        gate_strength.shape
                    )
                    != (
                        rgb.shape[0],
                        NUM_SCALES,
                    )
                ):
                    raise RuntimeError(
                        "Unexpected gate strength shape: "
                        f"{tuple(gate_strength.shape)}"
                    )

                if (
                    gate_logits.shape
                    != gate_strength.shape
                ):
                    raise RuntimeError(
                        "Gate logits/strength shape mismatch."
                    )

                ce = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=IGNORE_INDEX,
                )

                lovasz = darf_base.lovasz_softmax(
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

                gate_loss = (
                    F.binary_cross_entropy_with_logits(
                        gate_logits.float(),
                        target_matrix.float(),
                    )
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
                    f"Non-finite loss at "
                    f"epoch={epoch + 1}, "
                    f"batch={batch_index + 1}."
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

            total_valid += valid_pixels

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

            gate_loss_sum += float(
                gate_loss
                .detach()
                .item()
            )

            total_loss_sum += float(
                total_loss
                .detach()
                .item()
            )

            gate_cpu = (
                gate_strength
                .detach()
                .float()
                .cpu()
                .double()
            )

            gate_strength_sum += (
                gate_cpu
                .sum(
                    dim=0
                )
            )

            gate_strength_sq_sum += (
                (
                    gate_cpu
                    * gate_cpu
                )
                .sum(
                    dim=0
                )
            )

            gate_samples += int(
                gate_cpu.shape[0]
            )

            gate_target_sum += float(
                gate_target
                .detach()
                .float()
                .sum()
                .item()
            )

            gate_target_count += int(
                gate_target.numel()
            )

            mini_batches += 1

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
                        x.grad_clip,
                    )
                )

                last_grad_norm = float(
                    grad_norm
                    .detach()
                    .item()
                )

                before_scale = (
                    scaler.get_scale()
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                after_scale = (
                    scaler.get_scale()
                )

                step_happened = (
                    not amp
                    or after_scale
                    >= before_scale
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
                    x.base_lr,
                ),
                new_lr=lrs.get(
                    "new_decay",
                    x.new_lr,
                ),
                p_corrupt=p_corrupt,
            )

            if (
                batch_index
                % x.log_every
                == 0
                or batch_index
                + 1
                == len(loader)
            ):
                batch_gate_mean = (
                    gate_strength
                    .detach()
                    .float()
                    .mean(
                        dim=0
                    )
                    .cpu()
                    .tolist()
                )

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
                        "gate_bce": float(
                            gate_loss
                            .detach()
                            .item()
                        ),
                        "gate_target_mean": float(
                            gate_target
                            .detach()
                            .float()
                            .mean()
                            .item()
                        ),
                        "gate_strength_mean": [
                            float(v)
                            for v
                            in batch_gate_mean
                        ],
                        "corruption_probability": (
                            p_corrupt
                        ),
                        "family_counts_running": dict(
                            family_counts
                        ),
                        "lr_groups": lrs,
                        "batch_seconds": (
                            batch_seconds
                        ),
                    },
                )

            del (
                details,
                logits,
                raw_logits,
                gate_strength,
                gate_logits,
                gate_cpu,
                gate_target,
                target_matrix,
                gate_loss,
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
            total_valid <= 0
            or mini_batches <= 0
            or gate_samples <= 0
            or gate_target_count <= 0
        ):
            raise RuntimeError(
                "Epoch contains no valid training/gate samples."
            )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        gate_mean = (
            gate_strength_sum
            / gate_samples
        )

        gate_second_moment = (
            gate_strength_sq_sum
            / gate_samples
        )

        gate_var = torch.clamp(
            gate_second_moment
            - gate_mean
            * gate_mean,
            min=0.0,
        )

        gate_std = torch.sqrt(
            gate_var
        )

        gate_target_mean = (
            gate_target_sum
            / gate_target_count
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
            "train_gate_bce_mean": (
                gate_loss_sum
                / mini_batches
            ),
            "train_total_loss_mean": (
                total_loss_sum
                / mini_batches
            ),
            "gate_target_mean": (
                gate_target_mean
            ),
            "gate_nir_strength_mean_by_scale": [
                float(v)
                for v
                in gate_mean.tolist()
            ],
            "gate_nir_strength_std_by_scale": [
                float(v)
                for v
                in gate_std.tolist()
            ],
            "corruption_probability": (
                p_corrupt
            ),
            "corruption_family_counts": dict(
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
                progress_base
                .format_duration(
                    epoch_seconds
                )
            ),
            "eta_seconds": (
                eta_seconds
            ),
            "estimated_finish_local": (
                progress_base
                .format_local_datetime(
                    finish
                )
                if finish
                is not None
                else None
            ),
            "lr_groups": lrs,
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
                / (
                    f"epoch_"
                    f"{epoch + 1:03d}.pt"
                ),
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/{x.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"GateBCE={epoch_record['train_gate_bce_mean']:.6f} | "
            f"Total={epoch_record['train_total_loss_mean']:.6f} | "
            f"target={gate_target_mean:.4f} | "
            f"g={[round(float(v), 4) for v in gate_mean.tolist()]} | "
            f"p_corrupt={p_corrupt:.3f} | "
            f"families={family_counts} | "
            f"epoch={progress_base.format_duration(epoch_seconds)} | "
            f"ETA={progress_base.format_duration(eta_seconds)} | "
            f"finish="
            f"{epoch_record['estimated_finish_local']} | "
            f"global_step={global_step}",
            flush=True,
        )

    final = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=x.epochs - 1,
        global_step=global_step,
        protocol=protocol,
        model_meta=model_meta,
    )

    atomic_torch_save(
        final,
        ckpt_dir
        / "final.pt",
    )

    progress.finish(
        output_dir=out
    )

    total_seconds = (
        time.time()
        - training_start
    )

    print("=" * 126)
    print(
        f"[finished] {MODEL_NAME}"
    )
    print(
        f"[finished] duration   : "
        f"{progress_base.format_duration(total_seconds)}"
    )
    print(
        f"[finished] checkpoint : "
        f"{ckpt_dir / 'final.pt'}"
    )
    print(
        f"[finished] protocol   : "
        f"{out / 'protocol.json'}"
    )
    print(
        f"[finished] train log  : "
        f"{log_path}"
    )
    print(
        f"[finished] progress   : "
        f"{out / 'progress.json'}"
    )
    print("=" * 126)


if __name__ == "__main__":
    main()
