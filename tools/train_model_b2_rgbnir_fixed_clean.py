#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train SegFormer-B2 RGB+NIR fixed residual fusion on CLEAN Potsdam data.

Project placement
-----------------
Copy this file to:
    tools/train_model_b2_rgbnir_fixed_clean.py

This experiment intentionally contains NO synthetic imaging corruption:
- no Gaussian noise
- no Gaussian blur
- no underexposure
- no fog
- no modality dropout/misalignment

The frozen project data protocol is preserved, including deterministic training
crop selection and deterministic D4 geometric augmentation. D4 is retained
because it is part of the existing sampling protocol rather than an imaging
noise/degradation condition.

Default experiment
------------------
- SegFormer-B2 dual RGB/NIR encoders
- RGB-anchored NIR residual fusion
- fixed NIR strength = 1.0
- 120 epochs
- batch size 1
- gradient accumulation 16
- base/pretrained LR 3e-5
- new adapter/classifier LR 3e-4
- weight decay 0.01
- warmup 500 successful optimizer updates
- CE + 0.5 * Lovasz-Softmax
- AMP on CUDA
- gradient clip 1.0

Run
---
    python tools/train_model_b2_rgbnir_fixed_clean.py

Resume
------
    python tools/train_model_b2_rgbnir_fixed_clean.py --resume auto

Run another fixed coefficient later without editing code
--------------------------------------------------------
    python tools/train_model_b2_rgbnir_fixed_clean.py --fixed-nir-strength 0.5
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
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_pipeline.potsdam_dataloader import (  # noqa: E402
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from models.segformer_rgb import IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from train_model_a_rgb import (  # noqa: E402
    CachedPotsdamTrainDataset,
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_NUM_WORKERS,
    atomic_torch_save,
    build_poly_scheduler,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
import train_model_b2_rgb as progress_base  # noqa: E402
from models.segformer_b2_rgbnir_fixed_clean import (  # noqa: E402
    build_model_b2_rgbnir_fixed_clean,
)


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "B2 RGBNIR Fixed Clean Protocol v1"
MODEL_ID = "B2_RGBNIR_FIXED_CLEAN"
MODEL_NAME = "SegFormer-B2 RGB+NIR Fixed Residual Fusion Clean"

DEFAULT_SEED = 20260917
DEFAULT_FIXED_NIR_STRENGTH = 1.0
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_BASE_LR = 3e-5
DEFAULT_NEW_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP = 500
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LOVASZ_WEIGHT = 0.50
DEFAULT_LOG_EVERY = 50


def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def strength_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return "g" + text.replace("-", "m").replace(".", "p")


def default_output_dir(strength: float, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "training"
        / f"b2_rgbnir_fixed_clean_{strength_tag(strength)}_seed{int(seed)}"
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


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = int(gt_sorted.numel())
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard


def lovasz_softmax(
    raw_logits: torch.Tensor,
    labels_full: torch.Tensor,
) -> torch.Tensor:
    """Project-compatible Lovasz loss on raw 1/4-resolution SegFormer logits."""
    labels = (
        F.interpolate(
            labels_full.unsqueeze(1).float(),
            size=raw_logits.shape[-2:],
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )
    probabilities = torch.softmax(raw_logits.float(), dim=1)
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
    labels = labels.reshape(-1)
    valid = labels != IGNORE_INDEX
    probabilities = probabilities[valid]
    labels = labels[valid]

    if labels.numel() == 0:
        return raw_logits.sum() * 0.0

    losses = []
    for class_id in range(NUM_CLASSES):
        foreground = (labels == class_id).float()
        if foreground.sum().item() == 0:
            continue
        class_probability = probabilities[:, class_id]
        errors = torch.abs(foreground - class_probability)
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground_sorted)))

    if not losses:
        return raw_logits.sum() * 0.0
    return torch.stack(losses).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train clean Potsdam RGB+NIR fixed residual SegFormer-B2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fixed-nir-strength", type=float, default=DEFAULT_FIXED_NIR_STRENGTH)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--base-lr", type=float, default=DEFAULT_BASE_LR)
    p.add_argument("--new-lr", type=float, default=DEFAULT_NEW_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--lovasz-weight", type=float, default=DEFAULT_LOVASZ_WEIGHT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-cache-dir", type=Path, default=DEFAULT_DATA_CACHE_DIR)
    p.add_argument("--rebuild-data-cache", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--resume",
        default="",
        help='Checkpoint path, or "auto" for <output-dir>/checkpoints/latest.pt.',
    )
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    x = p.parse_args()

    if not (0.0 <= x.fixed_nir_strength <= 1.0):
        p.error("--fixed-nir-strength must be in [0,1]")
    for name, value in (
        ("epochs", x.epochs),
        ("batch-size", x.batch_size),
        ("grad-accum-steps", x.grad_accum_steps),
        ("base-lr", x.base_lr),
        ("new-lr", x.new_lr),
        ("grad-clip", x.grad_clip),
        ("log-every", x.log_every),
    ):
        if value <= 0:
            p.error(f"--{name} must be > 0")
    if x.weight_decay < 0:
        p.error("--weight-decay must be >= 0")
    if x.warmup_steps < 0:
        p.error("--warmup-steps must be >= 0")
    if x.lovasz_weight < 0:
        p.error("--lovasz-weight must be >= 0")
    if x.num_workers < 0:
        p.error("--num-workers must be >= 0")
    return x


def build_optimizer(
    model,
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
        is_new = (
            name.startswith("nir_adapters.")
            or name.startswith("decode_head.classifier.")
        )
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
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
    groups = [group for group in groups if group["params"]]
    optimizer = AdamW(groups)
    summary = {
        group["group_name"]: {
            "parameters": int(sum(p.numel() for p in group["params"])),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in groups
    }
    return optimizer, summary


def build_protocol(
    *,
    x: argparse.Namespace,
    model_meta: Mapping[str, Any],
    optimizer_summary: Mapping[str, Any],
    updates_per_epoch: int,
    total_updates: int,
    warmup_steps: int,
    device: torch.device,
    amp: bool,
    cache_dir: Path,
    checkpointing_status: Mapping[str, Any],
    nir_mean: float,
    nir_std: float,
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "backbone": "SegFormer-B2",
        "regime": "clean",
        "input_modalities": ["RGB", "NIR"],
        "dual_encoder": True,
        "fusion": "RGB-anchored fixed-strength NIR residual fusion",
        "fusion_rule": (
            f"F_i = F_RGB_i + {float(x.fixed_nir_strength):.8g} * Adapter_i(F_NIR_i)"
        ),
        "fixed_nir_strength": float(x.fixed_nir_strength),
        "quality_gate": False,
        "synthetic_corruption": {
            "enabled": False,
            "gaussian_noise": False,
            "gaussian_blur": False,
            "underexposure": False,
            "fog": False,
            "modality_dropout": False,
            "misregistration": False,
        },
        "data_protocol": {
            "train_dataset": "CachedPotsdamTrainDataset (same frozen PotsdamTrainDataset semantics)",
            "deterministic_epoch_sampler": True,
            "data_seed": int(DATA_SEED),
            "train_crop_size": 512,
            "train_geometric_d4": True,
            "note": "D4 is retained as frozen geometric augmentation; no imaging degradation is injected.",
            "nir_normalization": {
                "mean": float(nir_mean),
                "std": float(nir_std),
            },
        },
        "epochs": int(x.epochs),
        "batch_size": int(x.batch_size),
        "grad_accum_steps": int(x.grad_accum_steps),
        "effective_batch_size_nominal": int(x.batch_size * x.grad_accum_steps),
        "base_lr": float(x.base_lr),
        "new_lr": float(x.new_lr),
        "weight_decay": float(x.weight_decay),
        "warmup_steps": int(warmup_steps),
        "grad_clip": float(x.grad_clip),
        "loss": {
            "cross_entropy_weight": 1.0,
            "lovasz_softmax_weight": float(x.lovasz_weight),
        },
        "optimizer": dict(optimizer_summary),
        "updates_per_epoch": int(updates_per_epoch),
        "total_update_steps": int(total_updates),
        "seed": int(x.seed),
        "device": str(device),
        "amp": bool(amp),
        "gradient_checkpointing_requested": bool(x.gradient_checkpointing),
        "gradient_checkpointing_status": dict(checkpointing_status),
        "data_cache_dir": str(cache_dir),
        "model_meta": dict(model_meta),
    }


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    model_meta: Mapping[str, Any],
    fixed_nir_strength: float,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "regime": "clean",
        "fixed_nir_strength": float(fixed_nir_strength),
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


def load_resume(
    *,
    path: Path,
    expected_strength: float,
    model,
    optimizer,
    scheduler,
    scaler,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"Resume checkpoint model_id mismatch: {checkpoint.get('model_id')} != {MODEL_ID}"
        )
    if checkpoint.get("regime") != "clean":
        raise RuntimeError("Resume checkpoint is not a clean-regime checkpoint.")
    ckpt_strength = float(
        checkpoint.get(
            "fixed_nir_strength",
            checkpoint.get("protocol", {}).get("fixed_nir_strength", float("nan")),
        )
    )
    if not math.isclose(ckpt_strength, float(expected_strength), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Resume fixed NIR strength mismatch: checkpoint={ckpt_strength}, "
            f"requested={expected_strength}"
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["epoch"]) + 1, int(
        checkpoint.get("global_step", checkpoint.get("step", 0))
    )


def main() -> None:
    x = parse_args()
    seed_everything(x.seed)

    output_dir = (
        resolve(x.output_dir)
        if x.output_dir is not None
        else default_output_dir(x.fixed_nir_strength, x.seed)
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve(x.data_cache_dir)

    print("=" * 120)
    print(MODEL_NAME)
    print("=" * 120)
    print(f"regime              : clean (NO synthetic imaging degradation)")
    print(f"fixed NIR strength  : {x.fixed_nir_strength:.6f}")
    print(f"seed                : {x.seed}")
    print(f"output              : {output_dir}")
    print("=" * 120)

    dataset = CachedPotsdamTrainDataset(
        PROJECT_ROOT,
        epoch=0,
        cache_root=cache_dir,
    )
    dataset.prepare_cache(rebuild=x.rebuild_data_cache)

    nir_mean = float(dataset.spec.nir_mean)
    nir_std = float(dataset.spec.nir_std)
    if not math.isfinite(nir_mean) or not math.isfinite(nir_std) or nir_std <= 0:
        raise RuntimeError(f"Invalid frozen NIR normalization: mean={nir_mean}, std={nir_std}")

    sampler = DeterministicEpochSampler(dataset, seed=DATA_SEED, epoch=0)
    loader = build_train_dataloader(
        dataset,
        sampler,
        batch_size=x.batch_size,
        num_workers=x.num_workers,
        pin_memory=x.pin_memory,
    )

    updates_per_epoch = math.ceil(len(loader) / x.grad_accum_steps)
    total_updates = updates_per_epoch * x.epochs
    warmup_steps = min(x.warmup_steps, max(0, total_updates - 1))

    device = progress_base.get_device(x.device)
    model, model_meta = build_model_b2_rgbnir_fixed_clean(
        PROJECT_ROOT,
        fixed_strength=x.fixed_nir_strength,
    )
    model_meta = dict(model_meta)
    model_meta["experiment_model_id"] = MODEL_ID
    model_meta["experiment_model_name"] = MODEL_NAME
    model_meta["experiment_protocol_version"] = PROTOCOL_VERSION
    model_meta["fixed_nir_strength"] = float(x.fixed_nir_strength)
    model_meta["fusion_rule"] = (
        f"F_i = F_RGB_i + {float(x.fixed_nir_strength):.8g} * Adapter_i(F_NIR_i)"
    )

    optimizer, optimizer_summary = build_optimizer(
        model,
        base_lr=x.base_lr,
        new_lr=x.new_lr,
        weight_decay=x.weight_decay,
    )

    checkpointing_status: Dict[str, Any] = {
        "rgb_encoder": False,
        "nir_encoder": False,
    }
    if x.gradient_checkpointing:
        checkpointing_status = model.enable_gradient_checkpointing()

    model.to(device)
    scheduler = build_poly_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_updates,
        power=1.0,
    )

    amp = device.type == "cuda" and not x.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    protocol = build_protocol(
        x=x,
        model_meta=model_meta,
        optimizer_summary=optimizer_summary,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        warmup_steps=warmup_steps,
        device=device,
        amp=amp,
        cache_dir=cache_dir,
        checkpointing_status=checkpointing_status,
        nir_mean=nir_mean,
        nir_std=nir_std,
    )
    write_json_atomic(output_dir / "protocol.json", protocol)

    print(f"train samples/epoch : {len(dataset)}")
    print(f"batches/epoch       : {len(loader)}")
    print(f"updates/epoch       : {updates_per_epoch}")
    print(f"total updates       : {total_updates}")
    print(f"effective batch     : {x.batch_size * x.grad_accum_steps}")
    print(f"loss                : CE + {x.lovasz_weight} * Lovasz")
    print(f"AMP                 : {amp}")
    print(f"NIR norm            : mean={nir_mean:.9f}, std={nir_std:.9f}")

    start_epoch = 0
    global_step = 0
    resume_path = resolve_resume(x.resume, checkpoint_dir)
    if resume_path is not None:
        start_epoch, global_step = load_resume(
            path=resume_path,
            expected_strength=x.fixed_nir_strength,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(
            f"[resume] {resume_path} | start_epoch={start_epoch} | global_step={global_step}",
            flush=True,
        )

    if start_epoch >= x.epochs:
        print("[done] requested epochs already completed.")
        return

    log_path = output_dir / "train_log.jsonl"
    progress_path = output_dir / "progress.json"
    training_started = time.time()
    first_batch_checked = False

    for epoch in range(start_epoch, x.epochs):
        epoch_started = time.time()
        set_train_epoch(dataset, sampler, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_valid = 0
        ce_numerator = 0.0
        lovasz_sum = 0.0
        total_loss_sum = 0.0
        mini_batches = 0
        optimizer_updates = 0
        amp_skipped_updates = 0
        last_grad_norm = float("nan")
        residual_sum = torch.zeros(4, dtype=torch.float64)
        residual_batches = 0
        accumulation_target = x.grad_accum_steps

        for batch_index, batch in enumerate(loader):
            if batch_index % x.grad_accum_steps == 0:
                remaining = len(loader) - batch_index
                accumulation_target = min(x.grad_accum_steps, remaining)

            rgb = batch["rgb"].to(device, non_blocking=x.pin_memory)
            nir = batch["nir"].to(device, non_blocking=x.pin_memory)
            labels = batch["labels"].to(device, non_blocking=x.pin_memory).long()

            if not first_batch_checked:
                if rgb.ndim != 4 or rgb.shape[1] != 3:
                    raise RuntimeError(f"Bad RGB batch: {tuple(rgb.shape)}")
                if nir.ndim != 4 or nir.shape[1] != 1:
                    raise RuntimeError(f"Bad NIR batch: {tuple(nir.shape)}")
                if rgb.shape[0] != nir.shape[0] or rgb.shape[-2:] != nir.shape[-2:]:
                    raise RuntimeError("RGB/NIR batch alignment failed.")
                if labels.ndim != 3 or labels.shape[-2:] != rgb.shape[-2:]:
                    raise RuntimeError(f"Bad labels batch: {tuple(labels.shape)}")
                valid_values = labels[labels != IGNORE_INDEX]
                if valid_values.numel() <= 0:
                    raise RuntimeError("First batch contains no valid labels.")
                if int(valid_values.min()) < 0 or int(valid_values.max()) > 5:
                    raise RuntimeError("First batch labels are outside class range 0..5.")
                first_batch_checked = True
                print(
                    "[first batch] PASS | clean tensors used directly | "
                    f"RGB={tuple(rgb.shape)} | NIR={tuple(nir.shape)} | labels={tuple(labels.shape)}",
                    flush=True,
                )

            # IMPORTANT: clean experiment. No degradation function is called here.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                details = model(rgb, nir, return_details=True)
                logits = details["logits"]
                raw_logits = details["raw_logits"]
                ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
                lovasz = lovasz_softmax(raw_logits, labels)
                total_loss = ce + x.lovasz_weight * lovasz
                loss_for_backward = total_loss / float(accumulation_target)

            if not torch.isfinite(total_loss.detach()).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}: "
                    f"CE={float(ce.detach())}, Lovasz={float(lovasz.detach())}"
                )

            scaler.scale(loss_for_backward).backward()

            valid_pixels = int((labels != IGNORE_INDEX).sum().item())
            total_valid += valid_pixels
            ce_numerator += float(ce.detach().item()) * valid_pixels
            lovasz_sum += float(lovasz.detach().item())
            total_loss_sum += float(total_loss.detach().item())
            mini_batches += 1

            residual = details["residual_abs_mean"].detach().float().cpu()
            if residual.numel() != 4:
                raise RuntimeError(f"Expected 4 residual scales, got {residual.numel()}")
            residual_sum += residual.to(dtype=torch.float64)
            residual_batches += 1

            group_end = (
                ((batch_index + 1) % x.grad_accum_steps == 0)
                or (batch_index + 1 == len(loader))
            )
            if group_end:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), x.grad_clip)
                last_grad_norm = float(grad_norm.detach().cpu().item())

                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                step_succeeded = (not amp) or (scale_after >= scale_before)

                if step_succeeded:
                    scheduler.step()
                    global_step += 1
                    optimizer_updates += 1
                else:
                    amp_skipped_updates += 1

                optimizer.zero_grad(set_to_none=True)

            if (batch_index + 1) % x.log_every == 0 or (batch_index + 1) == len(loader):
                lrs = [float(group["lr"]) for group in optimizer.param_groups]
                elapsed = time.time() - epoch_started
                print(
                    f"[train] epoch={epoch + 1:03d}/{x.epochs:03d} "
                    f"batch={batch_index + 1:04d}/{len(loader):04d} "
                    f"loss={float(total_loss.detach().item()):.6f} "
                    f"ce={float(ce.detach().item()):.6f} "
                    f"lovasz={float(lovasz.detach().item()):.6f} "
                    f"step={global_step} lr={lrs} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            del details, logits, raw_logits, total_loss, loss_for_backward, rgb, nir, labels

        if total_valid <= 0 or mini_batches <= 0 or residual_batches <= 0:
            raise RuntimeError("Epoch contains no valid training data.")

        epoch_seconds = time.time() - epoch_started
        residual_mean = residual_sum / residual_batches
        epoch_record: Dict[str, Any] = {
            "event": "epoch_done",
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer_updates_this_epoch": optimizer_updates,
            "amp_skipped_updates": amp_skipped_updates,
            "mini_batches": mini_batches,
            "valid_pixels": total_valid,
            "train_ce": ce_numerator / total_valid,
            "train_lovasz_mean": lovasz_sum / mini_batches,
            "train_total_loss_mean": total_loss_sum / mini_batches,
            "fixed_nir_strength": float(x.fixed_nir_strength),
            "synthetic_corruption_enabled": False,
            "residual_abs_mean_by_scale": [float(v) for v in residual_mean.tolist()],
            "last_grad_norm_before_clip": last_grad_norm,
            "epoch_seconds": epoch_seconds,
        }
        append_jsonl(log_path, epoch_record)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            protocol=protocol,
            model_meta=model_meta,
            fixed_nir_strength=x.fixed_nir_strength,
        )
        atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        if x.save_every > 0 and (epoch + 1) % x.save_every == 0:
            atomic_torch_save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")

        progress = {
            "model_id": MODEL_ID,
            "regime": "clean",
            "fixed_nir_strength": float(x.fixed_nir_strength),
            "epoch_completed": epoch + 1,
            "epochs_total": x.epochs,
            "global_step": global_step,
            "latest_checkpoint": str(checkpoint_dir / "latest.pt"),
            "last_epoch": epoch_record,
            "elapsed_seconds": time.time() - training_started,
        }
        write_json_atomic(progress_path, progress)

        print(
            f"[epoch done] {epoch + 1:03d}/{x.epochs:03d} | "
            f"CE={epoch_record['train_ce']:.6f} | "
            f"Lovasz={epoch_record['train_lovasz_mean']:.6f} | "
            f"Total={epoch_record['train_total_loss_mean']:.6f} | "
            f"residual={[round(float(v), 6) for v in epoch_record['residual_abs_mean_by_scale']]} | "
            f"seconds={epoch_seconds:.1f} | global_step={global_step}",
            flush=True,
        )

    final_payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=x.epochs - 1,
        global_step=global_step,
        protocol=protocol,
        model_meta=model_meta,
        fixed_nir_strength=x.fixed_nir_strength,
    )
    final_path = checkpoint_dir / "final.pt"
    atomic_torch_save(final_payload, final_path)

    final_summary = {
        "status": "finished",
        "model_id": MODEL_ID,
        "regime": "clean",
        "fixed_nir_strength": float(x.fixed_nir_strength),
        "seed": int(x.seed),
        "epochs": int(x.epochs),
        "global_step": int(global_step),
        "duration_seconds": float(time.time() - training_started),
        "final_checkpoint": str(final_path),
        "validation_command": (
            "python tools/eval_model_b2_rgbnir_fixed_clean.py "
            f"--checkpoint {final_path}"
        ),
    }
    write_json_atomic(output_dir / "finished.json", final_summary)

    print("=" * 120)
    print("[finished] clean fixed-NIR training complete")
    print(f"[finished] checkpoint : {final_path}")
    print(f"[finished] protocol   : {output_dir / 'protocol.json'}")
    print(f"[finished] train log  : {log_path}")
    print(f"[finished] progress   : {progress_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
