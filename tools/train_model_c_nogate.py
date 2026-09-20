#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal training for Model C-noGate:
Dual SegFormer-B0 encoders + fixed 0.5/0.5 multi-scale fusion.

Controlled against Model A / Model B
------------------------------------
Training protocol remains:
- Clean training only.
- 100 epochs.
- mini-batch size = 2.
- gradient accumulation = 8.
- nominal effective batch size = 16.
- AdamW, lr=6e-5, weight_decay=0.01.
- 360 successful-update warmup.
- linear polynomial decay.
- FP16 AMP on CUDA.
- gradient clipping = 1.0.
- same seed and deterministic Potsdam sampler/crops/D4.
- same external CE loss.
- same mmap data cache already used by A/B.

Architecture-specific variable
------------------------------
Model C-noGate uses:
- pretrained RGB SegFormer encoder;
- pretrained NIR SegFormer encoder;
- fixed 0.5/0.5 fusion at all 4 encoder scales;
- one shared pretrained SegFormer decoder;
- NO Quality Gate.

The future Model C should use the same architecture and replace only fixed
0.5/0.5 weights with dynamic Quality Gate weights.

Expected location:
    tools/train_model_c_nogate.py

Required:
    models/segformer_dual_fixed.py
    tools/train_model_a_rgb.py

Run:
    python models/segformer_dual_fixed.py
    python tools/train_model_c_nogate.py

Outputs:
    outputs/training/model_c_nogate/
        protocol.json
        train_log.jsonl
        checkpoints/latest.pt
        checkpoints/final.pt
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


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)
TOOLS_DIR = (
    PROJECT_ROOT
    / "tools"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(TOOLS_DIR),
    )


from data_pipeline.potsdam_dataloader import (
    DATA_SEED,
    DeterministicEpochSampler,
    build_train_dataloader,
    set_train_epoch,
)
from models.segformer_dual_fixed import (
    FIXED_NIR_WEIGHT,
    FIXED_RGB_WEIGHT,
    IGNORE_INDEX,
    MODEL_ID,
    MODEL_NAME,
    NIR_INIT_METHOD,
    NUM_CLASSES,
    NUM_SCALES,
    build_model_c_nogate,
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
        "Could not import the successful tools/train_model_a_rgb.py. "
        "Keep that file in tools/ before training Model C-noGate."
    ) from exc


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_c_nogate"
)

DEFAULT_SEED = 20260917
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 8
DEFAULT_LR = 6e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 360
DEFAULT_GRAD_CLIP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Model C-noGate: dual encoder + fixed multi-scale fusion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=DEFAULT_GRAD_ACCUM_STEPS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
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
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            'Checkpoint path, or "auto" for '
            '<output-dir>/checkpoints/latest.pt.'
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
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.grad_clip <= 0:
        parser.error("--grad-clip must be > 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.save_every < 0:
        parser.error("--save-every must be >= 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    return args


def resolve_path(
    path: Path,
) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def get_device(
    device_arg: str,
) -> torch.device:
    device = torch.device(
        device_arg
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    if (
        device.type == "cuda"
        and device.index is not None
    ):
        torch.cuda.set_device(
            device.index
        )

    return device


def resolve_resume_path(
    resume_arg: str,
    checkpoint_dir: Path,
) -> Optional[Path]:
    if not resume_arg:
        return None

    if resume_arg.lower() == "auto":
        path = (
            checkpoint_dir
            / "latest.pt"
        )
    else:
        path = (
            Path(resume_arg)
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}"
        )

    return path


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def build_protocol(
    *,
    args: argparse.Namespace,
    model_meta: Mapping[str, Any],
    updates_per_epoch: int,
    total_update_steps: int,
    effective_warmup_steps: int,
    device: torch.device,
    amp_enabled: bool,
    data_cache_dir: Path,
) -> Dict[str, Any]:
    return {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "variant": "C-noGate",
        "scientific_role": (
            "dual-encoder fixed-fusion control for Quality Gate ablation"
        ),
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "rgb_channels": 3,
        "nir_channels": 1,
        "dual_encoder": True,
        "multiscale_fusion": True,
        "fusion_scales": NUM_SCALES,
        "fusion_rule": (
            "F_i = 0.5*F_RGB_i + 0.5*F_NIR_i"
        ),
        "fixed_rgb_weight": FIXED_RGB_WEIGHT,
        "fixed_nir_weight": FIXED_NIR_WEIGHT,
        "fusion_trainable_parameters": 0,
        "quality_gate": False,
        "shared_decoder": True,
        "nir_encoder_initialization": NIR_INIT_METHOD,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "seed": args.seed,
        "data_seed": int(
            DATA_SEED
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": (
            args.grad_accum_steps
        ),
        "effective_batch_size_nominal": (
            args.batch_size
            * args.grad_accum_steps
        ),
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": (
            args.weight_decay
        ),
        "warmup_steps_requested": (
            args.warmup_steps
        ),
        "warmup_steps_effective": (
            effective_warmup_steps
        ),
        "scheduler": (
            "linear_warmup_then_poly_decay"
        ),
        "scheduler_power": 1.0,
        "grad_clip": args.grad_clip,
        "amp_requested": (
            not args.no_amp
        ),
        "amp_enabled": (
            amp_enabled
        ),
        "device": str(
            device
        ),
        "num_workers": (
            args.num_workers
        ),
        "pin_memory": (
            args.pin_memory
        ),
        "data_cache_mode": (
            CachedPotsdamTrainDataset
            .CACHE_VERSION
        ),
        "data_cache_dir": str(
            data_cache_dir
        ),
        "updates_per_epoch": (
            updates_per_epoch
        ),
        "total_update_steps": (
            total_update_steps
        ),
        "torch_version": (
            torch.__version__
        ),
        "numpy_version": (
            np.__version__
        ),
        "model_meta": dict(
            model_meta
        ),
        "controlled_against_model_a_b": {
            "same_clean_training_distribution": True,
            "same_seed": True,
            "same_train_dataset": True,
            "same_sampler": True,
            "same_crop_and_d4": True,
            "same_optimizer": True,
            "same_lr": True,
            "same_weight_decay": True,
            "same_epochs": True,
            "same_batch_size": True,
            "same_grad_accum_steps": True,
            "same_warmup": True,
            "same_scheduler": True,
            "same_amp_policy": True,
            "same_grad_clip": True,
        },
        "future_quality_gate_control": {
            "require_same_rgb_encoder": True,
            "require_same_nir_encoder": True,
            "require_same_encoder_initialization": True,
            "require_same_decoder": True,
            "require_same_fusion_scales": True,
            "only_intended_change": (
                "replace fixed [0.5,0.5] per scale with dynamic "
                "Quality Gate weights summing to 1"
            ),
        },
    }


def make_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    model_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
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


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    print(
        f"[resume] loading checkpoint: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get(
        "model_id"
    ) not in (
        None,
        MODEL_ID,
    ):
        raise RuntimeError(
            "Resume checkpoint belongs to another model: "
            f"{checkpoint.get('model_id')!r}"
        )

    protocol = checkpoint.get(
        "protocol"
    )

    if isinstance(
        protocol,
        Mapping,
    ):
        if protocol.get(
            "model"
        ) != MODEL_ID:
            raise RuntimeError(
                "Resume protocol is not C-noGate."
            )

        if bool(
            protocol.get(
                "quality_gate",
                True,
            )
        ):
            raise RuntimeError(
                "Resume checkpoint unexpectedly has a Quality Gate."
            )

        if not bool(
            protocol.get(
                "dual_encoder",
                False,
            )
        ):
            raise RuntimeError(
                "Resume checkpoint is not dual-encoder."
            )

        if float(
            protocol.get(
                "fixed_rgb_weight",
                -1,
            )
        ) != 0.5:
            raise RuntimeError(
                "Resume RGB fixed fusion weight is not 0.5."
            )

        if float(
            protocol.get(
                "fixed_nir_weight",
                -1,
            )
        ) != 0.5:
            raise RuntimeError(
                "Resume NIR fixed fusion weight is not 0.5."
            )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )
    scaler.load_state_dict(
        checkpoint["scaler"]
    )

    restore_rng_state(
        checkpoint.get(
            "rng_state"
        )
    )

    start_epoch = (
        int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )
        + 1
    )

    global_step = int(
        checkpoint.get(
            "global_step",
            checkpoint.get(
                "step",
                0,
            ),
        )
    )

    print(
        f"[resume] restored epoch={start_epoch}, "
        f"global_step={global_step}"
    )

    return (
        start_epoch,
        global_step,
    )


def validate_first_batch(
    *,
    rgb: torch.Tensor,
    nir: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if (
        rgb.ndim != 4
        or rgb.shape[1] != 3
    ):
        raise RuntimeError(
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}."
        )

    if (
        nir.ndim != 4
        or nir.shape[1] != 1
    ):
        raise RuntimeError(
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}."
        )

    if labels.ndim != 3:
        raise RuntimeError(
            f"Labels must be [B,H,W], got {tuple(labels.shape)}."
        )

    if (
        rgb.shape[0]
        != nir.shape[0]
        or rgb.shape[0]
        != labels.shape[0]
        or rgb.shape[-2:]
        != nir.shape[-2:]
        or rgb.shape[-2:]
        != labels.shape[-2:]
    ):
        raise RuntimeError(
            "RGB/NIR/labels are not aligned."
        )

    if (
        rgb.dtype != torch.float32
        or nir.dtype != torch.float32
        or labels.dtype != torch.int64
    ):
        raise RuntimeError(
            "Expected dataset dtypes RGB=float32, NIR=float32, labels=int64; "
            f"got {rgb.dtype}, {nir.dtype}, {labels.dtype}."
        )

    valid = (
        labels
        != IGNORE_INDEX
    )

    if not torch.any(
        valid
    ):
        raise RuntimeError(
            "First batch contains no valid semantic labels."
        )

    values = labels[
        valid
    ]

    label_min = int(
        values.min().item()
    )
    label_max = int(
        values.max().item()
    )

    if (
        label_min < 0
        or label_max
        >= NUM_CLASSES
    ):
        raise RuntimeError(
            f"Unexpected label range [{label_min},{label_max}]."
        )

    print(
        "[check] first batch OK | "
        f"rgb={tuple(rgb.shape)} {rgb.dtype} | "
        f"nir={tuple(nir.shape)} {nir.dtype} | "
        f"labels={tuple(labels.shape)} {labels.dtype} | "
        f"label_range=[{label_min},{label_max}]"
    )


def validate_model_output(
    *,
    details: Mapping[str, Any],
    labels: torch.Tensor,
) -> None:
    logits = details[
        "logits"
    ]
    weights = details[
        "fusion_weights"
    ]

    if (
        logits.ndim != 4
        or logits.shape[1]
        != NUM_CLASSES
        or logits.shape[0]
        != labels.shape[0]
        or logits.shape[-2:]
        != labels.shape[-2:]
    ):
        raise RuntimeError(
            f"Model C-noGate logits invalid: {tuple(logits.shape)}."
        )

    if tuple(
        weights.shape
    ) != (
        labels.shape[0],
        NUM_SCALES,
        2,
    ):
        raise RuntimeError(
            f"Fixed fusion weights invalid: {tuple(weights.shape)}."
        )

    if not torch.equal(
        weights,
        torch.full_like(
            weights,
            0.5,
        ),
    ):
        raise RuntimeError(
            "C-noGate returned fusion weights other than exact 0.5/0.5."
        )

    print(
        "[check] model output OK | "
        f"logits={tuple(logits.shape)} {logits.dtype} | "
        f"fusion_weights={weights[0].detach().cpu().tolist()}"
    )


def main() -> None:
    args = parse_args()
    seed_everything(
        args.seed
    )

    output_dir = resolve_path(
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

    data_cache_dir = resolve_path(
        args.data_cache_dir
    )

    print("=" * 92)
    print(
        MODEL_NAME
    )
    print("=" * 92)
    print(
        f"project root : {PROJECT_ROOT}"
    )
    print(
        f"output dir   : {output_dir}"
    )
    print(
        f"data cache   : {data_cache_dir}"
    )

    print(
        "[1] loading training dataset"
    )

    dataset = CachedPotsdamTrainDataset(
        PROJECT_ROOT,
        epoch=0,
        cache_root=data_cache_dir,
    )

    dataset.prepare_cache(
        rebuild=args.rebuild_data_cache
    )

    if len(
        dataset
    ) <= 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    print(
        f"dataset length: {len(dataset)}"
    )

    sampler = (
        DeterministicEpochSampler(
            dataset,
            seed=DATA_SEED,
            epoch=0,
        )
    )

    print(
        "[2] building dataloader"
    )

    loader = (
        build_train_dataloader(
            dataset,
            sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    )

    if len(
        loader
    ) <= 0:
        raise RuntimeError(
            "Training dataloader contains zero batches."
        )

    print(
        f"batches per epoch: {len(loader)}"
    )

    updates_per_epoch = (
        math.ceil(
            len(loader)
            / args.grad_accum_steps
        )
    )
    total_update_steps = (
        updates_per_epoch
        * args.epochs
    )
    effective_warmup_steps = min(
        args.warmup_steps,
        max(
            0,
            total_update_steps
            - 1,
        ),
    )

    print(
        "[3] building Model C-noGate"
    )

    device = get_device(
        args.device
    )

    model, model_meta = (
        build_model_c_nogate(
            PROJECT_ROOT
        )
    )
    model.to(
        device
    )

    print(
        "  fusion rule             : "
        f"{model_meta['fusion_rule']}"
    )
    print(
        "  fusion trainable params : "
        f"{model_meta['fusion_trainable_parameters']}"
    )
    print(
        "  quality gate            : "
        f"{model_meta['quality_gate']}"
    )
    print(
        "  NIR encoder init        : "
        f"{model_meta['nir_encoder_initialization']['method']}"
    )
    print(
        "  copy audit              : "
        f"{model_meta['encoder_copy_audit']['only_first_patch_weight_changed']}"
    )
    print(
        "  parameters              : "
        f"{model_meta['parameters']['total']:,}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = (
        build_poly_scheduler(
            optimizer=optimizer,
            warmup_steps=(
                effective_warmup_steps
            ),
            total_steps=(
                total_update_steps
            ),
            power=1.0,
        )
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    protocol = build_protocol(
        args=args,
        model_meta=model_meta,
        updates_per_epoch=updates_per_epoch,
        total_update_steps=total_update_steps,
        effective_warmup_steps=(
            effective_warmup_steps
        ),
        device=device,
        amp_enabled=amp_enabled,
        data_cache_dir=data_cache_dir,
    )

    write_json(
        output_dir
        / "protocol.json",
        protocol,
    )

    print(
        "[protocol] "
        f"epochs={args.epochs}, "
        f"batch={args.batch_size}, "
        f"accum={args.grad_accum_steps}, "
        f"effective_batch≈"
        f"{args.batch_size * args.grad_accum_steps}, "
        f"updates/epoch={updates_per_epoch}, "
        f"total_updates={total_update_steps}"
    )
    print(
        "[protocol] "
        f"lr={args.lr:g}, "
        f"warmup={effective_warmup_steps}, "
        f"weight_decay={args.weight_decay:g}, "
        f"AMP={amp_enabled}, "
        f"device={device}"
    )

    start_epoch = 0
    global_step = 0

    resume_path = (
        resolve_resume_path(
            args.resume,
            checkpoint_dir,
        )
    )

    if resume_path is not None:
        (
            start_epoch,
            global_step,
        ) = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    if start_epoch >= args.epochs:
        print(
            "[done] checkpoint already reached "
            f"epoch {start_epoch}; requested {args.epochs}."
        )
        return

    log_path = (
        output_dir
        / "train_log.jsonl"
    )

    first_batch_checked = False
    training_start = (
        time.time()
    )

    print(
        "[4] start training"
    )

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

        loss_numerator = 0.0
        valid_pixel_count = 0
        mini_batches_seen = 0
        optimizer_updates_this_epoch = 0
        amp_skipped_updates = 0
        last_grad_norm = float(
            "nan"
        )

        accumulation_target = (
            args.grad_accum_steps
        )

        for i, batch in enumerate(
            loader
        ):
            if (
                i
                % args.grad_accum_steps
                == 0
            ):
                remaining = (
                    len(loader)
                    - i
                )
                accumulation_target = min(
                    args.grad_accum_steps,
                    remaining,
                )

            rgb = (
                batch["rgb"]
                .to(
                    device,
                    non_blocking=(
                        args.pin_memory
                    ),
                )
            )
            nir = (
                batch["nir"]
                .to(
                    device,
                    non_blocking=(
                        args.pin_memory
                    ),
                )
            )
            labels = (
                batch["labels"]
                .to(
                    device,
                    non_blocking=(
                        args.pin_memory
                    ),
                )
                .long()
            )

            if not first_batch_checked:
                validate_first_batch(
                    rgb=rgb,
                    nir=nir,
                    labels=labels,
                )

            with torch.autocast(
                device_type=(
                    device.type
                ),
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if not first_batch_checked:
                    details = model(
                        rgb,
                        nir,
                        return_details=True,
                    )

                    validate_model_output(
                        details=details,
                        labels=labels,
                    )

                    logits = details[
                        "logits"
                    ]
                    # Do not keep explicit references to the four RGB/NIR/fused
                    # feature pyramids beyond this one audit forward.
                    del details
                    first_batch_checked = (
                        True
                    )
                else:
                    logits = model(
                        rgb,
                        nir,
                    )

                raw_loss = (
                    F.cross_entropy(
                        logits,
                        labels,
                        ignore_index=(
                            IGNORE_INDEX
                        ),
                    )
                )

                loss_for_backward = (
                    raw_loss
                    / accumulation_target
                )

            if not torch.isfinite(
                raw_loss
            ).item():
                raise FloatingPointError(
                    "Non-finite loss at "
                    f"epoch={epoch + 1}, "
                    f"batch={i}: "
                    f"{raw_loss.detach().item()}."
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            batch_valid_pixels = int(
                (
                    labels
                    != IGNORE_INDEX
                )
                .sum()
                .item()
            )

            if batch_valid_pixels > 0:
                loss_numerator += (
                    float(
                        raw_loss
                        .detach()
                        .item()
                    )
                    * batch_valid_pixels
                )
                valid_pixel_count += (
                    batch_valid_pixels
                )

            mini_batches_seen += 1

            should_step = (
                (
                    (i + 1)
                    % args.grad_accum_steps
                    == 0
                )
                or (
                    (i + 1)
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

                scale_before = (
                    scaler
                    .get_scale()
                )

                scaler.step(
                    optimizer
                )
                scaler.update()

                scale_after = (
                    scaler
                    .get_scale()
                )

                optimizer_step_happened = (
                    (not amp_enabled)
                    or (
                        scale_after
                        >= scale_before
                    )
                )

                if optimizer_step_happened:
                    scheduler.step()
                    global_step += 1
                    optimizer_updates_this_epoch += 1
                else:
                    amp_skipped_updates += 1

                    print(
                        "[amp] skipped optimizer update due to overflow | "
                        f"epoch={epoch + 1}, batch={i}"
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

            if (
                i
                % args.log_every
                == 0
                or (
                    (i + 1)
                    == len(loader)
                )
            ):
                running_loss = (
                    loss_numerator
                    / valid_pixel_count
                    if valid_pixel_count > 0
                    else float("nan")
                )

                current_lr = float(
                    optimizer
                    .param_groups[0][
                        "lr"
                    ]
                )

                print(
                    f"epoch "
                    f"{epoch + 1:03d}/"
                    f"{args.epochs:03d} | "
                    f"batch "
                    f"{i + 1:05d}/"
                    f"{len(loader):05d} | "
                    f"raw_ce "
                    f"{raw_loss.detach().item():.6f} | "
                    f"running_ce "
                    f"{running_loss:.6f} | "
                    f"lr "
                    f"{current_lr:.3e} | "
                    f"step "
                    f"{global_step}/"
                    f"{total_update_steps}"
                )

        if valid_pixel_count <= 0:
            raise RuntimeError(
                f"Epoch {epoch + 1} contains no valid pixels."
            )

        epoch_loss = (
            loss_numerator
            / valid_pixel_count
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        current_lr = float(
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        epoch_record = {
            "epoch": (
                epoch + 1
            ),
            "global_step": (
                global_step
            ),
            "optimizer_updates_this_epoch": (
                optimizer_updates_this_epoch
            ),
            "amp_skipped_updates": (
                amp_skipped_updates
            ),
            "mini_batches": (
                mini_batches_seen
            ),
            "valid_pixels": (
                valid_pixel_count
            ),
            "train_ce": (
                epoch_loss
            ),
            "lr": (
                current_lr
            ),
            "last_grad_norm_before_clip": (
                last_grad_norm
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
            "fixed_rgb_weight": (
                FIXED_RGB_WEIGHT
            ),
            "fixed_nir_weight": (
                FIXED_NIR_WEIGHT
            ),
        }

        append_jsonl(
            log_path,
            epoch_record,
        )

        payload = (
            make_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                protocol=protocol,
                model_meta=model_meta,
            )
        )

        latest_path = (
            checkpoint_dir
            / "latest.pt"
        )

        atomic_torch_save(
            payload,
            latest_path,
        )

        if (
            args.save_every > 0
            and (
                (epoch + 1)
                % args.save_every
                == 0
            )
        ):
            archive_path = (
                checkpoint_dir
                / f"epoch_{epoch + 1:03d}.pt"
            )

            atomic_torch_save(
                payload,
                archive_path,
            )

        print(
            f"[epoch done] "
            f"{epoch + 1:03d}/"
            f"{args.epochs:03d} | "
            f"train_ce="
            f"{epoch_loss:.6f} | "
            f"updates="
            f"{optimizer_updates_this_epoch} | "
            f"amp_skips="
            f"{amp_skipped_updates} | "
            f"global_step="
            f"{global_step} | "
            f"lr="
            f"{current_lr:.3e} | "
            f"time="
            f"{epoch_seconds:.1f}s"
        )

    final_payload = (
        make_checkpoint_payload(
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
    )

    final_path = (
        checkpoint_dir
        / "final.pt"
    )

    atomic_torch_save(
        final_payload,
        final_path,
    )

    total_seconds = (
        time.time()
        - training_start
    )

    print("=" * 92)
    print(
        f"[finished] {MODEL_NAME} | "
        f"epochs={args.epochs} | "
        f"global_step={global_step} | "
        f"time="
        f"{total_seconds / 3600.0:.2f}h"
    )
    print(
        "[finished] latest checkpoint: "
        f"{checkpoint_dir / 'latest.pt'}"
    )
    print(
        "[finished] final checkpoint : "
        f"{final_path}"
    )
    print(
        "[finished] training log     : "
        f"{log_path}"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
