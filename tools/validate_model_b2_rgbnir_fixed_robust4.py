#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate M2: B2-RGBNIR-Fixed-Robust4.

Run:
    python tools/validate_model_b2_rgbnir_fixed_robust4.py

Requires:
    tools/train_model_b2_rgbnir_fixed_robust4.py
    tools/validate_model_b2_rgb.py
    tools/validate_model_d_fog.py

Default checkpoint:
    outputs/training/b2_rgbnir_fixed_robust4/checkpoints/final.pt

Validation:
    Clean
    Noise L1/L2/L3
    Blur L1/L2/L3
    Underexposure L1/L2/L3
    Fog L1/L2/L3

The metric protocol is identical to M0/M1:
512x512 windows -> full-res logits -> overlap mean-logit fusion
-> one 6000x6000 prediction/tile -> GLOBAL 6-tile confusion matrix.

M2-specific diagnostics:
    mean absolute NIR residual response at 4 scales.

Outputs:
    outputs/evaluation/b2_rgbnir_fixed_robust4/
        all_conditions_summary.json/csv
        comparison_vs_m1.json/csv
        residual_statistics.json/csv
        clean_val/
        robustness_val_v2/
        fog_seen_val/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgb as base
from models.segformer_rgb import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from train_model_b2_rgbnir_fixed_robust4 import (
    FIXED_NIR_STRENGTH,
    MODEL_ID,
    MODEL_NAME,
    NUM_SCALES,
    PROTOCOL_VERSION,
    build_model_b2_rgbnir_fixed,
)

REGIME = "robust4"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "training" /
    "b2_rgbnir_fixed_robust4" / "checkpoints" / "final.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "evaluation" /
    "b2_rgbnir_fixed_robust4"
)
DEFAULT_M1_SUMMARY = (
    PROJECT_ROOT / "outputs" / "evaluation" /
    "b2_rgb_robust4" / "all_conditions_summary.json"
)


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def training_relation(*, regime: str, family: str) -> str:
    if family == "clean":
        return "clean_reference"
    if family in {
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
        "fog",
    }:
        return "seen_degradation_family"
    return "unknown_relation"


def condition_dir(root: Path, item: Mapping[str, Any]) -> Path:
    if item["kind"] == "clean":
        return root / "clean_val"
    if item["kind"] == "fog":
        return root / "fog_seen_val" / str(item["condition"])
    return root / "robustness_val_v2" / str(item["condition"])


def validate_checkpoint(meta: Mapping[str, Any]) -> None:
    if meta.get("model_id") != MODEL_ID:
        raise RuntimeError(f"Wrong model_id: {meta.get('model_id')!r}")
    if meta.get("model_name") != MODEL_NAME:
        raise RuntimeError(f"Wrong model_name: {meta.get('model_name')!r}")
    if meta.get("regime") != REGIME:
        raise RuntimeError(f"Wrong regime: {meta.get('regime')!r}")

    p = meta.get("protocol")
    if not isinstance(p, Mapping):
        raise RuntimeError("Checkpoint protocol metadata missing.")
    if p.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("M2 protocol_version mismatch.")
    if list(p.get("input_modalities", [])) != ["RGB", "NIR"]:
        raise RuntimeError("Checkpoint is not RGB+NIR.")
    if bool(p.get("quality_gate", True)):
        raise RuntimeError("M2 must not use a quality gate.")
    if bool(p.get("gate_supervision", True)):
        raise RuntimeError("M2 must not use gate supervision.")

    g = float(p.get("fixed_nir_strength", float("nan")))
    if not math.isfinite(g) or abs(g - FIXED_NIR_STRENGTH) > 1e-12:
        raise RuntimeError(f"Formal M2 requires fixed g={FIXED_NIR_STRENGTH}.")

    c = p.get("corruption_training")
    if not isinstance(c, Mapping):
        raise RuntimeError("corruption_training metadata missing.")
    expected = {
        "gaussian_noise", "gaussian_blur",
        "rgb_underexposure", "fog",
    }
    actual = {str(x) for x in c.get("families", [])}
    if actual != expected or not bool(c.get("fog_in_training", False)):
        raise RuntimeError("Checkpoint is not the intended Robust-4 training run.")


def validate_batch(batch: Mapping[str, Any], tile_id: str) -> None:
    for key in ("rgb", "nir", "tile_id", "x", "y"):
        if key not in batch:
            raise RuntimeError(f"Missing batch key: {key}")

    if any(str(x) != tile_id for x in list(batch["tile_id"])):
        raise RuntimeError("Per-tile DataLoader mixed tile ids.")

    rgb, nir = batch["rgb"], batch["nir"]
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise RuntimeError(f"Bad RGB shape: {tuple(rgb.shape)}")
    if nir.ndim != 4 or nir.shape[1] != 1:
        raise RuntimeError(f"Bad NIR shape: {tuple(nir.shape)}")
    if rgb.shape[0] != nir.shape[0] or rgb.shape[-2:] != nir.shape[-2:]:
        raise RuntimeError("RGB/NIR are not aligned.")


def infer_one_tile_m2(
    *,
    model,
    dataset,
    tile_id,
    tile_index,
    batch_size,
    num_workers,
    pin_memory,
    device,
    amp_enabled,
    log_every,
):
    windows_per_tile = len(dataset.window_coordinates)
    start = tile_index * windows_per_tile
    stop = start + windows_per_tile

    loader = DataLoader(
        Subset(dataset, range(start, stop)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
    )

    tile_size = int(dataset.spec.tile_size)
    crop_size = int(dataset.spec.crop_size)

    logits_sum = torch.zeros(
        (NUM_CLASSES, tile_size, tile_size),
        dtype=torch.float32,
        device=device,
    )
    coverage = torch.zeros(
        (tile_size, tile_size),
        dtype=torch.float32,
        device=device,
    )

    residual_sum = torch.zeros(NUM_SCALES, dtype=torch.float64)
    residual_weight = 0
    windows_seen = 0
    started = time.time()

    for batch_idx, batch in enumerate(loader):
        validate_batch(batch, tile_id)

        rgb = batch["rgb"].to(
            device,
            non_blocking=(pin_memory and device.type == "cuda"),
        )
        nir = batch["nir"].to(
            device,
            non_blocking=(pin_memory and device.type == "cuda"),
        )

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            details = model(rgb, nir, return_details=True)

        logits = details["logits"].float()
        residual = details["residual_abs_mean"].detach().float().cpu().double()

        if tuple(logits.shape[-2:]) != (crop_size, crop_size):
            raise RuntimeError(f"Bad logits shape: {tuple(logits.shape)}")
        if logits.shape[1] != NUM_CLASSES:
            raise RuntimeError("Wrong number of output classes.")
        if residual.numel() != NUM_SCALES:
            raise RuntimeError("Residual diagnostic must have 4 scales.")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite logits.")

        n = int(logits.shape[0])
        residual_sum += residual * n
        residual_weight += n

        for j in range(n):
            x = int(batch["x"][j])
            y = int(batch["y"][j])
            logits_sum[:, y:y + crop_size, x:x + crop_size].add_(logits[j])
            coverage[y:y + crop_size, x:x + crop_size].add_(1.0)

        windows_seen += n

        if batch_idx % log_every == 0 or batch_idx + 1 == len(loader):
            r = residual_sum / max(residual_weight, 1)
            print(
                f"  tile {tile_id} | batch {batch_idx + 1:03d}/{len(loader):03d} | "
                f"windows {windows_seen:03d}/{windows_per_tile:03d} | "
                f"residual={[round(float(x), 5) for x in r.tolist()]}",
                flush=True,
            )

        del details, logits, residual, rgb, nir

    if windows_seen != windows_per_tile:
        raise RuntimeError("Window count mismatch.")
    if float(coverage.min().item()) <= 0:
        raise RuntimeError("Uncovered tile pixels found.")

    coverage_min = float(coverage.min().item())
    coverage_max = float(coverage.max().item())

    logits_sum.div_(coverage.unsqueeze(0))
    prediction = (
        logits_sum.argmax(dim=0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    residual_mean = residual_sum / residual_weight
    elapsed = time.time() - started

    del logits_sum, coverage
    if device.type == "cuda":
        torch.cuda.empty_cache()

    info = {
        "tile_id": tile_id,
        "windows": windows_seen,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "inference_seconds": elapsed,
        "fixed_nir_strength": FIXED_NIR_STRENGTH,
    }
    for s in range(NUM_SCALES):
        info[f"nir_residual_abs_mean_scale{s + 1}"] = float(residual_mean[s])

    return prediction, info


# Reuse the established evaluator, but swap in dual-modal inference.
base.infer_one_tile = infer_one_tile_m2
base.training_relation = training_relation


def add_semantics(result: Dict[str, Any]) -> None:
    result["regime"] = REGIME
    result["input_modalities"] = ["RGB", "NIR"]
    result["nir_used"] = True
    result["fusion"] = "RGB-anchored fixed-strength NIR residual fusion"
    result["fixed_nir_strength"] = FIXED_NIR_STRENGTH
    result["quality_gate"] = False
    result["gate_supervision"] = False
    result["training_relation"] = training_relation(
        regime=REGIME,
        family=str(result["family"]),
    )
    result["fog_seen_during_training"] = result["family"] == "fog"


def residual_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "condition": result["condition"],
        "family": result["family"],
        "severity_level": result.get("severity_level"),
        "fixed_nir_strength": FIXED_NIR_STRENGTH,
    }
    for s in range(1, NUM_SCALES + 1):
        key = f"nir_residual_abs_mean_scale{s}"
        values = np.asarray(
            [float(tile[key]) for tile in result["tiles"]],
            dtype=np.float64,
        )
        row[f"scale{s}_mean"] = float(values.mean())
        row[f"scale{s}_std"] = float(values.std())
        row[f"scale{s}_effective"] = float(
            FIXED_NIR_STRENGTH * values.mean()
        )
    return row


def summary_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "condition": result["condition"],
        "family": result["family"],
        "severity_level": result.get("severity_level"),
        "training_relation": result["training_relation"],
        "clean_miou": result["clean_reference_miou"],
        "miou": result["miou"],
        "drop_miou": result["drop_miou"],
        "relative_drop_pct": result["relative_drop_pct"],
        "retention_pct": result["retention_pct"],
        "pixel_accuracy": result["pixel_accuracy"],
        "mean_class_accuracy": result["mean_class_accuracy"],
        "validation_seconds": result["validation_seconds"],
    }


def compare_vs_m1(
    *,
    results,
    m1_summary_path: Path,
    output_root: Path,
) -> None:
    if not m1_summary_path.is_file():
        print(f"[comparison] M1 summary not found: {m1_summary_path}")
        return

    m1 = json.loads(m1_summary_path.read_text(encoding="utf-8"))
    lookup = {
        str(x["condition"]): x
        for x in m1.get("results", [])
    }

    rows = []
    for r in results:
        cond = str(r["condition"])
        if cond not in lookup:
            continue
        old = float(lookup[cond]["miou"])
        new = float(r["miou"])
        rows.append(
            {
                "condition": cond,
                "family": r["family"],
                "severity_level": r.get("severity_level"),
                "m1_rgb_robust4_miou": old,
                "m2_rgbnir_fixed_robust4_miou": new,
                "absolute_gain_miou": new - old,
                "relative_gain_pct": (
                    100.0 * (new - old) / old if old != 0 else None
                ),
            }
        )

    save_json(
        output_root / "comparison_vs_m1.json",
        {
            "scientific_question": (
                "Contribution of adding fixed NIR residual information "
                "while Robust-4 training is held constant."
            ),
            "m1_source": str(m1_summary_path),
            "rows": rows,
        },
    )
    write_csv(
        output_root / "comparison_vs_m1.csv",
        rows,
        [
            "condition",
            "family",
            "severity_level",
            "m1_rgb_robust4_miou",
            "m2_rgbnir_fixed_robust4_miou",
            "absolute_gain_miou",
            "relative_gain_pct",
        ],
    )

    degraded = [x["absolute_gain_miou"] for x in rows if x["condition"] != "Clean"]
    if degraded:
        print(
            f"[comparison] mean M2-M1 gain over 12 degraded conditions: "
            f"{float(np.mean(degraded)):+.6f} mIoU",
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate M2 B2-RGBNIR-Fixed-Robust4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--m1-summary", type=Path, default=DEFAULT_M1_SUMMARY)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=16)
    p.add_argument("--confusion-chunk-rows", type=int, default=512)
    p.add_argument("--fog-chunk-rows", type=int, default=128)
    p.add_argument("--save-predictions", action="store_true")
    args = p.parse_args()
    if args.batch_size <= 0:
        p.error("--batch-size must be > 0")
    return args


def main():
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    output_root = resolve(args.output_root)
    m1_summary_path = resolve(args.m1_summary)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_root.mkdir(parents=True, exist_ok=True)

    device = base.get_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, meta = base.unwrap_checkpoint_state_dict(ckpt)
    validate_checkpoint(meta)

    epoch = meta.get("epoch")
    step = meta.get("global_step", meta.get("step"))
    epoch = int(epoch) if epoch is not None else None
    step = int(step) if step is not None else None

    print("=" * 120)
    print(f"{MODEL_NAME} | 13-CONDITION VALIDATION")
    print("=" * 120)
    print(f"checkpoint   : {checkpoint_path}")
    print(f"epoch        : {epoch + 1 if epoch is not None else 'unknown'}")
    print(f"global_step  : {step}")
    print(f"device / AMP : {device} / {amp_enabled}")
    print(f"fixed g_NIR  : {FIXED_NIR_STRENGTH}")
    print(f"output       : {output_root}")
    print("=" * 120)

    model, model_meta = build_model_b2_rgbnir_fixed(
        PROJECT_ROOT,
        fixed_strength=FIXED_NIR_STRENGTH,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(
        f"[model] strict load PASS | "
        f"parameters={model_meta['parameters']['total']:,}"
    )

    # Save exact protocol snapshots.
    std_root = output_root / "robustness_val_v2"
    std_root.mkdir(parents=True, exist_ok=True)
    base.write_degradation_protocol(std_root / "degradation_protocol.json")

    fog_dir = output_root / "fog_seen_val"
    fog_dir.mkdir(parents=True, exist_ok=True)
    fog_protocol = dict(base.FOG_PROTOCOL)
    fog_protocol["sha256"] = base.fog_protocol_sha256()
    fog_protocol["evaluation_role_for_this_model"] = (
        "seen degradation family; RGB degraded, NIR clean"
    )
    save_json(fog_dir / "fog_protocol.json", fog_protocol)

    conditions = base.suite_conditions("all")
    clean_probe = base.PotsdamSlidingWindowDataset(PROJECT_ROOT, split="val")

    sample = clean_probe[0]
    if "nir" not in sample or sample["nir"].shape[0] != 1:
        raise RuntimeError("Frozen validation dataset does not provide 1-channel NIR.")

    progress = base.ValidationProgress(
        total_conditions=len(conditions),
        tiles_per_condition=6,
        output_path=output_root / "validation_progress.json",
    )

    results = []
    clean_result = None

    for condition_index, item in enumerate(conditions, start=1):
        condition = str(item["condition"])
        out_dir = condition_dir(output_root, item)

        print()
        print("-" * 120)
        print(
            f"[Condition {condition_index:02d}/{len(conditions):02d}] "
            f"{condition} | {training_relation(regime=REGIME, family=str(item['family']))}"
        )
        print("-" * 120)

        dataset = base.build_dataset(
            item=item,
            fog_chunk_rows=args.fog_chunk_rows,
        )

        if item["kind"] != "clean":
            base.degradation_probe(
                clean_dataset=clean_probe,
                degraded_dataset=dataset,
                condition=condition,
            )
            # M2 protocol: RGB degraded, NIR unchanged.
            if not torch.equal(clean_probe[0]["nir"], dataset[0]["nir"]):
                raise RuntimeError(f"{condition}: NIR changed unexpectedly.")

        result = base.evaluate_condition(
            model=model,
            dataset=dataset,
            item=item,
            output_dir=out_dir,
            model_id=MODEL_ID,
            model_name=MODEL_NAME,
            regime=REGIME,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=epoch,
            checkpoint_global_step=step,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            device=device,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
            confusion_chunk_rows=args.confusion_chunk_rows,
            save_predictions=args.save_predictions,
            condition_index=condition_index,
            progress=progress,
        )

        add_semantics(result)
        save_json(out_dir / "metrics.json", result)

        if condition == "Clean":
            clean_result = result

        results.append(result)
        base.clear_dataset_cache(dataset)
        del dataset

        rr = residual_row(result)
        print(
            f"[result] {condition:<24} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"residual={[round(rr[f'scale{s}_mean'], 5) for s in range(1, 5)]}",
            flush=True,
        )

    if clean_result is None:
        raise RuntimeError("Clean validation result missing.")

    clean_miou = float(clean_result["miou"])

    # Add clean-relative robustness metrics.
    for result in results:
        if result["condition"] == "Clean":
            result["clean_reference_miou"] = clean_miou
            result["drop_miou"] = 0.0
            result["delta_miou"] = 0.0
            result["relative_drop_pct"] = 0.0
            result["retention_pct"] = 100.0
        else:
            base.add_clean_relative_metrics(
                result,
                clean_miou=clean_miou,
            )
        add_semantics(result)

        item = {
            "kind": (
                "clean" if result["condition"] == "Clean"
                else "fog" if result["family"] == "fog"
                else "standard"
            ),
            "condition": result["condition"],
        }
        save_json(condition_dir(output_root, item) / "metrics.json", result)

    # Main summary.
    rows = [summary_row(r) for r in results]
    save_json(
        output_root / "all_conditions_summary.json",
        {
            "model": MODEL_ID,
            "model_name": MODEL_NAME,
            "regime": REGIME,
            "input_modalities": ["RGB", "NIR"],
            "fusion": "fixed NIR residual fusion",
            "fixed_nir_strength": FIXED_NIR_STRENGTH,
            "quality_gate": False,
            "checkpoint": str(checkpoint_path),
            "checkpoint_global_step": step,
            "results": rows,
        },
    )
    write_csv(
        output_root / "all_conditions_summary.csv",
        rows,
        [
            "condition",
            "family",
            "severity_level",
            "training_relation",
            "clean_miou",
            "miou",
            "drop_miou",
            "relative_drop_pct",
            "retention_pct",
            "pixel_accuracy",
            "mean_class_accuracy",
            "validation_seconds",
        ],
    )

    # Residual statistics.
    residual_rows = [residual_row(r) for r in results]
    save_json(
        output_root / "residual_statistics.json",
        {
            "model": MODEL_ID,
            "fixed_nir_strength": FIXED_NIR_STRENGTH,
            "definition": "mean absolute NIR adapter output per scale",
            "rows": residual_rows,
        },
    )
    residual_fields = [
        "condition",
        "family",
        "severity_level",
        "fixed_nir_strength",
    ]
    for s in range(1, 5):
        residual_fields.extend(
            [f"scale{s}_mean", f"scale{s}_std", f"scale{s}_effective"]
        )
    write_csv(
        output_root / "residual_statistics.csv",
        residual_rows,
        residual_fields,
    )

    compare_vs_m1(
        results=results,
        m1_summary_path=m1_summary_path,
        output_root=output_root,
    )

    progress.finish(output_root=output_root)

    print()
    print("=" * 132)
    print("FINAL M2 VALIDATION SUMMARY")
    print("=" * 132)
    print(
        f"{'Condition':<26} {'mIoU':>10} {'Drop':>10} "
        f"{'RelDrop%':>10} {'Retention%':>12}"
    )
    print("-" * 132)
    for r in results:
        print(
            f"{str(r['condition']):<26} "
            f"{float(r['miou']):>10.6f} "
            f"{float(r['drop_miou']):>10.6f} "
            f"{float(r['relative_drop_pct']):>10.3f} "
            f"{float(r['retention_pct']):>12.3f}"
        )
    print("-" * 132)
    print(f"Clean mIoU     : {clean_miou:.6f}")
    print(f"M2 vs M1      : {output_root / 'comparison_vs_m1.csv'}")
    print(f"Residual stats: {output_root / 'residual_statistics.csv'}")
    print(f"Summary       : {output_root / 'all_conditions_summary.csv'}")
    print("=" * 132)


if __name__ == "__main__":
    main()
