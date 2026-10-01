#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate M1: SegFormer-B2 RGB-only Robust-4 baseline.

Default:
    python tools/validate_model_b2_rgb_robust4.py

Checkpoint:
    outputs/training/b2_rgb_robust4/checkpoints/final.pt

Evaluation suite:
    Clean
    Gaussian Noise L1/L2/L3
    Gaussian Blur L1/L2/L3
    RGB Underexposure L1/L2/L3
    Fog L1/L2/L3

Total:
    13 conditions x 6 Potsdam validation tiles

Fairness:
- Same 512x512 sliding-window protocol as M0 / Model D.
- Same overlap mean-logit fusion.
- Same GLOBAL six-tile confusion-matrix metric.
- Same frozen standard degradation protocol.
- Same frozen deterministic full-tile Fog protocol.
- RGB-only model never consumes NIR.

Robust-4 semantics:
- Noise / Blur / Underexposure / Fog are ALL seen degradation families.
- Clean remains the reference condition.

Outputs:
outputs/evaluation/b2_rgb_robust4/
├── validation_progress.json
├── all_conditions_summary.json
├── all_conditions_summary.csv
├── comparison_vs_m0.json
├── comparison_vs_m0.csv
├── clean_val/
├── robustness_val_v2/
└── fog_seen_val/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgb as eval_base
from train_model_b2_rgb import build_b2_rgb_model
from train_model_b2_rgb_robust4 import MODEL_ID, MODEL_NAME

VALIDATOR_VERSION = "1.0.0"
REGIME = "robust4"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_rgb_robust4"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgb_robust4"
)

DEFAULT_M0_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgb_clean"
    / "all_conditions_summary.json"
)


# =============================================================================
# Helpers
# =============================================================================

def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fields),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def clean_dir(output_root: Path) -> Path:
    return output_root / "clean_val"


def robustness_root(output_root: Path) -> Path:
    return output_root / "robustness_val_v2"


def fog_root(output_root: Path) -> Path:
    # Fog is now seen during Robust-4 training, so do not call it OOD.
    return output_root / "fog_seen_val"


def condition_output_dir(
    output_root: Path,
    item: Mapping[str, Any],
) -> Path:
    kind = str(item["kind"])

    if kind == "clean":
        return clean_dir(output_root)

    if kind == "standard":
        return (
            robustness_root(output_root)
            / str(item["condition"])
        )

    if kind == "fog":
        return (
            fog_root(output_root)
            / str(item["condition"])
        )

    raise ValueError(kind)


def training_relation_robust4(
    *,
    regime: str,
    family: str,
) -> str:
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


# Patch the imported evaluator at runtime so its per-condition records use the
# correct Robust-4 interpretation, including Fog as SEEN rather than OOD.
eval_base.training_relation = training_relation_robust4


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate M1 / B2-RGB-Robust4 on Clean + standard degradations + Fog."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--m0-summary",
        type=Path,
        default=DEFAULT_M0_SUMMARY,
        help=(
            "Optional completed M0/B2-RGB-Clean summary. "
            "If present, M1-vs-M0 comparison files are generated."
        ),
    )

    parser.add_argument(
        "--suite",
        choices=(
            "all",
            "clean",
            "standard",
            "fog",
        ),
        default="all",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--fog-chunk-rows",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    if args.num_workers != 0:
        parser.error(
            "--num-workers must stay 0 for the frozen full-tile degradation protocol."
        )

    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    if args.fog_chunk_rows <= 0:
        parser.error("--fog-chunk-rows must be > 0")

    return args


# =============================================================================
# Checkpoint validation
# =============================================================================

def validate_checkpoint_metadata(
    meta: Mapping[str, Any],
) -> None:
    if meta.get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"model_id mismatch: {meta.get('model_id')!r} != {MODEL_ID!r}"
        )

    if meta.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            f"model_name mismatch: {meta.get('model_name')!r} != {MODEL_NAME!r}"
        )

    if meta.get("regime") != REGIME:
        raise RuntimeError(
            f"regime mismatch: {meta.get('regime')!r} != {REGIME!r}"
        )

    protocol = meta.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("Checkpoint protocol metadata missing.")

    if protocol.get("backbone") != "SegFormer-B2":
        raise RuntimeError("Checkpoint backbone is not SegFormer-B2.")

    if list(protocol.get("input_modalities", [])) != ["RGB"]:
        raise RuntimeError("Checkpoint is not RGB-only.")

    if bool(protocol.get("nir_used", True)):
        raise RuntimeError("Checkpoint metadata unexpectedly says NIR is used.")

    if protocol.get("regime") != REGIME:
        raise RuntimeError(
            f"protocol regime mismatch: {protocol.get('regime')!r}"
        )

    corruption = protocol.get("corruption_training")
    if not isinstance(corruption, Mapping):
        raise RuntimeError("corruption_training metadata missing.")

    if not bool(corruption.get("enabled", False)):
        raise RuntimeError("Robust-4 checkpoint says corruption training disabled.")

    if not bool(corruption.get("fog_in_training", False)):
        raise RuntimeError("Robust-4 checkpoint says Fog was not used in training.")

    expected = {
        "gaussian_noise",
        "gaussian_blur",
        "rgb_underexposure",
        "fog",
    }

    actual = {
        str(x)
        for x in corruption.get("families", [])
    }

    if actual != expected:
        raise RuntimeError(
            f"Robust-4 family mismatch: {sorted(actual)} != {sorted(expected)}"
        )


# =============================================================================
# Existing result compatibility
# =============================================================================

def compatible_existing(
    path: Path,
    *,
    condition: str,
    checkpoint_path: Path,
    checkpoint_step: Optional[int],
    item: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None

    checks = [
        obj.get("model") == MODEL_ID,
        obj.get("regime") == REGIME,
        obj.get("condition") == condition,
        obj.get("split") == "val",
        Path(str(obj.get("checkpoint", ""))).name
        == checkpoint_path.name,
    ]

    if checkpoint_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(checkpoint_step)
        )

    kind = str(item["kind"])

    if kind == "standard":
        checks.extend(
            [
                obj.get("degradation_protocol_sha256")
                == eval_base.degradation_protocol_sha256(),
                int(
                    obj.get(
                        "degradation_implementation_revision",
                        -1,
                    )
                )
                == eval_base.IMPLEMENTATION_REVISION,
            ]
        )

    if kind == "fog":
        checks.extend(
            [
                obj.get("fog_protocol_sha256")
                == eval_base.fog_protocol_sha256(),
                int(
                    obj.get(
                        "fog_implementation_revision",
                        -1,
                    )
                )
                == eval_base.FOG_IMPLEMENTATION_REVISION,
            ]
        )

    return obj if all(checks) else None


# =============================================================================
# Summary helpers
# =============================================================================

def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": result.get("model"),
        "regime": result.get("regime"),
        "condition": result.get("condition"),
        "family": result.get("family"),
        "severity_level": result.get("severity_level"),
        "severity_rank": result.get("severity_rank"),
        "training_relation": result.get("training_relation"),
        "clean_miou": result.get("clean_reference_miou"),
        "miou": result.get("miou"),
        "drop_miou": result.get("drop_miou"),
        "relative_drop_pct": result.get("relative_drop_pct"),
        "retention_pct": result.get("retention_pct"),
        "pixel_accuracy": result.get("pixel_accuracy"),
        "mean_class_accuracy": result.get("mean_class_accuracy"),
        "validation_seconds": result.get("validation_seconds"),
    }


def write_summary_bundle(
    *,
    output_root: Path,
    results: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_step: Optional[int],
) -> None:
    rows = [
        summary_row(result)
        for result in results
    ]

    payload = {
        "validator_version": VALIDATOR_VERSION,
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "backbone": "SegFormer-B2",
        "regime": REGIME,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint_step,
        "robust_training": {
            "families": [
                "gaussian_noise",
                "gaussian_blur",
                "rgb_underexposure",
                "fog",
            ],
            "fog_seen_during_training": True,
        },
        "results": rows,
    }

    save_json(
        output_root / "all_conditions_summary.json",
        payload,
    )

    fields = [
        "model",
        "regime",
        "condition",
        "family",
        "severity_level",
        "severity_rank",
        "training_relation",
        "clean_miou",
        "miou",
        "drop_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        "validation_seconds",
    ]

    write_csv(
        output_root / "all_conditions_summary.csv",
        rows,
        fields,
    )

    standard_rows = [
        row
        for row in rows
        if row["family"]
        in {
            "gaussian_noise",
            "gaussian_blur",
            "rgb_underexposure",
        }
    ]

    if standard_rows:
        save_json(
            robustness_root(output_root)
            / "robustness_summary.json",
            {
                "model": MODEL_ID,
                "model_name": MODEL_NAME,
                "regime": REGIME,
                "degradation_protocol_version": (
                    eval_base.DEGRADATION_PROTOCOL_VERSION
                ),
                "degradation_implementation_revision": (
                    eval_base.IMPLEMENTATION_REVISION
                ),
                "degradation_protocol_sha256": (
                    eval_base.degradation_protocol_sha256()
                ),
                "results": standard_rows,
            },
        )

        write_csv(
            robustness_root(output_root)
            / "robustness_summary.csv",
            standard_rows,
            fields,
        )

    fog_rows = [
        row
        for row in rows
        if row["family"] == "fog"
    ]

    if fog_rows:
        save_json(
            fog_root(output_root)
            / "fog_summary.json",
            {
                "model": MODEL_ID,
                "model_name": MODEL_NAME,
                "regime": REGIME,
                "scientific_role": (
                    "Seen atmospheric degradation under Robust-4 training."
                ),
                "fog_seen_during_training": True,
                "fog_protocol_version": (
                    eval_base.FOG_PROTOCOL_VERSION
                ),
                "fog_implementation_revision": (
                    eval_base.FOG_IMPLEMENTATION_REVISION
                ),
                "fog_protocol_sha256": (
                    eval_base.fog_protocol_sha256()
                ),
                "results": fog_rows,
            },
        )

        write_csv(
            fog_root(output_root)
            / "fog_summary.csv",
            fog_rows,
            fields,
        )


def write_m0_comparison(
    *,
    m1_results: Sequence[Mapping[str, Any]],
    m0_summary_path: Path,
    output_root: Path,
) -> None:
    if not m0_summary_path.is_file():
        print(
            f"[M0 comparison] skipped; not found: {m0_summary_path}",
            flush=True,
        )
        return

    try:
        m0_payload = json.loads(
            m0_summary_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        print(
            f"[M0 comparison] skipped; failed to parse: {exc}",
            flush=True,
        )
        return

    m0_lookup = {
        str(row["condition"]): row
        for row in m0_payload.get("results", [])
        if isinstance(row, Mapping)
        and "condition" in row
    }

    comparison = []

    for result in m1_results:
        condition = str(result["condition"])

        if condition not in m0_lookup:
            continue

        m0 = m0_lookup[condition]

        m0_miou = float(m0["miou"])
        m1_miou = float(result["miou"])

        comparison.append(
            {
                "condition": condition,
                "family": result.get("family"),
                "severity_level": result.get("severity_level"),
                "m0_b2_rgb_clean_miou": m0_miou,
                "m1_b2_rgb_robust4_miou": m1_miou,
                "absolute_gain_miou": (
                    m1_miou - m0_miou
                ),
                "relative_gain_vs_m0_pct": (
                    100.0
                    * (m1_miou - m0_miou)
                    / m0_miou
                    if m0_miou != 0
                    else None
                ),
                "m0_clean_relative_drop_pct": (
                    m0.get("relative_drop_pct")
                ),
                "m1_clean_relative_drop_pct": (
                    result.get("relative_drop_pct")
                ),
            }
        )

    if not comparison:
        print(
            "[M0 comparison] skipped; no matching conditions.",
            flush=True,
        )
        return

    save_json(
        output_root / "comparison_vs_m0.json",
        {
            "m0_source": str(m0_summary_path),
            "m0_model": m0_payload.get("model"),
            "m1_model": MODEL_ID,
            "comparison": comparison,
        },
    )

    write_csv(
        output_root / "comparison_vs_m0.csv",
        comparison,
        [
            "condition",
            "family",
            "severity_level",
            "m0_b2_rgb_clean_miou",
            "m1_b2_rgb_robust4_miou",
            "absolute_gain_miou",
            "relative_gain_vs_m0_pct",
            "m0_clean_relative_drop_pct",
            "m1_clean_relative_drop_pct",
        ],
    )

    print(
        f"[M0 comparison] written: "
        f"{output_root / 'comparison_vs_m0.csv'}",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    checkpoint_path = resolve(
        args.checkpoint
    )

    output_root = resolve(
        args.output_root
    )

    m0_summary_path = resolve(
        args.m0_summary
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = eval_base.get_device(
        args.device
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        eval_base.unwrap_checkpoint_state_dict(
            checkpoint_obj
        )
    )

    validate_checkpoint_metadata(
        checkpoint_meta
    )

    checkpoint_epoch = checkpoint_meta.get(
        "epoch"
    )

    checkpoint_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get(
            "step"
        ),
    )

    checkpoint_epoch_int = (
        int(checkpoint_epoch)
        if checkpoint_epoch is not None
        else None
    )

    checkpoint_step_int = (
        int(checkpoint_step)
        if checkpoint_step is not None
        else None
    )

    training_protocol = checkpoint_meta[
        "protocol"
    ]

    planned_steps = training_protocol.get(
        "total_update_steps"
    )

    print("=" * 126)
    print(
        f"{MODEL_NAME} | COMPLETE 13-CONDITION VALIDATION"
    )
    print("=" * 126)
    print(
        f"checkpoint      : {checkpoint_path}"
    )
    print(
        f"epoch           : "
        f"{checkpoint_epoch_int + 1 if checkpoint_epoch_int is not None else 'unknown'}"
    )
    print(
        f"global_step     : {checkpoint_step_int}"
    )
    print(
        f"planned updates : {planned_steps}"
    )
    print(
        f"device / AMP    : {device} / {amp_enabled}"
    )
    print(
        f"batch size      : {args.batch_size}"
    )
    print(
        "seen families   : Noise / Blur / Underexposure / Fog"
    )
    print(
        f"output root     : {output_root}"
    )
    print(
        f"start local     : "
        f"{eval_base.format_local_datetime(eval_base.local_now())}"
    )
    print("=" * 126)

    if (
        planned_steps is not None
        and checkpoint_step_int is not None
        and checkpoint_step_int != int(planned_steps)
    ):
        difference = (
            int(planned_steps)
            - checkpoint_step_int
        )

        print(
            "[checkpoint note] "
            f"successful optimizer steps are {difference} below the nominal "
            "schedule. This is accepted because all 120 epochs completed and "
            "small AMP overflow skips can reduce global_step.",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model, model_meta = (
        build_b2_rgb_model()
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading returned incompatibilities."
        )

    model.to(device)
    model.eval()

    print(
        f"[model] strict load PASS | "
        f"parameters={model_meta['parameters']['total']:,} | "
        "RGB only",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Save exact validation protocol snapshots
    # -------------------------------------------------------------------------

    standard_root = robustness_root(
        output_root
    )

    standard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_base.write_degradation_protocol(
        standard_root
        / "degradation_protocol.json"
    )

    fog_output_root = fog_root(
        output_root
    )

    fog_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fog_protocol_payload = dict(
        eval_base.FOG_PROTOCOL
    )

    fog_protocol_payload[
        "sha256"
    ] = eval_base.fog_protocol_sha256()

    fog_protocol_payload[
        "evaluation_role_for_this_model"
    ] = (
        "seen degradation family; Fog was included in Robust-4 training"
    )

    save_json(
        fog_output_root
        / "fog_protocol.json",
        fog_protocol_payload,
    )

    # -------------------------------------------------------------------------
    # Conditions / frozen clean reference
    # -------------------------------------------------------------------------

    conditions = eval_base.suite_conditions(
        args.suite
    )

    clean_probe_dataset = (
        eval_base.PotsdamSlidingWindowDataset(
            PROJECT_ROOT,
            split="val",
        )
    )

    if (
        len(clean_probe_dataset.tile_ids) != 6
        or len(
            clean_probe_dataset.window_coordinates
        )
        != 256
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    progress = eval_base.ValidationProgress(
        total_conditions=len(
            conditions
        ),
        tiles_per_condition=6,
        output_path=(
            output_root
            / "validation_progress.json"
        ),
    )

    results = []

    clean_result: Optional[
        Dict[str, Any]
    ] = None

    existing_clean_path = (
        clean_dir(output_root)
        / "metrics.json"
    )

    if existing_clean_path.is_file():
        try:
            candidate = json.loads(
                existing_clean_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                candidate.get("model") == MODEL_ID
                and candidate.get("regime") == REGIME
                and candidate.get("condition") == "Clean"
                and (
                    checkpoint_step_int is None
                    or int(
                        candidate.get(
                            "checkpoint_global_step",
                            -1,
                        )
                    )
                    == checkpoint_step_int
                )
            ):
                clean_result = candidate
        except Exception:
            clean_result = None

    # -------------------------------------------------------------------------
    # Evaluate all selected conditions
    # -------------------------------------------------------------------------

    print(
        f"[validation] {len(conditions)} conditions | "
        f"{len(conditions) * 6} full 6000x6000 tiles",
        flush=True,
    )

    for condition_index, item in enumerate(
        conditions,
        start=1,
    ):
        condition = str(
            item["condition"]
        )

        relation = (
            training_relation_robust4(
                regime=REGIME,
                family=str(
                    item["family"]
                ),
            )
        )

        output_dir = condition_output_dir(
            output_root,
            item,
        )

        metrics_path = (
            output_dir
            / "metrics.json"
        )

        print()
        print("-" * 126)
        print(
            f"[Condition {condition_index:02d}/{len(conditions):02d}] "
            f"{condition} | relation={relation}"
        )
        print("-" * 126)

        existing = None

        if not args.force:
            existing = compatible_existing(
                metrics_path,
                condition=condition,
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step_int,
                item=item,
            )

        if existing is not None:
            result = existing

            progress.mark_skipped_condition(
                condition_index=condition_index,
                condition=condition,
            )

        else:
            dataset = eval_base.build_dataset(
                item=item,
                fog_chunk_rows=args.fog_chunk_rows,
            )

            if item["kind"] != "clean":
                eval_base.degradation_probe(
                    clean_dataset=clean_probe_dataset,
                    degraded_dataset=dataset,
                    condition=condition,
                )

            result = eval_base.evaluate_condition(
                model=model,
                dataset=dataset,
                item=item,
                output_dir=output_dir,
                model_id=MODEL_ID,
                model_name=MODEL_NAME,
                regime=REGIME,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=checkpoint_epoch_int,
                checkpoint_global_step=checkpoint_step_int,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                device=device,
                amp_enabled=amp_enabled,
                log_every=args.log_every,
                confusion_chunk_rows=args.confusion_chunk_rows,
                save_predictions=args.save_predictions,
                condition_index=condition_index,
                progress=progress,
            )

            # Explicitly correct/add Robust-4 semantics.
            result["training_relation"] = relation
            result["fog_seen_during_training"] = (
                item["family"] == "fog"
            )
            result["robust4_training_families"] = [
                "gaussian_noise",
                "gaussian_blur",
                "rgb_underexposure",
                "fog",
            ]

            save_json(
                metrics_path,
                result,
            )

            eval_base.clear_dataset_cache(
                dataset
            )

            del dataset

        if condition == "Clean":
            clean_result = result

        results.append(
            result
        )

        print(
            f"[condition result] "
            f"{condition:<24} | "
            f"mIoU={float(result['miou']):.6f} | "
            f"pixel_acc={float(result['pixel_accuracy']):.6f} | "
            f"time={float(result['validation_seconds']):.1f}s",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Need Clean to compute drop / retention.
    # -------------------------------------------------------------------------

    if clean_result is None:
        raise RuntimeError(
            "Clean result is required. Run --suite all or --suite clean first."
        )

    clean_miou = float(
        clean_result["miou"]
    )

    for result in results:
        if result["condition"] == "Clean":
            result[
                "clean_reference_miou"
            ] = clean_miou

            result[
                "drop_miou"
            ] = 0.0

            result[
                "delta_miou"
            ] = 0.0

            result[
                "relative_drop_pct"
            ] = 0.0

            result[
                "retention_pct"
            ] = 100.0

        else:
            eval_base.add_clean_relative_metrics(
                result,
                clean_miou=clean_miou,
            )

        result["regime"] = REGIME

        result["training_relation"] = (
            training_relation_robust4(
                regime=REGIME,
                family=str(
                    result["family"]
                ),
            )
        )

        result_dir = condition_output_dir(
            output_root,
            {
                "kind": (
                    "clean"
                    if result["condition"] == "Clean"
                    else (
                        "fog"
                        if result["family"] == "fog"
                        else "standard"
                    )
                ),
                "condition": result["condition"],
            },
        )

        save_json(
            result_dir / "metrics.json",
            result,
        )

    write_summary_bundle(
        output_root=output_root,
        results=results,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step_int,
    )

    write_m0_comparison(
        m1_results=results,
        m0_summary_path=m0_summary_path,
        output_root=output_root,
    )

    progress.finish(
        output_root=output_root
    )

    # -------------------------------------------------------------------------
    # Final terminal table
    # -------------------------------------------------------------------------

    print()
    print("=" * 146)
    print(
        f"{MODEL_NAME} | FINAL VALIDATION SUMMARY"
    )
    print("=" * 146)
    print(
        f"{'Condition':<26} "
        f"{'Relation':<24} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'RelDrop%':>10} "
        f"{'Retention%':>12} "
        f"{'Time(s)':>10}"
    )
    print("-" * 146)

    for result in results:
        print(
            f"{str(result['condition']):<26} "
            f"{str(result['training_relation']):<24} "
            f"{float(result['miou']):>10.6f} "
            f"{float(result['drop_miou']):>10.6f} "
            f"{float(result['relative_drop_pct']):>10.3f} "
            f"{float(result['retention_pct']):>12.3f} "
            f"{float(result['validation_seconds']):>10.1f}"
        )

    print("-" * 146)
    print(
        f"Clean mIoU       : {clean_miou:.6f}"
    )
    print(
        f"Summary JSON     : "
        f"{output_root / 'all_conditions_summary.json'}"
    )
    print(
        f"Summary CSV      : "
        f"{output_root / 'all_conditions_summary.csv'}"
    )
    print(
        f"M1 vs M0 CSV     : "
        f"{output_root / 'comparison_vs_m0.csv'}"
    )
    print(
        f"Progress         : "
        f"{output_root / 'validation_progress.json'}"
    )
    print("=" * 146)


if __name__ == "__main__":
    main()
