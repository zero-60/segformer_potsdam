#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified validation for RR-DARF.

Runs:
1) Formal Joint RGB+NIR 13-condition protocol.
2) Cross-modal stress protocol used to test relative modality reliability.

The new model is compared against the already-trained Fixed g=1.0 baseline.
No validation condition changes the trained model or tunes the gate.

Expected location:
    tools/validate_model_b2_darf_rr.py

Dependencies already present in the project:
    tools/validate_model_b2_rgbnir_joint_robust4.py
    tools/validate_model_b2_rgbnir_crossmodal_stress.py
    models/segformer_b2_darf_rr.py

Examples
--------
Formal 13-condition main table only:
    python tools/validate_model_b2_darf_rr.py --suite formal

Cross-modal stress only:
    python tools/validate_model_b2_darf_rr.py --suite stress

Both:
    python tools/validate_model_b2_darf_rr.py --suite all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgbnir_joint_robust4 as joint_val
import validate_model_b2_rgbnir_crossmodal_stress as stress_val

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from models.segformer_b2_darf_rr import (
    MAX_GATE_LOGIT,
    MODEL_ID,
    MODEL_NAME,
    PROTOCOL_VERSION as MODEL_PROTOCOL_VERSION,
    build_model_b2_darf_rr,
)


VALIDATOR_VERSION = "1.0.0"
TRAINING_PROTOCOL_VERSION = "RR-DARF Joint RGB+NIR Robust-4 Protocol v1"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_darf_rr_joint_robust4"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_darf_rr_joint_robust4"
)

FIXED1_FORMAL_REFERENCE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgbnir_fixed1_joint_robust4"
    / "all_conditions_summary.json"
)

FIXED1_STRESS_REFERENCE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgbnir_fixed1_joint_robust4"
    / "crossmodal_stress"
    / "crossmodal_stress_summary.json"
)


def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def save_json(
    path: Path,
    obj: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(
        path.suffix + ".tmp"
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


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
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


def validate_checkpoint_metadata(
    meta: Mapping[str, Any],
) -> None:
    if meta.get("model_id") != MODEL_ID:
        raise RuntimeError(
            "RR-DARF checkpoint model_id mismatch: "
            f"{meta.get('model_id')!r} != {MODEL_ID!r}"
        )

    if meta.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            "RR-DARF checkpoint model_name mismatch."
        )

    if meta.get("variant") != "darf_rr":
        raise RuntimeError(
            "RR-DARF checkpoint variant must be 'darf_rr'."
        )

    if meta.get("regime") != "joint_robust4":
        raise RuntimeError(
            "RR-DARF checkpoint regime must be joint_robust4."
        )

    protocol = meta.get("protocol")
    if not isinstance(
        protocol,
        Mapping,
    ):
        raise RuntimeError(
            "RR-DARF protocol metadata missing."
        )

    if (
        protocol.get(
            "protocol_version"
        )
        != TRAINING_PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "RR-DARF training protocol_version mismatch."
        )

    if not bool(
        protocol.get(
            "quality_gate",
            False,
        )
    ):
        raise RuntimeError(
            "RR-DARF checkpoint says quality_gate=False."
        )

    gate_training = protocol.get(
        "gate_training"
    )
    if not isinstance(
        gate_training,
        Mapping,
    ):
        raise RuntimeError(
            "RR-DARF gate_training metadata missing."
        )

    if (
        gate_training.get(
            "absolute_severity_target"
        )
        is not False
    ):
        raise RuntimeError(
            "RR-DARF must not use an absolute severity gate target."
        )

    if (
        gate_training.get(
            "auxiliary_gate_bce"
        )
        is not False
    ):
        raise RuntimeError(
            "RR-DARF must not use old Gate BCE."
        )

    if (
        gate_training.get(
            "bounded_gate_logit"
        )
        is not True
    ):
        raise RuntimeError(
            "RR-DARF bounded gate metadata missing."
        )

    rel = protocol.get(
        "relative_reliability_auxiliary_training"
    )
    if not isinstance(
        rel,
        Mapping,
    ) or not bool(
        rel.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "Relative-reliability auxiliary training metadata missing."
        )


def build_model_from_checkpoint_meta(
    meta: Mapping[str, Any],
):
    protocol = meta["protocol"]
    gate_training = protocol[
        "gate_training"
    ]

    initial_gate = float(
        gate_training[
            "initial_gate"
        ]
    )
    max_gate_logit = float(
        gate_training.get(
            "max_gate_logit",
            MAX_GATE_LOGIT,
        )
    )

    return build_model_b2_darf_rr(
        PROJECT_ROOT,
        initial_gate=initial_gate,
        max_gate_logit=max_gate_logit,
    )


def clean_probe() -> PotsdamSlidingWindowDataset:
    dataset = PotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
    )

    if (
        len(dataset.tile_ids)
        != 6
        or len(
            dataset.window_coordinates
        )
        != 256
        or int(
            dataset.spec.tile_size
        )
        != 6000
        or int(
            dataset.spec.crop_size
        )
        != 512
    ):
        raise RuntimeError(
            "Frozen Potsdam validation protocol changed."
        )

    return dataset


def class_column_name(
    name: str,
) -> str:
    chars = [
        ch
        if ch.isalnum()
        else "_"
        for ch in name.lower()
    ]
    return "iou_" + "_".join(
        part
        for part in "".join(
            chars
        ).split("_")
        if part
    )


def result_row(
    result: Mapping[str, Any],
    *,
    suite: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "suite": suite,
        "condition": result["condition"],
        "family": result["family"],
        "severity_level": result.get(
            "severity_level"
        ),
        "stress_axis": result.get(
            "stress_axis"
        ),
        "mismatch_role": result.get(
            "mismatch_role"
        ),
        "rgb_level": result.get(
            "rgb_level"
        ),
        "nir_level": result.get(
            "nir_level"
        ),
        "shift_pixels": result.get(
            "shift_pixels"
        ),
        "miou": float(
            result["miou"]
        ),
        "pixel_accuracy": float(
            result["pixel_accuracy"]
        ),
        "mean_class_accuracy": float(
            result["mean_class_accuracy"]
        ),
        "clean_reference_miou": result.get(
            "clean_reference_miou"
        ),
        "drop_miou": result.get(
            "drop_miou"
        ),
        "relative_drop_pct": result.get(
            "relative_drop_pct"
        ),
        "retention_pct": result.get(
            "retention_pct"
        ),
    }

    for item in result[
        "per_class"
    ]:
        row[
            class_column_name(
                str(
                    item[
                        "class_name"
                    ]
                )
            )
        ] = float(
            item["iou"]
        )

    return row


def add_rr_semantics(
    result: Dict[str, Any],
    *,
    suite: str,
    item: Mapping[str, Any],
) -> None:
    result[
        "validator_version"
    ] = VALIDATOR_VERSION
    result[
        "model"
    ] = MODEL_ID
    result[
        "model_name"
    ] = MODEL_NAME
    result[
        "variant"
    ] = "darf_rr"
    result[
        "public_variant"
    ] = "darf_rr"
    result[
        "quality_gate"
    ] = True
    result[
        "gate_supervision"
    ] = (
        "relative_reliability_ranking"
    )
    result[
        "absolute_severity_gate_target"
    ] = False
    result[
        "fusion"
    ] = (
        "RR-DARF RGB-anchored NIR residual fusion"
    )
    result[
        "fusion_rule"
    ] = (
        "F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
    )
    result[
        "validation_suite"
    ] = suite

    if suite == "formal":
        result[
            "joint_validation"
        ] = True
        result[
            "joint_validation_protocol_version"
        ] = (
            joint_val.JOINT_VALIDATION_PROTOCOL_VERSION
        )
        result[
            "joint_validation_protocol_sha256"
        ] = (
            joint_val.joint_validation_protocol_sha256()
        )
    else:
        result[
            "crossmodal_stress"
        ] = True
        result[
            "crossmodal_stress_protocol_version"
        ] = (
            stress_val.STRESS_PROTOCOL_VERSION
        )
        result[
            "crossmodal_stress_protocol_sha256"
        ] = (
            stress_val.protocol_sha256()
        )
        result[
            "stress_axis"
        ] = str(
            item["stress_axis"]
        )
        result[
            "stress_kind"
        ] = str(
            item["kind"]
        )
        result[
            "rgb_state"
        ] = str(
            item["rgb_state"]
        )
        result[
            "nir_state"
        ] = str(
            item["nir_state"]
        )

        for key in (
            "mismatch_role",
            "rgb_level",
            "nir_level",
            "shift_pixels",
        ):
            if key in item:
                result[key] = item[key]


def compatible_existing(
    path: Path,
    *,
    condition: str,
    checkpoint_step: Optional[int],
    suite: str,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    checks = [
        obj.get(
            "model"
        )
        == MODEL_ID,
        obj.get(
            "condition"
        )
        == condition,
        obj.get(
            "variant"
        )
        == "darf_rr",
        obj.get(
            "validation_suite"
        )
        == suite,
    ]

    if checkpoint_step is not None:
        checks.append(
            int(
                obj.get(
                    "checkpoint_global_step",
                    -1,
                )
            )
            == int(
                checkpoint_step
            )
        )

    return (
        obj
        if all(
            checks
        )
        else None
    )


def condition_dir(
    output_root: Path,
    *,
    suite: str,
    item: Mapping[str, Any],
) -> Path:
    if suite == "formal":
        return joint_val.condition_output_dir(
            output_root,
            item,
        )

    return stress_val.output_dir_for_item(
        output_root,
        item,
    )


def build_dataset(
    *,
    suite: str,
    item: Mapping[str, Any],
    fog_chunk_rows: int,
):
    if suite == "formal":
        return joint_val.build_dataset(
            item=item,
            nir_fog_scatter_ratio=(
                joint_val.DEFAULT_NIR_FOG_SCATTER_RATIO
            ),
            fog_chunk_rows=(
                fog_chunk_rows
            ),
        )

    return stress_val.build_dataset(
        item=item,
        fog_chunk_rows=(
            fog_chunk_rows
        ),
    )


def conditions_for_suite(
    suite: str,
):
    if suite == "formal":
        return joint_val.suite_conditions(
            "all"
        )

    return stress_val.stress_conditions(
        "all"
    )


def probe_transform(
    *,
    suite: str,
    clean_dataset,
    dataset,
    item,
) -> None:
    if item["kind"] == "clean":
        return

    if suite == "formal":
        joint_val.joint_degradation_probe(
            clean_dataset=clean_dataset,
            degraded_dataset=dataset,
            condition=str(
                item["condition"]
            ),
        )
    else:
        stress_val.stress_probe(
            clean_dataset=clean_dataset,
            stressed_dataset=dataset,
            item=item,
        )


def gate_statistics(
    results: Sequence[
        Mapping[str, Any]
    ],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for result in results:
        for row in joint_val.darf_gate_rows(
            result
        ):
            x = dict(row)
            x["stress_axis"] = result.get(
                "stress_axis"
            )
            x["mismatch_role"] = result.get(
                "mismatch_role"
            )
            x["rgb_level"] = result.get(
                "rgb_level"
            )
            x["nir_level"] = result.get(
                "nir_level"
            )
            x["shift_pixels"] = result.get(
                "shift_pixels"
            )
            rows.append(x)

    clean_gate = {
        int(row["scale"]): float(
            row["g_nir_mean"]
        )
        for row in rows
        if row["condition"] == "Clean"
    }

    for row in rows:
        scale = int(
            row["scale"]
        )
        row[
            "delta_mean_vs_clean"
        ] = (
            float(
                row[
                    "g_nir_mean"
                ]
            )
            - clean_gate[
                scale
            ]
            if scale in clean_gate
            else None
        )

    return rows


def write_reference_comparison(
    *,
    rows: Sequence[
        Mapping[str, Any]
    ],
    reference_path: Path,
    output_path: Path,
) -> None:
    if not reference_path.is_file():
        print(
            f"[reference] missing, skipped: "
            f"{reference_path}",
            flush=True,
        )
        return

    payload = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )

    reference = {
        str(row["condition"]): row
        for row in payload.get(
            "results",
            []
        )
    }

    out = []

    for row in rows:
        condition = str(
            row["condition"]
        )

        if condition not in reference:
            continue

        old = float(
            reference[
                condition
            ][
                "miou"
            ]
        )
        new = float(
            row["miou"]
        )

        out.append(
            {
                "condition": condition,
                "suite": row["suite"],
                "fixed1_miou": old,
                "rr_darf_miou": new,
                "rr_darf_minus_fixed1_miou": (
                    new
                    - old
                ),
            }
        )

    if not out:
        return

    write_csv(
        output_path,
        out,
        [
            "condition",
            "suite",
            "fixed1_miou",
            "rr_darf_miou",
            "rr_darf_minus_fixed1_miou",
        ],
    )


def evaluate_suite(
    *,
    suite: str,
    model,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_step: Optional[int],
    output_root: Path,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    log_every: int,
    confusion_chunk_rows: int,
    fog_chunk_rows: int,
    save_predictions: bool,
    force: bool,
) -> List[Dict[str, Any]]:
    conditions = conditions_for_suite(
        suite
    )

    clean_dataset = clean_probe()

    progress = joint_val.JointValidationProgress(
        total_conditions=len(
            conditions
        ),
        tiles_per_condition=6,
        output_path=(
            output_root
            / "validation_progress.json"
        ),
        device=device,
    )

    results: List[
        Dict[str, Any]
    ] = []

    clean_result: Optional[
        Dict[str, Any]
    ] = None

    for condition_index, item in enumerate(
        conditions,
        start=1,
    ):
        condition = str(
            item["condition"]
        )

        out_dir = condition_dir(
            output_root,
            suite=suite,
            item=item,
        )
        metrics_path = (
            out_dir
            / "metrics.json"
        )

        print()
        print(
            "-" * 132
        )
        print(
            f"[{suite.upper()} "
            f"{condition_index:02d}/"
            f"{len(conditions):02d}] "
            f"{condition}"
        )
        print(
            "-" * 132
        )

        existing = None

        if not force:
            existing = compatible_existing(
                metrics_path,
                condition=condition,
                checkpoint_step=(
                    checkpoint_step
                ),
                suite=suite,
            )

        if existing is not None:
            result = existing
            progress.mark_skipped_condition(
                condition_index=(
                    condition_index
                ),
                condition=condition,
            )
            print(
                f"[resume] reused | "
                f"mIoU={float(result['miou']):.6f}",
                flush=True,
            )
        else:
            dataset = build_dataset(
                suite=suite,
                item=item,
                fog_chunk_rows=(
                    fog_chunk_rows
                ),
            )

            probe_transform(
                suite=suite,
                clean_dataset=(
                    clean_dataset
                ),
                dataset=dataset,
                item=item,
            )

            result = (
                joint_val.base.evaluate_condition(
                    model=model,
                    dataset=dataset,
                    item=item,
                    output_dir=out_dir,
                    model_id=MODEL_ID,
                    model_name=MODEL_NAME,
                    regime="joint_robust4",
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    checkpoint_epoch=(
                        checkpoint_epoch
                    ),
                    checkpoint_global_step=(
                        checkpoint_step
                    ),
                    batch_size=(
                        batch_size
                    ),
                    num_workers=(
                        num_workers
                    ),
                    pin_memory=(
                        pin_memory
                    ),
                    device=device,
                    amp_enabled=(
                        amp_enabled
                    ),
                    log_every=(
                        log_every
                    ),
                    confusion_chunk_rows=(
                        confusion_chunk_rows
                    ),
                    save_predictions=(
                        save_predictions
                    ),
                    condition_index=(
                        condition_index
                    ),
                    progress=progress,
                )
            )

            add_rr_semantics(
                result,
                suite=suite,
                item=item,
            )

            save_json(
                metrics_path,
                result,
            )

            joint_val.base.clear_dataset_cache(
                dataset
            )
            del dataset

        if condition == "Clean":
            clean_result = result

        results.append(
            result
        )

    if clean_result is None:
        raise RuntimeError(
            f"{suite}: Clean result missing."
        )

    clean_miou = float(
        clean_result[
            "miou"
        ]
    )

    item_lookup = {
        str(item["condition"]): item
        for item in conditions
    }

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
            joint_val.base.add_clean_relative_metrics(
                result,
                clean_miou=(
                    clean_miou
                ),
            )

        item = item_lookup[
            str(
                result[
                    "condition"
                ]
            )
        ]

        add_rr_semantics(
            result,
            suite=suite,
            item=item,
        )

        save_json(
            condition_dir(
                output_root,
                suite=suite,
                item=item,
            )
            / "metrics.json",
            result,
        )

    rows = [
        result_row(
            result,
            suite=suite,
        )
        for result in results
    ]

    class_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith(
                "iou_"
            )
        }
    )

    fields = [
        "suite",
        "condition",
        "family",
        "severity_level",
        "stress_axis",
        "mismatch_role",
        "rgb_level",
        "nir_level",
        "shift_pixels",
        "miou",
        "clean_reference_miou",
        "drop_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        *class_fields,
    ]

    summary_name = (
        "all_conditions_summary"
        if suite == "formal"
        else "crossmodal_stress_summary"
    )

    save_json(
        output_root
        / f"{summary_name}.json",
        {
            "validator_version": (
                VALIDATOR_VERSION
            ),
            "model": MODEL_ID,
            "model_name": MODEL_NAME,
            "variant": "darf_rr",
            "public_variant": "darf_rr",
            "suite": suite,
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_global_step": (
                checkpoint_step
            ),
            "clean_miou": clean_miou,
            "results": rows,
        },
    )

    write_csv(
        output_root
        / f"{summary_name}.csv",
        rows,
        fields,
    )

    gate_rows = gate_statistics(
        results
    )

    save_json(
        output_root
        / "gate_statistics.json",
        {
            "model": MODEL_ID,
            "suite": suite,
            "rows": gate_rows,
        },
    )

    write_csv(
        output_root
        / "gate_statistics.csv",
        gate_rows,
        [
            "condition",
            "stress_axis",
            "family",
            "severity_level",
            "mismatch_role",
            "rgb_level",
            "nir_level",
            "shift_pixels",
            "scale",
            "count",
            "g_nir_mean",
            "g_nir_std",
            "g_nir_min",
            "g_nir_max",
            "delta_mean_vs_clean",
        ],
    )

    reference = (
        FIXED1_FORMAL_REFERENCE
        if suite == "formal"
        else FIXED1_STRESS_REFERENCE
    )

    write_reference_comparison(
        rows=rows,
        reference_path=(
            reference
        ),
        output_path=(
            output_root
            / "rr_darf_vs_fixed1.csv"
        ),
    )

    progress.finish(
        output_root=(
            output_root
        )
    )

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate RR-DARF on formal Joint and/or cross-modal stress protocols."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    p.add_argument(
        "--suite",
        choices=(
            "formal",
            "stress",
            "all",
        ),
        default="all",
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=16,
    )

    p.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
    )

    p.add_argument(
        "--fog-chunk-rows",
        type=int,
        default=128,
    )

    p.add_argument(
        "--save-predictions",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    x = p.parse_args()

    if x.batch_size <= 0:
        p.error(
            "--batch-size must be > 0"
        )

    if x.num_workers != 0:
        p.error(
            "--num-workers must remain 0 for full-tile transformed cache correctness."
        )

    return x


def main() -> None:
    x = parse_args()

    checkpoint_path = resolve(
        x.checkpoint
    )
    output_root = resolve(
        x.output_root
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        joint_val.base.get_device(
            x.device
        )
    )
    amp_enabled = (
        device.type == "cuda"
        and not x.no_amp
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        joint_val.base.unwrap_checkpoint_state_dict(
            checkpoint_obj
        )
    )

    validate_checkpoint_metadata(
        checkpoint_meta
    )

    model, model_meta = (
        build_model_from_checkpoint_meta(
            checkpoint_meta
        )
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
            "Strict RR-DARF checkpoint load returned incompatibilities."
        )

    model.to(
        device
    )
    model.eval()

    checkpoint_epoch = checkpoint_meta.get(
        "epoch"
    )
    checkpoint_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get(
            "step"
        ),
    )

    checkpoint_epoch = (
        int(
            checkpoint_epoch
        )
        if checkpoint_epoch
        is not None
        else None
    )
    checkpoint_step = (
        int(
            checkpoint_step
        )
        if checkpoint_step
        is not None
        else None
    )

    print(
        "=" * 132
    )
    print(
        "RR-DARF VALIDATION"
    )
    print(
        "=" * 132
    )
    print(
        f"model               : {MODEL_ID}"
    )
    print(
        f"checkpoint          : {checkpoint_path}"
    )
    print(
        f"checkpoint step     : {checkpoint_step}"
    )
    print(
        f"suite               : {x.suite}"
    )
    print(
        f"device / AMP        : {device} / {amp_enabled}"
    )
    print(
        f"output              : {output_root}"
    )
    print(
        joint_val.gpu_memory_text(
            device
        )
    )
    print(
        "=" * 132
    )

    suites = (
        ["formal", "stress"]
        if x.suite == "all"
        else [
            x.suite
        ]
    )

    started = time.time()

    for suite in suites:
        suite_root = (
            output_root
            / (
                "formal_joint"
                if suite
                == "formal"
                else "crossmodal_stress"
            )
        )

        evaluate_suite(
            suite=suite,
            model=model,
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_epoch=(
                checkpoint_epoch
            ),
            checkpoint_step=(
                checkpoint_step
            ),
            output_root=(
                suite_root
            ),
            device=device,
            amp_enabled=(
                amp_enabled
            ),
            batch_size=(
                x.batch_size
            ),
            num_workers=(
                x.num_workers
            ),
            pin_memory=(
                x.pin_memory
            ),
            log_every=(
                x.log_every
            ),
            confusion_chunk_rows=(
                x.confusion_chunk_rows
            ),
            fog_chunk_rows=(
                x.fog_chunk_rows
            ),
            save_predictions=(
                x.save_predictions
            ),
            force=x.force,
        )

    elapsed = (
        time.time()
        - started
    )

    print()
    print(
        "=" * 132
    )
    print(
        "RR-DARF VALIDATION COMPLETE"
    )
    print(
        f"elapsed : "
        f"{joint_val.base.format_duration(elapsed)}"
    )
    print(
        f"output  : {output_root}"
    )
    print(
        "=" * 132
    )


if __name__ == "__main__":
    main()
