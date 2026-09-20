#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgrade Model A robustness results from RGB Degradation Protocol v1 -> v2.

What this script does
---------------------
1. Validates the already-completed v1 protocol + Model A v1 summary.
2. Proves that v2 leaves Gaussian Noise and Gaussian Blur definitions unchanged.
3. Carries forward those six Model A v1 results without recomputing inference.
4. Evaluates ONLY the new v2 RGB underexposure conditions:
       rgb_underexposure_L1  alpha=0.8
       rgb_underexposure_L2  alpha=0.6
       rgb_underexposure_L3  alpha=0.4
5. Produces a complete Model A v2 summary containing all nine degraded
   conditions plus the Clean reference.

The Clean checkpoint and metric implementation are unchanged.

Expected locations
------------------
evaluation/rgb_degradation_protocol.py     <- replace with v2 module
tools/validate_model_a_rgb.py              <- existing validated Clean evaluator
tools/validate_model_a_exposure_v2.py      <- this script

Existing v1 results expected at:
outputs/evaluation/model_a_rgb/robustness_val/

New v2 outputs:
outputs/evaluation/model_a_rgb/robustness_val_v2/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


from data_pipeline.potsdam_dataset import (
    PotsdamSlidingWindowDataset,
)
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL,
    DEGRADATION_PROTOCOL_VERSION,
    LEGACY_V1_PROTOCOL_SHA256,
    DegradedPotsdamSlidingWindowDataset,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    validate_v1_corruption_compatibility,
    write_degradation_protocol,
)
from models.segformer_rgb import build_model_a_rgb

try:
    from validate_model_a_rgb import (
        CLASS_NAMES,
        IGNORE_INDEX,
        NUM_CLASSES,
        confusion_from_prediction,
        infer_one_tile,
        metrics_from_confusion,
        unwrap_checkpoint_state_dict,
        validate_checkpoint_metadata,
        write_confusion_csv,
        write_per_class_csv,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import tools/validate_model_a_rgb.py. "
        "Keep the successful Clean validator in tools/."
    ) from exc


MODEL_NAME = "Model A (RGB baseline)"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "model_a_rgb"
    / "checkpoints"
    / "final.pt"
)

DEFAULT_CLEAN_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "clean_val"
    / "metrics.json"
)

DEFAULT_V1_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val_v2"
)

NEW_CORRUPTION = "rgb_underexposure"
NEW_LEVELS = ("L1", "L2", "L3")

LEGACY_CONDITIONS = (
    "gaussian_noise_L1",
    "gaussian_noise_L2",
    "gaussian_noise_L3",
    "gaussian_blur_L1",
    "gaussian_blur_L2",
    "gaussian_blur_L3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade Model A robustness results to degradation protocol v2 "
            "by evaluating only RGB underexposure L1-L3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--clean-metrics",
        type=Path,
        default=DEFAULT_CLEAN_METRICS,
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=DEFAULT_V1_DIR,
        help="Existing completed Model A protocol-v1 robustness directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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
        "--log-every",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--confusion-chunk-rows",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute exposure conditions even if compatible v2 metrics exist.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")
    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    return args


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return obj


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
    ) as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def get_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    if (
        device.type == "cuda"
        and device.index is not None
    ):
        torch.cuda.set_device(device.index)

    return device


def validate_clean_reference(
    clean: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if clean.get("model") != "A_RGB":
        raise RuntimeError(
            "Clean reference is not Model A."
        )

    if clean.get("condition") != "Clean":
        raise RuntimeError(
            "Clean reference condition is not Clean."
        )

    if clean.get("split") != "val":
        raise RuntimeError(
            "Clean reference split is not val."
        )

    if "GLOBAL confusion matrix" not in str(
        clean.get("metric_scope", "")
    ):
        raise RuntimeError(
            "Clean reference metric scope is not GLOBAL full-tile."
        )

    clean_miou = float(
        clean["miou"]
    )
    if not (
        0.0
        <= clean_miou
        <= 1.0
    ):
        raise RuntimeError(
            f"Invalid Clean mIoU: {clean_miou}"
        )

    recorded_checkpoint = clean.get(
        "checkpoint"
    )

    if recorded_checkpoint:
        if (
            Path(str(recorded_checkpoint)).name
            != checkpoint_path.name
        ):
            raise RuntimeError(
                "Clean checkpoint filename does not match requested "
                "Model A checkpoint."
            )


def validate_v1_summary(
    summary: Mapping[str, Any],
    *,
    clean_miou: float,
) -> List[Dict[str, Any]]:
    if summary.get("model") != "A_RGB":
        raise RuntimeError(
            "Existing v1 robustness summary is not Model A."
        )

    if summary.get(
        "degradation_protocol_version"
    ) != "RGB Degradation Protocol v1":
        raise RuntimeError(
            "Existing robustness summary is not protocol v1."
        )

    if summary.get(
        "degradation_protocol_sha256"
    ) != LEGACY_V1_PROTOCOL_SHA256:
        raise RuntimeError(
            "Existing robustness summary v1 hash differs from the "
            "completed frozen protocol."
        )

    clean_ref = summary.get(
        "clean_reference"
    )
    if not isinstance(
        clean_ref,
        Mapping,
    ):
        raise RuntimeError(
            "v1 summary has no clean_reference."
        )

    if not math.isclose(
        float(clean_ref["miou"]),
        clean_miou,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "v1 summary Clean mIoU differs from current Clean metrics."
        )

    rows = summary.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(
            "v1 summary has no results list."
        )

    by_condition: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(
                "v1 result row is not a JSON object."
            )

        cond = str(
            row.get("condition")
        )

        if cond in by_condition:
            raise RuntimeError(
                f"Duplicate v1 condition: {cond}"
            )

        by_condition[cond] = dict(row)

    if set(by_condition) != set(
        LEGACY_CONDITIONS
    ):
        raise RuntimeError(
            "Existing v1 summary does not contain exactly the six "
            "expected Noise/Blur conditions.\n"
            f"got={sorted(by_condition)}"
        )

    carried: List[Dict[str, Any]] = []

    for cond in LEGACY_CONDITIONS:
        row = dict(
            by_condition[cond]
        )

        row["provenance"] = (
            "carried_forward_from_v1_unchanged_definition"
        )
        row["source_protocol_version"] = (
            "RGB Degradation Protocol v1"
        )
        row["source_protocol_sha256"] = (
            LEGACY_V1_PROTOCOL_SHA256
        )

        carried.append(row)

    return carried


def clean_per_class_map(
    clean: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    rows = clean.get("per_class")

    if not isinstance(rows, list):
        raise RuntimeError(
            "Clean metrics missing per_class."
        )

    out: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        out[int(row["class_id"])] = dict(row)

    if set(out) != set(
        range(NUM_CLASSES)
    ):
        raise RuntimeError(
            "Clean per-class ids are incomplete."
        )

    return out


def validate_exposure_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: DegradedPotsdamSlidingWindowDataset,
    alpha: float,
) -> None:
    """
    Verify the new v2 condition is truly RGB-only and exposure-like.
    """
    clean_item = clean_dataset[0]
    degraded_item = degraded_dataset[0]

    for key in (
        "tile_id",
        "window_index",
        "x",
        "y",
        "height",
        "width",
    ):
        if (
            clean_item[key]
            != degraded_item[key]
        ):
            raise RuntimeError(
                f"Exposure changed frozen metadata {key}."
            )

    if not torch.equal(
        clean_item["nir"],
        degraded_item["nir"],
    ):
        max_diff = float(
            (
                clean_item["nir"]
                - degraded_item["nir"]
            )
            .abs()
            .max()
            .item()
        )

        raise RuntimeError(
            "NIR changed under RGB-only exposure degradation. "
            f"max_abs_diff={max_diff}"
        )

    if torch.equal(
        clean_item["rgb"],
        degraded_item["rgb"],
    ):
        raise RuntimeError(
            "Exposure-degraded RGB equals Clean RGB on the probe window."
        )

    # Because RGB has already gone through ImageNet normalization at this
    # stage, use mean absolute change as a robust sanity diagnostic rather
    # than trying to reconstruct alpha from normalized tensors.
    mean_abs_change = float(
        (
            clean_item["rgb"]
            - degraded_item["rgb"]
        )
        .abs()
        .mean()
        .item()
    )

    print(
        "[probe] PASS | "
        "same window metadata | "
        "NIR bit-identical | "
        f"underexposure alpha={alpha:.3f} | "
        f"RGB normalized mean_abs_change={mean_abs_change:.6f}"
    )


def write_per_class_robustness_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "class_id",
        "class_name",
        "clean_iou",
        "iou",
        "drop_iou",
        "delta_iou",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "gt_pixels",
        "pred_pixels",
        "union_pixels",
    ]

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
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def evaluate_exposure_condition(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_global_step: Optional[int],
    clean: Mapping[str, Any],
    clean_class_map: Mapping[int, Mapping[str, Any]],
    level: str,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    confusion_chunk_rows: int,
    save_predictions: bool,
) -> Dict[str, Any]:
    corruption = NEW_CORRUPTION
    cond = condition_name(
        corruption,
        level,
    )
    spec = condition_spec(
        corruption,
        level,
    )

    cond_dir = output_dir / cond
    cond_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print(f"CONDITION | {cond}")
    print("=" * 92)
    print(f"parameters: {spec}")

    dataset = DegradedPotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
        corruption=corruption,
        level=level,
    )

    clean_probe = PotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
    )

    validate_exposure_probe(
        clean_dataset=clean_probe,
        degraded_dataset=dataset,
        alpha=float(
            spec["alpha"]
        ),
    )

    del clean_probe

    tile_ids = list(
        dataset.tile_ids
    )
    windows_per_tile = len(
        dataset.window_coordinates
    )

    if (
        len(tile_ids) != 6
        or windows_per_tile != 256
    ):
        raise RuntimeError(
            "Frozen validation protocol changed: "
            f"tiles={len(tile_ids)}, "
            f"windows/tile={windows_per_tile}"
        )

    per_tile_path = (
        cond_dir
        / "per_tile_metrics.jsonl"
    )
    if per_tile_path.exists():
        per_tile_path.unlink()

    pred_dir = (
        cond_dir
        / "predictions"
    )
    if save_predictions:
        pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    global_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    per_tile_results: List[Dict[str, Any]] = []
    start = time.time()

    for tile_index, tile_id in enumerate(
        tile_ids
    ):
        print(
            f"[{cond}] "
            f"tile {tile_index + 1}/"
            f"{len(tile_ids)} | "
            f"{tile_id}",
            flush=True,
        )

        prediction, tile_runtime = infer_one_tile(
            model=model,
            dataset=dataset,
            tile_id=tile_id,
            tile_index=tile_index,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            device=device,
            amp_enabled=amp_enabled,
            log_every=log_every,
        )

        target_t = dataset.load_full_label(
            tile_id
        )
        target = target_t.numpy()

        tile_confusion = confusion_from_prediction(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            chunk_rows=confusion_chunk_rows,
        )

        global_confusion += (
            tile_confusion
        )

        tile_metrics = metrics_from_confusion(
            tile_confusion,
            CLASS_NAMES,
        )

        tile_record = {
            **tile_runtime,
            "condition": cond,
            "corruption": corruption,
            "severity_level": level,
            "severity_parameters": spec,
            "miou": tile_metrics["miou"],
            "pixel_accuracy": (
                tile_metrics[
                    "pixel_accuracy"
                ]
            ),
            "mean_class_accuracy": (
                tile_metrics[
                    "mean_class_accuracy"
                ]
            ),
            "valid_pixels": (
                tile_metrics[
                    "valid_pixels"
                ]
            ),
            "per_class_iou": {
                row["class_name"]: row["iou"]
                for row in tile_metrics[
                    "per_class"
                ]
            },
        }

        per_tile_results.append(
            tile_record
        )
        append_jsonl(
            per_tile_path,
            tile_record,
        )

        if save_predictions:
            np.save(
                pred_dir
                / f"{tile_id}_pred.npy",
                prediction,
                allow_pickle=False,
            )

        print(
            f"  tile mIoU="
            f"{tile_metrics['miou']:.6f} | "
            f"pixel_acc="
            f"{tile_metrics['pixel_accuracy']:.6f} | "
            f"time="
            f"{tile_runtime['inference_seconds']:.1f}s"
        )

        del prediction
        del target
        del target_t

        dataset.clear_degradation_cache()

    global_metrics = metrics_from_confusion(
        global_confusion,
        CLASS_NAMES,
    )

    clean_miou = float(
        clean["miou"]
    )
    degraded_miou = float(
        global_metrics["miou"]
    )

    drop_miou = (
        clean_miou
        - degraded_miou
    )
    delta_miou = (
        degraded_miou
        - clean_miou
    )

    relative_drop_pct = (
        100.0
        * drop_miou
        / clean_miou
    )

    retention_pct = (
        100.0
        * degraded_miou
        / clean_miou
    )

    per_class: List[Dict[str, Any]] = []

    for row in global_metrics[
        "per_class"
    ]:
        cid = int(
            row["class_id"]
        )

        clean_iou = float(
            clean_class_map[cid][
                "iou"
            ]
        )
        degraded_iou = float(
            row["iou"]
        )

        enriched = dict(row)
        enriched["clean_iou"] = clean_iou
        enriched["drop_iou"] = (
            clean_iou
            - degraded_iou
        )
        enriched["delta_iou"] = (
            degraded_iou
            - clean_iou
        )

        per_class.append(
            enriched
        )

    elapsed = (
        time.time()
        - start
    )

    result: Dict[str, Any] = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "condition": cond,
        "corruption": corruption,
        "severity_level": level,
        "severity_rank": int(
            level[1:]
        ),
        "severity_parameters": spec,
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL_VERSION
        ),
        "degradation_protocol_sha256": (
            degradation_protocol_sha256()
        ),
        "provenance": (
            "evaluated_new_under_protocol_v2"
        ),
        "rgb_degraded": True,
        "nir_used_by_model": False,
        "nir_source": "clean / unchanged",
        "split": "val",
        "metric_scope": (
            "GLOBAL confusion matrix after full-tile mean-logit fusion"
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch_zero_based": (
            checkpoint_epoch
        ),
        "checkpoint_global_step": (
            checkpoint_global_step
        ),
        "clean_reference_miou": (
            clean_miou
        ),
        "miou": degraded_miou,
        "drop_miou": drop_miou,
        "delta_miou": delta_miou,
        "relative_drop_pct": (
            relative_drop_pct
        ),
        "retention_pct": (
            retention_pct
        ),
        "pixel_accuracy": (
            global_metrics[
                "pixel_accuracy"
            ]
        ),
        "mean_class_accuracy": (
            global_metrics[
                "mean_class_accuracy"
            ]
        ),
        "valid_pixels": (
            global_metrics[
                "valid_pixels"
            ]
        ),
        "per_class": per_class,
        "confusion_matrix": (
            global_confusion.tolist()
        ),
        "tiles": per_tile_results,
        "num_tiles": len(
            tile_ids
        ),
        "windows_per_tile": (
            windows_per_tile
        ),
        "total_windows": len(
            dataset
        ),
        "validation_seconds": (
            elapsed
        ),
    }

    write_json(
        cond_dir
        / "metrics.json",
        result,
    )

    write_per_class_csv(
        cond_dir
        / "per_class_metrics.csv",
        per_class,
    )

    write_per_class_robustness_csv(
        cond_dir
        / "per_class_robustness.csv",
        per_class,
    )

    write_confusion_csv(
        cond_dir
        / "confusion_matrix.csv",
        global_confusion,
        CLASS_NAMES,
    )

    print("-" * 92)
    print(
        f"{cond} | "
        f"mIoU={degraded_miou:.6f} | "
        f"Drop={drop_miou:.6f} | "
        f"RelativeDrop={relative_drop_pct:.2f}% | "
        f"Retention={retention_pct:.2f}%"
    )
    print("-" * 92)

    return result


def load_existing_v2_exposure(
    path: Path,
    *,
    level: str,
    checkpoint_path: Path,
    clean_miou: float,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    try:
        obj = read_json(path)
    except Exception:
        return None

    expected_condition = condition_name(
        NEW_CORRUPTION,
        level,
    )

    checks = [
        obj.get("model") == "A_RGB",
        obj.get("condition") == expected_condition,
        obj.get("corruption") == NEW_CORRUPTION,
        obj.get("severity_level") == level,
        obj.get("split") == "val",
        obj.get(
            "degradation_protocol_sha256"
        )
        == degradation_protocol_sha256(),
        Path(
            str(
                obj.get(
                    "checkpoint",
                    "",
                )
            )
        ).name
        == checkpoint_path.name,
        math.isclose(
            float(
                obj.get(
                    "clean_reference_miou",
                    float("nan"),
                )
            ),
            clean_miou,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    ]

    if all(checks):
        return obj

    return None


def summary_row(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": result["model"],
        "condition": result["condition"],
        "corruption": result["corruption"],
        "severity_level": result["severity_level"],
        "severity_rank": result["severity_rank"],
        "clean_miou": result.get(
            "clean_reference_miou",
            result.get(
                "clean_miou"
            ),
        ),
        "miou": result["miou"],
        "drop_miou": result["drop_miou"],
        "delta_miou": result["delta_miou"],
        "relative_drop_pct": result[
            "relative_drop_pct"
        ],
        "retention_pct": result[
            "retention_pct"
        ],
        "pixel_accuracy": result[
            "pixel_accuracy"
        ],
        "mean_class_accuracy": result[
            "mean_class_accuracy"
        ],
        "validation_seconds": result[
            "validation_seconds"
        ],
        "provenance": result.get(
            "provenance",
            "",
        ),
        "source_protocol_version": result.get(
            "source_protocol_version",
            DEGRADATION_PROTOCOL_VERSION,
        ),
        "source_protocol_sha256": result.get(
            "source_protocol_sha256",
            degradation_protocol_sha256(),
        ),
    }


def write_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "model",
        "condition",
        "corruption",
        "severity_level",
        "severity_rank",
        "clean_miou",
        "miou",
        "drop_miou",
        "delta_miou",
        "relative_drop_pct",
        "retention_pct",
        "pixel_accuracy",
        "mean_class_accuracy",
        "validation_seconds",
        "provenance",
        "source_protocol_version",
        "source_protocol_sha256",
    ]

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
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fields
                }
            )


def trend_analysis(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_corruption: Dict[
        str,
        List[Mapping[str, Any]],
    ] = {}

    for row in rows:
        by_corruption.setdefault(
            str(
                row["corruption"]
            ),
            [],
        ).append(row)

    out: Dict[str, Any] = {}

    for corruption, group in by_corruption.items():
        group = sorted(
            group,
            key=lambda x: int(
                x["severity_rank"]
            ),
        )

        mious = [
            float(
                x["miou"]
            )
            for x in group
        ]
        drops = [
            float(
                x["drop_miou"]
            )
            for x in group
        ]

        out[corruption] = {
            "levels": [
                str(
                    x["severity_level"]
                )
                for x in group
            ],
            "miou": mious,
            "drop_miou": drops,
            "monotonic_nonincreasing_miou": all(
                mious[i + 1]
                <= mious[i] + 1e-12
                for i in range(
                    len(mious) - 1
                )
            ),
            "monotonic_nondecreasing_drop": all(
                drops[i + 1] + 1e-12
                >= drops[i]
                for i in range(
                    len(drops) - 1
                )
            ),
        }

    return out


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )
    clean_path = resolve_path(
        args.clean_metrics
    )
    v1_dir = resolve_path(
        args.v1_dir
    )
    output_dir = resolve_path(
        args.output_dir
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    clean = read_json(
        clean_path
    )
    validate_clean_reference(
        clean,
        checkpoint_path=checkpoint_path,
    )

    clean_miou = float(
        clean["miou"]
    )
    clean_class_map = (
        clean_per_class_map(
            clean
        )
    )

    v1_protocol_path = (
        v1_dir
        / "degradation_protocol.json"
    )
    v1_summary_path = (
        v1_dir
        / "robustness_summary.json"
    )

    print("=" * 92)
    print(
        "MODEL A | RGB DEGRADATION PROTOCOL v1 -> v2 UPGRADE"
    )
    print("=" * 92)
    print(
        f"Clean mIoU      : {clean_miou:.9f}"
    )
    print(
        f"v1 protocol     : {v1_protocol_path}"
    )
    print(
        f"v1 summary      : {v1_summary_path}"
    )
    print(
        f"v2 output       : {output_dir}"
    )
    print(
        f"v2 hash         : {degradation_protocol_sha256()}"
    )
    print("=" * 92)

    print("[1] validating v1 -> v2 backward compatibility")

    v1_protocol = read_json(
        v1_protocol_path
    )
    validate_v1_corruption_compatibility(
        v1_protocol
    )

    v1_summary = read_json(
        v1_summary_path
    )
    carried_results = validate_v1_summary(
        v1_summary,
        clean_miou=clean_miou,
    )

    print(
        "[compatibility] PASS | "
        "Noise/Blur definitions unchanged | "
        "6 v1 results approved for carry-forward"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_degradation_protocol(
        output_dir
        / "degradation_protocol.json"
    )

    print("[2] building Model A")
    model, model_meta = build_model_a_rgb(
        PROJECT_ROOT
    )

    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, checkpoint_meta = (
        unwrap_checkpoint_state_dict(
            checkpoint_obj
        )
    )
    validate_checkpoint_metadata(
        checkpoint_meta
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    device = get_device(
        args.device
    )
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    model.to(device)
    model.eval()

    checkpoint_epoch = (
        checkpoint_meta.get(
            "epoch"
        )
    )
    checkpoint_global_step = (
        checkpoint_meta.get(
            "global_step",
            checkpoint_meta.get(
                "step"
            ),
        )
    )

    print(
        f"[checkpoint] epoch_zero_based={checkpoint_epoch} | "
        f"global_step={checkpoint_global_step} | "
        f"device={device} | AMP={amp_enabled}"
    )

    print(
        "[3] evaluating NEW v2 condition family: RGB underexposure"
    )

    exposure_results: List[
        Dict[str, Any]
    ] = []

    for idx, level in enumerate(
        NEW_LEVELS,
        start=1,
    ):
        cond = condition_name(
            NEW_CORRUPTION,
            level,
        )

        print()
        print(
            f"[exposure {idx}/3] {cond}"
        )

        metrics_path = (
            output_dir
            / cond
            / "metrics.json"
        )

        existing = None

        if not args.force:
            existing = (
                load_existing_v2_exposure(
                    metrics_path,
                    level=level,
                    checkpoint_path=checkpoint_path,
                    clean_miou=clean_miou,
                )
            )

        if existing is not None:
            print(
                "[resume] compatible v2 exposure result exists; "
                f"skipping: {metrics_path}"
            )
            result = existing
        else:
            result = evaluate_exposure_condition(
                model=model,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=(
                    int(
                        checkpoint_epoch
                    )
                    if checkpoint_epoch
                    is not None
                    else None
                ),
                checkpoint_global_step=(
                    int(
                        checkpoint_global_step
                    )
                    if checkpoint_global_step
                    is not None
                    else None
                ),
                clean=clean,
                clean_class_map=clean_class_map,
                level=level,
                output_dir=output_dir,
                batch_size=args.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                log_every=args.log_every,
                confusion_chunk_rows=(
                    args.confusion_chunk_rows
                ),
                save_predictions=(
                    args.save_predictions
                ),
            )

        exposure_results.append(
            result
        )

    # ------------------------------------------------------------------
    # Build the complete v2 Model A summary.
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []

    for row in carried_results:
        # v1 summary uses clean_miou instead of clean_reference_miou.
        normalized = dict(row)
        normalized[
            "clean_reference_miou"
        ] = float(
            normalized[
                "clean_miou"
            ]
        )
        all_results.append(
            normalized
        )

    all_results.extend(
        exposure_results
    )

    corruption_order = {
        "gaussian_noise": 0,
        "gaussian_blur": 1,
        "rgb_underexposure": 2,
    }

    all_results.sort(
        key=lambda x: (
            corruption_order[
                str(
                    x["corruption"]
                )
            ],
            int(
                x["severity_rank"]
            ),
        )
    )

    summary_rows = [
        summary_row(
            result
        )
        for result in all_results
    ]

    trends = trend_analysis(
        all_results
    )

    summary = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "split": "val",
        "clean_reference": {
            "metrics_path": str(
                clean_path
            ),
            "miou": clean_miou,
            "pixel_accuracy": clean.get(
                "pixel_accuracy"
            ),
            "mean_class_accuracy": clean.get(
                "mean_class_accuracy"
            ),
            "checkpoint_global_step": clean.get(
                "checkpoint_global_step"
            ),
        },
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL_VERSION
        ),
        "degradation_protocol_sha256": (
            degradation_protocol_sha256()
        ),
        "upgrade_provenance": {
            "legacy_protocol_version": (
                "RGB Degradation Protocol v1"
            ),
            "legacy_protocol_sha256": (
                LEGACY_V1_PROTOCOL_SHA256
            ),
            "legacy_conditions_carried_forward": list(
                LEGACY_CONDITIONS
            ),
            "reason": (
                "v2 is a strict extension: Gaussian Noise and Gaussian Blur "
                "definitions, scope and seed are unchanged."
            ),
            "new_conditions_evaluated": [
                condition_name(
                    NEW_CORRUPTION,
                    level,
                )
                for level in NEW_LEVELS
            ],
        },
        "results": summary_rows,
        "trend_analysis": trends,
    }

    write_json(
        output_dir
        / "robustness_summary.json",
        summary,
    )

    write_summary_csv(
        output_dir
        / "robustness_summary.csv",
        summary_rows,
    )

    print()
    print("=" * 100)
    print("MODEL A ROBUSTNESS SUMMARY | PROTOCOL v2")
    print("=" * 100)
    print(
        f"{'Condition':<28} "
        f"{'mIoU':>10} "
        f"{'Drop':>10} "
        f"{'Rel.Drop':>10} "
        f"{'Retention':>10} "
        f"{'Source':>12}"
    )
    print("-" * 100)

    for row in summary_rows:
        source = (
            "v1 carried"
            if str(
                row["provenance"]
            ).startswith(
                "carried_forward"
            )
            else "v2 new"
        )

        print(
            f"{row['condition']:<28} "
            f"{float(row['miou']):>10.6f} "
            f"{float(row['drop_miou']):>10.6f} "
            f"{float(row['relative_drop_pct']):>9.2f}% "
            f"{float(row['retention_pct']):>9.2f}% "
            f"{source:>12}"
        )

    print("-" * 100)

    for corruption, trend in trends.items():
        print(
            f"{corruption}: "
            f"mIoU severity-monotonic="
            f"{trend['monotonic_nonincreasing_miou']} | "
            f"Drop severity-monotonic="
            f"{trend['monotonic_nondecreasing_drop']}"
        )

    print("-" * 100)
    print(
        f"summary JSON: "
        f"{output_dir / 'robustness_summary.json'}"
    )
    print(
        f"summary CSV : "
        f"{output_dir / 'robustness_summary.csv'}"
    )
    print(
        f"protocol    : "
        f"{output_dir / 'degradation_protocol.json'}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
