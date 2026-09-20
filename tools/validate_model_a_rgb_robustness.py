#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A RGB robustness validation: Gaussian Noise / Gaussian Blur, L1-L3.

Prerequisites
-------------
1. Model A has finished training.
2. Clean validation has already been run with:
       tools/validate_model_a_rgb.py
3. Place:
       evaluation/rgb_degradation_protocol.py
       tools/validate_model_a_rgb_robustness.py
   in the repository.

Default command
---------------
    python tools/validate_model_a_rgb_robustness.py

This evaluates:
    gaussian_noise L1, L2, L3
    gaussian_blur  L1, L2, L3

using the same:
    - Model A final checkpoint
    - 6 validation tiles
    - 256 windows/tile
    - full-tile mean-logit fusion
    - global confusion-matrix mIoU

The existing Clean result is used as the reference for:
    Drop_mIoU  = Clean_mIoU - Degraded_mIoU
    Delta_mIoU = Degraded_mIoU - Clean_mIoU
    RelativeDropPct = Drop_mIoU / Clean_mIoU * 100

Outputs
-------
outputs/evaluation/model_a_rgb/robustness_val/
    degradation_protocol.json
    robustness_summary.json
    robustness_summary.csv
    gaussian_noise_L1/
        metrics.json
        per_class_metrics.csv
        confusion_matrix.csv
        per_tile_metrics.jsonl
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Running a script from tools/ already adds tools/ to sys.path. This fallback
# also supports module-style execution from unusual launch contexts.
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL,
    DegradedPotsdamSlidingWindowDataset,
    available_corruptions,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    write_degradation_protocol,
)
from models.segformer_rgb import build_model_a_rgb

# Reuse the already-validated Clean evaluation implementation so metric/fusion
# code is identical between Clean and degraded conditions.
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
        "Keep the Clean validation script from the previous step in tools/ "
        "before running this robustness script."
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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Model A under frozen RGB degradation L1/L2/L3."
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
        help="Existing Clean validation metrics.json used as Drop reference.",
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
        "--corruptions",
        nargs="+",
        default=list(available_corruptions()),
        choices=list(available_corruptions()),
        help="Subset of frozen corruption families to evaluate.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["L1", "L2", "L3"],
        choices=["L1", "L2", "L3"],
        help="Subset of severity levels to evaluate.",
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
        help="Recompute conditions even if compatible metrics.json exists.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")
    if args.confusion_chunk_rows <= 0:
        parser.error("--confusion-chunk-rows must be > 0")

    # Remove duplicates while preserving requested order.
    args.corruptions = list(dict.fromkeys(args.corruptions))
    args.levels = list(dict.fromkeys(args.levels))

    return args


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def get_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    return device


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ) + "\n"
        )


def validate_clean_reference(
    clean: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if clean.get("model") != "A_RGB":
        raise RuntimeError(
            f"Clean reference is not Model A: {clean.get('model')!r}"
        )
    if clean.get("condition") != "Clean":
        raise RuntimeError(
            f"Reference condition is not Clean: "
            f"{clean.get('condition')!r}"
        )
    if clean.get("split") != "val":
        raise RuntimeError(
            f"Clean reference split must be val: "
            f"{clean.get('split')!r}"
        )

    metric_scope = str(clean.get("metric_scope", ""))
    if "GLOBAL confusion matrix" not in metric_scope:
        raise RuntimeError(
            "Clean reference does not appear to use the required GLOBAL "
            "full-tile metric scope."
        )

    clean_miou = clean.get("miou")
    if not isinstance(clean_miou, (int, float)):
        raise RuntimeError("Clean reference has no numeric mIoU.")

    if not (0.0 <= float(clean_miou) <= 1.0):
        raise RuntimeError(
            f"Clean reference mIoU is invalid: {clean_miou}"
        )

    clean_checkpoint = clean.get("checkpoint")
    if clean_checkpoint:
        if Path(str(clean_checkpoint)).name != checkpoint_path.name:
            raise RuntimeError(
                "Clean reference checkpoint filename differs from the "
                "requested degraded-evaluation checkpoint: "
                f"{clean_checkpoint} vs {checkpoint_path}"
            )


def clean_per_class_map(
    clean: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    rows = clean.get("per_class")
    if not isinstance(rows, list):
        raise RuntimeError("Clean metrics missing per_class list.")

    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = int(row["class_id"])
        out[cid] = dict(row)

    if set(out) != set(range(NUM_CLASSES)):
        raise RuntimeError(
            f"Clean per-class ids are invalid: {sorted(out)}"
        )

    return out


def validate_degradation_probe(
    *,
    clean_dataset: PotsdamSlidingWindowDataset,
    degraded_dataset: DegradedPotsdamSlidingWindowDataset,
) -> None:
    """
    Prove on the first validation window that:
      - window coordinates are identical
      - NIR is exactly unchanged
      - RGB is actually changed
    """
    clean_item = clean_dataset[0]
    degraded_item = degraded_dataset[0]

    for key in ("tile_id", "window_index", "x", "y", "height", "width"):
        if clean_item[key] != degraded_item[key]:
            raise RuntimeError(
                f"Degradation changed frozen window metadata {key}: "
                f"{clean_item[key]!r} != {degraded_item[key]!r}"
            )

    if not torch.equal(clean_item["nir"], degraded_item["nir"]):
        max_diff = float(
            (clean_item["nir"] - degraded_item["nir"])
            .abs()
            .max()
            .item()
        )
        raise RuntimeError(
            "NIR changed under an RGB-only degradation. "
            f"max_abs_diff={max_diff}"
        )

    if torch.equal(clean_item["rgb"], degraded_item["rgb"]):
        raise RuntimeError(
            "Degraded RGB is exactly equal to Clean RGB on the probe window; "
            "corruption implementation may not be active."
        )

    rgb_mean_abs_change = float(
        (clean_item["rgb"] - degraded_item["rgb"])
        .abs()
        .mean()
        .item()
    )

    print(
        "[probe] PASS | same window metadata | NIR bit-identical | "
        f"RGB mean_abs_change(normalized)={rgb_mean_abs_change:.6f}"
    )


def evaluate_condition(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_global_step: Optional[int],
    clean: Mapping[str, Any],
    clean_class_map: Mapping[int, Mapping[str, Any]],
    corruption: str,
    level: str,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    log_every: int,
    confusion_chunk_rows: int,
    save_predictions: bool,
) -> Dict[str, Any]:
    cond = condition_name(corruption, level)
    cond_dir = output_dir / cond
    cond_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(f"CONDITION | {cond}")
    print("=" * 92)
    print(f"parameters: {condition_spec(corruption, level)}")

    dataset = DegradedPotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
        corruption=corruption,
        level=level,
    )

    clean_probe_dataset = PotsdamSlidingWindowDataset(
        PROJECT_ROOT,
        split="val",
    )

    validate_degradation_probe(
        clean_dataset=clean_probe_dataset,
        degraded_dataset=dataset,
    )

    # Release the separate Clean probe TIFF cache before full evaluation.
    del clean_probe_dataset

    tile_ids = list(dataset.tile_ids)
    windows_per_tile = len(dataset.window_coordinates)

    if len(dataset) != len(tile_ids) * windows_per_tile:
        raise RuntimeError("Degraded dataset length is inconsistent.")
    if len(tile_ids) != 6 or windows_per_tile != 256:
        raise RuntimeError(
            "Frozen val protocol changed unexpectedly: "
            f"tiles={len(tile_ids)}, windows/tile={windows_per_tile}"
        )

    per_tile_path = cond_dir / "per_tile_metrics.jsonl"
    if per_tile_path.exists():
        per_tile_path.unlink()

    pred_dir = cond_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    global_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    per_tile_results: List[Dict[str, Any]] = []
    condition_start = time.time()

    for tile_index, tile_id in enumerate(tile_ids):
        print(
            f"[{cond}] tile {tile_index + 1}/{len(tile_ids)} | {tile_id}",
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

        target_t = dataset.load_full_label(tile_id)
        target = target_t.numpy()

        tile_confusion = confusion_from_prediction(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            chunk_rows=confusion_chunk_rows,
        )
        global_confusion += tile_confusion

        tile_metrics = metrics_from_confusion(
            tile_confusion,
            CLASS_NAMES,
        )

        tile_record = {
            **tile_runtime,
            "condition": cond,
            "corruption": corruption,
            "severity_level": level,
            "miou": tile_metrics["miou"],
            "pixel_accuracy": tile_metrics["pixel_accuracy"],
            "mean_class_accuracy": tile_metrics["mean_class_accuracy"],
            "valid_pixels": tile_metrics["valid_pixels"],
            "per_class_iou": {
                row["class_name"]: row["iou"]
                for row in tile_metrics["per_class"]
            },
        }
        per_tile_results.append(tile_record)
        append_jsonl(per_tile_path, tile_record)

        if save_predictions:
            np.save(
                pred_dir / f"{tile_id}_pred.npy",
                prediction,
                allow_pickle=False,
            )

        print(
            f"  tile mIoU={tile_metrics['miou']:.6f} | "
            f"pixel_acc={tile_metrics['pixel_accuracy']:.6f} | "
            f"time={tile_runtime['inference_seconds']:.1f}s"
        )

        del prediction, target, target_t

        # Explicitly release the previous full degraded tile before the next
        # tile is created.
        dataset.clear_degradation_cache()

    global_metrics = metrics_from_confusion(
        global_confusion,
        CLASS_NAMES,
    )

    clean_miou = float(clean["miou"])
    degraded_miou = float(global_metrics["miou"])

    drop_miou = clean_miou - degraded_miou
    delta_miou = degraded_miou - clean_miou
    relative_drop_pct = (
        100.0 * drop_miou / clean_miou
        if clean_miou != 0.0
        else float("nan")
    )
    retention_pct = (
        100.0 * degraded_miou / clean_miou
        if clean_miou != 0.0
        else float("nan")
    )

    per_class: List[Dict[str, Any]] = []
    for row in global_metrics["per_class"]:
        cid = int(row["class_id"])
        clean_row = clean_class_map[cid]
        clean_iou = float(clean_row["iou"])
        degraded_iou = float(row["iou"])

        enriched = dict(row)
        enriched["clean_iou"] = clean_iou
        enriched["drop_iou"] = clean_iou - degraded_iou
        enriched["delta_iou"] = degraded_iou - clean_iou
        per_class.append(enriched)

    elapsed = time.time() - condition_start

    result: Dict[str, Any] = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "condition": cond,
        "corruption": corruption,
        "severity_level": level,
        "severity_rank": int(level[1:]),
        "severity_parameters": condition_spec(corruption, level),
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL["protocol_version"]
        ),
        "degradation_protocol_sha256": degradation_protocol_sha256(),
        "rgb_degraded": True,
        "nir_used_by_model": False,
        "nir_source": "clean / unchanged",
        "split": "val",
        "metric_scope": (
            "GLOBAL confusion matrix after full-tile mean-logit fusion"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "clean_reference_miou": clean_miou,
        "miou": degraded_miou,
        "drop_miou": drop_miou,
        "delta_miou": delta_miou,
        "relative_drop_pct": relative_drop_pct,
        "retention_pct": retention_pct,
        "pixel_accuracy": global_metrics["pixel_accuracy"],
        "mean_class_accuracy": global_metrics["mean_class_accuracy"],
        "valid_pixels": global_metrics["valid_pixels"],
        "per_class": per_class,
        "confusion_matrix": global_confusion.tolist(),
        "tiles": per_tile_results,
        "num_tiles": len(tile_ids),
        "windows_per_tile": windows_per_tile,
        "total_windows": len(dataset),
        "validation_seconds": elapsed,
    }

    write_json(cond_dir / "metrics.json", result)
    # Keep the standard per-class metric table for compatibility with Clean,
    # and add an explicit robustness table with Clean-vs-degraded deltas.
    write_per_class_csv(
        cond_dir / "per_class_metrics.csv",
        per_class,
    )
    write_per_class_robustness_csv(
        cond_dir / "per_class_robustness.csv",
        per_class,
    )
    write_confusion_csv(
        cond_dir / "confusion_matrix.csv",
        global_confusion,
        CLASS_NAMES,
    )

    print("-" * 92)
    print(
        f"{cond} | mIoU={degraded_miou:.6f} | "
        f"Drop={drop_miou:.6f} | "
        f"RelativeDrop={relative_drop_pct:.2f}% | "
        f"Retention={retention_pct:.2f}%"
    )
    print("-" * 92)

    return result


def load_compatible_existing_result(
    path: Path,
    *,
    corruption: str,
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

    expected_condition = condition_name(corruption, level)

    checks = [
        obj.get("model") == "A_RGB",
        obj.get("condition") == expected_condition,
        obj.get("corruption") == corruption,
        obj.get("severity_level") == level,
        obj.get("split") == "val",
        obj.get("degradation_protocol_sha256")
        == degradation_protocol_sha256(),
        Path(str(obj.get("checkpoint", ""))).name
        == checkpoint_path.name,
        math.isclose(
            float(obj.get("clean_reference_miou", float("nan"))),
            clean_miou,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    ]

    if all(checks):
        return obj

    return None


def summary_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": result["model"],
        "condition": result["condition"],
        "corruption": result["corruption"],
        "severity_level": result["severity_level"],
        "severity_rank": result["severity_rank"],
        "clean_miou": result["clean_reference_miou"],
        "miou": result["miou"],
        "drop_miou": result["drop_miou"],
        "delta_miou": result["delta_miou"],
        "relative_drop_pct": result["relative_drop_pct"],
        "retention_pct": result["retention_pct"],
        "pixel_accuracy": result["pixel_accuracy"],
        "mean_class_accuracy": result["mean_class_accuracy"],
        "validation_seconds": result["validation_seconds"],
    }


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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in fields}
            )


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
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in fields}
            )


def trend_analysis(
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_corruption: Dict[str, List[Mapping[str, Any]]] = {}

    for result in results:
        by_corruption.setdefault(
            str(result["corruption"]),
            [],
        ).append(result)

    out: Dict[str, Any] = {}

    for corruption, rows in by_corruption.items():
        rows = sorted(
            rows,
            key=lambda x: int(x["severity_rank"]),
        )

        levels = [str(x["severity_level"]) for x in rows]
        mious = [float(x["miou"]) for x in rows]
        drops = [float(x["drop_miou"]) for x in rows]

        # For a full L1/L2/L3 run, this directly checks whether degradation
        # gets progressively worse. For a subset run, it checks the requested
        # ordered subset only.
        monotonic_nonincreasing_miou = all(
            mious[i + 1] <= mious[i] + 1e-12
            for i in range(len(mious) - 1)
        )
        monotonic_nondecreasing_drop = all(
            drops[i + 1] + 1e-12 >= drops[i]
            for i in range(len(drops) - 1)
        )

        out[corruption] = {
            "levels": levels,
            "miou": mious,
            "drop_miou": drops,
            "monotonic_nonincreasing_miou": (
                monotonic_nonincreasing_miou
            ),
            "monotonic_nondecreasing_drop": (
                monotonic_nondecreasing_drop
            ),
        }

    return out


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    clean_metrics_path = resolve_path(args.clean_metrics)
    output_dir = resolve_path(args.output_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not clean_metrics_path.is_file():
        raise FileNotFoundError(
            "Clean reference metrics not found. Run "
            "`python tools/validate_model_a_rgb.py` first.\n"
            f"Expected: {clean_metrics_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_degradation_protocol(
        output_dir / "degradation_protocol.json"
    )

    clean = read_json(clean_metrics_path)
    validate_clean_reference(
        clean,
        checkpoint_path=checkpoint_path,
    )
    clean_class_map = clean_per_class_map(clean)
    clean_miou = float(clean["miou"])

    device = get_device(args.device)
    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 92)
    print("MODEL A RGB ROBUSTNESS VALIDATION")
    print("=" * 92)
    print(f"checkpoint        : {checkpoint_path}")
    print(f"clean metrics     : {clean_metrics_path}")
    print(f"clean mIoU        : {clean_miou:.9f}")
    print(f"output dir        : {output_dir}")
    print(f"protocol          : {DEGRADATION_PROTOCOL['protocol_version']}")
    print(
        f"protocol sha256   : {degradation_protocol_sha256()}"
    )
    print(f"corruptions       : {args.corruptions}")
    print(f"levels            : {args.levels}")
    print(f"device / AMP      : {device} / {amp_enabled}")
    print(f"batch size        : {args.batch_size}")
    print("DataLoader workers: 0 (required for full-tile degradation cache)")
    print("=" * 92)

    print("[1] building Model A")
    model, model_meta = build_model_a_rgb(PROJECT_ROOT)

    print("[2] loading trained checkpoint")
    checkpoint_obj = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, checkpoint_meta = unwrap_checkpoint_state_dict(
        checkpoint_obj
    )
    validate_checkpoint_metadata(checkpoint_meta)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    checkpoint_epoch = checkpoint_meta.get("epoch")
    checkpoint_global_step = checkpoint_meta.get(
        "global_step",
        checkpoint_meta.get("step"),
    )

    print(
        f"[checkpoint] epoch_zero_based={checkpoint_epoch} | "
        f"global_step={checkpoint_global_step}"
    )

    all_results: List[Dict[str, Any]] = []

    total_conditions = len(args.corruptions) * len(args.levels)
    condition_counter = 0

    for corruption in args.corruptions:
        for level in args.levels:
            condition_counter += 1
            cond = condition_name(corruption, level)
            metrics_path = output_dir / cond / "metrics.json"

            print()
            print(
                f"[condition {condition_counter}/{total_conditions}] {cond}"
            )

            existing = None
            if not args.force:
                existing = load_compatible_existing_result(
                    metrics_path,
                    corruption=corruption,
                    level=level,
                    checkpoint_path=checkpoint_path,
                    clean_miou=clean_miou,
                )

            if existing is not None:
                print(
                    "[resume] compatible metrics already exist; "
                    f"skipping inference: {metrics_path}"
                )
                result = existing
            else:
                result = evaluate_condition(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    checkpoint_epoch=(
                        int(checkpoint_epoch)
                        if checkpoint_epoch is not None
                        else None
                    ),
                    checkpoint_global_step=(
                        int(checkpoint_global_step)
                        if checkpoint_global_step is not None
                        else None
                    ),
                    clean=clean,
                    clean_class_map=clean_class_map,
                    corruption=corruption,
                    level=level,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    log_every=args.log_every,
                    confusion_chunk_rows=args.confusion_chunk_rows,
                    save_predictions=args.save_predictions,
                )

            all_results.append(result)

    # Stable scientific ordering, regardless of CLI order.
    all_results.sort(
        key=lambda x: (
            list(DEGRADATION_PROTOCOL["corruptions"].keys()).index(
                str(x["corruption"])
            ),
            int(x["severity_rank"]),
        )
    )

    rows = [summary_row(result) for result in all_results]
    trends = trend_analysis(all_results)

    summary = {
        "model": "A_RGB",
        "model_name": MODEL_NAME,
        "split": "val",
        "clean_reference": {
            "metrics_path": str(clean_metrics_path),
            "miou": clean_miou,
            "pixel_accuracy": clean.get("pixel_accuracy"),
            "mean_class_accuracy": clean.get("mean_class_accuracy"),
            "checkpoint_global_step": clean.get(
                "checkpoint_global_step"
            ),
        },
        "degradation_protocol_version": (
            DEGRADATION_PROTOCOL["protocol_version"]
        ),
        "degradation_protocol_sha256": degradation_protocol_sha256(),
        "results": rows,
        "trend_analysis": trends,
    }

    write_json(
        output_dir / "robustness_summary.json",
        summary,
    )
    write_summary_csv(
        output_dir / "robustness_summary.csv",
        rows,
    )

    print()
    print("=" * 92)
    print("MODEL A ROBUSTNESS SUMMARY")
    print("=" * 92)
    print(
        f"{'Condition':<24} {'mIoU':>10} {'Drop':>10} "
        f"{'Rel.Drop':>10} {'Retention':>10}"
    )
    print("-" * 92)

    for row in rows:
        print(
            f"{row['condition']:<24} "
            f"{float(row['miou']):>10.6f} "
            f"{float(row['drop_miou']):>10.6f} "
            f"{float(row['relative_drop_pct']):>9.2f}% "
            f"{float(row['retention_pct']):>9.2f}%"
        )

    print("-" * 92)

    for corruption, trend in trends.items():
        print(
            f"{corruption}: "
            f"mIoU severity-monotonic="
            f"{trend['monotonic_nonincreasing_miou']} | "
            f"Drop severity-monotonic="
            f"{trend['monotonic_nondecreasing_drop']}"
        )

    print("-" * 92)
    print(
        f"summary JSON: {output_dir / 'robustness_summary.json'}"
    )
    print(
        f"summary CSV : {output_dir / 'robustness_summary.csv'}"
    )
    print(
        f"protocol    : {output_dir / 'degradation_protocol.json'}"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
