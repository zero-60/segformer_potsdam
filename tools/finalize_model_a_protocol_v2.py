#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalize Model A's existing Protocol-v2 metadata after the Gaussian Noise
RNG-compatibility implementation fix.

NO inference is run.
NO metric value is changed.

Why this is safe
----------------
- Model A Noise/Blur results were actually generated under v1 and carried into
  v2. They already use the desired v1 Gaussian-noise realization.
- Model A Underexposure L1-L3 is deterministic and does not use an RNG.
- The implementation fix affects only how FUTURE models generate Gaussian
  Noise so they reproduce Model A's existing v1 noise exactly.

This script:
1. validates the existing pre-fix v2 summary/protocol,
2. backs them up,
3. writes the final v2 protocol metadata,
4. updates the summary protocol hash/revision,
5. updates the three exposure metrics JSON metadata only,
6. rewrites robustness_summary.csv with unchanged numbers.

Expected:
    evaluation/rgb_degradation_protocol.py  <- final v2 file
    tools/finalize_model_a_protocol_v2.py   <- this file
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL_VERSION,
    IMPLEMENTATION_REVISION,
    PRE_FIX_V2_PROTOCOL_SHA256,
    degradation_protocol_sha256,
    write_degradation_protocol,
)


A_V2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model_a_rgb"
    / "robustness_val_v2"
)

EXPOSURE_CONDITIONS = (
    "rgb_underexposure_L1",
    "rgb_underexposure_L2",
    "rgb_underexposure_L3",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def backup_once(path: Path) -> Path:
    backup = path.with_name(
        path.stem + ".pre_rngfix" + path.suffix
    )
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in fields}
            )


def main() -> None:
    protocol_path = A_V2_DIR / "degradation_protocol.json"
    summary_path = A_V2_DIR / "robustness_summary.json"
    summary_csv_path = A_V2_DIR / "robustness_summary.csv"

    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    protocol = read_json(protocol_path)
    summary = read_json(summary_path)

    current_hash = degradation_protocol_sha256()

    # Idempotent: already finalized.
    if (
        protocol.get("sha256") == current_hash
        and summary.get("degradation_protocol_sha256") == current_hash
        and int(protocol.get("implementation_revision", -1))
        == IMPLEMENTATION_REVISION
    ):
        print("[done] Model A v2 metadata is already finalized.")
        print(f"protocol hash: {current_hash}")
        return

    if protocol.get("sha256") != PRE_FIX_V2_PROTOCOL_SHA256:
        raise RuntimeError(
            "Unexpected existing v2 protocol hash.\n"
            f"expected pre-fix: {PRE_FIX_V2_PROTOCOL_SHA256}\n"
            f"found: {protocol.get('sha256')}"
        )

    if summary.get("degradation_protocol_sha256") != PRE_FIX_V2_PROTOCOL_SHA256:
        raise RuntimeError(
            "Existing Model A v2 summary hash is not the known pre-fix hash."
        )

    if summary.get("model") != "A_RGB":
        raise RuntimeError("Existing v2 summary is not Model A.")

    rows = summary.get("results")
    if not isinstance(rows, list) or len(rows) != 9:
        raise RuntimeError(
            "Expected exactly nine Model A degraded result rows."
        )

    conditions = {str(row.get("condition")) for row in rows}
    expected = {
        "gaussian_noise_L1",
        "gaussian_noise_L2",
        "gaussian_noise_L3",
        "gaussian_blur_L1",
        "gaussian_blur_L2",
        "gaussian_blur_L3",
        *EXPOSURE_CONDITIONS,
    }
    if conditions != expected:
        raise RuntimeError(
            f"Unexpected Model A v2 conditions: {sorted(conditions)}"
        )

    backup_once(protocol_path)
    backup_once(summary_path)
    if summary_csv_path.exists():
        backup_once(summary_csv_path)

    # Update the three deterministic exposure condition metadata files.
    for condition in EXPOSURE_CONDITIONS:
        metrics_path = A_V2_DIR / condition / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)

        metrics = read_json(metrics_path)

        if metrics.get("degradation_protocol_sha256") not in (
            PRE_FIX_V2_PROTOCOL_SHA256,
            current_hash,
        ):
            raise RuntimeError(
                f"Unexpected protocol hash in {metrics_path}"
            )

        if metrics.get("corruption") != "rgb_underexposure":
            raise RuntimeError(
                f"Unexpected corruption in {metrics_path}"
            )

        backup_once(metrics_path)

        metrics["degradation_protocol_version"] = DEGRADATION_PROTOCOL_VERSION
        metrics["degradation_protocol_sha256"] = current_hash
        metrics["degradation_implementation_revision"] = IMPLEMENTATION_REVISION
        metrics["metadata_migration"] = {
            "previous_protocol_sha256": PRE_FIX_V2_PROTOCOL_SHA256,
            "metric_values_changed": False,
            "reason": (
                "Final v2 revision fixes only future Gaussian Noise RNG "
                "namespace compatibility; deterministic underexposure pixels "
                "and all resulting metrics are unchanged."
            ),
        }

        write_json(metrics_path, metrics)

    # Update summary rows without changing numeric metrics.
    new_rows = []
    for row in rows:
        row = dict(row)
        condition = str(row["condition"])

        if condition.startswith("rgb_underexposure_"):
            row["source_protocol_version"] = DEGRADATION_PROTOCOL_VERSION
            row["source_protocol_sha256"] = current_hash
            row["provenance"] = (
                "evaluated_under_v2; metadata finalized after RNG "
                "compatibility fix; numeric metrics unchanged"
            )

        new_rows.append(row)

    summary["degradation_protocol_version"] = DEGRADATION_PROTOCOL_VERSION
    summary["degradation_protocol_sha256"] = current_hash
    summary["degradation_implementation_revision"] = IMPLEMENTATION_REVISION
    summary["rng_compatibility_fix"] = {
        "previous_protocol_sha256": PRE_FIX_V2_PROTOCOL_SHA256,
        "new_protocol_sha256": current_hash,
        "metric_values_changed": False,
        "gaussian_noise_policy": (
            "Future models use protocol-v1 RNG namespace, reproducing the "
            "exact noise realization already used by Model A."
        ),
    }
    summary["results"] = new_rows

    write_json(summary_path, summary)
    write_summary_csv(summary_csv_path, new_rows)
    write_degradation_protocol(protocol_path)

    print("=" * 88)
    print("MODEL A PROTOCOL v2 METADATA FINALIZED")
    print("=" * 88)
    print("Inference rerun       : NO")
    print("Metric values changed : NO")
    print(f"Implementation rev    : {IMPLEMENTATION_REVISION}")
    print(f"Final protocol hash   : {current_hash}")
    print(f"Summary               : {summary_path}")
    print(f"Protocol              : {protocol_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
