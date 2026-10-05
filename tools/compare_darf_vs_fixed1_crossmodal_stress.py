#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Fixed g=1.0 against M3' DARF on the Cross-Modal Stress Protocol.

This script does NOT invent a significance threshold. It reports:
- condition-wise mIoU / per-class deltas
- suite-level mean deltas
- DARF gate changes relative to Clean
- directional mechanism checks:
    * NIR-only: does gate decrease from L1 -> L2 -> L3?
    * NIR dropout: is gate below Clean?
    * severity mismatch: is gate higher when NIR is relatively better?
    * misregistration: does gate decrease as shift grows?

Run after BOTH validators completed with --suite all.

Example
-------
python tools/compare_darf_vs_fixed1_crossmodal_stress.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIXED_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_rgbnir_fixed1_joint_robust4"
    / "crossmodal_stress"
)

DEFAULT_DARF_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "b2_darf_joint_robust4"
    / "crossmodal_stress"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "crossmodal_stress_darf_vs_fixed1"
)


def resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(
    path: Path,
    obj: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def index_results(
    payload: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["condition"]): row
        for row in payload.get(
            "results",
            []
        )
    }


def index_gates(
    payload: Mapping[str, Any],
) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    return {
        (
            str(row["condition"]),
            int(row["scale"]),
        ): row
        for row in payload.get(
            "rows",
            []
        )
    }


def mean_or_none(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None
    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def compare_condition_rows(
    fixed: Mapping[str, Any],
    darf: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "condition": fixed["condition"],
        "stress_axis": fixed["stress_axis"],
        "family": fixed["family"],
        "severity_level": fixed.get("severity_level"),
        "mismatch_role": fixed.get("mismatch_role"),
        "rgb_level": fixed.get("rgb_level"),
        "nir_level": fixed.get("nir_level"),
        "shift_pixels": fixed.get("shift_pixels"),
        "fixed1_miou": float(fixed["miou"]),
        "darf_miou": float(darf["miou"]),
    }

    row["darf_minus_fixed1_miou"] = (
        row["darf_miou"]
        - row["fixed1_miou"]
    )

    for key in sorted(
        set(fixed.keys())
        & set(darf.keys())
    ):
        if not key.startswith("iou_"):
            continue

        a = fixed.get(key)
        b = darf.get(key)

        if a is None or b is None:
            continue

        row[f"fixed1_{key}"] = float(a)
        row[f"darf_{key}"] = float(b)
        row[
            f"darf_minus_fixed1_{key}"
        ] = float(b) - float(a)

    return row


def gate_value(
    gates: Mapping[
        Tuple[str, int],
        Mapping[str, Any],
    ],
    condition: str,
    scale: int,
) -> float:
    return float(
        gates[(condition, scale)]["g_nir_mean"]
    )


def monotonic_nonincreasing(
    values: Sequence[float],
) -> bool:
    return all(
        values[i] >= values[i + 1]
        for i in range(
            len(values) - 1
        )
    )


def mechanism_report(
    gates: Mapping[
        Tuple[str, int],
        Mapping[str, Any],
    ],
) -> Dict[str, Any]:
    clean = {
        scale: gate_value(
            gates,
            "Clean",
            scale,
        )
        for scale in range(1, 5)
    }

    report: Dict[str, Any] = {
        "clean_gate_by_scale": clean,
    }

    nir_only_rows = []

    for family in (
        "gaussian_noise",
        "gaussian_blur",
        "underexposure",
        "fog",
    ):
        for scale in range(1, 5):
            values = [
                gate_value(
                    gates,
                    f"nir_only_{family}_{level}",
                    scale,
                )
                for level in (
                    "L1",
                    "L2",
                    "L3",
                )
            ]

            nir_only_rows.append(
                {
                    "family": family,
                    "scale": scale,
                    "g_L1": values[0],
                    "g_L2": values[1],
                    "g_L3": values[2],
                    "delta_L3_vs_clean": (
                        values[2]
                        - clean[scale]
                    ),
                    "monotonic_nonincreasing_L1_L2_L3": (
                        monotonic_nonincreasing(values)
                    ),
                }
            )

    report["nir_only"] = {
        "rows": nir_only_rows,
        "monotonic_pairs_passed": sum(
            bool(
                row[
                    "monotonic_nonincreasing_L1_L2_L3"
                ]
            )
            for row in nir_only_rows
        ),
        "monotonic_pairs_total": len(
            nir_only_rows
        ),
    }

    dropout_rows = []

    for scale in range(1, 5):
        g = gate_value(
            gates,
            "nir_dropout_full",
            scale,
        )

        dropout_rows.append(
            {
                "scale": scale,
                "g_clean": clean[scale],
                "g_dropout": g,
                "delta_dropout_vs_clean": (
                    g - clean[scale]
                ),
                "gate_below_clean": (
                    g < clean[scale]
                ),
            }
        )

    report["nir_dropout"] = {
        "rows": dropout_rows,
        "scales_below_clean": sum(
            bool(row["gate_below_clean"])
            for row in dropout_rows
        ),
        "scales_total": 4,
    }

    mismatch_rows = []

    for family in (
        "gaussian_noise",
        "gaussian_blur",
        "underexposure",
        "fog",
    ):
        nir_worse = (
            f"mismatch_{family}_rgbL1_nirL3"
        )
        rgb_worse = (
            f"mismatch_{family}_rgbL3_nirL1"
        )

        for scale in range(1, 5):
            g_nir_worse = gate_value(
                gates,
                nir_worse,
                scale,
            )
            g_rgb_worse = gate_value(
                gates,
                rgb_worse,
                scale,
            )

            mismatch_rows.append(
                {
                    "family": family,
                    "scale": scale,
                    "g_rgbL1_nirL3": g_nir_worse,
                    "g_rgbL3_nirL1": g_rgb_worse,
                    "expected_direction_delta": (
                        g_rgb_worse
                        - g_nir_worse
                    ),
                    "gate_higher_when_nir_relatively_better": (
                        g_rgb_worse
                        > g_nir_worse
                    ),
                }
            )

    report["severity_mismatch"] = {
        "rows": mismatch_rows,
        "direction_checks_passed": sum(
            bool(
                row[
                    "gate_higher_when_nir_relatively_better"
                ]
            )
            for row in mismatch_rows
        ),
        "direction_checks_total": len(
            mismatch_rows
        ),
    }

    misreg_rows = []

    for scale in range(1, 5):
        values = [
            gate_value(
                gates,
                f"misregistration_{level}",
                scale,
            )
            for level in (
                "L1",
                "L2",
                "L3",
            )
        ]

        misreg_rows.append(
            {
                "scale": scale,
                "g_L1": values[0],
                "g_L2": values[1],
                "g_L3": values[2],
                "delta_L3_vs_clean": (
                    values[2]
                    - clean[scale]
                ),
                "monotonic_nonincreasing_L1_L2_L3": (
                    monotonic_nonincreasing(values)
                ),
            }
        )

    report["misregistration"] = {
        "rows": misreg_rows,
        "monotonic_scales_passed": sum(
            bool(
                row[
                    "monotonic_nonincreasing_L1_L2_L3"
                ]
            )
            for row in misreg_rows
        ),
        "monotonic_scales_total": 4,
    }

    return report


def aggregate_performance(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    nonclean = [
        row
        for row in rows
        if row["condition"] != "Clean"
    ]

    by_axis: Dict[str, Any] = {}

    for axis in (
        "nir_only",
        "nir_dropout",
        "severity_mismatch",
        "misregistration",
    ):
        values = [
            float(
                row[
                    "darf_minus_fixed1_miou"
                ]
            )
            for row in nonclean
            if row["stress_axis"] == axis
        ]

        by_axis[axis] = {
            "num_conditions": len(values),
            "mean_darf_minus_fixed1_miou": (
                mean_or_none(values)
            ),
            "darf_wins": sum(
                value > 0
                for value in values
            ),
            "fixed1_wins": sum(
                value < 0
                for value in values
            ),
            "ties_exact": sum(
                value == 0
                for value in values
            ),
        }

    for role in (
        "nir_worse",
        "rgb_worse",
    ):
        values = [
            float(
                row[
                    "darf_minus_fixed1_miou"
                ]
            )
            for row in nonclean
            if (
                row["stress_axis"]
                == "severity_mismatch"
                and row.get("mismatch_role")
                == role
            )
        ]

        by_axis[
            f"severity_mismatch_{role}"
        ] = {
            "num_conditions": len(values),
            "mean_darf_minus_fixed1_miou": (
                mean_or_none(values)
            ),
        }

    all_values = [
        float(
            row["darf_minus_fixed1_miou"]
        )
        for row in nonclean
    ]

    return {
        "all_nonclean": {
            "num_conditions": len(all_values),
            "mean_darf_minus_fixed1_miou": (
                mean_or_none(all_values)
            ),
            "darf_wins": sum(
                value > 0
                for value in all_values
            ),
            "fixed1_wins": sum(
                value < 0
                for value in all_values
            ),
        },
        "by_axis": by_axis,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare Fixed g=1.0 and DARF on cross-modal stress validation."
        )
    )

    p.add_argument(
        "--fixed-root",
        type=Path,
        default=DEFAULT_FIXED_ROOT,
    )
    p.add_argument(
        "--darf-root",
        type=Path,
        default=DEFAULT_DARF_ROOT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return p.parse_args()


def main() -> None:
    x = parse_args()

    fixed_root = resolve(x.fixed_root)
    darf_root = resolve(x.darf_root)
    output_root = resolve(x.output_root)

    fixed_summary = load_json(
        fixed_root
        / "crossmodal_stress_summary.json"
    )
    darf_summary = load_json(
        darf_root
        / "crossmodal_stress_summary.json"
    )

    fixed_hash = fixed_summary.get(
        "protocol_sha256"
    )
    darf_hash = darf_summary.get(
        "protocol_sha256"
    )

    if not fixed_hash or fixed_hash != darf_hash:
        raise RuntimeError(
            "Fixed1/DARF stress protocol hashes differ."
        )

    fixed_rows = index_results(
        fixed_summary
    )
    darf_rows = index_results(
        darf_summary
    )

    if set(fixed_rows) != set(darf_rows):
        raise RuntimeError(
            "Fixed1/DARF condition sets differ. "
            "Run both with --suite all before comparison."
        )

    conditions = list(
        fixed_rows.keys()
    )

    comparison_rows = [
        compare_condition_rows(
            fixed_rows[condition],
            darf_rows[condition],
        )
        for condition in conditions
    ]

    gate_payload = load_json(
        darf_root / "gate_statistics.json"
    )

    if (
        gate_payload.get("protocol_sha256")
        != fixed_hash
    ):
        raise RuntimeError(
            "DARF gate statistics protocol hash mismatch."
        )

    gates = index_gates(
        gate_payload
    )
    mechanism = mechanism_report(
        gates
    )
    performance = aggregate_performance(
        comparison_rows
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_delta_fields = sorted(
        {
            key
            for row in comparison_rows
            for key in row
            if key.startswith(
                "darf_minus_fixed1_iou_"
            )
        }
    )
    class_fixed_fields = sorted(
        {
            key
            for row in comparison_rows
            for key in row
            if key.startswith(
                "fixed1_iou_"
            )
        }
    )
    class_darf_fields = sorted(
        {
            key
            for row in comparison_rows
            for key in row
            if key.startswith(
                "darf_iou_"
            )
        }
    )

    fields = [
        "condition",
        "stress_axis",
        "family",
        "severity_level",
        "mismatch_role",
        "rgb_level",
        "nir_level",
        "shift_pixels",
        "fixed1_miou",
        "darf_miou",
        "darf_minus_fixed1_miou",
        *class_fixed_fields,
        *class_darf_fields,
        *class_delta_fields,
    ]

    write_csv(
        output_root
        / "darf_vs_fixed1_condition_comparison.csv",
        comparison_rows,
        fields,
    )

    write_csv(
        output_root
        / "gate_nir_only_direction.csv",
        mechanism["nir_only"]["rows"],
        [
            "family",
            "scale",
            "g_L1",
            "g_L2",
            "g_L3",
            "delta_L3_vs_clean",
            "monotonic_nonincreasing_L1_L2_L3",
        ],
    )

    write_csv(
        output_root
        / "gate_dropout_direction.csv",
        mechanism["nir_dropout"]["rows"],
        [
            "scale",
            "g_clean",
            "g_dropout",
            "delta_dropout_vs_clean",
            "gate_below_clean",
        ],
    )

    write_csv(
        output_root
        / "gate_mismatch_direction.csv",
        mechanism["severity_mismatch"]["rows"],
        [
            "family",
            "scale",
            "g_rgbL1_nirL3",
            "g_rgbL3_nirL1",
            "expected_direction_delta",
            "gate_higher_when_nir_relatively_better",
        ],
    )

    write_csv(
        output_root
        / "gate_misregistration_direction.csv",
        mechanism["misregistration"]["rows"],
        [
            "scale",
            "g_L1",
            "g_L2",
            "g_L3",
            "delta_L3_vs_clean",
            "monotonic_nonincreasing_L1_L2_L3",
        ],
    )

    report = {
        "protocol_sha256": fixed_hash,
        "fixed_model": fixed_summary.get("model"),
        "darf_model": darf_summary.get("model"),
        "performance": performance,
        "mechanism": mechanism,
        "interpretation_rule": {
            "stronger_dynamic_fusion_evidence_requires": [
                (
                    "DARF performance advantage over Fixed g=1.0 in "
                    "relative-modality-quality stress conditions"
                ),
                (
                    "Gate movement in the expected direction when NIR "
                    "reliability changes"
                ),
            ],
            "not_sufficient_by_itself": [
                (
                    "DARF matching Fixed g=1.0 while gates remain saturated"
                ),
                (
                    "small gate changes without corresponding performance benefit"
                ),
            ],
            "significance_note": (
                "Directional evidence only; a single training seed does not "
                "establish statistical significance."
            ),
        },
        "condition_rows": comparison_rows,
    }

    save_json(
        output_root
        / "darf_vs_fixed1_crossmodal_stress_report.json",
        report,
    )

    print("=" * 120)
    print(
        "DARF vs FIXED g=1.0 CROSS-MODAL STRESS REPORT"
    )
    print("=" * 120)

    all_perf = performance["all_nonclean"]

    print("All non-clean stress conditions:")
    print(
        f"  mean DARF-Fixed1 mIoU = "
        f"{all_perf['mean_darf_minus_fixed1_miou']:+.8f}"
    )
    print(
        f"  DARF wins / Fixed1 wins = "
        f"{all_perf['darf_wins']} / "
        f"{all_perf['fixed1_wins']}"
    )

    print()
    print("By stress axis:")

    for axis, payload in performance[
        "by_axis"
    ].items():
        mean_delta = payload.get(
            "mean_darf_minus_fixed1_miou"
        )
        if mean_delta is None:
            continue
        print(
            f"  {axis:<32} "
            f"{mean_delta:+.8f}"
        )

    print()
    print("Gate directional checks:")
    print(
        "  NIR-only monotonic L1->L3: "
        f"{mechanism['nir_only']['monotonic_pairs_passed']}/"
        f"{mechanism['nir_only']['monotonic_pairs_total']}"
    )
    print(
        "  Dropout gate below clean:  "
        f"{mechanism['nir_dropout']['scales_below_clean']}/"
        f"{mechanism['nir_dropout']['scales_total']}"
    )
    print(
        "  Mismatch expected direction: "
        f"{mechanism['severity_mismatch']['direction_checks_passed']}/"
        f"{mechanism['severity_mismatch']['direction_checks_total']}"
    )
    print(
        "  Misregistration monotonic:  "
        f"{mechanism['misregistration']['monotonic_scales_passed']}/"
        f"{mechanism['misregistration']['monotonic_scales_total']}"
    )

    print()
    print(
        f"Report: "
        f"{output_root / 'darf_vs_fixed1_crossmodal_stress_report.json'}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
