#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict same-protocol comparison: RR-DARF vs Fixed g=1.0.

Place this file in:
    <repo>/tools/run_compare_rr_darf_vs_fixed1_strict.py

It uses the repository's existing frozen validators:
    tools/validate_model_b2_rgbnir_fixed1_joint_robust4.py
    tools/validate_model_b2_rgbnir_crossmodal_stress.py
    tools/validate_model_b2_darf_rr.py

Default experiment:
    1) Verify the two checkpoints share the same COMMON training protocol
       (same seed, epochs, LR, semantic loss, Joint Robust4 definition,
       normalization, data seed, etc.; model-specific RR gate losses are not
       incorrectly required to equal Fixed1).
    2) Verify the formal Joint validation protocol hash is identical.
    3) Verify the cross-modal stress protocol hash is identical.
    4) Run Fixed g=1.0 formal 13-condition validation.
    5) Run Fixed g=1.0 full 25-condition cross-modal stress validation.
    6) Run RR-DARF formal + full stress validation.
    7) Require exact condition-set equality.
    8) Report:
         - Clean mIoU
         - 12 degraded mean mIoU
         - 4 x L3 mean mIoU
         - condition-wise RR-DARF - Fixed1
         - per-class IoU deltas
         - Low vegetation / Tree focused table
         - all stress condition deltas
         - key stress summary
         - RR-DARF Gate directional diagnostics

No significance threshold is invented. This script reports measured deltas and
mechanism-direction checks only.

Example:
    python tools/run_compare_rr_darf_vs_fixed1_strict.py

Compare already-completed outputs without rerunning validation:
    python tools/run_compare_rr_darf_vs_fixed1_strict.py --skip-validation

Custom checkpoints:
    python tools/run_compare_rr_darf_vs_fixed1_strict.py \
      --fixed-checkpoint outputs/training/b2_rgbnir_fixed1_joint_robust4/checkpoints/final.pt \
      --rr-checkpoint outputs/training/b2_darf_rr_joint_robust4/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_FIXED_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "b2_rgbnir_fixed1_joint_robust4"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_RR_CHECKPOINT = (
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
    / "rr_darf_vs_fixed1_strict"
)

EXPECTED_FORMAL_COUNT = 13
EXPECTED_STRESS_COUNT = 25
FORMAL_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "underexposure",
    "fog",
)
MISMATCH_FAMILIES = FORMAL_FAMILIES

COMMON_TRAIN_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("backbone",),
    ("input_modalities",),
    ("nir_used",),
    ("dual_encoder",),
    ("regime",),
    ("epochs",),
    ("batch_size",),
    ("grad_accum_steps",),
    ("effective_batch_size_nominal",),
    ("base_lr",),
    ("new_lr",),
    ("weight_decay",),
    ("warmup_steps",),
    ("grad_clip",),
    ("loss", "ce_weight"),
    ("loss", "lovasz_weight"),
    ("loss", "gate_bce_weight"),
    ("joint_corruption_training",),
    ("updates_per_epoch",),
    ("total_update_steps",),
    ("seed",),
    ("data_seed",),
    ("amp",),
    ("gradient_checkpointing_requested",),
)


def resolve(path: Path | str) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(x) for x in cmd])


def run_streaming(cmd: Sequence[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 132, flush=True)
    print("[run] " + shell_join(cmd), flush=True)
    print("[log] " + str(log_path), flush=True)
    print("=" * 132, flush=True)
    started = time.time()

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n=== COMMAND ===\n")
        log.write(shell_join(cmd) + "\n")
        log.flush()

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()

    elapsed = time.time() - started
    if rc != 0:
        raise RuntimeError(
            f"Command failed with exit code {rc}: {shell_join(cmd)}"
        )
    print(f"[stage finished] elapsed={elapsed / 60.0:.2f} min", flush=True)


def deep_get(obj: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return "__MISSING__"
        cur = cur[key]
    return cur


def load_checkpoint_protocol(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required to audit checkpoint metadata."
        ) from exc

    obj = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Checkpoint is not a mapping: {checkpoint}")
    protocol = obj.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"Checkpoint protocol metadata missing: {checkpoint}")
    meta = {
        "model_id": obj.get("model_id"),
        "model_name": obj.get("model_name"),
        "variant": obj.get("variant"),
        "regime": obj.get("regime"),
        "epoch": obj.get("epoch"),
        "global_step": obj.get("global_step", obj.get("step")),
    }
    return dict(protocol), meta


def audit_common_training_protocol(
    fixed_checkpoint: Path,
    rr_checkpoint: Path,
    *,
    require_same_seed: bool,
) -> Dict[str, Any]:
    fixed_protocol, fixed_meta = load_checkpoint_protocol(fixed_checkpoint)
    rr_protocol, rr_meta = load_checkpoint_protocol(rr_checkpoint)

    checks: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []

    for path in COMMON_TRAIN_PATHS:
        if path == ("seed",) and not require_same_seed:
            continue
        a = deep_get(fixed_protocol, path)
        b = deep_get(rr_protocol, path)
        ok = a == b
        row = {
            "field": ".".join(path),
            "fixed1": a,
            "rr_darf": b,
            "match": ok,
        }
        checks.append(row)
        if not ok:
            mismatches.append(row)

    # Explicit mechanism differences are expected and recorded, not treated as failures.
    expected_differences = {
        "fixed1_fusion_rule": fixed_protocol.get("fusion_rule"),
        "rr_darf_fusion_rule": rr_protocol.get("fusion_rule"),
        "fixed1_fixed_nir_strength": fixed_protocol.get("fixed_nir_strength"),
        "rr_darf_gate_training": rr_protocol.get("gate_training"),
        "rr_darf_relative_reliability_auxiliary_training": rr_protocol.get(
            "relative_reliability_auxiliary_training"
        ),
    }

    payload = {
        "fixed_checkpoint": str(fixed_checkpoint),
        "rr_checkpoint": str(rr_checkpoint),
        "fixed_checkpoint_meta": fixed_meta,
        "rr_checkpoint_meta": rr_meta,
        "require_same_seed": require_same_seed,
        "common_checks": checks,
        "common_protocol_match": not mismatches,
        "mismatches": mismatches,
        "expected_method_specific_differences": expected_differences,
    }
    if mismatches:
        details = "\n".join(
            f"  - {x['field']}: fixed={x['fixed1']!r}, rr={x['rr_darf']!r}"
            for x in mismatches
        )
        raise RuntimeError(
            "COMMON training protocol mismatch. Strict comparison aborted:\n"
            + details
        )
    return payload


def import_protocol_modules():
    import validate_model_b2_rgbnir_fixed1_joint_robust4 as fixed_formal
    import validate_model_b2_rgbnir_crossmodal_stress as fixed_stress
    import validate_model_b2_darf_rr as rr_val

    return fixed_formal, fixed_stress, rr_val


def audit_validation_protocols() -> Dict[str, Any]:
    fixed_formal, fixed_stress, rr_val = import_protocol_modules()

    fixed_formal_hash = fixed_formal.joint_validation_protocol_sha256()
    rr_formal_hash = rr_val.joint_val.joint_validation_protocol_sha256()

    fixed_stress_hash = fixed_stress.protocol_sha256()
    rr_stress_hash = rr_val.stress_val.protocol_sha256()

    fixed_formal_conditions = [
        str(x["condition"]) for x in fixed_formal.suite_conditions("all")
    ]
    rr_formal_conditions = [
        str(x["condition"]) for x in rr_val.conditions_for_suite("formal")
    ]
    fixed_stress_conditions = [
        str(x["condition"]) for x in fixed_stress.stress_conditions("all")
    ]
    rr_stress_conditions = [
        str(x["condition"]) for x in rr_val.conditions_for_suite("stress")
    ]

    checks = {
        "formal_protocol_hash_equal": fixed_formal_hash == rr_formal_hash,
        "stress_protocol_hash_equal": fixed_stress_hash == rr_stress_hash,
        "formal_condition_sequence_equal": fixed_formal_conditions == rr_formal_conditions,
        "stress_condition_sequence_equal": fixed_stress_conditions == rr_stress_conditions,
        "formal_condition_count_is_13": len(fixed_formal_conditions) == EXPECTED_FORMAL_COUNT,
        "stress_condition_count_is_25": len(fixed_stress_conditions) == EXPECTED_STRESS_COUNT,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Frozen validation protocol equality check failed:\n"
            + json.dumps(checks, indent=2, ensure_ascii=False)
        )

    return {
        "checks": checks,
        "formal_protocol_sha256": fixed_formal_hash,
        "stress_protocol_sha256": fixed_stress_hash,
        "formal_conditions": fixed_formal_conditions,
        "stress_conditions": fixed_stress_conditions,
    }


def validator_common_args(x: argparse.Namespace) -> List[str]:
    args = [
        "--batch-size",
        str(x.batch_size),
        "--num-workers",
        "0",
        "--device",
        str(x.device),
        "--log-every",
        str(x.log_every),
        "--confusion-chunk-rows",
        str(x.confusion_chunk_rows),
        "--fog-chunk-rows",
        str(x.fog_chunk_rows),
    ]
    if x.no_amp:
        args.append("--no-amp")
    if x.save_predictions:
        args.append("--save-predictions")
    if x.force:
        args.append("--force")
    return args


def run_validations(
    *,
    x: argparse.Namespace,
    fixed_checkpoint: Path,
    rr_checkpoint: Path,
    output_root: Path,
) -> Dict[str, str]:
    fixed_root = output_root / "fixed1"
    rr_root = output_root / "rr_darf"
    fixed_formal_root = fixed_root / "formal_joint"
    fixed_stress_root = fixed_root / "crossmodal_stress"

    common = validator_common_args(x)

    commands = {
        "fixed_formal": [
            sys.executable,
            str(TOOLS_DIR / "validate_model_b2_rgbnir_fixed1_joint_robust4.py"),
            "--variant",
            "fixed",
            "--suite",
            "all",
            "--checkpoint",
            str(fixed_checkpoint),
            "--output-root",
            str(fixed_formal_root),
            "--nir-fog-scatter-ratio",
            "0.65",
            *common,
        ],
        "fixed_stress": [
            sys.executable,
            str(TOOLS_DIR / "validate_model_b2_rgbnir_crossmodal_stress.py"),
            "--variant",
            "fixed1",
            "--suite",
            "all",
            "--checkpoint",
            str(fixed_checkpoint),
            "--output-root",
            str(fixed_stress_root),
            *common,
        ],
        "rr_all": [
            sys.executable,
            str(TOOLS_DIR / "validate_model_b2_darf_rr.py"),
            "--suite",
            "all",
            "--checkpoint",
            str(rr_checkpoint),
            "--output-root",
            str(rr_root),
            *common,
        ],
    }

    if x.dry_run:
        print("\nDRY RUN — validators will not be executed.")
        for name, cmd in commands.items():
            print(f"[{name}] {shell_join(cmd)}")
        return {
            "fixed_formal_root": str(fixed_formal_root),
            "fixed_stress_root": str(fixed_stress_root),
            "rr_formal_root": str(rr_root / "formal_joint"),
            "rr_stress_root": str(rr_root / "crossmodal_stress"),
        }

    logs = output_root / "runner_logs"
    run_streaming(commands["fixed_formal"], log_path=logs / "fixed_formal.log")
    run_streaming(commands["fixed_stress"], log_path=logs / "fixed_stress.log")
    run_streaming(commands["rr_all"], log_path=logs / "rr_all.log")

    return {
        "fixed_formal_root": str(fixed_formal_root),
        "fixed_stress_root": str(fixed_stress_root),
        "rr_formal_root": str(rr_root / "formal_joint"),
        "rr_stress_root": str(rr_root / "crossmodal_stress"),
    }


def index_results(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["condition"]): row
        for row in payload.get("results", [])
        if isinstance(row, Mapping) and "condition" in row
    }


def assert_same_condition_set(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    expected_count: int,
    suite_name: str,
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    ia = index_results(a)
    ib = index_results(b)
    if len(ia) != expected_count or len(ib) != expected_count:
        raise RuntimeError(
            f"{suite_name}: expected {expected_count} unique conditions, "
            f"got fixed={len(ia)}, rr={len(ib)}"
        )
    if set(ia) != set(ib):
        raise RuntimeError(
            f"{suite_name}: condition sets differ.\n"
            f"fixed-only={sorted(set(ia) - set(ib))}\n"
            f"rr-only={sorted(set(ib) - set(ia))}"
        )
    return ia, ib


def formal_aggregate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    clean = [float(r["miou"]) for r in rows if r["condition"] == "Clean"]
    degraded = [float(r["miou"]) for r in rows if r["condition"] != "Clean"]
    l3 = [float(r["miou"]) for r in rows if r.get("severity_level") == "L3"]

    if len(clean) != 1 or len(degraded) != 12 or len(l3) != 4:
        raise RuntimeError(
            "Formal aggregate cardinality check failed: "
            f"clean={len(clean)}, degraded={len(degraded)}, L3={len(l3)}"
        )
    return {
        "clean_miou": clean[0],
        "mean_degraded_miou_12_conditions": mean(degraded),
        "mean_L3_miou_4_families": mean(l3),
    }


def class_key(name: str) -> str:
    return " ".join(
        str(name).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def per_class_map(metrics: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in metrics.get("per_class", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("class_name", "")).strip()
        if not name or item.get("iou") is None:
            continue
        out[name] = float(item["iou"])
    return out


def verify_metric_protocol_hashes(
    root: Path,
    *,
    key: str,
    expected_hash: str,
    expected_min_files: int,
) -> Dict[str, Any]:
    files = sorted(root.rglob("metrics.json"))
    if len(files) < expected_min_files:
        raise RuntimeError(
            f"Too few metrics files under {root}: {len(files)} < {expected_min_files}"
        )
    bad = []
    for path in files:
        obj = load_json(path)
        got = obj.get(key)
        if got != expected_hash:
            bad.append({"path": str(path), "got": got})
    if bad:
        raise RuntimeError(
            f"Protocol hash mismatch in metrics under {root}:\n"
            + json.dumps(bad[:10], indent=2, ensure_ascii=False)
        )
    return {
        "root": str(root),
        "metric_files_checked": len(files),
        "protocol_hash_key": key,
        "expected_hash": expected_hash,
        "all_match": True,
    }


def load_formal_per_class_comparison(
    *,
    fixed_formal_root: Path,
    rr_formal_root: Path,
) -> List[Dict[str, Any]]:
    fixed_formal, _, _ = import_protocol_modules()
    rows: List[Dict[str, Any]] = []

    for item in fixed_formal.suite_conditions("all"):
        condition = str(item["condition"])
        fixed_path = fixed_formal.condition_output_dir(
            fixed_formal_root, item
        ) / "metrics.json"
        rr_path = fixed_formal.condition_output_dir(
            rr_formal_root, item
        ) / "metrics.json"
        f = per_class_map(load_json(fixed_path))
        r = per_class_map(load_json(rr_path))
        if set(f) != set(r):
            raise RuntimeError(
                f"Per-class names differ at {condition}: "
                f"fixed={sorted(f)}, rr={sorted(r)}"
            )
        for class_name in sorted(f):
            rows.append(
                {
                    "condition": condition,
                    "family": item.get("family"),
                    "severity_level": item.get("level"),
                    "class_name": class_name,
                    "fixed1_iou": f[class_name],
                    "rr_darf_iou": r[class_name],
                    "rr_darf_minus_fixed1_iou": r[class_name] - f[class_name],
                }
            )
    return rows


def condition_compare_rows(
    fixed_index: Mapping[str, Mapping[str, Any]],
    rr_index: Mapping[str, Mapping[str, Any]],
    *,
    suite: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for condition in fixed_index:
        f = fixed_index[condition]
        r = rr_index[condition]
        row: Dict[str, Any] = {
            "suite": suite,
            "condition": condition,
            "family": f.get("family"),
            "severity_level": f.get("severity_level"),
            "stress_axis": f.get("stress_axis"),
            "mismatch_role": f.get("mismatch_role"),
            "rgb_level": f.get("rgb_level"),
            "nir_level": f.get("nir_level"),
            "shift_pixels": f.get("shift_pixels"),
            "fixed1_miou": float(f["miou"]),
            "rr_darf_miou": float(r["miou"]),
        }
        row["rr_darf_minus_fixed1_miou"] = (
            row["rr_darf_miou"] - row["fixed1_miou"]
        )

        # Stress summaries contain per-class iou_* fields; preserve all common ones.
        for key in sorted(set(f) & set(r)):
            if not key.startswith("iou_"):
                continue
            if f.get(key) is None or r.get(key) is None:
                continue
            row[f"fixed1_{key}"] = float(f[key])
            row[f"rr_darf_{key}"] = float(r[key])
            row[f"rr_darf_minus_fixed1_{key}"] = float(r[key]) - float(f[key])
        rows.append(row)
    return rows


def write_dynamic_csv(path: Path, rows: Sequence[Mapping[str, Any]], leading: Sequence[str]) -> None:
    extra = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in set(leading)
        }
    )
    write_csv(path, rows, [*leading, *extra])


def key_stress_metrics(index: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    def miou(condition: str) -> float:
        if condition not in index:
            raise KeyError(f"Stress condition missing: {condition}")
        return float(index[condition]["miou"])

    nir_worse = [
        miou(f"mismatch_{family}_rgbL1_nirL3")
        for family in MISMATCH_FAMILIES
    ]
    rgb_worse = [
        miou(f"mismatch_{family}_rgbL3_nirL1")
        for family in MISMATCH_FAMILIES
    ]
    return {
        "nir_noise_L3_miou": miou("nir_only_gaussian_noise_L3"),
        "nir_dropout_full_miou": miou("nir_dropout_full"),
        "mismatch_rgbL1_nirL3_mean_4families": mean(nir_worse),
        "mismatch_rgbL3_nirL1_mean_4families": mean(rgb_worse),
        "misregistration_16px_miou": miou("misregistration_L3"),
    }


def gate_index(payload: Mapping[str, Any]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row["condition"]), int(row["scale"])): row
        for row in payload.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("condition") is not None
        and row.get("scale") is not None
    }


def gate_value(gates: Mapping[Tuple[str, int], Mapping[str, Any]], condition: str, scale: int) -> float:
    return float(gates[(condition, scale)]["g_nir_mean"])


def gate_mechanism_report(gate_payload: Mapping[str, Any]) -> Dict[str, Any]:
    gates = gate_index(gate_payload)
    clean = {s: gate_value(gates, "Clean", s) for s in range(1, 5)}

    nir_only_checks = []
    for family in FORMAL_FAMILIES:
        for s in range(1, 5):
            vals = [
                gate_value(gates, f"nir_only_{family}_{level}", s)
                for level in ("L1", "L2", "L3")
            ]
            nir_only_checks.append(
                {
                    "family": family,
                    "scale": s,
                    "g_L1": vals[0],
                    "g_L2": vals[1],
                    "g_L3": vals[2],
                    "delta_L3_vs_clean": vals[2] - clean[s],
                    "monotonic_nonincreasing_L1_L2_L3": (
                        vals[0] >= vals[1] >= vals[2]
                    ),
                }
            )

    dropout_checks = []
    for s in range(1, 5):
        g = gate_value(gates, "nir_dropout_full", s)
        dropout_checks.append(
            {
                "scale": s,
                "g_clean": clean[s],
                "g_dropout": g,
                "delta_dropout_vs_clean": g - clean[s],
                "gate_below_clean": g < clean[s],
            }
        )

    mismatch_checks = []
    for family in MISMATCH_FAMILIES:
        for s in range(1, 5):
            nir_bad = gate_value(
                gates, f"mismatch_{family}_rgbL1_nirL3", s
            )
            rgb_bad = gate_value(
                gates, f"mismatch_{family}_rgbL3_nirL1", s
            )
            mismatch_checks.append(
                {
                    "family": family,
                    "scale": s,
                    "g_rgbL1_nirL3": nir_bad,
                    "g_rgbL3_nirL1": rgb_bad,
                    "delta_expected_direction": rgb_bad - nir_bad,
                    "gate_higher_when_nir_relatively_better": rgb_bad > nir_bad,
                }
            )

    misreg_checks = []
    for s in range(1, 5):
        vals = [
            gate_value(gates, f"misregistration_{level}", s)
            for level in ("L1", "L2", "L3")
        ]
        misreg_checks.append(
            {
                "scale": s,
                "g_2px": vals[0],
                "g_8px": vals[1],
                "g_16px": vals[2],
                "delta_16px_vs_clean": vals[2] - clean[s],
                "monotonic_nonincreasing_2_8_16px": (
                    vals[0] >= vals[1] >= vals[2]
                ),
            }
        )

    return {
        "clean_gate_by_scale": clean,
        "nir_only": {
            "rows": nir_only_checks,
            "passed": sum(
                bool(x["monotonic_nonincreasing_L1_L2_L3"])
                for x in nir_only_checks
            ),
            "total": len(nir_only_checks),
        },
        "dropout": {
            "rows": dropout_checks,
            "passed": sum(bool(x["gate_below_clean"]) for x in dropout_checks),
            "total": len(dropout_checks),
        },
        "mismatch": {
            "rows": mismatch_checks,
            "passed": sum(
                bool(x["gate_higher_when_nir_relatively_better"])
                for x in mismatch_checks
            ),
            "total": len(mismatch_checks),
        },
        "misregistration": {
            "rows": misreg_checks,
            "passed": sum(
                bool(x["monotonic_nonincreasing_2_8_16px"])
                for x in misreg_checks
            ),
            "total": len(misreg_checks),
        },
    }


def compare_outputs(
    *,
    output_root: Path,
    validation_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    fixed_formal_root = output_root / "fixed1" / "formal_joint"
    fixed_stress_root = output_root / "fixed1" / "crossmodal_stress"
    rr_formal_root = output_root / "rr_darf" / "formal_joint"
    rr_stress_root = output_root / "rr_darf" / "crossmodal_stress"

    fixed_formal = load_json(fixed_formal_root / "all_conditions_summary.json")
    rr_formal = load_json(rr_formal_root / "all_conditions_summary.json")
    fixed_stress = load_json(fixed_stress_root / "crossmodal_stress_summary.json")
    rr_stress = load_json(rr_stress_root / "crossmodal_stress_summary.json")

    formal_hash = str(validation_audit["formal_protocol_sha256"])
    stress_hash = str(validation_audit["stress_protocol_sha256"])

    hash_audits = [
        verify_metric_protocol_hashes(
            fixed_formal_root,
            key="joint_validation_protocol_sha256",
            expected_hash=formal_hash,
            expected_min_files=EXPECTED_FORMAL_COUNT,
        ),
        verify_metric_protocol_hashes(
            rr_formal_root,
            key="joint_validation_protocol_sha256",
            expected_hash=formal_hash,
            expected_min_files=EXPECTED_FORMAL_COUNT,
        ),
        verify_metric_protocol_hashes(
            fixed_stress_root,
            key="crossmodal_stress_protocol_sha256",
            expected_hash=stress_hash,
            expected_min_files=EXPECTED_STRESS_COUNT,
        ),
        verify_metric_protocol_hashes(
            rr_stress_root,
            key="crossmodal_stress_protocol_sha256",
            expected_hash=stress_hash,
            expected_min_files=EXPECTED_STRESS_COUNT,
        ),
    ]

    fi, ri = assert_same_condition_set(
        fixed_formal,
        rr_formal,
        expected_count=EXPECTED_FORMAL_COUNT,
        suite_name="formal",
    )
    fs, rs = assert_same_condition_set(
        fixed_stress,
        rr_stress,
        expected_count=EXPECTED_STRESS_COUNT,
        suite_name="stress",
    )

    formal_rows = condition_compare_rows(fi, ri, suite="formal")
    stress_rows = condition_compare_rows(fs, rs, suite="stress")

    fixed_agg = formal_aggregate(fi.values())
    rr_agg = formal_aggregate(ri.values())
    aggregate_rows = []
    for key in (
        "clean_miou",
        "mean_degraded_miou_12_conditions",
        "mean_L3_miou_4_families",
    ):
        aggregate_rows.append(
            {
                "metric": key,
                "fixed1": fixed_agg[key],
                "rr_darf": rr_agg[key],
                "rr_darf_minus_fixed1": rr_agg[key] - fixed_agg[key],
            }
        )

    per_class_rows = load_formal_per_class_comparison(
        fixed_formal_root=fixed_formal_root,
        rr_formal_root=rr_formal_root,
    )
    focus_rows = [
        row
        for row in per_class_rows
        if class_key(str(row["class_name"])) in {"low vegetation", "tree"}
    ]

    fixed_key = key_stress_metrics(fs)
    rr_key = key_stress_metrics(rs)
    key_stress_rows = []
    for key in fixed_key:
        key_stress_rows.append(
            {
                "metric": key,
                "fixed1": fixed_key[key],
                "rr_darf": rr_key[key],
                "rr_darf_minus_fixed1": rr_key[key] - fixed_key[key],
            }
        )

    gate_payload = load_json(rr_stress_root / "gate_statistics.json")
    gate_report = gate_mechanism_report(gate_payload)

    write_csv(
        output_root / "formal_aggregate_comparison.csv",
        aggregate_rows,
        ["metric", "fixed1", "rr_darf", "rr_darf_minus_fixed1"],
    )
    write_dynamic_csv(
        output_root / "formal_condition_comparison.csv",
        formal_rows,
        [
            "suite",
            "condition",
            "family",
            "severity_level",
            "fixed1_miou",
            "rr_darf_miou",
            "rr_darf_minus_fixed1_miou",
        ],
    )
    write_csv(
        output_root / "formal_per_class_comparison.csv",
        per_class_rows,
        [
            "condition",
            "family",
            "severity_level",
            "class_name",
            "fixed1_iou",
            "rr_darf_iou",
            "rr_darf_minus_fixed1_iou",
        ],
    )
    write_csv(
        output_root / "formal_focus_low_vegetation_tree.csv",
        focus_rows,
        [
            "condition",
            "family",
            "severity_level",
            "class_name",
            "fixed1_iou",
            "rr_darf_iou",
            "rr_darf_minus_fixed1_iou",
        ],
    )
    write_dynamic_csv(
        output_root / "stress_condition_comparison.csv",
        stress_rows,
        [
            "suite",
            "condition",
            "stress_axis",
            "family",
            "severity_level",
            "mismatch_role",
            "rgb_level",
            "nir_level",
            "shift_pixels",
            "fixed1_miou",
            "rr_darf_miou",
            "rr_darf_minus_fixed1_miou",
        ],
    )
    write_csv(
        output_root / "stress_key_metrics.csv",
        key_stress_rows,
        ["metric", "fixed1", "rr_darf", "rr_darf_minus_fixed1"],
    )

    save_json_atomic(output_root / "rr_gate_mechanism_report.json", gate_report)

    summary = {
        "comparison": "RR-DARF vs Fixed g=1.0",
        "strict_same_validation_protocol": True,
        "formal_protocol_sha256": formal_hash,
        "stress_protocol_sha256": stress_hash,
        "metric_protocol_hash_audit": hash_audits,
        "formal_aggregates": {
            "fixed1": fixed_agg,
            "rr_darf": rr_agg,
            "rr_minus_fixed1": {
                k: rr_agg[k] - fixed_agg[k] for k in fixed_agg
            },
        },
        "key_stress": {
            "fixed1": fixed_key,
            "rr_darf": rr_key,
            "rr_minus_fixed1": {
                k: rr_key[k] - fixed_key[k] for k in fixed_key
            },
        },
        "rr_gate_direction_checks": {
            "nir_only": {
                "passed": gate_report["nir_only"]["passed"],
                "total": gate_report["nir_only"]["total"],
            },
            "dropout": {
                "passed": gate_report["dropout"]["passed"],
                "total": gate_report["dropout"]["total"],
            },
            "mismatch": {
                "passed": gate_report["mismatch"]["passed"],
                "total": gate_report["mismatch"]["total"],
            },
            "misregistration": {
                "passed": gate_report["misregistration"]["passed"],
                "total": gate_report["misregistration"]["total"],
            },
        },
        "interpretation_rule": (
            "Gate direction alone is not treated as proof. The mechanism claim "
            "requires reasonable gate direction AND stable RR-DARF performance "
            "gain over Fixed g=1.0 in key relative-reliability stress conditions."
        ),
        "significance_policy": (
            "No significance threshold is invented by this script. Use the "
            "multi-seed experiment for stability/mean±std."
        ),
        "files": {
            "formal_aggregate_comparison_csv": str(
                output_root / "formal_aggregate_comparison.csv"
            ),
            "formal_condition_comparison_csv": str(
                output_root / "formal_condition_comparison.csv"
            ),
            "formal_per_class_comparison_csv": str(
                output_root / "formal_per_class_comparison.csv"
            ),
            "formal_focus_low_vegetation_tree_csv": str(
                output_root / "formal_focus_low_vegetation_tree.csv"
            ),
            "stress_condition_comparison_csv": str(
                output_root / "stress_condition_comparison.csv"
            ),
            "stress_key_metrics_csv": str(
                output_root / "stress_key_metrics.csv"
            ),
            "rr_gate_mechanism_report_json": str(
                output_root / "rr_gate_mechanism_report.json"
            ),
        },
    }
    save_json_atomic(output_root / "strict_comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict same-protocol RR-DARF vs Fixed g=1.0 comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fixed-checkpoint", type=Path, default=DEFAULT_FIXED_CHECKPOINT)
    p.add_argument("--rr-checkpoint", type=Path, default=DEFAULT_RR_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=16)
    p.add_argument("--confusion-chunk-rows", type=int, default=512)
    p.add_argument("--fog-chunk-rows", type=int, default=128)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Only audit and compare already-completed outputs.",
    )
    p.add_argument(
        "--skip-training-protocol-check",
        action="store_true",
        help="Do not inspect checkpoint protocol metadata.",
    )
    p.add_argument(
        "--allow-train-seed-mismatch",
        action="store_true",
        help=(
            "Allow Fixed1 and RR-DARF checkpoints with different training seeds. "
            "Not recommended for the main strict single-seed comparison."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print validator commands and protocol checks without running validation.",
    )
    return p.parse_args()


def main() -> None:
    x = parse_args()

    fixed_checkpoint = resolve(x.fixed_checkpoint)
    rr_checkpoint = resolve(x.rr_checkpoint)
    output_root = resolve(x.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for ckpt in (fixed_checkpoint, rr_checkpoint):
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

    started = time.time()

    validation_audit = audit_validation_protocols()
    save_json_atomic(
        output_root / "validation_protocol_audit.json",
        validation_audit,
    )
    print(
        "[protocol] formal hash:",
        validation_audit["formal_protocol_sha256"],
        flush=True,
    )
    print(
        "[protocol] stress hash:",
        validation_audit["stress_protocol_sha256"],
        flush=True,
    )

    training_audit = None
    if not x.skip_training_protocol_check:
        training_audit = audit_common_training_protocol(
            fixed_checkpoint,
            rr_checkpoint,
            require_same_seed=not x.allow_train_seed_mismatch,
        )
        save_json_atomic(
            output_root / "training_common_protocol_audit.json",
            training_audit,
        )
        print("[protocol] COMMON training protocol: MATCH", flush=True)

    manifest = {
        "experiment": "RR-DARF vs Fixed g=1.0 strict same-protocol comparison",
        "fixed_checkpoint": str(fixed_checkpoint),
        "rr_checkpoint": str(rr_checkpoint),
        "output_root": str(output_root),
        "validation_protocol": validation_audit,
        "training_common_protocol_checked": training_audit is not None,
        "validation_runtime": {
            "batch_size": x.batch_size,
            "num_workers": 0,
            "device": x.device,
            "amp_enabled_unless_no_amp": not x.no_amp,
            "log_every": x.log_every,
            "confusion_chunk_rows": x.confusion_chunk_rows,
            "fog_chunk_rows": x.fog_chunk_rows,
            "save_predictions": x.save_predictions,
        },
    }
    save_json_atomic(output_root / "strict_protocol_manifest.json", manifest)

    if not x.skip_validation:
        run_validations(
            x=x,
            fixed_checkpoint=fixed_checkpoint,
            rr_checkpoint=rr_checkpoint,
            output_root=output_root,
        )
        if x.dry_run:
            return

    summary = compare_outputs(
        output_root=output_root,
        validation_audit=validation_audit,
    )

    elapsed = time.time() - started
    print("\n" + "=" * 132)
    print("STRICT RR-DARF vs FIXED g=1.0 COMPARISON COMPLETE")
    print("=" * 132)
    print(f"elapsed      : {elapsed / 60.0:.2f} min")
    print(f"output       : {output_root}")
    print(
        "Clean delta  : "
        f"{summary['formal_aggregates']['rr_minus_fixed1']['clean_miou']:+.6f}"
    )
    print(
        "12-deg delta : "
        f"{summary['formal_aggregates']['rr_minus_fixed1']['mean_degraded_miou_12_conditions']:+.6f}"
    )
    print(
        "4xL3 delta   : "
        f"{summary['formal_aggregates']['rr_minus_fixed1']['mean_L3_miou_4_families']:+.6f}"
    )
    for key, value in summary["key_stress"]["rr_minus_fixed1"].items():
        print(f"{key:<44}: {value:+.6f}")
    print("=" * 132)


if __name__ == "__main__":
    main()
