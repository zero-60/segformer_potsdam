#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-seed Fixed g=1.0 / RR-DARF experiment runner and mean±std summarizer.

Place this file in:
    <repo>/tools/run_multiseed_rr_darf_vs_fixed1.py

Recommended seeds from the frozen thesis plan:
    20260917, 20260918, 20260919

Important behavior
------------------
1) Same data split, epochs, LR, augmentation/Joint Robust4 protocol, semantic
   loss, and validation protocol are kept fixed. Only --seed changes within
   each model.
2) If the original formal model already uses seed 20260917, it is reused and
   NOT retrained by default.
3) New run directories always include the seed:
       outputs/training/b2_rgbnir_fixed1_joint_robust4_seed20260918
       outputs/training/b2_darf_rr_joint_robust4_seed20260918
4) Two stress policies:
       full-all
           all three seeds run the full 25-condition stress protocol.
       main-full-others-key
           the first seed runs all 25 stress conditions;
           the other seeds run the required key subset:
             - Clean
             - NIR Gaussian Noise L3
             - NIR full dropout
             - all four RGB L1 / NIR L3 mismatch families
             - all four RGB L3 / NIR L1 mismatch families
             - misregistration 16 px
5) Final outputs include per-seed metrics and mean±sample-std (ddof=1) for:
       - Clean
       - 12 degraded mean
       - 4 x L3 mean
       - NIR Noise L3
       - NIR dropout
       - RGB L1 / NIR L3 mismatch mean (4 families)
       - RGB L3 / NIR L1 mismatch mean (4 families)
       - misregistration 16 px
   plus paired RR-DARF - Fixed1 deltas.

Examples
--------
Full reproducibility run:
    python tools/run_multiseed_rr_darf_vs_fixed1.py --stage all --stress-policy full-all

Resource-aware run:
    python tools/run_multiseed_rr_darf_vs_fixed1.py \
      --stage all \
      --stress-policy main-full-others-key

Only summarize completed runs:
    python tools/run_multiseed_rr_darf_vs_fixed1.py --stage summarize
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
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_SEEDS = (20260917, 20260918, 20260919)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "multiseed_rr_darf_vs_fixed1"
)

FIXED_BASE_NAME = "b2_rgbnir_fixed1_joint_robust4"
RR_BASE_NAME = "b2_darf_rr_joint_robust4"

ORIGINAL_FIXED_DIR = PROJECT_ROOT / "outputs" / "training" / FIXED_BASE_NAME
ORIGINAL_RR_DIR = PROJECT_ROOT / "outputs" / "training" / RR_BASE_NAME

FORMAL_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "underexposure",
    "fog",
)

KEY_STRESS_CONDITIONS = (
    "Clean",
    "nir_only_gaussian_noise_L3",
    "nir_dropout_full",
    "mismatch_gaussian_noise_rgbL1_nirL3",
    "mismatch_gaussian_noise_rgbL3_nirL1",
    "mismatch_gaussian_blur_rgbL1_nirL3",
    "mismatch_gaussian_blur_rgbL3_nirL1",
    "mismatch_underexposure_rgbL1_nirL3",
    "mismatch_underexposure_rgbL3_nirL1",
    "mismatch_fog_rgbL1_nirL3",
    "mismatch_fog_rgbL3_nirL1",
    "misregistration_L3",
)

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


def run_streaming(cmd: Sequence[str], *, log_path: Path, dry_run: bool) -> None:
    print("\n" + "=" * 132, flush=True)
    print("[run] " + shell_join(cmd), flush=True)
    print("[log] " + str(log_path), flush=True)
    print("=" * 132, flush=True)

    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_seed_list(text: str) -> Tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(seeds) < 3:
        raise ValueError("At least three seeds are required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def deep_get(obj: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return "__MISSING__"
        cur = cur[key]
    return cur


def protocol_seed(train_dir: Path) -> Optional[int]:
    p = train_dir / "protocol.json"
    if not p.is_file():
        return None
    obj = load_json(p)
    seed = obj.get("seed")
    return int(seed) if seed is not None else None


def final_checkpoint(train_dir: Path) -> Path:
    return train_dir / "checkpoints" / "final.pt"


def choose_train_dir(
    *,
    model: str,
    seed: int,
    reuse_original_seed_20260917: bool,
) -> Path:
    if model == "fixed1":
        original = ORIGINAL_FIXED_DIR
        base = FIXED_BASE_NAME
    elif model == "rr_darf":
        original = ORIGINAL_RR_DIR
        base = RR_BASE_NAME
    else:
        raise ValueError(model)

    if (
        seed == 20260917
        and reuse_original_seed_20260917
        and final_checkpoint(original).is_file()
        and protocol_seed(original) == seed
    ):
        return original

    return (
        PROJECT_ROOT
        / "outputs"
        / "training"
        / f"{base}_seed{seed}"
    )


def common_train_args(x: argparse.Namespace, seed: int, out_dir: Path) -> List[str]:
    args = [
        "--epochs", str(x.epochs),
        "--batch-size", str(x.train_batch_size),
        "--grad-accum-steps", str(x.grad_accum_steps),
        "--base-lr", str(x.base_lr),
        "--new-lr", str(x.new_lr),
        "--weight-decay", str(x.weight_decay),
        "--warmup-steps", str(x.warmup_steps),
        "--grad-clip", str(x.grad_clip),
        "--lovasz-weight", str(x.lovasz_weight),
        "--clean-warmup-epochs", str(x.clean_warmup_epochs),
        "--corruption-ramp-epochs", str(x.corruption_ramp_epochs),
        "--max-corruption-prob", str(x.max_corruption_prob),
        "--nir-fog-scatter-ratio", str(x.nir_fog_scatter_ratio),
        "--seed", str(seed),
        "--num-workers", str(x.train_num_workers),
        "--device", str(x.device),
        "--output-dir", str(out_dir),
        "--progress-every", str(x.progress_every),
        "--log-every", str(x.train_log_every),
    ]
    if x.no_amp:
        args.append("--no-amp")
    if not x.gradient_checkpointing:
        args.append("--no-gradient-checkpointing")
    if not x.pin_memory:
        args.append("--no-pin-memory")
    return args


def fixed_train_command(x: argparse.Namespace, seed: int, out_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(TOOLS_DIR / "train_model_b2_rgbnir_fixed1_joint_robust4.py"),
        "--variant",
        "fixed",
        *common_train_args(x, seed, out_dir),
    ]


def rr_train_command(x: argparse.Namespace, seed: int, out_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(TOOLS_DIR / "train_model_b2_darf_rr_joint_robust4.py"),
        "--variant",
        "darf_rr",
        *common_train_args(x, seed, out_dir),
        "--darf-initial-gate", str(x.darf_initial_gate),
        "--relative-aux-prob", str(x.relative_aux_prob),
        "--relative-aux-start-epoch", str(x.relative_aux_start_epoch),
        "--relative-min-severity", str(x.relative_min_severity),
        "--relative-rank-margin", str(x.relative_rank_margin),
        "--relative-rank-weight", str(x.relative_rank_weight),
        "--gate-saturation-weight", str(x.gate_saturation_weight),
        "--gate-saturation-free-logit", str(x.gate_saturation_free_logit),
    ]


def train_one(
    *,
    x: argparse.Namespace,
    model: str,
    seed: int,
    train_dir: Path,
    log_root: Path,
) -> Path:
    ckpt = final_checkpoint(train_dir)
    if ckpt.is_file() and protocol_seed(train_dir) == seed and x.resume_existing:
        print(
            f"[reuse] {model} seed={seed}: {ckpt}",
            flush=True,
        )
        return ckpt

    cmd = (
        fixed_train_command(x, seed, train_dir)
        if model == "fixed1"
        else rr_train_command(x, seed, train_dir)
    )
    run_streaming(
        cmd,
        log_path=log_root / f"train_{model}_seed{seed}.log",
        dry_run=x.dry_run,
    )
    if not x.dry_run and not ckpt.is_file():
        raise FileNotFoundError(
            f"Training command completed but final checkpoint is missing: {ckpt}"
        )
    return ckpt


def validation_common_args(x: argparse.Namespace) -> List[str]:
    args = [
        "--batch-size", str(x.val_batch_size),
        "--num-workers", "0",
        "--device", str(x.device),
        "--log-every", str(x.val_log_every),
        "--confusion-chunk-rows", str(x.confusion_chunk_rows),
        "--fog-chunk-rows", str(x.fog_chunk_rows),
    ]
    if x.no_amp:
        args.append("--no-amp")
    if not x.pin_memory:
        args.append("--no-pin-memory")
    if x.force_validation:
        args.append("--force")
    return args


def strict_runner_path() -> Path:
    p = TOOLS_DIR / "run_compare_rr_darf_vs_fixed1_strict.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"Required companion script not found: {p}\n"
            "Place both downloaded .py files in the repository tools/ directory."
        )
    return p


def run_full_seed_validation(
    *,
    x: argparse.Namespace,
    seed: int,
    fixed_ckpt: Path,
    rr_ckpt: Path,
    eval_root: Path,
    log_root: Path,
) -> None:
    cmd = [
        sys.executable,
        str(strict_runner_path()),
        "--fixed-checkpoint", str(fixed_ckpt),
        "--rr-checkpoint", str(rr_ckpt),
        "--output-root", str(eval_root),
        "--batch-size", str(x.val_batch_size),
        "--device", str(x.device),
        "--log-every", str(x.val_log_every),
        "--confusion-chunk-rows", str(x.confusion_chunk_rows),
        "--fog-chunk-rows", str(x.fog_chunk_rows),
    ]
    if x.no_amp:
        cmd.append("--no-amp")
    if x.force_validation:
        cmd.append("--force")
    if not x.pin_memory:
        # Companion validators understand BooleanOptionalAction only through
        # direct scripts; strict runner uses default pin-memory=True. For a
        # no-pin-memory run, use the resource-aware path below instead.
        raise RuntimeError(
            "--no-pin-memory is not supported through the companion strict "
            "runner. Use default pin-memory or resource-aware validation."
        )
    run_streaming(
        cmd,
        log_path=log_root / f"validate_full_seed{seed}.log",
        dry_run=x.dry_run,
    )


def run_formal_only(
    *,
    x: argparse.Namespace,
    model: str,
    checkpoint: Path,
    eval_root: Path,
    log_root: Path,
    seed: int,
) -> None:
    common = validation_common_args(x)
    if model == "fixed1":
        out = eval_root / "fixed1" / "formal_joint"
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "validate_model_b2_rgbnir_fixed1_joint_robust4.py"),
            "--variant", "fixed",
            "--suite", "all",
            "--checkpoint", str(checkpoint),
            "--output-root", str(out),
            "--nir-fog-scatter-ratio", "0.65",
            *common,
        ]
    elif model == "rr_darf":
        out = eval_root / "rr_darf"
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "validate_model_b2_darf_rr.py"),
            "--suite", "formal",
            "--checkpoint", str(checkpoint),
            "--output-root", str(out),
            *common,
        ]
    else:
        raise ValueError(model)

    run_streaming(
        cmd,
        log_path=log_root / f"validate_formal_{model}_seed{seed}.log",
        dry_run=x.dry_run,
    )


def python_literal_list(items: Sequence[str]) -> str:
    return repr(list(items))


def run_key_stress_fixed(
    *,
    x: argparse.Namespace,
    checkpoint: Path,
    eval_root: Path,
    log_root: Path,
    seed: int,
) -> None:
    output_root = eval_root / "fixed1" / "crossmodal_stress"
    keys = python_literal_list(KEY_STRESS_CONDITIONS)
    argv = [
        "validate_model_b2_rgbnir_crossmodal_stress.py",
        "--variant", "fixed1",
        "--suite", "all",
        "--checkpoint", str(checkpoint),
        "--output-root", str(output_root),
        *validation_common_args(x),
    ]
    code = f"""
import sys
from pathlib import Path
project_root = Path({str(PROJECT_ROOT)!r})
tools_dir = project_root / "tools"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))
import validate_model_b2_rgbnir_crossmodal_stress as m
_keys = set({keys})
_orig = m.stress_conditions
m.stress_conditions = lambda suite: [x for x in _orig("all") if str(x["condition"]) in _keys]
sys.argv = {argv!r}
m.main()
"""
    cmd = [sys.executable, "-c", code]
    run_streaming(
        cmd,
        log_path=log_root / f"validate_key_stress_fixed1_seed{seed}.log",
        dry_run=x.dry_run,
    )


def run_key_stress_rr(
    *,
    x: argparse.Namespace,
    checkpoint: Path,
    eval_root: Path,
    log_root: Path,
    seed: int,
) -> None:
    output_root = eval_root / "rr_darf"
    keys = python_literal_list(KEY_STRESS_CONDITIONS)
    argv = [
        "validate_model_b2_darf_rr.py",
        "--suite", "stress",
        "--checkpoint", str(checkpoint),
        "--output-root", str(output_root),
        *validation_common_args(x),
    ]
    code = f"""
import sys
from pathlib import Path
project_root = Path({str(PROJECT_ROOT)!r})
tools_dir = project_root / "tools"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))
import validate_model_b2_darf_rr as m
_keys = set({keys})
_orig = m.conditions_for_suite
def _filtered(suite):
    rows = _orig(suite)
    if suite == "stress":
        rows = [x for x in rows if str(x["condition"]) in _keys]
    return rows
m.conditions_for_suite = _filtered
sys.argv = {argv!r}
m.main()
"""
    cmd = [sys.executable, "-c", code]
    run_streaming(
        cmd,
        log_path=log_root / f"validate_key_stress_rr_seed{seed}.log",
        dry_run=x.dry_run,
    )


def run_resource_aware_seed_validation(
    *,
    x: argparse.Namespace,
    seed: int,
    fixed_ckpt: Path,
    rr_ckpt: Path,
    eval_root: Path,
    log_root: Path,
) -> None:
    run_formal_only(
        x=x,
        model="fixed1",
        checkpoint=fixed_ckpt,
        eval_root=eval_root,
        log_root=log_root,
        seed=seed,
    )
    run_formal_only(
        x=x,
        model="rr_darf",
        checkpoint=rr_ckpt,
        eval_root=eval_root,
        log_root=log_root,
        seed=seed,
    )
    run_key_stress_fixed(
        x=x,
        checkpoint=fixed_ckpt,
        eval_root=eval_root,
        log_root=log_root,
        seed=seed,
    )
    run_key_stress_rr(
        x=x,
        checkpoint=rr_ckpt,
        eval_root=eval_root,
        log_root=log_root,
        seed=seed,
    )


def index_results(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["condition"]): row
        for row in payload.get("results", [])
        if isinstance(row, Mapping) and "condition" in row
    }


def formal_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    rows = list(index_results(summary).values())
    clean = [float(r["miou"]) for r in rows if r["condition"] == "Clean"]
    degraded = [float(r["miou"]) for r in rows if r["condition"] != "Clean"]
    l3 = [float(r["miou"]) for r in rows if r.get("severity_level") == "L3"]
    if len(clean) != 1 or len(degraded) != 12 or len(l3) != 4:
        raise RuntimeError(
            "Formal summary is incomplete: "
            f"clean={len(clean)}, degraded={len(degraded)}, L3={len(l3)}"
        )
    return {
        "clean_miou": clean[0],
        "mean_degraded_miou_12_conditions": mean(degraded),
        "mean_L3_miou_4_families": mean(l3),
    }


def stress_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    idx = index_results(summary)

    def m(condition: str) -> float:
        if condition not in idx:
            raise KeyError(f"Required stress condition missing: {condition}")
        return float(idx[condition]["miou"])

    nir_bad = [
        m(f"mismatch_{family}_rgbL1_nirL3")
        for family in FORMAL_FAMILIES
    ]
    rgb_bad = [
        m(f"mismatch_{family}_rgbL3_nirL1")
        for family in FORMAL_FAMILIES
    ]
    return {
        "nir_noise_L3_miou": m("nir_only_gaussian_noise_L3"),
        "nir_dropout_full_miou": m("nir_dropout_full"),
        "mismatch_rgbL1_nirL3_mean_4families": mean(nir_bad),
        "mismatch_rgbL3_nirL1_mean_4families": mean(rgb_bad),
        "misregistration_16px_miou": m("misregistration_L3"),
    }


def evaluation_paths(eval_root: Path) -> Dict[str, Path]:
    return {
        "fixed_formal": eval_root / "fixed1" / "formal_joint" / "all_conditions_summary.json",
        "rr_formal": eval_root / "rr_darf" / "formal_joint" / "all_conditions_summary.json",
        "fixed_stress": eval_root / "fixed1" / "crossmodal_stress" / "crossmodal_stress_summary.json",
        "rr_stress": eval_root / "rr_darf" / "crossmodal_stress" / "crossmodal_stress_summary.json",
    }


def load_seed_metrics(seed: int, eval_root: Path) -> Dict[str, Dict[str, float]]:
    paths = evaluation_paths(eval_root)
    fixed_formal = formal_metrics(load_json(paths["fixed_formal"]))
    rr_formal = formal_metrics(load_json(paths["rr_formal"]))
    fixed_stress = stress_metrics(load_json(paths["fixed_stress"]))
    rr_stress = stress_metrics(load_json(paths["rr_stress"]))

    return {
        "fixed1": {**fixed_formal, **fixed_stress},
        "rr_darf": {**rr_formal, **rr_stress},
    }


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def aggregate_multiseed(
    *,
    seeds: Sequence[int],
    eval_roots: Mapping[int, Path],
    output_root: Path,
) -> Dict[str, Any]:
    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        metrics = load_seed_metrics(seed, eval_roots[seed])
        per_seed[seed] = metrics
        for model in ("fixed1", "rr_darf"):
            row = {"seed": seed, "model": model}
            row.update(metrics[model])
            rows.append(row)

        delta_row = {"seed": seed, "model": "rr_minus_fixed1"}
        for metric in metrics["fixed1"]:
            delta_row[metric] = (
                metrics["rr_darf"][metric] - metrics["fixed1"][metric]
            )
        rows.append(delta_row)

    metric_names = list(per_seed[seeds[0]]["fixed1"].keys())
    summary_rows: List[Dict[str, Any]] = []

    for metric in metric_names:
        for model in ("fixed1", "rr_darf", "rr_minus_fixed1"):
            if model == "rr_minus_fixed1":
                vals = [
                    per_seed[s]["rr_darf"][metric]
                    - per_seed[s]["fixed1"][metric]
                    for s in seeds
                ]
            else:
                vals = [per_seed[s][model][metric] for s in seeds]

            summary_rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "n_seeds": len(vals),
                    "mean": mean(vals),
                    "std_ddof1": sample_std(vals),
                    "mean_pm_std": f"{mean(vals):.6f} ± {sample_std(vals):.6f}",
                    "min": min(vals),
                    "max": max(vals),
                    "seeds": ",".join(str(s) for s in seeds),
                }
            )

    fields = ["seed", "model", *metric_names]
    write_csv(output_root / "multiseed_per_seed.csv", rows, fields)
    write_csv(
        output_root / "multiseed_summary.csv",
        summary_rows,
        [
            "metric",
            "model",
            "n_seeds",
            "mean",
            "std_ddof1",
            "mean_pm_std",
            "min",
            "max",
            "seeds",
        ],
    )

    payload = {
        "seeds": list(seeds),
        "std_definition": "sample standard deviation, ddof=1",
        "per_seed": per_seed,
        "summary_rows": summary_rows,
        "reporting_rule": "All requested seeds are reported; no best-seed selection.",
    }
    save_json_atomic(output_root / "multiseed_summary.json", payload)
    return payload


def load_protocol(train_dir: Path) -> Dict[str, Any]:
    return dict(load_json(train_dir / "protocol.json"))


def scrub_seed(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    x = json.loads(json.dumps(protocol, default=str))
    x.pop("seed", None)
    return x


def audit_within_model_only_seed_changes(
    *,
    model: str,
    seeds: Sequence[int],
    train_dirs: Mapping[int, Path],
) -> Dict[str, Any]:
    protocols = {seed: load_protocol(train_dirs[seed]) for seed in seeds}
    base_seed = seeds[0]
    base = scrub_seed(protocols[base_seed])
    mismatches = []
    for seed in seeds[1:]:
        current = scrub_seed(protocols[seed])
        if current != base:
            mismatches.append(
                {
                    "seed": seed,
                    "message": (
                        "Protocol differs beyond the seed. Compare protocol.json "
                        "files before using this run in the thesis aggregate."
                    ),
                }
            )
    if mismatches:
        raise RuntimeError(
            f"{model}: within-model protocol drift detected:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    return {
        "model": model,
        "seeds": list(seeds),
        "only_seed_changes": True,
        "base_seed": base_seed,
    }


def audit_cross_model_common_protocol(
    *,
    seed: int,
    fixed_dir: Path,
    rr_dir: Path,
) -> Dict[str, Any]:
    f = load_protocol(fixed_dir)
    r = load_protocol(rr_dir)
    checks = []
    mismatches = []
    for path in COMMON_TRAIN_PATHS:
        a = deep_get(f, path)
        b = deep_get(r, path)
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

    # The actual training seed must also match for paired seed comparison.
    seed_match = int(f.get("seed")) == int(r.get("seed")) == int(seed)
    if not seed_match:
        mismatches.append(
            {
                "field": "seed",
                "fixed1": f.get("seed"),
                "rr_darf": r.get("seed"),
                "expected": seed,
                "match": False,
            }
        )
    if mismatches:
        raise RuntimeError(
            f"Cross-model common protocol mismatch at seed {seed}:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    return {
        "seed": seed,
        "common_protocol_match": True,
        "checks": checks,
        "expected_method_specific_difference": (
            "Fixed1 has fixed g=1.0; RR-DARF has bounded dynamic gate plus "
            "relative-reliability gate-only auxiliary supervision."
        ),
    }


def build_manifest(
    *,
    x: argparse.Namespace,
    seeds: Sequence[int],
    fixed_dirs: Mapping[int, Path],
    rr_dirs: Mapping[int, Path],
    eval_roots: Mapping[int, Path],
) -> Dict[str, Any]:
    return {
        "experiment": "3-seed Fixed g=1.0 / RR-DARF",
        "seeds": list(seeds),
        "stress_policy": x.stress_policy,
        "main_seed_for_full_stress": seeds[0],
        "training": {
            "epochs": x.epochs,
            "batch_size": x.train_batch_size,
            "grad_accum_steps": x.grad_accum_steps,
            "base_lr": x.base_lr,
            "new_lr": x.new_lr,
            "weight_decay": x.weight_decay,
            "warmup_steps": x.warmup_steps,
            "grad_clip": x.grad_clip,
            "lovasz_weight": x.lovasz_weight,
            "clean_warmup_epochs": x.clean_warmup_epochs,
            "corruption_ramp_epochs": x.corruption_ramp_epochs,
            "max_corruption_prob": x.max_corruption_prob,
            "nir_fog_scatter_ratio": x.nir_fog_scatter_ratio,
            "train_num_workers": x.train_num_workers,
            "only_seed_should_change_within_model": True,
        },
        "rr_specific": {
            "darf_initial_gate": x.darf_initial_gate,
            "relative_aux_prob": x.relative_aux_prob,
            "relative_aux_start_epoch": x.relative_aux_start_epoch,
            "relative_min_severity": x.relative_min_severity,
            "relative_rank_margin": x.relative_rank_margin,
            "relative_rank_weight": x.relative_rank_weight,
            "gate_saturation_weight": x.gate_saturation_weight,
            "gate_saturation_free_logit": x.gate_saturation_free_logit,
        },
        "validation": {
            "val_batch_size": x.val_batch_size,
            "num_workers": 0,
            "device": x.device,
            "amp_enabled_unless_no_amp": not x.no_amp,
            "formal_conditions": 13,
            "full_stress_conditions": 25,
            "key_stress_conditions": list(KEY_STRESS_CONDITIONS),
        },
        "paths": {
            str(seed): {
                "fixed_train_dir": str(fixed_dirs[seed]),
                "rr_train_dir": str(rr_dirs[seed]),
                "evaluation_root": str(eval_roots[seed]),
            }
            for seed in seeds
        },
        "reporting_rule": "Report all seeds; never select only the best seed.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-seed Fixed1 / RR-DARF train-validate-summarize runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "train", "validate", "summarize"),
        default="all",
    )
    p.add_argument(
        "--seeds",
        default="20260917,20260918,20260919",
    )
    p.add_argument(
        "--stress-policy",
        choices=("full-all", "main-full-others-key"),
        default="main-full-others-key",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    # Frozen common training hyperparameters.
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--base-lr", type=float, default=3e-5)
    p.add_argument("--new-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lovasz-weight", type=float, default=0.5)
    p.add_argument("--clean-warmup-epochs", type=int, default=10)
    p.add_argument("--corruption-ramp-epochs", type=int, default=20)
    p.add_argument("--max-corruption-prob", type=float, default=0.5)
    p.add_argument("--nir-fog-scatter-ratio", type=float, default=0.65)
    p.add_argument("--train-num-workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--train-log-every", type=int, default=50)

    # RR-specific mechanism settings frozen by the current RR-DARF script.
    p.add_argument("--darf-initial-gate", type=float, default=0.5)
    p.add_argument("--relative-aux-prob", type=float, default=0.20)
    p.add_argument("--relative-aux-start-epoch", type=int, default=10)
    p.add_argument("--relative-min-severity", type=float, default=0.55)
    p.add_argument("--relative-rank-margin", type=float, default=0.08)
    p.add_argument("--relative-rank-weight", type=float, default=0.10)
    p.add_argument("--gate-saturation-weight", type=float, default=0.01)
    p.add_argument("--gate-saturation-free-logit", type=float, default=2.50)

    # Validation runtime only; protocol itself is frozen inside repository validators.
    p.add_argument("--val-batch-size", type=int, default=2)
    p.add_argument("--val-log-every", type=int, default=16)
    p.add_argument("--confusion-chunk-rows", type=int, default=512)
    p.add_argument("--fog-chunk-rows", type=int, default=128)

    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed seed-specific runs with matching protocol seed.",
    )
    p.add_argument(
        "--reuse-original-seed20260917",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the existing formal output directory already contains seed "
            "20260917 final.pt + protocol.json, count it as one seed and do not rerun."
        ),
    )
    p.add_argument("--force-validation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    x = parse_args()
    seeds = parse_seed_list(x.seeds)
    output_root = resolve(x.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "runner_logs"

    fixed_dirs = {
        seed: choose_train_dir(
            model="fixed1",
            seed=seed,
            reuse_original_seed_20260917=x.reuse_original_seed20260917,
        )
        for seed in seeds
    }
    rr_dirs = {
        seed: choose_train_dir(
            model="rr_darf",
            seed=seed,
            reuse_original_seed_20260917=x.reuse_original_seed20260917,
        )
        for seed in seeds
    }
    eval_roots = {
        seed: output_root / "evaluation" / f"seed{seed}"
        for seed in seeds
    }

    manifest = build_manifest(
        x=x,
        seeds=seeds,
        fixed_dirs=fixed_dirs,
        rr_dirs=rr_dirs,
        eval_roots=eval_roots,
    )
    save_json_atomic(output_root / "multiseed_protocol_manifest.json", manifest)

    fixed_ckpts: Dict[int, Path] = {}
    rr_ckpts: Dict[int, Path] = {}

    if x.stage in ("all", "train"):
        for seed in seeds:
            print("\n" + "#" * 132)
            print(f"TRAIN SEED {seed}")
            print("#" * 132)
            fixed_ckpts[seed] = train_one(
                x=x,
                model="fixed1",
                seed=seed,
                train_dir=fixed_dirs[seed],
                log_root=log_root,
            )
            rr_ckpts[seed] = train_one(
                x=x,
                model="rr_darf",
                seed=seed,
                train_dir=rr_dirs[seed],
                log_root=log_root,
            )

        if not x.dry_run:
            audit = {
                "within_fixed1": audit_within_model_only_seed_changes(
                    model="fixed1",
                    seeds=seeds,
                    train_dirs=fixed_dirs,
                ),
                "within_rr_darf": audit_within_model_only_seed_changes(
                    model="rr_darf",
                    seeds=seeds,
                    train_dirs=rr_dirs,
                ),
                "cross_model_common_protocol_by_seed": [
                    audit_cross_model_common_protocol(
                        seed=seed,
                        fixed_dir=fixed_dirs[seed],
                        rr_dir=rr_dirs[seed],
                    )
                    for seed in seeds
                ],
            }
            save_json_atomic(
                output_root / "multiseed_training_protocol_audit.json",
                audit,
            )
            print("[audit] training protocols passed.", flush=True)

    if x.stage in ("all", "validate"):
        for seed in seeds:
            fixed_ckpt = final_checkpoint(fixed_dirs[seed])
            rr_ckpt = final_checkpoint(rr_dirs[seed])
            if not fixed_ckpt.is_file():
                raise FileNotFoundError(fixed_ckpt)
            if not rr_ckpt.is_file():
                raise FileNotFoundError(rr_ckpt)

            print("\n" + "#" * 132)
            print(f"VALIDATE SEED {seed}")
            print("#" * 132)
            if x.stress_policy == "full-all" or seed == seeds[0]:
                run_full_seed_validation(
                    x=x,
                    seed=seed,
                    fixed_ckpt=fixed_ckpt,
                    rr_ckpt=rr_ckpt,
                    eval_root=eval_roots[seed],
                    log_root=log_root,
                )
            else:
                run_resource_aware_seed_validation(
                    x=x,
                    seed=seed,
                    fixed_ckpt=fixed_ckpt,
                    rr_ckpt=rr_ckpt,
                    eval_root=eval_roots[seed],
                    log_root=log_root,
                )

    if x.stage in ("all", "summarize"):
        if x.dry_run:
            print("[dry-run] summary skipped because no outputs were generated.")
            return

        # Audit completed training protocols even when --stage summarize is used.
        audit = {
            "within_fixed1": audit_within_model_only_seed_changes(
                model="fixed1",
                seeds=seeds,
                train_dirs=fixed_dirs,
            ),
            "within_rr_darf": audit_within_model_only_seed_changes(
                model="rr_darf",
                seeds=seeds,
                train_dirs=rr_dirs,
            ),
            "cross_model_common_protocol_by_seed": [
                audit_cross_model_common_protocol(
                    seed=seed,
                    fixed_dir=fixed_dirs[seed],
                    rr_dir=rr_dirs[seed],
                )
                for seed in seeds
            ],
        }
        save_json_atomic(
            output_root / "multiseed_training_protocol_audit.json",
            audit,
        )

        summary = aggregate_multiseed(
            seeds=seeds,
            eval_roots=eval_roots,
            output_root=output_root,
        )
        print("\n" + "=" * 132)
        print("MULTI-SEED SUMMARY COMPLETE")
        print("=" * 132)
        print(f"seeds  : {', '.join(str(s) for s in seeds)}")
        print(f"output : {output_root}")
        for row in summary["summary_rows"]:
            if row["model"] == "rr_minus_fixed1":
                print(
                    f"{row['metric']:<44} "
                    f"RR-Fixed = {row['mean_pm_std']}"
                )
        print("=" * 132)


if __name__ == "__main__":
    main()
