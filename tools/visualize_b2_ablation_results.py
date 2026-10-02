#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate and visualize the formal Potsdam ablation chain:

M0 = B2-RGB-Clean
M1 = B2-RGB-Robust4
M2 = B2-RGBNIR-Fixed-Robust4
M3 = B2-DARF-Robust4

Run:
    python tools/visualize_b2_ablation_results.py

Outputs:
    outputs/visualization/b2_robust4_ablation/
        tables/
        figures/

Required summaries:
    outputs/evaluation/b2_rgb_clean/all_conditions_summary.json
    outputs/evaluation/b2_rgb_robust4/all_conditions_summary.json
    outputs/evaluation/b2_rgbnir_fixed_robust4/all_conditions_summary.json
    outputs/evaluation/b2_darf_robust4/all_conditions_summary.json

Optional:
    outputs/evaluation/b2_darf_robust4/gate_statistics.csv
    outputs/training/b2_darf_robust4/train_log.jsonl
    M2/M3 per-condition metrics.json files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "outputs" / "visualization" / "b2_robust4_ablation"

MODELS = OrderedDict([
    ("M0", {
        "name": "B2-RGB-Clean",
        "summary": ROOT / "outputs/evaluation/b2_rgb_clean/all_conditions_summary.json",
        "eval": ROOT / "outputs/evaluation/b2_rgb_clean",
    }),
    ("M1", {
        "name": "B2-RGB-Robust4",
        "summary": ROOT / "outputs/evaluation/b2_rgb_robust4/all_conditions_summary.json",
        "eval": ROOT / "outputs/evaluation/b2_rgb_robust4",
    }),
    ("M2", {
        "name": "B2-RGBNIR-Fixed-Robust4",
        "summary": ROOT / "outputs/evaluation/b2_rgbnir_fixed_robust4/all_conditions_summary.json",
        "eval": ROOT / "outputs/evaluation/b2_rgbnir_fixed_robust4",
    }),
    ("M3", {
        "name": "B2-DARF-Robust4",
        "summary": ROOT / "outputs/evaluation/b2_darf_robust4/all_conditions_summary.json",
        "eval": ROOT / "outputs/evaluation/b2_darf_robust4",
    }),
])

M3_EVAL = MODELS["M3"]["eval"]
M3_LOG = ROOT / "outputs/training/b2_darf_robust4/train_log.jsonl"

CONDS = [
    "Clean",
    "gaussian_noise_L1", "gaussian_noise_L2", "gaussian_noise_L3",
    "gaussian_blur_L1", "gaussian_blur_L2", "gaussian_blur_L3",
    "rgb_underexposure_L1", "rgb_underexposure_L2", "rgb_underexposure_L3",
    "fog_L1", "fog_L2", "fog_L3",
]

LABEL = {
    "Clean": "Clean",
    "gaussian_noise_L1": "Noise L1",
    "gaussian_noise_L2": "Noise L2",
    "gaussian_noise_L3": "Noise L3",
    "gaussian_blur_L1": "Blur L1",
    "gaussian_blur_L2": "Blur L2",
    "gaussian_blur_L3": "Blur L3",
    "rgb_underexposure_L1": "Underexp. L1",
    "rgb_underexposure_L2": "Underexp. L2",
    "rgb_underexposure_L3": "Underexp. L3",
    "fog_L1": "Fog L1",
    "fog_L2": "Fog L2",
    "fog_L3": "Fog L3",
}

L3 = [
    "gaussian_noise_L3",
    "gaussian_blur_L3",
    "rgb_underexposure_L3",
    "fog_L3",
]

GATE_FAMILIES = OrderedDict([
    ("gaussian_noise", ["Clean", "gaussian_noise_L1", "gaussian_noise_L2", "gaussian_noise_L3"]),
    ("gaussian_blur", ["Clean", "gaussian_blur_L1", "gaussian_blur_L2", "gaussian_blur_L3"]),
    ("rgb_underexposure", ["Clean", "rgb_underexposure_L1", "rgb_underexposure_L2", "rgb_underexposure_L3"]),
    ("fog", ["Clean", "fog_L1", "fog_L2", "fog_L3"]),
])

GATE_TITLES = {
    "gaussian_noise": "DARF gate response under Gaussian noise",
    "gaussian_blur": "DARF gate response under Gaussian blur",
    "rgb_underexposure": "DARF gate response under RGB underexposure",
    "fog": "DARF gate response under atmospheric fog",
}

CLASS_ORDER = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]


def args():
    p = argparse.ArgumentParser(
        description="Create paper-ready visualizations for M0-M3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--show", action="store_true")
    x = p.parse_args()
    if x.dpi <= 0:
        p.error("--dpi must be > 0")
    return x


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL {path}:{i}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def fnum(v: Any):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def summary_map(path: Path):
    obj = read_json(path)
    rows = obj.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(f"No results list: {path}")
    out = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        c = row.get("condition")
        m = fnum(row.get("miou"))
        if c is not None and m is not None:
            out[str(c)] = dict(row)
    missing = [c for c in CONDS if c not in out]
    if missing:
        raise RuntimeError(f"Missing conditions in {path}: {missing}")
    return out


def save_fig(fig, base: Path, dpi: int, pdf: bool, show: bool):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("[figure]", base.with_suffix(".png"))


def style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def integrate(summaries, table_dir: Path):
    matrix = []
    for c in CONDS:
        row = {"condition": c, "label": LABEL[c]}
        for mid in MODELS:
            row[f"{mid}_miou"] = float(summaries[mid][c]["miou"])
        matrix.append(row)
    write_csv(
        table_dir / "model_condition_matrix.csv",
        matrix,
        ["condition", "label", "M0_miou", "M1_miou", "M2_miou", "M3_miou"],
    )

    degraded = [c for c in CONDS if c != "Clean"]
    ablation = []
    for mid, spec in MODELS.items():
        clean = float(summaries[mid]["Clean"]["miou"])
        vals = np.array([float(summaries[mid][c]["miou"]) for c in degraded])
        l3 = np.array([float(summaries[mid][c]["miou"]) for c in L3])
        ablation.append({
            "model_id": mid,
            "model_name": spec["name"],
            "clean_miou": clean,
            "mean_12_degraded_miou": float(vals.mean()),
            "mean_12_degraded_drop": float(clean - vals.mean()),
            "mean_12_degraded_retention_pct": float(100.0 * vals.mean() / clean),
            "mean_4_L3_miou": float(l3.mean()),
            "mean_4_L3_drop": float(clean - l3.mean()),
            "mean_4_L3_retention_pct": float(100.0 * l3.mean() / clean),
        })
    write_csv(
        table_dir / "main_ablation_summary.csv",
        ablation,
        [
            "model_id", "model_name", "clean_miou",
            "mean_12_degraded_miou", "mean_12_degraded_drop",
            "mean_12_degraded_retention_pct",
            "mean_4_L3_miou", "mean_4_L3_drop",
            "mean_4_L3_retention_pct",
        ],
    )

    gains = []
    for c in CONDS:
        m2 = float(summaries["M2"][c]["miou"])
        m3 = float(summaries["M3"][c]["miou"])
        gains.append({
            "condition": c,
            "label": LABEL[c],
            "m2_miou": m2,
            "m3_miou": m3,
            "m3_minus_m2": m3 - m2,
            "relative_gain_pct": 100.0 * (m3 - m2) / m2 if m2 else None,
        })
    write_csv(
        table_dir / "m3_vs_m2_gain.csv",
        gains,
        ["condition", "label", "m2_miou", "m3_miou", "m3_minus_m2", "relative_gain_pct"],
    )
    return ablation, gains


def fig_all(summaries, fig_dir, x):
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    xx = np.arange(len(CONDS))
    marks = ["o", "s", "^", "D"]
    for i, (mid, spec) in enumerate(MODELS.items()):
        yy = [float(summaries[mid][c]["miou"]) for c in CONDS]
        ax.plot(xx, yy, marker=marks[i], linewidth=1.8, markersize=5, label=f"{mid}: {spec['name']}")
    ax.set_title("Segmentation robustness across all validation conditions")
    ax.set_ylabel("mIoU")
    ax.set_xticks(xx, [LABEL[c] for c in CONDS], rotation=40, ha="right")
    ax.set_ylim(0.0, 0.82)
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig01_all_conditions_miou", x.dpi, not x.no_pdf, x.show)


def fig_l3(summaries, fig_dir, x):
    conds = ["Clean"] + L3
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    xx = np.arange(len(conds))
    width = 0.18
    offsets = (np.arange(4) - 1.5) * width
    for off, (mid, spec) in zip(offsets, MODELS.items()):
        yy = [float(summaries[mid][c]["miou"]) for c in conds]
        bars = ax.bar(xx + off, yy, width=width, label=f"{mid}: {spec['name']}")
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_title("Clean and severe L3 degradation performance")
    ax.set_ylabel("mIoU")
    ax.set_xticks(xx, [LABEL[c] for c in conds])
    ax.set_ylim(0.0, 0.84)
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig02_severe_l3_comparison", x.dpi, not x.no_pdf, x.show)


def fig_mean(ablation, fig_dir, x):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    xx = np.arange(len(ablation))
    width = 0.36
    clean = [float(r["clean_miou"]) for r in ablation]
    deg = [float(r["mean_12_degraded_miou"]) for r in ablation]
    b1 = ax.bar(xx - width/2, clean, width=width, label="Clean")
    b2 = ax.bar(xx + width/2, deg, width=width, label="Mean of 12 degraded conditions")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_title("Clean accuracy versus average robustness")
    ax.set_ylabel("mIoU")
    ax.set_xticks(xx, [r["model_id"] for r in ablation])
    ax.set_ylim(0.0, 0.82)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig03_clean_vs_mean_degraded", x.dpi, not x.no_pdf, x.show)


def fig_gain(gains, fig_dir, x):
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    xx = np.arange(len(gains))
    yy = [float(r["m3_minus_m2"]) for r in gains]
    bars = ax.bar(xx, yy)
    ax.axhline(0.0, linewidth=1)
    ax.bar_label(bars, fmt="%+.3f", fontsize=7, padding=2)
    ax.set_title("Adaptive DARF contribution relative to fixed NIR fusion")
    ax.set_ylabel("mIoU gain (M3 - M2)")
    ax.set_xticks(xx, [r["label"] for r in gains], rotation=40, ha="right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig04_m3_vs_m2_gain", x.dpi, not x.no_pdf, x.show)


def gate_rows():
    path = M3_EVAL / "gate_statistics.csv"
    rows = read_csv(path)
    parsed = []
    for r in rows:
        s = fnum(r.get("scale"))
        m = fnum(r.get("g_nir_mean"))
        if r.get("condition") and s is not None and m is not None:
            parsed.append({"condition": str(r["condition"]), "scale": int(s), "mean": m})
    return parsed


def fig_gates(rows, fig_dir, x):
    if not rows:
        print("[skip] gate_statistics.csv not found or empty", file=sys.stderr)
        return
    lookup = {(r["condition"], r["scale"]): r["mean"] for r in rows}
    for num, (fam, conds) in enumerate(GATE_FAMILIES.items(), start=5):
        if any((c, s) not in lookup for c in conds for s in range(1, 5)):
            print(f"[skip] incomplete Gate rows for {fam}", file=sys.stderr)
            continue
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        xx = np.arange(4)
        for s in range(1, 5):
            yy = [lookup[(c, s)] for c in conds]
            ax.plot(xx, yy, marker="o", linewidth=1.9, markersize=6, label=f"Scale {s}")
        ax.set_title(GATE_TITLES[fam])
        ax.set_ylabel("Mean predicted NIR gate strength")
        ax.set_xticks(xx, ["Clean", "L1", "L2", "L3"])
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        save_fig(fig, fig_dir / f"fig{num:02d}_gate_{fam}", x.dpi, not x.no_pdf, x.show)


def epoch_rows():
    rows = [r for r in read_jsonl(M3_LOG) if r.get("event") == "epoch_done"]
    rows.sort(key=lambda r: int(r.get("epoch", 0)))
    return rows


def fig_train(rows, fig_dir, x):
    if not rows:
        print("[skip] M3 train_log.jsonl not found or empty", file=sys.stderr)
        return

    epochs = [int(r["epoch"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for label, key in [
        ("Total loss", "train_total_loss_mean"),
        ("Cross entropy", "train_ce"),
        ("Lovasz", "train_lovasz_mean"),
        ("Gate BCE", "train_gate_bce_mean"),
    ]:
        yy = [fnum(r.get(key)) for r in rows]
        if any(v is not None for v in yy):
            ax.plot(epochs, [np.nan if v is None else v for v in yy], linewidth=1.7, label=label)
    ax.set_title("M3 training loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig09_m3_training_loss", x.dpi, not x.no_pdf, x.show)

    valid = [
        r for r in rows
        if isinstance(r.get("gate_nir_strength_mean_by_scale"), list)
        and len(r["gate_nir_strength_mean_by_scale"]) == 4
    ]
    if not valid:
        return
    ex = [int(r["epoch"]) for r in valid]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for s in range(4):
        yy = [float(r["gate_nir_strength_mean_by_scale"][s]) for r in valid]
        ax.plot(ex, yy, linewidth=1.7, label=f"Scale {s+1}")
    targets = [fnum(r.get("gate_target_mean")) for r in valid]
    if all(v is not None for v in targets):
        ax.plot(ex, [float(v) for v in targets], linestyle="--", linewidth=1.8, label="Mean gate target")
    ax.set_title("M3 Gate learning during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean NIR gate strength")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_fig(fig, fig_dir / "fig10_m3_training_gate", x.dpi, not x.no_pdf, x.show)


def metrics_path(eval_root: Path, condition: str) -> Path:
    if condition == "Clean":
        return eval_root / "clean_val" / "metrics.json"
    if condition.startswith("fog_"):
        return eval_root / "fog_seen_val" / condition / "metrics.json"
    return eval_root / "robustness_val_v2" / condition / "metrics.json"


def class_map(path: Path):
    obj = read_json(path)
    rows = obj.get("per_class")
    if not isinstance(rows, list):
        return {}
    out = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        name = r.get("class_name")
        iou = fnum(r.get("iou"))
        if name is not None and iou is not None:
            out[str(name)] = iou
    return out


def fig_classes(fig_dir, x):
    m2root = Path(MODELS["M2"]["eval"])
    m3root = Path(MODELS["M3"]["eval"])

    for num, cond in enumerate(L3, start=11):
        p2 = metrics_path(m2root, cond)
        p3 = metrics_path(m3root, cond)
        if not p2.is_file() or not p3.is_file():
            print(f"[skip] per-class metrics missing for {cond}", file=sys.stderr)
            continue

        m2 = class_map(p2)
        m3 = class_map(p3)
        common = set(m2) & set(m3)
        classes = [c for c in CLASS_ORDER if c in common] + sorted(common - set(CLASS_ORDER))
        if not classes:
            continue

        fig, ax = plt.subplots(figsize=(10.2, 5.8))
        xx = np.arange(len(classes))
        width = 0.36
        y2 = [m2[c] for c in classes]
        y3 = [m3[c] for c in classes]
        b2 = ax.bar(xx - width/2, y2, width=width, label="M2 Fixed fusion")
        b3 = ax.bar(xx + width/2, y3, width=width, label="M3 DARF")
        ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
        ax.bar_label(b3, fmt="%.3f", fontsize=7, padding=2)
        ax.set_title(f"Per-class IoU under {LABEL[cond]}")
        ax.set_ylabel("IoU")
        ax.set_xticks(xx, classes, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        short = LABEL[cond].lower().replace(" ", "_").replace(".", "")
        save_fig(fig, fig_dir / f"fig{num:02d}_per_class_{short}", x.dpi, not x.no_pdf, x.show)


def main():
    x = args()
    out = x.output_root.expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()

    table_dir = out / "tables"
    fig_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    style()

    summaries = {
        mid: summary_map(Path(spec["summary"]))
        for mid, spec in MODELS.items()
    }

    ablation, gains = integrate(summaries, table_dir)

    fig_all(summaries, fig_dir, x)
    fig_l3(summaries, fig_dir, x)
    fig_mean(ablation, fig_dir, x)
    fig_gain(gains, fig_dir, x)
    fig_gates(gate_rows(), fig_dir, x)
    fig_train(epoch_rows(), fig_dir, x)
    fig_classes(fig_dir, x)

    print()
    print("=" * 88)
    print("REPORT INTEGRATION CHECKLIST")
    print("=" * 88)
    print("1. M0 -> M1: quantify the contribution of Robust-4 training.")
    print("2. M1 -> M2: quantify the contribution of adding NIR.")
    print("3. M2 -> M3: quantify the contribution of adaptive DARF gating.")
    print("4. Report Clean, mean of 12 degraded conditions, and mean of four L3 conditions.")
    print("5. Use Gate curves to show whether NIR correction increases with degradation severity.")
    print("6. Use L3 per-class plots to explain which land-cover classes benefit most from DARF.")
    print("7. Use M3 training loss/Gate curves only as convergence and mechanism evidence.")
    print("8. For qualitative masks, rerun selected M2/M3 validations with --save-predictions.")
    print("-" * 88)

    for row in ablation:
        print(
            f"{row['model_id']}: "
            f"Clean={float(row['clean_miou']):.6f} | "
            f"Mean degraded={float(row['mean_12_degraded_miou']):.6f} | "
            f"Mean L3={float(row['mean_4_L3_miou']):.6f}"
        )

    degraded_gain = [
        float(r["m3_minus_m2"])
        for r in gains
        if r["condition"] != "Clean"
    ]
    print(
        "M3-M2 mean gain over 12 degraded conditions: "
        f"{float(np.mean(degraded_gain)):+.6f} mIoU"
    )
    print("=" * 88)
    print("[done] tables :", table_dir)
    print("[done] figures:", fig_dir)


if __name__ == "__main__":
    main()
