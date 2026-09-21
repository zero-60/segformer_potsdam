#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final validation for Model C / Quality Gate / Ours.

Runs:
  Clean
  Gaussian Noise L1/L2/L3
  Gaussian Blur L1/L2/L3
  RGB Underexposure L1/L2/L3

Uses the same 6-tile / 256-window / full-tile mean-logit / GLOBAL confusion
matrix protocol as Models A, B and C-noGate.

Also records every validation-window gate weight at all four scales and creates:
  - four_model_comparison.{json,csv}
  - per_class_four_model_comparison.csv
  - gate_statistics.{json,csv}
  - gate_monotonicity.json
  - rq4_rq5_rq6_summary.json

Expected location:
  tools/validate_model_c_quality_gate_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset
from evaluation.rgb_degradation_protocol import (
    DEGRADATION_PROTOCOL_VERSION,
    IMPLEMENTATION_REVISION,
    DegradedPotsdamSlidingWindowDataset,
    condition_name,
    condition_spec,
    degradation_protocol_sha256,
    write_degradation_protocol,
)
from models.segformer_dual_quality_gate import (
    GATE_TYPE,
    MODEL_ID,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_SCALES,
    build_model_c_quality_gate,
)
from validate_model_a_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    confusion_from_prediction,
    metrics_from_confusion,
    write_confusion_csv,
    write_per_class_csv,
)

A_ID = "A_RGB"
B_ID = "B_RGBNIR_4CH"
FIXED_ID = "C_NOGATE_FIXED_FUSION"

DEFAULT_CKPT = PROJECT_ROOT / "outputs/training/model_c_quality_gate/checkpoints/final.pt"
DEFAULT_OUT = PROJECT_ROOT / "outputs/evaluation/model_c_quality_gate"

A_CLEAN = PROJECT_ROOT / "outputs/evaluation/model_a_rgb/clean_val/metrics.json"
A_V1 = PROJECT_ROOT / "outputs/evaluation/model_a_rgb/robustness_val"
A_V2 = PROJECT_ROOT / "outputs/evaluation/model_a_rgb/robustness_val_v2"
B_CLEAN = PROJECT_ROOT / "outputs/evaluation/model_b_rgbnir/clean_val/metrics.json"
B_V2 = PROJECT_ROOT / "outputs/evaluation/model_b_rgbnir/robustness_val_v2"
F_CLEAN = PROJECT_ROOT / "outputs/evaluation/model_c_nogate/clean_val/metrics.json"
F_V2 = PROJECT_ROOT / "outputs/evaluation/model_c_nogate/robustness_val_v2"

CONDS = [
    ("gaussian_noise", "L1"),
    ("gaussian_noise", "L2"),
    ("gaussian_noise", "L3"),
    ("gaussian_blur", "L1"),
    ("gaussian_blur", "L2"),
    ("gaussian_blur", "L3"),
    ("rgb_underexposure", "L1"),
    ("rgb_underexposure", "L2"),
    ("rgb_underexposure", "L3"),
]


def args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=8)
    p.add_argument("--confusion-chunk-rows", type=int, default=512)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force-clean", action="store_true")
    p.add_argument("--force-robustness", action="store_true")
    x = p.parse_args()
    if x.batch_size <= 0 or x.log_every <= 0 or x.confusion_chunk_rows <= 0:
        p.error("batch-size, log-every and confusion-chunk-rows must be > 0")
    return x


def resolve(p: Path) -> Path:
    p = p.expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def loadj(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise TypeError(f"Expected JSON object: {p}")
    return x


def savej(p: Path, x: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, default=str) + "\n",
                 encoding="utf-8")


def appendj(p: Path, x: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")


def device_of(s: str) -> torch.device:
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return d


def validate_refs(a_clean, a_sum, b_clean, b_sum, f_clean, f_sum):
    for label, clean, mid in [
        ("A", a_clean, A_ID), ("B", b_clean, B_ID), ("C-noGate", f_clean, FIXED_ID)
    ]:
        if clean.get("model") != mid or clean.get("condition") != "Clean" or clean.get("split") != "val":
            raise RuntimeError(f"{label} Clean reference invalid.")
    for label, s, mid in [
        ("A", a_sum, A_ID), ("B", b_sum, B_ID), ("C-noGate", f_sum, FIXED_ID)
    ]:
        if s.get("model") != mid:
            raise RuntimeError(f"{label} summary model mismatch.")
        if s.get("degradation_protocol_sha256") != degradation_protocol_sha256():
            raise RuntimeError(f"{label} summary does not use FINAL v2 protocol hash.")
        if int(s.get("degradation_implementation_revision", -1)) != IMPLEMENTATION_REVISION:
            raise RuntimeError(f"{label} protocol implementation revision mismatch.")


def unwrap_ckpt(x):
    if not isinstance(x, Mapping) or "model" not in x:
        raise RuntimeError("Invalid Model C checkpoint.")
    meta = {k: v for k, v in x.items() if k != "model"}
    protocol = meta.get("protocol", {})
    if meta.get("model_id") not in (None, MODEL_ID):
        raise RuntimeError("Checkpoint model_id mismatch.")
    if protocol.get("model") != MODEL_ID or not protocol.get("quality_gate") or not protocol.get("dual_encoder"):
        raise RuntimeError("Checkpoint protocol is not Model C / Quality Gate.")
    return x["model"], meta


def gate_l2(model) -> float:
    vals = [p.detach().float().cpu() for n, p in model.named_parameters()
            if n.startswith("quality_gates.")]
    if not vals:
        raise RuntimeError("No Quality Gate parameters found.")
    return float(torch.sqrt(sum((v.double() ** 2).sum() for v in vals)).item())


class GateStats:
    def __init__(self):
        self.v = [[] for _ in range(NUM_SCALES)]

    def add(self, w: torch.Tensor):
        a = w.detach().float().cpu().numpy()
        if a.ndim != 3 or a.shape[1:] != (NUM_SCALES, 2):
            raise RuntimeError(f"Bad gate tensor shape {a.shape}")
        if np.max(np.abs(a.sum(-1) - 1.0)) > 1e-6:
            raise RuntimeError("Gate weights do not sum to 1.")
        for s in range(NUM_SCALES):
            self.v[s].extend(a[:, s, 0].astype(np.float64).tolist())

    def rows(self, condition, corruption, level, rank):
        out = []
        for s, vals in enumerate(self.v, 1):
            a = np.asarray(vals, dtype=np.float64)
            if a.size == 0:
                raise RuntimeError("No gate samples accumulated.")
            q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
            out.append({
                "condition": condition, "corruption": corruption,
                "severity_level": level, "severity_rank": rank, "scale": s,
                "count": int(a.size), "w_rgb_mean": float(a.mean()),
                "w_rgb_std": float(a.std()), "w_rgb_min": float(a.min()),
                "w_rgb_p05": float(q[0]), "w_rgb_p25": float(q[1]),
                "w_rgb_median": float(q[2]), "w_rgb_p75": float(q[3]),
                "w_rgb_p95": float(q[4]), "w_rgb_max": float(a.max()),
                "w_nir_mean": float(1.0 - a.mean()),
            })
        return out


GATE_FIELDS = [
    "condition","corruption","severity_level","tile_id","window_index","x","y",
    "scale","w_rgb","w_nir","gate_rgb_logit"
]


def gate_writer(path: Path):
    f = path.open("w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=GATE_FIELDS)
    w.writeheader()
    return f, w


def write_gate_batch(wr, condition, corruption, level, batch, weights, logits):
    ww = weights.detach().float().cpu().numpy()
    zz = logits.detach().float().cpu().numpy()
    for i in range(ww.shape[0]):
        for s in range(NUM_SCALES):
            wr.writerow({
                "condition": condition, "corruption": corruption, "severity_level": level,
                "tile_id": str(batch["tile_id"][i]),
                "window_index": int(batch["window_index"][i]),
                "x": int(batch["x"][i]), "y": int(batch["y"][i]),
                "scale": s + 1, "w_rgb": float(ww[i,s,0]),
                "w_nir": float(ww[i,s,1]), "gate_rgb_logit": float(zz[i,s]),
            })


def probe(clean_ds, degraded_ds):
    a, b = clean_ds[0], degraded_ds[0]
    for k in ("tile_id","window_index","x","y","height","width"):
        if a[k] != b[k]:
            raise RuntimeError(f"Degradation changed {k}.")
    if not torch.equal(a["nir"], b["nir"]):
        raise RuntimeError("NIR changed under RGB-only degradation.")
    if torch.equal(a["rgb"], b["rgb"]):
        raise RuntimeError("Degraded RGB equals Clean RGB.")
    print("[degradation probe] PASS | NIR bit-identical | RGB changed")


def infer_tile(model, ds, tile_id, tile_idx, batch_size, dev, amp, log_every,
               condition, corruption, level, gw, gs):
    nwin = len(ds.window_coordinates)
    subset = Subset(ds, range(tile_idx*nwin, (tile_idx+1)*nwin))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=False)
    ts, cs = int(ds.spec.tile_size), int(ds.spec.crop_size)
    logits_sum = torch.zeros((NUM_CLASSES, ts, ts), dtype=torch.float32, device=dev)
    coverage = torch.zeros((ts, ts), dtype=torch.float32, device=dev)
    seen, start = 0, time.time()

    for bi, batch in enumerate(loader):
        if any(str(t) != tile_id for t in list(batch["tile_id"])):
            raise RuntimeError("Per-tile loader mixed tile ids.")
        rgb = batch["rgb"].to(dev, non_blocking=(dev.type=="cuda"))
        nir = batch["nir"].to(dev, non_blocking=(dev.type=="cuda"))

        with torch.inference_mode():
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                d = model(rgb, nir, return_details=True)

        logits = d["logits"]
        weights = d["fusion_weights"]
        z = d["gate_rgb_logits"]

        if tuple(logits.shape[-2:]) != (cs,cs) or logits.shape[1] != NUM_CLASSES:
            raise RuntimeError(f"Bad logits shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Non-finite logits.")

        gs.add(weights)
        write_gate_batch(gw, condition, corruption, level, batch, weights, z)
        logits = logits.float()

        for i in range(logits.shape[0]):
            x, y = int(batch["x"][i]), int(batch["y"][i])
            logits_sum[:, y:y+cs, x:x+cs].add_(logits[i])
            coverage[y:y+cs, x:x+cs].add_(1.0)
        seen += int(logits.shape[0])

        if bi % log_every == 0 or bi + 1 == len(loader):
            m = weights[:,:,0].detach().float().mean(0).cpu().tolist()
            print(f"  {tile_id} batch {bi+1:03d}/{len(loader):03d} | "
                  f"windows {seen:03d}/{nwin:03d} | wRGB={[round(v,4) for v in m]}")
        del d, rgb, nir, logits, weights, z

    if seen != nwin or float(coverage.min().item()) <= 0:
        raise RuntimeError("Window coverage failure.")
    logits_sum.div_(coverage.unsqueeze(0))
    pred = logits_sum.argmax(0).to(torch.uint8).cpu().numpy()
    rt = {
        "tile_id": tile_id, "windows": seen,
        "coverage_min": float(coverage.min().item()),
        "coverage_max": float(coverage.max().item()),
        "inference_seconds": time.time() - start,
    }
    del logits_sum, coverage
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return pred, rt


def evaluate(model, ds, condition, corruption, level, rank, out, ckpt, epoch, step,
             batch_size, dev, amp, log_every, cm_rows, save_preds):
    out.mkdir(parents=True, exist_ok=True)
    ptile = out / "per_tile_metrics.jsonl"
    if ptile.exists():
        ptile.unlink()
    pdir = out / "predictions"
    if save_preds:
        pdir.mkdir(exist_ok=True)

    gf, gw = gate_writer(out / "gate_weights_windows.csv")
    gs = GateStats()
    global_cm = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int64)
    tile_rows, start = [], time.time()
    tile_ids = list(ds.tile_ids)
    if len(tile_ids) != 6 or len(ds.window_coordinates) != 256:
        gf.close()
        raise RuntimeError("Frozen validation protocol changed.")

    try:
        for ti, tid in enumerate(tile_ids):
            print(f"[{condition}] tile {ti+1}/6 | {tid}")
            pred, rt = infer_tile(
                model, ds, tid, ti, batch_size, dev, amp, log_every,
                condition, corruption, level, gw, gs
            )
            target_t = ds.load_full_label(tid)
            target = target_t.numpy()
            cm = confusion_from_prediction(
                pred, target, num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX, chunk_rows=cm_rows
            )
            global_cm += cm
            tm = metrics_from_confusion(cm, CLASS_NAMES)
            rec = {
                **rt, "condition": condition, "corruption": corruption,
                "severity_level": level, "miou": tm["miou"],
                "pixel_accuracy": tm["pixel_accuracy"],
                "mean_class_accuracy": tm["mean_class_accuracy"],
                "valid_pixels": tm["valid_pixels"],
                "per_class_iou": {r["class_name"]:r["iou"] for r in tm["per_class"]},
            }
            tile_rows.append(rec); appendj(ptile, rec)
            if save_preds:
                np.save(pdir / f"{tid}_pred.npy", pred, allow_pickle=False)
            print(f"  tile mIoU={tm['miou']:.6f} | time={rt['inference_seconds']:.1f}s")
            if isinstance(ds, DegradedPotsdamSlidingWindowDataset):
                ds.clear_degradation_cache()
            del pred, target, target_t
    finally:
        gf.close()

    gm = metrics_from_confusion(global_cm, CLASS_NAMES)
    gate_rows = gs.rows(condition, corruption, level, rank)
    result = {
        "model": MODEL_ID, "model_name": MODEL_NAME, "condition": condition,
        "corruption": corruption, "severity_level": level, "severity_rank": rank,
        "split": "val",
        "metric_scope": "GLOBAL confusion matrix after full-tile mean-logit fusion",
        "checkpoint": str(ckpt), "checkpoint_epoch_zero_based": epoch,
        "checkpoint_global_step": step, "quality_gate": True, "gate_type": GATE_TYPE,
        "miou": gm["miou"], "pixel_accuracy": gm["pixel_accuracy"],
        "mean_class_accuracy": gm["mean_class_accuracy"],
        "valid_pixels": gm["valid_pixels"], "per_class": gm["per_class"],
        "confusion_matrix": global_cm.tolist(), "tiles": tile_rows,
        "num_tiles": 6, "windows_per_tile": 256, "total_windows": len(ds),
        "validation_seconds": time.time()-start, "gate_statistics": gate_rows,
        "gate_weights_windows_csv": str(out / "gate_weights_windows.csv"),
    }
    if condition != "Clean":
        result.update({
            "severity_parameters": condition_spec(corruption, level),
            "degradation_protocol_version": DEGRADATION_PROTOCOL_VERSION,
            "degradation_implementation_revision": IMPLEMENTATION_REVISION,
            "degradation_protocol_sha256": degradation_protocol_sha256(),
            "rgb_degraded": True, "nir_source": "clean / unchanged",
        })
    savej(out/"metrics.json", result)
    write_per_class_csv(out/"per_class_metrics.csv", gm["per_class"])
    write_confusion_csv(out/"confusion_matrix.csv", global_cm, CLASS_NAMES)
    return result, gate_rows


def compatible(path, condition, ckpt, step, degraded):
    if not path.is_file():
        return None
    try:
        x = loadj(path)
    except Exception:
        return None
    ok = (
        x.get("model")==MODEL_ID and x.get("condition")==condition
        and x.get("split")=="val" and Path(str(x.get("checkpoint",""))).name==ckpt.name
        and isinstance(x.get("gate_statistics"), list)
    )
    if step is not None:
        ok = ok and int(x.get("checkpoint_global_step",-1)) == int(step)
    if degraded:
        ok = ok and x.get("degradation_protocol_sha256")==degradation_protocol_sha256()
        ok = ok and int(x.get("degradation_implementation_revision",-1))==IMPLEMENTATION_REVISION
    return x if ok else None


def summary_map(s, expected):
    if s.get("model") != expected:
        raise RuntimeError("Reference summary model mismatch.")
    rows = {str(r["condition"]):dict(r) for r in s["results"]}
    expected_conds = {condition_name(c,l) for c,l in CONDS}
    if set(rows) != expected_conds:
        raise RuntimeError("Reference summary condition mismatch.")
    return rows


def detail_a(cond):
    if cond=="Clean": return A_CLEAN
    if cond.startswith("gaussian_noise_") or cond.startswith("gaussian_blur_"):
        return A_V1/cond/"metrics.json"
    return A_V2/cond/"metrics.json"


def detail_std(cond, clean, v2):
    return clean if cond=="Clean" else v2/cond/"metrics.json"


def perclass(x):
    return {int(r["class_id"]):r for r in x["per_class"]}


def four_model(a_clean,a_sum,b_clean,b_sum,f_clean,f_sum,c_clean,cres):
    am,bm,fm = summary_map(a_sum,A_ID),summary_map(b_sum,B_ID),summary_map(f_sum,FIXED_ID)
    ac,bc,fc,cc = [float(x["miou"]) for x in (a_clean,b_clean,f_clean,c_clean)]
    rows=[{
        "condition":"Clean","corruption":"Clean","severity_level":"","severity_rank":0,
        "model_a_miou":ac,"model_b_miou":bc,"model_c_nogate_miou":fc,"model_c_miou":cc,
        "c_gain_vs_a":cc-ac,"c_gain_vs_b":cc-bc,"c_gain_vs_c_nogate":cc-fc,
        "model_a_drop":0.0,"model_b_drop":0.0,"model_c_nogate_drop":0.0,"model_c_drop":0.0,
        "c_drop_reduction_vs_a":0.0,"c_drop_reduction_vs_b":0.0,
        "c_drop_reduction_vs_c_nogate":0.0,
    }]
    for corr,lev in CONDS:
        cond=condition_name(corr,lev)
        av,bv,fv,cv = [float(x) for x in (
            am[cond]["miou"],bm[cond]["miou"],fm[cond]["miou"],cres[cond]["miou"]
        )]
        ad,bd,fd,cd = ac-av,bc-bv,fc-fv,cc-cv
        rows.append({
            "condition":cond,"corruption":corr,"severity_level":lev,"severity_rank":int(lev[1:]),
            "model_a_miou":av,"model_b_miou":bv,"model_c_nogate_miou":fv,"model_c_miou":cv,
            "c_gain_vs_a":cv-av,"c_gain_vs_b":cv-bv,"c_gain_vs_c_nogate":cv-fv,
            "model_a_drop":ad,"model_b_drop":bd,"model_c_nogate_drop":fd,"model_c_drop":cd,
            "c_drop_reduction_vs_a":ad-cd,"c_drop_reduction_vs_b":bd-cd,
            "c_drop_reduction_vs_c_nogate":fd-cd,
        })
    deg=rows[1:]
    mean_drop={
        "A":float(np.mean([r["model_a_drop"] for r in deg])),
        "B":float(np.mean([r["model_b_drop"] for r in deg])),
        "C_noGate":float(np.mean([r["model_c_nogate_drop"] for r in deg])),
        "C":float(np.mean([r["model_c_drop"] for r in deg])),
    }
    counts={k:0 for k in mean_drop}
    for r in deg:
        d={"A":r["model_a_drop"],"B":r["model_b_drop"],
           "C_noGate":r["model_c_nogate_drop"],"C":r["model_c_drop"]}
        mn=min(d.values())
        for k,v in d.items():
            if math.isclose(v,mn,abs_tol=1e-12,rel_tol=0):
                counts[k]+=1
    obj={
        "comparison":"A vs B vs C-noGate vs C/Ours",
        "degradation_protocol_version":DEGRADATION_PROTOCOL_VERSION,
        "degradation_implementation_revision":IMPLEMENTATION_REVISION,
        "degradation_protocol_sha256":degradation_protocol_sha256(),
        "clean":{"A":ac,"B":bc,"C_noGate":fc,"C":cc,"C_minus_C_noGate":cc-fc},
        "results":rows,"mean_degraded_drop":mean_drop,
        "smallest_drop_condition_counts":counts,
    }
    return rows,obj


FOUR_FIELDS=[
    "condition","corruption","severity_level","severity_rank",
    "model_a_miou","model_b_miou","model_c_nogate_miou","model_c_miou",
    "c_gain_vs_a","c_gain_vs_b","c_gain_vs_c_nogate",
    "model_a_drop","model_b_drop","model_c_nogate_drop","model_c_drop",
    "c_drop_reduction_vs_a","c_drop_reduction_vs_b","c_drop_reduction_vs_c_nogate"
]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})


def per_class_compare(four_rows,c_clean,cres):
    out=[]
    for r in four_rows:
        cond=r["condition"]
        ax,bx,fx = [loadj(p) for p in (
            detail_a(cond), detail_std(cond,B_CLEAN,B_V2), detail_std(cond,F_CLEAN,F_V2)
        )]
        cx=c_clean if cond=="Clean" else cres[cond]
        aa,bb,ff,cc=perclass(ax),perclass(bx),perclass(fx),perclass(cx)
        for k in range(NUM_CLASSES):
            av,bv,fv,cv=[float(z[k]["iou"]) for z in (aa,bb,ff,cc)]
            out.append({
                "condition":cond,"corruption":r["corruption"],"severity_level":r["severity_level"],
                "class_id":k,"class_name":CLASS_NAMES[k],"model_a_iou":av,"model_b_iou":bv,
                "model_c_nogate_iou":fv,"model_c_iou":cv,"c_gain_vs_a_iou":cv-av,
                "c_gain_vs_b_iou":cv-bv,"c_gain_vs_c_nogate_iou":cv-fv,
            })
    return out


def gate_monotonicity(rows):
    lookup={(r["condition"],int(r["scale"])):r for r in rows}
    out={}
    for corr in ("gaussian_noise","gaussian_blur","rgb_underexposure"):
        seq=["Clean"]+[condition_name(corr,l) for l in ("L1","L2","L3")]
        scales={}
        for s in range(1,NUM_SCALES+1):
            m=[float(lookup[(c,s)]["w_rgb_mean"]) for c in seq]
            mono=all(m[i+1] <= m[i]+1e-12 for i in range(3))
            scales[f"scale_{s}"]={
                "w_rgb_mean":m,"delta_from_clean":[v-m[0] for v in m],
                "monotonic_nonincreasing":mono,
            }
        out[corr]={
            "sequence":seq,"scales":scales,
            "all_scales_monotonic_nonincreasing":all(v["monotonic_nonincreasing"] for v in scales.values()),
        }
    return out


def main():
    x=args()
    ckpt,out=resolve(x.checkpoint),resolve(x.output_root)
    clean_dir,rob_dir=out/"clean_val",out/"robustness_val_v2"

    a_clean,a_sum=loadj(A_CLEAN),loadj(A_V2/"robustness_summary.json")
    b_clean,b_sum=loadj(B_CLEAN),loadj(B_V2/"robustness_summary.json")
    f_clean,f_sum=loadj(F_CLEAN),loadj(F_V2/"robustness_summary.json")
    validate_refs(a_clean,a_sum,b_clean,b_sum,f_clean,f_sum)

    dev=device_of(x.device);amp=(dev.type=="cuda" and not x.no_amp)
    print("="*100)
    print("MODEL C / QUALITY GATE / OURS | FINAL VALIDATION")
    print(f"checkpoint : {ckpt}")
    print(f"protocol   : {DEGRADATION_PROTOCOL_VERSION} rev={IMPLEMENTATION_REVISION}")
    print(f"hash       : {degradation_protocol_sha256()}")
    print("="*100)

    model,_=build_model_c_quality_gate(PROJECT_ROOT)
    state,meta=unwrap_ckpt(torch.load(ckpt,map_location="cpu",weights_only=False))
    model.load_state_dict(state,strict=True)
    gl2=gate_l2(model)
    epoch=meta.get("epoch");step=meta.get("global_step",meta.get("step"))
    print(f"[checkpoint] epoch_zero_based={epoch} | global_step={step} | gate_l2={gl2:.8f}")
    model.to(dev).eval()

    # Clean
    mp=clean_dir/"metrics.json"
    c_clean=None if x.force_clean else compatible(mp,"Clean",ckpt,step,False)
    if c_clean is None:
        ds=PotsdamSlidingWindowDataset(PROJECT_ROOT,split="val")
        c_clean,clean_g=evaluate(
            model,ds,"Clean","Clean","",0,clean_dir,ckpt,epoch,step,
            x.batch_size,dev,amp,x.log_every,x.confusion_chunk_rows,x.save_predictions
        )
    else:
        clean_g=list(c_clean["gate_statistics"])
        print(f"[resume] Clean: {mp}")
    cc=float(c_clean["miou"])
    print(f"[Clean] A={float(a_clean['miou']):.6f} | B={float(b_clean['miou']):.6f} | "
          f"C-noGate={float(f_clean['miou']):.6f} | C={cc:.6f} | "
          f"C-Fixed={cc-float(f_clean['miou']):+.6f}")
    print("[Clean wRGB]",[round(float(r["w_rgb_mean"]),6) for r in clean_g])

    rob_dir.mkdir(parents=True,exist_ok=True)
    write_degradation_protocol(rob_dir/"degradation_protocol.json")
    cres={};allg=[dict(r) for r in clean_g]

    for idx,(corr,lev) in enumerate(CONDS,1):
        cond=condition_name(corr,lev); cdir=rob_dir/cond; mp=cdir/"metrics.json"
        print(f"\n[{idx}/9] {cond} {condition_spec(corr,lev)}")
        r=None if x.force_robustness else compatible(mp,cond,ckpt,step,True)
        if r is None:
            ds=DegradedPotsdamSlidingWindowDataset(
                PROJECT_ROOT,split="val",corruption=corr,level=lev
            )
            clean_probe=PotsdamSlidingWindowDataset(PROJECT_ROOT,split="val")
            probe(clean_probe,ds);del clean_probe
            r,g=evaluate(
                model,ds,cond,corr,lev,int(lev[1:]),cdir,ckpt,epoch,step,
                x.batch_size,dev,amp,x.log_every,x.confusion_chunk_rows,x.save_predictions
            )
        else:
            g=list(r["gate_statistics"]);print(f"[resume] {mp}")
        dm=float(r["miou"]);drop=cc-dm
        r.update({
            "clean_reference_miou":cc,"drop_miou":drop,"delta_miou":-drop,
            "relative_drop_pct":100*drop/cc,"retention_pct":100*dm/cc,
        })
        savej(mp,r);cres[cond]=r;allg.extend(dict(z) for z in g)
        print(f"[{cond}] mIoU={dm:.6f} | Drop={drop:.6f} | "
              f"wRGB={[round(float(z['w_rgb_mean']),4) for z in g]}")

    # C robustness summary
    rr=[]
    for corr,lev in CONDS:
        cond=condition_name(corr,lev);r=cres[cond]
        rr.append({
            "model":MODEL_ID,"condition":cond,"corruption":corr,"severity_level":lev,
            "severity_rank":int(lev[1:]),"clean_miou":cc,"miou":r["miou"],
            "drop_miou":r["drop_miou"],"delta_miou":r["delta_miou"],
            "relative_drop_pct":r["relative_drop_pct"],"retention_pct":r["retention_pct"],
            "pixel_accuracy":r["pixel_accuracy"],"mean_class_accuracy":r["mean_class_accuracy"],
            "validation_seconds":r["validation_seconds"],
        })
    rs={
        "model":MODEL_ID,"model_name":MODEL_NAME,"split":"val",
        "clean_reference":{"metrics_path":str(clean_dir/"metrics.json"),"miou":cc,
                           "checkpoint_global_step":step},
        "degradation_protocol_version":DEGRADATION_PROTOCOL_VERSION,
        "degradation_implementation_revision":IMPLEMENTATION_REVISION,
        "degradation_protocol_sha256":degradation_protocol_sha256(),
        "quality_gate":{"type":GATE_TYPE,"gate_parameter_l2":gl2,"num_scales":NUM_SCALES},
        "results":rr,
    }
    savej(rob_dir/"robustness_summary.json",rs)
    write_csv(rob_dir/"robustness_summary.csv",rr,[
        "model","condition","corruption","severity_level","severity_rank","clean_miou",
        "miou","drop_miou","delta_miou","relative_drop_pct","retention_pct",
        "pixel_accuracy","mean_class_accuracy","validation_seconds"
    ])

    # Gate stats
    savej(rob_dir/"gate_statistics.json",{"model":MODEL_ID,"rows":allg})
    write_csv(rob_dir/"gate_statistics.csv",allg,[
        "condition","corruption","severity_level","severity_rank","scale","count",
        "w_rgb_mean","w_rgb_std","w_rgb_min","w_rgb_p05","w_rgb_p25","w_rgb_median",
        "w_rgb_p75","w_rgb_p95","w_rgb_max","w_nir_mean"
    ])
    gm=gate_monotonicity(allg)
    savej(rob_dir/"gate_monotonicity.json",gm)

    # Four models
    four,fobj=four_model(a_clean,a_sum,b_clean,b_sum,f_clean,f_sum,c_clean,cres)
    savej(rob_dir/"four_model_comparison.json",fobj)
    write_csv(rob_dir/"four_model_comparison.csv",four,FOUR_FIELDS)
    pc=per_class_compare(four,c_clean,cres)
    write_csv(rob_dir/"per_class_four_model_comparison.csv",pc,[
        "condition","corruption","severity_level","class_id","class_name",
        "model_a_iou","model_b_iou","model_c_nogate_iou","model_c_iou",
        "c_gain_vs_a_iou","c_gain_vs_b_iou","c_gain_vs_c_nogate_iou"
    ])

    deg=four[1:]
    gains=[float(r["c_gain_vs_c_nogate"]) for r in deg]
    dropred=[float(r["c_drop_reduction_vs_c_nogate"]) for r in deg]
    rq={
        "RQ4_quality_gate_vs_fixed_fusion":{
            "clean_gain_c_vs_c_nogate":float(fobj["clean"]["C_minus_C_noGate"]),
            "degraded_gain_by_condition":{r["condition"]:r["c_gain_vs_c_nogate"] for r in deg},
            "mean_degraded_gain":float(np.mean(gains)),
            "positive_degraded_gain_count":int(sum(v>0 for v in gains)),
            "mean_drop_reduction_vs_c_nogate":float(np.mean(dropred)),
        },
        "RQ5_gate_interpretability":{
            "criterion":"mean w_RGB non-increasing across Clean -> L1 -> L2 -> L3",
            "families":gm,
        },
        "RQ6_robustness_drop":{
            "mean_degraded_drop":fobj["mean_degraded_drop"],
            "smallest_drop_condition_counts":fobj["smallest_drop_condition_counts"],
            "num_degraded_conditions":9,
        },
    }
    savej(rob_dir/"rq4_rq5_rq6_summary.json",rq)

    print("\n"+"="*142)
    print("FINAL FOUR-MODEL SUMMARY | A vs B vs C-noGate vs C/Ours")
    print("="*142)
    print(f"{'Condition':<28} {'A':>9} {'B':>9} {'Fixed':>9} {'C':>9} "
          f"{'C-Fixed':>10} {'A Drop':>9} {'B Drop':>9} {'F Drop':>9} {'C Drop':>9}")
    print("-"*142)
    for r in four:
        print(f"{r['condition']:<28} {r['model_a_miou']:>9.6f} {r['model_b_miou']:>9.6f} "
              f"{r['model_c_nogate_miou']:>9.6f} {r['model_c_miou']:>9.6f} "
              f"{r['c_gain_vs_c_nogate']:>+10.6f} {r['model_a_drop']:>9.6f} "
              f"{r['model_b_drop']:>9.6f} {r['model_c_nogate_drop']:>9.6f} "
              f"{r['model_c_drop']:>9.6f}")
    print("-"*142)
    print("Mean degraded Drop:",fobj["mean_degraded_drop"])
    print("Smallest-Drop counts:",fobj["smallest_drop_condition_counts"])
    print("-"*142)
    print("QUALITY GATE mean w_RGB | Clean -> L1 -> L2 -> L3")
    for fam,v in gm.items():
        print(f"{fam}:")
        for s,z in v["scales"].items():
            print(f"  {s}: {[round(q,6) for q in z['w_rgb_mean']]} | "
                  f"monotonic={z['monotonic_nonincreasing']}")
        print("  all_scales_monotonic=",v["all_scales_monotonic_nonincreasing"])
    print("-"*142)
    q=rq["RQ4_quality_gate_vs_fixed_fusion"]
    print(f"RQ4 | Clean C-Fixed={q['clean_gain_c_vs_c_nogate']:+.6f} | "
          f"mean degraded C-Fixed={q['mean_degraded_gain']:+.6f} | "
          f"positive={q['positive_degraded_gain_count']}/9 | "
          f"mean DropRed={q['mean_drop_reduction_vs_c_nogate']:+.6f}")
    print(f"4-model CSV      : {rob_dir/'four_model_comparison.csv'}")
    print(f"Gate statistics  : {rob_dir/'gate_statistics.csv'}")
    print(f"Gate monotonicity: {rob_dir/'gate_monotonicity.json'}")
    print(f"RQ summary       : {rob_dir/'rq4_rq5_rq6_summary.json'}")
    print("="*142)


if __name__ == "__main__":
    main()
