#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-capable full-tile coverage audit for Potsdam tile 2_10.

Unlike the previous version, this script does not wrap the dataset or call the
full six-tile validator main(). It keeps the original six-tile dataset intact,
then directly invokes the repository's existing full-tile inference function
for the original index of tile 2_10 only.

Default condition mode is 'worst':
  Fixed model -> joint Fog L3
  RR-DARF     -> joint Gaussian Blur L3
"""
from __future__ import annotations
import argparse, csv, gc, json, os, re, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for p in (ROOT, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgbnir_fixed1_joint_robust4_coverage as fixedval
import validate_model_b2_darf_rr_coverage as rrval

TILE = "2_10"
GT_VEG_REFERENCE = 17_861_449
GT_COVERAGE_REFERENCE = 49.615
COLORS = np.array([
    [220,220,220], [220,40,40], [80,190,80],
    [20,100,40], [40,90,220], [230,180,30]
], dtype=np.uint8)

def resolve(p):
    p = Path(p).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def load_fixed(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = fixedval.base.unwrap_checkpoint_state_dict(obj)
    fixedval.validate_checkpoint_metadata(meta, variant="fixed")
    model, model_meta = fixedval.build_model("fixed")
    model.load_state_dict(state, strict=True)
    print(f"[model] Fixed strict load PASS | epoch={meta.get('epoch')} | step={meta.get('global_step')}", flush=True)
    del obj, state
    gc.collect()
    return model

def load_rr(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = rrval.joint_val.base.unwrap_checkpoint_state_dict(obj)
    rrval.validate_checkpoint_metadata(meta)
    model = rrval.build_model_from_checkpoint_meta(meta)
    model.load_state_dict(state, strict=True)
    print(f"[model] RR-DARF strict load PASS | epoch={meta.get('epoch')} | step={meta.get('global_step')}", flush=True)
    del obj, state
    gc.collect()
    return model

def to_np_label(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise RuntimeError(f"Expected 2D label, got {a.shape}")
    return a.astype(np.uint8, copy=False)

def colorize(mask, nclasses):
    out = np.zeros((*mask.shape,3), dtype=np.uint8)
    for i in range(nclasses):
        out[mask == i] = COLORS[i]
    out[~((mask >= 0) & (mask < nclasses))] = (0,0,0)
    return out

def error_map(gt, pred, nclasses, veg_ids):
    valid = (gt >= 0) & (gt < nclasses)
    gv, pv = np.isin(gt, list(veg_ids)), np.isin(pred, list(veg_ids))
    out = np.zeros((*gt.shape,3), dtype=np.uint8)
    out[valid & (gt == pred)] = (245,245,245)
    out[valid & gv & ~pv] = (230,50,50)
    out[valid & ~gv & pv] = (50,100,240)
    out[valid & (gt != pred) & (gv == pv)] = (245,200,40)
    return out

def save_preview(arrays, path):
    ims = []
    for arr in arrays:
        im = Image.fromarray(arr, "RGB")
        im.thumbnail((1100,1100), Image.Resampling.NEAREST)
        ims.append(im.copy())
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    canvas = Image.new("RGB", (3*w,h+40), "white")
    draw = ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(ims,["GT classes","Pred classes","Vegetation errors"])):
        draw.text((i*w+8,8),title,fill="black")
        canvas.paste(im,(i*w,35))
    canvas.save(path)

def diagnostics(dataset, pred, outdir, variant, condition, ckpt, runtime, save_npy):
    outdir.mkdir(parents=True, exist_ok=True)
    gt = to_np_label(dataset.load_full_label(TILE))
    pred = np.asarray(pred).astype(np.uint8, copy=False)
    if gt.shape != pred.shape:
        raise RuntimeError(f"GT/Pred shape mismatch: {gt.shape} != {pred.shape}")

    # Use the project's existing vegetation-coverage function.
    stats = fixedval.vegetation_tile_statistics(dataset=dataset, tile_id=TILE, prediction=pred)
    nclasses = int(fixedval.base.NUM_CLASSES)
    names = list(fixedval.CLASS_NAMES)
    veg_ids = tuple(fixedval.VEGETATION_CLASS_IDS)
    valid = (gt >= 0) & (gt < nclasses)
    gv, pv = gt[valid].astype(np.int64), pred[valid].astype(np.int64)
    cm = np.bincount(gv*nclasses + pv, minlength=nclasses*nclasses).reshape(nclasses,nclasses)

    Image.fromarray(gt, "L").save(outdir/"GT_labels_id.png")
    Image.fromarray(pred, "L").save(outdir/"Pred_labels_id.png")
    gt_rgb, pred_rgb = colorize(gt,nclasses), colorize(pred,nclasses)
    err_rgb = error_map(gt,pred,nclasses,veg_ids)
    Image.fromarray(gt_rgb,"RGB").save(outdir/"GT_classes_color.png")
    Image.fromarray(pred_rgb,"RGB").save(outdir/"Pred_classes_color.png")
    Image.fromarray(err_rgb,"RGB").save(outdir/"vegetation_error_map.png")
    save_preview([gt_rgb,pred_rgb,err_rgb], outdir/"comparison_preview.png")
    if save_npy:
        np.save(outdir/"GT_labels.npy", gt)
        np.save(outdir/"Pred_labels.npy", pred)

    with (outdir/"class_pixel_counts.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f,fieldnames=["class_id","class_name","gt_pixels","pred_pixels"])
        w.writeheader()
        w.writerows(stats["per_class_pixel_counts"])
    with (outdir/"confusion_matrix_GT_rows_Pred_cols.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["GT \\ Pred"]+names)
        for i,name in enumerate(names): w.writerow([name]+[int(x) for x in cm[i]])

    gt_match = int(stats["vegetation_gt_pixels"]) == GT_VEG_REFERENCE
    payload = {
        "tile_id":TILE, "variant":variant, "condition":condition, "checkpoint":str(ckpt),
        "coverage_metrics":stats, "runtime_diagnostics":dict(runtime),
        "notion_reference":{"gt_vegetation_pixels":GT_VEG_REFERENCE,
                            "gt_coverage_percent_rounded":GT_COVERAGE_REFERENCE,
                            "gt_vegetation_pixel_count_matches":gt_match},
        "confusion_matrix_rows_gt_cols_pred":cm.tolist(),
        "legend":{"white":"GT=Pred","red":"vegetation false negative",
                  "blue":"vegetation false positive","yellow":"other class mismatch",
                  "black":"ignored/invalid GT"}
    }
    (outdir/"coverage_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(f"[coverage] {variant} | {condition} | GT={stats['vegetation_gt_pixels']:,} "
          f"({stats['vegetation_gt_coverage_pct']:.4f}%) | Pred={stats['vegetation_pred_pixels']:,} "
          f"({stats['vegetation_pred_coverage_pct']:.4f}%) | abs error={stats['vegetation_coverage_abs_error_pp']:.4f} pp "
          f"| relative error={stats['vegetation_coverage_relative_error_pct']:.4f}%",flush=True)
    if not gt_match:
        print(f"[WARNING] GT vegetation pixels {stats['vegetation_gt_pixels']:,} != Notion reference {GT_VEG_REFERENCE:,}; verify labels/class mapping.",flush=True)
    return {
        "tile_id":TILE,"variant":variant,"condition":condition,"checkpoint":str(ckpt),
        "valid_pixels":stats["coverage_valid_pixels"],
        "GT_vegetation_pixels":stats["vegetation_gt_pixels"],
        "Pred_vegetation_pixels":stats["vegetation_pred_pixels"],
        "GT_coverage_percent":stats["vegetation_gt_coverage_pct"],
        "Pred_coverage_percent":stats["vegetation_pred_coverage_pct"],
        "signed_error_pp_Pred_minus_GT":stats["vegetation_coverage_diff_pp"],
        "absolute_error_pp":stats["vegetation_coverage_abs_error_pp"],
        "relative_error_percent":stats["vegetation_coverage_relative_error_pct"],
        "relative_error_lt_10pct":stats["vegetation_relative_error_lt_10pct"],
        "gt_pixel_count_matches_notion":gt_match,"diagnostic_dir":str(outdir)
    }

def condition_items(variant, mode):
    if variant == "fixed":
        items = fixedval.suite_conditions("all")
        clean = next(x for x in items if x["kind"]=="clean")
        worst = next(x for x in items if x.get("family")=="fog" and x.get("level")=="L3")
    else:
        items = rrval.conditions_for_suite("formal")
        clean = next(x for x in items if x["kind"]=="clean")
        worst = next(x for x in items if x.get("family")=="gaussian_blur" and x.get("level")=="L3")
    return [clean] if mode=="clean" else [worst] if mode=="worst" else [clean,worst]

def run_one_condition(variant, model, item, clean_ds, ckpt, device, batch_size, log_every, output_root, save_npy):
    if variant == "fixed":
        ds = fixedval.build_dataset(item=item,
             nir_fog_scatter_ratio=fixedval.DEFAULT_NIR_FOG_SCATTER_RATIO,
             fog_chunk_rows=fixedval.DEFAULT_FOG_CHUNK_ROWS)
        if item["kind"] != "clean":
            fixedval.joint_degradation_probe(clean_dataset=clean_ds,degraded_dataset=ds,condition=str(item["condition"]))
        infer_fn = fixedval.infer_one_tile_joint
    else:
        ds = rrval.build_dataset(suite="formal",item=item,fog_chunk_rows=128)
        if item["kind"] != "clean":
            rrval.probe_transform(suite="formal",clean_dataset=clean_ds,dataset=ds,item=item)
        infer_fn = rrval.joint_val.infer_one_tile_joint

    ids = list(ds.tile_ids)
    if TILE not in ids:
        raise RuntimeError(f"Tile {TILE} not in validation dataset: {ids}")
    idx = ids.index(TILE)
    print(f"[dataset] protocol unchanged: {len(ids)} tiles; target {TILE} at original index {idx}; windows/tile={len(ds.window_coordinates)}",flush=True)
    pred, runtime = infer_fn(model=model,dataset=ds,tile_id=TILE,tile_index=idx,
        batch_size=batch_size,num_workers=0,pin_memory=False,device=device,
        amp_enabled=False,log_every=log_every)
    dest = output_root/variant/safe(item["condition"])/TILE
    row = diagnostics(ds,pred,dest,variant,str(item["condition"]),ckpt,runtime,save_npy)
    del pred, ds
    gc.collect()
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--variant",choices=["fixed","rr","both"],default="fixed")
    ap.add_argument("--condition",choices=["worst","clean","both"],default="worst",
        help="worst=Fixed Fog L3 / RR Gaussian Blur L3; both also runs Clean.")
    ap.add_argument("--fixed-checkpoint",type=Path,default=Path("outputs/training/b2_rgbnir_fixed1_joint_robust4/checkpoints/final.pt"))
    ap.add_argument("--rr-checkpoint",type=Path,default=Path("outputs/training/b2_darf_rr_joint_robust4/checkpoints/final.pt"))
    ap.add_argument("--output-root",type=Path,default=Path("outputs/coverage_audit_2_10_direct"))
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--log-every",type=int,default=16)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--save-npy",action="store_true")
    a=ap.parse_args()
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run with --device cpu.")
    if a.batch_size<1 or a.log_every<1 or a.threads<1: ap.error("batch-size/log-every/threads must be positive")
    torch.set_num_threads(a.threads)
    device=torch.device(a.device)
    outroot=resolve(a.output_root); outroot.mkdir(parents=True,exist_ok=True)
    variants=["fixed","rr"] if a.variant=="both" else [a.variant]
    rows=[]
    for variant in variants:
        ckpt=resolve(a.fixed_checkpoint if variant=="fixed" else a.rr_checkpoint)
        if not ckpt.is_file(): raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
        model=load_fixed(ckpt) if variant=="fixed" else load_rr(ckpt)
        model.to(device); model.eval()
        if variant=="fixed":
            clean_item=next(x for x in fixedval.suite_conditions("all") if x["kind"]=="clean")
            clean_ds=fixedval.build_dataset(item=clean_item,
                nir_fog_scatter_ratio=fixedval.DEFAULT_NIR_FOG_SCATTER_RATIO,
                fog_chunk_rows=fixedval.DEFAULT_FOG_CHUNK_ROWS)
        else:
            clean_ds=rrval.clean_probe()
        for item in condition_items(variant,a.condition):
            print(f"\n===== {variant.upper()} | {item['condition']} | tile {TILE} | {device} =====",flush=True)
            rows.append(run_one_condition(variant,model,item,clean_ds,ckpt,device,a.batch_size,a.log_every,outroot,a.save_npy))
        del model, clean_ds
        gc.collect()
    summary=outroot/"summary_2_10_GT_Pred_coverage.csv"
    fields=["tile_id","variant","condition","checkpoint","valid_pixels","GT_vegetation_pixels","Pred_vegetation_pixels","GT_coverage_percent","Pred_coverage_percent","signed_error_pp_Pred_minus_GT","absolute_error_pp","relative_error_percent","relative_error_lt_10pct","gt_pixel_count_matches_notion","diagnostic_dir"]
    with summary.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n[done] Summary: {summary}\n[done] Output: {outroot}",flush=True)

if __name__=="__main__":
    main()
