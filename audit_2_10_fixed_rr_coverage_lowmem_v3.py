#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-memory, disk-backed full-tile audit for Potsdam tile 2_10.

Fixes the earlier CPU script's primary memory problem: it does not allocate
the full 6x6000x6000 FP32 logits sum and 6000x6000 FP32 coverage map in RAM.
Those accumulators are disk-backed memmaps, and argmax/statistics/PNG writing
are done in row chunks.

This script keeps the original six-tile validation dataset/protocol intact,
but predicts only the original full tile 2_10 (256 windows).
It reuses the repository's existing model, checkpoint validation, dataset,
and deterministic degradation implementations.

Default:
  Fixed g=1.0 -> Fog L3
  RR-DARF     -> Gaussian Blur L3

It has not been run against the user's AutoDL data/checkpoints; syntax checked
only. The first run should be one model / one condition.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for p in (ROOT, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import validate_model_b2_rgbnir_fixed1_joint_robust4_coverage as fixedval
import validate_model_b2_darf_rr_coverage as rrval

TILE_ID = "2_10"
GT_VEG_REFERENCE = 17_861_449
GT_COVERAGE_REFERENCE = 49.615
CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]
VEGETATION_IDS = (2, 3)
CLASS_COLORS = np.array([
    [220, 220, 220],
    [220, 40, 40],
    [80, 190, 80],
    [20, 100, 40],
    [40, 90, 220],
    [230, 180, 30],
], dtype=np.uint8)


def resolve_path(p):
    p = Path(p).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def safe_name(s):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def load_fixed_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = fixedval.base.unwrap_checkpoint_state_dict(obj)
    fixedval.validate_checkpoint_metadata(meta, variant="fixed")
    model, model_meta = fixedval.build_model("fixed")
    model.load_state_dict(state, strict=True)
    print(
        f"[model] Fixed strict load PASS | epoch={meta.get('epoch')} "
        f"| step={meta.get('global_step')} "
        f"| params={model_meta.get('parameters', {}).get('total', 'n/a')}",
        flush=True,
    )
    del obj, state
    gc.collect()
    return model


def load_rr_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = rrval.joint_val.base.unwrap_checkpoint_state_dict(obj)
    rrval.validate_checkpoint_metadata(meta)
    model = rrval.build_model_from_checkpoint_meta(meta)
    model.load_state_dict(state, strict=True)
    print(
        f"[model] RR-DARF strict load PASS | epoch={meta.get('epoch')} "
        f"| step={meta.get('global_step')}",
        flush=True,
    )
    del obj, state
    gc.collect()
    return model


def condition_items(variant, mode):
    if variant == "fixed":
        items = fixedval.suite_conditions("all")
        clean = next(x for x in items if x.get("kind") == "clean")
        worst = next(
            x for x in items
            if x.get("family") == "fog" and x.get("level") == "L3"
        )
    else:
        items = rrval.conditions_for_suite("formal")
        clean = next(x for x in items if x.get("kind") == "clean")
        worst = next(
            x for x in items
            if x.get("family") == "gaussian_blur" and x.get("level") == "L3"
        )
    if mode == "clean":
        return [clean]
    if mode == "worst":
        return [worst]
    return [clean, worst]


def build_dataset(variant, item):
    if variant == "fixed":
        return fixedval.build_dataset(
            item=item,
            nir_fog_scatter_ratio=fixedval.DEFAULT_NIR_FOG_SCATTER_RATIO,
            fog_chunk_rows=fixedval.DEFAULT_FOG_CHUNK_ROWS,
        )
    return rrval.build_dataset(
        suite="formal",
        item=item,
        fog_chunk_rows=128,
    )


def validate_batch(batch, tile_id):
    # Use the project's established batch validator.
    fixedval.validate_batch(batch, tile_id=tile_id)


def memmap_inference(
    *, variant, model, dataset, tile_id, checkpoint_path,
    output_dir, device, batch_size, log_every, threads,
):
    """Overlap-mean-logit full-tile inference with disk-backed accumulators."""
    tile_ids = list(dataset.tile_ids)
    if len(tile_ids) != 6 or tile_id not in tile_ids:
        raise RuntimeError(
            f"Frozen tile list mismatch: {tile_ids}; expected six tiles including {tile_id}"
        )
    windows_per_tile = len(dataset.window_coordinates)
    if windows_per_tile != 256:
        raise RuntimeError(
            f"Frozen window protocol mismatch: {windows_per_tile} != 256"
        )
    tile_index = tile_ids.index(tile_id)
    start_idx = tile_index * windows_per_tile
    stop_idx = start_idx + windows_per_tile

    tile_size = int(dataset.spec.tile_size)
    crop_size = int(dataset.spec.crop_size)
    if (tile_size, crop_size) != (6000, 512):
        raise RuntimeError(
            f"Unexpected tile/crop sizes: {tile_size}/{crop_size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    acc_dir = output_dir / "_lowmem_work"
    acc_dir.mkdir(parents=True, exist_ok=True)
    logits_path = acc_dir / "logits_sum_f32.dat"
    coverage_path = acc_dir / "coverage_u8.dat"
    pred_path = output_dir / "Pred_labels_id.png.dat"

    # 6*6000*6000*4 = ~824 MiB on disk, not a RAM tensor.
    logits_sum = np.memmap(
        logits_path, mode="w+", dtype=np.float32,
        shape=(6, tile_size, tile_size),
    )
    # A newly created/truncated memmap is zero-filled by the filesystem.
    # Do NOT explicitly write zeros across the 1 GiB mapping: that needlessly
    # dirties every page and can itself exceed a 2 GiB cgroup memory limit.
    # Coverage maximum is 9 in the frozen protocol; uint8 is sufficient.
    coverage = np.memmap(
        coverage_path, mode="w+", dtype=np.uint8,
        shape=(tile_size, tile_size),
    )

    loader = DataLoader(
        Subset(dataset, range(start_idx, stop_idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    started = time.time()
    windows_seen = 0
    diagnostic_kind = None
    residual_sum = np.zeros(4, dtype=np.float64)
    residual_weight = 0
    gate_sum = np.zeros(4, dtype=np.float64)
    gate_sumsq = np.zeros(4, dtype=np.float64)
    gate_count = 0
    cov_min, cov_max = 255, 0

    print(
        f"[protocol] intact dataset: {len(tile_ids)} tiles; "
        f"target={tile_id} original_index={tile_index}; "
        f"{windows_per_tile} windows; disk-backed accumulators",
        flush=True,
    )

    try:
        for batch_index, batch in enumerate(loader):
            validate_batch(batch, tile_id=tile_id)
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=False,
                ):
                    details = model(rgb, nir, return_details=True)

            logits = details["logits"].float()
            if logits.ndim != 4 or tuple(logits.shape[1:]) != (6, crop_size, crop_size):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Logits contain NaN/Inf")

            n = int(logits.shape[0])
            if "nir_gate_strength" in details:
                diagnostic_kind = "darf"
                gate = details["nir_gate_strength"].detach().double().cpu().numpy()
                if gate.shape != (n, 4):
                    raise RuntimeError(f"Bad gate shape: {gate.shape}")
                gate_sum += gate.sum(axis=0)
                gate_sumsq += np.square(gate).sum(axis=0)
                gate_count += n
                diag_text = "gate_mean=" + str(np.round(gate.mean(axis=0), 4).tolist())
            elif "residual_abs_mean" in details:
                diagnostic_kind = "fixed"
                residual = details["residual_abs_mean"].detach().double().cpu().numpy().reshape(-1)
                if residual.size != 4:
                    raise RuntimeError(f"Bad residual diagnostic shape: {residual.shape}")
                residual_sum += residual * n
                residual_weight += n
                diag_text = "residual_mean=" + str(np.round(residual_sum / max(residual_weight, 1), 5).tolist())
            else:
                raise RuntimeError("Model returned neither gate nor residual diagnostics")

            logits_np = logits.detach().cpu().numpy()
            for j in range(n):
                x = int(batch["x"][j])
                y = int(batch["y"][j])
                y2, x2 = y + crop_size, x + crop_size
                # Update disk-backed arrays; only the 6x512x512 current logits are in RAM.
                logits_sum[:, y:y2, x:x2] += logits_np[j]
                coverage[y:y2, x:x2] += 1
            windows_seen += n

            if (batch_index + 1) % log_every == 0 or batch_index + 1 == len(loader):
                logits_sum.flush()
                coverage.flush()
                frac = windows_seen / windows_per_tile
                elapsed = time.time() - started
                eta = elapsed * (1 - frac) / max(frac, 1e-12)
                print(
                    f"  {windows_seen:03d}/{windows_per_tile} windows | "
                    f"elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min | {diag_text}",
                    flush=True,
                )
                # Encourage release of inactive mapped pages between batches.
                try:
                    import ctypes
                    libc = ctypes.CDLL(None)
                    MADV_DONTNEED = 4
                    libc.madvise(
                        ctypes.c_void_p(logits_sum.ctypes.data),
                        ctypes.c_size_t(logits_sum.nbytes),
                        ctypes.c_int(MADV_DONTNEED),
                    )
                    libc.madvise(
                        ctypes.c_void_p(coverage.ctypes.data),
                        ctypes.c_size_t(coverage.nbytes),
                        ctypes.c_int(MADV_DONTNEED),
                    )
                except Exception:
                    pass

            del details, logits, logits_np, rgb, nir
            gc.collect()

        if windows_seen != windows_per_tile:
            raise RuntimeError(f"Expected {windows_per_tile} windows, got {windows_seen}")

        logits_sum.flush()
        coverage.flush()
        # Check coverage in small row strips to avoid large in-memory temporaries.
        for y in range(0, tile_size, 256):
            block = np.asarray(coverage[y:min(y+256, tile_size), :])
            bmin, bmax = int(block.min()), int(block.max())
            cov_min, cov_max = min(cov_min, bmin), max(cov_max, bmax)
            if bmin <= 0:
                raise RuntimeError(f"Uncovered pixels detected in rows {y}:{y+256}")

        pred = np.memmap(
            pred_path, mode="w+", dtype=np.uint8,
            shape=(tile_size, tile_size),
        )
        # Argmax of summed logits equals argmax of mean logits because coverage
        # is a positive scalar shared by all classes at each pixel.
        for y in range(0, tile_size, 64):
            y2 = min(y + 64, tile_size)
            block = np.asarray(logits_sum[:, y:y2, :])
            pred[y:y2, :] = np.argmax(block, axis=0).astype(np.uint8)
            pred.flush()
            print(f"  finalizing prediction rows {y2}/{tile_size}", flush=True)

        runtime = {
            "tile_id": tile_id,
            "windows": windows_seen,
            "coverage_min": cov_min,
            "coverage_max": cov_max,
            "inference_seconds": time.time() - started,
            "variant_detected": diagnostic_kind,
            "device": str(device),
            "batch_size": batch_size,
            "accumulator": "disk-backed float32 memmap; uint8 coverage memmap",
            "checkpoint": str(checkpoint_path),
        }
        if diagnostic_kind == "fixed":
            if residual_weight == 0:
                raise RuntimeError("No residual diagnostics collected")
            for i, value in enumerate(residual_sum / residual_weight, 1):
                runtime[f"nir_residual_abs_mean_scale{i}"] = float(value)
        else:
            if gate_count == 0:
                raise RuntimeError("No gate diagnostics collected")
            mean = gate_sum / gate_count
            std = np.sqrt(np.maximum(gate_sumsq / gate_count - mean * mean, 0))
            for i in range(4):
                runtime[f"nir_gate_mean_scale{i+1}"] = float(mean[i])
                runtime[f"nir_gate_std_scale{i+1}"] = float(std[i])

        pred.flush()
        del logits_sum, coverage
        del pred
        gc.collect()
        try:
            logits_path.unlink(missing_ok=True)
            coverage_path.unlink(missing_ok=True)
            acc_dir.rmdir()
        except Exception:
            pass
        return np.memmap(pred_path, mode="r", dtype=np.uint8, shape=(tile_size, tile_size)), runtime
    except Exception:
        try:
            logits_sum.flush()
            coverage.flush()
        except Exception:
            pass
        raise


def png_chunk(f, kind, data):
    f.write(struct.pack(">I", len(data)))
    f.write(kind)
    f.write(data)
    f.write(struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))


def write_png_rows(path, width, height, color_type, row_iter):
    """Stream a PNG by rows without constructing a full RGB image in RAM."""
    channels = 1 if color_type == 0 else 3
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_chunk(f, b"IHDR", ihdr)
        compressor = zlib.compressobj(level=3)
        for arr in row_iter:
            a = np.asarray(arr, dtype=np.uint8)
            if a.ndim == 2 and channels == 1:
                raw = b"".join(b"\x00" + row.tobytes() for row in a)
            elif a.ndim == 3 and a.shape[2] == 3 and channels == 3:
                raw = b"".join(b"\x00" + row.tobytes() for row in a)
            else:
                raise RuntimeError(f"Bad PNG row chunk shape {a.shape}")
            compressed = compressor.compress(raw)
            if compressed:
                png_chunk(f, b"IDAT", compressed)
        tail = compressor.flush()
        if tail:
            png_chunk(f, b"IDAT", tail)
        png_chunk(f, b"IEND", b"")


def gray_rows(arr, chunk_rows=128):
    h = arr.shape[0]
    for y in range(0, h, chunk_rows):
        yield np.asarray(arr[y:min(y+chunk_rows, h), :])


def color_rows(arr, nclasses, chunk_rows=128):
    h = arr.shape[0]
    for y in range(0, h, chunk_rows):
        block = np.asarray(arr[y:min(y+chunk_rows, h), :])
        rgb = np.zeros((*block.shape, 3), dtype=np.uint8)
        for c in range(nclasses):
            rgb[block == c] = CLASS_COLORS[c]
        rgb[~((block >= 0) & (block < nclasses))] = 0
        yield rgb


def error_rows(gt, pred, nclasses, chunk_rows=128):
    h = gt.shape[0]
    for y in range(0, h, chunk_rows):
        g = np.asarray(gt[y:min(y+chunk_rows, h), :])
        p = np.asarray(pred[y:min(y+chunk_rows, h), :])
        valid = (g >= 0) & (g < nclasses)
        gv, pv = np.isin(g, VEGETATION_IDS), np.isin(p, VEGETATION_IDS)
        out = np.zeros((*g.shape, 3), dtype=np.uint8)
        out[valid & (g == p)] = (245,245,245)
        out[valid & gv & ~pv] = (230,50,50)
        out[valid & ~gv & pv] = (50,100,240)
        out[valid & (g != p) & (gv == pv)] = (245,200,40)
        yield out


def make_preview(gt, pred, outpath, nclasses):
    # Preview is downsampled to <=1200x1200; full-resolution images are streamed separately.
    step = max(1, int(math.ceil(max(gt.shape) / 1200)))
    g = np.asarray(gt[::step, ::step])
    p = np.asarray(pred[::step, ::step])
    def col(a):
        o = np.zeros((*a.shape,3), dtype=np.uint8)
        for c in range(nclasses):
            o[a == c] = CLASS_COLORS[c]
        o[~((a >= 0) & (a < nclasses))] = 0
        return o
    grgb, prgb = col(g), col(p)
    gv, pv = np.isin(g, VEGETATION_IDS), np.isin(p, VEGETATION_IDS)
    valid = (g >= 0) & (g < nclasses)
    ergb = np.zeros((*g.shape,3), dtype=np.uint8)
    ergb[valid & (g == p)] = (245,245,245)
    ergb[valid & gv & ~pv] = (230,50,50)
    ergb[valid & ~gv & pv] = (50,100,240)
    ergb[valid & (g != p) & (gv == pv)] = (245,200,40)
    ims = [Image.fromarray(a, "RGB") for a in (grgb, prgb, ergb)]
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    canvas = Image.new("RGB", (3*w, h+40), "white")
    draw = ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(ims,["GT classes","Pred classes","Vegetation errors"])):
        draw.text((i*w+8,8),label,fill="black")
        canvas.paste(im,(i*w,35))
    canvas.save(outpath)


def calculate_stats(gt, pred, output_dir, variant, condition, checkpoint, runtime):
    nclasses = 6
    h, w = gt.shape
    class_gt = np.zeros(nclasses, dtype=np.int64)
    class_pred = np.zeros(nclasses, dtype=np.int64)
    cm = np.zeros((nclasses,nclasses), dtype=np.int64)
    valid_pixels = 0
    gt_veg = 0
    pred_veg = 0
    for y in range(0,h,128):
        g = np.asarray(gt[y:min(y+128,h),:])
        p = np.asarray(pred[y:min(y+128,h),:])
        valid = (g >= 0) & (g < nclasses)
        gv, pv = g[valid].astype(np.int64), p[valid].astype(np.int64)
        valid_pixels += int(valid.sum())
        if gv.size:
            class_gt += np.bincount(gv,minlength=nclasses)
            class_pred += np.bincount(pv,minlength=nclasses)
            cm += np.bincount(gv*nclasses+pv,minlength=nclasses*nclasses).reshape(nclasses,nclasses)
        gt_veg += int(np.isin(g[valid],VEGETATION_IDS).sum())
        pred_veg += int(np.isin(p[valid],VEGETATION_IDS).sum())

    if gt_veg == 0 or valid_pixels == 0:
        raise RuntimeError("GT vegetation count or valid pixel count is zero")
    gt_cov = gt_veg / valid_pixels * 100.0
    pred_cov = pred_veg / valid_pixels * 100.0
    signed_pp = pred_cov - gt_cov
    abs_pp = abs(signed_pp)
    relative = abs(pred_veg - gt_veg) / gt_veg * 100.0

    with (output_dir/"class_pixel_counts.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr = csv.writer(f); wr.writerow(["class_id","class_name","gt_pixels","pred_pixels"])
        for i,name in enumerate(CLASS_NAMES):
            wr.writerow([i,name,int(class_gt[i]),int(class_pred[i])])
    with (output_dir/"confusion_matrix_GT_rows_Pred_cols.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr = csv.writer(f); wr.writerow(["GT \\ Pred"]+CLASS_NAMES)
        for i,name in enumerate(CLASS_NAMES):
            wr.writerow([name]+[int(v) for v in cm[i]])

    payload = {
        "tile_id":TILE_ID, "variant":variant, "condition":condition,
        "checkpoint":str(checkpoint), "runtime":runtime,
        "valid_pixels":valid_pixels,
        "class_names":CLASS_NAMES, "vegetation_class_ids":list(VEGETATION_IDS),
        "GT_class_pixels":class_gt.tolist(), "Pred_class_pixels":class_pred.tolist(),
        "GT_vegetation_pixels":gt_veg, "Pred_vegetation_pixels":pred_veg,
        "GT_coverage_percent":gt_cov, "Pred_coverage_percent":pred_cov,
        "signed_error_pp_Pred_minus_GT":signed_pp,
        "absolute_error_pp":abs_pp, "relative_error_percent":relative,
        "relative_error_lt_10pct":bool(relative < 10.0),
        "Notion_reference":{"GT_vegetation_pixels":GT_VEG_REFERENCE,
                            "GT_coverage_percent_rounded":GT_COVERAGE_REFERENCE,
                            "GT_pixel_count_matches":gt_veg == GT_VEG_REFERENCE},
        "confusion_matrix_rows_GT_cols_Pred":cm.tolist(),
        "error_map_legend":{"white":"correct class","red":"vegetation false negative",
                            "blue":"vegetation false positive","yellow":"other class mismatch",
                            "black":"invalid/ignored GT"}
    }
    (output_dir/"coverage_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(
        f"[coverage] {variant} | {condition} | GT veg={gt_veg:,} ({gt_cov:.4f}%) | "
        f"Pred veg={pred_veg:,} ({pred_cov:.4f}%) | abs error={abs_pp:.4f} pp | "
        f"relative error={relative:.4f}% | GT matches Notion={gt_veg == GT_VEG_REFERENCE}",
        flush=True,
    )
    return {
        "tile_id":TILE_ID,"variant":variant,"condition":condition,"checkpoint":str(checkpoint),
        "valid_pixels":valid_pixels,"GT_vegetation_pixels":gt_veg,"Pred_vegetation_pixels":pred_veg,
        "GT_coverage_percent":gt_cov,"Pred_coverage_percent":pred_cov,
        "signed_error_pp_Pred_minus_GT":signed_pp,"absolute_error_pp":abs_pp,
        "relative_error_percent":relative,"relative_error_lt_10pct":relative < 10.0,
        "gt_pixel_count_matches_notion":gt_veg == GT_VEG_REFERENCE,"diagnostic_dir":str(output_dir)
    }


def write_diagnostics(gt, pred, output_dir, variant, condition, checkpoint, runtime):
    output_dir.mkdir(parents=True, exist_ok=True)
    h,w = gt.shape
    write_png_rows(output_dir/"GT_labels_id.png", w,h,0,gray_rows(gt))
    write_png_rows(output_dir/"Pred_labels_id.png", w,h,0,gray_rows(pred))
    write_png_rows(output_dir/"GT_classes_color.png", w,h,2,color_rows(gt,6))
    write_png_rows(output_dir/"Pred_classes_color.png", w,h,2,color_rows(pred,6))
    write_png_rows(output_dir/"vegetation_error_map.png", w,h,2,error_rows(gt,pred,6))
    make_preview(gt,pred,output_dir/"comparison_preview.png",6)
    return calculate_stats(gt,pred,output_dir,variant,condition,checkpoint,runtime)


def run_variant(variant, checkpoint, condition_mode, output_root, device, batch_size, log_every, threads):
    checkpoint = resolve_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = load_fixed_model(checkpoint) if variant == "fixed" else load_rr_model(checkpoint)
    model.to(device)
    model.eval()
    rows = []
    for item in condition_items(variant, condition_mode):
        condition = str(item.get("condition", "unknown"))
        output_dir = output_root / variant / safe_name(condition) / TILE_ID
        print(f"\n===== {variant.upper()} | {condition} | tile {TILE_ID} | {device} =====",flush=True)
        dataset = build_dataset(variant,item)
        # Do not construct a separate clean dataset: it wastes RAM and is not needed
        # for a coverage audit. The condition implementation and frozen protocol are
        # still the official repository implementations.
        pred_path = output_dir / "Pred_labels_id.png.dat"
        pred_path.parent.mkdir(parents=True,exist_ok=True)
        pred, runtime = memmap_inference(
            variant=variant,model=model,dataset=dataset,tile_id=TILE_ID,
            checkpoint_path=checkpoint,output_dir=output_dir,device=device,
            batch_size=batch_size,log_every=log_every,threads=threads,
        )
        gt = dataset.load_full_label(TILE_ID)
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        gt = np.asarray(gt)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        if gt.shape != pred.shape:
            raise RuntimeError(f"GT/Pred shape mismatch: {gt.shape} vs {pred.shape}")
        row = write_diagnostics(gt,pred,output_dir,variant,condition,checkpoint,runtime)
        rows.append(row)
        del gt,pred,dataset
        gc.collect()
    del model
    gc.collect()
    return rows


def main():
    ap=argparse.ArgumentParser(description="Low-memory disk-backed coverage audit for Potsdam tile 2_10")
    ap.add_argument("--variant",choices=["fixed","rr","both"],default="fixed")
    ap.add_argument("--condition",choices=["worst","clean","both"],default="worst",
                    help="worst=Fixed Fog L3 / RR Gaussian Blur L3; both also includes Clean.")
    ap.add_argument("--fixed-checkpoint",type=Path,default=Path("outputs/training/b2_rgbnir_fixed1_joint_robust4/checkpoints/final.pt"))
    ap.add_argument("--rr-checkpoint",type=Path,default=Path("outputs/training/b2_darf_rr_joint_robust4/checkpoints/final.pt"))
    ap.add_argument("--output-root",type=Path,default=Path("outputs/coverage_audit_2_10_lowmem_v3"))
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--log-every",type=int,default=8)
    ap.add_argument("--threads",type=int,default=2)
    a=ap.parse_args()
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    if min(a.batch_size,a.log_every,a.threads)<1:
        ap.error("batch-size, log-every and threads must be positive")
    torch.set_num_threads(a.threads)
    device=torch.device(a.device)
    output_root=resolve_path(a.output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    variants=["fixed","rr"] if a.variant=="both" else [a.variant]
    all_rows=[]
    for variant in variants:
        ckpt=a.fixed_checkpoint if variant=="fixed" else a.rr_checkpoint
        all_rows.extend(run_variant(variant,ckpt,a.condition,output_root,device,a.batch_size,a.log_every,a.threads))
    summary=output_root/"summary_2_10_GT_Pred_coverage.csv"
    fields=["tile_id","variant","condition","checkpoint","valid_pixels","GT_vegetation_pixels","Pred_vegetation_pixels",
            "GT_coverage_percent","Pred_coverage_percent","signed_error_pp_Pred_minus_GT","absolute_error_pp",
            "relative_error_percent","relative_error_lt_10pct","gt_pixel_count_matches_notion","diagnostic_dir"]
    with summary.open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(all_rows)
    print(f"\n[DONE] Summary: {summary}",flush=True)
    print(f"[DONE] Outputs: {output_root}",flush=True)

if __name__=="__main__":
    main()
