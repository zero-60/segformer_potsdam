#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only controlled registration experiment for HRSCD/Potsdam-style GeoTIFFs.

This experiment starts from one unchanged reference image, synthetically shifts a
copy by 1/2/4/8 pixels, and tests whether phase correlation and ORB+RANSAC recover
the known correction. It tests geometric recovery under controlled shifts; it does
NOT claim to measure true 2006-vs-2012 HRSCD alignment.

Example:
python tools/run_controlled_registration.py \
  --image data/HRSCD/images_2006/D14/14-2006-0415-6890-LA93-0M50-E080.tif \
  --out outputs/registration_controlled \
  --shifts 1 2 4 8 --max-side 1600 --max-images 20

If --image is omitted, --input-dir scans TIFF files and runs a small batch.
Requires opencv-python, numpy, pandas, tifffile, Pillow.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image

def read_rgb(path: Path, max_side: int):
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        # TIFs may be HWC or CHW; use first three visible bands.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported image shape {arr.shape}: {path}")
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}: {path}")
    arr = np.nan_to_num(arr)
    # Normalize each band for visualization/feature detection only.
    out = np.zeros(arr.shape, dtype=np.uint8)
    for c in range(3):
        band = arr[..., c].astype(np.float32)
        lo, hi = np.percentile(band, [1, 99])
        if hi <= lo: hi = lo + 1
        out[..., c] = np.clip((band - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1:
        out = cv2.resize(out, (max(32, int(w*scale)), max(32, int(h*scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def shift_image(img, dx, dy):
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def phase_recover(ref, mov):
    # phaseCorrelate(ref,mov) estimates the translation from ref toward mov.
    (dx, dy), response = cv2.phaseCorrelate(np.float32(gray(ref)), np.float32(gray(mov)))
    return -float(dx), -float(dy), float(response)

def orb_recover(ref, mov):
    g1, g2 = gray(ref), gray(mov)
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, 0, 0, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d2, d1, k=2)  # shifted-image points -> reference points
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75*n.distance]
    if len(good) < 4:
        return None, len(good), 0, None
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                             maxIters=3000, confidence=0.99, refineIters=10)
    if M is None:
        return None, len(good), 0, None
    dx, dy = float(M[0,2]), float(M[1,2])
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    ratio = inlier_count / len(good) if good else 0
    return (dx, dy), len(good), inlier_count, ratio

def save_pair_preview(ref, shifted, corrected, path):
    # side-by-side ref, shifted, corrected (all BGR)
    panel = np.concatenate([ref, shifted, corrected], axis=1)
    cv2.imwrite(str(path), panel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None, help="One TIFF to use as the unchanged source image.")
    ap.add_argument("--input-dir", type=Path, default=None, help="Batch scan .tif/.tiff files, e.g. data/HRSCD/images_2006")
    ap.add_argument("--out", type=Path, default=Path("outputs/registration_controlled"))
    ap.add_argument("--shifts", nargs="+", type=int, default=[1,2,4,8])
    ap.add_argument("--max-side", type=int, default=1600, help="Downsample long side to this for CPU feasibility.")
    ap.add_argument("--max-images", type=int, default=20)
    args = ap.parse_args()
    if args.image is None and args.input_dir is None:
        ap.error("Provide --image or --input-dir")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.image:
        files = [args.image.resolve()]
    else:
        files = sorted([*args.input_dir.rglob("*.tif"), *args.input_dir.rglob("*.tiff")])[:args.max_images]
    if not files:
        raise FileNotFoundError("No TIFF files found.")
    rows = []
    for fi, path in enumerate(files, 1):
        print(f"[{fi}/{len(files)}] {path}")
        try:
            ref = read_rgb(path, args.max_side)
        except Exception as e:
            print(f"  SKIP read error: {e}")
            continue
        for magnitude in args.shifts:
            # Test both horizontal and vertical translations; avoid repeating the same axis only.
            for dx, dy, axis in [(int(magnitude), 0, "x"), (0, int(magnitude), "y")]:
                moving = shift_image(ref, dx, dy)
                t0 = time.time()
                pdx, pdy, response = phase_recover(ref, moving)
                phase_err = float(np.hypot(pdx + dx, pdy + dy))
                # warpAffine uses the estimated correction to resample moving back onto reference grid
                phM = np.array([[1,0,pdx],[0,1,pdy]], dtype=np.float32)
                ph_corrected = cv2.warpAffine(moving, phM, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                phase_seconds = time.time() - t0
                t1 = time.time()
                orb_result, good_n, inlier_n, inlier_ratio = orb_recover(ref, moving)
                orb_seconds = time.time() - t1
                if orb_result is not None:
                    odx, ody = orb_result
                    orb_err = float(np.hypot(odx + dx, ody + dy))
                    orbM = np.array([[1,0,odx],[0,1,ody]], dtype=np.float32)
                    orb_corrected = cv2.warpAffine(moving, orbM, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                else:
                    odx = ody = orb_err = None
                    orb_corrected = moving
                case = out / f"{path.stem}_shift_{axis}{magnitude}"
                case.mkdir(parents=True, exist_ok=True)
                save_pair_preview(ref, moving, ph_corrected, case / "reference_shifted_phasecorrected.png")
                save_pair_preview(ref, moving, orb_corrected, case / "reference_shifted_orbcorrected.png")
                rows.append({
                    "image": str(path), "image_index": fi, "axis": axis,
                    "known_dx": dx, "known_dy": dy,
                    "phase_estimated_correction_dx": pdx, "phase_estimated_correction_dy": pdy,
                    "phase_translation_error_px": phase_err, "phase_response": response,
                    "phase_seconds": phase_seconds,
                    "orb_estimated_correction_dx": odx, "orb_estimated_correction_dy": ody,
                    "orb_good_matches": good_n, "orb_ransac_inliers": inlier_n,
                    "orb_inlier_ratio": inlier_ratio, "orb_translation_error_px": orb_err,
                    "orb_seconds": orb_seconds,
                    "interpretation": "synthetic known-shift test; not a real cross-date registration result"
                })
    csv_path = out / "registration_controlled_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    (out / "experiment_protocol.json").write_text(json.dumps({
        "purpose": "CPU controlled known-translation registration test",
        "input_image_count_processed": len(set(r["image"] for r in rows)),
        "shift_magnitudes_px": args.shifts,
        "axes": ["x", "y"],
        "methods": ["phase correlation", "ORB + RANSAC partial affine"],
        "max_side": args.max_side,
        "warning": "The shifted image is synthetically generated from the same source. This tests recovery of known translation only; it does not prove real 2006-vs-2012 HRSCD alignment."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Results: {csv_path}")
    print(f"Cases and previews: {out}")
if __name__ == "__main__":
    main()
