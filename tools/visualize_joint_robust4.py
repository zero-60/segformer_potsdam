#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Potsdam RGBIR Robust4 visualization, matching frozen Joint validation settings.
Place alongside your repo or run standalone.
Scene: full raw RGBIR tile -> deterministic corruption -> select same crop.
Outputs: 4x4 RGB and NIR grids (each row = corruption, cols = Clean/L1/L2/L3),
and individual condition PNGs. NO semantic predictions are generated.

Dependencies: pip install numpy opencv-python tifffile pillow
Examples:
 python visualize_joint_robust4.py --input data/raw/potsdam/4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif --output vis
 python visualize_joint_robust4.py --input-dir data/raw/potsdam/4_Ortho_RGBIR --max-tiles 3 --output vis
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

FAMILIES = ("gaussian_noise", "gaussian_blur", "underexposure", "fog")
LEVELS = ("L1", "L2", "L3")
NOISE = dict(L1=10.0, L2=25.0, L3=50.0)
BLUR = dict(L1=1.0, L2=2.0, L3=4.0)
EXPOSURE = dict(L1=0.8, L2=0.6, L3=0.4)
FOG = dict(L1=(0.80, 0.10, 0.70, 0.90),
           L2=(0.60, 0.15, 0.45, 0.75),
           L3=(0.40, 0.15, 0.25, 0.55))
NIR_FOG_RATIO = 0.65

def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],
                          "big", signed=False)

def round_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 255)
    return np.rint(x).astype(np.uint8)

def read_tile(path: Path) -> np.ndarray:
    try:
        arr = tifffile.memmap(str(path))
    except (ValueError, OSError):
        arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 4 and arr.shape[-1] != 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[2] != 4 or arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HxWx4 RGBIR TIFF: {path}; got {arr.shape} {arr.dtype}")
    return arr

def extract_tile_id(path: Path) -> str:
    match = re.search(r"(?<!\d)(\d+_\d+)(?!\d)", path.stem)
    return match.group(1) if match else path.stem

def noise_roi(tile: np.ndarray, y: int, x: int, size: int,
              sigma: float, tile_id: str, level: str, joint: bool) -> np.ndarray:
    """Generate random noise in 256-row FULL-width blocks; keep ROI only.
    RNG sequence remains the same as in the official full-tile validator."""
    h, w, _ = tile.shape
    result = np.array(tile[y:y+size, x:x+size, :], copy=True)
    rgb_seed = stable_seed(f"RGB Degradation Protocol v1|20260917|{tile_id}|gaussian_noise|{level}")
    nir_seed = stable_seed(f"Joint RGB+NIR Validation Protocol v1|20261002|{tile_id}|nir_independent_gaussian_noise|{level}")
    rgb_rng, nir_rng = np.random.default_rng(rgb_seed), np.random.default_rng(nir_seed)
    for y0 in range(0, min(h, y + size), 256):
        y1 = min(y0 + 256, h)
        rgb_noise = rgb_rng.normal(0.0, sigma, (y1-y0, w, 3)).astype(np.float32)
        nir_noise = nir_rng.normal(0.0, sigma, (y1-y0, w)).astype(np.float32) if joint else None
        lo, hi = max(y, y0), min(y+size, y1)
        if lo >= hi:
            continue
        dst = slice(lo-y, hi-y)
        src = slice(lo-y0, hi-y0)
        region = tile[lo:hi, x:x+size, :3].astype(np.float32)
        result[dst, :, :3] = round_uint8(region + rgb_noise[src, x:x+size, :])
        if joint:
            nir = tile[lo:hi, x:x+size, 3].astype(np.float32)
            result[dst, :, 3] = round_uint8(nir + nir_noise[src, x:x+size])
    return result

def fog_roi(tile: np.ndarray, y: int, x: int, size: int,
            tile_id: str, level: str, joint: bool) -> np.ndarray:
    """Compute the full-tile-position analytical field only over the ROI."""
    h, w, _ = tile.shape
    seed = stable_seed(f"Model D Atmospheric Fog OOD Protocol v1|20260928|{tile_id}|fog|{level}")
    rng = np.random.default_rng(seed)
    fx = rng.uniform(0.35, 1.60, 4).astype(np.float32)
    fy = rng.uniform(0.35, 1.60, 4).astype(np.float32)
    phase = rng.uniform(0.0, 2*math.pi, 4).astype(np.float32)
    weights = np.asarray([0.34, 0.27, 0.22, 0.17], np.float32)
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=4).astype(np.float32)

    xx = (np.arange(x, x+size, dtype=np.float32) / max(w-1, 1))[None, :]
    yy = (np.arange(y, y+size, dtype=np.float32) / max(h-1, 1))[:, None]
    field = np.zeros((size, size), np.float32)
    two_pi = np.float32(2 * math.pi)
    for k in range(4):
        arg = two_pi * (fx[k]*xx + fy[k]*yy) + phase[k]
        field += weights[k] * signs[k] * np.sin(arg).astype(np.float32)
    np.clip(field, -1.0, 1.0, out=field)
    mean, variation, lower, upper = FOG[level]
    t_rgb = np.clip(np.float32(mean) + np.float32(variation)*field,
                    np.float32(lower), np.float32(upper))
    original = tile[y:y+size, x:x+size, :]
    result = np.array(original, copy=True)
    rgb = original[..., :3].astype(np.float32)
    result[..., :3] = round_uint8(rgb*t_rgb[..., None] +
                                 np.float32(245.0)*(np.float32(1)-t_rgb[..., None]))
    if joint:
        t_nir = np.clip(np.float32(1.0) - np.float32(NIR_FOG_RATIO) *
                        (np.float32(1.0)-t_rgb), 0.20, 0.995)
        nir = original[..., 3].astype(np.float32)
        result[..., 3] = round_uint8(nir*t_nir +
                                     np.float32(245.0)*(np.float32(1.0)-t_nir))
    return result

def corrupt_roi(tile: np.ndarray, y: int, x: int, size: int,
                tile_id: str, family: str, level: str, joint: bool) -> np.ndarray:
    if family not in FAMILIES or level not in LEVELS:
        raise ValueError(f"Invalid family/level: {family!r}, {level!r}")
    if not (size > 0 and x >= 0 and y >= 0 and
            x+size <= tile.shape[1] and y+size <= tile.shape[0]):
        raise ValueError(f"ROI out of bounds: x={x}, y={y}, size={size}, tile={tile.shape}")
    crop = np.array(tile[y:y+size, x:x+size, :], copy=True)
    if family == "gaussian_noise":
        return noise_roi(tile, y, x, size, NOISE[level], tile_id, level, joint)
    if family == "gaussian_blur":
        # Full-tile blur first, then crop. Reflect-101 edge handling matches the repo.
        rgb_blur = cv2.GaussianBlur(tile[..., :3], (0, 0), BLUR[level],
                                   borderType=cv2.BORDER_REFLECT_101)
        crop[..., :3] = rgb_blur[y:y+size, x:x+size]
        del rgb_blur
        if joint:
            nir_blur = cv2.GaussianBlur(tile[..., 3], (0, 0), BLUR[level],
                                       borderType=cv2.BORDER_REFLECT_101)
            crop[..., 3] = nir_blur[y:y+size, x:x+size]
        return crop
    if family == "underexposure":
        crop[..., :3] = round_uint8(crop[..., :3].astype(np.float32) * EXPOSURE[level])
        if joint:
            crop[..., 3] = round_uint8(crop[..., 3].astype(np.float32) * EXPOSURE[level])
        return crop
    return fog_roi(tile, y, x, size, tile_id, level, joint)

def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()

def montage(rows: list[list[np.ndarray]], path: Path, labels: list[str],
            mode: str, tile_id: str, box_size: int = 235) -> None:
    margin, gutter, header, label_h = 24, 13, 100, 36
    width = margin*2 + 4*box_size + 3*gutter
    height = header + margin + 4*(box_size+label_h) + 3*gutter + margin
    canvas = Image.new("RGB", (width, height), "#fbfaf7")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 19), f"{mode} / {tile_id} / Joint Robust4", fill="#203d4d", font=font(23))
    draw.text((margin, 56), "Columns: Clean | L1 mild | L2 moderate | L3 severe",
              fill="#596872", font=font(14))
    for r, panels in enumerate(rows):
        for c, arr in enumerate(panels):
            px, py = margin+c*(box_size+gutter), header+margin+r*(box_size+label_h+gutter)
            if arr.ndim == 2:
                im = Image.fromarray(arr, "L").convert("RGB")
            else:
                im = Image.fromarray(arr, "RGB")
            im = im.resize((box_size, box_size), Image.Resampling.BILINEAR)
            canvas.paste(im, (px,py))
            draw.text((px+3,py+box_size+9), f"{labels[r]} / {['Clean','L1','L2','L3'][c]}",
                      fill="#284252", font=font(14))
    canvas.save(path)

def process_tile(path: Path, output: Path, size: int, x: int | None, y: int | None,
                 joint: bool) -> dict:
    tile = read_tile(path)
    h, w, _ = tile.shape
    if size <= 0 or size > min(h,w):
        raise ValueError(f"Crop size must be in 1..{min(h,w)}, got {size}")
    x = (w-size)//2 if x is None else x
    y = (h-size)//2 if y is None else y
    tile_id = extract_tile_id(path)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem)
    target = output/slug
    target.mkdir(parents=True, exist_ok=True)
    original = np.array(tile[y:y+size, x:x+size], copy=True)
    if original.shape != (size,size,4):
        raise ValueError("Selected crop is out of bounds")
    Image.fromarray(original[..., :3]).save(target/"Clean_RGB.png")
    Image.fromarray(original[..., 3], "L").save(target/"Clean_NIR.png")
    rgb_rows, nir_rows = [], []
    for family in FAMILIES:
        rgb_line = [original[..., :3]]
        nir_line = [original[..., 3]]
        for level in LEVELS:
            out = corrupt_roi(tile, y, x, size, tile_id, family, level, joint)
            rgb_line.append(out[..., :3])
            nir_line.append(out[..., 3])
            Image.fromarray(out[..., :3]).save(target/f"{family}_{level}_RGB.png")
            Image.fromarray(out[..., 3], "L").save(target/f"{family}_{level}_NIR.png")
        rgb_rows.append(rgb_line)
        nir_rows.append(nir_line)
    names = ["Gaussian Noise", "Gaussian Blur", "Underexposure", "Atmospheric Fog"]
    montage(rgb_rows, target/"RGB_robust4_grid.png", names, "RGB", tile_id)
    montage(nir_rows, target/"NIR_robust4_grid.png", names, "NIR", tile_id)
    meta = {
        "input": str(path), "tile_id": tile_id, "tile_shape": list(tile.shape),
        "crop": {"x":x,"y":y,"size":size}, "mode":"joint" if joint else "rgb_only",
        "formal_full_6000_tile": bool(h==6000 and w==6000),
        "families": list(FAMILIES), "levels": list(LEVELS),
        "noise_sigma_255":NOISE, "blur_sigma":BLUR, "underexposure_alpha":EXPOSURE,
        "fog_rgb_transmission": {k:{"mean":v[0],"variation":v[1],
                                 "min":v[2],"max":v[3]} for k,v in FOG.items()},
        "nir_fog_ratio": NIR_FOG_RATIO, "seed_policy": "matches frozen Joint validation",
        "notes":"Original RGBIR source is required; no semantic predictions are produced."
    }
    (target/"protocol.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2), encoding="utf-8")
    return {"tile_id":tile_id,"output":str(target),"conditions":len(FAMILIES)*len(LEVELS)}

def main():
    ap = argparse.ArgumentParser(description="Potsdam 4x3 Robust4 RGB/NIR image visualization")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",type=Path,help="Single RGBIR TIFF")
    src.add_argument("--input-dir",type=Path,help="Directory containing RGBIR TIFFs")
    ap.add_argument("--output",type=Path,default=Path("robust4_vis"))
    ap.add_argument("--size",type=int,default=512)
    ap.add_argument("--x",type=int,default=None)
    ap.add_argument("--y",type=int,default=None)
    ap.add_argument("--max-tiles",type=int,default=1)
    ap.add_argument("--mode",choices=["joint","rgb_only"],default="joint")
    args = ap.parse_args()
    if args.max_tiles <= 0:
        ap.error("--max-tiles must be positive")
    if args.input:
        paths = [args.input]
    else:
        paths = sorted(p for p in args.input_dir.rglob("*")
                       if p.suffix.lower() in (".tif",".tiff") and "RGBIR" in p.stem.upper())
        paths = paths[:args.max_tiles]
    if not paths:
        ap.error("No RGBIR TIFF found; supply original top_potsdam_*_RGBIR.tif images")
    for p in paths:
        if not p.is_file():
            ap.error(f"Missing TIFF: {p}")
        result = process_tile(p,args.output,args.size,args.x,args.y,args.mode=="joint")
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
