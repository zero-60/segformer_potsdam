#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether anomalous Potsdam GT is an alpha overlay "
            "of clean GT and RGB orthophoto."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Potsdam raw root, e.g. data/raw/potsdam",
    )

    parser.add_argument(
        "--tile",
        default="4_12",
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/dataset_check/label_overlay"
        ),
    )

    return parser.parse_args()


def find_one(
    base: Path,
    basename: str,
) -> Path:
    matches = [
        p
        for p in base.rglob(basename)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {basename} under "
            f"{base}, found {len(matches)}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def load_rgbir(
    path: Path,
) -> np.ndarray:
    """
    Read Potsdam RGBIR using tifffile.

    Pillow cannot identify these PackBits-compressed 4-sample TIFFs,
    while tifffile reads them correctly.

    No conversion or modification is performed.
    """
    arr = tifffile.imread(path)

    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected RGBIR image with 3 dimensions, "
            f"got shape={arr.shape}"
        )

    if arr.shape[2] != 4:
        raise RuntimeError(
            f"Expected RGBIR image with 4 channels, "
            f"got shape={arr.shape}"
        )

    if arr.dtype != np.uint8:
        raise RuntimeError(
            f"Expected uint8 RGBIR, got dtype={arr.dtype}"
        )

    return arr


def load_gt_rgb(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as img:
        rgb = np.asarray(
            img.convert("RGB"),
            dtype=np.uint8,
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(
            f"Expected RGB GT, got shape={rgb.shape}"
        )

    return rgb


def fit_alpha_for_permutation(
    corrupt: np.ndarray,
    clean: np.ndarray,
    rgbir: np.ndarray,
    permutation: tuple[int, int, int],
    chunk_rows: int,
) -> dict:
    sum_xy = 0.0
    sum_x2 = 0.0

    channel_xy = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_x2 = np.zeros(
        3,
        dtype=np.float64,
    )

    height, width, _ = corrupt.shape

    for y0 in range(
        0,
        height,
        chunk_rows,
    ):
        y1 = min(
            y0 + chunk_rows,
            height,
        )

        bad = corrupt[
            y0:y1
        ].astype(
            np.float64
        )

        gt = clean[
            y0:y1
        ].astype(
            np.float64
        )

        rgb = rgbir[
            y0:y1,
            :,
            list(permutation),
        ].astype(
            np.float64
        )

        # Model:
        #
        # bad = alpha * clean_GT
        #       + (1-alpha) * RGB
        #
        # Therefore:
        #
        # bad - RGB =
        #   alpha * (clean_GT - RGB)

        x = gt - rgb
        y = bad - rgb

        sum_xy += float(
            np.sum(
                x * y,
                dtype=np.float64,
            )
        )

        sum_x2 += float(
            np.sum(
                x * x,
                dtype=np.float64,
            )
        )

        for c in range(3):
            channel_xy[c] += float(
                np.sum(
                    x[..., c]
                    * y[..., c],
                    dtype=np.float64,
                )
            )

            channel_x2[c] += float(
                np.sum(
                    x[..., c]
                    * x[..., c],
                    dtype=np.float64,
                )
            )

    alpha = (
        sum_xy / sum_x2
        if sum_x2 > 0
        else float("nan")
    )

    alpha_per_channel = []

    for c in range(3):
        if channel_x2[c] > 0:
            value = (
                channel_xy[c]
                / channel_x2[c]
            )
        else:
            value = float("nan")

        alpha_per_channel.append(
            float(value)
        )

    return {
        "permutation": list(permutation),
        "alpha": float(alpha),
        "alpha_per_channel": (
            alpha_per_channel
        ),
    }


def evaluate_fit(
    corrupt: np.ndarray,
    clean: np.ndarray,
    rgbir: np.ndarray,
    permutation: tuple[int, int, int],
    alpha: float,
    chunk_rows: int,
) -> dict:
    abs_sum = 0.0
    sq_sum = 0.0

    max_error = 0.0

    total_values = 0

    within_0_5 = 0
    within_1 = 0
    within_2 = 0
    within_3 = 0
    within_5 = 0

    per_channel_abs_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    per_channel_count = np.zeros(
        3,
        dtype=np.uint64,
    )

    height, width, _ = corrupt.shape

    for y0 in range(
        0,
        height,
        chunk_rows,
    ):
        y1 = min(
            y0 + chunk_rows,
            height,
        )

        bad = corrupt[
            y0:y1
        ].astype(
            np.float64
        )

        gt = clean[
            y0:y1
        ].astype(
            np.float64
        )

        rgb = rgbir[
            y0:y1,
            :,
            list(permutation),
        ].astype(
            np.float64
        )

        predicted = (
            alpha * gt
            + (1.0 - alpha) * rgb
        )

        error = np.abs(
            bad - predicted
        )

        abs_sum += float(
            error.sum(
                dtype=np.float64
            )
        )

        sq_sum += float(
            np.square(
                error
            ).sum(
                dtype=np.float64
            )
        )

        max_error = max(
            max_error,
            float(error.max()),
        )

        total_values += int(
            error.size
        )

        within_0_5 += int(
            (error <= 0.5).sum()
        )

        within_1 += int(
            (error <= 1.0).sum()
        )

        within_2 += int(
            (error <= 2.0).sum()
        )

        within_3 += int(
            (error <= 3.0).sum()
        )

        within_5 += int(
            (error <= 5.0).sum()
        )

        for c in range(3):
            per_channel_abs_sum[c] += float(
                error[
                    ...,
                    c
                ].sum(
                    dtype=np.float64
                )
            )

            per_channel_count[c] += (
                error[
                    ...,
                    c
                ].size
            )

    mae = (
        abs_sum
        / total_values
    )

    rmse = (
        sq_sum
        / total_values
    ) ** 0.5

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "max_abs_error": float(
            max_error
        ),
        "fraction_abs_error_le_0_5": (
            within_0_5
            / total_values
        ),
        "fraction_abs_error_le_1": (
            within_1
            / total_values
        ),
        "fraction_abs_error_le_2": (
            within_2
            / total_values
        ),
        "fraction_abs_error_le_3": (
            within_3
            / total_values
        ),
        "fraction_abs_error_le_5": (
            within_5
            / total_values
        ),
        "per_channel_mae": [
            float(
                per_channel_abs_sum[c]
                / per_channel_count[c]
            )
            for c in range(3)
        ],
    }


def main() -> int:
    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.chunk_rows <= 0:
        raise SystemExit(
            "ERROR: --chunk-rows must be > 0"
        )

    tile = args.tile

    all_root = (
        root
        / "_expanded"
        / "5_Labels_all"
    )

    participant_root = (
        root
        / "_expanded"
        / "5_Labels_for_participants"
    )

    rgbir_root = (
        root
        / "_expanded"
        / "4_Ortho_RGBIR"
    )

    label_name = (
        f"top_potsdam_{tile}_label.tif"
    )

    rgbir_name = (
        f"top_potsdam_{tile}_RGBIR.tif"
    )

    corrupt_path = find_one(
        all_root,
        label_name,
    )

    clean_path = find_one(
        participant_root,
        label_name,
    )

    rgbir_path = find_one(
        rgbir_root,
        rgbir_name,
    )

    print("=== Files ===")
    print(
        f"Corrupt GT : {corrupt_path}"
    )
    print(
        f"Clean GT   : {clean_path}"
    )
    print(
        f"RGBIR      : {rgbir_path}"
    )

    print()
    print(
        "=== Loading images ==="
    )

    corrupt = load_gt_rgb(
        corrupt_path
    )

    clean = load_gt_rgb(
        clean_path
    )

    rgbir = load_rgbir(
        rgbir_path
    )

    print(
        f"Corrupt GT : "
        f"shape={corrupt.shape}, "
        f"dtype={corrupt.dtype}"
    )

    print(
        f"Clean GT   : "
        f"shape={clean.shape}, "
        f"dtype={clean.dtype}"
    )

    print(
        f"RGBIR      : "
        f"shape={rgbir.shape}, "
        f"dtype={rgbir.dtype}"
    )

    if (
        corrupt.shape[:2]
        != clean.shape[:2]
        or corrupt.shape[:2]
        != rgbir.shape[:2]
    ):
        raise RuntimeError(
            "Spatial dimensions do not match:\n"
            f"corrupt={corrupt.shape}\n"
            f"clean={clean.shape}\n"
            f"rgbir={rgbir.shape}"
        )

    print()
    print(
        "RGBIR reader: tifffile"
    )
    print(
        "No RGBIR conversion was performed."
    )

    candidates = []

    print()
    print(
        "=== Fitting scalar alpha ==="
    )

    # We test all permutations of the first three stored bands.
    # This avoids assuming channel order during the diagnostic.
    for permutation in itertools.permutations(
        (0, 1, 2)
    ):
        fit = fit_alpha_for_permutation(
            corrupt,
            clean,
            rgbir,
            permutation,
            args.chunk_rows,
        )

        evaluation = evaluate_fit(
            corrupt,
            clean,
            rgbir,
            permutation,
            fit["alpha"],
            args.chunk_rows,
        )

        candidate = {
            **fit,
            **evaluation,
        }

        candidates.append(
            candidate
        )

        print(
            f"perm={permutation} "
            f"alpha={fit['alpha']:.8f} "
            f"MAE={evaluation['mae']:.6f} "
            f"RMSE={evaluation['rmse']:.6f} "
            f"<=1={evaluation['fraction_abs_error_le_1']:.6%}"
        )

    candidates.sort(
        key=lambda x: x["mae"]
    )

    best = candidates[0]

    print()
    print(
        "=== Best overlay model ==="
    )

    print(
        f"RGB band permutation : "
        f"{tuple(best['permutation'])}"
    )

    print(
        f"alpha GT             : "
        f"{best['alpha']:.10f}"
    )

    print(
        f"alpha RGB            : "
        f"{1.0 - best['alpha']:.10f}"
    )

    print(
        "alpha per channel    : "
        + ", ".join(
            f"{x:.10f}"
            for x in best[
                "alpha_per_channel"
            ]
        )
    )

    print(
        f"MAE                  : "
        f"{best['mae']:.8f}"
    )

    print(
        f"RMSE                 : "
        f"{best['rmse']:.8f}"
    )

    print(
        f"Max abs error        : "
        f"{best['max_abs_error']:.8f}"
    )

    print(
        f"|error| <= 0.5       : "
        f"{best['fraction_abs_error_le_0_5']:.6%}"
    )

    print(
        f"|error| <= 1         : "
        f"{best['fraction_abs_error_le_1']:.6%}"
    )

    print(
        f"|error| <= 2         : "
        f"{best['fraction_abs_error_le_2']:.6%}"
    )

    print(
        f"|error| <= 3         : "
        f"{best['fraction_abs_error_le_3']:.6%}"
    )

    print(
        f"|error| <= 5         : "
        f"{best['fraction_abs_error_le_5']:.6%}"
    )

    print(
        "Per-channel MAE      : "
        + ", ".join(
            f"{x:.8f}"
            for x in best[
                "per_channel_mae"
            ]
        )
    )

    output = {
        "tile": tile,
        "corrupt_gt": str(
            corrupt_path
        ),
        "clean_gt": str(
            clean_path
        ),
        "rgbir": str(
            rgbir_path
        ),
        "rgbir_reader": (
            "tifffile"
        ),
        "rgbir_shape": list(
            rgbir.shape
        ),
        "rgbir_dtype": str(
            rgbir.dtype
        ),
        "model": (
            "corrupt = alpha * clean_GT "
            "+ (1-alpha) * RGB"
        ),
        "note": (
            "First three stored RGBIR bands were "
            "tested under all 6 permutations. "
            "No label repair/remapping performed."
        ),
        "candidates": candidates,
        "best": best,
    }

    output_path = (
        out_dir
        / f"overlay_{tile}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== Saved ==="
    )
    print(output_path)

    print()
    print(
        "No image or GT file was modified."
    )
    print(
        "No approximate label-color "
        "remapping was performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
