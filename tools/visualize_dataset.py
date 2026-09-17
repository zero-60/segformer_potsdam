import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


def extract_tile_id(path: Path):
    match = re.search(
        r"potsdam_(\d+_\d+)",
        path.as_posix().lower(),
    )

    if match is None:
        return None

    return match.group(1)


def natural_tile_key(tile_id):
    a, b = tile_id.split("_")
    return int(a), int(b)


def is_no_boundary(path: Path):
    text = path.as_posix().lower()

    return any(
        token in text
        for token in [
            "noboundary",
            "no_boundary",
            "no-boundary",
            "no boundary",
        ]
    )


def ensure_hwc(array):
    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {array.shape}"
        )

    if array.shape[-1] in (3, 4):
        return array

    if array.shape[0] in (3, 4):
        return np.moveaxis(array, 0, -1)

    raise ValueError(
        f"Cannot determine channel dimension: {array.shape}"
    )


def find_tiffs(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    )


def resize_for_display(array, max_side=1600, nearest=False):
    h, w = array.shape[:2]

    scale = min(
        1.0,
        max_side / max(h, w),
    )

    if scale == 1.0:
        return array

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    image = Image.fromarray(array)

    if nearest:
        resample = Image.Resampling.NEAREST
    else:
        resample = Image.Resampling.BILINEAR

    image = image.resize(
        (new_w, new_h),
        resample=resample,
    )

    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data/raw/potsdam",
        type=str,
    )

    parser.add_argument(
        "--tile",
        default=None,
        type=str,
        help="Example: 2_10. Default: first paired tile.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/dataset_check",
        type=str,
    )

    parser.add_argument(
        "--max-display-size",
        default=1600,
        type=int,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_tiffs(root)

    rgbir_files = [
        path
        for path in files
        if "rgbir" in path.name.lower()
    ]

    full_labels = [
        path
        for path in files
        if "label" in path.name.lower()
        and not is_no_boundary(path)
    ]

    rgbir_by_tile = defaultdict(list)
    label_by_tile = defaultdict(list)

    for path in rgbir_files:
        tile = extract_tile_id(path)

        if tile is not None:
            rgbir_by_tile[tile].append(path)

    for path in full_labels:
        tile = extract_tile_id(path)

        if tile is not None:
            label_by_tile[tile].append(path)

    paired = sorted(
        set(rgbir_by_tile) & set(label_by_tile),
        key=natural_tile_key,
    )

    if not paired:
        raise RuntimeError(
            "No RGBIR/label paired tile was found."
        )

    if args.tile is None:
        tile = paired[0]
    else:
        tile = args.tile

        if tile not in paired:
            raise ValueError(
                f"Tile {tile} is not available as an RGBIR/GT pair.\n"
                f"Available examples: {paired[:20]}"
            )

    image_path = rgbir_by_tile[tile][0]
    label_path = label_by_tile[tile][0]

    print("=" * 70)
    print(f"Tile       : {tile}")
    print(f"RGBIR      : {image_path}")
    print(f"Ground Truth: {label_path}")
    print("=" * 70)

    rgbir = tifffile.imread(image_path)
    rgbir = ensure_hwc(rgbir)

    if rgbir.shape[-1] != 4:
        raise RuntimeError(
            f"Expected four channels, got {rgbir.shape}"
        )

    rgb = rgbir[..., :3]
    nir = rgbir[..., 3]

    # NIR-R-G false-color composition
    false_color = np.stack(
        [
            nir,
            rgb[..., 0],
            rgb[..., 1],
        ],
        axis=-1,
    )

    with Image.open(label_path) as image:
        gt = np.asarray(image.convert("RGB"))

    if rgb.shape[:2] != gt.shape[:2]:
        raise RuntimeError(
            f"Spatial mismatch: RGBIR={rgb.shape[:2]}, "
            f"GT={gt.shape[:2]}"
        )

    rgb_show = resize_for_display(
        rgb,
        args.max_display_size,
    )

    nir_show = resize_for_display(
        nir,
        args.max_display_size,
    )

    false_show = resize_for_display(
        false_color,
        args.max_display_size,
    )

    gt_show = resize_for_display(
        gt,
        args.max_display_size,
        nearest=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 14),
    )

    axes[0, 0].imshow(rgb_show)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        nir_show,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[0, 1].set_title("NIR")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(false_show)
    axes[1, 0].set_title("False Color (NIR-R-G)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_show)
    axes[1, 1].set_title("Ground Truth")
    axes[1, 1].axis("off")

    fig.suptitle(
        f"ISPRS Potsdam Tile {tile}",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"potsdam_{tile}_rgb_nir_gt.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {output_path.resolve()}")
    print("Visualization: PASS")


if __name__ == "__main__":
    main()
