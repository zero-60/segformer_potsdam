import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
"""
查找 RGBIR TIFF
查找 GT TIFF
提取 tile ID
匹配 RGBIR ↔ GT
检查 shape
检查 dtype
检查是否为 4 通道
打印 R/G/B/NIR min/max/mean
检查 GT 的 RGB class colors
发现重复 label 数据源
发现 RGBIR 没有 GT 的 tile

"""

EXPECTED_LABEL_COLORS = {
    (255, 255, 255): "Impervious surfaces",
    (0, 0, 255): "Building",
    (0, 255, 255): "Low vegetation",
    (0, 255, 0): "Tree",
    (255, 255, 0): "Car",
    (255, 0, 0): "Clutter / Background",
}

CHANNEL_NAMES = ["R", "G", "B", "NIR"]


def extract_tile_id(path: Path):
    """
    Example:
        top_potsdam_2_10_RGBIR.tif
        -> 2_10
    """
    text = path.as_posix().lower()

    match = re.search(r"potsdam_(\d+_\d+)", text)

    if match is None:
        return None

    return match.group(1)


def natural_tile_key(tile_id):
    a, b = tile_id.split("_")
    return int(a), int(b)


def is_no_boundary(path: Path):
    text = path.as_posix().lower()

    patterns = [
        "noboundary",
        "no_boundary",
        "no-boundary",
        "no boundary",
    ]

    return any(pattern in text for pattern in patterns)


def find_tiff_files(root: Path):
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in {".tif", ".tiff"}:
            files.append(path)

    return sorted(files)


def ensure_hwc(array):
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D multi-channel TIFF, "
            f"but received shape {array.shape}"
        )

    # Normal HWC
    if array.shape[-1] in (3, 4):
        return array

    # Sometimes TIFF libraries may return CHW
    if array.shape[0] in (3, 4):
        return np.moveaxis(array, 0, -1)

    raise ValueError(
        f"Cannot identify channel dimension from shape {array.shape}"
    )


def tiff_metadata(path: Path):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(series.shape), np.dtype(series.dtype)


def shape_to_hw(shape):
    if len(shape) == 2:
        return shape

    if len(shape) != 3:
        return None

    if shape[-1] in (1, 3, 4):
        return shape[0], shape[1]

    if shape[0] in (1, 3, 4):
        return shape[1], shape[2]

    return None


def inspect_rgbir(path: Path):
    print()
    print("-" * 80)
    print("RGBIR")
    print("-" * 80)
    print(f"Path  : {path}")

    array = tifffile.imread(path)
    print(f"Raw shape : {array.shape}")
    print(f"Dtype     : {array.dtype}")

    array = ensure_hwc(array)

    print(f"HWC shape : {array.shape}")

    h, w, c = array.shape

    issues = []

    if c != 4:
        issues.append(
            f"RGBIR image has {c} channels instead of 4"
        )

    if (h, w) != (6000, 6000):
        issues.append(
            f"Image size is {(h, w)}, expected 6000x6000 "
            f"for original Potsdam tiles"
        )

    if np.issubdtype(array.dtype, np.floating):
        if np.isnan(array).any():
            issues.append("NaN detected")

        if np.isinf(array).any():
            issues.append("Inf detected")

    for index in range(c):
        channel = array[..., index]

        if index < len(CHANNEL_NAMES):
            name = CHANNEL_NAMES[index]
        else:
            name = f"channel_{index}"

        print(
            f"{name:>5s} | "
            f"min={channel.min():8.3f} "
            f"max={channel.max():8.3f} "
            f"mean={channel.mean():8.3f}"
        )

    return issues


def inspect_label(path: Path):
    print()
    print("-" * 80)
    print("LABEL")
    print("-" * 80)
    print(f"Path : {path}")

    issues = []

    with Image.open(path) as image:
        print(f"PIL mode : {image.mode}")
        print(f"Size     : {image.size}")

        rgb = image.convert("RGB")

        # Potsdam labels should contain only a very small number
        # of discrete colors.
        colors = rgb.getcolors(maxcolors=4096)

    if colors is None:
        issues.append(
            "More than 4096 colors found in label. "
            "This is unexpected for a class-index color mask."
        )
        print("Unique colors: >4096")
        return issues

    colors = sorted(colors, key=lambda x: x[0], reverse=True)

    total_pixels = sum(count for count, _ in colors)

    print(f"Unique RGB colors : {len(colors)}")
    print()

    unknown_colors = []

    for count, color in colors:
        class_name = EXPECTED_LABEL_COLORS.get(color)

        percentage = 100.0 * count / total_pixels

        if class_name is None:
            class_name = "UNKNOWN"
            unknown_colors.append((color, count))

        print(
            f"{color}  "
            f"{class_name:<22s}  "
            f"{count:>10d} px  "
            f"{percentage:7.3f}%"
        )

    if unknown_colors:
        issues.append(
            "Unknown label colors detected: "
            + ", ".join(str(color) for color, _ in unknown_colors)
        )

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw ISPRS Potsdam RGBIR and label files."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/potsdam",
        help="Root directory containing extracted Potsdam data.",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=3,
        help=(
            "Number of RGBIR and label files to fully inspect. "
            "Use 0 to inspect every file."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when an issue is detected.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    print("=" * 80)
    print("ISPRS Potsdam Dataset Inspection")
    print("=" * 80)
    print(f"Root: {root}")

    all_tiffs = find_tiff_files(root)

    rgbir_files = [
        path
        for path in all_tiffs
        if "rgbir" in path.name.lower()
    ]

    label_files = [
        path
        for path in all_tiffs
        if "label" in path.name.lower()
    ]

    full_label_files = [
        path
        for path in label_files
        if not is_no_boundary(path)
    ]

    no_boundary_files = [
        path
        for path in label_files
        if is_no_boundary(path)
    ]

    print()
    print("===== File summary =====")
    print(f"All TIFF files      : {len(all_tiffs)}")
    print(f"RGBIR TIFF files    : {len(rgbir_files)}")
    print(f"Full label files    : {len(full_label_files)}")
    print(f"No-boundary labels  : {len(no_boundary_files)}")

    print()
    print("===== RGBIR directories =====")

    rgbir_dirs = Counter(
        str(path.parent.relative_to(root))
        for path in rgbir_files
    )

    if rgbir_dirs:
        for directory, count in rgbir_dirs.most_common():
            print(f"{count:3d}  {directory}")
    else:
        print("No RGBIR TIFF found.")

    print()
    print("===== Label directories =====")

    label_dirs = Counter(
        str(path.parent.relative_to(root))
        for path in full_label_files
    )

    if label_dirs:
        for directory, count in label_dirs.most_common():
            print(f"{count:3d}  {directory}")
    else:
        print("No full label TIFF found.")

    rgbir_by_tile = defaultdict(list)
    label_by_tile = defaultdict(list)

    unparsed_rgbir = []
    unparsed_labels = []

    for path in rgbir_files:
        tile = extract_tile_id(path)

        if tile is None:
            unparsed_rgbir.append(path)
        else:
            rgbir_by_tile[tile].append(path)

    for path in full_label_files:
        tile = extract_tile_id(path)

        if tile is None:
            unparsed_labels.append(path)
        else:
            label_by_tile[tile].append(path)

    image_tiles = set(rgbir_by_tile)
    label_tiles = set(label_by_tile)

    paired_tiles = sorted(
        image_tiles & label_tiles,
        key=natural_tile_key,
    )

    image_only_tiles = sorted(
        image_tiles - label_tiles,
        key=natural_tile_key,
    )

    label_only_tiles = sorted(
        label_tiles - image_tiles,
        key=natural_tile_key,
    )

    print()
    print("===== Tile matching =====")
    print(f"Unique RGBIR tiles : {len(image_tiles)}")
    print(f"Unique label tiles : {len(label_tiles)}")
    print(f"Paired tiles       : {len(paired_tiles)}")

    if image_only_tiles:
        print(
            "RGBIR without label : "
            + ", ".join(image_only_tiles)
        )

    if label_only_tiles:
        print(
            "Label without RGBIR : "
            + ", ".join(label_only_tiles)
        )

    duplicate_rgbir = {
        tile: paths
        for tile, paths in rgbir_by_tile.items()
        if len(paths) > 1
    }

    duplicate_labels = {
        tile: paths
        for tile, paths in label_by_tile.items()
        if len(paths) > 1
    }

    issues = []

    if len(rgbir_files) == 0:
        issues.append("No RGBIR TIFF files found.")

    if len(full_label_files) == 0:
        issues.append("No full label TIFF files found.")

    if duplicate_rgbir:
        print()
        print("===== Duplicate RGBIR tile IDs =====")

        for tile, paths in duplicate_rgbir.items():
            print(tile)
            for path in paths:
                print(f"  {path}")

        issues.append(
            "Duplicate RGBIR files found for one or more tile IDs."
        )

    if duplicate_labels:
        print()
        print("===== Duplicate full-label tile IDs =====")

        for tile, paths in duplicate_labels.items():
            print(tile)
            for path in paths:
                print(f"  {path}")

        issues.append(
            "Multiple full-label sources exist for one or more tile IDs. "
            "Choose one label source before training."
        )

    # Metadata-only spatial shape check for every matched tile.
    print()
    print("===== RGBIR / GT spatial alignment =====")

    alignment_failures = []

    for tile in paired_tiles:
        image_path = rgbir_by_tile[tile][0]
        label_path = label_by_tile[tile][0]

        image_shape, image_dtype = tiff_metadata(image_path)
        image_hw = shape_to_hw(image_shape)

        with Image.open(label_path) as label_image:
            label_hw = (
                label_image.height,
                label_image.width,
            )

        if image_hw != label_hw:
            alignment_failures.append(
                (tile, image_hw, label_hw)
            )

    if alignment_failures:
        for tile, image_hw, label_hw in alignment_failures:
            print(
                f"[FAIL] tile={tile} "
                f"RGBIR={image_hw} GT={label_hw}"
            )

        issues.append(
            "RGBIR and GT spatial sizes do not match "
            "for one or more tiles."
        )
    else:
        print(
            f"All {len(paired_tiles)} paired tiles "
            f"have matching spatial dimensions."
        )

    # Full pixel/statistics inspection.
    if args.max_items == 0:
        selected_rgbir = rgbir_files
        selected_labels = full_label_files
    else:
        selected_rgbir = rgbir_files[: args.max_items]
        selected_labels = full_label_files[: args.max_items]

    print()
    print("=" * 80)
    print(
        f"Inspecting {len(selected_rgbir)} RGBIR files "
        f"and {len(selected_labels)} label files in detail"
    )
    print("=" * 80)

    for path in selected_rgbir:
        issues.extend(inspect_rgbir(path))

    for path in selected_labels:
        issues.extend(inspect_label(path))

    print()
    print("=" * 80)

    if issues:
        print("Inspection completed with warnings/issues:")
        print()

        for index, issue in enumerate(issues, start=1):
            print(f"{index:2d}. {issue}")

        print()
        print(
            "Do not start training until relevant dataset "
            "issues have been resolved."
        )

        if args.strict:
            sys.exit(2)

    else:
        print("Dataset quick inspection looks good.")

    print("=" * 80)


if __name__ == "__main__":
    main()
