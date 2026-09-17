#!/usr/bin/env python3
"""
Audit ISPRS Potsdam ground-truth label sources.

This script is diagnostic only.

It does NOT:
- remap approximate RGB colors
- modify label files
- select a training GT source
- crop patches
- start training

It does:
- discover all extracted 5_Labels* directories
- enumerate every Potsdam label TIFF
- group files by source and tile ID
- inspect image shape/mode/dtype/TIFF compression
- count exact RGB colors
- report every unknown RGB color and pixel count
- report colors close to official class colors WITHOUT remapping them
- locate target color (252, 255, 0)
- detect duplicate full-label sources
- compare duplicates using file SHA256 and decoded-pixel SHA256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


# Potsdam tiles are 6000 x 6000.
EXPECTED_SIZE = (6000, 6000)

# Official ISPRS semantic RGB colors.
# These are exact categorical colors, not approximate colors.
STANDARD_COLORS: dict[str, tuple[int, int, int]] = {
    "impervious_surface": (255, 255, 255),
    "building":           (0, 0, 255),
    "low_vegetation":     (0, 255, 255),
    "tree":               (0, 255, 0),
    "car":                (255, 255, 0),
    "clutter_background": (255, 0, 0),
}

# noBoundary / eroded reference masks commonly use black as an
# ignored/boundary/void marker. It is NOT a seventh semantic class here.
BOUNDARY_VOID_COLOR = (0, 0, 0)

# Specific suspicious color already observed by check_dataset.py.
TARGET_COLOR = (252, 255, 0)

TILE_RE = re.compile(
    r"top_potsdam_(\d+)_(\d+)",
    flags=re.IGNORECASE,
)

SOURCE_ORDER = [
    "5_Labels_all",
    "5_Labels_all_noBoundary",
    "5_Labels_for_participants",
    "5_Labels_for_participants_no_Boundary",
    "UNKNOWN_LABEL_SOURCE",
]

# 6000x6000 = 36M pixels. Disable Pillow's generic large-image warning;
# we explicitly expect this image size.
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect all ISPRS Potsdam GT label sources."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Potsdam raw-data root, e.g. data/raw/potsdam",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/dataset_check/label_audit"),
        help="Directory for audit reports.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=8,
        help=(
            "Diagnostic L-infinity RGB distance used only to REPORT colors "
            "near an official color. No remapping is performed."
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="Rows decoded at a time for exact RGB histogram.",
    )
    return parser.parse_args()


def tile_sort_key(tile_id: str) -> tuple[int, int]:
    a, b = tile_id.split("_", 1)
    return int(a), int(b)


def extract_tile_id(path: Path) -> str | None:
    match = TILE_RE.search(path.name)
    if match is None:
        return None
    return f"{int(match.group(1))}_{int(match.group(2))}"


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def canonical_source(path: Path) -> str:
    """
    Identify source package from the entire path.

    Order matters: noBoundary variants must be checked before ordinary variants.
    """
    s = compact_text(str(path))

    if "5labelsforparticipantsnoboundary" in s:
        return "5_Labels_for_participants_no_Boundary"

    if "5labelsallnoboundary" in s:
        return "5_Labels_all_noBoundary"

    if "5labelsforparticipants" in s:
        return "5_Labels_for_participants"

    if "5labelsall" in s:
        return "5_Labels_all"

    return "UNKNOWN_LABEL_SOURCE"


def is_no_boundary(path: Path) -> bool:
    return "noboundary" in compact_text(str(path))


def source_kind(path: Path) -> str:
    return "noBoundary" if is_no_boundary(path) else "full"


def rgb_to_id(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (int(r) << 16) | (int(g) << 8) | int(b)


def id_to_rgb(value: int) -> tuple[int, int, int]:
    value = int(value)
    return (
        (value >> 16) & 255,
        (value >> 8) & 255,
        value & 255,
    )


STANDARD_IDS = {
    rgb_to_id(rgb): name
    for name, rgb in STANDARD_COLORS.items()
}

TARGET_ID = rgb_to_id(TARGET_COLOR)
BOUNDARY_VOID_ID = rgb_to_id(BOUNDARY_VOID_COLOR)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def discover_label_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        name = compact_text(p.name)

        if name.startswith("5labels"):
            dirs.append(p)

    return sorted(set(dirs), key=lambda p: str(p))


def discover_label_files(root: Path) -> list[Path]:
    """
    Find Potsdam label TIFFs.

    Require:
    - TIFF suffix
    - "label" in filename
    - a valid top_potsdam_ROW_COL tile ID
    """
    found: dict[str, Path] = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {".tif", ".tiff"}:
            continue

        if "label" not in p.name.lower():
            continue

        if extract_tile_id(p) is None:
            continue

        # resolved path prevents accidental duplicate discovery through symlinks.
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p.absolute())

        found[key] = p

    return sorted(
        found.values(),
        key=lambda p: (
            tile_sort_key(extract_tile_id(p) or "999_999"),
            canonical_source(p),
            str(p),
        ),
    )


def count_label_tiffs(directory: Path) -> int:
    count = 0

    for p in directory.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and "label" in p.name.lower()
            and extract_tile_id(p) is not None
        ):
            count += 1

    return count


def find_outermost_label_dir(path: Path) -> str:
    """
    Example:
        .../_expanded/5_Labels_all/5_Labels_all/file.tif

    Returns the outermost matching 5_Labels* directory.
    """
    matches = [
        p
        for p in path.parents
        if compact_text(p.name).startswith("5labels")
    ]

    if not matches:
        return ""

    # path.parents is nearest -> farthest.
    return str(matches[-1])


def nearest_standard_batch(
    rgb_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rgb_array: shape [N, 3], uint8/int

    Returns:
        nearest_index
        L_inf distance
        L1 distance
    """
    standard_names = list(STANDARD_COLORS.keys())
    standard_rgb = np.asarray(
        [STANDARD_COLORS[name] for name in standard_names],
        dtype=np.int16,
    )

    x = rgb_array.astype(np.int16, copy=False)

    # [N, 6, 3]
    diff = np.abs(
        x[:, None, :] - standard_rgb[None, :, :]
    )

    l1 = diff.sum(axis=2)
    linf = diff.max(axis=2)

    # Nearest class by L1 distance.
    nearest_idx = np.argmin(l1, axis=1)

    rows = np.arange(len(x))
    nearest_l1 = l1[rows, nearest_idx]
    nearest_linf = linf[rows, nearest_idx]

    return nearest_idx, nearest_linf, nearest_l1


def inspect_one_label(
    path: Path,
    near_threshold: int,
    chunk_rows: int,
    unknown_writer: csv.DictWriter,
    near_writer: csv.DictWriter,
) -> dict[str, Any]:
    tile_id = extract_tile_id(path)
    source = canonical_source(path)
    kind = source_kind(path)
    no_boundary = kind == "noBoundary"

    record: dict[str, Any] = {
        "path": str(path),
        "tile_id": tile_id,
        "source": source,
        "kind": kind,
        "source_dir": find_outermost_label_dir(path),
        "status": "OK",
        "error": "",
        "width": None,
        "height": None,
        "shape": None,
        "mode": None,
        "dtype": None,
        "bands": None,
        "compression": None,
        "compression_tag_259": None,
        "photometric_tag_262": None,
        "unique_rgb_count": None,
        "unknown_unique_count": None,
        "unknown_pixel_count": None,
        "near_standard_unique_count": None,
        "near_standard_pixel_count": None,
        "target_252_255_0_count": None,
        "boundary_void_0_0_0_count": None,
        "file_sha256": None,
        "pixel_sha256": None,
        "standard_color_counts": {},
        "flags": [],
    }

    try:
        record["file_sha256"] = file_sha256(path)

        with Image.open(path) as img:
            img.seek(0)

            width, height = img.size
            mode = img.mode
            bands = tuple(img.getbands())

            one_pixel = np.asarray(
                img.crop((0, 0, 1, 1))
            )

            dtype = str(one_pixel.dtype)

            if len(bands) == 1:
                raw_shape = [height, width]
            else:
                raw_shape = [height, width, len(bands)]

            record["width"] = int(width)
            record["height"] = int(height)
            record["shape"] = raw_shape
            record["mode"] = mode
            record["dtype"] = dtype
            record["bands"] = list(bands)

            compression = img.info.get("compression")
            record["compression"] = (
                None if compression is None else str(compression)
            )

            try:
                record["compression_tag_259"] = str(
                    img.tag_v2.get(259)
                )
            except Exception:
                record["compression_tag_259"] = None

            try:
                record["photometric_tag_262"] = str(
                    img.tag_v2.get(262)
                )
            except Exception:
                record["photometric_tag_262"] = None

            if (width, height) != EXPECTED_SIZE:
                record["flags"].append(
                    f"unexpected_size:{width}x{height}"
                )

            if mode != "RGB":
                record["flags"].append(
                    f"unexpected_mode:{mode}"
                )

            if dtype != "uint8":
                record["flags"].append(
                    f"unexpected_dtype:{dtype}"
                )

            # Exact 24-bit RGB histogram:
            # 2^24 uint32 entries = 64 MiB.
            histogram = np.zeros(
                1 << 24,
                dtype=np.uint32,
            )

            pixel_hasher = hashlib.sha256()

            for y0 in range(0, height, chunk_rows):
                y1 = min(y0 + chunk_rows, height)

                crop = img.crop(
                    (0, y0, width, y1)
                ).convert("RGB")

                rgb = np.asarray(
                    crop,
                    dtype=np.uint8,
                )

                # Exact decoded pixel hash.
                pixel_hasher.update(
                    rgb.tobytes(order="C")
                )

                packed = (
                    (rgb[..., 0].astype(np.uint32) << 16)
                    | (rgb[..., 1].astype(np.uint32) << 8)
                    | rgb[..., 2].astype(np.uint32)
                )

                values, counts = np.unique(
                    packed.reshape(-1),
                    return_counts=True,
                )

                histogram[values] += counts.astype(
                    np.uint32,
                    copy=False,
                )

            record["pixel_sha256"] = pixel_hasher.hexdigest()

            used_ids = np.flatnonzero(histogram)
            unique_rgb_count = int(used_ids.size)

            record["unique_rgb_count"] = unique_rgb_count

            if unique_rgb_count > 4096:
                record["flags"].append(
                    f"more_than_4096_colors:{unique_rgb_count}"
                )

            standard_counts = {}

            for color_id, class_name in STANDARD_IDS.items():
                standard_counts[class_name] = int(
                    histogram[color_id]
                )

            record["standard_color_counts"] = standard_counts

            boundary_void_count = int(
                histogram[BOUNDARY_VOID_ID]
            )
            record["boundary_void_0_0_0_count"] = (
                boundary_void_count
            )

            target_count = int(
                histogram[TARGET_ID]
            )
            record["target_252_255_0_count"] = target_count

            if target_count > 0:
                record["flags"].append(
                    f"contains_252_255_0:{target_count}"
                )

            allowed_ids = set(STANDARD_IDS.keys())

            # Black is accepted only as a boundary/void diagnostic marker
            # for noBoundary masks. It is not a semantic training class.
            if no_boundary:
                allowed_ids.add(BOUNDARY_VOID_ID)

            allowed_array = np.asarray(
                sorted(allowed_ids),
                dtype=np.int64,
            )

            unknown_mask = ~np.isin(
                used_ids,
                allowed_array,
            )

            unknown_ids = used_ids[unknown_mask]

            record["unknown_unique_count"] = int(
                unknown_ids.size
            )

            if unknown_ids.size:
                unknown_pixel_count = int(
                    histogram[unknown_ids].astype(
                        np.uint64
                    ).sum()
                )
            else:
                unknown_pixel_count = 0

            record["unknown_pixel_count"] = (
                unknown_pixel_count
            )

            if unknown_ids.size > 0:
                record["flags"].append(
                    f"unknown_colors:{int(unknown_ids.size)}"
                )

            near_unique_count = 0
            near_pixel_count = 0

            standard_names = list(
                STANDARD_COLORS.keys()
            )
            standard_rgbs = [
                STANDARD_COLORS[name]
                for name in standard_names
            ]

            # Process unknown colors in batches so even a badly corrupted
            # image with huge color cardinality remains manageable.
            batch_size = 200_000

            for start in range(
                0,
                len(unknown_ids),
                batch_size,
            ):
                ids = unknown_ids[
                    start:start + batch_size
                ]

                if len(ids) == 0:
                    continue

                r = ((ids >> 16) & 255).astype(
                    np.uint8
                )
                g = ((ids >> 8) & 255).astype(
                    np.uint8
                )
                b = (ids & 255).astype(
                    np.uint8
                )

                rgb_batch = np.stack(
                    [r, g, b],
                    axis=1,
                )

                nearest_idx, nearest_linf, nearest_l1 = (
                    nearest_standard_batch(
                        rgb_batch
                    )
                )

                counts = histogram[ids]

                for i in range(len(ids)):
                    rgb_tuple = (
                        int(rgb_batch[i, 0]),
                        int(rgb_batch[i, 1]),
                        int(rgb_batch[i, 2]),
                    )

                    count = int(counts[i])
                    nearest_name = standard_names[
                        int(nearest_idx[i])
                    ]
                    nearest_rgb = standard_rgbs[
                        int(nearest_idx[i])
                    ]

                    linf = int(nearest_linf[i])
                    l1 = int(nearest_l1[i])

                    is_near = (
                        linf <= near_threshold
                    )

                    row = {
                        "path": str(path),
                        "tile_id": tile_id,
                        "source": source,
                        "kind": kind,
                        "r": rgb_tuple[0],
                        "g": rgb_tuple[1],
                        "b": rgb_tuple[2],
                        "pixel_count": count,
                        "nearest_standard_class": nearest_name,
                        "nearest_standard_rgb": str(
                            tuple(nearest_rgb)
                        ),
                        "distance_Linf": linf,
                        "distance_L1": l1,
                        "within_near_threshold": is_near,
                    }

                    # Every unknown RGB goes here.
                    unknown_writer.writerow(row)

                    # Diagnostic only. Still NOT mapped.
                    if is_near:
                        near_writer.writerow(row)
                        near_unique_count += 1
                        near_pixel_count += count

            record["near_standard_unique_count"] = (
                int(near_unique_count)
            )
            record["near_standard_pixel_count"] = (
                int(near_pixel_count)
            )

            # Add compression hint as a flag if Pillow identifies JPEG.
            compression_string = str(
                record["compression"]
            ).lower()

            if "jpeg" in compression_string:
                record["flags"].append(
                    f"lossy_tiff_compression:{record['compression']}"
                )

            if record["flags"]:
                record["status"] = "ANOMALY"

            # Explicit cleanup before the next 6000x6000 file.
            del histogram
            del used_ids
            del unknown_ids

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        RuntimeError,
        MemoryError,
    ) as exc:
        record["status"] = "ERROR"
        record["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        record["flags"].append(
            f"read_error:{type(exc).__name__}"
        )

    return record


def build_source_stats(
    files: list[Path],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in files:
        grouped[canonical_source(path)].append(path)

    stats: dict[str, dict[str, Any]] = {}

    for source in SOURCE_ORDER:
        source_files = grouped.get(source, [])

        tile_ids = sorted(
            {
                tile
                for p in source_files
                if (tile := extract_tile_id(p))
                is not None
            },
            key=tile_sort_key,
        )

        per_tile: dict[str, list[str]] = defaultdict(list)

        for p in source_files:
            tile = extract_tile_id(p)
            if tile is not None:
                per_tile[tile].append(str(p))

        duplicate_within_source = {
            tile: paths
            for tile, paths in per_tile.items()
            if len(paths) > 1
        }

        stats[source] = {
            "file_count": len(source_files),
            "tile_count": len(tile_ids),
            "tile_ids": tile_ids,
            "duplicate_tiles_within_source": (
                duplicate_within_source
            ),
        }

    return stats


def build_duplicate_full_groups(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_tile: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    for record in records:
        if record["kind"] != "full":
            continue

        tile_id = record["tile_id"]

        if tile_id is None:
            continue

        by_tile[tile_id].append(record)

    groups: list[dict[str, Any]] = []

    for tile_id in sorted(
        by_tile,
        key=tile_sort_key,
    ):
        items = by_tile[tile_id]

        sources = sorted(
            {item["source"] for item in items}
        )

        # "Multiple full-label sources" means one tile appears
        # in more than one canonical full-reference source.
        if len(sources) <= 1:
            continue

        pixel_hashes = {
            item["pixel_sha256"]
            for item in items
            if item["pixel_sha256"]
        }

        file_hashes = {
            item["file_sha256"]
            for item in items
            if item["file_sha256"]
        }

        pixel_identical = (
            len(pixel_hashes) == 1
            and len(pixel_hashes) > 0
        )

        file_identical = (
            len(file_hashes) == 1
            and len(file_hashes) > 0
        )

        groups.append(
            {
                "tile_id": tile_id,
                "sources": sources,
                "pixel_identical": pixel_identical,
                "file_identical": file_identical,
                "files": [
                    {
                        "source": item["source"],
                        "path": item["path"],
                        "file_sha256": item["file_sha256"],
                        "pixel_sha256": item["pixel_sha256"],
                        "status": item["status"],
                    }
                    for item in items
                ],
            }
        )

    return groups


def write_records_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    fields = [
        "path",
        "tile_id",
        "source",
        "kind",
        "source_dir",
        "status",
        "error",
        "shape",
        "mode",
        "dtype",
        "bands",
        "compression",
        "compression_tag_259",
        "photometric_tag_262",
        "unique_rgb_count",
        "unknown_unique_count",
        "unknown_pixel_count",
        "near_standard_unique_count",
        "near_standard_pixel_count",
        "target_252_255_0_count",
        "boundary_void_0_0_0_count",
        "file_sha256",
        "pixel_sha256",
        "flags",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for record in records:
            row = {
                key: record.get(key)
                for key in fields
            }

            row["flags"] = ";".join(
                record.get("flags", [])
            )

            writer.writerow(row)


def write_anomaly_report(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    anomaly_records = [
        r
        for r in records
        if r["status"] != "OK"
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        if not anomaly_records:
            f.write(
                "No anomalous label files detected.\n"
            )
            return

        for record in anomaly_records:
            f.write(
                f"{record['path']}\n"
            )
            f.write(
                f"  tile_id: {record['tile_id']}\n"
            )
            f.write(
                f"  source: {record['source']}\n"
            )
            f.write(
                f"  status: {record['status']}\n"
            )
            f.write(
                f"  flags: {record['flags']}\n"
            )
            if record["error"]:
                f.write(
                    f"  error: {record['error']}\n"
                )
            f.write("\n")


def write_text_report(
    path: Path,
    root: Path,
    label_dirs: list[Path],
    source_stats: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    near_threshold: int,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "ISPRS Potsdam Label Audit\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(f"Root: {root}\n")
        f.write(
            f"Near-color diagnostic threshold "
            f"(Linf): {near_threshold}\n"
        )
        f.write(
            "IMPORTANT: near colors are NOT remapped.\n\n"
        )

        f.write(
            "1. Extracted 5_Labels* directories\n"
        )
        f.write(
            "-" * 80 + "\n"
        )

        for d in label_dirs:
            f.write(
                f"{d}\n"
            )
            f.write(
                f"  recursive label TIFF count: "
                f"{count_label_tiffs(d)}\n"
            )

        f.write("\n")

        f.write(
            "2. Source statistics\n"
        )
        f.write(
            "-" * 80 + "\n"
        )

        for source in SOURCE_ORDER:
            stat = source_stats[source]

            f.write(
                f"{source}\n"
            )
            f.write(
                f"  file_count: "
                f"{stat['file_count']}\n"
            )
            f.write(
                f"  tile_count: "
                f"{stat['tile_count']}\n"
            )
            f.write(
                "  tile_ids: "
                + ", ".join(stat["tile_ids"])
                + "\n"
            )

            if stat[
                "duplicate_tiles_within_source"
            ]:
                f.write(
                    "  DUPLICATES WITHIN SOURCE:\n"
                )
                for tile, paths in stat[
                    "duplicate_tiles_within_source"
                ].items():
                    f.write(
                        f"    {tile}\n"
                    )
                    for p in paths:
                        f.write(
                            f"      {p}\n"
                        )

            f.write("\n")

        f.write(
            "3. Every inspected GT file\n"
        )
        f.write(
            "-" * 80 + "\n"
        )

        for record in records:
            f.write(
                f"[{record['status']}] "
                f"{record['tile_id']} "
                f"{record['source']} "
                f"{record['kind']}\n"
            )
            f.write(
                f"  path: {record['path']}\n"
            )
            f.write(
                f"  shape: {record['shape']}\n"
            )
            f.write(
                f"  mode/dtype: "
                f"{record['mode']} / "
                f"{record['dtype']}\n"
            )
            f.write(
                f"  compression: "
                f"{record['compression']}\n"
            )
            f.write(
                f"  TIFF tag 259: "
                f"{record['compression_tag_259']}\n"
            )
            f.write(
                f"  unique RGB colors: "
                f"{record['unique_rgb_count']}\n"
            )
            f.write(
                f"  unknown unique colors: "
                f"{record['unknown_unique_count']}\n"
            )
            f.write(
                f"  unknown pixels: "
                f"{record['unknown_pixel_count']}\n"
            )
            f.write(
                f"  near-standard unique colors: "
                f"{record['near_standard_unique_count']}\n"
            )
            f.write(
                f"  near-standard pixels: "
                f"{record['near_standard_pixel_count']}\n"
            )
            f.write(
                f"  (252,255,0) pixels: "
                f"{record['target_252_255_0_count']}\n"
            )
            f.write(
                f"  black/void pixels: "
                f"{record['boundary_void_0_0_0_count']}\n"
            )
            f.write(
                f"  flags: {record['flags']}\n"
            )

            if record["error"]:
                f.write(
                    f"  ERROR: {record['error']}\n"
                )

            f.write("\n")

        f.write(
            "4. Duplicate full-label sources\n"
        )
        f.write(
            "-" * 80 + "\n"
        )

        if not duplicate_groups:
            f.write(
                "No tile occurs in multiple "
                "full-label sources.\n"
            )
        else:
            for group in duplicate_groups:
                f.write(
                    f"tile {group['tile_id']}\n"
                )
                f.write(
                    f"  sources: "
                    f"{group['sources']}\n"
                )
                f.write(
                    f"  decoded RGB pixels identical: "
                    f"{group['pixel_identical']}\n"
                )
                f.write(
                    f"  TIFF files byte-identical: "
                    f"{group['file_identical']}\n"
                )

                for item in group["files"]:
                    f.write(
                        f"    {item['source']}\n"
                    )
                    f.write(
                        f"      {item['path']}\n"
                    )

                f.write("\n")


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser()

    if not root.exists():
        print(
            f"ERROR: root does not exist: {root}",
            file=sys.stderr,
        )
        return 2

    if not root.is_dir():
        print(
            f"ERROR: root is not a directory: {root}",
            file=sys.stderr,
        )
        return 2

    if args.near_threshold < 0:
        print(
            "ERROR: --near-threshold must be >= 0",
            file=sys.stderr,
        )
        return 2

    if args.chunk_rows <= 0:
        print(
            "ERROR: --chunk-rows must be > 0",
            file=sys.stderr,
        )
        return 2

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dirs = discover_label_dirs(root)
    label_files = discover_label_files(root)

    print(
        "\n=== Extracted 5_Labels* directories ==="
    )

    if not label_dirs:
        print("NONE")
    else:
        for d in label_dirs:
            print(
                f"{d} "
                f"[recursive label TIFFs: "
                f"{count_label_tiffs(d)}]"
            )

    print(
        "\n=== Discovered GT TIFF files ==="
    )
    print(
        f"Total discovered label TIFFs: "
        f"{len(label_files)}"
    )

    if not label_files:
        print(
            "\nERROR: no Potsdam label TIFFs found.",
            file=sys.stderr,
        )
        return 3

    source_stats = build_source_stats(
        label_files
    )

    print(
        "\n=== Source summary ==="
    )

    for source in SOURCE_ORDER:
        stat = source_stats[source]

        print(
            f"{source}: "
            f"files={stat['file_count']}, "
            f"tiles={stat['tile_count']}"
        )

        if stat["tile_ids"]:
            print(
                "  tile IDs: "
                + ", ".join(stat["tile_ids"])
            )

        if stat[
            "duplicate_tiles_within_source"
        ]:
            print(
                "  WARNING: duplicate tile IDs "
                "inside this source:"
            )

            for tile, paths in stat[
                "duplicate_tiles_within_source"
            ].items():
                print(
                    f"    {tile}"
                )
                for p in paths:
                    print(
                        f"      {p}"
                    )

    unknown_csv = (
        out_dir / "unknown_colors.csv"
    )
    near_csv = (
        out_dir / "near_standard_colors.csv"
    )

    color_fields = [
        "path",
        "tile_id",
        "source",
        "kind",
        "r",
        "g",
        "b",
        "pixel_count",
        "nearest_standard_class",
        "nearest_standard_rgb",
        "distance_Linf",
        "distance_L1",
        "within_near_threshold",
    ]

    records: list[dict[str, Any]] = []

    print(
        "\n=== Inspecting every GT file ==="
    )

    with unknown_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fu, near_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fn:

        unknown_writer = csv.DictWriter(
            fu,
            fieldnames=color_fields,
        )
        near_writer = csv.DictWriter(
            fn,
            fieldnames=color_fields,
        )

        unknown_writer.writeheader()
        near_writer.writeheader()

        total = len(label_files)

        for index, label_path in enumerate(
            label_files,
            start=1,
        ):
            record = inspect_one_label(
                path=label_path,
                near_threshold=args.near_threshold,
                chunk_rows=args.chunk_rows,
                unknown_writer=unknown_writer,
                near_writer=near_writer,
            )

            records.append(record)

            print(
                f"[{index:03d}/{total:03d}] "
                f"{record['status']:<7} "
                f"tile={record['tile_id']:<5} "
                f"source={record['source']}"
            )
            print(
                f"    path={record['path']}"
            )
            print(
                f"    shape={record['shape']} "
                f"mode={record['mode']} "
                f"dtype={record['dtype']} "
                f"compression={record['compression']}"
            )
            print(
                f"    unique={record['unique_rgb_count']} "
                f"unknown_unique="
                f"{record['unknown_unique_count']} "
                f"unknown_pixels="
                f"{record['unknown_pixel_count']} "
                f"target_252_255_0="
                f"{record['target_252_255_0_count']}"
            )

            if record["flags"]:
                print(
                    f"    FLAGS: {record['flags']}"
                )

            if record["error"]:
                print(
                    f"    ERROR: {record['error']}"
                )

    duplicate_groups = (
        build_duplicate_full_groups(records)
    )

    print(
        "\n=== Duplicate full-label sources ==="
    )

    if not duplicate_groups:
        print(
            "No tile occurs in multiple "
            "full-label sources."
        )
    else:
        for group in duplicate_groups:
            print(
                f"tile {group['tile_id']}: "
                f"sources={group['sources']}"
            )
            print(
                "  decoded RGB pixels identical: "
                f"{group['pixel_identical']}"
            )
            print(
                "  TIFF files byte-identical: "
                f"{group['file_identical']}"
            )

            for item in group["files"]:
                print(
                    f"    {item['source']}: "
                    f"{item['path']}"
                )

    target_records = [
        r
        for r in records
        if (
            r["target_252_255_0_count"]
            is not None
            and r["target_252_255_0_count"] > 0
        )
    ]

    over_4096_records = [
        r
        for r in records
        if (
            r["unique_rgb_count"] is not None
            and r["unique_rgb_count"] > 4096
        )
    ]

    anomaly_records = [
        r
        for r in records
        if r["status"] != "OK"
    ]

    print(
        "\n=== Key anomaly summary ==="
    )
    print(
        f"Anomalous files       : "
        f"{len(anomaly_records)}"
    )
    print(
        f">4096-color files     : "
        f"{len(over_4096_records)}"
    )
    print(
        f"(252,255,0) files     : "
        f"{len(target_records)}"
    )

    if over_4096_records:
        print(
            "\nFiles with >4096 exact RGB colors:"
        )

        for r in over_4096_records:
            print(
                f"  {r['path']}"
            )
            print(
                f"    unique="
                f"{r['unique_rgb_count']}"
            )
            print(
                f"    compression="
                f"{r['compression']}"
            )

    if target_records:
        print(
            "\nFiles containing exact "
            "(252,255,0):"
        )

        for r in target_records:
            print(
                f"  {r['path']}"
            )
            print(
                f"    pixel_count="
                f"{r['target_252_255_0_count']}"
            )
            print(
                f"    source={r['source']}"
            )
            print(
                f"    unique_colors="
                f"{r['unique_rgb_count']}"
            )
            print(
                f"    compression="
                f"{r['compression']}"
            )

    # Main per-file CSV.
    records_csv = (
        out_dir / "label_files.csv"
    )
    write_records_csv(
        records_csv,
        records,
    )

    anomaly_txt = (
        out_dir / "anomaly_files.txt"
    )
    write_anomaly_report(
        anomaly_txt,
        records,
    )

    text_report = (
        out_dir / "label_audit.txt"
    )
    write_text_report(
        path=text_report,
        root=root,
        label_dirs=label_dirs,
        source_stats=source_stats,
        records=records,
        duplicate_groups=duplicate_groups,
        near_threshold=args.near_threshold,
    )

    json_report = (
        out_dir / "label_audit.json"
    )

    json_payload = {
        "root": str(root),
        "near_threshold_Linf": (
            args.near_threshold
        ),
        "standard_colors": {
            name: list(rgb)
            for name, rgb
            in STANDARD_COLORS.items()
        },
        "boundary_void_color": list(
            BOUNDARY_VOID_COLOR
        ),
        "target_color": list(
            TARGET_COLOR
        ),
        "discovered_label_directories": [
            {
                "path": str(d),
                "recursive_label_tif_count": (
                    count_label_tiffs(d)
                ),
            }
            for d in label_dirs
        ],
        "source_stats": source_stats,
        "records": records,
        "duplicate_full_label_groups": (
            duplicate_groups
        ),
        "summary": {
            "total_label_files": (
                len(records)
            ),
            "anomalous_files": (
                len(anomaly_records)
            ),
            "over_4096_color_files": (
                len(over_4096_records)
            ),
            "target_252_255_0_files": (
                len(target_records)
            ),
        },
    }

    with json_report.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== Reports saved ==="
    )
    print(records_csv)
    print(unknown_csv)
    print(near_csv)
    print(anomaly_txt)
    print(text_report)
    print(json_report)

    print(
        "\nLabel inspection completed."
    )
    print(
        "No label files were modified."
    )
    print(
        "No approximate color mapping was performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
