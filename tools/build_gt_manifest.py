#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------
# Locked Potsdam tile inventory for this experiment.
# ---------------------------------------------------------------------

EXPECTED_ALL_TILES = {
    "2_10", "2_11", "2_12", "2_13", "2_14",
    "3_10", "3_11", "3_12", "3_13", "3_14",
    "4_10", "4_11", "4_12", "4_13", "4_14", "4_15",
    "5_10", "5_11", "5_12", "5_13", "5_14", "5_15",
    "6_7", "6_8", "6_9", "6_10", "6_11", "6_12",
    "6_13", "6_14", "6_15",
    "7_7", "7_8", "7_9", "7_10", "7_11", "7_12", "7_13",
}

EXPECTED_PARTICIPANT_TILES = {
    "2_10", "2_11", "2_12",
    "3_10", "3_11", "3_12",
    "4_10", "4_11", "4_12",
    "5_10", "5_11", "5_12",
    "6_7", "6_8", "6_9", "6_10", "6_11", "6_12",
    "7_7", "7_8", "7_9", "7_10", "7_11", "7_12",
}

EXPECTED_ALL_ONLY_TILES = (
    EXPECTED_ALL_TILES
    - EXPECTED_PARTICIPANT_TILES
)

# From the completed forensic audit:
#
# 4_12:
#   5_Labels_all version is an RGB/GT overlay.
#
# 6_7:
#   5_Labels_all contains 246,304 pixels of (252,255,0)
#   where the participant reference has (255,255,0).
#
# These are NOT repaired here.
# Their all-source copies are simply rejected by the global source policy.
EXPECTED_OVERLAP_MISMATCHES = {
    "4_12",
    "6_7",
}


# ---------------------------------------------------------------------
# Official categorical colors used by this experiment.
# ---------------------------------------------------------------------

STANDARD_COLORS: dict[str, tuple[int, int, int]] = {
    "impervious_surface": (255, 255, 255),
    "building":           (0, 0, 255),
    "low_vegetation":     (0, 255, 255),
    "tree":               (0, 255, 0),
    "car":                (255, 255, 0),
    "clutter_background": (255, 0, 0),
}

STANDARD_COLOR_SET = set(
    STANDARD_COLORS.values()
)

EXPECTED_SIZE = (
    6000,
    6000,
)

TILE_RE = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_label\.tif$",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic canonical GT manifest "
            "for ISPRS Potsdam."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Potsdam raw-data root, "
            "e.g. data/raw/potsdam"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/potsdam/"
            "labels_manifest.json"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/dataset_check/"
            "canonical_gt/"
            "canonical_gt_report.txt"
        ),
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
    )

    return parser.parse_args()


def tile_sort_key(
    tile_id: str,
) -> tuple[int, int]:
    a, b = tile_id.split("_", 1)

    return (
        int(a),
        int(b),
    )


def extract_tile_id(
    path: Path,
) -> str | None:
    match = TILE_RE.match(
        path.name
    )

    if match is None:
        return None

    return (
        f"{int(match.group(1))}_"
        f"{int(match.group(2))}"
    )


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def discover_source(
    source_root: Path,
) -> dict[str, Path]:
    if not source_root.is_dir():
        raise RuntimeError(
            f"Source directory does not exist: "
            f"{source_root}"
        )

    result: dict[str, Path] = {}

    for path in source_root.rglob(
        "top_potsdam_*_label.tif"
    ):
        if not path.is_file():
            continue

        tile_id = extract_tile_id(
            path
        )

        if tile_id is None:
            continue

        if tile_id in result:
            raise RuntimeError(
                "Duplicate tile inside one source:\n"
                f"tile={tile_id}\n"
                f"first={result[tile_id]}\n"
                f"second={path}"
            )

        result[tile_id] = path.resolve()

    return result


def relative_to_root(
    path: Path,
    root: Path,
) -> str:
    try:
        return str(
            path.relative_to(root)
        )
    except ValueError:
        raise RuntimeError(
            f"GT is outside Potsdam root:\n"
            f"root={root}\n"
            f"path={path}"
        )

def inspect_selected_gt(
    path: Path,
    chunk_rows: int,
) -> dict[str, Any]:
    """
    Exact categorical RGB validation.

    Fast implementation:
      - packs RGB into one uint32 value
      - compares exactly against the six allowed Potsdam colors
      - never performs approximate color matching
      - calls np.unique only if genuinely unknown pixels exist
    """

    pixel_hasher = hashlib.sha256()

    color_ids = {
        name: (
            (int(rgb[0]) << 16)
            | (int(rgb[1]) << 8)
            | int(rgb[2])
        )
        for name, rgb in STANDARD_COLORS.items()
    }

    standard_counts = {
        name: 0
        for name in STANDARD_COLORS
    }

    unknown_counts: Counter[int] = Counter()

    with Image.open(path) as img:
        img.seek(0)

        width, height = img.size
        mode = img.mode
        bands = tuple(img.getbands())

        sample = np.asarray(
            img.crop((0, 0, 1, 1))
        )

        dtype = str(sample.dtype)

        if (width, height) != EXPECTED_SIZE:
            raise RuntimeError(
                f"Unexpected GT size:\n"
                f"path={path}\n"
                f"size={img.size}"
            )

        if mode != "RGB":
            raise RuntimeError(
                f"Unexpected GT mode:\n"
                f"path={path}\n"
                f"mode={mode}"
            )

        if dtype != "uint8":
            raise RuntimeError(
                f"Unexpected GT dtype:\n"
                f"path={path}\n"
                f"dtype={dtype}"
            )

        if bands != ("R", "G", "B"):
            raise RuntimeError(
                f"Unexpected GT bands:\n"
                f"path={path}\n"
                f"bands={bands}"
            )

        for y0 in range(
            0,
            height,
            chunk_rows,
        ):
            y1 = min(
                y0 + chunk_rows,
                height,
            )

            rgb = np.asarray(
                img.crop(
                    (
                        0,
                        y0,
                        width,
                        y1,
                    )
                ),
                dtype=np.uint8,
            )

            if (
                rgb.ndim != 3
                or rgb.shape[2] != 3
            ):
                raise RuntimeError(
                    f"Unexpected RGB array shape:\n"
                    f"path={path}\n"
                    f"shape={rgb.shape}"
                )

            pixel_hasher.update(
                rgb.tobytes(order="C")
            )

            # Pack RGB exactly into:
            #
            # 0xRRGGBB
            #
            # uint32 is used to avoid overflow.
            r = rgb[..., 0].astype(
                np.uint32,
                copy=False,
            )

            g = rgb[..., 1].astype(
                np.uint32,
                copy=False,
            )

            b = rgb[..., 2].astype(
                np.uint32,
                copy=False,
            )

            packed = (
                (r << 16)
                | (g << 8)
                | b
            )

            valid = np.zeros(
                packed.shape,
                dtype=bool,
            )

            # Exact comparison against each of the
            # six categorical Potsdam colors.
            for class_name, color_id in (
                color_ids.items()
            ):
                mask = (
                    packed == color_id
                )

                count = int(
                    np.count_nonzero(
                        mask
                    )
                )

                standard_counts[
                    class_name
                ] += count

                valid |= mask

            # This branch should never execute for
            # a valid canonical GT.
            #
            # np.unique is intentionally applied ONLY
            # to genuinely unknown pixels.
            if not np.all(valid):
                unknown = packed[
                    ~valid
                ]

                values, counts = np.unique(
                    unknown,
                    return_counts=True,
                )

                for value, count in zip(
                    values.tolist(),
                    counts.tolist(),
                ):
                    unknown_counts[
                        int(value)
                    ] += int(count)

    def unpack_rgb(
        value: int,
    ) -> tuple[int, int, int]:
        return (
            (value >> 16) & 255,
            (value >> 8) & 255,
            value & 255,
        )

    if unknown_counts:
        unknown_items = sorted(
            unknown_counts.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )

        preview = [
            {
                "rgb": unpack_rgb(
                    value
                ),
                "count": count,
            }
            for value, count
            in unknown_items[:20]
        ]

        unknown_pixel_count = sum(
            unknown_counts.values()
        )

        raise RuntimeError(
            "Canonical GT contains unknown colors:\n"
            f"path={path}\n"
            f"unknown_unique_count="
            f"{len(unknown_counts)}\n"
            f"unknown_pixel_count="
            f"{unknown_pixel_count}\n"
            f"top_unknown={preview}"
        )

    present_standard_colors = [
        list(
            STANDARD_COLORS[
                class_name
            ]
        )
        for class_name
        in STANDARD_COLORS
        if standard_counts[
            class_name
        ] > 0
    ]

    unique_rgb_count = len(
        present_standard_colors
    )

    total_pixels = sum(
        standard_counts.values()
    )

    expected_pixels = (
        EXPECTED_SIZE[0]
        * EXPECTED_SIZE[1]
    )

    if total_pixels != expected_pixels:
        raise RuntimeError(
            "Pixel accounting mismatch:\n"
            f"path={path}\n"
            f"counted={total_pixels}\n"
            f"expected={expected_pixels}"
        )

    return {
        "shape": [
            EXPECTED_SIZE[1],
            EXPECTED_SIZE[0],
            3,
        ],
        "mode": "RGB",
        "dtype": "uint8",
        "bands": [
            "R",
            "G",
            "B",
        ],
        "unique_rgb_count": (
            unique_rgb_count
        ),
        "unique_rgb_colors": (
            present_standard_colors
        ),
        "unknown_unique_count": 0,
        "unknown_pixel_count": 0,
        "standard_color_counts": {
            name: int(
                standard_counts[
                    name
                ]
            )
            for name
            in STANDARD_COLORS
        },
        "file_sha256": (
            sha256_file(path)
        ),
        "pixel_sha256": (
            pixel_hasher.hexdigest()
        ),
    }


def validate_inventory(
    name: str,
    actual: set[str],
    expected: set[str],
) -> None:
    if actual == expected:
        return

    missing = sorted(
        expected - actual,
        key=tile_sort_key,
    )

    extra = sorted(
        actual - expected,
        key=tile_sort_key,
    )

    raise RuntimeError(
        f"{name} tile inventory mismatch.\n"
        f"Expected count: {len(expected)}\n"
        f"Actual count  : {len(actual)}\n"
        f"Missing       : {missing}\n"
        f"Extra         : {extra}"
    )


def main() -> int:
    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )

    report_path = (
        args.report
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        print(
            f"ERROR: root does not exist: "
            f"{root}",
            file=sys.stderr,
        )
        return 2

    if args.chunk_rows <= 0:
        print(
            "ERROR: --chunk-rows must be > 0",
            file=sys.stderr,
        )
        return 2

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

    all_files = discover_source(
        all_root
    )

    participant_files = discover_source(
        participant_root
    )

    print(
        "=== GT source inventory ==="
    )

    print(
        f"5_Labels_all             : "
        f"{len(all_files)}"
    )

    print(
        f"5_Labels_for_participants: "
        f"{len(participant_files)}"
    )

    validate_inventory(
        "5_Labels_all",
        set(all_files),
        EXPECTED_ALL_TILES,
    )

    validate_inventory(
        "5_Labels_for_participants",
        set(participant_files),
        EXPECTED_PARTICIPANT_TILES,
    )

    actual_all_only = (
        set(all_files)
        - set(participant_files)
    )

    if (
        actual_all_only
        != EXPECTED_ALL_ONLY_TILES
    ):
        raise RuntimeError(
            "Unexpected all-only tile set:\n"
            f"actual={sorted(actual_all_only, key=tile_sort_key)}\n"
            f"expected={sorted(EXPECTED_ALL_ONLY_TILES, key=tile_sort_key)}"
        )

    print()
    print(
        "Inventory check: PASS"
    )

    print()
    print(
        "=== Checking duplicate source files ==="
    )

    identical_overlap: list[str] = []
    mismatch_overlap: list[str] = []

    # Cache participant SHA values because these
    # will also appear in the final manifest.
    participant_file_sha: dict[
        str,
        str,
    ] = {}

    for tile_id in sorted(
        EXPECTED_PARTICIPANT_TILES,
        key=tile_sort_key,
    ):
        all_sha = sha256_file(
            all_files[tile_id]
        )

        participant_sha = sha256_file(
            participant_files[tile_id]
        )

        participant_file_sha[
            tile_id
        ] = participant_sha

        identical = (
            all_sha
            == participant_sha
        )

        if identical:
            identical_overlap.append(
                tile_id
            )
        else:
            mismatch_overlap.append(
                tile_id
            )

        print(
            f"{tile_id:<5} "
            f"byte_identical="
            f"{identical}"
        )

    mismatch_set = set(
        mismatch_overlap
    )

    if (
        mismatch_set
        != EXPECTED_OVERLAP_MISMATCHES
    ):
        raise RuntimeError(
            "Overlap mismatch set changed.\n"
            f"Expected mismatches: "
            f"{sorted(EXPECTED_OVERLAP_MISMATCHES, key=tile_sort_key)}\n"
            f"Actual mismatches  : "
            f"{sorted(mismatch_set, key=tile_sort_key)}\n"
            "Stop and review dataset version before continuing."
        )

    print()
    print(
        "Duplicate-source check: PASS"
    )

    print(
        f"Byte-identical overlaps : "
        f"{len(identical_overlap)}"
    )

    print(
        "Non-identical overlaps : "
        + ", ".join(
            sorted(
                mismatch_overlap,
                key=tile_sort_key,
            )
        )
    )

    print()
    print(
        "=== Building canonical selection ==="
    )

    tiles = []

    source_counts = Counter()

    for index, tile_id in enumerate(
        sorted(
            EXPECTED_ALL_TILES,
            key=tile_sort_key,
        ),
        start=1,
    ):
        if (
            tile_id
            in participant_files
        ):
            source = (
                "5_Labels_for_participants"
            )

            selected_path = (
                participant_files[
                    tile_id
                ]
            )

            selection_reason = (
                "participant_reference_available"
            )

        else:
            source = (
                "5_Labels_all"
            )

            selected_path = (
                all_files[
                    tile_id
                ]
            )

            selection_reason = (
                "all_only_reference"
            )

        print(
            f"[{index:02d}/38] "
            f"tile={tile_id:<5} "
            f"source={source}"
        )

        audit = inspect_selected_gt(
            selected_path,
            args.chunk_rows,
        )

        # Verify cached participant hash.
        if (
            source
            == "5_Labels_for_participants"
        ):
            expected_sha = (
                participant_file_sha[
                    tile_id
                ]
            )

            if (
                audit["file_sha256"]
                != expected_sha
            ):
                raise RuntimeError(
                    "File changed during manifest build:\n"
                    f"{selected_path}"
                )

        source_counts[
            source
        ] += 1

        tiles.append(
            {
                "tile_id": tile_id,
                "source": source,
                "selection_reason": (
                    selection_reason
                ),
                "gt_relpath": (
                    relative_to_root(
                        selected_path,
                        root,
                    )
                ),
                **audit,
            }
        )

    if len(tiles) != 38:
        raise RuntimeError(
            f"Expected 38 canonical GTs, "
            f"got {len(tiles)}"
        )

    if (
        source_counts[
            "5_Labels_for_participants"
        ]
        != 24
    ):
        raise RuntimeError(
            "Expected 24 participant GTs, "
            f"got "
            f"{source_counts['5_Labels_for_participants']}"
        )

    if (
        source_counts[
            "5_Labels_all"
        ]
        != 14
    ):
        raise RuntimeError(
            "Expected 14 all-only GTs, "
            f"got "
            f"{source_counts['5_Labels_all']}"
        )

    selected_ids = {
        item["tile_id"]
        for item in tiles
    }

    if (
        selected_ids
        != EXPECTED_ALL_TILES
    ):
        raise RuntimeError(
            "Canonical tile set is not "
            "exactly the expected 38 tiles."
        )

    manifest = {
        "schema_version": 1,
        "dataset": (
            "ISPRS Potsdam"
        ),
        "purpose": (
            "Canonical semantic-segmentation "
            "ground-truth selection"
        ),
        "selection_policy": (
            "Use 5_Labels_for_participants "
            "whenever that tile exists; "
            "otherwise use 5_Labels_all. "
            "Dataset code must consume this "
            "manifest and must not scan multiple "
            "GT sources dynamically."
        ),
        "path_semantics": (
            "gt_relpath is relative to "
            "the Potsdam raw-data root "
            "provided at runtime."
        ),
        "expected_tile_count": 38,
        "selected_tile_count": (
            len(tiles)
        ),
        "source_counts": {
            "5_Labels_for_participants": (
                source_counts[
                    "5_Labels_for_participants"
                ]
            ),
            "5_Labels_all": (
                source_counts[
                    "5_Labels_all"
                ]
            ),
        },
        "duplicate_source_audit": {
            "overlap_tile_count": (
                len(
                    EXPECTED_PARTICIPANT_TILES
                )
            ),
            "byte_identical_overlap_count": (
                len(
                    identical_overlap
                )
            ),
            "byte_identical_tiles": (
                sorted(
                    identical_overlap,
                    key=tile_sort_key,
                )
            ),
            "non_identical_overlap_count": (
                len(
                    mismatch_overlap
                )
            ),
            "non_identical_tiles": (
                sorted(
                    mismatch_overlap,
                    key=tile_sort_key,
                )
            ),
            "known_rejected_all_versions": {
                "4_12": (
                    "5_Labels_all version was "
                    "verified to be an RGB/GT "
                    "alpha overlay rather than "
                    "a categorical mask."
                ),
                "6_7": (
                    "5_Labels_all version contains "
                    "246304 pixels of (252,255,0); "
                    "coordinate-wise comparison "
                    "with participant GT maps these "
                    "to exact car color "
                    "(255,255,0)."
                ),
            },
        },
        "all_only_tiles": (
            sorted(
                EXPECTED_ALL_ONLY_TILES,
                key=tile_sort_key,
            )
        ),
        "standard_colors": {
            name: list(rgb)
            for name, rgb
            in STANDARD_COLORS.items()
        },
        "tiles": tiles,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "ISPRS Potsdam Canonical GT Report\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Selection policy:\n"
        )
        f.write(
            "  participants if available, "
            "otherwise all\n\n"
        )

        f.write(
            f"Canonical tile count : "
            f"{len(tiles)}\n"
        )

        f.write(
            "Participants selected: "
            f"{source_counts['5_Labels_for_participants']}\n"
        )

        f.write(
            "All-only selected     : "
            f"{source_counts['5_Labels_all']}\n"
        )

        f.write(
            "Identical overlaps    : "
            f"{len(identical_overlap)}\n"
        )

        f.write(
            "Mismatching overlaps  : "
            + ", ".join(
                sorted(
                    mismatch_overlap,
                    key=tile_sort_key,
                )
            )
            + "\n\n"
        )

        f.write(
            "All-only tiles:\n"
        )

        for tile_id in sorted(
            EXPECTED_ALL_ONLY_TILES,
            key=tile_sort_key,
        ):
            f.write(
                f"  {tile_id}\n"
            )

        f.write(
            "\nCanonical GT files:\n"
        )

        for item in tiles:
            f.write(
                f"  {item['tile_id']} "
                f"[{item['source']}]\n"
            )

            f.write(
                f"    {item['gt_relpath']}\n"
            )

            f.write(
                f"    unique_rgb="
                f"{item['unique_rgb_count']} "
                f"unknown_pixels="
                f"{item['unknown_pixel_count']}\n"
            )

            f.write(
                f"    file_sha256="
                f"{item['file_sha256']}\n"
            )

            f.write(
                f"    pixel_sha256="
                f"{item['pixel_sha256']}\n"
            )

    print()
    print(
        "=== Canonical GT summary ==="
    )

    print(
        f"Canonical GT tiles     : "
        f"{len(tiles)}"
    )

    print(
        f"Participants selected  : "
        f"{source_counts['5_Labels_for_participants']}"
    )

    print(
        f"All-only selected      : "
        f"{source_counts['5_Labels_all']}"
    )

    print(
        f"Identical overlaps     : "
        f"{len(identical_overlap)}"
    )

    print(
        "Rejected all mismatches: "
        + ", ".join(
            sorted(
                mismatch_overlap,
                key=tile_sort_key,
            )
        )
    )

    print(
        "Unknown colors in canonical GT: 0"
    )

    print()
    print(
        "Manifest:"
    )

    print(
        manifest_path
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Canonical GT manifest build: PASS"
    )

    print(
        "No GT file was copied or modified."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
