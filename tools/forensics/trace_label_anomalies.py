#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

TARGET_TILES = ("4_12", "6_7")

SOURCES = (
    "5_Labels_all",
    "5_Labels_for_participants",
)

STANDARD_COLORS = {
    "impervious_surface": (255, 255, 255),
    "building":           (0, 0, 255),
    "low_vegetation":     (0, 255, 255),
    "tree":               (0, 255, 0),
    "car":                (255, 255, 0),
    "clutter_background": (255, 0, 0),
}

CLASS_NAMES = list(STANDARD_COLORS.keys())

STANDARD_RGB = np.asarray(
    [STANDARD_COLORS[name] for name in CLASS_NAMES],
    dtype=np.int16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace anomalous Potsdam GT files back to ZIP members "
            "and compare duplicate GTs pixel by pixel."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Potsdam root, e.g. data/raw/potsdam",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/dataset_check/label_trace"
        ),
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
    )

    return parser.parse_args()


def rgb_to_id(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb

    return (
        (int(r) << 16)
        | (int(g) << 8)
        | int(b)
    )


def id_to_rgb(value: int) -> tuple[int, int, int]:
    value = int(value)

    return (
        (value >> 16) & 255,
        (value >> 8) & 255,
        value & 255,
    )


STANDARD_IDS = np.asarray(
    [
        rgb_to_id(STANDARD_COLORS[name])
        for name in CLASS_NAMES
    ],
    dtype=np.uint32,
)


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb32 = rgb.astype(
        np.uint32,
        copy=False,
    )

    return (
        (rgb32[..., 0] << 16)
        | (rgb32[..., 1] << 8)
        | rgb32[..., 2]
    )


def sha256_file(path: Path) -> str:
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


def sha256_zip_member(
    zf: zipfile.ZipFile,
    member: str,
) -> str:
    h = hashlib.sha256()

    with zf.open(member, "r") as f:
        while True:
            block = f.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def find_zip(
    root: Path,
    source: str,
) -> Path:
    expected = root / f"{source}.zip"

    if expected.is_file():
        return expected

    matches = [
        p
        for p in root.rglob(f"{source}.zip")
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {source}.zip, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def find_zip_member(
    zf: zipfile.ZipFile,
    tile: str,
) -> str:
    basename = (
        f"top_potsdam_{tile}_label.tif"
    ).lower()

    matches = [
        name
        for name in zf.namelist()
        if Path(name).name.lower()
        == basename
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one ZIP member "
            f"for tile {tile}, found "
            f"{len(matches)}:\n{matches}"
        )

    return matches[0]


def find_extracted(
    root: Path,
    source: str,
    tile: str,
) -> Path:
    base = (
        root
        / "_expanded"
        / source
    )

    basename = (
        f"top_potsdam_{tile}_label.tif"
    )

    if not base.exists():
        raise RuntimeError(
            f"Extracted source does not exist: "
            f"{base}"
        )

    matches = [
        p
        for p in base.rglob(basename)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one extracted "
            f"{source}/{basename}, found "
            f"{len(matches)}:\n{matches}"
        )

    return matches[0]


def nearest_standard_class(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Diagnostic nearest-standard classification.

    IMPORTANT:
    This function does NOT modify or repair any label.
    """

    x = rgb.astype(
        np.int16,
        copy=False,
    )

    shape = x.shape[:2]

    best_distance = np.full(
        shape,
        32767,
        dtype=np.int16,
    )

    best_class = np.full(
        shape,
        -1,
        dtype=np.int8,
    )

    for class_index, color in enumerate(
        STANDARD_RGB
    ):
        distance = np.abs(
            x - color
        ).sum(
            axis=2,
            dtype=np.int16,
        )

        update = (
            distance < best_distance
        )

        best_distance[update] = (
            distance[update]
        )
        best_class[update] = (
            class_index
        )

    return best_class


def exact_standard_class(
    packed: np.ndarray,
) -> np.ndarray:
    cls = np.full(
        packed.shape,
        -1,
        dtype=np.int8,
    )

    for class_index, color_id in enumerate(
        STANDARD_IDS
    ):
        cls[packed == color_id] = (
            class_index
        )

    return cls


def inspect_image(
    path: Path,
    chunk_rows: int,
) -> dict:
    histogram = np.zeros(
        1 << 24,
        dtype=np.uint32,
    )

    pixel_hasher = hashlib.sha256()

    with Image.open(path) as img:
        img.seek(0)

        width, height = img.size

        mode = img.mode

        compression = str(
            img.info.get("compression")
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
                    (0, y0, width, y1)
                ).convert("RGB"),
                dtype=np.uint8,
            )

            pixel_hasher.update(
                rgb.tobytes(order="C")
            )

            packed = pack_rgb(rgb)

            values, counts = np.unique(
                packed.reshape(-1),
                return_counts=True,
            )

            histogram[values] += (
                counts.astype(
                    np.uint32,
                    copy=False,
                )
            )

    used_ids = np.flatnonzero(
        histogram
    )

    standard_counts = {}

    for name, rgb in STANDARD_COLORS.items():
        standard_counts[name] = int(
            histogram[rgb_to_id(rgb)]
        )

    unknown_mask = ~np.isin(
        used_ids,
        STANDARD_IDS,
    )

    unknown_ids = (
        used_ids[unknown_mask]
    )

    unknown_pixels = int(
        histogram[
            unknown_ids
        ].astype(
            np.uint64
        ).sum()
    )

    target_count = int(
        histogram[
            rgb_to_id(
                (252, 255, 0)
            )
        ]
    )

    result = {
        "path": str(path),
        "size": [width, height],
        "mode": mode,
        "compression": compression,
        "file_sha256": sha256_file(path),
        "pixel_sha256": (
            pixel_hasher.hexdigest()
        ),
        "unique_rgb_count": int(
            used_ids.size
        ),
        "unknown_unique_count": int(
            unknown_ids.size
        ),
        "unknown_pixel_count": (
            unknown_pixels
        ),
        "target_252_255_0_count": (
            target_count
        ),
        "standard_color_counts": (
            standard_counts
        ),
    }

    del histogram

    return result


def inspect_zip_member(
    zf: zipfile.ZipFile,
    member: str,
    chunk_rows: int,
) -> dict:
    info = zf.getinfo(member)

    archive_sha256 = (
        sha256_zip_member(
            zf,
            member,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="potsdam_label_trace_"
    ) as tmp:
        temp_path = (
            Path(tmp)
            / Path(member).name
        )

        with zf.open(
            member,
            "r",
        ) as src, temp_path.open(
            "wb"
        ) as dst:
            shutil.copyfileobj(
                src,
                dst,
                length=8 * 1024 * 1024,
            )

        image_info = inspect_image(
            temp_path,
            chunk_rows,
        )

    return {
        "member": member,
        "zip_crc32": (
            f"{info.CRC:08x}"
        ),
        "zip_file_size": (
            info.file_size
        ),
        "zip_compress_size": (
            info.compress_size
        ),
        "zip_compress_type": (
            info.compress_type
        ),
        "member_sha256": (
            archive_sha256
        ),
        "image": image_info,
    }


def compare_duplicate_pair(
    all_path: Path,
    participant_path: Path,
    chunk_rows: int,
) -> dict:
    changed_pixels = 0
    same_pixels = 0

    participant_unknown_pixels = 0

    nearest_class_mismatch = 0

    participant_class_counts = np.zeros(
        len(CLASS_NAMES),
        dtype=np.uint64,
    )

    all_nearest_class_counts = np.zeros(
        len(CLASS_NAMES),
        dtype=np.uint64,
    )

    changed_transitions = Counter()

    with Image.open(
        all_path
    ) as img_all, Image.open(
        participant_path
    ) as img_ref:

        if img_all.size != img_ref.size:
            raise RuntimeError(
                "Image size mismatch:\n"
                f"all={img_all.size}\n"
                f"participant={img_ref.size}"
            )

        width, height = (
            img_all.size
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

            rgb_all = np.asarray(
                img_all.crop(
                    (0, y0, width, y1)
                ).convert("RGB"),
                dtype=np.uint8,
            )

            rgb_ref = np.asarray(
                img_ref.crop(
                    (0, y0, width, y1)
                ).convert("RGB"),
                dtype=np.uint8,
            )

            id_all = pack_rgb(
                rgb_all
            )

            id_ref = pack_rgb(
                rgb_ref
            )

            same = (
                id_all == id_ref
            )

            n_same = int(
                same.sum()
            )

            same_pixels += n_same
            changed_pixels += int(
                same.size - n_same
            )

            ref_class = (
                exact_standard_class(
                    id_ref
                )
            )

            participant_unknown_pixels += (
                int(
                    (
                        ref_class < 0
                    ).sum()
                )
            )

            valid_ref = (
                ref_class >= 0
            )

            if valid_ref.any():
                counts = np.bincount(
                    ref_class[
                        valid_ref
                    ].astype(
                        np.int64
                    ),
                    minlength=len(
                        CLASS_NAMES
                    ),
                )

                participant_class_counts += (
                    counts.astype(
                        np.uint64
                    )
                )

            all_nearest = (
                nearest_standard_class(
                    rgb_all
                )
            )

            counts = np.bincount(
                all_nearest.reshape(
                    -1
                ).astype(
                    np.int64
                ),
                minlength=len(
                    CLASS_NAMES
                ),
            )

            all_nearest_class_counts += (
                counts.astype(
                    np.uint64
                )
            )

            nearest_class_mismatch += int(
                (
                    (
                        all_nearest
                        != ref_class
                    )
                    & valid_ref
                ).sum()
            )

            changed = ~same

            if changed.any():
                pair_codes = (
                    (
                        id_all[
                            changed
                        ].astype(
                            np.uint64
                        )
                        << 24
                    )
                    |
                    id_ref[
                        changed
                    ].astype(
                        np.uint64
                    )
                )

                values, counts = np.unique(
                    pair_codes,
                    return_counts=True,
                )

                for value, count in zip(
                    values.tolist(),
                    counts.tolist(),
                ):
                    changed_transitions[
                        int(value)
                    ] += int(count)

    top_transitions = []

    for code, count in (
        changed_transitions.most_common(
            30
        )
    ):
        all_id = (
            code >> 24
        ) & 0xFFFFFF

        ref_id = (
            code
            & 0xFFFFFF
        )

        top_transitions.append(
            {
                "all_rgb": list(
                    id_to_rgb(all_id)
                ),
                "participant_rgb": list(
                    id_to_rgb(ref_id)
                ),
                "pixel_count": count,
            }
        )

    return {
        "all_path": str(all_path),
        "participant_path": str(
            participant_path
        ),
        "same_pixels": (
            same_pixels
        ),
        "changed_pixels": (
            changed_pixels
        ),
        "participant_unknown_pixels": (
            participant_unknown_pixels
        ),
        "nearest_class_mismatch_pixels": (
            nearest_class_mismatch
        ),
        "participant_exact_class_counts": {
            name: int(
                participant_class_counts[i]
            )
            for i, name in enumerate(
                CLASS_NAMES
            )
        },
        "all_nearest_class_counts_diagnostic": {
            name: int(
                all_nearest_class_counts[i]
            )
            for i, name in enumerate(
                CLASS_NAMES
            )
        },
        "changed_transition_count": (
            len(changed_transitions)
        ),
        "top_changed_transitions": (
            top_transitions
        ),
    }


def print_image_summary(
    prefix: str,
    info: dict,
) -> None:
    print(
        f"{prefix}: "
        f"unique={info['unique_rgb_count']} "
        f"unknown_unique="
        f"{info['unknown_unique_count']} "
        f"unknown_pixels="
        f"{info['unknown_pixel_count']} "
        f"target_252_255_0="
        f"{info['target_252_255_0_count']}"
    )


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise SystemExit(
            f"ERROR: root not found: "
            f"{root}"
        )

    if args.chunk_rows <= 0:
        raise SystemExit(
            "ERROR: --chunk-rows "
            "must be > 0"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "root": str(root),
        "tiles": {},
        "zip_checks": [],
    }

    print(
        "=== ZIP / extracted trace ==="
    )

    for source in SOURCES:
        zip_path = find_zip(
            root,
            source,
        )

        print()
        print(
            f"Source: {source}"
        )
        print(
            f"ZIP   : {zip_path}"
        )

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            bad_member = zf.testzip()

            if bad_member is None:
                print(
                    "ZIP CRC test: PASS"
                )
            else:
                print(
                    "ZIP CRC test: FAIL"
                )
                print(
                    f"Bad member: "
                    f"{bad_member}"
                )

            for tile in TARGET_TILES:
                member = find_zip_member(
                    zf,
                    tile,
                )

                extracted = find_extracted(
                    root,
                    source,
                    tile,
                )

                zip_info = (
                    inspect_zip_member(
                        zf,
                        member,
                        args.chunk_rows,
                    )
                )

                extracted_info = (
                    inspect_image(
                        extracted,
                        args.chunk_rows,
                    )
                )

                same_file = (
                    zip_info[
                        "member_sha256"
                    ]
                    ==
                    extracted_info[
                        "file_sha256"
                    ]
                )

                same_pixels = (
                    zip_info[
                        "image"
                    ][
                        "pixel_sha256"
                    ]
                    ==
                    extracted_info[
                        "pixel_sha256"
                    ]
                )

                print()
                print(
                    f"Tile {tile}"
                )
                print(
                    f"  member    : "
                    f"{member}"
                )
                print(
                    f"  extracted : "
                    f"{extracted}"
                )
                print(
                    "  archive member SHA256 "
                    "== extracted SHA256: "
                    f"{same_file}"
                )
                print(
                    "  decoded pixels identical: "
                    f"{same_pixels}"
                )

                print_image_summary(
                    "  archive",
                    zip_info["image"],
                )

                print_image_summary(
                    "  extracted",
                    extracted_info,
                )

                report[
                    "zip_checks"
                ].append(
                    {
                        "source": source,
                        "tile": tile,
                        "zip_path": (
                            str(zip_path)
                        ),
                        "archive": (
                            zip_info
                        ),
                        "extracted": (
                            extracted_info
                        ),
                        "archive_file_equals_extracted": (
                            same_file
                        ),
                        "archive_pixels_equal_extracted": (
                            same_pixels
                        ),
                    }
                )

    print()
    print(
        "=== Duplicate pixel comparison ==="
    )

    for tile in TARGET_TILES:
        all_path = find_extracted(
            root,
            "5_Labels_all",
            tile,
        )

        participant_path = find_extracted(
            root,
            "5_Labels_for_participants",
            tile,
        )

        comparison = (
            compare_duplicate_pair(
                all_path,
                participant_path,
                args.chunk_rows,
            )
        )

        report["tiles"][tile] = (
            comparison
        )

        print()
        print(
            f"Tile {tile}"
        )
        print(
            f"  same pixels    : "
            f"{comparison['same_pixels']}"
        )
        print(
            f"  changed pixels : "
            f"{comparison['changed_pixels']}"
        )
        print(
            "  participant unknown pixels: "
            f"{comparison['participant_unknown_pixels']}"
        )
        print(
            "  nearest-class spatial "
            "mismatches (diagnostic only): "
            f"{comparison['nearest_class_mismatch_pixels']}"
        )

        print(
            "  Participant exact "
            "class counts:"
        )

        for name, count in (
            comparison[
                "participant_exact_class_counts"
            ].items()
        ):
            print(
                f"    {name:<20} "
                f"{count}"
            )

        print(
            "  All nearest-standard "
            "class counts "
            "(diagnostic only):"
        )

        for name, count in (
            comparison[
                "all_nearest_class_counts_diagnostic"
            ].items()
        ):
            print(
                f"    {name:<20} "
                f"{count}"
            )

        print(
            "  Top changed RGB "
            "transitions:"
        )

        for item in (
            comparison[
                "top_changed_transitions"
            ]
        ):
            print(
                "    "
                f"{tuple(item['all_rgb'])}"
                " -> "
                f"{tuple(item['participant_rgb'])}"
                f" : "
                f"{item['pixel_count']}"
            )

    output_json = (
        out_dir
        / "anomaly_trace.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== Saved ==="
    )
    print(
        output_json
    )

    print()
    print(
        "Trace completed."
    )
    print(
        "No GT file was modified."
    )
    print(
        "No approximate RGB remapping "
        "was performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
