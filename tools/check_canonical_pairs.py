#!/usr/bin/env python3
"""
Strict RGBIR <-> canonical GT pairing check for ISPRS Potsdam.

Canonical GT policy:
    - Read ONLY data/processed/potsdam/labels_manifest.json
    - Read canonical entries ONLY from manifest["tiles"]
    - Resolve manifest["tiles"][i]["gt_relpath"] relative to the
      Potsdam raw-data root.
    - Never scan 5_Labels* directories to select or replace GT.
    - Never silently ignore missing, duplicate, or invalid files.

RGBIR:
    - Scan only 4_Ortho_RGBIR
    - Read TIFFs with tifffile, never Pillow

Checks:
    RGBIR:
        shape == (6000, 6000, 4)
        dtype == uint8

    GT:
        exact manifest path exists
        shape == (6000, 6000, 3)
        dtype == uint8
        file_sha256 matches manifest
        pixel_sha256 matches manifest

    Pair:
        one RGBIR per tile
        one canonical GT per tile
        equal height and width

Outputs:
    outputs/dataset_check/canonical_pairs/
        console.txt
        canonical_pairs.json
        canonical_pairs.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


EXPECTED_RGBIR_SHAPE = (6000, 6000, 4)
EXPECTED_GT_SHAPE = (6000, 6000, 3)
EXPECTED_DTYPE = np.dtype("uint8")
EXPECTED_TILE_COUNT = 38

RGBIR_RE = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_RGBIR\.tiff?$",
    re.IGNORECASE,
)

TILE_ID_RE = re.compile(r"^(\d+)_(\d+)$")

ALLOWED_GT_SOURCES = {
    "5_Labels_for_participants",
    "5_Labels_all",
}


class Tee:
    def __init__(self, terminal, output_path: Path):
        self.terminal = terminal
        self.file = output_path.open(
            "w",
            encoding="utf-8",
        )

    def write(self, text: str) -> None:
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Strict ISPRS Potsdam RGBIR <-> canonical GT "
            "pairing validation."
        )
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/potsdam/labels_manifest.json",
        help="Canonical GT manifest.",
    )

    parser.add_argument(
        "--potsdam-raw-root",
        default="data/raw/potsdam",
        help=(
            "Potsdam raw-data root. Manifest gt_relpath values "
            "are resolved relative to this directory."
        ),
    )

    parser.add_argument(
        "--rgbir-root",
        default="data/raw/potsdam/_expanded/4_Ortho_RGBIR",
        help="Extracted 4_Ortho_RGBIR root.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/dataset_check/canonical_pairs",
        help="Output report directory.",
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_TILE_COUNT,
        help="Expected Potsdam supervised tile count.",
    )

    args = parser.parse_args()
    args.project_root = project_root

    return args


def resolve_project_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def display_path(
    path: Path,
    project_root: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                project_root.resolve()
            )
        )
    except ValueError:
        return str(path.resolve())


def tile_sort_key(tile_id: str) -> tuple[int, int]:
    a, b = tile_id.split("_")
    return int(a), int(b)


def normalize_tile_id(value: Any) -> str:
    text = str(value).strip()

    match = TILE_ID_RE.fullmatch(text)

    if match is None:
        raise ValueError(
            f"Invalid tile_id: {value!r}"
        )

    return (
        f"{int(match.group(1))}_"
        f"{int(match.group(2))}"
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_pixels(array: np.ndarray) -> str:
    """
    SHA256 of decoded array pixels in C order.

    This matches the canonical manifest's pixel-level digest
    if that manifest was created from ndarray.tobytes().
    """
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        array.tobytes(order="C")
    ).hexdigest()


def validate_sha_string(
    value: Any,
    field_name: str,
) -> str:
    text = str(value).strip().lower()

    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(
            f"{field_name} is not a valid SHA256: "
            f"{value!r}"
        )

    return text


def load_manifest(
    manifest_path: Path,
    potsdam_raw_root: Path,
    expected_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    """
    Read the exact frozen manifest schema.

    No filesystem search is performed to discover GT files.
    """

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    global_errors: list[str] = []

    if not isinstance(manifest, dict):
        raise TypeError(
            "Manifest top level must be a dictionary."
        )

    tiles = manifest.get("tiles")

    if not isinstance(tiles, list):
        raise ValueError(
            'Manifest must contain a top-level "tiles" list.'
        )

    manifest_expected = manifest.get(
        "expected_tile_count"
    )
    manifest_selected = manifest.get(
        "selected_tile_count"
    )

    if manifest_expected != expected_count:
        global_errors.append(
            "manifest expected_tile_count is "
            f"{manifest_expected!r}, expected "
            f"{expected_count}"
        )

    if manifest_selected != expected_count:
        global_errors.append(
            "manifest selected_tile_count is "
            f"{manifest_selected!r}, expected "
            f"{expected_count}"
        )

    if len(tiles) != expected_count:
        global_errors.append(
            f'manifest["tiles"] has {len(tiles)} entries, '
            f"expected {expected_count}"
        )

    path_semantics = manifest.get(
        "path_semantics"
    )

    if not isinstance(path_semantics, str):
        global_errors.append(
            "manifest path_semantics is missing "
            "or is not a string"
        )

    records: list[dict[str, Any]] = []

    raw_root_resolved = potsdam_raw_root.resolve()

    for index, entry in enumerate(tiles):
        entry_errors: list[str] = []

        if not isinstance(entry, dict):
            global_errors.append(
                f"manifest tile entry #{index} "
                "is not a dictionary"
            )
            continue

        # ----------------------------------------------------------
        # tile_id
        # ----------------------------------------------------------
        try:
            tile_id = normalize_tile_id(
                entry["tile_id"]
            )
        except KeyError:
            global_errors.append(
                f"manifest entry #{index}: "
                "missing tile_id"
            )
            continue
        except Exception as exc:
            global_errors.append(
                f"manifest entry #{index}: {exc}"
            )
            continue

        # ----------------------------------------------------------
        # source
        # ----------------------------------------------------------
        source = entry.get("source")

        if source not in ALLOWED_GT_SOURCES:
            entry_errors.append(
                f"unexpected source {source!r}"
            )

        # ----------------------------------------------------------
        # gt_relpath
        # ----------------------------------------------------------
        gt_relpath_value = entry.get(
            "gt_relpath"
        )

        if not isinstance(
            gt_relpath_value,
            str,
        ) or not gt_relpath_value.strip():
            entry_errors.append(
                "gt_relpath missing or invalid"
            )
            gt_relpath = None
            gt_path = None

        else:
            gt_relpath = Path(
                gt_relpath_value.strip()
            )

            if gt_relpath.is_absolute():
                entry_errors.append(
                    "gt_relpath must be relative "
                    "to Potsdam raw-data root"
                )
                gt_path = None

            else:
                gt_path = (
                    raw_root_resolved
                    / gt_relpath
                ).resolve()

                # Security / correctness check:
                # manifest path must remain inside raw root.
                try:
                    gt_path.relative_to(
                        raw_root_resolved
                    )
                except ValueError:
                    entry_errors.append(
                        "gt_relpath escapes "
                        "Potsdam raw-data root"
                    )

        # ----------------------------------------------------------
        # Frozen metadata
        # ----------------------------------------------------------
        if entry.get("shape") != [6000, 6000, 3]:
            entry_errors.append(
                "manifest shape is "
                f"{entry.get('shape')!r}, "
                "expected [6000, 6000, 3]"
            )

        if entry.get("mode") != "RGB":
            entry_errors.append(
                "manifest mode is "
                f"{entry.get('mode')!r}, "
                "expected 'RGB'"
            )

        if entry.get("dtype") != "uint8":
            entry_errors.append(
                "manifest dtype is "
                f"{entry.get('dtype')!r}, "
                "expected 'uint8'"
            )

        if entry.get("bands") != ["R", "G", "B"]:
            entry_errors.append(
                "manifest bands are "
                f"{entry.get('bands')!r}, "
                "expected ['R', 'G', 'B']"
            )

        # ----------------------------------------------------------
        # SHA256 metadata
        # ----------------------------------------------------------
        try:
            expected_file_sha256 = (
                validate_sha_string(
                    entry["file_sha256"],
                    "file_sha256",
                )
            )
        except KeyError:
            entry_errors.append(
                "file_sha256 missing"
            )
            expected_file_sha256 = None
        except Exception as exc:
            entry_errors.append(str(exc))
            expected_file_sha256 = None

        try:
            expected_pixel_sha256 = (
                validate_sha_string(
                    entry["pixel_sha256"],
                    "pixel_sha256",
                )
            )
        except KeyError:
            entry_errors.append(
                "pixel_sha256 missing"
            )
            expected_pixel_sha256 = None
        except Exception as exc:
            entry_errors.append(str(exc))
            expected_pixel_sha256 = None

        records.append(
            {
                "manifest_index": index,
                "tile_id": tile_id,
                "source": source,
                "selection_reason": entry.get(
                    "selection_reason"
                ),
                "gt_relpath": (
                    str(gt_relpath)
                    if gt_relpath is not None
                    else None
                ),
                "gt_path": gt_path,
                "expected_file_sha256": (
                    expected_file_sha256
                ),
                "expected_pixel_sha256": (
                    expected_pixel_sha256
                ),
                "manifest_errors": entry_errors,
            }
        )

    metadata = {
        "schema_version": manifest.get(
            "schema_version"
        ),
        "dataset": manifest.get("dataset"),
        "purpose": manifest.get("purpose"),
        "selection_policy": manifest.get(
            "selection_policy"
        ),
        "path_semantics": path_semantics,
        "expected_tile_count": (
            manifest_expected
        ),
        "selected_tile_count": (
            manifest_selected
        ),
    }

    return records, global_errors, metadata


def discover_rgbir(
    rgbir_root: Path,
) -> tuple[
    list[Path],
    dict[str, list[Path]],
    list[Path],
]:
    if not rgbir_root.is_dir():
        raise FileNotFoundError(
            "RGBIR root does not exist: "
            f"{rgbir_root}"
        )

    all_tiffs = sorted(
        path.resolve()
        for path in rgbir_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in {".tif", ".tiff"}
        )
    )

    matching_files: list[Path] = []
    unexpected_tiffs: list[Path] = []
    by_tile: dict[str, list[Path]] = (
        defaultdict(list)
    )

    for path in all_tiffs:
        match = RGBIR_RE.fullmatch(
            path.name
        )

        if match is None:
            unexpected_tiffs.append(path)
            continue

        tile_id = (
            f"{int(match.group(1))}_"
            f"{int(match.group(2))}"
        )

        matching_files.append(path)
        by_tile[tile_id].append(path)

    return (
        matching_files,
        dict(by_tile),
        unexpected_tiffs,
    )


def inspect_rgbir(
    path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "readable": False,
        "shape": None,
        "dtype": None,
        "height": None,
        "width": None,
        "channels": None,
        "errors": [],
        "valid": False,
    }

    if not path.is_file():
        result["errors"].append(
            "RGBIR file does not exist"
        )
        return result

    try:
        array = tifffile.imread(path)

    except Exception as exc:
        result["errors"].append(
            "tifffile read failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return result

    result["readable"] = True
    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)

    if array.ndim >= 2:
        result["height"] = int(
            array.shape[0]
        )
        result["width"] = int(
            array.shape[1]
        )

    if array.ndim == 3:
        result["channels"] = int(
            array.shape[2]
        )

    if tuple(array.shape) != (
        EXPECTED_RGBIR_SHAPE
    ):
        result["errors"].append(
            "RGBIR shape is "
            f"{tuple(array.shape)}, expected "
            f"{EXPECTED_RGBIR_SHAPE}"
        )

    if array.dtype != EXPECTED_DTYPE:
        result["errors"].append(
            f"RGBIR dtype is {array.dtype}, "
            "expected uint8"
        )

    if array.ndim != 3:
        result["errors"].append(
            f"RGBIR ndim is {array.ndim}, "
            "expected 3"
        )

    elif array.shape[2] != 4:
        result["errors"].append(
            f"RGBIR channels = "
            f"{array.shape[2]}, expected 4"
        )

    result["valid"] = (
        len(result["errors"]) == 0
    )

    del array

    return result


def inspect_gt(
    record: dict[str, Any],
) -> dict[str, Any]:
    path = record["gt_path"]

    result: dict[str, Any] = {
        "path": (
            str(path)
            if path is not None
            else None
        ),
        "exists": False,
        "readable": False,
        "shape": None,
        "dtype": None,
        "height": None,
        "width": None,
        "channels": None,
        "file_sha256_expected": (
            record["expected_file_sha256"]
        ),
        "file_sha256_actual": None,
        "file_sha256_verified": False,
        "pixel_sha256_expected": (
            record["expected_pixel_sha256"]
        ),
        "pixel_sha256_actual": None,
        "pixel_sha256_verified": False,
        "errors": list(
            record["manifest_errors"]
        ),
        "valid": False,
    }

    if path is None:
        result["errors"].append(
            "GT path could not be resolved "
            "from manifest gt_relpath"
        )
        return result

    if not path.is_file():
        result["errors"].append(
            "canonical GT file does not exist"
        )
        return result

    result["exists"] = True

    # --------------------------------------------------------------
    # File-level SHA256
    # --------------------------------------------------------------
    expected_file_sha = (
        record["expected_file_sha256"]
    )

    if expected_file_sha is not None:
        try:
            actual_file_sha = sha256_file(
                path
            )

            result[
                "file_sha256_actual"
            ] = actual_file_sha

            result[
                "file_sha256_verified"
            ] = (
                actual_file_sha
                == expected_file_sha
            )

            if (
                actual_file_sha
                != expected_file_sha
            ):
                result["errors"].append(
                    "file SHA256 mismatch: "
                    f"expected={expected_file_sha}, "
                    f"actual={actual_file_sha}"
                )

        except Exception as exc:
            result["errors"].append(
                "file SHA256 calculation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------------
    # Decode with tifffile
    # --------------------------------------------------------------
    try:
        array = tifffile.imread(path)

    except Exception as exc:
        result["errors"].append(
            "tifffile read failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return result

    result["readable"] = True
    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)

    if array.ndim >= 2:
        result["height"] = int(
            array.shape[0]
        )
        result["width"] = int(
            array.shape[1]
        )

    if array.ndim == 3:
        result["channels"] = int(
            array.shape[2]
        )

    if tuple(array.shape) != EXPECTED_GT_SHAPE:
        result["errors"].append(
            "GT shape is "
            f"{tuple(array.shape)}, expected "
            f"{EXPECTED_GT_SHAPE}"
        )

    if array.dtype != EXPECTED_DTYPE:
        result["errors"].append(
            f"GT dtype is {array.dtype}, "
            "expected uint8"
        )

    if array.ndim != 3:
        result["errors"].append(
            f"GT ndim is {array.ndim}, "
            "expected 3"
        )

    elif array.shape[2] != 3:
        result["errors"].append(
            f"GT channels = "
            f"{array.shape[2]}, expected 3"
        )

    # --------------------------------------------------------------
    # Pixel SHA256
    # --------------------------------------------------------------
    expected_pixel_sha = (
        record["expected_pixel_sha256"]
    )

    if expected_pixel_sha is not None:
        try:
            actual_pixel_sha = (
                sha256_pixels(array)
            )

            result[
                "pixel_sha256_actual"
            ] = actual_pixel_sha

            result[
                "pixel_sha256_verified"
            ] = (
                actual_pixel_sha
                == expected_pixel_sha
            )

            if (
                actual_pixel_sha
                != expected_pixel_sha
            ):
                result["errors"].append(
                    "pixel SHA256 mismatch: "
                    f"expected={expected_pixel_sha}, "
                    f"actual={actual_pixel_sha}"
                )

        except Exception as exc:
            result["errors"].append(
                "pixel SHA256 calculation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    result["valid"] = (
        len(result["errors"]) == 0
    )

    del array

    return result


def format_tile_list(
    values: list[str],
) -> str:
    if not values:
        return "NONE"

    return ", ".join(values)


def run_checks(
    args: argparse.Namespace,
) -> dict[str, Any]:
    project_root = (
        args.project_root.resolve()
    )

    manifest_path = resolve_project_path(
        project_root,
        args.manifest,
    )

    potsdam_raw_root = resolve_project_path(
        project_root,
        args.potsdam_raw_root,
    )

    rgbir_root = resolve_project_path(
        project_root,
        args.rgbir_root,
    )

    print("=" * 80)
    print(
        "Potsdam RGBIR <-> Canonical GT "
        "Strict Pairing Check"
    )
    print("=" * 80)

    print(
        f"Project root     : {project_root}"
    )
    print(
        f"Manifest         : {manifest_path}"
    )
    print(
        f"Potsdam raw root : {potsdam_raw_root}"
    )
    print(
        f"RGBIR root       : {rgbir_root}"
    )
    print(
        f"Expected tiles   : "
        f"{args.expected_count}"
    )
    print()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Canonical manifest does not exist: "
            f"{manifest_path}"
        )

    if not potsdam_raw_root.is_dir():
        raise FileNotFoundError(
            "Potsdam raw-data root does not exist: "
            f"{potsdam_raw_root}"
        )

    # ==============================================================
    # 1. Manifest
    # ==============================================================

    (
        manifest_records,
        manifest_global_errors,
        manifest_metadata,
    ) = load_manifest(
        manifest_path=manifest_path,
        potsdam_raw_root=potsdam_raw_root,
        expected_count=args.expected_count,
    )

    gt_by_tile: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    gt_path_to_tiles: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for record in manifest_records:
        gt_by_tile[
            record["tile_id"]
        ].append(record)

        if record["gt_path"] is not None:
            gt_path_to_tiles[
                str(record["gt_path"])
            ].append(
                record["tile_id"]
            )

    duplicate_gt_tile_ids = {
        tile_id: records
        for tile_id, records
        in gt_by_tile.items()
        if len(records) > 1
    }

    duplicate_gt_paths = {
        path: tile_ids
        for path, tile_ids
        in gt_path_to_tiles.items()
        if len(tile_ids) > 1
    }

    source_counts = Counter(
        record["source"]
        for record in manifest_records
    )

    # ==============================================================
    # 2. RGBIR discovery
    # ==============================================================

    (
        rgbir_files,
        rgbir_by_tile,
        unexpected_rgbir_tiffs,
    ) = discover_rgbir(
        rgbir_root
    )

    duplicate_rgbir_tile_ids = {
        tile_id: paths
        for tile_id, paths
        in rgbir_by_tile.items()
        if len(paths) > 1
    }

    rgb_tile_ids = set(
        rgbir_by_tile.keys()
    )

    gt_tile_ids = set(
        gt_by_tile.keys()
    )

    missing_rgbir_tiles = sorted(
        gt_tile_ids - rgb_tile_ids,
        key=tile_sort_key,
    )

    extra_rgbir_tiles = sorted(
        rgb_tile_ids - gt_tile_ids,
        key=tile_sort_key,
    )

    missing_gt_tiles = sorted(
        rgb_tile_ids - gt_tile_ids,
        key=tile_sort_key,
    )

    # ==============================================================
    # 3. Inspect RGBIR
    # ==============================================================

    print(
        "Reading RGBIR TIFFs with tifffile..."
    )

    rgbir_checks: dict[
        str,
        dict[str, Any],
    ] = {}

    for index, path in enumerate(
        rgbir_files,
        start=1,
    ):
        print(
            f"  RGBIR "
            f"[{index:02d}/{len(rgbir_files):02d}] "
            f"{display_path(path, project_root)}"
        )

        rgbir_checks[str(path)] = (
            inspect_rgbir(path)
        )

    invalid_rgbir_files = []

    for path_text, check in (
        rgbir_checks.items()
    ):
        if not check["valid"]:
            invalid_rgbir_files.append(
                {
                    "path": display_path(
                        Path(path_text),
                        project_root,
                    ),
                    "errors": list(
                        check["errors"]
                    ),
                }
            )

    for path in unexpected_rgbir_tiffs:
        invalid_rgbir_files.append(
            {
                "path": display_path(
                    path,
                    project_root,
                ),
                "errors": [
                    "TIFF exists inside RGBIR "
                    "root but filename does not "
                    "match "
                    "top_potsdam_<x>_<y>_RGBIR.tif"
                ],
            }
        )

    # ==============================================================
    # 4. Inspect canonical GT
    # ==============================================================

    print()
    print(
        "Reading canonical GT TIFFs "
        "from manifest with tifffile..."
    )

    gt_checks: dict[
        int,
        dict[str, Any],
    ] = {}

    for index, record in enumerate(
        manifest_records,
        start=1,
    ):
        gt_path = record["gt_path"]

        path_for_display = (
            display_path(
                gt_path,
                project_root,
            )
            if gt_path is not None
            else "UNRESOLVED"
        )

        print(
            f"  GT "
            f"[{index:02d}/{len(manifest_records):02d}] "
            f"{record['tile_id']} -> "
            f"{path_for_display}"
        )

        gt_checks[
            record["manifest_index"]
        ] = inspect_gt(record)

    invalid_gt_files = []

    for record in manifest_records:
        check = gt_checks[
            record["manifest_index"]
        ]

        if not check["valid"]:
            invalid_gt_files.append(
                {
                    "tile_id": (
                        record["tile_id"]
                    ),
                    "source": (
                        record["source"]
                    ),
                    "path": (
                        display_path(
                            record["gt_path"],
                            project_root,
                        )
                        if record["gt_path"]
                        is not None
                        else None
                    ),
                    "errors": list(
                        check["errors"]
                    ),
                }
            )

    # ==============================================================
    # 5. Pairing
    # ==============================================================

    all_tile_ids = sorted(
        rgb_tile_ids | gt_tile_ids,
        key=tile_sort_key,
    )

    pair_results = []

    for tile_id in all_tile_ids:
        rgb_paths = rgbir_by_tile.get(
            tile_id,
            [],
        )

        gt_records = gt_by_tile.get(
            tile_id,
            [],
        )

        pair_errors: list[str] = []

        if len(rgb_paths) == 0:
            pair_errors.append(
                "missing RGBIR tile"
            )

        elif len(rgb_paths) > 1:
            pair_errors.append(
                "duplicate RGBIR tile ID: "
                f"{len(rgb_paths)} files"
            )

        if len(gt_records) == 0:
            pair_errors.append(
                "missing canonical GT tile "
                "in manifest"
            )

        elif len(gt_records) > 1:
            pair_errors.append(
                "duplicate canonical GT "
                "tile ID: "
                f"{len(gt_records)} entries"
            )

        rgb_path = (
            rgb_paths[0]
            if len(rgb_paths) == 1
            else None
        )

        gt_record = (
            gt_records[0]
            if len(gt_records) == 1
            else None
        )

        rgb_check = None
        gt_check = None

        same_height = None
        same_width = None

        if rgb_path is not None:
            rgb_check = rgbir_checks[
                str(rgb_path)
            ]

            if not rgb_check["valid"]:
                pair_errors.append(
                    "RGBIR file validation failed"
                )

        if gt_record is not None:
            gt_check = gt_checks[
                gt_record["manifest_index"]
            ]

            if not gt_check["valid"]:
                pair_errors.append(
                    "GT file validation failed"
                )

        if (
            rgb_check is not None
            and gt_check is not None
        ):
            if (
                rgb_check["height"] is not None
                and gt_check["height"]
                is not None
            ):
                same_height = (
                    rgb_check["height"]
                    == gt_check["height"]
                )

                if not same_height:
                    pair_errors.append(
                        "height mismatch: "
                        f"RGBIR="
                        f"{rgb_check['height']}, "
                        f"GT={gt_check['height']}"
                    )

            if (
                rgb_check["width"] is not None
                and gt_check["width"]
                is not None
            ):
                same_width = (
                    rgb_check["width"]
                    == gt_check["width"]
                )

                if not same_width:
                    pair_errors.append(
                        "width mismatch: "
                        f"RGBIR="
                        f"{rgb_check['width']}, "
                        f"GT={gt_check['width']}"
                    )

        pair_results.append(
            {
                "tile_id": tile_id,
                "rgbir_path": (
                    display_path(
                        rgb_path,
                        project_root,
                    )
                    if rgb_path is not None
                    else None
                ),
                "gt_path": (
                    display_path(
                        gt_record["gt_path"],
                        project_root,
                    )
                    if (
                        gt_record is not None
                        and gt_record[
                            "gt_path"
                        ] is not None
                    )
                    else None
                ),
                "gt_source": (
                    gt_record["source"]
                    if gt_record is not None
                    else None
                ),
                "same_height": same_height,
                "same_width": same_width,
                "file_sha256_verified": (
                    gt_check[
                        "file_sha256_verified"
                    ]
                    if gt_check is not None
                    else None
                ),
                "pixel_sha256_verified": (
                    gt_check[
                        "pixel_sha256_verified"
                    ]
                    if gt_check is not None
                    else None
                ),
                "valid": (
                    len(pair_errors) == 0
                ),
                "errors": pair_errors,
            }
        )

    successful_pairs = sum(
        pair["valid"]
        for pair in pair_results
    )

    # A supervised tile exists only if the
    # strict RGBIR+GT pair passes all checks.
    final_supervised_tile_count = (
        successful_pairs
    )

    file_sha_verified_count = sum(
        check["file_sha256_verified"]
        for check in gt_checks.values()
    )

    pixel_sha_verified_count = sum(
        check["pixel_sha256_verified"]
        for check in gt_checks.values()
    )

    # ==============================================================
    # 6. Overall status
    # ==============================================================

    overall_pass = all(
        [
            len(
                manifest_global_errors
            )
            == 0,
            len(manifest_records)
            == args.expected_count,
            len(gt_by_tile)
            == args.expected_count,
            len(rgbir_files)
            == args.expected_count,
            len(rgbir_by_tile)
            == args.expected_count,
            len(
                duplicate_gt_tile_ids
            )
            == 0,
            len(duplicate_gt_paths)
            == 0,
            len(
                duplicate_rgbir_tile_ids
            )
            == 0,
            len(
                unexpected_rgbir_tiffs
            )
            == 0,
            len(missing_rgbir_tiles)
            == 0,
            len(extra_rgbir_tiles)
            == 0,
            len(missing_gt_tiles)
            == 0,
            len(invalid_rgbir_files)
            == 0,
            len(invalid_gt_files)
            == 0,
            file_sha_verified_count
            == args.expected_count,
            pixel_sha_verified_count
            == args.expected_count,
            successful_pairs
            == args.expected_count,
            final_supervised_tile_count
            == args.expected_count,
        ]
    )

    return {
        "check_name": (
            "Potsdam RGBIR <-> "
            "Canonical GT pairing"
        ),
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "project_root": str(
            project_root
        ),
        "manifest": display_path(
            manifest_path,
            project_root,
        ),
        "potsdam_raw_root": display_path(
            potsdam_raw_root,
            project_root,
        ),
        "rgbir_root": display_path(
            rgbir_root,
            project_root,
        ),
        "expected_tile_count": (
            args.expected_count
        ),
        "manifest_metadata": (
            manifest_metadata
        ),
        "manifest_source_counts": dict(
            source_counts
        ),
        "summary": {
            "manifest_records": (
                len(manifest_records)
            ),
            "total_rgbir_files": (
                len(rgbir_files)
            ),
            "total_rgbir_unique_tiles": (
                len(rgbir_by_tile)
            ),
            "total_canonical_gt_tiles": (
                len(gt_by_tile)
            ),
            "successfully_paired_tiles": (
                successful_pairs
            ),
            "invalid_rgbir_files": (
                len(invalid_rgbir_files)
            ),
            "invalid_gt_files": (
                len(invalid_gt_files)
            ),
            "file_sha256_verified": (
                file_sha_verified_count
            ),
            "pixel_sha256_verified": (
                pixel_sha_verified_count
            ),
            "final_supervised_tile_count": (
                final_supervised_tile_count
            ),
        },
        "manifest_errors": (
            manifest_global_errors
        ),
        "missing_rgbir_tiles": (
            missing_rgbir_tiles
        ),
        "extra_rgbir_tiles": (
            extra_rgbir_tiles
        ),
        "missing_gt_tiles": (
            missing_gt_tiles
        ),
        "duplicate_rgbir_tile_ids": {
            tile_id: [
                display_path(
                    path,
                    project_root,
                )
                for path in paths
            ]
            for tile_id, paths
            in sorted(
                duplicate_rgbir_tile_ids.items(),
                key=lambda item:
                tile_sort_key(item[0]),
            )
        },
        "duplicate_gt_tile_ids": {
            tile_id: [
                display_path(
                    record["gt_path"],
                    project_root,
                )
                if record[
                    "gt_path"
                ] is not None
                else None
                for record in records
            ]
            for tile_id, records
            in sorted(
                duplicate_gt_tile_ids.items(),
                key=lambda item:
                tile_sort_key(item[0]),
            )
        },
        "duplicate_gt_paths": {
            display_path(
                Path(path),
                project_root,
            ): tile_ids
            for path, tile_ids
            in duplicate_gt_paths.items()
        },
        "invalid_rgbir_files": (
            invalid_rgbir_files
        ),
        "invalid_gt_files": (
            invalid_gt_files
        ),
        "pairs": pair_results,
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    }


def build_text_report(
    report: dict[str, Any],
) -> str:
    summary = report["summary"]

    lines = [
        "=" * 80,
        (
            "Potsdam RGBIR <-> Canonical GT "
            "Pairing Report"
        ),
        "=" * 80,
        "",
        f"Status: {report['status']}",
        "",
        "[Manifest]",
        (
            "Path semantics : "
            f"{report['manifest_metadata'].get('path_semantics')}"
        ),
        (
            "Selection policy: "
            f"{report['manifest_metadata'].get('selection_policy')}"
        ),
        (
            "GT source counts: "
            f"{report['manifest_source_counts']}"
        ),
        "",
        "[Summary]",
        (
            "Expected supervised tiles       : "
            f"{report['expected_tile_count']}"
        ),
        (
            "Manifest records                : "
            f"{summary['manifest_records']}"
        ),
        (
            "Total RGBIR TIFF files          : "
            f"{summary['total_rgbir_files']}"
        ),
        (
            "Total RGBIR tiles               : "
            f"{summary['total_rgbir_unique_tiles']}"
        ),
        (
            "Total canonical GT tiles        : "
            f"{summary['total_canonical_gt_tiles']}"
        ),
        (
            "Successfully paired tiles       : "
            f"{summary['successfully_paired_tiles']}"
        ),
        (
            "Invalid RGBIR files             : "
            f"{summary['invalid_rgbir_files']}"
        ),
        (
            "Invalid GT files                : "
            f"{summary['invalid_gt_files']}"
        ),
        (
            "GT file SHA256 verified         : "
            f"{summary['file_sha256_verified']}"
        ),
        (
            "GT pixel SHA256 verified        : "
            f"{summary['pixel_sha256_verified']}"
        ),
        (
            "Final supervised tiles          : "
            f"{summary['final_supervised_tile_count']}"
        ),
        "",
        "[Tile-set differences]",
        (
            "Missing RGBIR tiles : "
            + format_tile_list(
                report["missing_rgbir_tiles"]
            )
        ),
        (
            "Extra RGBIR tiles   : "
            + format_tile_list(
                report["extra_rgbir_tiles"]
            )
        ),
        (
            "Missing GT tiles    : "
            + format_tile_list(
                report["missing_gt_tiles"]
            )
        ),
        "",
        "[Manifest errors]",
    ]

    if report["manifest_errors"]:
        for error in report[
            "manifest_errors"
        ]:
            lines.append(
                f"- {error}"
            )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Duplicate RGBIR tile IDs]",
        ]
    )

    if report[
        "duplicate_rgbir_tile_ids"
    ]:
        for tile_id, paths in report[
            "duplicate_rgbir_tile_ids"
        ].items():
            lines.append(
                f"{tile_id}:"
            )

            for path in paths:
                lines.append(
                    f"  - {path}"
                )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Duplicate canonical GT tile IDs]",
        ]
    )

    if report[
        "duplicate_gt_tile_ids"
    ]:
        for tile_id, paths in report[
            "duplicate_gt_tile_ids"
        ].items():
            lines.append(
                f"{tile_id}:"
            )

            for path in paths:
                lines.append(
                    f"  - {path}"
                )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Duplicate canonical GT paths]",
        ]
    )

    if report[
        "duplicate_gt_paths"
    ]:
        for path, tile_ids in report[
            "duplicate_gt_paths"
        ].items():
            lines.append(
                f"{path}: "
                + ", ".join(tile_ids)
            )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Invalid RGBIR files]",
        ]
    )

    if report["invalid_rgbir_files"]:
        for item in report[
            "invalid_rgbir_files"
        ]:
            lines.append(
                f"- {item['path']}"
            )

            for error in item["errors"]:
                lines.append(
                    f"    {error}"
                )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Invalid canonical GT files]",
        ]
    )

    if report["invalid_gt_files"]:
        for item in report[
            "invalid_gt_files"
        ]:
            lines.append(
                f"- {item['tile_id']} "
                f"[{item['source']}]: "
                f"{item['path']}"
            )

            for error in item["errors"]:
                lines.append(
                    f"    {error}"
                )
    else:
        lines.append("NONE")

    lines.extend(
        [
            "",
            "[Per-tile pairing]",
        ]
    )

    for pair in report["pairs"]:
        status = (
            "PASS"
            if pair["valid"]
            else "FAIL"
        )

        lines.append(
            f"{pair['tile_id']}: "
            f"{status}"
        )

        lines.append(
            f"  RGBIR: "
            f"{pair['rgbir_path']}"
        )

        lines.append(
            f"  GT   : "
            f"{pair['gt_path']}"
        )

        lines.append(
            f"  Source: "
            f"{pair['gt_source']}"
        )

        lines.append(
            "  File SHA256: "
            f"{pair['file_sha256_verified']}"
        )

        lines.append(
            "  Pixel SHA256: "
            f"{pair['pixel_sha256_verified']}"
        )

        lines.append(
            "  Same H/W: "
            f"{pair['same_height']}/"
            f"{pair['same_width']}"
        )

        for error in pair["errors"]:
            lines.append(
                f"  ERROR: {error}"
            )

    lines.extend(
        [
            "",
            "=" * 80,
            (
                "FINAL STATUS: "
                f"{report['status']}"
            ),
            (
                "Final supervised tiles = "
                f"{summary['final_supervised_tile_count']}"
            ),
            "=" * 80,
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
    )

    output_dir = resolve_project_path(
        project_root,
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    console_path = (
        output_dir / "console.txt"
    )

    json_path = (
        output_dir / "canonical_pairs.json"
    )

    text_path = (
        output_dir / "canonical_pairs.txt"
    )

    original_stdout = sys.stdout

    tee = Tee(
        original_stdout,
        console_path,
    )

    sys.stdout = tee

    try:
        report = run_checks(args)

        json_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        text_report = build_text_report(
            report
        )

        text_path.write_text(
            text_report,
            encoding="utf-8",
        )

        print()
        print(
            text_report,
            end="",
        )

        print()
        print(
            f"JSON report : {json_path}"
        )
        print(
            f"Text report : {text_path}"
        )
        print(
            f"Console log : {console_path}"
        )

        return (
            0
            if report["status"] == "PASS"
            else 1
        )

    except Exception as exc:
        fatal_message = (
            f"{type(exc).__name__}: {exc}"
        )

        fatal_traceback = (
            traceback.format_exc()
        )

        fatal_report = {
            "check_name": (
                "Potsdam RGBIR <-> "
                "Canonical GT pairing"
            ),
            "generated_at": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            "status": "FATAL",
            "fatal_error": fatal_message,
            "traceback": fatal_traceback,
        }

        json_path.write_text(
            json.dumps(
                fatal_report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        text_path.write_text(
            "Potsdam RGBIR <-> "
            "Canonical GT Pairing Check\n"
            "STATUS: FATAL\n\n"
            f"{fatal_message}\n\n"
            f"{fatal_traceback}\n",
            encoding="utf-8",
        )

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)
        print(fatal_message)
        print()
        print(fatal_traceback)

        return 2

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())