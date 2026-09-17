#!/usr/bin/env python3
"""
Strict RGBIR band-semantics audit for the ISPRS Potsdam experiment.

Purpose
-------
Before any RGB/NIR Dataset, patching, corruption, or model code is written,
prove the channel semantics of 4_Ortho_RGBIR against the independent
2_Ortho_RGB and 3_Ortho_IRRG products.

The script:
  * takes the frozen 38 tile IDs from tile_split.json;
  * scans only the three ortho-product roots;
  * reads TIFFs with tifffile (never Pillow);
  * requires one file per tile in each source;
  * checks shape/dtype;
  * hashes each decoded 2-D channel and finds exact cross-product matches;
  * infers the RGBIR semantic channel order from:
        RGB  convention:  [R, G, B]
        IRRG convention:  [NIR, R, G]
  * requires the inferred RGBIR order to be identical for all 38 tiles;
  * reports sampled MAE matrices if exact channel matching fails;
  * modifies no source data.

No approximate matching is accepted as PASS.

Outputs
-------
outputs/dataset_check/rgbir_bands/
    console.txt
    rgbir_band_semantics.json
    rgbir_band_semantics.txt
"""

from __future__ import annotations

import argparse
import gc
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


EXPECTED_TILE_COUNT = 38
EXPECTED_HW = (6000, 6000)
EXPECTED_RGBIR_SHAPE = (6000, 6000, 4)
EXPECTED_RGB_SHAPE = (6000, 6000, 3)
EXPECTED_IRRG_SHAPE = (6000, 6000, 3)
EXPECTED_DTYPE = np.dtype("uint8")

RGBIR_PATTERN = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_RGBIR\.tiff?$",
    re.IGNORECASE,
)
RGB_PATTERN = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_RGB\.tiff?$",
    re.IGNORECASE,
)
IRRG_PATTERN = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_IRRG\.tiff?$",
    re.IGNORECASE,
)

RGB_SEMANTICS = ("R", "G", "B")
IRRG_SEMANTICS = ("NIR", "R", "G")


class Tee:
    def __init__(self, terminal, path: Path):
        self.terminal = terminal
        self.file = path.open("w", encoding="utf-8")

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
        description="Verify Potsdam RGBIR channel semantics against RGB and IRRG."
    )
    parser.add_argument(
        "--split",
        default="data/processed/potsdam/tile_split.json",
        help="Frozen tile split; used only as the authoritative 38-tile ID set.",
    )
    parser.add_argument(
        "--rgbir-root",
        default="data/raw/potsdam/_expanded/4_Ortho_RGBIR",
    )
    parser.add_argument(
        "--rgb-root",
        default="data/raw/potsdam/_expanded/2_Ortho_RGB",
    )
    parser.add_argument(
        "--irrg-root",
        default="data/raw/potsdam/_expanded/3_Ortho_IRRG",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dataset_check/rgbir_bands",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=32,
        help=(
            "Only used for diagnostic MAE matrices when exact hashing fails. "
            "PASS never depends on approximate matching."
        ),
    )

    args = parser.parse_args()
    args.project_root = project_root
    return args


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def display_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def tile_sort_key(tile_id: str) -> tuple[int, int]:
    a, b = tile_id.split("_")
    return int(a), int(b)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_split_tile_ids(split: dict[str, Any]) -> list[str]:
    if split.get("status") != "PASS":
        raise ValueError(
            f"Frozen tile split status is {split.get('status')!r}, expected 'PASS'."
        )

    splits = split.get("splits")
    if not isinstance(splits, dict):
        raise ValueError('tile_split.json has no "splits" dictionary.')

    ids: list[str] = []
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list):
            raise ValueError(f'tile_split.json splits["{name}"] is not a list.')
        ids.extend(values)

    if len(ids) != EXPECTED_TILE_COUNT:
        raise ValueError(
            f"Frozen split contains {len(ids)} tile references, "
            f"expected {EXPECTED_TILE_COUNT}."
        )

    if len(set(ids)) != EXPECTED_TILE_COUNT:
        counts = Counter(ids)
        duplicates = sorted(
            [tile_id for tile_id, count in counts.items() if count > 1],
            key=tile_sort_key,
        )
        raise ValueError(f"Frozen split contains duplicate tile IDs: {duplicates}")

    return sorted(ids, key=tile_sort_key)


def scan_source(
    root: Path,
    pattern: re.Pattern[str],
) -> tuple[dict[str, list[Path]], list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {root}")

    by_tile: dict[str, list[Path]] = defaultdict(list)
    unexpected_tiffs: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected_tiffs.append(path.resolve())
            continue

        tile_id = f"{int(match.group(1))}_{int(match.group(2))}"
        by_tile[tile_id].append(path.resolve())

    return dict(by_tile), unexpected_tiffs


def channel_sha256(channel: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(channel)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def channel_hashes(array: np.ndarray) -> list[str]:
    if array.ndim != 3:
        return []
    return [channel_sha256(array[..., c]) for c in range(array.shape[2])]


def sampled_mae_matrix(
    reference: np.ndarray,
    target: np.ndarray,
    stride: int,
) -> list[list[float]]:
    stride = max(1, int(stride))

    ref = reference[::stride, ::stride, :].astype(np.int16, copy=False)
    tgt = target[::stride, ::stride, :].astype(np.int16, copy=False)

    matrix: list[list[float]] = []
    for ref_c in range(ref.shape[2]):
        row = []
        for tgt_c in range(tgt.shape[2]):
            mae = np.abs(
                ref[..., ref_c] - tgt[..., tgt_c]
            ).mean(dtype=np.float64)
            row.append(float(mae))
        matrix.append(row)
    return matrix


def exact_mapping(
    reference_hashes: list[str],
    target_hashes: list[str],
) -> dict[int, list[int]]:
    """
    For every reference channel, return all exactly equal target channels.
    """
    mapping: dict[int, list[int]] = {}
    for ref_idx, ref_hash in enumerate(reference_hashes):
        mapping[ref_idx] = [
            target_idx
            for target_idx, target_hash in enumerate(target_hashes)
            if ref_hash == target_hash
        ]
    return mapping


def unique_mapping(mapping: dict[int, list[int]]) -> dict[int, int] | None:
    if not mapping:
        return None

    if any(len(matches) != 1 for matches in mapping.values()):
        return None

    collapsed = {
        ref_idx: matches[0]
        for ref_idx, matches in mapping.items()
    }

    if len(set(collapsed.values())) != len(collapsed):
        return None

    return collapsed


def inspect_array(
    path: Path,
    expected_shape: tuple[int, int, int],
    role: str,
) -> tuple[np.ndarray | None, list[str]]:
    errors: list[str] = []

    try:
        array = tifffile.imread(path)
    except Exception as exc:
        errors.append(
            f"{role} tifffile read failed: {type(exc).__name__}: {exc}"
        )
        return None, errors

    if tuple(array.shape) != expected_shape:
        errors.append(
            f"{role} shape is {tuple(array.shape)}, expected {expected_shape}"
        )

    if array.dtype != EXPECTED_DTYPE:
        errors.append(
            f"{role} dtype is {array.dtype}, expected uint8"
        )

    if array.ndim != 3:
        errors.append(
            f"{role} ndim is {array.ndim}, expected 3"
        )

    return array, errors


def infer_rgbir_semantics(
    rgb_to_rgbir: dict[int, int],
    irrg_to_rgbir: dict[int, int],
) -> tuple[list[str | None], list[str]]:
    """
    Infer semantic labels for the 4 RGBIR channels.

    RGB convention:
        RGB channel 0 -> R
        RGB channel 1 -> G
        RGB channel 2 -> B

    IRRG convention:
        IRRG channel 0 -> NIR
        IRRG channel 1 -> R
        IRRG channel 2 -> G
    """
    semantics: list[str | None] = [None, None, None, None]
    errors: list[str] = []

    assignments: list[tuple[str, int, str]] = []

    for ref_idx, target_idx in rgb_to_rgbir.items():
        assignments.append(("RGB", target_idx, RGB_SEMANTICS[ref_idx]))

    for ref_idx, target_idx in irrg_to_rgbir.items():
        assignments.append(("IRRG", target_idx, IRRG_SEMANTICS[ref_idx]))

    for source, target_idx, semantic in assignments:
        current = semantics[target_idx]

        if current is None:
            semantics[target_idx] = semantic
        elif current != semantic:
            errors.append(
                f"semantic conflict on RGBIR channel {target_idx}: "
                f"already {current}, {source} implies {semantic}"
            )

    expected_set = {"R", "G", "B", "NIR"}
    actual_set = {x for x in semantics if x is not None}

    if actual_set != expected_set or any(x is None for x in semantics):
        errors.append(
            f"incomplete/ambiguous RGBIR semantics: {semantics}; "
            f"expected exactly {sorted(expected_set)}"
        )

    return semantics, errors


def build_text_report(report: dict[str, Any]) -> str:
    s = report["summary"]

    lines = [
        "=" * 80,
        "Potsdam RGBIR Band-Semantics Audit",
        "=" * 80,
        "",
        f"Status: {report['status']}",
        "",
        "[Summary]",
        f"Frozen tile IDs                  : {s['frozen_tile_count']}",
        f"RGBIR unique tiles               : {s['rgbir_unique_tiles']}",
        f"RGB unique tiles                 : {s['rgb_unique_tiles']}",
        f"IRRG unique tiles                : {s['irrg_unique_tiles']}",
        f"Fully checked tiles              : {s['fully_checked_tiles']}",
        f"Exact semantic tiles             : {s['exact_semantic_tiles']}",
        f"Invalid tiles                    : {s['invalid_tiles']}",
        "",
        "[Source-set differences]",
        f"Missing RGBIR : {', '.join(report['missing_rgbir']) or 'NONE'}",
        f"Missing RGB   : {', '.join(report['missing_rgb']) or 'NONE'}",
        f"Missing IRRG  : {', '.join(report['missing_irrg']) or 'NONE'}",
        f"Extra RGBIR   : {', '.join(report['extra_rgbir']) or 'NONE'}",
        f"Extra RGB     : {', '.join(report['extra_rgb']) or 'NONE'}",
        f"Extra IRRG    : {', '.join(report['extra_irrg']) or 'NONE'}",
        "",
        "[Duplicates]",
        f"RGBIR duplicate tile IDs: "
        f"{sorted(report['duplicate_rgbir'].keys(), key=tile_sort_key) or 'NONE'}",
        f"RGB duplicate tile IDs  : "
        f"{sorted(report['duplicate_rgb'].keys(), key=tile_sort_key) or 'NONE'}",
        f"IRRG duplicate tile IDs : "
        f"{sorted(report['duplicate_irrg'].keys(), key=tile_sort_key) or 'NONE'}",
        "",
        "[Global inferred RGBIR channel order]",
        (
            ", ".join(
                f"ch{i}={name}"
                for i, name in enumerate(report["global_rgbir_semantics"])
            )
            if report["global_rgbir_semantics"] is not None
            else "UNRESOLVED"
        ),
        "",
        "[Per-tile results]",
    ]

    for item in report["tiles"]:
        lines.append(
            f"{item['tile_id']}: {'PASS' if item['valid'] else 'FAIL'}"
        )
        if item["rgb_to_rgbir"] is not None:
            lines.append(
                "  RGB -> RGBIR: "
                + ", ".join(
                    f"{RGB_SEMANTICS[int(k)]}->ch{v}"
                    for k, v in sorted(
                        ((int(k), v) for k, v in item["rgb_to_rgbir"].items()),
                        key=lambda pair: pair[0],
                    )
                )
            )
        if item["irrg_to_rgbir"] is not None:
            lines.append(
                "  IRRG -> RGBIR: "
                + ", ".join(
                    f"{IRRG_SEMANTICS[int(k)]}->ch{v}"
                    for k, v in sorted(
                        ((int(k), v) for k, v in item["irrg_to_rgbir"].items()),
                        key=lambda pair: pair[0],
                    )
                )
            )
        if item["rgbir_semantics"] is not None:
            lines.append(
                "  RGBIR order: "
                + ", ".join(
                    f"ch{i}={name}"
                    for i, name in enumerate(item["rgbir_semantics"])
                )
            )
        for error in item["errors"]:
            lines.append(f"  ERROR: {error}")

    lines.extend(
        [
            "",
            "=" * 80,
            f"FINAL STATUS: {report['status']}",
            (
                "RGBIR semantic order = "
                + (
                    ", ".join(
                        f"ch{i}:{name}"
                        for i, name in enumerate(report["global_rgbir_semantics"])
                    )
                    if report["global_rgbir_semantics"] is not None
                    else "UNRESOLVED"
                )
            ),
            "=" * 80,
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    split_path = resolve_project_path(project_root, args.split)
    rgbir_root = resolve_project_path(project_root, args.rgbir_root)
    rgb_root = resolve_project_path(project_root, args.rgb_root)
    irrg_root = resolve_project_path(project_root, args.irrg_root)
    output_dir = resolve_project_path(project_root, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    console_path = output_dir / "console.txt"
    json_path = output_dir / "rgbir_band_semantics.json"
    txt_path = output_dir / "rgbir_band_semantics.txt"

    original_stdout = sys.stdout
    tee = Tee(original_stdout, console_path)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("Potsdam RGBIR Band-Semantics Audit")
        print("=" * 80)
        print(f"Frozen split : {split_path}")
        print(f"RGBIR root   : {rgbir_root}")
        print(f"RGB root     : {rgb_root}")
        print(f"IRRG root    : {irrg_root}")
        print()

        if not split_path.is_file():
            raise FileNotFoundError(f"Frozen tile split does not exist: {split_path}")

        split = load_json(split_path)
        if not isinstance(split, dict):
            raise TypeError("tile_split.json top level must be a dictionary")

        frozen_ids = extract_split_tile_ids(split)
        frozen_set = set(frozen_ids)

        rgbir_by_tile, unexpected_rgbir = scan_source(
            rgbir_root, RGBIR_PATTERN
        )
        rgb_by_tile, unexpected_rgb = scan_source(
            rgb_root, RGB_PATTERN
        )
        irrg_by_tile, unexpected_irrg = scan_source(
            irrg_root, IRRG_PATTERN
        )

        rgbir_set = set(rgbir_by_tile)
        rgb_set = set(rgb_by_tile)
        irrg_set = set(irrg_by_tile)

        missing_rgbir = sorted(frozen_set - rgbir_set, key=tile_sort_key)
        missing_rgb = sorted(frozen_set - rgb_set, key=tile_sort_key)
        missing_irrg = sorted(frozen_set - irrg_set, key=tile_sort_key)

        extra_rgbir = sorted(rgbir_set - frozen_set, key=tile_sort_key)
        extra_rgb = sorted(rgb_set - frozen_set, key=tile_sort_key)
        extra_irrg = sorted(irrg_set - frozen_set, key=tile_sort_key)

        duplicate_rgbir = {
            tile_id: [display_path(p, project_root) for p in paths]
            for tile_id, paths in rgbir_by_tile.items()
            if len(paths) != 1
        }
        duplicate_rgb = {
            tile_id: [display_path(p, project_root) for p in paths]
            for tile_id, paths in rgb_by_tile.items()
            if len(paths) != 1
        }
        duplicate_irrg = {
            tile_id: [display_path(p, project_root) for p in paths]
            for tile_id, paths in irrg_by_tile.items()
            if len(paths) != 1
        }

        tile_results: list[dict[str, Any]] = []
        observed_orders: list[tuple[str, ...]] = []

        for index, tile_id in enumerate(frozen_ids, start=1):
            print(f"[{index:02d}/{len(frozen_ids):02d}] {tile_id}")

            errors: list[str] = []
            rgb_to_rgbir: dict[int, int] | None = None
            irrg_to_rgbir: dict[int, int] | None = None
            semantics: list[str | None] | None = None
            rgb_to_rgbir_matches: dict[int, list[int]] | None = None
            irrg_to_rgbir_matches: dict[int, list[int]] | None = None
            rgb_vs_irrg_rg_exact = None
            rgb_mae = None
            irrg_mae = None

            rgbir_paths = rgbir_by_tile.get(tile_id, [])
            rgb_paths = rgb_by_tile.get(tile_id, [])
            irrg_paths = irrg_by_tile.get(tile_id, [])

            if len(rgbir_paths) != 1:
                errors.append(
                    f"RGBIR file count for tile is {len(rgbir_paths)}, expected 1"
                )
            if len(rgb_paths) != 1:
                errors.append(
                    f"RGB file count for tile is {len(rgb_paths)}, expected 1"
                )
            if len(irrg_paths) != 1:
                errors.append(
                    f"IRRG file count for tile is {len(irrg_paths)}, expected 1"
                )

            rgbir = rgb = irrg = None

            if not errors:
                rgbir, e = inspect_array(
                    rgbir_paths[0], EXPECTED_RGBIR_SHAPE, "RGBIR"
                )
                errors.extend(e)

                rgb, e = inspect_array(
                    rgb_paths[0], EXPECTED_RGB_SHAPE, "RGB"
                )
                errors.extend(e)

                irrg, e = inspect_array(
                    irrg_paths[0], EXPECTED_IRRG_SHAPE, "IRRG"
                )
                errors.extend(e)

            if rgbir is not None and rgb is not None and irrg is not None:
                rgbir_hashes = channel_hashes(rgbir)
                rgb_hashes = channel_hashes(rgb)
                irrg_hashes = channel_hashes(irrg)

                rgb_to_rgbir_matches = exact_mapping(
                    rgb_hashes, rgbir_hashes
                )
                irrg_to_rgbir_matches = exact_mapping(
                    irrg_hashes, rgbir_hashes
                )

                rgb_to_rgbir = unique_mapping(rgb_to_rgbir_matches)
                irrg_to_rgbir = unique_mapping(irrg_to_rgbir_matches)

                if rgb_to_rgbir is None:
                    errors.append(
                        f"RGB channels do not have a unique exact mapping into RGBIR: "
                        f"{rgb_to_rgbir_matches}"
                    )
                    rgb_mae = sampled_mae_matrix(
                        rgb, rgbir, args.sample_stride
                    )

                if irrg_to_rgbir is None:
                    errors.append(
                        f"IRRG channels do not have a unique exact mapping into RGBIR: "
                        f"{irrg_to_rgbir_matches}"
                    )
                    irrg_mae = sampled_mae_matrix(
                        irrg, rgbir, args.sample_stride
                    )

                # Independent consistency check for the shared R/G bands:
                # RGB[R] == IRRG[R], RGB[G] == IRRG[G].
                rgb_vs_irrg_rg_exact = (
                    rgb_hashes[0] == irrg_hashes[1]
                    and rgb_hashes[1] == irrg_hashes[2]
                )

                if not rgb_vs_irrg_rg_exact:
                    errors.append(
                        "RGB and IRRG shared R/G channels are not exact matches "
                        "under the documented RGB=[R,G,B], IRRG=[NIR,R,G] conventions"
                    )

                if rgb_to_rgbir is not None and irrg_to_rgbir is not None:
                    semantics, semantic_errors = infer_rgbir_semantics(
                        rgb_to_rgbir,
                        irrg_to_rgbir,
                    )
                    errors.extend(semantic_errors)

                    if not semantic_errors and semantics is not None:
                        observed_orders.append(tuple(str(x) for x in semantics))

            valid = len(errors) == 0

            tile_results.append(
                {
                    "tile_id": tile_id,
                    "valid": valid,
                    "rgbir_path": display_path(
                        rgbir_paths[0], project_root
                    ) if len(rgbir_paths) == 1 else None,
                    "rgb_path": display_path(
                        rgb_paths[0], project_root
                    ) if len(rgb_paths) == 1 else None,
                    "irrg_path": display_path(
                        irrg_paths[0], project_root
                    ) if len(irrg_paths) == 1 else None,
                    "rgb_to_rgbir_exact_matches": (
                        {str(k): v for k, v in rgb_to_rgbir_matches.items()}
                        if rgb_to_rgbir_matches is not None
                        else None
                    ),
                    "irrg_to_rgbir_exact_matches": (
                        {str(k): v for k, v in irrg_to_rgbir_matches.items()}
                        if irrg_to_rgbir_matches is not None
                        else None
                    ),
                    "rgb_to_rgbir": (
                        {str(k): v for k, v in rgb_to_rgbir.items()}
                        if rgb_to_rgbir is not None
                        else None
                    ),
                    "irrg_to_rgbir": (
                        {str(k): v for k, v in irrg_to_rgbir.items()}
                        if irrg_to_rgbir is not None
                        else None
                    ),
                    "rgb_vs_irrg_shared_rg_exact": rgb_vs_irrg_rg_exact,
                    "rgbir_semantics": semantics,
                    "diagnostic_sampled_rgb_to_rgbir_mae": rgb_mae,
                    "diagnostic_sampled_irrg_to_rgbir_mae": irrg_mae,
                    "errors": errors,
                }
            )

            del rgbir, rgb, irrg
            gc.collect()

        unique_orders = sorted(set(observed_orders))

        global_semantics: list[str] | None
        global_order_consistent = False

        if (
            len(unique_orders) == 1
            and len(observed_orders) == EXPECTED_TILE_COUNT
        ):
            global_semantics = list(unique_orders[0])
            global_order_consistent = True
        else:
            global_semantics = None

        invalid_tiles = [
            item["tile_id"] for item in tile_results if not item["valid"]
        ]
        exact_semantic_tiles = sum(
            1 for item in tile_results if item["valid"]
        )

        source_structure_ok = all(
            [
                len(rgbir_by_tile) == EXPECTED_TILE_COUNT,
                len(rgb_by_tile) == EXPECTED_TILE_COUNT,
                len(irrg_by_tile) == EXPECTED_TILE_COUNT,
                not missing_rgbir,
                not missing_rgb,
                not missing_irrg,
                not extra_rgbir,
                not extra_rgb,
                not extra_irrg,
                not duplicate_rgbir,
                not duplicate_rgb,
                not duplicate_irrg,
                not unexpected_rgbir,
                not unexpected_rgb,
                not unexpected_irrg,
            ]
        )

        overall_pass = all(
            [
                source_structure_ok,
                len(tile_results) == EXPECTED_TILE_COUNT,
                exact_semantic_tiles == EXPECTED_TILE_COUNT,
                global_order_consistent,
            ]
        )

        report = {
            "check_name": "Potsdam RGBIR band semantics",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "PASS" if overall_pass else "FAIL",
            "inputs": {
                "tile_split": display_path(split_path, project_root),
                "rgbir_root": display_path(rgbir_root, project_root),
                "rgb_root": display_path(rgb_root, project_root),
                "irrg_root": display_path(irrg_root, project_root),
            },
            "semantic_conventions": {
                "RGB": list(RGB_SEMANTICS),
                "IRRG": list(IRRG_SEMANTICS),
            },
            "summary": {
                "frozen_tile_count": len(frozen_ids),
                "rgbir_unique_tiles": len(rgbir_by_tile),
                "rgb_unique_tiles": len(rgb_by_tile),
                "irrg_unique_tiles": len(irrg_by_tile),
                "fully_checked_tiles": len(tile_results),
                "exact_semantic_tiles": exact_semantic_tiles,
                "invalid_tiles": len(invalid_tiles),
            },
            "missing_rgbir": missing_rgbir,
            "missing_rgb": missing_rgb,
            "missing_irrg": missing_irrg,
            "extra_rgbir": extra_rgbir,
            "extra_rgb": extra_rgb,
            "extra_irrg": extra_irrg,
            "duplicate_rgbir": duplicate_rgbir,
            "duplicate_rgb": duplicate_rgb,
            "duplicate_irrg": duplicate_irrg,
            "unexpected_rgbir_tiffs": [
                display_path(p, project_root) for p in unexpected_rgbir
            ],
            "unexpected_rgb_tiffs": [
                display_path(p, project_root) for p in unexpected_rgb
            ],
            "unexpected_irrg_tiffs": [
                display_path(p, project_root) for p in unexpected_irrg
            ],
            "observed_rgbir_orders": [list(x) for x in unique_orders],
            "global_order_consistent": global_order_consistent,
            "global_rgbir_semantics": global_semantics,
            "invalid_tile_ids": invalid_tiles,
            "tiles": tile_results,
        }

        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        json_path.write_text(payload, encoding="utf-8")

        text_report = build_text_report(report)
        txt_path.write_text(text_report, encoding="utf-8")

        print()
        print(text_report, end="")
        print(f"JSON report : {json_path}")
        print(f"Text report : {txt_path}")
        print(f"Console log : {console_path}")

        return 0 if overall_pass else 1

    except Exception as exc:
        fatal = {
            "check_name": "Potsdam RGBIR band semantics",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "FATAL",
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

        json_path.write_text(
            json.dumps(fatal, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        txt_path.write_text(
            "Potsdam RGBIR Band-Semantics Audit\n"
            "STATUS: FATAL\n\n"
            f"{fatal['fatal_error']}\n\n"
            f"{fatal['traceback']}\n",
            encoding="utf-8",
        )

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)
        print(fatal["fatal_error"])
        print()
        print(fatal["traceback"])

        return 2

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
