#!/usr/bin/env python3
"""
Diagnose the systematic Potsdam 3_Ortho_IRRG channel anomaly.

This is a forensic/read-only script. It does NOT modify any dataset file.

Background
----------
The previous RGBIR band audit found, for all 38 tiles:

    RGB raw channels 0,1,2  -> RGBIR raw channels 0,1,2
    IRRG raw channels 0,1,2 -> RGBIR raw channels 3,1,2

The official ISPRS Potsdam documentation describes:
    RGB   = R-G-B
    IRRG  = IR-R-G
    RGBIR = R-G-B-IR

Therefore the documented raw-channel mapping would be:
    RGB   -> RGBIR : [0, 1, 2]
    IRRG  -> RGBIR : [3, 0, 1]

This script determines whether the unexpected IRRG mapping is caused by:
  1) extraction corruption / wrong extracted files, or
  2) the actual 3_Ortho_IRRG.zip contents themselves.

Checks
------
* Uses the frozen 38 tile IDs from tile_split.json.
* Locates all three extracted ortho products.
* Locates each IRRG TIFF inside the original 3_Ortho_IRRG.zip.
* Runs ZIP CRC validation.
* Compares extracted IRRG bytes SHA256 with original ZIP-member bytes SHA256.
* Decodes ZIP-member and extracted IRRG with tifffile and requires pixel identity.
* Computes exact per-channel SHA256 mappings:
      RGB  -> RGBIR
      IRRG -> RGBIR
      IRRG -> RGB
* Records TIFF metadata/tags for a representative tile.
* Does not use approximate matching to establish a conclusion.

Outputs
-------
outputs/dataset_check/irrg_anomaly/
    console.txt
    irrg_anomaly_report.json
    irrg_anomaly_report.txt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


EXPECTED_TILE_COUNT = 38

RGBIR_RE = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_RGBIR\.tiff?$",
    re.IGNORECASE,
)
RGB_RE = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_RGB\.tiff?$",
    re.IGNORECASE,
)
IRRG_RE = re.compile(
    r"^top_potsdam_(\d+)_(\d+)_IRRG\.tiff?$",
    re.IGNORECASE,
)

# Based on the official ISPRS channel-composition description.
EXPECTED_RGB_TO_RGBIR = (0, 1, 2)
EXPECTED_IRRG_TO_RGBIR = (3, 0, 1)


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
        description="Forensically diagnose Potsdam IRRG channel-order anomaly."
    )
    parser.add_argument(
        "--split",
        default="data/processed/potsdam/tile_split.json",
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
        "--irrg-zip",
        default="data/raw/potsdam/Potsdam/3_Ortho_IRRG.zip",
    )
    parser.add_argument(
        "--representative-tile",
        default="2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dataset_check/irrg_anomaly",
    )

    args = parser.parse_args()
    args.project_root = project_root
    return args


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def tile_sort_key(tile_id: str) -> tuple[int, int]:
    a, b = tile_id.split("_")
    return int(a), int(b)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def channel_sha256(channel: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(channel).tobytes(order="C")
    ).hexdigest()


def channel_hashes(array: np.ndarray) -> list[str]:
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {array.shape}")
    return [channel_sha256(array[..., i]) for i in range(array.shape[2])]


def exact_unique_mapping(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[list[int] | None, dict[int, list[int]]]:
    src_hashes = channel_hashes(source)
    tgt_hashes = channel_hashes(target)

    candidates: dict[int, list[int]] = {}
    for src_idx, src_hash in enumerate(src_hashes):
        candidates[src_idx] = [
            tgt_idx
            for tgt_idx, tgt_hash in enumerate(tgt_hashes)
            if src_hash == tgt_hash
        ]

    if any(len(v) != 1 for v in candidates.values()):
        return None, candidates

    mapping = [candidates[i][0] for i in range(len(src_hashes))]
    if len(set(mapping)) != len(mapping):
        return None, candidates

    return mapping, candidates


def load_split_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    if split.get("status") != "PASS":
        raise ValueError(
            f"tile_split status is {split.get('status')!r}, expected 'PASS'"
        )

    groups = split.get("splits")
    if not isinstance(groups, dict):
        raise ValueError('tile_split has no "splits" dictionary')

    ids: list[str] = []
    for name in ("train", "val", "test"):
        values = groups.get(name)
        if not isinstance(values, list):
            raise ValueError(f"split {name!r} is not a list")
        ids.extend(values)

    if len(ids) != EXPECTED_TILE_COUNT or len(set(ids)) != EXPECTED_TILE_COUNT:
        raise ValueError(
            f"Frozen split does not contain exactly {EXPECTED_TILE_COUNT} unique tiles"
        )

    return sorted(ids, key=tile_sort_key)


def scan_product(
    root: Path,
    pattern: re.Pattern[str],
) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing product root: {root}")

    by_tile: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = pattern.fullmatch(path.name)
        if match is None:
            continue

        tile_id = f"{int(match.group(1))}_{int(match.group(2))}"
        by_tile[tile_id].append(path.resolve())

    return dict(by_tile)


def zip_irrg_members(zf: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    by_tile: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)

    for info in zf.infolist():
        if info.is_dir():
            continue

        name = Path(info.filename).name
        match = IRRG_RE.fullmatch(name)
        if match is None:
            continue

        tile_id = f"{int(match.group(1))}_{int(match.group(2))}"
        by_tile[tile_id].append(info)

    return dict(by_tile)


def enum_name(value: Any) -> str:
    if value is None:
        return "None"
    return getattr(value, "name", str(value))


def safe_tag_value(page: tifffile.TiffPage, name: str) -> Any:
    tag = page.tags.get(name)
    if tag is None:
        return None

    value = tag.value

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return [
            item.item() if isinstance(item, np.generic) else str(item)
            if hasattr(item, "name")
            else item
            for item in value
        ]

    if hasattr(value, "name"):
        return value.name

    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value

    return str(value)


def tiff_metadata_from_path(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        series = tif.series[0]

        return {
            "shape": list(series.shape),
            "dtype": str(series.dtype),
            "axes": str(series.axes),
            "photometric": enum_name(page.photometric),
            "planarconfig": enum_name(page.planarconfig),
            "compression": enum_name(page.compression),
            "samplesperpixel": int(page.samplesperpixel),
            "bitspersample": (
                int(page.bitspersample)
                if isinstance(page.bitspersample, (int, np.integer))
                else list(page.bitspersample)
                if page.bitspersample is not None
                else None
            ),
            "extrasamples": [enum_name(x) for x in page.extrasamples],
            "software": safe_tag_value(page, "Software"),
            "imagedescription": safe_tag_value(page, "ImageDescription"),
            "documentname": safe_tag_value(page, "DocumentName"),
        }


def tiff_metadata_from_bytes(data: bytes) -> dict[str, Any]:
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        page = tif.pages[0]
        series = tif.series[0]

        return {
            "shape": list(series.shape),
            "dtype": str(series.dtype),
            "axes": str(series.axes),
            "photometric": enum_name(page.photometric),
            "planarconfig": enum_name(page.planarconfig),
            "compression": enum_name(page.compression),
            "samplesperpixel": int(page.samplesperpixel),
            "bitspersample": (
                int(page.bitspersample)
                if isinstance(page.bitspersample, (int, np.integer))
                else list(page.bitspersample)
                if page.bitspersample is not None
                else None
            ),
            "extrasamples": [enum_name(x) for x in page.extrasamples],
            "software": safe_tag_value(page, "Software"),
            "imagedescription": safe_tag_value(page, "ImageDescription"),
            "documentname": safe_tag_value(page, "DocumentName"),
        }


def mapping_text(mapping: list[int] | None) -> str:
    if mapping is None:
        return "UNRESOLVED"
    return "[" + ", ".join(f"ch{i}->ch{dst}" for i, dst in enumerate(mapping)) + "]"


def build_text_report(report: dict[str, Any]) -> str:
    s = report["summary"]

    lines = [
        "=" * 80,
        "Potsdam 3_Ortho_IRRG Channel-Anomaly Diagnostic",
        "=" * 80,
        "",
        f"Diagnostic status: {report['diagnostic_status']}",
        "",
        "[Official documented compositions used as expectation]",
        "RGB   : R-G-B",
        "IRRG  : IR-R-G",
        "RGBIR : R-G-B-IR",
        "",
        "[Summary]",
        f"Frozen tiles                         : {s['frozen_tiles']}",
        f"Tiles fully diagnosed                : {s['tiles_diagnosed']}",
        f"ZIP CRC check passed                 : {s['zip_crc_passed']}",
        f"Extracted IRRG == ZIP member bytes   : {s['zip_byte_identical_tiles']}/{s['frozen_tiles']}",
        f"Extracted IRRG == ZIP decoded pixels : {s['zip_pixel_identical_tiles']}/{s['frozen_tiles']}",
        f"RGB -> RGBIR documented mapping pass : {s['rgb_expected_mapping_tiles']}/{s['frozen_tiles']}",
        f"IRRG -> RGBIR documented mapping pass: {s['irrg_expected_mapping_tiles']}/{s['frozen_tiles']}",
        f"Observed IRRG mapping consistent     : {s['irrg_observed_mapping_consistent']}",
        "",
        "[Expected raw channel mappings]",
        f"RGB -> RGBIR  : {list(EXPECTED_RGB_TO_RGBIR)}",
        f"IRRG -> RGBIR : {list(EXPECTED_IRRG_TO_RGBIR)}",
        "",
        "[Observed global mappings]",
        f"RGB -> RGBIR  : {report['observed_global_rgb_to_rgbir']}",
        f"IRRG -> RGBIR : {report['observed_global_irrg_to_rgbir']}",
        f"IRRG -> RGB   : {report['observed_global_irrg_to_rgb']}",
        "",
        "[Interpretation]",
        report["interpretation"],
        "",
        "[Representative TIFF metadata]",
        f"Tile: {report['representative']['tile_id']}",
        "RGB:",
        json.dumps(report["representative"]["rgb_metadata"], indent=2, ensure_ascii=False),
        "IRRG extracted:",
        json.dumps(report["representative"]["irrg_extracted_metadata"], indent=2, ensure_ascii=False),
        "IRRG ZIP member:",
        json.dumps(report["representative"]["irrg_zip_metadata"], indent=2, ensure_ascii=False),
        "RGBIR:",
        json.dumps(report["representative"]["rgbir_metadata"], indent=2, ensure_ascii=False),
        "",
        "[Per-tile raw mappings]",
    ]

    for item in report["tiles"]:
        lines.append(
            f"{item['tile_id']}: "
            f"RGB->RGBIR={item['rgb_to_rgbir']} | "
            f"IRRG->RGBIR={item['irrg_to_rgbir']} | "
            f"IRRG->RGB={item['irrg_to_rgb']} | "
            f"ZIP-bytes={item['irrg_zip_bytes_identical']} | "
            f"ZIP-pixels={item['irrg_zip_pixels_identical']}"
        )
        for error in item["errors"]:
            lines.append(f"  ERROR: {error}")

    lines.extend(
        [
            "",
            "=" * 80,
            f"FINAL DIAGNOSTIC STATUS: {report['diagnostic_status']}",
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
    irrg_zip_path = resolve_project_path(project_root, args.irrg_zip)
    output_dir = resolve_project_path(project_root, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    console_path = output_dir / "console.txt"
    json_path = output_dir / "irrg_anomaly_report.json"
    txt_path = output_dir / "irrg_anomaly_report.txt"

    original_stdout = sys.stdout
    tee = Tee(original_stdout, console_path)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("Potsdam 3_Ortho_IRRG Channel-Anomaly Diagnostic")
        print("=" * 80)
        print(f"Frozen split : {split_path}")
        print(f"RGB root     : {rgb_root}")
        print(f"IRRG root    : {irrg_root}")
        print(f"RGBIR root   : {rgbir_root}")
        print(f"IRRG ZIP     : {irrg_zip_path}")
        print()

        if not split_path.is_file():
            raise FileNotFoundError(f"Missing frozen split: {split_path}")
        if not irrg_zip_path.is_file():
            raise FileNotFoundError(f"Missing original IRRG ZIP: {irrg_zip_path}")

        tile_ids = load_split_ids(split_path)
        tile_set = set(tile_ids)

        rgb_by_tile = scan_product(rgb_root, RGB_RE)
        irrg_by_tile = scan_product(irrg_root, IRRG_RE)
        rgbir_by_tile = scan_product(rgbir_root, RGBIR_RE)

        for name, mapping in (
            ("RGB", rgb_by_tile),
            ("IRRG", irrg_by_tile),
            ("RGBIR", rgbir_by_tile),
        ):
            missing = sorted(tile_set - set(mapping), key=tile_sort_key)
            extra = sorted(set(mapping) - tile_set, key=tile_sort_key)
            duplicates = sorted(
                [tile_id for tile_id, paths in mapping.items() if len(paths) != 1],
                key=tile_sort_key,
            )

            if missing or extra or duplicates:
                raise RuntimeError(
                    f"{name} source-set problem: "
                    f"missing={missing}, extra={extra}, duplicates={duplicates}"
                )

        with zipfile.ZipFile(irrg_zip_path, "r") as zf:
            bad_member = zf.testzip()
            zip_crc_passed = bad_member is None

            zip_by_tile = zip_irrg_members(zf)

            missing_zip = sorted(tile_set - set(zip_by_tile), key=tile_sort_key)
            extra_zip = sorted(set(zip_by_tile) - tile_set, key=tile_sort_key)
            duplicate_zip = sorted(
                [tile_id for tile_id, infos in zip_by_tile.items() if len(infos) != 1],
                key=tile_sort_key,
            )

            if missing_zip or extra_zip or duplicate_zip:
                raise RuntimeError(
                    "IRRG ZIP member-set problem: "
                    f"missing={missing_zip}, extra={extra_zip}, "
                    f"duplicates={duplicate_zip}"
                )

            tile_results: list[dict[str, Any]] = []
            rgb_maps: list[tuple[int, ...]] = []
            irrg_maps: list[tuple[int, ...]] = []
            irrg_to_rgb_maps: list[tuple[int, ...]] = []

            zip_byte_identical_count = 0
            zip_pixel_identical_count = 0
            rgb_expected_count = 0
            irrg_expected_count = 0

            representative_data: dict[str, Any] | None = None

            for index, tile_id in enumerate(tile_ids, start=1):
                print(f"[{index:02d}/{len(tile_ids):02d}] {tile_id}")

                errors: list[str] = []

                rgb_path = rgb_by_tile[tile_id][0]
                irrg_path = irrg_by_tile[tile_id][0]
                rgbir_path = rgbir_by_tile[tile_id][0]
                zip_info = zip_by_tile[tile_id][0]

                zip_bytes = zf.read(zip_info)
                extracted_sha = sha256_file(irrg_path)
                zip_sha = sha256_bytes(zip_bytes)

                zip_bytes_identical = extracted_sha == zip_sha
                if zip_bytes_identical:
                    zip_byte_identical_count += 1
                else:
                    errors.append(
                        "Extracted IRRG file bytes differ from original ZIP member"
                    )

                rgb = tifffile.imread(rgb_path)
                irrg = tifffile.imread(irrg_path)
                rgbir = tifffile.imread(rgbir_path)
                irrg_from_zip = tifffile.imread(io.BytesIO(zip_bytes))

                zip_pixels_identical = np.array_equal(irrg, irrg_from_zip)
                if zip_pixels_identical:
                    zip_pixel_identical_count += 1
                else:
                    errors.append(
                        "Decoded extracted IRRG pixels differ from decoded ZIP member"
                    )

                rgb_map, rgb_candidates = exact_unique_mapping(rgb, rgbir)
                irrg_map, irrg_candidates = exact_unique_mapping(irrg, rgbir)
                irrg_to_rgb, irrg_to_rgb_candidates = exact_unique_mapping(irrg, rgb)

                if rgb_map is None:
                    errors.append(
                        f"RGB->RGBIR exact mapping unresolved: {rgb_candidates}"
                    )
                else:
                    rgb_maps.append(tuple(rgb_map))
                    if tuple(rgb_map) == EXPECTED_RGB_TO_RGBIR:
                        rgb_expected_count += 1

                if irrg_map is None:
                    errors.append(
                        f"IRRG->RGBIR exact mapping unresolved: {irrg_candidates}"
                    )
                else:
                    irrg_maps.append(tuple(irrg_map))
                    if tuple(irrg_map) == EXPECTED_IRRG_TO_RGBIR:
                        irrg_expected_count += 1

                if irrg_to_rgb is not None:
                    irrg_to_rgb_maps.append(tuple(irrg_to_rgb))

                if tile_id == args.representative_tile:
                    representative_data = {
                        "tile_id": tile_id,
                        "rgb_path": display_path(rgb_path, project_root),
                        "irrg_path": display_path(irrg_path, project_root),
                        "rgbir_path": display_path(rgbir_path, project_root),
                        "irrg_zip_member": zip_info.filename,
                        "irrg_extracted_sha256": extracted_sha,
                        "irrg_zip_member_sha256": zip_sha,
                        "rgb_metadata": tiff_metadata_from_path(rgb_path),
                        "irrg_extracted_metadata": tiff_metadata_from_path(irrg_path),
                        "irrg_zip_metadata": tiff_metadata_from_bytes(zip_bytes),
                        "rgbir_metadata": tiff_metadata_from_path(rgbir_path),
                    }

                tile_results.append(
                    {
                        "tile_id": tile_id,
                        "rgb_to_rgbir": rgb_map,
                        "irrg_to_rgbir": irrg_map,
                        "irrg_to_rgb": irrg_to_rgb,
                        "irrg_zip_member": zip_info.filename,
                        "irrg_extracted_sha256": extracted_sha,
                        "irrg_zip_member_sha256": zip_sha,
                        "irrg_zip_bytes_identical": zip_bytes_identical,
                        "irrg_zip_pixels_identical": zip_pixels_identical,
                        "errors": errors,
                    }
                )

                del rgb, irrg, rgbir, irrg_from_zip, zip_bytes

        if representative_data is None:
            raise ValueError(
                f"Representative tile {args.representative_tile!r} was not found"
            )

        unique_rgb_maps = sorted(set(rgb_maps))
        unique_irrg_maps = sorted(set(irrg_maps))
        unique_irrg_to_rgb_maps = sorted(set(irrg_to_rgb_maps))

        observed_global_rgb = (
            list(unique_rgb_maps[0])
            if len(unique_rgb_maps) == 1 and len(rgb_maps) == EXPECTED_TILE_COUNT
            else None
        )
        observed_global_irrg = (
            list(unique_irrg_maps[0])
            if len(unique_irrg_maps) == 1 and len(irrg_maps) == EXPECTED_TILE_COUNT
            else None
        )
        observed_global_irrg_to_rgb = (
            list(unique_irrg_to_rgb_maps[0])
            if len(unique_irrg_to_rgb_maps) == 1
            and len(irrg_to_rgb_maps) == EXPECTED_TILE_COUNT
            else None
        )

        irrg_observed_consistent = observed_global_irrg is not None

        extraction_proven = all(
            item["irrg_zip_bytes_identical"]
            and item["irrg_zip_pixels_identical"]
            for item in tile_results
        )

        rgb_documented_mapping_proven = (
            observed_global_rgb == list(EXPECTED_RGB_TO_RGBIR)
        )

        irrg_documented_mapping_proven = (
            observed_global_irrg == list(EXPECTED_IRRG_TO_RGBIR)
        )

        if (
            zip_crc_passed
            and extraction_proven
            and rgb_documented_mapping_proven
            and irrg_observed_consistent
            and not irrg_documented_mapping_proven
        ):
            diagnostic_status = "SOURCE_ARCHIVE_CHANNEL_DISCREPANCY_CONFIRMED"
            interpretation = (
                "The unexpected IRRG channel mapping is present in the original "
                "3_Ortho_IRRG.zip itself, not introduced by extraction. The RGB "
                "product exactly follows the documented RGB->RGBIR raw-channel "
                "relationship, while the IRRG ZIP consistently does not follow "
                "the documented IR-R-G->R-G-B-IR mapping. Do not use the IRRG "
                "archive to redefine the already evidenced RGBIR channel order."
            )
        elif not extraction_proven:
            diagnostic_status = "EXTRACTION_OR_FILE_MISMATCH"
            interpretation = (
                "At least one extracted IRRG TIFF does not match its original ZIP "
                "member. Resolve the extraction/file mismatch before interpreting "
                "IRRG channel semantics."
            )
        elif not rgb_documented_mapping_proven:
            diagnostic_status = "RGB_REFERENCE_DISCREPANCY"
            interpretation = (
                "The RGB product does not consistently map to RGBIR channels "
                "0,1,2 as expected. More investigation is required before freezing "
                "RGBIR semantics."
            )
        elif irrg_documented_mapping_proven:
            diagnostic_status = "NO_IRRG_DISCREPANCY"
            interpretation = (
                "The IRRG product follows the documented raw-channel mapping."
            )
        else:
            diagnostic_status = "UNRESOLVED"
            interpretation = (
                "The available exact-match evidence is not sufficient to isolate "
                "the discrepancy to a single cause."
            )

        report = {
            "check_name": "Potsdam 3_Ortho_IRRG channel anomaly diagnosis",
            "generated_at": datetime.now().astimezone().isoformat(),
            "diagnostic_status": diagnostic_status,
            "official_expected_compositions": {
                "RGB": ["R", "G", "B"],
                "IRRG": ["IR", "R", "G"],
                "RGBIR": ["R", "G", "B", "IR"],
            },
            "expected_raw_mappings": {
                "RGB_to_RGBIR": list(EXPECTED_RGB_TO_RGBIR),
                "IRRG_to_RGBIR": list(EXPECTED_IRRG_TO_RGBIR),
            },
            "summary": {
                "frozen_tiles": len(tile_ids),
                "tiles_diagnosed": len(tile_results),
                "zip_crc_passed": zip_crc_passed,
                "zip_byte_identical_tiles": zip_byte_identical_count,
                "zip_pixel_identical_tiles": zip_pixel_identical_count,
                "rgb_expected_mapping_tiles": rgb_expected_count,
                "irrg_expected_mapping_tiles": irrg_expected_count,
                "irrg_observed_mapping_consistent": irrg_observed_consistent,
            },
            "observed_global_rgb_to_rgbir": observed_global_rgb,
            "observed_global_irrg_to_rgbir": observed_global_irrg,
            "observed_global_irrg_to_rgb": observed_global_irrg_to_rgb,
            "observed_unique_rgb_to_rgbir_mappings": [
                list(x) for x in unique_rgb_maps
            ],
            "observed_unique_irrg_to_rgbir_mappings": [
                list(x) for x in unique_irrg_maps
            ],
            "observed_unique_irrg_to_rgb_mappings": [
                list(x) for x in unique_irrg_to_rgb_maps
            ],
            "interpretation": interpretation,
            "representative": representative_data,
            "tiles": tile_results,
        }

        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        text = build_text_report(report)
        txt_path.write_text(text, encoding="utf-8")

        print()
        print(text, end="")
        print(f"JSON report : {json_path}")
        print(f"Text report : {txt_path}")
        print(f"Console log : {console_path}")

        # Exit 0 means the diagnostic itself completed and produced a resolved
        # forensic conclusion. It does NOT mean the documented IRRG mapping passed.
        resolved_statuses = {
            "SOURCE_ARCHIVE_CHANNEL_DISCREPANCY_CONFIRMED",
            "NO_IRRG_DISCREPANCY",
        }
        return 0 if diagnostic_status in resolved_statuses else 1

    except Exception as exc:
        fatal = {
            "check_name": "Potsdam 3_Ortho_IRRG channel anomaly diagnosis",
            "generated_at": datetime.now().astimezone().isoformat(),
            "diagnostic_status": "FATAL",
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

        json_path.write_text(
            json.dumps(fatal, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        txt_path.write_text(
            "Potsdam 3_Ortho_IRRG Channel-Anomaly Diagnostic\n"
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
