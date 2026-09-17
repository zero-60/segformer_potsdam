#!/usr/bin/env python3
"""
Freeze the RGBIR channel semantics for the ISPRS Potsdam experiment.

This script does NOT inspect or modify source TIFF pixels. It consumes the
already-completed forensic evidence and writes a single frozen semantics file
for all downstream Dataset/model code.

Required evidence
-----------------
1. data/processed/potsdam/tile_split.json
   - status == PASS
   - 38 unique supervised tiles

2. outputs/dataset_check/irrg_anomaly/irrg_anomaly_report.json
   - diagnostic_status == SOURCE_ARCHIVE_CHANNEL_DISCREPANCY_CONFIRMED
   - 38/38 extracted IRRG files byte-identical to ZIP members
   - 38/38 decoded IRRG pixels identical to ZIP members
   - RGB -> RGBIR documented raw mapping passes 38/38
   - observed global RGB -> RGBIR mapping == [0, 1, 2]
   - observed global IRRG -> RGBIR mapping == [3, 1, 2]
   - documented IRRG mapping passes 0/38
   - observed IRRG discrepancy is globally consistent

Frozen experiment semantics
---------------------------
RGBIR raw channel 0 = R
RGBIR raw channel 1 = G
RGBIR raw channel 2 = B
RGBIR raw channel 3 = NIR

Downstream input conventions
----------------------------
A  RGB baseline:                [0, 1, 2]
B  RGB+NIR 4-channel:          [0, 1, 2, 3]
C  RGB branch:                 [0, 1, 2]
C  NIR branch:                 [3]

The anomalous 3_Ortho_IRRG product is retained only as forensic evidence and
must not be used to redefine RGBIR channel semantics.

Outputs
-------
data/processed/potsdam/rgbir_semantics.json

outputs/dataset_check/rgbir_semantics/
    console.txt
    rgbir_semantics.txt
    rgbir_semantics.json

By default the script refuses to overwrite an existing frozen semantics file.
Use --overwrite only when intentionally rebuilding project metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_TILE_COUNT = 38
EXPECTED_DIAGNOSTIC_STATUS = "SOURCE_ARCHIVE_CHANNEL_DISCREPANCY_CONFIRMED"
EXPECTED_RGB_MAPPING = [0, 1, 2]
EXPECTED_IRRG_OBSERVED_MAPPING = [3, 1, 2]


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
        description="Freeze verified Potsdam RGBIR channel semantics."
    )
    parser.add_argument(
        "--split",
        default="data/processed/potsdam/tile_split.json",
    )
    parser.add_argument(
        "--irrg-report",
        default="outputs/dataset_check/irrg_anomaly/irrg_anomaly_report.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/potsdam/rgbir_semantics.json",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/dataset_check/rgbir_semantics",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow intentional replacement of an existing frozen semantics file.",
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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split(split: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if split.get("status") != "PASS":
        errors.append(
            f"tile split status is {split.get('status')!r}, expected 'PASS'"
        )

    groups = split.get("splits")
    if not isinstance(groups, dict):
        errors.append('tile split has no "splits" dictionary')
        return errors

    ids: list[str] = []
    for name in ("train", "val", "test"):
        values = groups.get(name)
        if not isinstance(values, list):
            errors.append(f"split {name!r} is not a list")
            continue
        ids.extend(values)

    if len(ids) != EXPECTED_TILE_COUNT:
        errors.append(
            f"split contains {len(ids)} tile references, "
            f"expected {EXPECTED_TILE_COUNT}"
        )

    if len(set(ids)) != EXPECTED_TILE_COUNT:
        errors.append(
            f"split contains {len(set(ids))} unique tile IDs, "
            f"expected {EXPECTED_TILE_COUNT}"
        )

    counts = split.get("counts")
    if isinstance(counts, dict):
        if counts.get("total") != EXPECTED_TILE_COUNT:
            errors.append(
                f"split counts.total={counts.get('total')!r}, "
                f"expected {EXPECTED_TILE_COUNT}"
            )
    else:
        errors.append('tile split has no "counts" dictionary')

    return errors


def validate_irrg_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if report.get("diagnostic_status") != EXPECTED_DIAGNOSTIC_STATUS:
        errors.append(
            "IRRG diagnostic status is "
            f"{report.get('diagnostic_status')!r}, expected "
            f"{EXPECTED_DIAGNOSTIC_STATUS!r}"
        )

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append('IRRG report has no "summary" dictionary')
        return errors

    expected_summary = {
        "frozen_tiles": EXPECTED_TILE_COUNT,
        "tiles_diagnosed": EXPECTED_TILE_COUNT,
        "zip_crc_passed": True,
        "zip_byte_identical_tiles": EXPECTED_TILE_COUNT,
        "zip_pixel_identical_tiles": EXPECTED_TILE_COUNT,
        "rgb_expected_mapping_tiles": EXPECTED_TILE_COUNT,
        "irrg_expected_mapping_tiles": 0,
        "irrg_observed_mapping_consistent": True,
    }

    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if actual != expected:
            errors.append(
                f"IRRG summary[{key!r}]={actual!r}, expected {expected!r}"
            )

    if report.get("observed_global_rgb_to_rgbir") != EXPECTED_RGB_MAPPING:
        errors.append(
            "observed_global_rgb_to_rgbir="
            f"{report.get('observed_global_rgb_to_rgbir')!r}, "
            f"expected {EXPECTED_RGB_MAPPING!r}"
        )

    if (
        report.get("observed_global_irrg_to_rgbir")
        != EXPECTED_IRRG_OBSERVED_MAPPING
    ):
        errors.append(
            "observed_global_irrg_to_rgbir="
            f"{report.get('observed_global_irrg_to_rgbir')!r}, "
            f"expected {EXPECTED_IRRG_OBSERVED_MAPPING!r}"
        )

    return errors


def build_text_report(metadata: dict[str, Any]) -> str:
    channels = metadata["rgbir_channels"]

    lines = [
        "=" * 80,
        "Potsdam RGBIR Frozen Channel Semantics",
        "=" * 80,
        "",
        f"Status: {metadata['status']}",
        "",
        "[Frozen raw channel order]",
    ]

    for item in channels:
        lines.append(
            f"RGBIR channel {item['index']}: {item['name']}"
        )

    lines.extend(
        [
            "",
            "[Downstream experiment inputs]",
            (
                "A RGB baseline             : "
                f"{metadata['experiment_inputs']['A_rgb']['channel_indices']}"
            ),
            (
                "B RGB+NIR                  : "
                f"{metadata['experiment_inputs']['B_rgb_nir']['channel_indices']}"
            ),
            (
                "C/C-noGate RGB branch      : "
                f"{metadata['experiment_inputs']['C_rgb_branch']['channel_indices']}"
            ),
            (
                "C/C-noGate NIR branch      : "
                f"{metadata['experiment_inputs']['C_nir_branch']['channel_indices']}"
            ),
            "",
            "[Evidence]",
            (
                "RGB -> RGBIR exact mapping : "
                f"{metadata['evidence']['rgb_to_rgbir_mapping']}"
            ),
            (
                "IRRG anomaly mapping       : "
                f"{metadata['evidence']['irrg_observed_mapping']}"
            ),
            (
                "IRRG archive discrepancy   : "
                f"{metadata['evidence']['irrg_source_archive_discrepancy_confirmed']}"
            ),
            (
                "IRRG ZIP byte identity     : "
                f"{metadata['evidence']['irrg_zip_byte_identical_tiles']}/38"
            ),
            (
                "IRRG ZIP pixel identity    : "
                f"{metadata['evidence']['irrg_zip_pixel_identical_tiles']}/38"
            ),
            "",
            "[Policy]",
            metadata["policy"],
            "",
            "=" * 80,
            "FINAL STATUS: PASS",
            "Frozen RGBIR order = [R, G, B, NIR]",
            "=" * 80,
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    split_path = resolve_project_path(project_root, args.split)
    irrg_report_path = resolve_project_path(project_root, args.irrg_report)
    output_path = resolve_project_path(project_root, args.output)
    report_dir = resolve_project_path(project_root, args.report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console_path = report_dir / "console.txt"
    txt_path = report_dir / "rgbir_semantics.txt"
    json_report_path = report_dir / "rgbir_semantics.json"

    if output_path.exists() and not args.overwrite:
        print(
            f"Refusing to overwrite existing frozen semantics: {output_path}\n"
            "If replacement is intentional, rerun with --overwrite."
        )
        return 2

    original_stdout = sys.stdout
    tee = Tee(original_stdout, console_path)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("Freeze Potsdam RGBIR Channel Semantics")
        print("=" * 80)
        print(f"Tile split       : {split_path}")
        print(f"IRRG diagnostic : {irrg_report_path}")
        print(f"Frozen output    : {output_path}")
        print()

        if not split_path.is_file():
            raise FileNotFoundError(f"Missing frozen tile split: {split_path}")

        if not irrg_report_path.is_file():
            raise FileNotFoundError(
                f"Missing IRRG anomaly diagnostic JSON: {irrg_report_path}"
            )

        split = load_json(split_path)
        irrg_report = load_json(irrg_report_path)

        if not isinstance(split, dict):
            raise TypeError("tile_split.json top level must be a dictionary")
        if not isinstance(irrg_report, dict):
            raise TypeError("IRRG anomaly report top level must be a dictionary")

        errors = validate_split(split)
        errors.extend(validate_irrg_report(irrg_report))

        if errors:
            print("Evidence validation FAILED:")
            for error in errors:
                print(f"  - {error}")
            print()
            print("Frozen semantics file will NOT be created.")
            return 1

        metadata = {
            "schema_version": 1,
            "dataset": "ISPRS Potsdam",
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "PASS",
            "purpose": (
                "Frozen raw-channel semantics for the 4_Ortho_RGBIR source used "
                "by all RGB/NIR robustness experiments."
            ),
            "source_product": "4_Ortho_RGBIR",
            "raw_shape": [6000, 6000, 4],
            "raw_dtype": "uint8",
            "rgbir_channels": [
                {"index": 0, "name": "R"},
                {"index": 1, "name": "G"},
                {"index": 2, "name": "B"},
                {"index": 3, "name": "NIR"},
            ],
            "channel_names": ["R", "G", "B", "NIR"],
            "rgb_channel_indices": [0, 1, 2],
            "nir_channel_index": 3,
            "experiment_inputs": {
                "A_rgb": {
                    "channel_indices": [0, 1, 2],
                    "channel_names": ["R", "G", "B"],
                },
                "B_rgb_nir": {
                    "channel_indices": [0, 1, 2, 3],
                    "channel_names": ["R", "G", "B", "NIR"],
                },
                "C_rgb_branch": {
                    "channel_indices": [0, 1, 2],
                    "channel_names": ["R", "G", "B"],
                },
                "C_nir_branch": {
                    "channel_indices": [3],
                    "channel_names": ["NIR"],
                },
            },
            "evidence": {
                "tile_split": display_path(split_path, project_root),
                "tile_split_sha256": sha256_file(split_path),
                "irrg_anomaly_report": display_path(
                    irrg_report_path, project_root
                ),
                "irrg_anomaly_report_sha256": sha256_file(irrg_report_path),
                "rgb_to_rgbir_mapping": [0, 1, 2],
                "rgb_to_rgbir_exact_tiles": 38,
                "irrg_documented_mapping_expected": [3, 0, 1],
                "irrg_observed_mapping": [3, 1, 2],
                "irrg_documented_mapping_tiles": 0,
                "irrg_source_archive_discrepancy_confirmed": True,
                "irrg_zip_crc_passed": True,
                "irrg_zip_byte_identical_tiles": 38,
                "irrg_zip_pixel_identical_tiles": 38,
            },
            "policy": (
                "Use 4_Ortho_RGBIR as the sole image source for the robustness "
                "experiments. Interpret raw channels as [R, G, B, NIR]. "
                "The 3_Ortho_IRRG archive is retained only as forensic evidence "
                "because its raw channel relationship is systematically "
                "inconsistent with the documented IR-R-G composition; it must "
                "not be used to redefine RGBIR semantics or as an alternate "
                "training image source."
            ),
        }

        payload = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        text = build_text_report(metadata)

        output_path.write_text(payload, encoding="utf-8")
        json_report_path.write_text(payload, encoding="utf-8")
        txt_path.write_text(text, encoding="utf-8")

        print(text, end="")
        print(f"Frozen semantics : {output_path}")
        print(f"JSON report      : {json_report_path}")
        print(f"Text report      : {txt_path}")
        print(f"Console log      : {console_path}")

        return 0

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
