#!/usr/bin/env python3
"""
Build and freeze a tile-level train/val/test split for the ISPRS Potsdam
SegFormer RGB/NIR robustness experiments.

Design
------
1. Canonical GT is read ONLY from:
       data/processed/potsdam/labels_manifest.json

2. The pairing gate must already have passed:
       outputs/dataset_check/canonical_pairs/canonical_pairs.json

3. Test split:
       all 14 tiles whose canonical GT source is "5_Labels_all"

   This preserves the already-frozen provenance boundary:
       - 24 participant-reference tiles form the development pool
       - 14 all-only tiles are held out for final test

4. Validation split:
       exactly 6 tiles are selected from the 24 participant-reference tiles
       by exhaustive deterministic search over C(24, 6) candidates.

   Selection criterion:
       minimize class-distribution divergence from the full 24-tile
       development pool using only manifest pixel counts.

   No model outputs, RGB/NIR image content, or future test metrics are used.

5. Training split:
       the remaining 18 participant-reference tiles.

6. No patches are created. No model is trained. No data files are modified.

Outputs
-------
data/processed/potsdam/tile_split.json

outputs/dataset_check/tile_split/
    console.txt
    tile_split.txt
    tile_split.json

The processed split file is treated as frozen experiment metadata. By default,
the script refuses to overwrite an existing split. Use --overwrite only when
you intentionally want to regenerate it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_TOTAL = 38
EXPECTED_DEV = 24
EXPECTED_TEST = 14
DEFAULT_VAL_SIZE = 6
EXPECTED_TRAIN = EXPECTED_DEV - DEFAULT_VAL_SIZE

PARTICIPANT_SOURCE = "5_Labels_for_participants"
ALL_SOURCE = "5_Labels_all"


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
        description="Build deterministic tile-level Potsdam train/val/test split."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/potsdam/labels_manifest.json",
    )
    parser.add_argument(
        "--pairs-report",
        default="outputs/dataset_check/canonical_pairs/canonical_pairs.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/potsdam/tile_split.json",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/dataset_check/tile_split",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing frozen split file.",
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


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tile_sort_key(tile_id: str) -> tuple[int, int]:
    a, b = tile_id.split("_")
    return int(a), int(b)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def proportions(counts: dict[str, int], classes: list[str]) -> dict[str, float]:
    total = sum(int(counts[c]) for c in classes)
    if total <= 0:
        raise ValueError("Class-pixel total must be positive.")
    return {c: int(counts[c]) / total for c in classes}


def aggregate_counts(
    tile_ids: list[str] | tuple[str, ...],
    tile_by_id: dict[str, dict[str, Any]],
    classes: list[str],
) -> dict[str, int]:
    out = {c: 0 for c in classes}
    for tile_id in tile_ids:
        tile = tile_by_id[tile_id]
        counts = tile.get("standard_color_counts")
        if not isinstance(counts, dict):
            raise ValueError(
                f"{tile_id}: missing standard_color_counts in canonical manifest"
            )
        for c in classes:
            if c not in counts:
                raise ValueError(
                    f"{tile_id}: class {c!r} missing from standard_color_counts"
                )
            out[c] += int(counts[c])
    return out


def distribution_error(
    candidate_ids: tuple[str, ...],
    train_ids: tuple[str, ...],
    tile_by_id: dict[str, dict[str, Any]],
    classes: list[str],
    reference_props: dict[str, float],
) -> tuple[float, float, float, float]:
    """
    Deterministic objective.

    Primary:
        minimize the maximum absolute class-proportion deviation for validation.
    Secondary:
        minimize mean absolute deviation for validation.
    Tertiary/quaternary:
        same two measures for training.

    All splits contain equal-size 6000x6000 tiles, but the objective is computed
    from actual canonical GT pixel counts rather than assuming class balance.
    """
    val_counts = aggregate_counts(candidate_ids, tile_by_id, classes)
    train_counts = aggregate_counts(train_ids, tile_by_id, classes)

    val_props = proportions(val_counts, classes)
    train_props = proportions(train_counts, classes)

    val_diffs = [abs(val_props[c] - reference_props[c]) for c in classes]
    train_diffs = [abs(train_props[c] - reference_props[c]) for c in classes]

    return (
        max(val_diffs),
        sum(val_diffs) / len(val_diffs),
        max(train_diffs),
        sum(train_diffs) / len(train_diffs),
    )


def choose_validation_tiles(
    dev_ids: list[str],
    val_size: int,
    tile_by_id: dict[str, dict[str, Any]],
    classes: list[str],
) -> tuple[list[str], dict[str, float], int]:
    if not (1 <= val_size < len(dev_ids)):
        raise ValueError(
            f"val_size must be between 1 and {len(dev_ids) - 1}, got {val_size}"
        )

    dev_ids = sorted(dev_ids, key=tile_sort_key)
    dev_counts = aggregate_counts(dev_ids, tile_by_id, classes)
    reference_props = proportions(dev_counts, classes)

    best_combo: tuple[str, ...] | None = None
    best_objective: tuple[Any, ...] | None = None
    candidate_count = 0

    dev_set = set(dev_ids)

    for combo in itertools.combinations(dev_ids, val_size):
        candidate_count += 1
        combo_set = set(combo)
        train_ids = tuple(
            tile_id for tile_id in dev_ids if tile_id not in combo_set
        )

        # Require every canonical class to occur in both train and val.
        val_counts = aggregate_counts(combo, tile_by_id, classes)
        train_counts = aggregate_counts(train_ids, tile_by_id, classes)

        if any(val_counts[c] <= 0 for c in classes):
            continue
        if any(train_counts[c] <= 0 for c in classes):
            continue

        numeric_error = distribution_error(
            combo,
            train_ids,
            tile_by_id,
            classes,
            reference_props,
        )

        # Lexicographic tile IDs are only the final deterministic tie-breaker.
        objective = numeric_error + (tuple(tile_sort_key(t) for t in combo),)

        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_combo = combo

    if best_combo is None:
        raise RuntimeError(
            "No valid validation subset satisfies the class-presence constraints."
        )

    assert set(best_combo).issubset(dev_set)

    objective_values = {
        "val_max_abs_class_fraction_error": float(best_objective[0]),
        "val_mean_abs_class_fraction_error": float(best_objective[1]),
        "train_max_abs_class_fraction_error": float(best_objective[2]),
        "train_mean_abs_class_fraction_error": float(best_objective[3]),
    }

    return list(best_combo), objective_values, candidate_count


def build_stats(
    ids: list[str],
    tile_by_id: dict[str, dict[str, Any]],
    classes: list[str],
) -> dict[str, Any]:
    counts = aggregate_counts(ids, tile_by_id, classes)
    props = proportions(counts, classes)
    return {
        "tile_count": len(ids),
        "pixel_count": int(sum(counts.values())),
        "class_pixel_counts": {c: int(counts[c]) for c in classes},
        "class_fractions": {c: float(props[c]) for c in classes},
    }


def validate_pairing_gate(
    pair_report: dict[str, Any],
    manifest_tile_ids: set[str],
) -> list[str]:
    errors: list[str] = []

    if pair_report.get("status") != "PASS":
        errors.append(
            f"canonical pair report status is {pair_report.get('status')!r}, not 'PASS'"
        )

    summary = pair_report.get("summary")
    if not isinstance(summary, dict):
        errors.append("canonical pair report has no summary dictionary")
        return errors

    expected_fields = {
        "manifest_records": EXPECTED_TOTAL,
        "total_rgbir_files": EXPECTED_TOTAL,
        "total_rgbir_unique_tiles": EXPECTED_TOTAL,
        "total_canonical_gt_tiles": EXPECTED_TOTAL,
        "successfully_paired_tiles": EXPECTED_TOTAL,
        "invalid_rgbir_files": 0,
        "invalid_gt_files": 0,
        "file_sha256_verified": EXPECTED_TOTAL,
        "pixel_sha256_verified": EXPECTED_TOTAL,
        "final_supervised_tile_count": EXPECTED_TOTAL,
    }

    for key, expected in expected_fields.items():
        actual = summary.get(key)
        if actual != expected:
            errors.append(
                f"pair report summary[{key!r}]={actual!r}, expected {expected!r}"
            )

    pair_entries = pair_report.get("pairs")
    if not isinstance(pair_entries, list):
        errors.append("canonical pair report has no pairs list")
        return errors

    pair_ids: set[str] = set()
    for entry in pair_entries:
        if not isinstance(entry, dict):
            errors.append("non-dictionary pair entry found")
            continue
        tile_id = entry.get("tile_id")
        if tile_id in pair_ids:
            errors.append(f"duplicate tile_id in pair report: {tile_id}")
        pair_ids.add(tile_id)

        if entry.get("valid") is not True:
            errors.append(f"pair is not valid: {tile_id}")

    if pair_ids != manifest_tile_ids:
        missing = sorted(manifest_tile_ids - pair_ids, key=tile_sort_key)
        extra = sorted(pair_ids - manifest_tile_ids, key=tile_sort_key)
        if missing:
            errors.append(f"pair report missing manifest tiles: {missing}")
        if extra:
            errors.append(f"pair report contains extra tiles: {extra}")

    return errors


def build_text_report(split: dict[str, Any]) -> str:
    lines: list[str] = [
        "=" * 80,
        "Potsdam Tile-Level Train / Val / Test Split",
        "=" * 80,
        "",
        f"Status: {split['status']}",
        "",
        "[Policy]",
        split["split_policy"],
        "",
        "[Counts]",
        f"Train tiles : {split['counts']['train']}",
        f"Val tiles   : {split['counts']['val']}",
        f"Test tiles  : {split['counts']['test']}",
        f"Total tiles : {split['counts']['total']}",
        "",
        "[Train]",
        ", ".join(split["splits"]["train"]),
        "",
        "[Validation]",
        ", ".join(split["splits"]["val"]),
        "",
        "[Test]",
        ", ".join(split["splits"]["test"]),
        "",
        "[Validation-search objective]",
    ]

    for key, value in split["validation_selection"]["objective"].items():
        lines.append(f"{key}: {value:.10f}")

    lines.extend(
        [
            f"Candidates evaluated: "
            f"{split['validation_selection']['candidate_count']}",
            "",
            "[Class fractions]",
        ]
    )

    classes = split["classes"]
    for split_name in ("train", "val", "test"):
        lines.append(f"{split_name}:")
        fractions = split["statistics"][split_name]["class_fractions"]
        for c in classes:
            lines.append(f"  {c}: {fractions[c]:.8f}")

    lines.extend(
        [
            "",
            "[Integrity checks]",
        ]
    )

    for key, value in split["checks"].items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "=" * 80,
            f"FINAL STATUS: {split['status']}",
            (
                f"Frozen split = "
                f"{split['counts']['train']} train / "
                f"{split['counts']['val']} val / "
                f"{split['counts']['test']} test"
            ),
            "=" * 80,
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    manifest_path = resolve_project_path(project_root, args.manifest)
    pairs_path = resolve_project_path(project_root, args.pairs_report)
    output_path = resolve_project_path(project_root, args.output)
    report_dir = resolve_project_path(project_root, args.report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console_path = report_dir / "console.txt"
    report_txt_path = report_dir / "tile_split.txt"
    report_json_path = report_dir / "tile_split.json"

    if output_path.exists() and not args.overwrite:
        print(
            f"Refusing to overwrite existing frozen split: {output_path}\n"
            "If regeneration is intentional, rerun with --overwrite."
        )
        return 2

    original_stdout = sys.stdout
    tee = Tee(original_stdout, console_path)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("Build Potsdam Tile-Level Split")
        print("=" * 80)
        print(f"Manifest     : {manifest_path}")
        print(f"Pair report  : {pairs_path}")
        print(f"Split output : {output_path}")
        print()

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing canonical manifest: {manifest_path}")
        if not pairs_path.is_file():
            raise FileNotFoundError(f"Missing canonical pair report: {pairs_path}")

        manifest = load_json(manifest_path)
        pair_report = load_json(pairs_path)

        if not isinstance(manifest, dict):
            raise TypeError("labels_manifest.json top level must be a dictionary")

        tiles = manifest.get("tiles")
        if not isinstance(tiles, list):
            raise ValueError('labels_manifest.json must contain top-level "tiles" list')
        if len(tiles) != EXPECTED_TOTAL:
            raise ValueError(
                f"Canonical manifest has {len(tiles)} tiles, expected {EXPECTED_TOTAL}"
            )

        standard_colors = manifest.get("standard_colors")
        if not isinstance(standard_colors, dict) or len(standard_colors) != 6:
            raise ValueError(
                "Manifest standard_colors must be a 6-class dictionary."
            )
        classes = list(standard_colors.keys())

        tile_by_id: dict[str, dict[str, Any]] = {}
        for tile in tiles:
            if not isinstance(tile, dict):
                raise TypeError("Every manifest tile entry must be a dictionary.")
            tile_id = tile.get("tile_id")
            if not isinstance(tile_id, str):
                raise ValueError("Every manifest tile entry must have string tile_id.")
            if tile_id in tile_by_id:
                raise ValueError(f"Duplicate tile_id in manifest: {tile_id}")
            tile_by_id[tile_id] = tile

        manifest_ids = set(tile_by_id)

        pair_errors = validate_pairing_gate(pair_report, manifest_ids)
        if pair_errors:
            print("Canonical pairing gate FAILED:")
            for error in pair_errors:
                print(f"  - {error}")
            print()
            print("Tile split will not be created.")
            return 1

        source_counts = Counter(tile["source"] for tile in tiles)
        if source_counts.get(PARTICIPANT_SOURCE, 0) != EXPECTED_DEV:
            raise ValueError(
                f"Expected {EXPECTED_DEV} {PARTICIPANT_SOURCE} tiles, "
                f"found {source_counts.get(PARTICIPANT_SOURCE, 0)}"
            )
        if source_counts.get(ALL_SOURCE, 0) != EXPECTED_TEST:
            raise ValueError(
                f"Expected {EXPECTED_TEST} {ALL_SOURCE} tiles, "
                f"found {source_counts.get(ALL_SOURCE, 0)}"
            )
        unexpected_sources = set(source_counts) - {PARTICIPANT_SOURCE, ALL_SOURCE}
        if unexpected_sources:
            raise ValueError(f"Unexpected GT sources: {sorted(unexpected_sources)}")

        dev_ids = sorted(
            [
                tile_id
                for tile_id, tile in tile_by_id.items()
                if tile["source"] == PARTICIPANT_SOURCE
            ],
            key=tile_sort_key,
        )

        test_ids = sorted(
            [
                tile_id
                for tile_id, tile in tile_by_id.items()
                if tile["source"] == ALL_SOURCE
            ],
            key=tile_sort_key,
        )

        if args.val_size != DEFAULT_VAL_SIZE:
            print(
                f"NOTE: val_size={args.val_size}; "
                f"default research split uses {DEFAULT_VAL_SIZE}."
            )

        print(
            f"Searching deterministic class-balanced validation subset: "
            f"C({len(dev_ids)}, {args.val_size}) = "
            f"{math.comb(len(dev_ids), args.val_size)} candidates"
        )

        val_ids, objective, candidate_count = choose_validation_tiles(
            dev_ids=dev_ids,
            val_size=args.val_size,
            tile_by_id=tile_by_id,
            classes=classes,
        )

        val_set = set(val_ids)
        train_ids = sorted(
            [tile_id for tile_id in dev_ids if tile_id not in val_set],
            key=tile_sort_key,
        )

        train_set = set(train_ids)
        test_set = set(test_ids)

        checks = {
            "pairing_gate_passed": len(pair_errors) == 0,
            "train_val_overlap": bool(train_set & val_set),
            "train_test_overlap": bool(train_set & test_set),
            "val_test_overlap": bool(val_set & test_set),
            "union_is_all_38_tiles": (
                train_set | val_set | test_set
            ) == manifest_ids,
            "train_count_ok": len(train_ids) == (EXPECTED_DEV - args.val_size),
            "val_count_ok": len(val_ids) == args.val_size,
            "test_count_ok": len(test_ids) == EXPECTED_TEST,
            "total_count_ok": (
                len(train_ids) + len(val_ids) + len(test_ids)
            ) == EXPECTED_TOTAL,
        }

        pass_status = all(
            [
                checks["pairing_gate_passed"],
                checks["train_val_overlap"] is False,
                checks["train_test_overlap"] is False,
                checks["val_test_overlap"] is False,
                checks["union_is_all_38_tiles"],
                checks["train_count_ok"],
                checks["val_count_ok"],
                checks["test_count_ok"],
                checks["total_count_ok"],
            ]
        )

        statistics = {
            "development_pool": build_stats(dev_ids, tile_by_id, classes),
            "train": build_stats(train_ids, tile_by_id, classes),
            "val": build_stats(val_ids, tile_by_id, classes),
            "test": build_stats(test_ids, tile_by_id, classes),
        }

        tile_records = []
        split_lookup = {}
        for tile_id in train_ids:
            split_lookup[tile_id] = "train"
        for tile_id in val_ids:
            split_lookup[tile_id] = "val"
        for tile_id in test_ids:
            split_lookup[tile_id] = "test"

        for tile_id in sorted(manifest_ids, key=tile_sort_key):
            tile = tile_by_id[tile_id]
            tile_records.append(
                {
                    "tile_id": tile_id,
                    "split": split_lookup[tile_id],
                    "gt_source": tile["source"],
                    "gt_relpath": tile["gt_relpath"],
                    "file_sha256": tile["file_sha256"],
                    "pixel_sha256": tile["pixel_sha256"],
                }
            )

        split = {
            "schema_version": 1,
            "dataset": "ISPRS Potsdam",
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "PASS" if pass_status else "FAIL",
            "split_policy": (
                "Hold out all 14 canonical tiles whose source is "
                "'5_Labels_all' as test. Use the 24 "
                "'5_Labels_for_participants' tiles as the development pool; "
                "select validation tiles by deterministic exhaustive search "
                "minimizing six-class pixel-distribution divergence from the "
                "development pool, then use the remaining development tiles "
                "for training. No image content or model metric is used."
            ),
            "inputs": {
                "labels_manifest": display_path(manifest_path, project_root),
                "labels_manifest_sha256": file_sha256(manifest_path),
                "canonical_pairs_report": display_path(pairs_path, project_root),
                "canonical_pairs_report_sha256": file_sha256(pairs_path),
            },
            "classes": classes,
            "source_counts": dict(source_counts),
            "counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "total": len(train_ids) + len(val_ids) + len(test_ids),
            },
            "splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "validation_selection": {
                "method": "exhaustive_class_distribution_balance",
                "val_size": args.val_size,
                "candidate_count": candidate_count,
                "objective": objective,
            },
            "statistics": statistics,
            "checks": checks,
            "tiles": tile_records,
        }

        text_report = build_text_report(split)

        if not pass_status:
            print(text_report)
            print("Split integrity checks failed; frozen output will not be written.")
            report_txt_path.write_text(text_report, encoding="utf-8")
            report_json_path.write_text(
                json.dumps(split, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return 1

        payload = json.dumps(split, indent=2, ensure_ascii=False) + "\n"

        # Write the frozen processed split and a report copy.
        output_path.write_text(payload, encoding="utf-8")
        report_json_path.write_text(payload, encoding="utf-8")
        report_txt_path.write_text(text_report, encoding="utf-8")

        print(text_report)
        print(f"Frozen split : {output_path}")
        print(f"JSON report  : {report_json_path}")
        print(f"Text report  : {report_txt_path}")
        print(f"Console log  : {console_path}")

        return 0

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
