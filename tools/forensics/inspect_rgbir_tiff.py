#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = None

TEST_TILES = ("2_10", "4_12")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Potsdam RGBIR TIFF integrity and reader compatibility."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Potsdam root, e.g. data/raw/potsdam",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
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
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def first_bytes_file(
    path: Path,
    n: int = 32,
) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def first_bytes_zip(
    zf: zipfile.ZipFile,
    member: str,
    n: int = 32,
) -> bytes:
    with zf.open(member, "r") as f:
        return f.read(n)


def classify_tiff_signature(data: bytes) -> str:
    if data.startswith(b"II*\x00"):
        return "Classic TIFF, little-endian"

    if data.startswith(b"MM\x00*"):
        return "Classic TIFF, big-endian"

    if data.startswith(b"II+\x00"):
        return "BigTIFF, little-endian"

    if data.startswith(b"MM\x00+"):
        return "BigTIFF, big-endian"

    return "NOT a recognized TIFF signature"


def find_extracted(
    root: Path,
    tile: str,
) -> Path:
    base = (
        root
        / "_expanded"
        / "4_Ortho_RGBIR"
    )

    basename = (
        f"top_potsdam_{tile}_RGBIR.tif"
    )

    matches = [
        p
        for p in base.rglob(basename)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one extracted {basename}, "
            f"found {len(matches)}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def find_zip(root: Path) -> Path:
    candidates = [
        root / "4_Ortho_RGBIR.zip",
        root / "Potsdam" / "4_Ortho_RGBIR.zip",
    ]

    for p in candidates:
        if p.is_file():
            return p

    matches = [
        p
        for p in root.rglob("4_Ortho_RGBIR.zip")
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one 4_Ortho_RGBIR.zip, "
            f"found {len(matches)}:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]


def find_member(
    zf: zipfile.ZipFile,
    tile: str,
) -> str:
    basename = (
        f"top_potsdam_{tile}_RGBIR.tif"
    ).lower()

    matches = [
        name
        for name in zf.namelist()
        if Path(name).name.lower() == basename
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one ZIP member for "
            f"{basename}, found {len(matches)}:\n"
            + "\n".join(matches)
        )

    return matches[0]


def test_pillow(path: Path) -> dict:
    result = {
        "available": True,
        "success": False,
        "error": None,
    }

    try:
        with Image.open(path) as img:
            result.update(
                {
                    "success": True,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "bands": img.getbands(),
                    "compression": img.info.get(
                        "compression"
                    ),
                }
            )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def test_tifffile(path: Path) -> dict:
    try:
        import tifffile
    except Exception as exc:
        return {
            "available": False,
            "success": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    result = {
        "available": True,
        "success": False,
        "error": None,
    }

    try:
        with tifffile.TiffFile(path) as tif:
            result["success"] = True
            result["is_bigtiff"] = tif.is_bigtiff
            result["pages"] = len(tif.pages)

            if tif.series:
                series = tif.series[0]
                result["shape"] = tuple(
                    int(x) for x in series.shape
                )
                result["dtype"] = str(
                    series.dtype
                )
                result["axes"] = str(
                    series.axes
                )

            if tif.pages:
                page = tif.pages[0]

                result["page_shape"] = tuple(
                    int(x) for x in page.shape
                )

                result["page_dtype"] = str(
                    page.dtype
                )

                result["compression"] = str(
                    page.compression
                )

                result["photometric"] = str(
                    page.photometric
                )

                result["samples_per_pixel"] = (
                    page.samplesperpixel
                )

                result["planarconfig"] = str(
                    page.planarconfig
                )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def test_rasterio(path: Path) -> dict:
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        return {
            "available": False,
            "success": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    result = {
        "available": True,
        "success": False,
        "error": None,
    }

    try:
        with rasterio.open(path) as src:
            result["success"] = True
            result["driver"] = src.driver
            result["width"] = src.width
            result["height"] = src.height
            result["count"] = src.count
            result["dtypes"] = list(
                src.dtypes
            )
            result["colorinterp"] = [
                str(x)
                for x in src.colorinterp
            ]
            result["crs"] = (
                None
                if src.crs is None
                else str(src.crs)
            )

            sample = src.read(
                window=Window(0, 0, 1, 1)
            )

            result["sample_shape"] = (
                tuple(
                    int(x)
                    for x in sample.shape
                )
            )

            result["sample_values"] = (
                sample[:, 0, 0].tolist()
            )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def print_reader(
    name: str,
    result: dict,
):
    print(f"  {name}:")

    for key, value in result.items():
        print(
            f"    {key}: {value}"
        )


def main() -> int:
    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise SystemExit(
            f"ERROR: root does not exist: {root}"
        )

    zip_path = find_zip(root)

    print("=== RGBIR archive ===")
    print(f"ZIP: {zip_path}")
    print()

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()

        print(
            "ZIP CRC test: "
            + (
                "PASS"
                if bad_member is None
                else f"FAIL ({bad_member})"
            )
        )

        print()

        for tile in TEST_TILES:
            print("=" * 80)
            print(f"Tile: {tile}")

            extracted = find_extracted(
                root,
                tile,
            )

            member = find_member(
                zf,
                tile,
            )

            zip_info = zf.getinfo(member)

            extracted_hash = sha256_file(
                extracted
            )

            archive_hash = sha256_zip_member(
                zf,
                member,
            )

            extracted_head = (
                first_bytes_file(
                    extracted
                )
            )

            archive_head = (
                first_bytes_zip(
                    zf,
                    member,
                )
            )

            print()
            print("Paths")
            print(
                f"  extracted: {extracted}"
            )
            print(
                f"  member   : {member}"
            )

            print()
            print("File sizes")
            print(
                f"  extracted bytes     : "
                f"{extracted.stat().st_size}"
            )
            print(
                f"  ZIP member bytes    : "
                f"{zip_info.file_size}"
            )
            print(
                f"  ZIP compressed bytes: "
                f"{zip_info.compress_size}"
            )
            print(
                f"  ZIP compression type: "
                f"{zip_info.compress_type}"
            )

            print()
            print("Integrity")
            print(
                f"  extracted SHA256: "
                f"{extracted_hash}"
            )
            print(
                f"  archive SHA256  : "
                f"{archive_hash}"
            )
            print(
                f"  archive == extracted: "
                f"{archive_hash == extracted_hash}"
            )

            print()
            print("Header")
            print(
                f"  extracted first 32 bytes: "
                f"{extracted_head.hex(' ')}"
            )
            print(
                f"  archive first 32 bytes  : "
                f"{archive_head.hex(' ')}"
            )
            print(
                f"  extracted signature: "
                f"{classify_tiff_signature(extracted_head)}"
            )
            print(
                f"  archive signature  : "
                f"{classify_tiff_signature(archive_head)}"
            )
            print(
                f"  header identical   : "
                f"{extracted_head == archive_head}"
            )

            print()
            print("Reader tests")

            pillow_result = test_pillow(
                extracted
            )

            tifffile_result = test_tifffile(
                extracted
            )

            rasterio_result = test_rasterio(
                extracted
            )

            print_reader(
                "Pillow",
                pillow_result,
            )

            print_reader(
                "tifffile",
                tifffile_result,
            )

            print_reader(
                "rasterio",
                rasterio_result,
            )

            print()

    print("=" * 80)
    print("Inspection completed.")
    print("No file was modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
