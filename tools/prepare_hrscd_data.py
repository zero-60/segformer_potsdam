#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRSCD five-archive safe extraction and inventory (read-only to ZIP files).

Placement:
  project/tools/prepare_hrscd_data.py
  project/data/HRSCD/{images_2006.zip,images_2012.zip,
     labels_land_cover_2006.zip,labels_land_cover_2012.zip,labels_change.zip}

Inspect ZIP contents first:
  python tools/prepare_hrscd_data.py --root data/HRSCD --output outputs/dataset_audit/HRSCD
Extract (do not delete the ZIP files):
  python tools/prepare_hrscd_data.py --root data/HRSCD --output outputs/dataset_audit/HRSCD --extract
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

ARCHIVES = (
    "images_2006.zip",
    "images_2012.zip",
    "labels_land_cover_2006.zip",
    "labels_land_cover_2012.zip",
    "labels_change.zip",
)
MAX_MEMBER_BYTES = 20 * 1024**3  # protective ceiling, override intentionally in source only
CHUNK = 2 * 1024**2


def safe_member_target(dest: Path, member: zipfile.ZipInfo) -> Path:
    # Do not allow path traversal, absolute paths, backslash windows-path ambiguity,
    # symbolic links, special files, or NUL characters.
    normalized = member.filename.replace("\\", "/")
    parts = normalized.rstrip("/").split("/")
    if not parts or any(x in ("", ".", "..") for x in parts):
        raise ValueError(f"Unsafe archive path: {member.filename!r}")
    if normalized.startswith("/") or ":" in parts[0] or "\x00" in normalized:
        raise ValueError(f"Unsafe absolute/drive path: {member.filename!r}")
    kind = (member.external_attr >> 16) & 0o170000
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Unsafe archive file type: {member.filename!r}")
    target = dest.joinpath(*parts)
    if not target.resolve().is_relative_to(dest.resolve()):
        raise ValueError(f"Archive entry escapes extraction root: {member.filename!r}")
    if member.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"Suspiciously large archive member: {member.filename!r}")
    return target


def existing_crc(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return crc & 0xFFFFFFFF
            crc = zlib.crc32(block, crc)


def inspect_archives(root: Path) -> tuple[list[dict], int, list[str]]:
    archives, total, errors = [], 0, []
    for name in ARCHIVES:
        src = root / name
        item = {"archive": name, "found": src.is_file(), "files": 0,
                "uncompressed_bytes": 0, "examples": [], "destination": name.removesuffix(".zip")}
        if not src.is_file():
            errors.append(f"Missing archive: {src}")
            archives.append(item)
            continue
        try:
            with zipfile.ZipFile(src) as z:
                seen = set()
                for member in z.infolist():
                    target = safe_member_target(root/name.removesuffix(".zip"), member)
                    if target in seen:
                        raise ValueError(f"Duplicate ZIP member target: {member.filename}")
                    seen.add(target)
                    if member.is_dir():
                        continue
                    item["files"] += 1
                    item["uncompressed_bytes"] += member.file_size
                    if len(item["examples"]) < 5:
                        item["examples"].append(member.filename)
                total += item["uncompressed_bytes"]
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
        archives.append(item)
    return archives, total, errors


def space_needed(root: Path, archives: list[dict]) -> int:
    needed = 0
    for a in archives:
        if not a["found"]:
            continue
        destination = root / a["destination"]
        with zipfile.ZipFile(root/a["archive"]) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                dst = safe_member_target(destination, member)
                if dst.exists() and dst.is_file() and dst.stat().st_size == member.file_size:
                    continue
                needed += member.file_size
    return needed


def unpack(root: Path, archives: list[dict], trust_existing_size: bool) -> dict:
    stats = {"extracted_files": 0, "skipped_verified_files": 0, "extracted_bytes": 0}
    for a in archives:
        dest = root / a["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        src = root / a["archive"]
        print(f"[EXTRACT] {src.name} -> {dest}")
        with zipfile.ZipFile(src) as z:
            for i, member in enumerate(z.infolist(), 1):
                target = safe_member_target(dest, member)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    if not target.is_file():
                        raise RuntimeError(f"Cannot overwrite non-file target: {target}")
                    if target.stat().st_size == member.file_size:
                        if trust_existing_size or existing_crc(target) == member.CRC:
                            stats["skipped_verified_files"] += 1
                            continue
                    raise RuntimeError(f"Existing file differs from archive: {target}. "
                                       "Back it up/move it manually before retrying.")
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temp = tempfile.mkstemp(prefix="._hrscd_partial_", dir=str(target.parent))
                os.close(fd)
                try:
                    with z.open(member) as fin, open(temp, "wb") as fout:
                        shutil.copyfileobj(fin, fout, length=CHUNK)
                        fout.flush()
                        os.fsync(fout.fileno())
                    if os.path.getsize(temp) != member.file_size:
                        raise IOError(f"Extraction size mismatch: {target}")
                    os.replace(temp, target)
                    stats["extracted_files"] += 1
                    stats["extracted_bytes"] += member.file_size
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
                if i % 50 == 0:
                    print(f"    {i}/{len(z.infolist())} archived entries examined", flush=True)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Safely inspect/unpack the five HRSCD original ZIPs")
    ap.add_argument("--root", type=Path, default=Path("data/HRSCD"))
    ap.add_argument("--output", type=Path, default=Path("outputs/dataset_audit/HRSCD"))
    ap.add_argument("--extract", action="store_true", help="Execute unpacking; otherwise inventory only")
    ap.add_argument("--trust-existing-size", action="store_true",
                    help="Skip existing files by size without verifying CRC (faster but less safe)")
    args = ap.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] No directory: {root}", file=sys.stderr)
        return 2
    info, total, errors = inspect_archives(root)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "HRSCD", "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "data_root": str(root), "archives": info, "total_uncompressed_bytes": total,
              "errors": errors, "extracted": None}
    print("[INVENTORY]")
    for item in info:
        print(f"  {item['archive']}: present={item['found']}, "
              f"files={item['files']}, uncompressed_GiB={item['uncompressed_bytes']/1024**3:.2f}")
        for sample in item["examples"][:2]:
            print("    e.g.", sample)
    print(f"[SUMMARY] Uncompressed size estimate: {total/1024**3:.2f} GiB")
    if errors:
        for err in errors:
            print("[ERROR]",err, file=sys.stderr)
    elif args.extract:
        try:
            required = space_needed(root, info)
            free = shutil.disk_usage(root).free
            print(f"[DISK] estimated extra space {required/1024**3:.2f} GiB; "
                  f"free {free/1024**3:.2f} GiB")
            if free < required + 256 * 1024**2:
                raise RuntimeError("Insufficient free space for extraction with a safety margin. "
                                   "Free disk space before retrying; do not delete the source ZIPs yet.")
            report["extracted"] = unpack(root, info, args.trust_existing_size)
            print("[DONE]", report["extracted"])
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            report["errors"] = errors
            print("[ERROR]", errors[-1], file=sys.stderr)
    elif not errors:
        print("[DRY RUN] Add --extract to unpack all five ZIP files.")
    path = args.output / "hrscd_archive_inventory.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[REPORT]", path.resolve())
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
