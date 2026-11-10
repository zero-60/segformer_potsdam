#!/usr/bin/env python3
"""Patch the HRSCD key normalizers in-place without relying on git patch context."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
FILES = {
    ROOT / 'tools' / 'inspect_hrscd_dataset.py': [
        ('    "2006", "2012", "image", "images", "img", "rgb", "label", "labels",\n',
         '    "2005", "2006", "2012", "image", "images", "img", "rgb", "label", "labels",\n'),
        ('    "change", "changes", "mask", "masks", "gt", "semantic", "segmentation",\n',
         '    "change", "changes", "mask", "masks", "gt", "semantic", "segmentation",\n    "0m50", "e080",\n'),
    ],
    ROOT / 'tools' / 'build_hrscd_cd0_protocol.py': [
        ('NONKEY = {"2006", "2012", "image", "images", "img", "label", "labels", "land",\n',
         'NONKEY = {"2005", "2006", "2012", "image", "images", "img", "label", "labels", "land",\n'),
        ('          "ortho", "semantic", "lcm", "ground", "truth"}',
         '          "ortho", "semantic", "lcm", "ground", "truth", "0m50", "e080"}'),
    ],
}

for path, replacements in FILES.items():
    if not path.is_file():
        raise SystemExit(f'[ERROR] 文件不存在: {path}')
    text = path.read_text(encoding='utf-8')
    original = text
    for old, new in replacements:
        if old in text and new not in text:
            text = text.replace(old, new, 1)
    if text == original:
        if all(new in text for _, new in replacements):
            print(f'[SKIP] 已经修复: {path}')
            continue
        raise SystemExit(f'[ERROR] 未找到预期代码片段，未修改: {path}')
    backup = path.with_suffix(path.suffix + '.before_key_fix')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding='utf-8')
    print(f'[OK] 已修改: {path}')
