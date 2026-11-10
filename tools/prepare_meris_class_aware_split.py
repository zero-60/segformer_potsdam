#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class-aware TLSNMSP split; compatible with existing training, source data read-only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
REQUIRED_FOLDERS = ("RGB", "multispectral", "mask", "mask_converted")
EXTENSIONS = {
    "RGB": {".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "multispectral": {".tif", ".tiff"},
    "mask": {".png", ".tif", ".tiff"},
    "mask_converted": {".png", ".tif", ".tiff"},
}
SPLIT_GROUPS = {
    "train": {"100", "101", "103", "106", "107"},
    "val": {"102"},
    "test": {"108"},
}
CLASS_NAMES = [
    "building", "impervious_surface", "water", "barren", "tree", "low_vegetation", "background_other"
]
RAW_LABEL_TO_TRAIN_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
RGB_COLOR_TO_RAW_ID = {
    (178, 74, 64): 1,       # #B24A40 building
    (128, 128, 128): 2,     # #808080 impervious surface
    (135, 206, 235): 3,    # #87CEEB water
    (210, 180, 100): 4,    # #D2B464 barren
    (34, 139, 34): 5,      # #228B22 tree
    (144, 238, 144): 6,    # #90EE90 low vegetation
    (255, 255, 0): 7,      # #FFFF00 background / other objects
}
GROUP_RE = re.compile(r"^(\d+)FPLAN_DJI_", re.IGNORECASE)


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"RGB 读取失败: {path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"不支持的 RGB shape {arr.shape}: {path}")
    if arr.dtype != np.uint8:
        raise RuntimeError(f"RGB 应为 uint8，实际 {arr.dtype}: {path}")
    return arr


def load_ms(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise RuntimeError(f"多光谱 TIFF 应为三维数组，实际 {arr.shape}: {path}")
    if arr.shape[0] == 5:
        chw = arr
    elif arr.shape[-1] == 5:
        chw = np.moveaxis(arr, -1, 0)
    else:
        raise RuntimeError(f"多光谱 TIFF 无法识别五波段布局，实际 {arr.shape}: {path}")
    if chw.dtype != np.uint16:
        raise RuntimeError(f"多光谱 TIFF 预期 uint16，实际 {chw.dtype}: {path}")
    return chw


def load_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败: {path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"转换标签必须是单通道 ID 图，实际 {arr.shape}: {path}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"标签类型应为 uint8/uint16，实际 {arr.dtype}: {path}")
    return arr


def verify_original_mask(original_path: Path, converted: np.ndarray, sample_id: str):
    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"原始彩色标签读取失败: {original_path}")
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    if original.shape[:2] != converted.shape:
        raise RuntimeError(f"{sample_id}: 原始 mask 与 converted 尺寸不一致")
    packed = (original[:, :, 0].astype(np.uint32) << 16) | (original[:, :, 1].astype(np.uint32) << 8) | original[:, :, 2].astype(np.uint32)
    color_to_id = {((r << 16) | (g << 8) | b): cls for (r, g, b), cls in RGB_COLOR_TO_RAW_ID.items()}
    expected = np.full(packed.shape, 0, dtype=np.uint8)
    unknown = np.zeros(packed.shape, dtype=bool)
    for color_key in np.unique(packed):
        cls = color_to_id.get(int(color_key))
        mask = packed == color_key
        if cls is None:
            unknown |= mask
        else:
            expected[mask] = cls
    if unknown.any():
        unknown_colors = [f"#{int(v):06X}" for v in np.unique(packed[unknown])[:10]]
        raise RuntimeError(f"{sample_id}: 原始 mask 有未知颜色 {unknown_colors}")
    mismatch = int(np.count_nonzero(expected != converted))
    if mismatch:
        raise RuntimeError(f"{sample_id}: 原始 mask 与 converted 有 {mismatch} 个像素不一致")


def get_group(sample_id: str) -> str:
    m = GROUP_RE.match(sample_id)
    if not m:
        raise RuntimeError(f"无法从样本名识别航带组: {sample_id}")
    return m.group(1)


def scan_dir(root: Path, folder: str):
    d = root / folder
    if not d.is_dir():
        raise FileNotFoundError(f"缺少目录: {d}")
    idx = defaultdict(list)
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSIONS[folder]:
            idx[p.stem.casefold()].append(p)
    dup = {k: v for k, v in idx.items() if len(v) != 1}
    if dup:
        example = list(dup.items())[:5]
        raise RuntimeError(f"{folder} 中存在重复文件名 stem: {[(k,[str(p) for p in v]) for k,v in example]}")
    return {k: v[0] for k, v in idx.items()}


SPLITS = ('train', 'val', 'test')

def choose_split(counts, sizes, ratios, priority, priority_train, seed, trials):
    """Search indivisible units. Exhaustive <=10 units, seeded search otherwise.

    Lexicographic objectives: nonempty sets; all available classes in train;
    class coverage in held-out sets; count/pixel targets and sample ratios.
    A zero pixel class contributes no objective.
    """
    n = len(sizes)
    if n < 3:
        raise ValueError('至少需要三个独立组；不能在保持组隔离的同时生成三个集合。')
    total = counts.sum(axis=0)
    active = total > 0
    positive = (counts > 0).astype(float)
    total_positive = positive.sum(axis=0)
    targets = np.repeat(np.asarray(ratios)[:, None], 7, axis=1)
    targets[0, priority] = priority_train
    targets[1:, priority] = (1-priority_train) * np.asarray(ratios[1:])[:, None] / sum(ratios[1:])
    rng = np.random.default_rng(seed)
    best_key, best = None, None
    iterator = itertools.product(range(3), repeat=n) if n <= 10 else None
    for offset in range(0, 3**n if iterator is not None else trials, 1024):
        if iterator is not None:
            batch = np.asarray(list(itertools.islice(iterator, 1024)), dtype=np.int8)
        else:
            batch = rng.choice(3, size=(min(1024, trials-offset), n), p=ratios)
        onehot = (batch[:, :, None] == np.arange(3)).astype(float)
        sc = np.einsum('bns,nc->bsc', onehot, counts)
        sp = np.einsum('bns,nc->bsc', onehot, positive)
        ss = np.einsum('bns,n->bs', onehot, sizes)
        # Coverage includes any real labeled pixel; meaningful-size counts audited separately.
        empty = (ss == 0).sum(axis=1)
        train_missing = ((sc[:, 0, :] == 0) & active).sum(axis=1)
        eval_missing = ((sc[:, 1:, :] == 0) & active).sum(axis=(1,2))
        weights = np.ones(7); weights[priority] = 3
        pixel_error = (((sc / np.maximum(total, 1) - targets)**2) * weights * active).sum(axis=(1,2))
        group_error = (((sp / np.maximum(total_positive, 1) - targets)**2) * weights * active).sum(axis=(1,2))
        size_error = ((ss / sizes.sum() - ratios)**2).sum(axis=1)
        error = pixel_error + 0.5*group_error + 5*size_error
        order = np.lexsort((error, eval_missing, train_missing, empty))
        i = order[0]
        key = (int(empty[i]), int(train_missing[i]), int(eval_missing[i]), float(error[i]))
        if best_key is None or key < best_key:
            best_key, best = key, batch[i].copy()
    if best_key[0]:
        raise ValueError('搜索未找到三个非空集合；增加 --search-trials 或检查分组。')
    return best, best_key


def main():
    ap = argparse.ArgumentParser(description='根据掩码统计重划 TLSNMSP 协议，不修改模型和原始数据')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'data/processed/meris_class_aware_v1')
    ap.add_argument('--ratios', type=float, nargs=3, default=[.7,.15,.15], metavar=('TRAIN','VAL','TEST'))
    ap.add_argument('--priority-raw-ids', type=int, nargs='+', default=[1,7])
    ap.add_argument('--priority-train-fraction', type=float, default=.8,
                    help='优先类训练像素/阳性组份额目标；软目标，不是保证值')
    ap.add_argument('--mode', choices=['group','image'], default='group')
    ap.add_argument('--groups-csv', type=Path, help='可选 sample_id,group；同一地点/重叠航带必须使用同组')
    ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--search-trials', type=int, default=30000)
    ap.add_argument('--nir-sample-stride', type=int, default=8)
    ap.add_argument('--presence-min-pixels', type=int, default=100,
                    help='额外审计有效阳性图像阈值；不把小目标从标签或划分中删除')
    ap.add_argument('--skip-color-mask-check', action='store_true')
    ap.add_argument('--strict-coverage', action='store_true', help='任一集合缺类别则仅输出审计，不生成训练协议')
    args = ap.parse_args()
    ratios = np.asarray(args.ratios, dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(),1):
        ap.error('--ratios 三个正数之和必须为1')
    if not 0 < args.priority_train_fraction < 1 or any(i not in range(1,8) for i in args.priority_raw_ids):
        ap.error('优先类原始 ID 必须在1..7，目标比例必须在0..1之间')
    if min(args.search_trials,args.nir_sample_stride,args.presence_min_pixels) < 1:
        ap.error('搜索次数、抽样步长和阳性阈值必须为正数')
    root, out = args.data_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'输出目录非空，拒绝覆盖历史协议；请换新目录：{out}')
    indexes = {folder: scan_dir(root,folder) for folder in REQUIRED_FOLDERS}
    mapping = {}
    if args.groups_csv:
        with args.groups_csv.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                k = r['sample_id'].casefold()
                if k in mapping or not r['group'].strip():
                    raise ValueError('groups-csv 有重复 sample_id 或空组名')
                mapping[k] = r['group'].strip()
    if args.mode == 'image' and args.groups_csv:
        ap.error('--groups-csv 只能与 --mode group 一起使用')
    keys = sorted(indexes['mask_converted'])
    missing = [{'sample_id':k, 'missing_folders':';'.join(f for f in REQUIRED_FOLDERS if k not in indexes[f])}
               for k in sorted(set(indexes['mask']) | set(keys))
               if any(k not in indexes[f] for f in REQUIRED_FOLDERS)]
    if missing:
        out.mkdir(parents=True,exist_ok=True)
        write_csv(out/'incomplete_pairs.csv',missing)
        raise ValueError('有标注样本配对不完整，见 incomplete_pairs.csv；修复后使用新输出目录')
    if not keys:
        raise ValueError('没有有标签的完整配对样本')
    rows, pixels, hashes = [], [], {}
    for i,k in enumerate(keys,1):
        paths = {f:indexes[f][k] for f in REQUIRED_FOLDERS}
        label = load_label(paths['mask_converted'])
        ids, nums = np.unique(label,return_counts=True)
        if set(map(int,ids)) - set(range(1,8)):
            raise ValueError(f'{k}: 标签需为1..7，实际{ids.tolist()}；不自动猜测0..6编码')
        h,w = label.shape
        if min(h,w) < 512:
            raise ValueError(f'{k}: 尺寸小于原训练裁剪512，拒绝生成不能训练的协议')
        rgb, ms = load_rgb(paths['RGB']), load_ms(paths['multispectral'])
        if rgb.shape[:2] != (h,w) or ms.shape[1:] != (h,w):
            raise ValueError(f'{k}: RGB/MS/label尺寸不一致')
        if not args.skip_color_mask_check:
            verify_original_mask(paths['mask'],label,k)
        # Exact duplicate multispectral scenes must not leak across sets.
        digest = hashlib.sha256(np.ascontiguousarray(ms).tobytes()).hexdigest()
        hashes[k] = digest
        if args.groups_csv:
            if k not in mapping:
                raise ValueError(f'groups-csv 缺少 {k}')
            group = mapping[k]
        else:
            group = get_group(paths['RGB'].stem)
        row = dict(sample_id=paths['RGB'].stem,group=group,split='',
                   rgb_path=paths['RGB'].relative_to(root).as_posix(),
                   multispectral_path=paths['multispectral'].relative_to(root).as_posix(),
                   mask_path=paths['mask'].relative_to(root).as_posix(),
                   label_path=paths['mask_converted'].relative_to(root).as_posix(),
                   width=w,height=h,rgb_channels_used=3,multispectral_band_count=5,nir_band_index_1based=5,
                   label_ids=';'.join(map(str,ids)),label_mapping_verified=not args.skip_color_mask_check)
        rows.append(row)
        counts = np.zeros(7,dtype=np.int64); counts[ids.astype(int)-1] = nums
        pixels.append(counts)
        if i % 10 == 0 or i == len(keys):
            print(f'[SCAN] {i}/{len(keys)}',flush=True)
    pixels = np.asarray(pixels)
    units = sorted(set(r['group'] if args.mode=='group' else r['sample_id'] for r in rows))
    unit_index = {u:i for i,u in enumerate(units)}
    ui = np.asarray([unit_index[r['group'] if args.mode=='group' else r['sample_id']] for r in rows])
    uc = np.zeros((len(units),7),dtype=np.int64)
    np.add.at(uc,ui,pixels)
    sizes = np.bincount(ui,minlength=len(units))
    assignment, score = choose_split(uc,sizes,ratios,np.asarray(args.priority_raw_ids)-1,
                                     args.priority_train_fraction,args.seed,args.search_trials)
    selected = assignment[ui]
    for row,s in zip(rows,selected): row['split'] = SPLITS[s]
    by_hash = defaultdict(set)
    for k,s in zip(keys,selected): by_hash[hashes[k]].add(int(s))
    if any(len(s)>1 for s in by_hash.values()):
        raise ValueError('发现完全相同的多光谱图像跨集合；请用 groups-csv 将重复影像放入同一组后再划分')
    out.mkdir(parents=True,exist_ok=True)
    audit = []
    warnings = []
    for s,name in enumerate(SPLITS):
        subset = pixels[selected==s]; total = int(subset.sum())
        for c in range(7):
            count = int(subset[:,c].sum())
            audit.append(dict(split=name,raw_label_id=c+1,train_label_id=c,class_name=CLASS_NAMES[c],
                              pixel_count=count,pixel_percent=100*count/max(total,1),total_pixels=total,
                              positive_images=int((subset[:,c]>0).sum()),
                              meaningful_positive_images=int((subset[:,c]>=args.presence_min_pixels).sum()),
                              positive_groups=len({r['group'] for j,r in enumerate(rows) if selected[j]==s and pixels[j,c]>0})))
            if count==0: warnings.append(f'{name} 缺少 {CLASS_NAMES[c]} (raw ID={c+1})')
    for c in range(7):
        groups = {r['group'] for j,r in enumerate(rows) if pixels[j,c]>0}
        if len(groups)<3:
            warnings.append(f'{CLASS_NAMES[c]} 仅见于 {len(groups)} 个航带组，组隔离下不能覆盖三个集合')
    if args.mode=='image':
        warnings.append('按图像划分：同航带/重叠区域可能跨集合；仅作探索实验，不能声称空间独立测试')
    warnings.append('航带名隔离不能证明地理独立；不同航带重叠时请通过 groups-csv 合并为同一地点组')
    write_csv(out/'class_pixel_counts_by_split.csv',audit)
    write_csv(out/'sample_class_counts.csv',[
        dict(sample_id=r['sample_id'],group=r['group'],split=r['split'],
             **{f'raw_{c+1}_pixels':int(pixels[j,c]) for c in range(7)}) for j,r in enumerate(rows)])
    write_csv(out/'group_assignment.csv',[dict(group_or_image=u,split=SPLITS[int(assignment[i])],image_count=int(sizes[i])) for i,u in enumerate(units)])
    report = dict(seed=args.seed,mode=args.mode,ratios_requested=ratios.tolist(),
                  priority_raw_ids=args.priority_raw_ids,priority_train_fraction=args.priority_train_fraction,
                  sample_counts={s:int((selected==i).sum()) for i,s in enumerate(SPLITS)},
                  search='exhaustive' if len(units)<=10 else 'seeded_random_approximate',objective=list(score),
                  scanned_files={f:len(idx) for f,idx in indexes.items()},
                  unlabeled_rgb_count=len(set(indexes['RGB'])-set(keys)),warnings=warnings,
                  protocol_ready=False)
    def save_report():
        (out/'split_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'preparation_report.txt').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save_report()
    if any(r['pixel_count']==0 and (args.strict_coverage or r['split']=='train') for r in audit):
        raise ValueError('存在缺类：已保存审计，但未生成可训练协议。检查报告、补标注或调整独立组。')
    # Recompute NIR mean/std solely from the new training set, streaming.
    count, mean, m2 = 0, 0., 0.
    for r in rows:
        if r['split']!='train': continue
        x = load_ms(root/r['multispectral_path'])[4,::args.nir_sample_stride,::args.nir_sample_stride].astype(np.float64)
        n = x.size; bm = float(x.mean()); b2 = float(np.square(x-bm).sum())
        delta = bm-mean; new = count+n
        m2 += b2+delta*delta*count*n/new; mean += delta*n/new; count=new
    std = float(np.sqrt(m2/max(count,1)))
    if count<1000 or not np.isfinite(std) or std<=0:
        raise ValueError('训练集 NIR 统计异常/样本不足；协议未生成')
    config = dict(protocol_version='TLSNMSP_RGB_NIR_7CLASS_v1',
                  created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  data_root=str(root),project_root=str(PROJECT_ROOT),sample_counts=report['sample_counts'],
                  split_groups={s:sorted({r['group'] for r in rows if r['split']==s}) for s in SPLITS},
                  class_names_train_id_order=CLASS_NAMES,raw_label_to_train_id={str(i):i-1 for i in range(1,8)},
                  ignore_index=255,crop_size=512,input_modalities=['RGB','NIR'],
                  rgb_normalization=dict(type='ImageNet',mean=[.485,.456,.406],std=[.229,.224,.225]),
                  nir_normalization=dict(method='train-only sampled streaming z-score on raw DN values',
                                         band_index_1based=5,sample_stride=args.nir_sample_stride,
                                         mean_dn=mean,std_dn=std,sampled_value_count=count),
                  split_method=args.mode,seed=args.seed,source_files_untouched=True,
                  original_mask_color_mapping_verified_samples=0 if args.skip_color_mask_check else len(rows))
    for s in SPLITS: write_csv(out/f'{s}.csv',[r for r in rows if r['split']==s])
    write_csv(out/'manifest_all.csv',rows)
    for name in ('train','val','test'):
        report.setdefault('manifest_sha256',{})[name] = hashlib.sha256((out/f'{name}.csv').read_bytes()).hexdigest()
    (out/'dataset_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    report['protocol_ready']=True; save_report()
    print('[READY]',report['sample_counts'])
    for w in warnings: print('[WARN]',w)
    print('[OUT]',out)
    print('请检查 class_pixel_counts_by_split.csv，然后用新 output-dir、resume none 先做 smoke test。')

if __name__=='__main__':
    main()
