#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inventory Potsdam files and rebuild missing metadata using the repository's frozen audit.
Does not move, modify, resample, or relabel any TIFF. No random data split.
"""
import argparse, importlib.util, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load_auditor(root):
    path=root/'tools/audit_dataset_protocol.py'
    if not path.is_file():raise FileNotFoundError(f'缺少仓库原始审计脚本: {path}')
    spec=importlib.util.spec_from_file_location('potsdam_frozen_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def scan_labels(root):
    result={}
    if not root.exists():return result
    pattern=re.compile(r'^top_potsdam_(\d+_\d+)_label\.(tif|tiff)$',re.I)
    for path in sorted(root.rglob('*')):
        match=pattern.match(path.name)
        if not path.is_file() or not match:continue
        tile=match.group(1)
        if tile in result:raise ValueError(f'同一标签目录内有重复tile，拒绝猜测: {result[tile]} / {path}')
        result[tile]=path.resolve()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--labels-all-dir',type=Path)
    p.add_argument('--labels-participants-dir',type=Path)
    p.add_argument('--prepare',action='store_true',help='生成缺少的元数据并执行仓库完整审计；已有文件不覆盖')
    args=p.parse_args();root=args.project_root.expanduser().resolve()
    audit=load_auditor(root)
    raw=root/'data/raw/potsdam/Potsdam'
    all_dir=args.labels_all_dir.resolve() if args.labels_all_dir else raw/'5_Labels_all'
    participant_dir=args.labels_participants_dir.resolve() if args.labels_participants_dir else raw/'5_Labels_for_participants'
    expected=list(audit.EXPECTED_ALL_TILES)
    # This also detects duplicates, missing RGBIR tiles and incorrect folder names.
    rgbroot=audit.locate_rgbir_root(root);rgbmap=audit.build_rgbir_map(rgbroot)
    all_labels=scan_labels(all_dir);participants=scan_labels(participant_dir)
    selected={};sources={};missing=[]
    trainval=set(audit.EXPECTED_TRAIN_TILES+audit.EXPECTED_VAL_TILES)
    for tile in expected:
        # Original public train/val GT preferred; all-label release supplies held-out test GT.
        if tile in trainval and tile in participants:selected[tile]=participants[tile];sources[tile]='participant'
        elif tile in all_labels:selected[tile]=all_labels[tile];sources[tile]='all'
        elif tile in participants:selected[tile]=participants[tile];sources[tile]='participant'
        else:missing.append(tile)
    report=dict(project_root=str(root),rgbir_dir=str(rgbroot),rgbir_tiles=len(rgbmap),
                labels_all_tiles=len(all_labels),participant_tiles=len(participants),missing_labels=missing,
                split={'train':audit.EXPECTED_TRAIN_TILES,'val':audit.EXPECTED_VAL_TILES,'test':audit.EXPECTED_TEST_TILES},
                selected_label_sources=sources,
                note='Uses exact repository frozen tile lists. Only ordinary *_label.tif; no eroded/no-boundary GT.')
    out=root/'outputs/dataset_check/potsdam_category_setup';write_json(out/'inventory.json',report)
    print(f'[RGBIR] {rgbroot} | {len(rgbmap)} tiles')
    print(f'[LABELS] all={len(all_labels)} participants={len(participants)} selected={len(selected)}/38')
    if missing:raise RuntimeError(f'缺少完整标签tile: {missing}。请将两个标签ZIP解压；只保留普通label，不用noBoundary替代。')
    import numpy as np
    import tifffile
    for tile in expected:
        for path,kind in [(rgbmap[tile],'RGBIR'),(selected[tile],'GT')]:
            with tifffile.TiffFile(path) as tf:
                series=tf.series[0];shape=series.shape;dtype=series.dtype
            good=shape==(6000,6000,4) if kind=='RGBIR' else shape in [(6000,6000,3),(6000,6000,4)]
            if not good or dtype!=np.dtype('uint8'):raise ValueError(f'{kind} shape/dtype不符: {path}: {shape}/{dtype}')
    print('[HEADERS PASS] 38 RGBIR + 38 GT；划分为18/6/14。')
    if not args.prepare:
        print('仅检查完成。执行同一命令并加 --prepare，生成缺少的协议并运行完整审计。');return
    meta=root/'data/processed/potsdam';meta.mkdir(parents=True,exist_ok=True)
    proposed={
      'tile_split.json':{'splits':report['split'],'source':'exact frozen lists from repository audit_dataset_protocol.py'},
      'rgbir_semantics.json':{'channel_names':['R','G','B','NIR'],'source':'ISPRS Potsdam 4_Ortho_RGBIR; original repository RGBIR channel convention'},
      'labels_manifest.json':{'tiles':[{'tile_id':t,'canonical_gt_path':selected[t].relative_to(root).as_posix() if selected[t].is_relative_to(root) else str(selected[t]),
                                       'canonical_source':sources[t]} for t in expected]},
    }
    for name,value in proposed.items():
        dest=meta/name
        if dest.exists():print(f'[KEEP EXISTING] {dest}')
        else:
            # Never inject new files into an existing frozen protocol with different hashes.
            if (meta/'dataset_protocol.json').exists():raise RuntimeError('已有冻结协议但其依赖元数据缺失。请从原实验复制缺失文件，不能混用新旧协议。')
            write_json(dest,value);print(f'[CREATED] {dest}')
    print('[AUDIT] 开始原仓库完整审计：颜色映射、采样、D4、滑窗、训练集NIR统计；可能耗时较长。',flush=True)
    audit.run_audit(root)
    # Verify generated metadata can actually be consumed by the unchanged loader.
    sys.path.insert(0,str(root))
    from data_pipeline.potsdam_dataset import FrozenPotsdamSpec
    spec=FrozenPotsdamSpec(root)
    print(f'[READY] frozen protocol valid; NIR mean={spec.nir_mean:.9f} std={spec.nir_std:.9f}')

if __name__=='__main__':main()
