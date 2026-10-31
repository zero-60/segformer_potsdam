#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate audited counts and launch original baseline with weighted CE only."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from audit_meris_fiveband import ROOT, digest, read_json, read_csv, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,default=ROOT/'outputs/diagnostics/meris_fiveband')
    p.add_argument('--baseline-run',type=Path,default=ROOT/'outputs/training/meris_b2_bgr_re_nir5_seed20261030')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/training/meris_b2_fiveband_weighted_ce')
    p.add_argument('--training-script',type=Path,default=ROOT/'tools/train_model_b2_bgr_re_nir5.py')
    p.add_argument('--epochs',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--early-stopping-patience',type=int,default=10)
    p.add_argument('--resume',default='none');p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--run',action='store_true',help='不加此参数仅生成命令；加此参数才启动训练')
    args=p.parse_args()
    args.audit_dir=args.audit_dir.resolve();args.baseline_run=args.baseline_run.resolve();args.output_dir=args.output_dir.resolve()
    if args.output_dir==args.baseline_run:raise ValueError('加权实验须使用独立输出目录')
    audit=read_json(args.audit_dir/'counts_summary.json')
    if not audit['raw_ids_valid']:raise ValueError('原始ID检查未通过')
    protocol=Path(audit['protocol_dir']);data=Path(audit['data_root'])
    if digest(protocol/'train.csv')!=audit['manifest_sha256']['train']:raise ValueError('train.csv已变化，请重新诊断')
    from audit_meris_fiveband import resolve
    for row in read_csv(protocol/'train.csv'):
        if digest(resolve(data,row['label_path']))!=audit['label_sha256'][f"train/{row['sample_id']}"]:
            raise ValueError('训练标签已变化，请重新执行stats')
    rows=read_csv(protocol/'class_pixel_counts_by_split.csv');counts=np.zeros(7,np.int64);seen=set()
    for row in rows:
        if row['split']!='train':continue
        i=int(row['train_label_id'])
        if i not in range(7) or i in seen:raise ValueError('协议类别统计有越界或重复train_label_id')
        value=float(row['pixel_count'])
        if not np.isfinite(value) or value<0 or value!=int(value):raise ValueError('协议像素数非法')
        counts[i]=int(value);seen.add(i)
    if seen!=set(range(7)) or counts.tolist()!=audit['totals']['train']['counts']:
        raise ValueError('原协议类别像素统计与本次原图扫描不一致；请先核对，脚本不会覆盖协议文件。')
    if np.any(counts<=0):raise ValueError('训练集有零像素类别；不能直接进行此七类加权实验')
    basefile=args.baseline_run/'protocol.json'
    if not basefile.exists():raise FileNotFoundError('缺少原基线protocol.json；请用--baseline-run指定真实实验目录')
    base=read_json(basefile)
    if base.get('train_manifest_sha256')!=audit['manifest_sha256']['train']:raise ValueError('基线训练清单与当前清单不同')
    bands=base.get('input_bands',base.get('model',{}).get('input_bands'))
    if bands!=['Blue','Green','Red','RedEdge','NIR']:raise ValueError('baseline-run不是预期的五波段基线')
    if base.get('crop_size',512)!=512:raise ValueError('原基线crop_size不是512')
    if 'category' in base.get('model',{}).get('architecture','').lower():raise ValueError('请指定原五波段基线，不能指定类别感知实验')
    if not args.training_script.is_file():raise FileNotFoundError(args.training_script)
    source=args.training_script.read_text(encoding='utf-8-sig')
    if 'sqrt_inverse' not in source or 'MerisFiveBandRandomCropDataset' not in source:raise ValueError('训练脚本接口与fixed2版本不匹配')
    epochs=args.epochs if args.epochs is not None else int(base.get('epochs',40))
    seed=args.seed if args.seed is not None else int(base.get('seed',20261029))
    replay=read_json(args.audit_dir/'crop_replay_settings.json')
    if replay['crop_size']!=512:raise ValueError('原训练脚本固定512裁剪，请按--crop-size 512重跑stats')
    if replay['seed']!=seed or replay['epochs']<epochs:raise ValueError(f'请先用--seed {seed} --epochs {epochs}重跑stats，以覆盖本次裁剪计划')
    if epochs<=0 or args.early_stopping_patience<0:p.error('epochs/early-stopping-patience非法')
    weights=np.sqrt(counts.sum()/counts.astype(float));weights/=weights.mean();weights=np.clip(weights,.5,3.);weights/=weights.mean()
    command=[sys.executable,str(args.training_script.resolve()),'--data-root',str(data),'--protocol-dir',str(protocol),
             '--output-dir',str(args.output_dir),'--class-weight-mode','sqrt_inverse','--epochs',str(epochs),
             '--seed',str(seed),'--early-stopping-patience',str(args.early_stopping_patience),'--resume',args.resume]
    for key,default in [('batch_size',1),('grad_accum_steps',4),('base_lr',3e-5),('new_lr',3e-4),('weight_decay',.01)]:
        command += ['--'+key.replace('_','-'),str(base.get(key,default))]
    command+=['--lovasz-weight',str(base.get('loss',{}).get('lovasz_softmax',.5)),'--num-workers','0']
    pretrained=base.get('model',{}).get('pretrained_source')
    if pretrained:command+=['--pretrained-checkpoint',pretrained]
    if base.get('amp') is False:command+=['--no-amp']
    if args.smoke_test:command+=['--smoke-test']
    plan=dict(command=command,class_weights=weights.tolist(),counts=counts.tolist(),baseline_protocol=base,
              training_script_sha256=digest(args.training_script),train_manifest_sha256=audit['manifest_sha256']['train'],
              change='Only CE class weighting; existing Lovasz weight, original five-band architecture and crop sampler retained.',
              caveat='Copy unchanged settings not stored in baseline protocol manually; defaults for warmup/early stopping must match.',
              semantics='Numeric IDs verified only; review official semantic names separately.')
    write_json(args.audit_dir/'weighted_training_plan.json',plan)
    print('[WEIGHTS raw ID 1..7]',[round(x,6) for x in weights]);print(subprocess.list2cmdline(command))
    print('语义检查仍需人工完成；覆盖度合并ID必须以官方定义为准。')
    if args.run:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        launch=args.output_dir/'weighted_launch.json'
        if launch.exists():
            old=read_json(launch)
            for key in ['training_script_sha256','train_manifest_sha256','class_weights']:
                if old[key]!=plan[key]:raise ValueError(f'输出目录已有不兼容实验: {key}')
        write_json(launch,plan)
        subprocess.run(command,check=True)
    else:print('[PLAN ONLY] 未启动训练；检查命令后添加 --run。')

if __name__=='__main__':main()
