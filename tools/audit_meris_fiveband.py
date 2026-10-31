#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only TLSNMSP raw-ID, exact crop replay and visual diagnosis. No torch needed."""
from __future__ import annotations
import argparse, ast, csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
IDS = tuple(range(1, 8))
PALETTE = np.array([[220,60,60],[160,160,160],[30,120,230],[185,125,65],
                    [20,145,60],[170,220,55],[180,65,190]], dtype=np.uint8)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def resolve(root,rel):
    p=Path(rel.replace('\\','/'))
    return p.resolve() if p.is_absolute() else (root/p).resolve()

def load_label(path):
    with Image.open(path) as im:
        # Palette PNG indices are the IDs. Do not convert palette images to RGB.
        arr=np.array(im)
    if arr.ndim==3:
        if arr.shape[2]>=3 and np.array_equal(arr[:,:,0],arr[:,:,1]) and np.array_equal(arr[:,:,0],arr[:,:,2]):arr=arr[:,:,0]
        else:raise ValueError(f'标签是彩色编码图，不是单通道 raw ID；不要自动猜颜色映射: {path}')
    if arr.ndim!=2 or not np.issubdtype(arr.dtype,np.integer):raise ValueError(f'非法标签格式: {path}: {arr.shape}/{arr.dtype}')
    return arr

def counts7(label):
    return np.array([np.count_nonzero(label==i) for i in IDS],dtype=np.int64)

def manifest(args):
    result={}; seen_ids=set();seen_paths=set()
    for split in ['train','val','test']:
        rows=read_csv(args.protocol_dir/f'{split}.csv')
        if not rows:raise ValueError(f'{split}.csv 为空')
        for row in rows:
            for key in ['sample_id','label_path']:
                if not row.get(key):raise ValueError(f'{split} 缺少字段 {key}')
            sid=row['sample_id'];path=str(resolve(args.data_root,row['label_path']))
            if sid in seen_ids or path in seen_paths:raise ValueError(f'划分存在重复 sample_id 或标签路径: {split}/{sid}')
            seen_ids.add(sid);seen_paths.add(path)
        result[split]=rows
    return result

def names_from_config(args):
    p=args.protocol_dir/'dataset_config.json'
    if p.exists():return read_json(p).get('class_names_train_id_order',[])
    return []

def check_counts(args):
    p=args.output_dir/'counts_summary.json'
    if not p.exists():raise ValueError('请先执行 --stage stats')
    summary=read_json(p)
    if Path(summary['data_root']).resolve()!=args.data_root:raise ValueError('data-root与统计记录不同，请重跑stats')
    for split,h in summary['manifest_sha256'].items():
        if digest(args.protocol_dir/f'{split}.csv')!=h:raise ValueError('数据清单已变化，请重跑 stats')
    if not summary['raw_ids_valid']:raise ValueError('检测到 1..7 以外的标签ID，请先核对映射；详见 invalid_label_ids.csv')
    return summary

def validate_crop_sampler(args):
    """Validate coordinate-relevant statements, not a hash of the whole method.

    This deliberately ignores unrelated formatting, comments and error messages.
    Reads AST only; never imports or executes the user's training script.
    """
    candidates=[ROOT/'tools/train_model_b2_bgr_re_nir5_category.py',
                ROOT/'tools/train_model_b2_bgr_re_nir5.py']
    trainer=args.training_script
    if trainer is None:
        trainer=next((p for p in candidates if p.is_file()),None)
    if trainer is None:
        print('[SAMPLER NOTE] 未找到训练脚本，本次仅按参考规则重放，不能视为已校验的实际采样。')
        return dict(status='reference_only_unverified',script=None)
    trainer=trainer.expanduser().resolve()
    if not trainer.is_file():raise FileNotFoundError(f'训练脚本不存在: {trainer}')
    tree=ast.parse(trainer.read_text(encoding='utf-8-sig'))
    cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MerisFiveBandRandomCropDataset'),None)
    if cls is None:raise ValueError(f'未找到随机裁剪数据集类: {trainer}; 原图统计已保存。')
    fn=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__'),None)
    if fn is None:raise ValueError('缺少 __getitem__，不能验证裁剪重放')
    expected={
        'rng':'np.random.default_rng(self.seed + self.epoch * 1000003 + idx * 9176)',
        'y0':'int(rng.integers(0, h - CROP_SIZE + 1))',
        'x0':'int(rng.integers(0, w - CROP_SIZE + 1))',
        'label_raw':'label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]',
    }
    canonical=lambda n:ast.dump(n,include_attributes=False)
    matched=[]
    for name,expression in expected.items():
        wanted=canonical(ast.parse(expression,mode='eval').body)
        hits=[n for n in fn.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
              and canonical(n.value)==wanted]
        if len(hits)!=1:
            raise ValueError(f'关键裁剪规则不支持: {name}; 文件={trainer}。原图统计已保存，请提供此训练脚本核对。')
        matched.append(hits[0].lineno)
    if matched!=sorted(matched):raise ValueError('裁剪随机数抽取顺序改变，停止重放；原图统计已保存。')
    # Additional draws or reassignments between creation of the RNG and cropping
    # could change the sampled coordinates despite matching individual expressions.
    block=[n for n in fn.body if matched[0]<=n.lineno<=matched[-1]]
    expected_block=ast.parse("""rng = np.random.default_rng(self.seed + self.epoch * 1000003 + idx * 9176)
y0 = int(rng.integers(0, h - CROP_SIZE + 1))
x0 = int(rng.integers(0, w - CROP_SIZE + 1))
bands = bands[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
label_raw = label_raw[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
""").body
    if [canonical(n) for n in block]!=[canonical(n) for n in expected_block]:
        raise ValueError('坐标采样区间存在额外语句/随机抽取，不能精确重放；原图统计已保存。')
    constants=[n.value.value for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='CROP_SIZE' for t in n.targets)
               and isinstance(n.value,ast.Constant)]
    if constants!=[args.crop_size]:raise ValueError(f'裁剪尺寸不同: 脚本={constants}, 审计={args.crop_size}')
    setter=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_epoch'),None)
    expected_setter=ast.parse('def set_epoch(self, epoch: int):\n    self.epoch = int(epoch)').body[0]
    if setter is None:raise ValueError('缺少set_epoch，无法确认轮次采样')
    actual_body=[n for n in setter.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    if [canonical(n) for n in actual_body]!=[canonical(n) for n in expected_setter.body]:
        raise ValueError('set_epoch赋值规则不同，无法确认重放轮次')
    print(f'[SAMPLER PASS] 种子公式、y/x抽取顺序、裁剪切片及尺寸一致: {trainer.name}')
    return dict(status='coordinate_rules_verified',script=str(trainer),sha256=digest(trainer),
                scope='Coordinate sampling and set_epoch rules; whole method hash is not required.')


def raw_counts(args):
    manifests=manifest(args);rows_out=[];invalid=[];totals={};names=names_from_config(args)
    label_hashes={}
    for split,rows in manifests.items():
        total=np.zeros(7,dtype=np.int64);present=np.zeros(7,dtype=np.int64)
        for n,row in enumerate(rows):
            path=resolve(args.data_root,row['label_path']);a=load_label(path);cnt=counts7(a)
            extra={str(int(i)):int(n) for i,n in zip(*np.unique(a,return_counts=True)) if int(i) not in IDS}
            for key,value in extra.items():invalid.append(dict(split=split,sample_id=row['sample_id'],raw_id=key,pixel_count=value))
            total+=cnt;present+=(cnt>0);label_hashes[f"{split}/{row['sample_id']}"]=digest(path)
            rows_out.append(dict(split=split,sample_id=row['sample_id'],width=a.shape[1],height=a.shape[0],
                                 total_pixels=a.size,invalid_pixels=sum(extra.values()),**{f'raw_{i}_pixels':int(cnt[i-1]) for i in IDS}))
            if (n+1)%25==0 or n+1==len(rows):print(f'[COUNTS] {split} {n+1}/{len(rows)}',flush=True)
        totals[split]=dict(images=len(rows),counts=total.tolist(),images_present=present.tolist())
    write_csv(args.output_dir/'per_image_class_counts.csv',rows_out)
    write_csv(args.output_dir/'invalid_label_ids.csv',invalid,['split','sample_id','raw_id','pixel_count'])
    perclass=[]
    for split,t in totals.items():
        for i in IDS:
            perclass.append(dict(split=split,raw_id=i,train_id=i-1,name_in_config=names[i-1] if len(names)==7 else '',
                semantic_status='UNVERIFIED',pixel_count=t['counts'][i-1],pixel_fraction=t['counts'][i-1]/max(1,sum(t['counts'])),
                images_present=t['images_present'][i-1],image_count=t['images']))
    write_csv(args.output_dir/'class_counts_by_split_audited.csv',perclass)
    summary=dict(raw_ids_valid=not invalid,totals=totals,data_root=str(args.data_root),protocol_dir=str(args.protocol_dir),
                 manifest_sha256={s:digest(args.protocol_dir/f'{s}.csv') for s in manifests},label_sha256=label_hashes,
                 semantics='Raw IDs are numeric facts; class names copied from config are NOT verified dataset definitions.')
    write_json(args.output_dir/'counts_summary.json',summary)
    if invalid:raise ValueError('发现非法 raw ID，已经输出清单；停止裁剪重放。')
    print('[COUNTS PASS] 已输出原始像素统计；类别语义仍需人工对照官方说明。')

def crop_origin(seed,epoch0,idx,h,w,crop):
    # Exact formula and draw order from train_model_b2_bgr_re_nir5.py fixed2.
    if min(h,w)<crop:raise ValueError(f'影像 {w}x{h} 小于训练裁剪 {crop}')
    rng=np.random.default_rng(seed+epoch0*1_000_003+idx*9_176)
    return int(rng.integers(0,h-crop+1)),int(rng.integers(0,w-crop+1))

def replay_crops(args):
    summary=check_counts(args);sampler=validate_crop_sampler(args)
    rows=read_csv(args.protocol_dir/'train.csv');entries=[]
    ep_pixels=np.zeros((args.epochs,7),dtype=np.int64);ep_present=np.zeros((args.epochs,7),dtype=np.int64)
    by_image=[]
    for idx,row in enumerate(rows):
        path=resolve(args.data_root,row['label_path'])
        if digest(path)!=summary['label_sha256'][f"train/{row['sample_id']}"]:raise ValueError('标签在统计后发生变化，请重跑 stats')
        label=load_label(path);h,w=label.shape;full=counts7(label);seen=np.zeros(7,dtype=np.int64);pix=np.zeros(7,dtype=np.int64)
        for epoch0 in range(args.epochs):
            y,x=crop_origin(args.seed,epoch0,idx,h,w,args.crop_size);cnt=counts7(label[y:y+args.crop_size,x:x+args.crop_size])
            ep_pixels[epoch0]+=cnt;ep_present[epoch0]+=(cnt>0);seen+=(cnt>0);pix+=cnt
            entries.append(dict(epoch=epoch0+1,epoch0=epoch0,manifest_index=idx,sample_id=row['sample_id'],y=y,x=x,
                **{f'raw_{i}_pixels':int(cnt[i-1]) for i in IDS}))
        for i in IDS:by_image.append(dict(sample_id=row['sample_id'],raw_id=i,full_image_pixels=int(full[i-1]),
            epochs_with_class=int(seen[i-1]),replayed_epochs=args.epochs,cumulative_crop_pixels=int(pix[i-1])))
    write_csv(args.output_dir/'crop_replay_per_sample_epoch.csv',entries)
    write_csv(args.output_dir/'crop_exposure_by_image.csv',by_image)
    epoch_rows=[];class_rows=[]
    for ep in range(args.epochs):
        for i in IDS:epoch_rows.append(dict(epoch=ep+1,raw_id=i,crop_pixels=int(ep_pixels[ep,i-1]),
            crops_with_class=int(ep_present[ep,i-1]),total_crops=len(rows),pixel_fraction=float(ep_pixels[ep,i-1]/(len(rows)*args.crop_size**2))))
    for i in IDS:
        full_present=int(summary['totals']['train']['images_present'][i-1]);items=[r for r in by_image if r['raw_id']==i and r['full_image_pixels']>0]
        class_rows.append(dict(raw_id=i,train_id=i-1,full_train_pixels=summary['totals']['train']['counts'][i-1],
            full_images_with_class=full_present,cumulative_crop_pixels=int(ep_pixels[:,i-1].sum()),
            crops_with_class=int(ep_present[:,i-1].sum()),total_crops=len(rows)*args.epochs,
            epochs_with_zero_class_crops=int((ep_present[:,i-1]==0).sum()),
            full_positive_images_never_seen=sum(r['epochs_with_class']==0 for r in items)))
    write_csv(args.output_dir/'crop_exposure_by_epoch.csv',epoch_rows)
    write_csv(args.output_dir/'crop_exposure_summary.csv',class_rows)
    write_json(args.output_dir/'crop_replay_settings.json',dict(seed=args.seed,epochs=args.epochs,crop_size=args.crop_size,
        mode='Deterministic coordinate replay, NOT observation of completed training.',sampler_verification=sampler,
        augmentation='D4 rotations/flips preserve class counts, so not applied.',
        epoch_note='CSV epoch=1 corresponds to trainer epoch0=0; early stopping truncates this schedule.',
        manifest_sha256=summary['manifest_sha256']['train']))
    print('[CROPS PASS] 裁剪位置重放完成；这不是历史训练运行记录。')

def image_rgb(args,row):
    for key in ['rgb_path','rgb_image_path','image_rgb_path']:
        if row.get(key):
            path=resolve(args.data_root,row[key])
            if path.exists():
                with Image.open(path) as im:return np.array(im.convert('RGB')),f'RGB: {path.name}'
    if not row.get('multispectral_path'):raise ValueError('清单缺少可用 rgb_path 和 multispectral_path')
    import tifffile
    path=resolve(args.data_root,row['multispectral_path']);a=tifffile.imread(path)
    if a.ndim!=3:raise ValueError(f'TIFF布局异常: {a.shape}')
    if a.shape[0]==5:a=np.moveaxis(a,0,-1)
    elif a.shape[-1]!=5:raise ValueError(f'无法确认五波段: {a.shape}')
    rgb=a[:,:,[2,1,0]].astype(np.float32);lo=np.percentile(rgb,2,axis=(0,1));hi=np.percentile(rgb,98,axis=(0,1))
    return (np.clip((rgb-lo)/np.maximum(hi-lo,1),0,1)*255).astype(np.uint8),f'MS R/G/B percentile display: {path.name}'

def panel(rgb,label,target,title):
    color=np.zeros((*label.shape,3),np.uint8)
    for i in IDS:color[label==i]=PALETTE[i-1]
    mask=label==target;overlay=rgb.copy();overlay[mask]=(0.45*rgb[mask]+0.55*np.array([255,0,255])).astype(np.uint8)
    views=[rgb,color,overlay];width=min(600,rgb.shape[1]);height=max(1,round(rgb.shape[0]*width/rgb.shape[1]))
    canvas=Image.new('RGB',(width*3,60+height+55),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,5),title,fill='black');draw.text((8,23),f'Target raw ID={target}; labels are numeric, semantics unverified',fill='black')
    for n,(a,name) in enumerate(zip(views,['Image','Raw ID colors','Target overlay'])):
        im=Image.fromarray(a).resize((width,height),Image.Resampling.NEAREST if n==1 else Image.Resampling.BILINEAR)
        canvas.paste(im,(n*width,60));draw.text((n*width+8,43),name,fill='black')
    for i in IDS:
        x=(i-1)*(width*3//7);draw.rectangle((x,65+height,x+12,77+height),fill=tuple(int(v) for v in PALETTE[i-1]));draw.text((x+16,65+height),f'ID {i}',fill='black')
    return canvas

def export_visual(args):
    summary=check_counts(args);manifests=manifest(args);rank=read_csv(args.output_dir/'per_image_class_counts.csv');review=[]
    for target in args.target_ids:
        # Training evidence only, so model selection does not depend on held-out targets.
        selected=sorted([r for r in rank if r['split']=='train' and int(r[f'raw_{target}_pixels'])>0],
                        key=lambda r:int(r[f'raw_{target}_pixels']),reverse=True)[:args.examples_per_class]
        rows_by_id={r['sample_id']:r for r in manifests['train']}
        for number,info in enumerate(selected):
            row=rows_by_id[info['sample_id']];label_path=resolve(args.data_root,row['label_path'])
            if digest(label_path)!=summary['label_sha256'][f"train/{row['sample_id']}"]:raise ValueError('标签已变化，请重新统计')
            label=load_label(label_path);rgb,source=image_rgb(args,row)
            if rgb.shape[:2]!=label.shape:raise ValueError(f"RGB/标签尺寸不一致，拒绝自动缩放: {row['sample_id']}")
            name=f'raw{target}_example{number+1:02d}'
            out=args.output_dir/'visual_review';out.mkdir(parents=True,exist_ok=True)
            panel(rgb,label,target,f"{row['sample_id']} | {source}").save(out/f'{name}_full.png')
            # Select a rare-positive crop among evenly spaced target pixels, not a random empty crop.
            ys,xs=np.where(label==target);best=None;crop=min(args.crop_size,*label.shape)
            for n in np.linspace(0,len(ys)-1,min(32,len(ys)),dtype=int):
                y=int(np.clip(ys[n]-crop//2,0,label.shape[0]-crop));x=int(np.clip(xs[n]-crop//2,0,label.shape[1]-crop))
                count=int(np.count_nonzero(label[y:y+crop,x:x+crop]==target))
                if best is None or count>best[0]:best=(count,y,x)
            count,y,x=best
            panel(rgb[y:y+crop,x:x+crop],label[y:y+crop,x:x+crop],target,f"{row['sample_id']} | crop x={x} y={y}").save(out/f'{name}_crop.png')
            review.append(dict(raw_id=target,sample_id=row['sample_id'],source_label=row['label_path'],image_source=source,
                full_figure=f'visual_review/{name}_full.png',crop_figure=f'visual_review/{name}_crop.png',
                crop_x=x,crop_y=y,crop_target_pixels=count,official_class_name='',official_evidence='',
                image_label_alignment='UNREVIEWED',semantic_correctness='UNREVIEWED',notes=''))
    if review:
        path=args.output_dir/'manual_label_review.csv'
        if path.exists():path=args.output_dir/'manual_label_review_new.csv' # preserve user annotations
        write_csv(path,review)
    print(f'[VISUAL PASS] {len(review)} 组；请人工填写 manual_label_review.csv，脚本不自动确认地物语义。')

def decision(args):
    summary=check_counts(args);exposure_path=args.output_dir/'crop_exposure_summary.csv'
    if not exposure_path.exists():raise ValueError('请先执行 --stage stats')
    exposure=read_csv(exposure_path);notes=['诊断建议（条件判断，不是已经确认的根因）','']
    for r in exposure:
        i=r['raw_id'];full=int(r['full_train_pixels']);seen=int(r['crops_with_class']);zero=int(r['epochs_with_zero_class_crops'])
        if full==0:msg='训练集无该ID：先核对标签/划分；类别加权无法弥补没有样本。'
        elif seen==0:msg='完整图有该ID但重放裁剪从未出现：优先做单独的类别感知采样实验。'
        else:msg=f'完整训练集 {full} 像素，{seen} 个裁剪包含该ID，{zero} 个重放epoch完全没有该ID。有曝光不代表充分学会；结合占比及人工核对。'
        notes.append(f'raw ID {i}: {msg}')
    notes+=['','先对照官方类别定义检查所有ID，尤其是5/6/7；未经确认不要把两个ID合并当植被覆盖度。',
            '核对通过且训练有曝光后，原五波段 baseline 与 sqrt_inverse CE 单因素对照；Lovasz、采样和架构保持原样。',
            '若加权改善少数类但虚警增多，依据验证集决定权重；不要删掉零IoU类别。',
            '若少数类能预测但边界破碎，再独立评估轻量频率模块；当前不自动叠加类别模块。']
    review=args.output_dir/'manual_label_review.csv'
    if not review.exists() or any(r.get('semantic_correctness') not in ['PASS','通过'] for r in read_csv(review)):
        notes+=['','人工标签核对尚未全部标为PASS；数值ID有效不能证明类别名称或植被定义正确。']
    if args.baseline_metrics and args.weighted_metrics:
        a=read_json(args.baseline_metrics);b=read_json(args.weighted_metrics)
        if a.get('split')!='val' or b.get('split')!='val':raise ValueError('方案选择只比较验证集 metrics_summary.json')
        for k in ['total_pixels','image_count','gt_vegetation_pixels']:
            if a.get(k)!=b.get(k):raise ValueError(f'比较口径不一致: {k}')
        if [r.get('gt_pixels') for r in a['per_class']]!=[r.get('gt_pixels') for r in b['per_class']]:raise ValueError('类别GT不一致')
        notes+=['','加权 vs baseline 验证集比较（请另核对相同样本清单及评价脚本）：']
        for k in ['OA','mIoU','Kappa','per_image_mean_absolute_error_pp']:
            if k in a and k in b:notes.append(f'{k}: {a[k]:.6f} -> {b[k]:.6f}; delta={b[k]-a[k]:+.6f}')
        for x,y in zip(a['per_class'],b['per_class']):notes.append(f"raw ID {x['raw_label_id']} IoU: {x['IoU']:.6f} -> {y['IoU']:.6f}; pred pixels {x['pred_pixels']} -> {y['pred_pixels']}")
        notes+=['单次结果只用于筛选；重要结论需要重复种子。']
    path=args.output_dir/'next_step_report.txt';path.write_text('\n'.join(notes),encoding='utf-8-sig');print('\n'.join(notes))

def get_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-script',type=Path,help='指定要核对的原基线或类别感知训练脚本；默认优先类别感知版本')
    p.add_argument('--stage',choices=['counts','stats','visual','plan','all'],default='stats')
    p.add_argument('--data-root',type=Path,default=Path(r'D:\MRISE\dataset\TLSNMSP'))
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'data/processed/meris_tlsnmsp')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/diagnostics/meris_fiveband')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--crop-size',type=int,default=512);p.add_argument('--target-ids',type=int,nargs='+',default=[1,7])
    p.add_argument('--examples-per-class',type=int,default=3)
    p.add_argument('--baseline-metrics',type=Path);p.add_argument('--weighted-metrics',type=Path)
    args=p.parse_args()
    if min(args.epochs,args.crop_size,args.examples_per_class)<=0 or any(i not in IDS for i in args.target_ids):p.error('正整数参数或 raw ID 非法')
    if bool(args.baseline_metrics)!=bool(args.weighted_metrics):p.error('需同时提供 baseline-metrics 和 weighted-metrics')
    for k in ['data_root','protocol_dir','output_dir']:setattr(args,k,getattr(args,k).expanduser().resolve())
    return args

def main():
    args=get_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage in ['counts','stats','all']:raw_counts(args)
    if args.stage in ['stats','all']:replay_crops(args)
    if args.stage in ['visual','all']:export_visual(args)
    if args.stage in ['plan','all']:decision(args)

if __name__=='__main__':main()
