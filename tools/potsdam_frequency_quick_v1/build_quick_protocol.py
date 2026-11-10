#!/usr/bin/env python3
"""Create an isolated, deterministic Potsdam subset; never alter raw data."""
import argparse, copy, hashlib, itertools, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parent/'source_protocol')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rgbir-dir', type=Path, help='Optional: recalculate NIR normalization using only the six subset training images (requires numpy,tifffile).')
    args=ap.parse_args()
    src, out=args.source.resolve(),args.out.resolve()
    if out.exists():
        raise SystemExit('Output already exists. Choose a NEW output directory; no overwrite is performed.')
    p=read(src/'dataset_protocol.json'); s=read(src/'tile_split.json'); m=read(src/'labels_manifest.json')
    assert p['split']==s['splits'], 'Source split files disagree'
    classes=s['classes']; tiles={t['tile_id']:t for t in m['tiles']}
    for a,b in itertools.combinations(s['splits'],2):
        assert not set(s['splits'][a])&set(s['splits'][b]), 'Source leakage'
    def stats(ids):
        counts={c:sum(tiles[t]['standard_color_counts'][c] for t in ids) for c in classes}
        n=sum(counts.values())
        return dict(tile_count=len(ids),pixel_count=n,class_pixel_counts=counts,
                    class_fractions={c:counts[c]/n for c in classes})
    # Exhaustive deterministic choice, separate original split pools. Test labels
    # never participate in subset selection. Tie-break: numeric tile IDs.
    selection={}
    def choose(split,k):
        pool=sorted(s['splits'][split],key=lambda t:tuple(map(int,t.split('_'))))
        target=stats(pool)['class_fractions']
        ranked=[]
        for ids in itertools.combinations(pool,k):
            st=stats(ids)
            if min(st['class_pixel_counts'].values())<=0: continue
            errors=[abs(st['class_fractions'][c]-target[c]) for c in classes]
            ranked.append(((max(errors),sum(errors)/len(errors)),ids))
        if not ranked: raise ValueError('No candidate covers all classes')
        score,ids=min(ranked,key=lambda v:v[0])
        selection[split]={'method':'exhaustive_minimax_absolute_class_fraction_error',
                          'target':'original '+split+' split distribution',
                          'pool_size':len(pool),'subset_size':k,'candidate_count':len(ranked),
                          'max_abs_fraction_error':score[0],'mean_abs_fraction_error':score[1]}
        return list(ids)
    splits={'train':choose('train',6),'val':choose('val',2),'test':s['splits']['test']}
    active=set(sum(splits.values(),[])); unused=[t['tile_id'] for t in m['tiles'] if t['tile_id'] not in active]
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    new=copy.deepcopy(s)
    new.update(created_at=now,status='METADATA_CHECKS_PASS_RAW_FILES_NOT_VERIFIED',
        split_policy='Frequency quick experiment v1. Select 6 original train and 2 original val tiles independently using class counts; keep original 14 test tiles frozen. Unselected development tiles are unused. Do not consult test metrics during module selection.',
        counts={**{k:len(v) for k,v in splits.items()},'total':len(active),'unused':len(unused),'source_total':len(tiles)},
        splits=splits, unused_tiles=unused, subset_selection=selection,
        inputs={'parent_tile_split_sha256':sha(src/'tile_split.json'), 'labels_manifest_sha256':sha(src/'labels_manifest.json')})
    new.pop('validation_selection',None)
    new['statistics']={k:stats(v) for k,v in splits.items()}
    new['statistics']['unused']=stats(unused)
    new['tiles']=[{**t,'split':next((k for k,v in splits.items() if t['tile_id'] in v),'unused')} for t in s['tiles']]
    new['checks']={'split_disjoint':all(not set(a)&set(b) for a,b in itertools.combinations(splits.values(),2)),
        'train_subset_of_parent_train':set(splits['train'])<=set(s['splits']['train']),
        'val_subset_of_parent_val':set(splits['val'])<=set(s['splits']['val']),
        'test_unchanged':splits['test']==s['splits']['test'],
        'six_classes_present_in_train_and_val':all(min(stats(splits[k])['class_pixel_counts'].values())>0 for k in ('train','val')),
        'raw_image_audit_performed':False}
    q=copy.deepcopy(p)
    q.update(protocol_version='Potsdam Frequency Quick Protocol v1',script_version='quick-1.0',
             status='METADATA_CHECKS_PASS_REQUIRES_TRAINER_INTEGRATION',split=splits)
    q['parent_protocol_sha256']=sha(src/'dataset_protocol.json')
    q['audit_fingerprints']={}
    q['invalidated_parent_audits']=['D4 real crop audit','sampling schedule audit','parent raw file audit status']
    q['unused_tiles']=unused
    q['training_sampling'].update(crops_per_tile_per_epoch=32,samples_per_epoch=192)
    q['training_sampling'].pop('audit_schedule_sha256_epochs_0_1_2',None)
    q['data_source']['split']='tile_split.json only; select splits.train/val/test; never infer membership from labels_manifest or all tiles'
    # Original fixed training normalization has no validation/test leakage, but
    # includes original training tiles unused in this quick subset. Say so.
    q['normalization']['nir']['scope']='fixed reference from parent 18 training tiles; NOT recalculated from subset; no parent val/test pixels'
    q['normalization']['nir']['reference_train_tiles']=s['splits']['train']
    q['normalization']['nir']['subset_recomputed']=False
    if args.rgbir_dir:
        import numpy as np
        import tifffile
        n=0; total=0.; total2=0.
        for tid in splits['train']:
            path=args.rgbir_dir/('top_potsdam_'+tid+'_RGBIR.tif')
            arr=tifffile.imread(path)
            if arr.shape!=(6000,6000,4) or arr.dtype!=np.uint8:
                raise ValueError(f'Unexpected image shape/dtype: {path}: {arr.shape}/{arr.dtype}')
            for y in range(0,6000,256):
                a=arr[y:y+256,:,3].astype(np.float64)/255.
                n+=a.size; total+=float(a.sum());total2+=float(np.square(a).sum())
            del arr
        mean=total/n;std=(max(0,total2/n-mean**2))**.5
        if std<=0: raise ValueError('Invalid NIR std')
        q['normalization']['nir'].update(mean=mean,std=std,pixel_count=n,sum=total,sumsq=total2,
            scope='train-only full pixels from six subset training tiles',reference_train_tiles=splits['train'],subset_recomputed=True)
    q['experiment_plan']={'purpose':'module screening; not final full-data benchmark',
        'suggested_epochs':30,'validation_every_epochs':5,'final_epoch_validation':True,
        'test_during_module_selection':False,'seed':20260917,
        'baseline_and_frequency_must_share':['split','sampling','augmentation','normalization','pretrained weights','training budget','evaluation fusion'],
        'trainer_auto_configured':False}
    # Only create destination after all required reads and optional NIR work pass.
    out.mkdir(parents=True)
    for f in ['labels_manifest.json','rgbir_semantics.json']:
        (out/f).write_bytes((src/f).read_bytes())
    write(out/'tile_split.json',new)
    q['source_metadata_sha256']={f:sha(out/f) for f in ['labels_manifest.json','rgbir_semantics.json','tile_split.json']}
    write(out/'dataset_protocol.json',q)
    for k,v in splits.items(): (out/(k+'.txt')).write_text('\n'.join(v)+'\n',encoding='utf-8')
    (out/'unused.txt').write_text('\n'.join(unused)+'\n',encoding='utf-8')
    report={'splits':splits,'unused':unused,'checks':new['checks'],'selection':selection,
       'samples_per_epoch':192,'validation_windows':512,'full_test_windows':3584,
       'nir_subset_recomputed':q['normalization']['nir']['subset_recomputed'],
       'training_run_performed':False,'raw_label_files_reaudited':False}
    write(out/'quick_protocol_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Created:',out)
if __name__=='__main__':main()
