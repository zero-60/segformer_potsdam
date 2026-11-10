"""Full 18/6/14 loader; exact SHA256 coordinate/D4 streams of the frozen protocol."""
import hashlib,json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def randbelow(parts,n,stream):
    limit=(1<<256)-((1<<256)%n);counter=0
    while True:
        v=int.from_bytes(hashlib.sha256(('|'.join(map(str,parts))+f'|{stream}|{counter}').encode()).digest(),'big')
        if v<limit:return v%n
        counter+=1

def d4(a,code):
    a=np.rot90(a,code%4,axes=(0,1))
    return np.ascontiguousarray(np.fliplr(a) if code>=4 else a)

def encode(a,colors):
    out=np.full(a.shape[:2],255,np.uint8)
    for i,c in enumerate(colors):out[np.all(a==c,axis=-1)]=i
    if np.any(out==255):raise ValueError('Unknown GT color: canonical label required')
    return out

class Spec:
    def __init__(self,protocol,raw_root,rgbir_dir):
        self.directory=Path(protocol);self.p=read(self.directory/'dataset_protocol.json')
        self.split=read(self.directory/'tile_split.json')['splits']
        self.manifest=read(self.directory/'labels_manifest.json')
        self.raw=Path(raw_root);self.rgbir=Path(rgbir_dir)
        if self.rgbir.name!='4_Ortho_RGBIR':raise ValueError('RGBIR directory must be named 4_Ortho_RGBIR')
        if self.p['split']!=self.split:raise ValueError('Split files disagree')
        for name,h in self.p['source_metadata_sha256'].items():
            if digest(self.directory/name)!=h:raise ValueError('Metadata hash mismatch: '+name)
        sem=read(self.directory/'rgbir_semantics.json')
        if sem['channel_order']!=['R','G','B','NIR']:raise ValueError('Unexpected channel order')
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if set(self.split[a])&set(self.split[b]):raise ValueError('Split leakage')
        self.tiles={t['tile_id']:t for t in self.manifest['tiles']}
        self.colors=np.array([c['rgb'] for c in self.p['labels']['classes']],np.uint8)
        self.sampling=self.p['training_sampling'];self.crop=self.sampling['crop_size'][0]
        self.seed=self.sampling['data_seed'];self.per_tile=self.sampling['crops_per_tile_per_epoch']
        if self.crop!=512 or len(self.colors)!=6:raise ValueError('Expected 512 crop, six classes')
        if self.sampling['samples_per_epoch']!=len(self.split['train'])*self.per_tile:raise ValueError('Wrong sample budget')
        if {k:len(v) for k,v in self.split.items()}!={'train':18,'val':6,'test':14}:
            raise ValueError('Full trainer requires the frozen 18/6/14 protocol, not the quick subset')
        members=[t for k in ('train','val','test') for t in self.split[k]]
        if len(set(members))!=38 or set(members)!=set(self.tiles):
            raise ValueError('Split must contain each of the 38 canonical tiles exactly once')
        if self.per_tile!=64 or self.sampling['samples_per_epoch']!=1152:
            raise ValueError('Full protocol requires 64 crops per train tile, 1152 samples/epoch')
        if self.p['normalization']['nir']['pixel_count']!=18*6000*6000:
            raise ValueError('NIR normalization must use all 18 training tiles')
        self.norm=self.p['normalization'];self.cache=OrderedDict()
        self.hash=digest(self.directory/'dataset_protocol.json')
    def array(self,p):
        key=str(p)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        try:a=tifffile.memmap(p,mode='r')
        except (ValueError,OSError):a=tifffile.imread(p)
        self.cache[key]=a
        while len(self.cache)>4:self.cache.popitem(last=False)
        return a
    def image(self,tid):
        a=self.array(self.rgbir/f'top_potsdam_{tid}_RGBIR.tif')
        if a.shape!=(6000,6000,4) or a.dtype!=np.uint8:raise ValueError(f'Invalid RGBIR {tid}: {a.shape}/{a.dtype}')
        return a
    def gt(self,tid):
        a=self.array(self.raw/self.tiles[tid]['gt_relpath'])
        if a.shape!=(6000,6000,3) or a.dtype!=np.uint8:raise ValueError('Invalid canonical RGB GT: '+tid)
        return a
    def normalize(self,a):
        a=np.array(a,dtype=np.float32,copy=True)/255.
        r=self.norm['rgb'];n=self.norm['nir']
        rgb=(a[...,:3]-np.array(r['mean'],np.float32))/np.array(r['std'],np.float32)
        nir=(a[...,3:4]-n['mean'])/n['std']
        return torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(nir.transpose(2,0,1).copy())

class TrainSet(Dataset):
    def __init__(self,spec):self.spec=spec;self.epoch=0
    def __len__(self):return len(self.spec.split['train'])*self.spec.per_tile
    def __getitem__(self,index):
        s=self.spec;tid=s.split['train'][index//s.per_tile];slot=index%s.per_tile
        gt=s.gt(tid);c=s.crop;best=None
        for candidate in range(s.sampling['max_candidates']):
            parts=(s.seed,self.epoch,tid,slot,candidate)
            x=randbelow(parts,6000-c+1,'x');y=randbelow(parts,6000-c+1,'y')
            lab=encode(gt[y:y+c,x:x+c],s.colors)
            ratio=np.bincount(lab.ravel(),minlength=6).max()/lab.size
            row=(ratio,x,y,lab)
            if best is None or ratio<best[0]:best=row
            if ratio<=s.sampling['cat_max_ratio']:best=row;break
        _,x,y,lab=best
        code=randbelow((s.seed,self.epoch,tid,slot),8,'d4')
        rgb,nir=s.normalize(d4(s.image(tid)[y:y+c,x:x+c],code))
        return rgb,nir,torch.from_numpy(d4(lab,code).astype(np.int64))

def starts(length,crop,stride):
    a=list(range(0,length-crop+1,stride))
    if a[-1]!=length-crop:a.append(length-crop)
    return a

# v2: persistent uint8 RGBIR/label-ID caches; exact same coordinates and D4.
import os,time
from cmx_full_runtime import close_mapping,duration,heartbeat,write_json

def _signature(s,tid):
    image=s.rgbir/f'top_potsdam_{tid}_RGBIR.tif';gt=s.raw/s.tiles[tid]['gt_relpath']
    def stamp(p):
        st=p.stat();return [str(p.resolve()),st.st_size,st.st_mtime_ns]
    return {'version':2,'tile':tid,'protocol':s.hash,'image':stamp(image),'gt':stamp(gt),
            'gt_sha256':s.tiles[tid]['file_sha256'],'colors':s.colors.tolist()}

class FastSpec(Spec):
    def __init__(self,protocol,raw_root,rgbir_dir,cache_dir,memory='ram'):
        super().__init__(protocol,raw_root,rgbir_dir)
        self.cache_dir=Path(cache_dir);self.memory=memory;self.fast_arrays={}
        self.eval_batch_size=1;self.fusion_storage='ram'

    def paths(self,tid):
        return [self.cache_dir/f'{tid}_{suffix}' for suffix in ['rgbir.npy','labels.npy','ready.json']]

    def check_sources(self,split):
        missing=[]
        for tid in self.split[split]:
            for p in (self.rgbir/f'top_potsdam_{tid}_RGBIR.tif',self.raw/self.tiles[tid]['gt_relpath']):
                if not p.is_file():missing.append(str(p))
        if missing:raise FileNotFoundError('Missing canonical data files:\n'+'\n'.join(missing))
        print(f'[PATH CHECK] {split}: {len(self.split[split])} RGBIR/GT pairs found',flush=True)

    def ready(self,tid):
        a,b,meta=self.paths(tid)
        if not all(p.is_file() for p in (a,b,meta)):return False
        try:
            d=read(meta)
            if d['signature']!=_signature(self,tid) or d['sizes']!=[a.stat().st_size,b.stat().st_size]:return False
            for p,shape in ((a,(6000,6000,4)),(b,(6000,6000))):
                x=np.load(p,mmap_mode='r',allow_pickle=False)
                ok=x.shape==shape and x.dtype==np.uint8
                close_mapping(x)
                if not ok:return False
            return True
        except (OSError,ValueError,KeyError,EOFError):return False

    def prepare(self,split):
        self.check_sources(split);self.cache_dir.mkdir(parents=True,exist_ok=True)
        begin=time.monotonic()
        for index,tid in enumerate(self.split[split],1):
            if self.ready(tid):
                print(f'[CACHE HIT] {split} {index}/{len(self.split[split])} tile={tid}',flush=True)
                continue
            a,b,meta=self.paths(tid);signature=_signature(self,tid)
            print(f'[CACHE BUILD] {split} {index}/{len(self.split[split])} tile={tid}',flush=True)
            with heartbeat(f'cache {split} tile={tid}'):
                if digest(self.raw/self.tiles[tid]['gt_relpath'])!=self.tiles[tid]['file_sha256']:
                    raise ValueError('Canonical GT SHA256 mismatch: '+tid+'; restore the manifest-selected TIFF; do not substitute 5_Labels_all')
                tag=f'.{os.getpid()}.tmp';ta=Path(str(a)+tag);tb=Path(str(b)+tag)
                mapped=None
                try:
                    print(f'[CACHE READ] tile={tid} RGBIR',flush=True)
                    image=super().image(tid)
                    mapped=np.lib.format.open_memmap(ta,mode='w+',dtype=np.uint8,shape=image.shape)
                    for y in range(0,6000,256):mapped[y:y+256]=image[y:y+256]
                    mapped.flush();close_mapping(mapped);mapped=None
                    print(f'[CACHE READ] tile={tid} canonical label',flush=True)
                    gt=super().gt(tid)
                    mapped=np.lib.format.open_memmap(tb,mode='w+',dtype=np.uint8,shape=(6000,6000))
                    for y in range(0,6000,256):
                        mapped[y:y+256]=encode(gt[y:y+256],self.colors)
                        if y%1536==0 or y+256>=6000:print(f'[CACHE LABEL] tile={tid} rows={min(y+256,6000)}/6000',flush=True)
                    mapped.flush();close_mapping(mapped);mapped=None
                    if signature!=_signature(self,tid):raise RuntimeError('Source changed during cache build')
                    ta.replace(a);tb.replace(b)
                    write_json(meta,{'signature':signature,'sizes':[a.stat().st_size,b.stat().st_size]})
                finally:
                    if mapped is not None:close_mapping(mapped)
                    self.cache.clear()
                    for p in (ta,tb):p.unlink(missing_ok=True)
            elapsed=time.monotonic()-begin
            print(f'[CACHE BUILT] tile={tid} {split}_cache_eta={duration(elapsed/index*(len(self.split[split])-index))}',flush=True)

    def preload(self,split):
        for index,tid in enumerate(self.split[split],1):
            if not self.ready(tid):raise RuntimeError('Cache missing/stale: '+tid)
            a,b,_=self.paths(tid)
            with heartbeat(f'preload tile={tid}'):
                self.fast_arrays[tid]=(np.load(a,mmap_mode=None if self.memory=='ram' else 'r'),
                                       np.load(b,mmap_mode=None if self.memory=='ram' else 'r'))
            print(f'[PRELOAD] {split} {index}/{len(self.split[split])} tile={tid} mode={self.memory}',flush=True)
        print(f'[CACHE READY] mode={self.memory} train_arrays_GiB={sum(a.nbytes+b.nbytes for a,b in self.fast_arrays.values())/2**30:.3f}',flush=True)

    def release_tile(self,tid):
        if tid in self.split['train']:return
        pair=self.fast_arrays.pop(tid,None)
        if pair is not None:
            for a in pair:close_mapping(a)

    def _pair(self,tid):
        if tid not in self.fast_arrays:
            if not self.ready(tid):raise RuntimeError('Cache missing; run preparation first: '+tid)
            a,b,_=self.paths(tid)
            self.fast_arrays[tid]=(np.load(a,mmap_mode='r'),np.load(b,mmap_mode='r'))
        return self.fast_arrays[tid]

    def image(self,tid):return self._pair(tid)[0]
    def label_ids(self,tid):return self._pair(tid)[1]

class FastTrainSet(TrainSet):
    def __getitem__(self,key):
        # Epoch travels in each sampler key, safe with persistent workers/prefetch.
        epoch,index=key if isinstance(key,tuple) else (self.epoch,key)
        s=self.spec;tid=s.split['train'][index//s.per_tile];slot=index%s.per_tile
        gt=s.label_ids(tid);c=s.crop;best=None
        for candidate in range(s.sampling['max_candidates']):
            parts=(s.seed,epoch,tid,slot,candidate)
            x=randbelow(parts,6000-c+1,'x');y=randbelow(parts,6000-c+1,'y')
            lab=gt[y:y+c,x:x+c]
            ratio=np.bincount(lab.ravel(),minlength=6).max()/lab.size
            row=(ratio,x,y,lab)
            if best is None or ratio<best[0]:best=row
            if ratio<=s.sampling['cat_max_ratio']:best=row;break
        _,x,y,lab=best;code=randbelow((s.seed,epoch,tid,slot),8,'d4')
        rgb,nir=s.normalize(d4(s.image(tid)[y:y+c,x:x+c],code))
        return rgb,nir,torch.from_numpy(d4(lab,code).astype(np.int64))

class EpochSampler:
    def __init__(self,size,seed):self.size=size;self.seed=seed;self.epoch=0
    def __len__(self):return self.size
    def __iter__(self):
        order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter((self.epoch,i) for i in order)

def worker_init(_):
    torch.set_num_threads(1)
