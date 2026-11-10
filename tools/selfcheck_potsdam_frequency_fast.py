"""Synthetic equivalence checks; no real imagery/pretrained weights needed."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pf_data import Spec,TrainSet,FastTrainSet,EpochSampler,worker_init
from pf_model import PotsdamModel
from train_potsdam_frequency_fast import seed
from transformers import SegformerConfig

class FakeSpec:
    split={'train':['2_12','4_12','5_10','6_9','6_11','7_9']}
    per_tile=32;crop=512;seed=20260917
    sampling={'max_candidates':10,'cat_max_ratio':.75}
    norm={'rgb':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'nir':{'mean':.385439,'std':.146142}}
    colors=np.array([[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]],np.uint8)
    normalize=Spec.normalize
    def image(self,_):return np.broadcast_to(np.array([60,80,90,130],np.uint8),(6000,6000,4))
    def gt(self,_):return np.broadcast_to(self.colors[np.arange(6000)%6][None],(6000,6000,3))
    def label_ids(self,_):return np.broadcast_to((np.arange(6000)%6).astype(np.uint8)[None],(6000,6000))

def numpy_collate(batch):
    # Avoid torch FD-sharing sockets in restricted local test runtimes.
    return tuple(np.stack([row[i].numpy() for row in batch]) for i in range(3))

def main():
    torch.set_num_threads(2)
    s=FakeSpec();old=TrainSet(s);new=FastTrainSet(s)
    for ep in [0,1,15]:
        old.epoch=ep
        for idx in [0,31,32,191]:
            assert all(torch.equal(a,b) for a,b in zip(old[idx],new[(ep,idx)]))
    sampler=EpochSampler(8,20260917)
    loader=DataLoader(new,batch_size=1,sampler=sampler,num_workers=2,persistent_workers=True,
                     worker_init_fn=worker_init,multiprocessing_context='fork',prefetch_factor=2,collate_fn=numpy_collate)
    for ep in [0,1,15]:
        sampler.epoch=ep
        for item,(_,idx) in zip(loader,list(sampler)):
            expected=new[(ep,idx)]
            assert all(np.array_equal(a[0],b.numpy()) for a,b in zip(item,expected))
    del loader
    cfg=SegformerConfig(num_labels=6,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],
        num_attention_heads=[1,2,4,8],sr_ratios=[8,4,2,1],decoder_hidden_size=16)
    seed(3);model=PotsdamModel(config=cfg,frequency=True)
    out=model(torch.randn(1,3,64,64),torch.randn(1,1,64,64));out.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: old/new crops, D4, normalization and labels exactly equal; persistent workers receive correct epochs; unchanged model forward/backward')
if __name__=='__main__':main()
