"""Dual SegFormer B2 + fixed NIR residual + optional content-gated FFT residual.
Simplified experimental design inspired by FreqFusion/FreDNet; NOT a reproduction.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class Adapter(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,max(32,c//4),1),nn.GELU(),nn.Conv2d(max(32,c//4),c,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)

class FrequencyResidual(nn.Module):
    def __init__(self,c,cutoff=.15):
        super().__init__(); self.cutoff=cutoff
        self.low=nn.Conv2d(c,c,3,padding=1,groups=c,padding_mode='reflect')
        self.high=nn.Conv2d(c,c,3,padding=1,groups=c,padding_mode='reflect')
        self.gate=nn.Conv2d(2*c,2*c,1,groups=c)
        self.out=nn.Conv2d(c,c,1)
        # Small nonzero projection gives frequency filters/gates gradients from step 1.
        nn.init.normal_(self.out.weight,std=1e-3);nn.init.zeros_(self.out.bias)
        self.alpha=nn.Parameter(torch.tensor(.1))
    def decompose(self,x):
        # AMP-safe FFT with reflect extension to limit periodic boundary artifacts.
        h,w=x.shape[-2:];pad=min(8,h-1,w-1)
        with torch.autocast(device_type=x.device.type,enabled=False):
            z=F.pad(x.float(),(pad,pad,pad,pad),mode='reflect')
            H,W=z.shape[-2:]
            fy=torch.fft.fftfreq(H,device=x.device)[:,None]
            fx=torch.fft.rfftfreq(W,device=x.device)[None,:]
            mask=torch.exp(-(fx.square()+fy.square())/(2*self.cutoff**2))
            low=torch.fft.irfft2(torch.fft.rfft2(z,norm='ortho')*mask,s=(H,W),norm='ortho')
            low=low[...,pad:pad+h,pad:pad+w]
            high=x.float()-low
        return low.to(x.dtype),high.to(x.dtype)
    def forward(self,x):
        lo,hi=self.decompose(x)
        # Interleave low/high per channel for the grouped content gate.
        paired=torch.stack((lo,hi),dim=2).flatten(1,2)
        gates=self.gate(paired).reshape(x.shape[0],x.shape[1],2,*x.shape[-2:]).softmax(dim=2)
        delta=gates[:,:,0]*self.low(lo)+gates[:,:,1]*self.high(hi)
        return x+self.alpha.tanh()*self.out(F.gelu(delta))

class PotsdamModel(nn.Module):
    def __init__(self,checkpoint=None,frequency=True,cutoff=.15,config=None):
        super().__init__()
        from transformers import SegformerConfig,SegformerForSemanticSegmentation
        if config is None:
            config=SegformerConfig.from_pretrained(str(checkpoint),local_files_only=True)
            if tuple(config.hidden_sizes)!=(64,128,320,512): raise ValueError('Expected SegFormer-B2 checkpoint')
            config.num_labels=6; config.id2label={i:str(i) for i in range(6)};config.label2id={str(i):i for i in range(6)}
            base=SegformerForSemanticSegmentation.from_pretrained(str(checkpoint),config=config,
                ignore_mismatched_sizes=True,local_files_only=True)
        else:
            # Offline random tiny configuration for local integration tests only.
            base=SegformerForSemanticSegmentation(config)
        self.rgb_encoder=base.segformer;self.nir_encoder=copy.deepcopy(base.segformer)
        old=self.nir_encoder.encoder.patch_embeddings[0].proj
        new=nn.Conv2d(1,old.out_channels,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(1,keepdim=True))
            if old.bias is not None:new.bias.copy_(old.bias)
        self.nir_encoder.encoder.patch_embeddings[0].proj=new
        self.nir_encoder.config.num_channels=1
        self.decode_head=base.decode_head
        self.adapters=nn.ModuleList([Adapter(c) for c in config.hidden_sizes])
        # Create modules after all shared baseline modules: identical initialization under same seed.
        self.frequency=nn.ModuleList([FrequencyResidual(c,cutoff) for c in config.hidden_sizes]) if frequency else None
    def forward(self,rgb,nir,raw=False):
        r=self.rgb_encoder(pixel_values=rgb,output_hidden_states=True).hidden_states
        n=self.nir_encoder(pixel_values=nir,output_hidden_states=True).hidden_states
        fused=[a+ad(b) for a,b,ad in zip(r,n,self.adapters)]
        if self.frequency is not None:fused=[f(x) for f,x in zip(self.frequency,fused)]
        logits=self.decode_head(fused)
        return logits if raw else F.interpolate(logits,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
