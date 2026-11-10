"""Original dual SegFormer B2 baseline, fixed NIR residual coefficient 1.0."""
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

class PotsdamModel(nn.Module):
    def __init__(self,checkpoint=None,config=None):
        super().__init__()
        from transformers import SegformerConfig,SegformerForSemanticSegmentation
        if config is None:
            config=SegformerConfig.from_pretrained(str(checkpoint),local_files_only=True)
            if tuple(config.hidden_sizes)!=(64,128,320,512): raise ValueError('Expected SegFormer-B2 checkpoint')
            config.num_labels=6; config.id2label={i:str(i) for i in range(6)};config.label2id={str(i):i for i in range(6)}
            base=SegformerForSemanticSegmentation.from_pretrained(str(checkpoint),config=config,
                ignore_mismatched_sizes=True,local_files_only=True)
        else:
            # Rebuild from a saved config; callers load the trained state before use.
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

    def forward(self,rgb,nir,raw=False):
        r=self.rgb_encoder(pixel_values=rgb,output_hidden_states=True).hidden_states
        n=self.nir_encoder(pixel_values=nir,output_hidden_states=True).hidden_states
        fused=[a+ad(b) for a,b,ad in zip(r,n,self.adapters)]
        logits=self.decode_head(fused)
        return logits if raw else F.interpolate(logits,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
