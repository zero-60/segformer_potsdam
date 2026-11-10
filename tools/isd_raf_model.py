"""RAF adaptation from cedricgsh/ISDNet (Apache-2.0).
Official commit 7341335677fd6201457b321e1e6368feb7ad647e,
mmseg/models/decode_heads/isdhead.py: ChannelAtt / RelationAwareFusion.
Changes: plain PyTorch, GN for single-GPU batch1, input/output projections,
ONE shallow/deep feature fusion added to dual SegFormer, no SR/distillation.
This is NOT the complete ISDNet, nor an implementation of global-image context.
"""
import torch
from torch import nn
from torch.nn import functional as F
from isd_base_model import PotsdamModel

def conv(cin,cout,k=3,act=True):
    layers=[nn.Conv2d(cin,cout,k,padding=k//2,bias=False),nn.GroupNorm(16,cout)]
    if act:layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)

class ChannelAttention(nn.Module):
    def __init__(self,cin,c):
        super().__init__();self.feature=conv(cin,c)
        self.descriptor=conv(c,c,k=1,act=False)
    def forward(self,x):
        f=self.feature(x)
        return f,self.descriptor(f.mean((2,3),keepdim=True))

class RelationAwareFusion(nn.Module):
    def __init__(self,c=128,r=16):
        super().__init__()
        if c%r:raise ValueError('channels must divide relation groups')
        self.r=r;self.g1=nn.Parameter(torch.zeros(1));self.g2=nn.Parameter(torch.zeros(1))
        self.spatial_att=ChannelAttention(2*c,c);self.context_att=ChannelAttention(c,c)
        self.spatial_mlp=nn.Sequential(nn.Linear(r*r,c),nn.ReLU(),nn.Linear(c,c))
        self.context_mlp=nn.Sequential(nn.Linear(r*r,c),nn.ReLU(),nn.Linear(c,c))
        self.context_head=conv(c,c);self.smooth=conv(c,c,act=False)
    def forward(self,shallow,deep):
        sf,sa=self.spatial_att(shallow);cf,ca=self.context_att(deep)
        b,c=sa.shape[:2]
        # Same grouped channel relationship construction as official RAF.
        with torch.autocast(device_type=sf.device.type,enabled=False):
            rel=torch.bmm(sa.float().reshape(b,self.r,c//self.r),
                          ca.float().reshape(b,self.r,c//self.r).transpose(1,2)).flatten(1)
        so=F.relu(self.spatial_mlp(rel));co=F.relu(self.context_mlp(rel))
        sw=torch.sigmoid(sa+self.g1*so[:,:,None,None]);cw=torch.sigmoid(ca+self.g2*co[:,:,None,None])
        sf=sf*sw;cf=cf*cw
        cf=self.context_head(F.interpolate(cf,size=sf.shape[-2:],mode='bilinear',align_corners=False))
        return self.smooth(sf+cf)

class PotsdamRAF(PotsdamModel):
    def __init__(self,checkpoint=None,use_raf=True,config=None):
        super().__init__(checkpoint=checkpoint,frequency=False,config=config)
        self.use_raf=use_raf
        if use_raf:
            c1=self.adapters[0].net[-1].out_channels;c4=self.adapters[-1].net[-1].out_channels
            self.shallow_projection=conv(c1,256,k=1)
            self.deep_projection=conv(c4,128,k=1)
            self.raf=RelationAwareFusion(128,16)
            self.raf_output=nn.Conv2d(128,c1,1)
            self.raf_scale=nn.Parameter(torch.tensor(.1))
    def forward(self,rgb,nir,raw=False,return_diagnostics=False):
        r=self.rgb_encoder(pixel_values=rgb,output_hidden_states=True).hidden_states
        n=self.nir_encoder(pixel_values=nir,output_hidden_states=True).hidden_states
        fused=[a+ad(b) for a,b,ad in zip(r,n,self.adapters)]
        details={}
        if self.use_raf:
            delta=self.raf_output(self.raf(self.shallow_projection(fused[0]),self.deep_projection(fused[-1])))
            contribution=self.raf_scale.tanh()*delta
            if return_diagnostics:
                details={'raf_scale':float(self.raf_scale.detach().tanh()),
                    'g1':float(self.raf.g1.detach()),'g2':float(self.raf.g2.detach()),
                    'relative_residual_l1':float(contribution.detach().float().abs().mean()/fused[0].detach().float().abs().mean().clamp_min(1e-8))}
            fused[0]=fused[0]+contribution
        logits=self.decode_head(fused)
        if not raw:logits=F.interpolate(logits,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        return (logits,details) if return_diagnostics else logits
