"""CMX modules added to the existing Potsdam fixed-fusion baseline.
Modes: baseline; frm (FRM -> fixed fusion); cmx (FRM -> fixed + residual FFM).
Encoder outputs are independently computed before rectification; unlike full CMX,
rectified features are NOT fed into later encoder stages. NIR remains one channel.
"""
import torch
from torch import nn
from torch.nn import functional as F
from cmx_full_base_model import PotsdamModel
from cmx_full_modules import FeatureRectifyModule,FeatureFusionModule

class PotsdamCMX(PotsdamModel):
    def __init__(self,checkpoint=None,variant='cmx',config=None):
        if variant not in ('baseline','frm','cmx'):raise ValueError('Invalid variant')
        super().__init__(checkpoint=checkpoint,config=config)
        self.variant=variant
        channels=[a.net[-1].out_channels for a in self.adapters]
        heads=list(self.rgb_encoder.config.num_attention_heads)
        if variant!='baseline':
            self.frms=nn.ModuleList([FeatureRectifyModule(c,reduction=1) for c in channels])
            # Official encoder initializes all child Linear/Conv/LayerNorm weights.
            for module in self.frms:module.apply(module._init_weights)
        if variant=='cmx':
            self.ffms=nn.ModuleList([FeatureFusionModule(c,reduction=1,num_heads=h,norm_layer=nn.BatchNorm2d)
                                    for c,h in zip(channels,heads)])
            self.ffm_scales=nn.Parameter(torch.full((4,),.1))
    def forward(self,rgb,nir,raw=False,return_diagnostics=False):
        r=self.rgb_encoder(pixel_values=rgb,output_hidden_states=True).hidden_states
        n=self.nir_encoder(pixel_values=nir,output_hidden_states=True).hidden_states
        fused=[];diagnostics=[]
        for i,(a,b,adapter) in enumerate(zip(r,n,self.adapters)):
            row={'stage':i+1}
            if self.variant!='baseline':
                original_a,original_b=a,b
                a,b=self.frms[i](a,b)
                if return_diagnostics:
                    with torch.no_grad():
                        cw=self.frms[i].channel_weights(original_a,original_b)
                        sw=self.frms[i].spatial_weights(original_a,original_b)
                        row.update(channel_gate_means=cw.float().mean((1,2,3,4)).tolist(),
                                   spatial_gate_means=sw.float().mean((1,2,3,4)).tolist(),
                                   rgb_rectification_relative_l1=float((a-original_a).float().abs().mean()/original_a.float().abs().mean().clamp_min(1e-8)),
                                   nir_rectification_relative_l1=float((b-original_b).float().abs().mean()/original_b.float().abs().mean().clamp_min(1e-8)))
            base=a+adapter(b)  # retain fixed NIR coefficient 1.0 after rectification
            if self.variant=='cmx':
                extra=self.ffm_scales[i].tanh()*self.ffms[i](a,b)
                if return_diagnostics:
                    row.update(ffm_scale=float(self.ffm_scales[i].detach().tanh()),
                               ffm_relative_l1=float(extra.float().abs().mean()/base.float().abs().mean().clamp_min(1e-8)))
                base=base+extra
            fused.append(base);diagnostics.append(row)
        logits=self.decode_head(fused)
        if not raw:logits=F.interpolate(logits,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        return (logits,diagnostics) if return_diagnostics else logits
