"""Potsdam fixed g=1 RGB+NIR model with an optional category prototype head.
The original dual encoders, NIR adapters and four fixed residual sums are reused.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from models.segformer_b2_rgbnir_fixed_clean import build_model_b2_rgbnir_fixed_clean
from models.segformer_rgb import NUM_CLASSES, IGNORE_INDEX, parameter_counts


class CategoryPrototypeClassifier(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        c=classifier.in_channels
        self.classifier=classifier
        # Preserve the original model's RNG stream for baseline comparisons.
        with torch.random.fork_rng(devices=[],enabled=True):
            self.prior=nn.Conv2d(c,NUM_CLASSES,1)
            self.gate=nn.Sequential(nn.Conv2d(2*c,c,1),nn.Sigmoid())
            self.project=nn.Conv2d(c,c,1,bias=False)
        nn.init.zeros_(self.project.weight)
        self.aux_logits=None

    def forward(self,feature):
        prior=self.prior(feature)
        self.aux_logits=prior if self.training else None
        b,c,h,w=feature.shape
        # Explicitly disable autocast: .float() alone does not keep bmm in FP32.
        with torch.autocast(device_type=feature.device.type,enabled=False):
            p=prior.float().softmax(1).flatten(2)
            f=feature.float().flatten(2)
            centers=torch.bmm(p,f.transpose(1,2))/p.sum(-1,keepdim=True).clamp_min(1e-6)
            context=torch.bmm(centers.transpose(1,2),p).reshape(b,c,h,w)
        context=context.to(feature.dtype)
        refined=feature+self.project(self.gate(torch.cat([feature,context],1))*context)
        return self.classifier(refined)


def category_aux_loss(model,labels):
    module=model.decode_head.classifier
    if not isinstance(module,CategoryPrototypeClassifier):
        return next(model.parameters()).sum()*0.0
    prior=module.aux_logits
    if prior is None:raise RuntimeError('辅助类别头缺少前向结果')
    prior=F.interpolate(prior.float(),size=labels.shape[-2:],mode='bilinear',align_corners=False)
    if not (labels!=IGNORE_INDEX).any():return prior.sum()*0.0
    return F.cross_entropy(prior,labels,ignore_index=IGNORE_INDEX)


def build_model_b2_rgbnir_fixed_category_clean(project_root: Path | str, *, checkpoint: str,
                                               fixed_strength: float=1.0, category_mode: str='full'):
    if float(fixed_strength)!=1.0:raise ValueError('本实验固定 NIR 系数必须为1.0')
    if category_mode not in ('full','baseline'):raise ValueError(category_mode)
    # A local ADE20K folder is required so this workflow never needs HF network.
    source=Path(checkpoint).expanduser().resolve()
    if not source.is_dir():raise FileNotFoundError(f'本地预训练目录不存在: {source}')
    model,meta=build_model_b2_rgbnir_fixed_clean(project_root,checkpoint=str(source),fixed_strength=1.0)
    if category_mode=='full':
        model.decode_head.classifier=CategoryPrototypeClassifier(model.decode_head.classifier)
    meta=dict(meta)
    meta.update(category_mode=category_mode,category_count=NUM_CLASSES,
                category_position='after original All-MLP fusion, before final classifier',
                category_design='predicted-probability prototypes + gated residual refinement',
                fixed_nir_strength=1.0,category_is_modality_gate=False,
                parameters=parameter_counts(model),checkpoint=str(source))
    return model,meta
