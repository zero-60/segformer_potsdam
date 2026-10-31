"""Category-prototype residual refinement for the existing five-band SegFormer.
An adaptation inspired by category-guided learning, not a reproduction of DSCA.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from models.segformer_b2_bgr_re_nir5 import (
    BAND_NAMES, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES,
    build_model_b2_bgr_re_nir5 as build_baseline,
)


class CategoryPrototypeClassifier(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        channels = classifier.in_channels
        self.prior = nn.Conv2d(channels, NUM_CLASSES, 1)
        self.classifier = classifier
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        # Start as the ordinary decode classifier; the residual learns gradually.
        nn.init.zeros_(self.project.weight)
        self.aux_logits = None

    def forward(self, feature):
        prior_logits = self.prior(feature)
        self.aux_logits = prior_logits if self.training else None
        b, c, h, w = feature.shape
        # Float32 accumulation prevents overflow at 128*128 pixels under AMP.
        probability = prior_logits.float().softmax(dim=1).flatten(2)
        values = feature.float().flatten(2)
        mass = probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prototypes = torch.bmm(probability, values.transpose(1, 2)) / mass
        context = torch.bmm(prototypes.transpose(1, 2), probability)
        context = context.reshape(b, c, h, w).to(feature.dtype)
        gate = self.gate(torch.cat([feature, context], dim=1))
        return self.classifier(feature + self.project(gate * context))


def category_aux_loss(model, labels, class_weights=None):
    logits = model.decode_head.classifier.aux_logits
    if logits is None:
        raise RuntimeError('Auxiliary category logits unavailable; call model.train() and forward first')
    logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode='bilinear', align_corners=False)
    if not (labels != IGNORE_INDEX).any():
        return logits.sum() * 0
    return F.cross_entropy(logits, labels, weight=class_weights, ignore_index=IGNORE_INDEX)


def build_model_b2_bgr_re_nir5(pretrained_checkpoint: Path | str):
    model, meta = build_baseline(pretrained_checkpoint)
    model.decode_head.classifier = CategoryPrototypeClassifier(model.decode_head.classifier)
    meta.update({
        'architecture': 'Five-band SegFormer-B2 + category-prototype residual refinement',
        'category_module_position': 'fused decoder features before final classifier',
        'category_count': NUM_CLASSES,
        'category_guidance': 'predicted soft probabilities; GT only for training auxiliary CE',
        'category_module_init': 'zero residual projection',
        'reproduction': False,
    })
    return model, meta
