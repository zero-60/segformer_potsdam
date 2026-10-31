#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-encoder SegFormer-B2 for B,G,R,RedEdge,NIR five-band TLSNMSP segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

NUM_CLASSES = 7
IGNORE_INDEX = 255
BAND_NAMES = ("Blue", "Green", "Red", "RedEdge", "NIR")
CLASS_NAMES = (
    "Building",
    "Impervious surface",
    "Water",
    "Bare soil",
    "Tree",
    "Low vegetation",
    "Background/Other",
)


def build_model_b2_bgr_re_nir5(pretrained_checkpoint: Path | str):
    """Load local ADE20K SegFormer-B2 weights and expand the first conv to 5 channels.

    The pretrained RGB kernels are preserved for B/G/R. The RedEdge and NIR kernels
    are initialized from the mean of the pretrained RGB kernels. The seven-class
    classifier is initialized for the TLSNMSP task by Transformers when its shape
    differs from the ADE20K classifier.
    """
    path = Path(pretrained_checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"本地预训练目录不存在：{path}")
    config = SegformerConfig.from_pretrained(path, local_files_only=True)
    config.num_labels = NUM_CLASSES
    config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    config.num_channels = 3  # load original pretrained RGB weights first
    model = SegformerForSemanticSegmentation.from_pretrained(
        path,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    # SegFormer first-stage patch embedding projection: RGB -> five-band.
    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    if not isinstance(old_conv, nn.Conv2d):
        raise RuntimeError(f"预期首层 patch projection 为 Conv2d，实际为 {type(old_conv)}")
    new_conv = nn.Conv2d(
        in_channels=5,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        mean_rgb_kernel = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:4].copy_(mean_rgb_kernel)
        new_conv.weight[:, 4:5].copy_(mean_rgb_kernel)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    model.segformer.encoder.patch_embeddings[0].num_channels = 5
    model.config.num_channels = 5
    model.segformer.config.num_channels = 5

    model_meta: dict[str, Any] = {
        "architecture": "SegFormer-B2 single encoder + All-MLP decode head",
        "input_bands": list(BAND_NAMES),
        "input_channels": 5,
        "num_classes": NUM_CLASSES,
        "pretrained_source": str(path),
        "first_conv_initialization": (
            "pretrained RGB kernels copied to B/G/R; RedEdge and NIR kernels "
            "initialized from mean RGB kernel"
        ),
        "fusion_design": "early fusion by channel concatenation; no separate NIR branch",
    }
    return model, model_meta
