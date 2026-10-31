#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SegFormer-B2 for TLSNMSP Blue + Green + Red + RedEdge (4-channel) segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

NUM_CLASSES = 7
IGNORE_INDEX = 255
CLASS_NAMES = [
    "building",
    "impervious_surface",
    "water",
    "barren",
    "tree",
    "low_vegetation",
    "background_other",
]
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge"]


def _replace_first_patch_projection(model: SegformerForSemanticSegmentation) -> dict[str, Any]:
    """Expand the pretrained RGB patch projection to 4 channels.

    The pretrained Blue/Green/Red-like input weights are copied into channels
    0..2; the RedEdge channel starts from the mean of the three pretrained
    input-channel kernels. This preserves the pretrained RGB features while
    giving RedEdge a sensible non-random initialization.
    """
    try:
        old_conv = model.segformer.encoder.patch_embeddings[0].proj
    except AttributeError as exc:
        raise RuntimeError(
            "无法定位 SegFormer 第一个 patch embedding 的卷积层；"
            "请确认 transformers 版本和本地 SegFormer-B2 模型结构。"
        ) from exc

    if not isinstance(old_conv, nn.Conv2d):
        raise RuntimeError(f"预期首层为 Conv2d，实际为 {type(old_conv).__name__}")
    if old_conv.in_channels == 4:
        return {"input_channels": 4, "adapted_from_rgb": False}
    if old_conv.in_channels != 3:
        raise RuntimeError(f"预训练首层输入通道应为 3，实际为 {old_conv.in_channels}")

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        new_conv.weight[:, 3:4].copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    model.config.num_channels = 4
    model.segformer.config.num_channels = 4
    return {
        "input_channels": 4,
        "adapted_from_rgb": True,
        "rededge_kernel_init": "mean of pretrained first-layer RGB kernels",
    }


def build_model_b2_bgr_re4(
    pretrained_checkpoint: str | Path,
    *,
    num_classes: int = NUM_CLASSES,
) -> tuple[SegformerForSemanticSegmentation, dict[str, Any]]:
    """Load local pretrained SegFormer-B2 and adapt it to 4 input bands."""
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"本地预训练模型目录不存在：{checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"本地预训练目录缺少 config.json：{checkpoint}")
    if not any(
        (checkpoint / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
    ):
        raise FileNotFoundError(f"本地预训练目录缺少权重文件：{checkpoint}")

    model, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        str(checkpoint),
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        local_files_only=True,
        output_loading_info=True,
    )
    input_info = _replace_first_patch_projection(model)
    model.config.num_labels = num_classes
    model.config.num_channels = 4
    meta = {
        "architecture": "SegFormer-B2",
        "model_id": "B2_BGR_REDEDGE_4BAND_MERIS7",
        "input_bands": BAND_NAMES,
        "num_input_channels": 4,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_loading_missing_keys_count": len(loading_info.get("missing_keys", [])),
        "pretrained_loading_unexpected_keys_count": len(loading_info.get("unexpected_keys", [])),
        "input_projection_adaptation": input_info,
        "note": (
            "ADE20K SegFormer-B2 initialization; the original RGB first-layer "
            "kernels are copied and the fourth RedEdge kernel is initialized "
            "from the mean RGB kernel. The seven-class segmentation head is "
            "reinitialized by Transformers when its shape differs."
        ),
    }
    return model, meta
