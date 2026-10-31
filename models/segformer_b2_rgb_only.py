#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standard 3-channel SegFormer-B2 for the TLSNMSP RGB-only baseline."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from transformers import SegformerForSemanticSegmentation

NUM_CLASSES = 7
CLASS_NAMES = [
    "building", "impervious_surface", "water", "barren",
    "tree", "low_vegetation", "background_other",
]


def build_model_b2_rgb_only(
    pretrained_checkpoint: str | Path,
    num_classes: int = NUM_CLASSES,
) -> tuple[SegformerForSemanticSegmentation, dict[str, Any]]:
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"本地预训练模型目录不存在：{checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"本地预训练模型缺少 config.json：{checkpoint}")
    if not any((checkpoint / n).is_file() for n in (
        "pytorch_model.bin", "model.safetensors",
        "pytorch_model.bin.index.json", "model.safetensors.index.json"
    )):
        raise FileNotFoundError(f"本地预训练目录中没有识别到权重文件：{checkpoint}")

    model, info = SegformerForSemanticSegmentation.from_pretrained(
        str(checkpoint),
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        local_files_only=True,
        output_loading_info=True,
    )
    if model.segformer.encoder.patch_embeddings[0].proj.in_channels != 3:
        raise RuntimeError("RGB-only 模型的首层输入通道不是 3，停止训练以避免通道配置错误。")
    model.config.num_labels = num_classes
    model.config.num_channels = 3
    meta = {
        "architecture": "SegFormer-B2",
        "model_id": "B2_RGB_ONLY_MERIS7",
        "input_bands": ["Red", "Green", "Blue"],
        "input_channels": 3,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "pretrained_checkpoint": str(checkpoint),
        "missing_keys_count": len(info.get("missing_keys", [])),
        "unexpected_keys_count": len(info.get("unexpected_keys", [])),
        "note": "Standard pretrained 3-channel SegFormer-B2; seven-class segmentation head is reinitialized.",
    }
    return model, meta
