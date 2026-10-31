#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Seven-class Meris/TLSNMSP wrapper around the repository's existing B2 RGB+NIR model.

This is a new file. It does not modify the Potsdam model or its six-class protocol.
"""
from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from models.segformer_b2_rgbnir_fixed_clean import (
    EXPECTED_HIDDEN_SIZES,
    SegFormerB2RGBNIRFixed,
    _adapt_nir_first_layer,
    _audit_zero_residual,
)
from models.segformer_rgb import (
    CLASS_NAMES as POTSDAM_CLASS_NAMES,
    IGNORE_INDEX as POTSDAM_IGNORE_INDEX,
    NUM_CLASSES as POTSDAM_NUM_CLASSES,
    parameter_counts,
    validate_pretrained_loading_info,
)

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
RAW_LABEL_TO_TRAIN_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


class SegFormerB2RGBNIRMeris(SegFormerB2RGBNIRFixed):
    """Same dual-encoder fixed residual architecture, with a seven-class head."""

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor, *, return_details: bool = False):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB must have shape [B,3,H,W], got {tuple(rgb.shape)}")
        if nir.ndim != 4 or nir.shape[1] != 1:
            raise ValueError(f"NIR must have shape [B,1,H,W], got {tuple(nir.shape)}")
        if rgb.shape[0] != nir.shape[0] or rgb.shape[-2:] != nir.shape[-2:]:
            raise ValueError("RGB and NIR batch/spatial shapes must match")

        rgb_features, nir_features = self._encode(rgb, nir)
        fused = []
        residual_abs_mean = []
        for rgb_feature, nir_feature, adapter in zip(
            rgb_features, nir_features, self.nir_adapters
        ):
            residual = adapter(nir_feature)
            fused_feature = rgb_feature + self.fixed_strength * residual
            if not torch.isfinite(fused_feature).all():
                raise RuntimeError("Fused feature contains NaN/Inf")
            fused.append(fused_feature)
            residual_abs_mean.append(residual.detach().float().abs().mean())

        raw_logits = self.decode_head(tuple(fused))
        if raw_logits.ndim != 4 or raw_logits.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"Expected raw logits with {NUM_CLASSES} classes, got {tuple(raw_logits.shape)}"
            )
        logits = F.interpolate(
            raw_logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Logits contain NaN/Inf")
        if not return_details:
            return logits
        return {
            "logits": logits,
            "raw_logits": raw_logits,
            "fixed_nir_strength": self.fixed_strength,
            "residual_abs_mean": torch.stack(residual_abs_mean),
        }


DEFAULT_LOCAL_CHECKPOINT = r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512"


def build_model_b2_rgbnir_meris(
    project_root,
    *,
    fixed_strength: float = 1.0,
    checkpoint: str | Path = DEFAULT_LOCAL_CHECKPOINT,
):
    """Build the seven-class RGB+NIR model from a local ADE20K B2 folder.

    The checkpoint must be a local directory containing config.json and model
    weights (pytorch_model.bin or model.safetensors). All Transformers loading
    calls use local_files_only=True, so this path cannot trigger a HF download.
    """
    checkpoint_dir = Path(checkpoint).expanduser()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"本地预训练模型目录不存在：{checkpoint_dir}\n"
            "请确认目录中包含 config.json 和 pytorch_model.bin，或通过 "
            "--pretrained-checkpoint 指定正确目录。"
        )
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"本地模型缺少 config.json：{checkpoint_dir}")
    weight_candidates = (
        "pytorch_model.bin", "model.safetensors",
        "pytorch_model.bin.index.json", "model.safetensors.index.json",
    )
    if not any((checkpoint_dir / name).is_file() for name in weight_candidates):
        raise FileNotFoundError(
            f"本地模型目录中未找到模型权重文件：{checkpoint_dir}\n"
            "需要 pytorch_model.bin 或 model.safetensors（也支持分片索引文件）。"
        )

    checkpoint_path = str(checkpoint_dir.resolve())
    config = SegformerConfig.from_pretrained(
        checkpoint_path,
        local_files_only=True,
    )
    if int(config.num_labels) != 150:
        raise RuntimeError(
            f"预期 ADE20K 预训练配置为 150 类，实际 num_labels={config.num_labels}。"
            "请确认该目录是 nvidia/segformer-b2-finetuned-ade-512-512。"
        )
    if int(getattr(config, "num_channels", 3)) != 3:
        raise RuntimeError(f"预训练模型不是三通道 RGB：num_channels={config.num_channels}")
    hidden_sizes = tuple(int(x) for x in config.hidden_sizes)
    if hidden_sizes != tuple(EXPECTED_HIDDEN_SIZES):
        raise RuntimeError(
            f"预期 SegFormer-B2 hidden_sizes={EXPECTED_HIDDEN_SIZES}，实际 {hidden_sizes}。"
        )

    # Load the ADE20K checkpoint with the repository's original six-class
    # Potsdam head shape first, so the project's loading-info validator can
    # verify that only the pretrained semantic classifier mismatches.
    id2label = {i: name for i, name in enumerate(POTSDAM_CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    config.num_labels = POTSDAM_NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = POTSDAM_IGNORE_INDEX

    base_model, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint_path,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        local_files_only=True,
    )
    loading_summary = validate_pretrained_loading_info(loading_info)

    rgb_encoder = base_model.segformer
    nir_encoder = copy.deepcopy(rgb_encoder)
    nir_init = _adapt_nir_first_layer(nir_encoder)
    decode_head = base_model.decode_head
    old_classifier = decode_head.classifier
    if not isinstance(old_classifier, nn.Conv2d):
        raise TypeError(f"Expected Conv2d decode classifier, got {type(old_classifier)!r}")

    new_classifier = nn.Conv2d(
        in_channels=old_classifier.in_channels,
        out_channels=NUM_CLASSES,
        kernel_size=old_classifier.kernel_size,
        stride=old_classifier.stride,
        padding=old_classifier.padding,
        dilation=old_classifier.dilation,
        groups=old_classifier.groups,
        bias=(old_classifier.bias is not None),
        padding_mode=old_classifier.padding_mode,
    ).to(device=old_classifier.weight.device, dtype=old_classifier.weight.dtype)
    decode_head.classifier = new_classifier

    for encoder in (rgb_encoder, nir_encoder):
        cfg = getattr(encoder, "config", None)
        if cfg is not None:
            cfg.num_labels = NUM_CLASSES
            cfg.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
            cfg.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    model = SegFormerB2RGBNIRMeris(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
        hidden_sizes=hidden_sizes,
        fixed_strength=fixed_strength,
    )
    residual_audit = _audit_zero_residual(model)
    meta: Dict[str, Any] = {
        "module_version": "1.0.1-local-only",
        "model_arch_id": "B2_RGBNIR_FIXED_RESIDUAL_MERIS7",
        "model_arch_name": "SegFormer-B2 RGB+NIR Fixed Residual Fusion - TLSNMSP 7 classes",
        "checkpoint": checkpoint_path,
        "checkpoint_source": "local_files_only",
        "resolved_revision": getattr(base_model.config, "_commit_hash", None),
        "backbone": "SegFormer-B2",
        "num_labels": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "raw_label_to_train_id": {str(k): v for k, v in RAW_LABEL_TO_TRAIN_ID.items()},
        "classifier_initialization": "newly_initialized_7_class_head; pretrained ADE20K classifier not reused",
        "fixed_nir_strength": float(fixed_strength),
        "hidden_sizes": list(hidden_sizes),
        "decoder_hidden_size": int(config.decoder_hidden_size),
        "input_modalities": ["RGB", "NIR"],
        "dual_encoder": True,
        "fusion": "RGB-anchored fixed-strength NIR residual fusion",
        "nir_encoder_initialization": nir_init,
        "residual_adapter_audit": residual_audit,
        "loading_info": loading_summary,
        "parameters": parameter_counts(model),
        "semantic_loss_ignore_index": IGNORE_INDEX,
    }
    return model, meta
