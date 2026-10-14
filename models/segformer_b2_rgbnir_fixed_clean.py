#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean fixed-fusion RGB+NIR SegFormer-B2 model definition.

Project placement
-----------------
Copy this file to:
    models/segformer_b2_rgbnir_fixed_clean.py

This module contains model construction only. It has no corruption, degradation,
training loop, or evaluation logic.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

DEFAULT_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
EXPECTED_HIDDEN_SIZES = (64, 128, 320, 512)


class CleanFixedProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanFixedProtocolError(message)


def _first_patch_projection(segformer_model: nn.Module) -> nn.Conv2d:
    try:
        projection = segformer_model.encoder.patch_embeddings[0].proj
    except Exception as exc:
        raise CleanFixedProtocolError(
            "Could not locate SegFormer first patch projection at "
            "encoder.patch_embeddings[0].proj."
        ) from exc
    require(
        isinstance(projection, nn.Conv2d),
        f"First patch projection is not Conv2d: {type(projection)!r}",
    )
    return projection


def _adapt_nir_first_layer(nir_encoder: nn.Module) -> Dict[str, Any]:
    old = _first_patch_projection(nir_encoder)
    require(
        old.in_channels == 3,
        f"Expected copied RGB first layer to have 3 channels, got {old.in_channels}.",
    )
    old_weight = old.weight.detach().clone()
    old_bias = old.bias.detach().clone() if old.bias is not None else None

    # Preserve the outer experiment RNG stream exactly as in the project model.
    with torch.random.fork_rng(devices=[], enabled=True):
        new = nn.Conv2d(
            1,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=(old.bias is not None),
            padding_mode=old.padding_mode,
        )
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.copy_(old_weight.mean(dim=1, keepdim=True))
        if old_bias is not None:
            new.bias.copy_(old_bias)

    nir_encoder.encoder.patch_embeddings[0].proj = new
    if hasattr(nir_encoder, "config"):
        nir_encoder.config.num_channels = 1
    if hasattr(nir_encoder.encoder, "config"):
        nir_encoder.encoder.config.num_channels = 1

    require(
        torch.equal(
            _first_patch_projection(nir_encoder).weight,
            old_weight.mean(dim=1, keepdim=True),
        ),
        "NIR first-layer kernel is not exact RGB-kernel mean.",
    )
    return {
        "method": "mean_of_pretrained_rgb_patch_embed_kernels",
        "old_in_channels": 3,
        "new_in_channels": 1,
        "kernel_size": list(new.kernel_size),
        "out_channels": int(new.out_channels),
    }


class NIRResidualAdapter(nn.Module):
    """Zero-output-initialized bottleneck adapter for one NIR feature scale."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(32, channels // 4)
        with torch.random.fork_rng(devices=[], enabled=True):
            self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
            self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        with torch.no_grad():
            self.expand.weight.zero_()
            self.expand.bias.zero_()
        self.channels = int(channels)
        self.hidden = int(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.act(self.reduce(x)))

from models.segformer_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    parameter_counts,
    validate_pretrained_loading_info,
)


MODULE_VERSION = "1.0.0"
MODEL_ARCH_ID = "B2_RGBNIR_FIXED_RESIDUAL"
MODEL_ARCH_NAME = "SegFormer-B2 RGB+NIR Fixed Residual Fusion"
NUM_SCALES = 4
DEFAULT_FIXED_NIR_STRENGTH = 1.0


class SegFormerB2RGBNIRFixed(nn.Module):
    """Dual B2 encoder with RGB-anchored fixed-strength NIR residual fusion."""

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        nir_encoder: nn.Module,
        decode_head: nn.Module,
        hidden_sizes: Sequence[int],
        fixed_strength: float = DEFAULT_FIXED_NIR_STRENGTH,
    ):
        super().__init__()
        hidden_sizes = tuple(int(x) for x in hidden_sizes)
        require(
            hidden_sizes == EXPECTED_HIDDEN_SIZES,
            f"Unexpected B2 hidden sizes: {hidden_sizes}",
        )
        fixed_strength = float(fixed_strength)
        require(0.0 <= fixed_strength <= 1.0, "fixed_strength must be in [0,1].")

        self.rgb_encoder = rgb_encoder
        self.nir_encoder = nir_encoder
        self.decode_head = decode_head
        self.hidden_sizes = hidden_sizes
        self.fixed_strength = fixed_strength
        self.nir_adapters = nn.ModuleList(
            [NIRResidualAdapter(channels) for channels in hidden_sizes]
        )

    def enable_gradient_checkpointing(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for name, encoder in (
            ("rgb_encoder", self.rgb_encoder),
            ("nir_encoder", self.nir_encoder),
        ):
            supported = bool(getattr(encoder, "supports_gradient_checkpointing", False))
            if not supported:
                status[name] = False
                continue
            method = getattr(encoder, "gradient_checkpointing_enable", None)
            if method is None:
                status[name] = False
                continue
            try:
                method()
            except (ValueError, NotImplementedError):
                status[name] = False
            else:
                status[name] = True
        return status

    def _encode(self, rgb: torch.Tensor, nir: torch.Tensor):
        rgb_out = self.rgb_encoder(
            pixel_values=rgb,
            output_hidden_states=True,
            return_dict=True,
        )
        nir_out = self.nir_encoder(
            pixel_values=nir,
            output_hidden_states=True,
            return_dict=True,
        )
        rgb_features = tuple(rgb_out.hidden_states)
        nir_features = tuple(nir_out.hidden_states)

        require(
            len(rgb_features) == NUM_SCALES,
            f"RGB encoder returned {len(rgb_features)} scales.",
        )
        require(
            len(nir_features) == NUM_SCALES,
            f"NIR encoder returned {len(nir_features)} scales.",
        )
        for index, (rgb_feature, nir_feature, channels) in enumerate(
            zip(rgb_features, nir_features, self.hidden_sizes), start=1
        ):
            require(
                rgb_feature.shape == nir_feature.shape,
                f"Scale {index}: RGB/NIR shapes differ.",
            )
            require(
                rgb_feature.shape[1] == channels,
                f"Scale {index}: expected C={channels}, got {rgb_feature.shape[1]}.",
            )
        return rgb_features, nir_features

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        require(
            rgb.ndim == 4 and rgb.shape[1] == 3,
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}",
        )
        require(
            nir.ndim == 4 and nir.shape[1] == 1,
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}",
        )
        require(
            rgb.shape[0] == nir.shape[0] and rgb.shape[-2:] == nir.shape[-2:],
            "RGB/NIR inputs are not aligned.",
        )

        rgb_features, nir_features = self._encode(rgb, nir)
        fused = []
        residual_abs_mean = []
        for rgb_feature, nir_feature, adapter in zip(
            rgb_features, nir_features, self.nir_adapters
        ):
            residual = adapter(nir_feature)
            fused_feature = rgb_feature + self.fixed_strength * residual
            require(
                torch.isfinite(fused_feature).all().item(),
                "Fused feature contains NaN/Inf.",
            )
            fused.append(fused_feature)
            with torch.no_grad():
                residual_abs_mean.append(residual.detach().float().abs().mean())

        raw_logits = self.decode_head(tuple(fused))
        require(
            raw_logits.ndim == 4
            and raw_logits.shape[0] == rgb.shape[0]
            and raw_logits.shape[1] == NUM_CLASSES,
            f"Bad raw logits shape: {tuple(raw_logits.shape)}",
        )
        logits = F.interpolate(
            raw_logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        require(torch.isfinite(logits).all().item(), "Logits contain NaN/Inf.")

        if not return_details:
            return logits
        return {
            "logits": logits,
            "raw_logits": raw_logits,
            "fixed_nir_strength": self.fixed_strength,
            "residual_abs_mean": torch.stack(residual_abs_mean),
        }


def _audit_zero_residual(model: SegFormerB2RGBNIRFixed) -> Dict[str, Any]:
    rows = []
    for scale, adapter in enumerate(model.nir_adapters, start=1):
        w_nonzero = int(torch.count_nonzero(adapter.expand.weight).item())
        b_nonzero = int(torch.count_nonzero(adapter.expand.bias).item())
        require(
            w_nonzero == 0 and b_nonzero == 0,
            f"Scale {scale}: residual adapter final projection is not zero-init.",
        )
        rows.append(
            {
                "scale": scale,
                "channels": adapter.channels,
                "bottleneck": adapter.hidden,
                "zero_output_initialized": True,
            }
        )
    return {"all_zero_output_initialized": True, "stages": rows}


def build_model_b2_rgbnir_fixed_clean(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    fixed_strength: float = DEFAULT_FIXED_NIR_STRENGTH,
) -> Tuple[SegFormerB2RGBNIRFixed, Dict[str, Any]]:
    _ = Path(project_root).resolve()
    require(
        checkpoint == DEFAULT_CHECKPOINT,
        "Clean fixed-fusion protocol requires the project's standard pretrained B2 checkpoint.",
    )

    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}

    config = SegformerConfig.from_pretrained(checkpoint)
    require(int(config.num_labels) == 150, "Expected ADE20K 150-label checkpoint.")
    require(
        int(getattr(config, "num_channels", 3)) == 3,
        "Pretrained B2 checkpoint is not RGB.",
    )
    require(
        tuple(int(x) for x in config.hidden_sizes) == EXPECTED_HIDDEN_SIZES,
        f"Unexpected B2 hidden sizes: {config.hidden_sizes}",
    )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    base_model, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    loading_summary = validate_pretrained_loading_info(loading_info)

    rgb_encoder = base_model.segformer
    nir_encoder = copy.deepcopy(rgb_encoder)
    nir_init = _adapt_nir_first_layer(nir_encoder)
    decode_head = base_model.decode_head

    model = SegFormerB2RGBNIRFixed(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
        hidden_sizes=config.hidden_sizes,
        fixed_strength=fixed_strength,
    )
    residual_audit = _audit_zero_residual(model)

    meta = {
        "module_version": MODULE_VERSION,
        "model_arch_id": MODEL_ARCH_ID,
        "model_arch_name": MODEL_ARCH_NAME,
        "checkpoint": checkpoint,
        "resolved_revision": getattr(base_model.config, "_commit_hash", None),
        "backbone": "SegFormer-B2",
        "num_labels": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "hidden_sizes": list(EXPECTED_HIDDEN_SIZES),
        "decoder_hidden_size": int(config.decoder_hidden_size),
        "input_modalities": ["RGB", "NIR"],
        "dual_encoder": True,
        "fusion": "RGB-anchored fixed-strength NIR residual fusion",
        "fusion_rule": (
            f"F_i = F_RGB_i + {float(fixed_strength):.8g} * Adapter_i(F_NIR_i)"
        ),
        "fixed_nir_strength": float(fixed_strength),
        "quality_gate": False,
        "gate_supervision": False,
        "nir_encoder_initialization": nir_init,
        "residual_adapter_audit": residual_audit,
        "loading_info": loading_summary,
        "parameters": parameter_counts(model),
        "semantic_loss_ignore_index": IGNORE_INDEX,
    }
    return model, meta
