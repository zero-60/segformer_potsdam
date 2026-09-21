#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model D / Proposed v2:
SegFormer-B2 + Degradation-Aware Residual Fusion (DARF).

Goal
----
Improve both clean mIoU and RGB-degradation robustness after the original
B0 convex Quality Gate showed two limitations:

1) B0 capacity capped clean performance around the mid-0.7 mIoU range.
2) Convex RGB/NIR mixing can disturb the pretrained RGB feature distribution.
3) A gate trained only on clean semantic CE is not forced to represent image
   quality, so its weights need not track degradation severity.

DARF changes the fusion rule from convex replacement to RGB-anchored residual
correction:

    R_i = Adapter_i(F_NIR_i)
    g_i = QualityGate_i(F_RGB_i, F_NIR_i) in [0,1]
    F_fused_i = F_RGB_i + g_i * R_i

The NIR residual adapter is zero-output initialized, so at initialization:

    F_fused_i == F_RGB_i

exactly, regardless of the initial gate value. This preserves the pretrained
RGB SegFormer-B2 path and decoder distribution at step 0.

The Quality Gate predicts NIR residual strength (not a convex RGB weight).
During training, a small auxiliary quality-supervision loss teaches g_i to
increase when RGB is synthetically degraded. At inference the gate uses only
features; corruption labels/severity are not required.

Backbone
--------
Hugging Face:
    nvidia/segformer-b2-finetuned-ade-512-512

B2 hidden sizes:
    [64, 128, 320, 512]

One RGB encoder and one NIR encoder are used. The NIR encoder is copied from
the pretrained RGB encoder; its first patch projection is converted 3->1 by
the mean of the pretrained RGB kernels.

Expected location
-----------------
models/segformer_b2_darf.py
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from models.segformer_rgb import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    parameter_counts,
    validate_pretrained_loading_info,
)


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Model D DARF-B2 Protocol v1"

MODEL_ID = "D_DARF_B2"
MODEL_NAME = "Model D (SegFormer-B2 + Degradation-Aware Residual Fusion)"

DEFAULT_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"

RGB_CHANNELS = 3
NIR_CHANNELS = 1
NUM_SCALES = 4
EXPECTED_HIDDEN_SIZES = (64, 128, 320, 512)

GATE_TYPE = "feature_quality_mlp_nir_residual_strength"
GATE_DESCRIPTOR = (
    "concat(mean_rgb, std_rgb, mean_nir, std_nir, mean_abs_rgb_nir_diff)"
)
INITIAL_NIR_GATE = 0.05
GATE_EPS = 1e-6


class DARFProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DARFProtocolError(message)


def _class_maps():
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def _first_patch_projection(segformer_model: nn.Module) -> nn.Conv2d:
    try:
        projection = segformer_model.encoder.patch_embeddings[0].proj
    except Exception as exc:
        raise DARFProtocolError(
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

    # Avoid changing the outer experiment RNG stream.
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
    """
    Lightweight bottleneck residual adapter.

    The final 1x1 Conv is exact-zero initialized, so adapter output is exactly
    zero at initialization and DARF starts as an RGB-only pretrained path.
    """

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


class ResidualQualityGate(nn.Module):
    """
    Predict one NIR residual strength per sample and scale.

    Descriptor dimension = 5*C:
        mean RGB
        std RGB
        mean NIR
        std NIR
        mean absolute RGB/NIR feature difference

    The final projection starts with zero weights and a bias corresponding to
    INITIAL_NIR_GATE. Gate supervision in training gives this module an
    explicit quality-learning signal.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.channels = int(channels)
        self.descriptor_dim = 5 * self.channels
        hidden = max(32, self.channels // 4)

        with torch.random.fork_rng(devices=[], enabled=True):
            self.fc1 = nn.Linear(self.descriptor_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)

        self.act = nn.GELU()

        prior = float(INITIAL_NIR_GATE)
        prior_logit = math.log(prior / (1.0 - prior))

        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.fill_(prior_logit)

    def _descriptor(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ) -> torch.Tensor:
        require(rgb.shape == nir.shape, "RGB/NIR feature shapes differ.")
        require(rgb.ndim == 4, "Gate expects BCHW features.")

        with torch.autocast(device_type=rgb.device.type, enabled=False):
            r = rgb.float()
            n = nir.float()

            r_var, r_mean = torch.var_mean(r, dim=(-2, -1), correction=0)
            n_var, n_mean = torch.var_mean(n, dim=(-2, -1), correction=0)

            r_std = torch.sqrt(torch.clamp(r_var, min=0.0) + GATE_EPS)
            n_std = torch.sqrt(torch.clamp(n_var, min=0.0) + GATE_EPS)
            diff = torch.mean(torch.abs(r - n), dim=(-2, -1))

            q = torch.cat(
                (r_mean, r_std, n_mean, n_std, diff),
                dim=1,
            )

        require(
            tuple(q.shape) == (rgb.shape[0], self.descriptor_dim),
            f"Bad gate descriptor shape: {tuple(q.shape)}",
        )
        return q

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._descriptor(rgb, nir)

        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logit = self.fc2(self.act(self.fc1(q.float()))).squeeze(-1)
            strength = torch.sigmoid(logit)

        require(
            torch.isfinite(strength).all().item(),
            "NIR gate strength contains NaN/Inf.",
        )

        return strength, logit


class SegFormerB2DARF(nn.Module):
    """
    Dual-encoder B2 with RGB-anchored degradation-aware residual NIR fusion.
    """

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        nir_encoder: nn.Module,
        decode_head: nn.Module,
        hidden_sizes: Sequence[int],
    ):
        super().__init__()

        require(
            tuple(int(x) for x in hidden_sizes) == EXPECTED_HIDDEN_SIZES,
            f"Unexpected B2 hidden sizes: {hidden_sizes}",
        )

        self.rgb_encoder = rgb_encoder
        self.nir_encoder = nir_encoder
        self.decode_head = decode_head
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.nir_adapters = nn.ModuleList(
            [NIRResidualAdapter(c) for c in self.hidden_sizes]
        )
        self.quality_gates = nn.ModuleList(
            [ResidualQualityGate(c) for c in self.hidden_sizes]
        )

    def enable_gradient_checkpointing(self) -> Dict[str, bool]:
        """
        Enable Hugging Face gradient checkpointing only when the installed
        Transformers version truly supports it.

        Some Transformers releases expose gradient_checkpointing_enable() on
        PreTrainedModel while SegformerModel itself advertises
        supports_gradient_checkpointing=False; calling the method then raises
        ValueError.  Treat that case as "unsupported" instead of crashing.
        """
        status: Dict[str, bool] = {}

        for name, encoder in (
            ("rgb_encoder", self.rgb_encoder),
            ("nir_encoder", self.nir_encoder),
        ):
            supported = bool(
                getattr(
                    encoder,
                    "supports_gradient_checkpointing",
                    False,
                )
            )

            if not supported:
                status[name] = False
                continue

            method = getattr(
                encoder,
                "gradient_checkpointing_enable",
                None,
            )

            if method is None:
                status[name] = False
                continue

            try:
                method()
            except (ValueError, NotImplementedError):
                # Compatibility path for older Transformers versions.
                status[name] = False
            else:
                status[name] = True

        return status

    def _encode(self, rgb: torch.Tensor, nir: torch.Tensor):
        ro = self.rgb_encoder(
            pixel_values=rgb,
            output_hidden_states=True,
            return_dict=True,
        )
        no = self.nir_encoder(
            pixel_values=nir,
            output_hidden_states=True,
            return_dict=True,
        )

        rf = tuple(ro.hidden_states)
        nf = tuple(no.hidden_states)

        require(len(rf) == NUM_SCALES, f"RGB encoder returned {len(rf)} scales.")
        require(len(nf) == NUM_SCALES, f"NIR encoder returned {len(nf)} scales.")

        for i, (r, n, c) in enumerate(zip(rf, nf, self.hidden_sizes)):
            require(r.shape == n.shape, f"Scale {i}: RGB/NIR shape mismatch.")
            require(r.shape[1] == c, f"Scale {i}: expected C={c}, got {r.shape[1]}.")

        return rf, nf

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
        gate_strengths = []
        gate_logits = []

        for r, n, adapter, gate in zip(
            rgb_features,
            nir_features,
            self.nir_adapters,
            self.quality_gates,
        ):
            strength, logit = gate(r, n)
            residual = adapter(n)

            s = strength.to(dtype=r.dtype).view(-1, 1, 1, 1)
            f = r + s * residual

            require(torch.isfinite(f).all().item(), "Fused feature has NaN/Inf.")

            fused.append(f)
            gate_strengths.append(strength)
            gate_logits.append(logit)

        fused_tuple = tuple(fused)

        raw_logits = self.decode_head(fused_tuple)

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

        if not return_details:
            return logits

        return {
            "logits": logits,
            "raw_logits": raw_logits,
            "nir_gate_strength": torch.stack(gate_strengths, dim=1),
            "nir_gate_logits": torch.stack(gate_logits, dim=1),
        }


def _audit_zero_residual(model: SegFormerB2DARF) -> Dict[str, Any]:
    stages = []

    for i, adapter in enumerate(model.nir_adapters, 1):
        w_nonzero = int(torch.count_nonzero(adapter.expand.weight).item())
        b_nonzero = int(torch.count_nonzero(adapter.expand.bias).item())

        require(
            w_nonzero == 0 and b_nonzero == 0,
            f"Stage {i} residual adapter final projection is not zero-init.",
        )

        stages.append(
            {
                "scale": i,
                "channels": adapter.channels,
                "bottleneck": adapter.hidden,
                "zero_output_initialized": True,
            }
        )

    return {
        "all_zero_output_initialized": True,
        "stages": stages,
    }


def build_model_d_darf_b2(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Tuple[SegFormerB2DARF, Dict[str, Any]]:
    project_root = Path(project_root).resolve()

    require(
        checkpoint == DEFAULT_CHECKPOINT,
        f"DARF-B2 v1 only permits checkpoint {DEFAULT_CHECKPOINT}",
    )

    id2label, label2id = _class_maps()

    config = SegformerConfig.from_pretrained(checkpoint)
    original_num_labels = int(config.num_labels)

    require(original_num_labels == 150, f"Expected 150 ADE labels, got {original_num_labels}.")
    require(int(getattr(config, "num_channels", 3)) == 3, "B2 checkpoint is not 3-channel.")
    require(
        tuple(int(x) for x in config.hidden_sizes) == EXPECTED_HIDDEN_SIZES,
        f"Unexpected B2 hidden sizes: {config.hidden_sizes}",
    )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    base, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )

    loading_summary = validate_pretrained_loading_info(loading_info)

    rgb_encoder = base.segformer
    nir_encoder = copy.deepcopy(rgb_encoder)
    nir_init = _adapt_nir_first_layer(nir_encoder)
    decode_head = base.decode_head

    model = SegFormerB2DARF(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
        hidden_sizes=config.hidden_sizes,
    )

    residual_audit = _audit_zero_residual(model)

    resolved_revision = getattr(base.config, "_commit_hash", None)

    meta = {
        "variant": "D",
        "model_id": MODEL_ID,
        "name": MODEL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": checkpoint,
        "resolved_revision": resolved_revision,
        "backbone": "SegFormer-B2",
        "num_labels": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "hidden_sizes": list(EXPECTED_HIDDEN_SIZES),
        "decoder_hidden_size": int(config.decoder_hidden_size),
        "dual_encoder": True,
        "fusion": "RGB-anchored gated NIR residual",
        "fusion_rule": "F_fused_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)",
        "quality_gate": True,
        "gate_type": GATE_TYPE,
        "gate_descriptor": GATE_DESCRIPTOR,
        "initial_nir_gate_strength": INITIAL_NIR_GATE,
        "nir_encoder_initialization": nir_init,
        "residual_adapter_audit": residual_audit,
        "loading_info": loading_summary,
        "parameters": parameter_counts(model),
        "semantic_loss_ignore_index": IGNORE_INDEX,
    }

    return model, meta


def _smoke_test(project_root: Path) -> None:
    from data_pipeline.potsdam_dataset import PotsdamTrainDataset

    print("=" * 100)
    print("Model D | SegFormer-B2 + Degradation-Aware Residual Fusion | smoke test")
    print("=" * 100)

    ds = PotsdamTrainDataset(project_root, epoch=0)
    sample = ds[0]

    rgb = sample["rgb"].unsqueeze(0)
    nir = sample["nir"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    model, meta = build_model_d_darf_b2(project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    with torch.inference_mode():
        d = model(
            rgb.to(device),
            nir.to(device),
            return_details=True,
        )

    require(
        tuple(d["logits"].shape) == (1, NUM_CLASSES, 512, 512),
        f"Bad full logits shape: {tuple(d['logits'].shape)}",
    )
    require(
        tuple(d["raw_logits"].shape) == (1, NUM_CLASSES, 128, 128),
        f"Bad raw logits shape: {tuple(d['raw_logits'].shape)}",
    )

    gates = d["nir_gate_strength"]

    require(
        torch.allclose(
            gates,
            torch.full_like(gates, INITIAL_NIR_GATE),
            atol=1e-6,
            rtol=0.0,
        ),
        f"Initial gate is not {INITIAL_NIR_GATE}: {gates}",
    )

    loss = F.cross_entropy(
        d["logits"],
        labels.to(device),
        ignore_index=IGNORE_INDEX,
    )

    print(f"checkpoint              : {meta['checkpoint']}")
    print(f"resolved revision       : {meta['resolved_revision']}")
    print(f"hidden sizes            : {meta['hidden_sizes']}")
    print(f"decoder hidden size     : {meta['decoder_hidden_size']}")
    print(f"parameters              : {meta['parameters']['total']:,}")
    print(f"initial NIR gate        : {gates[0].detach().cpu().tolist()}")
    print(f"zero residual adapters  : {meta['residual_adapter_audit']['all_zero_output_initialized']}")
    print(f"full logits             : {tuple(d['logits'].shape)}")
    print(f"CE                      : {float(loss.item()):.6f}")
    print("FINAL STATUS            : PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    x = parser.parse_args()

    root = x.project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    _smoke_test(root)


if __name__ == "__main__":
    main()
