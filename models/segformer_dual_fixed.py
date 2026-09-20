#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model C-noGate: Dual-Encoder SegFormer-B0 + Fixed Multi-Scale Fusion.

Scientific role
---------------
This is the fixed-fusion control for the proposed Quality Gate model.

Architecture
------------
                      RGB [B,3,H,W]
                           |
                    RGB SegFormer encoder
                           |
             R1, R2, R3, R4  (4 scales)
                           |
                           |
                     fixed fusion at every scale
                     Fi = 0.5*Ri + 0.5*Ni
                           |
             N1, N2, N3, N4
                           ^
                           |
                    NIR SegFormer encoder
                           |
                      NIR [B,1,H,W]

Four fused feature maps are then fed to ONE shared pretrained SegFormer decode
head, producing 6-class logits.

Critical controlled-variable rule
---------------------------------
The future Model C (Quality Gate / Ours) should keep:
- exactly the same RGB encoder;
- exactly the same NIR encoder;
- exactly the same encoder initialization;
- exactly the same SegFormer decoder;
- exactly the same four fusion locations;
- exactly the same training protocol;

and ONLY replace:

    fixed:
        w_RGB_i = 0.5
        w_NIR_i = 0.5

with:

    dynamic:
        w_RGB_i = QualityGate_i(...)
        w_NIR_i = 1 - w_RGB_i

This makes Model C-noGate the cleanest possible ablation for RQ4.

Pretrained initialization
-------------------------
A frozen Model-A SegFormer-B0 checkpoint is loaded through the existing
models.segformer_rgb.build_model_a_rgb() builder.

RGB encoder:
    exact ADE20K pretrained SegFormer encoder.

NIR encoder:
    deep copy of the RGB encoder. The first 3-channel patch-embedding Conv2d
    is converted to 1 channel using the MEAN of the pretrained RGB kernels:

        W_NIR = mean(W_RGB, dim=input_channel)

    All other NIR-encoder parameters are initially bit-identical to the RGB
    encoder.

Decoder:
    one pretrained SegFormer decode head from the same base model. Its final
    classifier is the same 150->6 Potsdam replacement used by Model A.

No trainable fusion layer is added in C-noGate.

Expected location:
    models/segformer_dual_fixed.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segformer_rgb import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    IGNORE_INDEX,
    NUM_CLASSES,
    build_model_a_rgb,
)


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Model C-noGate Fixed Multi-Scale Fusion Protocol v1"

MODEL_ID = "C_NOGATE_FIXED_FUSION"
MODEL_NAME = "Model C-noGate (Dual Encoder + Fixed Fusion)"

RGB_CHANNELS = 3
NIR_CHANNELS = 1
NUM_SCALES = 4
FIXED_RGB_WEIGHT = 0.5
FIXED_NIR_WEIGHT = 0.5
NIR_INIT_METHOD = "mean_of_pretrained_rgb_patch_embed_kernels"

EXPECTED_HIDDEN_SIZES = (32, 64, 160, 256)


class ModelCNoGateProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelCNoGateProtocolError(message)


def parameter_counts(module: nn.Module) -> Dict[str, int]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(
        int(p.numel())
        for p in module.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(
        arr.tobytes()
    ).hexdigest()


def _first_patch_projection(
    segformer_model: nn.Module,
) -> nn.Conv2d:
    """
    transformers==4.53.2:
        SegformerModel.encoder.patch_embeddings[0].proj
    """
    try:
        projection = (
            segformer_model
            .encoder
            .patch_embeddings[0]
            .proj
        )
    except Exception as exc:
        raise ModelCNoGateProtocolError(
            "Could not locate SegFormer first patch-embedding Conv2d at "
            "encoder.patch_embeddings[0].proj. "
            "Check the installed transformers version."
        ) from exc

    require(
        isinstance(projection, nn.Conv2d),
        "First patch projection is not nn.Conv2d: "
        f"{type(projection)!r}",
    )

    return projection


def _adapt_nir_encoder_first_layer(
    nir_encoder: nn.Module,
) -> Dict[str, Any]:
    """
    Convert the copied RGB encoder's first patch embedding from 3 -> 1 channel.

    The new module construction is wrapped in fork_rng() so parameter
    construction itself does not advance the global training RNG. Every new
    parameter is overwritten immediately afterwards.
    """
    old_proj = _first_patch_projection(
        nir_encoder
    )

    require(
        old_proj.in_channels == RGB_CHANNELS,
        "Expected copied NIR encoder to start from a 3-channel pretrained "
        f"projection, got in_channels={old_proj.in_channels}.",
    )
    require(
        old_proj.groups == 1,
        "Expected first patch projection groups=1.",
    )

    old_weight = (
        old_proj
        .weight
        .detach()
        .clone()
    )
    old_bias = (
        old_proj
        .bias
        .detach()
        .clone()
        if old_proj.bias is not None
        else None
    )

    rgb_hash = tensor_sha256(
        old_weight
    )

    with torch.random.fork_rng(
        devices=[],
        enabled=True,
    ):
        new_proj = nn.Conv2d(
            in_channels=NIR_CHANNELS,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            dilation=old_proj.dilation,
            groups=old_proj.groups,
            bias=(old_proj.bias is not None),
            padding_mode=old_proj.padding_mode,
        )

    new_proj = new_proj.to(
        device=old_proj.weight.device,
        dtype=old_proj.weight.dtype,
    )

    nir_kernel = old_weight.mean(
        dim=1,
        keepdim=True,
    )

    with torch.no_grad():
        new_proj.weight.copy_(
            nir_kernel
        )

        if old_bias is not None:
            require(
                new_proj.bias is not None,
                "New NIR patch projection unexpectedly has no bias.",
            )
            new_proj.bias.copy_(
                old_bias
            )

    nir_encoder.encoder.patch_embeddings[0].proj = (
        new_proj
    )

    # Keep all config objects audit-correct. In transformers 4.53.2 the
    # SegformerModel and its encoder hold the same logical configuration.
    if hasattr(
        nir_encoder,
        "config",
    ):
        nir_encoder.config.num_channels = (
            NIR_CHANNELS
        )

    if hasattr(
        nir_encoder.encoder,
        "config",
    ):
        nir_encoder.encoder.config.num_channels = (
            NIR_CHANNELS
        )

    adapted = _first_patch_projection(
        nir_encoder
    )

    require(
        adapted.in_channels == NIR_CHANNELS,
        "NIR encoder first Conv2d was not adapted to one channel.",
    )
    require(
        torch.equal(
            adapted.weight,
            nir_kernel,
        ),
        "NIR first-layer kernel is not the exact RGB-kernel mean.",
    )

    if old_bias is not None:
        require(
            adapted.bias is not None
            and torch.equal(
                adapted.bias,
                old_bias,
            ),
            "NIR first-layer bias changed unexpectedly.",
        )

    return {
        "method": NIR_INIT_METHOD,
        "old_in_channels": RGB_CHANNELS,
        "new_in_channels": NIR_CHANNELS,
        "out_channels": int(
            adapted.out_channels
        ),
        "kernel_size": list(
            adapted.kernel_size
        ),
        "stride": list(
            adapted.stride
        ),
        "padding": list(
            adapted.padding
        ),
        "source_rgb_kernel_sha256": (
            rgb_hash
        ),
        "nir_kernel_sha256": (
            tensor_sha256(
                adapted.weight
            )
        ),
        "nir_kernel_equals_rgb_mean_exactly": True,
        "bias_preserved_exactly": True,
    }


def _verify_encoder_copy(
    rgb_encoder: nn.Module,
    nir_encoder: nn.Module,
) -> Dict[str, Any]:
    """
    Verify every copied encoder tensor is identical except the first patch
    projection weight, whose input-channel shape intentionally differs.

    The first patch projection bias MUST still be identical.
    """
    rgb_state = (
        rgb_encoder.state_dict()
    )
    nir_state = (
        nir_encoder.state_dict()
    )

    require(
        set(rgb_state.keys())
        == set(nir_state.keys()),
        "RGB and NIR encoder state_dict keys differ.",
    )

    changed_keys = []
    equal_keys = []

    first_weight_key = (
        "encoder.patch_embeddings.0.proj.weight"
    )

    for key in rgb_state:
        rgb_tensor = rgb_state[key]
        nir_tensor = nir_state[key]

        if key == first_weight_key:
            require(
                tuple(rgb_tensor.shape)
                != tuple(nir_tensor.shape),
                "Expected first patch projection weight shape to differ.",
            )
            changed_keys.append(key)
            continue

        require(
            tuple(rgb_tensor.shape)
            == tuple(nir_tensor.shape),
            f"Unexpected RGB/NIR encoder shape mismatch: {key}: "
            f"{tuple(rgb_tensor.shape)} vs {tuple(nir_tensor.shape)}",
        )

        require(
            torch.equal(
                rgb_tensor,
                nir_tensor,
            ),
            "Copied NIR encoder parameter/buffer differs before training "
            f"outside first projection weight: {key}",
        )

        equal_keys.append(key)

    require(
        changed_keys
        == [first_weight_key],
        f"Unexpected changed encoder keys: {changed_keys}",
    )

    return {
        "only_first_patch_weight_changed": True,
        "changed_keys": changed_keys,
        "bit_identical_other_state_entries": len(
            equal_keys
        ),
    }


def _validate_feature_pyramid(
    features: Sequence[torch.Tensor],
    *,
    modality: str,
    batch_size: int,
) -> None:
    require(
        len(features) == NUM_SCALES,
        f"{modality} encoder must return {NUM_SCALES} stage features, "
        f"got {len(features)}.",
    )

    previous_h = None
    previous_w = None

    for index, (
        feature,
        expected_channels,
    ) in enumerate(
        zip(
            features,
            EXPECTED_HIDDEN_SIZES,
        )
    ):
        require(
            isinstance(
                feature,
                torch.Tensor,
            ),
            f"{modality} stage {index} is not a Tensor.",
        )
        require(
            feature.ndim == 4,
            f"{modality} stage {index} must be BCHW, "
            f"got {tuple(feature.shape)}.",
        )
        require(
            feature.shape[0] == batch_size,
            f"{modality} stage {index} batch mismatch.",
        )
        require(
            feature.shape[1]
            == expected_channels,
            f"{modality} stage {index} expected C={expected_channels}, "
            f"got C={feature.shape[1]}.",
        )
        require(
            torch.isfinite(
                feature
            ).all().item(),
            f"{modality} stage {index} contains NaN/Inf.",
        )

        height = int(
            feature.shape[-2]
        )
        width = int(
            feature.shape[-1]
        )

        if previous_h is not None:
            require(
                height < previous_h
                and width < previous_w,
                f"{modality} feature pyramid is not spatially descending.",
            )

        previous_h = height
        previous_w = width


class SegFormerDualFixedFusion(nn.Module):
    """
    Dual SegFormer encoders with fixed equal-weight multi-scale fusion.

    No trainable parameter is introduced by the fusion operator itself.
    """

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        nir_encoder: nn.Module,
        decode_head: nn.Module,
    ):
        super().__init__()

        self.rgb_encoder = (
            rgb_encoder
        )
        self.nir_encoder = (
            nir_encoder
        )
        self.decode_head = (
            decode_head
        )

        # Fixed weights are buffers for transparent serialization/auditing.
        # They are not trainable parameters.
        fixed_weights = torch.tensor(
            [
                FIXED_RGB_WEIGHT,
                FIXED_NIR_WEIGHT,
            ],
            dtype=torch.float32,
        ).repeat(
            NUM_SCALES,
            1,
        )

        self.register_buffer(
            "fixed_fusion_weights",
            fixed_weights,
            persistent=True,
        )

    def _encode(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ) -> tuple[
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        rgb_outputs = self.rgb_encoder(
            pixel_values=rgb,
            output_hidden_states=True,
            return_dict=True,
        )

        nir_outputs = self.nir_encoder(
            pixel_values=nir,
            output_hidden_states=True,
            return_dict=True,
        )

        rgb_features = tuple(
            rgb_outputs.hidden_states
        )
        nir_features = tuple(
            nir_outputs.hidden_states
        )

        _validate_feature_pyramid(
            rgb_features,
            modality="RGB",
            batch_size=int(
                rgb.shape[0]
            ),
        )
        _validate_feature_pyramid(
            nir_features,
            modality="NIR",
            batch_size=int(
                nir.shape[0]
            ),
        )

        for index, (
            rgb_feature,
            nir_feature,
        ) in enumerate(
            zip(
                rgb_features,
                nir_features,
            )
        ):
            require(
                rgb_feature.shape
                == nir_feature.shape,
                f"RGB/NIR feature shape mismatch at stage {index}: "
                f"{tuple(rgb_feature.shape)} vs "
                f"{tuple(nir_feature.shape)}",
            )

        return (
            rgb_features,
            nir_features,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        require(
            isinstance(
                rgb,
                torch.Tensor,
            ),
            "RGB input must be Tensor.",
        )
        require(
            isinstance(
                nir,
                torch.Tensor,
            ),
            "NIR input must be Tensor.",
        )
        require(
            rgb.ndim == 4
            and rgb.shape[1]
            == RGB_CHANNELS,
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}.",
        )
        require(
            nir.ndim == 4
            and nir.shape[1]
            == NIR_CHANNELS,
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}.",
        )
        require(
            rgb.shape[0]
            == nir.shape[0]
            and rgb.shape[-2:]
            == nir.shape[-2:],
            "RGB/NIR batch or spatial dimensions differ.",
        )
        require(
            rgb.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ),
            f"RGB dtype is not floating point: {rgb.dtype}.",
        )
        require(
            nir.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ),
            f"NIR dtype is not floating point: {nir.dtype}.",
        )
        require(
            rgb.dtype
            == nir.dtype,
            f"RGB/NIR dtype mismatch: {rgb.dtype} vs {nir.dtype}.",
        )
        require(
            torch.isfinite(
                rgb
            ).all().item(),
            "RGB contains NaN/Inf.",
        )
        require(
            torch.isfinite(
                nir
            ).all().item(),
            "NIR contains NaN/Inf.",
        )

        (
            rgb_features,
            nir_features,
        ) = self._encode(
            rgb,
            nir,
        )

        # Exact C-noGate definition:
        #   Fused_i = 0.5 * RGB_i + 0.5 * NIR_i
        fused_features = tuple(
            (
                FIXED_RGB_WEIGHT
                * rgb_feature
                + FIXED_NIR_WEIGHT
                * nir_feature
            )
            for rgb_feature, nir_feature
            in zip(
                rgb_features,
                nir_features,
            )
        )

        for index, feature in enumerate(
            fused_features
        ):
            require(
                torch.isfinite(
                    feature
                ).all().item(),
                f"Fused stage {index} contains NaN/Inf.",
            )

        raw_logits = self.decode_head(
            fused_features
        )

        require(
            raw_logits.ndim == 4
            and raw_logits.shape[0]
            == rgb.shape[0]
            and raw_logits.shape[1]
            == NUM_CLASSES,
            "Raw logits shape invalid: "
            f"{tuple(raw_logits.shape)}.",
        )
        require(
            torch.isfinite(
                raw_logits
            ).all().item(),
            "Raw logits contain NaN/Inf.",
        )

        full_logits = F.interpolate(
            raw_logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        require(
            full_logits.shape
            == (
                rgb.shape[0],
                NUM_CLASSES,
                rgb.shape[-2],
                rgb.shape[-1],
            ),
            "Full logits shape invalid: "
            f"{tuple(full_logits.shape)}.",
        )

        if not return_details:
            return full_logits

        batch_size = int(
            rgb.shape[0]
        )

        weights = (
            self
            .fixed_fusion_weights
            .to(
                device=full_logits.device,
                dtype=full_logits.dtype,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        return {
            "logits": full_logits,
            "raw_logits": raw_logits,
            "fusion_weights": weights,
            "rgb_features": rgb_features,
            "nir_features": nir_features,
            "fused_features": fused_features,
        }


def build_model_c_nogate(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Tuple[
    SegFormerDualFixedFusion,
    Dict[str, Any],
]:
    """
    Build Model C-noGate from the exact frozen Model-A pretrained source.
    """
    project_root = Path(
        project_root
    ).resolve()

    require(
        checkpoint
        == DEFAULT_CHECKPOINT,
        "C-noGate v1 only permits the frozen Model-A pretrained checkpoint: "
        f"{DEFAULT_CHECKPOINT}",
    )

    # Build exactly the same pretrained RGB SegFormer + Potsdam 6-class head
    # used by Model A. This also reuses Model A's frozen HF revision.
    base_wrapper, base_meta = (
        build_model_a_rgb(
            project_root,
            checkpoint=checkpoint,
        )
    )

    base_model = (
        base_wrapper.model
    )

    rgb_encoder = (
        base_model.segformer
    )
    decode_head = (
        base_model.decode_head
    )

    # Copy pretrained RGB encoder BEFORE adapting the first layer.
    nir_encoder = copy.deepcopy(
        rgb_encoder
    )

    nir_adaptation = (
        _adapt_nir_encoder_first_layer(
            nir_encoder
        )
    )

    copy_audit = (
        _verify_encoder_copy(
            rgb_encoder,
            nir_encoder,
        )
    )

    model = SegFormerDualFixedFusion(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
    )

    require(
        model.fixed_fusion_weights.requires_grad
        is False,
        "Fixed fusion weights must not require gradients.",
    )
    require(
        tuple(
            model.fixed_fusion_weights.shape
        )
        == (
            NUM_SCALES,
            2,
        ),
        "Fixed fusion weight shape must be [4,2].",
    )
    require(
        torch.equal(
            model.fixed_fusion_weights,
            torch.full(
                (
                    NUM_SCALES,
                    2,
                ),
                0.5,
                dtype=torch.float32,
            ),
        ),
        "C-noGate fixed fusion weights are not exactly 0.5/0.5.",
    )

    meta: Dict[str, Any] = {
        "variant": "C-noGate",
        "model_id": MODEL_ID,
        "name": MODEL_NAME,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "checkpoint": (
            base_meta["checkpoint"]
        ),
        "resolved_revision": (
            base_meta[
                "resolved_revision"
            ]
        ),
        "num_labels": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_modalities": [
            "RGB",
            "NIR",
        ],
        "rgb_channels": (
            RGB_CHANNELS
        ),
        "nir_channels": (
            NIR_CHANNELS
        ),
        "dual_encoder": True,
        "multiscale_fusion": True,
        "fusion_scales": (
            NUM_SCALES
        ),
        "fusion_rule": (
            "F_i = 0.5 * F_RGB_i + 0.5 * F_NIR_i"
        ),
        "fixed_rgb_weight": (
            FIXED_RGB_WEIGHT
        ),
        "fixed_nir_weight": (
            FIXED_NIR_WEIGHT
        ),
        "fusion_trainable_parameters": 0,
        "quality_gate": False,
        "shared_decoder": True,
        "nir_encoder_initialization": (
            nir_adaptation
        ),
        "encoder_copy_audit": (
            copy_audit
        ),
        "parameters": (
            parameter_counts(
                model
            )
        ),
        "base_model_a_meta": (
            base_meta
        ),
        "hidden_sizes": list(
            EXPECTED_HIDDEN_SIZES
        ),
        "decoder_hidden_size": int(
            decode_head
            .config
            .decoder_hidden_size
        ),
        "semantic_loss_ignore_index": (
            IGNORE_INDEX
        ),
    }

    return model, meta


def _smoke_test(
    project_root: Path,
) -> None:
    from data_pipeline.potsdam_dataset import (
        PotsdamTrainDataset,
    )

    print("=" * 88)
    print(
        "Model C-noGate | Dual Encoder + Fixed Multi-Scale Fusion | smoke test"
    )
    print("=" * 88)

    dataset = PotsdamTrainDataset(
        project_root,
        epoch=0,
    )

    sample = dataset[0]

    rgb = (
        sample["rgb"]
        .unsqueeze(0)
    )
    nir = (
        sample["nir"]
        .unsqueeze(0)
    )
    labels = (
        sample["labels"]
        .unsqueeze(0)
    )

    model, meta = build_model_c_nogate(
        project_root
    )

    print(
        "fusion rule             : "
        f"{meta['fusion_rule']}"
    )
    print(
        "fusion trainable params : "
        f"{meta['fusion_trainable_parameters']}"
    )
    print(
        "NIR init                : "
        f"{meta['nir_encoder_initialization']['method']}"
    )
    print(
        "encoder copy audit      : "
        f"{meta['encoder_copy_audit']['only_first_patch_weight_changed']}"
    )
    print(
        "parameters              : "
        f"{meta['parameters']['total']:,}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    with torch.inference_mode():
        details = model(
            rgb.to(device),
            nir.to(device),
            return_details=True,
        )

    logits = details["logits"]
    raw_logits = details[
        "raw_logits"
    ]
    weights = details[
        "fusion_weights"
    ]

    require(
        tuple(
            logits.shape
        )
        == (
            1,
            NUM_CLASSES,
            512,
            512,
        ),
        f"Full logits smoke-test shape invalid: {tuple(logits.shape)}",
    )

    require(
        tuple(
            raw_logits.shape
        )
        == (
            1,
            NUM_CLASSES,
            128,
            128,
        ),
        f"Raw logits smoke-test shape invalid: {tuple(raw_logits.shape)}",
    )

    require(
        tuple(
            weights.shape
        )
        == (
            1,
            NUM_SCALES,
            2,
        ),
        f"Fusion weights shape invalid: {tuple(weights.shape)}",
    )

    require(
        torch.equal(
            weights,
            torch.full_like(
                weights,
                0.5,
            ),
        ),
        "Smoke-test fusion weights are not exactly 0.5.",
    )

    loss = F.cross_entropy(
        logits,
        labels.to(device),
        ignore_index=IGNORE_INDEX,
    )

    require(
        torch.isfinite(
            loss
        ).item(),
        "Smoke-test CE is non-finite.",
    )

    print(
        f"device                  : {device}"
    )
    print(
        f"raw logits              : {tuple(raw_logits.shape)}"
    )
    print(
        f"full logits             : {tuple(logits.shape)}"
    )
    print(
        f"fusion weights          : {weights[0].detach().cpu().tolist()}"
    )
    print(
        f"CE                      : {float(loss.item()):.6f}"
    )
    print(
        "FINAL STATUS            : PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model C-noGate dual-encoder fixed-fusion smoke test."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=(
            Path(__file__)
            .resolve()
            .parents[1]
        ),
    )

    args = parser.parse_args()

    project_root = (
        args.project_root
        .resolve()
    )

    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    _smoke_test(
        project_root
    )


if __name__ == "__main__":
    main()
