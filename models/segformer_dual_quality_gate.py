#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model C / Ours: Dual-Encoder SegFormer-B0 + Dynamic Multi-Scale Quality Gate.

Scientific role
---------------
This is the proposed model.

The architecture is intentionally identical to Model C-noGate except for one
controlled change:

    C-noGate:
        F_i = 0.5 * F_RGB_i + 0.5 * F_NIR_i

    Model C / Ours:
        F_i = w_RGB_i * F_RGB_i + w_NIR_i * F_NIR_i
        w_NIR_i = 1 - w_RGB_i

The same:
- RGB encoder,
- NIR encoder,
- NIR first-layer initialization,
- four fusion locations,
- shared pretrained SegFormer decoder,
- six Potsdam classes,

are inherited directly from the C-noGate builder.

Quality Gate
------------
At each SegFormer scale i, the gate receives a compact global quality
descriptor built from BOTH modality feature maps:

    mu_RGB  = channel-wise spatial mean(F_RGB)
    sd_RGB  = channel-wise spatial std(F_RGB)
    mu_NIR  = channel-wise spatial mean(F_NIR)
    sd_NIR  = channel-wise spatial std(F_NIR)

    q_i = concat(mu_RGB, sd_RGB, mu_NIR, sd_NIR)

For stage width C_i:
    q_i has 4*C_i features.

A single linear projection produces one sample-wise RGB reliability logit:

    z_i = Linear_i(q_i)
    w_RGB_i = sigmoid(z_i)
    w_NIR_i = 1 - w_RGB_i

The scalar weights are broadcast over C/H/W for feature fusion.

Why global sample-wise gates?
-----------------------------
- Directly interpretable as modality reliability per scale.
- Easy to compare across Clean / Noise / Blur / Underexposure.
- Produces exactly the RQ5 object of interest:
      does w_RGB decrease as RGB quality deteriorates?
- Adds only a tiny number of parameters.
- Avoids confounding the ablation with a large spatial-attention network.

Initialization
--------------
Every Quality Gate Linear weight and bias is initialized to EXACTLY ZERO.

Therefore, at step 0:

    z_i = 0
    sigmoid(z_i) = 0.5

for every sample and scale. Model C starts as exact fixed 0.5/0.5 fusion.

Gate construction is wrapped with torch.random.fork_rng(), so creating the
new gate parameters does NOT advance the global PyTorch RNG stream. With the
same experiment seed, the underlying encoder/decoder initialization remains
aligned with Model C-noGate.

Training
--------
No auxiliary gate supervision.
No corruption-aware training.
No gate regularization.

The gate is trained only through the same semantic segmentation CE objective
as A/B/C-noGate. This preserves the clean-training controlled experiment.

Expected location:
    models/segformer_dual_quality_gate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segformer_dual_fixed import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    EXPECTED_HIDDEN_SIZES,
    IGNORE_INDEX,
    NIR_INIT_METHOD,
    NUM_CLASSES,
    NUM_SCALES,
    SegFormerDualFixedFusion,
    build_model_c_nogate,
)


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Model C Quality Gate Protocol v1"

MODEL_ID = "C_QUALITY_GATE"
MODEL_NAME = "Model C (Quality Gate / Ours)"

RGB_CHANNELS = 3
NIR_CHANNELS = 1

GATE_TYPE = "global_channel_mean_std_linear_sigmoid"
GATE_DESCRIPTOR = (
    "concat(channelwise_spatial_mean_RGB, channelwise_spatial_std_RGB, "
    "channelwise_spatial_mean_NIR, channelwise_spatial_std_NIR)"
)
GATE_INITIAL_RGB_WEIGHT = 0.5
GATE_INITIAL_NIR_WEIGHT = 0.5
GATE_EPS = 1e-6


class ModelCQualityGateProtocolError(RuntimeError):
    pass


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ModelCQualityGateProtocolError(
            message
        )


def parameter_counts(
    module: nn.Module,
) -> Dict[str, int]:
    total = sum(
        int(parameter.numel())
        for parameter
        in module.parameters()
    )

    trainable = sum(
        int(parameter.numel())
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


class ScaleQualityGate(nn.Module):
    """
    One sample-wise modality-quality gate for one SegFormer feature scale.

    Input:
        rgb_feature [B,C,H,W]
        nir_feature [B,C,H,W]

    Output:
        weights [B,2]
            [:,0] = w_RGB
            [:,1] = w_NIR = 1 - w_RGB

        rgb_logit [B]
    """

    def __init__(
        self,
        channels: int,
        *,
        eps: float = GATE_EPS,
    ):
        super().__init__()

        require(
            int(channels) > 0,
            f"Gate channels must be positive, got {channels}.",
        )

        self.channels = int(
            channels
        )
        self.eps = float(
            eps
        )
        self.descriptor_dim = (
            4
            * self.channels
        )

        # nn.Linear normally consumes global RNG even though we overwrite every
        # parameter. Preserve the outer experiment RNG exactly.
        with torch.random.fork_rng(
            devices=[],
            enabled=True,
        ):
            self.projection = nn.Linear(
                self.descriptor_dim,
                1,
                bias=True,
            )

        # Exact fixed-fusion-equivalent initialization.
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()

    def _descriptor(
        self,
        rgb_feature: torch.Tensor,
        nir_feature: torch.Tensor,
    ) -> torch.Tensor:
        require(
            rgb_feature.shape
            == nir_feature.shape,
            "RGB/NIR feature shapes differ in Quality Gate: "
            f"{tuple(rgb_feature.shape)} vs {tuple(nir_feature.shape)}.",
        )
        require(
            rgb_feature.ndim == 4,
            "Quality Gate expects BCHW features.",
        )
        require(
            int(
                rgb_feature.shape[1]
            )
            == self.channels,
            "Quality Gate feature channel mismatch: "
            f"expected C={self.channels}, "
            f"got C={rgb_feature.shape[1]}.",
        )

        # Compute quality statistics in FP32 even when outer inference/training
        # uses FP16 autocast. This keeps small variance estimates stable.
        with torch.autocast(
            device_type=(
                rgb_feature
                .device
                .type
            ),
            enabled=False,
        ):
            rgb32 = (
                rgb_feature
                .float()
            )
            nir32 = (
                nir_feature
                .float()
            )

            (
                rgb_var,
                rgb_mean,
            ) = torch.var_mean(
                rgb32,
                dim=(-2, -1),
                correction=0,
            )

            (
                nir_var,
                nir_mean,
            ) = torch.var_mean(
                nir32,
                dim=(-2, -1),
                correction=0,
            )

            rgb_std = torch.sqrt(
                torch.clamp(
                    rgb_var,
                    min=0.0,
                )
                + self.eps
            )

            nir_std = torch.sqrt(
                torch.clamp(
                    nir_var,
                    min=0.0,
                )
                + self.eps
            )

            descriptor = torch.cat(
                (
                    rgb_mean,
                    rgb_std,
                    nir_mean,
                    nir_std,
                ),
                dim=1,
            )

        require(
            tuple(
                descriptor.shape
            )
            == (
                rgb_feature.shape[0],
                self.descriptor_dim,
            ),
            "Quality Gate descriptor shape invalid: "
            f"{tuple(descriptor.shape)}.",
        )

        require(
            torch.isfinite(
                descriptor
            ).all().item(),
            "Quality Gate descriptor contains NaN/Inf.",
        )

        return descriptor

    def forward(
        self,
        rgb_feature: torch.Tensor,
        nir_feature: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        descriptor = self._descriptor(
            rgb_feature,
            nir_feature,
        )

        # Keep gate computation FP32 and outside AMP.
        with torch.autocast(
            device_type=(
                rgb_feature
                .device
                .type
            ),
            enabled=False,
        ):
            rgb_logit = (
                self.projection(
                    descriptor
                    .float()
                )
                .squeeze(-1)
            )

            w_rgb = torch.sigmoid(
                rgb_logit
            )

            w_nir = (
                1.0
                - w_rgb
            )

            weights = torch.stack(
                (
                    w_rgb,
                    w_nir,
                ),
                dim=-1,
            )

        require(
            torch.isfinite(
                weights
            ).all().item(),
            "Quality Gate weights contain NaN/Inf.",
        )

        require(
            bool(
                torch.all(
                    weights
                    >= 0.0
                ).item()
            )
            and bool(
                torch.all(
                    weights
                    <= 1.0
                ).item()
            ),
            "Quality Gate weights escaped [0,1].",
        )

        pair_sum_error = float(
            (
                weights.sum(
                    dim=-1
                )
                - 1.0
            )
            .abs()
            .max()
            .item()
        )

        require(
            pair_sum_error
            <= 1e-6,
            "Quality Gate modality weights do not sum to 1: "
            f"max_abs_error={pair_sum_error}.",
        )

        return (
            weights,
            rgb_logit,
        )


class SegFormerDualQualityGate(nn.Module):
    """
    Model C / Ours.

    Identical dual encoders + shared decoder to C-noGate.
    Only the fusion weights become data-dependent.
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
            len(
                hidden_sizes
            )
            == NUM_SCALES,
            f"Expected {NUM_SCALES} hidden sizes, got {len(hidden_sizes)}.",
        )

        self.rgb_encoder = (
            rgb_encoder
        )
        self.nir_encoder = (
            nir_encoder
        )
        self.decode_head = (
            decode_head
        )

        self.hidden_sizes = tuple(
            int(value)
            for value
            in hidden_sizes
        )

        self.quality_gates = nn.ModuleList(
            [
                ScaleQualityGate(
                    channels,
                )
                for channels
                in self.hidden_sizes
            ]
        )

    def _validate_inputs(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ) -> None:
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

        require(
            len(
                rgb_features
            )
            == NUM_SCALES,
            "RGB encoder did not return four scale features.",
        )
        require(
            len(
                nir_features
            )
            == NUM_SCALES,
            "NIR encoder did not return four scale features.",
        )

        previous_h = None
        previous_w = None

        for stage_index, (
            rgb_feature,
            nir_feature,
            expected_channels,
        ) in enumerate(
            zip(
                rgb_features,
                nir_features,
                self.hidden_sizes,
            )
        ):
            require(
                rgb_feature.shape
                == nir_feature.shape,
                f"Stage {stage_index} RGB/NIR feature shape mismatch: "
                f"{tuple(rgb_feature.shape)} vs "
                f"{tuple(nir_feature.shape)}.",
            )

            require(
                rgb_feature.ndim
                == 4,
                f"Stage {stage_index} feature must be BCHW.",
            )

            require(
                int(
                    rgb_feature.shape[1]
                )
                == expected_channels,
                f"Stage {stage_index} expected C={expected_channels}, "
                f"got C={rgb_feature.shape[1]}.",
            )

            require(
                torch.isfinite(
                    rgb_feature
                ).all().item()
                and torch.isfinite(
                    nir_feature
                ).all().item(),
                f"Stage {stage_index} feature contains NaN/Inf.",
            )

            height = int(
                rgb_feature.shape[-2]
            )
            width = int(
                rgb_feature.shape[-1]
            )

            if previous_h is not None:
                require(
                    height < previous_h
                    and width < previous_w,
                    "Feature pyramid is not spatially descending.",
                )

            previous_h = height
            previous_w = width

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
        self._validate_inputs(
            rgb,
            nir,
        )

        (
            rgb_features,
            nir_features,
        ) = self._encode(
            rgb,
            nir,
        )

        fused_features = []
        fusion_weights = []
        rgb_logits = []

        for stage_index, (
            rgb_feature,
            nir_feature,
            gate,
        ) in enumerate(
            zip(
                rgb_features,
                nir_features,
                self.quality_gates,
            )
        ):
            (
                weights,
                rgb_logit,
            ) = gate(
                rgb_feature,
                nir_feature,
            )

            w_rgb = (
                weights[
                    :,
                    0,
                ]
                .to(
                    dtype=(
                        rgb_feature
                        .dtype
                    )
                )
                .view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            w_nir = (
                weights[
                    :,
                    1,
                ]
                .to(
                    dtype=(
                        nir_feature
                        .dtype
                    )
                )
                .view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            fused = (
                w_rgb
                * rgb_feature
                + w_nir
                * nir_feature
            )

            require(
                torch.isfinite(
                    fused
                ).all().item(),
                f"Fused stage {stage_index} contains NaN/Inf.",
            )

            fused_features.append(
                fused
            )
            fusion_weights.append(
                weights
            )
            rgb_logits.append(
                rgb_logit
            )

        fused_features_tuple = tuple(
            fused_features
        )

        raw_logits = self.decode_head(
            fused_features_tuple
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

        weights_tensor = torch.stack(
            fusion_weights,
            dim=1,
        )

        gate_logits_tensor = torch.stack(
            rgb_logits,
            dim=1,
        )

        require(
            tuple(
                weights_tensor.shape
            )
            == (
                rgb.shape[0],
                NUM_SCALES,
                2,
            ),
            "Fusion weight tensor shape invalid: "
            f"{tuple(weights_tensor.shape)}.",
        )

        require(
            tuple(
                gate_logits_tensor.shape
            )
            == (
                rgb.shape[0],
                NUM_SCALES,
            ),
            "Gate logit tensor shape invalid: "
            f"{tuple(gate_logits_tensor.shape)}.",
        )

        return {
            "logits": (
                full_logits
            ),
            "raw_logits": (
                raw_logits
            ),
            "fusion_weights": (
                weights_tensor
            ),
            "gate_rgb_logits": (
                gate_logits_tensor
            ),
        }


def _gate_parameter_count(
    model: SegFormerDualQualityGate,
) -> int:
    return sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.quality_gates.parameters()
    )


def _audit_initial_gate_state(
    model: SegFormerDualQualityGate,
) -> Dict[str, Any]:
    stage_rows = []

    for stage_index, (
        gate,
        channels,
    ) in enumerate(
        zip(
            model.quality_gates,
            model.hidden_sizes,
        )
    ):
        weight = (
            gate
            .projection
            .weight
            .detach()
        )
        bias = (
            gate
            .projection
            .bias
            .detach()
        )

        require(
            torch.count_nonzero(
                weight
            ).item()
            == 0,
            f"Stage {stage_index} gate weight is not zero-initialized.",
        )

        require(
            torch.count_nonzero(
                bias
            ).item()
            == 0,
            f"Stage {stage_index} gate bias is not zero-initialized.",
        )

        expected_parameters = (
            4
            * int(
                channels
            )
            + 1
        )

        actual_parameters = sum(
            int(
                parameter.numel()
            )
            for parameter
            in gate.parameters()
        )

        require(
            actual_parameters
            == expected_parameters,
            f"Stage {stage_index} gate parameter count mismatch: "
            f"{actual_parameters} != {expected_parameters}.",
        )

        stage_rows.append(
            {
                "stage": (
                    stage_index + 1
                ),
                "channels": int(
                    channels
                ),
                "descriptor_dim": (
                    gate.descriptor_dim
                ),
                "parameters": (
                    actual_parameters
                ),
                "weight_zero_initialized": True,
                "bias_zero_initialized": True,
                "initial_w_rgb": (
                    GATE_INITIAL_RGB_WEIGHT
                ),
                "initial_w_nir": (
                    GATE_INITIAL_NIR_WEIGHT
                ),
            }
        )

    return {
        "all_gates_zero_initialized": True,
        "initial_weights_exactly_half": True,
        "stages": stage_rows,
        "total_gate_parameters": (
            _gate_parameter_count(
                model
            )
        ),
    }


def build_model_c_quality_gate(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Tuple[
    SegFormerDualQualityGate,
    Dict[str, Any],
]:
    """
    Build Model C from exactly the same base initialization as C-noGate.

    IMPORTANT:
    This does NOT load the trained C-noGate final checkpoint.
    It rebuilds the common pretrained initialization, preserving a fair
    independent training comparison.
    """
    project_root = Path(
        project_root
    ).resolve()

    # The fixed model builder is the single source of truth for:
    # - RGB pretrained encoder,
    # - NIR encoder copy,
    # - NIR first-layer mean-RGB initialization,
    # - pretrained shared decoder.
    fixed_model, fixed_meta = (
        build_model_c_nogate(
            project_root,
            checkpoint=checkpoint,
        )
    )

    require(
        isinstance(
            fixed_model,
            SegFormerDualFixedFusion,
        ),
        "C-noGate builder returned an unexpected model class.",
    )

    rgb_encoder = (
        fixed_model.rgb_encoder
    )
    nir_encoder = (
        fixed_model.nir_encoder
    )
    decode_head = (
        fixed_model.decode_head
    )

    model = SegFormerDualQualityGate(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
        hidden_sizes=(
            EXPECTED_HIDDEN_SIZES
        ),
    )

    gate_audit = (
        _audit_initial_gate_state(
            model
        )
    )

    fixed_parameter_count = int(
        fixed_meta[
            "parameters"
        ][
            "total"
        ]
    )

    quality_parameter_count = int(
        parameter_counts(
            model
        )[
            "total"
        ]
    )

    gate_parameter_count = int(
        gate_audit[
            "total_gate_parameters"
        ]
    )

    require(
        quality_parameter_count
        == (
            fixed_parameter_count
            + gate_parameter_count
        ),
        "Model C parameter count should equal C-noGate + Quality Gate params: "
        f"{quality_parameter_count} != "
        f"{fixed_parameter_count} + {gate_parameter_count}.",
    )

    meta: Dict[str, Any] = {
        "variant": "C",
        "model_id": (
            MODEL_ID
        ),
        "name": (
            MODEL_NAME
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "checkpoint": (
            fixed_meta[
                "checkpoint"
            ]
        ),
        "resolved_revision": (
            fixed_meta[
                "resolved_revision"
            ]
        ),
        "num_labels": (
            NUM_CLASSES
        ),
        "class_names": (
            CLASS_NAMES
        ),
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
        "quality_gate": True,
        "fusion_rule": (
            "F_i = w_RGB_i * F_RGB_i + "
            "(1-w_RGB_i) * F_NIR_i"
        ),
        "gate": {
            "type": (
                GATE_TYPE
            ),
            "descriptor": (
                GATE_DESCRIPTOR
            ),
            "per_sample": True,
            "per_scale": True,
            "spatially_uniform_within_scale": True,
            "output": (
                "w_RGB=sigmoid(linear(q)); "
                "w_NIR=1-w_RGB"
            ),
            "statistics_precision": (
                "float32"
            ),
            "initial_w_rgb": (
                GATE_INITIAL_RGB_WEIGHT
            ),
            "initial_w_nir": (
                GATE_INITIAL_NIR_WEIGHT
            ),
            "auxiliary_supervision": False,
            "regularization": None,
            "audit": (
                gate_audit
            ),
        },
        "shared_decoder": True,
        "nir_encoder_initialization": (
            NIR_INIT_METHOD
        ),
        "common_initialization_source": (
            "build_model_c_nogate; trained C-noGate checkpoint is NOT loaded"
        ),
        "c_nogate_base_meta": (
            fixed_meta
        ),
        "parameters": (
            parameter_counts(
                model
            )
        ),
        "gate_parameters": (
            gate_parameter_count
        ),
        "hidden_sizes": list(
            EXPECTED_HIDDEN_SIZES
        ),
        "semantic_loss_ignore_index": (
            IGNORE_INDEX
        ),
        "controlled_ablation": {
            "same_rgb_encoder_as_c_nogate": True,
            "same_nir_encoder_as_c_nogate": True,
            "same_nir_initialization_as_c_nogate": True,
            "same_decoder_as_c_nogate": True,
            "same_four_fusion_locations_as_c_nogate": True,
            "initial_function_matches_fixed_0.5_fusion": True,
            "only_intended_architecture_change": (
                "fixed 0.5/0.5 weights -> dynamic sample-wise per-scale "
                "Quality Gate weights"
            ),
        },
    }

    return (
        model,
        meta,
    )


def _smoke_test(
    project_root: Path,
) -> None:
    from data_pipeline.potsdam_dataset import (
        PotsdamTrainDataset,
    )

    print("=" * 96)
    print(
        "Model C / Ours | Dual Encoder + Dynamic Multi-Scale Quality Gate"
    )
    print("=" * 96)

    dataset = PotsdamTrainDataset(
        project_root,
        epoch=0,
    )

    sample = dataset[0]

    rgb = (
        sample[
            "rgb"
        ]
        .unsqueeze(0)
    )
    nir = (
        sample[
            "nir"
        ]
        .unsqueeze(0)
    )
    labels = (
        sample[
            "labels"
        ]
        .unsqueeze(0)
    )

    model, meta = (
        build_model_c_quality_gate(
            project_root
        )
    )

    print(
        "gate type              : "
        f"{meta['gate']['type']}"
    )
    print(
        "gate descriptor        : "
        f"{meta['gate']['descriptor']}"
    )
    print(
        "gate parameters        : "
        f"{meta['gate_parameters']:,}"
    )
    print(
        "total parameters       : "
        f"{meta['parameters']['total']:,}"
    )
    print(
        "initial gate audit     : "
        f"{meta['gate']['audit']['all_gates_zero_initialized']}"
    )
    print(
        "trained C-noGate loaded: NO"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(
        device
    )
    model.eval()

    with torch.inference_mode():
        details = model(
            rgb.to(
                device
            ),
            nir.to(
                device
            ),
            return_details=True,
        )

    logits = (
        details[
            "logits"
        ]
    )
    raw_logits = (
        details[
            "raw_logits"
        ]
    )
    weights = (
        details[
            "fusion_weights"
        ]
    )
    gate_logits = (
        details[
            "gate_rgb_logits"
        ]
    )

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
        "Smoke-test full logits shape invalid: "
        f"{tuple(logits.shape)}.",
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
        "Smoke-test raw logits shape invalid: "
        f"{tuple(raw_logits.shape)}.",
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
        "Smoke-test gate weight shape invalid.",
    )

    require(
        torch.equal(
            weights,
            torch.full_like(
                weights,
                0.5,
            ),
        ),
        "Zero-initialized Quality Gates do not output exact 0.5/0.5.",
    )

    require(
        torch.equal(
            gate_logits,
            torch.zeros_like(
                gate_logits
            ),
        ),
        "Zero-initialized Quality Gate logits are not exactly zero.",
    )

    loss = F.cross_entropy(
        logits,
        labels.to(
            device
        ),
        ignore_index=(
            IGNORE_INDEX
        ),
    )

    require(
        torch.isfinite(
            loss
        ).item(),
        "Smoke-test CE is non-finite.",
    )

    print(
        f"device                 : {device}"
    )
    print(
        f"raw logits             : {tuple(raw_logits.shape)}"
    )
    print(
        f"full logits            : {tuple(logits.shape)}"
    )
    print(
        "initial fusion weights : "
        f"{weights[0].detach().cpu().tolist()}"
    )
    print(
        "initial gate logits    : "
        f"{gate_logits[0].detach().cpu().tolist()}"
    )
    print(
        f"CE                     : {float(loss.item()):.6f}"
    )
    print(
        "FINAL STATUS           : PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model C / Quality Gate smoke test."
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

    if str(
        project_root
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                project_root
            ),
        )

    _smoke_test(
        project_root
    )


if __name__ == "__main__":
    main()
