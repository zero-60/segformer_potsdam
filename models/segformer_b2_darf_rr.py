#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RR-DARF:
SegFormer-B2 + Relative-Reliability Degradation-Aware Residual Fusion.

Why this revision exists
------------------------
The formal Joint RGB+NIR experiments showed that the previous DARF gate
saturated close to 1.0 and behaved almost identically to a separately-trained
Fixed g=1.0 baseline. Cross-modal stress tests further showed weak / inconsistent
gate response when NIR reliability changed.

This revision keeps the scientifically useful RGB-anchored residual rule:

    F_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)

but changes the gate implementation/training interface:

1) Gate starts at g=0.5.
2) Gate logit is smoothly bounded:
       z = z_max * tanh(z_raw / z_max)
       g = sigmoid(z)
   This prevents numerical/optimization saturation at exactly 0 or 1 while
   preserving a wide usable range.
3) gate_only(...) exposes the quality gates without running the decoder.
   Training can therefore apply RELATIVE reliability supervision:
       g(NIR worse) < g(neutral) < g(RGB worse)
   without assigning an absolute severity->gate target.
4) The auxiliary gate path can detach encoder features, so relative supervision
   updates the gates without turning the auxiliary corruption branch into a
   second segmentation-training distribution.

The NIR residual adapters, RGB/NIR encoders and SegFormer decode head stay
compatible with the previous DARF implementation.

Expected location:
    models/segformer_b2_darf_rr.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

import models.segformer_b2_darf as base


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "RR-DARF-B2 Protocol v1"

MODEL_ID = "B2_DARF_RR_JOINT_ROBUST4"
MODEL_NAME = "SegFormer-B2 RR-DARF Joint Robust-4"

DEFAULT_CHECKPOINT = base.DEFAULT_CHECKPOINT
NUM_CLASSES = base.NUM_CLASSES
IGNORE_INDEX = base.IGNORE_INDEX
NUM_SCALES = base.NUM_SCALES
EXPECTED_HIDDEN_SIZES = base.EXPECTED_HIDDEN_SIZES

INITIAL_NIR_GATE = 0.50
MAX_GATE_LOGIT = 4.0

GATE_TYPE = "relative_reliability_bounded_feature_quality_mlp"
GATE_DESCRIPTOR = base.GATE_DESCRIPTOR


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class RelativeReliabilityGate(base.ResidualQualityGate):
    """
    Same feature-quality descriptor as the previous DARF gate, but with:

        raw_logit -> bounded_logit -> sigmoid

    The bounded logit avoids the almost-irreversible sigmoid saturation observed
    in the previous M3' experiment. The training script adds relative ranking
    supervision; this module itself does not encode any severity target.
    """

    def __init__(
        self,
        channels: int,
        *,
        initial_gate: float = INITIAL_NIR_GATE,
        max_gate_logit: float = MAX_GATE_LOGIT,
    ):
        super().__init__(channels)

        initial_gate = float(initial_gate)
        max_gate_logit = float(max_gate_logit)

        require(
            0.0 < initial_gate < 1.0,
            f"initial_gate must be in (0,1), got {initial_gate}",
        )
        require(
            max_gate_logit > 0.0,
            f"max_gate_logit must be > 0, got {max_gate_logit}",
        )

        self.initial_gate = initial_gate
        self.max_gate_logit = max_gate_logit

        # For g=0.5 the desired bounded logit is exactly zero, therefore
        # raw_logit=0 is also exact and stable.
        with torch.no_grad():
            self.fc2.weight.zero_()
            if abs(initial_gate - 0.5) <= 1e-12:
                self.fc2.bias.zero_()
            else:
                target_logit = torch.logit(
                    torch.tensor(initial_gate, dtype=self.fc2.bias.dtype)
                ).item()
                # Invert z = M*tanh(raw/M).
                ratio = max(
                    min(target_logit / max_gate_logit, 0.999999),
                    -0.999999,
                )
                raw_bias = max_gate_logit * 0.5 * (
                    torch.log(
                        torch.tensor((1.0 + ratio) / (1.0 - ratio))
                    ).item()
                )
                self.fc2.bias.fill_(raw_bias)

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._descriptor(rgb, nir)

        with torch.autocast(
            device_type=rgb.device.type,
            enabled=False,
        ):
            raw_logit = (
                self.fc2(
                    self.act(
                        self.fc1(q.float())
                    )
                )
                .squeeze(-1)
            )

            bounded_logit = (
                float(self.max_gate_logit)
                * torch.tanh(
                    raw_logit
                    / float(self.max_gate_logit)
                )
            )

            strength = torch.sigmoid(
                bounded_logit
            )

        require(
            torch.isfinite(strength).all().item(),
            "RR-DARF gate strength contains NaN/Inf.",
        )
        require(
            torch.isfinite(bounded_logit).all().item(),
            "RR-DARF bounded gate logit contains NaN/Inf.",
        )

        return (
            strength,
            bounded_logit,
            raw_logit,
        )


class SegFormerB2DarfRR(base.SegFormerB2DARF):
    """
    Previous DARF backbone/adapters with revised relative-reliability gates.
    """

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        nir_encoder: nn.Module,
        decode_head: nn.Module,
        hidden_sizes: Sequence[int],
        initial_gate: float = INITIAL_NIR_GATE,
        max_gate_logit: float = MAX_GATE_LOGIT,
    ):
        super().__init__(
            rgb_encoder=rgb_encoder,
            nir_encoder=nir_encoder,
            decode_head=decode_head,
            hidden_sizes=hidden_sizes,
        )

        self.quality_gates = nn.ModuleList(
            [
                RelativeReliabilityGate(
                    c,
                    initial_gate=initial_gate,
                    max_gate_logit=max_gate_logit,
                )
                for c in self.hidden_sizes
            ]
        )

        self.initial_gate = float(initial_gate)
        self.max_gate_logit = float(max_gate_logit)

    def _gate_from_features(
        self,
        rgb_features,
        nir_features,
    ) -> Dict[str, torch.Tensor]:
        strengths = []
        bounded_logits = []
        raw_logits = []

        for r, n, gate in zip(
            rgb_features,
            nir_features,
            self.quality_gates,
        ):
            strength, bounded, raw = gate(
                r,
                n,
            )
            strengths.append(strength)
            bounded_logits.append(bounded)
            raw_logits.append(raw)

        return {
            "nir_gate_strength": torch.stack(
                strengths,
                dim=1,
            ),
            "nir_gate_logits": torch.stack(
                bounded_logits,
                dim=1,
            ),
            "nir_gate_raw_logits": torch.stack(
                raw_logits,
                dim=1,
            ),
        }

    def gate_only(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        *,
        detach_encoders: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute only gate outputs.

        detach_encoders=True:
            Encoders run without gradient and returned features are detached.
            The ranking branch then updates quality_gates only, keeping the
            auxiliary reliability curriculum separate from semantic feature
            learning.
        """
        if detach_encoders:
            with torch.no_grad():
                rgb_features, nir_features = self._encode(
                    rgb,
                    nir,
                )

            rgb_features = tuple(
                x.detach()
                for x in rgb_features
            )
            nir_features = tuple(
                x.detach()
                for x in nir_features
            )
        else:
            rgb_features, nir_features = self._encode(
                rgb,
                nir,
            )

        return self._gate_from_features(
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
            rgb.ndim == 4
            and rgb.shape[1] == 3,
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}",
        )
        require(
            nir.ndim == 4
            and nir.shape[1] == 1,
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}",
        )
        require(
            rgb.shape[0] == nir.shape[0]
            and rgb.shape[-2:] == nir.shape[-2:],
            "RGB/NIR inputs are not aligned.",
        )

        rgb_features, nir_features = self._encode(
            rgb,
            nir,
        )

        fused = []
        gate_strengths = []
        gate_logits = []
        gate_raw_logits = []
        residual_abs_mean = []

        for r, n, adapter, gate in zip(
            rgb_features,
            nir_features,
            self.nir_adapters,
            self.quality_gates,
        ):
            strength, bounded_logit, raw_logit = gate(
                r,
                n,
            )
            residual = adapter(n)

            s = (
                strength
                .to(dtype=r.dtype)
                .view(-1, 1, 1, 1)
            )
            f = r + s * residual

            require(
                torch.isfinite(f).all().item(),
                "RR-DARF fused feature has NaN/Inf.",
            )

            fused.append(f)
            gate_strengths.append(strength)
            gate_logits.append(bounded_logit)
            gate_raw_logits.append(raw_logit)
            residual_abs_mean.append(
                residual.detach().float().abs().mean()
            )

        fused_tuple = tuple(fused)
        raw_logits = self.decode_head(
            fused_tuple
        )

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
            "nir_gate_strength": torch.stack(
                gate_strengths,
                dim=1,
            ),
            "nir_gate_logits": torch.stack(
                gate_logits,
                dim=1,
            ),
            "nir_gate_raw_logits": torch.stack(
                gate_raw_logits,
                dim=1,
            ),
            "residual_abs_mean": torch.stack(
                residual_abs_mean,
            ),
        }


def build_model_b2_darf_rr(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    initial_gate: float = INITIAL_NIR_GATE,
    max_gate_logit: float = MAX_GATE_LOGIT,
) -> Tuple[SegFormerB2DarfRR, Dict[str, Any]]:
    project_root = Path(project_root).resolve()

    require(
        checkpoint == DEFAULT_CHECKPOINT,
        f"RR-DARF only permits checkpoint {DEFAULT_CHECKPOINT}",
    )

    id2label, label2id = base._class_maps()

    config = SegformerConfig.from_pretrained(
        checkpoint
    )

    require(
        int(config.num_labels) == 150,
        f"Expected 150 ADE labels, got {config.num_labels}.",
    )
    require(
        int(getattr(config, "num_channels", 3)) == 3,
        "B2 checkpoint is not 3-channel.",
    )
    require(
        tuple(int(x) for x in config.hidden_sizes)
        == EXPECTED_HIDDEN_SIZES,
        f"Unexpected B2 hidden sizes: {config.hidden_sizes}",
    )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    pretrained, loading_info = (
        SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
    )

    loading_summary = (
        base.validate_pretrained_loading_info(
            loading_info
        )
    )

    rgb_encoder = pretrained.segformer
    nir_encoder = copy.deepcopy(
        rgb_encoder
    )
    nir_init = base._adapt_nir_first_layer(
        nir_encoder
    )
    decode_head = pretrained.decode_head

    model = SegFormerB2DarfRR(
        rgb_encoder=rgb_encoder,
        nir_encoder=nir_encoder,
        decode_head=decode_head,
        hidden_sizes=config.hidden_sizes,
        initial_gate=initial_gate,
        max_gate_logit=max_gate_logit,
    )

    residual_audit = base._audit_zero_residual(
        model
    )

    meta = {
        "variant": "DARF_RR",
        "model_id": MODEL_ID,
        "name": MODEL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": checkpoint,
        "resolved_revision": getattr(
            pretrained.config,
            "_commit_hash",
            None,
        ),
        "backbone": "SegFormer-B2",
        "num_labels": NUM_CLASSES,
        "class_names": base.CLASS_NAMES,
        "hidden_sizes": list(
            EXPECTED_HIDDEN_SIZES
        ),
        "decoder_hidden_size": int(
            config.decoder_hidden_size
        ),
        "dual_encoder": True,
        "fusion": (
            "RGB-anchored bounded relative-reliability "
            "gated NIR residual"
        ),
        "fusion_rule": (
            "F_fused_i = F_RGB_i + g_i * Adapter_i(F_NIR_i)"
        ),
        "quality_gate": True,
        "gate_type": GATE_TYPE,
        "gate_descriptor": GATE_DESCRIPTOR,
        "initial_nir_gate_strength": float(
            initial_gate
        ),
        "max_gate_logit": float(
            max_gate_logit
        ),
        "gate_strength_range": [
            float(
                torch.sigmoid(
                    torch.tensor(
                        -float(max_gate_logit)
                    )
                ).item()
            ),
            float(
                torch.sigmoid(
                    torch.tensor(
                        float(max_gate_logit)
                    )
                ).item()
            ),
        ],
        "nir_encoder_initialization": nir_init,
        "residual_adapter_audit": residual_audit,
        "loading_info": loading_summary,
        "parameters": base.parameter_counts(
            model
        ),
        "semantic_loss_ignore_index": IGNORE_INDEX,
    }

    return model, meta


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    model, meta = build_model_b2_darf_rr(root)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device).eval()

    rgb = torch.zeros(
        1, 3, 512, 512,
        device=device,
    )
    nir = torch.zeros(
        1, 1, 512, 512,
        device=device,
    )

    with torch.inference_mode():
        details = model(
            rgb,
            nir,
            return_details=True,
        )

    gates = details[
        "nir_gate_strength"
    ]

    require(
        tuple(gates.shape)
        == (1, NUM_SCALES),
        f"Bad gate shape: {tuple(gates.shape)}",
    )

    print("RR-DARF smoke test: PASS")
    print("model_id:", meta["model_id"])
    print(
        "initial gates:",
        gates[0].detach().cpu().tolist(),
    )
