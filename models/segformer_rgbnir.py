#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B: RGB + NIR -> direct 4-channel SegFormer-B0.

Experiment role
---------------
This is the project's early/direct-fusion multimodal baseline:

    normalized RGB [B,3,H,W]
             +
    normalized NIR [B,1,H,W]
             |
             v
      concatenate channels
             |
             v
    SegFormer-B0 (4-channel input)
             |
             v
      6-class full-res logits

Controlled-variable design
--------------------------
- Same Hugging Face ADE20K SegFormer-B0 checkpoint as Model A.
- Same 6 Potsdam classes.
- Same decoder/head architecture as Model A.
- No dual encoder.
- No fixed fusion.
- No Quality Gate.
- No corruption inside the model.
- Dataset performs RGB and NIR normalization.
- External training script computes the same CE loss as Model A.

4-channel pretrained initialization
-----------------------------------
The ADE20K checkpoint is RGB-only. We therefore:

1. Load the pretrained 3-channel model exactly as Model A does.
2. Keep every pretrained parameter unchanged.
3. Replace only the FIRST overlapping patch-embedding Conv2d:
       [out, 3, 7, 7] -> [out, 4, 7, 7]
4. Copy RGB kernels exactly:
       W_4[:, 0:3] = W_RGB
5. Initialize the NIR kernel as the mean of the pretrained RGB kernels:
       W_4[:, 3:4] = mean(W_RGB, dim=input_channel)
6. Copy the bias exactly.

This avoids randomly discarding the pretrained first layer while giving NIR a
neutral, deterministic initialization. The extra Conv2d construction is done
inside torch.random.fork_rng(), so adapting the input layer does NOT advance
the global PyTorch RNG stream. This helps keep the training random stream as
comparable as possible with Model A.

Expected location:
    models/segformer_rgbnir.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Model B RGB+NIR Direct-4CH Protocol v1"

DEFAULT_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"

NUM_CLASSES = 6
IGNORE_INDEX = 255
RGB_CHANNELS = 3
NIR_CHANNELS = 1
INPUT_CHANNELS = 4

CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]

NIR_INIT_METHOD = "mean_of_pretrained_rgb_patch_embed_kernels"


class ModelBProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelBProtocolError(message)


def _class_maps() -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


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
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _normalize_loading_key(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (tuple, list)) and item:
        return str(item[0])
    return str(item)


def _classifier_key_allowed(key: str) -> bool:
    return (
        key.endswith("decode_head.classifier.weight")
        or key.endswith("decode_head.classifier.bias")
        or key == "decode_head.classifier.weight"
        or key == "decode_head.classifier.bias"
    )


def validate_pretrained_loading_info(
    info: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """
    At pretrained-load time the model is still 3-channel, so the ONLY allowed
    shape mismatch is ADE20K 150-class classifier -> Potsdam 6-class classifier.
    """
    mismatched = [
        _normalize_loading_key(x)
        for x in info.get("mismatched_keys", [])
    ]
    missing = [
        _normalize_loading_key(x)
        for x in info.get("missing_keys", [])
    ]
    unexpected = [
        _normalize_loading_key(x)
        for x in info.get("unexpected_keys", [])
    ]

    bad_mismatch = [
        key for key in mismatched
        if not _classifier_key_allowed(key)
    ]
    bad_missing = [
        key for key in missing
        if not _classifier_key_allowed(key)
    ]

    require(
        not bad_mismatch,
        "Unexpected pretrained shape mismatch outside final classifier: "
        f"{bad_mismatch}",
    )
    require(
        not bad_missing,
        "Unexpected pretrained missing key outside final classifier: "
        f"{bad_missing}",
    )
    require(
        not unexpected,
        f"Unexpected pretrained keys: {unexpected}",
    )

    allowed = set(mismatched) | set(missing)

    require(
        any(
            key.endswith("decode_head.classifier.weight")
            for key in allowed
        ),
        "Did not detect expected ADE150 -> Potsdam6 classifier.weight change.",
    )
    require(
        any(
            key.endswith("decode_head.classifier.bias")
            for key in allowed
        ),
        "Did not detect expected ADE150 -> Potsdam6 classifier.bias change.",
    )

    return {
        "mismatched_keys": mismatched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _load_model_a_frozen_revision(
    project_root: Path,
) -> Optional[str]:
    """
    Reuse Model A's resolved HF revision when its frozen model protocol exists.
    This prevents Model A and Model B from silently loading different revisions
    of the same checkpoint name.
    """
    protocol_path = (
        project_root
        / "data"
        / "processed"
        / "potsdam"
        / "model_a_rgb_protocol.json"
    )

    if not protocol_path.is_file():
        return None

    try:
        obj = json.loads(protocol_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelBProtocolError(
            f"Failed to read Model A protocol: {protocol_path}\n{exc}"
        ) from exc

    model_meta = obj.get("model", {})
    if not isinstance(model_meta, dict):
        return None

    checkpoint = model_meta.get("checkpoint")
    require(
        checkpoint in (None, DEFAULT_CHECKPOINT),
        "Model A protocol uses a different pretrained checkpoint: "
        f"{checkpoint!r}",
    )

    revision = model_meta.get("resolved_revision")
    return str(revision) if revision else None


def _first_patch_projection(
    model: SegformerForSemanticSegmentation,
) -> nn.Conv2d:
    try:
        proj = (
            model
            .segformer
            .encoder
            .patch_embeddings[0]
            .proj
        )
    except Exception as exc:
        raise ModelBProtocolError(
            "Could not locate SegFormer first patch-embedding Conv2d. "
            "The installed transformers SegFormer structure may have changed."
        ) from exc

    require(
        isinstance(proj, nn.Conv2d),
        f"First patch projection is not Conv2d: {type(proj)!r}",
    )
    return proj


def _expand_first_patch_embed_to_4ch(
    model: SegformerForSemanticSegmentation,
) -> Dict[str, Any]:
    """
    Deterministically convert first Conv2d input 3 -> 4 channels.

    Important: constructing nn.Conv2d normally consumes PyTorch RNG. Because
    all of its parameters are immediately overwritten, we use fork_rng() to
    restore the global RNG state afterwards. This avoids unnecessarily shifting
    later dropout/randomness relative to Model A.
    """
    old_proj = _first_patch_projection(model)

    require(
        old_proj.in_channels == RGB_CHANNELS,
        f"Expected pretrained first Conv2d in_channels=3, "
        f"got {old_proj.in_channels}.",
    )
    require(
        old_proj.groups == 1,
        f"Expected first Conv2d groups=1, got {old_proj.groups}.",
    )

    old_weight = old_proj.weight.detach().clone()
    old_bias = (
        old_proj.bias.detach().clone()
        if old_proj.bias is not None
        else None
    )

    rgb_weight_hash_before = tensor_sha256(old_weight)

    # Preserve global CPU RNG state across module construction.
    with torch.random.fork_rng(devices=[], enabled=True):
        new_proj = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            dilation=old_proj.dilation,
            groups=old_proj.groups,
            bias=(old_proj.bias is not None),
            padding_mode=old_proj.padding_mode,
        )

    # Keep dtype/device exactly aligned with the original layer.
    new_proj = new_proj.to(
        device=old_proj.weight.device,
        dtype=old_proj.weight.dtype,
    )

    with torch.no_grad():
        new_proj.weight[:, :RGB_CHANNELS].copy_(old_weight)

        nir_kernel = old_weight.mean(
            dim=1,
            keepdim=True,
        )
        new_proj.weight[:, RGB_CHANNELS:INPUT_CHANNELS].copy_(
            nir_kernel
        )

        if old_bias is not None:
            require(
                new_proj.bias is not None,
                "New 4-channel projection unexpectedly has no bias.",
            )
            new_proj.bias.copy_(old_bias)

    model.segformer.encoder.patch_embeddings[0].proj = new_proj

    # Hugging Face forward does not directly enforce config.num_channels, but
    # keeping config correct is important for serialization and auditing.
    model.config.num_channels = INPUT_CHANNELS
    model.segformer.config.num_channels = INPUT_CHANNELS
    model.segformer.encoder.config.num_channels = INPUT_CHANNELS

    adapted = _first_patch_projection(model)

    require(
        adapted.in_channels == INPUT_CHANNELS,
        "Failed to adapt first Conv2d to 4 input channels.",
    )
    require(
        torch.equal(
            adapted.weight[:, :RGB_CHANNELS],
            old_weight,
        ),
        "RGB pretrained kernels changed during 4-channel adaptation.",
    )

    expected_nir = old_weight.mean(
        dim=1,
        keepdim=True,
    )
    require(
        torch.equal(
            adapted.weight[:, RGB_CHANNELS:INPUT_CHANNELS],
            expected_nir,
        ),
        "NIR kernel is not the exact mean of pretrained RGB kernels.",
    )

    if old_bias is not None:
        require(
            adapted.bias is not None
            and torch.equal(adapted.bias, old_bias),
            "First patch-embedding bias changed during adaptation.",
        )

    return {
        "method": NIR_INIT_METHOD,
        "old_in_channels": RGB_CHANNELS,
        "new_in_channels": INPUT_CHANNELS,
        "out_channels": int(adapted.out_channels),
        "kernel_size": list(adapted.kernel_size),
        "stride": list(adapted.stride),
        "padding": list(adapted.padding),
        "rgb_kernel_preserved_exactly": True,
        "bias_preserved_exactly": True,
        "nir_kernel_equals_rgb_mean_exactly": True,
        "pretrained_rgb_kernel_sha256": rgb_weight_hash_before,
        "adapted_rgb_kernel_sha256": tensor_sha256(
            adapted.weight[:, :RGB_CHANNELS]
        ),
        "nir_kernel_sha256": tensor_sha256(
            adapted.weight[:, RGB_CHANNELS:INPUT_CHANNELS]
        ),
    }


class SegFormerRGBNIRDirect(nn.Module):
    """
    Model B wrapper.

    Inputs
    ------
    rgb:
        [B,3,H,W], float tensor, already normalized by the frozen Potsdam
        Dataset Protocol.
    nir:
        [B,1,H,W], float tensor, already normalized by the frozen Potsdam
        Dataset Protocol.

    Output
    ------
    Full-resolution 6-class logits [B,6,H,W].
    """

    def __init__(
        self,
        model: SegformerForSemanticSegmentation,
    ):
        super().__init__()
        self.model = model

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        *,
        return_raw_logits: bool = False,
    ):
        require(
            isinstance(rgb, torch.Tensor),
            "Model B RGB input must be torch.Tensor.",
        )
        require(
            isinstance(nir, torch.Tensor),
            "Model B NIR input must be torch.Tensor.",
        )
        require(
            rgb.ndim == 4,
            f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}.",
        )
        require(
            nir.ndim == 4,
            f"NIR must be [B,1,H,W], got {tuple(nir.shape)}.",
        )
        require(
            rgb.shape[1] == RGB_CHANNELS,
            f"RGB must have 3 channels, got {rgb.shape[1]}.",
        )
        require(
            nir.shape[1] == NIR_CHANNELS,
            f"NIR must have 1 channel, got {nir.shape[1]}.",
        )
        require(
            rgb.shape[0] == nir.shape[0]
            and rgb.shape[-2:] == nir.shape[-2:],
            "RGB/NIR batch or spatial shapes differ: "
            f"rgb={tuple(rgb.shape)}, nir={tuple(nir.shape)}.",
        )
        require(
            rgb.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ),
            f"RGB must be floating point, got {rgb.dtype}.",
        )
        require(
            nir.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ),
            f"NIR must be floating point, got {nir.dtype}.",
        )
        require(
            rgb.dtype == nir.dtype,
            f"RGB/NIR dtypes differ: {rgb.dtype} vs {nir.dtype}.",
        )
        require(
            torch.isfinite(rgb).all().item(),
            "RGB contains NaN/Inf.",
        )
        require(
            torch.isfinite(nir).all().item(),
            "NIR contains NaN/Inf.",
        )

        rgbnir = torch.cat(
            (rgb, nir),
            dim=1,
        )

        require(
            rgbnir.shape[1] == INPUT_CHANNELS,
            f"Concatenated RGBNIR must have 4 channels, "
            f"got {rgbnir.shape[1]}.",
        )

        outputs = self.model(
            pixel_values=rgbnir,
            return_dict=True,
        )
        raw_logits = outputs.logits

        require(
            raw_logits.ndim == 4
            and raw_logits.shape[0] == rgb.shape[0]
            and raw_logits.shape[1] == NUM_CLASSES,
            f"Model B raw logits shape invalid: "
            f"{tuple(raw_logits.shape)}.",
        )
        require(
            torch.isfinite(raw_logits).all().item(),
            "Model B raw logits contain NaN/Inf.",
        )

        full_logits = F.interpolate(
            raw_logits,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        require(
            full_logits.shape == (
                rgb.shape[0],
                NUM_CLASSES,
                rgb.shape[-2],
                rgb.shape[-1],
            ),
            f"Model B full logits shape invalid: "
            f"{tuple(full_logits.shape)}.",
        )
        require(
            torch.isfinite(full_logits).all().item(),
            "Model B full logits contain NaN/Inf.",
        )

        if return_raw_logits:
            return {
                "logits": full_logits,
                "raw_logits": raw_logits,
            }

        return full_logits


def build_model_b_rgbnir(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Tuple[SegFormerRGBNIRDirect, Dict[str, Any]]:
    project_root = Path(project_root).resolve()

    require(
        checkpoint == DEFAULT_CHECKPOINT,
        "Model B v1 only allows the frozen Model A checkpoint name: "
        f"{DEFAULT_CHECKPOINT}",
    )

    frozen_revision = _load_model_a_frozen_revision(
        project_root
    )

    id2label, label2id = _class_maps()

    config_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}

    if frozen_revision:
        config_kwargs["revision"] = frozen_revision
        model_kwargs["revision"] = frozen_revision

    # Load as 3-channel FIRST so all pretrained encoder weights, including the
    # first patch embedding, load without mismatch.
    config = SegformerConfig.from_pretrained(
        checkpoint,
        **config_kwargs,
    )

    original_num_labels = int(config.num_labels)

    require(
        int(getattr(config, "num_channels", 3)) == RGB_CHANNELS,
        "Pretrained SegFormer config is not 3-channel RGB.",
    )
    require(
        original_num_labels == 150,
        f"Expected ADE20K 150 classes, got {original_num_labels}.",
    )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    model, loading_info = (
        SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
            **model_kwargs,
        )
    )

    loading_summary = validate_pretrained_loading_info(
        loading_info
    )

    require(
        int(model.config.num_labels) == NUM_CLASSES,
        "Model B num_labels != 6 before input adaptation.",
    )
    require(
        int(
            getattr(
                model.config,
                "semantic_loss_ignore_index",
                IGNORE_INDEX,
            )
        )
        == IGNORE_INDEX,
        "Model B semantic_loss_ignore_index != 255.",
    )

    input_adaptation = _expand_first_patch_embed_to_4ch(
        model
    )

    require(
        int(model.config.num_channels) == INPUT_CHANNELS,
        "Model B config.num_channels != 4 after adaptation.",
    )
    require(
        model.decode_head.classifier.out_channels
        == NUM_CLASSES,
        "Model B classifier out_channels != 6.",
    )

    wrapper = SegFormerRGBNIRDirect(model)

    resolved_revision = getattr(
        model.config,
        "_commit_hash",
        None,
    )
    if resolved_revision is None:
        resolved_revision = frozen_revision

    metadata: Dict[str, Any] = {
        "variant": "B",
        "name": "RGB+NIR direct 4-channel SegFormer",
        "checkpoint": checkpoint,
        "resolved_revision": resolved_revision,
        "original_num_labels": original_num_labels,
        "num_labels": NUM_CLASSES,
        "input_modalities": ["RGB", "NIR"],
        "rgb_channels": 3,
        "nir_channels": 1,
        "input_channels": INPUT_CHANNELS,
        "fusion": "early/direct channel concatenation",
        "quality_gate": False,
        "dual_encoder": False,
        "semantic_loss_ignore_index": IGNORE_INDEX,
        "loading_info": loading_summary,
        "input_adaptation": input_adaptation,
        "parameters": parameter_counts(wrapper),
        "config": {
            "hidden_sizes": [
                int(x)
                for x in model.config.hidden_sizes
            ],
            "depths": [
                int(x)
                for x in model.config.depths
            ],
            "num_attention_heads": [
                int(x)
                for x in model.config.num_attention_heads
            ],
            "patch_sizes": [
                int(x)
                for x in model.config.patch_sizes
            ],
            "strides": [
                int(x)
                for x in model.config.strides
            ],
            "decoder_hidden_size": int(
                model.config.decoder_hidden_size
            ),
        },
    }

    return wrapper, metadata


def _smoke_test(project_root: Path) -> None:
    """
    Small real-data forward test. No optimizer/backward.
    """
    from data_pipeline.potsdam_dataset import (
        PotsdamTrainDataset,
    )

    print("=" * 78)
    print("Model B RGB+NIR direct-4CH smoke test")
    print("=" * 78)

    dataset = PotsdamTrainDataset(
        project_root,
        epoch=0,
    )
    sample = dataset[0]

    rgb = sample["rgb"].unsqueeze(0)
    nir = sample["nir"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    require(
        tuple(rgb.shape) == (1, 3, 512, 512),
        f"RGB smoke-test shape invalid: {tuple(rgb.shape)}",
    )
    require(
        tuple(nir.shape) == (1, 1, 512, 512),
        f"NIR smoke-test shape invalid: {tuple(nir.shape)}",
    )

    model, meta = build_model_b_rgbnir(project_root)

    print(
        "input adaptation:",
        meta["input_adaptation"]["method"],
    )
    print(
        "parameters:",
        f"{meta['parameters']['total']:,}",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    with torch.inference_mode():
        out = model(
            rgb.to(device),
            nir.to(device),
            return_raw_logits=True,
        )

    logits = out["logits"]
    raw_logits = out["raw_logits"]

    require(
        tuple(logits.shape) == (1, 6, 512, 512),
        f"Full logits smoke-test shape invalid: "
        f"{tuple(logits.shape)}",
    )
    require(
        tuple(raw_logits.shape[-2:]) == (128, 128),
        f"Raw logits smoke-test resolution invalid: "
        f"{tuple(raw_logits.shape)}",
    )

    loss = F.cross_entropy(
        logits,
        labels.to(device),
        ignore_index=IGNORE_INDEX,
    )
    require(
        torch.isfinite(loss).item(),
        "Smoke-test CE is non-finite.",
    )

    print(f"device      : {device}")
    print(f"raw logits  : {tuple(raw_logits.shape)}")
    print(f"full logits : {tuple(logits.shape)}")
    print(f"CE          : {float(loss.item()):.6f}")
    print("FINAL STATUS: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model B RGB+NIR direct-4CH smoke test.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _smoke_test(project_root)


if __name__ == "__main__":
    main()
