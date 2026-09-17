#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A: RGB -> SegFormer

本模块是 ISPRS Potsdam RGB baseline 的正式模型封装。
当前阶段只做：
- 从 Hugging Face SegFormer B0 ADE20K checkpoint 加载预训练权重；
- 把最终 semantic classifier 从 150 类替换为 Potsdam 6 类；
- 只接受已经由 Potsdam Dataset Protocol v1 完成 normalization 的 RGB tensor；
- 不调用 SegformerImageProcessor / AutoImageProcessor；
- 将 SegFormer raw logits 从 1/4 resolution 双线性上采样回输入分辨率；
- 运行真实 Potsdam RGB patch 的 GPU forward smoke-test；
- PASS 后冻结 model_a_rgb_protocol.json。

当前不训练，不实现 RGB+NIR / dual encoder / fusion / Quality Gate / corruption。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import SegformerConfig, SegformerForSemanticSegmentation


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "Model A RGB Protocol v1"

DEFAULT_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 6
IGNORE_INDEX = 255
EXPECTED_INPUT_CHANNELS = 3
EXPECTED_INPUT_SIZE = 512
EXPECTED_RAW_LOGIT_SIZE = 128

CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]


class ModelAProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelAProtocolError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"缺少文件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ModelAProtocolError(f"JSON 读取失败：{path}\n{e}") from e


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def parameter_counts(module: nn.Module) -> Dict[str, int]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(
        int(p.numel()) for p in module.parameters() if p.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


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


def validate_pretrained_loading_info(info: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    修改 num_labels=6 后，允许 semantic classifier 的 weight/bias shape mismatch。
    不允许 encoder / decoder fusion 等其他权重丢失或 shape mismatch。
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

    bad_mismatch = [k for k in mismatched if not _classifier_key_allowed(k)]
    bad_missing = [k for k in missing if not _classifier_key_allowed(k)]

    require(
        not bad_mismatch,
        "除最终 classifier 外出现 pretrained shape mismatch："
        f"{bad_mismatch}"
    )
    require(
        not bad_missing,
        "除最终 classifier 外出现 pretrained missing keys："
        f"{bad_missing}"
    )
    require(
        not unexpected,
        f"pretrained checkpoint 出现 unexpected keys：{unexpected}"
    )

    allowed_union = set(mismatched) | set(missing)
    require(
        any(k.endswith("decode_head.classifier.weight") for k in allowed_union),
        "没有检测到 150->6 classifier.weight 替换"
    )
    require(
        any(k.endswith("decode_head.classifier.bias") for k in allowed_union),
        "没有检测到 150->6 classifier.bias 替换"
    )

    return {
        "mismatched_keys": mismatched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _class_maps() -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def _load_frozen_revision_if_present(
    project_root: Path,
) -> Optional[str]:
    protocol_path = (
        project_root
        / "data/processed/potsdam/model_a_rgb_protocol.json"
    )
    if not protocol_path.is_file():
        return None

    protocol = load_json(protocol_path)
    require(
        protocol.get("protocol_version") == PROTOCOL_VERSION,
        "已有 model_a_rgb_protocol.json 版本不匹配"
    )
    require(
        protocol.get("status") == "FROZEN_AFTER_AUDIT_PASS",
        "已有 model_a_rgb_protocol.json 不是冻结 PASS 状态"
    )

    model = protocol["model"]
    require(
        model["checkpoint"] == DEFAULT_CHECKPOINT,
        "已有 Model A checkpoint 与当前代码默认 checkpoint 不一致"
    )
    revision = model.get("resolved_revision")
    return revision if revision else None


def build_model_a_rgb(
    project_root: Path | str,
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Tuple["SegFormerRGBBaseline", Dict[str, Any]]:
    project_root = Path(project_root).resolve()

    require(
        checkpoint == DEFAULT_CHECKPOINT,
        "Model A v1 当前只允许冻结 checkpoint："
        f"{DEFAULT_CHECKPOINT}"
    )

    frozen_revision = _load_frozen_revision_if_present(project_root)

    id2label, label2id = _class_maps()

    config_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    if frozen_revision:
        config_kwargs["revision"] = frozen_revision
        model_kwargs["revision"] = frozen_revision

    config = SegformerConfig.from_pretrained(
        checkpoint,
        **config_kwargs,
    )
    original_num_labels = int(config.num_labels)

    require(
        int(getattr(config, "num_channels", 3)) == 3,
        "pretrained SegFormer config num_channels != 3"
    )
    require(
        original_num_labels == 150,
        f"预期 ADE20K checkpoint 为 150 类，实际 {original_num_labels}"
    )

    config.num_labels = NUM_CLASSES
    config.id2label = id2label
    config.label2id = label2id
    config.semantic_loss_ignore_index = IGNORE_INDEX

    model, loading_info = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **model_kwargs,
    )

    loading_summary = validate_pretrained_loading_info(loading_info)

    require(model.config.num_labels == NUM_CLASSES,
            "Model A num_labels != 6")
    require(
        int(model.config.semantic_loss_ignore_index) == IGNORE_INDEX,
        "Model A semantic_loss_ignore_index != 255"
    )
    require(
        int(getattr(model.config, "num_channels", 3)) == 3,
        "Model A num_channels != 3"
    )
    require(
        model.decode_head.classifier.out_channels == NUM_CLASSES,
        "Model A classifier out_channels != 6"
    )

    wrapper = SegFormerRGBBaseline(model)

    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = frozen_revision

    metadata = {
        "checkpoint": checkpoint,
        "resolved_revision": resolved_revision,
        "original_num_labels": original_num_labels,
        "num_labels": NUM_CLASSES,
        "input_channels": 3,
        "semantic_loss_ignore_index": IGNORE_INDEX,
        "loading_info": loading_summary,
        "parameters": parameter_counts(wrapper),
        "config": {
            "hidden_sizes": [
                int(x) for x in model.config.hidden_sizes
            ],
            "depths": [
                int(x) for x in model.config.depths
            ],
            "num_attention_heads": [
                int(x) for x in model.config.num_attention_heads
            ],
            "patch_sizes": [
                int(x) for x in model.config.patch_sizes
            ],
            "strides": [
                int(x) for x in model.config.strides
            ],
            "decoder_hidden_size": int(
                model.config.decoder_hidden_size
            ),
        },
    }
    return wrapper, metadata


class SegFormerRGBBaseline(nn.Module):
    """
    Model A 正式封装。

    输入：
        rgb: float32 [B,3,H,W]
        已由 Potsdam Dataset 完成 /255 + ImageNet normalization。

    输出：
        full-resolution logits: float32 [B,6,H,W]

    注意：
    - 不接受 NIR；
    - 不调用 Hugging Face image processor；
    - 不在这里计算 loss，保证未来 A/B/C/C-noGate 使用统一外部 loss。
    """

    def __init__(self, model: SegformerForSemanticSegmentation):
        super().__init__()
        self.model = model

    def forward(
        self,
        rgb: torch.Tensor,
        *,
        return_raw_logits: bool = False,
    ):
        require(
            isinstance(rgb, torch.Tensor),
            "Model A 输入必须是 torch.Tensor"
        )
        require(
            rgb.ndim == 4,
            f"Model A RGB 输入必须为 [B,3,H,W]，实际 {tuple(rgb.shape)}"
        )
        require(
            rgb.shape[1] == EXPECTED_INPUT_CHANNELS,
            f"Model A 只接受 3-channel RGB，实际 channels={rgb.shape[1]}"
        )
        require(
            rgb.dtype in (torch.float16, torch.bfloat16, torch.float32),
            f"Model A RGB dtype 必须是浮点，实际 {rgb.dtype}"
        )
        require(
            torch.isfinite(rgb).all().item(),
            "Model A RGB 输入出现 NaN/Inf"
        )

        outputs = self.model(
            pixel_values=rgb,
            return_dict=True,
        )
        raw_logits = outputs.logits

        require(
            raw_logits.ndim == 4
            and raw_logits.shape[0] == rgb.shape[0]
            and raw_logits.shape[1] == NUM_CLASSES,
            f"Model A raw logits shape 异常：{tuple(raw_logits.shape)}"
        )
        require(
            torch.isfinite(raw_logits).all().item(),
            "Model A raw logits 出现 NaN/Inf"
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
            f"Model A full logits shape 异常：{tuple(full_logits.shape)}"
        )
        require(
            torch.isfinite(full_logits).all().item(),
            "Model A full logits 出现 NaN/Inf"
        )

        if return_raw_logits:
            return {
                "logits": full_logits,
                "raw_logits": raw_logits,
            }
        return full_logits


def external_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    只用于 smoke-test 验证 6-class tensor/loss 接口。
    当前阶段不执行 optimizer/backward。
    """
    require(
        logits.ndim == 4 and logits.shape[1] == NUM_CLASSES,
        "loss logits shape 异常"
    )
    require(
        labels.ndim == 3,
        "loss labels shape 必须为 [B,H,W]"
    )
    require(
        logits.shape[0] == labels.shape[0]
        and logits.shape[-2:] == labels.shape[-2:],
        "loss logits/labels shape 不对齐"
    )
    require(labels.dtype == torch.int64,
            "loss labels dtype 必须为 int64")

    return F.cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


def freeze_protocol(
    path: Path,
    protocol: Dict[str, Any],
) -> Dict[str, str]:
    new_fp = sha256_json_canonical(protocol)

    if path.exists():
        existing = load_json(path)
        old_fp = sha256_json_canonical(existing)
        require(
            old_fp == new_fp,
            "model_a_rgb_protocol.json 已存在且与本次结果不同；"
            "拒绝静默覆盖，请先人工审查。"
        )
        return {
            "action": "verified_existing",
            "sha256": sha256_file(path),
        }

    write_json(path, protocol)
    return {
        "action": "created",
        "sha256": sha256_file(path),
    }


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_smoke_test(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from data_pipeline.potsdam_dataset import PotsdamTrainDataset

    processed = project_root / "data/processed/potsdam"
    dataset_protocol_path = processed / "dataset_protocol.json"
    dataloader_protocol_path = processed / "dataloader_protocol.json"
    evaluation_protocol_path = processed / "evaluation_protocol.json"

    for p in (
        dataset_protocol_path,
        dataloader_protocol_path,
        evaluation_protocol_path,
    ):
        require(p.is_file(), f"缺少冻结 protocol：{p}")

    evaluation_protocol = load_json(evaluation_protocol_path)
    require(
        evaluation_protocol.get("status")
        == "FROZEN_AFTER_AUDIT_PASS",
        "evaluation_protocol.json 不是冻结 PASS 状态"
    )

    print("=" * 72)
    print("Model A: RGB -> SegFormer GPU forward smoke-test")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("本测试不训练，不执行 backward/optimizer。")
    print()

    print("[1/5] 读取真实 Train RGB sample...")
    dataset = PotsdamTrainDataset(project_root, epoch=0)
    sample = dataset[0]

    rgb = sample["rgb"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    require(tuple(rgb.shape) == (1, 3, 512, 512),
            f"RGB shape 异常：{tuple(rgb.shape)}")
    require(tuple(labels.shape) == (1, 512, 512),
            f"label shape 异常：{tuple(labels.shape)}")
    require(rgb.dtype == torch.float32,
            "RGB dtype != float32")
    require(labels.dtype == torch.int64,
            "label dtype != int64")

    print(
        f"  tile={sample['tile_id']} "
        f"slot={sample['sample_slot']} "
        f"x={sample['x']} y={sample['y']} d4={sample['d4_code']}"
    )
    print("  PASS: Dataset RGB/label tensor contract.")

    print("[2/5] 加载 Model A pretrained SegFormer...")
    wrapper, model_meta = build_model_a_rgb(project_root)
    print(f"  checkpoint = {model_meta['checkpoint']}")
    print(f"  resolved revision = {model_meta['resolved_revision']}")
    print(
        f"  parameters total/trainable = "
        f"{model_meta['parameters']['total']:,}/"
        f"{model_meta['parameters']['trainable']:,}"
    )
    print(
        "  PASS: pretrained encoder/decoder loaded；"
        "仅最终 150->6 classifier 允许 mismatch。"
    )

    print("[3/5] RTX GPU forward：raw logits + full-resolution logits...")
    require(torch.cuda.is_available(),
            "CUDA 不可用；本 smoke-test 要求 GPU")
    device = torch.device("cuda:0")

    wrapper = wrapper.to(device)
    wrapper.eval()

    rgb_gpu = rgb.to(device, non_blocking=False)
    labels_gpu = labels.to(device, non_blocking=False)

    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        out1 = wrapper(
            rgb_gpu,
            return_raw_logits=True,
        )
        out2 = wrapper(
            rgb_gpu,
            return_raw_logits=True,
        )

    raw = out1["raw_logits"]
    logits = out1["logits"]

    require(
        tuple(raw.shape) == (1, 6, 128, 128),
        f"预期 raw logits=(1,6,128,128)，实际={tuple(raw.shape)}"
    )
    require(
        tuple(logits.shape) == (1, 6, 512, 512),
        f"full logits shape 异常：{tuple(logits.shape)}"
    )

    require(
        torch.equal(out1["raw_logits"], out2["raw_logits"]),
        "eval inference 同输入两次 raw logits 未精确重现"
    )
    require(
        torch.equal(out1["logits"], out2["logits"]),
        "eval inference 同输入两次 full logits 未精确重现"
    )

    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    print(f"  raw logits shape = {tuple(raw.shape)}")
    print(f"  full logits shape = {tuple(logits.shape)}")
    print(
        f"  peak allocated CUDA memory = "
        f"{peak_bytes / (1024**2):.2f} MiB"
    )
    print("  PASS: forward finite + deterministic eval repeat.")

    print("[4/5] 外部 6-class CE loss 接口检查（不 backward）...")
    loss = external_cross_entropy_loss(logits, labels_gpu)
    require(torch.isfinite(loss).item(),
            f"CE loss 非有限：{float(loss.detach().cpu())}")
    loss_value = float(loss.detach().cpu())

    prediction = torch.argmax(logits, dim=1)
    unique_pred = sorted(
        int(x) for x in torch.unique(prediction).detach().cpu().tolist()
    )
    require(
        all(0 <= x <= 5 for x in unique_pred),
        f"prediction 出现非法 class：{unique_pred}"
    )

    print(f"  CE loss = {loss_value:.8f}")
    print(f"  predicted classes = {unique_pred}")
    print("  PASS: logits/label/loss contract；没有执行 backward。")

    print("[5/5] 冻结 model_a_rgb_protocol.json...")
    model_file = project_root / "models/segformer_rgb.py"
    require(model_file.is_file(),
            f"缺少 Model A 文件：{model_file}")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "status": "FROZEN_AFTER_AUDIT_PASS",
        "sources": {
            "dataset_protocol": (
                "data/processed/potsdam/dataset_protocol.json"
            ),
            "dataset_protocol_sha256": sha256_file(
                dataset_protocol_path
            ),
            "dataloader_protocol": (
                "data/processed/potsdam/dataloader_protocol.json"
            ),
            "dataloader_protocol_sha256": sha256_file(
                dataloader_protocol_path
            ),
            "evaluation_protocol": (
                "data/processed/potsdam/evaluation_protocol.json"
            ),
            "evaluation_protocol_sha256": sha256_file(
                evaluation_protocol_path
            ),
            "model_module": "models/segformer_rgb.py",
            "model_module_sha256": sha256_file(model_file),
        },
        "model": {
            **model_meta,
            "variant": "A",
            "name": "RGB -> SegFormer",
            "rgb_channels": [0, 1, 2],
            "nir_used": False,
            "num_classes": 6,
            "input_normalization": (
                "already applied by Potsdam Dataset Protocol v1"
            ),
            "huggingface_image_processor_used": False,
            "raw_logit_resolution_for_512_input": [128, 128],
            "full_logit_resolution": [512, 512],
            "upsample": {
                "mode": "bilinear",
                "align_corners": False,
            },
        },
        "smoke_test": {
            "device": str(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sample": {
                "tile_id": sample["tile_id"],
                "epoch": int(sample["epoch"]),
                "sample_slot": int(sample["sample_slot"]),
                "x": int(sample["x"]),
                "y": int(sample["y"]),
                "d4_code": int(sample["d4_code"]),
                "rgb_sha256": tensor_sha256(rgb),
                "label_sha256": tensor_sha256(labels),
            },
            "raw_logits_shape": list(raw.shape),
            "full_logits_shape": list(logits.shape),
            "raw_logits_sha256": tensor_sha256(raw),
            "full_logits_sha256": tensor_sha256(logits),
            "eval_repeat_exact": True,
            "ce_loss_finite": True,
            "ce_loss_value_untrained_head_diagnostic_only": loss_value,
            "prediction_classes": unique_pred,
            "peak_cuda_memory_allocated_bytes": peak_bytes,
            "training_performed": False,
            "backward_performed": False,
        },
    }

    protocol_path = processed / "model_a_rgb_protocol.json"
    freeze_info = freeze_protocol(protocol_path, protocol)
    print(f"  model_a_rgb_protocol.json: {freeze_info['action']}")
    print()
    print("FINAL STATUS: PASS")

    return {
        "status": "PASS",
        "module_version": MODULE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": model_meta["checkpoint"],
        "resolved_revision": model_meta["resolved_revision"],
        "parameters": model_meta["parameters"],
        "loading_info": model_meta["loading_info"],
        "sample": protocol["smoke_test"]["sample"],
        "raw_logits_shape": list(raw.shape),
        "full_logits_shape": list(logits.shape),
        "eval_repeat_exact": True,
        "ce_loss": loss_value,
        "prediction_classes": unique_pred,
        "peak_cuda_memory_allocated_bytes": peak_bytes,
        "frozen_model_protocol": {
            "path": "data/processed/potsdam/model_a_rgb_protocol.json",
            **freeze_info,
        },
    }


def report_to_text(report: Dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return (
            "Model A RGB smoke-test\n"
            + "=" * 72
            + "\n状态: FAIL\n"
            + f"error_type: {report.get('error_type')}\n"
            + f"error: {report.get('error')}\n"
        )

    lines = [
        "Model A: RGB -> SegFormer GPU forward smoke-test",
        "=" * 72,
        "状态: PASS",
        f"模块版本: {report['module_version']}",
        f"protocol: {report['protocol_version']}",
        "",
        "[Checkpoint]",
        f"name = {report['checkpoint']}",
        f"resolved_revision = {report['resolved_revision']}",
        "",
        "[Parameters]",
        f"total = {report['parameters']['total']}",
        f"trainable = {report['parameters']['trainable']}",
        "",
        "[Forward]",
        f"raw_logits_shape = {report['raw_logits_shape']}",
        f"full_logits_shape = {report['full_logits_shape']}",
        f"eval_repeat_exact = {report['eval_repeat_exact']}",
        f"CE loss (diagnostic only) = {report['ce_loss']}",
        f"prediction_classes = {report['prediction_classes']}",
        f"peak_cuda_memory_bytes = "
        f"{report['peak_cuda_memory_allocated_bytes']}",
        "",
        "[Frozen metadata]",
        f"path = {report['frozen_model_protocol']['path']}",
        f"action = {report['frozen_model_protocol']['action']}",
        f"sha256 = {report['frozen_model_protocol']['sha256']}",
        "",
        "FINAL STATUS: PASS",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model A RGB SegFormer GPU forward smoke-test"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录；默认 models/ 的父目录。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行真实 RGB patch GPU forward；不训练。",
    )
    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "Model A 模块已加载。运行 smoke-test：\n"
            "  python models/segformer_rgb.py --smoke-test"
        )
        return 0

    project_root = args.project_root.resolve()
    out_dir = project_root / "outputs/model_check/model_a_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    console_path = out_dir / "console.txt"
    json_path = out_dir / "model_a_rgb_smoke_test.json"
    txt_path = out_dir / "model_a_rgb_smoke_test.txt"

    with console_path.open("w", encoding="utf-8", buffering=1) as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)

        try:
            report = run_smoke_test(project_root)
            exit_code = 0
        except Exception as e:
            print()
            print()
            print("=" * 72)
            print("FINAL STATUS: FAIL")
            print(f"{type(e).__name__}: {e}")
            print("=" * 72)
            traceback.print_exc()
            report = {
                "status": "FAIL",
                "module_version": MODULE_VERSION,
                "project_root": str(project_root),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            exit_code = 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    write_json(json_path, report)
    txt_path.write_text(report_to_text(report), encoding="utf-8")

    print(f"console: {console_path}")
    print(f"json: {json_path}")
    print(f"txt: {txt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
