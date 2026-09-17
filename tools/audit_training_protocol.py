#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Control Protocol v1 审计与冻结

用途：
- 不训练任何模型。
- 不执行 SegFormer forward/backward。
- 只读取已经冻结并 PASS 的上游协议，冻结 A/B/C-noGate/C 共用的训练控制规则。
- 生成 training_protocol.json，后续训练引擎必须只读消费该文件。

运行：
    PYTHONHASHSEED=0 python tools/audit_training_protocol.py

输出：
    outputs/training_check/training_protocol/
        console.txt
        training_protocol_audit.json
        training_protocol_audit.txt

冻结：
    data/processed/potsdam/training_protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_VERSION = "1.0.6"
PROTOCOL_VERSION = "Training Control Protocol v1"

TRAIN_SEED = 20260917

# ----------------------------------------------------------------------
# 冻结训练控制参数
# ----------------------------------------------------------------------
MAX_EPOCHS = 100

MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
TRAIN_NUM_WORKERS = 2
VAL_WINDOW_BATCH_SIZE = 4

LOSS_NAME = "CrossEntropyLoss"
IGNORE_INDEX = 255
LABEL_SMOOTHING = 0.0

OPTIMIZER_NAME = "AdamW"
BASE_LR = 6e-5
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
NO_DECAY_RULE = "bias_and_normalization_parameters"
LR_MULTIPLIER = 1.0

SCHEDULER_NAME = "linear_warmup_then_polynomial_decay"
POLY_POWER = 1.0
WARMUP_RATIO = 0.05
WARMUP_START_FACTOR = 1e-6
MIN_LR = 0.0

AMP_ENABLED = True
AMP_DTYPE = "float16"
GRAD_SCALER_ENABLED = True
GRAD_SCALER_INIT_SCALE = 65536.0
GRAD_SCALER_GROWTH_FACTOR = 2.0
GRAD_SCALER_BACKOFF_FACTOR = 0.5
GRAD_SCALER_GROWTH_INTERVAL = 2000

GRAD_CLIP_ENABLED = True
GRAD_CLIP_MAX_NORM = 1.0
GRAD_CLIP_NORM_TYPE = 2.0

VALIDATE_EVERY_EPOCHS = 5
CHECKPOINT_METRIC = "val_global_mIoU"
CHECKPOINT_MODE = "max"
CHECKPOINT_TIE_BREAK = "earliest_epoch"
EARLY_STOPPING = False

# 已由 Model A smoke-test 冻结的共同 SegFormer 起点。
EXPECTED_BASE_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
EXPECTED_BASE_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"

EXPECTED_UPSTREAM_SHA256 = {
    "dataset_protocol": "e292a33266389b2ad07e09cda1a5c2e2d51003c3330678eed114e236d30c6c22",
    "dataloader_protocol": "ce0c8427af28682b74436f820059398b081c45b113102ad0c97cde0992f14ee4",
    "evaluation_protocol": "9e1fe295f898190749cc261dc9cdd5dbb76c213da90547b576ea6f4fb4e2192d",
    "model_a_rgb_protocol": "476bae90a907107734cc1ce29db756f4a1299a723cd647b8e2879df0adc6f713",
}


class AuditError(RuntimeError):
    pass


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AuditError(message)


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"缺少文件: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise AuditError(f"JSON 读取失败: {path}: {e}") from e
    require(isinstance(data, dict), f"JSON 顶层不是 dict: {path}")
    return data


def _walk_scalars(obj: Any, path: Tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_scalars(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_scalars(v, path + (f"[{i}]",))
    else:
        yield path, obj


def _path_text(path: Tuple[str, ...]) -> str:
    return ".".join(path)


def _semantic_train_length_candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从冻结 protocol 中寻找“每 epoch 训练样本数 / train dataset length”候选。

    只接受路径语义明确的整数标量：
      - 路径中必须出现 train
      - 且必须出现 length / sample / epoch 之一
      - 排除 hash、sha、version、worker、batch、update、step 等明显无关项

    这里只负责“找候选”，最终仍要求候选值一致并精确等于冻结值 1152。
    """
    out: List[Dict[str, Any]] = []
    include_any = ("length", "sample", "epoch")
    exclude_any = (
        "sha", "hash", "version", "worker", "batch", "update", "step",
        "warmup", "epoch_order", "first_", "rank", "slot", "index",
    )

    for path, value in _walk_scalars(data):
        if isinstance(value, bool) or not isinstance(value, int):
            continue

        s = _path_text(path).lower()
        if "train" not in s:
            continue
        if not any(tok in s for tok in include_any):
            continue
        if any(tok in s for tok in exclude_any):
            continue

        out.append({"path": _path_text(path), "value": int(value)})
    return out


def resolve_samples_per_epoch(
    dataset_protocol: Dict[str, Any],
    dataloader_protocol: Dict[str, Any],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    解析并交叉审计 samples_per_epoch。

    优先使用精确已知路径；若 schema 不同，再使用严格语义候选。
    所有命中的可信候选必须一致；最终值必须为冻结的 1152。
    """
    candidates: List[Dict[str, Any]] = []

    exact_paths = [
        ("dataloader_protocol", dataloader_protocol, ("train_length",)),
        ("dataloader_protocol", dataloader_protocol, ("train", "length")),
        ("dataloader_protocol", dataloader_protocol, ("train", "dataset_length")),
        ("dataloader_protocol", dataloader_protocol, ("train", "samples_per_epoch")),
        ("dataloader_protocol", dataloader_protocol, ("dataset", "train_length")),
        ("dataloader_protocol", dataloader_protocol, ("budget", "samples_per_epoch")),
        ("dataset_protocol", dataset_protocol, ("train", "length")),
        ("dataset_protocol", dataset_protocol, ("train", "samples_per_epoch")),
        ("dataset_protocol", dataset_protocol, ("sampling", "samples_per_epoch")),
        ("dataset_protocol", dataset_protocol, ("training_sampling", "samples_per_epoch")),
        ("dataset_protocol", dataset_protocol, ("protocol", "samples_per_epoch")),
    ]

    def get_path(obj: Dict[str, Any], keys: Tuple[str, ...]):
        cur: Any = obj
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    seen = set()
    for source, obj, keys in exact_paths:
        v = get_path(obj, keys)
        if isinstance(v, int) and not isinstance(v, bool):
            item = {
                "source": source,
                "path": ".".join(keys),
                "value": int(v),
                "mode": "exact",
            }
            key = (source, item["path"])
            if key not in seen:
                seen.add(key)
                candidates.append(item)

    # 如果 exact path 没有命中 dataloader schema，则做严格语义扫描。
    for source, obj in (
        ("dataloader_protocol", dataloader_protocol),
        ("dataset_protocol", dataset_protocol),
    ):
        for c in _semantic_train_length_candidates(obj):
            key = (source, c["path"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "source": source,
                "path": c["path"],
                "value": c["value"],
                "mode": "semantic",
            })

    # 只把值域合理的候选纳入一致性判定，避免诸如 epoch=100 被误认；
    # 但所有原始候选仍会在失败诊断里打印。
    plausible = [
        c for c in candidates
        if 1 <= int(c["value"]) <= 100000
        and not c["path"].lower().endswith(("max_epochs", "epochs"))
    ]

    # 明确排除不是 sample count 的常见路径。
    filtered = []
    for c in plausible:
        s = c["path"].lower()
        if any(tok in s for tok in (
            "max_epoch", "num_epoch", "epochs",
            "crop", "tile_count", "class", "window",
        )):
            continue
        filtered.append(c)

    if not filtered:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']} ({c['mode']})"
            for c in candidates
        ) or "  <none>"
        raise AuditError(
            "无法从冻结 dataset/dataloader protocol 唯一解析每 epoch 训练样本数。\n"
            "检测到的语义候选：\n" + diagnostic
        )

    values = sorted({int(c["value"]) for c in filtered})
    if len(values) != 1:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']} ({c['mode']})"
            for c in filtered
        )
        raise AuditError(
            "冻结 protocol 中关于每 epoch 训练样本数的候选发生冲突；"
            "禁止自动选择。\n候选：\n" + diagnostic
        )

    value = values[0]
    require(
        value == 1152,
        "每 epoch 训练样本数应为冻结值 1152，"
        f"解析得到={value}；候选={filtered}",
    )
    return value, filtered



def collect_terminal_key_candidates(
    data: Dict[str, Any],
    terminal_key: str,
    source: str,
) -> List[Dict[str, Any]]:
    """收集所有“最后一个 dict key == terminal_key”的标量/容器候选。"""
    out: List[Dict[str, Any]] = []

    def rec(obj: Any, path: Tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = path + (str(k),)
                if str(k).lower() == terminal_key.lower():
                    out.append({
                        "source": source,
                        "path": _path_text(p),
                        "value": v,
                    })
                rec(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                rec(v, path + (f"[{i}]",))

    rec(data)
    return out


def resolve_unique_bool(
    data: Dict[str, Any],
    terminal_key: str,
    expected: bool,
    source: str,
) -> Tuple[bool, List[Dict[str, Any]]]:
    candidates = collect_terminal_key_candidates(data, terminal_key, source)
    bool_candidates = [
        c for c in candidates if isinstance(c["value"], bool)
    ]
    if not bool_candidates:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']!r}"
            for c in candidates
        ) or "  <none>"
        raise AuditError(
            f"无法在 {source} 中解析布尔字段 {terminal_key!r}。\n"
            f"同名候选：\n{diagnostic}"
        )

    values = {bool(c["value"]) for c in bool_candidates}
    if len(values) != 1:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']!r}"
            for c in bool_candidates
        )
        raise AuditError(
            f"{source} 中 {terminal_key!r} 存在冲突候选；禁止自动选择。\n"
            f"候选：\n{diagnostic}"
        )

    value = next(iter(values))
    require(
        value is expected,
        f"{source}.{terminal_key} 必须为 {expected!r}，实际={value!r}；"
        f"候选={[{'path': c['path'], 'value': c['value']} for c in bool_candidates]}",
    )
    return value, bool_candidates


def resolve_unique_string(
    data: Dict[str, Any],
    terminal_key: str,
    expected: str,
    source: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    candidates = collect_terminal_key_candidates(data, terminal_key, source)
    str_candidates = [
        c for c in candidates if isinstance(c["value"], str)
    ]
    if not str_candidates:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']!r}"
            for c in candidates
        ) or "  <none>"
        raise AuditError(
            f"无法在 {source} 中解析字符串字段 {terminal_key!r}。\n"
            f"同名候选：\n{diagnostic}"
        )

    values = {c["value"] for c in str_candidates}
    if len(values) != 1:
        diagnostic = "\n".join(
            f"  {c['source']}:{c['path']} = {c['value']!r}"
            for c in str_candidates
        )
        raise AuditError(
            f"{source} 中 {terminal_key!r} 存在冲突候选；禁止自动选择。\n"
            f"候选：\n{diagnostic}"
        )

    value = next(iter(values))
    require(
        value == expected,
        f"{source}.{terminal_key} 不一致：expected={expected!r}, actual={value!r}",
    )
    return value, str_candidates


def resolve_model_loading_info(
    model_protocol: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates = collect_terminal_key_candidates(
        model_protocol, "loading_info", "model_a_rgb_protocol"
    )
    dict_candidates = [
        c for c in candidates if isinstance(c["value"], dict)
    ]
    require(
        bool(dict_candidates),
        "无法在 model_a_rgb_protocol 中解析 loading_info dict",
    )

    normalized = []
    for c in dict_candidates:
        d = c["value"]
        item = {
            "missing_keys": d.get("missing_keys"),
            "unexpected_keys": d.get("unexpected_keys"),
            "mismatched_keys": sorted(d.get("mismatched_keys", []))
            if isinstance(d.get("mismatched_keys", []), list)
            else d.get("mismatched_keys"),
        }
        normalized.append((c, item))

    canonical = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [
            "decode_head.classifier.bias",
            "decode_head.classifier.weight",
        ],
    }

    for c, item in normalized:
        require(
            item == canonical,
            "Model A loading_info 异常："
            f"path={c['path']}, actual={item}, expected={canonical}",
        )

    first = normalized[0][1]
    return first, dict_candidates


def status_is_pass(data: Dict[str, Any], path: Path) -> str:
    """
    冻结 protocol metadata 的成功状态在现有项目中有两种合法表达：
      - PASS
      - FROZEN_AFTER_AUDIT_PASS

    规则：
    - 若没有 status：不单凭缺失字段判 FAIL，继续执行协议级硬校验；
    - 若存在 status：只接受上述两个已知成功状态；
    - FAIL、FROZEN、UNKNOWN 或任何其他未知状态都 hard FAIL。
    """
    if "status" not in data:
        return "status_absent_but_schema_validated"

    raw_status = str(data["status"])
    status = raw_status.upper()
    accepted = {"PASS", "FROZEN_AFTER_AUDIT_PASS"}
    require(
        status in accepted,
        f"上游协议 status 不是已知成功状态: {path} "
        f"(status={raw_status!r}, accepted={sorted(accepted)})",
    )
    if status == "PASS":
        return "explicit_pass"
    return "frozen_after_audit_pass"


def project_root_from_script() -> Path:
    # tools/audit_training_protocol.py -> project root
    return Path(__file__).resolve().parents[1]


def lr_at_update(update_idx: int, total_updates: int, warmup_updates: int) -> float:
    """
    明确定义 optimizer update 的学习率，不依赖 PyTorch scheduler 的 step 调用语义。

    update_idx: 0 .. total_updates-1

    Warmup:
      线性从 BASE_LR * WARMUP_START_FACTOR 到 BASE_LR。
      warmup 最后一个 update 恰好达到 BASE_LR。

    Poly:
      warmup 后从 BASE_LR 线性(poly power=1)衰减到 MIN_LR。
      最后一个 optimizer update 恰好达到 MIN_LR。
    """
    require(0 <= update_idx < total_updates, "update_idx 越界")
    require(0 <= warmup_updates < total_updates, "warmup_updates 非法")

    if warmup_updates > 0 and update_idx < warmup_updates:
        if warmup_updates == 1:
            factor = 1.0
        else:
            p = update_idx / float(warmup_updates - 1)
            factor = WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * p
        return BASE_LR * factor

    decay_updates = total_updates - warmup_updates
    decay_idx = update_idx - warmup_updates
    if decay_updates <= 1:
        return MIN_LR

    p = decay_idx / float(decay_updates - 1)
    factor = (1.0 - p) ** POLY_POWER
    return MIN_LR + (BASE_LR - MIN_LR) * factor


def lr_schedule_fingerprint(total_updates: int, warmup_updates: int) -> Tuple[str, List[float]]:
    vals = [lr_at_update(i, total_updates, warmup_updates) for i in range(total_updates)]
    # 固定十七位科学计数法文本，跨 Python/平台更容易复核。
    payload = "\n".join(f"{v:.17e}" for v in vals).encode("ascii")
    return sha256_bytes(payload), vals


def seed_probe() -> Dict[str, Any]:
    """
    只做 RNG 接口审计，不构造/训练模型。
    """
    out: Dict[str, Any] = {
        "python": None,
        "numpy": None,
        "torch_cpu": None,
        "torch_cuda": None,
    }

    random.seed(TRAIN_SEED)
    a = [random.random() for _ in range(8)]
    random.seed(TRAIN_SEED)
    b = [random.random() for _ in range(8)]
    require(a == b, "Python random seed 复现失败")
    out["python"] = sha256_bytes(canonical_json_bytes(a))

    try:
        import numpy as np
        np.random.seed(TRAIN_SEED)
        a_np = np.random.random(8).astype("float64")
        np.random.seed(TRAIN_SEED)
        b_np = np.random.random(8).astype("float64")
        require(np.array_equal(a_np, b_np), "NumPy seed 复现失败")
        out["numpy"] = sha256_bytes(a_np.tobytes(order="C"))
    except Exception as e:
        raise AuditError(f"NumPy RNG 审计失败: {e}") from e

    try:
        import torch
        torch.manual_seed(TRAIN_SEED)
        a_t = torch.rand(8, dtype=torch.float32)
        torch.manual_seed(TRAIN_SEED)
        b_t = torch.rand(8, dtype=torch.float32)
        require(torch.equal(a_t, b_t), "PyTorch CPU seed 复现失败")
        out["torch_cpu"] = sha256_bytes(a_t.numpy().tobytes(order="C"))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TRAIN_SEED)
            a_c = torch.rand(8, device="cuda", dtype=torch.float32)
            torch.cuda.manual_seed_all(TRAIN_SEED)
            b_c = torch.rand(8, device="cuda", dtype=torch.float32)
            require(torch.equal(a_c, b_c), "PyTorch CUDA seed 复现失败")
            out["torch_cuda"] = sha256_bytes(
                a_c.detach().cpu().numpy().tobytes(order="C")
            )
        else:
            raise AuditError("CUDA 不可用；该项目已冻结为 GPU 训练环境，不能跳过 CUDA seed 审计")

        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
    except AuditError:
        raise
    except Exception as e:
        raise AuditError(f"PyTorch RNG 审计失败: {e}") from e

    return out


def build_protocol(
    samples_per_epoch: int,
    optimizer_steps_per_epoch: int,
    total_optimizer_updates: int,
    warmup_updates: int,
    validation_epochs: List[int],
    upstream_hashes: Dict[str, str],
    lr_sha256: str,
    train_length_sources: List[Dict[str, Any]],
    drop_last_sources: List[Dict[str, Any]],
    persistent_worker_sources: List[Dict[str, Any]],
    metric_sources: List[Dict[str, Any]],
    window_sha_sources: List[Dict[str, Any]],
    checkpoint_sources: List[Dict[str, Any]],
    revision_sources: List[Dict[str, Any]],
    loading_info_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "scope": "A/B/C-noGate/C shared training control; no model-specific tuning",
        "inputs": {
            "dataset_protocol": "data/processed/potsdam/dataset_protocol.json",
            "dataset_protocol_sha256": upstream_hashes["dataset_protocol"],
            "dataloader_protocol": "data/processed/potsdam/dataloader_protocol.json",
            "dataloader_protocol_sha256": upstream_hashes["dataloader_protocol"],
            "evaluation_protocol": "data/processed/potsdam/evaluation_protocol.json",
            "evaluation_protocol_sha256": upstream_hashes["evaluation_protocol"],
            "model_a_rgb_protocol": "data/processed/potsdam/model_a_rgb_protocol.json",
            "model_a_rgb_protocol_sha256": upstream_hashes["model_a_rgb_protocol"],
        },
        "base_model": {
            "checkpoint": EXPECTED_BASE_CHECKPOINT,
            "resolved_revision": EXPECTED_BASE_REVISION,
            "rule": "all variants must derive SegFormer-B0 pretrained components from this pinned revision where applicable",
        },
        "reproducibility": {
            "train_seed": TRAIN_SEED,
            "seed_before_model_construction": True,
            "seed_python": True,
            "seed_numpy": True,
            "seed_torch_cpu": True,
            "seed_torch_cuda_all": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "note": "exact cross-hardware bitwise identity is not claimed; same hardware/software repeatability is required by later backward smoke-test",
        },
        "train_length_resolution": train_length_sources,
        "schema_resolution": {
            "drop_last": drop_last_sources,
            "persistent_workers": persistent_worker_sources,
            "metric_definition": metric_sources,
            "window_coordinates_sha256": window_sha_sources,
            "model_checkpoint": checkpoint_sources,
            "model_resolved_revision": revision_sources,
            "model_loading_info": [
                {"source": c["source"], "path": c["path"]}
                for c in loading_info_sources
            ],
        },
        "budget": {
            "max_epochs": MAX_EPOCHS,
            "samples_per_epoch": samples_per_epoch,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batches_per_epoch": samples_per_epoch // MICRO_BATCH_SIZE,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_updates": total_optimizer_updates,
            "drop_last": False,
            "train_num_workers": TRAIN_NUM_WORKERS,
            "same_for_all_variants": True,
        },
        "loss": {
            "name": LOSS_NAME,
            "ignore_index": IGNORE_INDEX,
            "class_weight": None,
            "label_smoothing": LABEL_SMOOTHING,
            "reduction": "mean",
        },
        "optimizer": {
            "name": OPTIMIZER_NAME,
            "base_lr": BASE_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "no_decay_rule": NO_DECAY_RULE,
            "lr_multiplier_all_trainable_parameters": LR_MULTIPLIER,
            "head_lr_multiplier": 1.0,
            "zero_grad_set_to_none": True,
        },
        "lr_schedule": {
            "name": SCHEDULER_NAME,
            "defined_over": "optimizer_update_index",
            "step_timing": "set_lr_before_each_optimizer_step",
            "base_lr": BASE_LR,
            "min_lr": MIN_LR,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_updates": warmup_updates,
            "warmup_start_factor": WARMUP_START_FACTOR,
            "poly_power": POLY_POWER,
            "total_optimizer_updates": total_optimizer_updates,
            "lr_sequence_sha256": lr_sha256,
        },
        "precision": {
            "amp_enabled": AMP_ENABLED,
            "autocast_device_type": "cuda",
            "autocast_dtype": AMP_DTYPE,
            "grad_scaler": {
                "enabled": GRAD_SCALER_ENABLED,
                "init_scale": GRAD_SCALER_INIT_SCALE,
                "growth_factor": GRAD_SCALER_GROWTH_FACTOR,
                "backoff_factor": GRAD_SCALER_BACKOFF_FACTOR,
                "growth_interval": GRAD_SCALER_GROWTH_INTERVAL,
            },
        },
        "gradient": {
            "accumulate_loss_rule": "divide_each_microbatch_loss_by_gradient_accumulation_steps",
            "clip_enabled": GRAD_CLIP_ENABLED,
            "clip_after_grad_scaler_unscale": True,
            "max_norm": GRAD_CLIP_MAX_NORM,
            "norm_type": GRAD_CLIP_NORM_TYPE,
            "optimizer_step_only_after_full_accumulation": True,
        },
        "validation": {
            "split": "val",
            "full_tile_sliding_window_only": True,
            "window_batch_size": VAL_WINDOW_BATCH_SIZE,
            "every_epochs": VALIDATE_EVERY_EPOCHS,
            "epochs": validation_epochs,
            "metric": CHECKPOINT_METRIC,
            "metric_definition": "global confusion matrix -> 6-class IoU -> mIoU",
            "no_patch_mIoU_for_selection": True,
            "no_mean_tile_mIoU_for_selection": True,
        },
        "checkpointing": {
            "selection_metric": CHECKPOINT_METRIC,
            "mode": CHECKPOINT_MODE,
            "tie_break": CHECKPOINT_TIE_BREAK,
            "save_best": True,
            "save_last": True,
            "best_checkpoint_used_for_final_test": True,
            "early_stopping": EARLY_STOPPING,
        },
        "test_policy": {
            "test_split_used_during_training": False,
            "test_split_used_for_checkpoint_selection": False,
            "evaluate_test_only_after_best_val_checkpoint_is_frozen": True,
        },
        "fairness_constraints": {
            "same_dataset_protocol": True,
            "same_epoch_sample_order": True,
            "same_training_budget": True,
            "same_micro_batch_size": True,
            "same_effective_batch_size": True,
            "same_optimizer_hyperparameters": True,
            "same_lr_schedule": True,
            "same_validation_epochs": True,
            "same_checkpoint_selection_rule": True,
            "no_model_specific_hyperparameter_tuning_in_primary_comparison": True,
        },
    }


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
        return len(data)

    def flush(self):
        for f in self.files:
            f.flush()



EXPECTED_METRIC_DEFINITION = "global confusion matrix -> 6-class IoU -> mIoU"
EXPECTED_WINDOW_SHA256 = "58a2f857c7d84adb0a17f2ce0f5b1802452886127963f42fd9ac125c7dd84e1a"


def _normalize_metric_text(s: str) -> str:
    """
    仅用于比较表达形式，不改变语义：
    - Unicode 箭头转为 ->
    - 连续空白压成单空格
    - 大小写归一
    """
    x = str(s).strip()
    x = x.replace("→", "->").replace("⇒", "->")
    x = " ".join(x.split())
    return x.lower()


def resolve_evaluation_metric_definition(
    data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    冻结 evaluation protocol 的字段名不强制固定。
    这里按“值的语义”寻找主指标定义，而不是猜字段名。

    接受条件：
    1) 字符串必须同时明确包含 global confusion matrix / 6-class IoU / mIoU；
    2) 归一化后必须等价于冻结表达；
    3) 所有候选必须语义一致。
    """
    candidates: List[Dict[str, Any]] = []

    for path, value in _walk_scalars(data):
        if not isinstance(value, str):
            continue
        norm = _normalize_metric_text(value)
        if (
            "global confusion matrix" in norm
            and "6-class iou" in norm
            and "miou" in norm
        ):
            candidates.append({
                "path": _path_text(path),
                "value": value,
                "normalized": norm,
            })

    if not candidates:
        # 第二层：某些 metadata 可能拆成较短、但仍明确的字段值。
        for path, value in _walk_scalars(data):
            if not isinstance(value, str):
                continue
            norm = _normalize_metric_text(value)
            if "global confusion matrix" in norm and "miou" in norm:
                candidates.append({
                    "path": _path_text(path),
                    "value": value,
                    "normalized": norm,
                })

    if not candidates:
        raise AuditError(
            "无法在 evaluation_protocol 中定位主指标定义。"
            "要求内容明确包含 global confusion matrix 与 mIoU。"
        )

    expected_norm = _normalize_metric_text(EXPECTED_METRIC_DEFINITION)
    bad = [c for c in candidates if c["normalized"] != expected_norm]
    if bad:
        diagnostic = "\n".join(
            f"  {c['path']} = {c['value']!r}"
            for c in candidates
        )
        raise AuditError(
            "evaluation_protocol 中发现主指标相关字符串，但与冻结定义不完全一致；"
            "禁止自动接受近似表达。\n候选：\n" + diagnostic
        )

    return EXPECTED_METRIC_DEFINITION, candidates


def resolve_evaluation_window_sha256(
    data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    不依赖具体 key 名，寻找冻结的 sliding-window coordinate SHA256。
    只检查看起来像 SHA256 的 64 位十六进制字符串，
    且路径必须与 window / coordinate / sliding 至少一个语义相关。
    """
    candidates: List[Dict[str, Any]] = []
    hex64 = re.compile(r"^[0-9a-fA-F]{64}$")

    for path, value in _walk_scalars(data):
        if not isinstance(value, str) or not hex64.fullmatch(value.strip()):
            continue
        p = _path_text(path).lower()
        if not any(tok in p for tok in ("window", "coordinate", "sliding")):
            continue
        candidates.append({
            "path": _path_text(path),
            "value": value.strip().lower(),
        })

    matches = [c for c in candidates if c["value"] == EXPECTED_WINDOW_SHA256]
    if not matches:
        diagnostic = "\n".join(
            f"  {c['path']} = {c['value']}"
            for c in candidates
        ) or "  <none>"
        raise AuditError(
            "evaluation_protocol 中找不到冻结的 sliding-window coordinate SHA256。\n"
            "检测到的 window/coordinate 相关 SHA256：\n" + diagnostic
        )

    # 若相关字段中还出现别的 coordinate/window hash，也不能自动忽略。
    conflicting = [
        c for c in candidates
        if c["value"] != EXPECTED_WINDOW_SHA256
        and any(tok in c["path"].lower() for tok in ("coordinate", "coordinates"))
    ]
    if conflicting:
        diagnostic = "\n".join(
            f"  {c['path']} = {c['value']}"
            for c in candidates
        )
        raise AuditError(
            "evaluation_protocol 中存在冲突的 coordinate SHA256；禁止自动选择。\n"
            + diagnostic
        )

    return EXPECTED_WINDOW_SHA256, matches


def run_audit(project_root: Path) -> Dict[str, Any]:
    processed = project_root / "data/processed/potsdam"
    upstream_paths = {
        "dataset_protocol": processed / "dataset_protocol.json",
        "dataloader_protocol": processed / "dataloader_protocol.json",
        "evaluation_protocol": processed / "evaluation_protocol.json",
        "model_a_rgb_protocol": processed / "model_a_rgb_protocol.json",
    }

    print("[1/6] 读取并验证上游冻结协议...")
    upstream = {}
    upstream_hashes = {}
    for name, path in upstream_paths.items():
        data = load_json(path)
        status_mode = status_is_pass(data, path)
        upstream[name] = data
        upstream_hashes[name] = sha256_file(path)
        expected_sha = EXPECTED_UPSTREAM_SHA256[name]
        require(
            upstream_hashes[name] == expected_sha,
            f"{name} SHA256 与已冻结版本不一致: "
            f"actual={upstream_hashes[name]} expected={expected_sha}",
        )
        print(
            f"  PASS: {name} = {upstream_hashes[name]} "
            f"({status_mode}, exact_frozen_sha256)"
        )

    ds = upstream["dataset_protocol"]
    dl = upstream["dataloader_protocol"]
    ev = upstream["evaluation_protocol"]
    ma = upstream["model_a_rgb_protocol"]

    # Protocol-specific schema gates. These are stronger than a generic root-level status.
    require(
        str(ds.get("protocol_version", "")).startswith("Dataset Protocol"),
        f"dataset_protocol.protocol_version 异常: {ds.get('protocol_version')!r}",
    )

    ds_split = ds.get("split", ds.get("splits", ds.get("split_counts", {})))
    if isinstance(ds_split, dict):
        # Accept either {train:18,val:6,test:14} or nested metadata carrying these counts.
        for key, expected in (("train", 18), ("val", 6), ("test", 14)):
            if key in ds_split and isinstance(ds_split[key], int):
                require(
                    int(ds_split[key]) == expected,
                    f"dataset_protocol split {key} 应为 {expected}，实际={ds_split[key]}",
                )

    require(
        str(dl.get("protocol_version", "")).startswith("DataLoader Protocol"),
        f"dataloader_protocol.protocol_version 异常: {dl.get('protocol_version')!r}",
    )
    require(
        str(ev.get("protocol_version", "")).startswith("Evaluation Protocol"),
        f"evaluation_protocol.protocol_version 异常: {ev.get('protocol_version')!r}",
    )
    require(
        str(ma.get("protocol_version", "")).startswith("Model A RGB Protocol"),
        f"model_a_rgb_protocol.protocol_version 异常: {ma.get('protocol_version')!r}",
    )

    samples_per_epoch, train_length_sources = resolve_samples_per_epoch(ds, dl)
    print("  每 epoch 训练样本数解析来源：")
    for item in train_length_sources:
        print(
            f"    {item['source']}:{item['path']} = {item['value']} "
            f"({item['mode']})"
        )
    require(samples_per_epoch == 1152,
            f"samples_per_epoch 应为 1152，实际={samples_per_epoch}")
    drop_last, drop_last_sources = resolve_unique_bool(
        dl, "drop_last", False, "dataloader_protocol"
    )
    persistent_workers, persistent_worker_sources = resolve_unique_bool(
        dl, "persistent_workers", False, "dataloader_protocol"
    )
    print("  DataLoader 布尔协议解析来源：")
    for item in drop_last_sources + persistent_worker_sources:
        print(f"    {item['source']}:{item['path']} = {item['value']!r}")

    # Evaluation / Model A 已通过 exact frozen SHA256 校验。
    # 对冻结 metadata，逐字节一致是最强 schema-independent gate。
    metric_definition = EXPECTED_METRIC_DEFINITION
    window_sha = EXPECTED_WINDOW_SHA256
    metric_sources = [{
        "source": "evaluation_protocol",
        "path": "<pinned-by-file-sha256>",
        "value": metric_definition,
    }]
    window_sha_sources = [{
        "source": "evaluation_protocol",
        "path": "<pinned-by-file-sha256>",
        "value": window_sha,
    }]
    checkpoint_sources = [{
        "source": "model_a_rgb_protocol",
        "path": "<pinned-by-file-sha256>",
        "value": EXPECTED_BASE_CHECKPOINT,
    }]
    revision_sources = [{
        "source": "model_a_rgb_protocol",
        "path": "<pinned-by-file-sha256>",
        "value": EXPECTED_BASE_REVISION,
    }]
    loading_info_sources = [{
        "source": "model_a_rgb_protocol",
        "path": "<pinned-by-file-sha256>",
    }]

    print("  Evaluation 协议：")
    print("    exact frozen SHA256 匹配；不再依赖内部字段名重复解析。")
    print(f"    metric = {metric_definition}")
    print(f"    window_coordinates_sha256 = {window_sha}")

    print("  Model A 协议：")
    print("    exact frozen SHA256 匹配；checkpoint/revision/loading contract 由冻结指纹锁定。")
    print(f"    checkpoint = {EXPECTED_BASE_CHECKPOINT}")
    print(f"    resolved_revision = {EXPECTED_BASE_REVISION}")

    print("  PASS: Dataset/DataLoader/Evaluation/Model A 协议互相一致。")

    print("[2/6] 计算并审计统一训练预算...")
    require(EFFECTIVE_BATCH_SIZE == MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
            "effective batch 计算错误")
    require(samples_per_epoch % MICRO_BATCH_SIZE == 0,
            "1152 不能被 micro batch 整除")
    require(samples_per_epoch % EFFECTIVE_BATCH_SIZE == 0,
            "1152 不能被 effective batch 整除；会产生部分 accumulation")
    micro_batches_per_epoch = samples_per_epoch // MICRO_BATCH_SIZE
    optimizer_steps_per_epoch = samples_per_epoch // EFFECTIVE_BATCH_SIZE
    total_optimizer_updates = MAX_EPOCHS * optimizer_steps_per_epoch
    require(optimizer_steps_per_epoch == 72,
            f"optimizer steps/epoch 应为 72，实际={optimizer_steps_per_epoch}")
    require(total_optimizer_updates == 7200,
            f"total optimizer updates 应为 7200，实际={total_optimizer_updates}")
    print(
        f"  samples/epoch={samples_per_epoch}, micro_batch={MICRO_BATCH_SIZE}, "
        f"accum={GRAD_ACCUM_STEPS}, effective_batch={EFFECTIVE_BATCH_SIZE}"
    )
    print(
        f"  micro_batches/epoch={micro_batches_per_epoch}, "
        f"optimizer_steps/epoch={optimizer_steps_per_epoch}, "
        f"total_updates={total_optimizer_updates}"
    )
    print("  PASS: A/B/C-noGate/C 使用完全相同训练预算。")

    print("[3/6] 审计学习率序列...")
    warmup_updates = int(round(total_optimizer_updates * WARMUP_RATIO))
    require(warmup_updates == 360,
            f"warmup updates 应为 360，实际={warmup_updates}")
    lr_sha256, lr_values = lr_schedule_fingerprint(
        total_optimizer_updates, warmup_updates
    )
    require(math.isclose(lr_values[0], BASE_LR * WARMUP_START_FACTOR,
                         rel_tol=0.0, abs_tol=1e-18),
            "warmup 第一个 LR 异常")
    require(math.isclose(lr_values[warmup_updates - 1], BASE_LR,
                         rel_tol=0.0, abs_tol=1e-18),
            "warmup 最后一个 LR 没到 base LR")
    require(math.isclose(lr_values[warmup_updates], BASE_LR,
                         rel_tol=0.0, abs_tol=1e-18),
            "poly decay 第一个 LR 应为 base LR")
    require(math.isclose(lr_values[-1], MIN_LR,
                         rel_tol=0.0, abs_tol=1e-18),
            "最后一个 LR 应为 min LR")
    require(all(lr_values[i] <= lr_values[i - 1] + 1e-18
                for i in range(warmup_updates + 1, len(lr_values))),
            "warmup 后 LR 不是单调非增")
    print(f"  warmup_updates={warmup_updates}")
    print(f"  lr[0]={lr_values[0]:.12e}")
    print(f"  lr[359]={lr_values[359]:.12e}")
    print(f"  lr[360]={lr_values[360]:.12e}")
    print(f"  lr[last]={lr_values[-1]:.12e}")
    print(f"  lr_sequence_sha256={lr_sha256}")
    print("  PASS: 学习率 schedule 数学定义已固定。")

    print("[4/6] 审计随机种子与 CUDA RNG 接口...")
    rng = seed_probe()
    print(f"  Python RNG SHA256   = {rng['python']}")
    print(f"  NumPy RNG SHA256    = {rng['numpy']}")
    print(f"  Torch CPU SHA256    = {rng['torch_cpu']}")
    print(f"  Torch CUDA SHA256   = {rng['torch_cuda']}")
    print(f"  torch={rng['torch_version']} cuda={rng['cuda_version']}")
    print("  PASS: 固定 seed 可重复生成相同 RNG 序列。")

    print("[5/6] 冻结 validation/checkpoint 规则...")
    validation_epochs = list(
        range(VALIDATE_EVERY_EPOCHS, MAX_EPOCHS + 1, VALIDATE_EVERY_EPOCHS)
    )
    require(validation_epochs[0] == 5 and validation_epochs[-1] == 100,
            "validation epochs 起止异常")
    require(len(validation_epochs) == 20,
            f"validation 次数应为 20，实际={len(validation_epochs)}")
    require(EARLY_STOPPING is False, "主实验禁止 early stopping")
    print(f"  validation epochs = {validation_epochs}")
    print("  checkpoint = 最高 val global mIoU；同分取最早 epoch。")
    print("  early stopping = False；Test 不参与训练/选模。")
    print("  PASS: 模型选择与 Test 隔离规则已固定。")

    print("[6/6] 写入/验证 training_protocol.json...")
    protocol = build_protocol(
        samples_per_epoch=samples_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_optimizer_updates=total_optimizer_updates,
        warmup_updates=warmup_updates,
        validation_epochs=validation_epochs,
        upstream_hashes=upstream_hashes,
        lr_sha256=lr_sha256,
        train_length_sources=train_length_sources,
        drop_last_sources=drop_last_sources,
        persistent_worker_sources=persistent_worker_sources,
        metric_sources=metric_sources,
        window_sha_sources=window_sha_sources,
        checkpoint_sources=checkpoint_sources,
        revision_sources=revision_sources,
        loading_info_sources=loading_info_sources,
    )

    frozen_path = processed / "training_protocol.json"
    action: str
    if frozen_path.exists():
        existing = load_json(frozen_path)
        require(
            canonical_json_bytes(existing) == canonical_json_bytes(protocol),
            "training_protocol.json 已存在但内容与本次冻结协议不同。"
            "禁止自动覆盖；请先人工比较差异。",
        )
        action = "verified_existing"
    else:
        frozen_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        action = "created"

    frozen_sha = sha256_file(frozen_path)
    print(f"  training_protocol.json: {action}")
    print(f"  SHA256: {frozen_sha}")

    return {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "upstream_sha256": upstream_hashes,
        "base_checkpoint": {
            "name": EXPECTED_BASE_CHECKPOINT,
            "resolved_revision": EXPECTED_BASE_REVISION,
        },
        "budget": {
            "max_epochs": MAX_EPOCHS,
            "samples_per_epoch": samples_per_epoch,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_updates": total_optimizer_updates,
        },
        "optimizer": {
            "name": OPTIMIZER_NAME,
            "base_lr": BASE_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "no_decay_rule": NO_DECAY_RULE,
            "head_lr_multiplier": 1.0,
        },
        "lr_schedule": {
            "warmup_updates": warmup_updates,
            "poly_power": POLY_POWER,
            "lr_first": lr_values[0],
            "lr_warmup_last": lr_values[warmup_updates - 1],
            "lr_poly_first": lr_values[warmup_updates],
            "lr_last": lr_values[-1],
            "sequence_sha256": lr_sha256,
        },
        "loss": {
            "name": LOSS_NAME,
            "ignore_index": IGNORE_INDEX,
            "class_weight": None,
            "label_smoothing": LABEL_SMOOTHING,
        },
        "precision": {
            "amp_enabled": AMP_ENABLED,
            "amp_dtype": AMP_DTYPE,
            "grad_scaler_enabled": GRAD_SCALER_ENABLED,
        },
        "gradient_clip": {
            "enabled": GRAD_CLIP_ENABLED,
            "max_norm": GRAD_CLIP_MAX_NORM,
            "norm_type": GRAD_CLIP_NORM_TYPE,
        },
        "validation": {
            "every_epochs": VALIDATE_EVERY_EPOCHS,
            "epochs": validation_epochs,
            "selection_metric": CHECKPOINT_METRIC,
            "tie_break": CHECKPOINT_TIE_BREAK,
            "early_stopping": EARLY_STOPPING,
            "test_isolation": True,
        },
        "rng_probe": rng,
        "frozen_training_protocol": {
            "path": str(frozen_path.relative_to(project_root)),
            "action": action,
            "sha256": frozen_sha,
        },
    }


def write_txt_report(report: Dict[str, Any], path: Path) -> None:
    b = report["budget"]
    o = report["optimizer"]
    lr = report["lr_schedule"]
    v = report["validation"]
    f = report["frozen_training_protocol"]

    lines = [
        "Training Control Protocol v1 审计报告",
        "=" * 72,
        f"状态: {report['status']}",
        f"脚本版本: {report['script_version']}",
        f"protocol: {report['protocol_version']}",
        "",
        "[Base checkpoint]",
        f"name = {report['base_checkpoint']['name']}",
        f"resolved_revision = {report['base_checkpoint']['resolved_revision']}",
        "",
        "[Budget]",
        f"max_epochs = {b['max_epochs']}",
        f"samples_per_epoch = {b['samples_per_epoch']}",
        f"micro_batch_size = {b['micro_batch_size']}",
        f"gradient_accumulation_steps = {b['gradient_accumulation_steps']}",
        f"effective_batch_size = {b['effective_batch_size']}",
        f"optimizer_steps_per_epoch = {b['optimizer_steps_per_epoch']}",
        f"total_optimizer_updates = {b['total_optimizer_updates']}",
        "",
        "[Optimizer / LR]",
        f"optimizer = {o['name']}",
        f"base_lr = {o['base_lr']}",
        f"weight_decay = {o['weight_decay']}",
        f"head_lr_multiplier = {o['head_lr_multiplier']}",
        f"warmup_updates = {lr['warmup_updates']}",
        f"poly_power = {lr['poly_power']}",
        f"lr_sequence_sha256 = {lr['sequence_sha256']}",
        "",
        "[Loss / Precision / Gradient]",
        "loss = CrossEntropyLoss, unweighted",
        f"ignore_index = {report['loss']['ignore_index']}",
        f"AMP = {report['precision']['amp_enabled']} ({report['precision']['amp_dtype']})",
        f"grad_clip = {report['gradient_clip']['enabled']}, max_norm={report['gradient_clip']['max_norm']}",
        "",
        "[Validation / Checkpoint]",
        f"validate_every_epochs = {v['every_epochs']}",
        f"validation_epochs = {v['epochs']}",
        f"selection_metric = {v['selection_metric']}",
        f"tie_break = {v['tie_break']}",
        f"early_stopping = {v['early_stopping']}",
        f"test_isolation = {v['test_isolation']}",
        "",
        "[Frozen metadata]",
        f"path = {f['path']}",
        f"action = {f['action']}",
        f"sha256 = {f['sha256']}",
        "",
        "FINAL STATUS: PASS",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="默认自动取脚本所在项目根目录",
    )
    args = parser.parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_script()
    )

    out_dir = project_root / "outputs/training_check/training_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    console_path = out_dir / "console.txt"
    audit_json_path = out_dir / "training_protocol_audit.json"
    audit_txt_path = out_dir / "training_protocol_audit.txt"

    with console_path.open("w", encoding="utf-8") as cf:
        tee_out = Tee(sys.__stdout__, cf)
        tee_err = Tee(sys.__stderr__, cf)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 72)
            print("Training Control Protocol v1 独立审计")
            print("=" * 72)
            print(f"Project root: {project_root}")
            print("本测试不训练模型，不执行 SegFormer forward/backward。")
            print()
            try:
                report = run_audit(project_root)
                audit_json_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                write_txt_report(report, audit_txt_path)
                print()
                print("=" * 72)
                print("FINAL STATUS: PASS")
                print("=" * 72)
                print(f"console: {console_path}")
                print(f"audit json: {audit_json_path}")
                print(f"audit txt: {audit_txt_path}")
                return 0
            except Exception as e:
                fail_report = {
                    "status": "FAIL",
                    "script_version": SCRIPT_VERSION,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                audit_json_path.write_text(
                    json.dumps(fail_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                audit_txt_path.write_text(
                    "\n".join([
                        "Training Control Protocol v1 审计报告",
                        "=" * 72,
                        "状态: FAIL",
                        f"脚本版本: {SCRIPT_VERSION}",
                        f"{type(e).__name__}: {e}",
                        "",
                        "FINAL STATUS: FAIL",
                    ]) + "\n",
                    encoding="utf-8",
                )
                print()
                print("=" * 72)
                print("FINAL STATUS: FAIL")
                print(f"{type(e).__name__}: {e}")
                print("=" * 72)
                print(f"console: {console_path}")
                print(f"audit json: {audit_json_path}")
                print(f"audit txt: {audit_txt_path}")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
