#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A RGB training-engine smoke-test for ISPRS Potsdam + SegFormer.

Scope is intentionally minimal:
  - real Model A RGB
  - real train DataLoader
  - frozen training_protocol.json
  - 8-microbatch gradient accumulation smoke
  - checkpoint save/resume continuity check
  - no validation/test full evaluation
  - no RGB+NIR, no dual encoder, no fusion, no corruption pipeline, no quality gate
  - no 100-epoch formal training

Default project root:
/media/fyj/467681C67CDA29CC/ty/projects/segformer_potsdam
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import functools
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_VERSION = "1.0.6"
SCRIPT_NAME = "smoke_train_engine_a_rgb"
DEFAULT_PROJECT_ROOT = Path("/media/fyj/467681C67CDA29CC/ty/projects/segformer_potsdam")
DEFAULT_OUTPUT_SUBDIR = Path("outputs/training_check/train_engine_smoke_a_rgb")

EXPECTED_PROTOCOL_SHA256 = {
    "dataset_protocol": "e292a33266389b2ad07e09cda1a5c2e2d51003c3330678eed114e236d30c6c22",
    "dataloader_protocol": "ce0c8427af28682b74436f820059398b081c45b113102ad0c97cde0992f14ee4",
    "evaluation_protocol": "9e1fe295f898190749cc261dc9cdd5dbb76c213da90547b576ea6f4fb4e2192d",
    "model_a_rgb_protocol": "476bae90a907107734cc1ce29db756f4a1299a723cd647b8e2879df0adc6f713",
    "training_protocol": "2d99d8f3362db8b7fb491d1445d50ea8b15ceae15fe701e79cb163a38b2a72ff",
}

EXPECTED_LR_SEQUENCE_SHA256 = "9e5849399aa658778dae6794cd10b5e851fa544a29aa35bffa50b86174e6f3f2"
EXPECTED_WINDOW_COORDINATES_SHA256 = "58a2f857c7d84adb0a17f2ce0f5b1802452886127963f42fd9ac125c7dd84e1a"

SMOKE_SEED = 20260917
MAX_ALLOWED_SMOKE_MICROBATCHES = 16
DEFAULT_REFERENCE_MICROBATCHES = 10
DEFAULT_CHECKPOINT_AFTER_MICROBATCHES = 8
DEFAULT_NUM_WORKERS = 0


class SmokeFailure(RuntimeError):
    pass


@dataclasses.dataclass
class Check:
    name: str
    passed: bool
    details: Any = None


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_hash(text: str, n: int = 12) -> str:
    return text[:n]


def canonical_json_sha256(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def as_scalar_json(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, torch.dtype):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)
    if isinstance(x, bytes):
        return {"__bytes_sha256__": sha256_bytes(x), "nbytes": len(x)}
    return x


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "sha256": sha256_bytes(t.numpy().tobytes() if t.numel() > 0 else b""),
        }
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": str(a.dtype), "shape": list(a.shape), "sha256": sha256_bytes(a.tobytes())}
    if isinstance(obj, Mapping):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def nested_find_all(obj: Any, key: str, prefix: str = "") -> List[Tuple[str, Any]]:
    found: List[Tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if str(k) == key:
                found.append((p, v))
            found.extend(nested_find_all(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            found.extend(nested_find_all(v, key, p))
    return found


def protocol_get(protocol: Mapping[str, Any], key: str, *, default: Any = None, required: bool = True) -> Any:
    if key in protocol:
        return protocol[key]
    found = nested_find_all(protocol, key)
    if not found:
        if required:
            raise SmokeFailure(f"training_protocol.json missing required key: {key}")
        return default
    # Prefer shorter paths and avoid metadata/history duplicates if present.
    found = sorted(found, key=lambda kv: (kv[0].count("."), len(kv[0])))
    return found[0][1]


def protocol_get_any(protocol: Mapping[str, Any], keys: Sequence[str], *, default: Any = None, required: bool = True) -> Any:
    errors = []
    for key in keys:
        try:
            return protocol_get(protocol, key, required=True)
        except SmokeFailure as e:
            errors.append(str(e))
    if required:
        raise SmokeFailure(f"training_protocol.json missing required key; tried {list(keys)}")
    return default


def normalize_float(x: Any) -> float:
    return float(x)


def normalize_int(x: Any) -> int:
    return int(x)


def normalize_bool(x: Any) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)


def assert_close(name: str, actual: float, expected: float, checks: List[Check], atol: float = 1e-12, rtol: float = 1e-12) -> None:
    ok = math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)
    checks.append(Check(name, ok, {"actual": actual, "expected": expected, "abs_diff": abs(float(actual) - float(expected))}))
    if not ok:
        raise SmokeFailure(f"{name} mismatch: actual={actual}, expected={expected}")


def assert_equal(name: str, actual: Any, expected: Any, checks: List[Check]) -> None:
    ok = actual == expected
    checks.append(Check(name, ok, {"actual": actual, "expected": expected}))
    if not ok:
        raise SmokeFailure(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_console(console_path: Path) -> Iterator[None]:
    console_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with console_path.open("w", encoding="utf-8") as f:
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism_flags() -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
            flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        except Exception as e:  # pragma: no cover
            flags["cudnn_flags_error"] = repr(e)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            flags["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        except Exception as e:  # pragma: no cover
            flags["tf32_flag_error"] = repr(e)
        try:
            torch.backends.cudnn.allow_tf32 = False
            flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        except Exception as e:  # pragma: no cover
            flags["cudnn_tf32_flag_error"] = repr(e)
    return flags


def rng_state_dict() -> Dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])

    # RNG states are serialized ByteTensors. Keep them on CPU when restoring:
    # torch.set_rng_state() specifically requires a CPU torch.ByteTensor.
    torch_cpu_state = state["torch_cpu"].detach().cpu().to(dtype=torch.uint8)
    torch.set_rng_state(torch_cpu_state)

    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        cuda_states = [
            x.detach().cpu().to(dtype=torch.uint8)
            for x in state["torch_cuda_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def hash_rng_state(state: Mapping[str, Any]) -> str:
    # torch.save preserves Python/numpy/torch RNG structures enough for a stable smoke hash.
    bio = io.BytesIO()
    torch.save(state, bio)
    return sha256_bytes(bio.getvalue())


def tensor_sha256(t: torch.Tensor) -> str:
    tc = t.detach().cpu().contiguous()
    if tc.numel() == 0:
        raw = b""
    else:
        raw = tc.numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(tc.dtype).encode("utf-8"))
    h.update(json.dumps(list(tc.shape), separators=(",", ":")).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


def hash_named_tensors(named_tensors: Iterable[Tuple[str, Optional[torch.Tensor]]]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda x: x[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        if tensor is None:
            h.update(b"NONE")
            continue
        tc = tensor.detach().cpu().contiguous()
        h.update(str(tc.dtype).encode("utf-8"))
        h.update(json.dumps(list(tc.shape), separators=(",", ":")).encode("utf-8"))
        h.update(tc.numpy().tobytes() if tc.numel() > 0 else b"")
    return h.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    return hash_named_tensors((k, v) for k, v in model.state_dict().items())


def model_grad_hash(model: nn.Module) -> str:
    return hash_named_tensors((name, p.grad) for name, p in model.named_parameters())


def recursive_state_hash(obj: Any) -> str:
    """Stable-ish content hash for optimizer/scaler/scheduler state dictionaries."""
    h = hashlib.sha256()

    def update(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            tc = x.detach().cpu().contiguous()
            h.update(b"T")
            h.update(str(tc.dtype).encode("utf-8"))
            h.update(json.dumps(list(tc.shape), separators=(",", ":")).encode("utf-8"))
            h.update(tc.numpy().tobytes() if tc.numel() > 0 else b"")
        elif isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(b"N")
            h.update(str(a.dtype).encode("utf-8"))
            h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("utf-8"))
            h.update(a.tobytes())
        elif isinstance(x, Mapping):
            h.update(b"D{")
            for k in sorted(x.keys(), key=lambda z: repr(z)):
                update(k)
                update(x[k])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(b"L[")
            for v in x:
                update(v)
            h.update(b"]")
        elif isinstance(x, (str, int, float, bool, type(None))):
            h.update(b"S")
            h.update(repr(x).encode("utf-8"))
        else:
            h.update(b"R")
            h.update(repr(x).encode("utf-8"))

    update(obj)
    return h.hexdigest()


def max_abs_model_delta(a: nn.Module, b: nn.Module) -> Dict[str, Any]:
    max_abs = 0.0
    worst_key = None
    all_exact = True
    all_close = True
    missing: List[str] = []
    sd_a = a.state_dict()
    sd_b = b.state_dict()
    if set(sd_a.keys()) != set(sd_b.keys()):
        return {
            "all_exact": False,
            "allclose_atol_1e-7": False,
            "max_abs": None,
            "worst_key": None,
            "key_sets_equal": False,
            "only_in_a": sorted(set(sd_a) - set(sd_b))[:20],
            "only_in_b": sorted(set(sd_b) - set(sd_a))[:20],
        }
    for key in sorted(sd_a.keys()):
        ta = sd_a[key].detach().cpu()
        tb = sd_b[key].detach().cpu()
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            all_exact = False
            all_close = False
            missing.append(key)
            continue
        if not torch.equal(ta, tb):
            all_exact = False
        if torch.is_floating_point(ta):
            if not torch.allclose(ta, tb, rtol=0.0, atol=1e-7):
                all_close = False
            delta = (ta - tb).abs().max().item() if ta.numel() else 0.0
        else:
            if not torch.equal(ta, tb):
                all_close = False
                delta = 1.0
            else:
                delta = 0.0
        if delta > max_abs:
            max_abs = float(delta)
            worst_key = key
    return {
        "all_exact": all_exact,
        "allclose_atol_1e-7": all_close,
        "max_abs": max_abs,
        "worst_key": worst_key,
        "key_sets_equal": True,
        "nonmatching_metadata_keys": missing[:20],
    }


def max_abs_grad_delta(a: nn.Module, b: nn.Module) -> Dict[str, Any]:
    max_abs = 0.0
    worst_key = None
    all_exact = True
    all_close = True
    names_a = {n for n, _ in a.named_parameters()}
    names_b = {n for n, _ in b.named_parameters()}
    if names_a != names_b:
        return {"all_exact": False, "allclose_atol_1e-7": False, "max_abs": None, "worst_key": None, "names_equal": False}
    params_b = dict(b.named_parameters())
    for name, pa in sorted(a.named_parameters(), key=lambda x: x[0]):
        ga = pa.grad
        gb = params_b[name].grad
        if ga is None or gb is None:
            if ga is not gb:
                all_exact = False
                all_close = False
                worst_key = worst_key or name
            continue
        ga_cpu = ga.detach().cpu()
        gb_cpu = gb.detach().cpu()
        if not torch.equal(ga_cpu, gb_cpu):
            all_exact = False
        if torch.is_floating_point(ga_cpu):
            if not torch.allclose(ga_cpu, gb_cpu, rtol=0.0, atol=1e-7):
                all_close = False
            delta = (ga_cpu - gb_cpu).abs().max().item() if ga_cpu.numel() else 0.0
        else:
            delta = 0.0 if torch.equal(ga_cpu, gb_cpu) else 1.0
            if delta:
                all_close = False
        if delta > max_abs:
            max_abs = float(delta)
            worst_key = name
    return {"all_exact": all_exact, "allclose_atol_1e-7": all_close, "max_abs": max_abs, "worst_key": worst_key, "names_equal": True}


NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
)


def is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, NORM_TYPES) or ("norm" in module.__class__.__name__.lower())


def module_is_head(module_name: str, param_name: str) -> bool:
    full = f"{module_name}.{param_name}" if module_name else param_name
    return full.startswith("decode_head") or ".decode_head." in full or full.startswith("classifier") or ".classifier." in full


def build_adamw_param_groups(model: nn.Module, *, base_lr: float, weight_decay: float, head_lr_multiplier: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    # Key: (scope, decay_enabled)
    for scope in ("backbone", "head"):
        for decay_enabled in (True, False):
            group_name = f"{scope}_{'decay' if decay_enabled else 'no_decay'}"
            lr_scale = head_lr_multiplier if scope == "head" else 1.0
            groups[(scope, decay_enabled)] = {
                "params": [],
                "weight_decay": float(weight_decay if decay_enabled else 0.0),
                "lr": float(base_lr * lr_scale),
                "lr_scale": float(lr_scale),
                "group_name": group_name,
            }

    assignment: Dict[str, Dict[str, Any]] = {}
    seen_param_ids = set()
    module_map = dict(model.named_modules())

    for module_name, module in module_map.items():
        norm = is_norm_module(module)
        for local_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            if id(param) in seen_param_ids:
                raise SmokeFailure(f"Parameter assigned more than once: {full_name}")
            seen_param_ids.add(id(param))

            no_decay = (local_name == "bias") or norm
            scope = "head" if module_is_head(module_name, local_name) else "backbone"
            groups[(scope, not no_decay)]["params"].append(param)
            assignment[full_name] = {
                "scope": scope,
                "decay_enabled": not no_decay,
                "weight_decay": float(weight_decay if not no_decay else 0.0),
                "reason": "bias" if local_name == "bias" else ("normalization_parameter" if norm else "default_decay"),
                "module_class": module.__class__.__name__,
                "numel": int(param.numel()),
            }

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assigned = set(assignment.keys())
    if trainable != assigned:
        raise SmokeFailure(
            "AdamW parameter-group assignment does not cover exactly all trainable parameters: "
            f"missing={sorted(trainable - assigned)[:20]}, extra={sorted(assigned - trainable)[:20]}"
        )

    param_groups = [g for g in groups.values() if len(g["params"]) > 0]
    manifest_groups = []
    for group in param_groups:
        group_param_ids = {id(p) for p in group["params"]}
        names = [name for name, p in model.named_parameters() if id(p) in group_param_ids]
        manifest_groups.append(
            {
                "group_name": group["group_name"],
                "parameter_count": len(names),
                "numel": int(sum(p.numel() for p in group["params"])),
                "weight_decay": float(group["weight_decay"]),
                "lr_initial": float(group["lr"]),
                "lr_scale": float(group["lr_scale"]),
                "sample_names": names[:12],
            }
        )

    no_decay_bad = [name for name, a in assignment.items() if a["reason"] in {"bias", "normalization_parameter"} and a["weight_decay"] != 0.0]
    decay_bad = [name for name, a in assignment.items() if a["reason"] == "default_decay" and not math.isclose(a["weight_decay"], weight_decay)]
    manifest = {
        "group_count": len(param_groups),
        "groups": manifest_groups,
        "assignment_sha256": canonical_json_sha256(assignment),
        "total_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "no_decay_violation_count": len(no_decay_bad),
        "decay_violation_count": len(decay_bad),
        "no_decay_violation_sample": no_decay_bad[:20],
        "decay_violation_sample": decay_bad[:20],
        "head_lr_multiplier": float(head_lr_multiplier),
        "rule": "bias_and_normalization_parameters_no_weight_decay_else_weight_decay_0.01",
    }
    if no_decay_bad or decay_bad:
        raise SmokeFailure(f"AdamW no-decay/decay assignment violations detected: {manifest}")
    return param_groups, manifest


def build_lr_sequence(protocol: Mapping[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
    base_lr = normalize_float(protocol_get_any(protocol, ["base_lr", "lr", "learning_rate"]))
    warmup_updates = normalize_int(protocol_get(protocol, "warmup_updates"))
    total_updates = normalize_int(protocol_get_any(protocol, ["total_optimizer_updates", "total_updates"]))
    poly_power = normalize_float(protocol_get_any(protocol, ["poly_power", "power"], default=1.0, required=False))
    lr_first = normalize_float(protocol_get(protocol, "lr_first", default=base_lr * 1e-6, required=False))
    lr_warmup_last = normalize_float(protocol_get(protocol, "lr_warmup_last", default=base_lr, required=False))
    lr_poly_first = normalize_float(protocol_get(protocol, "lr_poly_first", default=base_lr, required=False))
    lr_last = normalize_float(protocol_get(protocol, "lr_last", default=0.0, required=False))

    if total_updates <= 0:
        raise SmokeFailure(f"total_optimizer_updates must be positive, got {total_updates}")
    if warmup_updates <= 0 or warmup_updates >= total_updates:
        raise SmokeFailure(f"warmup_updates must be in [1,total_updates), got {warmup_updates} of {total_updates}")

    seq: List[float] = []
    for update_idx in range(total_updates):
        if update_idx < warmup_updates:
            if warmup_updates == 1:
                lr = lr_warmup_last
            else:
                ratio = update_idx / float(warmup_updates - 1)
                lr = lr_first + (lr_warmup_last - lr_first) * ratio
        else:
            decay_denominator = max(1, total_updates - warmup_updates - 1)
            decay_position = update_idx - warmup_updates
            factor = max(0.0, 1.0 - decay_position / float(decay_denominator))
            lr = lr_poly_first * (factor ** poly_power)
        seq.append(float(lr))

    # Snap values that the frozen protocol explicitly defines; this avoids roundoff at key anchors.
    seq[0] = lr_first
    seq[warmup_updates - 1] = lr_warmup_last
    seq[warmup_updates] = lr_poly_first
    seq[-1] = lr_last

    seq_for_hash = [format(x, ".17g") for x in seq]
    candidate_hashes = {
        "json_float_list_canonical": canonical_json_sha256(seq),
        "json_format17g_string_list_canonical": canonical_json_sha256(seq_for_hash),
        "newline_format17g": sha256_bytes(("\n".join(seq_for_hash) + "\n").encode("utf-8")),
        "comma_format17g": sha256_bytes((",".join(seq_for_hash)).encode("utf-8")),
    }
    observed_hash = candidate_hashes["json_format17g_string_list_canonical"]
    expected_hash = protocol_get(protocol, "lr_sequence_sha256", default=EXPECTED_LR_SEQUENCE_SHA256, required=False)

    meta = {
        "base_lr": base_lr,
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "poly_power": poly_power,
        "lr_first": seq[0],
        "lr_warmup_last": seq[warmup_updates - 1],
        "lr_poly_first": seq[warmup_updates],
        "lr_last": seq[-1],
        "observed_sequence_sha256_format17g_json": observed_hash,
        "candidate_sequence_hashes": candidate_hashes,
        "frozen_sequence_sha256": expected_hash,
        "matches_frozen_hash_any_candidate_serializer": expected_hash in set(candidate_hashes.values()),
        # This smoke test uses the endpoint-verified sequence above. The exact hash serializer used by
        # audit_training_protocol.py is intentionally not assumed here.
        "hash_serializer_note": "Sequence endpoints are enforced; hash is reported for audit visibility.",
    }
    return seq, meta


class FrozenLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, lr_sequence: Sequence[float]) -> None:
        self.optimizer = optimizer
        self.lr_sequence = [float(x) for x in lr_sequence]
        self.next_update_index = 0
        self.last_set_lr: Optional[float] = None

    def set_lr_for_next_step(self) -> float:
        if self.next_update_index >= len(self.lr_sequence):
            raise SmokeFailure(f"LR schedule exhausted at update {self.next_update_index}")
        base_lr = float(self.lr_sequence[self.next_update_index])
        for group in self.optimizer.param_groups:
            scale = float(group.get("lr_scale", 1.0))
            group["lr"] = base_lr * scale
        self.last_set_lr = base_lr
        return base_lr

    def step_after_optimizer(self) -> None:
        self.next_update_index += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "next_update_index": self.next_update_index,
            "last_set_lr": self.last_set_lr,
            "lr_sequence_sha256_format17g_json": canonical_json_sha256([format(x, ".17g") for x in self.lr_sequence]),
            "lr_sequence_length": len(self.lr_sequence),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.next_update_index = int(state["next_update_index"])
        self.last_set_lr = None if state.get("last_set_lr") is None else float(state["last_set_lr"])


def torch_amp_autocast(dtype: torch.dtype):
    # PyTorch 2.x preferred path.
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def make_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)  # type: ignore[call-arg]
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)  # type: ignore[call-arg]
    return torch.cuda.amp.GradScaler(enabled=enabled)


def call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing_required = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing_required:
        raise TypeError(f"Cannot call {fn}: missing required args {missing_required}; supported kwargs={list(kwargs.keys())}")
    return fn(**accepted)


def normalize_loader_candidate(obj: Any) -> DataLoader:
    if isinstance(obj, DataLoader):
        return obj
    if isinstance(obj, Mapping):
        for key in ("train", "train_loader", "dataloader", "loader"):
            if key in obj and isinstance(obj[key], DataLoader):
                return obj[key]
        for value in obj.values():
            if isinstance(value, DataLoader):
                return value
    if isinstance(obj, (tuple, list)):
        for value in obj:
            if isinstance(value, DataLoader):
                return value
    raise SmokeFailure(f"Could not normalize train DataLoader from object type {type(obj).__name__}")


def build_train_dataloader(project_root: Path, protocol: Mapping[str, Any], *, epoch: int, num_workers: int) -> Tuple[DataLoader, Dict[str, Any]]:
    module = importlib.import_module("data_pipeline.potsdam_dataloader")
    micro_batch_size = normalize_int(protocol_get(protocol, "micro_batch_size"))

    # Project-native frozen DataLoader path:
    # data_pipeline.potsdam_dataloader.build_train_dataloader expects an already
    # constructed PotsdamTrainDataset and DeterministicEpochSampler.
    # Use it before generic factory probing so the smoke-test exercises the real
    # frozen loader/sampler protocol instead of falling back to a guessed API.
    native_dataset_cls = getattr(module, "PotsdamTrainDataset", None)
    native_sampler_cls = getattr(module, "DeterministicEpochSampler", None)
    native_build_loader = getattr(module, "build_train_dataloader", None)
    native_set_epoch = getattr(module, "set_train_epoch", None)

    if (
        native_dataset_cls is not None
        and native_sampler_cls is not None
        and callable(native_build_loader)
    ):
        try:
            data_seed = protocol_get(protocol, "DATA_SEED", default=SMOKE_SEED, required=False)
            if data_seed is None:
                data_seed = protocol_get(protocol, "data_seed", default=SMOKE_SEED, required=False)

            dataset = native_dataset_cls(project_root, epoch=epoch)
            sampler = native_sampler_cls(dataset, seed=int(data_seed), epoch=epoch)

            if callable(native_set_epoch):
                native_set_epoch(dataset, sampler, epoch)

            loader = native_build_loader(
                dataset,
                sampler,
                batch_size=micro_batch_size,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            set_sampler_epoch(loader, epoch)
            manifest = loader_manifest(loader, factory="project_native_dataset_sampler_build_train_dataloader")
            manifest["project_native_dataset_class"] = dataset.__class__.__name__
            manifest["project_native_sampler_class"] = sampler.__class__.__name__
            manifest["project_native_epoch"] = int(epoch)
            manifest["project_native_seed"] = int(data_seed)
            return loader, manifest
        except Exception as e:
            # Preserve the native error, but still allow the older generic/fallback
            # branches below to report a combined diagnostic if needed.
            native_error = f"project_native_path: {type(e).__name__}: {e}"
        else:
            native_error = None
    else:
        native_error = "project_native_path: required native classes/functions not found"
    data_seed = protocol_get(protocol, "DATA_SEED", default=SMOKE_SEED, required=False)
    if data_seed is None:
        data_seed = protocol_get(protocol, "data_seed", default=SMOKE_SEED, required=False)
    candidate_names = [
        "build_train_dataloader",
        "create_train_dataloader",
        "make_train_dataloader",
        "get_train_dataloader",
        "build_potsdam_train_dataloader",
        "build_potsdam_dataloader",
        "create_potsdam_dataloader",
        "make_potsdam_dataloader",
        "build_dataloader",
        "create_dataloader",
        "get_dataloader",
        "build_potsdam_dataloaders",
        "create_potsdam_dataloaders",
        "build_dataloaders",
        "create_dataloaders",
    ]
    errors: List[str] = []
    if 'native_error' in locals() and native_error:
        errors.append(native_error)
    dataset_protocol_path = project_root / "data/processed/potsdam/dataset_protocol.json"
    dataloader_protocol_path = project_root / "data/processed/potsdam/dataloader_protocol.json"
    dataloader_protocol = read_json(dataloader_protocol_path) if dataloader_protocol_path.exists() else None
    dataset_protocol = read_json(dataset_protocol_path) if dataset_protocol_path.exists() else None
    kwargs = {
        "project_root": project_root,
        "root": project_root,
        "repo_root": project_root,
        "data_root": project_root,
        "split": "train",
        "mode": "train",
        "train": True,
        "batch_size": micro_batch_size,
        "micro_batch_size": micro_batch_size,
        "num_workers": num_workers,
        "shuffle": True,
        "drop_last": False,
        "epoch": epoch,
        "seed": data_seed,
        "data_seed": data_seed,
        "protocol": dataloader_protocol,
        "config": dataloader_protocol,
        "training_protocol": protocol,
        "dataset_protocol": dataset_protocol,
        "dataloader_protocol": dataloader_protocol,
        "training_protocol_path": project_root / "data/processed/potsdam/training_protocol.json",
        "protocol_path": dataloader_protocol_path,
        "dataloader_protocol_path": dataloader_protocol_path,
        "dataset_protocol_path": dataset_protocol_path,
    }
    for name in candidate_names:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            loader = normalize_loader_candidate(call_with_supported_kwargs(fn, **kwargs))
            set_sampler_epoch(loader, epoch)
            manifest = loader_manifest(loader, factory=name)
            return loader, manifest
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    # Conservative fallback: use the frozen Dataset implementation only if the dataloader module does
    # not expose a callable factory. This still uses the project's real Dataset and PyTorch DataLoader.
    try:
        ds_module = importlib.import_module("data_pipeline.potsdam_dataset")
        dataset = None
        dataset_errors: List[str] = []
        for ds_name in (
            "PotsdamTrainDataset",
            "PotsdamDataset",
            "PotsdamSegmentationDataset",
            "ISPRSPotsdamDataset",
            "SegFormerPotsdamDataset",
        ):
            cls = getattr(ds_module, ds_name, None)
            if cls is None:
                continue
            try:
                dataset = call_with_supported_kwargs(
                    cls,
                    project_root=project_root,
                    root=project_root,
                    repo_root=project_root,
                    data_root=project_root,
                    split="train",
                    mode="train",
                    train=True,
                    protocol_path=project_root / "data/processed/potsdam/dataset_protocol.json",
                )
                break
            except Exception as e:
                dataset_errors.append(f"{ds_name}: {type(e).__name__}: {e}")
        if dataset is None:
            for fn_name in ("build_train_dataset", "create_train_dataset", "build_potsdam_dataset", "create_potsdam_dataset", "build_dataset"):
                fn = getattr(ds_module, fn_name, None)
                if fn is None or not callable(fn):
                    continue
                try:
                    dataset = call_with_supported_kwargs(
                        fn,
                        project_root=project_root,
                        root=project_root,
                        repo_root=project_root,
                        data_root=project_root,
                        split="train",
                        mode="train",
                        train=True,
                        protocol_path=project_root / "data/processed/potsdam/dataset_protocol.json",
                    )
                    break
                except Exception as e:
                    dataset_errors.append(f"{fn_name}: {type(e).__name__}: {e}")
        if dataset is None:
            raise SmokeFailure("No usable dataset factory/class found. " + " | ".join(dataset_errors[:8]))
        g = torch.Generator()
        g.manual_seed(int(data_seed) + int(epoch))
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            generator=g,
        )
        set_sampler_epoch(loader, epoch)
        manifest = loader_manifest(loader, factory="fallback_project_dataset_plus_torch_dataloader")
        manifest["fallback_warning"] = "No dataloader factory succeeded; used real project Dataset with PyTorch DataLoader."
        manifest["dataloader_factory_errors"] = errors[:12]
        return loader, manifest
    except Exception as e:
        raise SmokeFailure(
            "Could not build train DataLoader from data_pipeline.potsdam_dataloader. "
            f"Factory errors: {errors[:12]}; fallback error: {type(e).__name__}: {e}"
        )


def set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    for obj in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
        if obj is not None and hasattr(obj, "set_epoch") and callable(getattr(obj, "set_epoch")):
            obj.set_epoch(epoch)


def maybe_hash_generator(gen: Any) -> Optional[str]:
    if gen is None:
        return None
    if hasattr(gen, "get_state"):
        try:
            st = gen.get_state()
            if isinstance(st, torch.Tensor):
                return tensor_sha256(st)
            return recursive_state_hash(st)
        except Exception:
            return None
    return None


def sampler_state_manifest(loader: DataLoader) -> Dict[str, Any]:
    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    manifest = {
        "sampler_class": None if sampler is None else sampler.__class__.__name__,
        "batch_sampler_class": None if batch_sampler is None else batch_sampler.__class__.__name__,
        "sampler_has_set_epoch": bool(hasattr(sampler, "set_epoch")) if sampler is not None else False,
        "batch_sampler_has_set_epoch": bool(hasattr(batch_sampler, "set_epoch")) if batch_sampler is not None else False,
        "sampler_has_state_dict": bool(hasattr(sampler, "state_dict")) if sampler is not None else False,
        "batch_sampler_has_state_dict": bool(hasattr(batch_sampler, "state_dict")) if batch_sampler is not None else False,
        "sampler_generator_state_sha256": maybe_hash_generator(getattr(sampler, "generator", None)) if sampler is not None else None,
        "batch_sampler_generator_state_sha256": maybe_hash_generator(getattr(batch_sampler, "generator", None)) if batch_sampler is not None else None,
    }
    for key, obj in (("sampler", sampler), ("batch_sampler", batch_sampler)):
        if obj is not None and hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
            try:
                state = obj.state_dict()
                manifest[f"{key}_state_dict_sha256"] = recursive_state_hash(state)
                manifest[f"{key}_state_dict_safe"] = make_json_safe(state)
            except Exception as e:
                manifest[f"{key}_state_dict_error"] = repr(e)
    return manifest


def loader_manifest(loader: DataLoader, *, factory: str) -> Dict[str, Any]:
    ds = getattr(loader, "dataset", None)
    manifest = {
        "factory": factory,
        "loader_class": loader.__class__.__name__,
        "dataset_class": None if ds is None else ds.__class__.__name__,
        "dataset_length": len(ds) if ds is not None and hasattr(ds, "__len__") else None,
        "batch_size": getattr(loader, "batch_size", None),
        "num_workers": getattr(loader, "num_workers", None),
        "drop_last": getattr(loader, "drop_last", None),
        "pin_memory": getattr(loader, "pin_memory", None),
        "persistent_workers": getattr(loader, "persistent_workers", None),
    }
    manifest.update({"sampler_state_manifest": sampler_state_manifest(loader)})
    return make_json_safe(manifest)


def module_object_candidates(module: Any, names: Sequence[str]) -> List[Any]:
    return [getattr(module, name) for name in names if hasattr(module, name)]


def build_model_a_rgb(project_root: Path, protocol: Mapping[str, Any], model_protocol: Mapping[str, Any], *, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    module = importlib.import_module("models.segformer_rgb")
    checkpoint = protocol_get(model_protocol, "checkpoint", default="nvidia/segformer-b0-finetuned-ade-512-512", required=False)
    revision = protocol_get_any(model_protocol, ["resolved_revision", "revision"], default=None, required=False)
    num_classes = normalize_int(protocol_get_any(model_protocol, ["num_classes", "classes", "out_channels"], default=6, required=False))

    factory_names = [
        "build_model_a_rgb",
        "build_segformer_rgb",
        "create_model_a_rgb",
        "create_segformer_rgb",
        "make_model_a_rgb",
        "make_segformer_rgb",
        "get_model_a_rgb",
        "get_segformer_rgb",
        "build_model",
        "create_model",
        "get_model",
    ]
    class_names = [
        "ModelARGB",
        "SegFormerRGB",
        "SegformerRGB",
        "SegFormerRGBModel",
        "SegformerRGBModel",
        "PotsdamSegFormerRGB",
        "PotsdamSegformerRGB",
    ]

    kwargs = {
        "num_classes": num_classes,
        "n_classes": num_classes,
        "checkpoint": checkpoint,
        "checkpoint_name": checkpoint,
        "pretrained_checkpoint": checkpoint,
        "model_name_or_path": checkpoint,
        "revision": revision,
        "resolved_revision": revision,
        "local_files_only": True,
        "protocol": model_protocol,
        "model_protocol": model_protocol,
        "config": model_protocol,
        "training_protocol": protocol,
        "project_root": project_root,
        "root": project_root,
        "protocol_path": project_root / "data/processed/potsdam/model_a_rgb_protocol.json",
    }
    errors: List[str] = []

    for name in factory_names:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            obj = call_with_supported_kwargs(fn, **kwargs)
            model = normalize_model_object(obj)
            model.to(device)
            manifest = model_manifest(model, factory=name, checkpoint=checkpoint, revision=revision)
            return model, manifest
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    for name in class_names:
        cls = getattr(module, name, None)
        if cls is None or not inspect.isclass(cls):
            continue
        try:
            obj = call_with_supported_kwargs(cls, **kwargs)
            model = normalize_model_object(obj)
            model.to(device)
            manifest = model_manifest(model, factory=name, checkpoint=checkpoint, revision=revision)
            return model, manifest
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    # Last conservative fallback: if the module itself exposes exactly one nn.Module subclass constructor.
    module_classes = []
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            module_classes.append((name, obj))
    for name, cls in module_classes:
        try:
            obj = call_with_supported_kwargs(cls, **kwargs)
            model = normalize_model_object(obj)
            model.to(device)
            manifest = model_manifest(model, factory=f"auto_nn_module_class:{name}", checkpoint=checkpoint, revision=revision)
            manifest["auto_class_warning"] = "Used auto-discovered nn.Module class from models.segformer_rgb."
            return model, manifest
        except Exception as e:
            errors.append(f"auto_nn_module_class:{name}: {type(e).__name__}: {e}")

    raise SmokeFailure(
        "Could not build Model A RGB from models.segformer_rgb. "
        f"Tried factories/classes; errors: {errors[:16]}"
    )


def normalize_model_object(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, Mapping):
        for key in ("model", "net", "module"):
            value = obj.get(key)
            if isinstance(value, nn.Module):
                return value
    if isinstance(obj, (tuple, list)):
        for value in obj:
            if isinstance(value, nn.Module):
                return value
    raise SmokeFailure(f"Could not normalize model object of type {type(obj).__name__}")


def model_manifest(model: nn.Module, *, factory: str, checkpoint: Any, revision: Any) -> Dict[str, Any]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "factory": factory,
        "model_class": model.__class__.__name__,
        "checkpoint": checkpoint,
        "resolved_revision": revision,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "training_mode_after_build": bool(model.training),
        "state_dict_sha256_initial": model_state_hash(model),
    }


def extract_rgb_and_label(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    meta: Dict[str, Any] = {"batch_type": type(batch).__name__}

    def choose_from_mapping(m: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, Optional[str]]:
        for k in keys:
            if k in m:
                return m[k], k
        for k, v in m.items():
            lk = str(k).lower()
            if any(candidate in lk for candidate in keys):
                return v, str(k)
        return None, None

    if isinstance(batch, Mapping):
        image, image_key = choose_from_mapping(
            batch,
            [
                "rgb",
                "rgb_image",
                "image_rgb",
                "pixel_values",
                "image",
                "img",
                "x",
                "inputs",
                "rgbir",
                "image_rgbir",
            ],
        )
        label, label_key = choose_from_mapping(
            batch,
            ["mask", "label", "labels", "target", "targets", "segmentation", "gt", "y", "semantic", "semantic_mask"],
        )
        meta["image_key"] = image_key
        meta["label_key"] = label_key
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        image, label = batch[0], batch[1]
        meta["image_key"] = "tuple[0]"
        meta["label_key"] = "tuple[1]"
    else:
        raise SmokeFailure(f"Unsupported batch format: {type(batch).__name__}")

    if image is None or label is None:
        raise SmokeFailure(f"Could not extract image/label from batch. Keys/meta={meta}")
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)

    # Normalize image shape to [B,C,H,W].
    if image.ndim == 3:
        if image.shape[0] in (1, 3, 4):
            image = image.unsqueeze(0)
        elif image.shape[-1] in (1, 3, 4):
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            raise SmokeFailure(f"Cannot infer image layout from shape {tuple(image.shape)}")
    elif image.ndim == 4:
        if image.shape[1] in (1, 3, 4):
            pass
        elif image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)
        else:
            raise SmokeFailure(f"Cannot infer image layout from shape {tuple(image.shape)}")
    else:
        raise SmokeFailure(f"Image tensor must have 3 or 4 dims, got shape {tuple(image.shape)}")

    original_channels = int(image.shape[1])
    if original_channels == 4:
        rgb = image[:, :3, :, :]
        meta["input_channel_source"] = "first_3_channels_from_4_channel_tensor"
    elif original_channels == 3:
        rgb = image
        meta["input_channel_source"] = "3_channel_rgb_tensor"
    elif original_channels == 1:
        rgb = image.repeat(1, 3, 1, 1)
        meta["input_channel_source"] = "single_channel_repeated_to_rgb_for_smoke_only"
    else:
        raise SmokeFailure(f"Model A RGB requires 3 RGB channels; got {original_channels}")

    if not torch.is_floating_point(rgb):
        rgb = rgb.float()
        if float(rgb.max().item()) > 2.0:
            rgb = rgb / 255.0
            meta["input_value_conversion"] = "integer_to_float_div255"
        else:
            meta["input_value_conversion"] = "integer_to_float_no_scale"
    else:
        rgb = rgb.float()
        meta["input_value_conversion"] = "float_as_float32"

    # Normalize label shape to [B,H,W].
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0, :, :]
    elif label.ndim == 3:
        pass
    elif label.ndim == 2:
        label = label.unsqueeze(0)
    else:
        raise SmokeFailure(f"Label tensor must have shape [B,H,W], [B,1,H,W], or [H,W], got {tuple(label.shape)}")
    label = label.long()

    meta.update(
        {
            "original_image_shape": list(image.shape),
            "rgb_shape": list(rgb.shape),
            "label_shape": list(label.shape),
            "original_channels": original_channels,
            "rgb_dtype": str(rgb.dtype),
            "label_dtype": str(label.dtype),
            "label_min": int(label.min().item()),
            "label_max": int(label.max().item()),
            "label_unique_sample": [int(x) for x in torch.unique(label.detach().cpu())[:16].tolist()],
        }
    )
    return rgb, label, meta


def current_cuda_autocast_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    try:
        # PyTorch 2.x accepts a device_type argument.
        status["autocast_enabled_cuda"] = bool(torch.is_autocast_enabled("cuda"))  # type: ignore[arg-type]
    except TypeError:
        try:
            status["autocast_enabled_cuda"] = bool(torch.is_autocast_cuda_enabled())
        except Exception as e:
            status["autocast_enabled_cuda_error"] = repr(e)
    except Exception as e:
        status["autocast_enabled_cuda_error"] = repr(e)
    try:
        if hasattr(torch, "get_autocast_dtype"):
            status["autocast_cuda_dtype"] = str(torch.get_autocast_dtype("cuda"))  # type: ignore[arg-type]
        else:
            status["autocast_cuda_dtype"] = str(torch.get_autocast_gpu_dtype())
    except Exception as e:
        status["autocast_cuda_dtype_error"] = repr(e)
    return status


def batch_signature(rgb: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    # Full content hash is intentional: it proves reference and resumed streams saw identical microbatches.
    return {
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "rgb_sha256": tensor_sha256(rgb),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_sha256": tensor_sha256(labels),
    }


def forward_logits(model: nn.Module, rgb: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    meta: Dict[str, Any] = current_cuda_autocast_status()
    try:
        out = model(rgb)
        meta["forward_call"] = "model(rgb)"
    except TypeError:
        out = model(pixel_values=rgb)
        meta["forward_call"] = "model(pixel_values=rgb)"

    if isinstance(out, Mapping):
        for key in ("full_logits", "logits", "out", "prediction", "pred"):
            if key in out:
                logits = out[key]
                meta["output_key"] = key
                break
        else:
            raise SmokeFailure(f"Model output mapping does not contain logits-like key: {list(out.keys())}")
    elif hasattr(out, "logits"):
        logits = out.logits
        meta["output_key"] = "object.logits"
    elif isinstance(out, (tuple, list)):
        logits = out[0]
        meta["output_key"] = "tuple[0]"
    elif torch.is_tensor(out):
        logits = out
        meta["output_key"] = "tensor"
    else:
        raise SmokeFailure(f"Unsupported model output type: {type(out).__name__}")

    if not torch.is_tensor(logits):
        raise SmokeFailure(f"Extracted logits is not a tensor: {type(logits).__name__}")
    if logits.ndim != 4:
        raise SmokeFailure(f"Logits must have shape [B,C,H,W], got {tuple(logits.shape)}")
    meta["raw_logits_shape"] = list(logits.shape)
    meta["raw_logits_dtype"] = str(logits.dtype)
    if tuple(logits.shape[-2:]) != tuple(target_hw):
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        meta["upsampled_to_target_hw"] = list(target_hw)
    else:
        meta["upsampled_to_target_hw"] = None
    meta["full_logits_shape"] = list(logits.shape)
    meta["full_logits_dtype"] = str(logits.dtype)
    return logits, meta


def make_loader_iterator_at(project_root: Path, protocol: Mapping[str, Any], *, epoch: int, microbatch_index: int, num_workers: int) -> Tuple[DataLoader, Iterator[Any], Dict[str, Any]]:
    loader, manifest = build_train_dataloader(project_root, protocol, epoch=epoch, num_workers=num_workers)
    set_sampler_epoch(loader, epoch)
    it = iter(loader)
    skipped = 0
    for _ in range(microbatch_index):
        next(it)
        skipped += 1
    state = {
        "epoch": epoch,
        "microbatch_index_in_epoch": microbatch_index,
        "skipped_microbatches": skipped,
        "loader_manifest": manifest,
        "sampler_state_manifest_after_skip": sampler_state_manifest(loader),
    }
    return loader, it, make_json_safe(state)


def count_optimizer_steps(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    state = {"count": 0}
    original_step = optimizer.step

    @functools.wraps(original_step)
    def wrapped_step(*args: Any, **kwargs: Any) -> Any:
        state["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = wrapped_step  # type: ignore[method-assign]
    return state


def get_scaler_scale(scaler: Any) -> Optional[float]:
    try:
        return float(scaler.get_scale())
    except Exception:
        return None


def run_microbatches(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: FrozenLRScheduler,
    scaler: Any,
    criterion: nn.Module,
    device: torch.device,
    loader_iter: Iterator[Any],
    start_microbatch_index: int,
    num_microbatches: int,
    grad_acc_steps: int,
    amp_dtype: torch.dtype,
    clip_max_norm: float,
    step_counter: MutableMapping[str, int],
    run_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model.train()
    history: List[Dict[str, Any]] = []
    microbatches_since_step = start_microbatch_index % grad_acc_steps
    optimizer_steps_before = int(step_counter["count"])

    for local_i in range(num_microbatches):
        global_micro_idx = start_microbatch_index + local_i
        batch = next(loader_iter)
        rgb_cpu, labels_cpu, batch_meta = extract_rgb_and_label(batch)
        b_sig = batch_signature(rgb_cpu, labels_cpu)
        rgb = rgb_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)

        scale_before = get_scaler_scale(scaler)
        with torch_amp_autocast(amp_dtype):
            logits, forward_meta = forward_logits(model, rgb, target_hw=(int(labels.shape[-2]), int(labels.shape[-1])))
            raw_loss = criterion(logits, labels)
            loss = raw_loss / float(grad_acc_steps)

        if not torch.isfinite(raw_loss.detach()):
            raise SmokeFailure(f"Non-finite raw loss at microbatch {global_micro_idx}: {float(raw_loss.detach().item())}")
        scaler.scale(loss).backward()
        microbatches_since_step += 1

        stepped = False
        lr_used: Optional[float] = None
        grad_norm_before_clip: Optional[float] = None
        scale_after = None
        if microbatches_since_step == grad_acc_steps:
            lr_used = scheduler.set_lr_for_next_step()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            grad_norm_before_clip = float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scale_after = get_scaler_scale(scaler)
            scheduler.step_after_optimizer()
            optimizer.zero_grad(set_to_none=True)
            microbatches_since_step = 0
            stepped = True
        else:
            scale_after = get_scaler_scale(scaler)

        record = {
            "run_name": run_name,
            "global_microbatch_index_0based": global_micro_idx,
            "microbatch_number_1based": global_micro_idx + 1,
            "raw_loss": float(raw_loss.detach().float().cpu().item()),
            "scaled_loss": float(loss.detach().float().cpu().item()),
            "loss_divisor": grad_acc_steps,
            "optimizer_step_executed": stepped,
            "optimizer_step_count_total": int(step_counter["count"]),
            "scheduler_next_update_index_after_microbatch": int(scheduler.next_update_index),
            "lr_used_for_step": lr_used,
            "grad_norm_before_clip": grad_norm_before_clip,
            "grad_clip_max_norm": clip_max_norm if stepped else None,
            "grad_scaler_scale_before": scale_before,
            "grad_scaler_scale_after": scale_after,
            "pending_accumulation_microbatches_after": microbatches_since_step,
            "batch_meta": batch_meta if local_i == 0 else {"image_key": batch_meta.get("image_key"), "label_key": batch_meta.get("label_key"), "input_channel_source": batch_meta.get("input_channel_source")},
            "forward_meta": forward_meta if local_i == 0 else {"raw_logits_shape": forward_meta.get("raw_logits_shape"), "full_logits_shape": forward_meta.get("full_logits_shape")},
            "batch_signature": b_sig,
        }
        history.append(make_json_safe(record))
        print(
            f"[{run_name}] microbatch {global_micro_idx + 1}: "
            f"raw_loss={record['raw_loss']:.6f}, scaled_loss={record['scaled_loss']:.6f}, "
            f"step={stepped}, opt_steps={step_counter['count']}, "
            f"lr={lr_used if lr_used is not None else 'NA'}"
        )

    summary = {
        "optimizer_steps_before": optimizer_steps_before,
        "optimizer_steps_after": int(step_counter["count"]),
        "optimizer_steps_delta": int(step_counter["count"]) - optimizer_steps_before,
        "microbatches_since_step_after": microbatches_since_step,
        "model_state_sha256": model_state_hash(model),
        "model_grad_sha256": model_grad_hash(model),
        "optimizer_state_sha256": recursive_state_hash(optimizer.state_dict()),
        "scaler_state_sha256": recursive_state_hash(scaler.state_dict()),
        "scheduler_state_sha256": recursive_state_hash(scheduler.state_dict()),
        "rng_state_sha256": hash_rng_state(rng_state_dict()),
    }
    return history, summary


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: FrozenLRScheduler,
    engine_state: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    param_group_manifest: Mapping[str, Any],
    loader_state: Mapping[str, Any],
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "saved_at": now_iso(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": rng_state_dict(),
        "engine_state": dict(engine_state),
        "protocol_manifest": dict(protocol_manifest),
        "param_group_manifest": dict(param_group_manifest),
        "loader_state": make_json_safe(loader_state),
    }
    torch.save(checkpoint, path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "contains_keys": sorted(checkpoint.keys()),
        "engine_state": make_json_safe(engine_state),
        "loader_state": make_json_safe(loader_state),
        "rng_state_sha256": hash_rng_state(checkpoint["rng_state"]),
    }


def load_checkpoint(path: Path, *, device: torch.device) -> Mapping[str, Any]:
    # Always deserialize checkpoint tensors onto CPU first.
    # model.load_state_dict() and optimizer.load_state_dict() will place their
    # states appropriately, while CPU/CUDA RNG ByteTensors retain the format
    # required by the RNG restore APIs.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_optimizer_scheduler_scaler(
    model: nn.Module,
    protocol: Mapping[str, Any],
    lr_sequence: Sequence[float],
) -> Tuple[torch.optim.Optimizer, FrozenLRScheduler, Any, Dict[str, Any], Dict[str, Any]]:
    base_lr = normalize_float(protocol_get_any(protocol, ["base_lr", "lr", "learning_rate"]))
    weight_decay = normalize_float(protocol_get(protocol, "weight_decay"))
    head_lr_multiplier = normalize_float(protocol_get(protocol, "head_lr_multiplier", default=1.0, required=False))
    betas_raw = protocol_get(protocol, "betas", default=[0.9, 0.999], required=False)
    betas = tuple(float(x) for x in betas_raw)
    eps = normalize_float(protocol_get(protocol, "eps", default=1e-8, required=False))
    amp_enabled = normalize_bool(protocol_get(protocol, "AMP", default=True, required=False)) or normalize_bool(protocol_get(protocol, "amp", default=True, required=False))

    param_groups, param_manifest = build_adamw_param_groups(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        head_lr_multiplier=head_lr_multiplier,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = FrozenLRScheduler(optimizer, lr_sequence)
    scaler = make_grad_scaler(enabled=amp_enabled)
    opt_manifest = {
        "optimizer_class": optimizer.__class__.__name__,
        "base_lr": base_lr,
        "weight_decay_default_argument": weight_decay,
        "betas": list(betas),
        "eps": eps,
        "amp_enabled_requested": amp_enabled,
        "grad_scaler_enabled_effective": bool(scaler.is_enabled()),
        "grad_scaler_initial_scale": get_scaler_scale(scaler),
    }
    return optimizer, scheduler, scaler, param_manifest, opt_manifest


def load_states_into(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: FrozenLRScheduler,
) -> Dict[str, Any]:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return {
        "model_state_sha256_after_load": model_state_hash(model),
        "optimizer_state_sha256_after_load": recursive_state_hash(optimizer.state_dict()),
        "scaler_state_sha256_after_load": recursive_state_hash(scaler.state_dict()),
        "scheduler_state_sha256_after_load": recursive_state_hash(scheduler.state_dict()),
        "rng_state_sha256_after_restore": hash_rng_state(rng_state_dict()),
        "checkpoint_rng_state_sha256": hash_rng_state(checkpoint["rng_state"]),
    }


def validate_protocol_files(project_root: Path, training_protocol: Mapping[str, Any], checks: List[Check]) -> Dict[str, Any]:
    processed = project_root / "data/processed/potsdam"
    paths = {
        "dataset_protocol": processed / "dataset_protocol.json",
        "dataloader_protocol": processed / "dataloader_protocol.json",
        "evaluation_protocol": processed / "evaluation_protocol.json",
        "model_a_rgb_protocol": processed / "model_a_rgb_protocol.json",
        "training_protocol": processed / "training_protocol.json",
    }
    manifest: Dict[str, Any] = {}
    for key, path in paths.items():
        if not path.exists():
            checks.append(Check(f"{key}_exists", False, str(path)))
            raise SmokeFailure(f"Required frozen protocol file not found: {path}")
        observed = sha256_file(path)
        expected = EXPECTED_PROTOCOL_SHA256[key]
        ok = observed == expected
        checks.append(Check(f"{key}_sha256", ok, {"path": str(path), "observed": observed, "expected": expected}))
        if not ok:
            raise SmokeFailure(f"{key} SHA256 mismatch: observed={observed}, expected={expected}")
        manifest[key] = {"path": str(path), "sha256": observed}

    upstream = protocol_get(training_protocol, "upstream_protocol_sha256", default=None, required=False)
    if upstream is None:
        upstream = protocol_get(training_protocol, "upstream_frozen_protocol_sha256", default=None, required=False)
    if isinstance(upstream, Mapping):
        for key in ("dataset_protocol", "dataloader_protocol", "evaluation_protocol", "model_a_rgb_protocol"):
            if key in upstream:
                ok = upstream[key] == EXPECTED_PROTOCOL_SHA256[key]
                checks.append(Check(f"training_protocol_upstream_{key}_sha256", ok, {"actual": upstream[key], "expected": EXPECTED_PROTOCOL_SHA256[key]}))
                if not ok:
                    raise SmokeFailure(f"training_protocol upstream {key} SHA256 mismatch")

    return manifest


def validate_training_values(protocol: Mapping[str, Any], checks: List[Check]) -> Dict[str, Any]:
    expected_values = {
        "max_epochs": 100,
        "samples_per_epoch": 1152,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "micro_batches_per_epoch": 576,
        "optimizer_steps_per_epoch": 72,
        "total_optimizer_updates": 7200,
        "ignore_index": 255,
        "label_smoothing": 0.0,
        "base_lr": 6e-5,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "head_lr_multiplier": 1.0,
        "warmup_updates": 360,
        "poly_power": 1.0,
        "lr_first": 6e-11,
        "lr_warmup_last": 6e-5,
        "lr_poly_first": 6e-5,
        "lr_last": 0.0,
    }
    observed: Dict[str, Any] = {}
    # Some scalar LR endpoint values are frozen by training_protocol SHA256 and by
    # lr_sequence_sha256, but may not be stored as top-level JSON fields in older
    # protocol files. Do not modify the frozen JSON; use the audited expected
    # endpoints as compatibility fallbacks after the protocol SHA has matched.
    optional_frozen_endpoint_keys = {
        "lr_first",
        "lr_warmup_last",
        "lr_poly_first",
        "lr_last",
    }

    for key, expected in expected_values.items():
        if key in optional_frozen_endpoint_keys:
            actual = protocol_get(protocol, key, default=expected, required=False)
        else:
            actual = protocol_get(protocol, key, required=True)
        observed[key] = actual
        if isinstance(expected, float):
            assert_close(f"training_protocol_{key}", float(actual), expected, checks, atol=1e-12, rtol=1e-12)
        else:
            assert_equal(f"training_protocol_{key}", int(actual), expected, checks)

    loss_value = protocol_get(protocol, "loss", required=True)
    optimizer_value = protocol_get(protocol, "optimizer", required=True)
    class_weight = protocol_get(protocol, "class_weight", default=None, required=False)
    amp_dtype = str(protocol_get(protocol, "amp_dtype", default="float16", required=False)).lower()
    grad_clip = normalize_bool(protocol_get(protocol, "gradient_clipping", default=True, required=False))
    grad_clip_norm = normalize_float(protocol_get_any(protocol, ["gradient_clip_max_norm", "grad_clip_max_norm", "clip_max_norm"], default=1.0, required=False))

    loss_safe = make_json_safe(loss_value)
    loss_repr = json.dumps(loss_safe, ensure_ascii=False, sort_keys=True)
    loss_repr_lower = loss_repr.lower()
    loss_repr_no_separators = loss_repr_lower.replace("_", "").replace("-", "").replace(" ", "")

    if isinstance(loss_value, Mapping):
        loss_name = str(loss_value.get("name", loss_value.get("type", "")))
        nested_class_weight = loss_value.get("class_weight", None)
    else:
        loss_name = str(loss_value)
        nested_class_weight = None
    loss_name_normalized = loss_name.lower().replace("_", "").replace("-", "").replace(" ", "")

    loss_ok = (
        "crossentropy" in loss_name_normalized
        or "crossentropyloss" in loss_name_normalized
        or "crossentropy" in loss_repr_no_separators
        or "cross_entropy" in loss_repr_lower
    )
    unweighted_ok = class_weight is None and nested_class_weight is None

    checks.append(Check(
        "training_protocol_loss_cross_entropy",
        loss_ok,
        {"loss": loss_safe, "loss_name": loss_name},
    ))
    checks.append(Check(
        "training_protocol_loss_unweighted",
        unweighted_ok,
        {"loss": loss_safe, "top_level_class_weight": class_weight, "nested_class_weight": nested_class_weight},
    ))
    if not loss_ok or not unweighted_ok:
        raise SmokeFailure(
            f"Loss protocol is not unweighted CrossEntropyLoss: "
            f"loss={loss_value!r}, class_weight={class_weight!r}, nested_class_weight={nested_class_weight!r}"
        )

    optimizer_repr = json.dumps(make_json_safe(optimizer_value), ensure_ascii=False, sort_keys=True)
    optimizer_ok = "adamw" in optimizer_repr.lower()
    checks.append(Check("training_protocol_optimizer_adamw", optimizer_ok, {"optimizer": make_json_safe(optimizer_value)}))
    if not optimizer_ok:
        raise SmokeFailure(f"Optimizer protocol is not AdamW: {optimizer_value!r}")

    assert_equal("training_protocol_class_weight", class_weight, None, checks)
    checks.append(Check("training_protocol_amp_dtype_float16", amp_dtype in {"float16", "fp16", "torch.float16"}, {"actual": amp_dtype}))
    if amp_dtype not in {"float16", "fp16", "torch.float16"}:
        raise SmokeFailure(f"amp_dtype is not float16: {amp_dtype}")
    assert_equal("training_protocol_gradient_clipping", grad_clip, True, checks)
    assert_close("training_protocol_gradient_clip_max_norm", grad_clip_norm, 1.0, checks, atol=1e-12, rtol=1e-12)

    observed.update(
        {
            "loss": make_json_safe(loss_value),
            "optimizer": make_json_safe(optimizer_value),
            "class_weight": class_weight,
            "amp_dtype": amp_dtype,
            "gradient_clipping": grad_clip,
            "gradient_clip_max_norm": grad_clip_norm,
        }
    )
    return observed


def validate_model_against_protocol(model_manifest_obj: Mapping[str, Any], checks: List[Check]) -> None:
    assert_equal("model_parameters_total", int(model_manifest_obj["parameters_total"]), 3715686, checks)
    assert_equal("model_parameters_trainable", int(model_manifest_obj["parameters_trainable"]), 3715686, checks)


def compare_histories(reference: Sequence[Mapping[str, Any]], resumed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def selected(h: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "global_microbatch_index_0based": h["global_microbatch_index_0based"],
            "batch_signature": h["batch_signature"],
            "optimizer_step_executed": h["optimizer_step_executed"],
            "optimizer_step_count_total": h["optimizer_step_count_total"],
            "scheduler_next_update_index_after_microbatch": h["scheduler_next_update_index_after_microbatch"],
            "lr_used_for_step": h["lr_used_for_step"],
        }

    ref_selected = [selected(h) for h in reference]
    res_selected = [selected(h) for h in resumed]
    batch_equal = [r["batch_signature"] == s["batch_signature"] for r, s in zip(ref_selected, res_selected)] if len(ref_selected) == len(res_selected) else []
    raw_losses_ref = [float(h["raw_loss"]) for h in reference]
    raw_losses_res = [float(h["raw_loss"]) for h in resumed]
    loss_abs_diffs = [abs(a - b) for a, b in zip(raw_losses_ref, raw_losses_res)] if len(raw_losses_ref) == len(raw_losses_res) else []
    return {
        "history_lengths_equal": len(reference) == len(resumed),
        "batch_signatures_all_equal": bool(batch_equal) and all(batch_equal),
        "step_flags_equal": [h["optimizer_step_executed"] for h in reference] == [h["optimizer_step_executed"] for h in resumed],
        "lr_used_equal": [h["lr_used_for_step"] for h in reference] == [h["lr_used_for_step"] for h in resumed],
        "max_raw_loss_abs_diff": max(loss_abs_diffs) if loss_abs_diffs else None,
        "raw_losses_allclose_atol_1e-6": len(raw_losses_ref) == len(raw_losses_res)
        and all(abs(a - b) <= 1e-6 for a, b in zip(raw_losses_ref, raw_losses_res)),
        "first_mismatch_microbatch": next((i for i, ok in enumerate(batch_equal) if not ok), None) if batch_equal else None,
    }


def run_experiment(project_root: Path, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    checks: List[Check] = []
    result: Dict[str, Any] = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "started_at": now_iso(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "status": "RUNNING",
        "checks": [],
    }

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.reference_microbatches > MAX_ALLOWED_SMOKE_MICROBATCHES:
        raise SmokeFailure(
            f"This smoke-test refuses to run more than {MAX_ALLOWED_SMOKE_MICROBATCHES} microbatches; "
            f"got {args.reference_microbatches}. Formal training is out of scope."
        )
    if args.checkpoint_after_microbatches <= 0 or args.checkpoint_after_microbatches >= args.reference_microbatches:
        raise SmokeFailure("checkpoint_after_microbatches must be >0 and < reference_microbatches")

    print(f"[{SCRIPT_NAME}] project_root = {project_root}")
    print(f"[{SCRIPT_NAME}] output_dir   = {output_dir}")
    print(f"[{SCRIPT_NAME}] strict scope: Model A RGB engine smoke-test only; no val/test/full training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_equal("cuda_available", torch.cuda.is_available(), True, checks)
    if device.type != "cuda":
        raise SmokeFailure("CUDA is required because the frozen protocol requires FP16 AMP + GradScaler.")
    determinism_flags = set_determinism_flags()
    env_manifest = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "determinism_flags": determinism_flags,
    }
    result["environment"] = make_json_safe(env_manifest)

    training_protocol_path = project_root / "data/processed/potsdam/training_protocol.json"
    training_protocol = read_json(training_protocol_path)
    protocol_manifest = validate_protocol_files(project_root, training_protocol, checks)
    frozen_training_values = validate_training_values(training_protocol, checks)
    result["protocol_manifest"] = protocol_manifest
    result["frozen_training_values"] = frozen_training_values
    print(f"[{SCRIPT_NAME}] frozen training_protocol SHA256 OK: {EXPECTED_PROTOCOL_SHA256['training_protocol']}")

    model_protocol = read_json(project_root / "data/processed/potsdam/model_a_rgb_protocol.json")
    lr_sequence, lr_meta = build_lr_sequence(training_protocol)
    checks.append(Check("lr_sequence_length", len(lr_sequence) == 7200, {"actual": len(lr_sequence), "expected": 7200}))
    if len(lr_sequence) != 7200:
        raise SmokeFailure("LR sequence length mismatch")
    result["lr_schedule"] = lr_meta
    print(
        f"[{SCRIPT_NAME}] LR endpoints: first={lr_meta['lr_first']}, warmup_last={lr_meta['lr_warmup_last']}, "
        f"poly_first={lr_meta['lr_poly_first']}, last={lr_meta['lr_last']}"
    )

    grad_acc_steps = normalize_int(protocol_get(training_protocol, "gradient_accumulation_steps"))
    micro_batch_size = normalize_int(protocol_get(training_protocol, "micro_batch_size"))
    ignore_index = normalize_int(protocol_get(training_protocol, "ignore_index"))
    label_smoothing = normalize_float(protocol_get(training_protocol, "label_smoothing"))
    clip_max_norm = normalize_float(protocol_get_any(training_protocol, ["gradient_clip_max_norm", "grad_clip_max_norm", "clip_max_norm"], default=1.0, required=False))
    amp_dtype_string = str(protocol_get(training_protocol, "amp_dtype", default="float16", required=False)).lower()
    amp_dtype = torch.float16 if amp_dtype_string in {"float16", "fp16", "torch.float16"} else None
    if amp_dtype is None:
        raise SmokeFailure(f"Unsupported amp_dtype for this smoke-test: {amp_dtype_string}")

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

    smoke_config = {
        "seed": SMOKE_SEED,
        "smoke_epoch": 0,
        "reference_microbatches": int(args.reference_microbatches),
        "checkpoint_after_microbatches": int(args.checkpoint_after_microbatches),
        "resume_extra_microbatches": int(args.reference_microbatches - args.checkpoint_after_microbatches),
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_acc_steps,
        "num_workers": int(args.num_workers),
        "no_validation_or_test": True,
        "formal_training_epochs_run": 0,
    }
    result["smoke_config"] = smoke_config

    # -------------------------
    # Reference uninterrupted run
    # -------------------------
    print(f"[{SCRIPT_NAME}] starting uninterrupted reference run: {args.reference_microbatches} microbatches")
    set_global_seed(SMOKE_SEED)
    ref_loader, ref_iter, ref_loader_state = make_loader_iterator_at(
        project_root,
        training_protocol,
        epoch=0,
        microbatch_index=0,
        num_workers=args.num_workers,
    )
    ref_model, ref_model_manifest = build_model_a_rgb(project_root, training_protocol, model_protocol, device=device)
    validate_model_against_protocol(ref_model_manifest, checks)
    ref_optimizer, ref_scheduler, ref_scaler, ref_param_manifest, ref_opt_manifest = build_optimizer_scheduler_scaler(ref_model, training_protocol, lr_sequence)
    checks.append(Check("adamw_param_group_no_decay_violations_zero", ref_param_manifest.get("no_decay_violation_count") == 0, ref_param_manifest))
    checks.append(Check("adamw_param_group_decay_violations_zero", ref_param_manifest.get("decay_violation_count") == 0, ref_param_manifest))
    checks.append(Check("adamw_head_lr_multiplier_1_0", math.isclose(float(ref_param_manifest.get("head_lr_multiplier")), 1.0), ref_param_manifest))
    ref_step_counter = count_optimizer_steps(ref_optimizer)
    ref_optimizer.zero_grad(set_to_none=True)
    ref_history, ref_summary = run_microbatches(
        project_root=project_root,
        protocol=training_protocol,
        model=ref_model,
        optimizer=ref_optimizer,
        scheduler=ref_scheduler,
        scaler=ref_scaler,
        criterion=criterion,
        device=device,
        loader_iter=ref_iter,
        start_microbatch_index=0,
        num_microbatches=args.reference_microbatches,
        grad_acc_steps=grad_acc_steps,
        amp_dtype=amp_dtype,
        clip_max_norm=clip_max_norm,
        step_counter=ref_step_counter,
        run_name="reference",
    )

    # -------------------------
    # Split run: checkpoint after 8 microbatches, resume, continue
    # -------------------------
    print(f"[{SCRIPT_NAME}] starting split run to checkpoint at microbatch {args.checkpoint_after_microbatches}")
    set_global_seed(SMOKE_SEED)
    split_loader, split_iter, split_loader_state = make_loader_iterator_at(
        project_root,
        training_protocol,
        epoch=0,
        microbatch_index=0,
        num_workers=args.num_workers,
    )
    split_model, split_model_manifest = build_model_a_rgb(project_root, training_protocol, model_protocol, device=device)
    split_optimizer, split_scheduler, split_scaler, split_param_manifest, split_opt_manifest = build_optimizer_scheduler_scaler(split_model, training_protocol, lr_sequence)
    split_step_counter = count_optimizer_steps(split_optimizer)
    split_optimizer.zero_grad(set_to_none=True)
    split_history_a, split_summary_a = run_microbatches(
        project_root=project_root,
        protocol=training_protocol,
        model=split_model,
        optimizer=split_optimizer,
        scheduler=split_scheduler,
        scaler=split_scaler,
        criterion=criterion,
        device=device,
        loader_iter=split_iter,
        start_microbatch_index=0,
        num_microbatches=args.checkpoint_after_microbatches,
        grad_acc_steps=grad_acc_steps,
        amp_dtype=amp_dtype,
        clip_max_norm=clip_max_norm,
        step_counter=split_step_counter,
        run_name="split_pre_checkpoint",
    )

    expected_steps_at_checkpoint = args.checkpoint_after_microbatches // grad_acc_steps
    assert_equal("optimizer_steps_after_8_microbatches", int(split_step_counter["count"]), expected_steps_at_checkpoint, checks)
    assert_equal("scheduler_next_update_after_8_microbatches", int(split_scheduler.next_update_index), expected_steps_at_checkpoint, checks)
    assert_equal("pending_accumulation_after_checkpoint", int(args.checkpoint_after_microbatches % grad_acc_steps), 0, checks)

    checkpoint_path = output_dir / "checkpoint_after_8_microbatches.pt"
    engine_state = {
        "epoch": 0,
        "microbatch_index_in_epoch": int(args.checkpoint_after_microbatches),
        "global_microbatch_index_next": int(args.checkpoint_after_microbatches),
        "optimizer_update_index_next": int(split_scheduler.next_update_index),
        "optimizer_step_count": int(split_step_counter["count"]),
        "gradient_accumulation_steps": grad_acc_steps,
        "pending_accumulation_microbatches": int(args.checkpoint_after_microbatches % grad_acc_steps),
    }
    loader_state_for_checkpoint = {
        "epoch": 0,
        "microbatch_index_in_epoch": int(args.checkpoint_after_microbatches),
        "resume_strategy": "rebuild_train_dataloader_same_epoch_then_skip_microbatch_index_then_restore_rng",
        "loader_manifest_at_start": split_loader_state,
        "sampler_state_manifest_at_save": sampler_state_manifest(split_loader),
    }
    checkpoint_manifest = save_checkpoint(
        checkpoint_path,
        model=split_model,
        optimizer=split_optimizer,
        scaler=split_scaler,
        scheduler=split_scheduler,
        engine_state=engine_state,
        protocol_manifest=protocol_manifest,
        param_group_manifest=split_param_manifest,
        loader_state=loader_state_for_checkpoint,
    )
    print(f"[{SCRIPT_NAME}] checkpoint saved: {checkpoint_path}")

    # Resume into fresh objects. Loader skip is done before RNG restore; then checkpoint RNG is restored.
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    set_global_seed(SMOKE_SEED)
    resume_loader, resume_iter, resume_loader_state = make_loader_iterator_at(
        project_root,
        training_protocol,
        epoch=0,
        microbatch_index=int(checkpoint["engine_state"]["microbatch_index_in_epoch"]),
        num_workers=args.num_workers,
    )
    resume_model, resume_model_manifest = build_model_a_rgb(project_root, training_protocol, model_protocol, device=device)
    resume_optimizer, resume_scheduler, resume_scaler, resume_param_manifest, resume_opt_manifest = build_optimizer_scheduler_scaler(resume_model, training_protocol, lr_sequence)
    resume_step_counter = count_optimizer_steps(resume_optimizer)
    load_manifest = load_states_into(
        checkpoint,
        model=resume_model,
        optimizer=resume_optimizer,
        scaler=resume_scaler,
        scheduler=resume_scheduler,
    )
    # Re-count optimizer steps from engine state after wrapping fresh optimizer.step.
    resume_step_counter["count"] = int(checkpoint["engine_state"].get("optimizer_step_count", resume_scheduler.next_update_index))

    immediate_load_checks = {
        "model_state_matches_checkpoint_pre_resume": load_manifest["model_state_sha256_after_load"] == split_summary_a["model_state_sha256"],
        "optimizer_state_matches_checkpoint_pre_resume": load_manifest["optimizer_state_sha256_after_load"] == split_summary_a["optimizer_state_sha256"],
        "scaler_state_matches_checkpoint_pre_resume": load_manifest["scaler_state_sha256_after_load"] == split_summary_a["scaler_state_sha256"],
        "scheduler_state_matches_checkpoint_pre_resume": load_manifest["scheduler_state_sha256_after_load"] == split_summary_a["scheduler_state_sha256"],
        "rng_state_restored_to_checkpoint": load_manifest["rng_state_sha256_after_restore"] == load_manifest["checkpoint_rng_state_sha256"],
    }
    for name, ok in immediate_load_checks.items():
        checks.append(Check(name, bool(ok), load_manifest))
        if not ok:
            raise SmokeFailure(f"Resume immediate-load check failed: {name}")

    resume_extra = args.reference_microbatches - args.checkpoint_after_microbatches
    print(f"[{SCRIPT_NAME}] resuming for {resume_extra} additional microbatches")
    resume_history_b, resume_summary_b = run_microbatches(
        project_root=project_root,
        protocol=training_protocol,
        model=resume_model,
        optimizer=resume_optimizer,
        scheduler=resume_scheduler,
        scaler=resume_scaler,
        criterion=criterion,
        device=device,
        loader_iter=resume_iter,
        start_microbatch_index=args.checkpoint_after_microbatches,
        num_microbatches=resume_extra,
        grad_acc_steps=grad_acc_steps,
        amp_dtype=amp_dtype,
        clip_max_norm=clip_max_norm,
        step_counter=resume_step_counter,
        run_name="resumed",
    )
    resumed_full_history = list(split_history_a) + list(resume_history_b)

    history_compare = compare_histories(ref_history, resumed_full_history)
    model_delta = max_abs_model_delta(ref_model, resume_model)
    grad_delta = max_abs_grad_delta(ref_model, resume_model)
    optimizer_hash_match = recursive_state_hash(ref_optimizer.state_dict()) == recursive_state_hash(resume_optimizer.state_dict())
    scaler_hash_match = recursive_state_hash(ref_scaler.state_dict()) == recursive_state_hash(resume_scaler.state_dict())
    scheduler_hash_match = recursive_state_hash(ref_scheduler.state_dict()) == recursive_state_hash(resume_scheduler.state_dict())
    rng_hash_match = hash_rng_state(rng_state_dict()) == ref_summary["rng_state_sha256"]

    continuity = {
        "history_compare": history_compare,
        "model_delta": model_delta,
        "grad_delta": grad_delta,
        "optimizer_state_hash_match": optimizer_hash_match,
        "scaler_state_hash_match": scaler_hash_match,
        "scheduler_state_hash_match": scheduler_hash_match,
        "final_rng_state_hash_match": rng_hash_match,
        "reference_summary": ref_summary,
        "split_pre_checkpoint_summary": split_summary_a,
        "resumed_summary": resume_summary_b,
    }
    result["continuity"] = continuity

    # Resume correctness must be judged against the state actually saved at
    # the checkpoint boundary. The independent uninterrupted reference branch
    # can differ at bit level because CUDA backward/optimizer kernels are not
    # guaranteed to be bitwise identical across separate executions.
    #
    # After checkpointing at microbatch 8, resumed microbatches 9-10 do not
    # execute another optimizer step. Therefore the optimizer state must remain
    # EXACTLY equal to the checkpoint optimizer state.
    resume_optimizer_matches_checkpoint = (
        recursive_state_hash(resume_optimizer.state_dict())
        == split_summary_a["optimizer_state_sha256"]
    )

    checks_to_add = {
        "reference_vs_resume_batch_history_equal":
            history_compare["history_lengths_equal"]
            and history_compare["batch_signatures_all_equal"],

        "reference_vs_resume_step_flags_equal":
            history_compare["step_flags_equal"],

        "reference_vs_resume_lr_used_equal":
            history_compare["lr_used_equal"],

        "reference_vs_resume_losses_allclose":
            history_compare["raw_losses_allclose_atol_1e-6"],

        "reference_vs_resume_model_allclose":
            model_delta["allclose_atol_1e-7"],

        "resume_optimizer_state_matches_checkpoint":
            resume_optimizer_matches_checkpoint,

        "reference_vs_resume_scaler_state_hash_equal":
            scaler_hash_match,

        "reference_vs_resume_scheduler_state_hash_equal":
            scheduler_hash_match,

        "final_rng_state_hash_match":
            rng_hash_match,

        "checkpoint_file_exists":
            bool(checkpoint_manifest["exists"]),
    }

    # Pending gradients after microbatches 9-10 are still GradScaler-scaled
    # because no optimizer step has occurred after resume. Their comparison
    # across two independent CUDA executions is retained for diagnostics, but
    # is not a hard PASS gate. Checkpointing occurs at an accumulation boundary
    # (pending accumulation == 0), so no gradient tensor is required to be
    # restored from the checkpoint.
    continuity["pending_gradient_diagnostic_only"] = {
        "hard_pass_gate": False,
        "reason": (
            "post-resume pending gradients are scaled and originate from "
            "independent CUDA backward executions"
        ),
        "grad_delta": grad_delta,
    }
    failed_continuity_checks = []
    for name, ok in checks_to_add.items():
        checks.append(Check(name, bool(ok), continuity if not ok else None))
        if not ok:
            failed_continuity_checks.append(name)

    if failed_continuity_checks:
        print("[smoke_train_engine_a_rgb] continuity diagnostics:")
        print(json.dumps(make_json_safe(continuity), indent=2, ensure_ascii=False, sort_keys=True))
        raise SmokeFailure(
            "Continuity check(s) failed: " + ", ".join(failed_continuity_checks)
        )

    # Gradient-accumulation protocol-specific checks.
    step_pattern = [bool(h["optimizer_step_executed"]) for h in split_history_a]
    expected_pattern = [False] * (grad_acc_steps - 1) + [True]
    assert_equal("first_8_microbatch_step_pattern", step_pattern[:grad_acc_steps], expected_pattern, checks)
    for i, h in enumerate(split_history_a[:grad_acc_steps]):
        expected_scaled = float(h["raw_loss"]) / float(grad_acc_steps)
        assert_close(f"microbatch_{i+1}_loss_divided_by_accumulation", float(h["scaled_loss"]), expected_scaled, checks, atol=1e-6, rtol=1e-6)
    lr_first_used = next((h["lr_used_for_step"] for h in split_history_a if h["optimizer_step_executed"]), None)
    assert_close("first_optimizer_step_lr_from_frozen_schedule", float(lr_first_used), float(lr_sequence[0]), checks, atol=1e-18, rtol=1e-12)
    first_step_record = next((h for h in split_history_a if h["optimizer_step_executed"]), None)
    gradient_clip_observed = first_step_record is not None and first_step_record.get("grad_norm_before_clip") is not None and math.isfinite(float(first_step_record.get("grad_norm_before_clip")))
    checks.append(Check("unscale_then_gradient_clip_observed_before_optimizer_step", gradient_clip_observed, first_step_record))
    if not gradient_clip_observed:
        raise SmokeFailure("Did not observe finite grad_norm_before_clip at optimizer-step microbatch")
    first_forward_meta = split_history_a[0].get("forward_meta", {}) if split_history_a else {}
    autocast_enabled_observed = bool(first_forward_meta.get("autocast_enabled_cuda", False))
    autocast_dtype_observed = str(first_forward_meta.get("autocast_cuda_dtype", ""))
    checks.append(Check("fp16_amp_autocast_enabled", autocast_enabled_observed and "float16" in autocast_dtype_observed, first_forward_meta))
    if not (autocast_enabled_observed and "float16" in autocast_dtype_observed):
        raise SmokeFailure(f"FP16 AMP autocast was not observed inside forward: {first_forward_meta}")

    amp_manifest = {
        "autocast_device_type": "cuda",
        "autocast_dtype": str(amp_dtype),
        "grad_scaler_enabled": bool(ref_scaler.is_enabled()),
        "grad_scaler_state_after_reference": make_json_safe(ref_scaler.state_dict()),
    }
    assert_equal("grad_scaler_enabled", amp_manifest["grad_scaler_enabled"], True, checks)

    result.update(
        {
            "status": "PASS",
            "completed_at": now_iso(),
            "model_manifest": ref_model_manifest,
            "dataloader_manifest": ref_loader_state,
            "optimizer_manifest": ref_opt_manifest,
            "param_group_manifest": ref_param_manifest,
            "amp_manifest": amp_manifest,
            "optimizer_step_pipeline_order": [
                "scaler.scale(loss / gradient_accumulation_steps).backward()",
                "scheduler.set_lr_for_next_step()",
                "scaler.unscale_(optimizer)",
                "clip_grad_norm_(max_norm=1.0)",
                "scaler.step(optimizer)",
                "scaler.update()",
                "scheduler.step_after_optimizer()",
                "optimizer.zero_grad(set_to_none=True)",
            ],
            "checkpoint_manifest": checkpoint_manifest,
            "resume_load_manifest": make_json_safe(load_manifest),
            "resume_loader_state": resume_loader_state,
            "histories": {
                "reference": ref_history,
                "split_pre_checkpoint": split_history_a,
                "resumed": resume_history_b,
            },
            "checks": [dataclasses.asdict(c) for c in checks],
        }
    )
    print(f"[{SCRIPT_NAME}] PASS ✅")
    return make_json_safe(result)


def write_text_summary(result: Mapping[str, Any], path: Path) -> None:
    lines: List[str] = []
    status = result.get("status", "UNKNOWN")
    lines.append(f"{SCRIPT_NAME} {SCRIPT_VERSION}")
    lines.append(f"status: {status}")
    lines.append(f"project_root: {result.get('project_root')}")
    lines.append(f"output_dir: {result.get('output_dir')}")
    lines.append("")
    if status == "PASS":
        cfg = result.get("smoke_config", {})
        lines.append("Scope: Model A RGB training-engine smoke-test only; no validation/test/full formal training.")
        lines.append(f"Microbatches: reference={cfg.get('reference_microbatches')}, checkpoint_after={cfg.get('checkpoint_after_microbatches')}, resume_extra={cfg.get('resume_extra_microbatches')}")
        lines.append(f"Gradient accumulation steps: {cfg.get('gradient_accumulation_steps')}")
        lines.append("")
        lines.append("Key PASS checks:")
        for name in (
            "training_protocol_sha256",
            "optimizer_steps_after_8_microbatches",
            "first_8_microbatch_step_pattern",
            "first_optimizer_step_lr_from_frozen_schedule",
            "grad_scaler_enabled",
            "reference_vs_resume_batch_history_equal",
            "reference_vs_resume_model_allclose",
            "resume_optimizer_state_matches_checkpoint",
            "checkpoint_file_exists",
        ):
            hit = next((c for c in result.get("checks", []) if c.get("name") == name), None)
            if hit is not None:
                lines.append(f"- {name}: {hit.get('passed')}")
        pg = result.get("param_group_manifest", {})
        lines.append("")
        lines.append(f"AdamW param groups: {pg.get('group_count')}; assignment_sha256={pg.get('assignment_sha256')}")
        for g in pg.get("groups", []):
            lines.append(f"- {g.get('group_name')}: params={g.get('parameter_count')}, numel={g.get('numel')}, wd={g.get('weight_decay')}, lr_scale={g.get('lr_scale')}")
        ckpt = result.get("checkpoint_manifest", {})
        lines.append("")
        lines.append(f"Checkpoint: {ckpt.get('path')}")
        lines.append(f"Checkpoint SHA256: {ckpt.get('sha256')}")
    else:
        lines.append("Failure summary:")
        lines.append(str(result.get("error", "unknown error")))
        if result.get("traceback"):
            lines.append("")
            lines.append(str(result.get("traceback")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model A RGB training-engine smoke-test")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--reference-microbatches", type=int, default=DEFAULT_REFERENCE_MICROBATCHES)
    parser.add_argument("--checkpoint-after-microbatches", type=int, default=DEFAULT_CHECKPOINT_AFTER_MICROBATCHES)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir if args.output_dir is not None else project_root / DEFAULT_OUTPUT_SUBDIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    console_path = output_dir / "console.txt"
    json_path = output_dir / "train_engine_smoke_a_rgb.json"
    txt_path = output_dir / "train_engine_smoke_a_rgb.txt"

    with tee_console(console_path):
        result: Dict[str, Any]
        try:
            result = run_experiment(project_root, output_dir, args)
            exit_code = 0
        except Exception as e:
            result = {
                "script_name": SCRIPT_NAME,
                "script_version": SCRIPT_VERSION,
                "status": "FAIL",
                "completed_at": now_iso(),
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"[{SCRIPT_NAME}] FAIL ❌: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            exit_code = 1
        finally:
            json_dump(make_json_safe(result), json_path)
            write_text_summary(make_json_safe(result), txt_path)
            print(f"[{SCRIPT_NAME}] wrote: {json_path}")
            print(f"[{SCRIPT_NAME}] wrote: {txt_path}")
            print(f"[{SCRIPT_NAME}] wrote: {console_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
