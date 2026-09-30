#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the existing Model D / DARF-B2 checkpoint on unseen atmospheric fog.

Fog is NOT added to training. It is an OOD test condition.

Fog model:
    I(x) = J(x) * t(x) + A * (1 - t(x))

Levels:
    L1: mean t=0.80, range [0.70, 0.90]
    L2: mean t=0.60, range [0.45, 0.75]
    L3: mean t=0.40, range [0.25, 0.55]

RGB only is degraded. NIR and GT stay unchanged. Fog is synthesized on the
full raw 6000x6000 tile before crop/normalization, so overlapping windows see
pixel-identical degradation.

Expected location:
    tools/validate_model_d_fog.py

Run:
    python tools/validate_model_d_fog.py
    python tools/validate_model_d_fog.py --batch-size 1
    python tools/validate_model_d_fog.py --force-fog
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline.potsdam_dataset import PotsdamSlidingWindowDataset, require
from models.segformer_b2_darf import MODEL_ID, MODEL_NAME, NUM_SCALES, build_model_d_darf_b2
import validate_model_d_darf_b2_v2 as base


FOG_PROTOCOL_VERSION = "Model D Atmospheric Fog OOD Protocol v1"
FOG_IMPLEMENTATION_REVISION = 1
FOG_SEED = 20260928
ATMOSPHERIC_LIGHT_255 = (245.0, 245.0, 245.0)

FOG_LEVELS: Dict[str, Dict[str, float]] = {
    "L1": {
        "transmission_mean": 0.80,
        "transmission_min": 0.70,
        "transmission_max": 0.90,
        "transmission_variation": 0.10,
    },
    "L2": {
        "transmission_mean": 0.60,
        "transmission_min": 0.45,
        "transmission_max": 0.75,
        "transmission_variation": 0.15,
    },
    "L3": {
        "transmission_mean": 0.40,
        "transmission_min": 0.25,
        "transmission_max": 0.55,
        "transmission_variation": 0.15,
    },
}

FOG_PROTOCOL: Dict[str, Any] = {
    "protocol_version": FOG_PROTOCOL_VERSION,
    "implementation_revision": FOG_IMPLEMENTATION_REVISION,
    "seed": FOG_SEED,
    "scientific_role": "OOD/unseen degradation; fog is absent from Model-D Robust-3 training",
    "equation": "I(x) = J(x) * t(x) + A * (1 - t(x))",
    "scope": {
        "degrade": "RGB only",
        "nir": "clean / unchanged",
        "gt": "unchanged",
        "application_domain": "raw uint8 RGB before normalization",
        "spatial_scope": "full 6000x6000 tile before sliding-window crop",
        "overlap_consistency": True,
    },
    "atmospheric_light_255": list(ATMOSPHERIC_LIGHT_255),
    "levels": FOG_LEVELS,
    "transmission_field": {
        "type": "deterministic low-frequency analytic field",
        "components": 4,
        "frequency_cycles_range": [0.35, 1.60],
    },
}

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "training" / "model_d_darf_b2" / "checkpoints" / "final.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "evaluation" / "model_d_darf_b2"
)
DEFAULT_CLEAN_METRICS = DEFAULT_OUTPUT_ROOT / "clean_val" / "metrics.json"


def _canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def fog_protocol_sha256() -> str:
    return hashlib.sha256(_canonical(FOG_PROTOCOL)).hexdigest()


def fog_spec(level: str) -> Dict[str, float]:
    if level not in FOG_LEVELS:
        raise ValueError(f"Unknown fog level: {level}")
    return dict(FOG_LEVELS[level])


def fog_condition(level: str) -> str:
    fog_spec(level)
    return f"fog_{level}"


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key) for key in fields})


def fog_seed(tile_id: str, level: str) -> int:
    payload = (
        f"{FOG_PROTOCOL_VERSION}|{FOG_SEED}|{tile_id}|fog|{level}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _field_params(tile_id: str, level: str):
    rng = np.random.default_rng(fog_seed(tile_id, level))
    return {
        "fx": rng.uniform(0.35, 1.60, 4).astype(np.float32),
        "fy": rng.uniform(0.35, 1.60, 4).astype(np.float32),
        "phase": rng.uniform(0.0, 2.0 * math.pi, 4).astype(np.float32),
        "weights": np.asarray([0.34, 0.27, 0.22, 0.17], dtype=np.float32),
        "signs": rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32), size=4
        ).astype(np.float32),
    }


def _transmission_chunk(y0, y1, h, w, level, params):
    spec = fog_spec(level)
    x = (np.arange(w, dtype=np.float32) / max(w - 1, 1))[None, :]
    y = (np.arange(y0, y1, dtype=np.float32) / max(h - 1, 1))[:, None]
    field = np.zeros((y1 - y0, w), dtype=np.float32)
    two_pi = np.float32(2.0 * math.pi)

    for i in range(4):
        phase = two_pi * (params["fx"][i] * x + params["fy"][i] * y) + params["phase"][i]
        field += (
            params["weights"][i]
            * params["signs"][i]
            * np.sin(phase).astype(np.float32, copy=False)
        )

    np.clip(field, -1.0, 1.0, out=field)
    t = (
        np.float32(spec["transmission_mean"])
        + np.float32(spec["transmission_variation"]) * field
    )
    np.clip(
        t,
        np.float32(spec["transmission_min"]),
        np.float32(spec["transmission_max"]),
        out=t,
    )
    return t


def apply_fog_full_tile(
    rgb: np.ndarray,
    *,
    tile_id: str,
    level: str,
    chunk_rows: int = 128,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [H,W,3] RGB, got {rgb.shape} {rgb.dtype}")

    h, w, _ = rgb.shape
    params = _field_params(tile_id, level)
    atmosphere = np.asarray(ATMOSPHERIC_LIGHT_255, dtype=np.float32).reshape(1, 1, 3)
    out = np.empty_like(rgb)

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        t = _transmission_chunk(y0, y1, h, w, level, params)[..., None]
        src = rgb[y0:y1].astype(np.float32, copy=False)
        fogged = src * t + atmosphere * (1.0 - t)
        np.clip(fogged, 0.0, 255.0, out=fogged)
        np.rint(fogged, out=fogged)
        out[y0:y1] = fogged.astype(np.uint8)

    return np.ascontiguousarray(out)


class FogPotsdamSlidingWindowDataset(base.DegradedPotsdamSlidingWindowDataset):
    """Reuse the frozen degraded-dataset crop/normalization path with Fog RGB."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        split: str,
        level: str,
        fog_chunk_rows: int,
    ):
        # Deliberately bypass DegradedPotsdamSlidingWindowDataset.__init__,
        # because fog is intentionally NOT part of rgb_degradation_protocol.py.
        PotsdamSlidingWindowDataset.__init__(
            self, project_root=project_root, split=split
        )
        fog_spec(level)
        self.corruption = "fog"
        self.level = level
        self.condition = fog_condition(level)
        self.fog_chunk_rows = int(fog_chunk_rows)
        self._degraded_cache_tile_id = None
        self._degraded_cache_rgb = None

    def _get_degraded_rgb(self, tile_id: str, rgbir: np.ndarray) -> np.ndarray:
        if (
            self._degraded_cache_tile_id == tile_id
            and self._degraded_cache_rgb is not None
        ):
            return self._degraded_cache_rgb

        degraded = apply_fog_full_tile(
            rgbir[..., 0:3],
            tile_id=tile_id,
            level=self.level,
            chunk_rows=self.fog_chunk_rows,
        )

        require(
            degraded.shape == (self.spec.tile_size, self.spec.tile_size, 3),
            f"{tile_id}: fog RGB shape invalid: {degraded.shape}",
        )
        require(
            degraded.dtype == np.uint8,
            f"{tile_id}: fog RGB dtype invalid: {degraded.dtype}",
        )
        self._degraded_cache_tile_id = tile_id
        self._degraded_cache_rgb = degraded
        return degraded


def patch_base_validator_for_fog() -> None:
    """
    Reuse the established Model-D evaluator without editing the existing
    rgb_degradation_protocol.py. The patch is process-local to this script.
    """
    base.DEGRADATION_PROTOCOL_VERSION = FOG_PROTOCOL_VERSION
    base.IMPLEMENTATION_REVISION = FOG_IMPLEMENTATION_REVISION
    base.degradation_protocol_sha256 = fog_protocol_sha256

    original_condition_spec = base.condition_spec

    def _condition_spec(corruption: str, level: str):
        if corruption == "fog":
            return fog_spec(level)
        return original_condition_spec(corruption, level)

    base.condition_spec = _condition_spec


def assert_fog_unseen(checkpoint_meta: Mapping[str, Any]) -> None:
    protocol = checkpoint_meta.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("Checkpoint has no protocol metadata.")

    training = protocol.get("corruption_training")
    if not isinstance(training, Mapping):
        raise RuntimeError("Checkpoint has no corruption_training metadata.")

    families = [str(x).lower() for x in training.get("families", [])]
    if any(("fog" in x or "haze" in x) for x in families):
        raise RuntimeError(
            "This validator defines fog as OOD, but checkpoint training metadata "
            f"contains fog/haze: {families}"
        )

    print(f"[OOD check] PASS | training families={families} | fog/haze absent")


def load_clean_reference(path: Path, checkpoint_step):
    if not path.is_file():
        print(f"[clean reference] WARNING | not found: {path}")
        return None, []

    obj = base.load_json(path)
    if obj.get("model") != MODEL_ID or obj.get("condition") != "Clean":
        raise RuntimeError("Clean reference is not Model D / Clean.")

    if (
        checkpoint_step is not None
        and obj.get("checkpoint_global_step") is not None
        and int(obj["checkpoint_global_step"]) != int(checkpoint_step)
    ):
        raise RuntimeError("Clean reference checkpoint step does not match.")

    gates = [dict(row) for row in obj.get("gate_statistics", [])]
    print(f"[clean reference] mIoU={float(obj['miou']):.6f} | {path}")
    return obj, gates


def build_fog_monotonicity(clean_gates, fog_gates):
    lookup = {}
    for row in clean_gates:
        lookup[("Clean", int(row["scale"]))] = row
    for row in fog_gates:
        lookup[(str(row["condition"]), int(row["scale"]))] = row

    has_clean = all(("Clean", s) in lookup for s in range(1, NUM_SCALES + 1))
    sequence = (
        ["Clean", "fog_L1", "fog_L2", "fog_L3"]
        if has_clean
        else ["fog_L1", "fog_L2", "fog_L3"]
    )

    scales = {}
    for scale in range(1, NUM_SCALES + 1):
        means = [
            float(lookup[(condition, scale)]["g_nir_mean"])
            for condition in sequence
        ]
        mono = all(
            means[i + 1] + 1e-12 >= means[i]
            for i in range(len(means) - 1)
        )
        scales[f"scale_{scale}"] = {
            "g_nir_mean": means,
            "delta_from_first": [x - means[0] for x in means],
            "monotonic_nondecreasing": mono,
        }

    return {
        "scientific_question": (
            "Does the DARF gate increase NIR residual correction under an "
            "unseen atmospheric fog degradation as severity increases?"
        ),
        "fog_seen_during_training": False,
        "sequence": sequence,
        "clean_reference_available": has_clean,
        "scales": scales,
        "all_scales_monotonic_nondecreasing": all(
            x["monotonic_nondecreasing"] for x in scales.values()
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Model D / DARF-B2 unseen Fog L1/L2/L3 validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--clean-metrics", type=Path, default=DEFAULT_CLEAN_METRICS)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=8)
    p.add_argument("--confusion-chunk-rows", type=int, default=512)
    p.add_argument("--fog-chunk-rows", type=int, default=128)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force-fog", action="store_true")
    x = p.parse_args()

    for name in ("batch_size", "log_every", "confusion_chunk_rows", "fog_chunk_rows"):
        if int(getattr(x, name)) <= 0:
            p.error(f"--{name.replace('_', '-')} must be > 0")
    return x


def main():
    x = parse_args()
    patch_base_validator_for_fog()

    checkpoint_path = base.resolve(x.checkpoint)
    output_root = base.resolve(x.output_root)
    clean_metrics_path = base.resolve(x.clean_metrics)
    fog_root = output_root / "fog_ood_val"
    fog_root.mkdir(parents=True, exist_ok=True)

    protocol_out = dict(FOG_PROTOCOL)
    protocol_out["sha256"] = fog_protocol_sha256()
    save_json(fog_root / "fog_protocol.json", protocol_out)

    device = base.get_device(x.device)
    amp_enabled = device.type == "cuda" and not x.no_amp

    print("=" * 112)
    print("MODEL D / DARF-B2 | UNSEEN ATMOSPHERIC FOG OOD VALIDATION")
    print("=" * 112)
    print(f"checkpoint       : {checkpoint_path}")
    print(f"device / AMP     : {device} / {amp_enabled}")
    print(f"fog protocol     : {FOG_PROTOCOL_VERSION}")
    print(f"fog protocol hash: {fog_protocol_sha256()}")
    print("training change  : NONE")
    print("=" * 112)

    model, _ = build_model_d_darf_b2(PROJECT_ROOT)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict, checkpoint_meta = base.unwrap_checkpoint(checkpoint)
    base.validate_checkpoint_metadata(checkpoint_meta)
    assert_fog_unseen(checkpoint_meta)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    checkpoint_epoch = checkpoint_meta.get("epoch")
    checkpoint_step = checkpoint_meta.get(
        "global_step", checkpoint_meta.get("step")
    )
    checkpoint_epoch = int(checkpoint_epoch) if checkpoint_epoch is not None else None
    checkpoint_step = int(checkpoint_step) if checkpoint_step is not None else None

    clean_metrics, clean_gate_rows = load_clean_reference(
        clean_metrics_path, checkpoint_step
    )
    clean_miou = (
        float(clean_metrics["miou"]) if clean_metrics is not None else None
    )

    clean_probe = PotsdamSlidingWindowDataset(PROJECT_ROOT, split="val")
    results = {}
    all_fog_gate_rows = []

    for index, level in enumerate(("L1", "L2", "L3"), start=1):
        condition = fog_condition(level)
        condition_dir = fog_root / condition
        metrics_path = condition_dir / "metrics.json"

        print()
        print(f"[Fog {index}/3] {condition} {fog_spec(level)}")

        result = None
        if not x.force_fog:
            result = base.compatible_existing(
                metrics_path,
                condition=condition,
                checkpoint_path=checkpoint_path,
                checkpoint_global_step=checkpoint_step,
                degraded=True,
            )

        if result is None:
            dataset = FogPotsdamSlidingWindowDataset(
                PROJECT_ROOT,
                split="val",
                level=level,
                fog_chunk_rows=x.fog_chunk_rows,
            )

            # Reuse the established probe: same window/metadata, NIR identical,
            # RGB changed.
            base.degradation_probe(
                clean_dataset=clean_probe,
                degraded_dataset=dataset,
            )

            result, gate_rows = base.evaluate_condition(
                model=model,
                dataset=dataset,
                condition=condition,
                corruption="fog",
                severity_level=level,
                severity_rank=int(level[1:]),
                output_dir=condition_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_global_step=checkpoint_step,
                batch_size=x.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                log_every=x.log_every,
                confusion_chunk_rows=x.confusion_chunk_rows,
                save_predictions=x.save_predictions,
            )
            del dataset
        else:
            gate_rows = [dict(row) for row in result["gate_statistics"]]
            print(f"[resume] {metrics_path}")

        result["fog_seen_during_training"] = False
        result["fog_protocol_version"] = FOG_PROTOCOL_VERSION
        result["fog_implementation_revision"] = FOG_IMPLEMENTATION_REVISION
        result["fog_protocol_sha256"] = fog_protocol_sha256()
        result["fog_equation"] = "I(x) = J(x) * t(x) + A * (1 - t(x))"
        result["atmospheric_light_255"] = list(ATMOSPHERIC_LIGHT_255)

        if clean_miou is not None:
            fog_miou = float(result["miou"])
            drop = clean_miou - fog_miou
            result["clean_reference_miou"] = clean_miou
            result["drop_miou"] = drop
            result["delta_miou"] = -drop
            result["relative_drop_pct"] = 100.0 * drop / clean_miou
            result["retention_pct"] = 100.0 * fog_miou / clean_miou

        base.save_json(metrics_path, result)
        results[condition] = result
        all_fog_gate_rows.extend(dict(row) for row in gate_rows)

        print(
            f"[{condition}] mIoU={float(result['miou']):.6f} | "
            + (
                f"Drop={float(result['drop_miou']):.6f} | "
                if clean_miou is not None
                else ""
            )
            + f"gNIR={[round(float(r['g_nir_mean']), 4) for r in gate_rows]}"
        )

    summary_rows = []
    for level in ("L1", "L2", "L3"):
        condition = fog_condition(level)
        r = results[condition]
        row = {
            "model": MODEL_ID,
            "condition": condition,
            "corruption": "fog",
            "severity_level": level,
            "severity_rank": int(level[1:]),
            "clean_miou": r.get("clean_reference_miou"),
            "miou": r["miou"],
            "drop_miou": r.get("drop_miou"),
            "relative_drop_pct": r.get("relative_drop_pct"),
            "retention_pct": r.get("retention_pct"),
            "pixel_accuracy": r["pixel_accuracy"],
            "mean_class_accuracy": r["mean_class_accuracy"],
            "validation_seconds": r["validation_seconds"],
            "fog_seen_during_training": False,
        }
        summary_rows.append(row)

    save_json(
        fog_root / "fog_summary.json",
        {
            "model": MODEL_ID,
            "model_name": MODEL_NAME,
            "backbone": "SegFormer-B2",
            "split": "val",
            "scientific_role": "OOD / unseen atmospheric fog degradation",
            "fog_seen_during_training": False,
            "fog_protocol_version": FOG_PROTOCOL_VERSION,
            "fog_protocol_sha256": fog_protocol_sha256(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_global_step": checkpoint_step,
            "clean_reference": (
                {"path": str(clean_metrics_path), "miou": clean_miou}
                if clean_miou is not None
                else None
            ),
            "results": summary_rows,
        },
    )

    write_csv(
        fog_root / "fog_summary.csv",
        summary_rows,
        [
            "model",
            "condition",
            "corruption",
            "severity_level",
            "severity_rank",
            "clean_miou",
            "miou",
            "drop_miou",
            "relative_drop_pct",
            "retention_pct",
            "pixel_accuracy",
            "mean_class_accuracy",
            "validation_seconds",
            "fog_seen_during_training",
        ],
    )

    gate_rows = clean_gate_rows + all_fog_gate_rows
    save_json(
        fog_root / "fog_gate_statistics.json",
        {
            "model": MODEL_ID,
            "gate_semantics": "g_NIR residual correction strength",
            "fog_seen_during_training": False,
            "rows": gate_rows,
        },
    )
    write_csv(
        fog_root / "fog_gate_statistics.csv",
        gate_rows,
        [
            "condition",
            "corruption",
            "severity_level",
            "severity_rank",
            "scale",
            "count",
            "g_nir_mean",
            "g_nir_std",
            "g_nir_min",
            "g_nir_p05",
            "g_nir_p25",
            "g_nir_median",
            "g_nir_p75",
            "g_nir_p95",
            "g_nir_max",
        ],
    )

    monotonicity = build_fog_monotonicity(
        clean_gate_rows, all_fog_gate_rows
    )
    save_json(
        fog_root / "fog_gate_monotonicity.json",
        monotonicity,
    )

    print()
    print("=" * 112)
    print("FOG OOD SUMMARY")
    print("=" * 112)
    for row in summary_rows:
        message = f"{row['condition']}: mIoU={float(row['miou']):.6f}"
        if row["drop_miou"] is not None:
            message += (
                f" | Drop={float(row['drop_miou']):.6f}"
                f" | Retention={float(row['retention_pct']):.2f}%"
            )
        print(message)

    print(
        "Gate monotonic Clean/Fog: "
        f"{monotonicity['all_scales_monotonic_nondecreasing']}"
    )
    print(f"Protocol     : {fog_root / 'fog_protocol.json'}")
    print(f"Summary      : {fog_root / 'fog_summary.json'}")
    print(f"Summary CSV  : {fog_root / 'fog_summary.csv'}")
    print(f"Gate stats   : {fog_root / 'fog_gate_statistics.csv'}")
    print(f"Gate analysis: {fog_root / 'fog_gate_monotonicity.json'}")
    print("=" * 112)


if __name__ == "__main__":
    main()
