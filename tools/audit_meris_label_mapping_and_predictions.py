#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit TLSNMSP labels, seven-class mapping, classifier head and class predictions.

Run from the project root after copying this file into tools/.
The script imports the project's existing evaluation helpers, so preprocessing and
full-image sliding-window inference stay consistent with the baseline evaluator.
It does not modify training data or checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the exact preprocessing/model helpers from the validation script.
import eval_model_b2_rgbnir_meris_coverage as ev  # noqa: E402
from models.segformer_b2_rgbnir_meris import (  # noqa: E402
    CLASS_NAMES, NUM_CLASSES, RAW_LABEL_TO_TRAIN_ID, build_model_b2_rgbnir_meris,
)

DEFAULT_DATA_ROOT = Path(r"D:\MRISE\dataset\TLSNMSP")
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "data" / "processed" / "meris_tlsnmsp"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "training" / "meris_b2_rgbnir_fixed_g1_seed20261028" / "checkpoints" / "best.pt"
DEFAULT_PRETRAINED = Path(r"D:\MRISE\hf-cache\segformer-b2-finetuned-ade-512-512")
EXPECTED_RAW_TO_TRAIN = {i: i - 1 for i in range(1, 8)}
CHUNK_PIXELS = 400_000


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def load_label_only(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"标签读取失败：{path}")
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            raise RuntimeError(f"标签不是单通道类别 ID 图：{path}, shape={arr.shape}")
    if arr.ndim != 2:
        raise RuntimeError(f"标签维度异常：{path}, shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


def audit_protocol_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    model_map = {int(k): int(v) for k, v in RAW_LABEL_TO_TRAIN_ID.items()}
    config_map_raw = cfg.get("raw_label_to_train_id", {})
    try:
        config_map = {int(k): int(v) for k, v in config_map_raw.items()}
    except Exception:
        config_map = {}
    model_names = [str(x) for x in CLASS_NAMES]
    config_names = [str(x) for x in cfg.get("class_names_train_id_order", [])]
    expected_names = [
        "building", "impervious_surface", "water", "barren", "tree", "low_vegetation", "background_other"
    ]
    names_order_ok = len(model_names) == NUM_CLASSES and [norm_name(x) for x in model_names] == [norm_name(x) for x in expected_names]
    config_names_match_model = (
        len(config_names) == len(model_names)
        and [norm_name(x) for x in config_names] == [norm_name(x) for x in model_names]
    )
    return {
        "num_classes_model": int(NUM_CLASSES),
        "model_class_names_in_train_id_order": model_names,
        "config_class_names_in_train_id_order": config_names,
        "expected_semantic_order": expected_names,
        "model_class_order_matches_expected": bool(names_order_ok),
        "config_class_order_matches_model": bool(config_names_match_model),
        "expected_raw_id_to_train_id": {str(k): v for k, v in EXPECTED_RAW_TO_TRAIN.items()},
        "model_raw_id_to_train_id": {str(k): v for k, v in model_map.items()},
        "config_raw_id_to_train_id": {str(k): v for k, v in config_map.items()},
        "model_mapping_matches_expected": model_map == EXPECTED_RAW_TO_TRAIN,
        "config_mapping_matches_expected": config_map == EXPECTED_RAW_TO_TRAIN,
    }


def audit_labels(protocol_dir: Path, data_root: Path, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"split_counts": {}, "unknown_label_values": [], "missing_class_by_split": {}}
    for split in ("train", "val", "test"):
        rows = ev.read_csv_rows(protocol_dir / f"{split}.csv")
        totals = np.zeros(NUM_CLASSES, dtype=np.int64)
        image_count = 0
        for row in rows:
            label_path = ev.resolve_manifest_path(data_root, row["label_path"])
            label = load_label_only(label_path)
            ids, counts = np.unique(label, return_counts=True)
            found_ids = set(int(x) for x in ids.tolist())
            unknown = sorted(found_ids - set(EXPECTED_RAW_TO_TRAIN))
            if unknown:
                summary["unknown_label_values"].append({"split": split, "sample_id": row.get("sample_id", label_path.stem), "values": unknown, "path": str(label_path)})
            per_image: dict[str, Any] = {"split": split, "sample_id": row.get("sample_id", label_path.stem), "label_path": str(label_path), "height": int(label.shape[0]), "width": int(label.shape[1]), "pixel_count": int(label.size), "unknown_label_values": ";".join(map(str, unknown))}
            for raw_id in range(1, NUM_CLASSES + 1):
                count = int(counts[np.where(ids == raw_id)[0][0]]) if raw_id in found_ids else 0
                train_id = raw_id - 1
                totals[train_id] += count
                per_image[f"raw_{raw_id}_{CLASS_NAMES[train_id]}_pixels"] = count
                per_image[f"train_{train_id}_pixels"] = count
            image_rows.append(per_image)
            image_count += 1
        missing = [str(CLASS_NAMES[i]) for i, count in enumerate(totals) if count == 0]
        summary["split_counts"][split] = {
            "image_count": image_count,
            "total_pixels": int(totals.sum()),
            "raw_id_pixel_counts": {str(i + 1): int(totals[i]) for i in range(NUM_CLASSES)},
            "train_id_pixel_counts": {str(i): int(totals[i]) for i in range(NUM_CLASSES)},
            "missing_classes": missing,
        }
        summary["missing_class_by_split"][split] = missing
        for class_id, name in enumerate(CLASS_NAMES):
            raw_id = class_id + 1
            all_rows.append({
                "split": split,
                "class_id_train": class_id,
                "raw_label_id": raw_id,
                "class_name": str(name),
                "pixel_count": int(totals[class_id]),
                "pixel_share_pct": float(100.0 * totals[class_id] / max(1, int(totals.sum()))),
                "present": bool(totals[class_id] > 0),
            })
        print(f"[LABEL AUDIT] {split}: images={image_count}, pixels={int(totals.sum())}, missing={missing}")
    write_csv(output_dir / "label_counts_by_split.csv", all_rows)
    write_csv(output_dir / "per_image_label_counts.csv", image_rows)
    (output_dir / "label_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_rows, image_rows, summary


def checkpoint_head_audit(payload: dict[str, Any], state: dict[str, torch.Tensor], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    weight_key = next((k for k in state if k.endswith("decode_head.classifier.weight") or k.endswith("classifier.weight")), None)
    bias_key = next((k for k in state if k.endswith("decode_head.classifier.bias") or k.endswith("classifier.bias")), None)
    if weight_key is None:
        issues.append("未在 checkpoint 中找到分类头权重 key")
    else:
        w = state[weight_key].detach().float().cpu()
        if w.ndim < 1 or w.shape[0] != NUM_CLASSES:
            issues.append(f"分类头输出通道数异常：weight shape={tuple(w.shape)}，预期第一维={NUM_CLASSES}")
        elif not torch.isfinite(w).all():
            issues.append("分类头权重包含 NaN/Inf")
        for cid in range(min(NUM_CLASSES, int(w.shape[0]) if w.ndim else 0)):
            rows.append({
                "class_id_train": cid,
                "class_name": str(CLASS_NAMES[cid]),
                "classifier_weight_key": weight_key,
                "weight_l2_norm": float(torch.linalg.vector_norm(w[cid]).item()),
                "weight_mean": float(w[cid].mean().item()),
                "weight_std": float(w[cid].std(unbiased=False).item()),
                "bias": None,
            })
    if bias_key is not None:
        b = state[bias_key].detach().float().cpu().reshape(-1)
        if len(b) != NUM_CLASSES:
            issues.append(f"分类头 bias 数量异常：{len(b)}，预期={NUM_CLASSES}")
        elif not torch.isfinite(b).all():
            issues.append("分类头 bias 包含 NaN/Inf")
        by_id = {int(r["class_id_train"]): r for r in rows}
        for cid, value in enumerate(b.tolist()[:NUM_CLASSES]):
            if cid in by_id:
                by_id[cid]["bias"] = float(value)
            else:
                rows.append({"class_id_train": cid, "class_name": str(CLASS_NAMES[cid]), "classifier_weight_key": weight_key, "weight_l2_norm": None, "weight_mean": None, "weight_std": None, "bias": float(value)})
    elif weight_key is not None:
        issues.append("未在 checkpoint 中找到分类头 bias")
    model_id = payload.get("model_id") if isinstance(payload, dict) else None
    head_summary = {
        "model_id": model_id,
        "classifier_weight_key": weight_key,
        "classifier_bias_key": bias_key,
        "classifier_weight_shape": list(state[weight_key].shape) if weight_key else None,
        "classifier_bias_shape": list(state[bias_key].shape) if bias_key else None,
        "issues": issues,
    }
    write_csv(output_dir / "checkpoint_head_diagnostics.csv", rows)
    return rows, head_summary


def infer_with_logits(model: torch.nn.Module, rgb: np.ndarray, nir: np.ndarray, label_raw: np.ndarray,
                      cfg: dict[str, Any], device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = label_raw.shape
    ys, xs = ev.window_starts(height), ev.window_starts(width)
    logit_sum = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    votes = np.zeros((height, width), dtype=np.uint16)
    model.eval()
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                rgb_patch = np.ascontiguousarray(rgb[y0:y0 + ev.CROP_SIZE, x0:x0 + ev.CROP_SIZE])
                nir_patch = np.ascontiguousarray(nir[y0:y0 + ev.CROP_SIZE, x0:x0 + ev.CROP_SIZE])
                lab_patch = np.ascontiguousarray(label_raw[y0:y0 + ev.CROP_SIZE, x0:x0 + ev.CROP_SIZE])
                rgb_t, nir_t, _ = ev.pack_inputs(rgb_patch, nir_patch, lab_patch, cfg)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                nir_t = nir_t.unsqueeze(0).to(device, non_blocking=True)
                if amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        output = model(rgb_t, nir_t)
                else:
                    output = model(rgb_t, nir_t)
                if isinstance(output, dict):
                    output = output.get("logits", output.get("out"))
                if not torch.is_tensor(output) or output.ndim != 4:
                    raise RuntimeError("模型输出不是 [B,C,H,W] logits")
                if int(output.shape[1]) != NUM_CLASSES:
                    raise RuntimeError(f"模型输出通道={output.shape[1]}，期望={NUM_CLASSES}")
                if output.shape[-2:] != (ev.CROP_SIZE, ev.CROP_SIZE):
                    output = F.interpolate(output.float(), size=(ev.CROP_SIZE, ev.CROP_SIZE), mode="bilinear", align_corners=False)
                arr = output[0].float().cpu().numpy()
                y1, x1 = y0 + ev.CROP_SIZE, x0 + ev.CROP_SIZE
                logit_sum[:, y0:y1, x0:x1] += arr
                votes[y0:y1, x0:x1] += 1
                del rgb_t, nir_t, output, arr
    if np.any(votes == 0):
        raise RuntimeError("滑窗拼接后存在未预测像素")
    logit_sum /= votes[None, :, :].astype(np.float32)
    gt = ev.label_to_train_ids(label_raw)
    pred = np.argmax(logit_sum, axis=0).astype(np.int64)
    return gt, pred, logit_sum


def logits_diagnostics(logits: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    flat_logits = logits.reshape(NUM_CLASSES, -1)
    flat_gt = gt.reshape(-1)
    flat_pred = pred.reshape(-1)
    gt_counts = np.bincount(flat_gt, minlength=NUM_CLASSES)
    pred_counts = np.bincount(flat_pred, minlength=NUM_CLASSES)
    logit_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    prob_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    true_logit_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    true_prob_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    true_margin_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    true_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total = flat_gt.size
    for start in range(0, total, CHUNK_PIXELS):
        end = min(total, start + CHUNK_PIXELS)
        z = flat_logits[:, start:end].T.astype(np.float32, copy=False)
        y = flat_gt[start:end]
        logit_sums += z.sum(axis=0, dtype=np.float64)
        zmax = np.max(z, axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        probs = expz / np.maximum(expz.sum(axis=1, keepdims=True), 1e-12)
        prob_sums += probs.sum(axis=0, dtype=np.float64)
        for cid in range(NUM_CLASSES):
            mask = y == cid
            if not np.any(mask):
                continue
            zc = z[mask]
            pc = probs[mask, cid]
            other_max = np.max(np.delete(zc, cid, axis=1), axis=1)
            true_logit_sums[cid] += zc[:, cid].sum(dtype=np.float64)
            true_prob_sums[cid] += pc.sum(dtype=np.float64)
            true_margin_sums[cid] += (zc[:, cid] - other_max).sum(dtype=np.float64)
            true_counts[cid] += int(mask.sum())
    rows = []
    for cid, name in enumerate(CLASS_NAMES):
        n = int(true_counts[cid])
        rows.append({
            "class_id_train": cid,
            "raw_label_id": cid + 1,
            "class_name": str(name),
            "gt_pixels": int(gt_counts[cid]),
            "pred_pixels": int(pred_counts[cid]),
            "pred_share_pct": float(100.0 * pred_counts[cid] / max(1, total)),
            "mean_logit_all_pixels": float(logit_sums[cid] / max(1, total)),
            "mean_softmax_prob_all_pixels": float(prob_sums[cid] / max(1, total)),
            "mean_true_class_logit_on_gt_pixels": float(true_logit_sums[cid] / n) if n else None,
            "mean_true_class_probability_on_gt_pixels": float(true_prob_sums[cid] / n) if n else None,
            "mean_true_vs_best_other_logit_margin": float(true_margin_sums[cid] / n) if n else None,
            "gt_class_has_pixels": bool(n > 0),
            "pred_class_has_pixels": bool(pred_counts[cid] > 0),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="核验 TLSNMSP 七分类标签映射、分类头与建筑物/背景预测异常")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--max-images", type=int, default=0, help="每个 split 最多推理多少幅；0=全部。标签审计始终扫描 train/val/test 全部标签")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    pretrained_path = args.pretrained_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据根目录不存在：{data_root}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint 不存在：{checkpoint_path}")
    if not pretrained_path.is_dir() or not (pretrained_path / "config.json").is_file():
        raise SystemExit(f"本地预训练模型目录不存在或缺少 config.json：{pretrained_path}")
    if args.max_images < 0:
        raise SystemExit("--max-images 必须 >= 0")

    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "outputs" / "training" / "meris_b2_rgbnir_fixed_g1_seed20261028" / "audit" / checkpoint_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = ev.load_config(protocol_dir / "dataset_config.json")
    mapping_audit = audit_protocol_mapping(cfg)
    print("[MAPPING AUDIT]", json.dumps(mapping_audit, ensure_ascii=False))
    _, _, label_summary = audit_labels(protocol_dir, data_root, output_dir)

    payload = ev.load_checkpoint(checkpoint_path, "cpu")
    state = ev.extract_state_dict(payload)
    head_rows, head_summary = checkpoint_head_audit(payload, state, output_dir)
    model_id = payload.get("model_id") if isinstance(payload, dict) else None
    if model_id and model_id != "B2_RGBNIR_FIXED_RESIDUAL_MERIS7":
        raise RuntimeError(f"checkpoint model_id 不匹配：{model_id}")
    saved_hash = payload.get("train_manifest_sha256") if isinstance(payload, dict) else None
    train_csv = protocol_dir / "train.csv"
    if saved_hash and train_csv.is_file():
        import hashlib
        h = hashlib.sha256()
        with train_csv.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != saved_hash:
            raise RuntimeError("当前 train.csv 与 checkpoint 的 SHA-256 不一致，拒绝推理审计")

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定 cuda 但 torch.cuda.is_available() 为 False")
    amp = bool(device.type == "cuda" and not args.no_amp)
    fixed_strength = ev.find_fixed_strength(payload)
    model, model_meta = build_model_b2_rgbnir_meris(PROJECT_ROOT, checkpoint=str(pretrained_path), fixed_strength=float(fixed_strength))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"[MODEL LOAD PASS] strict=True | device={device} | AMP={amp} | model_id={model_id}")
    del payload, state

    selected = ["val", "test"] if args.split == "both" else [args.split]
    inference_summary: dict[str, Any] = {}
    findings: list[str] = []
    if not mapping_audit["model_mapping_matches_expected"] or not mapping_audit["config_mapping_matches_expected"]:
        findings.append("MAPPING_MISMATCH: 模型/协议中的 raw ID -> train ID 映射不符合 1..7 -> 0..6")
    if not mapping_audit["model_class_order_matches_expected"] or not mapping_audit["config_class_order_matches_model"]:
        findings.append("CLASS_ORDER_MISMATCH: 类别名称顺序与预期不一致")
    if head_summary["issues"]:
        findings.extend("CHECKPOINT_HEAD: " + x for x in head_summary["issues"])
    for split in selected:
        rows = ev.read_csv_rows(protocol_dir / f"{split}.csv")
        if args.max_images > 0:
            rows = rows[:args.max_images]
        confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        pred_rows: list[dict[str, Any]] = []
        all_diag: list[dict[str, Any]] = []
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(rows, 1):
            sample_id = row.get("sample_id", f"{split}_{idx}")
            rgb_path = ev.resolve_manifest_path(data_root, row["rgb_path"])
            ms_path = ev.resolve_manifest_path(data_root, row["multispectral_path"])
            label_path = ev.resolve_manifest_path(data_root, row["label_path"])
            rgb, nir, label_raw = ev.load_sample_arrays(rgb_path, ms_path, label_path)
            gt, pred, logits = infer_with_logits(model, rgb, nir, label_raw, cfg, device, amp)
            ev.confusion_update(confmat, gt, pred)
            class_diag = logits_diagnostics(logits, gt, pred)
            for item in class_diag:
                item["split"] = split
                item["sample_id"] = sample_id
                all_diag.append(item)
            image_row: dict[str, Any] = {"split": split, "sample_id": sample_id, "height": int(gt.shape[0]), "width": int(gt.shape[1]), "pixels": int(gt.size)}
            for cid, name in enumerate(CLASS_NAMES):
                image_row[f"gt_{cid}_{name}"] = int((gt == cid).sum())
                image_row[f"pred_{cid}_{name}"] = int((pred == cid).sum())
            image_row["building_gt_pixels"] = int((gt == 0).sum())
            image_row["building_pred_pixels"] = int((pred == 0).sum())
            image_row["background_gt_pixels"] = int((gt == 6).sum())
            image_row["background_pred_pixels"] = int((pred == 6).sum())
            pred_rows.append(image_row)
            print(f"[INFER {split}] {idx}/{len(rows)} {sample_id}: building GT/Pred={image_row['building_gt_pixels']}/{image_row['building_pred_pixels']}; background GT/Pred={image_row['background_gt_pixels']}/{image_row['background_pred_pixels']}")
            del rgb, nir, label_raw, gt, pred, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
        metrics = ev.metrics_from_confusion(confmat)
        write_csv(split_dir / "per_image_class_pixel_counts.csv", pred_rows)
        write_csv(split_dir / "per_class_logit_diagnostics.csv", all_diag)
        cm_rows = []
        for gt_id, gt_name in enumerate(CLASS_NAMES):
            row_cm = {"gt_class_id": gt_id, "gt_class_name": str(gt_name)}
            for pred_id, pred_name in enumerate(CLASS_NAMES):
                row_cm[f"pred_{pred_id}_{pred_name}"] = int(confmat[gt_id, pred_id])
            cm_rows.append(row_cm)
        write_csv(split_dir / "confusion_matrix.csv", cm_rows)
        (split_dir / "classification_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        class_pred_counts = confmat.sum(axis=0)
        class_gt_counts = confmat.sum(axis=1)
        zero_pred_classes = [str(CLASS_NAMES[i]) for i in range(NUM_CLASSES) if class_pred_counts[i] == 0]
        no_gt_classes = [str(CLASS_NAMES[i]) for i in range(NUM_CLASSES) if class_gt_counts[i] == 0]
        if zero_pred_classes:
            findings.append(f"{split.upper()}_ZERO_PREDICTIONS: {', '.join(zero_pred_classes)}")
        inference_summary[split] = {
            "images_inferred": len(rows),
            "is_full_split": args.max_images == 0,
            "metrics": metrics,
            "gt_pixel_counts_by_class": {str(CLASS_NAMES[i]): int(class_gt_counts[i]) for i in range(NUM_CLASSES)},
            "pred_pixel_counts_by_class": {str(CLASS_NAMES[i]): int(class_pred_counts[i]) for i in range(NUM_CLASSES)},
            "zero_prediction_classes": zero_pred_classes,
            "classes_absent_from_gt_in_inferred_subset": no_gt_classes,
        }
        print(f"[RESULT {split}] OA={metrics['OA']:.4f} mIoU={metrics['mIoU']:.4f} zero_pred={zero_pred_classes}")

    # Use train label totals to distinguish absence in the training data from prediction collapse.
    train_counts = label_summary["split_counts"]["train"]["raw_id_pixel_counts"]
    for raw_id, name in [(1, "building"), (7, "background_other")]:
        if int(train_counts.get(str(raw_id), 0)) == 0:
            findings.append(f"TRAIN_LABEL_MISSING: raw ID {raw_id} ({name}) has zero training pixels")
    if any("ZERO_PREDICTIONS" in x for x in findings) and not any("MAPPING_MISMATCH" in x or "CLASS_ORDER_MISMATCH" in x for x in findings):
        findings.append("INTERPRETATION: 若标签映射、类别顺序和 7 通道分类头检查均通过，但 GT 中存在建筑物/背景而 Pred 为 0，说明当前 checkpoint 未输出这些类别；这更像模型学习/类别不平衡问题，但仅凭本审计不能断定唯一原因。")

    final = {
        "checkpoint": str(checkpoint_path),
        "model_id": model_id,
        "device": str(device),
        "amp": amp,
        "mapping_audit": mapping_audit,
        "checkpoint_head_audit": head_summary,
        "label_audit": label_summary,
        "inference": inference_summary,
        "findings": findings,
        "elapsed_seconds": time.time() - started,
        "notes": [
            "标签审计统计的是协议清单 label_path 指向的转换后单通道标签（期望原始类别 ID 1..7）。",
            "本脚本检查转换标签与模型通道语义顺序；它不会重新解析原始彩色 mask 的颜色调色板。",
            "若要核验彩色原始 mask -> mask_converted 的逐像素转换，需要使用数据准备脚本中的原始颜色映射表。",
            "--max-images 大于 0 时，推理指标只代表所选子集；标签像素统计仍扫描 train/val/test 全部影像。",
        ],
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[AUDIT COMPLETE]")
    print(f"[SUMMARY] {output_dir / 'audit_summary.json'}")
    print("[FINDINGS]")
    for item in findings:
        print(" - " + item)
    if not findings:
        print(" - 未发现协议映射、标签缺失或零预测异常；请结合逐类 logits 与可视化继续判断。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n核验中断。已生成的审计文件保留。", file=sys.stderr)
        raise SystemExit(130)
