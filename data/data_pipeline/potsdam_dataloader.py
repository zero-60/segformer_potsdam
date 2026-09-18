#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Potsdam Dataset Protocol v1 的 DataLoader / epoch shuffle 实现与审计。

本阶段只解决：
1) 每个 epoch 的 1152 个 Train Dataset indices 如何做可复现 shuffle；
2) A/B/C/C-noGate 如何共享完全相同的 epoch sample order；
3) DataLoader worker 数变化时，样本 metadata / tensor 内容是否保持一致；
4) 避免 persistent worker 持有旧 epoch Dataset 状态；
5) 确保 drop_last=False，不丢弃任何一个 frozen sample slot。

明确不包含：
- batch size 的科研冻结（后续训练配置再决定）
- 模型
- optimizer / scheduler
- 训练循环
- corruption
- 4-channel model
- dual encoder / fusion / Quality Gate

Shuffle 不依赖 Python / NumPy / torch PRNG 的具体版本：
对每个 dataset index i 计算
    SHA256(f"{DATA_SEED}|epoch_shuffle|{epoch}|{i}")
然后按 digest + index 排序得到 permutation。
这样给定 seed/epoch/dataset length，顺序是 stateless、跨 worker、跨进程可重现的。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

try:
    from data_pipeline.potsdam_dataset import (
        MODULE_VERSION as DATASET_MODULE_VERSION,
        PotsdamTrainDataset,
        DatasetProtocolError,
        sha256_file,
    )
except ImportError:
    # 支持直接执行：
    # python data_pipeline/potsdam_dataloader.py --smoke-test
    from potsdam_dataset import (
        MODULE_VERSION as DATASET_MODULE_VERSION,
        PotsdamTrainDataset,
        DatasetProtocolError,
        sha256_file,
    )


MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "DataLoader Protocol v1"

DATA_SEED = 20260917
EXPECTED_TRAIN_LENGTH = 1152
AUDIT_EPOCHS = (0, 1, 2)

# 这些属于数据顺序/完整性协议，正式冻结。
DROP_LAST = False
PERSISTENT_WORKERS = False

# 下面只是本 smoke-test 的资源参数，不写成科研超参数。
AUDIT_BATCH_SIZE = 2
AUDIT_COMPARE_ITEMS = 8
AUDIT_MULTIWORKER_COUNT = 2


class DataLoaderProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataLoaderProtocolError(message)


def sha256_json_canonical(obj: Any) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# 稳定 epoch permutation
# ---------------------------------------------------------------------

def _shuffle_key(seed: int, epoch: int, index: int) -> bytes:
    return hashlib.sha256(
        f"{seed}|epoch_shuffle|{epoch}|{index}".encode("utf-8")
    ).digest()


def deterministic_epoch_permutation(
    length: int,
    seed: int,
    epoch: int,
) -> List[int]:
    length = int(length)
    seed = int(seed)
    epoch = int(epoch)

    require(length > 0, "Dataset length 必须 > 0")
    require(epoch >= 0, "epoch 必须 >= 0")

    keyed = [(_shuffle_key(seed, epoch, i), i) for i in range(length)]

    # digest collision 在实际中几乎不可能；仍显式检查，避免排序 tie 的隐含歧义。
    digests = [x[0] for x in keyed]
    require(
        len(set(digests)) == length,
        "SHA256 shuffle key 出现 collision，拒绝继续"
    )

    keyed.sort(key=lambda x: (x[0], x[1]))
    order = [i for _, i in keyed]

    require(len(order) == length, "permutation 长度错误")
    require(set(order) == set(range(length)),
            "permutation 不是 0..length-1 的完整排列")
    return order


def permutation_sha256(order: Sequence[int]) -> str:
    # 使用 JSON canonical bytes，未来脚本语言/平台也容易复现。
    return sha256_json_canonical([int(x) for x in order])


class DeterministicEpochSampler(Sampler[int]):
    """
    单 GPU / 单进程训练 sampler。

    每个 epoch 必须先调用 set_epoch(epoch)。
    Sampler 的顺序只依赖：
        seed
        epoch
        len(dataset)

    不依赖 DataLoader worker RNG。
    """

    def __init__(
        self,
        data_source,
        seed: int = DATA_SEED,
        epoch: int = 0,
    ):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        require(epoch >= 0, "sampler epoch 必须 >= 0")
        self.epoch = epoch

    def get_order(self) -> List[int]:
        return deterministic_epoch_permutation(
            len(self.data_source),
            self.seed,
            self.epoch,
        )

    def order_sha256(self) -> str:
        return permutation_sha256(self.get_order())

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_order())

    def __len__(self) -> int:
        return len(self.data_source)


# ---------------------------------------------------------------------
# worker seeding
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DeterministicWorkerSeeder:
    """
    Dataset 本身的 crop/D4 已经 stateless，不依赖 worker RNG。

    这里仍对 Python / NumPy / torch worker RNG 做确定性初始化，
    防止以后无意加入某个 worker-local 随机操作时完全失控。

    该 seed 不参与当前 Dataset crop/D4 定义。
    """
    base_seed: int
    epoch: int

    def __call__(self, worker_id: int) -> None:
        payload = (
            f"{int(self.base_seed)}|worker_seed|"
            f"{int(self.epoch)}|{int(worker_id)}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()

        # NumPy legacy seed 要求 uint32。
        seed32 = int.from_bytes(digest[:4], "big", signed=False)
        seed64 = int.from_bytes(digest[:8], "big", signed=False)

        random.seed(seed64)
        np.random.seed(seed32)
        torch.manual_seed(seed64)


# ---------------------------------------------------------------------
# Train DataLoader factory
# ---------------------------------------------------------------------

def build_train_dataloader(
    dataset: PotsdamTrainDataset,
    sampler: DeterministicEpochSampler,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    """
    科研协议固定：
      sampler = DeterministicEpochSampler
      shuffle = False（因为 sampler 已经决定顺序）
      drop_last = False
      persistent_workers = False

    batch_size / num_workers / pin_memory 是运行参数，不在此阶段科研冻结。
    """
    batch_size = int(batch_size)
    num_workers = int(num_workers)

    require(batch_size > 0, "batch_size 必须 > 0")
    require(num_workers >= 0, "num_workers 必须 >= 0")
    require(len(dataset) == len(sampler),
            "dataset 与 sampler 长度不一致")
    require(dataset.epoch == sampler.epoch,
            "dataset.epoch 与 sampler.epoch 不一致；"
            "请先调用 set_train_epoch()")

    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=DROP_LAST,
        persistent_workers=(PERSISTENT_WORKERS if num_workers > 0 else False),
        worker_init_fn=DeterministicWorkerSeeder(
            base_seed=sampler.seed,
            epoch=sampler.epoch,
        ),
    )

    return DataLoader(**kwargs)


def set_train_epoch(
    dataset: PotsdamTrainDataset,
    sampler: DeterministicEpochSampler,
    epoch: int,
) -> None:
    """
    每个 epoch 开始前必须同时更新 Dataset 与 Sampler。

    persistent_workers=False 是故意的：
    worker 会在新的 iterator 启动时从当前 Dataset 状态重新建立，
    避免 worker 持有旧 epoch 的 Dataset 副本。
    """
    epoch = int(epoch)
    dataset.set_epoch(epoch)
    sampler.set_epoch(epoch)
    require(dataset.epoch == sampler.epoch == epoch,
            "Dataset/Sampler epoch 同步失败")


# ---------------------------------------------------------------------
# audit helpers
# ---------------------------------------------------------------------

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


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _uncollate_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    只用于 smoke-test，把 default_collate 后的 batch 拆回逐 item summary。
    """
    batch_size = int(batch["rgb"].shape[0])
    rows: List[Dict[str, Any]] = []

    for i in range(batch_size):
        tile_id = batch["tile_id"][i]
        if not isinstance(tile_id, str):
            tile_id = str(tile_id)

        row = {
            "tile_id": tile_id,
            "epoch": int(batch["epoch"][i].item()),
            "sample_slot": int(batch["sample_slot"][i].item()),
            "x": int(batch["x"][i].item()),
            "y": int(batch["y"][i].item()),
            "d4_code": int(batch["d4_code"][i].item()),
            "candidate_id": int(batch["candidate_id"][i].item()),
            "fallback": bool(batch["fallback"][i].item()),
            "dominant_ratio": float(batch["dominant_ratio"][i].item()),
            "rgb_sha256": tensor_sha256(batch["rgb"][i]),
            "nir_sha256": tensor_sha256(batch["nir"][i]),
            "label_sha256": tensor_sha256(batch["labels"][i]),
        }
        rows.append(row)

    return rows


def collect_loader_probe(
    loader: DataLoader,
    max_items: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        rows.extend(_uncollate_batch(batch))
        if len(rows) >= max_items:
            return rows[:max_items]
    return rows


def index_metadata_from_dataset(
    dataset: PotsdamTrainDataset,
    order: Sequence[int],
    count: int,
) -> List[Dict[str, Any]]:
    """
    不读取图像，仅把 sampler index 映射成 tile/slot，便于报告首个顺序。
    """
    out = []
    for rank, dataset_index in enumerate(order[:count]):
        tile_id, slot = dataset.index_to_tile_slot(int(dataset_index))
        out.append({
            "rank": rank,
            "dataset_index": int(dataset_index),
            "tile_id": tile_id,
            "sample_slot": int(slot),
        })
    return out


def _freeze_protocol(
    path: Path,
    protocol: Dict[str, Any],
) -> Dict[str, str]:
    new_hash = sha256_json_canonical(protocol)

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_hash = sha256_json_canonical(existing)
        require(
            old_hash == new_hash,
            "dataloader_protocol.json 已存在但内容与本次结果不同；"
            "拒绝静默覆盖，请先人工审查。"
        )
        return {
            "action": "verified_existing",
            "sha256": sha256_file(path),
        }

    _write_json(path, protocol)
    return {
        "action": "created",
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------
# smoke-test
# ---------------------------------------------------------------------

def run_smoke_test(project_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()

    dataset_file = project_root / "data_pipeline/potsdam_dataset.py"
    dataset_protocol_file = (
        project_root / "data/processed/potsdam/dataset_protocol.json"
    )

    require(dataset_file.is_file(),
            f"缺少正式 Dataset 文件：{dataset_file}")
    require(dataset_protocol_file.is_file(),
            f"缺少冻结 dataset_protocol.json：{dataset_protocol_file}")

    print("=" * 72)
    print("Potsdam DataLoader / epoch shuffle reproducibility audit")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("本测试不会启动模型或训练。")
    print()

    print("[1/5] 加载正式 Train Dataset...")
    dataset = PotsdamTrainDataset(project_root, epoch=0)
    require(len(dataset) == EXPECTED_TRAIN_LENGTH,
            f"Train Dataset len={len(dataset)} != 1152")

    sampler = DeterministicEpochSampler(
        dataset,
        seed=DATA_SEED,
        epoch=0,
    )
    require(len(sampler) == 1152, "Sampler len != 1152")
    print("  PASS: Train Dataset / Sampler length = 1152")

    print("[2/5] 审计 epoch 0/1/2 deterministic shuffle...")
    epoch_rows: Dict[str, Any] = {}
    epoch_hashes: List[str] = []

    for epoch in AUDIT_EPOCHS:
        set_train_epoch(dataset, sampler, epoch)
        order_a = sampler.get_order()
        order_b = sampler.get_order()

        require(order_a == order_b,
                f"epoch {epoch}: permutation 重复生成不一致")
        require(len(order_a) == 1152,
                f"epoch {epoch}: permutation length != 1152")
        require(len(set(order_a)) == 1152,
                f"epoch {epoch}: permutation 存在重复 index")
        require(set(order_a) == set(range(1152)),
                f"epoch {epoch}: permutation 非完整 0..1151")

        order_hash = sampler.order_sha256()
        require(order_hash == permutation_sha256(order_a),
                f"epoch {epoch}: order SHA256 自检失败")
        epoch_hashes.append(order_hash)

        preview = index_metadata_from_dataset(dataset, order_a, 12)
        epoch_rows[str(epoch)] = {
            "order_sha256": order_hash,
            "first_12": preview,
        }

        first_str = ", ".join(
            f"{r['dataset_index']}({r['tile_id']}/s{r['sample_slot']})"
            for r in preview[:6]
        )
        print(f"  epoch {epoch}: {order_hash}")
        print(f"    first 6: {first_str}")

    require(len(set(epoch_hashes)) == len(AUDIT_EPOCHS),
            "epoch 0/1/2 得到了相同 permutation hash，异常")
    print("  PASS: 每个 epoch 是完整 permutation，重复生成一致，跨 epoch 顺序不同。")

    print("[3/5] 验证 DataLoader 不丢 sample...")
    # 用纯 sampler/order 数学检查，不必读取全部 1152 张 patch。
    # drop_last=False，因此任意 batch_size 下都应保留完整 sampler order。
    for audit_batch_size in (1, 2, 7, 32, 100):
        expected_batches = (
            len(dataset) + audit_batch_size - 1
        ) // audit_batch_size
        reconstructed_count = sum(
            min(audit_batch_size, len(dataset) - b * audit_batch_size)
            for b in range(expected_batches)
        )
        require(reconstructed_count == 1152,
                f"batch_size={audit_batch_size}: drop_last=False 数学检查失败")
    require(DROP_LAST is False, "DROP_LAST 必须为 False")
    require(PERSISTENT_WORKERS is False,
            "PERSISTENT_WORKERS 必须为 False")
    print("  PASS: drop_last=False；1152 个 sample 全部保留。")
    print("  PASS: persistent_workers=False，epoch 切换不会保留旧 Dataset worker 状态。")

    print("[4/5] 真实 DataLoader：num_workers=0 vs 2 内容与顺序对比...")
    # 固定 epoch 0 做 worker invariance 检查。
    dataset0 = PotsdamTrainDataset(project_root, epoch=0)
    sampler0 = DeterministicEpochSampler(dataset0, seed=DATA_SEED, epoch=0)
    set_train_epoch(dataset0, sampler0, 0)

    loader0 = build_train_dataloader(
        dataset0,
        sampler0,
        batch_size=AUDIT_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    rows0 = collect_loader_probe(loader0, AUDIT_COMPARE_ITEMS)
    require(len(rows0) == AUDIT_COMPARE_ITEMS,
            "num_workers=0 probe item 数不足")

    dataset2 = PotsdamTrainDataset(project_root, epoch=0)
    sampler2 = DeterministicEpochSampler(dataset2, seed=DATA_SEED, epoch=0)
    set_train_epoch(dataset2, sampler2, 0)

    loader2 = build_train_dataloader(
        dataset2,
        sampler2,
        batch_size=AUDIT_BATCH_SIZE,
        num_workers=AUDIT_MULTIWORKER_COUNT,
        pin_memory=False,
    )
    rows2 = collect_loader_probe(loader2, AUDIT_COMPARE_ITEMS)
    require(len(rows2) == AUDIT_COMPARE_ITEMS,
            "num_workers=2 probe item 数不足")

    require(
        rows0 == rows2,
        "num_workers=0 与 num_workers=2 的前 8 个真实样本内容/顺序不一致"
    )
    print(
        f"  PASS: 前 {AUDIT_COMPARE_ITEMS} 个真实样本在 "
        "num_workers=0 与 2 下 metadata/tensor SHA256 完全一致。"
    )

    print("[5/5] 冻结 DataLoader protocol metadata...")
    # 再回到 epoch 0，构造 protocol。
    set_train_epoch(dataset, sampler, 0)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "module_version": MODULE_VERSION,
        "status": "FROZEN_AFTER_AUDIT_PASS",
        "source": {
            "dataset_module": "data_pipeline/potsdam_dataset.py",
            "dataset_module_version": DATASET_MODULE_VERSION,
            "dataset_module_sha256": sha256_file(dataset_file),
            "dataset_protocol": "data/processed/potsdam/dataset_protocol.json",
            "dataset_protocol_sha256": sha256_file(dataset_protocol_file),
        },
        "train_order": {
            "data_seed": DATA_SEED,
            "dataset_length": 1152,
            "algorithm": (
                "sort dataset indices by "
                "SHA256(f'{seed}|epoch_shuffle|{epoch}|{index}') digest, "
                "then index as deterministic tie-break"
            ),
            "stateless": True,
            "complete_permutation_each_epoch": True,
            "audit_epochs": {
                str(epoch): epoch_rows[str(epoch)]["order_sha256"]
                for epoch in AUDIT_EPOCHS
            },
        },
        "dataloader_invariants": {
            "shuffle_argument": False,
            "sampler": "DeterministicEpochSampler",
            "drop_last": False,
            "persistent_workers": False,
            "worker_rng": (
                "DeterministicWorkerSeeder(seed, epoch, worker_id); "
                "current Dataset crop/D4 does not depend on worker RNG"
            ),
            "dataset_and_sampler_epoch_must_match": True,
        },
        "runtime_not_frozen_here": {
            "batch_size": True,
            "num_workers": True,
            "pin_memory": True,
        },
        "audit": {
            "epochs_checked": list(AUDIT_EPOCHS),
            "worker_invariance": {
                "compared_num_workers": [0, AUDIT_MULTIWORKER_COUNT],
                "batch_size_for_audit_only": AUDIT_BATCH_SIZE,
                "items_compared": AUDIT_COMPARE_ITEMS,
                "exact_metadata_and_tensor_hash_match": True,
            },
        },
    }

    protocol_path = (
        project_root / "data/processed/potsdam/dataloader_protocol.json"
    )
    freeze_info = _freeze_protocol(protocol_path, protocol)

    print(f"  dataloader_protocol.json: {freeze_info['action']}")
    print()
    print("FINAL STATUS: PASS")

    return {
        "status": "PASS",
        "module_version": MODULE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_module_version": DATASET_MODULE_VERSION,
        "dataset_protocol_sha256": sha256_file(dataset_protocol_file),
        "dataset_module_sha256": sha256_file(dataset_file),
        "train_length": 1152,
        "epochs": epoch_rows,
        "drop_last": DROP_LAST,
        "persistent_workers": PERSISTENT_WORKERS,
        "worker_invariance": {
            "num_workers_0_vs_2": True,
            "items_compared": AUDIT_COMPARE_ITEMS,
            "probe_rows": rows0,
        },
        "frozen_dataloader_protocol": {
            "path": "data/processed/potsdam/dataloader_protocol.json",
            **freeze_info,
        },
    }


def report_to_text(report: Dict[str, Any]) -> str:
    if report.get("status") != "PASS":
        return (
            "Potsdam DataLoader audit\n"
            + "=" * 72
            + "\n状态: FAIL\n"
            + f"error_type: {report.get('error_type')}\n"
            + f"error: {report.get('error')}\n"
        )

    lines = [
        "Potsdam DataLoader / epoch shuffle reproducibility audit",
        "=" * 72,
        "状态: PASS",
        f"模块版本: {report['module_version']}",
        f"protocol: {report['protocol_version']}",
        "",
        "[Train]",
        f"dataset length = {report['train_length']}",
        f"drop_last = {report['drop_last']}",
        f"persistent_workers = {report['persistent_workers']}",
        "",
        "[Epoch order SHA256]",
    ]

    for epoch in AUDIT_EPOCHS:
        lines.append(
            f"epoch {epoch} = "
            f"{report['epochs'][str(epoch)]['order_sha256']}"
        )

    lines.extend([
        "",
        "[Worker invariance]",
        f"num_workers 0 vs 2 = "
        f"{report['worker_invariance']['num_workers_0_vs_2']}",
        f"items compared = "
        f"{report['worker_invariance']['items_compared']}",
        "",
        "[Frozen metadata]",
        f"path = {report['frozen_dataloader_protocol']['path']}",
        f"action = {report['frozen_dataloader_protocol']['action']}",
        f"sha256 = {report['frozen_dataloader_protocol']['sha256']}",
        "",
        "FINAL STATUS: PASS",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Potsdam DataLoader / epoch shuffle reproducibility audit"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录；默认 data_pipeline/ 的父目录。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行真实 Dataset + DataLoader 审计；不启动模型或训练。",
    )
    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "DataLoader 模块已加载。运行审计：\n"
            "  python data_pipeline/potsdam_dataloader.py --smoke-test"
        )
        return 0

    project_root = args.project_root.resolve()
    out_dir = project_root / "outputs/dataset_check/dataloader_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    console_path = out_dir / "console.txt"
    json_path = out_dir / "dataloader_protocol_audit.json"
    txt_path = out_dir / "dataloader_protocol_audit.txt"

    with console_path.open("w", encoding="utf-8", buffering=1) as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)

        try:
            report = run_smoke_test(project_root)
            exit_code = 0
        except Exception as e:
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

    _write_json(json_path, report)
    txt_path.write_text(report_to_text(report), encoding="utf-8")

    print(f"console: {console_path}")
    print(f"json: {json_path}")
    print(f"txt: {txt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
