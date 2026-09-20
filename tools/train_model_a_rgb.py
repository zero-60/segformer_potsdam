#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A (RGB baseline) - formal training script for the ISPRS Potsdam experiment.

Purpose
-------
Train the RGB-only SegFormer-B0 baseline under a frozen protocol so later
RGB+NIR / Fixed Fusion / Quality Gate models can be compared fairly.

Default protocol
----------------
- Input: RGB only (3 channels)
- Classes: 6
- Ignore label: 255
- Epochs: 100
- Mini-batch size: 2
- Gradient accumulation: 8
- Effective batch size: 16 (except a possible final partial group)
- Optimizer: AdamW
- Base LR: 6e-5
- Weight decay: 0.01
- Warmup: 360 optimizer updates
- LR policy: linear warmup + linear polynomial decay (power=1.0)
- AMP: enabled on CUDA
- Gradient clipping: 1.0
- DataLoader workers: 4 by default (runtime-only; sample invariance is audited)

Key fixes over the earlier debug version
----------------------------------------
1. The final incomplete gradient-accumulation group is no longer discarded.
2. total optimizer-update steps use ceil(len(loader) / grad_accum_steps).
3. The last partial group is normalized by its actual accumulation size.
4. Logged loss is the real, unscaled cross-entropy loss.
5. Resume restores model / optimizer / scheduler / scaler / RNG state.
6. Checkpoints are written atomically.
7. First-batch checks enforce a clean RGB-only baseline and valid label range.
8. Non-finite loss and output-shape mistakes fail early with clear messages.
9. Per-epoch JSONL logging records loss, LR, update count and valid pixels.
10. Repository root is injected into sys.path, so PYTHONPATH=. is not required.
11. Compressed train TIFFs are decoded once into mmap-ready .npy runtime cache.
12. DataLoader defaults to multiple workers + pinned memory without changing
    the project's frozen deterministic sampler/sample-content protocol.

Expected project layout
-----------------------
repo_root/
├── src/
│   ├── train_model_a_rgb.py          <- this file
│   ├── data_pipeline/
│   │   ├── potsdam_dataset.py
│   │   └── potsdam_dataloader.py
│   └── models/
│       └── segformer_rgb.py
└── outputs/

The existing project interfaces are intentionally preserved:
- PotsdamTrainDataset(PROJECT_ROOT, epoch=0)
- DeterministicEpochSampler(...)
- build_train_dataloader(...)
- set_train_epoch(...)
- build_model_a_rgb(PROJECT_ROOT) -> (model, meta)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Frozen experiment constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# When this file is executed as:
#     python tools/train_model_a_rgb.py
# Python otherwise places only tools/ on sys.path.  Add the repository root
# explicitly so `data_pipeline` and `models` are importable without requiring
# `PYTHONPATH=.`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.potsdam_dataset import PotsdamTrainDataset

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training" / "model_a_rgb"
DEFAULT_DATA_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "potsdam" / "train_npy_mmap_cache_v1"
)
DEFAULT_NUM_WORKERS = min(4, max(1, os.cpu_count() or 1))

MODEL_NAME = "Model A (RGB baseline)"
NUM_CLASSES = 6
IGNORE_INDEX = 255
DEFAULT_SEED = 20260917

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 8
DEFAULT_LR = 6e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 360
DEFAULT_GRAD_CLIP = 1.0



class _NpyMMapStore:
    """
    Small per-process LRU of read-only NumPy memmaps.

    The original Potsdam TIFF files in this project can be compressed and are
    therefore not memory-mappable by tifffile.  Re-reading a 6000x6000 tile for
    every randomly shuffled crop stalls the GPU.  The cache built below stores
    the *same decoded uint8 pixels* once as an uncompressed .npy file.  NumPy
    can then mmap the file and every DataLoader worker reads only the requested
    crop pages.

    This is a runtime I/O optimization only.  It does not change:
      - train tile split
      - deterministic crop coordinates
      - D4 augmentation
      - RGB normalization
      - labels
      - sampler order
    """

    def __init__(self, max_open: int = 32):
        self.max_open = int(max_open)
        self._open: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle live mmap handles into DataLoader workers.
        state["_open"] = OrderedDict()
        return state

    @staticmethod
    def _close_array(arr: np.ndarray) -> None:
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass

    def clear(self) -> None:
        while self._open:
            _, arr = self._open.popitem(last=False)
            self._close_array(arr)

    def read(self, path: Path) -> np.ndarray:
        key = str(path.resolve())

        if key in self._open:
            arr = self._open.pop(key)
            self._open[key] = arr
            return arr

        arr = np.load(str(path), mmap_mode="r", allow_pickle=False)

        if len(self._open) >= self.max_open:
            _, old = self._open.popitem(last=False)
            self._close_array(old)

        self._open[key] = arr
        return arr


def _source_fingerprint(path: Path) -> str:
    """
    Fingerprint a raw source without hashing its entire multi-hundred-MB body.

    Raw Potsdam files are frozen experiment inputs.  Path + size + mtime_ns is
    enough to prevent accidentally reusing a cache after a local source file
    has been replaced or modified.
    """
    stat = path.stat()
    payload = (
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_valid_cache(
    path: Path,
    *,
    kind: str,
    tile_size: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        arr = np.load(str(path), mmap_mode="r", allow_pickle=False)

        valid = (
            arr.dtype == np.uint8
            and arr.ndim == 3
            and arr.shape[0] == tile_size
            and arr.shape[1] == tile_size
        )

        if kind == "rgbir":
            valid = valid and arr.shape[2] == 4
        elif kind == "gt":
            valid = valid and arr.shape[2] in (3, 4)
        else:
            raise ValueError(f"Unknown cache kind: {kind}")

        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

        return bool(valid)

    except Exception:
        return False


def _decode_raw_tile(
    source: Path,
    *,
    kind: str,
    tile_size: int,
) -> np.ndarray:
    """
    Decode one raw TIFF exactly once and normalize only array axis order.

    No numeric transform is applied: values and dtype remain uint8.
    """
    arr = tifffile.imread(str(source))

    if arr.ndim != 3:
        raise RuntimeError(
            f"{kind} TIFF must be 3D, got {arr.shape}: {source}"
        )

    if kind == "rgbir":
        if arr.shape[0] == 4 and arr.shape[-1] != 4:
            arr = np.moveaxis(arr, 0, -1)

        expected = (tile_size, tile_size, 4)
        if arr.shape != expected:
            raise RuntimeError(
                f"RGBIR shape mismatch for {source}: "
                f"expected {expected}, got {arr.shape}"
            )

    elif kind == "gt":
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

        if not (
            arr.shape[0] == tile_size
            and arr.shape[1] == tile_size
            and arr.shape[2] in (3, 4)
        ):
            raise RuntimeError(
                f"GT shape mismatch for {source}: got {arr.shape}"
            )
    else:
        raise ValueError(f"Unknown cache kind: {kind}")

    if arr.dtype != np.uint8:
        raise RuntimeError(
            f"{kind} dtype must be uint8, got {arr.dtype}: {source}"
        )

    return np.ascontiguousarray(arr)


class CachedPotsdamTrainDataset(PotsdamTrainDataset):
    """
    PotsdamTrainDataset with a transparent mmap cache for raw full-tile arrays.

    All sampling/augmentation/normalization logic remains inherited from the
    project's frozen PotsdamTrainDataset.  Only `_read_rgbir` and `_read_gt_rgb`
    are replaced so the inherited __getitem__ slices mmap-backed arrays instead
    of repeatedly decompressing TIFF files.
    """

    CACHE_VERSION = "npy-mmap-v1"

    def __init__(
        self,
        project_root: Path | str,
        *,
        epoch: int = 0,
        cache_root: Path,
    ):
        super().__init__(project_root=project_root, epoch=epoch)

        self.cache_root = Path(cache_root).resolve()
        self.rgbir_cache_dir = self.cache_root / "rgbir"
        self.gt_cache_dir = self.cache_root / "gt"

        self._rgbir_cache_paths: Dict[str, Path] = {}
        self._gt_cache_paths: Dict[str, Path] = {}

        self._rgbir_npy_store = _NpyMMapStore(max_open=32)
        self._gt_npy_store = _NpyMMapStore(max_open=32)

    def __getstate__(self):
        state = super().__getstate__()
        # Explicitly drop any live mmap handles if the main process happened to
        # touch the dataset before DataLoader workers are created.
        state["_rgbir_npy_store"] = _NpyMMapStore(max_open=32)
        state["_gt_npy_store"] = _NpyMMapStore(max_open=32)
        return state

    def _cache_path(
        self,
        *,
        tile_id: str,
        source: Path,
        kind: str,
    ) -> Path:
        fingerprint = _source_fingerprint(source)
        directory = (
            self.rgbir_cache_dir if kind == "rgbir" else self.gt_cache_dir
        )
        return directory / f"{tile_id}__{fingerprint}.npy"

    def _prepare_one(
        self,
        *,
        tile_id: str,
        source: Path,
        kind: str,
        rebuild: bool,
    ) -> tuple[Path, bool]:
        cache_path = self._cache_path(
            tile_id=tile_id,
            source=source,
            kind=kind,
        )

        if (not rebuild) and _load_valid_cache(
            cache_path,
            kind=kind,
            tile_size=self.spec.tile_size,
        ):
            return cache_path, False

        if cache_path.exists():
            cache_path.unlink()

        arr = _decode_raw_tile(
            source,
            kind=kind,
            tile_size=self.spec.tile_size,
        )
        _atomic_save_npy(cache_path, arr)

        # Drop the full decoded array immediately; training will reopen the
        # .npy as a read-only mmap instead of keeping 100+ MB in Python RAM.
        del arr

        if not _load_valid_cache(
            cache_path,
            kind=kind,
            tile_size=self.spec.tile_size,
        ):
            raise RuntimeError(
                f"Created cache failed validation: {cache_path}"
            )

        return cache_path, True

    def _estimate_required_cache_bytes(self) -> int:
        """
        Conservative upper bound:
          RGBIR = H*W*4 uint8
          GT    <= H*W*4 uint8
        for every train tile.
        """
        pixels = int(self.spec.tile_size) * int(self.spec.tile_size)
        return len(self.tile_ids) * pixels * 8

    def prepare_cache(self, *, rebuild: bool = False) -> None:
        self.rgbir_cache_dir.mkdir(parents=True, exist_ok=True)
        self.gt_cache_dir.mkdir(parents=True, exist_ok=True)

        # Only enforce the disk check when a substantial number of current
        # caches are missing. Existing valid caches consume no additional space.
        missing = 0
        for tile_id in self.tile_ids:
            rgbir_source = Path(self.spec.rgbir_paths[tile_id])
            gt_source = Path(self.spec.gt_paths[tile_id])

            rgbir_path = self._cache_path(
                tile_id=tile_id,
                source=rgbir_source,
                kind="rgbir",
            )
            gt_path = self._cache_path(
                tile_id=tile_id,
                source=gt_source,
                kind="gt",
            )

            if rebuild or not _load_valid_cache(
                rgbir_path,
                kind="rgbir",
                tile_size=self.spec.tile_size,
            ):
                missing += 1

            if rebuild or not _load_valid_cache(
                gt_path,
                kind="gt",
                tile_size=self.spec.tile_size,
            ):
                missing += 1

        if missing:
            total_items = max(1, len(self.tile_ids) * 2)
            conservative_total = self._estimate_required_cache_bytes()
            conservative_missing = math.ceil(
                conservative_total * (missing / total_items)
            )
            free = shutil.disk_usage(self.cache_root).free

            # Keep 1 GiB free after the conservative estimate.
            required_with_headroom = conservative_missing + (1 << 30)
            if free < required_with_headroom:
                raise RuntimeError(
                    "Not enough free disk space for the mmap data cache. "
                    f"free={free / (1024**3):.2f} GiB, "
                    f"estimated_required_with_headroom="
                    f"{required_with_headroom / (1024**3):.2f} GiB. "
                    "Use --data-cache-dir on a larger disk, or "
                    "--disable-data-cache (slower)."
                )

        created = 0
        reused = 0
        total = len(self.tile_ids)

        print(
            f"[cache] preparing {total} train tiles at: {self.cache_root}"
        )
        print(
            "[cache] first run decodes each compressed TIFF once; "
            "later runs reuse mmap-ready .npy files."
        )

        for idx, tile_id in enumerate(self.tile_ids, start=1):
            print(
                f"[cache] tile {idx:02d}/{total:02d} {tile_id} | RGBIR",
                flush=True,
            )
            rgbir_path, was_created = self._prepare_one(
                tile_id=tile_id,
                source=Path(self.spec.rgbir_paths[tile_id]),
                kind="rgbir",
                rebuild=rebuild,
            )
            self._rgbir_cache_paths[tile_id] = rgbir_path
            created += int(was_created)
            reused += int(not was_created)

            print(
                f"[cache] tile {idx:02d}/{total:02d} {tile_id} | GT",
                flush=True,
            )
            gt_path, was_created = self._prepare_one(
                tile_id=tile_id,
                source=Path(self.spec.gt_paths[tile_id]),
                kind="gt",
                rebuild=rebuild,
            )
            self._gt_cache_paths[tile_id] = gt_path
            created += int(was_created)
            reused += int(not was_created)

        print(
            f"[cache] ready | created={created}, reused={reused}, "
            f"format={self.CACHE_VERSION}"
        )

    def _require_cache_path(
        self,
        mapping: Dict[str, Path],
        tile_id: str,
        kind: str,
    ) -> Path:
        if tile_id not in mapping:
            raise RuntimeError(
                f"{kind} cache for tile {tile_id} was not prepared. "
                "prepare_cache() must run before DataLoader creation."
            )
        return mapping[tile_id]

    def _read_rgbir(self, tile_id: str) -> np.ndarray:
        path = self._require_cache_path(
            self._rgbir_cache_paths,
            tile_id,
            "RGBIR",
        )
        arr = self._rgbir_npy_store.read(path)

        expected = (
            self.spec.tile_size,
            self.spec.tile_size,
            4,
        )
        if arr.shape != expected or arr.dtype != np.uint8:
            raise RuntimeError(
                f"{tile_id}: cached RGBIR invalid: "
                f"shape={arr.shape}, dtype={arr.dtype}"
            )
        return arr

    def _read_gt_rgb(self, tile_id: str) -> np.ndarray:
        path = self._require_cache_path(
            self._gt_cache_paths,
            tile_id,
            "GT",
        )
        arr = self._gt_npy_store.read(path)

        if not (
            arr.ndim == 3
            and arr.shape[0] == self.spec.tile_size
            and arr.shape[1] == self.spec.tile_size
            and arr.shape[2] in (3, 4)
            and arr.dtype == np.uint8
        ):
            raise RuntimeError(
                f"{tile_id}: cached GT invalid: "
                f"shape={arr.shape}, dtype={arr.dtype}"
            )
        return arr

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Model A: RGB-only SegFormer-B0 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Frozen protocol defaults. They remain CLI-overridable for debugging only.
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # With the mmap cache enabled, workers no longer decompress whole TIFF
    # tiles. Four workers + pinned memory is a conservative default for a
    # single RTX 4080-class GPU and can be changed without altering the frozen
    # sample order/content protocol.
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU batches for asynchronous CUDA transfer.",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=DEFAULT_DATA_CACHE_DIR,
        help=(
            "Directory for mmap-ready .npy copies of the 18 train RGBIR/GT "
            "tiles. Relative paths are resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--rebuild-data-cache",
        action="store_true",
        help="Re-decode the current raw train TIFF files and replace their caches.",
    )
    parser.add_argument(
        "--disable-data-cache",
        action="store_true",
        help=(
            "Use the original TIFF reader directly. This is correct but can be "
            "extremely slow for compressed, non-mappable Potsdam TIFF files."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Examples: "cuda", "cuda:0", "cpu".',
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help='Checkpoint path, or "auto" to use <output-dir>/checkpoints/latest.pt.',
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also keep epoch_XXX.pt every N epochs. 0 disables archival checkpoints.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training status every N mini-batches.",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.grad_clip <= 0:
        parser.error("--grad-clip must be > 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.save_every < 0:
        parser.error("--save-every must be >= 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    return args


def seed_everything(seed: int) -> None:
    """
    Seed the RNGs used by this training process.

    PYTHONHASHSEED is recorded for child processes / reproducibility metadata.
    The dataset/sampler already receive their own deterministic epoch handling
    through set_train_epoch(...).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Stable deterministic cuDNN behavior where applicable.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_poly_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
) -> LambdaLR:
    """
    Linear warmup followed by polynomial decay to zero.

    With power=1.0, the decay is linear. This keeps the behavior of the
    previous script while making edge cases explicit and safe.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    warmup_steps = max(0, min(int(warmup_steps), max(0, total_steps - 1)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, (1.0 - progress) ** power)

    return LambdaLR(optimizer, lr_lambda)


def get_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu only for debugging, or fix the CUDA environment."
        )

    return device


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_resume_path(
    resume_arg: str,
    checkpoint_dir: Path,
) -> Optional[Path]:
    if not resume_arg:
        return None

    if resume_arg.lower() == "auto":
        path = checkpoint_dir / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(
                f'--resume auto requested, but checkpoint does not exist: {path}'
            )
        return path

    path = Path(resume_arg).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")

    return path


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def protocol_from_args(
    args: argparse.Namespace,
    updates_per_epoch: int,
    total_update_steps: int,
) -> Dict[str, Any]:
    return {
        "model": "A_RGB",
        "input_modalities": ["RGB"],
        "num_input_channels": 3,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size_nominal": args.batch_size * args.grad_accum_steps,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps_requested": args.warmup_steps,
        "scheduler": "linear_warmup_then_poly_decay",
        "scheduler_power": 1.0,
        "grad_clip": args.grad_clip,
        "amp_requested": not args.no_amp,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "updates_per_epoch": updates_per_epoch,
        "total_update_steps": total_update_steps,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def make_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    protocol: Dict[str, Any],
    model_meta: Any,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "model_name": MODEL_NAME,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        # Backward-compatible alias with the earlier script.
        "step": int(global_step),
        "protocol": protocol,
        "model_meta_repr": repr(model_meta),
        "rng_state": capture_rng_state(),
    }


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    """
    Restore a checkpoint.

    Returns:
        start_epoch, global_step
    """
    print(f"[resume] loading checkpoint: {path}")

    # PyTorch >= 2.6 defaults weights_only=True. This is our own trusted
    # training checkpoint and it also contains RNG / optimizer Python objects,
    # so weights_only=False is required.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=True)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    restore_rng_state(checkpoint.get("rng_state"))

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))

    if "protocol" not in checkpoint:
        print(
            "[resume] warning: legacy checkpoint has no protocol metadata; "
            "resume is allowed, but exact protocol matching cannot be verified."
        )

    print(
        f"[resume] restored epoch={start_epoch}, "
        f"global_step={global_step}"
    )
    return start_epoch, global_step


def validate_rgb_baseline_batch(
    rgb: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if rgb.ndim != 4:
        raise RuntimeError(
            f"Expected RGB tensor [B, 3, H, W], got shape={tuple(rgb.shape)}"
        )
    if rgb.shape[1] != 3:
        raise RuntimeError(
            "Model A must be RGB-only. "
            f"Expected exactly 3 input channels, got {rgb.shape[1]}."
        )
    if labels.ndim != 3:
        raise RuntimeError(
            f"Expected labels [B, H, W], got shape={tuple(labels.shape)}"
        )
    if tuple(rgb.shape[-2:]) != tuple(labels.shape[-2:]):
        raise RuntimeError(
            "RGB/label spatial shapes differ: "
            f"rgb={tuple(rgb.shape)}, labels={tuple(labels.shape)}"
        )

    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        raise RuntimeError("The first batch contains no valid semantic labels.")

    valid_labels = labels[valid]
    label_min = int(valid_labels.min().item())
    label_max = int(valid_labels.max().item())

    if label_min < 0 or label_max >= NUM_CLASSES:
        raise RuntimeError(
            "Unexpected Potsdam label id. "
            f"Expected valid labels in [0, {NUM_CLASSES - 1}] "
            f"plus ignore_index={IGNORE_INDEX}, "
            f"but found valid range [{label_min}, {label_max}]."
        )

    print(
        "[check] first batch OK | "
        f"rgb={tuple(rgb.shape)} {rgb.dtype} | "
        f"labels={tuple(labels.shape)} {labels.dtype} | "
        f"valid_label_range=[{label_min}, {label_max}]"
    )


def validate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "build_model_a_rgb(...).forward(rgb) must return a Tensor of logits. "
            f"Got {type(logits)!r}."
        )
    if logits.ndim != 4:
        raise RuntimeError(
            f"Expected logits [B, C, H, W], got shape={tuple(logits.shape)}"
        )
    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} output classes, got {logits.shape[1]}."
        )
    if tuple(logits.shape[-2:]) != tuple(labels.shape[-2:]):
        raise RuntimeError(
            "Logits and labels must have the same spatial resolution before "
            "cross-entropy. The model wrapper should upsample logits if needed. "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(MODEL_NAME)
    print("=" * 78)
    print(f"project root : {PROJECT_ROOT}")
    print(f"output dir   : {output_dir}")

    print("[1] importing project modules")
    from data_pipeline.potsdam_dataloader import (
        DATA_SEED,
        DeterministicEpochSampler,
        build_train_dataloader,
        set_train_epoch,
    )
    from models.segformer_rgb import build_model_a_rgb

    print("[2] loading training dataset")

    if args.data_cache_dir.is_absolute():
        data_cache_dir = args.data_cache_dir
    else:
        data_cache_dir = (PROJECT_ROOT / args.data_cache_dir).resolve()

    if args.disable_data_cache:
        print(
            "[warning] mmap data cache is DISABLED. Compressed Potsdam TIFF "
            "files may be fully decompressed repeatedly during shuffled training."
        )
        dataset = PotsdamTrainDataset(PROJECT_ROOT, epoch=0)
        data_cache_mode = "disabled"
    else:
        dataset = CachedPotsdamTrainDataset(
            PROJECT_ROOT,
            epoch=0,
            cache_root=data_cache_dir,
        )
        dataset.prepare_cache(rebuild=args.rebuild_data_cache)
        data_cache_mode = CachedPotsdamTrainDataset.CACHE_VERSION

    if len(dataset) <= 0:
        raise RuntimeError("Training dataset is empty.")
    print(f"dataset length: {len(dataset)}")

    sampler = DeterministicEpochSampler(
        dataset,
        seed=DATA_SEED,
        epoch=0,
    )

    print("[3] building dataloader")
    loader = build_train_dataloader(
        dataset,
        sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if len(loader) <= 0:
        raise RuntimeError("Training dataloader contains zero batches.")
    print(f"batches per epoch: {len(loader)}")

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_update_steps = updates_per_epoch * args.epochs

    if total_update_steps <= 0:
        raise RuntimeError("Computed total optimizer-update steps is zero.")

    effective_warmup_steps = min(
        args.warmup_steps,
        max(0, total_update_steps - 1),
    )

    print("[4] building model")
    device = get_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    model, model_meta = build_model_a_rgb(PROJECT_ROOT)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_poly_scheduler(
        optimizer=optimizer,
        warmup_steps=effective_warmup_steps,
        total_steps=total_update_steps,
        power=1.0,
    )

    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    protocol = protocol_from_args(
        args=args,
        updates_per_epoch=updates_per_epoch,
        total_update_steps=total_update_steps,
    )
    protocol["warmup_steps_effective"] = effective_warmup_steps
    protocol["device"] = str(device)
    protocol["amp_enabled"] = amp_enabled
    protocol["data_seed"] = int(DATA_SEED)
    protocol["model_meta_repr"] = repr(model_meta)
    protocol["data_cache_mode"] = data_cache_mode
    protocol["data_cache_dir"] = (
        str(data_cache_dir) if not args.disable_data_cache else None
    )
    protocol["data_cache_rebuilt"] = bool(args.rebuild_data_cache)

    write_json(output_dir / "protocol.json", protocol)

    print(
        "[protocol] "
        f"epochs={args.epochs}, batch={args.batch_size}, "
        f"accum={args.grad_accum_steps}, "
        f"effective_batch≈{args.batch_size * args.grad_accum_steps}, "
        f"updates/epoch={updates_per_epoch}, total_updates={total_update_steps}"
    )
    print(
        "[protocol] "
        f"lr={args.lr:g}, warmup={effective_warmup_steps}, "
        f"weight_decay={args.weight_decay:g}, "
        f"AMP={amp_enabled}, device={device}"
    )
    print(
        "[runtime] "
        f"num_workers={args.num_workers}, pin_memory={args.pin_memory}, "
        f"data_cache={data_cache_mode}"
    )

    start_epoch = 0
    global_step = 0

    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    if start_epoch >= args.epochs:
        print(
            f"[done] checkpoint already reached epoch {start_epoch}; "
            f"requested epochs={args.epochs}. Nothing to train."
        )
        return

    log_path = output_dir / "train_log.jsonl"

    print("[5] start training")
    first_batch_checked = False
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Important for deterministic per-epoch sampling / augmentation.
        set_train_epoch(dataset, sampler, epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Real (unscaled) CE statistics.
        loss_numerator = 0.0
        valid_pixel_count = 0
        mini_batches_seen = 0
        optimizer_updates_this_epoch = 0
        amp_skipped_updates = 0
        last_grad_norm = float("nan")

        accumulation_target = args.grad_accum_steps

        for i, batch in enumerate(loader):
            # At the beginning of each accumulation group, determine its real
            # size. This makes the final incomplete group mathematically valid.
            if i % args.grad_accum_steps == 0:
                remaining_batches = len(loader) - i
                accumulation_target = min(
                    args.grad_accum_steps,
                    remaining_batches,
                )

            rgb = batch["rgb"].to(
                device,
                non_blocking=args.pin_memory,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=args.pin_memory,
            ).long()

            if not first_batch_checked:
                validate_rgb_baseline_batch(rgb, labels)
                first_batch_checked = True

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(rgb)

                # Check only once per run; the architecture shape should be stable.
                if i == 0 and epoch == start_epoch:
                    validate_logits(logits, labels)
                    print(
                        "[check] model output OK | "
                        f"logits={tuple(logits.shape)} {logits.dtype}"
                    )

                raw_loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=IGNORE_INDEX,
                )

                # Normalize by the actual number of micro-batches in this
                # accumulation group, including the final partial group.
                loss_for_backward = raw_loss / accumulation_target

            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    "Non-finite training loss detected at "
                    f"epoch={epoch + 1}, batch={i}, "
                    f"loss={raw_loss.detach().item()}."
                )

            scaler.scale(loss_for_backward).backward()

            # Exact valid-pixel weighted logging of the real CE loss.
            batch_valid_pixels = int((labels != IGNORE_INDEX).sum().item())
            if batch_valid_pixels > 0:
                loss_numerator += float(raw_loss.detach().item()) * batch_valid_pixels
                valid_pixel_count += batch_valid_pixels

            mini_batches_seen += 1

            should_step = (
                ((i + 1) % args.grad_accum_steps == 0)
                or ((i + 1) == len(loader))
            )

            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
                last_grad_norm = float(grad_norm.detach().item())

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # If GradScaler lowers the scale, an overflow was found and the
                # optimizer step was skipped. Do not advance the LR schedule.
                optimizer_step_happened = (
                    (not amp_enabled)
                    or (scale_after >= scale_before)
                )

                if optimizer_step_happened:
                    scheduler.step()
                    global_step += 1
                    optimizer_updates_this_epoch += 1
                else:
                    amp_skipped_updates += 1
                    print(
                        "[amp] skipped optimizer update due to overflow | "
                        f"epoch={epoch + 1}, batch={i}"
                    )

                optimizer.zero_grad(set_to_none=True)

            if (i % args.log_every == 0) or ((i + 1) == len(loader)):
                running_loss = (
                    loss_numerator / valid_pixel_count
                    if valid_pixel_count > 0
                    else float("nan")
                )
                current_lr = float(optimizer.param_groups[0]["lr"])

                print(
                    f"epoch {epoch + 1:03d}/{args.epochs:03d} | "
                    f"batch {i + 1:05d}/{len(loader):05d} | "
                    f"raw_ce {raw_loss.detach().item():.6f} | "
                    f"running_ce {running_loss:.6f} | "
                    f"lr {current_lr:.3e} | "
                    f"step {global_step}/{total_update_steps}"
                )

        if valid_pixel_count <= 0:
            raise RuntimeError(
                f"Epoch {epoch + 1} contained no valid pixels."
            )

        epoch_loss = loss_numerator / valid_pixel_count
        epoch_seconds = time.time() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer_updates_this_epoch": optimizer_updates_this_epoch,
            "amp_skipped_updates": amp_skipped_updates,
            "mini_batches": mini_batches_seen,
            "valid_pixels": valid_pixel_count,
            "train_ce": epoch_loss,
            "lr": current_lr,
            "last_grad_norm_before_clip": last_grad_norm,
            "epoch_seconds": epoch_seconds,
        }
        append_jsonl(log_path, epoch_record)

        checkpoint_payload = make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            protocol=protocol,
            model_meta=model_meta,
        )

        latest_path = checkpoint_dir / "latest.pt"
        atomic_torch_save(checkpoint_payload, latest_path)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            archive_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            atomic_torch_save(checkpoint_payload, archive_path)

        print(
            f"[epoch done] {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_ce={epoch_loss:.6f} | "
            f"updates={optimizer_updates_this_epoch} | "
            f"global_step={global_step} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_seconds:.1f}s"
        )

    final_payload = make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=args.epochs - 1,
        global_step=global_step,
        protocol=protocol,
        model_meta=model_meta,
    )
    final_path = checkpoint_dir / "final.pt"
    atomic_torch_save(final_payload, final_path)

    total_seconds = time.time() - training_start
    print("=" * 78)
    print(
        f"[finished] {MODEL_NAME} | "
        f"epochs={args.epochs} | "
        f"global_step={global_step} | "
        f"time={total_seconds / 3600.0:.2f}h"
    )
    print(f"[finished] latest checkpoint: {checkpoint_dir / 'latest.pt'}")
    print(f"[finished] final checkpoint : {final_path}")
    print(f"[finished] training log     : {log_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
