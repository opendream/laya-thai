"""MASSIVE-th RLCD settings, train-split selection, and the 4-or-8 epoch rule.

This module does not load the 322M model. The training entry and the config
check both call these functions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BASE_EPOCHS = 4
MAX_EPOCHS = 8
MICROBATCH = 8
GRAD_ACCUM = 8
LR_ENCODER = 2.5e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ETA_MIN = 1e-6
GROUP_SIZE = 4
SIGMA_START = 0.4
SIGMA_END = 0.1
W_SPH = 0.75
CE_WEIGHT = 1.0
PRECISION = "fp32"
TRAIN_NAME = "massive_th.train.jsonl"
VAL_NAME = "massive_th.val.jsonl"


def train_jsonl(root: Path) -> Path:
    """The only file this run is allowed to learn from."""
    path = (Path(root) / "data" / TRAIN_NAME).resolve()
    if path.name != TRAIN_NAME:
        raise ValueError(f"refusing training input {path.name}")
    return path


def training_files(root: Path) -> list[Path]:
    return [train_jsonl(root)]


def val_jsonl(root: Path) -> Path:
    path = (Path(root) / "data" / VAL_NAME).resolve()
    if path.name != VAL_NAME:
        raise ValueError(f"refusing validation input {path.name}")
    return path


def load_train_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in training_files(root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no training rows in {training_files(root)}")
    return rows


def microbatch_count(n_items: int, microbatch: int = MICROBATCH) -> int:
    if n_items < 0:
        raise ValueError(n_items)
    if n_items == 0:
        return 0
    return (n_items + microbatch - 1) // microbatch


def optimizer_steps(
    n_items: int,
    epochs: int = BASE_EPOCHS,
    microbatch: int = MICROBATCH,
    grad_accum: int = GRAD_ACCUM,
) -> int:
    """Adam steps for `epochs` passes, including a short final accumulation group."""
    micros = microbatch_count(n_items, microbatch)
    if micros == 0 or epochs == 0:
        return 0
    per_epoch = (micros + grad_accum - 1) // grad_accum
    return per_epoch * epochs


def cosine_horizon(n_items: int) -> int:
    """Optimizer steps in the first four epochs. The schedule is not stretched to 8."""
    return optimizer_steps(n_items, epochs=BASE_EPOCHS)


def sigma_for_epoch(epoch_index: int) -> float:
    """0-based epoch. Epochs 1–4 anneal 0.4 → 0.1. Later epochs stay at 0.1."""
    if epoch_index <= 0:
        return SIGMA_START
    if epoch_index >= BASE_EPOCHS - 1:
        return SIGMA_END
    progress = epoch_index / (BASE_EPOCHS - 1)
    return SIGMA_START + (SIGMA_END - SIGMA_START) * progress


def learning_rate(base_lr: float, step_index: int, horizon: int, eta_min: float = ETA_MIN) -> float:
    """Cosine from `base_lr` toward `eta_min` across the 4-epoch horizon.

    `step_index` is the 0-based Adam step about to run. The last step of the
    horizon is about `eta_min`. Every later step stays there, so an extension
    cannot lift the learning rate.
    """
    if step_index < 0 or horizon <= 0:
        raise ValueError((step_index, horizon))
    if step_index >= horizon:
        return eta_min
    progress = step_index / horizon
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * progress)) / 2


def extend_after_four(epoch3_acc: float, epoch4_acc: float) -> bool:
    """Train epochs 5–8 only when epoch-4 validation accuracy is strictly higher."""
    return epoch4_acc > epoch3_acc


def select_epoch(accuracies: list[float]) -> int:
    """1-based epoch with the highest validation accuracy. Ties keep the earliest."""
    if not accuracies:
        raise ValueError("no validation scores")
    best = 0
    for index, acc in enumerate(accuracies):
        if acc > accuracies[best]:
            best = index
    return best + 1
