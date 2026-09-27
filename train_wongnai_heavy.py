"""Two more RLCD epochs from mix epoch 2, with every Wongnai train review.

The 6,203-review measure file is not an input. The 15k Wongnai slice stays
on disk. This file uses the full train split instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from train_massive import ROOT, fit, load_student, rows_to_items
from train_recipe import PRECISION

INIT = ROOT / "checkpoints" / "mix_rlcd" / "epoch_02"
TRAIN = ROOT / "data" / "train_wongnai_heavy.jsonl"
OUT = ROOT / "checkpoints" / "wongnai_heavy_rlcd"
EPOCHS = 2
SOURCES = (
    "massive_th.train.jsonl",
    "xnli_th.train.jsonl",
    "wongnai_full.train.jsonl",
    "prachathai.train.jsonl",
    "massive_en.train.jsonl",
)


def build_train_file() -> dict:
    counts = []
    total = 0
    with TRAIN.open("w", encoding="utf-8") as dest:
        for name in SOURCES:
            path = ROOT / "data" / name
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            n = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        dest.write(line if line.endswith("\n") else line + "\n")
                        n += 1
            counts.append({"file": name, "rows": n})
            total += n
    manifest = {"files": counts, "total": total, "init": str(INIT), "epochs": EPOCHS}
    path = ROOT / "data" / "train_wongnai_heavy_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"train_file={TRAIN} total={total}", flush=True)
    for item in counts:
        print(f"  {item['file']} rows={item['rows']}", flush=True)
    return manifest


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty training file: {path}")
    return rows


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise SystemExit("MPS is required for this run")
    if not (INIT / "model.safetensors").is_file():
        raise SystemExit(f"missing init checkpoint: {INIT}")
    build_train_file()
    print(
        f"train_file={TRAIN} init_ckpt={INIT} epochs={EPOCHS} precision={PRECISION}",
        flush=True,
    )
    model, tok, cfg = load_student(INIT, device)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    item_cfg = dict(cfg)
    item_cfg["max_len"] = 384
    items = rows_to_items(load_rows(TRAIN), tok, item_cfg)
    fit(items, model, tok, cfg, INIT, OUT, device, epoch_limit=EPOCHS)


if __name__ == "__main__":
    main()
