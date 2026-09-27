"""Continue RLCD from the MASSIVE epoch-6 checkpoint on the merged train file.

The recipe is unchanged: four epochs, eight only if MASSIVE validation at
epoch 4 is strictly above epoch 3. The merged file is the only training input.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from train_massive import ROOT, fit, load_student, rows_to_items
from train_recipe import BASE_EPOCHS, MAX_EPOCHS, PRECISION

INIT = ROOT / "checkpoints" / "massive_th_rlcd" / "epoch_06"
TRAIN = ROOT / "data" / "train.jsonl"
OUT = ROOT / "checkpoints" / "mix_rlcd"


def load_rows(path) -> list[dict]:
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
    print(
        f"train_file={TRAIN} init_ckpt={INIT} base_epochs={BASE_EPOCHS} "
        f"max_epochs={MAX_EPOCHS} precision={PRECISION}",
        flush=True,
    )
    model, tok, cfg = load_student(INIT, device)
    # Padded batches in this mix are several times longer than MASSIVE slips.
    # Checkpoint the encoder so the backward pass does not fill unified memory.
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    item_cfg = dict(cfg)
    # Keep slips inside a few hundred tokens. A 1024-token batch of 8
    # blew unified memory to 67 GB on this machine.
    item_cfg["max_len"] = 384
    items = rows_to_items(load_rows(TRAIN), tok, item_cfg)
    fit(items, model, tok, cfg, INIT, OUT, device)


if __name__ == "__main__":
    main()
