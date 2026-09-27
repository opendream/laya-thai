"""MASSIVE English train split -> Laya JSONL.

Same 20-name choice as the Thai file. Only the train split is written.
"""
from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, set_verbosity_error

from convert_massive_th import SEED, all_labels, sample_options, to_laya_row, write_jsonl

OUT = Path("data") / "massive_en.train.jsonl"
INSTRUCTIONS = "What is the user asking for in utterance?"


def training_split() -> str:
    return "train"


def convert() -> Path:
    disable_progress_bar()
    set_verbosity_error()
    split = training_split()
    ds = load_dataset("mteb/amazon_massive_intent", "en", split=split)
    labels = all_labels(ds)
    rng = random.Random(SEED)
    rows = []
    for row in ds:
        gold = row["label_text"]
        opts = sample_options(gold, labels, rng)
        rows.append(to_laya_row(row["text"], gold, opts, INSTRUCTIONS))
    write_jsonl(OUT, rows)
    print(f"split={split} rows={len(rows)} -> {OUT}")
    return OUT


if __name__ == "__main__":
    convert()
