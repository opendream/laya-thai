"""XNLI Thai train slice -> Laya JSONL.

One row is a premise/hypothesis pair with a 3-way choice and an entailment
noul. The training file is built from the train split only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path("data")
OUT_NAME = "xnli_th.train.jsonl"
DATASET = "facebook/xnli"
CONFIG = "th"
SEED = 13
SLICE = 20_000
LABELS = ("entailment", "neutral", "contradiction")
CHOICE_TEXT = {
    "entailment": "ประโยคที่สองสรุปได้จากประโยคแรก",
    "neutral": "ไม่สามารถสรุปได้",
    "contradiction": "ประโยคที่สองขัดแย้งกับประโยคแรก",
}
CHOICE_INSTRUCTIONS = "ความสัมพันธ์ระหว่าง premise และ hypothesis คืออะไร"
NOUL_INSTRUCTIONS = "hypothesis สรุปได้จาก premise หรือไม่"


def training_split() -> str:
    """The only XNLI split written into the training JSONL."""
    return "train"


def output_path(root: Path = OUT) -> Path:
    return Path(root) / OUT_NAME


def label_name(label: str | int) -> str:
    if isinstance(label, str):
        name = label
    else:
        name = LABELS[int(label)]
    if name not in CHOICE_TEXT:
        raise ValueError(f"unknown XNLI label {label!r}")
    return name


def to_laya_row(premise: str, hypothesis: str, label: str | int) -> dict:
    gold = label_name(label)
    entails = gold == "entailment"
    return {
        "state": {"premise": premise, "hypothesis": hypothesis},
        "questions": {
            "nli": {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_TEXT),
            },
            "entails": {
                "type": "noul",
                "instructions": NOUL_INSTRUCTIONS,
            },
        },
        "gold": {
            "nli": {
                "label": gold,
                "probabilities": {name: (1.0 if name == gold else 0.0) for name in LABELS},
            },
            "entails": {
                "label": entails,
                "probabilities": {
                    "false": 0.0 if entails else 1.0,
                    "true": 1.0 if entails else 0.0,
                },
            },
        },
    }


def select_indices(n_rows: int, k: int = SLICE, seed: int = SEED) -> list[int]:
    if k < 10_000 or k > 30_000:
        raise ValueError(f"slice {k} is outside 10000-30000")
    if n_rows < k:
        raise ValueError(f"need at least {k} train rows, got {n_rows}")
    chosen = random.Random(seed).sample(range(n_rows), k)
    chosen.sort()
    return chosen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(limit: int | None = None) -> Path:
    from datasets import load_dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()
    split = training_split()
    ds = load_dataset(DATASET, CONFIG, split=split)
    k = SLICE if limit is None else limit
    indices = select_indices(len(ds), k, SEED)
    rows = [
        to_laya_row(row["premise"], row["hypothesis"], row["label"])
        for row in ds.select(indices)
    ]
    dest = output_path()
    write_jsonl(dest, rows)
    print(f"split={split} rows={len(rows)} -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    convert(args.limit)


if __name__ == "__main__":
    main()
