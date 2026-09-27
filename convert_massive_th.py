"""MASSIVE-th -> Laya JSONL. test split is hold-out. Never train on it."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path("data")
N_OPTS = 20
SEED = 13


def load_th(split: str):
    return load_dataset("mteb/amazon_massive_intent", "th", split=split)


def all_labels(ds) -> list[str]:
    return sorted({row["label_text"] for row in ds})


def sample_options(gold: str, labels: list[str], rng: random.Random) -> list[str]:
    pool = [x for x in labels if x != gold]
    k = min(N_OPTS - 1, len(pool))
    opts = [gold] + rng.sample(pool, k)
    rng.shuffle(opts)
    return opts


def pretty(name: str) -> str:
    return name.replace("_", " ").replace(".", ": ")


def to_laya_row(
    text: str,
    gold: str,
    opts: list[str],
    instructions: str = "ผู้ใช้ต้องการทำอะไรใน utterance",
) -> dict:
    return {
        "state": {"utterance": text},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {k: pretty(k) for k in opts},
            }
        },
        "gold": {
            "intent": {
                "label": gold,
                "probabilities": {k: (1.0 if k == gold else 0.0) for k in opts},
            }
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(split: str, out_name: str, limit: int | None) -> None:
    ds = load_th(split)
    labels = all_labels(ds)
    rng = random.Random(SEED)
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for row in ds.select(range(n)):
        gold = row["label_text"]
        opts = sample_options(gold, labels, rng)
        rows.append(to_laya_row(row["text"], gold, opts))
    dest = OUT / out_name
    write_jsonl(dest, rows)
    print(f"{split:10} -> {dest}  rows={len(rows)}  labels={len(labels)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-train", type=int, default=None)
    args = p.parse_args()
    convert("train", "massive_th.train.jsonl", args.limit_train)
    convert("validation", "massive_th.val.jsonl", None)
    convert("test", "massive_th.test.jsonl", None)


if __name__ == "__main__":
    main()
