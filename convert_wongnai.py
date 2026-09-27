"""Wongnai review train slice -> Laya JSONL.

One row is a review with a 1-5 star score. The training file is built from
the train split only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("data")
OUT_NAME = "wongnai.train.jsonl"
DATASET = "Wongnai/wongnai_reviews"
SEED = 13
SLICE = 15_000
RUNGS = (
    "แย่มาก (1 ดาว)",
    "แย่ (2 ดาว)",
    "ปานกลาง (3 ดาว)",
    "ดี (4 ดาว)",
    "ดีมาก (5 ดาว)",
)
INSTRUCTIONS = "รีวิวนี้ให้คะแนนร้านกี่ดาว"


def training_split() -> str:
    """The only Wongnai split written into the training JSONL."""
    return "train"


def output_path(root: Path = OUT) -> Path:
    return Path(root) / OUT_NAME


def coerce_star(value, class_names: list[str] | None = None) -> int:
    """Map a dataset rating onto a star from 1 through 5.

    Class-label names win, so index 4 whose name is ``5`` is five stars.
    A bare 0 is the first class. A bare value already in 1-5 is that star.
    """
    if class_names is not None:
        name = str(class_names[int(value)]).strip()
        if name.isdigit() and 1 <= int(name) <= 5:
            return int(name)
    number = int(value)
    if number == 0:
        return 1
    if 1 <= number <= 5:
        return number
    raise ValueError(f"star rating {value!r}")


def to_laya_row(review: str, star: int) -> dict:
    star = coerce_star(star)
    return {
        "state": {"review": review},
        "questions": {
            "rating": {
                "type": "score",
                "instructions": INSTRUCTIONS,
                "criteria": list(RUNGS),
            }
        },
        "gold": {
            "rating": {
                "label": star,
                "probabilities": {str(i): (1.0 if i == star - 1 else 0.0) for i in range(5)},
            }
        },
    }


def select_indices(n_rows: int, k: int = SLICE, seed: int = SEED) -> list[int]:
    if k < 7_500 or k > 22_500:
        raise ValueError(f"slice {k} is outside 7500-22500")
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


def class_names(ds) -> list[str] | None:
    feature = ds.features.get("star_rating") if getattr(ds, "features", None) else None
    names = getattr(feature, "names", None)
    if not names:
        return None
    return [str(name) for name in names]


def convert_all() -> Path:
    """Every labeled train review. The 15k slice file is left as it is."""
    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()
    split = training_split()
    ds = load_dataset(DATASET, split=split)
    names = class_names(ds)
    rows = []
    for row in ds:
        if row.get("star_rating") is None or not row.get("review_body"):
            continue
        rows.append(to_laya_row(row["review_body"], coerce_star(row["star_rating"], names)))
    dest = Path("data") / "wongnai_full.train.jsonl"
    write_jsonl(dest, rows)
    print(f"split={split} rows={len(rows)} -> {dest}")
    return dest


def convert() -> Path:
    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()
    split = training_split()
    ds = load_dataset(DATASET, split=split)
    names = class_names(ds)
    indices = select_indices(len(ds), SLICE, SEED)
    rows = []
    for row in ds.select(indices):
        if row.get("star_rating") is None or not row.get("review_body"):
            continue
        rows.append(to_laya_row(row["review_body"], coerce_star(row["star_rating"], names)))
    if not (7_500 <= len(rows) <= 22_500):
        raise SystemExit(f"wongnai slice has {len(rows)} rows")
    dest = output_path()
    write_jsonl(dest, rows)
    print(f"split={split} rows={len(rows)} -> {dest}")
    return dest


if __name__ == "__main__":
    convert()
