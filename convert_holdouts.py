"""Write the measure files that stay out of data/train.jsonl.

XNLI and Prachathai use their validation splits. Wongnai has no validation
split, so the measure file is its test split.
"""
from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, set_verbosity_error

import convert_prachathai
import convert_wongnai
import convert_xnli_th

OUT = Path("data")


def _quiet() -> None:
    disable_progress_bar()
    set_verbosity_error()


def xnli_val() -> Path:
    _quiet()
    ds = load_dataset(convert_xnli_th.DATASET, convert_xnli_th.CONFIG, split="validation")
    rows = [
        convert_xnli_th.to_laya_row(row["premise"], row["hypothesis"], row["label"])
        for row in ds
    ]
    dest = OUT / "xnli_th.val.jsonl"
    convert_xnli_th.write_jsonl(dest, rows)
    print(f"split=validation rows={len(rows)} -> {dest}")
    return dest


def prachathai_val() -> Path:
    _quiet()
    ds = load_dataset(
        convert_prachathai.DATASET,
        split="validation",
        revision="refs/convert/parquet",
    )
    rows = [
        convert_prachathai.to_laya_row(
            row.get("title") or "",
            row.get("body_text"),
            convert_prachathai.flags_from_row(row),
        )
        for row in ds
    ]
    dest = OUT / "prachathai.val.jsonl"
    convert_prachathai.write_jsonl(dest, rows)
    print(f"split=validation rows={len(rows)} -> {dest}")
    return dest


def wongnai_measure() -> Path:
    _quiet()
    ds = load_dataset(convert_wongnai.DATASET, split="test")
    names = convert_wongnai.class_names(ds)
    rows = []
    for row in ds:
        if row.get("star_rating") is None or not row.get("review_body"):
            continue
        rows.append(
            convert_wongnai.to_laya_row(
                row["review_body"],
                convert_wongnai.coerce_star(row["star_rating"], names),
            )
        )
    dest = OUT / "wongnai_measure.jsonl"
    convert_wongnai.write_jsonl(dest, rows)
    print(f"split=test rows={len(rows)} -> {dest}")
    return dest


def main() -> None:
    xnli_val()
    prachathai_val()
    wongnai_measure()


if __name__ == "__main__":
    main()
