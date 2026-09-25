"""Prachathai train slice -> Laya JSONL.

One row is a news article. It carries topic yes/no questions, and a 12-way
topic choice when the article has exactly one topic. The training file is
built from the train split only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("data")
OUT_NAME = "prachathai.train.jsonl"
DATASET = "PyThaiNLP/prachathai67k"
SEED = 13
SLICE = 25_000
BODY_CHARS = 1_024
TOPICS = {
    "politics": "การเมือง",
    "human_rights": "สิทธิมนุษยชน",
    "quality_of_life": "คุณภาพชีวิต",
    "international": "ต่างประเทศ",
    "social": "สังคม",
    "environment": "สิ่งแวดล้อม",
    "economics": "เศรษฐกิจ",
    "culture": "วัฒนธรรม",
    "labor": "แรงงาน",
    "national_security": "ความมั่นคง",
    "ict": "ไอซีที",
    "education": "การศึกษา",
}
CHOICE_INSTRUCTIONS = "ข่าวนี้อยู่ในหมวดหลักใด"


def training_split() -> str:
    """The only Prachathai split written into the training JSONL."""
    return "train"


def output_path(root: Path = OUT) -> Path:
    return Path(root) / OUT_NAME


def clip_body(body: str | None) -> str:
    return (body or "")[:BODY_CHARS]


def _noul(topic: str, is_true: bool) -> tuple[dict, dict]:
    question = {
        "type": "noul",
        "instructions": f"ข่าวนี้เกี่ยวข้องกับหัวข้อ '{TOPICS[topic]}' หรือไม่",
    }
    gold = {
        "label": is_true,
        "probabilities": {
            "false": 0.0 if is_true else 1.0,
            "true": 1.0 if is_true else 0.0,
        },
    }
    return question, gold


def to_laya_row(title: str, body: str | None, topics: dict) -> dict:
    positives = [key for key in TOPICS if topics.get(key)]
    negatives = [key for key in TOPICS if key not in positives]
    questions: dict = {}
    gold: dict = {}
    if not positives:
        question, answer = _noul(next(iter(TOPICS)), False)
        questions[f"is_{next(iter(TOPICS))}"] = question
        gold[f"is_{next(iter(TOPICS))}"] = answer
    elif negatives:
        yes, no = positives[0], negatives[0]
        question, answer = _noul(yes, True)
        questions[f"is_{yes}"] = question
        gold[f"is_{yes}"] = answer
        question, answer = _noul(no, False)
        questions[f"is_{no}"] = question
        gold[f"is_{no}"] = answer
        if len(positives) == 1:
            questions["topic"] = {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(TOPICS),
            }
            gold["topic"] = {
                "label": yes,
                "probabilities": {key: (1.0 if key == yes else 0.0) for key in TOPICS},
            }
    else:
        question, answer = _noul(positives[0], True)
        questions[f"is_{positives[0]}"] = question
        gold[f"is_{positives[0]}"] = answer
    return {
        "state": {"title": title or "", "body": clip_body(body)},
        "questions": questions,
        "gold": gold,
    }


def select_indices(n_rows: int, k: int = SLICE, seed: int = SEED) -> list[int]:
    if k < 12_500 or k > 37_500:
        raise ValueError(f"slice {k} is outside 12500-37500")
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


def flags_from_row(row: dict) -> dict:
    return {key: bool(row.get(key)) for key in TOPICS}


def convert() -> Path:
    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()
    split = training_split()
    # The Hub copy is an old dataset script. The parquet revision is the loadable train split.
    ds = load_dataset(DATASET, split=split, revision="refs/convert/parquet")
    indices = select_indices(len(ds), SLICE, SEED)
    rows = [
        to_laya_row(row.get("title") or "", row.get("body_text"), flags_from_row(row))
        for row in ds.select(indices)
    ]
    if not (12_500 <= len(rows) <= 37_500):
        raise SystemExit(f"prachathai slice has {len(rows)} rows")
    dest = output_path()
    write_jsonl(dest, rows)
    print(f"split={split} rows={len(rows)} -> {dest}")
    return dest


if __name__ == "__main__":
    convert()
