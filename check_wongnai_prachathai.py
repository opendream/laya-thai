"""Assert the shipped Wongnai and Prachathai transforms and train-only selections."""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import convert_prachathai
import convert_wongnai
import train_massive
from transformers import AutoTokenizer


def _thai(text: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


def check_wongnai() -> None:
    assert convert_wongnai.training_split() == "train"
    assert 7_500 <= convert_wongnai.SLICE <= 22_500
    assert convert_wongnai.coerce_star(0) == 1
    for star in range(1, 6):
        assert convert_wongnai.coerce_star(star) == star
    assert convert_wongnai.coerce_star(0, ["1", "2", "3", "4", "5"]) == 1
    assert convert_wongnai.coerce_star(4, ["1", "2", "3", "4", "5"]) == 5
    indices = convert_wongnai.select_indices(40_000)
    assert indices == convert_wongnai.select_indices(40_000)
    assert len(indices) == convert_wongnai.SLICE
    for star in range(1, 6):
        row = convert_wongnai.to_laya_row("ร้านนี้ดี", star)
        assert "review" in row["state"]
        question = row["questions"]["rating"]
        assert question["type"] == "score"
        assert len(question["criteria"]) == 5
        assert all(_thai(text) and str(n) in text for n, text in enumerate(question["criteria"], start=1))
        gold = row["gold"]["rating"]
        assert gold["label"] == star
        # Same index keys rows_to_items reads: rung 0 is star 1, rung 4 is star 5.
        target = [float(gold["probabilities"].get(str(i), 0.0)) for i in range(5)]
        assert target == [1.0 if i == star - 1 else 0.0 for i in range(5)]
        assert gold["probabilities"]["4"] == (1.0 if star == 5 else 0.0)
        assert "5" not in gold["probabilities"]
    src = train_massive.multilingual_snapshot()
    tok = AutoTokenizer.from_pretrained(str(src / "tokenizer"))
    built = [
        convert_wongnai.to_laya_row("ร้านนี้ดี", star)
        for star in range(1, 6)
    ]
    items = train_massive.rows_to_items(built, tok, {"max_len": 1024, "head_max_len": 256})
    assert len(items) == 5
    for star, item in zip(range(1, 6), items):
        assert item["target"] == [1.0 if i == star - 1 else 0.0 for i in range(5)]
        assert item["label"] == star - 1


def check_prachathai() -> None:
    assert convert_prachathai.training_split() == "train"
    assert 12_500 <= convert_prachathai.SLICE <= 37_500
    topics = convert_prachathai.TOPICS
    assert list(topics.values()) == [
        "การเมือง",
        "สิทธิมนุษยชน",
        "คุณภาพชีวิต",
        "ต่างประเทศ",
        "สังคม",
        "สิ่งแวดล้อม",
        "เศรษฐกิจ",
        "วัฒนธรรม",
        "แรงงาน",
        "ความมั่นคง",
        "ไอซีที",
        "การศึกษา",
    ]
    none = {key: False for key in topics}
    one = {key: key == "politics" for key in topics}
    two = {key: key in {"politics", "social"} for key in topics}
    empty = convert_prachathai.to_laya_row("หัวข้อ", "ก" * 2000, none)
    assert empty["state"]["body"] == "ก" * 1024
    assert len(empty["state"]["body"]) <= 1024
    nouls = [q for q in empty["questions"].values() if q["type"] == "noul"]
    assert len(nouls) == 1
    assert list(empty["gold"].values())[0]["label"] is False
    assert "topic" not in empty["questions"]

    single = convert_prachathai.to_laya_row("หัวข้อ", "เนื้อ", one)
    single_nouls = {key: q for key, q in single["questions"].items() if q["type"] == "noul"}
    assert len(single_nouls) == 2
    assert single["gold"]["is_politics"]["label"] is True
    assert any(answer["label"] is False for key, answer in single["gold"].items() if key.startswith("is_"))
    choice = single["questions"]["topic"]
    assert choice["type"] == "choice"
    assert list(choice["criteria"]) == list(topics)
    assert list(choice["criteria"].values()) == list(topics.values())
    assert all(_thai(text) for text in choice["criteria"].values())
    assert single["gold"]["topic"]["label"] == "politics"
    assert single["gold"]["topic"]["probabilities"]["politics"] == 1.0

    both = convert_prachathai.to_laya_row("หัวข้อ", "เนื้อ", two)
    assert "topic" not in both["questions"]
    assert len([q for q in both["questions"].values() if q["type"] == "noul"]) == 2
    assert both["gold"]["is_politics"]["label"] is True
    assert both["gold"]["is_human_rights"]["label"] is False
    indices = convert_prachathai.select_indices(54_379)
    assert indices == convert_prachathai.select_indices(54_379)
    assert len(indices) == convert_prachathai.SLICE


def main() -> None:
    check_wongnai()
    check_prachathai()
    print(
        f"wongnai_split={convert_wongnai.training_split()} wongnai_slice={convert_wongnai.SLICE} "
        f"prachathai_split={convert_prachathai.training_split()} prachathai_slice={convert_prachathai.SLICE} "
        f"train_only=1"
    )


if __name__ == "__main__":
    main()
