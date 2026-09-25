"""Assert the shipped XNLI Thai transform and train-only split selection."""
from __future__ import annotations

import convert_xnli_th


def main() -> None:
    assert convert_xnli_th.training_split() == "train"
    assert convert_xnli_th.SLICE == 20_000
    assert 10_000 <= convert_xnli_th.SLICE <= 30_000
    indices = convert_xnli_th.select_indices(400_000)
    assert indices == convert_xnli_th.select_indices(400_000)
    assert len(indices) == convert_xnli_th.SLICE
    assert indices == sorted(indices)

    out = convert_xnli_th.output_path()
    assert out.name == "xnli_th.train.jsonl"
    assert out.name != "massive_th.train.jsonl"
    assert out.name != "massive_th.test.jsonl"

    for label in convert_xnli_th.LABELS:
        row = convert_xnli_th.to_laya_row("ประโยคแรก", "ประโยคที่สอง", label)
        assert set(row["state"]) == {"premise", "hypothesis"}
        choice = row["questions"]["nli"]
        assert choice["type"] == "choice"
        assert list(choice["criteria"]) == ["entailment", "neutral", "contradiction"]
        assert all(any("\u0e00" <= ch <= "\u0e7f" for ch in text) for text in choice["criteria"].values())
        noul = row["questions"]["entails"]
        assert noul["type"] == "noul"
        gold = row["gold"]["nli"]
        assert gold["label"] == label
        assert gold["probabilities"][label] == 1.0
        assert sum(gold["probabilities"].values()) == 1.0
        entails = row["gold"]["entails"]
        assert entails["label"] is (label == "entailment")
        assert entails["probabilities"]["true"] == (1.0 if label == "entailment" else 0.0)
        assert entails["probabilities"]["false"] == (0.0 if label == "entailment" else 1.0)

    print(
        f"split={convert_xnli_th.training_split()} "
        f"slice={convert_xnli_th.SLICE} "
        f"out={out} "
        f"labels={','.join(convert_xnli_th.LABELS)} "
        f"train_only=1"
    )


if __name__ == "__main__":
    main()
