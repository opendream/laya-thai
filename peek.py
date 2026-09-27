"""Inspect converted JSONL without the network."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def peek(path: Path) -> None:
    if not path.exists():
        print("missing", path)
        return
    n = 0
    n_opts = Counter()
    with path.open(encoding="utf-8") as f:
        first = None
        for line in f:
            row = json.loads(line)
            n += 1
            if first is None:
                first = row
            keys = list(row["questions"]["intent"]["criteria"])
            n_opts[len(keys)] += 1
            gold = row["gold"]["intent"]["label"]
            assert gold in keys, gold
    print(f"\n{path}")
    print("  rows", n)
    print("  option-count", dict(n_opts))
    if first:
        print("  sample utterance:", first["state"]["utterance"][:80])
        print("  sample gold:", first["gold"]["intent"]["label"])
        print("  sample n_opts:", len(first["questions"]["intent"]["criteria"]))


def main() -> None:
    for name in (
        "massive_th.train.jsonl",
        "massive_th.val.jsonl",
        "massive_th.test.jsonl",
    ):
        peek(Path("data") / name)


if __name__ == "__main__":
    main()
