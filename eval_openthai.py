"""Score OpenThai-SystemOne on the same JSONL and the same right/wrong rule as eval_decisions.py.

One option order, the order written in the file. Their client otherwise averages
eight cyclic orders whenever a choice menu has 11 or more names.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openthai_systemone import SystemOneClient


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def correct(question: dict, gold: dict, answer) -> bool:
    kind = question["type"]
    if kind == "choice":
        return answer.choice == gold["label"]
    if kind == "noul":
        return (answer.noul >= 0.5) == bool(gold["label"])
    if kind == "score":
        predicted = max(answer.probabilities, key=lambda key: answer.probabilities[key])
        return int(predicted) == int(gold["label"]) - 1
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="iapp/OpenThai-SystemOne")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    client = SystemOneClient(args.model, device="mps")
    rows = load_jsonl(Path(args.file))
    ok = 0
    n = 0
    by_type: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        responses = client.system_one_batch(
            [(row["state"], row["questions"]) for row in chunk],
        )
        for row, resp in zip(chunk, responses):
            for qid, question in row["questions"].items():
                hit = correct(question, row["gold"][qid], resp.answers[qid])
                ok += int(hit)
                n += 1
                bucket = by_type.setdefault(question["type"], [0, 0])
                bucket[0] += int(hit)
                bucket[1] += 1
        done = min(start + args.batch, len(rows))
        if done % 200 < args.batch or done == len(rows):
            print(f"  {done}/{len(rows)} acc={ok / n:.3f}", flush=True)
    parts = " ".join(
        f"{kind}={hit / total:.3f}({total})" for kind, (hit, total) in sorted(by_type.items())
    )
    print(
        f"model={args.model} file={args.file} n_rows={len(rows)} n={n} "
        f"acc={ok / n:.3f} {parts} "
        f"ms/row={(time.perf_counter() - t0) / len(rows) * 1000:.0f} order=as_written"
    )


if __name__ == "__main__":
    main()
