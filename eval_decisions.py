"""Score a checkpoint on a Laya JSONL of choice, score, and noul questions."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import laya


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def correct(question: dict, gold: dict, answer: dict) -> bool:
    kind = question["type"]
    if kind == "choice":
        return answer["choice"] == gold["label"]
    if kind == "noul":
        return (answer["noul"] >= 0.5) == bool(gold["label"])
    if kind == "score":
        predicted = max(answer["probabilities"], key=lambda key: answer["probabilities"][key])
        return int(predicted) == int(gold["label"]) - 1
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    agent = laya.Agent(args.ckpt, device=device)
    rows = load_jsonl(Path(args.file))
    ok = 0
    n = 0
    by_type = {}
    t0 = time.perf_counter()
    for row in rows:
        answers = agent.predict(row["state"], row["questions"])["answers"]
        for qid, question in row["questions"].items():
            hit = correct(question, row["gold"][qid], answers[qid])
            ok += int(hit)
            n += 1
            bucket = by_type.setdefault(question["type"], [0, 0])
            bucket[0] += int(hit)
            bucket[1] += 1
    if n == 0:
        raise SystemExit(f"no questions in {args.file}")
    parts = " ".join(f"{kind}={hit / total:.3f}({total})" for kind, (hit, total) in sorted(by_type.items()))
    print(
        f"ckpt={args.ckpt} file={args.file} n_rows={len(rows)} n={n} "
        f"acc={ok / n:.3f} {parts} "
        f"ms/row={(time.perf_counter() - t0) / len(rows) * 1000:.0f} device={device}"
    )


if __name__ == "__main__":
    main()
