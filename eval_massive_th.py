"""Evaluate a Laya checkpoint on a MASSIVE-th JSONL file."""
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


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="data/massive_th.test.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ckpt", default="")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.ckpt:
        agent = laya.Agent(args.ckpt, device=device)
        tag = args.ckpt
    else:
        agent = laya.load(
            "convaiinnovations/laya", subfolder="multilingual", device=device
        )
        tag = "multilingual-zero-shot"

    rows = load_jsonl(Path(args.file), args.limit)
    if not rows:
        raise SystemExit(f"empty file: {args.file}")

    ok, t0 = 0, time.perf_counter()
    for i, row in enumerate(rows, 1):
        out = agent.predict(row["state"], row["questions"])["answers"]["intent"]
        ok += int(out["choice"] == row["gold"]["intent"]["label"])
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} running_acc={ok/i:.3f}")

    acc = ok / len(rows)
    print(
        f"ckpt={tag} file={args.file} n={len(rows)} "
        f"acc={acc:.3f} published_20way=0.480 mac40=0.475 "
        f"ms/row={(time.perf_counter()-t0)/len(rows)*1000:.0f} device={device}"
    )


if __name__ == "__main__":
    main()
