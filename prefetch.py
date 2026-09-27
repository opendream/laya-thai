"""Download weights + MASSIVE-th while you still have wifi."""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main() -> None:
    print("downloading laya multilingual snapshot…")
    path = snapshot_download(
        "convaiinnovations/laya",
        allow_patterns=["multilingual/*"],
    )
    print("weights", path)

    print("downloading MASSIVE-th splits…")
    for split in ("train", "validation", "test"):
        ds = load_dataset("mteb/amazon_massive_intent", "th", split=split)
        print(f"  {split:10} {len(ds)} rows")

    import laya
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    out = agent.predict(
        {"body": "คิดเงินซ้ำ กรุณาคืนเงิน"},
        {"refund": {"type": "noul", "instructions": "ขอคืนเงินหรือไม่"}},
    )
    print("probe", out["answers"]["refund"]["noul"])
    print("prefetch ok")


if __name__ == "__main__":
    main()
