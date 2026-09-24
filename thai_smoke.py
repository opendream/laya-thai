"""Quick Thai + English routing smoke test."""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import laya

device = "mps" if torch.backends.mps.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="", help="Score this checkpoint instead of the base multilingual loader")
args = parser.parse_args()
print("device", device, "laya", laya.__version__)
if args.ckpt:
    ml = laya.Agent(args.ckpt, device=device)
    print("ckpt", args.ckpt)
else:
    ml = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
q = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle body?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
            "sales": "pricing and new plans",
        },
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer ask for a refund in body?",
    },
}
cases = [
    ("th_refund", {"body": "คิดเงินซ้ำสองครั้งจากใบแจ้งหนี้ 4411 กรุณาคืนเงินวันนี้"}),
    ("th_outage", {"body": "แอปเข้าไม่ได้ตั้งแต่เช้า ระบบล่มทั้งบริษัท"}),
    ("en_refund", {"body": "Charged twice on invoice 4411. Please refund today."}),
]
for name, state in cases:
    t = time.perf_counter()
    out = ml.predict(state, q)["answers"]
    print(
        name,
        f"{(time.perf_counter()-t)*1000:.0f}ms",
        f"department={out['department']['choice']}",
        round(out["refund_requested"]["noul"], 3),
    )
