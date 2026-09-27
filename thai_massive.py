"""Official-style 20-way MASSIVE-th protocol (same as published 0.480)."""
from __future__ import annotations

import os
import random
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import laya
from datasets import load_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
n, n_opts, seed = 40, 20, 0
ds = load_dataset("mteb/amazon_massive_intent", "th", split="test")
labels = sorted({r["label_text"] for r in ds})
rng = random.Random(seed)
agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
n = min(n, len(ds))
ok, confs = 0, []
t0 = time.perf_counter()
for r in ds.select(range(n)):
    pool = [x for x in labels if x != r["label_text"]]
    keys = [r["label_text"]] + rng.sample(pool, n_opts - 1)
    rng.shuffle(keys)
    out = agent.predict(
        {"utterance": r["text"]},
        {
            "intent": {
                "type": "choice",
                "instructions": "What is the user asking for in utterance?",
                "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys},
            }
        },
    )["answers"]["intent"]
    ok += int(out["choice"] == r["label_text"])
    confs.append(float(out["confidence"]))
print(f"acc={ok/n:.3f} mean_conf={np.mean(confs):.3f} published=0.480")
print(f"{n} rows {(time.perf_counter()-t0)/n*1000:.0f} ms/row device={device}")
