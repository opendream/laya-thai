"""Fit one temperature per Laya option-count bucket on validation slips.

Reads epoch 2 with its shipped temperatures (all 1.0), so the probabilities
are the raw softmax. Writes a sibling directory with the same weights and a
new rl_agent_config.json. Argmax is unchanged by a single temperature.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import laya
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "checkpoints" / "wongnai_heavy_rlcd" / "epoch_02"
DST = ROOT / "checkpoints" / "wongnai_heavy_rlcd" / "epoch_02_calibrated"
VAL_FILES = (
    ROOT / "data" / "massive_th.val.jsonl",
    ROOT / "data" / "xnli_th.val.jsonl",
    ROOT / "data" / "prachathai.val.jsonl",
)
TEMP_LO = 0.5
TEMP_HI = 5.0


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_index(question: dict, gold: dict) -> int:
    kind = question["type"]
    if kind == "choice":
        keys = list(question["criteria"].keys())
        return keys.index(gold["label"])
    if kind == "noul":
        return 1 if bool(gold["label"]) else 0
    if kind == "score":
        return int(gold["label"]) - 1
    raise ValueError(kind)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    out = np.exp(scaled)
    return out / out.sum()


def row_logits(agent, state, questions: dict) -> dict[str, np.ndarray]:
    ids = list(questions)
    items = []
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    for qid in ids:
        internal = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, internal, max_len, head_max_len)
        if len(markers) != len(render_options(internal)):
            raise ValueError(f"{qid} options exceed head_max_len")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[internal["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad():
        logits, _act = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
    logits = logits.float().cpu().numpy()
    out = {}
    for row_index, qid in enumerate(ids):
        k = len(items[row_index]["markers"])
        out[qid] = logits[row_index, :k].astype(np.float64)
    return out


def nll(rows: list[tuple[np.ndarray, int]], temperature: float) -> float:
    total = 0.0
    for probs, gold in rows:
        cooled = apply_temperature(probs, temperature)
        total -= np.log(max(float(cooled[gold]), 1e-12))
    return total / len(rows)


def ece(rows: list[tuple[np.ndarray, int]], temperature: float, bins: int = 10) -> float:
    conf = np.empty(len(rows))
    hit = np.empty(len(rows))
    for i, (probs, gold) in enumerate(rows):
        cooled = apply_temperature(probs, temperature)
        conf[i] = cooled.max()
        hit[i] = float(int(cooled.argmax()) == gold)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / len(rows)) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return total


def fit_scalar(rows: list[tuple[np.ndarray, int]]) -> float:
    grid = np.linspace(TEMP_LO, TEMP_HI, 46)
    best_t = 1.0
    best_loss = nll(rows, best_t)
    for temperature in grid:
        loss = nll(rows, float(temperature))
        if loss < best_loss:
            best_loss = loss
            best_t = float(temperature)
    # Local polish around the grid winner.
    for temperature in np.linspace(max(TEMP_LO, best_t - 0.1), min(TEMP_HI, best_t + 0.1), 21):
        loss = nll(rows, float(temperature))
        if loss < best_loss:
            best_loss = loss
            best_t = float(temperature)
    return best_t


def collect(agent) -> dict[str, list[tuple[np.ndarray, int]]]:
    grouped: dict[str, list[tuple[np.ndarray, int]]] = {}
    for path in VAL_FILES:
        for row in load_rows(path):
            logits = row_logits(agent, row["state"], row["questions"])
            for qid, question in row["questions"].items():
                if question["type"] == "score":
                    continue
                gold = gold_index(question, row["gold"][qid])
                vector = logits[qid]
                bucket = temp_bucket(QTYPES[question["type"]], len(vector))
                grouped.setdefault(bucket, []).append((vector, gold))
    return grouped


def write_checkpoint(temperatures: dict[str, float]) -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for name in ("model.safetensors", "encoder", "tokenizer"):
        os.symlink(SRC / name, DST / name)
    cfg = json.loads((SRC / "rl_agent_config.json").read_text())
    by_options = dict(cfg.get("temperature_by_options") or {})
    by_options.update(temperatures)
    cfg["temperature_by_options"] = by_options
    cfg["calibration"] = {
        "fit_on": [str(path.relative_to(ROOT)) for path in VAL_FILES],
        "buckets": temperatures,
    }
    (DST / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    agent = laya.Agent(str(SRC), device=device)
    grouped = collect(agent)
    fitted = {}
    for bucket, rows in sorted(grouped.items()):
        temperature = fit_scalar(rows)
        before_nll = nll(rows, 1.0)
        after_nll = nll(rows, temperature)
        before_ece = ece(rows, 1.0)
        after_ece = ece(rows, temperature)
        fitted[bucket] = temperature
        print(
            f"bucket={bucket} n={len(rows)} T={temperature:.3f} "
            f"nll={before_nll:.4f}->{after_nll:.4f} "
            f"ece={before_ece:.4f}->{after_ece:.4f}",
            flush=True,
        )
    write_checkpoint(fitted)
    print(f"checkpoint={DST}", flush=True)


if __name__ == "__main__":
    main()
