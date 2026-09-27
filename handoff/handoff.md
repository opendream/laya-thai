# Handoff: Fine-tune Laya for general Thai (not tickets-only)

**Date:** 2026-09-24  
**Operator:** Keng (macOS 27, Apple Silicon)  
**Status:** Local eval done. MASSIVE-th converter written. Training not started.  
**Goal:** Make `laya-multilingual` generally good at Thai typed decisions (`choice` / `score` / `noul`), not only support tickets.

---

## 1. Mission and non-goals

Stay inside the official Laya family. Fine-tune `convaiinnovations/laya` with `subfolder="multilingual"` (mmBERT-base, 322M). Do not wrap OpenThai-SystemOne and call it Laya.

Non-goals: 5B-token continued pretrain, ticket-only data, training the English-root checkpoint on Thai script, a single Laya `choice` with 60–77 options, inventing `agent.train()`.

---

## 2. What Laya is

Non-autoregressive System 1 model. No text generation. Input = `state` + typed `questions`. Output = probabilities in one forward pass.

| Primitive | Returns |
|---|---|
| `choice` | one label from a named list + distribution |
| `score` | ordinal level (e.g. 1–5 stars) |
| `noul` | P(yes) in `[0, 1]` |

Three checkpoints in one Hub repo:

| Checkpoint | Encoder | When to use |
|---|---|---|
| repo root | ModernBERT-large 421M | English only |
| `multilingual` | mmBERT-base 322M | **this project** |
| `typed-decisions` | ModernBERT-large 421M | four English workflows; do not start from here |

```python
agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="mps")
```

`laya>=0.3.6` (user ran **0.3.6**). MPS autocast was broken before that. There is **no** public `agent.train()`. Official training is RLCD on 2×T4:

https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

---

## 3. Environment that already works

macOS 27, `arm64` (not Rosetta), `uv`, Python 3.12, PyTorch MPS.

```zsh
export USE_TF=0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
uv add "laya>=0.3.6" datasets numpy huggingface-hub
```

`USE_TF=0` is mandatory or `laya.load()` can hang. No CUDA wheel. uv installer: `curl -LsSf https://astral.sh/uv/install.sh | sh` then `source "$HOME/.local/bin/env"`.

---

## 4. Verified Mac results (2026-09-24)

Smoke (`department` choice, then refund `noul`):

| Case | Output | Verdict |
|---|---|---|
| `th_refund` | `billing` 0.996 | correct |
| `th_outage` | `technical` 0.0 | correct |
| `en_refund` | `billing` 0.994 | English still works |

MASSIVE-th, 20-way, N=40, MPS:

```
acc=0.475 mean_conf=0.764 published=0.480
40 rows 20 ms/row device=mps
```

Published 51-language table (`BENCHMARKS.md` row `th`): English checkpoint **0.080**, multilingual **0.480**, Δ +0.400. Chance = 0.05.

Overconfident: conf 0.764 vs acc 0.475. Multilingual ships uncalibrated (card: ECE 0.314 → 0.106 after temperature fit). Do not gate on raw confidence.

**Do not use 0.475 as the post-train bar.** Re-eval the full test JSONL and beat *that* number.

---

## 5. OpenThai-SystemOne (syllabus only)

Same contract, different body: Qwen3.5-0.8B + 256-slot head, ~5B Thai CPT, 2–3M decision SFT, Brier + per-type temperatures. v0.3 public macro 74.3. MASSIVE-th **60-way 90.0**. That is not Laya’s 20-way test.

Copy: multi-task mix, Thai instruction text, option shuffle, dual primitive on one state, hold-outs, measure-then-synth, calibrate last.

Do not copy: Qwen surgery, 255 options, 5B CPT, 2–3M rows on day one.

iApp Hub “datasets” are mostly LLM exams. Only `iapp/iapp_wiki_qa_squad` belongs in Laya **train**. Keep `iapp/openthaieval` as hold-out.

Converter to copy mappings from:

https://github.com/iapp-technology/openthai-systemone/blob/main/scripts/03_decision_data.py

---

## 6. Files in the kit

| File | Role |
|---|---|
| `convert_massive_th.py` | MASSIVE-th → JSONL, 20 opts, seed 13 |
| `eval_massive_th.py` | `--file` `--limit` `--ckpt` |
| `prefetch.py` | cache weights + MASSIVE-th |
| `peek.py` | validate JSONL |
| `thai_smoke.py` / `thai_massive.py` | original probes |
| `check_mac.py` | MPS / versions |
| `KAGGLE.md` | notebook deltas |
| `README.md` | Thai operator checklist |
| `HANDOFF.md` | this document |

Schema already in use:

```json
{"state": {"utterance": "…"}, "questions": {"intent": {"type": "choice", "instructions": "…", "criteria": {}}}, "gold": {"intent": {"label": "alarm_set", "probabilities": {"alarm_set": 1.0}}}}
```

Source dataset: `mteb/amazon_massive_intent`, config `th`, fields `text`, `label_text`.

---

## 7. Dataset plan

**Train (phase 1), ~50k–150k decisions**

| Source | Type | Split |
|---|---|---|
| MASSIVE-th | choice ≤20 | train only |
| `facebook/xnli` `th` | 3-way choice + entail noul | train |
| Prachathai67k | topic choice + is-topic noul | train |
| Wongnai reviews | score 1–5 | train |
| `iapp/iapp_wiki_qa_squad` | answerable noul | train |
| MASSIVE-en or AG News | choice | ~10–20% of mix |

Suggested mix: MASSIVE-th 30%, Prachathai 25%, XNLI-th 20%, Wongnai 15%, Wiki QA 10%, then carve ~10% English.

**Hold out**

MASSIVE-th test · XNLI-th val/test · Prachathai val · Wongnai test · SIB-200 Thai whole · Wisesight whole · OpenThaiEval whole.

**Skip for now**

Handwriting, AIME/math, HumanEval-TH, chat SFT/DPO, images, 5B CPT, Banking77 as 77-way.

**Phase 2** only after a scoreboard exists: XQuAD-th, Thai toxicity, a Thai exam slice, targeted synth (OpenThai v0.2/v0.3 pattern).

---

## 8. Training path

1. `eval_massive_th.py` on **full** `data/massive_th.test.jsonl` → write `n` and `acc`.
2. Official Kaggle notebook. Base = multilingual snapshot. Train file = our JSONL. Never open test in the loop. Save `checkpoint_latest/` (`model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`).
3. `eval_massive_th.py --ckpt /path/to/checkpoint_latest`.
4. Fit temperatures on val. Then consider phase 2.

Mac head-only CE (freeze `encoder.*`, ~26M head, AdamW) is unofficial. Internals: `Agent._infer` batch keys `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`. Do not invent a public fit API. Do not report Mac-CE numbers as the project result.

Honest phase-1 targets on **Laya’s 20-way protocol**: MASSIVE-th 0.65–0.80; XNLI-th beat 0.55 first; English refund smoke must stay correct.

---

## 9. Hard rules

- Never train on test.
- Never put 60 options in one Laya `choice`.
- Keep an English slice.
- Do not start from `typed-decisions` or the English root.
- Do not compare OpenThai 90% @ 60-way to Laya 47.5% @ 20-way.
- Do not `Router(preload=True)` during this work (~2.3 GB).
- Keep eval on PyPI `laya`, not `laya-mlx`, so numbers stay comparable to 0.480.

---

## 10. Next actions

1. Confirm `data/massive_th.{train,val,test}.jsonl` (`peek.py`). If missing, `prefetch.py` then `convert_massive_th.py`.
2. Full `eval_massive_th.py` → record baseline.
3. Write converters for XNLI-th, Wongnai, Prachathai, Wiki QA (same schema).
4. Merge `data/train.jsonl` + manifest of source counts.
5. Kaggle RLCD on the multilingual snapshot (`KAGGLE.md`).
6. `eval --ckpt` vs step-2 baseline + `thai_smoke.py`.
7. Only then synth weak spots and calibrate.

If the next operator has only a Mac: stop after step 4. Converters plus a frozen baseline beat a broken MPS trainer.

---

## 11. References

- https://huggingface.co/convaiinnovations/laya
- https://huggingface.co/convaiinnovations/laya-multilingual
- https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md
- Fine-tune notebook path in §2
- MASSIVE: arXiv:2204.08582 · `mteb/amazon_massive_intent`
- mmBERT: `jhu-clsp/mmBERT-base`
- https://huggingface.co/iapp/OpenThai-SystemOne
- https://github.com/iapp-technology/openthai-systemone
- `iapp/iapp_wiki_qa_squad` · PyThaiNLP/prachathai-67k · wongnai-corpus · `facebook/xnli` · Wisesight (hold-out)
- Guo et al. 2017 calibration, arXiv:1706.04599
- Head-only CE precedent (unofficial): https://n.demir.io/articles/testing-and-fine-tuning-laya/

User Mac numbers in §4 beat any remembered 40-row figure if they disagree.
