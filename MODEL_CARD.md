# Laya multilingual, Thai typed decisions

Fine-tune of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (`subfolder="multilingual"`) for Thai typed decisions. The model does not write text. One forward pass stamps `choice`, `score`, and `noul` (yes/no) answers and returns a probability for each.

| | |
|---|---|
| Checkpoint to use | `checkpoints/wongnai_heavy_rlcd/epoch_02` |
| Same weights, softened confidence | `checkpoints/wongnai_heavy_rlcd/epoch_02_calibrated` |
| Base | mmBERT-base, 322M parameters, context 1,024 |
| Library | `laya` 0.3.6 |
| Training device | Apple M4 Max, PyTorch, fp32 |
| Date | 2026-09-26 |

`selected` in `checkpoints/wongnai_heavy_rlcd/` points at `epoch_02`. The calibrated directory symlinks the same weight file and changes only `rl_agent_config.json`.

## What it is for

Route a Thai sentence to a label the caller already named.

- Intent among a menu of at most 20 names.
- News topic among the 12 Prachathai topics, and a yes/no "is this that topic?"
- A rough yes/no on whether a hypothesis follows a premise, and a 3-way entailment label.
- A rough 1–5 star guess on a Thai review. This is the weak stamp.

The English refund check still lands on `billing`. Short English intent menus are in scope. Long English work is not.

## What it is not for

- Writing or extracting an answer span.
- A published star rating. Exact-rung accuracy on the held-out reviews is 0.599.
- Exams, math, code, handwriting, or images.
- A gate that treats a raw confidence of 0.99 as certainty. Use the calibrated copy when a number will accept or reject an action. The 3-way choice and the yes/no stamp are still somewhat overconfident after calibration.

## How to run it

```python
import os
os.environ["USE_TF"] = "0"

import laya

agent = laya.Agent(
    "checkpoints/wongnai_heavy_rlcd/epoch_02_calibrated",
    device="mps",
)
out = agent.predict(
    {"utterance": "คิดเงินซ้ำสองครั้ง กรุณาคืนเงินวันนี้"},
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs and outages",
                "sales": "pricing and new plans",
            },
        }
    },
)
print(out["answers"]["department"]["choice"])
```

`choice` returns the key it picked and a probability for every name on that slip. `noul` returns the probability the answer is yes. `score` returns an expected rung in `0 .. K-1` plus a probability per rung. A 1–5 star label is stored as 1–5 in the training files. The probability keys the model reads are `"0"` through `"4"`.

## Training

Three stages, each continuing from the previous weights. The loss is the upstream RLCD objective: Gaussian noise on the logits (group size 4), `proper_reward` with spherical weight 0.75, plus soft cross-entropy of weight 1.0. Optimizer is AdamW, encoder learning rate 2.5e-5, head learning rate 1e-4, weight decay 0.01, gradient clip 1.0. Effective batch size is 64 (microbatch 8, gradient accumulation 8). Training sequences were cut at 384 tokens. The saved config still advertises `max_len` 1024.

Noise starts at 0.4 and is scheduled down to 0.1 across a 4-epoch horizon. The last stage ran only two epochs, so it stopped at 0.30.

| Stage | Start | Train rows | Epochs kept |
|---|---|---|---|
| MASSIVE Thai only | multilingual base | 11,514 | epoch 6 of 8 |
| Mixed stamps | that epoch 6 | 83,028 rows, 139,752 questions | epoch 2 of 3 finished |
| Extra Wongnai reviews | that epoch 2 | 108,028 rows, 164,752 questions | epoch 2 of 2 |

The mixed file was MASSIVE Thai 11,514, XNLI Thai 20,000, Wongnai 15,000, Prachathai 25,000, and MASSIVE English 11,514. The last stage replaced the 15,000 Wongnai rows with all 40,000 labeled training reviews.

Rows the optimizer never saw:

- MASSIVE Thai test, 2,974. Menus are 20 names: the true intent plus 19 decoys, seed 13, Thai instruction. This is not the published 60-way number and not the 40-slip English-instruction probe.
- XNLI Thai validation, 2,490. The XNLI test split was not used.
- Prachathai validation, 6,721.
- Wongnai measure file, 6,203. This is the dataset's test split. There is no separate Wongnai validation split.

## Evaluation

Held-out accuracy of `epoch_02`. A question is correct when the argmax matches the gold label. A row with two questions counts twice in `n`.

| Set | Questions | Accuracy |
|---|---|---|
| MASSIVE Thai test, 20-way intent | 2,974 | 0.888 |
| XNLI Thai validation, 3-way choice | 2,490 | 0.714 |
| XNLI Thai validation, yes/no | 2,490 | 0.820 |
| XNLI Thai validation, both | 4,980 | 0.767 |
| Prachathai validation, 12-way topic | 3,501 | 0.888 |
| Prachathai validation, yes/no | 13,119 | 0.881 |
| Prachathai validation, both | 16,620 | 0.882 |
| Wongnai measure, exact star | 6,203 | 0.599 |

Reference points for the MASSIVE test file only: zero-shot multilingual 0.395, the earlier MASSIVE-only fine-tune 0.878. Chance on that file is 0.05. Chance on the 12-way topic menu is 0.083, on the 3-way entailment menu 0.333, and on five stars 0.20.

Smoke, uncalibrated `epoch_02`:

| Slip | Department | P(refund) |
|---|---|---|
| Thai double charge | billing | 1.000 |
| Thai outage | technical | 0.000 |
| English "Charged twice… refund today." | billing | 1.000 |

## Speed

Milliseconds per row on an Apple M4 Max, Metal, after the weights are loaded. One row is one slip: a MASSIVE intent, an XNLI pair with two questions, a Prachathai article with two or three questions, or a Wongnai review. Stock is the untouched `multilingual` checkpoint. Ours is `epoch_02`. iApp is `iapp/OpenThai-SystemOne` run locally.

Laya was timed one row per call. iApp was timed in batches of 4, and batches of 2 on Prachathai, which makes its per-row time look better than a one-row call would.

| File | Rows | Stock | Ours | iApp |
|---|---|---|---|---|
| MASSIVE Thai test | 2,974 | 18 ms | 16 ms | 195 ms |
| XNLI Thai validation | 2,490 | 18 ms | 18 ms | 175 ms |
| Prachathai validation | 6,721 | 59 ms | 62 ms | 616 ms |
| Wongnai measure | 6,203 | 21 ms | 20 ms | 547 ms |

Fine-tuning did not change the speed. The body is still the 322M encoder. iApp's 0.8B model takes about 10× as long on a short slip and about 25× as long on a review or a news article.

## Calibration

`epoch_02_calibrated` is `epoch_02` with temperatures written into `temperature_by_options`. Fit on MASSIVE validation, XNLI validation, and Prachathai validation. The Wongnai measure file was not used, so `score:3-5` stays at 1.0.

A temperature divides the logits of one question. It does not change the argmax. Laya clamps temperatures to [0.5, 5.0].

| Bucket | Questions | Temperature | ECE before → after |
|---|---|---|---|
| `choice:11+` (12- and 20-way) | 5,534 | 3.30 | 0.096 → 0.030 |
| `choice:3-5` | 2,490 | 5.00 | 0.251 → 0.092 |
| `noul:2` | 15,609 | 5.00 | 0.109 → 0.032 |

The 3-way and yes/no fits both sat on the ceiling of 5, so those confidences are still a bit sharp. On the calibrated copy the English refund is still `billing`, and P(refund) is 0.949 rather than 1.000.

## Limits

- Menus longer than 20 names were not trained. MASSIVE has 60 intents. The 0.888 number is the 20-name protocol.
- Star errors are exact-match. The card does not say whether the misses are off by one.
- Training saw at most 384 tokens. Longer inputs are cut by the library at 1,024.
- No Thai exam, toxicity, wiki answerability, Wisesight sentiment, or SIB-200 result is claimed.
- Confidence on `choice:3-5` and `noul:2` is improved and still not tight.

## License

CC BY-NC 4.0. Non-commercial use only.

The base Laya checkpoint is Apache-2.0. This fine-tune also trained on XNLI (CC BY-NC 4.0) and on Prachathai, whose Hugging Face card says CC BY-NC. The Wongnai review corpus is LGPL-3.0. Those terms are why these weights are not Apache-2.0.
