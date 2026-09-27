# Laya Thai

Fine-tune of multilingual Laya for Thai typed decisions. The model stamps `choice`, `score`, and `noul`. It does not write text.

Published weights, calibrated: [team-od/laya-thai](https://huggingface.co/team-od/laya-thai).  
The same weights also sit at [team-od/laya-thai](https://huggingface.co/team-od/laya-thai).  
License: CC BY-NC 4.0.

Local checkpoint to continue from:

- `checkpoints/wongnai_heavy_rlcd/epoch_02`
- `checkpoints/wongnai_heavy_rlcd/epoch_02_calibrated` (same weights, softer confidence)
- `selected` points at `epoch_02`

## Scores

Held-out, exact label match, `epoch_02`.

| Set | Accuracy |
|---|---|
| MASSIVE Thai test, 20-way, 2,974 | 0.888 |
| XNLI validation, 3-way / yes-no | 0.714 / 0.820 |
| Prachathai validation, topic / yes-no | 0.888 / 0.881 |
| Wongnai measure, exact star, 6,203 | 0.599 |

Stock multilingual Laya on the same files: intent 0.395, topic 0.371, topic yes/no 0.600, entailment yes/no 0.805, 3-way 0.677, stars 0.255.  
iApp OpenThai-SystemOne, one option order: intent 0.905, topic 0.943, topic yes/no 0.904, entailment yes/no 0.843, 3-way 0.772, stars 0.630.  
On this Mac a short slip is about 16 ms here and about 195 ms for iApp.

## Do not train on

- `data/massive_th.test.jsonl`
- `data/xnli_th.val.jsonl`
- `data/prachathai.val.jsonl`
- `data/wongnai_measure.jsonl` (Wongnai's test split)

The heavy mix that produced the published checkpoint is `data/train_wongnai_heavy.jsonl`. Manifest: `data/train_wongnai_heavy_manifest.json`. `data/train.jsonl` is the earlier mix with 15,000 Wongnai rows. Those two merged files are not in git. Each is over GitHub's 100 MB limit. Rebuild them with `merge_train.py` and `train_wongnai_heavy.py`. The source JSONL files are in `data/`.

## What to do next

More reviews of the same kind barely moved stars (0.589 to 0.599) and did not move the other scores. The next change, if stars must improve, is the score loss: a far miss and a near miss are the same error now. Fit temperatures only after a new scoreboard, and not on `wongnai_measure.jsonl`.

```zsh
export USE_TF=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
uv run python eval_massive_th.py --ckpt checkpoints/wongnai_heavy_rlcd/epoch_02 --file data/massive_th.test.jsonl
uv run python eval_decisions.py --ckpt checkpoints/wongnai_heavy_rlcd/epoch_02 --file data/wongnai_measure.jsonl
uv run python thai_smoke.py --ckpt checkpoints/wongnai_heavy_rlcd/epoch_02
```

`MODEL_CARD.md` is the long card. `KAGGLE.md` is the old two-T4 note and is not how this checkpoint was trained.
