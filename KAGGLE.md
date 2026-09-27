# After landing: official RLCD on Kaggle 2xT4

Laya has no `agent.train()`. The official recipe is:

https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

## What to change

1. Base checkpoint = multilingual, not English root.

```python
import laya
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
```

The notebook then trains from that snapshot folder (`rl_agent_config.json`, `encoder/`, `tokenizer/`, `model.safetensors`).

2. Training rows = `data/massive_th.train.jsonl` only.
3. Validation = `data/massive_th.val.jsonl`.
4. Never open `data/massive_th.test.jsonl` inside the training loop.

Each JSONL line is already:

```json
{"state": {...}, "questions": {...}, "gold": {...}}
```

`gold.intent.probabilities` is the soft target the notebook calls `target`.

## After Kaggle finishes

Download `checkpoint_latest/` to the Mac, then:

```zsh
export USE_TF=0 PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python eval_massive_th.py --ckpt /path/to/checkpoint_latest
```

Compare `acc=` to the zero-shot line from the same file. That is the only fair delta.

Do not start from `laya-typed-decisions`. That specialist is for four English workflows.
