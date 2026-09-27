"""RLCD fine-tune of multilingual Laya on the MASSIVE-th train split.

Four epochs on the notebook schedule. Epochs 5–8 run only when validation
accuracy at epoch 4 is strictly above epoch 3. One checkpoint is kept per
epoch. This script learns only from the train split.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import (
    QTYPES,
    build_model,
    build_sequence,
    collate_items,
    proper_reward,
    render_options,
)
from train_recipe import (
    BASE_EPOCHS,
    CE_WEIGHT,
    ETA_MIN,
    GRAD_ACCUM,
    GRAD_CLIP,
    GROUP_SIZE,
    LR_ENCODER,
    LR_HEAD,
    MAX_EPOCHS,
    MICROBATCH,
    PRECISION,
    SIGMA_END,
    SIGMA_START,
    W_SPH,
    WEIGHT_DECAY,
    cosine_horizon,
    extend_after_four,
    learning_rate,
    load_train_rows,
    select_epoch,
    sigma_for_epoch,
    training_files,
    val_jsonl,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "checkpoints" / "massive_th_rlcd"


def multilingual_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            "convaiinnovations/laya",
            allow_patterns=["multilingual/*"],
        )
    )
    path = root / "multilingual"
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return path


def rows_to_items(rows: list[dict], tok, cfg: dict) -> list[dict]:
    max_len = int(cfg.get("max_len", 1024))
    head_max_len = int(cfg.get("head_max_len", 256))
    items: list[dict] = []
    skipped = 0
    for row in rows:
        state = row["state"]
        golds = row["gold"]
        for qid, question in row["questions"].items():
            if qid not in golds:
                continue
            gold = golds[qid]
            kind = question["type"]
            crit = question.get("criteria")
            internal = {"t": kind, "ins": question["instructions"], "crit": crit}
            if kind == "choice":
                keys = list(crit.keys())
                target = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            elif kind == "noul":
                probs = gold["probabilities"]
                target = [float(probs.get("false", 0.5)), float(probs.get("true", 0.5))]
            elif kind == "score":
                n_levels = len(crit) if isinstance(crit, list) else 4
                target = [float(gold["probabilities"].get(str(i), 0.0)) for i in range(n_levels)]
            else:
                skipped += 1
                continue
            total = sum(target)
            target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)
            seq, markers = build_sequence(tok, state, internal, max_len, head_max_len)
            if len(markers) != len(render_options(internal)):
                skipped += 1
                continue
            items.append(
                {
                    "ids": seq,
                    "markers": markers,
                    "qtype": QTYPES[kind],
                    "target": target,
                    "label": max(range(len(target)), key=target.__getitem__),
                }
            )
    print(f"items={len(items)} skipped={skipped}", flush=True)
    if not items:
        raise SystemExit("no training items")
    return items


def rlcd_loss(logits, act, target, qtype, mask, sigma: float):
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float().clamp(min=1)
    eps = torch.randn((GROUP_SIZE,) + tuple(logits.shape), device=logits.device, dtype=logits.dtype)
    eps = eps * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    reported = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(reported, target.unsqueeze(0), qtype, mask, w_sph=W_SPH, w_rps=1.0)
        adv = reward - reward.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    raw = loss_rl + CE_WEIGHT * loss_ce
    scaled = raw / GRAD_ACCUM + act.sum() * 0.0
    return scaled, raw.detach(), reward.detach().mean()


def load_student(src: Path, device: torch.device):
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    model = build_model(cfg, encoder_dir=str(src / "encoder"))
    model.load_state_dict(load_file(str(src / "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    model.to(device)
    model.train()
    tok = AutoTokenizer.from_pretrained(str(src / "tokenizer"))
    return model, tok, cfg


def save_checkpoint(model, tok, cfg: dict, src: Path, out_dir: Path, epoch: int, avg_loss: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous().clone()
        if tensor.is_floating_point():
            tensor = tensor.half()
        tensors[name] = tensor
    save_file(tensors, str(out_dir / "model.safetensors"))
    encoder_dir = out_dir / "encoder"
    encoder_dir.mkdir(exist_ok=True)
    shutil.copyfile(src / "encoder" / "config.json", encoder_dir / "config.json")
    tok.save_pretrained(out_dir / "tokenizer")
    written = dict(cfg)
    written["fine_tuned"] = True
    written["training"] = {
        "epochs_completed": epoch,
        "base_epochs": BASE_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "avg_loss": avg_loss,
        "world_size": 1,
        "fine_tuned_from_checkpoint": True,
        "precision": PRECISION,
    }
    (out_dir / "rl_agent_config.json").write_text(json.dumps(written, indent=2))


def parse_eval_line(stdout: str) -> tuple[int, float, str]:
    for line in stdout.splitlines():
        if " n=" in line and " acc=" in line:
            n = int(line.split(" n=")[1].split()[0])
            acc = float(line.split(" acc=")[1].split()[0])
            return n, acc, line
    raise SystemExit(f"eval produced no accuracy line:\n{stdout}")


def score_validation(ckpt: Path) -> tuple[int, float, str]:
    val_path = val_jsonl(ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval_massive_th.py"),
            "--ckpt",
            str(ckpt),
            "--file",
            str(val_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"validation eval failed\n{proc.stdout}\n{proc.stderr}")
    n, acc, line = parse_eval_line(proc.stdout)
    if n != 2033:
        raise SystemExit(f"validation eval returned n={n}, expected 2033\n{line}")
    return n, acc, line


def link_selected(out_dir: Path, epoch: int) -> Path:
    target = out_dir / f"epoch_{epoch:02d}"
    link = out_dir / "selected"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)
    return link


def fit(
    items: list[dict],
    model,
    tok,
    cfg: dict,
    src: Path,
    out_dir: Path,
    device: torch.device,
    epoch_limit: int | None = None,
) -> Path:
    enc = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    if not enc or not head:
        raise SystemExit("expected a trainable encoder and a decision head")
    optimizer = torch.optim.AdamW(
        [
            {"params": enc, "lr": LR_ENCODER},
            {"params": head, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    horizon = cosine_horizon(len(items))
    print(
        f"base_epochs={BASE_EPOCHS} max_epochs={MAX_EPOCHS} n_items={len(items)} "
        f"microbatch={MICROBATCH} grad_accum={GRAD_ACCUM} "
        f"lr_encoder={LR_ENCODER} lr_head={LR_HEAD} weight_decay={WEIGHT_DECAY} "
        f"grad_clip={GRAD_CLIP} eta_min={ETA_MIN} group_size={GROUP_SIZE} "
        f"sigma={SIGMA_START}->{SIGMA_END} w_sph={W_SPH} ce_weight={CE_WEIGHT} "
        f"precision={PRECISION} device={device.type} cosine_horizon={horizon}",
        flush=True,
    )
    opt_steps = 0
    limit = BASE_EPOCHS if epoch_limit is None else epoch_limit
    val_accs: list[float] = []
    epoch_index = 0
    while epoch_index < limit:
        random.Random(42 + epoch_index).shuffle(items)
        sigma = sigma_for_epoch(epoch_index)
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        n_micro = 0
        last_enc = LR_ENCODER
        last_head = LR_HEAD
        t0 = time.perf_counter()
        for start in range(0, len(items), MICROBATCH):
            chunk = items[start : start + MICROBATCH]
            batch = collate_items([chunk], tok.pad_token_id)
            logits, act = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            scaled, raw, reward = rlcd_loss(
                logits,
                act,
                batch["target"].to(device),
                batch["qtype"].to(device),
                batch["marker_mask"].to(device),
                sigma,
            )
            if not torch.isfinite(raw):
                raise SystemExit(f"non-finite loss at epoch {epoch_index + 1} micro {n_micro}")
            scaled.backward()
            loss_sum += float(raw)
            n_micro += 1
            accum += 1
            end = start + MICROBATCH >= len(items)
            if accum == GRAD_ACCUM or end:
                last_enc = learning_rate(LR_ENCODER, opt_steps, horizon)
                last_head = learning_rate(LR_HEAD, opt_steps, horizon)
                optimizer.param_groups[0]["lr"] = last_enc
                optimizer.param_groups[1]["lr"] = last_head
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                opt_steps += 1
                if opt_steps % 20 == 0:
                    print(
                        f"epoch={epoch_index + 1} opt_step={opt_steps} "
                        f"micro_loss={float(raw):.4f} reward={float(reward):.3f} "
                        f"sigma={sigma:.4f} lr_encoder={last_enc:.6e}",
                        flush=True,
                    )
        if device.type == "mps":
            torch.mps.synchronize()
        avg = loss_sum / max(1, n_micro)
        epoch_num = epoch_index + 1
        ckpt = out_dir / f"epoch_{epoch_num:02d}"
        save_checkpoint(model, tok, cfg, src, ckpt, epoch_num, avg)
        print(
            f"epoch_done={epoch_num} avg_loss={avg:.4f} sigma={sigma:.4f} "
            f"lr_encoder={last_enc:.6e} lr_head={last_head:.6e} "
            f"opt_step={opt_steps} sec={time.perf_counter() - t0:.1f} checkpoint={ckpt}",
            flush=True,
        )
        _n, acc, line = score_validation(ckpt)
        val_accs.append(acc)
        print(line, flush=True)
        if epoch_limit is None and epoch_num == BASE_EPOCHS:
            extend = extend_after_four(val_accs[2], val_accs[3])
            print(
                f"extend={int(extend)} epoch3_acc={val_accs[2]:.3f} epoch4_acc={val_accs[3]:.3f}",
                flush=True,
            )
            if extend:
                limit = MAX_EPOCHS
        epoch_index += 1
    chosen = select_epoch(val_accs)
    selected = link_selected(out_dir, chosen)
    print(
        f"selected_epoch={chosen} selected_acc={val_accs[chosen - 1]:.3f} "
        f"finished_epochs={epoch_index} checkpoint={selected.resolve()}",
        flush=True,
    )
    return selected


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise SystemExit("MPS is required for this run")
    train_path = training_files(ROOT)[0]
    print(
        f"train_file={train_path} base_epochs={BASE_EPOCHS} max_epochs={MAX_EPOCHS} "
        f"precision={PRECISION}",
        flush=True,
    )
    src = multilingual_snapshot()
    model, tok, cfg = load_student(src, device)
    items = rows_to_items(load_train_rows(ROOT), tok, cfg)
    fit(items, model, tok, cfg, src, DEFAULT_OUT, device)


if __name__ == "__main__":
    main()
