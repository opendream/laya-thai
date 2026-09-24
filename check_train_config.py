"""Assert the shipped training setup: hyperparameters, 4-or-8 rule, train split only."""
from __future__ import annotations

from pathlib import Path

import train_massive
import train_recipe

ROOT = Path(__file__).resolve().parent


def main() -> None:
    files = train_recipe.training_files(ROOT)
    rows = train_recipe.load_train_rows(ROOT)
    counted = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            counted += sum(1 for line in handle if line.strip())

    assert train_recipe.BASE_EPOCHS == 4
    assert train_recipe.MAX_EPOCHS == 8
    assert train_massive.BASE_EPOCHS == 4
    assert train_massive.MAX_EPOCHS == 8
    assert train_massive.MICROBATCH == train_recipe.MICROBATCH == 8
    assert train_massive.GRAD_ACCUM == train_recipe.GRAD_ACCUM == 8
    assert train_massive.LR_ENCODER == train_recipe.LR_ENCODER == 2.5e-5
    assert train_massive.LR_HEAD == train_recipe.LR_HEAD == 1e-4
    assert train_recipe.WEIGHT_DECAY == 0.01
    assert train_recipe.GRAD_CLIP == 1.0
    assert train_recipe.GROUP_SIZE == 4
    assert train_recipe.W_SPH == 0.75
    assert train_recipe.CE_WEIGHT == 1.0
    assert train_recipe.PRECISION == "fp32"
    assert train_recipe.ETA_MIN == 1e-6

    assert len(files) == 1
    assert files[0].name == train_recipe.TRAIN_NAME
    assert files[0].is_file()
    assert len(rows) == counted
    assert train_recipe.val_jsonl(ROOT).name == train_recipe.VAL_NAME
    assert train_recipe.val_jsonl(ROOT).name not in {path.name for path in files}

    assert train_recipe.sigma_for_epoch(0) == 0.4
    assert train_recipe.sigma_for_epoch(3) == 0.1
    assert train_recipe.sigma_for_epoch(4) == 0.1
    assert train_recipe.sigma_for_epoch(7) == 0.1

    horizon = train_recipe.cosine_horizon(len(rows))
    assert horizon == train_recipe.optimizer_steps(len(rows), epochs=4)
    encoder_lrs = [train_recipe.learning_rate(train_recipe.LR_ENCODER, i, horizon) for i in range(horizon + 3)]
    assert encoder_lrs[0] == train_recipe.LR_ENCODER
    assert abs(encoder_lrs[horizon - 1] - train_recipe.ETA_MIN) < 1e-8
    assert encoder_lrs[horizon] == train_recipe.ETA_MIN
    assert encoder_lrs[horizon + 2] == train_recipe.ETA_MIN
    assert all(earlier >= later - 1e-15 for earlier, later in zip(encoder_lrs, encoder_lrs[1:]))

    assert train_recipe.extend_after_four(0.40, 0.41) is True
    assert train_recipe.extend_after_four(0.41, 0.41) is False
    assert train_recipe.extend_after_four(0.42, 0.41) is False
    assert train_recipe.select_epoch([0.2, 0.5, 0.5, 0.1]) == 2
    assert train_recipe.select_epoch([0.4, 0.2, 0.3, 0.1]) == 1

    held_out = ROOT / "data" / "massive_th.test.jsonl"
    assert all(path.resolve() != held_out.resolve() for path in files)
    assert all(path.name != held_out.name for path in files)

    print(
        f"base_epochs={train_massive.BASE_EPOCHS} max_epochs={train_massive.MAX_EPOCHS} "
        f"train_file={files[0]} n={len(rows)} "
        f"microbatch={train_recipe.MICROBATCH} grad_accum={train_recipe.GRAD_ACCUM} "
        f"lr_encoder={train_recipe.LR_ENCODER} lr_head={train_recipe.LR_HEAD} "
        f"sigma_epoch1={train_recipe.sigma_for_epoch(0):.3f} "
        f"sigma_epoch4={train_recipe.sigma_for_epoch(3):.3f} "
        f"sigma_epoch8={train_recipe.sigma_for_epoch(7):.3f} "
        f"horizon_steps={horizon} lr_epoch4={encoder_lrs[horizon - 1]:.6e} "
        f"lr_after={encoder_lrs[horizon]:.6e} "
        f"precision={train_recipe.PRECISION} test_path_absent=1"
    )


if __name__ == "__main__":
    main()
