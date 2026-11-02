# Held-out validation for fine-tuning

This repo keeps validation as a sidecar process instead of changing `scripts/train.py`.
Train writes checkpoints as usual; the validator watches those checkpoint folders,
computes loss on held-out LeRobot data, and logs `val/*` metrics to Weights & Biases.

## Prepare a validation set

Use full held-out episodes or tasks, not random frames from the same episodes used for
training. Put them in LeRobot format and make sure they use the same observation/action
contract as the training config.

For one held-out dataset:

```bash
VAL_REPO=/path/to/heldout_lerobot
```

For multiple held-out datasets, repeat `--val-repo-id`.

## Launch training

```bash
export HF_HOME=/tmp/hf-home-$USER
CONFIG=pi05_genie_sim_manip_20260613
EXP=my_pi05_run
WANDB_MODE=online bash scripts/train.sh "$CONFIG" "$EXP"
```

Training writes checkpoints to:

```text
checkpoints/<CONFIG>/<EXP>/<step>/
```

## Run validation once

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config-name "$CONFIG" \
  --checkpoint-dir "checkpoints/$CONFIG/$EXP/5000" \
  --val-repo-id "$VAL_REPO" \
  --max-batches 20 \
  --batch-size 32 \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

## Watch checkpoints while fine-tuning

Run this in a second shell on the workstation:

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --watch \
  --poll-seconds 60 \
  --max-batches 20 \
  --batch-size 32 \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Equivalent with uv if your shell has it on PATH. The `HF_HOME` export gives Hugging Face a writable cache on this workstation:

```bash
/home/datamentors/.local/bin/uv run python scripts/validate_checkpoint.py --help
```

Logged metrics:

- `val/loss`
- `val/loss_std`
- `val/batches`
- `val/examples`
- `val/elapsed_s`
- `val/checkpoint_step`

The validator loads normalization stats from each checkpoint's `assets/norm_stats.json`,
so validation uses the same normalization as the checkpoint being scored.


## Action-chunk validation

`validate_actions.py` runs real policy inference on held-out observations, then
compares predicted action chunks against recorded future actions.

Modes:

- `all-frames`: score many frames from the held-out dataset. This is dense and is the default.
- `first-frame`: score only episode starts where `frame_index == 0` is available. If episode starts are not discoverable, it falls back to `--sample-stride`.

Dense validation:

```bash
.venv/bin/python scripts/validate_actions.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --mode all-frames \
  --max-samples 128 \
  --watch \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-action-val" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Episode-start validation:

```bash
.venv/bin/python scripts/validate_actions.py \
  --config-name "$CONFIG" \
  --checkpoint-root "checkpoints/$CONFIG/$EXP" \
  --val-repo-id "$VAL_REPO" \
  --mode first-frame \
  --max-samples 64 \
  --watch \
  --wandb-project ACoT-VLA \
  --wandb-run-name "${EXP}-action-val-first" \
  --wandb-group "$EXP" \
  --wandb-mode online
```

Action metrics logged:

- `val_actions/mse`
- `val_actions/mae`
- `val_actions/first_mse`
- `val_actions/first_mae`
- `val_actions/final_mse`
- `val_actions/final_mae`
- `val_actions/dim_XX_mae`
- `val_actions/sample_table`

### Hold baseline and gain scores

Raw MAE has no reference point. On slow tasks (towel folding moves ~0.04 rad on
average over a 1 s chunk) a dummy policy that outputs "stay at the current state"
for all 30 steps (the robot does not move) already scores a low MAE. Every
checkpoint is therefore also compared against that hold baseline, on the same
samples and dims:

    gain_vs_hold = 1 - model_mae / baseline_hold_mae
    0 = no better than not moving, 1 = perfect, < 0 = worse than not moving

"Ground truth chunk" = the next 30 recorded actions from the val episode, starting
at the sampled frame.

- `val_actions/baseline_hold_mae`, `baseline_hold_first_mae`, `baseline_hold_final_mae`
- `val_actions/gain_vs_hold` (whole chunk), `val_actions/gain_vs_hold_last_step` (last step only,
  where the chunk has drifted furthest from the current state)
- `val_actions/step_error/tXX_mae` and `tXX_baseline_hold_mae`: error at each chunk step
  (t00 = first action, t29 = last). Shows where in the chunk the error grows.
- `val_actions/motion_chunks/{n,mae,baseline_hold_mae,gain_vs_hold}`: samples whose ground-truth
  chunk moves more than `--moving-threshold` (default 0.3) on some live dim. Does the policy predict the motion?
- `val_actions/static_chunks/{...}`: the remaining samples. Does the policy hold position instead of drifting?
- `val_actions/gripper/{mae,baseline_hold_mae}`: `--gripper-dims` (default `14,15`).
- `val_actions/gripper/{gt_events,pred_events,event_recall,event_precision,event_f1,onset_error_ms,false_toggles}`:
  open/close timing. An event is a crossing of the gripper's mid-range (±10% hysteresis) within
  [current state, chunk]. A predicted event matches a ground-truth one with the same direction
  within 6 steps (200 ms at 30 fps). `onset_error_ms` = mean timing error of matched events;
  `false_toggles` = unmatched predicted events per chunk.

Renamed 2026-09-30 (old -> new): `holdstill_*` -> `baseline_hold_*`, `skill` -> `gain_vs_hold`,
`final_skill` -> `gain_vs_hold_last_step`, `moving_*` / `still_*` -> `motion_chunks/*` / `static_chunks/*`
(`*_samples` -> `/n`), `gripper_mae` -> `gripper/mae`, `gripper_holdstill_mae` -> `gripper/baseline_hold_mae`,
`gripper_skill` -> removed (the model never sees gripper state with `mask_gripper_state=True`, while
the hold baseline does, so that comparison was unfair); use the gripper event metrics instead.
Default `--max-samples` 128 -> 1024.

The prune loop still ranks checkpoints by `val_actions/mae`. Reference numbers (old key names) for
`pi05_g2_mixed_vision_action` on the 13-episode `lerobot_v21_mp_val` set (2026-09-28):

| step | mae | holdstill | skill | moving_skill | still_skill | gripper_skill |
|---|---|---|---|---|---|---|
| 1000 | 0.0385 | 0.0388 | 0.009 | 0.183 | -1.035 | -0.529 |
| 9000 | 0.0371 | 0.0388 | 0.045 | 0.211 | -0.951 | -0.439 |
| 10000 | 0.0370 | 0.0388 | 0.046 | 0.207 | -0.914 | -0.426 |

Offline skill is a filter, not proof: only closed-loop runs (sim or robot) measure task success.
