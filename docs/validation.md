# Held-out validation for fine-tuning

This repo keeps validation as a sidecar process instead of changing `scripts/train.py`.
Train writes checkpoints as usual; the validator watches those checkpoint folders,
computes loss on held-out LeRobot data, and logs `val/*` metrics to MLflow.

## Tracking server

All scripts log to MLflow. The tracking URI comes from, in order: the `--mlflow-tracking-uri`
flag, the `MLFLOW_TRACKING_URI` env var, else MLflow's default local store
`sqlite:///mlflow.db` in the working directory (run from the repo root so train and
validators share it). With a sqlite store, artifacts (images, tables, config.json) go to
`./mlruns/` under the working directory, not next to the `.db` file. Browse a local store with:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

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
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000   # optional; omit for the local sqlite store
bash scripts/train.sh "$CONFIG" "$EXP"
# or: .venv/bin/python scripts/train.py "$CONFIG" --exp-name="$EXP" --mlflow-tracking-uri=http://<mlflow-host>:5000
```

Training writes checkpoints to:

```text
checkpoints/<CONFIG>/<EXP>/<step>/
```

The MLflow run id is saved in `checkpoints/<CONFIG>/<EXP>/mlflow_run_id.txt`; `--resume`
continues that run. Train and validation runs share the tag `group=<EXP>`.

## Run validation once

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config-name "$CONFIG" \
  --checkpoint-dir "checkpoints/$CONFIG/$EXP/5000" \
  --val-repo-id "$VAL_REPO" \
  --max-batches 20 \
  --batch-size 32 \
  --mlflow-run-name "${EXP}-val" \
  --mlflow-group "$EXP"
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
  --mlflow-run-name "${EXP}-val" \
  --mlflow-group "$EXP"
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
  --mlflow-run-name "${EXP}-action-val" \
  --mlflow-group "$EXP"
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
  --mlflow-run-name "${EXP}-action-val-first" \
  --mlflow-group "$EXP"
```

Action metrics logged:

- `val_actions/mse`
- `val_actions/mae`
- `val_actions/first_mse`
- `val_actions/first_mae`
- `val_actions/final_mse`
- `val_actions/final_mae`
- `val_actions/dim_XX_mae`
- `val_actions/sample_table/step_XXXXXX.json` (table artifact, Evaluation tab)

To put action metrics in the training run instead of a separate run, start training with
`VAL_METRICS_JSONL=<file>` and the validator with `--metrics-jsonl <file> --no-mlflow-enabled`;
`val_actions/*` points land at the checkpoint step they scored.
