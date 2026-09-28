#!/usr/bin/env python3
"""Validate predicted action chunks against held-out demonstration actions.

Modes:
- all-frames: score many frames from the validation dataset.
- first-frame: score one starting frame per held-out episode when frame_index is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import wandb

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config name used for the checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", help="One checkpoint step dir, e.g. checkpoints/<config>/<exp>/5000.")
    group.add_argument("--checkpoint-root", help="Checkpoint root containing numeric step dirs to validate/watch.")
    parser.add_argument(
        "--val-repo-id",
        action="append",
        default=None,
        help="Held-out LeRobot repo/path. Repeat for a MultiLeRobot validation set. Defaults to config repo_id.",
    )
    parser.add_argument(
        "--mode",
        choices=("all-frames", "first-frame"),
        default="all-frames",
        help="all-frames scores dense held-out frames; first-frame scores episode starts only.",
    )
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum validation samples per checkpoint.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Stride used by all-frames mode and fallback first-frame mode. "
        "Default: len(dataset) // max_samples, so samples span the whole held-out set.",
    )
    parser.add_argument("--action-dim", type=int, default=None, help="Optional action dimension cap before scoring.")
    parser.add_argument("--max-dim-metrics", type=int, default=32, help="Maximum per-dimension MAE metrics to log.")
    parser.add_argument("--watch", action="store_true", help="Keep polling checkpoint-root and validate new steps.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Seconds between checkpoint-root polls.")
    parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Append one JSON line per scored checkpoint here. scripts/train.py relays these into the "
        "training wandb run when started with VAL_METRICS_JSONL pointing at the same file "
        "(pair with --wandb-mode disabled to avoid a second run).",
    )
    parser.add_argument("--wandb-project", default=None, help="W&B project. Defaults to TrainConfig.project_name.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name for validation metrics.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group, useful to pair train/action-val runs.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Optional W&B tag. Repeatable.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override W&B mode. If omitted, wandb/env defaults apply.",
    )
    return parser.parse_args()


def _with_validation_repo(data_factory: Any, repo_ids: Sequence[str] | None) -> Any:
    if not repo_ids:
        return data_factory
    repo_id: str | list[str] = repo_ids[0] if len(repo_ids) == 1 else list(repo_ids)
    if not dataclasses.is_dataclass(data_factory) or not hasattr(data_factory, "repo_id"):
        raise TypeError(f"Config data factory {type(data_factory).__name__} does not expose repo_id")
    return dataclasses.replace(data_factory, repo_id=repo_id)


def _checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    try:
        return int(checkpoint_dir.name)
    except ValueError:
        return 0


def _checkpoint_ready(checkpoint_dir: pathlib.Path) -> bool:
    return (checkpoint_dir / "params").exists() and (checkpoint_dir / "assets").exists()


def _find_checkpoint_dirs(checkpoint_root: pathlib.Path) -> list[pathlib.Path]:
    if not checkpoint_root.exists():
        return []
    checkpoints = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit() and _checkpoint_ready(p)]
    return sorted(checkpoints, key=_checkpoint_step)


def _scalar_from_item(item: dict[str, Any], key: str) -> int | None:
    if key not in item:
        return None
    value = item[key]
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _select_indices(dataset: _data_loader.Dataset, *, mode: str, max_samples: int, sample_stride: int | None) -> list[int]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if sample_stride is None:
        # A stride of 1 would score only the first max_samples frames of the first episode.
        sample_stride = max(1, len(dataset) // max_samples)
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")

    if mode == "all-frames":
        return list(range(0, min(len(dataset), max_samples * sample_stride), sample_stride))[:max_samples]

    indices: list[int] = []
    seen_episodes: set[int] = set()
    for index in range(len(dataset)):
        item = dataset[index]
        episode_index = _scalar_from_item(item, "episode_index")
        frame_index = _scalar_from_item(item, "frame_index")
        if episode_index is None or frame_index is None:
            break
        if frame_index == 0 and episode_index not in seen_episodes:
            indices.append(index)
            seen_episodes.add(episode_index)
            if len(indices) >= max_samples:
                return indices

    if indices:
        return indices

    logging.warning("Could not identify episode starts; falling back to sample_stride=%d", sample_stride)
    return list(range(0, min(len(dataset), max_samples * sample_stride), sample_stride))[:max_samples]


def _action_layout(data_config: _config.DataConfig) -> tuple[list[int] | None, np.ndarray | None]:
    """Return (raw-action indices in model order, live model dims) from the policy input transform.

    The policy outputs actions in model order (e.g. Go1: arms, grippers, waist), while the
    raw dataset `action` is in recording order. Without this remap, targets and predictions
    are compared dimension-by-dimension across different joints. Masked dims (action_mask
    True) are not trained, so they are excluded from the aggregate metrics.
    """
    for transform in data_config.data_transforms.inputs:
        indices = getattr(transform, "action_indices", None)
        if indices is not None:
            mask = getattr(transform, "action_mask", None)
            live = None if mask is None else ~np.asarray(mask, dtype=bool)
            return list(indices), live
    logging.warning("No action_indices found in data transforms; comparing raw action order as-is")
    return None, None


def _prepare_raw_dataset(
    train_config: _config.TrainConfig, val_repo_ids: Sequence[str] | None
) -> tuple[_data_loader.Dataset, _transforms.DataTransformFn, list[int] | None, np.ndarray | None]:
    data_factory = _with_validation_repo(train_config.data, val_repo_ids)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model)
    # With several val repos, create_torch_dataset returns a MultiLeRobotDataset whose
    # children are wrapped for prompt_from_task, which breaks its own __len__/__getitem__
    # (children lack num_frames). TransformedDataset walks the children itself, exactly as
    # the training loader does; with no transforms it is an identity wrapper.
    dataset = _data_loader.TransformedDataset(dataset, [])
    repack = _transforms.compose(data_config.repack_transforms.inputs)
    action_indices, live_dims = _action_layout(data_config)
    return dataset, repack, action_indices, live_dims


def _align_actions(predicted: np.ndarray, target: np.ndarray, action_dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    if target.ndim == 1:
        target = target[None, :]
    horizon = min(predicted.shape[0], target.shape[0])
    dims = min(predicted.shape[-1], target.shape[-1])
    if action_dim is not None:
        dims = min(dims, action_dim)
    if horizon <= 0 or dims <= 0:
        raise ValueError(f"Cannot align predicted {predicted.shape} and target {target.shape}")
    return predicted[:horizon, :dims], target[:horizon, :dims]


def _score_sample(
    policy: Any,
    sample: dict[str, Any],
    *,
    action_dim: int | None,
    action_indices: list[int] | None = None,
    live_dims: np.ndarray | None = None,
) -> dict[str, Any]:
    target_actions = sample.get("actions")
    if target_actions is None:
        raise ValueError("Validation sample does not contain actions")
    # Policy input transforms may mutate the sample in place; keep GT immutable.
    target_actions = np.asarray(target_actions).copy()
    if action_indices is not None:
        if target_actions.shape[-1] <= max(action_indices):
            raise ValueError(f"Target action dim {target_actions.shape[-1]} too small for indices {action_indices}")
        target_actions = target_actions[..., action_indices]

    start = time.time()
    result = policy.infer(sample)
    infer_ms = (time.time() - start) * 1000.0
    predicted_actions = result.get("actions")
    if predicted_actions is None:
        raise ValueError("Policy output does not contain actions")

    pred, target = _align_actions(predicted_actions, target_actions, action_dim)
    err = pred - target
    live = np.ones(pred.shape[1], dtype=bool)
    if live_dims is not None:
        live[: min(len(live_dims), len(live))] = live_dims[: len(live)]
    # Aggregates over live (trained) dims only; per-dim MAE still reports every dim.
    squared = np.square(err[:, live])
    absolute = np.abs(err[:, live])
    return {
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(absolute)),
        "first_mse": float(np.mean(squared[0])),
        "first_mae": float(np.mean(absolute[0])),
        "final_mse": float(np.mean(squared[-1])),
        "final_mae": float(np.mean(absolute[-1])),
        "per_dim_mae": np.mean(np.abs(err), axis=0),
        "horizon": float(pred.shape[0]),
        "dims": float(live.sum()),
        "infer_ms": float(infer_ms),
    }


def validate_actions(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    val_repo_ids: Sequence[str] | None,
    *,
    mode: str,
    max_samples: int,
    sample_stride: int,
    action_dim: int | None,
    max_dim_metrics: int,
) -> dict[str, Any]:
    logging.info("Loading policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    dataset, repack, action_indices, live_dims = _prepare_raw_dataset(train_config, val_repo_ids)
    indices = _select_indices(dataset, mode=mode, max_samples=max_samples, sample_stride=sample_stride)
    if not indices:
        raise RuntimeError("No validation samples selected")

    rows: list[list[Any]] = []
    sample_metrics: list[dict[str, Any]] = []
    for raw_index in indices:
        raw_item = dataset[raw_index]
        sample = repack(raw_item)
        metrics = _score_sample(
            policy, sample, action_dim=action_dim, action_indices=action_indices, live_dims=live_dims
        )
        episode_index = _scalar_from_item(raw_item, "episode_index")
        frame_index = _scalar_from_item(raw_item, "frame_index")
        sample_metrics.append(metrics)
        rows.append([
            raw_index,
            episode_index,
            frame_index,
            metrics["mse"],
            metrics["mae"],
            metrics["first_mse"],
            metrics["first_mae"],
            metrics["horizon"],
            metrics["dims"],
            metrics["infer_ms"],
        ])

    per_dim_stack = np.stack([m["per_dim_mae"] for m in sample_metrics], axis=0)
    metrics_out: dict[str, Any] = {
        "val_actions/mse": float(np.mean([m["mse"] for m in sample_metrics])),
        "val_actions/mae": float(np.mean([m["mae"] for m in sample_metrics])),
        "val_actions/first_mse": float(np.mean([m["first_mse"] for m in sample_metrics])),
        "val_actions/first_mae": float(np.mean([m["first_mae"] for m in sample_metrics])),
        "val_actions/final_mse": float(np.mean([m["final_mse"] for m in sample_metrics])),
        "val_actions/final_mae": float(np.mean([m["final_mae"] for m in sample_metrics])),
        "val_actions/samples": float(len(sample_metrics)),
        "val_actions/horizon_mean": float(np.mean([m["horizon"] for m in sample_metrics])),
        "val_actions/dims_mean": float(np.mean([m["dims"] for m in sample_metrics])),
        "val_actions/infer_ms_mean": float(np.mean([m["infer_ms"] for m in sample_metrics])),
        "val_actions/checkpoint_step": float(_checkpoint_step(checkpoint_dir)),
    }
    for dim_index, value in enumerate(np.mean(per_dim_stack, axis=0)[:max_dim_metrics]):
        metrics_out[f"val_actions/dim_{dim_index:02d}_mae"] = float(value)

    metrics_out["val_actions/sample_table"] = wandb.Table(
        columns=[
            "raw_index",
            "episode_index",
            "frame_index",
            "mse",
            "mae",
            "first_mse",
            "first_mae",
            "horizon",
            "dims",
            "infer_ms",
        ],
        data=rows,
    )
    logging.info(
        "Action validation %s mode=%s: mse=%.6f mae=%.6f samples=%d",
        checkpoint_dir,
        mode,
        metrics_out["val_actions/mse"],
        metrics_out["val_actions/mae"],
        len(sample_metrics),
    )
    return metrics_out


def _append_metrics_jsonl(path: str, metrics: dict[str, Any]) -> None:
    """One complete line per checkpoint, written in a single call so readers never see half a record."""
    scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}
    record: dict[str, Any] = {"metrics": scalars}
    table = metrics.get("val_actions/sample_table")
    if table is not None:
        record["sample_table"] = {"columns": list(table.columns), "data": table.data}
    line = json.dumps(record, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()


def _init_wandb(args: argparse.Namespace, train_config: _config.TrainConfig) -> None:
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project or train_config.project_name,
        "name": args.wandb_run_name or f"{args.config_name}-action-validation",
        "group": args.wandb_group,
        "tags": args.wandb_tag,
        "config": vars(args),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    wandb.init(**init_kwargs)


def main() -> None:
    # force=True: openpi imports configure the root logger first, which made this a no-op
    # and silently dropped every INFO line (including the per-checkpoint summary).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args = _parse_args()
    train_config = _config.get_config(args.config_name)
    _init_wandb(args, train_config)

    validated_steps: set[int] = set()
    if args.metrics_jsonl and pathlib.Path(args.metrics_jsonl).exists():
        # Resumed run appending to the same file: do not re-score checkpoints already in it.
        for line in open(args.metrics_jsonl):
            try:
                validated_steps.add(int(json.loads(line)["metrics"]["val_actions/checkpoint_step"]))
            except (ValueError, KeyError):
                continue
        logging.info("Skipping %d checkpoint steps already in %s", len(validated_steps), args.metrics_jsonl)
    while True:
        if args.checkpoint_dir:
            checkpoint_dirs = [_path(args.checkpoint_dir)]
        else:
            checkpoint_dirs = _find_checkpoint_dirs(_path(args.checkpoint_root))

        new_dirs = [p for p in checkpoint_dirs if _checkpoint_step(p) not in validated_steps]
        for checkpoint_dir in new_dirs:
            step = _checkpoint_step(checkpoint_dir)
            metrics = validate_actions(
                train_config,
                checkpoint_dir,
                args.val_repo_id,
                mode=args.mode,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
                action_dim=args.action_dim,
                max_dim_metrics=args.max_dim_metrics,
            )
            metrics["val_actions/mode_id"] = 0.0 if args.mode == "all-frames" else 1.0
            if args.metrics_jsonl:
                _append_metrics_jsonl(args.metrics_jsonl, metrics)
            wandb.log(metrics, step=step if step > 0 else None)
            validated_steps.add(step)

        if not args.watch:
            break
        if not new_dirs:
            logging.info("No new checkpoints under %s; sleeping %.1fs", args.checkpoint_root, args.poll_seconds)
        time.sleep(args.poll_seconds)

    wandb.finish()


if __name__ == "__main__":
    main()
