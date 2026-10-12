"""MLflow experiment tracking shared by scripts/train.py and the validation sidecars.

Tracking URI precedence: explicit argument (--mlflow-tracking-uri) > $MLFLOW_TRACKING_URI > MLflow's
default (sqlite:///mlflow.db in the working directory, i.e. a local store you can browse with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

Starting or resuming a run raises, so a bad URI or a missing run fails before any GPU work. Logging
calls never raise: a failed write is a warning, so a tracking-server hiccup cannot take a run down.
"""

import argparse
from collections.abc import Mapping, Sequence
import json
import logging
from typing import Any

import mlflow
import numpy as np


class Tracker:
    """Handle to one MLflow run. A disabled tracker (run_id None) turns every call into a no-op."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id

    @property
    def enabled(self) -> bool:
        return self.run_id is not None

    def log_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        """Log the scalar entries of `metrics` at `step`; non-scalar values are skipped."""
        if not self.enabled:
            return
        scalars = scalar_metrics(metrics)
        if not scalars:
            return
        try:
            # Async: the HTTP round trip to a remote server never blocks the training loop.
            mlflow.log_metrics(scalars, step=int(step), synchronous=False, run_id=self.run_id)
        except Exception as e:  # never take a run down over a log write
            logging.warning(f"mlflow log_metrics failed at step {step}: {e}")

    def log_table(self, artifact_file: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Log a row-major table as an MLflow table artifact (shown in the run's Evaluation tab)."""
        if not self.enabled:
            return
        try:
            data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            mlflow.log_table(data, artifact_file=artifact_file, run_id=self.run_id)
        except Exception as e:
            logging.warning(f"mlflow log_table {artifact_file} failed: {e}")

    def log_images(self, key: str, images: Sequence[np.ndarray], step: int) -> None:
        """Log uint8 HxWx3 images as `<key>/<index>` at `step`."""
        if not self.enabled:
            return
        try:
            for i, image in enumerate(images):
                mlflow.log_image(np.asarray(image), key=f"{key}/{i}", step=int(step))
        except Exception as e:
            logging.warning(f"mlflow log_image {key} failed: {e}")

    def finish(self, status: str = "FINISHED") -> None:
        if self.enabled:
            end_active_run(status)


def start(
    *,
    enabled: bool,
    experiment: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    resume_run_id: str | None = None,
    tags: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Tracker:
    """Start (or resume) an MLflow run and make it the active run.

    `config` is logged once, on a fresh run only: flattened into params plus the full nested dict as
    config.json. MLflow params are write-once, so re-logging a changed value on resume would raise.
    """
    if not enabled:
        logging.info("MLflow tracking disabled")
        return Tracker()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    uri = mlflow.get_tracking_uri()

    if resume_run_id is not None:
        try:
            run = mlflow.start_run(run_id=resume_run_id)
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Cannot resume MLflow run {resume_run_id} at {uri}. Point --mlflow-tracking-uri or "
                "MLFLOW_TRACKING_URI at the store the run was created in."
            ) from e
        logging.info(f"Resumed MLflow run {run.info.run_id} ({uri})")
        return Tracker(run.info.run_id)

    mlflow.set_experiment(experiment)
    run = mlflow.start_run(
        run_name=run_name, tags={k: str(v) for k, v in (tags or {}).items() if v is not None}
    )
    if config is not None:
        mlflow.log_params(flatten_params(config))
        mlflow.log_text(json.dumps(config, indent=2, default=str), "config.json")
    logging.info(f"Started MLflow run {run.info.run_id} '{run_name}' in experiment '{experiment}' ({uri})")
    return Tracker(run.info.run_id)


def end_active_run(status: str = "FINISHED") -> None:
    """Flush pending async writes and close the active run with `status` (FINISHED, FAILED, KILLED)."""
    if mlflow.active_run() is None:
        return
    try:
        mlflow.flush_async_logging()
    except Exception as e:
        logging.warning(f"mlflow flush failed: {e}")
    mlflow.end_run(status=status)


def run_main(main_fn, *args) -> None:
    """Call `main_fn(*args)`; on an exception close the active run as FAILED (KILLED on Ctrl-C) and re-raise."""
    try:
        main_fn(*args)
    except BaseException as e:
        end_active_run("KILLED" if isinstance(e, KeyboardInterrupt) else "FAILED")
        raise


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep the entries that convert to a float (python/numpy/0-d array scalars); drop strings and arrays."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (str, bytes)) or np.ndim(v) != 0:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted MLflow param keys; non-primitive leaves become strings."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and v:
            out.update(flatten_params(v, f"{key}."))
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[key] = v
        else:
            out[key] = str(v)
    return out


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """MLflow flags shared by the validation scripts."""
    parser.add_argument(
        "--mlflow-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to MLflow. --no-mlflow-enabled skips the run entirely (e.g. when train.py relays --metrics-jsonl).",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to $MLFLOW_TRACKING_URI, else sqlite:///mlflow.db in the working dir.",
    )
    parser.add_argument("--mlflow-experiment", default=None, help="MLflow experiment. Defaults to TrainConfig.project_name.")
    parser.add_argument("--mlflow-run-name", default=None, help="MLflow run name for validation metrics.")
    parser.add_argument(
        "--mlflow-group", default=None, help="Value of the run's 'group' tag; set it to the training exp name to pair runs."
    )
    parser.add_argument(
        "--mlflow-tag", action="append", default=None, help="Extra run tag, KEY=VALUE or a bare label. Repeatable."
    )


def parse_tags(group: str | None, tags: Sequence[str] | None) -> dict[str, str]:
    out = {"group": group} if group else {}
    for tag in tags or ():
        key, sep, value = tag.partition("=")
        out[key] = value if sep else "true"
    return out
