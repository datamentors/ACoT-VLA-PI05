import math

import mlflow
import numpy as np
import pytest

import openpi.training.tracking as _tracking


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    yield uri
    _tracking.end_active_run()


def test_disabled_tracker_is_noop(store):
    tracker = _tracking.start(enabled=False, experiment="exp", tracking_uri=store)
    assert not tracker.enabled
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.log_table("t.json", ["a"], [[1]])
    tracker.finish()
    assert mlflow.active_run() is None


def test_fresh_run_then_resume_keeps_run_and_history(store):
    config = {"exp_name": "e", "model": {"action_dim": 32, "filter": object()}, "lr": None}
    tracker = _tracking.start(
        enabled=True, experiment="exp", run_name="e", tracking_uri=store, tags={"group": "e"}, config=config
    )
    run_id = tracker.run_id
    tracker.log_metrics({"loss": np.float32(2.0), "grad_norm": np.array(1.5), "skip_me": "x", "vec": np.ones(3)}, step=0)
    tracker.log_metrics({"val/x": 0.3}, step=1000)
    tracker.log_metrics({"val/x": 0.2}, step=500)  # out of order: steps are per key
    tracker.log_metrics({"val/nan": float("nan")}, step=0)
    tracker.log_table("val_actions/sample_table/step_000500.json", ["a", "b"], [[1, 2.0], [3, 4.0]])
    tracker.finish()

    resumed = _tracking.start(enabled=True, experiment="exp", tracking_uri=store, resume_run_id=run_id)
    assert resumed.run_id == run_id
    resumed.log_metrics({"loss": 1.0}, step=10)
    resumed.finish()

    client = mlflow.MlflowClient(tracking_uri=store)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["group"] == "e"
    assert run.data.params["model.action_dim"] == "32"
    assert set(run.data.metrics) == {"loss", "grad_norm", "val/x", "val/nan"}
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, "loss")] == [(0, 2.0), (10, 1.0)]
    assert sorted((m.step, m.value) for m in client.get_metric_history(run_id, "val/x")) == [(500, 0.2), (1000, 0.3)]
    assert math.isnan(client.get_metric_history(run_id, "val/nan")[0].value)
    artifacts = [a.path for a in client.list_artifacts(run_id, "val_actions/sample_table")]
    assert artifacts == ["val_actions/sample_table/step_000500.json"]
    assert "config.json" in [a.path for a in client.list_artifacts(run_id)]


def test_resume_unknown_run_fails_fast(store):
    with pytest.raises(RuntimeError, match="Cannot resume MLflow run"):
        _tracking.start(enabled=True, experiment="exp", tracking_uri=store, resume_run_id="0" * 32)


def test_run_main_marks_failed(store):
    tracker = _tracking.start(enabled=True, experiment="exp", tracking_uri=store)

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        _tracking.run_main(boom)
    assert mlflow.MlflowClient(tracking_uri=store).get_run(tracker.run_id).info.status == "FAILED"


def test_parse_tags():
    assert _tracking.parse_tags("g", ["k=v", "label"]) == {"group": "g", "k": "v", "label": "true"}
    assert _tracking.parse_tags(None, None) == {}
