import numpy as np
import pytest

import openpi.training.optimizer as _optimizer


@pytest.mark.parametrize("decay_type", ["sqrt", "linear", "cosine"])
def test_wsd_schedule_phases(decay_type):
    schedule = _optimizer.WSDSchedule(
        warmup_steps=1_000, peak_lr=1e-5, total_steps=100_000, decay_steps=10_000, end_lr=1e-6, decay_type=decay_type
    ).create()

    assert float(schedule(0)) < 1e-7
    np.testing.assert_allclose(float(schedule(1_000)), 1e-5, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(50_000)), 1e-5, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(90_000)), 1e-5, rtol=1e-6)
    assert 1e-6 < float(schedule(95_000)) < 1e-5
    np.testing.assert_allclose(float(schedule(100_000)), 1e-6, rtol=1e-5)
    np.testing.assert_allclose(float(schedule(120_000)), 1e-6, rtol=1e-5)


def test_wsd_schedule_decay_is_monotonic():
    schedule = _optimizer.WSDSchedule().create()
    lrs = np.array([float(schedule(s)) for s in range(90_000, 100_001, 500)])
    assert np.all(np.diff(lrs) <= 0)


def test_wsd_schedule_rejects_decay_before_warmup():
    with pytest.raises(ValueError, match="decay_start"):
        _optimizer.WSDSchedule(warmup_steps=5_000, total_steps=10_000, decay_steps=8_000).create()
