"""Sanity tests for the filters. Run with:  python -m pytest tests/"""

import numpy as np

from sensor_dsp.filters import kalman_filter, moving_average
from sensor_dsp.metrics import rmse
from sensor_dsp.simulation import simulate


def test_moving_average_known_input():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = moving_average(x, 2)
    assert np.allclose(y, [1.0, 1.5, 2.5, 3.5])


def test_filters_preserve_length():
    _t, _truth, _vel, meas = simulate(sigma=2.0, seed=0)
    assert len(moving_average(meas, 10)) == len(meas)
    assert len(kalman_filter(meas, dt=0.1, sigma_meas=2.0, q=5.0)) == len(meas)


def test_kalman_beats_moving_average_and_raw():
    _t, truth, _vel, meas = simulate(sigma=2.0, seed=0)
    ma = moving_average(meas, 10)
    kal = kalman_filter(meas, dt=0.1, sigma_meas=2.0, q=5.0)
    assert rmse(kal, truth) < rmse(ma, truth) < rmse(meas, truth)
