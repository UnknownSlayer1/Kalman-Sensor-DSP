"""Synthetic distance-sensor simulation.

A target moves smoothly in front of an optical distance sensor (think
LiDAR or laser rangefinder). The true trajectory is analytic, so velocity
is known exactly and filter error can be measured against ground truth.
"""

import numpy as np


def true_trajectory(t):
    """Position (cm) and velocity (cm/s) of the target at times t (s)."""
    position = 120.0 + 40.0 * np.sin(2 * np.pi * 0.02 * t)
    velocity = 40.0 * 2 * np.pi * 0.02 * np.cos(2 * np.pi * 0.02 * t)
    return position, velocity


def simulate(dt=0.1, duration=60.0, sigma=2.0, seed=0):
    """Return (t, true_position, true_velocity, measurements).

    Measurements are the true position plus zero-mean Gaussian noise with
    standard deviation sigma (cm). The default sigma of 2.0 cm matches the
    noise level used in the earlier MATLAB simulation this project follows
    up on.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration + dt / 2, dt)
    position, velocity = true_trajectory(t)
    measurements = position + rng.normal(0.0, sigma, size=t.shape)
    return t, position, velocity, measurements
