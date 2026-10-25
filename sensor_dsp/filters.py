"""Moving-average and constant-velocity Kalman filters."""

import numpy as np


def moving_average(x, window):
    """Centered-in-the-past moving average.

    Output i is the mean of the last ``window`` samples up to and including
    i. Early samples average over however many samples exist so far, so the
    output has the same length as the input and no NaN warm-up.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    partial_sums = np.convolve(x, kernel, mode="full")[: len(x)] * window
    counts = np.minimum(np.arange(1, len(x) + 1), window)
    return partial_sums / counts


class ConstantVelocityKalman:
    """Discrete constant-velocity Kalman filter (DWNA process model).

    State is [position, velocity]. Process noise follows the discrete
    white-noise acceleration model with spectral density ``q``:

        Q = q * [[dt^4/4, dt^3/2],
                 [dt^3/2, dt^2  ]]

    Only position is measured, so H = [1, 0] and R = sigma_meas**2.
    """

    def __init__(self, dt, sigma_meas, q, x0=None, p0=None):
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[sigma_meas**2]])
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = q * np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]])
        self.x = np.zeros(2) if x0 is None else np.asarray(x0, dtype=float)
        self.P = (
            np.eye(2) * 1000.0 if p0 is None else np.asarray(p0, dtype=float)
        )

    def step(self, z):
        """Predict, then update on measurement z. Returns position estimate."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]
        K = (self.P @ self.H.T / S).ravel()
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, self.H.ravel())) @ self.P
        return self.x[0]


def kalman_filter(measurements, dt, sigma_meas, q, x0=None):
    """Run the constant-velocity Kalman filter over all measurements."""
    measurements = np.asarray(measurements, dtype=float)
    if x0 is None:
        x0 = np.array([measurements[0], 0.0])
    kf = ConstantVelocityKalman(dt, sigma_meas, q, x0=x0)
    return np.array([kf.step(z) for z in measurements])
