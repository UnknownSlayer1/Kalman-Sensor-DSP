"""Error metrics."""

import numpy as np


def rmse(estimate, truth):
    """Root-mean-square error between an estimate and ground truth."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))
