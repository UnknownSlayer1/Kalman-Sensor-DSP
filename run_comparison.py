"""Main experiment: raw vs moving average vs Kalman on simulated sensor data.

Scenario: an optical distance sensor tracks a target moving smoothly
(position = 120 + 40*sin(2*pi*0.02*t) cm, sampled at 10 Hz for 60 s).
Measurements carry Gaussian noise; the default sigma of 2.0 cm matches the
noise level in the earlier MATLAB simulation this project follows up on.

Prints an RMSE table for the main run and for a noise sweep, and writes
figures/trajectory.svg and figures/rmse_vs_noise.svg.

Usage:  python run_comparison.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sensor_dsp.filters import kalman_filter, moving_average
from sensor_dsp.metrics import rmse
from sensor_dsp.simulation import simulate

DT = 0.1
SIGMA = 2.0
MA_WINDOW = 10  # samples = 1 second of data
Q = 5.0  # Kalman process-noise spectral density. Tuned on this scenario;
# the RMSE curve is flat for q in 5-8, so this is not a knife-edge value.
SEED = 0
NOISE_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0]


def run_once(sigma, q=Q, seed=SEED):
    """Return (t, truth, raw, ma, kalman, rmse_dict) for one noise level."""
    t, truth, _velocity, meas = simulate(dt=DT, sigma=sigma, seed=seed)
    ma = moving_average(meas, MA_WINDOW)
    kal = kalman_filter(meas, dt=DT, sigma_meas=sigma, q=q)
    errors = {
        "raw": rmse(meas, truth),
        "moving_average": rmse(ma, truth),
        "kalman": rmse(kal, truth),
    }
    return t, truth, meas, ma, kal, errors


def plot_trajectory(t, truth, meas, ma, kal, path):
    plt.figure(figsize=(10, 5))
    plt.plot(t, meas, linewidth=0.7, alpha=0.4, label="Raw measurements")
    plt.plot(t, truth, "k-", linewidth=2, label="True position")
    plt.plot(t, ma, label=f"Moving average (window {MA_WINDOW})")
    plt.plot(t, kal, label="Kalman filter")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (cm)")
    plt.title("Distance-sensor tracking: raw vs moving average vs Kalman")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sweep(sigmas, table, path):
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, table["raw"], "o-", label="Raw")
    plt.plot(sigmas, table["moving_average"], "s-", label="Moving average")
    plt.plot(sigmas, table["kalman"], "^-", label="Kalman")
    plt.xscale("log")
    plt.xlabel("Measurement noise sigma (cm, log scale)")
    plt.ylabel("RMSE (cm)")
    plt.title("RMSE vs measurement noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)
    t, truth, meas, ma, kal, errors = run_once(SIGMA)
    print(f"Main run: sigma = {SIGMA} cm, seed = {SEED}")
    print(f"  raw            RMSE = {errors['raw']:.3f} cm")
    print(f"  moving average RMSE = {errors['moving_average']:.3f} cm")
    print(f"  kalman         RMSE = {errors['kalman']:.3f} cm")
    plot_trajectory(t, truth, meas, ma, kal, "figures/trajectory.svg")
    plot_trajectory(t, truth, meas, ma, kal, "figures/trajectory.png")

    table = {"raw": [], "moving_average": [], "kalman": []}
    print("\nNoise sweep:")
    print("  sigma | raw | moving avg | kalman")
    for sigma in NOISE_SWEEP:
        _t, _truth, _m, _ma, _k, err = run_once(sigma)
        for key in table:
            table[key].append(err[key])
        print(
            f"  {sigma:5.1f} | {err['raw']:.3f} | "
            f"{err['moving_average']:.3f} | {err['kalman']:.3f}"
        )
    plot_sweep(NOISE_SWEEP, table, "figures/rmse_vs_noise.svg")


if __name__ == "__main__":
    main()
