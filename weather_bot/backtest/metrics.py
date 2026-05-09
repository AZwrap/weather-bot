"""Forecast verification metrics.

Point-forecast metrics (bias, MAE, RMSE) measure how close a single number is
to truth. Probabilistic metrics (Brier, reliability, CRPS) measure how well
*distributions* match outcomes — which is what actually matters for trading,
since Polymarket prices are probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------- Deterministic metrics ----------


def bias(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """Mean signed error: positive means model is too warm."""
    return float(np.mean(forecasts - observations))


def mae(forecasts: np.ndarray, observations: np.ndarray) -> float:
    return float(np.mean(np.abs(forecasts - observations)))


def rmse(forecasts: np.ndarray, observations: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecasts - observations) ** 2)))


# ---------- Probabilistic metrics ----------


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and {0,1} outcomes.

    Lower is better. A perfect deterministic forecast scores 0; predicting a
    constant climatological base rate gives the climatology baseline.
    """
    return float(np.mean((probs - outcomes.astype(float)) ** 2))


def crps_gaussian(
    forecast_mean: np.ndarray,
    forecast_sigma: float | np.ndarray,
    observation: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for a Gaussian predictive distribution.

    CRPS generalises MAE to probability distributions. Lower is better.
    Closed form: σ * (z(2Φ(z)-1) + 2φ(z) - 1/√π) where z = (obs-μ)/σ.
    """
    sigma = np.asarray(forecast_sigma, dtype=float)
    z = (observation - forecast_mean) / sigma
    crps = sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


@dataclass
class ReliabilityBin:
    bin_low: float
    bin_high: float
    n: int
    mean_predicted: float
    observed_freq: float


def reliability_table(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Group (probability, outcome) pairs into bins and report observed frequency.

    A perfectly calibrated forecaster has observed_freq ≈ mean_predicted in
    every bin. Systematic deviations reveal under- or over-confidence.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                bin_low=float(lo),
                bin_high=float(hi),
                n=int(mask.sum()),
                mean_predicted=float(probs[mask].mean()),
                observed_freq=float(outcomes[mask].mean()),
            )
        )
    return out
