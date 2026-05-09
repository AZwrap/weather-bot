"""Forecast-skill backtest harness.

Pipeline for one location:
  1. Fetch ECMWF (or other model) historical daily-max forecast for the past N days.
  2. Fetch ERA5 observed daily max over the same window.
  3. Compute deterministic metrics (bias, MAE, RMSE) and compare against two
     trivial baselines (persistence, climatology mean).
  4. Build a probabilistic forecast: predicted ~ N(forecast - bias, σ_residual)
     where σ_residual is the std of bias-corrected forecast errors.
  5. Generate (probability, outcome) pairs over a grid of thresholds and
     compute Brier score, CRPS, and a reliability table.

Caveat: the historical-forecast-api returns "best-available" concatenated
short-range outputs, NOT a true T-1d-issued forecast. Skill measured here is
an UPPER BOUND on what we'll see in live 1-day-ahead trading. Reinforce this
by also running a forward log going forward (TODO).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
from scipy import stats

from ..forecast.fetcher import (
    DailyAgg,
    Location,
    fetch_historical_forecast_range,
    fetch_observed_range,
)
from ..observations import fetch_observed_truth
from .metrics import (
    ReliabilityBin,
    bias,
    brier_score,
    crps_gaussian,
    mae,
    reliability_table,
    rmse,
)


@dataclass
class SkillReport:
    location: str
    target: DailyAgg  # "max" or "min"
    model: str
    start: date
    end: date
    n_days: int

    # Deterministic skill of the model
    forecast_bias_c: float
    forecast_mae_c: float
    forecast_rmse_c: float

    # Baselines (lower MAE/RMSE = harder to beat)
    persistence_mae_c: float
    persistence_rmse_c: float
    climatology_mae_c: float
    climatology_rmse_c: float

    # Probabilistic skill (Gaussian predictive built from forecast residuals)
    residual_sigma_c: float
    brier: float
    crps_c: float
    reliability: list[ReliabilityBin] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [
            f"━━ {self.location} [{self.target}] ({self.model}) ━━",
            f"Window: {self.start} → {self.end}  (n={self.n_days} days)",
            "",
            f"  Forecast vs observed (deterministic):",
            f"    bias = {self.forecast_bias_c:+.2f} °C   "
            f"MAE = {self.forecast_mae_c:.2f} °C   "
            f"RMSE = {self.forecast_rmse_c:.2f} °C",
            "",
            f"  Baselines (these are what we have to beat):",
            f"    persistence    MAE = {self.persistence_mae_c:.2f}   RMSE = {self.persistence_rmse_c:.2f}",
            f"    climatology    MAE = {self.climatology_mae_c:.2f}   RMSE = {self.climatology_rmse_c:.2f}",
            "",
            f"  Probabilistic (Gaussian predictive, σ = {self.residual_sigma_c:.2f} °C):",
            f"    Brier = {self.brier:.4f}   CRPS = {self.crps_c:.3f} °C",
            "",
            f"  Reliability (predicted prob → observed freq):",
            f"    {'bin':>11s}  {'n':>4s}  {'predicted':>10s}  {'observed':>9s}",
        ]
        for b in self.reliability:
            lines.append(
                f"    [{b.bin_low:.2f}, {b.bin_high:.2f}]  "
                f"{b.n:4d}  {b.mean_predicted:10.3f}  {b.observed_freq:9.3f}"
            )
        return "\n".join(lines)


async def _fetch_pair(
    location: Location,
    start: date,
    end: date,
    agg: DailyAgg,
    model: str,
    icao: str | None = None,
) -> pd.DataFrame:
    """Fetch forecast and observed series and return them merged on date.

    Observations come from METAR (Iowa State ASOS) when an `icao` is given —
    same source Polymarket resolves on. Falls back to ERA5 for non-ASOS
    stations (HKO).
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        forecast_df = await fetch_historical_forecast_range(
            location, start, end, agg, model, client
        )
        if icao is not None:
            obs_df = await fetch_observed_truth(
                location, icao, start, end, agg, source="metar", client=client,
            )
        else:
            obs_df = await fetch_observed_range(location, start, end, agg, client)
    df = pd.merge(forecast_df, obs_df, on="date", how="inner").dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _compute(
    df: pd.DataFrame,
    location: Location,
    target: DailyAgg,
    model: str,
    start: date,
    end: date,
    bias_c: float = 0.0,
) -> SkillReport:
    """Compute skill metrics for the bias-corrected forecast.

    The predictive distribution at day t is N(forecast_t - bias_c, σ) where σ
    is the std of post-correction residuals. Probabilistic metrics are
    evaluated at FIXED integer thresholds spanning the observed range, so
    Brier score actually responds to errors in the predictive mean.
    """
    f_raw = df["forecast_c"].to_numpy()
    o = df["observed_c"].to_numpy()
    f = f_raw - bias_c  # apply user-supplied bias correction once

    # Deterministic metrics on the corrected forecast
    f_bias = bias(f, o)  # remaining bias (≈0 if `bias_c` is a good estimate)
    f_mae = mae(f, o)
    f_rmse = rmse(f, o)

    # Persistence baseline: predict tomorrow = today
    pers_pred = o[:-1]
    pers_obs = o[1:]
    pers_mae = mae(pers_pred, pers_obs)
    pers_rmse = rmse(pers_pred, pers_obs)

    # Climatology baseline: in-window mean
    clim = float(np.mean(o))
    clim_mae = mae(np.full_like(o, clim), o)
    clim_rmse = rmse(np.full_like(o, clim), o)

    # Predictive distribution: N(f, σ) where σ is std of post-correction residuals
    residuals = o - f
    sigma = float(np.std(residuals, ddof=1))
    if sigma == 0.0:
        sigma = 1e-6  # guard for degenerate cases
    mu = f

    # Fixed thresholds spanning the observed range, half-degree resolution.
    # Using a fixed grid (rather than mu-relative) means a wrong predictive
    # mean produces wrong probabilities, which Brier penalises.
    obs_lo = float(np.floor(o.min())) - 1.0
    obs_hi = float(np.ceil(o.max())) + 1.0
    thresholds = np.arange(obs_lo, obs_hi + 0.001, 0.5)

    mu_b = mu[:, None]                          # (n_days, 1)
    t_b = thresholds[None, :]                   # (1, n_thresh)
    probs = 1.0 - stats.norm.cdf(t_b, loc=mu_b, scale=sigma)
    outcomes = (o[:, None] > t_b).astype(float)

    probs_flat = probs.ravel()
    outcomes_flat = outcomes.ravel()

    return SkillReport(
        location=location.name,
        target=target,
        model=model,
        start=start,
        end=end,
        n_days=len(df),
        forecast_bias_c=f_bias,
        forecast_mae_c=f_mae,
        forecast_rmse_c=f_rmse,
        persistence_mae_c=pers_mae,
        persistence_rmse_c=pers_rmse,
        climatology_mae_c=clim_mae,
        climatology_rmse_c=clim_rmse,
        residual_sigma_c=sigma,
        brier=brier_score(probs_flat, outcomes_flat),
        crps_c=crps_gaussian(mu, sigma, o),
        reliability=reliability_table(probs_flat, outcomes_flat, n_bins=10),
    )


async def run_skill_backtest(
    location: Location,
    days: int = 90,
    end: date | None = None,
    model: str = "ecmwf_ifs025",
    target: DailyAgg = "max",
    bias_c: float = 0.0,
    icao: str | None = None,
) -> SkillReport:
    """Run the full skill backtest for one location's daily max or min.

    `bias_c` is subtracted from the raw forecast before metrics are computed.
    For honest evaluation, pass a bias trained on data that does NOT overlap
    with the test window [end - days + 1, end].

    `icao` enables METAR truth source (recommended); leave None to fall back
    to ERA5.
    """
    if end is None:
        # ERA5 archive lags by ~5 days; back off a bit.
        end = date.today() - timedelta(days=6)
    start = end - timedelta(days=days - 1)

    df = await _fetch_pair(location, start, end, target, model, icao=icao)
    if len(df) < 10:
        raise RuntimeError(
            f"Only {len(df)} matching days for {location.name} [{target}] "
            f"({start} to {end}); need ≥10 for meaningful stats."
        )
    return _compute(df, location, target, model, start, end, bias_c=bias_c)
