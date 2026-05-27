"""Per-(station, target) forecast-bias correction.

The model has a systematic mean offset at every station that varies by daily
target (max vs min). HKO has +0.39 °C on max but +1.42 °C on min; LA has
+2.06 °F on max. A global correction averages these to zero and helps no one.

This module:
  1. Trains a `BiasTable` by fetching historical forecast vs observation pairs
     per (station, target), computing the mean residual.
  2. Persists the table as JSON.
  3. Provides `corrected_members(...)` to apply the offset to ensemble members
     before probability computations at inference time.

Honest evaluation: when comparing pre/post correction in a backtest, train the
bias on a window that does NOT overlap with the test window. The CLI in
`train_bias.py` enforces this with a `--train-end` flag.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np
import pandas as pd

from .forecast.fetcher import (
    DailyAgg,
    fetch_historical_forecast_range,
    fetch_observed_range,
)
from .locations import Station
from .observations import fetch_observed_truth


@dataclass
class BiasEntry:
    station_id: str
    target: DailyAgg
    bias_c: float            # mean(forecast - observation), °C
    n_days: int              # number of (forecast, obs) pairs used
    rmse_c: float            # residual RMSE for sanity check
    trained_through: date    # last day in training window

    @property
    def sigma_residual_c(self) -> float:
        """Std of forecast residuals after bias correction (°C).

        Derived from rmse and bias: rmse² = bias² + σ². This is the
        "irreducible" model-error uncertainty at this station, *additional*
        to whatever flow-dependent spread the live ensemble captures. Use
        it to inflate the live ensemble's σ:

            σ_total² = σ_ensemble² + σ_residual²

        Live ECMWF ENS is known to be under-dispersed at short leads, so
        this addition is necessary for honest probability calibration.
        """
        return float(np.sqrt(max(self.rmse_c ** 2 - self.bias_c ** 2, 1e-6)))

    def as_jsonable(self) -> dict:
        d = asdict(self)
        d["trained_through"] = self.trained_through.isoformat()
        return d

    @classmethod
    def from_jsonable(cls, d: dict) -> "BiasEntry":
        return cls(
            station_id=d["station_id"],
            target=d["target"],
            bias_c=float(d["bias_c"]),
            n_days=int(d["n_days"]),
            rmse_c=float(d["rmse_c"]),
            trained_through=date.fromisoformat(d["trained_through"]),
        )


@dataclass
class BiasTable:
    entries: dict[tuple[str, str], BiasEntry]

    def get(self, station_id: str, target: DailyAgg) -> float:
        e = self.entries.get((station_id, target))
        return e.bias_c if e else 0.0

    def get_entry(self, station_id: str, target: DailyAgg) -> BiasEntry | None:
        return self.entries.get((station_id, target))

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, path: Path) -> None:
        """Atomic save via weather_bot.atomic_write — bias_table.json is
        read at the start of every scan; a non-atomic write left a
        concurrency window where the weekly retrain cron could corrupt
        the file mid-write, causing scan_markets / place_orders to load
        partial JSON and skew probability calibration."""
        from weather_bot.atomic_write import atomic_write_json
        atomic_write_json(path, [e.as_jsonable() for e in self.entries.values()])

    @classmethod
    def load(cls, path: Path) -> "BiasTable":
        data = json.loads(path.read_text())
        entries: dict[tuple[str, str], BiasEntry] = {}
        for d in data:
            e = BiasEntry.from_jsonable(d)
            entries[(e.station_id, e.target)] = e
        return cls(entries=entries)

    @classmethod
    def empty(cls) -> "BiasTable":
        return cls(entries={})


def corrected_members(
    members: np.ndarray,
    table: BiasTable,
    station_id: str,
    target: DailyAgg,
) -> np.ndarray:
    """Subtract the bias from every ensemble member. No-op if not in table."""
    return members - table.get(station_id, target)


def predictive_members(
    members: np.ndarray,
    table: BiasTable,
    station_id: str,
    target: DailyAgg,
    *,
    inflate_sigma: bool = True,
    inflation_factor: float = 1.4,
    n_resample: int = 10,
    rng_seed: int = 0,
) -> np.ndarray:
    """Build a calibrated predictive sample from raw ensemble members.

    Steps:
      1. Subtract per-(station, target) bias.
      2. Optionally convolve with N(0, inflation_factor × σ_residual) noise
         to widen the distribution to its calibrated width.

    `inflation_factor` is a safety margin on top of σ_residual. The
    training-period σ_residual comes from concatenated short-range
    historical-forecast data, not true 1-day-lead, so it under-estimates
    real predictive uncertainty. Default 1.4× is a paranoid first guess;
    forward-log resolutions will let us calibrate empirically.

    Returns an array of `n_resample × len(members)` samples (when inflated)
    or just bias-corrected members (when not).
    """
    corrected = members - table.get(station_id, target)
    if not inflate_sigma:
        return corrected

    entry = table.get_entry(station_id, target)
    if entry is None or entry.sigma_residual_c <= 0:
        return corrected

    sigma = max(entry.sigma_residual_c * float(inflation_factor), 1e-6)
    rng = np.random.default_rng(rng_seed)
    tiled = np.tile(corrected, max(1, n_resample))
    noise = rng.normal(0.0, sigma, size=tiled.shape)
    return tiled + noise


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────


async def _train_one(
    station: Station,
    target: DailyAgg,
    train_start: date,
    train_end: date,
    model: str,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    truth_source: str = "metar",
) -> BiasEntry | None:
    """Fetch forecast/obs pair and compute mean residual.

    Truth source defaults to METAR (matches Polymarket resolution).
    Falls back to ERA5 for stations not on the ASOS network (HKO).
    """
    async with sem:
        try:
            forecast_df = await fetch_historical_forecast_range(
                station.to_location(), train_start, train_end, target, model, client
            )
            obs_df = await fetch_observed_truth(
                station.to_location(), station.station_id,
                train_start, train_end, target,
                source=truth_source, client=client,
            )
        except Exception as exc:
            print(f"!! train {station.station_id} [{target}]: {exc}")
            return None

    df = pd.merge(forecast_df, obs_df, on="date", how="inner").dropna()
    if len(df) < 30:
        print(
            f"!! train {station.station_id} [{target}]: only {len(df)} days, "
            f"need ≥30 — skipping"
        )
        return None

    f = df["forecast_c"].to_numpy()
    o = df["observed_c"].to_numpy()
    residual = f - o
    return BiasEntry(
        station_id=station.station_id,
        target=target,
        bias_c=float(np.mean(residual)),
        n_days=len(df),
        rmse_c=float(np.sqrt(np.mean(residual ** 2))),
        trained_through=train_end,
    )


async def train_bias_table(
    market_pairs: Iterable[tuple[Station, DailyAgg]],
    train_end: date,
    train_days: int = 365,
    model: str = "ecmwf_ifs025",
    concurrency: int = 2,
    truth_source: str = "metar",
) -> BiasTable:
    """Train a `BiasTable` for the supplied (station, target) pairs.

    Args:
        market_pairs: each (Station, "max"|"min") pair to train.
        train_end: last day in the training window. Make sure this is BEFORE
            any window you'll evaluate on, to keep the test honest.
        train_days: training window length in days (default 365).
        model: Open-Meteo model id.
        concurrency: max parallel HTTP requests (Open-Meteo throttles bursts).
    """
    train_start = train_end - timedelta(days=train_days - 1)
    sem = asyncio.Semaphore(concurrency)
    entries: dict[tuple[str, str], BiasEntry] = {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        pairs = list(market_pairs)
        results = await asyncio.gather(
            *(
                _train_one(s, t, train_start, train_end, model, client, sem,
                           truth_source=truth_source)
                for s, t in pairs
            )
        )

    for (s, t), e in zip(pairs, results):
        if e is not None:
            entries[(s.station_id, t)] = e
    return BiasTable(entries=entries)
