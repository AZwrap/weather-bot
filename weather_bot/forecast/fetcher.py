"""Open-Meteo ensemble forecast fetcher.

Open-Meteo aggregates ensemble forecasts from ECMWF, NOAA GFS, DWD ICON, and
ECCC GEM, free and without API key. Each model returns N hourly per-member
temperature traces; we extract the daily max per member to build an empirical
probability distribution over tomorrow's high.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Awaitable, Callable, Literal, TypeVar

import httpx
import numpy as np
import pandas as pd

T = TypeVar("T")


async def _with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    initial_backoff: float = 1.5,
) -> T:
    """Retry an async HTTP call on 429 / 5xx with exponential backoff."""
    backoff = initial_backoff
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except httpx.HTTPStatusError as exc:
            last = exc
            status = exc.response.status_code
            if status not in (429, 500, 502, 503, 504):
                raise
            if attempt == max_attempts - 1:
                raise
            retry_after = exc.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else backoff
            await asyncio.sleep(wait)
            backoff *= 2
    raise last  # type: ignore[misc]

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

EnsembleModel = Literal[
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "gem_global",
]

DEFAULT_MODELS: list[EnsembleModel] = [
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
]

# Matches both "temperature_2m" (control) and "temperature_2m_memberNN".
_MEMBER_KEY_RE = re.compile(r"^temperature_2m(_member\d+)?$")


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str  # IANA tz name, e.g. "America/New_York"


@dataclass
class EnsembleForecast:
    """Per-member hourly 2m temperature forecast for a location."""

    location: Location
    model: str
    issued_at: datetime  # when this object was built (UTC)
    times: pd.DatetimeIndex  # hourly timestamps in location-local time
    members: np.ndarray  # shape (n_members, n_hours), degrees Celsius

    @property
    def n_members(self) -> int:
        return self.members.shape[0]

    def daily_max(self, target: date) -> np.ndarray:
        """Daily max temperature per ensemble member for `target` (local date).

        Returns array of shape (n_members,) in °C.
        """
        mask = np.array([t.date() == target for t in self.times])
        if not mask.any():
            available = sorted({t.date() for t in self.times})
            raise ValueError(
                f"No forecast data for {target}. "
                f"Available dates: {available[0]} to {available[-1]}"
            )
        return np.nanmax(self.members[:, mask], axis=1)

    def daily_min(self, target: date) -> np.ndarray:
        mask = np.array([t.date() == target for t in self.times])
        if not mask.any():
            raise ValueError(f"No forecast data for {target}")
        return np.nanmin(self.members[:, mask], axis=1)


async def fetch_ensemble(
    location: Location,
    model: EnsembleModel = "ecmwf_ifs025",
    forecast_days: int = 7,
    client: httpx.AsyncClient | None = None,
) -> EnsembleForecast:
    """Fetch hourly per-member 2m temperature forecast from Open-Meteo."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": "temperature_2m",
        "models": model,
        "timezone": location.timezone,
        "forecast_days": forecast_days,
        "temperature_unit": "celsius",
    }

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=30.0)

    async def _fetch() -> dict:
        r = await client.get(ENSEMBLE_URL, params=params)
        r.raise_for_status()
        return r.json()

    try:
        data = await _with_retry(_fetch)
    finally:
        if owns_client:
            await client.aclose()

    hourly = data["hourly"]
    times = pd.DatetimeIndex(hourly["time"])

    member_keys = sorted(k for k in hourly if _MEMBER_KEY_RE.match(k))
    if not member_keys:
        raise RuntimeError(
            f"Open-Meteo response had no temperature_2m keys: {list(hourly)}"
        )

    rows = []
    for k in member_keys:
        rows.append([np.nan if v is None else v for v in hourly[k]])
    members = np.asarray(rows, dtype=float)

    return EnsembleForecast(
        location=location,
        model=model,
        issued_at=datetime.now(timezone.utc),
        times=times,
        members=members,
    )


async def fetch_multi_model(
    location: Location,
    models: list[EnsembleModel] | None = None,
    forecast_days: int = 7,
) -> dict[str, EnsembleForecast]:
    """Fetch ensembles from multiple models concurrently."""
    if models is None:
        models = list(DEFAULT_MODELS)

    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_ensemble(location, m, forecast_days, client) for m in models),
            return_exceptions=True,
        )

    out: dict[str, EnsembleForecast] = {}
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            # Don't kill the whole pipeline if one model is briefly down.
            print(f"[fetch_multi_model] {model} failed: {result}")
            continue
        out[model] = result
    return out


async def fetch_observed_max(
    location: Location,
    target: date,
    client: httpx.AsyncClient | None = None,
) -> float | None:
    """Fetch the actual observed daily max temperature for backtesting.

    Uses Open-Meteo's archive (ERA5-based reanalysis). Returns None if data
    is not yet available (typically a 2–5 day lag).
    """
    df = await fetch_observed_max_range(location, target, target, client)
    if df.empty or pd.isna(df.iloc[0]["observed_max_c"]):
        return None
    return float(df.iloc[0]["observed_max_c"])


DailyAgg = Literal["max", "min"]


def _daily_var(agg: DailyAgg) -> str:
    return "temperature_2m_max" if agg == "max" else "temperature_2m_min"


async def fetch_observed_range(
    location: Location,
    start: date,
    end: date,
    agg: DailyAgg = "max",
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch daily max or min observations from ERA5 archive over [start, end].

    Returns a DataFrame with columns: date, observed_c.
    """
    var = _daily_var(agg)
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": var,
        "timezone": location.timezone,
        "temperature_unit": "celsius",
    }
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=60.0)

    async def _fetch() -> dict:
        r = await client.get(ARCHIVE_URL, params=params)
        r.raise_for_status()
        return r.json()

    try:
        data = await _with_retry(_fetch)
    finally:
        if owns_client:
            await client.aclose()

    daily = data.get("daily", {})
    times = daily.get("time", [])
    values = daily.get(var, [])
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(t).date() for t in times],
            "observed_c": [np.nan if v is None else float(v) for v in values],
        }
    )


async def fetch_historical_forecast_range(
    location: Location,
    start: date,
    end: date,
    agg: DailyAgg = "max",
    model: str = "ecmwf_ifs025",
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch best-available historical daily max/min forecast for `model` over [start, end].

    Open-Meteo's historical-forecast-api stores past forecasts as a continuous
    time series built by concatenating short-range outputs from successive
    model runs. This is close in lead time to an evening-before-target trade.

    Returns a DataFrame with columns: date, forecast_c.
    """
    var = _daily_var(agg)
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": var,
        "models": model,
        "timezone": location.timezone,
        "temperature_unit": "celsius",
    }
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=60.0)

    async def _fetch() -> dict:
        r = await client.get(HISTORICAL_FORECAST_URL, params=params)
        r.raise_for_status()
        return r.json()

    try:
        data = await _with_retry(_fetch)
    finally:
        if owns_client:
            await client.aclose()

    daily = data.get("daily", {})
    times = daily.get("time", [])
    values = daily.get(var, [])
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(t).date() for t in times],
            "forecast_c": [np.nan if v is None else float(v) for v in values],
        }
    )


# Backwards-compatible aliases used by older modules and demos.
async def fetch_observed_max_range(
    location: Location,
    start: date,
    end: date,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    df = await fetch_observed_range(location, start, end, "max", client)
    return df.rename(columns={"observed_c": "observed_max_c"})


async def fetch_historical_forecast_max(
    location: Location,
    start: date,
    end: date,
    model: str = "ecmwf_ifs025",
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    df = await fetch_historical_forecast_range(location, start, end, "max", model, client)
    return df.rename(columns={"forecast_c": "forecast_max_c"})
