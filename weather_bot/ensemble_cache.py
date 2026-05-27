"""Cache for ECMWF ensemble forecasts.

Mirrors `multimodel_cache.py` but for `fetch_ensemble` (the ECMWF ensemble
fetch in log_forecasts). Open-Meteo's ensemble endpoint serves the same
data between upstream model runs (ECMWF re-runs every 6 UTC hours with
~5h availability lag). The bot's */20-min log_forecasts cron was
refetching every tick — burning ~49 Open-Meteo ensemble calls per run
on data identical to 17 of every 18 ticks. Audit 2026-05-16 found 536
429-rate-limit errors over the live window as a result.

Pattern (mirrors `multimodel_cache.py`):
  load_cache(path) -> dict[station_id, entry]
  get_fresh(cache, station_id, current_run_init) -> EnsembleForecast | None
  put(cache, station_id, forecast, current_run_init, fetched_at_utc)
  save_cache(cache, path)

Math (49 stations, ECMWF ensemble only):
  Without cache: 72 ticks/day × 49 stations =  3,528 ensemble calls/day
  With cache:     4 ticks/day × 49 stations =    196 ensemble calls/day  (~18× reduction)
  → eliminates the 429 cascade; cron ticks shed retry-wait latency.

The cache file holds exactly one entry per station — entries are
overwritten on each refresh, so the file does not grow over time.

Serialization: ECMWF ensemble payloads carry numpy ndarrays + a
pandas DatetimeIndex. We round-trip via JSON-compatible types
(list[list[float]] and list[str] of ISO timestamps) so the cache file
stays inspectable + version-portable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .forecast.fetcher import EnsembleForecast

DEFAULT_CACHE_PATH = Path("data/ensemble_cache.json")


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> dict[str, dict]:
    """Load the per-station cache. Missing or malformed file returns {}."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, dict], path: Path = DEFAULT_CACHE_PATH) -> None:
    """Atomic save via weather_bot.atomic_write — see that module for
    why non-atomic writes can corrupt this cache on process crash
    (truncated file → reload returns {} → 49 stations × 6 fresh API
    calls per cycle, burning Open-Meteo rate-limit budget)."""
    from weather_bot.atomic_write import atomic_write_json
    atomic_write_json(path, cache)


def _parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _serialize_forecast(forecast: "EnsembleForecast") -> dict:
    """EnsembleForecast → JSON-compatible dict.

    Stores location fields, model id, issued_at, times as ISO strings,
    and members as list-of-lists of floats. Re-parsing is cheap; the
    network call we avoided is what matters."""
    return {
        "location": {
            "name": forecast.location.name,
            "latitude": float(forecast.location.latitude),
            "longitude": float(forecast.location.longitude),
            "timezone": forecast.location.timezone,
        },
        "model": str(forecast.model),
        "issued_at": forecast.issued_at.isoformat(),
        "times": [t.isoformat() for t in forecast.times],
        "members": [
            [None if (v != v) else float(v) for v in row]  # v != v handles NaN
            for row in forecast.members
        ],
    }


def _deserialize_forecast(payload: dict) -> "EnsembleForecast":
    """JSON-compatible dict → EnsembleForecast (re-instantiates numpy arrays)."""
    # Local imports to avoid forecast/fetcher import-time cost when the
    # cache is just being loaded but not deserialized.
    import numpy as np
    import pandas as pd
    from .forecast.fetcher import EnsembleForecast, Location

    loc_d = payload["location"]
    location = Location(
        name=str(loc_d["name"]),
        latitude=float(loc_d["latitude"]),
        longitude=float(loc_d["longitude"]),
        timezone=str(loc_d["timezone"]),
    )
    times = pd.DatetimeIndex(payload["times"])
    rows = [
        [float("nan") if v is None else float(v) for v in row]
        for row in payload["members"]
    ]
    members = np.asarray(rows, dtype=float)
    issued_at = _parse_iso_dt(payload.get("issued_at")) or datetime.now(timezone.utc)
    return EnsembleForecast(
        location=location,
        model=str(payload["model"]),
        issued_at=issued_at,
        times=times,
        members=members,
    )


def get_fresh(
    cache: dict[str, dict],
    station_id: str,
    current_run_init: datetime,
    model: str,
) -> "EnsembleForecast | None":
    """Return cached EnsembleForecast iff it matches the current run-init AND model.

    A cached entry is fresh when:
      - its `run_init_utc` is >= the current "latest available" run init
        (no new model run has landed since we cached), AND
      - its `model` matches the requested model.
    """
    entry = cache.get(station_id)
    if entry is None:
        return None
    cached_run = _parse_iso_dt(entry.get("run_init_utc"))
    if cached_run is None or cached_run < current_run_init:
        return None
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return None
    if str(payload.get("model")) != str(model):
        return None
    try:
        return _deserialize_forecast(payload)
    except (KeyError, ValueError, TypeError):
        return None


def put(
    cache: dict[str, dict],
    station_id: str,
    forecast: "EnsembleForecast",
    current_run_init: datetime,
    fetched_at_utc: datetime | None = None,
) -> None:
    """Store a freshly-fetched EnsembleForecast in the cache, replacing any
    prior entry for the station."""
    if fetched_at_utc is None:
        fetched_at_utc = datetime.now(timezone.utc)
    cache[station_id] = {
        "run_init_utc": current_run_init.isoformat(),
        "fetched_at_utc": fetched_at_utc.isoformat(),
        "payload": _serialize_forecast(forecast),
    }
