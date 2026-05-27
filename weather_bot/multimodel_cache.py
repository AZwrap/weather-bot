"""Cache for multi-model deterministic forecasts.

Open-Meteo's multi-model `/v1/forecast` returns identical values between
upstream model run cycles (ECMWF/AIFS/GEM at 00,12 UTC; GFS/ICON at
00,06,12,18 UTC, with ~5h availability lag). The bot's */20-min
log_forecasts cron previously refetched on every tick — burning ~228 OM
calls per run on data that hadn't changed since the last cron.

This cache invalidates on the *deterministic* schedule used by
`ecmwf_run_init_utc(now)` (most recent 6-hour multiple ≥5h ago). When the
"latest available" run advances past the cached one, every station refetches
on the next cron — picking up the new run within ≤20 min, same as before.
For the other 17 of every 18 ticks, the cached payload is reused.

Math (57 stations, 4 multimodel models):
  Without cache: 72 runs/day × 228 = 16,416 multimodel calls/day
  With cache:     4 runs/day × 228 =    912 multimodel calls/day  (~18× reduction)

The cache file holds exactly one entry per station — entries are
overwritten on each refresh, so the file does not grow over time.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_CACHE_PATH = Path("data/multimodel_cache.json")


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> dict[str, dict]:
    """Load the per-station cache. Missing or malformed file returns {}."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, dict], path: Path = DEFAULT_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def get_fresh(
    cache: dict[str, dict],
    station_id: str,
    current_run_init: datetime,
    dates: list[date],
) -> dict | None:
    """Return cached payload iff it matches the current run-init AND date range.

    A cached entry is fresh when:
      - its `run_init_utc` is >= the current "latest available" run init
        (i.e. no new model run has landed since we cached), AND
      - its `dates` exactly matches the requested date range (so we don't
        serve yesterday's [today, tomorrow] payload after midnight rolls).
    """
    entry = cache.get(station_id)
    if entry is None:
        return None
    cached_run = _parse_iso_dt(entry.get("run_init_utc"))
    if cached_run is None or cached_run < current_run_init:
        return None
    cached_dates = entry.get("dates")
    if cached_dates != [d.isoformat() for d in dates]:
        return None
    return entry.get("payload")


def put(
    cache: dict[str, dict],
    station_id: str,
    current_run_init: datetime,
    dates: list[date],
    payload: dict,
    fetched_at_utc: datetime,
) -> None:
    cache[station_id] = {
        "run_init_utc": current_run_init.isoformat(),
        "dates": [d.isoformat() for d in dates],
        "fetched_at_utc": fetched_at_utc.isoformat(),
        "payload": payload,
    }
