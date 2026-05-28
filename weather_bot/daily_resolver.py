"""Daily resolver — fetch Wunderground actual_obs_c for settled events.

Runs once per events-refresh tick (= every 5 min by default). For each
(station, target, target_date) where:
  - midend-local-utc has passed by at least RESOLUTION_GRACE_HOURS
  - we don't already have actual_obs_c in forward_log.jsonl

we hit api.weather.com (via weather_bot.wunderground), extract the daily
extreme in the target direction (max or min), and append a resolved
record to forward_log.

This feeds:
  - persistence_tail (uses forward_log priors to fire on tail buckets)
  - analyze_publication_window (joins snapshots with resolutions)
  - any future PnL accounting

Idempotency
===========
forward_log.jsonl is append-only. Before fetching WUG for a (station,
target, date) tuple, we scan forward_log for an existing entry with
actual_obs_c set — if present, skip.

Failure modes
=============
- WUG returns no_data (early in resolution day, before observations
  publish): skip this tick, try next refresh.
- WUG returns http_error: log, retry next tick.
- Both: never crash the daemon — the resolver is fail-soft.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .publication_window import midend_local_utc
from .wunderground import clear_cache as _wug_clear_cache
from .wunderground import fetch_wunderground_daily

DEFAULT_FORWARD_LOG_PATH = Path("data/forward_log.jsonl")
DEFAULT_RESOLVER_AUDIT_PATH = Path("data/daily_resolver_audit.jsonl")

RESOLUTION_GRACE_HOURS: float = 2.0
"""Hours past midend-local to wait before trying to resolve. WUG can
lag the actual end-of-day by 30-90 min for the final hourly METAR to
land and the daily summary to update."""

LOOKBACK_DAYS: int = 3
"""How many days back to scan for unresolved events. Anything older
either resolved or is permanently missing. 3 days covers weekend +
2-day delay on the publication-window tail."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_resolved_keys(
    forward_log_path: Path = DEFAULT_FORWARD_LOG_PATH,
) -> set[tuple[str, str, str]]:
    """Return the set of (station_id, target, target_date_iso) tuples
    that already have actual_obs_c set in forward_log.jsonl."""
    out: set[tuple[str, str, str]] = set()
    if not forward_log_path.exists():
        return out
    with forward_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("actual_obs_c") is None:
                continue
            sid = r.get("station_id")
            tgt = r.get("target")
            td = r.get("target_date")
            if sid and tgt and td:
                out.add((sid, tgt, td))
    return out


def _append_resolved_record(
    *,
    station_id: str,
    target: str,
    target_date_iso: str,
    actual_obs_c: float,
    source: str,
    forward_log_path: Path = DEFAULT_FORWARD_LOG_PATH,
) -> None:
    record = {
        "station_id": station_id,
        "target": target,
        "target_date": target_date_iso,
        "actual_obs_c": float(actual_obs_c),
        "resolved_at_utc": _now_utc_iso(),
        "source": source,
    }
    try:
        forward_log_path.parent.mkdir(parents=True, exist_ok=True)
        with forward_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _audit(record: dict, path: Path = DEFAULT_RESOLVER_AUDIT_PATH) -> None:
    """Append-only audit so we can debug resolver progress without
    polluting forward_log."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


async def resolve_settled_events(
    *,
    events_by_sk: dict[tuple[str, str, str], Any],
    stations_by_sk: dict[tuple[str, str, str], Any],
    http: httpx.AsyncClient,
    grace_hours: float = RESOLUTION_GRACE_HOURS,
    lookback_days: int = LOOKBACK_DAYS,
    forward_log_path: Path = DEFAULT_FORWARD_LOG_PATH,
) -> dict[str, int]:
    """Walk the active event set + history, resolve anything ready.

    Returns counts dict:
      attempted     candidates that passed the time / dedupe gates
      resolved      successfully fetched + appended actual_obs_c
      skipped_no_data  WUG returned no_data (will retry next tick)
      skipped_error  WUG returned http_error / parse_error
      skipped_already already in forward_log with actual_obs_c set
      skipped_too_early not yet midend + grace
    """
    counts: dict[str, int] = {
        "attempted": 0, "resolved": 0,
        "skipped_no_data": 0, "skipped_error": 0,
        "skipped_already": 0, "skipped_too_early": 0,
    }
    now_utc = datetime.now(timezone.utc)
    cutoff_date = (now_utc.date() - timedelta(days=lookback_days)).isoformat()
    resolved_keys = _load_resolved_keys(forward_log_path)

    # Build a set of (sid, target, target_date_iso) tuples to attempt.
    # Includes events in the active set + recent unresolved past days
    # gleaned from station presence.
    candidates: set[tuple[str, str, str]] = set()
    for sk in events_by_sk.keys():
        sid, target, td_iso = sk
        if td_iso < cutoff_date:
            continue
        candidates.add(sk)

    # Also include yesterday's tuples for stations we know about — they
    # might have resolved overnight without us picking up a present-day
    # event for them yet. We use the stations_by_sk active set as the
    # universe.
    yesterday_iso = (now_utc.date() - timedelta(days=1)).isoformat()
    for sk in stations_by_sk.keys():
        sid, target, _ = sk
        candidates.add((sid, target, yesterday_iso))

    for sk in candidates:
        sid, target, td_iso = sk
        if sk in resolved_keys:
            counts["skipped_already"] += 1
            continue

        station = stations_by_sk.get(sk)
        if station is None:
            # Try to find a station in any (sid, target, *) bucket
            for k, s in stations_by_sk.items():
                if k[0] == sid:
                    station = s
                    break
            if station is None:
                continue

        try:
            target_date = date.fromisoformat(td_iso)
        except ValueError:
            continue

        midend = midend_local_utc(target_date, station)
        if now_utc < midend + timedelta(hours=grace_hours):
            counts["skipped_too_early"] += 1
            continue

        counts["attempted"] += 1
        # Clear cache so we re-hit (in case a prior tick got partial data)
        _wug_clear_cache()
        wug = await fetch_wunderground_daily(sid, target_date, client=http)
        actual_c = wug.daily_max_c if target == "max" else wug.daily_min_c

        if actual_c is None:
            if wug.raw_status == "no_data":
                counts["skipped_no_data"] += 1
            else:
                counts["skipped_error"] += 1
            _audit({
                "ts_utc": _now_utc_iso(),
                "result": "skipped",
                "station_id": sid, "target": target, "target_date": td_iso,
                "wug_status": wug.raw_status, "n_obs": wug.n_observations,
            })
            continue

        _append_resolved_record(
            station_id=sid, target=target, target_date_iso=td_iso,
            actual_obs_c=float(actual_c), source="wunderground",
            forward_log_path=forward_log_path,
        )
        counts["resolved"] += 1
        _audit({
            "ts_utc": _now_utc_iso(),
            "result": "resolved",
            "station_id": sid, "target": target, "target_date": td_iso,
            "actual_obs_c": float(actual_c),
            "wug_n_obs": wug.n_observations,
        })
        # Avoid double-resolve within the same tick
        resolved_keys.add(sk)

    return counts
