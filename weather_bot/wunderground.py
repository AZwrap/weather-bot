"""Wunderground daily-summary fetcher.

Wunderground is the source Polymarket reads for weather-market settlement.
This module fetches the daily max/min for a station-date directly from
Wunderground, so the bot's view of the answer matches the oracle's view —
eliminating the 0.5-1°C METAR-vs-Wunderground disagreement risk that
caused the RKSI/ZGSZ incidents on the previous deployment.

## Data source

We use the api.weather.com v1 historical observations endpoint that
backs the Wunderground web app:

  https://api.weather.com/v1/location/{ICAO}:9:US/observations/historical.json

The endpoint requires an `apiKey` query param. The web app exposes a
public key visible in its bundle — it has been used by many open-source
weather tools for years. We document it here transparently so the
operator knows what they're using.

If the key changes, Wunderground changes the endpoint, or the request
is rate-limited / blocked, this module returns (None, None) so the
calling strategy can fall back to METAR-only behaviour. **Do not let a
Wunderground failure crash the bot.**

## Why not METAR directly?

METAR feeds the SAME underlying ASOS data but Wunderground applies
post-processing (rounding, sensor selection, sustained vs instantaneous,
QA flags). The "official" daily max that appears on Wunderground's
history page — and that Polymarket reads — can differ from a naive
max-of-hourly-METAR by up to ~1°C. See the May 2026 RKSI incident
documented in `project_decommission_2026-05-26.md`.

## Caching

Per-(icao, date) responses are cached in-memory for the lifetime of the
process. Polite throttling: at most one request per second across the
whole module. Honour HTTP 429 with exponential backoff.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx

# Public key exposed by the Wunderground web app (visible in its
# JavaScript bundle, used by many community tools). If Wunderground
# rotates this we fail-soft and the calling code falls back to METAR.
DEFAULT_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

ENDPOINT_TEMPLATE = (
    "https://api.weather.com/v1/location/{icao}:9:US/observations/historical.json"
)
USER_AGENT = "weather-bot-shadow-harness/0.1 (research)"

# Politeness throttle: serialise requests via an asyncio.Lock + last-call
# timestamp guard so we never hit Wunderground more often than once per
# second across the entire process.
_throttle_lock = asyncio.Lock()
_last_call_ts: float = 0.0
_MIN_INTERVAL_S = 1.0


_cache: dict[tuple[str, str], "WundergroundDaily"] = {}


@dataclass
class WundergroundDaily:
    icao: str
    target_date_iso: str
    daily_max_c: float | None
    daily_min_c: float | None
    n_observations: int
    last_observation_utc: str | None
    raw_status: str   # "ok" | "no_data" | "http_error" | "parse_error" | "rate_limited"
    fetched_at_utc: str


def _parse_response(payload: dict[str, Any]) -> tuple[float | None, float | None, int, str | None]:
    """Extract (daily_max_c, daily_min_c, n_obs, last_obs_iso_utc) from
    the api.weather.com historical observations JSON.

    The endpoint returns `{"observations": [...]}` where each observation
    has `valid_time_gmt` (unix), `temp` (Celsius when units=m), `wx_phrase`,
    etc. We compute the daily extreme from the temp series — Wunderground
    derives its "high" / "low" the same way.
    """
    obs = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(obs, list) or not obs:
        return None, None, 0, None
    temps_c: list[float] = []
    last_ts: int = 0
    for o in obs:
        if not isinstance(o, dict):
            continue
        t = o.get("temp")
        if t is None:
            continue
        try:
            temps_c.append(float(t))
        except (TypeError, ValueError):
            continue
        try:
            vt = int(o.get("valid_time_gmt", 0))
            if vt > last_ts:
                last_ts = vt
        except (TypeError, ValueError):
            pass
    if not temps_c:
        return None, None, len(obs), None
    last_iso = (
        datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
        if last_ts else None
    )
    return max(temps_c), min(temps_c), len(temps_c), last_iso


async def _throttled_get(
    url: str,
    params: dict[str, Any],
    client: httpx.AsyncClient,
    max_retries: int = 3,
) -> httpx.Response | None:
    """Throttle, fetch, retry on 429. None on terminal failure."""
    global _last_call_ts
    backoff_s = 2.0
    for attempt in range(max_retries):
        async with _throttle_lock:
            now = asyncio.get_event_loop().time()
            wait_for = max(0.0, _MIN_INTERVAL_S - (now - _last_call_ts))
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            _last_call_ts = asyncio.get_event_loop().time()
        try:
            r = await client.get(
                url, params=params, headers={"User-Agent": USER_AGENT},
            )
        except httpx.RequestError:
            await asyncio.sleep(backoff_s)
            backoff_s *= 2
            continue
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            try:
                wait_for = float(ra) if ra is not None else backoff_s
            except ValueError:
                wait_for = backoff_s
            await asyncio.sleep(wait_for)
            backoff_s *= 2
            continue
        return r
    return None


async def fetch_wunderground_daily(
    icao: str,
    target_date: date,
    client: httpx.AsyncClient | None = None,
    api_key: str = DEFAULT_API_KEY,
) -> WundergroundDaily:
    """Fetch the Wunderground daily summary for (icao, target_date).

    Returns a WundergroundDaily record with `raw_status` indicating
    success ("ok") or the failure mode ("no_data", "http_error",
    "parse_error", "rate_limited"). On any failure both max/min are
    None and the caller should fall back to METAR.
    """
    key = (icao.upper(), target_date.isoformat())
    if key in _cache:
        return _cache[key]

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=30.0)

    ymd_compact = target_date.strftime("%Y%m%d")
    url = ENDPOINT_TEMPLATE.format(icao=icao.upper())
    params = {
        "apiKey": api_key,
        "units": "m",       # Celsius
        "startDate": ymd_compact,
        "endDate": ymd_compact,
    }
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        r = await _throttled_get(url, params, client)
    finally:
        if owns:
            await client.aclose()

    if r is None:
        rec = WundergroundDaily(
            icao=icao.upper(), target_date_iso=target_date.isoformat(),
            daily_max_c=None, daily_min_c=None, n_observations=0,
            last_observation_utc=None, raw_status="rate_limited",
            fetched_at_utc=now_iso,
        )
        _cache[key] = rec
        return rec

    if r.status_code != 200:
        rec = WundergroundDaily(
            icao=icao.upper(), target_date_iso=target_date.isoformat(),
            daily_max_c=None, daily_min_c=None, n_observations=0,
            last_observation_utc=None, raw_status=f"http_{r.status_code}",
            fetched_at_utc=now_iso,
        )
        _cache[key] = rec
        return rec

    try:
        payload = r.json()
    except ValueError:
        rec = WundergroundDaily(
            icao=icao.upper(), target_date_iso=target_date.isoformat(),
            daily_max_c=None, daily_min_c=None, n_observations=0,
            last_observation_utc=None, raw_status="parse_error",
            fetched_at_utc=now_iso,
        )
        _cache[key] = rec
        return rec

    daily_max, daily_min, n_obs, last_iso = _parse_response(payload)
    rec = WundergroundDaily(
        icao=icao.upper(), target_date_iso=target_date.isoformat(),
        daily_max_c=daily_max, daily_min_c=daily_min,
        n_observations=n_obs, last_observation_utc=last_iso,
        raw_status="ok" if daily_max is not None else "no_data",
        fetched_at_utc=now_iso,
    )
    _cache[key] = rec
    return rec


def clear_cache() -> None:
    """Drop the in-process cache. Useful for re-running a scan within
    the same long-lived process (daemon mode)."""
    _cache.clear()
