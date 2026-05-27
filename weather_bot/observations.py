"""Station-level observations from the Iowa State University ASOS archive.

Same METAR/ASOS feed Wunderground uses → matches what Polymarket resolves
on. Free, no signup, worldwide ICAO coverage going back decades. Verified
head-to-head against ERA5 on 2025-12-31:

    ┌─────────┬──────────────┬──────────────┬─────────────────────┐
    │ station │ ERA5 °C max  │ METAR °C max │ Polymarket resolved │
    ├─────────┼──────────────┼──────────────┼─────────────────────┤
    │ EGLC    │ 2.80         │ 5.00         │ 5°C                 │  ← METAR matches
    │ CYYZ    │ -4.30        │ -4.00        │ -4°C or below       │  ← METAR matches
    │ RKSI    │ -1.70        │ -1.00        │ -1°C or higher      │  ← METAR matches
    └─────────┴──────────────┴──────────────┴─────────────────────┘

If METAR is temporarily unavailable (Iowa State down, 429 burst, etc.) we
return an empty DataFrame rather than fall back to ERA5 — ERA5 disagrees
with airport observations enough to be worse than no data. Stations not on
the ASOS network at all (e.g. HKO Hong Kong Observatory) are explicitly
excluded from `weather_bot.locations.MARKETS`; we don't trade them.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from io import StringIO
from typing import Literal

import httpx
import numpy as np
import pandas as pd

from .forecast.fetcher import DailyAgg, Location

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

ObservationSource = Literal["metar"]


async def fetch_metar_daily(
    location: Location,
    icao: str,
    start: date,
    end: date,
    agg: DailyAgg = "max",
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch daily max or min from Iowa State ASOS METAR archive.

    Returns a DataFrame: date (datetime.date), observed_c (float).
    Days with no data come back missing from the result, not NaN, so the
    caller can treat absence as "skip".
    """
    # Iowa State's `year2/month2/day2` is exclusive — bump end by 1 day so
    # we get observations through the end of `end` in local tz.
    end_excl = end + timedelta(days=1)
    params = {
        "station": icao,
        "data": "tmpc",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "tz": location.timezone,
        "format": "onlycomma",
        "latlon": "no",
        "missing": "null",
        "trace": "null",
        "direct": "no",
    }

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=60.0)

    # Iowa State rate-limits hard (429 on bursts). Retry with backoff.
    backoff = 2.0
    text: str | None = None
    try:
        for attempt in range(5):
            try:
                r = await client.get(ASOS_URL, params=params)
                r.raise_for_status()
                text = r.text
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 4:
                    wait = float(exc.response.headers.get("retry-after", backoff))
                    await asyncio.sleep(wait)
                    backoff *= 2
                    continue
                raise
    finally:
        if owns:
            await client.aclose()
    if text is None:
        return pd.DataFrame(columns=["date", "observed_c"])

    # Some ASOS responses prepend a comment line. Skip until "station,valid,tmpc"
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("station,valid")),
        None,
    )
    if header_idx is None:
        return pd.DataFrame(columns=["date", "observed_c"])

    csv = "\n".join(lines[header_idx:])
    raw = pd.read_csv(StringIO(csv))
    if raw.empty or "tmpc" not in raw.columns:
        return pd.DataFrame(columns=["date", "observed_c"])

    raw["valid"] = pd.to_datetime(raw["valid"], errors="coerce")
    raw["tmpc"] = pd.to_numeric(raw["tmpc"], errors="coerce")
    raw = raw.dropna(subset=["valid", "tmpc"])
    if raw.empty:
        return pd.DataFrame(columns=["date", "observed_c"])

    raw["date"] = raw["valid"].dt.date
    if agg == "max":
        out = raw.groupby("date")["tmpc"].max()
    else:
        out = raw.groupby("date")["tmpc"].min()

    df = out.reset_index().rename(columns={"tmpc": "observed_c"})
    return df


async def fetch_metar_hourly_today(
    location: Location,
    icao: str,
    target_date: date,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch hourly METAR observations for `target_date` at `icao`.

    Returns DataFrame with columns:
      - local_dt (pd.Timestamp): observation time in station-local tz
      - temp_c (float): observed temperature in Celsius

    Used by the intraday METAR-feedback strategy to compute peak-so-far
    on the resolution day.

    As of Layer 2 (2026-05-17), this uses a multi-source fanout: races
    NOAA Aviation Weather Center + Iowa State ASOS in parallel and
    returns whichever responds first with non-empty data. Per-source
    health is tracked in `data/rate_limit_state.json` (a 429 from one
    source temporarily routes around it to the other). See
    `weather_bot/metar_sources.py` for details.

    Returns empty DataFrame on failure.
    """
    from weather_bot.metar_sources import fetch_metar_hourly_fanout

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=60.0)
    try:
        return await fetch_metar_hourly_fanout(
            client=client,
            icao=icao,
            target_date=target_date,
            tz=location.timezone,
        )
    finally:
        if owns:
            await client.aclose()


async def fetch_metar_hourly_range(
    location: Location,
    icao: str,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch hourly METAR observations between `start_date` and `end_date`
    (both inclusive) at `icao`.

    Returns DataFrame with columns:
      - local_dt (pd.Timestamp): observation time in station-local tz
      - temp_c (float): observed temperature in Celsius
      - date (date): station-local date (for groupby)

    Used by backtests that need many days of hourly METAR (e.g., the
    365-day peak-settlement analysis in `backtest_peak_settlement_365.py`).
    Issuing one query for the whole range is dramatically faster than
    one-per-day fetching (single Iowa State response vs N round-trips).
    """
    end_excl = end_date + timedelta(days=1)
    params = {
        "station": icao,
        "data": "tmpc",
        "year1": start_date.year, "month1": start_date.month, "day1": start_date.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "tz": location.timezone,
        "format": "onlycomma",
        "latlon": "no",
        "missing": "null",
        "trace": "null",
        "direct": "no",
    }
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=120.0)
    backoff = 2.0
    text: str | None = None
    try:
        for attempt in range(5):
            try:
                r = await client.get(ASOS_URL, params=params)
                r.raise_for_status()
                text = r.text
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 4:
                    wait = float(exc.response.headers.get("retry-after", backoff))
                    await asyncio.sleep(wait)
                    backoff *= 2
                    continue
                return pd.DataFrame(columns=["local_dt", "temp_c", "date"])
            except Exception:
                return pd.DataFrame(columns=["local_dt", "temp_c", "date"])
    finally:
        if owns:
            await client.aclose()
    if not text:
        return pd.DataFrame(columns=["local_dt", "temp_c", "date"])

    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("station,valid")),
        None,
    )
    if header_idx is None:
        return pd.DataFrame(columns=["local_dt", "temp_c", "date"])

    raw = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    if raw.empty or "tmpc" not in raw.columns:
        return pd.DataFrame(columns=["local_dt", "temp_c", "date"])

    raw["valid"] = pd.to_datetime(raw["valid"], errors="coerce")
    raw["tmpc"] = pd.to_numeric(raw["tmpc"], errors="coerce")
    raw = raw.dropna(subset=["valid", "tmpc"])
    if raw.empty:
        return pd.DataFrame(columns=["local_dt", "temp_c", "date"])

    out = raw[["valid", "tmpc"]].rename(columns={"valid": "local_dt", "tmpc": "temp_c"})
    out["date"] = out["local_dt"].dt.date
    return out


async def fetch_observed_truth(
    location: Location,
    icao: str,
    start: date,
    end: date,
    agg: DailyAgg = "max",
    source: ObservationSource = "metar",
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Truth-source router. Always METAR — no ERA5 fallback.

    Returns empty DataFrame if METAR fails. Callers handle missing data as
    "no resolution this round; try again next cron". No silent ERA5
    substitution: an unresolved record is more honest than a wrong one.
    """
    try:
        return await fetch_metar_daily(location, icao, start, end, agg, client)
    except Exception as exc:
        print(f"!! METAR fetch {icao} {start}..{end}: {exc}")
        return pd.DataFrame(columns=["date", "observed_c"])
