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
