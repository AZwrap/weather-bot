"""Layer 2: Multi-source METAR fanout for low-latency observation ingestion.

The bot previously used a single source (Iowa State ASOS archive) for
all METAR fetches. Iowa State is reliable but is a mirror — it adds
~10-30s of latency vs querying NOAA more directly. For the live bot,
faster observation ingestion means cross-up cancel and Layer 7
guaranteed-NO-buy fire earlier, capturing more EV before market reprice.

Source priority (per the tomorrow-morning checklist):
  1. NOAA Aviation Weather Center (AWC) — primary, ~50-70s sensor lag
  2. Iowa State ASOS — tertiary fallback, ~60-90s sensor lag (and the
     only source we'll keep on the 1-min ASOS endpoint for sub-hourly).

MADIS was a third primary candidate but requires registration/auth;
deferring until empirically needed. Two-source fanout provides good
redundancy already.

The fanout issues parallel requests and returns the first successful
response. If both sources are blocked / failing, returns empty.

Circuit breaker: per-source health is tracked in
`data/rate_limit_state.json`. On sustained 429s or 403s, a source is
temporarily skipped for the next call. State persists across cron runs
and across daemon restarts.

The aggressive sub-second cadence + burst polling features described
in the Layer 2 plan live in the Layer 1 daemon (long-running process
can sustain a tight loop). This module handles single-call fanout and
circuit-breaker bookkeeping; daemon orchestrates the polling cadence.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import httpx
import pandas as pd

NOAA_AWC_URL = "https://aviationweather.gov/api/data/metar"
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

DEFAULT_RATE_LIMIT_STATE_PATH = Path("data/rate_limit_state.json")

# Block durations on rate-limit signals
BLOCK_AFTER_429_SECONDS = 60.0  # short cool-down after one 429
BLOCK_AFTER_403_SECONDS = 1800.0  # IP-block-class; back off 30 min


# ──────────────────────────────────────────────────────────────────────────
# Source health tracking
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SourceHealth:
    """Per-source health tracking. Persisted across runs."""
    name: str
    request_count: int = 0
    success_count: int = 0
    error_429: int = 0
    error_403: int = 0
    error_other: int = 0
    last_success_at_utc: Optional[str] = None
    last_error_at_utc: Optional[str] = None
    blocked_until_utc: Optional[str] = None  # ISO timestamp

    def is_blocked(self, now: Optional[datetime] = None) -> bool:
        if self.blocked_until_utc is None:
            return False
        try:
            blocked_until = datetime.fromisoformat(self.blocked_until_utc)
            if blocked_until.tzinfo is None:
                blocked_until = blocked_until.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False
        now = now or datetime.now(timezone.utc)
        return now < blocked_until

    def mark_success(self) -> None:
        self.request_count += 1
        self.success_count += 1
        self.last_success_at_utc = datetime.now(timezone.utc).isoformat()

    def mark_error(self, status_code: int) -> None:
        self.request_count += 1
        self.last_error_at_utc = datetime.now(timezone.utc).isoformat()
        if status_code == 429:
            self.error_429 += 1
            self._block_for(BLOCK_AFTER_429_SECONDS)
        elif status_code == 403:
            self.error_403 += 1
            self._block_for(BLOCK_AFTER_403_SECONDS)
        else:
            self.error_other += 1

    def _block_for(self, seconds: float) -> None:
        until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        self.blocked_until_utc = until.isoformat()


def load_rate_limit_state(
    path: Path = DEFAULT_RATE_LIMIT_STATE_PATH,
) -> dict[str, SourceHealth]:
    """Load per-source health state from disk. Missing file → empty dict."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, SourceHealth] = {}
    sources = data.get("sources", {})
    if not isinstance(sources, dict):
        return {}
    for name, fields in sources.items():
        if not isinstance(fields, dict):
            continue
        try:
            out[name] = SourceHealth(name=name, **{
                k: v for k, v in fields.items() if k in SourceHealth.__dataclass_fields__
            })
        except Exception:
            continue
    return out


def save_rate_limit_state(
    state: dict[str, SourceHealth],
    path: Path = DEFAULT_RATE_LIMIT_STATE_PATH,
) -> None:
    """Atomically write per-source health state to disk."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {name: asdict(h) for name, h in state.items()},
    }
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_or_create_health(state: dict[str, SourceHealth], name: str) -> SourceHealth:
    if name not in state:
        state[name] = SourceHealth(name=name)
    return state[name]


# ──────────────────────────────────────────────────────────────────────────
# Source-specific fetchers
# ──────────────────────────────────────────────────────────────────────────

EMPTY_DF = pd.DataFrame(columns=["local_dt", "temp_c"])


async def fetch_iowa_state_hourly(
    client: httpx.AsyncClient,
    icao: str,
    target_date: date,
    tz: str,
    health: SourceHealth,
) -> pd.DataFrame:
    """Fetch hourly METAR for `target_date` at `icao` from Iowa State ASOS.

    Returns DataFrame[local_dt, temp_c] (empty on failure or empty result).
    Updates `health` to record success / error / rate-limit.
    """
    if health.is_blocked():
        return EMPTY_DF
    end_excl = target_date + timedelta(days=1)
    params = {
        "station": icao,
        "data": "tmpc",
        "year1": target_date.year, "month1": target_date.month, "day1": target_date.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "tz": tz,
        "format": "onlycomma",
        "latlon": "no",
        "missing": "null",
        "trace": "null",
        "direct": "no",
    }
    try:
        r = await client.get(ASOS_URL, params=params, timeout=60.0)
        r.raise_for_status()
        text = r.text
    except httpx.HTTPStatusError as exc:
        health.mark_error(exc.response.status_code)
        return EMPTY_DF
    except asyncio.CancelledError:
        # Fanout loser — task cancelled because the other source won.
        # Re-raise so asyncio sees a proper cancel; caller (drain loop)
        # is responsible for swallowing this without logging tracebacks.
        raise
    except BaseException:
        # Any other failure (incl. NetworkError, TimeoutException, etc.)
        # is a soft failure — fanout will fall back to other sources.
        health.mark_error(0)
        return EMPTY_DF

    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("station,valid")),
        None,
    )
    if header_idx is None:
        # Empty or malformed — count as soft failure (not an error)
        return EMPTY_DF

    try:
        raw = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    except Exception:
        return EMPTY_DF
    if raw.empty or "tmpc" not in raw.columns:
        return EMPTY_DF

    raw["valid"] = pd.to_datetime(raw["valid"], errors="coerce")
    raw["tmpc"] = pd.to_numeric(raw["tmpc"], errors="coerce")
    raw = raw.dropna(subset=["valid", "tmpc"])
    if raw.empty:
        return EMPTY_DF

    raw = raw[raw["valid"].dt.date == target_date]
    df = raw[["valid", "tmpc"]].rename(columns={"valid": "local_dt", "tmpc": "temp_c"})
    if not df.empty:
        health.mark_success()
    return df


async def fetch_noaa_awc_hourly(
    client: httpx.AsyncClient,
    icao: str,
    target_date: date,
    tz: str,
    health: SourceHealth,
) -> pd.DataFrame:
    """Fetch recent hourly METAR for `icao` from NOAA Aviation Weather Center.

    NOAA AWC returns the last N hours of METAR observations as JSON. We
    filter to observations on `target_date` in station-local time.

    Returns DataFrame[local_dt, temp_c] (empty on failure or empty result).
    Updates `health` to record success / error.
    """
    if health.is_blocked():
        return EMPTY_DF
    # Request the last 36 hours to ensure we capture all of today's
    # observations (handles UTC ↔ local timezone offsets up to ±12h).
    params = {
        "ids": icao,
        "format": "json",
        "hours": 36,
        "taf": "false",
    }
    try:
        r = await client.get(NOAA_AWC_URL, params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as exc:
        health.mark_error(exc.response.status_code)
        return EMPTY_DF
    except asyncio.CancelledError:
        raise  # propagate; drain loop swallows
    except BaseException:
        health.mark_error(0)
        return EMPTY_DF

    if not isinstance(data, list) or not data:
        return EMPTY_DF

    rows: list[tuple[pd.Timestamp, float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        temp = entry.get("temp")
        report_time = entry.get("reportTime") or entry.get("obsTime")
        if temp is None or report_time is None:
            continue
        try:
            temp_c = float(temp)
        except (TypeError, ValueError):
            continue
        try:
            # reportTime is ISO-ish UTC; obsTime can be epoch
            if isinstance(report_time, (int, float)):
                ts = pd.Timestamp(report_time, unit="s", tz="UTC")
            else:
                ts = pd.Timestamp(report_time)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
        except Exception:
            continue
        # Convert to station-local time
        try:
            local_ts = ts.tz_convert(tz)
        except Exception:
            continue
        if local_ts.date() != target_date:
            continue
        rows.append((local_ts.tz_localize(None), temp_c))

    if not rows:
        return EMPTY_DF

    df = pd.DataFrame(rows, columns=["local_dt", "temp_c"])
    df = df.sort_values("local_dt").reset_index(drop=True)
    health.mark_success()
    return df


# ──────────────────────────────────────────────────────────────────────────
# Fanout
# ──────────────────────────────────────────────────────────────────────────

async def fetch_metar_hourly_fanout(
    client: httpx.AsyncClient,
    icao: str,
    target_date: date,
    tz: str,
    *,
    state_path: Path = DEFAULT_RATE_LIMIT_STATE_PATH,
    persist_state: bool = True,
) -> pd.DataFrame:
    """Race NOAA AWC and Iowa State for the first successful response.

    Strategy:
      1. Start both primary sources in parallel.
      2. Wait for the FIRST non-empty result; return it.
      3. If both come back empty (or error), return empty DataFrame.

    Health state is persisted (so a 429 in this run blocks the source
    on the next run too, until the cool-down expires).
    """
    state = load_rate_limit_state(state_path)
    h_noaa = get_or_create_health(state, "noaa_awc")
    h_iowa = get_or_create_health(state, "iowa_state")

    # Launch both. asyncio.wait with FIRST_COMPLETED lets us return on
    # whichever finishes first with non-empty data.
    tasks = {
        asyncio.create_task(fetch_noaa_awc_hourly(client, icao, target_date, tz, h_noaa)): "noaa_awc",
        asyncio.create_task(fetch_iowa_state_hourly(client, icao, target_date, tz, h_iowa)): "iowa_state",
    }

    result: pd.DataFrame = EMPTY_DF
    pending = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                df = task.result()
            except BaseException:
                # Task raised; either an exception or it was cancelled
                # (the other source's race). Either way, move on to the
                # next done task / wait round.
                continue
            if not df.empty:
                result = df
                # Cancel remaining tasks — we have what we need
                for p in pending:
                    p.cancel()
                pending = set()
                break

    # Drain any cancelled tasks to suppress warnings. Both Exception
    # subclasses AND asyncio.CancelledError (BaseException) may arise
    # depending on which await was interrupted by .cancel().
    for task in tasks:
        if task.cancelled() or task.done():
            continue
        try:
            await task
        except BaseException:
            # Includes CancelledError (BaseException) and any other
            # mid-flight failure from the cancelled task. Swallow all.
            pass

    if persist_state:
        try:
            save_rate_limit_state(state, state_path)
        except Exception:
            # Don't fail the fetch on a state-save error
            pass

    return result
