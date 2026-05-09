r"""Fill in actual observations on records whose target date is past.

ERA5 archive lags by ~5 days. This script scans the forward log for
unresolved records whose target_date is at least `--lag-days` ago and
fills `actual_obs_c` from the Open-Meteo archive.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python resolve_log.py
    python resolve_log.py --lag-days 7
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
import json
import re
import pandas as pd

from weather_bot.forecast.fetcher import fetch_observed_range
from weather_bot.forward_log import (
    DEFAULT_LOG_PATH,
    ForwardLogRecord,
    load_records,
    write_all_records,
)
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.observations import fetch_observed_truth

GAMMA_BASE = "https://gamma-api.polymarket.com"
_INT_RE = re.compile(r"-?\d+")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    p.add_argument("--lag-days", type=int, default=1,
                   help="Resolution lag — only resolve records whose "
                        "target_date is at least this many days ago. "
                        "Default 1 + cron at 12 UTC catches all stations "
                        "(west-coast US local day ends ~07 UTC of day+1, "
                        "so 12 UTC gives a 5h buffer). Was 6 in ERA5 era.")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()

    log_path = Path(args.log)
    records = load_records(log_path)
    if not records:
        sys.exit(f"No records in {log_path}")

    today_utc = datetime.now(timezone.utc).date()
    cutoff = today_utc - timedelta(days=args.lag_days)
    pending = [
        r for r in records
        if not r.is_resolved and r.target_date <= cutoff
    ]
    print(
        f"{len(records)} total records, {len(pending)} pending resolution "
        f"(target_date ≤ {cutoff})"
    )
    if not pending:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def fetch_polymarket_winner(
        slug: str | None, client: httpx.AsyncClient
    ) -> tuple[str, int] | None:
        """If the event is closed and one bucket resolved Yes, return its label
        and threshold. Used to cross-check our rounding rule."""
        if not slug:
            return None
        async with sem:
            try:
                rr = await client.get(
                    f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=20.0,
                )
                rr.raise_for_status()
                events = rr.json()
            except Exception as exc:
                print(f"!! gamma fetch {slug}: {exc}")
                return None
        if not events:
            return None
        ev = events[0]
        if not ev.get("closed"):
            return None
        for m in ev.get("markets", []):
            prices_raw = m.get("outcomePrices")
            try:
                prices = (
                    json.loads(prices_raw) if isinstance(prices_raw, str)
                    else (prices_raw or [])
                )
            except json.JSONDecodeError:
                continue
            if prices and float(prices[0]) >= 0.5:
                label = m.get("groupItemTitle") or ""
                ints = _INT_RE.findall(label)
                threshold = int(ints[0]) if ints else int(m.get("groupItemThreshold") or 0)
                return label, threshold
        return None

    async def resolve_one(r: ForwardLogRecord, client: httpx.AsyncClient) -> bool:
        station = STATIONS_BY_ID.get(r.station_id)
        if station is None:
            print(f"!! unknown station_id {r.station_id} — skipping")
            return False
        async with sem:
            try:
                # Use METAR (Iowa State ASOS) as truth — matches Polymarket
                # resolution. Falls back to ERA5 for non-ASOS stations (HKO).
                df = await fetch_observed_truth(
                    station.to_location(),
                    station.station_id,
                    r.target_date, r.target_date,
                    agg=r.target,
                    source="metar",
                    client=client,
                )
            except Exception as exc:
                print(f"!! fetch {r.station_id} {r.target_date} [{r.target}]: {exc}")
                return False
        if df.empty or pd.isna(df.iloc[0]["observed_c"]):
            return False
        r.actual_obs_c = float(df.iloc[0]["observed_c"])
        r.resolved_at_utc = datetime.now(timezone.utc)

        # Cross-check: also fetch Polymarket's actual winning bucket if the
        # event has closed. Lets us measure the rounding rule + ERA5 gap.
        winner = await fetch_polymarket_winner(r.event_slug, client)
        if winner is not None:
            r.polymarket_won_bucket, r.polymarket_won_threshold = winner
        return True

    async with httpx.AsyncClient(timeout=60.0) as client:
        outcomes = await asyncio.gather(*(resolve_one(r, client) for r in pending))

    n_resolved = sum(1 for ok in outcomes if ok)
    if n_resolved:
        write_all_records(records, log_path)
    print(f"Resolved {n_resolved}/{len(pending)} records.")


if __name__ == "__main__":
    asyncio.run(main())
