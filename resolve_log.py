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

    # Dedupe pending by (station_id, target, target_date) — multiple records
    # at different issue times on the same station-target-date all get the
    # SAME resolution. With same-day logging + 20-min cron, ~20-50 records
    # exist per (station, target, date), so deduping cuts METAR + Gamma
    # calls by ~20-50× and resolve_log runtime from ~30min to ~1min.
    # Added 2026-05-10.
    unique_keys: dict[tuple[str, str, "date"], list[ForwardLogRecord]] = {}
    for r in pending:
        key = (r.station_id, r.target, r.target_date)
        unique_keys.setdefault(key, []).append(r)
    print(
        f"  deduped to {len(unique_keys)} unique (station, target, date) "
        f"resolutions ({len(pending)}/{len(unique_keys):.1f}× duplication)"
    )

    async def resolve_group(
        key: tuple[str, str, "date"],
        records_for_key: list[ForwardLogRecord],
        client: httpx.AsyncClient,
    ) -> int:
        """Resolve all records sharing a (station, target, date) with ONE
        METAR fetch + ONE Gamma fetch. Returns count of records updated."""
        sid, tgt, td = key
        station = STATIONS_BY_ID.get(sid)
        if station is None:
            print(f"!! unknown station_id {sid} — skipping {len(records_for_key)} records")
            return 0
        async with sem:
            try:
                df = await fetch_observed_truth(
                    station.to_location(),
                    station.station_id,
                    td, td,
                    agg=tgt,
                    source="metar",
                    client=client,
                )
            except Exception as exc:
                print(f"!! fetch {sid} {td} [{tgt}]: {exc}")
                return 0
        if df.empty or pd.isna(df.iloc[0]["observed_c"]):
            return 0
        actual = float(df.iloc[0]["observed_c"])
        now = datetime.now(timezone.utc)

        # Polymarket gamma: one call per unique event slug (deduped within group).
        unique_slugs = {r.event_slug for r in records_for_key if r.event_slug}
        winners_by_slug: dict[str, tuple[str, int]] = {}
        for slug in unique_slugs:
            w = await fetch_polymarket_winner(slug, client)
            if w is not None:
                winners_by_slug[slug] = w

        # Apply to all records in the group
        for r in records_for_key:
            r.actual_obs_c = actual
            r.resolved_at_utc = now
            if r.event_slug and r.event_slug in winners_by_slug:
                r.polymarket_won_bucket, r.polymarket_won_threshold = (
                    winners_by_slug[r.event_slug]
                )
        return len(records_for_key)

    async with httpx.AsyncClient(timeout=60.0) as client:
        per_group_counts = await asyncio.gather(
            *(resolve_group(k, rs, client) for k, rs in unique_keys.items())
        )

    n_resolved = sum(per_group_counts)
    if n_resolved:
        write_all_records(records, log_path)
    print(f"Resolved {n_resolved}/{len(pending)} records "
          f"({len(unique_keys)} unique station-date groups).")


if __name__ == "__main__":
    asyncio.run(main())
