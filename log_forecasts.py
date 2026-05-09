r"""Append a forward-log record per (station, target) for tomorrow's date.

Run this on a daily schedule (cron / Task Scheduler / Polymarket-aligned
trigger) to accumulate true 1-day-lead forecast data for skill validation.
The script is idempotent within a UTC day — re-running won't duplicate.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python log_forecasts.py

CLI flags:
    --log     path to JSONL log (default data/forward_log.jsonl)
    --bias    path to bias_table.json (default bias_table.json)
    --model   ensemble model (default ecmwf_ifs025)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
import numpy as np

from weather_bot.bias import BiasTable, predictive_members
from weather_bot.forecast.fetcher import fetch_ensemble
from weather_bot.forecast.probability import TempDistribution, bucket_prob
from weather_bot.forward_log import (
    DEFAULT_LOG_PATH,
    BucketSnapshot,
    ForwardLogRecord,
    append_record,
    existing_keys,
    existing_keys_hourly,
    load_records,
)
from weather_bot.locations import MARKETS, STATIONS_BY_ID
from weather_bot.polymarket import (
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    match_event_to_station,
    parse_bucket,
)
from weather_bot.unmatched import (
    DEFAULT_UNMATCHED_PATH,
    append_unmatched,
    make_unmatched_record,
)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    p.add_argument("--bias", default="bias_table.json")
    p.add_argument("--model", default="ecmwf_ifs025")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()

    bias_path = Path(args.bias)
    if not bias_path.exists():
        sys.exit(f"bias table not found at {bias_path}; run train_bias.py first.")
    bias_table = BiasTable.load(bias_path)

    log_path = Path(args.log)
    records = load_records(log_path)
    seen = existing_keys_hourly(records)
    now_utc = datetime.now(timezone.utc)
    issue_hour = now_utc.strftime("%Y%m%d%H")

    # Dedupe (station_id, target) pairs from the markets list
    pairs: list[tuple[str, str]] = []
    seen_pair: set[tuple[str, str]] = set()
    for sid, t in MARKETS:
        target = "max" if t == "highest" else "min"
        key = (sid, target)
        if key not in seen_pair:
            seen_pair.add(key)
            pairs.append(key)

    unique_station_ids = sorted({sid for sid, _ in pairs})
    print(
        f"Logging forecasts at {now_utc.isoformat()} for {len(unique_station_ids)} "
        f"stations × {len(pairs)} (station, target) pairs"
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def fetch_one(sid: str):
        station = STATIONS_BY_ID[sid]
        async with sem:
            try:
                fc = await fetch_ensemble(
                    station.to_location(), model=args.model, forecast_days=3
                )
                return sid, fc
            except Exception as exc:
                print(f"!! ensemble fetch {sid}: {exc}")
                return sid, None

    # Fetch ensembles and Polymarket events concurrently.
    async with httpx.AsyncClient(timeout=30.0) as gamma_client:
        events_task = asyncio.create_task(fetch_all_temperature_events(gamma_client))
        results = await asyncio.gather(*(fetch_one(s) for s in unique_station_ids))
        events = await events_task

        # Refresh prices via CLOB orderbook (more accurate than gamma's
        # cached bestBid/bestAsk; can drift 1-2¢ from real-time book).
        yes_tokens = [
            m.yes_token_id
            for ev in events for m in ev.markets
            if m.yes_token_id
        ]
        fresh = await fetch_clob_prices_batch(yes_tokens, gamma_client)
        n_upd, _ = apply_clob_prices(events, fresh)
        print(f"  CLOB-refreshed prices: {n_upd}/{len(yes_tokens)} markets updated")

    forecasts = {sid: fc for sid, fc in results if fc is not None}

    # Build (station_id, target, target_date) → event index for snapshotting.
    # Also collect unmatched events (Polymarket lists a city we don't have a
    # station for) so the user can triage and decide whether to add them.
    event_index: dict[tuple[str, str, "date"], object] = {}
    unmatched_records = []
    for ev in events:
        st = match_event_to_station(ev)
        if st is None:
            unmatched_records.append(make_unmatched_record(ev, now_utc))
            continue
        ev_target = "max" if ev.target == "highest" else "min"
        td = event_target_date(ev, st)
        existing = event_index.get((st.station_id, ev_target, td))
        if existing is None or ev.volume_24hr > existing.volume_24hr:
            event_index[(st.station_id, ev_target, td)] = ev
    print(f"  matched {len(event_index)} polymarket events to registry stations")
    if unmatched_records:
        n_appended = append_unmatched(unmatched_records, DEFAULT_UNMATCHED_PATH)
        cities = sorted({r.city_parsed or "?" for r in unmatched_records})
        print(
            f"  ⚠ {n_appended} unmatched events from {len(cities)} new "
            f"cit{'y' if len(cities) == 1 else 'ies'}: "
            f"{', '.join(cities[:8])}"
            f"{' …' if len(cities) > 8 else ''}"
        )

    n_added = 0
    n_skipped_dup = 0
    n_skipped_no_forecast = 0

    for sid, target in pairs:
        forecast = forecasts.get(sid)
        if forecast is None:
            n_skipped_no_forecast += 1
            continue

        station = STATIONS_BY_ID[sid]
        local_now = now_utc.astimezone(ZoneInfo(station.timezone))
        target_date = local_now.date() + timedelta(days=1)

        key = (sid, target, target_date, issue_hour)
        if key in seen:
            n_skipped_dup += 1
            continue

        try:
            if target == "max":
                raw_members = forecast.daily_max(target_date)
            else:
                raw_members = forecast.daily_min(target_date)
        except ValueError:
            n_skipped_no_forecast += 1
            continue

        bias_c = bias_table.get(sid, target)
        entry = bias_table.get_entry(sid, target)
        sigma_resid = entry.sigma_residual_c if entry else 0.0

        sigma_ens = float(np.std(raw_members - bias_c, ddof=1))
        sigma_total = float(np.sqrt(sigma_ens ** 2 + sigma_resid ** 2))

        inflated = predictive_members(
            raw_members, bias_table, sid, target, inflate_sigma=True
        )
        dist = TempDistribution(
            location_name=station.name, target_date=target_date, members=inflated
        )

        # Snapshot the matching Polymarket event's per-bucket prices.
        ev = event_index.get((sid, target, target_date))
        bucket_snaps: list[BucketSnapshot] | None = None
        event_slug: str | None = None
        event_id: int | None = None
        if ev is not None:
            event_slug = ev.slug
            event_id = ev.event_id
            bucket_snaps = []
            for m in ev.markets:
                kind, threshold = parse_bucket(m)
                p = bucket_prob(dist, kind, threshold, station.unit)
                bucket_snaps.append(BucketSnapshot(
                    bucket_label=m.bucket_label,
                    kind=kind,
                    threshold=threshold,
                    our_prob=p,
                    yes_bid=m.yes_bid,
                    yes_ask=m.yes_ask,
                    yes_last=m.last_trade_price,
                    volume_24hr=m.volume_24hr,
                    market_id=m.market_id,
                    yes_token_id=m.yes_token_id,
                    no_token_id=m.no_token_id,
                ))

        record = ForwardLogRecord(
            issue_time_utc=now_utc,
            target_date=target_date,
            station_id=sid,
            target=target,
            model=args.model,
            raw_members_c=raw_members.tolist(),
            bias_applied_c=bias_c,
            sigma_residual_c=sigma_resid,
            sigma_ensemble_c=sigma_ens,
            sigma_total_c=sigma_total,
            p10=float(np.quantile(inflated, 0.10)),
            p50=float(np.quantile(inflated, 0.50)),
            p90=float(np.quantile(inflated, 0.90)),
            event_slug=event_slug,
            event_id=event_id,
            bucket_snapshots=bucket_snaps,
        )
        append_record(record, log_path)
        n_added += 1

    print(
        f"Added {n_added} record(s) to {log_path}  "
        f"(skipped {n_skipped_dup} dup, {n_skipped_no_forecast} missing-forecast)"
    )


if __name__ == "__main__":
    asyncio.run(main())
