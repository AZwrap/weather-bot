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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
import numpy as np

from weather_bot.bias import BiasTable, predictive_members
from weather_bot.forecast.fetcher import (
    EnsembleForecast,
    ecmwf_run_init_utc,
    fetch_ensemble,
    fetch_multimodel_deterministic,
)
from weather_bot.forecast.probability import TempDistribution, bucket_prob
from weather_bot.forward_log import (
    DEFAULT_LOG_PATH,
    BucketSnapshot,
    ForwardLogRecord,
    append_record,
    existing_keys,
    existing_keys_hourly,
    existing_keys_slot,
    load_records,
    slot_id_for,
)
from weather_bot.locations import MARKETS, STATIONS_BY_ID, Station
from weather_bot.multimodel_cache import (
    DEFAULT_CACHE_PATH as DEFAULT_MM_CACHE_PATH,
    get_fresh as mm_cache_get,
    load_cache as mm_cache_load,
    put as mm_cache_put,
    save_cache as mm_cache_save,
)
from weather_bot.polymarket import (
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    fetch_orderbook_depths_batch,
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
    p.add_argument("--slot-minutes", type=int, default=20,
                   help="dedup slot size in minutes (default 20, matching the "
                        "*/20 cron). Set to 60 for the legacy hourly behaviour.")
    p.add_argument("--multimodel-cache", default=str(DEFAULT_MM_CACHE_PATH),
                   help="path to the multi-model OM cache (avoids refetching "
                        "between model runs). Pass empty string to disable.")
    p.add_argument("--ensemble-cache",
                   default="data/ensemble_cache.json",
                   help="path to the ECMWF ensemble cache (avoids refetching "
                        "the ensemble between ECMWF model runs ~every 6h). "
                        "Pass empty string to disable.")
    p.add_argument("--skip-forecasts", action="store_true", default=True,
                   help="Skip Open-Meteo ECMWF ensemble + multimodel "
                        "deterministic fetches entirely. The bot does NOT "
                        "use forecasts for live decisions (per standing "
                        "rule), so this avoids burning Open-Meteo API budget "
                        "on unused data. Records still get written with "
                        "empty raw_members_c so resolve_log + "
                        "poll_resolutions can still match position resolutions. "
                        "Use --no-skip-forecasts to re-enable (e.g., for "
                        "research / backtest data collection).")
    p.add_argument("--no-skip-forecasts", dest="skip_forecasts",
                   action="store_false",
                   help="Re-enable Open-Meteo forecast fetches.")
    p.add_argument("--no-capture-depth", action="store_true",
                   help="Disable top-of-book depth capture in BucketSnapshot. "
                        "By default (since 2026-05-13), every paper scan also "
                        "fetches /book for RESOLUTION-DAY events to populate "
                        "top_yes_ask_size / top_yes_bid_size — gives us "
                        "calibration data for the live --depth-aware-metar "
                        "synthetic ladder. Disable if Polymarket /book becomes "
                        "rate-limited or for debugging. Tomorrow's-events "
                        "snapshots never fetch depth (METAR can't fire yet).")
    args = p.parse_args()

    bias_path = Path(args.bias)
    if not bias_path.exists():
        sys.exit(f"bias table not found at {bias_path}; run train_bias.py first.")
    bias_table = BiasTable.load(bias_path)

    log_path = Path(args.log)
    records = load_records(log_path)
    seen = existing_keys_slot(records, args.slot_minutes)
    now_utc = datetime.now(timezone.utc)
    issue_slot = slot_id_for(now_utc, args.slot_minutes)

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

    # ── ECMWF ensemble cache (added 2026-05-16 audit) ────────────────────
    # ECMWF runs every 6h with ~5h availability lag; the */20 cron used to
    # re-fetch on every tick (~3,528 calls/day) and trigger Open-Meteo
    # 429s. With this cache, 49 stations × 4 ECMWF runs/day = 196 fetches
    # — well under the 10k/day non-commercial limit.
    from weather_bot.ensemble_cache import (
        load_cache as ec_load,
        save_cache as ec_save,
        get_fresh as ec_get_fresh,
        put as ec_put,
    )
    from weather_bot.forecast.fetcher import ecmwf_run_init_utc
    ensemble_cache_path = Path(args.ensemble_cache) if args.ensemble_cache else None
    ensemble_cache = ec_load(ensemble_cache_path) if ensemble_cache_path else {}
    current_ecmwf_run = ecmwf_run_init_utc(now_utc)
    ec_stats = {"hit": 0, "miss": 0, "skipped": 0}

    async def fetch_one(sid: str):
        # When --skip-forecasts is set (default per the standing "no
        # forecast logic" rule), short-circuit without hitting Open-Meteo
        # at all. Returns None so downstream record-writing uses empty
        # raw_members_c (a separate code branch handles this gracefully).
        if args.skip_forecasts:
            ec_stats["skipped"] += 1
            return sid, None
        station = STATIONS_BY_ID[sid]
        # Try cache first — only re-fetch when the current ECMWF run-init
        # has advanced past what we have cached.
        if ensemble_cache_path:
            cached = ec_get_fresh(
                ensemble_cache, sid, current_ecmwf_run, args.model,
            )
            if cached is not None:
                ec_stats["hit"] += 1
                return sid, cached
        async with sem:
            try:
                fc = await fetch_ensemble(
                    station.to_location(), model=args.model, forecast_days=3
                )
                ec_stats["miss"] += 1
                if ensemble_cache_path:
                    ec_put(ensemble_cache, sid, fc, current_ecmwf_run, now_utc)
                return sid, fc
            except Exception as exc:
                print(f"!! ensemble fetch {sid}: {exc}")
                return sid, None

    async def fetch_multimodel_one(sid: str, target_date):
        """Fetch deterministic multi-model forecasts for one station+date.
        Failures return None — multi-model is auxiliary data, not critical."""
        station = STATIONS_BY_ID[sid]
        async with sem:
            try:
                return sid, await fetch_multimodel_deterministic(
                    station.to_location(), target_date,
                )
            except Exception as exc:
                print(f"!! multimodel fetch {sid}: {exc}")
                return sid, None

    # Fetch ensembles and Polymarket events concurrently.
    # HTTP/2 + connection pooling eliminates the ~54ms TLS handshake on every
    # subsequent request to the same host (Polymarket / Open-Meteo / Iowa
    # State / etc). Pool config from weather_bot/http_clients.py — when
    # the daemon (Layer 1) ships, this block will switch to using the
    # shared module's `get_http_client()` directly. For cron-based execution
    # the per-block lifetime is fine (process exits after the scan anyway).
    from weather_bot.http_clients import make_local_client
    async with make_local_client() as gamma_client:
        events_task = asyncio.create_task(fetch_all_temperature_events(gamma_client))
        results = await asyncio.gather(*(fetch_one(s) for s in unique_station_ids))
        events = await events_task

        # Persist ensemble cache (added 2026-05-16). Only writes if something
        # was added; pure-cache-hit ticks leave the file untouched.
        if ensemble_cache_path and ec_stats["miss"] > 0:
            ec_save(ensemble_cache, ensemble_cache_path)
        print(
            f"  ensemble cache (run init {current_ecmwf_run.strftime('%Y-%m-%d %HZ')}): "
            f"{ec_stats['hit']} hit, {ec_stats['miss']} need refresh"
        )

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

        # ── Depth capture (RESOLUTION-DAY only) ─────────────────────────
        # Restrict /book fetches to events resolving today (station-local)
        # to bound API calls: tomorrow's events can't fire METAR yet, so
        # depth there doesn't help calibrate the ladder. Even with this
        # filter, expect ~30-50 events × ~11 buckets = ~300-550 /book
        # calls per scan. Concurrency-6 keeps each scan under 90s.
        depth_by_token: dict[str, "OrderBookDepth | None"] = {}  # noqa: F821
        if not args.no_capture_depth:
            today_station_local = {
                sid: datetime.now(ZoneInfo(STATIONS_BY_ID[sid].timezone)).date()
                for sid in unique_station_ids
            }
            resday_tokens: list[str] = []
            skipped_no_live_market = 0
            for ev in events:
                station = match_event_to_station(ev)
                if station is None:
                    continue
                ev_target_date = event_target_date(ev, station)
                if ev_target_date != today_station_local.get(station.station_id):
                    continue
                for m in ev.markets:
                    if not m.yes_token_id:
                        continue
                    # Filter out tokens with no live market (yes_bid/yes_ask
                    # both None after apply_clob_prices). Catches some but
                    # not all dead tokens — gamma often caches non-None
                    # prices for resolved markets.
                    if m.yes_bid is None and m.yes_ask is None:
                        skipped_no_live_market += 1
                        continue
                    resday_tokens.append(m.yes_token_id)

            # Persistent dead-token cache: skip tokens that returned 404
            # from /book in the last 24h. Audit 2026-05-16 found the
            # bid/ask filter above only caught ~3 of ~170 dead tokens
            # per tick because gamma's cached prices are stale-but-not-
            # None. The disk cache catches the rest.
            from weather_bot.dead_token_cache import (
                load_cache as dt_load,
                save_cache as dt_save,
                filter_live_tokens as dt_filter,
                mark_dead as dt_mark,
                prune_expired as dt_prune,
                DEFAULT_CACHE_PATH as DT_DEFAULT,
            )
            dt_cache = dt_load(DT_DEFAULT)
            n_pruned = dt_prune(dt_cache)
            live_tokens, n_cached_skip = dt_filter(dt_cache, resday_tokens)

            if live_tokens:
                msg = f"  fetching /book depth for {len(live_tokens)} markets"
                if skipped_no_live_market or n_cached_skip:
                    msg += (
                        f" (skipped {skipped_no_live_market} no-bid/ask"
                        f" + {n_cached_skip} cached-dead"
                        f"{f', pruned {n_pruned} expired' if n_pruned else ''})"
                    )
                print(msg + "...")
                depth_by_token = await fetch_orderbook_depths_batch(
                    live_tokens, gamma_client, concurrency=6,
                )
                # Pre-populate None entries for the skipped (cached-dead)
                # tokens so downstream code that iterates them gets None
                # consistently.
                for tid in resday_tokens:
                    if tid not in depth_by_token:
                        depth_by_token[tid] = None
                # Update dead-token cache with any 404s from this tick
                n_new_dead = 0
                for tid, depth in depth_by_token.items():
                    if depth is None and tid in live_tokens:
                        dt_mark(dt_cache, tid)
                        n_new_dead += 1
                if n_new_dead or n_pruned:
                    dt_save(dt_cache, DT_DEFAULT)
                n_ok = sum(1 for d in depth_by_token.values() if d is not None)
                print(f"  got depth for {n_ok}/{len(live_tokens)} live markets "
                      f"({n_new_dead} new dead-token cache entries)")

    forecasts = {sid: fc for sid, fc in results if fc is not None}

    # Compute per-station target_dates: BOTH today and tomorrow station-local.
    # `today` enables resolution-day market-price observation (required for
    # METAR-feedback strategy and convergence-exit observability). Without
    # this, the bot has no market data on the day each market resolves.
    # Added 2026-05-10.
    target_dates_per_station: dict[str, list["date"]] = {}
    for sid in unique_station_ids:
        station = STATIONS_BY_ID[sid]
        local_now = now_utc.astimezone(ZoneInfo(station.timezone))
        today = local_now.date()
        target_dates_per_station[sid] = [today, today + timedelta(days=1)]

    # Fetch multi-model deterministic forecasts (for future bias retraining).
    # ONE API call per station covers BOTH today + tomorrow via date-range
    # mode (2026-05-10). Run-aware cache (also 2026-05-10) reuses the
    # payload between upstream model runs — Open-Meteo returns identical
    # values until the next ECMWF/GFS/ICON cycle (~6h), so refetching every
    # 20 min was burning ~17 of every 18 calls on unchanged data.
    cache_path = Path(args.multimodel_cache) if args.multimodel_cache else None
    mm_cache = mm_cache_load(cache_path) if cache_path else {}
    current_run_init = ecmwf_run_init_utc(now_utc)

    stations_needing_fetch: list[str] = []
    cached_payloads: dict[str, dict] = {}
    for sid in forecasts.keys():
        cached = (
            mm_cache_get(mm_cache, sid, current_run_init, target_dates_per_station[sid])
            if cache_path else None
        )
        if cached is not None:
            cached_payloads[sid] = cached
        else:
            stations_needing_fetch.append(sid)

    print(
        f"  multimodel cache (run init {current_run_init.strftime('%Y-%m-%d %HZ')}): "
        f"{len(cached_payloads)} hit, {len(stations_needing_fetch)} need refresh"
    )

    async def fetch_mm_for_station(sid: str):
        station = STATIONS_BY_ID[sid]
        dates = target_dates_per_station[sid]
        # dates is [today, today+1]. Pass as range to fetch both in one call.
        start_d, end_d = dates[0], dates[-1]
        async with sem:
            try:
                mm = await fetch_multimodel_deterministic(
                    station.to_location(), start_d, end_date=end_d,
                )
                return sid, mm
            except Exception as exc:
                print(f"!! multimodel fetch {sid}: {exc}")
                return sid, None

    mm_results: list = []
    if stations_needing_fetch:
        print(
            f"  fetching multi-model deterministic forecasts "
            f"({len(stations_needing_fetch)} stations, today+tomorrow per call)..."
        )
        mm_results = await asyncio.gather(
            *(fetch_mm_for_station(sid) for sid in stations_needing_fetch),
            return_exceptions=True,
        )
        if cache_path:
            for r in mm_results:
                if isinstance(r, Exception):
                    continue
                sid, mm_data = r
                if mm_data:
                    mm_cache_put(
                        mm_cache, sid, current_run_init,
                        target_dates_per_station[sid], mm_data, now_utc,
                    )
            mm_cache_save(mm_cache, cache_path)

    # Nested map: multimodel[sid][target_date_iso] = {model: {max/min: temp}}
    # Range-mode response is dict[model, dict[date_iso, {max, min}]] —
    # invert to dict[date_iso, dict[model, {max, min}]] for log_forecasts'
    # per-record lookup.
    multimodel: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    n_with_mm = 0

    def _ingest(sid: str, mm_data: dict | None) -> None:
        nonlocal n_with_mm
        if not mm_data:
            return
        per_date: dict[str, dict[str, dict[str, float]]] = {}
        for model, by_date in mm_data.items():
            for date_iso, vals in by_date.items():
                per_date.setdefault(date_iso, {})[model] = vals
        if per_date:
            multimodel[sid] = per_date
            n_with_mm += len(per_date)

    for r in mm_results:
        if isinstance(r, Exception):
            continue
        sid, mm_data = r
        _ingest(sid, mm_data)
    for sid, mm_data in cached_payloads.items():
        _ingest(sid, mm_data)

    print(
        f"  multi-model returned data for "
        f"{n_with_mm}/{len(forecasts) * 2} (station, date) pairs "
        f"({len(stations_needing_fetch)} fresh API calls + "
        f"{len(cached_payloads)} cache hits)"
    )

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
        if forecast is None and not args.skip_forecasts:
            # Forecast was attempted but failed — skip the record.
            n_skipped_no_forecast += 1
            continue
        # With --skip-forecasts, forecast is intentionally None: we
        # still write a market-state-only record (CLOB prices + depth)
        # so resolve_log + poll_resolutions can resolve filled positions.

        station = STATIONS_BY_ID[sid]
        # Iterate over BOTH today (resolution day) and tomorrow (1-day-lead).
        # Today gives us market-price observability throughout the resolution
        # day — required for METAR-feedback strategy and convergence-exit
        # backtesting. Tomorrow continues the existing 1-day-lead bias
        # validation pipeline.
        for target_date in target_dates_per_station[sid]:
            ok = _log_one_record(
                forecast, station, target, target_date,
                issue_slot, now_utc, seen, log_path,
                bias_table, multimodel, event_index, args.model,
                depth_by_token,
            )
            if ok == "added":
                n_added += 1
            elif ok == "dup":
                n_skipped_dup += 1
            elif ok == "no_forecast":
                n_skipped_no_forecast += 1

    print(
        f"Added {n_added} record(s) to {log_path}  "
        f"(skipped {n_skipped_dup} dup, {n_skipped_no_forecast} missing-forecast)"
    )


def _log_one_record(
    forecast: "EnsembleForecast | None",
    station: Station,
    target: Literal["max", "min"],
    target_date: date,
    issue_slot: str,
    now_utc: datetime,
    seen: set[tuple[str, str, date, str]],
    log_path: Path,
    bias_table: BiasTable,
    multimodel: dict[str, dict[str, dict[str, dict[str, float]]]],
    event_index: dict[tuple[str, str, date], Any],
    ensemble_model: str,
    depth_by_token: dict[str, Any],
) -> str:
    """Log a single (station, target, target_date) record.

    When `forecast` is None (--skip-forecasts mode), writes a
    "market-state-only" record with empty raw_members_c + zero
    forecast stats. The bucket_snaps section still captures CLOB
    prices + depth, which is what resolve_log + poll_resolutions
    need to match positions to resolutions.

    Returns one of "added", "dup", "no_forecast" so the caller can
    increment the right counter.
    """
    sid = station.station_id
    key = (sid, target, target_date, issue_slot)
    if key in seen:
        return "dup"

    if forecast is None:
        # --skip-forecasts: market-state-only record
        raw_members = np.array([], dtype=float)
        bias_c = 0.0
        sigma_resid = 0.0
        sigma_ens = 0.0
        sigma_total = 0.0
        inflated = np.array([], dtype=float)
    else:
        try:
            if target == "max":
                raw_members = forecast.daily_max(target_date)
            else:
                raw_members = forecast.daily_min(target_date)
        except ValueError:
            return "no_forecast"

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
        # With empty forecast distribution (--skip-forecasts), our_prob
        # cannot be computed — record 0.0 placeholder. Market-state-only
        # records consumers (resolve_log, poll_resolutions) don't read
        # this field anyway.
        no_dist = len(inflated) == 0
        for m in ev.markets:
            kind, threshold = parse_bucket(m)
            p = 0.0 if no_dist else bucket_prob(dist, kind, threshold, station.unit)
            # Depth-of-book + per-market tick size (resolution-day only;
            # None otherwise). Captures top 5 levels per side so we can
            # walk a real ladder in --depth-aware-metar instead of the
            # synthetic decay model.
            top_ask_sz: float | None = None
            top_bid_sz: float | None = None
            ask_levels: list[list[float]] | None = None
            bid_levels: list[list[float]] | None = None
            tick_size_val: float | None = None
            depth = depth_by_token.get(m.yes_token_id) if depth_by_token else None
            if depth is not None:
                tick_size_val = float(depth.tick_size)
                if depth.asks:
                    top_ask_sz = depth.asks[0].size_shares
                    ask_levels = [
                        [float(lv.price), float(lv.size_shares)]
                        for lv in depth.asks[:5]
                    ]
                if depth.bids:
                    top_bid_sz = depth.bids[0].size_shares
                    bid_levels = [
                        [float(lv.price), float(lv.size_shares)]
                        for lv in depth.bids[:5]
                    ]
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
                top_yes_ask_size=top_ask_sz,
                top_yes_bid_size=top_bid_sz,
                yes_ask_levels=ask_levels,
                yes_bid_levels=bid_levels,
                tick_size=tick_size_val,
            ))

    # Extract per-model deterministic prediction for THIS record's target.
    # multimodel[sid][target_date.isoformat()] is dict like
    #   {"icon_seamless": {"max": 24.5, "min": 12.3}, ...}
    mm_for_record: dict[str, float] | None = None
    td_iso = target_date.isoformat()
    if sid in multimodel and td_iso in multimodel[sid]:
        mm_for_record = {}
        for model_id, vals in multimodel[sid][td_iso].items():
            if target in vals:
                mm_for_record[model_id] = vals[target]
        if not mm_for_record:
            mm_for_record = None  # nothing usable for this target

    # Quantile of empty array would raise; placeholder zeros for
    # market-state-only records (--skip-forecasts mode).
    if len(inflated) > 0:
        p10 = float(np.quantile(inflated, 0.10))
        p50 = float(np.quantile(inflated, 0.50))
        p90 = float(np.quantile(inflated, 0.90))
    else:
        p10 = p50 = p90 = 0.0

    record = ForwardLogRecord(
        issue_time_utc=now_utc,
        target_date=target_date,
        station_id=sid,
        target=target,
        model=ensemble_model,
        raw_members_c=raw_members.tolist(),
        bias_applied_c=bias_c,
        sigma_residual_c=sigma_resid,
        sigma_ensemble_c=sigma_ens,
        sigma_total_c=sigma_total,
        p10=p10,
        p50=p50,
        p90=p90,
        event_slug=event_slug,
        event_id=event_id,
        bucket_snapshots=bucket_snaps,
        multimodel_forecasts_c=mm_for_record,
        ecmwf_run_init_utc=ecmwf_run_init_utc(now_utc),
    )
    append_record(record, log_path)
    return "added"


if __name__ == "__main__":
    asyncio.run(main())
