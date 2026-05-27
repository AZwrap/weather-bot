"""Slim scan — lite rebuild entry point.

Replaces the 956-line `intraday_scan.py`. Runs three strategies only:

  1. METAR early-tail        — fire on monotonically-locked-in tail buckets
  2. Layer 7 guaranteed_no_buy — FAK NO on dead buckets (peak past edge)
  3. V2 conditional preposit — GTC NO maker, gated on consensus bucket
                                (paper-only by default; V2_ENABLED=False)

CLI:

  python slim_scan.py                      # dry-run; paper logs only
  python slim_scan.py --live               # live trading (requires LIVE_OK=1)
  python slim_scan.py --strategies metar   # subset
  python slim_scan.py --strategies metar,layer7,v2

KILL_SWITCH:
  Touch a file named KILL_SWITCH at the repo root to make any future
  scan exit 0 before touching the network.

This scanner intentionally does NOT carry the patches accumulated
during the 2026-05-15 → 2026-05-26 live run. It is a fresh
orchestrator. See plans/clever-riding-church.md for the rebuild plan
and project_decommission_2026-05-26.md for what was learned.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import httpx

from weather_bot.exclusions import load_active_exclusions
from weather_bot.fees import fetch_live_fee_config, warn_if_fee_config_changed
from weather_bot.guaranteed_no_buy import detect_and_execute_guaranteed_buys
from weather_bot.intraday import (
    DEFAULT_INTRADAY_LOG_PATH,
    IntradayDecision,
    already_decided,
    append_intraday_decision,
    find_early_tail_winner,
    load_intraday_log,
    metar_has_critical_gap,
)
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.observations import fetch_metar_hourly_today
from weather_bot.pnl import _rounded_observation
from weather_bot.polymarket import (
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    match_event_to_station,
    parse_bucket,
)
from weather_bot.portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio
from weather_bot.v2_conditional_preposit import (
    submit_v2_conditional_preposit_orders,
)

KILL_SWITCH = Path("KILL_SWITCH")
ALL_STRATEGIES = ("metar", "layer7", "v2")

# ── HARD PAPER-ONLY GATE ──────────────────────────────────────────────
# Force every strategy to paper mode (dry-run client) regardless of
# CLI flags or environment. Set to False ONLY after the publication-
# window shadow harness has accumulated enough data to justify live
# trading. While True, `--live` is ignored — the build_client factory
# always returns a dry-run client.
PAPER_ONLY: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slim scan — lite rebuild.")
    p.add_argument(
        "--live",
        action="store_true",
        help="Submit live orders. Requires LIVE_OK=1 in env as a second gate.",
    )
    p.add_argument(
        "--strategies",
        default=",".join(ALL_STRATEGIES),
        help=f"Comma-separated subset of {ALL_STRATEGIES}",
    )
    p.add_argument(
        "--portfolio-path",
        type=Path,
        default=DEFAULT_PORTFOLIO_PATH,
        help="Path to portfolio.json state file.",
    )
    return p.parse_args()


def build_client(live: bool):
    """Return an ExecutionClient. Dry-run unless --live AND LIVE_OK=1
    AND PAPER_ONLY is False. The PAPER_ONLY constant short-circuits
    everything below."""
    from weather_bot.execution.client import ExecutionClient
    from weather_bot.execution.safety import TradingConfig

    cfg = TradingConfig(enabled=live and not PAPER_ONLY)

    if PAPER_ONLY:
        if live:
            print(
                "[paper-only] --live requested but PAPER_ONLY=True is set in "
                "slim_scan.py. Falling back to dry-run client. Flip PAPER_ONLY "
                "to False to enable live submissions.",
                file=sys.stderr,
            )
        return ExecutionClient.dry_run(cfg)

    if not live:
        return ExecutionClient.dry_run(cfg)

    if os.environ.get("LIVE_OK") != "1":
        print(
            "[abort] --live requires LIVE_OK=1 in the environment. "
            "Refusing to submit real orders without the second gate.",
            file=sys.stderr,
        )
        sys.exit(2)

    return ExecutionClient.dry_run(cfg)  # placeholder — wire a real CLOB client when ready


def kill_switch_active() -> bool:
    if KILL_SWITCH.exists():
        print(f"[abort] KILL_SWITCH file present at {KILL_SWITCH.resolve()}")
        return True
    return False


async def run_scan(args: argparse.Namespace) -> int:
    strategies = {s.strip() for s in args.strategies.split(",") if s.strip()}
    unknown = strategies - set(ALL_STRATEGIES)
    if unknown:
        print(f"[abort] unknown strategies: {unknown}", file=sys.stderr)
        return 2

    if kill_switch_active():
        return 0

    client = build_client(args.live)
    portfolio = Portfolio.load(args.portfolio_path)
    today_utc = datetime.now(timezone.utc).date()
    excluded_pairs = load_active_exclusions(today_utc)
    excluded_stations = {sid for sid, _t in excluded_pairs}
    intraday_log = load_intraday_log(DEFAULT_INTRADAY_LOG_PATH)

    print(
        f"[scan] {datetime.now(timezone.utc).isoformat()} "
        f"strategies={sorted(strategies)} live={args.live} "
        f"dry_run={client.is_dry_run}"
    )

    async with httpx.AsyncClient(timeout=30.0) as http:
        # Cheap one-time live fee sanity check (cached 24h, polite).
        fee_cfg = await fetch_live_fee_config(http)
        if fee_cfg is not None:
            warn_if_fee_config_changed(fee_cfg)
            print(
                f"[fees] taker_rate={fee_cfg.taker_fee_rate:.4f} "
                f"rebate={fee_cfg.maker_rebate_rate} source={fee_cfg.source}"
            )

        events = await fetch_all_temperature_events(http)

        # Refresh CLOB top-of-book — gamma's bestBid/bestAsk are stale.
        all_tokens: list[str] = []
        for ev in events:
            for m in ev.markets:
                if m.yes_token_id:
                    all_tokens.append(m.yes_token_id)
        clob_prices = await fetch_clob_prices_batch(all_tokens, http)
        for ev in events:
            apply_clob_prices(ev, clob_prices)

        print(f"[scan] fetched {len(events)} events, {len(all_tokens)} tokens")

        v2_events: list = []  # built up for the V2 batch call after the per-event loop

        for ev in events:
            station = match_event_to_station(ev)
            if station is None:
                continue
            if station.station_id in excluded_stations:
                continue

            target = "max" if ev.target == "highest" else "min"
            target_date = event_target_date(ev, station)

            # Lite rebuild horizon: today + tomorrow only (matches original).
            if target_date > today_utc + timedelta(days=1):
                continue

            # METAR early-tail — only runs on the resolution-day event.
            if "metar" in strategies and target_date == today_utc:
                await run_metar_early_tail(
                    ev=ev,
                    station=station,
                    target=target,
                    target_date=target_date,
                    http=http,
                    intraday_log=intraday_log,
                )

            # Layer 7 — needs current observed extreme on resolution day.
            if "layer7" in strategies and target_date == today_utc:
                await run_layer7(
                    ev=ev,
                    station=station,
                    target=target,
                    target_date=target_date,
                    http=http,
                    client=client,
                    portfolio=portfolio,
                )

            v2_events.append(ev)

        # V2 preposit — batched call across all candidate events.
        if "v2" in strategies and v2_events:
            v2_counts = submit_v2_conditional_preposit_orders(
                events=v2_events,
                client=client,
                portfolio=portfolio,
                portfolio_path=args.portfolio_path,
                verbose=True,
            )
            print(f"[v2] {v2_counts}")

    portfolio.save(args.portfolio_path)
    print("[scan] done")
    return 0


async def run_metar_early_tail(
    *,
    ev,
    station,
    target: str,
    target_date,
    http: httpx.AsyncClient,
    intraday_log: list[IntradayDecision],
) -> None:
    """Lock-in YES detection. WUG primary, METAR fallback.

    WUG = Polymarket's oracle source. We always try it first; only if
    WUG returns no_data do we fall back to METAR + critical-gap check.
    """
    from weather_bot.wunderground import fetch_wunderground_daily

    sid = station.station_id
    target_date_iso = target_date.isoformat()

    if already_decided(sid, target, target_date_iso, intraday_log):
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    extreme_c: float | None = None
    source: str = ""
    n_observations: int = 0

    # ── Primary: WUG ──
    wug = await fetch_wunderground_daily(sid, target_date, client=http)
    wug_ext = wug.daily_max_c if target == "max" else wug.daily_min_c
    if wug_ext is not None:
        extreme_c = float(wug_ext)
        source = "wug"
        n_observations = wug.n_observations

    # ── Fallback: METAR ──
    if extreme_c is None:
        df = await fetch_metar_hourly_today(
            station.location, station.icao, target_date, client=http,
        )
        if df is None or df.empty:
            decision = IntradayDecision(
                scan_time_utc=now_iso, station_id=sid, target=target,
                target_date=target_date_iso, decision="NO_OBS",
                reason=f"WUG {wug.raw_status} + empty METAR fallback",
            )
            append_intraday_decision(decision)
            intraday_log.append(decision)
            return
        has_gap, gap_min, gap_at = metar_has_critical_gap(df, target)
        if has_gap:
            decision = IntradayDecision(
                scan_time_utc=now_iso, station_id=sid, target=target,
                target_date=target_date_iso, decision="CRITICAL_GAP",
                reason=f"WUG {wug.raw_status} + METAR gap {gap_min:.0f}min @ {gap_at}",
                n_observations_used=len(df),
            )
            append_intraday_decision(decision)
            intraday_log.append(decision)
            return
        extreme_c = float(
            df["temp_c"].max() if target == "max" else df["temp_c"].min()
        )
        source = "metar_fallback"
        n_observations = len(df)

    bucket_kinds_thresholds = [parse_bucket(m) for m in ev.markets]
    win = find_early_tail_winner(extreme_c, target, bucket_kinds_thresholds, station.unit)

    if win is None:
        decision = IntradayDecision(
            scan_time_utc=now_iso, station_id=sid, target=target,
            target_date=target_date_iso, decision="NO_TAIL_CROSSED",
            extreme_so_far_c=extreme_c, n_observations_used=n_observations,
            reason=f"source={source}",
        )
        append_intraday_decision(decision)
        intraday_log.append(decision)
        return

    win_kind, win_thr = win
    winning_market = next(
        (m for m in ev.markets if parse_bucket(m) == (win_kind, win_thr)),
        None,
    )

    decision = IntradayDecision(
        scan_time_utc=now_iso, station_id=sid, target=target,
        target_date=target_date_iso, decision="BUY_EARLY_TAIL",
        reason=(
            f"{source} observed {extreme_c:.1f}°C crossed {win_kind} threshold "
            f"{win_thr} (locked-in winner)"
        ),
        extreme_so_far_c=extreme_c, n_observations_used=n_observations,
        winning_bucket_kind=win_kind, winning_bucket_threshold=win_thr,
        winning_bucket_label=getattr(winning_market, "bucket_label", None),
        event_slug=ev.slug, event_id=ev.id,
        yes_token_id=getattr(winning_market, "yes_token_id", None),
        market_yes_ask=getattr(winning_market, "yes_ask", None),
        market_yes_bid=getattr(winning_market, "yes_bid", None),
    )
    append_intraday_decision(decision)
    intraday_log.append(decision)
    print(
        f"  [lock-in:{source}] {sid}/{target} {target_date_iso}: "
        f"BUY_EARLY_TAIL {win_kind} {win_thr} (extreme {extreme_c:.1f}°C)"
    )


async def run_layer7(
    *,
    ev,
    station,
    target: str,
    target_date,
    http: httpx.AsyncClient,
    client,
    portfolio: Portfolio,
) -> None:
    df = await fetch_metar_hourly_today(
        station.location, station.icao, target_date, client=http,
    )
    if df is None or df.empty:
        return
    extreme_c = float(df["temp_c"].max() if target == "max" else df["temp_c"].min())

    counts = detect_and_execute_guaranteed_buys(
        station_id=station.station_id,
        target_date_iso=target_date.isoformat(),
        observed_extreme_c=extreme_c,
        target=target,
        bucket_snapshots=list(ev.markets),
        client=client,
        portfolio=portfolio,
        verbose=True,
    )
    if counts:
        print(f"  [layer7] {station.station_id}/{target}: {dict(counts)}")


def main() -> int:
    args = parse_args()
    return asyncio.run(run_scan(args))


if __name__ == "__main__":
    sys.exit(main())
