"""Slim daemon — long-running event-driven bot.

Replaces the 15-min cron with a single long-running process. Three
background sources drive strategy evaluation:

  1. **WUG pollers** (one per active station-target-date, 60s cadence).
     Emit a `WUGUpdate` event when the daily extreme moves outward.
     Each event triggers Layer 7 + high-bucket NO + lock-in-YES for that
     (station, target, date).

  2. **CLOB WebSocket** (single connection across all NO tokens we care
     about). Maintains a `BookCache` so strategies read sub-second-fresh
     prices instead of REST snapshots that can be 1-30s stale.

  3. **Events refresh** (every 5 min). Re-fetches gamma + CLOB prices,
     re-registers WUG pollers for new (station, date) tuples, prunes
     resolved tuples. Also re-fetches the live Polymarket fee config
     and runs V2 conditional preposit (paper-only) on each event.

KILL_SWITCH and PAPER_ONLY semantics from slim_scan.py apply.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import httpx

from weather_bot.basket_sweep import log_basket_sweep
from weather_bot.consensus_basket import detect_and_execute_consensus_basket
from weather_bot.consensus_yes import (
    detect_and_execute_consensus_yes,
    evaluate_consensus_yes_exits,
    evaluate_single_consensus_yes_exit,
)
from weather_bot.consistency_arb import detect_and_execute_consistency_arb
from weather_bot.daily_resolver import resolve_settled_events
from weather_bot.exclusions import load_active_exclusions
from weather_bot.fees import fetch_live_fee_config, warn_if_fee_config_changed
from weather_bot.guaranteed_no_buy import detect_and_execute_guaranteed_buys
from weather_bot.high_bucket_no import detect_and_execute_high_bucket_no
from weather_bot.persistence_tail import (
    _load_yesterday_actuals,
    detect_and_execute_persistence_tail,
)
from weather_bot.intraday import (
    DEFAULT_INTRADAY_LOG_PATH,
    IntradayDecision,
    already_decided,
    append_intraday_decision,
    find_early_tail_winner,
    load_intraday_log,
)
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.polymarket import (
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    match_event_to_station,
    parse_bucket,
)
from weather_bot.polymarket_ws import BookCache, subscribe_and_watch
from weather_bot.portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio
from weather_bot.publication_window import midend_local_utc, snapshot_one
from weather_bot.v2_conditional_preposit import (
    V2_ENABLED,
    submit_v2_conditional_preposit_orders,
)
from weather_bot.wug_poller import WUGPollerPool, WUGUpdate

KILL_SWITCH = Path("KILL_SWITCH")

# Hard paper-only gate — mirrors slim_scan.PAPER_ONLY. Flip to False
# only after the publication-window harness validates strategies.
PAPER_ONLY: bool = True

# consensus_yes DISABLED 2026-05-31 — paper control returned its verdict:
# −$312 over 304 trades @ 24% win (the NO_momentum spread-bleed again).
# Entry + both exit paths are gated off; logs retained. The 5 open
# positions just ride to resolution. Flip True to re-enable.
CONSENSUS_YES_ENABLED: bool = False

# Layer 7 (guaranteed_no_buy) DISABLED 2026-06-02 by operator request.
# Post-decommission data had it at ~92% redeem vs ~97% breakeven at the
# $0.98 cap — a structural bleeder. Both call sites are gated on this flag
# (code + logs retained, tab dropped from the dashboard) so a single flip
# re-enables it without re-adding the calls.
LAYER7_ENABLED: bool = False

# Consistency arb DISABLED 2026-06-02 by operator request. After hardening
# (REST depth + fees + empty-book/implausible-margin artifact filters), real
# net-of-fee executable arbs ran at ~0 — the old headline was artifacts +
# gross/top-of-book optimism. Call site gated on this flag; code + logs
# retained, dashboard tab dropped. Flip True to re-enable.
CONSISTENCY_ARB_ENABLED: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--wug-poll-interval", type=float, default=60.0,
        help="Seconds between WUG polls per station-date. Default 60s.",
    )
    p.add_argument(
        "--events-refresh-interval", type=float, default=300.0,
        help="Seconds between gamma events refresh. Default 300s (5 min).",
    )
    p.add_argument(
        "--portfolio-path", type=Path, default=DEFAULT_PORTFOLIO_PATH,
    )
    p.add_argument(
        "--live", action="store_true",
        help="Submit live orders. Ignored while PAPER_ONLY=True.",
    )
    return p.parse_args()


# ── Shared state ─────────────────────────────────────────────────────
class DaemonState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.http: httpx.AsyncClient | None = None
        self.book_cache = BookCache()
        self.portfolio = Portfolio.load(args.portfolio_path)
        self.events: list = []
        self.events_by_sk: dict[tuple[str, str, str], object] = {}
        self.stations_by_sk: dict[tuple[str, str, str], object] = {}
        # token_id -> sk, for the WS-delta basket trigger to resolve which
        # event a price_change belongs to (rebuilt each refresh).
        self.sk_by_token: dict[str, tuple[str, str, str]] = {}
        # In-memory high-water penny per sk (basket WS-trigger dedupe):
        # only spawn a fire task when the leader crosses a NEW penny.
        self.basket_hw: dict[tuple[str, str, str], int] = {}
        # Guard against unbounded concurrent fire tasks.
        self.basket_fire_inflight: set[tuple[str, str, str]] = set()
        # Current leading bucket per sk as (bucket_label, peak_ask_while_leading,
        # became_at_ts) — used to log leader FLIPS (data/leader_flips.jsonl),
        # classify "late" flips (a ≥0.90 leader dethroned), and supply the
        # leader-dwell (stability-gate) field to the basket sweep.
        self.current_leader: dict[tuple[str, str, str], tuple[str, float, float]] = {}
        self.client = None  # ExecutionClient — see build_client()
        self.intraday_log = load_intraday_log(DEFAULT_INTRADAY_LOG_PATH)
        # Cached prior-day actuals for persistence_tail. Refreshed on
        # each events refresh tick from data/forward_log.jsonl.
        self.yesterday_actuals: dict[tuple[str, str], float] = {}
        # Latest WUG-observed extreme per (sk) — fed to the fast NO-side
        # sweep so it can re-run Layer7/high-bucket-NO between WUG ticks.
        self.latest_extreme: dict[tuple[str, str, str], float] = {}
        self.shutdown_event = asyncio.Event()
        self.ws_task: asyncio.Task | None = None
        self.wug_pool: WUGPollerPool | None = None
        self.resolver_task: asyncio.Task | None = None
        self.no_side_sweep_task: asyncio.Task | None = None


def build_client(live: bool):
    """Always dry-run while PAPER_ONLY=True. Mirrors slim_scan logic."""
    from weather_bot.execution.client import ExecutionClient
    from weather_bot.execution.safety import TradingConfig

    cfg = TradingConfig(enabled=live and not PAPER_ONLY)
    if PAPER_ONLY:
        if live:
            print(
                "[paper-only] --live ignored: PAPER_ONLY=True in slim_daemon.py",
                file=sys.stderr,
            )
        return ExecutionClient.dry_run(cfg)
    if not live:
        return ExecutionClient.dry_run(cfg)
    if os.environ.get("LIVE_OK") != "1":
        print("[abort] --live requires LIVE_OK=1", file=sys.stderr)
        sys.exit(2)
    return ExecutionClient.dry_run(cfg)


def kill_switch_active() -> bool:
    return KILL_SWITCH.exists()


# ── Strategy evaluators ──────────────────────────────────────────────
async def on_wug_update(state: DaemonState, upd: WUGUpdate) -> None:
    """Single entry point for every WUG-driven strategy.

    Runs (in order):
      1. Lock-in YES (was METAR early-tail; now WUG-driven)
      2. Layer 7 progressive dead-bucket NO
      3. High-bucket NO (fires only past trigger-local-hour)
    """
    if kill_switch_active():
        return

    sk = (upd.station_id, upd.target, upd.target_date_iso)
    ev = state.events_by_sk.get(sk)
    if ev is None:
        return
    station = state.stations_by_sk.get(sk)
    if station is None:
        return

    # Remember the latest observed extreme so the fast NO-side sweep can
    # re-fire Layer 7 / high-bucket-NO between WUG ticks (≤10s) instead
    # of only on the ≤60s WUG poll.
    state.latest_extreme[sk] = upd.observed_extreme_c

    print(
        f"[wug] {upd.station_id}/{upd.target} {upd.target_date_iso}  "
        f"obs={upd.observed_extreme_c:.1f}°C (int={upd.observed_int}, "
        f"prev={upd.previous_int}, n={upd.n_observations})"
    )

    # DEPTH PRE-FETCH. on_wug_update only fires on OUTWARD extreme moves
    # (not every poll tick), so fetching the event's full order books
    # here is bounded. The depth_map feeds the taker strategies so paper
    # fills reflect the depth-walked average, not an optimistic
    # top-of-book full fill. Fetch both NO and YES tokens (NO strategies
    # + consensus_yes). Best-effort: on failure depth_map is empty and
    # strategies record skipped_no_depth.
    depth_map: dict = {}
    try:
        from weather_bot.polymarket import fetch_orderbook_depths_batch
        tokens = []
        for m in ev.markets:
            if m.no_token_id:
                tokens.append(m.no_token_id)
            if m.yes_token_id:
                tokens.append(m.yes_token_id)
        if tokens:
            depth_map = await fetch_orderbook_depths_batch(tokens, state.http)
            # Seed the WS cache with this REST depth so the 5s sweep can
            # depth-walk this just-moved event for ~90s without re-fetching
            # (Polymarket WS doesn't snapshot on subscribe → ~2% WS-fresh,
            # so this is what actually feeds the sweep's depth-walk).
            for tok, d in depth_map.items():
                if d is not None:
                    state.book_cache.seed_depth(tok, d)
    except Exception as exc:
        print(f"  [depth] fetch failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 1) Lock-in YES — if a tail bucket just locked in, fire YES on it.
    try:
        await _run_lockin_yes(state, station, ev, upd)
    except Exception as exc:
        print(f"  [lock-in] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 2) Layer 7 progressive — gated on LAYER7_ENABLED (disabled). When off,
    # `counts` is falsy and the print guard below short-circuits.
    try:
        counts = LAYER7_ENABLED and detect_and_execute_guaranteed_buys(
            station_id=upd.station_id,
            target_date_iso=upd.target_date_iso,
            observed_extreme_c=upd.observed_extreme_c,
            target=upd.target,
            bucket_snapshots=list(ev.markets),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
            book_cache=state.book_cache,
            depth_map=depth_map,
            verbose=False,
        )
        if counts and counts.get("placed", 0) > 0:
            print(f"  [layer7] placed={counts['placed']} "
                  f"other={ {k: v for k, v in counts.items() if k != 'placed'} }")
    except Exception as exc:
        print(f"  [layer7] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 3) High-bucket NO (trigger-time-gated)
    try:
        counts = detect_and_execute_high_bucket_no(
            station_id=upd.station_id,
            target_date_iso=upd.target_date_iso,
            observed_extreme_c=upd.observed_extreme_c,
            target=upd.target,
            bucket_snapshots=list(ev.markets),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
            book_cache=state.book_cache,
            depth_map=depth_map,
            verbose=False,
        )
        if counts and counts.get("placed", 0) > 0:
            print(f"  [hbn] placed={counts['placed']}")
    except Exception as exc:
        print(f"  [hbn] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 4) Persistence-tail NO — bets against extreme tail buckets
    # 4+ buckets away from yesterday's actual. Uses the cached
    # yesterday_actuals dict on state (refreshed each events refresh).
    try:
        counts = detect_and_execute_persistence_tail(
            station_id=upd.station_id,
            target_date_iso=upd.target_date_iso,
            target=upd.target,
            bucket_snapshots=list(ev.markets),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
            yesterday_actuals=state.yesterday_actuals,
            depth_map=depth_map,
            verbose=False,
        )
        if counts and counts.get("placed", 0) > 0:
            print(f"  [pers-tail] placed={counts['placed']}")
    except Exception as exc:
        print(f"  [pers-tail] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 5) Consensus YES momentum — DISABLED (see CONSENSUS_YES_ENABLED).
    # Bled −$312/304 trades in paper; the NO_momentum spread-bleed again.
    try:
        if not CONSENSUS_YES_ENABLED:
            counts = {}
        else:
            counts = detect_and_execute_consensus_yes(
                station_id=upd.station_id,
                target_date_iso=upd.target_date_iso,
                target=upd.target,
                bucket_snapshots=list(ev.markets),
                client=state.client,
                portfolio=state.portfolio,
                portfolio_path=state.args.portfolio_path,
                depth_map=depth_map,
                verbose=False,
            )
        if counts and counts.get("placed", 0) > 0:
            print(f"  [cons-yes] placed={counts['placed']}")
    except Exception as exc:
        print(f"  [cons-yes] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 6) Publication-window snapshot (post-midend only, idempotent
    # per (station, target, date, 30-min-offset-bin). snapshot_one
    # short-circuits BEFORE midend-local — so this is cheap to call
    # on every WUG tick and only writes when we're in the snapshot
    # window.
    try:
        from datetime import date as _date
        target_date = _date.fromisoformat(upd.target_date_iso)
        await snapshot_one(
            station=station, target=upd.target,
            target_date=target_date, ev=ev, http=state.http,
        )
    except Exception as exc:
        print(f"  [pub-window] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)


async def _run_lockin_yes(state: DaemonState, station, ev, upd: WUGUpdate) -> None:
    """Detect locked-in tail buckets (WUG primary) and write a paper-log
    decision. No live submission — lock-in fires write to intraday_log
    only until we wire submit_metar_live or similar."""
    if already_decided(
        upd.station_id, upd.target, upd.target_date_iso, state.intraday_log
    ):
        return
    bucket_kinds_thresholds = [parse_bucket(m) for m in ev.markets]
    win = find_early_tail_winner(
        upd.observed_extreme_c, upd.target,
        bucket_kinds_thresholds, station.unit,
    )
    if win is None:
        return
    win_kind, win_thr = win
    winning_market = next(
        (m for m in ev.markets if parse_bucket(m) == (win_kind, win_thr)),
        None,
    )
    decision = IntradayDecision(
        scan_time_utc=datetime.now(timezone.utc).isoformat(),
        station_id=upd.station_id, target=upd.target,
        target_date=upd.target_date_iso,
        decision="BUY_EARLY_TAIL",
        reason=(
            f"WUG observed {upd.observed_extreme_c:.1f}°C crossed "
            f"{win_kind} threshold {win_thr} (locked-in winner)"
        ),
        extreme_so_far_c=upd.observed_extreme_c,
        n_observations_used=upd.n_observations,
        winning_bucket_kind=win_kind, winning_bucket_threshold=win_thr,
        winning_bucket_label=getattr(winning_market, "bucket_label", None),
        event_slug=ev.slug, event_id=ev.event_id,
        yes_token_id=getattr(winning_market, "yes_token_id", None),
        market_yes_ask=getattr(winning_market, "yes_ask", None),
        market_yes_bid=getattr(winning_market, "yes_bid", None),
    )
    append_intraday_decision(decision)
    state.intraday_log.append(decision)
    print(f"  [lock-in] BUY_EARLY_TAIL {win_kind} {win_thr}")


# ── Refresh: events + WUG pollers + WS subscription ──────────────────
async def refresh_events_and_pollers(state: DaemonState) -> None:
    """Re-fetch gamma events, update pollers, refresh CLOB book subscription."""
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()

    fee_cfg = await fetch_live_fee_config(state.http)
    if fee_cfg is not None:
        warn_if_fee_config_changed(fee_cfg)

    events = await fetch_all_temperature_events(state.http)
    all_tokens: list[str] = []
    for ev in events:
        for m in ev.markets:
            if m.yes_token_id:
                all_tokens.append(m.yes_token_id)
            if m.no_token_id:
                all_tokens.append(m.no_token_id)
    clob = await fetch_clob_prices_batch(all_tokens, state.http)
    apply_clob_prices(events, clob)
    state.events = events

    # Rebuild event-by-sk lookup + active station set.
    excluded = {sid for sid, _t in load_active_exclusions(today_utc)}
    new_events_by_sk = {}
    new_stations_by_sk = {}
    for ev in events:
        station = match_event_to_station(ev)
        if station is None or station.station_id in excluded:
            continue
        target = "max" if ev.target == "highest" else "min"
        target_date = event_target_date(ev, station)
        if target_date > today_utc + timedelta(days=1):
            continue
        sk = (station.station_id, target, target_date.isoformat())
        new_events_by_sk[sk] = ev
        new_stations_by_sk[sk] = station
    state.events_by_sk = new_events_by_sk
    state.stations_by_sk = new_stations_by_sk

    # Reverse index token_id -> sk so the WS price_change handler can map a
    # delta straight to its event for the sub-second basket trigger.
    new_sk_by_token: dict[str, tuple[str, str, str]] = {}
    for sk, ev in new_events_by_sk.items():
        for m in ev.markets:
            if m.yes_token_id:
                new_sk_by_token[m.yes_token_id] = sk
            if m.no_token_id:
                new_sk_by_token[m.no_token_id] = sk
    state.sk_by_token = new_sk_by_token

    # Refresh WUG pollers — only run pollers for events that are
    # currently "in flight" in STATION-LOCAL time. The previous version
    # filtered by today_utc, which silently broke US stations after
    # 00:00 UTC: their May-27-local target_date no longer matched
    # May-28-UTC and the poller got reassigned to May-28-local, which
    # hadn't started yet — Wunderground returned http_400 NDF-0001 for
    # the entire timezone band west of UTC until the next station-local
    # midnight rolled in. (Verified empirically 2026-05-28 00:50 UTC.)
    #
    # New rule: a poller runs while now_utc is between the start of
    # the station-local target day and end-of-day-local + 6h (the
    # 6h trailing window keeps publication-window snapshots flowing
    # past midend-local — see weather_bot/publication_window.py).
    desired: set[tuple[str, str, str]] = set()
    for sk, station in new_stations_by_sk.items():
        sid, target, td_iso = sk
        from datetime import date as _date, timedelta as _td
        try:
            target_date = _date.fromisoformat(td_iso)
        except ValueError:
            continue
        midend_utc = midend_local_utc(target_date, station)
        start_polling = midend_utc - _td(days=1)   # start-of-day in station-local
        end_polling = midend_utc + _td(hours=6)    # 6h past end-of-day for pub-window
        if not (start_polling <= now_utc < end_polling):
            continue
        desired.add(sk)
        state.wug_pool.ensure_running(
            station=station, target=target,
            target_date=target_date,   # actual event date, not today_utc
        )
    await state.wug_pool.prune(keep=desired)

    # Refresh CLOB WS subscription on the union of:
    #   - NO tokens (Layer 7 + high-bucket NO read no_ask from book_cache)
    #   - YES tokens (consensus_yes exits fire on each YES book update,
    #     so its tokens MUST be in the subscription for sub-second exits)
    # We subscribe to both YES and NO for every active market — the
    # bandwidth cost is small and removes any "which side is needed"
    # state-tracking.
    sub_tokens: list[str] = []
    for sk in new_events_by_sk:
        for m in new_events_by_sk[sk].markets:
            if m.no_token_id:
                sub_tokens.append(m.no_token_id)
            if m.yes_token_id:
                sub_tokens.append(m.yes_token_id)
    # Also include YES tokens of any open consensus_yes positions whose
    # event isn't in the active set anymore (e.g. resolved-soon edge case)
    open_pos_tokens = {
        p.token_id for p in state.portfolio.positions
        if (p.strategy == "consensus_yes"
            and p.side == "YES"
            and p.status == "filled")
    }
    for tok in open_pos_tokens:
        if tok and tok not in sub_tokens:
            sub_tokens.append(tok)
    await _refresh_ws_subscription(state, sub_tokens)

    # V2 conditional preposit — run once per refresh on all events.
    # The function is sync; offload to a thread so we don't block the loop.
    try:
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(
                pool,
                lambda: submit_v2_conditional_preposit_orders(
                    events=list(new_events_by_sk.values()),
                    client=state.client,
                    portfolio=state.portfolio,
                    portfolio_path=state.args.portfolio_path,
                ),
            )
    except Exception as exc:
        print(f"[v2] failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Refresh yesterday_actuals cache for persistence_tail. Reads
    # data/forward_log.jsonl (~once per 5 min, cheap).
    try:
        state.yesterday_actuals = _load_yesterday_actuals()
    except Exception as exc:
        print(f"[pers-tail load] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # Consistency arb — scans paired (max, min) events for
    # implementable arb (buy YES on both sides). Sync, run inline (cheap).
    try:
        counts = CONSISTENCY_ARB_ENABLED and await detect_and_execute_consistency_arb(
            events=list(new_events_by_sk.values()),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
            book_cache=state.book_cache,
            http=state.http,
        )
        rej = {k: v for k, v in (counts or {}).items() if k.startswith("rej_")}
        if counts and (counts.get("placed", 0) > 0 or sum(rej.values()) > 0):
            # Funnel: how many candidates cleared the cheap top-of-book
            # pre-filter, then how many survived depth + fees + artifact.
            cleared_tob = counts.get("placed", 0) + sum(rej.values())
            print(f"[cons-arb] confirmed={counts.get('placed', 0)} "
                  f"(net-of-fee, depth-checked)  "
                  f"cleared_top_of_book={cleared_tob}  "
                  f"rejected={rej}  pairs={counts.get('pairs_scanned', 0)}")
    except Exception as exc:
        print(f"[cons-arb] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # Consensus-YES trailing exit — DISABLED with the strategy. The 5 open
    # positions ride to resolution rather than bleeding more spread on exit.
    try:
        exit_counts = {} if not CONSENSUS_YES_ENABLED else evaluate_consensus_yes_exits(
            events=list(new_events_by_sk.values()),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
        )
        if exit_counts and exit_counts.get("sold", 0) > 0:
            print(f"[cons-yes-exit] sold={exit_counts['sold']}  "
                  f"holding={exit_counts.get('holding', 0)}")
    except Exception as exc:
        print(f"[cons-yes-exit] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # Consensus BASKET — when a bucket's YES ≥ 0.85 (the emerged winner),
    # buy YES $5 on it + NO $5 on every other bucket, then HOLD to
    # resolution (no exit). This is the operator's hold-to-resolution
    # answer to consensus_yes's spread-bleed. Runs here on the freshly
    # repriced event set (5-min cadence) so it catches a 0.85 crossing
    # whether it came from an extreme move or pure market repricing.
    # Fetches depth itself, only for events that actually trigger
    # (rare), so the extra REST is bounded. Paper-only.
    try:
        basket_counts = await detect_and_execute_consensus_basket(
            events=list(new_events_by_sk.values()),
            client=state.client,
            portfolio=state.portfolio,
            http=state.http,
            book_cache=state.book_cache,
            portfolio_path=state.args.portfolio_path,
            verbose=False,
        )
        if basket_counts and basket_counts.get("baskets_placed", 0) > 0:
            print(f"[cons-basket] baskets={basket_counts['baskets_placed']} "
                  f"legs_filled={basket_counts.get('legs_filled', 0)}")
    except Exception as exc:
        print(f"[cons-basket] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # Basket threshold SWEEP — SHADOW logger (no orders). Each time an
    # event's leading-bucket YES reaches a new high-water penny in
    # [0.70, 0.99], record a shadow basket (winner YES + all fade NO,
    # depth-walked + fee-applied) so analyze_basket_sweep.py can build
    # the per-station P&L-vs-trigger curve for tuning. Bounded REST:
    # one depth fetch per event per new penny level (≤30/event/day).
    try:
        sweep_counts = await log_basket_sweep(
            events=list(new_events_by_sk.values()),
            http=state.http,
            book_cache=state.book_cache,
            leader_state=state.current_leader,
            extreme_state=state.latest_extreme,
        )
        if sweep_counts and sweep_counts.get("snapshots", 0) > 0:
            print(f"[sweep-log] snapshots={sweep_counts['snapshots']} "
                  f"levels_advanced={sweep_counts.get('levels_advanced', 0)}")
    except Exception as exc:
        print(f"[sweep-log] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # NOTE: the daily resolver used to run HERE, inside every 5-min
    # refresh. It does 50+ sequential 1/sec-throttled WUG fetches, so
    # it blocked refresh for ~5-6 min AND re-fetched the same
    # resolutions every 5 min (6472 audit lines / 23h). It now runs as
    # its own background task on a 30-min cadence — see
    # _resolver_loop() spawned in main_async. Resolutions only update
    # once/day so 30 min is plenty.

    # ws_fresh = tokens whose WS book is fresh enough to serve a read
    # RIGHT NOW. If this stays ~0, Polymarket isn't sending book
    # snapshots on subscribe and we need a cold-start REST seed (the
    # arb playbook's L1 lesson) — the sweep would otherwise fall back to
    # top-of-book everywhere.
    try:
        ws_fresh = state.book_cache.fresh_size(max_age_seconds=120.0)
    except Exception:
        ws_fresh = -1
    print(
        f"[refresh] {now_utc.isoformat()}  events={len(events)}  "
        f"active_sk={len(new_events_by_sk)}  wug_pollers={len(state.wug_pool.keys())}  "
        f"ws_tokens={len(sub_tokens)}  ws_fresh={ws_fresh}"
    )


async def _resolver_loop(state: "DaemonState", interval_s: float = 1800.0) -> None:
    """Background daily-resolver loop. Runs every `interval_s` (default
    30 min). Resolutions only change once per station per day, so a
    tight cadence just wastes WUG calls — 30 min catches each station
    within half an hour of its midend+grace window opening.

    Runs independently of refresh_events_and_pollers so the resolver's
    50+ sequential throttled WUG fetches never block the main loop or
    the WUG pollers. Reads the latest events/stations snapshot off
    `state` (populated by the most recent refresh).
    """
    while not state.shutdown_event.is_set():
        try:
            if state.events_by_sk:
                res_counts = await resolve_settled_events(
                    events_by_sk=state.events_by_sk,
                    stations_by_sk=state.stations_by_sk,
                    http=state.http,
                )
                if res_counts.get("resolved", 0) > 0:
                    print(f"[resolver] {res_counts}")
                    # A fresh resolution means persistence_tail has a new
                    # prior — refresh the cached yesterday_actuals.
                    try:
                        state.yesterday_actuals = _load_yesterday_actuals()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[resolver] loop failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _no_side_sweep_loop(state: "DaemonState", interval_s: float = 5.0) -> None:
    """Fast NO-side sweep — the 'fire faster' path.

    The WUG poller only re-evaluates a station when its extreme moves
    (≤60s, and throttle-bound). Between ticks, a fillable NO offer can
    appear and we'd miss it until the next tick. This loop re-runs
    Layer 7 + high-bucket-NO + persistence-tail for every active
    (station,target,date) every ~`interval_s`s, depth-walking
    SYNCHRONOUSLY from the WS BookCache ladder (no REST). Strategy dedupe
    / progressive-eval / trigger-time gates make re-runs cheap no-ops;
    the win is catching a fillable offer within seconds instead of ~60s
    — e.g. grabbing a Layer 7 dead-bucket reprice before it climbs to
    the $0.99 cap.

    Cadence note: 5s on a 1-vCPU VPS (the arb playbook runs ~2s, but its
    per-iteration cost is lighter than our 3-strategy × depth-walk ×
    dedupe scans). Drop lower only after confirming steady-state CPU.
    """
    sweep_n = 0
    while not state.shutdown_event.is_set():
        sweep_n += 1
        depth_hits = depth_misses = 0  # WS-ladder availability this sweep
        try:
            for sk, ev in list(state.events_by_sk.items()):
                station = state.stations_by_sk.get(sk)
                if station is None:
                    continue
                sid, target, td_iso = sk
                # Build a depth_map from the WS ladder for this event's
                # tokens (None entries → strategy falls back to top-of-book).
                depth_map = {}
                for m in ev.markets:
                    for tok in (m.no_token_id, m.yes_token_id):
                        if tok:
                            d = state.book_cache.get_depth(tok, max_age_seconds=30.0)
                            if d is not None:
                                depth_map[tok] = d
                                depth_hits += 1
                            else:
                                depth_misses += 1
                extreme = state.latest_extreme.get(sk)
                # persistence-tail doesn't need the extreme (uses prior).
                try:
                    detect_and_execute_persistence_tail(
                        station_id=sid, target_date_iso=td_iso, target=target,
                        bucket_snapshots=list(ev.markets), client=state.client,
                        portfolio=state.portfolio,
                        portfolio_path=state.args.portfolio_path,
                        yesterday_actuals=state.yesterday_actuals,
                        depth_map=depth_map, verbose=False,
                    )
                except Exception as exc:
                    print(f"[sweep] pers-tail {sid}: {exc}", file=sys.stderr)
                if extreme is None:
                    continue  # Layer7 + HBN need the observed extreme
                try:
                    LAYER7_ENABLED and detect_and_execute_guaranteed_buys(
                        station_id=sid, target_date_iso=td_iso,
                        observed_extreme_c=extreme, target=target,
                        bucket_snapshots=list(ev.markets), client=state.client,
                        portfolio=state.portfolio,
                        portfolio_path=state.args.portfolio_path,
                        book_cache=state.book_cache, depth_map=depth_map,
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"[sweep] layer7 {sid}: {exc}", file=sys.stderr)
                try:
                    detect_and_execute_high_bucket_no(
                        station_id=sid, target_date_iso=td_iso,
                        observed_extreme_c=extreme, target=target,
                        bucket_snapshots=list(ev.markets), client=state.client,
                        portfolio=state.portfolio,
                        portfolio_path=state.args.portfolio_path,
                        book_cache=state.book_cache, depth_map=depth_map,
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"[sweep] hbn {sid}: {exc}", file=sys.stderr)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[sweep] loop failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        # Periodically report WS-ladder availability so we can see whether
        # get_depth is actually feeding the sweep (vs falling back to
        # top-of-book everywhere → cold-start REST seed needed).
        if sweep_n % 24 == 0:
            tot = depth_hits + depth_misses
            pct = (100.0 * depth_hits / tot) if tot else 0.0
            print(f"[sweep] n={sweep_n} ws-depth hits={depth_hits} "
                  f"misses={depth_misses} ({pct:.0f}% fresh)")
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def _maybe_fire_consensus_yes_exit(state: "DaemonState", msg: dict) -> None:
    """Fire the trailing-stop exit check on a single book message.

    Triggered from the WS on_message handler. Sub-second latency from
    Polymarket book update → exit decision. The function is sync (matches
    on_message's sync contract); paper-mode submit_order is also sync.

    For a `book` event (full snapshot), the BookCache was just updated
    with fresh bid/ask values. For a `price_change` delta, BookCache
    marks the book as invalidated so fresh_best_ask returns None — in
    that case we skip until the next snapshot.
    """
    if not CONSENSUS_YES_ENABLED:
        return  # strategy disabled — open positions ride to resolution
    asset_id = msg.get("asset_id") or msg.get("market")
    if not asset_id:
        return
    # Find any open consensus_yes position on this YES token
    target_pos = None
    for p in state.portfolio.positions:
        if (p.strategy == "consensus_yes"
                and p.token_id == asset_id
                and p.side == "YES"
                and p.status == "filled"):
            target_pos = p
            break
    if target_pos is None:
        return

    # Pull fresh ask/bid from BookCache. fresh_best_ask returns None if
    # the book was invalidated by a delta — we skip and wait for the
    # next `book` snapshot.
    yes_ask = state.book_cache.fresh_best_ask(asset_id, max_age_seconds=5.0)
    yes_bid = state.book_cache.fresh_best_bid(asset_id, max_age_seconds=5.0)
    # Exit trigger evaluates on the bid (the price we sell into), so the
    # bid is what we require here.
    if yes_bid is None:
        return

    outcome = evaluate_single_consensus_yes_exit(
        position=target_pos,
        yes_ask=yes_ask, yes_bid=yes_bid,
        client=state.client,
        portfolio=state.portfolio,
        portfolio_path=state.args.portfolio_path,
        trigger="ws_push",
    )
    if outcome == "sold":
        # The exit logger already wrote the detailed record. Emit a
        # short stdout line for journalctl visibility.
        print(f"[ws] cons-yes exit fired on {target_pos.station_id} "
              f"{target_pos.bucket_label} (entry ${target_pos.entry_price:.3f})")


_LEADER_FLIP_LOG = Path("data/leader_flips.jsonl")
# Log a leader flip if the OUTGOING leader peaked ≥ this. Set to 0.70 to
# match the threshold-SWEEP's 0.70–0.99 band, so the flip dataset is
# COMPLETE for calibration: we capture every leadership change of a bucket
# that entered the 0.70–0.99 contention zone (needed to tune the per-station
# trigger anywhere in that range, not just at 0.85). Below 0.70 is pre-
# contention flicker (e.g. a 0.30 bucket losing the lead) — pure noise we
# never act on, so it stays unlogged.
FLIP_LOG_MIN_PEAK = 0.70


def _log_leader_flip(record: dict) -> None:
    """Append a leader-change record for the station-integrity monitor."""
    try:
        _LEADER_FLIP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LEADER_FLIP_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _maybe_fire_basket_on_cross(state: "DaemonState", msg: dict) -> None:
    """WS-delta basket trigger — the 'sub-second, no polling' path.

    On every `price_change` (which carries fresh best_bid/best_ask per
    token), map the affected token(s) to their event, recompute the
    leading YES bucket's fresh best ask, and if it just crossed a NEW
    penny in [0.70, 0.99], spawn a fire task. The task fires the basket
    legs as independent FAK orders (each fills whatever depth is there,
    the rest abort) and logs the per-penny sweep snapshot.

    Sync + fast (runs inline in the WS receive loop); all REST/order work
    is offloaded to an asyncio task so the socket keeps draining.
    """
    if msg.get("event_type") != "price_change":
        return
    affected: set = set()
    for ch in msg.get("price_changes", []) or []:
        aid = ch.get("asset_id")
        if not aid:
            continue
        sk = state.sk_by_token.get(aid)
        if sk is not None:
            affected.add(sk)
    if not affected:
        return

    for sk in affected:
        ev = state.events_by_sk.get(sk)
        if ev is None:
            continue
        # Fresh leader best-ask across the event's YES tokens (WS cache).
        leader_ya = 0.0
        leader_m = None
        runner_up = 0.0
        for m in ev.markets:
            if not m.yes_token_id:
                continue
            a = state.book_cache.best_ask(m.yes_token_id)
            if a is None:
                continue
            if a > leader_ya:
                runner_up = leader_ya
                leader_ya = a
                leader_m = m
            elif a > runner_up:
                runner_up = a
        if leader_ya <= 0 or leader_m is None:
            continue

        # Leader-flip log — every time the leading bucket CHANGES, record it
        # (data/leader_flips.jsonl). Feeds the station-integrity monitor.
        # We track the OUTGOING leader's PEAK ask while it led, so a flip can
        # be classified "late" = a bucket that REACHED ≥0.90 (looked locked)
        # then got dethroned — the UUWW/Moscow oracle-risk signature. (A flip
        # merely INTO a high price is just normal convergence, not suspect.)
        leader_label = leader_m.bucket_label
        now_dt = datetime.now(timezone.utc)
        now_ts = now_dt.timestamp()
        cur = state.current_leader.get(sk)   # (label, peak_ask, became_at_ts) or None
        if cur is None:
            state.current_leader[sk] = (leader_label, leader_ya, now_ts)
        elif cur[0] == leader_label:
            if leader_ya > cur[1]:           # same leader — track its peak
                state.current_leader[sk] = (leader_label, leader_ya, cur[2])
        else:
            prev_label, prev_peak, prev_since = cur   # leader CHANGED
            state.current_leader[sk] = (leader_label, leader_ya, now_ts)
            # Log if the OUTGOING leader entered the 0.70–0.99 contention
            # band (complete calibration dataset); skip sub-0.70 flicker.
            # from_dwell_s = how long the dethroned leader held = direct
            # stability signal (short dwell before losing = unstable).
            if prev_peak >= FLIP_LOG_MIN_PEAK:
                _log_leader_flip({
                    "ts_utc": now_dt.isoformat(),
                    "station_id": sk[0], "target": sk[1], "target_date": sk[2],
                    "from_bucket": prev_label, "from_peak_ask": round(prev_peak, 3),
                    "from_dwell_s": round(now_ts - prev_since, 1),
                    "to_bucket": leader_label, "to_leader_ask": round(leader_ya, 3),
                    "runner_up_ask": round(runner_up, 3),
                })

        hw_now = min(int(leader_ya * 100 + 1e-9), 99)
        if hw_now < 70:
            continue
        prev = state.basket_hw.get(sk, 69)
        if hw_now <= prev:
            continue
        # New penny crossing → advance high-water + spawn one fire task.
        state.basket_hw[sk] = hw_now
        if sk in state.basket_fire_inflight:
            continue
        state.basket_fire_inflight.add(sk)
        try:
            asyncio.get_running_loop().create_task(
                _fire_basket_cross(state, sk, hw_now))
        except RuntimeError:
            state.basket_fire_inflight.discard(sk)


async def _fire_basket_cross(state: "DaemonState", sk, hw_now: int) -> None:
    """Off-socket task: depth-walk + parallel-FAK fire for one event whose
    leader just crossed a new penny. Reuses the tested basket functions
    (which fetch REST depth + dedupe); book_cache feeds them the WS-fresh
    trigger price so the crossing is acted on immediately."""
    try:
        ev = state.events_by_sk.get(sk)
        if ev is None:
            return
        # Per-penny shadow sweep snapshot (REST depth, fee-applied).
        try:
            await log_basket_sweep(
                events=[ev], http=state.http, book_cache=state.book_cache,
                leader_state=state.current_leader,
                extreme_state=state.latest_extreme)
        except Exception as exc:
            print(f"[basket-cross sweep] {sk}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        # Real paper basket — fires only if the winner is ≥ trigger and the
        # event hasn't been attempted yet. Legs are independent FAK: each
        # fills to available depth, the rest abort ($0).
        try:
            counts = await detect_and_execute_consensus_basket(
                events=[ev], client=state.client, portfolio=state.portfolio,
                http=state.http, book_cache=state.book_cache,
                portfolio_path=state.args.portfolio_path,
            )
            if counts.get("baskets_placed", 0) > 0:
                print(f"[basket-cross] {sk[0]}/{sk[1]} fired at {hw_now}¢ "
                      f"legs_filled={counts.get('legs_filled', 0)}")
        except Exception as exc:
            print(f"[basket-cross fire] {sk}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    finally:
        state.basket_fire_inflight.discard(sk)


async def _refresh_ws_subscription(state: DaemonState, no_tokens: list[str]) -> None:
    """Restart the WS subscription with the current token set. Older
    task is cancelled first; clean cancellation is the priority over
    minimizing book-cache gap."""
    if state.ws_task is not None and not state.ws_task.done():
        state.ws_task.cancel()
        try:
            await asyncio.wait_for(state.ws_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    if not no_tokens:
        state.ws_task = None
        return

    def on_message(msg):
        # subscribe_and_watch dispatches each parsed JSON message here
        # SYNCHRONOUSLY (it just calls `on_message(message)`). Keep this
        # fast + non-blocking. Polymarket sometimes batches updates as
        # a list — handle both. After updating the book cache, also fire
        # consensus_yes exit checks for any tracked YES tokens.
        try:
            if isinstance(msg, list):
                for sub in msg:
                    if isinstance(sub, dict):
                        state.book_cache.on_book_message(sub)
                        _maybe_fire_consensus_yes_exit(state, sub)
                        _maybe_fire_basket_on_cross(state, sub)
            elif isinstance(msg, dict):
                state.book_cache.on_book_message(msg)
                _maybe_fire_consensus_yes_exit(state, msg)
                _maybe_fire_basket_on_cross(state, msg)
        except Exception as exc:
            print(f"[ws] on_book_message failed: {exc}", file=sys.stderr)

    state.ws_task = asyncio.create_task(
        subscribe_and_watch(
            token_ids=no_tokens,
            on_message=on_message,
        ),
        name="clob-ws",
    )


# ── Main loop ────────────────────────────────────────────────────────
async def main_async() -> int:
    args = parse_args()
    if kill_switch_active():
        print(f"[abort] KILL_SWITCH present at {KILL_SWITCH.resolve()}")
        return 0

    state = DaemonState(args)
    state.client = build_client(args.live)
    print(
        f"[daemon] start  paper_only={PAPER_ONLY}  "
        f"dry_run={state.client.is_dry_run}  "
        f"wug_interval={args.wug_poll_interval}s  "
        f"events_refresh={args.events_refresh_interval}s  "
        f"layer7={'on' if LAYER7_ENABLED else 'OFF'}  "
        f"consensus_yes={'on' if CONSENSUS_YES_ENABLED else 'OFF'}  "
        f"v2={'on' if V2_ENABLED else 'OFF'}  "
        f"consistency_arb={'on' if CONSISTENCY_ARB_ENABLED else 'OFF'}"
    )

    # Install signal handlers (UNIX). Windows skips silently.
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGTERM", "SIGINT"):
        try:
            sig = getattr(signal, sig_name)
            loop.add_signal_handler(sig, state.shutdown_event.set)
        except (NotImplementedError, AttributeError):
            pass

    async with httpx.AsyncClient(timeout=30.0) as http:
        state.http = http

        async def _callback(upd):
            await on_wug_update(state, upd)

        state.wug_pool = WUGPollerPool(
            callback=_callback, http=http, interval_s=args.wug_poll_interval,
        )

        # Startup refresh — wrapped so any error here can't prevent the
        # daemon from entering its main loop (where the in-loop refresh
        # retries every events_refresh_interval). A bare crash here was
        # the 2026-05-29 NameError crash-loop: the daemon never reached
        # the main loop, restarting every ~6 min (= resolver runtime).
        try:
            await refresh_events_and_pollers(state)
        except Exception as exc:
            import traceback
            print(f"[startup-refresh] failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            traceback.print_exc()

        # Spawn the daily resolver as its own background task (30-min
        # cadence) so its slow throttled WUG fetches never block the
        # main loop or the WUG pollers.
        state.resolver_task = asyncio.create_task(
            _resolver_loop(state), name="daily-resolver",
        )
        # Fast NO-side sweep (~10s) — re-fires Layer7/HBN/persistence
        # between WUG ticks using WS-cached depth. The "fire faster" path.
        state.no_side_sweep_task = asyncio.create_task(
            _no_side_sweep_loop(state), name="no-side-sweep",
        )

        # Main loop: periodic refresh + kill-switch + signal poll
        last_refresh = datetime.now(timezone.utc)
        try:
            while not state.shutdown_event.is_set():
                if kill_switch_active():
                    print("[daemon] KILL_SWITCH detected — shutting down")
                    break
                # Refresh events every refresh_interval seconds
                if (
                    (datetime.now(timezone.utc) - last_refresh).total_seconds()
                    >= args.events_refresh_interval
                ):
                    try:
                        await refresh_events_and_pollers(state)
                    except Exception as exc:
                        print(
                            f"[refresh] failed: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                    last_refresh = datetime.now(timezone.utc)
                try:
                    await asyncio.wait_for(state.shutdown_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
        finally:
            print("[daemon] stopping pollers + WS subscription + resolver")
            await state.wug_pool.stop_all()
            if state.ws_task is not None and not state.ws_task.done():
                state.ws_task.cancel()
                try:
                    await asyncio.wait_for(state.ws_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            if state.resolver_task is not None and not state.resolver_task.done():
                state.resolver_task.cancel()
                try:
                    await asyncio.wait_for(state.resolver_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            if state.no_side_sweep_task is not None and not state.no_side_sweep_task.done():
                state.no_side_sweep_task.cancel()
                try:
                    await asyncio.wait_for(state.no_side_sweep_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            try:
                state.portfolio.save(args.portfolio_path)
            except Exception:
                pass

    print("[daemon] shutdown complete")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
