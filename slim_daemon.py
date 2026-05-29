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
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import httpx

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
    submit_v2_conditional_preposit_orders,
)
from weather_bot.wug_poller import WUGPollerPool, WUGUpdate

KILL_SWITCH = Path("KILL_SWITCH")

# Hard paper-only gate — mirrors slim_scan.PAPER_ONLY. Flip to False
# only after the publication-window harness validates strategies.
PAPER_ONLY: bool = True


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
        self.client = None  # ExecutionClient — see build_client()
        self.intraday_log = load_intraday_log(DEFAULT_INTRADAY_LOG_PATH)
        # Cached prior-day actuals for persistence_tail. Refreshed on
        # each events refresh tick from data/forward_log.jsonl.
        self.yesterday_actuals: dict[tuple[str, str], float] = {}
        self.shutdown_event = asyncio.Event()
        self.ws_task: asyncio.Task | None = None
        self.wug_pool: WUGPollerPool | None = None
        self.resolver_task: asyncio.Task | None = None


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

    print(
        f"[wug] {upd.station_id}/{upd.target} {upd.target_date_iso}  "
        f"obs={upd.observed_extreme_c:.1f}°C (int={upd.observed_int}, "
        f"prev={upd.previous_int}, n={upd.n_observations})"
    )

    # 1) Lock-in YES — if a tail bucket just locked in, fire YES on it.
    try:
        await _run_lockin_yes(state, station, ev, upd)
    except Exception as exc:
        print(f"  [lock-in] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 2) Layer 7 progressive
    try:
        counts = detect_and_execute_guaranteed_buys(
            station_id=upd.station_id,
            target_date_iso=upd.target_date_iso,
            observed_extreme_c=upd.observed_extreme_c,
            target=upd.target,
            bucket_snapshots=list(ev.markets),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
            book_cache=state.book_cache,
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
            verbose=False,
        )
        if counts and counts.get("placed", 0) > 0:
            print(f"  [pers-tail] placed={counts['placed']}")
    except Exception as exc:
        print(f"  [pers-tail] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # 5) Consensus YES momentum — buy the leading mid bucket while it's
    # in the trade-able band. HIGHEST-RISK strategy in the rebuild —
    # closest path back to the prior NO_momentum bleed. Paper-only.
    try:
        counts = detect_and_execute_consensus_yes(
            station_id=upd.station_id,
            target_date_iso=upd.target_date_iso,
            target=upd.target,
            bucket_snapshots=list(ev.markets),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
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
        counts = detect_and_execute_consistency_arb(
            events=list(new_events_by_sk.values()),
            client=state.client,
            portfolio=state.portfolio,
            portfolio_path=state.args.portfolio_path,
        )
        if counts and counts.get("placed", 0) > 0:
            print(f"[cons-arb] opportunities={counts['placed']} "
                  f"pairs_scanned={counts.get('pairs_scanned', 0)}")
    except Exception as exc:
        print(f"[cons-arb] failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    # Consensus-YES trailing exit — walks open consensus_yes positions
    # and sells the ones whose price came back down ≥ peak_decline_ticks
    # below their peak while still ≥ entry + 5pp profit. Paper-only.
    try:
        exit_counts = evaluate_consensus_yes_exits(
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

    # NOTE: the daily resolver used to run HERE, inside every 5-min
    # refresh. It does 50+ sequential 1/sec-throttled WUG fetches, so
    # it blocked refresh for ~5-6 min AND re-fetched the same
    # resolutions every 5 min (6472 audit lines / 23h). It now runs as
    # its own background task on a 30-min cadence — see
    # _resolver_loop() spawned in main_async. Resolutions only update
    # once/day so 30 min is plenty.

    print(
        f"[refresh] {now_utc.isoformat()}  events={len(events)}  "
        f"active_sk={len(new_events_by_sk)}  wug_pollers={len(state.wug_pool.keys())}  "
        f"ws_tokens={len(sub_tokens)}"
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
            elif isinstance(msg, dict):
                state.book_cache.on_book_message(msg)
                _maybe_fire_consensus_yes_exit(state, msg)
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
        f"events_refresh={args.events_refresh_interval}s"
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
