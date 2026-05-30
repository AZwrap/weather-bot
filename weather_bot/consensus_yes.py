"""Consensus-YES momentum — buy the highest-YES mid bucket while it's
still in the trade-able band.

⚠️  HIGHEST-RISK STRATEGY IN THE LITE REBUILD.
This is the closest path back to the 2026-05 NO_momentum bleed. Per
the user's explicit request to add anyway. Run paper-only until N≥30
resolutions validate the win rate at each entry-price band.

Premise
=======
Consensus-winner buckets have the most depth — fills are easy. The
strategy buys YES on the bucket the market is converging on, BEFORE
it reaches the $0.95+ ceiling. Sells at a convergence exit (handled
separately).

Mechanism
=========
For each event, identify the bucket with the highest mid-bucket YES
ask. Fire YES if:
  1. yes_ask is in [MIN_YES_ASK, MAX_YES_ASK] band — out of band means
     consensus hasn't formed (too low) or has fully converged (too high)
  2. We don't already hold YES on this token
  3. Per-event, at most one YES position (the leading bucket)

Sizing: $5/fire (matches other strategies).
Sell logic: NOT yet implemented. The buy is paper-only via the dry-run
client; resolution PnL is computed by the analyzer.

Risks
=====
- The bucket the market is converging on may be wrong. ~30% of the
  time, the daily extreme lands in a neighbour bucket. At entry price
  $0.65 YES, breakeven win rate is 65% — bot loses if true hit-rate
  is below that.
- Mid-bucket momentum is the failure mode of NO_momentum inverted.
  Watch the realized win rate closely.

Output: data/consensus_yes_log.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .locations import STATIONS_BY_ID
from .polymarket import parse_bucket
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position, region_for
from .scanner import TradeSignal


DEFAULT_LOG_PATH = Path("data/consensus_yes_log.jsonl")
DEFAULT_EXIT_LOG_PATH = Path("data/consensus_yes_exit_log.jsonl")

MIN_YES_ASK: float = 0.40
"""Below this, consensus hasn't formed — too many buckets still in
contention. Wait."""

MAX_YES_ASK: float = 0.85
"""Above this we're in convergence-late territory. The bot has lost
the run-up; YES at $0.90+ is the same trade Layer 7 would have done
much cheaper had it fired. Skip."""

DEFAULT_SIZE_USD: float = 5.0

# ── Exit logic: ratcheting 5¢-grid floor + hard stop ──────────────────
# Per user spec (2026-05-29): a moving floor that snaps up in 5¢ steps
# as the bid rises, selling when the bid crosses below it. Plus a hard
# stop below entry so a never-profits position doesn't ride to a full
# loss at resolution.
#
# The trailing stop tracks the BID (the price we actually sell into) —
# measuring on the ask would let a "+profit on ask" realize as a loss
# on the bid when books are wide.

FLOOR_GRID: float = 0.05
"""Floor granularity. The ratchet floor = peak_bid rounded DOWN to the
nearest FLOOR_GRID. peak $0.57 → floor $0.55; peak $0.63 → floor $0.60."""

RATCHET_ARM: float = 0.05
"""The profit ratchet only ARMS once peak_bid ≥ entry + RATCHET_ARM
(one grid step of genuine profit). Until armed, only the hard stop is
active — this prevents a knee-jerk sell right at entry on entry-noise.
After arming, the floor trails the peak and only moves up."""

HARD_STOP_BUFFER: float = 0.10
"""Hard stop: sell if bid ≤ entry − HARD_STOP_BUFFER, regardless of
ratchet state. Caps the held-to-resolution tail (a consensus bucket
that's wrong marches to a $0 resolution = full ~$5 loss without this).
At $0.10 a stopped-out loser costs ~10¢/share instead of the full
entry price."""

# Legacy names kept for any external import; no longer drive the logic.
MIN_PROFIT_FROM_ENTRY: float = RATCHET_ARM
PEAK_DECLINE_TICKS: float = 0.01


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def detect_and_execute_consensus_yes(
    *,
    station_id: str,
    target_date_iso: str,
    target: str,
    bucket_snapshots: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    min_yes_ask: float = MIN_YES_ASK,
    max_yes_ask: float = MAX_YES_ASK,
    size_usd: float = DEFAULT_SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    depth_map: dict | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    """Fire YES on the leading mid bucket if yes_ask is in band.

    Returns counts:
      placed                  successful fires
      skipped_no_mid          no mid buckets in event
      skipped_no_ask          leading bucket missing yes_ask
      skipped_below_band      leading bucket yes_ask < min_yes_ask
                              (consensus hasn't formed)
      skipped_above_band      leading bucket yes_ask > max_yes_ask
                              (already converged)
      skipped_already_open    dedupe via Portfolio.is_open
      submit_failed           SDK / Polymarket rejection
    """
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0

    station = STATIONS_BY_ID.get(station_id)
    if station is None:
        return dict(counts)

    # Find the leading mid bucket (highest yes_ask among mids)
    mids: list[tuple[Any, str, int, float]] = []
    for m in bucket_snapshots:
        if not m.yes_token_id:
            continue
        try:
            kind, thr = parse_bucket(m)
        except (ValueError, TypeError, KeyError):
            continue
        if kind != "mid":
            continue
        if m.yes_ask is None:
            continue
        mids.append((m, kind, int(thr), float(m.yes_ask)))

    if not mids:
        counts["skipped_no_mid"] += 1
        return dict(counts)

    # Pick the highest yes_ask
    mids.sort(key=lambda x: -x[3])
    leader_m, leader_kind, leader_thr, leader_ya = mids[0]

    if leader_ya < min_yes_ask:
        counts["skipped_below_band"] += 1
        return dict(counts)
    if leader_ya > max_yes_ask:
        counts["skipped_above_band"] += 1
        return dict(counts)

    if portfolio.is_open(leader_m.yes_token_id, "YES"):
        counts["skipped_already_open"] += 1
        return dict(counts)

    # Submit FAK BUY YES at the 2-decimal rounded ask. Polymarket walks
    # the book; we'd fill at the cheapest available offer ≤ limit.
    # Round UP via the round-to-2-decimals so the engine matches the
    # current ask (avoid rounding down and missing the fill).
    import math
    submitted_limit = math.ceil(leader_ya * 100) / 100.0
    # Clamp to max_yes_ask just in case rounding pushes above the cap.
    submitted_limit = min(submitted_limit, round(max_yes_ask, 2))

    # DEPTH-AWARE FILL — walk the YES ask ladder up to the limit. This is
    # the strategy where it matters most: buying YES at the ask on a wide
    # book, a real FAK walks UP, so the depth-walked avg is worse than the
    # top tick — exactly the cost that makes consensus_yes a spread-bleed.
    from .polymarket import simulate_buy_fill
    depth = (depth_map or {}).get(leader_m.yes_token_id)
    sim = simulate_buy_fill(depth, size_usd, submitted_limit)
    if sim is None:
        if depth_map is not None:
            counts["skipped_no_depth"] += 1
            return dict(counts)
        fill_avg = leader_ya
        shares = float(max(1, int(size_usd / submitted_limit)))
        fully_filled = True
        depth_source = "top_of_book_fallback"
    else:
        fill_avg, shares, fully_filled = sim
        depth_source = "depth_walk"

    signal = TradeSignal(
        station=station, event_title="", event_slug="",
        target=target,
        target_date=datetime.fromisoformat(target_date_iso).date(),
        bucket_label=leader_m.bucket_label, bucket_kind=leader_kind,
        market_id=int(leader_m.market_id), token_id=leader_m.yes_token_id,
        our_prob=leader_ya, yes_implied=leader_ya,
        yes_bid=leader_m.yes_bid, yes_ask=leader_m.yes_ask,
        side="YES", edge=0.0,
        fill_price=fill_avg, volume_24hr=0.0,
        bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
        kelly_full=1.0, position_usd=fill_avg * shares,
    )
    try:
        result = client.submit_order(
            signal, order_type="FAK", sdk_side="BUY",
            limit_price=submitted_limit, override_shares=shares,
        )
    except Exception as exc:
        counts["submit_failed"] += 1
        _log_event({
            "ts_utc": _now_utc_iso(), "result": "submit_exception",
            "station_id": station_id, "target": target,
            "target_date": target_date_iso,
            "bucket_label": leader_m.bucket_label,
            "yes_ask_snapshot": leader_ya, "exc": str(exc)[:200],
        }, log_path)
        return dict(counts)
    if not result.ok:
        counts["submit_failed"] += 1
        _log_event({
            "ts_utc": _now_utc_iso(), "result": "rejected",
            "station_id": station_id, "target": target,
            "target_date": target_date_iso,
            "bucket_label": leader_m.bucket_label,
            "yes_ask_snapshot": leader_ya,
            "message": (result.message or "")[:200],
        }, log_path)
        return dict(counts)

    position = Position(
        token_id=leader_m.yes_token_id, side="YES",
        station_id=station_id, region=region_for(station_id),
        market_id=int(leader_m.market_id),
        bucket_label=leader_m.bucket_label, bucket_kind=leader_kind,
        bucket_threshold=leader_thr,
        target_date=target_date_iso,
        shares=shares, entry_price=fill_avg, position_usd=fill_avg * shares,
        submitted_at=_now_utc_iso(), status="filled",
        order_id=result.order_id, strategy="consensus_yes",
    )
    try:
        portfolio.add(position)
        portfolio.save(portfolio_path)
    except Exception:
        pass
    counts["placed"] += 1
    _log_event({
        "ts_utc": _now_utc_iso(), "result": "filled",
        "station_id": station_id, "target": target,
        "target_date": target_date_iso,
        "bucket_label": leader_m.bucket_label,
        "bucket_kind": leader_kind, "bucket_threshold": leader_thr,
        "yes_ask_snapshot": leader_ya,
        "submitted_limit": submitted_limit,
        "fill_price": result.fill_price,        # depth-walked avg fill
        "depth_source": depth_source,
        "fully_filled": fully_filled,
        "shares": shares, "size_usd": fill_avg * shares,
        "order_id": result.order_id,
        "n_mids_in_event": len(mids),
        # Rank-2 yes_ask to gauge gap to second-place (signal-strength proxy)
        "second_yes_ask": mids[1][3] if len(mids) > 1 else None,
    }, log_path)
    if verbose:
        print(f"  [cons-yes] {station_id}/{target} {target_date_iso} "
              f"BUY YES {leader_m.bucket_label} @ {leader_ya:.3f} "
              f"(limit ${submitted_limit:.2f})")

    return dict(counts)


# ──────────────────────────────────────────────────────────────────────
# Trailing-stop exit logic
# ──────────────────────────────────────────────────────────────────────

def _peak_key(token_id: str, side: str) -> str:
    return f"{token_id}|{side}"


def evaluate_single_consensus_yes_exit(
    *,
    position: Position,
    yes_ask: float | None,
    yes_bid: float | None,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    floor_grid: float = FLOOR_GRID,
    ratchet_arm: float = RATCHET_ARM,
    hard_stop_buffer: float = HARD_STOP_BUFFER,
    log_path: Path = DEFAULT_EXIT_LOG_PATH,
    trigger: str = "ws_push",
    verbose: bool = False,
) -> str:
    """Evaluate the ratcheting-floor + hard-stop exit on a single open
    consensus_yes position given a fresh (yes_ask, yes_bid) snapshot.

    Exit logic (per user spec 2026-05-29):
      - Track peak_bid (highest bid since entry).
      - Profit ratchet: once peak_bid ≥ entry + ratchet_arm, a floor sits
        at peak_bid rounded DOWN to floor_grid (e.g. peak 0.63 → 0.60).
        It only moves up. Sell when bid < floor.
      - Hard stop (always on): sell when bid ≤ entry − hard_stop_buffer.
      - Fill at the bid (the price we cross down through).

    Returns: "peak_updated" | "holding" | "sold" | "no_quote" |
    "skipped_status" | "submit_failed".

    Called from the WS book-update callback (sub-second) AND the 5-min
    sweep (backup). Idempotent once status → "filled_closed".
    """
    import math

    if position.strategy != "consensus_yes":
        return "skipped_status"
    if position.side != "YES":
        return "skipped_status"
    if position.status != "filled":
        return "skipped_status"
    # Track on the BID — the price we actually sell into.
    if yes_bid is None:
        return "no_quote"

    key = _peak_key(position.token_id, position.side)
    peak = portfolio.consensus_yes_peak_by_pos.get(key, position.entry_price)

    if yes_bid > peak:
        peak = float(yes_bid)
        portfolio.consensus_yes_peak_by_pos[key] = peak

    entry = position.entry_price

    # Hard stop — always active.
    hard_stop_level = entry - hard_stop_buffer
    hard_stop_hit = yes_bid <= hard_stop_level

    # Profit ratchet — arms only once peak rose ≥ ratchet_arm above entry.
    ratchet_armed = peak >= entry + ratchet_arm
    ratchet_floor = None
    ratchet_hit = False
    if ratchet_armed:
        # peak rounded DOWN to the floor grid (round() clears float dust)
        ratchet_floor = round(math.floor((peak + 1e-9) / floor_grid) * floor_grid, 2)
        ratchet_hit = yes_bid < ratchet_floor

    if not (hard_stop_hit or ratchet_hit):
        return "holding"

    which = "hard_stop" if hard_stop_hit and not ratchet_hit else (
        "ratchet_floor" if ratchet_hit and not hard_stop_hit else "both")

    # Sell at the current bid (rounded down to $0.01).
    limit_sell = max(0.01, math.floor(yes_bid * 100) / 100.0)

    signal = TradeSignal(
        station=STATIONS_BY_ID.get(position.station_id) or _stub_station(position),
        event_title="", event_slug="",
        target="max" if "max" in str(position.target_date) else position.bucket_kind,
        target_date=datetime.fromisoformat(position.target_date).date(),
        bucket_label=position.bucket_label,
        bucket_kind=position.bucket_kind,
        market_id=int(position.market_id),
        token_id=position.token_id,
        our_prob=yes_ask, yes_implied=yes_ask,
        yes_bid=yes_bid, yes_ask=yes_ask,
        side="YES", edge=0.0,
        fill_price=limit_sell, volume_24hr=0.0,
        bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
        kelly_full=1.0, position_usd=position.position_usd,
    )
    try:
        result = client.submit_order(
            signal, order_type="FAK", sdk_side="SELL",
            limit_price=limit_sell, override_shares=position.shares,
        )
    except Exception as exc:
        _log_exit({
            "ts_utc": _now_utc_iso(), "result": "submit_exception",
            "station_id": position.station_id, "token_id": position.token_id,
            "bucket_label": position.bucket_label, "exc": str(exc)[:200],
        }, log_path)
        return "submit_failed"
    if not result.ok:
        _log_exit({
            "ts_utc": _now_utc_iso(), "result": "rejected",
            "station_id": position.station_id, "token_id": position.token_id,
            "bucket_label": position.bucket_label,
            "message": (result.message or "")[:200],
        }, log_path)
        return "submit_failed"

    position.status = "filled_closed"
    try:
        portfolio.save(portfolio_path)
    except Exception:
        pass
    net_per_share = limit_sell - position.entry_price
    _log_exit({
        "ts_utc": _now_utc_iso(), "result": "sold",
        "station_id": position.station_id,
        "target_date": position.target_date,
        "bucket_label": position.bucket_label,
        "bucket_threshold": position.bucket_threshold,
        "entry_price": position.entry_price,
        "peak_yes_bid": peak,            # trailing stop tracks the BID
        "exit_reason": which,            # "ratchet_floor" | "hard_stop" | "both"
        "ratchet_floor": ratchet_floor,  # None if ratchet never armed
        "hard_stop_level": hard_stop_level,
        "current_yes_ask": yes_ask,
        "current_yes_bid": yes_bid,
        "submitted_sell_limit": limit_sell,
        "fill_price": result.fill_price,
        "shares": position.shares,
        "net_per_share_usd": net_per_share,
        "net_total_usd": net_per_share * position.shares,
        "order_id": result.order_id,
        "trigger": trigger,
    }, log_path)
    if verbose:
        print(f"  [cons-yes-exit] {position.station_id} {position.bucket_label} "
              f"entry ${position.entry_price:.3f} → peak ${peak:.3f} → "
              f"sell ${limit_sell:.3f} (net ${net_per_share*position.shares:+.2f})")
    return "sold"


def _log_exit(record: dict, log_path: Path = DEFAULT_EXIT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def evaluate_consensus_yes_exits(
    *,
    events: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    log_path: Path = DEFAULT_EXIT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Sweep all open consensus_yes positions. Designed as a periodic
    backup to the WS-push path — catches any positions whose YES token
    isn't currently in the WS subscription or whose updates were missed.

    Per-position logic lives in `evaluate_single_consensus_yes_exit`.
    """
    counts: dict[str, int] = defaultdict(int)

    # Build token_id → current (yes_ask, yes_bid) lookup from events
    quotes: dict[str, tuple[float | None, float | None]] = {}
    for ev in events:
        for m in ev.markets:
            if m.yes_token_id:
                quotes[m.yes_token_id] = (m.yes_ask, m.yes_bid)

    for p in list(portfolio.positions):
        ya, yb = quotes.get(p.token_id, (None, None))
        outcome = evaluate_single_consensus_yes_exit(
            position=p, yes_ask=ya, yes_bid=yb,
            client=client, portfolio=portfolio, portfolio_path=portfolio_path,
            log_path=log_path, trigger="sweep", verbose=verbose,
        )
        counts[outcome] += 1

    return dict(counts)


def _stub_station(p: Position):
    """Build a placeholder Station for TradeSignal when STATIONS_BY_ID
    lookup fails (shouldn't happen but defensive)."""
    class _S:
        name = ""
        station_id = p.station_id
        timezone = "UTC"
        unit = "C"
    return _S()
