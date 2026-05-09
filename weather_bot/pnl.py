"""Hypothetical P&L from forward-log records.

Given a resolved record (forecast + market snapshot + actual observation),
walk every bucket and compute:

  1. Did the bucket WIN?  Polymarket rounds the observed temperature to whole
     degrees of the market's unit (1 °C or 1 °F).  Mid °F buckets cover two
     integers (e.g. "60-61 °F"); °C mid buckets cover exactly one.
  2. Would we have traded it?  YES if our_prob > yes_ask + min_edge,
     NO if our_prob < yes_bid - min_edge.
  3. What was the would-have-been P&L?  Use deci-Kelly sizing.

The aggregate stats (total bet, total profit, ROI, win rate, per-station
breakdown) are what you'd want to see before risking real capital.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

from .forward_log import BucketSnapshot, ForwardLogRecord
from .locations import STATIONS_BY_ID
from .sizing import position_size_usd
from .units import Unit, c_to_f


Side = Literal["YES", "NO"]

# METAR (Iowa State ASOS) reports temperatures in whole-degree Celsius
# natively, so the daily max from the truth source is already an integer
# matching Polymarket's bucket. We use floor on the converted value
# defensively (and because non-METAR stations are no longer traded).


@dataclass
class BucketTradeResult:
    """Outcome of one hypothetical bucket trade.

    `won` and `actual_obs_c` are None until the underlying record is resolved
    by `resolve_log.py`. Pending trades have `profit_usd = 0` and contribute
    to exposure totals but not to P&L until they resolve.
    """

    station_id: str
    target: str           # "max" or "min"
    target_date: str      # ISO date string
    issue_time_utc: str   # ISO timestamp string
    bucket_label: str
    bucket_kind: str      # "low_tail" / "mid" / "high_tail"
    threshold: int
    our_prob: float
    market_yes_bid: float | None
    market_yes_ask: float | None
    side: Side | None    # None when no edge / not actionable
    fill_price: float | None
    edge: float | None
    position_usd: float
    actual_obs_c: float | None
    won: bool | None      # None = not yet resolved
    profit_usd: float     # 0 when pending or we didn't trade


# ──────────────────────────────────────────────────────────────────────────
# Resolution: did the bucket win?
# ──────────────────────────────────────────────────────────────────────────


def _rounded_observation(actual_c: float, unit: Unit) -> int:
    """Convert observed temperature to the integer used for bucket resolution.

    METAR truth values are whole-degree Celsius natively, so floor on the
    converted value gives the integer Polymarket resolves on.
    """
    actual = c_to_f(actual_c) if unit == "F" else actual_c
    return int(math.floor(actual))


def bucket_won(snap: BucketSnapshot, actual_int: int, unit: Unit) -> bool:
    """Decide whether `snap`'s bucket WON, given the rounded observation.

    Polymarket bucket conventions (verified empirically against live markets):
      °C: low_tail "X°C or below" → actual ≤ X
          mid     "X°C"          → actual == X
          high_tail "X°C or higher" → actual ≥ X
      °F: low_tail "X°F or below" → actual ≤ X
          mid     "X-Y°F"         → X ≤ actual ≤ X+1   (always 2 wide)
          high_tail "X°F or higher" → actual ≥ X
    """
    if snap.kind == "low_tail":
        return actual_int <= snap.threshold
    if snap.kind == "high_tail":
        return actual_int >= snap.threshold
    # mid
    if unit == "C":
        return actual_int == snap.threshold
    return snap.threshold <= actual_int <= snap.threshold + 1


# ──────────────────────────────────────────────────────────────────────────
# Per-bucket trade simulation
# ──────────────────────────────────────────────────────────────────────────


def _decide_side(
    our_prob: float,
    yes_bid: float | None,
    yes_ask: float | None,
    min_edge: float,
) -> tuple[Side, float, float] | None:
    """Pick the side with positive edge ≥ min_edge (None if neither)."""
    edge_yes = -1.0
    fill_yes = None
    if yes_ask is not None and 0.0 < yes_ask < 1.0:
        edge_yes = our_prob - yes_ask
        fill_yes = yes_ask

    edge_no = -1.0
    fill_no = None
    if yes_bid is not None and 0.0 < yes_bid < 1.0:
        # NO wins when YES loses; fill price for buying NO is (1 − yes_bid).
        # Edge_no expressed in NO-prob terms: (1 − our_prob) − (1 − yes_bid)
        #                                  = yes_bid − our_prob.
        edge_no = yes_bid - our_prob
        fill_no = 1.0 - yes_bid

    if edge_yes >= edge_no and edge_yes >= min_edge and fill_yes is not None:
        return "YES", edge_yes, fill_yes
    if edge_no > edge_yes and edge_no >= min_edge and fill_no is not None:
        return "NO", edge_no, fill_no
    return None


def _profit_usd(side: Side, fill_price: float, position_usd: float, won: bool) -> float:
    """Signed P&L given side, fill price, position, and outcome.

    Per dollar invested at price c:
        profit if side wins  = (1 − c) / c
        profit if side loses = −1
    """
    side_won = won if side == "YES" else not won
    if side_won:
        return position_usd * (1.0 - fill_price) / fill_price
    return -position_usd


def simulate_record(
    record: ForwardLogRecord,
    *,
    bankroll_usd: float = 1000.0,
    kelly_multiplier: float = 0.1,
    max_position_usd: float = 50.0,
    min_edge: float = 0.05,
    max_edge: float = 0.25,
    min_yes_price: float = 0.05,
    max_yes_price: float = 0.95,
    liquidity_cap_fraction: float = 0.1,
) -> list[BucketTradeResult]:
    """Walk every bucket in the record, decide whether we'd have traded, and
    compute hypothetical P&L.

    Filters that catch over-confident model errors:
      * min_edge / max_edge — skip if edge is too small (noise) or too big (model error at tails)
      * min_yes_price / max_yes_price — skip extreme-tail markets where probabilities are unreliable

    Resolution-agnostic: works on records that haven't been resolved yet (P&L
    is reported as 0 until `actual_obs_c` is filled by resolve_log).
    """
    if record.bucket_snapshots is None:
        return []

    station = STATIONS_BY_ID.get(record.station_id)
    if station is None:
        return []
    unit: Unit = station.unit

    # Resolution-aware bucket-won lookup. Returns None for pending records.
    if record.actual_obs_c is None:
        def lookup_won(_snap):
            return None
    else:
        actual_int = _rounded_observation(record.actual_obs_c, unit)
        def lookup_won(snap):
            return bucket_won(snap, actual_int, unit)

    issue_iso = record.issue_time_utc.isoformat()
    out: list[BucketTradeResult] = []
    for snap in record.bucket_snapshots:
        won = lookup_won(snap)

        def make_no_trade() -> BucketTradeResult:
            return BucketTradeResult(
                station_id=record.station_id, target=record.target,
                target_date=record.target_date.isoformat(),
                issue_time_utc=issue_iso,
                bucket_label=snap.bucket_label, bucket_kind=snap.kind,
                threshold=snap.threshold,
                our_prob=snap.our_prob,
                market_yes_bid=snap.yes_bid, market_yes_ask=snap.yes_ask,
                side=None, fill_price=None, edge=None,
                position_usd=0.0,
                actual_obs_c=record.actual_obs_c, won=won, profit_usd=0.0,
            )

        decision = _decide_side(snap.our_prob, snap.yes_bid, snap.yes_ask, min_edge)
        if decision is None:
            out.append(make_no_trade())
            continue

        side, edge, fill_price = decision

        # Skip absurd edges (almost always model errors at the tails)
        if edge > max_edge:
            out.append(make_no_trade())
            continue
        # Skip extreme-tail markets where bid/ask is unreliable
        if not (min_yes_price <= fill_price <= max_yes_price):
            out.append(make_no_trade())
            continue
        pos = position_size_usd(
            snap.our_prob, fill_price, side,
            bankroll_usd=bankroll_usd,
            kelly_multiplier=kelly_multiplier,
            max_position_usd=max_position_usd,
            liquidity_cap_usd=snap.volume_24hr * liquidity_cap_fraction,
        )
        if pos <= 0:
            out.append(make_no_trade())
            continue

        # P&L only when resolved; pending trades report 0.
        profit = _profit_usd(side, fill_price, pos, won) if won is not None else 0.0
        out.append(BucketTradeResult(
            station_id=record.station_id, target=record.target,
            target_date=record.target_date.isoformat(),
            issue_time_utc=issue_iso,
            bucket_label=snap.bucket_label, bucket_kind=snap.kind,
            threshold=snap.threshold,
            our_prob=snap.our_prob,
            market_yes_bid=snap.yes_bid, market_yes_ask=snap.yes_ask,
            side=side, fill_price=fill_price, edge=edge,
            position_usd=pos,
            actual_obs_c=record.actual_obs_c, won=won, profit_usd=profit,
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PnLSummary:
    n_trades: int            # actionable trades (resolved + pending)
    n_resolved: int
    n_pending: int
    n_wins: int
    n_losses: int
    total_pos_usd: float     # all-trade exposure
    resolved_pos_usd: float  # exposure on resolved trades only
    total_profit_usd: float  # P&L on resolved trades only
    win_rate: float | None   # None when no resolved trades
    roi_pct: float | None    # profit ÷ resolved exposure


def dedupe_trades_by_bucket(
    results: Iterable[BucketTradeResult],
) -> list[BucketTradeResult]:
    """Keep one actionable trade per unique (station, target, target_date,
    kind, threshold) — the earliest by issue_time.

    `simulate_record` returns one BucketTradeResult per (record, bucket) pair,
    so a bucket logged across N hourly snapshots produces N copies of the
    same paper trade. For an honest P&L accounting we want exactly ONE — the
    one the bot would have placed when it first detected edge. Subsequent
    snapshots are the SAME trade still held, not new trades.

    Drops non-actionable (side=None) rows since they don't affect aggregates.
    """
    sorted_trades = sorted(results, key=lambda t: t.issue_time_utc)
    by_bucket: dict[tuple[str, str, str, str, int], BucketTradeResult] = {}
    for t in sorted_trades:
        if t.side is None:
            continue
        key = (t.station_id, t.target, t.target_date, t.bucket_kind, t.threshold)
        if key not in by_bucket:
            by_bucket[key] = t
    return list(by_bucket.values())


def cap_per_station_per_day(
    results: Iterable[BucketTradeResult],
    cap_usd: float,
) -> list[BucketTradeResult]:
    """Cap exposure per (station, target, target_date). 0 = no cap.

    Without this, a single forecast we're confident about can put 3-5
    bets on the same event, which concentrates risk on one weather call.
    """
    if cap_usd <= 0:
        return list(results)

    by_event: dict[tuple[str, str, str], list[BucketTradeResult]] = {}
    for r in results:
        by_event.setdefault((r.station_id, r.target, r.target_date), []).append(r)

    out: list[BucketTradeResult] = []
    for trades in by_event.values():
        out.extend(t for t in trades if t.side is None)
        actionable = sorted(
            (t for t in trades if t.side is not None),
            key=lambda t: t.edge or 0.0,
            reverse=True,
        )
        running = 0.0
        for t in actionable:
            if running + t.position_usd <= cap_usd + 1e-9:
                out.append(t)
                running += t.position_usd
    return out


def cap_daily_exposure(
    results: Iterable[BucketTradeResult],
    daily_cap_usd: float,
) -> list[BucketTradeResult]:
    """Optional per-target-date exposure cap.

    `daily_cap_usd <= 0` → no cap, all trades pass through (paper-trade
    research mode — useful when you want every trade as a calibration data
    point regardless of bankroll constraints).

    `daily_cap_usd > 0` → drop the lowest-edge actionable trades on each
    target_date until that day's total exposure fits within the cap. Mimics
    the execution layer's `max_total_exposure_usd` constraint, applied
    per-day since each target_date's markets resolve independently.
    """
    if daily_cap_usd <= 0:
        return list(results)

    by_date: dict[str, list[BucketTradeResult]] = {}
    for r in results:
        by_date.setdefault(r.target_date, []).append(r)

    out: list[BucketTradeResult] = []
    for day_trades in by_date.values():
        out.extend(t for t in day_trades if t.side is None)
        actionable = sorted(
            (t for t in day_trades if t.side is not None),
            key=lambda t: t.edge or 0.0,
            reverse=True,
        )
        running = 0.0
        for t in actionable:
            if running + t.position_usd <= daily_cap_usd + 1e-9:
                out.append(t)
                running += t.position_usd
            # else: dropped (over-cap); the simulation pretends we didn't take it
    return out


def aggregate(results: Iterable[BucketTradeResult]) -> PnLSummary:
    actionable = [r for r in results if r.side is not None]
    if not actionable:
        return PnLSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, None, None)
    resolved = [r for r in actionable if r.won is not None]
    total_pos = sum(r.position_usd for r in actionable)
    resolved_pos = sum(r.position_usd for r in resolved)
    total_profit = sum(r.profit_usd for r in resolved)
    wins = sum(1 for r in resolved if r.profit_usd > 0)
    return PnLSummary(
        n_trades=len(actionable),
        n_resolved=len(resolved),
        n_pending=len(actionable) - len(resolved),
        n_wins=wins,
        n_losses=len(resolved) - wins,
        total_pos_usd=total_pos,
        resolved_pos_usd=resolved_pos,
        total_profit_usd=total_profit,
        win_rate=(wins / len(resolved)) if resolved else None,
        roi_pct=(total_profit / resolved_pos * 100.0) if resolved_pos > 0 else None,
    )
