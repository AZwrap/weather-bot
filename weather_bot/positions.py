"""Position simulator: replays the multi-snapshot forward log and applies
EV-based open / hold / sell decisions with churn protection.

Why this exists
---------------
The plain `pnl.py` simulator treats each forward-log record as an independent
one-shot bet. That ignores two things real markets do continuously:

  1. Markets reprice. A bucket we entered at 0.001 might be 0.215 an hour
     later. Holding to expiration vs. selling at the new bid have different
     EVs — we should pick the higher-EV branch.
  2. Our own forecast updates. ECMWF re-runs every 6 hours; bias correction
     is constant but the underlying ensemble shifts. Sometimes our `our_prob`
     drops below the price we paid, flipping EV(hold) negative.

This module simulates a paper portfolio over the time-series of snapshots:
  * On the FIRST snapshot of a (station, target, target_date, bucket), if
    edge > min_edge and the bucket isn't already held, OPEN a position with
    deci-Kelly sizing.
  * On EVERY subsequent snapshot, re-evaluate held positions:
        EV(hold) = our_prob_now × $1 − entry_price            (per share)
        EV(sell) = current_bid    − entry_price                (per share)
        sell if EV(sell) ≥ EV(hold) + threshold (take-profit) or
                EV(hold) drops below stop-loss threshold.
  * Final resolution P&L if not already closed.

Churn protection: once a position closes, do not re-enter the SAME bucket
unless our_prob has improved by `re_entry_threshold` (default 0.05) over
the closing price. Stops the bot from cycling in/out of noisy markets.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Literal

import numpy as np

from .forecast.probability import TempDistribution, bucket_prob
from .forward_log import BucketSnapshot, ForwardLogRecord
from .locations import STATIONS_BY_ID
from .pnl import _decide_side, _rounded_observation, bucket_won
from .sizing import position_size_usd
from .units import Unit

Side = Literal["YES", "NO"]
Action = Literal["open", "hold", "sell_take_profit", "sell_stop_loss", "expire"]


@dataclass
class PositionEvent:
    """One step in a position's life cycle."""

    action: Action
    issue_time_utc: str
    fill_price: float | None      # price at the action (for sells = bid)
    shares: float
    cash_flow_usd: float           # negative on open, positive on sell/win, 0 on hold
    our_prob_at_step: float
    market_yes_implied_at_step: float | None


@dataclass
class Position:
    station_id: str
    target: str
    target_date: str
    bucket_label: str
    bucket_kind: str
    threshold: int
    side: Side
    entry_price: float
    shares: float
    open_event: PositionEvent
    events: list[PositionEvent] = field(default_factory=list)
    closed: bool = False
    realized_profit_usd: float = 0.0

    @property
    def position_usd(self) -> float:
        return self.entry_price * self.shares

    @property
    def status(self) -> str:
        if not self.closed:
            return "open"
        for ev in reversed(self.events):
            if ev.action.startswith("sell") or ev.action == "expire":
                return ev.action
        return "closed"


# ──────────────────────────────────────────────────────────────────────────
# Probability recomputation (so the σ slider is faithful)
# ──────────────────────────────────────────────────────────────────────────


def recompute_our_prob(
    record: ForwardLogRecord,
    snap: BucketSnapshot,
    unit: Unit,
    *,
    sigma_inflation_factor: float = 1.4,
    n_resample: int = 10,
    rng_seed: int = 0,
) -> float:
    """Recompute the bucket probability from raw_members at a chosen σ factor.

    The position simulator used to read the cached `snap.our_prob` (frozen at
    log time with σ_factor=1.4), which made the σ slider in the dashboard
    inert for the Positions tab. This recomputation closes that loop so the
    slider is genuinely live.

    Steps mirror `bias.predictive_members` but use the bias and σ_residual
    that were FROZEN ON THE RECORD (not the current BiasTable), so we replay
    the historical state — not retroactively rewrite history with new bias.
    """
    members = np.asarray(record.raw_members_c, dtype=float) - record.bias_applied_c

    if record.sigma_residual_c > 0 and sigma_inflation_factor > 0:
        sigma = record.sigma_residual_c * float(sigma_inflation_factor)
        rng = np.random.default_rng(rng_seed)
        tiled = np.tile(members, max(1, n_resample))
        noise = rng.normal(0.0, sigma, size=tiled.shape)
        samples = tiled + noise
    else:
        samples = members

    dist = TempDistribution(
        location_name=record.station_id,
        target_date=record.target_date,
        members=samples,
    )
    return bucket_prob(dist, snap.kind, snap.threshold, unit)


# ──────────────────────────────────────────────────────────────────────────
# Decision logic
# ──────────────────────────────────────────────────────────────────────────


def _ev_per_share(our_prob: float, entry_price: float) -> float:
    """E[profit per share if we hold to expiration]."""
    return our_prob - entry_price  # win=$1−entry with prob p, lose=−entry with prob 1−p


def _ev_per_share_sell(bid: float, entry_price: float) -> float:
    """Realized profit per share if we sell at the bid."""
    return bid - entry_price


def _decide_close(
    pos: Position,
    snap: BucketSnapshot,
    our_prob: float,
    *,
    take_profit_threshold: float,
    stop_loss_threshold: float,
    stop_loss_pct: float,
) -> tuple[Action, float] | None:
    """Decide whether to close `pos` given the current snapshot.

    Two stop-loss conditions, OR'd together (whichever fires first):

      1. Absolute EV stop:  EV(hold)  ≤  stop_loss_threshold  (e.g. -0.10)
         Catches mid-priced bets (entry ~0.30–0.70) where absolute EV can
         move into negative territory while still being well above
         maximum-possible-loss (-entry_price).

      2. Relative MTM stop:  (entry − bid) / entry  ≥  stop_loss_pct (e.g. 0.50)
         Catches low-entry-price tail bets (entry < 0.20) where the
         absolute stop-loss is structurally unreachable because the max
         possible loss (-entry) is shallower than the threshold.

    `our_prob` is the FRESHLY-RECOMPUTED bucket probability (caller's choice
    of σ factor), not `snap.our_prob`. Returns (action, sell_price) or None
    to keep holding.
    """
    if pos.side == "YES":
        bid = snap.yes_bid
        prob = our_prob
    else:
        bid = (1.0 - snap.yes_ask) if snap.yes_ask is not None else None
        prob = 1.0 - our_prob

    if bid is None or bid <= 0.0 or bid >= 1.0:
        return None

    ev_hold = _ev_per_share(prob, pos.entry_price)
    ev_sell = _ev_per_share_sell(bid, pos.entry_price)

    # Take profit: realized P&L noticeably exceeds our model's hold EV
    if ev_sell >= ev_hold + take_profit_threshold:
        return "sell_take_profit", bid

    # Stop loss A: absolute EV floor
    if ev_hold <= stop_loss_threshold:
        return "sell_stop_loss", bid

    # Stop loss B: relative MTM loss as a fraction of entry price
    if pos.entry_price > 0:
        mtm_loss_pct = (pos.entry_price - bid) / pos.entry_price
        if mtm_loss_pct >= stop_loss_pct:
            return "sell_stop_loss", bid

    return None


# ──────────────────────────────────────────────────────────────────────────
# Replay
# ──────────────────────────────────────────────────────────────────────────


def _bucket_key(snap: BucketSnapshot) -> tuple[str, int]:
    return snap.kind, snap.threshold


def replay(
    records: Iterable[ForwardLogRecord],
    *,
    bankroll_usd: float = 1000.0,
    kelly_multiplier: float = 0.1,
    max_position_usd: float = 50.0,
    min_edge: float = 0.05,
    max_edge: float = 0.25,
    min_yes_price: float = 0.05,
    max_yes_price: float = 0.95,
    liquidity_cap_fraction: float = 0.1,
    take_profit_threshold: float = 0.05,
    stop_loss_threshold: float = -0.10,
    stop_loss_pct: float = 0.50,
    re_entry_threshold: float = 0.05,
    sigma_inflation_factor: float = 1.4,
) -> list[Position]:
    """Walk every snapshot in chronological order, opening / holding / closing
    positions per the EV-based rules above.

    `sigma_inflation_factor` is applied LIVE during replay — `our_prob` is
    recomputed from raw ensemble members for every (record, bucket) pair, so
    moving the σ slider in the dashboard genuinely changes which positions
    open and close.

    Returns every Position that was ever opened (including currently-held ones).
    """
    # Group records by event slot, then walk chronologically.
    buckets: dict[tuple[str, str, str, str, int], list[tuple[ForwardLogRecord, BucketSnapshot]]] = defaultdict(list)
    for r in records:
        if r.bucket_snapshots is None:
            continue
        for snap in r.bucket_snapshots:
            key = (r.station_id, r.target, r.target_date.isoformat(), snap.kind, snap.threshold)
            buckets[key].append((r, snap))

    positions: list[Position] = []
    for key, series in buckets.items():
        series.sort(key=lambda rs: rs[0].issue_time_utc)
        positions.extend(_replay_bucket(
            series,
            bankroll_usd=bankroll_usd,
            kelly_multiplier=kelly_multiplier,
            max_position_usd=max_position_usd,
            min_edge=min_edge,
            max_edge=max_edge,
            min_yes_price=min_yes_price,
            max_yes_price=max_yes_price,
            liquidity_cap_fraction=liquidity_cap_fraction,
            take_profit_threshold=take_profit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            stop_loss_pct=stop_loss_pct,
            re_entry_threshold=re_entry_threshold,
            sigma_inflation_factor=sigma_inflation_factor,
        ))
    return positions


def _replay_bucket(
    series: list[tuple[ForwardLogRecord, BucketSnapshot]],
    *,
    bankroll_usd: float,
    kelly_multiplier: float,
    max_position_usd: float,
    min_edge: float,
    max_edge: float,
    min_yes_price: float,
    max_yes_price: float,
    liquidity_cap_fraction: float,
    take_profit_threshold: float,
    stop_loss_threshold: float,
    stop_loss_pct: float,
    re_entry_threshold: float,
    sigma_inflation_factor: float,
) -> list[Position]:
    """Replay one bucket's snapshot timeline."""
    if not series:
        return []
    station = STATIONS_BY_ID.get(series[0][0].station_id)
    if station is None:
        return []

    out: list[Position] = []
    open_pos: Position | None = None
    last_close_price: float | None = None

    for record, snap in series:
        # Recompute our_prob fresh at the chosen σ factor — this is what
        # makes the σ slider actually live for the position simulator.
        our_prob = recompute_our_prob(
            record, snap, station.unit,
            sigma_inflation_factor=sigma_inflation_factor,
        )

        # 1) If we hold a position, evaluate close
        if open_pos is not None:
            decision = _decide_close(
                open_pos, snap, our_prob,
                take_profit_threshold=take_profit_threshold,
                stop_loss_threshold=stop_loss_threshold,
                stop_loss_pct=stop_loss_pct,
            )
            if decision is not None:
                action, sell_price = decision
                proceeds = sell_price * open_pos.shares
                cost_basis = open_pos.entry_price * open_pos.shares
                profit = proceeds - cost_basis
                ev = PositionEvent(
                    action=action,
                    issue_time_utc=record.issue_time_utc.isoformat(),
                    fill_price=sell_price,
                    shares=open_pos.shares,
                    cash_flow_usd=proceeds,
                    our_prob_at_step=our_prob,
                    market_yes_implied_at_step=_market_yes_mid(snap),
                )
                open_pos.events.append(ev)
                open_pos.closed = True
                open_pos.realized_profit_usd = profit
                last_close_price = sell_price if open_pos.side == "YES" else 1.0 - sell_price
                out.append(open_pos)
                open_pos = None

        # 2) If we have no position, consider opening
        if open_pos is None:
            decision = _decide_side(our_prob, snap.yes_bid, snap.yes_ask, min_edge)
            if decision is None:
                continue
            side, edge, fill_price = decision
            if edge > max_edge:
                continue
            if not (min_yes_price <= fill_price <= max_yes_price):
                continue
            # Churn protection: don't re-enter immediately after a close
            # unless our prob has improved meaningfully past the prior price.
            if last_close_price is not None:
                prob_for_side = our_prob if side == "YES" else 1.0 - our_prob
                if prob_for_side - last_close_price < re_entry_threshold:
                    continue
            pos_usd = position_size_usd(
                our_prob, fill_price, side,
                bankroll_usd=bankroll_usd,
                kelly_multiplier=kelly_multiplier,
                max_position_usd=max_position_usd,
                liquidity_cap_usd=snap.volume_24hr * liquidity_cap_fraction,
            )
            if pos_usd <= 0:
                continue
            shares = pos_usd / fill_price
            ev = PositionEvent(
                action="open",
                issue_time_utc=record.issue_time_utc.isoformat(),
                fill_price=fill_price,
                shares=shares,
                cash_flow_usd=-pos_usd,
                our_prob_at_step=our_prob,
                market_yes_implied_at_step=_market_yes_mid(snap),
            )
            open_pos = Position(
                station_id=record.station_id,
                target=record.target,
                target_date=record.target_date.isoformat(),
                bucket_label=snap.bucket_label,
                bucket_kind=snap.kind,
                threshold=snap.threshold,
                side=side,
                entry_price=fill_price,
                shares=shares,
                open_event=ev,
            )
            open_pos.events.append(ev)

    # 3) End of timeline. If still holding, mark expire and compute realised
    # P&L when the underlying record has actual_obs_c.
    if open_pos is not None:
        last_record, last_snap = series[-1]
        last_prob = recompute_our_prob(
            last_record, last_snap, station.unit,
            sigma_inflation_factor=sigma_inflation_factor,
        )
        if last_record.actual_obs_c is not None:
            actual_int = _rounded_observation(
                last_record.actual_obs_c, station.unit
            )
            won_yes = bucket_won(last_snap, actual_int, station.unit)
            won = won_yes if open_pos.side == "YES" else not won_yes
            payoff = open_pos.shares if won else 0.0
            cost_basis = open_pos.entry_price * open_pos.shares
            ev = PositionEvent(
                action="expire",
                issue_time_utc=last_record.issue_time_utc.isoformat(),
                fill_price=1.0 if won else 0.0,
                shares=open_pos.shares,
                cash_flow_usd=payoff,
                our_prob_at_step=last_prob,
                market_yes_implied_at_step=_market_yes_mid(last_snap),
            )
            open_pos.events.append(ev)
            open_pos.closed = True
            open_pos.realized_profit_usd = payoff - cost_basis
        out.append(open_pos)
    return out


def _market_yes_mid(snap: BucketSnapshot) -> float | None:
    if snap.yes_bid is not None and snap.yes_ask is not None:
        return (float(snap.yes_bid) + float(snap.yes_ask)) / 2
    return snap.yes_last


# ──────────────────────────────────────────────────────────────────────────
# Maker-ladder simulator — canonical execution strategy
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_TICK_SIZE = 0.001
DEFAULT_MIN_SHARES = 5
DEFAULT_N_RUNGS = 4


def _build_ladder_prices(
    snap: BucketSnapshot,
    side: Side,
    n_rungs: int,
    tick: float,
) -> list[float]:
    """Generate evenly-spaced limit prices INSIDE the bid-ask spread.

    For YES: prices in [yes_bid+tick, yes_ask-tick], snapped to tick grid.
    For NO:  same idea in NO-space (NO ask = 1-yes_bid, NO bid = 1-yes_ask).

    Returns [] if spread is too tight to place any rung inside.
    """
    if snap.yes_bid is None or snap.yes_ask is None:
        return []

    if side == "YES":
        lo = float(snap.yes_bid) + tick
        hi = float(snap.yes_ask) - tick
    else:
        # NO side: NO ask = 1 - yes_bid, NO bid = 1 - yes_ask
        lo = (1.0 - float(snap.yes_ask)) + tick
        hi = (1.0 - float(snap.yes_bid)) - tick

    if lo > hi:
        return []  # spread is exactly 1 tick; nowhere to place a maker rung
    if hi - lo < tick / 2:
        return [round(lo / tick) * tick]

    if n_rungs == 1:
        return [round(((lo + hi) / 2) / tick) * tick]

    # Evenly spaced in [lo, hi], snapped to grid, deduplicated
    raw = np.linspace(lo, hi, n_rungs)
    snapped = sorted({round(round(p / tick) * tick, 6) for p in raw})
    return snapped


def _maker_fill_price_at(side: Side, limit_price: float, snap: BucketSnapshot) -> bool:
    """Did our resting limit fill at this snapshot's market state?

    For YES buy at L: filled iff yes_ask ≤ L (someone willing to sell at L or below).
    For NO  buy at L: filled iff (1 - yes_bid) ≤ L (someone willing to sell NO at L).
    """
    if side == "YES":
        return snap.yes_ask is not None and float(snap.yes_ask) <= limit_price
    return snap.yes_bid is not None and (1.0 - float(snap.yes_bid)) <= limit_price


def _replay_maker_bucket(
    series: list[tuple[ForwardLogRecord, BucketSnapshot]],
    *,
    bankroll_usd: float,
    kelly_multiplier: float,
    max_position_usd: float,
    min_edge: float,
    max_edge: float,
    min_yes_price: float,
    max_yes_price: float,
    n_rungs: int,
    tick_size: float,
    min_shares: int,
    sigma_inflation_factor: float,
    take_profit_threshold: float,
    stop_loss_threshold: float,
    stop_loss_pct: float,
    taker_fallback: bool,
) -> list[Position]:
    """Maker-ladder replay for one bucket timeline.

    Steps per bucket:
      1. At first snapshot with edge, build ladder of n_rungs limit orders.
      2. Walk subsequent snapshots: rungs fill when market moves to their price.
      3. Each filled rung becomes a Position, runs the same hold/sell logic as taker mode.
      4. Optional: at the last snapshot, take any unfilled remainder if edge still positive.
    """
    if not series:
        return []
    station = STATIONS_BY_ID.get(series[0][0].station_id)
    if station is None:
        return []

    out: list[Position] = []
    # Each rung is tracked as: (limit_price, size_usd, position_or_none).
    # `position_or_none` is None until the rung fills, then a Position object.
    rungs: list[dict] | None = None  # set after first valid signal snapshot
    ladder_side: Side | None = None

    for record, snap in series:
        our_prob = recompute_our_prob(
            record, snap, station.unit,
            sigma_inflation_factor=sigma_inflation_factor,
        )

        # Step 1 — place ladder if we don't have one yet
        if rungs is None:
            decision = _decide_side(our_prob, snap.yes_bid, snap.yes_ask, min_edge)
            if decision is None:
                continue
            side, edge, fill_price = decision
            if edge > max_edge or not (min_yes_price <= fill_price <= max_yes_price):
                continue

            total_size = position_size_usd(
                our_prob, fill_price, side,
                bankroll_usd=bankroll_usd,
                kelly_multiplier=kelly_multiplier,
                max_position_usd=max_position_usd,
                liquidity_cap_usd=snap.volume_24hr * 0.1,
            )
            if total_size <= 0:
                continue

            limit_prices = _build_ladder_prices(snap, side, n_rungs, tick_size)
            if not limit_prices:
                continue

            # Equal-split sizing, drop rungs where shares < min_shares
            per_rung = total_size / len(limit_prices)
            placed: list[dict] = []
            for limit in limit_prices:
                if limit <= 0 or limit >= 1:
                    continue
                shares = per_rung / limit
                if shares < min_shares:
                    continue
                placed.append({
                    "limit": limit,
                    "size_usd": per_rung,
                    "shares": shares,
                    "position": None,
                    "open_record": record,
                })
            if not placed:
                continue
            rungs = placed
            ladder_side = side
            continue  # placed; check fills on next snapshot

        # Step 2 — check fills + manage filled rungs
        for rung in rungs:
            if rung["position"] is None:
                # Try to fill this rung
                if _maker_fill_price_at(ladder_side, rung["limit"], snap):
                    pos = Position(
                        station_id=record.station_id,
                        target=record.target,
                        target_date=record.target_date.isoformat(),
                        bucket_label=snap.bucket_label,
                        bucket_kind=snap.kind,
                        threshold=snap.threshold,
                        side=ladder_side,
                        entry_price=rung["limit"],
                        shares=rung["shares"],
                        open_event=PositionEvent(
                            action="open",
                            issue_time_utc=record.issue_time_utc.isoformat(),
                            fill_price=rung["limit"],
                            shares=rung["shares"],
                            cash_flow_usd=-rung["size_usd"],
                            our_prob_at_step=our_prob,
                            market_yes_implied_at_step=_market_yes_mid(snap),
                        ),
                    )
                    pos.events.append(pos.open_event)
                    rung["position"] = pos
            else:
                # Rung is filled and we hold a position. Apply hold/sell.
                pos = rung["position"]
                if pos.closed:
                    continue
                decision = _decide_close(
                    pos, snap, our_prob,
                    take_profit_threshold=take_profit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    stop_loss_pct=stop_loss_pct,
                )
                if decision is not None:
                    action, sell_price = decision
                    proceeds = sell_price * pos.shares
                    cost_basis = pos.entry_price * pos.shares
                    pos.events.append(PositionEvent(
                        action=action,
                        issue_time_utc=record.issue_time_utc.isoformat(),
                        fill_price=sell_price,
                        shares=pos.shares,
                        cash_flow_usd=proceeds,
                        our_prob_at_step=our_prob,
                        market_yes_implied_at_step=_market_yes_mid(snap),
                    ))
                    pos.closed = True
                    pos.realized_profit_usd = proceeds - cost_basis

    if rungs is None:
        return out

    # Step 3 — end of timeline. Settle filled rungs at expiration if resolved;
    # optionally take unfilled rungs at ask if taker_fallback enabled.
    last_record, last_snap = series[-1]
    last_prob = recompute_our_prob(
        last_record, last_snap, station.unit,
        sigma_inflation_factor=sigma_inflation_factor,
    )

    for rung in rungs:
        if rung["position"] is not None:
            pos = rung["position"]
            if pos.closed:
                out.append(pos)
                continue
            # Held to expiration — settle
            if last_record.actual_obs_c is not None:
                actual_int = _rounded_observation(last_record.actual_obs_c, station.unit)
                won_yes = bucket_won(last_snap, actual_int, station.unit)
                won = won_yes if pos.side == "YES" else not won_yes
                payoff = pos.shares if won else 0.0
                cost_basis = pos.entry_price * pos.shares
                pos.events.append(PositionEvent(
                    action="expire",
                    issue_time_utc=last_record.issue_time_utc.isoformat(),
                    fill_price=1.0 if won else 0.0,
                    shares=pos.shares,
                    cash_flow_usd=payoff,
                    our_prob_at_step=last_prob,
                    market_yes_implied_at_step=_market_yes_mid(last_snap),
                ))
                pos.closed = True
                pos.realized_profit_usd = payoff - cost_basis
            out.append(pos)
        elif taker_fallback:
            # Unfilled rung — fire taker at ask if edge still positive
            decision = _decide_side(last_prob, last_snap.yes_bid, last_snap.yes_ask, min_edge)
            if decision is None:
                continue  # no edge anymore; skip
            side, edge, taker_fill = decision
            if side != ladder_side:
                continue  # edge flipped sides; don't take
            shares = rung["size_usd"] / taker_fill
            if shares < min_shares:
                continue
            pos = Position(
                station_id=last_record.station_id,
                target=last_record.target,
                target_date=last_record.target_date.isoformat(),
                bucket_label=last_snap.bucket_label,
                bucket_kind=last_snap.kind,
                threshold=last_snap.threshold,
                side=side,
                entry_price=taker_fill,
                shares=shares,
                open_event=PositionEvent(
                    action="open",
                    issue_time_utc=last_record.issue_time_utc.isoformat(),
                    fill_price=taker_fill,
                    shares=shares,
                    cash_flow_usd=-rung["size_usd"],
                    our_prob_at_step=last_prob,
                    market_yes_implied_at_step=_market_yes_mid(last_snap),
                ),
            )
            pos.events.append(pos.open_event)
            # Settle if resolved
            if last_record.actual_obs_c is not None:
                actual_int = _rounded_observation(last_record.actual_obs_c, station.unit)
                won_yes = bucket_won(last_snap, actual_int, station.unit)
                won = won_yes if side == "YES" else not won_yes
                payoff = shares if won else 0.0
                cost_basis = taker_fill * shares
                pos.events.append(PositionEvent(
                    action="expire",
                    issue_time_utc=last_record.issue_time_utc.isoformat(),
                    fill_price=1.0 if won else 0.0,
                    shares=shares,
                    cash_flow_usd=payoff,
                    our_prob_at_step=last_prob,
                    market_yes_implied_at_step=_market_yes_mid(last_snap),
                ))
                pos.closed = True
                pos.realized_profit_usd = payoff - cost_basis
            out.append(pos)
    return out


def replay_maker(
    records: Iterable[ForwardLogRecord],
    *,
    bankroll_usd: float = 1000.0,
    kelly_multiplier: float = 0.1,
    max_position_usd: float = 50.0,
    min_edge: float = 0.05,
    max_edge: float = 0.25,
    min_yes_price: float = 0.05,
    max_yes_price: float = 0.95,
    n_rungs: int = DEFAULT_N_RUNGS,
    tick_size: float = DEFAULT_TICK_SIZE,
    min_shares: int = DEFAULT_MIN_SHARES,
    sigma_inflation_factor: float = 1.4,
    take_profit_threshold: float = 0.05,
    stop_loss_threshold: float = -0.10,
    stop_loss_pct: float = 0.50,
    taker_fallback: bool = False,
) -> list[Position]:
    """Maker-ladder replay over multi-snapshot forward log.

    Returns Position objects (one per filled rung). Unfilled rungs cost $0
    and don't appear in the result. Use `summarize()` to aggregate.

    `taker_fallback=True` enables the "if maker missed, fire taker at ask"
    rule at the last snapshot of each bucket. Off by default.
    """
    buckets: dict[
        tuple[str, str, str, str, int],
        list[tuple[ForwardLogRecord, BucketSnapshot]]
    ] = defaultdict(list)
    for r in records:
        if r.bucket_snapshots is None:
            continue
        for snap in r.bucket_snapshots:
            key = (r.station_id, r.target, r.target_date.isoformat(), snap.kind, snap.threshold)
            buckets[key].append((r, snap))

    positions: list[Position] = []
    for key, series in buckets.items():
        series.sort(key=lambda rs: rs[0].issue_time_utc)
        positions.extend(_replay_maker_bucket(
            series,
            bankroll_usd=bankroll_usd,
            kelly_multiplier=kelly_multiplier,
            max_position_usd=max_position_usd,
            min_edge=min_edge,
            max_edge=max_edge,
            min_yes_price=min_yes_price,
            max_yes_price=max_yes_price,
            n_rungs=n_rungs,
            tick_size=tick_size,
            min_shares=min_shares,
            sigma_inflation_factor=sigma_inflation_factor,
            take_profit_threshold=take_profit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            stop_loss_pct=stop_loss_pct,
            taker_fallback=taker_fallback,
        ))
    return positions


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PortfolioSummary:
    n_positions: int
    n_open: int
    n_take_profit: int
    n_stop_loss: int
    n_expire_won: int
    n_expire_lost: int
    total_invested_usd: float
    total_realized_pnl_usd: float
    open_exposure_usd: float


def summarize(positions: Iterable[Position]) -> PortfolioSummary:
    n_pos = 0
    n_open = n_tp = n_sl = n_ew = n_el = 0
    total_invested = 0.0
    total_realized = 0.0
    open_exp = 0.0
    for p in positions:
        n_pos += 1
        total_invested += p.position_usd
        if not p.closed:
            n_open += 1
            open_exp += p.position_usd
            continue
        total_realized += p.realized_profit_usd
        last = p.events[-1].action if p.events else ""
        if last == "sell_take_profit":
            n_tp += 1
        elif last == "sell_stop_loss":
            n_sl += 1
        elif last == "expire":
            if p.realized_profit_usd > 0:
                n_ew += 1
            else:
                n_el += 1
    return PortfolioSummary(
        n_positions=n_pos,
        n_open=n_open,
        n_take_profit=n_tp,
        n_stop_loss=n_sl,
        n_expire_won=n_ew,
        n_expire_lost=n_el,
        total_invested_usd=total_invested,
        total_realized_pnl_usd=total_realized,
        open_exposure_usd=open_exp,
    )
