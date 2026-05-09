"""Backtest: take-profit when market converges to our_prob.

Three exit policies tested:
  A. Baseline (current): TP only fires when bid - our_prob >= 0.05
  B. Convergence-only: exit when bid - our_prob >= -tolerance
  C. Combined: A OR B (whichever fires first)

For each policy, computes total realized P&L, mean position duration
(in snapshots), and trade count. Goal: does convergence-exit free up
capital faster without sacrificing total P&L?
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from dataclasses import dataclass

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _decide_side, _rounded_observation, bucket_won
from weather_bot.positions import (
    Position, PositionEvent, _build_ladder_prices, _market_yes_mid,
    _maker_fill_price_at, recompute_our_prob,
    DEFAULT_TICK_SIZE, DEFAULT_MIN_SHARES, DEFAULT_N_RUNGS,
)
from weather_bot.sizing import position_size_usd


@dataclass
class ExitPolicy:
    name: str
    tp_threshold: float          # baseline take-profit (overshoot)
    sl_threshold: float          # absolute EV stop loss
    sl_pct: float                # relative MTM stop loss
    convergence_tolerance: float | None  # exit if bid >= our_prob - tol
                                          # None = no convergence rule


def decide_close_with_policy(
    pos, snap, our_prob, policy: ExitPolicy
) -> tuple[str, float] | None:
    if pos.side == "YES":
        bid = snap.yes_bid
        prob = our_prob
    else:
        bid = (1.0 - snap.yes_ask) if snap.yes_ask is not None else None
        prob = 1.0 - our_prob
    if bid is None or bid <= 0.0 or bid >= 1.0:
        return None
    bid = float(bid)
    ev_hold = prob - pos.entry_price
    ev_sell = bid - pos.entry_price

    # Baseline take-profit (overshoot)
    if ev_sell >= ev_hold + policy.tp_threshold:
        return "sell_take_profit", bid
    # Convergence rule: exit when market matches our_prob within tolerance
    if policy.convergence_tolerance is not None:
        # bid >= prob - tolerance means market priced AT or ABOVE our view
        if bid >= prob - policy.convergence_tolerance and ev_sell > 0:
            return "sell_convergence", bid
    # Stop loss A: absolute EV
    if ev_hold <= policy.sl_threshold:
        return "sell_stop_loss", bid
    # Stop loss B: relative MTM
    if pos.entry_price > 0:
        mtm_loss = (pos.entry_price - bid) / pos.entry_price
        if mtm_loss >= policy.sl_pct:
            return "sell_stop_loss", bid
    return None


def replay_with_policy(records, policy: ExitPolicy, *, sigma=1.4) -> list:
    buckets = defaultdict(list)
    for r in records:
        if r.bucket_snapshots is None:
            continue
        for snap in r.bucket_snapshots:
            key = (r.station_id, r.target, r.target_date.isoformat(),
                   snap.kind, snap.threshold)
            buckets[key].append((r, snap))
    for key in buckets:
        buckets[key].sort(key=lambda rs: rs[0].issue_time_utc)

    positions = []
    for key, series in buckets.items():
        if not series:
            continue
        station = STATIONS_BY_ID.get(series[0][0].station_id)
        if station is None:
            continue
        rungs = None
        ladder_side = None

        for record, snap in series:
            our_prob = recompute_our_prob(
                record, snap, station.unit, sigma_inflation_factor=sigma
            )
            if rungs is None:
                decision = _decide_side(our_prob, snap.yes_bid, snap.yes_ask, 0.05)
                if decision is None:
                    continue
                side, edge, fill_price = decision
                if edge > 0.25:
                    continue
                if not (0.05 <= fill_price <= 0.95):
                    continue
                total_size = position_size_usd(
                    our_prob, fill_price, side,
                    bankroll_usd=1000, kelly_multiplier=0.1,
                    max_position_usd=50,
                    liquidity_cap_usd=snap.volume_24hr * 0.1,
                )
                if total_size <= 0:
                    continue
                limit_prices = _build_ladder_prices(snap, side, DEFAULT_N_RUNGS, DEFAULT_TICK_SIZE)
                if not limit_prices:
                    continue
                per_rung = total_size / len(limit_prices)
                placed = []
                for limit in limit_prices:
                    if limit <= 0 or limit >= 1:
                        continue
                    shares = per_rung / limit
                    if shares < DEFAULT_MIN_SHARES:
                        continue
                    placed.append({"limit": limit, "size_usd": per_rung,
                                   "shares": shares, "position": None,
                                   "open_snap_idx": None})
                if not placed:
                    continue
                rungs = placed
                ladder_side = side
                continue

            for rung in rungs:
                if rung["position"] is None:
                    if _maker_fill_price_at(ladder_side, rung["limit"], snap):
                        idx = len(rungs[0].get("snap_history", [0]))  # not used
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
                    pos = rung["position"]
                    if pos.closed:
                        continue
                    decision = decide_close_with_policy(pos, snap, our_prob, policy)
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
            continue
        last_record, last_snap = series[-1]
        last_prob = recompute_our_prob(
            last_record, last_snap, station.unit, sigma_inflation_factor=sigma
        )
        for rung in rungs:
            if rung["position"] is not None:
                pos = rung["position"]
                if pos.closed:
                    positions.append(pos)
                    continue
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
                positions.append(pos)
    return positions


def report(label: str, positions) -> None:
    n = len(positions)
    n_resolved = sum(1 for p in positions if p.closed)
    pnl = sum(p.realized_profit_usd for p in positions if p.closed)
    expo = sum(p.position_usd for p in positions)
    res_expo = sum(p.position_usd for p in positions if p.closed)
    n_w = sum(1 for p in positions if p.closed and p.realized_profit_usd > 0)
    n_l = sum(1 for p in positions if p.closed and p.realized_profit_usd < 0)
    n_tp = sum(1 for p in positions if p.closed and p.events
               and p.events[-1].action == "sell_take_profit")
    n_conv = sum(1 for p in positions if p.closed and p.events
                 and p.events[-1].action == "sell_convergence")
    n_sl = sum(1 for p in positions if p.closed and p.events
               and p.events[-1].action == "sell_stop_loss")
    n_exp = sum(1 for p in positions if p.closed and p.events
                and p.events[-1].action == "expire")
    avg_events = sum(len(p.events) for p in positions) / max(1, n)
    wr = n_w / max(1, n_w + n_l)
    print(f"{label:42s} n={n:4d}  R={n_resolved:4d}  W={n_w:3d}  L={n_l:3d}  "
          f"wr={wr*100:5.1f}%  PnL=${pnl:+8.2f}  rexp=${res_expo:6.0f}")
    print(f"{'  exits:':42s} TP={n_tp:3d}  CONV={n_conv:3d}  SL={n_sl:3d}  "
          f"EXP={n_exp:3d}  avg_events={avg_events:.1f}")


recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
print(f"records: {len(recs)}, eligible: {len(elig)}\n")

# Baseline: current TP/SL/expire
baseline = ExitPolicy("baseline", 0.05, -0.10, 0.50, None)
report("Baseline (TP=overshoot 5pp, no convergence)", replay_with_policy(elig, baseline))
print()

# Convergence-only sweep
for tol in [0.00, 0.02, 0.05]:
    pol = ExitPolicy(f"conv {tol}", 0.05, -0.10, 0.50, tol)
    report(f"Combined: TP=5pp + convergence at tol={tol}", replay_with_policy(elig, pol))
    print()

# Convergence-only (no overshoot TP at all)
for tol in [0.00, 0.02, 0.05]:
    pol = ExitPolicy(f"pure conv {tol}", 99.0, -0.10, 0.50, tol)
    report(f"Pure convergence only at tol={tol} (no overshoot TP)", replay_with_policy(elig, pol))
    print()
