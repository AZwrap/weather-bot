"""Backtest v2: combine three fixes the user proposed.

1. Price filter relaxed to [0.001, 0.999] — let the model take all trades.
2. For inverted YES-tail bets, fire a SINGLE limit at NO-ask (taker-style).
   No maker ladder. Fills immediately or near-immediately.
3. Override Kelly sizing for inverted bets with fixed $5 each, since
   model-derived sizing rejects them by construction.

Compares against baseline (maker, no inversion) and skip-tails-only.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from typing import Iterable

from weather_bot.forward_log import ForwardLogRecord, load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _decide_side, _rounded_observation, bucket_won
from weather_bot.positions import (
    Position, PositionEvent, _build_ladder_prices, _decide_close,
    _market_yes_mid, _maker_fill_price_at, recompute_our_prob, summarize,
    DEFAULT_TICK_SIZE, DEFAULT_MIN_SHARES, DEFAULT_N_RUNGS,
    replay_maker,
)
from weather_bot.sizing import position_size_usd

# Wide price filter
MIN_PRICE = 0.001
MAX_PRICE = 0.999


def replay_with_inversion(
    records: Iterable[ForwardLogRecord],
    *,
    invert_yes_below: float,
    invert_no_above: float,
    inverted_size_usd: float,
    bankroll_usd: float = 1000.0,
    kelly_multiplier: float = 0.1,
    max_position_usd: float = 50.0,
    min_edge: float = 0.05,
    max_edge: float = 0.25,
    n_rungs: int = DEFAULT_N_RUNGS,
    tick_size: float = DEFAULT_TICK_SIZE,
    min_shares: int = DEFAULT_MIN_SHARES,
    sigma_inflation_factor: float = 1.4,
    take_profit_threshold: float = 0.05,
    stop_loss_threshold: float = -0.10,
    stop_loss_pct: float = 0.50,
    skip_only: bool = False,  # if True, just skip tail bets, no inversion
) -> tuple[list[Position], int, int]:
    """Returns (positions, n_inverted, n_skipped)."""
    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r.bucket_snapshots is None:
            continue
        for snap in r.bucket_snapshots:
            key = (r.station_id, r.target, r.target_date.isoformat(),
                   snap.kind, snap.threshold)
            buckets[key].append((r, snap))

    positions: list[Position] = []
    n_inverted = 0
    n_skipped = 0

    for key, series in buckets.items():
        series.sort(key=lambda rs: rs[0].issue_time_utc)
        if not series:
            continue
        station = STATIONS_BY_ID.get(series[0][0].station_id)
        if station is None:
            continue

        rungs = None
        ladder_side = None

        for record, snap in series:
            our_prob = recompute_our_prob(
                record, snap, station.unit,
                sigma_inflation_factor=sigma_inflation_factor,
            )

            if rungs is None:
                decision = _decide_side(our_prob, snap.yes_bid, snap.yes_ask, min_edge)
                if decision is None:
                    continue
                side, edge, fill_price = decision
                if edge > max_edge:
                    continue
                if not (MIN_PRICE <= fill_price <= MAX_PRICE):
                    continue

                # Should this be inverted?
                inverted = False
                if side == "YES" and fill_price < invert_yes_below:
                    inverted = True
                    side = "NO"
                    if snap.yes_bid is None:
                        n_skipped += 1
                        continue
                    fill_price = 1.0 - float(snap.yes_bid)
                elif side == "NO" and fill_price > invert_no_above:
                    inverted = True
                    side = "YES"
                    if snap.yes_ask is None:
                        n_skipped += 1
                        continue
                    fill_price = float(snap.yes_ask)

                if not (MIN_PRICE <= fill_price <= MAX_PRICE):
                    if inverted:
                        n_skipped += 1
                    continue

                # Skip-only mode: don't trade tail bets at all
                if skip_only and inverted:
                    n_skipped += 1
                    continue

                if inverted:
                    # Aggressive single-rung limit at the ask: taker-style
                    # fill. Bypass Kelly with fixed sizing.
                    total_size = inverted_size_usd
                    limit_prices = [fill_price]
                    n_inverted += 1
                else:
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
                    })
                if not placed:
                    continue

                rungs = placed
                ladder_side = side
                continue

            for rung in rungs:
                if rung["position"] is None:
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
            continue
        last_record, last_snap = series[-1]
        last_prob = recompute_our_prob(
            last_record, last_snap, station.unit,
            sigma_inflation_factor=sigma_inflation_factor,
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
    return positions, n_inverted, n_skipped


def report(label: str, positions: list[Position]) -> None:
    n_resolved = sum(1 for p in positions if p.closed)
    n_w = sum(1 for p in positions if p.closed and p.realized_profit_usd > 0)
    n_l = sum(1 for p in positions if p.closed and p.realized_profit_usd < 0)
    pnl = sum(p.realized_profit_usd for p in positions if p.closed)
    expo = sum(p.position_usd for p in positions)
    res_expo = sum(p.position_usd for p in positions if p.closed)
    wr = n_w / max(1, n_w + n_l)
    print(f"{label:62s} n={len(positions):4d}  R={n_resolved:4d}  "
          f"W={n_w:3d}  L={n_l:3d}  wr={wr*100:5.1f}%  "
          f"PnL=${pnl:+8.2f}  expo=${expo:7.0f}")


recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
print(f"records: {len(recs)}, eligible: {len(elig)}\n")

# --- Baselines ---
base = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
report("baseline maker (filter 0.05-0.95, no inversion)", base)

base_wide = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.001, max_yes_price=0.999,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
report("baseline maker (filter 0.001-0.999, no inversion)", base_wide)

print()

# --- Skip-only sweeps (no inversion, just skip tails) ---
for thr in [0.10, 0.20, 0.30]:
    pos, ni, ns = replay_with_inversion(
        elig, invert_yes_below=thr, invert_no_above=1 - thr,
        inverted_size_usd=5.0, skip_only=True,
    )
    report(f"SKIP only YES<{thr:.2f} / NO>{1-thr:.2f}  (skipped={ns})", pos)

print()

# --- Invert with fixed-size taker ---
for thr in [0.10, 0.20, 0.30]:
    for inv_size in [2.0, 5.0, 10.0]:
        pos, ni, ns = replay_with_inversion(
            elig, invert_yes_below=thr, invert_no_above=1 - thr,
            inverted_size_usd=inv_size,
        )
        report(f"INVERT taker YES<{thr:.2f} ${inv_size:.0f}/inv  (n_inv={ni})", pos)
    print()

# --- Diagnostic on the inverted positions in YES<0.20 / $5 case ---
print("=== INVERTED POSITIONS DETAIL (YES<0.20, $5 each, taker) ===")
pos_inv, _, _ = replay_with_inversion(
    elig, invert_yes_below=0.20, invert_no_above=0.80, inverted_size_usd=5.0,
)
# heuristic: inverted positions are the ones where side is NO with high
# entry price, OR side is YES with low entry price.
inv_pos = [
    p for p in pos_inv
    if (p.side == "NO" and p.entry_price > 0.80)
    or (p.side == "YES" and p.entry_price < 0.20)
]
print(f"identified inverted positions: n={len(inv_pos)}")
if inv_pos:
    n_w = sum(1 for p in inv_pos if p.closed and p.realized_profit_usd > 0)
    n_l = sum(1 for p in inv_pos if p.closed and p.realized_profit_usd < 0)
    pnl = sum(p.realized_profit_usd for p in inv_pos if p.closed)
    print(f"  resolved: {n_w + n_l}, W={n_w}, L={n_l}, wr={n_w/max(1,n_w+n_l)*100:.1f}%, PnL=${pnl:+.2f}")
    print(f"  avg entry: ${sum(p.entry_price for p in inv_pos)/len(inv_pos):.3f}")
    print(f"  by side:")
    for s in ["YES", "NO"]:
        ss = [p for p in inv_pos if p.side == s]
        if ss:
            sw = sum(1 for p in ss if p.closed and p.realized_profit_usd > 0)
            sl = sum(1 for p in ss if p.closed and p.realized_profit_usd < 0)
            spnl = sum(p.realized_profit_usd for p in ss if p.closed)
            print(f"    {s}: n={len(ss)}, W={sw}, L={sl}, PnL=${spnl:+.2f}")
