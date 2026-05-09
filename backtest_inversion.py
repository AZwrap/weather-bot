"""Backtest: apply calibration-inversion to YES tail bets, see if it actually
makes money on May 8 data.

Rule (simple, structural — not fit to May 8 cohort outcomes):
    - If signal says BUY YES with entry_price < INVERT_THRESHOLD,
      flip the side to BUY NO instead.
    - Build the maker ladder on the inverted side; everything else is identical.

This re-runs the maker simulator with the side-flip baked in, so the fill
dynamics on the inverted side are real (not estimated from the original fills).
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from typing import Iterable

from weather_bot.forward_log import BucketSnapshot, ForwardLogRecord, load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _decide_side, _rounded_observation, bucket_won
from weather_bot.positions import (
    Position, PositionEvent, _build_ladder_prices, _decide_close,
    _market_yes_mid, _maker_fill_price_at, recompute_our_prob, summarize,
    DEFAULT_TICK_SIZE, DEFAULT_MIN_SHARES, DEFAULT_N_RUNGS,
)
from weather_bot.sizing import position_size_usd


def replay_maker_with_inversion(
    records: Iterable[ForwardLogRecord],
    *,
    invert_yes_below: float,  # invert YES bets with fill_price < this
    invert_no_above: float,   # invert NO bets with fill_price > this (e.g. 0.80)
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
                if not (min_yes_price <= fill_price <= max_yes_price):
                    continue

                # Inversion rule
                inverted = False
                if side == "YES" and fill_price < invert_yes_below:
                    side = "NO"
                    fill_price = (1.0 - float(snap.yes_bid)
                                  if snap.yes_bid is not None else None)
                    inverted = True
                elif side == "NO" and fill_price > invert_no_above:
                    side = "YES"
                    fill_price = (float(snap.yes_ask)
                                  if snap.yes_ask is not None else None)
                    inverted = True

                # Validity check. For INVERTED bets, the whole point is to
                # trade at extreme prices, so we bypass the symmetric
                # min_yes_price/max_yes_price filter — those were designed
                # for the original-signal case.
                if fill_price is None:
                    if inverted:
                        n_skipped += 1
                    continue
                if not inverted and not (min_yes_price <= fill_price <= max_yes_price):
                    continue
                if inverted and not (0.001 <= fill_price <= 0.999):
                    n_skipped += 1
                    continue

                # For inverted bets, the model is — by hypothesis — wrong.
                # Override our_prob with the inverted estimate before sizing,
                # so Kelly sees a positive-edge bet rather than the model's
                # negative-edge view. This is what the calibration-inversion
                # rule effectively does in production.
                sizing_prob = (1.0 - our_prob) if inverted else our_prob
                total_size = position_size_usd(
                    sizing_prob, fill_price, side,
                    bankroll_usd=bankroll_usd,
                    kelly_multiplier=kelly_multiplier,
                    max_position_usd=max_position_usd,
                    liquidity_cap_usd=snap.volume_24hr * 0.1,
                )
                if total_size <= 0:
                    if inverted:
                        n_skipped += 1
                    continue

                limit_prices = _build_ladder_prices(snap, side, n_rungs, tick_size)
                if not limit_prices:
                    if inverted:
                        n_skipped += 1
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
                    if inverted:
                        n_skipped += 1
                    continue

                rungs = placed
                ladder_side = side
                if inverted:
                    n_inverted += 1
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
    s = summarize(positions)
    n_resolved = sum(1 for p in positions if p.closed)
    n_w = sum(1 for p in positions if p.closed and p.realized_profit_usd > 0)
    n_l = sum(1 for p in positions if p.closed and p.realized_profit_usd < 0)
    pnl = sum(p.realized_profit_usd for p in positions if p.closed)
    expo = sum(p.position_usd for p in positions)
    res_expo = sum(p.position_usd for p in positions if p.closed)
    wr = n_w / max(1, n_w + n_l)
    print(f"{label:42s} pos={s.n_positions:5d}  resolved={n_resolved:4d}  "
          f"W={n_w:4d}  L={n_l:4d}  wr={wr*100:5.1f}%  "
          f"PnL=${pnl:+9.2f}  expo=${expo:8.0f}  rexp=${res_expo:7.0f}")


recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
print(f"records: {len(recs)}, eligible: {len(elig)}\n")

# Baseline: no inversion (= current bot behaviour)
from weather_bot.positions import replay_maker
base = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
report("baseline (no inversion)", base)

# Sweep invert-threshold
for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    pos, ni, ns = replay_maker_with_inversion(
        elig, invert_yes_below=thr, invert_no_above=1.0 - thr,
        bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
        min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
        sigma_inflation_factor=1.4,
    )
    report(f"MAKER  invert YES<{thr:.2f} / NO>{1-thr:.2f}  (n_inv={ni})", pos)

print()
# Taker-mode equivalent: replace _build_ladder with single rung AT THE ASK,
# so inverted bets always cross the spread and fill immediately (instead of
# waiting for adverse drift).
import weather_bot.positions as P
_orig_build = P._build_ladder_prices
def _taker_ladder(snap, side, n_rungs, tick):
    if snap.yes_bid is None or snap.yes_ask is None:
        return []
    if side == "YES":
        return [float(snap.yes_ask)]
    return [1.0 - float(snap.yes_bid)]

# Only the TAKER variant — patch in
P._build_ladder_prices = _taker_ladder
import backtest_inversion as M  # self-import to use the function with the patch
for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    pos, ni, ns = replay_maker_with_inversion(
        elig, invert_yes_below=thr, invert_no_above=1.0 - thr,
        bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
        min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
        sigma_inflation_factor=1.4,
        n_rungs=1,  # only one rung needed; it's AT the ask
    )
    report(f"TAKER  invert YES<{thr:.2f} / NO>{1-thr:.2f}  (n_inv={ni})", pos)
P._build_ladder_prices = _orig_build  # restore

# Diagnostic: of the positions opened in the inverted=0.20 case, which ones
# had ladder_side flipped? Are they the trades that fill?
pos_inv, _, _ = replay_maker_with_inversion(
    elig, invert_yes_below=0.20, invert_no_above=0.80,
    bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4,
)
print()
print("=== INVERTED CASE (YES<0.20): NO-side positions analysis ===")
no_side = [p for p in pos_inv if p.side == "NO" and p.entry_price > 0.80]
yes_side = [p for p in pos_inv if not (p.side == "NO" and p.entry_price > 0.80)]
print(f"high-NO positions (likely inverted from YES tail): n={len(no_side)}")
if no_side:
    pnl_no = sum(p.realized_profit_usd for p in no_side if p.closed)
    res_no = [p for p in no_side if p.closed]
    w = sum(1 for p in res_no if p.realized_profit_usd > 0)
    l = sum(1 for p in res_no if p.realized_profit_usd < 0)
    print(f"  resolved: {len(res_no)}, W={w}, L={l}, wr={w/max(1,w+l)*100:.1f}%, PnL=${pnl_no:+.2f}")
    print(f"  avg entry: ${sum(p.entry_price for p in no_side)/len(no_side):.3f}")
    print(f"  → answers: do high-NO inverted bets actually fill, and do they win?")
