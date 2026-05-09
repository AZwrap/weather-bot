"""Are ladders worth it? Direct comparison on May 8 data.

Same signals, same hold/sell logic — just vary the entry execution:
  1. Maker 4-rung ladder (current canonical)
  2. Maker 1-rung at midprice (single resting limit, no spread spreading)
  3. Pure taker (fill at ask immediately, never a maker)

Each against three filter regimes (no skip / skip<0.20 / skip<0.30) so we
see whether ladders' value depends on the tail-vs-mid composition.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from weather_bot.forward_log import load_records
from weather_bot.positions import replay, replay_maker, summarize


def report(label: str, positions) -> None:
    s = summarize(positions)
    n_resolved = sum(1 for p in positions if p.closed)
    n_w = sum(1 for p in positions if p.closed and p.realized_profit_usd > 0)
    n_l = sum(1 for p in positions if p.closed and p.realized_profit_usd < 0)
    pnl = sum(p.realized_profit_usd for p in positions if p.closed)
    expo = sum(p.position_usd for p in positions)
    res_expo = sum(p.position_usd for p in positions if p.closed)
    wr = n_w / max(1, n_w + n_l)
    avg_entry = (sum(p.entry_price for p in positions) / len(positions)
                 if positions else 0.0)
    print(f"{label:50s} n={len(positions):4d}  R={n_resolved:4d}  "
          f"W={n_w:3d}  L={n_l:3d}  wr={wr*100:5.1f}%  "
          f"PnL=${pnl:+8.2f}  rexp=${res_expo:6.0f}  "
          f"avgE=${avg_entry:.3f}")


recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
print(f"records: {len(recs)}, eligible: {len(elig)}\n")

COMMON = dict(
    bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25,
    sigma_inflation_factor=1.4,
    take_profit_threshold=0.05, stop_loss_threshold=-0.10, stop_loss_pct=0.50,
)

for filter_label, min_p, max_p in [
    ("filter 0.05-0.95", 0.05, 0.95),
    ("filter 0.20-0.80 (skip tails)", 0.20, 0.80),
    ("filter 0.30-0.70 (skip wide tails)", 0.30, 0.70),
]:
    print(f"=== {filter_label} ===")

    pos_maker_4 = replay_maker(
        elig, n_rungs=4, min_yes_price=min_p, max_yes_price=max_p,
        taker_fallback=False, **COMMON,
    )
    report("maker 4-rung ladder", pos_maker_4)

    pos_maker_1 = replay_maker(
        elig, n_rungs=1, min_yes_price=min_p, max_yes_price=max_p,
        taker_fallback=False, **COMMON,
    )
    report("maker 1-rung (single midprice limit)", pos_maker_1)

    pos_taker = replay(
        elig, min_yes_price=min_p, max_yes_price=max_p,
        re_entry_threshold=0.05, **COMMON,
    )
    report("pure taker (fill at ask)", pos_taker)

    print()

# Also compute spread-paid metrics for taker to make the cost concrete
print("=== Taker spread cost on filter 0.05-0.95 ===")
pos_taker_full = replay(
    elig, min_yes_price=0.05, max_yes_price=0.95,
    re_entry_threshold=0.05, **COMMON,
)
spread_paid = 0.0
mid_distance_count = 0
for p in pos_taker_full:
    open_ev = p.events[0]
    if open_ev.market_yes_implied_at_step is not None:
        if p.side == "YES":
            spread = p.entry_price - open_ev.market_yes_implied_at_step
        else:
            no_mid = 1.0 - open_ev.market_yes_implied_at_step
            spread = p.entry_price - no_mid
        spread_paid += spread * p.shares
        mid_distance_count += 1
print(f"taker fills: {mid_distance_count}")
print(f"total spread paid (vs midprice): ${spread_paid:+.2f}")
print(f"avg spread cost per position: ${spread_paid/max(1,mid_distance_count):.4f}")
