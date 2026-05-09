"""Diagnostic: how is the stop-loss firing on live forward-log data?"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from weather_bot.forward_log import load_records
from weather_bot.positions import replay_maker, summarize

recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
pos = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
    take_profit_threshold=0.05, stop_loss_threshold=-0.10, stop_loss_pct=0.50,
)
s = summarize(pos)
print(f"positions: {s.n_positions}")
print(f"  open: {s.n_open}")
print(f"  TP: {s.n_take_profit}")
print(f"  SL: {s.n_stop_loss}")
print(f"  expired won: {s.n_expire_won}")
print(f"  expired lost: {s.n_expire_lost}")
print(f"  realized PnL: ${s.total_realized_pnl_usd:+,.2f}")

print()
print("PnL distribution by outcome:")
buckets: dict[str, list[float]] = {"TP": [], "SL": [], "won": [], "lost": [], "open": []}
for p in pos:
    if not p.closed:
        buckets["open"].append(p.position_usd)
    else:
        last = p.events[-1].action if p.events else ""
        if last == "sell_take_profit":
            buckets["TP"].append(p.realized_profit_usd)
        elif last == "sell_stop_loss":
            buckets["SL"].append(p.realized_profit_usd)
        elif p.realized_profit_usd > 0:
            buckets["won"].append(p.realized_profit_usd)
        else:
            buckets["lost"].append(p.realized_profit_usd)

for k, vs in buckets.items():
    if not vs:
        continue
    s_total = sum(vs)
    print(f"  {k}: n={len(vs)}, total=${s_total:+,.2f}, avg=${s_total/len(vs):+,.2f}")

print()
print("Sample of expired-lost positions that DIDN'T stop out:")
print(f"{'station':10s} {'target':12s} {'date':12s} {'bucket':22s} {'side':4s} "
      f"{'entry':>7s} {'evs':>5s} {'pnl':>8s}")
losers_no_sl = [
    p for p in pos
    if p.closed and p.realized_profit_usd < 0
    and (not p.events or p.events[-1].action == "expire")
]
losers_no_sl.sort(key=lambda p: p.realized_profit_usd)
for p in losers_no_sl[:15]:
    print(f"{p.station_id:10s} {p.target:12s} {p.target_date:12s} {p.bucket_label:22s} "
          f"{p.side:4s} {p.entry_price:7.3f} {len(p.events):5d} "
          f"${p.realized_profit_usd:+8.2f}")

print()
print("How many of those lost-and-expired had any chance to stop-loss?")
print("(i.e., bid dropped >= 50% from entry at SOME point during the position)")
n_could_have_sl = 0
n_total_lost_expired = len(losers_no_sl)
for p in losers_no_sl:
    saw_drop = False
    for ev in p.events[1:]:  # skip open
        if ev.fill_price is not None and p.entry_price > 0:
            mtm = (p.entry_price - ev.fill_price) / p.entry_price
            if mtm >= 0.50:
                saw_drop = True
                break
    if saw_drop:
        n_could_have_sl += 1
print(f"  {n_could_have_sl} / {n_total_lost_expired} positions saw >=50% MTM drop "
      f"at some intermediate snapshot")
print("  (if low: stop-loss can't fire because the drop happens AT expiration, not before)")
