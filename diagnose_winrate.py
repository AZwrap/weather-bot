"""Diagnose where the bot's losses are concentrated. Goal: find structural
patterns we can act on (exclude bad stations, bad bucket types, bad sides)
without waiting weeks for more data."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

from weather_bot.forward_log import load_records
from weather_bot.positions import replay_maker

recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]
positions = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
resolved = [p for p in positions if p.closed]
print(f"resolved positions: {len(resolved)}")
total_pnl = sum(p.realized_profit_usd for p in resolved)
print(f"total realized P&L: ${total_pnl:+,.2f}")
print()


def report(title, groups: dict[str, list]) -> None:
    print(f"\n=== {title} ===")
    rows = []
    for k, ps in groups.items():
        n = len(ps)
        wins = sum(1 for p in ps if p.realized_profit_usd > 0)
        loss = sum(1 for p in ps if p.realized_profit_usd < 0)
        wr = wins / max(1, wins + loss)
        pnl = sum(p.realized_profit_usd for p in ps)
        rows.append((k, n, wins, loss, wr, pnl))
    rows.sort(key=lambda r: r[5])  # sort by P&L
    print(f"{'group':28s} {'n':>5s} {'W':>4s} {'L':>4s} {'wr':>6s} {'pnl':>10s}")
    for k, n, w, l, wr, pnl in rows:
        print(f"{k:28s} {n:5d} {w:4d} {l:4d} {wr*100:5.1f}% ${pnl:+9.2f}")


# By side
by_side: dict[str, list] = defaultdict(list)
for p in resolved:
    by_side[p.side].append(p)
report("BY SIDE", by_side)

# By bucket-kind
by_kind: dict[str, list] = defaultdict(list)
for p in resolved:
    by_kind[p.bucket_kind].append(p)
report("BY BUCKET KIND", by_kind)

# By bucket-kind × side
by_ks: dict[str, list] = defaultdict(list)
for p in resolved:
    by_ks[f"{p.bucket_kind}/{p.side}"].append(p)
report("BY KIND × SIDE", by_ks)

# By station
by_stn: dict[str, list] = defaultdict(list)
for p in resolved:
    by_stn[p.station_id].append(p)
report("BY STATION", by_stn)

# By target (max vs min)
by_tgt: dict[str, list] = defaultdict(list)
for p in resolved:
    by_tgt[p.target].append(p)
report("BY TARGET", by_tgt)

# By entry-price bin (predicted-prob proxy)
def pp_bin(p) -> str:
    e = p.entry_price
    for hi, label in [(0.10, "0-10%"), (0.20, "10-20%"), (0.30, "20-30%"),
                      (0.50, "30-50%"), (0.70, "50-70%"), (1.0, "70-100%")]:
        if e < hi:
            return label
    return "??"
by_pp: dict[str, list] = defaultdict(list)
for p in resolved:
    by_pp[pp_bin(p)].append(p)
report("BY ENTRY PRICE BIN", by_pp)

# Calibration: predicted vs actual hit rate per entry-price bin
print("\n=== CALIBRATION CHECK (entry price ≈ implied prob we paid) ===")
print(f"{'bin':10s} {'n':>5s} {'paid_avg':>9s} {'hit_rate':>9s} {'gap':>9s}")
for label in ["0-10%", "10-20%", "20-30%", "30-50%", "50-70%", "70-100%"]:
    ps = by_pp.get(label, [])
    if not ps:
        continue
    n = len(ps)
    paid_avg = sum(p.entry_price for p in ps) / n
    hits = sum(1 for p in ps if p.realized_profit_usd > 0)
    hit_rate = hits / n
    gap = hit_rate - paid_avg
    arrow = "↑" if gap > 0.02 else ("↓" if gap < -0.02 else "≈")
    print(f"{label:10s} {n:5d} {paid_avg*100:7.1f}% {hit_rate*100:7.1f}% "
          f"{gap*100:+7.1f}pp {arrow}")
print("(if hit_rate < paid_avg by a lot → we're paying more than we win)")
