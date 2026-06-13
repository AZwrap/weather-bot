#!/usr/bin/env python3
"""Is the NO 0.7-0.9 favorite-longshot edge BROAD and SAFE, or selected + fat-tailed?

Two checks on the located edge:
  1. Per-trader consistency — if all 11 books independently profit on NO 0.7-0.9,
     it's a market-wide bias (capturable by anyone), not one book's selection skill.
  2. Correlated-bust / fat-tail — the 98.7% win rate hides rare days where a
     regional heat front busts many NO bets at once (the prior study's finding).
     Measures daily PnL dispersion + worst-day drawdown.
"""
import json
from collections import defaultdict
from statistics import mean, pstdev
from pathlib import Path

sub = json.loads(Path("data/poligarch/cohort_substrate.json").read_text())
def isC(r): return "°c" in r["title"].lower()
LO, HI = 0.70, 0.90

# 1. per-trader consistency
pt = defaultdict(list)
for r in sub:
    if r["outcome"] == "No" and isC(r) and LO <= r["vwap"] < HI:
        pt[r["trader"]].append((r["vwap"], 1.0 if r["won"] else 0.0))
print(f"Per-trader NO [{LO},{HI}) — is the bias broad across books?")
print(f"  {'trader':15}{'n':>5}{'win%':>7}{'ROI/$':>8}")
for t, rows in sorted(pt.items(), key=lambda x: -len(x[1])):
    if len(rows) < 10:
        continue
    win = mean(w for _, w in rows)
    roi = mean(((1.0 if w else 0.0) - p) / p for p, w in rows)
    print(f"  {t:15}{len(rows):>5}{100*win:>6.1f}%{100*roi:>+7.1f}%")

# 2. dedup unique markets, correlated busts
mk = {}
for r in sub:
    if r["outcome"] == "No" and isC(r) and LO <= r["vwap"] < HI:
        e = mk.setdefault(r["condition_id"], {"won": r["won"], "date": r["date"], "city": r["city"], "vw": []})
        e["vw"].append(r["vwap"])
markets = [(v["date"], v["city"], v["won"], mean(v["vw"])) for v in mk.values()]
n = len(markets)
losers = [(d, c) for d, c, w, p in markets if not w]
print(f"\nUnique NO [{LO},{HI}) markets: {n}  | losers: {len(losers)} ({100*len(losers)/n:.1f}%)")

byday = defaultdict(lambda: [0, 0])
daily_pnl = defaultdict(float)
for d, c, w, p in markets:
    byday[d][0] += 1
    byday[d][1] += 0 if w else 1
    daily_pnl[d] += ((1.0 - p) / p) if w else -1.0   # $1 NO per market, held to resolution

print("Worst correlated-bust days (losses / markets that day):")
for d, (tot, lost) in sorted(byday.items(), key=lambda x: -x[1][1])[:6]:
    print(f"  {d}: {lost:2d}/{tot:2d} lost")

pnls = sorted(daily_pnl.values())
tot = sum(pnls)
print(f"\nDaily PnL, $1 NO on every market ({len(pnls)} days, {n} bets):")
print(f"  total ${tot:+.1f}  mean/day ${mean(pnls):+.2f}  std ${pstdev(pnls):.2f}  "
      f"worst ${min(pnls):+.1f}  best ${max(pnls):+.1f}")
neg = [x for x in pnls if x < 0]
print(f"  losing days: {len(neg)}/{len(pnls)}  ({100*len(neg)/len(pnls):.0f}%)  "
      f"sum of losing days ${sum(neg):+.1f} vs winning ${tot - sum(neg):+.1f}")
