#!/usr/bin/env python3
"""Where is the cohort's edge? Reliability table: realized win-rate vs entry price.

If the market were fairly priced, win-rate == price (zero edge). A systematic
GAP (NO wins more than its price implies / YES wins less) IS the edge: the
favorite-longshot bias. This needs NO forecast — it's a price-level inefficiency.

Reads the cohort substrate (every book's resolved bets). Pure stdlib + numpy.
"""
import json, sys
from collections import defaultdict
from statistics import mean
from pathlib import Path

sub = json.loads(Path("data/poligarch/cohort_substrate.json").read_text())

# Dedup to unique (market, outcome) so we measure the MARKET's mispricing,
# not how many times the cohort piled into it. won == did that outcome resolve true.
mk = {}
for r in sub:
    k = (r["condition_id"], r["outcome"])
    e = mk.setdefault(k, {"vwaps": [], "won": r["won"],
                          "unit": "C" if "°c" in r["title"].lower() else "F"})
    e["vwaps"].append(r["vwap"])
rows = [(o, mean(v["vwaps"]), v["won"], v["unit"]) for (c, o), v in mk.items()]
print(f"Unique (market,outcome) cells: {len(rows)}  "
      f"(NO={sum(1 for o,*_ in rows if o=='No')}, YES={sum(1 for o,*_ in rows if o=='Yes')})")


def table(outcome, unit_filter=None):
    sel = [(p, w) for o, p, w, u in rows if o == outcome and (unit_filter is None or u == unit_filter)]
    if not sel:
        return
    tag = f"{outcome} [{unit_filter or 'all'}]"
    print(f"\n{tag} reliability  (n={len(sel)}):")
    print(f"  {'price bin':11}{'n':>5}{'avg_px':>8}{'win%':>7}{'edge=win-px':>13}{'ROI/$':>8}")
    tot = []
    for i in range(10):
        lo, hi = i / 10, i / 10 + 0.1
        b = [(p, w) for p, w in sel if lo <= p < hi]
        if not b:
            continue
        n = len(b); ap = mean(p for p, w in b); wr = mean(1.0 if w else 0.0 for p, w in b)
        roi = mean(((1.0 if w else 0.0) - p) / p for p, w in b)
        flag = "  <== edge" if (wr - ap) > 0.05 and n >= 20 else ""
        print(f"  [{lo:.1f},{hi:.1f}){n:>5}{ap:>8.3f}{100*wr:>6.1f}%{100*(wr-ap):>+11.1f}pp{100*roi:>+7.1f}%{flag}")
        tot.append((ap, wr, n))
    return sel


# The headline: buy NO on every unique market in the fade band, equal weight
for outcome in ("No", "Yes"):
    table(outcome, "C")
print("\n" + "-" * 60)
for band, lo, hi in [("NO 0.70-0.90", 0.70, 0.90), ("NO 0.80-0.95", 0.80, 0.95),
                     ("YES 0.02-0.15", 0.02, 0.15)]:
    out = "No" if band.startswith("NO") else "Yes"
    b = [(p, w) for o, p, w, u in rows if o == out and u == "C" and lo <= p < hi]
    if not b:
        continue
    n = len(b); wr = mean(1.0 if w else 0.0 for p, w in b); ap = mean(p for p, w in b)
    roi = mean(((1.0 if w else 0.0) - p) / p for p, w in b)
    print(f"{band}: n={n}  avg_px {ap:.3f}  win {100*wr:.1f}%  underpriced {100*(wr-ap):+.1f}pp  "
          f"equal-weight ROI {100*roi:+.1f}%/mkt")
