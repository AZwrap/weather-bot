#!/usr/bin/env python3
"""Sharpen the live-test basket from paper data: per-city + per-band edge + capacity.

Designs the eventual min-stake live test: which cities to fade, which exact price
band, and roughly how much $ the recreational flow absorbs (capacity).
Reads the cohort substrate (all 11 books, 5d). Pure stdlib.
"""
import json
from collections import defaultdict
from statistics import mean, median
from pathlib import Path

sub = json.loads(Path("data/poligarch/cohort_substrate.json").read_text())
def isC(r): return "°c" in r["title"].lower()

# Unique NO °C markets in the fade band, with cohort TOTAL deployment summed across books.
mk = {}
for r in sub:
    if r["outcome"] == "No" and isC(r) and 0.70 <= r["vwap"] < 0.90:
        e = mk.setdefault(r["condition_id"], {"city": r["city"], "date": r["date"],
                                              "won": r["won"], "vw": [], "dep": 0.0})
        e["vw"].append(r["vwap"]); e["dep"] += (r.get("deployed") or 0)
for e in mk.values():
    e["px"] = mean(e["vw"])
M = list(mk.values())
print(f"NO °C 0.70-0.90 unique markets: {len(M)}  over {len({m['date'] for m in M})} days")


def stats(rows):
    n = len(rows)
    wr = mean(1.0 if m["won"] else 0.0 for m in rows)
    ap = mean(m["px"] for m in rows)
    roi = mean(((1.0 if m["won"] else 0.0) - m["px"]) / m["px"] for m in rows)
    dep = sum(m["dep"] for m in rows)
    return n, wr, ap, roi, dep


print("\nPER-CITY (n>=8), sorted by volume   [* = win>=95% & n>=15: core basket]")
print(f"  {'city':18}{'n':>4}{'win%':>7}{'edge':>8}{'ROI%':>7}{'cohort$':>9}{'$/mkt':>7}")
basket = []
for c, rows in sorted(defaultdict(list, {c: [m for m in M if m["city"] == c]
                                         for c in {m["city"] for m in M}}).items(),
                      key=lambda x: -len(x[1])):
    if len(rows) < 8:
        continue
    n, wr, ap, roi, dep = stats(rows)
    star = "  *" if wr >= 0.95 and n >= 15 else ""
    print(f"  {c:18}{n:>4}{100*wr:>6.1f}%{100*(wr-ap):>+7.1f}pp{100*roi:>+6.1f}%{dep:>9,.0f}{dep/n:>7,.0f}{star}")
    if wr >= 0.93 and n >= 12:
        basket.append(c)

print("\nFINE BANDS (all cities):")
print(f"  {'band':13}{'n':>5}{'win%':>7}{'edge':>8}{'ROI%':>7}")
for lo in (0.70, 0.75, 0.80, 0.85):
    rows = [m for m in M if lo <= m["px"] < lo + 0.05]
    if rows:
        n, wr, ap, roi, dep = stats(rows)
        print(f"  [{lo:.2f},{lo+0.05:.2f}){n:>5}{100*wr:>6.1f}%{100*(wr-ap):>+7.1f}pp{100*roi:>+6.1f}%")

print("\nCAPACITY — edge by per-market cohort-deployment quartile (does edge survive heavy $?):")
M2 = sorted(M, key=lambda m: m["dep"]); q = len(M2) // 4
for i, name in enumerate(["Q1 smallest$", "Q2", "Q3", "Q4 largest$"]):
    rows = M2[i*q:(i+1)*q] if i < 3 else M2[3*q:]
    n, wr, ap, roi, dep = stats(rows)
    print(f"  {name:13} n={n:>4}  ${rows[0]['dep']:>5.0f}-${rows[-1]['dep']:>6.0f}/mkt  win {100*wr:.1f}%  ROI {100*roi:+.1f}%")

deps = sorted(m["dep"] for m in M); ndays = len({m["date"] for m in M})
print(f"\n  per-market cohort NO $: median ${median(deps):.0f}  p90 ${deps[int(0.9*len(deps))]:.0f}  max ${max(deps):.0f}")
print(f"  ~{len(M)/ndays:.0f} band-markets/day; cohort fades ~${sum(deps)/ndays:,.0f}/day total in this band")
print(f"\nCORE BASKET (win>=93%, n>=12): {sorted(basket)}")
