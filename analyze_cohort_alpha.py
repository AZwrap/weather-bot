#!/usr/bin/env python3
"""Decompose the cohort's edge: price-implied premium vs SELECTION ALPHA.

For each cohort trade we know the entry price (vwap = the bought outcome's
market-implied win prob) and whether it won. Define, deploy-weighted:

    ALPHA = realized_win_rate - avg_entry_price   (percentage points)

  alpha ~ 0  -> they win what they pay for; any profit is EXECUTION (maker $0-fee +
              25% rebate) on calibrated fades, NOT selection -> nothing for us to
              select on, and not replicable without being a real low-latency maker.
  alpha > 0  -> they pick winners the price underprices -> SELECTION skill. Then the
              job is to find WHERE it concentrates (band / city / target / day) and
              whether that filter is available forward (forecast-free) at our scan.

Hold-to-resolution ROI ~= alpha / price, so the cohort study's '+11% vs +0.5%'
should surface here as a real alpha gap IF it's not just small-sample. N is printed
next to every number for exactly that reason.

Reads data/poligarch/cohort_substrate.json (regenerate via analyze_traders.py).
Usage: python analyze_cohort_alpha.py [--side No] [--min-dep 3] [--min-n 15]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

SUB = Path("data/poligarch/cohort_substrate.json")


def block(rows):
    """-> (n, dep, avg_px, win, alpha, roi) all deploy-weighted."""
    dep = sum(r["deployed"] for r in rows)
    if not rows or dep <= 0:
        return (len(rows), 0.0, 0.0, 0.0, 0.0, 0.0)
    vw = sum(r["vwap"] * r["deployed"] for r in rows) / dep
    win = sum((1.0 if r["won"] else 0.0) * r["deployed"] for r in rows) / dep
    roi = sum(((1.0 if r["won"] else 0.0) - r["vwap"]) * r["deployed"] for r in rows) / dep
    return (len(rows), dep, vw, win, win - vw, roi)


def line(label, rows, w=16):
    n, dep, vw, win, al, roi = block(rows)
    return (f"{label:{w}}{n:>4}{dep:>9,.0f}{vw:>7.3f}{100*win:>6.1f}%"
            f"{100*al:>+7.1f}{100*roi:>+6.1f}%")


HEAD = lambda w=16: f"{'':{w}}{'n':>4}{'dep$':>9}{'avgPx':>7}{'win%':>7}{'ALPHA':>8}{'ROI%':>7}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="No")
    ap.add_argument("--min-dep", type=float, default=3.0)
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args()
    rows = json.loads(SUB.read_text())
    rows = [r for r in rows if r.get("deployed", 0) >= args.min_dep and r.get("outcome") == args.side]
    print(f"=== cohort {args.side}-side edge: implied premium vs SELECTION ALPHA "
          f"(n={len(rows)} trades >= ${args.min_dep}) ===\n")

    by_tr = defaultdict(list)
    for r in rows:
        by_tr[r["trader"]].append(r)
    print(HEAD())
    ranked = sorted(by_tr.items(), key=lambda x: -block(x[1])[4])
    for tr, rs in ranked:
        print(line(tr, rs))
    print("\nALPHA = realized win% - avg entry price (pp). >0 = selection skill; ~0 = pays for what it gets.")
    print("(ROI ~= alpha/price; small n => treat as noise. Watch the n column.)")

    # Decompose the traders that clear the n gate, by entry-price band and by city.
    BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    focus = [(tr, rs) for tr, rs in ranked if len(rs) >= args.min_n]
    for tr, rs in focus[:4]:
        print(f"\n--- {tr}  (n={len(rs)}) by entry band ---")
        print(HEAD(12))
        for lo, hi in BANDS:
            sub = [r for r in rs if lo <= r["vwap"] < hi]
            if sub:
                print(line(f"{lo:.1f}-{hi:.1f}", sub, 12))
        bycity = defaultdict(list)
        for r in rs:
            bycity[r["city"]].append(r)
        big = sorted(bycity.items(), key=lambda x: -block(x[1])[4])
        big = [(c, cr) for c, cr in big if len(cr) >= 4]
        if big:
            print(f"  top/bottom cities (n>=4):")
            for c, cr in big[:4] + big[-3:]:
                print("  " + line(c, cr, 14))


if __name__ == "__main__":
    main()
