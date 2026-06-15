#!/usr/bin/env python3
"""Hunt badatmath's residual ~+5pp NO-fade alpha: which forward-computable feature
carries it? Decomposes alpha by entry sub-band, target (highest/lowest temp), and
trade SIZE (conviction proxy), with poligarch (~0 alpha) as a control.

Reads:
  - size alpha rising with size  -> private conviction signal = real skill, NOT
    replicable by us (we don't have their conviction).
  - a feature with alpha for badatmath but flat ~0 for poligarch -> badatmath-
    specific selection; if the feature is structural/forward-computable it's a
    candidate filter, if it's size/conviction it isn't.
  - flat alpha everywhere -> +5pp is uniform => unidentifiable (skill we can't
    feature-ize, or noise); directional hunt is exhausted.

Order-book features (spread/imbalance at THEIR entry) are NOT recoverable -- we have
no historical book snapshots for their trades -- so this is the trade-record-only cut.

Usage: python analyze_badatmath_features.py [--days 21] [--lo 0.68] [--hi 0.92]
"""
from __future__ import annotations
import argparse, time
import analyze_poligarch_trades as A

TRADERS = [("badatmath", "0x8fbd7cf5f806f563080864694415829f7229a959"),
           ("poligarch", "0xb40e89677d59665d5188541ad860450a6e2a7cc9")]


def cell(rows):
    """rows: list of (price, won, usd) -> (n, avgPx, win, alpha)."""
    dep = sum(r[2] for r in rows)
    if not rows or dep <= 0:
        return (len(rows), 0.0, 0.0, 0.0)
    px = sum(r[0] * r[2] for r in rows) / dep
    win = sum((1.0 if r[1] else 0.0) * r[2] for r in rows) / dep
    return (len(rows), px, win, win - px)


def show(label, rows):
    n, px, win, al = cell(rows)
    return f"  {label:16}{n:>6}{px:>8.3f}{100*win:>7.1f}%{100*al:>+7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=21)
    ap.add_argument("--lo", type=float, default=0.68)
    ap.add_argument("--hi", type=float, default=0.92)
    args = ap.parse_args()
    now = int(time.time())
    for label, addr in TRADERS:
        raw = A.fetch_window(addr, now - int(args.days * 86400), now)
        trades = []
        for r in raw:
            if r["side"] != "BUY" or r["outcome"] != "No":
                continue
            p = r.get("price")
            if not p or not (args.lo <= p < args.hi):
                continue
            res = A.resolve_market(r["condition_id"])
            if not res or not res.get("closed"):
                continue
            trades.append({"p": p, "won": bool(res.get("No", False)),
                           "usd": r.get("usdc") or 0.0, "target": r.get("target")})
        A.save_cache()
        R = lambda t: (t["p"], t["won"], t["usd"])
        print(f"\n=== {label}: {len(trades)} NO-fade trades {args.lo:.2f}-{args.hi:.2f} ({args.days:.0f}d) ===")
        print(f"  {'':16}{'n':>6}{'avgPx':>8}{'win%':>7}{'ALPHA':>7}")
        print(show("OVERALL", [R(t) for t in trades]))
        print("  -- by entry sub-band --")
        for lo, hi in [(0.68, 0.74), (0.74, 0.80), (0.80, 0.86), (0.86, 0.92)]:
            sub = [R(t) for t in trades if lo <= t["p"] < hi]
            if sub:
                print(show(f"{lo:.2f}-{hi:.2f}", sub))
        print("  -- by target --")
        for tg in ("highest", "lowest"):
            sub = [R(t) for t in trades if t["target"] == tg]
            if sub:
                print(show(tg, sub))
        print("  -- by trade size (conviction proxy) --")
        sizes = sorted(t["usd"] for t in trades)
        if len(sizes) >= 6:
            q1 = sizes[len(sizes) // 3]
            q2 = sizes[2 * len(sizes) // 3]
            print(f"     terciles: small<=${q1:.0f}  mid<=${q2:.0f}  large>${q2:.0f}")
            print(show("small", [R(t) for t in trades if t["usd"] <= q1]))
            print(show("mid", [R(t) for t in trades if q1 < t["usd"] <= q2]))
            print(show("large", [R(t) for t in trades if t["usd"] > q2]))


if __name__ == "__main__":
    main()
