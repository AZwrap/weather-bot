#!/usr/bin/env python3
"""DECISIVE TEST: is the cohort's NO-fade alpha a forward signal or a same-day
(knowledge-game) timing effect?

The pooled substrate shows ~+19pp alpha fading at 0.7-0.9 across nearly every
city -- yet our next-day pure-forward test is -15%. Hypothesis: the cohort enter
SAME-DAY, after the daily high is largely realized, so NO @ 0.79 is actually
~0.98 to win. This splits each trade by entry-time-vs-target-date and recomputes
alpha per bucket. If same-day alpha >> forward alpha, the edge is timing (and
matches our same-day-am vs next-day split), i.e. it needs LIVE intraday execution,
not a next-day paper bet.

Usage: python analyze_cohort_timing.py [--days 21] [--lo 0.70] [--hi 0.90]
"""
from __future__ import annotations
import argparse, datetime, time
from collections import defaultdict
import analyze_poligarch_trades as A

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
TRADERS = [("poligarch", "0xb40e89677d59665d5188541ad860450a6e2a7cc9"),
           ("badatmath", "0x8fbd7cf5f806f563080864694415829f7229a959")]


def target_date(s):
    p = s.split("-")
    return datetime.date(int(p[2]), MONTHS[p[0]], int(p[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=21)
    ap.add_argument("--lo", type=float, default=0.70)
    ap.add_argument("--hi", type=float, default=0.90)
    args = ap.parse_args()
    now = int(time.time())
    for label, addr in TRADERS:
        rows = A.fetch_window(addr, now - int(args.days * 86400), now)
        recs = []
        for r in rows:
            if r["side"] != "BUY" or r["outcome"] != "No":
                continue
            p = r.get("price")
            if not p or not (args.lo <= p < args.hi):
                continue
            res = A.resolve_market(r["condition_id"])
            if not res or not res.get("closed"):
                continue
            won = bool(res.get("No", False))
            try:
                td = target_date(r["date"])
            except Exception:
                continue
            ed = datetime.datetime.fromtimestamp(r["ts"], datetime.timezone.utc).date()
            dd = (td - ed).days   # >0 = entered days BEFORE target (forward); 0 = same UTC day
            recs.append((dd, p, won, r.get("usdc") or 0.0))
        A.save_cache()

        buck = defaultdict(lambda: {"dep": 0.0, "w": 0.0, "vw": 0.0, "n": 0})
        order = ["same-day (<=0)", "1d forward", "2d forward", "3+d forward"]
        def key(dd):
            return order[0] if dd <= 0 else (order[1] if dd == 1 else (order[2] if dd == 2 else order[3]))
        for dd, p, won, usd in recs:
            b = buck[key(dd)]
            d = max(usd, 1e-9)
            b["dep"] += d; b["w"] += (1.0 if won else 0.0) * d; b["vw"] += p * d; b["n"] += 1
        print(f"\n=== {label}: NO-fade {args.lo:.2f}-{args.hi:.2f}, alpha by entry timing "
              f"({len(recs)} trades, {args.days:.0f}d) ===")
        print(f"  {'window':18}{'n':>5}{'avgPx':>8}{'win%':>8}{'ALPHA':>8}")
        for k in order:
            b = buck.get(k)
            if not b or b["dep"] <= 0:
                continue
            win = b["w"] / b["dep"]; px = b["vw"] / b["dep"]
            print(f"  {k:18}{b['n']:>5}{px:>8.3f}{100*win:>7.1f}%{100*(win-px):>+7.1f}")


if __name__ == "__main__":
    main()
