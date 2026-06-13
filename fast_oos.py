#!/usr/bin/env python3
"""Fast OOS bias + fat-tail test: poligarch only, CONCURRENT CLOB resolution.

Answers two questions over a long window (default 15d, reaching the prior
study's June-2 correlated-bust day):
  1. Does the NO favorite-longshot bias hold out-of-sample?
  2. Does a genuinely bad heat-wave day turn a diversified NO-fade day net-negative?

Poligarch is the ideal single book: pure hold-to-resolution (RT=0), highest
volume (huge N), traded through June 2. Reuses analyze_poligarch_trades helpers;
resolves markets with a thread pool (the slow part, made parallel).

Usage:  python fast_oos.py [days=15]
"""
import sys, json, time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from statistics import mean
from pathlib import Path
import analyze_poligarch_trades as A

POLI = "0xb40e89677d59665d5188541ad860450a6e2a7cc9"
RES_CACHE = Path("data/poligarch/resolutions.json")


def resolve_one(cid):
    try:
        m = A.get_json(f"{A.CLOB}/markets/{cid}")
    except Exception:
        return cid, None
    if isinstance(m, dict) and m.get("tokens"):
        r = {"closed": bool(m.get("closed"))}
        for t in m["tokens"]:
            r[str(t.get("outcome"))] = bool(t.get("winner")) or (t.get("price") in (1, 1.0, "1"))
        return cid, r
    return cid, None


def main():
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 15
    now = int(time.time()); start = now - int(days * 86400); end = now - 86400
    print(f"Pulling poligarch {days:.0f}d ...", file=sys.stderr)
    rows = A.fetch_window(POLI, start, end)
    agg, meta = A.aggregate(rows)
    cache = json.loads(RES_CACHE.read_text()) if RES_CACHE.exists() else {}
    cids = list({c for c, o in agg})
    todo = [c for c in cids if c not in cache]
    print(f"{len(rows)} trades, {len(cids)} markets, resolving {len(todo)} concurrently ...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (cid, res) in enumerate(ex.map(resolve_one, todo)):
            cache[cid] = res
            if (i + 1) % 1000 == 0:
                print(f"  resolved {i+1}/{len(todo)}", file=sys.stderr)
    RES_CACHE.write_text(json.dumps(cache))

    recs = []  # (outcome, vwap, won, unit, date, city) — one per unique (market,outcome)
    for (cid, o), a in agg.items():
        if a["buy_sh"] <= 0:
            continue
        res = cache.get(cid)
        if not res or not res.get("closed"):
            continue
        m = meta[(cid, o)]
        unit = "C" if "°c" in (m["title"] or "").lower() else "F"
        recs.append((o, a["buy_usdc"] / a["buy_sh"], bool(res.get(str(o), False)), unit, m["date"], m["city"]))
    dates = sorted({d for *_, d, c in recs})
    print(f"\nResolved records: {len(recs)}  | date span {dates[0]} .. {dates[-1]}")

    print(f"\nNO °C reliability (poligarch, {days:.0f}d OOS):")
    print(f"  {'bin':11}{'n':>6}{'avg_px':>8}{'win%':>7}{'edge':>9}{'ROI/$':>8}")
    for i in range(10):
        lo, hi = i / 10, i / 10 + 0.1
        b = [(p, w) for o, p, w, u, d, c in recs if o == "No" and u == "C" and lo <= p < hi]
        if not b:
            continue
        n = len(b); ap = mean(p for p, w in b); wr = mean(1.0 if w else 0.0 for p, w in b)
        roi = mean(((1.0 if w else 0.0) - p) / p for p, w in b)
        print(f"  [{lo:.1f},{hi:.1f}){n:>6}{ap:>8.3f}{100*wr:>6.1f}%{100*(wr-ap):>+8.1f}pp{100*roi:>+7.1f}%")

    # Fat tail: NO °C 0.7-0.9 (each rec is one unique market), daily PnL incl bad days
    band = [(d, w, p) for o, p, w, u, d, c in recs if o == "No" and u == "C" and 0.70 <= p < 0.90]
    daily = defaultdict(float); byday = defaultdict(lambda: [0, 0])
    for d, w, p in band:
        daily[d] += ((1.0 - p) / p) if w else -1.0
        byday[d][0] += 1; byday[d][1] += 0 if w else 1
    pnls = list(daily.values()); neg = [x for x in pnls if x < 0]
    print(f"\nNO °C 0.7-0.9 fat-tail: {len(band)} bets over {len(daily)} days  ($1/bet, held to resolution)")
    print(f"  total ${sum(pnls):+.1f}  mean/day ${mean(pnls):+.2f}  worst day ${min(pnls):+.1f}  "
          f"losing days {len(neg)}/{len(pnls)}")
    print("  worst days (PnL, losses/bets):")
    for d, v in sorted(daily.items(), key=lambda x: x[1])[:6]:
        print(f"    {d}: ${v:+6.1f}  ({byday[d][1]}/{byday[d][0]} lost)")


if __name__ == "__main__":
    main()
