#!/usr/bin/env python3
"""Compare realized weather edge + STYLE across the top-cohort weather traders.

One process, shared in-process resolution cache. Cash-flow-correct PnL (handles
exiters, not just hold-to-resolution). Reuses helpers from analyze_poligarch_trades.

Per trader: modal name (identity), breadth (#resolved markets), turnover ($ bought),
round-trip% (sell$/buy$ -> hold vs flip), realized PnL/ROI, and an edge "signature"
(ROI by side+entry band). Answers: is the NO-fade engine universal, and what
distinct styles exist?

Usage:  python analyze_traders.py [--start-days-ago 7] [--end-days-ago 1]
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import analyze_poligarch_trades as A

TRADERS = [
    ("poligarch",      "0xb40e89677d59665d5188541ad860450a6e2a7cc9"),
    ("opopv",          "0x116db6298abcdefe06f9f5458c293c7de185fbf1"),
    ("link2-ec86",     "0xec86a2d3f69015b1a9382e4dfa8695e1b48760e4"),
    ("link3-8fbd",     "0x8fbd7cf5f806f563080864694415829f7229a959"),
    ("hightemptation", "0x6011655c4afb76f36dd1b08a137a1ba73466b31e"),
    ("michi1",         "0xa89518aca5a633a79ad1e9737209c9689f83faac"),
    ("shyguy1",        "0x1f66796b45581868376365aef54b51eb84184c8d"),
    ("weatherstappen", "0xb9012e0d9b60d3920286309328b935cdfa609fc4"),
    ("weatherhk",      "0x488c725253fc21c7a9ca812030dc2f6343f98c1c"),
    ("opopv2",         "0xafde461fce5aa0fabdb7711c59db93b65e343e1d"),
    ("sailor82",       "0xbbb72a812cfbc5217d77c0a0018c71f174d3a11a"),
]


def trader_bands(rows):
    agg, meta = A.aggregate(rows)
    bands = defaultdict(lambda: {"deployed": 0.0, "pnl": 0.0, "won": 0.0, "n": 0})
    substrate, resolved = [], 0
    for (cid, outcome), a in agg.items():
        if a["buy_sh"] <= 0:
            continue
        res = A.resolve_market(cid)
        if not res or not res.get("closed"):
            continue
        resolved += 1
        won = bool(res.get(str(outcome), False))
        rp = 1.0 if won else 0.0
        dep, pnl, vwap, net = A.realized_pnl(a, rp)
        band = min(int(vwap * 10), 9) / 10
        b = bands[(outcome, band)]
        b["deployed"] += dep; b["pnl"] += pnl; b["won"] += dep if won else 0.0; b["n"] += 1
        m = meta[(cid, outcome)]
        substrate.append({
            "city": m["city"], "date": m["date"], "target": m["target"], "title": m["title"],
            "condition_id": cid, "asset": m["asset"], "outcome": outcome,
            "vwap": round(vwap, 4), "deployed": round(dep, 2), "won": won,
        })
    return bands, resolved, substrate


def sig(bands, side, lo, hi):
    dep = sum(b["deployed"] for (o, bd), b in bands.items() if o == side and lo - 1e-9 <= bd < hi - 1e-9)
    pnl = sum(b["pnl"] for (o, bd), b in bands.items() if o == side and lo - 1e-9 <= bd < hi - 1e-9)
    return 100 * pnl / dep if dep else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-days-ago", type=float, default=7)
    ap.add_argument("--end-days-ago", type=float, default=1)
    args = ap.parse_args()
    now = int(time.time())
    start = now - int(args.start_days_ago * 86400)
    end = now - int(args.end_days_ago * 86400)

    results, all_sub = [], []
    for label, addr in TRADERS:
        print(f"\n### {label} {addr}", file=sys.stderr)
        rows = A.fetch_window(addr, start, end)
        if not rows:
            results.append((label, "?", 0, 0.0, 0.0, 0.0, 0.0, None)); continue
        names = Counter(r.get("name") for r in rows if r.get("name"))
        modal = names.most_common(1)[0][0] if names else "?"
        buy = sum(r["usdc"] for r in rows if r["side"] == "BUY" and r["usdc"])
        sell = sum(r["usdc"] for r in rows if r["side"] == "SELL" and r["usdc"])
        rt = 100 * sell / buy if buy else 0.0
        bands, resolved, sub = trader_bands(rows)
        for s in sub:
            s["trader"] = label
        all_sub += sub
        dep = sum(b["deployed"] for b in bands.values())
        pnl = sum(b["pnl"] for b in bands.values())
        roi = 100 * pnl / dep if dep else 0.0
        results.append((label, modal, resolved, buy, rt, pnl, roi, bands))
        A.save_cache()
        print(f"  {modal}: {resolved} mkts  turnover ${buy:,.0f}  RT {rt:.0f}%  "
              f"pnl ${pnl:,.0f}  ROI {roi:+.1f}%", file=sys.stderr)

    win = args.start_days_ago - args.end_days_ago
    print("\n" + "=" * 122)
    print(f"TOP WEATHER TRADERS — realized edge + style, ~{win:.0f}d window  (cash-flow PnL, no maker rebates)")
    print(f"{'handle':15}{'name':15}{'#mkt':>5}{'turn$':>10}{'RT%':>5}{'$pnl':>9}{'ROI%':>7}   "
          f"{'NOfade':>7}{'NOmid':>7}{'YESlo':>7}{'YESmid':>7}")
    print("-" * 122)
    for label, modal, resolved, buy, rt, pnl, roi, bands in sorted(results, key=lambda x: -x[6]):
        if bands is None:
            print(f"{label:15}{'(no trades in window)':>30}"); continue
        nf = sig(bands, "No", 0.6, 1.0); nm = sig(bands, "No", 0.2, 0.6)
        yl = sig(bands, "Yes", 0.0, 0.3); ym = sig(bands, "Yes", 0.3, 0.7)
        print(f"{label:15}{str(modal)[:14]:15}{resolved:>5}{buy:>10,.0f}{rt:>5.0f}{pnl:>9,.0f}{roi:>+7.1f}   "
              f"{nf:>+6.0f}%{nm:>+6.0f}%{yl:>+6.0f}%{ym:>+6.0f}%")
    print("=" * 122)
    print("RT%=sell$/buy$ (0=pure hold-to-resolution, high=actively flips). "
          "NOfade=NO@.6-1 | NOmid=NO@.2-.6 | YESlo=YES@0-.3 | YESmid=YES@.3-.7  (ROI per band)")

    Path("data/poligarch").mkdir(parents=True, exist_ok=True)
    Path("data/poligarch/cohort_substrate.json").write_text(json.dumps(all_sub))
    print(f"\nSaved {len(all_sub)} (trader,market,outcome) rows -> data/poligarch/cohort_substrate.json")


if __name__ == "__main__":
    main()
