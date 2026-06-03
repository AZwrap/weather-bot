"""Test the 3-bucket hedge (operator idea, 2026-06-03): when the favorite (e.g.
24°C) reaches the trigger, buy YES on favorite ±1 (23/24/25) + NO on the rest —
to hedge the off-by-one bust (the June-2 failure mode: max lands one bucket
hotter than the favorite). SHADOW only, no trading change.

1-bucket basket and 3-bucket basket differ ONLY on the favorite's two neighbors:
1-bucket holds NO on them, 3-bucket holds YES. So we compute the differential on
those two legs per event: diff = (3-bucket YES-neighbor net) − (1-bucket
NO-neighbor net). Total diff = the net effect of the hedge.

Prices: forecast_log top-of-book (yes_ask to BUY YES; NO_ask=1−yes_bid to BUY
NO), $5 notional/leg, taker fees, WUG resolution.
CAVEATS: top-of-book (NOT depth-walked); the fire snapshot is the first
forecast_log snapshot where the favorite's yes_ask ≥ TRIG (forecast cadence, not
the exact WS crossing). So this is a first-order read, not the live number.

Run:  python analyze_basket_3bucket.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import slim_dashboard as sd
from weather_bot.fees import taker_fee_usd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

DATA = Path("data")
TRIG = 0.82
SIZE = 5.0
EXCL = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM"}


def unit(s):
    st = STATIONS_BY_ID.get(s)
    return (getattr(st, "unit", None) or "C")


def leg_net(price, won):
    """Buy $SIZE of an outcome at top-of-book `price`; pays $1/share if `won`."""
    if price is None or not (0.0 < price < 1.0):
        return None
    sh = SIZE / price
    return (sh if won else 0.0) - SIZE - taker_fee_usd(sh, price)


def main():
    resmap = sd.resolution_map()
    fc = defaultdict(list)
    for line in (DATA / "forecast_log.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        bs = r.get("bucket_snapshots")
        if not bs:
            continue
        rows = []
        for s in bs:
            thr = s.get("threshold")
            if thr is None:
                continue
            rows.append((int(thr), s.get("kind"), s.get("bucket_label"),
                         s.get("yes_ask"), s.get("yes_bid")))
        fc[(r.get("station_id"), r.get("target"), r.get("target_date"))].append(
            (r.get("issue_time_utc") or "", rows))

    n = arb_ok = adj_won_events = 0
    tot = 0.0
    detail = []
    for sk, snaps in fc.items():
        sid, tgt, date = sk
        if sid in EXCL:
            continue
        ac = resmap.get(sk)
        if ac is None:
            continue
        u = unit(sid)
        ai = _rounded_observation(ac, u)
        snaps.sort()
        fire = None
        for _it, rows in snaps:
            yas = [b[3] for b in rows if b[3] is not None]
            if yas and max(yas) >= TRIG:
                fire = rows
                break
        if fire is None:
            continue
        rows = sorted(fire, key=lambda b: b[0])
        favi = max(range(len(rows)), key=lambda i: rows[i][3] if rows[i][3] is not None else -1)
        n += 1
        core = [i for i in (favi - 1, favi, favi + 1) if 0 <= i < len(rows)]
        yes_sum = sum(rows[i][3] for i in core if rows[i][3] is not None)
        if yes_sum < 1.0:
            arb_ok += 1
        neigh = [i for i in (favi - 1, favi + 1) if 0 <= i < len(rows)]
        d = 0.0
        adjwon = False
        for i in neigh:
            thr, kind, label, ya, yb = rows[i]
            won = bucket_won(kind, thr, ai, u)
            if won:
                adjwon = True
            yes_n = leg_net(ya, won)                       # 3-bucket: YES-neighbor
            no_n = leg_net((1.0 - yb) if yb is not None else None, not won)  # 1-bucket: NO-neighbor
            if yes_n is None or no_n is None:
                continue
            d += yes_n - no_n
        if adjwon:
            adj_won_events += 1
        tot += d
        detail.append((d, sk, adjwon, rows[favi][2]))

    print("=" * 74)
    print("3-BUCKET HEDGE TEST — buy YES on favorite ±1 instead of NO (shadow)")
    print("  diff = (3-bucket YES-neighbor net) − (1-bucket NO-neighbor net), $5/leg,")
    print("  top-of-book forecast_log prices, taker fees, WUG resolution.")
    print("=" * 74)
    print("events fired (fav≥%.2f, resolved, clean): %d | arb-gate (3 YES sum<$1): %d | an adjacent bucket WON: %d"
          % (TRIG, n, arb_ok, adj_won_events))
    print()
    print("NET 3-bucket effect (the ±1 legs only): $%+.2f" % tot)
    won_d = sum(x[0] for x in detail if x[2])
    notwon_d = sum(x[0] for x in detail if not x[2])
    print("  when an adjacent bucket WON (off-by-one bust): $%+8.2f over %d events" % (won_d, adj_won_events))
    print("  when neither adjacent won (favorite hit / far): $%+8.2f over %d events" % (notwon_d, n - adj_won_events))
    print()
    print("biggest helps / hurts:")
    for d, sk, aw, fav in sorted(detail, key=lambda x: -x[0])[:5]:
        print("   +  %-5s %-3s %s fav='%s' adj_won=%s  diff $%+.2f" % (sk[0], sk[1], sk[2], fav, aw, d))
    for d, sk, aw, fav in sorted(detail, key=lambda x: x[0])[:5]:
        print("   -  %-5s %-3s %s fav='%s' adj_won=%s  diff $%+.2f" % (sk[0], sk[1], sk[2], fav, aw, d))


if __name__ == "__main__":
    main()
