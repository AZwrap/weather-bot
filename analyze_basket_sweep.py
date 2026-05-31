"""Basket threshold-sweep analyzer — P&L vs entry threshold, overall +
per-station, net of Polymarket taker fees.

Joins the shadow sweep log with WUG resolutions:
  data/basket_sweep_log.jsonl   — one row per (event, observed-penny) basket
  data/forward_log.jsonl        — WUG resolutions (actual_obs_c)

For each logged basket (a winner-YES leg + N fade-NO legs captured at a
given entry threshold T), once the event resolves we know the winning
bucket, so we can compute realized payouts:
  - winner-YES leg pays $1/share iff its bucket WON
  - each fade-NO leg pays $1/share iff its bucket did NOT win
Entry cost (incl. fee) is already baked into each leg's net_cost; payout
at resolution is fee-free. So leg P&L = payout − net_cost.

Three P&L series per (threshold, [station]):
  WINNER-YES alone  — buy the favorite at T and hold (your exact bet)
  FULL basket       — winner + fade every other bucket
  FADE-NO alone     — just the NO legs

Reports:
  1. OVERALL: each penny 70..99 → n, FULL-basket avg net, WINNER-YES avg
     net, basket win-rate.
  2. PER-STATION: same table per station + a recommended trigger (the T
     with the best FULL-basket avg net at n ≥ --min-n).

Run:
  python analyze_basket_sweep.py
  python analyze_basket_sweep.py --min-n 5 --station KNYC
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

DATA = Path("data")
SWEEP_LOG = DATA / "basket_sweep_log.jsonl"
FORWARD_LOG = DATA / "forward_log.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _rounded_obs(actual_c: float, unit: str) -> int:
    if unit == "F":
        return int(math.floor(actual_c * 9.0 / 5.0 + 32.0 + 0.1))
    return int(math.floor(actual_c + 0.05))


def _bucket_won(kind: str, threshold: int, actual_int: int, unit: str) -> bool:
    if kind == "low_tail":
        return actual_int <= threshold
    if kind == "high_tail":
        return actual_int >= threshold
    if unit == "C":
        return actual_int == threshold
    return threshold <= actual_int <= threshold + 1


def _leg_pnl(leg: dict, actual_int: int, unit: str) -> float | None:
    if not leg:
        return None
    # Exclude any leg that was a fabricated top-of-book fill (pre-fix
    # logging). Only real depth-walked fills count toward results.
    if leg.get("depth_source") == "top_of_book_fallback":
        return None
    kind = leg.get("bucket_kind")
    thr = leg.get("bucket_threshold")
    shares = float(leg.get("shares", 0))
    net_cost = float(leg.get("net_cost", 0))
    if thr is None or shares <= 0:
        return None
    won = _bucket_won(kind, int(thr), actual_int, unit)
    if leg.get("side") == "YES":
        payout = shares if won else 0.0
    else:  # NO
        payout = shares if not won else 0.0
    return payout - net_cost


def summarize(label: str, xs: list[float]) -> str:
    if not xs:
        return f"  {label:22s} n=0"
    n = len(xs)
    total = sum(xs)
    wins = sum(1 for v in xs if v > 0)
    return (f"  {label:22s} n={n:>3}  net=${total:+8.2f}  "
            f"avg=${total/n:+7.3f}  win={100*wins/n:3.0f}%")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=5,
                    help="Min baskets at a threshold for a per-station "
                         "recommendation (default 5).")
    ap.add_argument("--station", default=None,
                    help="Restrict per-station detail to one station id.")
    args = ap.parse_args(argv)

    rows = load_jsonl(SWEEP_LOG)
    resolutions = load_jsonl(FORWARD_LOG)

    units: dict[str, str] = {}
    try:
        sys.path.insert(0, ".")
        from weather_bot.locations import STATIONS_BY_ID
        units = {sid: s.unit for sid, s in STATIONS_BY_ID.items()}
    except Exception:
        pass

    res_by_key: dict[tuple, float] = {}
    for r in resolutions:
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            res_by_key[k] = float(r["actual_obs_c"])

    # Accumulators: [scope][threshold] -> list of (full, winner_yes, fade_no)
    overall: dict[int, list[tuple]] = defaultdict(list)
    per_station: dict[str, dict[int, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    scored = 0
    unresolved = 0

    for row in rows:
        sid = row.get("station_id")
        target = row.get("target")
        td = row.get("target_date")
        T = row.get("entry_threshold")
        if T is None:
            continue
        actual_c = res_by_key.get((sid, target, td))
        if actual_c is None:
            unresolved += 1
            continue
        unit = units.get(sid, "C")
        ai = _rounded_obs(actual_c, unit)

        wy = _leg_pnl(row.get("winner"), ai, unit)
        if wy is None:
            continue
        no_sum = 0.0
        for leg in row.get("fade_no", []):
            p = _leg_pnl(leg, ai, unit)
            if p is not None:
                no_sum += p
        full = wy + no_sum
        scored += 1
        overall[int(T)].append((full, wy, no_sum))
        per_station[sid][int(T)].append((full, wy, no_sum))

    print("=" * 72)
    print("BASKET THRESHOLD SWEEP  —  P&L vs entry threshold (net of fees)")
    print("=" * 72)
    print(f"Logged baskets: {len(rows)}   scored: {scored}   "
          f"unresolved (still open): {unresolved}")
    print()
    print("OVERALL  (each row = a candidate trigger threshold)")
    print(f"  {'T':>4} {'n':>4} {'FULL avg':>10} {'FULL net':>10} "
          f"{'YESonly avg':>12} {'basket win%':>11}")
    for T in range(70, 100):
        lst = overall.get(T, [])
        if not lst:
            continue
        n = len(lst)
        full = [x[0] for x in lst]
        wy = [x[1] for x in lst]
        wins = sum(1 for v in full if v > 0)
        print(f"  {T:>4} {n:>4} {sum(full)/n:>+10.3f} {sum(full):>+10.2f} "
              f"{sum(wy)/n:>+12.3f} {100*wins/n:>10.0f}%")
    print()

    # Per-station detail + recommended trigger
    print("=" * 72)
    print("PER-STATION  (recommended trigger = best FULL avg at n ≥ "
          f"{args.min_n})")
    print("=" * 72)
    stations = sorted(per_station)
    if args.station:
        stations = [s for s in stations if s == args.station]
    for sid in stations:
        by_T = per_station[sid]
        total_n = sum(len(v) for v in by_T.values())
        # recommendation
        best_T, best_avg = None, None
        for T, lst in by_T.items():
            if len(lst) < args.min_n:
                continue
            avg = sum(x[0] for x in lst) / len(lst)
            if best_avg is None or avg > best_avg:
                best_T, best_avg = T, avg
        rec = (f"trigger≈${best_T/100:.2f}  (FULL avg ${best_avg:+.3f})"
               if best_T is not None
               else f"insufficient data (need n≥{args.min_n} at some T)")
        print(f"\n{sid}  [{units.get(sid,'C')}]  total baskets={total_n}  →  {rec}")
        # detail table (only if a single station requested, else compact)
        show_T = sorted(by_T)
        for T in show_T:
            lst = by_T[T]
            n = len(lst)
            full = [x[0] for x in lst]
            wy = [x[1] for x in lst]
            wins = sum(1 for v in full if v > 0)
            flag = "  <== rec" if T == best_T else ""
            print(f"    T={T:>3}  n={n:>3}  FULL avg ${sum(full)/n:+7.3f}  "
                  f"YESonly avg ${sum(wy)/n:+7.3f}  win {100*wins/n:3.0f}%{flag}")
    if not stations:
        print("  (no scored baskets yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
