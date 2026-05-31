"""Consensus-BASKET P&L analyzer (hold-to-resolution).

The consensus_basket strategy fires once a bucket's YES ≥ 0.85 (the
emerged winner): BUY YES $5 on the winner + BUY NO $5 on every other
bucket, then HOLDS every leg to resolution (no early exit). This scores
the realized P&L net of Polymarket taker fees by joining:

  data/consensus_basket_log.jsonl   — leg fills (one row per leg)
  data/forward_log.jsonl            — WUG resolutions (actual_obs_c)

A "basket" is keyed (station_id, target, target_date). For each leg at
resolution:
  - YES leg pays $1/share iff its bucket WON.
  - NO  leg pays $1/share iff its bucket did NOT win.

Three views, because they answer different questions:
  1. WINNER-YES leg ALONE, held to resolution
     → this is EXACTLY "buy the favorite at ~0.85 and hold". If favorites
       are fairly priced this lands near breakeven; if there's a
       favorite-longshot mispricing it goes positive.
  2. NO legs (fade every other bucket) — the rest of the basket.
  3. FULL basket = (1)+(2) summed per event.

Run:
  python analyze_consensus_basket.py
  python analyze_consensus_basket.py --since 2026-05-31
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
BASKET_LOG = DATA / "consensus_basket_log.jsonl"
FORWARD_LOG = DATA / "forward_log.jsonl"

FEE_RATE = 0.05  # Polymarket weather taker fee rate (see weather_bot/fees.py)


def taker_fee(shares: float, price: float) -> float:
    if not (0.0 < price < 1.0) or shares <= 0:
        return 0.0
    return shares * FEE_RATE * price * (1.0 - price)


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
    """Realized P&L for one held-to-resolution leg, net of taker fee.
    Returns None if the bucket can't be scored."""
    # Exclude fabricated top-of-book fills (pre-fix): only count real
    # depth-walked fills — an empty book is a no-op, never a $5 trade.
    if leg.get("depth_source") == "top_of_book_fallback":
        return None
    kind = leg.get("bucket_kind")
    thr = leg.get("bucket_threshold")
    side = leg.get("side")
    shares = float(leg.get("shares", 0))
    fill = float(leg.get("fill_price", 0))
    if thr is None or shares <= 0 or not (0.0 < fill < 1.0):
        return None
    won = _bucket_won(kind, int(thr), actual_int, unit)
    cost = shares * fill + taker_fee(shares, fill)
    if side == "YES":
        payout = shares if won else 0.0
    else:  # NO leg pays when the bucket does NOT win
        payout = shares if not won else 0.0
    return payout - cost


def stat(label: str, xs: list[float]) -> None:
    if not xs:
        print(f"  {label:30s} n=0")
        return
    n = len(xs)
    total = sum(xs)
    wins = sum(1 for v in xs if v > 0)
    print(f"  {label:30s} n={n:>3}  net=${total:+9.2f}  "
          f"avg=${total/n:+7.3f}  win={wins}/{n} ({100*wins/n:.0f}%)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="Only count legs with ts_utc >= this ISO prefix.")
    args = ap.parse_args(argv)

    legs = [r for r in load_jsonl(BASKET_LOG)
            if r.get("result") == "filled"
            and not (args.since and r.get("ts_utc", "") < args.since)]
    resolutions = load_jsonl(FORWARD_LOG)

    # Station unit lookup
    units: dict[str, str] = {}
    try:
        sys.path.insert(0, ".")
        from weather_bot.locations import STATIONS_BY_ID
        units = {sid: s.unit for sid, s in STATIONS_BY_ID.items()}
    except Exception:
        pass

    # Resolutions keyed by (sid, target, date) → actual_obs_c
    res_by_key: dict[tuple, float] = {}
    for r in resolutions:
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            res_by_key[k] = float(r["actual_obs_c"])

    # Group legs into baskets keyed (sid, target, date)
    baskets: dict[tuple, list[dict]] = defaultdict(list)
    for lg in legs:
        k = (lg.get("station_id"), lg.get("target"), lg.get("target_date"))
        baskets[k].append(lg)

    winner_yes_pnl: list[float] = []   # buy-the-favorite-and-hold, ALONE
    no_legs_pnl: list[float] = []      # fade-the-rest, summed per basket
    full_basket_pnl: list[float] = []  # the whole thing, summed per basket
    open_baskets = 0
    scored_baskets = 0

    for k, blegs in baskets.items():
        sid, target, td = k
        actual_c = res_by_key.get((sid, target, td))
        if actual_c is None:
            open_baskets += 1
            continue
        unit = units.get(sid, "C")
        ai = _rounded_obs(actual_c, unit)

        yes_sum = 0.0
        no_sum = 0.0
        scored_any = False
        for lg in blegs:
            pnl = _leg_pnl(lg, ai, unit)
            if pnl is None:
                continue
            scored_any = True
            if lg.get("side") == "YES":
                yes_sum += pnl
            else:
                no_sum += pnl
        if not scored_any:
            open_baskets += 1
            continue
        scored_baskets += 1
        winner_yes_pnl.append(yes_sum)
        no_legs_pnl.append(no_sum)
        full_basket_pnl.append(yes_sum + no_sum)

    print("=" * 70)
    print("Consensus-BASKET P&L  (hold-to-resolution, net of taker fees)")
    if args.since:
        print(f"Filtered to ts_utc >= {args.since}")
    print("=" * 70)
    print(f"Legs in scope: {len(legs)}   "
          f"Baskets: {len(baskets)} "
          f"(scored {scored_baskets}, still-open {open_baskets})")
    print()
    print("Per-basket P&L (each n = one event):")
    stat("WINNER-YES alone (hold)", winner_yes_pnl)
    stat("NO legs (fade the rest)", no_legs_pnl)
    stat("FULL basket", full_basket_pnl)
    print()
    print("Interpretation:")
    print("  'WINNER-YES alone (hold)' IS the buy-the-favorite-and-hold bet.")
    print("  Near $0 avg ⇒ favorite is fairly priced (price = win prob).")
    print("  Persistently +$ ⇒ favorite-longshot mispricing worth a look.")
    print("  'FULL basket' adds the cost/benefit of fading every other leg.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
