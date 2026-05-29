"""Consensus-YES P&L analyzer.

Joins the three logs the consensus_yes strategy produces:
  data/consensus_yes_log.jsonl       — entries (BUY YES fills)
  data/consensus_yes_exit_log.jsonl  — trailing-stop exits (SELL YES)
  data/forward_log.jsonl             — WUG resolutions (actual_obs_c)

and reports realized + mark-to-resolution P&L, net of Polymarket taker
fees, broken down by:
  - outcome (sold / resolved-held / still-open)
  - exit trigger (ws_push vs sweep)
  - pre/post the 2026-05-29 bid-fix (commit c830257) so we can measure
    whether tracking the trailing stop on the BID flipped the strategy
    from net-negative to net-positive.

Run:
  python analyze_consensus_yes.py
  python analyze_consensus_yes.py --since 2026-05-29T12:33   # post-fix only

Entry↔exit are matched on (station_id, target_date, bucket_label),
which is unique per consensus_yes position (one YES bet per bucket per
day via the is_open dedupe).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

DATA = Path("data")
ENTRY_LOG = DATA / "consensus_yes_log.jsonl"
EXIT_LOG = DATA / "consensus_yes_exit_log.jsonl"
FORWARD_LOG = DATA / "forward_log.jsonl"

# Deploy timestamp of the bid-fix (commit c830257). Exits at/after this
# used the corrected bid-tracking logic.
BID_FIX_CUTOFF = "2026-05-29T12:33:00"

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
    import math
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="Only count entries/exits with ts_utc >= this ISO prefix.")
    args = ap.parse_args(argv)

    entries = load_jsonl(ENTRY_LOG)
    exits = load_jsonl(EXIT_LOG)
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
    res_by_key: dict[tuple[str, str, str], float] = {}
    for r in resolutions:
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            res_by_key[k] = float(r["actual_obs_c"])

    # Exits keyed by (sid, date, bucket_label) → exit record
    exit_by_key: dict[tuple[str, str, str], dict] = {}
    for x in exits:
        if x.get("result") != "sold":
            continue
        if args.since and (x.get("ts_utc", "") < args.since):
            continue
        k = (x.get("station_id"), x.get("target_date"), x.get("bucket_label"))
        exit_by_key[k] = x

    filled_entries = [
        e for e in entries
        if e.get("result") == "filled"
        and not (args.since and e.get("ts_utc", "") < args.since)
    ]

    # Tally
    sold_pnl: list[float] = []
    resolved_pnl: list[float] = []
    open_count = 0
    by_trigger: dict[str, list[float]] = defaultdict(list)

    for e in filled_entries:
        sid = e.get("station_id")
        td = e.get("target_date")
        target = e.get("target")
        label = e.get("bucket_label")
        kind = e.get("bucket_kind")
        thr = e.get("bucket_threshold")
        shares = float(e.get("shares", 0))
        entry_fill = float(e.get("fill_price", 0))
        entry_fee = taker_fee(shares, entry_fill)
        cost = shares * entry_fill + entry_fee

        key = (sid, td, label)
        x = exit_by_key.get(key)
        if x is not None:
            # Realized via SELL
            exit_fill = float(x.get("fill_price", 0))
            exit_fee = taker_fee(shares, exit_fill)
            proceeds = shares * exit_fill - exit_fee
            pnl = proceeds - cost
            sold_pnl.append(pnl)
            by_trigger[x.get("trigger", "?")].append(pnl)
            continue

        # Not sold — mark to resolution if available
        actual_c = res_by_key.get((sid, target, td))
        if actual_c is not None:
            unit = units.get(sid, "C")
            ai = _rounded_obs(actual_c, unit)
            won = _bucket_won(kind, int(thr), ai, unit) if thr is not None else False
            value = shares * (1.0 if won else 0.0)
            pnl = value - cost
            resolved_pnl.append(pnl)
            continue

        open_count += 1

    # Report
    def stat(label: str, xs: list[float]):
        if not xs:
            print(f"  {label:24s} n=0")
            return
        n = len(xs)
        total = sum(xs)
        wins = sum(1 for v in xs if v > 0)
        print(f"  {label:24s} n={n:>3}  net=${total:+8.2f}  "
              f"avg=${total/n:+6.3f}  win={wins}/{n} ({100*wins/n:.0f}%)")

    print("=" * 64)
    print("Consensus-YES P&L  (net of Polymarket taker fees)")
    if args.since:
        print(f"Filtered to ts_utc >= {args.since}")
    print("=" * 64)
    print(f"Filled entries in scope: {len(filled_entries)}")
    print()
    print("By outcome:")
    stat("Sold (trailing exit)", sold_pnl)
    stat("Held to resolution", resolved_pnl)
    print(f"  {'Still open':24s} n={open_count:>3}  (unsold + unresolved)")
    print()

    combined = sold_pnl + resolved_pnl
    stat("ALL realized", combined)
    print()

    print("Sold exits by trigger:")
    for trig in sorted(by_trigger):
        stat(f"  {trig}", by_trigger[trig])
    print()

    # Pre/post bid-fix comparison (only meaningful without --since)
    if not args.since:
        print("=" * 64)
        print("Bid-fix effect (exits before vs after 2026-05-29 12:33 UTC)")
        print("=" * 64)
        pre = [x for x in exits if x.get("result") == "sold"
               and x.get("ts_utc", "") < BID_FIX_CUTOFF]
        post = [x for x in exits if x.get("result") == "sold"
                and x.get("ts_utc", "") >= BID_FIX_CUTOFF]

        def exit_net(xs):
            # Use the log's net_total_usd (entry vs exit price * shares,
            # pre-fee) for a like-for-like comparison with what the bot
            # recorded at exit time.
            return sum(float(x.get("net_total_usd", 0)) for x in xs)

        for tag, xs in (("PRE-fix (ask-tracked)", pre), ("POST-fix (bid-tracked)", post)):
            if xs:
                n = len(xs)
                net = exit_net(xs)
                wins = sum(1 for x in xs if float(x.get("net_total_usd", 0)) > 0)
                print(f"  {tag:26s} n={n:>3}  net=${net:+8.2f}  "
                      f"avg=${net/n:+6.3f}  win={wins}/{n} ({100*wins/n:.0f}%)")
            else:
                print(f"  {tag:26s} n=0")
        print()
        print("  (PRE net uses the recorded net_total_usd; the bug was that")
        print("   it could be negative even when the trigger 'fired in profit'")
        print("   because the sale hit the bid. POST should show fewer/no")
        print("   sub-entry sales.)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
