"""Backtest #11: how much $$ could have been salvaged on resolved-lost
positions if we'd exited at the bid T-4h before resolution?

For each Position p where p.closed and p.realized_profit_usd < 0:
  - Find the bucket's snapshot series (station, target, date, kind, thr)
  - Locate the snapshot at ~T-4h before the last snapshot
  - Compute salvage = bid * shares at that snapshot
    (for YES side: bid = yes_bid; for NO side: bid = 1 - yes_ask)
  - Real outcome was payoff=$0, so salvage represents money saved
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from datetime import timedelta

from weather_bot.forward_log import load_records
from weather_bot.positions import replay_maker

recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]

# Build bucket-key → snapshot series map
buckets: dict[tuple, list] = defaultdict(list)
for r in recs:
    if r.bucket_snapshots is None:
        continue
    for snap in r.bucket_snapshots:
        key = (r.station_id, r.target, r.target_date.isoformat(),
               snap.kind, snap.threshold)
        buckets[key].append((r, snap))
for key in buckets:
    buckets[key].sort(key=lambda rs: rs[0].issue_time_utc)

# Run replay_maker on the SAME settings as the dashboard / main backtest
positions = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
lost = [p for p in positions
        if p.closed and p.realized_profit_usd < 0
        and p.events and p.events[-1].action == "expire"]
print(f"resolved-lost expiry positions: {len(lost)}")
print(f"total realized loss on these:   ${sum(p.realized_profit_usd for p in lost):+.2f}")
print()

# For each lost position, find salvage at multiple time-to-resolution points
def lookup_at(snaps, last_time, hours_before):
    target_time = last_time - timedelta(hours=hours_before)
    # find snap closest to target_time but not after
    best = None
    for r, s in snaps:
        if r.issue_time_utc > target_time:
            break
        best = (r, s)
    return best

print(f"{'time':>5s} {'n_with_snap':>11s} {'avg_bid':>8s} "
      f"{'salvage_total':>13s} {'avg_salvage':>11s} "
      f"{'pct_recovered':>13s}")
for hrs in [1, 2, 4, 6, 8, 12]:
    salvages = []
    for p in lost:
        key = (p.station_id, p.target, p.target_date,
               p.bucket_kind, p.threshold)
        series = buckets.get(key, [])
        if len(series) < 2:
            continue
        last_r, _ = series[-1]
        result = lookup_at(series, last_r.issue_time_utc, hrs)
        if result is None:
            continue
        _, snap = result
        if p.side == "YES":
            bid = snap.yes_bid if snap.yes_bid is not None else 0.0
        else:
            bid = (1.0 - snap.yes_ask) if snap.yes_ask is not None else 0.0
        salvage_per_share = max(0.0, float(bid))
        salvage_value = salvage_per_share * p.shares
        salvages.append((salvage_value, p.position_usd, salvage_per_share))
    if not salvages:
        continue
    n = len(salvages)
    total_salvage = sum(v for v, _, _ in salvages)
    avg_salvage = total_salvage / n
    avg_bid = sum(b for _, _, b in salvages) / n
    total_lost = sum(c for _, c, _ in salvages)
    pct = total_salvage / total_lost * 100.0 if total_lost else 0
    print(f"T-{hrs}h   {n:>11d}   {avg_bid:>6.3f}   "
          f"${total_salvage:>10.2f}   ${avg_salvage:>8.2f}   "
          f"{pct:>10.1f}%")

# Bucket-by-station: where is the salvage concentrated?
print()
print("Salvage at T-4h, by station (where applicable):")
by_station: dict[str, list] = defaultdict(list)
for p in lost:
    key = (p.station_id, p.target, p.target_date,
           p.bucket_kind, p.threshold)
    series = buckets.get(key, [])
    if len(series) < 2:
        continue
    last_r, _ = series[-1]
    result = lookup_at(series, last_r.issue_time_utc, 4)
    if result is None:
        continue
    _, snap = result
    if p.side == "YES":
        bid = snap.yes_bid if snap.yes_bid is not None else 0.0
    else:
        bid = (1.0 - snap.yes_ask) if snap.yes_ask is not None else 0.0
    salvage = max(0.0, float(bid)) * p.shares
    by_station[p.station_id].append(salvage)

rows = sorted(
    [(stn, sum(svs), len(svs), sum(svs)/len(svs)) for stn, svs in by_station.items()],
    key=lambda r: -r[1]
)
print(f"{'station':10s} {'total_salv':>11s} {'n_pos':>6s} {'avg':>8s}")
for stn, total, n, avg in rows[:15]:
    print(f"{stn:10s} ${total:>9.2f}   {n:>4d}   ${avg:>6.2f}")
