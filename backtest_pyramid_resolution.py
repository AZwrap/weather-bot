"""Pyramid on resolution-day confirmation backtest.

For each resolved (station, target, target_date), identify the
winning bucket. Simulate buying YES on that bucket at multiple time
points before resolution (T-12h, T-6h, T-2h, T-1h). Compute per-
share and per-$5-fill profit at each layer.

Compares pyramid (multiple layers) vs single-layer (one entry only).
The pyramid only fires AFTER METAR confirmation, so losing buckets
don't enter — that's the alpha source.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from datetime import timedelta
from collections import defaultdict

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

LAYER_TIMES = [12, 6, 4, 2, 1]  # hours before resolution
SIZE_PER_LAYER = 5.0

recs = load_records()

# Group by bucket key
buckets = defaultdict(list)
for r in recs:
    if r.bucket_snapshots is None:
        continue
    if r.actual_obs_c is None:
        continue
    for snap in r.bucket_snapshots:
        if snap.yes_ask is None or snap.yes_bid is None:
            continue
        key = (r.station_id, r.target, r.target_date.isoformat(),
               snap.kind, snap.threshold)
        buckets[key].append((r.issue_time_utc, snap, r.actual_obs_c))

# Find winning buckets (the one bucket per station-day where bucket_won=True)
winning_buckets = []
for key, snaps in buckets.items():
    if not snaps:
        continue
    snaps.sort()
    sid, target, date, kind, thr = key
    station = STATIONS_BY_ID.get(sid)
    if station is None:
        continue
    actual_obs = snaps[-1][2]
    actual_int = _rounded_observation(actual_obs, station.unit)
    fake_snap = type("S", (), {"kind": kind, "threshold": thr})
    if bucket_won(fake_snap, actual_int, station.unit):
        winning_buckets.append((key, snaps))

print(f"resolved bucket timelines: {len(buckets)}")
print(f"winning buckets: {len(winning_buckets)}")
print()


def lookup_at_t(snaps, target_time):
    """Find snap closest to target_time but not after."""
    best = None
    for ts, snap, _ in snaps:
        if ts > target_time:
            break
        best = (ts, snap)
    return best


# For each winning bucket, simulate layer fills at each LAYER_TIME
print(f"{'layer (hrs to res)':20s} {'n':>5s} {'avg_ask':>8s} "
      f"{'win/share':>10s} {'pnl':>8s} {'roi':>7s}")
total_per_layer = {}
for hrs in LAYER_TIMES:
    fills = []
    for key, snaps in winning_buckets:
        last_ts = snaps[-1][0]
        target_t = last_ts - timedelta(hours=hrs)
        result = lookup_at_t(snaps, target_t)
        if result is None:
            continue
        ts, snap = result
        ask = float(snap.yes_ask)
        if ask <= 0 or ask >= 1:
            continue
        # Taker fill at ask
        shares = SIZE_PER_LAYER / ask
        # Bucket wins → payoff $1/share, cost $ask/share
        per_share_profit = 1.0 - ask
        fills.append((ask, per_share_profit, SIZE_PER_LAYER))
    if not fills:
        continue
    n = len(fills)
    avg_ask = sum(a for a, _, _ in fills) / n
    avg_per_share = sum(p for _, p, _ in fills) / n
    total_pnl = sum(p / a * s for a, p, s in fills)
    total_size = sum(s for _, _, s in fills)
    roi = total_pnl / total_size * 100 if total_size else 0
    total_per_layer[hrs] = (n, total_pnl, total_size)
    print(f"T-{hrs:2d}h                {n:>5d}   ${avg_ask:>5.3f}   "
          f"${avg_per_share:>7.3f}   ${total_pnl:>6.2f}   {roi:>5.1f}%")

print()
print("Pyramid combinations (sum of layers, $5 each):")
print(f"{'layers':30s} {'total_size':>11s} {'total_pnl':>10s} {'roi':>7s}")
combos = [
    [12], [6], [4], [2], [1],
    [12, 6], [12, 4], [12, 2], [12, 1],
    [6, 2], [6, 1], [4, 1], [2, 1],
    [12, 6, 2], [12, 6, 1], [12, 4, 1], [6, 2, 1], [4, 2, 1],
    [12, 6, 2, 1], [12, 4, 2, 1], [6, 4, 2, 1],
    [12, 6, 4, 2, 1],
]
for layers in combos:
    total_pnl = 0
    total_size = 0
    for h in layers:
        if h in total_per_layer:
            n, pnl, sz = total_per_layer[h]
            total_pnl += pnl
            total_size += sz
    if total_size == 0:
        continue
    roi = total_pnl / total_size * 100
    layer_str = ",".join(f"T-{h}h" for h in layers)
    print(f"{layer_str:30s} ${total_size:>9.0f}   ${total_pnl:>7.2f}   {roi:>5.1f}%")

# Compare against the alternative: METAR feedback already counted
# this profit. Pyramid is layer-additive ON TOP of METAR feedback baseline.
# So the "marginal" pyramid value = adding T-12h or T-6h on top of T-2h/T-1h alone.
print()
print("Marginal value of EARLIER layers (vs T-2h alone):")
if 2 in total_per_layer:
    base_n, base_pnl, base_sz = total_per_layer[2]
    print(f"Baseline T-2h alone: ${base_pnl:.2f} on {base_n} fills (${base_sz:.0f} deployed)")
    for hrs in [12, 6, 4]:
        if hrs not in total_per_layer:
            continue
        n_e, pnl_e, sz_e = total_per_layer[hrs]
        print(f"  Adding T-{hrs}h:  +${pnl_e:.2f} pnl, +${sz_e:.0f} size, "
              f"marginal ROI = {pnl_e/sz_e*100:.1f}%")
