"""Backtest #12: how fast do Polymarket weather markets converge to truth
in the final hours before resolution?

For each (station, target, target_date, bucket_kind, threshold) where the
underlying bucket is resolved, compute outcome (1 if bucket actually won,
0 otherwise). For each snapshot in that bucket's timeline, compute
time-to-resolution and |market_yes_implied - outcome|.

Bin by time-to-resolution; report mean absolute error per bin.

If error at T-2h is ≥10pp, our intraday-METAR-feedback window is large
enough to exploit. If it's ≤5pp, markets reprice fast and the window
is too small.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from datetime import timedelta
import statistics as stats

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

recs = load_records()
print(f"records: {len(recs)}")

# Build bucket-key → snapshot series map
buckets: dict[tuple, list] = defaultdict(list)
for r in recs:
    if r.bucket_snapshots is None or r.actual_obs_c is None:
        continue
    for snap in r.bucket_snapshots:
        key = (r.station_id, r.target, r.target_date.isoformat(),
               snap.kind, snap.threshold)
        buckets[key].append((r, snap))
for key in buckets:
    buckets[key].sort(key=lambda rs: rs[0].issue_time_utc)

print(f"resolved bucket timelines: {len(buckets)}")

# For each bucket, compute outcome (1 if YES won, 0 if not)
# Then for each snapshot, log (time_to_resolution, market_implied, outcome)
data_points = []
for key, series in buckets.items():
    # Compute outcome from the latest record's actual_obs_c
    last_record, last_snap = series[-1]
    station = STATIONS_BY_ID.get(last_record.station_id)
    if station is None:
        continue
    actual_int = _rounded_observation(last_record.actual_obs_c, station.unit)
    outcome = 1 if bucket_won(last_snap, actual_int, station.unit) else 0
    last_time = last_record.issue_time_utc

    for r, snap in series:
        if snap.yes_bid is None or snap.yes_ask is None:
            continue
        market_yes_implied = (float(snap.yes_bid) + float(snap.yes_ask)) / 2
        time_to_resolution = (last_time - r.issue_time_utc).total_seconds() / 3600
        data_points.append((time_to_resolution, market_yes_implied, outcome))

print(f"data points (snapshot, outcome): {len(data_points)}")
print()

# Bin by time-to-resolution
bins = [(0, 1, "T-0 to T-1h"),
        (1, 3, "T-1h to T-3h"),
        (3, 6, "T-3h to T-6h"),
        (6, 12, "T-6h to T-12h"),
        (12, 24, "T-12h to T-24h"),
        (24, 48, "T-24h to T-48h"),
        (48, 168, "T-48h to T-7d")]

print(f"{'window':18s} {'n':>5s} {'mean_implied':>13s} {'mean_outcome':>13s} "
      f"{'mae':>8s} {'rmse':>8s}")
for lo, hi, label in bins:
    points = [(im, oc) for ttr, im, oc in data_points if lo <= ttr < hi]
    if not points:
        continue
    n = len(points)
    mean_implied = sum(im for im, _ in points) / n
    mean_outcome = sum(oc for _, oc in points) / n
    mae = sum(abs(im - oc) for im, oc in points) / n
    rmse = (sum((im - oc)**2 for im, oc in points) / n) ** 0.5
    print(f"{label:18s} {n:>5d}   {mean_implied:>10.3f}   "
          f"{mean_outcome:>10.3f}   {mae:>6.3f}   {rmse:>6.3f}")

print()
print("Interpretation: MAE = mean absolute error of midprice vs actual outcome.")
print("  MAE near 0 = markets near truth, no info advantage.")
print("  MAE 0.05 (5pp) = small window, marginal strategy.")
print("  MAE 0.10+ = significant info advantage, large window.")

# Stratify by side: are losers (outcome=0, market still > 0) the salvage opportunity?
print()
print("Stratified by outcome (true=hit / false=miss):")
print(f"{'window':18s} {'true_n':>7s} {'true_imp':>9s} {'false_n':>8s} {'false_imp':>10s}")
for lo, hi, label in bins:
    points = [(im, oc) for ttr, im, oc in data_points if lo <= ttr < hi]
    if not points:
        continue
    true_pts = [im for im, oc in points if oc == 1]
    false_pts = [im for im, oc in points if oc == 0]
    n_t, n_f = len(true_pts), len(false_pts)
    avg_t = sum(true_pts)/n_t if n_t else 0
    avg_f = sum(false_pts)/n_f if n_f else 0
    print(f"{label:18s} {n_t:>7d}   {avg_t:>6.3f}   {n_f:>8d}   {avg_f:>7.3f}")

print()
print("If true_imp << 1.0 close to T-0: market underprices winners → ADD opportunity.")
print("If false_imp >> 0.0 close to T-0: market overprices losers → SALVAGE opportunity.")
