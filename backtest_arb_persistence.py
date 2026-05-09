"""How long do bucket-sum-to-1 YES-side arbs persist?

For each detected arb (sum_yes_ask < 0.97), check if it's still
present in the NEXT snapshot for that (station, target, target_date).
Tracks the margin trajectory over time — does the arb stay open, or
close immediately?

Also computes:
- Frequency by hour-of-day (do arbs cluster around ECMWF rolls?)
- Per-station distribution
- Best persistent arbs (margin × duration)
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from datetime import datetime

from weather_bot.forward_log import load_records

recs = load_records()

# Group all snapshots into time-series per (station, target, date)
groups: dict[tuple, list] = defaultdict(list)
for r in recs:
    if r.bucket_snapshots is None:
        continue
    if not all(s.yes_ask is not None for s in r.bucket_snapshots):
        continue
    if len(r.bucket_snapshots) < 3:
        continue
    sum_yes_ask = sum(float(s.yes_ask) for s in r.bucket_snapshots)
    n = len(r.bucket_snapshots)
    key = (r.station_id, r.target, r.target_date.isoformat())
    groups[key].append((r.issue_time_utc, sum_yes_ask, n))

# Sort each group by time
for key in groups:
    groups[key].sort()

# Find arbs (sum < 1.00) and check persistence
print("Persistence analysis: how long does each arb stay open?\n")
print(f"{'station':10s} {'target':4s} {'date':12s} {'first_t':5s} {'first_sum':>10s} "
      f"{'persisted':>10s} {'final_sum':>10s} {'best_margin':>11s}")

persistent_arbs = []
flash_arbs = []
n_total_arbs = 0
n_arbs_persisted_1plus = 0
n_arbs_persisted_2plus = 0

for key, series in groups.items():
    in_arb = False
    arb_start_idx = -1
    arb_start_sum = 0
    arb_min_sum = 0

    for i, (ts, s, n) in enumerate(series):
        if s < 0.97 and not in_arb:
            # New arb detected
            in_arb = True
            arb_start_idx = i
            arb_start_sum = s
            arb_min_sum = s
            n_total_arbs += 1
        elif s < 0.97 and in_arb:
            # Arb persists
            arb_min_sum = min(arb_min_sum, s)
        elif s >= 0.97 and in_arb:
            # Arb closed
            duration_snapshots = i - arb_start_idx
            in_arb = False
            margin = 1.0 - arb_min_sum

            if duration_snapshots >= 2:
                n_arbs_persisted_2plus += 1
                persistent_arbs.append((key, series[arb_start_idx][0],
                                        duration_snapshots, margin, arb_start_sum, arb_min_sum))
            if duration_snapshots >= 1:
                n_arbs_persisted_1plus += 1

            if duration_snapshots == 1:
                flash_arbs.append((key, series[arb_start_idx][0], margin))

    # Handle arb still open at end of series
    if in_arb:
        duration_snapshots = len(series) - arb_start_idx
        margin = 1.0 - arb_min_sum
        if duration_snapshots >= 2:
            n_arbs_persisted_2plus += 1
            persistent_arbs.append((key, series[arb_start_idx][0],
                                    duration_snapshots, margin, arb_start_sum, arb_min_sum))
        if duration_snapshots >= 1:
            n_arbs_persisted_1plus += 1

print()
print(f"Total YES-side arb episodes detected: {n_total_arbs}")
print(f"Persisted ≥1 snapshot (≥20 min): {n_arbs_persisted_1plus} ({n_arbs_persisted_1plus/max(1,n_total_arbs)*100:.0f}%)")
print(f"Persisted ≥2 snapshots (≥40 min): {n_arbs_persisted_2plus} ({n_arbs_persisted_2plus/max(1,n_total_arbs)*100:.0f}%)")
print(f"Flash arbs (closed within 1 snapshot, ≤20 min): {len(flash_arbs)}")
print()

if persistent_arbs:
    print("Persistent arb episodes (≥40 min open):")
    print(f"{'station':10s} {'target':4s} {'date':12s} {'first_t':>15s} "
          f"{'snaps':>5s} {'best_margin':>11s}")
    persistent_arbs.sort(key=lambda x: -x[3])
    for (sid, target, date), ts, dur, margin, start_s, min_s in persistent_arbs[:20]:
        print(f"{sid:10s} {target:4s} {date:12s} {ts.strftime('%m-%d %H:%M'):>15s} "
              f"{dur:>5d}   ${margin:>+8.4f}")

# Frequency by hour
print()
print("Arb frequency by hour-of-day (UTC):")
arbs_by_hour = defaultdict(int)
total_by_hour = defaultdict(int)
for key, series in groups.items():
    for ts, s, n in series:
        hour = ts.hour
        total_by_hour[hour] += 1
        if s < 0.97:
            arbs_by_hour[hour] += 1

print(f"{'hour':>4s} {'n_arbs':>7s} {'n_total':>8s} {'arb_rate':>9s}")
for hour in range(24):
    n_arb = arbs_by_hour.get(hour, 0)
    n_total = total_by_hour.get(hour, 0)
    rate = n_arb / max(1, n_total) * 100
    bar = "█" * int(rate * 5)
    print(f"{hour:>4d} {n_arb:>7d} {n_total:>8d} {rate:>7.1f}% {bar}")

# Per-station
print()
print("Per-station arb frequency (top 15):")
arbs_by_station = defaultdict(int)
total_by_station = defaultdict(int)
for (sid, target, date), series in groups.items():
    for ts, s, n in series:
        total_by_station[sid] += 1
        if s < 0.97:
            arbs_by_station[sid] += 1
print(f"{'station':10s} {'n_arbs':>7s} {'n_total':>8s} {'arb_rate':>9s}")
rows = sorted(arbs_by_station.items(), key=lambda x: -x[1])
for sid, n_arb in rows[:15]:
    n_total = total_by_station[sid]
    rate = n_arb / max(1, n_total) * 100
    print(f"{sid:10s} {n_arb:>7d} {n_total:>8d} {rate:>7.1f}%")
