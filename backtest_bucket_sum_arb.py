"""Bucket sum-to-1 arbitrage scan.

For each (station, target, target_date) at each snapshot, sum YES asks
and NO asks across all logged buckets. Arb conditions:

  Buy all YES:  sum(yes_ask) < 1.00  (pay X, receive $1, profit = 1-X)
  Buy all NO:   sum(no_ask)  < (N-1) (pay X, receive $(N-1), profit = N-1-X)

This is pure structural arb — works regardless of model accuracy. If it
shows up across multiple markets, it's free money. If it never does,
the markets are too efficient for this to be exploitable.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

from weather_bot.forward_log import load_records

recs = load_records()
print(f"records: {len(recs)}")

# Group snapshots by (station, target, target_date, issue_time)
# Each group should have ALL buckets for that station-target-date at one moment
groups: dict[tuple, list] = defaultdict(list)
for r in recs:
    if r.bucket_snapshots is None:
        continue
    issue_iso = r.issue_time_utc.isoformat()
    target_iso = r.target_date.isoformat()
    key = (r.station_id, r.target, target_iso, issue_iso)
    for snap in r.bucket_snapshots:
        groups[key].append(snap)

print(f"groups (station × target × date × snap): {len(groups)}")

yes_sums = []
no_sums = []
yes_arb_count = 0
no_arb_count = 0
yes_best_arb = None  # (margin, key, sum)
no_best_arb = None

for key, snaps in groups.items():
    # Need both yes_ask AND yes_bid (= 1 - no_ask) for every bucket
    have_all = all(s.yes_ask is not None and s.yes_bid is not None for s in snaps)
    if not have_all:
        continue
    n = len(snaps)
    if n < 3:
        continue  # skip degenerate (need at least 3 buckets to be meaningful)
    sum_yes_ask = sum(float(s.yes_ask) for s in snaps)
    sum_no_ask = sum(1.0 - float(s.yes_bid) for s in snaps)  # NO ask = 1 - YES bid
    yes_sums.append((n, sum_yes_ask, key))
    no_sums.append((n, sum_no_ask, key))
    if sum_yes_ask < 1.00:
        yes_arb_count += 1
        margin = 1.00 - sum_yes_ask
        if yes_best_arb is None or margin > yes_best_arb[0]:
            yes_best_arb = (margin, key, sum_yes_ask, n)
    if sum_no_ask < (n - 1):
        no_arb_count += 1
        margin = (n - 1) - sum_no_ask
        if no_best_arb is None or margin > no_best_arb[0]:
            no_best_arb = (margin, key, sum_no_ask, n)

print(f"\ngroups with full bid/ask data: {len(yes_sums)}")
print(f"YES arb opportunities (sum_yes_ask < 1.00):  {yes_arb_count}")
print(f"NO arb opportunities  (sum_no_ask < N-1):    {no_arb_count}")

# Distribution
import statistics as stats
yes_ratios = [(s/n) for n, s, _ in yes_sums]  # avg yes_ask per bucket
no_ratios = [s/(n-1) for n, s, _ in no_sums]  # ratio to no-arb threshold

print()
print("YES-ask SUM distribution (across N buckets):")
sums_only = [s for _, s, _ in yes_sums]
print(f"  min:    {min(sums_only):.4f}")
print(f"  median: {stats.median(sums_only):.4f}")
print(f"  mean:   {stats.mean(sums_only):.4f}")
print(f"  max:    {max(sums_only):.4f}")
print(f"  fair (=1.00) at quantile: {sum(1 for s in sums_only if s < 1.00)/len(sums_only)*100:.1f}%")

print()
print("NO-ask SUM / (N-1) ratio:")
print(f"  min:    {min(no_ratios):.4f}  (< 1.00 = arb)")
print(f"  median: {stats.median(no_ratios):.4f}")
print(f"  mean:   {stats.mean(no_ratios):.4f}")
print(f"  max:    {max(no_ratios):.4f}")
print(f"  arb (< 1.00) quantile: {sum(1 for r in no_ratios if r < 1.00)/len(no_ratios)*100:.1f}%")

# Best arb examples
print()
if yes_best_arb:
    margin, key, sum_, n = yes_best_arb
    print(f"BEST YES arb: margin=${margin:.4f} on {key}, sum=${sum_:.4f}, N={n}")
else:
    print("No YES-side arb opportunities found.")
if no_best_arb:
    margin, key, sum_, n = no_best_arb
    print(f"BEST NO arb:  margin=${margin:.4f} on {key}, sum=${sum_:.4f}, N={n}")
else:
    print("No NO-side arb opportunities found.")

# Tightest markets — closest to fair
yes_sums.sort(key=lambda t: t[1])
print()
print("Tightest YES-sums (closest to $1.00 from above):")
for n, s, key in yes_sums[:10]:
    over = s - 1.00
    print(f"  {key[0]:6s} {key[1]:5s} {key[2]} {key[3][:13]}  N={n:2d}  sum_yes_ask=${s:.4f}  excess=${over:+.4f}")

# Wide markets — bucket-sum-as-marketmaker-rake
yes_sums.sort(key=lambda t: t[1], reverse=True)
print()
print("Widest YES-sums (most overpriced bucket sets):")
for n, s, key in yes_sums[:10]:
    over = s - 1.00
    print(f"  {key[0]:6s} {key[1]:5s} {key[2]} {key[3][:13]}  N={n:2d}  sum_yes_ask=${s:.4f}  excess=${over:+.4f}")
