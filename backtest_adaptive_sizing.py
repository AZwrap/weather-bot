"""Does the bot's win/loss series have serial correlation?

If wins cluster in time, adaptive sizing helps (size up after winners,
down after losers). If outcomes are independent, adaptive sizing just
adds noise.

Test:
  1. Order resolved positions chronologically.
  2. Compute autocorrelation of win/loss series at lags 1, 5, 10, 20.
  3. Compare win rate AFTER recent winners vs AFTER recent losers
     (rolling window of 10).
  4. Test mean reversion explicitly.

Caveat: we only have 1 resolved day (May 8). Intra-day correlation
will be dominated by station-clustering (synoptic patterns affect
multiple bets at once). True multi-day autocorrelation needs N≥7.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import statistics as stats
from collections import defaultdict
import math

from weather_bot.forward_log import load_records
from weather_bot.positions import replay_maker

recs = load_records()
elig = [r for r in recs if r.bucket_snapshots is not None]

positions = replay_maker(
    elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
    min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
    sigma_inflation_factor=1.4, taker_fallback=False,
)
resolved = [p for p in positions if p.closed]
print(f"resolved positions (with rung duplication): {len(resolved)}")

# DEDUPE: 4-rung ladders give 4 positions per bucket with identical outcomes,
# logged at consecutive issue times. That alone produces high lag-1 autocorr
# without any "hot hand" effect. Take ONE position per unique bucket.
seen = set()
unique_resolved = []
for p in sorted(resolved, key=lambda p: p.open_event.issue_time_utc):
    key = (p.station_id, p.target, p.target_date, p.bucket_kind, p.threshold)
    if key in seen:
        continue
    seen.add(key)
    unique_resolved.append(p)
print(f"unique buckets (one position each): {len(unique_resolved)}")
resolved = unique_resolved

# Order chronologically by open event time
resolved.sort(key=lambda p: p.open_event.issue_time_utc)
outcomes = [1 if p.realized_profit_usd > 0 else 0 for p in resolved]

n = len(outcomes)
mean_wr = sum(outcomes) / n
print(f"overall win rate: {mean_wr*100:.1f}%")
print()


def autocorrelation(series: list[int], lag: int) -> float:
    """Pearson autocorrelation at given lag."""
    if lag >= len(series):
        return 0.0
    x = series[:-lag]
    y = series[lag:]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


print("Autocorrelation of win/loss series:")
print(f"{'lag':>5s} {'autocorr':>10s} {'interpretation'}")
for lag in [1, 2, 5, 10, 20, 50]:
    if lag >= n:
        break
    r = autocorrelation(outcomes, lag)
    if abs(r) < 0.05:
        interp = "noise (no serial correlation)"
    elif r > 0:
        interp = "winners cluster — adaptive sizing helps"
    else:
        interp = "mean-reverting — adaptive sizing hurts"
    print(f"{lag:>5d} {r:>+9.4f}   {interp}")

print()
print("Conditional win rate by previous-10 outcomes:")
print(f"{'recent wins (last 10)':>22s} {'next_n':>7s} {'next_wr':>9s}")
window = 10
buckets_by_recent = defaultdict(list)  # recent_wr_bucket -> next_outcomes
for i in range(window, n):
    recent_wr = sum(outcomes[i-window:i]) / window
    if recent_wr < 0.10:
        bucket = "0-10% (cold)"
    elif recent_wr < 0.30:
        bucket = "10-30%"
    elif recent_wr < 0.50:
        bucket = "30-50%"
    elif recent_wr < 0.70:
        bucket = "50-70%"
    else:
        bucket = "70-100% (hot)"
    buckets_by_recent[bucket].append(outcomes[i])

for bucket in ["0-10% (cold)", "10-30%", "30-50%", "50-70%", "70-100% (hot)"]:
    nexts = buckets_by_recent.get(bucket, [])
    if not nexts:
        continue
    next_n = len(nexts)
    next_wr = sum(nexts) / next_n
    print(f"{bucket:>22s} {next_n:>7d}   {next_wr*100:>6.1f}%")

print()
print("Rolling-window adaptive sizing test:")
print("Strategy: size_multiplier = recent_wr_10 / overall_wr")
print("Skipping first 10 trades for warm-up.")
total_pnl_baseline = 0.0
total_pnl_adaptive = 0.0
total_size_baseline = 0.0
total_size_adaptive = 0.0
for i in range(window, n):
    p = resolved[i]
    base_size = p.position_usd
    recent_wr = sum(outcomes[i-window:i]) / window
    # Adaptive multiplier: scale by ratio to overall mean (clamped)
    mult = max(0.2, min(2.0, recent_wr / max(mean_wr, 0.01)))
    adaptive_size = base_size * mult
    # If win, profit scales linearly with size; if loss, loss scales linearly
    if outcomes[i] == 1:
        # profit = position_usd × per_dollar_return where per_dollar_return is
        # current realized_profit / position_usd
        per_dollar = p.realized_profit_usd / max(base_size, 0.001)
        total_pnl_baseline += base_size * per_dollar
        total_pnl_adaptive += adaptive_size * per_dollar
    else:
        per_dollar = p.realized_profit_usd / max(base_size, 0.001)  # negative
        total_pnl_baseline += base_size * per_dollar
        total_pnl_adaptive += adaptive_size * per_dollar
    total_size_baseline += base_size
    total_size_adaptive += adaptive_size

print(f"{'baseline':10s}  size=${total_size_baseline:7.0f}   pnl=${total_pnl_baseline:+8.2f}   "
      f"roi={total_pnl_baseline/total_size_baseline*100:+5.2f}%")
print(f"{'adaptive':10s}  size=${total_size_adaptive:7.0f}   pnl=${total_pnl_adaptive:+8.2f}   "
      f"roi={total_pnl_adaptive/max(total_size_adaptive,1)*100:+5.2f}%")
delta = total_pnl_adaptive - total_pnl_baseline
print(f"adaptive sizing delta: ${delta:+.2f}")
