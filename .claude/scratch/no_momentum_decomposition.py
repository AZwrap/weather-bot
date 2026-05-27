"""NO_momentum decomposition — find a profitable sub-cohort if one exists.

Question: 164 out_window resolved NO_momentum positions since 2026-05-14
have a 52.4% win rate (-$37.88 net). Breakeven at $0.78 entry needs 78%+
win rate. Is there a slice where we hit 78%+?

Dimensions to decompose by:
  1. station (49)
  2. region
  3. bucket_kind (mid / high_tail / low_tail)
  4. bucket position relative to extreme (distance from peak)
  5. target_date offset (today vs tomorrow)
  6. hours_to_peak at submit time
  7. day-of-week
  8. entry_price (fills below $0.78 = depth-walked)

Output:
  - For each dimension, table of n / win% / total_pnl / avg_pnl per bucket
  - Highlight any cohort with n≥10 AND win_rate≥0.78
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from zoneinfo import ZoneInfo
from weather_bot.intraday import trigger_local_hour
from weather_bot.locations import STATIONS_BY_ID

BREAKEVEN_WIN_RATE = 0.78  # at $0.78 entry

# ────────────────────────────────────────────────────────────────────────
# Load + filter dataset
# ────────────────────────────────────────────────────────────────────────
p = json.load(open("data/portfolio.json"))
positions = p["positions"]

# All resolved NO_momentum positions
nm = [
    pos for pos in positions
    if pos.get("strategy") == "NO_momentum"
    and pos.get("status") == "resolved"
    and pos.get("realized_pnl") is not None
    and pos.get("submitted_at")
]
print(f"Total resolved NO_momentum positions: {len(nm)}")

# Filter to post-threshold-change (May 14)
nm = [x for x in nm if x.get("submitted_at", "") >= "2026-05-14"]
print(f"  since 2026-05-14: {len(nm)}")


# Helper: was this position blocked by Layer 6 in_window?
def in_adverse_window(pos):
    sid = pos.get("station_id", "")
    station = STATIONS_BY_ID.get(sid)
    if not station: return None
    try:
        sub_dt = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        tdate = datetime.fromisoformat(pos["target_date"]).date()
        local_now = sub_dt.astimezone(ZoneInfo(station.timezone))
        if local_now.date() != tdate:
            return False  # forward placement = not in window
        peak_hour = trigger_local_hour(sid, tdate)
        if peak_hour is None:
            return None
        now_h = local_now.hour + local_now.minute / 60.0
        hours_to_peak = peak_hour - now_h
        return -1.0 <= hours_to_peak <= 2.0
    except Exception:
        return None


def hours_to_peak_at_submit(pos):
    sid = pos.get("station_id", "")
    station = STATIONS_BY_ID.get(sid)
    if not station: return None
    try:
        sub_dt = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        tdate = datetime.fromisoformat(pos["target_date"]).date()
        local_now = sub_dt.astimezone(ZoneInfo(station.timezone))
        if local_now.date() != tdate:
            return None
        peak_hour = trigger_local_hour(sid, tdate)
        if peak_hour is None: return None
        now_h = local_now.hour + local_now.minute / 60.0
        return peak_hour - now_h
    except Exception:
        return None


def target_date_offset(pos):
    """0 = same-day, 1 = tomorrow, etc. None if cannot compute station-local."""
    sid = pos.get("station_id", "")
    station = STATIONS_BY_ID.get(sid)
    if not station: return None
    try:
        sub_dt = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        tdate = datetime.fromisoformat(pos["target_date"]).date()
        local_today = sub_dt.astimezone(ZoneInfo(station.timezone)).date()
        return (tdate - local_today).days
    except Exception:
        return None


# Restrict to out_window only (apples-to-apples with the 52.4% finding)
out_window = []
for pos in nm:
    w = in_adverse_window(pos)
    if w is False:  # explicitly out-of-window
        out_window.append(pos)
    elif w is None:  # couldn't determine — treat as out-of-window
        out_window.append(pos)

print(f"  out_window (passes Layer 6 filter): {len(out_window)}")


def summary(label: str, rows: list, indent: str = "  ") -> dict:
    if not rows:
        return {}
    n = len(rows)
    wins = sum(1 for x in rows if x.get("realized_pnl", 0) > 0)
    total = sum(x.get("realized_pnl", 0) for x in rows)
    win_rate = wins / n
    return {
        "n": n, "wins": wins, "win_rate": win_rate,
        "total": total, "avg": total / n,
        "passes_breakeven": (n >= 10 and win_rate >= BREAKEVEN_WIN_RATE),
    }


def print_decomp(title: str, groups: dict[str, list]):
    print()
    print("=" * 76)
    print(f"BY {title}")
    print("=" * 76)
    print(f"{'group':<30} {'n':>5} {'wins':>5} {'win%':>7} {'total':>9} {'avg':>9} {'pass?':>6}")
    print("-" * 76)
    summaries = []
    for k, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
        s = summary(k, rows)
        if not s: continue
        marker = " ★ " if s["passes_breakeven"] else ""
        print(f"{str(k):<30} {s['n']:>5} {s['wins']:>5} "
              f"{s['win_rate']*100:>6.1f}% ${s['total']:>+7.2f} ${s['avg']:>+7.3f} {marker:>6}")
        summaries.append((k, s))
    # Highlight winners
    winners = [(k, s) for k, s in summaries if s["passes_breakeven"]]
    if winners:
        print()
        print(f"  ★ PROFITABLE COHORTS (n≥10, win_rate≥{BREAKEVEN_WIN_RATE*100:.0f}%):")
        for k, s in winners:
            print(f"    {k}: n={s['n']}, win={s['win_rate']*100:.1f}%, "
                  f"total=${s['total']:+.2f}, avg=${s['avg']:+.3f}")
    return summaries


# 1. BY STATION
by_station = defaultdict(list)
for pos in out_window:
    by_station[pos.get("station_id", "?")].append(pos)
print_decomp("STATION", by_station)

# 2. BY REGION
by_region = defaultdict(list)
for pos in out_window:
    by_region[pos.get("region", "?")].append(pos)
print_decomp("REGION", by_region)

# 3. BY BUCKET KIND
by_kind = defaultdict(list)
for pos in out_window:
    by_kind[pos.get("bucket_kind", "?")].append(pos)
print_decomp("BUCKET KIND", by_kind)

# 4. BY TARGET_DATE OFFSET (same-day / tomorrow / future)
by_offset = defaultdict(list)
for pos in out_window:
    off = target_date_offset(pos)
    if off is None:
        by_offset["unknown"].append(pos)
    elif off == 0:
        by_offset["same-day"].append(pos)
    elif off == 1:
        by_offset["tomorrow"].append(pos)
    elif off >= 2:
        by_offset["2+ days"].append(pos)
    else:
        by_offset[f"past ({off}d)"].append(pos)
print_decomp("TARGET_DATE OFFSET", by_offset)

# 5. BY HOURS-TO-PEAK BIN
by_h2p = defaultdict(list)
for pos in out_window:
    h = hours_to_peak_at_submit(pos)
    if h is None:
        by_h2p["no-peak-data"].append(pos)
    elif h < -3:
        by_h2p["past peak >3h"].append(pos)
    elif h < 0:
        by_h2p["past peak 0-3h"].append(pos)
    elif h < 2:
        by_h2p["pre-peak 0-2h"].append(pos)
    elif h < 6:
        by_h2p["pre-peak 2-6h"].append(pos)
    elif h < 12:
        by_h2p["pre-peak 6-12h"].append(pos)
    else:
        by_h2p["pre-peak 12+h"].append(pos)
print_decomp("HOURS-TO-PEAK AT SUBMIT", by_h2p)

# 6. BY ENTRY PRICE BUCKET
by_entry = defaultdict(list)
for pos in out_window:
    e = pos.get("entry_price", 0)
    if e < 0.70:
        by_entry["< $0.70"].append(pos)
    elif e < 0.78:
        by_entry["$0.70-0.78"].append(pos)
    elif e == 0.78:
        by_entry["$0.78 (target)"].append(pos)
    elif e < 0.85:
        by_entry["$0.78-0.85"].append(pos)
    else:
        by_entry["≥ $0.85"].append(pos)
print_decomp("ENTRY PRICE", by_entry)

# 7. BY DAY-OF-WEEK
by_dow = defaultdict(list)
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for pos in out_window:
    try:
        tdate = datetime.fromisoformat(pos["target_date"]).date()
        by_dow[DOW_NAMES[tdate.weekday()]].append(pos)
    except Exception:
        pass
print_decomp("DAY OF WEEK (target_date)", by_dow)

# 8. INTERSECTION: tomorrow-only (most likely to be the "safe" cohort)
tomorrow_only = [pos for pos in out_window if target_date_offset(pos) == 1]
print()
print("=" * 76)
print("FOCUSED COHORTS")
print("=" * 76)
s = summary("tomorrow_only", tomorrow_only)
if s:
    print(f"tomorrow_only: n={s['n']}, win={s['win_rate']*100:.1f}%, "
          f"total=${s['total']:+.2f}, avg=${s['avg']:+.3f}, "
          f"passes={s['passes_breakeven']}")

# tomorrow_only × bucket_kind
print()
print("tomorrow_only × bucket_kind:")
for kind in ("mid", "high_tail", "low_tail"):
    cohort = [p for p in tomorrow_only if p.get("bucket_kind") == kind]
    s = summary("", cohort)
    if s:
        marker = " ★" if s["passes_breakeven"] else ""
        print(f"  {kind:12} n={s['n']:>3} win={s['win_rate']*100:>5.1f}% "
              f"total=${s['total']:>+7.2f}{marker}")

# Same-day × hours_to_peak deep-dive (the actually-played slice)
same_day = [pos for pos in out_window if target_date_offset(pos) == 0]
print()
print("same-day positions:")
s = summary("", same_day)
if s:
    print(f"  n={s['n']}, win={s['win_rate']*100:.1f}%, "
          f"total=${s['total']:+.2f}, avg=${s['avg']:+.3f}")

# Final verdict
print()
print("=" * 76)
print("VERDICT")
print("=" * 76)
print(f"Whole out_window cohort: n={len(out_window)}, "
      f"win_rate={sum(1 for x in out_window if x.get('realized_pnl',0)>0)/max(1,len(out_window))*100:.1f}%")
print(f"Breakeven at $0.78 entry: 78%")
print(f"Deficit: {78 - sum(1 for x in out_window if x.get('realized_pnl',0)>0)/max(1,len(out_window))*100:.1f}pp")
print()
print("Check the ★ markers above. If multiple dimensions have a profitable cohort,")
print("intersect them. If no cohort hits 78%, the strategy is structurally unprofitable.")
