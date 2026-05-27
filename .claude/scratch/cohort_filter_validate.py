"""Re-validate A∩C cohort filter against accumulated resolved positions.

Run when N has grown materially (≥30 more fills since 2026-05-25).
Compares:
  - Resolved NO_momentum positions that PASSED the cohort filter
    (their win rate should hold at ~78%)
  - Resolved positions logged as BLOCKED (these were placed BEFORE
    the filter shipped 2026-05-25 — useful for retrospective bias check)

If pass-rate drops below 75%, fall back to B-only (2-6h pre-peak)
or disable NO_momentum entirely.
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from weather_bot.cohort_filter import LOSER_STATIONS

# ────────────────────────────────────────────────────────────────────────
# Load portfolio resolved positions + cohort filter log
# ────────────────────────────────────────────────────────────────────────
p = json.load(open("data/portfolio.json"))
nm = [
    pos for pos in p["positions"]
    if pos.get("strategy") == "NO_momentum"
    and pos.get("status") == "resolved"
    and pos.get("realized_pnl") is not None
    and pos.get("submitted_at")
]

# Pre/post filter ship date
SHIP_DATE = "2026-05-25"
pre_ship = [x for x in nm if x.get("submitted_at", "") < SHIP_DATE]
post_ship = [x for x in nm if x.get("submitted_at", "") >= SHIP_DATE]
print(f"resolved NO_momentum: {len(nm)} total ({len(pre_ship)} pre-ship, {len(post_ship)} post-ship)")


def passes_cohort(pos) -> bool:
    """Replay the filter against an already-resolved position."""
    sid = pos.get("station_id", "")
    if sid in LOSER_STATIONS:
        return False
    sys.path.insert(0, ".")
    from weather_bot.locations import STATIONS_BY_ID
    from zoneinfo import ZoneInfo
    station = STATIONS_BY_ID.get(sid)
    if not station: return False
    try:
        sub = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        td = datetime.fromisoformat(pos["target_date"]).date()
        local_today = sub.astimezone(ZoneInfo(station.timezone)).date()
        return td == local_today  # same-day
    except Exception:
        return False


def stats(label: str, rows: list):
    if not rows:
        print(f"  {label:45} n=0"); return
    n = len(rows)
    wins = sum(1 for x in rows if x.get("realized_pnl", 0) > 0)
    total = sum(x.get("realized_pnl", 0) for x in rows)
    marker = " ★" if (n >= 10 and wins / n >= 0.78) else ""
    print(f"  {label:45} n={n:>3}  win={100*wins/n:>5.1f}%  "
          f"total=${total:>+7.2f}  avg=${total/n:>+6.3f}{marker}")


# Retrospective: what the filter WOULD have done on pre-ship data
print("\n=== Retrospective check on pre-ship data ===")
pre_pass = [x for x in pre_ship if passes_cohort(x)]
pre_block = [x for x in pre_ship if not passes_cohort(x)]
stats("Would-pass (filter says good)", pre_pass)
stats("Would-block (filter says bad)", pre_block)

# Live check: post-ship pass cohort
print("\n=== Live check on post-ship data ===")
post_pass = [x for x in post_ship if passes_cohort(x)]
stats("Pass-cohort actual fills", post_pass)
if len(post_pass) >= 30:
    win_rate = sum(1 for x in post_pass if x.get("realized_pnl", 0) > 0) / len(post_pass)
    if win_rate >= 0.78:
        print(f"\n  ✓ Filter holds: {win_rate*100:.1f}% win rate on n={len(post_pass)}")
    elif win_rate >= 0.75:
        print(f"\n  ⚠ Filter degraded but acceptable: {win_rate*100:.1f}% on n={len(post_pass)}")
    else:
        print(f"\n  ✗ Filter broken: {win_rate*100:.1f}% on n={len(post_pass)} — fall back to B-only")
else:
    print(f"\n  (need ≥30 post-ship fills to validate; have {len(post_pass)})")

# Cohort filter log analysis (live data on blocked decisions)
print("\n=== Cohort filter log breakdown ===")
log_path = Path("data/cohort_filter_log.jsonl")
if log_path.exists():
    with open(log_path) as f:
        decisions = [json.loads(l) for l in f if l.strip()]
    print(f"  total decisions logged: {len(decisions)}")
    by_reason = Counter(d.get("reason", "?") for d in decisions)
    for k, v in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"    {k:25}: {v}")
    blocked = sum(d.get("n_buckets", 1) for d in decisions if d.get("block"))
    passed = sum(d.get("n_buckets", 1) for d in decisions if not d.get("block"))
    print(f"  bucket-weighted: blocked={blocked}, passed={passed}, "
          f"block_rate={blocked/(blocked+passed)*100:.1f}%")
else:
    print("  (log file not yet created — filter runs at next NO_momentum cycle)")
