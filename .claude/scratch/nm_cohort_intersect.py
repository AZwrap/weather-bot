"""Cohort intersection: A, B, C and their overlaps."""
import json, sys
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, ".")
from weather_bot.intraday import trigger_local_hour
from weather_bot.locations import STATIONS_BY_ID

p = json.load(open("data/portfolio.json"))
nm = [
    pos for pos in p["positions"]
    if pos.get("strategy") == "NO_momentum"
    and pos.get("status") == "resolved"
    and pos.get("realized_pnl") is not None
    and pos.get("submitted_at", "") >= "2026-05-14"
]


def h2p(pos):
    sid = pos.get("station_id", "")
    s = STATIONS_BY_ID.get(sid)
    if not s: return None
    try:
        sub = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        td = datetime.fromisoformat(pos["target_date"]).date()
        local = sub.astimezone(ZoneInfo(s.timezone))
        if local.date() != td: return None
        peak = trigger_local_hour(sid, td)
        if peak is None: return None
        return peak - (local.hour + local.minute / 60.0)
    except Exception:
        return None


def is_same_day(pos):
    sid = pos.get("station_id", "")
    s = STATIONS_BY_ID.get(sid)
    if not s: return None
    try:
        sub = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
        td = datetime.fromisoformat(pos["target_date"]).date()
        return sub.astimezone(ZoneInfo(s.timezone)).date() == td
    except Exception:
        return None


def in_adverse(pos):
    h = h2p(pos)
    return h is not None and -1.0 <= h <= 2.0


# Tag every position
ASIA_EAST = {"RJTT", "RCSS", "ZUCK", "SBGR", "RKSI", "ZSPD", "ZSQD", "ZHHH", "ZGGG"}
# (SBGR is LatAm not Asia_East, but I'll keep the loser-set definition broad)
LOSERS_BY_STATION = {"RJTT", "RCSS", "ZUCK", "SBGR", "ZHHH", "OEJN", "LFPB", "KSFO",
                     "LLBG", "VILK", "MPMG", "KAUS", "ZUUU", "LTFM"}

# Filter to out_window (apples-to-apples with the 52.4% number)
out = [pos for pos in nm if not in_adverse(pos)]
print(f"out_window: n={len(out)}")


def stats(rows, label):
    if not rows:
        print(f"  {label:30}: n=0")
        return
    n = len(rows)
    wins = sum(1 for x in rows if x.get("realized_pnl", 0) > 0)
    total = sum(x.get("realized_pnl", 0) for x in rows)
    marker = "★" if (n >= 10 and wins / n >= 0.78) else " "
    print(f"  {label:42}: n={n:>3}  win={100*wins/n:>5.1f}%  total=${total:>+7.2f}  avg=${total/n:>+6.3f}  {marker}")


print("\n=== BASELINE ===")
stats(out, "out_window (all)")

print("\n=== A (disable tomorrow): same-day OR 2+ days ===")
stats([x for x in out if is_same_day(x) is True], "A: same-day only")
stats([x for x in out if is_same_day(x) is False], "(excluded by A: forward placements)")

print("\n=== B (pre-peak 2-6h): subset of same-day ===")
stats([x for x in out if (h := h2p(x)) is not None and 2 <= h < 6], "B: same-day & 2-6h pre-peak")

print("\n=== C (drop loser stations) ===")
stats([x for x in out if x.get("station_id") not in LOSERS_BY_STATION], "C: non-loser stations")
stats([x for x in out if x.get("station_id") in LOSERS_BY_STATION], "(excluded by C)")

print("\n=== INTERSECTIONS ===")
A = [x for x in out if is_same_day(x) is True]
B = [x for x in out if (h := h2p(x)) is not None and 2 <= h < 6]
C = [x for x in out if x.get("station_id") not in LOSERS_BY_STATION]

stats([x for x in A if x in C], "A ∩ C: same-day, non-loser")
stats([x for x in B if x in C], "B ∩ C: same-day 2-6h, non-loser")
stats([x for x in A if x not in B and x in C], "A ∩ C \\ B: same-day non-(2-6h) non-loser")

print("\n=== B's stations (which stations contribute to the 92.3% cohort?) ===")
b_stations = Counter(x.get("station_id", "?") for x in B)
for sid, n in sorted(b_stations.items(), key=lambda x: -x[1]):
    wins = sum(1 for x in B if x.get("station_id") == sid and x.get("realized_pnl", 0) > 0)
    print(f"  {sid}: n={n}, wins={wins}")

print("\n=== A \\ B: same-day, NOT in 2-6h window ===")
ab_diff = [x for x in A if x not in B]
stats(ab_diff, "A \\ B (same-day, outside 2-6h window)")

# Per-cohort hours_to_peak distribution
print("\n=== distribution of h2p in A vs B ===")
from collections import Counter
buckets_a = Counter()
buckets_b = Counter()
for pos in A:
    h = h2p(pos)
    bucket = f"{int(h):>+3d}h" if h is not None else "none"
    buckets_a[bucket] += 1
for pos in B:
    h = h2p(pos)
    bucket = f"{int(h):>+3d}h" if h is not None else "none"
    buckets_b[bucket] += 1
print(f"  A breakdown by h2p bin: {dict(sorted(buckets_a.items()))}")
print(f"  B breakdown:            {dict(sorted(buckets_b.items()))}")
