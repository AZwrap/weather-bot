"""One-shot analysis: Layer 5 break point + Layer 6 re-validation + poll_resolutions audit.

Run on VPS: python3 .claude/scratch/daily_check_analysis.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, ".")

# ───────────────────────────────────────────────────────────────────────
# Layer 5: where did the paper log stop, and which call sites use it?
# ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LAYER 5 — paper log break investigation")
print("=" * 70)
try:
    with open("data/obs_distance_paper_log.jsonl") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"records: {len(rows)}")
    by_day = Counter(r.get("scan_time_utc", "")[:10] for r in rows)
    for d in sorted(by_day.keys()):
        print(f"  {d}: {by_day[d]}")
    if rows:
        print(f"\nfirst entry: {rows[0].get('scan_time_utc','-')[:19]}")
        print(f"last entry:  {rows[-1].get('scan_time_utc','-')[:19]}")
        print(f"\nlast entry full record:")
        print(json.dumps(rows[-1], default=str, indent=2)[:600])
except Exception as e:
    print(f"err: {e}")

# ───────────────────────────────────────────────────────────────────────
# Layer 6: re-validate against current resolved positions
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("LAYER 6 — adverse-info filter re-validation")
print("=" * 70)

try:
    from zoneinfo import ZoneInfo
    from weather_bot.intraday import trigger_local_hour
    from weather_bot.locations import STATIONS_BY_ID

    p = json.load(open("data/portfolio.json"))
    positions = p["positions"]
    nm_resolved = [
        pos for pos in positions
        if pos.get("strategy") == "NO_momentum"
        and pos.get("status") == "resolved"
        and pos.get("realized_pnl") is not None
        and pos.get("submitted_at")
    ]
    print(f"resolved NO_momentum positions: {len(nm_resolved)}")

    # Threshold-change cutoff: 0.78 NO threshold shipped 2026-05-14
    cutoff = "2026-05-14"
    nm_recent = [p for p in nm_resolved if p.get("submitted_at", "") >= cutoff]
    print(f"  with submitted_at >= {cutoff} (post-0.78-threshold): {len(nm_recent)}")
    print()

    # Need market target (max vs min). Heuristic: max for max-target events.
    # Get from market_id → look up event in raw data isn't readily available.
    # Fall back: assume all are max-target (NO_momentum is max-heavy).
    in_window_pnl, out_window_pnl, skipped = [], [], 0
    for pos in nm_recent:
        sid = pos.get("station_id", "")
        station = STATIONS_BY_ID.get(sid)
        if not station:
            skipped += 1
            continue
        try:
            sub_dt = datetime.fromisoformat(pos["submitted_at"].replace("Z", "+00:00"))
            tdate = datetime.fromisoformat(pos["target_date"]).date()
            local_now = sub_dt.astimezone(ZoneInfo(station.timezone))
            if local_now.date() != tdate:
                # Forward-placed → not in adverse window by definition
                out_window_pnl.append(pos["realized_pnl"])
                continue
            peak_hour = trigger_local_hour(sid, tdate)
            if peak_hour is None:
                skipped += 1
                continue
            now_h = local_now.hour + local_now.minute / 60.0
            hours_to_peak = peak_hour - now_h
            in_window = -1.0 <= hours_to_peak <= 2.0
            if in_window:
                in_window_pnl.append(pos["realized_pnl"])
            else:
                out_window_pnl.append(pos["realized_pnl"])
        except Exception:
            skipped += 1

    def summary(label, pnls):
        if not pnls:
            print(f"  {label}: n=0")
            return
        n = len(pnls)
        wins = sum(1 for x in pnls if x > 0)
        total = sum(pnls)
        print(f"  {label}: n={n} wins={wins} ({100*wins/n:.1f}%) "
              f"total=${total:+.2f} avg=${total/n:+.3f}")

    summary("in_window  (would block) ", in_window_pnl)
    summary("out_window (passes filter)", out_window_pnl)
    print(f"  unclassified: {skipped}")
except Exception as e:
    print(f"err: {e}")
    import traceback
    traceback.print_exc()

# ───────────────────────────────────────────────────────────────────────
# Poll_resolutions: why did 2 RKSI May-23 positions get missed?
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("POLL_RESOLUTIONS — missed-positions filter audit")
print("=" * 70)
try:
    # Find which positions are currently filled-but-not-resolved with old target_date
    p = json.load(open("data/portfolio.json"))
    today = datetime.now(timezone.utc).date()
    stuck = []
    for pos in p["positions"]:
        if pos.get("status") != "filled":
            continue
        td = pos.get("target_date")
        if not td:
            continue
        try:
            tdate = datetime.fromisoformat(td).date()
            if (today - tdate).days >= 1:
                stuck.append(pos)
        except Exception:
            pass
    print(f"currently stuck (filled, target_date older than today): {len(stuck)}")
    for pos in stuck[:10]:
        print(f"  {pos.get('station_id','?'):5} {pos.get('bucket_label','?'):16} "
              f"target_date={pos.get('target_date','?')} "
              f"strategy={pos.get('strategy','?')} "
              f"filled_at={pos.get('filled_at')} resolved_at={pos.get('resolved_at')}")

    # Inspect poll_resolutions source code for the filter
    print("\n--- poll_resolutions source: filter that selects positions to check ---")
    import subprocess
    try:
        with open("weather_bot/poll_resolutions.py") as f:
            src = f.read()
        # Find filter logic
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if any(k in line for k in ("status ==", "status==", "filled_at", "is_closed",
                                        "for pos in", "iter_filled", "iter_open")):
                if "#" not in line[:line.find(line.lstrip()[:1])] if line.strip() else True:
                    print(f"  {i+1:4}: {line}")
    except FileNotFoundError:
        print("  poll_resolutions.py not found")
except Exception as e:
    print(f"err: {e}")
    import traceback
    traceback.print_exc()
