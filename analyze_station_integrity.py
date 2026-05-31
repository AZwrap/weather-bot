"""Station-integrity monitor — flag dodgy / possibly-manipulated stations.

Operator note (2026-05-31): Polymarket comments allege the Moscow (UUWW)
station feed is manipulated. Russia-hosted oracle source → plausible. We
want to surface stations whose books behave erratically so they can be
EXCLUDED MANUALLY (added to data/excluded_stations.json) — this script
does NOT auto-exclude; it flags candidates for review.

Signals per station (higher = more suspect):
  - flip_rate     : leader changes per event (data/leader_flips.jsonl).
                    A clean market converges monotonically (≈0-1 flips);
                    lots of flips = churn / spoofing / thin manipulation.
  - reversal_rate : flips that go A→B→A (the leader bounces back) — a
                    spoofing / wash signature, not honest convergence.
  - miss_rate     : fraction of fired baskets whose entry bucket ≠ the
                    resolved winner (the early/most-traded leader was wrong)
                    — combines genuine volatility AND manipulation.
  - late_flip     : leader changed AFTER crossing 0.90 (winner looked
                    locked, then moved) — strong dodgy/oracle-risk signal.

Inputs: data/leader_flips.jsonl, data/basket_sweep_log.jsonl,
        data/consensus_basket_log.jsonl, data/forward_log.jsonl

Run:  python analyze_station_integrity.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

DATA = Path("data")
FLIP_LOG = DATA / "leader_flips.jsonl"
SWEEP_LOG = DATA / "basket_sweep_log.jsonl"
BASKET_LOG = DATA / "consensus_basket_log.jsonl"
FORWARD_LOG = DATA / "forward_log.jsonl"

# Flag thresholds (review candidates; tune as data accumulates).
# The flip log is PRE-FILTERED by the daemon to flips where the OUTGOING
# leader peaked ≥0.70 — the full 0.70–0.99 contention band (matches the
# threshold-sweep) so the calibration dataset is complete. Sub-0.70 flicker
# (market honestly undecided) is not logged. So flips/event = dethronings of
# a 0.70+ leader per event; some convergence flips through the 0.70s are
# NORMAL, so flip-rate alone is moderate signal. The HIGH-confidence dodgy
# signal is LATE-FLIP (outgoing leader peaked ≥0.90 = locked-then-moved, the
# UUWW/Moscow oracle-risk pattern), alongside a sustained MISS-rate.
FLAG_FLIP_RATE = 2.5     # >2.5 dethronings of a 0.70+ leader / event (churny)
FLAG_MISS_RATE = 0.40    # >40% of fires on a bucket that didn't win
FLAG_LATE_FLIP = 1       # any flip where a ≥0.90 leader got dethroned
LATE_PEAK = 0.90


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _rounded_obs(actual_c, unit):
    if unit == "F":
        return int(math.floor(actual_c * 9.0 / 5.0 + 32.0 + 0.1))
    return int(math.floor(actual_c + 0.05))


def main() -> int:
    flips = load_jsonl(FLIP_LOG)
    basket = [r for r in load_jsonl(BASKET_LOG) if r.get("result") == "filled"]
    res = load_jsonl(FORWARD_LOG)
    units = {}
    try:
        sys.path.insert(0, ".")
        from weather_bot.locations import STATIONS_BY_ID
        units = {sid: s.unit for sid, s in STATIONS_BY_ID.items()}
    except Exception:
        pass

    # --- flip metrics per station ---
    flips_by_event: dict[tuple, list[dict]] = defaultdict(list)
    for f in flips:
        flips_by_event[(f.get("station_id"), f.get("target"), f.get("target_date"))].append(f)

    st_flip_total = defaultdict(int)
    st_events = defaultdict(int)
    st_reversals = defaultdict(int)
    st_late = defaultdict(int)
    for k, fs in flips_by_event.items():
        sid = k[0]
        st_events[sid] += 1
        st_flip_total[sid] += len(fs)
        seq = [f.get("to_bucket") for f in fs]
        # reversal: a bucket reappears as a destination (A→B→A…)
        seen = set()
        for b in seq:
            if b in seen:
                st_reversals[sid] += 1
            seen.add(b)
        for f in fs:
            # LATE = the OUTGOING leader had peaked ≥0.90 (looked locked) and
            # then lost the lead. This is the locked-then-moved signature, not
            # a normal flip INTO a high price.
            if float(f.get("from_peak_ask", 0) or 0) >= LATE_PEAK:
                st_late[sid] += 1

    # --- miss rate: fired-bucket vs resolved winner ---
    res_by_key = {}
    for r in res:
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            res_by_key[k] = float(r["actual_obs_c"])
    # the YES (winner) leg of each fired basket = the bucket we backed
    st_fires = defaultdict(int)
    st_miss = defaultdict(int)
    for leg in basket:
        if leg.get("side") != "YES":
            continue
        sid = leg.get("station_id")
        k = (sid, leg.get("target"), leg.get("target_date"))
        actual_c = res_by_key.get(k)
        if actual_c is None:
            continue
        unit = units.get(sid, "C")
        ai = _rounded_obs(actual_c, unit)
        thr = leg.get("bucket_threshold")
        kind = leg.get("bucket_kind")
        if thr is None:
            continue
        st_fires[sid] += 1
        won = (ai <= thr if kind == "low_tail" else ai >= thr if kind == "high_tail"
               else (ai == thr if unit == "C" else thr <= ai <= thr + 1))
        if not won:
            st_miss[sid] += 1

    stations = sorted(set(st_events) | set(st_fires))
    print("=" * 78)
    print("STATION INTEGRITY  —  flip / reversal / miss signals (flag for MANUAL review)")
    print("=" * 78)
    print(f"  {'station':8} {'events':>6} {'flips/ev':>8} {'reversals':>9} "
          f"{'late':>5} {'fires':>5} {'miss%':>6}  flags")
    flagged = []
    for sid in stations:
        ev = st_events.get(sid, 0)
        fpe = (st_flip_total.get(sid, 0) / ev) if ev else 0.0
        rev = st_reversals.get(sid, 0)
        late = st_late.get(sid, 0)
        fires = st_fires.get(sid, 0)
        miss = st_miss.get(sid, 0)
        miss_rate = (miss / fires) if fires else 0.0
        flags = []
        if fpe > FLAG_FLIP_RATE:
            flags.append("FLIPPY")
        if fires >= 3 and miss_rate > FLAG_MISS_RATE:
            flags.append("MISSES")
        if late >= FLAG_LATE_FLIP:
            flags.append("LATE-FLIP")
        if flags:
            flagged.append(sid)
        print(f"  {sid:8} {ev:>6} {fpe:>8.1f} {rev:>9} {late:>5} {fires:>5} "
              f"{100*miss_rate:>5.0f}%  {' '.join(flags)}")

    print()
    if flagged:
        print("⚠️  REVIEW for exclusion (add to data/excluded_stations.json if confirmed):")
        print("    " + ", ".join(flagged))
    else:
        print("No stations flagged yet (need more resolved events / flip data).")
    print("\nNOTE: thresholds are provisional; counts are low until data accumulates.")
    print("This flags candidates only — exclusion is a manual decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
