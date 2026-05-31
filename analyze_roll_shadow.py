"""Roll-shadow — would a 7-tick-stop "roll to the new leader" have beaten
the static fire-once basket?

Motivation (UUWW/Moscow, 2026-05-31): the basket fired at 09:05 on 16°C
@0.85; the leader flipped to 17°C at 09:31; 17°C was the actual max →
static basket = full loss. A dynamic strategy would have stopped out of
16°C and rolled into 17°C. This measures, per fired basket, the static
P&L vs a "roll to the eventual leader" P&L, net of fees + a modeled
7-tick-stop exit + roll churn — OVERALL and PER-STATION.

Data: it reuses the threshold-SWEEP log, which already snapshots a full
basket (winner + fade legs, depth-walked, fee-applied) at EVERY leader
the market presented during the day. So for one event we already have:
  - the basket at the FIRST-≥trigger leader  (what we actually fire), and
  - the basket at the FINAL/highest leader    (what a perfect roll lands on).
We compare their resolved P&Ls and net the roll cost.

IMPORTANT — this is an OPTIMISTIC bound on the roll's benefit: it assumes
the roll perfectly tracks to the final observed leader (no extra whipsaw
beyond the modeled per-roll cost). If even this upper bound says "doesn't
help," rolling isn't worth building. If it says "helps a lot" (UUWW
should), it's worth a real paper test.

7-tick buffer (operator, 2026-05-31): the stop on the held leg fires at
entry − 0.07; only flips where the held bucket fell ≥7 ticks count as a
roll (the buffer that stops buy/sell/buy/sell whipsaw).

Run:  python analyze_roll_shadow.py
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
SWEEP_LOG = DATA / "basket_sweep_log.jsonl"
FORWARD_LOG = DATA / "forward_log.jsonl"

TRIGGER = 85          # the live basket trigger (penny)
STOP_BUFFER = 0.07    # 7-tick stop on the held leg
ROLL_COST_PER_LEG = 0.03  # modeled taker spread+fee to unwind/re-enter a leg


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


def _rounded_obs(actual_c: float, unit: str) -> int:
    if unit == "F":
        return int(math.floor(actual_c * 9.0 / 5.0 + 32.0 + 0.1))
    return int(math.floor(actual_c + 0.5))  # round-half-up: oracle uses rounded °C


def _bucket_won(kind, threshold, actual_int, unit) -> bool:
    if kind == "low_tail":
        return actual_int <= threshold
    if kind == "high_tail":
        return actual_int >= threshold
    if unit == "C":
        return actual_int == threshold
    return threshold <= actual_int <= threshold + 1


def _basket_pnl(row: dict, ai: int, unit: str) -> float | None:
    """Resolved P&L of one sweep-snapshot basket (winner YES + fade NO),
    net of the fee already baked into each leg's net_cost."""
    w = row.get("winner")
    if not w or w.get("depth_source") == "top_of_book_fallback":
        return None
    legs = [("YES", w)] + [("NO", l) for l in row.get("fade_no", [])]
    pnl = 0.0
    for side, leg in legs:
        if leg.get("depth_source") == "top_of_book_fallback":
            continue
        thr = leg.get("bucket_threshold")
        sh = float(leg.get("shares", 0))
        net_cost = float(leg.get("net_cost", 0))
        if thr is None or sh <= 0:
            continue
        won = _bucket_won(leg.get("bucket_kind"), int(thr), ai, unit)
        payout = sh if ((side == "YES") == won) else 0.0
        pnl += payout - net_cost
    return pnl


def main() -> int:
    rows = load_jsonl(SWEEP_LOG)
    res = load_jsonl(FORWARD_LOG)
    units = {}
    try:
        sys.path.insert(0, ".")
        from weather_bot.locations import STATIONS_BY_ID
        units = {sid: s.unit for sid, s in STATIONS_BY_ID.items()}
    except Exception:
        pass
    res_by_key = {}
    for r in res:
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            res_by_key[k] = float(r["actual_obs_c"])

    # Group sweep rows by event (sk).
    by_event: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        k = (row.get("station_id"), row.get("target"), row.get("target_date"))
        by_event[k].append(row)

    per_station: dict[str, list[tuple]] = defaultdict(list)  # (static, roll, flipped)
    n_events = n_scored = n_flipped = 0

    for k, evrows in by_event.items():
        sid, target, td = k
        actual_c = res_by_key.get(k)
        if actual_c is None:
            continue
        unit = units.get(sid, "C")
        ai = _rounded_obs(actual_c, unit)
        evrows = sorted(evrows, key=lambda r: r.get("entry_threshold", 0))

        # FIRE basket = the first snapshot at/above the live trigger.
        fire = next((r for r in evrows if r.get("entry_threshold", 0) >= TRIGGER), None)
        if fire is None:
            continue
        # FINAL leader basket = the highest-threshold snapshot (best info).
        final = evrows[-1]
        static_pnl = _basket_pnl(fire, ai, unit)
        if static_pnl is None:
            continue
        n_events += 1; n_scored += 1

        fire_leader = (fire.get("winner") or {}).get("bucket_label")
        final_leader = (final.get("winner") or {}).get("bucket_label")
        flipped = fire_leader != final_leader

        if not flipped:
            roll_pnl = static_pnl  # never rolled
        else:
            n_flipped += 1
            final_pnl = _basket_pnl(final, ai, unit)
            if final_pnl is None:
                roll_pnl = static_pnl
            else:
                # Cost of stopping out the fire-basket legs + entering the
                # final-leader legs (per-leg churn).
                n_legs = 1 + len(fire.get("fade_no", [])) + 1 + len(final.get("fade_no", []))
                roll_pnl = final_pnl - ROLL_COST_PER_LEG * n_legs
        per_station[sid].append((static_pnl, roll_pnl, flipped))

    def line(label, rowset):
        if not rowset:
            print(f"  {label:18} n=0"); return
        n = len(rowset)
        s = sum(x[0] for x in rowset); r = sum(x[1] for x in rowset)
        fl = sum(1 for x in rowset if x[2])
        helped = sum(1 for x in rowset if x[1] > x[0] + 1e-9)
        print(f"  {label:18} n={n:>3}  static=${s:+8.2f}  roll=${r:+8.2f}  "
              f"Δ=${r-s:+8.2f}  flips={fl}  roll_helped={helped}")

    allrows = [x for v in per_station.values() for x in v]
    print("=" * 74)
    print("ROLL-SHADOW  —  static fire-once vs 7-tick-stop roll-to-leader")
    print(f"(trigger={TRIGGER}¢, stop=−{STOP_BUFFER:.2f}, roll_cost=${ROLL_COST_PER_LEG}/leg)")
    print("=" * 74)
    print(f"scored events: {n_scored}   flipped (leader changed after fire): {n_flipped}")
    print("\nOVERALL")
    line("ALL", allrows)
    print("\nPER-STATION")
    for sid in sorted(per_station):
        line(sid, per_station[sid])
    print("\nΔ = roll − static. Positive ⇒ rolling would have helped (UPPER BOUND:")
    print("assumes perfect tracking to the final leader, net of modeled churn).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
