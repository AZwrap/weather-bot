"""3-bucket arb test (operator idea 2026-06-03, REVISED to be arb-gated).

The hedge: when the favorite hits the trigger, ALSO buy YES on the two
threshold-neighbors (fav-1, fav+1) + NO on the rest. Operator constraint
(2026-06-03): "it has to be an arb otherwise we lose money anyway." So we
ONLY count it when the three adjacent YES asks sum < $1 — then you pay
<$1 for a 3-bucket basket that wins ~99% of the time (the off-by-one is
covered: 82% favorite-hit + 18% adjacent = 99% within ±1, measured from
the real sweep fires). If Σ3 ≥ $1 the hedge bleeds → skip.

This reads the `yes3_arb` block now logged per snapshot in
basket_sweep_log.jsonl (favorite + both neighbor YES *ask ladders*), so
the arb is sized DEPTH-AWARE: matched (equal) shares across the three
legs, bounded by the THINNEST of the three books — the fix for the fake
$5000 top-of-book fill that wrecked the first attempt.

Sized matched-share K = min(top-ask size across the 3 legs), floored at
the 5-share exchange minimum. Cost/set = Σ(3 best asks). At resolution:
  winner ∈ {3 adjacent}:  net = K·(1 − Σ) − fees     (≈99% of the time)
  winner ∉ {3}:           net = −K·Σ  − fees          (the ~1% far tail)

Run (on VPS venv):  python analyze_basket_3bucket.py [trigger]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import slim_dashboard as sd
from weather_bot.fees import taker_fee_usd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

DATA = Path("data")
TRIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.82
EXCL = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM"}
MIN_SHARES = 5.0       # exchange minimum; below this the arb isn't fillable
MAX_SET_COST = 0.999   # Σ3 must be strictly < $1 to be an arb at all


def unit(s):
    st = STATIONS_BY_ID.get(s)
    return (getattr(st, "unit", None) or "C")


def matched_arb(legs):
    """legs = [leader_arb, *neighbor_arbs], each with best_ask + ask_ladder.
    Returns (K, set_cost, fee_per_set) for the depth-aware matched-share arb,
    or None if not 3 complete legs / not fillable / Σ≥$1.

    K = min top-ask size across the 3 (the thinnest book bounds the matched
    buy); set_cost = Σ best asks; fee/set = Σ taker_fee(1 share @ best ask)."""
    if len(legs) != 3 or any(l is None for l in legs):
        return None
    tops = []          # (price, size) at top of each ask ladder
    for l in legs:
        lad = l.get("ask_ladder") or []
        if not lad:
            ba = l.get("best_ask")
            if ba is None:
                return None
            tops.append((float(ba), 0.0))   # price known, size unknown → 0 → unfillable
        else:
            tops.append((float(lad[0][0]), float(lad[0][1])))
    set_cost = sum(p for p, _ in tops)
    if set_cost >= MAX_SET_COST:
        return None
    K = min(s for _, s in tops)
    if K < MIN_SHARES:
        return None
    fee_per_set = sum(taker_fee_usd(1.0, p) for p, _ in tops)
    return K, set_cost, fee_per_set


def main():
    resmap = sd.resolution_map()
    sweep = defaultdict(list)
    path = DATA / "basket_sweep_log.jsonl"
    if not path.exists():
        print("no basket_sweep_log.jsonl"); return
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sweep[(r.get("station_id"), r.get("target"), r.get("target_date"))].append(r)

    rows_with_field = 0
    # fire-level read: first snapshot at/above TRIG per event
    fired = arb_gate = scored = win_in3 = 0
    tot_net = 0.0
    by_penny_seen = defaultdict(int)
    by_penny_arb = defaultdict(int)
    detail = []

    for sk, rows in sweep.items():
        sid, tgt, date = sk
        if sid in EXCL:
            continue
        rows = sorted([r for r in rows if r.get("yes3_arb")], key=lambda r: r.get("ts_utc") or "")
        # gate-frequency by penny across ALL snapshots that carry the field
        for r in rows:
            rows_with_field += 1
            y3 = r["yes3_arb"]
            pen = r.get("entry_threshold")
            if y3.get("n_yes_legs") == 3 and y3.get("best_ask_sum") is not None:
                by_penny_seen[pen] += 1
                if y3.get("arb_ok"):
                    by_penny_arb[pen] += 1

        # fire snapshot = first at/above TRIG
        fire = next((r for r in rows
                     if float(r.get("leader_yes_ask") or 0) >= TRIG), None)
        if fire is None:
            continue
        y3 = fire["yes3_arb"]
        fired += 1
        legs = [y3.get("leader")] + list(y3.get("neighbors") or [])
        m = matched_arb(legs)
        if m is None:
            continue
        arb_gate += 1
        K, set_cost, fee_set = m

        ac = resmap.get(sk)
        if ac is None:
            continue          # gate passed but not resolved yet → can't score
        u = unit(sid)
        ai = _rounded_observation(ac, u)
        in3 = any(bucket_won(l["bucket_kind"], l["bucket_threshold"], ai, u)
                  for l in legs if l is not None)
        scored += 1
        if in3:
            win_in3 += 1
            net = K * (1.0 - set_cost) - K * fee_set
        else:
            net = -K * set_cost - K * fee_set
        tot_net += net
        favlab = legs[0]["bucket_label"] if legs[0] else "?"
        detail.append((net, sk, favlab, round(set_cost, 3), round(K, 1), in3))

    print("=" * 78)
    print("3-BUCKET ARB TEST (arb-gated: Σ3 YES ask < $1) — depth-aware matched shares")
    print("  source: basket_sweep_log.jsonl `yes3_arb` block | trigger %.2f | excl %s"
          % (TRIG, ",".join(sorted(EXCL))))
    print("=" * 78)
    if rows_with_field == 0:
        print("\nNO `yes3_arb` rows yet — the neighbor-YES depth logging was just")
        print("deployed; this fills in as the sweep runs. Re-run at N=7/14 days.")
        print("(Old sweep rows predate the field, so they're skipped.)")
        return

    print("snapshots carrying yes3_arb: %d" % rows_with_field)
    print("\nARB-GATE FREQUENCY by entry penny (how often Σ3 YES ask < $1):")
    for pen in sorted(by_penny_seen):
        seen = by_penny_seen[pen]; ok = by_penny_arb[pen]
        bar = "#" * int(round(20 * ok / seen)) if seen else ""
        print("   %3d¢  %3d/%-3d  %4.0f%%  %s" % (pen, ok, seen, 100 * ok / seen, bar))

    print("\nAt trigger %.2f:" % TRIG)
    print("  events fired (favorite ≥ trig):        %d" % fired)
    print("  ... where the 3-leg arb gate PASSED:   %d  (Σ3<$1 AND ≥5 sh fillable)" % arb_gate)
    print("  ... and resolved (scoreable):          %d" % scored)
    if scored:
        print("  winner landed in the 3 buckets:        %d / %d  (%.0f%%)"
              % (win_in3, scored, 100 * win_in3 / scored))
        print("\n  NET P&L of the arb-gated 3-bucket buy:  $%+.2f over %d events  ($%+.3f/event)"
              % (tot_net, scored, tot_net / scored))
        print("\n  biggest / worst single events:")
        for net, sk, fav, sc, K, in3 in sorted(detail, key=lambda x: -x[0])[:4]:
            print("    + %-5s %-3s fav='%s' Σ3=%.3f K=%.0fsh in3=%s  $%+.2f"
                  % (sk[0], sk[1], fav, sc, K, in3, net))
        for net, sk, fav, sc, K, in3 in sorted(detail, key=lambda x: x[0])[:4]:
            print("    - %-5s %-3s fav='%s' Σ3=%.3f K=%.0fsh in3=%s  $%+.2f"
                  % (sk[0], sk[1], fav, sc, K, in3, net))
    else:
        print("  (no gate-passing events resolved yet — accumulating)")


if __name__ == "__main__":
    main()
