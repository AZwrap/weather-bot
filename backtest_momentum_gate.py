"""Backtest: conditional preposit / momentum-gated entry strategies.

For each bucket timeline, track peak yes_ask reached. Test multiple
entry policies:

  Single-stage entries (place limit at threshold X, fill if peak ≥ X):
    - $0.85, $0.88, $0.90, $0.92, $0.95

  Two-stage entry (user idea):
    - Stage 1: $5 at limit $0.92 (filled if peak ≥ $0.92)
    - Stage 2: $5 at limit $0.95 (filled if peak ≥ $0.95)

For each strategy, compute hit rate at fill and total realized P&L.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

SIZE_PER_STAGE = 5.0  # dollars

recs = load_records()
print(f"records: {len(recs)}")

# Build bucket timelines with outcomes
buckets: dict[tuple, dict] = {}
for r in recs:
    if r.bucket_snapshots is None:
        continue
    if r.actual_obs_c is None:
        continue  # only test on resolved buckets
    for snap in r.bucket_snapshots:
        if snap.yes_ask is None or snap.yes_bid is None:
            continue
        key = (r.station_id, r.target, r.target_date.isoformat(),
               snap.kind, snap.threshold)
        if key not in buckets:
            buckets[key] = {"snaps": [], "label": snap.bucket_label,
                            "station_id": r.station_id, "kind": snap.kind,
                            "threshold": snap.threshold, "obs_c": r.actual_obs_c,
                            "target": r.target}
        buckets[key]["snaps"].append((r.issue_time_utc, float(snap.yes_ask),
                                       float(snap.yes_bid)))

# Determine outcomes per bucket
for key, b in buckets.items():
    station = STATIONS_BY_ID.get(b["station_id"])
    if station is None:
        b["won_yes"] = None
        continue
    actual_int = _rounded_observation(b["obs_c"], station.unit)
    # Need a snap for bucket_won — use any
    if not b["snaps"]:
        continue
    fake_snap = type("S", (), {"kind": b["kind"], "threshold": b["threshold"]})
    b["won_yes"] = bucket_won(fake_snap, actual_int, station.unit)
    b["peak_yes_ask"] = max(s[1] for s in b["snaps"])
    b["min_yes_bid"] = min(s[2] for s in b["snaps"])
    b["peak_no_ask"] = max(1.0 - s[2] for s in b["snaps"])  # NO ask = 1 - yes_bid
    b["min_yes_ask"] = min(s[1] for s in b["snaps"])

resolved = {k: b for k, b in buckets.items() if b.get("won_yes") is not None}
print(f"resolved bucket timelines: {len(resolved)}")
print()


def report(label: str, fills: list[tuple[float, bool, float]]) -> None:
    """fills = [(entry_price, won, size_usd), ...]"""
    if not fills:
        print(f"{label:50s} no fills")
        return
    n = len(fills)
    n_won = sum(1 for _, w, _ in fills if w)
    hit_rate = n_won / n
    total_size = sum(s for _, _, s in fills)
    pnl = sum((s/e - s) if w else (-s) for e, w, s in fills)
    avg_entry = sum(e for e, _, _ in fills) / n
    print(f"{label:50s} n={n:4d}  hit={hit_rate*100:5.1f}%  "
          f"avg_entry=${avg_entry:.3f}  size=${total_size:6.0f}  "
          f"PnL=${pnl:+8.2f}  ROI={pnl/total_size*100:+6.1f}%")


# --- Single-stage YES side: limit at threshold X, $5 each ---
print("=== SINGLE-STAGE YES preposit (place buy YES at limit $X, $5 size) ===")
for thr in [0.85, 0.88, 0.90, 0.92, 0.95]:
    fills = []
    for key, b in resolved.items():
        if b["peak_yes_ask"] >= thr:
            fills.append((thr, b["won_yes"], SIZE_PER_STAGE))
    report(f"YES limit at ${thr:.2f}", fills)
print()

# --- Single-stage NO side: place buy NO at threshold X (= NO ask at X) ---
print("=== SINGLE-STAGE NO preposit (place buy NO at limit $X, $5 size) ===")
for thr in [0.85, 0.88, 0.90, 0.92, 0.95]:
    fills = []
    for key, b in resolved.items():
        if b["peak_no_ask"] >= thr:
            fills.append((thr, not b["won_yes"], SIZE_PER_STAGE))
    report(f"NO limit at ${thr:.2f}", fills)
print()

# --- Two-stage YES: $5 at $0.92 + $5 at $0.95 ---
print("=== TWO-STAGE YES preposit (size scales with confirmation) ===")
fills = []
for key, b in resolved.items():
    if b["peak_yes_ask"] >= 0.92:
        fills.append((0.92, b["won_yes"], SIZE_PER_STAGE))
    if b["peak_yes_ask"] >= 0.95:
        fills.append((0.95, b["won_yes"], SIZE_PER_STAGE))
report("Two-stage YES (0.92 + 0.95)", fills)

# --- Three-stage YES: $5 at $0.88 + $5 at $0.92 + $5 at $0.95 ---
fills = []
for key, b in resolved.items():
    if b["peak_yes_ask"] >= 0.88:
        fills.append((0.88, b["won_yes"], SIZE_PER_STAGE))
    if b["peak_yes_ask"] >= 0.92:
        fills.append((0.92, b["won_yes"], SIZE_PER_STAGE))
    if b["peak_yes_ask"] >= 0.95:
        fills.append((0.95, b["won_yes"], SIZE_PER_STAGE))
report("Three-stage YES (0.88 + 0.92 + 0.95)", fills)
print()

# --- Hit rate by peak-ask bin (calibration check) ---
print("=== Calibration: hit rate by bucket peak yes_ask reached ===")
print("(Tells us: for buckets that reached price X, did they actually hit?)")
print(f"{'peak bin':18s} {'n':>5s} {'hit_rate':>9s} {'mid':>6s}")
bins = [(0.0, 0.5, "0-50%"), (0.5, 0.7, "50-70%"), (0.7, 0.85, "70-85%"),
        (0.85, 0.92, "85-92%"), (0.92, 0.97, "92-97%"), (0.97, 1.0, "97-99.9%")]
for lo, hi, label in bins:
    items = [b for b in resolved.values() if lo <= b["peak_yes_ask"] < hi]
    if not items:
        continue
    n = len(items)
    hits = sum(1 for b in items if b["won_yes"])
    print(f"{label:18s} {n:>5d}   {hits/n*100:>6.1f}%   {(lo+hi)/2:.3f}")

# --- For NO side ---
print()
print("=== Calibration: hit rate (NO winning) by peak NO ask reached ===")
print(f"{'peak bin':18s} {'n':>5s} {'hit_rate':>9s} {'mid':>6s}")
for lo, hi, label in bins:
    items = [b for b in resolved.values() if lo <= b["peak_no_ask"] < hi]
    if not items:
        continue
    n = len(items)
    no_wins = sum(1 for b in items if not b["won_yes"])
    print(f"{label:18s} {n:>5d}   {no_wins/n*100:>6.1f}%   {(lo+hi)/2:.3f}")
