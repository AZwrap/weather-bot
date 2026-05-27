"""Overlap check: V2 conditional preposit fills vs Layer 7 live fires.

For each V2 hypothetical fill (paper):
  - Identify (station_id, target_date, bucket_label)
  - Check if Layer 7 fired live on the same tuple
  - Report:
      timing: V2 fill_ts vs Layer 7 actual fire ts
      overlap: same tuple, both fired
      incremental: V2 fired, Layer 7 didn't

Result interpretation:
  - V2 ⊆ Layer 7  → V2 is redundant; no new edge
  - V2 ⊃ Layer 7  → V2 catches more; investigate the delta
  - V2 ≠ Layer 7  → partial overlap; quantify both directions
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

PAPER_LOG = Path("data/paper_no_momentum_log.jsonl")
FORWARD_LOG = Path("data/forward_log.jsonl")
SUBMITTED = Path("data/submitted_orders.jsonl")


def passes_v2_gate(intent: dict, gate: float = 0.80) -> bool:
    other = intent.get("other_max_yes_ask")
    return other is not None and other >= gate


# 1. Load resolutions
res: dict[tuple[str, str, str], float] = {}
with open(FORWARD_LOG) as f:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("actual_obs_c") is None:
            continue
        sid = r.get("station_id")
        target = r.get("target")
        td = r.get("target_date")
        if sid and target and td:
            key = (sid, target, td)
            if key not in res:
                res[key] = float(r["actual_obs_c"])


# 2. Load V2 candidates from paper log (just first record per token-day)
v2_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
with open(PAPER_LOG) as f:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        token = r.get("token_id")
        td = r.get("target_date")
        if token and td:
            v2_groups[(token, td)].append(r)

# Sort each group chronologically
for k in v2_groups:
    v2_groups[k].sort(key=lambda r: r.get("ts_utc", ""))


# 3. Identify V2 fills (passed gate AND maker order would have filled)
v2_fills: list[dict] = []
for (token, td), snaps in v2_groups.items():
    first = snaps[0]
    if not passes_v2_gate(first):
        continue

    sid = first.get("station_id")
    target = first.get("target")
    bk = first.get("bucket_kind", "mid")
    bt = first.get("bucket_threshold")
    if bt is None:
        continue
    unit = first.get("station_unit", "C")
    bucket_label = first.get("bucket_label", "")

    # Resolution
    res_obs_c = res.get((sid, target, td))
    if res_obs_c is None:
        continue
    actual_int = _rounded_observation(res_obs_c, unit)
    bucket_did_win = bucket_won(bk, int(bt), actual_int, unit)
    # Our NO bet wins iff bucket loses
    no_won = not bucket_did_win

    # Maker fill: yes_bid >= 1 - 0.82 = 0.18 in any snapshot
    fill_ts = None
    for s in snaps:
        yb = s.get("yes_bid")
        if yb is not None and yb >= 0.18 - 1e-9:
            fill_ts = s.get("ts_utc")
            break
    if fill_ts is None:
        continue

    v2_fills.append({
        "token_id": token,
        "station_id": sid,
        "target": target,
        "target_date": td,
        "bucket_label": bucket_label,
        "bucket_threshold": int(bt),
        "bucket_kind": bk,
        "v2_first_ts": first.get("ts_utc"),
        "v2_fill_ts": fill_ts,
        "no_won": no_won,
    })


# 4. Load Layer 7 fires from submitted_orders.jsonl
# Layer 7 is identifiable: order_type=FAK, side=NO, fill_price typically >= 0.90
# (NO ask high = bucket past trigger). The strategy field isn't logged, so
# infer from price + order_type.
l7_fires: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
with open(SUBMITTED) as f:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("order_type") != "FAK":
            continue
        if r.get("side") != "NO":
            continue
        # Treat FAK NO at fill_price >= 0.85 as Layer 7
        fp = r.get("fill_price")
        if fp is None or fp < 0.85:
            continue
        sid = r.get("station_id")
        td = r.get("target_date")
        bl = r.get("bucket_label")
        # Layer 7 doesn't include target; infer max for now (default trigger)
        if sid and td and bl:
            # Match by (station, target_date, bucket_label) — target inferred
            key = (sid, td, bl)
            l7_fires[key].append(r)


# 5. Cross-reference
n_v2 = len(v2_fills)
n_overlap = 0
n_incremental = 0
v2_won_total = 0
overlap_records: list[dict] = []
incremental_records: list[dict] = []

for v in v2_fills:
    if v["no_won"]:
        v2_won_total += 1
    key = (v["station_id"], v["target_date"], v["bucket_label"])
    l7_matches = l7_fires.get(key, [])
    if l7_matches:
        n_overlap += 1
        # Get earliest L7 fire ts on this bucket
        l7_first_ts = min(o.get("ts_utc", "") for o in l7_matches)
        v.update({
            "l7_fired": True,
            "l7_first_ts": l7_first_ts,
            "v2_earlier_than_l7": v["v2_fill_ts"] < l7_first_ts,
        })
        overlap_records.append(v)
    else:
        n_incremental += 1
        v.update({"l7_fired": False, "l7_first_ts": None})
        incremental_records.append(v)


# 6. Report
print(f"V2 hypothetical fills: {n_v2}")
print(f"V2 win count: {v2_won_total}")
print()
print(f"Overlap with Layer 7 (same station/date/bucket): {n_overlap}")
print(f"Incremental to Layer 7 (V2 only):                {n_incremental}")
print()

if overlap_records:
    print("=== OVERLAP records (V2 + Layer 7 same bucket) ===")
    print(f"{'station':<6} {'target_date':<11} {'bucket':<18} "
          f"{'V2_fill':<25} {'L7_fire':<25} {'V2_earlier':<11} {'won':<4}")
    for r in overlap_records:
        print(
            f"{r['station_id']:<6} {r['target_date']:<11} "
            f"{r['bucket_label'][:18]:<18} "
            f"{(r['v2_fill_ts'] or '')[:19]:<25} "
            f"{(r['l7_first_ts'] or '')[:19]:<25} "
            f"{str(r.get('v2_earlier_than_l7','-')):<11} "
            f"{'WIN' if r['no_won'] else 'LOSS':<4}"
        )

if incremental_records:
    print()
    print("=== INCREMENTAL records (V2 fired, Layer 7 didn't) ===")
    print(f"{'station':<6} {'target_date':<11} {'bucket':<18} "
          f"{'V2_fill':<25} {'won':<4}")
    for r in incremental_records:
        print(
            f"{r['station_id']:<6} {r['target_date']:<11} "
            f"{r['bucket_label'][:18]:<18} "
            f"{(r['v2_fill_ts'] or '')[:19]:<25} "
            f"{'WIN' if r['no_won'] else 'LOSS':<4}"
        )

print()
print("=== Interpretation guide ===")
if n_overlap == n_v2:
    print(f"  V2 ⊆ Layer 7 — V2 is REDUNDANT. No incremental edge.")
elif n_overlap == 0:
    print(f"  V2 ∩ Layer 7 = ∅ — V2 is fully INCREMENTAL.")
else:
    pct_overlap = 100 * n_overlap / n_v2
    print(f"  Partial overlap: {pct_overlap:.0f}% of V2 also fired Layer 7.")
    if n_incremental > 0:
        inc_wins = sum(1 for r in incremental_records if r["no_won"])
        inc_rate = 100 * inc_wins / n_incremental if n_incremental else 0
        # Hypothetical PnL for incremental cohort at $0.82 entry
        # WIN payout = 6.1 * (1 - 0.82) = $1.10; LOSS = -6.1 * 0.82 = -$5.00
        inc_pnl = inc_wins * 1.10 - (n_incremental - inc_wins) * 5.00
        print(f"    Incremental cohort: n={n_incremental}, wins={inc_wins} ({inc_rate:.1f}%), "
              f"hypothetical PnL=${inc_pnl:+.2f}")

# Also report v2_earlier counts
n_v2_earlier = sum(1 for r in overlap_records if r.get("v2_earlier_than_l7"))
n_l7_earlier = sum(1 for r in overlap_records if r.get("v2_earlier_than_l7") is False)
if n_overlap > 0:
    print(f"  Timing: V2 fired earlier on {n_v2_earlier}/{n_overlap} overlapping buckets.")
    print(f"          Layer 7 fired earlier on {n_l7_earlier}/{n_overlap}.")
