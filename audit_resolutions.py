"""Resolution audit: compare Polymarket's resolution vs our ASOS reading.

⚠ ⚠ ⚠  CAVEAT (after 2026-05-14 bug discovery):
The forward log records have an off-by-one timezone bug for stations
with high UTC offset (e.g., NZWN/Wellington UTC+12). `target_date`
was computed from `end_date.astimezone(tz).date()` which lands ONE
DAY AFTER the actual measurement day for events ending at midnight
station-local. Result: NZWN's `actual_obs_c` (for our target_date)
and `polymarket_won_bucket` (from the event_slug ONE DAY EARLIER)
were comparing different weather days entirely. Fix shipped in
weather_bot/polymarket.py event_target_date 2026-05-14.

This audit reads the STORED data — for records logged BEFORE the
fix, the day pairing may be wrong. Re-run resolve_log AFTER the
fix lands to repopulate, OR cross-check audit findings against
event_slug to verify the day match.

Surfaces cases where Polymarket resolved a different bucket than our
station observation would have. Useful for:

  - Detecting oracle-source disagreements (the DNMM Lagos pattern)
  - Identifying stations to consider excluding via data/excluded_stations.json
  - Deciding whether to manually file a UMA dispute (rarely economic — see
    the dispute-economics summary at top of output)

Does NOT auto-dispute. UMA disputes require posting ~$750 USD-equivalent
bond. At our $5/trade sizing the bond is 150× a single position. Even if
multiple trades on one event resolve wrong, the bond is rarely covered
unless multi-bucket exposure is large.

Run:
    python audit_resolutions.py                    # all resolved records
    python audit_resolutions.py --since 2026-05-08 # within date range
    python audit_resolutions.py --station DNMM     # one station
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import date

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

UMA_DISPUTE_BOND_USD = 750.0  # rough — varies by market; check current at dispute time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", help="ISO date inclusive, e.g. 2026-05-08")
    p.add_argument("--until", help="ISO date inclusive")
    p.add_argument("--station", help="filter to one station_id")
    args = p.parse_args()

    since_d = date.fromisoformat(args.since) if args.since else None
    until_d = date.fromisoformat(args.until) if args.until else None

    recs = load_records()
    print(f"loaded {len(recs)} forward-log records")

    # Filter to records with BOTH our ASOS reading AND Polymarket's resolution
    matched: list = []
    for r in recs:
        if r.target != "max":
            continue
        if r.actual_obs_c is None or r.polymarket_won_bucket is None:
            continue
        if since_d and r.target_date < since_d:
            continue
        if until_d and r.target_date > until_d:
            continue
        if args.station and r.station_id != args.station:
            continue
        matched.append(r)

    print(f"resolved records with both readings: {len(matched)}")
    if not matched:
        return

    # Dedup by (station, target_date) — multiple forward-log records exist
    # per market (one per scan); the resolution is the same.
    seen_keys: set = set()
    unique: list = []
    for r in matched:
        key = (r.station_id, r.target_date.isoformat())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(r)

    print(f"unique (station, target_date) groups: {len(unique)}")
    print()

    # ── Audit each ────────────────────────────────────────────────────────
    mismatches: list = []
    by_station = defaultdict(lambda: {"total": 0, "mismatch": 0})

    for r in unique:
        station = STATIONS_BY_ID.get(r.station_id)
        if station is None:
            continue

        # SLUG-DAY CROSS-CHECK: if the event_slug encodes a different
        # day than target_date, the comparison would be cross-day and
        # MEANINGLESS. Skip those records with a warning. This catches
        # the pre-fix Wellington off-by-one issue.
        if r.event_slug:
            import re
            # Slug format: "highest-temperature-in-<city>-on-may-<DD>-YYYY"
            m = re.search(r"-on-([a-z]+)-(\d+)-(\d{4})$", r.event_slug)
            if m:
                month_name, day, year = m.groups()
                MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                          "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
                month = MONTHS.get(month_name.lower())
                if month:
                    from datetime import date as _date
                    slug_date = _date(int(year), month, int(day))
                    if slug_date != r.target_date:
                        # Day mismatch — skip with warning
                        print(
                            f"  ⚠ SKIPPING {r.station_id} {r.target_date}: "
                            f"slug={r.event_slug} encodes {slug_date}, "
                            f"target_date is {r.target_date}. "
                            f"Off-by-one timezone bug — re-run resolve_log "
                            f"after the event_target_date fix lands."
                        )
                        continue

        # What does OUR ASOS reading imply for the winning bucket?
        our_int = _rounded_observation(r.actual_obs_c, station.unit)

        # Polymarket's winning bucket
        poly_label = r.polymarket_won_bucket
        poly_thr = r.polymarket_won_threshold

        # Find OUR predicted winning bucket from the bucket_snapshots list
        our_winner_label = None
        our_winner_threshold = None
        if r.bucket_snapshots:
            for s in r.bucket_snapshots:
                if bucket_won(s.kind, s.threshold, our_int, station.unit):
                    our_winner_label = s.bucket_label
                    our_winner_threshold = s.threshold
                    break

        by_station[r.station_id]["total"] += 1

        # Compare
        if poly_thr is None or our_winner_threshold is None:
            continue  # can't compare
        if poly_thr != our_winner_threshold:
            mismatches.append({
                "station_id": r.station_id,
                "station_name": station.name,
                "target_date": r.target_date.isoformat(),
                "actual_obs_c": r.actual_obs_c,
                "our_rounded": our_int,
                "unit": station.unit,
                "our_winner_label": our_winner_label,
                "our_winner_threshold": our_winner_threshold,
                "poly_label": poly_label,
                "poly_threshold": poly_thr,
                "gap": abs(poly_thr - our_winner_threshold),
            })
            by_station[r.station_id]["mismatch"] += 1

    # ── Report ────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"UMA dispute bond reference: ${UMA_DISPUTE_BOND_USD:.0f}+ USD-equivalent")
    print("Disputing is rarely economic at our $5/trade sizing.")
    print("This audit is for: (1) detecting bad stations, (2) escalating only")
    print("clear-cut multi-bucket-exposure cases where bond is justified.")
    print("=" * 80)
    print()

    if not mismatches:
        print(f"✓ NO MISMATCHES found across {len(unique)} resolved markets.")
        print("  All Polymarket resolutions agree with our ASOS readings.")
        return

    print(f"✗ {len(mismatches)} mismatches found ({len(mismatches)}/{len(unique)} = "
          f"{100*len(mismatches)/max(1,len(unique)):.1f}%):\n")

    # Sort by gap descending (biggest disagreements first)
    mismatches.sort(key=lambda m: -m["gap"])

    print(f"{'STATION':6s} {'CITY':<15s} {'DATE':10s} "
          f"{'OBS':>6s} {'OUR_PICK':<18s} {'POLY_PICK':<18s} {'GAP':>4s}")
    print("-" * 95)
    for m in mismatches:
        print(
            f"{m['station_id']:6s} "
            f"{m['station_name']:<15s} "
            f"{m['target_date']:10s} "
            f"{m['actual_obs_c']:>5.1f}°C "
            f"{m['our_winner_label']:<18s} "
            f"{m['poly_label']:<18s} "
            f"{m['gap']:>4d}"
        )

    # ── Per-station summary ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Per-station mismatch rate (only stations with ≥1 mismatch):")
    print("=" * 80)
    rows = [
        (sid, info["mismatch"], info["total"])
        for sid, info in by_station.items() if info["mismatch"] > 0
    ]
    rows.sort(key=lambda r: -r[1])  # by mismatch count desc
    for sid, mm, total in rows:
        rate = 100 * mm / total
        station_name = STATIONS_BY_ID[sid].name if sid in STATIONS_BY_ID else "?"
        print(f"  {sid} ({station_name}): {mm}/{total} = {rate:.1f}% mismatch")

    # ── Action guidance ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Action guidance:")
    print("=" * 80)
    for sid, mm, total in rows:
        rate = mm / total
        if rate > 0.20 and mm >= 2:
            print(
                f"  HIGH RISK — {sid}: {rate*100:.0f}% mismatch rate over "
                f"{total} resolved days. Consider adding to "
                f"data/excluded_stations.json (see project_oracle_source_risk.md "
                f"for the DNMM precedent)."
            )
        elif mm >= 2:
            print(
                f"  WATCH — {sid}: {mm} mismatches over {total} days. "
                f"Monitor; consider exclusion at next review."
            )
        else:
            print(
                f"  ONE-OFF — {sid}: {mm}/{total}. Likely single-incident; "
                f"keep trading but record this for pattern tracking."
            )

    # ── Dispute economics ─────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Dispute decision tree:")
    print("=" * 80)
    print("  1. Is the gap LARGE (≥2°C / ≥4°F)? If small (1°C off), likely")
    print("     rounding/sensor noise — don't dispute.")
    print("  2. Do public sources (Wunderground, NWS, METAR archive) all")
    print("     agree with OUR reading, not Polymarket's? Need 2-of-3 to win.")
    print("  3. Did we have ≥$750 of position exposure on the disputed event?")
    print("     If less, the bond costs more than the recovery. Don't dispute.")
    print("  4. Have we won a UMA dispute before? Cold-start dispute is risky;")
    print("     test process on a clear $750+ case first.")
    print()
    print("If ALL four are YES, file dispute via UMA Voter App on uma.xyz")
    print("(MANUAL — bot does NOT auto-file). Document outcome here for the")
    print("next decision.")


if __name__ == "__main__":
    main()
