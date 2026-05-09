r"""Report Polymarket weather events whose city/station we don't yet trade.

Run any time to see candidate cities the bot has noticed but won't act on.
Decide which (if any) are worth evaluating with `evaluate_station.py`.

Usage:
    python check_new_stations.py                  # default: last 7 days
    python check_new_stations.py --since-days 30
    python check_new_stations.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.unmatched import load_unmatched, summarise_by_city


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since-days", type=float, default=7.0,
                   help="Look at unmatched events seen in the past N days (default 7)")
    p.add_argument("--min-observations", type=int, default=1,
                   help="Hide cities seen fewer than N times (default 1)")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()

    records = load_unmatched()
    if not records:
        print("No unmatched events recorded yet. Either:")
        print("  • the cron hasn't logged enough yet, or")
        print("  • Polymarket isn't listing any cities we don't already trade.")
        return

    since_utc = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    summaries = summarise_by_city(records, since_utc=since_utc)
    summaries = [s for s in summaries if s.n_observations >= args.min_observations]

    if not summaries:
        print(f"No new cities in the past {args.since_days:.1f} days.")
        return

    if args.json:
        out = []
        for s in summaries:
            d = asdict(s)
            d["first_seen_utc"] = s.first_seen_utc.isoformat()
            d["last_seen_utc"] = s.last_seen_utc.isoformat()
            d["targets"] = sorted(s.targets)
            d["days_seen"] = round(s.days_seen, 2)
            out.append(d)
        print(json.dumps(out, indent=2))
        return

    print(f"Unmatched cities seen in the past {args.since_days:.1f} days:\n")
    print(f"{'city':<22s}  {'obs':>4s}  {'days':>5s}  {'targets':<10s}  "
          f"{'url ICAO':<8s}  {'sample event slug'}")
    print("-" * 110)
    for s in summaries:
        targets_str = "+".join(sorted(s.targets))
        icao_hint = s.sample_url_icao or "?"
        print(
            f"{s.city:<22s}  {s.n_observations:>4d}  "
            f"{s.days_seen:>5.1f}  {targets_str:<10s}  "
            f"{icao_hint:<8s}  {s.sample_event_slug}"
        )
    print()
    print(f"Total: {len(summaries)} unique unmatched cit"
          f"{'y' if len(summaries) == 1 else 'ies'}.")
    print()
    print("Next step: pick a city worth investigating and run")
    print("    python evaluate_station.py --name 'CityName' --icao XXXX "
          "--lat LAT --lon LON --tz IANA/Tz --unit C")
    print("…to run a 180-day METAR-based skill backtest on it. If it lands")
    print("in tier ★★★ or ★★ with bias correction, add it to MARKETS.")


if __name__ == "__main__":
    main()
