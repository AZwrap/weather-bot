"""Publication-window shadow harness — cron entry point.

Designed to run every 30 minutes via cron. Each invocation:
  1. Honors KILL_SWITCH.
  2. Fetches Polymarket events for today + yesterday.
  3. For each event matched to a station, captures a publication-window
     snapshot if (and only if) end-of-resolution-day in station-local
     time has already passed AND we haven't already logged a snapshot
     for the same 30-min bucket on that (station, target, date).
  4. Writes records to data/publication_window_log.jsonl.

Idempotent. No live trading. No portfolio mutation.

After ~5-7 days of runtime, run `python analyze_publication_window.py`
to read the log and decide whether a Wunderground-race strategy is
worth building.

Usage:
  python publication_window_log.py             # snapshot once
  python publication_window_log.py --verbose   # print each snapshot

Cron line (every 30 min):
  */30 * * * * cd <repo> && python publication_window_log.py >> data/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import httpx

from weather_bot.exclusions import load_active_exclusions
from weather_bot.polymarket import (
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    match_event_to_station,
)
from weather_bot.publication_window import (
    DEFAULT_LOG_PATH,
    snapshot_one,
)

KILL_SWITCH = Path("KILL_SWITCH")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per snapshot written.")
    p.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH,
                   help="Override the JSONL log path.")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    if KILL_SWITCH.exists():
        print(f"[abort] KILL_SWITCH present at {KILL_SWITCH.resolve()}")
        return 0

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    yesterday_utc = today_utc - timedelta(days=1)
    excluded = {sid for sid, _t in load_active_exclusions(today_utc)}

    n_attempts = 0
    n_written = 0

    async with httpx.AsyncClient(timeout=60.0) as http:
        events = await fetch_all_temperature_events(http)

        all_tokens: list[str] = []
        for ev in events:
            for m in ev.markets:
                if m.yes_token_id:
                    all_tokens.append(m.yes_token_id)
        clob = await fetch_clob_prices_batch(all_tokens, http)
        for ev in events:
            apply_clob_prices(ev, clob)

        print(
            f"[pubwin] {now_utc.isoformat()} fetched {len(events)} events; "
            f"snapshotting (station, target_date) pairs past midend-local"
        )

        for ev in events:
            station = match_event_to_station(ev)
            if station is None:
                continue
            if station.station_id in excluded:
                continue
            target = "max" if ev.target == "highest" else "min"
            target_date = event_target_date(ev, station)

            # Only snapshot today/yesterday/day-before. Older events are
            # likely already settled and gone from the events feed; if
            # they linger we still log them up to 48h post-midend.
            if target_date < yesterday_utc - timedelta(days=1):
                continue
            if target_date > today_utc:
                continue

            n_attempts += 1
            rec = await snapshot_one(
                station=station,
                target=target,
                target_date=target_date,
                ev=ev,
                http=http,
                log_path=args.log_path,
                now_utc=now_utc,
            )
            if rec is not None:
                n_written += 1
                if args.verbose:
                    matched = (
                        f"{rec.matched_bucket_kind}/{rec.matched_bucket_threshold}"
                        if rec.matched_bucket_kind else "no_match"
                    )
                    print(
                        f"  [{station.station_id}/{target}] "
                        f"{target_date.isoformat()}  "
                        f"+{rec.offset_h_after_midend:.1f}h  "
                        f"extreme={rec.metar_final_extreme_c}  "
                        f"matched={matched}"
                    )

    print(f"[pubwin] done — attempted {n_attempts}, wrote {n_written}")
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
