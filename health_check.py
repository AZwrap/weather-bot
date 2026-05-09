r"""Health check for the deployed bot. Verifies cron is firing, data is fresh,
APIs are reachable, and disk has headroom.

Exit codes:
    0  — all checks pass
    1  — at least one WARN (degraded but bot still works)
    2  — at least one FAIL (something is broken)

Usage (PowerShell or VPS shell):
    python health_check.py
    python health_check.py --quiet            # print only WARN / FAIL rows
    python health_check.py --json             # machine-readable output
    python health_check.py --no-net           # skip network reachability checks

Cron-friendly check every hour (won't spam logs unless something is wrong):
    0 * * * *  /path/to/Weather_Bot/.venv/bin/python /path/to/Weather_Bot/health_check.py --quiet || \
               echo "weather-bot health WARN/FAIL at $(date -u)" >> data/health.out
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx

from weather_bot.forward_log import DEFAULT_LOG_PATH, load_records
from weather_bot.unmatched import (
    DEFAULT_UNMATCHED_PATH,
    load_unmatched,
    summarise_by_city,
)

Status = Literal["OK", "WARN", "FAIL"]
EXPECTED_RECORDS_PER_RUN = 57  # one per (station, target) in MARKETS (HKO excluded)


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str

    @property
    def severity(self) -> int:
        return {"OK": 0, "WARN": 1, "FAIL": 2}[self.status]


# ──────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────


def check_bias_table(path: Path = Path("bias_table.json"), max_age_days: int = 14) -> CheckResult:
    if not path.exists():
        return CheckResult("bias_table", "FAIL", f"{path} missing — run train_bias.py")
    age_days = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 86400
    if age_days > max_age_days:
        return CheckResult(
            "bias_table",
            "WARN",
            f"{age_days:.1f} days old (weekly retrain expected)",
        )
    return CheckResult("bias_table", "OK", f"{age_days:.1f} days old")


def check_forward_log_freshness(max_age_hours: float = 26.0) -> CheckResult:
    """Latest log record should be < 26 hours old (daily cron + slack)."""
    if not DEFAULT_LOG_PATH.exists():
        return CheckResult("forward_log_freshness", "FAIL", f"{DEFAULT_LOG_PATH} missing")
    records = load_records(DEFAULT_LOG_PATH)
    if not records:
        return CheckResult("forward_log_freshness", "FAIL", "log file is empty")
    latest = max(r.issue_time_utc for r in records)
    age_h = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
    if age_h > max_age_hours:
        return CheckResult(
            "forward_log_freshness",
            "FAIL",
            f"latest record is {age_h:.1f}h old — cron likely missed (latest={latest.isoformat()})",
        )
    return CheckResult(
        "forward_log_freshness", "OK", f"latest {age_h:.1f}h ago, total {len(records)} records"
    )


def check_recent_completeness(
    expected: int = EXPECTED_RECORDS_PER_RUN, threshold_pct: float = 0.9
) -> CheckResult:
    """Latest run should produce ~all expected records."""
    records = load_records(DEFAULT_LOG_PATH)
    if not records:
        return CheckResult("recent_completeness", "FAIL", "no records")

    latest = max(r.issue_time_utc for r in records)
    # Records sharing the same issue_time are from the same cron run.
    latest_run = [r for r in records if r.issue_time_utc == latest]
    n = len(latest_run)
    if n < expected * threshold_pct:
        return CheckResult(
            "recent_completeness",
            "WARN",
            f"latest run logged {n}/{expected} records — some forecasts failed",
        )
    return CheckResult("recent_completeness", "OK", f"{n}/{expected} records on latest run")


def check_bucket_snapshots() -> CheckResult:
    """Latest run should have bucket prices for most records."""
    records = load_records(DEFAULT_LOG_PATH)
    if not records:
        return CheckResult("bucket_snapshots", "FAIL", "no records")
    latest = max(r.issue_time_utc for r in records)
    latest_run = [r for r in records if r.issue_time_utc == latest]
    n = len(latest_run)
    with_snaps = sum(1 for r in latest_run if r.bucket_snapshots)
    if with_snaps == 0:
        return CheckResult(
            "bucket_snapshots",
            "FAIL",
            "no bucket prices in latest run — polymarket fetch failing?",
        )
    if with_snaps < n * 0.6:
        return CheckResult(
            "bucket_snapshots", "WARN", f"only {with_snaps}/{n} records have bucket data"
        )
    return CheckResult("bucket_snapshots", "OK", f"{with_snaps}/{n} records have bucket data")


def check_resolution_lag(max_unresolved_age_days: int = 14) -> CheckResult:
    """Records older than ~6 days should be resolvable; if many old records
    remain unresolved, the resolve_log cron is probably broken."""
    records = load_records(DEFAULT_LOG_PATH)
    if not records:
        return CheckResult("resolution_lag", "OK", "no records yet")
    today = datetime.now(timezone.utc).date()
    old_unresolved = [
        r for r in records
        if not r.is_resolved and (today - r.target_date).days >= max_unresolved_age_days
    ]
    if old_unresolved:
        return CheckResult(
            "resolution_lag",
            "WARN",
            f"{len(old_unresolved)} records >{max_unresolved_age_days} days old still unresolved — run resolve_log.py",
        )
    return CheckResult("resolution_lag", "OK", "no stale unresolved records")


def check_disk_space(min_mb: int = 100) -> CheckResult:
    free_mb = shutil.disk_usage(".").free / 1_000_000
    if free_mb < min_mb:
        return CheckResult("disk_space", "FAIL", f"only {free_mb:.0f} MB free in cwd")
    return CheckResult("disk_space", "OK", f"{free_mb:.0f} MB free")


def check_unmatched_cities(min_age_hours: float = 24.0) -> CheckResult:
    """WARN if any unmatched city has been observed for ≥24h.

    Triggers a soft alert so the user knows to run check_new_stations.py.
    Does NOT auto-add stations — that's deliberately a manual decision.
    """
    if not DEFAULT_UNMATCHED_PATH.exists():
        return CheckResult("unmatched_cities", "OK", "no unmatched events logged")
    records = load_unmatched(DEFAULT_UNMATCHED_PATH)
    if not records:
        return CheckResult("unmatched_cities", "OK", "no unmatched events logged")
    summaries = summarise_by_city(records)
    persistent = [s for s in summaries if s.days_seen * 24 >= min_age_hours]
    if not persistent:
        return CheckResult(
            "unmatched_cities", "OK",
            f"{len(summaries)} cit{'y' if len(summaries)==1 else 'ies'} seen briefly, none ≥{min_age_hours:.0f}h",
        )
    cities = sorted(s.city for s in persistent)
    preview = ", ".join(cities[:5]) + (" …" if len(cities) > 5 else "")
    return CheckResult(
        "unmatched_cities", "WARN",
        f"{len(persistent)} new cit{'y' if len(persistent)==1 else 'ies'} seen for ≥{min_age_hours:.0f}h: {preview} — run `python check_new_stations.py`",
    )


async def check_open_meteo(client: httpx.AsyncClient) -> CheckResult:
    try:
        r = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 51.5,
                "longitude": 0.0,
                "hourly": "temperature_2m",
                "forecast_days": 1,
            },
        )
    except Exception as exc:
        return CheckResult("open_meteo", "FAIL", f"unreachable: {exc}")
    if r.status_code == 200:
        return CheckResult("open_meteo", "OK", "reachable")
    return CheckResult("open_meteo", "WARN", f"HTTP {r.status_code}")


async def check_polymarket(client: httpx.AsyncClient) -> CheckResult:
    try:
        r = await client.get(
            "https://gamma-api.polymarket.com/events",
            params={"tag_slug": "highest-temperature", "active": "true", "limit": 1},
        )
    except Exception as exc:
        return CheckResult("polymarket", "FAIL", f"unreachable: {exc}")
    if r.status_code == 200:
        n = len(r.json() or [])
        return CheckResult("polymarket", "OK", f"reachable, {n} event in test page")
    return CheckResult("polymarket", "WARN", f"HTTP {r.status_code}")


async def check_metar(client: httpx.AsyncClient) -> CheckResult:
    """Iowa State ASOS — our truth source for Polymarket resolution."""
    try:
        r = await client.get(
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            params={
                "station": "EGLC", "data": "tmpc",
                "year1": 2025, "month1": 12, "day1": 31,
                "year2": 2026, "month2": 1, "day2": 1,
                "tz": "Etc/UTC", "format": "onlycomma",
                "latlon": "no", "missing": "null", "trace": "null",
            },
        )
    except Exception as exc:
        return CheckResult("iowa_state_asos", "FAIL", f"unreachable: {exc}")
    if r.status_code != 200:
        return CheckResult("iowa_state_asos", "WARN", f"HTTP {r.status_code}")
    # Sanity-check the response actually has data
    if "EGLC" not in r.text:
        return CheckResult("iowa_state_asos", "WARN", "response missing test data")
    return CheckResult("iowa_state_asos", "OK", "reachable, EGLC test sample present")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


async def gather_checks(skip_net: bool) -> list[CheckResult]:
    results = [
        check_bias_table(),
        check_forward_log_freshness(),
        check_recent_completeness(),
        check_bucket_snapshots(),
        check_resolution_lag(),
        check_disk_space(),
        check_unmatched_cities(),
    ]
    if not skip_net:
        async with httpx.AsyncClient(timeout=10.0) as client:
            results.extend(
                await asyncio.gather(
                    check_open_meteo(client),
                    check_polymarket(client),
                    check_metar(client),
                )
            )
    return results


def _marker(status: Status) -> str:
    return {"OK": "✓", "WARN": "!", "FAIL": "✗"}[status]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true", help="suppress OK lines")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--no-net", action="store_true", help="skip network checks")
    args = p.parse_args()

    results = await gather_checks(skip_net=args.no_net)
    worst = max((r.severity for r in results), default=0)

    if args.json:
        payload = {
            "overall": ["OK", "WARN", "FAIL"][worst],
            "checks": [asdict(r) for r in results],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(payload, indent=2))
    else:
        rows = [r for r in results if not (args.quiet and r.status == "OK")]
        if rows:
            print(f"{'check':<24s}  {'status':<6s}  message")
            print("-" * 90)
            for r in rows:
                print(f"{r.name:<24s}  {_marker(r.status)} {r.status:<4s}  {r.message}")
        if worst == 2:
            print("\nFAIL — at least one critical check failed.")
        elif worst == 1:
            print("\nWARN — passed with warnings.")
        elif not args.quiet:
            print("\nAll checks pass.")

    sys.exit(worst)


if __name__ == "__main__":
    asyncio.run(main())
