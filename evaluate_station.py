r"""Run a 180-day METAR-based skill backtest on a candidate station that
isn't yet in `weather_bot.locations`. Tells you whether it's worth adding.

Designed for the new-station triage flow:
    1. `check_new_stations.py` shows cities Polymarket lists that we don't trade
    2. You manually look up: ICAO, lat/lon, IANA timezone, °C or °F
    3. Run this script with those details
    4. Get tier verdict (★★★ / ★★ / ★ / ✗) — same classification as the
       original station registry was built from
    5. If tradable, manually add to `weather_bot/locations.py` MARKETS list

Usage:
    python evaluate_station.py \
        --name "Mumbai" \
        --icao VABB \
        --lat 19.0887 --lon 72.8682 \
        --tz Asia/Kolkata \
        --unit C \
        --target max \
        --days 180
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.backtest import run_skill_backtest
from weather_bot.forecast.fetcher import Location


def _classify(report) -> tuple[str, str]:
    """Return (tier, recommendation) — same scheme as backtest_skill.py."""
    if report.forecast_mae_c >= report.persistence_mae_c:
        return "✗ skip", "Skill ≤ persistence — model has no edge over 'same as today'."
    if report.forecast_mae_c < 0.5 and abs(report.forecast_bias_c) < 0.5:
        return "★★★", "Trade now — clean MAE and small bias."
    if report.forecast_mae_c < 1.0:
        return "★★ bias-fix", (
            f"Trade with per-station bias correction "
            f"(apply +{-report.forecast_bias_c:+.2f}°C to forecasts)."
        )
    return "★ check", "Investigate before trading — MAE > 1°C suggests station mismatch."


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="City display name (e.g. 'Mumbai')")
    p.add_argument("--icao", required=True, help="ICAO airport code (e.g. VABB)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True, help="IANA timezone (e.g. Asia/Kolkata)")
    p.add_argument("--unit", choices=["C", "F"], default="C",
                   help="Polymarket resolution unit for this station")
    p.add_argument("--target", choices=["max", "min"], default="max")
    p.add_argument("--days", type=int, default=180,
                   help="Backtest window length in days (default 180)")
    p.add_argument("--model", default="ecmwf_ifs025")
    args = p.parse_args()

    location = Location(
        name=args.name, latitude=args.lat, longitude=args.lon, timezone=args.tz,
    )
    end = date.today() - timedelta(days=6)
    print(
        f"Evaluating candidate station: {args.name} ({args.icao})  "
        f"{args.lat:+.4f},{args.lon:+.4f}  {args.tz}  unit={args.unit}\n"
        f"Backtest window: {end - timedelta(days=args.days - 1)} → {end}  "
        f"({args.days} days, target={args.target}, model={args.model}, METAR truth)\n"
    )

    try:
        report = await run_skill_backtest(
            location,
            days=args.days,
            end=end,
            model=args.model,
            target=args.target,
            icao=args.icao,
        )
    except Exception as exc:
        print(f"!! Backtest failed: {exc}")
        sys.exit(1)

    print(report.pretty())
    print()
    tier, rec = _classify(report)
    print(f"Tier: {tier}")
    print(f"Recommendation: {rec}")
    print()
    if tier in ("★★★", "★★ bias-fix"):
        print("To add this market, edit weather_bot/locations.py:")
        print(f'   1. Add to STATIONS_BY_ID:')
        print(f'      Station("{args.name}", "{args.icao}", '
              f'"<station-name-as-on-polymarket>", '
              f'{args.lat}, {args.lon}, "{args.tz}", "{args.unit}"),')
        print(f'   2. Add to MARKETS:')
        print(f'      ("{args.icao}", "{ "highest" if args.target == "max" else "lowest" }"),')
        print(f'   3. Then run:  python train_bias.py')
        print(f'      to refresh the bias_table.json with the new station.')


if __name__ == "__main__":
    asyncio.run(main())
