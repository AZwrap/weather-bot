r"""Quick demo of unit-aware bucket pricing.

For each Polymarket market, fetch the live ensemble forecast, apply the
station's bias correction, and print the implied probability per Polymarket
bucket — the same numbers the bot will compare against live market prices.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from weather_bot.bias import BiasTable, corrected_members
from weather_bot.forecast import distribution_from_forecast, fetch_ensemble
from weather_bot.forecast.probability import TempDistribution
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.units import from_celsius

# Demo: a few cities across both units
DEMO_MARKETS = [
    ("EGLC", "max"),  # London max (°C)
    ("LFPB", "max"),  # Paris max (°C)
    ("HKO",  "max"),  # Hong Kong max — bias correction should matter
    ("KLGA", "max"),  # NYC max (°F)
    ("KMIA", "min"),  # Miami min (°F)
]


def _print_pmf(label: str, dist: TempDistribution, unit: str, mode_band: int = 4) -> None:
    """Print a Polymarket-style bucket PMF centred on the modal forecast.

    `mode_band` is in BUCKETS — so for °F (2°F per bucket) we span 8°F either side.
    """
    step = 1 if unit == "C" else 2
    mode_c = dist.quantile(0.5)
    mode = round(from_celsius(mode_c, unit) / step) * step  # snap to bucket centre
    lo, hi = mode - mode_band * step, mode + mode_band * step
    pmf = dist.bucket_pmf(lo, hi, unit=unit, method="empirical")
    print(f"\n{label}")
    print(f"  modal forecast ≈ {mode}{'°C' if unit == 'C' else '°F'}, "
          f"σ_ensemble = {dist.std:.2f} °C  →  bucket probabilities:")
    total = sum(p for _, p in pmf)
    for bucket, p in pmf:
        bar = "█" * round(p * 40)
        print(f"    {bucket:>4s}  {p:6.1%}  {bar}")
    print(f"    sum = {total:.3f}")


async def main() -> None:
    bias_path = Path("bias_table.json")
    if not bias_path.exists():
        sys.exit("bias_table.json not found — run `python train_bias.py` first.")
    bias_table = BiasTable.load(bias_path)

    target_date = date.today() + timedelta(days=1)
    print(f"Target date: {target_date} (each station's local date)")

    for station_id, target in DEMO_MARKETS:
        station = STATIONS_BY_ID[station_id]
        forecast = await fetch_ensemble(station.to_location(), forecast_days=3)

        # Apply bias correction to ensemble members
        if target == "max":
            members = forecast.daily_max(target_date)
        else:
            members = forecast.daily_min(target_date)

        corrected = corrected_members(members, bias_table, station_id, target)
        dist = TempDistribution(
            location_name=station.name,
            target_date=target_date,
            members=corrected,
        )

        bias_c = bias_table.get(station_id, target)
        label = (
            f"━━ {station.name} [{target}] ({station.station_label}) "
            f"bias_correction={bias_c:+.2f}°C ━━"
        )
        _print_pmf(label, dist, station.unit)


if __name__ == "__main__":
    asyncio.run(main())
