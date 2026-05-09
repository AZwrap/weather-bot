r"""Demo: fetch multi-model ensemble forecasts and print the implied
distribution of tomorrow's daily max temperature for several cities.

Usage (PowerShell):
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python main.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta

# Force UTF-8 on Windows consoles so °C renders correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.forecast import (
    Location,
    blend_distributions,
    distribution_from_forecast,
    fetch_multi_model,
)

CITIES = [
    Location("New York",  40.7128,  -74.0060, "America/New_York"),
    Location("London",    51.5074,   -0.1278, "Europe/London"),
    Location("Paris",     48.8566,    2.3522, "Europe/Paris"),
    Location("Tokyo",     35.6762,  139.6503, "Asia/Tokyo"),
]


async def run() -> None:
    target = date.today() + timedelta(days=1)  # tomorrow, in each city's tz
    print(f"Forecast target: {target} (each city's local date)\n")

    for city in CITIES:
        print(f"=== {city.name} ===")
        forecasts = await fetch_multi_model(city, forecast_days=7)

        per_model = []
        for model, fc in forecasts.items():
            try:
                d = distribution_from_forecast(fc, target)
            except ValueError as exc:
                print(f"  {model:14s}  -- {exc}")
                continue
            print(f"  {model:14s}  {d.summary()}")
            per_model.append(d)

        if len(per_model) >= 2:
            blended = blend_distributions(per_model)
            print(f"  {'BLENDED':14s}  {blended.summary()}")
            mu = blended.mean
            print(f"    P(max >  {mu+2:.0f}°C) = {blended.prob_above(mu + 2):.3f}")
            print(f"    P(max >  {mu  :.0f}°C) = {blended.prob_above(mu):.3f}")
            print(f"    P(max >  {mu-2:.0f}°C) = {blended.prob_above(mu - 2):.3f}")
        print()


if __name__ == "__main__":
    asyncio.run(run())
