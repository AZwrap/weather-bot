"""Preview test: does multi-model agreement correlate with hit rate?

Re-fetches historical forecasts from 4 models (ECMWF, GFS, ICON, GEM)
via Open-Meteo's historical-forecast API for each resolved
(station, target_date) tuple. Computes the std-dev of predicted
max/min across models = agreement_std.

Cross-references against the bot's positions on May 8 to check whether:
  - High-agreement + bot-disagrees-market trades win at higher rate
  - Low-agreement trades suffer regardless

With N=1 resolved day this is a directional preview, not definitive.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import asyncio
import httpx
import statistics as stats
from collections import defaultdict

from weather_bot.forward_log import load_records
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.positions import replay_maker

MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]


async def fetch_multimodel(lat: float, lon: float, target_date: str) -> dict:
    """Fetch daily max/min from each model for target_date."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "UTC",
        "models": ",".join(MODELS),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def parse_models(response: dict, target: str) -> dict[str, float]:
    """Extract per-model temperature for max or min."""
    daily = response.get("daily", {})
    field = "temperature_2m_max" if target == "max" else "temperature_2m_min"
    out = {}
    for model in MODELS:
        key = f"{field}_{model}"
        vals = daily.get(key, [])
        if vals and vals[0] is not None:
            out[model] = float(vals[0])
    return out


async def main():
    recs = load_records()
    elig = [r for r in recs if r.bucket_snapshots is not None]

    # Get unique (station, target, target_date) for resolved records
    resolved_keys: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if r.actual_obs_c is None:
            continue
        key = (r.station_id, r.target, r.target_date.isoformat())
        if key not in resolved_keys:
            resolved_keys[key] = []
        resolved_keys[key].append(r)

    print(f"unique resolved (station, target, date) tuples: {len(resolved_keys)}")

    # Run replay_maker to get positions
    positions = replay_maker(
        elig, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
        min_edge=0.05, max_edge=0.25, min_yes_price=0.05, max_yes_price=0.95,
        sigma_inflation_factor=1.4, taker_fallback=False,
    )
    resolved_positions = [p for p in positions if p.closed]
    print(f"resolved positions: {len(resolved_positions)}")

    # Fetch multi-model forecasts for each unique (station, target_date)
    print(f"\nFetching multi-model forecasts ({len(resolved_keys)} stations)...")
    multimodel_data: dict[tuple, dict[str, float]] = {}
    fetched = 0
    for (sid, target, td), _ in resolved_keys.items():
        station = STATIONS_BY_ID.get(sid)
        if station is None:
            continue
        try:
            resp = await fetch_multimodel(station.latitude, station.longitude, td)
            preds = parse_models(resp, target)
            if len(preds) >= 2:
                multimodel_data[(sid, target, td)] = preds
                fetched += 1
        except Exception as e:
            print(f"  error fetching {sid} {target} {td}: {e}")
        # Throttle to avoid rate limits
        await asyncio.sleep(0.1)

    print(f"successfully fetched: {fetched}/{len(resolved_keys)}")

    # Compute agreement std for each
    agreements = []
    for (sid, target, td), preds in multimodel_data.items():
        std = stats.stdev(preds.values()) if len(preds) >= 2 else 0
        mean = stats.mean(preds.values())
        agreements.append((sid, target, td, std, mean, preds))

    print()
    print("Agreement-std distribution across stations:")
    stds = [a[3] for a in agreements]
    if stds:
        print(f"  min:    {min(stds):.3f}°C")
        print(f"  median: {stats.median(stds):.3f}°C")
        print(f"  mean:   {stats.mean(stds):.3f}°C")
        print(f"  max:    {max(stds):.3f}°C")
    print()

    # Tag each resolved position with its station-target-date agreement_std
    pos_with_agree = []
    for p in resolved_positions:
        key = (p.station_id, p.target, p.target_date)
        if key not in multimodel_data:
            continue
        preds = multimodel_data[key]
        if len(preds) < 2:
            continue
        std = stats.stdev(preds.values())
        # Compute "bot-vs-market direction" — did our_prob disagree with market?
        # Use the open event's recorded our_prob and market_implied
        open_ev = p.open_event
        bot_prob = open_ev.our_prob_at_step
        market_implied = open_ev.market_yes_implied_at_step
        if market_implied is None:
            continue
        if p.side == "NO":
            bot_prob = 1.0 - bot_prob
            market_implied = 1.0 - market_implied
        diff = bot_prob - market_implied
        won = p.realized_profit_usd > 0
        pos_with_agree.append((p, std, diff, won))

    print(f"resolved positions tagged with agreement: {len(pos_with_agree)}")

    # Categorize: agreement (low/high), market-direction (match/disagree)
    median_std = stats.median([t[1] for t in pos_with_agree]) if pos_with_agree else 0.5
    print(f"split agreement at median std = {median_std:.3f}°C")

    cells: dict[str, list] = defaultdict(list)
    for p, std, diff, won in pos_with_agree:
        agreement_label = "high_agree" if std < median_std else "low_agree"
        # bot disagrees with market means (bot_prob - market_implied) is large
        # but we entered the trade because of edge, so always bot > market on chosen side
        # use magnitude as "how much the bot disagreed"
        edge_magnitude = abs(diff)
        # split edge at 0.05 (small vs large disagreement)
        edge_label = "big_edge" if edge_magnitude >= 0.05 else "small_edge"
        cells[f"{agreement_label} / {edge_label}"].append(won)

    print()
    print("Hit rate by cell (agreement × edge size):")
    print(f"{'cell':35s} {'n':>5s} {'wins':>5s} {'losses':>7s} {'wr':>6s}")
    for cell_name in ["high_agree / big_edge", "high_agree / small_edge",
                      "low_agree / big_edge",  "low_agree / small_edge"]:
        results = cells.get(cell_name, [])
        if not results:
            print(f"{cell_name:35s} {'-':>5s} {'-':>5s} {'-':>7s} {'-':>6s}")
            continue
        n = len(results)
        w = sum(results)
        l = n - w
        wr = w / n if n else 0
        print(f"{cell_name:35s} {n:>5d} {w:>5d} {l:>7d} {wr*100:>5.1f}%")

    # Cross-check: agreement_std vs predicted-vs-actual error
    # (sanity: do high-agreement forecasts actually have lower error?)
    print()
    print("Sanity: model agreement vs forecast error (lower = better):")
    print(f"{'std bin':18s} {'n':>4s} {'mean_err':>9s} {'rmse':>8s}")
    bins = [(0, 0.5, "0-0.5°C tight"), (0.5, 1.0, "0.5-1°C"),
            (1.0, 2.0, "1-2°C wide"), (2.0, 99, "2°C+ very wide")]
    for lo, hi, label in bins:
        items = []
        for sid, target, td, std, mean, _preds in agreements:
            if lo <= std < hi:
                # Find actual outcome for this station-target-date
                key_records = resolved_keys.get((sid, target, td), [])
                if not key_records:
                    continue
                actual = key_records[-1].actual_obs_c
                if actual is None:
                    continue
                err = mean - actual
                items.append(err)
        if not items:
            continue
        n = len(items)
        mae = stats.mean([abs(e) for e in items])
        rmse = (stats.mean([e**2 for e in items])) ** 0.5
        print(f"{label:18s} {n:>4d}  {mae:>6.3f}°C  {rmse:>5.3f}°C")


if __name__ == "__main__":
    asyncio.run(main())
