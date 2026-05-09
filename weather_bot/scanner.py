"""Market scanner: compute trade signals from Polymarket vs our model.

Pipeline:
  1. Fetch all active highest- and lowest-temperature events.
  2. Match each to a station in our registry.
  3. Fetch the live ensemble forecast once per station (deduplicated, parallel).
  4. Apply per-(station, target) bias correction to the ensemble members.
  5. For each event's buckets, compute our probability and compare to the
     market's yes-side bid/ask. Emit a `TradeSignal` when |edge| > threshold.
  6. Rank by edge × min(volume24hr, cap) so a 50% edge in a thin market
     doesn't crowd out a 5% edge in a deep one.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone as tz
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from .bias import BiasTable, corrected_members, predictive_members
from .forecast.fetcher import EnsembleForecast, fetch_ensemble
from .forecast.probability import TempDistribution, bucket_prob
from .locations import STATIONS_BY_ID, Station
from .polymarket import (
    PolymarketEvent,
    PolymarketMarket,
    apply_clob_prices,
    event_target_date,
    fetch_all_temperature_events,
    fetch_clob_prices_batch,
    match_event_to_station,
    parse_bucket,
)
from .sizing import position_size_usd, kelly_fraction
from .units import Unit

Side = Literal["YES", "NO"]


@dataclass
class TradeSignal:
    station: Station
    event_title: str
    event_slug: str
    target: Literal["max", "min"]
    target_date: date
    bucket_label: str
    bucket_kind: str             # "low_tail", "mid", "high_tail"
    market_id: int
    token_id: str                # the side we'd buy
    our_prob: float
    yes_implied: float           # market's mid yes-probability
    yes_bid: float | None
    yes_ask: float | None
    side: Side                   # "YES" or "NO" — which side has positive edge
    edge: float                  # in probability units (0.0 - 1.0)
    fill_price: float            # the price we'd pay on the chosen side
    volume_24hr: float
    bias_applied_c: float        # the bias correction we applied
    sigma_ensemble_c: float      # raw ensemble spread (informational)
    sigma_total_c: float         # σ after combining ensemble + residual variance
    kelly_full: float            # full-Kelly fraction (0..1)
    position_usd: float          # USD size after fractional-Kelly sizing

    @property
    def score(self) -> float:
        """Edge × bounded liquidity. Caps liquidity at $10k 24h to avoid
        rewarding stale-volume markets."""
        return self.edge * min(self.volume_24hr, 10_000.0)


# ──────────────────────────────────────────────────────────────────────────
# Per-bucket probability under our model
# ──────────────────────────────────────────────────────────────────────────


# Backwards-compat alias; the implementation lives in forecast.probability.
_bucket_probability = bucket_prob


# ──────────────────────────────────────────────────────────────────────────
# Edge calculation
# ──────────────────────────────────────────────────────────────────────────


def _edge_for_market(
    our_prob: float, market: PolymarketMarket
) -> tuple[Side, float, float] | None:
    """Decide which side has more edge and by how much.

    Returns (side, edge, fill_price) or None if no actionable signal.
    `fill_price` is the ask we'd pay (yes_ask for YES, 1-yes_bid for NO).
    """
    yes_ask = market.yes_ask
    yes_bid = market.yes_bid
    if yes_ask is None and yes_bid is None and market.last_trade_price is None:
        return None

    # Buying YES: we pay yes_ask, profit if outcome resolves yes.
    # Edge = our_prob - yes_ask  (positive ⇒ EV+).
    yes_fill = yes_ask if yes_ask is not None else market.yes_implied
    if yes_fill is None:
        edge_yes = -1.0
    else:
        edge_yes = our_prob - yes_fill

    # Buying NO: we pay (1 - yes_bid), profit if outcome resolves no.
    # Equivalent edge in YES space: (1 - our_prob) - (1 - yes_bid) = yes_bid - our_prob.
    no_fill = (1 - yes_bid) if yes_bid is not None else (
        (1 - market.yes_implied) if market.yes_implied is not None else None
    )
    if no_fill is None:
        edge_no = -1.0
    else:
        edge_no = (1 - our_prob) - no_fill

    if edge_yes >= edge_no:
        return "YES", edge_yes, yes_fill if yes_fill is not None else 0.0
    return "NO", edge_no, no_fill if no_fill is not None else 0.0


# ──────────────────────────────────────────────────────────────────────────
# The pipeline
# ──────────────────────────────────────────────────────────────────────────


def _is_actionable(
    event: PolymarketEvent, station: Station, now_utc: datetime
) -> bool:
    """True if the target day hasn't started yet in the station's local time.

    Filters out already-resolved markets (today) and stale ones (yesterday).
    The day's high/low temperature is determined within the local calendar
    day, so once that day starts there's already partial observation —
    forecasting it post-hoc isn't honest trading.
    """
    target_date = event_target_date(event, station)
    now_local = now_utc.astimezone(ZoneInfo(station.timezone))
    return target_date > now_local.date()


async def scan(
    bias_table: BiasTable,
    *,
    min_edge: float = 0.03,
    max_edge: float = 0.25,
    min_yes_price: float = 0.05,
    max_yes_price: float = 0.95,
    min_volume_24hr: float = 0.0,
    forecast_concurrency: int = 2,
    only_station_ids: set[str] | None = None,
    include_today: bool = False,
    inflate_sigma: bool = True,
    sigma_inflation_factor: float = 1.4,
    bankroll_usd: float = 1000.0,
    kelly_multiplier: float = 0.1,
    max_position_usd: float = 50.0,
    liquidity_cap_fraction: float = 0.1,
    per_event_cap_usd: float = 0.0,
    use_clob_prices: bool = True,
) -> list[TradeSignal]:
    """Scan every Polymarket weather event and return ranked trade signals.

    By default, filters out events whose target day has already started in
    the station's local timezone (those are already observed/resolved).
    Pass `include_today=True` to override.
    """
    async with httpx.AsyncClient(timeout=30.0) as gamma_client:
        events = await fetch_all_temperature_events(gamma_client)
        if use_clob_prices:
            yes_tokens = [
                m.yes_token_id
                for ev in events for m in ev.markets
                if m.yes_token_id
            ]
            fresh = await fetch_clob_prices_batch(yes_tokens, gamma_client)
            n_upd, n_unch = apply_clob_prices(events, fresh)
            print(f"  CLOB-refreshed prices: {n_upd} markets updated, "
                  f"{n_unch} unchanged or missing")

    now_utc = datetime.now(tz.utc)
    n_total = len(events)

    # Group: station_id → list of events for that station
    by_station: dict[str, list[PolymarketEvent]] = {}
    n_unmatched = 0
    n_resolved = 0
    for ev in events:
        st = match_event_to_station(ev)
        if st is None:
            n_unmatched += 1
            continue
        if only_station_ids and st.station_id not in only_station_ids:
            continue
        if not include_today and not _is_actionable(ev, st, now_utc):
            n_resolved += 1
            continue
        by_station.setdefault(st.station_id, []).append(ev)

    n_kept = sum(len(v) for v in by_station.values())
    print(
        f"  events: {n_total} total, {n_unmatched} unmatched, "
        f"{n_resolved} already-started, {n_kept} actionable"
    )

    if not by_station:
        return []

    # Fetch ensemble forecasts once per station (max 7-day window covers all upcoming events)
    sem = asyncio.Semaphore(forecast_concurrency)

    async def fetch_for_station(sid: str) -> tuple[str, EnsembleForecast | Exception]:
        station = STATIONS_BY_ID[sid]
        async with sem:
            try:
                fc = await fetch_ensemble(station.to_location(), forecast_days=7)
                return sid, fc
            except Exception as exc:
                return sid, exc

    forecasts_results = await asyncio.gather(
        *(fetch_for_station(sid) for sid in by_station)
    )
    forecasts: dict[str, EnsembleForecast] = {}
    for sid, result in forecasts_results:
        if isinstance(result, Exception):
            print(f"!! ensemble fetch {sid}: {result}")
            continue
        forecasts[sid] = result

    signals: list[TradeSignal] = []

    for sid, station_events in by_station.items():
        forecast = forecasts.get(sid)
        if forecast is None:
            continue
        station = STATIONS_BY_ID[sid]

        for ev in station_events:
            target_date = event_target_date(ev, station)
            target = "max" if ev.target == "highest" else "min"

            try:
                if target == "max":
                    raw_members = forecast.daily_max(target_date)
                else:
                    raw_members = forecast.daily_min(target_date)
            except ValueError:
                # Forecast doesn't extend to this date (event is too far out)
                continue

            bias_c = bias_table.get(sid, target)
            sigma_ensemble = float(np.std(raw_members - bias_c, ddof=1))
            members = predictive_members(
                raw_members, bias_table, sid, target,
                inflate_sigma=inflate_sigma,
                inflation_factor=sigma_inflation_factor,
            )
            sigma_total = float(np.std(members, ddof=1))
            dist = TempDistribution(
                location_name=station.name,
                target_date=target_date,
                members=members,
            )

            event_signals: list[TradeSignal] = []
            for m in ev.markets:
                if m.volume_24hr < min_volume_24hr:
                    continue
                kind, threshold = parse_bucket(m)
                our_prob = _bucket_probability(dist, kind, threshold, station.unit)
                edge_result = _edge_for_market(our_prob, m)
                if edge_result is None:
                    continue
                side, edge, fill_price = edge_result
                if edge < min_edge:
                    continue
                # Skip absurd edges (model error at the tails)
                if edge > max_edge:
                    continue
                # Skip extreme-tail markets where bid/ask is unreliable
                if not (min_yes_price <= fill_price <= max_yes_price):
                    continue

                kf = kelly_fraction(our_prob, fill_price, side)
                pos_usd = position_size_usd(
                    our_prob, fill_price, side,
                    bankroll_usd=bankroll_usd,
                    kelly_multiplier=kelly_multiplier,
                    max_position_usd=max_position_usd,
                    liquidity_cap_usd=m.volume_24hr * liquidity_cap_fraction,
                )

                event_signals.append(
                    TradeSignal(
                        station=station,
                        event_title=ev.title,
                        event_slug=ev.slug,
                        target=target,
                        target_date=target_date,
                        bucket_label=m.bucket_label,
                        bucket_kind=kind,
                        market_id=m.market_id,
                        token_id=m.yes_token_id if side == "YES" else m.no_token_id,
                        our_prob=our_prob,
                        yes_implied=float(m.yes_implied) if m.yes_implied is not None else 0.0,
                        yes_bid=m.yes_bid,
                        yes_ask=m.yes_ask,
                        side=side,
                        edge=edge,
                        fill_price=fill_price,
                        volume_24hr=m.volume_24hr,
                        bias_applied_c=bias_c,
                        sigma_ensemble_c=sigma_ensemble,
                        sigma_total_c=sigma_total,
                        kelly_full=kf,
                        position_usd=pos_usd,
                    )
                )

            # Per-event (station, target, target_date) cap: take only highest-edge
            # trades up to per_event_cap_usd. Prevents concentrating too much risk
            # on a single forecast.
            if per_event_cap_usd > 0 and event_signals:
                event_signals.sort(key=lambda s: -s.edge)
                running = 0.0
                kept = []
                for s in event_signals:
                    if running + s.position_usd <= per_event_cap_usd + 1e-9:
                        kept.append(s)
                        running += s.position_usd
                signals.extend(kept)
            else:
                signals.extend(event_signals)

    signals.sort(key=lambda s: -s.score)
    return signals
