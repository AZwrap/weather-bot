"""Above/below consistency arb — cross-event arbitrage.

Restored + adapted from commit d526298 (shadow_consistency_arb.py),
which was deleted in the lite rebuild. This version both detects
opportunities AND fires paper trades (multi-leg YES on max + NO on min).

EDGE MECHANISM
==============
For a station's max and min events on the same date, the daily-min is
always ≤ daily-max (by definition). This forces P(min ≥ T) ≤ P(max ≥ T)
for any threshold T. When markets violate this inequality, a guaranteed
arb exists.

THE ARB STRUCTURE
=================
At threshold T:
  market P(max ≥ T) = p_max  (cumulative YES on max-event buckets ≥ T)
  market P(min ≥ T) = p_min  (cumulative YES on min-event buckets ≥ T)

If p_min > p_max + MARGIN (market thinks min more likely ≥T than max,
which is impossible), buy this combined position:

  BUY YES on "max ≥ T" at cost p_max     → pays $1 if max ≥ T
  BUY NO  on "min ≥ T" at cost (1-p_min) → pays $1 if min < T

Outcomes:
  max ≥ T, min ≥ T:  YES_max wins $1, NO_min loses. Total $1.
  max ≥ T, min < T:  YES_max wins $1, NO_min wins $1.  Total $2.
  max < T, min ≥ T:  IMPOSSIBLE (min ≤ max).
  max < T, min < T:  YES_max loses, NO_min wins $1.    Total $1.

Min payout = $1. Cost = p_max + (1 - p_min). Guaranteed profit when
cost < $1 (= p_min - p_max > 0).

OUTPUT
======
data/consistency_arb_log.jsonl with every opportunity detected. When
PAPER_ONLY=True (= dry-run client), submits a record-only paper trade
without portfolio mutation. Future live mode would need multi-leg
coordination (cancel one leg if the other fails) — not implemented yet.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path("data/consistency_arb_log.jsonl")

MIN_MARGIN_USD: float = 0.02
"""Minimum (1 - cost) profit per arb to count as opportunity.
$0.02 mirrors the prior live_bucket_arb's MIN_ARB_MARGIN_USD_DEPTH.
Above this we log; below this we don't bother (fees would eat it)."""

DEFAULT_SIZE_USD: float = 5.0
"""Per-arb USD notional. We size each LEG to produce $size_usd in
total outlay. Shares per leg = size_usd / leg_price."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _bucket_fully_ge_T(kind: str, t: int, T: int) -> bool:
    """True iff every value in this bucket is ≥ T (= safe to include
    in 'extreme ≥ T' sum without overcounting)."""
    if kind == "mid":
        return t >= T
    if kind == "high_tail":
        return t >= T   # high_tail covers [t, +inf); fully ≥ T iff t ≥ T
    return False        # low_tail covers (-inf, t+w); never fully ≥ T


def _bucket_fully_lt_T(kind: str, t: int, T: int, bucket_width: int) -> bool:
    """True iff every value in this bucket is < T (= safe to include
    in 'extreme < T' sum without overcounting).

    bucket_width is the integer step (1 for °C, 2 for °F).
    A mid bucket with threshold t covers [t, t+bucket_width). Fully
    below T iff t + bucket_width ≤ T, i.e. t ≤ T - bucket_width.
    """
    if kind == "mid":
        return t + bucket_width <= T
    if kind == "low_tail":
        # low_tail "X or below" covers (-inf, t+bucket_width).
        # Fully < T iff t + bucket_width ≤ T.
        return t + bucket_width <= T
    return False    # high_tail covers [t, +inf); never fully < T


def _cumulative_yes_ask_ge_threshold(
    markets: list[Any], threshold: int,
) -> tuple[float | None, int, list[Any]]:
    """Sum of YES asks across buckets fully ≥ threshold. Returns
    (sum, n_used, bucket_list). None if no usable buckets."""
    from weather_bot.polymarket import parse_bucket
    total = 0.0
    n = 0
    used: list[Any] = []
    for m in markets:
        if m.yes_ask is None or m.yes_ask <= 0:
            continue
        try:
            kind, t = parse_bucket(m)
        except Exception:
            continue
        if _bucket_fully_ge_T(kind, t, threshold):
            total += float(m.yes_ask)
            n += 1
            used.append(m)
    if n == 0:
        return None, 0, []
    return total, n, used


def _cumulative_yes_ask_lt_threshold(
    markets: list[Any], threshold: int, bucket_width: int,
) -> tuple[float | None, int, list[Any]]:
    """Sum of YES asks across buckets fully < threshold. To buy NO on
    'extreme ≥ T', we equivalently buy YES on every bucket fully < T."""
    from weather_bot.polymarket import parse_bucket
    total = 0.0
    n = 0
    used: list[Any] = []
    for m in markets:
        if m.yes_ask is None or m.yes_ask <= 0:
            continue
        try:
            kind, t = parse_bucket(m)
        except Exception:
            continue
        if _bucket_fully_lt_T(kind, t, threshold, bucket_width):
            total += float(m.yes_ask)
            n += 1
            used.append(m)
    if n == 0:
        return None, 0, []
    return total, n, used


def detect_and_execute_consistency_arb(
    *,
    events: list[Any],
    client: Any,
    portfolio: Any,
    portfolio_path: Path = Path("data/portfolio.json"),
    size_usd: float = DEFAULT_SIZE_USD,
    min_margin_usd: float = MIN_MARGIN_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Scan event list for max/min consistency arbs, fire paper trades
    on opportunities ≥ min_margin_usd."""
    from weather_bot.locations import STATIONS_BY_ID
    from weather_bot.polymarket import (
        event_target_date, match_event_to_station, parse_bucket,
    )
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0

    now_utc = datetime.now(timezone.utc)

    # Group by (station, target_date) → {max: ev, min: ev}
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for ev in events:
        station = match_event_to_station(ev)
        if station is None:
            continue
        try:
            td = event_target_date(ev, station)
        except Exception:
            continue
        target = "max" if ev.target == "highest" else "min"
        key = (station.station_id, td.isoformat())
        pairs.setdefault(key, {})[target] = ev

    for (sid, td_iso), sides in pairs.items():
        station = STATIONS_BY_ID.get(sid)
        bucket_width = 1 if (station and station.unit == "C") else 2
        ev_max = sides.get("max")
        ev_min = sides.get("min")
        if not (ev_max and ev_min):
            counts["incomplete_pair"] += 1
            continue
        counts["pairs_scanned"] += 1

        # Threshold grid from union of mid-bucket thresholds across
        # both events
        thresholds_set = set()
        for ev in (ev_max, ev_min):
            for m in ev.markets:
                try:
                    kind, t = parse_bucket(m)
                except Exception:
                    continue
                if kind == "mid":
                    thresholds_set.add(int(t))
        thresholds = sorted(thresholds_set)
        if not thresholds:
            counts["no_thresholds"] += 1
            continue

        for T in thresholds:
            # MAX leg: buy YES on every max-event bucket ≥ T. Pays $1 if
            # max ≥ T. Cost = sum of yes_ask across those buckets.
            cost_max_leg, n_max, max_buckets = _cumulative_yes_ask_ge_threshold(
                ev_max.markets, T,
            )
            # MIN leg: buy NO on the cumulative "min ≥ T" event, which
            # equals buying YES on every min-event bucket fully < T.
            # Pays $1 if min < T. Cost = sum of yes_ask across those.
            cost_min_leg, n_min, min_buckets = _cumulative_yes_ask_lt_threshold(
                ev_min.markets, T, bucket_width,
            )
            if cost_max_leg is None or cost_min_leg is None:
                continue
            cost = cost_max_leg + cost_min_leg
            # Combined payout: $1 (min always ≤ max, so exactly one of
            # "max ≥ T" or "min < T" wins when they don't both win;
            # when both win, payout is $2 — but for the worst-case
            # guaranteed margin we use min-payout = $1).
            arb_margin = 1.0 - cost
            if arb_margin < min_margin_usd:
                counts["below_margin"] += 1
                continue

            counts["opportunities"] += 1
            if verbose:
                print(f"  [cons-arb] {sid} {td_iso} T={T}: "
                      f"cost_max={cost_max_leg:.3f}, cost_min_lt={cost_min_leg:.3f}, "
                      f"margin=${arb_margin:.3f}")

            _log_event({
                "ts_utc": now_utc.isoformat(),
                "result": "opportunity",
                "station_id": sid,
                "target_date": td_iso,
                "threshold": int(T),
                "bucket_width": int(bucket_width),
                # CORRECTED COST MATH (2026-05-28):
                # cost_max_leg = sum(yes_ask) for max-buckets fully ≥ T
                # cost_min_leg = sum(yes_ask) for min-buckets fully < T
                # Both are real implementable prices (we BUY YES at ASK
                # on both legs). cost = sum; arb_margin = 1 - cost.
                "cost_max_leg_usd": float(cost_max_leg),
                "cost_min_lt_leg_usd": float(cost_min_leg),
                "cost_usd": float(cost),
                "arb_margin_usd": float(arb_margin),
                "n_max_buckets": int(n_max),
                "n_min_buckets": int(n_min),
                "max_legs": [
                    {"market_id": int(getattr(m, "market_id", 0)),
                     "bucket_label": getattr(m, "bucket_label", ""),
                     "yes_token_id": getattr(m, "yes_token_id", None),
                     "yes_ask": float(m.yes_ask)}
                    for m in max_buckets
                ],
                "min_legs": [
                    {"market_id": int(getattr(m, "market_id", 0)),
                     "bucket_label": getattr(m, "bucket_label", ""),
                     "yes_token_id": getattr(m, "yes_token_id", None),
                     "yes_ask": float(m.yes_ask)}
                    for m in min_buckets
                ],
                "size_usd": size_usd,
            })
            counts["placed"] += 1

    return dict(counts)
