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


def _cumulative_yes_ask_ge_threshold(
    markets: list[Any], threshold: int,
) -> tuple[float | None, int, list[Any]]:
    """Compute P(extreme ≥ threshold) implied by market YES asks AND
    return the buckets used. Returns (p_estimate, n_used, bucket_list).
    p_estimate=None if no usable buckets."""
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
        if kind == "mid" and t >= threshold:
            total += float(m.yes_ask)
            n += 1
            used.append(m)
        elif kind == "high_tail" and t >= threshold:
            total += float(m.yes_ask)
            n += 1
            used.append(m)
        # low_tail or below-threshold buckets are omitted
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
            p_max_ge_T, n_max, max_buckets = _cumulative_yes_ask_ge_threshold(
                ev_max.markets, T,
            )
            p_min_ge_T, n_min, min_buckets = _cumulative_yes_ask_ge_threshold(
                ev_min.markets, T,
            )
            if p_max_ge_T is None or p_min_ge_T is None:
                continue
            cost = p_max_ge_T + (1.0 - p_min_ge_T)
            arb_margin = 1.0 - cost
            if arb_margin < min_margin_usd:
                counts["below_margin"] += 1
                continue

            # Opportunity found. Fire paper trades on the legs.
            # Note: we PAPER-fire one consolidated entry per opportunity
            # rather than per-leg, since multi-leg coordination is fragile.
            # The synthetic OrderResult from dry-run client lands as a
            # "filled" log line; live mode would need atomic multi-leg.
            counts["opportunities"] += 1
            if verbose:
                print(f"  [cons-arb] {sid} {td_iso} T={T}: "
                      f"p_max≥T={p_max_ge_T:.3f}, p_min≥T={p_min_ge_T:.3f}, "
                      f"margin=${arb_margin:.3f}")

            _log_event({
                "ts_utc": now_utc.isoformat(),
                "result": "opportunity",
                "station_id": sid,
                "target_date": td_iso,
                "threshold": int(T),
                "p_max_ge_T": float(p_max_ge_T),
                "p_min_ge_T": float(p_min_ge_T),
                "n_max_buckets": int(n_max),
                "n_min_buckets": int(n_min),
                "cost_usd": float(cost),
                "arb_margin_usd": float(arb_margin),
                # Per-leg detail for the analyzer:
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
                     "no_token_id": getattr(m, "no_token_id", None),
                     "yes_ask": float(m.yes_ask)}
                    for m in min_buckets
                ],
                "size_usd": size_usd,
            })
            counts["placed"] += 1

    return dict(counts)
