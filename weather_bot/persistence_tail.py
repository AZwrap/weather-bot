"""Persistence-tail NO — bet against extreme tail buckets using yesterday-as-prior.

Premise
=======
Weather is sticky day-to-day. Typical day-over-day shift in daily extreme
is 1-3°C on stable stations, 3-5°C on more variable ones. Tail buckets
("X°F or higher", "X°F or below") that are 4+ buckets away from
yesterday's actual will almost certainly NOT win — they're priced
speculatively at $0.05-$0.15 YES (= $0.85-$0.95 NO), where the
liquidity sits.

Mechanism
=========
For each station-event resolving today or tomorrow:
  1. Find yesterday's actual_obs_c for the same (station, target) from
     forward_log.jsonl. If unavailable, skip.
  2. Convert to market-unit integer via _rounded_observation.
  3. For each TAIL bucket in the event:
       - low_tail "X or below":  fire NO if X is >= 4 buckets below yesterday's actual
       - high_tail "X or higher": fire NO if X is >= 4 buckets above yesterday's actual
  4. Submit FAK BUY NO at the existing $0.99 cap (2-decimal clean).
  5. Dedupe via Portfolio.is_open.

Sizing: $5/fire (matches Layer 7 / V2 / HBN).
Price band: [$0.50, $0.99]. Below $0.50 = market disagrees strongly →
refuse. At $0.99 = cap.

Risks
=====
- Fat-tail weather: heat waves and cold fronts can shift daily extreme
  5+°C overnight. One bad day eats weeks of small wins. The 4-bucket
  minimum distance is the first line of defense; size cap is the
  second.
- Yesterday's resolution might not yet be in forward_log if we haven't
  built/run the resolver — strategy will silently skip until that
  exists. Future improvement: pull from Polymarket trade history.

Output: data/persistence_tail_log.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any

from .locations import STATIONS_BY_ID
from .obs_distance_filter import bucket_edges_c
from .pnl import _rounded_observation
from .polymarket import parse_bucket
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position, region_for
from .scanner import TradeSignal


DEFAULT_LOG_PATH = Path("data/persistence_tail_log.jsonl")
FORWARD_LOG_PATH = Path("data/forward_log.jsonl")

N_BUCKETS_MIN_DISTANCE: int = 4
"""Minimum bucket-step distance from yesterday's actual before we'll
fire on a tail bucket. At 4, we're betting the day-over-day shift is
less than 4 buckets (4°F for °F markets, 4°C for °C markets). On stable
stations this is >95% reliable; on continental/desert stations more
variable. Tune after N≥30 paper resolutions."""

MIN_NO_ASK: float = 0.50
MAX_NO_ASK: float = 0.99
DEFAULT_SIZE_USD: float = 5.0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _load_yesterday_actuals(
    forward_log_path: Path = FORWARD_LOG_PATH,
    lookback_days: int = 14,
) -> dict[tuple[str, str], float]:
    """Return {(station_id, target): actual_obs_c} for the most recent
    resolved record per (station, target) within `lookback_days`.

    forward_log records have `actual_obs_c` set after resolution and
    `target_date` (ISO string). Returns Celsius regardless of station
    unit (forward_log stores Celsius)."""
    if not forward_log_path.exists():
        return {}
    by_key: dict[tuple[str, str], tuple[str, float]] = {}
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=lookback_days)).isoformat()
    with forward_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("station_id")
            target = r.get("target")
            td = r.get("target_date")
            actual = r.get("actual_obs_c")
            if not (sid and target and td and actual is not None):
                continue
            if td < cutoff:
                continue
            try:
                actual_c = float(actual)
            except (TypeError, ValueError):
                continue
            key = (sid, target)
            cur = by_key.get(key)
            if cur is None or cur[0] < td:
                by_key[key] = (td, actual_c)
    return {k: v[1] for k, v in by_key.items()}


def detect_and_execute_persistence_tail(
    *,
    station_id: str,
    target_date_iso: str,
    target: str,                # "max" | "min"
    bucket_snapshots: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    n_buckets_min_distance: int = N_BUCKETS_MIN_DISTANCE,
    min_no_ask: float = MIN_NO_ASK,
    max_no_ask: float = MAX_NO_ASK,
    size_usd: float = DEFAULT_SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    yesterday_actuals: dict[tuple[str, str], float] | None = None,
    depth_map: dict | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    """Fire NO on tail buckets that are N_BUCKETS_MIN_DISTANCE+ steps
    away from yesterday's actual extreme for the (station, target).

    Returns counts dict:
      placed              successful fires
      skipped_no_prior    no resolved actual for (station, target) in
                          forward_log within lookback window
      skipped_already_open dedupe via Portfolio.is_open
      skipped_no_book     no NO ask available
      skipped_ask_too_high NO ask > max_no_ask
      skipped_ask_too_low  NO ask < min_no_ask
      skipped_too_close    bucket within N_BUCKETS_MIN_DISTANCE of prior
      submit_failed       SDK / Polymarket rejection
      no_candidates       no tail buckets met the distance criterion
    """
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0

    station = STATIONS_BY_ID.get(station_id)
    if station is None:
        return dict(counts)

    if yesterday_actuals is None:
        yesterday_actuals = _load_yesterday_actuals()

    prior_c = yesterday_actuals.get((station_id, target))
    if prior_c is None:
        counts["skipped_no_prior"] += 1
        return dict(counts)

    prior_int = _rounded_observation(prior_c, station.unit)

    # Find tail bucket candidates by distance.
    candidates: list[tuple[Any, str, int, float, float, int]] = []  # ..., distance_buckets
    for m in bucket_snapshots:
        if not m.no_token_id:
            continue
        try:
            kind, thr = parse_bucket(m)
            low_c, high_c = bucket_edges_c(kind, int(thr), station.unit)
        except (ValueError, TypeError, KeyError):
            continue
        if kind not in ("low_tail", "high_tail"):
            continue

        # Distance in market-unit bucket steps:
        # low_tail "X or below": dies when prior > X. Distance = prior - X.
        # high_tail "X or higher": dies when prior < X. Distance = X - prior.
        if kind == "low_tail":
            distance = prior_int - int(thr)
        else:  # high_tail
            distance = int(thr) - prior_int

        if distance < n_buckets_min_distance:
            counts["skipped_too_close"] += 1
            continue
        candidates.append((m, kind, int(thr), low_c, high_c, distance))

    if not candidates:
        counts["no_candidates"] += 1
        return dict(counts)

    if verbose:
        print(f"  [pers-tail] {station_id} {target} prior={prior_int} "
              f"({prior_c:.1f}°C) → {len(candidates)} candidate tail(s)")

    for m, kind, thr, low_c, high_c, distance in candidates:
        if portfolio.is_open(m.no_token_id, "NO"):
            counts["skipped_already_open"] += 1
            continue

        # NO ask ≈ 1 − yes_bid for Polymarket binary outcomes
        if m.yes_bid is None:
            counts["skipped_no_book"] += 1
            continue
        no_ask = 1.0 - float(m.yes_bid)
        if no_ask > max_no_ask:
            counts["skipped_ask_too_high"] += 1
            continue
        if no_ask < min_no_ask:
            counts["skipped_ask_too_low"] += 1
            continue

        # Submit at the 2-decimal cap; matching engine walks the book.
        submitted_limit = round(float(max_no_ask), 2)

        # DEPTH-AWARE FILL — walk the NO ask ladder up to the limit.
        from .polymarket import simulate_buy_fill
        depth = (depth_map or {}).get(m.no_token_id)
        sim = simulate_buy_fill(depth, size_usd, submitted_limit)
        if sim is None:
            if depth_map is not None:
                counts["skipped_no_depth"] += 1
                continue
            fill_avg = no_ask
            shares = float(max(1, int(size_usd / submitted_limit)))
            fully_filled = True
            depth_source = "top_of_book_fallback"
        else:
            fill_avg, shares, fully_filled = sim
            depth_source = "depth_walk"

        signal = TradeSignal(
            station=station, event_title="", event_slug="",
            target=target,
            target_date=datetime.fromisoformat(target_date_iso).date(),
            bucket_label=m.bucket_label, bucket_kind=kind,
            market_id=int(m.market_id), token_id=m.no_token_id,
            our_prob=1.0 - no_ask, yes_implied=float(m.yes_ask or 0.0),
            yes_bid=m.yes_bid, yes_ask=m.yes_ask,
            side="NO", edge=1.0 - no_ask,
            fill_price=fill_avg, volume_24hr=0.0,
            bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
            kelly_full=1.0, position_usd=fill_avg * shares,
        )
        try:
            result = client.submit_order(
                signal, order_type="FAK", sdk_side="BUY",
                limit_price=submitted_limit, override_shares=shares,
            )
        except Exception as exc:
            counts["submit_failed"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(), "result": "submit_exception",
                "station_id": station_id, "target": target,
                "target_date": target_date_iso, "bucket_label": m.bucket_label,
                "prior_c": prior_c, "distance_buckets": distance,
                "no_ask": no_ask, "exc": str(exc)[:200],
            }, log_path)
            continue
        if not result.ok:
            counts["submit_failed"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(), "result": "rejected",
                "station_id": station_id, "target": target,
                "target_date": target_date_iso, "bucket_label": m.bucket_label,
                "prior_c": prior_c, "distance_buckets": distance,
                "no_ask": no_ask,
                "message": (result.message or "")[:200],
            }, log_path)
            continue

        position = Position(
            token_id=m.no_token_id, side="NO",
            station_id=station_id, region=region_for(station_id),
            market_id=int(m.market_id), bucket_label=m.bucket_label,
            bucket_kind=kind, bucket_threshold=int(thr),
            target_date=target_date_iso,
            shares=shares, entry_price=fill_avg, position_usd=fill_avg * shares,
            submitted_at=_now_utc_iso(), status="filled",
            order_id=result.order_id, strategy="persistence_tail",
        )
        try:
            portfolio.add(position)
            portfolio.save(portfolio_path)
        except Exception:
            pass
        counts["placed"] += 1
        _log_event({
            "ts_utc": _now_utc_iso(), "result": "filled",
            "station_id": station_id, "target": target,
            "target_date": target_date_iso, "bucket_label": m.bucket_label,
            "bucket_kind": kind, "bucket_threshold": int(thr),
            "prior_c": prior_c, "prior_int": prior_int,
            "distance_buckets": distance,
            "bucket_low_c": low_c if low_c != float("-inf") else None,
            "bucket_high_c": high_c if high_c != float("inf") else None,
            "no_ask_snapshot": no_ask,
            "submitted_limit": submitted_limit,
            "fill_price": result.fill_price,
            "depth_source": depth_source,
            "fully_filled": fully_filled,
            "shares": shares, "size_usd": fill_avg * shares,
            "order_id": result.order_id,
        }, log_path)

    return dict(counts)
