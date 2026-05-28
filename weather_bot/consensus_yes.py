"""Consensus-YES momentum — buy the highest-YES mid bucket while it's
still in the trade-able band.

⚠️  HIGHEST-RISK STRATEGY IN THE LITE REBUILD.
This is the closest path back to the 2026-05 NO_momentum bleed. Per
the user's explicit request to add anyway. Run paper-only until N≥30
resolutions validate the win rate at each entry-price band.

Premise
=======
Consensus-winner buckets have the most depth — fills are easy. The
strategy buys YES on the bucket the market is converging on, BEFORE
it reaches the $0.95+ ceiling. Sells at a convergence exit (handled
separately).

Mechanism
=========
For each event, identify the bucket with the highest mid-bucket YES
ask. Fire YES if:
  1. yes_ask is in [MIN_YES_ASK, MAX_YES_ASK] band — out of band means
     consensus hasn't formed (too low) or has fully converged (too high)
  2. We don't already hold YES on this token
  3. Per-event, at most one YES position (the leading bucket)

Sizing: $5/fire (matches other strategies).
Sell logic: NOT yet implemented. The buy is paper-only via the dry-run
client; resolution PnL is computed by the analyzer.

Risks
=====
- The bucket the market is converging on may be wrong. ~30% of the
  time, the daily extreme lands in a neighbour bucket. At entry price
  $0.65 YES, breakeven win rate is 65% — bot loses if true hit-rate
  is below that.
- Mid-bucket momentum is the failure mode of NO_momentum inverted.
  Watch the realized win rate closely.

Output: data/consensus_yes_log.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .locations import STATIONS_BY_ID
from .polymarket import parse_bucket
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position, region_for
from .scanner import TradeSignal


DEFAULT_LOG_PATH = Path("data/consensus_yes_log.jsonl")

MIN_YES_ASK: float = 0.40
"""Below this, consensus hasn't formed — too many buckets still in
contention. Wait."""

MAX_YES_ASK: float = 0.85
"""Above this we're in convergence-late territory. The bot has lost
the run-up; YES at $0.90+ is the same trade Layer 7 would have done
much cheaper had it fired. Skip."""

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


def detect_and_execute_consensus_yes(
    *,
    station_id: str,
    target_date_iso: str,
    target: str,
    bucket_snapshots: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    min_yes_ask: float = MIN_YES_ASK,
    max_yes_ask: float = MAX_YES_ASK,
    size_usd: float = DEFAULT_SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Fire YES on the leading mid bucket if yes_ask is in band.

    Returns counts:
      placed                  successful fires
      skipped_no_mid          no mid buckets in event
      skipped_no_ask          leading bucket missing yes_ask
      skipped_below_band      leading bucket yes_ask < min_yes_ask
                              (consensus hasn't formed)
      skipped_above_band      leading bucket yes_ask > max_yes_ask
                              (already converged)
      skipped_already_open    dedupe via Portfolio.is_open
      submit_failed           SDK / Polymarket rejection
    """
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0

    station = STATIONS_BY_ID.get(station_id)
    if station is None:
        return dict(counts)

    # Find the leading mid bucket (highest yes_ask among mids)
    mids: list[tuple[Any, str, int, float]] = []
    for m in bucket_snapshots:
        if not m.yes_token_id:
            continue
        try:
            kind, thr = parse_bucket(m)
        except (ValueError, TypeError, KeyError):
            continue
        if kind != "mid":
            continue
        if m.yes_ask is None:
            continue
        mids.append((m, kind, int(thr), float(m.yes_ask)))

    if not mids:
        counts["skipped_no_mid"] += 1
        return dict(counts)

    # Pick the highest yes_ask
    mids.sort(key=lambda x: -x[3])
    leader_m, leader_kind, leader_thr, leader_ya = mids[0]

    if leader_ya < min_yes_ask:
        counts["skipped_below_band"] += 1
        return dict(counts)
    if leader_ya > max_yes_ask:
        counts["skipped_above_band"] += 1
        return dict(counts)

    if portfolio.is_open(leader_m.yes_token_id, "YES"):
        counts["skipped_already_open"] += 1
        return dict(counts)

    # Submit FAK BUY YES at the 2-decimal rounded ask. Polymarket walks
    # the book; we'd fill at the cheapest available offer ≤ limit.
    # Round UP via the round-to-2-decimals so the engine matches the
    # current ask (avoid rounding down and missing the fill).
    import math
    submitted_limit = math.ceil(leader_ya * 100) / 100.0
    # Clamp to max_yes_ask just in case rounding pushes above the cap.
    submitted_limit = min(submitted_limit, round(max_yes_ask, 2))
    shares = float(max(1, int(size_usd / submitted_limit)))

    signal = TradeSignal(
        station=station, event_title="", event_slug="",
        target=target,
        target_date=datetime.fromisoformat(target_date_iso).date(),
        bucket_label=leader_m.bucket_label, bucket_kind=leader_kind,
        market_id=int(leader_m.market_id), token_id=leader_m.yes_token_id,
        our_prob=leader_ya, yes_implied=leader_ya,
        yes_bid=leader_m.yes_bid, yes_ask=leader_m.yes_ask,
        side="YES", edge=0.0,
        fill_price=leader_ya, volume_24hr=0.0,
        bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
        kelly_full=1.0, position_usd=size_usd,
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
            "target_date": target_date_iso,
            "bucket_label": leader_m.bucket_label,
            "yes_ask_snapshot": leader_ya, "exc": str(exc)[:200],
        }, log_path)
        return dict(counts)
    if not result.ok:
        counts["submit_failed"] += 1
        _log_event({
            "ts_utc": _now_utc_iso(), "result": "rejected",
            "station_id": station_id, "target": target,
            "target_date": target_date_iso,
            "bucket_label": leader_m.bucket_label,
            "yes_ask_snapshot": leader_ya,
            "message": (result.message or "")[:200],
        }, log_path)
        return dict(counts)

    position = Position(
        token_id=leader_m.yes_token_id, side="YES",
        station_id=station_id, region=region_for(station_id),
        market_id=int(leader_m.market_id),
        bucket_label=leader_m.bucket_label, bucket_kind=leader_kind,
        bucket_threshold=leader_thr,
        target_date=target_date_iso,
        shares=shares, entry_price=leader_ya, position_usd=size_usd,
        submitted_at=_now_utc_iso(), status="filled",
        order_id=result.order_id, strategy="consensus_yes",
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
        "target_date": target_date_iso,
        "bucket_label": leader_m.bucket_label,
        "bucket_kind": leader_kind, "bucket_threshold": leader_thr,
        "yes_ask_snapshot": leader_ya,
        "submitted_limit": submitted_limit,
        "fill_price": result.fill_price,
        "shares": shares, "size_usd": size_usd,
        "order_id": result.order_id,
        "n_mids_in_event": len(mids),
        # Rank-2 yes_ask to gauge gap to second-place (signal-strength proxy)
        "second_yes_ask": mids[1][3] if len(mids) > 1 else None,
    }, log_path)
    if verbose:
        print(f"  [cons-yes] {station_id}/{target} {target_date_iso} "
              f"BUY YES {leader_m.bucket_label} @ {leader_ya:.3f} "
              f"(limit ${submitted_limit:.2f})")

    return dict(counts)
