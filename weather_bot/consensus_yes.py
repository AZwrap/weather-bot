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
DEFAULT_EXIT_LOG_PATH = Path("data/consensus_yes_exit_log.jsonl")

MIN_YES_ASK: float = 0.40
"""Below this, consensus hasn't formed — too many buckets still in
contention. Wait."""

MAX_YES_ASK: float = 0.85
"""Above this we're in convergence-late territory. The bot has lost
the run-up; YES at $0.90+ is the same trade Layer 7 would have done
much cheaper had it fired. Skip."""

DEFAULT_SIZE_USD: float = 5.0

# Exit (trailing-stop) parameters
MIN_PROFIT_FROM_ENTRY: float = 0.05
"""Sell only once current_ask is at least 5pp above entry price.
Below this we hold even on declines (would be a loss after fees)."""

PEAK_DECLINE_TICKS: float = 0.01
"""Minimum decline below the peak (in $) before we treat a drop as
'turning down'. Default $0.01 = one tick of Polymarket's price grid.
Lower = more sensitive (faster exits, less profit per fire).
Higher = need a larger drawdown before triggering."""


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


# ──────────────────────────────────────────────────────────────────────
# Trailing-stop exit logic
# ──────────────────────────────────────────────────────────────────────

def _peak_key(token_id: str, side: str) -> str:
    return f"{token_id}|{side}"


def _log_exit(record: dict, log_path: Path = DEFAULT_EXIT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def evaluate_consensus_yes_exits(
    *,
    events: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    min_profit_from_entry: float = MIN_PROFIT_FROM_ENTRY,
    peak_decline_ticks: float = PEAK_DECLINE_TICKS,
    log_path: Path = DEFAULT_EXIT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Walk all open consensus_yes positions, update peak tracker,
    fire SELL when the trailing-stop trigger fires.

    Trigger:
      current_yes_ask < (peak − peak_decline_ticks)
      AND current_yes_ask >= entry_price + min_profit_from_entry

    Sells at current_yes_bid (= taker hits the bid). All paper-only
    under dry-run client.
    """
    counts: dict[str, int] = defaultdict(int)
    counts["sold"] = 0

    # Build token_id → current (yes_ask, yes_bid) lookup
    quotes: dict[str, tuple[float | None, float | None]] = {}
    for ev in events:
        for m in ev.markets:
            if m.yes_token_id:
                quotes[m.yes_token_id] = (m.yes_ask, m.yes_bid)

    for p in list(portfolio.positions):
        if p.strategy != "consensus_yes":
            continue
        if p.status != "filled":
            continue
        if p.side != "YES":
            continue
        ya, yb = quotes.get(p.token_id, (None, None))
        if ya is None:
            counts["skipped_no_quote"] += 1
            continue

        key = _peak_key(p.token_id, p.side)
        peak = portfolio.consensus_yes_peak_by_pos.get(key, p.entry_price)

        # Update peak when current_ask climbs
        if ya > peak:
            portfolio.consensus_yes_peak_by_pos[key] = float(ya)
            counts["peak_updated"] += 1
            continue

        # Trailing-stop trigger
        decline_below_peak = peak - ya
        profit_above_entry = ya - p.entry_price
        if (decline_below_peak >= peak_decline_ticks
                and profit_above_entry >= min_profit_from_entry):
            # Submit a SELL YES. Use yes_bid as the achievable price
            # (taker hits bid); fall back to ya - 0.01 if bid missing.
            sell_target = yb if yb is not None else max(0.01, ya - 0.01)
            # Submit at limit = sell_target rounded down to 2 decimals
            import math
            limit_sell = max(0.01, math.floor(sell_target * 100) / 100.0)

            signal = TradeSignal(
                station=STATIONS_BY_ID.get(p.station_id) or _stub_station(p),
                event_title="", event_slug="",
                target="max" if "max" in str(p.target_date) else p.bucket_kind,
                target_date=datetime.fromisoformat(p.target_date).date(),
                bucket_label=p.bucket_label, bucket_kind=p.bucket_kind,
                market_id=int(p.market_id), token_id=p.token_id,
                our_prob=ya, yes_implied=ya,
                yes_bid=yb, yes_ask=ya,
                side="YES", edge=0.0,
                fill_price=limit_sell, volume_24hr=0.0,
                bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
                kelly_full=1.0, position_usd=p.position_usd,
            )
            try:
                result = client.submit_order(
                    signal, order_type="FAK", sdk_side="SELL",
                    limit_price=limit_sell, override_shares=p.shares,
                )
            except Exception as exc:
                counts["submit_failed"] += 1
                _log_exit({
                    "ts_utc": _now_utc_iso(), "result": "submit_exception",
                    "station_id": p.station_id, "token_id": p.token_id,
                    "bucket_label": p.bucket_label, "exc": str(exc)[:200],
                }, log_path)
                continue
            if not result.ok:
                counts["submit_failed"] += 1
                _log_exit({
                    "ts_utc": _now_utc_iso(), "result": "rejected",
                    "station_id": p.station_id, "token_id": p.token_id,
                    "bucket_label": p.bucket_label,
                    "message": (result.message or "")[:200],
                }, log_path)
                continue

            # Mark position closed in-memory (no extra dataclass field;
            # we update status). Real portfolio mark_resolved isn't
            # appropriate (this is a sell, not a resolution).
            p.status = "filled_closed"
            try:
                portfolio.save(portfolio_path)
            except Exception:
                pass
            counts["sold"] += 1
            net_per_share = limit_sell - p.entry_price
            _log_exit({
                "ts_utc": _now_utc_iso(), "result": "sold",
                "station_id": p.station_id,
                "target_date": p.target_date,
                "bucket_label": p.bucket_label,
                "bucket_threshold": p.bucket_threshold,
                "entry_price": p.entry_price,
                "peak_yes_ask": peak,
                "current_yes_ask": ya,
                "current_yes_bid": yb,
                "submitted_sell_limit": limit_sell,
                "fill_price": result.fill_price,
                "shares": p.shares,
                "net_per_share_usd": net_per_share,
                "net_total_usd": net_per_share * p.shares,
                "order_id": result.order_id,
            }, log_path)
            if verbose:
                print(f"  [cons-yes-exit] {p.station_id} {p.bucket_label} "
                      f"entry ${p.entry_price:.3f} → peak ${peak:.3f} → "
                      f"sell ${limit_sell:.3f} (net ${net_per_share*p.shares:+.2f})")
        else:
            counts["holding"] += 1

    return dict(counts)


def _stub_station(p: Position):
    """Build a placeholder Station for TradeSignal when STATIONS_BY_ID
    lookup fails (shouldn't happen but defensive)."""
    class _S:
        name = ""
        station_id = p.station_id
        timezone = "UTC"
        unit = "C"
    return _S()
