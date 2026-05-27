"""Live execution helpers — bridges the scanner/intraday decision layer
to the ExecutionClient + Portfolio.

Currently exposes one function: `submit_metar_live()`, called by
`intraday_scan.py` immediately after a METAR BUY (or BUY_EARLY_TAIL)
decision is logged to the paper-trade audit trail. The paper log
keeps recording every decision; this function only ADDS the live
submission + portfolio bookkeeping when --live is enabled.

Phase 1 scope (METAR-only, per project_live_deployment_roadmap.md):
  - YES-side taker FAK on the winning bucket
  - Per-event / per-region / portfolio cap checks
  - Dedupe via Portfolio.should_skip
  - Persists Position to portfolio.json on success

NOT covered yet:
  - NO_momentum live submission (Phase 2)
  - Cross-up cancel (Phase 2)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .alerts import record_alert
from .exclusions import load_active_exclusions
from .locations import STATIONS_BY_ID
from .portfolio import (
    DEFAULT_PORTFOLIO_PATH,
    Portfolio,
    Position,
    region_for,
)
from .scanner import TradeSignal


def submit_metar_live(
    *,
    sid: str,
    target: str,
    target_date,  # datetime.date
    winning_market,  # PolymarketMarket
    win_kind: str,
    win_thr: int,
    ask: float,
    bid: float | None,
    size_usd: float,
    shares: float,
    ev,  # PolymarketEvent
    decision_kind: str,  # "BUY" or "BUY_EARLY_TAIL"
    client,  # ExecutionClient
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
) -> bool:
    """Submit a live METAR order and persist on success.

    Returns True if the order was submitted and accepted by Polymarket,
    False on any gate failure or submission error. Idempotent against
    portfolio.json — if a position on this (token, YES) already exists,
    skips without re-submitting.
    """
    token_id = winning_market.yes_token_id
    if not token_id:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: no yes_token_id")
        return False

    # SANITY GATE (Fix 2026-05-21): market-disagreement protection.
    # METAR_peak/early_tail fires when WE observe peak landing in/past
    # the bucket. We bet bucket is the winner via YES at the depth-walked
    # price up to $0.85-$0.95 ceiling. If market YES_ask is suspiciously
    # LOW (e.g., $0.02), market thinks bucket is NOT the winner --
    # disagreement is so wide that OUR observation is likely wrong
    # (sensor glitch, unit confusion, oracle source disagreement, stale
    # data). Refuse rather than depth-walk into a near-certain loss.
    #
    # Same pattern as Layer 7's MIN_NO_ASK_SANITY + live-arb's
    # MIN_ALIVE_YES_SUM. Audit pass 2026-05-21 after Layer 7 incident.
    MIN_YES_ASK_SANITY = 0.10
    if ask is not None and float(ask) < MIN_YES_ASK_SANITY:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: "
              f"YES ask ${float(ask):.4f} < ${MIN_YES_ASK_SANITY:.2f} "
              f"-- market disagrees with 'winning bucket' determination; skipping")
        return False

    # EXCLUDED-STATIONS GUARD (2026-05-22 ZGSZ incident). METAR_peak /
    # early_tail previously did not honor data/excluded_stations.json
    # -- only NO_momentum did. Excluded stations (e.g., ZGSZ Shenzhen
    # due to ASOS feed disagreement with Polymarket oracle) have
    # unreliable observations; a "peak landed in bucket X" call from
    # our data might be off by 1-3°C, putting us on the wrong bucket.
    try:
        active_exclusions = load_active_exclusions(today=datetime.now(timezone.utc).date())
    except Exception:
        active_exclusions = set()
    if (sid, target) in active_exclusions:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: "
              f"station-target excluded")
        return False

    # Dedupe / cooldown / permanent-block gate
    should_skip, skip_reason = portfolio.should_skip(token_id, "YES")
    if should_skip:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: {skip_reason}")
        return False

    # Cap check — scale to current effective bankroll
    bankroll = float(getattr(client.config, "bankroll_usd", 500.0))
    scaled = portfolio.scaled_caps(bankroll)
    # market_id: prefer the Polymarket market identifier (int) when available
    raw_mid = getattr(winning_market, "condition_id", None) or getattr(ev, "event_id", None) or 0
    try:
        market_id = int(raw_mid) if isinstance(raw_mid, (int, str)) and str(raw_mid).isdigit() else hash(str(raw_mid)) & 0x7FFFFFFF
    except (ValueError, TypeError):
        market_id = 0

    exceed, exceed_reason = portfolio.would_exceed_cap(
        station_id=sid,
        market_id=market_id,
        position_usd=size_usd,
        portfolio_cap_usd=scaled["portfolio_cap"],
        per_region_cap_usd=scaled["per_region_cap"],
        per_event_cap_usd=scaled["per_event_cap"],
    )
    if exceed:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: {exceed_reason}")
        return False

    # HARD CIRCUIT BREAKER: cumulative daily deployment cap.
    # Caps the day's max possible loss at this value (= every fire loses).
    # Default $150 → guaranteed max -$150/day on $500 bankroll.
    daily_limit = float(getattr(client.config, "daily_deployment_limit_usd", 150.0))
    today_deployed = portfolio.today_deployed_usd()
    if today_deployed + size_usd > daily_limit:
        print(f"    [skip-live] {sid} {winning_market.bucket_label}: "
              f"daily deployment cap ${daily_limit:.0f} would be exceeded "
              f"(today: ${today_deployed:.2f} + new ${size_usd:.2f} = "
              f"${today_deployed + size_usd:.2f}). Existing positions hold "
              f"to resolution; new submissions halted for the UTC day.")
        return False

    # Build minimal TradeSignal — only the fields submit_order reads are
    # meaningfully set; ensemble/bias fields are zeroed for METAR fires
    # (no model prediction involved; we OBSERVED the winning bucket).
    station = STATIONS_BY_ID[sid]
    signal = TradeSignal(
        station=station,
        event_title=getattr(ev, "title", "") or "",
        event_slug=getattr(ev, "slug", "") or "",
        target=target,
        target_date=target_date,
        bucket_label=winning_market.bucket_label,
        bucket_kind=win_kind,
        market_id=market_id,
        token_id=token_id,
        our_prob=1.0,  # observed → win locked in
        yes_implied=float(ask),
        yes_bid=bid,
        yes_ask=ask,
        side="YES",
        edge=1.0 - float(ask),
        fill_price=float(ask),
        volume_24hr=float(getattr(ev, "volume_24hr", 0.0) or 0.0),
        bias_applied_c=0.0,
        sigma_ensemble_c=0.0,
        sigma_total_c=0.0,
        kelly_full=1.0,
        position_usd=size_usd,
    )

    # Submit (taker FAK on YES side, depth-walking enabled)
    #
    # Depth-walking: we submit with limit = client.config.metar_max_ask
    # ($0.95 default) NOT at top_ask. Polymarket's matching engine fills
    # the cheapest asks first up to our limit, so this naturally captures
    # the backtest's depth-aware-metar fills. Verified empirically via
    # the 2026-05-14 smoke order: GTC BUY limit $0.01 filled 500 shares
    # at $0.0019 average (Polymarket walks the book).
    ceiling = float(getattr(client.config, "metar_max_ask", 0.92))
    print(f"    [LIVE] submitting {sid} {winning_market.bucket_label} "
          f"target {shares:.1f}sh × top_ask ${float(ask):.3f} = ${size_usd:.2f}, "
          f"FAK BUY YES @ limit ${ceiling:.3f} (depth-walk)")
    result = client.submit_order(
        signal, order_type="FAK", sdk_side="BUY",
        limit_price=ceiling,
    )

    if not result.ok:
        print(f"      ✗ submit failed: {result.message}")
        return False

    # Capture ACTUAL fill data from the response (separate from the
    # submitted limit price). On filled FAK orders, result.fill_price =
    # avg fill (taking/making), result.shares = filled shares. If status
    # is delayed/partial, we may not have these — falls back to limit + target.
    actual_fill_price = result.fill_price
    actual_shares = result.shares
    actual_usd = actual_fill_price * actual_shares
    print(f"      ✓ order_id={(result.order_id or '?')[:20]}…  "
          f"avg_fill=${actual_fill_price:.4f}  shares={actual_shares:.2f}  "
          f"usd=${actual_usd:.2f}  limit=${ceiling:.3f}")
    print(f"        {result.message}")

    # Persist position
    strategy_label = (
        "METAR_early_tail" if decision_kind == "BUY_EARLY_TAIL" else "METAR_peak"
    )
    position = Position(
        token_id=token_id,
        side="YES",
        station_id=sid,
        region=region_for(sid),
        market_id=market_id,
        bucket_label=winning_market.bucket_label,
        bucket_kind=win_kind,
        bucket_threshold=win_thr,
        target_date=target_date.isoformat(),
        shares=float(actual_shares),
        entry_price=float(actual_fill_price),
        position_usd=float(actual_usd),
        submitted_at=datetime.now(timezone.utc).isoformat(),
        status="submitted",
        order_id=result.order_id,
        strategy=strategy_label,
    )
    # ORPHAN-ORDER GUARD (2026-05-21 incident): submit_order succeeded
    # so 549+ shares may now be in our Polymarket account. If add/save
    # raises, the bot has the position but doesn't know it -- the same
    # silent-failure pattern that caused the KLGA NO incident. Record a
    # critical alert (persisted + loud) and re-raise so the failure
    # propagates rather than getting swallowed.
    try:
        portfolio.add(position)
        portfolio.save(portfolio_path)
    except Exception as exc:
        record_alert(
            kind="orphan_order_save_failed",
            severity="critical",
            summary=(
                f"METAR {decision_kind} {sid} {winning_market.bucket_label}: "
                f"order {result.order_id} placed on Polymarket "
                f"({actual_shares:.2f}sh @ ${actual_fill_price:.4f}) "
                f"but portfolio.save FAILED -- bot does not know it owns these shares"
            ),
            details={
                "strategy": strategy_label,
                "station_id": sid,
                "bucket_label": winning_market.bucket_label,
                "token_id": token_id,
                "order_id": result.order_id,
                "shares": float(actual_shares),
                "entry_price": float(actual_fill_price),
                "position_usd": float(actual_usd),
                "portfolio_path": str(portfolio_path),
                "exception": str(exc),
                "exception_type": type(exc).__name__,
            },
        )
        raise
    return True
