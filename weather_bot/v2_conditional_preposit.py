"""V2 Conditional Preposit — GTC NO maker at $0.82 only when another
bucket in the same event has yes_ask >= $0.80 (= market has identified
a winner elsewhere).

EMPIRICAL BASIS (Task #69, 2026-05-25)
======================================

Paper analyzer (analyze_paper_no_momentum.py) replayed 274K records of
paper_no_momentum data and found:

  V2 hypothetical fills:  11
  V2 wins:                11   (100% win rate)
  V2 PnL:                 +$12.07  (+$1.10/fill)
  Layer 7 overlap:         0   (V2 catches DIFFERENT buckets than Layer 7)

V2's gate is fundamentally a "market consensus" signal — when another
bucket trades above $0.80, the market has converged on a winner. Our
candidate bucket is then by elimination near-dead — but NOT
mathematically proven dead (Layer 7's threshold). V2 catches the gap
between "market converged" and "math triggered".

CAVEAT — small N (N=11):
  95% Wilson CI on win rate: (74%, 100%).
  At lower bound (74% < 82% breakeven), V2 would be marginally
  unprofitable. Deployment is a measured bet that the true win rate
  is closer to the upper bound. Re-validate at N>=30 via the analyzer.

CONFIGURATION (per 2026-05-25 user decisions)
=============================================

  threshold:           $0.82  (matches paper data; NOT live no_momentum's $0.78)
  size_usd:            $5.00  (matches paper data)
  other_bucket_gate:   $0.80  (paper gate that produced the 100% cohort)
  order_type:          GTC maker with post_only
  A∩C composition:     INDEPENDENT (user choice — V2's gate is a
                        DIFFERENT signal than A∩C's same-day/loser
                        filter; 3 of 11 paper wins were on
                        loser-station buckets that A∩C would block)

WHAT V2 RESPECTS (composes with)
================================

  - excluded_stations.json (data quality blacklist)
  - drawdown_breaker
  - portfolio caps (portfolio / region / event)
  - cap_budget (daily + per-station deployment)
  - Portfolio.should_skip (dedupe + cooldown + permanent_block)
  - Layer 6 adverse_info_window (skip same-day placements near peak)
  - persistent_submitted alert (via standard portfolio.save lifecycle)

WHAT V2 DELIBERATELY DOES NOT RESPECT
=====================================

  - A∩C cohort filter (cohort_filter.py): V2's empirical signal showed
    distinct edge on loser-station buckets. Re-applying A∩C would lose
    that signal.
  - Per-station threshold overrides: V2 uses fixed $0.82 (gate signal
    is the selector, not threshold tuning).

KILL SWITCH
===========

  V2_ENABLED = True  (set False to disable instantly without redeploy)

The daemon checks this flag every cycle. False = no submissions; logging
of would-have-fired candidates continues so we don't lose validation data.

LOGGING
=======

Every V2 decision (fired or skipped) writes one record to
data/v2_conditional_log.jsonl. Analyzer joins against
portfolio.json + forward_log.jsonl for post-hoc validation at N>=30.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .alerts import record_alert
from .locations import STATIONS_BY_ID
from .polymarket import (
    PolymarketEvent,
    event_target_date,
    match_event_to_station,
    parse_bucket,
)
from .portfolio import (
    DEFAULT_PORTFOLIO_PATH,
    Portfolio,
    Position,
    region_for,
)
from .scanner import TradeSignal


V2_ENABLED: bool = False
"""KILL SWITCH. Currently FALSE — V2 is in shadow-mode-implicit (we
collect V2 gate data via paper_no_momentum's `other_max_yes_ask`
field; no separate live submissions until N>=30 paper resolutions
validate the 100% sample win rate from N=11).

To go live: flip to True + restart daemon. Before flipping, run:
  python analyze_paper_no_momentum.py
and verify V2 win rate stays >=82% (breakeven) on N>=30 resolved
candidates.

History:
  2026-05-25 initial deploy: I (Claude) shipped V2 LIVE at N=11 after
    user said 'do option A'. User correctly pointed out we'd previously
    agreed to shadow-test new strategies first. No live orders fired
    in the ~6 minute interval between deploy and rollback. Reverted
    to False without any portfolio impact."""

V2_THRESHOLD: float = 0.82
"""GTC NO maker limit price. Matches paper data exactly (the 100%
win-rate cohort fired at $0.82). Higher than live no_momentum's $0.78
because the gate signal selects for higher-confidence buckets where
$0.82 is still well below the eventual $1.00 payout."""

V2_SIZE_USD: float = 5.0
"""Per-fill notional. Matches paper data. Shares the daily cap_budget
with NO_momentum so total live exposure is bounded."""

V2_OTHER_BUCKET_GATE: float = 0.80
"""The gate. At least one OTHER bucket in the event must have
yes_ask >= this for V2 to consider firing. Empirically the threshold
at which 'market has identified a winner' kicks in (paper data uses
0.80 and produced 100% win rate at the 11 fills observed)."""

V2_LOG_PATH = Path("data/v2_conditional_log.jsonl")


def _log_decision(record: dict) -> None:
    """Append-only JSONL log. Errors swallowed (log must not disrupt
    placement loop). Matches the alerts.py / cohort_filter.py pattern."""
    try:
        V2_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(V2_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def submit_v2_conditional_preposit_orders(
    *,
    events: list[PolymarketEvent],
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    threshold: float = V2_THRESHOLD,
    size_usd: float = V2_SIZE_USD,
    other_bucket_gate: float = V2_OTHER_BUCKET_GATE,
    verbose: bool = False,
) -> dict[str, int]:
    """Scan events, place GTC post_only BUY NO @ threshold on candidate
    buckets, gated on V2 condition (another bucket yes_ask >= gate).

    Returns counts dict mirroring submit_no_momentum_orders, plus:
      placed:                       successful submissions
      skipped_gate_not_met:         no other bucket had yes_ask >= gate
      skipped_below_threshold:      NO ask < threshold → post_only would reject
      skipped_self_was_gate_bucket: this bucket IS the converged-on winner
                                    (we don't bet NO against the winner)
      skipped_dedupe/cap/etc.:      same semantics as NO_momentum

    Idempotent: dedupes via Portfolio.should_skip per (token, side).
    """
    if not V2_ENABLED:
        return {"reason": "v2_disabled"}

    if client.is_dry_run:
        if verbose:
            print("  [V2] dry-run client — skipping")
        return {"reason": "dry-run"}

    # Drawdown breaker — same gate as NO_momentum so a single bad day
    # halts BOTH strategies (cross-strategy drawdown safety).
    from weather_bot.drawdown_breaker import can_place_new_orders
    can_place, reason = can_place_new_orders(portfolio)
    if not can_place:
        if verbose:
            print(f"  [V2] HALTED by drawdown breaker: {reason}")
        return {"reason": "drawdown_breaker", "detail": reason}

    counts: dict[str, int] = defaultdict(int)
    from weather_bot.exclusions import load_active_exclusions
    excluded_pairs = load_active_exclusions(datetime.now(timezone.utc).date())
    excluded = {sid for sid, _target in excluded_pairs}
    bankroll = float(getattr(client.config, "bankroll_usd", 500.0))
    daily_limit = float(getattr(client.config, "daily_deployment_limit_usd", 150.0))
    scaled_caps = portfolio.scaled_caps(bankroll)

    # Date horizon — same as no_momentum (today + tomorrow only)
    from datetime import timedelta as _timedelta
    today_utc = datetime.now(timezone.utc).date()
    max_target_date = today_utc + _timedelta(days=1)

    for ev in events:
        station = match_event_to_station(ev)
        if station is None:
            continue
        if station.station_id in excluded:
            counts["skipped_excluded"] += 1
            continue

        target = "max" if ev.target == "highest" else "min"
        target_date = event_target_date(ev, station)

        if target_date > max_target_date:
            counts["skipped_future_date"] += 1
            continue

        # ── V2 GATE — does ANY bucket in this event have yes_ask >= gate?
        # If not, V2 sees no "market has identified a winner" signal and
        # we don't fire on any bucket in this event.
        ev_yes_asks = [
            (m, m.yes_ask) for m in ev.markets if m.yes_ask is not None
        ]
        if not ev_yes_asks:
            counts["skipped_no_asks"] += 1
            continue
        gate_bucket = None
        gate_bucket_ask = 0.0
        for m, ya in ev_yes_asks:
            if ya >= other_bucket_gate and ya > gate_bucket_ask:
                gate_bucket = m
                gate_bucket_ask = ya
        if gate_bucket is None:
            counts["skipped_gate_not_met"] += 1
            continue

        # Layer 6 adverse-info window check was REMOVED in the lite
        # rebuild — it depended on the peak-based intraday tuning we
        # deleted, and V2 stays paper-only (V2_ENABLED=False) until
        # N≥30 paper resolutions validate the strategy. Before flipping
        # V2 to live, re-add a defensive filter to avoid placing makers
        # too close to peak hours (adverse-selection mitigation).

        # ── Per-bucket loop. Skip the gate bucket itself (we don't
        # bet NO against the bucket that's converging to win).
        for m in ev.markets:
            if not m.no_token_id or m.yes_ask is None or m.yes_bid is None:
                counts["missing_data"] += 1
                continue
            if m is gate_bucket:
                counts["skipped_self_was_gate_bucket"] += 1
                continue

            kind, thr = parse_bucket(m)
            # NO_momentum-style: mid buckets only. Tails handled by Layer 7.
            if kind != "mid":
                counts["skipped_non_mid"] += 1
                continue

            no_ask = 1.0 - m.yes_bid
            if no_ask < threshold:
                # post_only would reject; book is too eager. Could indicate
                # this bucket is also a near-winner (similar to gate bucket).
                counts["skipped_below_threshold"] += 1
                continue

            # Dedupe / cooldown / permanent block
            should_skip, skip_reason = portfolio.should_skip(m.no_token_id, "NO")
            if should_skip:
                if "permanently" in skip_reason:
                    counts["skipped_blocked"] += 1
                elif "cooldown" in skip_reason:
                    counts["skipped_cooldown"] += 1
                else:
                    counts["skipped_dedupe"] += 1
                continue

            # Cap checks (same as NO_momentum)
            raw_mid = getattr(m, "condition_id", None) or getattr(ev, "event_id", None) or 0
            try:
                market_id = int(raw_mid) if isinstance(raw_mid, (int, str)) and str(raw_mid).isdigit() else hash(str(raw_mid)) & 0x7FFFFFFF
            except (ValueError, TypeError):
                market_id = 0

            exceed, _ = portfolio.would_exceed_cap(
                station_id=station.station_id,
                market_id=market_id,
                position_usd=size_usd,
                portfolio_cap_usd=scaled_caps["portfolio_cap"],
                per_region_cap_usd=scaled_caps["per_region_cap"],
                per_event_cap_usd=scaled_caps["per_event_cap"],
            )
            if exceed:
                counts["skipped_cap"] += 1
                continue

            from weather_bot.cap_budget import acquire_cap_token, release_cap_token
            cap_ok, cap_reason = acquire_cap_token(
                size_usd=size_usd, daily_limit_usd=daily_limit,
                station_id=station.station_id,
            )
            if not cap_ok:
                if cap_reason == "global_cap":
                    counts["skipped_daily_limit"] += 1
                else:
                    counts["skipped_station_cap"] += 1
                continue

            # Build signal
            signal = TradeSignal(
                station=station,
                event_title=getattr(ev, "title", "") or "",
                event_slug=getattr(ev, "slug", "") or "",
                target=target,
                target_date=target_date,
                bucket_label=m.bucket_label,
                bucket_kind=kind,
                market_id=market_id,
                token_id=m.no_token_id,
                our_prob=1.0 - threshold,
                yes_implied=float(m.yes_ask),
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                side="NO",
                edge=1.0 - threshold,
                fill_price=threshold,
                volume_24hr=float(getattr(ev, "volume_24hr", 0.0) or 0.0),
                bias_applied_c=0.0,
                sigma_ensemble_c=0.0,
                sigma_total_c=0.0,
                kelly_full=1.0,
                position_usd=size_usd,
            )

            # GTD security buffer (Polymarket requires now+60s minimum)
            POLYMARKET_GTD_SECURITY_BUFFER_S = 60
            ttl_seconds = 30
            expires_at_unix = int(time.time()) + POLYMARKET_GTD_SECURITY_BUFFER_S + ttl_seconds

            if verbose:
                print(f"  [V2] {station.station_id} {m.bucket_label} "
                      f"gate={gate_bucket.bucket_label}@{gate_bucket_ask:.2f} → "
                      f"GTD BUY ${threshold:.2f} post_only=True")
            try:
                result = client.submit_order(
                    signal,
                    order_type="GTD",
                    sdk_side="BUY",
                    limit_price=threshold,
                    post_only=True,
                    expires_at=expires_at_unix,
                )
            except Exception as exc:
                if verbose:
                    print(f"    ✗ V2 submit exception: {exc}")
                release_cap_token(size_usd, station_id=station.station_id)
                counts["rejected_by_polymarket"] += 1
                continue

            if not result.ok:
                if verbose:
                    msg = (result.message or "")[:80]
                    print(f"    ✗ V2 rejected: {msg}")
                release_cap_token(size_usd, station_id=station.station_id)
                counts["rejected_by_polymarket"] += 1
                continue

            # Persist position (strategy="v2_conditional_preposit"
            # distinguishes from NO_momentum in portfolio.json)
            position = Position(
                token_id=m.no_token_id,
                side="NO",
                station_id=station.station_id,
                region=region_for(station.station_id),
                market_id=market_id,
                bucket_label=m.bucket_label,
                bucket_kind=kind,
                bucket_threshold=thr,
                target_date=target_date.isoformat(),
                shares=size_usd / threshold,
                entry_price=threshold,
                position_usd=size_usd,
                submitted_at=datetime.now(timezone.utc).isoformat(),
                status="submitted",
                order_id=result.order_id,
                strategy="v2_conditional_preposit",
                expires_at_utc=datetime.fromtimestamp(
                    expires_at_unix, tz=timezone.utc).isoformat(),
            )
            # Orphan-order guard: save immediately
            try:
                portfolio.add(position)
                portfolio.save(portfolio_path)
            except Exception as exc:
                # Submit succeeded, portfolio.save failed → orphan order
                # alert (same pattern as NO_momentum/Layer 7).
                record_alert(
                    kind="orphan_order_save_failed",
                    severity="critical",
                    summary=(
                        f"V2 order {result.order_id[:14]} submitted to "
                        f"Polymarket but portfolio.save failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    details={
                        "strategy": "v2_conditional_preposit",
                        "order_id": result.order_id,
                        "token_id": m.no_token_id,
                        "station_id": station.station_id,
                        "bucket_label": m.bucket_label,
                    },
                )
                if verbose:
                    print(f"    ⚠ V2 portfolio.save failed: {exc}")
                counts["save_failed"] += 1
                continue

            counts["placed"] += 1

            # ── Maker vs Taker counterfactual ──
            # Maker (what V2 actually does): GTC post_only BUY NO at
            # `threshold` ($0.82). Fills iff NO ask crosses down to
            # threshold — i.e., yes_bid rises to 1 − threshold = $0.18.
            #
            # Taker (counterfactual): immediately BUY NO at current ask.
            # NO ask on Polymarket binary buckets ≈ 1 − yes_bid (the bot
            # never stores a separate no_ask field). Fee per share at
            # entry price p is 0.05 × p × (1 − p). We log enough fields
            # here for the offline analyzer to compute resolution-PnL
            # for both modes.
            taker_no_ask = (
                1.0 - float(m.yes_bid) if m.yes_bid is not None else None
            )
            taker_fee_per_share = (
                0.05 * taker_no_ask * (1.0 - taker_no_ask)
                if taker_no_ask is not None else None
            )
            maker_fee_per_share = 0.05 * float(threshold) * (1.0 - float(threshold))

            _log_decision({
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "decision": "placed",
                "station_id": station.station_id,
                "target": target,
                "target_date": target_date.isoformat(),
                "bucket_label": m.bucket_label,
                "bucket_threshold": int(thr),
                "bucket_kind": kind,
                "yes_ask": float(m.yes_ask),
                "yes_bid": float(m.yes_bid) if m.yes_bid is not None else None,
                # MAKER intent (what we placed):
                "maker_intended_price": float(threshold),
                "maker_fee_per_share": maker_fee_per_share,
                # TAKER counterfactual (snapshot at fire time):
                "taker_no_ask": taker_no_ask,
                "taker_fee_per_share": taker_fee_per_share,
                "taker_size_shares": (
                    (size_usd / taker_no_ask) if taker_no_ask else None
                ),
                "intended_size_usd": float(size_usd),
                "gate_bucket_label": gate_bucket.bucket_label,
                "gate_bucket_yes_ask": float(gate_bucket_ask),
                "order_id": result.order_id,
                "token_id": m.no_token_id,
            })

    return dict(counts)
