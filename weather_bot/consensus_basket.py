"""Consensus basket — back the emerged winner, fade the rest, HOLD.

Operator spec (2026-05-31): the day's temperature firms up and one
bucket's YES climbs out of the pack. When a bucket's YES ask reaches
the trigger (default $0.85), treat it as the emerged winner and place a
whole-event basket, then HOLD every leg to resolution (no early exit):

  - BUY YES on the winner bucket, $5 (depth-aware FAK taker — walk the
    ask ladder, scale shares to deploy ~$5, partial/drop per the
    5-share floor).
  - BUY NO on EVERY OTHER bucket in the event, $5 each (same depth-aware
    FAK).

At resolution: if the winner is right, the YES leg pays $1/sh AND every
NO-on-a-loser leg pays $1/sh. If a different bucket wins, the YES leg
and the NO-on-the-true-winner leg lose, the other NO legs pay.

This is a hold-to-resolution basket — the OPPOSITE of consensus_yes
(which sold early and bled the round-trip spread). No exit logic here;
the daily resolver fills actuals and analyze_consensus_basket.py scores
it. Paper-only (dry-run client) like everything else.

One basket per (station, target, date); deduped via portfolio.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .polymarket import (
    event_target_date,
    fetch_orderbook_depths_batch,
    match_event_to_station,
    parse_bucket,
    simulate_buy_fill,
)
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position, region_for
from .scanner import TradeSignal

DEFAULT_LOG_PATH = Path("data/consensus_basket_log.jsonl")
ATTEMPTED_PATH = Path("data/consensus_basket_attempted.json")
"""Persisted set of (station|target|date) keys we've already tried a
basket for. A basket is a once-per-event-per-day shot: when the winner
emerges at $0.85 we build it once. Without this guard, an event whose
legs all fail to fill (dead /book near resolution → 0 positions → the
position-based dedupe can't catch it) would re-trigger every refresh
and re-fetch depth for ~10 dead tokens, risking a Cloudflare rate-limit
ban (the arb playbook warns ≳5 RPS trips 429/1015)."""

TRIGGER_YES: float = 0.85
"""A bucket's YES ask must reach this for it to count as the emerged
winner and trigger the basket."""

SIZE_USD: float = 5.0
"""Per-leg notional (winner YES + each loser NO)."""

YES_SLIPPAGE_TICKS: float = 0.02
"""How far above the winner's current YES ask we'll let the taker walk
(don't chase the price up — 'go up the shares', not up the price)."""

NO_LIMIT_CAP: float = 0.99
"""Cap for the NO legs (NO on losers is cheap-ish; take up to here)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _load_attempted(path: Path = ATTEMPTED_PATH) -> set[str]:
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _save_attempted(keys: set[str], path: Path = ATTEMPTED_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(keys)), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def _already_placed(portfolio: Portfolio, event_tokens: set[str]) -> bool:
    """One basket per event. Position has no `target` field, and a
    station's max and min events share (station_id, target_date) — so we
    dedupe on the event's TOKEN SET, which is disjoint between max/min.
    If we already hold any consensus_basket leg on one of this event's
    tokens, the basket was placed."""
    for p in portfolio.positions:
        if p.strategy == "consensus_basket" and p.token_id in event_tokens:
            return True
    return False


def _place_leg(
    *, m, side: str, limit: float, size_usd: float, depth, station,
    target: str, target_date_iso: str, client, portfolio, portfolio_path,
    log_path: Path,
) -> str:
    """Place one depth-aware FAK leg (YES or NO). Returns outcome string."""
    token_id = m.yes_token_id if side == "YES" else m.no_token_id
    if not token_id:
        return "no_token"
    kind, thr = parse_bucket(m)
    sim = simulate_buy_fill(depth, size_usd, limit)
    if sim is None:
        if depth is not None:
            return "no_depth"
        # no depth fetched → fall back to top-of-book single fill
        top = (float(m.yes_ask) if side == "YES"
               else (1.0 - float(m.yes_bid)) if m.yes_bid is not None else None)
        if top is None or not (0.0 < top <= limit):
            return "no_book"
        fill_avg = top
        shares = float(max(1, int(size_usd / max(top, 0.01))))
        fully = True
        depth_source = "top_of_book_fallback"
    else:
        fill_avg, shares, fully = sim
        depth_source = "depth_walk"

    signal = TradeSignal(
        station=station, event_title="", event_slug="",
        target=target, target_date=datetime.fromisoformat(target_date_iso).date(),
        bucket_label=m.bucket_label, bucket_kind=kind,
        market_id=int(m.market_id), token_id=token_id,
        our_prob=fill_avg, yes_implied=float(m.yes_ask or 0.0),
        yes_bid=m.yes_bid, yes_ask=m.yes_ask,
        side=side, edge=0.0, fill_price=fill_avg, volume_24hr=0.0,
        bias_applied_c=0.0, sigma_ensemble_c=0.0, sigma_total_c=0.0,
        kelly_full=1.0, position_usd=fill_avg * shares,
    )
    try:
        result = client.submit_order(
            signal, order_type="FAK", sdk_side="BUY",
            limit_price=round(limit, 2), override_shares=shares,
        )
    except Exception as exc:
        _log({"ts_utc": _now(), "result": "submit_exception", "side": side,
              "station_id": station.station_id, "bucket_label": m.bucket_label,
              "exc": str(exc)[:200]}, log_path)
        return "submit_failed"
    if not result.ok:
        return "rejected"

    position = Position(
        token_id=token_id, side=side,
        station_id=station.station_id, region=region_for(station.station_id),
        market_id=int(m.market_id), bucket_label=m.bucket_label,
        bucket_kind=kind, bucket_threshold=int(thr),
        target_date=target_date_iso,
        shares=shares, entry_price=fill_avg, position_usd=fill_avg * shares,
        submitted_at=_now(), status="filled",
        order_id=result.order_id, strategy="consensus_basket",
    )
    try:
        portfolio.add(position)
    except Exception:
        pass
    _log({
        "ts_utc": _now(), "result": "filled", "side": side,
        "station_id": station.station_id, "target": target,
        "target_date": target_date_iso, "bucket_label": m.bucket_label,
        "bucket_kind": kind, "bucket_threshold": int(thr),
        "limit": round(limit, 2), "fill_price": fill_avg, "shares": shares,
        "size_usd": fill_avg * shares, "depth_source": depth_source,
        "fully_filled": fully, "order_id": result.order_id,
    }, log_path)
    return "filled"


async def detect_and_execute_consensus_basket(
    *,
    events: list,
    client: Any,
    portfolio: Portfolio,
    http,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    trigger_yes: float = TRIGGER_YES,
    size_usd: float = SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Scan events; when a bucket's YES ≥ trigger, place the hold-to-
    resolution basket (YES winner + NO every other bucket)."""
    counts: dict[str, int] = defaultdict(int)
    today_utc = datetime.now(timezone.utc).date()
    attempted = _load_attempted()
    attempted_dirty = False

    for ev in events:
        station = match_event_to_station(ev)
        if station is None:
            continue
        target = "max" if ev.target == "highest" else "min"
        try:
            td = event_target_date(ev, station)
        except Exception:
            continue
        if td > today_utc + timedelta(days=1):
            continue
        td_iso = td.isoformat()

        # Event token set (for per-event dedupe; disjoint between max/min).
        event_tokens: set[str] = set()
        for m in ev.markets:
            if m.yes_token_id:
                event_tokens.add(m.yes_token_id)
            if m.no_token_id:
                event_tokens.add(m.no_token_id)

        # Emerged winner = highest-YES bucket, if it's reached the trigger.
        winner = None
        winner_ya = 0.0
        for m in ev.markets:
            if m.yes_ask is None:
                continue
            if m.yes_ask > winner_ya:
                winner = m
                winner_ya = float(m.yes_ask)
        if winner is None or winner_ya < trigger_yes:
            counts["skipped_no_winner"] += 1
            continue
        if not winner.yes_token_id:
            counts["skipped_winner_no_token"] += 1
            continue
        attempt_key = f"{station.station_id}|{target}|{td_iso}"
        if attempt_key in attempted or _already_placed(portfolio, event_tokens):
            counts["skipped_already_placed"] += 1
            continue

        # Fetch depth for the legs we'll trade: winner YES + every other NO.
        tokens = [winner.yes_token_id]
        for m in ev.markets:
            if m is not winner and m.no_token_id:
                tokens.append(m.no_token_id)
        try:
            depth_map = await fetch_orderbook_depths_batch(tokens, http)
        except Exception:
            depth_map = {}

        if verbose:
            print(f"  [basket] {station.station_id}/{target} {td_iso}: winner "
                  f"{winner.bucket_label}@{winner_ya:.2f} → placing basket")

        # Winner YES leg (don't chase price up — cap at ask + slippage ticks).
        # 0.98 ceiling: normally we fire on first 0.85 crossing so the ask is
        # ~0.85; the high cap only matters on a cold-start where we first see
        # the event already converged, and we still want the YES leg filled.
        yes_limit = min(round(winner_ya + YES_SLIPPAGE_TICKS, 2), 0.98)
        out = _place_leg(
            m=winner, side="YES", limit=yes_limit, size_usd=size_usd,
            depth=depth_map.get(winner.yes_token_id), station=station,
            target=target, target_date_iso=td_iso, client=client,
            portfolio=portfolio, portfolio_path=portfolio_path, log_path=log_path,
        )
        counts[f"winner_{out}"] += 1
        legs_filled = 1 if out == "filled" else 0

        # NO legs on every other bucket.
        for m in ev.markets:
            if m is winner:
                continue
            out = _place_leg(
                m=m, side="NO", limit=NO_LIMIT_CAP, size_usd=size_usd,
                depth=depth_map.get(m.no_token_id), station=station,
                target=target, target_date_iso=td_iso, client=client,
                portfolio=portfolio, portfolio_path=portfolio_path,
                log_path=log_path,
            )
            counts[f"no_{out}"] += 1
            if out == "filled":
                legs_filled += 1

        # Persist once after the whole basket (legs added in-memory above).
        try:
            portfolio.save(portfolio_path)
        except Exception:
            pass
        # Mark attempted regardless of fill outcome — a basket is a
        # once-per-event shot. Even a 0-fill (dead /book) attempt must
        # not re-trigger every refresh (REST-storm / rate-limit guard).
        attempted.add(attempt_key)
        attempted_dirty = True
        counts["baskets_placed"] += 1
        counts["legs_filled"] += legs_filled
        if verbose:
            print(f"  [basket] {station.station_id}: {legs_filled} legs filled")

    if attempted_dirty:
        _save_attempted(attempted)
    return dict(counts)
