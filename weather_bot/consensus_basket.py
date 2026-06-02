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
"""Persisted per-event state: {f"station|target|date": "locked" | int}.
  - "locked"  → committed (winner YES filled, OR we gave up after
                MAX_NOFILL_ATTEMPTS no-fill tries). Never re-fire.
  - int n     → n consecutive attempts where the winner was ≥0.85 but its
                YES leg could NOT fill (illiquid/uncertain). We did NOT
                place anything and did NOT lock — the event stays eligible
                so a LATER, fillable bucket (the corrected leader) can be
                taken instead of being blocked on the first cross.
We only lock on a FILLED winner (or the cap), so a phantom/unfillable
0.85 cross never burns the event. The cap bounds REST on a genuinely
dead winner (a sticky ≥0.85 quote with no depth) — the Cloudflare guard
(arb playbook: ≳5 RPS trips 429/1015). Backward-compatible with the old
list-of-locked-keys format."""

MAX_NOFILL_ATTEMPTS: int = 10
"""After this many attempts where the winner is ≥0.85 but unfillable,
give up and lock the event (it's genuinely too illiquid). Generous so it
doesn't pre-empt a winner that becomes fillable later; only a transient
quote that REVERTS below 0.85 is skipped earlier (no count), so this only
bites a persistently-stuck dead book."""

TRIGGER_YES: float = 0.82
"""A bucket's YES ask must reach this for it to count as the emerged
winner and trigger the basket. Lowered 0.85→0.82 (2026-06-03): the
per-trigger sweep curve put 0.85 in a NEGATIVE pocket (−$24, N=39) while
0.82 sat in a positive one (+$33 static / +$28 clean). Small-N, PAPER —
this is a live A/B of the better-measured trigger, not a proven edge."""

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


def _load_attempted(path: Path = ATTEMPTED_PATH) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(d, list):          # old format: list of locked keys
            return {k: "locked" for k in d}
        return dict(d)
    except (OSError, ValueError, TypeError):
        return {}


def _save_attempted(state: dict, path: Path = ATTEMPTED_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
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
        # No walkable depth (empty/dead book, failed /book, or under the
        # 5-share exchange minimum). Do NOT fabricate a top-of-book fill —
        # an empty bucket is a no-op ($0), never a $5 trade.
        return "no_depth" if depth is not None else "no_book"
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


def _fresh_yes_ask(m, book_cache) -> float | None:
    """Winner trigger price — WS top-of-book (sub-second) when available,
    else the REST-refreshed snapshot price."""
    if book_cache is not None and m.yes_token_id:
        a = book_cache.best_ask(m.yes_token_id)
        if a is not None:
            return float(a)
    return float(m.yes_ask) if m.yes_ask is not None else None


async def detect_and_execute_consensus_basket(
    *,
    events: list,
    client: Any,
    portfolio: Portfolio,
    http,
    book_cache=None,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    trigger_yes: float = TRIGGER_YES,
    size_usd: float = SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Scan events; when a bucket's YES ≥ trigger, place the hold-to-
    resolution basket (YES winner + NO every other bucket).

    `book_cache`, when given, supplies the WS-fresh trigger price so this
    can fire the instant a delta pushes the winner across `trigger_yes`."""
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

        # Emerged winner = highest-YES bucket, if it's reached the trigger
        # (WS-fresh price when book_cache supplied).
        winner = None
        winner_ya = 0.0
        for m in ev.markets:
            ya = _fresh_yes_ask(m, book_cache)
            if ya is None:
                continue
            if ya > winner_ya:
                winner = m
                winner_ya = ya
        if winner is None or winner_ya < trigger_yes:
            counts["skipped_no_winner"] += 1
            continue
        if not winner.yes_token_id:
            counts["skipped_winner_no_token"] += 1
            continue
        attempt_key = f"{station.station_id}|{target}|{td_iso}"
        if attempted.get(attempt_key) == "locked" or _already_placed(portfolio, event_tokens):
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

        # Winner YES leg FIRST (don't chase price up — cap at ask + slippage
        # ticks; 0.98 ceiling for cold-start already-converged events).
        # We commit the basket (fade the field + lock the event) ONLY if the
        # winner actually fills. A winner we can't fill means the favorite is
        # still illiquid/uncertain — so we place NOTHING and DON'T lock, and
        # the event stays eligible for a LATER fillable bucket (the corrected
        # leader) instead of being permanently blocked on this first cross.
        yes_limit = min(round(winner_ya + YES_SLIPPAGE_TICKS, 2), 0.98)
        wout = _place_leg(
            m=winner, side="YES", limit=yes_limit, size_usd=size_usd,
            depth=depth_map.get(winner.yes_token_id), station=station,
            target=target, target_date_iso=td_iso, client=client,
            portfolio=portfolio, portfolio_path=portfolio_path, log_path=log_path,
        )
        counts[f"winner_{wout}"] += 1
        if wout != "filled":
            # Winner unfillable → place nothing, don't lock. Count the miss;
            # give up only after MAX_NOFILL_ATTEMPTS (dead-book REST backstop).
            n = int(attempted.get(attempt_key) or 0) + 1
            attempted[attempt_key] = "locked" if n >= MAX_NOFILL_ATTEMPTS else n
            attempted_dirty = True
            counts["winner_unfilled"] += 1
            continue
        legs_filled = 1

        # Winner filled → fade every other bucket with NO.
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

        # Persist + LOCK the event now that we've backed a fillable winner.
        try:
            portfolio.save(portfolio_path)
        except Exception:
            pass
        attempted[attempt_key] = "locked"
        attempted_dirty = True
        counts["baskets_placed"] += 1
        counts["legs_filled"] += legs_filled
        if verbose:
            print(f"  [basket] {station.station_id}: {legs_filled} legs filled")

    if attempted_dirty:
        _save_attempted(attempted)
    return dict(counts)
