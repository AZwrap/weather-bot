"""Basket threshold sweep — SHADOW logger for trigger-level tuning.

Operator spec (2026-05-31): instead of firing the basket once at a fixed
$0.85, log a *shadow* basket at EVERY penny the leading bucket reaches
from $0.70 to $0.99 — the winner-YES leg AND all the associated NO legs
— with depth-walked fills and Polymarket taker fees applied. After N
days, analyze_basket_sweep.py turns this into a P&L-vs-entry-threshold
curve, OVERALL and PER-STATION, so each station's optimal trigger can be
tuned independently.

This places NO orders and creates NO portfolio positions — it is pure
observation. It records, each time an event's leading-bucket YES ask
first reaches a new high-water penny T in [70, 99], a snapshot of:

  winner : the current max-YES bucket — what a YES $5 entry would cost
           (depth-walked, fee-applied) if you triggered at T.
  fade_no: every OTHER bucket — what a NO $5 entry would cost on each.

Honesty note on granularity: prices are refreshed every ~5 min, so on a
fast move the leader can jump several pennies between samples. We log
ONE snapshot at the penny actually OBSERVED (the high-water level) with
the real price at that moment, and bin it to that penny. We do NOT
fabricate prices for skipped intermediate pennies — that would corrupt
the per-threshold P&L the whole exercise exists to measure. Across many
events and N days, every penny level fills in with real observations.

Output:
  data/basket_sweep_log.jsonl   — one row per (event, observed-penny) snapshot
  data/basket_sweep_state.json  — per-event highest penny logged so far
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .polymarket import (
    event_target_date,
    fetch_orderbook_depths_batch,
    match_event_to_station,
    parse_bucket,
    simulate_buy_fill,
)

DEFAULT_LOG_PATH = Path("data/basket_sweep_log.jsonl")
DEFAULT_STATE_PATH = Path("data/basket_sweep_state.json")

LOW_PENNY: int = 70    # start logging once the leader's YES reaches $0.70
HIGH_PENNY: int = 99   # stop after $0.99
SIZE_USD: float = 5.0  # per-leg notional (winner YES + each fade NO)
FEE_RATE: float = 0.05  # Polymarket weather/culture taker rate (fees.py)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _taker_fee(shares: float, price: float) -> float:
    if not (0.0 < price < 1.0) or shares <= 0:
        return 0.0
    return shares * FEE_RATE * price * (1.0 - price)


def _load_state(path: Path) -> dict[str, int]:
    try:
        return {str(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _save_state(state: dict[str, int], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def _log(record: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _leg_snapshot(*, m, side: str, depth, observed_ask: float | None,
                  size_usd: float) -> dict | None:
    """Depth-walked $size_usd entry snapshot for one leg (YES or NO).
    Returns a dict with fill_price/shares/fee/net_cost, or None if no book."""
    try:
        kind, thr = parse_bucket(m)
    except (ValueError, TypeError, KeyError):
        return None
    # Walk cheapest-first to $size_usd; 0.99 hard cap (don't cross above it).
    sim = simulate_buy_fill(depth, size_usd, 0.99)
    if sim is not None:
        fill_avg, shares, fully = sim
        depth_source = "depth_walk"
    elif observed_ask is not None and 0.0 < observed_ask <= 0.99:
        # No depth (dead/flaky /book) → top-of-book single-fill estimate.
        fill_avg = float(observed_ask)
        shares = float(max(1, int(size_usd / max(observed_ask, 0.01))))
        fully = True
        depth_source = "top_of_book_fallback"
    else:
        return None
    fee = _taker_fee(shares, fill_avg)
    return {
        "side": side,
        "bucket_label": m.bucket_label,
        "bucket_kind": kind,
        "bucket_threshold": int(thr),
        "observed_ask": observed_ask,
        "fill_price": fill_avg,
        "shares": shares,
        "size_usd": fill_avg * shares,
        "fee": fee,
        "net_cost": fill_avg * shares + fee,
        "fully_filled": fully,
        "depth_source": depth_source,
    }


async def log_basket_sweep(
    *,
    events: list,
    http,
    low: int = LOW_PENNY,
    high: int = HIGH_PENNY,
    size_usd: float = SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """For each event, if the leading bucket's YES ask has reached a new
    high-water penny in [low, high], log a shadow basket at that level."""
    counts: dict[str, int] = defaultdict(int)
    today_utc = datetime.now(timezone.utc).date()
    state = _load_state(state_path)
    dirty = False

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
        key = f"{station.station_id}|{target}|{td_iso}"

        # Leading bucket = highest YES ask.
        leader = None
        leader_ya = 0.0
        for m in ev.markets:
            if m.yes_ask is None:
                continue
            if m.yes_ask > leader_ya:
                leader = m
                leader_ya = float(m.yes_ask)
        if leader is None or not leader.yes_token_id:
            continue

        hw_now = min(int(math.floor(leader_ya * 100 + 1e-9)), high)
        if hw_now < low:
            counts["below_range"] += 1
            continue
        prev = state.get(key, low - 1)
        if hw_now <= prev:
            counts["no_new_level"] += 1
            continue

        # New high-water penny reached — fetch depth and snapshot the basket.
        tokens = [leader.yes_token_id]
        for m in ev.markets:
            if m is not leader and m.no_token_id:
                tokens.append(m.no_token_id)
        try:
            depth_map = await fetch_orderbook_depths_batch(tokens, http)
        except Exception:
            depth_map = {}

        winner = _leg_snapshot(
            m=leader, side="YES", depth=depth_map.get(leader.yes_token_id),
            observed_ask=leader_ya, size_usd=size_usd,
        )
        fade_no = []
        for m in ev.markets:
            if m is leader or not m.no_token_id:
                continue
            no_ask = (1.0 - float(m.yes_bid)) if m.yes_bid is not None else None
            snap = _leg_snapshot(
                m=m, side="NO", depth=depth_map.get(m.no_token_id),
                observed_ask=no_ask, size_usd=size_usd,
            )
            if snap is not None:
                fade_no.append(snap)

        total_net_cost = (winner["net_cost"] if winner else 0.0) + sum(
            l["net_cost"] for l in fade_no)
        _log({
            "ts_utc": _now(),
            "station_id": station.station_id,
            "target": target,
            "target_date": td_iso,
            "entry_threshold": hw_now,          # penny bin actually observed
            "prev_high_water": prev,            # last penny logged (skip-gap audit)
            "leader_yes_ask": leader_ya,
            "winner": winner,                   # None only if winner book vanished
            "fade_no": fade_no,
            "n_no_legs": len(fade_no),
            "total_net_cost": total_net_cost,
        }, log_path)

        state[key] = hw_now
        dirty = True
        counts["snapshots"] += 1
        counts["levels_advanced"] += (hw_now - prev)
        if verbose:
            print(f"  [sweep-log] {key} reached {hw_now}¢ "
                  f"(winner {leader.bucket_label}, {len(fade_no)} NO legs)")

    if dirty:
        _save_state(state, state_path)
    return dict(counts)
