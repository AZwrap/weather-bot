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
from .publication_window import midend_local_utc

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
    # If there's no walkable depth (empty book, dead/failed /book, or less
    # than the 5-share exchange minimum), this leg is NOT tradable right
    # now → return None so it counts as $0 / no position. We NEVER fabricate
    # a top-of-book fill: an empty bucket must count 0, not a $5 trade.
    sim = simulate_buy_fill(depth, size_usd, 0.99)
    if sim is None:
        return None
    fill_avg, shares, fully = sim
    depth_source = "depth_walk"
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


def _ask_ladder(depth, max_levels: int = 8) -> list[list[float]]:
    """Top `max_levels` of a YES ask ladder as [[price, shares], ...]
    (cheapest-first). Lets the analyzer size a matched-share arb against
    the REAL book instead of assuming infinite depth at top-of-book —
    the failure that produced the fake $5000 fill in the first 3-bucket
    test. Empty when there's no book."""
    if depth is None or not depth.asks:
        return []
    return [[round(lv.price, 4), round(lv.size_shares, 2)] for lv in depth.asks[:max_levels]]


def _yes_arb_leg(m, depth, size_usd: float) -> dict | None:
    """One YES leg of the 3-bucket arb (the favorite or a ±1 neighbor):
    best ask, the ask ladder (for depth-aware matched-share sizing), and a
    $size_usd depth-walk for a quick read. None if the bucket can't parse."""
    if m is None:
        return None
    try:
        kind, thr = parse_bucket(m)
    except (ValueError, TypeError, KeyError):
        return None
    best_ask = None
    if depth is not None and depth.asks:
        best_ask = round(depth.asks[0].price, 4)
    elif m.yes_ask is not None:
        best_ask = float(m.yes_ask)
    fill = None
    sim = simulate_buy_fill(depth, size_usd, 0.99)
    if sim is not None:
        fa, sh, fully = sim
        fill = {"fill_price": fa, "shares": round(sh, 2), "fully_filled": fully}
    return {
        "bucket_label": m.bucket_label,
        "bucket_kind": kind,
        "bucket_threshold": int(thr),
        "best_ask": best_ask,
        "ask_ladder": _ask_ladder(depth),
        "fill_5usd": fill,
    }


def _neighbor_markets(ev, leader) -> list:
    """The leader's two threshold-adjacent buckets (fav−1, fav+1) — the
    off-by-one buckets the 3-bucket hedge buys YES on. Found by sorting all
    buckets by threshold and taking the leader's index ±1, so it's
    unit-agnostic (°C 1° steps and °F 2° steps both work)."""
    parsed = []
    for m in ev.markets:
        try:
            _k, t = parse_bucket(m)
        except (ValueError, TypeError, KeyError):
            continue
        parsed.append((int(t), m))
    parsed.sort(key=lambda x: x[0])
    li = next((i for i, (_t, m) in enumerate(parsed) if m is leader), None)
    if li is None:
        return []
    out = []
    for j in (li - 1, li + 1):
        if 0 <= j < len(parsed):
            out.append(parsed[j][1])
    return out


def _fresh_yes_ask(m, book_cache) -> float | None:
    """Leading-bucket trigger price. Prefer the WS cache's best ask
    (refreshed sub-second on every price_change delta); fall back to the
    REST-refreshed snapshot price on the market object."""
    if book_cache is not None and m.yes_token_id:
        a = book_cache.best_ask(m.yes_token_id)
        if a is not None:
            return float(a)
    return float(m.yes_ask) if m.yes_ask is not None else None


async def log_basket_sweep(
    *,
    events: list,
    http,
    book_cache=None,
    leader_state=None,
    extreme_state=None,
    low: int = LOW_PENNY,
    high: int = HIGH_PENNY,
    size_usd: float = SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """For each event, if the leading bucket's YES ask has reached a new
    high-water penny in [low, high], log a shadow basket at that level.

    When `book_cache` is supplied, the leader's trigger price is read from
    the WS top-of-book (sub-second fresh via price_change deltas) instead
    of the 5-min snapshot — this is what lets the WS-delta path catch a
    crossing the instant it happens. Fills still depth-walk REST depth."""
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

        # Leading bucket = highest YES ask (WS-fresh when book_cache given).
        leader = None
        leader_ya = 0.0
        for m in ev.markets:
            ya = _fresh_yes_ask(m, book_cache)
            if ya is None:
                continue
            if ya > leader_ya:
                leader = m
                leader_ya = ya
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

        # Settled / no real live book (YES pinned at ~1.0): nothing left to
        # trade, the /book is dead. Skip the depth fetch and mark done so we
        # stop probing it every refresh (and don't pollute the log with
        # empty rows).
        if leader_ya >= 0.995:
            state[key] = hw_now
            dirty = True
            counts["skipped_settled"] += 1
            continue

        # New high-water penny reached — fetch depth and snapshot the basket.
        # 3-bucket arb: also pull the two threshold-neighbors' YES books so
        # we can measure how often the favorite ±1 YES asks sum < $1 (the
        # arb gate the operator requires — only then is the hedge +EV) and
        # how many shares actually fill at that price.
        neighbors = _neighbor_markets(ev, leader)
        tokens = [leader.yes_token_id]
        for m in ev.markets:
            if m is not leader and m.no_token_id:
                tokens.append(m.no_token_id)
        for nm in neighbors:
            if nm.yes_token_id and nm.yes_token_id not in tokens:
                tokens.append(nm.yes_token_id)
        try:
            depth_map = await fetch_orderbook_depths_batch(tokens, http)
        except Exception:
            depth_map = {}

        winner = _leg_snapshot(
            m=leader, side="YES", depth=depth_map.get(leader.yes_token_id),
            observed_ask=leader_ya, size_usd=size_usd,
        )
        if winner is None:
            # Transient empty winner book (leader_ya < 0.995, so NOT a
            # settled market). Don't advance state or log — retry next
            # refresh when the /book may be back.
            counts["winner_no_book"] += 1
            continue
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

        # Time-gate + stability context, so all three "when to fire" levers
        # (raise-trigger = entry_threshold, time-gate, stability-gate) are
        # calibratable from this one per-crossing row across 0.70–0.99.
        now_dt = datetime.now(timezone.utc)
        local_hour = hours_to_midend = None
        try:
            midend = midend_local_utc(td, station)
            hours_to_midend = round((midend - now_dt).total_seconds() / 3600.0, 2)
            local_hour = round((24.0 - hours_to_midend) % 24.0, 1)  # 0=day start
        except Exception:
            pass
        leader_dwell_s = None   # how long this leader has held the lead
        if leader_state is not None:
            ls = leader_state.get((station.station_id, target, td_iso))
            if ls is not None and len(ls) >= 3 and ls[0] == leader.bucket_label:
                leader_dwell_s = round(now_dt.timestamp() - float(ls[2]), 1)

        # Observed max/min so far (WUG-primary, METAR-fallback) at snapshot
        # time — enables the OBS-CONFIRMED roll replay (analyze_roll_realistic):
        # detect when the observed extreme has passed the held YES bucket so a
        # roll fires only on a genuinely-dead favorite, not a transient market
        # flip. WUG stays the resolution oracle; this is signal-only.
        observed_extreme_c = None
        if extreme_state is not None:
            observed_extreme_c = extreme_state.get((station.station_id, target, td_iso))

        # 3-bucket arb snapshot: leader YES + the two threshold-neighbors'
        # YES, each with its ask ladder. The hedge (buy YES on fav±1 too) is
        # only +EV when these three asks sum < $1 — then you pay <$1 for a
        # basket that wins ~99% (the off-by-one is covered), an empirical
        # arb. The analyzer sizes it depth-aware (equal shares across the 3,
        # bounded by the thinnest book) and scores it on the real winner.
        leader_arb = _yes_arb_leg(leader, depth_map.get(leader.yes_token_id), size_usd)
        neigh_arb = [a for a in (
            _yes_arb_leg(nm, depth_map.get(nm.yes_token_id), size_usd)
            for nm in neighbors) if a is not None]
        best_asks = [a["best_ask"] for a in ([leader_arb] + neigh_arb)
                     if a is not None and a["best_ask"] is not None]
        best_sum = round(sum(best_asks), 4) if len(best_asks) == 3 else None
        yes3_arb = {
            "leader": leader_arb,
            "neighbors": neigh_arb,
            "n_yes_legs": (1 if leader_arb else 0) + len(neigh_arb),
            "best_ask_sum": best_sum,                # arb gate: <1.0 ⇒ +EV
            "arb_ok": (best_sum is not None and best_sum < 1.0),
        }

        _log({
            "ts_utc": now_dt.isoformat(),
            "station_id": station.station_id,
            "target": target,
            "target_date": td_iso,
            "entry_threshold": hw_now,          # penny bin actually observed
            "prev_high_water": prev,            # last penny logged (skip-gap audit)
            "leader_yes_ask": leader_ya,
            "local_hour": local_hour,           # time-gate: station-local hour
            "hours_to_midend": hours_to_midend, # time-gate: hrs to window-end
            "leader_dwell_s": leader_dwell_s,   # stability-gate: leader dwell
            "observed_extreme_c": observed_extreme_c,  # obs max/min so far (roll death-signal)
            "yes3_arb": yes3_arb,               # 3-bucket arb gate (fav±1 YES ladders)
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
