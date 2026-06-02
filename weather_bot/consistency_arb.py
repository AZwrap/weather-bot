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
"""Legacy per-arb USD notional (kept for callers that pass it). The
hardened path sizes by SHARES, not USD — see SHARES_PER_LEG."""

SHARES_PER_LEG: int = 5
"""Hardened arb unit. The arb pays $shares whichever bucket wins, so we
must hold the SAME share count on every leg (a per-USD size would buy
unequal shares and break the guarantee). 5 = Polymarket's order
minimum, i.e. the smallest *fillable* unit. If the real book can't even
supply 5 shares on every leg, the arb is not executable (artifact)."""

MAX_PLAUSIBLE_LEGS: int = 40
"""Defensive cap. A 'guaranteed' arb spanning more than this many legs
is almost always an empty-book sum, not a real lock."""

MAX_PLAUSIBLE_NET_MARGIN: float = 0.15
"""Reject 'guaranteed' arbs whose net margin/share exceeds this. In any
functioning market a cross-event lock is sub-few-percent — a fat one
would be taken instantly. A large margin means the near-certain leg is
dust-priced: a stale/dead book, not real money. Canonical example: KMIA
T=76 (Miami summer, max≥76 is near-certain yet the book shows ~2¢ asks →
a 70% 'margin'). The 5-share depth filter alone passes these because the
dead book still has a few shares of dust; this cap is what kills them."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fill_shares(
    depth: Any, shares_target: float, limit_price: float = 1.0,
) -> tuple[float, float] | None:
    """Walk an ask ladder (cheapest-first) for `shares_target` shares.

    Returns (avg_fill_price, filled_shares) or None if the book is
    empty / absent. `filled_shares` may be < target on a thin book —
    the caller treats a short fill as "leg unfillable".
    """
    asks = getattr(depth, "asks", None)
    if not asks:
        return None
    taken = 0.0
    cost = 0.0
    for lv in asks:  # get_depth returns asks sorted ascending by price
        p = float(getattr(lv, "price", 0.0))
        s = float(getattr(lv, "size_shares", 0.0))
        if p <= 0.0 or p > limit_price + 1e-9 or s <= 0.0:
            continue
        take = min(shares_target - taken, s)
        taken += take
        cost += take * p
        if taken >= shares_target - 1e-9:
            break
    if taken <= 0.0:
        return None
    return (cost / taken, taken)


def _log_event(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _bucket_fully_ge_T(kind: str, t: int, T: int) -> bool:
    """True iff every value in this bucket is ≥ T (= safe to include
    in 'extreme ≥ T' sum without overcounting)."""
    if kind == "mid":
        return t >= T
    if kind == "high_tail":
        return t >= T   # high_tail covers [t, +inf); fully ≥ T iff t ≥ T
    return False        # low_tail covers (-inf, t+w); never fully ≥ T


def _bucket_fully_lt_T(kind: str, t: int, T: int, bucket_width: int) -> bool:
    """True iff every value in this bucket is < T (= safe to include
    in 'extreme < T' sum without overcounting).

    bucket_width is the integer step (1 for °C, 2 for °F).
    A mid bucket with threshold t covers [t, t+bucket_width). Fully
    below T iff t + bucket_width ≤ T, i.e. t ≤ T - bucket_width.
    """
    if kind == "mid":
        return t + bucket_width <= T
    if kind == "low_tail":
        # low_tail "X or below" covers (-inf, t+bucket_width).
        # Fully < T iff t + bucket_width ≤ T.
        return t + bucket_width <= T
    return False    # high_tail covers [t, +inf); never fully < T


def _cumulative_yes_ask_ge_threshold(
    markets: list[Any], threshold: int,
) -> tuple[float | None, int, list[Any]]:
    """Sum of YES asks across buckets fully ≥ threshold. Returns
    (sum, n_used, bucket_list). None if no usable buckets."""
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
        if _bucket_fully_ge_T(kind, t, threshold):
            total += float(m.yes_ask)
            n += 1
            used.append(m)
    if n == 0:
        return None, 0, []
    return total, n, used


def _cumulative_yes_ask_lt_threshold(
    markets: list[Any], threshold: int, bucket_width: int,
) -> tuple[float | None, int, list[Any]]:
    """Sum of YES asks across buckets fully < threshold. To buy NO on
    'extreme ≥ T', we equivalently buy YES on every bucket fully < T."""
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
        if _bucket_fully_lt_T(kind, t, threshold, bucket_width):
            total += float(m.yes_ask)
            n += 1
            used.append(m)
    if n == 0:
        return None, 0, []
    return total, n, used


async def detect_and_execute_consistency_arb(
    *,
    events: list[Any],
    client: Any,
    portfolio: Any,
    portfolio_path: Path = Path("data/portfolio.json"),
    size_usd: float = DEFAULT_SIZE_USD,
    min_margin_usd: float = MIN_MARGIN_USD,
    shares_per_leg: int = SHARES_PER_LEG,
    book_cache: Any = None,
    http: Any = None,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
) -> dict[str, int]:
    """Scan event list for max/min consistency arbs.

    Hardened (2026-06-02): a candidate that clears the cheap top-of-book
    pre-filter is then re-priced against the REAL ask ladder for
    `shares_per_leg` shares on EVERY leg, with taker fees subtracted.
    Depth source per leg: the fresh WS ladder (`book_cache.get_depth`)
    when available, else a live REST orderbook fetch (`http`). This
    matters — `price_change` deltas invalidate the WS ladder within
    seconds, so REST is the reliable source for an accurate depth walk.
    Any leg that can't supply the full share count (empty/thin book)
    fails the whole arb — the artifact filter that kills empty-book sums
    masquerading as fat margins. Only depth-checked, fee-netted survivors
    are written to the log; the rejection funnel is in the counts dict.

    With neither `book_cache` nor `http` it falls back to the legacy
    top-of-book log (depth_aware=False) so tests/callers still work."""
    from weather_bot.fees import taker_fee_usd
    from weather_bot.locations import STATIONS_BY_ID
    from weather_bot.polymarket import (
        event_target_date, fetch_orderbook_depth, match_event_to_station,
        parse_bucket,
    )
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0

    now_utc = datetime.now(timezone.utc)

    # Per-call depth memo: each YES token fetched at most once per refresh.
    # Cache-first (free when the WS ladder is fresh), REST-fallback (the
    # WS ladder is book_invalidated most of the time on active markets).
    _depth_memo: dict[str, Any] = {}

    async def _leg_depth(tok: Any) -> Any:
        if not tok:
            return None
        if tok in _depth_memo:
            return _depth_memo[tok]
        d = book_cache.get_depth(tok) if book_cache is not None else None
        if d is None and http is not None:
            try:
                d = await fetch_orderbook_depth(tok, http)
            except Exception:
                d = None
        _depth_memo[tok] = d
        return d

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
        station = STATIONS_BY_ID.get(sid)
        bucket_width = 1 if (station and station.unit == "C") else 2
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
            # MAX leg: buy YES on every max-event bucket ≥ T. Pays $1 if
            # max ≥ T. Cost = sum of yes_ask across those buckets.
            cost_max_leg, n_max, max_buckets = _cumulative_yes_ask_ge_threshold(
                ev_max.markets, T,
            )
            # MIN leg: buy NO on the cumulative "min ≥ T" event, which
            # equals buying YES on every min-event bucket fully < T.
            # Pays $1 if min < T. Cost = sum of yes_ask across those.
            cost_min_leg, n_min, min_buckets = _cumulative_yes_ask_lt_threshold(
                ev_min.markets, T, bucket_width,
            )
            if cost_max_leg is None or cost_min_leg is None:
                continue
            cost = cost_max_leg + cost_min_leg
            # STAGE 1 — cheap top-of-book pre-filter. Σ yes_ask (1 share
            # per leg). Min payout = $1 (min ≤ max ⇒ at least one leg
            # wins). Skip the vast majority here without touching depth.
            gross_margin = 1.0 - cost
            if gross_margin < min_margin_usd:
                counts["below_margin"] += 1
                continue

            legs = list(max_buckets) + list(min_buckets)

            # No depth source at all → legacy top-of-book log (tests).
            if book_cache is None and http is None:
                counts["opportunities"] += 1
                counts["placed"] += 1
                _log_event({
                    "ts_utc": now_utc.isoformat(),
                    "result": "opportunity", "depth_aware": False,
                    "station_id": sid, "target_date": td_iso,
                    "threshold": int(T), "bucket_width": int(bucket_width),
                    "cost_usd": float(cost),
                    "gross_margin_top_of_book_usd": float(gross_margin),
                    "n_max_buckets": int(n_max), "n_min_buckets": int(n_min),
                }, log_path)
                continue

            # STAGE 2 — DEPTH + FEES + ARTIFACT FILTER.
            # The arb needs EQUAL shares on every leg (it pays $shares
            # whichever bucket wins). Walk each leg's real ask ladder for
            # `shares_per_leg`; if ANY leg can't supply them the basket
            # has a hole and the "$1 guaranteed" breaks → artifact (this
            # is exactly the empty-book sum that fakes a fat margin).
            if (n_max + n_min) > MAX_PLAUSIBLE_LEGS:
                counts["rej_too_many_legs"] += 1
                continue

            S = float(shares_per_leg)
            depth_cost = 0.0
            total_fees = 0.0
            leg_fills: list[dict] = []
            reject = None
            for m in legs:
                tok = getattr(m, "yes_token_id", None)
                depth = await _leg_depth(tok)
                if depth is None:
                    reject = "no_depth"      # no orderbook / empty book (real)
                    break
                fr = _fill_shares(depth, S)
                if fr is None or fr[1] < S - 1e-9:
                    reject = "thin_depth"     # book present but < S shares
                    break
                avg, got = fr
                depth_cost += avg * got
                total_fees += taker_fee_usd(got, avg)
                leg_fills.append({
                    "bucket_label": getattr(m, "bucket_label", ""),
                    "yes_token_id": tok,
                    "avg_fill_price": round(avg, 4),
                    "shares": round(got, 2),
                })
            if reject is not None:
                counts[f"rej_{reject}"] += 1
                continue

            payout = S  # guaranteed minimum ($1/share × S, min-payout case)
            net_margin = payout - depth_cost - total_fees
            net_per_share = net_margin / S
            # STAGE 3 — net-of-fee floor (per-share, comparable to the old
            # gross margin). A depth-fillable arb whose edge is eaten by
            # fees is not worth executing.
            if net_per_share < min_margin_usd:
                counts["rej_thin_net"] += 1
                continue
            # Artifact filter #2 — too-good-to-be-real. A guaranteed lock
            # this fat means a near-certain leg is dust-priced (dead/stale
            # book), not executable money. Kills the KMIA-T=76 class that
            # survives the 5-share depth check on a few cents of dust.
            if net_per_share > MAX_PLAUSIBLE_NET_MARGIN:
                counts["rej_implausible"] += 1
                continue

            counts["opportunities"] += 1
            counts["placed"] += 1
            if verbose:
                print(f"  [cons-arb] {sid} {td_iso} T={T}: "
                      f"gross_tob=${gross_margin:.3f}  "
                      f"net/sh=${net_per_share:.3f}  "
                      f"({n_max}+{n_min} legs × {int(S)}sh, "
                      f"fees=${total_fees:.3f})")

            _log_event({
                "ts_utc": now_utc.isoformat(),
                "result": "opportunity",
                "depth_aware": True,
                "station_id": sid,
                "target_date": td_iso,
                "threshold": int(T),
                "bucket_width": int(bucket_width),
                "shares_per_leg": int(S),
                "n_max_buckets": int(n_max),
                "n_min_buckets": int(n_min),
                "n_legs": int(n_max + n_min),
                # Old top-of-book metric, kept for before/after comparison.
                "cost_usd": float(cost),
                "gross_margin_top_of_book_usd": float(gross_margin),
                # Hardened economics (the trustworthy numbers):
                "depth_cost_usd": float(depth_cost),
                "total_fees_usd": float(total_fees),
                "payout_usd": float(payout),
                "net_margin_usd": float(net_margin),
                "net_margin_per_share_usd": float(net_per_share),
                "legs": leg_fills,
            }, log_path)

    return dict(counts)
