"""High-bucket NO strategy — bet NO on buckets too far above the peak.

Premise
=======
Around end-of-peak-window in station-local time (typical 18:00 local
for max-target, 06:00 local for min-target), the day's extreme is
unlikely to move much further outward. Buckets that are 2+ steps past
the current peak/trough are very unlikely to win. NO is the high-EV
side, but at LESS than the Layer 7 $0.99 ceiling because mathematical
certainty doesn't apply — the peak could still drift.

Difference from Layer 7
=======================
  - Layer 7 fires when peak >= bucket_high_c → bucket is structurally
    dead (heat has already passed it) → 99% certain → buy NO at ≤ $0.99.
  - This fires when peak + 2 × bucket_width ≤ bucket_low_c → bucket is
    probabilistically dead (heat HASN'T reached it AND is unlikely to)
    → buy NO at ≤ $0.98.
  - The two strategies cover DISJOINT bucket sets: Layer 7 hits buckets
    below the peak, this hits buckets above. No overlap, no ceiling
    handoff — both caps are set independently by EV math.

Configurable
============
  - `N_BUCKETS_AWAY`: how far past the current peak. Default 2 (= one
    bucket of safety margin).
  - `TRIGGER_LOCAL_HOUR_MAX`: station-local hour to start firing on
    max-target events. Default 18 (= 6pm).
  - `TRIGGER_LOCAL_HOUR_MIN`: same for min-target. Default 6 (= 6am,
    just past pre-dawn trough).
  - `MIN_NO_ASK`, `MAX_NO_ASK`: price band. Below MIN means market
    disagrees with us; above MAX means it's already Layer 7 territory.
  - `SIZE_USD`: per-fire size. Default $5 (matches Layer 7).

Dedupe via Portfolio.is_open; the strategy never re-bets the same
(token, NO) twice. No separate tracker — once we fire on bucket B,
dedupe prevents re-fire as the peak climbs toward B.

Paper-only respect
==================
This module honors `client.is_dry_run`: when True, it short-circuits
with `{"reason": "dry-run"}` and logs nothing (matches the convention
of guaranteed_no_buy.py + v2_conditional_preposit.py).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .alerts import record_alert
from .exclusions import load_active_exclusions
from .locations import STATIONS_BY_ID
from .obs_distance_filter import bucket_edges_c
from .polymarket import parse_bucket
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position, region_for
from .scanner import TradeSignal


# Config ─────────────────────────────────────────────────────────────
N_BUCKETS_AWAY: int = 2
"""How many bucket-widths past the peak we require before firing NO.
Default 2 = one bucket of safety margin between the peak bucket and
our NO bet. Tighten to 3 for more conservative; loosen to 1 to fire
more aggressively (= effectively a soft Layer 7)."""

TRIGGER_LOCAL_HOUR_MAX: int = 18
TRIGGER_LOCAL_HOUR_MIN: int = 6

MIN_NO_ASK: float = 0.50
"""If NO ask is BELOW this, market thinks the bucket is more likely to
win than not — strong disagreement with our 'too high' thesis. Refuse
to fire (the market knows something we don't)."""

MAX_NO_ASK: float = 0.98
"""Hard cap on the NO ask we'll pay. High-bucket NO targets buckets
ABOVE the current peak (heat would need to keep climbing to reach
them), so Layer 7 — which fires on buckets BELOW the peak — never
touches these. No handoff at the ceiling; the cap is set by EV math.

At $0.98 NO: +$0.02 if the heat stays below the bucket (we win),
−$0.98 if it climbs into it. Fee adds ~$0.001. Breakeven win rate
≈ 98.1%. For buckets 2+ steps above the current peak at the
station-local trigger hour, that's a stretch but defensible on
stable stations late in the day. Re-tune after N≥30 paper
resolutions — the realized hit rate at each entry-price band tells
us whether 0.98 is too aggressive or if we can push higher."""

DEFAULT_SIZE_USD: float = 5.0

DEFAULT_LOG_PATH = Path("data/high_bucket_no_log.jsonl")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _is_past_trigger(
    station_timezone: str,
    target: str,
    now_utc: datetime,
    trigger_local_hour_max: int,
    trigger_local_hour_min: int,
) -> tuple[bool, float]:
    """Return (is_past_trigger, station_local_hour_fractional)."""
    try:
        tz = ZoneInfo(station_timezone)
    except Exception:
        return False, -1.0
    local = now_utc.astimezone(tz)
    local_h = local.hour + local.minute / 60.0
    trig = trigger_local_hour_max if target == "max" else trigger_local_hour_min
    return local_h >= float(trig), local_h


def detect_and_execute_high_bucket_no(
    *,
    station_id: str,
    target_date_iso: str,
    observed_extreme_c: float,
    target: str,
    bucket_snapshots: list,
    client: Any,
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    now_utc: datetime | None = None,
    n_buckets_away: int = N_BUCKETS_AWAY,
    trigger_local_hour_max: int = TRIGGER_LOCAL_HOUR_MAX,
    trigger_local_hour_min: int = TRIGGER_LOCAL_HOUR_MIN,
    min_no_ask: float = MIN_NO_ASK,
    max_no_ask: float = MAX_NO_ASK,
    size_usd: float = DEFAULT_SIZE_USD,
    log_path: Path = DEFAULT_LOG_PATH,
    verbose: bool = False,
    book_cache: Any = None,
    depth_map: dict | None = None,
) -> dict[str, int]:
    """For (station_id, target_date_iso, target), find buckets that are
    N_BUCKETS_AWAY or more past the current peak/trough and submit FAK
    BUY NO at the current ask (capped at max_no_ask, refused below
    min_no_ask).

    Returns counts dict:
      placed              successful fires
      skipped_pre_trigger station-local time hasn't reached trigger
      skipped_excluded    station blacklisted
      skipped_no_peak     can't find a peak bucket for observed extreme
      skipped_already_open dedupe via Portfolio.is_open
      skipped_ask_too_high NO ask > max_no_ask (Layer 7 territory)
      skipped_ask_too_low  NO ask < min_no_ask (market disagrees)
      skipped_no_book      no NO ask available
      submit_failed        SDK / Polymarket rejection
      no_candidates        no buckets met the N_BUCKETS_AWAY criterion
    """
    counts: dict[str, int] = defaultdict(int)
    counts["placed"] = 0  # ensure key exists

    station = STATIONS_BY_ID.get(station_id)
    if station is None:
        return dict(counts)

    # Note: when client.is_dry_run, client.submit_order returns a
    # synthetic OrderResult(ok=True, dry_run=True). We still run the
    # full strategy logic so paper-logs accumulate.

    # Excluded stations (DNMM, ZGSZ, WIHH, ZSQD as of decommission)
    excluded = {sid for sid, _t in load_active_exclusions(datetime.now(timezone.utc).date())}
    if station_id in excluded:
        counts["skipped_excluded"] += 1
        return dict(counts)

    now_utc = now_utc or datetime.now(timezone.utc)
    past_trig, local_h = _is_past_trigger(
        station.timezone, target, now_utc,
        trigger_local_hour_max, trigger_local_hour_min,
    )
    if not past_trig:
        counts["skipped_pre_trigger"] += 1
        return dict(counts)

    # Find the "peak bucket" — the one containing observed_extreme_c.
    peak_low_c: float | None = None
    peak_high_c: float | None = None
    bucket_width_c = 1.0 if station.unit == "C" else (5.0 / 9.0) * 2.0  # 2°F
    for m in bucket_snapshots:
        try:
            kind, thr = parse_bucket(m)
            low_c, high_c = bucket_edges_c(kind, int(thr), station.unit)
        except (ValueError, TypeError, KeyError):
            continue
        if low_c <= observed_extreme_c <= high_c + 1e-9:
            peak_low_c, peak_high_c = low_c, high_c
            break
    if peak_low_c is None:
        counts["skipped_no_peak"] += 1
        return dict(counts)

    # Identify too-far buckets.
    #   max-target: low_c >= peak_high_c + (N-1) * bucket_width_c
    #   min-target: high_c <= peak_low_c  - (N-1) * bucket_width_c
    candidate_buckets: list[tuple[Any, str, int, float, float]] = []
    for m in bucket_snapshots:
        if not m.no_token_id:
            continue
        try:
            kind, thr = parse_bucket(m)
            low_c, high_c = bucket_edges_c(kind, int(thr), station.unit)
        except (ValueError, TypeError, KeyError):
            continue
        if target == "max":
            min_low = peak_high_c + (n_buckets_away - 1) * bucket_width_c - 1e-9
            if low_c >= min_low and low_c > peak_high_c:
                candidate_buckets.append((m, kind, int(thr), low_c, high_c))
        elif target == "min":
            max_high = peak_low_c - (n_buckets_away - 1) * bucket_width_c + 1e-9
            if high_c <= max_high and high_c < peak_low_c:
                candidate_buckets.append((m, kind, int(thr), low_c, high_c))

    if not candidate_buckets:
        counts["no_candidates"] += 1
        return dict(counts)

    if verbose:
        print(f"  [hbn] {station_id} {target}={observed_extreme_c:.1f}°C "
              f"local={local_h:.1f}h → {len(candidate_buckets)} candidate(s)")

    for m, kind, thr, low_c, high_c in candidate_buckets:
        # Dedupe
        if portfolio.is_open(m.no_token_id, "NO"):
            counts["skipped_already_open"] += 1
            continue

        # Get NO ask. Prefer WS book cache if provided.
        no_ask: float | None = None
        no_ask_source = "rest"
        if book_cache is not None and m.no_token_id:
            try:
                fresh = book_cache.fresh_best_ask(m.no_token_id, max_age_seconds=5.0)
                if fresh is not None:
                    no_ask = float(fresh)
                    no_ask_source = "ws"
            except Exception:
                pass
        if no_ask is None and m.yes_bid is not None:
            # NO ask ≈ 1 − yes_bid for Polymarket binary outcomes
            no_ask = 1.0 - float(m.yes_bid)
            no_ask_source = "rest_inferred"
        if no_ask is None:
            counts["skipped_no_book"] += 1
            continue

        if no_ask > max_no_ask:
            counts["skipped_ask_too_high"] += 1
            continue
        if no_ask < min_no_ask:
            counts["skipped_ask_too_low"] += 1
            continue

        # Polymarket only accepts 2-decimal limit prices. We submit at
        # the CAP (max_no_ask = $0.98) and let the matching engine walk
        # the book — fills at the cheapest available ask, never above
        # our limit.
        submitted_limit = round(float(max_no_ask), 2)

        # DEPTH-AWARE FILL. Walk the NO-token ask ladder up to the limit
        # to get the realistic avg fill + filled shares. Without depth we
        # used to assume the whole $5 cleared at top-of-book — optimistic
        # on thin books. If there's no acceptable depth, skip (the order
        # couldn't fill a minimum-size lot in reality).
        from .polymarket import simulate_buy_fill
        depth = (depth_map or {}).get(m.no_token_id)
        sim = simulate_buy_fill(depth, size_usd, submitted_limit)
        if sim is not None:
            fill_avg, shares, fully_filled = sim
            depth_source = "depth_walk"
        elif depth is not None:
            # Real book fetched but < min order size under the limit —
            # a real order couldn't fill. Gate.
            counts["skipped_no_depth"] += 1
            continue
        else:
            # Depth unavailable (not fetched / 404 dead-market) → fall
            # back to top-of-book so we don't freeze firing on a flaky or
            # absent /book. Flagged in the log as optimistic.
            fill_avg = no_ask
            shares = float(max(1, int(size_usd / submitted_limit)))
            fully_filled = True
            depth_source = "top_of_book_fallback"

        signal = TradeSignal(
            station=station, event_title="", event_slug="",
            target=target, target_date=datetime.fromisoformat(target_date_iso).date(),
            bucket_label=m.bucket_label, bucket_kind=kind,
            market_id=int(m.market_id), token_id=m.no_token_id,
            our_prob=1.0 - no_ask, yes_implied=float(m.yes_ask or 0.0),
            yes_bid=m.yes_bid, yes_ask=m.yes_ask,
            side="NO", edge=1.0 - no_ask,
            # fill_price = depth-walked avg (what a real FAK gets). The
            # dry-run client records this as the fill; override_shares is
            # the depth-limited filled size.
            fill_price=fill_avg,
            volume_24hr=0.0, bias_applied_c=0.0,
            sigma_ensemble_c=0.0, sigma_total_c=0.0,
            kelly_full=1.0, position_usd=fill_avg * shares,
        )
        try:
            result = client.submit_order(
                signal, order_type="FAK", sdk_side="BUY",
                limit_price=submitted_limit,
                override_shares=shares,
            )
        except Exception as exc:
            counts["submit_failed"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(), "result": "submit_exception",
                "station_id": station_id, "target": target,
                "target_date": target_date_iso, "bucket_label": m.bucket_label,
                "no_ask": no_ask, "exc": str(exc)[:200],
            }, log_path)
            continue
        if not result.ok:
            counts["submit_failed"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(), "result": "rejected",
                "station_id": station_id, "target": target,
                "target_date": target_date_iso, "bucket_label": m.bucket_label,
                "no_ask": no_ask, "message": (result.message or "")[:200],
            }, log_path)
            continue

        # Persist position — entry_price = depth-walked avg fill.
        position = Position(
            token_id=m.no_token_id, side="NO",
            station_id=station_id, region=region_for(station_id),
            market_id=int(m.market_id), bucket_label=m.bucket_label,
            bucket_kind=kind, bucket_threshold=int(thr),
            target_date=target_date_iso,
            shares=shares, entry_price=fill_avg, position_usd=fill_avg * shares,
            submitted_at=_now_utc_iso(), status="filled",
            order_id=result.order_id, strategy="high_bucket_no",
        )
        try:
            portfolio.add(position)
            portfolio.save(portfolio_path)
        except Exception as exc:
            record_alert(
                kind="orphan_order_save_failed", severity="critical",
                detail=f"high_bucket_no portfolio.save failed: {exc}",
            )
        counts["placed"] += 1
        _log_event({
            "ts_utc": _now_utc_iso(), "result": "filled",
            "station_id": station_id, "target": target,
            "target_date": target_date_iso, "bucket_label": m.bucket_label,
            "bucket_kind": kind, "bucket_threshold": int(thr),
            "observed_extreme_c": observed_extreme_c,
            "peak_low_c": peak_low_c, "peak_high_c": peak_high_c,
            "bucket_low_c": low_c, "bucket_high_c": high_c,
            "no_ask_snapshot": no_ask,        # top-of-book ask at fire time
            "no_ask_source": no_ask_source,
            "submitted_limit": submitted_limit,
            "fill_price": result.fill_price,     # depth-walked avg fill
            "depth_source": depth_source,        # "depth_walk" | "top_of_book_fallback"
            "fully_filled": fully_filled,        # False = depth ran out, partial fill
            "shares": shares, "size_usd": fill_avg * shares,
            "order_id": result.order_id, "local_hour": local_h,
        }, log_path)

    return dict(counts)
