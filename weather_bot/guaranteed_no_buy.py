"""Layer 7 — Guaranteed NO buy on bucket cross (post-cross harvest).

Symmetric counterpart of weather_bot/cross_up_cancel.py.

When the observed peak (max-target) or trough (min-target) crosses
PAST a bucket's relevant edge, that bucket is structurally dead:
  - max-target: peak ≥ bucket_high_c → bucket can never win (peak can't
    go back down)
  - min-target: trough ≤ bucket_low_c → bucket can never win (trough
    can't go back up)

For any such dead bucket, NO is a STRUCTURALLY guaranteed winner. The
SDK / market participants typically reprice NO to ~$0.95-$0.99 within
seconds of the cross. Layer 7's job is to aggressively BUY NO at any
market ask ≤ $0.98 on these dead buckets — collecting ~2-5pp of edge
on each fill.

Hard cap: $0.98 per share. Above that, expected EV is negative once
oracle-dispute and ASOS data-revision risk is priced in (≤2% failure
rate; net EV at $0.99 ≈ 0.99 × $0.01 - 0.01 × $0.99 = ~0).

Net-of-fee EV (Polymarket fee model, confirmed 2026-05-25):
  Taker fee per fill = shares × 0.05 × p × (1-p).
  At p=$0.98: 5sh × 0.05 × 0.98 × 0.02 = $0.0049/fire (~0.1% of $5)
  At p=$0.95: 5sh × 0.05 × 0.95 × 0.05 = $0.0119/fire (~0.25% of $5)
  At p=$0.99: 5sh × 0.05 × 0.99 × 0.01 = $0.0025/fire (~0.05% of $5)
  Effectively negligible at Layer 7's high-price regime — the p(1-p)
  curve sends fees toward zero exactly where Layer 7 operates. No
  threshold change needed; the $0.98 cap stays as the EV ceiling.

Key properties (vs Layer 5 paper-log entry filter):
  - LIVE (not paper-log). The math is the math — peak past bucket
    means bucket dead, no predictive risk.
  - Structural guarantee, not predictive signal. Low ASOS / oracle
    failure rate is the only real risk.
  - Single-use per (token, target_date). Once we've bought NO on a
    confirmed-dead bucket, we don't re-buy.

Confirmation log (data/guaranteed_no_buy_log.jsonl):
  Per fill: timestamp, station, bucket, ask, fill_price, shares,
  expected_payout. After 3 days of data we measure actual fill-to-
  redeem rate; if ≥98% redeem successfully, the cap stays at $0.98
  and the layer is fully validated.

Where Layer 7 fires:
  - Inside intraday_scan, right after METAR fetch + cross_up_cancel
    detection. Reuses the same observation event.
  - Also intended to be invoked by the Layer 1 daemon (when it ships)
    on the 30s polling cadence — for racing scenarios where market
    repricing happens faster than 15-min cron tick.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .alerts import record_alert
from .exclusions import load_active_exclusions
from .locations import STATIONS_BY_ID
from .obs_distance_filter import bucket_edges_c
from .polymarket import parse_bucket
from .portfolio import DEFAULT_PORTFOLIO_PATH, Portfolio, Position
from .scanner import TradeSignal


def _station_local_date_iso(station_timezone: str) -> str | None:
    """Return the station's local calendar date as ISO 'YYYY-MM-DD'.

    Layer 7 fires on a (station, target_date) combo iff the station-local
    current date == target_date. Without this, UTC-based date matching
    lets stations in UTC+N timezones trip a "tomorrow UTC" market that
    is actually "today" in station-local time -- which is correct -- BUT
    also lets stations in UTC-N timezones trip a "tomorrow UTC" market
    that is FUTURE locally (today's observation cannot constrain it).

    Returns None on timezone-lookup failure (caller should fall back to
    UTC date to avoid blocking all of Layer 7 on a single bad config).
    """
    try:
        tz = ZoneInfo(station_timezone)
    except Exception:
        return None
    return datetime.now(tz).date().isoformat()


HARD_CAP_PRICE = 0.99
"""Maximum we'll pay for NO on a confirmed-dead bucket. Above this,
expected EV turns negative once oracle-dispute / data-revision risk
(≤2% failure) is priced in.

CHANGED 2026-05-18 from $0.98 → $0.99 to unblock 5.5% of Layer 7
evaluations that were `skipped_ask_high` at $0.98. Stays 2-decimal
clean (Polymarket maker_amount precision requirement)."""

MIN_NO_ASK_SANITY = 0.80
"""SANITY GATE (added 2026-05-21 after catastrophic bug):
if NO_ask is BELOW this threshold, the market disagrees with our
'dead bucket' determination by a huge margin. We bet NO is certain
to win ($0.99 cap); market thinks NO has <20% probability. Trust the
market -- our bucket-classification is wrong (oracle disagreement,
peak rounding, or stale METAR).

The 2026-05-21 incident: Layer 7 detected KLGA 66-67°F as 'dead'
(peak=20.0°C at bucket_high). NO_ask was $0.015 (market priced
bucket at 98.5% YES). Bot sent BUY NO at $0.99 limit; Polymarket
depth-walked filled at $0.015 -> bought 324 shares of an almost-
certainly-losing position for $4.86. Total damage across two cron
runs: ~$9.81 of expected loss.

Same pattern as live_bucket_arb's MIN_ALIVE_YES_SUM gate: when
market consensus contradicts our determination, trust the market."""

DEFAULT_GUARANTEED_LOG_PATH = Path("data/guaranteed_no_buy_log.jsonl")
DEFAULT_SIZE_USD = 5.0
"""Per-fill notional size. Matches NO_momentum sizing so capital
allocation feels symmetric. Adjustable via caller."""

MAX_PER_BUCKET_USD = 10.0
"""Per-bucket total exposure cap (NO_momentum entry + Layer 7 add-on).
Prevents Layer 7 from over-concentrating on a single bucket when adding
to an existing NO_momentum position (audit 2026-05-18)."""

MIN_MARGIN_PAST_BUCKET_C = 1.0
"""ORACLE-DISAGREEMENT MARGIN (added 2026-05-23 after 2 incidents).

LOWERED 2026-05-24 from 1.5°C → 1.0°C. At 1.5°C the filter blocked
53,620 Layer 7 dead-bucket evaluations per day -- way too aggressive.
At 1.0°C the filter still skips today's RKSI-style 1°C misses but
allows the larger pool of legitimate fires past 1°C margin.

Layer 7 fires only when observed extreme is at least this many °C past
the bucket edge. Within this margin, the bot's ASOS/METAR data and
Polymarket's Wunderground oracle can disagree by 0.5-1.0°C (rounding,
sensor selection, sustained vs peak), flipping the bucket-dead call.

Empirical: 2026-05-22 ZGSZ + 2026-05-23 RKSI both fired with ~1°C
margin past the bucket edge -- ZGSZ won (Polymarket agreed), RKSI lost
(Polymarket disagreed, oracle picked 22°C while our METAR read 23°C).
Both at 1.0°C — exactly the boundary. The data is genuinely 50/50
at the 1°C mark; per-station calibration TBD will give a sharper cut.

Filtered fires are logged to MARGIN_FILTER_LOG_PATH for later analysis:
if the skipped-trade hit-rate proves >>80%, the threshold is too tight
and we leave money on the table."""

MARGIN_FILTER_LOG_PATH = Path("data/margin_filter_log.jsonl")
"""Append-only log of Layer 7 fires SKIPPED by the margin filter (i.e.,
within MIN_MARGIN_PAST_BUCKET_C of the bucket edge). Each entry has the
station, bucket, observed value, bucket edge, margin, and the NO ask
at the moment. After N=30+ entries, analyze whether the filtered trades
would have won (= filter is too tight, lower the threshold) or lost
(= filter is doing its job)."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_margin_filtered(record: dict, path: Path = MARGIN_FILTER_LOG_PATH) -> None:
    """Append one filtered-trade record to the margin filter log."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # never disrupt the live path on logging failure


def _log_event(
    record: dict,
    log_path: Path = DEFAULT_GUARANTEED_LOG_PATH,
) -> None:
    """Append one event record to the JSONL confirmation log.

    All exceptions are swallowed. Logging must NEVER disrupt order
    placement / cancellation. If disk is full, permission denied, or
    the record isn't JSON-serializable, we silently drop the entry —
    the bot's real state survives via portfolio.json + journalctl.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _presigned_cache_key(token_id: str, price: float, size_shares: float) -> str:
    """Stable cache key for a pre-signed FAK BUY NO order."""
    return f"L7|{token_id}|BUY|{price:.4f}|{size_shares:.4f}|FAK"


def detect_and_execute_guaranteed_buys(
    *,
    station_id: str,
    target_date_iso: str,
    observed_extreme_c: float,
    target: str,                  # "max" or "min"
    bucket_snapshots: list,        # current PolymarketEvent.markets list
    client: Any,                   # ExecutionClient
    portfolio: Portfolio,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
    size_usd: float = DEFAULT_SIZE_USD,
    hard_cap_price: float = HARD_CAP_PRICE,
    log_path: Path = DEFAULT_GUARANTEED_LOG_PATH,
    verbose: bool = True,
    # Pre-signed order cache (optional). When provided, Layer 7 tries to
    # broadcast a pre-signed order before falling back to inline sign+broadcast.
    # Saves ~200ms per fill on racing scenarios.
    pre_signed_cache: Any = None,  # PreSignedOrderCache | None
    # WebSocket BookCache (optional). When provided, Layer 7 prefers the
    # cached fresh best-ask over the (potentially stale) REST snapshot in
    # `bucket_snapshots`. The cycle's REST data can be 1-30s old; WS push
    # updates land in <1s. On dead-bucket reactions this matters because
    # other participants are repricing simultaneously and we want the
    # truest "now" price for the ≤$0.98 cap check.
    book_cache: Any = None,         # BookCache | None
    ws_max_age_seconds: float = 5.0,
) -> dict[str, int]:
    """For (station_id, target_date_iso, target), find structurally-dead
    buckets and submit FAK BUY NO at market ask (capped at hard_cap_price).

    Returns counts:
      placed:               successful FAK BUY NO submissions
      skipped_have_pos:     we already hold a NO position on this bucket
      skipped_ask_high:     NO ask > hard_cap_price → would be negative EV
      skipped_no_ask:       no NO ask available (illiquid)
      skipped_alive:        bucket is not actually dead yet
      skipped_daily_limit:  would exceed the $/day deployment cap
      submit_failed:        SDK / Polymarket rejection
      no_action:            nothing to do (no dead buckets found)
    """
    counts: dict[str, int] = defaultdict(int)
    station = STATIONS_BY_ID.get(station_id)
    if station is None:
        return dict(counts)
    if client.is_dry_run:
        if verbose:
            print(f"  [layer7] dry-run client — skipping for {station_id}")
        return {"reason": "dry-run"}

    # STATION-LOCAL DATE GUARD (2026-05-22 LTFM incident). Layer 7 reasons
    # about "today's observed extreme has passed bucket X, so bucket X is
    # dead". This is only valid if `target_date_iso` is the station's
    # CURRENT calendar day -- not a future day in station-local time.
    #
    # The LTFM incident illustrates BOTH sides:
    #   - target_date=2026-05-22 fired at UTC 21:16 (= local 00:16 in
    #     Istanbul UTC+3). Station-local date WAS 2026-05-22. Valid. ✓
    #   - Same instant for KLAX (UTC-7): local 14:16 on 2026-05-21.
    #     Station-local date is 2026-05-21. A 2026-05-22 market is
    #     FUTURE locally -- today's peak doesn't constrain it. ✗
    #
    # UTC-based date matching gets the Istanbul case right by accident.
    # Station-local makes both right by construction.
    station_today = _station_local_date_iso(station.timezone)
    if station_today is None:
        # Lookup failed -- fall through using UTC date (preserves prior
        # behavior; better than blocking on a station-config issue).
        pass
    elif target_date_iso != station_today:
        counts["skipped_not_station_local_today"] += 1
        if verbose:
            print(f"  [layer7] {station_id} {target}={observed_extreme_c:.1f}°C: "
                  f"target_date={target_date_iso} ≠ station-local-today={station_today} "
                  f"({station.timezone}); skipping")
        return dict(counts)

    # EXCLUDED-STATIONS GUARD (2026-05-22 ZGSZ incident). Layer 7
    # previously did not honor data/excluded_stations.json -- only
    # NO_momentum did. ZGSZ (Shenzhen) is excluded because the ASOS
    # feed disagrees with the Polymarket oracle by 1-3°C in mixed
    # directions; an "observed peak past bucket X" call from our data
    # might be wrong, so trades on excluded stations carry data-quality
    # risk. ZGSZ 28°C 2026-05-22 filled today before this guard shipped.
    try:
        active_exclusions = load_active_exclusions(today=datetime.now(timezone.utc).date())
    except Exception:
        active_exclusions = set()
    if (station_id, target) in active_exclusions:
        counts["skipped_excluded"] += 1
        if verbose:
            print(f"  [layer7] {station_id} {target}: station-target excluded; skipping")
        return dict(counts)

    # Daily-deployment circuit breaker (mirrors NO_momentum's gate).
    # Each fill is ≤$5; without this, a tick that finds 30 dead buckets
    # would deploy $150+ in one shot, breaching the daily cap.
    daily_limit = float(getattr(client.config, "daily_deployment_limit_usd", 150.0))

    # Find ALL dead buckets in this event (could be multiple — every
    # bucket below the peak is dead for max-target)
    dead_buckets: list[tuple] = []  # (market, kind, threshold, low_c, high_c)
    for m in bucket_snapshots:
        if not m.no_token_id:
            continue
        try:
            kind, thr = parse_bucket(m)
            low_c, high_c = bucket_edges_c(kind, int(thr), station.unit)
        except (ValueError, TypeError, KeyError):
            continue

        is_dead = False
        margin_c = 0.0  # how far past the bucket edge the observation is
        if target == "max":
            # max-target: bucket dead if peak is at or past upper edge
            if observed_extreme_c >= high_c:
                is_dead = True
                margin_c = observed_extreme_c - high_c
        elif target == "min":
            # min-target: bucket dead if trough is at or past lower edge
            if observed_extreme_c <= low_c:
                is_dead = True
                margin_c = low_c - observed_extreme_c

        if is_dead:
            # ORACLE-DISAGREEMENT MARGIN FILTER (2026-05-23). If we're
            # within MIN_MARGIN_PAST_BUCKET_C of the bucket edge, the
            # observation is close enough to the boundary that
            # Polymarket's Wunderground oracle and our ASOS/METAR can
            # legitimately disagree. Skip and LOG so we can later
            # measure whether these "filtered" trades would have won.
            if margin_c < MIN_MARGIN_PAST_BUCKET_C:
                counts["skipped_within_margin"] += 1
                _log_margin_filtered({
                    "ts_utc": _now_utc_iso(),
                    "station_id": station_id,
                    "target": target,
                    "target_date": target_date_iso,
                    "bucket_label": m.bucket_label,
                    "bucket_kind": kind,
                    "bucket_threshold": int(thr),
                    "bucket_low_c": low_c if low_c != float("-inf") else None,
                    "bucket_high_c": high_c if high_c != float("inf") else None,
                    "observed_extreme_c": observed_extreme_c,
                    "margin_c": margin_c,
                    "margin_threshold_c": MIN_MARGIN_PAST_BUCKET_C,
                    "no_token_id": m.no_token_id,
                    "yes_token_id": m.yes_token_id,
                    "yes_bid": m.yes_bid,
                    "yes_ask": m.yes_ask,
                    "no_ask_implied": (
                        1.0 - m.yes_bid if m.yes_bid is not None else None
                    ),
                })
                if verbose:
                    edge_str = (
                        f"high={high_c:.1f}" if target == "max"
                        else f"low={low_c:.1f}"
                    )
                    print(f"    ⊘ margin-filter: {m.bucket_label} "
                          f"obs={observed_extreme_c:.1f} {edge_str} "
                          f"margin={margin_c:.2f}°C < {MIN_MARGIN_PAST_BUCKET_C}°C; "
                          f"skipping (oracle-disagreement risk zone)")
                continue
            dead_buckets.append((m, kind, thr, low_c, high_c))

    if not dead_buckets:
        counts["no_action"] += 1
        return dict(counts)

    if verbose:
        print(f"  [layer7] {station_id} {target}={observed_extreme_c:.1f}°C → "
              f"{len(dead_buckets)} dead bucket(s) for {target_date_iso}")

    for market, kind, thr, low_c, high_c in dead_buckets:
        # DEDUPE GUARD (2026-05-22 LTFM incident). Layer 7 had no
        # any-open-position check -- only a cumulative-USD cap
        # (MAX_PER_BUCKET_USD=$10). Result: when the cron and daemon both
        # see the same dead bucket, the first fill puts $5 exposure, the
        # next fill sees $5 < $10 cap and fires AGAIN. 5 fills happened
        # in 31 seconds before the global daily cap stopped it. Now: if
        # we already have ANY filled or submitted position on this token,
        # skip outright. This is the same pattern NO_momentum uses via
        # portfolio.should_skip / is_open. The cumulative-USD cap below
        # stays in place as a secondary guard for the NO_momentum + Layer 7
        # stacking case (different code paths can each place once).
        if portfolio.is_open(market.no_token_id, "NO"):
            counts["skipped_already_open"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "cap_price": hard_cap_price,
                "result": "skipped_already_open",
            }, log_path)
            continue

        # Compute total existing exposure on this token+date so Layer 7
        # can add structurally-guaranteed NO on top of any existing
        # NO_momentum position UP TO `MAX_PER_BUCKET_USD`. Previously we
        # blanket-skipped if we held any position; that starved Layer 7
        # of opportunities because NO_momentum had usually pre-placed
        # on the same buckets (audit 2026-05-18 root-cause).
        existing_exposure_usd = 0.0
        for p in portfolio.positions:
            if p.token_id != market.no_token_id:
                continue
            if p.target_date != target_date_iso:
                continue
            if p.side != "NO":
                continue
            if p.status in ("submitted", "filled"):
                existing_exposure_usd += float(p.position_usd or 0.0)
        already_at_cap = existing_exposure_usd >= MAX_PER_BUCKET_USD
        already_holding = existing_exposure_usd > 0  # for back-compat in log records

        # Compute current NO ask up front so the skipped-candidates log
        # is populated regardless of which gate fires. Prefer fresh WS data
        # when available; otherwise derive from REST yes_bid snapshot.
        current_no_ask: Optional[float] = None
        ws_ask: Optional[float] = None
        if book_cache is not None and market.no_token_id:
            try:
                ws_ask = book_cache.fresh_best_ask(
                    market.no_token_id, max_age_seconds=ws_max_age_seconds,
                )
            except Exception:
                ws_ask = None
        if ws_ask is not None:
            current_no_ask = float(ws_ask)
        elif market.yes_bid is not None:
            current_no_ask = 1.0 - float(market.yes_bid)

        if already_at_cap:
            counts["skipped_have_pos"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": current_no_ask,
                "cap_price": hard_cap_price,
                "result": "skipped_have_pos",
            }, log_path)
            continue

        # ──────────────────────────────────────────────────────────────
        # ELIGIBILITY CHECKS — must all run BEFORE acquire_cap_token to
        # avoid silent cap leaks. Each `continue` below would otherwise
        # leave a $5 reservation un-refunded against both the global daily
        # cap and the per-station cap (audit finding C1, 2026-05-17).
        # ──────────────────────────────────────────────────────────────

        # NO ask: prefer fresh WS data (sub-second), fall back to REST
        # (yes_bid → 1-yes_bid) snapshot. current_no_ask was already
        # populated above with the WS-preferred value.
        if current_no_ask is None:
            counts["skipped_no_ask"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": None,
                "no_ask_source": None,
                "cap_price": hard_cap_price,
                "result": "skipped_no_bid",
            }, log_path)
            continue
        no_ask = float(current_no_ask)
        no_ask_source = "ws" if ws_ask is not None else "rest"
        if ws_ask is not None:
            counts["ws_ask_used"] += 1
        if no_ask > hard_cap_price:
            counts["skipped_ask_high"] += 1
            if verbose:
                print(f"    ✗ {market.bucket_label}: NO ask ${no_ask:.3f} "
                      f"({no_ask_source}) > cap ${hard_cap_price:.2f}; skipping")
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": no_ask,
                "no_ask_source": no_ask_source,
                "cap_price": hard_cap_price,
                "result": "skipped_ask_high",
            }, log_path)
            continue
        if no_ask <= 0.001:
            counts["skipped_no_ask"] += 1
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": no_ask,
                "no_ask_source": no_ask_source,
                "cap_price": hard_cap_price,
                "result": "skipped_dust_ask",
            }, log_path)
            continue

        # SANITY GATE (Fix 2026-05-21): if NO_ask is far below the cap,
        # the market strongly disagrees with our 'dead bucket'
        # determination. Refuse the buy -- otherwise the depth-walk
        # fills at the actual ask, getting many shares of an almost-
        # certainly-losing position. The 2026-05-21 incident lost ~$9.81
        # on a single bucket where NO_ask was $0.015 vs our $0.99 cap.
        if no_ask < MIN_NO_ASK_SANITY:
            counts["skipped_market_disagreement"] += 1
            if verbose:
                print(f"    ✗ {market.bucket_label}: NO ask ${no_ask:.3f} "
                      f"({no_ask_source}) FAR BELOW cap ${hard_cap_price:.2f} -- "
                      f"market disagrees with 'dead' determination; skipping")
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": no_ask,
                "no_ask_source": no_ask_source,
                "cap_price": hard_cap_price,
                "min_no_ask_sanity": MIN_NO_ASK_SANITY,
                "result": "skipped_market_disagreement",
            }, log_path)
            continue

        # Parse target_date_obj before cap acquire (ValueError shouldn't
        # eat cap budget either — same audit finding C1).
        from datetime import date as _date
        try:
            target_date_obj = _date.fromisoformat(target_date_iso)
        except ValueError:
            counts["submit_failed"] += 1
            continue

        market_id = getattr(market, "market_id", 0) or 0

        # ──────────────────────────────────────────────────────────────
        # SIZE — respects per-bucket exposure cap so Layer 7 doesn't
        # over-concentrate on a single bucket when adding to an existing
        # NO_momentum position.
        # ──────────────────────────────────────────────────────────────
        remaining_bucket_headroom = max(0.0, MAX_PER_BUCKET_USD - existing_exposure_usd)
        effective_size_usd = min(float(size_usd), remaining_bucket_headroom)
        if effective_size_usd < 1.00:
            # Below Polymarket marketable minimum — skip
            counts["skipped_bucket_cap"] += 1
            if verbose:
                print(f"    ✗ {market.bucket_label}: only ${effective_size_usd:.2f} "
                      f"headroom (existing ${existing_exposure_usd:.2f}, cap "
                      f"${MAX_PER_BUCKET_USD}); skip")
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "existing_exposure_usd": existing_exposure_usd,
                "remaining_headroom_usd": remaining_bucket_headroom,
                "result": "skipped_bucket_cap",
            }, log_path)
            continue

        # ──────────────────────────────────────────────────────────────
        # CAP ACQUIRE — only reached when the bucket is genuinely placeable.
        # All early-`continue` paths above this line are SAFE (no cap leak).
        # ──────────────────────────────────────────────────────────────
        from weather_bot.cap_budget import acquire_cap_token, release_cap_token
        # Layer 7 is a fast-reaction caller — uses the $30 reserve carved
        # out from no_momentum's effective cap (Day 3 fix 2026-05-18).
        cap_ok, cap_reason = acquire_cap_token(
            size_usd=effective_size_usd, daily_limit_usd=daily_limit,
            station_id=station_id,
            caller_kind="fast_reaction",
        )
        if not cap_ok:
            counter_key = (
                "skipped_daily_limit" if cap_reason == "global_cap"
                else "skipped_station_cap"
            )
            counts[counter_key] += 1
            if verbose:
                print(f"    ✗ {market.bucket_label}: cap refused ({cap_reason}); skipping")
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": no_ask,
                "no_ask_source": no_ask_source,
                "cap_price": hard_cap_price,
                "daily_limit_usd": daily_limit,
                "cap_reason": cap_reason,
                "result": counter_key,
            }, log_path)
            continue

        signal = TradeSignal(
            station=station,
            event_title="",
            event_slug="",
            target=target,
            target_date=target_date_obj,
            bucket_label=market.bucket_label,
            bucket_kind=kind,
            market_id=market_id,
            token_id=market.no_token_id,
            our_prob=1.0,  # structurally guaranteed
            yes_implied=float(market.yes_ask) if market.yes_ask is not None else None,
            yes_bid=market.yes_bid,
            yes_ask=market.yes_ask,
            side="NO",
            edge=1.0 - no_ask,
            fill_price=no_ask,
            volume_24hr=0.0,
            bias_applied_c=0.0,
            sigma_ensemble_c=0.0,
            sigma_total_c=0.0,
            kelly_full=1.0,
            position_usd=effective_size_usd,
        )

        # Compute size_shares with Polymarket FAK BUY decimal-precision
        # constraints in mind (root cause of 100% Layer 7 rejections per
        # 2026-05-18 investigation):
        #   - maker_amount (USDC) = shares × price must have ≤2 decimals
        #   - taker_amount (shares) must have ≤4-5 decimals (per market)
        #
        # The cleanest way to satisfy both is to use an integer share
        # count: shares × price is automatically ≤2-decimal as long as
        # price itself is ≤2-decimal (which is Polymarket's tick size).
        #
        # `hard_cap_price` is the limit price the FAK walks the book up to.
        # We deploy slightly less than effective_size_usd (e.g., 5 shares
        # × $0.99 = $4.95 instead of $5.00) but the order will be
        # accepted by Polymarket.
        limit_price_clean = round(float(hard_cap_price), 2)
        # Integer share count, at least 1, at most effective_size_usd / price floor
        size_shares_target = float(effective_size_usd) / limit_price_clean
        size_shares = float(max(1, int(size_shares_target)))
        # Sanity-check: verify maker_amount is clean
        maker_amount_check = round(size_shares * limit_price_clean, 4)
        if abs(maker_amount_check - round(maker_amount_check, 2)) > 1e-6:
            # Should never happen with integer shares × 2-decimal price,
            # but log if it does so we catch precision drift.
            if verbose:
                print(f"  [layer7] precision check failed: {size_shares} × "
                      f"{limit_price_clean} = {maker_amount_check}")
        cache_key = _presigned_cache_key(market.no_token_id, hard_cap_price, size_shares)
        result = None
        used_presigned = False
        # Audit finding M3: the pre-signed order has FIXED size_shares
        # computed from the cap price. If current ask is materially below
        # the cap, the FAK fills `size_shares × current_ask` USD — well
        # under the intended size_usd. The inline path correctly sizes to
        # `size_usd / current_ask`. Only use the pre-signed fast path when
        # the ask is close enough to the cap that the size delta is small.
        # Threshold $0.03 (≈ 3% under-deploy max) — racing-tight buckets
        # (the actual hot path for pre-sign) sit at $0.95-$0.98.
        PRESIGN_ASK_MARGIN = 0.03
        ask_near_cap = (no_ask >= hard_cap_price - PRESIGN_ASK_MARGIN)
        if pre_signed_cache is not None and not ask_near_cap:
            counts["presigned_skipped_far_from_cap"] += 1
            if verbose:
                print(f"  [layer7-fast] {market.bucket_label}: ask ${no_ask:.3f} "
                      f"too far below cap ${hard_cap_price:.2f}; using inline "
                      f"for correct sizing")
        if pre_signed_cache is not None and ask_near_cap:
            try:
                entry = pre_signed_cache.peek(cache_key)
                if entry is not None:
                    if verbose:
                        print(f"  [layer7-fast] {market.bucket_label} dead — broadcasting "
                              f"pre-signed FAK BUY NO @ cap ${hard_cap_price:.2f}")
                    raw_resp = pre_signed_cache.broadcast(cache_key)
                    # Adapt raw SDK response into an OrderResult-compatible shape.
                    # client.submit_order returns an OrderResult; the pre-signed
                    # broadcast returns the raw SDK dict. We need to bridge.
                    from weather_bot.execution.client import OrderResult
                    ok = bool(raw_resp.get("success", False))
                    order_id = raw_resp.get("orderID") or raw_resp.get("order_id")
                    result = OrderResult(
                        ok=ok, order_id=order_id, side="NO",
                        token_id=market.no_token_id,
                        fill_price=no_ask, size_usd=effective_size_usd, shares=size_shares,
                        dry_run=False,
                        message=f"pre-signed broadcast: {raw_resp.get('status', 'unknown')}",
                        limit_price=limit_price_clean, target_shares=size_shares,
                    )
                    used_presigned = True
            except Exception as exc:
                # Distinguish stale-signature drop from other failures so
                # logs make it clear when we fell back due to TTL vs error.
                from weather_bot.execution.pre_signed import StaleSignatureError
                if isinstance(exc, StaleSignatureError):
                    counts["presigned_stale_fallback"] += 1
                    if verbose:
                        print(f"  [layer7-fast] pre-sign STALE — refusing broadcast, "
                              f"falling back to inline submit")
                else:
                    counts["presigned_error_fallback"] += 1
                    if verbose:
                        print(f"  [layer7-fast] pre-signed broadcast error: {exc}; "
                              f"falling back to inline submit")
                result = None  # fall through to inline path below

        if result is None:
            if verbose:
                print(f"  [layer7] {market.bucket_label} dead (peak={observed_extreme_c:.1f} "
                      f"past {high_c if target == 'max' else low_c:.1f}); "
                      f"FAK BUY NO @ ask=${no_ask:.3f} ({no_ask_source}) "
                      f"cap=${hard_cap_price:.2f}")
            try:
                result = client.submit_order(
                    signal,
                    order_type="FAK",
                    sdk_side="BUY",
                    limit_price=limit_price_clean,  # 2-decimal clean (FAK precision req)
                    post_only=False,
                    override_shares=size_shares,    # integer shares (FAK precision req)
                )
            except Exception as exc:
                if verbose:
                    print(f"    ✗ submit exception: {exc}")
                # Refund the cap reservation (no deployment happened)
                release_cap_token(effective_size_usd, station_id=station_id)
                counts["submit_failed"] += 1
                _log_event({
                    "ts_utc": _now_utc_iso(),
                    "station_id": station_id,
                    "target_date": target_date_iso,
                    "bucket_label": market.bucket_label,
                    "observed_extreme_c": observed_extreme_c,
                    "bucket_low_c": low_c if low_c != float("-inf") else None,
                    "bucket_high_c": high_c if high_c != float("inf") else None,
                    "no_ask_at_attempt": no_ask,
                    "cap_price": hard_cap_price,
                    "result": "submit_exception",
                    "exception": str(exc),
                }, log_path)
                continue

        if not result.ok:
            # Refund the cap reservation (placement was rejected)
            release_cap_token(size_usd, station_id=station_id)
            counts["submit_failed"] += 1
            if verbose:
                msg = (result.message or "")[:80]
                print(f"    ✗ rejected: {msg}")
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "observed_extreme_c": observed_extreme_c,
                "no_ask_at_attempt": no_ask,
                "cap_price": hard_cap_price,
                "result": "rejected",
                "message": result.message,
            }, log_path)
            continue

        # Persist position. Use ACTUAL filled amounts (not the intended
        # size_usd) — FAK can partially fill if book depth runs out
        # before our limit. Record the real position_usd = fill_price *
        # filled_shares so portfolio.today_deployed_usd / total_exposure
        # are accurate. Refund any unfilled portion to cap_budget.
        fill_price = float(getattr(result, "fill_price", no_ask) or no_ask)
        filled_shares = float(getattr(result, "shares", effective_size_usd / fill_price) or (effective_size_usd / fill_price))
        actual_position_usd = fill_price * filled_shares

        # Near-zero fill (< $0.50 worth) — Polymarket's marketable minimum
        # is $1 so this is rare, but it does happen on certain SDK
        # response shapes. Previously we treated this as a rejection AND
        # didn't record the position, leaving Polymarket with shares the
        # bot has no record of → ghost shares at resolution (audit F2-H4).
        #
        # New behavior: refund the UNFILLED portion to cap, but RECORD
        # the position so poll_resolutions can redeem it normally. The
        # position is tagged `near_zero_fill=True` for offline filtering.
        if actual_position_usd < 0.50:
            unfilled_near_zero = max(0.0, float(effective_size_usd) - actual_position_usd)
            if unfilled_near_zero > 0.01:
                release_cap_token(unfilled_near_zero, station_id=station_id)
            counts["near_zero_fill_recorded"] += 1
            if verbose:
                print(f"    ⚠ near-zero fill (${actual_position_usd:.2f}, "
                      f"{filled_shares:.4f} shares); recording position so "
                      f"poll_resolutions redeems normally")
            from weather_bot.portfolio import region_for as _region_for
            tiny_position = Position(
                token_id=market.no_token_id,
                side="NO",
                station_id=station_id,
                region=_region_for(station_id),
                market_id=market_id,
                bucket_label=market.bucket_label,
                bucket_kind=kind,
                bucket_threshold=int(thr),
                target_date=target_date_iso,
                shares=filled_shares,
                entry_price=fill_price,
                position_usd=actual_position_usd,
                submitted_at=_now_utc_iso(),
                status="filled",
                order_id=result.order_id,
                strategy="guaranteed_no_buy_near_zero",
            )
            # ORPHAN-ORDER GUARD (2026-05-21 incident): save IMMEDIATELY
            # after add so a crash later in the loop can't leave the
            # position un-persisted. Alert + re-raise on save failure.
            try:
                portfolio.add(tiny_position)
                portfolio.save(portfolio_path)
            except Exception as exc:
                record_alert(
                    kind="orphan_order_save_failed",
                    severity="critical",
                    summary=(
                        f"Layer7 near-zero {station_id} {market.bucket_label}: "
                        f"order {result.order_id} placed on Polymarket "
                        f"({filled_shares:.4f}sh @ ${fill_price:.4f}) "
                        f"but portfolio.save FAILED -- bot does not know it owns these shares"
                    ),
                    details={
                        "strategy": "guaranteed_no_buy_near_zero",
                        "station_id": station_id,
                        "bucket_label": market.bucket_label,
                        "token_id": market.no_token_id,
                        "order_id": result.order_id,
                        "shares": float(filled_shares),
                        "entry_price": float(fill_price),
                        "position_usd": float(actual_position_usd),
                        "portfolio_path": str(portfolio_path),
                        "exception": str(exc),
                        "exception_type": type(exc).__name__,
                    },
                )
                raise
            _log_event({
                "ts_utc": _now_utc_iso(),
                "station_id": station_id,
                "target_date": target_date_iso,
                "bucket_label": market.bucket_label,
                "actual_position_usd": actual_position_usd,
                "fill_price": fill_price,
                "filled_shares": filled_shares,
                "unfilled_refunded_usd": unfilled_near_zero,
                "order_id": result.order_id,
                "result": "near_zero_fill_recorded",
            }, log_path)
            continue

        # Partial fill: refund the unfilled portion to the cap budget so
        # tomorrow's accounting matches reality. Threshold $0.10 to
        # avoid micro-refunds from floating-point dust.
        unfilled_usd = float(effective_size_usd) - actual_position_usd
        if unfilled_usd > 0.10:
            release_cap_token(unfilled_usd, station_id=station_id)
            counts["partial_fill_refund"] += 1
            if verbose:
                print(f"    ⚠ partial fill: requested ${effective_size_usd:.2f}, got "
                      f"${actual_position_usd:.2f}; refunding ${unfilled_usd:.2f}")

        from weather_bot.portfolio import region_for as _region_for
        position = Position(
            token_id=market.no_token_id,
            side="NO",
            station_id=station_id,
            region=_region_for(station_id),
            market_id=market_id,
            bucket_label=market.bucket_label,
            bucket_kind=kind,
            bucket_threshold=int(thr),
            target_date=target_date_iso,
            shares=filled_shares,
            entry_price=fill_price,
            position_usd=actual_position_usd,  # actual, not intended
            submitted_at=_now_utc_iso(),
            status="filled",  # FAK either fills or rejects
            order_id=result.order_id,
            strategy="guaranteed_no_buy",  # distinguishes from NO_momentum
        )
        # ORPHAN-ORDER GUARD (2026-05-21 incident): save IMMEDIATELY after
        # add so a later crash in the loop can't leave the position
        # un-persisted. The KLGA incident was caused by exactly this gap.
        try:
            portfolio.add(position)
            portfolio.save(portfolio_path)
        except Exception as exc:
            record_alert(
                kind="orphan_order_save_failed",
                severity="critical",
                summary=(
                    f"Layer7 {station_id} {market.bucket_label}: "
                    f"order {result.order_id} placed on Polymarket "
                    f"({filled_shares:.2f}sh @ ${fill_price:.4f}) "
                    f"but portfolio.save FAILED -- bot does not know it owns these shares"
                ),
                details={
                    "strategy": "guaranteed_no_buy",
                    "station_id": station_id,
                    "bucket_label": market.bucket_label,
                    "token_id": market.no_token_id,
                    "order_id": result.order_id,
                    "shares": float(filled_shares),
                    "entry_price": float(fill_price),
                    "position_usd": float(actual_position_usd),
                    "portfolio_path": str(portfolio_path),
                    "exception": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
            raise
        counts["placed"] += 1
        # Track pre-sign hit-rate for tuning PRE_SIGN_VULNERABLE_MARGIN_C
        # and the 60-90% YES probability range. Higher hit-rate = wasted
        # pre-signing on buckets that never actually crossed.
        if used_presigned:
            counts["placed_via_presigned"] += 1
        else:
            counts["placed_via_inline"] += 1
        if verbose:
            tag = " [pre-signed]" if used_presigned else ""
            print(f"    ✓ filled order_id={(result.order_id or '?')[:14]}…  "
                  f"@${fill_price:.3f} shares={filled_shares:.2f}{tag}")

        _log_event({
            "ts_utc": _now_utc_iso(),
            "station_id": station_id,
            "target_date": target_date_iso,
            "bucket_label": market.bucket_label,
            "observed_extreme_c": observed_extreme_c,
            "bucket_low_c": low_c if low_c != float("-inf") else None,
            "bucket_high_c": high_c if high_c != float("inf") else None,
            "no_ask_at_attempt": no_ask,
            "no_ask_source": no_ask_source,  # "ws" or "rest"
            "fill_price": fill_price,
            "shares": filled_shares,
            "size_usd_requested": effective_size_usd,
            "size_usd_actual": actual_position_usd,
            "existing_bucket_exposure_usd": existing_exposure_usd,
            "unfilled_usd": unfilled_usd,
            "partial_fill": unfilled_usd > 0.10,
            "expected_payout_usd": filled_shares,  # NO redeems at $1 per share
            "expected_gross_pnl_usd": filled_shares * (1.0 - fill_price),
            "cap_price": hard_cap_price,
            "order_id": result.order_id,
            "fast_path": used_presigned,  # True if broadcast pre-signed (saves ~200ms)
            "result": "filled",
        }, log_path)

    # NOTE: per-iteration portfolio.save() runs after each add (orphan-order
    # guard, see ORPHAN-ORDER GUARD comments above). No post-loop save needed.

    return dict(counts)
