"""METAR feedback strategy — intraday module.

Runs at 30-min cadence on the resolution day. Two firing conditions:

  **Early-tail trigger (added 2026-05-12):** for high_tail buckets on
  max markets (and low_tail on min markets), a single METAR reading
  that crosses the threshold is sufficient to lock in the bucket as a
  winner. The strategy fires as soon as that crossing is observed —
  often hours before the peak-based T-6h trigger would have. Captures
  alpha that the peak-based logic misses (brief threshold-cross-then-
  fallback, or simply early certainty).

  **Peak-based trigger (T-6h default, per-station overrides):** for
  mid-range buckets where the bucket can be knocked out by subsequent
  readings, we wait until the daily peak is likely past before firing.
  Per-station map in `PEAK_TRIGGER_T_MINUS_H` handles known late-peak
  stations.

For each (station, target, target_date=today):
  1. Fetch hourly METAR observed-so-far for the station-day
  2. If any tail bucket's threshold has been crossed → fire on it
     immediately (no per-station trigger required)
  3. Else if past per-station trigger: find peak, identify mid bucket,
     fire if the price has margin
  4. Else: PRE_TRIGGER (skip, retry next cron)
  5. Log every decision to data/intraday_log.jsonl (idempotent per-day)

Bot does NOT submit orders today (`place_orders.py` is dry-run; the
v2 SDK migration is still pending). This is paper-trade — the goal is
to accumulate logged decisions and compare against eventual outcomes.

See `project_strategy_ideas.md` § C for the strategy rationale, the
trigger map derivation, and the early-tail rationale.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

DEFAULT_INTRADAY_LOG_PATH = Path("data/intraday_log.jsonl")

# Per-station trigger time: how many hours BEFORE station-local midnight
# do we start firing the strategy. Lower = later trigger (less safe but
# earlier than peak for some stations); higher = earlier trigger (safer
# but more time risk for late peaks).
#
# Variant B trigger map (applied 2026-05-15). Derived from 365-day
# peak-settlement backtest @ 97% threshold + 7-day market-data validation.
# 25 stations moved earlier vs the prior empirical map; 15 keep current;
# 5 are METAR-skipped (see METAR_SKIP_STATIONS below — peak distribution
# too wide for any single trigger to reliably pre-empt false positives;
# they still receive NO_momentum orders).
# Re-derive at N≥30 days of resolution data. Seasonal overrides below
# take precedence per-month for tropical/desert stations.
PEAK_TRIGGER_T_MINUS_H: dict[str, float] = {
    # T-4h (latest trigger — for stations with 15% post-18h peaks)
    "CYYZ": 4,
    # T-5h (DNMM excluded via data/excluded_stations.json — kept here
    # for completeness; runtime exclusion check shorts before any trade)
    "DNMM": 5,
    # T-6h (Variant B: 1h earlier than prior T-5h baseline)
    "LEMD": 6, "KSEA": 6,
    # T-7h (Variant B: 1h earlier than prior T-6h baseline)
    "EDDM": 7, "KATL": 7, "KAUS": 7, "KDAL": 7,
    "LIMC": 7, "LTAC": 7, "LTFM": 7, "SAEZ": 7,
    "ZHHH": 7, "ZUCK": 7, "ZUUU": 7,
    # T-8h (Variant B: 2h earlier OR original early-peak stations kept here)
    "KBKF": 8, "KHOU": 8, "KSFO": 8, "MMMX": 8,
    "SBGR": 8, "VILK": 8, "ZBAA": 8, "ZGGG": 8,
    "FACT": 8, "KMIA": 8, "NZWN": 8, "MPMG": 8,
    "RKSI": 8, "RPLL": 8, "WMKK": 8, "WSSS": 8,
    "ZGSZ": 8,  # excluded; see DNMM note above
    # T-9h
    "KLAX": 9, "OEJN": 9, "WIHH": 9, "RCSS": 9,
    "ZSPD": 9, "ZSQD": 9,
    # T-10h (earliest trigger — long safety margin)
    "LLBG": 10, "RKPK": 10,
    # All others default to T-6h via DEFAULT_TRIGGER_T_MINUS_H below.
}

# METAR-skip stations (2026-05-15): high-latitude stations whose 365-day
# peak distribution is too wide for any single trigger time to reliably
# pre-empt false-positives. Excluded from the METAR BUY/EXTREME decision
# branch in intraday_scan.py but still receive NO_momentum orders (Phase 2)
# when that path runs. Distinct from data/excluded_stations.json (that
# blocks ALL trading for oracle-source reasons).
METAR_SKIP_STATIONS: frozenset[str] = frozenset({
    "KLGA", "KORD", "EFHK", "EHAM", "UUWW",
})

DEFAULT_TRIGGER_T_MINUS_H: float = 6.0
"""Default trigger for stations not in the explicit map. Fractional
allowed (e.g. 6.5 = T-6.5h = 17:30 local trigger)."""


# Per-station, per-month seasonal overrides for PEAK_TRIGGER_T_MINUS_H.
# Added 2026-05-13 after measure_metar_fp_rate.py revealed a 1.37% FP rate
# concentrated on tropical-station heat-wave events (DNMM May 12: +6°C rise
# after T-5h trigger). Tropical stations have larger month-to-month peak-
# hour drift than temperate ones — hot dry months see peaks shift later,
# wet/cool months shift earlier.
#
# Format:  station_id → {month_1_to_12: t_minus_h_override}
# Lookup order: this seasonal override first, then PEAK_TRIGGER_T_MINUS_H,
# then DEFAULT_TRIGGER_T_MINUS_H.
#
# CURRENT VALUES ARE CONSERVATIVE CLIMATOLOGICAL GUESSES, not empirically
# derived. Refine when N≥30 days per (station, month) accumulate. The
# derivation script is `backtest_peak_hour_per_month.py` (TODO).
#
# Hot-month bias = trigger pushed slightly later (smaller t_minus_h) than
# annual default; cool-month bias = trigger pushed earlier (larger value).
SEASONAL_TRIGGER_OVERRIDES: dict[str, dict[int, float]] = {
    # DNMM (Lagos, 6.5°N): hot dry season Nov-Apr (peaks 32-34°C, can spike
    # higher in heat waves). May 12 2026 had +6°C rise after T-5h —
    # tentatively push Feb-May to T-4h (= 20:00 local). Re-evaluate after
    # N≥10 May days.
    "DNMM": {2: 4, 3: 4, 4: 4, 5: 4},

    # KMIA (Miami, 25.8°N): hot/humid Jun-Sep, peaks shift later in
    # afternoon. Push Jun-Sep to T-6h (= 18:00 local) from default T-8h.
    "KMIA": {6: 6, 7: 6, 8: 6, 9: 6},

    # OEJN (Jeddah, 21.5°N): extreme desert summer May-Sep, peaks early
    # afternoon but heat persists late. Default T-9h is very early — push
    # to T-7h May-Sep to catch tail-warming days.
    "OEJN": {5: 7, 6: 7, 7: 7, 8: 7, 9: 7},

    # RPLL (Manila, 14.6°N): hottest Mar-May (dry/hot season),
    # cooler-wet Jun-Oct. Push Mar-May to T-7h (= 17:00 local) for the
    # late afternoon heat surges.
    "RPLL": {3: 7, 4: 7, 5: 7},
}


def trigger_t_minus_h(station_id: str, target_date: "date | None" = None) -> float:
    """Per-station trigger hours before station-local midnight.

    With `target_date`, checks `SEASONAL_TRIGGER_OVERRIDES` first, then
    falls back to the annual `PEAK_TRIGGER_T_MINUS_H` value, then to
    `DEFAULT_TRIGGER_T_MINUS_H`. Backward-compatible: callers that don't
    pass target_date get the annual default (today's pre-2026-05-13 behavior).

    Returns float to support fractional triggers (T-5.5h, T-9.8h, etc).
    Int values are also valid and treated as float for downstream comparison.
    """
    if target_date is not None:
        season_map = SEASONAL_TRIGGER_OVERRIDES.get(station_id)
        if season_map is not None:
            override = season_map.get(target_date.month)
            if override is not None:
                return float(override)
    return float(PEAK_TRIGGER_T_MINUS_H.get(station_id, DEFAULT_TRIGGER_T_MINUS_H))

DecisionType = Literal[
    "BUY",             # peak-based: placed a BUY on the bucket containing the peak
    "BUY_EARLY_TAIL",  # tail-based: peak crossed a tail bucket threshold; locked-in winner
    "PRE_TRIGGER",     # local hour < trigger AND no tail crossing yet
    "NO_OBS",          # METAR fetch failed or no observations yet
    "NO_EVENT",        # no Polymarket event for this (station, target, date)
    "NO_BUCKET",       # peak doesn't fall in any logged bucket
    "EXTREME_PRICE",   # winning bucket already at >=0.97 ask, no margin to capture
    "ALREADY_LOGGED",  # decision exists in log for this (station, target_date)
]


@dataclass
class IntradayDecision:
    """One logged decision from a single intraday scan tick."""

    scan_time_utc: str               # ISO datetime
    station_id: str
    target: str                       # "max" only for now
    target_date: str                  # ISO date
    trigger_t_minus_h: float          # per-station trigger used (fractional OK)

    decision: DecisionType
    reason: str = ""

    # Populated when past trigger
    peak_so_far_c: float | None = None
    peak_at_local_hour: int | None = None
    n_observations_used: int = 0

    # Populated when we identified a winning bucket
    winning_bucket_kind: str | None = None
    winning_bucket_threshold: int | None = None
    winning_bucket_label: str | None = None

    # Populated when we matched a Polymarket event
    event_slug: str | None = None
    event_id: int | None = None
    yes_token_id: str | None = None
    market_yes_ask: float | None = None
    market_yes_bid: float | None = None

    # Populated for BUY decisions — paper-trade only
    hypothetical_size_usd: float | None = None
    hypothetical_shares: float | None = None

    # Filled by resolve step (later script)
    actual_obs_c: float | None = None
    bucket_actually_won: bool | None = None
    realized_pnl_usd: float | None = None
    resolved_at_utc: str | None = None

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, d: dict) -> "IntradayDecision":
        return cls(**d)


def trigger_local_hour(station_id: str, target_date: "date | None" = None) -> float:
    """Station-local hour (fractional) at which the trigger fires.

    Examples:
      T-6h    → 24 - 6  = 18.0  (18:00 local)
      T-6.5h  → 24 - 6.5 = 17.5 (17:30 local)
      T-9.8h  → 24 - 9.8 = 14.2 (14:12 local)
    """
    return 24.0 - trigger_t_minus_h(station_id, target_date)


def is_past_trigger(
    station_id: str,
    local_hour: int | float,
    target_date: "date | None" = None,
    local_minute: int = 0,
) -> bool:
    """Has station-local time crossed the per-station trigger hour?

    For fractional triggers, pass `local_minute` for precision.
    With minute=0 (default), comparison is to the trigger HOUR;
    so a T-6.5h trigger (= 17:30 local) considered "past" at local hour 18.
    """
    current = float(local_hour) + (float(local_minute) / 60.0)
    return current >= trigger_local_hour(station_id, target_date)


# ──────────────────────────────────────────────────────────────────────────
# Early-tail detection (added 2026-05-12)
#
# For high_tail buckets on max markets and low_tail on min markets, a single
# METAR reading that crosses the threshold locks in the bucket as a winner.
# Daily max is monotonically non-decreasing as the day progresses, so once we
# observe ≥ high_tail.threshold, the daily max can only stay ≥ threshold
# (or go higher — both keep the bucket winning). Mirror logic for min/low_tail.
#
# This fires HOURS before the peak-based T-6h trigger would, capturing alpha
# the peak-based logic misses (brief threshold crossings that fall back,
# OR just earlier certainty for the orderbook to refresh against).
# ──────────────────────────────────────────────────────────────────────────


def metar_has_critical_gap(
    df,  # pd.DataFrame with local_dt + temp_c columns
    target: str,
    threshold_minutes: float = 90.0,
) -> tuple[bool, float, str | None]:
    """Detect a significant METAR observation gap during peak/trough hours.

    `find_early_tail_winner` relies on the assumption that the daily
    extreme is monotonic — peak can only stay flat or rise as the day
    progresses. This breaks if METAR has gaps (station outage, network
    interruption, ASOS sensor failure): the true peak might have
    occurred during the gap and we wouldn't observe it.

    Critical-hours windows:
      max target: 10am-6pm local — typical heating curve peak window
      min target: 1am-7am local  — typical pre-dawn trough window

    A gap > `threshold_minutes` (default 90) during these hours means
    early-tail detection is untrustworthy for this scan. Caller should
    skip the early-tail path and fall through to the peak-based
    trigger which doesn't rely on the monotonicity invariant.

    Returns (has_critical_gap, max_gap_minutes, gap_window_str).
    """
    if df is None:
        return True, float("inf"), "no dataframe"
    if df.empty or len(df) < 2:
        return True, float("inf"), "no data or single observation"

    # Critical hours window per target
    if target == "max":
        critical_start, critical_end = 10, 18  # 10am-6pm local
    else:
        critical_start, critical_end = 1, 7  # 1am-7am local

    # Sort defensively (caller usually sorts but cheap to verify)
    df_sorted = df.sort_values("local_dt").reset_index(drop=True)

    max_gap_min = 0.0
    gap_at: str | None = None
    for i in range(1, len(df_sorted)):
        t_prev = df_sorted.iloc[i - 1]["local_dt"]
        t_curr = df_sorted.iloc[i]["local_dt"]
        try:
            gap_min = (t_curr - t_prev).total_seconds() / 60.0
        except Exception:
            continue
        # Only count gaps that overlap the critical hours window
        try:
            prev_hour = t_prev.hour
            curr_hour = t_curr.hour
        except AttributeError:
            continue
        overlaps_critical = (
            prev_hour <= critical_end and curr_hour >= critical_start
        )
        if overlaps_critical and gap_min > max_gap_min:
            max_gap_min = gap_min
            try:
                gap_at = (
                    f"{t_prev.strftime('%H:%M')} → "
                    f"{t_curr.strftime('%H:%M')}"
                )
            except Exception:
                gap_at = "unknown"

    return (max_gap_min > threshold_minutes, max_gap_min, gap_at)


def find_early_tail_winner(
    extreme_so_far_c: float,
    target: str,
    bucket_kinds_thresholds: list[tuple[str, int]],
    unit: str,
) -> tuple[str, int] | None:
    """Check if any tail bucket has been locked in by observations so far.

    Args:
        extreme_so_far_c: max-so-far (target="max") or min-so-far (target="min")
            in Celsius — the bot's canonical unit for observation comparison.
        target: "max" or "min".
        bucket_kinds_thresholds: list of (kind, threshold) for this event's buckets.
            Threshold is in the MARKET's unit (°C or °F, per `unit`).
        unit: market unit ("C" or "F"). Determines how `extreme_so_far_c`
            compares to threshold — see `_rounded_observation` in `pnl.py`.

    Returns (kind, threshold) of the locked-in tail bucket, or None if no
    tail crossing has occurred yet.

    Asymmetry: max → only high_tail can lock in early; min → only low_tail.
    Contralateral tails (low_tail on max, high_tail on min) need the full
    day to confirm and are NOT covered by this function.
    """
    from .pnl import _rounded_observation, bucket_won

    actual_int = _rounded_observation(extreme_so_far_c, unit)  # type: ignore[arg-type]
    relevant_kind = "high_tail" if target == "max" else "low_tail"

    for kind, threshold in bucket_kinds_thresholds:
        if kind != relevant_kind:
            continue
        # bucket_won handles °C/°F semantics for the kind correctly
        if bucket_won(kind, threshold, actual_int, unit):  # type: ignore[arg-type]
            return (kind, threshold)
    return None


# ──────────────────────────────────────────────────────────────────────────
# Log I/O
# ──────────────────────────────────────────────────────────────────────────


def load_intraday_log(path: Path = DEFAULT_INTRADAY_LOG_PATH) -> list[IntradayDecision]:
    """Load all decisions from the JSONL log. Missing file returns []."""
    if not path.exists():
        return []
    out: list[IntradayDecision] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(IntradayDecision.from_jsonable(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # skip malformed lines
    return out


def append_intraday_decision(
    decision: IntradayDecision,
    path: Path = DEFAULT_INTRADAY_LOG_PATH,
) -> None:
    """Append a single decision to the JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision.to_jsonable()) + "\n")


def already_decided(
    station_id: str,
    target: str,
    target_date: str,
    log: list[IntradayDecision],
) -> bool:
    """Has a terminal decision already been logged for this (station, target,
    target_date)? Used to make the cron idempotent — once we've made a
    decision we won't revise (e.g. BUY), subsequent scans the same day skip
    without re-firing.

    Retryable (NOT terminal): PRE_TRIGGER, NO_OBS, EXTREME_PRICE. The first
    two are transient. EXTREME_PRICE means the winning bucket is currently
    >=$0.97 ask; price can drop later in the day, so we want to keep
    re-checking until either the price reverts (BUY) or the day ends.
    """
    retryable = {"PRE_TRIGGER", "NO_OBS", "EXTREME_PRICE"}
    for d in log:
        if (d.station_id == station_id
            and d.target == target
            and d.target_date == target_date
            and d.decision not in retryable):
            return True
    return False
