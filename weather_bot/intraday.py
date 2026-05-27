"""METAR early-tail strategy — lite rebuild.

This module is the EARLY-TAIL ONLY portion of the original
`weather_bot/intraday.py`. The peak-based T-6h branch (and its
per-station / per-month trigger tuning) was removed in the lite
rebuild on 2026-05-27 — it bled empirically and the user opted to
drop it.

Early-tail trigger (kept):
  For high_tail buckets on max markets (and low_tail on min markets),
  a single METAR reading that crosses the threshold is sufficient to
  lock in the bucket as a winner. Daily max is monotonically non-
  decreasing as the day progresses, so once we observe
  >= high_tail.threshold, the daily max can only stay >= threshold or
  go higher — both keep the bucket winning. Mirror logic applies for
  min target + low_tail.

  This is the only METAR firing condition in the lite rebuild. It
  corresponds to the "early-tail / locked-in math" cohort, which had
  0% FP rate in the original measurement (N=219 fires across 5 days
  in May 2026 — see `project_metar_fp_rate.md`).

Per scan tick, for each (station, target, target_date=today):
  1. Caller fetches hourly METAR observed-so-far for the station-day.
  2. Caller passes the extreme-so-far + bucket thresholds to
     `find_early_tail_winner()`. If it returns a tail bucket, that
     bucket is locked in.
  3. `metar_has_critical_gap()` is checked first — if there's a gap
     of > 90min during the critical hours, the monotonicity
     assumption is unreliable and the caller should skip.
  4. Log every decision to `data/intraday_log.jsonl` (idempotent
     per-day via `already_decided`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

DEFAULT_INTRADAY_LOG_PATH = Path("data/intraday_log.jsonl")


DecisionType = Literal[
    "BUY_EARLY_TAIL",  # tail bucket crossed; locked-in winner
    "NO_TAIL_CROSSED", # extreme-so-far hasn't reached any tail threshold
    "NO_OBS",          # METAR fetch failed or no observations yet
    "NO_EVENT",        # no Polymarket event for this (station, target, date)
    "CRITICAL_GAP",    # METAR gap > threshold; monotonicity invariant broken
    "EXTREME_PRICE",   # winning bucket already at >= ceiling, no margin
    "ALREADY_LOGGED",  # decision exists in log for this (station, target_date)
]


@dataclass
class IntradayDecision:
    """One logged decision from a single intraday scan tick."""

    scan_time_utc: str               # ISO datetime
    station_id: str
    target: str                       # "max" or "min"
    target_date: str                  # ISO date

    decision: DecisionType
    reason: str = ""

    # Populated when extreme-so-far is known
    extreme_so_far_c: float | None = None
    n_observations_used: int = 0

    # Populated when we identified a locked-in tail bucket
    winning_bucket_kind: str | None = None
    winning_bucket_threshold: int | None = None
    winning_bucket_label: str | None = None

    # Populated when we matched a Polymarket event
    event_slug: str | None = None
    event_id: int | None = None
    yes_token_id: str | None = None
    market_yes_ask: float | None = None
    market_yes_bid: float | None = None

    # Populated for BUY_EARLY_TAIL decisions — paper-trade only by default
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
        # Tolerate legacy fields from the pre-lite-rebuild log format —
        # silently drop any keys we no longer carry.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ──────────────────────────────────────────────────────────────────────────
# Early-tail detection
#
# For high_tail buckets on max markets and low_tail on min markets, a single
# METAR reading that crosses the threshold locks in the bucket as a winner.
# Daily max is monotonically non-decreasing as the day progresses, so once we
# observe >= high_tail.threshold, the daily max can only stay >= threshold
# (or go higher — both keep the bucket winning). Mirror logic for min/low_tail.
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
    skip and re-try on a later cron tick.

    Returns (has_critical_gap, max_gap_minutes, gap_window_str).
    """
    if df is None:
        return True, float("inf"), "no dataframe"
    if df.empty or len(df) < 2:
        return True, float("inf"), "no data or single observation"

    if target == "max":
        critical_start, critical_end = 10, 18
    else:
        critical_start, critical_end = 1, 7

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
        bucket_kinds_thresholds: list of (kind, threshold) for this event's
            buckets. Threshold is in the MARKET's unit (°C or °F per `unit`).
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
                continue
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
    locked-in BUY decision we won't revise.

    Retryable (NOT terminal): NO_OBS, NO_TAIL_CROSSED, CRITICAL_GAP,
    EXTREME_PRICE. These can flip later in the day so we keep re-checking.
    """
    retryable = {"NO_OBS", "NO_TAIL_CROSSED", "CRITICAL_GAP", "EXTREME_PRICE"}
    for d in log:
        if (d.station_id == station_id
            and d.target == target
            and d.target_date == target_date
            and d.decision not in retryable):
            return True
    return False
