"""Layer 5 — Adaptive observation-distance entry filter (PAPER-LOG ONLY).

Filters NO_momentum placements based on the distance between the
current observed extreme (peak for max-target, trough for min-target)
and the bucket's edges, BEFORE placing the order.

The principle: a peak temperature is monotonically non-decreasing
through the day. If we're placing a NO bet on bucket [low, high), we
benefit from observing where the peak-so-far is RIGHT NOW relative
to the bucket:

For max-target:
  - peak_so_far ≥ high   → bucket DEAD (can't win) → PLACE_GUARANTEED
  - peak_so_far ∈ [low, high) → peak in bucket   → SKIP_IN_BUCKET
  - peak < low, gap < margin → vulnerable        → SKIP_VULNERABLE
  - peak < low - margin  → safe buffer           → PLACE

For min-target (mirror — trough monotonically decreases):
  - trough ≤ low         → bucket DEAD → PLACE_GUARANTEED
  - trough ∈ [low, high) → trough in bucket → SKIP_IN_BUCKET
  - trough > high, gap < margin → vulnerable → SKIP_VULNERABLE
  - trough > high + margin → safe buffer → PLACE

CRITICAL: This module is PAPER-LOG ONLY. The live bot continues to
place orders on ALL eligible buckets regardless of filter output. The
filter's decision is recorded to `data/obs_distance_paper_log.jsonl`
so we can analyze (after ≥7 days of data) whether filter-PLACE
placements have a higher win rate than filter-SKIP placements. Only
after that empirical validation does the filter get wired LIVE.

Pass criterion to go live:
  - filter-PLACE NO win rate ≥ 70% AND filter-SKIP NO win rate ≤ 40%
    over ≥7 trading days.
  - Anything weaker → filter is noise; don't ship.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_OBS_DISTANCE_LOG_PATH = Path("data/obs_distance_paper_log.jsonl")
DEFAULT_SAFETY_MARGIN_C = 2.5
"""Empirical optimum from May 16 calibration (see
analyze_cross_up_distances.py): 2.5°C margin would have caught 68% of
losses with ~3h avg lead time. Plateaus 2.5→3.0; 4.0 adds only one
more save but materially raises false-positive risk. See
project_tomorrow_morning_checklist_2026-05-17.md Q6 for the full table."""


# ──────────────────────────────────────────────────────────────────────────
# Bucket edge math
# ──────────────────────────────────────────────────────────────────────────

# Fahrenheit buckets are 2°F wide.  2°F = (2 * 5/9)°C ≈ 1.111°C
_F_BUCKET_WIDTH_C = 10.0 / 9.0
_C_BUCKET_WIDTH_C = 1.0


def _threshold_to_c(threshold_int: int, unit: str) -> float:
    """Convert a market-unit integer threshold to °C."""
    if unit == "F":
        return (float(threshold_int) - 32.0) * 5.0 / 9.0
    return float(threshold_int)


def bucket_edges_c(
    bucket_kind: str,  # "mid" | "low_tail" | "high_tail"
    threshold_int: int,
    unit: str,
) -> tuple[float, float]:
    """Return (low_c, high_c) for the bucket as a half-open interval [low, high) in °C.

    Conventions:
      - "low_tail"  ("X or below"):   peak ≤ X → (-inf, X+width_c)
      - "mid"       ("X"):            peak ∈ [X, X+width) → (X, X+width_c)
      - "high_tail" ("X or higher"):  peak ≥ X → (X, +inf)

    Width is 1°C for °C markets and ~1.111°C (= 2°F) for °F markets.
    """
    width_c = _C_BUCKET_WIDTH_C if unit == "C" else _F_BUCKET_WIDTH_C
    base_c = _threshold_to_c(threshold_int, unit)

    if bucket_kind == "mid":
        return base_c, base_c + width_c
    if bucket_kind == "low_tail":
        return float("-inf"), base_c + width_c
    if bucket_kind == "high_tail":
        return base_c, float("inf")
    raise ValueError(f"Unknown bucket_kind: {bucket_kind!r}")


# ──────────────────────────────────────────────────────────────────────────
# Filter decision
# ──────────────────────────────────────────────────────────────────────────

DECISION_PLACE = "PLACE"                       # safe buffer; place
DECISION_PLACE_GUARANTEED = "PLACE_GUARANTEED" # bucket dead; place (NO is guaranteed)
DECISION_SKIP_IN_BUCKET = "SKIP_IN_BUCKET"    # extreme is inside bucket; risky
DECISION_SKIP_VULNERABLE = "SKIP_VULNERABLE"  # extreme close to bucket edge; vulnerable


@dataclass
class FilterDecision:
    """One filter evaluation result. Logged to paper log for analysis."""
    decision: str                    # one of DECISION_* constants above
    reason: str                      # short human-readable explanation
    extreme_c: float                 # observed peak (max-target) or trough (min-target)
    bucket_low_c: float              # inclusive lower edge of bucket
    bucket_high_c: float             # exclusive upper edge of bucket
    safety_margin_c: float           # margin applied
    distance_c: float                # signed distance from extreme to relevant bucket edge
                                     # >0: gap to entry (safe direction)
                                     # <0: inside bucket or past it

    def to_jsonable(self) -> dict:
        d = asdict(self)
        # Replace +/-inf with sentinel strings (JSON doesn't support inf)
        if d["bucket_low_c"] == float("-inf"):
            d["bucket_low_c"] = "-inf"
        if d["bucket_high_c"] == float("inf"):
            d["bucket_high_c"] = "+inf"
        return d


def filter_decision(
    *,
    extreme_so_far_c: float,
    bucket_low_c: float,
    bucket_high_c: float,
    target: str,                     # "max" or "min"
    safety_margin_c: float = DEFAULT_SAFETY_MARGIN_C,
) -> FilterDecision:
    """Compute the filter decision for one (extreme, bucket, target) tuple.

    Args:
      extreme_so_far_c: peak (for max-target) or trough (for min-target)
        observed so far today, in °C.
      bucket_low_c, bucket_high_c: half-open bucket interval [low, high) in °C.
      target: "max" or "min".
      safety_margin_c: distance below the bucket's relevant edge within
        which placement is considered too vulnerable. Default 2°C.

    Returns a FilterDecision capturing the full reasoning trace.
    """
    if target == "max":
        # Peak monotonically rises.
        # PLACE_GUARANTEED: peak already past upper edge — bucket cannot win.
        if extreme_so_far_c >= bucket_high_c:
            return FilterDecision(
                decision=DECISION_PLACE_GUARANTEED,
                reason="peak past bucket upper edge",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=extreme_so_far_c - bucket_high_c,
            )
        # SKIP_IN_BUCKET: peak currently inside bucket.
        if extreme_so_far_c >= bucket_low_c:
            return FilterDecision(
                decision=DECISION_SKIP_IN_BUCKET,
                reason="peak currently inside bucket",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=extreme_so_far_c - bucket_low_c,
            )
        # Peak below bucket. Gap measured in °C.
        gap = bucket_low_c - extreme_so_far_c
        if gap < safety_margin_c:
            return FilterDecision(
                decision=DECISION_SKIP_VULNERABLE,
                reason=f"peak {gap:.2f}°C below bucket; margin {safety_margin_c:.2f}°C",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=gap,
            )
        return FilterDecision(
            decision=DECISION_PLACE,
            reason=f"peak {gap:.2f}°C below bucket; safe buffer",
            extreme_c=extreme_so_far_c,
            bucket_low_c=bucket_low_c,
            bucket_high_c=bucket_high_c,
            safety_margin_c=safety_margin_c,
            distance_c=gap,
        )

    elif target == "min":
        # Trough monotonically falls.
        # PLACE_GUARANTEED: trough already past (below) the bucket's lower
        # edge — bucket cannot win (further decrease moves us further past).
        if extreme_so_far_c <= bucket_low_c:
            return FilterDecision(
                decision=DECISION_PLACE_GUARANTEED,
                reason="trough past bucket lower edge",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=bucket_low_c - extreme_so_far_c,
            )
        # SKIP_IN_BUCKET: trough currently inside bucket.
        if extreme_so_far_c < bucket_high_c:
            return FilterDecision(
                decision=DECISION_SKIP_IN_BUCKET,
                reason="trough currently inside bucket",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=bucket_high_c - extreme_so_far_c,
            )
        # Trough above bucket. Gap measured in °C.
        gap = extreme_so_far_c - bucket_high_c
        if gap < safety_margin_c:
            return FilterDecision(
                decision=DECISION_SKIP_VULNERABLE,
                reason=f"trough {gap:.2f}°C above bucket; margin {safety_margin_c:.2f}°C",
                extreme_c=extreme_so_far_c,
                bucket_low_c=bucket_low_c,
                bucket_high_c=bucket_high_c,
                safety_margin_c=safety_margin_c,
                distance_c=gap,
            )
        return FilterDecision(
            decision=DECISION_PLACE,
            reason=f"trough {gap:.2f}°C above bucket; safe buffer",
            extreme_c=extreme_so_far_c,
            bucket_low_c=bucket_low_c,
            bucket_high_c=bucket_high_c,
            safety_margin_c=safety_margin_c,
            distance_c=gap,
        )

    raise ValueError(f"target must be 'max' or 'min', got {target!r}")


# ──────────────────────────────────────────────────────────────────────────
# Paper log
# ──────────────────────────────────────────────────────────────────────────


def log_filter_decision(
    *,
    scan_time_utc: str,
    station_id: str,
    target: str,
    target_date_iso: str,
    bucket_label: str,
    bucket_kind: str,
    threshold_int: int,
    unit: str,
    decision: FilterDecision,
    # Whether we actually placed the order despite the filter decision
    # (paper-log mode: always True until the filter is wired live)
    placed: bool,
    # Optional metadata for debugging
    no_token_id: Optional[str] = None,
    extra: Optional[dict] = None,
    log_path: Path = DEFAULT_OBS_DISTANCE_LOG_PATH,
) -> None:
    """Append one filter-decision record to the paper log (JSONL).

    Caller is responsible for invoking this only on candidates that
    passed all OTHER gates (cap, dedupe, etc.) — we only want to log
    decisions on placements that would have been made anyway.
    """
    record: dict = {
        "scan_time_utc": scan_time_utc,
        "station_id": station_id,
        "target": target,
        "target_date": target_date_iso,
        "bucket_label": bucket_label,
        "bucket_kind": bucket_kind,
        "threshold_int": threshold_int,
        "unit": unit,
        "placed": bool(placed),
        "decision": decision.to_jsonable(),
    }
    if no_token_id is not None:
        record["no_token_id"] = no_token_id
    if extra:
        record["extra"] = extra

    # Swallow all logging errors — paper log must never disrupt the
    # live NO_momentum placement loop. Failures here mean we miss a
    # paper-log entry; bot operation continues normally.
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
