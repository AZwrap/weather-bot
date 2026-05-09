"""Forward-log harness: persist live forecasts and resolve them later.

Production validation cycle:
  1. Each evening, `log_forecasts.py` fetches the live ensemble for every
     (station, target) market in the registry and appends a `ForwardLogRecord`
     with the predictive distribution for tomorrow's daily max/min.
  2. ERA5 archive observations lag by ~5 days. Once a target date is in the
     past long enough, `resolve_log.py` fills `actual_obs_c` and
     `resolved_at_utc` on the record.
  3. `analyze_log.py` walks the resolved subset and reports calibration:
     per-station bias, MAE, RMSE, CRPS — measured on TRUE 1-day-lead
     forecasts (not the historical-forecast-api proxy used to train the
     bias table).

The log is JSONL — one record per line, append-only writes, full-file
rewrite when resolving. Storage cost is trivial (~few KB per day).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast.fetcher import DailyAgg

DEFAULT_LOG_PATH = Path("data/forward_log.jsonl")
DEFAULT_INFLATE_RESAMPLE = 10
DEFAULT_INFLATE_SEED = 0


@dataclass
class BucketSnapshot:
    """Per-bucket forecast + market state at issue time."""

    bucket_label: str                 # "10°C or below", "14°C", etc.
    kind: str                         # "low_tail", "mid", "high_tail"
    threshold: int                    # integer threshold parsed from label
    our_prob: float                   # our predicted probability for this bucket
    yes_bid: float | None             # market best bid for YES
    yes_ask: float | None             # market best ask for YES
    yes_last: float | None            # last trade price for YES
    volume_24hr: float
    market_id: int
    yes_token_id: str
    no_token_id: str

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, d: dict) -> "BucketSnapshot":
        return cls(
            bucket_label=d["bucket_label"],
            kind=d["kind"],
            threshold=int(d["threshold"]),
            our_prob=float(d["our_prob"]),
            yes_bid=(float(d["yes_bid"]) if d.get("yes_bid") is not None else None),
            yes_ask=(float(d["yes_ask"]) if d.get("yes_ask") is not None else None),
            yes_last=(float(d["yes_last"]) if d.get("yes_last") is not None else None),
            volume_24hr=float(d.get("volume_24hr", 0.0)),
            market_id=int(d["market_id"]),
            yes_token_id=str(d["yes_token_id"]),
            no_token_id=str(d["no_token_id"]),
        )


@dataclass
class ForwardLogRecord:
    issue_time_utc: datetime          # when this prediction was generated
    target_date: date                 # date we're forecasting for (station-local)
    station_id: str
    target: DailyAgg                  # "max" or "min"
    model: str                        # ensemble model id, e.g. "ecmwf_ifs025"

    # Forecast inputs (compact: just the raw 51 members + the calibration params)
    raw_members_c: list[float]        # ensemble members BEFORE bias correction
    bias_applied_c: float             # bias subtracted to get bias-corrected members
    sigma_residual_c: float           # σ used for inflation, from BiasTable

    # Derived diagnostics (cheap to recompute, but useful at glance)
    sigma_ensemble_c: float           # std of raw_members - bias
    sigma_total_c: float              # √(σ_ens² + σ_residual²)
    p10: float                        # 10th percentile of inflated predictive
    p50: float
    p90: float

    # Linked Polymarket event + per-bucket market snapshot at issue time.
    # `None` when no matching active market exists for this (station, target,
    # target_date) tuple — e.g. station forecast is logged but Polymarket
    # hasn't listed that market yet.
    event_slug: str | None = None
    event_id: int | None = None
    bucket_snapshots: list[BucketSnapshot] | None = None

    # Filled in later by resolve_log
    actual_obs_c: float | None = None
    resolved_at_utc: datetime | None = None

    # Polymarket's own resolution (the winning bucket label per the gamma API).
    # Filled by resolve_log when the underlying event has closed and one of
    # its bucket markets has yes_price ≥ 0.5. Used to cross-check our rounding
    # rule and the ERA5-vs-station-truth gap.
    polymarket_won_bucket: str | None = None
    polymarket_won_threshold: int | None = None

    @property
    def is_resolved(self) -> bool:
        return self.actual_obs_c is not None

    @property
    def predictive_mean_c(self) -> float:
        return float(np.mean(self.raw_members_c)) - self.bias_applied_c

    def inflated_members(
        self,
        n_resample: int = DEFAULT_INFLATE_RESAMPLE,
        rng_seed: int = DEFAULT_INFLATE_SEED,
    ) -> np.ndarray:
        """Reconstruct the predictive samples from stored params."""
        members = np.asarray(self.raw_members_c, dtype=float) - self.bias_applied_c
        if self.sigma_residual_c <= 0 or n_resample <= 0:
            return members
        rng = np.random.default_rng(rng_seed)
        tiled = np.tile(members, max(1, n_resample))
        noise = rng.normal(0.0, self.sigma_residual_c, size=tiled.shape)
        return tiled + noise

    # ── Serialisation ───────────────────────────────────────────────────

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["issue_time_utc"] = self.issue_time_utc.isoformat()
        d["target_date"] = self.target_date.isoformat()
        d["resolved_at_utc"] = (
            self.resolved_at_utc.isoformat() if self.resolved_at_utc else None
        )
        # bucket_snapshots is already serialised by asdict()
        return d

    @classmethod
    def from_jsonable(cls, d: dict) -> "ForwardLogRecord":
        snaps_raw = d.get("bucket_snapshots")
        snaps = (
            [BucketSnapshot.from_jsonable(s) for s in snaps_raw]
            if snaps_raw is not None
            else None
        )
        return cls(
            issue_time_utc=datetime.fromisoformat(d["issue_time_utc"]),
            target_date=date.fromisoformat(d["target_date"]),
            station_id=d["station_id"],
            target=d["target"],
            model=d["model"],
            raw_members_c=list(d["raw_members_c"]),
            bias_applied_c=float(d["bias_applied_c"]),
            sigma_residual_c=float(d["sigma_residual_c"]),
            sigma_ensemble_c=float(d["sigma_ensemble_c"]),
            sigma_total_c=float(d["sigma_total_c"]),
            p10=float(d["p10"]),
            p50=float(d["p50"]),
            p90=float(d["p90"]),
            event_slug=d.get("event_slug"),
            event_id=(int(d["event_id"]) if d.get("event_id") is not None else None),
            bucket_snapshots=snaps,
            actual_obs_c=(
                float(d["actual_obs_c"]) if d.get("actual_obs_c") is not None else None
            ),
            resolved_at_utc=(
                datetime.fromisoformat(d["resolved_at_utc"])
                if d.get("resolved_at_utc")
                else None
            ),
            polymarket_won_bucket=d.get("polymarket_won_bucket"),
            polymarket_won_threshold=(
                int(d["polymarket_won_threshold"])
                if d.get("polymarket_won_threshold") is not None
                else None
            ),
        )


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────


def load_records(path: Path = DEFAULT_LOG_PATH) -> list[ForwardLogRecord]:
    """Load all records from the JSONL log. Missing file returns []."""
    if not path.exists():
        return []
    out: list[ForwardLogRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(ForwardLogRecord.from_jsonable(json.loads(line)))
    return out


def append_record(record: ForwardLogRecord, path: Path = DEFAULT_LOG_PATH) -> None:
    """Append a single record to the log. Creates the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_jsonable()) + "\n")


def write_all_records(
    records: list[ForwardLogRecord], path: Path = DEFAULT_LOG_PATH
) -> None:
    """Rewrite the entire log. Used by resolve_log to update in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_jsonable()) + "\n")
    tmp.replace(path)


def existing_keys(records: list[ForwardLogRecord]) -> set[tuple[str, str, date, date]]:
    """Set of (station_id, target, target_date, issue_utc_date) for daily dedup."""
    return {
        (r.station_id, r.target, r.target_date, r.issue_time_utc.astimezone(timezone.utc).date())
        for r in records
    }


def existing_keys_hourly(
    records: list[ForwardLogRecord],
) -> set[tuple[str, str, date, str]]:
    """Set of (station_id, target, target_date, issue_utc_yyyymmddhh) for
    hourly dedup. Allows multiple snapshots per UTC day, one per hour."""
    return {
        (
            r.station_id, r.target, r.target_date,
            r.issue_time_utc.astimezone(timezone.utc).strftime("%Y%m%d%H"),
        )
        for r in records
    }
