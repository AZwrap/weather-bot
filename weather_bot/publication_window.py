"""Publication-window shadow harness.

Forward-looking measurement of the gap between:
  (a) end-of-resolution-day in station-local time (= the moment the
      day's max/min is mathematically determined), and
  (b) Polymarket oracle settlement (= the moment payouts are issued).

The original analysis of May 14-17 trade data shows the median gap is
~4h with p90 around +16h, but used our-own trades only and so couldn't
measure how much of that window remained TRADABLE for any participant.
This harness fills the gap: it snapshots Polymarket bucket prices at
multiple offsets past end-of-day-local for each station/date, plus our
final METAR-derived daily extreme, so we can later answer:

  1. How long does the market remain tradable past end-of-day-local?
  2. Do bucket prices converge to {0, 1} immediately or slowly?
  3. At each offset, was a fast Wunderground-race strategy in the money?

Output JSONL: data/publication_window_log.jsonl.
Idempotency: at most one snapshot per (station, target, target_date,
offset_bin) where offset_bin is the snapshot time rounded down to a
30-min bucket.

This module is read by `analyze_publication_window.py` once a few days
of data have accumulated.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_LOG_PATH = Path("data/publication_window_log.jsonl")

# Per-station UTC offset (mid-2026 DST baseline). Loaded once via
# locations.py; this fallback table is just for stations whose IANA tz
# can't be resolved (unlikely on a working install but defensive).
FALLBACK_OFFSET_H: dict[str, float] = {}


@dataclass
class BucketSnapshot:
    kind: str            # "low_tail" | "mid" | "high_tail"
    threshold: int
    label: str
    yes_token_id: str | None
    no_token_id: str | None
    yes_ask: float | None
    yes_bid: float | None
    no_ask: float | None
    no_bid: float | None
    volume_24h_usd: float | None


@dataclass
class PublicationWindowRecord:
    snapshot_ts_utc: str
    station_id: str
    target: str
    target_date: str
    midend_local_utc: str
    offset_h_after_midend: float

    # METAR-derived (our view of the truth)
    metar_final_extreme_c: float | None = None
    metar_n_observations: int = 0
    metar_max_gap_min: float | None = None
    metar_last_obs_local_iso: str | None = None

    # Polymarket
    event_slug: str | None = None
    event_id: int | None = None
    buckets: list[BucketSnapshot] = field(default_factory=list)
    matched_bucket_kind: str | None = None
    matched_bucket_threshold: int | None = None

    # Filled later by analyzer when forward_log resolves the (station, date)
    settlement_ts_utc: str | None = None
    settlement_obs_c: float | None = None
    settlement_bucket_kind: str | None = None
    settlement_bucket_threshold: int | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def midend_local_utc(target_date: date, station) -> datetime:
    """UTC datetime corresponding to the end of `target_date` in station-local tz."""
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(station.timezone)
        midend_local = datetime(
            target_date.year, target_date.month, target_date.day, tzinfo=tz,
        ) + timedelta(days=1)
        return midend_local.astimezone(timezone.utc)
    except Exception:
        off = FALLBACK_OFFSET_H.get(station.station_id, 0.0)
        midend_local_naive = datetime(
            target_date.year, target_date.month, target_date.day,
        ) + timedelta(days=1)
        return (midend_local_naive - timedelta(hours=off)).replace(tzinfo=timezone.utc)


def _offset_bin_30min(offset_h: float) -> int:
    """Round-down offset hours to integer 30-min bins. -0.4h → -1 bin."""
    import math
    return int(math.floor(offset_h * 2))


def already_logged(
    *,
    station_id: str,
    target: str,
    target_date_iso: str,
    offset_h: float,
    log_path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """Return True if a record already exists at this (station, target,
    date, offset-30min-bin) — used to enforce idempotency under a 30-min cron."""
    if not log_path.exists():
        return False
    want_bin = _offset_bin_30min(offset_h)
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("station_id") == station_id
                and r.get("target") == target
                and r.get("target_date") == target_date_iso
                and _offset_bin_30min(r.get("offset_h_after_midend", -999)) == want_bin):
                return True
    return False


def append(record: PublicationWindowRecord, log_path: Path = DEFAULT_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_jsonable(), default=str) + "\n")


async def snapshot_one(
    *,
    station,
    target: str,          # "max" | "min"
    target_date: date,
    ev,                   # PolymarketEvent
    http: httpx.AsyncClient,
    log_path: Path = DEFAULT_LOG_PATH,
    now_utc: datetime | None = None,
) -> PublicationWindowRecord | None:
    """Capture one snapshot for this (station, target, target_date).

    Returns the record on success, None if we skipped (not yet past
    midend, already-logged for this 30-min bin, or fetch failure).
    """
    from .observations import fetch_metar_hourly_range
    from .pnl import _rounded_observation, bucket_won
    from .polymarket import parse_bucket

    now_utc = now_utc or datetime.now(timezone.utc)
    midend_utc = midend_local_utc(target_date, station)
    offset_h = (now_utc - midend_utc).total_seconds() / 3600.0

    # Only snapshot AFTER end-of-resolution-day-local.
    if offset_h < 0:
        return None

    target_date_iso = target_date.isoformat()
    if already_logged(
        station_id=station.station_id, target=target,
        target_date_iso=target_date_iso, offset_h=offset_h, log_path=log_path,
    ):
        return None

    # Fetch the full station-day of METAR. Past midend, this is the day
    # that just closed.
    df = await fetch_metar_hourly_range(
        location=station.to_location(),
        icao=station.station_id,
        start_date=target_date,
        end_date=target_date,
        client=http,
    )

    metar_final_extreme_c: float | None = None
    metar_n = 0
    metar_max_gap_min: float | None = None
    metar_last_obs_iso: str | None = None
    if df is not None and not df.empty:
        metar_n = len(df)
        df_sorted = df.sort_values("local_dt").reset_index(drop=True)
        if target == "max":
            metar_final_extreme_c = float(df_sorted["temp_c"].max())
        else:
            metar_final_extreme_c = float(df_sorted["temp_c"].min())
        # Largest gap between successive observations
        max_gap = 0.0
        for i in range(1, len(df_sorted)):
            try:
                gap = (df_sorted.iloc[i]["local_dt"]
                       - df_sorted.iloc[i - 1]["local_dt"]).total_seconds() / 60.0
                if gap > max_gap:
                    max_gap = gap
            except Exception:
                continue
        metar_max_gap_min = max_gap
        try:
            metar_last_obs_iso = df_sorted.iloc[-1]["local_dt"].isoformat()
        except Exception:
            pass

    # Snapshot all buckets in the event.
    buckets: list[BucketSnapshot] = []
    matched_kind: str | None = None
    matched_thr: int | None = None
    if metar_final_extreme_c is not None:
        actual_int = _rounded_observation(metar_final_extreme_c, station.unit)
    else:
        actual_int = None

    for m in ev.markets:
        try:
            kind, thr = parse_bucket(m)
        except Exception:
            continue
        buckets.append(BucketSnapshot(
            kind=kind,
            threshold=thr,
            label=getattr(m, "bucket_label", ""),
            yes_token_id=getattr(m, "yes_token_id", None),
            no_token_id=getattr(m, "no_token_id", None),
            yes_ask=getattr(m, "yes_ask", None),
            yes_bid=getattr(m, "yes_bid", None),
            no_ask=getattr(m, "no_ask", None),
            no_bid=getattr(m, "no_bid", None),
            volume_24h_usd=getattr(m, "volume_24hr", None),
        ))
        if actual_int is not None and matched_kind is None:
            if bucket_won(kind, thr, actual_int, station.unit):
                matched_kind, matched_thr = kind, thr

    record = PublicationWindowRecord(
        snapshot_ts_utc=now_utc.isoformat(),
        station_id=station.station_id,
        target=target,
        target_date=target_date_iso,
        midend_local_utc=midend_utc.isoformat(),
        offset_h_after_midend=offset_h,
        metar_final_extreme_c=metar_final_extreme_c,
        metar_n_observations=metar_n,
        metar_max_gap_min=metar_max_gap_min,
        metar_last_obs_local_iso=metar_last_obs_iso,
        event_slug=ev.slug,
        event_id=ev.id,
        buckets=buckets,
        matched_bucket_kind=matched_kind,
        matched_bucket_threshold=matched_thr,
    )
    append(record, log_path)
    return record
