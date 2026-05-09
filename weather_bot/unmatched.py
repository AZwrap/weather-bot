"""Track Polymarket weather events that don't match any station in our registry.

The hourly cron picks up *new dates* for known stations automatically, but if
Polymarket starts listing a *new city* (Mumbai, Lima, Athens, etc.), the
matcher returns None and the event is silently skipped. To avoid both
silently missing markets AND auto-adding stations with wrong coords/units,
this module logs unmatched events for the user to triage.

Storage: append-only JSONL at `data/unmatched_events.jsonl`. One line per
unmatched (event, observation_time). Cheap to grow; the analyser dedups
by event_slug at read time.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .locations import EXCLUDED_CITIES_LOWER, EXCLUDED_STATION_IDS
from .polymarket import PolymarketEvent

DEFAULT_UNMATCHED_PATH = Path("data/unmatched_events.jsonl")

_TITLE_RE = re.compile(r"^(?:Highest|Lowest) temperature in (.+?) on ", re.IGNORECASE)


@dataclass
class UnmatchedEvent:
    observed_at_utc: datetime
    event_slug: str
    event_title: str
    event_id: int
    target: str                # "highest" | "lowest"
    resolution_url: str | None
    url_icao: str | None       # last URL path segment if it looks like an ICAO
    volume_24hr: float
    city_parsed: str | None    # city name extracted from event title

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["observed_at_utc"] = self.observed_at_utc.isoformat()
        return d

    @classmethod
    def from_jsonable(cls, d: dict) -> "UnmatchedEvent":
        return cls(
            observed_at_utc=datetime.fromisoformat(d["observed_at_utc"]),
            event_slug=d.get("event_slug", ""),
            event_title=d.get("event_title", ""),
            event_id=int(d.get("event_id") or 0),
            target=d.get("target", ""),
            resolution_url=d.get("resolution_url"),
            url_icao=d.get("url_icao"),
            volume_24hr=float(d.get("volume_24hr") or 0.0),
            city_parsed=d.get("city_parsed"),
        )


def _extract_city(title: str) -> str | None:
    m = _TITLE_RE.match(title)
    return m.group(1).strip() if m else None


def _extract_url_icao(url: str | None) -> str | None:
    """Return the last path segment of the URL if it looks like an ICAO (4 caps).

    Wunderground URLs are like .../pk/karachi/OPKC — the trailing OPKC is the
    ICAO, useful as a hint for what station Polymarket might be using.
    """
    if not url:
        return None
    last = url.rstrip("/").rsplit("/", 1)[-1]
    return last if (len(last) == 4 and last.isupper() and last.isalpha()) else None


def make_unmatched_record(
    event: PolymarketEvent, observed_at: datetime
) -> UnmatchedEvent:
    return UnmatchedEvent(
        observed_at_utc=observed_at.astimezone(timezone.utc),
        event_slug=event.slug,
        event_title=event.title,
        event_id=event.event_id,
        target=event.target,
        resolution_url=event.resolution_url,
        url_icao=_extract_url_icao(event.resolution_url),
        volume_24hr=event.volume_24hr,
        city_parsed=_extract_city(event.title),
    )


def append_unmatched(
    records: Iterable[UnmatchedEvent], path: Path = DEFAULT_UNMATCHED_PATH
) -> int:
    """Append records to the JSONL log. Returns count appended."""
    records = list(records)
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_jsonable()) + "\n")
    return len(records)


def load_unmatched(path: Path = DEFAULT_UNMATCHED_PATH) -> list[UnmatchedEvent]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(UnmatchedEvent.from_jsonable(json.loads(line)))
    return out


@dataclass
class CitySummary:
    city: str
    n_observations: int
    first_seen_utc: datetime
    last_seen_utc: datetime
    sample_event_slug: str
    sample_url: str | None
    sample_url_icao: str | None
    targets: set[str]
    max_volume_24hr: float

    @property
    def days_seen(self) -> float:
        return (self.last_seen_utc - self.first_seen_utc).total_seconds() / 86400


def summarise_by_city(
    records: Iterable[UnmatchedEvent],
    since_utc: datetime | None = None,
    exclude_cities: frozenset[str] | None = None,
    exclude_station_ids: frozenset[str] | None = None,
) -> list[CitySummary]:
    """Group unmatched records by parsed city name. Filter to ≥ since_utc if given.

    `exclude_cities` (lowercase) and `exclude_station_ids` suppress entries
    we've intentionally chosen not to trade — defaults pull from
    `weather_bot.locations.EXCLUDED_*` so HKO won't trigger perpetual alerts.
    """
    if exclude_cities is None:
        exclude_cities = EXCLUDED_CITIES_LOWER
    if exclude_station_ids is None:
        exclude_station_ids = EXCLUDED_STATION_IDS

    by_city: dict[str, list[UnmatchedEvent]] = {}
    for r in records:
        if since_utc and r.observed_at_utc < since_utc:
            continue
        city_lc = (r.city_parsed or "").lower().strip()
        if city_lc and city_lc in exclude_cities:
            continue
        if r.url_icao and r.url_icao in exclude_station_ids:
            continue
        key = (r.city_parsed or r.event_slug or r.event_title or "?").strip()
        by_city.setdefault(key, []).append(r)

    out: list[CitySummary] = []
    for city, items in by_city.items():
        items.sort(key=lambda r: r.observed_at_utc)
        sample = items[-1]
        out.append(CitySummary(
            city=city,
            n_observations=len(items),
            first_seen_utc=items[0].observed_at_utc,
            last_seen_utc=items[-1].observed_at_utc,
            sample_event_slug=sample.event_slug,
            sample_url=sample.resolution_url,
            sample_url_icao=sample.url_icao,
            targets={r.target for r in items},
            max_volume_24hr=max(r.volume_24hr for r in items),
        ))
    out.sort(key=lambda s: -s.last_seen_utc.timestamp())
    return out
