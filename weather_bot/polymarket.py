"""Polymarket gamma API client (read-only, public endpoints).

Discovers active weather events and parses each into bucket-level binary
markets the bot can score against its own probability model. Order placement
lives in a separate module — this file deliberately stays read-only.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone as tz
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

import httpx

from .locations import STATION_BY_CITY, STATIONS_BY_ID, Station

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
EventTarget = Literal["highest", "lowest"]
BucketKind = Literal["low_tail", "mid", "high_tail"]


@dataclass
class PolymarketMarket:
    """A single binary outcome (one bucket) inside an event."""

    market_id: int
    question: str
    bucket_label: str           # "10°C or below", "14°C", "20°C or higher", etc.
    threshold_index: int        # 0-indexed position from groupItemThreshold
    yes_token_id: str
    no_token_id: str
    yes_price: float | None     # outcomePrices[0] (last consensus price)
    yes_bid: float | None       # bestBid
    yes_ask: float | None       # bestAsk
    last_trade_price: float | None
    volume_total: float
    volume_24hr: float

    @property
    def yes_implied(self) -> float | None:
        """Best estimate of the market's implied yes-probability.

        Mid of best bid/ask if both exist; else last trade; else outcomePrices.
        """
        if self.yes_bid is not None and self.yes_ask is not None:
            return (float(self.yes_bid) + float(self.yes_ask)) / 2
        if self.last_trade_price is not None:
            return float(self.last_trade_price)
        return self.yes_price


@dataclass
class PolymarketEvent:
    event_id: int
    slug: str
    title: str
    end_date: datetime          # UTC, when the market closes
    resolution_url: str | None
    target: EventTarget
    markets: list[PolymarketMarket]
    volume_24hr: float


# ──────────────────────────────────────────────────────────────────────────
# Fetching
# ──────────────────────────────────────────────────────────────────────────


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_market(raw: dict) -> PolymarketMarket:
    # Some fields come back as JSON strings (Polymarket quirk)
    prices_raw = raw.get("outcomePrices")
    if isinstance(prices_raw, str):
        try:
            prices = json.loads(prices_raw)
        except json.JSONDecodeError:
            prices = []
    else:
        prices = prices_raw or []

    tokens_raw = raw.get("clobTokenIds")
    if isinstance(tokens_raw, str):
        try:
            tokens = json.loads(tokens_raw)
        except json.JSONDecodeError:
            tokens = []
    else:
        tokens = tokens_raw or []

    yes_price = _safe_float(prices[0]) if len(prices) > 0 else None
    yes_token = tokens[0] if len(tokens) > 0 else ""
    no_token = tokens[1] if len(tokens) > 1 else ""

    return PolymarketMarket(
        market_id=int(raw["id"]),
        question=raw.get("question", ""),
        bucket_label=raw.get("groupItemTitle") or "",
        threshold_index=int(raw.get("groupItemThreshold") or 0),
        yes_token_id=str(yes_token),
        no_token_id=str(no_token),
        yes_price=yes_price,
        yes_bid=_safe_float(raw.get("bestBid")),
        yes_ask=_safe_float(raw.get("bestAsk")),
        last_trade_price=_safe_float(raw.get("lastTradePrice")),
        volume_total=_safe_float(raw.get("volume")) or 0.0,
        volume_24hr=_safe_float(raw.get("volume24hr")) or 0.0,
    )


def _parse_event(raw: dict, target: EventTarget) -> PolymarketEvent | None:
    end_iso = raw.get("endDate")
    if not end_iso:
        return None
    try:
        end_date = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    markets = [_parse_market(m) for m in raw.get("markets", []) if m.get("id")]
    if not markets:
        return None
    return PolymarketEvent(
        event_id=int(raw["id"]),
        slug=raw.get("slug", ""),
        title=raw.get("title", ""),
        end_date=end_date,
        resolution_url=raw.get("resolutionSource") or None,
        target=target,
        markets=sorted(markets, key=lambda m: m.threshold_index),
        volume_24hr=_safe_float(raw.get("volume24hr")) or 0.0,
    )


async def fetch_temperature_events(
    target: EventTarget,
    client: httpx.AsyncClient | None = None,
    page_size: int = 200,
) -> list[PolymarketEvent]:
    """Fetch all active highest- or lowest-temperature events, paginated."""
    tag_slug = "highest-temperature" if target == "highest" else "lowest-temperature"

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=30.0)

    out: list[PolymarketEvent] = []
    try:
        offset = 0
        while True:
            r = await client.get(
                f"{GAMMA_BASE}/events",
                params={
                    "tag_slug": tag_slug,
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                },
            )
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            for raw in page:
                ev = _parse_event(raw, target)
                if ev is not None:
                    out.append(ev)
            if len(page) < page_size:
                break
            offset += page_size
    finally:
        if owns:
            await client.aclose()
    return out


async def fetch_all_temperature_events(
    client: httpx.AsyncClient | None = None,
) -> list[PolymarketEvent]:
    """Fetch both highest and lowest events concurrently."""
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        max_events, min_events = await asyncio.gather(
            fetch_temperature_events("highest", client),
            fetch_temperature_events("lowest", client),
        )
    finally:
        if owns:
            await client.aclose()
    return max_events + min_events


# ──────────────────────────────────────────────────────────────────────────
# CLOB orderbook prices (more accurate than gamma's cached bestBid/bestAsk)
# ──────────────────────────────────────────────────────────────────────────


async def fetch_clob_prices_batch(
    yes_token_ids: list[str],
    client: httpx.AsyncClient | None = None,
    *,
    chunk_size: int = 100,
) -> dict[str, tuple[float | None, float | None]]:
    """Batch-fetch best (bid, ask) for the YES side of each token via CLOB.

    Returns {yes_token_id: (yes_bid, yes_ask)}. Missing or error tokens get
    (None, None). The CLOB `/prices` POST endpoint accepts a list of
    {token_id, side} pairs and returns {token_id: {BUY: x, SELL: y}}, where
    BUY = best bid and SELL = best ask. Limit is undocumented; we chunk to
    `chunk_size` tokens per request (= 2*chunk_size pairs) for safety.
    """
    out: dict[str, tuple[float | None, float | None]] = {tid: (None, None) for tid in yes_token_ids}
    if not yes_token_ids:
        return out

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=30.0)

    try:
        for i in range(0, len(yes_token_ids), chunk_size):
            chunk = yes_token_ids[i:i + chunk_size]
            body = []
            for tid in chunk:
                body.append({"token_id": tid, "side": "BUY"})
                body.append({"token_id": tid, "side": "SELL"})
            try:
                r = await client.post(f"{CLOB_BASE}/prices", json=body)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                print(f"!! CLOB /prices chunk {i}-{i+len(chunk)}: {exc}")
                continue
            for tid, sides in (data or {}).items():
                bid = sides.get("BUY")
                ask = sides.get("SELL")
                out[tid] = (
                    float(bid) if bid not in (None, "", "0") else None,
                    float(ask) if ask not in (None, "", "0") else None,
                )
    finally:
        if owns:
            await client.aclose()
    return out


def apply_clob_prices(
    events: list[PolymarketEvent],
    fresh: dict[str, tuple[float | None, float | None]],
) -> tuple[int, int]:
    """Mutate events in place to use CLOB bid/ask where available.

    Returns (n_updated_markets, n_unchanged_markets) for telemetry.
    """
    n_updated = 0
    n_unchanged = 0
    for ev in events:
        for m in ev.markets:
            tid = m.yes_token_id
            if not tid or tid not in fresh:
                n_unchanged += 1
                continue
            bid, ask = fresh[tid]
            changed = False
            if bid is not None and bid != m.yes_bid:
                m.yes_bid = bid
                changed = True
            if ask is not None and ask != m.yes_ask:
                m.yes_ask = ask
                changed = True
            if changed:
                n_updated += 1
            else:
                n_unchanged += 1
    return n_updated, n_unchanged


# ──────────────────────────────────────────────────────────────────────────
# Parsing helpers — event → station, bucket → integer threshold
# ──────────────────────────────────────────────────────────────────────────


_TITLE_RE = re.compile(
    r"^(?:Highest|Lowest) temperature in (.+?) on ", re.IGNORECASE
)


def match_event_to_station(event: PolymarketEvent) -> Station | None:
    """Resolve an event to a station in our registry.

    Tries the resolution URL's last path segment (Wunderground ICAO format)
    first; falls back to the event title's city name.
    """
    if event.resolution_url:
        last = event.resolution_url.rstrip("/").rsplit("/", 1)[-1]
        if last in STATIONS_BY_ID:
            return STATIONS_BY_ID[last]

    m = _TITLE_RE.match(event.title)
    if m:
        city = m.group(1).strip().lower()
        if city in STATION_BY_CITY:
            return STATION_BY_CITY[city]
    return None


def event_target_date(event: PolymarketEvent, station: Station) -> date:
    """Derive the resolution date in the station's local timezone."""
    return event.end_date.astimezone(ZoneInfo(station.timezone)).date()


_BUCKET_INT_RE = re.compile(r"-?\d+")


def parse_bucket(market: PolymarketMarket) -> tuple[BucketKind, int]:
    """Return (kind, integer_threshold) parsed from the bucket label.

    Examples (°C and °F use the same int — `Station.unit` disambiguates):
        "10°C or below"     → ("low_tail",  10)
        "14°C"              → ("mid",       14)
        "20°C or higher"    → ("high_tail", 20)
        "70°F or below"     → ("low_tail",  70)
    """
    label_lc = market.bucket_label.lower()
    nums = _BUCKET_INT_RE.findall(market.bucket_label)
    threshold = int(nums[0]) if nums else market.threshold_index

    if "or below" in label_lc:
        return "low_tail", threshold
    if "or higher" in label_lc or "or above" in label_lc:
        return "high_tail", threshold
    return "mid", threshold


def group_events_by_station(
    events: Iterable[PolymarketEvent],
) -> dict[str, list[PolymarketEvent]]:
    """Bucket events by station_id (skipping unmatchable ones)."""
    out: dict[str, list[PolymarketEvent]] = {}
    for ev in events:
        st = match_event_to_station(ev)
        if st is None:
            continue
        out.setdefault(st.station_id, []).append(ev)
    return out
