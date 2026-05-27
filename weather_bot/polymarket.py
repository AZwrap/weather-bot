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

# ──────────────────────────────────────────────────────────────────────────
# Exchange-level minimums (2026-05-13):
#   Marketable orders: ≥ $1 notional. No share floor.
#   Resting limits:    ≥ 5 shares. No dollar floor. Can be sub-$1.
# ──────────────────────────────────────────────────────────────────────────

POLYMARKET_DEFAULT_TICK_SIZE: float = 0.001
"""Common tick size on weather markets, BUT tick size varies per market
(0.01, 0.001, 0.0001, or 0.1). Prefer the per-market `tick_size` from
OrderBookDepth in live code."""

POLYMARKET_RESTING_MIN_SHARES: int = 5
"""Minimum for RESTING limit orders. Resting has no dollar floor."""

POLYMARKET_MARKETABLE_MIN_USD: float = 1.0
"""Minimum for MARKETABLE orders. Marketable has no share floor."""


def is_marketable_order(limit_price: float, opposite_top_of_book: float, side: str) -> bool:
    """Return True if a limit order at `limit_price` would immediately cross."""
    if side.upper() == "BUY":
        return limit_price >= opposite_top_of_book
    if side.upper() == "SELL":
        return limit_price <= opposite_top_of_book
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def marketable_passes_min(notional_usd: float) -> bool:
    """Marketable order minimum: ≥ $1 notional. No share floor."""
    return notional_usd >= POLYMARKET_MARKETABLE_MIN_USD


def resting_passes_min(shares: float) -> bool:
    """Resting limit minimum: ≥ 5 shares. No dollar floor."""
    return shares >= POLYMARKET_RESTING_MIN_SHARES


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

    chunks_failed = 0
    chunks_total = 0
    tokens_with_stale_prices: list[str] = []
    try:
        for i in range(0, len(yes_token_ids), chunk_size):
            chunks_total += 1
            chunk = yes_token_ids[i:i + chunk_size]
            body = []
            for tid in chunk:
                body.append({"token_id": tid, "side": "BUY"})
                body.append({"token_id": tid, "side": "SELL"})

            # Try once + ONE retry with backoff. A persistent failure
            # means these tokens keep their stale Gamma cached prices —
            # the bot would otherwise trade on potentially-hours-old
            # data without knowing. The loud warning at the end lets us
            # see the rate offline and tune as needed.
            data = None
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    r = await client.post(f"{CLOB_BASE}/prices", json=body)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0:
                        await asyncio.sleep(0.5)

            if data is None:
                chunks_failed += 1
                tokens_with_stale_prices.extend(chunk)
                print(f"!! CLOB /prices chunk {i}-{i+len(chunk)} FAILED after "
                      f"1 retry: {last_exc}")
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

    # Loud failure-rate signal — caller / operator can see when
    # /prices reliability degrades and adjust (skip placements, alert).
    if chunks_failed > 0:
        print(f"!! CLOB /prices: {chunks_failed}/{chunks_total} chunks failed, "
              f"{len(tokens_with_stale_prices)} tokens KEPT STALE Gamma prices "
              f"this cycle. Bot may trade on outdated bid/ask for those tokens.")

    return out


# ──────────────────────────────────────────────────────────────────────────
# CLOB orderbook depth (full price ladder, not just top-of-book)
#
# Top-of-book bid/ask via /prices is fast but doesn't reflect fillable depth.
# A signal that "looks profitable" at top-of-book may have only 5-10 shares
# at the displayed price — filling $20-50 walks through worse prices.
#
# Critical for:
#   - Tail-bucket trades where top-of-book has thin liquidity
#   - Bucket-sum-to-1 arbitrage (margins are 3-6¢; slippage of 1-2¢ kills it)
#   - Convergence-exit fills (often happen as liquidity thins near resolution)
#
# Added 2026-05-10.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class OrderBookLevel:
    price: float
    size_shares: float


@dataclass
class OrderBookDepth:
    """Full bid/ask ladder for one Polymarket binary token.

    Polymarket's /book endpoint returns prices/sizes as strings; we
    convert and sort canonically:
      - bids sorted DESCENDING (best bid first)
      - asks sorted ASCENDING  (best ask first)

    Polymarket's API may return them in either order; we always re-sort
    so callers can rely on the canonical layout.
    """

    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    tick_size: float
    min_order_size: float

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def simulate_buy(
        self, target_usd: float
    ) -> tuple[float, float, bool] | None:
        """Walk asks (lowest first) to fill `target_usd` of buy orders.

        Returns (avg_fill_price, shares_filled, fully_filled).
        If asks empty or zero sized, returns None.

        Note: respects min_order_size — won't return fills below it.
        """
        return _walk_levels(self.asks, target_usd, self.min_order_size)

    def simulate_sell(
        self, target_usd: float
    ) -> tuple[float, float, bool] | None:
        """Walk bids (highest first) to fill `target_usd` worth of sell orders.

        Returns (avg_proceeds_price, shares_sold, fully_filled).
        """
        return _walk_levels(self.bids, target_usd, self.min_order_size)

    def sweep_buy_to_ceiling(
        self, price_ceiling: float, max_usd: float | None = None,
    ) -> tuple[float, float, float] | None:
        """Walk asks up to `price_ceiling`, taking whatever depth is available.

        Returns (avg_fill_price, shares_filled, dollar_cost) or None if no
        asks below the ceiling.

        Different from `simulate_buy`:
          - No USD target required; walks until ceiling or out of book.
          - Accepts partial fills (no fully_filled boolean — caller checks
            against POLYMARKET_MIN_ORDER_SHARES / POLYMARKET_MIN_ORDER_USD).
          - Optionally caps total spend at `max_usd` so the bot never
            over-deploys when depth is unusually deep.

        Use for METAR feedback (and any future strategy where the trade is
        EV+ at any price below the ceiling). Not appropriate for
        model-driven trades — use `simulate_buy` + slippage cap for those.
        """
        if not self.asks:
            return None
        remaining_usd = float(max_usd) if max_usd is not None else float("inf")
        total_shares = 0.0
        total_cost = 0.0
        for lvl in self.asks:
            if lvl.price <= 0 or lvl.size_shares <= 0:
                continue
            if lvl.price > price_ceiling:
                break
            if remaining_usd <= 0:
                break
            max_shares_at_level = lvl.size_shares
            max_usd_at_level = lvl.price * max_shares_at_level
            if remaining_usd < max_usd_at_level:
                shares_taken = remaining_usd / lvl.price
                total_shares += shares_taken
                total_cost += shares_taken * lvl.price
                remaining_usd = 0.0
                break
            else:
                total_shares += max_shares_at_level
                total_cost += max_usd_at_level
                remaining_usd -= max_usd_at_level
        if total_shares <= 0:
            return None
        avg_price = total_cost / total_shares
        return (avg_price, total_shares, total_cost)


def _walk_levels(
    levels: list[OrderBookLevel],
    target_usd: float,
    min_order_size: float,
) -> tuple[float, float, bool] | None:
    """Walk a sorted-best-first level list, accumulating up to target_usd."""
    if not levels or target_usd <= 0:
        return None
    remaining_usd = float(target_usd)
    total_shares = 0.0
    total_cost = 0.0
    for lvl in levels:
        if lvl.price <= 0 or lvl.size_shares <= 0:
            continue
        max_shares_at_level = lvl.size_shares
        max_usd_at_level = lvl.price * max_shares_at_level
        if remaining_usd <= max_usd_at_level:
            shares_taken = remaining_usd / lvl.price
            total_shares += shares_taken
            total_cost += shares_taken * lvl.price
            remaining_usd = 0.0
            break
        else:
            total_shares += max_shares_at_level
            total_cost += max_usd_at_level
            remaining_usd -= max_usd_at_level
    if total_shares < min_order_size:
        return None
    fully_filled = remaining_usd <= 1e-6
    avg_price = total_cost / total_shares if total_shares > 0 else 0.0
    return (avg_price, total_shares, fully_filled)


async def fetch_orderbook_depth(
    token_id: str,
    client: httpx.AsyncClient | None = None,
) -> OrderBookDepth | None:
    """Fetch the full orderbook for one Polymarket token. Returns None on failure."""
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=10.0)
    try:
        try:
            r = await client.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"!! fetch_orderbook_depth({token_id[:16]}…): {exc}")
            return None
    finally:
        if owns:
            await client.aclose()

    raw_bids = data.get("bids", []) or []
    raw_asks = data.get("asks", []) or []

    def _parse(levels):
        out = []
        for lvl in levels:
            try:
                p = float(lvl.get("price", 0))
                s = float(lvl.get("size", 0))
                if p > 0 and s > 0:
                    out.append(OrderBookLevel(price=p, size_shares=s))
            except (TypeError, ValueError):
                continue
        return out

    bids = sorted(_parse(raw_bids), key=lambda lv: -lv.price)  # desc, best first
    asks = sorted(_parse(raw_asks), key=lambda lv: lv.price)   # asc, best first

    return OrderBookDepth(
        token_id=token_id,
        bids=bids,
        asks=asks,
        tick_size=float(data.get("tick_size", 0.001)),
        min_order_size=float(data.get("min_order_size", 5)),
    )


async def fetch_orderbook_depths_batch(
    token_ids: list[str],
    client: httpx.AsyncClient | None = None,
    *,
    concurrency: int = 6,
) -> dict[str, OrderBookDepth | None]:
    """Concurrently fetch depths for many tokens. Failures get None entries.

    Each /book call is a separate HTTP request (no batch endpoint exists);
    we limit concurrency to be polite to the API.
    """
    out: dict[str, OrderBookDepth | None] = {tid: None for tid in token_ids}
    if not token_ids:
        return out

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=15.0)

    sem = asyncio.Semaphore(concurrency)

    async def _one(tid: str):
        async with sem:
            depth = await fetch_orderbook_depth(tid, client)
            out[tid] = depth

    try:
        await asyncio.gather(*(_one(tid) for tid in token_ids))
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
    Markets with empty / null yes_token_id are counted separately and
    logged loudly so we can detect data-quality regressions in
    Gamma's event payloads.
    """
    n_updated = 0
    n_unchanged = 0
    n_missing_token_id = 0
    n_not_in_fresh = 0
    for ev in events:
        for m in ev.markets:
            tid = m.yes_token_id
            if not tid:
                # Distinguish empty / null token_id from "not in fresh"
                # so we can detect Gamma payload regressions vs missed
                # CLOB chunks.
                n_missing_token_id += 1
                n_unchanged += 1
                continue
            if tid not in fresh:
                n_not_in_fresh += 1
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
    if n_missing_token_id > 0:
        # Loud signal — Gamma should never return markets without
        # yes_token_id. If it does, something is wrong upstream.
        print(f"!! apply_clob_prices: {n_missing_token_id} markets have empty/null "
              f"yes_token_id (Gamma payload regression?)")
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
    """Derive the WEATHER day this market measures, in station-local timezone.

    Polymarket event `end_date` is the moment the market closes for
    settlement — typically AT or just past midnight station-local on
    the day AFTER the measurement period. For example, "Wellington
    May 11" market measures Wellington May 11 local weather and ends
    at midnight May 11/12 Wellington = 12:00 UTC May 11.

    The naive `.astimezone(tz).date()` would return May 12 (because
    end_date in NZ is at 00:00 May 12 boundary). That's the day AFTER
    the weather we care about — produces a day-off mismatch with
    `polymarket_won_bucket` (which is the May 11 resolution).

    Fix: subtract 1 second from end_date before converting. This pulls
    boundary-midnight values back to 23:59:59 of the prior day, which
    is what we want. Events whose end_date is mid-afternoon (some
    stations) are unaffected because 1 second doesn't cross the date
    boundary for them.

    Bug discovered 2026-05-14: prior version produced ALL Wellington
    audit mismatches (5/5 NZWN) which were really off-by-one date
    pairings, not oracle disagreements. After this fix the audit
    should re-evaluate cleanly.
    """
    from datetime import timedelta as _td
    just_before = event.end_date - _td(seconds=1)
    return just_before.astimezone(ZoneInfo(station.timezone)).date()


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
