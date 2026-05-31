"""Layer 3 — Polymarket WebSocket client for sub-second market data.

Polymarket exposes a public WebSocket endpoint for streaming order
book updates. For our use case (cross-up cancel reacting to bid drops,
Layer 7 guaranteed-NO-buy on dead buckets), receiving book updates
via push notification is dramatically faster than polling REST:

  - REST polling (current Layer 1 daemon path): up to 30s lag
  - WebSocket push: <1s from match → client

The persistent WebSocket connection ALSO solves the TLS-handshake-
per-request problem for Polymarket reads. Once subscribed, all book
updates flow over the same connection — no per-request 54ms handshake.

This module exposes:
  - `PolymarketWS`: async context manager that connects, subscribes,
    and yields book updates as parsed dicts.
  - `subscribe_and_watch()`: helper that wraps the loop with
    auto-reconnect + exponential backoff.

Integration with Layer 1 daemon (Week 2):
  - Daemon subscribes to NO tokens of held positions on startup.
  - On book update, daemon updates its in-memory best-bid/ask cache.
  - Cross-up cancel + Layer 7 use the cached values instead of
    polling /book each cycle.
  - When portfolio changes (new placement / resolution), daemon
    updates the subscription set.

Endpoint (per Polymarket docs):
  wss://ws-subscriptions-clob.polymarket.com/ws/market
  (Read-only market data; no authentication required.)

Message types (per 2026-05-17 empirical probe):
  - `[]` (empty array, immediately after subscribe): subscription ack.
  - `book` (full snapshot): event_type="book", asset_id, bids[], asks[].
    NOT sent automatically on subscribe; only on first book change for
    the subscribed asset. To get initial state, fetch /book via REST.
  - `price_change` (delta): event_type="price_change", asset_id,
    changes[{price, size, side}]. Most common message in steady state.
  - `tick_size_change`: event_type="tick_size_change", asset_id,
    new_tick_size, old_tick_size.

Less interesting for our use case:
  - `last_trade_price` (we already get this via gamma)

Daemon integration pattern (when wired in Week 2):
  1. On daemon start / portfolio change: REST-fetch /book for each
     held NO token (one-time cold start).
  2. Subscribe via PolymarketWS to the same tokens.
  3. Apply incoming `price_change` deltas to the cached book to
     maintain accurate best-bid/ask.
  4. On `book` (full snapshot received later), replace cached state.
  5. Cross-up cancel + Layer 7 consume the cached values (no REST
     /book per cycle — pure in-memory lookups).

Reconnect strategy:
  - Connection drop → wait 1s, retry
  - Subsequent failures double the wait (exponential backoff)
  - Cap at 60s between attempts
  - After 10 consecutive failures, log a loud warning (probably the
    Polymarket WS endpoint is down; daemon's REST polling continues
    to work meanwhile)
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional


POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

MAX_BACKOFF_SECONDS = 60.0
INITIAL_BACKOFF_SECONDS = 1.0
LOUD_WARNING_AFTER_FAILURES = 10


class PolymarketWS:
    """Async WebSocket client for Polymarket CLOB market data.

    Usage:
        async with PolymarketWS(token_ids=[...]) as ws:
            async for message in ws:
                # message is a parsed JSON dict
                print(message.get("event_type"), message.get("asset_id"))

    Single connection lifetime is bounded by the `async with` block.
    For long-lived subscriptions with auto-reconnect, use
    `subscribe_and_watch()` instead.
    """

    def __init__(self, token_ids: list[str]):
        self.token_ids = list(token_ids)
        self._ws = None  # set in __aenter__

    async def __aenter__(self) -> "PolymarketWS":
        import websockets

        self._ws = await websockets.connect(
            POLYMARKET_WS_URL,
            ping_interval=20,    # send ping every 20s (server keeps connection)
            ping_timeout=10,     # disconnect if pong not received in 10s
            close_timeout=5,
            max_size=2**20,      # 1 MB max frame size
        )

        # Subscribe to the requested tokens
        subscribe_msg = {"type": "MARKET", "assets_ids": self.token_ids}
        await self._ws.send(json.dumps(subscribe_msg))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._receive_loop()

    async def _receive_loop(self) -> AsyncIterator[dict]:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected — use async with")
        async for raw in self._ws:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # Polymarket sends arrays sometimes (batched updates)
            if isinstance(msg, list):
                for m in msg:
                    if isinstance(m, dict):
                        yield m
            elif isinstance(msg, dict):
                yield msg


async def subscribe_and_watch(
    token_ids: list[str],
    *,
    on_message: Callable[[dict], None],
    on_reconnect: Optional[Callable[[], None]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Long-lived subscription with auto-reconnect.

    Args:
      token_ids: list of NO/YES token IDs to subscribe to.
      on_message: callback invoked once per received message.
        Should be FAST (non-blocking) — heavy work should be queued
        to another task. Callbacks that raise are logged + swallowed.
      on_reconnect: optional callback invoked after each successful
        (re)connect. Useful for re-subscribing to a refreshed token
        set (e.g., on portfolio change, recompute held tokens and
        pass them in).
      stop_event: optional asyncio.Event that, when set, breaks the
        reconnect loop and returns cleanly.

    Returns when stop_event is set, or never returns (loops forever
    on reconnect failures).
    """
    backoff = INITIAL_BACKOFF_SECONDS
    consecutive_failures = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            async with PolymarketWS(token_ids) as ws:
                # Successful connection — reset backoff
                backoff = INITIAL_BACKOFF_SECONDS
                consecutive_failures = 0
                if on_reconnect is not None:
                    try:
                        on_reconnect()
                    except Exception:
                        pass

                async for message in ws:
                    try:
                        on_message(message)
                    except Exception as exc:
                        # Don't let a bad callback kill the connection.
                        print(f"[polymarket_ws] on_message error: {exc}")

                    if stop_event is not None and stop_event.is_set():
                        return

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            consecutive_failures += 1
            # Log on the threshold AND every multiple of it thereafter,
            # so a sustained WS outage stays visible in journalctl (audit
            # finding M4: previously `==` meant only the 10th failure
            # logged; failures 11+ were silent).
            if (consecutive_failures >= LOUD_WARNING_AFTER_FAILURES
                    and consecutive_failures % LOUD_WARNING_AFTER_FAILURES == 0):
                ts = datetime.now(timezone.utc).isoformat()
                print(f"[polymarket_ws {ts}] WARNING: {consecutive_failures} "
                      f"consecutive connection failures: {type(exc).__name__}: {exc}")

        # Wait before retry (exponential backoff)
        if stop_event is not None and stop_event.is_set():
            return
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, MAX_BACKOFF_SECONDS)


# ──────────────────────────────────────────────────────────────────────────
# In-memory book cache (Layer 1 daemon will use this)
# ──────────────────────────────────────────────────────────────────────────

class BookCache:
    """In-memory best-bid/ask cache keyed by token_id.

    Designed to be updated by `subscribe_and_watch` callbacks. The Layer 1
    daemon's cross-up cancel + Layer 7 guaranteed-NO-buy then consume
    the cached values (no /book REST call per cycle).

    Thread/async safety: single-threaded async use only. No locking.
    """

    def __init__(self):
        # token_id -> {"bid": float, "ask": float, "updated_at": iso}
        self._cache: dict[str, dict] = {}

    def on_book_message(self, msg: dict) -> None:
        """Callback compatible with subscribe_and_watch.on_message."""
        event_type = msg.get("event_type")
        asset_id = msg.get("asset_id") or msg.get("market") or ""
        if not asset_id:
            return

        if event_type == "book":
            # Full snapshot. Polymarket sends `bids` and `asks` as lists
            # of {"price": "0.78", "size": "100"} dicts. This CLEARS any
            # prior `book_invalidated` flag — a fresh full snapshot is
            # the source of truth, replacing whatever the delta stream
            # had previously made stale (audit finding C1).
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            bid = self._best_price(bids, side="bid")
            ask = self._best_price(asks, side="ask")
            self._update(asset_id, bid=bid, ask=ask, bids=bids, asks=asks)
        elif event_type == "price_change":
            # Delta arrival — we don't maintain full book state, so we
            # INVALIDATE the cached bid/ask. Callers asking for a fresh
            # reading will see None and fall back to REST. This is
            # SAFER than serving stale prices. A future iteration could
            # apply the delta to maintain the book. A subsequent `book`
            # snapshot clears the flag (handled by _update).
            entry = self._cache.setdefault(asset_id, {})
            entry["last_event"] = "price_change"
            entry["last_seen_at_ts"] = time.time()
            entry["book_invalidated"] = True

    def best_bid(self, token_id: str) -> Optional[float]:
        entry = self._cache.get(token_id, {})
        if entry.get("book_invalidated"):
            return None
        return entry.get("bid")

    def best_ask(self, token_id: str) -> Optional[float]:
        entry = self._cache.get(token_id, {})
        if entry.get("book_invalidated"):
            return None
        return entry.get("ask")

    def fresh_best_ask(
        self, token_id: str, max_age_seconds: float = 30.0,
    ) -> Optional[float]:
        """Return cached best ask ONLY if it's fresh enough.

        Returns None if:
          - token not in cache
          - book has been invalidated by a price_change delta
          - last book update was > max_age_seconds ago

        Caller should fall back to REST data when None is returned.
        Use case: Layer 7 / Layer 8 prefer sub-second WS data when
        available; otherwise use the cycle's REST-refreshed snapshot.
        """
        entry = self._cache.get(token_id, {})
        if entry.get("book_invalidated"):
            return None
        ask = entry.get("ask")
        ts = entry.get("updated_at_ts")
        if ask is None or ts is None:
            return None
        age_seconds = time.time() - float(ts)
        if age_seconds < 0 or age_seconds > max_age_seconds:
            return None
        return float(ask)

    def fresh_best_bid(
        self, token_id: str, max_age_seconds: float = 30.0,
    ) -> Optional[float]:
        """Like fresh_best_ask but for the bid side."""
        entry = self._cache.get(token_id, {})
        if entry.get("book_invalidated"):
            return None
        bid = entry.get("bid")
        ts = entry.get("updated_at_ts")
        if bid is None or ts is None:
            return None
        age_seconds = time.time() - float(ts)
        if age_seconds < 0 or age_seconds > max_age_seconds:
            return None
        return float(bid)

    def get_depth(self, token_id: str, max_age_seconds: float = 30.0):
        """Return an OrderBookDepth built from the cached ladder, or None
        if the book is invalidated / stale / never seen. Lets the NO-side
        sweep depth-walk SYNCHRONOUSLY from the WS stream — no REST call.

        Returns None (not a partial) when invalidated by a price_change
        delta, so the caller falls back to top-of-book rather than walking
        a stale ladder.
        """
        from weather_bot.polymarket import OrderBookDepth, OrderBookLevel
        entry = self._cache.get(token_id, {})
        if entry.get("book_invalidated"):
            return None
        ts = entry.get("updated_at_ts")
        if ts is None or (time.time() - float(ts)) > max_age_seconds:
            return None
        raw_bids = entry.get("bids_raw")
        raw_asks = entry.get("asks_raw")
        if raw_bids is None and raw_asks is None:
            return None

        def _parse(levels):
            out = []
            for lv in levels or []:
                try:
                    p = float(lv.get("price", 0)); s = float(lv.get("size", 0))
                    if p > 0 and s > 0:
                        out.append(OrderBookLevel(price=p, size_shares=s))
                except (TypeError, ValueError, AttributeError):
                    continue
            return out

        bids = sorted(_parse(raw_bids), key=lambda lv: -lv.price)  # best first
        asks = sorted(_parse(raw_asks), key=lambda lv: lv.price)
        return OrderBookDepth(
            token_id=token_id, bids=bids, asks=asks,
            tick_size=0.01, min_order_size=5,
        )

    def updated_at(self, token_id: str) -> Optional[str]:
        entry = self._cache.get(token_id, {})
        return entry.get("updated_at")

    def size(self) -> int:
        return len(self._cache)

    def fresh_size(self, max_age_seconds: float = 30.0) -> int:
        """Count of entries that would serve a fresh read RIGHT NOW.

        Distinct from `size()` which includes invalidated/stale entries.
        Used for cycle observability — `cache_size=37` looks healthy but
        `fresh_size=0` reveals all entries are price_change-invalidated
        (audit finding H4)."""
        now = time.time()
        n = 0
        for entry in self._cache.values():
            if entry.get("book_invalidated"):
                continue
            ts = entry.get("updated_at_ts")
            if ts is None:
                continue
            age = now - float(ts)
            if 0 <= age <= max_age_seconds:
                n += 1
        return n

    def invalidated_size(self) -> int:
        """Count of entries currently flagged book_invalidated."""
        return sum(1 for e in self._cache.values() if e.get("book_invalidated"))

    def _update(self, asset_id: str, *, bid: Optional[float], ask: Optional[float],
                bids: Optional[list] = None, asks: Optional[list] = None) -> None:
        entry = self._cache.setdefault(asset_id, {})
        if bid is not None:
            entry["bid"] = bid
        if ask is not None:
            entry["ask"] = ask
        # Retain the full ladder from the snapshot so consumers can do a
        # synchronous depth-walk (no REST). Stored as raw {price,size}
        # dicts; get_depth() parses them into an OrderBookDepth.
        if bids is not None:
            entry["bids_raw"] = bids
        if asks is not None:
            entry["asks_raw"] = asks
        # Store as float seconds (time.time) — robust to refactors that
        # would otherwise break datetime.fromisoformat parsing (audit M2).
        entry["updated_at_ts"] = time.time()
        # Keep ISO form too for observability (the daemon's log line),
        # but freshness math uses the float.
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        entry["last_event"] = "book"
        # CRITICAL: clear the invalidation flag set by a prior price_change
        # delta. Without this, the cache becomes permanently invalidated
        # after the first delta and `fresh_*` returns None forever (audit
        # finding C1).
        entry["book_invalidated"] = False

    @staticmethod
    def _best_price(level_list: list, side: str) -> Optional[float]:
        """Extract the best price from a list of {price, size} dicts.
        Best bid = highest price; best ask = lowest price."""
        if not isinstance(level_list, list) or not level_list:
            return None
        prices = []
        for lv in level_list:
            try:
                p = float(lv.get("price"))
                sz = float(lv.get("size", 0))
                if sz > 0:
                    prices.append(p)
            except (TypeError, ValueError):
                continue
        if not prices:
            return None
        return max(prices) if side == "bid" else min(prices)
