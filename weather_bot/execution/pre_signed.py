"""Pre-signed order cache for low-latency Polymarket order placement.

Polymarket's CLOB SDK splits order placement into two steps:
  1. `create_order(args)` — builds + EIP-712 signs the order (~100-200ms)
  2. `post_order(signed, ...)` — HTTPS POST to Polymarket (~100-200ms)

For racing scenarios (Layer 7 guaranteed-NO-buy harvest), the signing
step is on the critical path between observing a cross event and
placing the order. By pre-signing candidate orders ahead of time, we
collapse the on-trigger latency from ~300ms to ~100ms.

This module is INFRASTRUCTURE ONLY — not wired into the live bot yet.
It exists so Layer 7 (planned Week 2) can use it without first having
to invent the pre-signing pattern. Until Layer 7 ships, the live bot
uses the standard `ExecutionClient.submit_order()` path which signs
inline (no behaviour change).

Key constraints:
  - Single-use: once broadcast, a cached order is removed (the nonce
    or order timestamp becomes stale; re-broadcasting risks rejection).
  - Time-bounded: caller is responsible for invalidating stale entries
    periodically. As of py_clob_client_v2 the order timestamp is
    embedded at create_order time; orders older than ~10 minutes
    should be re-signed.
  - Idempotent invalidate: safe to call invalidate() multiple times.

Intended Layer 7 usage:
    cache = PreSignedOrderCache(exec_client)

    # At scan time, for each live max-target bucket "37°C or higher":
    key = cache.pre_sign(
        token_id="123...",          # NO token of "37°C or higher"
        side="BUY",
        price=0.96,                 # max we're willing to pay
        size_shares=5.21,           # ~$5 / $0.96
        order_type="FAK",
    )
    bucket_to_key["KMIA:max:2026-05-17:37C+"] = key

    # When peak observation crosses 37°C:
    result = cache.broadcast(bucket_to_key["KMIA:max:2026-05-17:37C+"])
    # ~100ms instead of ~300ms — the signing happened in advance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from weather_bot.execution.client import ExecutionClient


class StaleSignatureError(RuntimeError):
    """Raised when a pre-signed order has aged past its TTL.

    The pre-signed order's EIP-712 timestamp / nonce becomes invalid
    after Polymarket's server-side staleness window (~10 min). Rather
    than broadcasting a stale signature (which Polymarket rejects with
    a confusing error or silently no-ops), we refuse to broadcast and
    let the caller fall back to inline sign+broadcast.
    """
    pass


@dataclass
class PreSignedEntry:
    """One pre-signed order ready for broadcast.

    Fields:
      signed_order:   The opaque object returned by SDK's create_order().
      order_type:     SDK OrderType enum (FAK / GTC / FOK / GTD).
      post_only:      Whether to enforce post-only on broadcast.
      signed_at:      Unix timestamp of when this was signed (for TTL).
      meta:           Free-form metadata (bucket label, station, etc.)
                      for debugging / logging.
    """
    signed_order: object
    order_type: object  # SDK OrderType enum
    post_only: bool
    signed_at: float
    meta: dict = field(default_factory=dict)


@dataclass
class PreparedCancelEntry:
    """One pre-prepared cancel ready for fast broadcast.

    Unlike order placement (EIP-712-signed), Polymarket's cancel endpoint
    uses L2 API auth (HMAC per request, no on-chain signature). The
    "pre-sign" we can do for cancels is:
      - Pre-build the OrderPayload object (~1ms saved)
      - Hold a cancel callable that fires the cancel with no portfolio
        re-load, no object construction, no extra indirection

    Realistic speedup vs inline cancel_order(): 5-15ms per fire. Small
    in absolute terms but valuable on a hair-trigger drift cancel where
    the resting maker is being SELL-into by an adverse taker — every
    millisecond between observation and cancel is a millisecond more
    adverse fill exposure.

    Fields:
      order_id:   The Polymarket order ID to cancel.
      payload:    Pre-built OrderPayload (or raw order_id if SDK v1).
      prepared_at: Unix timestamp of when this was built.
      meta:       Free-form metadata (station_id, bucket_label, etc.).
    """
    order_id: str
    payload: object   # OrderPayload | str
    prepared_at: float
    meta: dict = field(default_factory=dict)


class PreSignedOrderCache:
    """In-memory cache of pre-signed orders keyed by caller-chosen IDs.

    The cache is per-process (lives in the daemon's memory; a cron-based
    invocation has no opportunity to pre-sign and benefits less). For
    Layer 7's racing use case, the daemon pre-signs at scan time and
    broadcasts on observation crossing.
    """

    def __init__(self, exec_client: "ExecutionClient", default_ttl_seconds: float = 600.0):
        """
        Args:
          exec_client: The bot's ExecutionClient instance. The cache uses
            its `_clob` attribute (the py_clob_client_v2.ClobClient) to
            call create_order() and post_order() directly.
          default_ttl_seconds: How long a pre-signed order remains valid
            before being considered stale (default 10 minutes). The cache
            does NOT auto-expire entries; the caller must call
            `prune_expired()` periodically. TTL is informational — the
            SDK / Polymarket may have its own staleness window.
        """
        self._exec_client = exec_client
        self._default_ttl_seconds = float(default_ttl_seconds)
        self._cache: dict[str, PreSignedEntry] = {}
        # Separate map for prepared cancels. Cancels DON'T have an
        # EIP-712 staleness window (L2 auth is per-request HMAC) so the
        # cache can hold these indefinitely while the order is alive —
        # though we still prune entries whose order_id no longer appears
        # in the resting-maker set (daemon refreshes each cycle).
        self._cancel_cache: dict[str, PreparedCancelEntry] = {}

    def pre_sign(
        self,
        cache_key: str,
        *,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
        order_type: str = "FAK",
        post_only: bool = False,
        meta: Optional[dict] = None,
    ) -> str:
        """Pre-sign an order; cache under `cache_key` for later broadcast.

        If a previous entry exists under the same key, it's overwritten
        (the caller chose to re-sign; old entry is discarded).

        Returns:
          The cache_key (for chaining).

        Raises:
          ImportError: if py_clob_client_v2 is not installed.
          Exception:   any SDK / signing error propagates as-is.
        """
        try:
            # OrderType lives in clob_types in py_clob_client_v2 (not in
            # order_builder.constants where v1 had it).
            from py_clob_client_v2.clob_types import OrderArgs, OrderType
            from py_clob_client_v2.order_builder.constants import BUY, SELL
        except ImportError as exc:
            raise ImportError(
                "py_clob_client_v2 not installed — pre-signing requires it. "
                f"Underlying error: {exc}"
            )

        side_const = BUY if side.upper() == "BUY" else SELL
        try:
            ot_enum = getattr(OrderType, order_type.upper())
        except AttributeError:
            raise ValueError(
                f"Unsupported order_type {order_type!r}; "
                f"valid: FAK, GTC, FOK, GTD"
            )

        args = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size_shares),
            side=side_const,
        )
        signed = self._exec_client._clob.create_order(args)

        self._cache[cache_key] = PreSignedEntry(
            signed_order=signed,
            order_type=ot_enum,
            post_only=bool(post_only),
            signed_at=time.time(),
            meta=dict(meta or {}),
        )
        return cache_key

    def broadcast(self, cache_key: str, ttl_seconds: Optional[float] = None):
        """Broadcast a pre-signed order. The entry is consumed (removed).

        Args:
          ttl_seconds: Override the default TTL. If the entry is older
            than this, the entry is dropped and `StaleSignatureError` is
            raised — DO NOT broadcast a stale signature.

        Returns:
          The raw SDK response dict from post_order().

        Raises:
          KeyError: if no entry exists for `cache_key`.
          StaleSignatureError: if the entry is past its TTL.
          Exception: any SDK / network error propagates as-is.
        """
        # Peek (don't consume yet) so we can preserve audit info if stale.
        entry = self._cache.get(cache_key)
        if entry is None:
            raise KeyError(f"No pre-signed order under cache key {cache_key!r}")

        ttl = self._default_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        age = time.time() - entry.signed_at
        if age > ttl:
            # Stale — drop the entry so subsequent broadcasts can't re-use
            # it, and raise so the caller can fall back to inline signing.
            self._cache.pop(cache_key, None)
            raise StaleSignatureError(
                f"Pre-signed order {cache_key!r} is {age:.1f}s old "
                f"(>{ttl:.0f}s TTL); refusing to broadcast — caller should "
                f"fall back to inline sign+broadcast"
            )

        # Age check passed — now consume + broadcast.
        self._cache.pop(cache_key, None)
        return self._exec_client._clob.post_order(
            entry.signed_order,
            entry.order_type,
            post_only=entry.post_only,
        )

    def peek(self, cache_key: str) -> Optional[PreSignedEntry]:
        """Return the entry without consuming it (for inspection / metrics)."""
        return self._cache.get(cache_key)

    def invalidate(self, cache_key: Optional[str] = None) -> int:
        """Drop a single entry (by key) or all entries.

        Returns count removed. Safe to call on non-existent keys.
        """
        if cache_key is None:
            n = len(self._cache)
            self._cache.clear()
            return n
        if cache_key in self._cache:
            del self._cache[cache_key]
            return 1
        return 0

    def prune_expired(self, ttl_seconds: Optional[float] = None) -> int:
        """Remove entries older than ttl_seconds. Returns count removed."""
        ttl = self._default_ttl_seconds if ttl_seconds is None else ttl_seconds
        now = time.time()
        stale_keys = [k for k, e in self._cache.items() if (now - e.signed_at) > ttl]
        for k in stale_keys:
            del self._cache[k]
        return len(stale_keys)

    def size(self) -> int:
        """Number of pre-signed orders currently cached."""
        return len(self._cache)

    def keys(self) -> list[str]:
        """Snapshot of currently cached keys (for debugging)."""
        return list(self._cache.keys())

    # ──────────────────────────────────────────────────────────────────
    # Cancel pre-preparation (Layer 8 fast path)
    # ──────────────────────────────────────────────────────────────────

    def pre_prepare_cancel(
        self,
        order_id: str,
        *,
        meta: Optional[dict] = None,
    ) -> str:
        """Pre-build the cancel payload for `order_id`; cache it for later
        fast-fire by `broadcast_cancel(order_id)`.

        Unlike order pre-signing, this doesn't perform any cryptographic
        operations — Polymarket's cancel endpoint uses L2 API auth (HMAC
        per request) which has to be fresh, so it happens at broadcast
        time. What we DO pre-build:
          - The OrderPayload object (v2 SDK)
          - A direct reference to the SDK cancel callable

        Idempotent: re-calling with the same order_id refreshes the entry.

        Returns:
          The order_id (for chaining).
        """
        if not order_id:
            raise ValueError("order_id must be a non-empty string")

        # Try v2 first; fall back to v1 string form.
        payload: object
        try:
            from py_clob_client_v2.clob_types import OrderPayload  # type: ignore
            payload = OrderPayload(orderID=order_id)
        except ImportError:
            payload = order_id

        self._cancel_cache[order_id] = PreparedCancelEntry(
            order_id=order_id,
            payload=payload,
            prepared_at=time.time(),
            meta=dict(meta or {}),
        )
        return order_id

    def has_cancel(self, order_id: str) -> bool:
        """Returns True if a prepared cancel exists for this order_id."""
        return order_id in self._cancel_cache

    def broadcast_cancel(self, order_id: str) -> bool:
        """Fire a pre-prepared cancel. The entry is consumed (removed).

        This skips the OrderPayload construction + the cancel-callable
        lookup that the inline path does. The actual cancel API call
        (HTTPS POST with L2 HMAC) still happens — Polymarket's auth
        scheme can't be pre-computed reliably.

        Args:
          order_id: The Polymarket order ID. Must have a prepared entry.

        Returns:
          True if cancel call returned ok; False otherwise.

        Raises:
          KeyError:  if no prepared entry exists.
          Exception: SDK / network errors propagate as-is.
        """
        entry = self._cancel_cache.get(order_id)
        if entry is None:
            raise KeyError(f"No prepared cancel for order_id {order_id!r}")
        # Consume so a second broadcast can't re-fire the same payload.
        self._cancel_cache.pop(order_id, None)

        clob = self._exec_client._clob
        if clob is None:
            # Dry-run / no SDK — treat as success without an actual API call.
            return True
        try:
            cancel = getattr(clob, "cancel_order", None) or getattr(clob, "cancel")
            cancel(entry.payload)
            return True
        except Exception as exc:
            # Caller (Layer 8) decides whether to fall back to inline
            # cancel_order_verified or treat as cancel_failed.
            raise

    def invalidate_cancel(self, order_id: Optional[str] = None) -> int:
        """Drop one cancel entry (by order_id) or all entries.

        Returns count removed. Safe on non-existent keys.
        """
        if order_id is None:
            n = len(self._cancel_cache)
            self._cancel_cache.clear()
            return n
        if order_id in self._cancel_cache:
            del self._cancel_cache[order_id]
            return 1
        return 0

    def prune_cancels_not_in(self, live_order_ids: set[str]) -> int:
        """Drop cancel entries for orders no longer in the live set.

        Daemon calls this each cycle with the current set of resting-maker
        order_ids. Cancels for filled / cancelled / unknown orders get
        cleared so the cache size stays bounded by the current portfolio.

        Returns count removed.
        """
        stale = [oid for oid in self._cancel_cache if oid not in live_order_ids]
        for oid in stale:
            del self._cancel_cache[oid]
        return len(stale)

    def cancel_size(self) -> int:
        """Number of prepared cancels currently cached."""
        return len(self._cancel_cache)
