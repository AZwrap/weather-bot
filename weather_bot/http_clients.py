"""Shared HTTP/2 client pool for the weather-bot.

Provides a module-level `httpx.AsyncClient` instance configured for
HTTP/2 with persistent connection pooling. Every cron run / daemon
process should use this singleton — TLS connections stay open for the
lifetime of the process, eliminating the ~54ms TLS handshake on every
subsequent request to the same host.

Measured impact (from 2026-05-17 VPS latency probe):
  - TCP connect to Polymarket CLOB:    ~2.4ms
  - TLS handshake to Polymarket CLOB:  ~54ms (cumulative)
  - Polygon RPC public endpoint:       ~75ms
  - Network RTT (Amsterdam→Polymarket): 1.27ms

Without persistent connections, every REST call eats ~56ms of TLS
handshake even though the network is essentially adjacent. With HTTP/2
+ keep-alive pooling, only the FIRST request to a given host pays this
cost; subsequent requests reuse the established connection.

The module exposes:
  - `get_http_client()`:    lazy-initialized shared client (use this in
                            cron entry points + the future daemon)
  - `make_local_client()`:  create a fresh AsyncClient with the same
                            HTTP/2 + pool config; for cases that want a
                            per-block client closed via `async with`
  - `close_all()`:          cleanly close the shared client at exit

The shared client maintains separate per-host connection pools
internally (httpx behaviour), so one client handles Polymarket,
METAR sources, and Polygon RPC concurrently without interference.

Usage (cron entry points):
    from weather_bot.http_clients import get_http_client, close_all

    async def main():
        client = get_http_client()
        # pass `client` to all internal functions that accept it
        ...
        # at program exit
        await close_all()

NOTE: This module does NOT change any internal function signatures.
Functions throughout the codebase that accept a `client` kwarg
(scanner, polymarket, observations, etc.) keep their existing API.
The only change is that the TOP-LEVEL caller passes the shared
client instead of creating a per-run one.
"""
from __future__ import annotations

from typing import Optional

import httpx

# Connection pool sized for a single cron run or daemon process.
# - max_connections:           hard cap across all hosts
# - max_keepalive_connections: how many idle connections to keep warm
# - keepalive_expiry:          seconds to keep an idle connection alive
#
# 20 keepalive is comfortable: we typically talk to ~5 distinct hosts
# (Polymarket clob/gamma/data, Iowa State, Polygon RPC) and may run
# multiple concurrent requests per host. 120s expiry is well below
# typical server timeouts.
_POOL_LIMITS = httpx.Limits(
    max_connections=30,
    max_keepalive_connections=20,
    keepalive_expiry=120.0,
)

# Generous default timeout. Individual callers can override per-request
# via `client.get(..., timeout=X)` if they need a tighter or looser bound.
# - read timeout 60s: METAR sources can be slow; Polymarket usually <5s
# - connect timeout 10s: TCP+TLS handshake should be sub-second from
#   Amsterdam; 10s catches transient network hiccups without hanging.
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared HTTP/2 AsyncClient (lazy-initialized).

    Subsequent calls return the same instance. If the client was
    previously closed, a fresh one is created. The client maintains
    per-host connection pools internally, so it can be shared across
    Polymarket, METAR, and Polygon RPC calls without interference.

    Raises ImportError at runtime if the `h2` package is not installed
    (HTTP/2 is optional in httpx; the bot's requirements.txt installs it).
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=_DEFAULT_TIMEOUT,
            limits=_POOL_LIMITS,
            headers={"User-Agent": "weather-bot/1.0"},
            follow_redirects=False,
        )
    return _http_client


def make_local_client() -> httpx.AsyncClient:
    """Create a fresh AsyncClient with the same HTTP/2 + pool config.

    Use this for code paths that want `async with httpx.AsyncClient(...) as
    client:` block semantics (per-block lifetime, auto-close at exit).
    All HTTP/2 / pool / timeout settings match the shared client, so the
    in-block latency profile is the same.

    For long-running processes (the future daemon), prefer the shared
    `get_http_client()` instead — connections live across the daemon
    lifetime, not just one block.
    """
    return httpx.AsyncClient(
        http2=True,
        timeout=_DEFAULT_TIMEOUT,
        limits=_POOL_LIMITS,
        headers={"User-Agent": "weather-bot/1.0"},
        follow_redirects=False,
    )


async def close_all() -> None:
    """Cleanly close the shared client at program exit.

    Safe to call multiple times. Safe to call before any client has
    been initialized.
    """
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None
