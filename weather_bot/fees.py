"""Polymarket fee model (Weather markets).

The hardcoded constants below were empirically validated against 281
real trades on 2026-05-25 (formula matched 281/281). They are also
periodically cross-checked against live Polymarket config via
`fetch_live_fee_config()` — if Polymarket changes the rate, the bot
logs a warning and the operator should update the constants here.

ORIGINAL VALIDATION NOTES (preserved):

EMPIRICAL VALIDATION
====================

Verified against 281 actual Polymarket trades from
Polymarket-History-2026-05-26.csv. Formula matched fee charged
exactly on 281/281 trades.

Total fees paid since live (from CSV): $13.7192
Total Buy notional in same window: $1,129.03
Effective rate: 1.215% of notional (close to theoretical 1.25% peak
at p=0.5, indicating most trades happen near mid-price).

FORMULA
=======

  fee_USDC = shares × FEE_RATE × p × (1 - p)

Where:
  - shares: number of shares in the trade
  - FEE_RATE: 0.05 for Weather markets (confirmed)
  - p: trade fill price (not the bucket probability — the
    actual price paid per share of the token bought)

Properties:
  - Peak fee at p=0.5: shares × 0.0125 = 1.25% of shares value
  - Zero at p=0 or p=1 (certain outcomes)
  - Symmetric (fee on a BUY at $0.30 = fee on a BUY at $0.70 if
    same share count)

REBATE
======

  - takerOnly: True — makers pay $0
  - rebateRate: 0.25 — 25% of taker fees collected daily are
    redistributed to makers proportionally to their maker volume
  - Tracked via daily_maker_rebates dict in Portfolio + the
    sync_maker_rebates.py cron job pulls actuals from Polymarket

PER-STRATEGY EXPECTED FEE
=========================

  NO_momentum (A∩C):   maker, $0 fee, small rebate (~$0.003/fill)
  V2 conditional:      maker, $0 fee, small rebate
  Layer 7 @ $0.99:     taker, ~$0.0025/fire  (negligible)
  Layer 7 @ $0.95:     taker, ~$0.013/fire   (small)
  METAR @ $0.92:       taker, ~$0.020/fire   (small)
  Arb leg @ $0.30-0.70: taker, $0.06-$0.25/LEG  (SIGNIFICANT —
                       can exceed thin arb margins)

CRITICAL: arb was firing at MIN_ARB_MARGIN_USD_DEPTH=$0.02 with no
fee awareness. Typical 2-leg arb pays $0.20-$0.50 in fees on $5-20
notional. Every arb fire was negative-EV before our partial-fill bug
ate the rest. See live_bucket_arb.py for dynamic fee-aware gate.
"""
from __future__ import annotations

# Weather/Culture markets fee rate. Confirmed empirically 2026-05-25.
FEE_RATE_WEATHER: float = 0.05

# Rebate program: 25% of taker fees collected redistributed daily to
# makers proportional to their maker volume.
REBATE_RATE_WEATHER: float = 0.25


def taker_fee_usd(shares: float, price: float, fee_rate: float = FEE_RATE_WEATHER) -> float:
    """Compute the taker fee for a single trade.

    Args:
      shares: number of shares filled
      price: trade fill price (0..1)
      fee_rate: category-specific rate (default Weather=0.05)

    Returns: fee in USDC. 0 if price out of range.
    """
    if not (0.0 < price < 1.0) or shares <= 0:
        return 0.0
    return float(shares) * fee_rate * float(price) * (1.0 - float(price))


def estimated_maker_rebate_usd(
    shares: float, price: float,
    fee_rate: float = FEE_RATE_WEATHER,
    rebate_rate: float = REBATE_RATE_WEATHER,
) -> float:
    """Estimate the per-fill maker rebate.

    NOTE: this is an APPROXIMATION. The actual rebate is a daily-pool
    distribution proportional to your fraction of total maker volume.
    The simple "25% of what your counterparty paid" estimate is the
    upper bound — actual rebate may be lower if more makers compete
    for the same pool.

    For accounting purposes, the actual rebate should come from
    Portfolio.daily_maker_rebates (populated by sync_maker_rebates.py)
    rather than this estimate.
    """
    return taker_fee_usd(shares, price, fee_rate) * rebate_rate


def arb_total_fees_usd(legs: list[tuple[float, float]]) -> float:
    """Compute total taker fee for a multi-leg arb.

    Args:
      legs: list of (shares, price) per leg

    Returns: sum of taker fees across all legs.
    """
    return sum(taker_fee_usd(sh, p) for sh, p in legs)


# ──────────────────────────────────────────────────────────────────────
# Live fee config validation
#
# Polymarket exposes per-market fee info on its gamma + clob APIs. The
# field names + structure aren't stable in the public docs, so the
# fetcher below probes several candidate field paths and returns
# whatever it finds. Result is cached for the process lifetime; the
# sanity-check helper emits a warning if the live rate differs from the
# hardcoded constants.
# ──────────────────────────────────────────────────────────────────────

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"

FEE_CONFIG_CACHE_PATH = Path("data/fee_config_cache.json")
# How long the cached live-fetch is considered fresh. Default 24h.
FEE_CONFIG_TTL_HOURS = 24.0


@dataclass
class LiveFeeConfig:
    """Snapshot of Polymarket's fee config for a sample Weather market.

    `source` tells us where the data came from:
      - "gamma_market.{field}"  → pulled from gamma /markets/{id}
      - "clob_market.{field}"   → pulled from CLOB /markets/{token}
      - "fallback_constant"     → no live endpoint exposed the field
    """
    taker_fee_rate: float
    maker_rebate_rate: float | None
    sample_market_id: int | None
    source: str
    fetched_at_utc: str

    def to_jsonable(self) -> dict:
        return {
            "taker_fee_rate": self.taker_fee_rate,
            "maker_rebate_rate": self.maker_rebate_rate,
            "sample_market_id": self.sample_market_id,
            "source": self.source,
            "fetched_at_utc": self.fetched_at_utc,
        }


def _extract_fee_from_dict(d: dict) -> tuple[float | None, str | None]:
    """Probe known field-name variants. Returns (rate, source_key)."""
    candidates = [
        # (path, scale_to_fraction)
        ("feeRate", 1.0),
        ("fee_rate", 1.0),
        ("feeFraction", 1.0),
        ("takerFeeBps", 1e-4),  # basis points → fraction
        ("taker_fee_bps", 1e-4),
        ("fee", 1.0),
        ("feeBps", 1e-4),
    ]
    for key, scale in candidates:
        v = d.get(key)
        if v is None:
            continue
        try:
            f = float(v) * scale
        except (TypeError, ValueError):
            continue
        if 0.0 < f < 1.0:
            return f, key
    return None, None


def _extract_rebate_from_dict(d: dict) -> float | None:
    for key in ("rebateRate", "rebate_rate", "makerRebateRate",
                "maker_rebate_rate"):
        v = d.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 < f < 1.0:
            return f
    return None


async def _fetch_sample_weather_market(client) -> dict | None:
    """Get one currently-active Weather market from gamma. Used as the
    fee probe sample. Returns the raw market dict or None on failure."""
    try:
        r = await client.get(
            f"{GAMMA_BASE_URL}/events",
            params={
                "tag_slug": "highest-temperature",
                "active": "true",
                "closed": "false",
                "limit": 1,
            },
            timeout=15.0,
        )
        r.raise_for_status()
        events = r.json()
        for ev in events or []:
            for m in ev.get("markets", []):
                if m.get("id"):
                    return m
    except Exception:
        return None
    return None


async def fetch_live_fee_config(client=None) -> LiveFeeConfig | None:
    """Best-effort fetch of Polymarket's live fee config.

    Algorithm:
      1. Cache-hit if fee_config_cache.json is < TTL old.
      2. Otherwise fetch one active Weather market from gamma.
      3. Probe known fee/rebate fields. If found, return.
      4. Fall back to the hardcoded constants with source="fallback_constant".

    Returns None only if `client` is None and we can't construct one.
    Otherwise always returns a LiveFeeConfig (possibly the fallback).
    """
    cached = _load_cache_if_fresh()
    if cached is not None:
        return cached

    import httpx
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        raw = await _fetch_sample_weather_market(client)
    finally:
        if owns:
            await client.aclose()

    now_iso = datetime.now(timezone.utc).isoformat()
    if raw is None:
        cfg = LiveFeeConfig(
            taker_fee_rate=FEE_RATE_WEATHER,
            maker_rebate_rate=REBATE_RATE_WEATHER,
            sample_market_id=None,
            source="fallback_constant",
            fetched_at_utc=now_iso,
        )
    else:
        rate, source_key = _extract_fee_from_dict(raw)
        rebate = _extract_rebate_from_dict(raw)
        if rate is None:
            cfg = LiveFeeConfig(
                taker_fee_rate=FEE_RATE_WEATHER,
                maker_rebate_rate=rebate or REBATE_RATE_WEATHER,
                sample_market_id=int(raw.get("id", 0)) or None,
                source="fallback_constant",
                fetched_at_utc=now_iso,
            )
        else:
            cfg = LiveFeeConfig(
                taker_fee_rate=rate,
                maker_rebate_rate=rebate,
                sample_market_id=int(raw.get("id", 0)) or None,
                source=f"gamma_market.{source_key}",
                fetched_at_utc=now_iso,
            )

    _save_cache(cfg)
    return cfg


def _load_cache_if_fresh() -> LiveFeeConfig | None:
    if not FEE_CONFIG_CACHE_PATH.exists():
        return None
    try:
        d = json.loads(FEE_CONFIG_CACHE_PATH.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(d["fetched_at_utc"])
        age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
        if age_h > FEE_CONFIG_TTL_HOURS:
            return None
        return LiveFeeConfig(
            taker_fee_rate=float(d["taker_fee_rate"]),
            maker_rebate_rate=d.get("maker_rebate_rate"),
            sample_market_id=d.get("sample_market_id"),
            source=d["source"],
            fetched_at_utc=d["fetched_at_utc"],
        )
    except (OSError, ValueError, KeyError):
        return None


def _save_cache(cfg: LiveFeeConfig) -> None:
    try:
        FEE_CONFIG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEE_CONFIG_CACHE_PATH.write_text(
            json.dumps(cfg.to_jsonable(), indent=2), encoding="utf-8",
        )
    except OSError:
        pass


def warn_if_fee_config_changed(cfg: LiveFeeConfig, tol: float = 1e-4) -> None:
    """Emit a stderr warning if the live taker rate differs from the
    hardcoded `FEE_RATE_WEATHER` by more than `tol`. Cheap startup-time
    sanity check — call once per scan, after `fetch_live_fee_config()`."""
    import sys
    if cfg.source == "fallback_constant":
        return  # nothing to compare against
    delta = abs(cfg.taker_fee_rate - FEE_RATE_WEATHER)
    if delta > tol:
        print(
            f"[fees] WARNING: live taker rate {cfg.taker_fee_rate:.4f} "
            f"(source={cfg.source}) differs from hardcoded "
            f"FEE_RATE_WEATHER={FEE_RATE_WEATHER:.4f}. "
            f"Update weather_bot/fees.py if intentional.",
            file=sys.stderr,
        )
