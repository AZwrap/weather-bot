"""Polymarket fee model (Weather markets, confirmed 2026-05-25).

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
