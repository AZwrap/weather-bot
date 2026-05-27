"""Position sizing via fractional Kelly.

Polymarket binary mechanics (per share):
    pay `c` (the fill price), receive 1 if outcome resolves Yes, else 0.

Per-dollar invested:
    profit = (1 - c)/c  with prob p   (Yes wins)
            = -1                with prob 1 - p (No wins)

Kelly: bet a fraction f of bankroll. Maximising E[log(bankroll)] gives

    f* = (p · b − q) / b           (canonical form, b = (1−c)/c)
       = (p − c) / (1 − c)         (after simplification)

Symmetric for No: substitute (1−p) for p and (1 − yes_bid) for c.

Why fractional Kelly: full Kelly is variance-greedy and assumes p is exact.
Our p has residual estimation error (bias correction, σ inflation are both
imperfect), so we discount Kelly by `kelly_multiplier`:

    1.0  — full Kelly (theoretically optimal, high-variance)
    0.5  — half Kelly (~75 % of growth at half variance)
    0.25 — quarter Kelly (typical systematic-strategy choice)
    0.1  — deci Kelly (paranoid; correct setting before live skill is
           validated against forward-log data)

## Dynamic Kelly (added 2026-05-13)

The base Kelly assumes our `p` is point-known. In practice:
  - Forecast skill rises as the resolution day approaches
  - Wider ensemble spread = more residual uncertainty about `p`

`dynamic_kelly_multiplier()` returns a [0..1.x] factor that combines
both. Per `project_pricing_engine.md` Stage 3. Paper-observed only
until N≥30 days validate the assumption that high-multiplier trades
genuinely outperform low-multiplier ones.
"""
from __future__ import annotations

from typing import Literal

Side = Literal["YES", "NO"]


# Dynamic-Kelly tuning constants. Set conservatively; sweep when paper data
# accumulates (N≥30 days). See project_pricing_engine.md Stage 3.
DYNAMIC_KELLY_TIME_CEILING_H: float = 168.0
"""Horizon (hours) beyond which the time factor saturates at its floor. 7 days."""

DYNAMIC_KELLY_TIME_FLOOR: float = 0.1
"""Minimum value of the time factor — applies at and beyond the ceiling horizon."""

DYNAMIC_KELLY_TIGHTNESS_K: float = 1.0
"""Denominator constant in ensemble_tightness = 1 / (1 + k·σ_total_c).
Lower k = less sensitive to ensemble spread; higher k = more sensitive."""


def dynamic_kelly_multiplier(
    hours_to_resolution: float | None,
    sigma_total_c: float | None,
) -> float:
    """Multiplier in approx [0.02, 1.0] applied on top of the base
    `kelly_multiplier`.

    Formula (per `project_pricing_engine.md` Stage 3):
        confidence_factor = clamp(1 − hrs/168, 0.1, 1.0)
        ensemble_tightness = 1 / (1 + σ_total_c)
        multiplier = confidence_factor × ensemble_tightness

    Concrete examples (σ_total in °C):
        T-12h, σ=0.5: confidence=0.93, tightness=0.67 → mult ≈ 0.62
        T-3d,  σ=1.5: confidence=0.57, tightness=0.40 → mult ≈ 0.23
        T-7d,  σ=3.0: confidence=0.10, tightness=0.25 → mult ≈ 0.025

    Either arg `None` → that factor is 1.0 (graceful degradation). This
    keeps the function callable when caller has partial data and avoids
    surprise zero-sizing.
    """
    if hours_to_resolution is None:
        confidence_factor = 1.0
    else:
        hrs = max(0.0, float(hours_to_resolution))
        raw = 1.0 - (hrs / DYNAMIC_KELLY_TIME_CEILING_H)
        confidence_factor = max(DYNAMIC_KELLY_TIME_FLOOR, min(1.0, raw))

    if sigma_total_c is None or sigma_total_c <= 0:
        tightness = 1.0
    else:
        tightness = 1.0 / (1.0 + DYNAMIC_KELLY_TIGHTNESS_K * float(sigma_total_c))

    return confidence_factor * tightness


def kelly_fraction(our_prob: float, fill_price: float, side: Side) -> float:
    """Full-Kelly fraction of bankroll. Returns 0 when there's no edge."""
    if side == "YES":
        p = our_prob
    elif side == "NO":
        p = 1.0 - our_prob
    else:
        raise ValueError(f"side must be YES or NO, got {side!r}")

    if fill_price <= 0.0 or fill_price >= 1.0:
        return 0.0
    edge = p - fill_price
    if edge <= 0.0:
        return 0.0
    return edge / (1.0 - fill_price)


def position_size_usd(
    our_prob: float,
    fill_price: float,
    side: Side,
    bankroll_usd: float,
    *,
    kelly_multiplier: float = 0.1,
    max_position_usd: float | None = None,
    min_position_usd: float = 1.0,
    liquidity_cap_usd: float | None = None,
    hours_to_resolution: float | None = None,
    sigma_total_c: float | None = None,
) -> float:
    """USD size for a single trade using fractional Kelly.

    Args:
        our_prob: our probability the side wins.
        fill_price: price we'd pay per share for that side
                    (yes_ask for YES, 1 - yes_bid for NO).
        side: "YES" or "NO".
        bankroll_usd: total bankroll allocated to the bot.
        kelly_multiplier: fraction of full Kelly (default 0.1 = deci Kelly).
        max_position_usd: hard per-trade cap.
        min_position_usd: ignore trades below this size (default $1).
        liquidity_cap_usd: don't exceed this fraction of available liquidity
            (typically passed as 0.1 × market 24h volume).
        hours_to_resolution: hours until target_date local midnight close.
            When provided, applies a confidence factor scaling: bigger size
            close to resolution, smaller size 7+ days out. See
            `dynamic_kelly_multiplier`.
        sigma_total_c: predictive distribution stdev (°C). When provided,
            tighter ensembles get bigger size, looser ensembles smaller.

    Both `hours_to_resolution` and `sigma_total_c` are optional — when
    neither is provided, behaviour is identical to pre-2026-05-13 sizing
    (preserving model-driven-paper observation continuity).
    """
    if bankroll_usd <= 0:
        return 0.0
    base = kelly_fraction(our_prob, fill_price, side) * kelly_multiplier
    dyn = dynamic_kelly_multiplier(hours_to_resolution, sigma_total_c)
    f = base * dyn
    size = bankroll_usd * f
    if max_position_usd is not None:
        size = min(size, max_position_usd)
    if liquidity_cap_usd is not None:
        size = min(size, liquidity_cap_usd)
    if size < min_position_usd:
        return 0.0
    return size
