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
"""
from __future__ import annotations

from typing import Literal

Side = Literal["YES", "NO"]


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
    """
    if bankroll_usd <= 0:
        return 0.0
    f = kelly_fraction(our_prob, fill_price, side) * kelly_multiplier
    size = bankroll_usd * f
    if max_position_usd is not None:
        size = min(size, max_position_usd)
    if liquidity_cap_usd is not None:
        size = min(size, liquidity_cap_usd)
    if size < min_position_usd:
        return 0.0
    return size
