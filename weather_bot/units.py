"""Temperature unit handling.

Internal representation throughout the codebase is degrees Celsius. Use these
helpers (or the unit-aware methods on `TempDistribution`) at the boundary
where Polymarket markets are parsed — Polymarket uses °F for US cities and
°C elsewhere.
"""
from __future__ import annotations

from typing import Literal

Unit = Literal["C", "F"]


def to_celsius(value: float, unit: Unit) -> float:
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    raise ValueError(f"Unknown unit: {unit!r} (expected 'C' or 'F')")


def from_celsius(value_c: float, unit: Unit) -> float:
    if unit == "C":
        return value_c
    if unit == "F":
        return value_c * 9.0 / 5.0 + 32.0
    raise ValueError(f"Unknown unit: {unit!r} (expected 'C' or 'F')")


def f_to_c(value_f: float) -> float:
    return (value_f - 32.0) * 5.0 / 9.0


def c_to_f(value_c: float) -> float:
    return value_c * 9.0 / 5.0 + 32.0
