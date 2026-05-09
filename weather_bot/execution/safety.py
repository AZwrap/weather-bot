"""Pre-trade validation and hard limits for live trading.

Default config is paranoid: tiny exposure caps, tier-1 only, kill switch
checked. The bot must NEVER trade without these gates passing.

Production checklist before flipping `enabled=True`:
  □ Bot wallet is dedicated (NOT your main MetaMask).
  □ Bot wallet holds only the capital you can afford to lose to a VPS compromise.
  □ Forward-log has ≥30 days of resolved records and reliability is calibrated.
  □ Bias table retrained within the last 14 days.
  □ Kill switch tested (touch KILL_SWITCH; bot exits without trading).
  □ Confirm USDC allowance approved on Polygon for the CTF exchange.
  □ Max-total-exposure starts small and increases only after live verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..scanner import TradeSignal


# Stations the backtest classified as ★★★ (MAE < 0.5°C, |bias| < 0.5°C, beats
# persistence). Trading is restricted to these by default; expand explicitly.
TIER_1_STATIONS: frozenset[tuple[str, str]] = frozenset({
    ("EGLC", "max"), ("LFPB", "max"), ("LEMD", "max"), ("KLGA", "max"),
    ("UUWW", "max"), ("EPWA", "max"), ("EDDM", "max"), ("KORD", "max"),
    ("KHOU", "max"), ("KSEA", "max"), ("KATL", "max"), ("KAUS", "max"),
    ("KDAL", "max"), ("SAEZ", "max"), ("EHAM", "max"), ("ZBAA", "max"),
    ("ZHHH", "max"), ("ZUUU", "max"), ("ZUCK", "max"), ("ZSQD", "max"),
    ("RKPK", "max"), ("OPMR", "max"), ("WMKK", "max"), ("VILK", "max"),
    ("RCSS", "max"), ("NZWN", "max"),
    ("EGLC", "min"), ("LFPB", "min"), ("KLGA", "min"),
})


@dataclass
class TradingConfig:
    """Hard limits enforced before any order is submitted."""

    enabled: bool = False
    """MUST be set to True (and confirmed at the CLI) to actually submit orders."""

    kill_switch_path: Path = Path("KILL_SWITCH")
    """Touch this file to force-disable trading without stopping the process."""

    max_total_exposure_usd: float = 100.0
    """Hard cap on the sum of open + new positions in USD."""

    max_per_trade_usd: float = 25.0
    """Hard cap on any single order in USD."""

    min_edge: float = 0.05
    """Skip signals with edge below this fraction."""

    min_volume_24hr: float = 500.0
    """Skip signals in low-liquidity markets."""

    only_tier_1: bool = True
    """Restrict to stations in TIER_1_STATIONS. Override with care."""

    bankroll_usd: float = 1000.0
    """Notional bankroll for position sizing."""

    kelly_multiplier: float = 0.1
    """Deci-Kelly default — DO NOT raise without forward-log validation."""

    require_min_n_resolved: int = 30
    """If `forward_log_records_resolved` is below this, skip live trading.
    Verified upstream by the CLI; this is informational here."""

    extra_allowed_stations: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    """Additional (station_id, target) pairs explicitly opted in."""


@dataclass
class TradeValidation:
    ok: bool
    reason: str = ""


def is_kill_switched(config: TradingConfig) -> bool:
    return config.kill_switch_path.exists()


def validate_signal(
    signal: TradeSignal,
    config: TradingConfig,
    current_exposure_usd: float = 0.0,
) -> TradeValidation:
    """Run all hard-limit checks on a single signal."""
    if not config.enabled:
        return TradeValidation(False, "trading is not enabled in config")
    if is_kill_switched(config):
        return TradeValidation(False, f"kill switch present at {config.kill_switch_path}")

    pair = (signal.station.station_id, signal.target)
    if config.only_tier_1 and pair not in (TIER_1_STATIONS | config.extra_allowed_stations):
        return TradeValidation(False, f"{pair} not in tier-1 allow-list")

    if signal.edge < config.min_edge:
        return TradeValidation(False, f"edge {signal.edge:.3f} < min_edge {config.min_edge}")
    if signal.volume_24hr < config.min_volume_24hr:
        return TradeValidation(
            False, f"vol24 ${signal.volume_24hr:,.0f} < min ${config.min_volume_24hr:,.0f}"
        )
    if signal.position_usd <= 0:
        return TradeValidation(False, "computed position is $0")
    if signal.position_usd > config.max_per_trade_usd:
        return TradeValidation(
            False,
            f"position ${signal.position_usd:.2f} > per-trade cap ${config.max_per_trade_usd}",
        )
    if current_exposure_usd + signal.position_usd > config.max_total_exposure_usd:
        return TradeValidation(
            False,
            f"would exceed total cap ${config.max_total_exposure_usd} "
            f"(current ${current_exposure_usd:.2f} + new ${signal.position_usd:.2f})",
        )
    return TradeValidation(True)
