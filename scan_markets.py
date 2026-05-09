r"""Scan every active Polymarket weather market for mispriced buckets.

Loads `bias_table.json`, fetches live ensemble forecasts, computes our
probability per bucket, compares to the market's bid/ask, and prints
ranked trade signals.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python scan_markets.py
    python scan_markets.py --min-edge 0.05 --min-volume 500
    python scan_markets.py --only EGLC,LFPB,KLGA
    python scan_markets.py --top 30
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.bias import BiasTable
from weather_bot.scanner import TradeSignal, scan


def _format_signal(s: TradeSignal) -> str:
    bid = f"{s.yes_bid:.3f}" if s.yes_bid is not None else "  -  "
    ask = f"{s.yes_ask:.3f}" if s.yes_ask is not None else "  -  "
    return (
        f"{s.station.name:<13s}  "
        f"{s.target:<3s}  "
        f"{str(s.target_date):<10s}  "
        f"{s.bucket_label:<13s}  "
        f"ours={s.our_prob:>5.1%}  "
        f"mkt={s.yes_implied:>5.1%}  "
        f"{s.side:<3s} @ {s.fill_price:>5.3f}  "
        f"edge={s.edge:>+5.1%}  "
        f"K={s.kelly_full:>5.1%}  "
        f"size=${s.position_usd:>5.2f}  "
        f"vol24={s.volume_24hr:>8,.0f}  "
        f"σ_tot={s.sigma_total_c:.2f}  "
        f"bias={s.bias_applied_c:+.2f}"
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bias-table", default="bias_table.json")
    p.add_argument("--min-edge", type=float, default=0.05,
                   help="minimum probability edge (default 0.05 = 5pp)")
    p.add_argument("--min-volume", type=float, default=0.0,
                   help="minimum 24h market volume (USD)")
    p.add_argument("--only", default=None,
                   help="comma-separated station IDs to restrict to (e.g. EGLC,LFPB)")
    p.add_argument("--top", type=int, default=50,
                   help="show only the top N ranked signals (default 50)")
    p.add_argument("--include-today", action="store_true",
                   help="include events whose target day has already started "
                        "(by default these are filtered out as already-observed)")
    p.add_argument("--no-inflate", action="store_true",
                   help="disable σ inflation (use raw ensemble spread). Default: "
                        "σ_total² = σ_ensemble² + σ_residual² for honest calibration.")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="USD bankroll for position sizing (default 1000)")
    p.add_argument("--kelly", type=float, default=0.1,
                   help="Kelly multiplier (default 0.1 = deci Kelly)")
    p.add_argument("--max-position", type=float, default=50.0,
                   help="hard cap per trade in USD (default 50)")
    p.add_argument("--liquidity-cap", type=float, default=0.1,
                   help="cap position at this fraction of market 24h vol (default 0.1)")
    p.add_argument("--no-zero-pos", action="store_true",
                   help="hide signals whose deci-Kelly position is below the min")
    args = p.parse_args()

    bias_path = Path(args.bias_table)
    if not bias_path.exists():
        sys.exit(
            f"{bias_path} not found. Train one first with `python train_bias.py`."
        )
    bias_table = BiasTable.load(bias_path)
    print(f"Loaded bias table: {len(bias_table)} entries from {bias_path}")

    only_ids = (
        set(s.strip().upper() for s in args.only.split(",")) if args.only else None
    )

    print(
        f"Scanning Polymarket weather markets… "
        f"(bankroll=${args.bankroll:,.0f}  Kelly={args.kelly:g}×  "
        f"max=${args.max_position:.0f}  σ_inflate={not args.no_inflate})"
    )
    signals = await scan(
        bias_table,
        min_edge=args.min_edge,
        min_volume_24hr=args.min_volume,
        only_station_ids=only_ids,
        include_today=args.include_today,
        inflate_sigma=not args.no_inflate,
        bankroll_usd=args.bankroll,
        kelly_multiplier=args.kelly,
        max_position_usd=args.max_position,
        liquidity_cap_fraction=args.liquidity_cap,
    )
    if args.no_zero_pos:
        signals = [s for s in signals if s.position_usd > 0]

    print()
    print(f"Found {len(signals)} signals with edge ≥ {args.min_edge:.1%} and vol24 ≥ ${args.min_volume:,.0f}")
    print()
    if not signals:
        return

    print("rank  " + _format_signal_header())
    print("-" * 175)
    for i, s in enumerate(signals[: args.top], start=1):
        print(f"{i:>4d}  {_format_signal(s)}")


def _format_signal_header() -> str:
    return (
        f"{'station':<13s}  "
        f"{'tgt':<3s}  "
        f"{'date':<10s}  "
        f"{'bucket':<13s}  "
        f"{'ours':>6s}  "
        f"{'mkt':>6s}  "
        f"{'side @ fill':<13s}  "
        f"{'edge':>6s}  "
        f"{'kelly':>6s}  "
        f"{'size$':>7s}  "
        f"{'vol24':>9s}  σ_tot bias"
    )


if __name__ == "__main__":
    asyncio.run(main())
