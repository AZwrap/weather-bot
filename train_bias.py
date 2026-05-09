r"""Train per-(station, target) forecast biases and persist as JSON.

The bias table contains one entry per Polymarket market, each with the mean
forecast residual (forecast - observation) over a configurable training window.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python train_bias.py                      # train on default window
    python train_bias.py --train-end 2025-11-02 --days 365
    python train_bias.py --output bias_table.json

For honest backtesting, set --train-end to one day BEFORE the start of your
test window so the periods don't overlap.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_bot.bias import train_bias_table
from weather_bot.locations import MARKETS, STATIONS_BY_ID


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", default=None,
                   help="last day in training window (YYYY-MM-DD). Default: 6 months ago.")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--model", default="ecmwf_ifs025")
    p.add_argument("--output", default="bias_table.json")
    args = p.parse_args()

    if args.train_end:
        train_end = date.fromisoformat(args.train_end)
    else:
        # Default training window is the 365 days ENDING the day before our
        # standard backtest window starts. Backtest end = today - 6 days, span = 180.
        # So test window starts at today - 6 - 179. Training ends one day prior.
        test_start = date.today() - timedelta(days=6 + 179)
        train_end = test_start - timedelta(days=1)

    market_pairs = []
    for sid, t in MARKETS:
        target = "max" if t == "highest" else "min"
        market_pairs.append((STATIONS_BY_ID[sid], target))

    print(
        f"Training bias on {args.days} days ending {train_end} "
        f"(model={args.model}, n={len(market_pairs)} markets)\n"
    )

    table = await train_bias_table(
        market_pairs=market_pairs,
        train_end=train_end,
        train_days=args.days,
        model=args.model,
    )

    print(f"Trained {len(table)} entries:\n")
    print(f"  {'station':<8s}  {'target':<6s}  {'n':>4s}  {'bias_°C':>9s}  {'rmse_°C':>9s}")
    print("  " + "-" * 50)
    for (sid, t), e in sorted(
        table.entries.items(), key=lambda kv: -abs(kv[1].bias_c)
    ):
        print(f"  {sid:<8s}  {t:<6s}  {e.n_days:>4d}  {e.bias_c:>+9.3f}  {e.rmse_c:>9.3f}")

    out = Path(args.output)
    table.save(out)
    print(f"\nSaved → {out.resolve()}")

    # Date-stamped archival copy so we always have the historical bias table
    # for any past forward-log record. Replayed simulations can then use the
    # exact bias-state that was live at any prior time.
    archive_dir = Path("data/bias_table_history")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{train_end.isoformat()}.json"
    table.save(archive_path)
    print(f"Archived → {archive_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
