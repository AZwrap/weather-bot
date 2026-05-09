r"""CLI: scan markets, validate against safety config, place orders.

Default mode is **dry-run**. Real submission requires:
  * `--live` flag
  * `enabled=True` in the constructed TradingConfig
  * Interactive confirmation at the prompt
  * No `KILL_SWITCH` file in the project root
  * py-clob-client installed and POLYMARKET_PRIVATE_KEY env var set

Usage:
    .\.venv\Scripts\Activate.ps1
    python place_orders.py                    # dry-run, default safety config
    python place_orders.py --bankroll 500     # smaller bankroll
    python place_orders.py --live             # interactive confirm, then submit

Kill switch:
    touch KILL_SWITCH      # disables all submission immediately
    rm KILL_SWITCH         # re-enables after manual review
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
from weather_bot.execution import (
    ExecutionClient,
    TradingConfig,
    is_kill_switched,
    validate_signal,
)
from weather_bot.scanner import scan


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bias-table", default="bias_table.json")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.1)
    p.add_argument("--max-position", type=float, default=25.0,
                   help="hard per-trade cap (default 25 — execution layer "
                        "uses tighter caps than the scanner default)")
    p.add_argument("--max-total", type=float, default=100.0,
                   help="hard cap on total open + new position USD")
    p.add_argument("--min-edge", type=float, default=0.05)
    p.add_argument("--min-volume", type=float, default=500.0)
    p.add_argument("--allow-non-tier-1", action="store_true",
                   help="trade outside the tier-1 station allow-list — DON'T "
                        "use without forward-log validation")
    p.add_argument("--live", action="store_true",
                   help="actually place orders (default: dry-run only)")
    args = p.parse_args()

    config = TradingConfig(
        enabled=args.live,
        max_total_exposure_usd=args.max_total,
        max_per_trade_usd=args.max_position,
        min_edge=args.min_edge,
        min_volume_24hr=args.min_volume,
        only_tier_1=not args.allow_non_tier_1,
        bankroll_usd=args.bankroll,
        kelly_multiplier=args.kelly,
    )

    if is_kill_switched(config):
        sys.exit(f"!! KILL_SWITCH file present at {config.kill_switch_path} — aborting.")

    bias_path = Path(args.bias_table)
    if not bias_path.exists():
        sys.exit(f"bias_table missing at {bias_path}; run train_bias.py first.")
    bias_table = BiasTable.load(bias_path)

    # Step 1 — generate signals via the scanner with execution-grade caps.
    print("Scanning Polymarket weather markets…")
    signals = await scan(
        bias_table,
        min_edge=args.min_edge,
        min_volume_24hr=args.min_volume,
        bankroll_usd=args.bankroll,
        kelly_multiplier=args.kelly,
        max_position_usd=args.max_position,
    )
    if not signals:
        print("No signals to consider.")
        return

    # Step 2 — pass each through the safety validator with running exposure.
    accepted: list = []
    rejected: list[tuple] = []
    running_exposure = 0.0
    for s in signals:
        # In dry-run we still want to see what WOULD pass. Temporarily flip
        # `enabled` so validate_signal doesn't blanket-reject.
        cfg_for_check = (
            config if config.enabled else _enabled_copy(config)
        )
        v = validate_signal(s, cfg_for_check, running_exposure)
        if v.ok:
            accepted.append(s)
            running_exposure += s.position_usd
        else:
            rejected.append((s, v.reason))

    print(f"\nAccepted: {len(accepted)}  /  Rejected: {len(rejected)}  "
          f"of {len(signals)} signals")
    print(f"Total proposed exposure: ${running_exposure:.2f} "
          f"(cap ${config.max_total_exposure_usd:.2f})")

    if accepted:
        print("\nAccepted trades:")
        for s in accepted:
            print(f"  • {s.station.name:<13s} {s.target:<3s} {s.target_date}  "
                  f"{s.bucket_label:<13s}  {s.side} @ {s.fill_price:.3f}  "
                  f"edge={s.edge:+.1%}  size=${s.position_usd:.2f}")
    if rejected and len(rejected) <= 20:
        print("\nRejected (first 20):")
        for s, reason in rejected[:20]:
            print(f"  ✗ {s.station.name:<13s} {s.target:<3s} {s.bucket_label:<13s}  "
                  f"— {reason}")

    # Step 3 — submit (or dry-run-print).
    if not args.live:
        print("\n[DRY RUN] No orders submitted. Pass --live to submit (with confirmation).")
        return

    if not accepted:
        return

    print("\n!! LIVE MODE !!")
    print(f"  About to submit {len(accepted)} orders totalling ${running_exposure:.2f}")
    print("  This is REAL money. Type 'yes' to confirm, anything else to abort.")
    response = input("> ").strip().lower()
    if response != "yes":
        print("Aborted.")
        return

    client = ExecutionClient.from_env(config)
    bal = client.get_balance_usdc()
    if bal is not None and bal < running_exposure:
        sys.exit(f"!! USDC balance ${bal:.2f} < required exposure ${running_exposure:.2f}")

    print("\nSubmitting…")
    for s in accepted:
        result = client.submit_order(s)
        marker = "✓" if result.ok else "✗"
        print(f"  {marker} {result.message or result.order_id}")


def _enabled_copy(config: TradingConfig) -> TradingConfig:
    """Return a config copy with enabled=True, for validation in dry-run."""
    from dataclasses import replace
    return replace(config, enabled=True)


if __name__ == "__main__":
    asyncio.run(main())
