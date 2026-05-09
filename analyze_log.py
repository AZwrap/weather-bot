r"""Print calibration metrics on resolved forward-log records.

Reports per-(station, target) skill on TRUE 1-day-lead forecasts — the
honest test the historical-forecast-api proxy can't deliver. Build up
≥30 days per station before drawing conclusions; under that the metrics
are noisy.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python analyze_log.py
    python analyze_log.py --min-n 14   # show stations with ≥14 resolved records
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from weather_bot.backtest.metrics import bias, crps_gaussian, mae, rmse
from weather_bot.forward_log import DEFAULT_LOG_PATH, load_records
from weather_bot.pnl import aggregate, simulate_record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    p.add_argument("--min-n", type=int, default=1,
                   help="only include (station, target) pairs with at least N "
                        "resolved records (default 1)")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="USD bankroll for hypothetical PnL (default 1000)")
    p.add_argument("--kelly", type=float, default=0.1,
                   help="Kelly multiplier for PnL sizing (default 0.1)")
    p.add_argument("--max-position", type=float, default=50.0,
                   help="hard per-trade cap for PnL sizing (default 50)")
    p.add_argument("--min-edge", type=float, default=0.05,
                   help="minimum edge for PnL sizing (default 0.05 = 5pp)")
    args = p.parse_args()

    records = load_records(Path(args.log))
    resolved = [r for r in records if r.is_resolved]
    print(
        f"Forward log: {len(records)} total, {len(resolved)} resolved, "
        f"{len(records) - len(resolved)} pending"
    )
    if not resolved:
        print("\nNothing resolved yet. Wait ≥6 days after the first log entry, "
              "then run `python resolve_log.py`.")
        return

    # Group by (station_id, target)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in resolved:
        groups[(r.station_id, r.target)].append(r)

    print()
    print(
        f"{'station':<8s} {'tgt':<3s} {'n':>4s}  "
        f"{'bias':>6s} {'MAE':>5s} {'RMSE':>5s} {'CRPS':>5s}  "
        f"{'σ_total':>7s}  {'training σ':>10s}  Δ"
    )
    print("-" * 80)

    overall_f, overall_o = [], []
    for (sid, target), recs in sorted(groups.items()):
        if len(recs) < args.min_n:
            continue
        f = np.array([r.predictive_mean_c for r in recs])
        o = np.array([r.actual_obs_c for r in recs])

        b = bias(f, o)
        m = mae(f, o)
        r_ = rmse(f, o)

        # CRPS using each record's own σ_total
        crps_vals = []
        for rec in recs:
            crps_vals.append(
                crps_gaussian(
                    np.array([rec.predictive_mean_c]),
                    rec.sigma_total_c,
                    np.array([rec.actual_obs_c]),
                )
            )
        crps_avg = float(np.mean(crps_vals))

        sigma_total_avg = float(np.mean([rec.sigma_total_c for rec in recs]))
        # Compare to training-period σ (the σ used for inflation)
        training_sigma = float(np.mean([rec.sigma_residual_c for rec in recs]))

        # If actual error std much larger than training σ, the inflation factor is too low
        delta_marker = ""
        if r_ > sigma_total_avg * 1.3:
            delta_marker = "↑ inflate more"
        elif r_ < sigma_total_avg * 0.7:
            delta_marker = "↓ over-inflated"

        print(
            f"{sid:<8s} {target:<3s} {len(recs):>4d}  "
            f"{b:>+6.2f} {m:>5.2f} {r_:>5.2f} {crps_avg:>5.2f}  "
            f"{sigma_total_avg:>7.2f}  {training_sigma:>10.2f}  {delta_marker}"
        )

        overall_f.extend(f.tolist())
        overall_o.extend(o.tolist())

    if overall_f:
        f = np.array(overall_f)
        o = np.array(overall_o)
        print("-" * 80)
        print(
            f"{'OVERALL':<8s} {'':3s} {len(f):>4d}  "
            f"{bias(f, o):>+6.2f} {mae(f, o):>5.2f} {rmse(f, o):>5.2f}"
        )

    # ── Hypothetical PnL section ────────────────────────────────────────
    records_with_buckets = [
        r for r in resolved if r.bucket_snapshots is not None
    ]
    if not records_with_buckets:
        print(
            "\nNo resolved records have bucket snapshots yet — PnL section skipped."
        )
        return

    print(
        f"\n=== Hypothetical PnL "
        f"(bankroll=${args.bankroll:,.0f}  Kelly={args.kelly:g}×  "
        f"max=${args.max_position:.0f}  min_edge={args.min_edge:.2%}) ==="
    )
    all_trades = []
    for r in records_with_buckets:
        all_trades.extend(simulate_record(
            r,
            bankroll_usd=args.bankroll,
            kelly_multiplier=args.kelly,
            max_position_usd=args.max_position,
            min_edge=args.min_edge,
        ))

    summary = aggregate(all_trades)
    if summary.n_trades == 0:
        print("  No trades passed the edge filter.")
        return

    print(f"  trades:        {summary.n_trades:,}  "
          f"({summary.n_wins} wins / {summary.n_losses} losses)")
    print(f"  total wagered: ${summary.total_pos_usd:>12,.2f}")
    print(f"  total profit:  ${summary.total_profit_usd:>+12,.2f}")
    print(f"  ROI:           {summary.roi_pct:>+12.2f}%")
    print(f"  win rate:      {summary.win_rate:>12.1%}")

    # Per-station PnL
    print("\n  per-station:")
    by_station: dict[tuple[str, str], list] = {}
    for t in all_trades:
        by_station.setdefault((t.station_id, t.target), []).append(t)

    print(f"    {'station':<8s} {'tgt':<3s}  {'n':>4s}  "
          f"{'wagered':>10s}  {'profit':>10s}  {'ROI':>7s}  win%")
    for (sid, tgt), trades in sorted(by_station.items()):
        s = aggregate(trades)
        if s.n_trades == 0:
            continue
        roi_str = f"{s.roi_pct:+.1f}%" if s.roi_pct is not None else "  —  "
        wr_str = f"{s.win_rate:.0%}" if s.win_rate is not None else " — "
        print(f"    {sid:<8s} {tgt:<3s}  {s.n_trades:>4d}  "
              f"${s.total_pos_usd:>9,.0f}  ${s.total_profit_usd:>+9,.0f}  "
              f"{roi_str:>7s}  {wr_str}")


if __name__ == "__main__":
    main()
