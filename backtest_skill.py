r"""Run the forecast-skill backtest across every Polymarket weather market.

Iterates `weather_bot.locations.MARKETS` (59 markets as of 2026-05-07),
fetching the appropriate target (highest/lowest = daily max/min) for each.

Usage (PowerShell):
    .\.venv\Scripts\Activate.ps1
    python backtest_skill.py

CLI flags:
    --days N        backtest window length in days (default 180)
    --model NAME    Open-Meteo model id (default ecmwf_ifs025)
    --full          print the full per-market report including reliability table
    --only S        restrict to a single station id substring (e.g. "EGLC", "HKO")
    --target T      restrict to "max" or "min" (default: both)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path

from weather_bot.backtest import run_skill_backtest
from weather_bot.bias import BiasTable
from weather_bot.locations import MARKETS, STATIONS_BY_ID


def _summary_line(market_label: str, report) -> str:
    return (
        f"{market_label:<24s}  n={report.n_days:>3d}  "
        f"bias={report.forecast_bias_c:+.2f}  "
        f"MAE={report.forecast_mae_c:.2f}  "
        f"RMSE={report.forecast_rmse_c:.2f}  "
        f"σ={report.residual_sigma_c:.2f}  "
        f"Brier={report.brier:.4f}  "
        f"CRPS={report.crps_c:.3f}  |  "
        f"persist={report.persistence_mae_c:.2f}  "
        f"clim={report.climatology_mae_c:.2f}"
    )


def _verdict(report) -> str:
    """Quick visual tier marker."""
    if report.forecast_mae_c >= report.persistence_mae_c:
        return "✗ skip"
    if report.forecast_mae_c < 0.5 and abs(report.forecast_bias_c) < 0.5:
        return "★★★"
    if report.forecast_mae_c < 1.0:
        return "★★ bias-fix"
    return "★ check"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--model", default="ecmwf_ifs025")
    p.add_argument("--full", action="store_true")
    p.add_argument("--only", default=None)
    p.add_argument("--target", choices=["max", "min", "both"], default="both")
    p.add_argument("--bias-table", default=None,
                   help="Path to bias_table.json. If set, the run is "
                        "post-correction; defaults to no correction.")
    p.add_argument("--compare", action="store_true",
                   help="Run twice (with and without bias correction from "
                        "--bias-table) and print delta per market.")
    args = p.parse_args()

    bias_table: BiasTable | None = None
    if args.bias_table or args.compare:
        if not args.bias_table:
            sys.exit("--compare requires --bias-table")
        bias_table = BiasTable.load(Path(args.bias_table))

    selected: list[tuple[str, str]] = []
    for sid, t in MARKETS:
        target = "max" if t == "highest" else "min"
        if args.target != "both" and target != args.target:
            continue
        if args.only and args.only.lower() not in sid.lower():
            continue
        selected.append((sid, target))

    if not selected:
        sys.exit("No markets matched filters")

    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=args.days - 1)
    print(
        f"Skill backtest: {start} → {end}  "
        f"(model={args.model}, n={len(selected)} markets across "
        f"{len({sid for sid, _ in selected})} stations)\n"
    )

    sem = asyncio.Semaphore(2)

    async def bounded(sid: str, target: str, bias_c: float = 0.0):
        station = STATIONS_BY_ID[sid]
        async with sem:
            return await run_skill_backtest(
                station.to_location(),
                days=args.days,
                end=end,
                model=args.model,
                target=target,
                bias_c=bias_c,
                icao=station.station_id,
            )

    if args.compare:
        # Two passes: without correction, with correction. Same fetch each time
        # but Open-Meteo caches at edge so this isn't 2× cost in practice.
        results_raw = await asyncio.gather(
            *(bounded(sid, t) for sid, t in selected),
            return_exceptions=True,
        )
        results_corr = await asyncio.gather(
            *(
                bounded(sid, t, bias_table.get(sid, t))
                for sid, t in selected
            ),
            return_exceptions=True,
        )
        _print_compare(selected, results_raw, results_corr, bias_table)
        return

    bias_lookup = bias_table.get if bias_table else (lambda *_: 0.0)
    results = await asyncio.gather(
        *(bounded(sid, t, bias_lookup(sid, t)) for sid, t in selected),
        return_exceptions=True,
    )

    label_post = " (bias-corrected)" if bias_table else ""
    print(
        f"{'market' + label_post:<24s}  {'n':>5s}  {'bias':>5s}  {'MAE':>4s}  {'RMSE':>4s}  "
        f"{'σ':>4s}  {'Brier':>6s}  {'CRPS':>5s}  |  {'persist':>7s}  {'clim':>4s}  verdict"
    )
    print("-" * 130)

    by_verdict: dict[str, list] = {"★★★": [], "★★ bias-fix": [], "★ check": [], "✗ skip": []}

    for (sid, target), r in zip(selected, results):
        station = STATIONS_BY_ID[sid]
        label = f"{station.name} [{target}]" + ("" if station.unit == "C" else " (°F)")
        if isinstance(r, Exception):
            print(f"{label:<24s}  -- {r}")
            continue
        verdict = _verdict(r)
        print(f"{_summary_line(label, r)}  {verdict}")
        by_verdict[verdict].append((label, r))

    print()
    print("Tier counts:")
    for v in ("★★★", "★★ bias-fix", "★ check", "✗ skip"):
        print(f"  {v:<14s} {len(by_verdict[v])} markets")


def _print_compare(selected, results_raw, results_corr, bias_table) -> None:
    print(
        f"{'market':<24s}  {'bias_train':>10s}  "
        f"{'MAE_raw':>7s} → {'MAE_corr':>8s}  "
        f"{'Brier_raw':>9s} → {'Brier_corr':>10s}  "
        f"{'Δ MAE':>7s}  {'Δ Brier':>8s}"
    )
    print("-" * 110)
    n_better, n_worse, total_brier_delta = 0, 0, 0.0
    for (sid, target), r_raw, r_corr in zip(selected, results_raw, results_corr):
        station = STATIONS_BY_ID[sid]
        label = f"{station.name} [{target}]" + ("" if station.unit == "C" else " (°F)")
        if isinstance(r_raw, Exception) or isinstance(r_corr, Exception):
            print(f"{label:<24s}  --")
            continue
        bias_train = bias_table.get(sid, target)
        d_mae = r_corr.forecast_mae_c - r_raw.forecast_mae_c
        d_brier = r_corr.brier - r_raw.brier
        marker = "✓" if d_brier < 0 else ("=" if abs(d_brier) < 1e-4 else "✗")
        print(
            f"{label:<24s}  {bias_train:>+10.3f}  "
            f"{r_raw.forecast_mae_c:>7.2f} → {r_corr.forecast_mae_c:>8.2f}  "
            f"{r_raw.brier:>9.4f} → {r_corr.brier:>10.4f}  "
            f"{d_mae:>+7.2f}  {d_brier:>+8.4f}  {marker}"
        )
        if d_brier < -1e-4:
            n_better += 1
        elif d_brier > 1e-4:
            n_worse += 1
        total_brier_delta += d_brier
    n = len(selected)
    print()
    print(f"Brier improved on {n_better}/{n} markets, worsened on {n_worse}/{n}, "
          f"net Δ = {total_brier_delta:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
