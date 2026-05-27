# Weather Bot — lite rebuild

A trimmed Polymarket weather-market bot. Three strategies only:

1. **METAR early-tail** — fire when a tail bucket is mathematically locked
   in by the day's monotonic extreme.
2. **Layer 7 guaranteed_no_buy** — FAK NO at ≤$0.99 on dead buckets
   (peak past edge).
3. **V2 conditional preposit** — GTC NO maker @ $0.82 when another bucket
   in the same event reaches yes_ask ≥ $0.80. **Paper-only at launch
   (`V2_ENABLED=False`).**

## Before you trust this

This bot was previously deployed live from 2026-05-15 → 2026-05-26 and
**bled −$41.02 over 11 days**. It was decommissioned and rebuilt in
this slim form. The decommission memory at
`.claude/projects/<project>/memory/project_decommission_2026-05-26.md`
contains the full post-mortem. Key empirical numbers from that run:

| strategy | live win rate | breakeven | verdict |
| --- | --- | --- | --- |
| NO_momentum @ $0.78 | 52.7% (n=165) | 78% | **bled**, removed |
| Layer 7 @ $0.97 | 92.3% (n=13) | 97% | **bled −5pp**, kept but watch |
| Live-bucket arb | 46.7% (n=15) | 47% | marginal, removed |
| METAR early-tail | 0% FP cohort | — | **the one that worked** |

The rebuild assumes nothing about today's market conditions. Before
flipping any live switch, accumulate ≥30 paper resolutions per strategy
and re-validate.

## Layout

- `slim_scan.py` — single entry point. Runs all three strategies in one
  pass, paper-trade by default.
- `weather_bot/` — package. Strategy modules: `guaranteed_no_buy.py`,
  `v2_conditional_preposit.py`, `intraday.py` (early-tail only).
  Defensive: `portfolio.py`, `cap_budget.py`, `drawdown_breaker.py`,
  `exclusions.py`, `alerts.py`, `fees.py`.
- `data/` — runtime state. `excluded_stations.json` is the hard
  station blacklist (DNMM, ZGSZ, WIHH, ZSQD as of decommission).
- `analyze_log.py`, `resolve_log.py`, `audit_resolutions.py` — paper
  audit + post-hoc resolution.

## Running

```
python slim_scan.py                       # dry-run; writes paper logs only
python slim_scan.py --strategies metar     # METAR only
LIVE_OK=1 python slim_scan.py --live       # live (requires both flags)
```

A file named `KILL_SWITCH` at the repo root halts every scan before any
network call.

## Cross-references in memory

- `project_decommission_2026-05-26.md` — why the previous run halted
- `project_strategy_levers_post_day2.md` — the Polymarket oracle bug +
  market archive (May 2026) that changed the trading environment
- `polymarket_sdk_v2_migration.md` — the SDK v2 requirements that still
  apply if you wire a real CLOB client
- `polymarket_live_trading_lessons.md` — the operational gotchas

## What is NOT here on purpose

`no_momentum`, `live_bucket_arb`, `cross_up_cancel`,
`shadow_*`, `cohort_filter`, `per_station_thresholds`,
`market_drift_cancel`, `time_of_day_filter`, `value_avg_paper`,
`polymarket_csv_ingest`, `polymarket_ws`, `daemon.py` — all on the
`multimodel-logging` branch if you ever need to compare.
