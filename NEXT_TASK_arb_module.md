# Next task: Build the bucket-sum-to-1 arbitrage module

## What you need to know

On 2026-05-09, we discovered Polymarket weather markets have **real
structural arbitrage** in their bucket sum-to-1 pricing. For each
(station, target, target_date), the YES asks across all buckets
should sum to ~$1.00 (since exactly one bucket wins). When they sum
to less than $1.00, buying every bucket's YES token guarantees $1
return for less than $1 cost.

We scanned 1,806 snapshots of forward-log data and found:
- **30 YES-side arb opportunities** (sum < $1.00)
- Best margin: $0.062 (6.6% guaranteed ROI per capture)
- Concentrated on lower-volume stations: KHOU, LTAC, WMKK
- Estimated EV: $700-1500/year on a $1k bankroll, with **zero model risk**

Test script that produced these numbers: `backtest_bucket_sum_arb.py`
(in project root, also on VPS at `/root/Weather_Bot/`). Re-run any
time to confirm arbs still appear in fresh data.

## Why this matters separately from the main bot

The forecast-driven bot needs ~30 days of resolved data to validate
calibration before going live. This arb module needs none of that —
it's pure math. It can go live FIRST and run alongside the main bot.
Even if the main bot has zero edge (worst case), the arb module
prints small consistent money. Insurance against model failure.

## Implementation goal

Build a standalone module `weather_bot/arb_scanner.py` that:

1. Polls Polymarket CLOB every ~60 seconds for all weather markets.
2. For each (station, target, target_date) market group, computes
   `sum(yes_ask)` across all buckets.
3. Flags any group where `sum < 0.97` (margin > 3¢ to clear execution
   friction; we measured 6¢ best so 3¢ is a reasonable floor).
4. For flagged groups, submits **simultaneous** buy-YES limit orders
   on every bucket at the current ask, sized so total cost = e.g. $20.
5. Records the attempt (asks at detection, asks at submission, fills,
   actual P&L on resolution).

## The CRITICAL unknown — live persistence

Our test was on snapshot data (every 20 min). We don't know whether
detected arbs SURVIVE the time it takes to submit 11 orders to
Polymarket. Three possible outcomes from the live smoke test:

- **Arb persists for ~10 seconds**: easy capture, automate it.
- **Arb persists for ~1 second**: hard but doable with batched
  submission via py-clob-client.
- **Arb is "snapshot artifact"**: bid/ask data was stale at one or
  more buckets; by the time we'd detect, prices have already updated.
  No real opportunity.

**You MUST do a manual smoke test before automating.** Steps:

1. Run `backtest_bucket_sum_arb.py` against fresh data.
2. When it flags an arb, immediately go to the Polymarket UI for that
   market.
3. Check whether the asks shown match the snapshot, AND whether the
   sizes available at the asks total at least $20-50 across all buckets.
4. If yes both → arb is real and capturable. Automate.
5. If no → arb is data-quality only. Add a "freshness" filter (only
   trust asks updated within last N minutes) and re-test.

## Files to read first

- `backtest_bucket_sum_arb.py` — the existing test script
- `weather_bot/polymarket.py` — has `fetch_clob_prices_batch()` already,
  reuse for the arb scanner's polling
- `weather_bot/forward_log.py` — understand how bucket snapshots are
  structured (each record has `bucket_snapshots: list[BucketSnapshot]`)
- Memory: `project_strategy_ideas.md` § A — the documented arb finding
- Memory: `project_simulator_bugs_for_live.md` — bankroll tracking
  applies here too: cap aggregate arb capital at e.g. $200 max deployed

## Out-of-scope for this task

- Don't touch the main forecast-driven bot logic. This is a parallel
  module.
- Don't try to integrate with the maker/taker debate. Arb is taker-only
  by definition (need to grab liquidity at the ask).
- Don't add fancy filtering. The math is the math: if sum < threshold,
  buy. No model logic.
- Don't try to automate before the manual smoke test confirms
  persistence.

## Acceptance criteria

1. `arb_scanner.py` runs as a separate cron job (every 1-5 min).
2. Logs every detected arb with detection-time asks, post-submission
   asks, and outcome.
3. Manual smoke test of at least 5 detected arbs, with documented
   outcomes (filled / partial / missed).
4. Daily summary in dashboard or log: # arbs detected, # captured,
   total realized P&L from arb module specifically.
5. Cap deployed capital at $200 across all open arb positions until
   proven over 2 weeks of running.

## Start here

```bash
# Re-confirm arbs still appear in fresh data
ssh root@209.250.227.207 'cd /root/Weather_Bot && .venv/bin/python backtest_bucket_sum_arb.py'

# Then: build arb_scanner.py based on the test script's logic.
# Then: smoke test before automating order submission.
```
