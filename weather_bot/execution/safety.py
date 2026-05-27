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

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from ..polymarket import (
    OrderBookDepth,
    POLYMARKET_MARKETABLE_MIN_USD,
    POLYMARKET_RESTING_MIN_SHARES,
    fetch_orderbook_depths_batch,
    marketable_passes_min,
    resting_passes_min,
)
from ..scanner import TradeSignal

if TYPE_CHECKING:
    from ..bias import BiasTable
    from ..portfolio import Portfolio


# Tier-1 stations: the 42 markets classified ★★ bias-fix in the
# 2026-05-13 skill backtest (180-day window, post-bias-correction).
# Criteria: MAE < 1.0°C AND beats persistence baseline.
# Trading is restricted to these by default; expand explicitly via
# `extra_allowed_stations` (e.g. for ★ check stations like NYC max,
# Taipei, Tokyo max, Chongqing, NYC min).
#
# Re-derive with: `python backtest_skill.py --days 180 --bias-table bias_table.json`
# Cadence: quarterly, or after any bias retrain. See project_backtest_tiers.md.
#
# NOTE: Good forecast skill (★★) does NOT mean reliable Polymarket
# resolution. Watchlist verification 2026-05-14: OEJN, MPMG, FACT,
# RPLL all show 100% ASOS=Polymarket agreement over 5-6 days. CLEARED.
# All TIER_1 stations are now trusted for live except the explicit
# exclusions below.
#
# HARD-EXCLUDED via data/excluded_stations.json (runtime block at
# validate_signal time — these still appear in TIER_1 for
# forecast-skill classification, but won't trade live):
#
#   DNMM (Lagos)    — never in TIER_1 (★ check). Excluded for 3-way
#                      source disagreement on May 12 2026 (Wunderground
#                      91°F, ASOS 39°C, Polymarket 37°C+).
#   ZGSZ (Shenzhen) — in TIER_1 (★★). User-verified May 9-13 2026:
#                      our ASOS disagrees with Polymarket by 1-3°C
#                      mixed direction; on May 9 even WG (24.4°C) and
#                      Polymarket (25°C bucket) disagreed. Multi-source
#                      uncertainty — can't model reliably with current
#                      truth feed. Future: Wunderground PWS fallback
#                      (see project_oracle_source_risk.md).
#
# NZWN (Wellington) WAS briefly excluded but it was a false alarm
# caused by a timezone off-by-one in event_target_date. Fixed 2026-05-14
# (subtract 1s from end_date before astimezone). NZWN is fine to trade.
#
# See project_oracle_source_risk.md for the audit details + future
# Wunderground PWS fallback design (post-launch enhancement).
TIER_1_STATIONS: frozenset[tuple[str, str]] = frozenset({
    # ── United States (°F) — 8 markets ─────────────────────────────────
    ("KORD", "max"), ("KMIA", "max"), ("KDAL", "max"), ("KHOU", "max"),
    ("KSEA", "max"), ("KAUS", "max"), ("KATL", "max"),
    ("KMIA", "min"),
    # ── Europe (°C) — 12 markets ───────────────────────────────────────
    ("EGLC", "max"), ("LEMD", "max"), ("LFPB", "max"), ("EPWA", "max"),
    ("EDDM", "max"), ("EHAM", "max"), ("LIMC", "max"),
    ("UUWW", "max"), ("LTFM", "max"), ("LTAC", "max"),
    ("EGLC", "min"), ("LFPB", "min"),
    # ── Asia (°C) — 15 markets ─────────────────────────────────────────
    ("ZBAA", "max"), ("ZSPD", "max"), ("ZHHH", "max"), ("ZUUU", "max"),
    ("ZGSZ", "max"), ("ZSQD", "max"),
    ("RKSI", "max"), ("RKPK", "max"),
    ("WSSS", "max"), ("WMKK", "max"), ("RPLL", "max"), ("VILK", "max"),
    ("OEJN", "max"),  # oracle-watch
    ("RJTT", "min"), ("RKSI", "min"),
    # ── Oceania — 1 market ─────────────────────────────────────────────
    ("NZWN", "max"),
    # ── Africa — 1 market ──────────────────────────────────────────────
    ("FACT", "max"),  # oracle-watch
    # ── Latin America — 4 markets ──────────────────────────────────────
    ("MMMX", "max"), ("MPMG", "max"),  # MPMG = oracle-watch
    ("SAEZ", "max"), ("SBGR", "max"),
    # ── North America (Canada) — 1 market ──────────────────────────────
    ("CYYZ", "max"),
})


@dataclass
class TradingConfig:
    """Hard limits enforced before any order is submitted."""

    enabled: bool = False
    """MUST be set to True (and confirmed at the CLI) to actually submit orders."""

    kill_switch_path: Path = Path("KILL_SWITCH")
    """Touch this file to force-disable trading without stopping the process."""

    max_total_exposure_usd: float = 100.0
    """PER-SCAN hard cap on the sum of NEW positions in ONE invocation
    of place_orders.py. Reset to 0 at each cron tick — does NOT track
    concurrent open positions across scans. For portfolio-level caps
    use `portfolio_cap_usd` below."""

    max_per_trade_usd: float = 25.0
    """Hard cap on any single order in USD."""

    # ── Portfolio-level caps (added 2026-05-14, revised 2026-05-14) ────
    # All four enforced via Portfolio.would_exceed_cap() at submit time.
    # See weather_bot/portfolio.py for the cluster taxonomy + correlation
    # weighting that backs these.
    #
    # Defaults sized for $500 test bankroll. With --adaptive-bankroll
    # (default ON in place_orders.py), these are recomputed each scan
    # from realized PnL so caps grow with the bankroll up to $2k.
    portfolio_cap_usd: float = 150.0
    """Total $ across all CURRENTLY OPEN positions (persistent across
    cron runs via data/portfolio.json). 30% of $500 test bankroll.
    SHRUNK 2026-05-19 from $400 → $150 for the re-validation phase after
    the Polymarket archive event. Sized to ~3× the $50/day deployment so
    a 3-day overlap is possible but total exposure is capped at 30%
    bankroll. Original $400 sizing returns once 7+ days of post-archive
    stable trading + bankroll growth justify scaling caps back up."""

    per_region_cap_usd: float = 35.0
    """Max $ of open positions in any one synoptic-weather region
    (US_East, Europe_West, Asia_East, etc.). 7% of $500 test bankroll.
    SHRUNK 2026-05-19 from $100 → $35. The Miami archive event ($46
    cost basis on one city) showed single-region wipe-out is the dominant
    tail risk; capping at $35 means no single platform-level event on a
    single city can erase more than $35. Asia_East has 12 stations vs
    Oceania's 1; uniform cap means Asia naturally saturates first under
    correlated synoptic events, which is the right behavior for cluster-
    bust protection. Original $100 returns when caps scale back."""

    per_event_cap_usd: float = 20.0
    """Max $ across all 11 bucket positions of one Polymarket event.
    SHRUNK 2026-05-19 from $55 → $20 for the re-validation phase. Original
    $55 was sized at 11 buckets × $5/trade to cover every bucket as a
    structural diversifier (exactly 9 win, 1 cross-up). $20 still permits
    ~4 buckets per event — preserves diversification on the most-likely
    buckets while limiting exposure when oracle-bug or archive events
    can void the whole event in one shot. Returns to $55 alongside the
    portfolio_cap restoration."""

    enable_portfolio_kelly: bool = True
    """When True, scale per-trade size down by
    `Portfolio.portfolio_kelly_multiplier()` which discounts capital
    that's already concentrated in correlated positions. Disable to
    revert to naive per-trade Kelly (only the hard caps apply)."""

    min_edge: float = 0.05
    """Skip signals with edge below this fraction."""

    min_volume_24hr: float = 500.0
    """Skip signals in low-liquidity markets."""

    only_tier_1: bool = True
    """Restrict to stations in TIER_1_STATIONS. Override with care."""

    bankroll_usd: float = 500.0
    """Notional bankroll for position sizing. Phase-1/Week-1 live default: $500.
    Adaptive bankroll lifts effective caps with realized PnL up to $2k ceiling
    (see `weather_bot/portfolio.py` ADAPTIVE_BANKROLL_CEILING)."""

    metar_max_ask: float = 0.92
    """Max ask price the bot will pay on METAR FAK fires.
    History:
      - 2026-05-15: lowered $0.95 → $0.85 (Variant B shipping config).
      - 2026-05-25: raised $0.85 → $0.92. Post-Polymarket-archive (May 18),
        book depth and width regressed; books on past-bucket markets
        consistently clear at $0.86-0.95, gating out METAR entirely
        (0 fires in 7 days). Empirical FP-rate work (project_metar_fp_rate.md)
        validated $0.95 as the EV ceiling; $0.92 keeps a $0.03 safety
        buffer vs that ceiling while reopening the fire path.
    Rationale: per project_metar_fp_rate.md the FP rate is ~0.95%; markets
    at $0.92+ already reflect near-certain resolution and the remaining
    $0.08 captures fee + slippage + FP-rate variance with margin. Used both
    as the gate (skip if top_ask >= this) AND as the FAK limit price
    (= ceiling for depth-
    walking). Polymarket fills cheapest asks first, so submitting at
    ceiling captures the backtest's depth-aware-metar behavior — exchange
    walks the book for us.

    Net-of-fee EV (2026-05-25 fee model integration):
      At p=$0.92: taker fee = 5 sh × 0.05 × 0.92 × 0.08 = $0.0184/fire
      At p=$0.95: taker fee = $0.0119/fire
      At p=$0.99: taker fee = $0.0025/fire
      Small relative to $5 size and $0.10+ expected per fill on wins;
      doesn't change METAR economics meaningfully. No threshold change
      needed."""

    daily_deployment_limit_usd: float = 50.0
    """HARD CIRCUIT BREAKER on the day's cumulative capital deployment.
    Sums `position_usd` for all positions filled today (UTC day) — when
    today's deployment + a new fire would exceed this, the bot skips the
    submission. Existing positions are NOT affected.

    Mathematical guarantee: worst-case daily loss ≤ this value. At $50
    on a $500 bankroll, that's 10% max drawdown (= -$50) in the absolute
    worst case where EVERY fire loses. In practice with 95% hit rate,
    expected daily PnL is ~+$15-35 on $50 deployment.

    SHRUNK 2026-05-19 from $150 → $50 for the re-validation phase after
    the Polymarket archive event (per strategy-levers memo lines 82+120
    and the user-requested "smaller default deployment" tightening). Real
    strategy losses pre-archive were ~$46/day on $150 deployment (~30%
    loss rate, likely distorted by oracle misresolutions on Miami/Seoul/
    HK so true rate is unknown). $50/day caps the unknown-strategy
    downside while we re-validate; restore to $150 only after 7+ days
    of stable post-archive trading prove the strategy is net-positive.

    This is the answer to 'what if hybrid produces -$200/day?' — it
    cannot. The CAP is the guarantee, not a target.
    """

    kelly_multiplier: float = 0.1
    """Deci-Kelly default — DO NOT raise without forward-log validation."""

    require_min_n_resolved: int = 30
    """If `forward_log_records_resolved` is below this, skip live trading.
    Verified upstream by the CLI; this is informational here."""

    extra_allowed_stations: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    """Additional (station_id, target) pairs explicitly opted in."""

    max_slippage_pp: float = 0.02
    """Max acceptable slippage from top-of-book ask, in price points (= dollars
    per share). 0.02 = 2¢. If the depth-walked average fill price exceeds the
    signal's original `fill_price` by more than this, the signal is rejected
    rather than executed at the worse price.

    Rationale: top-of-book often has only 5-10 shares; filling $20-50 walks
    through worse levels, especially on tail buckets. Without this gate,
    the bot executes apparent edges that disappear after slippage.
    Empirically (2026-05-10): tail buckets can show 100+pp slippage on
    $50 fills; mid-buckets typically ≤10pp."""

    min_edge_after_slippage: float = 0.03
    """Minimum edge that must remain AFTER applying depth-walked fill price.
    A 5pp edge that becomes 1pp after slippage isn't worth the spread. Set
    this lower than `min_edge` (the pre-slippage filter) to allow some
    erosion but reject what the slippage eats."""

    spread_anomaly_factor: float = 2.5
    """Reject trades where today's ensemble spread (σ_ensemble_c) exceeds
    `spread_anomaly_factor` times the trained σ_residual_c for that
    (station, target). This auto-detects regime shifts — tropical cyclones,
    atmospheric rivers, etc. — where the model has lost confidence and
    bias correction is no longer valid. Set to 0 or a very large value to
    disable. See `weather_bot.exclusions` for the complementary manual
    exclusion list."""

    max_position_pct_of_depth: float = 0.30
    """Maximum fraction of the current ask-side orderbook depth our trade
    is allowed to consume. Bounds market impact: if we'd take >30% of all
    visible asks, we're a price-mover and the trade's apparent EV decays
    as our own buying pushes the price up against us.

    Distinct from `max_slippage_pp`:
      - max_slippage_pp limits how bad the fill price can be on THIS trade
      - max_position_pct_of_depth limits how visible we are to other
        participants who might react (pull quotes, front-run, etc.)

    Applies in `apply_depth_check` (model-driven trades, where EV can
    actually degrade with market impact). NOT applied in
    `apply_depth_sweep_metar` — METAR-confirmed trades have our_prob ≈ 1.0
    so resolution is fixed by reality, not by orderbook movement; market
    impact doesn't hurt EV there. Slippage is the only relevant constraint
    for METAR and that's bounded by the price ceiling."""


@dataclass
class TradeValidation:
    ok: bool
    reason: str = ""


def is_kill_switched(config: TradingConfig) -> bool:
    return config.kill_switch_path.exists()


# ──────────────────────────────────────────────────────────────────────────
# Depth-of-book check (added 2026-05-10)
#
# Top-of-book ask via /prices is fast but doesn't reflect fillable depth.
# Before submitting any order, walk the actual orderbook to compute the
# realistic average fill price for the signal's intended position size.
# Reject signals where slippage would erode edge past the threshold.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class DepthAdjustedSignal:
    """A TradeSignal paired with depth-walked fill assumptions.

    The wrapped `signal` is left untouched (its `fill_price` remains the
    original top-of-book ask). Downstream code that wants the realistic
    post-slippage values reads them from this wrapper directly:
      - `realistic_fill_price` instead of `signal.fill_price`
      - `new_edge`              instead of `signal.edge`
    """

    signal: TradeSignal
    original_fill_price: float
    realistic_fill_price: float
    realistic_shares: float
    slippage_pp: float
    fully_filled_at_size: bool
    new_edge: float


async def apply_depth_check(
    signals: list[TradeSignal],
    config: TradingConfig,
    *,
    client: httpx.AsyncClient | None = None,
    concurrency: int = 6,
) -> tuple[list[DepthAdjustedSignal], list[tuple[TradeSignal, str]]]:
    """For each signal, fetch CLOB depth and recompute fill price for size.

    Returns (kept_signals_with_realistic_prices, rejected_with_reasons).
    The original TradeSignal objects are NOT mutated — kept signals are
    wrapped in `DepthAdjustedSignal` and downstream code reads
    `adj.realistic_fill_price` / `adj.new_edge` for the post-slippage values.

    Rejection criteria (in order):
      1. Depth fetch failed → "depth fetch failed"
      2. Insufficient depth → "insufficient depth for $X size"
      3. Slippage > max_slippage_pp → "slippage Xpp > max_slippage_pp Ypp"
      4. New edge < min_edge_after_slippage → "edge after slippage too small"
      5. Realized fill below marketable minimum ($1 notional, no share floor)
      6. Market impact > max_position_pct_of_depth (default 30% of book)
    """
    if not signals:
        return [], []

    # Batch-fetch depths concurrently. One token per signal.
    token_ids = [s.token_id for s in signals]
    depths = await fetch_orderbook_depths_batch(
        token_ids, client=client, concurrency=concurrency,
    )

    kept: list[DepthAdjustedSignal] = []
    rejected: list[tuple[TradeSignal, str]] = []

    for sig in signals:
        depth = depths.get(sig.token_id)
        if depth is None:
            rejected.append((sig, "depth fetch failed"))
            continue
        if not depth.asks:
            rejected.append((sig, "no asks on book"))
            continue

        sim = depth.simulate_buy(sig.position_usd)
        if sim is None:
            rejected.append((sig, f"insufficient depth for ${sig.position_usd:.2f}"))
            continue

        realistic_price, realistic_shares, full = sim
        slip_pp = realistic_price - sig.fill_price

        if slip_pp > config.max_slippage_pp:
            rejected.append((
                sig,
                f"slippage {slip_pp*100:.1f}pp > max {config.max_slippage_pp*100:.1f}pp "
                f"(top ${sig.fill_price:.3f} → walked ${realistic_price:.3f})"
            ))
            continue

        # Recompute edge with the new fill price.
        # For YES: edge = our_prob − fill_price.
        # For NO:  edge = (1 − our_prob)_for_NO − fill_price_for_NO; but the
        # signal already stores the NO-side fill_price (= 1 − yes_bid). The
        # realistic_price returned here is for the no_token_id (since we
        # buy NO via the NO token), so the formula stays consistent:
        # new_edge = (probability_chosen_side) − realistic_price.
        prob_chosen_side = sig.our_prob if sig.side == "YES" else 1.0 - sig.our_prob
        new_edge = prob_chosen_side - realistic_price

        if new_edge < config.min_edge_after_slippage:
            rejected.append((
                sig,
                f"edge after slippage {new_edge*100:.1f}pp < "
                f"min {config.min_edge_after_slippage*100:.1f}pp"
            ))
            continue

        # Final exchange-floor gate: realized fill must clear the $1
        # marketable minimum. Marketable has no share floor.
        realized_usd = realistic_shares * realistic_price
        if not marketable_passes_min(realized_usd):
            rejected.append((
                sig,
                f"realized fill {realistic_shares:.1f} sh × ${realistic_price:.3f} "
                f"= ${realized_usd:.2f} below marketable min "
                f"${POLYMARKET_MARKETABLE_MIN_USD:.2f}"
            ))
            continue

        # Market-impact cap: limit our share of total ask-side depth so we're
        # not a price-mover. Being too big a fraction of the book signals
        # our presence and invites adverse reaction (other makers pull
        # quotes; later snapshots show worse spreads). For model-driven
        # trades this can flip the strategy EV-negative even when each
        # individual fill looks fine.
        if config.max_position_pct_of_depth > 0:
            total_ask_shares = sum(
                lvl.size_shares for lvl in depth.asks if lvl.size_shares > 0
            )
            if total_ask_shares > 0:
                impact_pct = realistic_shares / total_ask_shares
                if impact_pct > config.max_position_pct_of_depth:
                    rejected.append((
                        sig,
                        f"market impact {impact_pct*100:.0f}% of ask depth "
                        f"({realistic_shares:.1f}/{total_ask_shares:.0f} sh) "
                        f"> max {config.max_position_pct_of_depth*100:.0f}% "
                        f"— would move the book against us"
                    ))
                    continue

        adj = DepthAdjustedSignal(
            signal=sig,
            original_fill_price=sig.fill_price,
            realistic_fill_price=realistic_price,
            realistic_shares=realistic_shares,
            slippage_pp=slip_pp,
            fully_filled_at_size=full,
            new_edge=new_edge,
        )
        # position_usd stays the same (we're committing the same dollar
        # amount, just at a worse average fill); shares change implicitly.
        kept.append(adj)

    return kept, rejected


# ──────────────────────────────────────────────────────────────────────────
# SELL-side depth check (added 2026-05-13)
#
# Used by the cross-up cancellation path: when METAR early-tail (or T-6h
# cleanup sweep) confirms a winner, any open NO position on that bucket
# needs to be SOLD before resolution at $0. The sell is marketable (we
# cross the spread immediately to exit), so it walks the BID side and
# must clear the marketable-$1 floor.
#
# Distinct from `apply_depth_check` (BUY/asks) and `apply_depth_sweep_metar`
# (BUY/asks/ceiling). This walks bids and accepts whatever proceeds the
# book gives — there's no slippage gate because the alternative (hold to
# resolution at $0) is worse than any positive proceeds.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SellDepthAdjusted:
    """Result of walking the bid side for a marketable SELL.

    Mirrors `DepthAdjustedSignal` but for the exit-from-NO path. The
    bot keeps its original NO position size; the avg proceeds is what
    a marketable sell of `shares_to_sell` would realize against current
    bids.
    """

    token_id: str
    shares_to_sell: float
    avg_proceeds_price: float
    proceeds_usd: float
    shares_sellable: float       # = shares_to_sell if book is deep enough
    fully_sellable: bool


@dataclass
class SellOrderRequest:
    """A pending marketable SELL the cancel-on-confirmation path wants
    to submit. Decoupled from TradeSignal because cancels operate on
    existing positions, not on fresh model signals.
    """
    token_id: str
    shares_to_sell: float
    bucket_label: str = ""       # for log lines only


async def apply_depth_check_sell(
    requests: list[SellOrderRequest],
    *,
    client: httpx.AsyncClient | None = None,
    concurrency: int = 6,
) -> tuple[list[SellDepthAdjusted], list[tuple[SellOrderRequest, str]]]:
    """For each SELL request, fetch CLOB depth and compute realistic proceeds.

    Returns (kept_with_proceeds, rejected_with_reasons).

    Used by the cross-up cancellation path. The alternative to selling
    is holding the NO position to a $0 resolution, so any positive
    proceeds is strictly better. The only gates are:

      1. Depth fetch must succeed.
      2. There must be at least one bid on the book.
      3. The realized fill must clear the marketable-$1 floor; if it
         doesn't, the bot can't legally submit the order and must hold.

    No slippage gate (every cent recovered beats $0), no edge gate (we
    already know the bucket has crossed), and no market-impact gate
    (we're exiting a known-loser; pushing the book down is irrelevant
    to a position that resolves at $0).
    """
    if not requests:
        return [], []

    token_ids = [r.token_id for r in requests]
    depths = await fetch_orderbook_depths_batch(
        token_ids, client=client, concurrency=concurrency,
    )

    kept: list[SellDepthAdjusted] = []
    rejected: list[tuple[SellOrderRequest, str]] = []

    for req in requests:
        depth = depths.get(req.token_id)
        if depth is None:
            rejected.append((req, "depth fetch failed"))
            continue
        if not depth.bids:
            rejected.append((req, "no bids on book"))
            continue

        # Walk bids: take up to shares_to_sell of the best-bid-first ladder.
        # We need a shares-based walk (not USD-target), so reproduce the
        # math locally rather than wrapping simulate_sell (which takes a
        # USD target).
        remaining_shares = float(req.shares_to_sell)
        total_proceeds = 0.0
        shares_sold = 0.0
        for lvl in depth.bids:
            if lvl.price <= 0 or lvl.size_shares <= 0:
                continue
            take = min(remaining_shares, lvl.size_shares)
            shares_sold += take
            total_proceeds += take * lvl.price
            remaining_shares -= take
            if remaining_shares <= 1e-6:
                break

        if shares_sold <= 0:
            rejected.append((req, "bid side empty after filtering"))
            continue

        avg_price = total_proceeds / shares_sold
        proceeds_usd = total_proceeds

        if not marketable_passes_min(proceeds_usd):
            rejected.append((
                req,
                f"realized sell {shares_sold:.1f} sh × ${avg_price:.3f} "
                f"= ${proceeds_usd:.2f} below marketable min "
                f"${POLYMARKET_MARKETABLE_MIN_USD:.2f} — must hold to $0"
            ))
            continue

        kept.append(SellDepthAdjusted(
            token_id=req.token_id,
            shares_to_sell=req.shares_to_sell,
            avg_proceeds_price=avg_price,
            proceeds_usd=proceeds_usd,
            shares_sellable=shares_sold,
            fully_sellable=(remaining_shares <= 1e-6),
        ))

    return kept, rejected


async def apply_depth_sweep_metar(
    signals: list[TradeSignal],
    *,
    price_ceiling: float = 0.95,
    max_usd_per_trade: float = 20.0,
    client: httpx.AsyncClient | None = None,
    concurrency: int = 6,
) -> tuple[list[DepthAdjustedSignal], list[tuple[TradeSignal, str]]]:
    """Depth check for METAR-confirmed trades — sweep semantics.

    METAR-confirmed trades have `our_prob ≈ 1.0` (the bucket WILL win at $1.00).
    Slippage is therefore irrelevant: any fill price below the ceiling is EV+.
    These orders cross the spread immediately so they're MARKETABLE — only
    the $1 notional floor applies. No share floor for marketable.

    The constraints:
      - Walked fill must clear marketable minimum (≥ $1 notional).
      - Walked fill stays at or below `price_ceiling` (default $0.95) to
        leave a safety margin against resolution slippage / weird tail cases.
      - Walked fill spends ≤ `max_usd_per_trade` to bound per-trade exposure.

    **Market-impact cap is intentionally NOT applied here.** METAR-confirmed
    trades have a fixed resolution at $1.00 regardless of how much we move
    the book during the fill. Taking 100% of available depth at $0.05 is
    fine: we get however many shares the book provides at the cheap price,
    and they all pay $1.00 at resolution. The cap in `apply_depth_check`
    exists because model-driven EV decays with market impact (other makers
    react, spread widens against us). That mechanism doesn't apply to
    oracle-confirmed trades.

    For each signal: walk the book up to the ceiling, take whatever fills.
    Partial fills are accepted. The returned `DepthAdjustedSignal.realistic_*`
    fields reflect the swept fill; downstream submit_order uses these for
    a marketable-limit-with-ceiling order on the live path.

    Rejection criteria (in order):
      1. Depth fetch failed → "depth fetch failed"
      2. No asks at or below ceiling → "no fillable depth at ≤ $X"
      3. Swept fill < min order → "swept N sh × $X = $Y below min order"

    DO NOT use this for model-driven trades — they need the strict
    `apply_depth_check` with slippage and edge gates. METAR is the
    intended caller (and any future "we KNOW the outcome" strategy).
    """
    if not signals:
        return [], []

    token_ids = [s.token_id for s in signals]
    depths = await fetch_orderbook_depths_batch(
        token_ids, client=client, concurrency=concurrency,
    )

    kept: list[DepthAdjustedSignal] = []
    rejected: list[tuple[TradeSignal, str]] = []

    for sig in signals:
        depth = depths.get(sig.token_id)
        if depth is None:
            rejected.append((sig, "depth fetch failed"))
            continue
        if not depth.asks:
            rejected.append((sig, "no asks on book"))
            continue

        sweep = depth.sweep_buy_to_ceiling(price_ceiling, max_usd=max_usd_per_trade)
        if sweep is None:
            rejected.append((sig, f"no fillable depth at ≤ ${price_ceiling:.2f}"))
            continue

        avg_price, shares, cost = sweep

        if not marketable_passes_min(cost):
            rejected.append((
                sig,
                f"swept {shares:.1f} sh × ${avg_price:.3f} = ${cost:.2f} "
                f"below marketable min ${POLYMARKET_MARKETABLE_MIN_USD:.2f}"
            ))
            continue

        # METAR has our_prob ≈ 1.0, so new_edge = 1.0 - avg_price.
        # No slippage gate; the strategy is already EV+ by construction.
        prob_chosen_side = sig.our_prob if sig.side == "YES" else 1.0 - sig.our_prob
        new_edge = prob_chosen_side - avg_price
        slip_pp = avg_price - sig.fill_price  # informational, not gated

        adj = DepthAdjustedSignal(
            signal=sig,
            original_fill_price=sig.fill_price,
            realistic_fill_price=avg_price,
            realistic_shares=shares,
            slippage_pp=slip_pp,
            fully_filled_at_size=False,  # sweep is always "best effort"
            new_edge=new_edge,
        )
        kept.append(adj)

    return kept, rejected


def validate_signal(
    signal: TradeSignal,
    config: TradingConfig,
    current_exposure_usd: float = 0.0,
    *,
    bias_table: "BiasTable | None" = None,
    active_exclusions: set[tuple[str, str]] | None = None,
    portfolio: "Portfolio | None" = None,
) -> TradeValidation:
    """Run all hard-limit checks on a single signal.

    Optional gates:
      - `active_exclusions`: set of (station_id, target) pairs that the user
        has manually excluded via `data/excluded_stations.json` because of
        an active extreme-weather event. See `weather_bot.exclusions`.
      - `bias_table`: when provided, σ_ensemble vs σ_residual is checked
        against `config.spread_anomaly_factor` to auto-detect regime
        shifts. Signals from stations where the model has lost confidence
        are rejected.
      - `portfolio`: when provided (2026-05-14), enforces portfolio-level
        dedupe (don't re-submit the same (token_id, side) across scans)
        AND the portfolio_cap_usd / per_region_cap_usd / per_event_cap_usd
        caps. The portfolio also feeds correlation-aware Kelly sizing
        when `config.enable_portfolio_kelly` is True — but that's applied
        UPSTREAM in scanner.py, not here (this is just a gate).
    """
    if not config.enabled:
        return TradeValidation(False, "trading is not enabled in config")
    if is_kill_switched(config):
        return TradeValidation(False, f"kill switch present at {config.kill_switch_path}")

    pair = (signal.station.station_id, signal.target)
    if active_exclusions is not None and pair in active_exclusions:
        return TradeValidation(
            False,
            f"{pair} on excluded_stations.json (active event-driven exclusion)",
        )

    if config.only_tier_1 and pair not in (TIER_1_STATIONS | config.extra_allowed_stations):
        return TradeValidation(False, f"{pair} not in tier-1 allow-list")

    if bias_table is not None and config.spread_anomaly_factor > 0:
        entry = bias_table.get_entry(signal.station.station_id, signal.target)
        if entry is not None and entry.sigma_residual_c > 0:
            threshold = config.spread_anomaly_factor * entry.sigma_residual_c
            if signal.sigma_ensemble_c > threshold:
                return TradeValidation(
                    False,
                    f"σ_ensemble {signal.sigma_ensemble_c:.2f}°C > "
                    f"{config.spread_anomaly_factor:.1f}× σ_residual "
                    f"{entry.sigma_residual_c:.2f}°C — anomalous regime "
                    f"(likely extreme weather event)",
                )

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

    # Polymarket order-type floors (verified empirically 2026-05-14):
    #   Marketable (taker): ≥ $1 notional, NO share floor
    #   Resting maker:      ≥ 5 shares, NO dollar floor
    #
    # Current `validate_signal` is called by `place_orders.py` for
    # MODEL-DRIVEN TAKER trades (and METAR which is also taker). So we
    # enforce the marketable $1 floor. NO_momentum's maker leg uses a
    # different execution path (TBD when live-wired) which must
    # separately enforce resting_passes_min(shares) before submission.
    if not marketable_passes_min(signal.position_usd):
        return TradeValidation(
            False,
            f"position ${signal.position_usd:.2f} below Polymarket "
            f"marketable minimum ${POLYMARKET_MARKETABLE_MIN_USD:.2f} notional",
        )
    # Belt-and-suspenders for the rare case where this path is repurposed
    # for maker orders: also check the 5-share resting floor when the
    # share count is low. Doesn't hurt taker orders (most pass with
    # margin); catches the maker edge case if ever routed here.
    shares = (
        signal.position_usd / signal.fill_price
        if signal.fill_price > 0 else 0.0
    )
    if not resting_passes_min(shares):
        # Soft warning, not a rejection — the order may still go through
        # as taker (which has no share floor). Just log for visibility.
        # If this becomes a frequent issue, surface as a metric and
        # add a hard floor.
        pass  # intentional no-op until live data shows whether this matters
    if current_exposure_usd + signal.position_usd > config.max_total_exposure_usd:
        return TradeValidation(
            False,
            f"would exceed per-scan cap ${config.max_total_exposure_usd} "
            f"(current ${current_exposure_usd:.2f} + new ${signal.position_usd:.2f})",
        )

    # ── Portfolio-level gates (added 2026-05-14) ────────────────────────
    if portfolio is not None:
        # 1. Dedupe + retry policy:
        #    - Already open/filled → skip
        #    - Permanently blocked (≥3 cancellations) → skip
        #    - In cooldown window after a cancel → skip
        skip, reason = portfolio.should_skip(signal.token_id, signal.side)
        if skip:
            return TradeValidation(False, reason)
        # 2. Portfolio + per-region + per-event hard caps
        exceeds, reason = portfolio.would_exceed_cap(
            station_id=signal.station.station_id,
            market_id=signal.market_id,
            position_usd=signal.position_usd,
            portfolio_cap_usd=config.portfolio_cap_usd,
            per_region_cap_usd=config.per_region_cap_usd,
            per_event_cap_usd=config.per_event_cap_usd,
        )
        if exceeds:
            return TradeValidation(False, reason)
    return TradeValidation(True)
