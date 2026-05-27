"""Comprehensive bot audit: where did the −$41 go, and why?

Decomposes lifetime PnL into:
  1. Per-strategy contribution
  2. Per-strategy win rate ACTUAL vs theoretical breakeven
  3. Per-strategy fee bleed (using confirmed formula)
  4. Per-strategy expected EV at entry vs realized — exposes execution gaps
  5. Open positions (unrealized exposure)

Outputs the SPECIFIC questions that should drive the next fix.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PORTFOLIO_PATH = "data/portfolio.json"
CSV_PATH = "/mnt/c/Users/Kevin/Downloads/Polymarket-History-2026-05-26.csv"

FEE_RATE = 0.05  # Polymarket Weather confirmed


def taker_fee(shares, price):
    if not (0 < price < 1) or shares <= 0:
        return 0.0
    return shares * FEE_RATE * price * (1 - price)


def main():
    p = json.load(open(PORTFOLIO_PATH))
    positions = p["positions"]

    # ── 1. PER-STRATEGY BREAKDOWN ──────────────────────────────────────
    by_strat = defaultdict(lambda: {
        "n_total": 0, "n_resolved": 0, "n_filled_open": 0, "n_cancelled": 0,
        "n_wins": 0, "n_losses": 0,
        "realized_pnl": 0.0,
        "deployed_usd": 0.0,
        "winning_shares": 0.0, "winning_entry_sum": 0.0,
        "losing_shares": 0.0, "losing_entry_sum": 0.0,
    })

    for pos in positions:
        strat = pos.get("strategy", "unknown") or "unknown"
        d = by_strat[strat]
        d["n_total"] += 1
        status = pos.get("status", "")
        shares = pos.get("shares", 0) or 0
        entry = pos.get("entry_price", 0) or 0
        pnl = pos.get("realized_pnl", 0) or 0

        if status == "resolved":
            d["n_resolved"] += 1
            d["realized_pnl"] += pnl
            d["deployed_usd"] += shares * entry
            if pnl > 0:
                d["n_wins"] += 1
                d["winning_shares"] += shares
                d["winning_entry_sum"] += shares * entry
            else:
                d["n_losses"] += 1
                d["losing_shares"] += shares
                d["losing_entry_sum"] += shares * entry
        elif status == "filled":
            d["n_filled_open"] += 1
        elif status == "cancelled":
            d["n_cancelled"] += 1

    print("=" * 90)
    print("PER-STRATEGY PnL BREAKDOWN (resolved positions)")
    print("=" * 90)
    print(f"{'strategy':<24} {'n':>4} {'wins':>4} {'win%':>6} {'avg_entry':>9} "
          f"{'realized':>11} {'$/fill':>8}")
    total_realized = 0.0
    total_n = 0
    for strat, d in sorted(by_strat.items(), key=lambda x: x[1]["realized_pnl"]):
        n = d["n_resolved"]
        if n == 0:
            continue
        wr = 100 * d["n_wins"] / n
        avg_entry = (d["winning_entry_sum"] + d["losing_entry_sum"]) / (d["winning_shares"] + d["losing_shares"]) if (d["winning_shares"] + d["losing_shares"]) > 0 else 0
        per_fill = d["realized_pnl"] / n
        total_realized += d["realized_pnl"]
        total_n += n
        print(f"{strat:<24} {n:>4} {d['n_wins']:>4} {wr:>5.1f}% "
              f"${avg_entry:>8.3f} ${d['realized_pnl']:>+10.2f} ${per_fill:>+7.3f}")
    print(f"{'TOTAL':<24} {total_n:>4} {'':>4} {'':>6} {'':>9} "
          f"${total_realized:>+10.2f}")

    # ── 2. BREAKEVEN ANALYSIS PER STRATEGY ────────────────────────────
    print()
    print("=" * 90)
    print("BREAKEVEN vs ACTUAL — is the strategy structurally profitable?")
    print("=" * 90)
    print(f"{'strategy':<24} {'avg_entry':>9} {'breakeven win%':>14} "
          f"{'actual win%':>12} {'gap':>8} {'verdict':>10}")
    for strat, d in sorted(by_strat.items()):
        n = d["n_resolved"]
        if n < 5:
            continue
        wr = 100 * d["n_wins"] / n
        avg_entry = (d["winning_entry_sum"] + d["losing_entry_sum"]) / (d["winning_shares"] + d["losing_shares"]) if (d["winning_shares"] + d["losing_shares"]) > 0 else 0
        # Breakeven: at avg entry price p, breakeven win rate = p (since wins pay $1-p, losses cost $p)
        # → w × (1-p) = (1-w) × p → w = p
        breakeven = avg_entry * 100
        gap = wr - breakeven
        verdict = "PROFITABLE" if gap > 0 else "BLEEDING" if gap < -2 else "MARGINAL"
        print(f"{strat:<24} ${avg_entry:>8.3f} {breakeven:>13.1f}% "
              f"{wr:>11.1f}% {gap:>+7.1f}pp {verdict:>10}")

    # ── 3. FEE BLEED ESTIMATE ─────────────────────────────────────────
    print()
    print("=" * 90)
    print("FEE BLEED (taker fees paid — formula: shares × 0.05 × p × (1-p))")
    print("=" * 90)
    fee_by_strat = defaultdict(float)
    for pos in positions:
        if pos.get("status") not in ("resolved", "filled"):
            continue
        strat = pos.get("strategy", "unknown") or "unknown"
        shares = pos.get("shares", 0) or 0
        entry = pos.get("entry_price", 0) or 0
        # Maker strategies pay $0; takers pay fee. Approximation:
        # - NO_momentum (GTD post_only) → maker
        # - Layer 7 / guaranteed_no_buy → taker (FAK)
        # - live_bucket_arb → taker (FAK)
        # - v2_conditional_preposit → maker (GTD post_only)
        is_maker = strat in ("NO_momentum", "v2_conditional_preposit", "cross_up_cancel")
        if is_maker:
            continue
        fee = taker_fee(shares, entry)
        fee_by_strat[strat] += fee

    total_fees = 0
    for strat, fee in sorted(fee_by_strat.items(), key=lambda x: -x[1]):
        total_fees += fee
        print(f"  {strat:<24} ${fee:>+8.4f}")
    print(f"  {'TOTAL FEES':<24} ${total_fees:>+8.4f}")

    # ── 4. EXECUTION-GAP ANALYSIS ─────────────────────────────────────
    # For each strategy: compute theoretical EV at fair-price assumption
    # (= breakeven), compare to actual. Gap reveals execution bleed beyond
    # what the strategy's fundamental EV explains.
    print()
    print("=" * 90)
    print("EXECUTION GAP — actual realized PnL vs naive theoretical at win rate")
    print("=" * 90)
    print(f"{'strategy':<24} {'theory_pnl':>12} {'actual_pnl':>12} {'gap':>10} {'gap/$deployed':>14}")
    for strat, d in sorted(by_strat.items()):
        n = d["n_resolved"]
        if n < 5:
            continue
        # Theoretical PnL at observed win rate (no fees, no slippage):
        theory_pnl = d["winning_shares"] * 1.0 - d["winning_entry_sum"] - d["losing_entry_sum"]
        actual_pnl = d["realized_pnl"]
        gap = actual_pnl - theory_pnl  # negative = bot underperformed theory
        deployed = d["deployed_usd"]
        gap_pct = (gap / deployed * 100) if deployed > 0 else 0
        print(f"{strat:<24} ${theory_pnl:>+11.2f} ${actual_pnl:>+11.2f} "
              f"${gap:>+9.2f} {gap_pct:>+13.2f}%")

    # ── 5. OPEN POSITIONS ─────────────────────────────────────────────
    print()
    print("=" * 90)
    print("OPEN POSITIONS (unrealized exposure)")
    print("=" * 90)
    open_positions = [pos for pos in positions if pos.get("status") in ("filled", "submitted")]
    open_by_strat = defaultdict(lambda: {"n": 0, "deployed": 0.0, "potential_payout": 0.0})
    for pos in open_positions:
        strat = pos.get("strategy", "unknown") or "unknown"
        shares = pos.get("shares", 0) or 0
        entry = pos.get("entry_price", 0) or 0
        d = open_by_strat[strat]
        d["n"] += 1
        d["deployed"] += shares * entry
        d["potential_payout"] += shares  # max payout = shares × $1
    for strat, d in sorted(open_by_strat.items(), key=lambda x: -x[1]["deployed"]):
        print(f"  {strat:<24} n={d['n']:>3}  deployed=${d['deployed']:>7.2f}  "
              f"max_payout=${d['potential_payout']:>7.2f}  "
              f"max_gain=${d['potential_payout'] - d['deployed']:>+7.2f}")

    # ── 6. STRUCTURAL DIAGNOSIS ──────────────────────────────────────
    print()
    print("=" * 90)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * 90)
    print()
    print("For each strategy, the question is: BLEEDING (gap < -2pp from breakeven)")
    print("or PROFITABLE (gap >= 0)? Strategies below are BLEEDING:")
    print()
    issues = []
    for strat, d in by_strat.items():
        n = d["n_resolved"]
        if n < 5:
            continue
        wr = 100 * d["n_wins"] / n
        avg_entry = (d["winning_entry_sum"] + d["losing_entry_sum"]) / (d["winning_shares"] + d["losing_shares"]) if (d["winning_shares"] + d["losing_shares"]) > 0 else 0
        breakeven = avg_entry * 100
        gap = wr - breakeven
        if gap < -2:
            issues.append((strat, n, wr, breakeven, gap, d["realized_pnl"]))

    for strat, n, wr, be, gap, pnl in sorted(issues, key=lambda x: x[5]):
        print(f"  • {strat}: needs {be:.0f}% to break even, hitting {wr:.0f}% "
              f"({gap:+.0f}pp short) on n={n}, lost ${pnl:+.2f}")
    if not issues:
        print("  (no strategies bleeding > 2pp)")
    print()
    print(f"Total realized: ${total_realized:+.2f}")
    print(f"Total fee bleed (estimate): ${-total_fees:.2f}")
    print(f"Resolved trade bleed: ${total_realized + total_fees:+.2f}")
    print()


if __name__ == "__main__":
    main()
