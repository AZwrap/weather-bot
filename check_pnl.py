"""Quick diagnostic: how much PnL has materialised after resolutions?"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from weather_bot.forward_log import load_records
from weather_bot.pnl import simulate_record, aggregate
from weather_bot.positions import replay, summarize

recs = load_records()
n_resolved = sum(1 for r in recs if r.is_resolved)
print(f"records: {len(recs)} total, {n_resolved} resolved")

trades = []
for r in recs:
    trades.extend(simulate_record(
        r, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
        min_edge=0.01, max_edge=0.50, min_yes_price=0.0, max_yes_price=1.0,
    ))
s = aggregate(trades)
print()
print("PnL simulator (taker, hold to expiration):")
print(f"  trades: {s.n_trades}, resolved: {s.n_resolved}, pending: {s.n_pending}")
print(f"  exposure: ${s.total_pos_usd:,.2f}, resolved exposure: ${s.resolved_pos_usd:,.2f}")
print(f"  gross wins:   ${s.gross_wins_usd:+,.2f}")
print(f"  gross losses: ${s.gross_losses_usd:+,.2f}")
print(f"  realised P&L: ${s.total_profit_usd:+,.2f}")
print(f"  pending EV:   ${s.pending_ev_usd:+,.2f}  ({s.n_pending} pending trades)")
print(f"  projected:    ${s.projected_total_usd:+,.2f}  (realised + pending EV)")
if s.win_rate is not None:
    print(f"  win rate: {s.win_rate:.1%} ({s.n_wins}W / {s.n_losses}L)")
if s.roi_pct is not None:
    print(f"  ROI on resolved exposure: {s.roi_pct:+.2f}%")

print()
print("Position simulator (multi-snapshot replay):")
for sl_pct in (0.50, 0.90):
    pt = replay(
        recs, bankroll_usd=1000, kelly_multiplier=0.1, max_position_usd=50,
        min_edge=0.01, max_edge=0.50, min_yes_price=0.0, max_yes_price=1.0,
        stop_loss_pct=sl_pct, sigma_inflation_factor=1.4,
    )
    ss = summarize(pt)
    pnl = ss.total_realized_pnl_usd
    print(
        f"  stop_loss_pct={sl_pct}: {ss.n_positions} pos, "
        f"SL={ss.n_stop_loss}, TP={ss.n_take_profit}, "
        f"won={ss.n_expire_won}, lost={ss.n_expire_lost}, "
        f"PnL=${pnl:+,.2f}"
    )
