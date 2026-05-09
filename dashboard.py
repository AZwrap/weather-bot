"""Streamlit dashboard.

Run locally:
    streamlit run dashboard.py

Run on the VPS (headless, listening on 8501):
    streamlit run dashboard.py \
      --server.port 8501 --server.address 0.0.0.0 --server.headless true

Then open http://VPS_IP:8501 in your browser.

Tabs:
  Overview  — bot health, log volume, last cron, system info
  Skill     — per-station calibration: MAE / bias / RMSE / CRPS / reliability
  PnL       — hypothetical P&L from deci-Kelly trades on resolved records
  Signals   — live scanner output (cached 5 min) with edge/Kelly/size
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from weather_bot.bias import BiasTable
from weather_bot.forward_log import (
    DEFAULT_LOG_PATH,
    ForwardLogRecord,
    load_records,
)
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.positions import Position, replay_maker, summarize
from weather_bot.scanner import TradeSignal, scan

st.set_page_config(
    page_title="Weather Bot Dashboard",
    page_icon="🌤️",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar-settings persistence
# ──────────────────────────────────────────────────────────────────────────

SETTINGS_PATH = Path("data/dashboard_settings.json")


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_settings(settings: dict) -> None:
    """Write current sidebar values to disk so they persist across reloads."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except OSError:
        pass  # non-fatal — sidebar still works without persistence

# ──────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def cached_records() -> list[ForwardLogRecord]:
    return load_records(DEFAULT_LOG_PATH)


@st.cache_data(ttl=300)
def cached_signals(
    _bias_path: str, _bankroll: float, _kelly: float, _max_pos: float,
    _max_edge: float, _min_yes: float, _max_yes: float,
    _sigma_factor: float, _per_event_cap: float,
) -> list[TradeSignal]:
    """Live scan, cached 5 minutes. Underscore-prefixed args = hashable cache keys."""
    bias_table = BiasTable.load(Path(_bias_path))
    return asyncio.run(
        scan(
            bias_table,
            min_edge=0.05,
            max_edge=_max_edge,
            min_yes_price=_min_yes,
            max_yes_price=_max_yes,
            min_volume_24hr=100.0,
            bankroll_usd=_bankroll,
            kelly_multiplier=_kelly,
            max_position_usd=_max_pos,
            sigma_inflation_factor=_sigma_factor,
            per_event_cap_usd=_per_event_cap,
        )
    )


@st.cache_data(ttl=600)
def cached_bias_table_meta(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    return {
        "exists": True,
        "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc),
        "n_entries": len(BiasTable.load(p)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Tab renderers
# ──────────────────────────────────────────────────────────────────────────


def render_overview(records: list[ForwardLogRecord], bias_meta: dict) -> None:
    st.subheader("System status")

    cols = st.columns(4)
    n = len(records)
    n_resolved = sum(1 for r in records if r.is_resolved)
    cols[0].metric("Forward-log records", f"{n:,}")
    cols[1].metric("Resolved", f"{n_resolved:,}", f"{n - n_resolved} pending")
    if records:
        last = max(r.issue_time_utc for r in records)
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        cols[2].metric("Last log entry", last.strftime("%Y-%m-%d %H:%M UTC"),
                       f"{age_h:.1f}h ago")
    else:
        cols[2].metric("Last log entry", "—")

    if bias_meta.get("exists"):
        age_d = (datetime.now(timezone.utc) - bias_meta["mtime"]).days
        cols[3].metric("Bias table", f"{bias_meta['n_entries']} entries",
                       f"{age_d}d old")
    else:
        cols[3].metric("Bias table", "missing", "run train_bias.py")

    if not records:
        st.info("No forward-log records yet. Run `python log_forecasts.py` to seed.")
        return

    st.divider()
    st.subheader("Records by station and target")
    df = pd.DataFrame([{
        "station_id": r.station_id,
        "target": r.target,
        "target_date": r.target_date,
        "issue_time_utc": r.issue_time_utc,
        "resolved": r.is_resolved,
    } for r in records])
    pivot = (
        df.groupby(["station_id", "target"])
        .agg(records=("target_date", "count"),
             resolved=("resolved", "sum"),
             latest_target=("target_date", "max"))
        .reset_index()
        .sort_values(["records", "station_id"], ascending=[False, True])
    )
    pivot["pending"] = pivot["records"] - pivot["resolved"]
    st.dataframe(pivot, use_container_width=True, hide_index=True)


def render_skill(records: list[ForwardLogRecord]) -> None:
    resolved = [r for r in records if r.is_resolved]
    st.subheader("Forecast skill on resolved records (true 1-day-lead)")
    if not resolved:
        st.info(
            "No resolved records yet. ERA5 archive lags ~5 days, so the first "
            "resolutions appear ~6 days after `log_forecasts.py` first ran. "
            "Run `python resolve_log.py` to fill them in."
        )
        return

    rows = []
    for r in resolved:
        rows.append({
            "station_id": r.station_id,
            "target": r.target,
            "target_date": r.target_date,
            "predicted_mean": r.predictive_mean_c,
            "actual": r.actual_obs_c,
            "error": r.predictive_mean_c - r.actual_obs_c,
            "sigma_total": r.sigma_total_c,
        })
    df = pd.DataFrame(rows)

    # Per-(station, target) skill table
    by = df.groupby(["station_id", "target"]).agg(
        n=("error", "count"),
        bias_c=("error", "mean"),
        mae_c=("error", lambda e: np.mean(np.abs(e))),
        rmse_c=("error", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        sigma_avg=("sigma_total", "mean"),
    ).reset_index().sort_values(["mae_c"])
    by["calibration"] = np.where(
        by["rmse_c"] > by["sigma_avg"] * 1.3, "↑ inflate more",
        np.where(by["rmse_c"] < by["sigma_avg"] * 0.7, "↓ over-inflated", "OK"))
    st.dataframe(by, use_container_width=True, hide_index=True)

    st.subheader("Forecast vs observed")
    if len(df) >= 5:
        fig = px.scatter(
            df, x="actual", y="predicted_mean", color="station_id",
            hover_data=["target", "target_date", "error"],
            labels={"actual": "Observed (°C)", "predicted_mean": "Predicted mean (°C)"},
            title="Predicted vs actual daily max/min",
        )
        rng_lo, rng_hi = df[["actual", "predicted_mean"]].min().min(), df[["actual", "predicted_mean"]].max().max()
        fig.add_shape(type="line", x0=rng_lo, y0=rng_lo, x1=rng_hi, y1=rng_hi,
                      line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"Need ≥5 resolved records for the scatter plot (have {len(df)}).")


def render_pnl(
    records: list[ForwardLogRecord], bankroll: float, kelly: float,
    max_pos: float, min_edge: float, max_edge: float,
    min_yes: float, max_yes: float, sigma_factor: float,
) -> None:
    eligible = [r for r in records if r.bucket_snapshots is not None]
    st.subheader("Paper-trade ledger (maker-ladder, deci-Kelly)")
    if not eligible:
        st.info("No records with bucket snapshots yet — log_forecasts hasn't run.")
        return

    st.caption(
        "📐 **Maker-ladder simulator** — canonical execution strategy. "
        "Each accepted signal places a 4-rung limit ladder inside the "
        "bid-ask spread; rungs only fill when the market drifts to our "
        "price. Unfilled rungs cost $0 (asymmetric payoff). "
        "Taker-mode simulation is excluded from the dashboard but still "
        "logged via `pnl.simulate_record` for offline analysis."
    )

    positions = replay_maker(
        eligible,
        bankroll_usd=bankroll, kelly_multiplier=kelly,
        max_position_usd=max_pos, min_edge=min_edge, max_edge=max_edge,
        min_yes_price=min_yes, max_yes_price=max_yes,
        sigma_inflation_factor=sigma_factor,
        taker_fallback=False,
    )

    if not positions:
        st.warning(
            "No maker-ladder fills yet. The forward log needs more snapshot "
            "density (typically several days of hourly cron) before the "
            "market drifts through enough rungs to register fills. "
            "Once the cron runs longer, this tab will populate."
        )
        return

    n_pos = len(positions)
    n_resolved = sum(1 for p in positions if p.closed)
    n_pending = n_pos - n_resolved
    n_wins = sum(1 for p in positions if p.closed and p.realized_profit_usd > 0)
    n_losses = sum(1 for p in positions if p.closed and p.realized_profit_usd < 0)
    total_exposure = sum(p.position_usd for p in positions)
    resolved_exposure = sum(p.position_usd for p in positions if p.closed)
    realized_pnl = sum(p.realized_profit_usd for p in positions if p.closed)
    win_rate = (n_wins / (n_wins + n_losses)) if (n_wins + n_losses) > 0 else None
    roi_pct = (realized_pnl / resolved_exposure * 100.0) if resolved_exposure > 0 else None

    cols = st.columns(6)
    cols[0].metric("Positions", f"{n_pos:,}",
                   f"{n_pending} pending  ·  {n_resolved} resolved")
    cols[1].metric("Total exposure", f"${total_exposure:,.0f}")
    cols[2].metric("Resolved exposure", f"${resolved_exposure:,.0f}")
    cols[3].metric("Realised P&L", f"${realized_pnl:+,.0f}")
    cols[4].metric("ROI (resolved)",
                   f"{roi_pct:+.1f}%" if roi_pct is not None else "—")
    cols[5].metric("Win rate",
                   f"{win_rate:.0%}" if win_rate is not None else "—",
                   f"{n_wins}W / {n_losses}L" if n_resolved else "no resolved yet")

    # Build the position ledger DataFrame
    def _status(p: Position) -> str:
        if not p.closed:
            return "open"
        last = p.events[-1].action if p.events else ""
        if last == "sell_take_profit":
            return "take_profit"
        if last == "sell_stop_loss":
            return "stop_loss"
        if last == "expire":
            return "won" if p.realized_profit_usd > 0 else "lost"
        return "closed"

    rows = [{
        "issue_time_utc": p.open_event.issue_time_utc,
        "target_date": p.target_date,
        "station": p.station_id,
        "target": p.target,
        "bucket": p.bucket_label,
        "side": p.side,
        "entry_price": p.entry_price,
        "size_usd": p.position_usd,
        "n_events": len(p.events),
        "status": _status(p),
        "profit_usd": p.realized_profit_usd if p.closed else 0.0,
    } for p in positions]
    df = pd.DataFrame(rows)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"])
    df = df.sort_values("target_date")

    # Cumulative chart: exposure (gray, includes pending) + P&L (resolved only)
    closed_statuses = {"won", "lost", "take_profit", "stop_loss"}
    chart_df = df.copy()
    chart_df["resolved_profit"] = chart_df["profit_usd"].where(
        chart_df["status"].isin(closed_statuses), 0.0
    )
    daily = chart_df.groupby("target_date").agg(
        daily_exposure=("size_usd", "sum"),
        daily_profit=("resolved_profit", "sum"),
    ).reset_index().sort_values("target_date")
    daily["cum_exposure"] = daily["daily_exposure"].cumsum()
    daily["cum_profit"] = daily["daily_profit"].cumsum()

    st.subheader("Cumulative paper exposure & realised P&L")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["target_date"], y=daily["cum_exposure"],
        mode="lines", name="Cumulative exposure",
        line=dict(color="lightgray", width=2),
        fill="tozeroy",
    ))
    fig.add_trace(go.Scatter(
        x=daily["target_date"], y=daily["cum_profit"],
        mode="lines+markers", name="Cumulative realised P&L",
        line=dict(color="#1f8b4c", width=3),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Target date",
        yaxis_title="USD",
        title=f"Bankroll ${bankroll:,.0f}  ·  {kelly:g}× Kelly  ·  cap ${max_pos:.0f}/trade",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Gray area = total $ committed across all maker fills (grows as new "
        "ladder rungs fill). Green line = realised P&L on positions whose "
        "target day has passed and the METAR observation has been pulled. "
        "The gap is money 'in flight'."
    )

    # Status breakdown
    cnt = df["status"].value_counts()
    parts = []
    for status in ("open", "won", "lost", "take_profit", "stop_loss"):
        if status in cnt and cnt[status] > 0:
            parts.append(f"{int(cnt[status])} {status}")
    if parts:
        st.caption("Status: " + "  ·  ".join(parts))

    # Position ledger table
    with st.expander(f"Position ledger ({len(df)} rows)", expanded=False):
        show = df[["issue_time_utc", "target_date", "station", "target", "bucket",
                   "side", "entry_price", "size_usd", "n_events", "status",
                   "profit_usd"]].copy()
        st.dataframe(
            show.style.format({
                "entry_price": "{:.3f}",
                "size_usd": "${:,.2f}",
                "profit_usd": "${:+,.2f}",
            }),
            use_container_width=True, hide_index=True, height=400,
        )

    # Per-station P&L (only meaningful where there's a resolved position)
    st.subheader("By station")
    by_stn = df.groupby("station").agg(
        positions=("size_usd", "count"),
        exposure=("size_usd", "sum"),
        profit=("profit_usd", "sum"),
    ).reset_index().sort_values("profit", ascending=False)
    by_stn["resolved_count"] = (
        df[df["status"].isin(closed_statuses)].groupby("station").size()
    ).reindex(by_stn["station"]).fillna(0).astype(int).values
    fig2 = px.bar(by_stn, x="station", y="profit",
                  hover_data=["positions", "exposure", "resolved_count"],
                  labels={"profit": "Realised P&L ($)"})
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(by_stn, use_container_width=True, hide_index=True)


def render_positions(
    records: list[ForwardLogRecord], bankroll: float, kelly: float,
    max_pos: float, min_edge: float, max_edge: float,
    min_yes: float, max_yes: float,
    sigma_factor: float,
    saved: dict,
) -> None:
    st.subheader("Position simulator (multi-snapshot replay)")
    eligible = [r for r in records if r.bucket_snapshots is not None]
    if len(eligible) < 2:
        st.info(
            "Position simulator needs at least 2 snapshots of the same "
            "(station, target, target_date, bucket) — once hourly cron has "
            "been firing for a few hours this tab will populate."
        )
        return

    st.caption(
        f"σ inflation factor (from sidebar) = {sigma_factor:.1f}× — applied "
        "live: every replay recomputes our_prob from raw_members, so changing "
        "σ retroactively changes which positions open and close."
    )
    st.caption(
        "**Execution = maker-ladder (canonical).** 4-rung limit ladder inside "
        "the spread; rungs fill only when the market drifts to our price. "
        "Taker-mode replay has been removed from the dashboard — it remains "
        "available in `weather_bot.positions.replay` for offline use."
    )
    take_profit = st.slider(
        "Take-profit threshold (EV gap)", 0.0, 0.30,
        float(saved.get("take_profit", 0.05)), 0.01, key="take_profit",
        help="Sell when realised EV exceeds hold EV by this much.",
    )
    stop_loss = st.slider(
        "Stop-loss: absolute EV floor", -0.50, 0.0,
        float(saved.get("stop_loss", -0.10)), 0.01, key="stop_loss",
        help="Sell when hold-EV per share drops below this. "
             "Effective for mid-priced bets (entry ≈ 0.30–0.70). "
             "Structurally unreachable for tail bets (entry < |threshold|).",
    )
    stop_loss_pct = st.slider(
        "Stop-loss: relative MTM loss", 0.0, 1.0,
        float(saved.get("stop_loss_pct", 0.50)), 0.05, key="stop_loss_pct",
        help="Sell when (entry_price − current_bid) / entry_price exceeds "
             "this. Catches tail bets where the absolute stop can't fire. "
             "0.5 = exit when half the position is lost.",
    )
    n_rungs = st.slider(
        "Ladder rungs", 1, 8,
        int(saved.get("n_rungs", 4)), 1, key="n_rungs",
        help="Number of evenly-spaced limit orders inside "
             "the bid-ask spread. Polymarket tick size = 0.001.",
    )

    # Persist position-tab slider values
    saved.update({
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "n_rungs": n_rungs,
    })

    positions = replay_maker(
        eligible,
        bankroll_usd=bankroll, kelly_multiplier=kelly,
        max_position_usd=max_pos, min_edge=min_edge, max_edge=max_edge,
        min_yes_price=min_yes, max_yes_price=max_yes,
        n_rungs=n_rungs,
        sigma_inflation_factor=sigma_factor,
        take_profit_threshold=take_profit,
        stop_loss_threshold=stop_loss,
        stop_loss_pct=stop_loss_pct,
        taker_fallback=False,
    )
    summary = summarize(positions)

    cols = st.columns(7)
    cols[0].metric("Positions opened", f"{summary.n_positions}")
    cols[1].metric("Open now", f"{summary.n_open}",
                   f"${summary.open_exposure_usd:,.0f} exposure")
    cols[2].metric("Take-profit exits", f"{summary.n_take_profit}")
    cols[3].metric("Stop-loss exits", f"{summary.n_stop_loss}")
    cols[4].metric("Expired won", f"{summary.n_expire_won}")
    cols[5].metric("Expired lost", f"{summary.n_expire_lost}")
    cols[6].metric("Realized P&L", f"${summary.total_realized_pnl_usd:+,.0f}")

    if summary.n_positions == 0:
        st.warning("No positions opened — try lowering Min edge.")
        return

    rows = []
    for p in positions:
        rows.append({
            "station": p.station_id,
            "target": p.target,
            "target_date": p.target_date,
            "bucket": p.bucket_label,
            "side": p.side,
            "entry_price": p.entry_price,
            "shares": p.shares,
            "size_usd": p.position_usd,
            "n_events": len(p.events),
            "status": p.status,
            "realized_pnl": p.realized_profit_usd,
        })
    df = pd.DataFrame(rows)

    # Grouping by exit type
    counts = df["status"].value_counts().reindex(
        ["open", "sell_take_profit", "sell_stop_loss", "expire"]
    ).fillna(0).astype(int)
    fig = go.Figure(data=[go.Bar(
        x=counts.index, y=counts.values,
        marker_color=["lightgray", "#1f8b4c", "#d83b01", "#0078d4"],
    )])
    fig.update_layout(yaxis_title="positions", title="Position outcomes")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Position ledger ({len(df)} positions)", expanded=False):
        st.dataframe(
            df.style.format({
                "entry_price": "{:.3f}",
                "shares": "{:,.1f}",
                "size_usd": "${:,.2f}",
                "realized_pnl": "${:+,.2f}",
            }),
            use_container_width=True, hide_index=True, height=400,
        )

    # ─────────────────────────────────────────────────────────────────
    # Stop-loss sweep — find the SL threshold that maximises realised P&L
    # ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Stop-loss sweep")
    st.caption(
        "Re-run the simulator across a range of `stop_loss_pct` values to find "
        "where the realised P&L peaks. The default 0.50 cuts losers early but "
        "may also cut eventual winners — sweep finds the empirical optimum."
    )
    if st.button("🔍 Run stop-loss sweep", key="sl_sweep_run"):
        sl_values = [round(0.05 * i, 2) for i in range(1, 21)]
        sweep_rows: list[dict] = []
        progress = st.progress(0.0, text="Running sweep…")
        for i, sl in enumerate(sl_values):
            pos_list = replay_maker(
                eligible,
                bankroll_usd=bankroll, kelly_multiplier=kelly,
                max_position_usd=max_pos, min_edge=min_edge, max_edge=max_edge,
                min_yes_price=min_yes, max_yes_price=max_yes,
                n_rungs=n_rungs,
                take_profit_threshold=take_profit,
                stop_loss_threshold=stop_loss,
                stop_loss_pct=sl,
                sigma_inflation_factor=sigma_factor,
                taker_fallback=False,
            )
            ss = summarize(pos_list)
            sweep_rows.append({
                "stop_loss_pct": sl,
                "n_positions": ss.n_positions,
                "n_sl_exits": ss.n_stop_loss,
                "n_tp_exits": ss.n_take_profit,
                "n_won": ss.n_expire_won,
                "n_lost": ss.n_expire_lost,
                "pnl_usd": ss.total_realized_pnl_usd,
            })
            progress.progress((i + 1) / len(sl_values),
                              text=f"sl_pct={sl:.2f} done…")
        progress.empty()

        df = pd.DataFrame(sweep_rows)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["stop_loss_pct"], y=df["pnl_usd"],
            mode="lines+markers", name="Realised P&L",
            line=dict(color="#1f8b4c", width=3),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="stop_loss_pct (relative MTM exit threshold)",
            yaxis_title="Realised P&L ($)",
            hovermode="x unified",
            title="P&L vs stop-loss threshold (maker-ladder)",
        )
        st.plotly_chart(fig, use_container_width=True)

        best = df.loc[df["pnl_usd"].idxmax()]
        worst = df.loc[df["pnl_usd"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best stop_loss_pct",
                  f"{best['stop_loss_pct']:.2f}",
                  f"P&L ${best['pnl_usd']:+,.2f}")
        c2.metric("Worst stop_loss_pct",
                  f"{worst['stop_loss_pct']:.2f}",
                  f"P&L ${worst['pnl_usd']:+,.2f}")
        c3.metric("Spread",
                  f"${best['pnl_usd'] - worst['pnl_usd']:+,.2f}",
                  "value of optimisation")

        with st.expander(f"Sweep table ({len(df)} rows)", expanded=False):
            st.dataframe(
                df.style.format({
                    "stop_loss_pct": "{:.2f}",
                    "pnl_usd": "${:+,.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

    # Drill-down: pick a position to inspect its event timeline
    st.subheader("Inspect a position's event timeline")
    if positions:
        labels = [
            f"{i}: {p.station_id} {p.target} {p.target_date} {p.bucket_label} ({p.status})"
            for i, p in enumerate(positions)
        ]
        sel = st.selectbox("Pick a position", labels, index=0)
        idx = int(sel.split(":")[0])
        p = positions[idx]
        ev_rows = [{
            "step": i + 1,
            "time": ev.issue_time_utc,
            "action": ev.action,
            "fill_price": ev.fill_price,
            "shares": ev.shares,
            "cash_flow": ev.cash_flow_usd,
            "our_prob": ev.our_prob_at_step,
            "market_mid": ev.market_yes_implied_at_step,
        } for i, ev in enumerate(p.events)]
        st.dataframe(
            pd.DataFrame(ev_rows).style.format({
                "fill_price": "{:.3f}",
                "shares": "{:,.1f}",
                "cash_flow": "${:+,.2f}",
                "our_prob": "{:.1%}",
                "market_mid": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )


def render_signals(
    bias_path: str, bankroll: float, kelly: float, max_pos: float,
    min_edge: float, max_edge: float, min_yes: float, max_yes: float,
    min_volume: float, sigma_factor: float, per_event_cap: float,
) -> None:
    st.subheader("Live trade signals (5-min cache)")
    if st.button("🔄 Refresh signals"):
        cached_signals.clear()  # invalidate
        st.rerun()              # force re-render so the next call refetches

    cache_key = (bias_path, bankroll, kelly, max_pos, max_edge, min_yes, max_yes,
                 sigma_factor, per_event_cap)
    with st.spinner("Scanning Polymarket and fetching forecasts…"):
        signals = cached_signals(*cache_key)

    # Apply user filters
    signals = [
        s for s in signals
        if s.edge >= min_edge and s.volume_24hr >= min_volume
    ]
    if not signals:
        st.warning("No signals match current filters.")
        return

    rows = [{
        "rank": i + 1,
        "station": s.station.name,
        "target": s.target,
        "date": str(s.target_date),
        "bucket": s.bucket_label,
        "ours": s.our_prob,
        "market": s.yes_implied,
        "side": s.side,
        "fill": s.fill_price,
        "edge": s.edge,
        "kelly": s.kelly_full,
        "size_usd": s.position_usd,
        "vol24": s.volume_24hr,
        "sigma_tot": s.sigma_total_c,
        "bias": s.bias_applied_c,
    } for i, s in enumerate(signals[:50])]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "ours": "{:.1%}", "market": "{:.1%}", "fill": "{:.3f}",
            "edge": "{:+.1%}", "kelly": "{:.0%}", "size_usd": "${:,.2f}",
            "vol24": "${:,.0f}", "sigma_tot": "{:.2f}", "bias": "{:+.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    total_size = sum(s.position_usd for s in signals[:50])
    st.caption(f"Top 50 signals  ·  total proposed exposure: ${total_size:,.2f}")


# ──────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("🌤️ Weather Bot Dashboard")

    saved = _load_settings()

    with st.sidebar:
        st.header("Sizing")
        bankroll = st.number_input(
            "Bankroll ($)", value=float(saved.get("bankroll", 1000.0)),
            step=100.0, min_value=10.0, key="bankroll",
        )
        kelly = st.number_input(
            "Kelly multiplier", value=float(saved.get("kelly", 0.1)),
            step=0.05, min_value=0.0, max_value=1.0, key="kelly",
            help="0.1 = deci-Kelly. Don't raise without forward-log validation.",
        )
        max_pos = st.number_input(
            "Max position ($)", value=float(saved.get("max_pos", 50.0)),
            step=10.0, min_value=1.0, key="max_pos",
        )
        per_event_cap = st.number_input(
            "Per-event cap ($)", value=float(saved.get("per_event_cap", 30.0)),
            step=10.0, min_value=0.0, key="per_event_cap",
            help="Max exposure across all buckets of a single (station, target, "
                 "date) event. Prevents Wellington-style over-concentration.",
        )
        daily_cap = st.number_input(
            "Daily exposure cap ($)", value=float(saved.get("daily_cap", 0.0)),
            step=25.0, min_value=0.0, key="daily_cap",
            help="0 = no cap (paper-trade research mode). Set >0 to preview "
                 "the live execution cap from TradingConfig.",
        )

        st.header("Filters")
        min_edge = st.slider(
            "Min edge", 0.0, 0.5, float(saved.get("min_edge", 0.05)), 0.01,
            key="min_edge",
        )
        max_edge = st.slider(
            "Max edge", 0.05, 1.0, float(saved.get("max_edge", 0.25)), 0.05,
            key="max_edge",
            help="Drop signals with edge above this — large edges (>25%) "
                 "almost always indicate model error at the tails.",
        )
        min_yes = st.slider(
            "Min fill price", 0.0, 0.5, float(saved.get("min_yes", 0.05)), 0.01,
            key="min_yes",
            help="Skips trades where the price we'd PAY (yes_ask for YES, "
                 "1−yes_bid for NO) is below this. Filters out extreme-tail "
                 "markets where bid/ask is unreliable. Symmetric across both sides.",
        )
        max_yes = st.slider(
            "Max fill price", 0.5, 1.0, float(saved.get("max_yes", 0.95)), 0.01,
            key="max_yes",
            help="Skips trades where the price we'd PAY is above this. "
                 "Symmetric across both YES and NO sides.",
        )
        min_volume = st.number_input(
            "Min 24h volume ($)", value=float(saved.get("min_volume", 100.0)),
            step=100.0, min_value=0.0, key="min_volume",
        )

        st.header("Calibration")
        sigma_factor = st.slider(
            "σ inflation factor", 1.0, 2.5, float(saved.get("sigma_factor", 1.4)), 0.1,
            key="sigma_factor",
            help="Multiplies σ_residual to widen the predictive distribution. "
                 "1.0 = use BiasTable σ as-is (likely under-estimates "
                 "real 1-day-lead error). Default 1.4 is paranoid until "
                 "forward-log calibrates the true factor.",
        )

        st.divider()
        bias_path = st.text_input(
            "bias_table.json path",
            value=saved.get("bias_path", "bias_table.json"),
            key="bias_path",
        )

    # Merge current sidebar values into `saved` so (a) render_positions
    # sees the live values, and (b) the final save below has the full set.
    # Critical: do NOT _save_settings() yet — render_positions may add
    # position-tab keys to `saved`, and we want a single canonical write
    # at the end. (Earlier code did a save here AND a save after tabs;
    # the second one clobbered the first with stale `saved` keys.)
    saved.update({
        "bankroll": bankroll, "kelly": kelly, "max_pos": max_pos,
        "per_event_cap": per_event_cap, "daily_cap": daily_cap,
        "min_edge": min_edge, "max_edge": max_edge,
        "min_yes": min_yes, "max_yes": max_yes,
        "min_volume": min_volume, "sigma_factor": sigma_factor,
        "bias_path": bias_path,
    })

    records = cached_records()
    bias_meta = cached_bias_table_meta(bias_path)

    tab_o, tab_s, tab_p, tab_pos, tab_sig = st.tabs(
        ["Overview", "Skill", "P&L", "Positions", "Live signals"]
    )
    with tab_o:
        render_overview(records, bias_meta)
    with tab_s:
        render_skill(records)
    with tab_p:
        render_pnl(records, bankroll, kelly, max_pos, min_edge, max_edge,
                   min_yes, max_yes, sigma_factor)
    with tab_pos:
        render_positions(records, bankroll, kelly, max_pos, min_edge,
                         max_edge, min_yes, max_yes, sigma_factor, saved)
    # Single canonical save: includes both sidebar values (merged above)
    # and any position-tab keys that render_positions added to `saved`.
    _save_settings(saved)
    with tab_sig:
        render_signals(
            bias_path, bankroll, kelly, max_pos,
            min_edge, max_edge, min_yes, max_yes, min_volume,
            sigma_factor, per_event_cap,
        )


if __name__ == "__main__":
    main()
