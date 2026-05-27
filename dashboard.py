"""Streamlit dashboard for the live weather bot.

Run on the VPS under systemd (deploy/weather-bot-dashboard.service)
or directly for local development:
    streamlit run dashboard.py

Access from your laptop via SSH tunnel:
    ssh -L 8501:localhost:8501 root@VPS_IP
    # then open http://localhost:8501

Tabs:
  Live trades  — REAL Polymarket positions from data/portfolio.json
                 (current bankroll, realized PnL, active orders, region
                 exposure vs caps, recent fills, maker rebates,
                 cancellations)
  Overview     — bot health: forward-log volume, last cron tick,
                 bias_table freshness, per-station record counts
  Skill        — per-station calibration: MAE / bias / RMSE on resolved
                 records (feeds bias-table tuning + helps spot stations
                 drifting away from spec)

History:
  Earlier versions had P&L (sim), Positions (sim), and Live signals tabs
  driven by `weather_bot.scanner` (model-driven YES/NO edges). Removed
  2026-05-14 because (a) that strategy was killed after N=4 backtests
  showed -$412/day with convergence, and (b) the live bot uses
  METAR + NO_momentum + cross-up cancel, not model-driven entries.
  If/when calibration map at N=30 unlocks model-driven entry, re-add
  via the git history (commit before the cleanup commit).
"""
from __future__ import annotations

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
# ──────────────────────────────────────────────────────────────────────────
# Live trades tab — real Polymarket positions from data/portfolio.json
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def cached_portfolio_raw(path: str, _bust: int = 0) -> dict | None:
    """Load portfolio.json (raw dict). 30-second cache so the dashboard
    stays responsive but reflects new fills/resolutions promptly.
    `_bust` is a manual cache-buster; bump to force a reload."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _fetch_live_balance() -> tuple[float | None, str]:
    """Best-effort fetch of Polymarket Cash balance. Returns (balance, status_msg).
    Returns (None, msg) if SDK not configured (dry-run mode)."""
    try:
        from weather_bot.execution.client import ExecutionClient
        from weather_bot.execution.safety import TradingConfig
        cfg = TradingConfig(enabled=False, bankroll_usd=500.0)
        client = ExecutionClient.from_env(cfg)
        bal = client.get_balance_usdc()
        if bal is None:
            return None, "balance call returned None"
        return bal, "live"
    except RuntimeError as exc:
        return None, f"SDK not configured: {exc}"
    except Exception as exc:
        return None, f"error: {exc}"


def render_live_trades(portfolio_path: str, starting_bankroll: float) -> None:
    """Live monitoring view: real positions from data/portfolio.json.

    Sections:
      [A] Status strip (4 metrics)
      [B] Cumulative realized PnL over time
      [C] Active positions (submitted + filled)
      [D] Per-region exposure with cap utilization
      [E] Recently resolved trades
      [F] Daily maker rebates
      [G] Cancellations / permanent blocks
    """
    st.subheader("Live trades — real Polymarket positions")
    st.caption(
        "Data source: `data/portfolio.json` (persisted by place_orders.py "
        "and updated by poll_fills/poll_resolutions/sync_maker_rebates crons). "
        "30s cache — bump the refresh counter to force reload."
    )

    # Refresh control
    col_refresh, col_bal_btn = st.columns([1, 5])
    if col_refresh.button("🔄 Refresh"):
        cached_portfolio_raw.clear()
        st.rerun()

    raw = cached_portfolio_raw(portfolio_path)
    if raw is None:
        st.info(
            f"No portfolio data yet at `{portfolio_path}`. This is expected "
            f"if `place_orders.py --live` hasn't run. Once it has, this tab "
            f"shows real positions + realized PnL."
        )
        return

    # Build the same domain objects place_orders/poll_resolutions use, so
    # the displayed numbers are guaranteed identical to the running bot.
    from weather_bot.portfolio import (
        Portfolio, PER_EVENT_CAP_RATIO, PER_REGION_CAP_RATIO,
        PORTFOLIO_CAP_RATIO, PERMANENT_BLOCK_AFTER_N_CANCELS,
    )
    portfolio = Portfolio.load(Path(portfolio_path))

    # ── [A] Status strip ──────────────────────────────────────────────
    st.markdown("### A. Current state")
    realized = portfolio.realized_pnl_total()
    rebates = portfolio.total_maker_rebates()
    effective = portfolio.effective_bankroll(starting_bankroll)
    n_open = len(portfolio.open_positions())
    n_filled = len(portfolio.filled_positions())
    filled_usd = portfolio.total_exposure_usd()

    cols = st.columns(4)
    cols[0].metric(
        "Effective bankroll",
        f"${effective:,.2f}",
        f"{(effective - starting_bankroll):+,.2f} vs base ${starting_bankroll:,.0f}",
    )
    cols[1].metric(
        "Realized PnL",
        f"${realized:+,.2f}",
        f"rebates ${rebates:+,.2f}" if rebates else "no rebates yet",
    )
    cols[2].metric(
        "Open positions",
        f"{n_open}",
        f"{n_filled} filled · {n_open - n_filled} resting",
    )
    bal, bal_status = _fetch_live_balance()
    if bal is not None:
        expected = starting_bankroll - filled_usd + realized
        drift = bal - expected
        cols[3].metric(
            "Wallet (Polymarket Cash)",
            f"${bal:,.2f}",
            f"drift ${drift:+.2f} vs expected" if abs(drift) > 0.01 else "matches",
            delta_color="off" if abs(drift) < 0.50 else "inverse",
        )
    else:
        cols[3].metric("Wallet", "—", bal_status)

    # ── [B] Cumulative realized PnL over time ──────────────────────────
    st.divider()
    st.markdown("### B. Cumulative realized PnL")
    resolved = [
        p for p in portfolio.positions
        if p.status == "resolved" and p.resolved_at is not None
           and p.realized_pnl is not None
    ]
    if len(resolved) < 2:
        st.caption(f"Need ≥2 resolved positions for a PnL series "
                   f"(have {len(resolved)}). Will populate as positions resolve.")
    else:
        # Per-day aggregation. Include maker rebates as same-day positive.
        rows = [{
            "date": p.resolved_at[:10],
            "pnl": p.realized_pnl,
            "kind": "resolution",
        } for p in resolved]
        for date_iso, amt in portfolio.daily_maker_rebates.items():
            if amt > 0:
                rows.append({"date": date_iso, "pnl": amt, "kind": "rebate"})
        df_pnl = pd.DataFrame(rows)
        daily = df_pnl.groupby("date")["pnl"].sum().reset_index().sort_values("date")
        daily["cumulative"] = daily["pnl"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["pnl"], name="Daily PnL",
            marker_color=["green" if v >= 0 else "red" for v in daily["pnl"]],
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cumulative"], name="Cumulative",
            mode="lines+markers", line=dict(color="blue", width=3),
            yaxis="y2",
        ))
        fig.update_layout(
            xaxis_title="Date (UTC)",
            yaxis_title="Daily PnL ($)",
            yaxis2=dict(title="Cumulative ($)", overlaying="y", side="right"),
            hovermode="x unified", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── [C] Active positions (submitted + filled) ──────────────────────
    st.divider()
    st.markdown("### C. Active positions")
    open_pos = portfolio.open_positions()
    if not open_pos:
        st.caption("No active positions. Bot is idle (or all trades have resolved).")
    else:
        rows = []
        now = datetime.now(timezone.utc)
        for p in open_pos:
            try:
                age = (now - datetime.fromisoformat(p.submitted_at)).total_seconds() / 3600
            except (ValueError, TypeError):
                age = None
            rows.append({
                "station": p.station_id,
                "region": p.region,
                "date": p.target_date,
                "side": p.side,
                "bucket": p.bucket_label,
                "price": p.entry_price,
                "shares": p.shares,
                "usd": p.position_usd,
                "status": p.status,
                "strategy": p.strategy,
                "age_h": age,
                "order_id": (p.order_id or "")[:14] + ("…" if p.order_id and len(p.order_id) > 14 else ""),
            })
        df = pd.DataFrame(rows).sort_values(["status", "date", "station"])
        st.dataframe(
            df.style.format({
                "price": "{:.3f}", "shares": "{:.1f}", "usd": "${:.2f}",
                "age_h": lambda v: f"{v:.1f}h" if pd.notna(v) else "—",
            }),
            use_container_width=True, hide_index=True,
        )

    # ── [D] Per-region exposure ────────────────────────────────────────
    st.divider()
    st.markdown("### D. Region exposure vs caps")
    caps = portfolio.scaled_caps(starting_bankroll)
    per_region_cap = caps["per_region_cap"]
    portfolio_cap = caps["portfolio_cap"]

    # Build region exposure (filled only — that's what the caps gate)
    region_exposure: dict[str, float] = {}
    for p in portfolio.filled_positions():
        region_exposure[p.region] = region_exposure.get(p.region, 0.0) + p.position_usd
    if not region_exposure:
        st.caption(
            f"No filled exposure yet. Region cap: ${per_region_cap:.2f}, "
            f"portfolio cap: ${portfolio_cap:.2f} "
            f"(based on effective bankroll ${effective:,.2f})."
        )
    else:
        regions = sorted(region_exposure.keys())
        exposures = [region_exposure[r] for r in regions]
        utilizations = [e / per_region_cap * 100 if per_region_cap > 0 else 0
                        for e in exposures]
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=regions, y=exposures, name="Exposure",
            marker_color=["red" if u > 100 else ("orange" if u > 75 else "steelblue")
                          for u in utilizations],
            text=[f"${e:.0f}<br>({u:.0f}%)" for e, u in zip(exposures, utilizations)],
            textposition="outside",
        ))
        fig_r.add_hline(
            y=per_region_cap, line_dash="dash", line_color="red",
            annotation_text=f"Per-region cap ${per_region_cap:.0f}",
            annotation_position="top right",
        )
        fig_r.update_layout(
            xaxis_title="Region", yaxis_title="Exposure ($, filled positions)",
            height=380, showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(
            f"Portfolio total: ${filled_usd:,.2f} / ${portfolio_cap:,.2f} = "
            f"{filled_usd / portfolio_cap * 100 if portfolio_cap > 0 else 0:.0f}% "
            f"utilization. Caps scale with effective bankroll automatically."
        )

    # ── [E] Recently resolved ──────────────────────────────────────────
    st.divider()
    st.markdown("### E. Recently resolved (latest 20)")
    if not resolved:
        st.caption("No resolved positions yet.")
    else:
        # Sort by resolved_at desc, take 20
        sorted_resolved = sorted(
            resolved, key=lambda p: p.resolved_at or "", reverse=True
        )[:20]
        rows = []
        for p in sorted_resolved:
            won = (p.realized_pnl or 0) > 0
            rows.append({
                "resolved": (p.resolved_at or "")[:16].replace("T", " "),
                "station": p.station_id,
                "date": p.target_date,
                "side": p.side,
                "bucket": p.bucket_label,
                "shares": p.shares,
                "entry": p.entry_price,
                "result": "WIN" if won else "LOSS",
                "pnl": p.realized_pnl,
            })
        df_r = pd.DataFrame(rows)

        # Color-code WIN / LOSS rows
        def _highlight(row):
            color = "background-color: #d4edda" if row["result"] == "WIN" else "background-color: #f8d7da"
            return [color] * len(row)

        st.dataframe(
            df_r.style.format({
                "shares": "{:.1f}", "entry": "{:.3f}", "pnl": "${:+,.2f}",
            }).apply(_highlight, axis=1),
            use_container_width=True, hide_index=True,
        )

        # Aggregate WR + ROI
        n_w = sum(1 for p in resolved if (p.realized_pnl or 0) > 0)
        n_l = sum(1 for p in resolved if (p.realized_pnl or 0) <= 0)
        deployed = sum(p.position_usd for p in resolved)
        wr = n_w / (n_w + n_l) if (n_w + n_l) else 0
        roi = sum(p.realized_pnl or 0 for p in resolved) / deployed if deployed else 0
        cols_e = st.columns(4)
        cols_e[0].metric("Resolved total", f"{len(resolved)}")
        cols_e[1].metric("Win rate", f"{wr:.0%}", f"{n_w}W / {n_l}L")
        cols_e[2].metric("Total deployed", f"${deployed:,.2f}")
        cols_e[3].metric("ROI", f"{roi:+.1%}")

    # ── [F] Daily maker rebates ────────────────────────────────────────
    st.divider()
    st.markdown("### F. Daily maker rebates")
    if not portfolio.daily_maker_rebates:
        st.caption(
            "No maker rebates recorded yet. `sync_maker_rebates.py` cron "
            "runs daily at 00:10 UTC; Polymarket pays at $1 minimum, so "
            "small days may show $0."
        )
    else:
        rebate_df = pd.DataFrame([
            {"date": d, "rebate": v}
            for d, v in sorted(portfolio.daily_maker_rebates.items())
        ])
        fig_reb = px.bar(
            rebate_df, x="date", y="rebate",
            labels={"rebate": "Rebate ($)", "date": "Date (UTC)"},
            height=300,
        )
        fig_reb.update_layout(showlegend=False)
        st.plotly_chart(fig_reb, use_container_width=True)
        st.caption(f"Total maker rebates: ${rebates:+,.2f}")

    # ── [G] Cancellations / blocked ────────────────────────────────────
    st.divider()
    st.markdown("### G. Cancellations & blocks")
    cancelled = [p for p in portfolio.positions if p.status == "cancelled"]
    blocked = [
        p for p in cancelled
        if p.cancellation_count >= PERMANENT_BLOCK_AFTER_N_CANCELS
    ]
    if not cancelled:
        st.caption("No cancellations recorded. Bot has had clean execution.")
    else:
        st.write(f"Total cancellations: **{len(cancelled)}**  ·  "
                 f"Permanently blocked: **{len(blocked)}**")
        rows = [{
            "cancelled": (p.last_cancelled_at or "")[:16].replace("T", " "),
            "station": p.station_id,
            "date": p.target_date,
            "side": p.side,
            "bucket": p.bucket_label,
            "count": p.cancellation_count,
            "blocked": "YES" if p.cancellation_count >= PERMANENT_BLOCK_AFTER_N_CANCELS else "",
            "reason": (p.cancellation_reason or "")[:60],
        } for p in sorted(
            cancelled, key=lambda p: p.last_cancelled_at or "", reverse=True
        )[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────


KILL_SWITCH_PATH = Path("KILL_SWITCH")


def _kill_switch_active() -> bool:
    """True if the bot is currently halted via KILL_SWITCH file."""
    return KILL_SWITCH_PATH.exists()


def _kill_switch_mtime() -> str:
    """Human-readable timestamp of when KILL_SWITCH was last toggled."""
    if not KILL_SWITCH_PATH.exists():
        return ""
    try:
        ts = datetime.fromtimestamp(
            KILL_SWITCH_PATH.stat().st_mtime, tz=timezone.utc
        )
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return ""


def _toggle_kill_switch(halt: bool) -> tuple[bool, str]:
    """Create or remove the KILL_SWITCH file.
    Returns (success, message). Failures usually mean the dashboard
    process doesn't have write permission to the project root."""
    try:
        if halt:
            KILL_SWITCH_PATH.write_text(
                f"Halted via dashboard at {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            return True, "Halt request written to KILL_SWITCH file."
        else:
            if KILL_SWITCH_PATH.exists():
                KILL_SWITCH_PATH.unlink()
            return True, "KILL_SWITCH removed. Live submissions resume on next cron tick."
    except OSError as exc:
        return False, f"Filesystem error: {exc}"


def main() -> None:
    st.title("🌤️ Weather Bot — Live Dashboard")

    saved = _load_settings()

    with st.sidebar:
        # ── Bot status / kill switch ─────────────────────────────────
        # Highest-priority sidebar element — always visible regardless
        # of which tab is open. Toggles the same KILL_SWITCH file that
        # intraday_scan.py checks at the start of every cron tick.
        st.header("🔌 Bot status")
        halted = _kill_switch_active()
        if halted:
            st.error(
                "🛑 **HALTED**\n\n"
                "Live submissions are disabled. Existing positions are "
                "untouched; paper logging continues. Next cron tick will "
                "see the KILL_SWITCH file and skip live submission."
            )
            mtime = _kill_switch_mtime()
            if mtime:
                st.caption(f"Halted at: {mtime}")
            if st.button(
                "▶️ Resume live trading", type="primary",
                use_container_width=True, key="resume_btn",
            ):
                ok, msg = _toggle_kill_switch(halt=False)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.success(
                "🟢 **LIVE**\n\n"
                "Submissions enabled. Cron fires at :01, :16, :31, :46 UTC."
            )
            if st.button(
                "🛑 Halt bot", type="secondary",
                use_container_width=True, key="halt_btn",
                help="Creates the KILL_SWITCH file. Next intraday_scan tick "
                     "will skip live submissions (paper logging continues). "
                     "One click — re-enable by clicking Resume.",
            ):
                ok, msg = _toggle_kill_switch(halt=True)
                if ok:
                    st.warning(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()
        st.header("Bankroll")
        bankroll = st.number_input(
            "Starting bankroll ($)", value=float(saved.get("bankroll", 500.0)),
            step=100.0, min_value=10.0, key="bankroll",
            help="Production setting: $500. Used by the Live Trades tab "
                 "as the floor for adaptive-cap math: effective bankroll "
                 "= starting + realized PnL, capped at $2k ceiling. "
                 "Should match TradingConfig.bankroll_usd on the bot.",
        )
        st.caption(
            "Caps derived live (see Live Trades → Region exposure):\n"
            "- Portfolio cap = 80% of effective bankroll\n"
            "- Per-region cap = 20%\n"
            "- Per-event cap = 11%"
        )

        st.divider()
        st.header("Paths")
        bias_path = st.text_input(
            "bias_table.json path",
            value=saved.get("bias_path", "bias_table.json"),
            key="bias_path",
            help="Used by the Overview tab to show bias-table freshness.",
        )

    # Persist just the two settings we still expose
    saved.update({"bankroll": bankroll, "bias_path": bias_path})
    _save_settings(saved)

    records = cached_records()
    bias_meta = cached_bias_table_meta(bias_path)

    tab_live, tab_o, tab_s = st.tabs(
        ["🔴 Live trades", "Overview", "Skill"]
    )
    with tab_live:
        render_live_trades("data/portfolio.json", bankroll)
    with tab_o:
        render_overview(records, bias_meta)
    with tab_s:
        render_skill(records)


if __name__ == "__main__":
    main()
