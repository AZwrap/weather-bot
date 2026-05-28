"""Streamlit dashboard for the slim_daemon.

Run on the VPS under systemd (deploy/slim-dashboard.service) or
directly for local development:

    streamlit run slim_dashboard.py

Access from your laptop / WSL via SSH tunnel:

    ssh -L 8501:localhost:8501 weather-vps2
    # then open http://localhost:8501 in your Windows browser

Read-only. Never mutates portfolio.json or any log file.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DATA = Path("data")

LOGS = {
    "intraday": DATA / "intraday_log.jsonl",
    "layer7": DATA / "guaranteed_no_buy_log.jsonl",
    "layer7_margin_filtered": DATA / "margin_filter_log.jsonl",
    "v2": DATA / "v2_conditional_log.jsonl",
    "hbn": DATA / "high_bucket_no_log.jsonl",
    "publication_window": DATA / "publication_window_log.jsonl",
    "portfolio_audit": DATA / "portfolio_save_audit.jsonl",
}

PORTFOLIO_PATH = DATA / "portfolio.json"
FEE_CACHE_PATH = DATA / "fee_config_cache.json"
EXCLUSIONS_PATH = DATA / "excluded_stations.json"

st.set_page_config(
    page_title="Weather Bot Dashboard",
    page_icon="🌤️",
    layout="wide",
)


# ── Helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_jsonl(path_str: str) -> list[dict]:
    p = Path(path_str)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@st.cache_data(ttl=15)
def load_json(path_str: str) -> dict | None:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=15)
def systemd_status(service: str) -> dict[str, str]:
    """Read systemd state via systemctl show. Returns a flat dict of
    Property=Value pairs we care about."""
    try:
        r = subprocess.run(
            ["systemctl", "show", service,
             "--property=ActiveState,SubState,MainPID,MemoryCurrent,ExecMainStartTimestamp,NRestarts"],
            capture_output=True, text=True, timeout=5,
        )
        out: dict[str, str] = {}
        for line in r.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
        return out
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


@st.cache_data(ttl=15)
def recent_journal(service: str, lines: int = 200) -> list[str]:
    try:
        r = subprocess.run(
            ["journalctl", "-u", service, "-n", str(lines),
             "--no-pager", "-o", "short-iso"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def fmt_age(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - t).total_seconds()
        if age_s < 60:
            return f"{age_s:.0f}s"
        if age_s < 3600:
            return f"{age_s/60:.1f}m"
        if age_s < 86400:
            return f"{age_s/3600:.1f}h"
        return f"{age_s/86400:.1f}d"
    except (ValueError, TypeError):
        return "—"


def fmt_mem(bytes_str: str) -> str:
    try:
        b = int(bytes_str)
        if b < 1024**2:
            return f"{b/1024:.1f} KB"
        if b < 1024**3:
            return f"{b/1024**2:.0f} MB"
        return f"{b/1024**3:.2f} GB"
    except (ValueError, TypeError):
        return "—"


# ── Sidebar controls ────────────────────────────────────────────────
st.sidebar.title("🌤️ Weather Bot")
st.sidebar.caption(f"Now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=True)
if st.sidebar.button("Force refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Quick commands")
st.sidebar.code(
    "sudo systemctl restart slim-daemon\n"
    "sudo journalctl -u slim-daemon -f\n"
    "touch ~/Weather_Bot/KILL_SWITCH",
    language="bash",
)

# ── Top status bar ──────────────────────────────────────────────────
daemon = systemd_status("slim-daemon.service")
portfolio = load_json(str(PORTFOLIO_PATH)) or {}
intraday = load_jsonl(str(LOGS["intraday"]))
layer7 = load_jsonl(str(LOGS["layer7"]))
v2 = load_jsonl(str(LOGS["v2"]))
hbn = load_jsonl(str(LOGS["hbn"]))
pub_window = load_jsonl(str(LOGS["publication_window"]))
audit = load_jsonl(str(LOGS["portfolio_audit"]))

active_state = daemon.get("ActiveState", "?")
sub_state = daemon.get("SubState", "?")
status_color = "🟢" if active_state == "active" else "🔴"

last_audit_ts = audit[-1].get("ts_utc") if audit else None

st.title("Weather Bot — Live Dashboard")
top = st.columns(6)
top[0].metric(
    "Daemon",
    f"{status_color} {active_state}",
    f"{sub_state}, restarts: {daemon.get('NRestarts', '?')}",
)
top[1].metric("Memory", fmt_mem(daemon.get("MemoryCurrent", "")))
top[2].metric(
    "Uptime since",
    daemon.get("ExecMainStartTimestamp", "—")[:19] or "—",
)
top[3].metric(
    "Last save",
    fmt_age(last_audit_ts) + " ago" if last_audit_ts else "—",
)

positions = portfolio.get("positions", []) if isinstance(portfolio, dict) else []
total_fires_today = sum(
    1 for r in intraday + layer7 + v2 + hbn
    if (r.get("result") == "filled" or r.get("decision") in ("placed", "BUY_EARLY_TAIL"))
    and (r.get("ts_utc") or r.get("scan_time_utc") or "").startswith(
        datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
)
top[4].metric("Synthetic positions", len(positions))
top[5].metric("Strategy fires today", total_fires_today)


# ── Tabs ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Strategy fires",
    "WUG / lock-in",
    "Layer 7",
    "V2 preposit",
    "High-bucket NO",
    "Publication window",
    "Portfolio",
    "Logs",
])

# Tab 0 — Combined strategy fires
with tabs[0]:
    rows: list[dict] = []
    for r in intraday:
        if r.get("decision") == "BUY_EARLY_TAIL":
            rows.append({
                "strategy": "lock-in YES",
                "ts": r.get("scan_time_utc"),
                "station": r.get("station_id"),
                "target": r.get("target"),
                "date": r.get("target_date"),
                "bucket": f"{r.get('winning_bucket_kind')}/{r.get('winning_bucket_threshold')}",
                "size_usd": None,
                "reason": (r.get("reason") or "")[:60],
            })
    for r in layer7:
        if r.get("result") == "filled":
            rows.append({
                "strategy": "Layer 7",
                "ts": r.get("ts_utc"),
                "station": r.get("station_id"),
                "target": r.get("target") or "—",
                "date": r.get("target_date"),
                "bucket": r.get("bucket_label"),
                "size_usd": r.get("size_usd"),
                "reason": f"obs={r.get('observed_extreme_c')}",
            })
    for r in v2:
        if r.get("decision") == "placed":
            rows.append({
                "strategy": "V2",
                "ts": r.get("ts_utc"),
                "station": r.get("station_id"),
                "target": r.get("target"),
                "date": r.get("target_date"),
                "bucket": r.get("bucket_label"),
                "size_usd": r.get("intended_size_usd"),
                "reason": f"maker@${r.get('maker_intended_price')} vs taker@{r.get('taker_no_ask'):.3f}"
                if r.get("taker_no_ask") is not None else "",
            })
    for r in hbn:
        if r.get("result") == "filled":
            rows.append({
                "strategy": "high-bucket NO",
                "ts": r.get("ts_utc"),
                "station": r.get("station_id"),
                "target": r.get("target"),
                "date": r.get("target_date"),
                "bucket": r.get("bucket_label"),
                "size_usd": r.get("size_usd"),
                "reason": f"no_ask=${r.get('no_ask'):.3f}",
            })
    if rows:
        df = pd.DataFrame(rows).sort_values("ts", ascending=False)
        st.write(f"**{len(df)} fires** across all strategies (paper)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        counts = df["strategy"].value_counts().reset_index()
        counts.columns = ["strategy", "fires"]
        st.bar_chart(counts.set_index("strategy"))
    else:
        st.info("No strategy fires yet. Daemon just started or no qualifying events.")


# Tab 1 — WUG + lock-in
with tabs[1]:
    st.subheader("Lock-in YES decisions (WUG primary, METAR fallback)")
    if intraday:
        df = pd.DataFrame(intraday).sort_values("scan_time_utc", ascending=False).head(50)
        cols = [c for c in ["scan_time_utc", "station_id", "target", "target_date",
                            "decision", "extreme_so_far_c",
                            "winning_bucket_kind", "winning_bucket_threshold",
                            "reason", "n_observations_used"]
                if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

        st.subheader("Decision counts")
        counts = Counter(r.get("decision") for r in intraday)
        st.write(dict(counts))
    else:
        st.info("No intraday decisions logged yet.")


# Tab 2 — Layer 7 detail
with tabs[2]:
    if layer7:
        results = Counter(r.get("result") for r in layer7)
        st.subheader("Outcome distribution")
        st.write(dict(results.most_common(10)))

        st.subheader("Recent fills (last 50)")
        fills = [r for r in layer7 if r.get("result") == "filled"]
        if fills:
            df = pd.DataFrame(fills).sort_values("ts_utc", ascending=False).head(50)
            cols = [c for c in ["ts_utc", "station_id", "target_date", "bucket_label",
                                "fill_price", "shares", "size_usd",
                                "observed_extreme_c", "no_ask_at_attempt",
                                "no_ask_source"]
                    if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No filled Layer 7 fires yet.")

        st.subheader("Margin-filtered skips (oracle-disagreement guard)")
        margin = load_jsonl(str(LOGS["layer7_margin_filtered"]))
        if margin:
            df = pd.DataFrame(margin).tail(30)
            cols = [c for c in ["ts_utc", "station_id", "bucket_label",
                                "observed_extreme_c", "margin_c",
                                "yes_ask", "no_ask_implied"]
                    if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No margin-filtered records yet.")
    else:
        st.info("No Layer 7 records logged yet.")


# Tab 3 — V2 preposit
with tabs[3]:
    if v2:
        placed = [r for r in v2 if r.get("decision") == "placed"]
        st.write(f"**{len(placed)}** maker fires (paper); maker vs taker counterfactuals logged.")
        if placed:
            df = pd.DataFrame(placed).sort_values("ts_utc", ascending=False).head(50)
            cols = [c for c in ["ts_utc", "station_id", "target_date", "bucket_label",
                                "yes_ask", "yes_bid",
                                "maker_intended_price", "maker_fee_per_share",
                                "taker_no_ask", "taker_fee_per_share",
                                "intended_size_usd",
                                "gate_bucket_label", "gate_bucket_yes_ask"]
                    if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

            # Maker vs taker price summary
            st.subheader("Maker (GTC @ $0.82) vs taker (FAK @ no_ask) snapshot")
            mt = pd.DataFrame([
                {
                    "maker_price": r["maker_intended_price"],
                    "taker_no_ask": r["taker_no_ask"],
                }
                for r in placed if r.get("taker_no_ask") is not None
            ])
            if not mt.empty:
                col1, col2 = st.columns(2)
                col1.metric("Avg maker price", f"${mt['maker_price'].mean():.4f}")
                col2.metric(
                    "Avg taker no_ask",
                    f"${mt['taker_no_ask'].mean():.4f}",
                    delta=f"{(mt['taker_no_ask']-mt['maker_price']).mean():+.4f}",
                )
    else:
        st.info("No V2 records yet. V2 fires when at least one bucket in an event has yes_ask >= $0.80.")


# Tab 4 — High-bucket NO
with tabs[4]:
    if hbn:
        st.write(f"**{len(hbn)}** high-bucket NO log entries.")
        df = pd.DataFrame(hbn).sort_values("ts_utc", ascending=False).head(50)
        cols = [c for c in ["ts_utc", "result", "station_id", "target", "target_date",
                            "bucket_label", "no_ask", "no_ask_source",
                            "observed_extreme_c", "peak_low_c", "peak_high_c",
                            "bucket_low_c", "bucket_high_c", "local_hour"]
                if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No high-bucket NO fires yet. Strategy fires past 18:00 station-local "
                "(or 06:00 for min-target).")


# Tab 5 — Publication window
with tabs[5]:
    if pub_window:
        st.write(f"**{len(pub_window)}** publication-window snapshots.")
        df = pd.DataFrame([
            {
                "snapshot_ts": r.get("snapshot_ts_utc"),
                "station": r.get("station_id"),
                "target": r.get("target"),
                "target_date": r.get("target_date"),
                "offset_h": r.get("offset_h_after_midend"),
                "metar_c": r.get("metar_final_extreme_c"),
                "wug_max_c": r.get("wug_daily_max_c"),
                "wug_min_c": r.get("wug_daily_min_c"),
                "wug_status": r.get("wug_status"),
                "matched_bucket": (
                    f"{r.get('matched_bucket_kind')}/{r.get('matched_bucket_threshold')}"
                    if r.get("matched_bucket_kind") else "—"
                ),
            }
            for r in pub_window
        ]).sort_values("snapshot_ts", ascending=False).head(100)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Coverage")
        distinct_sk = {(r["station_id"], r["target"], r["target_date"]) for r in pub_window}
        st.metric("Distinct (station, target, date) tuples", len(distinct_sk))
        statuses = Counter(r.get("wug_status") for r in pub_window)
        st.write(f"WUG status distribution: {dict(statuses)}")
    else:
        st.info(
            "No publication-window snapshots yet. First snapshot lands when a "
            "station crosses end-of-resolution-day in local time (≤24h after "
            "daemon start, depending on station's longitude)."
        )


# Tab 6 — Portfolio
with tabs[6]:
    if isinstance(portfolio, dict) and portfolio.get("positions"):
        pos = portfolio["positions"]
        st.write(f"**{len(pos)}** synthetic positions (paper, dry_run=True).")
        df = pd.DataFrame(pos)
        cols = [c for c in ["submitted_at", "station_id", "target_date",
                            "bucket_label", "side", "shares", "entry_price",
                            "position_usd", "status", "strategy"]
                if c in df.columns]
        if cols:
            st.dataframe(
                df[cols].sort_values("submitted_at", ascending=False),
                use_container_width=True, hide_index=True,
            )

        # Per-strategy summary
        if "strategy" in df.columns:
            st.subheader("Per-strategy size")
            by_strat = df.groupby("strategy")["position_usd"].agg(["count", "sum"])
            by_strat.columns = ["positions", "size_usd_total"]
            st.dataframe(by_strat, use_container_width=True)

        # Tracker — Layer 7 progressive eval state
        le = portfolio.get("last_evaluated_max_by_sk", {})
        if le:
            st.subheader("Layer 7 progressive-eval tracker")
            st.caption(
                "Highest temp (int, market unit) we've already evaluated dead "
                "buckets up to. New WUG readings above this trigger evaluation "
                "of the next-up bucket."
            )
            tracker_df = pd.DataFrame([
                {"key": k, "last_evaluated_int": v} for k, v in le.items()
            ]).sort_values("key")
            st.dataframe(tracker_df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions in portfolio.json yet.")


# Tab 7 — Logs
with tabs[7]:
    n_lines = st.slider("Lines to show", min_value=50, max_value=1000,
                        value=200, step=50)
    grep_filter = st.text_input("Filter (substring match)", value="")
    lines = recent_journal("slim-daemon.service", lines=n_lines)
    if grep_filter:
        lines = [l for l in lines if grep_filter.lower() in l.lower()]
    if lines:
        st.code("\n".join(lines[-n_lines:]), language="log")
    else:
        st.info("No journal output (or systemctl not accessible).")


# ── Fee + exclusions footer ─────────────────────────────────────────
with st.expander("Live Polymarket fee config + excluded stations"):
    fee_cfg = load_json(str(FEE_CACHE_PATH))
    if fee_cfg:
        st.write({
            "taker_fee_rate": fee_cfg.get("taker_fee_rate"),
            "maker_rebate_rate": fee_cfg.get("maker_rebate_rate"),
            "source": fee_cfg.get("source"),
            "fetched_at_utc": fee_cfg.get("fetched_at_utc"),
            "age": fmt_age(fee_cfg.get("fetched_at_utc")) + " ago",
        })
    else:
        st.caption("Fee config not yet cached.")

    excl = load_json(str(EXCLUSIONS_PATH))
    if excl:
        st.write("Excluded stations:", excl)


# ── Auto-refresh ────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
