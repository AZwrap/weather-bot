#!/usr/bin/env python3
"""Weather Bot — custom dark dashboard (lite rebuild).

A dependency-free stdlib HTTP server (no Streamlit). Serves:
  GET /          → the dark, tabbed, auto-refreshing HTML page
  GET /api/data  → the JSON the page renders from

Hero = Positions + P&L. Net-of-fee P&L is computed by joining each live
strategy's fill log against forward_log resolutions with the canonical
scorers (weather_bot.pnl._rounded_observation + bucket_won) and the
weather_bot.fees taker-fee model — the same maths the analyzers use.

Runs as the slim-dashboard systemd service:
  ExecStart=/root/Weather_Bot/.venv/bin/python /root/Weather_Bot/slim_dashboard.py
Binds 127.0.0.1:8501 (reach it via SSH tunnel, same as before).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from weather_bot.fees import taker_fee_usd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

DATA = Path("data")
HOST = "127.0.0.1"
PORT = 8501
REFRESH_S = 20  # client auto-refresh cadence

# Live strategies (the only ones that still fire). Each maps to its fill log
# and the side it bets. consensus_basket carries an explicit per-leg side.
STRATEGIES = {
    "consensus_basket": {"log": "consensus_basket_log.jsonl", "side": None,  "label": "Consensus basket"},
    "high_bucket_no":   {"log": "high_bucket_no_log.jsonl",   "side": "NO",  "label": "High-bucket NO"},
    "persistence_tail": {"log": "persistence_tail_log.jsonl", "side": "NO",  "label": "Persistence tail"},
}


# ─────────────────────────────────────────── loaders

def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _unit(sid: str) -> str:
    s = STATIONS_BY_ID.get(sid)
    return (getattr(s, "unit", None) or "C")


# ─────────────────────────────────────────── caches (slow calls)

_cache: dict = {"sys": (0.0, None), "journal": (0.0, None)}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    ts, val = _cache.get(key, (0.0, None))
    if val is not None and (now - ts) < ttl:
        return val
    val = fn()
    _cache[key] = (now, val)
    return val


def systemd_show(service: str) -> dict:
    def go():
        try:
            r = subprocess.run(
                ["systemctl", "show", service,
                 "--property=ActiveState,SubState,ExecMainStartTimestamp,NRestarts,MemoryCurrent"],
                capture_output=True, text=True, timeout=5,
            )
            d = {}
            for ln in r.stdout.splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    d[k] = v
            return d
        except Exception:
            return {}
    return _cached(f"sys:{service}", 8.0, go)


def journal_tail(service: str, n: int = 250) -> list[str]:
    def go():
        try:
            r = subprocess.run(
                ["journalctl", "-u", service, "-n", str(n), "--no-pager", "-o", "short-iso"],
                capture_output=True, text=True, timeout=6,
            )
            return r.stdout.splitlines()
        except Exception:
            return []
    return _cached(f"jrnl:{service}", 8.0, go)


def read_flags() -> dict:
    """Parse strategy enable flags from source (no heavy import)."""
    flags = {"paper_only": True}
    try:
        src = Path("slim_daemon.py").read_text(encoding="utf-8")
        for name, key in [("PAPER_ONLY", "paper_only"), ("LAYER7_ENABLED", "layer7"),
                          ("CONSENSUS_YES_ENABLED", "consensus_yes"),
                          ("CONSISTENCY_ARB_ENABLED", "consistency_arb")]:
            m = re.search(rf"^{name}: bool = (True|False)", src, re.M)
            if m:
                flags[key] = (m.group(1) == "True")
    except OSError:
        pass
    try:
        v = Path("weather_bot/v2_conditional_preposit.py").read_text(encoding="utf-8")
        m = re.search(r"^V2_ENABLED: bool = (True|False)", v, re.M)
        if m:
            flags["v2"] = (m.group(1) == "True")
    except OSError:
        pass
    return flags


# ─────────────────────────────────────────── P&L engine

def resolution_map() -> dict:
    """(station, target, date) → actual_obs_c, from forward_log."""
    out: dict = {}
    for r in load_jsonl(DATA / "forward_log.jsonl"):
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            out[k] = float(r["actual_obs_c"])
    return out


def _score(sid, target, date, kind, thr, side, shares, fill, resmap):
    """Net-of-fee P&L for one filled leg. status open|resolved."""
    cost = float(shares) * float(fill)
    fee = taker_fee_usd(float(shares), float(fill))
    actual_c = resmap.get((sid, target, date))
    if actual_c is None or kind is None or thr is None:
        return {"status": "open", "net": None, "cost": cost, "fee": fee, "won": None}
    unit = _unit(sid)
    try:
        actual_int = _rounded_observation(actual_c, unit)
        bwon = bucket_won(kind, int(thr), actual_int, unit)
    except Exception:
        return {"status": "open", "net": None, "cost": cost, "fee": fee, "won": None}
    side_won = bwon if side == "YES" else (not bwon)
    payout = float(shares) if side_won else 0.0
    return {"status": "resolved", "net": payout - cost - fee, "cost": cost,
            "fee": fee, "won": side_won, "actual_int": actual_int}


def compute_positions(resmap) -> list[dict]:
    """One row per filled leg across the 3 live strategies (deduped)."""
    rows: list[dict] = []
    seen = set()
    for strat, cfg in STRATEGIES.items():
        for r in load_jsonl(DATA / cfg["log"]):
            if r.get("result") != "filled":
                continue
            side = r.get("side") or cfg["side"] or "NO"
            sid = r.get("station_id"); target = r.get("target"); date = r.get("target_date")
            kind = r.get("bucket_kind"); thr = r.get("bucket_threshold")
            shares = r.get("shares"); fill = r.get("fill_price")
            if shares is None or fill is None:
                continue
            key = (strat, sid, target, date, kind, thr, side)
            if key in seen:
                continue
            seen.add(key)
            sc = _score(sid, target, date, kind, thr, side, shares, fill, resmap)
            rows.append({
                "strategy": strat, "station": sid, "target": target, "date": date,
                "bucket": r.get("bucket_label"), "side": side,
                "shares": round(float(shares), 2), "fill": round(float(fill), 4),
                "ts": r.get("ts_utc"), "depth": r.get("depth_source"),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()},
            })
    rows.sort(key=lambda x: x.get("ts") or "", reverse=True)
    return rows


def compute(resmap=None) -> dict:
    resmap = resmap if resmap is not None else resolution_map()
    positions = compute_positions(resmap)
    today = datetime.now(timezone.utc).date().isoformat()

    by_strat: dict = {}
    by_day: dict = defaultdict(float)
    tot_net = 0.0; tot_open_exp = 0.0; n_open = 0; n_res = 0; wins = 0; losses = 0; today_net = 0.0
    for p in positions:
        s = by_strat.setdefault(p["strategy"], {
            "strategy": p["strategy"], "label": STRATEGIES.get(p["strategy"], {}).get("label", p["strategy"]),
            "net": 0.0, "resolved": 0, "wins": 0, "open": 0, "open_exposure": 0.0})
        if p["status"] == "resolved":
            s["net"] += p["net"]; s["resolved"] += 1
            tot_net += p["net"]; n_res += 1
            by_day[p["date"]] += p["net"]
            if p["date"] == today:
                today_net += p["net"]
            if p["won"]:
                s["wins"] += 1; wins += 1
            else:
                losses += 1
        else:
            s["open"] += 1; s["open_exposure"] += p["cost"]
            n_open += 1; tot_open_exp += p["cost"]
    for s in by_strat.values():
        s["net"] = round(s["net"], 2)
        s["open_exposure"] = round(s["open_exposure"], 2)
        s["win_rate"] = round(s["wins"] / s["resolved"], 3) if s["resolved"] else None

    days = sorted(by_day.items())[-14:]

    # activity feed (recent fills, any field set)
    activity = [{
        "ts": p["ts"], "strategy": p["strategy"], "station": p["station"],
        "target": p["target"], "date": p["date"], "bucket": p["bucket"],
        "side": p["side"], "shares": p["shares"], "fill": p["fill"],
        "status": p["status"], "net": p["net"],
    } for p in positions[:60]]

    resolutions = []
    for r in load_jsonl(DATA / "forward_log.jsonl"):
        if r.get("actual_obs_c") is None:
            continue
        resolutions.append({
            "station": r.get("station_id"), "target": r.get("target"),
            "date": r.get("target_date"), "actual_c": round(float(r["actual_obs_c"]), 2),
            "resolved_at": r.get("resolved_at_utc"), "source": r.get("source"),
        })
    resolutions.sort(key=lambda x: x.get("resolved_at") or "", reverse=True)

    # health
    sysd = systemd_show("slim-daemon")
    fee_cfg = load_json(DATA / "fee_config_cache.json") or {}
    excl = load_json(DATA / "excluded_stations.json") or {}
    flags = read_flags()
    # freshness: newest mtime among the live logs + forward_log
    newest = 0.0
    for cfg in STRATEGIES.values():
        p = DATA / cfg["log"]
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    fl = DATA / "forward_log.jsonl"
    if fl.exists():
        newest = max(newest, fl.stat().st_mtime)
    data_age = int(time.time() - newest) if newest else None

    health = {
        "paper_only": bool(flags.get("paper_only", True)),
        "kill_switch": Path("KILL_SWITCH").exists(),
        "daemon_active": sysd.get("ActiveState", "?"),
        "daemon_sub": sysd.get("SubState", "?"),
        "daemon_since": sysd.get("ExecMainStartTimestamp", ""),
        "daemon_restarts": sysd.get("NRestarts", "?"),
        "data_age_s": data_age,
        "flags": flags,
        "live_strategies": [v["label"] for v in STRATEGIES.values()],
        "taker_fee_rate": fee_cfg.get("taker_fee_rate"),
        "n_excluded": len(excl) if isinstance(excl, (list, dict)) else 0,
    }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refresh_s": REFRESH_S,
        "health": health,
        "pnl": {
            "total_net": round(tot_net, 2),
            "today_net": round(today_net, 2),
            "open_exposure": round(tot_open_exp, 2),
            "n_open": n_open, "n_resolved": n_res,
            "wins": wins, "losses": losses,
            "win_rate": round(wins / n_res, 3) if n_res else None,
            "by_strategy": sorted(by_strat.values(), key=lambda s: -s["net"]),
            "by_day": [{"date": d, "net": round(v, 2)} for d, v in days],
        },
        "positions": {
            "open": [p for p in positions if p["status"] == "open"][:300],
            "resolved": [p for p in positions if p["status"] == "resolved"][:300],
        },
        "activity": activity,
        "resolutions": resolutions[:120],
        "journal": journal_tail("slim-daemon", 250),
    }


# ─────────────────────────────────────────── HTML (inline CSS + JS)

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weather Bot</title>
<style>
:root{
  --bg:#0b0e14; --panel:#141a24; --panel2:#1b2230; --line:#222c3a;
  --txt:#e6edf3; --mut:#8b98a9; --accent:#4aa8ff; --accent2:#2b3445;
  --pos:#3fb950; --neg:#f85149; --warn:#d29922;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
.mono{font-family:"SF Mono","Consolas",ui-monospace,monospace}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;
  background:linear-gradient(90deg,#11161f,#0b0e14);border-bottom:1px solid var(--line);
  position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0;letter-spacing:.3px;font-weight:600}
header .dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto;align-items:center}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line);
  background:var(--panel);color:var(--mut)}
.badge.on{color:var(--pos);border-color:#1d3a25}
.badge.off{color:var(--mut)}
.badge.paper{color:var(--accent);border-color:#1d3550}
.badge.bad{color:var(--neg);border-color:#3a1d1d}
.updated{font-size:11px;color:var(--mut)}
nav{display:flex;gap:4px;padding:10px 22px 0;background:var(--bg);
  border-bottom:1px solid var(--line);position:sticky;top:53px;z-index:4}
nav button{background:none;border:none;color:var(--mut);padding:9px 16px;cursor:pointer;
  font-size:13px;border-bottom:2px solid transparent;border-radius:6px 6px 0 0}
nav button:hover{color:var(--txt);background:var(--panel)}
nav button.active{color:var(--txt);border-bottom-color:var(--accent);font-weight:600}
main{padding:20px 22px 60px;max-width:1500px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:26px;font-weight:700;margin-top:6px;font-family:ui-monospace,monospace}
.card .sub{font-size:12px;color:var(--mut);margin-top:4px}
.pos{color:var(--pos)} .neg{color:var(--neg)} .mut{color:var(--mut)} .warn{color:var(--warn)}
.section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:6px 0 0;margin-bottom:22px;overflow:hidden}
.section h2{font-size:13px;margin:0;padding:14px 18px 10px;color:var(--mut);
  text-transform:uppercase;letter-spacing:.5px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.4px;padding:8px 14px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
td{padding:8px 14px;border-bottom:1px solid #1a212c}
tr:hover td{background:var(--panel2)}
td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
.pill{font-size:11px;padding:2px 8px;border-radius:10px;background:var(--accent2);color:var(--txt)}
.pill.no{background:#2a2030;color:#e6a8d0} .pill.yes{background:#1d3550;color:#9fd0ff}
.tag{font-size:11px;color:var(--mut)}
.bars{display:flex;align-items:flex-end;gap:5px;height:70px;padding:6px 18px 14px}
.bars .b{flex:1;min-width:6px;border-radius:3px 3px 0 0;position:relative}
.bars .b span{position:absolute;bottom:-15px;left:50%;transform:translateX(-50%);
  font-size:9px;color:var(--mut);white-space:nowrap}
.scroll{max-height:520px;overflow:auto}
pre.log{margin:0;padding:14px 18px;font-size:12px;line-height:1.5;color:#b9c4d0;
  max-height:600px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.hidden{display:none}
.empty{padding:26px 18px;color:var(--mut);text-align:center}
.flex{display:flex;gap:22px;flex-wrap:wrap}.flex>.section{flex:1;min-width:340px}
.right{margin-left:auto}
a.refresh{color:var(--accent);cursor:pointer;font-size:12px;text-decoration:none}
</style></head>
<body>
<header>
  <span id="hdot" class="dot"></span>
  <h1>Weather Bot <span class="tag mono" id="mode"></span></h1>
  <div class="badges" id="badges"></div>
</header>
<nav id="nav"></nav>
<main>
  <div class="updated" style="margin-bottom:14px">
    Updated <span id="updated" class="mono"></span> · auto-refresh <span id="rs"></span>s ·
    <a class="refresh" onclick="load()">refresh now</a>
  </div>
  <div id="view"></div>
</main>
<script>
const TABS = ["Overview","Positions & P&L","Strategies","Resolutions","System"];
let TAB = 0, D = null;
const $ = (h)=>{const t=document.createElement('template');t.innerHTML=h.trim();return t.content.firstChild;};
const money=(v)=> v==null?'—':(v>=0?'+':'')+'$'+v.toFixed(2);
const cls=(v)=> v==null?'mut':(v>0?'pos':(v<0?'neg':'mut'));
const pct=(v)=> v==null?'—':(v*100).toFixed(0)+'%';
const esc=(s)=> (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ago=(s)=> s==null?'—':(s<90?s+'s':(s<5400?Math.round(s/60)+'m':Math.round(s/3600)+'h'))+' ago';

function badges(h){
  const f=h.flags||{};
  const b=[];
  b.push(`<span class="badge ${h.paper_only?'paper':'bad'}">${h.paper_only?'PAPER-ONLY':'LIVE ⚠'}</span>`);
  if(h.kill_switch) b.push(`<span class="badge bad">KILL_SWITCH</span>`);
  const da=(h.daemon_active==='active');
  b.push(`<span class="badge ${da?'on':'bad'}">daemon ${esc(h.daemon_active)}</span>`);
  for(const [k,label] of [['consensus_basket','basket'],['high_bucket_no','hi-NO'],['persistence_tail','tail']])
    b.push(`<span class="badge on">${label}</span>`);
  for(const k of ['layer7','v2','consensus_yes','consistency_arb'])
    b.push(`<span class="badge off">${k} off</span>`);
  return b.join('');
}

function navbar(){
  const n=document.getElementById('nav'); n.innerHTML='';
  TABS.forEach((t,i)=>{const btn=$(`<button class="${i===TAB?'active':''}">${t}</button>`);
    btn.onclick=()=>{TAB=i;render();}; n.appendChild(btn);});
}

function tbl(cols, rows, render){
  if(!rows||!rows.length) return `<div class="empty">Nothing yet.</div>`;
  let h='<div class="scroll"><table><thead><tr>'+cols.map(c=>`<th class="${c.n?'num':''}">${c.t}</th>`).join('')+'</tr></thead><tbody>';
  h+=rows.map(r=>'<tr>'+render(r)+'</tr>').join('');
  return h+'</tbody></table></div>';
}

function sidePill(s){return `<span class="pill ${s==='NO'?'no':'yes'}">${s}</span>`;}

function overview(){
  const p=D.pnl, h=D.health;
  let html=`<div class="cards">
    <div class="card"><div class="k">Realized P&L (net of fee)</div>
      <div class="v ${cls(p.total_net)}">${money(p.total_net)}</div>
      <div class="sub">${p.n_resolved} resolved · win ${pct(p.win_rate)}</div></div>
    <div class="card"><div class="k">Today</div>
      <div class="v ${cls(p.today_net)}">${money(p.today_net)}</div>
      <div class="sub">UTC ${esc(D.generated_utc.slice(0,10))}</div></div>
    <div class="card"><div class="k">Open exposure</div>
      <div class="v">$${p.open_exposure.toFixed(2)}</div>
      <div class="sub">${p.n_open} open positions</div></div>
    <div class="card"><div class="k">Record</div>
      <div class="v">${p.wins}<span class="mut" style="font-size:16px">/${p.wins+p.losses}</span></div>
      <div class="sub">${p.losses} losses</div></div>
    <div class="card"><div class="k">Data freshness</div>
      <div class="v ${h.data_age_s!=null&&h.data_age_s<600?'pos':'warn'}" style="font-size:22px">${ago(h.data_age_s)}</div>
      <div class="sub">daemon restarts: ${esc(h.daemon_restarts)}</div></div>
  </div>`;

  // by-day bars
  const days=p.by_day||[];
  if(days.length){
    const mx=Math.max(1,...days.map(d=>Math.abs(d.net)));
    html+=`<div class="section"><h2>Realized P&L by event date (net of fee)</h2><div class="bars">`+
      days.map(d=>{const hgt=Math.max(3,Math.abs(d.net)/mx*52);
        return `<div class="b" title="${d.date}: ${money(d.net)}" style="height:${hgt}px;background:${d.net>=0?'var(--pos)':'var(--neg)'}"><span>${d.date.slice(5)}</span></div>`;}).join('')+
      `</div></div>`;
  }

  html+=`<div class="section"><h2>Per-strategy P&L</h2>`+tbl(
    [{t:'Strategy'},{t:'Net',n:1},{t:'Resolved',n:1},{t:'Win rate',n:1},{t:'Open',n:1},{t:'Open $',n:1}],
    p.by_strategy,
    s=>`<td>${esc(s.label)}</td><td class="num ${cls(s.net)}">${money(s.net)}</td>
        <td class="num">${s.resolved}</td><td class="num">${pct(s.win_rate)}</td>
        <td class="num">${s.open}</td><td class="num">$${s.open_exposure.toFixed(2)}</td>`)+`</div>`;

  html+=`<div class="section"><h2>Recent activity</h2>`+tbl(
    [{t:'Time'},{t:'Strategy'},{t:'Station'},{t:'Bucket'},{t:'Side'},{t:'Shares',n:1},{t:'Fill',n:1},{t:'Status'},{t:'Net',n:1}],
    D.activity,
    r=>`<td class="mono tag">${esc((r.ts||'').slice(5,19).replace('T',' '))}</td>
        <td>${esc((r.strategy||'').replace('consensus_basket','basket').replace('high_bucket_no','hi-NO').replace('persistence_tail','tail'))}</td>
        <td>${esc(r.station)} <span class="tag">${esc(r.target)}</span></td>
        <td>${esc(r.bucket)}</td><td>${sidePill(r.side)}</td>
        <td class="num">${r.shares}</td><td class="num">${r.fill}</td>
        <td><span class="tag">${esc(r.status)}</span></td>
        <td class="num ${cls(r.net)}">${r.net==null?'—':money(r.net)}</td>`)+`</div>`;
  return html;
}

function positionsTab(){
  const o=D.positions.open, r=D.positions.resolved;
  const row=(p,resolved)=>`<td class="mono tag">${esc((p.ts||'').slice(5,16).replace('T',' '))}</td>
      <td>${esc(p.strategy.replace('consensus_basket','basket').replace('high_bucket_no','hi-NO').replace('persistence_tail','tail'))}</td>
      <td>${esc(p.station)} <span class="tag">${esc(p.target)} ${esc(p.date)}</span></td>
      <td>${esc(p.bucket)}</td><td>${sidePill(p.side)}</td>
      <td class="num">${p.shares}</td><td class="num">${p.fill}</td>
      <td class="num">$${p.cost.toFixed(2)}</td>`+
      (resolved?`<td class="num">${p.won?'<span class="pos">WON</span>':'<span class="neg">lost</span>'}</td>
      <td class="num ${cls(p.net)}">${money(p.net)}</td>`:'');
  let html=`<div class="flex">
    <div class="section"><h2>Open positions · ${o.length} · $${D.pnl.open_exposure.toFixed(2)} exposure</h2>`+
    tbl([{t:'Filled'},{t:'Strat'},{t:'Event'},{t:'Bucket'},{t:'Side'},{t:'Sh',n:1},{t:'Fill',n:1},{t:'Cost',n:1}],o,p=>row(p,false))+`</div></div>`;
  html+=`<div class="section"><h2>Resolved positions · ${r.length} · net ${money(D.pnl.total_net)}</h2>`+
    tbl([{t:'Filled'},{t:'Strat'},{t:'Event'},{t:'Bucket'},{t:'Side'},{t:'Sh',n:1},{t:'Fill',n:1},{t:'Cost',n:1},{t:'Result',n:1},{t:'Net',n:1}],r,p=>row(p,true))+`</div>`;
  return html;
}

function strategiesTab(){
  let html='';
  for(const s of D.pnl.by_strategy){
    const open=D.positions.open.filter(p=>p.strategy===s.strategy);
    const res=D.positions.resolved.filter(p=>p.strategy===s.strategy).slice(0,40);
    html+=`<div class="section"><h2>${esc(s.label)} — net ${money(s.net)} · win ${pct(s.win_rate)} · ${s.open} open ($${s.open_exposure.toFixed(2)})</h2>`+
      tbl([{t:'Filled'},{t:'Event'},{t:'Bucket'},{t:'Side'},{t:'Fill',n:1},{t:'Status'},{t:'Net',n:1}],
        [...open.slice(0,15),...res],
        p=>`<td class="mono tag">${esc((p.ts||'').slice(5,16).replace('T',' '))}</td>
            <td>${esc(p.station)} <span class="tag">${esc(p.target)} ${esc(p.date)}</span></td>
            <td>${esc(p.bucket)}</td><td>${sidePill(p.side)}</td><td class="num">${p.fill}</td>
            <td><span class="tag">${p.status}</span></td>
            <td class="num ${cls(p.net)}">${p.net==null?'—':money(p.net)}</td>`)+`</div>`;
  }
  return html||`<div class="empty">No strategy data.</div>`;
}

function resolutionsTab(){
  return `<div class="section"><h2>Wunderground resolutions (truth source)</h2>`+tbl(
    [{t:'Resolved at'},{t:'Station'},{t:'Target'},{t:'Date'},{t:'Actual °C',n:1},{t:'Source'}],
    D.resolutions,
    r=>`<td class="mono tag">${esc((r.resolved_at||'').slice(0,19).replace('T',' '))}</td>
        <td>${esc(r.station)}</td><td>${esc(r.target)}</td><td>${esc(r.date)}</td>
        <td class="num">${r.actual_c}</td><td class="tag">${esc(r.source)}</td>`)+`</div>`;
}

function systemTab(){
  const h=D.health,f=h.flags||{};
  let html=`<div class="cards">
    <div class="card"><div class="k">Mode</div><div class="v ${h.paper_only?'pos':'neg'}" style="font-size:20px">${h.paper_only?'PAPER-ONLY':'LIVE'}</div>
      <div class="sub">kill switch: ${h.kill_switch?'<span class="neg">ON</span>':'off'}</div></div>
    <div class="card"><div class="k">Daemon</div><div class="v" style="font-size:20px">${esc(h.daemon_active)}/${esc(h.daemon_sub)}</div>
      <div class="sub">since ${esc((h.daemon_since||'').slice(0,19))}</div></div>
    <div class="card"><div class="k">Taker fee rate</div><div class="v" style="font-size:20px">${h.taker_fee_rate==null?'—':h.taker_fee_rate}</div>
      <div class="sub">${h.n_excluded} excluded stations</div></div>
    <div class="card"><div class="k">Strategy flags</div>
      <div class="sub" style="margin-top:8px;line-height:1.9">
        ${Object.entries({layer7:f.layer7,v2:f.v2,consensus_yes:f.consensus_yes,consistency_arb:f.consistency_arb})
          .map(([k,v])=>`${k}: <span class="${v?'pos':'mut'}">${v?'ON':'off'}</span>`).join('<br>')}</div></div>
  </div>`;
  html+=`<div class="section"><h2>slim-daemon journal (last 250)</h2><pre class="log" id="log">`+
    (D.journal||[]).map(esc).join('\n')+`</pre></div>`;
  return html;
}

function render(){
  navbar();
  document.getElementById('badges').innerHTML=badges(D.health);
  document.getElementById('mode').textContent='';
  document.getElementById('hdot').style.background = (D.health.daemon_active==='active' && D.health.paper_only)?'var(--pos)':'var(--neg)';
  document.getElementById('updated').textContent=(D.generated_utc||'').replace('T',' ');
  document.getElementById('rs').textContent=D.refresh_s;
  const v=document.getElementById('view');
  v.innerHTML=[overview,positionsTab,strategiesTab,resolutionsTab,systemTab][TAB]();
  const lg=document.getElementById('log'); if(lg) lg.scrollTop=lg.scrollHeight;
}

async function load(){
  try{
    const r=await fetch('/api/data',{cache:'no-store'}); D=await r.json(); render();
  }catch(e){ document.getElementById('view').innerHTML='<div class="empty">Failed to load /api/data: '+esc(e)+'</div>'; }
}
load(); setInterval(load, __REFRESH__*1000);
</script>
</body></html>"""


# ─────────────────────────────────────────── server

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path.startswith("/api/data"):
                self._send(200, json.dumps(compute()), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE.replace("__REFRESH__", str(REFRESH_S)), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._send(200, "ok", "text/plain")
            else:
                self._send(404, "not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # never crash the server on one bad request
            try:
                self._send(500, json.dumps({"error": str(exc)}), "application/json")
            except Exception:
                pass

    def log_message(self, *a):  # quiet
        return


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[dashboard] serving http://{HOST}:{PORT}  (refresh {REFRESH_S}s)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
