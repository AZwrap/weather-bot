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
from weather_bot.units import c_to_f

DATA = Path("data")
HOST = "127.0.0.1"
PORT = 8501
REFRESH_S = 20  # client auto-refresh cadence

STRATEGIES = {
    "consensus_basket": {"log": "consensus_basket_log.jsonl", "side": None, "label": "Consensus basket"},
    "high_bucket_no":   {"log": "high_bucket_no_log.jsonl",   "side": "NO", "label": "High-bucket NO"},
    "persistence_tail": {"log": "persistence_tail_log.jsonl", "side": "NO", "label": "Persistence tail"},
    "layer7":           {"log": "guaranteed_no_buy_log.jsonl", "side": "NO", "label": "Layer 7 (guaranteed NO)"},
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

_cache: dict = {}


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
    out: dict = {}
    for r in load_jsonl(DATA / "forward_log.jsonl"):
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            out[k] = float(r["actual_obs_c"])
    return out


def _score(sid, target, date, kind, thr, side, shares, fill, resmap):
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
            "fee": fee, "won": side_won}


def compute_positions(resmap) -> list[dict]:
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
            # Skip unscoreable rows. Legacy Layer 7 fills (pre-2026-06-03) lack
            # target/kind/threshold; without them the engine can't resolve them
            # and they'd show as permanently-open zombies. The other strategy
            # logs always carry all three, so this only drops the old L7 rows.
            if (shares is None or fill is None or kind is None
                    or thr is None or target is None):
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


def _agg(rows: list[dict], today: str) -> dict:
    net = open_exp = today_net = 0.0
    n_open = n_res = wins = losses = 0
    byday: dict = defaultdict(float)
    for p in rows:
        if p["status"] == "resolved":
            net += p["net"]; n_res += 1; byday[p["date"]] += p["net"]
            if p["date"] == today:
                today_net += p["net"]
            if p["won"]:
                wins += 1
            else:
                losses += 1
        else:
            n_open += 1; open_exp += p["cost"]
    days = sorted(byday.items())[-14:]
    return {
        "total_net": round(net, 2), "today_net": round(today_net, 2),
        "open_exposure": round(open_exp, 2),
        "n_open": n_open, "n_resolved": n_res, "wins": wins, "losses": losses,
        "win_rate": round(wins / n_res, 3) if n_res else None,
        "by_day": [{"date": d, "net": round(v, 2)} for d, v in days],
    }


def compute(resmap=None) -> dict:
    resmap = resmap if resmap is not None else resolution_map()
    positions = compute_positions(resmap)
    today = datetime.now(timezone.utc).date().isoformat()

    # Prune unresolvable ZOMBIE open positions on excluded dodgy-source stations
    # (Istanbul/Moscow/Tel Aviv/etc. resolve on weather.gov/NOAA/HKO, not our WUG
    # oracle, so they can NEVER resolve and would hang "open" forever, cluttering
    # the view). They're already dropped from all P&L analysis; this just hides
    # them from the open list. Count is surfaced in health.n_open_excluded.
    _excl_raw = load_json(DATA / "excluded_stations.json") or []
    _excl_ids = {r.get("station_id") for r in _excl_raw if isinstance(r, dict)} \
        if isinstance(_excl_raw, list) else set()
    n_open_excluded = sum(1 for p in positions
                          if p["status"] == "open" and p["station"] in _excl_ids)
    positions = [p for p in positions
                 if not (p["status"] == "open" and p["station"] in _excl_ids)]

    all_agg = _agg(positions, today)
    by_strategy = {}
    for strat in STRATEGIES:
        a = _agg([p for p in positions if p["strategy"] == strat], today)
        a["label"] = STRATEGIES[strat]["label"]
        by_strategy[strat] = a

    activity = [{
        "ts": p["ts"], "strategy": p["strategy"], "station": p["station"],
        "target": p["target"], "date": p["date"], "bucket": p["bucket"],
        "side": p["side"], "shares": p["shares"], "fill": p["fill"],
        "status": p["status"], "net": p["net"],
    } for p in positions[:80]]

    resolutions = []
    for r in load_jsonl(DATA / "forward_log.jsonl"):
        if r.get("actual_obs_c") is None:
            continue
        actual_c = float(r["actual_obs_c"])
        unit = _unit(r.get("station_id"))
        native = round(c_to_f(actual_c), 1) if unit == "F" else round(actual_c, 1)
        resolutions.append({
            "station": r.get("station_id"), "target": r.get("target"),
            "date": r.get("target_date"), "actual_c": round(actual_c, 1),
            "actual_native": native, "unit": unit,
            "resolved_at": r.get("resolved_at_utc"), "source": r.get("source"),
        })
    resolutions.sort(key=lambda x: x.get("resolved_at") or "", reverse=True)

    sysd = systemd_show("slim-daemon")
    fee_cfg = load_json(DATA / "fee_config_cache.json") or {}
    excl_raw = load_json(DATA / "excluded_stations.json") or []
    excluded = []
    if isinstance(excl_raw, list):
        for r in excl_raw:
            tag = (r.get("reason") or "").split("(")[0].strip()
            excluded.append({"station": r.get("station_id"), "reason": tag[:64]})
    flags = read_flags()

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
        "taker_fee_rate": fee_cfg.get("taker_fee_rate"),
        "n_excluded": len(excluded),
        "n_open_excluded": n_open_excluded,
        "excluded": excluded,
    }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refresh_s": REFRESH_S,
        "health": health,
        "pnl": {"all": all_agg, "by_strategy": by_strategy, "order": list(STRATEGIES.keys())},
        "positions": {
            "open": [p for p in positions if p["status"] == "open"][:500],
            "resolved": [p for p in positions if p["status"] == "resolved"][:500],
        },
        "activity": activity,
        "resolutions": resolutions[:200],
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
  --txt:#e6edf3; --mut:#8b98a9; --accent:#4aa8ff;
  --pos:#3fb950; --neg:#f85149; --warn:#d29922;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;
  background:linear-gradient(90deg,#11161f,#0b0e14);border-bottom:1px solid var(--line);
  position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0;letter-spacing:.3px;font-weight:600}
header .dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto;align-items:center}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line);
  background:var(--panel);color:var(--mut)}
.badge.on{color:var(--pos);border-color:#1d3a25}
.badge.paper{color:var(--accent);border-color:#1d3550}
.badge.bad{color:var(--neg);border-color:#3a1d1d}
.updated{font-size:11px;color:var(--mut)}
nav{display:flex;gap:4px;padding:10px 22px 0;background:var(--bg);
  border-bottom:1px solid var(--line);position:sticky;top:53px;z-index:4}
nav button{background:none;border:none;color:var(--mut);padding:9px 16px;cursor:pointer;
  font-size:13px;border-bottom:2px solid transparent;border-radius:6px 6px 0 0}
nav button:hover{color:var(--txt);background:var(--panel)}
nav button.active{color:var(--txt);border-bottom-color:var(--accent);font-weight:600}
main{padding:18px 22px 60px;max-width:1560px}
.filterbar{display:flex;align-items:center;gap:10px;margin-bottom:16px;color:var(--mut);font-size:13px}
.filterbar select{background:var(--panel);color:var(--txt);border:1px solid var(--line);
  border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;outline:none}
.filterbar select:hover{border-color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:26px;font-weight:700;margin-top:6px;font-family:ui-monospace,monospace}
.card .sub{font-size:12px;color:var(--mut);margin-top:4px}
.pos{color:var(--pos)} .neg{color:var(--neg)} .mut{color:var(--mut)} .warn{color:var(--warn)}
.section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  margin-bottom:18px;overflow:hidden}
.section h2{font-size:13px;margin:0;padding:13px 18px;color:var(--mut);
  text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.section h2.fold{cursor:pointer;user-select:none;display:flex;align-items:center;gap:9px}
.section h2.fold:hover{color:var(--txt);background:var(--panel2)}
.caret{display:inline-block;width:12px;color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.4px;padding:8px 14px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--panel);white-space:nowrap}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--txt)}
th .sarr{color:#3a4658;font-size:10px;margin-left:3px}
td{padding:8px 14px;border-bottom:1px solid #1a212c}
tr:hover td{background:var(--panel2)}
td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
.pill{font-size:11px;padding:2px 8px;border-radius:10px}
.pill.no{background:#2a2030;color:#e6a8d0} .pill.yes{background:#1d3550;color:#9fd0ff}
.tag{font-size:11px;color:var(--mut)}
.bars{display:flex;align-items:flex-end;gap:5px;height:78px;padding:8px 18px 18px}
.bars .b{flex:1;min-width:6px;border-radius:3px 3px 0 0;position:relative}
.bars .b span{position:absolute;bottom:-15px;left:50%;transform:translateX(-50%);
  font-size:9px;color:var(--mut);white-space:nowrap}
.scroll{max-height:560px;overflow:auto}
pre.log{margin:0;padding:14px 18px;font-size:12px;line-height:1.5;color:#b9c4d0;
  max-height:620px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.hidden{display:none}
.empty{padding:24px 18px;color:var(--mut);text-align:center}
a.refresh{color:var(--accent);cursor:pointer;font-size:12px;text-decoration:none}
</style></head>
<body>
<header>
  <span id="hdot" class="dot"></span>
  <h1>Weather Bot</h1>
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
const STRATS=[["all","All"],["consensus_basket","Basket"],["high_bucket_no","Hi-NO"],["persistence_tail","Tail"]];
let TAB=0, D=null, FILTER="all";
const SORTS={}, FOLDS={};

const money=(v)=> v==null?"—":(v>=0?"+":"")+"$"+(+v).toFixed(2);
const cls=(v)=> v==null?"mut":(v>0?"pos":(v<0?"neg":"mut"));
const pct=(v)=> v==null?"—":(v*100).toFixed(0)+"%";
const esc=(s)=> (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const ago=(s)=> s==null?"—":((s<90?s+"s":(s<5400?Math.round(s/60)+"m":Math.round(s/3600)+"h"))+" ago");
const sshort=(s)=> ({consensus_basket:"basket",high_bucket_no:"hi-NO",persistence_tail:"tail"}[s]||s);
const sidePill=(s)=> `<span class="pill ${s==="NO"?"no":"yes"}">${s}</span>`;
const tmin=(t)=> esc((t||"").slice(5,16).replace("T"," "));
const cur=()=> FILTER==="all"? D.pnl.all : D.pnl.by_strategy[FILTER];
const fp=(arr)=> FILTER==="all"?arr:arr.filter(x=>x.strategy===FILTER);

function setFilter(v){FILTER=v;render();}
function filterBar(){
  return `<div class="filterbar">Strategy
    <select onchange="setFilter(this.value)">`+
    STRATS.map(([v,l])=>`<option value="${v}" ${FILTER===v?"selected":""}>${l}</option>`).join("")+
    `</select></div>`;
}
function toggleFold(id){FOLDS[id]=(FOLDS[id]===false);render();}
function fold(id,title,body){
  const open=FOLDS[id]!==false;
  return `<div class="section"><h2 class="fold" onclick="toggleFold('${id}')">
      <span class="caret">${open?"▾":"▸"}</span>${title}</h2>
      <div class="${open?"":"hidden"}">${body}</div></div>`;
}
function sortBy(tid,col){
  const s=SORTS[tid];
  if(s&&s.col===col) s.dir=(s.dir==="asc"?"desc":"asc");
  else SORTS[tid]={col,dir:"desc"};
  render();
}
// cols: [{t, key, num, html(row), cls(row)}]
function table(tid, cols, rows){
  if(!rows||!rows.length) return `<div class="empty">Nothing here.</div>`;
  const s=SORTS[tid];
  let rs=rows.slice();
  if(s){
    const c=cols[s.col], k=c.key;
    rs.sort((a,b)=>{
      let va=a[k], vb=b[k];
      if(va==null&&vb==null) return 0;
      if(va==null) return 1;
      if(vb==null) return -1;
      if(c.num){va=+va;vb=+vb;} else {va=(""+va).toLowerCase();vb=(""+vb).toLowerCase();}
      const r=va<vb?-1:(va>vb?1:0);
      return s.dir==="asc"?r:-r;
    });
  }
  let h=`<div class="scroll"><table><thead><tr>`;
  cols.forEach((c,i)=>{
    const arr=(s&&s.col===i)?(s.dir==="asc"?" ▲":" ▼"):`<span class="sarr">⇅</span>`;
    h+=`<th class="${c.num?"num ":""}sortable" onclick="sortBy('${tid}',${i})">${c.t}${arr}</th>`;
  });
  h+=`</tr></thead><tbody>`;
  for(const r of rs){
    h+="<tr>";
    for(const c of cols){
      const cc=(c.num?"num ":"")+(c.cls?c.cls(r):"");
      h+=`<td class="${cc.trim()}">${c.html?c.html(r):esc(r[c.key])}</td>`;
    }
    h+="</tr>";
  }
  return h+`</tbody></table></div>`;
}

const C={
  eventCell:(r)=>`${esc(r.station)} <span class="tag">${esc(r.target)} ${esc(r.date)}</span>`,
};

function badges(h){
  const b=[];
  b.push(`<span class="badge ${h.paper_only?"paper":"bad"}">${h.paper_only?"PAPER-ONLY":"LIVE ⚠"}</span>`);
  if(h.kill_switch) b.push(`<span class="badge bad">KILL_SWITCH</span>`);
  b.push(`<span class="badge ${h.daemon_active==="active"?"on":"bad"}">daemon ${esc(h.daemon_active)}</span>`);
  b.push(`<span class="badge on">basket · hi-NO · tail</span>`);
  return b.join("");
}
function navbar(){
  const n=document.getElementById("nav"); n.innerHTML="";
  TABS.forEach((t,i)=>{const btn=document.createElement("button");
    btn.textContent=t; if(i===TAB)btn.className="active";
    btn.onclick=()=>{TAB=i;render();}; n.appendChild(btn);});
}

function cards(a){
  const h=D.health;
  return `<div class="cards">
    <div class="card"><div class="k">Realized P&L (net of fee)</div>
      <div class="v ${cls(a.total_net)}">${money(a.total_net)}</div>
      <div class="sub">${a.n_resolved} resolved · win ${pct(a.win_rate)}</div></div>
    <div class="card"><div class="k">Today</div>
      <div class="v ${cls(a.today_net)}">${money(a.today_net)}</div>
      <div class="sub">UTC ${esc(D.generated_utc.slice(0,10))}</div></div>
    <div class="card"><div class="k">Open exposure</div>
      <div class="v">$${a.open_exposure.toFixed(2)}</div>
      <div class="sub">${a.n_open} open positions</div></div>
    <div class="card"><div class="k">Record</div>
      <div class="v">${a.wins}<span class="mut" style="font-size:16px">/${a.wins+a.losses}</span></div>
      <div class="sub">${a.losses} losses</div></div>
    <div class="card"><div class="k">Data freshness</div>
      <div class="v ${h.data_age_s!=null&&h.data_age_s<600?"pos":"warn"}" style="font-size:22px">${ago(h.data_age_s)}</div>
      <div class="sub">daemon restarts: ${esc(h.daemon_restarts)}</div></div>
  </div>`;
}
function barsBody(a){
  const days=a.by_day||[];
  if(!days.length) return `<div class="empty">No resolved P&L yet.</div>`;
  const mx=Math.max(1,...days.map(d=>Math.abs(d.net)));
  return `<div class="bars">`+days.map(d=>{const hgt=Math.max(3,Math.abs(d.net)/mx*56);
    return `<div class="b" title="${d.date}: ${money(d.net)}" style="height:${hgt}px;background:${d.net>=0?"var(--pos)":"var(--neg)"}"><span>${d.date.slice(5)}</span></div>`;}).join("")+`</div>`;
}

function overview(){
  const a=cur();
  let html=filterBar()+cards(a);
  html+=fold("ov-bars","Realized P&L by event date (net of fee)", barsBody(a));

  let stratRows=D.pnl.order.map(s=>({strategy:s, ...D.pnl.by_strategy[s]}));
  if(FILTER!=="all") stratRows=stratRows.filter(r=>r.strategy===FILTER);
  html+=fold("ov-strat","Per-strategy P&L", table("ov-strat-t",[
    {t:"Strategy",key:"label"},
    {t:"Net",key:"total_net",num:1,html:r=>money(r.total_net),cls:r=>cls(r.total_net)},
    {t:"Resolved",key:"n_resolved",num:1},
    {t:"Win rate",key:"win_rate",num:1,html:r=>pct(r.win_rate)},
    {t:"Open",key:"n_open",num:1},
    {t:"Open $",key:"open_exposure",num:1,html:r=>"$"+r.open_exposure.toFixed(2)},
  ],stratRows));

  html+=fold("ov-act","Recent activity", table("ov-act-t",[
    {t:"Time",key:"ts",html:r=>`<span class="tag mono">${tmin(r.ts)}</span>`},
    {t:"Strategy",key:"strategy",html:r=>sshort(r.strategy)},
    {t:"Station",key:"station",html:C.eventCell},
    {t:"Bucket",key:"bucket"},
    {t:"Side",key:"side",html:r=>sidePill(r.side)},
    {t:"Shares",key:"shares",num:1},
    {t:"Fill",key:"fill",num:1},
    {t:"Status",key:"status",html:r=>`<span class="tag">${esc(r.status)}</span>`},
    {t:"Net",key:"net",num:1,html:r=>r.net==null?"—":money(r.net),cls:r=>cls(r.net)},
  ],fp(D.activity)));
  return html;
}

function posCols(resolved){
  const c=[
    {t:"Filled",key:"ts",html:r=>`<span class="tag mono">${tmin(r.ts)}</span>`},
    {t:"Strat",key:"strategy",html:r=>sshort(r.strategy)},
    {t:"Event",key:"date",html:C.eventCell},
    {t:"Bucket",key:"bucket"},
    {t:"Side",key:"side",html:r=>sidePill(r.side)},
    {t:"Sh",key:"shares",num:1},
    {t:"Fill",key:"fill",num:1},
    {t:"Cost",key:"cost",num:1,html:r=>"$"+(r.cost||0).toFixed(2)},
  ];
  if(resolved){
    c.push({t:"Result",key:"won",num:1,html:r=>r.won?`<span class="pos">WON</span>`:`<span class="neg">lost</span>`});
    c.push({t:"Net",key:"net",num:1,html:r=>money(r.net),cls:r=>cls(r.net)});
  }
  return c;
}
function positionsTab(){
  const a=cur();
  const open=fp(D.positions.open), res=fp(D.positions.resolved);
  let html=filterBar();
  html+=fold("pos-open",`Open positions · ${a.n_open} · $${a.open_exposure.toFixed(2)} exposure`,
    table("pos-open-t",posCols(false),open));
  html+=fold("pos-res",`Resolved positions · ${a.n_resolved} · net ${money(a.total_net)} · win ${pct(a.win_rate)}`,
    table("pos-res-t",posCols(true),res));
  return html;
}

function strategiesTab(){
  let html="";
  for(const s of D.pnl.order){
    const a=D.pnl.by_strategy[s];
    const rows=[...D.positions.open.filter(p=>p.strategy===s),
                ...D.positions.resolved.filter(p=>p.strategy===s)];
    html+=fold("str-"+s,
      `${esc(a.label)} — net <span class="${cls(a.total_net)}">${money(a.total_net)}</span> · win ${pct(a.win_rate)} · ${a.n_open} open ($${a.open_exposure.toFixed(2)})`,
      table("str-"+s+"-t",[
        {t:"Filled",key:"ts",html:r=>`<span class="tag mono">${tmin(r.ts)}</span>`},
        {t:"Event",key:"date",html:C.eventCell},
        {t:"Bucket",key:"bucket"},
        {t:"Side",key:"side",html:r=>sidePill(r.side)},
        {t:"Fill",key:"fill",num:1},
        {t:"Status",key:"status",html:r=>`<span class="tag">${esc(r.status)}</span>`},
        {t:"Net",key:"net",num:1,html:r=>r.net==null?"—":money(r.net),cls:r=>cls(r.net)},
      ],rows));
  }
  return html||`<div class="empty">No strategy data.</div>`;
}

function resolutionsTab(){
  return fold("res-all","Wunderground resolutions (truth source · station native unit)",
    table("res-t",[
      {t:"Resolved at",key:"resolved_at",html:r=>`<span class="tag mono">${esc((r.resolved_at||"").slice(0,19).replace("T"," "))}</span>`},
      {t:"Station",key:"station"},
      {t:"Target",key:"target"},
      {t:"Date",key:"date"},
      {t:"Actual",key:"actual_native",num:1,html:r=>`${r.actual_native}°${r.unit}`},
      {t:"Source",key:"source",html:r=>`<span class="tag">${esc(r.source)}</span>`},
    ],D.resolutions));
}

function systemTab(){
  const h=D.health,f=h.flags||{};
  let html=`<div class="cards">
    <div class="card"><div class="k">Mode</div><div class="v ${h.paper_only?"pos":"neg"}" style="font-size:20px">${h.paper_only?"PAPER-ONLY":"LIVE"}</div>
      <div class="sub">kill switch: ${h.kill_switch?'<span class="neg">ON</span>':"off"}</div></div>
    <div class="card"><div class="k">Daemon</div><div class="v" style="font-size:20px">${esc(h.daemon_active)}/${esc(h.daemon_sub)}</div>
      <div class="sub">since ${esc((h.daemon_since||"").slice(0,19))}</div></div>
    <div class="card"><div class="k">Taker fee rate</div><div class="v" style="font-size:20px">${h.taker_fee_rate==null?"—":h.taker_fee_rate}</div>
      <div class="sub">${h.n_excluded} excluded stations (below)</div></div>
    <div class="card"><div class="k">Strategy flags</div>
      <div class="sub" style="margin-top:8px;line-height:1.9">
        ${Object.entries({layer7:f.layer7,v2:f.v2,consensus_yes:f.consensus_yes,consistency_arb:f.consistency_arb})
          .map(([k,v])=>`${k}: <span class="${v?"pos":"mut"}">${v?"ON":"off"}</span>`).join("<br>")}</div></div>
  </div>`;
  html+=fold("sys-excl","Excluded stations ("+(h.excluded||[]).length+")",
    table("sys-excl-t",[
      {t:"Station",key:"station"},
      {t:"Reason",key:"reason"},
    ],h.excluded||[]));
  html+=fold("sys-journal","slim-daemon journal (last 250)",
    `<pre class="log" id="log">`+(D.journal||[]).map(esc).join("\n")+`</pre>`);
  return html;
}

function render(){
  navbar();
  document.getElementById("badges").innerHTML=badges(D.health);
  document.getElementById("hdot").style.background=(D.health.daemon_active==="active"&&D.health.paper_only)?"var(--pos)":"var(--neg)";
  document.getElementById("updated").textContent=(D.generated_utc||"").replace("T"," ");
  document.getElementById("rs").textContent=D.refresh_s;
  document.getElementById("view").innerHTML=[overview,positionsTab,strategiesTab,resolutionsTab,systemTab][TAB]();
  const lg=document.getElementById("log"); if(lg) lg.scrollTop=lg.scrollHeight;
}

async function load(){
  try{
    const r=await fetch("/api/data",{cache:"no-store"}); D=await r.json(); render();
  }catch(e){ document.getElementById("view").innerHTML='<div class="empty">Failed to load /api/data: '+esc(e)+"</div>"; }
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
        except Exception as exc:
            try:
                self._send(500, json.dumps({"error": str(exc)}), "application/json")
            except Exception:
                pass

    def log_message(self, *a):
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
