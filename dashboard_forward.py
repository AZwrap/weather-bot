#!/usr/bin/env python3
"""Dark HTML dashboard for the longshot-fade money-free forward test.

Self-contained (stdlib only). Binds 127.0.0.1:PORT — NOT public; reach it via an
SSH tunnel:  ssh -L 8765:127.0.0.1:8765 root@199.247.29.13  then http://localhost:8765

Reads data/longshot_fade/signals.jsonl (written by the scan cron), resolves
matured signals via the public gamma API (cached to disk, resolutions never
change), and renders: realized NO-win% vs the ~97% paper bar, realized ROI,
signals by city / band, fill-gap, and the most recent signals. A background
thread refreshes every 10 min so page loads are instant.
"""
from __future__ import annotations
import json, threading, time, urllib.request
from collections import defaultdict
from datetime import datetime, timezone, date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", 8765
BASE = Path(__file__).resolve().parent / "data" / "longshot_fade"
SIGNALS = BASE / "signals.jsonl"
RESCACHE = BASE / "dash_resolutions.json"
GAMMA = "https://gamma-api.polymarket.com/markets"
REFRESH_SEC = 600
PAPER_WIN, PAPER_ROI = 97.0, 20.0  # the bar to beat

_state = {"html": "<h1 style='color:#ccc;font-family:sans-serif'>starting…</h1>"}
_lock = threading.Lock()

CSS = """
*{box-sizing:border-box} body{margin:0;background:#0d1117;color:#c9d1d9;
font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8b949e;font-size:12px;margin-bottom:20px}
.kpis{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:24px}
.kpi{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;min-width:150px;flex:1}
.kpi .lbl{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.kpi .val{font-size:26px;font-weight:600;margin-top:6px}
.kpi .note{font-size:11px;color:#8b949e;margin-top:4px}
.g{color:#3fb950}.r{color:#f85149}.y{color:#d29922}.m{color:#58a6ff}
h2{font-size:14px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:26px 0 10px}
table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}
th{color:#8b949e;font-weight:500;font-size:11px;text-transform:uppercase;text-align:right;background:#0d1117}
td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:none}
.bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%}
"""


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def resolve_token(tok):
    """Return 'no' / 'yes' (winning side) for a closed market, or None."""
    try:
        arr = _get(f"{GAMMA}?clob_token_ids={tok}&closed=true")
    except Exception:
        return None
    if not arr:
        return None
    prices = arr[0].get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            return None
    if not prices or len(prices) < 2:
        return None
    return "no" if str(prices[1]) in ("1", "1.0") else "yes"


def compute():
    if not SIGNALS.exists():
        return {"n": 0}
    recs, seen = [], set()
    for line in SIGNALS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = (r.get("no_token_id"), r.get("target_date"))
        if k in seen:
            continue
        seen.add(k)
        recs.append(r)
    cache = json.loads(RESCACHE.read_text()) if RESCACHE.exists() else {}
    today = datetime.now(timezone.utc).date()
    changed = False
    settled, maturing = [], 0
    for r in recs:
        try:
            matured = date.fromisoformat(r["target_date"]) < today
        except Exception:
            matured = False
        if not matured:
            maturing += 1
            continue
        tok = r.get("no_token_id")
        res = cache.get(tok)
        if res is None:
            res = resolve_token(tok)
            if res is not None:
                cache[tok] = res
                changed = True
        if res is None:
            continue
        entry = r.get("decision_quote") or 0.0
        no_won = res == "no"
        pnl = (1.0 - entry) if no_won else -entry
        settled.append((r, no_won, pnl, entry))
    if changed:
        RESCACHE.parent.mkdir(parents=True, exist_ok=True)
        RESCACHE.write_text(json.dumps(cache))

    n = len(settled)
    wins = sum(1 for _, w, _, _ in settled if w)
    dep = sum(e for *_, e in settled)
    pnl = sum(p for _, _, p, _ in settled)
    gaps = [r["fill_gap"] for r in recs if isinstance(r.get("fill_gap"), (int, float))]
    # per-city
    city = defaultdict(lambda: {"sig": 0, "set": 0, "win": 0, "dep": 0.0, "pnl": 0.0})
    for r in recs:
        city[r.get("city", "?")]["sig"] += 1
    for r, w, p, e in settled:
        c = city[r.get("city", "?")]
        c["set"] += 1; c["win"] += 1 if w else 0; c["dep"] += e; c["pnl"] += p
    # per-timing window (next_day vs same_day_am) — the comparison
    tg = defaultdict(lambda: {"sig": 0, "set": 0, "win": 0, "dep": 0.0, "pnl": 0.0})
    for r in recs:
        tg[r.get("timing", "?")]["sig"] += 1
    for r, w, p, e in settled:
        t = tg[r.get("timing", "?")]
        t["set"] += 1; t["win"] += 1 if w else 0; t["dep"] += e; t["pnl"] += p
    dates = sorted({r.get("target_date") for r in recs})
    return {
        "n": len(recs), "cities": len({r.get("city") for r in recs}),
        "dates": (dates[0], dates[-1]) if dates else ("", ""),
        "settled": n, "maturing": maturing,
        "win_pct": (100 * wins / n) if n else None,
        "roi_pct": (100 * pnl / dep) if dep else None,
        "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "city": city,
        "timing": tg,
        "recent": recs[-15:][::-1],
    }


def _cls(val, good, ok):
    if val is None:
        return "y"
    return "g" if val >= good else ("y" if val >= ok else "r")


def render(s):
    if s.get("n", 0) == 0:
        return f"<!doctype html><html><head><meta http-equiv=refresh content=30><style>{CSS}</style></head><body><div class=wrap><h1>Longshot-fade forward test</h1><div class=sub>No signals logged yet — the scan cron runs every 8h. Check back shortly.</div></div></body></html>"
    win = s["win_pct"]; roi = s["roi_pct"]
    wintxt = f"{win:.1f}%" if win is not None else "—"
    roitxt = f"{roi:+.1f}%" if roi is not None else "—"
    wbar = min(100, win) if win is not None else 0
    gaptxt = f"{s['mean_gap']:+.3f}" if s.get('mean_gap') is not None else "—"
    kpis = (
        f"<div class=kpi><div class=lbl>Signals logged</div><div class='val m'>{s['n']}</div>"
        f"<div class=note>{s['cities']} cities · {s['dates'][0]}→{s['dates'][1]}</div></div>"
        f"<div class=kpi><div class=lbl>Settled</div><div class=val>{s['settled']}</div>"
        f"<div class=note>{s['maturing']} still maturing</div></div>"
        f"<div class=kpi><div class=lbl>NO-win rate</div><div class='val {_cls(win,95,90)}'>{wintxt}</div>"
        f"<div class=note>paper bar {PAPER_WIN:.0f}%</div><div class=bar><i class='{_cls(win,95,90)}' style='width:{wbar:.0f}%;background:currentColor'></i></div></div>"
        f"<div class=kpi><div class=lbl>Realized ROI / mkt</div><div class='val {_cls(roi,15,5)}'>{roitxt}</div>"
        f"<div class=note>paper bar +{PAPER_ROI:.0f}%</div></div>"
        f"<div class=kpi><div class=lbl>Fill-gap (sim)</div><div class=val>{gaptxt}</div>"
        f"<div class=note>quote→fill slippage</div></div>"
    )
    # city table
    crows = ""
    for c, d in sorted(s["city"].items(), key=lambda x: -x[1]["sig"])[:25]:
        wp = (100 * d["win"] / d["set"]) if d["set"] else None
        rp = (100 * d["pnl"] / d["dep"]) if d["dep"] else None
        crows += (f"<tr><td>{c}</td><td>{d['sig']}</td><td>{d['set']}</td>"
                  f"<td class='{_cls(wp,95,90) if wp is not None else ''}'>{f'{wp:.0f}%' if wp is not None else '—'}</td>"
                  f"<td class='{_cls(rp,15,0) if rp is not None else ''}'>{f'{rp:+.0f}%' if rp is not None else '—'}</td></tr>")
    # timing comparison table (next_day vs same_day_am)
    trows = ""
    for tkey, tlabel in (("next_day", "next-day (pure forward)"), ("same_day_am", "same-day a.m. (pre-peak)")):
        d = s["timing"].get(tkey)
        if not d:
            continue
        wp = (100 * d["win"] / d["set"]) if d["set"] else None
        rp = (100 * d["pnl"] / d["dep"]) if d["dep"] else None
        trows += (f"<tr><td>{tlabel}</td><td>{d['sig']}</td><td>{d['set']}</td>"
                  f"<td class='{_cls(wp,95,90) if wp is not None else ''}'>{f'{wp:.0f}%' if wp is not None else '—'}</td>"
                  f"<td class='{_cls(rp,15,0) if rp is not None else ''}'>{f'{rp:+.0f}%' if rp is not None else '—'}</td></tr>")
    # recent table
    rrows = ""
    for r in s["recent"]:
        ts = (r.get("ts", "") or "")[11:16]
        gap = r.get("fill_gap")
        win = "nd" if r.get("timing") == "next_day" else ("am" if r.get("timing") == "same_day_am" else "?")
        rrows += (f"<tr><td>{ts}</td><td>{r.get('city','?')}</td><td>{win}</td><td>{r.get('bucket_label','?')}</td>"
                  f"<td>${r.get('decision_quote',0):.3f}</td>"
                  f"<td>{f'{gap:+.3f}' if isinstance(gap,(int,float)) else '—'}</td></tr>")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=60>"
            f"<title>weatherbot forward test</title><style>{CSS}</style></head><body><div class=wrap>"
            f"<h1>Longshot-fade forward test <span class=m>· paper</span></h1>"
            f"<div class=sub>NO @ 0.75–0.85 on the basket cities, held to resolution · no orders, no money · refreshed {now}</div>"
            f"<div class=kpis>{kpis}</div>"
            f"<h2>By timing — does same-day a.m. differ from next-day?</h2><table><tr><th>window</th><th>signals</th><th>settled</th><th>win%</th><th>roi</th></tr>{trows}</table>"
            f"<h2>By city</h2><table><tr><th>city</th><th>signals</th><th>settled</th><th>win%</th><th>roi</th></tr>{crows}</table>"
            f"<h2>Recent signals</h2><table><tr><th>utc</th><th>city</th><th>win</th><th>bucket</th><th>NO quote</th><th>fill-gap</th></tr>{rrows}</table>"
            f"</div></body></html>")


def refresh_loop():
    while True:
        try:
            html = render(compute())
        except Exception as e:
            html = f"<pre style='color:#f85149'>dashboard error: {e}</pre>"
        with _lock:
            _state["html"] = html
        time.sleep(REFRESH_SEC)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _lock:
            html = _state["html"]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=refresh_loop, daemon=True).start()
    print(f"dashboard on http://{HOST}:{PORT}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
