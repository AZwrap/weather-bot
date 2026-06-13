#!/usr/bin/env python3
"""Measure @Poligarch's realized weather edge from Polymarket public data.

Step-1 measurement for the "own forecast + their book as benchmark" study.
Pure stdlib. NO forecast involved -- this is THEIR realized edge by price band,
from on-chain trades + CLOB resolution only. It (a) confirms the premise and
(b) produces the resolved (city,date,bucket,outcome) substrate for the forecast
replay that follows.

Pipeline:
  1. End-walk data-api /activity over a trade-time window (offset caps at ~3000,
     so we page backward via the `end` Unix-seconds param).
  2. Keep weather TRADE rows; aggregate per (conditionId, outcome) -> net shares,
     volume-weighted avg buy price, $ deployed.
  3. Resolve each market via CLOB markets/<conditionId> (winner flag). Cached.
  4. Report realized PnL / win-rate / $ deployed by entry-price band x outcome.

Usage:
  python analyze_poligarch_trades.py [--start-days-ago 4] [--end-days-ago 1]
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path

WALLET = "0xb40e89677d59665d5188541ad860450a6e2a7cc9"  # @Poligarch
DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
WEATHER_RE = re.compile(r"temperature", re.I)
EVENT_RE = re.compile(r"^(?P<target>highest|lowest)-temperature-in-(?P<city>.+?)-on-(?P<date>[a-z]+-\d{1,2}-\d{4})$")

RES_CACHE = Path("data/poligarch/resolutions.json")
_res_cache: dict | None = None


def get_json(url: str, retries: int = 4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
        time.sleep(1.0 * (attempt + 1))
    raise last


def parse_event(event_slug: str):
    m = EVENT_RE.match(event_slug or "")
    return (m.group("target"), m.group("city"), m.group("date")) if m else None


def fetch_window(address: str, start_ts: int, end_ts: int):
    """End-walk /activity backward, collecting weather TRADE rows in [start_ts, end_ts]."""
    rows, seen = [], set()
    cur_end = end_ts
    pages = 0
    while True:
        batch = get_json(f"{DATA_API}/activity?user={address}&limit=500&end={cur_end}")
        pages += 1
        if not isinstance(batch, list) or not batch:
            break
        ts_list = [b.get("timestamp", 0) for b in batch]
        page_min = min(ts_list)
        for b in batch:
            if b.get("type") != "TRADE":
                continue
            ts = b.get("timestamp", 0)
            if ts < start_ts or ts > end_ts:
                continue
            if not WEATHER_RE.search(b.get("title", "") or ""):
                continue
            ev = parse_event(b.get("eventSlug", ""))
            if not ev:
                continue
            key = (b.get("transactionHash"), b.get("asset"))
            if key in seen:
                continue
            seen.add(key)
            target, city, date_str = ev
            rows.append({
                "ts": ts, "target": target, "city": city, "date": date_str,
                "side": b.get("side"), "outcome": b.get("outcome"),
                "outcome_index": b.get("outcomeIndex"), "price": b.get("price"),
                "size": b.get("size"), "usdc": b.get("usdcSize"),
                "condition_id": b.get("conditionId"), "asset": b.get("asset"),
                "title": b.get("title"), "slug": b.get("slug"), "event_slug": b.get("eventSlug"),
                "name": b.get("name"),
            })
        print(f"  page {pages}: +{len(batch)} rows, window-kept {len(rows)}, "
              f"page_min {time.strftime('%m-%d %H:%M', time.gmtime(page_min))}", file=sys.stderr)
        if page_min <= start_ts:
            break
        nxt = page_min - 1
        if nxt >= cur_end:  # no progress guard
            break
        cur_end = nxt
        time.sleep(0.25)
    return rows


def resolve_market(cid: str):
    """Return dict {outcome_str: won_bool, 'closed': bool} via CLOB, cached."""
    global _res_cache
    if _res_cache is None:
        _res_cache = json.loads(RES_CACHE.read_text()) if RES_CACHE.exists() else {}
    if cid in _res_cache:
        return _res_cache[cid]
    m = get_json(f"{CLOB}/markets/{cid}")
    res = None
    if isinstance(m, dict) and m.get("tokens"):
        res = {"closed": bool(m.get("closed"))}
        for t in m["tokens"]:
            won = bool(t.get("winner")) or (t.get("price") in (1, 1.0, "1"))
            res[str(t.get("outcome"))] = won
    _res_cache[cid] = res
    return res


def save_cache():
    if _res_cache is not None:
        RES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        RES_CACHE.write_text(json.dumps(_res_cache))


def aggregate(rows):
    """Per (conditionId, outcome): net shares, vwap buy price, $ deployed."""
    agg = defaultdict(lambda: {"buy_sh": 0.0, "buy_usdc": 0.0, "sell_sh": 0.0, "sell_usdc": 0.0, "n": 0})
    meta = {}
    for r in rows:
        k = (r["condition_id"], r["outcome"])
        a = agg[k]
        sh, usdc = r["size"] or 0, r["usdc"] or 0
        if r["side"] == "BUY":
            a["buy_sh"] += sh; a["buy_usdc"] += usdc
        else:
            a["sell_sh"] += sh; a["sell_usdc"] += usdc
        a["n"] += 1
        meta.setdefault(k, r)
    return agg, meta


def realized_pnl(a, resolved_price):
    """Cash-flow realized PnL for one (market,outcome) cell.
    Counts sell proceeds + net shares still held into resolution -> correct for
    round-trippers (not just hold-to-resolution). Denominator = gross $ bought."""
    net = a["buy_sh"] - a["sell_sh"]
    deployed = a["buy_usdc"]
    pnl = a["sell_usdc"] - a["buy_usdc"] + (net * resolved_price if net > 0 else 0.0)
    vwap = a["buy_usdc"] / a["buy_sh"] if a["buy_sh"] else 0.0
    return deployed, pnl, vwap, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-days-ago", type=float, default=4)
    ap.add_argument("--end-days-ago", type=float, default=1)
    ap.add_argument("--substrate-out", default="data/poligarch/resolved_substrate.json")
    args = ap.parse_args()
    now = int(time.time())
    start_ts = now - int(args.start_days_ago * 86400)
    end_ts = now - int(args.end_days_ago * 86400)
    print(f"Window: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(start_ts))} .. "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(end_ts))}", file=sys.stderr)

    rows = fetch_window(WALLET, start_ts, end_ts)
    print(f"\nCollected {len(rows)} weather trades. Aggregating + resolving...", file=sys.stderr)
    agg, meta = aggregate(rows)
    print(f"  {len(agg)} (market,outcome) cells across {len({k[0] for k in agg})} markets", file=sys.stderr)

    # band table: keyed by (outcome, price-band 0.1)
    bands = defaultdict(lambda: {"deployed": 0.0, "pnl": 0.0, "won": 0.0, "n": 0})
    substrate, resolved_n, unresolved_n = [], 0, 0
    for i, (k, a) in enumerate(agg.items()):
        cid, outcome = k
        if a["buy_sh"] <= 0:
            continue
        res = resolve_market(cid)
        if not res or not res.get("closed"):
            unresolved_n += 1
            continue
        resolved_n += 1
        won = bool(res.get(str(outcome), False))
        resolved_price = 1.0 if won else 0.0
        deployed, pnl, vwap, net = realized_pnl(a, resolved_price)
        band = min(int(vwap * 10), 9) / 10
        bk = (outcome, band)
        bands[bk]["deployed"] += deployed
        bands[bk]["pnl"] += pnl
        bands[bk]["won"] += deployed if won else 0.0
        bands[bk]["n"] += 1
        m = meta[k]
        substrate.append({
            "city": m["city"], "date": m["date"], "target": m["target"],
            "title": m["title"], "slug": m["slug"], "condition_id": cid, "asset": m["asset"],
            "their_outcome": outcome, "their_vwap": round(vwap, 4),
            "their_net_usdc": round(deployed, 2), "won": won,
        })
        if (i + 1) % 100 == 0:
            print(f"    resolved {resolved_n}/{i+1}...", file=sys.stderr)
            save_cache()
    save_cache()

    # report
    print("\n" + "=" * 72)
    print(f"@Poligarch realized weather edge  (resolved markets: {resolved_n}, "
          f"unresolved-skipped: {unresolved_n})")
    tot_dep = sum(b["deployed"] for b in bands.values())
    tot_pnl = sum(b["pnl"] for b in bands.values())
    print(f"  TOTAL: deployed ${tot_dep:,.0f}  realized PnL ${tot_pnl:,.0f}  "
          f"ROI {100*tot_pnl/tot_dep if tot_dep else 0:+.1f}%")
    print(f"\n  {'side':4} {'priceband':9} {'$deployed':>11} {'$pnl':>9} {'ROI%':>7} {'win%$':>7} {'nMkts':>6}")
    for (outcome, band) in sorted(bands, key=lambda x: (x[0], x[1])):
        b = bands[(outcome, band)]
        roi = 100 * b["pnl"] / b["deployed"] if b["deployed"] else 0
        winp = 100 * b["won"] / b["deployed"] if b["deployed"] else 0
        print(f"  {str(outcome):4} {band:.1f}-{band+0.1:.1f}  {b['deployed']:>11,.0f} "
              f"{b['pnl']:>9,.0f} {roi:>+7.1f} {winp:>7.1f} {b['n']:>6}")
    print("=" * 72)

    out = Path(args.substrate_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(substrate, indent=2))
    print(f"\nSaved {len(substrate)} resolved (market,outcome) rows -> {out}")


if __name__ == "__main__":
    main()
