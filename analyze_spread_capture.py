#!/usr/bin/env python3
"""Spread-capture market-maker sim for the longshot-fade forward test.

The directional sims (static + dynamic maker, taker) all model a NO bet HELD to
resolution -- and all came out negative/thin. A real market-maker doesn't hold
directionally: it quotes BOTH sides, buys NO at the bid and sells NO at the ask,
earns the spread + 25% rebate on two-way recreational flow, and carries only the
residual inventory to resolution. This measures that mechanism -- the one thing the
cohort actually do that we haven't tested.

Flow is normalized across BOTH tokens (the data reports each trade from the taker's
chosen token, no double-count):
  sell-NO flow (fills our resting BID): NO-token SELL  +  YES-token BUY  @ (1-yes_px)
  buy-NO  flow (fills our resting ASK): NO-token BUY   +  YES-token SELL @ (1-yes_px)

MODEL (first-order, hourly, optimistic on queue):
  Keep ~size_usd of NO resting on EACH side at the touch, refilled every trajectory
  tick (bid/ask path). Per interval, fill up to our quote from the flow at our price:
    sell-NO at price <= our bid -> we BUY  (inv +=, cash -=, rebate +=)
    buy-NO  at price >= our ask -> we SELL (inv -=, cash +=, rebate +=)
  $0 maker fees. Residual inventory pays out at resolution (1 if NO won else 0).
  PnL = realized cash + rebate + residual payout.

Caveats: ignores queue competition (assumes we get our quote of the flow at our price
-> OPTIMISTIC); hourly granularity; per-tick quote cap. A first-order read of whether
the spread edge exists, not a live-MM guarantee.

Usage:  python analyze_spread_capture.py
"""
from __future__ import annotations
import json, urllib.request
from datetime import datetime, timezone
from pathlib import Path

GAMMA = "https://gamma-api.polymarket.com/markets"
DATA = "https://data-api.polymarket.com/trades"
LOG = Path("data/longshot_fade/signals.jsonl")
TRAJ = Path("data/longshot_fade/trajectory.jsonl")
CIDCACHE = Path("data/longshot_fade/cid_cache.json")
RESCACHE = Path("data/longshot_fade/maker_rescache.json")
SUMMARY = Path("data/longshot_fade/spread_capture_summary.json")
REBATE = 0.25
FEE = 0.05


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def condition_id(tok, cache):
    if tok in cache:
        return cache[tok]
    cid = None
    for q in ("&closed=true", "&closed=false", ""):
        try:
            g = get(f"{GAMMA}?clob_token_ids={tok}{q}")
        except Exception:
            g = None
        if g:
            cid = g[0].get("conditionId")
            break
    cache[tok] = cid
    return cid


def resolve_no(tok, cache):
    if tok in cache:
        return cache[tok]
    try:
        g = get(f"{GAMMA}?clob_token_ids={tok}&closed=true")
    except Exception:
        return None
    if not g:
        return None
    pr = g[0].get("outcomePrices")
    if isinstance(pr, str):
        try:
            pr = json.loads(pr)
        except Exception:
            return None
    if not pr or len(pr) < 2:
        return None
    cache[tok] = "no" if str(pr[1]) in ("1", "1.0") else "yes"
    return cache[tok]


def no_flow(cid, no_tok, yes_tok, entry):
    """All flow normalized to NO terms: list of (ts, side, no_price, size).
    side 'S' = sell-NO (fills our bid); 'B' = buy-NO (fills our ask)."""
    trades, off = [], 0
    while off < 10000:
        try:
            page = get(f"{DATA}?market={cid}&limit=500&offset={off}")
        except Exception:
            break
        if not page:
            break
        trades += page
        if min(t["timestamp"] for t in page) < entry:
            break
        off += 500
    out = []
    for t in trades:
        if t["timestamp"] < entry:
            continue
        a, side, p, sz, ts = (t.get("asset"), t.get("side"), t.get("price"),
                              t.get("size"), t["timestamp"])
        if a == no_tok:
            out.append((ts, "S" if side == "SELL" else "B", p, sz))
        elif yes_tok and a == yes_tok:
            # buy YES == sell NO ; sell YES == buy NO ; NO price = 1 - yes price
            out.append((ts, "S" if side == "BUY" else "B", 1.0 - p, sz))
    out.sort(key=lambda x: x[0])
    return out


def load_traj():
    traj = {}
    if not TRAJ.exists():
        return traj
    for l in TRAJ.read_text().splitlines():
        if not l.strip():
            continue
        try:
            t = json.loads(l)
        except Exception:
            continue
        sid = t.get("sig_id")
        if sid and t.get("no_bid") and t.get("no_ask"):
            traj.setdefault(sid, []).append(t)
    for sid in traj:
        traj[sid].sort(key=lambda x: x.get("ts", ""))
    return traj


def spread_sim(sig, flow, ticks, no_won):
    stake = sig.get("size_usd") or 5.0
    entry = datetime.fromisoformat(sig["ts"]).timestamp()
    path = [(entry, sig["no_best_bid"], sig.get("decision_quote"))]
    for t in ticks:
        try:
            ts = datetime.fromisoformat(t["ts"]).timestamp()
        except Exception:
            continue
        if ts > entry:
            path.append((ts, t["no_bid"], t["no_ask"]))
    path.append((float("inf"), None, None))
    fi = 0
    inv = cash = rebate = bought = sold = 0.0
    for k in range(len(path) - 1):
        ts_i, bid_i, ask_i = path[k]
        ts_next = path[k + 1][0]
        if not bid_i or not ask_i:
            continue
        svol = bvol = 0.0
        while fi < len(flow) and flow[fi][0] < ts_next:
            ts_f, side, p, sz = flow[fi]
            fi += 1
            if ts_f < ts_i:
                continue
            if side == "S" and p <= bid_i + 1e-9:
                svol += sz
            elif side == "B" and p >= ask_i - 1e-9:
                bvol += sz
        buy_fill = min(svol, stake / bid_i)     # we BUY NO at the bid
        sell_fill = min(bvol, stake / ask_i)    # we SELL NO at the ask
        inv += buy_fill
        cash -= buy_fill * bid_i
        rebate += REBATE * FEE * bid_i * (1 - bid_i) * buy_fill
        bought += buy_fill
        inv -= sell_fill
        cash += sell_fill * ask_i
        rebate += REBATE * FEE * ask_i * (1 - ask_i) * sell_fill
        sold += sell_fill
    payout = inv * (1.0 if no_won else 0.0)
    return {"pnl": cash + rebate + payout, "cash": cash, "rebate": rebate,
            "inv": inv, "payout": payout, "bought": bought, "sold": sold}


def main():
    recs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    seen = {}
    for r in recs:
        if r.get("no_best_bid") and r.get("decision_quote") and r.get("ts"):
            seen[(r.get("no_token_id"), r.get("target_date"))] = r
    sigs = list(seen.values())
    cidc = json.loads(CIDCACHE.read_text()) if CIDCACHE.exists() else {}
    resc = json.loads(RESCACHE.read_text()) if RESCACHE.exists() else {}
    traj = load_traj()
    today = datetime.now(timezone.utc).date()

    out = []
    for r in sigs:
        try:
            td = datetime.fromisoformat(r["target_date"] + "T00:00:00").date()
        except Exception:
            continue
        if td > today:
            continue
        tok = r["no_token_id"]
        no = resolve_no(tok, resc)
        if no is None:
            continue
        cid = condition_id(tok, cidc)
        if not cid:
            continue
        entry = datetime.fromisoformat(r["ts"]).timestamp()
        flow = no_flow(cid, tok, r.get("yes_token_id"), entry)
        res = spread_sim(r, flow, traj.get(r.get("sig_id"), []), no == "no")
        res["city"] = r.get("city")
        res["no_won"] = no == "no"
        out.append(res)
    CIDCACHE.write_text(json.dumps(cidc))
    RESCACHE.write_text(json.dumps(resc))

    n = len(out)
    if not n:
        print("no settled signals")
        return
    tot = sum(o["pnl"] for o in out)
    reb = sum(o["rebate"] for o in out)
    cash = sum(o["cash"] for o in out)
    pay = sum(o["payout"] for o in out)
    churn = sum(o["bought"] + o["sold"] for o in out)
    pos = sum(1 for o in out if o["pnl"] > 0)
    print(f"SPREAD-CAPTURE MM (first-order, ignores queue) | {n} settled signals")
    print(f"  total PnL ${tot:+.2f} = ${tot/n:+.3f}/signal   |   net-positive {pos}/{n} signals")
    print(f"  decomposition: realized cash ${cash:+.2f} + rebate ${reb:+.2f} + residual payout ${pay:+.2f}")
    print(f"  two-way churn {churn:.0f} sh filled; mean residual inv {sum(o['inv'] for o in out)/n:+.1f} sh/sig")
    SUMMARY.write_text(json.dumps({
        "n": n, "total": tot, "per_sig": tot / n, "cash": cash, "rebate": reb,
        "payout": pay, "churn": churn, "pos": pos,
        "generated": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  -> wrote {SUMMARY.name}")
    mf = Path("data/longshot_fade/maker_fill_summary.json")
    if mf.exists():
        m = json.loads(mf.read_text())
        print(f"\n  vs static maker ${m['maker_total']:+.2f} / dynamic "
              f"${m.get('dyn_maker_total', 0):+.2f} / taker ${m['taker_total']:+.2f}")


if __name__ == "__main__":
    main()
