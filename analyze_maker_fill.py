#!/usr/bin/env python3
"""Queue-aware maker-fill simulator for the longshot-fade forward test (Task #13).

The dashboard's "maker = rest at bid, assumes fill" line is an optimistic
placeholder: it assumes a resting NO buy always fills at the bid. This measures
whether it actually WOULD, using real per-market taker trade flow.

MODEL (FIFO, conservative, STATIC bid -- no re-quoting):
  At scan time we logged the NO best-bid B, the queue ahead Q (shares resting at
  B), and our size (size_usd/B shares). A resting BUY-NO at B fills only once
  taker SELL-NO volume at price <= B (after our entry ts) clears Q + our_size
  shares (FIFO: everyone ahead of us fills first). Taker sells at price > B mean
  the touch-bid rose above us -> we're left behind, no fill. We do NOT model
  chasing the market up; this is the static-bid floor on maker capture.

THE ADVERSE-SELECTION CHECK:
  A winning NO firms toward 1.0 -> sells trade above our bid -> we miss the win.
  A losing  NO falls toward 0.0 -> sells trade down through our bid -> we fill,
  then lose. If losers fill >> winners, the resting-bid maker is adversely
  selected and the "assumes fill" EV is illusory.

Caveat: the cohort quote CONTINUOUSLY (raise the bid as the market firms), which
our periodic snapshots can't replicate. So this is a lower bound = "what a naive
static maker captures", not the ceiling a live market-maker could reach.

Usage:  python analyze_maker_fill.py [--limit N] [--show N]
"""
from __future__ import annotations
import argparse, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path

GAMMA = "https://gamma-api.polymarket.com/markets"
DATA = "https://data-api.polymarket.com/trades"
LOG = Path("data/longshot_fade/signals.jsonl")
CIDCACHE = Path("data/longshot_fade/cid_cache.json")
RESCACHE = Path("data/longshot_fade/maker_rescache.json")
TRAJ = Path("data/longshot_fade/trajectory.jsonl")
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
    """'no'/'yes' for a closed market, None if not closed yet."""
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


def fetch_no_sells(cid, no_tok, yes_tok, entry):
    """Taker SELL-NO flow at/after entry, normalized across BOTH tokens:
    a NO-token SELL is selling NO directly; a YES-token BUY is buying YES == selling
    NO at price (1 - yes_price). These are distinct trades (the data reports each from
    the taker's chosen token, no double-count), so ~40% of sell-NO flow lives on the
    YES leg and was previously missed."""
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
        a = t.get("asset")
        if a == no_tok and t.get("side") == "SELL":
            out.append({"timestamp": t["timestamp"], "price": t["price"], "size": t["size"]})
        elif yes_tok and a == yes_tok and t.get("side") == "BUY":
            out.append({"timestamp": t["timestamp"], "price": 1.0 - t["price"], "size": t["size"]})
    return out


def sim_fill(sells, B, Q, our_sh):
    """FIFO: fill once cumulative SELL vol at price<=B clears Q+our_sh."""
    cum = 0.0
    for t in sorted(sells, key=lambda x: x["timestamp"]):
        if t["price"] <= B + 1e-9:
            cum += t["size"]
            if cum >= Q + our_sh:
                return True, t["timestamp"]
    return False, None


def load_trajectory():
    """sig_id -> time-sorted list of book ticks (no_bid, q_bid, ...)."""
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
        if sid and t.get("no_bid"):
            traj.setdefault(sid, []).append(t)
    for sid in traj:
        traj[sid].sort(key=lambda x: x.get("ts", ""))
    return traj


def dynamic_fill(sig, sells, ticks, ignore_queue=False):
    """Re-quoting maker: re-bid to the touch at each trajectory tick, FIFO fill.
    Returns (filled, fill_bid, fill_ts). Resets queue progress on each re-quote
    (we join the back of the new price level) and accumulates while the bid is
    unchanged. Uses per-tick q_bid if logged, else the entry queue as a proxy.
    With ignore_queue=True it's the upper bound (fill on the first sell at <= the
    current touch). Chasing UP catches winners static misses; chasing DOWN can
    duck the loser fills static eats."""
    B0 = sig["no_best_bid"]
    Q0 = sig["maker_queue_ahead"]
    stake = sig.get("size_usd") or 5.0
    entry = datetime.fromisoformat(sig["ts"]).timestamp()
    path = [(entry, B0, Q0)]
    for t in ticks:
        try:
            ts = datetime.fromisoformat(t["ts"]).timestamp()
        except Exception:
            continue
        if ts > entry and t.get("no_bid"):
            path.append((ts, t["no_bid"], t.get("q_bid")))
    path.append((float("inf"), None, None))
    ss = sorted(sells, key=lambda x: x["timestamp"])
    si = 0
    cum = 0.0
    last_bid = None
    for k in range(len(path) - 1):
        ts_i, bid_i, q_i = path[k]
        ts_next = path[k + 1][0]
        if not bid_i:
            continue
        if bid_i != last_bid:           # re-quoted -> re-join the new level's queue
            cum = 0.0
            last_bid = bid_i
        need = (0.0 if ignore_queue else (q_i if q_i is not None else Q0)) + stake / bid_i
        while si < len(ss) and ss[si]["timestamp"] < ts_next:
            s = ss[si]
            si += 1
            if s["timestamp"] < ts_i:
                continue
            if s["price"] <= bid_i + 1e-9:
                cum += s["size"]
                if cum >= need:
                    return True, bid_i, s["timestamp"]
    return False, None, None


def maker_pnl(B, won, stake):
    sh = stake / B
    per = ((1.0 - B) if won else -B) + REBATE * FEE * B * (1 - B)  # $0 taker fee + rebate
    return sh * per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    recs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    seen = {}
    for r in recs:
        if r.get("no_best_bid") and r.get("maker_queue_ahead") is not None and r.get("ts"):
            seen[(r.get("no_token_id"), r.get("target_date"))] = r
    sigs = list(seen.values())

    cidc = json.loads(CIDCACHE.read_text()) if CIDCACHE.exists() else {}
    resc = json.loads(RESCACHE.read_text()) if RESCACHE.exists() else {}
    today = datetime.now(timezone.utc).date()

    traj = load_trajectory()
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
            continue                       # not settled yet -> flow window incomplete
        cid = condition_id(tok, cidc)
        if not cid:
            continue
        B, Q = r["no_best_bid"], r["maker_queue_ahead"]
        stake = r.get("size_usd") or 5.0
        our_sh = stake / B
        entry = datetime.fromisoformat(r["ts"]).timestamp()
        sells = fetch_no_sells(cid, tok, r.get("yes_token_id"), entry)
        filled, fts = sim_fill(sells, B, Q, our_sh)
        tk = traj.get(r.get("sig_id"), [])
        dfilled, dbid, _ = dynamic_fill(r, sells, tk, ignore_queue=False)
        ufilled, _, _ = dynamic_fill(r, sells, tk, ignore_queue=True)
        out.append({
            "city": r.get("city"), "date": r.get("target_date"), "B": B, "Q": Q,
            "ask": r.get("decision_quote"), "no_won": no == "no", "filled": filled,
            "t2fill_h": ((fts - entry) / 3600) if fts else None,
            "sell_at_B": sum(t["size"] for t in sells if t["price"] <= B + 1e-9),
            "stake": stake, "dyn_filled": dfilled, "dyn_bid": dbid, "ub_filled": ufilled,
            "ticks": len(tk),
        })
        if args.limit and len(out) >= args.limit:
            break

    CIDCACHE.write_text(json.dumps(cidc))
    RESCACHE.write_text(json.dumps(resc))

    n = len(out)
    if not n:
        print("no settled signals with bid+queue yet")
        return
    filled = [o for o in out if o["filled"]]
    win = [o for o in out if o["no_won"]]
    los = [o for o in out if not o["no_won"]]
    fr = lambda L: (100 * sum(o["filled"] for o in L) / len(L)) if L else 0.0

    print(f"settled signals analysed: {n}\n")
    print(f"STATIC-MAKER FILL RATE: {100*len(filled)/n:.0f}%  ({len(filled)}/{n})")
    print(f"  by outcome | NO winners: {fr(win):.0f}% ({sum(o['filled'] for o in win)}/{len(win)})"
          f"   NO losers: {fr(los):.0f}% ({sum(o['filled'] for o in los)}/{len(los)})")
    print("  >> adverse selection if losers fill much more than winners\n")

    blended = sum(maker_pnl(o["B"], o["no_won"], o["stake"]) for o in filled)  # unfilled = $0
    print(f"MAKER (measured fills): total ${blended:+.2f} over {n} signals "
          f"= ${blended/n:+.3f}/signal (unfilled positions contribute $0)")
    if filled:
        fpnl = sum(maker_pnl(o["B"], o["no_won"], o["stake"]) for o in filled)
        print(f"  conditional on fill: {len(filled)} fills, "
              f"win {100*sum(o['no_won'] for o in filled)/len(filled):.0f}%, "
              f"${fpnl/len(filled):+.3f}/fill")
    # taker baseline on the same set: always fills at the ask
    tpnl = 0.0
    for o in out:
        a = o["ask"] or o["B"]
        sh = o["stake"] / a
        tpnl += sh * (((1.0 - a) if o["no_won"] else -a) - FEE * a * (1 - a))
    print(f"TAKER baseline (always fills at ask): total ${tpnl:+.2f} = ${tpnl/n:+.3f}/signal\n")

    # DYNAMIC re-quoting maker: re-bid to the touch at each trajectory tick (chase the market)
    dfil = [o for o in out if o["dyn_filled"]]
    dfr = lambda L: (100 * sum(o["dyn_filled"] for o in L) / len(L)) if L else 0.0
    dyn_total = sum(maker_pnl(o["dyn_bid"], o["no_won"], o["stake"]) for o in dfil)
    ub_rate = 100 * sum(o["ub_filled"] for o in out) / n
    med_ticks = sorted(o["ticks"] for o in out)[n // 2] if n else 0
    print(f"DYNAMIC re-quoting maker (chase the touch, median {med_ticks} ticks/sig): "
          f"fill {100*len(dfil)/n:.0f}% (win {dfr(win):.0f}% / lose {dfr(los):.0f}%)  "
          f"PnL ${dyn_total:+.2f} = ${dyn_total/n:+.3f}/sig")
    print(f"  upper bound (ignore queue): fill {ub_rate:.0f}%   |   "
          f"vs static ${blended:+.2f} / taker ${tpnl:+.2f}\n")

    summary = {
        "n": n, "fill_rate": 100 * len(filled) / n,
        "win_fill_rate": fr(win), "los_fill_rate": fr(los),
        "n_win": len(win), "n_los": len(los),
        "win_filled": sum(o["filled"] for o in win), "los_filled": sum(o["filled"] for o in los),
        "maker_total": blended, "maker_per_sig": blended / n,
        "taker_total": tpnl, "taker_per_sig": tpnl / n,
        "dyn_fill_rate": 100 * len(dfil) / n, "dyn_win_fill": dfr(win), "dyn_los_fill": dfr(los),
        "dyn_maker_total": dyn_total, "dyn_maker_per_sig": dyn_total / n, "ub_fill_rate": ub_rate,
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    Path("data/longshot_fade/maker_fill_summary.json").write_text(json.dumps(summary, indent=2))
    print("-> wrote data/longshot_fade/maker_fill_summary.json (dashboard reads this)\n")

    miss = [o for o in win if not o["filled"]]
    print(f"missed winners (NO won, maker bid never filled): {len(miss)}/{len(win)}")
    for o in sorted(miss, key=lambda x: -(x["ask"] or 0))[:args.show]:
        print(f"  MISS  {o['city']:11} {o['date']}  bid={o['B']:.2f} ask={o['ask']}  sell_at_bid={o['sell_at_B']:.0f}sh")
    badfills = [o for o in filled if not o["no_won"]]
    print(f"\nfilled losers (maker bid filled, then NO lost): {len(badfills)}")
    for o in badfills[:args.show]:
        print(f"  FILL->LOSS {o['city']:11} {o['date']}  bid={o['B']:.2f}  t2fill={o['t2fill_h']:.1f}h")


if __name__ == "__main__":
    main()
