#!/usr/bin/env python3
"""Longshot-fade live-test harness — DRY-RUN by default, zero money.

The forecast-free strategy located in this study: systematically taker-buy NO
on °C weather buckets priced 0.75-0.85 (the favorite-longshot bias sweet spot),
diversified across the strong basket cities, held to resolution.

This harness scans LIVE Polymarket books and measures the one thing a backtest
cannot: the fill-gap. For each qualifying bucket it logs the DECISION QUOTE
(NO best ask right now) and the DEPTH-WALKED fill for our size (from the live
order book) — so even before spending a cent we learn (a) whether the
opportunity actually exists live, (b) how many signals/day, (c) whether there's
depth to fill, and (d) the slippage from quote to fill.

Modes:
  (default)  DRY-RUN: scan + simulate fills off the live book + append to log.
  --live     submit real FAK NO orders. Needs Polymarket funds + py_clob_client_v2
             + POLY_* env (see execution/client.py). Flips ExecutionClient.dry_run
             -> from_env; nothing else changes. GATED, will not run without setup.
  --resolve  read the log, resolve matured markets, report realized win-rate / ROI
             vs the logged decision quote — the actual paper-vs-live answer.

Run dry-run (reusable; schedule it a few×/day to accumulate):
    "<repo>\\.venv\\Scripts\\python.exe" longshot_fade_harness.py
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from pathlib import Path

import httpx

from weather_bot.polymarket import (
    fetch_all_temperature_events, fetch_clob_prices_batch,
    fetch_orderbook_depths_batch, simulate_buy_fill,
    match_event_to_station, parse_bucket, event_target_date,
    GAMMA_BASE,
)
from weather_bot.fees import taker_fee_usd  # weather taker fee = shares·0.05·p·(1−p)

# ── Basket spec (from sharpen_basket.py) ──────────────────────────────────────
BAND_LO, BAND_HI = 0.75, 0.85       # NO entry-price sweet spot (+16-18pp, 96-99% win)
SIZE_USD = 5.0                       # per-bucket taker size (min-stake)
MAX_PER_CITY_PER_DAY = 4             # diversification cap
PEAK_HOUR = {"highest": 13, "lowest": 5}  # local hour before which the daily extreme isn't set yet
PAPER_BENCHMARK = "paper: ~97% win, ~+20% ROI/mkt — the bar live fills must clear"

# Strong, +edge, CLEAN-oracle °C cities (Station.name.lower()). Oracle-risky
# (shenzhen/istanbul/moscow/hong kong) and flat/negative (cape town/wuhan/
# madrid/milan/munich/mexico city) are EXCLUDED by virtue of the allowlist.
BASKET = {
    "guangzhou", "qingdao", "beijing", "shanghai", "tokyo", "taipei", "busan",
    "seoul", "chengdu", "paris", "helsinki", "warsaw", "london", "wellington",
    "lucknow", "kuala lumpur", "singapore", "buenos aires", "sao paulo",
    "amsterdam", "panama city",
}

LOG = Path("data/longshot_fade/signals.jsonl")
TRAJ = Path("data/longshot_fade/trajectory.jsonl")  # per-open-position NO bid/ask path (resell sim)
MAKER_REBATE_FRAC = 0.25  # 25% of taker fee redistributed to makers (fees.py)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_seen():
    """(no_token_id, target_date) already logged — avoid double-counting re-scans."""
    seen = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r["no_token_id"], r["target_date"]))
            except Exception:
                continue
    return seen


def _append(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


async def scan(live: bool):
    seen = _load_seen()
    per_city_day = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        events = await fetch_all_temperature_events(client)
        # Candidate buckets in basket °C cities with a NO token
        cand = []
        for ev in events:
            st = match_event_to_station(ev)
            if st is None or st.unit != "C" or st.name.lower() not in BASKET:
                continue
            for m in ev.markets:
                if m.no_token_id:
                    cand.append((st, ev, m))
        print(f"{len(events)} weather events, {len(cand)} basket-°C buckets to price", file=sys.stderr)

        # Refresh YES prices, pre-filter to NO≈1-yes_bid near the band (cheap),
        # then fetch real NO order books only for those.
        fresh = await fetch_clob_prices_batch([m.yes_token_id for _, _, m in cand if m.yes_token_id], client)
        pre = []
        for st, ev, m in cand:
            yb, _ = fresh.get(m.yes_token_id, (m.yes_bid, m.yes_ask))
            if yb is None:
                continue
            if BAND_LO - 0.06 <= (1.0 - yb) <= BAND_HI + 0.06:
                pre.append((st, ev, m))
        books = await fetch_orderbook_depths_batch([m.no_token_id for _, _, m in pre], client)

        live_client = None
        if live:
            live_client = _build_live_client()

        emitted = 0
        n_not_forward = 0
        for st, ev, m in pre:
            book = books.get(m.no_token_id)
            if book is None or book.best_ask is None:
                continue
            no_ask = book.best_ask
            if not (BAND_LO <= no_ask <= BAND_HI):
                continue
            td = event_target_date(ev, st)
            # Forward-only, TAGGED so we can compare two pre-outcome windows:
            #   next_day    = target day is a future local day (pure forward)
            #   same_day_am = target day is today local, before the daily peak (still
            #                 pre-outcome). Skip post-peak / past-day (that's lookahead).
            local_now = datetime.now(ZoneInfo(st.timezone))
            if td > local_now.date():
                timing = "next_day"
            elif td == local_now.date() and local_now.hour < PEAK_HOUR.get(st_target(ev), 12):
                timing = "same_day_am"
            else:
                n_not_forward += 1
                continue
            key = (m.no_token_id, td.isoformat())
            if key in seen:
                continue
            ck = (st.name, td.isoformat())
            if per_city_day.get(ck, 0) >= MAX_PER_CITY_PER_DAY:
                continue
            kind, thr = parse_bucket(m)

            # Depth-walked dry-run fill for our size, capped at the band ceiling.
            sim = simulate_buy_fill(book, SIZE_USD, BAND_HI, min_order_size=book.min_order_size)
            sim_fill, sim_sh, sim_full = (sim if sim else (None, 0.0, False))

            rec = {
                "ts": _now_iso(), "mode": "live" if live else "dry",
                "sig_id": f"{m.no_token_id}|{td.isoformat()}",
                "city": st.name, "station_id": st.station_id, "target": st_target(ev),
                "timing": timing,
                "target_date": td.isoformat(), "bucket_label": m.bucket_label,
                "kind": kind, "threshold": thr, "no_token_id": m.no_token_id,
                "yes_token_id": m.yes_token_id, "event_slug": ev.slug,
                "decision_quote": round(no_ask, 4),
                "no_best_bid": round(book.best_bid, 4) if book.best_bid else None,
                "maker_queue_ahead": round(book.bids[0].size_shares, 1) if book.bids else None,
                "no_spread": round(book.spread, 4) if book.spread is not None else None,
                "sim_avg_fill": round(sim_fill, 4) if sim_fill else None,
                "sim_shares": round(sim_sh, 2), "sim_fully_filled": sim_full,
                "fill_gap": round(sim_fill - no_ask, 4) if sim_fill else None,
                "taker_fee": round(taker_fee_usd(sim_sh, sim_fill), 5) if sim_fill else 0.0,
                "size_usd": SIZE_USD,
            }

            if live and live_client is not None:
                rec.update(_submit_live(live_client, st, ev, m, no_ask, kind, thr, td))

            _append(rec)
            per_city_day[ck] = per_city_day.get(ck, 0) + 1
            seen.add(key)
            emitted += 1
            gap = rec["fill_gap"]
            print(f"  {st.name:14} {m.bucket_label:16} NO ask ${no_ask:.3f}  "
                  f"sim_fill ${rec['sim_avg_fill'] or 0:.3f} (gap {gap:+.3f}, "
                  f"{'full' if sim_full else 'PARTIAL'} {sim_sh:.1f}sh)")

        print(f"\n{emitted} new signals logged -> {LOG}  [{'LIVE' if live else 'DRY-RUN'}]")
        print(f"  (skipped {n_not_forward} non-forward: target day already started in city-local time)")
        print(f"{PAPER_BENCHMARK}")


def st_target(ev):
    return "highest" if ev.target == "highest" else "lowest"


def _build_live_client():
    """Construct the real ExecutionClient. Verify TradingConfig fields at funding time."""
    from weather_bot.execution.client import ExecutionClient
    from weather_bot.execution.safety import TradingConfig
    cfg = TradingConfig()  # defaults; tune caps/bankroll when funding
    client = ExecutionClient.from_env(cfg)
    bal = client.get_balance_usdc()
    print(f"[LIVE] ExecutionClient ready. USDC balance: {bal}", file=sys.stderr)
    return client


def _submit_live(client, st, ev, m, no_ask, kind, thr, td):
    """Submit a real FAK NO taker order. Returns dict of live-fill fields."""
    from weather_bot.scanner import TradeSignal
    sig = TradeSignal(
        station=st, event_title=ev.title, event_slug=ev.slug, target=st_target(ev),
        target_date=td, bucket_label=m.bucket_label, bucket_kind=kind,
        market_id=m.market_id, token_id=m.no_token_id, our_prob=0.0,
        yes_implied=m.yes_implied or 0.0, yes_bid=m.yes_bid, yes_ask=m.yes_ask,
        side="NO", edge=0.0, fill_price=no_ask,
        volume_24hr=ev.volume_24hr, bias_applied_c=0.0, sigma_ensemble_c=0.0,
        sigma_total_c=0.0, kelly_full=0.0, position_usd=SIZE_USD,
    )
    res = client.submit_order(sig, order_type="FAK", sdk_side="BUY", limit_price=BAND_HI)
    return {"live_ok": res.ok, "live_order_id": res.order_id,
            "live_fill_price": res.fill_price, "live_shares": res.shares,
            "live_msg": res.message}


async def resolve():
    """Resolve matured logged signals via gamma (by NO token) and report
    realized win-rate / ROI vs the logged decision quote."""
    if not LOG.exists():
        print("no signals logged yet"); return
    recs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    today = datetime.now(timezone.utc).date()
    # target_date <= today: Asia/EU markets close mid-UTC-day (hours before UTC
    # midnight), so include same-day targets. closed=true below is the real gate —
    # markets not yet closed return empty and are skipped, never mis-resolved.
    matured = [r for r in recs if date.fromisoformat(r["target_date"]) <= today]
    print(f"{len(recs)} logged, {len(matured)} reached target day (<= today UTC)")
    settled = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for r in matured:
            try:
                g = await client.get(f"{GAMMA_BASE}/markets",
                                     params={"clob_token_ids": r["no_token_id"], "closed": "true"})
                g.raise_for_status(); arr = g.json()
            except Exception:
                continue
            if not arr:
                continue
            mkt = arr[0]
            prices = mkt.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if not prices:
                continue
            # outcomes order = [Yes, No]; NO won if outcomePrices[1] == "1"
            no_won = str(prices[1]) in ("1", "1.0")
            entry = r["decision_quote"]
            pnl = ((1.0 - entry) if no_won else -entry)
            settled.append((r, no_won, pnl))
    if not settled:
        print("none resolvable yet"); return
    n = len(settled); wins = sum(1 for _, w, _ in settled if w)
    tot_pnl = sum(p for *_, p in settled)
    avg_entry = sum(r["decision_quote"] for r, _, _ in settled) / n
    gaps = [r["fill_gap"] for r, _, _ in settled if r.get("fill_gap") is not None]
    print(f"\nSETTLED {n} bets | win {100*wins/n:.1f}%  avg entry {avg_entry:.3f}  "
          f"realized ROI {100*tot_pnl/sum(r['decision_quote'] for r,_,_ in settled):+.1f}%")
    if gaps:
        print(f"fill-gap (sim_fill − quote): mean {sum(gaps)/len(gaps):+.4f}  "
              f"max {max(gaps):+.4f}  (>0 = paid up from quote)")
    print(f"vs {PAPER_BENCHMARK}")


async def track_open():
    """Re-fetch the NO book for every still-open logged signal and append a bid/ask
    tick to trajectory.jsonl — builds the price path used to simulate resell exits.
    Runs automatically after each scan."""
    if not LOG.exists():
        return
    today = datetime.now(timezone.utc).date()
    open_sigs = {}
    for line in LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        sid = r.get("sig_id")
        if not sid:
            continue
        try:
            if date.fromisoformat(r["target_date"]) < today:   # matured -> stop tracking
                continue
        except Exception:
            continue
        open_sigs[sid] = r["no_token_id"]
    if not open_sigs:
        print("no open positions to track", file=sys.stderr)
        return
    async with httpx.AsyncClient(timeout=30.0) as client:
        books = await fetch_orderbook_depths_batch(list(set(open_sigs.values())), client)
    ts = _now_iso()
    n = 0
    TRAJ.parent.mkdir(parents=True, exist_ok=True)
    with TRAJ.open("a") as f:
        for sid, tok in open_sigs.items():
            b = books.get(tok)
            if b is None:
                continue
            qb = round(b.bids[0].size_shares, 1) if getattr(b, "bids", None) else None
            qa = round(b.asks[0].size_shares, 1) if getattr(b, "asks", None) else None
            f.write(json.dumps({"sig_id": sid, "ts": ts,
                                "no_bid": round(b.best_bid, 4) if b.best_bid else None,
                                "no_ask": round(b.best_ask, 4) if b.best_ask else None,
                                "q_bid": qb, "q_ask": qa}) + "\n")
            n += 1
    print(f"tracked {n} open positions -> {TRAJ}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Longshot-fade live-test harness (dry-run default)")
    ap.add_argument("--live", action="store_true", help="submit real orders (needs funds+SDK+env)")
    ap.add_argument("--resolve", action="store_true", help="score matured logged signals")
    ap.add_argument("--track-only", action="store_true", help="only update open-position trajectories")
    args = ap.parse_args()
    if args.resolve:
        asyncio.run(resolve())
    elif args.track_only:
        asyncio.run(track_open())
    else:
        if args.live:
            print("!! --live: real money. Ctrl-C now if unintended.", file=sys.stderr)

        async def _run():
            await scan(live=args.live)
            await track_open()

        asyncio.run(_run())


if __name__ == "__main__":
    main()
