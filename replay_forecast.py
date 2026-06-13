#!/usr/bin/env python3
"""Forecast replay — does OUR ensemble reproduce the cohort's NO-fade selectivity?

For each resolved °C cohort NO-fade market, reconstruct our forecast AS-OF
forecast time (genuine historical, NO lookahead — forces the historical-forecast
API) -> multi-model Gaussian -> bucket probability (reusing the bot's debugged
probability.py) -> our NO-side edge. Then test whether OUR edge separates the
cohort's winning NO-fades from the losing ones (the selectivity that distinguishes
badatmath +11% from poligarch +0.5%).

Run with the project venv (has httpx/numpy/scipy/pandas + the weather_bot package
is importable from cwd):
    "<repo>\\.venv\\Scripts\\python.exe" replay_forecast.py
"""
from __future__ import annotations
import argparse, asyncio, json, re, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import httpx

from weather_bot.forecast.probability import TempDistribution, bucket_prob
from weather_bot.polymarket import parse_bucket
from weather_bot.locations import STATION_BY_CITY

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
SIGMA_BASE = 1.3  # °C irreducible ~1-day forecast error beyond model disagreement
SUBSTRATE = "data/poligarch/cohort_substrate.json"
FC_CACHE = Path("data/poligarch/forecast_cache.json")
GEO_CACHE = Path("data/poligarch/geo_cache.json")

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def parse_date(s: str) -> date:
    mon, day, yr = s.split("-")
    return date(int(yr), _MONTHS[mon.lower()], int(day))


def parse_bucket_from_title(title: str):
    """(kind, threshold, unit) from a market title, reusing the bot's parse_bucket.
    Returns None if not parseable or not a recognisable bucket."""
    low = title.lower()
    unit = "C" if "°c" in low else ("F" if "°f" in low else None)
    if unit is None:
        return None
    m = re.search(r"\bbe\s+(.+?)\s+on\b", title, re.I)
    if not m:
        return None
    shim = SimpleNamespace(bucket_label=m.group(1), threshold_index=0)
    kind, thr = parse_bucket(shim)
    return kind, thr, unit


async def _get(client, url, params, tries=4):
    backoff = 1.0
    for i in range(tries):
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        await asyncio.sleep(backoff)
        backoff *= 2


async def resolve_location(slug, client, cache):
    if slug in cache:
        return cache[slug]
    name = slug.replace("-", " ")
    st = STATION_BY_CITY.get(name)
    if st:
        loc = {"lat": st.latitude, "lon": st.longitude, "tz": st.timezone, "src": "registry"}
    else:
        try:
            d = await _get(client, GEO_URL, {"name": name, "count": 1, "language": "en"})
            res = (d or {}).get("results") or []
            if res:
                r = res[0]
                loc = {"lat": r["latitude"], "lon": r["longitude"],
                       "tz": r.get("timezone", "UTC"), "src": "geocode"}
            else:
                loc = None
        except Exception:
            loc = None
    cache[slug] = loc
    return loc


async def fetch_models_hist(loc, d: date, client):
    """Genuine historical multi-model deterministic max+min (no lookahead)."""
    params = {
        "latitude": loc["lat"], "longitude": loc["lon"],
        "start_date": d.isoformat(), "end_date": d.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": ",".join(MODELS), "timezone": loc["tz"], "temperature_unit": "celsius",
    }
    data = await _get(client, HIST_URL, params)
    daily = (data or {}).get("daily", {})
    out = {"max": [], "min": []}
    for mdl in MODELS:
        for agg in ("max", "min"):
            vals = daily.get(f"temperature_2m_{agg}_{mdl}", [])
            if vals and vals[0] is not None:
                out[agg].append(float(vals[0]))
    return out


def dist_from_points(points, target_date, seed=0, n=4000):
    pts = np.array([p for p in points if p == p], dtype=float)
    if len(pts) == 0:
        return None
    mu = float(pts.mean())
    spread = float(pts.std(ddof=1)) if len(pts) > 1 else 0.0
    sigma = (spread ** 2 + SIGMA_BASE ** 2) ** 0.5
    members = np.random.default_rng(seed).normal(mu, sigma, n)
    return TempDistribution(location_name="", target_date=target_date, members=members)


def roi_stats(rows):
    """rows: list of (entry_price, won_bool). Returns (n, win%, mean ROI per $)."""
    if not rows:
        return (0, 0.0, 0.0)
    n = len(rows)
    win = sum(1 for _, w in rows if w) / n
    roi = sum(((1.0 if w else 0.0) - p) / p for p, w in rows) / n
    return (n, 100 * win, 100 * roi)


def auc(vals, wins):
    """AUC for 'higher val -> more likely won'. 0.5 = no skill."""
    v = np.asarray(vals, float); w = np.asarray(wins, bool)
    pos, neg = v[w], v[~w]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


async def main(args):
    global SIGMA_BASE
    SIGMA_BASE = args.sigma
    sub = json.loads(Path(SUBSTRATE).read_text())
    fc_cache = json.loads(FC_CACHE.read_text()) if FC_CACHE.exists() else {}
    geo_cache = json.loads(GEO_CACHE.read_text()) if GEO_CACHE.exists() else {}

    # Dedup cohort NO bets -> unique markets. won (for outcome=No) == "did NO win".
    no_mkts = {}
    n_c, n_f, n_skip = 0, 0, 0
    for r in sub:
        if r["outcome"] != "No":
            continue
        pb = parse_bucket_from_title(r["title"])
        if pb is None:
            n_skip += 1; continue
        kind, thr, unit = pb
        if unit == "F":
            n_f += 1; continue
        n_c += 1
        cid = r["condition_id"]
        e = no_mkts.get(cid)
        if e is None:
            e = no_mkts[cid] = {"city": r["city"], "date": r["date"], "target": r["target"],
                                "kind": kind, "thr": thr, "won": r["won"], "vwaps": []}
        e["vwaps"].append(r["vwap"])
    print(f"NO bets: {n_c} °C rows, {n_f} °F rows skipped, {n_skip} unparseable")
    print(f"Unique °C NO markets: {len(no_mkts)}")

    # Resolve locations + fetch forecasts for unique (city,date)
    async with httpx.AsyncClient(timeout=60.0) as client:
        cities = sorted({m["city"] for m in no_mkts.values()})
        for slug in cities:
            await resolve_location(slug, client, geo_cache)
        GEO_CACHE.write_text(json.dumps(geo_cache))
        unresolved = [c for c in cities if not geo_cache.get(c)]
        print(f"Cities: {len(cities)} ({sum(1 for c in cities if geo_cache.get(c) and geo_cache[c]['src']=='registry')} registry, "
              f"{sum(1 for c in cities if geo_cache.get(c) and geo_cache[c]['src']=='geocode')} geocoded); unresolved {unresolved}")

        keys = sorted({(m["city"], m["date"]) for m in no_mkts.values() if geo_cache.get(m["city"])})
        sem = asyncio.Semaphore(5)
        miss = [k for k in keys if f"{k[0]}|{k[1]}" not in fc_cache]
        print(f"Forecast fetches: {len(keys)} unique (city,date); {len(miss)} to fetch, {len(keys)-len(miss)} cached")

        async def one(slug, dstr):
            async with sem:
                try:
                    pts = await fetch_models_hist(geo_cache[slug], parse_date(dstr), client)
                    fc_cache[f"{slug}|{dstr}"] = pts
                except Exception as e:
                    fc_cache[f"{slug}|{dstr}"] = {"err": str(e)[:80]}
        await asyncio.gather(*(one(c, d) for c, d in miss))
        FC_CACHE.write_text(json.dumps(fc_cache))

    # Score
    scored = []  # (cid, entry_price, won_no, our_p_no, our_edge, city, kind)
    no_fc = 0
    for cid, m in no_mkts.items():
        fc = fc_cache.get(f"{m['city']}|{m['date']}")
        if not fc or "err" in fc:
            no_fc += 1; continue
        agg = "max" if m["target"] == "highest" else "min"
        pts = fc.get(agg, [])
        dist = dist_from_points(pts, parse_date(m["date"]))
        if dist is None or dist.n_members == 0:
            no_fc += 1; continue
        p_bucket = bucket_prob(dist, m["kind"], m["thr"], unit="C")
        p_no = 1.0 - p_bucket
        entry = float(np.mean(m["vwaps"]))
        edge = p_no - entry
        scored.append((cid, entry, bool(m["won"]), p_no, edge, m["city"], m["kind"]))
    print(f"Scored {len(scored)} markets ({no_fc} had no usable forecast)\n")

    # ---- Forecast skill (Brier on the bucket-happens event) ----
    briers = [((1.0 - p_no) - (0.0 if won else 1.0)) ** 2 for _, _, won, p_no, _, _, _ in scored]
    base_rate = np.mean([0.0 if won else 1.0 for _, _, won, *_ in scored])  # P(bucket happens)
    brier_base = base_rate * (1 - base_rate)
    print(f"Forecast Brier (bucket-happens): {np.mean(briers):.4f}  vs base-rate {brier_base:.4f}  "
          f"(skill {1 - np.mean(briers)/brier_base:+.1%})")

    # ---- Separation: our P(NO wins) on cohort winners vs losers ----
    pno_win = [p for _, _, won, p, _, _, _ in scored if won]
    pno_lose = [p for _, _, won, p, _, _, _ in scored if not won]
    print(f"Our P(NO wins): on cohort WINNERS {np.mean(pno_win):.3f} (n={len(pno_win)})  "
          f"vs LOSERS {np.mean(pno_lose):.3f} (n={len(pno_lose)})  "
          f"gap {np.mean(pno_win)-np.mean(pno_lose):+.3f}")
    aw, al = np.array(pno_win), np.array(pno_lose)
    allv = np.concatenate([aw, al]); order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    auc_all = (ranks[:len(aw)].sum() - len(aw) * (len(aw) + 1) / 2) / (len(aw) * len(al))
    print(f"Our P(NO) AUC (winner>loser ranking, all markets): {auc_all:.3f}  (0.5 = no skill)")

    # ---- Selectivity test on the NO-fade band [0.6, 0.9] ----
    band = [(p, won, edge, pno) for _, p, won, pno, edge, _, _ in scored if 0.6 <= p < 0.9]
    prices = [b[0] for b in band]; wins = [b[1] for b in band]
    edges = [b[2] for b in band]; pnos = [b[3] for b in band]
    nlose = len(band) - sum(wins)
    base = roi_stats([(p, w) for p, w, _, _ in band])
    print(f"\nNO-fade band [0.60,0.90): {len(band)} markets ({sum(wins)} win / {nlose} lose), "
          f"baseline win {base[1]:.1f}% ROI {base[2]:+.1f}%")
    # THE decisive number: does our forecast rank winners better than the PRICE already does?
    print(f"  AUC(predict NO-win):   market PRICE = {auc(prices, wins):.3f}   "
          f"our P_no = {auc(pnos, wins):.3f}   edge=P_no-price = {auc(edges, wins):.3f}")
    print(f"  -> if edge-AUC ~0.5, our forecast adds nothing beyond the price the cohort already trades on")
    # Ranking-drop: cut the q% worst NO-fades by our edge — do we catch losers + lift ROI?
    order = sorted(range(len(band)), key=lambda i: edges[i])  # ascending: worst edge first
    for q in (0.10, 0.20, 0.30):
        k = int(len(band) * q); dropped = set(order[:k])
        kept = [(prices[i], wins[i]) for i in range(len(band)) if i not in dropped]
        ld = sum(1 for i in order[:k] if not wins[i])
        ks = roi_stats(kept)
        print(f"  drop worst {int(q*100)}% by our edge (n={k}): kept ROI {ks[2]:+.1f}% win {ks[1]:.1f}%  | "
              f"losers caught {ld}/{nlose} ({100*ld/max(nlose,1):.0f}%, vs {int(q*100)}% if random)")
    losers = [(p, e) for p, w, e, _ in band if not w]

    k0 = roi_stats([(p, w) for p, w, e, _ in band if e > 0])
    s0 = roi_stats([(p, w) for p, w, e, _ in band if e <= 0])
    lc0 = sum(1 for p, e in losers if e <= 0)
    print(f"SUMMARY sigma={SIGMA_BASE} brier_skill={1-np.mean(briers)/brier_base:+.1%} "
          f"pno_auc={auc_all:.3f} edge_auc={auc(edges, wins):.3f} base_roi={base[2]:+.1f}% "
          f"keep>0_roi={k0[2]:+.1f}% skip>0_roi={s0[2]:+.1f}% losers_caught={lc0}/{len(losers)}")

    Path("data/poligarch/replay_scored.json").write_text(json.dumps(
        [{"cid": c, "entry": p, "won_no": w, "our_p_no": pn, "edge": e, "city": ct, "kind": k}
         for c, p, w, pn, e, ct, k in scored]))
    print(f"\nSaved {len(scored)} scored markets -> data/poligarch/replay_scored.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.3)
    asyncio.run(main(ap.parse_args()))
