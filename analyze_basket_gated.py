"""Gated-stack basket backtest — does firing only the GOOD fires beat the spread?

Applies the parked preventive gates (individually + stacked) to the realized
consensus_basket P&L and reports the selected-subset result. The one tool to run
at the N-gate to decide "fire smarter" vs "retire".

Gates (edit the CONFIG block or pass nothing = defaults):
  time-gate    : skip MAX-target fires before MIN_LOCAL_HOUR (station-local)
  region-cap   : keep only the first REGION_CAP_K same-region fires per UTC day
  kill-2-leg   : drop favorite-YES + single-NO baskets (the no-offset shape)
  trigger-band : fire only when favorite fill price in [TRIG_LO, TRIG_HI]
  nowcast/obs  : skip fires where the obs-max-so-far is already past the favorite
                 bucket's kill boundary (observed_extreme_c; only from 2026-06-02)

Each gate is reported ALONE (marginal Δ vs baseline) and STACKED. Realized P&L
comes from slim_dashboard.compute_positions (actual fills + WUG resolution).

FIRST READ (2026-06-05, 238 clean resolved fires, N≈6 days — small, confirm at N):
  baseline (all fires) = −$125.  Gate marginals:
    trigger-band .88-.94  Δ +$161  → +$36, 98% win   ← STRONGEST; flips positive
    time-gate ≥16h local  Δ +$89   → −$36, 84% win
    region-cap K=3        Δ +$52   → −$73, 83% win
    kill-2-leg            Δ −$44   ← COUNTERPRODUCTIVE: 2-leg baskets are net-
                                     POSITIVE; killing them drops wins (overturns
                                     the parked lever → default DROP_2LEG=False).
    nowcast/obs           Δ −$2    ← ~0 effect yet (obs_extreme only from 06-02).
  STACKED (time≥16 + region K=3 + trig .88-.94, no 2-leg) = +$11.35, 100% win,
  but only 12 fires (5%) → "fire near-certain favorites", thin (+$0.88/fire).
  Takeaway: gates RECOVER the bleed and a narrow high-confidence subset goes
  +EV — NOT retire — but it's thin; the decisive "every leg loses" was the
  UN-gated average. Re-run at N=14/30; tune the CONFIG below.

Run (VPS venv):  python analyze_basket_gated.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import slim_dashboard as sd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.portfolio import region_for
from weather_bot.pnl import _rounded_observation, bucket_won

# ─────────────────────────── GATE CONFIG (edit me) ───────────────────────────
MIN_LOCAL_HOUR = 16       # time-gate: skip MAX fires before this local hour
REGION_CAP_K = 3          # region-cap: keep first K same-region fires / UTC day
DROP_2LEG = False         # kill 2-leg no-offset baskets — DATA SAYS COUNTERPRODUCTIVE
                          # (2-leg baskets are net-positive; killing them costs ~−$44)
TRIG_LO, TRIG_HI = 0.00, 1.00   # trigger band (0,1 = off). +EV pockets: .73-.76 / .88-.94
NOWCAST_SKIP = True       # skip fires whose obs-max already passed the favorite bucket
EXCL = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM", "WIHH"}
# ─────────────────────────────────────────────────────────────────────────────

DATA = Path("data")
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


def local_hour(ts_iso, sid):
    """Station-local hour of a UTC ISO timestamp. tz name → ZoneInfo; else solar
    approx from longitude."""
    st = STATIONS_BY_ID.get(sid)
    try:
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    tzname = getattr(st, "timezone", None)
    if ZoneInfo is not None and tzname:
        try:
            return dt.astimezone(ZoneInfo(tzname)).hour
        except Exception:
            pass
    lon = getattr(st, "longitude", None)
    if lon is not None:
        return int((dt.hour + lon / 15.0) % 24)
    return None


def load(p):
    out = []
    if not (DATA / p).exists():
        return out
    for l in (DATA / p).open(encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def unit_for(sid):
    return getattr(STATIONS_BY_ID.get(sid), "unit", None) or "C"


def obs_margin(obs_c, target, kind, thr, unit):
    """° headroom from obs-max-so-far to the favorite bucket's kill boundary
    (mid buckets). <0 = obs already past the favorite → it's busting."""
    if obs_c is None or kind != "mid" or thr is None:
        return None
    o = float(obs_c) if unit == "C" else float(obs_c) * 9.0 / 5.0 + 32.0
    t = int(thr)
    if target == "max":
        return (t + 0.5) - o
    if target == "min":
        return o - (t - 0.5)
    return None


def build_events():
    resmap = sd.resolution_map()
    # realized net per (sk)
    net = defaultdict(float)
    resolved = set()
    for p in sd.compute_positions(resmap):
        if p["strategy"] != "consensus_basket":
            continue
        k = (p["station"], p["target"], p["date"])
        if p["status"] == "resolved":
            net[k] += p["net"]; resolved.add(k)
    # fire features from the basket log
    fire = defaultdict(lambda: {"ts": "", "fav": None, "yes": 0, "no": 0,
                                "kind": None, "thr": None})
    for r in load("consensus_basket_log.jsonl"):
        if r.get("result") != "filled":
            continue
        k = (r["station_id"], r["target"], r["target_date"])
        e = fire[k]
        if r["side"] == "YES":
            e["yes"] += 1; e["fav"] = r.get("fill_price")
            e["kind"] = r.get("bucket_kind"); e["thr"] = r.get("bucket_threshold")
        else:
            e["no"] += 1
        if not e["ts"] or (r.get("ts_utc") or "") < e["ts"]:
            e["ts"] = r.get("ts_utc") or ""
    # obs_extreme at fire from the sweep
    sw = defaultdict(list)
    for r in load("basket_sweep_log.jsonl"):
        if r.get("winner"):
            sw[(r["station_id"], r["target"], r["target_date"])].append(r)
    obsf = {}
    for k, rs in sw.items():
        rs.sort(key=lambda r: r.get("ts_utc") or "")
        f = next((r for r in rs if float(r.get("leader_yes_ask") or 0) >= 0.70), None)
        if f:
            obsf[k] = f.get("observed_extreme_c")

    evs = []
    for k in resolved:
        sid, tgt, date = k
        if sid in EXCL:
            continue
        e = fire.get(k)
        if not e or e["fav"] is None:
            continue
        u = unit_for(sid)
        ai = _rounded_observation(resmap[k], u)
        evs.append({
            "sk": k, "sid": sid, "target": tgt, "date": date,
            "net": net[k], "fav": float(e["fav"]),
            "legs": e["yes"] + e["no"], "is2leg": (e["yes"] == 1 and e["no"] == 1),
            "lhour": local_hour(e["ts"], sid), "region": region_for(sid),
            "ts": e["ts"],
            "obs_margin": obs_margin(obsf.get(k), tgt, e["kind"], e["thr"], u),
        })
    # region concurrency rank (same region + UTC day, ordered by fire ts)
    grp = defaultdict(list)
    for e in evs:
        grp[(e["region"], e["date"])].append(e)
    for g in grp.values():
        g.sort(key=lambda e: e["ts"])
        for i, e in enumerate(g):
            e["region_rank"] = i
    return evs


# ── gates ──
def g_time(e):    return not (e["target"] == "max" and (e["lhour"] is not None) and e["lhour"] < MIN_LOCAL_HOUR)
def g_region(e):  return e["region_rank"] < REGION_CAP_K
def g_2leg(e):    return not (DROP_2LEG and e["is2leg"])
def g_trig(e):    return TRIG_LO <= e["fav"] <= TRIG_HI
def g_now(e):     return not (NOWCAST_SKIP and e["obs_margin"] is not None and e["obs_margin"] < 0)

GATES = [("time-gate", g_time), ("region-cap", g_region), ("kill-2-leg", g_2leg),
         ("trigger-band", g_trig), ("nowcast/obs", g_now)]


def summ(evs):
    n = len(evs)
    if not n:
        return (0, 0.0, 0.0, 0.0)
    tot = sum(e["net"] for e in evs)
    wins = sum(1 for e in evs if e["net"] > 0)
    return (n, tot, tot / n, 100 * wins / n)


def main():
    evs = build_events()
    print("=" * 78)
    print("GATED-STACK BASKET BACKTEST  (realized net, clean stations)")
    print("config: time>=%dh local · region-cap K=%d · drop2leg=%s · trig[%.2f,%.2f] · nowcast=%s"
          % (MIN_LOCAL_HOUR, REGION_CAP_K, DROP_2LEG, TRIG_LO, TRIG_HI, NOWCAST_SKIP))
    print("=" * 78)
    n, tot, avg, wr = summ(evs)
    print("BASELINE (all fires):   n=%3d  net=$%+8.2f  avg=$%+.3f  win=%2.0f%%" % (n, tot, avg, wr))
    print("\nEACH GATE ALONE (marginal):")
    for name, fn in GATES:
        kept = [e for e in evs if fn(e)]
        kn, kt, ka, kw = summ(kept)
        print("  %-13s keep %3d/%-3d  net=$%+8.2f  Δ=$%+7.2f  avg=$%+.3f  win=%2.0f%%"
              % (name, kn, n, kt, kt - tot, ka, kw))
    stacked = [e for e in evs if all(fn(e) for _, fn in GATES)]
    sn, st_, sa, sw = summ(stacked)
    print("\nFULL STACK:             n=%3d  net=$%+8.2f  Δ=$%+7.2f  avg=$%+.3f  win=%2.0f%%"
          % (sn, st_, st_ - tot, sa, sw))
    print("  → fires kept %d of %d (%.0f%%); P&L %s baseline by $%+.2f"
          % (sn, n, 100 * sn / n if n else 0, "beats" if st_ > tot else "trails", st_ - tot))
    print("\nNote: realized fills + WUG resolution. region-cap keeps the first-K")
    print("  chronological fires/region/day (a later 'keep-highest-confidence' rule")
    print("  could differ). nowcast/obs only covers fires from 2026-06-02 (field onset).")


if __name__ == "__main__":
    main()
