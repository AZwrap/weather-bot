"""Off-by-one bust DETECTOR — is the favorite's ±1 slip pre-observable AT FIRE?

The 18% adjacent-bucket busts hold the ENTIRE basket loss (HIT 82% /
ADJACENT 18% / FAR 1%). This asks the only question that matters for a
skip-or-hedge gate: at fire time, is there an observable signal that the
favorite will LOSE (slip to a neighbor)? If a "contestedness" metric
separates HITs from BUSTs, it becomes the gate (and tells the 3-bucket
hedge WHEN to fire). If nothing separates, the bust isn't pre-observable
and we lean on the region cap instead — that's also a real finding.

Signals tested (the two available on resolved history):
  1. FAVORITE PRICE at fire (own-market) — leader's fill_price. The market's
     own P(favorite wins); does a lower-priced (more contested) fire bust more?
  2. FORECAST-implied distribution (measurement-use only; forecast TRADING
     stays gated) — from forecast_log bucket_snapshots at the latest issue
     before the fire: P(favorite bucket), top-2 margin, entropy, argmax-agrees,
     and forecast→favorite bucket distance. Does forecast SEE the bust coming?

(The market-implied neighbor gap + obs-boundary proximity — the mechanically
strongest signals — need the new yes3_arb / observed_extreme_c logs to
accumulate; add them here once there's data. yes3_arb has 1 row today.)

Outcome = HIT (favorite bucket won, via WUG) vs BUST (favorite lost).
Run (VPS venv):  python analyze_basket_offbyone.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import slim_dashboard as sd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

DATA = Path("data")
EXCL = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM"}  # dodgy-source, untradeable


def load(p):
    out = []
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def unit_for(sid):
    return getattr(STATIONS_BY_ID.get(sid), "unit", None) or "C"


def sweep_fire_map(trigger=0.82):
    """(sk) -> the fire snapshot (first sweep row at/above trigger), for reading
    the per-snapshot yes3_arb / observed_extreme_c at fire time."""
    by_sk = defaultdict(list)
    for r in load(DATA / "basket_sweep_log.jsonl"):
        by_sk[(r.get("station_id"), r.get("target"), r.get("target_date"))].append(r)
    out = {}
    for sk, rows in by_sk.items():
        rows.sort(key=lambda r: r.get("ts_utc") or "")
        fire = next((r for r in rows if float(r.get("leader_yes_ask") or 0) >= trigger), None)
        if fire is not None:
            out[sk] = fire
    return out


def neighbor_gap(fire):
    """Market-implied contestedness: favorite YES ask − max ±1-neighbor YES ask.
    SMALL gap = the market gives a neighbor real probability = contested. From
    the yes3_arb block (deployed 2026-06-03)."""
    y3 = fire.get("yes3_arb") if fire else None
    if not y3:
        return None
    leader = y3.get("leader")
    if not leader or leader.get("best_ask") is None:
        return None
    nb_asks = [float(nb["best_ask"]) for nb in (y3.get("neighbors") or [])
               if nb and nb.get("best_ask") is not None]
    if not nb_asks:
        return None
    return float(leader["best_ask"]) - max(nb_asks)


def obs_margin(fire, target, unit, kind, thr):
    """Degrees of headroom from the obs-so-far to the favorite bucket's KILL
    boundary (mid buckets only — the bulk of busts). SMALL/negative = the
    observed extreme is at/over the edge = imminent slip. From observed_extreme_c
    (deployed 2026-06-02). max → headroom up to T+0.5; min → headroom down to
    T−0.5. Rounding boundary is ±0.5 (floor(c+0.5))."""
    if not fire or kind != "mid" or thr is None:
        return None
    oc = fire.get("observed_extreme_c")
    if oc is None:
        return None
    obs_u = float(oc) if unit == "C" else float(oc) * 9.0 / 5.0 + 32.0
    try:
        t = int(thr)
    except (TypeError, ValueError):
        return None
    if target == "max":
        return (t + 0.5) - obs_u
    if target == "min":
        return obs_u - (t - 0.5)
    return None


def forecast_by_sk():
    """(sk) -> sorted [(issue_time, [(prob,label,threshold)...normalised])]."""
    fc = defaultdict(list)
    for r in load(DATA / "forecast_log.jsonl"):
        bs = r.get("bucket_snapshots")
        if not bs:
            continue
        dist = []
        for s in bs:
            op = s.get("our_prob")
            if op is None:
                continue
            dist.append((float(op), s.get("bucket_label"), s.get("threshold")))
        if not dist:
            continue
        sk = (r.get("station_id"), r.get("target"), r.get("target_date"))
        fc[sk].append((r.get("issue_time_utc") or "", dist))
    for k in fc:
        fc[k].sort(key=lambda x: x[0])
    return fc


def fc_metrics(snap, fav_label, fav_thr):
    """From a [(prob,label,threshold)] snapshot at fire, derive contestedness."""
    total = sum(p for p, _, _ in snap) or 1.0
    probs = [(p / total, l, t) for p, l, t in snap]
    p_fav = sum(p for p, l, _ in probs if l == fav_label)
    order = sorted(probs, key=lambda x: -x[0])
    top, second = order[0], (order[1] if len(order) > 1 else (0.0, None, None))
    ent = -sum(p * math.log(p) for p, _, _ in probs if p > 0)
    dist = None
    try:
        if fav_thr is not None and top[2] is not None:
            dist = abs(int(fav_thr) - int(top[2]))
    except (TypeError, ValueError):
        pass
    return {"p_fav": p_fav, "top2": top[0] - second[0], "ent": ent,
            "agree": top[1] == fav_label, "dist": dist}


def grp(label, hit_vals, bust_vals, fmt="%.3f"):
    """Print mean/median of a metric for HIT vs BUST groups."""
    def m(v):
        return ("n=%2d mean=" + fmt + " median=" + fmt) % (len(v), statistics.mean(v), statistics.median(v)) if v else "n=0"
    print("  %-16s HIT [%s]   BUST [%s]" % (label, m(hit_vals), m(bust_vals)))


def rate(label, items, pred):
    """Bust rate among items matching pred(rec)."""
    sel = [r for r in items if pred(r)]
    if not sel:
        print("  %-28s n=0" % label)
        return
    busts = sum(1 for r in sel if not r["hit"])
    print("  %-28s n=%3d  busts=%2d  bust-rate=%4.0f%%" % (label, len(sel), busts, 100 * busts / len(sel)))


def main():
    resmap = sd.resolution_map()
    fc = forecast_by_sk()
    sweep = sweep_fire_map()

    # favorite YES leg per event (first fill) — has kind/threshold/fill_price
    fires = {}
    for r in load(DATA / "consensus_basket_log.jsonl"):
        if r.get("result") == "filled" and r.get("side") == "YES":
            k = (r.get("station_id"), r.get("target"), r.get("target_date"))
            if k not in fires:
                fires[k] = (r.get("ts_utc") or "", r.get("bucket_label"),
                            r.get("bucket_threshold"), r.get("bucket_kind"),
                            float(r.get("fill_price") or 0.0))

    recs = []
    no_fc = 0
    for k, (fts, fb, fthr, fkind, fprice) in fires.items():
        sid = k[0]
        if sid in EXCL:
            continue
        ac = resmap.get(k)
        if ac is None:
            continue  # unresolved
        u = unit_for(sid)
        ai = _rounded_observation(ac, u)
        hit = bool(bucket_won(fkind, int(fthr), ai, u)) if fthr is not None else None
        if hit is None:
            continue
        m = None
        preds = fc.get(k)
        if preds:
            before = [x for x in preds if x[0] <= fts]
            snap = (before[-1] if before else preds[0])[1]
            m = fc_metrics(snap, fb, fthr)
        else:
            no_fc += 1
        deg_off = None
        try:
            deg_off = abs(int(ai) - int(fthr))  # |actual - fav threshold|, station unit
        except (TypeError, ValueError):
            pass
        fire = sweep.get(k)
        recs.append({"sk": k, "sid": sid, "hit": hit, "fprice": fprice, "fc": m,
                     "actual": ai, "fthr": fthr, "fkind": fkind, "fb": fb,
                     "deg_off": deg_off,
                     "nb_gap": neighbor_gap(fire),
                     "obs_margin": obs_margin(fire, k[1], u, fkind, fthr)})

    n = len(recs)
    busts = [r for r in recs if not r["hit"]]
    hits = [r for r in recs if r["hit"]]
    print("=" * 88)
    print("OFF-BY-ONE DETECTOR — clean set, resolved baskets  (N=%d: %d HIT / %d BUST = %.0f%% bust)"
          % (n, len(hits), len(busts), 100 * len(busts) / n if n else 0))
    print("=" * 88)
    if not n:
        print("no resolved baskets"); return

    # ---- bust shape (deg-off distribution) ----
    adj = sum(1 for r in busts if r["deg_off"] is not None and r["deg_off"] <= 1)
    far = sum(1 for r in busts if r["deg_off"] is not None and r["deg_off"] >= 2)
    print("\nBUST shape: ADJACENT(|actual-fav|<=1u)=%d  FAR(>=2u)=%d  (the ±1 hedge would catch ADJACENT)"
          % (adj, far))

    # ---- signal 1: FAVORITE PRICE (own-market) ----
    print("\n[1] FAVORITE PRICE at fire (own-market) — does a more-contested (cheaper) fire bust more?")
    grp("fav price", [r["fprice"] for r in hits], [r["fprice"] for r in busts])
    for lo, hi in [(0.0, 0.85), (0.85, 0.92), (0.92, 1.01)]:
        rate("  price [%.2f,%.2f)" % (lo, hi), recs, lambda r, lo=lo, hi=hi: lo <= r["fprice"] < hi)

    # ---- signal 2: FORECAST distribution (measurement-use) ----
    frecs = [r for r in recs if r["fc"]]
    fhits = [r for r in frecs if r["hit"]]
    fbusts = [r for r in frecs if not r["hit"]]
    print("\n[2] FORECAST-implied distribution  (matched to a pre-fire forecast: %d of %d; no-fc=%d)"
          % (len(frecs), n, no_fc))
    if frecs:
        grp("P(fav bucket)", [r["fc"]["p_fav"] for r in fhits], [r["fc"]["p_fav"] for r in fbusts])
        grp("top-2 margin", [r["fc"]["top2"] for r in fhits], [r["fc"]["top2"] for r in fbusts])
        grp("entropy", [r["fc"]["ent"] for r in fhits], [r["fc"]["ent"] for r in fbusts])
        print("  -- forecast as a SKIP/HEDGE gate --")
        rate("forecast DISAGREES (argmax!=fav)", frecs, lambda r: not r["fc"]["agree"])
        rate("forecast AGREES   (argmax==fav)", frecs, lambda r: r["fc"]["agree"])
        rate("forecast dist>=1 (fav!=argmax thr)", frecs, lambda r: r["fc"]["dist"] is not None and r["fc"]["dist"] >= 1)
        # the decisive question: of the busts, how many did the forecast AGREE on (saw nothing)?
        agreed_busts = sum(1 for r in fbusts if r["fc"]["agree"])
        print("  >> of %d forecast-matched BUSTS, forecast AGREED (missed it) on %d (%.0f%%) "
              "— the higher this is, the LESS the forecast can pre-flag the bust"
              % (len(fbusts), agreed_busts, 100 * agreed_busts / len(fbusts) if fbusts else 0))

    # ---- signal 3: MARKET neighbor gap (yes3_arb) — accumulating ----
    gr = [r for r in recs if r["nb_gap"] is not None]
    print("\n[3] MARKET neighbor gap at fire (fav ask − max ±1-neighbor ask; SMALL = contested)")
    print("    matched to yes3_arb: %d of %d  (deployed 2026-06-03 — accumulating)" % (len(gr), n))
    if gr:
        grp("neighbor gap", [r["nb_gap"] for r in gr if r["hit"]], [r["nb_gap"] for r in gr if not r["hit"]])

    # ---- signal 4: OBS-boundary margin (observed_extreme_c) — accumulating ----
    om = [r for r in recs if r["obs_margin"] is not None]
    print("\n[4] OBS-boundary margin at fire (° from obs-so-far to kill boundary, mid buckets; SMALL=imminent)")
    print("    matched to observed_extreme_c: %d of %d  (deployed 2026-06-02 — accumulating)" % (len(om), n))
    if om:
        grp("obs margin", [r["obs_margin"] for r in om if r["hit"]], [r["obs_margin"] for r in om if not r["hit"]], fmt="%+.2f")

    # ---- per-bust detail ----
    print("\nBUSTS (favorite lost):")
    for r in sorted(busts, key=lambda r: r["fprice"]):
        fcs = ("p_fav=%.2f top2=%+.2f agree=%s dist=%s" % (
            r["fc"]["p_fav"], r["fc"]["top2"], r["fc"]["agree"], r["fc"]["dist"])) if r["fc"] else "no-forecast"
        extra = ""
        if r["nb_gap"] is not None:
            extra += " gap=%+.2f" % r["nb_gap"]
        if r["obs_margin"] is not None:
            extra += " obsM=%+.2f" % r["obs_margin"]
        print("  %-5s %-3s %s  fav '%s'@%.2f actual=%s (|off|=%s)  %s%s"
              % (r["sk"][0], r["sk"][1], r["sk"][2][5:], r["fb"], r["fprice"],
                 r["actual"], r["deg_off"], fcs, extra))
    print("\nNotes: forecast use is MEASUREMENT-only (TRADING gated N>=30). Market-implied")
    print("  neighbor gap (yes3_arb) + obs-boundary proximity (observed_extreme_c) are the")
    print("  mechanically strongest signals — add here once those logs accumulate.")


if __name__ == "__main__":
    main()
