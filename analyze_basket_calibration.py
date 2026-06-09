"""FILL-INDEPENDENT calibration / inversion audit — the knowledge-edge test.

Question: are the market's +-1 neighbor buckets priced BELOW their realized
off-by-one win frequency? i.e. is there a KNOWLEDGE edge in being LONG the bust
(buying the cheap neighbors), or is the market already calibrated?

Why this matters: every PAPER P&L we have fills at the quoted price, so it is
structurally blind to adverse selection (the "mirage"). This audit sidesteps that
entirely — it compares the ASK you would PAY against whether the bucket actually
WON. Prices vs realized frequencies, no fill model. If the prices themselves are
calibrated, there is no directional edge for ANY execution, fast or slow.

Reference point per market = the fire snapshot (first sweep snapshot whose
leader_yes_ask >= TRIG), read from the yes3_arb block (leader + neighbor ask
ladders). Resolution from forward_log actual_obs_c via bucket_won.

FIRST READ (2026-06-09, N=275 markets / 534 neighbor legs, clean stations):
  Realized: favorite 80.4% · bust HIGH (+1) 13.8% · bust LOW (-1) 2.5% · far 3.3%.
  INVERSION (buy every neighbor at its fire ask): NET -$21.89 (-4.1c/leg) — you
    pay avg 12.1c for neighbors that win 8.4%. DEAD.
  Calibration curve: cheap neighbors (<=0.10) FAIR; expensive (0.20+) OVERPRICED
    (pay 33.7c, win 19.7c ~ 14pp over). Nothing underpriced to buy.
  Direction: max busts HIGH 5.5x more than LOW (13.8 vs 2.5) AND the market prices
    it in (+1 avg 18c vs -1 avg 6c). No naive-direction edge.
  VERDICT: the weather bucket market is CALIBRATED — no knowledge edge, by a
    fill-independent method that cannot be fooled by the mirage. The only measured
    mispricing is the over-feared bust (expensive neighbor overpriced) -> a FADE,
    which is exactly the consensus_basket NO-leg thesis that already loses to
    spread + bust-correlation. Un-capturable. Re-run at higher N to confirm.

Run (VPS venv):  python analyze_basket_calibration.py
"""
from __future__ import annotations
import json
import statistics
import sys
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from weather_bot.pnl import _rounded_observation, bucket_won
from weather_bot.locations import STATIONS_BY_ID

# ─────────────────────────── CONFIG ───────────────────────────
TRIG = 0.85   # fire reference: first sweep snapshot with leader_yes_ask >= this
EXCL = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM", "WIHH"}   # dodgy-source / excluded
# ───────────────────────────────────────────────────────────────


def load_actuals():
    out = {}
    with open("data/forward_log.jsonl") as fh:
        for l in fh:
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("actual_obs_c") is not None:
                out[(r["station_id"], r["target"], r["target_date"])] = r["actual_obs_c"]
    return out


def load_fire_snapshots():
    """First yes3_arb snapshot per market with leader ask >= TRIG."""
    fire = {}
    with open("data/basket_sweep_log.jsonl") as fh:
        for l in fh:
            if "yes3" not in l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            y = r.get("yes3_arb")
            if not y or not y.get("leader") or not y.get("neighbors"):
                continue
            if float(r.get("leader_yes_ask") or 0) < TRIG:
                continue
            k = (r["station_id"], r["target"], r["target_date"])
            if k not in fire:
                fire[k] = r
    return fire


def main():
    fwd = load_actuals()
    fire = load_fire_snapshots()

    realized = defaultdict(int)
    neigh = []   # (price_you_pay, won, signed_delta_from_favorite)
    n = 0
    for k, r in fire.items():
        sid, tgt, date = k
        if sid in EXCL:
            continue
        ac = fwd.get(k)
        if ac is None:
            continue
        unit = getattr(STATIONS_BY_ID.get(sid), "unit", None) or "C"
        robs = _rounded_observation(ac, unit)
        y = r["yes3_arb"]
        lead, nbrs = y["leader"], y["neighbors"]
        if lead.get("bucket_kind") != "mid" or lead.get("bucket_threshold") is None:
            continue
        n += 1
        lt = int(lead["bucket_threshold"])
        won_label = "fav" if bucket_won("mid", lt, robs, unit) else "far"
        for nb in nbrs:
            if nb.get("bucket_kind") != "mid" or nb.get("bucket_threshold") is None:
                continue
            nt = int(nb["bucket_threshold"])
            nb_won = bucket_won("mid", nt, robs, unit)
            neigh.append((float(nb.get("best_ask") or 0.0), 1 if nb_won else 0, nt - lt))
            if nb_won:
                won_label = "+nbr" if nt > lt else "-nbr"
        realized[won_label] += 1

    print("CALIBRATION / INVERSION AUDIT  (fill-independent)  leader>=%.2f  N=%d markets" % (TRIG, n))
    print("=" * 72)
    print("REALIZED outcome distribution:")
    for lbl in ["fav", "+nbr", "-nbr", "far"]:
        print("  %-5s: %3d  (%4.1f%%)" % (lbl, realized[lbl], 100 * realized[lbl] / n if n else 0))

    prices = [p for p, w, d in neigh]
    won = [w for p, w, d in neigh]
    print("\nALL neighbor legs: %d   avg ASK you'd pay=$%.4f   realized win=%.1f%%   raw edge=%+.4f/sh"
          % (len(neigh), statistics.mean(prices), 100 * statistics.mean(won),
             statistics.mean(won) - statistics.mean(prices)))

    print("\nDIRECTION (does the max bust HIGH or LOW?):")
    for dd, name in [(1, "+1 (bust HIGH)"), (-1, "-1 (bust LOW)")]:
        sub = [(p, w) for p, w, d in neigh if (d > 0) == (dd > 0) and d != 0]
        if sub:
            ap = statistics.mean(p for p, w in sub)
            aw = statistics.mean(w for p, w in sub)
            print("  %-14s n=%-4d avg_ask=$%.4f realized=%5.1f%%  edge=%+.4f/sh"
                  % (name, len(sub), ap, 100 * aw, aw - ap))

    print("\nCALIBRATION CURVE  (neighbor ask bin -> realized win%; UNDERPRICED if realized > price):")
    for lo, hi in [(0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 1.0)]:
        sub = [(p, w) for p, w, d in neigh if lo <= p < hi]
        if sub:
            ap = statistics.mean(p for p, w in sub)
            aw = statistics.mean(w for p, w in sub)
            flag = "UNDERPRICED" if aw > ap + 0.005 else ("overpriced" if aw < ap - 0.005 else "fair")
            print("  [%.2f,%.2f) n=%-4d avg_ask=$%.4f realized=%5.1f%%  -> %s"
                  % (lo, hi, len(sub), ap, 100 * aw, flag))

    ev = sum((w - p) for p, w, d in neigh)
    fee = sum(0.05 * p * (1 - p) for p, w, d in neigh)   # Polymarket fee approx
    print("\nINVERSION bottom line (buy EVERY neighbor at its fire ask, $1 payout):")
    print("  gross EV=$%+.2f  -fees=$%.2f  NET=$%+.2f  over %d legs (avg %+.4f/leg)"
          % (ev, fee, ev - fee, len(neigh), (ev - fee) / len(neigh) if neigh else 0))


if __name__ == "__main__":
    main()
