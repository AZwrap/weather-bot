"""Roll-REALISTIC — does rolling out of a beaten favorite still win AFTER
realistic (collapsed-bid) exits?

The roll IDEA (operator, 2026-06-02): when the held YES bucket is beaten —
the observed max climbs past it (max is monotonic) — sell the dead leg and
re-enter on the new bucket once it crosses the trigger. Resolution stays
WUG (the oracle); the death-signal would be the fast obs (METAR if proven
faster) — that part is gated on THIS measurement looking good.

`analyze_roll_shadow.py` already showed the OPTIMISTIC upper bound (exit at
entry−7 ticks, perfect landing): +$143 across 111 baskets, every flip
helped. The open question is whether that survives REALISTIC exits — by the
time the favorite is beaten, the market has usually already crashed its YES
bid toward $0 (the consensus_yes −$312 exit-bleed lesson).

This measures the realistic version on the SWEEP log, which already snapshots
the basket at every leader the market presented. Key: the sweep logs each
fade leg's `observed_ask = 1 − yes_bid`, so the REAL top-of-book SELL price of
any held leg is recoverable from a later snapshot — no approximation:
  - sell held YES-Bold at  yes_bid_Bold = 1 − (NO-Bold fade observed_ask)
  - sell held NO-Bnew  at  no_bid_Bnew  = 1 − (YES-Bnew winner observed_ask)
Both legs you're selling have collapsed (Bold dying, Bnew rising) → you
recover scrap, exactly the realism the upper bound skipped.

Trigger here = MARKET leader-flip (what the sweep loggably captures). The
obs-trigger (the operator's actual idea) could only do BETTER — exit earlier,
before the bid fully collapses — so this is a CONSERVATIVE floor on the roll's
benefit. If even this beats static, the obs/METAR refinement is upside.

Model: minimal swap per flip — sell {YES-Bold, NO-Bnew}, buy {YES-Bnew,
NO-Bold}, keep the rest (NO legs on dead-low buckets are guaranteed winners).
Buys use the logged depth-walked fills; sells use real top-of-book bids; taker
fees on every leg; only re-enter when the new leader ≥ TRIGGER. Final held
basket resolved against WUG (forward_log).

Run:  python analyze_roll_realistic.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from weather_bot.fees import taker_fee_usd
from weather_bot.locations import STATIONS_BY_ID
from weather_bot.pnl import _rounded_observation, bucket_won

DATA = Path("data")
SWEEP = DATA / "basket_sweep_log.jsonl"
FWD = DATA / "forward_log.jsonl"
TRIGGER = 0.85


def load_jsonl(p):
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
    s = STATIONS_BY_ID.get(sid)
    return (getattr(s, "unit", None) or "C")


def resolution_map():
    out = {}
    for r in load_jsonl(FWD):
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            out[k] = float(r["actual_obs_c"])
    return out


def leg_won(leg, actual_int, unit):
    """Did this held leg pay $1? YES wins if its bucket won; NO wins if not."""
    b = bucket_won(leg["bucket_kind"], int(leg["bucket_threshold"]), actual_int, unit)
    return b if leg["side"] == "YES" else (not b)


def payout(held, actual_int, unit):
    return sum(leg["shares"] for leg in held.values() if leg_won(leg, actual_int, unit))


def basket_cost(held):
    return sum(leg["net_cost"] for leg in held.values())


def mk_leg(side, snap):
    return {"side": side, "bucket_label": snap["bucket_label"],
            "bucket_kind": snap["bucket_kind"], "bucket_threshold": int(snap["bucket_threshold"]),
            "shares": float(snap["shares"]), "entry": float(snap["fill_price"]),
            "net_cost": float(snap["net_cost"])}


def sell_proceeds(shares, bid):
    """Cash from selling `shares` at top-of-book `bid`, net of taker fee."""
    if bid is None or bid <= 0.0:
        return 0.0
    return shares * bid - taker_fee_usd(shares, bid)


def simulate_event(rows, resmap):
    """Return (static_pnl, roll_pnl, n_flips) for one event, or None."""
    rows = sorted(rows, key=lambda r: r.get("ts_utc") or "")
    # fire = first snapshot whose leader reached the trigger and has a winner
    fire = None
    for r in rows:
        if r.get("winner") and float(r.get("leader_yes_ask", 0)) >= TRIGGER:
            fire = r
            break
    if fire is None:
        return None
    sid, tgt, date = fire["station_id"], fire["target"], fire["target_date"]
    actual_c = resmap.get((sid, tgt, date))
    if actual_c is None:
        return None
    unit = unit_for(sid)
    actual_int = _rounded_observation(actual_c, unit)

    # --- build the fire basket ---
    held0 = {}
    held0[fire["winner"]["bucket_label"]] = mk_leg("YES", fire["winner"])
    for l in fire.get("fade_no", []):
        held0[l["bucket_label"]] = mk_leg("NO", l)
    static_pnl = -basket_cost(held0) + payout(held0, actual_int, unit)

    # --- roll through subsequent leader flips ---
    held = dict(held0)
    cash = -basket_cost(held0)
    cur = fire["winner"]["bucket_label"]
    n_flips = 0
    fire_seen = False
    for r in rows:
        if r is fire:
            fire_seen = True
            continue
        if not fire_seen or not r.get("winner"):
            continue
        new_label = r["winner"]["bucket_label"]
        if new_label == cur or float(r.get("leader_yes_ask", 0)) < TRIGGER:
            continue
        # ---- ROLL  cur -> new_label ----
        fade_by = {l["bucket_label"]: l for l in r.get("fade_no", [])}
        # 1) sell held YES-cur at its real top-of-book bid (= 1 - NO-cur ask)
        if cur in held and held[cur]["side"] == "YES":
            ncur = fade_by.get(cur)
            yes_bid_cur = (1.0 - float(ncur["observed_ask"])) if (ncur and ncur.get("observed_ask") is not None) else 0.0
            cash += sell_proceeds(held[cur]["shares"], yes_bid_cur)
            del held[cur]
        # 2) sell held NO-new at its real top-of-book bid (= 1 - YES-new ask)
        if new_label in held and held[new_label]["side"] == "NO":
            no_bid_new = 1.0 - float(r["winner"]["observed_ask"]) if r["winner"].get("observed_ask") is not None else 0.0
            cash += sell_proceeds(held[new_label]["shares"], no_bid_new)
            del held[new_label]
        # 3) buy YES-new (the new favorite) at its logged depth-walked fill
        wl = mk_leg("YES", r["winner"]); cash -= wl["net_cost"]; held[new_label] = wl
        # 4) buy NO-cur (old favorite is now a fade) at its logged fill, if available
        if cur in fade_by:
            nl = mk_leg("NO", fade_by[cur]); cash -= nl["net_cost"]; held[cur] = nl
        cur = new_label
        n_flips += 1

    roll_pnl = cash + payout(held, actual_int, unit)
    return static_pnl, roll_pnl, n_flips


def main():
    rows = load_jsonl(SWEEP)
    resmap = resolution_map()
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[(r.get("station_id"), r.get("target"), r.get("target_date"))].append(r)

    per_station = defaultdict(lambda: {"n": 0, "flips": 0, "static": 0.0, "roll": 0.0, "helped": 0, "hurt": 0})
    tot = {"n": 0, "flips": 0, "static": 0.0, "roll": 0.0, "helped": 0, "hurt": 0}
    flip_detail = []
    for (sid, tgt, date), evrows in by_ev.items():
        res = simulate_event(evrows, resmap)
        if res is None:
            continue
        static_pnl, roll_pnl, n_flips = res
        s = per_station[sid]
        for d in (s, tot):
            d["n"] += 1; d["flips"] += n_flips
            d["static"] += static_pnl; d["roll"] += roll_pnl
        if n_flips > 0:
            delta = roll_pnl - static_pnl
            if delta > 0.01:
                s["helped"] += 1; tot["helped"] += 1
            elif delta < -0.01:
                s["hurt"] += 1; tot["hurt"] += 1
            flip_detail.append((sid, tgt, date, static_pnl, roll_pnl, delta, n_flips))

    print("=" * 78)
    print("ROLL-REALISTIC — static fire-once vs roll-on-flip with REAL collapsed-bid exits")
    print("(trigger=%.2f, sells at real top-of-book bid, fees both sides, WUG resolution)" % TRIGGER)
    print("=" * 78)
    print("scored events: %d   flipped: %d   roll helped: %d  hurt: %d" % (
        tot["n"], len(flip_detail), tot["helped"], tot["hurt"]))
    print()
    print("OVERALL")
    print("  static = $%+8.2f   roll = $%+8.2f   Δ = $%+8.2f" % (tot["static"], tot["roll"], tot["roll"] - tot["static"]))
    print()
    print("FLIPPED EVENTS (the only ones the roll touches)")
    for sid, tgt, date, st, rl, dl, nf in sorted(flip_detail, key=lambda x: x[5]):
        print("  %-5s %-3s %s  static $%+7.2f -> roll $%+7.2f  Δ $%+7.2f  (%d flip)" % (sid, tgt, date, st, rl, dl, nf))
    print()
    print("PER-STATION (only stations with a flip shown)")
    for sid in sorted(per_station, key=lambda x: (per_station[x]["roll"] - per_station[x]["static"])):
        s = per_station[sid]
        if s["flips"] == 0:
            continue
        print("  %-5s n=%2d flips=%d  static $%+7.2f -> roll $%+7.2f  Δ $%+7.2f  (helped %d/hurt %d)" % (
            sid, s["n"], s["flips"], s["static"], s["roll"], s["roll"] - s["static"], s["helped"], s["hurt"]))


if __name__ == "__main__":
    main()
