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
    """Cash from selling `shares` at depth-aware avg `bid`, net of taker fee."""
    if bid is None or bid <= 0.0:
        return 0.0
    return shares * bid - taker_fee_usd(shares, bid)


def simulate_event(rows, resmap, stop="any", trigger=TRIGGER):
    """Return (static_pnl, roll_pnl, n_flips) for one event, or None.

    obs_confirmed=True models the operator's actual idea: only roll OUT of a
    bucket the observed max has genuinely passed (the favorite is dead). We
    proxy "obs would have confirmed death" with perfect foresight — skip the
    roll if the currently-held YES bucket is the eventual WUG winner. This is
    an upper bound on the no-whipsaw benefit (perfect death-detection, but
    still REAL collapsed-bid exits) and it never rolls a would-be winner."""
    rows = sorted(rows, key=lambda r: r.get("ts_utc") or "")
    # fire = first snapshot whose leader reached the trigger and has a winner
    fire = None
    for r in rows:
        if r.get("winner") and float(r.get("leader_yes_ask", 0)) >= trigger:
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
        if new_label == cur or float(r.get("leader_yes_ask", 0)) < trigger:
            continue
        fade_by = {l["bucket_label"]: l for l in r.get("fade_no", [])}
        # ---- STOP-MODE GATE: decide whether to roll ----
        if stop == "obs":
            # only roll out of a genuinely-dead favorite (perfect foresight)
            if cur in held and held[cur]["side"] == "YES" and leg_won(held[cur], actual_int, unit):
                continue
        elif stop == "pp10":
            # roll only when the held YES dropped >=10pp from entry (a big,
            # informative move — filters small leader-flicker whipsaws). NOTE:
            # the sell is still priced at the flip-time snapshot (we lack
            # sub-flip price snapshots), so this isolates the THRESHOLD effect,
            # not the "sell earlier at a less-collapsed price" effect.
            ncur0 = fade_by.get(cur)
            cur_px = (1.0 - float(ncur0["observed_ask"])) if (ncur0 and ncur0.get("observed_ask") is not None) else 0.0
            if not (cur in held and held[cur]["side"] == "YES" and cur_px <= held[cur]["entry"] - 0.10):
                continue
        # stop == "any" → roll on every flip (no gate)
        # ---- ROLL  cur -> new_label ----
        # 1) sell held YES-cur — DEPTH-AWARE. YES_bid ladder ≡ NO_ask ladder
        # (same resting orders), and the fade NO-cur leg logs the DEPTH-WALKED
        # buy fill, so 1 - fill_price is the avg price selling ~$5 of YES-cur
        # into the collapsed book (≥ a top-of-book overestimate → conservative).
        if cur in held and held[cur]["side"] == "YES":
            ncur = fade_by.get(cur)
            yes_bid_cur = (1.0 - float(ncur["fill_price"])) if (ncur and ncur.get("fill_price") is not None) else 0.0
            cash += sell_proceeds(held[cur]["shares"], yes_bid_cur)
            del held[cur]
        # 2) sell held NO-new — DEPTH-AWARE. winner YES-new's depth-walked buy
        # fill walks the YES-new ask = NO-new bid ladder, so 1 - fill_price is
        # the avg NO-new sell price into depth.
        if new_label in held and held[new_label]["side"] == "NO":
            no_bid_new = (1.0 - float(r["winner"]["fill_price"])) if r["winner"].get("fill_price") is not None else 0.0
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


EXCLUDED = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM"}  # now untradeable (dodgy source)


def main():
    TRIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    rows = load_jsonl(SWEEP)
    resmap = resolution_map()
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[(r.get("station_id"), r.get("target"), r.get("target_date"))].append(r)

    # accumulators: [all, clean] x [static, mkt-roll, obs-roll]
    agg = {scope: {"static": 0.0, "mkt": 0.0, "obs": 0.0, "n": 0, "flips": 0}
           for scope in ("all", "clean")}
    detail = []
    for (sid, tgt, date), evrows in by_ev.items():
        rm = simulate_event(evrows, resmap, stop="any", trigger=TRIG)
        ro = simulate_event(evrows, resmap, stop="obs", trigger=TRIG)
        if rm is None or ro is None:
            continue
        static_pnl, mkt_pnl, n_flips = rm
        _, obs_pnl, _ = ro
        scopes = ["all"] + (["clean"] if sid not in EXCLUDED else [])
        for sc in scopes:
            a = agg[sc]
            a["static"] += static_pnl; a["mkt"] += mkt_pnl; a["obs"] += obs_pnl
            a["n"] += 1; a["flips"] += n_flips
        if n_flips > 0:
            detail.append((sid, tgt, date, static_pnl, mkt_pnl, obs_pnl, n_flips))

    print("=" * 84)
    print("ROLL-REALISTIC — REAL collapsed-bid exits, taker fees both sides, WUG resolution")
    print("  static     = fire once, hold to resolution")
    print("  mkt-roll   = roll on every market leader-flip (whipsaw-prone)")
    print("  obs-roll   = roll ONLY when the held favorite is genuinely dead (operator's idea;")
    print("               perfect death-detection upper bound, no whipsaw)")
    print("  reference  : analyze_roll_shadow upper bound (cheap exits) = +$143.72")
    print("=" * 84)
    for sc in ("all", "clean"):
        a = agg[sc]
        lbl = "ALL STATIONS" if sc == "all" else "CLEAN (tradeable; excl Moscow/Istanbul/etc.)"
        print("\n%s — %d events, %d flips" % (lbl, a["n"], a["flips"]))
        print("  static   $%+8.2f" % a["static"])
        print("  mkt-roll $%+8.2f   Δ $%+7.2f" % (a["mkt"], a["mkt"] - a["static"]))
        print("  obs-roll $%+8.2f   Δ $%+7.2f   (+$%.2f vs mkt-roll: whipsaw avoided)"
              % (a["obs"], a["obs"] - a["static"], a["obs"] - a["mkt"]))
    print()
    # ---- multi-trigger sweep (clean tradeable set) ----
    print()
    print("STOP MODES ACROSS TRIGGERS (clean set)  static / mkt-flip / pp10-drop / obs-death")
    print("  trig   static      mkt(Δ)        pp10(Δ)       obs(Δ)")
    for trig in [0.78, 0.80, 0.82, 0.85, 0.90, 0.95]:
        st=mk=pp=ob=0.0
        for (sid, tgt, date), evrows in by_ev.items():
            if sid in EXCLUDED:
                continue
            rm = simulate_event(evrows, resmap, stop="any", trigger=trig)
            rp = simulate_event(evrows, resmap, stop="pp10", trigger=trig)
            ro = simulate_event(evrows, resmap, stop="obs", trigger=trig)
            if rm is None:
                continue
            st += rm[0]; mk += rm[1]; pp += rp[1]; ob += ro[1]
        print("  %.2f  $%+7.2f  $%+7.2f(%+5.1f)  $%+7.2f(%+5.1f)  $%+7.2f(%+5.1f)" % (trig, st, mk, mk-st, pp, pp-st, ob, ob-st))
    print()
    print("FLIPPED EVENTS  (static -> mkt-roll / obs-roll)")
    for sid, tgt, date, st, mk, ob, nf in sorted(detail, key=lambda x: x[5] - x[3]):
        flag = "  <excluded>" if sid in EXCLUDED else ""
        print("  %-5s %-3s %s  $%+7.2f -> mkt $%+7.2f / obs $%+7.2f  (%d flip)%s"
              % (sid, tgt, date, st, mk, ob, nf, flag))


if __name__ == "__main__":
    main()
