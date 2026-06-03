"""Price-stop backtest on the sub-second favorite-bid log — the decisive test
of the DECOUPLED STOP (operator, 2026-06-02).

The decoupled stop (memory: project_basket_roll_realistic_2026-06-02): SELL the
held basket-favorite YES leg when its bid drops, then re-enter ONLY on the
normal 0.82 trigger (never forced). Selling at the MARKET bid is ~EV-NEUTRAL
minus spread+fee, so a bare price-stop is a VARIANCE / left-tail tool, not an
EV tool — its value is cutting volatile losers early at fair value (lower
drawdown → size up under Kelly). The ONLY +EV path is a death signal FASTER
than the market (obs/METAR), measured elsewhere. So this analyzer's job is to
answer the MECHANISM questions the coarse sweep could not, at REAL sub-second
execution speed (data/basket_favorite_ticks.jsonl):

  1. SEPARATOR: do winners stay high while losers fall? (whipsaw risk)
  2. STOPPABILITY: when a favorite busts, does its bid DECLINE GRADUALLY
     through a stoppable price zone, or JUMP 0.999→0 at resolution
     (unstoppable — no intermediate price to sell at)?
  3. $ DELTA + TAIL: P&L of "stop + sit flat" vs "hold", and the left-tail
     reduction (the actual point of the stop).

Re-entry is NOT modelled here: in the decoupled design re-entry is just a
normal new basket fire (already in the system's P&L), so the stop's marginal
effect is exactly "sell the dropping leg + sit flat" — the conservative,
honest scope of this log.

TWO stop references (each swept over X):
  entry-X : sell at first tick bid <= entry(ask) - X. The spec'd rule, BUT
            entry is the taker ASK we filled at; best_bid sits below it by the
            spread, so small X can fire on the spread alone (artifact) — we
            flag immediate-fire positions.
  peak-X  : sell at first tick bid <= (running peak bid) - X. A trailing stop;
            spread-immune (peak and current are both bids) → the robust
            drawdown detector. LEAD WITH THIS.

Resolution = WUG (forward_log actual_obs_c → rounded → bucket_won). Dollars use
the real shares + fill_price from consensus_basket_log. Taker fees both sides.

Run:  python analyze_basket_price_stop.py        (on the VPS venv)
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
TICKS = DATA / "basket_favorite_ticks.jsonl"
FWD = DATA / "forward_log.jsonl"
CBL = DATA / "consensus_basket_log.jsonl"

EXCLUDED = {"LTFM", "LLBG", "UUWW", "VHHH", "DNMM"}  # dodgy-source, untradeable
STOPS_PP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]      # drop thresholds swept


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
    return getattr(s, "unit", None) or "C"


def resolution_map():
    out = {}
    for r in load_jsonl(FWD):
        if r.get("actual_obs_c") is None:
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"))
        if all(k):
            out[k] = float(r["actual_obs_c"])
    return out


def basket_leg_map():
    """(sid,target,date,bucket_label) -> {kind, threshold, shares, fill_price}.

    Aggregates the YES favorite legs (sum shares, share-weighted fill)."""
    acc = defaultdict(lambda: {"kind": None, "threshold": None, "sh": 0.0, "fv": 0.0})
    for r in load_jsonl(CBL):
        if r.get("side") != "YES" or r.get("result") != "filled":
            continue
        k = (r.get("station_id"), r.get("target"), r.get("target_date"), r.get("bucket_label"))
        if not all(k[:3]) or k[3] is None:
            continue
        sh = float(r.get("shares") or 0.0)
        a = acc[k]
        a["kind"] = r.get("bucket_kind")
        a["threshold"] = r.get("bucket_threshold")
        a["sh"] += sh
        a["fv"] += sh * float(r.get("fill_price") or 0.0)
    out = {}
    for k, a in acc.items():
        if a["sh"] <= 0 or a["kind"] is None or a["threshold"] is None:
            continue
        out[k] = {"kind": a["kind"], "threshold": int(a["threshold"]),
                  "shares": a["sh"], "fill_price": a["fv"] / a["sh"]}
    return out


def build_positions():
    """One row per held favorite position with its full bid trajectory +
    resolved outcome. Returns list of dicts; only RESOLVED + joinable kept."""
    legmap = basket_leg_map()
    resmap = resolution_map()
    by_tok = defaultdict(list)
    for r in load_jsonl(TICKS):
        by_tok[r.get("yes_token_id")].append(r)

    positions = []
    skipped = {"unresolved": 0, "no_leg": 0}
    for tok, rows in by_tok.items():
        rows = sorted(rows, key=lambda r: r.get("ts_utc") or "")
        r0 = rows[0]
        sid, tgt, date, label = r0["station_id"], r0["target"], r0["target_date"], r0["bucket_label"]
        leg = legmap.get((sid, tgt, date, label))
        if leg is None:
            skipped["no_leg"] += 1
            continue
        actual_c = resmap.get((sid, tgt, date))
        if actual_c is None:
            skipped["unresolved"] += 1
            continue
        unit = unit_for(sid)
        won = bucket_won(leg["kind"], leg["threshold"], _rounded_observation(actual_c, unit), unit)
        raw = [float(r["best_bid"]) for r in rows]
        # best_bid == 0.0 = EMPTY bid side (no resting buy orders) → NOT a
        # tradeable sell price. This removes (a) the terminal settlement
        # collapse a winning favorite's feed shows at resolution and (b)
        # already-dead favorites logged at 0. Real intermediate prices (a
        # loser's gradual decline, a winner's dip-and-recover) are >0 → kept.
        bids = [b for b in raw if b > 0.0]
        positions.append({
            "sid": sid, "target": tgt, "date": date, "label": label,
            "entry": float(leg["fill_price"]), "shares": float(leg["shares"]),
            "won": bool(won), "bids": bids, "n_raw": len(raw),
            "live": bool(bids),
            "first_bid": (bids[0] if bids else None),
            "peak": (max(bids) if bids else None),
            "min": (min(bids) if bids else None),
            "n": len(bids),
        })
    return positions, skipped


def hold_pnl(pos):
    s, e = pos["shares"], pos["entry"]
    payout = s if pos["won"] else 0.0
    return payout - s * e - taker_fee_usd(s, e)  # incl. buy fee


def stop_pnl(pos, x, ref):
    """Sell at first tick bid <= ref - x; flat after. ref in {'entry','peak'}.

    Returns (pnl, fired, sell_bid). If never fires, == hold_pnl (held to end)."""
    s, e = pos["shares"], pos["entry"]
    peak = -1.0
    for b in pos["bids"]:
        if b > peak:
            peak = b
        thr = (e if ref == "entry" else peak) - x
        if b <= thr:
            pnl = s * b - s * e - taker_fee_usd(s, e) - taker_fee_usd(s, b)
            return pnl, True, b
    return hold_pnl(pos), False, None


def fired_immediately(pos, x):
    """entry-ref stop that fires on the FIRST tick (spread/lag artifact)."""
    return bool(pos["bids"]) and pos["bids"][0] <= pos["entry"] - x


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def report(positions, scope_label):
    n = len(positions)
    live = [p for p in positions if p["live"]]
    dead = [p for p in positions if not p["live"]]
    wins = [p for p in live if p["won"]]
    losers = [p for p in live if not p["won"]]
    print("=" * 92)
    print("%s — %d resolved favorite positions  (%d won / %d lost)"
          % (scope_label, n, sum(p["won"] for p in positions), sum(not p["won"] for p in positions)))
    print("   live (a tradeable bid was ever observed): %d   |   no-live-bid (dead before logging, "
          "unstoppable): %d" % (len(live), len(dead)))
    if dead:
        print("   no-live-bid: " + ", ".join("%s/%s%s" % (p["sid"], p["target"],
              "" if p["won"] else "·LOST") for p in sorted(dead, key=lambda p: p["sid"])))
    print("=" * 92)
    if not live:
        print("  (no live positions to analyse)\n")
        return

    # ---- 1. SEPARATOR: drawdown-from-peak, winners vs losers (live only) ----
    def dd(p):  # max drawdown from running peak, in pp
        return (p["peak"] - p["min"]) * 100.0
    print("\n[1] SEPARATOR — intraday drawdown from running peak bid (pp), live positions")
    for grp, name in ((wins, "WON "), (losers, "LOST")):
        if grp:
            dds = sorted(dd(p) for p in grp)
            print("  %s n=%2d   min=%4.0f  median=%4.0f  max=%4.0f   | dipped >=10pp: %d/%d  >=15pp: %d/%d"
                  % (name, len(grp), dds[0], dds[len(dds)//2], dds[-1],
                     sum(d >= 10 for d in dds), len(grp),
                     sum(d >= 15 for d in dds), len(grp)))
    big_dip_winners = [p for p in wins if dd(p) >= 15]
    if big_dip_winners:
        print("  WINNERS that dipped >=15pp then recovered (a price stop would WHIPSAW these):")
        for p in sorted(big_dip_winners, key=lambda p: p["min"]):
            print("    %-5s %-3s %s  entry=%.2f peak=%.3f min=%.3f -> WON  (dip %.0fpp)"
                  % (p["sid"], p["target"], p["date"][5:], p["entry"], p["peak"], p["min"], dd(p)))

    # ---- 2. STOPPABILITY of losers: gradual vs terminal-jump ----
    print("\n[2] LOSER STOPPABILITY — did the bid pass through a stoppable zone before dying?")
    if losers:
        for p in sorted(losers, key=lambda p: p["sid"]):
            zone = sum(1 for b in p["bids"] if 0.30 <= b <= p["peak"] - 0.10)
            verdict = "GRADUAL (stoppable)" if zone >= 1 else "JUMP->0 (unstoppable: no mid-band tick)"
            print("  %-5s %-3s %s  entry=%.2f peak=%.3f min=%.3f n=%-4d mid-band_ticks=%-3d  %s"
                  % (p["sid"], p["target"], p["date"][5:], p["entry"], p["peak"], p["min"], p["n"], zone, verdict))
    else:
        print("  (no live losers in this set)")

    # ---- 3. $ DELTA + TAIL across stop thresholds, both references ----
    base_hold = sum(hold_pnl(p) for p in positions)
    worst_hold = min(hold_pnl(p) for p in positions)
    print("\n[3] STOP P&L vs HOLD  (hold total = $%+.2f, worst single = $%+.2f)" % (base_hold, worst_hold))
    for ref in ("peak", "entry"):
        tag = "PEAK-trailing (robust)" if ref == "peak" else "ENTRY-relative (spec'd; spread-prone)"
        print("  -- %s --" % tag)
        print("     X     fired  win/los  $stop_total   Δ vs hold   Δwin(cost)  Δlos(save)  worst_single")
        for x in STOPS_PP:
            tot = wlost = wfired = lfired = 0.0
            dwin = dlos = 0.0
            worst = 0.0
            first = True
            for p in positions:
                sp, fired, _ = stop_pnl(p, x, ref)
                hp = hold_pnl(p)
                tot += sp
                if first or sp < worst:
                    worst = sp; first = False
                if fired:
                    wfired += 1
                    if p["won"]:
                        wlost += 1
                        dwin += sp - hp
                    else:
                        lfired += 1
                        dlos += sp - hp
            extra = ""
            if ref == "entry":
                imm = sum(fired_immediately(p, x) for p in positions)
                extra = "  [%d immediate/spread]" % imm
            print("    %4.0fpp  %5.0f  %3.0f/%-3.0f  $%+9.2f   $%+8.2f   $%+8.2f   $%+8.2f   $%+8.2f%s"
                  % (x * 100, wfired, wlost, lfired, tot, tot - base_hold, dwin, dlos, worst, extra))


def main():
    positions, skipped = build_positions()
    print("Loaded %d resolved+joinable favorite positions "
          "(skipped: %d unresolved, %d no basket-leg join)\n"
          % (len(positions), skipped["unresolved"], skipped["no_leg"]))
    clean = [p for p in positions if p["sid"] not in EXCLUDED]
    report(clean, "CLEAN (tradeable; excl Moscow/Istanbul/etc.)")
    if len(positions) != len(clean):
        print()
        report(positions, "ALL STATIONS (incl. dodgy-source)")
    print("\nNotes: bid trajectory = top-of-book best_bid (tick log has NO depth) →")
    print("  sell price assumes ~$5 notional fills at top-of-book (slightly optimistic on thin books).")
    print("  A bare stop is ~EV-neutral minus spread+fee; its value is LEFT-TAIL reduction, not EV.")
    print("  Re-entry is excluded by design (decoupled stop = sell + sit flat; re-entry = normal trigger).")


if __name__ == "__main__":
    main()
