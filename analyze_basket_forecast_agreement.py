"""Does the weather forecast flag the basket's busts? (SHADOW — no trading change.)

Operator idea (2026-06-02, flagged "on the edge"): use the forecast to avoid
buying into a transient spike — i.e., if the 0.85 favorite the basket fired on
DISAGREES with the forecast's most-likely bucket, that fire is more likely a
spike that busts. This MEASURES whether disagreeing fires lose more. It changes
NOTHING live. (Memory: forecast-driven TRADING is rejected/gated at N>=30; this
is measurement only.)

Join: consensus_basket fires (the YES favorite) × forecast_log (argmax our_prob
bucket at the latest issue time BEFORE the fire) × resolution (net-of-fee basket
P&L via slim_dashboard). Reports AGREE vs DISAGREE net/avg/win%, and the
disagreeing losers.

Run:  python analyze_basket_forecast_agreement.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import slim_dashboard as sd

DATA = Path("data")


def load(p: Path):
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


def main():
    # 1) forecast predicted bucket (argmax our_prob) per (sk) over time
    fc = defaultdict(list)
    for r in load(DATA / "forecast_log.jsonl"):
        bs = r.get("bucket_snapshots")
        if not bs:
            continue
        best = None
        for s in bs:
            op = s.get("our_prob")
            if op is None:
                continue
            if best is None or op > best[0]:
                best = (op, s.get("bucket_label"), s.get("threshold"))
        if best is None:
            continue
        sk = (r.get("station_id"), r.get("target"), r.get("target_date"))
        fc[sk].append((r.get("issue_time_utc") or "", best[1], best[2]))
    for k in fc:
        fc[k].sort()

    # 2) net-of-fee basket P&L per event (resolved) via the dashboard engine
    resmap = sd.resolution_map()
    posrows = sd.compute_positions(resmap)
    by_ev = defaultdict(list)
    for p in posrows:
        if p["strategy"] == "consensus_basket":
            by_ev[(p["station"], p["target"], p["date"])].append(p)

    # 3) fires (first YES favorite per event) from the basket log (has threshold)
    fires = {}
    for r in load(DATA / "consensus_basket_log.jsonl"):
        if r.get("result") == "filled" and r.get("side") == "YES":
            k = (r.get("station_id"), r.get("target"), r.get("target_date"))
            if k not in fires:
                fires[k] = (r.get("ts_utc") or "", r.get("bucket_label"), r.get("bucket_threshold"))

    agree, disagree, nofc = [], [], 0
    for k, legs in by_ev.items():
        if not any(l["status"] == "resolved" for l in legs):
            continue
        net = sum(l["net"] for l in legs if l["status"] == "resolved")
        if k not in fires:
            continue
        fts, fb, fthr = fires[k]
        preds = fc.get(k)
        if not preds:
            nofc += 1
            continue
        before = [x for x in preds if x[0] <= fts]
        pred = before[-1] if before else preds[0]
        dist = None
        try:
            if fthr is not None and pred[2] is not None:
                dist = abs(int(fthr) - int(pred[2]))
        except (TypeError, ValueError):
            pass
        rec = (net, k, fb, pred[1], dist)
        (agree if fb == pred[1] else disagree).append(rec)

    def summ(rs, lab):
        if not rs:
            print("  %-9s none" % lab)
            return
        nets = [x[0] for x in rs]
        wins = sum(1 for n in nets if n > 0)
        print("  %-9s n=%3d  net=$%+8.2f  avg=$%+6.2f  win%%=%3.0f" % (
            lab, len(rs), sum(nets), sum(nets) / len(rs), 100 * wins / len(rs)))

    print("=" * 70)
    print("BASKET vs FORECAST — does disagreement flag busts? (shadow)")
    print("=" * 70)
    print("fires matched to a pre-fire forecast: agree=%d disagree=%d (no-forecast=%d)" % (
        len(agree), len(disagree), nofc))
    summ(agree, "AGREE")
    summ(disagree, "DISAGREE")
    # split disagree by distance (1 bucket vs >=2)
    d1 = [r for r in disagree if r[4] == 1]
    d2 = [r for r in disagree if r[4] is not None and r[4] >= 2]
    summ(d1, " dist=1")
    summ(d2, " dist>=2")
    print("disagreeing losers (fired vs forecast):")
    for n, k, fb, pb, dist in sorted(disagree, key=lambda x: x[0])[:10]:
        print("   %-5s %-3s %s  fired '%s' vs forecast '%s' (dist %s)  net $%+.2f" % (
            k[0], k[1], k[2], fb, pb, dist, n))


if __name__ == "__main__":
    main()
