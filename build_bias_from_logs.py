"""Build a bias_table.json from logged forecasts + actuals.

Unlike train_bias.py (which RE-FETCHES historical forecasts from
Open-Meteo at training time, possibly at a different lead than we log
in production), this trains on the EXACT production forecasts the bot
recorded — same model, same lead, same issue cadence. That's the
faithful "rebuild from the data we accumulated."

Two input shapes, both supported:
  - JOINED  (archived forward_log): each record has both raw_members_c
            and actual_obs_c.
  - SPLIT   (fresh lite-rebuild): forecasts in --forecasts files
            (raw_members_c, actual_obs_c=None) joined to actuals in
            --actuals files (actual_obs_c set) by (station,target,date).

Per (station, target):
  residual_d = median(members on day d) - actual_d   (one per resolved day)
  bias_c  = mean(residual_d)
  rmse_c  = sqrt(mean(residual_d^2))
  n_days  = number of resolved days
Output matches weather_bot.bias.BiasEntry.as_jsonable so BiasTable.load
reads it directly. Pure stdlib (no numpy) so it runs anywhere.

Usage:
  python build_bias_from_logs.py --forecasts "data_archive/data/forward_log*.jsonl" --output bias_table.json
  python build_bias_from_logs.py --forecasts data/forecast_log.jsonl --actuals data/forward_log.jsonl --output bias_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _iter(files):
    for f in files:
        try:
            fh = open(f, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forecasts", required=True,
                   help="glob of JSONL files holding forecast records (raw_members_c).")
    p.add_argument("--actuals", default=None,
                   help="optional glob of JSONL holding actual_obs_c (for the SPLIT case). "
                        "If omitted, actuals are read from the forecast records themselves.")
    p.add_argument("--output", default="bias_table.json")
    p.add_argument("--min-days", type=int, default=5,
                   help="skip (station,target) pairs with fewer resolved days.")
    p.add_argument("--merge-into", default=None,
                   help="existing bias_table.json to merge over. A freshly-"
                        "computed entry only REPLACES the existing one when it "
                        "has >= --replace-min-days resolved days; otherwise the "
                        "existing (seed) entry is kept. Prevents a thin re-train "
                        "from wiping the archived seed.")
    p.add_argument("--replace-min-days", type=int, default=14,
                   help="min fresh resolved days before a fresh entry replaces "
                        "the merged-in (seed) entry for that station.")
    args = p.parse_args()

    fc_files = sorted(glob.glob(args.forecasts))
    if not fc_files:
        print(f"no forecast files matched {args.forecasts!r}")
        return 1

    # Actuals lookup (SPLIT case): (sid,target,date) -> actual_obs_c
    actuals: dict[tuple[str, str, str], float] = {}
    if args.actuals:
        for r in _iter(sorted(glob.glob(args.actuals))):
            a = r.get("actual_obs_c")
            if a is None:
                continue
            k = (r.get("station_id"), r.get("target"), r.get("target_date"))
            if all(k):
                actuals[k] = float(a)

    # Latest-issue forecast median per (sid,target,date), with its actual.
    # best[(sid,t,date)] = {"issue": str, "p50": float, "actual": float}
    best: dict[tuple[str, str, str], dict] = {}
    rows = 0
    for r in _iter(fc_files):
        members = r.get("raw_members_c")
        if not members:
            continue
        sid = r.get("station_id"); tgt = r.get("target"); td = r.get("target_date")
        if not (sid and tgt and td):
            continue
        # actual: from the record (joined) or the side lookup (split)
        actual = r.get("actual_obs_c")
        if actual is None and args.actuals:
            actual = actuals.get((sid, tgt, td))
        if actual is None:
            continue
        rows += 1
        issue = r.get("issue_time_utc", "")
        k = (sid, tgt, td)
        cur = best.get(k)
        if cur is None or issue > cur["issue"]:
            try:
                p50 = statistics.median(float(m) for m in members)
            except (TypeError, ValueError, statistics.StatisticsError):
                continue
            best[k] = {"issue": issue, "p50": p50, "actual": float(actual)}

    # Aggregate per (station, target)
    resid = defaultdict(list)   # (sid,t) -> [residual,...]
    last_day = defaultdict(str)
    for (sid, tgt, td), v in best.items():
        resid[(sid, tgt)].append(v["p50"] - v["actual"])
        if td > last_day[(sid, tgt)]:
            last_day[(sid, tgt)] = td

    fresh = {}
    for (sid, tgt), errs in sorted(resid.items()):
        n = len(errs)
        if n < args.min_days:
            continue
        bias = statistics.mean(errs)
        rmse = (statistics.mean(e * e for e in errs)) ** 0.5
        fresh[(sid, tgt)] = {
            "station_id": sid,
            "target": tgt,
            "bias_c": round(bias, 4),
            "n_days": n,
            "rmse_c": round(rmse, 4),
            "trained_through": last_day[(sid, tgt)],
        }

    # Merge over an existing seed table if requested. A fresh entry only
    # replaces the seed once it has >= replace_min_days resolved days, so
    # a thin weekly re-train can't wipe the archived seed.
    n_replaced = n_kept_seed = n_new = 0
    if args.merge_into:
        try:
            with open(args.merge_into, encoding="utf-8") as f:
                seed_list = json.load(f)
            merged = {(e["station_id"], e["target"]): e for e in seed_list}
        except (OSError, ValueError, KeyError):
            merged = {}
        for k, fe in fresh.items():
            if k not in merged:
                merged[k] = fe; n_new += 1
            elif fe["n_days"] >= args.replace_min_days:
                merged[k] = fe; n_replaced += 1
            else:
                n_kept_seed += 1
        entries = sorted(merged.values(), key=lambda x: (x["station_id"], x["target"]))
    else:
        entries = sorted(fresh.values(), key=lambda x: (x["station_id"], x["target"]))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"forecast+actual rows used: {rows:,}")
    print(f"resolved (station,target,date) triples: {len(best):,}")
    print(f"fresh entries computed (n>={args.min_days}): {len(fresh)}")
    if args.merge_into:
        print(f"merge: {n_replaced} replaced (fresh n>={args.replace_min_days}), "
              f"{n_kept_seed} kept seed (fresh too thin), {n_new} new")
    print(f"total entries written: {len(entries)} → {args.output}")
    print()
    print(f"  {'station':<8s} {'tgt':<4s} {'n':>3s} {'bias_C':>8s} {'rmse_C':>8s}")
    print("  " + "-" * 40)
    for e in sorted(entries, key=lambda x: -abs(x["bias_c"]))[:20]:
        print(f"  {e['station_id']:<8s} {e['target']:<4s} {e['n_days']:>3d} "
              f"{e['bias_c']:>+8.3f} {e['rmse_c']:>8.3f}")
    if len(entries) > 20:
        print(f"  ... (+{len(entries)-20} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
