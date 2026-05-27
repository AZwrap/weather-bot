"""Analyzer for the publication-window shadow harness.

Reads data/publication_window_log.jsonl (written by
publication_window_log.py) and data/forward_log.jsonl (the resolution
truth) and answers:

  1. Tradable-window distribution: how long past end-of-day-local do
     buckets remain quoted with non-trivial spreads?
  2. METAR↔Polymarket agreement: how often does our final METAR-derived
     extreme land in the same bucket as Polymarket eventually resolved?
  3. Mispricing in the window: at each offset bin, what was the WINNING
     bucket's yes_ask vs the eventual $1 redemption? That delta is the
     ceiling on a Wunderground-race strategy's per-fill edge.
  4. Per-station/per-offset hypothetical PnL: if we bought the winning
     bucket at the first observed yes_ask in each offset bin, after fees.

Run after >= 5 days of harness data have accumulated.

Usage:
  python analyze_publication_window.py [--min-n 10]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


DEFAULT_LOG_PATH = Path("data/publication_window_log.jsonl")
FORWARD_LOG_PATH = Path("data/forward_log.jsonl")

# Polymarket taker fee formula (confirmed empirically 2026-05-25):
#   shares * 0.05 * p * (1 - p)
# At p=0.95 the per-share fee on a YES buy is 0.05 * 0.95 * 0.05 = $0.0024
# i.e. ~0.25% of the $1.00 redemption.
def taker_fee_per_share(price: float) -> float:
    return 0.05 * price * (1.0 - price)


OFFSET_BINS_H = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0),
                 (4.0, 8.0), (8.0, 16.0), (16.0, 36.0)]


def bin_label(offset_h: float) -> str | None:
    for lo, hi in OFFSET_BINS_H:
        if lo <= offset_h < hi:
            return f"+{lo:.1f}–{hi:.1f}h"
    return None


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "min": s[0], "p10": s[int(n * 0.1)],
        "p25": s[int(n * 0.25)], "p50": s[int(n * 0.5)],
        "p75": s[int(n * 0.75)], "p90": s[int(n * 0.9)],
        "max": s[-1],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-n", type=int, default=5,
                   help="Suppress bins / stations with fewer than this many records.")
    p.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    snaps = load_jsonl(args.log_path)
    resolutions = load_jsonl(FORWARD_LOG_PATH)

    if not snaps:
        print(f"[empty] no snapshots at {args.log_path}. "
              f"Run publication_window_log.py periodically first.")
        return 1

    # Build (station_id, target_date) → resolution
    by_resolution: dict[tuple[str, str], dict] = {}
    for r in resolutions:
        sid = r.get("station_id")
        td = r.get("target_date")
        if not sid or not td:
            continue
        if r.get("actual_obs_c") is None:
            continue
        key = (sid, td)
        if key not in by_resolution:
            by_resolution[key] = r

    # Group snapshots by (station, target, target_date)
    per_market: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in snaps:
        sid = s.get("station_id")
        tgt = s.get("target")
        td = s.get("target_date")
        if not (sid and tgt and td):
            continue
        per_market[(sid, tgt, td)].append(s)

    print(f"[pubwin] loaded {len(snaps)} snapshots across "
          f"{len(per_market)} (station, target, date) tuples")
    print(f"[pubwin] {len(by_resolution)} resolved (station, date) pairs in forward_log")

    # ── 1) Tradable-window distribution ───────────────────────────────
    # A bucket "remains tradable" if at the snapshot offset there exists
    # at least one bucket with yes_ask != null AND yes_ask < 0.99 AND
    # yes_bid != null AND (yes_ask - yes_bid) <= 0.10.
    has_active_book: dict[str, list[float]] = defaultdict(list)
    for s in snaps:
        bin_l = bin_label(s.get("offset_h_after_midend", -1))
        if bin_l is None:
            continue
        active = 0
        for b in s.get("buckets", []):
            ya, yb = b.get("yes_ask"), b.get("yes_bid")
            if ya is None or yb is None:
                continue
            if 0.0 < ya < 0.99 and (ya - yb) <= 0.10:
                active += 1
        has_active_book[bin_l].append(1.0 if active > 0 else 0.0)

    print()
    print("=== 1) Tradable-book presence by offset bin ===")
    print(f"{'bin':<14} {'n':>4} {'%any-active':>13}")
    for lo, hi in OFFSET_BINS_H:
        bin_l = f"+{lo:.1f}–{hi:.1f}h"
        xs = has_active_book.get(bin_l, [])
        if not xs:
            continue
        n = len(xs)
        pct = 100.0 * sum(xs) / n
        if n >= args.min_n:
            print(f"{bin_l:<14} {n:>4} {pct:>12.1f}%")

    # ── 2) Bucket agreement: METAR vs Polymarket, and WUG vs Polymarket
    # Since WUG is Polymarket's source, WUG↔PM should be ~100%.
    # METAR↔PM measures how often raw METAR-derived buckets agree with
    # the oracle (the 0.5-1°C disagreement that motivated the WUG fetch).
    metar_agreement: list[bool] = []
    wug_agreement: list[bool] = []
    metar_disagreements: list[tuple[str, str, int | None, int | None]] = []
    wug_disagreements: list[tuple[str, str, int | None, int | None]] = []

    def _bucket_for_value(forward_record: dict, value_c: float) -> tuple[str | None, int | None]:
        return _pm_resolved_bucket(forward_record, value_c)

    for (sid, tgt, td), market_snaps in per_market.items():
        res = by_resolution.get((sid, td))
        if not res or res.get("actual_obs_c") is None:
            continue
        pm_kind, pm_thr = _pm_resolved_bucket(res, res["actual_obs_c"])
        if pm_kind is None:
            continue
        latest = max(market_snaps, key=lambda s: s.get("snapshot_ts_utc", ""))
        # METAR-derived bucket from latest snapshot's metar extreme
        m_c = latest.get("metar_final_extreme_c")
        if m_c is not None:
            m_kind, m_thr = _bucket_for_value(res, m_c)
            if m_kind is not None:
                match = (m_kind == pm_kind and m_thr == pm_thr)
                metar_agreement.append(match)
                if not match:
                    metar_disagreements.append((sid, td, m_thr, pm_thr))
        # WUG-derived bucket from latest snapshot
        wug_c = (latest.get("wug_daily_max_c") if tgt == "max"
                 else latest.get("wug_daily_min_c"))
        if wug_c is not None:
            w_kind, w_thr = _bucket_for_value(res, wug_c)
            if w_kind is not None:
                match = (w_kind == pm_kind and w_thr == pm_thr)
                wug_agreement.append(match)
                if not match:
                    wug_disagreements.append((sid, td, w_thr, pm_thr))

    print()
    print("=== 2) Bucket agreement vs Polymarket oracle ===")
    if metar_agreement:
        rate = 100.0 * sum(1 for a in metar_agreement if a) / len(metar_agreement)
        print(f"  METAR↔PM   N={len(metar_agreement)}   agreement: {rate:.1f}%")
        if metar_disagreements:
            print("    sample disagreements (our bucket | pm bucket):")
            for sid, td, mt, pt in metar_disagreements[:5]:
                print(f"      {sid:6s} {td:10s}  metar:{mt}  pm:{pt}")
    if wug_agreement:
        rate = 100.0 * sum(1 for a in wug_agreement if a) / len(wug_agreement)
        print(f"  WUG↔PM     N={len(wug_agreement)}   agreement: {rate:.1f}%")
        if wug_disagreements:
            print("    sample disagreements (our bucket | pm bucket):")
            for sid, td, mt, pt in wug_disagreements[:5]:
                print(f"      {sid:6s} {td:10s}  wug:{mt}  pm:{pt}")
    if not metar_agreement and not wug_agreement:
        print("  (no joinable resolutions yet)")

    # ── 3) Winning-bucket price by offset (mispricing) ───────────────
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for (sid, tgt, td), market_snaps in per_market.items():
        res = by_resolution.get((sid, td))
        if not res or res.get("actual_obs_c") is None:
            continue
        pm_kind, pm_thr = _pm_resolved_bucket(res, res["actual_obs_c"])
        if pm_kind is None:
            continue
        for s in market_snaps:
            bin_l = bin_label(s.get("offset_h_after_midend", -1))
            if bin_l is None:
                continue
            # Winning bucket's yes_ask at this snapshot
            for b in s.get("buckets", []):
                if b.get("kind") == pm_kind and b.get("threshold") == pm_thr:
                    by_bin[bin_l].append({
                        "yes_ask": b.get("yes_ask"),
                        "yes_bid": b.get("yes_bid"),
                        "sid": sid, "td": td,
                    })
                    break

    print()
    print("=== 3) Winning bucket's YES ask, by offset bin ===")
    print(f"{'bin':<14} {'n':>4} {'p10':>6} {'p50':>6} {'p90':>6} "
          f"{'%<0.97':>8} {'avg-edge':>10}")
    for lo, hi in OFFSET_BINS_H:
        bin_l = f"+{lo:.1f}–{hi:.1f}h"
        rows = [r for r in by_bin.get(bin_l, []) if r["yes_ask"] is not None]
        if len(rows) < args.min_n:
            continue
        asks = [r["yes_ask"] for r in rows]
        p = percentiles(asks)
        pct_under = 100.0 * sum(1 for a in asks if a < 0.97) / len(asks)
        # Avg edge = 1 - mean(ask) - mean(fee)
        avg_ask = statistics.mean(asks)
        avg_fee = statistics.mean(taker_fee_per_share(a) for a in asks)
        avg_edge = 1.0 - avg_ask - avg_fee
        print(f"{bin_l:<14} {len(asks):>4} {p['p10']:>6.3f} {p['p50']:>6.3f} "
              f"{p['p90']:>6.3f} {pct_under:>7.1f}% {avg_edge:>+10.3f}")

    # ── 4) Hypothetical PnL — buy YES at first observed ask per offset bin
    print()
    print("=== 4) Hypothetical PnL: buy YES on winner at FIRST ask seen in bin ===")
    print("  (size = $5 per fire; redeems at $1; fees at p*(1-p)*0.05 per share)")
    print(f"{'bin':<14} {'fires':>5} {'wins':>5} {'gross':>8} {'fees':>7} {'net':>8}")
    fire_amount_usd = 5.0
    summary_total = {"fires": 0, "gross": 0.0, "fees": 0.0, "net": 0.0}
    for lo, hi in OFFSET_BINS_H:
        bin_l = f"+{lo:.1f}–{hi:.1f}h"
        rows = by_bin.get(bin_l, [])
        # Per market: first snapshot in this bin
        first_per_market: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = (r["sid"], r["td"])
            if r["yes_ask"] is None:
                continue
            if r["yes_ask"] >= 0.99:
                continue
            if key not in first_per_market:
                first_per_market[key] = r
        if len(first_per_market) < args.min_n:
            continue
        fires = len(first_per_market)
        wins = fires  # every fire is on the WINNING bucket; redeem at $1
        gross = sum(
            (1.0 - r["yes_ask"]) * (fire_amount_usd / r["yes_ask"])
            for r in first_per_market.values()
        )
        fees = sum(
            taker_fee_per_share(r["yes_ask"]) * (fire_amount_usd / r["yes_ask"])
            for r in first_per_market.values()
        )
        net = gross - fees
        print(f"{bin_l:<14} {fires:>5} {wins:>5} {gross:>+8.2f} {fees:>7.2f} {net:>+8.2f}")
        summary_total["fires"] += fires
        summary_total["gross"] += gross
        summary_total["fees"] += fees
        summary_total["net"] += net

    if summary_total["fires"]:
        print(f"{'TOTAL':<14} {summary_total['fires']:>5} {'':>5} "
              f"{summary_total['gross']:>+8.2f} {summary_total['fees']:>7.2f} "
              f"{summary_total['net']:>+8.2f}")
        print(
            f"  per-fire avg: ${summary_total['net']/summary_total['fires']:+.3f} "
            f"net (= {100.0*summary_total['net']/(fire_amount_usd*summary_total['fires']):+.1f}% ROI)"
        )

    return 0


def _pm_resolved_bucket(forward_record: dict, actual_c: float) -> tuple[str | None, int | None]:
    """Find which bucket Polymarket would have settled given actual_obs_c."""
    from weather_bot.pnl import _rounded_observation, bucket_won
    sid = forward_record.get("station_id", "")
    # Need station unit to round properly. Fallback "C".
    unit = "C"
    try:
        sys.path.insert(0, ".")
        from weather_bot.locations import STATIONS_BY_ID
        s = STATIONS_BY_ID.get(sid)
        if s is not None:
            unit = s.unit
    except Exception:
        pass
    actual_int = _rounded_observation(actual_c, unit)  # type: ignore[arg-type]
    for b in forward_record.get("bucket_snapshots", []):
        k = b.get("kind")
        t = b.get("threshold")
        if k is None or t is None:
            continue
        if bucket_won(k, int(t), actual_int, unit):  # type: ignore[arg-type]
            return k, int(t)
    return None, None


if __name__ == "__main__":
    sys.exit(main())
