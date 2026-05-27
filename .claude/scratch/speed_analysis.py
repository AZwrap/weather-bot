"""Bot reaction-speed analysis.

Questions answered from on-disk data:
  1. Layer 7 fill-price distribution — how close were we to the $0.99
     cap on actual fires? Near-cap = we arrived late, the market had
     already converged before our cron tick.
  2. METAR fetch latency (per-source) — from data/benchmark_metar_results.jsonl
     written by benchmark_metar_rate_limit.py.
  3. Implied scan cycle time given current parallelism.
"""
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORKTREE = Path(r"C:\Users\Kevin\Claude Bots\Weather_Bot\.claude\worktrees\epic-merkle-391602")
MAIN     = Path(r"C:\Users\Kevin\Claude Bots\Weather_Bot")
SUBMITTED = WORKTREE / "data_archive" / "data" / "submitted_orders.jsonl"
L7_LOG    = WORKTREE / "data_archive" / "data" / "guaranteed_no_buy_log.jsonl"
BENCH     = MAIN / "data" / "benchmark_metar_results.jsonl"


def load_jsonl(p):
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pcts(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {q: s[min(int(n * q), n - 1)] for q in qs}


# ── 1) Layer 7 fill-price distribution ─────────────────────────────
print("=" * 70)
print("1) Layer 7 fill-price distribution")
print("=" * 70)
orders = load_jsonl(SUBMITTED)
l7_fires = [o for o in orders
            if o.get("side") == "NO"
            and (o.get("fill_price") or 0) >= 0.85]
prices = [float(o["fill_price"]) for o in l7_fires]
if prices:
    print(f"  N Layer-7-style NO fills: {len(prices)}")
    print(f"  mean:    {statistics.mean(prices):.4f}")
    print(f"  median:  {statistics.median(prices):.4f}")
    p = pcts(prices)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"  p{int(q*100):>2}:    {p[q]:.4f}")
    near_cap = sum(1 for x in prices if x >= 0.98)
    at_cap = sum(1 for x in prices if x >= 0.99)
    print(f"  >= $0.98: {near_cap}/{len(prices)} ({100*near_cap/len(prices):.0f}%)")
    print(f"  >= $0.99: {at_cap}/{len(prices)} ({100*at_cap/len(prices):.0f}%)")
else:
    print("  (no Layer-7-style NO fills found)")


# ── 2) METAR latency by source ─────────────────────────────────────
print()
print("=" * 70)
print("2) METAR fetch latency (per source)")
print("=" * 70)
bench = load_jsonl(BENCH)
by_source = {}
for b in bench:
    src = b.get("source") or "unknown"
    lat = b.get("latency_ms")
    if lat is None or not b.get("success"):
        continue
    by_source.setdefault(src, []).append(float(lat))
for src, lats in sorted(by_source.items()):
    p = pcts(lats)
    print(f"  [{src}] N={len(lats)}  "
          f"p50={p[0.5]:>5.0f}ms  "
          f"p90={p[0.9]:>5.0f}ms  "
          f"p99={p[0.99]:>5.0f}ms  "
          f"max={max(lats):>5.0f}ms")


# ── 3) Implied scan cycle time ─────────────────────────────────────
print()
print("=" * 70)
print("3) Implied total scan cycle time")
print("=" * 70)
# Stages: gamma events fetch, CLOB prices batch, METAR for N stations
# (parallel), strategy decisions (CPU only, negligible).
# Assumptions from production: ~50 stations, METAR fetch parallel-per-station,
# gamma fetch ~1s, CLOB batch ~2s.
gamma_s = 1.0
clob_s = 2.0
if by_source:
    fastest_src = min(by_source, key=lambda s: pcts(by_source[s])[0.5])
    metar_p90_ms = pcts(by_source[fastest_src])[0.9]
    metar_p50_ms = pcts(by_source[fastest_src])[0.5]
    # Parallel-across-stations: total ≈ p90 of one fetch
    # (assumes >1 parallel slot, which the bot uses)
    metar_parallel_s = metar_p90_ms / 1000.0
    print(f"  Stage budget (parallel where possible):")
    print(f"    gamma events     ~{gamma_s:.1f}s")
    print(f"    CLOB price batch ~{clob_s:.1f}s")
    print(f"    METAR (p90 of {fastest_src}, 1 of N parallel) "
          f"~{metar_parallel_s:.1f}s")
    total_s = gamma_s + clob_s + metar_parallel_s
    print(f"  → Total scan cycle: ~{total_s:.1f}s")
    print()
    print(f"  Implication: cycle time ({total_s:.0f}s) is "
          f"{900/total_s:.0f}× faster than a 15-min cron tick.")
    print(f"  A daemon polling every ~{int(total_s*1.5):.0f}s would react "
          f"~{(15*60)/(total_s*1.5):.0f}× faster than cron-15.")


# ── 4) Layer 7 reaction lag indicator ──────────────────────────────
# If we are "late" to Layer 7 fires, fill_price clusters at $0.99 cap.
# Distribution skew toward $0.99 = sign we missed earlier $0.85-$0.95
# opportunities.
print()
print("=" * 70)
print("4) Layer 7 lateness indicator")
print("=" * 70)
if prices:
    bins = Counter()
    for p in prices:
        if p < 0.90:
            bins["<0.90"] += 1
        elif p < 0.92:
            bins["0.90-0.92"] += 1
        elif p < 0.95:
            bins["0.92-0.95"] += 1
        elif p < 0.97:
            bins["0.95-0.97"] += 1
        elif p < 0.99:
            bins["0.97-0.99"] += 1
        else:
            bins[">=0.99"] += 1
    order = ["<0.90", "0.90-0.92", "0.92-0.95", "0.95-0.97",
             "0.97-0.99", ">=0.99"]
    print(f"  Layer 7 fill-price bins:")
    for k in order:
        v = bins.get(k, 0)
        pct = 100 * v / len(prices)
        bar = "█" * int(pct / 2)
        print(f"    {k:<10} {v:>4} ({pct:>5.1f}%) {bar}")

    if bins.get(">=0.99", 0) + bins.get("0.97-0.99", 0) > 0.5 * len(prices):
        print()
        print("  >50% of fills clustered above $0.97 → we WERE late.")
        print("  A faster reaction (sub-minute daemon) should pull the")
        print("  fill-price distribution down to $0.92-$0.96 range, ")
        print("  improving per-fire EV by 3-7pp gross of fees.")
    else:
        print("  Distribution looks reasonable; arriving in time on most fires.")


# ── 5) Histogram of L7 evaluations (placed vs skipped reasons) ──────
print()
print("=" * 70)
print("5) Layer 7 evaluation outcome distribution")
print("=" * 70)
results = Counter()
for line in open(L7_LOG, "r", encoding="utf-8"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    res = r.get("result", "unknown")
    results[res] += 1
total = sum(results.values())
print(f"  Total evaluations: {total}")
for k, v in results.most_common(10):
    pct = 100 * v / total
    print(f"    {k:<32s} {v:>8}  ({pct:>5.1f}%)")
