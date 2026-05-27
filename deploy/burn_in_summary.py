"""Summarize the state of a slim_daemon burn-in.

Reads the paper-trail JSONL files written by the strategies + harness
and prints a one-page health report. Run after a multi-hour burn-in to
decide whether to promote to VPS.

Usage:
  python deploy/burn_in_summary.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

DATA = Path("data")

LOGS = {
    "publication_window_log.jsonl": "Publication-window snapshots",
    "intraday_log.jsonl": "Lock-in YES decisions",
    "high_bucket_no_log.jsonl": "High-bucket NO fires",
    "v2_conditional_log.jsonl": "V2 preposit decisions",
    "guaranteed_no_buy_log.jsonl": "Layer 7 evaluations",
    "fee_config_cache.json": "Live Polymarket fee config",
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out: list[dict] = []
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


def fmt_age(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - t).total_seconds()
        if age_s < 60:
            return f"{age_s:.0f}s ago"
        if age_s < 3600:
            return f"{age_s/60:.1f}m ago"
        if age_s < 86400:
            return f"{age_s/3600:.1f}h ago"
        return f"{age_s/86400:.1f}d ago"
    except ValueError:
        return iso[:19]


# ── Per-log summary ──────────────────────────────────────────────────
print("=" * 70)
print("Burn-in summary @", datetime.now(timezone.utc).isoformat())
print("=" * 70)

for fname, label in LOGS.items():
    p = DATA / fname
    if fname.endswith(".jsonl"):
        rows = load_jsonl(p)
        if not rows:
            print(f"  [empty]   {label:40s} — {fname}")
            continue
        first_ts = rows[0].get("snapshot_ts_utc") or rows[0].get("ts_utc") or rows[0].get("scan_time_utc")
        last_ts = rows[-1].get("snapshot_ts_utc") or rows[-1].get("ts_utc") or rows[-1].get("scan_time_utc")
        print(f"  {label:40s}  N={len(rows):>5}  first={fmt_age(first_ts):>10}  last={fmt_age(last_ts):>10}")
    else:
        # fee_config_cache.json
        if not p.exists():
            print(f"  [empty]   {label}")
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            print(
                f"  {label:40s}  rate={d.get('taker_fee_rate'):.4f}  "
                f"rebate={d.get('maker_rebate_rate')}  "
                f"source={d.get('source')}  fetched={fmt_age(d.get('fetched_at_utc'))}"
            )
        except (OSError, ValueError):
            print(f"  [unreadable] {label}")


# ── Pub-window detail: WUG status counts ─────────────────────────────
print()
print("=" * 70)
print("Publication-window detail")
print("=" * 70)
pw = load_jsonl(DATA / "publication_window_log.jsonl")
if pw:
    statuses = Counter(r.get("wug_status") for r in pw)
    print(f"  WUG status counts: {dict(statuses)}")
    matched_count = sum(1 for r in pw if r.get("matched_bucket_kind"))
    print(f"  Snapshots with matched_bucket: {matched_count}/{len(pw)}")
    distinct_sk = {(r["station_id"], r["target"], r["target_date"]) for r in pw}
    print(f"  Distinct (station, target, date) tuples covered: {len(distinct_sk)}")


# ── Intraday lock-in detail ──────────────────────────────────────────
print()
print("=" * 70)
print("Lock-in YES detail")
print("=" * 70)
intraday = load_jsonl(DATA / "intraday_log.jsonl")
if intraday:
    decisions = Counter(r.get("decision") for r in intraday)
    print(f"  Decision counts: {dict(decisions)}")
    sources = Counter()
    for r in intraday:
        reason = r.get("reason") or ""
        if "wug" in reason.lower():
            sources["wug"] += 1
        elif "metar" in reason.lower():
            sources["metar"] += 1
        else:
            sources["other"] += 1
    print(f"  Source split: {dict(sources)}")


# ── Layer 7 detail ───────────────────────────────────────────────────
print()
print("=" * 70)
print("Layer 7 detail")
print("=" * 70)
l7 = load_jsonl(DATA / "guaranteed_no_buy_log.jsonl")
if l7:
    results = Counter(r.get("result") for r in l7)
    print(f"  Evaluation outcomes: {dict(results.most_common())}")


# ── High-bucket NO detail ────────────────────────────────────────────
print()
print("=" * 70)
print("High-bucket NO detail")
print("=" * 70)
hbn = load_jsonl(DATA / "high_bucket_no_log.jsonl")
if hbn:
    results = Counter(r.get("result") for r in hbn)
    print(f"  Outcome counts: {dict(results)}")


# ── V2 detail (taker vs maker counterfactual) ────────────────────────
print()
print("=" * 70)
print("V2 conditional preposit detail")
print("=" * 70)
v2 = load_jsonl(DATA / "v2_conditional_log.jsonl")
if v2:
    decisions = Counter(r.get("decision") for r in v2)
    print(f"  Decision counts: {dict(decisions)}")
    placed = [r for r in v2 if r.get("decision") == "placed"]
    if placed:
        avg_maker_price = sum(r.get("maker_intended_price", 0.0) for r in placed) / len(placed)
        taker_asks = [r["taker_no_ask"] for r in placed if r.get("taker_no_ask") is not None]
        avg_taker_ask = sum(taker_asks) / len(taker_asks) if taker_asks else 0.0
        print(f"  Avg maker intended: ${avg_maker_price:.4f}")
        print(f"  Avg taker no_ask:   ${avg_taker_ask:.4f}  (delta = "
              f"{avg_taker_ask-avg_maker_price:+.4f})")


print()
print("=" * 70)
print("Pass/fail signals to look for")
print("=" * 70)
print("  ✓ Publication-window N grows steadily (≥ N stations × 24h / 30min)")
print("  ✓ WUG status is mostly 'ok' (not http_429 / parse_error)")
print("  ✓ Lock-in YES source split is mostly 'wug' (METAR fallback is rare)")
print("  ✓ Layer 7 evaluation count doesn't explode (progressive eval works)")
print("  ✓ No exception spew in daemon stdout")
