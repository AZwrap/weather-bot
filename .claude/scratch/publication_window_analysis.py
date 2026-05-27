"""Measure: do Polymarket weather markets stay tradable AFTER the
resolution day ends (= after the daily max/min is mathematically known)?

For each market we redeemed:
  - midend_local_ts = UTC timestamp of midnight at end of resolution day
                      in station-local tz (mid-May 2026 DST hard-coded)
  - last_trade_h    = (our last Buy/Sell timestamp − midend_local_ts) / 3600
  - first_redeem_h  = (first Redeem timestamp     − midend_local_ts) / 3600

Positive last_trade_h = we ourselves traded AFTER resolution day ended
= direct proof the market stayed tradable past midnight local.

Positive first_redeem_h = settlement landed AFTER end-of-day-local.
Settlement gap to last-trade lower-bounds how late the market remained
tradable for someone (we'd need full PM book history to know who, but
this gap is what an edge-window strategy would target).
"""
import csv
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")

# UTC offset in hours for each city, valid mid-May 2026 (DST baked in).
CITY_OFFSET_H = {
    "Tokyo": 9, "Singapore": 8, "Seoul": 9, "Hong Kong": 8,
    "Beijing": 8, "Shanghai": 8, "Shenzhen": 8, "Guangzhou": 8,
    "Chengdu": 8, "Wuhan": 8, "Chongqing": 8, "Qingdao": 8,
    "Manila": 8, "Kuala Lumpur": 8, "Taipei": 8, "Busan": 9,
    "Jakarta": 7, "Bangkok": 7,
    "New Delhi": 5.5, "Mumbai": 5.5, "Lucknow": 5.5,
    "Dubai": 4, "Riyadh": 3, "Jeddah": 3,
    "Tel Aviv": 3, "Cairo": 2, "Istanbul": 3, "Helsinki": 3,
    "Moscow": 3,
    # Europe — May = DST in nearly all
    "London": 1, "Lisbon": 1,
    "Paris": 2, "Berlin": 2, "Rome": 2, "Madrid": 2, "Amsterdam": 2,
    "Munich": 2, "Milan": 2,
    # US — May = DST
    "New York City": -4, "Miami": -4, "Atlanta": -4, "Boston": -4,
    "Washington, D.C.": -4, "Washington DC": -4,
    "Chicago": -5, "Dallas": -5, "Houston": -5, "Austin": -5,
    "Denver": -6, "Seattle": -7, "Los Angeles": -7,
    "San Francisco": -7, "Phoenix": -7,  # AZ no DST
    # Latin America
    "Mexico City": -6, "Panama City": -5,
    "Bogota": -5, "Bogotá": -5, "Lima": -5, "Caracas": -4,
    "Buenos Aires": -3, "São Paulo": -3, "Sao Paulo": -3,
    # Africa
    "Lagos": 1, "Cape Town": 2,
    # Oceania
    "Toronto": -4, "Sydney": 10, "Wellington": 12, "Auckland": 12,
}


def midend_local_unix(d, offset_h):
    """UTC unix timestamp of midnight at END of date `d` in local tz."""
    midend_local = datetime(d.year, d.month, d.day) + timedelta(days=1)
    midend_utc = midend_local - timedelta(hours=offset_h)
    return midend_utc.replace(tzinfo=timezone.utc).timestamp()


rows = []
for f in ["data/polymarket_export.csv", "data/polymarket_export_day3.csv"]:
    with open(f, encoding="utf-8-sig") as fh:
        rows.extend(csv.DictReader(fh))

pat_tail_hi = re.compile(r"(highest|lowest) temperature in (.+?) be (\d+).(C|F) or higher on (\w+ \d+)")
pat_tail_lo = re.compile(r"(highest|lowest) temperature in (.+?) be (\d+).(C|F) or (?:lower|below) on (\w+ \d+)")
pat_mid_one = re.compile(r"(highest|lowest) temperature in (.+?) be (\d+).(C|F) on (\w+ \d+)")
pat_mid_rng = re.compile(r"(highest|lowest) temperature in (.+?) be between (\d+)-(\d+).(C|F) on (\w+ \d+)")

for r in rows:
    name = r["marketName"]
    for p, kind in (
        (pat_tail_hi, "high_tail"),
        (pat_tail_lo, "low_tail"),
        (pat_mid_rng, "mid_rng"),
        (pat_mid_one, "mid_one"),
    ):
        m = p.search(name)
        if m:
            r["_city"] = m.group(2)
            r["_date"] = m.group(6) if kind == "mid_rng" else m.group(5)
            r["_kind"] = kind
            break

markets = defaultdict(lambda: {"trades": [], "redeems": []})
for r in rows:
    if "_city" not in r:
        continue
    key = (r["_city"], r["_date"], r["marketName"])
    if r["action"] == "Redeem":
        markets[key]["redeems"].append(int(r["timestamp"]))
    elif r["action"] in ("Buy", "Sell"):
        markets[key]["trades"].append(int(r["timestamp"]))

results = []
unknown_tz = Counter()
for (city, date_str, name), m in markets.items():
    off = CITY_OFFSET_H.get(city)
    if off is None:
        unknown_tz[city] += 1
        continue
    if not m["redeems"]:
        continue
    try:
        d = datetime.strptime(date_str + " 2026", "%B %d %Y").date()
    except ValueError:
        try:
            d = datetime.strptime(date_str + " 2026", "%b %d %Y").date()
        except ValueError:
            continue
    midend_ts = midend_local_unix(d, off)
    last_trade = max(m["trades"]) if m["trades"] else None
    first_redeem = min(m["redeems"])
    results.append({
        "city": city,
        "date": date_str,
        "midend_utc": datetime.fromtimestamp(midend_ts, tz=timezone.utc).isoformat(),
        "last_trade_h": ((last_trade - midend_ts) / 3600) if last_trade else None,
        "first_redeem_h": (first_redeem - midend_ts) / 3600,
        "redeem_minus_last_trade_h": (
            (first_redeem - last_trade) / 3600 if last_trade else None
        ),
    })


def percentiles(xs, labels=("p10", "p25", "p50", "p75", "p90")):
    if not xs:
        return
    n = len(xs)
    s = sorted(xs)
    out = {}
    for label, frac in zip(labels, (0.1, 0.25, 0.5, 0.75, 0.9)):
        out[label] = s[int(n * frac)]
    out["min"], out["max"] = s[0], s[-1]
    return out


last_trade_hrs = [r["last_trade_h"] for r in results if r["last_trade_h"] is not None]
redeem_hrs = [r["first_redeem_h"] for r in results]
gap_hrs = [r["redeem_minus_last_trade_h"] for r in results if r["redeem_minus_last_trade_h"] is not None]

print(f"N markets analyzed (timezone known + redeemed): {len(results)}")
print(f"N with our trades:                              {len(last_trade_hrs)}")
print()
print("Our LAST TRADE timestamp − end-of-resolution-day-local (hours):")
print("  (negative = before midnight local; positive = after midnight local)")
p = percentiles(last_trade_hrs)
if p:
    for k in ("p10", "p25", "p50", "p75", "p90", "min", "max"):
        print(f"  {k}: {p[k]:+6.2f}h")
    n_post = sum(1 for x in last_trade_hrs if x > 0)
    print(f"  POST resolution-day-end: {n_post}/{len(last_trade_hrs)} "
          f"({100*n_post/len(last_trade_hrs):.0f}%)")
    n_close = sum(1 for x in last_trade_hrs if -2 <= x <= 0)
    print(f"  Within 2h BEFORE end-of-day-local: {n_close}/{len(last_trade_hrs)} "
          f"({100*n_close/len(last_trade_hrs):.0f}%)")

print()
print("FIRST REDEEM timestamp − end-of-resolution-day-local (hours):")
p = percentiles(redeem_hrs)
if p:
    for k in ("p10", "p25", "p50", "p75", "p90", "min", "max"):
        print(f"  {k}: {p[k]:+6.2f}h")

print()
print("REDEEM − LAST TRADE gap (hours), per market with both:")
p = percentiles(gap_hrs)
if p:
    for k in ("p10", "p25", "p50", "p75", "p90", "min", "max"):
        print(f"  {k}: {p[k]:6.2f}h")
    n_long = sum(1 for x in gap_hrs if x > 6)
    print(f"  Settlement landed >6h after our last trade: {n_long}/{len(gap_hrs)} "
          f"({100*n_long/len(gap_hrs):.0f}%)")

print()
print("Sample of markets where OUR last trade was AFTER end-of-day-local:")
late = sorted([r for r in results if r["last_trade_h"] and r["last_trade_h"] > 0],
              key=lambda r: -r["last_trade_h"])
for r in late[:10]:
    print(f"  {r['city']:20s} {r['date']:8s}  last_trade {r['last_trade_h']:+6.1f}h  "
          f"first_redeem {r['first_redeem_h']:+6.1f}h")

print()
print("Sample of markets where settlement landed >12h after our last trade:")
slow = sorted(
    [r for r in results if r["redeem_minus_last_trade_h"] and r["redeem_minus_last_trade_h"] > 12],
    key=lambda r: -r["redeem_minus_last_trade_h"],
)
for r in slow[:10]:
    print(f"  {r['city']:20s} {r['date']:8s}  last_trade {r['last_trade_h']:+6.1f}h  "
          f"gap-to-redeem {r['redeem_minus_last_trade_h']:5.1f}h")

if unknown_tz:
    print()
    print(f"Unmapped cities (skipped): {dict(unknown_tz.most_common(10))}")
