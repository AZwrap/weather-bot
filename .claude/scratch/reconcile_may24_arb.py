"""Reconcile 6 stuck RKSI May-24 arb positions.

Polymarket confirmed:
  market 2328287: YES won (prices=["1","0"])
  market 2328288: NO won (prices=["0","1"])

But we don't know which market_id maps to 24°C vs 25°C+ from the API alone.
We DO know from poll_resolutions today's output:
  RKSI 2026-05-24 25°C or higher → bucket_won (YES won)
  RKSI 2026-05-24 24°C → bucket_lost (YES lost)

So: 25°C+ positions = WIN, 24°C positions = LOSS.
"""
import json
import os
import tempfile
from datetime import datetime, timezone

p = json.load(open("data/portfolio.json"))
positions = p["positions"]

fixed = []
for pos in positions:
    if pos.get("station_id") != "RKSI": continue
    if pos.get("target_date") != "2026-05-24": continue
    if pos.get("status") != "filled": continue

    bucket = pos.get("bucket_label", "")
    shares = pos.get("shares", 0.0)
    position_usd = pos.get("position_usd", 0.0)

    if bucket == "25°C or higher":
        # YES won → payout = shares × $1, pnl = payout - position_usd
        realized = round(shares * 1.0 - position_usd, 4)
    elif bucket == "24°C":
        # YES lost → payout = 0, pnl = -position_usd
        realized = round(-position_usd, 4)
    else:
        print(f"  UNEXPECTED bucket: {bucket}")
        continue

    pos["status"] = "resolved"
    pos["resolved_at"] = "2026-05-24T15:30:00+00:00"
    pos["realized_pnl"] = realized
    pos["last_modified_utc"] = datetime.now(timezone.utc).isoformat()
    sub = (pos.get("submitted_at") or "")[:19]
    fixed.append((bucket, realized, sub))
    print(f"  reconciled {bucket:18} sub={sub} sh={shares:.2f} entry={pos.get('entry_price',0):.3f} → pnl={realized:+.2f}")

if fixed:
    tmp = "data/portfolio.json.tmp_reconcile"
    with open(tmp, "w") as f:
        json.dump(p, f, default=str, indent=2)
    os.replace(tmp, "data/portfolio.json")
    print(f"\nreconciled {len(fixed)} positions; net = {sum(r for _,r,_ in fixed):+.2f}")
else:
    print("nothing to reconcile")
