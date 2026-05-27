"""Reconcile 6 stuck May-25 positions by querying Polymarket directly.

Logic:
  1. For each filled position with target_date=2026-05-25 and station in
     {RKSI, ZSPD, KLGA}, fetch market state from Gamma API.
  2. If market is closed, determine winning outcome from `outcomePrices`.
  3. Compute realized PnL based on position side + market outcome.
  4. Mark resolved in portfolio.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

PORTFOLIO_PATH = Path("data/portfolio.json")
GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_market_state(client: httpx.Client, condition_id: int) -> dict | None:
    """Query the Gamma API for a specific market by condition/market ID."""
    try:
        # Try both endpoints: by ID and by condition_id
        r = client.get(f"{GAMMA_BASE}/markets/{condition_id}", timeout=20.0)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        print(f"  !! gamma fetch {condition_id}: {exc}")
    return None


def find_market_by_token(client: httpx.Client, token_id: str) -> dict | None:
    """Fallback: query Gamma /markets with token_id filter."""
    try:
        r = client.get(
            f"{GAMMA_BASE}/markets",
            params={"clob_token_ids": token_id},
            timeout=20.0,
        )
        if r.status_code == 200:
            mkts = r.json()
            if isinstance(mkts, list) and mkts:
                return mkts[0]
    except Exception as exc:
        print(f"  !! gamma by-token fetch {token_id[:20]}: {exc}")
    return None


def main() -> int:
    p = json.load(open(PORTFOLIO_PATH))
    stuck = [
        pos for pos in p["positions"]
        if pos.get("status") == "filled"
        and pos.get("target_date") == "2026-05-25"
        and pos.get("station_id") in {"RKSI", "ZSPD", "KLGA"}
    ]
    print(f"Stuck positions: {len(stuck)}")
    if not stuck:
        return 0

    total_pnl = 0.0
    n_resolved = 0
    with httpx.Client() as client:
        for pos in stuck:
            station = pos["station_id"]
            bucket = pos["bucket_label"]
            side = pos["side"]
            shares = float(pos["shares"])
            entry = float(pos["entry_price"])
            token_id = pos.get("token_id", "")
            market_id = pos.get("market_id")

            # Fetch market state via token_id (most reliable)
            mkt = find_market_by_token(client, token_id)
            if not mkt:
                # Fallback: by market_id
                if market_id:
                    mkt = fetch_market_state(client, market_id)
            if not mkt:
                print(f"  ✗ {station} {bucket}: no Gamma data found")
                continue

            closed = mkt.get("closed", False)
            if not closed:
                print(f"  ⏸ {station} {bucket}: market not yet closed on Polymarket")
                continue

            # outcomePrices is a JSON-encoded string list: ["1","0"] or ["0","1"]
            prices_raw = mkt.get("outcomePrices") or ""
            if isinstance(prices_raw, str):
                try:
                    prices = json.loads(prices_raw)
                except json.JSONDecodeError:
                    prices = []
            else:
                prices = prices_raw
            if not prices or len(prices) < 2:
                print(f"  ✗ {station} {bucket}: outcomePrices unparseable: {prices_raw}")
                continue

            # outcomes is typically ["Yes","No"] — convention
            outcomes = mkt.get("outcomes")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = ["Yes", "No"]

            # Determine YES outcome from prices: ["1","0"] = YES wins; ["0","1"] = NO wins
            yes_won = float(prices[0]) >= 0.5

            # Compute PnL
            if side == "YES":
                if yes_won:
                    realized = shares * (1.0 - entry)  # win: pay $1 - entry per share
                else:
                    realized = -shares * entry  # lose: lost entry per share
            else:  # NO
                if yes_won:
                    realized = -shares * entry  # NO loses
                else:
                    realized = shares * (1.0 - entry)  # NO wins

            outcome_str = "YES wins" if yes_won else "NO wins"
            result_str = "WIN" if realized > 0 else "LOSS"
            print(
                f"  {result_str:4} {station} {bucket:16} side={side} sh={shares:6.2f} "
                f"entry=${entry:.3f} → {outcome_str} → pnl ${realized:+7.2f}"
            )
            total_pnl += realized

            # Mark resolved
            pos["status"] = "resolved"
            pos["resolved_at"] = datetime.now(timezone.utc).isoformat()
            pos["realized_pnl"] = round(realized, 4)
            pos["last_modified_utc"] = datetime.now(timezone.utc).isoformat()
            n_resolved += 1

    if n_resolved > 0:
        # Atomic save
        tmp_path = PORTFOLIO_PATH.with_suffix(".json.tmp_reconcile_may25")
        with open(tmp_path, "w") as f:
            json.dump(p, f, default=str, indent=2)
        import os
        os.replace(tmp_path, PORTFOLIO_PATH)
        print()
        print(f"Reconciled {n_resolved} positions; net PnL ${total_pnl:+.2f}")
    else:
        print("Nothing to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
