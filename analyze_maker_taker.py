#!/usr/bin/env python3
"""Determine whether cohort weather traders are MAKERS or TAKERS — on-chain truth.

The public data-api exposes no maker/taker or fee field (usdcSize == size*price
exactly), so we decode the on-chain OrderFilled event from each trade's receipt.
Polymarket multi-outcome (weather) markets settle via the NegRisk exchange, which
emits OrderFilled(orderHash, maker indexed, taker indexed, ...) — sig 0xd543adfd,
4 topics, topic2=maker topic3=taker. If the trader's padded address is topic2 the
fill was RESTING (maker, $0 fee + rebate); topic3 means they CROSSED (taker, pays fee).

Result 2026-06-13: poligarch / badatmath / sailor82 = 100% MAKER (25/25 each).

Usage:  python analyze_maker_taker.py [n_trades_per_wallet=25]
"""
from __future__ import annotations
import json, sys, urllib.request

RPC = "https://polygon-bor-rpc.publicnode.com"
ORDERFILLED = "0xd543adfd"  # OrderFilled topic0 prefix (NegRisk exchange)
DATA_API = "https://data-api.polymarket.com"

TRADERS = [
    ("poligarch", "0xb40e89677d59665d5188541ad860450a6e2a7cc9"),
    ("badatmath", "0x8fbd7cf5f806f563080864694415829f7229a959"),
    ("sailor82",  "0xbbb72a812cfbc5217d77c0a0018c71f174d3a11a"),
    ("opopv",     "0x116db6298abcdefe06f9f5458c293c7de185fbf1"),
    ("link2-ec86","0xec86a2d3f69015b1a9382e4dfa8695e1b48760e4"),
    ("shyguy1",   "0x1f66796b45581868376365aef54b51eb84184c8d"),
    ("weatherhk", "0x488c725253fc21c7a9ca812030dc2f6343f98c1c"),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, data=body, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def role_in_tx(txhash, padded):
    """maker / taker / None for the trader in this trade's OrderFilled events."""
    try:
        rec = rpc("eth_getTransactionReceipt", [txhash]).get("result")
    except Exception:
        return None
    if not rec:
        return None
    role = None
    for lg in rec["logs"]:
        tp = lg["topics"]
        if len(tp) == 4 and tp[0].lower().startswith(ORDERFILLED):
            if tp[2].lower() == padded:
                return "maker"          # resting order filled
            if tp[3].lower() == padded:
                role = "taker"          # crossed the book
    return role


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print(f"{'trader':12}{'n':>4}{'MAKER':>9}{'TAKER':>9}{'unclass':>9}")
    for label, addr in TRADERS:
        padded = ("0x" + "0" * 24 + addr[2:]).lower()
        acts = get(f"{DATA_API}/activity?user={addr}&limit=300")
        txs, seen = [], set()
        for x in acts:
            if (x.get("type") == "TRADE" and x.get("side") == "BUY"
                    and "temperature" in (x.get("title", "") or "").lower()):
                h = x.get("transactionHash")
                if h and h not in seen:
                    seen.add(h); txs.append(h)
            if len(txs) >= n:
                break
        mk = tk = oth = 0
        for h in txs:
            r = role_in_tx(h, padded)
            mk += r == "maker"; tk += r == "taker"; oth += r is None
        tot = mk + tk
        print(f"{label:12}{len(txs):>4}{f'{mk} ({100*mk//tot if tot else 0}%)':>9}"
              f"{f'{tk} ({100*tk//tot if tot else 0}%)':>9}{oth:>9}")


if __name__ == "__main__":
    main()
