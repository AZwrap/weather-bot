"""Cancel all open Polymarket orders. Run once on VPS as part of shutdown."""
import os
import subprocess
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.constants import POLYGON


def env_from_file(path: str) -> dict[str, str]:
    out = subprocess.check_output(["bash", "-c", f". {path} && env"]).decode()
    d = {}
    for ln in out.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k] = v
    return d


def main() -> None:
    key = os.environ.get("POLY_PRIVATE_KEY")
    funder = os.environ.get("POLY_FUNDER_ADDRESS")
    if not key:
        env = env_from_file("/root/.env.polymarket")
        key = env.get("POLY_PRIVATE_KEY", "")
        funder = env.get("POLY_FUNDER_ADDRESS", "")
    if not key:
        print("!! no POLY_PRIVATE_KEY found")
        return

    c = ClobClient(
        "https://clob.polymarket.com",
        key=key, chain_id=POLYGON,
        signature_type=3, funder=funder,
    )
    c.set_api_creds(c.create_or_derive_api_key())

    print("=== open orders ===")
    orders = c.get_open_orders()
    print(f"count: {len(orders)}")
    for o in orders[:20]:
        oid = str(o.get("id") or o.get("orderID") or "?")
        side = o.get("side", "?")
        sz = o.get("original_size", "?")
        px = o.get("price", "?")
        print(f"  {oid[:18]}... side={side} size={sz} price={px}")

    if not orders:
        print("nothing to cancel")
        return

    print()
    print("=== cancelling all ===")
    try:
        result = c.cancel_all()
        print(f"cancel_all result: {result}")
    except Exception as exc:
        print(f"cancel_all err: {exc}")
        for o in orders:
            oid = o.get("id") or o.get("orderID")
            if not oid:
                continue
            try:
                c.cancel_order(oid)
                print(f"  cancelled {str(oid)[:18]}")
            except Exception as exc2:
                print(f"  err {str(oid)[:18]}: {exc2}")

    print()
    print("=== verify ===")
    remaining = c.get_open_orders()
    print(f"remaining open orders: {len(remaining)}")


if __name__ == "__main__":
    main()
