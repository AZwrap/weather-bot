"""Execution client wrapping py-clob-client.

Two construction paths:

  ExecutionClient.dry_run(config)
      Pure simulator. Safe to instantiate without py-clob-client installed.
      submit_order() returns a fake-success OrderResult and prints what it
      would have done. Use for development and CI.

  ExecutionClient.from_env(config)
      Real client. Requires:
        - `pip install py-clob-client`
        - POLYMARKET_PRIVATE_KEY env var (the bot wallet's private key)
        - Optionally POLYMARKET_API_KEY / SECRET / PASSPHRASE if you've
          already created API credentials; otherwise they're derived from
          the private key on first use.

The real client path is intentionally minimal — it's a skeleton you complete
once you've passed all the items in safety.py's pre-flight checklist. Filling
it in is mechanical but the consequences of a bug are real money. Do not
flip the switch lightly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..scanner import TradeSignal
from .safety import TradingConfig

if TYPE_CHECKING:  # py-clob-client is optional
    from py_clob_client.client import ClobClient


Side = Literal["YES", "NO", "BUY", "SELL"]


@dataclass
class OrderResult:
    ok: bool
    order_id: str | None
    side: str            # "YES" or "NO" (the side we bought)
    token_id: str        # Polymarket binary outcome token id
    fill_price: float    # the price we paid
    size_usd: float      # USD invested
    shares: float        # 1/fill_price * size_usd
    dry_run: bool
    message: str = ""    # info or error detail


class ExecutionClient:
    """Wraps a py-clob-client.ClobClient with safety-aware submit/cancel/balance."""

    def __init__(self, config: TradingConfig, clob: "ClobClient | None" = None):
        self.config = config
        self._clob = clob

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def dry_run(cls, config: TradingConfig) -> "ExecutionClient":
        return cls(config=config, clob=None)

    @classmethod
    def from_env(cls, config: TradingConfig) -> "ExecutionClient":
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import POLYGON
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "py-clob-client is not installed. Install with:\n"
                "    .venv/bin/pip install py-clob-client\n"
                "and re-run."
            ) from exc

        host = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            raise RuntimeError(
                "POLYMARKET_PRIVATE_KEY env var is required. "
                "Set it from a SystemD EnvironmentFile (chmod 600), never in code."
            )

        funder = os.environ.get("POLYMARKET_PROXY_ADDRESS")  # optional for proxy wallets
        clob = ClobClient(
            host=host,
            key=private_key,
            chain_id=POLYGON,
            funder=funder,
            signature_type=int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "2")),
        )

        # API credentials: derive from the private key (or use existing env vars).
        api_key = os.environ.get("POLYMARKET_API_KEY")
        api_secret = os.environ.get("POLYMARKET_API_SECRET")
        api_pass = os.environ.get("POLYMARKET_API_PASSPHRASE")
        if api_key and api_secret and api_pass:
            from py_clob_client.clob_types import ApiCreds  # type: ignore[import-not-found]
            clob.set_api_creds(ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass))
        else:
            clob.set_api_creds(clob.create_or_derive_api_creds())

        return cls(config=config, clob=clob)

    # ── Read-only helpers ───────────────────────────────────────────────

    def get_balance_usdc(self) -> float | None:
        """Return USDC balance available for trading on the proxy. None in dry-run."""
        if self._clob is None:
            return None
        try:  # pragma: no cover — depends on live network
            bal = self._clob.get_balance_allowance({"asset_type": "COLLATERAL"})
            # Some library versions return float USDC, others return wei-like int.
            return float(bal.get("balance", 0.0)) if isinstance(bal, dict) else float(bal)
        except Exception as exc:
            print(f"!! get_balance_usdc failed: {exc}")
            return None

    def get_open_orders(self) -> list[dict]:
        if self._clob is None:
            return []
        try:  # pragma: no cover
            return list(self._clob.get_orders())
        except Exception as exc:
            print(f"!! get_open_orders failed: {exc}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        if self._clob is None:
            print(f"  [DRY RUN] would cancel order {order_id}")
            return True
        try:  # pragma: no cover
            self._clob.cancel(order_id)
            return True
        except Exception as exc:
            print(f"!! cancel_order failed: {exc}")
            return False

    # ── Submit ──────────────────────────────────────────────────────────

    def submit_order(self, signal: TradeSignal) -> OrderResult:
        """Submit a single order derived from a TradeSignal.

        Always prints what it intends to do. In dry-run mode it returns
        a synthetic OrderResult; in live mode it submits via py-clob-client.

        IMPORTANT: this does NOT re-validate against TradingConfig. Validation
        belongs in `place_orders.py` so a single failed gate aborts the whole
        batch rather than silently dropping individual orders.
        """
        side = signal.side
        token_id = signal.token_id
        fill_price = signal.fill_price
        size_usd = signal.position_usd
        shares = size_usd / fill_price if fill_price > 0 else 0.0

        descriptor = (
            f"{signal.station.name} [{signal.target}] {signal.target_date} "
            f"{signal.bucket_label} — {side} @ {fill_price:.3f}, "
            f"size ${size_usd:.2f} ({shares:.2f} shares), "
            f"edge {signal.edge:+.1%}"
        )

        if self._clob is None:
            print(f"  [DRY RUN] {descriptor}")
            return OrderResult(
                ok=True, order_id="dry-run", side=side, token_id=token_id,
                fill_price=fill_price, size_usd=size_usd, shares=shares,
                dry_run=True, message="dry-run (no order submitted)",
            )

        # ── LIVE ORDER — implementation TODO ────────────────────────────
        # The following is the SHAPE of the call, not a complete implementation.
        # Filling it in safely requires:
        #   * Choosing order type: GTC for resting, FOK for fill-or-kill.
        #     For a scanner-driven bot, FOK at the ask is safer (no resting risk).
        #   * Choosing buy vs sell semantics correctly. On Polymarket's CLOB,
        #     buying NO is implemented by buying the no_token_id (Yes share of
        #     the complementary outcome) — already what TradeSignal.token_id
        #     resolves to.
        #   * Handling partial fills, post-only rejection, and rate limiting.
        #   * Logging the resulting order_id to a persistent trade ledger so
        #     reconciliation against fills/balances is possible.
        #
        # Pseudocode:
        #     from py_clob_client.clob_types import OrderArgs, OrderType
        #     args = OrderArgs(
        #         token_id=token_id,
        #         price=fill_price,
        #         size=shares,
        #         side="BUY",                  # always BUY in this convention
        #     )
        #     signed = self._clob.create_order(args)
        #     resp = self._clob.post_order(signed, OrderType.FOK)
        #     return OrderResult(ok=resp.get("success", False),
        #                        order_id=resp.get("orderID"),
        #                        side=side, token_id=token_id,
        #                        fill_price=fill_price, size_usd=size_usd,
        #                        shares=shares, dry_run=False,
        #                        message=str(resp))
        raise NotImplementedError(
            "Live order submission is intentionally left as a TODO. "
            "Complete this method only after the safety checklist in "
            "safety.py is satisfied. Run with --dry-run until then."
        )
