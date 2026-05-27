"""Execution client wrapping py-clob-client (SDK v2 since 2026-05-11).

Two construction paths:

  ExecutionClient.dry_run(config)
      Pure simulator. Safe to instantiate without py-clob-client installed.
      submit_order() returns a fake-success OrderResult and prints what it
      would have done. Use for development and CI.

  ExecutionClient.from_env(config)
      Real client. Requires:
        - `pip install py_clob_client_v2`  (v1 archived 2026-05-11)
        - POLY_PRIVATE_KEY env var (the bot wallet's EOA private key)
        - POLY_FUNDER_ADDRESS env var (DepositWallet contract address)
        - POLY_SIGNATURE_TYPE env var (defaults to 3 = POLY_1271)
        - Optionally POLY_API_KEY / SECRET / PASSPHRASE if you've already
          created API credentials; otherwise they're derived on first use.

Both POLY_* (canonical, post-rename) and POLYMARKET_* (legacy) env var
prefixes are accepted for backward compat during the transition.

See polymarket_sdk_v2_migration.md for context on the v1 → v2 migration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..scanner import TradeSignal
from .safety import TradingConfig

if TYPE_CHECKING:  # py-clob-client is optional
    from py_clob_client_v2.client import ClobClient


Side = Literal["YES", "NO", "BUY", "SELL"]


@dataclass
class OrderResult:
    ok: bool
    order_id: str | None
    side: str            # "YES" or "NO" (the side we bought)
    token_id: str        # Polymarket binary outcome token id
    fill_price: float    # the AVG fill price (what we actually paid per share)
                         # — distinct from limit_price (the cap we submitted).
                         # Set from Polymarket response takingAmount/makingAmount
                         # if available, else falls back to the limit price.
    size_usd: float      # USD invested
    shares: float        # shares actually filled (from response if available;
                         # else target_shares = size_usd / fill_price)
    dry_run: bool
    message: str = ""    # info or error detail
    limit_price: float | None = None  # the LIMIT we submitted (= ceiling for FAK
                                       # depth-walks). Distinct from fill_price.
    target_shares: float = 0.0         # what we ASKED for, before any partial-fill


class ExecutionClient:
    """Wraps a py-clob-client.ClobClient with safety-aware submit/cancel/balance."""

    def __init__(self, config: TradingConfig, clob: "ClobClient | None" = None):
        self.config = config
        self._clob = clob

    @property
    def is_dry_run(self) -> bool:
        """True if this client cannot submit live orders. Use to gate
        fill-polling and similar reconciliation steps that need a real
        Polymarket API connection."""
        return self._clob is None

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def dry_run(cls, config: TradingConfig) -> "ExecutionClient":
        """Factory for a dry-run (non-submitting) client. Safe to use
        without py-clob-client installed."""
        return cls(config=config, clob=None)

    @classmethod
    def from_env(cls, config: TradingConfig) -> "ExecutionClient":
        """Build a live ExecutionClient from environment variables.

        Required env (set via SystemD EnvironmentFile, chmod 600):
            POLY_PRIVATE_KEY            — bot wallet EOA private key
            POLY_FUNDER_ADDRESS         — DepositWallet contract address
                                          (the wallet that holds Polymarket
                                          Cash; visible in Polymarket UI)
            POLY_SIGNATURE_TYPE         — should be "3" (POLY_1271 for
                                          DepositWallet path); default 3
            POLY_API_KEY,
            POLY_API_SECRET,
            POLY_API_PASSPHRASE         — L2 API creds; derived from
                                          private key on first run if absent

        Uses py_clob_client_v2 (renamed from py-clob-client v1, archived
        2026-05-11). See polymarket_sdk_v2_migration.md.
        """
        try:
            # py_clob_client_v2 is the post-2026-05-11 package; v1 was
            # archived. The structure is largely the same, with stronger
            # types and SignatureTypeV2 enum.
            from py_clob_client_v2.client import ClobClient
            from py_clob_client_v2.constants import POLYGON
        except ImportError:
            # Fallback to v1 if v2 isn't installed yet — useful during
            # transition. Remove this fallback once v2 is stable.
            try:
                from py_clob_client.client import ClobClient  # type: ignore
                from py_clob_client.constants import POLYGON  # type: ignore
                print("!! Using legacy py_clob_client v1 — install "
                      "py_clob_client_v2 for full DepositWallet support")
            except ImportError as exc:
                raise RuntimeError(
                    "Neither py_clob_client_v2 nor py-clob-client is installed.\n"
                    "Install with:\n"
                    "    .venv/bin/pip install py_clob_client_v2\n"
                    "and re-run."
                ) from exc

        host = os.environ.get("POLY_CLOB_HOST", "https://clob.polymarket.com")
        # Accept both POLY_PRIVATE_KEY (canonical) and the legacy
        # POLYMARKET_PRIVATE_KEY name during transition.
        private_key = os.environ.get("POLY_PRIVATE_KEY") or os.environ.get(
            "POLYMARKET_PRIVATE_KEY"
        )
        if not private_key:
            raise RuntimeError(
                "POLY_PRIVATE_KEY env var is required. "
                "Set it from a SystemD EnvironmentFile (chmod 600), never in code."
            )

        # DepositWallet address — the Polymarket-managed wallet that
        # holds your Cash balance. Different from your EOA. Required for
        # signature_type=3 (POLY_1271). Accept both new and legacy names.
        funder = os.environ.get("POLY_FUNDER_ADDRESS") or os.environ.get(
            "POLYMARKET_PROXY_ADDRESS"
        )
        if not funder:
            raise RuntimeError(
                "POLY_FUNDER_ADDRESS env var is required (DepositWallet "
                "contract address). Find it in the Polymarket UI under "
                "Deposit, or in your EOA's PolygonScan transaction history "
                "as the recipient of your USDC.e deposit."
            )

        # Default signature_type=3 (POLY_1271, the DepositWallet path).
        # Legacy wallets used type 0/1/2.
        sig_type = int(
            os.environ.get("POLY_SIGNATURE_TYPE")
            or os.environ.get("POLYMARKET_SIGNATURE_TYPE")
            or "3"
        )

        clob = ClobClient(
            host=host,
            key=private_key,
            chain_id=POLYGON,
            funder=funder,
            signature_type=sig_type,
        )

        # API credentials: prefer explicit env vars; derive from key as fallback.
        api_key = os.environ.get("POLY_API_KEY") or os.environ.get("POLYMARKET_API_KEY")
        api_secret = os.environ.get("POLY_API_SECRET") or os.environ.get("POLYMARKET_API_SECRET")
        api_pass = os.environ.get("POLY_API_PASSPHRASE") or os.environ.get(
            "POLYMARKET_API_PASSPHRASE"
        )
        if api_key and api_secret and api_pass:
            try:
                from py_clob_client_v2.clob_types import ApiCreds  # type: ignore
            except ImportError:
                from py_clob_client.clob_types import ApiCreds  # type: ignore
            clob.set_api_creds(ApiCreds(
                api_key=api_key, api_secret=api_secret, api_passphrase=api_pass
            ))
        else:
            # SDK v2 renamed create_or_derive_api_creds → create_or_derive_api_key.
            # Try v2 name first, fall back to v1 for legacy installs.
            derive = getattr(
                clob,
                "create_or_derive_api_key",
                getattr(clob, "create_or_derive_api_creds", None),
            )
            if derive is None:
                raise RuntimeError(
                    "ClobClient has neither create_or_derive_api_key (v2) "
                    "nor create_or_derive_api_creds (v1). SDK version mismatch."
                )
            clob.set_api_creds(derive())

        return cls(config=config, clob=clob)

    # ── Read-only helpers ───────────────────────────────────────────────

    def get_balance_usdc(self) -> float | None:
        """Return USDC balance available for trading on the proxy. None in dry-run."""
        if self._clob is None:
            return None
        try:  # pragma: no cover — depends on live network
            # v2 takes a typed BalanceAllowanceParams; v1 took a plain dict.
            try:
                from py_clob_client_v2.clob_types import (
                    BalanceAllowanceParams, AssetType,
                )
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            except ImportError:
                params = {"asset_type": "COLLATERAL"}  # type: ignore
            bal = self._clob.get_balance_allowance(params)
            # Polymarket returns USDC scaled by 1e6 in the "balance" field.
            if isinstance(bal, dict):
                raw = bal.get("balance", 0.0)
                try:
                    raw_f = float(raw)
                except (TypeError, ValueError):
                    raw_f = 0.0
                # Heuristic: raw>1e4 means wei-scaled (divide by 1e6); else already dollars.
                return raw_f / 1e6 if raw_f > 1e4 else raw_f
            return float(bal)
        except Exception as exc:
            print(f"!! get_balance_usdc failed: {exc}")
            return None

    def get_open_orders(self) -> list[dict]:
        if self._clob is None:
            return []
        try:  # pragma: no cover
            # v2: get_open_orders(params=None). v1: get_orders().
            getter = getattr(
                self._clob,
                "get_open_orders",
                getattr(self._clob, "get_orders", None),
            )
            if getter is None:
                print("!! ClobClient has no get_open_orders/get_orders method")
                return []
            return list(getter() or [])
        except Exception as exc:
            print(f"!! get_open_orders failed: {exc}")
            return []

    def _lookup_actual_fill(
        self, order_id: str, max_retries: int = 3,
    ) -> tuple[float, float] | None:
        """Find the actual avg fill price + shares for an order by querying
        get_trades. Returns (avg_fill_price, total_shares_filled) or None.

        Used by submit_order's SELL path when the order-submit response
        omits takingAmount/makingAmount. Without this, the bot's recorded
        realized_pnl on FAK SELLs is pessimistic by the slippage-buffer
        amount (5pp typical) — Polymarket's matching engine often fills
        at a higher price than our limit.

        Retry rationale: Polymarket's trade index has eventual-consistency
        lag of 0.5-2s after order match. A first-call get_trades() may
        not yet show our just-filled trade, returning empty matching set
        → caller falls back to pessimistic limit price. With retry, we
        give the index time to catch up and get the actual fill price
        → realized PnL is accurate.

        Each retry waits 0.5s × (attempt+1): 0.5s, 1.0s, 1.5s = up to
        ~3s total. Worth it: every cross-up exit's realized PnL gains
        2-5pp accuracy, which compounds to $20-50/week saved from
        under-reporting losses.
        """
        if self._clob is None:
            return None
        import time
        matching: list = []
        for attempt in range(max_retries):
            try:
                trades = self._clob.get_trades()
            except Exception:
                # Transient API issue — back off and retry, or give up.
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
            if not isinstance(trades, list):
                return None
            matching = []
            for t in trades:
                if not isinstance(t, dict):
                    continue
                # Accept either taker_order_id (we crossed the book) or any
                # order_id field that matches our submitted order. The exact
                # key name varies across SDK versions; try several.
                tid = (
                    t.get("taker_order_id")
                    or t.get("takerOrderID")
                    or t.get("order_id")
                )
                if tid == order_id:
                    matching.append(t)
            if matching:
                break
            # No matches yet — likely indexing lag. Back off + retry.
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))

        if not matching:
            return None
        total_size = 0.0
        weighted_price_sum = 0.0
        for t in matching:
            try:
                sz = float(t.get("size", 0))
                pr = float(t.get("price", 0))
            except (TypeError, ValueError):
                continue
            if sz > 0 and pr > 0:
                total_size += sz
                weighted_price_sum += sz * pr
        if total_size <= 0:
            return None
        return weighted_price_sum / total_size, total_size

    def get_order(self, order_id: str) -> dict | None:
        """Fetch a single order by id. Returns the SDK dict or None on error.

        Used by poll_fills to disambiguate orders that have disappeared
        from the open-orders list. Possible status values:
          "LIVE"      — still resting (shouldn't normally hit get_order;
                        we already see these in open_orders)
          "MATCHED"   — fully filled
          "CANCELED"  — cancelled (by us or externally)
          "EXPIRED"   — expired without filling
        """
        if self._clob is None:
            return None
        try:  # pragma: no cover
            return self._clob.get_order(order_id)
        except Exception as exc:
            print(f"!! get_order({order_id[:14]}…) failed: {exc}")
            return None

    def get_maker_rebate_for_date(self, target_date) -> float:
        """Return the USD maker-rebate total earned on `target_date` (UTC).

        Polymarket pays maker rewards daily at midnight UTC with a $1
        minimum (per `polymarket_live_trading_lessons.md`). Below threshold
        the API typically returns $0 even when you earned non-zero —
        accrued but not paid.

        `target_date` is a `datetime.date` or ISO string (YYYY-MM-DD).
        Returns 0.0 on any error / dry-run, so calling unconditionally
        is safe.

        TODO (when SDK v2 exposes the endpoint): the underlying call is
        documented as `get_rewards_user(date=X)`. Verify the exact method
        name on py_clob_client_v2 — it may be `get_rewards`, `get_user_rewards`,
        or similar. Until then this stub returns 0.0.
        """
        if self._clob is None:
            return 0.0
        try:  # pragma: no cover
            from datetime import date as _date
            iso = (
                target_date.isoformat() if hasattr(target_date, "isoformat")
                else str(target_date)
            )
            # Best-effort SDK call. Try common names; ignore AttributeError.
            for method_name in ("get_rewards_user", "get_user_rewards", "get_rewards"):
                method = getattr(self._clob, method_name, None)
                if method is None:
                    continue
                result = method(date=iso)
                # Common response shapes: {"total": "1.23"}, {"amount": 1.23},
                # or a list of per-market rebates that sum to total.
                if isinstance(result, dict):
                    for key in ("total", "amount", "earnings"):
                        if key in result:
                            return float(result[key] or 0.0)
                    return 0.0
                if isinstance(result, (int, float, str)):
                    return float(result)
                if isinstance(result, list):
                    return sum(
                        float(item.get("amount", item.get("total", 0.0)) or 0.0)
                        for item in result if isinstance(item, dict)
                    )
                return 0.0
            return 0.0
        except Exception as exc:
            print(f"!! get_maker_rebate_for_date failed: {exc}")
            return 0.0

    def cancel_order(self, order_id: str) -> bool:
        if self._clob is None:
            print(f"  [DRY RUN] would cancel order {order_id}")
            return True
        try:  # pragma: no cover
            # v2 renamed cancel → cancel_order and takes an OrderPayload.
            # Try v2 first; fall back to v1's cancel(order_id).
            cancel = getattr(self._clob, "cancel_order", None)
            if cancel is not None:
                try:
                    from py_clob_client_v2.clob_types import OrderPayload  # type: ignore
                    cancel(OrderPayload(orderID=order_id))
                except ImportError:
                    cancel(order_id)
            else:
                self._clob.cancel(order_id)
            return True
        except Exception as exc:
            print(f"!! cancel_order failed: {exc}")
            return False

    def cancel_order_verified(self, order_id: str) -> tuple[bool, str | None]:
        """Cancel an order and VERIFY the resulting state.

        After calling cancel_order, queries the order's final state via
        get_order. Returns (was_actually_cancelled, actual_status):
          - (True,  "CANCELED")  → cancel succeeded; mark_cancelled is safe
          - (False, "MATCHED")   → race: order filled before cancel hit;
                                   caller MUST treat as filled (don't mark
                                   cancelled — let poll_fills / cross-up
                                   handle it)
          - (False, "EXPIRED")   → order self-expired before/during cancel
          - (False, "LIVE")      → cancel API said ok but order still alive
                                   (shouldn't happen; treat as cancel_failed)
          - (False, None)        → cancel API rejected, OR couldn't query
                                   state. Caller should treat as "unknown"
                                   and skip portfolio mutation.

        Adds ~50ms per cancel (one extra HTTP round trip). Worth it to
        prevent silent position-state corruption from filled-not-cancelled
        races.
        """
        if self._clob is None:
            print(f"  [DRY RUN] would cancel order {order_id}")
            return True, "CANCELED"

        # Try cancel first
        cancel_ok = self.cancel_order(order_id)
        if not cancel_ok:
            # SDK already logged the error. Try to figure out actual state.
            actual = self.get_order(order_id)
            if isinstance(actual, dict):
                status = actual.get("status")
                return False, status
            return False, None

        # Cancel call returned ok. Verify the order's actual final state
        # to guard against the "cancel returned ok=True but order was
        # already filled" race. Polymarket's SDK has been observed to
        # treat no-op cancels (filled orders) as success.
        actual = self.get_order(order_id)
        if not isinstance(actual, dict):
            # Couldn't verify — trust the cancel return value. This is
            # the historical behaviour; logged here so we can spot
            # patterns if get_order is flaky for specific tokens.
            return True, None
        status = actual.get("status")
        if status == "CANCELED" or status == "EXPIRED":
            return True, status
        if status == "MATCHED":
            # Race: order filled between portfolio.load and our cancel.
            # The cancel call was a no-op on Polymarket's side. Caller
            # MUST NOT mark this position as cancelled — it has shares.
            return False, "MATCHED"
        if status == "LIVE":
            # Weird state: cancel said ok but order still alive. Could
            # be eventual consistency. Caller should retry on next cycle.
            return False, "LIVE"
        return True, status  # other statuses: trust the cancel call

    def verify_cancel_status(self, order_id: str) -> tuple[bool, str | None]:
        """Verify the post-cancel state of an order without re-firing.

        Used by Layer 8's pre-signed cancel fast path: the cancel was
        already fired via PreSignedOrderCache.broadcast_cancel(); we just
        need to confirm the actual final state via get_order, with the
        same race-aware return-tuple semantics as cancel_order_verified.

        Returns (was_actually_cancelled, actual_status). See
        cancel_order_verified docstring for the exact semantics.
        """
        if self._clob is None:
            return True, "CANCELED"
        actual = self.get_order(order_id)
        if not isinstance(actual, dict):
            return True, None
        status = actual.get("status")
        if status == "CANCELED" or status == "EXPIRED":
            return True, status
        if status == "MATCHED":
            return False, "MATCHED"
        if status == "LIVE":
            return False, "LIVE"
        return True, status

    # ── Submit ──────────────────────────────────────────────────────────

    def submit_order(
        self,
        signal: TradeSignal,
        *,
        order_type: str = "FAK",
        sdk_side: str = "BUY",
        limit_price: float | None = None,
        post_only: bool = False,
        override_shares: float | None = None,
        expires_at: int | None = None,
    ) -> OrderResult:
        """Submit a single order derived from a TradeSignal.

        Always prints what it intends to do. In dry-run mode it returns
        a synthetic OrderResult; in live mode it submits via py-clob-client.

        Args:
            signal: TradeSignal with token_id, fill_price (= top-of-book ask,
                used for share-sizing), position_usd, side.
            order_type: Polymarket OrderType. FAK / GTC / FOK / GTD.
            sdk_side: "BUY" or "SELL".
            limit_price: OPTIONAL — the LIMIT price submitted to Polymarket.
                If None, uses signal.fill_price (legacy behaviour, no depth walk).
                If set HIGHER than fill_price (e.g. = $0.95 ceiling while
                fill_price = $0.40 top ask), Polymarket's matching engine
                walks the book from top_ask UP to limit_price, filling
                cheapest first. This captures the backtest's depth-aware-metar
                fills automatically. Verified via smoke order (2026-05-14):
                submitting limit $0.01 filled at avg $0.0019. Exchange walks
                the book; we just need to submit at the ceiling.

        Returns OrderResult with:
            fill_price = avg fill from Polymarket response (takingAmount /
              makingAmount), or limit_price if response doesn't expose
              fill detail (e.g. status=delayed). Distinct from the submitted
              limit price (recorded as result.limit_price).
            shares = filled shares from response, or target_shares if N/A
            target_shares = what we ASKED for (size_usd / signal.fill_price)
        """
        side = signal.side
        token_id = signal.token_id
        target_fill_price = signal.fill_price  # = top ask, used for sizing
        size_usd = signal.position_usd
        submit_limit = float(limit_price) if limit_price is not None else float(target_fill_price)
        # For SELL orders (cross-up exits): caller must pass the actual owned
        # shares via override_shares, because computing target_shares from
        # size_usd / fill_price produces a value WAY larger than we own when
        # the floor price is low (e.g. $5 / $0.01 = 500 shares vs the 6.4
        # we actually have). Polymarket rejects with "not enough balance".
        # For BUY orders, fall back to the legacy size_usd / fill_price calc.
        if override_shares is not None:
            target_shares = float(override_shares)
            # Recompute size_usd to match (= what we'd pay/receive at submit_limit)
            size_usd = target_shares * submit_limit
        else:
            target_shares = size_usd / target_fill_price if target_fill_price > 0 else 0.0

        # Descriptor: show both target (top ask) and limit (ceiling) when they differ
        if abs(submit_limit - target_fill_price) > 1e-6:
            price_str = f"limit ${submit_limit:.3f} (top_ask ${target_fill_price:.3f}, depth-walk)"
        else:
            price_str = f"@ ${submit_limit:.3f}"
        descriptor = (
            f"{signal.station.name} [{signal.target}] {signal.target_date} "
            f"{signal.bucket_label} — {sdk_side} {side} {price_str} "
            f"({order_type}), size ${size_usd:.2f} (~{target_shares:.2f} sh target), "
            f"edge {signal.edge:+.1%}"
        )

        if self._clob is None:
            print(f"  [DRY RUN] {descriptor}")
            return OrderResult(
                ok=True, order_id="dry-run", side=side, token_id=token_id,
                fill_price=target_fill_price, size_usd=size_usd,
                shares=target_shares, dry_run=True,
                message="dry-run (no order submitted)",
                limit_price=submit_limit, target_shares=target_shares,
            )

        # ── LIVE ORDER PATH ──────────────────────────────────────────────
        print(f"  [LIVE] {descriptor}")
        try:
            try:
                # v2 renamed OrderArgs → OrderArgsV2.
                from py_clob_client_v2.clob_types import (  # type: ignore
                    OrderArgsV2 as OrderArgs,
                    OrderType,
                )
                from py_clob_client_v2.order_builder.constants import BUY, SELL
            except ImportError:
                from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore
                from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore
        except ImportError as exc:
            return OrderResult(
                ok=False, order_id=None, side=side, token_id=token_id,
                fill_price=target_fill_price, size_usd=size_usd, shares=target_shares,
                dry_run=False, message=f"SDK import error: {exc}",
                limit_price=submit_limit, target_shares=target_shares,
            )

        # Map our sdk_side string to the SDK's BUY/SELL constant
        sdk_side_const = BUY if sdk_side.upper() == "BUY" else SELL

        # Map our order_type string to the SDK's OrderType enum
        try:
            ot_enum = getattr(OrderType, order_type.upper())
        except AttributeError:
            return OrderResult(
                ok=False, order_id=None, side=side, token_id=token_id,
                fill_price=target_fill_price, size_usd=size_usd, shares=target_shares,
                dry_run=False,
                message=f"Unsupported order_type {order_type!r}; "
                        f"valid: FAK, GTC, FOK, GTD",
                limit_price=submit_limit, target_shares=target_shares,
            )

        try:
            # GTD orders need a Unix-timestamp expiration. Other order types
            # use the SDK default (0 = no expiration).
            args_kwargs = dict(
                token_id=token_id,
                price=submit_limit,  # ← the LIMIT (= ceiling for FAK depth walks)
                size=float(target_shares),
                side=sdk_side_const,
            )
            if expires_at is not None and order_type.upper() == "GTD":
                args_kwargs["expiration"] = int(expires_at)
            args = OrderArgs(**args_kwargs)
            signed = self._clob.create_order(args)
            # post_only=True: SDK rejects if order would taker-fill.
            # Only valid for GTC + GTD (resting maker types); FAK/FOK reject.
            resp = self._clob.post_order(signed, ot_enum, post_only=post_only)
        except Exception as exc:  # pragma: no cover — network/SDK errors
            return OrderResult(
                ok=False, order_id=None, side=side, token_id=token_id,
                fill_price=target_fill_price, size_usd=size_usd, shares=target_shares,
                dry_run=False, message=f"submit error: {exc!s}",
                limit_price=submit_limit, target_shares=target_shares,
            )

        # ── Parse response. SDK / Polymarket response shapes vary:
        #   v1:  {"success": True, "orderID": "0x...", "transactionsHashes": [...]}
        #   v2:  {"success": True, "orderID": "0x...", "status": "matched",
        #         "takingAmount": "X", "makingAmount": "Y"}
        # For BUYs:  takingAmount = USDC paid;  makingAmount = shares acquired
        # → avg_fill_price = takingAmount / makingAmount
        # When status="delayed" or partial, these may be empty — fall back to limit.
        avg_fill_price = submit_limit  # conservative fallback
        filled_shares = target_shares  # conservative fallback
        if isinstance(resp, dict):
            ok = bool(resp.get("success", False))
            order_id = resp.get("orderID") or resp.get("order_id")
            msg_parts = []
            if "status" in resp:
                msg_parts.append(f"status={resp['status']}")

            # Extract actual avg fill price from response if available
            taking = resp.get("takingAmount")
            making = resp.get("makingAmount")
            fill_extracted_from_response = False
            try:
                if taking and making:
                    taking_f = float(taking)
                    making_f = float(making)
                    # 2026-05-22 fix: the BUY/SELL interpretation of
                    # takingAmount/makingAmount was inverted in the
                    # previous code. The actual CLOB convention:
                    #   takingAmount = the amount the order TAKER
                    #                  receives (consumed liquidity)
                    #   makingAmount = the amount the order TAKER
                    #                  committed (the maker side)
                    # For LIMIT BUY that filled:
                    #   takingAmount = SHARES received (we took the
                    #                  resting sell offer)
                    #   makingAmount = USDC paid (our committed side)
                    # For LIMIT SELL that filled:
                    #   takingAmount = USDC received (we took the
                    #                  resting buy offer)
                    #   makingAmount = SHARES delivered (our committed side)
                    #
                    # Empirical confirmation: ZGSZ 28C BUY on 2026-05-22
                    # had response with taking=5.0, making=4.95. UI shows
                    # we received 5.1 shares for $4.95. Old code computed
                    # avg_fill_price=$1.01 + shares=4.95 (inverted).
                    # New code computes avg_fill_price=$0.99 + shares=5.0
                    # (correct; UI displays 5.1sh@98c due to depth-walk
                    # rounding but product matches: 5.0*0.99 = $4.95).
                    #
                    # The old SELL interpretation was ALSO inverted; cross_up_cancel.py
                    # had a product-based workaround for the SELL case (using
                    # position_usd as entry_cost). After this fix, the SELL
                    # values from response are correct and that workaround
                    # becomes a no-op (still safe -- it uses the product).
                    if sdk_side.upper() == "BUY":
                        if taking_f > 0:
                            avg_fill_price = making_f / taking_f
                            filled_shares = taking_f
                            fill_extracted_from_response = True
                    else:
                        # SELL
                        if making_f > 0:
                            avg_fill_price = taking_f / making_f
                            filled_shares = making_f
                            fill_extracted_from_response = True
                    if fill_extracted_from_response:
                        msg_parts.append(
                            f"avg_fill=${avg_fill_price:.4f}"
                            f"  filled={filled_shares:.2f}sh"
                            f"  vs limit=${submit_limit:.3f}"
                        )
            except (TypeError, ValueError):
                pass

            # Fallback for SELL orders: Polymarket's response often omits
            # takingAmount/makingAmount for FAK SELLs, leaving us with the
            # limit price as a pessimistic estimate. Query get_trades to
            # find the actual fill price. Audit 2026-05-16 found this was
            # under-reporting realized PnL on every cross-up SELL by
            # ~$0.30-$0.50 per win vs the Polymarket UI's truth.
            if ok and order_id and sdk_side.upper() == "SELL" and not fill_extracted_from_response:
                try:
                    actual = self._lookup_actual_fill(order_id)
                except Exception as exc:
                    actual = None
                    msg_parts.append(f"actual-fill-lookup-failed={exc!s}"[:80])
                if actual is not None:
                    avg_fill_price, filled_shares = actual
                    msg_parts.append(
                        f"avg_fill=${avg_fill_price:.4f}"
                        f"  filled={filled_shares:.2f}sh"
                        f"  vs limit=${submit_limit:.3f} (from get_trades)"
                    )

            if "transactionsHashes" in resp:
                hashes = resp["transactionsHashes"]
                if hashes:
                    msg_parts.append(f"tx={hashes[0][:12]}…")
            if "errorMsg" in resp:
                msg_parts.append(f"err={resp['errorMsg']}")
            message = ", ".join(msg_parts) if msg_parts else str(resp)[:200]
        else:
            ok = False
            order_id = None
            message = f"unexpected response type: {type(resp).__name__}"

        # APPEND-ONLY SUBMISSION LOG (2026-05-22 incident). Before
        # returning, persist a record of this submission to
        # data/submitted_orders.jsonl. This is independent of
        # portfolio.json and catches the failure mode where
        # portfolio.save() returns success but the position doesn't
        # persist (LTFM: 5 saves succeeded, only 1 on disk -> 4 orphans).
        # Best-effort: never block the return on logging failure.
        if ok and order_id:
            try:
                from weather_bot.alerts import log_submitted_order
                _sig_station_id = getattr(getattr(signal, "station", None), "station_id", "?")
                _sig_bucket = getattr(signal, "bucket_label", "?")
                _sig_td = getattr(signal, "target_date", None)
                _sig_td_iso = _sig_td.isoformat() if _sig_td is not None else "?"
                log_submitted_order(
                    order_id=order_id,
                    token_id=token_id,
                    side=side,
                    station_id=_sig_station_id,
                    bucket_label=_sig_bucket,
                    target_date=_sig_td_iso,
                    fill_price=avg_fill_price,
                    shares=filled_shares,
                    size_usd=size_usd,
                    sdk_side=sdk_side,
                    order_type=order_type,
                )
            except Exception as _log_exc:
                print(f"!! log_submitted_order failed for {order_id}: {_log_exc!s}")

        return OrderResult(
            ok=ok, order_id=order_id, side=side, token_id=token_id,
            fill_price=avg_fill_price, size_usd=size_usd, shares=filled_shares,
            dry_run=False, message=message,
            limit_price=submit_limit, target_shares=target_shares,
        )
