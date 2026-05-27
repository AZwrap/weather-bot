"""Drawdown circuit breaker — halts NEW placements when realized PnL
breaches a configured threshold. Cross-up cancel + Layer 7 + redemption
continue to fire (we still want to exit losing positions and harvest
guaranteed-NO buckets); only NEW NO_momentum placements get blocked.

State stored in `data/drawdown_state.json`:
  {
    "realized_pnl_at_last_check_usd": -42.50,
    "breaker_tripped": false,
    "breaker_tripped_at_utc": null,
    "manual_override": false,
    "last_check_utc": "2026-05-17T16:00:00+00:00"
  }

The breaker is one-way: once tripped, it stays tripped until a manual
override (touch `data/drawdown_override` file) OR realized PnL recovers
above the threshold (auto-reset).

Default threshold: −$100 (20% of $500 bankroll). Tunable via
`DRAWDOWN_BREAKER_USD` env var or override the function arg.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:  # Windows local dev
    _FCNTL_AVAILABLE = False


DEFAULT_DRAWDOWN_BREAKER_USD = -50.0
"""Realized PnL threshold below which new placements halt. Negative
value: −$50 means halt when realized < −$50. Tunable per session
via env var DRAWDOWN_BREAKER_USD.

TIGHTENED 2026-05-19 from −$100 → −$50 (10% of $500 bankroll). Sized
for the re-validation phase after the Polymarket archive event: at the
new $50/day deployment cap, expected daily burn under the pre-archive
~30% loss rate would be ~$15/day, so −$50 = ~3-4 bad days before the
breaker forces a human decision. Original −$100 (20% bankroll) returns
once the strategy is empirically net-positive over 7+ days.

NOTE on resume: lifetime position-only realized PnL may already be
below −$50 when trading resumes (the cross-up SELL fix yielded
−$16.58 lifetime per reconcile, but archived markets count as $0
resolution losses). If the breaker trips on the first cycle after
resume, that's working as designed — touch `data/drawdown_override`
once to clear, or do the reconcile-portfolio TODO first to reset
lifetime PnL to the post-refund truth (memo: project_strategy_levers
_post_day2.md lines 130-132)."""

DEFAULT_STATE_PATH = Path("data/drawdown_state.json")
OVERRIDE_FILE = Path("data/drawdown_override")
"""Touch this file to manually re-enable placements after the breaker
trips. Bot deletes the override after a successful re-enable, so it's
single-use (you must touch it again if breaker re-trips)."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked_state(path: Path):
    """Open the breaker state file under an fcntl exclusive lock for
    atomic read-modify-write.

    Without this, the daemon and cron racing to evaluate the breaker
    would each load → decide → write back, and the second writer
    overwrites the first's decision (audit finding C3, 2026-05-17).
    Worst case: the breaker silently fails to enforce during the very
    anomaly conditions it exists for.

    Falls back to a best-effort non-locked read+write on Windows
    (local dev only — live bot is Linux).

    Yields (state_dict, write_fn). Caller mutates `state_dict` in place
    and calls `write_fn()` to persist before the context exits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not _FCNTL_AVAILABLE:
        # Best-effort non-atomic path
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if not isinstance(state, dict):
                state = {}
        except (json.JSONDecodeError, OSError):
            state = {}

        def _write_unlocked() -> None:
            from weather_bot.atomic_write import atomic_write_json
            atomic_write_json(path, state)

        yield state, _write_unlocked
        return

    mode = "r+" if path.exists() else "w+"
    with open(path, mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            try:
                state = json.loads(content) if content else {}
                if not isinstance(state, dict):
                    state = {}
            except json.JSONDecodeError:
                state = {}

            def _write_locked() -> None:
                f.seek(0)
                f.truncate()
                json.dump(state, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            yield state, _write_locked
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_state(path: Path) -> dict:
    """Read-only state snapshot (for observability — NOT for decision making).
    Decision-making callers must go through `_locked_state` for atomicity."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_threshold(threshold_usd: float | None) -> float:
    """Resolve the breaker threshold from arg, env var, or default.

    Audit F4-M1: a positive value would silently invert the breaker
    (trip at +$100 — IMMEDIATELY on day one). Reject + warn instead.
    """
    if threshold_usd is None:
        env = os.environ.get("DRAWDOWN_BREAKER_USD")
        if env is not None:
            try:
                threshold_usd = float(env)
            except ValueError:
                print(f"!! [drawdown] DRAWDOWN_BREAKER_USD={env!r} unparseable; "
                      f"using default ${DEFAULT_DRAWDOWN_BREAKER_USD:+.2f}")
                threshold_usd = DEFAULT_DRAWDOWN_BREAKER_USD
        else:
            threshold_usd = DEFAULT_DRAWDOWN_BREAKER_USD

    # Reject positive thresholds — drawdown is by definition a negative
    # PnL value. A positive threshold would trip the breaker immediately
    # at PnL just below the (positive) threshold, halting all placements.
    if threshold_usd >= 0:
        print(f"!! [drawdown] threshold ${threshold_usd:+.2f} is non-negative; "
              f"refusing to invert the breaker. Using default "
              f"${DEFAULT_DRAWDOWN_BREAKER_USD:+.2f}.")
        threshold_usd = DEFAULT_DRAWDOWN_BREAKER_USD
    return float(threshold_usd)


def check_drawdown_breaker(
    realized_pnl_usd: float,
    *,
    threshold_usd: float | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    verbose: bool = True,
) -> tuple[bool, str]:
    """Check the drawdown circuit breaker.

    Args:
      realized_pnl_usd: position-resolution PnL (USD, negative if down).
                       Should EXCLUDE maker rebates — the operator's
                       mental model of "drawdown" is position loss, not
                       net cashflow. Use `can_place_new_orders` for the
                       canonical wiring (audit F4-H5).
      threshold_usd: cap (negative). Defaults to env var DRAWDOWN_BREAKER_USD
                     or −$100.
      state_path: where the breaker state file lives
      verbose: print state changes

    Returns:
      (can_place, reason)
        can_place=True   → safe to place new NO_momentum / Layer 7 orders
        can_place=False  → halted; `reason` describes why
    """
    threshold_usd = _resolve_threshold(threshold_usd)
    is_under_threshold = realized_pnl_usd < threshold_usd

    # All read+decide+write is serialized by an fcntl exclusive lock on
    # state_path — no daemon/cron race can lose a decision (audit C3).
    with _locked_state(state_path) as (state, write_state):
        was_tripped = bool(state.get("breaker_tripped", False))

        # State transitions:
        #   not tripped + under threshold     → TRIP
        #   tripped + override file present   → CLEAR (and delete override)
        #   tripped + above threshold         → CLEAR (auto-reset)
        #   else: maintain state

        if not was_tripped and is_under_threshold:
            # TRIP the breaker
            state.update({
                "breaker_tripped": True,
                "breaker_tripped_at_utc": _now_utc_iso(),
                "tripped_at_realized_pnl_usd": realized_pnl_usd,
                "threshold_usd": threshold_usd,
                "last_check_utc": _now_utc_iso(),
                "realized_pnl_at_last_check_usd": realized_pnl_usd,
            })
            write_state()
            msg = (f"DRAWDOWN BREAKER TRIPPED: realized PnL ${realized_pnl_usd:+.2f} "
                   f"below threshold ${threshold_usd:+.2f}. NEW placements halted. "
                   f"Cross-up / Layer 7 / redemption continue. "
                   f"Touch {OVERRIDE_FILE} to manually re-enable, or wait for "
                   f"PnL to recover above threshold.")
            if verbose:
                print()
                print("!! " + "=" * 70)
                print(f"!! {msg}")
                print("!! " + "=" * 70)
                print()
            return False, msg

        if was_tripped:
            # Check for manual override — unlink under the same lock to
            # avoid two processes both observing + consuming the override.
            if OVERRIDE_FILE.exists():
                try:
                    OVERRIDE_FILE.unlink()
                except OSError:
                    pass
                state.update({
                    "breaker_tripped": False,
                    "breaker_cleared_at_utc": _now_utc_iso(),
                    "cleared_via": "manual_override",
                    "last_check_utc": _now_utc_iso(),
                    "realized_pnl_at_last_check_usd": realized_pnl_usd,
                })
                write_state()
                if verbose:
                    print(f"  [drawdown] breaker CLEARED via manual override "
                          f"(realized=${realized_pnl_usd:+.2f})")
                return True, "cleared via manual override"

            # Auto-clear if PnL recovered above threshold
            if not is_under_threshold:
                state.update({
                    "breaker_tripped": False,
                    "breaker_cleared_at_utc": _now_utc_iso(),
                    "cleared_via": "auto_recovery",
                    "last_check_utc": _now_utc_iso(),
                    "realized_pnl_at_last_check_usd": realized_pnl_usd,
                })
                write_state()
                if verbose:
                    print(f"  [drawdown] breaker AUTO-CLEARED: realized "
                          f"${realized_pnl_usd:+.2f} recovered above "
                          f"${threshold_usd:+.2f}")
                return True, "auto-cleared (PnL recovered)"

            # Still tripped, still under threshold
            return False, (
                f"breaker tripped at ${state.get('tripped_at_realized_pnl_usd', 0):+.2f}, "
                f"realized still ${realized_pnl_usd:+.2f} below ${threshold_usd:+.2f}"
            )

        # Not tripped, not under threshold — normal operation
        # Update the last-check timestamp + value
        state.update({
            "breaker_tripped": False,
            "last_check_utc": _now_utc_iso(),
            "realized_pnl_at_last_check_usd": realized_pnl_usd,
            "threshold_usd": threshold_usd,
        })
        write_state()
        return True, "normal"


def can_place_new_orders(
    portfolio: Any,  # weather_bot.portfolio.Portfolio
    threshold_usd: float | None = None,
) -> tuple[bool, str]:
    """Convenience wrapper: pull realized PnL from portfolio + check breaker.

    Uses POSITION-only realized PnL (excludes maker rebates) as the trip
    metric. The operator's mental model of "drawdown" is position loss;
    rebates compound the bankroll separately but shouldn't mask a real
    losing streak from the breaker (audit F4-H5).

    Returns (allowed, reason). Call from any placement gate
    (NO_momentum, Layer 7) before submitting.

    Fail-CLOSED on portfolio read failure: portfolio.json corruption /
    transient OSError is precisely the kind of anomaly the breaker
    exists to halt during. 15-min cron retry means worst case is one
    missed cycle on a transient error — correct tradeoff (audit F4-H2).
    """
    try:
        net_pnl = portfolio.realized_pnl_total()
        total_rebates = portfolio.total_maker_rebates()
        position_pnl = net_pnl - total_rebates
    except Exception as exc:
        msg = f"breaker FAIL-CLOSED: portfolio read failed ({type(exc).__name__}: {exc})"
        print(f"!! [drawdown] {msg}")
        return False, msg
    return check_drawdown_breaker(position_pnl, threshold_usd=threshold_usd)
