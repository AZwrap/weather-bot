"""Critical-alert system for the bot — surfaces silent failures loudly.

THE PROBLEM THIS SOLVES
=======================

2026-05-21 incident: intraday_scan crashed with ImportError AFTER
placing live orders on Polymarket. The orders filled (549 shares of
NYC NO at $0.015), but the bot's portfolio.json was never updated —
the bot didn't know it owned those shares. No automated management
possible. The crash was logged to a generic file that no one was
watching.

The general failure mode: bot performs an irreversible action (submit
order, send transaction) -> something fails between the success and
the local-state update -> local state is stale -> bot continues as
if nothing happened.

This module provides a single place to write critical alerts that:
  1. Persist to data/alerts.jsonl (append-only audit trail)
  2. Are loud in the daemon log (print + traceback)
  3. Block daemon startup if there are unhandled alerts from a previous
     session (forces operator to acknowledge before resuming)

USAGE
=====

```python
from weather_bot.alerts import record_alert, has_unhandled_alerts

# After a critical operation succeeds:
try:
    result = client.submit_order(...)
    if result.ok:
        portfolio.add(position)
        portfolio.save(path)
except Exception as exc:
    record_alert(
        kind="orphan_order_save_failed",
        severity="critical",
        summary=f"Order {result.order_id} placed on Polymarket but local save failed",
        details={
            "order_id": result.order_id,
            "token_id": position.token_id,
            "exception": str(exc),
            "exception_type": type(exc).__name__,
        },
    )
    raise  # don't swallow; let it propagate

# At daemon startup:
if has_unhandled_alerts():
    print("!! UNHANDLED ALERTS PRESENT — refusing to start.")
    print("!! Review data/alerts.jsonl and touch data/alerts_ack to resume.")
    sys.exit(1)
```

ALERT LIFECYCLE
===============

1. record_alert(...) writes a JSONL record + prints to stdout
2. Operator reviews data/alerts.jsonl
3. Operator acknowledges by touching data/alerts_ack (or by running
   ack_alerts() programmatically)
4. has_unhandled_alerts() returns False after acknowledgement
5. Bot can resume
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ALERTS_PATH = Path("data/alerts.jsonl")
DEFAULT_ACK_PATH = Path("data/alerts_ack")
DEFAULT_SUBMITTED_ORDERS_PATH = Path("data/submitted_orders.jsonl")
"""APPEND-ONLY log of every successful order submission to Polymarket.

Independent of portfolio.json -- written by `log_submitted_order()` at
the point the SDK returns a valid order_id, BEFORE the strategy module
tries to `portfolio.add(...).save(...)`. This catches the 2026-05-22 LTFM
failure mode: 5 saves succeeded (no exception), audit log got 5 entries,
but only 1 position made it to portfolio.json. The append-only log can't
be silently overwritten by a stale concurrent write.

`find_orphan_orders()` reconciles this log against portfolio.json's
open positions and reports any submission whose order_id isn't tracked."""


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_alert(
    *,
    kind: str,
    severity: str = "critical",   # "critical" | "warning" | "info"
    summary: str,
    details: dict[str, Any] | None = None,
    alerts_path: Path = DEFAULT_ALERTS_PATH,
    print_to_stdout: bool = True,
) -> None:
    """Record a critical alert to data/alerts.jsonl + stdout.

    Designed to never raise — if writing fails, prints the alert
    inline so the user still has a chance to see it.
    """
    record = {
        "ts_utc": _utc_iso(),
        "kind": kind,
        "severity": severity,
        "summary": summary,
        "details": details or {},
        "traceback": traceback.format_exc() if sys.exc_info()[0] else None,
    }
    if print_to_stdout:
        print()
        print("!!" + "=" * 78)
        print(f"!! ALERT ({severity.upper()}): {kind}")
        print(f"!! {summary}")
        if details:
            for k, v in details.items():
                print(f"!!   {k}: {v}")
        print("!!" + "=" * 78)
        print()
    try:
        alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Last-resort: print is the best we can do
        print(f"!! ALSO: failed to persist alert to {alerts_path}")


def has_unhandled_alerts(
    alerts_path: Path = DEFAULT_ALERTS_PATH,
    ack_path: Path = DEFAULT_ACK_PATH,
) -> bool:
    """True iff alerts.jsonl exists AND was modified after alerts_ack
    (or alerts_ack doesn't exist).

    Use at daemon startup to refuse running with stale alerts. The
    operator acknowledges by running `touch data/alerts_ack` after
    reviewing the alerts.
    """
    if not alerts_path.exists():
        return False
    if not alerts_path.stat().st_size:
        return False
    if not ack_path.exists():
        return True
    alerts_mtime = alerts_path.stat().st_mtime
    ack_mtime = ack_path.stat().st_mtime
    return alerts_mtime > ack_mtime


def ack_alerts(ack_path: Path = DEFAULT_ACK_PATH) -> None:
    """Acknowledge all current alerts by touching the ack file.

    Equivalent of `touch data/alerts_ack` but importable from Python.
    Doesn't delete or modify alerts.jsonl — preserves the audit trail.
    """
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    ack_path.touch()
    # Bump mtime explicitly in case touch is a no-op on existing file
    now = datetime.now(timezone.utc).timestamp()
    os.utime(ack_path, (now, now))


def unhandled_alert_summary(
    alerts_path: Path = DEFAULT_ALERTS_PATH,
    ack_path: Path = DEFAULT_ACK_PATH,
    max_lines: int = 10,
) -> str:
    """Return a short summary of unhandled alerts for printing on startup."""
    if not has_unhandled_alerts(alerts_path, ack_path):
        return ""
    ack_mtime = ack_path.stat().st_mtime if ack_path.exists() else 0
    new_alerts = []
    try:
        with alerts_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = r.get("ts_utc", "")
                try:
                    ts_obj = datetime.fromisoformat(ts).timestamp()
                except Exception:
                    continue
                if ts_obj > ack_mtime:
                    new_alerts.append(r)
    except OSError:
        return f"(could not read {alerts_path})"
    if not new_alerts:
        return ""
    lines = [f"{len(new_alerts)} unhandled alert(s):"]
    for r in new_alerts[-max_lines:]:
        lines.append(f"  [{r.get('severity','?')}] {r.get('ts_utc','?')[:19]} "
                     f"{r.get('kind','?')}: {r.get('summary','?')[:80]}")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────
# Append-only submission log + orphan reconciliation (2026-05-22 incident)
# ───────────────────────────────────────────────────────────────────────


def log_submitted_order(
    *,
    order_id: str,
    token_id: str,
    side: str,
    station_id: str = "?",
    bucket_label: str = "?",
    target_date: str = "?",
    fill_price: float | None = None,
    shares: float | None = None,
    size_usd: float | None = None,
    sdk_side: str = "?",
    order_type: str = "?",
    submitted_orders_path: Path = DEFAULT_SUBMITTED_ORDERS_PATH,
) -> None:
    """Append one record of a successful order submission. Append-only;
    never modified or overwritten. The ground truth for "did the bot
    place this order on Polymarket".

    Called by ExecutionClient.submit_order at the point the SDK returns
    a valid order_id. Independent of portfolio.json -- catches the
    failure mode where portfolio.save() returns successfully but the
    position doesn't actually persist (2026-05-22 LTFM: 5 saves, only
    1 position on disk, 4 orphans).

    Best-effort write: if disk fails, prints a warning but doesn't
    raise. The order is already placed; refusing to log would be worse.
    """
    record = {
        "ts_utc": _utc_iso(),
        "order_id": order_id,
        "token_id": token_id,
        "side": side,
        "sdk_side": sdk_side,
        "order_type": order_type,
        "station_id": station_id,
        "bucket_label": bucket_label,
        "target_date": target_date,
        "fill_price": fill_price,
        "shares": shares,
        "size_usd": size_usd,
    }
    try:
        submitted_orders_path.parent.mkdir(parents=True, exist_ok=True)
        with submitted_orders_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        print(f"!! ALSO: failed to log submission of order {order_id} "
              f"to {submitted_orders_path}")


def find_orphan_orders(
    portfolio_path: Path | None = None,
    submitted_orders_path: Path = DEFAULT_SUBMITTED_ORDERS_PATH,
    *,
    since_iso: str | None = None,
    grace_seconds: float = 60.0,
) -> list[dict]:
    """Compare append-only submitted_orders log to portfolio.json's
    open positions. Return records for any submitted order_id NOT
    tracked in portfolio -- those are orphans (placed on Polymarket
    but the bot lost track).

    Args:
      portfolio_path: path to portfolio.json. Default: DEFAULT_PORTFOLIO_PATH
        from weather_bot.portfolio.
      submitted_orders_path: append-only log path.
      since_iso: only check submissions with ts_utc >= this ISO string.
        Default: 2 days ago (skip already-resolved positions, which
        will have been pruned out of portfolio).
      grace_seconds: skip submissions within the last N seconds so we
        don't false-positive on an in-flight portfolio.save still on
        its way to disk. Default 60s.

    Returns: list of submitted-order records (dicts) that are orphans.
    Empty list = healthy.
    """
    if portfolio_path is None:
        from .portfolio import DEFAULT_PORTFOLIO_PATH as _DPP
        portfolio_path = _DPP
    if not submitted_orders_path.exists():
        return []

    now = datetime.now(timezone.utc)
    if since_iso is None:
        from datetime import timedelta
        since_iso = (now - timedelta(days=2)).isoformat()
    cutoff_iso = (now - timedelta(seconds=grace_seconds)).isoformat() if grace_seconds > 0 else None

    # Build map of order_id -> latest record (since cutoff)
    submitted: dict[str, dict] = {}
    try:
        with submitted_orders_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = r.get("ts_utc", "")
                if ts < since_iso:
                    continue
                if cutoff_iso is not None and ts >= cutoff_iso:
                    # Within grace window -- skip to avoid false positives
                    continue
                oid = r.get("order_id")
                if oid:
                    submitted[oid] = r
    except OSError:
        return []

    # Set of order_ids known to portfolio (any non-final status)
    portfolio_oids: set[str] = set()
    try:
        with portfolio_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("positions", []):
            oid = p.get("order_id")
            if oid:
                portfolio_oids.add(oid)
    except (OSError, json.JSONDecodeError):
        # If portfolio can't be read, treat all submissions as orphans
        # (safer than treating none as orphans on a broken portfolio).
        return list(submitted.values())

    return [r for oid, r in submitted.items() if oid not in portfolio_oids]


# Add timedelta import at module level (used by find_orphan_orders)
from datetime import timedelta  # noqa: E402


# ───────────────────────────────────────────────────────────────────────
# Persistent-submitted detection (2026-05-25, Task #65)
# ───────────────────────────────────────────────────────────────────────


def find_persistent_submitted(
    portfolio_path: Path | None = None,
    *,
    threshold_minutes: float = 30.0,
) -> list[dict]:
    """Return positions in 'submitted' state for longer than threshold_minutes.

    Why: poll_fills has TWO branches that can leave a position stuck in
    'submitted' indefinitely:

      1. Primary "info is None, no expires_at_utc" branch (line ~864 in
         portfolio.py): get_order returns nothing AND no TTL info → can't
         presume dead.

      2. Secondary "unrecognized status, no expires_at_utc" branch
         (line ~954): get_order returns a status outside the known set
         {MATCHED, CANCELED, EXPIRED, INVALID, REJECTED} AND no TTL info.

    Both branches were patched 2026-05-22 (CYYZ) and 2026-05-25 (Task #39)
    for the WITH-expires_at_utc case. Positions WITHOUT expires_at_utc
    can still get stuck — currently theoretical (every order we submit
    today either sets the TTL or doesn't rest), but if a future code path
    adds a resting order without TTL, this catches it.

    Threshold rationale: GTD orders have a 90s TTL + 5min grace + slack →
    self-resolve within ~6 min in normal operation. 30 min default is a
    5× safety margin over the worst legitimate case.

    This is OBSERVABILITY, not a fix: callers (e.g., intraday_scan startup
    gate, daemon periodic health check) should record_alert + optionally
    demote to paper mode. Don't auto-cancel — without TTL info we can't
    safely presume an order is dead, and the right fix for a new
    stuck-pattern bug is to investigate it, not paper over it.

    Args:
      portfolio_path: path to portfolio.json. Default: DEFAULT_PORTFOLIO_PATH.
      threshold_minutes: positions older than this in 'submitted' are stuck.
        Default 30. Tune up if false-positives appear, but the root cause
        should be diagnosed first.

    Returns: list of stuck-position records (dicts). Empty list = healthy.
    """
    if portfolio_path is None:
        from .portfolio import DEFAULT_PORTFOLIO_PATH as _DPP
        portfolio_path = _DPP
    try:
        with portfolio_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    now = datetime.now(timezone.utc)
    cutoff_iso = (now - timedelta(minutes=threshold_minutes)).isoformat()

    stuck: list[dict] = []
    for p in data.get("positions", []):
        if p.get("status") != "submitted":
            continue
        oid = p.get("order_id")
        if not oid or oid == "dry-run":
            continue
        sub_at = p.get("submitted_at", "")
        if not sub_at or sub_at >= cutoff_iso:
            continue
        try:
            sub_dt = datetime.fromisoformat(sub_at.replace("Z", "+00:00"))
            age_min = (now - sub_dt).total_seconds() / 60.0
        except Exception:
            age_min = None
        stuck.append({
            "order_id": oid,
            "token_id": p.get("token_id"),
            "side": p.get("side"),
            "station_id": p.get("station_id"),
            "bucket_label": p.get("bucket_label"),
            "target_date": p.get("target_date"),
            "submitted_at": sub_at,
            "expires_at_utc": p.get("expires_at_utc"),
            "stuck_minutes": round(age_min, 1) if age_min is not None else None,
            "strategy": p.get("strategy"),
        })
    return stuck
