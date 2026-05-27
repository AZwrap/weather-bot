"""Atomic daily-deployment cap counter (cross-process safe).

The bot's `$150/day deployment cap` was previously enforced via
`Portfolio.today_deployed_usd()`, which sums position_usd over filled+
resolved positions in `data/portfolio.json`. This works WITHIN a single
process call (each placement updates positions in memory, next iteration
sees the new total), but breaks ACROSS processes:

  - Daemon (30s cycle) and cron (15-min cycle) both load portfolio.json
    independently at the start of their cycle.
  - During the overlap window (cron tick fires while daemon cycle is
    running), both see the same stale `today_deployed`.
  - Both can independently approve a $5 placement against the same
    "$5 below cap" reading → cap overshoots by $5 per concurrent fill.

The fix: a small JSON file at `data/cap_budget.json` whose updates are
gated by an fcntl exclusive lock. Each placement caller does:

    cap_ok, reason = acquire_cap_token(
        size_usd=5.0, daily_limit_usd=150.0, station_id="KMIA",
    )
    if cap_ok:
        try:
            result = client.submit_order(...)
            if not result.ok:
                # refund the reservation since we didn't actually deploy
                release_cap_token(size_usd=5.0, station_id="KMIA")
        except Exception:
            release_cap_token(size_usd=5.0, station_id="KMIA")
            raise

The reservation-then-confirm pattern ensures that capital is only
counted toward the cap when an actual fill (or attempted submit)
happens, with refunds on rejection. The fcntl lock is OS-level and
atomic across the daemon and cron processes.

State file schema (`data/cap_budget.json`):
  {
    "date_utc": "2026-05-17",
    "deployed_usd": 142.50,
    "per_station_deployed_usd": {"KMIA": 25.0, "RKSI": 15.0},
    "reservation_count": 28,         # number of distinct acquires today
    "release_count": 2,              # number of refunds (submit failures)
    "last_updated_utc": "2026-05-17T16:00:00+00:00"
  }

Date rollover semantics (audit F5-M4): when the stored `date_utc`
doesn't match the current UTC date, BOTH the global counter AND the
per-station map reset to zero. This is UTC-based — a station in UTC+10
will see its per-station counter reset mid-afternoon local time.
This is intentional and matches `Portfolio.today_deployed_usd()`'s
UTC semantics. If station-local rollover is ever required, it would
need a per-station date tracker (deferred).

Falls back gracefully on non-Linux (Windows local dev): does a
best-effort non-atomic check. Live bot runs on Linux so this is fine.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:  # Windows local dev
    _FCNTL_AVAILABLE = False


DEFAULT_CAP_BUDGET_PATH = Path("data/cap_budget.json")

DEFAULT_PER_STATION_CAP_USD = 30.0
"""Per-station daily deployment limit. Defaults to $30 (= 20% of the
$150 daily global cap). Prevents a single hot-day region (e.g., 5
LFPB adverse fills on a heatwave day) from consuming the entire
daily budget. Tune via env var PER_STATION_CAP_USD."""


DEFAULT_FAST_REACTION_RESERVE_USD = 30.0
"""USD of the daily cap reserved EXCLUSIVELY for fast-reaction layers
(Layer 7 guaranteed_no_buy + METAR_peak + METAR_early_tail). NO_momentum
callers see an effective cap of `daily_limit_usd - this`, so they can
never consume more than (daily_cap - reserve). The reserve is available
ONLY to callers passing `caller_kind="fast_reaction"`.

Rationale (Day 3 finding 2026-05-18): NO_momentum placements consumed
the entire $150 daily cap in the first minute of the day (30 placements
× $5), leaving Layer 7 with $0 to work with. Fast-reaction layers had
1,657 `skipped_daily_limit` events while sitting on an otherwise valid
opportunity stream. Reserving $30 (~6 fast-reaction fills at $5 each)
ensures Layer 7 + METAR layers always have room to fire."""


def _today_utc_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_or_init(f) -> dict:
    """Read JSON state from a file handle (positioned at start). If
    parsing fails or date has rolled over, return a fresh state for today.
    """
    try:
        f.seek(0)
        content = f.read()
        if content:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("not a dict")
        else:
            data = {}
    except (json.JSONDecodeError, ValueError, OSError):
        data = {}

    today = _today_utc_iso()
    if data.get("date_utc") != today:
        # Date rolled over (or first init) — reset counter
        data = {
            "date_utc": today,
            "deployed_usd": 0.0,
            "reservation_count": 0,
            "release_count": 0,
            "last_updated_utc": _now_utc_iso(),
        }
    return data


def _write_state(f, data: dict) -> None:
    """Write state to file handle (overwrites in place)."""
    data["last_updated_utc"] = _now_utc_iso()
    f.seek(0)
    f.truncate()
    json.dump(data, f, indent=2)


def acquire_cap_token(
    size_usd: float,
    daily_limit_usd: float,
    state_path: Path = DEFAULT_CAP_BUDGET_PATH,
    *,
    station_id: str,
    per_station_cap_usd: float | None = None,
    caller_kind: str = "no_momentum",
    fast_reaction_reserve_usd: float | None = None,
) -> tuple[bool, str]:
    """Attempt to reserve `size_usd` of budget against the daily GLOBAL
    cap and the per-station cap.

    Args:
      size_usd: USD to reserve
      daily_limit_usd: global daily cap (default $150)
      station_id: REQUIRED. The station to charge the per-station counter
        against.
      per_station_cap_usd: per-station daily cap (default $30)
      caller_kind: "no_momentum" (default) or "fast_reaction". Fast-
        reaction callers (Layer 7 / METAR_peak / METAR_early_tail) can
        consume up to the full daily cap. no_momentum callers see an
        effective limit of `daily_limit_usd - fast_reaction_reserve_usd`,
        guaranteeing the fast-reaction layers always have room.
      fast_reaction_reserve_usd: how much of daily_limit_usd is reserved
        for fast-reaction. None → DEFAULT_FAST_REACTION_RESERVE_USD ($30).

    Returns:
      (success, reason):
        (True,  "ok")            → safe to place; counters consumed
        (False, "global_cap")    → global cap would be exceeded
        (False, "no_momentum_cap_exhausted") → no_momentum reserve full
        (False, "station_cap(X)") → per-station cap would be exceeded

    The cap-reservation rule (Day 3 finding 2026-05-18): without this,
    NO_momentum at 30 placements/day × $5 = $150 fully consumes the cap
    in the first minute of the UTC day, leaving Layer 7 + METAR layers
    with nothing. Now NO_momentum effectively caps at $120 and the $30
    reserve is held for fast-reaction.

    fcntl exclusive lock guarantees atomicity across daemon + cron.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if per_station_cap_usd is None:
        per_station_cap_usd = DEFAULT_PER_STATION_CAP_USD
    if fast_reaction_reserve_usd is None:
        fast_reaction_reserve_usd = DEFAULT_FAST_REACTION_RESERVE_USD

    # Audit F5-M3: assert per-station ≤ daily-limit invariant.
    if per_station_cap_usd > daily_limit_usd:
        print(f"!! [cap_budget] per_station_cap_usd ${per_station_cap_usd:.2f} "
              f"> daily_limit_usd ${daily_limit_usd:.2f} — clamping per-station "
              f"to global to preserve invariant")
        per_station_cap_usd = daily_limit_usd

    if not station_id:
        raise ValueError("station_id is required (must be a non-empty string)")
    if caller_kind not in ("no_momentum", "fast_reaction"):
        raise ValueError(
            f"caller_kind must be 'no_momentum' or 'fast_reaction', got {caller_kind!r}"
        )

    # Effective limit depends on caller:
    #   - no_momentum: must leave the reserve untouched → limit = daily - reserve
    #   - fast_reaction: full cap available
    if caller_kind == "no_momentum":
        effective_limit = max(0.0, daily_limit_usd - fast_reaction_reserve_usd)
    else:
        effective_limit = daily_limit_usd

    if not _FCNTL_AVAILABLE:
        ok = _acquire_non_atomic(
            size_usd, effective_limit, state_path,
            station_id=station_id, per_station_cap_usd=per_station_cap_usd,
        )
        return ok, ("ok" if ok else "non_atomic_check")

    mode = "r+" if state_path.exists() else "w+"
    with open(state_path, mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = _read_or_init(f)
            # Global cap check uses caller-specific effective limit
            if data["deployed_usd"] + size_usd > effective_limit:
                reason = (
                    "no_momentum_cap_exhausted" if caller_kind == "no_momentum"
                    else "global_cap"
                )
                return False, reason
            # Per-station cap check
            per_station = data.setdefault("per_station_deployed_usd", {})
            cur = float(per_station.get(station_id, 0.0))
            if cur + size_usd > per_station_cap_usd:
                return False, f"station_cap({station_id})"
            per_station[station_id] = cur + float(size_usd)
            data["deployed_usd"] += float(size_usd)
            data["reservation_count"] += 1
            _write_state(f, data)
            return True, "ok"
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def release_cap_token(
    size_usd: float,
    state_path: Path = DEFAULT_CAP_BUDGET_PATH,
    *,
    station_id: str,
) -> None:
    """Refund `size_usd` back to BOTH the global counter and the
    per-station counter. Called when a previously successful acquire
    turned out to not result in a real deployment (order rejected,
    exception, etc.).

    Args:
      size_usd: USD to refund (must match the acquire amount)
      station_id: REQUIRED. The station to refund the per-station counter
        for. Was previously optional with a None default that would
        silently refund only the global counter — leaving the per-station
        counter inflated and eventually permanently locking out the
        station (audit F5-H4). Now keyword-required to force callers
        to pair acquire/release symmetrically.

    Safe to call when no acquire was made (counters floor at 0).
    """
    if not station_id:
        raise ValueError("station_id is required (must be a non-empty string)")
    if not state_path.exists():
        return
    if not _FCNTL_AVAILABLE:
        _release_non_atomic(size_usd, state_path, station_id=station_id)
        return

    with open(state_path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = _read_or_init(f)
            data["deployed_usd"] = max(0.0, data["deployed_usd"] - float(size_usd))
            data["release_count"] += 1
            if station_id:
                per_station = data.setdefault("per_station_deployed_usd", {})
                cur = float(per_station.get(station_id, 0.0))
                per_station[station_id] = max(0.0, cur - float(size_usd))
            _write_state(f, data)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_current_state(state_path: Path = DEFAULT_CAP_BUDGET_PATH) -> dict:
    """Read-only snapshot of the current cap state. For monitoring."""
    if not state_path.exists():
        return {
            "date_utc": _today_utc_iso(),
            "deployed_usd": 0.0,
            "reservation_count": 0,
            "release_count": 0,
            "last_updated_utc": None,
        }
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            if _FCNTL_AVAILABLE:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.loads(f.read())
            finally:
                if _FCNTL_AVAILABLE:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data
    except (json.JSONDecodeError, OSError):
        return {}


# ──────────────────────────────────────────────────────────────────────────
# Non-atomic fallbacks (Windows local dev only — live bot is Linux)
# ──────────────────────────────────────────────────────────────────────────

def _acquire_non_atomic(
    size_usd: float, daily_limit_usd: float, state_path: Path,
    station_id: str | None = None,
    per_station_cap_usd: float | None = None,
) -> bool:
    """Best-effort acquire without fcntl. Used on Windows local dev."""
    if per_station_cap_usd is None:
        per_station_cap_usd = DEFAULT_PER_STATION_CAP_USD
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    today = _today_utc_iso()
    if data.get("date_utc") != today:
        data = {"date_utc": today, "deployed_usd": 0.0,
                "reservation_count": 0, "release_count": 0,
                "per_station_deployed_usd": {}}
    if data.get("deployed_usd", 0.0) + size_usd > daily_limit_usd:
        return False
    if station_id:
        per_station = data.setdefault("per_station_deployed_usd", {})
        cur = float(per_station.get(station_id, 0.0))
        if cur + size_usd > per_station_cap_usd:
            return False
        per_station[station_id] = cur + float(size_usd)
    data["deployed_usd"] = data.get("deployed_usd", 0.0) + float(size_usd)
    data["reservation_count"] = data.get("reservation_count", 0) + 1
    data["last_updated_utc"] = _now_utc_iso()
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
    return True


def _release_non_atomic(
    size_usd: float, state_path: Path, station_id: str | None = None,
) -> None:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    data["deployed_usd"] = max(0.0, data.get("deployed_usd", 0.0) - float(size_usd))
    data["release_count"] = data.get("release_count", 0) + 1
    if station_id:
        per_station = data.setdefault("per_station_deployed_usd", {})
        cur = float(per_station.get(station_id, 0.0))
        per_station[station_id] = max(0.0, cur - float(size_usd))
    data["last_updated_utc"] = _now_utc_iso()
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
