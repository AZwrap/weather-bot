"""Portfolio-level position tracking + correlation-aware caps.

Added 2026-05-14 to support live trading. The bot's safety layer
previously had only `max_total_exposure_usd` which was a PER-SCAN cap.
That left no protection against:

  1. **Re-submission across scans** — same NO_momentum opportunity sat
     in the book hours; each 20-min cron tick would re-fire it.
  2. **Cluster correlation** — cold front busts NYC + BOS + DC + EWR
     simultaneously. Naive per-trade Kelly treats those as independent
     bets, gets caught with 4× correlated exposure.
  3. **No portfolio-level concurrent cap** — total open positions could
     drift to 100%+ of bankroll without tripping any guard.

This module addresses all of them via a single persisted `Portfolio`
state file (`data/portfolio.json`) loaded fresh at the start of each
scan and updated on every submission / resolution.

Five guarantees:
  - DEDUPE: `is_open(token_id, side)` skips signals already submitted
  - PORTFOLIO CAP: total $$$ cap on the sum of `position_usd` across
    open positions (DIFFERENT from `max_total_exposure_usd`, which is
    per-scan and resets each cron tick)
  - PER-REGION CAP: hard cap on total $ in any one weather region
    (cluster correlation proxy)
  - PER-EVENT CAP: max $ across all buckets of one event
  - PORTFOLIO KELLY: position size scaled down by correlation-adjusted
    bankroll utilization (vs naive per-trade Kelly that ignores other
    open positions)

State file format (`data/portfolio.json`):
  {
    "positions": [
      {Position fields...},
      ...
    ],
    "version": 1
  }

Stale positions (target_date past + resolved_at None) are pruned at
load time and logged. Resolution is the caller's responsibility — a
separate `resolve_portfolio.py` script should mark positions resolved
after end-of-day, freeing up their cap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:  # Windows local dev
    _FCNTL_AVAILABLE = False


DEFAULT_PORTFOLIO_PATH = Path("data/portfolio.json")

Side = Literal["YES", "NO"]
PositionStatus = Literal["submitted", "filled", "resolved", "cancelled"]


# Region taxonomy — coarse synoptic-weather correlation groups.
# Stations in the same region share large-scale weather systems
# (heat waves, cold fronts, etc.) so positions there should be
# treated as correlated for sizing purposes.
REGION_BY_STATION: dict[str, str] = {
    # US East — Northeast + SE + East Canada (Toronto)
    "KLGA": "US_East",   # NYC
    "KMIA": "US_East",   # Miami
    "KATL": "US_East",   # Atlanta
    "CYYZ": "US_East",   # Toronto (correlated with NYC weather)
    # US Central
    "KORD": "US_Central",  # Chicago
    "KDAL": "US_Central",  # Dallas
    "KHOU": "US_Central",  # Houston
    "KAUS": "US_Central",  # Austin
    # US West
    "KBKF": "US_West",   # Denver
    "KLAX": "US_West",   # Los Angeles
    "KSEA": "US_West",   # Seattle
    "KSFO": "US_West",   # San Francisco
    # Europe West
    "EGLC": "Europe_West",  # London
    "LEMD": "Europe_West",  # Madrid
    "LFPB": "Europe_West",  # Paris
    "EHAM": "Europe_West",  # Amsterdam
    # Europe Central / South
    "EDDM": "Europe_Central",  # Munich
    "EPWA": "Europe_Central",  # Warsaw
    "LIMC": "Europe_Central",  # Milan
    # Europe Nordic
    "EFHK": "Europe_Nordic",   # Helsinki
    # Europe East
    "UUWW": "Europe_East",  # Moscow
    "LTFM": "Europe_East",  # Istanbul
    "LTAC": "Europe_East",  # Ankara
    # Asia East
    "RJTT": "Asia_East",  # Tokyo
    "ZBAA": "Asia_East",  # Beijing
    "ZSPD": "Asia_East",  # Shanghai
    "ZHHH": "Asia_East",  # Wuhan
    "ZUUU": "Asia_East",  # Chengdu
    "ZUCK": "Asia_East",  # Chongqing
    "ZGGG": "Asia_East",  # Guangzhou
    "ZGSZ": "Asia_East",  # Shenzhen
    "ZSQD": "Asia_East",  # Qingdao
    "RCSS": "Asia_East",  # Taipei
    "RKSI": "Asia_East",  # Seoul
    "RKPK": "Asia_East",  # Busan
    # Asia SE
    "WSSS": "Asia_SE",   # Singapore
    "WMKK": "Asia_SE",   # Kuala Lumpur
    "WIHH": "Asia_SE",   # Jakarta
    "RPLL": "Asia_SE",   # Manila
    # Asia S
    "VILK": "Asia_S",    # Lucknow
    "OPMR": "Asia_S",    # Karachi
    # Middle East
    "LLBG": "Middle_East",  # Tel Aviv
    "OEJN": "Middle_East",  # Jeddah
    # Oceania
    "NZWN": "Oceania",   # Wellington
    # Africa
    "DNMM": "Africa",    # Lagos (excluded from live — see oracle_source_risk)
    "FACT": "Africa",    # Cape Town
    # Latin America
    "MMMX": "LatAm",     # Mexico City
    "MPMG": "LatAm",     # Panama City
    "SAEZ": "LatAm",     # Buenos Aires
    "SBGR": "LatAm",     # São Paulo
}


def region_for(station_id: str) -> str:
    """Return the region cluster for a station, or 'Unknown' if unmapped."""
    return REGION_BY_STATION.get(station_id, "Unknown")


# Cluster correlation coefficients. Used by `portfolio_kelly_multiplier`
# to scale down position sizes when concentrated in one region.
# Values are rough synoptic-scale estimates; refine empirically when
# we have N>=30 days of multi-station resolved data.
SAME_EVENT_CORRELATION: float = 0.95
"""Two NO_momentum positions on different buckets of the SAME event
are nearly perfectly correlated: exactly one bucket wins, so 'is this
bucket a loser' bets co-vary almost perfectly."""

SAME_REGION_CORRELATION: float = 0.5
"""Two positions on different stations in the same region (e.g., NYC
and Boston) share the synoptic weather pattern. Estimate; tune later."""

CROSS_REGION_CORRELATION: float = 0.05
"""Different regions are nearly independent on a daily basis."""


# ── Cancellation retry policy ──────────────────────────────────────────
CANCELLATION_COOLDOWN_MIN: int = 60
"""After Polymarket rejects an order, wait at least this many minutes
before re-submitting the same (token, side). Prevents tight retry
loops on structurally-bad orders (insufficient balance, invalid price,
stale tick size, etc.)."""

PERMANENT_BLOCK_AFTER_N_CANCELS: int = 3
"""After this many cancellations on the same (token, side) in the
same lifetime, the dedupe layer permanently blocks further attempts.
Manual review required to unblock (edit portfolio.json or wait
for `prune_resolved`)."""


# ── Adaptive bankroll ──────────────────────────────────────────────────
ADAPTIVE_BANKROLL_CEILING: float = 2000.0
"""Hard ceiling on the effective bankroll once realized PnL compounds.
Locks-in conservative posture even after big wins — the bot won't size
beyond $2k worth of caps regardless of how much it earns."""

# Cap ratios (fraction of effective bankroll). Used by `scaled_caps()`
# to derive portfolio + region + event caps from current bankroll.
PORTFOLIO_CAP_RATIO: float = 0.80
"""Concurrent open positions can sum to 80% of effective bankroll.
Leaves 20% headroom for in-flight orders + unsettled cancellations."""

PER_REGION_CAP_RATIO: float = 0.20
"""Max 20% of bankroll in any one synoptic-weather region. Bounds
cluster correlation: cold front busts US_East simultaneously across
KLGA, KMIA, KATL, CYYZ."""

PER_EVENT_CAP_RATIO: float = 0.11
"""Max 11% of bankroll across all buckets of one Polymarket event.
Sized to cover all 11 buckets at $5/trade for a $500 bankroll
($500 × 0.11 = $55 = 11 × $5)."""


@dataclass
class Position:
    """One open or recently-closed live position.

    NOTE: this is the LIVE position record, distinct from `BucketSnapshot`
    (which is a passive forecast log). Created when the bot submits an
    order; resolved at end-of-day by `resolve_portfolio.py`.
    """

    # Identity (used for dedupe)
    token_id: str            # Polymarket CLOB token id (the side we hold)
    side: Side               # "YES" or "NO"
    # Market metadata (for cluster grouping + cap checks)
    station_id: str
    region: str              # derived from station_id at submit time
    market_id: int
    bucket_label: str
    bucket_kind: str         # "low_tail" | "mid" | "high_tail"
    bucket_threshold: int
    target_date: str         # ISO; the date the market resolves
    # Trade economics
    shares: float
    entry_price: float       # avg fill price (after depth walk)
    position_usd: float      # capital deployed = shares * entry_price
    # Lifecycle
    submitted_at: str        # ISO datetime UTC
    status: PositionStatus = "submitted"
    order_id: str | None = None   # Polymarket order ID, when SDK is wired
    resolved_at: str | None = None
    realized_pnl: float | None = None
    strategy: str = ""       # "NO_momentum" | "METAR_peak" | "METAR_early_tail" | etc.
    # GTD-with-TTL support (added 2026-05-18). For GTD orders:
    #   - expires_at_utc: ISO timestamp when the on-chain order expires.
    #     When this time passes, Polymarket auto-cancels the order
    #     (status → EXPIRED). poll_fills marks the position as
    #     cancelled with reason="gtd_expired".
    #   - filled_at: ISO timestamp when status transitioned from
    #     submitted → filled (instrumentation for tuning TTL based on
    #     observed fill-time distribution).
    expires_at_utc: str | None = None
    filled_at: str | None = None
    # Retry tracking (added 2026-05-14): counts cancellations on this
    # (token, side) lifetime. Once it hits PERMANENT_BLOCK_AFTER_N_CANCELS
    # the order is blocked indefinitely. last_cancelled_at gates the
    # CANCELLATION_COOLDOWN_MIN retry-cooldown.
    cancellation_count: int = 0
    last_cancelled_at: str | None = None
    cancellation_reason: str | None = None

    # Maker-rebate attribution (added 2026-05-14). For positions filled
    # as MAKER (resting limit crossed by a taker), Polymarket pays a
    # liquidity-program rebate. The aggregate daily rebate is tracked
    # on Portfolio.daily_maker_rebates; this per-position field is
    # populated when (or if) Polymarket exposes per-trade attribution
    # via a future SDK endpoint. Leave None until that's wired.
    maker_rebate_usd: float | None = None

    # Cross-up SELL retry suppression (added 2026-05-19, Fix A).
    # When True, cross_up_cancel will SKIP attempting to SELL this
    # position even if the cross-up condition is detected. Rationale:
    # analyze_cross_up_failures.py (2026-05-19) showed that of 1,426
    # cross-up SELL attempts, only 10 were successful (0.7%) -- and
    # ALL 10 succeeded on the FIRST attempt for the position. The
    # remaining 1,416 attempts were retries on 5 stuck positions where
    # the market had already collapsed NO_bid to near zero. Retrying
    # only generates log noise + API calls -- no recovery is possible
    # once the bid has dropped below the proceeds floor.
    # Flag is set to True by cross_up_cancel after the FIRST failed
    # SELL (below_min / no_bid / rejected). Reset to False only via
    # external action (e.g., a fresh position on the same token after
    # cancel + re-place).
    cross_up_sell_failed: bool = False

    # last_modified_utc — set on EVERY mutation (add, mark_resolved,
    # mark_cancelled, poll_fills status transitions, manual reconcile).
    # Portfolio.save() uses this to tiebreak the merge for shared-key
    # positions: later timestamp wins, instead of always "in-memory wins".
    #
    # Root cause of the 2026-05-21 LTFM merge bug + 2026-05-22 reconcile
    # reversion + 2026-05-23 poll_resolutions not persisting: when a
    # long-running process loaded the portfolio with stale values, then
    # called save() AFTER another process had updated those positions on
    # disk, the merge picked the stale in-memory copy. With a
    # last_modified_utc field, the merge picks whichever copy has the
    # newer timestamp -- the fresh one always wins.
    last_modified_utc: str | None = None

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, d: dict) -> "Position":
        return cls(
            token_id=d["token_id"],
            side=d["side"],
            station_id=d["station_id"],
            region=d.get("region", region_for(d["station_id"])),
            market_id=int(d["market_id"]),
            bucket_label=d.get("bucket_label", ""),
            bucket_kind=d.get("bucket_kind", "mid"),
            bucket_threshold=int(d.get("bucket_threshold", 0)),
            target_date=d["target_date"],
            shares=float(d["shares"]),
            entry_price=float(d["entry_price"]),
            position_usd=float(d["position_usd"]),
            submitted_at=d["submitted_at"],
            status=d.get("status", "submitted"),
            order_id=d.get("order_id"),
            resolved_at=d.get("resolved_at"),
            realized_pnl=(
                float(d["realized_pnl"]) if d.get("realized_pnl") is not None else None
            ),
            strategy=d.get("strategy", ""),
            cancellation_count=int(d.get("cancellation_count", 0)),
            last_cancelled_at=d.get("last_cancelled_at"),
            cancellation_reason=d.get("cancellation_reason"),
            maker_rebate_usd=(
                float(d["maker_rebate_usd"])
                if d.get("maker_rebate_usd") is not None else None
            ),
            expires_at_utc=d.get("expires_at_utc"),
            filled_at=d.get("filled_at"),
            cross_up_sell_failed=bool(d.get("cross_up_sell_failed", False)),
            last_modified_utc=d.get("last_modified_utc"),
        )


@dataclass
class Portfolio:
    """In-memory state, mirrors `data/portfolio.json` on disk.

    Use `Portfolio.load(path)` to get the current state at the start of
    each scan; `add()`/`mark_resolved()` to mutate; `save(path)` to
    persist. Concurrent cron runs should be rare (cron is sequential)
    but a file lock could be added if needed.
    """

    positions: list[Position] = field(default_factory=list)
    version: int = 1

    # Maker-rebate accounting (added 2026-05-14). Polymarket's liquidity
    # rewards program pays makers daily at midnight UTC ($1 minimum
    # threshold per `polymarket_live_trading_lessons.md`). We track
    # daily totals here keyed by date_iso (yyyy-mm-dd UTC). Populated
    # by `scripts/sync_maker_rebates.py` or manually via the CLI tool.
    # Feeds `realized_pnl_total()` so adaptive bankroll compounds the
    # rebate income alongside resolution PnL.
    daily_maker_rebates: dict[str, float] = field(default_factory=dict)

    # Progressive-eval tracker for Layer 7 + high-bucket NO (2026-05-28).
    # Key: f"{station_id}|{target}|{target_date_iso}". Value: integer
    # max-so-far observation in the station's market unit (°C or °F),
    # i.e. the last `_rounded_observation` we've already evaluated dead
    # buckets up to. Used to avoid re-checking buckets that became dead
    # in earlier WUG ticks.
    #
    # On WUG update W > last_evaluated[key]: iterate dead steps from
    # last_evaluated + 1 to W inclusive, evaluate each containing bucket
    # exactly once, then advance the tracker.
    last_evaluated_max_by_sk: dict[str, int] = field(default_factory=dict)

    # Trailing-stop peak tracker for consensus_yes exits. Keyed by
    # f"{token_id}|{side}" (matches Position dedupe key). Records the
    # highest yes_ask we've observed on this position since entry.
    # consensus_yes exit triggers when current_yes_ask < peak AND
    # current_yes_ask >= entry_price + 0.05. See
    # weather_bot/consensus_yes.evaluate_consensus_yes_exits().
    consensus_yes_peak_by_pos: dict[str, float] = field(default_factory=dict)

    # ── Persistence ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path = DEFAULT_PORTFOLIO_PATH) -> "Portfolio":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # On parse error, return an empty portfolio rather than crash.
            # The caller should log a warning.
            # AUDIT (2026-05-22): record this so we catch silent data
            # loss caused by load-returns-empty paths.
            try:
                import os as _os
                audit_path = Path("data/portfolio_save_audit.jsonl")
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "event": "load_failed_returned_empty",
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "pid": _os.getpid(),
                        "path": str(path),
                    }) + "\n")
            except Exception:
                pass
            return cls()
        positions = [Position.from_jsonable(p) for p in data.get("positions", [])]
        # daily_maker_rebates: dict of date_iso -> float. Older files won't have it.
        rebates_raw = data.get("daily_maker_rebates", {}) or {}
        rebates: dict[str, float] = {}
        if isinstance(rebates_raw, dict):
            for k, v in rebates_raw.items():
                try:
                    rebates[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        # last_evaluated_max_by_sk: dict of "sid|target|date" → int.
        # Older files won't have it. Skip malformed values silently.
        le_raw = data.get("last_evaluated_max_by_sk", {}) or {}
        last_eval: dict[str, int] = {}
        if isinstance(le_raw, dict):
            for k, v in le_raw.items():
                try:
                    last_eval[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
        # consensus_yes_peak_by_pos: dict of "token|side" → float.
        # Older portfolio.json files won't have it.
        peak_raw = data.get("consensus_yes_peak_by_pos", {}) or {}
        peaks: dict[str, float] = {}
        if isinstance(peak_raw, dict):
            for k, v in peak_raw.items():
                try:
                    peaks[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            positions=positions,
            version=int(data.get("version", 1)),
            daily_maker_rebates=rebates,
            last_evaluated_max_by_sk=last_eval,
            consensus_yes_peak_by_pos=peaks,
        )

    def _audit_save(self, event: dict) -> None:
        """Append a save-audit record to data/portfolio_save_audit.jsonl.

        Best-effort: never raises on logging failure. Designed to capture
        enough state to debug the 2026-05-22 LTFM-style 'positions
        disappear silently' failure mode if it recurs after the
        dedupe + station-local-date + append-only-log defenses.

        Each record captures: PID, caller (top 3 stack frames),
        in-memory count before, on-disk count read inside lock, merged
        count, post-write verified count. Mismatches between merged and
        verified counts = smoking gun for where positions get lost.
        """
        try:
            import os as _os
            import traceback as _tb
            event["pid"] = _os.getpid()
            event["ts_utc"] = datetime.now(timezone.utc).isoformat()
            # Top 3 frames of caller stack (excluding _audit_save + save)
            event["stack"] = [
                f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
                for f in _tb.extract_stack()[-6:-2]
            ]
            audit_path = Path("data/portfolio_save_audit.jsonl")
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass  # never disrupt save on logging failure

    def save(self, path: Path = DEFAULT_PORTFOLIO_PATH) -> None:
        """Cross-process safe save (2026-05-21 race fix) + atomic write.

        TWO failure modes are guarded:

        1. Torn write (process killed mid-write). `atomic_write_json`
           writes a tempfile then os.replaces -- concurrent readers
           always see either the old file or the new, never partial.

        2. Read-modify-write race across processes. Daemon (5s cycle)
           and intraday_scan cron (15min) both load portfolio.json,
           mutate in memory, save. Without a lock, the LATER saver
           overwrites earlier-saver positions it never knew existed.

           Fix: acquire fcntl.LOCK_EX on a separate .lock file, RELOAD
           the on-disk state inside the lock, MERGE in any positions
           added by the other process since we last loaded, then write.

        Merge policy:
          - Positions in BOTH: in-memory wins (we just mutated them;
            stale-disk status would overwrite our intended change)
          - Positions ONLY on disk: keep them (added by other process)
          - Positions ONLY in memory: keep them (added by us)

        Trade-off: concurrent mutations of the SAME (token, side) lose
        the older one. Acceptable because (a) it's rare -- typically
        only one process mutates a given position, and (b) poll_fills
        re-syncs from Polymarket within one cycle.

        Without merge, the catastrophic case is concurrent ADDS of
        DIFFERENT positions: each process's add disappears in the other's
        save -- silent orphan-order, same severity as the 2026-05-21
        incident. The merge guarantees both adds survive.

        Lock file path: `<portfolio.json>.lock`. Separate from the data
        file because `atomic_write_json` does an os.replace, which
        invalidates any open file handle held by another process for
        locking purposes.

        Windows fallback: when fcntl is unavailable (local dev), do an
        atomic write without the lock/merge -- best-effort, matches the
        prior behavior. Live bot runs on Linux so the lock is active.
        """
        from weather_bot.atomic_write import atomic_write_json

        in_memory_count = len(self.positions)

        if not _FCNTL_AVAILABLE:
            payload = {
                "version": self.version,
                "positions": [p.to_jsonable() for p in self.positions],
                "daily_maker_rebates": dict(self.daily_maker_rebates),
                "last_evaluated_max_by_sk": dict(self.last_evaluated_max_by_sk),
                "consensus_yes_peak_by_pos": dict(self.consensus_yes_peak_by_pos),
            }
            atomic_write_json(path, payload)
            self._audit_save({
                "path": str(path), "fcntl": False,
                "in_memory_count": in_memory_count,
                "wrote_count": in_memory_count,
            })
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")

        with open(lock_path, "a+", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read on-disk state inside the lock. Any positions
                # added by another process since we last loaded will
                # appear here -- we MUST preserve them or they become
                # orphans on Polymarket.
                on_disk_positions: list[Position] = []
                on_disk_rebates: dict[str, float] = {}
                disk_read_error: str | None = None
                disk_raw_count: int | None = None
                if path.exists():
                    try:
                        on_disk = json.loads(path.read_text(encoding="utf-8"))
                        disk_raw_count = len(on_disk.get("positions", []))
                        on_disk_positions = [
                            Position.from_jsonable(p)
                            for p in on_disk.get("positions", [])
                        ]
                        on_disk_rebates_raw = on_disk.get("daily_maker_rebates", {}) or {}
                        if isinstance(on_disk_rebates_raw, dict):
                            for k, v in on_disk_rebates_raw.items():
                                try:
                                    on_disk_rebates[str(k)] = float(v)
                                except (TypeError, ValueError):
                                    continue
                    except (json.JSONDecodeError, OSError) as _disk_exc:
                        # Corrupt / unreadable disk state -- treat as empty
                        # so our in-memory positions are preserved.
                        on_disk_positions = []
                        on_disk_rebates = {}
                        disk_read_error = f"{type(_disk_exc).__name__}: {_disk_exc}"
                    except Exception as _disk_exc:
                        # NEW (2026-05-22): catch broader exceptions from
                        # from_jsonable (KeyError, ValueError, TypeError).
                        # Previously these propagated and could mask data
                        # loss in concurrent-save scenarios. Audit the
                        # exception so we can see if this fires.
                        on_disk_positions = []
                        on_disk_rebates = {}
                        disk_read_error = f"{type(_disk_exc).__name__}: {_disk_exc}"

                # TIMESTAMP-AWARE MERGE (2026-05-23 fix). Earlier merge
                # picked "in-memory wins" for shared keys, which broke
                # when a long-running process had stale in-memory data
                # and saved AFTER another process updated those positions
                # on disk. Caused: (a) 2026-05-21 LTFM 4/5 lost fills,
                # (b) 2026-05-22 reconcile reversion, (c) 2026-05-23
                # poll_resolutions not persisting. All same root cause.
                #
                # New rule: for shared keys, pick the position with the
                # later last_modified_utc. Every mutation method
                # (add/mark_resolved/mark_cancelled/poll_fills status
                # transitions) calls _stamp(p) to set this timestamp.
                # Legacy positions without the field fall back to
                # in-memory-wins (preserves prior behavior).
                in_memory_by_key = {
                    (p.token_id, p.side, p.submitted_at): p
                    for p in self.positions
                }
                on_disk_by_key = {
                    (p.token_id, p.side, p.submitted_at): p
                    for p in on_disk_positions
                }
                all_keys = set(in_memory_by_key) | set(on_disk_by_key)
                merged_positions: list[Position] = []
                disk_only_count = 0
                disk_wins_count = 0  # shared key where disk's last_modified is newer
                for key in all_keys:
                    in_mem = in_memory_by_key.get(key)
                    disk = on_disk_by_key.get(key)
                    if in_mem is None:
                        merged_positions.append(disk)  # type: ignore[arg-type]
                        disk_only_count += 1
                    elif disk is None:
                        merged_positions.append(in_mem)
                    else:
                        in_ts = in_mem.last_modified_utc
                        disk_ts = disk.last_modified_utc
                        if in_ts is None and disk_ts is None:
                            merged_positions.append(in_mem)
                        elif in_ts is None:
                            merged_positions.append(disk)
                            disk_wins_count += 1
                        elif disk_ts is None:
                            merged_positions.append(in_mem)
                        elif in_ts >= disk_ts:
                            merged_positions.append(in_mem)
                        else:
                            merged_positions.append(disk)
                            disk_wins_count += 1

                # Maker rebates: take the max per-day across both
                # (rebates accumulate monotonically; max is safe).
                merged_rebates = dict(self.daily_maker_rebates)
                for k, v in on_disk_rebates.items():
                    merged_rebates[k] = max(merged_rebates.get(k, 0.0), v)

                # last_evaluated_max_by_sk: monotonically rising per
                # (station,target,date) since the daily extreme only
                # moves outward → take max across both sides.
                on_disk_le: dict[str, int] = {}
                try:
                    on_disk_le_raw = on_disk.get("last_evaluated_max_by_sk", {}) or {}
                    if isinstance(on_disk_le_raw, dict):
                        for k, v in on_disk_le_raw.items():
                            try:
                                on_disk_le[str(k)] = int(v)
                            except (TypeError, ValueError):
                                continue
                except Exception:
                    on_disk_le = {}
                merged_le = dict(self.last_evaluated_max_by_sk)
                for k, v in on_disk_le.items():
                    merged_le[k] = max(merged_le.get(k, v), v)

                # consensus_yes_peak_by_pos: each key is per (token,
                # side). Take max across both sides since peak is
                # monotonically rising.
                on_disk_peaks: dict[str, float] = {}
                try:
                    on_disk_peaks_raw = on_disk.get("consensus_yes_peak_by_pos", {}) or {}
                    if isinstance(on_disk_peaks_raw, dict):
                        for k, v in on_disk_peaks_raw.items():
                            try:
                                on_disk_peaks[str(k)] = float(v)
                            except (TypeError, ValueError):
                                continue
                except Exception:
                    on_disk_peaks = {}
                merged_peaks = dict(self.consensus_yes_peak_by_pos)
                for k, v in on_disk_peaks.items():
                    merged_peaks[k] = max(merged_peaks.get(k, v), v)

                payload = {
                    "version": self.version,
                    "positions": [p.to_jsonable() for p in merged_positions],
                    "daily_maker_rebates": merged_rebates,
                    "last_evaluated_max_by_sk": merged_le,
                    "consensus_yes_peak_by_pos": merged_peaks,
                }
                atomic_write_json(path, payload)

                # POST-WRITE VERIFY (2026-05-22 instrumentation): re-read
                # disk and confirm count matches what we just wrote. Any
                # mismatch = data loss bug to catch in the audit log.
                verified_count: int | None = None
                try:
                    verified = json.loads(path.read_text(encoding="utf-8"))
                    verified_count = len(verified.get("positions", []))
                except Exception:
                    pass

                self._audit_save({
                    "path": str(path), "fcntl": True,
                    "in_memory_count": in_memory_count,
                    "disk_raw_count": disk_raw_count,
                    "disk_parsed_count": len(on_disk_positions),
                    "disk_read_error": disk_read_error,
                    "disk_only_count": disk_only_count,
                    "merged_count": len(merged_positions),
                    "verified_count": verified_count,
                    "loss_detected": (
                        verified_count is not None
                        and verified_count < len(merged_positions)
                    ),
                })

                # Reflect merged state in memory so subsequent mutations
                # see disk-only entries we just absorbed (prevents the
                # next save() from re-merging the same disk positions
                # AGAIN as "disk-only").
                self.positions = merged_positions
                self.daily_maker_rebates = merged_rebates
                self.last_evaluated_max_by_sk = merged_le
                self.consensus_yes_peak_by_pos = merged_peaks
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    # ── Dedupe + retry policy ──────────────────────────────────────────

    def is_open(self, token_id: str, side: Side) -> bool:
        """Already have an open or pending position on this (token, side)?"""
        for p in self.positions:
            if p.token_id == token_id and p.side == side and p.status in (
                "submitted", "filled"
            ):
                return True
        return False

    # ── Progressive-eval tracker (Layer 7 + high-bucket NO) ────────────

    @staticmethod
    def _sk_key(station_id: str, target: str, target_date_iso: str) -> str:
        return f"{station_id}|{target}|{target_date_iso}"

    def get_last_evaluated_max(
        self, station_id: str, target: str, target_date_iso: str,
    ) -> int | None:
        """Return the last integer extreme we evaluated dead buckets up
        to for this (station, target, date), or None if unseen.

        For `target="max"`: the integer °C/°F that the running max has
        reached. New WUG readings strictly higher than this trigger
        progressive evaluation of newly-dead buckets.
        For `target="min"`: the integer °C/°F the running min has
        reached (which moves DOWNWARD over the day). New WUG readings
        strictly lower than this trigger evaluation.
        """
        return self.last_evaluated_max_by_sk.get(
            self._sk_key(station_id, target, target_date_iso)
        )

    def set_last_evaluated_max(
        self, station_id: str, target: str, target_date_iso: str, value: int,
    ) -> None:
        """Advance the tracker. Monotone: for max-target only writes if
        value > existing; for min-target only writes if value < existing.
        """
        key = self._sk_key(station_id, target, target_date_iso)
        cur = self.last_evaluated_max_by_sk.get(key)
        if cur is None:
            self.last_evaluated_max_by_sk[key] = int(value)
            return
        if target == "max" and int(value) > cur:
            self.last_evaluated_max_by_sk[key] = int(value)
        elif target == "min" and int(value) < cur:
            self.last_evaluated_max_by_sk[key] = int(value)

    def _last_cancel_for(self, token_id: str, side: Side) -> Position | None:
        """Most-recent cancelled record for (token, side), or None."""
        latest: Position | None = None
        for p in self.positions:
            if p.token_id != token_id or p.side != side:
                continue
            if p.status != "cancelled":
                continue
            if latest is None or (
                p.last_cancelled_at and latest.last_cancelled_at and
                p.last_cancelled_at > latest.last_cancelled_at
            ):
                latest = p
        return latest

    def is_permanently_blocked(self, token_id: str, side: Side) -> bool:
        """True if this (token, side) has been cancelled ≥ N times."""
        # Tally across all records (may be multiple Position rows if
        # we re-attempted after cooldown). Use the last record's count
        # as the running tally — mark_cancelled increments it.
        last = self._last_cancel_for(token_id, side)
        if last is None:
            return False
        return last.cancellation_count >= PERMANENT_BLOCK_AFTER_N_CANCELS

    def is_in_cooldown(
        self, token_id: str, side: Side,
        cooldown_minutes: int = CANCELLATION_COOLDOWN_MIN,
        now: datetime | None = None,
    ) -> bool:
        """True if last cancellation is within `cooldown_minutes`."""
        last = self._last_cancel_for(token_id, side)
        if last is None or last.last_cancelled_at is None:
            return False
        try:
            ts = datetime.fromisoformat(last.last_cancelled_at)
        except ValueError:
            return False
        ref = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        elapsed_min = (ref - ts).total_seconds() / 60.0
        return elapsed_min < cooldown_minutes

    def should_skip(
        self, token_id: str, side: Side,
        *,
        cooldown_minutes: int = CANCELLATION_COOLDOWN_MIN,
        max_cancellations: int = PERMANENT_BLOCK_AFTER_N_CANCELS,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Combined gate: should the bot skip this signal entirely?

        Skips if:
          1. Already open or filled → dedupe
          2. Permanently blocked (≥ N cancellations)
          3. In cooldown window after most recent cancel

        Returns (skip, reason).
        """
        if self.is_open(token_id, side):
            return True, "already-open position dedupe"
        last = self._last_cancel_for(token_id, side)
        if last is not None:
            if last.cancellation_count >= max_cancellations:
                return True, (
                    f"permanently blocked after {last.cancellation_count} "
                    f"cancellations (last reason: {last.cancellation_reason or 'unknown'})"
                )
            if self.is_in_cooldown(token_id, side, cooldown_minutes, now):
                return True, (
                    f"cooldown active — cancelled {last.cancellation_count}× "
                    f"recently; retry after {cooldown_minutes} min"
                )
        return False, ""

    # ── Mutation ───────────────────────────────────────────────────────

    @staticmethod
    def _stamp(position: Position) -> None:
        """Set position.last_modified_utc to now. Called by every mutation
        method so the save() merge can pick the newer copy when shared-key
        races occur (see Portfolio.save docstring for the bug this guards)."""
        position.last_modified_utc = datetime.now(timezone.utc).isoformat()

    def add(self, position: Position) -> None:
        # Defensive: don't double-add. Caller should have checked is_open.
        if self.is_open(position.token_id, position.side):
            return
        self._stamp(position)
        self.positions.append(position)

    # ── Fill polling (added 2026-05-14) ────────────────────────────────

    def poll_fills(
        self,
        client,
        *,
        verbose: bool = False,
    ) -> dict[str, int]:
        """Reconcile our `submitted` positions against Polymarket's
        open-orders list. Mutates the portfolio in place.

        Logic:
          - Get the live open-orders list from Polymarket.
          - For each of our positions with status=='submitted' and a
            real order_id:
              * Still in open list → no change (resting)
              * NOT in open list → presume FILLED (most common case)

        Returns: dict of {filled, still_open, no_id} counts for logging.

        Caveat — can't distinguish FILLED vs user-CANCELLED:
          If you cancel an order via the Polymarket UI (not via the
          bot), this method will mark it as filled because it's no
          longer in open_orders. To avoid bad accounting:
            1. Don't cancel manually when the bot has the order on
               its books. Use a bot-side cancel that calls
               `mark_cancelled()` directly.
            2. Future improvement: cross-check `get_balance(token_id)`
               — if we have 0 of that token AFTER polling, override
               'filled' to 'cancelled'. Requires the SDK v2 balance
               endpoint.

        Caveat — pagination:
          Polymarket's `get_orders()` may truncate at some page size.
          If we have >N submitted orders, some "still_open" might be
          misclassified as filled. The current SDK call doesn't
          paginate. Bound risk by keeping total open orders below
          the limit; the portfolio cap helps here.

        Dry-run safety:
          If the client returns no open orders AND we have submitted
          positions with order_ids, that's ambiguous (real client
          returned empty vs dry-run returned empty). We don't mark
          ANYTHING filled in that case unless `client.dry_run is False`.
          Callers should pass a live client only.
        """
        submitted = [
            p for p in self.positions
            if p.status == "submitted" and p.order_id is not None
            and p.order_id != "dry-run"
        ]
        # Always count submitted-but-no-id positions for caller visibility,
        # regardless of whether the rest of the poll runs.
        no_id = sum(
            1 for p in self.positions
            if p.status == "submitted" and (
                p.order_id is None or p.order_id == "dry-run"
            )
        )
        result = {"filled": 0, "still_open": 0, "no_id": no_id, "skipped": 0}

        if not submitted:
            return result

        # If client is a dry-run shell, skip — we can't trust empty results
        is_dry_run = getattr(client, "_clob", None) is None
        if is_dry_run:
            result["skipped"] = len(submitted)
            return result

        try:
            open_orders = client.get_open_orders()
        except Exception as exc:
            if verbose:
                print(f"!! poll_fills get_open_orders failed: {exc}")
            result["skipped"] = len(submitted)
            return result

        # Polymarket order objects use either 'id' or 'orderID' keys
        # depending on SDK version; accept both. Build a dict so we can
        # also read size_matched for partial-fill detection.
        open_by_id: dict[str, dict] = {}
        for o in open_orders:
            if isinstance(o, dict):
                oid = o.get("id") or o.get("orderID")
                if oid:
                    open_by_id[str(oid)] = o

        result.setdefault("partial_fill", 0)
        result.setdefault("cancelled_externally", 0)
        result.setdefault("unknown", 0)

        for p in submitted:
            if p.order_id in open_by_id:
                # Order still resting. Detect PARTIAL fills via size_matched
                # — a maker order can be partially-matched while staying on
                # the book for the unfilled remainder. Reflect that in our
                # cap stack by transitioning status to 'filled' and shrinking
                # shares/position_usd to the actually-filled portion.
                info = open_by_id[p.order_id]
                try:
                    size_matched = float(info.get("size_matched", 0) or 0)
                except (ValueError, TypeError):
                    size_matched = 0.0
                if size_matched > 0 and p.entry_price > 0:
                    if p.filled_at is None:
                        # First time we observe this fill — stamp timestamp
                        # for fill-time-distribution analysis (GTD-TTL tuning).
                        p.filled_at = datetime.now(timezone.utc).isoformat()
                    p.status = "filled"
                    p.shares = size_matched
                    p.position_usd = size_matched * p.entry_price
                    self._stamp(p)
                    result["partial_fill"] += 1
                    if verbose:
                        print(
                            f"  partial-fill: {p.station_id} {p.side} {p.bucket_label} "
                            f"@ ${p.entry_price:.3f} ({p.shares:.2f} sh = "
                            f"${p.position_usd:.2f})  order_id={p.order_id[:12]}…"
                        )
                else:
                    result["still_open"] += 1
                continue

            # Order no longer on the book — query for the final disposition
            # so we can distinguish a true fill from an external cancellation
            # (Polymarket UI cancel, expiry, etc.). Without this, an
            # externally-cancelled order would be misaccounted as filled.
            info = None
            getter = getattr(client, "get_order", None)
            if getter is not None:
                try:
                    info = getter(p.order_id)
                except Exception as exc:
                    if verbose:
                        print(f"  warn: get_order({p.order_id[:14]}…) failed: {exc}")
                    info = None

            if info is None:
                # Couldn't determine from Polymarket. For GTD orders past
                # their TTL by a comfortable grace, presume they expired
                # (Polymarket forgets old orders past some retention
                # window, so get_order returns nothing for them).
                # Without this branch, stale 'submitted' positions
                # accumulate forever and inflate per-station / per-event
                # / per-region exposure caps -- silently throttling new
                # fires (2026-05-22 finding: 2 CYYZ NO_momentum positions
                # sat submitted for 10 hours, blocking new fires on CYYZ).
                if p.expires_at_utc:
                    try:
                        expires_dt = datetime.fromisoformat(p.expires_at_utc)
                        if expires_dt.tzinfo is None:
                            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        grace = 300.0  # 5min past expiry before presuming dead
                        if (now - expires_dt).total_seconds() > grace:
                            self.mark_cancelled(
                                p.token_id, p.side,
                                reason="gtd_ttl_presumed_expired "
                                       "(get_order returned nothing past TTL+grace)",
                            )
                            result["gtd_expired_presumed"] = (
                                result.get("gtd_expired_presumed", 0) + 1
                            )
                            if verbose:
                                age_min = (now - expires_dt).total_seconds() / 60
                                print(
                                    f"  gtd-presumed-expired: {p.station_id} "
                                    f"{p.side} {p.bucket_label} "
                                    f"(past TTL by {age_min:.0f}min) "
                                    f"order_id={p.order_id[:12]}…"
                                )
                            continue
                    except ValueError:
                        pass  # bad expires_at_utc string -- fall through
                # No TTL info or still within grace -- leave as submitted.
                result["unknown"] += 1
                continue

            status_str = str(info.get("status", "")).upper()
            if status_str == "MATCHED":
                # Fully filled
                try:
                    size_matched = float(info.get("size_matched", 0) or 0)
                except (ValueError, TypeError):
                    size_matched = 0.0
                if size_matched > 0 and p.entry_price > 0:
                    p.shares = size_matched
                    p.position_usd = size_matched * p.entry_price
                if p.filled_at is None:
                    p.filled_at = datetime.now(timezone.utc).isoformat()
                p.status = "filled"
                self._stamp(p)
                result["filled"] += 1
                if verbose:
                    print(
                        f"  fill: {p.station_id} {p.side} {p.bucket_label} "
                        f"@ ${p.entry_price:.3f} ({p.shares:.2f} sh)  "
                        f"order_id={p.order_id[:12]}…"
                    )
            elif (
                "CANC" in status_str
                or "EXPIR" in status_str
                or status_str == "INVALID"
                or status_str == "REJECTED"
            ):
                # EXPIRED: GTD order's TTL passed without a fill — expected
                # part of the GTD-with-TTL design (#2). Next cron tick
                # re-evaluates against all placement gates and re-places
                # if still eligible. Tracked separately from other cancels
                # for tuning the optimal TTL.
                # INVALID: Polymarket auto-invalidated (e.g., collateral
                # exhausted while resting).
                # CANCELED / REJECTED: external cancel from UI / API.
                is_expired = "EXPIR" in status_str
                reason_str = (
                    "gtd_expired" if is_expired
                    else f"externally {status_str.lower()} (poll_fills detected)"
                )
                self.mark_cancelled(p.token_id, p.side, reason=reason_str)
                result_key = "gtd_expired" if is_expired else "cancelled_externally"
                result[result_key] = result.get(result_key, 0) + 1
                if verbose:
                    tag = "gtd-expired" if is_expired else "external-cancel"
                    print(
                        f"  {tag}: {p.station_id} {p.side} {p.bucket_label}  "
                        f"order_id={p.order_id[:12]}…  ({status_str})"
                    )
            else:
                # Unexpected status. Mirror the past-TTL handling in the
                # primary (info-is-None) branch above (Task #39 audit,
                # 2026-05-25): if the order's GTD TTL is past+grace and
                # we got back a status we don't recognize, presume it's
                # dead and free the cap slot. Otherwise the position
                # accumulates indefinitely in `submitted`, consuming
                # per-station/per-region/per-event cap budget and
                # silently throttling new fires.
                #
                # Production evidence: data/log.out shows `'unknown': 2`
                # persisting across multiple poll_fills runs — without
                # the symmetric TTL gate, those 2 positions were stuck
                # blocking cap budget. Same root cause as the CYYZ
                # 10-hour stuck-submitted incident (2026-05-22), one
                # control flow over.
                presumed_dead = False
                if p.expires_at_utc:
                    try:
                        expires_dt = datetime.fromisoformat(p.expires_at_utc)
                        if expires_dt.tzinfo is None:
                            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        grace = 300.0  # 5min past expiry (matches primary branch)
                        if (now - expires_dt).total_seconds() > grace:
                            self.mark_cancelled(
                                p.token_id, p.side,
                                reason=(
                                    f"unrecognized_status_past_ttl "
                                    f"(status={status_str!r}, "
                                    f"get_order returned non-standard status "
                                    f"past TTL+grace)"
                                ),
                            )
                            result["gtd_expired_presumed_secondary"] = (
                                result.get("gtd_expired_presumed_secondary", 0) + 1
                            )
                            presumed_dead = True
                            if verbose:
                                age_min = (now - expires_dt).total_seconds() / 60
                                print(
                                    f"  unrecognized-status-past-ttl: "
                                    f"{p.station_id} {p.side} {p.bucket_label} "
                                    f"(status={status_str!r}, past TTL by "
                                    f"{age_min:.0f}min) "
                                    f"order_id={p.order_id[:12]}…"
                                )
                    except ValueError:
                        pass  # bad expires_at_utc string — fall through to unknown

                if not presumed_dead:
                    result["unknown"] += 1
                    if verbose:
                        print(
                            f"  warn: unexpected status '{status_str}' for "
                            f"{p.station_id} {p.bucket_label} "
                            f"order_id={p.order_id[:14]}…"
                        )

        return result

    # ── Resolution mutation ────────────────────────────────────────────

    def mark_resolved(
        self, token_id: str, side: Side, realized_pnl: float,
        resolved_at: str | None = None,
    ) -> bool:
        """Mark a position resolved and free its slot in the caps.
        Returns True if a position was found and updated."""
        when = resolved_at or datetime.now(timezone.utc).isoformat()
        for p in self.positions:
            if p.token_id == token_id and p.side == side and p.status in (
                "submitted", "filled"
            ):
                p.status = "resolved"
                p.resolved_at = when
                p.realized_pnl = realized_pnl
                self._stamp(p)
                return True
        return False

    def mark_cancelled(
        self, token_id: str, side: Side,
        reason: str | None = None,
        when: str | None = None,
    ) -> bool:
        """Convert an in-flight submitted position to cancelled.
        Increments cancellation_count + records last_cancelled_at + reason.
        Returns True if a record was updated."""
        ts = when or datetime.now(timezone.utc).isoformat()
        # Find any submitted-but-not-yet-filled record for this (token, side)
        for p in reversed(self.positions):  # most recent first
            if p.token_id == token_id and p.side == side and p.status == "submitted":
                # Carry the running cancel count forward
                prior = self._last_cancel_for(token_id, side)
                p.cancellation_count = (
                    (prior.cancellation_count + 1) if prior else 1
                )
                p.status = "cancelled"
                p.resolved_at = ts
                p.last_cancelled_at = ts
                p.cancellation_reason = reason
                p.realized_pnl = 0.0
                self._stamp(p)
                return True
        return False

    def prune_resolved(self, keep_days: int = 7) -> int:
        """Drop resolved positions older than `keep_days`. Returns count
        dropped. Use to keep `positions.json` size bounded."""
        cutoff = (datetime.now(timezone.utc).timestamp() - keep_days * 86400)
        kept = []
        n_dropped = 0
        for p in self.positions:
            if p.status == "resolved" and p.resolved_at is not None:
                try:
                    ts = datetime.fromisoformat(p.resolved_at).timestamp()
                except ValueError:
                    ts = cutoff + 1  # keep on parse error
                if ts < cutoff:
                    n_dropped += 1
                    continue
            kept.append(p)
        self.positions = kept
        return n_dropped

    # ── Exposure queries ───────────────────────────────────────────────

    def open_positions(self) -> list[Position]:
        """Positions in any active state (submitted OR filled). Used for
        DEDUPE — we don't want to re-submit on a token where we have
        an unfilled resting limit OR a filled position."""
        return [p for p in self.positions if p.status in ("submitted", "filled")]

    def filled_positions(self) -> list[Position]:
        """Positions where capital is actually deployed.

        Polymarket empirical behavior (verified 2026-05-14 via manual
        test on Brazil election market): RESTING limit orders away from
        the spread do NOT lock capital. Only orders that fill (taker
        crosses our quote) consume cash. See
        `polymarket_live_trading_lessons.md` § "Capital lock semantics".

        TODO (before $1k promotion): add fill-status polling so we can
        distinguish 'submitted-but-unfilled' (no capital) from 'filled'
        (capital locked). Then `total_exposure_usd` can use this method
        instead of `open_positions` for far more aggressive submission
        without phantom-blocking the cap on resting limits that will
        never fill.

        Currently the bot transitions submitted → filled only via the
        order submission acknowledgment, which is wrong for makers —
        the limit might sit unfilled for hours. Once fill polling is
        wired, this becomes the source of truth for capital deployment.
        """
        return [p for p in self.positions if p.status == "filled"]

    def today_deployed_usd(self, now: datetime | None = None) -> float:
        """Sum of `position_usd` for positions FILLED today (UTC day).

        Distinct from `total_exposure_usd` (= currently-open filled
        positions) and from any single-position cap. This is the day's
        CUMULATIVE risk exposure including positions that already
        resolved.

        Used by the daily-deployment circuit breaker: bot refuses new
        submissions if `today_deployed + new_size > daily_deployment_limit`.
        Worst-case daily loss ≤ this value (= every fire loses).

        Counts:
          - status = 'filled' (still open, capital deployed)
          - status = 'resolved' (closed; counted because the day's risk
            was real even if it ended favorably)

        Does NOT count:
          - status = 'submitted' (resting limit, no capital deployed)
          - status = 'cancelled' (also no capital deployed)
          - positions submitted on a different UTC day
        """
        ref = now or datetime.now(timezone.utc)
        today_iso = ref.date().isoformat()
        total = 0.0
        for p in self.positions:
            if p.status not in ("filled", "resolved"):
                continue
            if not p.submitted_at:
                continue
            try:
                sub_date = p.submitted_at[:10]
                if sub_date == today_iso:
                    total += p.position_usd
            except (ValueError, IndexError):
                continue
        return total

    def total_exposure_usd(self) -> float:
        """Sum of position_usd for FILLED positions only.

        Polymarket only locks capital when an order fills (resting
        limits are weightless until a taker crosses). Verified
        empirically 2026-05-14 — see polymarket_live_trading_lessons.md
        § "Capital lock semantics".

        With fill-status polling in place (poll_fills() called at
        scan start), this accurately reflects deployed capital. Resting
        submitted orders don't count toward this cap — they only count
        toward DEDUPE (via is_open).

        Dedupe still uses `open_positions()` (both submitted + filled)
        so we don't re-submit on a token where we already have a
        resting limit OR a filled position."""
        return sum(p.position_usd for p in self.filled_positions())

    def exposure_in_region(self, region: str) -> float:
        """Filled-only — see total_exposure_usd docstring."""
        return sum(
            p.position_usd for p in self.filled_positions() if p.region == region
        )

    def exposure_in_event(self, market_id: int) -> float:
        """Filled-only — see total_exposure_usd docstring."""
        return sum(
            p.position_usd for p in self.filled_positions() if p.market_id == market_id
        )

    def n_open_in_region(self, region: str) -> int:
        """Includes submitted+filled (count, not exposure)."""
        return sum(1 for p in self.open_positions() if p.region == region)

    # ── Cap-check helpers ──────────────────────────────────────────────

    def would_exceed_cap(
        self,
        *,
        station_id: str,
        market_id: int,
        position_usd: float,
        portfolio_cap_usd: float,
        per_region_cap_usd: float,
        per_event_cap_usd: float,
    ) -> tuple[bool, str]:
        """Return (would_exceed, reason). All caps include the new
        position itself, so the check is `current + new > cap`."""
        new_total = self.total_exposure_usd() + position_usd
        if new_total > portfolio_cap_usd:
            return True, (
                f"portfolio cap ${portfolio_cap_usd:.0f} would be exceeded: "
                f"current ${self.total_exposure_usd():.2f} + new ${position_usd:.2f}"
            )
        region = region_for(station_id)
        new_region = self.exposure_in_region(region) + position_usd
        if new_region > per_region_cap_usd:
            return True, (
                f"region '{region}' cap ${per_region_cap_usd:.0f} would be exceeded: "
                f"current ${self.exposure_in_region(region):.2f} + new ${position_usd:.2f}"
            )
        new_event = self.exposure_in_event(market_id) + position_usd
        if new_event > per_event_cap_usd:
            return True, (
                f"event {market_id} cap ${per_event_cap_usd:.0f} would be exceeded: "
                f"current ${self.exposure_in_event(market_id):.2f} + new ${position_usd:.2f}"
            )
        return False, ""

    # ── Correlation-aware Kelly sizing ─────────────────────────────────

    def portfolio_kelly_multiplier(
        self,
        *,
        station_id: str,
        market_id: int,
        bankroll_usd: float,
    ) -> float:
        """Scale-down factor in (0, 1] reflecting correlation with currently-
        open positions. Apply to base Kelly size.

        Idea: if 80% of bankroll is already deployed in 'Europe West',
        a new Europe West position should be sized down vs an
        independent (e.g., Asia East) position.

        Formula:
            ρ-weighted exposure
              = Σ (position_usd × correlation(this, existing))
              over all open positions
            multiplier = max(0, 1 - ρ_weighted / bankroll)

        Caps to [0.05, 1.0] to avoid zero-sizing (use the cap functions
        for hard limits; this is the soft de-sizing layer).
        """
        if bankroll_usd <= 0:
            return 1.0
        target_region = region_for(station_id)
        rho_weighted = 0.0
        for p in self.open_positions():
            if p.market_id == market_id:
                rho = SAME_EVENT_CORRELATION
            elif p.region == target_region:
                rho = SAME_REGION_CORRELATION
            else:
                rho = CROSS_REGION_CORRELATION
            rho_weighted += p.position_usd * rho
        multiplier = 1.0 - (rho_weighted / bankroll_usd)
        return max(0.05, min(1.0, multiplier))

    # ── Maker rebates (added 2026-05-14) ────────────────────────────────

    def record_daily_rebate(self, date_iso: str, amount_usd: float) -> None:
        """Record (or overwrite) the maker rebate for a given UTC date.
        Idempotent: calling twice for the same date overwrites.

        `date_iso` should be 'YYYY-MM-DD' (UTC). Use the UTC date the
        rebate was PAID, not the trade date. Polymarket pays at midnight
        UTC per `polymarket_live_trading_lessons.md`.
        """
        if amount_usd < 0:
            return  # silently ignore negatives — rebates can't be negative
        self.daily_maker_rebates[date_iso] = float(amount_usd)

    def total_maker_rebates(self) -> float:
        """Sum of all recorded maker rebates across all dates."""
        return sum(self.daily_maker_rebates.values())

    def maker_rebate_for(self, date_iso: str) -> float:
        """Fetch the recorded rebate for a specific UTC date (0.0 if none)."""
        return float(self.daily_maker_rebates.get(date_iso, 0.0))

    # ── Adaptive bankroll (added 2026-05-14) ───────────────────────────

    def realized_pnl_total(self) -> float:
        """Total realized PnL = resolution PnL + maker rebates.

        Resolution PnL = sum of `realized_pnl` across resolved positions
        (bucket-resolved wins/losses, cancel-mode SELL proceeds, etc.).

        Maker rebates = Polymarket's liquidity-rewards program payouts
        for resting orders that filled. Tracked separately in
        `daily_maker_rebates` because Polymarket reports them as daily
        aggregates, not per-trade.

        Both contribute to the adaptive bankroll: a $50 bucket win and
        a $5 maker rebate both compound the next day's caps the same way.
        """
        resolution_pnl = sum(
            (p.realized_pnl or 0.0)
            for p in self.positions
            if p.status == "resolved" and p.realized_pnl is not None
        )
        return resolution_pnl + self.total_maker_rebates()

    def effective_bankroll(
        self, base_bankroll: float,
        ceiling: float = ADAPTIVE_BANKROLL_CEILING,
    ) -> float:
        """Current bankroll = starting bankroll + cumulative realized PnL.
        Capped at `ceiling` to keep risk posture conservative even
        after compounding wins.

        Example:
          Day 0: bankroll=$500, realized=$0  → effective=$500
          Day 1: bankroll=$500, realized=+$87 → effective=$587
          Day 7: bankroll=$500, realized=+$600 → effective=$1100
          Day 30: bankroll=$500, realized=+$2000 → effective=$2000 (capped)
        """
        return max(0.0, min(ceiling, base_bankroll + self.realized_pnl_total()))

    def scaled_caps(
        self,
        base_bankroll: float,
        *,
        ceiling: float = ADAPTIVE_BANKROLL_CEILING,
        portfolio_ratio: float = PORTFOLIO_CAP_RATIO,
        per_region_ratio: float = PER_REGION_CAP_RATIO,
        per_event_ratio: float = PER_EVENT_CAP_RATIO,
    ) -> dict[str, float]:
        """Return all four caps scaled to the current effective bankroll.

        Output keys:
          effective_bankroll, portfolio_cap, per_region_cap, per_event_cap

        Wire into TradingConfig at scan-start so each scan uses the
        most-recent realized-PnL-adjusted bankroll.
        """
        eff = self.effective_bankroll(base_bankroll, ceiling)
        return {
            "effective_bankroll": eff,
            "portfolio_cap": eff * portfolio_ratio,
            "per_region_cap": eff * per_region_ratio,
            "per_event_cap": eff * per_event_ratio,
        }

    # ── Reporting ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Diagnostic dict for logs / dashboard."""
        open_pos = self.open_positions()
        by_region: dict[str, float] = {}
        for p in open_pos:
            by_region[p.region] = by_region.get(p.region, 0.0) + p.position_usd
        return {
            "n_open": len(open_pos),
            "n_resolved_in_state": sum(
                1 for p in self.positions if p.status == "resolved"
            ),
            "total_exposure_usd": self.total_exposure_usd(),
            "exposure_by_region": by_region,
        }
