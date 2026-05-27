"""Atomic JSON file writes — crash-safe state persistence.

The bot has several state files that survive across process restarts:
  - data/portfolio.json (positions, cap accounting, rebates)
  - data/dead_tokens.json (Polymarket-404 token cache)
  - data/ensemble_cache.json (Open-Meteo ensemble cache)
  - bias_table.json (per-station bias correction)
  - data/cap_budget.json (already uses atomic via fcntl, kept separate)
  - data/rate_limit_state.json (already uses tempfile+os.replace, kept separate)

Previously these were written via `path.write_text(json.dumps(...))`,
which is NOT atomic. If the process gets SIGKILL'd (OOM, systemd
timeout, host reboot) mid-write, the file is truncated/corrupted.
Next load() returns `{}` or raises JSONDecodeError → the bot loses
ALL of its state.

For portfolio.json specifically, the impact is catastrophic:
  - All open positions become invisible
  - Dedup check passes → bot re-places existing orders
  - Cap accounting resets → daily limit silently bypassed
  - Cross-up cancel has nothing to cancel
  - Bot effectively starts fresh with full bankroll committed

Atomic write via tempfile + os.replace is the standard POSIX
solution: writes go to a tmpfile, then a single atomic rename
swaps it into place. Concurrent readers see either the old file
or the new — never a partial write.

Use:
    from weather_bot.atomic_write import atomic_write_json

    def save(self, path):
        atomic_write_json(path, self.to_dict())
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write `payload` as JSON to `path` atomically.

    Mechanics:
      1. Create a unique temp file in the same directory as `path`
         (so the rename can be atomic — cross-filesystem renames are
         not atomic on most systems).
      2. Write JSON to the temp file.
      3. fsync to force OS buffer flush to disk before rename.
         Without this, a crash AFTER the rename but BEFORE the disk
         flush could still leave the on-disk content stale/empty.
      4. os.replace atomically swaps temp → target.
      5. On any exception, the temp file is removed; original is
         untouched.

    Raises:
      OSError, TypeError, ValueError: propagated to caller. Callers
      that don't want disruption (e.g., observability logs) should
      wrap in try/except.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_str = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            try:
                # Force OS buffer to disk so a crash AFTER the rename
                # still leaves the new content on disk. fsync can fail
                # on some filesystems (e.g., NFS); we tolerate failure
                # here because os.replace is still atomic enough for
                # our purposes — the rename itself is the critical
                # atomicity guarantee.
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_str, path)
    except BaseException:
        # Cleanup temp file on ANY failure (Exception + CancelledError).
        # Original file (if any) is unmodified.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
