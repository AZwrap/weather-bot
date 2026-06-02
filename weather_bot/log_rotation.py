"""Log rotation for append-only JSONL files.

The bot writes to several append-only JSONL files that grow unbounded:
  - data/intraday_log.jsonl            (paper-trade decisions per scan)
  - data/obs_distance_paper_log.jsonl  (Layer 5 filter decisions)
  - data/market_drift_eval_log.jsonl   (Layer 8 evaluations)
  - data/market_drift_cancel_log.jsonl (Layer 8 cancellations)
  - data/guaranteed_no_buy_log.jsonl   (Layer 7 fills + skips)
  - data/cross_up_log.jsonl            (cross-up SELL attempts)
  - data/no_momentum_placement_log.jsonl  (NO_momentum placement context)
  - data/forward_log.jsonl             (forecast snapshots, written by log_forecasts)

Each grows at 0.5-10 MB/day. Over months, read performance degrades
(analyzers re-read whole file) and disk usage accumulates. This
module provides a simple size-based rotation: when a log exceeds
`max_size_mb`, rename it to a timestamped archive and start fresh.

No process needs to coordinate — the rotation is atomic via os.rename,
and the bot just opens a fresh file on next write.

Usage:
    from weather_bot.log_rotation import rotate_log_if_large

    # Call periodically (e.g., once per cron tick at start)
    rotate_log_if_large(Path("data/intraday_log.jsonl"), max_size_mb=50.0)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MAX_SIZE_MB = 100.0
"""Rotate when file exceeds this. 100MB is a comfortable threshold —
analyzers can still read it in ~1s, but starts to be noticeable. Each
rotation archives the file with a timestamp suffix so all data is
preserved (just split into chunks)."""


def rotate_log_if_large(
    path: Path,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    quiet: bool = False,
) -> bool:
    """Rotate `path` if it exceeds max_size_mb.

    Returns True if a rotation occurred, False otherwise.

    Rotation mechanic: rename `path` to `path.YYYY-MM-DD_HHMMSS.jsonl`
    (preserving extension). The original path is left missing so the
    next write creates a fresh file.

    Safe to call frequently — no-op when file is small or absent.
    """
    if not path.exists():
        return False
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return False
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < max_size_mb:
        return False

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    # Insert timestamp before the extension chain (handle .jsonl, .json, etc.)
    suffix = "".join(path.suffixes) or path.suffix or ""
    stem = path.name[: -len(suffix)] if suffix else path.name
    rotated = path.parent / f"{stem}.{ts}{suffix}"

    try:
        os.rename(str(path), str(rotated))
    except OSError as exc:
        if not quiet:
            print(f"!! rotate_log_if_large({path}) failed: {exc}")
        return False

    if not quiet:
        print(f"[log-rotation] {path.name} ({size_mb:.1f} MB) → {rotated.name}")
    return True


def rotate_all_logs(
    log_paths: list[Path],
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    quiet: bool = False,
) -> int:
    """Rotate every log in `log_paths` that exceeds `max_size_mb`.

    Returns count of files rotated. Use at the start of each cron tick
    to keep log files bounded without needing a separate cron job.
    """
    n = 0
    for p in log_paths:
        if rotate_log_if_large(p, max_size_mb=max_size_mb, quiet=quiet):
            n += 1
    return n


# Default set of log files the bot maintains. Caller can override.
DEFAULT_LOG_PATHS = [
    Path("data/intraday_log.jsonl"),
    Path("data/obs_distance_paper_log.jsonl"),
    Path("data/market_drift_eval_log.jsonl"),
    Path("data/market_drift_cancel_log.jsonl"),
    Path("data/guaranteed_no_buy_log.jsonl"),
    Path("data/cross_up_log.jsonl"),
    Path("data/no_momentum_placement_log.jsonl"),
    Path("data/forward_log.jsonl"),
    Path("data/time_of_day_filter_log.jsonl"),
    Path("data/daemon_cycle_metrics.jsonl"),
    Path("data/filled_position_no_ask_trajectory.jsonl"),
    Path("data/basket_favorite_ticks.jsonl"),
]
