"""Persistent cache of Polymarket token IDs whose /book endpoint returns 404.

Audit 2026-05-16 found ~170 of ~594 /book queries per cron tick return
404. These are tokens whose underlying market has resolved (UMA cleared)
or was never listed. Each tick the bot was re-querying the same dead
tokens, wasting ~170 API calls and cascading into 429s on the live
/book endpoint.

This cache persists "known dead" token IDs to disk. The bot skips
/book queries for cached tokens until a TTL expires (default 24h —
markets very rarely re-list, but the TTL lets us recover from
transient 404s).

Schema (`data/dead_tokens.json`):
  {
    "tokens": {
      "<token_id>": {
        "marked_at_utc": "2026-05-16T01:50:00+00:00",
        "ttl_hours": 24
      },
      ...
    },
    "version": 1
  }
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_CACHE_PATH = Path("data/dead_tokens.json")
DEFAULT_TTL_HOURS = 24
"""How long to consider a token dead before re-testing /book. Polymarket
markets do not unresolve (so 24h is conservative); set to a smaller
value during development to validate the cache invalidates correctly."""


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("tokens", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, dict], path: Path = DEFAULT_CACHE_PATH) -> None:
    """Atomic save via weather_bot.atomic_write — see that module for
    why non-atomic writes can corrupt this cache on process crash
    (truncated file → reload returns {} → ~170 dead tokens re-fire
    on /book each cron tick → cascade of 429s)."""
    from weather_bot.atomic_write import atomic_write_json
    payload = {"tokens": cache, "version": 1}
    atomic_write_json(path, payload)


def filter_live_tokens(
    cache: dict[str, dict],
    token_ids: list[str],
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> tuple[list[str], int]:
    """Split token_ids into (still-live, n_filtered).

    Returns:
      - live: tokens not in cache, or whose cache entry has expired
      - n_filtered: count of tokens skipped due to cached dead-status
    """
    now = datetime.now(timezone.utc)
    live: list[str] = []
    skipped = 0
    for tid in token_ids:
        entry = cache.get(tid)
        if entry is None:
            live.append(tid)
            continue
        try:
            marked_at = datetime.fromisoformat(entry["marked_at_utc"])
        except (KeyError, ValueError):
            # Corrupt entry — re-test
            live.append(tid)
            continue
        entry_ttl = float(entry.get("ttl_hours", ttl_hours))
        if now - marked_at > timedelta(hours=entry_ttl):
            # Expired — re-test, will refresh if still dead
            live.append(tid)
        else:
            skipped += 1
    return live, skipped


def mark_dead(
    cache: dict[str, dict],
    token_id: str,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> None:
    """Add a token to the dead cache with current UTC timestamp."""
    cache[token_id] = {
        "marked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ttl_hours": ttl_hours,
    }


def prune_expired(
    cache: dict[str, dict],
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> int:
    """Drop expired entries. Returns count removed."""
    now = datetime.now(timezone.utc)
    to_remove: list[str] = []
    for tid, entry in cache.items():
        try:
            marked_at = datetime.fromisoformat(entry["marked_at_utc"])
        except (KeyError, ValueError):
            to_remove.append(tid)
            continue
        entry_ttl = float(entry.get("ttl_hours", ttl_hours))
        if now - marked_at > timedelta(hours=entry_ttl):
            to_remove.append(tid)
    for tid in to_remove:
        cache.pop(tid, None)
    return len(to_remove)
