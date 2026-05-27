"""Wunderground polling background task with change detection.

Wunderground doesn't push, so we poll. For each active (station, target_date)
we fetch the daily extreme from api.weather.com once per poll interval
and emit a `WUGUpdate` event ONLY when the extreme moves outward from
the previous reading. The daemon subscribes to these events and re-runs
the strategy evaluators on each one.

Design
======
- One background asyncio task PER (station, target_date) tuple that's
  currently relevant — i.e. target_date in {today_local, tomorrow_local}
  for each station. Old (station, date) tuples are pruned when the day
  rolls over locally.
- Polite throttle is enforced by weather_bot.wunderground (1 req/sec
  globally), so even 50 concurrent station-pollers self-serialize.
- Cache invalidation: each tick calls `wunderground.clear_cache()` for
  the specific (icao, date) before the fetch, so we re-hit the API
  instead of getting the cached value from earlier in the process.
- Failure isolation: per-poller exceptions are logged and the poller
  retries on the next interval. One bad station never kills the daemon.

Event payload (`WUGUpdate`)
===========================
  station_id, target, target_date_iso, observed_extreme_c, observed_int,
  previous_int (None on first reading), n_observations, fetched_at_utc.

Consumer
========
The daemon wires up a callback per WUGUpdate that runs:
  1. METAR early-tail equivalent — check lock-in math (high_tail crossed)
  2. Layer 7 — progressive single-bucket eval
  3. High-bucket NO — fires only after trigger-local-hour
WUG-update-driven evaluation replaces the previous cron-15 sweep.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Awaitable, Callable

import httpx

from .wunderground import fetch_wunderground_daily, clear_cache as _wug_clear_cache


@dataclass
class WUGUpdate:
    station_id: str
    target: str                # "max" | "min"
    target_date_iso: str
    observed_extreme_c: float
    observed_int: int          # _rounded_observation in market unit
    previous_int: int | None
    n_observations: int
    fetched_at_utc: str
    wug_status: str            # "ok" | "no_data" | "http_error" | etc.


# Type alias for the callback the daemon registers.
UpdateCallback = Callable[[WUGUpdate], Awaitable[None]]


class WUGPoller:
    """Per-(station, target_date) background poller.

    Usage from the daemon:
      poller = WUGPoller(station=..., target=..., target_date=...,
                        callback=on_wug_update, http=shared_http_client)
      task = asyncio.create_task(poller.run())

    Stop via `poller.stop_event.set()`.
    """

    def __init__(
        self,
        *,
        station,
        target: str,
        target_date: date,
        callback: UpdateCallback,
        http: httpx.AsyncClient,
        interval_s: float = 60.0,
    ):
        self.station = station
        self.target = target
        self.target_date = target_date
        self.target_date_iso = target_date.isoformat()
        self.callback = callback
        self.http = http
        self.interval_s = interval_s
        self.stop_event = asyncio.Event()
        # Track the last integer extreme we observed so we only fire
        # callbacks on outward movement.
        self._last_int: int | None = None
        self._last_status: str | None = None
        self._tick_count = 0

    async def run(self) -> None:
        """Main loop. Polls every interval_s until stop_event is set."""
        from .pnl import _rounded_observation
        sid = self.station.station_id
        target = self.target
        td = self.target_date

        while not self.stop_event.is_set():
            self._tick_count += 1
            try:
                # Re-fetch (clear cache so we always hit the API again)
                _wug_clear_cache()
                wug = await fetch_wunderground_daily(
                    sid, td, client=self.http,
                )
                ext_c = (
                    wug.daily_max_c if target == "max" else wug.daily_min_c
                )
                if ext_c is None:
                    self._last_status = wug.raw_status
                else:
                    obs_int = int(_rounded_observation(ext_c, self.station.unit))
                    moved_outward = self._has_moved_outward(obs_int)
                    if moved_outward or self._last_int is None:
                        update = WUGUpdate(
                            station_id=sid, target=target,
                            target_date_iso=self.target_date_iso,
                            observed_extreme_c=float(ext_c),
                            observed_int=obs_int,
                            previous_int=self._last_int,
                            n_observations=wug.n_observations,
                            fetched_at_utc=wug.fetched_at_utc,
                            wug_status=wug.raw_status,
                        )
                        prev = self._last_int
                        self._last_int = obs_int
                        try:
                            await self.callback(update)
                        except Exception as exc:
                            print(
                                f"[wug-poller {sid}/{target}] callback failed: "
                                f"{type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                            # Roll back so next tick re-fires the same event
                            self._last_int = prev
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[wug-poller {sid}/{target}] tick {self._tick_count} "
                    f"failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            # Sleep interruptibly so stop_event.set() wakes us promptly.
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.interval_s,
                )
            except asyncio.TimeoutError:
                pass

    def _has_moved_outward(self, new_int: int) -> bool:
        """Did the extreme move in the direction we care about?

        For max-target: outward means new_int > last_int.
        For min-target: outward means new_int < last_int.
        """
        if self._last_int is None:
            return True
        if self.target == "max":
            return new_int > self._last_int
        return new_int < self._last_int


class WUGPollerPool:
    """Owns a set of WUGPoller tasks, one per (station, target_date) we
    care about. Add/remove tuples as events become active / expire.

    Designed for a long-running daemon: register at startup, refresh on
    each gamma fetch, prune entries past their resolution date.
    """

    def __init__(
        self,
        callback: UpdateCallback,
        http: httpx.AsyncClient,
        interval_s: float = 60.0,
    ):
        self.callback = callback
        self.http = http
        self.interval_s = interval_s
        self._pollers: dict[tuple[str, str, str], WUGPoller] = {}
        self._tasks: dict[tuple[str, str, str], asyncio.Task] = {}

    def keys(self) -> set[tuple[str, str, str]]:
        return set(self._pollers.keys())

    def ensure_running(self, *, station, target: str, target_date: date) -> None:
        """Start a poller for this tuple if not already running."""
        key = (station.station_id, target, target_date.isoformat())
        if key in self._pollers:
            return
        poller = WUGPoller(
            station=station, target=target, target_date=target_date,
            callback=self.callback, http=self.http,
            interval_s=self.interval_s,
        )
        self._pollers[key] = poller
        self._tasks[key] = asyncio.create_task(
            poller.run(), name=f"wug-poller-{key[0]}-{key[1]}-{key[2]}",
        )

    async def stop(self, key: tuple[str, str, str]) -> None:
        """Stop a single poller and await its exit."""
        poller = self._pollers.pop(key, None)
        task = self._tasks.pop(key, None)
        if poller is not None:
            poller.stop_event.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    async def stop_all(self) -> None:
        for key in list(self._pollers.keys()):
            await self.stop(key)

    async def prune(self, keep: set[tuple[str, str, str]]) -> None:
        """Stop any poller whose key isn't in `keep`."""
        to_stop = [k for k in self._pollers.keys() if k not in keep]
        for k in to_stop:
            await self.stop(k)
