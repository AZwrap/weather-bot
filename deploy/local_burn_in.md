# Local 24h burn-in

Before promoting `slim_daemon.py` to a VPS, run it on this machine for
24h to catch the obvious failure modes: crashes, exception spew, file-
descriptor leaks, WS reconnect loops. The bot is paper-only, so this
costs nothing but laptop battery + bandwidth.

## Prerequisites

```
python --version          # need ≥ 3.11
pip install -r requirements.txt
```

Required dependencies (already in `requirements.txt`):
`httpx[http2]`, `numpy`, `pandas`, `scipy`, `python-dateutil`,
`openpyxl`. The daemon also imports `websockets` (pulled in by the
upstream py-clob-client deps); if you hit `ModuleNotFoundError:
websockets`, do `pip install websockets`.

Windows-only: `pip install tzdata` (Python's zoneinfo fallback).

## Start the burn-in

In one terminal:

```
python slim_daemon.py --wug-poll-interval 60 --events-refresh-interval 300
```

Watch the first few seconds of output. You should see:

- `[daemon] start  paper_only=True  dry_run=True  wug_interval=60.0s  events_refresh=300.0s`
- `[refresh] <iso-ts>  events=N  active_sk=M  wug_pollers=M  ws_tokens=K`
- `[fees] taker_rate=0.0500  rebate=0.25  source=...`

If those land, you're good. The daemon will be quiet for a while after
that — most WUG ticks won't see the daily extreme move, so most ticks
just refresh state silently.

## What to watch

In a second terminal:

```
# tail the strategy logs as they accumulate
tail -f data/intraday_log.jsonl data/high_bucket_no_log.jsonl data/v2_conditional_log.jsonl data/publication_window_log.jsonl 2>/dev/null

# count records over time
watch -n 60 'wc -l data/*.jsonl'
```

After 24h, run:

```
python deploy/burn_in_summary.py
```

It reports: total records per log, WUG poller success/failure counts,
Layer 7 evaluation outcomes, any exceptions caught.

## Success criteria

- Daemon process still running after 24h (`ps -p $(pgrep -f slim_daemon)`).
- No `Traceback` lines in the daemon stdout/stderr — except expected
  network blips ("[wug-poller] tick N failed: HTTPError" once or twice
  is fine; sustained means something is broken).
- `data/publication_window_log.jsonl` has > 50 records (one per
  active station-date snapshot tick).
- WS task is alive: `state.ws_task is not None and not state.ws_task.done()`
  — visible by inspecting the daemon's output for repeated
  `[refresh]` lines without `[ws] on_book_message failed`.

## Halt

`Ctrl-C` once for a clean shutdown (signal handler triggers
`shutdown_event`, daemon awaits pollers + WS task with a 5s timeout).
Or `touch KILL_SWITCH` and wait up to 5s for the next polling tick.

## If burn-in passes

Provision the Amsterdam VPS and run `bash deploy/setup_vps.sh` there.
The systemd unit (`deploy/slim-daemon.service`) handles auto-restart.
