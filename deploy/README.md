# VPS deployment — slim daemon

End-to-end checklist for running `slim_daemon.py` on an Amsterdam VPS
under systemd. The daemon is paper-only by default (`PAPER_ONLY=True`),
so this guide is safe to follow without funding.

## Prerequisite: passed 24h local burn-in

Don't skip this. Run `python slim_daemon.py` on the laptop for 24h
first per [deploy/local_burn_in.md](local_burn_in.md). Then run
`python deploy/burn_in_summary.py` and confirm:

- Daemon was still running at the end
- No `Traceback` lines in stdout/stderr
- `data/publication_window_log.jsonl` has > 50 records
- WUG status is mostly `ok` (some `rate_limited` / `http_*` is fine)

## 1. Provision the VPS

| provider | recommended plan | region | rough cost |
| --- | --- | --- | --- |
| **Hetzner** (recommended) | CX22 (2 vCPU, 4 GB) | Falkenstein FSN1 or Helsinki HEL1 — both peer well with Polymarket AMS | €4.50/mo |
| DigitalOcean | s-1vcpu-1gb-amd | AMS3 | $6/mo |
| Vultr | $5 cloud compute | Amsterdam | $5/mo |
| AWS Lightsail | 1 GB RAM | eu-central-1 (Frankfurt) | $5/mo |

OS: **Ubuntu 24.04 LTS** (or Debian 12). SSH key auth, no root password.

> The prior decommissioned VPS was Amsterdam at 1.27ms to Polymarket
> CLOB. Hetzner FSN1 / HEL1 measure similar; either is fine.

## 2. First-time setup on the VPS

SSH in as a non-root sudoer (or root — either works):

```
ssh kevin@<vps-ip>
git clone <your-fork-of-the-repo> ~/Weather_Bot
cd ~/Weather_Bot
bash deploy/setup_vps.sh
```

`setup_vps.sh` is idempotent — re-run any time. It installs apt deps,
creates `.venv`, pip-installs `requirements.txt`, drops the systemd
unit at `/etc/systemd/system/slim-daemon.service`, runs a smoke-test
Polymarket fee fetch, and enables the service (without starting it).

## 3. Start the daemon

```
sudo systemctl start slim-daemon
sudo systemctl status slim-daemon          # confirm active (running)
sudo journalctl -u slim-daemon -f          # live logs
```

You should see the startup banner within seconds:

```
[daemon] start  paper_only=True  dry_run=True  wug_interval=60.0s  events_refresh=300.0s
[fees] taker_rate=0.0500  rebate=0.25  source=...
[refresh] <iso-ts>  events=N  active_sk=M  wug_pollers=M  ws_tokens=K
```

After that the daemon is mostly quiet — most WUG ticks won't see the
extreme move. WUGUpdate events fire whenever the daily extreme moves
outward; each fires the strategy evaluators (Layer 7 progressive,
high-bucket NO at trigger-local-hour, lock-in YES on tail crossings).

## 4. Operating

| action | command |
| --- | --- |
| live logs | `sudo journalctl -u slim-daemon -f` |
| last hour | `sudo journalctl -u slim-daemon --since '1 hour ago'` |
| daily summary | `~/Weather_Bot/.venv/bin/python deploy/burn_in_summary.py` |
| publication analyzer | `~/Weather_Bot/.venv/bin/python analyze_publication_window.py` |
| restart (config change) | `sudo systemctl restart slim-daemon` |
| graceful halt | `sudo systemctl stop slim-daemon` |
| emergency halt | `touch ~/Weather_Bot/KILL_SWITCH && sudo systemctl stop slim-daemon` |
| auto-restart on crash | yes (systemd `Restart=on-failure`, RestartSec=10) |

## 5. After 7 days of data

Run the analyzer:

```
~/Weather_Bot/.venv/bin/python analyze_publication_window.py
```

The 4-section report tells you:

1. **Tradable-book presence by offset bin** — does the market stay
   tradable past end-of-resolution-day-local?
2. **WUG↔Polymarket and METAR↔Polymarket agreement** — does our truth
   source match the oracle? (WUG should be ~100%.)
3. **Winning bucket's YES ask distribution per offset** — what entry
   price could a publication-race strategy achieve?
4. **Hypothetical PnL** — buy YES on winner at first observed ask per
   bin, after fees.

Decision: if section 1 shows >30% active book past +30 min AND section 3
median ask <$0.97, the race strategy is buildable. Else, the slim
strategies (Layer 7 progressive + V2 + high-bucket NO) are the complete
surface. Either way, the data answers the question.

## 6. Going live (later)

When (and if) you choose to flip from paper to live:

1. Edit `slim_daemon.py` and `slim_scan.py`, change `PAPER_ONLY = True`
   to `PAPER_ONLY = False`. Commit.
2. Pull on the VPS.
3. Wire a real `ExecutionClient` in `build_client()` — currently the
   `--live` branch still returns `ExecutionClient.dry_run(cfg)` as a
   placeholder (see the `# placeholder — wire a real CLOB client` comment).
4. Set `LIVE_OK=1` and add `--live` to the systemd unit's `ExecStart`.
5. Fund Polymarket with the smoke-test amount ($50-100).
6. Restart the service.
7. Watch the first 24h closely.

For the SDK v2 details — funder, signature type, geo restrictions — see
the memory note `polymarket_sdk_v2_migration.md`.
