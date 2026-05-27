#!/usr/bin/env bash
# Idempotent VPS bootstrap for the slim daemon.
#
# Assumptions: fresh Ubuntu 24.04 (or Debian 12) on Hetzner / DigitalOcean
# in Amsterdam. SSH'd in as a non-root sudoer (or root — both work).
#
# Usage on the VPS:
#   git clone <repo> ~/Weather_Bot
#   cd ~/Weather_Bot
#   bash deploy/setup_vps.sh
#
# Re-run anytime to repair / upgrade. The systemd unit is reinstalled
# every time so edits to deploy/slim-daemon.service propagate.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Use the invoking user, even when run via sudo.
RUN_USER="${SUDO_USER:-$USER}"
USER_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

echo "==> Project dir: $PROJECT_DIR"
echo "==> Run user:    $RUN_USER"

# ── 1. System packages ────────────────────────────────────────────────
echo "==> Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    git tzdata curl ca-certificates build-essential

# ── 2. UTC timezone ───────────────────────────────────────────────────
echo "==> Setting timezone to UTC…"
sudo timedatectl set-timezone UTC

# ── 3. Python ≥ 3.11 ──────────────────────────────────────────────────
PY_VERSION="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "==> System Python: $PY_VERSION"
if [[ "$(printf '%s\n3.11' "$PY_VERSION" | sort -V | head -1)" != "3.11" ]]; then
  echo "!! Python ≥ 3.11 required (you have $PY_VERSION)."
  echo "   On Ubuntu 22.04, install python3.12 from deadsnakes PPA."
  exit 1
fi

# ── 4. venv + deps ────────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  echo "==> Creating venv…"
  python3 -m venv .venv
fi
echo "==> Upgrading pip + installing requirements…"
.venv/bin/pip install --upgrade -q pip wheel
.venv/bin/pip install -q -r requirements.txt
# Daemon needs the WS client. Some installs don't pull this in via the
# main requirements; add it explicitly.
.venv/bin/pip install -q "websockets>=12"

# ── 5. Data dir + KILL_SWITCH staging ─────────────────────────────────
mkdir -p data
# Ensure no stale KILL_SWITCH from a prior run blocks startup
if [[ -f KILL_SWITCH ]]; then
  echo "==> Removing pre-existing KILL_SWITCH file."
  rm KILL_SWITCH
fi

# ── 6. Fee config + initial latency probe ─────────────────────────────
echo "==> Smoke test: live Polymarket fee fetch (caches to data/)…"
.venv/bin/python -c "
import asyncio
from weather_bot.fees import fetch_live_fee_config, warn_if_fee_config_changed
async def main():
    cfg = await fetch_live_fee_config()
    if cfg is not None:
        warn_if_fee_config_changed(cfg)
        print(f'  taker_rate={cfg.taker_fee_rate:.4f}  rebate={cfg.maker_rebate_rate}  source={cfg.source}')
asyncio.run(main())
" || echo "!! fee fetch failed (non-fatal; daemon will retry)"

# ── 7. Install systemd unit ───────────────────────────────────────────
SVC_SRC="$PROJECT_DIR/deploy/slim-daemon.service"
SVC_DST="/etc/systemd/system/slim-daemon.service"

if [[ ! -f "$SVC_SRC" ]]; then
  echo "!! $SVC_SRC missing"; exit 1
fi

echo "==> Installing $SVC_DST…"
sudo cp "$SVC_SRC" "$SVC_DST"
sudo sed -i \
  -e "s|/USER|$RUN_USER|g" \
  -e "s|/PROJECT_PATH/Weather_Bot|$PROJECT_DIR|g" \
  "$SVC_DST"

sudo systemctl daemon-reload
sudo systemctl enable slim-daemon.service
echo "==> Enabled. Start with:  sudo systemctl start slim-daemon"

# ── 8. Verification cheatsheet ────────────────────────────────────────
cat <<EOF

==> Setup complete.

Start:           sudo systemctl start slim-daemon
Status:          sudo systemctl status slim-daemon
Live logs:       sudo journalctl -u slim-daemon -f
Last hour:       sudo journalctl -u slim-daemon --since '1 hour ago'

Halt (drain):    sudo systemctl stop slim-daemon
Hard halt:       touch $PROJECT_DIR/KILL_SWITCH        # daemon exits within ~5s
                 sudo systemctl stop slim-daemon

Daily summary:   .venv/bin/python deploy/burn_in_summary.py
PnL / analyzer:  .venv/bin/python analyze_publication_window.py

The PAPER_ONLY=True constant in slim_daemon.py is the third gate after
--live and LIVE_OK=1. To enable real orders later, edit slim_daemon.py +
slim_scan.py and set PAPER_ONLY=False, then restart the service.
EOF
