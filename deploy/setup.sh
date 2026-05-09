#!/usr/bin/env bash
# One-time bootstrap on a fresh Ubuntu/Debian VPS.
# Idempotent — safe to re-run.
#
# Usage (after uploading the project to the VPS):
#   cd ~/Weather_Bot
#   bash deploy/setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> Project: $PROJECT_DIR"

# ── System packages ────────────────────────────────────────────────────────
echo "==> Installing system packages (sudo required)…"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git tzdata curl

# ── Timezone: UTC for predictable cron behaviour ───────────────────────────
echo "==> Setting timezone to UTC…"
sudo timedatectl set-timezone UTC

# ── Python venv ────────────────────────────────────────────────────────────
PY_VERSION="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "==> System Python: $PY_VERSION"
if [[ "$(printf '%s\n3.11' "$PY_VERSION" | sort -V | head -1)" != "3.11" ]]; then
  echo "!! Python ≥ 3.11 recommended (you have $PY_VERSION). Continuing anyway."
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating venv…"
  python3 -m venv .venv
fi

echo "==> Installing dependencies…"
.venv/bin/pip install --upgrade -q pip
.venv/bin/pip install -q -r requirements.txt

# ── Data directory ─────────────────────────────────────────────────────────
mkdir -p data

# ── Initial bias-table training ────────────────────────────────────────────
if [[ ! -f bias_table.json ]]; then
  echo "==> Training initial bias table (~5 min, fetches 365 days × 59 markets)…"
  .venv/bin/python train_bias.py
else
  echo "==> bias_table.json already exists — skipping training. Re-train with:"
  echo "    .venv/bin/python train_bias.py"
fi

# ── Verify everything works end-to-end ─────────────────────────────────────
echo "==> Smoke test: log one set of forecasts…"
.venv/bin/python log_forecasts.py

echo
echo "==> Setup complete."
echo
echo "Next steps:"
echo
echo "  1. Install the cron schedule"
echo "     sed -i \"s|/PROJECT_PATH/Weather_Bot|$PROJECT_DIR|g\" deploy/crontab.txt"
echo "     crontab deploy/crontab.txt"
echo "     crontab -l                # verify"
echo
echo "  2. (Optional) Install the dashboard as a systemd service"
echo "     sudo cp deploy/weather-bot-dashboard.service /etc/systemd/system/"
echo "     sudo sed -i \"s|/PROJECT_PATH/Weather_Bot|$PROJECT_DIR|g; s|/USER|\$USER|g\" \\"
echo "        /etc/systemd/system/weather-bot-dashboard.service"
echo "     sudo systemctl daemon-reload"
echo "     sudo systemctl enable --now weather-bot-dashboard"
echo
echo "     Access from your laptop via SSH tunnel:"
echo "       ssh -L 8501:localhost:8501 user@vps"
echo "       open http://localhost:8501"
echo
echo "Logs:                      data/log.out"
echo "Forward-log records:       data/forward_log.jsonl"
echo "Dashboard logs:            sudo journalctl -u weather-bot-dashboard -f"
