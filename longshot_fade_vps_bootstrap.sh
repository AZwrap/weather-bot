#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Weatherbot money-free FORWARD TEST — VPS bootstrap (paper-only).
#
# What it does: clones the public repo, makes a venv, and installs two cron jobs
# that run the longshot-fade DRY-RUN scanner (NO @ 0.75-0.85 on the basket
# cities) + a daily resolve. It places NO orders, needs NO wallet key, and puts
# NO secrets on the box — it only reads public Polymarket/Open-Meteo APIs and
# writes signal logs to disk.
#
# Run as root on a fresh Ubuntu VPS:
#   curl -fsSL https://raw.githubusercontent.com/AZwrap/weather-bot/claude/funny-neumann-86b263/longshot_fade_vps_bootstrap.sh | bash
# (or review it first, then: bash longshot_fade_vps_bootstrap.sh)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="https://github.com/AZwrap/weather-bot.git"
BRANCH="claude/funny-neumann-86b263"
DIR="/root/weatherbot"
PY="$DIR/.venv/bin/python"

echo "[1/6] apt deps (git, python venv)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv python3-pip >/dev/null

echo "[2/6] clone/update repo on $BRANCH ..."
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q origin "$BRANCH"
  git -C "$DIR" checkout -q "$BRANCH"
  git -C "$DIR" reset -q --hard "origin/$BRANCH"
else
  git clone -q -b "$BRANCH" "$REPO" "$DIR"
fi

echo "[3/6] python venv + deps (httpx numpy pandas scipy)..."
[ -d "$DIR/.venv" ] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q httpx numpy pandas scipy
mkdir -p "$DIR/data/longshot_fade"

echo "[4/6] dry-run smoke test (no orders) ..."
cd "$DIR"
PYTHONUTF8=1 "$PY" longshot_fade_harness.py | tail -3

echo "[5/7] install cron: scan hourly, resolve daily 23:30 UTC ..."
SCAN="cd $DIR && PYTHONUTF8=1 $PY longshot_fade_harness.py >> $DIR/data/longshot_fade/scan.log 2>&1"
RES="cd $DIR && PYTHONUTF8=1 $PY longshot_fade_harness.py --resolve >> $DIR/data/longshot_fade/resolve_report.txt 2>&1"
{ crontab -l 2>/dev/null | grep -v 'longshot_fade_harness.py' || true
  echo "0 * * * * $SCAN"
  echo "30 23 * * * $RES"
} | crontab -

echo "[6/7] dashboard systemd service on 127.0.0.1:8765 ..."
cat > /etc/systemd/system/weatherbot-dash.service <<UNIT
[Unit]
Description=Weatherbot forward-test dashboard
After=network.target
[Service]
WorkingDirectory=$DIR
ExecStart=$PY $DIR/dashboard_forward.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q weatherbot-dash.service
systemctl restart weatherbot-dash.service
sleep 1
if systemctl is-active --quiet weatherbot-dash.service; then
  echo "  dashboard ACTIVE -> http://127.0.0.1:8765"
else
  echo "  dashboard NOT active -- check: journalctl -u weatherbot-dash -n 20"
fi

echo "[7/7] DONE — forward test + dashboard live on this box (paper-only)."
echo "  signals   : $DIR/data/longshot_fade/signals.jsonl"
echo "  watch scan: tail -f $DIR/data/longshot_fade/scan.log"
echo "  DASHBOARD : tunnel from your laptop, then open http://localhost:8765 :"
echo "              ssh -L 8765:127.0.0.1:8765 root@199.247.29.13"
echo "  stop all  : crontab -l|grep -v longshot_fade_harness.py|crontab -; systemctl disable --now weatherbot-dash"
echo "--- installed cron ---"
crontab -l | grep longshot_fade_harness.py
