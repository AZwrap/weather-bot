#!/usr/bin/env bash
# Wrapper invoked by cron. cd's to the project root, activates the venv, runs
# the named script with a timestamped header so log files stay readable.
#
# Usage:  deploy/run.sh <script-name-without-.py>
# Example: deploy/run.sh log_forecasts
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${1:?usage: $0 <script-name-without-.py>}"

cd "$PROJECT_DIR"

if [[ ! -f "$SCRIPT.py" ]]; then
  echo "!! $SCRIPT.py not found in $PROJECT_DIR"
  exit 1
fi

echo
echo "================================================================"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')  $SCRIPT.py"
echo "================================================================"
exec .venv/bin/python "$SCRIPT.py" "${@:2}"
