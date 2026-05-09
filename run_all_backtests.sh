#!/bin/bash
# Run all diagnostic / backtest scripts in sequence.
# See memory/project_backtest_scripts.md for what each one tests.
#
# Usage:
#   ./run_all_backtests.sh                 (locally, requires .venv)
#   ssh root@VPS 'cd /root/Weather_Bot && bash run_all_backtests.sh'

set -e
cd "$(dirname "$0")"

VENV_PYTHON=".venv/bin/python"
[ -f "$VENV_PYTHON" ] || VENV_PYTHON=".venv/Scripts/python.exe"  # windows venv

run() {
    echo
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
    "$VENV_PYTHON" "$2"
}

run "1. PnL health check"           check_pnl.py
run "2. Stop-loss diagnostic"       check_sl.py
run "3. Win-rate breakdown"         diagnose_winrate.py
run "4. Inversion backtest (v2)"    backtest_inversion_v2.py
run "5. Maker vs taker comparison"  backtest_maker_vs_taker.py

echo
echo "=============================================================="
echo "All backtests complete."
echo "=============================================================="
