#!/bin/zsh
set -euo pipefail

REPO_DIR="/Users/hakeemshindy/cvx_options"
PYTHON_BIN="/Users/hakeemshindy/miniconda3/envs/cvx_options/bin/python"
LOG_DIR="$REPO_DIR/logs"
MARKET_TZ="America/New_York"

mkdir -p "$LOG_DIR"

# Skip weekends; daily launchd scheduling is simpler and more reliable than
# encoding market-day logic in the plist itself.
weekday="$(TZ="$MARKET_TZ" date +%u)"
if [[ "$weekday" -gt 5 ]]; then
  exit 0
fi

hhmm="$(TZ="$MARKET_TZ" date +%H%M)"
if [[ "$hhmm" -lt 0930 || "$hhmm" -gt 1605 ]]; then
  exit 0
fi

cd "$REPO_DIR"
exec "$PYTHON_BIN" "$REPO_DIR/run_data_pipeline.py" snapshot --universe sp100_plus_spy
