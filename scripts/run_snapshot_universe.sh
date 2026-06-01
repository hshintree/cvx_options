#!/bin/zsh
set -euo pipefail

REPO_DIR="/Users/hakeemshindy/cvx_options"
PYTHON_BIN="/Users/hakeemshindy/miniconda3/envs/cvx_options/bin/python"
LOG_DIR="$REPO_DIR/logs"
MARKET_TZ="America/New_York"
LOCK_DIR="$LOG_DIR/snapshot.lock"
SNAPSHOT_UNIVERSE="${SNAPSHOT_UNIVERSE:-liquid_research_plus_spy}"
SNAPSHOT_DTE_MIN="${SNAPSHOT_DTE_MIN:-7}"
SNAPSHOT_DTE_MAX="${SNAPSHOT_DTE_MAX:-35}"
SNAPSHOT_STRIKE_PCT="${SNAPSHOT_STRIKE_PCT:-0.10}"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Snapshot job already running; skipping overlapping launch." >&2
  exit 0
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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
"$PYTHON_BIN" "$REPO_DIR/run_data_pipeline.py" snapshot \
  --universe "$SNAPSHOT_UNIVERSE" \
  --dte-min "$SNAPSHOT_DTE_MIN" \
  --dte-max "$SNAPSHOT_DTE_MAX" \
  --strike-pct "$SNAPSHOT_STRIKE_PCT"
