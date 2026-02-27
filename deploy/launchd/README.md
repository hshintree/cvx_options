## launchd setup

This repo includes a ready-to-install LaunchAgent:

- Source plist: `/Users/hakeemshindy/cvx_options/deploy/launchd/com.hakeemshindy.cvx-options-snapshot.plist`
- Wrapper script: `/Users/hakeemshindy/cvx_options/scripts/run_snapshot_universe.sh`

Install it on macOS:

```bash
mkdir -p ~/Library/LaunchAgents
cp /Users/hakeemshindy/cvx_options/deploy/launchd/com.hakeemshindy.cvx-options-snapshot.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist
launchctl enable gui/$(id -u)/com.hakeemshindy.cvx-options-snapshot
launchctl kickstart -k gui/$(id -u)/com.hakeemshindy.cvx-options-snapshot
```

Inspect status:

```bash
launchctl print gui/$(id -u)/com.hakeemshindy.cvx-options-snapshot
tail -n 50 /Users/hakeemshindy/cvx_options/logs/launchd.snapshot.stdout.log
tail -n 50 /Users/hakeemshindy/cvx_options/logs/launchd.snapshot.stderr.log
```

Remove it:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist
rm ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist
```

How it works:

- `launchd` is the current macOS-native scheduler; it is preferred over `cron`.
- The agent wakes up every 30 minutes while your user session is active.
- The wrapper script skips weekends, then runs:
  `python run_data_pipeline.py snapshot --universe sp100_plus_spy`
- The wrapper script also gates execution to regular market hours in
  `America/New_York` (9:30 AM to 4:05 PM ET), so the 30-minute cadence only
  produces snapshots during the session.
- That captures live Alpaca option-chain snapshots (quotes, IV, greeks) for the
  frozen research basket and stores them under the per-symbol chain directories.

Important limitation:

- A LaunchAgent does **not** reliably run while your Mac is asleep or your user
  session is logged out.
- If you want dense intraday sampling, your Mac needs to stay awake during
  market hours. The job will resume after wake, but missed intervals are gone.
- If you later want truly reliable market-hours coverage without keeping the Mac
  awake, move this same script to an always-on host (Mac mini, cloud VM, etc.).
