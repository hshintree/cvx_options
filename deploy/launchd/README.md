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

If you edit the plist later, reload it like this:

```bash
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

Status interpretation:

- `state = not running` is normal for an interval-based job between launches.
- `runs > 0` and `last exit code = 0` means the job has been firing and exiting cleanly.
- `Bootstrap failed: 5: Input/output error` usually means the job is already loaded.
  Use `bootout` first, then `bootstrap`, instead of calling `bootstrap` twice.

Remove it:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist
rm ~/Library/LaunchAgents/com.hakeemshindy.cvx-options-snapshot.plist
```

How it works:

- `launchd` is the current macOS-native scheduler; it is preferred over `cron`.
- The agent wakes up every 30 minutes while your user session is active.
- The wrapper script skips weekends, then runs:
  `python run_data_pipeline.py snapshot --universe liquid_research_plus_spy --dte-min 7 --dte-max 35 --strike-pct 0.10`
- The wrapper script also gates execution to regular market hours in
  `America/New_York` (9:30 AM to 4:05 PM ET), so the 30-minute cadence only
  produces snapshots during the session.
- That captures live Alpaca option-chain snapshots (quotes, IV, greeks) for a
  smaller liquid research basket and stores them under the per-symbol chain
  directories.
- Each symbol/side still uses one parquet per calendar day, but new intraday
  scrapes append rows with a `snapshot_time` column instead of overwriting the
  earlier same-day snapshot.
- The wrapper also uses a lock directory under `logs/` so a long-running scrape
  cannot overlap with the next interval or a manual `kickstart`.

Important limitation:

- A LaunchAgent does **not** reliably run while your Mac is asleep or your user
  session is logged out.
- If you want dense intraday sampling, your Mac needs to stay awake during
  market hours. The job will resume after wake, but missed intervals are gone.
- Internet connectivity still matters. If Wi-Fi is down when the job fires, the
  process will launch but the Alpaca requests will fail for that interval.
- If you later want truly reliable market-hours coverage without keeping the Mac
  awake, move this same script to an always-on host (Mac mini, cloud VM, etc.).
