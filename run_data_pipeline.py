"""
Run the full Alpaca data pipeline.

Usage:
    python run_data_pipeline.py                          # legacy rebalance-date pipeline
    python run_data_pipeline.py --spy-only               # just SPY bars + cash rate
    python run_data_pipeline.py --no-history             # SPY + current chain, skip historical
    python run_data_pipeline.py --extend-dte             # add longer-dated expiries to existing chains

    # --- Backfill: every trading day, all DTE, wide strikes ---
    python run_data_pipeline.py backfill                 # SPY, incremental from 2024-02-01
    python run_data_pipeline.py backfill --symbols AAPL  # any symbol
    python run_data_pipeline.py backfill --symbols SPY AAPL CRWD  # multiple
    python run_data_pipeline.py backfill --force          # re-fetch even if cached
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from config import DEFAULT_START_DATE, DEFAULT_END_DATE, TARGET_IDEAL_DTE


def main():
    parser = argparse.ArgumentParser(description="Alpaca data pipeline for cvx_options")
    sub = parser.add_subparsers(dest="cmd")

    # --- backfill subcommand ---
    p_bf = sub.add_parser("backfill",
                          help="Backfill option chains for every trading day (incremental)")
    p_bf.add_argument("--symbols", nargs="+", default=["SPY"],
                       help="Underlying symbols (default: SPY)")
    p_bf.add_argument("--start", default="2024-02-01",
                       help="First date to backfill (default: %(default)s)")
    p_bf.add_argument("--end", default=None,
                       help="Last date to backfill (default: today)")
    p_bf.add_argument("--max-dte", type=int, default=90,
                       help="Max DTE for expiry candidates (default: %(default)s)")
    p_bf.add_argument("--strike-pct", type=float, default=0.25,
                       help="Strike range as pct of spot, e.g. 0.25 = ±25%% (default: %(default)s)")
    p_bf.add_argument("--strike-step", type=float, default=1.0,
                       help="Dollar step between strikes (default: %(default)s)")
    p_bf.add_argument("--api-sleep", type=float, default=0.35,
                       help="Seconds between API batches (default: %(default)s)")
    p_bf.add_argument("--force", action="store_true",
                       help="Re-fetch even if chain files already exist")

    # --- legacy flags (no subcommand) ---
    parser.add_argument("--spy-only", action="store_true",
                        help="Only fetch SPY bars + cash rate")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip historical chain reconstruction")
    parser.add_argument("--start", default=DEFAULT_START_DATE, dest="start_legacy",
                        help="SPY start date (default: %(default)s)")
    parser.add_argument("--end", default=DEFAULT_END_DATE, dest="end_legacy",
                        help="SPY end date (default: today)")
    parser.add_argument("--period", type=int, default=TARGET_IDEAL_DTE,
                        help="Rebalance period in trading days")
    parser.add_argument("--extend-dte", action="store_true",
                        help="Fetch longer-dated expiries and merge into existing chains")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.cmd == "backfill":
        from data.fetch_alpaca import backfill_daily_chains, fetch_stock_bars

        symbols = [s.upper() for s in args.symbols]
        for sym in symbols:
            # Always refresh equity bars so we have the full trading calendar
            logging.info("Refreshing equity bars for %s ...", sym)
            fetch_stock_bars(sym, start="2020-01-01", end=args.end, save=True)

            backfill_daily_chains(
                symbol=sym,
                start_date=args.start,
                end_date=args.end,
                max_dte=args.max_dte,
                strike_pct_range=args.strike_pct,
                strike_step=args.strike_step,
                api_sleep=args.api_sleep,
                force=args.force,
            )
        return

    # --- Legacy pipeline ---
    from data.fetch_alpaca import run_full_pipeline, fetch_spy_bars, build_cash_rate

    if args.spy_only:
        spy_df = fetch_spy_bars(start=args.start_legacy, end=args.end_legacy, save=True)
        build_cash_rate(spy_df, save=True)
        return

    run_full_pipeline(
        spy_start=args.start_legacy,
        spy_end=args.end_legacy,
        fetch_historical=not args.no_history,
        period_days=args.period,
        extend_dte=args.extend_dte,
    )


if __name__ == "__main__":
    main()
