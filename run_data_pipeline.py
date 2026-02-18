"""
Run the full Alpaca data pipeline.

Usage:
    python run_data_pipeline.py                  # fetch everything
    python run_data_pipeline.py --spy-only       # just SPY bars + cash rate
    python run_data_pipeline.py --no-history     # SPY + current chain, skip historical
    python run_data_pipeline.py --extend-dte     # add longer-dated expiries to existing chains
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
    parser.add_argument("--spy-only", action="store_true", help="Only fetch SPY bars + cash rate")
    parser.add_argument("--no-history", action="store_true", help="Skip historical chain reconstruction")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="SPY start date (default: %(default)s)")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="SPY end date (default: today)")
    parser.add_argument("--period", type=int, default=TARGET_IDEAL_DTE, help="Rebalance period in trading days")
    parser.add_argument("--extend-dte", action="store_true",
                        help="Fetch longer-dated expiries and merge into existing chains")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from data.fetch_alpaca import run_full_pipeline, fetch_spy_bars, build_cash_rate

    if args.spy_only:
        spy_df = fetch_spy_bars(start=args.start, end=args.end, save=True)
        build_cash_rate(spy_df, save=True)
        return

    run_full_pipeline(
        spy_start=args.start,
        spy_end=args.end,
        fetch_historical=not args.no_history,
        period_days=args.period,
        extend_dte=args.extend_dte,
    )


if __name__ == "__main__":
    main()
