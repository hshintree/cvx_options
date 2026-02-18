#!/usr/bin/env python3
"""
Run the full data pipeline: fetch raw data, then clean and build processed returns/volumes/prices.

Usage (from project root):
  conda activate cvx_options
  python run_data_pipeline.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--no-options]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.fetch import run_full_fetch
from data.clean import run_clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch and clean SPY + options + cash data")
    p.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--no-options",
        action="store_true",
        help="Skip fetching option contract histories (faster; call/put returns will be empty)",
    )
    p.add_argument(
        "--historical-options",
        action="store_true",
        help="Also fetch historical option symbols (last 24 months). Default: current chain only (~120 contracts).",
    )
    args = p.parse_args()
    run_full_fetch(
        start_date=args.start,
        end_date=args.end,
        include_option_histories=not args.no_options,
        include_historical_options=args.historical_options,
    )
    run_clean(save=True)
    print("Done. Processed data in data/processed/")


if __name__ == "__main__":
    main()
