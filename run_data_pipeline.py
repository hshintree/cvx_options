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

    # --- Local free historical chain ingestion ---
    python run_data_pipeline.py free-history --source optionsdx_csv --input data/raw/free_history
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from config import DEFAULT_START_DATE, DEFAULT_END_DATE, TARGET_IDEAL_DTE
from data.universes import UNIVERSE_BUILDERS, get_named_universe


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

    # --- snapshot subcommand ---
    p_snap = sub.add_parser(
        "snapshot",
        help="Capture today's live option chain snapshot (quotes + IV + greeks)",
    )
    p_snap.add_argument("--symbols", nargs="+", default=["SPY"],
                        help="Underlying symbols (default: SPY)")
    p_snap.add_argument("--universe", choices=sorted(UNIVERSE_BUILDERS),
                        help="Named fixed universe to snapshot instead of --symbols")
    p_snap.add_argument("--dte-min", type=int, default=None,
                        help="Min DTE for chain snapshot (default: config TARGET_MIN_DTE)")
    p_snap.add_argument("--dte-max", type=int, default=None,
                        help="Max DTE for chain snapshot (default: config TARGET_MAX_DTE)")
    p_snap.add_argument("--strike-pct", type=float, default=0.10,
                        help="Strike range as pct of spot for live snapshots, e.g. 0.10 = +/-10%% (default: %(default)s)")
    p_snap.add_argument("--refresh-bars", action="store_true",
                        help="Refresh daily stock bars before snapshot capture (off by default for intraday runs)")

    # --- free-history subcommand ---
    p_free = sub.add_parser(
        "free-history",
        help="Normalize a local free historical option-chain dataset into a research panel",
    )
    p_free.add_argument("--source", choices=["optionsdx_csv"], default="optionsdx_csv",
                        help="Provider schema for the local files (default: %(default)s)")
    p_free.add_argument("--input", required=True,
                        help="Path, directory, or glob pattern for local CSV/Parquet files")
    p_free.add_argument("--universe", choices=sorted(UNIVERSE_BUILDERS),
                        help="Optional named universe filter applied after normalization")
    p_free.add_argument("--output-dir", default=str(_ROOT / "data" / "processed" / "free_history"),
                        help="Directory for normalized and panel parquet outputs")
    p_free.add_argument("--target-dte", type=int, default=21,
                        help="Target DTE for candidate selection (default: %(default)s)")
    p_free.add_argument("--infer-missing-greeks", action="store_true",
                        help="Infer missing IV/greeks from bid/ask mids before building panels (off by default)")

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

    if args.cmd == "snapshot":
        from data.fetch_alpaca import fetch_current_chain, fetch_stock_bars

        symbols = get_named_universe(args.universe) if args.universe else [s.upper() for s in args.symbols]
        dte_min = args.dte_min if args.dte_min is not None else None
        dte_max = args.dte_max if args.dte_max is not None else None
        started = time.perf_counter()

        for idx, sym in enumerate(symbols, start=1):
            sym_started = time.perf_counter()
            if args.refresh_bars:
                logging.info("Refreshing equity bars for %s before snapshot capture ...", sym)
                fetch_stock_bars(sym, start="2020-01-01", end=None, save=True)
            fetch_current_chain(
                underlying_symbol=sym,
                dte_min=dte_min if dte_min is not None else TARGET_IDEAL_DTE // 2,
                dte_max=dte_max if dte_max is not None else TARGET_IDEAL_DTE * 2,
                strike_pct_range=args.strike_pct,
                save=True,
            )
            logging.info(
                "[%d/%d] Captured %s snapshot in %.2fs",
                idx,
                len(symbols),
                sym,
                time.perf_counter() - sym_started,
            )
        logging.info(
            "Snapshot run complete: %d symbols in %.2fs",
            len(symbols),
            time.perf_counter() - started,
        )
        return

    if args.cmd == "free-history":
        from data.fetch_free_history import (
            build_daily_research_candidates,
            build_research_clean_panel,
            compute_missing_greeks_from_mid,
            load_raw_historical_chain,
            normalize_historical_chain,
            serialize_contract_key_columns,
            summarize_source_coverage,
        )

        if args.universe:
            universe = set(get_named_universe(args.universe))
        else:
            universe = None

        raw = load_raw_historical_chain(args.source, args.input, target_symbols=universe)
        normalized = normalize_historical_chain(raw, source=args.source)
        if args.infer_missing_greeks:
            normalized = compute_missing_greeks_from_mid(normalized)

        if universe is not None:
            normalized = normalized[normalized["symbol"].isin(universe)].reset_index(drop=True)

        coverage = summarize_source_coverage(
            normalized,
            target_symbols=sorted(universe) if universe is not None else None,
        )
        candidates = build_daily_research_candidates(normalized, target_dte=args.target_dte)
        panel = build_research_clean_panel(normalized, target_dte=args.target_dte)

        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = output_dir / f"normalized_{args.source}.parquet"
        coverage_path = output_dir / f"coverage_{args.source}.parquet"
        candidates_path = output_dir / f"research_candidates_{args.source}.parquet"
        panel_path = output_dir / f"research_panel_{args.source}.parquet"

        normalized.to_parquet(normalized_path, index=False)
        coverage.to_parquet(coverage_path, index=False)
        serialize_contract_key_columns(candidates).to_parquet(candidates_path, index=False)
        serialize_contract_key_columns(panel).to_parquet(panel_path, index=False)

        logging.info("Saved normalized dataset to %s (%d rows)", normalized_path, len(normalized))
        logging.info("Saved coverage summary to %s (%d rows)", coverage_path, len(coverage))
        logging.info("Saved research candidates to %s (%d rows)", candidates_path, len(candidates))
        logging.info("Saved research panel to %s (%d rows)", panel_path, len(panel))
        logging.info("Candidate build failure counts: %s", candidates.attrs.get("failure_counts", {}))
        logging.info("Panel build failure counts: %s", panel.attrs.get("failure_counts", {}))
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
