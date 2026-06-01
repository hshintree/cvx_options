"""
Data loading utilities for the min_var module.
Migrated from log_norm.ipynb Cells 5-6, 35.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    SPY_DAILY_FILE, RAW_DIR, OPTION_CHAINS_DIR, PANEL_PATH,
    SYMS, MIN_REALIZED_DAYS, LOOKBACK_DAYS,
)


def load_spy_prices(asset: str = "SPY") -> pd.DataFrame:
    """
    Load daily OHLCV + log-return for the given asset.
    Returns DataFrame indexed by date with columns including 'close', 'log_ret'.
    """
    price_file = SPY_DAILY_FILE if asset == "SPY" else RAW_DIR / f"{asset.lower()}_daily.parquet"
    prices = pd.read_parquet(price_file)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index()
    prices["log_ret"] = np.log(prices["close"] / prices["close"].shift(1))
    return prices


def discover_chain_dates(asset: str = "SPY") -> list:
    """
    Return sorted list of all chain date strings that have both calls_ and puts_ files.
    """
    chain_dir = OPTION_CHAINS_DIR if asset == "SPY" else OPTION_CHAINS_DIR / asset
    all_files = sorted(os.listdir(chain_dir))
    dates = sorted({
        f.replace("calls_", "").replace("puts_", "").replace(".parquet", "")
        for f in all_files
        if (f.startswith("calls_") or f.startswith("puts_")) and f.endswith(".parquet")
    })
    return dates


def eligible_chain_dates(asset: str = "SPY", min_days_back: int = MIN_REALIZED_DAYS) -> list:
    """
    Return chain dates that have at least min_days_back calendar days of realized data
    relative to the last available close price.
    """
    prices = load_spy_prices(asset)
    last_close = prices.index.max()
    all_dates = discover_chain_dates(asset)
    return [
        d for d in all_dates
        if (last_close - pd.Timestamp(d)).days >= min_days_back
    ]


def pick_fit_date(asset: str = "SPY", chain_date: str | None = None) -> tuple:
    """
    Pick a chain date for single-date fitting (Sections 2-3 of notebook).

    Returns (fit_date_str, spot, mu_guess_ann, sigma_guess_ann, days_realized).
    """
    prices   = load_spy_prices(asset)
    eligible = eligible_chain_dates(asset)
    assert eligible, "No eligible chain dates found."

    if chain_date is not None:
        assert chain_date in eligible, f"{chain_date} not eligible."
        fit_date = chain_date
    else:
        fit_date = eligible[-1]

    fit_ts  = pd.Timestamp(fit_date)
    idx_fit = prices.index.get_indexer([fit_ts], method="ffill")[0]
    spot    = float(prices.iloc[idx_fit]["close"])

    recent_rets = prices.loc[prices.index <= fit_ts, "log_ret"].iloc[-LOOKBACK_DAYS:].dropna()
    mu_guess    = float(recent_rets.mean() * 252) if len(recent_rets) else 0.0
    sigma_guess = max(
        float(recent_rets.std() * np.sqrt(252)) if len(recent_rets) >= 2 else 0.15, 1e-4
    )
    days_realized = (prices.index.max() - fit_ts).days
    return fit_date, spot, mu_guess, sigma_guess, days_realized


def load_research_panel(panel_path: Path | None = None) -> pd.DataFrame:
    """
    Load the optionsdx free-history research panel (2023 data with Greeks).
    Returns sorted DataFrame with 'date_t' as datetime column.
    """
    path = panel_path or PANEL_PATH
    rp = pd.read_parquet(path)
    rp["date_t"] = pd.to_datetime(rp["date_t"])
    return rp.sort_values(["symbol", "date_t"]).reset_index(drop=True)


def build_returns_matrix(rp: pd.DataFrame, syms: list | None = None) -> pd.DataFrame:
    """
    Build wide returns DataFrame: date × all assets (underlying + option legs).
    Inner-joins on common dates where all underlying symbols are present.

    Columns: SPY, SPY_call1, SPY_put1, SPY_call2, SPY_put2, AAPL, AAPL_call1, ...
    """
    if syms is None:
        syms = SYMS

    ret_blocks = []
    for sym in syms:
        sub = rp[rp["symbol"] == sym].set_index("date_t")
        col_map = [
            ("R_und_real",    sym),
            ("call_1_R_real", f"{sym}_call1"),
            ("put_1_R_real",  f"{sym}_put1"),
            ("call_2_R_real", f"{sym}_call2"),
            ("put_2_R_real",  f"{sym}_put2"),
        ]
        for src_col, new_col in col_map:
            if src_col in sub.columns:
                ret_blocks.append(sub[[src_col]].rename(columns={src_col: new_col}))

    rets_all = pd.concat(ret_blocks, axis=1).sort_index()

    # Inner-join on dates where all underlying symbols are present
    common_dates = rets_all.dropna(subset=syms).index
    return rets_all.loc[common_dates]


def build_rp_index(rp: pd.DataFrame, syms: list | None = None) -> dict:
    """
    Pre-index the research panel by symbol for fast lookup during the backtest loop.
    Returns dict: {symbol: DataFrame indexed by date_t}.
    """
    if syms is None:
        syms = SYMS
    return {
        sym: rp[rp["symbol"] == sym].set_index("date_t").sort_index()
        for sym in syms
    }
