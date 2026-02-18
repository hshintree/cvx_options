"""
Align SPY, option (call/put), and cash data to a common daily calendar.
Produce returns, volumes, and prices DataFrames for cvxportfolio (daily;
can resample to weekly in backtest or here).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    CASH_RATE_FILE,
    PROCESSED_DIR,
    PRICES_FILE,
    RETURNS_FILE,
    SPY_DAILY_FILE,
    VOLUMES_FILE,
)
from data.options_returns import build_call_put_returns

logger = logging.getLogger(__name__)

# Asset names for cvxportfolio (cash last)
SPY_COL = "SPY"
CALL_COL = "SPY_CALL"
PUT_COL = "SPY_PUT"
CASH_COL = "USDOLLAR"
ASSET_COLUMNS = [SPY_COL, CALL_COL, PUT_COL, CASH_COL]


def load_raw_spy(path: Optional[Path] = None) -> pd.DataFrame:
    """SPY daily with columns open, high, low, close, volume."""
    path = path or SPY_DAILY_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def load_raw_cash(path: Optional[Path] = None) -> pd.DataFrame:
    """Cash rate daily with columns rate, return."""
    path = path or CASH_RATE_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def build_returns(
    spy_daily_path: Optional[Path] = None,
    cash_path: Optional[Path] = None,
    option_contracts_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build daily open-to-open returns for SPY, SPY_CALL, SPY_PUT, USDOLLAR.
    Index = business daily; columns = ASSET_COLUMNS.
    """
    spy = load_raw_spy(spy_daily_path)
    if spy.empty:
        logger.warning("No SPY data; returning empty returns")
        return pd.DataFrame(columns=ASSET_COLUMNS)

    # Calendar = SPY index (already business days)
    calendar = spy.index.sort_values()
    out = pd.DataFrame(index=calendar, columns=ASSET_COLUMNS, dtype=float)

    # SPY: open-to-open return
    open_ = spy["open"].reindex(calendar).ffill()
    out[SPY_COL] = open_.shift(-1) / open_ - 1.0
    out.loc[out.index[-1], SPY_COL] = pd.NA  # last row no next open

    # Cash: per-day return from rate
    cash_df = load_raw_cash(cash_path)
    if not cash_df.empty and "return" in cash_df.columns:
        cash_ret = cash_df["return"].reindex(calendar).ffill()
        out[CASH_COL] = cash_ret
    else:
        out[CASH_COL] = 0.0

    # Call / put from option contract history (real data only; last ~24 months where Yahoo has it)
    call_ret, put_ret = build_call_put_returns(
        spy_daily_path=spy_daily_path,
        option_contracts_path=option_contracts_path,
    )
    if not call_ret.empty:
        out[CALL_COL] = call_ret.reindex(calendar)
    if not put_ret.empty:
        out[PUT_COL] = put_ret.reindex(calendar)

    return out


def build_volumes(spy_daily_path: Optional[Path] = None) -> pd.DataFrame:
    """Daily volumes: SPY from data; others NaN (cvxportfolio can handle)."""
    spy = load_raw_spy(spy_daily_path)
    if spy.empty:
        return pd.DataFrame(columns=ASSET_COLUMNS)
    calendar = spy.index.sort_values()
    out = pd.DataFrame(index=calendar, columns=ASSET_COLUMNS, dtype=float)
    out[SPY_COL] = spy["volume"].reindex(calendar)
    # Options: no volume in our call/put series; use NaN or 0
    out[CALL_COL] = pd.NA
    out[PUT_COL] = pd.NA
    out[CASH_COL] = 0.0
    return out


def build_prices(spy_daily_path: Optional[Path] = None) -> pd.DataFrame:
    """Daily open prices (for rounding trades). SPY from data; others NaN or placeholder."""
    spy = load_raw_spy(spy_daily_path)
    if spy.empty:
        return pd.DataFrame(columns=ASSET_COLUMNS)
    calendar = spy.index.sort_values()
    out = pd.DataFrame(index=calendar, columns=ASSET_COLUMNS, dtype=float)
    out[SPY_COL] = spy["open"].reindex(calendar)
    # Option "price" could be last close of front contract; we don't have it here, use NaN
    out[CALL_COL] = pd.NA
    out[PUT_COL] = pd.NA
    out[CASH_COL] = 1.0
    return out


def run_clean(
    spy_daily_path: Optional[Path] = None,
    cash_path: Optional[Path] = None,
    option_contracts_path: Optional[Path] = None,
    *,
    save: bool = True,
    min_valid_assets: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build returns, volumes, and prices; optionally drop rows with too many NaNs;
    save to PROCESSED_DIR.

    Parameters
    ----------
    min_valid_assets : int
        Drop rows where we have fewer than this many non-NaN returns (e.g. 2 = at least SPY + cash).
    """
    returns = build_returns(
        spy_daily_path=spy_daily_path,
        cash_path=cash_path,
        option_contracts_path=option_contracts_path,
    )
    volumes = build_volumes(spy_daily_path)
    prices = build_prices(spy_daily_path)

    if not returns.empty and min_valid_assets > 0:
        valid = returns.notna().sum(axis=1) >= min_valid_assets
        returns = returns.loc[valid]
        volumes = volumes.reindex(returns.index).ffill()
        prices = prices.reindex(returns.index).ffill()

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(RETURNS_FILE, index=True)
        volumes.to_parquet(VOLUMES_FILE, index=True)
        prices.to_parquet(PRICES_FILE, index=True)
        logger.info(
            "Saved processed data to %s: returns %s, volumes %s, prices %s",
            PROCESSED_DIR,
            RETURNS_FILE.name,
            VOLUMES_FILE.name,
            PRICES_FILE.name,
        )
    return returns, volumes, prices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ret, vol, pr = run_clean(save=True)
    print("Returns shape:", ret.shape)
    print("Returns head:\n", ret.head())
    print("Returns (non-null counts):\n", ret.notna().sum())
