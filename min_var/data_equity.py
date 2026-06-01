"""
Equity + bond ETF data loader for extended backtests.

Wraps the existing data/fetch_alpaca.fetch_stock_bars convention:
  data/raw/{symbol.lower()}_daily.parquet

No new fetch logic — just ensures files exist and loads aligned returns.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import RAW_DIR  # noqa: E402
from min_var.config import EXTENDED_SYMBOLS, EXTENDED_DATA_START  # noqa: E402

logger = logging.getLogger(__name__)


def ensure_equity_data(
    symbols: list[str] = EXTENDED_SYMBOLS,
    start: str = EXTENDED_DATA_START,
    force: bool = False,
) -> None:
    """
    Fetch and cache daily bars for symbols not yet in data/raw/.
    Skips symbols whose parquet already exists (unless force=True).
    """
    from data.fetch_alpaca import fetch_stock_bars  # noqa: E402 (lazy import)

    for sym in symbols:
        path = RAW_DIR / f"{sym.lower()}_daily.parquet"
        if not force and path.exists():
            logger.debug("%s already cached at %s", sym, path)
            continue
        logger.info("Fetching %s from %s …", sym, start)
        df = fetch_stock_bars(sym, start=start, save=True)
        if df.empty:
            logger.warning("No data returned for %s", sym)
        else:
            logger.info("  %s: %d rows (%s → %s)", sym, len(df),
                        df.index.min().date(), df.index.max().date())


def load_equity_returns(
    symbols: list[str] = EXTENDED_SYMBOLS,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load daily log-returns for each symbol from data/raw/{sym}_daily.parquet.
    Returns a wide DataFrame (date × symbol) aligned on the intersection of
    trading days.  Missing data for individual symbols raises FileNotFoundError.
    """
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        path = RAW_DIR / f"{sym.lower()}_daily.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached data for {sym} at {path}. "
                f"Run ensure_equity_data(['{sym}']) first."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        frames[sym] = log_ret

    rets = pd.DataFrame(frames).dropna(how="all")

    if start:
        rets = rets[rets.index >= pd.Timestamp(start)]
    if end:
        rets = rets[rets.index <= pd.Timestamp(end)]

    # Keep only rows where ALL symbols have data (strict intersection)
    rets = rets.dropna(how="any")
    return rets


def build_equity_panel(
    symbols: list[str] = EXTENDED_SYMBOLS,
    start: str = EXTENDED_DATA_START,
    end: str | None = None,
    fetch_missing: bool = True,
) -> pd.DataFrame:
    """
    Convenience: ensure data exists, then load aligned returns.
    Returns date × symbol log-return DataFrame.
    """
    if fetch_missing:
        ensure_equity_data(symbols, start=start)
    return load_equity_returns(symbols, start=start, end=end)
