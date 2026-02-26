"""
Diagnostics and coverage summaries for SPY option-chain data.

This module provides a helper to answer:

- How many trading days in a given period have option-chain data?
- For each day with data:
  - How many distinct expiries / DTE values are present?
  - How many in-the-money (ITM) vs out-of-the-money (OTM) contracts do we have
    for calls and puts (relative to spot on that day)?
  - How far do strikes extend from deepest ITM to farthest OTM, for calls and
    for puts?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import OPTION_CHAINS_DIR, OPTION_DATA_START, SPY_DAILY_FILE


@dataclass
class DailyOptionCoverage:
    """Per-day summary of option-chain coverage."""

    trade_date: pd.Timestamp
    has_chain: bool
    n_dte: int = 0
    n_calls: int = 0
    n_calls_itm: int = 0
    n_calls_otm: int = 0
    n_puts: int = 0
    n_puts_itm: int = 0
    n_puts_otm: int = 0
    call_itm_min_strike: float | None = None
    call_itm_min_moneyness: float | None = None
    call_otm_max_strike: float | None = None
    call_otm_max_moneyness: float | None = None
    put_itm_max_strike: float | None = None
    put_itm_max_moneyness: float | None = None
    put_otm_min_strike: float | None = None
    put_otm_min_moneyness: float | None = None


def _list_chain_dates(chain_dir: Path) -> list[pd.Timestamp]:
    """Return all trade dates for which both calls_*.parquet and puts_*.parquet exist."""
    call_files = sorted(chain_dir.glob("calls_*.parquet"))
    dates = []
    for path in call_files:
        name = path.name
        # Expect names like calls_YYYY-MM-DD.parquet
        try:
            date_str = name.split("_", 1)[1].split(".parquet")[0]
            calls_date = pd.Timestamp(date_str)
        except Exception:
            continue
        # Require a matching puts file for the same date
        puts_path = chain_dir / f"puts_{date_str}.parquet"
        if puts_path.exists():
            dates.append(calls_date)
    return sorted(set(dates))


def summarize_option_chain_quality(
    start_date: str | pd.Timestamp = OPTION_DATA_START,
    end_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Summarize SPY option-chain coverage and moneyness by trade date.

    Parameters
    ----------
    start_date:
        First trade date to consider (inclusive). Defaults to OPTION_DATA_START.
    end_date:
        Last trade date to consider (inclusive). If None, uses the last date
        for which chain data exists.

    Returns
    -------
    pd.DataFrame
        One row per business day in [start_date, end_date], with columns:

        - trade_date
        - has_chain: bool
        - n_dte: number of distinct DTE values (expiries) on days with data
        - n_calls, n_calls_itm, n_calls_otm
        - n_puts, n_puts_itm, n_puts_otm
        - call_itm_min_strike, call_itm_min_moneyness
        - call_otm_max_strike, call_otm_max_moneyness
        - put_itm_max_strike, put_itm_max_moneyness
        - put_otm_min_strike, put_otm_min_moneyness

    Notes
    -----
    - ITM / OTM is defined relative to spot (close) on the trade date:

        * Calls:  strike < spot → ITM, strike > spot → OTM
        * Puts:   strike > spot → ITM, strike < spot → OTM

      Strikes exactly equal to spot are treated as at-the-money and excluded
      from the ITM/OTM counts.

    - Moneyness is reported as (strike / spot - 1), so negative values are
      ITM for calls and OTM for puts, and positive values are OTM for calls
      and ITM for puts.
    """
    chain_dir = OPTION_CHAINS_DIR
    chain_dates = _list_chain_dates(chain_dir)
    if not chain_dates:
        raise FileNotFoundError(f"No SPY option-chain files found in {chain_dir}")

    prices = pd.read_parquet(SPY_DAILY_FILE)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date is not None else chain_dates[-1]

    all_days = pd.bdate_range(start_ts, end_ts)
    have_chain = set(chain_dates)

    summaries: list[DailyOptionCoverage] = []

    for day in all_days:
        if day not in prices.index:
            # No equity close for this day; still record that we lack chains.
            summaries.append(DailyOptionCoverage(trade_date=day, has_chain=False))
            continue

        has = day in have_chain
        if not has:
            summaries.append(DailyOptionCoverage(trade_date=day, has_chain=False))
            continue

        # We have both calls_*.parquet and puts_*.parquet for this day.
        date_str = day.strftime("%Y-%m-%d")
        calls_path = chain_dir / f"calls_{date_str}.parquet"
        puts_path = chain_dir / f"puts_{date_str}.parquet"
        if not calls_path.exists() or not puts_path.exists():
            summaries.append(DailyOptionCoverage(trade_date=day, has_chain=False))
            continue

        calls = pd.read_parquet(calls_path)
        puts = pd.read_parquet(puts_path)

        # Normalize column names (lowercase, consistent with other data modules).
        calls.columns = [c.lower() for c in calls.columns]
        puts.columns = [c.lower() for c in puts.columns]

        # Ensure required columns are present.
        if "expiry" not in calls.columns or "strike" not in calls.columns:
            summaries.append(DailyOptionCoverage(trade_date=day, has_chain=False))
            continue
        if "expiry" not in puts.columns or "strike" not in puts.columns:
            summaries.append(DailyOptionCoverage(trade_date=day, has_chain=False))
            continue

        # Coerce strikes to float.
        calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
        puts["strike"] = pd.to_numeric(puts["strike"], errors="coerce")
        calls = calls.dropna(subset=["strike"])
        puts = puts.dropna(subset=["strike"])

        # Compute DTE for this trade date.
        calls["expiry"] = pd.to_datetime(calls["expiry"])
        puts["expiry"] = pd.to_datetime(puts["expiry"])
        calls["dte"] = (calls["expiry"] - day).dt.days.astype(int)
        puts["dte"] = (puts["expiry"] - day).dt.days.astype(int)
        n_dte = int(pd.Index(np.union1d(calls["dte"].unique(), puts["dte"].unique())).nunique())

        # Spot: SPY close on this day.
        spot = float(prices.loc[day, "close"])

        # Classify ITM / OTM.
        call_itm_mask = calls["strike"] < spot
        call_otm_mask = calls["strike"] > spot
        put_itm_mask = puts["strike"] > spot
        put_otm_mask = puts["strike"] < spot

        n_calls = int(len(calls))
        n_calls_itm = int(call_itm_mask.sum())
        n_calls_otm = int(call_otm_mask.sum())
        n_puts = int(len(puts))
        n_puts_itm = int(put_itm_mask.sum())
        n_puts_otm = int(put_otm_mask.sum())

        # Extremes for calls.
        call_itm_min_strike = None
        call_itm_min_moneyness = None
        if n_calls_itm > 0:
            s_min = float(calls.loc[call_itm_mask, "strike"].min())
            call_itm_min_strike = s_min
            call_itm_min_moneyness = s_min / spot - 1.0

        call_otm_max_strike = None
        call_otm_max_moneyness = None
        if n_calls_otm > 0:
            s_max = float(calls.loc[call_otm_mask, "strike"].max())
            call_otm_max_strike = s_max
            call_otm_max_moneyness = s_max / spot - 1.0

        # Extremes for puts.
        put_itm_max_strike = None
        put_itm_max_moneyness = None
        if n_puts_itm > 0:
            s_max_put = float(puts.loc[put_itm_mask, "strike"].max())
            put_itm_max_strike = s_max_put
            put_itm_max_moneyness = s_max_put / spot - 1.0

        put_otm_min_strike = None
        put_otm_min_moneyness = None
        if n_puts_otm > 0:
            s_min_put = float(puts.loc[put_otm_mask, "strike"].min())
            put_otm_min_strike = s_min_put
            put_otm_min_moneyness = s_min_put / spot - 1.0

        summaries.append(
            DailyOptionCoverage(
                trade_date=day,
                has_chain=True,
                n_dte=n_dte,
                n_calls=n_calls,
                n_calls_itm=n_calls_itm,
                n_calls_otm=n_calls_otm,
                n_puts=n_puts,
                n_puts_itm=n_puts_itm,
                n_puts_otm=n_puts_otm,
                call_itm_min_strike=call_itm_min_strike,
                call_itm_min_moneyness=call_itm_min_moneyness,
                call_otm_max_strike=call_otm_max_strike,
                call_otm_max_moneyness=call_otm_max_moneyness,
                put_itm_max_strike=put_itm_max_strike,
                put_itm_max_moneyness=put_itm_max_moneyness,
                put_otm_min_strike=put_otm_min_strike,
                put_otm_min_moneyness=put_otm_min_moneyness,
            )
        )

    df = pd.DataFrame([s.__dict__ for s in summaries])
    df.sort_values("trade_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


__all__ = ["DailyOptionCoverage", "summarize_option_chain_quality"]

