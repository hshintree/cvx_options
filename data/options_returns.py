"""
Build daily return series for "SPY call" and "SPY put" from
option contract price history.

At each date we define the front-month ATM call and put (by expiry and strike),
look up their prices at t and t+1 in the option history panel, and compute
one-period return. Result is aligned to the same daily calendar as SPY so
you can resample to weekly later.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import OPTION_CONTRACTS_DIR, SPY_DAILY_FILE

logger = logging.getLogger(__name__)

# Option symbol format: SPY + YYMMDD + C|P + 8-digit strike (e.g. 00600000 = 600)
_OPTION_SYMBOL_PATTERN = re.compile(
    r"^([A-Z]+)(\d{6})([CP])(\d{8})$", re.IGNORECASE
)


def _parse_option_symbol(symbol: str) -> Optional[Tuple[str, str, str, float]]:
    """Return (underlying, expiry_YYMMDD, type, strike) or None."""
    m = _OPTION_SYMBOL_PATTERN.match(symbol.strip())
    if not m:
        return None
    underlying, exp, opt_type, strike_str = m.groups()
    strike = int(strike_str) / 1000.0  # 00600000 -> 600.0
    return (underlying, exp, opt_type.upper(), strike)


def _expiry_to_date(yy_mm_dd: str) -> pd.Timestamp:
    """Convert YYMMDD to date."""
    return pd.Timestamp(
        int("20" + yy_mm_dd[:2]),
        int(yy_mm_dd[2:4]),
        int(yy_mm_dd[4:6]),
    )


def load_option_contracts(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load option contract history panel: columns date, symbol, close, (open, high, low, volume)."""
    path = path or (OPTION_CONTRACTS_DIR / "option_contracts.parquet")
    if not path.exists():
        logger.warning("Option contracts file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def build_option_metadata(contracts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse contract symbols into expiry and strike. Returns DataFrame with
    columns: symbol, underlying, expiry_date, type (C/P), strike.
    """
    rows = []
    for sym in contracts_df["symbol"].unique():
        parsed = _parse_option_symbol(str(sym))
        if parsed is None:
            continue
        underlying, yymmdd, opt_type, strike = parsed
        exp_dt = _expiry_to_date(yymmdd)
        rows.append({
            "symbol": sym,
            "underlying": underlying,
            "expiry_date": exp_dt,
            "type": opt_type,
            "strike": strike,
        })
    return pd.DataFrame(rows)


def select_front_atm_contracts(
    spot_series: pd.Series,
    metadata: pd.DataFrame,
    kind: str,
) -> pd.Series:
    """
    For each date in spot_series.index, select the contract that is
    front-month (expiry >= date, nearest in time) and ATM (strike closest to spot).

    Parameters
    ----------
    spot_series : Series with DatetimeIndex (e.g. SPY close).
    metadata : DataFrame from build_option_metadata (symbol, expiry_date, type, strike).
    kind : "C" for call, "P" for put.

    Returns
    -------
    Series index = date, value = option symbol (or NaN if none found).
    """
    meta = metadata[metadata["type"] == kind].copy()
    if meta.empty:
        return pd.Series(dtype=object)
    meta["expiry_date"] = pd.to_datetime(meta["expiry_date"])
    out = pd.Series(index=spot_series.index, dtype=object)
    for t in spot_series.index:
        spot = spot_series.loc[t]
        # Expiries strictly after t (we hold until next period)
        future = meta[meta["expiry_date"] > t]
        if future.empty:
            continue
        # Front month = smallest expiry after t
        front_exp = future["expiry_date"].min()
        front = future[future["expiry_date"] == front_exp]
        # ATM = strike closest to spot
        front = front.assign(dist=(front["strike"] - spot).abs())
        front = front.sort_values("dist")
        if not front.empty:
            out.loc[t] = front.iloc[0]["symbol"]
    return out


def compute_option_returns(
    contracts_df: pd.DataFrame,
    spot_series: pd.Series,
    kind: str,
) -> pd.Series:
    """
    Compute daily return series for the rolling front-month ATM option (call or put).

    For each date t we have spot_t. We select the front-month ATM contract for t,
    get its close at t and at t+1 (next trading day). Return = (close_{t+1} - close_t) / close_t.
    If the contract expires before t+1 we use expiry settlement (option value at expiry);
    for simplicity we use close at t+1 if available, else NaN for that day.

    Parameters
    ----------
    contracts_df : from load_option_contracts (date, symbol, close).
    spot_series : SPY close, DatetimeIndex.
    kind : "C" or "P".

    Returns
    -------
    Series of daily returns, same index as spot_series (NaNs where not available).
    """
    if contracts_df.empty:
        return pd.Series(index=spot_series.index, dtype=float)

    metadata = build_option_metadata(contracts_df)
    symbol_series = select_front_atm_contracts(spot_series, metadata, kind)
    # Pivot contracts to (date x symbol) close prices
    close_pivot = contracts_df.pivot(index="date", columns="symbol", values="close")
    close_pivot.index = pd.to_datetime(close_pivot.index).normalize()

    returns = pd.Series(index=spot_series.index, dtype=float)
    dates = spot_series.index.sort_values()
    for i, t in enumerate(dates):
        sym = symbol_series.loc[t]
        if pd.isna(sym) or sym not in close_pivot.columns:
            continue
        if i + 1 >= len(dates):
            continue
        t_next = dates[i + 1]
        if t_next not in close_pivot.index:
            continue
        p_t = close_pivot.loc[t, sym] if t in close_pivot.index else None
        p_next = close_pivot.loc[t_next, sym] if t_next in close_pivot.index else None
        if p_next is None:
            # Contract may have expired; we could use payoff at expiry here
            continue
        if p_t is None or p_t <= 0:
            continue
        ret = (float(p_next) - float(p_t)) / float(p_t)
        returns.loc[t] = ret
    return returns


def build_call_put_returns(
    spy_daily_path: Optional[Path] = None,
    option_contracts_path: Optional[Path] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build daily return series for the "SPY call" and "SPY put" assets (front-month ATM).

    Returns
    -------
    call_returns : Series, DatetimeIndex, daily return of front-month ATM call.
    put_returns  : Series, DatetimeIndex, daily return of front-month ATM put.
    """
    spy_path = spy_daily_path or SPY_DAILY_FILE
    if not spy_path.exists():
        logger.warning("SPY daily not found: %s", spy_path)
        return pd.Series(dtype=float), pd.Series(dtype=float)
    spy = pd.read_parquet(spy_path)
    spy = spy[["close"]].squeeze()
    spy.index = pd.to_datetime(spy.index).normalize()

    contracts = load_option_contracts(option_contracts_path)
    if contracts.empty:
        logger.warning("No option contracts; returning empty call/put returns")
        return pd.Series(index=spy.index, dtype=float), pd.Series(index=spy.index, dtype=float)

    call_ret = compute_option_returns(contracts, spy, "C")
    put_ret = compute_option_returns(contracts, spy, "P")
    return call_ret, put_ret


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cr, pr = build_call_put_returns()
    print("Call returns:", cr.dropna().describe())
    print("Put returns:", pr.dropna().describe())
