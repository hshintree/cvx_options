"""
Fetch raw market data: SPY daily, risk-free rate, option chain snapshots,
and historical prices for selected option contracts.

All functions write to config.RAW_DIR. Use daily frequency so that
weekly or daily trading can be supported later.

Options data: how often and how many
------------------------------------
- **Option chain snapshot**: Pulled once per run of the pipeline (e.g. when you
  run `run_data_pipeline.py` or `python -m data.fetch`). yfinance only provides
  the *current* chain, so each run saves one snapshot (calls_YYYY-MM-DD.parquet,
  puts_YYYY-MM-DD.parquet). To build a history of chains, run fetch daily (e.g. cron).

- **Contracts per chain**: The chain includes *all* expirations and *all* strikes
  returned by yfinance for SPY (often dozens of expirations × hundreds of strikes
  per expiry). We do not limit this snapshot; we save the full chain.

- **Option contract histories** (for building call/put return series): We do *not*
  fetch history for every contract in the chain. We select a subset and fetch
  daily OHLCV for each:
  - **Expirations**: First `OPTION_EXPIRATIONS_TO_FETCH` (default 4) from front month.
  - **Strikes**: Within ±`OPTION_ATM_STRIKE_PCT` (default 10%) of ATM.
  - **Per expiry per type**: Up to `OPTION_MAX_CONTRACTS_PER_EXPIRY` (default 15)
    calls and 15 puts.
  So at most 4 × 2 × 15 = **120 contracts** per run, each with one history request.
  Typical run: ~60–120 API calls for option histories, plus one chain request
  (which itself does one request per expiration in the chain).
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow importing config when run as script from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

from config import (
    CASH_RATE_FILE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    OPTION_CHAINS_DIR,
    OPTION_CONTRACTS_DIR,
    OPTION_ATM_STRIKE_PCT,
    OPTION_EXPIRATIONS_TO_FETCH,
    OPTION_HISTORICAL_MAX_MONTHS_BACK,
    OPTION_HISTORICAL_MONTHLY,
    OPTION_MAX_CONTRACTS_PER_EXPIRY,
    SPY_DAILY_FILE,
    UNDERLYING,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SPY daily
# -----------------------------------------------------------------------------


def fetch_spy_daily(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch SPY daily OHLCV from Yahoo Finance. Uses adjusted close for consistency.

    Parameters
    ----------
    start_date, end_date : str (YYYY-MM-DD) or None
        Range for history. Defaults from config.
    save : bool
        If True, write to parquet.
    output_path : Path, optional
        Override save path.

    Returns
    -------
    DataFrame with DatetimeIndex and columns: open, high, low, close, adj_close, volume.
    """
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    ticker = yf.Ticker(UNDERLYING)
    # period="max" gets full history; then we can slice
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        logger.warning("SPY history is empty for range %s to %s", start_date, end_date)
        return df
    # Standardize column names (yfinance uses Capitalized)
    df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))
    # Prefer adjusted close when available
    if "adj_close" in df.columns and "close" in df.columns:
        df["close"] = df["adj_close"]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if save:
        out = output_path or SPY_DAILY_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=True)
        logger.info("Saved SPY daily to %s (%d rows)", out, len(df))
    return df


# -----------------------------------------------------------------------------
# Cash / risk-free rate
# -----------------------------------------------------------------------------

# Yahoo symbol for 13-week Treasury yield (risk-free proxy)
RISKFREE_YAHOO_SYMBOL = "^IRX"


def fetch_cash_rate_daily(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch daily risk-free rate (13-week T-bill yield). Converts to per-day
    simple return for use as cash return.

    Returns
    -------
    DataFrame with DatetimeIndex and columns: rate (annual yield as decimal), return (per-day).
    """
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    ticker = yf.Ticker(RISKFREE_YAHOO_SYMBOL)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        logger.warning("Cash rate history empty; using 0.0")
        # Build a date range and fill with 0
        dr = pd.date_range(start=start_date, end=end_date or pd.Timestamp.now(), freq="B")
        df = pd.DataFrame(index=dr)
        df["rate"] = 0.0
    else:
        # yfinance returns Capitalized columns (Close, Open, ...)
        df.columns = df.columns.str.lower()
        price_col = "close" if "close" in df.columns else df.columns[0]
        df = df[[price_col]].rename(columns={price_col: "rate"})
        df.index = pd.to_datetime(df.index).tz_localize(None)
    # Per-day return: (1 + rate/100)^(1/252) - 1 ≈ rate/(100*252) for small rate
    df["return"] = (1 + df["rate"] / 100) ** (1 / 252) - 1
    df = df[["rate", "return"]]
    if save:
        out = output_path or CASH_RATE_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=True)
        logger.info("Saved cash rate to %s (%d rows)", out, len(df))
    return df


# -----------------------------------------------------------------------------
# Option chain snapshot (as of a given date; yfinance only provides "today")
# -----------------------------------------------------------------------------


def fetch_option_chain_spy(
    as_of_date: Optional[str] = None,
    *,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch SPY option chain. yfinance only returns current chain; as_of_date is
    used only for the filename when saving (use today if None).

    Returns
    -------
    calls : DataFrame (contractSymbol, strike, expiry, bid, ask, last, volume, implVol, etc.)
    puts  : DataFrame (same structure)
    """
    ticker = yf.Ticker(UNDERLYING)
    # Get all expirations and concatenate chains
    expirations = ticker.options
    if not expirations:
        logger.warning("No option expirations found for %s", UNDERLYING)
        return pd.DataFrame(), pd.DataFrame()

    all_calls = []
    all_puts = []
    for exp in expirations:
        try:
            chain = ticker.option_chain(exp)
            chain.calls["expiry"] = exp
            chain.puts["expiry"] = exp
            all_calls.append(chain.calls)
            all_puts.append(chain.puts)
            time.sleep(0.2)  # be nice to API
        except Exception as e:
            logger.warning("Option chain for expiry %s failed: %s", exp, e)

    calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    if save and (not calls.empty or not puts.empty):
        date_str = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = OPTION_CHAINS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path_calls = out_dir / f"calls_{date_str}.parquet"
        path_puts = out_dir / f"puts_{date_str}.parquet"
        if not calls.empty:
            calls.to_parquet(path_calls, index=False)
        if not puts.empty:
            puts.to_parquet(path_puts, index=False)
        logger.info("Saved option chain for %s to %s", date_str, out_dir)
    return calls, puts


# -----------------------------------------------------------------------------
# Option contract symbols for history (subset: front expiries, ATM strikes)
# -----------------------------------------------------------------------------


def _select_contracts_for_history(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    n_expirations: int = OPTION_EXPIRATIONS_TO_FETCH,
    atm_pct: float = OPTION_ATM_STRIKE_PCT,
    max_per_expiry: int = OPTION_MAX_CONTRACTS_PER_EXPIRY,
) -> list[str]:
    """Return list of option contract symbols to fetch history for."""
    symbols = []
    for exp in sorted(calls["expiry"].unique())[:n_expirations]:
        for kind, df in [("calls", calls), ("puts", puts)]:
            sub = df[df["expiry"] == exp].copy()
            if sub.empty:
                continue
            sub = sub.assign(
                dist_pct=((sub["strike"] - spot) / spot).abs(),
            )
            sub = sub[sub["dist_pct"] <= atm_pct].sort_values("dist_pct")
            sub = sub.head(max_per_expiry)
            if "contractSymbol" in sub.columns:
                symbols.extend(sub["contractSymbol"].tolist())
            else:
                # Build symbol if not present (e.g. Yahoo format SPY + YYMMDD + C/P + strike)
                for _, row in sub.iterrows():
                    exp_dt = pd.Timestamp(exp)
                    sym = f"{UNDERLYING}{exp_dt.strftime('%y%m%d')}{'C' if kind == 'calls' else 'P'}{int(row['strike']*1000):08d}"
                    symbols.append(sym)
    return list(dict.fromkeys(symbols))


def get_option_contract_symbols_for_history(
    spot: Optional[float] = None,
    n_expirations: int = OPTION_EXPIRATIONS_TO_FETCH,
    atm_pct: float = OPTION_ATM_STRIKE_PCT,
    max_per_expiry: int = OPTION_MAX_CONTRACTS_PER_EXPIRY,
) -> list[str]:
    """
    From current SPY option chain, select a subset of contract symbols
    (front expirations, ATM-ish strikes) for which we will fetch historical prices.

    Parameters
    ----------
    spot : float, optional
        Current SPY price for ATM. If None, fetched from yfinance.
    n_expirations, atm_pct, max_per_expiry
        Selection limits (see config).

    Returns
    -------
    List of option contract symbols (e.g. SPY250117C00600000).
    """
    if spot is None:
        spot = float(yf.Ticker(UNDERLYING).fast_info.get("lastPrice", 450.0))
    calls, puts = fetch_option_chain_spy(save=False)
    if calls.empty and puts.empty:
        return []
    return _select_contracts_for_history(
        calls, puts, spot,
        n_expirations=n_expirations,
        atm_pct=atm_pct,
        max_per_expiry=max_per_expiry,
    )


# -----------------------------------------------------------------------------
# Historical option symbols (for multi-year backfill)
# -----------------------------------------------------------------------------


def _third_fridays(start_date: str, end_date: str) -> list[pd.Timestamp]:
    """Return list of third-Friday-of-month dates in [start_date, end_date]."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    out = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            # Third Friday: first day of month, add 14 days to get to 15th, then adjust to Friday
            d = pd.Timestamp(year=year, month=month, day=1)
            # 0=Monday, 4=Friday; third Friday is 15 + (4 - weekday(15))
            fifteenth = d + pd.Timedelta(days=14)
            # days until next Friday
            weekday = fifteenth.dayofweek  # 0=Mon, 4=Fri
            friday = fifteenth + pd.Timedelta(days=(4 - weekday) % 7)
            if weekday <= 4:
                # 15th is Mon-Thu, so we add to get to that week's Friday
                pass
            else:
                # 15th is Fri or weekend: 15th is Fri -> friday is 15th; 15th Sat -> 16th; 15th Sun -> 17th
                if fifteenth.dayofweek == 4:
                    friday = fifteenth
                else:
                    friday = fifteenth + pd.Timedelta(days=(4 - fifteenth.dayofweek) % 7)
            # Actually: "third Friday" = the Friday that falls on 15th, 16th, 17th, 18th, 19th
            # Simplest: iterate from day 1, count Fridays, take 3rd
            first = pd.Timestamp(year=year, month=month, day=1)
            fridays = []
            for d in range(1, 32):
                try:
                    t = pd.Timestamp(year=year, month=month, day=d)
                    if t.dayofweek == 4:
                        fridays.append(t)
                except ValueError:
                    break
            if len(fridays) >= 3:
                third_friday = fridays[2]
                if start <= third_friday <= end:
                    out.append(third_friday)
    return out


def generate_historical_option_symbols(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spy_daily_path: Optional[Path] = None,
) -> list[str]:
    """
    Generate option contract symbols for past expirations (one ATM call + one ATM put
    per monthly expiry) so we can fetch their history and build multi-year option returns.

    Uses SPY daily close to approximate ATM strike on each expiry date. SPY options
    use whole-number strikes; format is SPY + YYMMDD + C/P + 8-digit strike.

    Returns
    -------
    List of symbols (e.g. SPY150117C00200000 for Jan 17 2015 call strike 200).
    """
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    path = spy_daily_path or SPY_DAILY_FILE
    if not path.exists():
        logger.warning("SPY daily not found at %s; cannot generate historical option symbols", path)
        return []
    spy = pd.read_parquet(path)
    spy = spy[["close"]].squeeze()
    spy.index = pd.to_datetime(spy.index).normalize()
    expiries = _third_fridays(start_date, end_date)
    # Yahoo does not keep history for long-expired options (404). Only include
    # expiries within the last N months so requested symbols might still exist.
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=OPTION_HISTORICAL_MAX_MONTHS_BACK)
    expiries = [e for e in expiries if e >= cutoff]
    if not expiries:
        logger.info("No historical option expiries in last %d months; skipping historical symbols", OPTION_HISTORICAL_MAX_MONTHS_BACK)
        return []
    symbols = []
    for exp in expiries:
        # SPY close on or just before expiry (option still traded)
        before = spy.index[spy.index <= exp]
        if len(before) == 0:
            continue
        ref_date = before[-1]
        spot = float(spy.loc[ref_date])
        strike = max(1, round(spot))  # whole number; SPY minimum strike 1
        strike_str = f"{int(strike * 1000):08d}"  # 200 -> 00200000
        yy_mm_dd = exp.strftime("%y%m%d")
        symbols.append(f"{UNDERLYING}{yy_mm_dd}C{strike_str}")
        symbols.append(f"{UNDERLYING}{yy_mm_dd}P{strike_str}")
    logger.info("Generated %d historical option symbols (%d monthly expiries)", len(symbols), len(expiries))
    return symbols


# -----------------------------------------------------------------------------
# Historical prices for option contracts
# -----------------------------------------------------------------------------


def fetch_option_contract_histories(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    save: bool = True,
    output_path: Optional[Path] = None,
    throttle_seconds: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for each option contract symbol. Results are stored in
    long format: (date, symbol, open, high, low, close, volume).

    Parameters
    ----------
    symbols : list of str
        Option contract symbols (e.g. from get_option_contract_symbols_for_history).
    start_date, end_date : str or None
        Date range.
    save : bool
        If True, write to parquet under OPTION_CONTRACTS_DIR.
    output_path : Path, optional
        If provided, write single parquet here (otherwise one file per symbol or combined).
    throttle_seconds : float
        Delay between API calls to avoid rate limits.
    """
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    # Reduce yfinance 404/delisted log noise (we already log per-symbol failures)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    rows = []
    for i, sym in enumerate(symbols):
        try:
            t = yf.Ticker(sym)
            hist = t.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                continue
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist.columns = [c.lower() for c in hist.columns]
            hist["symbol"] = sym
            hist = hist.reset_index()
            rows.append(hist)
            time.sleep(throttle_seconds)
        except Exception as e:
            logger.warning("History for %s failed: %s", sym, e)
        if (i + 1) % 20 == 0:
            logger.info("Fetched %d / %d option histories", i + 1, len(symbols))

    if not rows:
        logger.warning("No option contract history collected")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.tz_localize(None)
    combined = combined.rename(columns={"Date": "date"})

    if save:
        out = output_path or (OPTION_CONTRACTS_DIR / "option_contracts.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out, index=False)
        logger.info("Saved option contract histories to %s (%d rows)", out, len(combined))
    return combined


# -----------------------------------------------------------------------------
# One-shot: run full pipeline (SPY + cash + option chain snapshot + option histories)
# -----------------------------------------------------------------------------


def run_full_fetch(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    include_option_histories: bool = True,
    option_symbols: Optional[list[str]] = None,
    include_historical_options: bool = False,
) -> None:
    """
    Fetch and save all raw data: SPY daily, cash rate, today's option chain,
    and (optionally) historical prices for selected option contracts.

    Option histories:
    - If option_symbols is provided, only those symbols are fetched.
    - Otherwise: by default only current-chain symbols (~120) are fetched.
      Set include_historical_options=True to also add historical monthly symbols
      (last 24 months); many of those may 404 on Yahoo once expired.
    """
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    fetch_spy_daily(start_date, end_date, save=True)
    fetch_cash_rate_daily(start_date, end_date, save=True)
    fetch_option_chain_spy(save=True)

    if include_option_histories:
        if option_symbols is not None:
            symbols = option_symbols
        else:
            current = get_option_contract_symbols_for_history()
            if include_historical_options:
                historical = generate_historical_option_symbols(
                    start_date=start_date,
                    end_date=end_date,
                    spy_daily_path=SPY_DAILY_FILE,
                )
                symbols = list(dict.fromkeys(historical + current))
                logger.info("Fetching option histories for %d contracts (historical + current)", len(symbols))
            else:
                symbols = current
                logger.info("Fetching option histories for %d contracts (current chain only)", len(symbols))
        if symbols:
            fetch_option_contract_histories(
                symbols,
                start_date=start_date,
                end_date=end_date,
                save=True,
                throttle_seconds=0.35,
            )
        else:
            logger.warning("No option symbols to fetch history for")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_fetch()
