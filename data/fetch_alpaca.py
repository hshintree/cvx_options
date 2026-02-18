"""
Alpaca-based data pipeline for SPY equity + option chain data.

Replaces the yfinance pipeline.  Uses:
  - StockHistoricalDataClient  → SPY daily OHLCV bars
  - OptionHistoricalDataClient → option chain snapshots (current),
    historical daily bars for past rebalance dates

Historical option bars are available since ~Feb 2024.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    CASH_RATE_FILE,
    OPTION_CHAINS_DIR,
    RAW_DIR,
    SPY_DAILY_FILE,
    TARGET_IDEAL_DTE,
    TARGET_MAX_DTE,
    TARGET_MIN_DTE,
)

logger = logging.getLogger(__name__)

load_dotenv(_ROOT / ".env")

_API_KEY = os.getenv("ALPACA_API_KEY")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Rate-limit: brief sleep between Alpaca API calls (ms)
_API_SLEEP = 0.25
# Max option symbols per bars request (Alpaca cap)
_BATCH_SIZE = 100
# How far back historical option data is available
OPTION_DATA_START = "2024-02-01"


# ---------------------------------------------------------------------------
# Alpaca SDK clients (lazy-initialised)
# ---------------------------------------------------------------------------

_stock_client = None
_option_client = None


def _get_stock_client():
    global _stock_client
    if _stock_client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        _stock_client = StockHistoricalDataClient(_API_KEY, _SECRET_KEY)
    return _stock_client


def _get_option_client():
    global _option_client
    if _option_client is None:
        from alpaca.data.historical import OptionHistoricalDataClient
        _option_client = OptionHistoricalDataClient(_API_KEY, _SECRET_KEY)
    return _option_client


# ---------------------------------------------------------------------------
# OCC option symbology helpers
# ---------------------------------------------------------------------------

def make_occ_symbol(underlying: str, expiry: date, strike: float, is_call: bool) -> str:
    """Build OCC option symbol.  e.g. SPY250228C00550000"""
    t = "C" if is_call else "P"
    strike_int = int(round(strike * 1000))
    return f"{underlying}{expiry.strftime('%y%m%d')}{t}{strike_int:08d}"


def parse_occ_symbol(sym: str) -> dict:
    """Parse an OCC symbol into components."""
    root = sym[:3]  # works for SPY
    yy, mm, dd = int(sym[3:5]), int(sym[5:7]), int(sym[7:9])
    is_call = sym[9] == "C"
    strike = int(sym[10:18]) / 1000.0
    expiry = date(2000 + yy, mm, dd)
    return {
        "root": root,
        "expiry": expiry,
        "is_call": is_call,
        "strike": strike,
        "symbol": sym,
    }


# ---------------------------------------------------------------------------
# 1.  SPY daily bars
# ---------------------------------------------------------------------------

def fetch_spy_bars(
    start: str = "2020-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetch SPY daily bars from Alpaca and save as parquet."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_stock_client()
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=end_dt,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    # bars.df has MultiIndex (symbol, timestamp).  Flatten.
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SPY_DAILY_FILE, index=True)
        logger.info("Saved SPY bars: %d rows → %s", len(df), SPY_DAILY_FILE)
    return df


# ---------------------------------------------------------------------------
# 2.  Cash / risk-free rate (synthetic from constant or Fed data)
# ---------------------------------------------------------------------------

def build_cash_rate(spy_df: pd.DataFrame, annual_rate: float = 0.05, save: bool = True) -> pd.DataFrame:
    """Build a daily cash-rate series aligned to the SPY calendar."""
    cal = spy_df.index
    daily_return = (1.0 + annual_rate) ** (1.0 / 252) - 1.0
    df = pd.DataFrame({"rate": annual_rate, "return": daily_return}, index=cal)
    df.index.name = "date"
    if save:
        df.to_parquet(CASH_RATE_FILE, index=True)
        logger.info("Saved cash rate: %d rows → %s", len(df), CASH_RATE_FILE)
    return df


# ---------------------------------------------------------------------------
# 3.  Current option chain snapshot  (get_option_chain)
# ---------------------------------------------------------------------------

def fetch_current_chain(
    spot: float | None = None,
    dte_min: int = TARGET_MIN_DTE,
    dte_max: int = TARGET_MAX_DTE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the live SPY option chain from Alpaca (snapshot: bid, ask, IV, greeks).
    Saves to OPTION_CHAINS_DIR as calls_{date}.parquet / puts_{date}.parquet.
    """
    from alpaca.data.requests import OptionChainRequest

    client = _get_option_client()
    today = date.today()

    if spot is None and SPY_DAILY_FILE.exists():
        spy = pd.read_parquet(SPY_DAILY_FILE)
        spot = float(spy["close"].iloc[-1])
    spot = spot or 550.0

    exp_gte = (today + timedelta(days=dte_min)).isoformat()
    exp_lte = (today + timedelta(days=dte_max)).isoformat()

    request = OptionChainRequest(
        underlying_symbol="SPY",
        expiration_date_gte=exp_gte,
        expiration_date_lte=exp_lte,
        strike_price_gte=spot * 0.85,
        strike_price_lte=spot * 1.15,
    )
    chain = client.get_option_chain(request)
    logger.info("Fetched option chain: %d contracts", len(chain))

    rows = []
    for sym, snap in chain.items():
        parsed = parse_occ_symbol(sym)
        bid = snap.latest_quote.bid_price if snap.latest_quote else 0.0
        ask = snap.latest_quote.ask_price if snap.latest_quote else 0.0
        last_price = snap.latest_trade.price if snap.latest_trade else 0.0
        iv = snap.implied_volatility or 0.0
        delta = snap.greeks.delta if snap.greeks else None
        gamma = snap.greeks.gamma if snap.greeks else None
        theta = snap.greeks.theta if snap.greeks else None
        vega = snap.greeks.vega if snap.greeks else None

        rows.append({
            "contractSymbol": sym,
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": last_price,
            "bid": bid,
            "ask": ask,
            "impliedVolatility": iv,
            "is_call": parsed["is_call"],
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        })

    df = pd.DataFrame(rows)
    calls = df[df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
    puts = df[~df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

    if save:
        OPTION_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = today.isoformat()
        calls.to_parquet(OPTION_CHAINS_DIR / f"calls_{date_str}.parquet", index=False)
        puts.to_parquet(OPTION_CHAINS_DIR / f"puts_{date_str}.parquet", index=False)
        logger.info(
            "Saved chain %s: %d calls, %d puts",
            date_str, len(calls), len(puts),
        )
    return calls, puts


# ---------------------------------------------------------------------------
# 4.  Historical chain reconstruction from option bars
# ---------------------------------------------------------------------------

def _candidate_expiries(ref_date: date, min_dte: int, max_dte: int) -> List[date]:
    """Generate candidate SPY expiry dates (Mon/Wed/Fri within DTE range)."""
    candidates = []
    for delta in range(min_dte, max_dte + 1):
        d = ref_date + timedelta(days=delta)
        if d.weekday() in (0, 2, 4):  # Mon, Wed, Fri
            candidates.append(d)
    return candidates


def _generate_symbols(
    expiries: List[date],
    spot: float,
    pct_range: float = 0.10,
    strike_step: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """Generate OCC call and put symbols for strikes near ATM."""
    lo = int(spot * (1 - pct_range))
    hi = int(spot * (1 + pct_range)) + 1
    strikes = np.arange(lo, hi, strike_step)

    call_syms, put_syms = [], []
    for exp in expiries:
        for k in strikes:
            call_syms.append(make_occ_symbol("SPY", exp, float(k), True))
            put_syms.append(make_occ_symbol("SPY", exp, float(k), False))
    return call_syms, put_syms


def _fetch_bars_batch(
    symbols: List[str],
    bar_date: date,
) -> pd.DataFrame:
    """Fetch daily bars for a list of option symbols on a single date.

    Returns a DataFrame with columns: symbol, open, high, low, close, volume.
    Batches into groups of _BATCH_SIZE to stay within API limits.
    """
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_option_client()
    start_dt = datetime.combine(bar_date, datetime.min.time())
    end_dt = datetime.combine(bar_date + timedelta(days=3), datetime.min.time())

    frames = []
    for i in range(0, len(symbols), _BATCH_SIZE):
        batch = symbols[i : i + _BATCH_SIZE]
        try:
            request = OptionBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )
            bars = client.get_option_bars(request)
            df = bars.df
            if len(df) > 0:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    if "symbol" in df.columns:
                        df = df.rename(columns={"symbol": "contract"})
                frames.append(df)
        except Exception as e:
            logger.warning("Bars batch %d failed: %s", i // _BATCH_SIZE, e)
        time.sleep(_API_SLEEP)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Keep only bars on the target date
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined[combined["timestamp"].dt.date == bar_date]

    return combined


def fetch_historical_chain(
    chain_date: date,
    spot: float,
    min_dte: int = TARGET_MIN_DTE,
    max_dte: int = TARGET_MAX_DTE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct the option chain for a past date by querying option bars.

    1. Generate candidate expiry dates (M/W/F within DTE range).
    2. Generate OCC symbols for ATM ± 10% strikes.
    3. Fetch daily bars on chain_date for those symbols.
    4. Parse into calls/puts DataFrames matching the format forecasts.py expects.
    """
    expiries = _candidate_expiries(chain_date, min_dte, max_dte)
    if not expiries:
        logger.warning("No candidate expiries for %s (DTE %d-%d)", chain_date, min_dte, max_dte)
        return pd.DataFrame(), pd.DataFrame()

    call_syms, put_syms = _generate_symbols(expiries, spot)
    all_syms = call_syms + put_syms
    logger.info(
        "Fetching bars for %s: spot=%.0f, %d expiries, %d symbols",
        chain_date, spot, len(expiries), len(all_syms),
    )

    bars_df = _fetch_bars_batch(all_syms, chain_date)
    if bars_df.empty:
        logger.warning("No option bars returned for %s", chain_date)
        return pd.DataFrame(), pd.DataFrame()

    # Identify which column holds the symbol
    sym_col = "contract" if "contract" in bars_df.columns else "symbol"
    if sym_col not in bars_df.columns:
        logger.warning("No symbol column in bars response; columns=%s", bars_df.columns.tolist())
        return pd.DataFrame(), pd.DataFrame()

    # Parse each bar into chain format
    rows = []
    for _, row in bars_df.iterrows():
        sym = row[sym_col]
        parsed = parse_occ_symbol(sym)
        close_price = row.get("close", 0.0)
        open_price = row.get("open", 0.0)
        mid = (open_price + close_price) / 2 if (open_price > 0 and close_price > 0) else close_price
        rows.append({
            "contractSymbol": sym,
            "expiry": parsed["expiry"].isoformat(),
            "strike": parsed["strike"],
            "lastPrice": close_price,
            "bid": 0.0,
            "ask": 0.0,
            "impliedVolatility": 0.0,  # will be derived via BS inversion
            "is_call": parsed["is_call"],
            "volume": row.get("volume", 0),
        })

    chain_df = pd.DataFrame(rows)
    calls = chain_df[chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)
    puts = chain_df[~chain_df["is_call"]].drop(columns=["is_call"]).reset_index(drop=True)

    logger.info(
        "Chain %s: %d calls, %d puts (from %d bars)",
        chain_date, len(calls), len(puts), len(bars_df),
    )

    if save:
        OPTION_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = chain_date.isoformat()
        calls.to_parquet(OPTION_CHAINS_DIR / f"calls_{date_str}.parquet", index=False)
        puts.to_parquet(OPTION_CHAINS_DIR / f"puts_{date_str}.parquet", index=False)

    return calls, puts


# ---------------------------------------------------------------------------
# 5.  Compute rebalance dates from SPY calendar
# ---------------------------------------------------------------------------

def compute_rebalance_dates(
    spy_df: pd.DataFrame,
    period_days: int = TARGET_IDEAL_DTE,
    start_date: str = OPTION_DATA_START,
) -> List[date]:
    """Pick rebalance dates every `period_days` trading days, starting from
    start_date or the first available date with option data."""
    cal = spy_df.index.sort_values()
    cal = cal[cal >= pd.Timestamp(start_date)]
    dates = []
    i = 0
    while i < len(cal):
        dates.append(cal[i].date())
        i += period_days
    return dates


# ---------------------------------------------------------------------------
# 6.  Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    spy_start: str = "2020-01-01",
    spy_end: str | None = None,
    fetch_historical: bool = True,
    period_days: int = TARGET_IDEAL_DTE,
):
    """
    End-to-end Alpaca data fetch:
      1. SPY daily bars
      2. Cash rate series
      3. Current option chain snapshot
      4. Historical chain reconstructions for each rebalance date
    """
    logger.info("=== Alpaca Data Pipeline ===")

    # --- SPY bars ---
    spy_df = fetch_spy_bars(start=spy_start, end=spy_end, save=True)

    # --- Cash rate ---
    build_cash_rate(spy_df, save=True)

    # --- Current chain ---
    spot = float(spy_df["close"].iloc[-1])
    logger.info("Current spot: %.2f", spot)
    fetch_current_chain(spot=spot, save=True)

    # --- Historical chains ---
    if fetch_historical:
        rebal_dates = compute_rebalance_dates(spy_df, period_days=period_days)
        logger.info("Rebalance dates: %d (from %s to %s)", len(rebal_dates), rebal_dates[0], rebal_dates[-1])

        for i, rd in enumerate(rebal_dates):
            chain_file = OPTION_CHAINS_DIR / f"calls_{rd.isoformat()}.parquet"
            if chain_file.exists():
                logger.info("[%d/%d] %s — already cached, skipping", i + 1, len(rebal_dates), rd)
                continue

            rd_ts = pd.Timestamp(rd)
            spot_row = spy_df.index.get_indexer([rd_ts], method="ffill")
            if spot_row[0] >= 0:
                spot_rd = float(spy_df.iloc[spot_row[0]]["close"])
            else:
                spot_rd = spot

            logger.info("[%d/%d] Fetching chain for %s (spot=%.0f) ...", i + 1, len(rebal_dates), rd, spot_rd)
            fetch_historical_chain(rd, spot_rd, save=True)
            time.sleep(_API_SLEEP)

    logger.info("=== Pipeline complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_full_pipeline()
